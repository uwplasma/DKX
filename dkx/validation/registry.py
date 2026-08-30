"""The one generic runner for the DKX validation registry.

`validation/registry.toml` binds each registered evidence artifact to its
capability, its inputs, the command that produced it, and the limits of what it
establishes. This module loads that file, checks the parts of every entry that
can be checked the same way for all of them, and dispatches into the entry's own
`audit()` for the parts that are campaign-specific physics.

The split matters. Before this module existed, each campaign carried its own
audit script *and* its own test module, and the generic half -- does the
artifact exist, does its checksum still match, does it declare a claim scope and
its exclusions, does its capability exist, does its audit still pass -- was
written out once per campaign. That half now lives here, once.

Run the whole registry from a checkout::

    python -m dkx.validation.registry

or a single entry::

    python -m dkx.validation.registry --entry w7x_seeded_bracket_discovery
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised by the 3.10 CI floor
    import tomli as tomllib

REGISTRY_RELATIVE_PATH = Path("validation/registry.toml")
CAPABILITIES_RELATIVE_PATH = Path("validation/capabilities.toml")
HARDWARE_RELATIVE_PATH = Path("validation/hardware.toml")

#: Every artifact must say what it does not establish. An entry that declares no
#: limitation is a claim without a boundary, which plan.md section 4.3 forbids.
REQUIRED_ENTRY_FIELDS = (
    "id",
    "capability",
    "status",
    "claim",
    "claim_scope",
    "artifact",
    "artifact_sha256",
    "command",
    "limitations",
)


def repository_root(start: Path | None = None) -> Path:
    """Return the checkout that owns ``validation/registry.toml``.

    Walks up from this file so the runner works from an installed package inside
    a checkout, from the checkout itself, and from a test module, without any of
    them passing a path.
    """
    here = (start or Path(__file__)).resolve()
    for candidate in (here, *here.parents):
        if (candidate / REGISTRY_RELATIVE_PATH).is_file():
            return candidate
    raise FileNotFoundError(
        f"no {REGISTRY_RELATIVE_PATH} found at or above {here}; the validation "
        "registry is only available from a DKX checkout, not from a wheel"
    )


@dataclass(frozen=True)
class Entry:
    """One registered evidence artifact."""

    id: str
    capability: str
    status: str
    claim: str
    claim_scope: str
    artifact: str
    artifact_sha256: str
    command: str
    limitations: tuple[str, ...]
    audit_script: str = ""
    artifact_schema: str = ""
    inputs: tuple[str, ...] = ()
    dkx_commit: str = ""
    generated_on_host: str = ""
    generated_on_host_recorded: str = ""
    external_commits: dict[str, str] = field(default_factory=dict)
    corruption: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> Entry:
        missing = [name for name in REQUIRED_ENTRY_FIELDS if not payload.get(name)]
        if missing:
            raise ValueError(
                f"registry entry {payload.get('id', '<unnamed>')!r} is missing "
                f"required fields: {', '.join(missing)}"
            )
        return cls(
            id=payload["id"],
            capability=payload["capability"],
            status=payload["status"],
            claim=payload["claim"],
            claim_scope=payload["claim_scope"],
            artifact=payload["artifact"],
            artifact_sha256=payload["artifact_sha256"],
            command=payload["command"],
            limitations=tuple(payload["limitations"]),
            audit_script=payload.get("audit_script", ""),
            artifact_schema=payload.get("artifact_schema", ""),
            inputs=tuple(payload.get("inputs", ())),
            dkx_commit=payload.get("dkx_commit", ""),
            generated_on_host=payload.get("generated_on_host", ""),
            generated_on_host_recorded=payload.get("generated_on_host_recorded", ""),
            external_commits=dict(payload.get("external_commits", {})),
            corruption=dict(payload.get("corruption", {})),
        )


@dataclass(frozen=True)
class Registry:
    """The parsed `validation/registry.toml` plus the checkout that owns it."""

    root: Path
    schema_version: int
    recorded_at: str
    registry_commit: str
    status_values: tuple[str, ...]
    entries: tuple[Entry, ...]

    def __getitem__(self, entry_id: str) -> Entry:
        for entry in self.entries:
            if entry.id == entry_id:
                return entry
        raise KeyError(entry_id)

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(entry.id for entry in self.entries)


def load_registry(root: Path | None = None) -> Registry:
    """Parse the registry and reject a malformed or duplicated entry list."""
    root = root or repository_root()
    with (root / REGISTRY_RELATIVE_PATH).open("rb") as stream:
        payload = tomllib.load(stream)

    entries = tuple(Entry.from_mapping(item) for item in payload["entry"])
    identifiers = [entry.id for entry in entries]
    duplicates = sorted({name for name in identifiers if identifiers.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate registry entry ids: {', '.join(duplicates)}")

    return Registry(
        root=root,
        schema_version=int(payload["schema_version"]),
        recorded_at=str(payload["recorded_at"]),
        registry_commit=str(payload["registry_commit"]),
        status_values=tuple(payload["status_values"]),
        entries=entries,
    )


def load_capability_ids(root: Path | None = None) -> frozenset[str]:
    """Return the capability ids the registry is allowed to reference."""
    root = root or repository_root()
    with (root / CAPABILITIES_RELATIVE_PATH).open("rb") as stream:
        payload = tomllib.load(stream)
    return frozenset(str(item["id"]) for item in payload["capability"])


def load_hardware_ids(root: Path | None = None) -> frozenset[str]:
    """Return the measurement hosts the registry is allowed to reference."""
    root = root or repository_root()
    with (root / HARDWARE_RELATIVE_PATH).open("rb") as stream:
        payload = tomllib.load(stream)
    return frozenset(str(item["id"]) for item in payload["host"])


def read_artifact(entry: Entry, root: Path) -> dict[str, Any]:
    """Return the artifact payload for ``entry``."""
    text = (root / entry.artifact).read_text(encoding="utf-8")
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise TypeError(f"{entry.artifact} is not a JSON object")
    return payload


def artifact_digest(entry: Entry, root: Path) -> str:
    """Return the sha256 of the artifact as it currently sits on disk."""
    return hashlib.sha256((root / entry.artifact).read_bytes()).hexdigest()


def artifact_exclusions(payload: dict[str, Any]) -> tuple[str, ...]:
    """Return an artifact's declared exclusions under either spelling.

    Campaigns settled on two names for the same field, `exclusions` and
    `claim_exclusions`. Both are read here rather than rewriting sealed
    artifacts, whose checksums are the point.
    """
    raw = payload.get("exclusions", payload.get("claim_exclusions", ()))
    return tuple(str(item) for item in raw)


def load_audit_callable(entry: Entry, root: Path):
    """Import the entry's audit script and return its ``audit`` function.

    The scripts live under `tools/`, which is deliberately not an importable
    package: they are checkout-local evidence code, not shipped API. Loading
    them by path keeps that boundary while still letting one runner drive them.
    """
    if not entry.audit_script:
        return None
    script = root / entry.audit_script
    spec = importlib.util.spec_from_file_location(script.stem, script)
    if spec is None or spec.loader is None:  # pragma: no cover - importlib guard
        raise ImportError(f"cannot load audit script {entry.audit_script}")
    module = importlib.util.module_from_spec(spec)
    # Several auditors import a sibling by bare name, which works when the
    # script is run directly because Python puts its directory on sys.path.
    # Reproduce that, and register the module under its own stem so a sibling
    # import resolves to this instance instead of executing the file twice.
    directory = str(script.parent)
    added = directory not in sys.path
    if added:
        sys.path.insert(0, directory)
    try:
        sys.modules.setdefault(script.stem, module)
        spec.loader.exec_module(module)
    finally:
        if added:
            sys.path.remove(directory)
    audit = getattr(module, "audit", None)
    if audit is None:
        raise AttributeError(f"{entry.audit_script} defines no audit() entry point")
    return audit


def check_entry(
    entry: Entry,
    registry: Registry,
    capabilities: frozenset[str],
    hosts: frozenset[str] | None = None,
) -> list[str]:
    """Return the problems with one entry; an empty list means it is sound.

    Every check here is one that is identical for all twenty campaigns. Anything
    campaign-specific belongs in that campaign's own ``audit()``.
    """
    root = registry.root
    problems: list[str] = []

    if entry.status not in registry.status_values:
        problems.append(
            f"status {entry.status!r} is not one of {', '.join(registry.status_values)}"
        )
    if entry.capability not in capabilities:
        problems.append(f"capability {entry.capability!r} is not in capabilities.toml")
    if hosts is not None and entry.generated_on_host not in hosts:
        problems.append(
            f"generated_on_host {entry.generated_on_host!r} is not a host in "
            "hardware.toml; every claim has to name the machine that produced it"
        )

    artifact_path = root / entry.artifact
    if not artifact_path.is_file():
        problems.append(f"artifact {entry.artifact} does not exist")
        return problems

    digest = artifact_digest(entry, root)
    if digest != entry.artifact_sha256:
        problems.append(
            f"artifact {entry.artifact} has sha256 {digest}, registry records "
            f"{entry.artifact_sha256}"
        )

    payload = read_artifact(entry, root)
    if payload.get("claim_scope") != entry.claim_scope:
        problems.append(
            f"claim_scope mismatch: registry {entry.claim_scope!r}, artifact "
            f"{payload.get('claim_scope')!r}"
        )
    if entry.artifact_schema and str(payload.get("schema", "")) != entry.artifact_schema:
        problems.append(
            f"schema mismatch: registry {entry.artifact_schema!r}, artifact "
            f"{payload.get('schema')!r}"
        )

    if entry.dkx_commit:
        source = payload.get("source") or {}
        environment = payload.get("environment") or {}
        recorded = (
            source.get("dkx_commit")
            or source.get("dkx_base_commit")
            or environment.get("dkx_commit")
        )
        if recorded != entry.dkx_commit:
            problems.append(
                f"dkx_commit mismatch: registry {entry.dkx_commit!r}, artifact "
                f"{recorded!r}"
            )

    exclusions = artifact_exclusions(payload)
    if tuple(entry.limitations) != exclusions:
        problems.append(
            "limitations do not match the artifact's own exclusions: registry "
            f"{list(entry.limitations)}, artifact {list(exclusions)}"
        )

    for relative in entry.inputs:
        if not (root / relative).is_file():
            problems.append(f"declared input {relative} does not exist")

    if entry.audit_script:
        if not (root / entry.audit_script).is_file():
            problems.append(f"audit script {entry.audit_script} does not exist")
        elif entry.audit_script not in entry.command:
            problems.append(
                f"command does not invoke the declared audit script "
                f"{entry.audit_script}: {entry.command!r}"
            )

    return problems


def run_entry(entry: Entry, root: Path) -> dict[str, Any]:
    """Execute one entry's own audit and return its report.

    Entries without an audit script -- an operational no-go whose evidence is a
    stopped run rather than a comparison -- report ``pass`` from the fact that
    their generic checks held, and say so.
    """
    audit = load_audit_callable(entry, root)
    if audit is None:
        return {
            "pass": True,
            "audited": False,
            "reason": "entry declares no audit script; generic checks only",
        }
    report = audit(root / entry.artifact)
    if not isinstance(report, dict):
        raise TypeError(f"{entry.audit_script} audit() returned {type(report)!r}")
    return report


def _traverse(payload: Any, path: str) -> tuple[Any, str | int]:
    """Return the container and key that ``path`` addresses.

    ``path`` is dot-separated; a segment that parses as an integer indexes a
    list, so ``comparisons.-1.root_movements.0.field`` reaches the last
    comparison's first root movement. Deliberately small: this addresses
    artifact JSON in a registry file, not arbitrary documents.
    """
    segments = path.split(".")
    cursor = payload
    for segment in segments[:-1]:
        key: str | int = int(segment) if segment.lstrip("-").isdigit() else segment
        cursor = cursor[key]
    last = segments[-1]
    return cursor, int(last) if last.lstrip("-").isdigit() else last


def corrupt_payload(payload: dict[str, Any], probe: dict[str, Any]) -> dict[str, Any]:
    """Return ``payload`` with the probe's single edit applied.

    Two operations are enough for every registered probe: ``set`` replaces a
    value outright, and ``scale`` multiplies a stored number, which is how a
    plausible-looking output tamper is expressed.
    """
    container, key = _traverse(payload, probe["path"])
    operation = probe["operation"]
    if operation == "set":
        container[key] = probe["value"]
    elif operation == "scale":
        container[key] = container[key] * probe["value"]
    else:
        raise ValueError(f"unknown corruption operation {operation!r}")
    return payload


def run_corruption_probe(entry: Entry, root: Path) -> list[str]:
    """Return the problems with an entry's corruption probe.

    An auditor that accepts a tampered artifact is worse than no auditor, so
    every entry that declares a probe must fail on it, and must fail with the
    error the registry names rather than some unrelated complaint.
    """
    if not entry.corruption:
        return []

    audit = load_audit_callable(entry, root)
    if audit is None:
        return [f"{entry.id} declares a corruption probe but no audit script"]

    payload = corrupt_payload(read_artifact(entry, root), entry.corruption)
    with tempfile.TemporaryDirectory() as directory:
        tampered = Path(directory) / f"{entry.id}_tampered.json"
        tampered.write_text(json.dumps(payload), encoding="utf-8")
        report = audit(tampered)

    problems: list[str] = []
    if report.get("pass") is not False:
        problems.append(
            f"audit accepted a tampered artifact "
            f"({entry.corruption['operation']} {entry.corruption['path']})"
        )
    reported = " | ".join(str(item) for item in report.get("errors", ()))
    for expected in entry.corruption.get("expected_errors", ()):
        if expected not in reported:
            problems.append(f"tampered audit did not report {expected!r}; got {reported[:300]!r}")
    return problems


def audit_registry(root: Path | None = None, *, entry_id: str | None = None) -> dict[str, Any]:
    """Check every registered entry and run its audit.

    Returns a report whose ``pass`` is true only when no entry reported a
    problem and every audit passed.
    """
    registry = load_registry(root)
    capabilities = load_capability_ids(registry.root)
    hosts = load_hardware_ids(registry.root)
    selected = registry.entries
    if entry_id is not None:
        selected = (registry[entry_id],)

    results: list[dict[str, Any]] = []
    for entry in selected:
        problems = check_entry(entry, registry, capabilities, hosts)
        report: dict[str, Any] = {}
        if not problems:
            report = run_entry(entry, registry.root)
            if report.get("pass") is not True:
                problems.append(f"audit did not pass: {json.dumps(report)[:400]}")
            problems.extend(run_corruption_probe(entry, registry.root))
        results.append(
            {
                "id": entry.id,
                "capability": entry.capability,
                "status": entry.status,
                "pass": not problems,
                "problems": problems,
                "audit": report,
            }
        )

    return {
        "schema": "dkx.validation.registry.audit.v1",
        "registry_commit": registry.registry_commit,
        "recorded_at": registry.recorded_at,
        "entries": len(results),
        "failed": sum(1 for item in results if not item["pass"]),
        "pass": all(item["pass"] for item in results),
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m dkx.validation.registry",
        description="Check every registered DKX validation artifact.",
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--entry", default=None, help="check one registry entry by id.")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="print one summary line instead of the full report.",
    )
    args = parser.parse_args(argv)

    report = audit_registry(args.root, entry_id=args.entry)
    if args.quiet:
        print(
            f"{report['entries'] - report['failed']}/{report['entries']} registry "
            f"entries pass"
        )
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
