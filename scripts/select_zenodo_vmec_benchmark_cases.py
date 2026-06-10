#!/usr/bin/env python3
"""Select deterministic Zenodo VMEC benchmark cases from a compact manifest."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_MANIFEST = Path("docs/_static/figures/vmec_jax_finite_beta/zenodo_vmec_manifest.json")


def _grid_size(entry: dict[str, Any]) -> int:
    resolution = entry.get("resolution", {})
    values = [resolution.get(name) for name in ("Ntheta", "Nzeta", "Nxi", "Nx")]
    if any(value is None for value in values):
        return 0
    size = 1
    for value in values:
        size *= int(value)
    return size


def _resolution_label(entry: dict[str, Any]) -> str:
    return str(entry.get("resolution", {}).get("label", "unknown"))


def _surface(entry: dict[str, Any]) -> float | None:
    value = entry.get("surface_s")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _eligible_entries(manifest: dict[str, Any], *, families: set[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for entry in manifest.get("entries", []):
        if families and entry.get("family") not in families:
            continue
        if entry.get("rhs_mode") != 1:
            continue
        if not entry.get("fortran_output_summary", {}).get("exists", False):
            continue
        if _surface(entry) is None:
            continue
        out.append(entry)
    return out


def _selected_resolution_labels(entries: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Return deterministic low/intermediate/production resolution labels."""
    by_label: dict[str, int] = {}
    for entry in entries:
        label = _resolution_label(entry)
        by_label[label] = max(by_label.get(label, 0), _grid_size(entry))
    labels = sorted(by_label, key=lambda label: (by_label[label], label))
    if not labels:
        return []
    if len(labels) == 1:
        return [("single", labels[0])]
    if len(labels) == 2:
        return [("low", labels[0]), ("production", labels[-1])]
    middle = labels[len(labels) // 2]
    selected = [("low", labels[0]), ("intermediate", middle), ("production", labels[-1])]
    deduped: list[tuple[str, str]] = []
    seen: set[str] = set()
    for rung, label in selected:
        if label in seen:
            continue
        deduped.append((rung, label))
        seen.add(label)
    return deduped


def _closest_surface(entries: Iterable[dict[str, Any]], target: float) -> dict[str, Any] | None:
    candidates = [entry for entry in entries if _surface(entry) is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda entry: (abs(float(_surface(entry)) - target), str(entry.get("input", ""))))


def _case_record(entry: dict[str, Any], *, family: str, rung: str, surface_role: str, target_surface: float) -> dict[str, Any]:
    return {
        "family": family,
        "rung": rung,
        "surface_role": surface_role,
        "target_surface_s": target_surface,
        "input": entry["input"],
        "fortran_output": entry["fortran_output"],
        "case": entry.get("case"),
        "surface_s": entry.get("surface_s"),
        "resolution": entry.get("resolution"),
        "grid_size": _grid_size(entry),
        "geometry_scheme": entry.get("geometry_scheme"),
        "collision_operator": entry.get("collision_operator"),
        "solver_tolerance": entry.get("solver_tolerance"),
        "nu_n": entry.get("nu_n"),
        "er": entry.get("er"),
    }


def select_zenodo_vmec_benchmark_cases(
    manifest: dict[str, Any],
    *,
    families: Iterable[str] = ("qa", "qh", "w7x"),
    surface_targets: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Select compact, deterministic benchmark cases from a Zenodo manifest."""
    family_set = set(families)
    targets = surface_targets or {"inner_edge": 0.05, "central": 0.5, "outer_edge": 0.95}
    entries = _eligible_entries(manifest, families=family_set)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        by_family[str(entry.get("family"))].append(entry)

    selected: list[dict[str, Any]] = []
    seen_inputs: set[str] = set()
    for family in sorted(by_family):
        family_entries = by_family[family]
        for rung, label in _selected_resolution_labels(family_entries):
            rung_entries = [entry for entry in family_entries if _resolution_label(entry) == label]
            for role, target in targets.items():
                chosen = _closest_surface(rung_entries, target)
                if chosen is None:
                    continue
                input_key = str(chosen["input"])
                if input_key in seen_inputs:
                    continue
                seen_inputs.add(input_key)
                selected.append(_case_record(chosen, family=family, rung=rung, surface_role=role, target_surface=target))

    counts_by_family: dict[str, int] = defaultdict(int)
    counts_by_rung: dict[str, int] = defaultdict(int)
    for item in selected:
        counts_by_family[item["family"]] += 1
        counts_by_rung[item["rung"]] += 1

    return {
        "schema_version": 1,
        "source_manifest_input_count": manifest.get("input_count"),
        "source_manifest_with_fortran_output_count": manifest.get("with_fortran_output_count"),
        "selected_count": len(selected),
        "families": sorted(family_set),
        "surface_targets": targets,
        "counts_by_family": dict(sorted(counts_by_family.items())),
        "counts_by_rung": dict(sorted(counts_by_rung.items())),
        "cases": selected,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--family", action="append", default=None, help="Family to include; may be repeated")
    args = parser.parse_args(argv)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    payload = select_zenodo_vmec_benchmark_cases(manifest, families=args.family or ("qa", "qh", "w7x"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote Zenodo VMEC benchmark selection: {args.out} ({payload['selected_count']} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
