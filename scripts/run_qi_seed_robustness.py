#!/usr/bin/env python
"""Build and optionally run deterministic QI seed-robustness cases.

The checked-in quasi-isodynamic VMEC example is expensive at authored
resolution, so this lane creates reproducible neighboring smoke decks around
that input. By default it only writes inputs and a manifest; pass ``--execute``
to run each seed through ``sfincs_jax write-output``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QI_INPUT = REPO_ROOT / "examples" / "additional_examples" / "input.namelist"
DEFAULT_OUT_ROOT = REPO_ROOT / "tests" / "qi_seed_robustness"
RESOLUTION_KEYS = ("NTHETA", "NZETA", "NX", "NXI")


def _read_resolution(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for key in RESOLUTION_KEYS:
        value = _read_number_parameter(text, key)
        if value is not None:
            out[key] = int(round(float(value)))
    return out


def _read_number_parameter(text: str, key: str) -> float | None:
    match = re.search(rf"(?im)^\s*{re.escape(key)}\s*=\s*([-+0-9.eEdD]+)", text)
    if match is None:
        return None
    try:
        return float(match.group(1).replace("D", "E").replace("d", "e"))
    except ValueError:
        return None


def _read_string_parameter(text: str, key: str) -> str | None:
    match = re.search(rf"(?im)^\s*{re.escape(key)}\s*=\s*([^!\n]+)", text)
    if match is None:
        return None
    value = match.group(1).strip().rstrip(",").strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _replace_or_append_parameter(text: str, *, group: str, key: str, value: str) -> str:
    pattern = re.compile(rf"(?im)^(\s*{re.escape(key)}\s*=\s*)([^!\n]*?)(\s*(?:!.*)?)$")
    if pattern.search(text):
        return pattern.sub(rf"\g<1>{value}\3", text, count=1)

    group_pattern = re.compile(rf"(?ims)(^\s*&{re.escape(group)}\b.*?)(^\s*/\s*$)")
    group_match = group_pattern.search(text)
    if group_match is not None:
        return text[: group_match.start(2)] + f"  {key} = {value}\n" + text[group_match.start(2) :]

    return text.rstrip() + f"\n\n&{group}\n  {key} = {value}\n/\n"


def _normalize_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines()).rstrip() + "\n"


def _hash_unit(seed: int, label: str) -> float:
    digest = hashlib.sha256(f"{int(seed)}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _signed_jitter(seed: int, label: str) -> float:
    return 2.0 * _hash_unit(seed, label) - 1.0


def _scaled_resolution(
    resolution: dict[str, int],
    *,
    scale: float,
    min_ntheta: int,
    min_nzeta: int,
    min_nx: int,
    min_nxi: int,
) -> dict[str, int]:
    def scaled_value(key: str, minimum: int) -> int:
        source = int(resolution.get(key, minimum))
        return max(int(minimum), int(round(source * float(scale))))

    out = {
        "NTHETA": scaled_value("NTHETA", min_ntheta),
        "NZETA": scaled_value("NZETA", min_nzeta),
        "NX": scaled_value("NX", min_nx),
        "NXI": scaled_value("NXI", min_nxi),
    }
    for key in ("NTHETA", "NZETA"):
        if int(resolution.get(key, out[key])) % 2 == 1 and out[key] % 2 == 0:
            out[key] += 1
    return out


def _resolve_equilibrium(input_path: Path, text: str) -> Path | None:
    raw = _read_string_parameter(text, "equilibriumFile")
    if not raw:
        return None
    candidate = Path(raw)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    relative = (input_path.parent / candidate).resolve()
    if relative.exists():
        return relative
    by_basename = input_path.parent / candidate.name
    if by_basename.exists():
        return by_basename.resolve()
    return None


def _case_command(case_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "sfincs_jax",
        "write-output",
        "--input",
        str(case_dir / "input.namelist"),
        "--out",
        str(case_dir / "sfincsOutput_jax.h5"),
        "--solver-trace",
        str(case_dir / "sfincsOutput_jax.solver_trace.json"),
    ]


def _materialize_case(
    *,
    seed: int,
    source_input: Path,
    source_text: str,
    source_resolution: dict[str, int],
    source_equilibrium: Path | None,
    out_root: Path,
    resolution_scale: float,
    min_ntheta: int,
    min_nzeta: int,
    min_nx: int,
    min_nxi: int,
    nu_jitter: float,
    er_jitter: float,
) -> dict[str, object]:
    case_name = f"qi_seed_{int(seed):04d}"
    case_dir = out_root / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    text = source_text
    resolution = _scaled_resolution(
        source_resolution,
        scale=resolution_scale,
        min_ntheta=min_ntheta,
        min_nzeta=min_nzeta,
        min_nx=min_nx,
        min_nxi=min_nxi,
    )
    for key, value in resolution.items():
        text = _replace_or_append_parameter(text, group="resolutionParameters", key=key, value=str(int(value)))

    base_nu = _read_number_parameter(source_text, "nu_n")
    base_er = _read_number_parameter(source_text, "Er")
    nu_factor = 1.0 + float(nu_jitter) * _signed_jitter(seed, "nu_n")
    er_delta = float(er_jitter) * _signed_jitter(seed, "Er")
    nu_value = None if base_nu is None else float(base_nu) * nu_factor
    er_value = None if base_er is None else float(base_er) + er_delta
    if nu_value is not None:
        text = _replace_or_append_parameter(text, group="physicsParameters", key="nu_n", value=f"{nu_value:.12g}")
    if er_value is not None:
        text = _replace_or_append_parameter(text, group="physicsParameters", key="Er", value=f"{er_value:.12g}")

    copied_equilibrium = None
    if source_equilibrium is not None:
        copied_equilibrium = case_dir / source_equilibrium.name
        if source_equilibrium.resolve() != copied_equilibrium.resolve():
            shutil.copy2(source_equilibrium, copied_equilibrium)
        text = _replace_or_append_parameter(
            text,
            group="geometryParameters",
            key="equilibriumFile",
            value=f"'{copied_equilibrium.name}'",
        )

    input_path = case_dir / "input.namelist"
    input_path.write_text(_normalize_text(text), encoding="utf-8")
    (case_dir / "input.source.namelist").write_text(_normalize_text(source_text), encoding="utf-8")
    command = _case_command(case_dir)
    return {
        "case": case_name,
        "seed": int(seed),
        "input": str(input_path.relative_to(out_root)),
        "output": str((case_dir / "sfincsOutput_jax.h5").relative_to(out_root)),
        "solver_trace": str((case_dir / "sfincsOutput_jax.solver_trace.json").relative_to(out_root)),
        "source_input": str(source_input),
        "source_equilibrium": str(source_equilibrium) if source_equilibrium is not None else None,
        "copied_equilibrium": str(copied_equilibrium.relative_to(out_root)) if copied_equilibrium is not None else None,
        "resolution": resolution,
        "perturbations": {
            "nu_n": nu_value,
            "nu_factor": nu_factor if nu_value is not None else None,
            "Er": er_value,
            "Er_delta": er_delta if er_value is not None else None,
        },
        "command": command,
    }


def _execute_cases(out_root: Path, cases: Iterable[dict[str, object]], *, timeout_s: float, fail_fast: bool) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for case in cases:
        command = [str(part) for part in case["command"]]  # type: ignore[index]
        case_dir = out_root / str(case["case"])
        stdout_path = case_dir / "sfincs_jax.stdout.log"
        stderr_path = case_dir / "sfincs_jax.stderr.log"
        start = time.perf_counter()
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                stdout=stdout,
                stderr=stderr,
                timeout=float(timeout_s),
                check=False,
            )
        elapsed_s = time.perf_counter() - start
        result = {
            "case": case["case"],
            "seed": case["seed"],
            "returncode": int(completed.returncode),
            "elapsed_s": elapsed_s,
            "stdout": str(stdout_path.relative_to(out_root)),
            "stderr": str(stderr_path.relative_to(out_root)),
        }
        results.append(result)
        if completed.returncode != 0 and fail_fast:
            break
    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_QI_INPUT, help="Base QI input.namelist.")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT, help="Directory for generated seed cases.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Deterministic seed ids to materialize.")
    parser.add_argument("--resolution-scale", type=float, default=0.25, help="Scale applied to NTHETA/NZETA/NX/NXI.")
    parser.add_argument("--min-ntheta", type=int, default=7)
    parser.add_argument("--min-nzeta", type=int, default=11)
    parser.add_argument("--min-nx", type=int, default=4)
    parser.add_argument("--min-nxi", type=int, default=16)
    parser.add_argument("--nu-jitter", type=float, default=0.05, help="Relative symmetric nu_n jitter per seed.")
    parser.add_argument("--er-jitter", type=float, default=0.02, help="Additive symmetric Er jitter per seed.")
    parser.add_argument("--execute", action="store_true", help="Run each generated seed through sfincs_jax write-output.")
    parser.add_argument("--timeout-s", type=float, default=300.0, help="Per-seed execution timeout.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop executing after the first failed seed.")
    parser.add_argument("--clean", action="store_true", help="Remove --out-root before materializing cases.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    source_input = Path(args.input).resolve()
    if not source_input.exists():
        raise FileNotFoundError(source_input)

    out_root = Path(args.out_root).resolve()
    if args.clean and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    source_text = source_input.read_text(encoding="utf-8")
    source_resolution = _read_resolution(source_text)
    source_equilibrium = _resolve_equilibrium(source_input, source_text)
    cases = [
        _materialize_case(
            seed=int(seed),
            source_input=source_input,
            source_text=source_text,
            source_resolution=source_resolution,
            source_equilibrium=source_equilibrium,
            out_root=out_root,
            resolution_scale=float(args.resolution_scale),
            min_ntheta=int(args.min_ntheta),
            min_nzeta=int(args.min_nzeta),
            min_nx=int(args.min_nx),
            min_nxi=int(args.min_nxi),
            nu_jitter=float(args.nu_jitter),
            er_jitter=float(args.er_jitter),
        )
        for seed in args.seeds
    ]
    manifest: dict[str, object] = {
        "schema_version": 1,
        "lane": "qi_seed_robustness",
        "source_input": str(source_input),
        "source_equilibrium": str(source_equilibrium) if source_equilibrium is not None else None,
        "resolution_scale": float(args.resolution_scale),
        "nu_jitter": float(args.nu_jitter),
        "er_jitter": float(args.er_jitter),
        "case_count": len(cases),
        "cases": cases,
    }
    if bool(args.execute):
        results = _execute_cases(out_root, cases, timeout_s=float(args.timeout_s), fail_fast=bool(args.fail_fast))
        manifest["execution"] = {
            "timeout_s": float(args.timeout_s),
            "fail_fast": bool(args.fail_fast),
            "results": results,
            "passed": sum(1 for result in results if int(result["returncode"]) == 0),
            "failed": sum(1 for result in results if int(result["returncode"]) != 0),
        }

    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path}")
    print(f"Cases: {len(cases)}")
    if bool(args.execute):
        execution = manifest["execution"]  # type: ignore[index]
        print(f"Executed: {execution['passed']} passed, {execution['failed']} failed")
        return 0 if int(execution["failed"]) == 0 else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
