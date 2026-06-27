#!/usr/bin/env python
"""Create production-resolution SFINCS-JAX benchmark inputs.

This script separates fast parity/smoke examples from the benchmark tier used
for runtime and memory claims. It copies selected input decks into a standalone
tree, raises under-resolved cases to a documented minimum grid, and writes a
machine-readable manifest for CPU/GPU/Fortran benchmark runners.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = REPO_ROOT / "benchmarks" / "production_resolution_inputs_2026-05-04"
DEFAULT_EXAMPLES_ROOT = REPO_ROOT / "examples" / "sfincs_examples"
DEFAULT_ADDITIONAL_INPUT = REPO_ROOT / "examples" / "data" / "qi_nfp2_reference.input.namelist"
# Archived downstream NTX decks can still be imported explicitly for local
# reproduction, but the production benchmark tier is SFINCS_JAX-owned by
# default and should not depend on another repository.
DEFAULT_ARCHIVED_NTX_INPUTS = (
    Path(
        "/Users/rogeriojorge/local/NTX/examples/outputs/"
        "owned_finite_beta_sfincs_jax_profile_current_audit/"
        "finite_beta_qa_pressure_current/rho_0p142857/nu_n_0p00831565/input.namelist"
    ),
    Path(
        "/Users/rogeriojorge/local/NTX/examples/outputs/"
        "sfincs_jax_rhsmode1_profile_current_profiling/cpu_17x21x12_deck/"
        "finite_beta_qa_pressure_current/rho_0p142857/nu_n_0p00831565/input.namelist"
    ),
    Path(
        "/Users/rogeriojorge/local/NTX/examples/outputs/"
        "owned_finite_beta_sfincs_jax_inputs/finite_beta_qa_pressure_current/"
        "rho_0p5/nuPrime_0p01/EStar_0/input.namelist"
    ),
    Path(
        "/Users/rogeriojorge/local/NTX/examples/outputs/"
        "owned_finite_beta_sfincs_jax_inputs/finite_beta_nfp3_qh_stage1/"
        "rho_0p5/nuPrime_0p01/EStar_0/input.namelist"
    ),
    Path(
        "/Users/rogeriojorge/local/NTX/examples/outputs/"
        "owned_finite_beta_sfincs_jax_inputs_production_probe/grid_35_43_48/"
        "finite_beta_qa_pressure_current/rho_0p142857/nuPrime_0p01/EStar_0/input.namelist"
    ),
)

RESOLUTION_KEYS = ("NTHETA", "NZETA", "NX", "NXI")
DEFAULT_3D_MINIMUM = {"NTHETA": 25, "NZETA": 51, "NX": 4, "NXI": 100}
DEFAULT_TOKAMAK_MINIMUM = {"NTHETA": 33, "NX": 12, "NXI": 140}
DEFAULT_TOKAMAK_PAS_NOER_MINIMUM = {"NTHETA": 89, "NX": 24, "NXI": 300}
DEFAULT_TARGET_FORTRAN_MIN_RUNTIME_S = 10.0


def _read_resolution(text: str) -> dict[str, int]:
    resolution: dict[str, int] = {}
    for key in RESOLUTION_KEYS:
        match = re.search(rf"(?im)^\s*{key}\s*=\s*([-+0-9.eEdD]+)", text)
        if match is None:
            continue
        token = match.group(1).replace("D", "E").replace("d", "e")
        try:
            resolution[key] = int(float(token))
        except ValueError:
            continue
    return resolution


def _read_int_parameter(text: str, key: str, default: int | None = None) -> int | None:
    match = re.search(rf"(?im)^\s*{key}\s*=\s*([-+0-9.eEdD]+)", text)
    if match is None:
        return default
    token = match.group(1).replace("D", "E").replace("d", "e")
    try:
        return int(float(token))
    except ValueError:
        return default


def _read_float_parameter(text: str, key: str, default: float | None = None) -> float | None:
    match = re.search(rf"(?im)^\s*{key}\s*=\s*([-+0-9.eEdD]+)", text)
    if match is None:
        return default
    token = match.group(1).replace("D", "E").replace("d", "e")
    try:
        return float(token)
    except ValueError:
        return default


def _read_logical_parameter(text: str, key: str, default: bool = False) -> bool:
    match = re.search(rf"(?im)^\s*{key}\s*=\s*(\.[TtFf][A-Za-z]*\.)", text)
    if match is None:
        return bool(default)
    return match.group(1).lower().startswith(".t")


def _count_parameter_values(text: str, key: str, default: int = 1) -> int:
    match = re.search(rf"(?im)^\s*{key}\s*=\s*([^!\n/]+)", text)
    if match is None:
        return int(default)
    tokens = [token for token in re.split(r"[,\s]+", match.group(1).strip()) if token]
    return max(1, len(tokens))


def _replace_or_append_resolution(text: str, updates: dict[str, int]) -> str:
    updated = text
    missing: dict[str, int] = {}
    for key, value in updates.items():
        pattern = re.compile(rf"(?im)^(\s*{key}\s*=\s*)([-+0-9.eEdD]+)(.*)$")
        if pattern.search(updated):
            updated = pattern.sub(rf"\g<1>{int(value)}\3", updated, count=1)
        else:
            missing[key] = int(value)
    if not missing:
        return updated
    block = "\n&resolutionParameters\n" + "".join(f"  {key} = {value}\n" for key, value in missing.items()) + "/\n"
    return updated.rstrip() + block


def _replace_or_append_namelist_parameter(text: str, *, group: str, key: str, value: str) -> str:
    """Replace a scalar namelist parameter or append it to the requested group."""
    pattern = re.compile(rf"(?im)^(\s*{re.escape(key)}\s*=\s*)([^!\n/]+)(.*)$")
    if pattern.search(text):
        return pattern.sub(rf"\g<1>{value}\3", text, count=1)

    group_pattern = re.compile(rf"(?ims)(^\s*&{re.escape(group)}\b.*?)(^\s*/\s*$)")
    match = group_pattern.search(text)
    if match is not None:
        return text[: match.start(2)] + f"  {key} = {value}\n" + text[match.start(2) :]

    return text.rstrip() + f"\n\n&{group}\n  {key} = {value}\n/\n"


def _apply_benchmark_solver_hints(text: str) -> str:
    """Apply non-physics solver hints needed for robust production benchmarks.

    The public benchmark floor lifts several old small tokamak FP decks far above
    their authored resolution. SFINCS v3 itself warns that ``preconditioner_x = 1``
    is usually the best option; without it, the full ``Nxi=100`` no-Er FP deck can
    run to the PETSc iteration ceiling and abort before a reference comparison is
    possible. This changes only the linear-solver preconditioner, not the physics
    model or requested outputs.
    """

    rhs_mode = int(_read_int_parameter(text, "RHSMode", 1) or 1)
    collision_operator = int(_read_int_parameter(text, "collisionOperator", 0) or 0)
    if rhs_mode == 1 and collision_operator == 0:
        text = _replace_or_append_namelist_parameter(
            text,
            group="preconditionerOptions",
            key="preconditioner_x",
            value="1",
        )
        text = _replace_or_append_namelist_parameter(
            text,
            group="otherNumericalParameters",
            key="whichParallelSolverToFactorPreconditioner",
            value="1",
        )
        resolution = _read_resolution(text)
        phase_points = _phase_points(resolution) or 0
        tokamak_like = int(resolution.get("NZETA", 1)) == 1
        include_phi1 = _read_logical_parameter(text, "includePhi1", False)
        if tokamak_like and (not include_phi1) and int(phase_points) <= 30_000:
            text = _replace_or_append_namelist_parameter(
                text,
                group="otherNumericalParameters",
                key="useIterativeLinearSolver",
                value=".false.",
            )
    return text


def _normalize_generated_text(text: str) -> str:
    """Return deterministic text for checked-in benchmark input artifacts."""
    return "\n".join(line.rstrip() for line in str(text).splitlines()).rstrip() + "\n"


def _normalize_text_file_if_possible(path: Path) -> None:
    if path.suffix.lower() in {".h5", ".hdf5", ".nc", ".npz", ".npy"}:
        return
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return
    path.write_text(_normalize_generated_text(text), encoding="utf-8")


def _safe_case_name(prefix: str, input_path: Path) -> str:
    parts = [part for part in input_path.resolve().parts if part not in {"", "/"}]
    tail = "_".join(parts[-6:-1])
    raw = f"{prefix}_{tail}" if tail else f"{prefix}_{input_path.parent.name}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_")


def _case_kind(case: str, resolution: dict[str, int]) -> str:
    n_zeta = int(resolution.get("NZETA", 1))
    if "tokamak" in case.lower() or n_zeta == 1:
        return "tokamak"
    return "3d"


def _benchmark_resolution(
    resolution: dict[str, int],
    *,
    kind: str,
    text: str,
    min_3d: dict[str, int],
    min_tokamak: dict[str, int],
    min_tokamak_pas_noer: dict[str, int],
) -> dict[str, int]:
    updated = dict(resolution)
    minimum = min_tokamak if kind == "tokamak" else min_3d
    if kind == "tokamak":
        rhs_mode = int(_read_int_parameter(text, "RHSMode", 1) or 1)
        collision_operator = int(_read_int_parameter(text, "collisionOperator", 0) or 0)
        if rhs_mode == 1 and collision_operator == 1 and not _effective_xdot_requested(text):
            minimum = {
                key: max(int(minimum.get(key, 0)), int(value))
                for key, value in min_tokamak_pas_noer.items()
            }
    for key, value in minimum.items():
        if key in updated:
            updated[key] = max(int(updated[key]), int(value))
        else:
            updated[key] = int(value)
    return updated


def _copy_parent_files(src_input: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for item in src_input.parent.iterdir():
        if item.is_file():
            target = dst_dir / item.name
            if item.resolve() == target.resolve():
                continue
            shutil.copy2(item, target)
            _normalize_text_file_if_possible(target)


def _copy_referenced_absolute_files(text: str, dst_dir: Path) -> None:
    """Copy absolute file references in an input deck when they exist locally."""
    for match in re.finditer(r"['\"](?P<path>/[^'\"]+?)['\"]", text):
        path = Path(match.group("path"))
        if not path.is_file():
            continue
        target = dst_dir / path.name
        if path.resolve() == target.resolve():
            continue
        shutil.copy2(path, target)
        _normalize_text_file_if_possible(target)


def _localize_copied_absolute_paths(text: str, dst_dir: Path) -> str:
    """Replace absolute file paths with copied sibling basenames when possible."""

    def repl(match: re.Match[str]) -> str:
        quote = match.group("quote")
        raw = match.group("path")
        path = Path(raw)
        if not path.is_absolute():
            return match.group(0)
        sibling = dst_dir / path.name
        if not sibling.exists():
            return match.group(0)
        return f"{quote}{path.name}{quote}"

    return re.sub(
        r"(?P<quote>['\"])(?P<path>/[^'\"]+?)(?P=quote)",
        repl,
        text,
    )


def _phase_points(resolution: dict[str, int]) -> int | None:
    try:
        return int(resolution["NTHETA"]) * int(resolution.get("NZETA", 1)) * int(resolution["NX"]) * int(resolution["NXI"])
    except KeyError:
        return None


def _average_legendre_bandwidth(n_xi: int, *, radius: int = 2) -> float:
    n_xi = max(1, int(n_xi))
    total = 0
    for ell in range(n_xi):
        total += min(n_xi, ell + int(radius) + 1) - max(0, ell - int(radius))
    return float(total) / float(n_xi)


def _effective_xdot_requested(text: str) -> bool:
    """Return whether `includeXDotTerm` implies a nonzero electric-field x-coupling."""
    if not _read_logical_parameter(text, "includeXDotTerm", False):
        return False
    for key in ("dPhiHatdpsiHat", "dPhiHatdpsiN", "dPhiHatdrHat", "dPhiHatdrN", "Er"):
        value = _read_float_parameter(text, key, None)
        if value is not None and abs(float(value)) > 1.0e-15:
            return True
    return False


def _estimate_case_size(text: str, resolution: dict[str, int]) -> dict[str, object]:
    """Return a conservative matrix-size estimate for benchmark scheduling.

    This is a preflight estimate only. It intentionally over-approximates the
    sparse pattern in the same direction as the sparse-host builder so that
    benchmark launchers can avoid unsafe dense or sparse-direct runs.
    """

    n_theta = int(resolution["NTHETA"])
    n_zeta = int(resolution.get("NZETA", 1))
    n_x = int(resolution["NX"])
    n_xi = int(resolution["NXI"])
    n_species = _count_parameter_values(text, "Zs")
    collision_operator = int(_read_int_parameter(text, "collisionOperator", 0) or 0)
    include_xdot_requested = _read_logical_parameter(text, "includeXDotTerm", False)
    include_xdot = _effective_xdot_requested(text)
    include_phi1 = _read_logical_parameter(text, "includePhi1", False)
    constraint_scheme = _read_int_parameter(text, "constraintScheme", None)
    if constraint_scheme is None:
        constraint_scheme = 1 if collision_operator == 0 else 2

    f_unknowns = int(n_species * n_x * n_xi * n_theta * n_zeta)
    phi1_unknowns = int(n_theta * n_zeta) if include_phi1 else 0
    if constraint_scheme == 2:
        extra_unknowns = int(n_species * n_x)
    elif constraint_scheme in {1, 3, 4}:
        extra_unknowns = int(2 * n_species)
    else:
        extra_unknowns = 0
    total_unknowns = int(f_unknowns + phi1_unknowns + extra_unknowns)

    spatial_stencil = int(n_theta + n_zeta - 1)
    if collision_operator == 0:
        avg_row_nnz = float(n_species * n_x * n_xi * spatial_stencil)
    else:
        x_band = n_x if include_xdot else 1
        avg_row_nnz = float(x_band * _average_legendre_bandwidth(n_xi) * spatial_stencil)
    conservative_nnz = int(f_unknowns * avg_row_nnz + extra_unknowns * n_theta * n_zeta)
    dense_nbytes = int(total_unknowns * total_unknowns * 8)
    csr_nbytes = int(conservative_nnz * (8 + 4) + (total_unknowns + 1) * 4)

    run_recommendation = "bounded_remote"
    if total_unknowns < 100_000 and csr_nbytes < 2_000_000_000:
        run_recommendation = "bounded_local_ok"
    elif csr_nbytes > 8_000_000_000 or total_unknowns > 250_000:
        run_recommendation = "remote_or_cluster_only"

    return {
        "species_count": int(n_species),
        "collision_operator": int(collision_operator),
        "include_xdot_requested": bool(include_xdot_requested),
        "include_xdot": bool(include_xdot),
        "include_xdot_effective": bool(include_xdot),
        "include_phi1": bool(include_phi1),
        "constraint_scheme": int(constraint_scheme),
        "f_unknowns_estimate": int(f_unknowns),
        "phi1_unknowns_estimate": int(phi1_unknowns),
        "extra_unknowns_estimate": int(extra_unknowns),
        "total_unknowns_estimate": int(total_unknowns),
        "dense_matrix_nbytes_estimate": int(dense_nbytes),
        "conservative_sparse_nnz_estimate": int(conservative_nnz),
        "conservative_csr_nbytes_estimate": int(csr_nbytes),
        "run_recommendation": run_recommendation,
    }


def _resolution_policy_text(
    *,
    enforce_minimum: bool,
    min_3d: dict[str, int],
    min_tokamak: dict[str, int],
    min_tokamak_pas_noer: dict[str, int],
    target_fortran_min_runtime_s: float,
) -> str:
    if not enforce_minimum:
        return "preserve authored external resolution"
    min_3d_text = (
        f"{int(min_3d['NTHETA'])}x{int(min_3d['NZETA'])}x"
        f"{int(min_3d['NX'])}x{int(min_3d['NXI'])}"
    )
    min_tokamak_text = (
        f"{int(min_tokamak['NTHETA'])}x1x"
        f"{int(min_tokamak['NX'])}x{int(min_tokamak['NXI'])}"
    )
    min_tokamak_pas_noer_text = (
        f"{int(min_tokamak_pas_noer['NTHETA'])}x1x"
        f"{int(min_tokamak_pas_noer['NX'])}x{int(min_tokamak_pas_noer['NXI'])}"
    )
    return (
        f"preserve nominal grid, but enforce 3D >= {min_3d_text} "
        f"and tokamak >= {min_tokamak_text} "
        f"(RHSMode=1 PAS/no-Er tokamak >= {min_tokamak_pas_noer_text}); "
        f"production timing rows target "
        f"Fortran v3 >= {float(target_fortran_min_runtime_s):g} s"
    )


def _case_name_with_benchmark_resolution(case: str, resolution: dict[str, int]) -> str:
    try:
        token = (
            f"{int(resolution['NTHETA'])}x{int(resolution['NZETA'])}x"
            f"{int(resolution['NX'])}x{int(resolution['NXI'])}"
        )
    except KeyError:
        return case
    return re.sub(r"(?i)cpu_\d+x\d+x\d+(?:x\d+)?_deck", f"cpu_{token}_deck", case)


def _iter_example_inputs(examples_root: Path, additional_input: Path | None) -> Iterable[tuple[str, Path]]:
    for path in sorted(Path(examples_root).glob("*/input.namelist")):
        yield path.parent.name, path
    if additional_input is not None and Path(additional_input).exists():
        yield "additional_examples", Path(additional_input)


def _build_entry(
    *,
    case: str,
    src_input: Path,
    dst_root: Path,
    source_group: str,
    min_3d: dict[str, int],
    min_tokamak: dict[str, int],
    min_tokamak_pas_noer: dict[str, int],
    enforce_minimum: bool,
    target_fortran_min_runtime_s: float,
) -> dict[str, object] | None:
    src_input = Path(src_input).resolve()
    if not src_input.exists():
        return None
    text = src_input.read_text(encoding="utf-8", errors="replace")
    original = _read_resolution(text)
    kind = _case_kind(case, original)
    if source_group == "examples" and kind not in {"tokamak", "3d"}:
        return None
    benchmark = (
        _benchmark_resolution(
            original,
            kind=kind,
            text=text,
            min_3d=min_3d,
            min_tokamak=min_tokamak,
            min_tokamak_pas_noer=min_tokamak_pas_noer,
        )
        if enforce_minimum
        else dict(original)
    )
    case = _case_name_with_benchmark_resolution(case, benchmark)
    dst_dir = dst_root / "inputs" / case
    _copy_parent_files(src_input, dst_dir)
    _copy_referenced_absolute_files(text, dst_dir)
    dst_input = dst_dir / "input.namelist"
    text = _localize_copied_absolute_paths(text, dst_dir)
    generated_text = _replace_or_append_resolution(text, benchmark)
    generated_text = _apply_benchmark_solver_hints(generated_text)
    dst_input.write_text(_normalize_generated_text(generated_text), encoding="utf-8")
    size_estimate = _estimate_case_size(text, benchmark)
    return {
        "case": case,
        "source_group": source_group,
        "kind": kind,
        "source_input": str(src_input),
        "input": str(dst_input.relative_to(dst_root)),
        "original_resolution": original,
        "benchmark_resolution": benchmark,
        "phase_points": _phase_points(benchmark),
        "size_estimate": size_estimate,
        "resolution_policy": _resolution_policy_text(
            enforce_minimum=enforce_minimum,
            min_3d=min_3d,
            min_tokamak=min_tokamak,
            min_tokamak_pas_noer=min_tokamak_pas_noer,
            target_fortran_min_runtime_s=float(target_fortran_min_runtime_s),
        ),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples-root", type=Path, default=DEFAULT_EXAMPLES_ROOT)
    parser.add_argument("--additional-input", type=Path, default=DEFAULT_ADDITIONAL_INPUT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument(
        "--external-input",
        type=Path,
        action="append",
        default=[],
        help="Additional production-resolution input deck to copy into the manifest.",
    )
    parser.add_argument(
        "--include-archived-ntx-defaults",
        action="store_true",
        help=(
            "Compatibility/debug option: also import archived downstream NTX decks "
            "when they exist locally. Not used by the default SFINCS_JAX benchmark tier."
        ),
    )
    parser.add_argument(
        "--include-ntx-defaults",
        dest="include_archived_ntx_defaults",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--ntx-input",
        dest="external_input",
        type=Path,
        action="append",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--enforce-minimum-on-external",
        action="store_true",
        default=True,
        help="Also lift external inputs to the production minimum instead of preserving authored grids.",
    )
    parser.add_argument(
        "--enforce-minimum-on-ntx",
        dest="enforce_minimum_on_external",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--preserve-external-resolution",
        dest="enforce_minimum_on_external",
        action="store_false",
        help="Compatibility/debug mode: keep authored external grids instead of lifting them to the production minimum.",
    )
    parser.add_argument(
        "--preserve-ntx-resolution",
        dest="enforce_minimum_on_external",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--min-3d-ntheta", type=int, default=DEFAULT_3D_MINIMUM["NTHETA"])
    parser.add_argument("--min-3d-nzeta", type=int, default=DEFAULT_3D_MINIMUM["NZETA"])
    parser.add_argument("--min-3d-nx", type=int, default=DEFAULT_3D_MINIMUM["NX"])
    parser.add_argument("--min-3d-nxi", type=int, default=DEFAULT_3D_MINIMUM["NXI"])
    parser.add_argument("--min-tokamak-ntheta", type=int, default=DEFAULT_TOKAMAK_MINIMUM["NTHETA"])
    parser.add_argument("--min-tokamak-nx", type=int, default=DEFAULT_TOKAMAK_MINIMUM["NX"])
    parser.add_argument("--min-tokamak-nxi", type=int, default=DEFAULT_TOKAMAK_MINIMUM["NXI"])
    parser.add_argument(
        "--min-tokamak-pas-noer-ntheta",
        type=int,
        default=DEFAULT_TOKAMAK_PAS_NOER_MINIMUM["NTHETA"],
    )
    parser.add_argument(
        "--min-tokamak-pas-noer-nx",
        type=int,
        default=DEFAULT_TOKAMAK_PAS_NOER_MINIMUM["NX"],
    )
    parser.add_argument(
        "--min-tokamak-pas-noer-nxi",
        type=int,
        default=DEFAULT_TOKAMAK_PAS_NOER_MINIMUM["NXI"],
    )
    parser.add_argument(
        "--target-fortran-min-runtime-s",
        type=float,
        default=DEFAULT_TARGET_FORTRAN_MIN_RUNTIME_S,
        help=(
            "Documented Fortran v3 lower runtime target for production benchmark rows. "
            "The fixed resolution floor is chosen to make public timing cases nontrivial; "
            "benchmark runners should still report the measured per-case runtime."
        ),
    )
    parser.add_argument("--clean", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    out_root = Path(args.out_root).resolve()
    if args.clean and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    min_3d = {
        "NTHETA": int(args.min_3d_ntheta),
        "NZETA": int(args.min_3d_nzeta),
        "NX": int(args.min_3d_nx),
        "NXI": int(args.min_3d_nxi),
    }
    min_tokamak = {
        "NTHETA": int(args.min_tokamak_ntheta),
        "NX": int(args.min_tokamak_nx),
        "NXI": int(args.min_tokamak_nxi),
    }
    min_tokamak_pas_noer = {
        "NTHETA": int(args.min_tokamak_pas_noer_ntheta),
        "NX": int(args.min_tokamak_pas_noer_nx),
        "NXI": int(args.min_tokamak_pas_noer_nxi),
    }
    target_fortran_min_runtime_s = float(args.target_fortran_min_runtime_s)
    entries: list[dict[str, object]] = []
    for case, src_input in _iter_example_inputs(Path(args.examples_root), Path(args.additional_input)):
        entry = _build_entry(
            case=case,
            src_input=src_input,
            dst_root=out_root,
            source_group="examples",
            min_3d=min_3d,
            min_tokamak=min_tokamak,
            min_tokamak_pas_noer=min_tokamak_pas_noer,
            enforce_minimum=True,
            target_fortran_min_runtime_s=target_fortran_min_runtime_s,
        )
        if entry is not None:
            entries.append(entry)
    external_inputs = list(args.external_input or [])
    if args.include_archived_ntx_defaults:
        external_inputs.extend(path for path in DEFAULT_ARCHIVED_NTX_INPUTS if path.exists())
    for src_input in external_inputs:
        case = _safe_case_name("external", Path(src_input))
        entry = _build_entry(
            case=case,
            src_input=Path(src_input),
            dst_root=out_root,
            source_group="external",
            min_3d=min_3d,
            min_tokamak=min_tokamak,
            min_tokamak_pas_noer=min_tokamak_pas_noer,
            enforce_minimum=bool(args.enforce_minimum_on_external),
            target_fortran_min_runtime_s=target_fortran_min_runtime_s,
        )
        if entry is not None:
            entries.append(entry)
    entries.sort(key=lambda item: (str(item["source_group"]), int(item["phase_points"] or 0), str(item["case"])))
    manifest = {
        "schema_version": 1,
        "minimum_3d_resolution": min_3d,
        "minimum_tokamak_resolution": min_tokamak,
        "minimum_tokamak_pas_noer_resolution": min_tokamak_pas_noer,
        "target_fortran_min_runtime_s": target_fortran_min_runtime_s,
        "resolution_policy": _resolution_policy_text(
            enforce_minimum=True,
            min_3d=min_3d,
            min_tokamak=min_tokamak,
            min_tokamak_pas_noer=min_tokamak_pas_noer,
            target_fortran_min_runtime_s=target_fortran_min_runtime_s,
        ),
        "case_count": len(entries),
        "cases": entries,
    }
    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path}")
    print(f"Cases: {len(entries)}")
    if entries:
        largest = sorted(entries, key=lambda item: int(item["phase_points"] or 0), reverse=True)[:8]
        print("Largest cases by Ntheta*Nzeta*Nx*Nxi:")
        for item in largest:
            print(f"- {item['case']}: {item['benchmark_resolution']} phase_points={item['phase_points']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
