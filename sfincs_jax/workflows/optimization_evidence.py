"""Evidence-campaign helpers for optimization promotion runs.

The proxy optimization lane is intentionally cheap and differentiable.  A
candidate only becomes a kinetic claim after the same electric-field scan has
been evaluated on the requested execution backends and, when available, against
the SFINCS Fortran v3 executable.  This module contains the reusable pieces for
that promotion campaign without making the public scripts depend on slow tests.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shlex
import sys
import time
from typing import Any

from ..fortran import run_sfincs_fortran
from ..io import localize_equilibrium_file_in_place
from ..namelist import read_sfincs_input
from ..scans import ScanResult, _er_scan_var_name, _patch_scalar_in_group


@dataclass(frozen=True)
class PromotionEvidenceLane:
    """One backend lane in a CPU/GPU/Fortran promotion campaign."""

    label: str
    backend: str
    scan_dir: Path
    promotion_dir: Path
    scan_command: tuple[str, ...] | None
    promotion_command: tuple[str, ...]
    env: dict[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "backend": self.backend,
            "scan_dir": str(self.scan_dir),
            "promotion_dir": str(self.promotion_dir),
            "scan_command": None if self.scan_command is None else list(self.scan_command),
            "scan_command_string": None
            if self.scan_command is None
            else shlex.join(self.scan_command),
            "promotion_command": list(self.promotion_command),
            "promotion_command_string": shlex.join(self.promotion_command),
            "env": dict(sorted(self.env.items())),
        }


@dataclass(frozen=True)
class PromotionEvidencePlan:
    """Serializable command plan for a promotion evidence campaign."""

    input_namelist: Path
    out_dir: Path
    er_values: tuple[float, ...]
    lanes: tuple[PromotionEvidenceLane, ...]
    comparison_command: tuple[str, ...] | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "workflow": "sfincs_jax_optimization_promotion_evidence_plan",
            "input_namelist": str(self.input_namelist),
            "out_dir": str(self.out_dir),
            "er_values": [float(value) for value in self.er_values],
            "lanes": [lane.as_dict() for lane in self.lanes],
            "comparison_command": None
            if self.comparison_command is None
            else list(self.comparison_command),
            "comparison_command_string": None
            if self.comparison_command is None
            else shlex.join(self.comparison_command),
            "claim_boundary": (
                "A campaign plan is execution provenance, not promotion evidence. "
                "Promotion requires completed scan outputs, passing residual and "
                "ambipolar gates, and passing backend/reference comparisons."
            ),
        }


def build_promotion_evidence_plan(
    *,
    input_namelist: str | Path,
    out_dir: str | Path,
    er_values: tuple[float, ...],
    include_cpu: bool = True,
    include_gpu: bool = False,
    include_fortran: bool = False,
    fortran_exe: str | Path | None = None,
    gpu_device: str | None = None,
    jobs: int = 1,
    compute_solution: bool = True,
    compute_transport_matrix: bool = False,
    skip_existing: bool = True,
    require_electron_root: bool = True,
    impurity_species_index: int | None = None,
    target_impurity_flux: float = 0.0,
    require_fortran_residuals: bool = False,
    promotion_stem: str = "candidate_promotion",
    compare_stem: str = "candidate_promotion_comparison",
) -> PromotionEvidencePlan:
    """Build commands for CPU/GPU/Fortran high-fidelity promotion evidence."""

    values = tuple(float(value) for value in er_values)
    if len(values) < 2:
        raise ValueError("at least two Er values are required")
    if int(jobs) < 1:
        raise ValueError("jobs must be >= 1")
    if not (include_cpu or include_gpu or include_fortran):
        raise ValueError("at least one backend lane must be requested")

    input_path = Path(input_namelist).resolve()
    root = Path(out_dir).resolve()
    lanes: list[PromotionEvidenceLane] = []

    if include_cpu:
        lanes.append(
            _build_jax_lane(
                label="cpu",
                input_path=input_path,
                out_dir=root,
                values=values,
                env={"JAX_PLATFORM_NAME": "cpu"},
                jobs=int(jobs),
                compute_solution=compute_solution,
                compute_transport_matrix=compute_transport_matrix,
                skip_existing=skip_existing,
                require_electron_root=require_electron_root,
                impurity_species_index=impurity_species_index,
                target_impurity_flux=target_impurity_flux,
                require_residuals=True,
                promotion_stem=promotion_stem,
            )
        )

    if include_gpu:
        env = {"JAX_PLATFORM_NAME": "gpu"}
        if gpu_device is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
        lanes.append(
            _build_jax_lane(
                label="gpu",
                input_path=input_path,
                out_dir=root,
                values=values,
                env=env,
                jobs=int(jobs),
                compute_solution=compute_solution,
                compute_transport_matrix=compute_transport_matrix,
                skip_existing=skip_existing,
                require_electron_root=require_electron_root,
                impurity_species_index=impurity_species_index,
                target_impurity_flux=target_impurity_flux,
                require_residuals=True,
                promotion_stem=promotion_stem,
            )
        )

    if include_fortran:
        env: dict[str, str] = {}
        if fortran_exe is not None:
            env["SFINCS_FORTRAN_EXE"] = str(Path(fortran_exe).expanduser())
        lane_dir = root / "fortran_v3_scan"
        lanes.append(
            PromotionEvidenceLane(
                label="fortran_v3",
                backend="fortran_v3",
                scan_dir=lane_dir,
                promotion_dir=root / "fortran_v3_promotion",
                scan_command=None,
                promotion_command=_promotion_command(
                    scan_dir=lane_dir,
                    promotion_dir=root / "fortran_v3_promotion",
                    promotion_stem=promotion_stem,
                    require_electron_root=require_electron_root,
                    impurity_species_index=impurity_species_index,
                    target_impurity_flux=target_impurity_flux,
                    require_residuals=bool(require_fortran_residuals),
                ),
                env=env,
            )
        )

    comparison_command = _comparison_command(
        lanes=tuple(lanes),
        out_dir=root / "comparison",
        stem=compare_stem,
        promotion_stem=promotion_stem,
        require_flux_objective=impurity_species_index is not None,
    )
    return PromotionEvidencePlan(
        input_namelist=input_path,
        out_dir=root,
        er_values=values,
        lanes=tuple(lanes),
        comparison_command=comparison_command,
    )


def write_promotion_evidence_plan(path: str | Path, plan: PromotionEvidencePlan) -> Path:
    """Write a promotion evidence plan to JSON."""

    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def prepare_fortran_er_scan_inputs(
    *,
    input_namelist: str | Path,
    out_dir: str | Path,
    values: tuple[float, ...],
) -> tuple[Path, ...]:
    """Write Fortran-v3 Er-scan input directories without executing solves."""

    input_path = Path(input_namelist).resolve()
    root = Path(out_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    template_txt = input_path.read_text(encoding="utf-8")
    nml = read_sfincs_input(input_path)
    var = _er_scan_var_name(nml=nml)
    vals = tuple(sorted((float(value) for value in values), reverse=True))

    scan_txt = template_txt
    if not scan_txt.endswith("\n"):
        scan_txt += "\n"
    scan_txt += f"!ss NErs = {len(vals)}\n"
    scan_txt += f"!ss {var}Min = {min(vals):.16g}\n"
    scan_txt += f"!ss {var}Max = {max(vals):.16g}\n"
    (root / "input.namelist").write_text(scan_txt, encoding="utf-8")

    input_paths: list[Path] = []
    for value in vals:
        run_dir = root / f"{var}{value:.4g}"
        run_dir.mkdir(parents=True, exist_ok=True)
        patched = _patch_scalar_in_group(
            txt=template_txt,
            group="physicsParameters",
            key=var,
            value=float(value),
        )
        run_input = run_dir / "input.namelist"
        run_input.write_text(patched, encoding="utf-8")
        localize_equilibrium_file_in_place(input_namelist=run_input, overwrite=False)
        input_paths.append(run_input)
    return tuple(input_paths)


def run_fortran_er_scan(
    *,
    input_namelist: str | Path,
    out_dir: str | Path,
    values: tuple[float, ...],
    exe: str | Path | None = None,
    timeout_s: float | None = None,
    skip_existing: bool = True,
    emit: Any | None = None,
) -> ScanResult:
    """Run the same Er scan with the external SFINCS Fortran v3 executable."""

    root = Path(out_dir).resolve()
    input_paths = prepare_fortran_er_scan_inputs(
        input_namelist=input_namelist,
        out_dir=root,
        values=values,
    )
    outputs: list[Path] = []
    run_dirs: list[Path] = []
    var = _er_scan_var_name(nml=read_sfincs_input(Path(input_namelist).resolve()))
    total = len(input_paths)
    t0 = time.perf_counter()
    for idx, run_input in enumerate(input_paths, start=1):
        run_dir = run_input.parent
        output_path = run_dir / "sfincsOutput.h5"
        run_dirs.append(run_dir)
        if bool(skip_existing) and output_path.exists():
            if emit is not None:
                emit(0, f"fortran-scan: progress {idx}/{total} {run_dir.name} reused existing output")
            outputs.append(output_path)
            continue
        if emit is not None:
            emit(0, f"fortran-scan: [{idx}/{total}] {run_dir.name}")
        run_sfincs_fortran(
            input_namelist=run_input,
            exe=None if exe is None else Path(exe).expanduser(),
            workdir=run_dir,
            timeout_s=timeout_s,
        )
        if not output_path.exists():
            raise FileNotFoundError(f"Fortran v3 did not create {output_path}")
        outputs.append(output_path)
        if emit is not None:
            elapsed = time.perf_counter() - t0
            remaining = total - idx
            avg = elapsed / max(idx, 1)
            emit(
                0,
                f"fortran-scan: progress {idx}/{total} {run_dir.name} "
                f"avg_point={avg:.1f}s est_remaining={avg * remaining:.1f}s",
            )
    vals = tuple(sorted((float(value) for value in values), reverse=True))
    return ScanResult(
        scan_dir=root,
        run_dirs=tuple(run_dirs),
        outputs=tuple(outputs),
        variable=var,
        values=vals,
    )


def _build_jax_lane(
    *,
    label: str,
    input_path: Path,
    out_dir: Path,
    values: tuple[float, ...],
    env: dict[str, str],
    jobs: int,
    compute_solution: bool,
    compute_transport_matrix: bool,
    skip_existing: bool,
    require_electron_root: bool,
    impurity_species_index: int | None,
    target_impurity_flux: float,
    require_residuals: bool,
    promotion_stem: str,
) -> PromotionEvidenceLane:
    scan_dir = out_dir / f"{label}_scan"
    promotion_dir = out_dir / f"{label}_promotion"
    scan_cmd = [
        sys.executable,
        "-m",
        "sfincs_jax",
        "scan-er",
        "--input",
        str(input_path),
        "--out-dir",
        str(scan_dir),
        "--values",
        *[f"{value:.16g}" for value in values],
    ]
    if compute_solution:
        scan_cmd.append("--compute-solution")
    if compute_transport_matrix:
        scan_cmd.append("--compute-transport-matrix")
    if skip_existing:
        scan_cmd.append("--skip-existing")
    if int(jobs) > 1:
        scan_cmd.extend(["--jobs", str(int(jobs))])
    return PromotionEvidenceLane(
        label=label,
        backend="sfincs_jax",
        scan_dir=scan_dir,
        promotion_dir=promotion_dir,
        scan_command=tuple(scan_cmd),
        promotion_command=_promotion_command(
            scan_dir=scan_dir,
            promotion_dir=promotion_dir,
            promotion_stem=promotion_stem,
            require_electron_root=require_electron_root,
            impurity_species_index=impurity_species_index,
            target_impurity_flux=target_impurity_flux,
            require_residuals=bool(require_residuals),
        ),
        env=dict(env),
    )


def _promotion_command(
    *,
    scan_dir: Path,
    promotion_dir: Path,
    promotion_stem: str,
    require_electron_root: bool,
    impurity_species_index: int | None,
    target_impurity_flux: float,
    require_residuals: bool,
) -> tuple[str, ...]:
    cmd = [
        sys.executable,
        "examples/optimization/evaluate_sfincs_jax_promotion_scan.py",
        "--scan-dir",
        str(scan_dir),
        "--out-dir",
        str(promotion_dir),
        "--stem",
        str(promotion_stem),
        "--target-impurity-flux",
        f"{float(target_impurity_flux):.16g}",
    ]
    if require_electron_root:
        cmd.append("--require-electron-root")
    else:
        cmd.append("--allow-no-electron-root")
    if impurity_species_index is not None:
        cmd.extend(["--impurity-species-index", str(int(impurity_species_index))])
    if not bool(require_residuals):
        cmd.append("--allow-missing-residuals")
    return tuple(cmd)


def _comparison_command(
    *,
    lanes: tuple[PromotionEvidenceLane, ...],
    out_dir: Path,
    stem: str,
    promotion_stem: str,
    require_flux_objective: bool,
) -> tuple[str, ...] | None:
    by_label = {lane.label: lane for lane in lanes}
    if "cpu" not in by_label or "gpu" not in by_label:
        return None
    cmd = [
        sys.executable,
        "examples/optimization/compare_sfincs_jax_promotion_runs.py",
        "--cpu",
        str(by_label["cpu"].promotion_dir / f"{promotion_stem}.json"),
        "--gpu",
        str(by_label["gpu"].promotion_dir / f"{promotion_stem}.json"),
        "--out-dir",
        str(out_dir),
        "--stem",
        str(stem),
    ]
    if "fortran_v3" in by_label:
        cmd.extend(
            [
                "--fortran",
                str(by_label["fortran_v3"].promotion_dir / f"{promotion_stem}.json"),
            ]
        )
    if not bool(require_flux_objective):
        cmd.append("--allow-missing-flux")
    return tuple(cmd)


__all__ = [
    "PromotionEvidenceLane",
    "PromotionEvidencePlan",
    "build_promotion_evidence_plan",
    "prepare_fortran_er_scan_inputs",
    "run_fortran_er_scan",
    "write_promotion_evidence_plan",
]
