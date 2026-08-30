"""Electric-field scan workflows with progress reporting and output writing."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import re
import concurrent.futures
import subprocess
import sys
import textwrap
import time

import numpy as np

from ..io import localize_equilibrium_file_in_place
from ..run import run_from_namelist
from ..namelist import Namelist, read_sfincs_input
from ..paths import repository_root


EmitFn = Callable[[int, str], None]


@dataclass(frozen=True)
class ScanResult:
    scan_dir: Path
    run_dirs: tuple[Path, ...]
    outputs: tuple[Path, ...]
    variable: str
    values: tuple[float, ...]


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h{minutes:02d}m"
    days, hours = divmod(hours, 24)
    return f"{days}d{hours:02d}h"


def _emit_scan_progress(
    *,
    emit: EmitFn | None,
    completed: int,
    total: int,
    run_name: str,
    point_elapsed_s: float,
    total_elapsed_s: float,
    solved_elapsed_s: list[float],
    skipped_existing: bool,
) -> None:
    if emit is None:
        return
    remaining = max(0, int(total) - int(completed))
    if skipped_existing:
        emit(
            0,
            f"scan-er: progress {completed}/{total} {run_name} reused existing output "
            f"total_elapsed={_format_duration(total_elapsed_s)} remaining_points={remaining}",
        )
        return
    avg_s = float(sum(solved_elapsed_s) / max(1, len(solved_elapsed_s)))
    eta_s = float(avg_s * remaining)
    emit(
        0,
        f"scan-er: progress {completed}/{total} {run_name} point_elapsed={_format_duration(point_elapsed_s)} "
        f"avg_point={_format_duration(avg_s)} elapsed={_format_duration(total_elapsed_s)} "
        f"est_remaining={_format_duration(eta_s)}",
    )


def _er_scan_var_name(*, nml: Namelist) -> str:
    geom = nml.group("geometryParameters")
    v = geom.get("INPUTRADIALCOORDINATEFORGRADIENTS", None)
    if v is None:
        # v3 default in many examples is 4 (Er).
        igrad = 4
    else:
        igrad = int(v if not isinstance(v, list) else v[0])

    if igrad == 0:
        return "dPhiHatdpsiHat"
    if igrad == 1:
        return "dPhiHatdpsiN"
    if igrad == 2:
        return "dPhiHatdrHat"
    if igrad == 3:
        return "dPhiHatdrN"
    if igrad == 4:
        return "Er"
    raise ValueError(f"Invalid inputRadialCoordinateForGradients={igrad}")


def _patch_scalar_in_group(*, txt: str, group: str, key: str, value: float) -> str:
    """Patch a scalar assignment inside a Fortran namelist group.

    If the key is not present, it is appended before the group terminator `/`.
    """
    g = str(group)
    k = str(key)

    start = re.search(rf"(?im)^\s*&{re.escape(g)}\s*$", txt)
    if start is None:
        raise ValueError(f"Missing namelist group &{g}")

    # Find the group terminator "/" after the group start.
    end = re.search(r"(?m)^\s*/\s*$", txt[start.end() :])
    if end is None:
        raise ValueError(f"Missing '/' terminator for &{g}")
    end_pos = start.end() + end.start()
    group_txt = txt[start.end() : end_pos]

    # Replace if present (handle quoted/unquoted, spacing, and fortran D exponents).
    pat = re.compile(rf"(?im)^[ \t]*{re.escape(k)}[ \t]*=[ \t]*([^!\n\r]+)[ \t]*$")
    m = pat.search(group_txt)
    new_line = f"  {k} = {value:.16g}"
    if m is not None:
        group_txt2 = group_txt.replace(m.group(0), new_line)
    else:
        # Insert just before the "/" line.
        if not group_txt.endswith("\n"):
            group_txt = group_txt + "\n"
        group_txt2 = group_txt + new_line + "\n"

    return txt[: start.end()] + group_txt2 + txt[end_pos:]


def _can_batch_profile_scan(
    *,
    input_namelist: Path,
    variable: str,
    compute_transport_matrix: bool,
    jobs: int,
    solve_method: str,
) -> bool:
    """Whether the compatibility scan can safely share one operator."""

    # The fast path is deliberately narrow. Unsupported cases retain the
    # scalar runner instead of silently approximating different physics.
    method = str(solve_method).strip().lower()
    if (
        variable != "Er"
        or compute_transport_matrix
        or jobs != 1
        or method
        not in {
            "auto",
            "block_tridiagonal",
            "block_tridiagonal_truncated",
            "gmres",
            "iterative",
        }
    ):
        return False
    try:
        from ..inputs import load_sfincs_input  # noqa: PLC0415

        inp = load_sfincs_input(input_namelist)
    except Exception:  # noqa: BLE001 - scalar path owns legacy validation/errors
        return False
    if (
        inp.general.rhs_mode != 1
        or inp.physics.include_phi1
        or inp.geometry.geometry_scheme not in {1, 2, 3, 4, 5, 11, 12, 13}
    ):
        return False
    if inp.raw is None or not any(
        str(key).upper() == "GEOMETRYSCHEME"
        for key in inp.raw.groups.get("geometryparameters", {})
    ):
        return False
    return True


def _run_er_scan_batched(
    *,
    out_dir: Path,
    template_txt: str,
    values: list[float],
    skip_existing: bool,
    solve_method: str,
    emit: EmitFn | None,
    scan_t0: float,
) -> ScanResult:
    """Solve a compatible profile Er scan with one shared operator/JIT program."""
    from types import SimpleNamespace  # noqa: PLC0415

    from ..batch import batched_er_scan  # noqa: PLC0415
    from ..drift_kinetic import kinetic_operator_build_from_namelist  # noqa: PLC0415
    from ..er import ErProblem, operator_at_er  # noqa: PLC0415
    from ..inputs import load_sfincs_input, parse_sfincs_input_text  # noqa: PLC0415
    from ..moments import classical_fluxes  # noqa: PLC0415
    from ..run import _ntv_kernel_for  # noqa: PLC0415
    from ..writer import (  # noqa: PLC0415
        _geometry_extras,
        operator_containers,
        write_profile_output,
        write_run_solver_trace,
    )

    records: list[tuple[float, Path, Path, Path, bool]] = []
    pending: list[tuple[float, Path, Path, Path]] = []
    for i, value in enumerate(values, start=1):
        run_dir = out_dir / f"Er{value:.4g}"
        run_dir.mkdir(parents=True, exist_ok=True)
        out_path = run_dir / "sfincsOutput.h5"
        input_path = run_dir / "input.namelist"
        reused = bool(skip_existing) and out_path.exists()
        records.append((value, run_dir, out_path, input_path, reused))
        if reused:
            if emit is not None:
                emit(1, f"scan-er: [{i}/{len(values)}] {run_dir.name} already complete; skipping")
            continue
        if emit is not None:
            emit(0, f"scan-er: [{i}/{len(values)}] {run_dir.name} Er={value:.16g} queued")
        point_txt = _patch_scalar_in_group(
            txt=template_txt, group="physicsParameters", key="Er", value=value
        )
        input_path.write_text(point_txt)
        localize_equilibrium_file_in_place(input_namelist=input_path, overwrite=False)
        pending.append((value, run_dir, out_path, input_path))

    if pending:
        first_input = pending[0][3]
        first_txt = first_input.read_text()
        reference_txt = _patch_scalar_in_group(
            txt=first_txt, group="physicsParameters", key="Er", value=1.0
        )
        reference_raw = parse_sfincs_input_text(reference_txt, source_path=first_input)
        reference_build = kinetic_operator_build_from_namelist(reference_raw)
        op_base = reference_build.operator
        dphi_per_er = float(
            np.asarray(op_base.dphi_hat_dpsi_hat_kinetic).reshape(())
        )
        first_typed = load_sfincs_input(first_input)
        tolerance = float(first_typed.resolution.solver_tolerance)
        problem = ErProblem(
            operator=op_base,
            dphi_per_er=dphi_per_er,
            z_s=np.asarray(op_base.z_s, dtype=np.float64).reshape((-1,)),
            er_initial=float(pending[0][0]),
            er_min=float(min(value for value, *_ in pending)),
            er_max=float(max(value for value, *_ in pending)),
            solve_method=str(solve_method),
            tol=tolerance,
        )
        if emit is not None:
            emit(
                0,
                f"scan-er: shared-operator batch n={len(pending)}; one geometry build "
                "and one compiled scan replace per-point cold starts",
            )
        batch_t0 = time.perf_counter()
        batch = batched_er_scan(
            problem,
            np.asarray([value for value, *_ in pending], dtype=np.float64),
            solve_method=str(solve_method),
            tol=tolerance,
            ntv_kernel_tz=_ntv_kernel_for(
                first_typed, op_base, reference_build.geometry
            ),
        )
        states = np.asarray(batch.states, dtype=np.float64)
        residuals = np.asarray(batch.residual_norms, dtype=np.float64).reshape((-1,))
        batch_seconds = float(time.perf_counter() - batch_t0)
        moment_arrays = {
            key: np.asarray(value, dtype=np.float64)
            for key, value in batch.moments.items()
        }
        grids = reference_build.grids
        geom = reference_build.geometry
        radial = reference_build.radial
        gpsipsi, diotadpsi_hat = _geometry_extras(
            inp=first_typed, grids=grids, geom=geom, radial=radial
        )
        _, _, surface, species = operator_containers(op_base)
        classical_particle, classical_heat = classical_fluxes(
            use_phi1=False,
            surface=surface,
            species=species,
            gpsipsi=gpsipsi,
            phi1_hat=np.zeros_like(gpsipsi),
            alpha=op_base.alpha,
            delta=op_base.delta,
            nu_n=first_typed.physics.nu_n,
            dn_hat_dpsi_hat=op_base.dn_hat_dpsi_hat,
            dt_hat_dpsi_hat=op_base.dt_hat_dpsi_hat,
        )
        per_point_seconds = batch_seconds / len(pending)
        for batch_index, (value, _run_dir, out_path, point_input) in enumerate(pending):
            inp = load_sfincs_input(point_input)
            op = operator_at_er(op_base, value, dphi_per_er=dphi_per_er)
            state = states[batch_index]
            residual = float(residuals[batch_index])
            rhs_norm = float(np.linalg.norm(np.asarray(op.rhs(), dtype=np.float64)))
            converged = bool(np.isfinite(residual) and residual <= tolerance * rhs_norm)
            if not converged:
                raise RuntimeError(
                    "scan-er: shared-operator solve did not converge for "
                    f"Er={value:.16g} (residual={residual:.6g}, "
                    f"target={tolerance * rhs_norm:.6g})"
                )
            moments = {
                key: array[batch_index] for key, array in moment_arrays.items()
            }
            moments["classicalParticleFlux_psiHat"] = np.asarray(
                classical_particle, dtype=np.float64
            )
            moments["classicalHeatFlux_psiHat"] = np.asarray(
                classical_heat, dtype=np.float64
            )
            write_profile_output(
                path=out_path,
                inp=inp,
                op=op,
                grids=grids,
                geom=geom,
                radial=radial,
                state_vector=state,
                elapsed_seconds=per_point_seconds,
                solver_method=str(batch.method),
                solver_requested_method=str(solve_method),
                residual_norm=residual,
                geometry_extras=(gpsipsi, diotadpsi_hat),
                moments_table=moments,
                overwrite=True,
                fortran_layout=True,
            )
            solve_result = SimpleNamespace(
                x=np.asarray(state).reshape((-1, 1)),
                method=str(batch.method),
                iterations=None,
                residual_norms=np.asarray([residual]),
                converged=True,
                timings={"solve": per_point_seconds},
            )
            write_run_solver_trace(
                path=out_path.with_name("sfincsOutput.solver_trace.json"),
                inp=inp,
                op=op,
                solve_result=solve_result,
                rhs_norm=rhs_norm,
                solver_tol=tolerance,
                selected_path="rhsmode1_solution_batched_er",
                elapsed_seconds=per_point_seconds,
                input_namelist=point_input,
                output_path=out_path,
                compute_solution=True,
                compute_transport_matrix=False,
            )
        if emit is not None:
            emit(
                0,
                f"scan-er: shared-operator batch complete chunks={batch.n_chunks} "
                f"chunk_size={batch.chunk_size} solve_elapsed={_format_duration(batch_seconds)}",
            )

    solved_elapsed = [
        (float(batch_seconds) / len(pending)) if pending else 0.0
        for _ in pending
    ]
    solved_index = 0
    completed = 0
    for _value, run_dir, _out_path, _input_path, reused in records:
        completed += 1
        point_elapsed = 0.0 if reused else solved_elapsed[solved_index]
        if not reused:
            solved_index += 1
        _emit_scan_progress(
            emit=emit,
            completed=completed,
            total=len(records),
            run_name=run_dir.name,
            point_elapsed_s=point_elapsed,
            total_elapsed_s=float(time.perf_counter() - scan_t0),
            solved_elapsed_s=solved_elapsed[:solved_index],
            skipped_existing=reused,
        )
    return ScanResult(
        scan_dir=out_dir,
        run_dirs=tuple(record[1] for record in records),
        outputs=tuple(record[2] for record in records),
        variable="Er",
        values=tuple(values),
    )


def run_er_scan(
    *,
    input_namelist: Path,
    out_dir: Path,
    values: Sequence[float],
    compute_transport_matrix: bool = False,
    compute_solution: bool = False,
    skip_existing: bool = False,
    solve_method: str = "auto",
    differentiable: bool | None = False,
    jobs: int | None = None,
    index: int | None = None,
    stride: int | None = None,
    emit: EmitFn | None = None,
) -> ScanResult:
    """Run an E_r (or dPhiHatd*) scan using `dkx` and write `sfincsOutput.h5` in each run dir.

    The directory naming convention follows upstream `utils/sfincsScan_2`:
    - the varied variable name is determined by `inputRadialCoordinateForGradients`
    - directories are named like `Er{:.4g}` / `dPhiHatdpsiHat{:.4g}` etc.
    """
    input_namelist = Path(input_namelist).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    template_txt = input_namelist.read_text()

    nml0 = read_sfincs_input(input_namelist)
    var = _er_scan_var_name(nml=nml0)
    # Use a deterministic order that matches upstream `sfincsScan_2`, which generates values
    # via linspace(max, min, N).
    vals = sorted([float(v) for v in values], reverse=True)
    stride_val = max(1, int(stride) if stride is not None else 1)
    if index is not None:
        idx = int(index)
        if idx < 0 or idx >= stride_val:
            raise ValueError(f"scan-er: index={idx} out of range for stride={stride_val}")
        vals = [v for i, v in enumerate(vals) if i % stride_val == idx]
    jobs_val = max(1, int(jobs) if jobs is not None else 1)
    if emit is not None:
        emit(0, f"scan-er: input={input_namelist}")
        emit(0, f"scan-er: out_dir={out_dir}")
        emit(
            0,
            f"scan-er: variable={var} n={len(vals)} compute_solution={bool(compute_solution)} compute_transport_matrix={bool(compute_transport_matrix)} skip_existing={bool(skip_existing)}",
        )
        if len(vals) > 0:
            emit(
                0,
                "scan-er: ETA becomes available after the first completed point. "
                "The first point may include one-time JIT compilation, so later points can be faster.",
            )
        if index is not None:
            emit(1, f"scan-er: subset index={index} stride={stride_val}")
        if jobs_val > 1:
            emit(1, f"scan-er: jobs={jobs_val} (parallel)")

    # Write a scan-style `input.namelist` in the scan directory so vendored upstream
    # `utils/sfincsScanPlot_*` scripts can infer the directory list.
    scan_txt = template_txt
    if not scan_txt.endswith("\n"):
        scan_txt += "\n"
    scan_txt += f"!ss NErs = {len(vals)}\n"
    scan_txt += f"!ss {var}Min = {min(vals):.16g}\n"
    scan_txt += f"!ss {var}Max = {max(vals):.16g}\n"
    (out_dir / "input.namelist").write_text(scan_txt)

    run_dirs: list[Path] = []
    outputs: list[Path] = []
    scan_t0 = time.perf_counter()
    solved_elapsed_s: list[float] = []
    use_batched_profile = _can_batch_profile_scan(
        input_namelist=input_namelist,
        variable=var,
        compute_transport_matrix=bool(compute_transport_matrix),
        jobs=jobs_val,
        solve_method=str(solve_method),
    )
    if use_batched_profile:
        return _run_er_scan_batched(
            out_dir=out_dir,
            template_txt=template_txt,
            values=vals,
            skip_existing=bool(skip_existing),
            solve_method=str(solve_method),
            emit=emit,
            scan_t0=scan_t0,
        )
    del compute_solution, differentiable  # Canonical runs always solve; solve_method picks the path.

    def _run_one(v: float, i: int) -> tuple[Path, Path, float, bool]:
        point_t0 = time.perf_counter()
        run_dir = out_dir / f"{var}{v:.4g}"
        run_dir.mkdir(parents=True, exist_ok=True)
        out_path = run_dir / "sfincsOutput.h5"
        if bool(skip_existing) and out_path.exists():
            if emit is not None:
                emit(1, f"scan-er: [{i}/{len(vals)}] {run_dir.name} already complete; skipping")
            return run_dir, out_path, 0.0, True
        if emit is not None:
            emit(0, f"scan-er: [{i}/{len(vals)}] {run_dir.name} {var}={v:.16g}")

        txt2 = _patch_scalar_in_group(txt=template_txt, group="physicsParameters", key=var, value=float(v))
        w_input = run_dir / "input.namelist"
        w_input.write_text(txt2)
        localize_equilibrium_file_in_place(input_namelist=w_input, overwrite=False)

        nml_point = read_sfincs_input(w_input)
        rhs_mode = int(nml_point.group("general").get("RHSMODE", 1))
        if bool(compute_transport_matrix) and rhs_mode == 1:
            raise ValueError(
                "scan-er: --compute-transport-matrix requires an RHSMode=2/3 deck; "
                f"{w_input} has RHSMode=1."
            )
        emit_line = None if emit is None else (lambda line: emit(2, line))
        run_from_namelist(
            w_input,
            out_path=out_path,
            overwrite=True,
            solve_method=str(solve_method),
            solver_trace_path=run_dir / "sfincsOutput.solver_trace.json",
            emit=emit_line,
        )
        return run_dir, out_path, float(time.perf_counter() - point_t0), False

    def _run_one_parallel(payload: tuple[float, int]) -> tuple[Path, Path, float, bool]:
        v, i = payload
        return _run_one(v, i)

    if jobs_val > 1 and len(vals) > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=jobs_val) as pool:
            payloads = [(v, i) for i, v in enumerate(vals, start=1)]
            futures = [pool.submit(_run_one_parallel, payload) for payload in payloads]
            results: list[tuple[Path, Path, float, bool]] = []
            completed = 0
            for fut in concurrent.futures.as_completed(futures):
                run_dir, out_path, point_elapsed_s, skipped_existing = fut.result()
                results.append((run_dir, out_path, point_elapsed_s, skipped_existing))
                completed += 1
                if not skipped_existing:
                    solved_elapsed_s.append(float(point_elapsed_s))
                _emit_scan_progress(
                    emit=emit,
                    completed=completed,
                    total=len(vals),
                    run_name=run_dir.name,
                    point_elapsed_s=float(point_elapsed_s),
                    total_elapsed_s=float(time.perf_counter() - scan_t0),
                    solved_elapsed_s=solved_elapsed_s,
                    skipped_existing=bool(skipped_existing),
                )
            results.sort(key=lambda item: item[0].name)
            run_dirs = [r for r, _, _, _ in results]
            outputs = [o for _, o, _, _ in results]
    else:
        completed = 0
        for i, v in enumerate(vals, start=1):
            run_dir, out_path, point_elapsed_s, skipped_existing = _run_one(v, i)
            run_dirs.append(run_dir)
            outputs.append(out_path)
            completed += 1
            if not skipped_existing:
                solved_elapsed_s.append(float(point_elapsed_s))
            _emit_scan_progress(
                emit=emit,
                completed=completed,
                total=len(vals),
                run_name=run_dir.name,
                point_elapsed_s=float(point_elapsed_s),
                total_elapsed_s=float(time.perf_counter() - scan_t0),
                solved_elapsed_s=solved_elapsed_s,
                skipped_existing=bool(skipped_existing),
            )

    return ScanResult(
        scan_dir=out_dir,
        run_dirs=tuple(run_dirs),
        outputs=tuple(outputs),
        variable=var,
        values=tuple(vals),
    )


def linspace_including_endpoints(min_value: float, max_value: float, n: int) -> np.ndarray:
    if int(n) < 2:
        raise ValueError("n must be >= 2")
    return np.linspace(float(min_value), float(max_value), int(n), dtype=np.float64)


def find_upstream_utils_dir(*, override: Path | None = None) -> Path:
    """Locate upstream SFINCS v3 ``utils/`` scripts used for scan postprocessing.

    Search order is explicit override, ``DKX_UPSTREAM_UTILS_DIR``, then
    the vendored example-suite path in a source checkout.
    """

    if override is not None:
        path = Path(override)
        if not path.exists():
            raise FileNotFoundError(f"utils dir does not exist: {path}")
        return path

    env = os.environ.get("DKX_UPSTREAM_UTILS_DIR", "").strip()
    if env:
        path = Path(env)
        if not path.exists():
            raise FileNotFoundError(f"DKX_UPSTREAM_UTILS_DIR does not exist: {path}")
        return path

    repo_root = repository_root()
    if repo_root is not None:
        candidate = repo_root / "examples" / "sfincs_examples" / "utils"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate upstream v3 utils/ scripts. Set DKX_UPSTREAM_UTILS_DIR or run from a repo checkout."
    )


def run_upstream_util(
    *,
    util: str,
    case_dir: Path,
    args: Sequence[str] = (),
    utils_dir: Path | None = None,
    noninteractive: bool = True,
    emit: Callable[[int, str], None] | None = None,
) -> None:
    """Run an upstream v3 ``utils/<util>`` script without interactive prompts.

    The wrapper executes the script in a subprocess with ``MPLBACKEND=Agg`` and
    optionally overrides ``input()`` so plotting utilities can run in CI,
    tutorials, and batch environments.
    """

    resolved_utils_dir = find_upstream_utils_dir(override=utils_dir)
    script_path = (resolved_utils_dir / util).resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"Upstream util not found: {script_path}")

    resolved_case_dir = Path(case_dir).resolve()
    if not resolved_case_dir.exists():
        raise FileNotFoundError(f"case_dir does not exist: {resolved_case_dir}")

    harness = textwrap.dedent(
        """
        import builtins
        import runpy
        import sys

        noninteractive = bool(int(sys.argv[1]))
        script = sys.argv[2]
        argv = sys.argv[3:]
        if noninteractive:
            builtins.input = lambda *a, **k: ""
        sys.argv = [script] + argv
        runpy.run_path(script, run_name="__main__")
        """
    ).strip()

    cmd = [
        sys.executable,
        "-c",
        harness,
        "1" if noninteractive else "0",
        str(script_path),
        *list(args),
    ]
    env = dict(os.environ)
    env.setdefault("MPLBACKEND", "Agg")

    if emit is not None:
        emit(0, f"postprocess-upstream: running {script_path.name} in {resolved_case_dir}")
    subprocess.run(cmd, cwd=str(resolved_case_dir), env=env, check=True)
