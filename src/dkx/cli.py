from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from . import runtime as _runtime
from .cli_front import (  # noqa: F401
    _CONVERGE_AXES,
    _KNOWN_COMMANDS,
    _USER_COMMANDS,
    _density_override,
    _maybe_handle_plot,
    _maybe_reexec_for_early_runtime,
    _merge_global_cli_args,
    _normalize_default_argv,
)
from .cli_parsers import (  # noqa: F401
    _add_common_cli_args,
    _add_compat_parsers,
    _add_equilibrium_override_args,
    _add_parallel_cli_args,
    _HiddenAliases,
)
from .cli_user import (  # noqa: F401
    _cmd_compare,
    _cmd_converge,
    _cmd_convert,
    _cmd_inspect_result,
    _cmd_plot,
    _cmd_roots,
    _cmd_run_case,
    _cmd_scan,
    _compare_result_arrays,
    _emit,
    _looks_like_sfincs_h5,
    _now,
)
from .input_compat import with_equilibrium_override
from .namelist import read_sfincs_input


def _emit_namelist_summary(*, nml, args: argparse.Namespace) -> None:
    geom = nml.group("geometryParameters")
    phys = nml.group("physicsParameters")
    res = nml.group("resolutionParameters")
    general = nml.group("general")

    def _g(group: dict, key: str, default=None):
        return group.get(key.upper(), default)

    _emit("----------------------------------------------------------------", level=0, args=args)
    _emit(" input.namelist summary", level=0, args=args)
    _emit(f" geometryScheme={_g(geom, 'geometryScheme', '?')}", level=0, args=args)
    _emit(f" RHSMode={_g(general, 'RHSMode', '?')}", level=0, args=args)
    _emit(f" collisionOperator={_g(phys, 'collisionOperator', '?')}", level=0, args=args)
    _emit(f" includePhi1={bool(_g(phys, 'includePhi1', False))}", level=0, args=args)
    _emit(f" includePhi1InKineticEquation={bool(_g(phys, 'includePhi1InKineticEquation', False))}", level=2, args=args)
    _emit(f" includePhi1InCollisionOperator={bool(_g(phys, 'includePhi1InCollisionOperator', False))}", level=2, args=args)
    _emit(f" useDKESExBDrift={bool(_g(phys, 'useDKESExBDrift', False))}", level=2, args=args)
    _emit(
        " resolution:"
        f" Ntheta={_g(res, 'Ntheta', '?')}"
        f" Nzeta={_g(res, 'Nzeta', '?')}"
        f" Nxi={_g(res, 'Nxi', '?')}"
        f" NL={_g(res, 'NL', '?')}"
        f" Nx={_g(res, 'Nx', '?')}",
        level=0,
        args=args,
    )
    _emit(f" solverTolerance={_g(res, 'solverTolerance', '?')}", level=2, args=args)

def _emit_runtime_info(*, args: argparse.Namespace) -> None:
    """Emit basic runtime info helpful for benchmarking and bug reports."""
    try:
        import jax  # noqa: PLC0415
        import jax.numpy as _jnp  # noqa: PLC0415

        _emit(f" jax={jax.__version__} backend={jax.default_backend()} devices={jax.devices()}", level=2, args=args)
        _emit(f" jax_enable_x64={bool(_jnp.array(0.0).dtype == _jnp.float64)}", level=3, args=args)
    except Exception:  # noqa: BLE001
        return


def _cmd_validate_case(args: argparse.Namespace) -> int:
    """Validate a case without touching JAX kernels or external files."""
    from .config import Case, CaseValidationError  # noqa: PLC0415

    try:
        case = Case.from_file(args.case)
    except (CaseValidationError, OSError) as exc:
        print(f"dkx validate failed: {exc}", file=sys.stderr)
        return 2
    # The schema is deliberately wider than the executor: it admits
    # magnetic_drifts = "full", workflow = "monoenergetic" and
    # phi1 = "kinetic"/"full", none of which execution implements yet. Running
    # the executor's own preflight here is what makes `dkx validate` mean "this
    # will run" rather than "this parses" -- otherwise a user learns which of
    # the advertised enum values are real only after `dkx run` sets up and
    # fails.
    from dataclasses import replace as _replace  # noqa: PLC0415

    from .execution import _validate_native_slice  # noqa: PLC0415

    # The preflight is the single-run one and refuses a [scan] table, which
    # `dkx scan` exists to expand. Check the physics of a scan case with the
    # table removed: refusing it here would make `dkx validate` reject every
    # case `dkx scan` is designed to run.
    try:
        _validate_native_slice(case if case.scan is None else _replace(case, scan=None))
    except CaseValidationError as exc:
        print(f"dkx validate failed: {exc}", file=sys.stderr)
        return 2

    print(f"valid DKX case: {case.name}")
    print(f"case_id: {case.case_id}")
    print(
        f"workflow: {case.run.workflow}; "
        f"surfaces: {len(case.geometry.surfaces)}; species: {len(case.species)}"
    )
    if case.scan is not None:
        print(
            f"scan: {case.scan.case_count} cases "
            f"(limit {case.scan.max_cases}, resume={str(case.scan.resume).lower()})"
        )
    if case.run.workflow == "ambipolar_profile":
        from .workflows.ambipolar_native import (  # noqa: PLC0415
            preflight_ambipolar_case,
        )

        try:
            preflight = preflight_ambipolar_case(case)
        except ValueError as exc:
            print(f"dkx validate failed: {exc}", file=sys.stderr)
            return 2
        print(
            "ambipolar preflight: "
            f"hierarchy_points={preflight.hierarchy_points}; "
            f"max_evaluations_per_surface={preflight.evaluations_per_surface}; "
            f"max_profile_evaluations={preflight.profile_evaluations}"
        )
        print(
            "retained evidence upper bound: "
            f"{preflight.retained_profile_bytes} B profile "
            f"({preflight.retained_bytes_per_surface} B/surface); "
            "runtime not estimated"
        )
        if preflight.search_strategy == "seeded_brackets":
            print(
                "seeded bracket scope: "
                f"endpoint_counts={list(preflight.search_points_by_surface)}; "
                "explicit intervals only; unsampled crossings not excluded"
            )
    return 0


def _cmd_template(args: argparse.Namespace) -> int:
    """Print the commented case template or the case JSON Schema."""
    from .config import COMMENTED_TOML_EXAMPLE, case_json_schema  # noqa: PLC0415

    if args.format == "toml":
        print(COMMENTED_TOML_EXAMPLE, end="")
    else:
        print(json.dumps(case_json_schema(), indent=2, sort_keys=True))
    return 0

def _doctor_checks() -> list[tuple[str, str, str]]:
    """Collect one ``(status, name, detail)`` row per environment check.

    Every row reports what this process *observed*, not what it was asked for.
    The distinction matters: ``JAX_ENABLE_X64`` being set in the environment is
    not evidence that float64 is active, because a backend already initialized
    by an earlier import ignores it. So the check below allocates an array and
    reads its dtype. The same rule applies to the accelerator row, which lists
    the devices JAX actually enumerates rather than the platform requested.

    Status is one of ``ok``, ``warn`` or ``fail``. Only ``fail`` means the
    install cannot run correctly; ``warn`` marks something absent that limits
    what is available without breaking the core solver.
    """
    from importlib.metadata import PackageNotFoundError, version  # noqa: PLC0415

    rows: list[tuple[str, str, str]] = []

    py = ".".join(str(n) for n in sys.version_info[:3])
    rows.append(
        ("ok", "python", py) if sys.version_info >= (3, 11)
        else ("fail", "python", f"{py} is below the 3.11 floor")
    )

    try:
        rows.append(("ok", "dkx", version("dkx")))
    except PackageNotFoundError:
        rows.append(("warn", "dkx", "not installed as a distribution (running from a checkout)"))

    # solvax carries the solver routes, and a version below the declared floor
    # fails deep inside a solve rather than at import, so it is checked here.
    floor = (0, 19, 0)
    try:
        raw = version("solvax")
        parsed = tuple(int(part) for part in raw.split(".")[:3])
        rows.append(
            ("ok", "solvax", raw) if parsed >= floor
            else ("fail", "solvax", f"{raw} is below the {'.'.join(map(str, floor))} floor")
        )
    except PackageNotFoundError:
        rows.append(("fail", "solvax", "missing; every canonical solve needs it"))
    except ValueError:
        rows.append(("warn", "solvax", f"{raw} could not be compared against the floor"))

    for name, required in (
        ("jax", True), ("jaxlib", True), ("numpy", True), ("scipy", True),
        ("h5py", False), ("netCDF4", False), ("matplotlib", False), ("rich", True),
    ):
        try:
            rows.append(("ok", name, version(name)))
        except PackageNotFoundError:
            rows.append(("fail" if required else "warn", name, "missing"))

    # Observed float64, not the environment variable that requests it.
    try:
        import warnings  # noqa: PLC0415

        import jax.numpy as jnp  # noqa: PLC0415

        # JAX warns when it truncates a requested float64 to float32. Probing
        # for exactly that is the point here, so the warning is the finding,
        # not a problem to surface twice.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            dtype = str(jnp.zeros(1, dtype=jnp.float64).dtype)
        rows.append(
            ("ok", "float64", "active") if dtype == "float64"
            else ("fail", "float64", f"arrays materialize as {dtype}; results will be wrong")
        )
    except Exception as exc:  # noqa: BLE001 - any backend failure is the finding
        rows.append(("fail", "float64", f"could not allocate a JAX array: {exc}"))

    # Observed devices, not the requested platform.
    try:
        import jax  # noqa: PLC0415

        devices = jax.devices()
        kinds = sorted({d.platform for d in devices})
        rows.append(("ok", "devices", f"{len(devices)} x {','.join(kinds)}"))
    except Exception as exc:  # noqa: BLE001
        rows.append(("fail", "devices", f"JAX enumerated no devices: {exc}"))

    return rows


def _cmd_doctor(args: argparse.Namespace) -> int:
    """Report whether this install can run, and exit non-zero when it cannot.

    Exists because the failure modes users actually hit are environmental, not
    physical: a conda numpy 1.x that dies inside ``ml_dtypes``, a solvax below
    the floor that fails mid-solve, or float64 silently off so every number is
    plausible and wrong. Each of those produces a traceback that names neither
    dkx nor the real cause.
    """
    from rich.console import Console  # noqa: PLC0415
    from rich.table import Table  # noqa: PLC0415

    rows = _doctor_checks()

    if args.format == "json":
        print(json.dumps(
            {name: {"status": status, "detail": detail} for status, name, detail in rows},
            indent=2, sort_keys=True,
        ))
    else:
        console = Console()
        table = Table(show_lines=False)
        table.add_column("")
        table.add_column("check", style="bold")
        table.add_column("observed")
        marks = {"ok": "[green]ok[/green]", "warn": "[yellow]warn[/yellow]", "fail": "[red]fail[/red]"}
        for status, name, detail in rows:
            table.add_row(marks[status], name, detail)
        console.print(table)

    failures = [name for status, name, _ in rows if status == "fail"]
    if failures:
        print(f"dkx doctor: {len(failures)} blocking problem(s): {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


def _emit_parallel_runtime_info(*, args: argparse.Namespace) -> None:
    def _env(name: str, default: str = "") -> str:
        return os.environ.get(name, default).strip()

    cores = _env("DKX_CORES")
    threads = _env("NPROC")
    cpu_devices = _env("DKX_CPU_DEVICES")
    transport_parallel = _env("DKX_TRANSPORT_PARALLEL", "off") or "off"
    transport_workers = _env("DKX_TRANSPORT_PARALLEL_WORKERS", "1") or "1"
    distributed = _env("DKX_DISTRIBUTED")

    if not any(
        (
            cores,
            cpu_devices,
            transport_parallel not in {"", "off"},
            distributed,
        )
    ):
        return

    _emit(
        " parallel:"
        f" cores={cores or '-'}"
        f" threads={threads or '-'}"
        f" cpu_devices={cpu_devices or '-'}",
        level=1,
        args=args,
    )
    _emit(
        f" transport_parallel: mode={transport_parallel} workers={transport_workers}",
        level=1,
        args=args,
    )
    if distributed in {"1", "true", "yes", "on"}:
        _emit(
            " multi_host:"
            " enabled=1"
            f" process_id={_env('DKX_PROCESS_ID', '-') or '-'}"
            f" process_count={_env('DKX_PROCESS_COUNT', '-') or '-'}"
            f" coordinator={_env('DKX_COORDINATOR_ADDRESS', '-') or '-'}"
            f" port={_env('DKX_COORDINATOR_PORT', '-') or '-'}",
            level=1,
            args=args,
        )

def _nml_with_cli_equilibrium_override(nml, args: argparse.Namespace):
    return with_equilibrium_override(
        nml=nml,
        equilibrium_file=getattr(args, "equilibrium_file", None),
        wout_path=getattr(args, "wout_path", None),
    )

@contextlib.contextmanager
def _canonical_namelist_path(*, nml, input_path: Path, args: argparse.Namespace):
    """Yield a namelist path for the canonical driver, honoring CLI equilibrium overrides.

    The canonical :func:`dkx.run.run_transport_matrix` reads the input
    file itself, so a ``--equilibrium-file``/``--wout-path`` override is
    materialized as a sibling temporary namelist (same directory, so any other
    relative paths keep resolving) and removed afterwards.
    """
    from .input_compat import canonical_equilibrium_override  # noqa: PLC0415

    override = canonical_equilibrium_override(
        equilibrium_file=getattr(args, "equilibrium_file", None),
        wout_path=getattr(args, "wout_path", None),
    )
    if override is None:
        yield input_path
        return
    if nml.source_text is None:
        raise ValueError(
            "--equilibrium-file/--wout-path require a readable input.namelist source text."
        )
    tmp = tempfile.NamedTemporaryFile(
        "w",
        dir=str(input_path.parent),
        prefix=f".{input_path.stem}.override.",
        suffix=".namelist",
        delete=False,
        encoding="utf-8",
    )
    try:
        tmp.write(nml.source_text)
        tmp.close()
        yield Path(tmp.name)
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

def _cmd_solve_v3(args: argparse.Namespace) -> int:
    t0 = _now()
    nml = _nml_with_cli_equilibrium_override(read_sfincs_input(Path(args.input)), args)
    rhs_mode = int(nml.group("general").get("RHSMODE", 1))
    _emit("################################################################", level=0, args=args)
    _emit(" dkx solve-v3", level=0, args=args)
    _emit(f" input={Path(args.input).resolve()}", level=0, args=args)
    _emit_namelist_summary(nml=nml, args=args)
    _emit_runtime_info(args=args)
    _emit_parallel_runtime_info(args=args)
    _emit(f" tol={args.tol} atol={args.atol} restart={args.restart} maxiter={args.maxiter} solve_method={args.solve_method}", level=1, args=args)
    if args.which_rhs is not None:
        _emit(f" whichRHS={args.which_rhs}", level=0, args=args)

    out_state = Path(args.out_state)

    # The canonical stack (dkx.run) owns every supported deck.  Invalid
    # namelist values (RHSMode outside 1-3, out-of-range option values) surface
    # as load-time validation errors.
    quiet = bool(getattr(args, "quiet", False))
    emit_line = None if quiet else (lambda line: _emit(line, level=0, args=args))
    try:
        with _canonical_namelist_path(nml=nml, input_path=Path(args.input), args=args) as namelist_path:
            if rhs_mode == 1:
                from .run import run_profile  # noqa: PLC0415

                run = run_profile(
                    namelist_path,
                    solve_method=str(args.solve_method),
                    tol=float(args.tol),
                    emit=emit_line,
                )
                state = np.asarray(run.state_vector)
                residual = float(np.atleast_1d(np.asarray(run.solve_result.residual_norms, dtype=np.float64))[0])
            else:
                from .run import run_transport_matrix  # noqa: PLC0415

                run = run_transport_matrix(
                    namelist_path,
                    solve_method=str(args.solve_method),
                    tol=float(args.tol),
                    emit=emit_line,
                )
                state_vectors = np.asarray(run.state_vectors)
                residual_norms = np.atleast_1d(np.asarray(run.solve_result.residual_norms, dtype=np.float64))
                col = (int(args.which_rhs) - 1) if args.which_rhs is not None else 0
                if not (0 <= col < state_vectors.shape[0]):
                    raise ValueError(
                        f"whichRHS={args.which_rhs} is out of range for the "
                        f"{state_vectors.shape[0]} transport-matrix RHS columns"
                    )
                state = state_vectors[col]
                residual = float(residual_norms[col])
    except (NotImplementedError, ValueError, RuntimeError) as exc:
        if os.environ.get("DKX_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}:
            raise
        print(f"dkx solve-v3 failed: {exc}", file=sys.stderr, flush=True)
        return 2
    out_state.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_state, state)
    _emit(f" wrote stateVector -> {out_state.resolve()}", level=0, args=args)
    _emit(f" residual_norm={residual:.6e}", level=0, args=args)
    _emit(f" elapsed_s={_now()-t0:.3f}", level=1, args=args)
    return 0

def _cmd_run_fortran(args: argparse.Namespace) -> int:
    t0 = _now()
    from .validation.fortran import run_sfincs_fortran  # noqa: PLC0415

    _emit("################################################################", level=0, args=args)
    _emit(" dkx run-fortran", level=0, args=args)
    _emit(f" input={Path(args.input).resolve()}", level=0, args=args)
    output_path = run_sfincs_fortran(
        input_namelist=Path(args.input),
        exe=Path(args.exe) if args.exe else None,
        workdir=Path(args.workdir) if args.workdir else None,
    )
    _emit(f" wrote sfincsOutput.h5 -> {output_path}", level=0, args=args)
    _emit(f" elapsed_s={_now()-t0:.3f}", level=1, args=args)
    return 0

def _cmd_write_output(args: argparse.Namespace) -> int:
    t0 = _now()
    from .writer import output_format_from_suffix  # noqa: PLC0415

    nml = _nml_with_cli_equilibrium_override(read_sfincs_input(Path(args.input)), args)
    rhs_mode = int(nml.group("general").get("RHSMODE", 1))
    try:
        output_format = output_format_from_suffix(Path(args.out))
    except ValueError as exc:
        print(f"dkx write-output failed: {exc}", file=sys.stderr, flush=True)
        return 2
    _emit("################################################################", level=0, args=args)
    _emit(" dkx write-output", level=0, args=args)
    _emit(f" input={Path(args.input).resolve()}", level=0, args=args)
    _emit(f" output={Path(args.out).resolve()} format={output_format}", level=0, args=args)
    _emit_namelist_summary(nml=nml, args=args)
    _emit_runtime_info(args=args)
    _emit_parallel_runtime_info(args=args)

    # Default to upstream v3 behavior: full solve/write appropriate to RHSMode
    # (--geometry-only skips the solve), all on the canonical stack
    # (dkx.run).  Invalid namelist values (RHSMode outside 1-3,
    # out-of-range option values) surface as load-time validation errors.
    geometry_only = bool(getattr(args, "geometry_only", False))
    res_group = nml.group("resolutionParameters")
    try:
        solver_tol = float(res_group.get("SOLVERTOLERANCE", 1e-10))
    except (TypeError, ValueError):
        solver_tol = 1e-10
    quiet = bool(getattr(args, "quiet", False))
    emit_line = None if quiet else (lambda line: _emit(line, level=0, args=args))
    solver_trace_path = Path(args.solver_trace) if getattr(args, "solver_trace", None) else None
    overwrite = bool(args.overwrite)
    fortran_layout = bool(args.fortran_layout)
    try:
        with _canonical_namelist_path(nml=nml, input_path=Path(args.input), args=args) as namelist_path:
            if geometry_only:
                from .run import run_geometry  # noqa: PLC0415

                run = run_geometry(
                    namelist_path,
                    out_path=Path(args.out),
                    overwrite=overwrite,
                    fortran_layout=fortran_layout,
                    solver_trace_path=solver_trace_path,
                    emit=emit_line,
                )
            elif rhs_mode == 1:
                from .run import run_profile  # noqa: PLC0415

                run = run_profile(
                    namelist_path,
                    solve_method=str(getattr(args, "solve_method", "auto")),
                    tol=solver_tol,
                    out_path=Path(args.out),
                    overwrite=overwrite,
                    fortran_layout=fortran_layout,
                    solver_trace_path=solver_trace_path,
                    emit=emit_line,
                )
            else:
                from .run import run_transport_matrix  # noqa: PLC0415

                run = run_transport_matrix(
                    namelist_path,
                    solve_method=str(getattr(args, "solve_method", "auto")),
                    tol=solver_tol,
                    out_path=Path(args.out),
                    overwrite=overwrite,
                    fortran_layout=fortran_layout,
                    solver_trace_path=solver_trace_path,
                    emit=emit_line,
                )
    except (NotImplementedError, ValueError, FileExistsError, RuntimeError) as exc:
        if os.environ.get("DKX_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}:
            raise
        print(f"dkx write-output failed: {exc}", file=sys.stderr, flush=True)
        return 2
    _emit(f" wrote output -> {run.output_path}", level=0, args=args)
    if solver_trace_path is not None:
        _emit(f" wrote solver trace -> {solver_trace_path.resolve()}", level=0, args=args)
    _emit(f" elapsed_s={_now()-t0:.3f}", level=1, args=args)
    return 0

def _cmd_transport_matrix_v3(args: argparse.Namespace) -> int:
    """RHSMode=2/3 transport-matrix runs on the canonical stack (:mod:`dkx.run`)."""
    t0 = _now()
    from .run import run_transport_matrix  # noqa: PLC0415

    input_path = Path(args.input)
    nml = _nml_with_cli_equilibrium_override(read_sfincs_input(input_path), args)
    _emit("################################################################", level=0, args=args)
    _emit(" dkx transport-matrix-v3", level=0, args=args)
    _emit(f" input={input_path.resolve()}", level=0, args=args)
    _emit_namelist_summary(nml=nml, args=args)
    _emit_runtime_info(args=args)
    _emit_parallel_runtime_info(args=args)
    _emit(f" tol={args.tol} solve_method={args.solve_method}", level=1, args=args)
    quiet = bool(getattr(args, "quiet", False))
    with _canonical_namelist_path(nml=nml, input_path=input_path, args=args) as namelist_path:
        run = run_transport_matrix(
            namelist_path,
            solve_method=str(args.solve_method),
            tol=float(args.tol),
            out_path=Path(args.out) if getattr(args, "out", None) else None,
            emit=None if quiet else (lambda line: _emit(line, level=0, args=args)),
        )

    out_tm = Path(args.out_matrix)
    out_tm.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_tm, np.asarray(run.transport_matrix))
    _emit(f" wrote transportMatrix -> {out_tm.resolve()}", level=0, args=args)
    if run.output_path is not None:
        _emit(f" wrote output -> {Path(run.output_path).resolve()}", level=0, args=args)

    if args.out_state_prefix is not None:
        pref = Path(args.out_state_prefix)
        pref.parent.mkdir(parents=True, exist_ok=True)
        for idx, x in enumerate(np.asarray(run.state_vectors), start=1):
            p = pref.with_name(f"{pref.name}.whichRHS{idx}.npy")
            np.save(p, np.asarray(x))
            _emit(f" wrote stateVector(whichRHS={idx}) -> {p.resolve()}", level=1, args=args)

    residual_norms = np.atleast_1d(np.asarray(run.solve_result.residual_norms, dtype=np.float64))
    for idx, rn in enumerate(residual_norms, start=1):
        _emit(f" whichRHS={idx} residual_norm={float(rn):.6e}", level=0, args=args)
    _emit(f" elapsed_s={_now()-t0:.3f}", level=1, args=args)
    return 0

def _cmd_monoenergetic_database(args: argparse.Namespace) -> int:
    """Scan (nuPrime, EStar) and write the monoenergetic-coefficient database."""
    t0 = _now()
    from .monoenergetic import monoenergetic_database, save_database  # noqa: PLC0415

    input_path = Path(args.input)
    nu_values = [float(v) for v in args.nu_prime]
    er_values = [float(v) for v in args.e_star]
    _emit("################################################################", level=0, args=args)
    _emit(" dkx monoenergetic-database", level=0, args=args)
    _emit(f" input={input_path.resolve()}", level=0, args=args)
    _emit(f" nuPrime grid ({len(nu_values)}): {nu_values}", level=0, args=args)
    _emit(f" EStar grid ({len(er_values)}): {er_values}", level=0, args=args)
    quiet = bool(getattr(args, "quiet", False))
    db = monoenergetic_database(
        input_path,
        nu_values,
        er_values,
        solve_method=str(args.solve_method),
        tol=float(args.tol),
        emit=None if quiet else (lambda line: _emit(line, level=0, args=args)),
    )
    out = save_database(Path(args.out), db)
    _emit(f" wrote database -> {out.resolve()}", level=0, args=args)
    header = f" {'nuPrime':>12} {'EStar':>12} {'nu_star':>12} {'D11*':>13} {'D31*':>13} {'D13*':>13} {'D33*':>13}"
    _emit(header, level=0, args=args)
    nu_star = np.asarray(db.nu_star)
    for i, nu in enumerate(np.asarray(db.nu_prime)):
        for j, er in enumerate(np.asarray(db.e_star)):
            _emit(
                f" {nu:12.5e} {er:12.5e} {nu_star[i]:12.5e}"
                f" {float(np.asarray(db.d11_star)[i, j]):13.6e}"
                f" {float(np.asarray(db.d31_star)[i, j]):13.6e}"
                f" {float(np.asarray(db.d13_star)[i, j]):13.6e}"
                f" {float(np.asarray(db.d33_star)[i, j]):13.6e}",
                level=0,
                args=args,
            )
    _emit(f" elapsed_s={_now()-t0:.3f}", level=1, args=args)
    return 0

def _cmd_dump_h5(args: argparse.Namespace) -> int:
    from .io import read_sfincs_h5  # noqa: PLC0415

    data = read_sfincs_h5(Path(args.sfincs_output))
    if args.keys_only:
        for k in sorted(data.keys()):
            print(k)
        return 0
    out = {k: v.tolist() if hasattr(v, "tolist") else v for k, v in data.items()}
    Path(args.out_json).write_text(json.dumps(out, indent=2, sort_keys=True))
    return 0

def _default_plot_output_path(input_h5: Path) -> Path:
    input_h5 = Path(input_h5)
    stem = input_h5.stem
    if stem.endswith(".sfincsOutput"):
        stem = stem[: -len(".sfincsOutput")]
    return input_h5.with_name(f"{stem}_summary.pdf")

def _cmd_plot_output(args: argparse.Namespace) -> int:
    t0 = _now()
    from .plotting import plot_sfincs_output_summary  # noqa: PLC0415

    input_h5 = Path(args.input_h5)
    out_path = Path(args.out) if args.out else _default_plot_output_path(input_h5)
    _emit("################################################################", level=0, args=args)
    _emit(" dkx plot-output", level=0, args=args)
    _emit(f" input={input_h5.resolve()}", level=0, args=args)
    _emit(f" out={out_path.resolve()}", level=0, args=args)
    plot_path = plot_sfincs_output_summary(input_h5=input_h5, output_png=out_path)
    _emit(f" wrote plot -> {plot_path}", level=0, args=args)
    _emit(f" elapsed_s={_now()-t0:.3f}", level=1, args=args)
    return 0

def _cmd_compare_h5(args: argparse.Namespace) -> int:
    from .compare import compare_sfincs_outputs  # noqa: PLC0415

    tolerances = None
    if args.tolerances_json:
        with open(args.tolerances_json, "r", encoding="utf-8") as f:
            tolerances = json.load(f)
    results = compare_sfincs_outputs(
        a_path=Path(args.a),
        b_path=Path(args.b),
        rtol=float(args.rtol),
        atol=float(args.atol),
        tolerances=tolerances,
    )
    bad = [r for r in results if not r.ok]
    if args.show_all:
        for r in results:
            status = "OK" if r.ok else "FAIL"
            print(f"{status} {r.key}: max_abs={r.max_abs:.3e} max_rel={r.max_rel:.3e}")
    else:
        for r in bad[:50]:
            print(f"FAIL {r.key}: max_abs={r.max_abs:.3e} max_rel={r.max_rel:.3e}")
        if len(bad) > 50:
            print(f"... {len(bad) - 50} more failing keys omitted")
    return 0 if not bad else 2

def _cmd_scan_er(args: argparse.Namespace) -> int:
    t0 = _now()
    from .workflows.scans import (  # noqa: PLC0415
        linspace_including_endpoints,
        run_er_scan,
    )

    _emit("################################################################", level=0, args=args)
    _emit(" dkx scan-er", level=0, args=args)
    _emit(f" input={Path(args.input).resolve()}", level=0, args=args)
    _emit(f" out-dir={Path(args.out_dir).resolve()}", level=0, args=args)
    _emit_runtime_info(args=args)
    _emit_parallel_runtime_info(args=args)

    if args.values is not None:
        values = [float(x) for x in args.values]
    else:
        values = list(linspace_including_endpoints(float(args.min), float(args.max), int(args.n)))

    result = run_er_scan(
        input_namelist=Path(args.input),
        out_dir=Path(args.out_dir),
        values=values,
        compute_transport_matrix=bool(args.compute_transport_matrix),
        compute_solution=bool(getattr(args, "compute_solution", False)),
        skip_existing=bool(getattr(args, "skip_existing", False)),
        solve_method=str(getattr(args, "solve_method", "auto")),
        differentiable=False,
        jobs=int(args.jobs) if getattr(args, "jobs", None) is not None else None,
        index=int(args.index) if getattr(args, "index", None) is not None else None,
        stride=int(args.stride) if getattr(args, "stride", None) is not None else None,
        emit=lambda level, msg: _emit(msg, level=level, args=args),
    )
    _emit(f" elapsed_s={_now()-t0:.3f}", level=1, args=args)
    if result.failures:
        _emit(
            f"scan-er: {len(result.failures)} of {len(result.values)} points failed "
            f"(see {Path(args.out_dir) / 'scan_failures.json'}): "
            + ", ".join(f"{v:g}" for v, _ in result.failures),
            level=0,
            args=args,
        )
        return 1
    return 0

def _cmd_ambipolar_solve(args: argparse.Namespace) -> int:
    t0 = _now()
    from .ambipolar import solve_ambipolar_from_scan_dir  # noqa: PLC0415

    _emit("################################################################", level=0, args=args)
    _emit(" dkx ambipolar-solve", level=0, args=args)
    _emit(f" scan-dir={Path(args.scan_dir).resolve()}", level=0, args=args)
    _emit_runtime_info(args=args)
    _emit_parallel_runtime_info(args=args)

    res = solve_ambipolar_from_scan_dir(
        scan_dir=Path(args.scan_dir),
        write_pickle=True,
        write_json=True,
        n_fine=int(args.n_fine),
    )

    if res.roots_er.size == 0:
        _emit(" ambipolar-solve: no sign change found (no roots).", level=0, args=args)
    else:
        for i, (rv, re, rt) in enumerate(zip(res.roots_var, res.roots_er, res.root_types, strict=False), start=1):
            _emit(f" root[{i}] {res.var_name}={float(rv):.16g} Er={float(re):.16g} type={rt}", level=0, args=args)

    _emit(f" wrote {Path(args.scan_dir).resolve() / 'ambipolarSolutions.dat'}", level=1, args=args)
    _emit(f" wrote {Path(args.scan_dir).resolve() / 'ambipolarSolutions.json'}", level=2, args=args)
    _emit(f" elapsed_s={_now()-t0:.3f}", level=1, args=args)
    return 0

def _cmd_ambipolar(args: argparse.Namespace) -> int:
    t0 = _now()
    from .er import find_ambipolar_er  # noqa: PLC0415

    _emit("################################################################", level=0, args=args)
    _emit(" dkx ambipolar", level=0, args=args)
    _emit(f" input={Path(args.input).resolve()}", level=0, args=args)
    _emit(f" out-dir={Path(args.out_dir).resolve()}", level=0, args=args)
    _emit_runtime_info(args=args)
    _emit_parallel_runtime_info(args=args)
    _emit(
        " ambipolar:"
        f" method=brent er_min={float(args.er_min):.16g}"
        f" er_max={float(args.er_max):.16g}"
        f" er_initial={float(args.er_initial):.16g}"
        f" max_evaluations={int(args.max_evaluations)}"
        f" current_tolerance={float(args.current_tolerance):.3e}",
        level=0,
        args=args,
    )

    result = find_ambipolar_er(
        Path(args.input),
        er_bracket=(float(args.er_min), float(args.er_max)),
        er_initial=float(args.er_initial),
        max_iter=int(args.max_evaluations),
        current_tol=float(args.current_tolerance),
        solve_method=str(args.solve_method),
        emit=lambda msg: _emit(msg, level=1, args=args),
    )

    summary_path = Path(args.summary_json) if args.summary_json else Path(args.out_dir) / "ambipolar_result.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "converged": bool(result.converged),
        "method": result.method,
        "status": result.status,
        "message": result.message,
        "root_er": result.er,
        "root_radial_current": result.radial_current,
        "root_type": result.root_type,
        "iterations": [
            {
                "index": item.index,
                "er": item.er,
                "radial_current": item.radial_current,
                "stage": item.stage,
            }
            for item in result.iterations
        ],
        "roots": [
            {
                "er": root.er,
                "radial_current": root.radial_current,
                "slope": root.slope,
                "root_type": root.root_type,
            }
            for root in result.roots
        ],
        "n_solves": result.n_solves,
        "failures": list(result.failures),
        "elapsed_s": float(_now() - t0),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    if result.converged and result.er is not None:
        _emit(
            f" ambipolar root: Er={float(result.er):.16g} "
            f"radial_current={float(result.radial_current):.6e} type={result.root_type}",
            level=0,
            args=args,
        )
    else:
        _emit(f" ambipolar status={result.status}: {result.message}", level=0, args=args)
    _emit(f" wrote summary -> {summary_path.resolve()}", level=0, args=args)
    _emit(f" elapsed_s={_now()-t0:.3f}", level=1, args=args)
    return 0 if result.converged else 2

def _apply_cores_setting(cores: int | None) -> None:
    """Record the requested solver thread count in the process environment.

    ``cores > 0`` pins the XLA host threadpool (``NPROC`` — the variable XLA
    actually reads when its CPU backend initializes), and the host BLAS pools
    default to one thread (:func:`dkx.runtime._default_single_thread_blas`
    explains why); ``cores ==
    0`` requests XLA's own full-width sizing (``DKX_CORES=0`` suppresses the
    package default clamp of ``min(8, cpu_count)``).  Thread counts only take
    effect before JAX initializes, so the CLI re-execs itself with
    ``DKX_CORES`` exported (:func:`_maybe_reexec_for_early_runtime`) and the
    package applies the variables at import; the assignments here keep child
    processes (transport workers, spawned tools) consistent.
    """
    if cores is None:
        return
    try:
        cores_val = int(cores)
    except (TypeError, ValueError):
        return
    if cores_val < 0:
        return
    os.environ["DKX_CORES"] = str(cores_val)
    if cores_val > 0:
        os.environ["NPROC"] = str(cores_val)
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(name, "1")

def _apply_runtime_env_defaults() -> None:
    # Avoid large eager GPU preallocation in CLI workflows so solver/benchmark
    # runs coexist better with other accelerator jobs by default.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

def _apply_parallel_runtime_settings(args: argparse.Namespace) -> None:
    transport_workers = getattr(args, "transport_workers", None)
    if transport_workers is not None:
        workers_val = max(1, int(transport_workers))
        os.environ["DKX_TRANSPORT_PARALLEL"] = "process" if workers_val > 1 else "off"
        os.environ["DKX_TRANSPORT_PARALLEL_WORKERS"] = str(workers_val)

    if bool(getattr(args, "distributed", False)):
        from . import initialize_distributed_runtime_from_env  # noqa: PLC0415

        os.environ["DKX_DISTRIBUTED"] = "1"
        process_id = getattr(args, "process_id", None)
        process_count = getattr(args, "process_count", None)
        coordinator_address = getattr(args, "coordinator_address", None)
        coordinator_port = getattr(args, "coordinator_port", None)
        if process_id is not None:
            os.environ["DKX_PROCESS_ID"] = str(int(process_id))
        if process_count is not None:
            os.environ["DKX_PROCESS_COUNT"] = str(int(process_count))
        if coordinator_address:
            os.environ["DKX_COORDINATOR_ADDRESS"] = str(coordinator_address)
        if coordinator_port is not None:
            os.environ["DKX_COORDINATOR_PORT"] = str(int(coordinator_port))
        initialize_distributed_runtime_from_env()

def main(argv: list[str] | None = None) -> int:
    """Run the dkx command-line interface."""
    # The CLI bootstrap, as plan.md section 6.4 requires: the runtime is applied
    # here, once, before anything imports the JAX backend. Doing it first is what
    # lets --cores reach XLA's threadpool, which reads NPROC when the CPU backend
    # initialises and never again.
    _runtime.configure()
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    rc = _maybe_handle_plot(argv)
    if rc is not None:
        return rc
    parser = argparse.ArgumentParser(prog="dkx")
    _add_common_cli_args(parser)
    _add_parallel_cli_args(parser)
    # metavar hides the compatibility aliases from the choice list. Without it
    # argparse prints all 21 names even though their help is suppressed.
    sub = parser.add_subparsers(
        dest="cmd", required=True, metavar="{" + ",".join(_USER_COMMANDS) + "}"
    )

    p_validate = sub.add_parser(
        "validate",
        help="Validate a case file and print its deterministic ID.",
    )
    _add_common_cli_args(p_validate)
    _add_parallel_cli_args(p_validate)
    p_validate.add_argument("case", help="Path to a .toml or .json case file.")
    p_validate.set_defaults(func=_cmd_validate_case)

    p_doctor = sub.add_parser(
        "doctor",
        help="Check that this install can run, and say what is wrong when it cannot.",
    )
    _add_common_cli_args(p_doctor)
    _add_parallel_cli_args(p_doctor)
    p_doctor.add_argument("--format", choices=("table", "json"), default="table")
    p_doctor.set_defaults(func=_cmd_doctor)

    p_converge = sub.add_parser(
        "converge",
        help="Refine each phase-space axis of a case and report observable convergence.",
    )
    _add_common_cli_args(p_converge)
    _add_parallel_cli_args(p_converge)
    p_converge.add_argument("case", help="Path to a native .toml/.json case or a linear .namelist deck.")
    p_converge.add_argument(
        "--axes", nargs="+", default=list(_CONVERGE_AXES),
        choices=list(_CONVERGE_AXES),
        help="Phase-space axes to refine (default: all).",
    )
    p_converge.add_argument("--factor", type=float, default=1.5, help="Refinement factor per axis.")
    p_converge.add_argument("--tolerance", type=float, default=0.02, help="Relative-change tolerance.")
    p_converge.add_argument(
        "--no-joint", action="store_true",
        help="Skip the all-axes-together run. Faster, and unable to detect axis coupling.",
    )
    p_converge.add_argument("--format", choices=("table", "json"), default="table")
    p_converge.set_defaults(func=_cmd_converge)

    p_roots = sub.add_parser(
        "roots",
        help="Print the ambipolar root table and branch events stored in a result.",
    )
    _add_common_cli_args(p_roots)
    _add_parallel_cli_args(p_roots)
    p_roots.add_argument("result", help="Path to a NetCDF Result written by dkx run.")
    p_roots.add_argument("--format", choices=("table", "json"), default="table")
    p_roots.set_defaults(func=_cmd_roots)

    p_compare = sub.add_parser(
        "compare",
        help="Compare two results (dkx NetCDF or SFINCS HDF5) and exit non-zero if they differ.",
    )
    _add_common_cli_args(p_compare)
    _add_parallel_cli_args(p_compare)
    p_compare.add_argument("a")
    p_compare.add_argument("b")
    p_compare.add_argument("--rtol", type=float, default=1e-9)
    p_compare.add_argument("--atol", type=float, default=0.0)
    p_compare.add_argument("--verbose-keys", action="store_true",
                           help="List every array, not only the differing ones.")
    p_compare.add_argument("--format", choices=("table", "json"), default="table")
    p_compare.set_defaults(func=_cmd_compare)

    p_plot_result = sub.add_parser(
        "plot",
        help="Plot a result (dkx NetCDF radial profiles, or a SFINCS HDF5 panel).",
    )
    _add_common_cli_args(p_plot_result)
    _add_parallel_cli_args(p_plot_result)
    p_plot_result.add_argument("result")
    p_plot_result.add_argument("--out", default=None, help="Output image path (default: alongside the input).")
    p_plot_result.add_argument(
        "--kind", choices=("auto", "summary", "search", "scan"), default="auto",
        help="auto: pick from what the result holds. summary: radial profiles. "
             "search: radial current against E_r with every evaluation, bracket "
             "and root (ambipolar results). scan: observables against the scan axis.",
    )
    p_plot_result.set_defaults(func=_cmd_plot)

    p_scan_case = sub.add_parser(
        "scan",
        help="Expand a case's [scan] axes, run every point, and write one Result.",
    )
    _add_common_cli_args(p_scan_case)
    _add_parallel_cli_args(p_scan_case)
    p_scan_case.add_argument("case", help="Path to a .toml or .json case file with a [scan] table.")
    p_scan_case.add_argument("--out", default=None, help="Output NetCDF path (default: scan.output).")
    p_scan_case.add_argument(
        "--no-resume", action="store_true",
        help="Rerun every point even if the output already holds its result.",
    )
    p_scan_case.set_defaults(func=_cmd_scan)

    # ``schema`` is a hidden alias of ``template``, kept for one release.
    for name, extra in (
        ("template", {"help": "Print the commented case template or its JSON Schema."}),
        ("schema", {}),
    ):
        p_template = sub.add_parser(name, **extra)
        _add_common_cli_args(p_template)
        _add_parallel_cli_args(p_template)
        p_template.add_argument("--format", choices=("toml", "json"), default="toml")
        p_template.set_defaults(func=_cmd_template)

    p_run = sub.add_parser(
        "run",
        help="Execute a case file and write its NetCDF Result.",
    )
    _add_common_cli_args(p_run)
    _add_parallel_cli_args(p_run)
    p_run.add_argument("case", help="Path to a .toml or .json case file.")
    p_run.add_argument(
        "--out",
        default=None,
        help=(
            "Write the NetCDF Result here. Omitted, the case's [output].file is used "
            "(relative to the case file). With [output].plots = true a .png summary "
            "is written beside it."
        ),
    )
    p_run.set_defaults(func=_cmd_run_case)

    p_inspect = sub.add_parser(
        "inspect",
        help="Print what a saved NetCDF Result contains.",
    )
    _add_common_cli_args(p_inspect)
    _add_parallel_cli_args(p_inspect)
    p_inspect.add_argument("result", help="Path to a DKX .nc result.")
    p_inspect.set_defaults(func=_cmd_inspect_result)

    p_convert = sub.add_parser(
        "convert",
        help="Convert a SFINCS input.namelist into a native .toml or .json case.",
    )
    _add_common_cli_args(p_convert)
    _add_parallel_cli_args(p_convert)
    p_convert.add_argument("source", help="Path to a SFINCS v3 input.namelist.")
    p_convert.add_argument(
        "destination",
        help="Case file to write; the .toml or .json extension picks the format.",
    )
    p_convert.add_argument(
        "--name", default=None, help="Case name (default: derived from SOURCE)."
    )
    p_convert.add_argument(
        "--force", action="store_true", help="Overwrite DESTINATION if it exists."
    )
    p_convert.set_defaults(func=_cmd_convert)

    # SFINCS-compatibility commands: visible under `dkx sfincs`, and still
    # accepted at the top level as hidden aliases so existing scripts run.
    p_sfincs = sub.add_parser(
        "sfincs",
        help="SFINCS v3 compatibility commands (namelist and HDF5 workflows).",
    )
    _add_compat_parsers(
        p_sfincs.add_subparsers(dest="sfincs_cmd", required=True)
    )
    _add_compat_parsers(_HiddenAliases(sub))


    argv = _normalize_default_argv(argv, set(sub.choices))
    _maybe_reexec_for_early_runtime(argv)
    args = parser.parse_args(argv)
    args = _merge_global_cli_args(argv, args)
    _apply_runtime_env_defaults()
    # No CLI-side default core count: when --cores/DKX_CORES is absent the
    # package import already clamped the threadpool to min(8, cpu_count).
    _apply_cores_setting(args.cores)
    _apply_parallel_runtime_settings(args)
    if args.fortran_stdout is True:
        os.environ["DKX_FORTRAN_STDOUT"] = "1"
    elif args.fortran_stdout is False:
        os.environ["DKX_FORTRAN_STDOUT"] = "0"
    else:
        os.environ.setdefault("DKX_FORTRAN_STDOUT", "1" if not getattr(args, "quiet", False) else "0")
    return int(args.func(args))

if __name__ == "__main__":
    raise SystemExit(main())
