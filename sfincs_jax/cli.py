from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from .input_compat import with_equilibrium_override
from .namelist import read_sfincs_input


def _now() -> float:
    return time.perf_counter()


def _emit(msg: str, *, level: int, args: argparse.Namespace) -> None:
    """Simple structured stdout logging for the CLI.

    We intentionally avoid the stdlib `logging` module here to keep CLI output
    deterministic across platforms and to make it easy to compare with upstream
    SFINCS logs.
    """
    verbose = int(getattr(args, "verbose", 0) or 0)
    quiet = bool(getattr(args, "quiet", False))
    if quiet:
        return
    if verbose >= level:
        print(msg, flush=True)


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


def _emit_parallel_runtime_info(*, args: argparse.Namespace) -> None:
    def _env(name: str, default: str = "") -> str:
        return os.environ.get(name, default).strip()

    cores = _env("SFINCS_JAX_CORES")
    cpu_devices = _env("SFINCS_JAX_CPU_DEVICES")
    shard_axis = _env("SFINCS_JAX_MATVEC_SHARD_AXIS")
    auto_shard = _env("SFINCS_JAX_AUTO_SHARD")
    shard_pad = _env("SFINCS_JAX_SHARD_PAD")
    transport_parallel = _env("SFINCS_JAX_TRANSPORT_PARALLEL", "off") or "off"
    transport_workers = _env("SFINCS_JAX_TRANSPORT_PARALLEL_WORKERS", "1") or "1"
    gmres_distributed = _env("SFINCS_JAX_GMRES_DISTRIBUTED")
    distributed_krylov = _env("SFINCS_JAX_DISTRIBUTED_KRYLOV")
    distributed = _env("SFINCS_JAX_DISTRIBUTED")

    if not any(
        (
            cores,
            cpu_devices,
            shard_axis,
            auto_shard,
            shard_pad,
            transport_parallel not in {"", "off"},
            gmres_distributed,
            distributed_krylov,
            distributed,
        )
    ):
        return

    _emit(
        " parallel:"
        f" cores={cores or '-'}"
        f" cpu_devices={cpu_devices or '-'}"
        f" shard_axis={shard_axis or '-'}"
        f" auto_shard={auto_shard or '-'}"
        f" shard_pad={shard_pad or '-'}",
        level=1,
        args=args,
    )
    _emit(
        f" transport_parallel: mode={transport_parallel} workers={transport_workers}",
        level=1,
        args=args,
    )
    _emit(
        " distributed_solver:"
        f" gmres={gmres_distributed or '-'}"
        f" krylov={distributed_krylov or '-'}",
        level=1,
        args=args,
    )
    if distributed in {"1", "true", "yes", "on"}:
        _emit(
            " multi_host:"
            " enabled=1"
            f" process_id={_env('SFINCS_JAX_PROCESS_ID', '-') or '-'}"
            f" process_count={_env('SFINCS_JAX_PROCESS_COUNT', '-') or '-'}"
            f" coordinator={_env('SFINCS_JAX_COORDINATOR_ADDRESS', '-') or '-'}"
            f" port={_env('SFINCS_JAX_COORDINATOR_PORT', '-') or '-'}",
            level=1,
            args=args,
        )


def _nml_with_cli_equilibrium_override(nml, args: argparse.Namespace):
    return with_equilibrium_override(
        nml=nml,
        equilibrium_file=getattr(args, "equilibrium_file", None),
        wout_path=getattr(args, "wout_path", None),
    )


def _add_equilibrium_override_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--equilibrium-file",
        default=None,
        help="Override geometryParameters.equilibriumFile without editing input.namelist.",
    )
    parser.add_argument(
        "--wout-path",
        default=None,
        help="Compatibility alias for --equilibrium-file, commonly used for geometryScheme=5 VMEC runs.",
    )


def _add_common_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=1,
        help="Increase verbosity (repeatable).",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Reduce output to a minimum.")
    parser.add_argument(
        "--cores",
        type=int,
        default=None,
        help="Number of host CPU devices/cores to use (sets SFINCS_JAX_CORES).",
    )
    parser.add_argument(
        "--fortran-stdout",
        dest="fortran_stdout",
        action="store_true",
        help="Mirror upstream v3 stdout line-for-line (including KSP/SNES iteration lines).",
    )
    parser.add_argument(
        "--no-fortran-stdout",
        dest="fortran_stdout",
        action="store_false",
        help="Disable strict Fortran-style stdout mirroring.",
    )
    parser.set_defaults(fortran_stdout=None)


def _add_parallel_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--transport-workers",
        type=int,
        default=None,
        help="Parallel worker processes for independent transport-RHS solves.",
    )
    parser.add_argument(
        "--shard-axis",
        default=None,
        choices=("auto", "off", "theta", "zeta", "x", "flat"),
        help="Single-solve sharding axis for the executable path.",
    )
    parser.add_argument(
        "--distributed-gmres",
        default=None,
        help="Override SFINCS_JAX_GMRES_DISTRIBUTED for distributed RHSMode=1 solves.",
    )
    parser.add_argument(
        "--distributed-krylov",
        default=None,
        help="Override SFINCS_JAX_DISTRIBUTED_KRYLOV for sharded solves.",
    )
    parser.add_argument(
        "--shard-pad",
        dest="shard_pad",
        action="store_true",
        default=None,
        help="Allow neutral padding when sharded dimensions are not divisible by device count.",
    )
    parser.add_argument(
        "--no-shard-pad",
        dest="shard_pad",
        action="store_false",
        help="Disable neutral padding for sharded dimensions.",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable JAX multi-host distributed initialization for this CLI run.",
    )
    parser.add_argument("--process-id", type=int, default=None, help="Multi-host JAX process id.")
    parser.add_argument("--process-count", type=int, default=None, help="Multi-host JAX process count.")
    parser.add_argument(
        "--coordinator-address",
        default=None,
        help="Multi-host JAX coordinator host or host:port.",
    )
    parser.add_argument(
        "--coordinator-port",
        type=int,
        default=None,
        help="Coordinator port when --coordinator-address omits it.",
    )


def _cmd_solve_v3(args: argparse.Namespace) -> int:
    t0 = _now()
    from .v3_driver import solve_v3_full_system_linear_gmres  # noqa: PLC0415

    nml = _nml_with_cli_equilibrium_override(read_sfincs_input(Path(args.input)), args)
    _emit("################################################################", level=0, args=args)
    _emit(" sfincs_jax solve-v3", level=0, args=args)
    _emit(f" input={Path(args.input).resolve()}", level=0, args=args)
    _emit_namelist_summary(nml=nml, args=args)
    _emit_runtime_info(args=args)
    _emit_parallel_runtime_info(args=args)
    _emit(f" tol={args.tol} atol={args.atol} restart={args.restart} maxiter={args.maxiter} solve_method={args.solve_method}", level=1, args=args)
    if args.which_rhs is not None:
        _emit(f" whichRHS={args.which_rhs}", level=0, args=args)
    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        which_rhs=int(args.which_rhs) if args.which_rhs is not None else None,
        tol=float(args.tol),
        atol=float(args.atol),
        restart=int(args.restart),
        maxiter=int(args.maxiter) if args.maxiter is not None else None,
        solve_method=str(args.solve_method),
        differentiable=False,
        emit=lambda level, msg: _emit(msg, level=level, args=args),
    )
    out_state = Path(args.out_state)
    out_state.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_state, np.asarray(result.x))
    _emit(f" wrote stateVector -> {out_state.resolve()}", level=0, args=args)
    _emit(f" residual_norm={float(result.residual_norm):.6e}", level=0, args=args)
    _emit(f" elapsed_s={_now()-t0:.3f}", level=1, args=args)
    return 0


def _cmd_run_fortran(args: argparse.Namespace) -> int:
    t0 = _now()
    from .fortran import run_sfincs_fortran  # noqa: PLC0415

    _emit("################################################################", level=0, args=args)
    _emit(" sfincs_jax run-fortran", level=0, args=args)
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
    from .io import _output_file_format, write_sfincs_jax_output_h5  # noqa: PLC0415

    nml = _nml_with_cli_equilibrium_override(read_sfincs_input(Path(args.input)), args)
    rhs_mode = int(nml.group("general").get("RHSMODE", 1))
    output_format = _output_file_format(Path(args.out))
    _emit("################################################################", level=0, args=args)
    _emit(" sfincs_jax write-output", level=0, args=args)
    _emit(f" input={Path(args.input).resolve()}", level=0, args=args)
    _emit(f" output={Path(args.out).resolve()} format={output_format}", level=0, args=args)
    _emit_namelist_summary(nml=nml, args=args)
    _emit_runtime_info(args=args)
    _emit_parallel_runtime_info(args=args)

    # Default to upstream v3 behavior: full solve/write appropriate to RHSMode.
    geometry_only = bool(getattr(args, "geometry_only", False))
    compute_solution = (not geometry_only) and (bool(getattr(args, "compute_solution", False)) or rhs_mode == 1)
    compute_transport_matrix = (not geometry_only) and (bool(args.compute_transport_matrix) or rhs_mode in (2, 3))

    try:
        out_path = write_sfincs_jax_output_h5(
            input_namelist=Path(args.input),
            output_path=Path(args.out),
            equilibrium_file=getattr(args, "equilibrium_file", None),
            wout_path=getattr(args, "wout_path", None),
            fortran_layout=bool(args.fortran_layout),
            overwrite=bool(args.overwrite),
            compute_transport_matrix=bool(compute_transport_matrix),
            compute_solution=bool(compute_solution),
            differentiable=False,
            solver_trace_path=Path(args.solver_trace) if getattr(args, "solver_trace", None) else None,
            solve_method=str(getattr(args, "solve_method", "auto")),
            emit=lambda level, msg: _emit(msg, level=level, args=args),
            verbose=not bool(getattr(args, "quiet", False)),
        )
    except RuntimeError as exc:
        if os.environ.get("SFINCS_JAX_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}:
            raise
        print(f"sfincs_jax write-output failed: {exc}", file=sys.stderr, flush=True)
        return 2
    _emit(f" wrote output -> {out_path}", level=0, args=args)
    _emit(f" elapsed_s={_now()-t0:.3f}", level=1, args=args)
    return 0


def _cmd_transport_matrix_v3(args: argparse.Namespace) -> int:
    t0 = _now()
    from .v3_driver import solve_v3_transport_matrix_linear_gmres  # noqa: PLC0415

    nml = _nml_with_cli_equilibrium_override(read_sfincs_input(Path(args.input)), args)
    _emit("################################################################", level=0, args=args)
    _emit(" sfincs_jax transport-matrix-v3", level=0, args=args)
    _emit(f" input={Path(args.input).resolve()}", level=0, args=args)
    _emit_namelist_summary(nml=nml, args=args)
    _emit_runtime_info(args=args)
    _emit_parallel_runtime_info(args=args)
    _emit(f" tol={args.tol} atol={args.atol} restart={args.restart} maxiter={args.maxiter} solve_method={args.solve_method}", level=1, args=args)
    result = solve_v3_transport_matrix_linear_gmres(
        nml=nml,
        tol=float(args.tol),
        atol=float(args.atol),
        restart=int(args.restart),
        maxiter=int(args.maxiter) if args.maxiter is not None else None,
        solve_method=str(args.solve_method),
        differentiable=False,
        emit=lambda level, msg: _emit(msg, level=level, args=args),
    )

    out_tm = Path(args.out_matrix)
    out_tm.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_tm, np.asarray(result.transport_matrix))
    _emit(f" wrote transportMatrix -> {out_tm.resolve()}", level=0, args=args)

    if args.out_state_prefix is not None:
        pref = Path(args.out_state_prefix)
        pref.parent.mkdir(parents=True, exist_ok=True)
        for which_rhs, x in sorted(result.state_vectors_by_rhs.items()):
            p = pref.with_name(f"{pref.name}.whichRHS{which_rhs}.npy")
            np.save(p, np.asarray(x))
            _emit(f" wrote stateVector(whichRHS={which_rhs}) -> {p.resolve()}", level=1, args=args)

    for which_rhs, rn in sorted(result.residual_norms_by_rhs.items()):
        _emit(f" whichRHS={which_rhs} residual_norm={float(rn):.6e}", level=0, args=args)
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
    _emit(" sfincs_jax plot-output", level=0, args=args)
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
    from .scans import linspace_including_endpoints, run_er_scan  # noqa: PLC0415

    _emit("################################################################", level=0, args=args)
    _emit(" sfincs_jax scan-er", level=0, args=args)
    _emit(f" input={Path(args.input).resolve()}", level=0, args=args)
    _emit(f" out-dir={Path(args.out_dir).resolve()}", level=0, args=args)
    _emit_runtime_info(args=args)
    _emit_parallel_runtime_info(args=args)

    if args.values is not None:
        values = [float(x) for x in args.values]
    else:
        values = list(linspace_including_endpoints(float(args.min), float(args.max), int(args.n)))

    run_er_scan(
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
    return 0


def _cmd_ambipolar_solve(args: argparse.Namespace) -> int:
    t0 = _now()
    from .ambipolar import solve_ambipolar_from_scan_dir  # noqa: PLC0415

    _emit("################################################################", level=0, args=args)
    _emit(" sfincs_jax ambipolar-solve", level=0, args=args)
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
    from .problems.ambipolar import solve_sfincs_jax_ambipolar_brent  # noqa: PLC0415

    _emit("################################################################", level=0, args=args)
    _emit(" sfincs_jax ambipolar", level=0, args=args)
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
        f" current_tolerance={float(args.current_tolerance):.3e}"
        f" step_tolerance={float(args.step_tolerance):.3e}",
        level=0,
        args=args,
    )

    result, evaluator = solve_sfincs_jax_ambipolar_brent(
        input_namelist=Path(args.input),
        work_dir=Path(args.out_dir),
        er_min=float(args.er_min),
        er_max=float(args.er_max),
        er_initial=float(args.er_initial),
        max_evaluations=int(args.max_evaluations),
        current_tolerance=float(args.current_tolerance),
        step_tolerance=float(args.step_tolerance),
        solve_method=str(args.solve_method),
        differentiable=False,
        reuse_output_geometry_cache=not bool(getattr(args, "no_output_cache", False)),
        reuse_solver_state=not bool(getattr(args, "no_solver_state", False)),
        emit=lambda level, msg: _emit(msg, level=level, args=args),
    )

    summary_path = Path(args.summary_json) if args.summary_json else Path(args.out_dir) / "ambipolar_result.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "converged": bool(result.converged),
        "method": result.method,
        "status": result.status,
        "message": result.message,
        "root_er": result.root_er,
        "root_radial_current": result.root_radial_current,
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
        "evaluations": [
            {
                "er": item.er,
                "radial_current": item.radial_current,
                "input_path": str(item.input_path),
                "output_path": str(item.output_path),
                "solver_trace_path": None if item.solver_trace_path is None else str(item.solver_trace_path),
                "selected_path": item.selected_path,
                "solve_method": item.solve_method,
                "preconditioner": item.preconditioner,
                "residual_norm": item.residual_norm,
                "residual_target": item.residual_target,
                "converged": item.converged,
                "setup_s": item.setup_s,
                "solve_s": item.solve_s,
                "elapsed_s": item.elapsed_s,
                "total_size": item.total_size,
                "active_size": item.active_size,
                "cache_enabled": item.cache_enabled,
                "cache_dir": None if item.cache_dir is None else str(item.cache_dir),
                "solver_state_reuse_enabled": item.solver_state_reuse_enabled,
                "solver_state_path": None if item.solver_state_path is None else str(item.solver_state_path),
                "solver_state_input_exists": item.solver_state_input_exists,
                "solver_state_input_used": item.solver_state_input_used,
                "solver_state_output_exists": item.solver_state_output_exists,
                "fixed_shape_input_signature": (
                    None
                    if item.fixed_shape_input_signature is None
                    else list(item.fixed_shape_input_signature)
                ),
                "fixed_shape_signature": (
                    None if item.fixed_shape_signature is None else list(item.fixed_shape_signature)
                ),
                "fixed_shape_reuse_enabled": item.fixed_shape_reuse_enabled,
                "fixed_shape_reuse_admitted": item.fixed_shape_reuse_admitted,
                "fixed_shape_reuse_reason": item.fixed_shape_reuse_reason,
                "fixed_shape_reuse_count": item.fixed_shape_reuse_count,
            }
            for item in evaluator.records
        ],
        "elapsed_s": float(_now() - t0),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    if result.converged:
        _emit(
            f" ambipolar root: Er={float(result.root_er):.16g} "
            f"radial_current={float(result.root_radial_current):.6e} type={result.root_type}",
            level=0,
            args=args,
        )
    else:
        _emit(f" ambipolar status={result.status}: {result.message}", level=0, args=args)
    _emit(f" wrote summary -> {summary_path.resolve()}", level=0, args=args)
    _emit(f" elapsed_s={_now()-t0:.3f}", level=1, args=args)
    return 0 if result.converged else 2


def _apply_cores_setting(cores: int | None) -> None:
    if cores is None:
        return
    try:
        cores_val = int(cores)
    except (TypeError, ValueError):
        return
    if cores_val <= 0:
        return
    os.environ["SFINCS_JAX_CORES"] = str(cores_val)
    backend_hint = os.environ.get("JAX_PLATFORM_NAME", "").strip().lower()
    if cores_val > 1 and backend_hint not in {"gpu", "cuda", "rocm"}:
        os.environ.setdefault("SFINCS_JAX_GMRES_DISTRIBUTED", "auto")
    # Ensure host device count and XLA threading reflect the requested cores.
    xla_flags = os.environ.get("XLA_FLAGS", "")
    xla_parts = [p for p in xla_flags.split() if not p.startswith("--xla_force_host_platform_device_count=")]
    xla_parts.append(f"--xla_force_host_platform_device_count={cores_val}")
    os.environ["XLA_FLAGS"] = " ".join(xla_parts).strip()
    os.environ.setdefault("SFINCS_JAX_CPU_DEVICES", str(cores_val))
    # Enable auto-sharding unless explicitly disabled.
    shard_env = os.environ.get("SFINCS_JAX_SHARD", "").strip().lower()
    if cores_val > 1 and shard_env not in {"0", "false", "no", "off"}:
        os.environ.setdefault("SFINCS_JAX_MATVEC_SHARD_AXIS", "auto")
        os.environ.setdefault("SFINCS_JAX_AUTO_SHARD", "1")


def _apply_runtime_env_defaults() -> None:
    # Avoid large eager GPU preallocation in CLI workflows so solver/benchmark
    # runs coexist better with other accelerator jobs by default.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def _apply_parallel_runtime_settings(args: argparse.Namespace) -> None:
    transport_workers = getattr(args, "transport_workers", None)
    if transport_workers is not None:
        workers_val = max(1, int(transport_workers))
        os.environ["SFINCS_JAX_TRANSPORT_PARALLEL"] = "process" if workers_val > 1 else "off"
        os.environ["SFINCS_JAX_TRANSPORT_PARALLEL_WORKERS"] = str(workers_val)

    shard_axis = getattr(args, "shard_axis", None)
    if shard_axis is not None:
        shard_axis = str(shard_axis).strip().lower()
        if shard_axis == "off":
            os.environ["SFINCS_JAX_SHARD"] = "0"
            os.environ["SFINCS_JAX_AUTO_SHARD"] = "0"
            os.environ["SFINCS_JAX_MATVEC_SHARD_AXIS"] = "off"
        elif shard_axis == "auto":
            os.environ["SFINCS_JAX_SHARD"] = "1"
            os.environ["SFINCS_JAX_AUTO_SHARD"] = "1"
            os.environ["SFINCS_JAX_MATVEC_SHARD_AXIS"] = "auto"
        else:
            os.environ["SFINCS_JAX_SHARD"] = "1"
            os.environ["SFINCS_JAX_AUTO_SHARD"] = "0"
            os.environ["SFINCS_JAX_MATVEC_SHARD_AXIS"] = shard_axis

    distributed_gmres = getattr(args, "distributed_gmres", None)
    if distributed_gmres is not None:
        os.environ["SFINCS_JAX_GMRES_DISTRIBUTED"] = str(distributed_gmres)

    distributed_krylov = getattr(args, "distributed_krylov", None)
    if distributed_krylov is not None:
        os.environ["SFINCS_JAX_DISTRIBUTED_KRYLOV"] = str(distributed_krylov)

    shard_pad = getattr(args, "shard_pad", None)
    if shard_pad is not None:
        os.environ["SFINCS_JAX_SHARD_PAD"] = "1" if shard_pad else "0"

    if bool(getattr(args, "distributed", False)):
        from . import initialize_distributed_runtime_from_env  # noqa: PLC0415

        os.environ["SFINCS_JAX_DISTRIBUTED"] = "1"
        process_id = getattr(args, "process_id", None)
        process_count = getattr(args, "process_count", None)
        coordinator_address = getattr(args, "coordinator_address", None)
        coordinator_port = getattr(args, "coordinator_port", None)
        if process_id is not None:
            os.environ["SFINCS_JAX_PROCESS_ID"] = str(int(process_id))
        if process_count is not None:
            os.environ["SFINCS_JAX_PROCESS_COUNT"] = str(int(process_count))
        if coordinator_address:
            os.environ["SFINCS_JAX_COORDINATOR_ADDRESS"] = str(coordinator_address)
        if coordinator_port is not None:
            os.environ["SFINCS_JAX_COORDINATOR_PORT"] = str(int(coordinator_port))
        initialize_distributed_runtime_from_env()


def _auto_cores_for_args(args: argparse.Namespace) -> int:
    """Choose a conservative default core count by workload type.

    On CPU, single-RHS RHSMode=1 solves (especially nonlinear includePhi1)
    can regress with multi-device host sharding due synchronization/launch
    overhead. Prefer one core by default there; keep a few cores for
    transport-matrix/multi-RHS throughput workloads.
    """
    cpu_count = max(1, int(os.cpu_count() or 1))
    cmd = getattr(getattr(args, "func", None), "__name__", "")
    if cmd in {"_cmd_transport_matrix_v3", "_cmd_scan_er"}:
        return min(3, cpu_count)
    input_path = getattr(args, "input", None)
    if input_path is None:
        return min(3, cpu_count)
    try:
        nml = read_sfincs_input(Path(input_path))
        rhs_mode = int(nml.group("general").get("RHSMODE", 1))
    except Exception:  # noqa: BLE001
        return min(3, cpu_count)
    if rhs_mode in (2, 3):
        return min(3, cpu_count)
    return 1


def _normalize_default_argv(argv: list[str]) -> list[str]:
    if not argv:
        return argv
    known_cmds = {
        "solve-v3",
        "ambipolar",
        "scan-er",
        "ambipolar-solve",
        "run-fortran",
        "write-output",
        "transport-matrix-v3",
        "dump-h5",
        "plot-output",
        "compare-h5",
        "postprocess-upstream",
    }
    if any(tok in known_cmds for tok in argv):
        return argv
    global_opts_with_val = {
        "--cores",
        "--transport-workers",
        "--shard-axis",
        "--distributed-gmres",
        "--distributed-krylov",
        "--process-id",
        "--process-count",
        "--coordinator-address",
        "--coordinator-port",
    }
    global_opts_no_val = {
        "-v",
        "--verbose",
        "-q",
        "--quiet",
        "--fortran-stdout",
        "--no-fortran-stdout",
        "--shard-pad",
        "--no-shard-pad",
        "--distributed",
    }
    if "--plot" in argv:
        global_args: list[str] = []
        rest: list[str] = []
        input_h5: str | None = None
        idx = 0
        while idx < len(argv):
            tok = argv[idx]
            if tok in global_opts_with_val:
                if idx + 1 < len(argv):
                    global_args.extend([tok, argv[idx + 1]])
                    idx += 2
                    continue
            if tok.startswith("--cores="):
                global_args.append(tok)
                idx += 1
                continue
            if tok in global_opts_no_val:
                global_args.append(tok)
                idx += 1
                continue
            if tok == "--plot":
                if idx + 1 < len(argv):
                    input_h5 = argv[idx + 1]
                    idx += 2
                    continue
            rest.append(tok)
            idx += 1
        if input_h5 is not None:
            return [*global_args, "plot-output", "--input-h5", input_h5, *rest]
    global_args: list[str] = []
    rest: list[str] = []
    input_path: str | None = None
    idx = 0
    while idx < len(argv):
        tok = argv[idx]
        if tok in global_opts_with_val:
            if idx + 1 < len(argv):
                global_args.extend([tok, argv[idx + 1]])
                idx += 2
                continue
        if tok.startswith("--cores="):
            global_args.append(tok)
            idx += 1
            continue
        if tok in global_opts_no_val:
            global_args.append(tok)
            idx += 1
            continue
        if tok.startswith("-"):
            rest.append(tok)
            idx += 1
            continue
        if input_path is None:
            input_path = tok
        else:
            rest.append(tok)
        idx += 1
    if input_path is None:
        return argv
    return [*global_args, "write-output", "--input", input_path, *rest]


def _maybe_reexec_for_early_runtime(argv: list[str]) -> None:
    """Re-exec with early runtime env so host device count/bootstrap take effect.

    The CLI is imported after the package, so JAX may already be imported before
    flags like `--cores` or `--distributed` are parsed. When those flags would
    change pre-import runtime state, restart the process once with the relevant
    env vars set before package import.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--cores", type=int, default=None)
    pre.add_argument("--distributed", action="store_true")
    pre.add_argument("--process-id", type=int, default=None)
    pre.add_argument("--process-count", type=int, default=None)
    pre.add_argument("--coordinator-address", default=None)
    pre.add_argument("--coordinator-port", type=int, default=None)
    args, _ = pre.parse_known_args(argv)

    desired: dict[str, str] = {}
    if args.cores is not None and int(args.cores) > 0:
        desired["SFINCS_JAX_CORES"] = str(int(args.cores))
        desired["SFINCS_JAX_CPU_DEVICES"] = str(int(args.cores))
    if bool(args.distributed):
        desired["SFINCS_JAX_DISTRIBUTED"] = "1"
        if args.process_id is not None:
            desired["SFINCS_JAX_PROCESS_ID"] = str(int(args.process_id))
        if args.process_count is not None:
            desired["SFINCS_JAX_PROCESS_COUNT"] = str(int(args.process_count))
        if args.coordinator_address is not None:
            desired["SFINCS_JAX_COORDINATOR_ADDRESS"] = str(args.coordinator_address)
        if args.coordinator_port is not None:
            desired["SFINCS_JAX_COORDINATOR_PORT"] = str(int(args.coordinator_port))

    if not desired:
        return

    if all(os.environ.get(key, "") == value for key, value in desired.items()):
        return

    env = os.environ.copy()
    env.update(desired)
    env["SFINCS_JAX_CLI_BOOTSTRAPPED"] = "1"
    os.execvpe(sys.executable, [sys.executable, "-m", "sfincs_jax", *argv], env)


def _merge_global_cli_args(argv: list[str], args: argparse.Namespace) -> argparse.Namespace:
    """Preserve global CLI flags regardless of whether they appear before or after the subcommand.

    Argparse defaults on both the root parser and subparsers can otherwise cause
    root-level values to be overwritten by subparser defaults when a flag is
    supplied before the subcommand. Parse the shared global options once more
    from the full argv and reapply them onto the final namespace.
    """
    pre = argparse.ArgumentParser(add_help=False)
    _add_common_cli_args(pre)
    _add_parallel_cli_args(pre)
    pre_args, _ = pre.parse_known_args(argv)
    for name in (
        "verbose",
        "quiet",
        "cores",
        "fortran_stdout",
        "transport_workers",
        "shard_axis",
        "distributed_gmres",
        "distributed_krylov",
        "shard_pad",
        "distributed",
        "process_id",
        "process_count",
        "coordinator_address",
        "coordinator_port",
    ):
        setattr(args, name, getattr(pre_args, name))
    return args


def main(argv: list[str] | None = None) -> int:
    """Run the sfincs_jax command-line interface."""
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    argv = _normalize_default_argv(argv)
    _maybe_reexec_for_early_runtime(argv)
    parser = argparse.ArgumentParser(prog="sfincs_jax")
    _add_common_cli_args(parser)
    _add_parallel_cli_args(parser)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_solve = sub.add_parser("solve-v3", help="Solve a supported v3 linear problem matrix-free and write stateVector.npy.")
    _add_common_cli_args(p_solve)
    _add_parallel_cli_args(p_solve)
    p_solve.add_argument("--input", required=True, help="Path to input.namelist")
    p_solve.add_argument("--out-state", default="stateVector.npy", help="Where to write the solution vector (NumPy .npy)")
    p_solve.add_argument("--tol", default="1e-10", help="GMRES relative tolerance")
    p_solve.add_argument("--atol", default="0.0", help="GMRES absolute tolerance")
    p_solve.add_argument("--restart", default="80", help="GMRES restart")
    p_solve.add_argument("--maxiter", default=None, help="GMRES maxiter (default: library default)")
    p_solve.add_argument(
        "--solve-method",
        default="auto",
        help="Advanced solver override. Default 'auto' is recommended for normal runs; see docs/usage.rst.",
    )
    p_solve.add_argument(
        "--which-rhs",
        default=None,
        help="For RHSMode=2/3 transport-matrix runs, select whichRHS (v3 loops over multiple RHS).",
    )
    _add_equilibrium_override_args(p_solve)
    p_solve.set_defaults(func=_cmd_solve_v3)

    p_scan = sub.add_parser(
        "scan-er",
        help="Run an Er (or dPhiHatd*) scan by writing sfincsOutput.h5 in multiple run directories.",
    )
    _add_common_cli_args(p_scan)
    _add_parallel_cli_args(p_scan)
    p_scan.add_argument("--input", required=True, help="Path to input.namelist (template).")
    p_scan.add_argument("--out-dir", required=True, help="Directory to create scan subdirectories inside.")
    p_scan.add_argument(
        "--compute-transport-matrix",
        action="store_true",
        help="Also compute RHSMode=2/3 transport-matrix outputs (slow).",
    )
    p_scan.add_argument(
        "--compute-solution",
        action="store_true",
        help="For RHSMode=1 runs, also solve and write solution-derived fields (may be slow).",
    )
    p_scan.add_argument("--min", default="-1.0", help="Minimum value (ignored if --values is provided).")
    p_scan.add_argument("--max", default="1.0", help="Maximum value (ignored if --values is provided).")
    p_scan.add_argument("--n", default="5", help="Number of points (ignored if --values is provided).")
    p_scan.add_argument("--values", default=None, nargs="+", help="Explicit list of values to use.")
    p_scan.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing sfincsOutput.h5 files and only solve missing scan points.",
    )
    p_scan.add_argument("--jobs", type=int, default=1, help="Parallel worker processes for scan points.")
    p_scan.add_argument("--index", type=int, default=None, help="Optional job-array index (0-based).")
    p_scan.add_argument("--stride", type=int, default=1, help="Stride for job-array slicing.")
    p_scan.add_argument(
        "--solve-method",
        default="auto",
        help="Advanced RHSMode=1 solution solve override for --compute-solution scan points.",
    )
    p_scan.set_defaults(func=_cmd_scan_er)

    p_ambi = sub.add_parser(
        "ambipolar-solve",
        help="Given an existing scan-er directory, solve for ambipolar Er roots and write ambipolarSolutions.dat.",
    )
    _add_common_cli_args(p_ambi)
    _add_parallel_cli_args(p_ambi)
    p_ambi.add_argument("--scan-dir", required=True, help="Scan directory produced by `sfincs_jax scan-er`.")
    p_ambi.add_argument("--n-fine", default="500", help="Number of fine-grid points for bracketing (default: 500).")
    p_ambi.set_defaults(func=_cmd_ambipolar_solve)

    p_ambi_direct = sub.add_parser(
        "ambipolar",
        help="Run an in-process Brent ambipolar Er solve from input.namelist.",
    )
    _add_common_cli_args(p_ambi_direct)
    _add_parallel_cli_args(p_ambi_direct)
    p_ambi_direct.add_argument("--input", required=True, help="Path to input.namelist.")
    p_ambi_direct.add_argument(
        "--out-dir",
        default="ambipolar_run",
        help="Directory for per-evaluation inputs, outputs, traces, and the summary JSON.",
    )
    p_ambi_direct.add_argument("--er-min", default="-100.0", help="Lower Er bracket.")
    p_ambi_direct.add_argument("--er-max", default="100.0", help="Upper Er bracket.")
    p_ambi_direct.add_argument("--er-initial", default="0.0", help="Initial Er evaluation.")
    p_ambi_direct.add_argument("--max-evaluations", default="20", help="Maximum Brent radial-current evaluations.")
    p_ambi_direct.add_argument("--current-tolerance", default="1e-10", help="Radial-current convergence tolerance.")
    p_ambi_direct.add_argument("--step-tolerance", default="1e-8", help="Reserved Er-step tolerance for Newton-compatible APIs.")
    p_ambi_direct.add_argument(
        "--solve-method",
        default="auto",
        help="Advanced RHSMode=1 solver override for each radial-current evaluation.",
    )
    p_ambi_direct.add_argument(
        "--summary-json",
        default=None,
        help="Optional summary JSON path. Default: <out-dir>/ambipolar_result.json.",
    )
    p_ambi_direct.add_argument(
        "--no-output-cache",
        action="store_true",
        help="Disable the per-run geometry/output cache used across Er evaluations.",
    )
    p_ambi_direct.add_argument(
        "--no-solver-state",
        action="store_true",
        help="Disable shape-checked Krylov state reuse across Er evaluations.",
    )
    p_ambi_direct.set_defaults(func=_cmd_ambipolar)

    p_run = sub.add_parser("run-fortran", help="Run the compiled Fortran SFINCS v3 executable.")
    _add_common_cli_args(p_run)
    _add_parallel_cli_args(p_run)
    p_run.add_argument("--input", required=True, help="Path to input.namelist")
    p_run.add_argument("--exe", default=None, help="Path to Fortran v3 sfincs executable")
    p_run.add_argument("--workdir", default=None, help="Directory to run in (default: temp dir)")
    p_run.set_defaults(func=_cmd_run_fortran)

    p_out = sub.add_parser(
        "write-output",
        help="Write a SFINCS output file; the --out suffix selects HDF5, NetCDF4, or NPZ.",
    )
    _add_common_cli_args(p_out)
    _add_parallel_cli_args(p_out)
    p_out.add_argument("--input", required=True, help="Path to input.namelist")
    p_out.add_argument(
        "--out",
        default="sfincsOutput.h5",
        help="Output path. Suffix selects format: .h5/.hdf5, .nc/.netcdf, or .npz.",
    )
    p_out.add_argument(
        "--no-fortran-layout",
        dest="fortran_layout",
        action="store_false",
        default=True,
        help="Disable Fortran-compatible array layout (not recommended for parity)",
    )
    p_out.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        default=True,
        help="Fail if output already exists",
    )
    p_out.add_argument(
        "--compute-transport-matrix",
        action="store_true",
        help="Force transport-matrix solves for RHSMode=2/3 (default: enabled when RHSMode=2/3).",
    )
    p_out.add_argument(
        "--compute-solution",
        action="store_true",
        help="Force RHSMode=1 solves (default: enabled when RHSMode=1).",
    )
    p_out.add_argument(
        "--geometry-only",
        action="store_true",
        help="Only write geometry/grid outputs (skip RHSMode=1 solve and RHSMode=2/3 transport-matrix loop).",
    )
    p_out.add_argument(
        "--solver-trace",
        default=None,
        help="Optional JSON sidecar path for solver/backend/timing metadata.",
    )
    p_out.add_argument(
        "--solve-method",
        default="auto",
        help="Advanced RHSMode=1 solver override. Default 'auto' is recommended for normal runs; see docs/usage.rst.",
    )
    _add_equilibrium_override_args(p_out)
    p_out.set_defaults(func=_cmd_write_output)

    p_tm = sub.add_parser("transport-matrix-v3", help="Solve RHSMode=2/3 transport-matrix systems and write transportMatrix.npy.")
    _add_common_cli_args(p_tm)
    _add_parallel_cli_args(p_tm)
    p_tm.add_argument("--input", required=True, help="Path to input.namelist (must have RHSMode=2 or 3)")
    p_tm.add_argument("--out-matrix", default="transportMatrix.npy", help="Where to write the transport matrix (NumPy .npy)")
    p_tm.add_argument(
        "--out-state-prefix",
        default=None,
        help="Optional prefix for saving solution vectors as <prefix>.whichRHS{k}.npy",
    )
    p_tm.add_argument("--tol", default="1e-10", help="GMRES relative tolerance")
    p_tm.add_argument("--atol", default="0.0", help="GMRES absolute tolerance")
    p_tm.add_argument("--restart", default="80", help="GMRES restart")
    p_tm.add_argument("--maxiter", default=None, help="GMRES maxiter (default: library default)")
    p_tm.add_argument(
        "--solve-method",
        default="auto",
        help="Advanced transport solver override. Default 'auto' is recommended for normal runs; see docs/usage.rst.",
    )
    _add_equilibrium_override_args(p_tm)
    p_tm.set_defaults(func=_cmd_transport_matrix_v3)

    p_dump = sub.add_parser("dump-h5", help="Dump SFINCS HDF5 output to JSON (small files only).")
    _add_common_cli_args(p_dump)
    _add_parallel_cli_args(p_dump)
    p_dump.add_argument("--sfincs-output", required=True, help="Path to sfincsOutput.h5")
    p_dump.add_argument("--out-json", required=True, help="Where to write JSON")
    p_dump.add_argument("--keys-only", action="store_true", help="Only print dataset names")
    p_dump.set_defaults(func=_cmd_dump_h5)

    p_plot = sub.add_parser("plot-output", help="Write a diagnostics PDF/figure panel from a SFINCS output file.")
    _add_common_cli_args(p_plot)
    _add_parallel_cli_args(p_plot)
    p_plot.add_argument("--input-h5", required=True, help="Path to sfincsOutput.h5/.nc/.npz")
    p_plot.add_argument(
        "--out",
        default=None,
        help="Where to write the diagnostics panel (default: <input>_summary.pdf next to the output file).",
    )
    p_plot.set_defaults(func=_cmd_plot_output)

    p_cmp = sub.add_parser("compare-h5", help="Compare two SFINCS HDF5 output files.")
    _add_common_cli_args(p_cmp)
    _add_parallel_cli_args(p_cmp)
    p_cmp.add_argument("--a", required=True, help="First sfincsOutput.h5")
    p_cmp.add_argument("--b", required=True, help="Second sfincsOutput.h5")
    p_cmp.add_argument("--rtol", default="1e-12")
    p_cmp.add_argument("--atol", default="1e-12")
    p_cmp.add_argument("--tolerances-json", default=None, help="Optional JSON file of per-key tolerances")
    p_cmp.add_argument("--show-all", action="store_true", help="Print all keys (not just failures)")
    p_cmp.set_defaults(func=_cmd_compare_h5)

    p_pp = sub.add_parser(
        "postprocess-upstream",
        help="Run a vendored upstream v3 utils/ postprocessing script (best-effort, requires sfincsOutput.h5).",
    )
    _add_common_cli_args(p_pp)
    _add_parallel_cli_args(p_pp)
    p_pp.add_argument("--case-dir", required=True, help="Directory containing sfincsOutput.h5")
    p_pp.add_argument("--util", required=True, help="Upstream util script name (e.g. sfincsScanPlot_1)")
    p_pp.add_argument("--utils-dir", default=None, help="Override utils/ directory (else auto-detect / env var)")
    p_pp.add_argument("--interactive", action="store_true", help="Do not override input() (may hang in CI)")
    p_pp.add_argument(
        "util_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the upstream script (e.g. 'pdf'). Prefix with '--' to separate args.",
    )

    def _cmd_postprocess_upstream(args: argparse.Namespace) -> int:
        t0 = _now()
        from .postprocess_upstream import run_upstream_util  # noqa: PLC0415

        _emit("################################################################", level=0, args=args)
        _emit(" sfincs_jax postprocess-upstream", level=0, args=args)
        _emit(f" case_dir={Path(args.case_dir).resolve()}", level=0, args=args)
        _emit(f" util={args.util}", level=0, args=args)
        if args.utils_dir is not None:
            _emit(f" utils_dir={Path(args.utils_dir).resolve()}", level=1, args=args)
        util_args = list(args.util_args or [])
        if util_args and util_args[0] == "--":
            util_args = util_args[1:]
        run_upstream_util(
            util=str(args.util),
            case_dir=Path(args.case_dir),
            args=util_args,
            utils_dir=Path(args.utils_dir) if args.utils_dir is not None else None,
            noninteractive=not bool(args.interactive),
            emit=lambda level, msg: _emit(msg, level=level, args=args),
        )
        _emit(f" elapsed_s={_now()-t0:.3f}", level=1, args=args)
        return 0

    p_pp.set_defaults(func=_cmd_postprocess_upstream)

    args = parser.parse_args(argv)
    args = _merge_global_cli_args(argv, args)
    _apply_runtime_env_defaults()
    if args.cores is None and not os.environ.get("SFINCS_JAX_CORES"):
        if not (os.environ.get("SFINCS_JAX_CI") or os.environ.get("CI")):
            args.cores = _auto_cores_for_args(args)
    _apply_cores_setting(args.cores)
    _apply_parallel_runtime_settings(args)
    if args.fortran_stdout is True:
        os.environ["SFINCS_JAX_FORTRAN_STDOUT"] = "1"
    elif args.fortran_stdout is False:
        os.environ["SFINCS_JAX_FORTRAN_STDOUT"] = "0"
    else:
        os.environ.setdefault("SFINCS_JAX_FORTRAN_STDOUT", "1" if not getattr(args, "quiet", False) else "0")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
