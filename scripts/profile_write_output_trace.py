#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import jax
from jax import profiler as jax_profiler

from sfincs_jax.io import localize_equilibrium_file_in_place, write_sfincs_jax_output_h5


def _prepare_input(input_path: Path, *, equilibrium_file: str | None = None, wout_path: str | None = None) -> tuple[Path, Path]:
    tmpdir = Path(tempfile.mkdtemp(prefix="sfincs_jax_write_output_trace_"))
    dst = tmpdir / "input.namelist"
    shutil.copy2(input_path, dst)
    if equilibrium_file or wout_path:
        return dst, tmpdir
    old_search = os.environ.get("SFINCS_JAX_EQUILIBRIA_DIRS")
    search_dirs = [str(input_path.parent.resolve())]
    if old_search:
        search_dirs.append(old_search)
    os.environ["SFINCS_JAX_EQUILIBRIA_DIRS"] = os.pathsep.join(search_dirs)
    try:
        localize_equilibrium_file_in_place(input_namelist=dst, overwrite=False)
    finally:
        if old_search is None:
            os.environ.pop("SFINCS_JAX_EQUILIBRIA_DIRS", None)
        else:
            os.environ["SFINCS_JAX_EQUILIBRIA_DIRS"] = old_search
    return dst, tmpdir


def _run_write_output(
    *,
    input_path: Path,
    output_path: Path,
    compute_solution: bool,
    compute_transport_matrix: bool,
    equilibrium_file: str | None,
    wout_path: str | None,
    solver_trace_path: Path | None = None,
) -> None:
    write_sfincs_jax_output_h5(
        input_namelist=input_path,
        output_path=output_path,
        compute_solution=bool(compute_solution),
        compute_transport_matrix=bool(compute_transport_matrix),
        equilibrium_file=equilibrium_file,
        wout_path=wout_path,
        overwrite=True,
        verbose=True,
        solver_trace_path=solver_trace_path,
    )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON sidecar atomically so timeout/debug runs keep usable state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _exception_summary(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Profile a full sfincs_jax write-output run with JAX trace capture. "
            "Use --warmup 0 to include compile/lowering time, or --warmup 1 to focus on steady-state kernels."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="Path to input.namelist.")
    parser.add_argument(
        "--trace-dir",
        type=Path,
        required=True,
        help="Directory for the JAX/XProf trace output.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("sfincsOutput_profiled.h5"),
        help="Output H5 written by the profiled run.",
    )
    parser.add_argument("--warmup", type=int, default=1, help="Number of untraced warmup runs.")
    parser.add_argument(
        "--perfetto",
        action="store_true",
        help="Also emit perfetto_trace.json.gz for upload to ui.perfetto.dev.",
    )
    parser.add_argument(
        "--no-jax-trace",
        action="store_true",
        help=(
            "Skip the JAX/XProf trace context while keeping the phase log, "
            "output solve, and optional device-memory snapshot. Use this for "
            "long low-overhead production audits."
        ),
    )
    parser.add_argument(
        "--device-memory-profile",
        type=Path,
        default=None,
        help="Optional pprof-format device-memory snapshot written after the traced run.",
    )
    parser.add_argument(
        "--phase-log",
        type=Path,
        default=None,
        help=(
            "Optional JSON sidecar for timeout-safe phase timings. Defaults to "
            "<trace-dir>/profile_write_output_trace_phases.json."
        ),
    )
    parser.add_argument(
        "--phase-log-interval-s",
        type=float,
        default=10.0,
        help="Heartbeat interval for refreshing the phase log while a long solve is running; use 0 to disable.",
    )
    parser.add_argument(
        "--solver-trace",
        type=Path,
        default=None,
        help="Optional JSON solver-trace sidecar written by write_sfincs_jax_output_h5.",
    )
    parser.add_argument(
        "--strict-profiler",
        action="store_true",
        help=(
            "Return a nonzero status if profiler finalization or device-memory "
            "snapshotting fails after the solve. By default, a completed output "
            "file is treated as solve success and profiler teardown failures are "
            "recorded in the phase log."
        ),
    )
    parser.add_argument(
        "--compute-solution",
        action="store_true",
        help="Force solution arrays into the output file.",
    )
    parser.add_argument(
        "--compute-transport-matrix",
        action="store_true",
        help="Force transport-matrix arrays into the output file.",
    )
    parser.add_argument(
        "--equilibrium-file",
        default=None,
        help="Optional equilibrium file override, matching the CLI.",
    )
    parser.add_argument(
        "--wout-path",
        default=None,
        help="Compatibility alias for --equilibrium-file.",
    )
    args = parser.parse_args(argv)

    trace_dir = args.trace_dir.resolve()
    trace_dir.mkdir(parents=True, exist_ok=True)
    phase_log = (
        args.phase_log.resolve()
        if args.phase_log is not None
        else trace_dir / "profile_write_output_trace_phases.json"
    )
    overall_t0 = time.perf_counter()
    phase_payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "initializing",
        "input": str(args.input.resolve()),
        "trace_dir": str(trace_dir),
        "output": str(args.out.resolve()),
        "warmup": int(args.warmup),
        "perfetto": bool(args.perfetto),
        "jax_trace": not bool(args.no_jax_trace),
        "compute_solution": bool(args.compute_solution),
        "compute_transport_matrix": bool(args.compute_transport_matrix),
        "phase_log_interval_s": float(args.phase_log_interval_s),
        "solver_trace": str(args.solver_trace.resolve()) if args.solver_trace is not None else None,
        "device_memory_profile": str(args.device_memory_profile.resolve())
        if args.device_memory_profile is not None
        else None,
        "phases": [],
    }
    phase_log_lock = threading.RLock()

    def _flush(status: str | None = None) -> None:
        with phase_log_lock:
            if status is not None:
                phase_payload["status"] = status
            phase_payload["elapsed_s"] = time.perf_counter() - overall_t0
            _atomic_write_json(phase_log, phase_payload)

    def _start_phase(name: str, **extra: Any) -> dict[str, Any]:
        phase = {
            "name": name,
            "status": "running",
            "started_s": time.perf_counter() - overall_t0,
            **extra,
        }
        with phase_log_lock:
            phase_payload["phases"].append(phase)
        _flush("running")
        return phase

    def _finish_phase(phase: dict[str, Any], status: str = "ok", **extra: Any) -> None:
        with phase_log_lock:
            if phase.get("status") != "running":
                return
            phase["status"] = status
            phase["elapsed_s"] = time.perf_counter() - overall_t0 - float(phase.get("started_s", 0.0))
            phase.update(extra)
        _flush()

    os.environ.setdefault("JAX_ENABLE_X64", "True")
    cache_dir = (
        Path(os.environ["JAX_COMPILATION_CACHE_DIR"]).resolve()
        if os.environ.get("JAX_COMPILATION_CACHE_DIR", "").strip()
        else (trace_dir / ".jax_compilation_cache").resolve()
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(cache_dir))
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")
    jax.config.update("jax_compilation_cache_dir", str(cache_dir))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)

    output_path = args.out.resolve()
    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()
    phase_payload["output"] = str(output_path)
    _flush("initializing")

    phase = _start_phase("prepare_input")
    try:
        work_input, tmpdir = _prepare_input(
            args.input.resolve(),
            equilibrium_file=args.equilibrium_file,
            wout_path=args.wout_path,
        )
    except Exception as exc:  # noqa: BLE001
        _finish_phase(phase, "failed", exception=_exception_summary(exc))
        _flush("failed")
        print(f"Input preparation failed: {_exception_summary(exc)}", flush=True)
        return 1
    _finish_phase(phase, localized_input=str(work_input), localized_dir=str(tmpdir))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for warmup_index in range(max(0, int(args.warmup))):
        warmup_out = output_path.with_name(output_path.stem + ".warmup.h5")
        phase = _start_phase("warmup", index=warmup_index, output=str(warmup_out))
        try:
            _run_write_output(
                input_path=work_input,
                output_path=warmup_out,
                compute_solution=bool(args.compute_solution),
            compute_transport_matrix=bool(args.compute_transport_matrix),
            equilibrium_file=args.equilibrium_file,
            wout_path=args.wout_path,
            solver_trace_path=None,
        )
            if warmup_out.exists():
                warmup_out.unlink()
        except Exception as exc:  # noqa: BLE001
            _finish_phase(phase, "failed", exception=_exception_summary(exc))
            _flush("failed")
            print(f"Warmup {warmup_index} failed: {_exception_summary(exc)}", flush=True)
            return 1
        _finish_phase(phase)

    try:
        heartbeat_interval_s = max(0.0, float(args.phase_log_interval_s))
    except (TypeError, ValueError):
        heartbeat_interval_s = 10.0
    heartbeat_stop = threading.Event()

    def _heartbeat() -> None:
        while not heartbeat_stop.wait(heartbeat_interval_s):
            _flush()

    heartbeat_thread: threading.Thread | None = None
    if heartbeat_interval_s > 0.0:
        heartbeat_thread = threading.Thread(target=_heartbeat, name="sfincs_jax_phase_log_heartbeat", daemon=True)
        heartbeat_thread.start()

    trace_phase = _start_phase(
        "jax_trace",
        perfetto=bool(args.perfetto),
        enabled=not bool(args.no_jax_trace),
    )
    solve_phase: dict[str, Any] | None = None
    block_phase: dict[str, Any] | None = None
    solve_completed = False
    profiler_error: BaseException | None = None
    try:
        if args.no_jax_trace:
            solve_phase = _start_phase("write_output_solve", output=str(output_path))
            _run_write_output(
                input_path=work_input,
                output_path=output_path,
                compute_solution=bool(args.compute_solution),
                compute_transport_matrix=bool(args.compute_transport_matrix),
                equilibrium_file=args.equilibrium_file,
                wout_path=args.wout_path,
                solver_trace_path=args.solver_trace.resolve() if args.solver_trace is not None else None,
            )
            solve_completed = True
            _finish_phase(solve_phase, output_exists=output_path.exists())
            block_phase = _start_phase("block_until_ready")
            jax.block_until_ready(0)
            _finish_phase(block_phase)
        else:
            with jax_profiler.trace(
                str(trace_dir),
                create_perfetto_trace=bool(args.perfetto),
            ):
                solve_phase = _start_phase("write_output_solve", output=str(output_path))
                _run_write_output(
                    input_path=work_input,
                    output_path=output_path,
                    compute_solution=bool(args.compute_solution),
                    compute_transport_matrix=bool(args.compute_transport_matrix),
                    equilibrium_file=args.equilibrium_file,
                    wout_path=args.wout_path,
                    solver_trace_path=args.solver_trace.resolve() if args.solver_trace is not None else None,
                )
                solve_completed = True
                _finish_phase(solve_phase, output_exists=output_path.exists())
                block_phase = _start_phase("block_until_ready")
                jax.block_until_ready(0)
                _finish_phase(block_phase)
    except Exception as exc:  # noqa: BLE001
        profiler_error = exc
        if solve_phase is not None and solve_phase.get("status") == "running":
            _finish_phase(solve_phase, "failed", exception=_exception_summary(exc))
        if block_phase is not None and block_phase.get("status") == "running":
            _finish_phase(block_phase, "failed", exception=_exception_summary(exc))
        _finish_phase(trace_phase, "failed", exception=_exception_summary(exc))
    else:
        _finish_phase(trace_phase)
    heartbeat_stop.set()
    if heartbeat_thread is not None:
        heartbeat_thread.join(timeout=1.0)

    device_memory_error: BaseException | None = None
    if args.device_memory_profile is not None:
        mem_phase = _start_phase("device_memory_profile", output=str(args.device_memory_profile.resolve()))
        try:
            args.device_memory_profile.parent.mkdir(parents=True, exist_ok=True)
            jax_profiler.save_device_memory_profile(str(args.device_memory_profile))
        except Exception as exc:  # noqa: BLE001
            device_memory_error = exc
            _finish_phase(mem_phase, "failed", exception=_exception_summary(exc))
        else:
            _finish_phase(mem_phase)

    output_exists = output_path.exists()
    profiler_failed_after_solve = profiler_error is not None and solve_completed and output_exists
    device_profile_failed_after_solve = device_memory_error is not None and solve_completed and output_exists
    if profiler_error is not None and not profiler_failed_after_solve:
        _flush("failed")
        print(f"Trace failed before solve completion: {_exception_summary(profiler_error)}", flush=True)
        return 1
    if profiler_failed_after_solve and args.strict_profiler:
        _flush("failed")
        print(f"Profiler finalization failed: {_exception_summary(profiler_error)}", flush=True)
        return 1
    if device_memory_error is not None and args.strict_profiler:
        _flush("failed")
        print(f"Device-memory profiling failed: {_exception_summary(device_memory_error)}", flush=True)
        return 1

    status = "completed"
    if profiler_failed_after_solve or device_profile_failed_after_solve:
        status = "solve_completed_profile_incomplete"
    _flush(status)

    print(f"Wrote trace -> {trace_dir}")
    print(f"Wrote output -> {output_path}")
    print(f"Wrote phase log -> {phase_log}")
    print(f"Trace elapsed {float(trace_phase.get('elapsed_s', 0.0)):.3f}s")
    print(f"Localized input dir -> {tmpdir}")
    if profiler_failed_after_solve:
        print(
            "Profiler finalization failed after the output file was written; "
            f"recorded and returning solve success: {_exception_summary(profiler_error)}",
            flush=True,
        )
    if device_profile_failed_after_solve:
        print(
            "Device-memory profiling failed after the output file was written; "
            f"recorded and returning solve success: {_exception_summary(device_memory_error)}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
