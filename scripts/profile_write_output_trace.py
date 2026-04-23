#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import time
from pathlib import Path

import jax
from jax import profiler as jax_profiler

from sfincs_jax.io import localize_equilibrium_file_in_place, write_sfincs_jax_output_h5


def _prepare_input(input_path: Path) -> tuple[Path, Path]:
    tmpdir = Path(tempfile.mkdtemp(prefix="sfincs_jax_write_output_trace_"))
    dst = tmpdir / "input.namelist"
    shutil.copy2(input_path, dst)
    localize_equilibrium_file_in_place(input_namelist=dst, overwrite=False)
    return dst, tmpdir


def _run_write_output(
    *,
    input_path: Path,
    output_path: Path,
    compute_solution: bool,
    compute_transport_matrix: bool,
    equilibrium_file: str | None,
    wout_path: str | None,
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
    )


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
        "--device-memory-profile",
        type=Path,
        default=None,
        help="Optional pprof-format device-memory snapshot written after the traced run.",
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

    work_input, tmpdir = _prepare_input(args.input.resolve())
    output_path = args.out.resolve()
    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for _ in range(max(0, int(args.warmup))):
        warmup_out = output_path.with_name(output_path.stem + ".warmup.h5")
        _run_write_output(
            input_path=work_input,
            output_path=warmup_out,
            compute_solution=bool(args.compute_solution),
            compute_transport_matrix=bool(args.compute_transport_matrix),
            equilibrium_file=args.equilibrium_file,
            wout_path=args.wout_path,
        )
        if warmup_out.exists():
            warmup_out.unlink()

    t0 = time.perf_counter()
    with jax_profiler.trace(
        str(trace_dir),
        create_perfetto_trace=bool(args.perfetto),
    ):
        _run_write_output(
            input_path=work_input,
            output_path=output_path,
            compute_solution=bool(args.compute_solution),
            compute_transport_matrix=bool(args.compute_transport_matrix),
            equilibrium_file=args.equilibrium_file,
            wout_path=args.wout_path,
        )
        jax.block_until_ready(0)
    elapsed = time.perf_counter() - t0

    if args.device_memory_profile is not None:
        args.device_memory_profile.parent.mkdir(parents=True, exist_ok=True)
        jax_profiler.save_device_memory_profile(str(args.device_memory_profile))

    print(f"Wrote trace -> {trace_dir}")
    print(f"Wrote output -> {output_path}")
    print(f"Trace elapsed {elapsed:.3f}s")
    print(f"Localized input dir -> {tmpdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
