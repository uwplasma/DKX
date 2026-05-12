from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.v3_driver import full_system_operator_from_namelist
from sfincs_jax.v3_system import apply_v3_full_system_operator_cached


def _display_path(path: Path, *, repo_root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(repo_root))
    except ValueError:
        return str(path)


def _normalize_device_counts(requested_devices: list[int]) -> list[int]:
    devices = sorted({int(d) for d in requested_devices if int(d) >= 1})
    return devices or [1]


def _run_once(input_path: Path, *, nrep: int) -> float:
    os.environ["SFINCS_JAX_FORTRAN_STDOUT"] = "0"
    os.environ["SFINCS_JAX_SOLVER_ITER_STATS"] = "0"

    nml = read_sfincs_input(input_path)
    op = full_system_operator_from_namelist(nml=nml)

    rng = np.random.default_rng(0)
    x = rng.normal(size=(int(op.total_size),)).astype(np.float64)

    import jax
    import jax.numpy as jnp

    x_jax = jnp.asarray(x)
    x_jax.block_until_ready()

    def matvec(v):
        return apply_v3_full_system_operator_cached(op, v)

    matvec_jit = jax.jit(matvec)

    # Warmup (compile):
    y = matvec_jit(x_jax)
    y.block_until_ready()

    t0 = time.perf_counter()
    for _ in range(max(1, int(nrep))):
        y = matvec_jit(x_jax)
    y.block_until_ready()
    t1 = time.perf_counter()
    return (t1 - t0) / float(max(1, int(nrep)))


def _run_once_args(*, input_path: Path | str, nrep: int) -> list[str]:
    return [
        "--run-once",
        "--input",
        str(input_path),
        "--nrep",
        str(int(nrep)),
    ]


def _run_once_command(*, input_path: Path, nrep: int) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        *_run_once_args(input_path=input_path, nrep=nrep),
    ]


def _device_env_preview(
    *,
    devices: int,
    cache_dir: Path | None,
    axis: str,
    pad: bool,
    repo_root: Path,
) -> dict[str, str]:
    env = {
        "SFINCS_JAX_CPU_DEVICES": str(int(devices)),
        "SFINCS_JAX_MATVEC_SHARD_AXIS": str(axis),
        "SFINCS_JAX_AUTO_SHARD": "1",
        "SFINCS_JAX_SHARD_PAD": "1" if pad else "0",
        "SFINCS_JAX_FORTRAN_STDOUT": "0",
        "SFINCS_JAX_SOLVER_ITER_STATS": "0",
    }
    if cache_dir is not None:
        env["JAX_COMPILATION_CACHE_DIR"] = _display_path(cache_dir, repo_root=repo_root)
    return env


def _build_sharded_matvec_benchmark_plan(
    *,
    input_path: Path,
    devices: list[int],
    nrep: int,
    repeats: int,
    global_warmup: int,
    axis: str,
    pad: bool,
    out_dir: Path,
    cache_dir: Path,
) -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[2]
    normalized_devices = _normalize_device_counts(devices)
    input_display = _display_path(input_path, repo_root=repo_root)
    device_plan = []
    for d in normalized_devices:
        device_plan.append(
            {
                "devices": int(d),
                "timed_repeats": max(int(repeats), 1),
                "nrep_per_timing": max(int(nrep), 1),
                "env": _device_env_preview(
                    devices=d,
                    cache_dir=cache_dir,
                    axis=str(axis),
                    pad=bool(pad),
                    repo_root=repo_root,
                ),
                "run_once_command": [
                    "python",
                    "examples/performance/benchmark_sharded_matvec_scaling.py",
                    *_run_once_args(input_path=input_display, nrep=int(nrep)),
                ],
            }
        )
    return {
        "artifact_kind": "benchmark_plan",
        "benchmark_kind": "sharded_matvec_scaling",
        "launches_solves": False,
        "input": input_path.name,
        "input_path": input_display,
        "case": input_path.stem.replace(".input", ""),
        "axis": str(axis),
        "pad": bool(pad),
        "devices": normalized_devices,
        "nrep": int(nrep),
        "repeats": int(repeats),
        "global_warmup": int(global_warmup),
        "timing_semantics": "compiled_matvec_hot_loop",
        "device_plan": device_plan,
        "estimated_child_process_samples": int(max(global_warmup, 0))
        + len(normalized_devices) * max(int(repeats), 1),
        "speedup_gate_semantics": {
            "release_gate": False,
            "metric": "one_device_mean_s / device_mean_s",
            "matvec_microbenchmark_only": True,
        },
        "memory_gate_semantics": {
            "status": "not_measured_in_plan",
            "cpu_devices_per_child": "SFINCS_JAX_CPU_DEVICES",
            "padding_enabled": bool(pad),
        },
        "benchmark_command": [
            "python",
            "examples/performance/benchmark_sharded_matvec_scaling.py",
            "--input",
            input_display,
            "--axis",
            str(axis),
            "--devices",
            *[str(int(d)) for d in normalized_devices],
            "--nrep",
            str(int(nrep)),
            "--repeats",
            str(int(repeats)),
            "--global-warmup",
            str(int(global_warmup)),
            "--out-dir",
            _display_path(out_dir, repo_root=repo_root),
            "--cache-dir",
            _display_path(cache_dir, repo_root=repo_root),
            *(["--pad"] if pad else []),
        ],
    }


def _write_plan_json(plan: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")


def _run_once_subprocess(
    *, input_path: Path, devices: int, nrep: int, cache_dir: Path | None, axis: str, pad: bool
) -> float:
    env = os.environ.copy()
    env["SFINCS_JAX_CPU_DEVICES"] = str(int(devices))
    env["SFINCS_JAX_MATVEC_SHARD_AXIS"] = axis
    env["SFINCS_JAX_AUTO_SHARD"] = "1"
    env["SFINCS_JAX_SHARD_PAD"] = "1" if pad else "0"
    env["SFINCS_JAX_FORTRAN_STDOUT"] = "0"
    env["SFINCS_JAX_SOLVER_ITER_STATS"] = "0"
    if cache_dir is not None:
        env["JAX_COMPILATION_CACHE_DIR"] = str(cache_dir)

    cmd = _run_once_command(input_path=input_path, nrep=int(nrep))
    out = subprocess.check_output(cmd, env=env, text=True)
    return float(out.strip().splitlines()[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark sharded matvec scaling.")
    repo_root = Path(__file__).resolve().parents[2]
    default_input = repo_root / "examples" / "performance" / "transport_parallel_xxlarge.input.namelist"
    default_out = repo_root / "examples" / "performance" / "output" / "sharded_matvec_scaling"
    default_cache = default_out / "jax_cache"

    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help="Path to input.namelist for the full-system operator.",
    )
    parser.add_argument(
        "--axis",
        type=str,
        default="auto",
        choices=("auto", "theta", "zeta", "x", "flat"),
        help="Sharding axis to request (default auto).",
    )
    parser.add_argument(
        "--pad",
        action="store_true",
        help="Enable padding so odd grids can shard on even device counts.",
    )
    parser.add_argument(
        "--devices",
        type=int,
        nargs="+",
        default=list(range(1, 5)),
        help="CPU device counts to benchmark (default 1..4).",
    )
    parser.add_argument("--nrep", type=int, default=20, help="Matvec repeats per timing.")
    parser.add_argument("--repeats", type=int, default=2, help="Timing repeats per device count.")
    parser.add_argument(
        "--global-warmup",
        type=int,
        default=1,
        help="Warmup runs before benchmarking (uses devices=1).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=default_out,
        help="Output directory for JSON and figure.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=default_cache,
        help="Persistent JAX cache directory.",
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Run once and print wall time (internal).",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Write a deterministic benchmark plan JSON without launching matvec child processes.",
    )
    parser.add_argument(
        "--plan-json",
        type=Path,
        default=None,
        help="Path for --plan-only JSON (default: --out-dir/sharded_matvec_benchmark_plan.json).",
    )
    args = parser.parse_args()

    input_path = args.input
    if not input_path.exists():
        raise FileNotFoundError(str(input_path))

    if args.run_once:
        dt = _run_once(input_path, nrep=args.nrep)
        print(f"{dt:.6f}")
        return

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir

    devices = _normalize_device_counts([int(d) for d in args.devices])

    if args.plan_only:
        plan = _build_sharded_matvec_benchmark_plan(
            input_path=input_path,
            devices=devices,
            nrep=int(args.nrep),
            repeats=int(args.repeats),
            global_warmup=int(args.global_warmup),
            axis=str(args.axis),
            pad=bool(args.pad),
            out_dir=out_dir,
            cache_dir=cache_dir,
        )
        plan_path = args.plan_json if args.plan_json is not None else out_dir / "sharded_matvec_benchmark_plan.json"
        _write_plan_json(plan, plan_path)
        print(json.dumps(plan, indent=2, sort_keys=True))
        return

    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.global_warmup and args.global_warmup > 0:
        for _ in range(int(args.global_warmup)):
            _run_once_subprocess(
                input_path=input_path,
                devices=1,
                nrep=args.nrep,
                cache_dir=cache_dir,
                axis=args.axis,
                pad=args.pad,
            )

    results = []
    for d in devices:
        times = []
        for _ in range(max(args.repeats, 1)):
            dt = _run_once_subprocess(
                input_path=input_path,
                devices=d,
                nrep=args.nrep,
                cache_dir=cache_dir,
                axis=args.axis,
                pad=args.pad,
            )
            times.append(dt)
        times = np.asarray(times, dtype=float)
        results.append(
            {
                "devices": d,
                "mean_s": float(np.mean(times)),
                "std_s": float(np.std(times, ddof=1)) if times.size > 1 else 0.0,
                "samples": [float(v) for v in times],
            }
        )
        print(f"devices={d} mean_s={results[-1]['mean_s']:.6f} std_s={results[-1]['std_s']:.6f}", flush=True)

    base = next((r for r in results if r["devices"] == 1), None)
    if base is not None and base["mean_s"] > 0:
        for r in results:
            r["speedup"] = float(base["mean_s"] / r["mean_s"])
    else:
        for r in results:
            r["speedup"] = None

    payload = {
        "input": input_path.name,
        "axis": args.axis,
        "pad": bool(args.pad),
        "case": input_path.stem.replace(".input", ""),
        "devices": devices,
        "results": results,
        "nrep": int(args.nrep),
    }

    json_path = out_dir / "sharded_matvec_scaling.json"
    json_path.write_text(json.dumps(payload, indent=2))

    try:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt  # noqa: PLC0415

        d = np.array([r["devices"] for r in results], dtype=int)
        mean_s = np.array([r["mean_s"] for r in results], dtype=float)
        speedup = np.array([r.get("speedup", np.nan) for r in results], dtype=float)

        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
        axes[0].plot(d, mean_s, "o-", label="measured")
        axes[0].set_xlabel("CPU devices")
        axes[0].set_ylabel("time per matvec (s)")
        axes[0].set_title("Matvec time vs devices")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(d, speedup, "o-", label="measured")
        axes[1].plot(d, d, "--", label="ideal")
        axes[1].set_xlabel("CPU devices")
        axes[1].set_ylabel("speedup")
        axes[1].set_title("Matvec speedup vs devices")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(frameon=False)

        fig.suptitle(f"Sharded matvec scaling: {payload['case']}", y=1.02)
        fig.tight_layout()
        fig_path = out_dir / "sharded_matvec_scaling.png"
        fig.savefig(fig_path, dpi=200)
        print(f"Saved figure -> {fig_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"Matplotlib unavailable: {exc}")


if __name__ == "__main__":
    main()
