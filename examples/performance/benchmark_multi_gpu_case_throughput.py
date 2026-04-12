from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _base_env(cache_dir: Path | None, rhs1_precond: str, coarse_levels: int | None) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
    env["SFINCS_JAX_MATVEC_SHARD_AXIS"] = "theta"
    env["SFINCS_JAX_GMRES_DISTRIBUTED"] = "0"
    env["SFINCS_JAX_DISTRIBUTED_KRYLOV"] = "auto"
    env["SFINCS_JAX_AUTO_SHARD"] = "0"
    env["SFINCS_JAX_IMPLICIT_SOLVE"] = "1"
    env["SFINCS_JAX_SHARD_PAD"] = "1"
    env["SFINCS_JAX_FORTRAN_STDOUT"] = "0"
    env["SFINCS_JAX_SOLVER_ITER_STATS"] = "0"
    env["SFINCS_JAX_RHSMODE1_PRECONDITIONER"] = rhs1_precond
    if coarse_levels is None:
        env.pop("SFINCS_JAX_RHSMODE1_SCHWARZ_COARSE_LEVELS", None)
    else:
        env["SFINCS_JAX_RHSMODE1_SCHWARZ_COARSE_LEVELS"] = str(int(coarse_levels))
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        env["JAX_COMPILATION_CACHE_DIR"] = str(cache_dir)
    return env


def _run_case_once(*, input_path: Path, visible_devices: str, nsolve: int, env: dict[str, str]) -> float:
    local_env = dict(env)
    local_env["CUDA_VISIBLE_DEVICES"] = visible_devices
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("benchmark_sharded_solve_scaling.py")),
        "--run-once",
        "--input",
        str(input_path),
        "--nsolve",
        str(int(nsolve)),
    ]
    out = subprocess.check_output(cmd, env=local_env, text=True)
    return float(out.strip().splitlines()[-1])


def _warm_gpu(*, input_path: Path, gpu_id: int, nsolve: int, env: dict[str, str]) -> None:
    _run_case_once(
        input_path=input_path,
        visible_devices=str(int(gpu_id)),
        nsolve=nsolve,
        env=env,
    )


def _sequential_two_cases(*, input_path: Path, nsolve: int, env: dict[str, str]) -> dict[str, float]:
    t0 = time.perf_counter()
    dt0 = _run_case_once(input_path=input_path, visible_devices="0", nsolve=nsolve, env=env)
    dt1 = _run_case_once(input_path=input_path, visible_devices="0", nsolve=nsolve, env=env)
    wall = time.perf_counter() - t0
    return {
        "case0_s": float(dt0),
        "case1_s": float(dt1),
        "wall_s": float(wall),
    }


def _parallel_two_cases(*, input_path: Path, nsolve: int, env: dict[str, str]) -> dict[str, float]:
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("benchmark_sharded_solve_scaling.py")),
        "--run-once",
        "--input",
        str(input_path),
        "--nsolve",
        str(int(nsolve)),
    ]
    env0 = dict(env)
    env1 = dict(env)
    env0["CUDA_VISIBLE_DEVICES"] = "0"
    env1["CUDA_VISIBLE_DEVICES"] = "1"
    t0 = time.perf_counter()
    p0 = subprocess.Popen(cmd, env=env0, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    p1 = subprocess.Popen(cmd, env=env1, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out0, err0 = p0.communicate()
    out1, err1 = p1.communicate()
    wall = time.perf_counter() - t0
    if p0.returncode != 0:
        raise RuntimeError(f"GPU0 throughput run failed: {err0.strip()}")
    if p1.returncode != 0:
        raise RuntimeError(f"GPU1 throughput run failed: {err1.strip()}")
    dt0 = float(out0.strip().splitlines()[-1])
    dt1 = float(out1.strip().splitlines()[-1])
    return {
        "case0_s": float(dt0),
        "case1_s": float(dt1),
        "wall_s": float(wall),
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Benchmark one-GPU-per-case throughput on a 2-GPU node.")
    parser.add_argument(
        "--input",
        type=Path,
        default=repo_root / "examples" / "performance" / "rhsmode1_sharded_scaling.input.namelist",
        help="RHSMode=1 input.namelist to run twice.",
    )
    parser.add_argument(
        "--nsolve",
        type=int,
        default=4,
        help="Number of solves per timed case.",
    )
    parser.add_argument(
        "--rhs1-precond",
        type=str,
        default="theta_schwarz",
        help="Explicit RHSMode=1 preconditioner for the benchmark.",
    )
    parser.add_argument(
        "--schwarz-coarse-levels",
        type=int,
        default=2,
        help="Explicit multilevel Schwarz setting for the benchmark.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root / "examples" / "performance" / "output" / "gpu_case_throughput",
        help="Output directory for JSON and figure.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=repo_root / "examples" / "performance" / "output" / "gpu_case_throughput" / "jax_cache",
        help="Persistent JAX cache directory.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(str(args.input))

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    env = _base_env(args.cache_dir, args.rhs1_precond, args.schwarz_coarse_levels)

    # Warm both devices individually before timing steady-state throughput.
    _warm_gpu(input_path=args.input, gpu_id=0, nsolve=args.nsolve, env=env)
    _warm_gpu(input_path=args.input, gpu_id=1, nsolve=args.nsolve, env=env)

    sequential = _sequential_two_cases(input_path=args.input, nsolve=args.nsolve, env=env)
    parallel = _parallel_two_cases(input_path=args.input, nsolve=args.nsolve, env=env)
    speedup = sequential["wall_s"] / parallel["wall_s"] if parallel["wall_s"] > 0 else float("nan")

    payload = {
        "input": args.input.name,
        "case": args.input.stem.replace(".input", ""),
        "nsolve": int(args.nsolve),
        "rhs1_precond": str(args.rhs1_precond),
        "schwarz_coarse_levels": int(args.schwarz_coarse_levels),
        "sequential_one_gpu": sequential,
        "parallel_two_gpu": parallel,
        "throughput_speedup": float(speedup),
    }
    json_path = out_dir / "gpu_case_throughput.json"
    json_path.write_text(json.dumps(payload, indent=2))

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import numpy as np

        labels = ["1 GPU sequential\n(2 cases)", "2 GPUs parallel\n(2 cases)"]
        wall = np.array([sequential["wall_s"], parallel["wall_s"]], dtype=float)
        cases = np.array(
            [
                [sequential["case0_s"], sequential["case1_s"]],
                [parallel["case0_s"], parallel["case1_s"]],
            ],
            dtype=float,
        )

        fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3), constrained_layout=True)
        axes[0].bar(labels, wall, color=["#1D4ED8", "#0F766E"])
        axes[0].set_ylabel("wall time (s)")
        axes[0].set_title("Two-case GPU throughput")
        axes[0].grid(True, axis="y", alpha=0.25)

        x = np.arange(len(labels))
        width = 0.32
        axes[1].bar(x - width / 2, cases[:, 0], width=width, label="case 0", color="#7C3AED")
        axes[1].bar(x + width / 2, cases[:, 1], width=width, label="case 1", color="#DC2626")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels)
        axes[1].set_ylabel("per-case solve time (s)")
        axes[1].set_title(f"Throughput speedup = {speedup:.2f}x")
        axes[1].legend(frameon=False)
        axes[1].grid(True, axis="y", alpha=0.25)

        fig.suptitle(f"One-GPU-per-case throughput: {payload['case']}", y=1.03)
        fig_path = out_dir / "gpu_case_throughput.png"
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        print(f"Saved figure -> {fig_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"Matplotlib unavailable: {exc}")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
