from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


def partition_transport_rhs(values: list[int], workers: int) -> list[list[int]]:
    chunks: list[list[int]] = [[] for _ in range(max(1, int(workers)))]
    for i, val in enumerate(values):
        chunks[i % len(chunks)].append(int(val))
    return [chunk for chunk in chunks if chunk]


def run_transport_parallel_gpu_subprocesses(
    *,
    payloads: list[dict[str, object]],
    parallel_workers: int,
    visible_gpu_ids: Callable[[int], list[str]],
    gpu_worker_env: Callable[..., dict[str, str]],
    emit: Callable[[int, str], None] | None = None,
) -> list[dict[str, object]]:
    gpu_ids = visible_gpu_ids(int(parallel_workers))
    if not gpu_ids:
        raise RuntimeError("GPU transport parallel backend requested but no visible GPU ids were found.")
    use_workers = min(int(parallel_workers), len(payloads), len(gpu_ids))
    gpu_ids = gpu_ids[:use_workers]
    if emit is not None and use_workers < int(parallel_workers):
        emit(
            1,
            "solve_v3_transport_matrix_linear_gmres: GPU transport workers capped by visible devices "
            f"({use_workers}/{int(parallel_workers)})",
        )

    results: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="sfincs_jax_transport_gpu_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        procs: list[tuple[subprocess.Popen[str], Path, list[int], str]] = []
        for i, payload in enumerate(payloads[:use_workers]):
            rhs_vals = [int(v) for v in payload.get("which_rhs_values", [])]
            payload_path = tmpdir_path / f"payload_{i}.json"
            output_path = tmpdir_path / f"result_{i}.npz"
            payload_path.write_text(json.dumps(payload))
            gpu_id = gpu_ids[i]
            env = gpu_worker_env(gpu_id=gpu_id)
            cmd = [
                sys.executable,
                "-m",
                "sfincs_jax.transport_parallel_worker",
                "--payload",
                str(payload_path),
                "--output",
                str(output_path),
            ]
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            procs.append((proc, output_path, rhs_vals, gpu_id))

        for proc, output_path, rhs_vals, gpu_id in procs:
            out, err = proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(
                    "GPU transport worker failed "
                    f"(gpu={gpu_id} whichRHS={rhs_vals} code={proc.returncode})\n"
                    f"stdout:\n{out}\n"
                    f"stderr:\n{err}"
                )
            data = np.load(output_path)
            rhs_values = [int(v) for v in np.asarray(data["which_rhs_values"], dtype=np.int32)]
            state_vectors = np.asarray(data["state_vectors"], dtype=np.float64)
            residual_norms = np.asarray(data["residual_norms"], dtype=np.float64)
            elapsed_time_s = np.asarray(data["elapsed_time_s"], dtype=np.float64)
            results.append(
                {
                    "which_rhs_values": rhs_values,
                    "state_vectors_by_rhs": {
                        int(rhs): state_vectors[idx] for idx, rhs in enumerate(rhs_values)
                    },
                    "residual_norms_by_rhs": {
                        int(rhs): float(residual_norms[idx]) for idx, rhs in enumerate(rhs_values)
                    },
                    "elapsed_time_s": elapsed_time_s,
                }
            )
    return results


def merge_transport_parallel_results(
    *,
    n_rhs: int,
    results: list[dict[str, object]],
) -> tuple[dict[int, np.ndarray], dict[int, float], np.ndarray]:
    state_vectors: dict[int, np.ndarray] = {}
    residual_norms: dict[int, float] = {}
    elapsed_s = np.zeros((int(n_rhs),), dtype=np.float64)
    for res in results:
        rhs_vals = [int(v) for v in res.get("which_rhs_values", [])]
        idxs = [v - 1 for v in rhs_vals]
        elapsed_chunk = np.asarray(res.get("elapsed_time_s", np.zeros((n_rhs,))), dtype=np.float64)
        if elapsed_chunk.ndim == 0:
            if idxs:
                elapsed_s[idxs[0]] = float(elapsed_chunk)
        elif elapsed_chunk.shape[0] == len(rhs_vals):
            elapsed_s[idxs] = elapsed_chunk
        elif elapsed_chunk.shape[0] > max(idxs, default=-1):
            elapsed_s[idxs] = elapsed_chunk[idxs]
        else:
            count = min(len(rhs_vals), int(elapsed_chunk.shape[0]))
            if count > 0:
                elapsed_s[idxs[:count]] = elapsed_chunk[:count]
        residual_norms.update({int(k): float(v) for k, v in res.get("residual_norms_by_rhs", {}).items()})
        state_vectors.update({int(k): np.asarray(v, dtype=np.float64) for k, v in res.get("state_vectors_by_rhs", {}).items()})
    return state_vectors, residual_norms, elapsed_s
