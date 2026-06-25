from __future__ import annotations

from collections.abc import Callable
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

from sfincs_jax.problems.transport_matrix.parallel.policy import (
    transport_parallel_gpu_worker_env,
    transport_parallel_visible_gpu_ids,
    validate_transport_parallel_worker_count,
)
from sfincs_jax.problems.transport_matrix.parallel.validation import (
    validate_complete_transport_worker_rhs_coverage,
    validate_distinct_transport_worker_rhs,
    validate_gpu_transport_worker_arrays,
    validate_transport_worker_result_payload,
)
from sfincs_jax.problems.transport_matrix.residual_quality import (
    transport_residual_gate_failures_from_arrays,
    transport_residual_gate_thresholds_from_env,
)


_GPU_WORKER_LOG_MARKERS = (
    "whichRHS=",
    "rhs_norm=",
    "preconditioner=",
    "active-DOF",
    "host sparse",
    "sparse LU",
    "fallback",
    "retry",
    "residual_norm=",
    "elapsed_s=",
)


def summarize_transport_worker_output(text: str, *, max_lines: int = 24) -> list[str]:
    """Return the useful progress lines from a successful transport worker log."""
    selected: list[str] = []
    for raw_line in str(text).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if any(marker in line for marker in _GPU_WORKER_LOG_MARKERS):
            selected.append(line)
    if len(selected) <= int(max_lines):
        return selected
    head = max(1, int(max_lines) // 3)
    tail = max(1, int(max_lines) - head - 1)
    return [*selected[:head], "...", *selected[-tail:]]


def transport_worker_subprocess_env(base_env: dict[str, str]) -> dict[str, str]:
    """Return worker env with the source checkout importable from scan subdirs."""
    env = dict(base_env)
    file_path = Path(__file__).resolve()
    repo_path = next(
        (
            parent
            for parent in file_path.parents
            if (parent / "pyproject.toml").is_file() and (parent / "sfincs_jax").is_dir()
        ),
        file_path.parents[4],
    )
    repo_root = str(repo_path)
    existing = env.get("PYTHONPATH", "").strip()
    parts = [repo_root]
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def _quality_failure_from_worker_result(output_path: Path) -> str | None:
    max_abs, max_rel = transport_residual_gate_thresholds_from_env()
    if max_abs <= 0.0 and max_rel <= 0.0:
        return None
    if not output_path.exists():
        return None
    with np.load(output_path) as data:
        rhs_values = np.asarray(data["which_rhs_values"], dtype=np.int32)
        residual_norms = np.asarray(data["residual_norms"], dtype=np.float64)
        rhs_norms = (
            np.asarray(data["rhs_norms"], dtype=np.float64)
            if "rhs_norms" in data.files
            else np.full_like(residual_norms, np.nan, dtype=np.float64)
        )
    failures = transport_residual_gate_failures_from_arrays(
        which_rhs_values=rhs_values,
        residual_norms=residual_norms,
        rhs_norms=rhs_norms,
        max_abs=float(max_abs),
        max_relative=float(max_rel),
    )
    if not failures:
        return None
    return "; ".join(failures)


def _quality_failure_from_worker_text(stdout: str, stderr: str) -> str | None:
    text = "\n".join([str(stdout), str(stderr)])
    marker = "transport residual gate failed:"
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if marker in line:
            return line.split(marker, 1)[1].strip()
    return None


def _terminate_pending_workers(
    procs: list[tuple[subprocess.Popen[str], Path, list[int], str]],
    pending: set[int],
) -> None:
    for idx in list(pending):
        proc = procs[idx][0]
        if proc.poll() is None:
            proc.terminate()
    deadline = time.perf_counter() + 5.0
    for idx in list(pending):
        proc = procs[idx][0]
        while proc.poll() is None and time.perf_counter() < deadline:
            time.sleep(0.1)
        if proc.poll() is None:
            proc.kill()


def partition_transport_rhs(values: list[int], workers: int) -> list[list[int]]:
    worker_count = validate_transport_parallel_worker_count(workers)
    chunks: list[list[int]] = [[] for _ in range(worker_count)]
    for i, val in enumerate(values):
        chunks[i % len(chunks)].append(int(val))
    return [chunk for chunk in chunks if chunk]


def _unique_gpu_ids(gpu_ids: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for raw_id in gpu_ids:
        gpu_id = str(raw_id).strip()
        if gpu_id and gpu_id not in seen:
            unique.append(gpu_id)
            seen.add(gpu_id)
    return unique


def _coalesce_transport_payloads(
    payloads: list[dict[str, object]],
    workers: int,
) -> list[dict[str, object]]:
    if int(workers) <= 0:
        return []
    scheduled = [dict(payload) for payload in payloads[: int(workers)]]
    for i, payload in enumerate(payloads[int(workers) :], start=int(workers)):
        target = i % int(workers)
        existing_rhs = [int(v) for v in scheduled[target].get("which_rhs_values", [])]
        extra_rhs = [int(v) for v in payload.get("which_rhs_values", [])]
        scheduled[target]["which_rhs_values"] = [*existing_rhs, *extra_rhs]
    return scheduled


def plan_transport_parallel_gpu_subprocesses(
    *,
    payloads: list[dict[str, object]],
    parallel_workers: int,
    visible_gpu_ids: list[str],
) -> dict[str, object]:
    """Return a deterministic GPU transport worker schedule without launching it."""

    requested_workers = validate_transport_parallel_worker_count(
        parallel_workers,
        context="GPU transport",
    )
    if not payloads:
        return {
            "requested_workers": requested_workers,
            "active_workers": 0,
            "raw_visible_gpu_ids": [str(gpu_id) for gpu_id in visible_gpu_ids],
            "unique_visible_gpu_ids": [],
            "capped": False,
            "cap_reasons": [],
            "worker_assignments": [],
        }

    gpu_ids = _unique_gpu_ids(visible_gpu_ids)
    if not gpu_ids:
        raise RuntimeError("GPU transport parallel backend requested but no visible GPU ids were found.")

    unique_gpu_count = len(gpu_ids)
    use_workers = min(requested_workers, len(payloads), unique_gpu_count)
    cap_reasons: list[str] = []
    if len(payloads) < requested_workers:
        cap_reasons.append(f"independent RHS chunks={len(payloads)}")
    if unique_gpu_count < requested_workers:
        cap_reasons.append(f"unique visible GPU ids={unique_gpu_count}")

    scheduled_payloads = _coalesce_transport_payloads(payloads, use_workers)
    assignments: list[dict[str, object]] = []
    for i, payload in enumerate(scheduled_payloads):
        rhs_values = [int(v) for v in payload.get("which_rhs_values", [])]
        assignments.append(
            {
                "worker_index": i,
                "gpu_id": gpu_ids[i],
                "which_rhs_values": rhs_values,
                "payload": dict(payload),
            }
        )

    return {
        "requested_workers": requested_workers,
        "active_workers": use_workers,
        "raw_visible_gpu_ids": [str(gpu_id) for gpu_id in visible_gpu_ids],
        "unique_visible_gpu_ids": gpu_ids[:use_workers],
        "capped": use_workers < requested_workers,
        "cap_reasons": cap_reasons,
        "worker_assignments": assignments,
    }


def run_transport_parallel_gpu_subprocesses(
    *,
    payloads: list[dict[str, object]],
    parallel_workers: int,
    visible_gpu_ids: Callable[[int], list[str]],
    gpu_worker_env: Callable[..., dict[str, str]],
    emit: Callable[[int, str], None] | None = None,
) -> list[dict[str, object]]:
    requested_workers = validate_transport_parallel_worker_count(
        parallel_workers,
        context="GPU transport",
    )
    if not payloads:
        return []
    plan = plan_transport_parallel_gpu_subprocesses(
        payloads=payloads,
        parallel_workers=requested_workers,
        visible_gpu_ids=visible_gpu_ids(requested_workers),
    )
    use_workers = int(plan["active_workers"])
    assignments = list(plan["worker_assignments"])  # type: ignore[arg-type]
    if emit is not None and bool(plan["capped"]):
        cap_reasons = [str(reason) for reason in plan["cap_reasons"]]  # type: ignore[union-attr]
        reason = ", ".join(cap_reasons) if cap_reasons else "available work/devices"
        emit(
            1,
            "solve_v3_transport_matrix_linear_gmres: GPU transport worker plan capped "
            f"(active={use_workers} requested={requested_workers}; {reason})",
        )

    results: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="sfincs_jax_transport_gpu_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        procs: list[tuple[subprocess.Popen[str], Path, list[int], str]] = []
        for i, assignment in enumerate(assignments):
            payload = dict(assignment["payload"])  # type: ignore[index]
            rhs_vals = [int(v) for v in payload.get("which_rhs_values", [])]
            payload_path = tmpdir_path / f"payload_{i}.json"
            output_path = tmpdir_path / f"result_{i}.npz"
            payload_path.write_text(json.dumps(payload))
            gpu_id = str(assignment["gpu_id"])  # type: ignore[index]
            env = gpu_worker_env(gpu_id=gpu_id)
            cmd = [
                sys.executable,
                "-m",
                "sfincs_jax.problems.transport_matrix.parallel.worker",
                "--payload",
                str(payload_path),
                "--output",
                str(output_path),
            ]
            env = transport_worker_subprocess_env(env)
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            procs.append((proc, output_path, rhs_vals, gpu_id))

        interval_env = os.environ.get("SFINCS_JAX_TRANSPORT_PARALLEL_STATUS_INTERVAL", "").strip()
        try:
            status_interval_s = float(interval_env) if interval_env else 30.0
        except ValueError:
            status_interval_s = 30.0
        started = time.perf_counter()
        last_status = started
        pending = set(range(len(procs)))
        completed: dict[int, tuple[str, str]] = {}
        while pending:
            for idx in list(pending):
                proc, output_path, rhs_vals, gpu_id = procs[idx]
                if proc.poll() is None:
                    continue
                out, err = proc.communicate()
                completed[idx] = (out, err)
                pending.remove(idx)
                if emit is not None:
                    emit(
                        0,
                        "solve_v3_transport_matrix_linear_gmres: GPU transport worker done "
                        f"(gpu={gpu_id} whichRHS={rhs_vals} elapsed={time.perf_counter() - started:.1f}s)",
                    )
                if proc.returncode == 0:
                    quality_failure = _quality_failure_from_worker_result(output_path)
                    if quality_failure is not None:
                        if emit is not None:
                            emit(
                                1,
                                "solve_v3_transport_matrix_linear_gmres: GPU transport worker "
                                "residual gate failed; terminating remaining workers "
                                f"({quality_failure})",
                            )
                        _terminate_pending_workers(procs, pending)
                        raise RuntimeError(
                            "GPU transport worker residual gate failed: "
                            f"gpu={gpu_id} whichRHS={rhs_vals}: {quality_failure}"
                        )
                else:
                    quality_failure = _quality_failure_from_worker_text(out, err)
                    if quality_failure is not None:
                        if emit is not None:
                            emit(
                                1,
                                "solve_v3_transport_matrix_linear_gmres: GPU transport worker "
                                "residual gate failed; terminating remaining workers "
                                f"({quality_failure})",
                            )
                        _terminate_pending_workers(procs, pending)
                        raise RuntimeError(
                            "GPU transport worker residual gate failed: "
                            f"gpu={gpu_id} whichRHS={rhs_vals}: {quality_failure}"
                        )
            if pending:
                now = time.perf_counter()
                if emit is not None and status_interval_s > 0.0 and now - last_status >= status_interval_s:
                    running = ", ".join(
                        f"gpu={procs[idx][3]} rhs={procs[idx][2]}" for idx in sorted(pending)
                    )
                    emit(
                        0,
                        "solve_v3_transport_matrix_linear_gmres: GPU transport workers running "
                        f"({running}; elapsed={now - started:.1f}s)",
                    )
                    last_status = now
                time.sleep(0.5)

        for idx, (proc, output_path, rhs_vals, gpu_id) in enumerate(procs):
            out, err = completed.get(idx, ("", ""))
            if proc.returncode != 0:
                raise RuntimeError(
                    "GPU transport worker failed "
                    f"(gpu={gpu_id} whichRHS={rhs_vals} code={proc.returncode})\n"
                    f"stdout:\n{out}\n"
                    f"stderr:\n{err}"
                )
            with np.load(output_path) as data:
                rhs_values = [int(v) for v in np.asarray(data["which_rhs_values"], dtype=np.int32)]
                state_vectors = np.asarray(data["state_vectors"], dtype=np.float64)
                residual_norms = np.asarray(data["residual_norms"], dtype=np.float64)
                if "rhs_norms" in data.files:
                    rhs_norms = np.asarray(data["rhs_norms"], dtype=np.float64)
                else:
                    rhs_norms = np.full_like(residual_norms, np.nan, dtype=np.float64)
                elapsed_time_s = np.asarray(data["elapsed_time_s"], dtype=np.float64)
            validate_gpu_transport_worker_arrays(
                requested_rhs_values=rhs_vals,
                output_rhs_values=rhs_values,
                state_vectors=state_vectors,
                residual_norms=residual_norms,
                rhs_norms=rhs_norms,
                elapsed_time_s=elapsed_time_s,
                gpu_id=gpu_id,
            )
            if emit is not None:
                for line in summarize_transport_worker_output(out):
                    emit(
                        1,
                        "solve_v3_transport_matrix_linear_gmres: GPU transport worker log "
                        f"(gpu={gpu_id} whichRHS={rhs_vals}): {line}",
                    )
                for rhs_value, residual_norm, rhs_norm, elapsed in zip(
                    rhs_values,
                    residual_norms,
                    rhs_norms,
                    elapsed_time_s,
                    strict=False,
                ):
                    rel = (
                        float(residual_norm) / float(rhs_norm)
                        if np.isfinite(float(rhs_norm)) and float(rhs_norm) > 0.0
                        else float("nan")
                    )
                    emit(
                        0,
                        "solve_v3_transport_matrix_linear_gmres: GPU transport worker result "
                        f"(gpu={gpu_id} whichRHS={int(rhs_value)} residual_norm={float(residual_norm):.6e} "
                        f"rhs_norm={float(rhs_norm):.6e} relative_residual={rel:.6e} "
                        f"elapsed_s={float(elapsed):.3f})",
                    )
            results.append(
                {
                    "which_rhs_values": rhs_values,
                    "state_vectors_by_rhs": {
                        int(rhs): state_vectors[idx] for idx, rhs in enumerate(rhs_values)
                    },
                    "residual_norms_by_rhs": {
                        int(rhs): float(residual_norms[idx]) for idx, rhs in enumerate(rhs_values)
                    },
                    "rhs_norms_by_rhs": {
                        int(rhs): float(rhs_norms[idx]) for idx, rhs in enumerate(rhs_values)
                    },
                    "elapsed_time_s": elapsed_time_s,
                }
            )
    return results


def run_transport_parallel_gpu_subprocesses_with_policy(
    *,
    payloads: list[dict[str, object]],
    parallel_workers: int,
    emit: Callable[[int, str], None] | None = None,
) -> list[dict[str, object]]:
    """Run GPU transport workers using the standard environment policy."""
    return run_transport_parallel_gpu_subprocesses(
        payloads=payloads,
        parallel_workers=int(parallel_workers),
        visible_gpu_ids=transport_parallel_visible_gpu_ids,
        gpu_worker_env=transport_parallel_gpu_worker_env,
        emit=emit,
    )


def merge_transport_parallel_results(
    *,
    n_rhs: int,
    results: list[dict[str, object]],
    require_complete_coverage: bool = False,
) -> tuple[dict[int, np.ndarray], dict[int, float], dict[int, float], np.ndarray]:
    state_vectors: dict[int, np.ndarray] = {}
    residual_norms: dict[int, float] = {}
    rhs_norms: dict[int, float] = {}
    elapsed_s = np.zeros((int(n_rhs),), dtype=np.float64)
    seen_rhs: set[int] = set()
    for res in results:
        rhs_vals = [int(v) for v in res.get("which_rhs_values", [])]
        validate_distinct_transport_worker_rhs(rhs_values=rhs_vals, seen_rhs=seen_rhs)
        validate_transport_worker_result_payload(rhs_values=rhs_vals, result=res, n_rhs=int(n_rhs))
        seen_rhs.update(rhs_vals)
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
        rhs_norms.update({int(k): float(v) for k, v in res.get("rhs_norms_by_rhs", {}).items()})
        state_vectors.update({int(k): np.asarray(v, dtype=np.float64) for k, v in res.get("state_vectors_by_rhs", {}).items()})
    if require_complete_coverage:
        validate_complete_transport_worker_rhs_coverage(seen_rhs=seen_rhs, n_rhs=int(n_rhs))
    return state_vectors, residual_norms, rhs_norms, elapsed_s
