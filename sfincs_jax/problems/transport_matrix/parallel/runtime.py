"""Runtime orchestration for transport-matrix parallel worker solves."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import atexit
import concurrent.futures
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

import jax.numpy as jnp
import numpy as np

from sfincs_jax.namelist import Namelist
from sfincs_jax.problems.transport_matrix.diagnostics import (
    v3_transport_matrix_from_flux_arrays,
    v3_transport_output_fields_vm_only,
)
from sfincs_jax.problems.transport_matrix.parallel.policy import (
    rewrite_xla_flags,
    transport_parallel_gpu_worker_env,
    transport_parallel_pool_executor_kwargs as _transport_parallel_pool_executor_kwargs,
    transport_parallel_pool_key,
    transport_parallel_visible_gpu_ids,
    transport_parallel_worker_env as _transport_parallel_worker_env,
    validate_transport_parallel_worker_count,
)
from sfincs_jax.problems.transport_matrix.policies import (
    transport_residual_gate_failures_from_arrays,
    transport_residual_gate_thresholds_from_env,
)
from sfincs_jax.problems.transport_matrix.finalize import V3TransportMatrixSolveResult
from sfincs_jax.discretization.v3 import geometry_from_namelist, grids_from_namelist
from sfincs_jax.operators.profile_response.system import V3FullSystemOperator


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
    """Split ``whichRHS`` values into deterministic round-robin worker chunks."""

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
    """Launch one-GPU-per-worker transport subprocesses and collect payloads."""

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
    """Merge worker result dictionaries into per-``whichRHS`` arrays."""

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

# Worker-result validation helpers.
def format_transport_rhs_list(values: Sequence[int]) -> str:
    """Format ``whichRHS`` values for validation errors."""

    return "[" + ", ".join(str(v) for v in values) + "]"


def validate_distinct_transport_worker_rhs(
    *,
    rhs_values: Sequence[int],
    seen_rhs: set[int],
) -> None:
    """Reject duplicate ``whichRHS`` values across worker results."""

    duplicate_rhs = [int(rhs) for rhs in rhs_values if int(rhs) in seen_rhs]
    if duplicate_rhs:
        raise ValueError(
            "transport parallel worker results contain duplicate whichRHS values "
            f"{format_transport_rhs_list(duplicate_rhs)}"
        )


def validate_transport_worker_result_payload(
    *,
    rhs_values: Sequence[int],
    result: Mapping[str, object],
    n_rhs: int | None,
) -> None:
    """Validate one merge-ready worker result payload."""

    rhs_values = [int(rhs) for rhs in rhs_values]
    if any(rhs < 1 for rhs in rhs_values):
        invalid = [rhs for rhs in rhs_values if rhs < 1]
        raise ValueError(
            "transport parallel worker reported invalid whichRHS values "
            f"{format_transport_rhs_list(invalid)}"
        )
    if n_rhs is not None and any(rhs > int(n_rhs) for rhs in rhs_values):
        invalid = [rhs for rhs in rhs_values if rhs > int(n_rhs)]
        raise ValueError(
            "transport parallel worker reported out-of-range whichRHS values "
            f"{format_transport_rhs_list(invalid)} for n_rhs={int(n_rhs)}"
        )

    required_maps = ("state_vectors_by_rhs", "residual_norms_by_rhs", "rhs_norms_by_rhs")
    for key in required_maps:
        value = result.get(key, {})
        if not isinstance(value, dict):
            raise ValueError(f"transport parallel worker result field {key!r} must be a mapping")
        present = {int(k) for k in value}
        missing = [rhs for rhs in rhs_values if rhs not in present]
        if missing:
            raise ValueError(
                "transport parallel worker result is missing "
                f"{key} entries for whichRHS={format_transport_rhs_list(missing)}"
            )


def validate_complete_transport_worker_rhs_coverage(
    *,
    seen_rhs: set[int],
    n_rhs: int,
) -> None:
    """Require worker results to cover every transport right-hand side once."""

    expected = set(range(1, int(n_rhs) + 1))
    missing = sorted(expected - set(seen_rhs))
    extra = sorted(set(seen_rhs) - expected)
    if missing:
        raise ValueError(
            "transport parallel worker results are missing whichRHS values "
            f"{format_transport_rhs_list(missing)}"
        )
    if extra:
        raise ValueError(
            "transport parallel worker results contain out-of-range whichRHS values "
            f"{format_transport_rhs_list(extra)} for n_rhs={int(n_rhs)}"
        )


def validate_gpu_transport_worker_arrays(
    *,
    requested_rhs_values: Sequence[int],
    output_rhs_values: Sequence[int],
    state_vectors: np.ndarray,
    residual_norms: np.ndarray,
    rhs_norms: np.ndarray,
    elapsed_time_s: np.ndarray,
    gpu_id: str,
) -> None:
    """Validate NPZ arrays emitted by a GPU transport worker subprocess."""

    requested_rhs_values = [int(rhs) for rhs in requested_rhs_values]
    output_rhs_values = [int(rhs) for rhs in output_rhs_values]
    if output_rhs_values != requested_rhs_values:
        raise RuntimeError(
            "GPU transport worker returned unexpected whichRHS coverage "
            f"(gpu={gpu_id} requested={requested_rhs_values} returned={output_rhs_values})"
        )
    expected = len(output_rhs_values)
    lengths = {
        "state_vectors": int(state_vectors.shape[0]) if state_vectors.ndim > 0 else 0,
        "residual_norms": int(residual_norms.shape[0]) if residual_norms.ndim > 0 else 0,
        "rhs_norms": int(rhs_norms.shape[0]) if rhs_norms.ndim > 0 else 0,
        "elapsed_time_s": int(elapsed_time_s.shape[0]) if elapsed_time_s.ndim > 0 else 0,
    }
    bad_lengths = {key: length for key, length in lengths.items() if length != expected}
    if bad_lengths:
        details = ", ".join(f"{key}={length}" for key, length in sorted(bad_lengths.items()))
        raise RuntimeError(
            "GPU transport worker returned inconsistent result array lengths "
            f"(gpu={gpu_id} whichRHS={output_rhs_values} expected={expected}; {details})"
        )


# Worker payload execution helpers.
def solve_transport_parallel_payload(
    payload: dict[str, object],
    *,
    read_input: Callable[[Path], object],
    solve_transport: Callable[..., object],
    emit: Callable[[int, str], None] | None = None,
    set_child_environment: bool = True,
) -> dict[str, object]:
    """Run one transport worker payload and return the merge-ready result dict."""
    input_path = Path(str(payload["input_path"]))
    which_rhs_values = [int(v) for v in payload["which_rhs_values"]]  # type: ignore[assignment]
    tol = float(payload.get("tol", 1e-10))
    atol = float(payload.get("atol", 0.0))
    restart = int(payload.get("restart", 80))
    maxiter = payload.get("maxiter")
    solve_method = str(payload.get("solve_method", "auto"))
    identity_shift = float(payload.get("identity_shift", 0.0))
    differentiable_payload = payload.get("differentiable", None)
    if differentiable_payload is not None:
        differentiable_payload = bool(differentiable_payload)
    phi1_hat_base = payload.get("phi1_hat_base")
    if phi1_hat_base is not None:
        phi1_hat_base = jnp.asarray(phi1_hat_base, dtype=jnp.float64)

    if set_child_environment:
        # Prevent recursive process/GPU worker launches from inside workers.
        os.environ["SFINCS_JAX_TRANSPORT_PARALLEL"] = "off"
        os.environ["SFINCS_JAX_TRANSPORT_PARALLEL_CHILD"] = "1"

    solve_kwargs: dict[str, Any] = {
        "nml": read_input(input_path),
        "tol": tol,
        "atol": atol,
        "restart": restart,
        "maxiter": maxiter,
        "solve_method": solve_method,
        "identity_shift": identity_shift,
        "phi1_hat_base": phi1_hat_base,
        "differentiable": differentiable_payload,
        "input_namelist": input_path,
        "which_rhs_values": which_rhs_values,
        "force_stream_diagnostics": True,
        "force_store_state": True,
        "collect_transport_output_fields": False,
        "parallel_workers": 1,
    }
    if emit is not None:
        solve_kwargs["emit"] = emit
    result = solve_transport(**solve_kwargs)
    return pack_transport_parallel_result(which_rhs_values=which_rhs_values, result=result)


def pack_transport_parallel_result(
    *,
    which_rhs_values: list[int],
    result: object,
) -> dict[str, object]:
    """Convert a transport solve result object to the parent merge payload."""
    return {
        "which_rhs_values": [int(v) for v in which_rhs_values],
        "state_vectors_by_rhs": {
            int(k): np.asarray(v) for k, v in result.state_vectors_by_rhs.items()
        },
        "residual_norms_by_rhs": {
            int(k): float(np.asarray(v)) for k, v in result.residual_norms_by_rhs.items()
        },
        "rhs_norms_by_rhs": {
            int(k): float(np.asarray(v))
            for k, v in (getattr(result, "rhs_norms_by_rhs", None) or {}).items()
        },
        "elapsed_time_s": np.asarray(result.elapsed_time_s, dtype=np.float64),
    }


def transport_parallel_result_to_npz_arrays(result: dict[str, object]) -> dict[str, np.ndarray]:
    """Convert a merge-ready worker result into the subprocess NPZ schema."""
    rhs_values = np.asarray(result.get("which_rhs_values", []), dtype=np.int32)
    if rhs_values.size == 0:
        return {
            "which_rhs_values": rhs_values,
            "state_vectors": np.zeros((0, 0), dtype=np.float64),
            "residual_norms": np.zeros((0,), dtype=np.float64),
            "rhs_norms": np.zeros((0,), dtype=np.float64),
            "elapsed_time_s": np.zeros((0,), dtype=np.float64),
        }

    state_vectors_by_rhs = result.get("state_vectors_by_rhs", {})
    residual_norms_by_rhs = result.get("residual_norms_by_rhs", {})
    rhs_norms_by_rhs = result.get("rhs_norms_by_rhs", {})
    state_vectors = np.stack(
        [np.asarray(state_vectors_by_rhs[int(rhs)], dtype=np.float64) for rhs in rhs_values],
        axis=0,
    )
    residual_norms = np.asarray(
        [float(np.asarray(residual_norms_by_rhs[int(rhs)], dtype=np.float64)) for rhs in rhs_values],
        dtype=np.float64,
    )
    rhs_norms = np.asarray(
        [
            float(np.asarray(rhs_norms_by_rhs.get(int(rhs), np.nan), dtype=np.float64))
            for rhs in rhs_values
        ],
        dtype=np.float64,
    )
    elapsed_full = np.asarray(result.get("elapsed_time_s", np.zeros((0,))), dtype=np.float64)
    if elapsed_full.ndim == 0:
        elapsed_time_s = np.full((rhs_values.size,), float(elapsed_full), dtype=np.float64)
    elif elapsed_full.shape[0] == rhs_values.size:
        elapsed_time_s = elapsed_full.astype(np.float64, copy=False)
    elif elapsed_full.shape[0] > int(np.max(rhs_values - 1)):
        elapsed_time_s = elapsed_full[rhs_values - 1].astype(np.float64, copy=False)
    else:
        elapsed_time_s = np.zeros((rhs_values.size,), dtype=np.float64)
        count = min(rhs_values.size, int(elapsed_full.shape[0]))
        if count > 0:
            elapsed_time_s[:count] = elapsed_full[:count]
    return {
        "which_rhs_values": rhs_values,
        "state_vectors": state_vectors,
        "residual_norms": residual_norms,
        "rhs_norms": rhs_norms,
        "elapsed_time_s": elapsed_time_s,
    }


# Parent-side execution policy helpers.
def should_run_transport_parallel(
    *,
    parallel_child: bool,
    parallel_workers: int,
    which_rhs_values: Sequence[int],
    input_namelist: Path | None,
) -> bool:
    """Return whether the parent should launch parallel transport workers."""

    return (
        (not bool(parallel_child))
        and int(parallel_workers) > 1
        and len(which_rhs_values) > 1
        and (input_namelist is not None)
    )


def build_transport_parallel_payloads(
    *,
    chunks: Sequence[Sequence[int]],
    input_namelist: Path,
    tol: float,
    atol: float,
    restart: int,
    maxiter: int | None,
    solve_method: str,
    identity_shift: float,
    collect_transport_output_fields: bool,
    phi1_hat_base,
    differentiable: bool | None,
) -> list[dict[str, object]]:
    """Build JSON-like payloads for transport worker chunks."""

    phi1_payload = np.asarray(phi1_hat_base) if phi1_hat_base is not None else None
    payloads: list[dict[str, object]] = []
    for chunk in chunks:
        payloads.append(
            {
                "input_path": str(input_namelist),
                "which_rhs_values": [int(v) for v in chunk],
                "tol": float(tol),
                "atol": float(atol),
                "restart": int(restart),
                "maxiter": maxiter,
                "solve_method": str(solve_method),
                "identity_shift": float(identity_shift),
                "collect_transport_output_fields": bool(collect_transport_output_fields),
                "phi1_hat_base": phi1_payload,
                "differentiable": differentiable,
            }
        )
    return payloads


def _collect_pool_results(*, pool, payloads, worker) -> list[dict[str, object]]:
    future_to_index = {pool.submit(worker, payload): i for i, payload in enumerate(payloads)}
    results: list[dict[str, object] | None] = [None] * len(future_to_index)
    for fut in concurrent.futures.as_completed(future_to_index):
        results[future_to_index[fut]] = fut.result()
    ordered: list[dict[str, object]] = []
    for res in results:
        if res is None:
            raise RuntimeError("transport parallel worker result was not collected")
        ordered.append(res)
    return ordered


def run_transport_parallel_payloads(
    *,
    payloads: list[dict[str, object]],
    parallel_workers: int,
    parallel_backend: str,
    run_gpu_subprocesses: Callable[..., list[dict[str, object]]],
    persistent_pool_enabled: bool,
    get_pool: Callable[..., object],
    shutdown_pool: Callable[[], None],
    worker: Callable[[dict[str, object]], dict[str, object]],
    worker_env: Callable[[int], object],
    executor_class,
    executor_kwargs: Callable[..., dict[str, object]],
    emit: Callable[[int, str], None] | None = None,
) -> list[dict[str, object]]:
    """Run transport worker payloads using GPU subprocesses or CPU pools."""

    worker_count = validate_transport_parallel_worker_count(parallel_workers)
    if str(parallel_backend) == "gpu":
        return run_gpu_subprocesses(
            payloads=payloads,
            parallel_workers=worker_count,
            emit=emit,
        )

    if bool(persistent_pool_enabled):
        try:
            pool = get_pool(parallel_workers=worker_count, emit=emit)
            return _collect_pool_results(pool=pool, payloads=payloads, worker=worker)
        except BrokenProcessPool as exc:
            if emit is not None:
                emit(
                    1,
                    "solve_v3_transport_matrix_linear_gmres: persistent transport pool broke "
                    f"({type(exc).__name__}: {exc}); restarting pool once",
                )
            shutdown_pool()
            try:
                pool = get_pool(parallel_workers=worker_count, emit=emit)
                return _collect_pool_results(pool=pool, payloads=payloads, worker=worker)
            except Exception as retry_exc:
                if emit is not None:
                    emit(
                        1,
                        "solve_v3_transport_matrix_linear_gmres: persistent transport pool retry failed "
                        f"({type(retry_exc).__name__}: {retry_exc}); falling back to sequential whichRHS",
                    )
                return [worker(payload) for payload in payloads]
        except (PermissionError, NotImplementedError, OSError) as exc:
            if emit is not None:
                emit(
                    1,
                    "solve_v3_transport_matrix_linear_gmres: process parallelism unavailable "
                    f"({type(exc).__name__}: {exc}); falling back to sequential whichRHS",
                )
            return [worker(payload) for payload in payloads]

    with worker_env(worker_count):
        try:
            with executor_class(**executor_kwargs(parallel_workers=worker_count, emit=emit)) as pool:
                return _collect_pool_results(pool=pool, payloads=payloads, worker=worker)
        except (PermissionError, NotImplementedError, OSError) as exc:
            if emit is not None:
                emit(
                    1,
                    "solve_v3_transport_matrix_linear_gmres: process parallelism unavailable "
                    f"({type(exc).__name__}: {exc}); falling back to sequential whichRHS",
                )
            return [worker(payload) for payload in payloads]


# Persistent process-pool cache.
class TransportParallelPoolCache:
    """Persistent process-pool cache keyed by transport worker configuration."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pool = None
        self._key: tuple[object, ...] | None = None

    def shutdown(self) -> None:
        with self._lock:
            pool = self._pool
            self._pool = None
            self._key = None
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)

    def get(
        self,
        *,
        parallel_workers: int,
        key_fn: Callable[[int], tuple[object, ...]],
        worker_env: Callable[[int], object],
        executor_kwargs: Callable[..., dict[str, object]],
        executor_class: Callable[..., object],
        emit: Callable[[int, str], None] | None = None,
    ):
        key = key_fn(int(parallel_workers))
        with self._lock:
            if self._pool is not None and self._key == key:
                return self._pool
            old_pool = self._pool
            self._pool = None
            self._key = None

        if old_pool is not None:
            old_pool.shutdown(wait=True, cancel_futures=True)

        with worker_env(int(parallel_workers)):
            pool = executor_class(**executor_kwargs(parallel_workers=int(parallel_workers), emit=emit))

        with self._lock:
            self._pool = pool
            self._key = key
        return pool


_TRANSPORT_PARALLEL_POOL_CACHE = TransportParallelPoolCache()


def transport_parallel_worker_env(parallel_workers: int):
    """Return the process-pool worker environment context for transport solves."""
    return _transport_parallel_worker_env(
        parallel_workers=int(parallel_workers),
        rewrite_xla_flags=rewrite_xla_flags,
    )


def transport_parallel_pool_executor_kwargs(
    *,
    parallel_workers: int,
    emit: Callable[[int, str], None] | None = None,
) -> dict[str, object]:
    """Build ``ProcessPoolExecutor`` kwargs for transport worker pools."""
    return _transport_parallel_pool_executor_kwargs(
        parallel_workers=int(parallel_workers),
        get_context=mp.get_context,
        emit=emit,
    )


def shutdown_transport_parallel_pool() -> None:
    """Shut down the persistent transport process pool, if one exists."""
    _TRANSPORT_PARALLEL_POOL_CACHE.shutdown()


def get_transport_parallel_pool(
    *,
    parallel_workers: int,
    emit: Callable[[int, str], None] | None = None,
) -> concurrent.futures.ProcessPoolExecutor:
    """Return the persistent process pool for CPU transport-worker solves."""
    return _TRANSPORT_PARALLEL_POOL_CACHE.get(
        parallel_workers=int(parallel_workers),
        key_fn=transport_parallel_pool_key,
        worker_env=transport_parallel_worker_env,
        executor_kwargs=transport_parallel_pool_executor_kwargs,
        executor_class=concurrent.futures.ProcessPoolExecutor,
        emit=emit,
    )


def transport_parallel_process_pool_executor(**kwargs: object) -> concurrent.futures.ProcessPoolExecutor:
    """Construct the process-pool executor used by one-shot transport workers."""
    return concurrent.futures.ProcessPoolExecutor(**kwargs)


atexit.register(shutdown_transport_parallel_pool)


# Parent-side parallel solve orchestration.
@dataclass(frozen=True)
class TransportParallelSolveRuntime:
    """Injected runtime hooks needed to launch and merge transport workers."""

    run_gpu_subprocesses: Callable[..., list[dict[str, object]]]
    persistent_pool_enabled: bool
    get_pool: Callable[..., object]
    shutdown_pool: Callable[[], None]
    worker: Callable[[dict[str, object]], dict[str, object]]
    worker_env: Callable[[int], object]
    executor_class: Any
    executor_kwargs: Callable[..., dict[str, object]]
    elapsed_s: Callable[[], float]


def maybe_run_transport_parallel_solve(
    *,
    nml: Namelist,
    op0: V3FullSystemOperator,
    rhs_mode: int,
    n_rhs: int,
    which_rhs_values: Sequence[int],
    parallel_child: bool,
    parallel_workers: int,
    parallel_backend: str,
    input_namelist: Path | None,
    tol: float,
    atol: float,
    restart: int,
    maxiter: int | None,
    solve_method: str,
    identity_shift: float,
    collect_transport_output_fields: bool,
    phi1_hat_base: jnp.ndarray | None,
    differentiable: bool | None,
    runtime: TransportParallelSolveRuntime,
    emit: Callable[[int, str], None] | None = None,
) -> V3TransportMatrixSolveResult | None:
    """Run the parent-side parallel whichRHS branch, or return ``None``.

    The worker payload format and process/GPU execution helpers live in the
    transport-parallel modules. This function owns the parent orchestration that
    was historically embedded in ``v3_driver.py``: partitioning, worker launch,
    result merge, transport diagnostic assembly, and early result construction.
    """

    if not should_run_transport_parallel(
        parallel_child=bool(parallel_child),
        parallel_workers=int(parallel_workers),
        which_rhs_values=which_rhs_values,
        input_namelist=input_namelist,
    ):
        return None

    if input_namelist is None:
        raise RuntimeError("parallel transport solve requires input_namelist")

    if emit is not None:
        emit(
            0,
            "solve_v3_transport_matrix_linear_gmres: parallel whichRHS "
            f"(backend={parallel_backend} workers={int(parallel_workers)} "
            f"rhs_count={len(which_rhs_values)}/{int(n_rhs)})",
        )

    chunks = partition_transport_rhs(list(which_rhs_values), int(parallel_workers))
    payloads = build_transport_parallel_payloads(
        chunks=chunks,
        input_namelist=input_namelist,
        tol=float(tol),
        atol=float(atol),
        restart=int(restart),
        maxiter=maxiter,
        solve_method=str(solve_method),
        identity_shift=float(identity_shift),
        collect_transport_output_fields=bool(collect_transport_output_fields),
        phi1_hat_base=phi1_hat_base,
        differentiable=differentiable,
    )

    results = run_transport_parallel_payloads(
        payloads=payloads,
        parallel_workers=int(parallel_workers),
        parallel_backend=str(parallel_backend),
        run_gpu_subprocesses=runtime.run_gpu_subprocesses,
        persistent_pool_enabled=bool(runtime.persistent_pool_enabled),
        get_pool=runtime.get_pool,
        shutdown_pool=runtime.shutdown_pool,
        worker=runtime.worker,
        worker_env=runtime.worker_env,
        executor_class=runtime.executor_class,
        executor_kwargs=runtime.executor_kwargs,
        emit=emit,
    )

    state_vectors_np, residual_norms_np, rhs_norms_np, elapsed_s = merge_transport_parallel_results(
        n_rhs=int(n_rhs),
        results=results,
    )
    state_vectors = {
        int(k): jnp.asarray(v, dtype=jnp.float64)
        for k, v in state_vectors_np.items()
    }
    residual_norms = {
        int(k): jnp.asarray(v, dtype=jnp.float64)
        for k, v in residual_norms_np.items()
    }
    rhs_norms = {
        int(k): jnp.asarray(v, dtype=jnp.float64)
        for k, v in rhs_norms_np.items()
    }

    missing_rhs = [which_rhs for which_rhs in range(1, int(n_rhs) + 1) if which_rhs not in state_vectors]
    if missing_rhs:
        raise RuntimeError(f"parallel transport solve missing state vectors for whichRHS={missing_rhs}")

    if emit is not None:
        for which_rhs in range(1, int(n_rhs) + 1):
            rn = float(np.asarray(residual_norms.get(which_rhs, np.nan), dtype=np.float64))
            rhsn = float(np.asarray(rhs_norms.get(which_rhs, np.nan), dtype=np.float64))
            rel = rn / rhsn if np.isfinite(rhsn) and rhsn > 0.0 else float("nan")
            emit(
                0,
                f"whichRHS={which_rhs}: residual_norm={rn:.6e} rhs_norm={rhsn:.6e} "
                f"relative_residual={rel:.6e} elapsed_s={float(elapsed_s[which_rhs - 1]):.3f}",
            )
        emit(0, "solve_v3_transport_matrix_linear_gmres: computing whichRHS diagnostics (batched)")

    transport_fields_full = v3_transport_output_fields_vm_only(
        op0=op0,
        state_vectors_by_rhs=state_vectors,
    )
    diag_pf = jnp.asarray(transport_fields_full["particleFlux_vm_psiHat"], dtype=jnp.float64)
    diag_hf = jnp.asarray(transport_fields_full["heatFlux_vm_psiHat"], dtype=jnp.float64)
    diag_flow = jnp.asarray(transport_fields_full["FSABFlow"], dtype=jnp.float64)
    geom = geometry_from_namelist(nml=nml, grids=grids_from_namelist(nml))
    tm = v3_transport_matrix_from_flux_arrays(
        op=op0,
        geom=geom,
        particle_flux_vm_psi_hat=diag_pf,
        heat_flux_vm_psi_hat=diag_hf,
        fsab_flow=diag_flow,
    )
    transport_output_fields = transport_fields_full if bool(collect_transport_output_fields) else None
    if emit is not None:
        emit(0, "solve_v3_transport_matrix_linear_gmres: done")
        emit(1, f"solve_v3_transport_matrix_linear_gmres: elapsed_s={runtime.elapsed_s():.3f}")
    return V3TransportMatrixSolveResult(
        op0=op0,
        transport_matrix=tm,
        state_vectors_by_rhs=state_vectors,
        residual_norms_by_rhs=residual_norms,
        fsab_flow=diag_flow,
        particle_flux_vm_psi_hat=diag_pf,
        heat_flux_vm_psi_hat=diag_hf,
        elapsed_time_s=jnp.asarray(elapsed_s, dtype=jnp.float64),
        transport_output_fields=transport_output_fields,
        rhs_norms_by_rhs=rhs_norms,
    )
