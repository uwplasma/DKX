from __future__ import annotations

from collections.abc import Callable, Sequence
import concurrent.futures
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np

from .transport_parallel_policy import validate_transport_parallel_worker_count


def should_run_transport_parallel(
    *,
    parallel_child: bool,
    parallel_workers: int,
    which_rhs_values: Sequence[int],
    input_namelist: Path | None,
) -> bool:
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
