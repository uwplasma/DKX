from __future__ import annotations

from collections.abc import Callable
import atexit
import concurrent.futures
import multiprocessing as mp
import threading

from sfincs_jax.problems.transport_matrix.parallel.policy import (
    rewrite_xla_flags,
    transport_parallel_pool_executor_kwargs as _transport_parallel_pool_executor_kwargs,
    transport_parallel_pool_key,
    transport_parallel_worker_env as _transport_parallel_worker_env,
)


class TransportParallelPoolCache:
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


__all__ = (
    "TransportParallelPoolCache",
    "get_transport_parallel_pool",
    "shutdown_transport_parallel_pool",
    "transport_parallel_pool_executor_kwargs",
    "transport_parallel_pool_key",
    "transport_parallel_process_pool_executor",
    "transport_parallel_worker_env",
)
