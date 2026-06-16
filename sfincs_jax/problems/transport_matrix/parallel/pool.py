from __future__ import annotations

from collections.abc import Callable
import threading


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
