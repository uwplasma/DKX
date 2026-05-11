from __future__ import annotations

from collections.abc import Callable, Iterator
import contextlib
import os


def validate_transport_parallel_worker_count(
    parallel_workers: int,
    *,
    context: str = "transport parallel",
) -> int:
    """Return a validated positive transport worker count."""
    try:
        workers = int(parallel_workers)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} worker count must be an integer >= 1; got {parallel_workers!r}") from exc
    if workers < 1:
        raise ValueError(f"{context} worker count must be >= 1; got {workers}")
    return workers


@contextlib.contextmanager
def transport_parallel_worker_env(
    *,
    parallel_workers: int,
    rewrite_xla_flags: Callable[[str, int | None, int | None], str],
) -> Iterator[None]:
    """Cap XLA threads + disable sharding in transport worker processes."""
    workers = validate_transport_parallel_worker_count(parallel_workers)
    saved: dict[str, str | None] = {}

    def _set(key: str, value: str | None) -> None:
        saved[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

    try:
        cores_env = os.environ.get("SFINCS_JAX_CORES", "").strip()
        total_cores = int(cores_env) if cores_env else 0
    except ValueError:
        total_cores = 0
    if total_cores <= 0:
        total_cores = os.cpu_count() or 1
    threads = max(1, int(total_cores) // workers)

    _set("SFINCS_JAX_SHARD", "0")
    _set("SFINCS_JAX_MATVEC_SHARD_AXIS", "off")
    _set("SFINCS_JAX_AUTO_SHARD", "0")
    _set("SFINCS_JAX_CPU_DEVICES", "1")
    pin_threads_env = os.environ.get("SFINCS_JAX_TRANSPORT_PIN_THREADS", "").strip().lower()
    if pin_threads_env in {"1", "true", "yes", "on"}:
        flags = rewrite_xla_flags(os.environ.get("XLA_FLAGS", ""), None, 1)
        _set("XLA_FLAGS", flags or None)
        _set("OMP_NUM_THREADS", str(int(threads)))
        _set("OPENBLAS_NUM_THREADS", str(int(threads)))
        _set("MKL_NUM_THREADS", str(int(threads)))
        _set("VECLIB_MAXIMUM_THREADS", str(int(threads)))
        _set("NUMEXPR_NUM_THREADS", str(int(threads)))

    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def transport_parallel_start_method() -> str:
    env = os.environ.get("SFINCS_JAX_TRANSPORT_MP_START_METHOD", "").strip().lower()
    if env in {"", "auto"}:
        return "spawn"
    if env in {"spawn", "fork", "forkserver"}:
        return env
    return "spawn"


def transport_parallel_backend() -> str:
    env = os.environ.get("SFINCS_JAX_TRANSPORT_PARALLEL_BACKEND", "").strip().lower()
    if env in {"", "auto", "cpu", "process"}:
        return "cpu"
    if env in {"gpu", "gpu_process", "process_gpu"}:
        return "gpu"
    return "cpu"


def transport_parallel_persistent_pool_enabled() -> bool:
    env = os.environ.get("SFINCS_JAX_TRANSPORT_POOL_PERSIST", "").strip().lower()
    return env not in {"0", "false", "no", "off"}


def transport_parallel_pool_key(parallel_workers: int) -> tuple[object, ...]:
    return (
        validate_transport_parallel_worker_count(parallel_workers),
        transport_parallel_backend(),
        transport_parallel_start_method(),
        os.environ.get("SFINCS_JAX_TRANSPORT_PIN_THREADS", "").strip().lower(),
        os.environ.get("SFINCS_JAX_CORES", "").strip(),
    )


def transport_parallel_visible_gpu_ids(parallel_workers: int) -> list[str]:
    workers = validate_transport_parallel_worker_count(parallel_workers, context="GPU transport")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        ids: list[str] = []
        seen: set[str] = set()
        for token in visible.split(","):
            gpu_id = token.strip()
            if gpu_id and gpu_id not in seen:
                ids.append(gpu_id)
                seen.add(gpu_id)
        return ids
    return [str(i) for i in range(workers)]


def transport_parallel_gpu_worker_env(*, gpu_id: str) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["SFINCS_JAX_TRANSPORT_PARALLEL"] = "off"
    env["SFINCS_JAX_TRANSPORT_PARALLEL_CHILD"] = "1"
    env["SFINCS_JAX_TRANSPORT_PARALLEL_BACKEND"] = "gpu"
    env["SFINCS_JAX_SHARD"] = "0"
    env["SFINCS_JAX_MATVEC_SHARD_AXIS"] = "off"
    env["SFINCS_JAX_AUTO_SHARD"] = "0"
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
    return env


def transport_parallel_pool_executor_kwargs(
    *,
    parallel_workers: int,
    get_context: Callable[[str], object],
    emit: Callable[[int, str], None] | None = None,
) -> dict[str, object]:
    workers = validate_transport_parallel_worker_count(parallel_workers)
    kwargs: dict[str, object] = {"max_workers": workers}
    start_method = transport_parallel_start_method()
    try:
        kwargs["mp_context"] = get_context(start_method)
    except ValueError:
        kwargs["mp_context"] = get_context("spawn")
        if emit is not None:
            emit(
                1,
                "solve_v3_transport_matrix_linear_gmres: invalid "
                f"SFINCS_JAX_TRANSPORT_MP_START_METHOD={start_method!r}; using 'spawn'.",
            )
    return kwargs
