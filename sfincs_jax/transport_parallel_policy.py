from __future__ import annotations

from collections.abc import Callable, Iterator
import contextlib
from dataclasses import dataclass
import math
import os


@dataclass(frozen=True)
class TransportParallelScalingAudit:
    """Pure audit result for transport-worker benchmark summaries."""

    backend: str
    task_count: int
    worker_counts: tuple[int, ...]
    device_count: int | None
    baseline_s: float
    claim_workers: int
    claim_speedup: float
    claim_efficiency: float
    claim_finite_task_ideal_speedup: float
    deterministic_payload_coverage: bool
    release_scaling_claim: bool
    failures: tuple[str, ...]
    notes: tuple[str, ...]


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


def _finite_positive_float(value: object, *, name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number; got {value!r}") from exc
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be a finite positive number; got {value!r}")
    return numeric


def _optional_positive_int(value: object, *, name: str) -> int | None:
    if value is None:
        return None
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer; got {value!r}") from exc
    if numeric < 1:
        raise ValueError(f"{name} must be a positive integer; got {numeric}")
    return numeric


def _scaling_task_count(summary: dict[str, object]) -> int:
    for key in ("rhs_count", "task_count", "which_rhs_count", "n_rhs"):
        if key in summary:
            task_count = _optional_positive_int(summary[key], name=key)
            if task_count is not None:
                return task_count
    rhs_values = summary.get("which_rhs_values")
    if rhs_values is not None:
        try:
            return len(tuple(rhs_values))  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError("which_rhs_values must be an iterable of RHS indices") from exc
    raise ValueError("transport-worker scaling summary must include rhs_count or which_rhs_values")


def _scaling_device_count(summary: dict[str, object], *, backend: str) -> tuple[int | None, str | None]:
    for key in ("device_count", "devices", "gpu_count", "gpu_device_count"):
        if key in summary:
            return _optional_positive_int(summary[key], name=key), None
    visible_gpu_ids = summary.get("visible_gpu_ids")
    if visible_gpu_ids is not None:
        try:
            return len(tuple(visible_gpu_ids)), None  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError("visible_gpu_ids must be an iterable of device ids") from exc
    if backend == "cpu":
        return None, "device count not required for CPU process-worker scaling"
    return None, "GPU device count was not recorded; audit cannot prove one worker per device"


def _deterministic_payload_coverage(summary: dict[str, object], *, task_count: int) -> tuple[bool, str | None]:
    explicit = summary.get("deterministic_payload_coverage")
    if explicit is not None:
        return bool(explicit), None

    payloads = summary.get("payloads")
    if payloads is None:
        return False, "deterministic payload coverage was not recorded"

    expected_values = summary.get("which_rhs_values")
    if expected_values is None:
        expected = set(range(1, task_count + 1))
    else:
        try:
            expected = {int(v) for v in expected_values}  # type: ignore[union-attr]
        except (TypeError, ValueError) as exc:
            raise ValueError("which_rhs_values must contain integer RHS indices") from exc

    seen: list[int] = []
    try:
        payload_iter = tuple(payloads)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError("payloads must be an iterable of payload dictionaries") from exc
    for i, payload in enumerate(payload_iter):
        if not isinstance(payload, dict):
            raise ValueError(f"payloads[{i}] must be a dictionary")
        rhs_values = payload.get("which_rhs_values")
        if rhs_values is None:
            raise ValueError(f"payloads[{i}] must include which_rhs_values")
        try:
            seen.extend(int(v) for v in rhs_values)  # type: ignore[union-attr]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"payloads[{i}].which_rhs_values must contain integer RHS indices") from exc

    coverage_ok = len(seen) == len(set(seen)) and set(seen) == expected
    if coverage_ok:
        return True, None
    return False, f"payload RHS coverage must be deterministic and exact; expected {sorted(expected)}, got {seen}"


def audit_transport_parallel_scaling_summary(
    summary: dict[str, object],
    *,
    min_speedup: float = 1.2,
    min_efficiency: float = 0.50,
    min_parallel_workers: int = 2,
) -> TransportParallelScalingAudit:
    """Audit whether a transport-worker benchmark can support a release scaling claim.

    The audit is intentionally pure: it consumes a saved benchmark summary and
    does not inspect hardware, launch workers, or read benchmark artifacts.
    """
    if not isinstance(summary, dict):
        raise ValueError("transport-worker scaling summary must be a dictionary")

    backend = str(summary.get("backend", "cpu")).strip().lower() or "cpu"
    task_count = _scaling_task_count(summary)
    device_count, device_note = _scaling_device_count(summary, backend=backend)

    raw_results = summary.get("results")
    if not isinstance(raw_results, list) or not raw_results:
        raise ValueError("transport-worker scaling summary must include a non-empty results list")

    results: dict[int, dict[str, float]] = {}
    for i, result in enumerate(raw_results):
        if not isinstance(result, dict):
            raise ValueError(f"results[{i}] must be a dictionary")
        workers = validate_transport_parallel_worker_count(result.get("workers"), context=f"results[{i}]")
        mean_s = _finite_positive_float(result.get("mean_s"), name=f"results[{i}].mean_s")
        speedup_raw = result.get("speedup")
        speedup = _finite_positive_float(speedup_raw, name=f"results[{i}].speedup") if speedup_raw is not None else math.nan
        results[workers] = {"mean_s": mean_s, "speedup": speedup}

    if 1 not in results:
        raise ValueError("transport-worker scaling summary must include a 1-worker baseline")
    baseline_s = results[1]["mean_s"]
    for workers, result in results.items():
        if math.isnan(result["speedup"]):
            result["speedup"] = baseline_s / result["mean_s"]

    worker_counts = tuple(sorted(results))
    claim_workers = max(worker_counts)
    claim_speedup = results[claim_workers]["speedup"]
    claim_efficiency = claim_speedup / float(claim_workers)

    ideal_values = summary.get("ideal_speedup_finite_rhs")
    if ideal_values is not None:
        try:
            ideal_by_workers = {workers: float(v) for workers, v in zip(summary.get("workers", worker_counts), ideal_values, strict=False)}  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError("ideal_speedup_finite_rhs must be an iterable of finite positive numbers") from exc
        claim_ideal = ideal_by_workers.get(claim_workers)
        if claim_ideal is None:
            claim_ideal = float(task_count) / float(math.ceil(task_count / claim_workers))
        claim_finite_task_ideal_speedup = _finite_positive_float(claim_ideal, name="claim finite-task ideal speedup")
    else:
        claim_finite_task_ideal_speedup = float(task_count) / float(math.ceil(task_count / claim_workers))

    deterministic_payload_coverage, coverage_note = _deterministic_payload_coverage(summary, task_count=task_count)

    failures: list[str] = []
    notes: list[str] = []
    if task_count < min_parallel_workers:
        failures.append(f"only {task_count} independent transport tasks; need at least {min_parallel_workers}")
    if claim_workers < min_parallel_workers:
        failures.append(f"only {claim_workers} worker count audited; need at least {min_parallel_workers}")
    if backend == "gpu" and device_count is None:
        failures.append("GPU device count was not recorded")
    if backend == "gpu" and device_count is not None and device_count < claim_workers:
        failures.append(f"only {device_count} GPU devices recorded for {claim_workers} workers")
    if claim_speedup < float(min_speedup):
        failures.append(f"speedup {claim_speedup:.3g}x is below release gate {float(min_speedup):.3g}x")
    if claim_efficiency < float(min_efficiency):
        failures.append(f"efficiency {claim_efficiency:.3g} is below release gate {float(min_efficiency):.3g}")
    if not deterministic_payload_coverage:
        failures.append("deterministic payload coverage was not proven")
    if claim_speedup > claim_finite_task_ideal_speedup * 1.05:
        failures.append(
            f"speedup {claim_speedup:.3g}x exceeds finite-task ideal {claim_finite_task_ideal_speedup:.3g}x by more than 5%"
        )

    if device_note is not None:
        notes.append(device_note)
    if coverage_note is not None:
        notes.append(coverage_note)
    if task_count < claim_workers:
        notes.append(f"{task_count} transport tasks cannot keep {claim_workers} workers fully occupied")

    return TransportParallelScalingAudit(
        backend=backend,
        task_count=task_count,
        worker_counts=worker_counts,
        device_count=device_count,
        baseline_s=baseline_s,
        claim_workers=claim_workers,
        claim_speedup=claim_speedup,
        claim_efficiency=claim_efficiency,
        claim_finite_task_ideal_speedup=claim_finite_task_ideal_speedup,
        deterministic_payload_coverage=deterministic_payload_coverage,
        release_scaling_claim=not failures,
        failures=tuple(failures),
        notes=tuple(notes),
    )


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
