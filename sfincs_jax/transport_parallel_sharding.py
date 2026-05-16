"""Pure planning helpers for experimental single-case sharded solves."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math


_SHARDED_SOLVE_KINDS = {
    "single_case_sharded_solve",
    "sharded_solve_scaling",
    "sharded_solve",
}
_EXPERIMENTAL_STATUSES = {
    "experimental",
    "experimental_single_case_sharding",
    "regression_snapshot",
    "non_release_snapshot",
}
_SUPPORTED_SINGLE_CASE_AXES = {"theta", "zeta"}


@dataclass(frozen=True)
class ShardedSolveDeviceAssignment:
    """Deterministic contiguous shard assigned to one logical device."""

    device_index: int
    device_id: str | None
    shard_start: int
    shard_stop: int
    work_units: int
    workload_fraction: float


@dataclass(frozen=True)
class ShardedSolveBalanceDiagnostics:
    """Per-device workload balance summary for a planned sharded solve."""

    total_work_units: int
    min_work_units: int
    max_work_units: int
    imbalance_units: int
    max_to_mean_ratio: float
    idle_device_count: int


@dataclass(frozen=True)
class ShardedSolveExecutionPlan:
    """Release-safe metadata for experimental single-case sharding."""

    benchmark_kind: str
    backend: str
    rhs_mode: int
    shard_axis: str
    task_count: int
    requested_devices: int
    active_devices: int
    available_device_count: int
    available_device_ids: tuple[str, ...]
    capped: bool
    cap_reasons: tuple[str, ...]
    eligible_for_single_case_sharding: bool
    release_scaling_claim: bool
    release_scaling_supported: bool
    experimental_single_case_scaling: bool
    failures: tuple[str, ...]
    notes: tuple[str, ...]
    device_assignments: tuple[ShardedSolveDeviceAssignment, ...]
    balance_diagnostics: ShardedSolveBalanceDiagnostics

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable plan dictionary."""

        return asdict(self)


def _positive_int(value: object, *, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer; got {value!r}") from exc
    if parsed < 1:
        raise ValueError(f"{name} must be a positive integer; got {parsed}")
    return parsed


def _nonnegative_int(value: object, *, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer; got {value!r}") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer; got {parsed}")
    return parsed


def _normalized_token(value: object) -> str:
    return str(value).strip().lower().replace("-", "_")


def _normalize_backend(value: object) -> str:
    backend = _normalized_token(value or "auto")
    if backend in {"auto", "cpu", "gpu"}:
        return backend
    raise ValueError(f"backend must be one of 'auto', 'cpu', or 'gpu'; got {value!r}")


def _unique_device_ids(values: object) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        raw_values = [part.strip() for part in values.split(",")]
    else:
        try:
            raw_values = [str(part).strip() for part in values]  # type: ignore[operator]
        except TypeError as exc:
            raise ValueError("available_device_ids must be an iterable of device ids") from exc

    seen: set[str] = set()
    unique: list[str] = []
    for device_id in raw_values:
        if device_id and device_id not in seen:
            unique.append(device_id)
            seen.add(device_id)
    return tuple(unique)


def _balanced_assignments(
    *,
    active_devices: int,
    available_device_ids: tuple[str, ...],
    total_work_units: int,
) -> tuple[ShardedSolveDeviceAssignment, ...]:
    if active_devices < 1:
        return ()
    base, remainder = divmod(int(total_work_units), int(active_devices))
    start = 0
    assignments: list[ShardedSolveDeviceAssignment] = []
    for idx in range(int(active_devices)):
        work_units = base + (1 if idx < remainder else 0)
        stop = start + work_units
        assignments.append(
            ShardedSolveDeviceAssignment(
                device_index=idx,
                device_id=available_device_ids[idx] if idx < len(available_device_ids) else None,
                shard_start=start,
                shard_stop=stop,
                work_units=work_units,
                workload_fraction=float(work_units) / float(total_work_units)
                if total_work_units > 0
                else 0.0,
            )
        )
        start = stop
    return tuple(assignments)


def _balance_diagnostics(
    *,
    assignments: tuple[ShardedSolveDeviceAssignment, ...],
    total_work_units: int,
    available_device_count: int,
) -> ShardedSolveBalanceDiagnostics:
    if not assignments:
        return ShardedSolveBalanceDiagnostics(
            total_work_units=int(total_work_units),
            min_work_units=0,
            max_work_units=0,
            imbalance_units=0,
            max_to_mean_ratio=math.inf,
            idle_device_count=max(int(available_device_count), 0),
        )
    work = [assignment.work_units for assignment in assignments]
    mean = float(sum(work)) / float(len(work))
    return ShardedSolveBalanceDiagnostics(
        total_work_units=int(total_work_units),
        min_work_units=min(work),
        max_work_units=max(work),
        imbalance_units=max(work) - min(work),
        max_to_mean_ratio=(max(work) / mean) if mean > 0.0 else math.inf,
        idle_device_count=max(int(available_device_count) - len(assignments), 0),
    )


def plan_single_case_sharded_solve(
    *,
    requested_devices: int,
    backend: str = "auto",
    available_device_count: int | None = None,
    available_device_ids: object = None,
    rhs_mode: int = 1,
    shard_axis: str = "theta",
    shard_axis_size: int | None = None,
    benchmark_kind: str = "single_case_sharded_solve",
    task_count: int = 1,
    release_scaling_claim: bool = False,
    experimental_single_case_scaling: bool = False,
    scaling_status: str = "",
    min_parallel_devices: int = 2,
) -> ShardedSolveExecutionPlan:
    """Build a fail-closed device plan for a single-case sharded solve.

    The helper is intentionally pure. It never probes JAX or starts workers; the
    caller must pass visible device metadata from a benchmark plan or smoke test.
    """

    requested = _positive_int(requested_devices, name="requested_devices")
    min_devices = _positive_int(min_parallel_devices, name="min_parallel_devices")
    backend_norm = _normalize_backend(backend)
    rhs_mode_value = _positive_int(rhs_mode, name="rhs_mode")
    task_count_value = _positive_int(task_count, name="task_count")
    axis = _normalized_token(shard_axis)
    benchmark_kind_norm = _normalized_token(benchmark_kind)
    device_ids = _unique_device_ids(available_device_ids)

    if available_device_count is None:
        available_count = len(device_ids) if device_ids else requested
    else:
        available_count = _nonnegative_int(available_device_count, name="available_device_count")
    if not device_ids and available_count > 0:
        device_ids = tuple(str(idx) for idx in range(available_count))

    active_devices = min(requested, available_count)
    cap_reasons: list[str] = []
    if available_count < requested:
        cap_reasons.append(f"available devices={available_count}")

    if shard_axis_size is not None:
        axis_size = _positive_int(shard_axis_size, name="shard_axis_size")
        if axis_size < active_devices:
            active_devices = axis_size
            cap_reasons.append(f"{axis} shard axis size={axis_size}")
        total_work_units = axis_size
    else:
        total_work_units = max(active_devices, 1)

    marked_experimental = bool(experimental_single_case_scaling) or _normalized_token(
        scaling_status
    ) in _EXPERIMENTAL_STATUSES

    failures: list[str] = []
    notes: list[str] = []
    if benchmark_kind_norm not in _SHARDED_SOLVE_KINDS:
        failures.append(
            "single-case sharding plan requires benchmark_kind='single_case_sharded_solve'"
        )
    if task_count_value != 1:
        failures.append(
            f"single-case sharding plan requires task_count=1; got {task_count_value}"
        )
    if rhs_mode_value != 1:
        failures.append(
            f"single-case sharded solve planning is only eligible for RHSMode=1; got {rhs_mode_value}"
        )
    if axis not in _SUPPORTED_SINGLE_CASE_AXES:
        failures.append(
            f"single-case sharded solve requires shard_axis in {sorted(_SUPPORTED_SINGLE_CASE_AXES)}; got {axis!r}"
        )
    if active_devices < min_devices:
        failures.append(
            f"only {active_devices} active devices planned; need at least {min_devices}"
        )
    if bool(release_scaling_claim):
        failures.append("single-case sharded solve cannot set release_scaling_claim=true")
    if not marked_experimental:
        failures.append("single-case sharded solve must be marked experimental/non-release")
    if backend_norm == "gpu" and available_count < requested:
        notes.append("GPU plan capped requested devices to visible unique devices")
    if active_devices < requested:
        notes.append(
            f"requested_devices={requested} was capped to active_devices={active_devices}"
        )
    notes.append(
        "single-case sharded solve is planning metadata only and is not release scaling evidence"
    )

    assignments = _balanced_assignments(
        active_devices=active_devices,
        available_device_ids=device_ids,
        total_work_units=total_work_units,
    )
    balance = _balance_diagnostics(
        assignments=assignments,
        total_work_units=total_work_units,
        available_device_count=available_count,
    )

    return ShardedSolveExecutionPlan(
        benchmark_kind=benchmark_kind_norm,
        backend=backend_norm,
        rhs_mode=rhs_mode_value,
        shard_axis=axis,
        task_count=task_count_value,
        requested_devices=requested,
        active_devices=active_devices,
        available_device_count=available_count,
        available_device_ids=device_ids,
        capped=active_devices < requested,
        cap_reasons=tuple(cap_reasons),
        eligible_for_single_case_sharding=not failures,
        release_scaling_claim=False,
        release_scaling_supported=False,
        experimental_single_case_scaling=marked_experimental,
        failures=tuple(failures),
        notes=tuple(notes),
        device_assignments=assignments,
        balance_diagnostics=balance,
    )


__all__ = [
    "ShardedSolveBalanceDiagnostics",
    "ShardedSolveDeviceAssignment",
    "ShardedSolveExecutionPlan",
    "plan_single_case_sharded_solve",
]
