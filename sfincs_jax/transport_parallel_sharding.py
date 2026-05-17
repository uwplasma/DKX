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


@dataclass(frozen=True)
class ShardedSolveAmortizationDiagnostics:
    """Communication/setup model for a single-case sharded solve claim."""

    active_devices: int
    serial_work_units: float
    per_device_work_units: float
    setup_work_units: float
    krylov_iterations: int
    collectives_per_iteration: int
    collective_latency_units: float
    halo_bytes_per_iteration: float
    bandwidth_bytes_per_unit: float
    communication_work_units: float
    predicted_parallel_units: float
    predicted_speedup: float
    parallel_efficiency: float
    communication_fraction: float
    release_scaling_supported: bool
    failures: tuple[str, ...]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable diagnostics dictionary."""

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


def estimate_sharded_solve_amortization(
    *,
    active_devices: int,
    serial_work_units: float,
    setup_work_units: float = 0.0,
    krylov_iterations: int = 0,
    collectives_per_iteration: int = 2,
    collective_latency_units: float = 0.0,
    halo_bytes_per_iteration: float = 0.0,
    bandwidth_bytes_per_unit: float = math.inf,
    min_work_units_per_device: float = 1.0,
    min_speedup: float = 1.25,
    min_efficiency: float = 0.55,
    max_communication_fraction: float = 0.35,
) -> ShardedSolveAmortizationDiagnostics:
    """Estimate whether single-case sharding can be promoted as strong scaling.

    The units are intentionally abstract: callers can pass seconds, matvec-cost
    units, or normalized benchmark work. The gate captures the basic
    communication-avoiding/domain-decomposition requirement that useful strong
    scaling needs enough local work to amortize setup, halo exchange, and Krylov
    collectives.
    """

    devices = _positive_int(active_devices, name="active_devices")
    iterations = _nonnegative_int(krylov_iterations, name="krylov_iterations")
    collectives = _nonnegative_int(collectives_per_iteration, name="collectives_per_iteration")
    serial = float(serial_work_units)
    setup = max(0.0, float(setup_work_units))
    latency = max(0.0, float(collective_latency_units))
    halo_bytes = max(0.0, float(halo_bytes_per_iteration))
    bandwidth = float(bandwidth_bytes_per_unit)
    if not math.isfinite(serial) or serial <= 0.0:
        raise ValueError("serial_work_units must be positive and finite")
    if bandwidth <= 0.0:
        raise ValueError("bandwidth_bytes_per_unit must be positive")

    bandwidth_cost = 0.0 if math.isinf(bandwidth) else halo_bytes / bandwidth
    communication = float(iterations) * (float(collectives) * latency + bandwidth_cost)
    per_device_work = serial / float(devices)
    parallel_units = per_device_work + setup + communication
    predicted_speedup = serial / parallel_units if parallel_units > 0.0 else math.inf
    efficiency = predicted_speedup / float(devices)
    communication_fraction = communication / parallel_units if parallel_units > 0.0 else math.inf

    failures: list[str] = []
    notes: list[str] = []
    if devices < 2:
        failures.append("single-case strong scaling requires at least 2 active devices")
    if per_device_work < float(min_work_units_per_device):
        failures.append(
            "per-device work below amortization floor "
            f"({per_device_work:.3g} < {float(min_work_units_per_device):.3g})"
        )
    if predicted_speedup < float(min_speedup):
        failures.append(
            f"predicted speedup {predicted_speedup:.3g}x below gate {float(min_speedup):.3g}x"
        )
    if efficiency < float(min_efficiency):
        failures.append(
            f"parallel efficiency {efficiency:.3g} below gate {float(min_efficiency):.3g}"
        )
    if communication_fraction > float(max_communication_fraction):
        failures.append(
            "communication fraction above gate "
            f"({communication_fraction:.3g} > {float(max_communication_fraction):.3g})"
        )
    if setup > per_device_work:
        notes.append("setup cost exceeds per-device compute work")
    if communication > per_device_work:
        notes.append("communication cost exceeds per-device compute work")

    return ShardedSolveAmortizationDiagnostics(
        active_devices=devices,
        serial_work_units=serial,
        per_device_work_units=per_device_work,
        setup_work_units=setup,
        krylov_iterations=iterations,
        collectives_per_iteration=collectives,
        collective_latency_units=latency,
        halo_bytes_per_iteration=halo_bytes,
        bandwidth_bytes_per_unit=bandwidth,
        communication_work_units=communication,
        predicted_parallel_units=parallel_units,
        predicted_speedup=predicted_speedup,
        parallel_efficiency=efficiency,
        communication_fraction=communication_fraction,
        release_scaling_supported=not failures,
        failures=tuple(failures),
        notes=tuple(notes),
    )


__all__ = [
    "ShardedSolveAmortizationDiagnostics",
    "ShardedSolveBalanceDiagnostics",
    "ShardedSolveDeviceAssignment",
    "ShardedSolveExecutionPlan",
    "estimate_sharded_solve_amortization",
    "plan_single_case_sharded_solve",
]
