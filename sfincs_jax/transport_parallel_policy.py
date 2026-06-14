"""Pure transport parallelism, scaling-audit, and worker-env policy helpers."""

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
    benchmark_kind: str = "transport_worker_scaling"
    timing_semantics: str | None = None
    deterministic_output_check: bool = False
    compile_amortization_gate: bool = False
    min_speedup: float = 1.2
    min_efficiency: float = 0.50
    min_parallel_workers: int = 2


@dataclass(frozen=True)
class ShardedSolveScalingAudit:
    """Pure audit result for single-case sharded-solve benchmark summaries."""

    backend: str
    device_counts: tuple[int, ...]
    baseline_s: float
    claim_devices: int
    claim_speedup: float
    claim_efficiency: float
    release_scaling_claim: bool
    experimental_single_case_scaling: bool
    ci_gate_pass: bool
    failures: tuple[str, ...]
    notes: tuple[str, ...]
    benchmark_kind: str = "single_case_sharded_solve"
    timing_semantics: str | None = None
    deterministic_output_check: bool = False
    operator_reuse_gate: bool = False
    deterministic_output_gate: bool = False
    release_promotion_supported: bool = False
    release_promotion_blockers: tuple[str, ...] = ()


@dataclass(frozen=True)
class MultiGpuCaseThroughputAudit:
    """Pure audit result for one-GPU-per-case throughput benchmark summaries."""

    backend: str
    required_gpu_count: int
    sequential_wall_s: float
    parallel_wall_s: float
    throughput_speedup: float
    release_scaling_claim: bool
    ci_gate_pass: bool
    failures: tuple[str, ...]
    notes: tuple[str, ...]
    benchmark_kind: str = "multi_gpu_case_throughput"
    timing_semantics: str | None = None
    min_throughput_speedup: float = 1.0


@dataclass(frozen=True)
class ParallelScalingClaimScopeAudit:
    """Pure audit separating throughput evidence from single-case strong scaling."""

    benchmark_kind: str
    claim_scope: str
    independent_task_count: int
    parallel_count: int
    backend: str
    artifact_kind: str | None
    launches_solves: bool | None
    plan_only_scope_evidence: bool
    measured_results_present: bool
    release_gate_required: str | None
    claim_scope_release_eligible: bool
    release_scaling_supported: bool
    unsupported_single_case_strong_scaling: bool
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


def rewrite_xla_flags(flags: str, cpu_threads: int | None, host_devices: int | None) -> str:
    """Rewrite worker-local XLA flags without duplicating stale thread/device caps.

    Transport worker pools set CPU thread and host-device limits in child
    processes. Rewriting instead of appending avoids conflicting XLA tokens when
    workers are reused or launched from shells that already export ``XLA_FLAGS``.
    """
    parts = []
    for token in str(flags).split():
        if token.startswith("--xla_cpu_parallelism_threads="):
            continue
        if token.startswith("--xla_cpu_multi_thread_eigen_num_threads="):
            continue
        if token.startswith("--xla_cpu_multi_thread_eigen="):
            continue
        if token.startswith("--xla_force_host_platform_device_count="):
            continue
        parts.append(token)
    if cpu_threads is not None:
        parts.append("--xla_cpu_multi_thread_eigen=true")
        parts.append(f"--xla_cpu_multi_thread_eigen_num_threads={int(cpu_threads)}")
    if host_devices is not None:
        parts.append(f"--xla_force_host_platform_device_count={int(host_devices)}")
    return " ".join(parts).strip()


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


def _optional_nonnegative_int(value: object, *, name: str) -> int | None:
    if value is None:
        return None
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer; got {value!r}") from exc
    if numeric < 0:
        raise ValueError(f"{name} must be a non-negative integer; got {numeric}")
    return numeric


def _optional_bool(value: object, *, name: str) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean; got {value!r}")


def _speedup_threshold(value: object, *, name: str) -> float:
    threshold = _finite_positive_float(value, name=name)
    if threshold < 1.0:
        raise ValueError(f"{name} must be >= 1.0; got {threshold!r}")
    return threshold


def _efficiency_threshold(value: object, *, name: str) -> float:
    threshold = _finite_positive_float(value, name=name)
    if threshold > 1.0:
        raise ValueError(f"{name} must be <= 1.0; got {threshold!r}")
    return threshold


def _normalized_benchmark_kind(summary: dict[str, object], *, default: str) -> str:
    return str(summary.get("benchmark_kind", default)).strip().lower().replace("-", "_")


def _timing_semantics(summary: dict[str, object]) -> tuple[str | None, str | None]:
    raw = summary.get("timing_semantics", summary.get("timing_mode", summary.get("cache_state")))
    if raw is not None:
        normalized = str(raw).strip().lower().replace("-", "_")
        if normalized in {"warm", "cache_warm", "warm_cache", "hot", "hot_solve", "hot_run"}:
            return normalized, None
        if normalized in {"cold", "cold_start", "cold_cache"}:
            return normalized, None
        if normalized in {"mixed", "mixed_warm_cold"}:
            return normalized, None
        return None, f"unrecognized timing semantics {raw!r}"

    warm_keys = (
        "global_warmup",
        "warmup",
        "per_worker_warmup",
        "per_device_warmup",
        "inner_warmup_solves",
    )
    warmups: list[int] = []
    for key in warm_keys:
        if key in summary:
            value = _optional_nonnegative_int(summary[key], name=key)
            if value is not None:
                warmups.append(value)
    if any(value > 0 for value in warmups):
        return "cache_warm", "timing semantics inferred from recorded warmup counts"
    if warmups:
        return "cold_start", "timing semantics inferred from zero warmup counts"
    return None, "timing semantics were not recorded"


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
    for key in ("device_count", "gpu_device_count", "gpu_count", "devices"):
        if key in summary:
            try:
                return _optional_positive_int(summary[key], name=key), None
            except ValueError:
                if key != "devices":
                    raise
                try:
                    values = tuple(int(value) for value in summary[key])  # type: ignore[index]
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{key} must be a positive integer or device-count list") from exc
                if not values or any(value < 1 for value in values):
                    raise ValueError(f"{key} must contain positive device counts")
                return max(values), None
    visible_gpu_ids = summary.get("visible_gpu_ids", summary.get("device_ids"))
    if visible_gpu_ids is not None:
        try:
            ids = tuple(str(value).strip() for value in visible_gpu_ids)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError("visible_gpu_ids must be an iterable of device ids") from exc
        unique_count = len({gpu_id for gpu_id in ids if gpu_id})
        if unique_count < 1:
            return None, "visible_gpu_ids did not contain any device ids"
        return unique_count, None
    if backend == "cpu":
        return None, "device count not required for CPU process-worker scaling"
    return None, "GPU device count was not recorded; audit cannot prove one worker per device"


def _payloads_for_claim(
    summary: dict[str, object],
    *,
    claim_workers: int,
) -> object:
    payloads_by_workers = summary.get("payloads_by_workers")
    if isinstance(payloads_by_workers, dict):
        for key in (claim_workers, str(claim_workers)):
            if key in payloads_by_workers:
                return payloads_by_workers[key]

    raw_results = summary.get("results")
    if isinstance(raw_results, list):
        for result in raw_results:
            if not isinstance(result, dict):
                continue
            try:
                workers = validate_transport_parallel_worker_count(result.get("workers"), context="results")
            except ValueError:
                continue
            if workers == int(claim_workers) and "payloads" in result:
                return result["payloads"]

    return summary.get("payloads")


def _deterministic_payload_coverage(
    summary: dict[str, object],
    *,
    task_count: int,
    claim_workers: int,
) -> tuple[bool, str | None]:
    explicit = _optional_bool(summary.get("deterministic_payload_coverage"), name="deterministic_payload_coverage")

    payloads = _payloads_for_claim(summary, claim_workers=claim_workers)
    if payloads is None:
        if explicit:
            return False, "deterministic payload coverage was asserted but payload chunks were not recorded"
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
    if coverage_ok and explicit is not False:
        return True, None
    if coverage_ok:
        return False, "deterministic payload coverage was explicitly marked false"
    return False, f"payload RHS coverage must be deterministic and exact; expected {sorted(expected)}, got {seen}"


def _deterministic_output_check(
    summary: dict[str, object],
    *,
    deterministic_payload_coverage: bool,
) -> tuple[bool, str | None]:
    explicit = _optional_bool(summary.get("deterministic_output_check"), name="deterministic_output_check")
    if explicit is not None:
        if explicit:
            return True, None
        return False, "deterministic output check was explicitly marked false"
    if deterministic_payload_coverage:
        return True, "deterministic output check inferred from exact payload coverage"
    return False, "deterministic output check was not recorded"


def _compile_amortization_gate(
    summary: dict[str, object],
    *,
    timing_semantics: str | None,
) -> tuple[bool, bool, tuple[str, ...], tuple[str, ...]]:
    raw = summary.get("compile_amortization_gate", summary.get("compile_amortization"))
    if raw is None:
        return False, False, (), ("compile-amortization gate was not recorded",)
    if not isinstance(raw, dict):
        raise ValueError("compile_amortization_gate must be a dictionary")

    failures: list[str] = []
    notes: list[str] = []
    passes = _optional_bool(raw.get("passes"), name="compile_amortization_gate.passes")
    if passes is None:
        failures.append("compile-amortization gate must record passes=true/false")
    elif not passes:
        failures.append("compile-amortization gate did not pass")

    gate_timing_raw = raw.get("timing_semantics")
    if gate_timing_raw is not None:
        gate_timing = str(gate_timing_raw).strip().lower().replace("-", "_")
        if timing_semantics is not None and gate_timing != timing_semantics:
            failures.append(
                "compile-amortization timing semantics "
                f"{gate_timing!r} do not match summary timing semantics {timing_semantics!r}"
            )

    compile_in_timed_region = _optional_bool(
        raw.get("compile_in_timed_region"),
        name="compile_amortization_gate.compile_in_timed_region",
    )
    if compile_in_timed_region:
        failures.append("compile-amortization gate records compilation inside the timed region")

    warm_run_amortization = _optional_bool(
        raw.get("warm_run_amortization_pass", raw.get("warm_run_amortization")),
        name="compile_amortization_gate.warm_run_amortization_pass",
    )
    if warm_run_amortization is False:
        failures.append("compile-amortization gate did not prove warm-run amortization")

    cache_required = _optional_bool(
        raw.get("cache_required"),
        name="compile_amortization_gate.cache_required",
    )
    persistent_compile_cache = _optional_bool(
        raw.get("persistent_compile_cache"),
        name="compile_amortization_gate.persistent_compile_cache",
    )
    compile_cache_dir = raw.get("compile_cache_dir")
    has_compile_cache_dir = compile_cache_dir is not None and bool(str(compile_cache_dir).strip())
    if cache_required is True and persistent_compile_cache is not True:
        failures.append("compile-amortization gate requires persistent_compile_cache=true")
    if cache_required is True and not has_compile_cache_dir:
        failures.append("compile-amortization gate requires compile_cache_dir metadata")
    if persistent_compile_cache is True and not has_compile_cache_dir:
        failures.append("persistent_compile_cache=true requires compile_cache_dir metadata")

    timed_repeats = _optional_positive_int(
        raw.get("timed_repeats", raw.get("repeats")),
        name="compile_amortization_gate.timed_repeats",
    )
    min_timed_repeats = _optional_positive_int(
        raw.get("min_timed_repeats", 1),
        name="compile_amortization_gate.min_timed_repeats",
    )
    if timed_repeats is not None and min_timed_repeats is not None and timed_repeats < min_timed_repeats:
        failures.append(
            f"compile-amortization gate recorded only {timed_repeats} timed repeats; "
            f"need at least {min_timed_repeats}"
        )

    if timing_semantics in {"cold", "cold_start", "cold_cache", "mixed", "mixed_warm_cold"}:
        failures.append(f"compile-amortization gate cannot pass timing semantics {timing_semantics!r}")

    reason = raw.get("reason")
    if reason is not None:
        notes.append(str(reason))
    raw_notes = raw.get("notes")
    if raw_notes is not None:
        try:
            notes.extend(str(note) for note in raw_notes)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError("compile_amortization_gate.notes must be an iterable of strings") from exc

    return True, not failures, tuple(failures), tuple(notes)


def _operator_reuse_gate(
    summary: dict[str, object],
    *,
    timing_semantics: str | None,
) -> tuple[bool, bool, tuple[str, ...], tuple[str, ...]]:
    raw = summary.get("operator_reuse_gate", summary.get("compile_amortization_gate"))
    if raw is None:
        return False, False, (), ("compiled sharded operator-reuse gate was not recorded",)
    return _compile_amortization_gate(
        {"compile_amortization_gate": raw},
        timing_semantics=timing_semantics,
    )


def _sharded_deterministic_output_gate(
    summary: dict[str, object],
) -> tuple[bool, tuple[str, ...], tuple[str, ...]]:
    raw = summary.get("deterministic_output_gate")
    explicit = _optional_bool(
        summary.get("deterministic_output_check"),
        name="deterministic_output_check",
    )
    if raw is None:
        if explicit:
            return True, (), ("legacy deterministic_output_check=true without gate schema",)
        return False, (), ("deterministic residual/output gate was not recorded",)
    if not isinstance(raw, dict):
        raise ValueError("deterministic_output_gate must be a dictionary")

    passes = _optional_bool(raw.get("passes"), name="deterministic_output_gate.passes")
    if passes is None:
        raise ValueError("deterministic_output_gate must record passes=true/false")

    failures: list[str] = []
    notes: list[str] = []
    threshold_raw = raw.get("residual_tolerance", raw.get("max_relative_residual_norm_gate"))
    observed_raw = raw.get("max_relative_residual_norm")
    if passes:
        if threshold_raw is None:
            failures.append("deterministic_output_gate.passes=true requires residual_tolerance")
        else:
            threshold = _finite_positive_float(
                threshold_raw,
                name="deterministic_output_gate.residual_tolerance",
            )
            if observed_raw is None:
                failures.append(
                    "deterministic_output_gate.passes=true requires max_relative_residual_norm"
                )
            else:
                observed = float(observed_raw)
                if not math.isfinite(observed) or observed > threshold:
                    failures.append(
                        "deterministic output residual exceeds tolerance "
                        f"({observed!r} > {threshold:g})"
                    )
        output_digest = raw.get("output_digest")
        baseline_digest = raw.get("baseline_output_digest")
        comparison_digest = raw.get("comparison_output_digest")
        has_output_digest = output_digest is not None and bool(str(output_digest).strip())
        has_baseline_digest = baseline_digest is not None and bool(str(baseline_digest).strip())
        has_comparison_digest = comparison_digest is not None and bool(str(comparison_digest).strip())
        if not has_output_digest and not (has_baseline_digest and has_comparison_digest):
            failures.append("deterministic_output_gate.passes=true requires output_digest")
        if has_baseline_digest != has_comparison_digest:
            failures.append(
                "deterministic_output_gate.passes=true requires both baseline and comparison digests"
            )
        if has_output_digest and has_comparison_digest and str(output_digest).strip() != str(comparison_digest).strip():
            failures.append("deterministic_output_gate.output_digest must match comparison_output_digest")
    else:
        raw_failures = raw.get("failures")
        if raw_failures is not None:
            try:
                notes.extend(str(failure) for failure in raw_failures)  # type: ignore[arg-type]
            except TypeError as exc:
                raise ValueError("deterministic_output_gate.failures must be an iterable") from exc
        notes.append("deterministic residual/output parity is not proven")

    raw_notes = raw.get("notes")
    if raw_notes is not None:
        try:
            notes.extend(str(note) for note in raw_notes)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError("deterministic_output_gate.notes must be an iterable") from exc
    evidence_source = raw.get("evidence_source")
    if evidence_source is not None:
        notes.append(f"deterministic output evidence_source={evidence_source}")

    return passes and not failures, tuple(failures), tuple(notes)


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

    min_speedup_value = _speedup_threshold(min_speedup, name="min_speedup")
    min_efficiency_value = _efficiency_threshold(min_efficiency, name="min_efficiency")
    min_parallel_workers_value = validate_transport_parallel_worker_count(
        min_parallel_workers,
        context="minimum parallel",
    )

    benchmark_kind = _normalized_benchmark_kind(summary, default="transport_worker_scaling")
    if benchmark_kind in {"single_case_sharded_solve", "sharded_solve_scaling", "sharded_solve"}:
        raise ValueError(
            "single-case sharded-solve summaries must use audit_sharded_solve_scaling_summary; "
            "they are not transport-worker scaling claims"
        )
    if benchmark_kind not in {"transport_worker_scaling", "transport_parallel_scaling", "whichrhs_worker_scaling"}:
        raise ValueError(f"unsupported transport-worker benchmark_kind={benchmark_kind!r}")

    backend = str(summary.get("backend", "cpu")).strip().lower() or "cpu"
    task_count = _scaling_task_count(summary)
    device_count, device_note = _scaling_device_count(summary, backend=backend)
    timing_semantics, timing_note = _timing_semantics(summary)
    requested_release = _optional_bool(summary.get("release_scaling_claim"), name="release_scaling_claim")

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

    deterministic_payload_coverage, coverage_note = _deterministic_payload_coverage(
        summary,
        task_count=task_count,
        claim_workers=claim_workers,
    )
    deterministic_output_check, output_note = _deterministic_output_check(
        summary,
        deterministic_payload_coverage=deterministic_payload_coverage,
    )
    compile_gate_recorded, compile_gate_ok, compile_failures, compile_notes = _compile_amortization_gate(
        summary,
        timing_semantics=timing_semantics,
    )

    failures: list[str] = []
    notes: list[str] = []
    if task_count < min_parallel_workers_value:
        failures.append(
            f"only {task_count} independent transport tasks; need at least {min_parallel_workers_value}"
        )
    if claim_workers < min_parallel_workers_value:
        failures.append(
            f"only {claim_workers} worker count audited; need at least {min_parallel_workers_value}"
        )
    if claim_workers > task_count:
        failures.append(f"{claim_workers} workers cannot be claimed for only {task_count} independent transport tasks")
    if backend == "gpu" and device_count is None:
        failures.append("GPU device count was not recorded")
    if backend == "gpu" and device_count is not None and device_count < claim_workers:
        failures.append(f"only {device_count} GPU devices recorded for {claim_workers} workers")
    if timing_semantics is None:
        failures.append("timing semantics were not recorded")
    elif timing_semantics in {"cold", "cold_start", "cold_cache"}:
        failures.append(f"timing semantics {timing_semantics!r} include cold setup and cannot support a warm scaling claim")
    elif timing_semantics in {"mixed", "mixed_warm_cold"}:
        failures.append("mixed warm/cold timing semantics cannot support a release scaling claim")
    if claim_speedup < min_speedup_value:
        failures.append(f"speedup {claim_speedup:.3g}x is below release gate {min_speedup_value:.3g}x")
    if claim_efficiency < min_efficiency_value:
        failures.append(f"efficiency {claim_efficiency:.3g} is below release gate {min_efficiency_value:.3g}")
    if not deterministic_payload_coverage:
        failures.append("deterministic payload coverage was not proven")
    if not deterministic_output_check:
        failures.append("deterministic output check was not proven")
    if compile_gate_recorded:
        failures.extend(compile_failures)
    elif requested_release is True:
        failures.append("explicit release_scaling_claim=true requires compile-amortization gate metadata")
    if claim_speedup > claim_finite_task_ideal_speedup * 1.05:
        failures.append(
            f"speedup {claim_speedup:.3g}x exceeds finite-task ideal {claim_finite_task_ideal_speedup:.3g}x by more than 5%"
        )

    if device_note is not None:
        notes.append(device_note)
    if timing_note is not None:
        notes.append(timing_note)
    if coverage_note is not None:
        notes.append(coverage_note)
    if output_note is not None:
        notes.append(output_note)
    notes.extend(compile_notes)
    if requested_release is False:
        notes.append("summary explicitly records release_scaling_claim=false")
    if task_count < claim_workers:
        notes.append(f"{task_count} transport tasks cannot keep {claim_workers} workers fully occupied")

    release_scaling_claim = not failures and requested_release is not False
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
        release_scaling_claim=release_scaling_claim,
        failures=tuple(failures),
        notes=tuple(notes),
        benchmark_kind=benchmark_kind,
        timing_semantics=timing_semantics,
        deterministic_output_check=deterministic_output_check,
        compile_amortization_gate=compile_gate_ok,
        min_speedup=min_speedup_value,
        min_efficiency=min_efficiency_value,
        min_parallel_workers=min_parallel_workers_value,
    )


def _wall_time_from_mapping(summary: dict[str, object], *, key: str) -> float:
    raw = summary.get(key)
    if not isinstance(raw, dict):
        raise ValueError(f"{key} must be a dictionary containing wall_s")
    return _finite_positive_float(raw.get("wall_s"), name=f"{key}.wall_s")


def _positive_int_sequence(summary: dict[str, object], *, keys: tuple[str, ...]) -> tuple[int, ...]:
    for key in keys:
        raw = summary.get(key)
        if raw is None:
            continue
        if isinstance(raw, (str, bytes)):
            values = [raw]
        else:
            try:
                values = list(raw)  # type: ignore[arg-type]
            except TypeError as exc:
                raise ValueError(f"{key} must be an iterable of positive integers") from exc
        parsed = tuple(_optional_positive_int(value, name=key) for value in values)
        if any(value is None for value in parsed):
            raise ValueError(f"{key} must contain positive integers")
        return tuple(int(value) for value in parsed if value is not None)
    return ()


def _parallel_count_from_results(
    summary: dict[str, object],
    *,
    result_key: str,
) -> tuple[int, ...]:
    raw_results = summary.get("results")
    if not isinstance(raw_results, list):
        return ()
    values: list[int] = []
    for i, result in enumerate(raw_results):
        if not isinstance(result, dict) or result_key not in result:
            continue
        values.append(
            validate_transport_parallel_worker_count(
                result[result_key],
                context=f"results[{i}]",
            )
        )
    return tuple(values)


def _parallel_scope_metadata(
    summary: dict[str, object],
) -> tuple[str, str | None, bool | None, bool, bool]:
    backend = str(summary.get("backend", "unspecified")).strip().lower().replace("-", "_") or "unspecified"
    artifact_kind_raw = summary.get("artifact_kind")
    artifact_kind = (
        str(artifact_kind_raw).strip().lower().replace("-", "_")
        if artifact_kind_raw is not None
        else None
    )
    launches_solves = (
        _optional_bool(summary.get("launches_solves"), name="launches_solves")
        if "launches_solves" in summary
        else None
    )
    measured_results_present = isinstance(summary.get("results"), list) and bool(summary.get("results"))
    plan_only = artifact_kind == "benchmark_plan" or launches_solves is False
    return backend, artifact_kind, launches_solves, plan_only, measured_results_present


def _declared_parallel_claim_scope(summary: dict[str, object]) -> str | None:
    raw_scope = summary.get("claim_scope")
    if raw_scope is None:
        raw_nested = summary.get("parallel_claim_scope")
        if isinstance(raw_nested, dict):
            raw_scope = raw_nested.get("claim_scope")
    if raw_scope is None:
        return None
    return str(raw_scope).strip().lower().replace("-", "_")


def _append_common_parallel_scope_failures(
    *,
    failures: list[str],
    notes: list[str],
    expected_claim_scope: str,
    declared_claim_scope: str | None,
    explicit_benchmark_kind: bool,
    requested_release: bool,
    artifact_kind: str | None,
    launches_solves: bool | None,
    plan_only: bool,
    measured_results_present: bool,
) -> None:
    if not explicit_benchmark_kind:
        failures.append("parallel scaling scope audit requires explicit benchmark_kind metadata")
    if declared_claim_scope is not None and declared_claim_scope != expected_claim_scope:
        failures.append(
            f"declared claim_scope={declared_claim_scope!r} conflicts with "
            f"benchmark_kind claim_scope={expected_claim_scope!r}"
        )
    if plan_only and requested_release:
        failures.append("plan-only parallel benchmark artifacts cannot set release_scaling_claim=true")
    if plan_only and measured_results_present:
        failures.append("plan-only parallel benchmark artifacts must not include measured timing results")
    if artifact_kind is not None:
        notes.append(f"artifact_kind={artifact_kind}")
    if launches_solves is not None:
        notes.append(f"launches_solves={launches_solves}")
    if plan_only:
        notes.append("plan-only artifact checks launch scope; measured speedup and memory gates still require a run")


def audit_parallel_scaling_claim_scope(summary: dict[str, object]) -> ParallelScalingClaimScopeAudit:
    """Audit the semantic scope of a parallel benchmark without running solves.

    The release-facing scaling claim in ``sfincs_jax`` is independent-work
    throughput: transport ``whichRHS`` columns, scan points, or separate cases
    can run concurrently. A single RHSMode=1 linear system split across devices
    is a different claim. This helper fails closed when a saved artifact or
    benchmark plan tries to promote that single-case path as release-grade
    strong scaling.
    """
    if not isinstance(summary, dict):
        raise ValueError("parallel scaling claim scope summary must be a dictionary")

    explicit_benchmark_kind = "benchmark_kind" in summary
    benchmark_kind = _normalized_benchmark_kind(summary, default="transport_worker_scaling")
    requested_release = bool(_optional_bool(summary.get("release_scaling_claim"), name="release_scaling_claim") or False)
    failures: list[str] = []
    notes: list[str] = []
    backend, artifact_kind, launches_solves, plan_only, measured_results_present = _parallel_scope_metadata(summary)
    declared_claim_scope = _declared_parallel_claim_scope(summary)

    if benchmark_kind in {"transport_worker_scaling", "transport_parallel_scaling", "whichrhs_worker_scaling"}:
        expected_claim_scope = "independent_transport_worker_throughput"
        _append_common_parallel_scope_failures(
            failures=failures,
            notes=notes,
            expected_claim_scope=expected_claim_scope,
            declared_claim_scope=declared_claim_scope,
            explicit_benchmark_kind=explicit_benchmark_kind,
            requested_release=requested_release,
            artifact_kind=artifact_kind,
            launches_solves=launches_solves,
            plan_only=plan_only,
            measured_results_present=measured_results_present,
        )
        task_count = _scaling_task_count(summary)
        worker_counts = _positive_int_sequence(summary, keys=("workers", "requested_workers"))
        if not worker_counts:
            worker_counts = _parallel_count_from_results(summary, result_key="workers")
        parallel_count = max(worker_counts) if worker_counts else 1
        if backend not in {"cpu", "gpu"}:
            failures.append("transport-worker scope must record backend='cpu' or backend='gpu'")
        if parallel_count > task_count:
            failures.append(
                f"{parallel_count} transport workers cannot support a release claim with only "
                f"{task_count} independent whichRHS tasks"
            )
        if parallel_count < 2:
            failures.append("transport-worker release scaling needs at least two workers")
        if requested_release and not measured_results_present:
            failures.append("transport-worker release_scaling_claim=true requires measured timing results")
        scope_release_eligible = not failures
        release_supported = scope_release_eligible and measured_results_present and not plan_only
        notes.append("transport-worker scaling is independent whichRHS throughput")
        return ParallelScalingClaimScopeAudit(
            benchmark_kind=benchmark_kind,
            claim_scope=expected_claim_scope,
            independent_task_count=int(task_count),
            parallel_count=int(parallel_count),
            backend=backend,
            artifact_kind=artifact_kind,
            launches_solves=launches_solves,
            plan_only_scope_evidence=plan_only,
            measured_results_present=measured_results_present,
            release_gate_required="audit_transport_parallel_scaling_summary",
            claim_scope_release_eligible=scope_release_eligible,
            release_scaling_supported=release_supported,
            unsupported_single_case_strong_scaling=False,
            failures=tuple(failures),
            notes=tuple(notes),
        )

    if benchmark_kind in {"single_case_sharded_solve", "sharded_solve_scaling", "sharded_solve"}:
        expected_claim_scope = "single_case_sharded_solve_experimental"
        _append_common_parallel_scope_failures(
            failures=failures,
            notes=notes,
            expected_claim_scope=expected_claim_scope,
            declared_claim_scope=declared_claim_scope,
            explicit_benchmark_kind=explicit_benchmark_kind,
            requested_release=requested_release,
            artifact_kind=artifact_kind,
            launches_solves=launches_solves,
            plan_only=plan_only,
            measured_results_present=measured_results_present,
        )
        device_counts = _positive_int_sequence(summary, keys=("devices", "device_counts", "requested_devices"))
        if not device_counts:
            device_counts = _parallel_count_from_results(summary, result_key="devices")
        parallel_count = max(device_counts) if device_counts else 1
        experimental_marker = _optional_bool(
            summary.get("experimental_single_case_scaling"),
            name="experimental_single_case_scaling",
        )
        scaling_status = str(summary.get("scaling_status", "")).strip().lower().replace("-", "_")
        marked_experimental = bool(experimental_marker) or scaling_status in {
            "experimental",
            "experimental_single_case_sharding",
            "regression_snapshot",
            "non_release_snapshot",
        }
        if requested_release:
            failures.append("single-case sharded solve cannot set release_scaling_claim=true")
        if not marked_experimental:
            failures.append("single-case sharded solve must be marked experimental/non-release")
        if parallel_count < 2:
            failures.append("single-case sharded solve needs at least two devices to demonstrate sharding")
        notes.append("single-case sharded solve is not independent throughput")
        notes.append("this helper never promotes single-case sharding to release strong-scaling evidence")
        return ParallelScalingClaimScopeAudit(
            benchmark_kind=benchmark_kind,
            claim_scope=expected_claim_scope,
            independent_task_count=1,
            parallel_count=int(parallel_count),
            backend=backend,
            artifact_kind=artifact_kind,
            launches_solves=launches_solves,
            plan_only_scope_evidence=plan_only,
            measured_results_present=measured_results_present,
            release_gate_required=None,
            claim_scope_release_eligible=not failures,
            release_scaling_supported=False,
            unsupported_single_case_strong_scaling=parallel_count > 1,
            failures=tuple(failures),
            notes=tuple(notes),
        )

    if benchmark_kind in {"multi_gpu_case_throughput", "gpu_case_throughput"}:
        expected_claim_scope = "independent_case_throughput_non_release"
        _append_common_parallel_scope_failures(
            failures=failures,
            notes=notes,
            expected_claim_scope=expected_claim_scope,
            declared_claim_scope=declared_claim_scope,
            explicit_benchmark_kind=explicit_benchmark_kind,
            requested_release=requested_release,
            artifact_kind=artifact_kind,
            launches_solves=launches_solves,
            plan_only=plan_only,
            measured_results_present=measured_results_present,
        )
        required_gpu_count = _optional_positive_int(
            summary.get("required_gpu_count", 2),
            name="required_gpu_count",
        )
        parallel_count = int(required_gpu_count or 2)
        if requested_release:
            failures.append("multi-GPU case-throughput artifacts must not set release_scaling_claim=true")
        if backend != "gpu":
            failures.append("multi-GPU case-throughput scope must record backend='gpu'")
        if parallel_count < 2:
            failures.append("multi-GPU case throughput needs at least two GPU workers/cases")
        notes.append("multi-GPU case throughput covers independent cases, not a single sharded solve")
        return ParallelScalingClaimScopeAudit(
            benchmark_kind=benchmark_kind,
            claim_scope=expected_claim_scope,
            independent_task_count=parallel_count,
            parallel_count=parallel_count,
            backend=backend,
            artifact_kind=artifact_kind,
            launches_solves=launches_solves,
            plan_only_scope_evidence=plan_only,
            measured_results_present=measured_results_present,
            release_gate_required="audit_multi_gpu_case_throughput_summary",
            claim_scope_release_eligible=not failures,
            release_scaling_supported=False,
            unsupported_single_case_strong_scaling=False,
            failures=tuple(failures),
            notes=tuple(notes),
        )

    raise ValueError(f"unsupported parallel benchmark_kind={benchmark_kind!r}")


def audit_multi_gpu_case_throughput_summary(
    summary: dict[str, object],
    *,
    min_throughput_speedup: float = 1.0,
) -> MultiGpuCaseThroughputAudit:
    """Audit two-case multi-GPU throughput evidence without making a release claim."""
    if not isinstance(summary, dict):
        raise ValueError("multi-GPU case-throughput summary must be a dictionary")

    min_speedup_value = _speedup_threshold(min_throughput_speedup, name="min_throughput_speedup")
    benchmark_kind = _normalized_benchmark_kind(summary, default="multi_gpu_case_throughput")
    if benchmark_kind not in {"multi_gpu_case_throughput", "gpu_case_throughput"}:
        raise ValueError(f"unsupported multi-GPU throughput benchmark_kind={benchmark_kind!r}")

    backend = str(summary.get("backend", "gpu")).strip().lower() or "gpu"
    timing_semantics, timing_note = _timing_semantics(summary)
    required_gpu_count = int(
        _optional_positive_int(summary.get("required_gpu_count", 2), name="required_gpu_count") or 2
    )
    requested_release = bool(_optional_bool(summary.get("release_scaling_claim"), name="release_scaling_claim") or False)
    sequential_wall_s = _wall_time_from_mapping(summary, key="sequential_one_gpu")
    parallel_wall_s = _wall_time_from_mapping(summary, key="parallel_two_gpu")

    raw_speedup = summary.get("throughput_speedup")
    computed_speedup = sequential_wall_s / parallel_wall_s
    throughput_speedup = (
        _finite_positive_float(raw_speedup, name="throughput_speedup")
        if raw_speedup is not None
        else computed_speedup
    )

    failures: list[str] = []
    notes: list[str] = []
    if requested_release:
        failures.append("multi-GPU case-throughput summaries must not set release_scaling_claim=true")
    if backend != "gpu":
        failures.append(f"multi-GPU case-throughput backend must be 'gpu'; got {backend!r}")
    if required_gpu_count < 2:
        failures.append(f"required_gpu_count={required_gpu_count} cannot prove two-GPU throughput")
    if timing_semantics is None:
        failures.append("timing semantics were not recorded")
    elif timing_semantics in {"cold", "cold_start", "cold_cache", "mixed", "mixed_warm_cold"}:
        failures.append(f"timing semantics {timing_semantics!r} cannot support throughput evidence")
    if throughput_speedup < min_speedup_value:
        failures.append(
            f"throughput speedup {throughput_speedup:.3g}x is below evidence gate {min_speedup_value:.3g}x"
        )
    if abs(throughput_speedup - computed_speedup) > max(1.0e-9, abs(computed_speedup) * 1.0e-6):
        failures.append(
            f"throughput speedup {throughput_speedup:.3g}x does not match wall-time ratio {computed_speedup:.3g}x"
        )
    if timing_note is not None:
        notes.append(timing_note)
    notes.append("multi-GPU case throughput is evidence for batched case throughput, not single-case strong scaling")
    notes.append("this audit never converts throughput evidence into a release transport scaling claim")

    return MultiGpuCaseThroughputAudit(
        backend=backend,
        required_gpu_count=required_gpu_count,
        sequential_wall_s=sequential_wall_s,
        parallel_wall_s=parallel_wall_s,
        throughput_speedup=throughput_speedup,
        release_scaling_claim=False,
        ci_gate_pass=not failures,
        failures=tuple(failures),
        notes=tuple(notes),
        benchmark_kind=benchmark_kind,
        timing_semantics=timing_semantics,
        min_throughput_speedup=min_speedup_value,
    )


def audit_sharded_solve_scaling_summary(
    summary: dict[str, object],
    *,
    min_parallel_devices: int = 2,
) -> ShardedSolveScalingAudit:
    """Audit a single-case sharded-solve benchmark for schema honesty.

    This intentionally does not mint a release scaling claim. A sharded
    RHSMode=1 benchmark is one coupled solve spread across devices, so it is a
    different class from independent transport-worker throughput.
    """
    if not isinstance(summary, dict):
        raise ValueError("sharded-solve scaling summary must be a dictionary")

    min_parallel_devices_value = validate_transport_parallel_worker_count(
        min_parallel_devices,
        context="minimum parallel device",
    )
    benchmark_kind = _normalized_benchmark_kind(summary, default="single_case_sharded_solve")
    if benchmark_kind in {"transport_worker_scaling", "transport_parallel_scaling", "whichrhs_worker_scaling"}:
        raise ValueError(
            "transport-worker summaries must use audit_transport_parallel_scaling_summary; "
            "they are not single-case sharded-solve scaling claims"
        )
    if benchmark_kind not in {"single_case_sharded_solve", "sharded_solve_scaling", "sharded_solve"}:
        raise ValueError(f"unsupported sharded-solve benchmark_kind={benchmark_kind!r}")

    backend = str(summary.get("backend", "cpu")).strip().lower() or "cpu"
    timing_semantics, timing_note = _timing_semantics(summary)
    legacy_deterministic_output_check = bool(
        _optional_bool(summary.get("deterministic_output_check"), name="deterministic_output_check") or False
    )
    deterministic_output_probe_requested = bool(
        _optional_bool(
            summary.get("deterministic_output_probe_requested"),
            name="deterministic_output_probe_requested",
        )
        or False
    )
    requested_release = bool(_optional_bool(summary.get("release_scaling_claim"), name="release_scaling_claim") or False)

    raw_results = summary.get("results")
    if not isinstance(raw_results, list) or not raw_results:
        raise ValueError("sharded-solve scaling summary must include a non-empty results list")

    results: dict[int, dict[str, float]] = {}
    for i, result in enumerate(raw_results):
        if not isinstance(result, dict):
            raise ValueError(f"results[{i}] must be a dictionary")
        devices = validate_transport_parallel_worker_count(result.get("devices"), context=f"results[{i}]")
        mean_s = _finite_positive_float(result.get("mean_s"), name=f"results[{i}].mean_s")
        speedup_raw = result.get("speedup")
        speedup = _finite_positive_float(speedup_raw, name=f"results[{i}].speedup") if speedup_raw is not None else math.nan
        results[devices] = {"mean_s": mean_s, "speedup": speedup}

    if 1 not in results:
        raise ValueError("sharded-solve scaling summary must include a 1-device baseline")
    baseline_s = results[1]["mean_s"]
    for devices, result in results.items():
        if math.isnan(result["speedup"]):
            result["speedup"] = baseline_s / result["mean_s"]

    device_counts = tuple(sorted(results))
    claim_devices = max(device_counts)
    claim_speedup = results[claim_devices]["speedup"]
    claim_efficiency = claim_speedup / float(claim_devices)

    experimental_marker = _optional_bool(
        summary.get("experimental_single_case_scaling"),
        name="experimental_single_case_scaling",
    )
    scaling_status = str(summary.get("scaling_status", "")).strip().lower().replace("-", "_")
    experimental_single_case_scaling = bool(experimental_marker) or scaling_status in {
        "experimental",
        "experimental_single_case_sharding",
        "regression_snapshot",
        "non_release_snapshot",
    }
    operator_gate_recorded, operator_gate_ok, operator_failures, operator_notes = _operator_reuse_gate(
        summary,
        timing_semantics=timing_semantics,
    )
    deterministic_gate_ok, deterministic_failures, deterministic_notes = (
        _sharded_deterministic_output_gate(summary)
    )
    deterministic_output_check = bool(legacy_deterministic_output_check or deterministic_gate_ok)

    failures: list[str] = []
    notes: list[str] = []
    release_promotion_blockers: list[str] = [
        "single-case sharded solve is experimental and not independent transport throughput"
    ]
    if claim_devices < min_parallel_devices_value:
        failures.append(f"only {claim_devices} device count audited; need at least {min_parallel_devices_value}")
        release_promotion_blockers.append(
            f"only {claim_devices} device count audited; need at least {min_parallel_devices_value}"
        )
    if requested_release:
        failures.append("single-case sharded-solve summaries must not set release_scaling_claim=true")
        release_promotion_blockers.append("artifact requested release_scaling_claim=true")
    else:
        release_promotion_blockers.append("artifact records release_scaling_claim=false")
    if not experimental_single_case_scaling:
        failures.append("single-case sharded-solve summary must be marked experimental/non-release")
        release_promotion_blockers.append("artifact is not marked experimental/non-release")
    if timing_semantics is None:
        failures.append("timing semantics were not recorded")
        release_promotion_blockers.append("timing semantics were not recorded")
    elif timing_semantics in {"cold", "cold_start", "cold_cache"}:
        notes.append(f"timing semantics {timing_semantics!r} include cold setup")
        release_promotion_blockers.append(
            f"timing semantics {timing_semantics!r} include cold setup"
        )
    if operator_gate_recorded:
        failures.extend(operator_failures)
        if not operator_gate_ok:
            release_promotion_blockers.append("compiled sharded operator-reuse gate did not pass")
    else:
        failures.append("compiled sharded operator-reuse gate metadata was not recorded")
        release_promotion_blockers.append(
            "compiled sharded operator-reuse gate metadata was not recorded"
        )
    if backend == "gpu":
        device_count, device_note = _scaling_device_count(summary, backend=backend)
        if device_note is not None:
            notes.append(device_note)
        if device_count is not None and device_count < claim_devices:
            failures.append(f"only {device_count} GPU devices recorded for {claim_devices} sharded devices")
            release_promotion_blockers.append(
                f"only {device_count} GPU devices recorded for {claim_devices} sharded devices"
            )
    for result in raw_results:
        if not isinstance(result, dict):
            continue
        devices = validate_transport_parallel_worker_count(result.get("devices"), context="results")
        sample_failures = result.get("sample_failures", result.get("failures", ()))
        try:
            failure_list = tuple(str(item) for item in sample_failures)  # type: ignore[arg-type]
        except TypeError:
            failure_list = (str(sample_failures),)
        timed_out = bool(_optional_bool(result.get("timed_out"), name="results.timed_out") or False)
        failed_samples = result.get("failed_samples")
        failed_sample_count = 0
        if failed_samples is not None:
            try:
                failed_sample_count = len(tuple(failed_samples))  # type: ignore[arg-type]
            except TypeError:
                failed_sample_count = 1
        if timed_out or failure_list or failed_sample_count:
            failures.append(f"devices={devices} recorded failed/timed-out sharded-solve samples")
            release_promotion_blockers.append(
                f"devices={devices} recorded failed/timed-out sharded-solve samples"
            )
    if not deterministic_output_check:
        notes.append("deterministic output parity was not recorded for this timing-only sharded benchmark")
        release_promotion_blockers.append("deterministic residual/output parity was not proven")
    if deterministic_output_probe_requested and not deterministic_gate_ok:
        failures.append("requested deterministic output probe did not pass")
        release_promotion_blockers.append("requested deterministic output probe did not pass")
    failures.extend(deterministic_failures)
    if deterministic_failures:
        release_promotion_blockers.append("deterministic residual/output gate metadata is invalid")
    elif not deterministic_gate_ok:
        release_promotion_blockers.append("deterministic residual/output gate did not pass")
    notes.extend(operator_notes)
    notes.extend(deterministic_notes)
    if timing_note is not None:
        notes.append(timing_note)
    notes.append("single-case sharded solve remains experimental and is not a release scaling claim")
    notes.append(
        "release promotion blocked: "
        + "; ".join(dict.fromkeys(release_promotion_blockers))
    )

    return ShardedSolveScalingAudit(
        backend=backend,
        device_counts=device_counts,
        baseline_s=baseline_s,
        claim_devices=claim_devices,
        claim_speedup=claim_speedup,
        claim_efficiency=claim_efficiency,
        release_scaling_claim=False,
        experimental_single_case_scaling=experimental_single_case_scaling,
        ci_gate_pass=not failures,
        failures=tuple(failures),
        notes=tuple(notes),
        benchmark_kind=benchmark_kind,
        timing_semantics=timing_semantics,
        deterministic_output_check=deterministic_output_check,
        operator_reuse_gate=operator_gate_ok,
        deterministic_output_gate=deterministic_gate_ok,
        release_promotion_supported=False,
        release_promotion_blockers=tuple(dict.fromkeys(release_promotion_blockers)),
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
    """Resolve the multiprocessing start method used by transport workers."""
    env = os.environ.get("SFINCS_JAX_TRANSPORT_MP_START_METHOD", "").strip().lower()
    if env in {"", "auto"}:
        return "spawn"
    if env in {"spawn", "fork", "forkserver"}:
        return env
    return "spawn"


def transport_parallel_backend() -> str:
    """Resolve whether transport worker parallelism uses CPU or GPU workers."""
    env = os.environ.get("SFINCS_JAX_TRANSPORT_PARALLEL_BACKEND", "").strip().lower()
    if env in {"", "auto", "cpu", "process"}:
        return "cpu"
    if env in {"gpu", "gpu_process", "process_gpu"}:
        return "gpu"
    return "cpu"


def transport_parallel_persistent_pool_enabled() -> bool:
    """Return whether CPU transport workers may reuse a persistent process pool."""
    env = os.environ.get("SFINCS_JAX_TRANSPORT_POOL_PERSIST", "").strip().lower()
    return env not in {"0", "false", "no", "off"}


def transport_parallel_pool_key(parallel_workers: int) -> tuple[object, ...]:
    """Return the cache key for a persistent transport worker pool."""
    return (
        validate_transport_parallel_worker_count(parallel_workers),
        transport_parallel_backend(),
        transport_parallel_start_method(),
        os.environ.get("SFINCS_JAX_TRANSPORT_PIN_THREADS", "").strip().lower(),
        os.environ.get("SFINCS_JAX_CORES", "").strip(),
    )


def transport_parallel_visible_gpu_ids(parallel_workers: int) -> list[str]:
    """Return the visible GPU ids assigned to GPU transport workers."""
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
    """Return an isolated environment for one GPU transport child process."""
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
    """Build ``ProcessPoolExecutor`` kwargs with a safe spawn fallback."""
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
