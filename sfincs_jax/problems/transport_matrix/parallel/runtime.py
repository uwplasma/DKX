"""Transport-matrix parallel policy, sharding plans, and runtime orchestration."""

from __future__ import annotations

from collections.abc import Callable, Iterator
import contextlib
from dataclasses import dataclass
import math
import os
from dataclasses import asdict
from collections.abc import Mapping, Sequence
import atexit
import concurrent.futures
from concurrent.futures.process import BrokenProcessPool
import json
import multiprocessing as mp
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
from sfincs_jax.problems.transport_matrix.policies import (
    transport_residual_gate_failures_from_arrays,
    transport_residual_gate_thresholds_from_env,
)
from sfincs_jax.problems.transport_matrix.finalize import V3TransportMatrixSolveResult
from sfincs_jax.discretization.v3 import geometry_from_namelist, grids_from_namelist
from sfincs_jax.operators.profile_response.system import V3FullSystemOperator

# --- Parallel policy, scaling audits, and worker environments ---


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
def _transport_parallel_worker_env(
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


def _transport_parallel_pool_executor_kwargs(
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


# --- Single-case sharding plans and promotion gates ---


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
_COMPILED_SHARDED_OPERATOR_KINDS = _SHARDED_SOLVE_KINDS | {
    "sharded_matvec_scaling",
    "sharded_matvec",
}
_WARM_OPERATOR_TIMING_SEMANTICS = {
    "cache_warm",
    "compiled_matvec_hot_loop",
    "hot",
    "hot_run",
    "hot_solve",
    "warm",
    "warm_cache",
}


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


@dataclass(frozen=True)
class CompiledShardedOperatorReuseGate:
    """Deterministic metadata for compiled sharded operator reuse."""

    benchmark_kind: str
    timing_semantics: str
    strategy: str
    persistent_compile_cache: bool
    compile_cache_dir: str | None
    cache_required: bool
    global_warmup_runs: int
    per_device_warmup_runs: int
    inner_warmup_runs: int
    timed_repeats: int
    min_timed_repeats: int
    work_units_per_sample: int
    compile_in_timed_region: bool
    passes: bool
    warm_run_amortization_pass: bool
    failures: tuple[str, ...]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable gate dictionary."""

        return asdict(self)


@dataclass(frozen=True)
class ShardedSolveDeterministicOutputGate:
    """Portable schema for residual/output parity before promoting sharded runs."""

    schema_version: int
    status: str
    passes: bool
    baseline_devices: int
    comparison_devices: int
    residual_tolerance: float
    max_relative_residual_norm: float | None
    output_digest_algorithm: str
    output_digest: str | None
    failures: tuple[str, ...]
    notes: tuple[str, ...]
    baseline_output_digest: str | None = None
    comparison_output_digest: str | None = None
    output_digest_match: bool | None = None
    evidence_source: str = "not_measured"

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable gate dictionary."""

        return asdict(self)


@dataclass(frozen=True)
class SingleCaseOperatorCoarseReusePlan:
    """Fail-closed architecture plan for single-case compiled operator reuse."""

    schema_version: int
    benchmark_kind: str
    backend: str
    rhs_mode: int
    shard_axis: str
    active_devices: int
    experimental_single_case_scaling: bool
    operator_reuse_enabled: bool
    operator_reuse_gate_pass: bool
    deterministic_output_gate_pass: bool
    measured_hot_speedup: float | None
    min_hot_speedup: float
    memory_growth_fraction: float | None
    max_memory_growth_fraction: float
    coarse_strategy: str
    coarse_levels: int
    max_coarse_rank: int | None
    operator_build_scope: str
    operator_action_scope: str
    preconditioner_scope: str
    coarse_operator_scope: str
    coarse_solve_scope: str
    compiled_components: tuple[str, ...]
    reused_components: tuple[str, ...]
    per_device_components: tuple[str, ...]
    replicated_components: tuple[str, ...]
    required_runtime_gates: tuple[str, ...]
    plan_valid: bool
    promotion_ready: bool
    failures: tuple[str, ...]
    promotion_blockers: tuple[str, ...]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable architecture plan."""

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


def _optional_bool_value(value: object, *, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = _normalized_token(value)
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


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


def plan_compiled_sharded_operator_reuse(
    *,
    benchmark_kind: str,
    timing_semantics: str,
    global_warmup_runs: int = 0,
    per_device_warmup_runs: int = 0,
    inner_warmup_runs: int = 0,
    timed_repeats: int = 1,
    work_units_per_sample: int = 1,
    compile_cache_dir: str | None = None,
    persistent_compile_cache: bool | None = None,
    compile_in_timed_region: bool = False,
    min_timed_repeats: int = 1,
    require_cache_for_global_warmup: bool = True,
) -> CompiledShardedOperatorReuseGate:
    """Plan a fail-closed compiled-operator reuse gate for sharded benchmarks.

    The gate is pure metadata. It records whether timed samples are intended to
    reuse an already-compiled sharded operator via an inner/per-device warmup or
    via a persistent compilation cache populated by a global warmup.
    """

    kind = _normalized_token(benchmark_kind)
    timing = _normalized_token(timing_semantics)
    global_warmup = _nonnegative_int(global_warmup_runs, name="global_warmup_runs")
    per_device_warmup = _nonnegative_int(
        per_device_warmup_runs,
        name="per_device_warmup_runs",
    )
    inner_warmup = _nonnegative_int(inner_warmup_runs, name="inner_warmup_runs")
    repeats = _positive_int(timed_repeats, name="timed_repeats")
    min_repeats = _positive_int(min_timed_repeats, name="min_timed_repeats")
    work_units = _positive_int(work_units_per_sample, name="work_units_per_sample")
    cache_dir = None if compile_cache_dir is None else str(compile_cache_dir).strip() or None
    persistent_cache = _optional_bool_value(
        persistent_compile_cache,
        default=cache_dir is not None,
    )
    compile_inside = _optional_bool_value(compile_in_timed_region, default=False)

    if inner_warmup > 0:
        strategy = "inner_warmup"
    elif per_device_warmup > 0:
        strategy = "per_device_warmup"
    elif global_warmup > 0:
        strategy = "global_persistent_compile_cache"
    else:
        strategy = "none"

    cache_required = bool(
        require_cache_for_global_warmup
        and strategy == "global_persistent_compile_cache"
    )
    warm_run_amortized = bool(
        strategy in {"inner_warmup", "per_device_warmup"}
        or (strategy == "global_persistent_compile_cache" and persistent_cache and cache_dir)
    )

    failures: list[str] = []
    notes: list[str] = []
    if kind not in _COMPILED_SHARDED_OPERATOR_KINDS:
        failures.append(f"unsupported compiled sharded operator benchmark_kind={kind!r}")
    if timing not in _WARM_OPERATOR_TIMING_SEMANTICS:
        failures.append(f"timing semantics {timing!r} do not describe a warm compiled operator")
    if repeats < min_repeats:
        failures.append(f"only {repeats} timed repeats recorded; need at least {min_repeats}")
    if compile_inside:
        failures.append("compiled sharded operator gate records compilation inside timed samples")
    if not warm_run_amortized:
        failures.append(
            "compiled sharded operator gate requires an inner/per-device warmup "
            "or a persistent-cache-backed global warmup"
        )
    if cache_required and not persistent_cache:
        failures.append("global warmup strategy requires persistent_compile_cache=true")
    if cache_required and cache_dir is None:
        failures.append("global warmup strategy requires compile_cache_dir metadata")
    if persistent_cache and cache_dir is None:
        failures.append("persistent_compile_cache=true requires compile_cache_dir metadata")
    if work_units == 1:
        notes.append("one work unit per timed sample; speedup evidence is sensitive to launch overhead")
    if strategy == "global_persistent_compile_cache":
        notes.append("global warmup depends on cross-process persistent compilation cache reuse")
    elif strategy != "none":
        notes.append(f"timed samples reuse operators via {strategy}")

    return CompiledShardedOperatorReuseGate(
        benchmark_kind=kind,
        timing_semantics=timing,
        strategy=strategy,
        persistent_compile_cache=bool(persistent_cache),
        compile_cache_dir=cache_dir,
        cache_required=cache_required,
        global_warmup_runs=global_warmup,
        per_device_warmup_runs=per_device_warmup,
        inner_warmup_runs=inner_warmup,
        timed_repeats=repeats,
        min_timed_repeats=min_repeats,
        work_units_per_sample=work_units,
        compile_in_timed_region=bool(compile_inside),
        passes=not failures,
        warm_run_amortization_pass=warm_run_amortized and not failures,
        failures=tuple(failures),
        notes=tuple(notes),
    )


def plan_sharded_solve_deterministic_output_gate(
    *,
    baseline_devices: int = 1,
    comparison_devices: int,
    residual_tolerance: float = 1.0e-10,
    max_relative_residual_norm: float | None = None,
    output_digest: str | None = None,
    baseline_output_digest: str | None = None,
    comparison_output_digest: str | None = None,
    output_digest_algorithm: str = "sha256",
    evidence_source: str = "not_measured",
) -> ShardedSolveDeterministicOutputGate:
    """Build a deterministic residual/output parity gate for sharded solve artifacts."""

    baseline = _positive_int(baseline_devices, name="baseline_devices")
    comparison = _positive_int(comparison_devices, name="comparison_devices")
    tolerance = float(residual_tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("residual_tolerance must be positive and finite")

    observed = None
    if max_relative_residual_norm is not None:
        observed = float(max_relative_residual_norm)

    digest = None if output_digest is None else str(output_digest).strip() or None
    baseline_digest = (
        None
        if baseline_output_digest is None
        else str(baseline_output_digest).strip() or None
    )
    comparison_digest = (
        None
        if comparison_output_digest is None
        else str(comparison_output_digest).strip() or None
    )
    if digest is None and comparison_digest is not None:
        digest = comparison_digest
    elif digest is None and baseline_digest is not None:
        digest = baseline_digest
    digest_algorithm = str(output_digest_algorithm).strip().lower() or "sha256"
    source = _normalized_token(evidence_source or "not_measured")
    digest_match = (
        baseline_digest == comparison_digest
        if baseline_digest is not None and comparison_digest is not None
        else None
    )
    failures: list[str] = []
    notes: list[str] = []
    if comparison < 2:
        failures.append("deterministic output gate needs at least two comparison devices")
    if observed is None:
        failures.append("max_relative_residual_norm was not measured")
    elif not math.isfinite(observed) or observed > tolerance:
        failures.append(
            f"max_relative_residual_norm {observed!r} exceeds tolerance {tolerance:g}"
        )
    if baseline_digest is None and comparison_digest is None and digest is None:
        failures.append("output digest was not recorded")
    if baseline_digest is not None or comparison_digest is not None:
        if baseline_digest is None:
            failures.append("baseline output digest was not recorded")
        if comparison_digest is None:
            failures.append("comparison output digest was not recorded")
        if digest_match is False:
            notes.append("baseline/comparison output digests differ; residual gate decides parity")
    if baseline == comparison:
        notes.append("baseline and comparison device counts are identical")

    status = "pass" if not failures else "not_measured" if observed is None and digest is None else "fail"
    if status == "not_measured":
        notes.append("deterministic output parity must be measured before a scaling claim")
    if source == "not_measured" and status == "pass":
        source = "measured"

    return ShardedSolveDeterministicOutputGate(
        schema_version=1,
        status=status,
        passes=not failures,
        baseline_devices=baseline,
        comparison_devices=comparison,
        residual_tolerance=tolerance,
        max_relative_residual_norm=observed,
        output_digest_algorithm=digest_algorithm,
        output_digest=digest,
        failures=tuple(failures),
        notes=tuple(notes),
        baseline_output_digest=baseline_digest,
        comparison_output_digest=comparison_digest,
        output_digest_match=digest_match,
        evidence_source=source,
    )


def plan_single_case_operator_coarse_reuse(
    *,
    active_devices: int,
    backend: str = "auto",
    rhs_mode: int = 1,
    shard_axis: str = "theta",
    benchmark_kind: str = "single_case_sharded_solve",
    experimental_single_case_scaling: bool = True,
    operator_reuse_enabled: bool = True,
    operator_reuse_gate_pass: bool = False,
    deterministic_output_gate_pass: bool = False,
    measured_hot_speedup: float | None = None,
    min_hot_speedup: float = 1.15,
    memory_growth_fraction: float | None = None,
    max_memory_growth_fraction: float = 0.0,
    coarse_strategy: str = "replicated_schur",
    coarse_levels: int = 0,
    max_coarse_rank: int | None = None,
) -> SingleCaseOperatorCoarseReusePlan:
    """Plan the next sharded RHSMode=1 operator/coarse-reuse architecture.

    The helper is intentionally pure. It does not claim speedup; it records the
    executable architecture that a measured single-case scaling artifact must
    use before being considered for promotion.
    """

    devices = _positive_int(active_devices, name="active_devices")
    rhs_mode_value = _positive_int(rhs_mode, name="rhs_mode")
    backend_norm = _normalize_backend(backend)
    axis = _normalized_token(shard_axis)
    kind = _normalized_token(benchmark_kind)
    coarse_level_count = _nonnegative_int(coarse_levels, name="coarse_levels")
    coarse_rank = (
        None
        if max_coarse_rank is None
        else _positive_int(max_coarse_rank, name="max_coarse_rank")
    )
    speedup_gate = float(min_hot_speedup)
    memory_gate = float(max_memory_growth_fraction)
    if not math.isfinite(speedup_gate) or speedup_gate < 1.0:
        raise ValueError("min_hot_speedup must be finite and >= 1.0")
    if not math.isfinite(memory_gate):
        raise ValueError("max_memory_growth_fraction must be finite")

    hot_speedup = None if measured_hot_speedup is None else float(measured_hot_speedup)
    memory_growth = (
        None if memory_growth_fraction is None else float(memory_growth_fraction)
    )

    failures: list[str] = []
    blockers: list[str] = []
    notes: list[str] = []
    if kind not in _SHARDED_SOLVE_KINDS:
        failures.append("operator/coarse reuse plan requires single-case sharded-solve benchmark_kind")
    if rhs_mode_value != 1:
        failures.append("operator/coarse reuse plan currently applies only to RHSMode=1")
    if axis not in _SUPPORTED_SINGLE_CASE_AXES:
        failures.append(
            f"operator/coarse reuse plan requires shard_axis in {sorted(_SUPPORTED_SINGLE_CASE_AXES)}"
        )
    if devices < 2:
        failures.append("operator/coarse reuse plan needs at least two active devices")
    if not bool(experimental_single_case_scaling):
        failures.append("single-case operator/coarse reuse must remain marked experimental")
    if not bool(operator_reuse_enabled):
        blockers.append("assembled operator reuse is disabled")
    if not bool(operator_reuse_gate_pass):
        blockers.append("compiled operator-reuse gate has not passed")
    if not bool(deterministic_output_gate_pass):
        blockers.append("deterministic 1-vs-N output gate has not passed")
    if hot_speedup is None:
        blockers.append("hot 1-vs-N speedup has not been measured")
    elif not math.isfinite(hot_speedup) or hot_speedup < speedup_gate:
        blockers.append(
            f"hot speedup {hot_speedup:.3g}x is below gate {speedup_gate:.3g}x"
        )
    if memory_growth is None:
        blockers.append("1-vs-N peak-memory growth has not been measured")
    elif not math.isfinite(memory_growth) or memory_growth > memory_gate:
        blockers.append(
            "peak-memory growth exceeds gate "
            f"({memory_growth:.3g} > {memory_gate:.3g})"
        )
    if coarse_level_count == 0:
        notes.append("coarse reuse is planned but no Schwarz/coarse levels are requested")
    if coarse_rank is None:
        notes.append("coarse rank is not capped in this plan; measured runs must record actual rank")

    compiled_components = (
        "sharded_full_system_matvec",
        "local_slab_preconditioner_apply",
        "coarse_projection",
        "coarse_correction_apply",
    )
    reused_components = (
        "full_system_operator_signature",
        "operator_constants",
        "local_preconditioner_blocks",
        "coarse_basis",
        "projected_coarse_operator",
    )
    per_device_components = (
        f"{axis}_slab_state",
        f"{axis}_slab_matvec_workspace",
        "local_preconditioner_workspace",
    )
    replicated_components = (
        "coarse_basis_metadata",
        "projected_coarse_operator",
        "coarse_solution_vector",
    )
    required_runtime_gates = (
        "compiled_operator_reuse_gate",
        "deterministic_output_gate",
        "hot_1_vs_n_speedup_gate",
        "peak_memory_nonincrease_gate",
    )

    valid = not failures
    ready = valid and not blockers
    if valid:
        notes.append(
            "planned path keeps the expensive operator/coarse setup outside timed hot solves"
        )
        notes.append("coarse solve is replicated; only slab residual data should cross devices")

    return SingleCaseOperatorCoarseReusePlan(
        schema_version=1,
        benchmark_kind=kind,
        backend=backend_norm,
        rhs_mode=rhs_mode_value,
        shard_axis=axis,
        active_devices=devices,
        experimental_single_case_scaling=bool(experimental_single_case_scaling),
        operator_reuse_enabled=bool(operator_reuse_enabled),
        operator_reuse_gate_pass=bool(operator_reuse_gate_pass),
        deterministic_output_gate_pass=bool(deterministic_output_gate_pass),
        measured_hot_speedup=hot_speedup,
        min_hot_speedup=speedup_gate,
        memory_growth_fraction=memory_growth,
        max_memory_growth_fraction=memory_gate,
        coarse_strategy=_normalized_token(coarse_strategy or "replicated_schur"),
        coarse_levels=coarse_level_count,
        max_coarse_rank=coarse_rank,
        operator_build_scope=(
            "once_per_child_process"
            if bool(operator_reuse_enabled)
            else "per_timed_solve"
        ),
        operator_action_scope="compiled_sharded_device_function",
        preconditioner_scope=f"local_{axis}_slab_apply",
        coarse_operator_scope="replicated_small_dense_operator",
        coarse_solve_scope="replicated_device_dense_solve",
        compiled_components=compiled_components,
        reused_components=reused_components,
        per_device_components=per_device_components,
        replicated_components=replicated_components,
        required_runtime_gates=required_runtime_gates,
        plan_valid=valid,
        promotion_ready=ready,
        failures=tuple(failures),
        promotion_blockers=tuple(dict.fromkeys(blockers)),
        notes=tuple(notes),
    )


# --- Parallel worker runtime orchestration ---




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


def transport_parallel_worker_env(
    parallel_workers: int,
    rewrite_xla_flags: Callable[[str, int | None, int | None], str] = rewrite_xla_flags,
):
    """Return the process-pool worker environment context for transport solves."""
    return _transport_parallel_worker_env(
        parallel_workers=int(parallel_workers),
        rewrite_xla_flags=rewrite_xla_flags,
    )


def transport_parallel_pool_executor_kwargs(
    *,
    parallel_workers: int,
    get_context: Callable[[str], object] = mp.get_context,
    emit: Callable[[int, str], None] | None = None,
) -> dict[str, object]:
    """Build ``ProcessPoolExecutor`` kwargs for transport worker pools."""
    return _transport_parallel_pool_executor_kwargs(
        parallel_workers=int(parallel_workers),
        get_context=get_context,
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
