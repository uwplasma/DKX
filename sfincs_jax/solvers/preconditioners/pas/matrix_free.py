"""Matrix-free RHSMode=1 PAS correction helpers.

The routines here are intentionally opt-in building blocks. They do not build
dense angular patch inverses and only accept a correction when a fresh matrix-
free residual check shows a finite improvement.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
import math

import jax.numpy as jnp

ArrayLike = object


@dataclass(frozen=True)
class Rhs1PasMatrixFreeConfig:
    """Acceptance controls for a bounded matrix-free PAS correction."""

    max_steps: int = 1
    omega: float = 1.0
    min_residual_reduction: float = 1.0e-3
    block_size: int | None = None
    max_update_norm_ratio: float | None = None
    min_update_norm_ratio: float | None = None
    max_candidate_elements: int | None = None
    max_candidate_bytes: int | None = None
    max_reduction_bytes: int | None = None
    stream_update_chunks: bool = False
    max_update_chunk_bytes: int | None = None

    def __post_init__(self) -> None:
        if int(self.max_steps) < 1:
            raise ValueError("max_steps must be >= 1")
        if not math.isfinite(float(self.omega)):
            raise ValueError("omega must be finite")
        if not math.isfinite(float(self.min_residual_reduction)):
            raise ValueError("min_residual_reduction must be finite")
        if float(self.min_residual_reduction) < 0.0:
            raise ValueError("min_residual_reduction must be >= 0")
        if self.block_size is not None and int(self.block_size) < 1:
            raise ValueError("block_size must be >= 1 when provided")
        if self.max_update_norm_ratio is not None:
            ratio = float(self.max_update_norm_ratio)
            if not math.isfinite(ratio) or ratio <= 0.0:
                raise ValueError("max_update_norm_ratio must be finite and > 0 when provided")
        if self.min_update_norm_ratio is not None:
            ratio = float(self.min_update_norm_ratio)
            if not math.isfinite(ratio) or ratio < 0.0:
                raise ValueError("min_update_norm_ratio must be finite and >= 0 when provided")
        if self.max_candidate_elements is not None and int(self.max_candidate_elements) < 1:
            raise ValueError("max_candidate_elements must be >= 1 when provided")
        if self.max_candidate_bytes is not None and int(self.max_candidate_bytes) < 1:
            raise ValueError("max_candidate_bytes must be >= 1 when provided")
        if self.max_reduction_bytes is not None and int(self.max_reduction_bytes) < 1:
            raise ValueError("max_reduction_bytes must be >= 1 when provided")
        if self.max_update_chunk_bytes is not None and int(self.max_update_chunk_bytes) < 1:
            raise ValueError("max_update_chunk_bytes must be >= 1 when provided")


@dataclass(frozen=True)
class PasRuntimeChunkPlan:
    """Bounded chunk plan for PAS vector reductions and live-array guards."""

    element_count: int
    itemsize: int
    array_bytes: int
    requested_block_size: int | None
    block_size: int | None
    live_array_count: int
    estimated_live_array_bytes: int
    max_live_bytes: int | None
    live_byte_margin: int | None
    reduction_work_arrays: int
    max_reduction_bytes: int | None
    estimated_reduction_bytes: int | None
    safe: bool
    reason: str

    def as_metadata(self) -> dict[str, object]:
        """Return a plain metadata mapping suitable for diagnostics."""

        return asdict(self)


@dataclass(frozen=True)
class Rhs1PasMatrixFreeResult:
    """Result from a guarded matrix-free PAS correction attempt."""

    x: jnp.ndarray
    residual_norm: float
    initial_residual_norm: float
    residual_history: tuple[float, ...]
    accepted_steps: int
    accepted: bool
    reason: str
    diagnostics: dict[str, object] = field(default_factory=dict)


def _finite_or_none(value: float) -> float | None:
    parsed = float(value)
    if not math.isfinite(parsed):
        return None
    return parsed


def _gate_diagnostics(
    *,
    reason: str,
    initial_residual_norm: float,
    candidate_residual_norm: float | None = None,
    min_residual_reduction: float,
    accepted_steps: int,
) -> dict[str, object]:
    initial = _finite_or_none(initial_residual_norm)
    candidate = None if candidate_residual_norm is None else _finite_or_none(candidate_residual_norm)
    diagnostics: dict[str, object] = {
        "reason": reason,
        "accepted_steps": int(accepted_steps),
        "initial_residual_norm": initial,
        "candidate_residual_norm": candidate,
        "min_residual_reduction": float(min_residual_reduction),
    }
    if initial is not None:
        diagnostics["required_residual_norm"] = initial * max(0.0, 1.0 - float(min_residual_reduction))
    if initial is not None and candidate is not None and initial > 0.0:
        diagnostics["residual_reduction"] = (initial - candidate) / initial
    diagnostics["initial_residual_finite"] = initial is not None
    diagnostics["candidate_residual_finite"] = candidate is not None if candidate_residual_norm is not None else None
    return diagnostics


def _array_metadata(value: jnp.ndarray, *, block_size: int | None) -> dict[str, object]:
    arr = jnp.asarray(value)
    element_count = int(arr.size)
    itemsize = int(getattr(arr.dtype, "itemsize", 0))
    return {
        "shape": tuple(int(dim) for dim in arr.shape),
        "dtype": str(arr.dtype),
        "element_count": element_count,
        "array_bytes": int(element_count * itemsize),
        "block_size": None if block_size is None else int(block_size),
    }


def _reduction_chunk_bytes_for_config(
    config: Rhs1PasMatrixFreeConfig,
    *,
    live_arrays: int,
) -> int | None:
    if config.max_reduction_bytes is not None:
        return int(config.max_reduction_bytes)
    if config.max_candidate_bytes is None:
        return None
    return max(1, int(config.max_candidate_bytes) // max(1, int(live_arrays)))


def _update_chunk_bytes_for_config(config: Rhs1PasMatrixFreeConfig) -> int | None:
    if config.max_update_chunk_bytes is not None:
        return int(config.max_update_chunk_bytes)
    return None


def plan_pas_runtime_chunks(
    value: ArrayLike,
    *,
    requested_block_size: int | None = None,
    max_live_bytes: int | None = None,
    live_arrays: int = 1,
    max_reduction_bytes: int | None = None,
    reduction_work_arrays: int = 2,
) -> PasRuntimeChunkPlan:
    """Plan bounded PAS vector chunks without changing numerical operations.

    ``max_live_bytes`` guards how many full vectors may be live at once.
    ``max_reduction_bytes`` bounds each streaming reduction chunk. If a
    non-empty vector cannot fit even one element of reduction work, the plan is
    unsafe so callers can fail closed before launching larger PAS work.
    """

    arr = jnp.asarray(value)
    element_count = int(arr.size)
    itemsize = max(1, int(getattr(arr.dtype, "itemsize", 0)))
    array_bytes = int(element_count * itemsize)
    live_array_count = max(1, int(live_arrays))
    estimated_live_bytes = int(array_bytes * live_array_count)
    max_live = None if max_live_bytes is None else int(max_live_bytes)
    live_margin = None if max_live is None else int(max_live - estimated_live_bytes)
    requested_block = None if requested_block_size is None else max(1, int(requested_block_size))
    work_arrays = max(1, int(reduction_work_arrays))
    max_reduction = None if max_reduction_bytes is None else int(max_reduction_bytes)

    if max_live is not None and estimated_live_bytes > max_live:
        return PasRuntimeChunkPlan(
            element_count=element_count,
            itemsize=itemsize,
            array_bytes=array_bytes,
            requested_block_size=requested_block,
            block_size=requested_block,
            live_array_count=live_array_count,
            estimated_live_array_bytes=estimated_live_bytes,
            max_live_bytes=max_live,
            live_byte_margin=live_margin,
            reduction_work_arrays=work_arrays,
            max_reduction_bytes=max_reduction,
            estimated_reduction_bytes=None,
            safe=False,
            reason="live-memory-limit-exceeded",
        )

    if element_count == 0:
        return PasRuntimeChunkPlan(
            element_count=0,
            itemsize=itemsize,
            array_bytes=0,
            requested_block_size=requested_block,
            block_size=0 if requested_block is not None or max_reduction is not None else None,
            live_array_count=live_array_count,
            estimated_live_array_bytes=0,
            max_live_bytes=max_live,
            live_byte_margin=live_margin,
            reduction_work_arrays=work_arrays,
            max_reduction_bytes=max_reduction,
            estimated_reduction_bytes=0,
            safe=True,
            reason="within-pas-runtime-memory-limit",
        )

    bytes_per_reduction_element = int(itemsize * work_arrays)
    if max_reduction is None:
        block_size = requested_block
        estimated_reduction = (
            int(min(element_count, block_size) * bytes_per_reduction_element)
            if block_size is not None
            else int(element_count * bytes_per_reduction_element)
        )
    else:
        if max_reduction < bytes_per_reduction_element:
            return PasRuntimeChunkPlan(
                element_count=element_count,
                itemsize=itemsize,
                array_bytes=array_bytes,
                requested_block_size=requested_block,
                block_size=0,
                live_array_count=live_array_count,
                estimated_live_array_bytes=estimated_live_bytes,
                max_live_bytes=max_live,
                live_byte_margin=live_margin,
                reduction_work_arrays=work_arrays,
                max_reduction_bytes=max_reduction,
                estimated_reduction_bytes=bytes_per_reduction_element,
                safe=False,
                reason="reduction-memory-limit-exceeded",
            )
        budget_block = max(1, int(max_reduction // bytes_per_reduction_element))
        block_size = min(element_count, budget_block if requested_block is None else min(requested_block, budget_block))
        estimated_reduction = int(block_size * bytes_per_reduction_element)

    return PasRuntimeChunkPlan(
        element_count=element_count,
        itemsize=itemsize,
        array_bytes=array_bytes,
        requested_block_size=requested_block,
        block_size=block_size,
        live_array_count=live_array_count,
        estimated_live_array_bytes=estimated_live_bytes,
        max_live_bytes=max_live,
        live_byte_margin=live_margin,
        reduction_work_arrays=work_arrays,
        max_reduction_bytes=max_reduction,
        estimated_reduction_bytes=estimated_reduction,
        safe=True,
        reason="within-pas-runtime-memory-limit",
    )


def rhs1_pas_matrixfree_preflight_gate(
    x_template: ArrayLike,
    *,
    config: Rhs1PasMatrixFreeConfig = Rhs1PasMatrixFreeConfig(),
    live_arrays: int = 5,
) -> tuple[bool, str, dict[str, object]]:
    """Return a fail-fast memory gate for a matrix-free PAS candidate step.

    The guarded PAS correction needs several vectors live at once around the
    candidate update and residual check. Rejecting at this preflight avoids
    calling expensive correction builders, materializing candidate updates, or
    running another matvec when the configured element/byte budget cannot hold.
    """

    reduction_budget = _reduction_chunk_bytes_for_config(config, live_arrays=live_arrays)
    chunk_plan = plan_pas_runtime_chunks(
        x_template,
        requested_block_size=config.block_size,
        max_live_bytes=config.max_candidate_bytes,
        live_arrays=live_arrays,
        max_reduction_bytes=reduction_budget,
    )
    metadata = _array_metadata(jnp.asarray(x_template), block_size=chunk_plan.block_size)
    live_array_count = max(1, int(live_arrays))
    estimated_live_bytes = int(metadata["array_bytes"]) * live_array_count
    enriched = dict(metadata)
    enriched["estimated_live_array_count"] = int(live_array_count)
    enriched["estimated_live_array_bytes"] = int(estimated_live_bytes)
    enriched["preflight_kind"] = "rhs1_pas_matrixfree_candidate"
    enriched["max_candidate_elements"] = (
        None if config.max_candidate_elements is None else int(config.max_candidate_elements)
    )
    enriched["max_candidate_bytes"] = (
        None if config.max_candidate_bytes is None else int(config.max_candidate_bytes)
    )
    enriched["candidate_element_budget_configured"] = config.max_candidate_elements is not None
    enriched["candidate_byte_budget_configured"] = config.max_candidate_bytes is not None
    enriched["candidate_byte_budget_margin"] = (
        None if config.max_candidate_bytes is None else int(config.max_candidate_bytes) - estimated_live_bytes
    )
    enriched["max_reduction_bytes"] = reduction_budget
    enriched["planned_norm_block_size"] = chunk_plan.block_size
    enriched["reduction_chunk_plan"] = chunk_plan.as_metadata()
    enriched["stream_update_chunks"] = bool(config.stream_update_chunks)
    enriched["max_update_chunk_bytes"] = _update_chunk_bytes_for_config(config)
    enriched["planned_update_block_size"] = None
    enriched["stream_update_chunk_plan"] = None
    enriched["full_update_materialized"] = not bool(config.stream_update_chunks)
    enriched["min_update_norm_ratio"] = (
        None if config.min_update_norm_ratio is None else float(config.min_update_norm_ratio)
    )
    if config.stream_update_chunks:
        update_chunk_plan = plan_pas_runtime_chunks(
            x_template,
            requested_block_size=config.block_size,
            max_reduction_bytes=_update_chunk_bytes_for_config(config),
            reduction_work_arrays=1,
        )
        enriched["planned_update_block_size"] = update_chunk_plan.block_size
        enriched["stream_update_chunk_plan"] = update_chunk_plan.as_metadata()
        if not update_chunk_plan.safe:
            enriched["safe"] = False
            enriched["reason"] = "update-chunk-memory-limit-exceeded"
            return False, "update-chunk-memory-limit-exceeded", enriched
    if config.max_candidate_elements is not None and int(metadata["element_count"]) > int(
        config.max_candidate_elements
    ):
        enriched["safe"] = False
        enriched["reason"] = "candidate-size-limit-exceeded"
        return False, "candidate-size-limit-exceeded", enriched
    if config.max_candidate_bytes is not None and estimated_live_bytes > int(config.max_candidate_bytes):
        enriched["safe"] = False
        enriched["reason"] = "candidate-memory-limit-exceeded"
        return False, "candidate-memory-limit-exceeded", enriched
    if not chunk_plan.safe:
        enriched["safe"] = False
        enriched["reason"] = str(chunk_plan.reason)
        return False, str(chunk_plan.reason), enriched
    enriched["safe"] = True
    enriched["reason"] = "within-candidate-memory-limit"
    return True, "within-candidate-memory-limit", enriched


def _with_matrixfree_metadata(
    diagnostics: dict[str, object],
    *,
    x_template: jnp.ndarray,
    config: Rhs1PasMatrixFreeConfig,
    live_arrays: int,
    candidate_matvecs: int,
) -> dict[str, object]:
    enriched = dict(diagnostics)
    _safe, _reason, metadata = rhs1_pas_matrixfree_preflight_gate(
        x_template,
        config=config,
        live_arrays=live_arrays,
    )
    metadata["candidate_matvecs"] = int(candidate_matvecs)
    enriched["matrix_free_metadata"] = metadata
    return enriched


def streaming_l2_norm(
    value: ArrayLike,
    *,
    block_size: int | None = None,
    max_chunk_bytes: int | None = None,
) -> float:
    """Return an L2 norm using optional flat chunks for bounded accumulation."""

    arr = jnp.ravel(jnp.asarray(value))
    if arr.size == 0:
        return 0.0
    plan = plan_pas_runtime_chunks(
        arr,
        requested_block_size=block_size,
        max_reduction_bytes=max_chunk_bytes,
    )
    if not plan.safe:
        raise MemoryError(f"PAS streaming norm rejected: {plan.reason}")
    planned_block_size = plan.block_size
    if planned_block_size is None:
        norm = jnp.linalg.norm(arr)
        return float(norm)
    block = max(1, int(planned_block_size))
    total = 0.0
    for start in range(0, int(arr.size), block):
        chunk = arr[start : start + block]
        chunk_total = float(jnp.real(jnp.vdot(chunk, chunk)))
        if not math.isfinite(chunk_total):
            return chunk_total
        total += chunk_total
        if not math.isfinite(total):
            return total
    return math.sqrt(max(0.0, total))


def _configured_streaming_l2_norm(
    value: ArrayLike,
    *,
    config: Rhs1PasMatrixFreeConfig,
    live_arrays: int,
) -> float:
    return streaming_l2_norm(
        value,
        block_size=config.block_size,
        max_chunk_bytes=_reduction_chunk_bytes_for_config(config, live_arrays=live_arrays),
    )


@dataclass(frozen=True)
class _StreamedUpdateResult:
    candidate: jnp.ndarray | None
    update_norm: float
    reason: str | None
    metadata: dict[str, object]


def _with_streamed_update_metadata(
    diagnostics: dict[str, object],
    streamed_update_metadata: dict[str, object] | None,
) -> dict[str, object]:
    if streamed_update_metadata is None:
        return diagnostics
    enriched = dict(diagnostics)
    enriched["streamed_update_metadata"] = streamed_update_metadata
    return enriched


def _streamed_update_candidate(
    *,
    x_best: jnp.ndarray,
    residual: jnp.ndarray,
    chunked_correction: Callable[[jnp.ndarray, int, int], jnp.ndarray] | None,
    omega: float,
    config: Rhs1PasMatrixFreeConfig,
) -> _StreamedUpdateResult:
    """Build a candidate from correction chunks without a dense update vector."""

    plan = plan_pas_runtime_chunks(
        x_best,
        requested_block_size=config.block_size,
        max_reduction_bytes=_update_chunk_bytes_for_config(config),
        reduction_work_arrays=1,
    )
    metadata: dict[str, object] = {
        "stream_update_chunks": True,
        "full_update_materialized": False,
        "update_chunk_plan": plan.as_metadata(),
        "planned_update_block_size": plan.block_size,
        "max_update_chunk_bytes": _update_chunk_bytes_for_config(config),
        "update_chunk_count": 0,
        "max_update_chunk_elements": 0,
    }
    if not plan.safe:
        metadata["reason"] = "update-chunk-memory-limit-exceeded"
        return _StreamedUpdateResult(
            candidate=None,
            update_norm=float("nan"),
            reason="update-chunk-memory-limit-exceeded",
            metadata=metadata,
        )
    if chunked_correction is None:
        metadata["reason"] = "stream-update-correction-missing"
        return _StreamedUpdateResult(
            candidate=None,
            update_norm=float("nan"),
            reason="stream-update-correction-missing",
            metadata=metadata,
        )

    residual_flat = jnp.ravel(jnp.asarray(residual))
    candidate_flat = jnp.ravel(jnp.asarray(x_best))
    element_count = int(candidate_flat.size)
    block = int(plan.block_size) if plan.block_size is not None else element_count
    block = max(1, block)
    total = 0.0
    chunk_count = 0
    max_chunk_elements = 0
    for start in range(0, element_count, block):
        stop = min(element_count, start + block)
        residual_chunk = residual_flat[start:stop]
        update_chunk = jnp.ravel(jnp.asarray(chunked_correction(residual_chunk, start, stop)))
        expected_shape = tuple(int(dim) for dim in residual_chunk.shape)
        observed_shape = tuple(int(dim) for dim in update_chunk.shape)
        if observed_shape != expected_shape:
            metadata.update(
                {
                    "reason": "update-chunk-shape-mismatch",
                    "expected_chunk_shape": expected_shape,
                    "observed_chunk_shape": observed_shape,
                    "update_chunk_count": int(chunk_count),
                    "max_update_chunk_elements": int(max_chunk_elements),
                }
            )
            return _StreamedUpdateResult(
                candidate=None,
                update_norm=float("nan"),
                reason="update-chunk-shape-mismatch",
                metadata=metadata,
            )
        update_chunk = jnp.asarray(update_chunk, dtype=x_best.dtype)
        chunk_total = float(jnp.real(jnp.vdot(update_chunk, update_chunk)))
        chunk_count += 1
        max_chunk_elements = max(max_chunk_elements, int(update_chunk.size))
        if not math.isfinite(chunk_total):
            metadata.update(
                {
                    "reason": "nonfinite-update",
                    "update_chunk_count": int(chunk_count),
                    "max_update_chunk_elements": int(max_chunk_elements),
                    "nonfinite_chunk_start": int(start),
                    "nonfinite_chunk_stop": int(stop),
                }
            )
            return _StreamedUpdateResult(
                candidate=None,
                update_norm=chunk_total,
                reason="nonfinite-update",
                metadata=metadata,
            )
        total += chunk_total
        if not math.isfinite(total):
            metadata.update(
                {
                    "reason": "nonfinite-update",
                    "update_chunk_count": int(chunk_count),
                    "max_update_chunk_elements": int(max_chunk_elements),
                }
            )
            return _StreamedUpdateResult(
                candidate=None,
                update_norm=total,
                reason="nonfinite-update",
                metadata=metadata,
            )
        if float(omega) != 0.0:
            scaled_update = jnp.asarray(float(omega) * update_chunk, dtype=x_best.dtype)
            candidate_flat = candidate_flat.at[start:stop].add(scaled_update)

    update_norm = math.sqrt(max(0.0, total))
    metadata.update(
        {
            "reason": "streamed-update-built",
            "update_chunk_count": int(chunk_count),
            "max_update_chunk_elements": int(max_chunk_elements),
            "update_norm": float(update_norm),
        }
    )
    return _StreamedUpdateResult(
        candidate=jnp.asarray(candidate_flat.reshape(x_best.shape), dtype=x_best.dtype),
        update_norm=float(update_norm),
        reason=None,
        metadata=metadata,
    )


def rhs1_pas_matrixfree_acceptance_gate(
    *,
    initial_residual_norm: float,
    candidate_residual_norm: float,
    min_residual_reduction: float,
) -> tuple[bool, str]:
    """Return whether a candidate residual is finite and sufficiently improved."""

    initial = float(initial_residual_norm)
    candidate = float(candidate_residual_norm)
    if not math.isfinite(initial):
        return False, "nonfinite-initial-residual"
    if not math.isfinite(candidate):
        return False, "nonfinite-candidate-residual"
    if initial < 0.0 or candidate < 0.0:
        return False, "negative-residual"
    required = initial * max(0.0, 1.0 - float(min_residual_reduction))
    if candidate < required:
        return True, "accepted"
    return False, "insufficient-residual-improvement"


def rhs1_pas_matrixfree_correction(
    *,
    matvec: Callable[[jnp.ndarray], jnp.ndarray],
    rhs: ArrayLike,
    x0: ArrayLike,
    correction: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    chunked_correction: Callable[[jnp.ndarray, int, int], jnp.ndarray] | None = None,
    config: Rhs1PasMatrixFreeConfig = Rhs1PasMatrixFreeConfig(),
) -> Rhs1PasMatrixFreeResult:
    """Apply a bounded matrix-free PAS correction only when residual gates pass.

    ``correction`` receives the current residual and returns an update
    direction. With ``config.stream_update_chunks=True``, ``chunked_correction``
    instead receives flat residual chunks plus ``start``/``stop`` offsets, so
    this helper can build the candidate without materializing a full dense
    update vector. Every proposed step is checked with ``matvec`` and rejected
    on non-finite values, shape changes, or failure to improve the residual by
    ``min_residual_reduction``.
    """

    x_initial = jnp.asarray(x0)
    rhs_arr = jnp.asarray(rhs)
    if rhs_arr.shape != x_initial.shape:
        raise ValueError("rhs and x0 must have the same shape")

    preflight_safe, preflight_reason, preflight_metadata = rhs1_pas_matrixfree_preflight_gate(
        x_initial,
        config=config,
        live_arrays=5,
    )
    if not preflight_safe:
        metadata = dict(preflight_metadata)
        metadata["candidate_matvecs"] = 0
        initial_norm = float("nan")
        return Rhs1PasMatrixFreeResult(
            x=x_initial,
            residual_norm=initial_norm,
            initial_residual_norm=initial_norm,
            residual_history=(),
            accepted_steps=0,
            accepted=False,
            reason=preflight_reason,
            diagnostics={
                "reason": preflight_reason,
                "accepted_steps": 0,
                "initial_residual_norm": None,
                "candidate_residual_norm": None,
                "min_residual_reduction": float(config.min_residual_reduction),
                "initial_residual_finite": None,
                "candidate_residual_finite": None,
                "matrix_free_metadata": metadata,
            },
        )

    x_best = x_initial
    residual = rhs_arr - matvec(x_best)
    if jnp.asarray(residual).shape != x_initial.shape:
        raise ValueError("matvec(x0) must have the same shape as x0")
    initial_norm = _configured_streaming_l2_norm(residual, config=config, live_arrays=2)
    history: list[float] = [initial_norm]
    if not math.isfinite(initial_norm):
        return Rhs1PasMatrixFreeResult(
            x=x_initial,
            residual_norm=initial_norm,
            initial_residual_norm=initial_norm,
            residual_history=tuple(history),
            accepted_steps=0,
            accepted=False,
            reason="nonfinite-initial-residual",
            diagnostics=_with_matrixfree_metadata(
                _gate_diagnostics(
                    reason="nonfinite-initial-residual",
                    initial_residual_norm=initial_norm,
                    min_residual_reduction=float(config.min_residual_reduction),
                    accepted_steps=0,
                ),
                x_template=x_initial,
                config=config,
                live_arrays=2,
                candidate_matvecs=0,
            ),
        )

    accepted_steps = 0
    best_norm = initial_norm
    current_residual = jnp.asarray(residual)
    omega_use = float(config.omega)
    max_steps = max(1, int(config.max_steps))
    candidate_matvecs = 0
    last_streamed_update_metadata: dict[str, object] | None = None
    for step_index in range(max_steps):
        streamed_update_metadata: dict[str, object] | None = None
        if config.stream_update_chunks:
            streamed = _streamed_update_candidate(
                x_best=x_best,
                residual=current_residual,
                chunked_correction=chunked_correction,
                omega=omega_use,
                config=config,
            )
            streamed_update_metadata = streamed.metadata
            last_streamed_update_metadata = streamed_update_metadata
            update_norm = streamed.update_norm
            candidate = streamed.candidate
            if streamed.reason is not None:
                return Rhs1PasMatrixFreeResult(
                    x=x_best,
                    residual_norm=best_norm,
                    initial_residual_norm=initial_norm,
                    residual_history=tuple(history),
                    accepted_steps=accepted_steps,
                    accepted=accepted_steps > 0,
                    reason=streamed.reason,
                    diagnostics=_with_matrixfree_metadata(
                        _with_streamed_update_metadata(
                            {
                                "reason": streamed.reason,
                                "accepted_steps": int(accepted_steps),
                                "update_norm": None,
                                "update_norm_finite": False,
                            },
                            streamed_update_metadata,
                        ),
                        x_template=x_initial,
                        config=config,
                        live_arrays=3,
                        candidate_matvecs=candidate_matvecs,
                    ),
                )
            if candidate is None:
                raise RuntimeError("streamed update reported success without a candidate")
        else:
            if correction is None:
                raise ValueError("correction must be provided unless stream_update_chunks is enabled")
            update = jnp.asarray(correction(current_residual))
            if update.shape != x_initial.shape:
                return Rhs1PasMatrixFreeResult(
                    x=x_best,
                    residual_norm=best_norm,
                    initial_residual_norm=initial_norm,
                    residual_history=tuple(history),
                    accepted_steps=accepted_steps,
                    accepted=accepted_steps > 0,
                    reason="update-shape-mismatch",
                    diagnostics=_with_matrixfree_metadata(
                        {
                            "reason": "update-shape-mismatch",
                            "accepted_steps": int(accepted_steps),
                            "expected_shape": tuple(int(dim) for dim in x_initial.shape),
                            "observed_shape": tuple(int(dim) for dim in update.shape),
                        },
                        x_template=x_initial,
                        config=config,
                        live_arrays=3,
                        candidate_matvecs=candidate_matvecs,
                    ),
                )
            update_norm = _configured_streaming_l2_norm(update, config=config, live_arrays=3)
            candidate = x_best + jnp.asarray(omega_use * update, dtype=x_initial.dtype)
            candidate = jnp.asarray(candidate, dtype=x_initial.dtype)
            del update
        if not math.isfinite(update_norm):
            return Rhs1PasMatrixFreeResult(
                x=x_best,
                residual_norm=best_norm,
                initial_residual_norm=initial_norm,
                residual_history=tuple(history),
                accepted_steps=accepted_steps,
                accepted=accepted_steps > 0,
                reason="nonfinite-update",
                diagnostics=_with_matrixfree_metadata(
                    _with_streamed_update_metadata(
                        {
                            "reason": "nonfinite-update",
                            "accepted_steps": int(accepted_steps),
                            "update_norm": None,
                            "update_norm_finite": False,
                        },
                        streamed_update_metadata,
                    ),
                    x_template=x_initial,
                    config=config,
                    live_arrays=3,
                    candidate_matvecs=candidate_matvecs,
                ),
            )
        if update_norm == 0.0 or omega_use == 0.0:
            history.append(best_norm)
            return Rhs1PasMatrixFreeResult(
                x=x_best,
                residual_norm=best_norm,
                initial_residual_norm=initial_norm,
                residual_history=tuple(history),
                accepted_steps=accepted_steps,
                accepted=accepted_steps > 0,
                reason="insufficient-residual-improvement",
                diagnostics=_with_matrixfree_metadata(
                    _with_streamed_update_metadata(
                        _gate_diagnostics(
                            reason="insufficient-residual-improvement",
                            initial_residual_norm=best_norm,
                            candidate_residual_norm=best_norm,
                            min_residual_reduction=float(config.min_residual_reduction),
                            accepted_steps=accepted_steps,
                        ),
                        streamed_update_metadata,
                    ),
                    x_template=x_initial,
                    config=config,
                    live_arrays=3,
                    candidate_matvecs=candidate_matvecs,
                ),
            )
        if config.min_update_norm_ratio is not None:
            min_update_ratio = float(config.min_update_norm_ratio)
            min_update_norm = max(float(best_norm), 1.0) * min_update_ratio
            damped_update_norm = abs(omega_use) * float(update_norm)
            if damped_update_norm <= min_update_norm:
                history.append(best_norm)
                return Rhs1PasMatrixFreeResult(
                    x=x_best,
                    residual_norm=best_norm,
                    initial_residual_norm=initial_norm,
                    residual_history=tuple(history),
                    accepted_steps=accepted_steps,
                    accepted=accepted_steps > 0,
                    reason="update-norm-too-small",
                    diagnostics=_with_matrixfree_metadata(
                        _with_streamed_update_metadata(
                            {
                                "reason": "update-norm-too-small",
                                "accepted_steps": int(accepted_steps),
                                "update_norm": float(update_norm),
                                "damped_update_norm": float(damped_update_norm),
                                "min_update_norm": float(min_update_norm),
                                "min_update_norm_ratio": float(min_update_ratio),
                            },
                            streamed_update_metadata,
                        ),
                        x_template=x_initial,
                        config=config,
                        live_arrays=3,
                        candidate_matvecs=candidate_matvecs,
                    ),
                )
        if config.max_update_norm_ratio is not None:
            max_update_ratio = float(config.max_update_norm_ratio)
            if update_norm > max_update_ratio:
                x_scale = max(_configured_streaming_l2_norm(x_best, config=config, live_arrays=3), 1.0)
            else:
                x_scale = 1.0
            if update_norm > x_scale * max_update_ratio:
                update_limit = x_scale * max_update_ratio
                return Rhs1PasMatrixFreeResult(
                    x=x_best,
                    residual_norm=best_norm,
                    initial_residual_norm=initial_norm,
                    residual_history=tuple(history),
                    accepted_steps=accepted_steps,
                    accepted=accepted_steps > 0,
                    reason="update-norm-too-large",
                    diagnostics=_with_matrixfree_metadata(
                        _with_streamed_update_metadata(
                            {
                                "reason": "update-norm-too-large",
                                "accepted_steps": int(accepted_steps),
                                "update_norm": float(update_norm),
                                "update_norm_limit": float(update_limit),
                                "x_scale": float(x_scale),
                                "max_update_norm_ratio": float(max_update_ratio),
                            },
                            streamed_update_metadata,
                        ),
                        x_template=x_initial,
                        config=config,
                        live_arrays=3,
                        candidate_matvecs=candidate_matvecs,
                    ),
                )
        candidate_residual = rhs_arr - matvec(candidate)
        candidate_matvecs += 1
        if jnp.asarray(candidate_residual).shape != x_initial.shape:
            return Rhs1PasMatrixFreeResult(
                x=x_best,
                residual_norm=best_norm,
                initial_residual_norm=initial_norm,
                residual_history=tuple(history),
                accepted_steps=accepted_steps,
                accepted=accepted_steps > 0,
                reason="candidate-residual-shape-mismatch",
                diagnostics=_with_matrixfree_metadata(
                    _with_streamed_update_metadata(
                        {
                            "reason": "candidate-residual-shape-mismatch",
                            "accepted_steps": int(accepted_steps),
                            "expected_shape": tuple(int(dim) for dim in x_initial.shape),
                            "observed_shape": tuple(int(dim) for dim in jnp.asarray(candidate_residual).shape),
                        },
                        streamed_update_metadata,
                    ),
                    x_template=x_initial,
                    config=config,
                    live_arrays=5,
                    candidate_matvecs=candidate_matvecs,
                ),
            )
        candidate_norm = _configured_streaming_l2_norm(candidate_residual, config=config, live_arrays=5)
        history.append(candidate_norm)
        accepted, reason = rhs1_pas_matrixfree_acceptance_gate(
            initial_residual_norm=best_norm,
            candidate_residual_norm=candidate_norm,
            min_residual_reduction=float(config.min_residual_reduction),
        )
        if not accepted:
            return Rhs1PasMatrixFreeResult(
                x=x_best,
                residual_norm=best_norm,
                initial_residual_norm=initial_norm,
                residual_history=tuple(history),
                accepted_steps=accepted_steps,
                accepted=accepted_steps > 0,
                reason=reason,
                diagnostics=_with_matrixfree_metadata(
                    _with_streamed_update_metadata(
                        _gate_diagnostics(
                            reason=reason,
                            initial_residual_norm=best_norm,
                            candidate_residual_norm=candidate_norm,
                            min_residual_reduction=float(config.min_residual_reduction),
                            accepted_steps=accepted_steps,
                        ),
                        streamed_update_metadata,
                    ),
                    x_template=x_initial,
                    config=config,
                    live_arrays=5,
                    candidate_matvecs=candidate_matvecs,
                ),
            )
        x_best = candidate
        best_norm = candidate_norm
        accepted_steps += 1
        if step_index + 1 < max_steps:
            current_residual = jnp.asarray(candidate_residual)
        else:
            del candidate_residual

    return Rhs1PasMatrixFreeResult(
        x=x_best,
        residual_norm=best_norm,
        initial_residual_norm=initial_norm,
        residual_history=tuple(history),
        accepted_steps=accepted_steps,
        accepted=accepted_steps > 0,
        reason="accepted",
        diagnostics=_with_matrixfree_metadata(
            _with_streamed_update_metadata(
                _gate_diagnostics(
                    reason="accepted",
                    initial_residual_norm=initial_norm,
                    candidate_residual_norm=best_norm,
                    min_residual_reduction=float(config.min_residual_reduction),
                    accepted_steps=accepted_steps,
                ),
                last_streamed_update_metadata,
            ),
            x_template=x_initial,
            config=config,
            live_arrays=4 if max_steps == 1 else 5,
            candidate_matvecs=candidate_matvecs,
        ),
    )


__all__ = [
    "Rhs1PasMatrixFreeConfig",
    "Rhs1PasMatrixFreeResult",
    "PasRuntimeChunkPlan",
    "plan_pas_runtime_chunks",
    "rhs1_pas_matrixfree_acceptance_gate",
    "rhs1_pas_matrixfree_correction",
    "rhs1_pas_matrixfree_preflight_gate",
    "streaming_l2_norm",
]
