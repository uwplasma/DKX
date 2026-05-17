"""Device-resident local smoothers for RHSMode=1 QI preconditioners.

The true device-QI lane needs a replacement for host x-block LU/ILU inside
hard-seed probes.  This module provides a bounded first building block: a
CSR-backed damped Jacobi/stationary smoother whose ``apply`` path is pure JAX
and reuses the assembled device operator.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .rhs1_device_operator import DeviceCSR

ArrayLike = Any


@dataclass(frozen=True)
class RHS1QIDeviceJacobiSmootherMetadata:
    """Diagnostics for a device CSR-backed QI local smoother."""

    shape: tuple[int, int]
    nnz: int
    damping: float
    sweeps: int
    step_policy: str
    max_step_damping: float
    min_step_denominator: float
    diagonal_floor: float
    valid_diagonal_count: int
    missing_diagonal_count: int
    duplicate_diagonal_count: int
    small_diagonal_count: int
    nonfinite_diagonal_count: int
    diagonal_min_abs: float
    diagonal_max_abs: float
    device_resident: bool
    source: str
    reason: str

    def to_dict(self) -> dict[str, object]:
        """Return JSON-friendly smoother diagnostics."""

        return {
            "shape": tuple(int(v) for v in self.shape),
            "nnz": int(self.nnz),
            "damping": float(self.damping),
            "sweeps": int(self.sweeps),
            "step_policy": self.step_policy,
            "max_step_damping": float(self.max_step_damping),
            "min_step_denominator": float(self.min_step_denominator),
            "diagonal_floor": float(self.diagonal_floor),
            "valid_diagonal_count": int(self.valid_diagonal_count),
            "missing_diagonal_count": int(self.missing_diagonal_count),
            "duplicate_diagonal_count": int(self.duplicate_diagonal_count),
            "small_diagonal_count": int(self.small_diagonal_count),
            "nonfinite_diagonal_count": int(self.nonfinite_diagonal_count),
            "diagonal_min_abs": float(self.diagonal_min_abs),
            "diagonal_max_abs": float(self.diagonal_max_abs),
            "device_resident": bool(self.device_resident),
            "source": self.source,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RHS1QIDeviceJacobiSmoother:
    """Pure-JAX damped Jacobi/stationary local smoother."""

    operator: DeviceCSR
    inverse_diagonal: ArrayLike
    diagonal: ArrayLike
    metadata: RHS1QIDeviceJacobiSmootherMetadata

    def apply(self, residual: ArrayLike) -> ArrayLike:
        """Apply bounded Jacobi sweeps to ``A correction = residual``.

        ``stationary`` uses the fixed damping requested at construction.
        ``residual_minimizing`` chooses a nonnegative per-sweep scalar that
        minimizes ``||r - alpha A D^{-1} r||`` and clips it to the configured
        maximum, making the local seed a measured residual-reducing candidate.
        """

        residual_vec = jnp.asarray(residual, dtype=self.operator.data.dtype).reshape((-1,))
        if int(residual_vec.shape[0]) != int(self.operator.shape[0]):
            raise ValueError(
                f"residual length {residual_vec.shape[0]} does not match "
                f"operator rows {self.operator.shape[0]}"
            )

        correction = jnp.zeros_like(residual_vec)
        remaining = residual_vec
        fixed_damping = jnp.asarray(float(self.metadata.damping), dtype=residual_vec.dtype)
        inverse_diagonal = jnp.asarray(self.inverse_diagonal, dtype=residual_vec.dtype)
        for _ in range(max(1, int(self.metadata.sweeps))):
            raw_step = inverse_diagonal * remaining
            if self.metadata.step_policy == "residual_minimizing":
                raw_action = self.operator.matvec(raw_step)
                step_scale = _residual_minimizing_step_scale(
                    raw_action,
                    remaining,
                    max_step_damping=float(self.metadata.max_step_damping),
                    min_step_denominator=float(self.metadata.min_step_denominator),
                )
                step = step_scale * raw_step
                action = step_scale * raw_action
            elif self.metadata.step_policy == "stationary":
                step = fixed_damping * raw_step
                action = self.operator.matvec(step)
            else:
                raise ValueError(f"unsupported device smoother step policy {self.metadata.step_policy!r}")
            correction = correction + step
            remaining = remaining - action
        return correction

    def as_preconditioner(self) -> Callable[[ArrayLike], ArrayLike]:
        """Return a callable suitable for QI local-smoother hooks."""

        return self.apply


@dataclass(frozen=True)
class RHS1QIDeviceSmootherProbe:
    """True-residual acceptance result for a standalone device smoother seed."""

    accepted: bool
    reason: str
    residual_before_norm: float
    residual_after_norm: float
    improvement_ratio: float | None
    metadata: RHS1QIDeviceJacobiSmootherMetadata

    def to_dict(self) -> dict[str, object]:
        """Return JSON-friendly probe diagnostics."""

        return {
            "accepted": bool(self.accepted),
            "reason": self.reason,
            "residual_before_norm": float(self.residual_before_norm),
            "residual_after_norm": float(self.residual_after_norm),
            "improvement_ratio": None if self.improvement_ratio is None else float(self.improvement_ratio),
            "metadata": self.metadata.to_dict(),
        }


def _normalize_step_policy(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "fixed": "stationary",
        "jacobi": "stationary",
        "stationary": "stationary",
        "damped_jacobi": "stationary",
        "minres": "residual_minimizing",
        "minimum_residual": "residual_minimizing",
        "residual_minimizing": "residual_minimizing",
        "residual_reducing": "residual_minimizing",
    }
    if normalized not in aliases:
        raise ValueError("step_policy must be 'stationary' or 'residual_minimizing'")
    return aliases[normalized]


def _residual_minimizing_step_scale(
    action: ArrayLike,
    residual: ArrayLike,
    *,
    max_step_damping: float,
    min_step_denominator: float,
) -> ArrayLike:
    action_vec = jnp.asarray(action).reshape((-1,))
    residual_vec = jnp.asarray(residual, dtype=action_vec.dtype).reshape((-1,))
    numerator = jnp.real(jnp.vdot(action_vec, residual_vec))
    denominator = jnp.real(jnp.vdot(action_vec, action_vec))
    floor = jnp.asarray(max(0.0, float(min_step_denominator)), dtype=denominator.dtype)
    denominator_valid = denominator > floor
    safe_denominator = jnp.where(denominator_valid, denominator, 1.0)
    raw_scale = numerator / safe_denominator
    finite_positive = (
        denominator_valid
        & jnp.isfinite(raw_scale)
        & jnp.isfinite(numerator)
        & jnp.isfinite(denominator)
        & (raw_scale > 0.0)
    )
    safe_scale = jnp.where(finite_positive, raw_scale, 0.0)
    cap = jnp.asarray(max(0.0, float(max_step_damping)), dtype=safe_scale.dtype)
    return jnp.minimum(safe_scale, cap)


def extract_device_csr_diagonal(device_operator: DeviceCSR) -> tuple[ArrayLike, ArrayLike]:
    """Return ``(diagonal, hit_count)`` from a square device CSR operator."""

    if int(device_operator.shape[0]) != int(device_operator.shape[1]):
        raise ValueError("device Jacobi smoother requires a square operator")

    n_rows = int(device_operator.shape[0])
    if int(device_operator.nnz) == 0:
        return (
            jnp.zeros((n_rows,), dtype=device_operator.data.dtype),
            jnp.zeros((n_rows,), dtype=jnp.int32),
        )

    row_lengths = jnp.diff(device_operator.indptr)
    rows = jnp.repeat(
        jnp.arange(n_rows, dtype=device_operator.indices.dtype),
        row_lengths,
        total_repeat_length=int(device_operator.nnz),
    )
    diagonal_mask = device_operator.indices == rows
    diagonal_values = jnp.where(diagonal_mask, device_operator.data, 0)
    diagonal = jnp.zeros((n_rows,), dtype=device_operator.data.dtype)
    diagonal = diagonal.at[rows].add(diagonal_values)
    hit_count = jnp.zeros((n_rows,), dtype=jnp.int32)
    hit_count = hit_count.at[rows].add(diagonal_mask.astype(jnp.int32))
    return diagonal, hit_count


def _diagonal_metadata(
    *,
    device_operator: DeviceCSR,
    diagonal: ArrayLike,
    hit_count: ArrayLike,
    damping: float,
    sweeps: int,
    step_policy: str,
    max_step_damping: float,
    min_step_denominator: float,
    diagonal_floor: float,
    require_all_diagonal: bool,
) -> RHS1QIDeviceJacobiSmootherMetadata:
    diagonal_np = np.asarray(jax.device_get(diagonal))
    hit_count_np = np.asarray(jax.device_get(hit_count), dtype=np.int64)
    diagonal_abs = np.abs(diagonal_np)
    finite = np.isfinite(diagonal_np)
    present = hit_count_np > 0
    small = present & finite & (diagonal_abs <= float(diagonal_floor))
    valid = present & finite & (diagonal_abs > float(diagonal_floor))

    missing_count = int(np.sum(~present))
    duplicate_count = int(np.sum(hit_count_np > 1))
    small_count = int(np.sum(small))
    nonfinite_count = int(np.sum(present & ~finite))
    valid_count = int(np.sum(valid))
    invalid_count = missing_count + small_count + nonfinite_count
    if bool(require_all_diagonal) and invalid_count > 0:
        raise ValueError(
            "device Jacobi smoother invalid diagonal: "
            f"missing={missing_count} small={small_count} nonfinite={nonfinite_count}"
        )

    finite_present_abs = diagonal_abs[present & finite]
    diagonal_min_abs = float(np.min(finite_present_abs)) if finite_present_abs.size else 0.0
    diagonal_max_abs = float(np.max(finite_present_abs)) if finite_present_abs.size else 0.0
    if valid_count == int(device_operator.shape[0]):
        reason = "built"
    elif valid_count > 0:
        reason = "partial_diagonal"
    else:
        reason = "empty_or_invalid_diagonal"

    return RHS1QIDeviceJacobiSmootherMetadata(
        shape=tuple(int(v) for v in device_operator.shape),
        nnz=int(device_operator.nnz),
        damping=float(damping),
        sweeps=int(sweeps),
        step_policy=str(step_policy),
        max_step_damping=float(max_step_damping),
        min_step_denominator=float(min_step_denominator),
        diagonal_floor=float(diagonal_floor),
        valid_diagonal_count=valid_count,
        missing_diagonal_count=missing_count,
        duplicate_diagonal_count=duplicate_count,
        small_diagonal_count=small_count,
        nonfinite_diagonal_count=nonfinite_count,
        diagonal_min_abs=diagonal_min_abs,
        diagonal_max_abs=diagonal_max_abs,
        device_resident=True,
        source=str(device_operator.metadata.source),
        reason=reason,
    )


def build_rhs1_qi_device_jacobi_smoother(
    device_operator: DeviceCSR,
    *,
    damping: float = 0.7,
    sweeps: int = 1,
    step_policy: str = "stationary",
    max_step_damping: float | None = None,
    min_step_denominator: float = 1.0e-30,
    diagonal_floor: float = 1.0e-14,
    require_all_diagonal: bool = True,
) -> RHS1QIDeviceJacobiSmoother:
    """Build a fail-closed device CSR-backed Jacobi smoother.

    By default every row must have a finite diagonal entry above
    ``diagonal_floor``.  Set ``require_all_diagonal=False`` only for diagnostic
    probes that intentionally want a partial smoother with zero correction on
    invalid rows.
    """

    damping_use = float(damping)
    if not np.isfinite(damping_use) or damping_use <= 0.0:
        raise ValueError("damping must be finite and positive")
    step_policy_use = _normalize_step_policy(step_policy)
    max_step_damping_use = (
        damping_use if max_step_damping is None else float(max_step_damping)
    )
    if not np.isfinite(max_step_damping_use) or max_step_damping_use <= 0.0:
        raise ValueError("max_step_damping must be finite and positive")
    min_step_denominator_use = max(0.0, float(min_step_denominator))
    if not np.isfinite(min_step_denominator_use):
        raise ValueError("min_step_denominator must be finite")
    sweeps_use = max(1, int(sweeps))
    floor_use = max(0.0, float(diagonal_floor))
    diagonal, hit_count = extract_device_csr_diagonal(device_operator)
    metadata = _diagonal_metadata(
        device_operator=device_operator,
        diagonal=diagonal,
        hit_count=hit_count,
        damping=damping_use,
        sweeps=sweeps_use,
        step_policy=step_policy_use,
        max_step_damping=max_step_damping_use,
        min_step_denominator=min_step_denominator_use,
        diagonal_floor=floor_use,
        require_all_diagonal=bool(require_all_diagonal),
    )
    valid = (hit_count > 0) & jnp.isfinite(diagonal) & (jnp.abs(diagonal) > floor_use)
    safe_diagonal = jnp.where(valid, diagonal, 1.0)
    inverse_diagonal = jnp.where(valid, 1.0 / safe_diagonal, 0.0)
    return RHS1QIDeviceJacobiSmoother(
        operator=device_operator,
        inverse_diagonal=inverse_diagonal,
        diagonal=diagonal,
        metadata=metadata,
    )


def probe_rhs1_qi_device_smoother_correction(
    *,
    rhs: ArrayLike,
    x0: ArrayLike,
    smoother: RHS1QIDeviceJacobiSmoother,
    operator: Callable[[ArrayLike], ArrayLike] | None = None,
    min_relative_improvement: float = 0.0,
    acceptance_atol: float = 0.0,
) -> tuple[ArrayLike, RHS1QIDeviceSmootherProbe]:
    """Apply the device smoother as a seed and fail closed without improvement."""

    matvec = smoother.operator.matvec if operator is None else operator
    rhs_vec = jnp.asarray(rhs, dtype=smoother.operator.data.dtype).reshape((-1,))
    x_initial = jnp.asarray(x0, dtype=rhs_vec.dtype).reshape((-1,))
    if rhs_vec.shape != x_initial.shape:
        raise ValueError("rhs and x0 must have the same shape")

    residual_before = rhs_vec - jnp.asarray(matvec(x_initial), dtype=rhs_vec.dtype).reshape((-1,))
    residual_before_norm = float(jnp.linalg.norm(residual_before))
    if residual_before_norm == 0.0:
        probe = RHS1QIDeviceSmootherProbe(
            accepted=False,
            reason="zero_residual",
            residual_before_norm=0.0,
            residual_after_norm=0.0,
            improvement_ratio=None,
            metadata=smoother.metadata,
        )
        return x_initial, probe

    dx = jnp.asarray(smoother.apply(residual_before), dtype=rhs_vec.dtype).reshape((-1,))
    x_candidate = x_initial + dx
    residual_after = rhs_vec - jnp.asarray(matvec(x_candidate), dtype=rhs_vec.dtype).reshape((-1,))
    residual_after_norm_measured = float(jnp.linalg.norm(residual_after))
    finite = bool(np.isfinite(residual_after_norm_measured))
    improvement_ratio = residual_after_norm_measured / residual_before_norm if finite else None
    required_drop = max(
        float(acceptance_atol),
        residual_before_norm * max(0.0, float(min_relative_improvement)),
    )
    accepted = finite and residual_after_norm_measured < residual_before_norm - required_drop
    if accepted:
        reason = "residual_reduced"
        residual_after_norm = residual_after_norm_measured
    elif not finite:
        reason = "nonfinite_candidate"
        residual_after_norm = residual_before_norm
    else:
        reason = "residual_not_reduced"
        residual_after_norm = residual_after_norm_measured
    probe = RHS1QIDeviceSmootherProbe(
        accepted=bool(accepted),
        reason=reason,
        residual_before_norm=residual_before_norm,
        residual_after_norm=residual_after_norm,
        improvement_ratio=improvement_ratio,
        metadata=smoother.metadata,
    )
    return x_candidate if accepted else x_initial, probe


__all__ = [
    "RHS1QIDeviceJacobiSmoother",
    "RHS1QIDeviceJacobiSmootherMetadata",
    "RHS1QIDeviceSmootherProbe",
    "build_rhs1_qi_device_jacobi_smoother",
    "extract_device_csr_diagonal",
    "probe_rhs1_qi_device_smoother_correction",
]
