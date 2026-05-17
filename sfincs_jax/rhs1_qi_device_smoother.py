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
        """Apply bounded Jacobi sweeps to ``A correction = residual``."""

        residual_vec = jnp.asarray(residual, dtype=self.operator.data.dtype).reshape((-1,))
        if int(residual_vec.shape[0]) != int(self.operator.shape[0]):
            raise ValueError(
                f"residual length {residual_vec.shape[0]} does not match "
                f"operator rows {self.operator.shape[0]}"
            )

        correction = jnp.zeros_like(residual_vec)
        remaining = residual_vec
        damping = jnp.asarray(float(self.metadata.damping), dtype=residual_vec.dtype)
        inverse_diagonal = jnp.asarray(self.inverse_diagonal, dtype=residual_vec.dtype)
        for _ in range(max(1, int(self.metadata.sweeps))):
            step = damping * inverse_diagonal * remaining
            correction = correction + step
            remaining = remaining - self.operator.matvec(step)
        return correction

    def as_preconditioner(self) -> Callable[[ArrayLike], ArrayLike]:
        """Return a callable suitable for QI local-smoother hooks."""

        return self.apply


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
    sweeps_use = max(1, int(sweeps))
    floor_use = max(0.0, float(diagonal_floor))
    diagonal, hit_count = extract_device_csr_diagonal(device_operator)
    metadata = _diagonal_metadata(
        device_operator=device_operator,
        diagonal=diagonal,
        hit_count=hit_count,
        damping=damping_use,
        sweeps=sweeps_use,
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


__all__ = [
    "RHS1QIDeviceJacobiSmoother",
    "RHS1QIDeviceJacobiSmootherMetadata",
    "build_rhs1_qi_device_jacobi_smoother",
    "extract_device_csr_diagonal",
]
