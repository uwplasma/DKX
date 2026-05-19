"""Device-resident two-level preconditioner for RHSMode=1 QI lanes.

This module is the first production-shaped primitive for the true device-QI
research lane.  It keeps the preconditioner state as JAX arrays plus static
metadata:

``M^{-1} r = S_local^{-1} r + P_c A_c^{-1} R_c (r - A S_local^{-1} r)``.

``S_local`` is a CSR-backed device Jacobi/stationary smoother when a device CSR
operator is available.  For larger QI seeds the module can also run a coarse-only
matrix-free path: it builds only ``A Q`` by applying a JAX matvec to rank-gated
coarse vectors, avoiding full CSR materialization.  Both variants keep the timed
apply path free of SciPy, Python callbacks, and host factors.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from .rhs1_device_operator import DeviceCSR
from .rhs1_qi_coarse import RHS1QICoarseBasis, orthonormalize_rhs1_qi_coarse_basis
from .rhs1_qi_device_smoother import (
    RHS1QIDeviceJacobiSmoother,
    build_rhs1_qi_device_jacobi_smoother,
)

ArrayLike = Any


@dataclass(frozen=True)
class RHS1QIDevicePreconditionerConfig:
    """Static controls for a device QI two-level preconditioner."""

    regularization_rcond: float = 1.0e-12
    damping: float = 1.0
    coarse_solver: str = "action_lstsq"
    composition: str = "multiplicative"
    basis_rtol: float = 1.0e-10
    basis_atol: float = 0.0
    max_rank: int | None = None
    jacobi_damping: float = 0.7
    jacobi_sweeps: int = 1
    jacobi_step_policy: str = "stationary"
    jacobi_diagonal_floor: float = 1.0e-14
    jacobi_require_all_diagonal: bool = True
    local_smoother_kind: str = "auto"
    matrix_free_smoother_sweeps: int = 1
    matrix_free_smoother_damping: float = 1.0
    matrix_free_smoother_step_policy: str = "residual_minimizing"
    matrix_free_smoother_alpha_clip: float = 10.0
    matrix_free_smoother_min_denominator: float = 1.0e-300
    matrix_free_block_smoother_max_groups: int = 32
    matrix_free_block_smoother_include_tail: bool = True
    matrix_free_block_smoother_rcond: float = 1.0e-12
    matrix_free_block_smoother_grouping: str = "contiguous"
    residual_enrichment: bool = False
    residual_enrichment_depth: int = 0
    residual_enrichment_include_residual: bool = True
    recycle_enrichment: bool = False
    recycle_enrichment_cycles: int = 0


@dataclass(frozen=True)
class RHS1QIDevicePreconditionerMetadata:
    """JSON-friendly diagnostics for the device-QI preconditioner state."""

    shape: tuple[int, int]
    nnz: int
    rank: int
    operator_source: str
    coarse_operator_shape: tuple[int, int]
    operator_on_basis_shape: tuple[int, int]
    coarse_operator_norm: float
    operator_on_basis_norm: float
    regularization_rcond: float
    damping: float
    coarse_solver: str
    composition: str
    local_smoother_kind: str
    local_smoother_reason: str
    device_resident: bool
    host_fallback_used: bool
    host_callback_free: bool
    operator_metadata_keys: tuple[str, ...]
    geometry_metadata_keys: tuple[str, ...]
    accepted_basis_labels: tuple[str, ...]
    residual_enrichment_enabled: bool
    residual_enrichment_depth: int
    residual_enrichment_candidate_count: int
    recycle_enrichment_enabled: bool
    recycle_enrichment_cycles: int
    recycle_enrichment_candidate_count: int
    reason: str

    def to_dict(self) -> dict[str, object]:
        """Return a plain mapping suitable for solver traces and JSON."""

        payload = asdict(self)
        payload["shape"] = tuple(int(v) for v in self.shape)
        payload["coarse_operator_shape"] = tuple(int(v) for v in self.coarse_operator_shape)
        payload["operator_on_basis_shape"] = tuple(int(v) for v in self.operator_on_basis_shape)
        return payload


@dataclass(frozen=True)
class RHS1QIDevicePreconditionerState:
    """Reusable pure-JAX local-plus-coarse QI preconditioner action."""

    operator: DeviceCSR | None
    operator_matvec: Callable[[ArrayLike], ArrayLike]
    dtype: Any
    shape: tuple[int, int]
    local_smoother: (
        RHS1QIDeviceJacobiSmoother
        | "RHS1QIMatrixFreeResidualSmoother"
        | "RHS1QIMatrixFreeProjectedResidualSmoother"
        | None
    )
    basis: RHS1QICoarseBasis
    operator_on_basis: ArrayLike
    coarse_operator: ArrayLike
    metadata: RHS1QIDevicePreconditionerMetadata

    def solve_coarse(self, residual: ArrayLike) -> ArrayLike:
        """Solve the small coarse problem for the current residual."""

        residual_vec = jnp.asarray(residual, dtype=self.dtype).reshape((-1,))
        rank = int(self.metadata.rank)
        if rank <= 0:
            return jnp.zeros((0,), dtype=residual_vec.dtype)
        if self.metadata.coarse_solver == "action_lstsq":
            return _regularized_least_squares(
                self.operator_on_basis,
                residual_vec,
                rcond=float(self.metadata.regularization_rcond),
            )
        projected = jnp.conjugate(self.basis.vectors).T @ residual_vec
        return _regularized_least_squares(
            self.coarse_operator,
            projected,
            rcond=float(self.metadata.regularization_rcond),
        )

    def apply(self, residual: ArrayLike) -> ArrayLike:
        """Apply one field-split/two-level correction to ``residual``."""

        residual_vec = jnp.asarray(residual, dtype=self.dtype).reshape((-1,))
        if int(residual_vec.shape[0]) != int(self.shape[0]):
            raise ValueError(
                f"residual length {residual_vec.shape[0]} does not match operator rows {self.shape[0]}"
            )
        if self.local_smoother is None:
            local = jnp.zeros_like(residual_vec)
        else:
            local = jnp.asarray(self.local_smoother.apply(residual_vec), dtype=residual_vec.dtype).reshape((-1,))
        if int(self.metadata.rank) <= 0:
            return float(self.metadata.damping) * local

        if self.metadata.composition == "multiplicative" and self.local_smoother is not None:
            coarse_input = residual_vec - jnp.asarray(self.operator_matvec(local), dtype=residual_vec.dtype).reshape((-1,))
        else:
            coarse_input = residual_vec
        coefficients = self.solve_coarse(coarse_input)
        coarse = jnp.asarray(self.basis.vectors, dtype=residual_vec.dtype) @ coefficients
        return float(self.metadata.damping) * (local + coarse)

    def as_preconditioner(self):
        """Return a callable for Krylov hooks."""

        return self.apply


@dataclass(frozen=True)
class RHS1QIDevicePreconditionerProbe:
    """Fail-closed true-residual probe for a device-QI candidate."""

    accepted: bool
    reason: str
    residual_before_norm: float
    residual_after_norm: float
    improvement_ratio: float | None
    metadata: RHS1QIDevicePreconditionerMetadata
    cycles: int = 0
    residual_history: tuple[float, ...] = ()
    step_history: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return JSON-friendly probe diagnostics."""

        return {
            "accepted": bool(self.accepted),
            "reason": self.reason,
            "residual_before_norm": float(self.residual_before_norm),
            "residual_after_norm": float(self.residual_after_norm),
            "improvement_ratio": None if self.improvement_ratio is None else float(self.improvement_ratio),
            "metadata": self.metadata.to_dict(),
            "cycles": int(self.cycles),
            "residual_history": tuple(float(value) for value in self.residual_history),
            "step_history": tuple(float(value) for value in self.step_history),
        }


@dataclass(frozen=True)
class RHS1QIMatrixFreeResidualSmootherMetadata:
    """Diagnostics for a bounded matrix-free local smoother."""

    shape: tuple[int, int]
    sweeps: int
    damping: float
    step_policy: str
    alpha_clip: float
    min_denominator: float
    device_resident: bool
    source: str
    reason: str

    def to_dict(self) -> dict[str, object]:
        """Return JSON-friendly matrix-free smoother diagnostics."""

        return {
            "shape": tuple(int(v) for v in self.shape),
            "sweeps": int(self.sweeps),
            "damping": float(self.damping),
            "step_policy": self.step_policy,
            "alpha_clip": float(self.alpha_clip),
            "min_denominator": float(self.min_denominator),
            "device_resident": bool(self.device_resident),
            "source": self.source,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RHS1QIMatrixFreeResidualSmoother:
    """Pure-JAX matrix-free local smoother for large QI seeds.

    Each sweep applies a Richardson/minimal-residual step using the current
    residual as the search direction.  This is deliberately bounded: it is a
    local preconditioner component, not an unbounded Krylov solve.
    """

    operator_matvec: Callable[[ArrayLike], ArrayLike]
    dtype: Any
    metadata: RHS1QIMatrixFreeResidualSmootherMetadata

    def apply(self, residual: ArrayLike) -> ArrayLike:
        """Apply fixed-count residual-polynomial smoothing to ``residual``."""

        residual_vec = jnp.asarray(residual, dtype=self.dtype).reshape((-1,))
        if int(residual_vec.shape[0]) != int(self.metadata.shape[0]):
            raise ValueError(
                f"residual length {residual_vec.shape[0]} does not match "
                f"operator rows {self.metadata.shape[0]}"
            )
        correction = jnp.zeros_like(residual_vec)
        remaining = residual_vec
        damping = jnp.asarray(float(self.metadata.damping), dtype=residual_vec.dtype)
        min_denominator = jnp.asarray(max(0.0, float(self.metadata.min_denominator)), dtype=residual_vec.dtype)
        alpha_clip = jnp.asarray(max(0.0, float(self.metadata.alpha_clip)), dtype=residual_vec.dtype)
        for _ in range(max(1, int(self.metadata.sweeps))):
            direction = remaining
            action = jnp.asarray(self.operator_matvec(direction), dtype=residual_vec.dtype).reshape((-1,))
            if self.metadata.step_policy == "residual_minimizing":
                numerator = jnp.real(jnp.vdot(action, remaining))
                denominator = jnp.real(jnp.vdot(action, action))
                valid = (
                    jnp.isfinite(numerator)
                    & jnp.isfinite(denominator)
                    & (denominator > min_denominator)
                )
                raw_alpha = numerator / jnp.where(valid, denominator, jnp.asarray(1.0, dtype=denominator.dtype))
                if float(self.metadata.alpha_clip) > 0.0:
                    raw_alpha = jnp.clip(raw_alpha, -alpha_clip, alpha_clip)
                alpha = jnp.where(valid & jnp.isfinite(raw_alpha), raw_alpha, jnp.asarray(0.0, dtype=raw_alpha.dtype))
            elif self.metadata.step_policy == "stationary":
                alpha = jnp.asarray(1.0, dtype=residual_vec.dtype)
            else:
                raise ValueError(f"unsupported matrix-free smoother step policy {self.metadata.step_policy!r}")
            step_scale = damping * alpha
            correction = correction + step_scale * direction
            remaining = remaining - step_scale * action
        return correction

    def as_preconditioner(self) -> Callable[[ArrayLike], ArrayLike]:
        """Return a callable suitable for local-smoother hooks."""

        return self.apply


@dataclass(frozen=True)
class RHS1QIMatrixFreeProjectedResidualSmootherMetadata:
    """Diagnostics for projected block residual smoothing."""

    shape: tuple[int, int]
    group_slices: tuple[tuple[int, int], ...]
    group_partitions: tuple[tuple[tuple[int, int], ...], ...]
    sweeps: int
    damping: float
    regularization_rcond: float
    block_count: int
    group_count: int
    max_groups: int
    include_tail: bool
    grouping: str
    device_resident: bool
    source: str
    reason: str

    def to_dict(self) -> dict[str, object]:
        """Return JSON-friendly projected smoother diagnostics."""

        return {
            "shape": tuple(int(v) for v in self.shape),
            "group_slices": tuple((int(start), int(stop)) for start, stop in self.group_slices),
            "group_partitions": tuple(
                tuple((int(start), int(stop)) for start, stop in partition)
                for partition in self.group_partitions
            ),
            "sweeps": int(self.sweeps),
            "damping": float(self.damping),
            "regularization_rcond": float(self.regularization_rcond),
            "block_count": int(self.block_count),
            "group_count": int(self.group_count),
            "max_groups": int(self.max_groups),
            "include_tail": bool(self.include_tail),
            "grouping": self.grouping,
            "device_resident": bool(self.device_resident),
            "source": self.source,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RHS1QIMatrixFreeProjectedResidualSmoother:
    """Pure-JAX projected block/angular/radial residual smoother.

    Each sweep splits the current residual into structural x/species block
    pieces, optional x/species aggregate pieces, applies the matrix-free
    operator to those pieces, and solves the small problem
    ``min_c ||r - A D c||_2``.  The lifted correction ``D c`` is a bounded
    additive-Schwarz-like local action that keeps all angular content inside
    each selected partition while avoiding full CSR materialization.
    """

    operator_matvec: Callable[[ArrayLike], ArrayLike]
    dtype: Any
    metadata: RHS1QIMatrixFreeProjectedResidualSmootherMetadata

    def _project_partition(
        self,
        residual: ArrayLike,
        partition: Sequence[tuple[int, int]],
    ) -> ArrayLike:
        residual_vec = jnp.asarray(residual, dtype=self.dtype).reshape((-1,))
        direction = jnp.zeros_like(residual_vec)
        for start, stop in partition:
            direction = direction.at[int(start) : int(stop)].set(residual_vec[int(start) : int(stop)])
        return direction

    def apply(self, residual: ArrayLike) -> ArrayLike:
        """Apply fixed-count projected residual smoothing to ``residual``."""

        residual_vec = jnp.asarray(residual, dtype=self.dtype).reshape((-1,))
        if int(residual_vec.shape[0]) != int(self.metadata.shape[0]):
            raise ValueError(
                f"residual length {residual_vec.shape[0]} does not match "
                f"operator rows {self.metadata.shape[0]}"
            )
        correction = jnp.zeros_like(residual_vec)
        remaining = residual_vec
        damping = jnp.asarray(float(self.metadata.damping), dtype=residual_vec.dtype)
        for _ in range(max(1, int(self.metadata.sweeps))):
            directions = tuple(
                self._project_partition(remaining, partition)
                for partition in self.metadata.group_partitions
            )
            direction_matrix = jnp.stack(directions, axis=1)
            action_matrix = _operator_on_basis(
                self.operator_matvec,
                direction_matrix,
                shape=self.metadata.shape,
                dtype=residual_vec.dtype,
            )
            coefficients = _regularized_least_squares(
                action_matrix,
                remaining,
                rcond=float(self.metadata.regularization_rcond),
            )
            step = direction_matrix @ coefficients
            action = action_matrix @ coefficients
            correction = correction + damping * step
            remaining = remaining - damping * action
        return correction

    def as_preconditioner(self) -> Callable[[ArrayLike], ArrayLike]:
        """Return a callable suitable for local-smoother hooks."""

        return self.apply


def _normalize_coarse_solver(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "action_lstsq": "action_lstsq",
        "action_ls": "action_lstsq",
        "lstsq": "action_lstsq",
        "least_squares": "action_lstsq",
        "projected": "projected",
        "galerkin": "projected",
        "qtaq": "projected",
    }
    if normalized not in aliases:
        raise ValueError("coarse_solver must be 'action_lstsq' or 'projected'")
    return aliases[normalized]


def _normalize_composition(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "add": "additive",
        "additive": "additive",
        "mult": "multiplicative",
        "multiplicative": "multiplicative",
        "field_split": "multiplicative",
        "schur": "multiplicative",
    }
    if normalized not in aliases:
        raise ValueError("composition must be 'multiplicative' or 'additive'")
    return aliases[normalized]


def _normalize_matrix_free_smoother_step_policy(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "fixed": "stationary",
        "richardson": "stationary",
        "stationary": "stationary",
        "minres": "residual_minimizing",
        "minimum_residual": "residual_minimizing",
        "residual_minimizing": "residual_minimizing",
        "residual_reducing": "residual_minimizing",
    }
    if normalized not in aliases:
        raise ValueError("matrix_free_smoother_step_policy must be 'stationary' or 'residual_minimizing'")
    return aliases[normalized]


def _normalize_matrix_free_block_smoother_grouping(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "contiguous": "contiguous",
        "block": "contiguous",
        "blocks": "contiguous",
        "block_contiguous": "contiguous",
        "hybrid": "block_x_species",
        "aggregate": "block_x_species",
        "aggregates": "block_x_species",
        "block_x": "block_x_species",
        "block_species": "block_x_species",
        "x_species": "block_x_species",
        "block_x_species": "block_x_species",
        "radial_species": "block_x_species",
        "block_radial_species": "block_x_species",
    }
    if normalized not in aliases:
        raise ValueError("matrix_free_block_smoother_grouping must be 'contiguous' or 'block_x_species'")
    return aliases[normalized]


def _metadata_int_tuple(metadata: Mapping[str, object] | None, key: str) -> tuple[int, ...]:
    if metadata is None or key not in metadata:
        return ()
    value = metadata[key]
    if isinstance(value, str):
        raw_values: Sequence[object] = tuple(part for part in value.replace(",", " ").split() if part)
    elif isinstance(value, Sequence):
        raw_values = value
    else:
        return ()
    result: list[int] = []
    for item in raw_values:
        try:
            result.append(int(item))
        except (TypeError, ValueError):
            return ()
    return tuple(result)


def _merge_group_slices(
    slices: Sequence[tuple[int, int]],
    *,
    max_groups: int,
) -> tuple[tuple[int, int], ...]:
    valid = tuple((int(start), int(stop)) for start, stop in slices if int(stop) > int(start))
    if not valid:
        return ()
    group_limit = max(1, int(max_groups))
    if len(valid) <= group_limit:
        return valid
    merged: list[tuple[int, int]] = []
    n_slices = len(valid)
    for group_index in range(group_limit):
        first = int(group_index * n_slices // group_limit)
        last = int((group_index + 1) * n_slices // group_limit)
        if last <= first:
            continue
        merged.append((valid[first][0], valid[last - 1][1]))
    return tuple(merged)


def _partition_bounds(partition: Sequence[tuple[int, int]]) -> tuple[int, int]:
    starts = [int(start) for start, _ in partition]
    stops = [int(stop) for _, stop in partition]
    return (min(starts), max(stops))


def _matrix_free_block_group_partitions(
    *,
    shape: tuple[int, int],
    geometry_metadata: Mapping[str, object] | None,
    config: RHS1QIDevicePreconditionerConfig,
) -> tuple[
    tuple[tuple[tuple[int, int], ...], ...],
    tuple[tuple[int, int], ...],
    int,
    str,
]:
    block_sizes = _metadata_int_tuple(geometry_metadata, "qi_block_sizes")
    if not block_sizes:
        raise ValueError(
            "matrix_free_block_minres local smoother requires geometry_metadata['qi_block_sizes']"
        )
    if any(size <= 0 for size in block_sizes):
        raise ValueError("qi_block_sizes entries must be positive")
    n_rows = int(shape[0])
    offsets = [0]
    for size in block_sizes:
        offsets.append(offsets[-1] + int(size))
    if offsets[-1] > n_rows:
        raise ValueError(f"qi_block_sizes sum {offsets[-1]} exceeds operator rows {n_rows}")
    block_slices = tuple((offsets[index], offsets[index + 1]) for index in range(len(block_sizes)))
    tail_slice = (offsets[-1], n_rows) if offsets[-1] < n_rows else None
    grouping = _normalize_matrix_free_block_smoother_grouping(config.matrix_free_block_smoother_grouping)
    max_groups = max(1, int(config.matrix_free_block_smoother_max_groups))
    if grouping == "contiguous":
        slices = list(block_slices)
        if bool(config.matrix_free_block_smoother_include_tail) and tail_slice is not None:
            slices.append(tail_slice)
        group_slices = _merge_group_slices(slices, max_groups=max_groups)
        group_partitions = tuple(((int(start), int(stop)),) for start, stop in group_slices)
    else:
        block_x = _metadata_int_tuple(geometry_metadata, "qi_block_x")
        block_species = _metadata_int_tuple(geometry_metadata, "qi_block_species")
        aggregate_partitions: list[tuple[tuple[int, int], ...]] = []
        if len(block_x) == len(block_slices):
            for x_index in sorted(set(int(value) for value in block_x)):
                aggregate_partitions.append(
                    tuple(
                        block_slices[index]
                        for index, value in enumerate(block_x)
                        if int(value) == int(x_index)
                    )
                )
        if len(block_species) == len(block_slices):
            for species_index in sorted(set(int(value) for value in block_species)):
                aggregate_partitions.append(
                    tuple(
                        block_slices[index]
                        for index, value in enumerate(block_species)
                        if int(value) == int(species_index)
                    )
                )
        tail_partitions: list[tuple[tuple[int, int], ...]] = []
        if bool(config.matrix_free_block_smoother_include_tail) and tail_slice is not None:
            tail_partitions.append((tail_slice,))
        reserved_groups = len(aggregate_partitions) + len(tail_partitions)
        block_group_limit = max(1, max_groups - reserved_groups) if max_groups > reserved_groups else max_groups
        block_partitions = tuple(
            ((int(start), int(stop)),)
            for start, stop in _merge_group_slices(block_slices, max_groups=block_group_limit)
        )
        group_partitions = tuple(block_partitions + tuple(aggregate_partitions) + tuple(tail_partitions))
        if len(group_partitions) > max_groups:
            group_partitions = group_partitions[:max_groups]
        group_slices = tuple(_partition_bounds(partition) for partition in group_partitions)
    if not group_slices:
        raise ValueError("matrix_free_block_minres local smoother found no non-empty groups")
    return group_partitions, group_slices, len(block_sizes), grouping


def _build_matrix_free_residual_smoother(
    *,
    operator_matvec: Callable[[ArrayLike], ArrayLike],
    shape: tuple[int, int],
    dtype: Any,
    config: RHS1QIDevicePreconditionerConfig,
) -> RHS1QIMatrixFreeResidualSmoother:
    """Build a bounded device-compatible matrix-free local smoother."""

    if int(shape[0]) != int(shape[1]):
        raise ValueError("matrix-free residual smoother requires a square operator")
    sweeps = max(1, int(config.matrix_free_smoother_sweeps))
    damping = float(config.matrix_free_smoother_damping)
    if not np.isfinite(damping) or damping <= 0.0:
        raise ValueError("matrix_free_smoother_damping must be finite and positive")
    alpha_clip = max(0.0, float(config.matrix_free_smoother_alpha_clip))
    min_denominator = max(0.0, float(config.matrix_free_smoother_min_denominator))
    metadata = RHS1QIMatrixFreeResidualSmootherMetadata(
        shape=tuple(int(value) for value in shape),
        sweeps=sweeps,
        damping=damping,
        step_policy=_normalize_matrix_free_smoother_step_policy(config.matrix_free_smoother_step_policy),
        alpha_clip=alpha_clip,
        min_denominator=min_denominator,
        device_resident=True,
        source="matrix_free_matvec",
        reason="built",
    )
    return RHS1QIMatrixFreeResidualSmoother(
        operator_matvec=operator_matvec,
        dtype=dtype,
        metadata=metadata,
    )


def _build_matrix_free_projected_residual_smoother(
    *,
    operator_matvec: Callable[[ArrayLike], ArrayLike],
    shape: tuple[int, int],
    dtype: Any,
    geometry_metadata: Mapping[str, object] | None,
    config: RHS1QIDevicePreconditionerConfig,
) -> RHS1QIMatrixFreeProjectedResidualSmoother:
    """Build a bounded projected block residual smoother."""

    if int(shape[0]) != int(shape[1]):
        raise ValueError("matrix-free block smoother requires a square operator")
    sweeps = max(1, int(config.matrix_free_smoother_sweeps))
    damping = float(config.matrix_free_smoother_damping)
    if not np.isfinite(damping) or damping <= 0.0:
        raise ValueError("matrix_free_smoother_damping must be finite and positive")
    rcond = float(config.matrix_free_block_smoother_rcond)
    if not np.isfinite(rcond) or rcond <= 0.0:
        raise ValueError("matrix_free_block_smoother_rcond must be finite and positive")
    group_partitions, group_slices, block_count, grouping = _matrix_free_block_group_partitions(
        shape=shape,
        geometry_metadata=geometry_metadata,
        config=config,
    )
    metadata = RHS1QIMatrixFreeProjectedResidualSmootherMetadata(
        shape=tuple(int(value) for value in shape),
        group_slices=group_slices,
        group_partitions=group_partitions,
        sweeps=sweeps,
        damping=damping,
        regularization_rcond=rcond,
        block_count=int(block_count),
        group_count=int(len(group_slices)),
        max_groups=max(1, int(config.matrix_free_block_smoother_max_groups)),
        include_tail=bool(config.matrix_free_block_smoother_include_tail),
        grouping=grouping,
        device_resident=True,
        source="matrix_free_block_projections",
        reason="built",
    )
    return RHS1QIMatrixFreeProjectedResidualSmoother(
        operator_matvec=operator_matvec,
        dtype=dtype,
        metadata=metadata,
    )


def _regularized_least_squares(matrix: ArrayLike, rhs: ArrayLike, *, rcond: float) -> ArrayLike:
    a = jnp.asarray(matrix)
    b = jnp.asarray(rhs, dtype=a.dtype).reshape((-1,))
    if a.ndim != 2:
        raise ValueError("coarse least-squares matrix must be two-dimensional")
    if int(a.shape[1]) == 0:
        return jnp.zeros((0,), dtype=b.dtype)
    gram = jnp.conjugate(a).T @ a
    normal_rhs = jnp.conjugate(a).T @ b
    scale = jnp.maximum(jnp.linalg.norm(gram), jnp.asarray(1.0, dtype=gram.dtype))
    ridge = jnp.asarray(max(float(rcond), 0.0), dtype=gram.dtype) * scale
    eye = jnp.eye(int(gram.shape[0]), dtype=gram.dtype)
    return jnp.linalg.solve(gram + ridge * eye, normal_rhs)


def _operator_on_basis(
    operator_matvec: Callable[[ArrayLike], ArrayLike],
    basis_vectors: ArrayLike,
    *,
    shape: tuple[int, int],
    dtype: Any,
) -> ArrayLike:
    q = jnp.asarray(basis_vectors, dtype=dtype)
    if q.ndim != 2:
        raise ValueError("basis vectors must be a matrix")
    if int(q.shape[0]) != int(shape[1]):
        raise ValueError(f"basis row count {q.shape[0]} does not match operator columns {shape[1]}")
    if int(q.shape[1]) == 0:
        return jnp.zeros((int(shape[0]), 0), dtype=dtype)
    columns = [jnp.asarray(operator_matvec(q[:, idx]), dtype=dtype).reshape((-1,)) for idx in range(int(q.shape[1]))]
    return jnp.stack(columns, axis=1)


def _append_normalized_candidate(
    columns: list[ArrayLike],
    labels: list[str],
    values: ArrayLike,
    label: str,
    *,
    total_size: int,
    dtype: Any,
) -> None:
    vector = jnp.asarray(values, dtype=dtype).reshape((-1,))
    if int(vector.shape[0]) != int(total_size):
        raise ValueError(f"candidate {label!r} has length {vector.shape[0]}, expected {total_size}")
    norm = float(jnp.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0.0:
        return
    columns.append(vector / norm)
    labels.append(str(label))


def _enrich_basis_with_residual(
    *,
    operator_matvec: Callable[[ArrayLike], ArrayLike],
    basis: RHS1QICoarseBasis,
    residual_seed: ArrayLike,
    shape: tuple[int, int],
    dtype: Any,
    config: RHS1QIDevicePreconditionerConfig,
) -> tuple[RHS1QICoarseBasis, int]:
    """Add residual-generated matrix-free directions to a physics coarse basis.

    For large QI seeds the full CSR operator can be too expensive to keep on
    device.  This enrichment builds a bounded correction-space Krylov basis
    ``{r, A r, A^2 r, ...}`` using only matrix-vector products.  The resulting
    basis still goes through the same rank gate and true-residual acceptance
    probe, so weak or harmful enrichments fail closed.
    """

    residual = jnp.asarray(residual_seed, dtype=dtype).reshape((-1,))
    total_size = int(shape[1])
    if int(residual.shape[0]) != total_size:
        raise ValueError(f"residual_seed length {residual.shape[0]} does not match operator size {total_size}")

    columns: list[ArrayLike] = []
    labels: list[str] = []
    base_vectors = jnp.asarray(basis.vectors, dtype=dtype)
    for index, label in enumerate(basis.metadata.accepted_labels):
        _append_normalized_candidate(
            columns,
            labels,
            base_vectors[:, int(index)],
            f"base:{label}",
            total_size=total_size,
            dtype=dtype,
        )

    residual_candidate_count = 0
    current = residual
    if bool(config.residual_enrichment_include_residual):
        before = len(columns)
        _append_normalized_candidate(
            columns,
            labels,
            current,
            "residual:0",
            total_size=total_size,
            dtype=dtype,
        )
        residual_candidate_count += len(columns) - before

    for depth in range(max(0, int(config.residual_enrichment_depth))):
        current = jnp.asarray(operator_matvec(current), dtype=dtype).reshape((-1,))
        before = len(columns)
        _append_normalized_candidate(
            columns,
            labels,
            current,
            f"operator_power:{depth + 1}",
            total_size=total_size,
            dtype=dtype,
        )
        residual_candidate_count += len(columns) - before

    if columns:
        candidates = jnp.stack(tuple(columns), axis=1)
    else:
        candidates = jnp.zeros((total_size, 0), dtype=dtype)
    enriched = orthonormalize_rhs1_qi_coarse_basis(
        candidates,
        labels=tuple(labels),
        rtol=float(config.basis_rtol),
        atol=float(config.basis_atol),
        max_rank=config.max_rank,
    )
    return enriched, residual_candidate_count


def _coarse_action_residual(
    *,
    operator_matvec: Callable[[ArrayLike], ArrayLike],
    basis: RHS1QICoarseBasis,
    residual: ArrayLike,
    shape: tuple[int, int],
    dtype: Any,
    rcond: float,
) -> ArrayLike:
    """Return the true residual after the best current coarse correction."""

    residual_vec = jnp.asarray(residual, dtype=dtype).reshape((-1,))
    if int(basis.metadata.rank) <= 0:
        return residual_vec
    aq = _operator_on_basis(operator_matvec, basis.vectors, shape=shape, dtype=dtype)
    coefficients = _regularized_least_squares(aq, residual_vec, rcond=float(rcond))
    correction = jnp.asarray(basis.vectors, dtype=dtype) @ coefficients
    return residual_vec - jnp.asarray(operator_matvec(correction), dtype=dtype).reshape((-1,))


def _enrich_basis_with_recycle_residuals(
    *,
    operator_matvec: Callable[[ArrayLike], ArrayLike],
    basis: RHS1QICoarseBasis,
    residual_seed: ArrayLike,
    shape: tuple[int, int],
    dtype: Any,
    config: RHS1QIDevicePreconditionerConfig,
) -> tuple[RHS1QICoarseBasis, int]:
    """Append residuals left by bounded coarse corrections.

    This is a device-compatible GCRO-style seed: the current coarse space first
    removes what it can from the true residual, then the remaining slow residual
    is appended as a new candidate direction.  Repeating this for a small number
    of cycles builds a recycle space targeted at the actual hard seed without
    host factors, dense full operators, or unbounded Krylov work.
    """

    residual = jnp.asarray(residual_seed, dtype=dtype).reshape((-1,))
    total_size = int(shape[1])
    if int(residual.shape[0]) != total_size:
        raise ValueError(f"residual_seed length {residual.shape[0]} does not match operator size {total_size}")

    current_basis = basis
    current_residual = residual
    candidate_count = 0
    for cycle in range(max(0, int(config.recycle_enrichment_cycles))):
        current_residual = _coarse_action_residual(
            operator_matvec=operator_matvec,
            basis=current_basis,
            residual=current_residual,
            shape=shape,
            dtype=dtype,
            rcond=float(config.regularization_rcond),
        )
        columns = [jnp.asarray(current_basis.vectors, dtype=dtype)[:, idx] for idx in range(int(current_basis.metadata.rank))]
        labels = [str(label) for label in current_basis.metadata.accepted_labels]
        before = len(columns)
        _append_normalized_candidate(
            columns,
            labels,
            current_residual,
            f"recycle_residual:{cycle}",
            total_size=total_size,
            dtype=dtype,
        )
        candidate_count += len(columns) - before
        if not columns:
            break
        current_basis = orthonormalize_rhs1_qi_coarse_basis(
            jnp.stack(tuple(columns), axis=1),
            labels=tuple(labels),
            rtol=float(config.basis_rtol),
            atol=float(config.basis_atol),
            max_rank=config.max_rank,
        )
    return current_basis, candidate_count


def _basis_from_value(
    coarse_basis: RHS1QICoarseBasis | ArrayLike | None,
    *,
    total_size: int,
    dtype: Any,
    labels: Sequence[str] | None,
    config: RHS1QIDevicePreconditionerConfig,
) -> RHS1QICoarseBasis:
    if isinstance(coarse_basis, RHS1QICoarseBasis):
        basis = coarse_basis
    elif coarse_basis is None:
        basis = orthonormalize_rhs1_qi_coarse_basis(
            jnp.zeros((int(total_size), 0), dtype=dtype),
            labels=(),
            rtol=float(config.basis_rtol),
            atol=float(config.basis_atol),
            max_rank=config.max_rank,
        )
    else:
        basis = orthonormalize_rhs1_qi_coarse_basis(
            jnp.asarray(coarse_basis, dtype=dtype),
            labels=labels,
            rtol=float(config.basis_rtol),
            atol=float(config.basis_atol),
            max_rank=config.max_rank,
        )
    vectors = jnp.asarray(basis.vectors, dtype=dtype)
    if vectors.ndim != 2 or int(vectors.shape[0]) != int(total_size):
        raise ValueError(f"coarse basis must have shape ({total_size}, rank)")
    if vectors.dtype != jnp.dtype(dtype):
        basis = RHS1QICoarseBasis(vectors=vectors, metadata=basis.metadata)
    return basis


def _metadata_keys(value: Mapping[str, object] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(sorted(str(key) for key in value.keys()))


def setup_rhs1_qi_device_preconditioner(
    *,
    operator: DeviceCSR | Callable[[ArrayLike], ArrayLike],
    coarse_basis: RHS1QICoarseBasis | ArrayLike | None = None,
    local_smoother: RHS1QIDeviceJacobiSmoother | None = None,
    coarse_labels: Sequence[str] | None = None,
    residual_seed: ArrayLike | None = None,
    total_size: int | None = None,
    dtype: Any = jnp.float64,
    operator_metadata: Mapping[str, object] | None = None,
    geometry_metadata: Mapping[str, object] | None = None,
    config: RHS1QIDevicePreconditionerConfig | None = None,
) -> RHS1QIDevicePreconditionerState:
    """Build a device-resident QI preconditioner state.

    The returned state is intentionally standalone and does not change any
    production solver defaults.  It can be closed over by JAX transforms and later
    wired into ``v3_driver.py`` behind an explicit opt-in.
    """

    config_use = RHS1QIDevicePreconditionerConfig() if config is None else config
    coarse_solver = _normalize_coarse_solver(config_use.coarse_solver)
    composition = _normalize_composition(config_use.composition)
    local_smoother_kind_requested = str(config_use.local_smoother_kind).strip().lower().replace("-", "_")
    matrix_free_residual_smoother_tokens = {
        "matrix_free",
        "matrix_free_residual",
        "matrix_free_minres",
        "matrix_free_richardson",
        "residual_polynomial",
    }
    matrix_free_block_smoother_tokens = {
        "block_minres",
        "matrix_free_block",
        "matrix_free_block_minres",
        "block_angular_radial",
        "projected_block_minres",
    }
    matrix_free_smoother_tokens = matrix_free_residual_smoother_tokens | matrix_free_block_smoother_tokens
    if local_smoother_kind_requested not in {
        "auto",
        "device_jacobi",
        "jacobi",
        "none",
        "coarse_only",
        *matrix_free_smoother_tokens,
    }:
        raise ValueError("local_smoother_kind must be 'auto', 'device_jacobi', 'matrix_free_minres', or 'none'")
    damping = float(config_use.damping)
    if not np.isfinite(damping) or damping <= 0.0:
        raise ValueError("damping must be finite and positive")
    rcond = float(config_use.regularization_rcond)
    if not np.isfinite(rcond) or rcond < 0.0:
        raise ValueError("regularization_rcond must be finite and non-negative")

    if isinstance(operator, DeviceCSR):
        operator_csr: DeviceCSR | None = operator
        operator_matvec: Callable[[ArrayLike], ArrayLike] = operator.matvec
        shape = tuple(int(v) for v in operator.shape)
        dtype_use = operator.data.dtype
        nnz = int(operator.nnz)
        operator_source = "device_csr"
    else:
        if total_size is None:
            if isinstance(coarse_basis, RHS1QICoarseBasis):
                total_size = int(coarse_basis.vectors.shape[0])
            elif coarse_basis is not None:
                total_size = int(jnp.asarray(coarse_basis).shape[0])
        if total_size is None or int(total_size) <= 0:
            raise ValueError("total_size is required for a matrix-free QI device preconditioner")
        operator_csr = None
        operator_matvec = operator
        shape = (int(total_size), int(total_size))
        dtype_use = jnp.dtype(dtype)
        nnz = 0
        operator_source = "matrix_free"

    if int(shape[0]) != int(shape[1]):
        raise ValueError("device QI preconditioner requires a square operator")

    smoother = local_smoother
    if smoother is None and operator_csr is not None and local_smoother_kind_requested not in {"none", "coarse_only"}:
        smoother = build_rhs1_qi_device_jacobi_smoother(
            operator_csr,
            damping=float(config_use.jacobi_damping),
            sweeps=int(config_use.jacobi_sweeps),
            step_policy=str(config_use.jacobi_step_policy),
            diagonal_floor=float(config_use.jacobi_diagonal_floor),
            require_all_diagonal=bool(config_use.jacobi_require_all_diagonal),
        )
    elif (
        smoother is not None
        and isinstance(smoother, RHS1QIDeviceJacobiSmoother)
        and smoother.operator.shape != shape
    ):
        raise ValueError("local_smoother operator shape must match operator shape")
    elif smoother is None and operator_csr is None and local_smoother_kind_requested in matrix_free_block_smoother_tokens:
        smoother = _build_matrix_free_projected_residual_smoother(
            operator_matvec=operator_matvec,
            shape=shape,
            dtype=dtype_use,
            geometry_metadata=geometry_metadata,
            config=config_use,
        )
    elif (
        smoother is None
        and operator_csr is None
        and local_smoother_kind_requested in matrix_free_residual_smoother_tokens
    ):
        smoother = _build_matrix_free_residual_smoother(
            operator_matvec=operator_matvec,
            shape=shape,
            dtype=dtype_use,
            config=config_use,
        )
    if smoother is None and local_smoother_kind_requested in {"device_jacobi", "jacobi"}:
        raise ValueError("device_jacobi local smoother requires a DeviceCSR operator")

    basis = _basis_from_value(
        coarse_basis,
        total_size=int(shape[1]),
        dtype=dtype_use,
        labels=coarse_labels,
        config=config_use,
    )
    residual_enrichment_candidate_count = 0
    if bool(config_use.residual_enrichment):
        if residual_seed is None:
            raise ValueError("residual_seed is required when residual_enrichment=True")
        basis, residual_enrichment_candidate_count = _enrich_basis_with_residual(
            operator_matvec=operator_matvec,
            basis=basis,
            residual_seed=residual_seed,
            shape=shape,
            dtype=dtype_use,
            config=config_use,
        )
    recycle_enrichment_candidate_count = 0
    if bool(config_use.recycle_enrichment) and int(config_use.recycle_enrichment_cycles) > 0:
        if residual_seed is None:
            raise ValueError("residual_seed is required when recycle_enrichment=True")
        basis, recycle_enrichment_candidate_count = _enrich_basis_with_recycle_residuals(
            operator_matvec=operator_matvec,
            basis=basis,
            residual_seed=residual_seed,
            shape=shape,
            dtype=dtype_use,
            config=config_use,
        )
    rank = int(basis.metadata.rank)
    if rank > 0:
        aq = _operator_on_basis(operator_matvec, basis.vectors, shape=shape, dtype=dtype_use)
        coarse_operator = jnp.conjugate(jnp.asarray(basis.vectors, dtype=dtype_use)).T @ aq
    else:
        aq = jnp.zeros((int(shape[0]), 0), dtype=dtype_use)
        coarse_operator = jnp.zeros((0, 0), dtype=dtype_use)

    if smoother is None:
        local_smoother_kind = "none"
        local_smoother_reason = "matrix_free_coarse_only"
    elif isinstance(smoother, RHS1QIDeviceJacobiSmoother):
        local_smoother_kind = "device_jacobi"
        local_smoother_reason = str(smoother.metadata.reason)
    elif isinstance(smoother, RHS1QIMatrixFreeProjectedResidualSmoother):
        local_smoother_kind = "matrix_free_block_minres"
        local_smoother_reason = str(smoother.metadata.reason)
    else:
        local_smoother_kind = "matrix_free_residual"
        local_smoother_reason = str(smoother.metadata.reason)
    if rank > 0 and smoother is None:
        reason = "built_matrix_free_coarse_only"
    elif rank > 0:
        reason = "built_with_coarse"
    else:
        reason = "built_local_only" if smoother is not None else "built_empty"
    metadata = RHS1QIDevicePreconditionerMetadata(
        shape=shape,
        nnz=nnz,
        rank=rank,
        operator_source=operator_source,
        coarse_operator_shape=tuple(int(v) for v in coarse_operator.shape),
        operator_on_basis_shape=tuple(int(v) for v in aq.shape),
        coarse_operator_norm=float(jnp.linalg.norm(coarse_operator)) if rank > 0 else 0.0,
        operator_on_basis_norm=float(jnp.linalg.norm(aq)) if rank > 0 else 0.0,
        regularization_rcond=rcond,
        damping=damping,
        coarse_solver=coarse_solver,
        composition=composition,
        local_smoother_kind=local_smoother_kind,
        local_smoother_reason=local_smoother_reason,
        device_resident=True,
        host_fallback_used=False,
        host_callback_free=True,
        operator_metadata_keys=_metadata_keys(operator_metadata),
        geometry_metadata_keys=_metadata_keys(geometry_metadata),
        accepted_basis_labels=tuple(str(label) for label in basis.metadata.accepted_labels),
        residual_enrichment_enabled=bool(config_use.residual_enrichment),
        residual_enrichment_depth=max(0, int(config_use.residual_enrichment_depth)),
        residual_enrichment_candidate_count=int(residual_enrichment_candidate_count),
        recycle_enrichment_enabled=bool(config_use.recycle_enrichment),
        recycle_enrichment_cycles=max(0, int(config_use.recycle_enrichment_cycles)),
        recycle_enrichment_candidate_count=int(recycle_enrichment_candidate_count),
        reason=reason,
    )
    return RHS1QIDevicePreconditionerState(
        operator=operator_csr,
        operator_matvec=operator_matvec,
        dtype=dtype_use,
        shape=shape,
        local_smoother=smoother,
        basis=basis,
        operator_on_basis=aq,
        coarse_operator=coarse_operator,
        metadata=metadata,
    )


def probe_rhs1_qi_device_preconditioner(
    *,
    rhs: ArrayLike,
    x0: ArrayLike,
    state: RHS1QIDevicePreconditionerState,
    operator: Callable[[ArrayLike], ArrayLike] | None = None,
    min_relative_improvement: float = 0.0,
    acceptance_atol: float = 0.0,
    max_cycles: int = 1,
    residual_minimizing_step: bool = False,
    alpha_clip: float = 10.0,
) -> tuple[ArrayLike, RHS1QIDevicePreconditionerProbe]:
    """Apply bounded preconditioner corrections accepted only by true residual.

    ``max_cycles`` is intentionally small and fail-closed.  Each cycle applies
    the reusable device-QI action to the current true residual, accepts only a
    material residual drop, and stops as soon as a candidate is non-finite or no
    longer improves.  When ``residual_minimizing_step`` is enabled, each
    correction direction is scaled by the scalar that minimizes
    ``||r - alpha A d||_2`` before the true-residual gate is evaluated.  This
    gives the GPU hard-seed lane a real residual-reducing sequence without
    installing the coarse action as an unbounded Krylov preconditioner.
    """

    matvec = state.operator_matvec if operator is None else operator
    rhs_vec = jnp.asarray(rhs, dtype=state.dtype).reshape((-1,))
    x_initial = jnp.asarray(x0, dtype=rhs_vec.dtype).reshape((-1,))
    if rhs_vec.shape != x_initial.shape:
        raise ValueError("rhs and x0 must have the same shape")
    residual_before = rhs_vec - jnp.asarray(matvec(x_initial), dtype=rhs_vec.dtype).reshape((-1,))
    residual_before_norm = float(jnp.linalg.norm(residual_before))
    if residual_before_norm == 0.0:
        probe = RHS1QIDevicePreconditionerProbe(
            accepted=False,
            reason="zero_residual",
            residual_before_norm=0.0,
            residual_after_norm=0.0,
            improvement_ratio=None,
            metadata=state.metadata,
            cycles=0,
            residual_history=(0.0,),
            step_history=(),
        )
        return x_initial, probe

    x_best = x_initial
    residual_current = residual_before
    residual_current_norm = residual_before_norm
    history: list[float] = [float(residual_before_norm)]
    accepted_cycles = 0
    last_finite = True
    last_candidate_norm = residual_before_norm
    max_cycles_use = max(1, int(max_cycles))
    step_history: list[float] = []
    alpha_clip_use = max(0.0, float(alpha_clip))
    for _ in range(max_cycles_use):
        dx = jnp.asarray(state.apply(residual_current), dtype=rhs_vec.dtype).reshape((-1,))
        alpha = 1.0
        if bool(residual_minimizing_step):
            a_dx = jnp.asarray(matvec(dx), dtype=rhs_vec.dtype).reshape((-1,))
            denom = float(jnp.real(jnp.vdot(a_dx, a_dx)))
            if (not np.isfinite(denom)) or denom <= 1.0e-300:
                last_finite = False
                break
            numer = float(jnp.real(jnp.vdot(a_dx, residual_current)))
            alpha = numer / denom
            if alpha_clip_use > 0.0:
                alpha = max(-alpha_clip_use, min(alpha_clip_use, float(alpha)))
            if (not np.isfinite(alpha)) or alpha == 0.0:
                last_finite = False
                break
            x_candidate = x_best + float(alpha) * dx
            residual_after = residual_current - float(alpha) * a_dx
        else:
            x_candidate = x_best + dx
            residual_after = rhs_vec - jnp.asarray(matvec(x_candidate), dtype=rhs_vec.dtype).reshape((-1,))
        residual_after_norm_measured = float(jnp.linalg.norm(residual_after))
        finite = bool(np.isfinite(residual_after_norm_measured))
        last_finite = finite
        if finite:
            last_candidate_norm = residual_after_norm_measured
        required_drop = max(
            float(acceptance_atol),
            residual_current_norm * max(0.0, float(min_relative_improvement)),
        )
        if not (finite and residual_after_norm_measured < residual_current_norm - required_drop):
            break
        x_best = x_candidate
        residual_current = residual_after
        residual_current_norm = residual_after_norm_measured
        history.append(float(residual_current_norm))
        step_history.append(float(alpha))
        accepted_cycles += 1

    accepted = accepted_cycles > 0
    if accepted:
        reason = "residual_reduced"
        residual_after_norm = residual_current_norm
    elif not last_finite:
        reason = "nonfinite_candidate"
        residual_after_norm = residual_before_norm
    else:
        reason = "residual_not_reduced"
        residual_after_norm = last_candidate_norm
    improvement_ratio = residual_after_norm / residual_before_norm if last_finite else None
    probe = RHS1QIDevicePreconditionerProbe(
        accepted=bool(accepted),
        reason=reason,
        residual_before_norm=residual_before_norm,
        residual_after_norm=residual_after_norm,
        improvement_ratio=improvement_ratio,
        metadata=state.metadata,
        cycles=int(accepted_cycles),
        residual_history=tuple(float(value) for value in history),
        step_history=tuple(float(value) for value in step_history),
    )
    return x_best if accepted else x_initial, probe


__all__ = [
    "RHS1QIMatrixFreeResidualSmoother",
    "RHS1QIMatrixFreeResidualSmootherMetadata",
    "RHS1QIMatrixFreeProjectedResidualSmoother",
    "RHS1QIMatrixFreeProjectedResidualSmootherMetadata",
    "RHS1QIDevicePreconditionerConfig",
    "RHS1QIDevicePreconditionerMetadata",
    "RHS1QIDevicePreconditionerProbe",
    "RHS1QIDevicePreconditionerState",
    "probe_rhs1_qi_device_preconditioner",
    "setup_rhs1_qi_device_preconditioner",
]
