"""Two-level RHSMode=1 QI preconditioner building blocks.

The hard QI GPU lane needs a preconditioner that improves the true residual on
device before a long Krylov solve is launched.  This module provides a small,
JAX-compatible primitive for that strategy without changing production defaults:

``local smoother`` + ``coarse Schur/Galerkin correction``.

The action is multiplicative:

``M^{-1} r = S^{-1} r + Q A_c^{-1} Q^T (r - A S^{-1} r)``,

where ``S^{-1}`` is a cheap local block smoother and ``Q`` is a rank-gated QI
coarse basis.  The helper records enough metadata for the production driver to
fail closed when a residual probe does not improve.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import os
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .rhs1_qi_coarse import (
    RHS1QICoarseBasis,
    build_rhs1_xblock_global_coarse_basis,
    build_rhs1_xblock_global_coupling_load_basis,
)

ArrayLike = Any
LinearOperator = Callable[[ArrayLike], ArrayLike]


def _normalize_coarse_solver(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "projected": "projected",
        "galerkin": "projected",
        "qtaq": "projected",
        "action_lstsq": "action_lstsq",
        "action_ls": "action_lstsq",
        "least_squares": "action_lstsq",
        "lstsq": "action_lstsq",
    }
    if normalized not in aliases:
        raise ValueError(
            "coarse_solver must be one of 'projected' or 'action_lstsq' "
            f"(got {value!r})"
        )
    return aliases[normalized]


@dataclass(frozen=True)
class RHS1QITwoLevelMetadata:
    """Diagnostics for a two-level QI preconditioner candidate."""

    rank: int
    coarse_operator_shape: tuple[int, int]
    coarse_operator_norm: float
    operator_on_basis_shape: tuple[int, int]
    operator_on_basis_norm: float
    regularization_rcond: float
    damping: float
    coarse_solver: str = "projected"
    composition: str = "multiplicative"

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": int(self.rank),
            "coarse_operator_shape": tuple(int(v) for v in self.coarse_operator_shape),
            "coarse_operator_norm": float(self.coarse_operator_norm),
            "operator_on_basis_shape": tuple(
                int(v) for v in self.operator_on_basis_shape
            ),
            "operator_on_basis_norm": float(self.operator_on_basis_norm),
            "regularization_rcond": float(self.regularization_rcond),
            "damping": float(self.damping),
            "coarse_solver": self.coarse_solver,
            "composition": self.composition,
        }


@dataclass(frozen=True)
class RHS1QITwoLevelPreconditioner:
    """Device-compatible local-smoother plus coarse-correction action."""

    operator: LinearOperator
    local_smoother: LinearOperator
    basis: RHS1QICoarseBasis
    coarse_operator: ArrayLike
    operator_on_basis: ArrayLike
    metadata: RHS1QITwoLevelMetadata

    def solve_coarse(self, residual: ArrayLike) -> ArrayLike:
        """Solve the small projected problem in regularized least-squares form."""

        residual_vec = jnp.asarray(residual).reshape((-1,))
        rank = int(self.metadata.rank)
        if rank <= 0:
            return jnp.zeros((0,), dtype=residual_vec.dtype)
        if self.metadata.coarse_solver == "action_lstsq":
            return _regularized_coarse_solve(
                self.operator_on_basis,
                residual_vec,
                rcond=float(self.metadata.regularization_rcond),
            )
        projected = jnp.conjugate(self.basis.vectors).T @ residual_vec
        return _regularized_coarse_solve(
            self.coarse_operator,
            projected,
            rcond=float(self.metadata.regularization_rcond),
        )

    def apply(self, residual: ArrayLike) -> ArrayLike:
        """Apply one multiplicative two-level preconditioner action."""

        residual_vec = jnp.asarray(residual).reshape((-1,))
        local = jnp.asarray(self.local_smoother(residual_vec)).reshape((-1,))
        if int(self.metadata.rank) <= 0:
            return float(self.metadata.damping) * local
        remaining = residual_vec - jnp.asarray(self.operator(local)).reshape((-1,))
        coarse_coefficients = self.solve_coarse(remaining)
        coarse = self.basis.vectors @ coarse_coefficients
        return float(self.metadata.damping) * (local + coarse)

    def as_preconditioner(self) -> Callable[[ArrayLike], ArrayLike]:
        """Return a callable suitable for Krylov preconditioner hooks."""

        return self.apply


@dataclass(frozen=True)
class RHS1QITwoLevelProbe:
    """True-residual probe for a two-level preconditioner candidate."""

    accepted: bool
    reason: str
    residual_before_norm: float
    residual_after_norm: float
    improvement_ratio: float | None
    metadata: RHS1QITwoLevelMetadata

    def to_dict(self) -> dict[str, object]:
        return {
            "accepted": bool(self.accepted),
            "reason": self.reason,
            "residual_before_norm": float(self.residual_before_norm),
            "residual_after_norm": float(self.residual_after_norm),
            "improvement_ratio": None
            if self.improvement_ratio is None
            else float(self.improvement_ratio),
            "metadata": self.metadata.to_dict(),
        }


def _regularized_coarse_solve(
    a: ArrayLike, rhs: ArrayLike, *, rcond: float
) -> ArrayLike:
    a_arr = jnp.asarray(a)
    rhs_arr = jnp.asarray(rhs).reshape((-1,))
    if a_arr.size == 0:
        return jnp.zeros((0,), dtype=rhs_arr.dtype)
    gram = jnp.conjugate(a_arr).T @ a_arr
    normal_rhs = jnp.conjugate(a_arr).T @ rhs_arr
    scale = jnp.maximum(jnp.linalg.norm(gram), jnp.asarray(1.0, dtype=gram.dtype))
    ridge = jnp.asarray(max(0.0, float(rcond)), dtype=gram.dtype) * scale
    eye = jnp.eye(int(gram.shape[0]), dtype=gram.dtype)
    return jnp.linalg.solve(gram + ridge * eye, normal_rhs)


def _apply_operator_to_basis(
    operator: LinearOperator, basis_vectors: ArrayLike
) -> ArrayLike:
    q = jnp.asarray(basis_vectors)
    columns = [
        jnp.asarray(operator(q[:, i])).reshape((-1,)) for i in range(int(q.shape[1]))
    ]
    if not columns:
        return jnp.zeros_like(q)
    return jnp.stack(columns, axis=1)


def build_rhs1_qi_two_level_preconditioner(
    *,
    operator: LinearOperator,
    local_smoother: LinearOperator,
    basis: RHS1QICoarseBasis,
    regularization_rcond: float = 1.0e-12,
    damping: float = 1.0,
    coarse_solver: str = "projected",
) -> RHS1QITwoLevelPreconditioner:
    """Build a reusable two-level QI preconditioner from a coarse basis."""

    q = jnp.asarray(basis.vectors)
    rank = int(q.shape[1]) if q.ndim == 2 else 0
    coarse_solver_norm = _normalize_coarse_solver(coarse_solver)
    if rank <= 0:
        coarse_operator = jnp.zeros((0, 0), dtype=q.dtype)
        operator_on_basis = jnp.zeros_like(q)
    else:
        operator_on_basis = _apply_operator_to_basis(operator, q)
        coarse_operator = jnp.conjugate(q).T @ operator_on_basis
    metadata = RHS1QITwoLevelMetadata(
        rank=rank,
        coarse_operator_shape=tuple(int(v) for v in coarse_operator.shape),
        coarse_operator_norm=float(jnp.linalg.norm(coarse_operator))
        if rank > 0
        else 0.0,
        operator_on_basis_shape=tuple(int(v) for v in operator_on_basis.shape),
        operator_on_basis_norm=float(jnp.linalg.norm(operator_on_basis))
        if rank > 0
        else 0.0,
        regularization_rcond=float(regularization_rcond),
        damping=float(damping),
        coarse_solver=coarse_solver_norm,
    )
    return RHS1QITwoLevelPreconditioner(
        operator=operator,
        local_smoother=local_smoother,
        basis=basis,
        coarse_operator=coarse_operator,
        operator_on_basis=operator_on_basis,
        metadata=metadata,
    )


def probe_rhs1_qi_two_level_correction(
    *,
    operator: LinearOperator,
    rhs: ArrayLike,
    x0: ArrayLike,
    preconditioner: RHS1QITwoLevelPreconditioner,
    min_relative_improvement: float = 0.0,
) -> tuple[ArrayLike, RHS1QITwoLevelProbe]:
    """Apply one correction and accept it only if the true residual improves."""

    rhs_vec = jnp.asarray(rhs).reshape((-1,))
    x_initial = jnp.asarray(x0).reshape((-1,))
    residual_before = rhs_vec - jnp.asarray(operator(x_initial)).reshape((-1,))
    residual_before_norm = float(jnp.linalg.norm(residual_before))
    dx = jnp.asarray(preconditioner.apply(residual_before)).reshape((-1,))
    x_candidate = x_initial + dx
    residual_after = rhs_vec - jnp.asarray(operator(x_candidate)).reshape((-1,))
    residual_after_norm = float(jnp.linalg.norm(residual_after))
    improvement_ratio = (
        residual_after_norm / residual_before_norm
        if residual_before_norm > 0.0
        else None
    )
    required = residual_before_norm * max(0.0, 1.0 - float(min_relative_improvement))
    accepted = bool(
        jnp.isfinite(jnp.asarray(residual_after_norm))
        and residual_after_norm < required
    )
    probe = RHS1QITwoLevelProbe(
        accepted=accepted,
        reason="residual_reduced" if accepted else "residual_not_reduced",
        residual_before_norm=residual_before_norm,
        residual_after_norm=residual_after_norm,
        improvement_ratio=improvement_ratio,
        metadata=preconditioner.metadata,
    )
    return x_candidate if accepted else x_initial, probe


def build_rhs1_xblock_two_level_preconditioner(
    *,
    op: Any,
    rhs: ArrayLike,
    matvec: LinearOperator,
    base_preconditioner: LinearOperator,
    direction_projector: LinearOperator | None = None,
    expected_size: int | None = None,
    mode: str,
    fsavg_lmax: int,
    max_extra_units: int,
    max_directions: int,
    rcond: float,
    include_rhs: bool,
    emit: Callable[[int, str], None] | None = None,
) -> tuple[LinearOperator, dict[str, object], dict[str, int]]:
    """Wrap an x-block preconditioner with a fixed global coarse inverse."""

    from scipy.linalg import qr, solve_triangular  # noqa: PLC0415

    mode_use = str(mode).strip().lower().replace("-", "_")
    if mode_use not in {"additive", "multiplicative"}:
        mode_use = "additive"
    max_dirs_use = max(1, int(max_directions))
    expected_size_use = (
        int(op.total_size) if expected_size is None else int(expected_size)
    )
    raw_directions = build_rhs1_xblock_global_coarse_basis(
        op=op,
        rhs=rhs,
        preconditioner=base_preconditioner,
        include_rhs=bool(include_rhs),
        fsavg_lmax=int(fsavg_lmax),
        max_extra_units=int(max_extra_units),
        max_directions=max_dirs_use,
    )
    basis_cols: list[np.ndarray] = []
    abasis_cols: list[np.ndarray] = []
    names: list[str] = []
    for name, direction in raw_directions[:max_dirs_use]:
        if direction_projector is not None:
            direction = direction_projector(direction)
        direction_np = np.asarray(jax.device_get(direction), dtype=np.float64).reshape(
            (-1,)
        )
        if direction_np.shape != (expected_size_use,) or not np.all(
            np.isfinite(direction_np)
        ):
            continue
        norm = float(np.linalg.norm(direction_np))
        if (not np.isfinite(norm)) or norm <= 0.0:
            continue
        direction_np = direction_np / norm
        a_direction = np.asarray(
            jax.device_get(matvec(jnp.asarray(direction_np, dtype=jnp.float64))),
            dtype=np.float64,
        ).reshape((-1,))
        if a_direction.shape != direction_np.shape or not np.all(
            np.isfinite(a_direction)
        ):
            continue
        a_norm = float(np.linalg.norm(a_direction))
        if (not np.isfinite(a_norm)) or a_norm <= 0.0:
            continue
        names.append(str(name))
        basis_cols.append(direction_np)
        abasis_cols.append(a_direction)

    if not basis_cols:
        raise RuntimeError(
            "two-level x-block preconditioner found no valid coarse directions"
        )

    basis = np.column_stack(basis_cols)
    abasis = np.column_stack(abasis_cols)
    q, r, piv = qr(abasis, mode="economic", pivoting=True)
    diag = np.abs(np.diag(r))
    if diag.size == 0:
        raise RuntimeError("two-level x-block preconditioner coarse QR is empty")
    threshold = max(float(rcond), 0.0) * max(float(diag[0]), 1.0)
    rank = int(np.count_nonzero(diag > threshold))
    if rank <= 0:
        raise RuntimeError(
            "two-level x-block preconditioner coarse QR is rank deficient"
        )
    piv_use = np.asarray(piv[:rank], dtype=np.int32)
    q_use = np.asarray(q[:, :rank], dtype=np.float64)
    r_use = np.asarray(r[:rank, :rank], dtype=np.float64)
    basis_use = np.asarray(basis[:, piv_use], dtype=np.float64)
    names_use = tuple(names[int(i)] for i in piv_use)
    stats = {"applies": 0, "coarse_applies": 0}

    def coarse_correction(vec_np: np.ndarray) -> np.ndarray:
        rhs_small = q_use.T @ np.asarray(vec_np, dtype=np.float64).reshape((-1,))
        coeff = solve_triangular(r_use, rhs_small, lower=False, check_finite=False)
        return basis_use @ np.asarray(coeff, dtype=np.float64).reshape((-1,))

    def apply(value: ArrayLike) -> ArrayLike:
        stats["applies"] += 1
        value_np = np.asarray(jax.device_get(value), dtype=np.float64).reshape((-1,))
        base = np.asarray(
            jax.device_get(
                base_preconditioner(jnp.asarray(value_np, dtype=jnp.float64))
            ),
            dtype=np.float64,
        ).reshape((-1,))
        if mode_use == "multiplicative":
            abase = np.asarray(
                jax.device_get(matvec(jnp.asarray(base, dtype=jnp.float64))),
                dtype=np.float64,
            ).reshape((-1,))
            coarse_input = value_np - abase
        else:
            coarse_input = value_np
        correction = coarse_correction(coarse_input)
        stats["coarse_applies"] += 1
        return jnp.asarray(base + correction, dtype=jnp.float64)

    metadata: dict[str, object] = {
        "mode": mode_use,
        "basis_size": int(len(basis_cols)),
        "rank": int(rank),
        "requested_max_directions": int(max_dirs_use),
        "fsavg_lmax": int(fsavg_lmax),
        "include_rhs": bool(include_rhs),
        "rcond": float(rcond),
        "basis_names": names_use,
        "active_projected": bool(direction_projector is not None),
        "expected_size": int(expected_size_use),
    }
    if emit is not None:
        emit(
            0,
            "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres two-level coarse "
            f"built mode={mode_use} basis={len(basis_cols)} rank={rank}",
        )
    return apply, metadata, stats


def build_rhs1_xblock_smoothed_global_coupling_preconditioner(
    *,
    op: Any,
    rhs: ArrayLike,
    matvec: LinearOperator,
    base_preconditioner: LinearOperator,
    direction_projector: LinearOperator | None = None,
    expected_size: int | None = None,
    mode: str,
    fsavg_lmax: int,
    angular_lmax: int,
    max_extra_units: int,
    max_directions: int,
    rcond: float,
    include_rhs: bool,
    max_setup_s: float = 0.0,
    emit: Callable[[int, str], None] | None = None,
) -> tuple[LinearOperator, dict[str, object], dict[str, int]]:
    """Build a host-QR smoothed low-rank global-coupling preconditioner."""

    from scipy.linalg import qr, solve_triangular  # noqa: PLC0415

    mode_use = str(mode).strip().lower().replace("-", "_")
    if mode_use not in {"additive", "multiplicative"}:
        mode_use = "additive"
    smoother = (
        os.environ.get("SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_SMOOTHER", "base")
        .strip()
        .lower()
        .replace("-", "_")
    )
    if smoother in {"none", "raw", "identity", "load", "loads"}:
        smoother = "identity"
    elif smoother in {"preconditioner", "preconditioned"}:
        smoother = "base"
    elif smoother != "base":
        smoother = "base"
    max_dirs_use = max(1, int(max_directions))
    max_setup_s_use = max(0.0, float(max_setup_s))
    setup_deadline = (
        time.perf_counter() + max_setup_s_use if max_setup_s_use > 0.0 else None
    )
    setup_budget_reached = False
    expected_size_use = (
        int(op.total_size) if expected_size is None else int(expected_size)
    )
    raw_loads = build_rhs1_xblock_global_coupling_load_basis(
        op=op,
        rhs=rhs,
        include_rhs=bool(include_rhs),
        fsavg_lmax=int(fsavg_lmax),
        angular_lmax=int(angular_lmax),
        max_extra_units=int(max_extra_units),
        max_directions=max_dirs_use,
    )
    z_cols: list[np.ndarray] = []
    az_cols: list[np.ndarray] = []
    names: list[str] = []
    for name, load in raw_loads[:max_dirs_use]:
        if direction_projector is not None:
            load = direction_projector(load)
        load_np = np.asarray(jax.device_get(load), dtype=np.float64).reshape((-1,))
        if load_np.shape != (expected_size_use,) or not np.all(np.isfinite(load_np)):
            continue
        load_norm = float(np.linalg.norm(load_np))
        if (not np.isfinite(load_norm)) or load_norm <= 0.0:
            continue
        if smoother == "identity":
            smoothed = np.asarray(load_np / load_norm, dtype=np.float64)
        else:
            smoothed = np.asarray(
                jax.device_get(
                    base_preconditioner(
                        jnp.asarray(load_np / load_norm, dtype=jnp.float64)
                    )
                ),
                dtype=np.float64,
            ).reshape((-1,))
        if smoothed.shape != load_np.shape or not np.all(np.isfinite(smoothed)):
            continue
        z_norm = float(np.linalg.norm(smoothed))
        if (not np.isfinite(z_norm)) or z_norm <= 0.0:
            continue
        smoothed = smoothed / z_norm
        a_smoothed = np.asarray(
            jax.device_get(matvec(jnp.asarray(smoothed, dtype=jnp.float64))),
            dtype=np.float64,
        ).reshape((-1,))
        if a_smoothed.shape != smoothed.shape or not np.all(np.isfinite(a_smoothed)):
            continue
        a_norm = float(np.linalg.norm(a_smoothed))
        if (not np.isfinite(a_norm)) or a_norm <= 0.0:
            continue
        names.append(str(name))
        z_cols.append(smoothed)
        az_cols.append(a_smoothed)
        if setup_deadline is not None and time.perf_counter() >= setup_deadline:
            setup_budget_reached = True
            if emit is not None:
                emit(
                    1,
                    "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres global-coupling "
                    f"setup budget reached after basis={len(z_cols)}/{len(raw_loads)} "
                    f"max_setup_s={max_setup_s_use:.3f}; using partial basis",
                )
            break

    if not z_cols:
        raise RuntimeError(
            "global-coupling x-block preconditioner found no valid smoothed directions"
        )

    z_basis = np.column_stack(z_cols)
    az_basis = np.column_stack(az_cols)
    q, r, piv = qr(az_basis, mode="economic", pivoting=True)
    diag = np.abs(np.diag(r))
    if diag.size == 0:
        raise RuntimeError("global-coupling x-block preconditioner coarse QR is empty")
    threshold = max(float(rcond), 0.0) * max(float(diag[0]), 1.0)
    rank = int(np.count_nonzero(diag > threshold))
    if rank <= 0:
        raise RuntimeError(
            "global-coupling x-block preconditioner coarse QR is rank deficient"
        )
    piv_use = np.asarray(piv[:rank], dtype=np.int32)
    q_use = np.asarray(q[:, :rank], dtype=np.float64)
    r_use = np.asarray(r[:rank, :rank], dtype=np.float64)
    z_use = np.asarray(z_basis[:, piv_use], dtype=np.float64)
    names_use = tuple(names[int(i)] for i in piv_use)
    stats = {"applies": 0, "coarse_applies": 0}

    def coarse_correction(vec_np: np.ndarray) -> np.ndarray:
        rhs_small = q_use.T @ np.asarray(vec_np, dtype=np.float64).reshape((-1,))
        coeff = solve_triangular(r_use, rhs_small, lower=False, check_finite=False)
        return z_use @ np.asarray(coeff, dtype=np.float64).reshape((-1,))

    def apply(value: ArrayLike) -> ArrayLike:
        stats["applies"] += 1
        value_np = np.asarray(jax.device_get(value), dtype=np.float64).reshape((-1,))
        base = np.asarray(
            jax.device_get(
                base_preconditioner(jnp.asarray(value_np, dtype=jnp.float64))
            ),
            dtype=np.float64,
        ).reshape((-1,))
        if mode_use == "multiplicative":
            abase = np.asarray(
                jax.device_get(matvec(jnp.asarray(base, dtype=jnp.float64))),
                dtype=np.float64,
            ).reshape((-1,))
            coarse_input = value_np - abase
        else:
            coarse_input = value_np
        correction = coarse_correction(coarse_input)
        stats["coarse_applies"] += 1
        return jnp.asarray(base + correction, dtype=jnp.float64)

    metadata: dict[str, object] = {
        "mode": mode_use,
        "load_basis_size": int(len(raw_loads)),
        "basis_size": int(len(z_cols)),
        "rank": int(rank),
        "requested_max_directions": int(max_dirs_use),
        "fsavg_lmax": int(fsavg_lmax),
        "angular_lmax": int(angular_lmax),
        "include_rhs": bool(include_rhs),
        "rcond": float(rcond),
        "smoother": smoother,
        "setup_budget_s": float(max_setup_s_use),
        "setup_budget_reached": bool(setup_budget_reached),
        "basis_names": names_use,
    }
    if emit is not None:
        emit(
            0,
            "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres global-coupling "
            f"built mode={mode_use} smoother={smoother} loads={len(raw_loads)} basis={len(z_cols)} rank={rank}",
        )
    return apply, metadata, stats


def build_rhs1_xblock_device_global_coupling_preconditioner(
    *,
    op: Any,
    rhs: ArrayLike,
    matvec: LinearOperator,
    base_preconditioner: LinearOperator,
    direction_projector: LinearOperator | None = None,
    expected_size: int | None = None,
    mode: str,
    fsavg_lmax: int,
    angular_lmax: int,
    max_extra_units: int,
    max_directions: int,
    rcond: float,
    include_rhs: bool,
    max_setup_s: float = 0.0,
    emit: Callable[[int, str], None] | None = None,
) -> tuple[LinearOperator, dict[str, object], dict[str, int]]:
    """Build a device-resident global-coupling correction for Krylov paths."""

    from scipy.linalg import qr  # noqa: PLC0415

    mode_use = str(mode).strip().lower().replace("-", "_")
    if mode_use not in {"additive", "multiplicative"}:
        mode_use = "additive"
    coarse_solver = (
        os.environ.get(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_DEVICE_SOLVER", "qr"
        )
        .strip()
        .lower()
        .replace("-", "_")
    )
    if coarse_solver in {"normal", "normal_equation", "normal_equations", "ridge"}:
        coarse_solver = "normal_equations"
    elif coarse_solver not in {"qr", "pivoted_qr"}:
        coarse_solver = "qr"
    smoother = (
        os.environ.get(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_SMOOTHER", "identity"
        )
        .strip()
        .lower()
        .replace("-", "_")
    )
    if smoother in {"none", "raw", "identity", "load", "loads"}:
        smoother = "identity"
    elif smoother in {"base", "preconditioner", "preconditioned"}:
        smoother = "base"
    else:
        smoother = "identity"
    expected_size_use = (
        int(op.total_size) if expected_size is None else int(expected_size)
    )
    max_dirs_use = max(1, int(max_directions))
    max_setup_s_use = max(0.0, float(max_setup_s))
    setup_deadline = (
        time.perf_counter() + max_setup_s_use if max_setup_s_use > 0.0 else None
    )
    setup_budget_reached = False
    raw_loads = build_rhs1_xblock_global_coupling_load_basis(
        op=op,
        rhs=rhs,
        include_rhs=bool(include_rhs),
        fsavg_lmax=int(fsavg_lmax),
        angular_lmax=int(angular_lmax),
        max_extra_units=int(max_extra_units),
        max_directions=max_dirs_use,
    )
    z_cols: list[ArrayLike] = []
    az_cols: list[ArrayLike] = []
    names: list[str] = []
    for name, load in raw_loads[:max_dirs_use]:
        if direction_projector is not None:
            load = direction_projector(load)
        load_j = jnp.asarray(load, dtype=jnp.float64).reshape((-1,))
        if load_j.shape != (expected_size_use,):
            continue
        load_np = np.asarray(jax.device_get(load_j), dtype=np.float64)
        if not np.all(np.isfinite(load_np)):
            continue
        load_norm = float(np.linalg.norm(load_np))
        if (not np.isfinite(load_norm)) or load_norm <= 0.0:
            continue
        load_unit = jnp.asarray(load_np / load_norm, dtype=jnp.float64)
        if smoother == "identity":
            z = load_unit
        else:
            z = jnp.asarray(base_preconditioner(load_unit), dtype=jnp.float64).reshape(
                (-1,)
            )
        z_np = np.asarray(jax.device_get(z), dtype=np.float64)
        if z_np.shape != load_np.shape or not np.all(np.isfinite(z_np)):
            continue
        z_norm = float(np.linalg.norm(z_np))
        if (not np.isfinite(z_norm)) or z_norm <= 0.0:
            continue
        z = jnp.asarray(z_np / z_norm, dtype=jnp.float64)
        az = jnp.asarray(matvec(z), dtype=jnp.float64).reshape((-1,))
        az_np = np.asarray(jax.device_get(az), dtype=np.float64)
        if az_np.shape != z_np.shape or not np.all(np.isfinite(az_np)):
            continue
        az_norm = float(np.linalg.norm(az_np))
        if (not np.isfinite(az_norm)) or az_norm <= 0.0:
            continue
        names.append(str(name))
        z_cols.append(z)
        az_cols.append(az)
        if setup_deadline is not None and time.perf_counter() >= setup_deadline:
            setup_budget_reached = True
            if emit is not None:
                emit(
                    1,
                    "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres device global-coupling "
                    f"setup budget reached after basis={len(z_cols)}/{len(raw_loads)} "
                    f"max_setup_s={max_setup_s_use:.3f}; using partial basis",
                )
            break

    if not z_cols:
        raise RuntimeError(
            "device global-coupling x-block preconditioner found no valid smoothed directions"
        )

    z_basis_np = np.column_stack(
        [np.asarray(jax.device_get(col), dtype=np.float64) for col in z_cols]
    )
    az_basis_np = np.column_stack(
        [np.asarray(jax.device_get(col), dtype=np.float64) for col in az_cols]
    )
    if not np.all(np.isfinite(z_basis_np)) or not np.all(np.isfinite(az_basis_np)):
        raise RuntimeError(
            "device global-coupling x-block preconditioner found non-finite coarse columns"
        )

    ridge_value = 0.0
    names_use = tuple(names)
    singular_values: np.ndarray
    r_diag: np.ndarray
    if coarse_solver == "normal_equations":
        z_basis = jnp.asarray(z_basis_np, dtype=jnp.float64)
        az_basis = jnp.asarray(az_basis_np, dtype=jnp.float64)
        gram = az_basis.T @ az_basis
        diag_scale = jnp.maximum(
            jnp.max(jnp.abs(jnp.diag(gram))), jnp.asarray(1.0, dtype=jnp.float64)
        )
        ridge = jnp.asarray(max(float(rcond), 1.0e-14), dtype=jnp.float64) * diag_scale
        coarse_inv = jnp.linalg.inv(
            gram + ridge * jnp.eye(int(gram.shape[0]), dtype=jnp.float64)
        )
        gram_eigs = jnp.maximum(
            jnp.linalg.eigvalsh(gram), jnp.asarray(0.0, dtype=jnp.float64)
        )
        singular_values = np.asarray(
            jax.device_get(jnp.sqrt(gram_eigs[::-1])), dtype=np.float64
        )
        if singular_values.size == 0 or not np.all(np.isfinite(singular_values)):
            raise RuntimeError(
                "device global-coupling x-block preconditioner coarse spectrum is invalid"
            )
        rank_threshold = max(float(rcond), 0.0) * max(
            float(np.max(singular_values, initial=0.0)), 1.0
        )
        rank = int(np.count_nonzero(singular_values > rank_threshold))
        if rank <= 0:
            raise RuntimeError(
                "device global-coupling x-block preconditioner coarse system is rank deficient"
            )
        ridge_value = float(jax.device_get(ridge))
        r_diag = singular_values

        def coarse_coeff(coarse_input: ArrayLike) -> ArrayLike:
            return coarse_inv @ (az_basis.T @ coarse_input)

    else:
        q, r, piv = qr(az_basis_np, mode="economic", pivoting=True)
        diag = np.abs(np.diag(r))
        if diag.size == 0 or not np.all(np.isfinite(diag)):
            raise RuntimeError(
                "device global-coupling x-block preconditioner coarse QR is empty"
            )
        rank_threshold = max(float(rcond), 0.0) * max(float(diag[0]), 1.0)
        rank = int(np.count_nonzero(diag > rank_threshold))
        if rank <= 0:
            raise RuntimeError(
                "device global-coupling x-block preconditioner coarse QR is rank deficient"
            )
        piv_use = np.asarray(piv[:rank], dtype=np.int32)
        q_use = jnp.asarray(q[:, :rank], dtype=jnp.float64)
        r_use = jnp.asarray(r[:rank, :rank], dtype=jnp.float64)
        z_basis = jnp.asarray(z_basis_np[:, piv_use], dtype=jnp.float64)
        r_diag = np.asarray(diag[:rank], dtype=np.float64)
        singular_values = r_diag
        names_use = tuple(names[int(i)] for i in piv_use)

        def coarse_coeff(coarse_input: ArrayLike) -> ArrayLike:
            return jnp.linalg.solve(r_use, q_use.T @ coarse_input)

    stats = {"applies": 0, "coarse_applies": 0}

    def apply(value: ArrayLike) -> ArrayLike:
        stats["applies"] += 1
        value_j = jnp.asarray(value, dtype=jnp.float64)
        base = jnp.asarray(base_preconditioner(value_j), dtype=jnp.float64)
        if mode_use == "multiplicative":
            coarse_input = value_j - jnp.asarray(matvec(base), dtype=jnp.float64)
        else:
            coarse_input = value_j
        coeff = coarse_coeff(coarse_input)
        stats["coarse_applies"] += 1
        return base + z_basis @ coeff

    metadata: dict[str, object] = {
        "mode": mode_use,
        "load_basis_size": int(len(raw_loads)),
        "basis_size": int(z_basis.shape[1]),
        "rank": int(rank),
        "requested_max_directions": int(max_dirs_use),
        "fsavg_lmax": int(fsavg_lmax),
        "angular_lmax": int(angular_lmax),
        "include_rhs": bool(include_rhs),
        "rcond": float(rcond),
        "coarse_solver": coarse_solver,
        "smoother": smoother,
        "ridge": float(ridge_value),
        "setup_budget_s": float(max_setup_s_use),
        "setup_budget_reached": bool(setup_budget_reached),
        "basis_names": names_use,
        "device_resident": True,
        "singular_values": tuple(
            float(v) for v in singular_values[: min(int(singular_values.size), 16)]
        ),
        "r_diag": tuple(float(v) for v in r_diag[: min(int(r_diag.size), 16)]),
    }
    if emit is not None:
        emit(
            0,
            "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres device global-coupling "
            f"built mode={mode_use} smoother={smoother} coarse_solver={coarse_solver} "
            f"loads={len(raw_loads)} basis={int(z_basis.shape[1])} rank={rank}",
        )
    return apply, metadata, stats


__all__ = [
    "RHS1QITwoLevelMetadata",
    "RHS1QITwoLevelPreconditioner",
    "RHS1QITwoLevelProbe",
    "build_rhs1_xblock_device_global_coupling_preconditioner",
    "build_rhs1_xblock_smoothed_global_coupling_preconditioner",
    "build_rhs1_xblock_two_level_preconditioner",
    "build_rhs1_qi_two_level_preconditioner",
    "probe_rhs1_qi_two_level_correction",
]
