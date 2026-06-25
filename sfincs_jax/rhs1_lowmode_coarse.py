"""Low-mode, moment, and tail coarse spaces for RHSMode=1 residual corrections.

The RHSMode=1 driver uses these helpers to build compact Galerkin residual
equations on top of structured f-block preconditioners.  Keeping this code out
of ``v3_driver.py`` makes the coarse-space algebra directly testable without
constructing a full SFINCS solve.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from sfincs_jax.operators.profile_response.layout import (
    RHS1MatrixFreeGalerkinResidualCorrection,
    RHS1MatrixFreeLeastSquaresResidualCorrection,
)


def _rhs1_lowmode_angular_features(
    *,
    n_theta: int,
    n_zeta: int,
    theta_modes: int,
    zeta_modes: int,
) -> np.ndarray:
    """Return normalized low angular feature fields with shape ``(F,T,Z)``."""

    theta = 2.0 * np.pi * np.arange(n_theta, dtype=np.float64) / max(n_theta, 1)
    zeta = 2.0 * np.pi * np.arange(n_zeta, dtype=np.float64) / max(n_zeta, 1)
    theta_grid = theta[:, None]
    zeta_grid = zeta[None, :]

    feature_fields: list[np.ndarray] = [np.ones((n_theta, n_zeta), dtype=np.float64)]
    for mode in range(1, int(theta_modes) + 1):
        feature_fields.append(np.cos(float(mode) * theta_grid) * np.ones((1, n_zeta), dtype=np.float64))
        feature_fields.append(np.sin(float(mode) * theta_grid) * np.ones((1, n_zeta), dtype=np.float64))
    for mode in range(1, int(zeta_modes) + 1):
        feature_fields.append(np.ones((n_theta, 1), dtype=np.float64) * np.cos(float(mode) * zeta_grid))
        feature_fields.append(np.ones((n_theta, 1), dtype=np.float64) * np.sin(float(mode) * zeta_grid))

    usable_features: list[np.ndarray] = []
    for field in feature_fields:
        norm = float(np.linalg.norm(field.reshape((-1,))))
        if norm > 0.0:
            usable_features.append(field / norm)
    if not usable_features:
        raise ValueError("at least one nonzero low angular feature is required")
    return np.stack(usable_features, axis=0)


def _rhs1_cap_lowmode_features(
    *,
    features: np.ndarray,
    n_species: int,
    n_x: int,
    n_xi: int,
    max_coarse_size: int,
) -> tuple[np.ndarray, dict[str, int | bool]]:
    """Trim optional low-mode features to keep the coarse solve bounded."""

    stride = int(n_species) * int(n_x) * int(n_xi)
    if stride <= 0:
        raise ValueError("low-mode coarse stride must be positive")
    requested_features = int(features.shape[0])
    max_features = int(max_coarse_size) // stride
    if max_features < 1:
        raise MemoryError(
            "structured f-block FP low-mode Schur coarse space too large: "
            f"minimum {stride} > {int(max_coarse_size)}; raise "
            "SFINCS_JAX_RHSMODE1_STRUCTURED_FBLOCK_FP_LOWMODE_MAX_COARSE to override"
        )
    retained_features = max(1, min(requested_features, max_features))
    capped = np.asarray(features[:retained_features], dtype=np.float64)
    return capped, {
        "requested_features": requested_features,
        "retained_features": retained_features,
        "truncated_features": bool(retained_features < requested_features),
        "coarse_stride": stride,
        "requested_coarse_size": int(stride * requested_features),
        "retained_coarse_size": int(stride * retained_features),
    }


def _build_rhs1_lowmode_angular_matrix_free_correction(
    *,
    op: Any,
    operator: Any,
    theta_modes: int,
    zeta_modes: int,
    max_coarse_size: int,
    max_basis_batch_nbytes: int,
    basis_batch_size: int,
    regularization: float,
    damping: float,
) -> tuple[RHS1MatrixFreeGalerkinResidualCorrection, dict[str, int | bool]]:
    """Build the low-mode Galerkin correction without storing ``N_f x N_c``."""

    n_species = int(op.n_species)
    n_x = int(op.n_x)
    n_xi = int(op.n_xi)
    n_theta = int(op.n_theta)
    n_zeta = int(op.n_zeta)
    features_requested = _rhs1_lowmode_angular_features(
        n_theta=n_theta,
        n_zeta=n_zeta,
        theta_modes=int(theta_modes),
        zeta_modes=int(zeta_modes),
    )
    capped_features, feature_metadata = _rhs1_cap_lowmode_features(
        features=features_requested,
        n_species=n_species,
        n_x=n_x,
        n_xi=n_xi,
        max_coarse_size=int(max_coarse_size),
    )
    features = jnp.asarray(capped_features, dtype=jnp.float64)
    n_features = int(features.shape[0])
    n_coarse = n_species * n_x * n_xi * n_features

    def _restrict(vector: jnp.ndarray) -> jnp.ndarray:
        arr = jnp.asarray(vector, dtype=jnp.float64)
        if arr.ndim == 1:
            f = arr.reshape((n_species, n_x, n_xi, n_theta, n_zeta))
            return jnp.einsum("sxltz,ftz->sxlf", f, features).reshape((n_coarse,))
        if arr.ndim == 2:
            f = arr.reshape((n_species, n_x, n_xi, n_theta, n_zeta, int(arr.shape[1])))
            return jnp.einsum("sxltzb,ftz->sxlfb", f, features).reshape((n_coarse, int(arr.shape[1])))
        raise ValueError("low-mode restriction expects a vector or column matrix")

    def _prolong(coefficients: jnp.ndarray) -> jnp.ndarray:
        coeff = jnp.asarray(coefficients, dtype=jnp.float64)
        if coeff.ndim == 1:
            c = coeff.reshape((n_species, n_x, n_xi, n_features))
            return jnp.einsum("sxlf,ftz->sxltz", c, features).reshape((int(op.f_size),))
        if coeff.ndim == 2:
            c = coeff.reshape((n_species, n_x, n_xi, n_features, int(coeff.shape[1])))
            return jnp.einsum("sxlfb,ftz->sxltzb", c, features).reshape((int(op.f_size), int(coeff.shape[1])))
        raise ValueError("low-mode prolongation expects a vector or column matrix")

    correction = RHS1MatrixFreeGalerkinResidualCorrection.from_callbacks(
        operator=operator,
        restrict_fn=_restrict,
        prolong_fn=_prolong,
        n_coarse=int(n_coarse),
        regularization=float(regularization),
        damping=float(damping),
        basis_batch_size=int(basis_batch_size),
        max_coarse_size=int(max_coarse_size),
        max_basis_batch_nbytes=int(max_basis_batch_nbytes),
    )
    return correction, feature_metadata


def _rhs1_polynomial_moment_features(*, n_points: int, n_moments: int) -> np.ndarray:
    """Return small orthonormal polynomial moments on a discrete grid."""

    count = max(1, min(int(n_moments), int(n_points)))
    if int(n_points) <= 0:
        raise ValueError("moment feature grid must be nonempty")
    grid = np.linspace(-1.0, 1.0, int(n_points), dtype=np.float64) if int(n_points) > 1 else np.zeros((1,))
    candidates = [np.ones((int(n_points),), dtype=np.float64)]
    for power in range(1, count + 3):
        candidates.append(np.asarray(grid**power, dtype=np.float64))

    features: list[np.ndarray] = []
    for candidate in candidates:
        vec = np.asarray(candidate, dtype=np.float64)
        for existing in features:
            vec = vec - float(np.dot(existing, vec)) * existing
        norm = float(np.linalg.norm(vec))
        if norm > 1.0e-14:
            features.append(vec / norm)
        if len(features) >= count:
            break
    if len(features) < count:
        raise ValueError("could not build requested polynomial moment features")
    return np.stack(features, axis=0)


def _rhs1_low_legendre_index_features(*, n_xi: int, n_moments: int) -> np.ndarray:
    """Return low-L selector moments for the pitch/Legendre axis."""

    count = max(1, min(int(n_moments), int(n_xi)))
    features = np.zeros((count, int(n_xi)), dtype=np.float64)
    for ell in range(count):
        features[ell, ell] = 1.0
    return features


def _build_rhs1_moment_angular_matrix_free_correction(
    *,
    op: Any,
    operator: Any,
    theta_modes: int,
    zeta_modes: int,
    x_moments: int,
    xi_moments: int,
    max_coarse_size: int,
    max_basis_batch_nbytes: int,
    basis_batch_size: int,
    regularization: float,
    damping: float,
) -> tuple[RHS1MatrixFreeGalerkinResidualCorrection, dict[str, int | bool]]:
    """Build a compact physics-moment Galerkin correction."""

    n_species = int(op.n_species)
    n_x = int(op.n_x)
    n_xi = int(op.n_xi)
    n_theta = int(op.n_theta)
    n_zeta = int(op.n_zeta)
    x_features_np = _rhs1_polynomial_moment_features(n_points=n_x, n_moments=int(x_moments))
    xi_features_np = _rhs1_low_legendre_index_features(n_xi=n_xi, n_moments=int(xi_moments))
    features_requested = _rhs1_lowmode_angular_features(
        n_theta=n_theta,
        n_zeta=n_zeta,
        theta_modes=int(theta_modes),
        zeta_modes=int(zeta_modes),
    )
    capped_features, feature_metadata = _rhs1_cap_lowmode_features(
        features=features_requested,
        n_species=n_species,
        n_x=int(x_features_np.shape[0]),
        n_xi=int(xi_features_np.shape[0]),
        max_coarse_size=int(max_coarse_size),
    )
    x_features = jnp.asarray(x_features_np, dtype=jnp.float64)
    xi_features = jnp.asarray(xi_features_np, dtype=jnp.float64)
    angular_features = jnp.asarray(capped_features, dtype=jnp.float64)
    n_x_features = int(x_features.shape[0])
    n_xi_features = int(xi_features.shape[0])
    n_angular_features = int(angular_features.shape[0])
    n_coarse = n_species * n_x_features * n_xi_features * n_angular_features

    def _restrict(vector: jnp.ndarray) -> jnp.ndarray:
        arr = jnp.asarray(vector, dtype=jnp.float64)
        if arr.ndim == 1:
            f = arr.reshape((n_species, n_x, n_xi, n_theta, n_zeta))
            return jnp.einsum(
                "sxltz,px,ql,ftz->spqf",
                f,
                x_features,
                xi_features,
                angular_features,
            ).reshape((n_coarse,))
        if arr.ndim == 2:
            f = arr.reshape((n_species, n_x, n_xi, n_theta, n_zeta, int(arr.shape[1])))
            return jnp.einsum(
                "sxltzb,px,ql,ftz->spqfb",
                f,
                x_features,
                xi_features,
                angular_features,
            ).reshape((n_coarse, int(arr.shape[1])))
        raise ValueError("moment-space restriction expects a vector or column matrix")

    def _prolong(coefficients: jnp.ndarray) -> jnp.ndarray:
        coeff = jnp.asarray(coefficients, dtype=jnp.float64)
        if coeff.ndim == 1:
            c = coeff.reshape((n_species, n_x_features, n_xi_features, n_angular_features))
            return jnp.einsum(
                "spqf,px,ql,ftz->sxltz",
                c,
                x_features,
                xi_features,
                angular_features,
            ).reshape((int(op.f_size),))
        if coeff.ndim == 2:
            c = coeff.reshape(
                (n_species, n_x_features, n_xi_features, n_angular_features, int(coeff.shape[1]))
            )
            return jnp.einsum(
                "spqfb,px,ql,ftz->sxltzb",
                c,
                x_features,
                xi_features,
                angular_features,
            ).reshape((int(op.f_size), int(coeff.shape[1])))
        raise ValueError("moment-space prolongation expects a vector or column matrix")

    correction = RHS1MatrixFreeGalerkinResidualCorrection.from_callbacks(
        operator=operator,
        restrict_fn=_restrict,
        prolong_fn=_prolong,
        n_coarse=int(n_coarse),
        regularization=float(regularization),
        damping=float(damping),
        basis_batch_size=int(basis_batch_size),
        max_coarse_size=int(max_coarse_size),
        max_basis_batch_nbytes=int(max_basis_batch_nbytes),
    )
    moment_metadata = {
        **feature_metadata,
        "x_moments_requested": int(x_moments),
        "x_moments_retained": int(n_x_features),
        "xi_moments_requested": int(xi_moments),
        "xi_moments_retained": int(n_xi_features),
    }
    return correction, moment_metadata


def _build_rhs1_coupled_moment_matrix_free_correction(
    *,
    op: Any,
    operator: Any,
    theta_modes: int,
    zeta_modes: int,
    x_moments: int,
    xi_moments: int,
    max_tail_size: int,
    max_coarse_size: int,
    max_basis_batch_nbytes: int,
    basis_batch_size: int,
    regularization: float,
    damping: float,
) -> tuple[RHS1MatrixFreeGalerkinResidualCorrection, dict[str, int | bool | str]]:
    """Build a coupled f/tail moment residual equation for the full system."""

    n_species = int(op.n_species)
    n_x = int(op.n_x)
    n_xi = int(op.n_xi)
    n_theta = int(op.n_theta)
    n_zeta = int(op.n_zeta)
    f_size = int(op.f_size)
    total_size = int(op.total_size)
    phi1_size = int(op.phi1_size)
    extra_size = int(op.extra_size)
    tail_size = total_size - f_size
    x_features_np = _rhs1_polynomial_moment_features(n_points=n_x, n_moments=int(x_moments))
    xi_features_np = _rhs1_low_legendre_index_features(n_xi=n_xi, n_moments=int(xi_moments))
    features_requested = _rhs1_lowmode_angular_features(
        n_theta=n_theta,
        n_zeta=n_zeta,
        theta_modes=int(theta_modes),
        zeta_modes=int(zeta_modes),
    )

    if tail_size <= int(max_tail_size):
        tail_indices_np = np.arange(f_size, total_size, dtype=np.int32)
        tail_policy = "all_tail"
    elif extra_size > 0:
        tail_indices_np = np.arange(f_size + phi1_size, total_size, dtype=np.int32)
        tail_policy = "constraints_only"
    else:
        tail_indices_np = np.zeros((0,), dtype=np.int32)
        tail_policy = "none"
    tail_count = int(tail_indices_np.size)
    moment_stride = n_species * int(x_features_np.shape[0]) * int(xi_features_np.shape[0])
    angular_max = max(1, int(max_coarse_size) - tail_count)
    capped_features, feature_metadata = _rhs1_cap_lowmode_features(
        features=features_requested,
        n_species=n_species,
        n_x=int(x_features_np.shape[0]),
        n_xi=int(xi_features_np.shape[0]),
        max_coarse_size=int(angular_max),
    )
    x_features = jnp.asarray(x_features_np, dtype=jnp.float64)
    xi_features = jnp.asarray(xi_features_np, dtype=jnp.float64)
    angular_features = jnp.asarray(capped_features, dtype=jnp.float64)
    tail_indices = jnp.asarray(tail_indices_np, dtype=jnp.int32)
    n_x_features = int(x_features.shape[0])
    n_xi_features = int(xi_features.shape[0])
    n_angular_features = int(angular_features.shape[0])
    n_f_coarse = n_species * n_x_features * n_xi_features * n_angular_features
    n_coarse = int(n_f_coarse + tail_count)
    if n_coarse > int(max_coarse_size):
        raise MemoryError(
            "structured f-block coupled moment Schur coarse space too large: "
            f"{n_coarse} > {int(max_coarse_size)}; reduce moments or raise "
            "SFINCS_JAX_RHSMODE1_STRUCTURED_FBLOCK_FP_COUPLED_MAX_COARSE"
        )

    def _restrict(vector: jnp.ndarray) -> jnp.ndarray:
        arr = jnp.asarray(vector, dtype=jnp.float64)
        if arr.ndim == 1:
            f = arr[:f_size].reshape((n_species, n_x, n_xi, n_theta, n_zeta))
            moments = jnp.einsum(
                "sxltz,px,ql,ftz->spqf",
                f,
                x_features,
                xi_features,
                angular_features,
            ).reshape((n_f_coarse,))
            tail = arr[tail_indices] if tail_count > 0 else jnp.zeros((0,), dtype=jnp.float64)
            return jnp.concatenate([moments, tail], axis=0)
        if arr.ndim == 2:
            f = arr[:f_size, :].reshape((n_species, n_x, n_xi, n_theta, n_zeta, int(arr.shape[1])))
            moments = jnp.einsum(
                "sxltzb,px,ql,ftz->spqfb",
                f,
                x_features,
                xi_features,
                angular_features,
            ).reshape((n_f_coarse, int(arr.shape[1])))
            tail = arr[tail_indices, :] if tail_count > 0 else jnp.zeros((0, int(arr.shape[1])), dtype=jnp.float64)
            return jnp.concatenate([moments, tail], axis=0)
        raise ValueError("coupled moment restriction expects a vector or column matrix")

    def _prolong(coefficients: jnp.ndarray) -> jnp.ndarray:
        coeff = jnp.asarray(coefficients, dtype=jnp.float64)
        if coeff.ndim == 1:
            moment_coeff = coeff[:n_f_coarse].reshape(
                (n_species, n_x_features, n_xi_features, n_angular_features)
            )
            f = jnp.einsum(
                "spqf,px,ql,ftz->sxltz",
                moment_coeff,
                x_features,
                xi_features,
                angular_features,
            ).reshape((f_size,))
            out = jnp.concatenate([f, jnp.zeros((tail_size,), dtype=jnp.float64)], axis=0)
            if tail_count > 0:
                out = out.at[tail_indices].set(coeff[n_f_coarse:], unique_indices=True)
            return out
        if coeff.ndim == 2:
            n_cols = int(coeff.shape[1])
            moment_coeff = coeff[:n_f_coarse, :].reshape(
                (n_species, n_x_features, n_xi_features, n_angular_features, n_cols)
            )
            f = jnp.einsum(
                "spqfb,px,ql,ftz->sxltzb",
                moment_coeff,
                x_features,
                xi_features,
                angular_features,
            ).reshape((f_size, n_cols))
            out = jnp.concatenate([f, jnp.zeros((tail_size, n_cols), dtype=jnp.float64)], axis=0)
            if tail_count > 0:
                out = out.at[tail_indices, :].set(coeff[n_f_coarse:, :], unique_indices=True)
            return out
        raise ValueError("coupled moment prolongation expects a vector or column matrix")

    correction = RHS1MatrixFreeGalerkinResidualCorrection.from_callbacks(
        operator=operator,
        restrict_fn=_restrict,
        prolong_fn=_prolong,
        n_coarse=int(n_coarse),
        regularization=float(regularization),
        damping=float(damping),
        basis_batch_size=int(basis_batch_size),
        max_coarse_size=int(max_coarse_size),
        max_basis_batch_nbytes=int(max_basis_batch_nbytes),
    )
    metadata: dict[str, int | bool | str] = {
        **feature_metadata,
        "x_moments_requested": int(x_moments),
        "x_moments_retained": int(n_x_features),
        "xi_moments_requested": int(xi_moments),
        "xi_moments_retained": int(n_xi_features),
        "f_coarse_size": int(n_f_coarse),
        "tail_size": int(tail_size),
        "tail_count": int(tail_count),
        "tail_policy": tail_policy,
        "max_tail_size": int(max_tail_size),
        "moment_stride": int(moment_stride),
    }
    return correction, metadata


def _build_rhs1_tail_matrix_free_correction(
    *,
    op: Any,
    operator: Any,
    max_tail_size: int,
    max_coarse_size: int,
    max_basis_batch_nbytes: int,
    max_action_nbytes: int,
    basis_batch_size: int,
    regularization: float,
    damping: float,
) -> tuple[RHS1MatrixFreeLeastSquaresResidualCorrection, dict[str, int | str]]:
    """Build a full-system residual equation that only prolongs tail variables."""

    f_size = int(op.f_size)
    total_size = int(op.total_size)
    phi1_size = int(op.phi1_size)
    extra_size = int(op.extra_size)
    tail_size = total_size - f_size
    if tail_size <= int(max_tail_size):
        tail_indices_np = np.arange(f_size, total_size, dtype=np.int32)
        tail_policy = "all_tail"
    elif extra_size > 0:
        tail_indices_np = np.arange(f_size + phi1_size, total_size, dtype=np.int32)
        tail_policy = "constraints_only"
    else:
        tail_indices_np = np.zeros((0,), dtype=np.int32)
        tail_policy = "none"
    tail_count = int(tail_indices_np.size)
    if tail_count <= 0:
        raise NotImplementedError("tail-coupled Schur correction requires non-f tail variables")
    if tail_count > int(max_coarse_size):
        raise MemoryError(
            "structured f-block tail-coupled Schur coarse space too large: "
            f"{tail_count} > {int(max_coarse_size)}; raise "
            "SFINCS_JAX_RHSMODE1_STRUCTURED_FBLOCK_FP_TAIL_COUPLED_MAX_COARSE "
            "or reduce selected tail variables"
        )
    tail_indices = jnp.asarray(tail_indices_np, dtype=jnp.int32)

    def _prolong(coefficients: jnp.ndarray) -> jnp.ndarray:
        coeff = jnp.asarray(coefficients, dtype=jnp.float64)
        if coeff.ndim == 1:
            out = jnp.zeros((total_size,), dtype=jnp.float64)
            return out.at[tail_indices].set(coeff, unique_indices=True)
        if coeff.ndim == 2:
            out = jnp.zeros((total_size, int(coeff.shape[1])), dtype=jnp.float64)
            return out.at[tail_indices, :].set(coeff, unique_indices=True)
        raise ValueError("tail-coupled prolongation expects a vector or column matrix")

    correction = RHS1MatrixFreeLeastSquaresResidualCorrection.from_callbacks(
        operator=operator,
        prolong_fn=_prolong,
        n_coarse=int(tail_count),
        regularization=float(regularization),
        damping=float(damping),
        basis_batch_size=int(basis_batch_size),
        max_coarse_size=int(max_coarse_size),
        max_basis_batch_nbytes=int(max_basis_batch_nbytes),
        max_action_nbytes=int(max_action_nbytes),
    )
    metadata: dict[str, int | str] = {
        "tail_size": int(tail_size),
        "tail_count": int(tail_count),
        "tail_policy": tail_policy,
        "max_tail_size": int(max_tail_size),
        "max_action_nbytes": int(max_action_nbytes),
    }
    return correction, metadata


__all__ = [
    "_build_rhs1_coupled_moment_matrix_free_correction",
    "_build_rhs1_lowmode_angular_matrix_free_correction",
    "_build_rhs1_moment_angular_matrix_free_correction",
    "_build_rhs1_tail_matrix_free_correction",
    "_rhs1_cap_lowmode_features",
    "_rhs1_low_legendre_index_features",
    "_rhs1_lowmode_angular_features",
    "_rhs1_polynomial_moment_features",
]
