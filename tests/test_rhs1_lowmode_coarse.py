from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from sfincs_jax.rhs1_lowmode_coarse import (
    _build_rhs1_lowmode_angular_matrix_free_correction,
    _build_rhs1_moment_angular_matrix_free_correction,
    _rhs1_cap_lowmode_features,
    _rhs1_low_legendre_index_features,
    _rhs1_lowmode_angular_features,
    _rhs1_polynomial_moment_features,
)


class _IdentityOperator:
    def __init__(self, size: int) -> None:
        self.shape = (int(size), int(size))
        self.blocks = jnp.ones((1,), dtype=jnp.float64)

    def matmat(self, x):
        return jnp.asarray(x, dtype=jnp.float64)


def test_lowmode_angular_features_are_normalized_and_bounded() -> None:
    features = _rhs1_lowmode_angular_features(n_theta=5, n_zeta=7, theta_modes=1, zeta_modes=1)

    assert features.shape == (5, 5, 7)
    norms = np.linalg.norm(features.reshape((features.shape[0], -1)), axis=1)
    np.testing.assert_allclose(norms, np.ones_like(norms), rtol=1.0e-13, atol=1.0e-13)

    capped, metadata = _rhs1_cap_lowmode_features(
        features=features,
        n_species=2,
        n_x=3,
        n_xi=4,
        max_coarse_size=48,
    )
    assert capped.shape[0] == 2
    assert metadata["requested_features"] == 5
    assert metadata["retained_features"] == 2
    assert metadata["truncated_features"] is True
    assert metadata["retained_coarse_size"] == 48


def test_lowmode_feature_cap_rejects_impossible_minimum() -> None:
    features = np.ones((1, 2, 2), dtype=np.float64)

    with pytest.raises(MemoryError, match="low-mode Schur coarse space too large"):
        _rhs1_cap_lowmode_features(
            features=features,
            n_species=2,
            n_x=3,
            n_xi=4,
            max_coarse_size=23,
        )


def test_polynomial_and_low_legendre_moments_are_orthonormal_selectors() -> None:
    poly = _rhs1_polynomial_moment_features(n_points=5, n_moments=3)
    np.testing.assert_allclose(poly @ poly.T, np.eye(3), rtol=1.0e-13, atol=1.0e-13)

    low_l = _rhs1_low_legendre_index_features(n_xi=4, n_moments=6)
    assert low_l.shape == (4, 4)
    np.testing.assert_allclose(low_l, np.eye(4), rtol=0.0, atol=0.0)


def test_lowmode_matrix_free_correction_projects_identity_residual() -> None:
    op = SimpleNamespace(n_species=1, n_x=1, n_xi=1, n_theta=5, n_zeta=7, f_size=35)
    correction, metadata = _build_rhs1_lowmode_angular_matrix_free_correction(
        op=op,
        operator=_IdentityOperator(op.f_size),
        theta_modes=0,
        zeta_modes=0,
        max_coarse_size=8,
        max_basis_batch_nbytes=4096,
        basis_batch_size=1,
        regularization=0.0,
        damping=1.0,
    )

    residual = jnp.ones((op.f_size,), dtype=jnp.float64)
    projected = correction.apply(residual)
    np.testing.assert_allclose(np.asarray(projected), np.asarray(residual), rtol=1.0e-12, atol=1.0e-12)
    assert metadata["retained_features"] == 1
    assert correction.to_dict()["basis_storage_nbytes"] == 0


def test_moment_matrix_free_correction_keeps_compact_coarse_metadata() -> None:
    op = SimpleNamespace(n_species=1, n_x=3, n_xi=2, n_theta=3, n_zeta=5, f_size=90)
    correction, metadata = _build_rhs1_moment_angular_matrix_free_correction(
        op=op,
        operator=_IdentityOperator(op.f_size),
        theta_modes=0,
        zeta_modes=0,
        x_moments=1,
        xi_moments=2,
        max_coarse_size=8,
        max_basis_batch_nbytes=8192,
        basis_batch_size=2,
        regularization=0.0,
        damping=1.0,
    )

    residual = jnp.ones((op.f_size,), dtype=jnp.float64)
    projected = correction.apply(residual)
    assert projected.shape == residual.shape
    assert np.all(np.isfinite(np.asarray(projected)))
    assert metadata["x_moments_retained"] == 1
    assert metadata["xi_moments_retained"] == 2
    assert correction.n_coarse == 2
