from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
from jax import config as jax_config
import numpy as np

jax_config.update("jax_enable_x64", True)

from sfincs_jax.operators.profile_response.sources import (  # noqa: E402
    build_rhs1_xblock_constraint1_moment_schur_preconditioner,
    constraint_scheme1_inject_source,
    constraint_scheme1_moments_from_f,
    constraint_scheme2_inject_source,
    constraint_scheme2_source_from_f,
)


def _op(*, point_at_x0: bool = False) -> SimpleNamespace:
    n_species = 2
    n_x = 3
    n_xi = 2
    n_theta = 2
    n_zeta = 2
    return SimpleNamespace(
        n_species=n_species,
        point_at_x0=point_at_x0,
        fblock=SimpleNamespace(f_shape=(n_species, n_x, n_xi, n_theta, n_zeta)),
        theta_weights=jnp.asarray([0.25, 0.75], dtype=jnp.float64),
        zeta_weights=jnp.asarray([0.4, 0.6], dtype=jnp.float64),
        d_hat=jnp.ones((n_theta, n_zeta), dtype=jnp.float64),
        x=jnp.asarray([0.0, 1.0, 2.0], dtype=jnp.float64),
        x_weights=jnp.asarray([0.5, 1.5, 2.5], dtype=jnp.float64),
    )


def test_constraint_scheme2_source_from_f_matches_flux_surface_average() -> None:
    op = _op()
    f = jnp.arange(np.prod(op.fblock.f_shape), dtype=jnp.float64).reshape(
        op.fblock.f_shape
    )

    source = constraint_scheme2_source_from_f(op, f)

    factor = np.outer(np.asarray(op.theta_weights), np.asarray(op.zeta_weights))
    expected = np.einsum("tz,sxtz->sx", factor, np.asarray(f[:, :, 0, :, :]))
    np.testing.assert_allclose(np.asarray(source), expected)


def test_constraint_scheme2_inject_source_respects_point_at_x0() -> None:
    op = _op(point_at_x0=True)
    src = jnp.asarray([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]], dtype=jnp.float64)

    injected = np.asarray(constraint_scheme2_inject_source(op, src)).reshape(
        op.fblock.f_shape
    )

    np.testing.assert_allclose(injected[:, 0, :, :, :], 0.0)
    np.testing.assert_allclose(injected[:, 1:, 1, :, :], 0.0)
    np.testing.assert_allclose(injected[0, 1, 0, :, :], 11.0)
    np.testing.assert_allclose(injected[1, 2, 0, :, :], 22.0)


def test_constraint_scheme1_moments_from_f_matches_velocity_weighted_average() -> None:
    op = _op()
    f = jnp.arange(np.prod(op.fblock.f_shape), dtype=jnp.float64).reshape(
        op.fblock.f_shape
    )

    moments = constraint_scheme1_moments_from_f(op, f)

    factor = np.outer(np.asarray(op.theta_weights), np.asarray(op.zeta_weights))
    x = np.asarray(op.x)
    x_weights = np.asarray(op.x_weights)
    f0 = np.asarray(f[:, :, 0, :, :])
    expected_density = np.einsum("x,tz,sxtz->s", x**2 * x_weights, factor, f0)
    expected_pressure = np.einsum("x,tz,sxtz->s", x**4 * x_weights, factor, f0)
    np.testing.assert_allclose(
        np.asarray(moments),
        np.stack([expected_density, expected_pressure], axis=1),
    )


def test_constraint_scheme1_inject_source_uses_documented_source_basis() -> None:
    op = _op(point_at_x0=True)
    src = jnp.asarray([[1.5, -0.25], [0.5, 2.0]], dtype=jnp.float64)

    injected = np.asarray(constraint_scheme1_inject_source(op, src)).reshape(
        op.fblock.f_shape
    )

    x = np.asarray(op.x)
    coef = np.exp(-(x**2)) / (np.pi * np.sqrt(np.pi))
    basis1 = (-(x**2) + 2.5) * coef
    basis2 = ((2.0 / 3.0) * x**2 - 1.0) * coef
    expected_x1_species0 = basis1[1] * 1.5 + basis2[1] * (-0.25)
    expected_x2_species1 = basis1[2] * 0.5 + basis2[2] * 2.0

    np.testing.assert_allclose(injected[:, 0, :, :, :], 0.0)
    np.testing.assert_allclose(injected[:, 1:, 1, :, :], 0.0)
    np.testing.assert_allclose(injected[0, 1, 0, :, :], expected_x1_species0)
    np.testing.assert_allclose(injected[1, 2, 0, :, :], expected_x2_species1)


def test_constraint_scheme1_moment_schur_wraps_xblock_preconditioner() -> None:
    op = _op()
    op.rhs_mode = 1
    op.constraint_scheme = 1
    op.phi1_size = 0
    op.f_size = int(np.prod(op.fblock.f_shape))
    op.extra_size = 2 * int(op.n_species)
    op.total_size = op.f_size + op.extra_size

    def identity_preconditioner(v: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(v, dtype=jnp.float64)

    logs: list[str] = []
    preconditioner, metadata, stats = (
        build_rhs1_xblock_constraint1_moment_schur_preconditioner(
            op=op,
            base_preconditioner=identity_preconditioner,
            emit=lambda _level, msg: logs.append(str(msg)),
        )
    )

    rhs = jnp.arange(op.total_size, dtype=jnp.float64) / 10.0
    out = preconditioner(rhs)

    assert out.shape == (op.total_size,)
    assert np.all(np.isfinite(np.asarray(out)))
    assert metadata["mode"] == "constraint1_moment_schur"
    assert metadata["extra_size"] == op.extra_size
    assert metadata["rank"] == op.extra_size
    assert metadata["device_resident"] is True
    assert stats["applies"] == 1
    assert stats["base_applies"] == 2
    assert any("constraint1 moment-Schur built" in msg for msg in logs)
