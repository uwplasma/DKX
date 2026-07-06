from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sfincs_jax.discretization.adaptive_maps import (
    AffineXMap,
    MappedXGrid,
    RationalTailXMap,
    SoftplusCellXMap,
    SplineDensityXMap,
    barycentric_diff_matrix,
    barycentric_diff_matrices,
    chain_rule_diff_matrices,
    make_reference_eta_grid,
    maxwellian_tail_integral_proxy,
    mapped_grid_regularization,
)


def _finite_difference(f, x: float, *, eps: float = 1.0e-6) -> float:
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


def test_reference_eta_grid_gauss_integrates_low_polynomials():
    eta, weights = make_reference_eta_grid(8)

    assert eta.shape == (8,)
    assert weights.shape == (8,)
    assert bool(jnp.all(eta > 0.0))
    assert bool(jnp.all(eta < 1.0))
    np.testing.assert_allclose(float(jnp.sum(weights)), 1.0, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(float(jnp.sum(weights * eta)), 0.5, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(float(jnp.sum(weights * eta**2)), 1.0 / 3.0, rtol=0.0, atol=1.0e-14)


def test_affine_map_identity_like_quadrature_and_derivatives():
    eta, weights = make_reference_eta_grid(9)
    grid = AffineXMap()(0.0, eta=eta, eta_weights=weights)

    np.testing.assert_allclose(np.asarray(grid.x), np.asarray(eta), rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(np.asarray(grid.x_weights), np.asarray(weights), rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(float(jnp.sum(grid.x_weights)), 1.0, rtol=0.0, atol=1.0e-14)

    p = grid.x**4 - 2.0 * grid.x**2 + 3.0 * grid.x + 1.0
    dp = 4.0 * grid.x**3 - 4.0 * grid.x + 3.0
    d2p = 12.0 * grid.x**2 - 4.0
    np.testing.assert_allclose(np.asarray(grid.ddx @ p), np.asarray(dp), rtol=1.0e-10, atol=1.0e-10)
    np.testing.assert_allclose(np.asarray(grid.d2dx2 @ p), np.asarray(d2p), rtol=1.0e-9, atol=1.0e-9)


def test_barycentric_and_chain_rule_derivatives_agree_for_affine_map():
    eta, _ = make_reference_eta_grid(8)
    scale = 2.5
    x = scale * eta
    jac = jnp.full_like(eta, scale)
    jac_eta = jnp.zeros_like(eta)

    d_bary, d2_bary = barycentric_diff_matrices(x)
    d_chain, d2_chain = chain_rule_diff_matrices(eta, jac, jac_eta)

    np.testing.assert_allclose(np.asarray(d_chain), np.asarray(d_bary), rtol=1.0e-11, atol=1.0e-11)
    np.testing.assert_allclose(np.asarray(d2_chain), np.asarray(d2_bary), rtol=1.0e-9, atol=1.0e-9)


def test_barycentric_diff_matrix_differentiates_polynomials_on_irregular_nodes():
    x = jnp.asarray([0.0, 0.2, 0.7, 1.3, 2.0], dtype=jnp.float64)
    d = barycentric_diff_matrix(x)

    f = x**3 - 2.0 * x + 1.0
    expected = 3.0 * x**2 - 2.0

    np.testing.assert_allclose(np.asarray(d @ f), np.asarray(expected), rtol=1.0e-11, atol=1.0e-11)


def test_mapped_grid_regularization_reports_spacing_and_tail_metrics():
    x = jnp.asarray([0.0, 0.25, 0.75, 1.5], dtype=jnp.float64)
    jac = jnp.asarray([0.25, 0.25, 0.5, 0.75], dtype=jnp.float64)

    reg = mapped_grid_regularization(x, jac)

    assert reg["min_dx"] == pytest.approx(0.25)
    assert reg["width_ratio"] == pytest.approx(3.0)
    assert float(reg["smoothness"]) >= 0.0
    assert float(reg["jac_roughness"]) >= 0.0
    np.testing.assert_allclose(float(reg["tail_mass_proxy"]), np.exp(-(1.5**2)), rtol=1.0e-14)


def test_rational_tail_map_is_monotone_and_differentiable():
    eta, weights = make_reference_eta_grid(12)
    mapper = RationalTailXMap(eps=1.0e-4)
    grid = mapper(jnp.log(2.0), eta=eta, eta_weights=weights)

    assert bool(jnp.all(jnp.diff(grid.x) > 0.0))
    assert bool(jnp.all(grid.jac > 0.0))
    assert float(grid.regularization["width_ratio"]) > 1.0
    assert float(maxwellian_tail_integral_proxy(grid)) > 0.0

    def objective(log_length):
        g = mapper(log_length, eta=eta, eta_weights=weights)
        return jnp.sum(g.x_weights * jnp.exp(-(g.x**2)))

    ad = float(jax.grad(objective)(jnp.asarray(0.2, dtype=jnp.float64)))
    fd = _finite_difference(lambda z: objective(jnp.asarray(z, dtype=jnp.float64)), 0.2)
    np.testing.assert_allclose(ad, fd, rtol=1.0e-6, atol=1.0e-6)


def test_softplus_cell_map_has_positive_widths_and_bounded_extent():
    eta, weights = make_reference_eta_grid(10, kind="uniform")
    params = jnp.linspace(-1.0, 1.0, eta.size)
    mapper = SoftplusCellXMap(x_min=0.0, x_max=3.0)
    grid = mapper(params, eta=eta, eta_weights=weights)

    assert bool(jnp.all(jnp.diff(grid.x) > 0.0))
    assert bool(jnp.all(grid.jac > 0.0))
    np.testing.assert_allclose(float(jnp.sum(grid.x_weights)), 3.0, rtol=1.0e-13, atol=1.0e-13)

    def objective(p):
        g = mapper(p, eta=eta, eta_weights=weights)
        return g.regularization["smoothness"] + 0.01 * jnp.sum(g.x)

    ad = np.asarray(jax.grad(objective)(params))
    eps = 1.0e-6
    idx = 3
    perturb = jnp.zeros_like(params).at[idx].set(eps)
    fd = float((objective(params + perturb) - objective(params - perturb)) / (2.0 * eps))
    np.testing.assert_allclose(ad[idx], fd, rtol=1.0e-5, atol=1.0e-5)


def test_spline_density_map_gradients_are_finite():
    eta, weights = make_reference_eta_grid(11)
    coeffs = jnp.asarray([0.1, -0.2, 0.05], dtype=jnp.float64)
    mapper = SplineDensityXMap()
    grid = mapper(coeffs, eta=eta, eta_weights=weights)

    assert bool(jnp.all(jnp.diff(grid.x) > 0.0))
    assert bool(jnp.all(grid.jac > 0.0))

    def objective(c):
        g = mapper(c, eta=eta, eta_weights=weights)
        return jnp.sum(g.x_weights * (1.0 + g.x + g.x**2)) + g.regularization["smoothness"]

    grad = jax.grad(objective)(coeffs)
    assert bool(jnp.all(jnp.isfinite(grad)))

    eps = 1.0e-6
    idx = 1
    perturb = jnp.zeros_like(coeffs).at[idx].set(eps)
    fd = float((objective(coeffs + perturb) - objective(coeffs - perturb)) / (2.0 * eps))
    np.testing.assert_allclose(float(grad[idx]), fd, rtol=1.0e-5, atol=1.0e-5)


def test_mapped_grid_pytree_roundtrip_preserves_regularization() -> None:
    eta, weights = make_reference_eta_grid(5, kind="uniform")
    grid = AffineXMap()(0.4, eta=eta, eta_weights=weights)

    children, aux = grid.tree_flatten()
    restored = MappedXGrid.tree_unflatten(aux, children)

    np.testing.assert_allclose(np.asarray(restored.x), np.asarray(grid.x), rtol=0.0, atol=0.0)
    assert set(restored.regularization) == set(grid.regularization)
    for key in grid.regularization:
        np.testing.assert_allclose(
            np.asarray(restored.regularization[key]),
            np.asarray(grid.regularization[key]),
            rtol=0.0,
            atol=0.0,
        )


def test_reference_and_barycentric_inputs_fail_fast() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        make_reference_eta_grid(1)
    with pytest.raises(ValueError, match="kind"):
        make_reference_eta_grid(3, kind="chebyshev")
    with pytest.raises(ValueError, match="one-dimensional"):
        barycentric_diff_matrix(jnp.zeros((2, 2), dtype=jnp.float64))
    with pytest.raises(ValueError, match="at least two"):
        barycentric_diff_matrix(jnp.asarray([0.0], dtype=jnp.float64))


def test_mapped_xgrid_builders_validate_shape_and_derivative_options() -> None:
    eta, weights = make_reference_eta_grid(4, kind="uniform")

    with pytest.raises(ValueError, match="derivative"):
        AffineXMap(derivative="spectral-ish")(0.0, eta=eta, eta_weights=weights)

    softplus = SoftplusCellXMap()
    with pytest.raises(ValueError, match="one-dimensional"):
        softplus(jnp.zeros((2, 2), dtype=jnp.float64), eta=eta, eta_weights=weights)
    with pytest.raises(ValueError, match="same length"):
        softplus(jnp.zeros((3,), dtype=jnp.float64), eta=eta, eta_weights=weights)

    spline = SplineDensityXMap()
    with pytest.raises(ValueError, match="one-dimensional"):
        spline(jnp.zeros((2, 2), dtype=jnp.float64), eta=eta, eta_weights=weights)
    with pytest.raises(ValueError, match="at least two"):
        spline(jnp.asarray([0.0], dtype=jnp.float64), eta=eta[:1], eta_weights=weights[:1])
