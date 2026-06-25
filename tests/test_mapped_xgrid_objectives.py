from __future__ import annotations

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from sfincs_jax.workflows.mapped_xgrid_objectives import (  # noqa: E402
    brute_force_rational_tail_moment_baseline,
    maxwellian_speed_moment,
    rational_tail_transport_grid,
    transport_moment_objective,
    transport_moment_report,
)


def _finite_difference(f, x: float, *, eps: float = 1.0e-6) -> float:
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


def test_analytic_maxwellian_speed_moments():
    np.testing.assert_allclose(float(maxwellian_speed_moment(0.0)), 0.5 * np.sqrt(np.pi), rtol=1.0e-14)
    np.testing.assert_allclose(float(maxwellian_speed_moment(1.0)), 0.5, rtol=1.0e-14)
    np.testing.assert_allclose(float(maxwellian_speed_moment(2.0)), 0.25 * np.sqrt(np.pi), rtol=1.0e-14)


def test_transport_moment_report_shapes_and_regularization():
    grid = rational_tail_transport_grid(12, log_length=-0.2)
    report = transport_moment_report(
        grid,
        powers=(2.0, 4.0, 6.0),
        moment_weights=(1.0, 0.5, 0.25),
        regularization_weights={"smoothness": 1.0e-6, "jac_roughness": 1.0e-6},
    )

    assert report.powers.shape == (3,)
    assert report.moments.shape == (3,)
    assert report.references.shape == (3,)
    assert report.relative_errors.shape == (3,)
    assert float(report.objective) >= float(report.moment_loss) >= 0.0
    assert float(report.regularization["min_dx"]) > 0.0


def test_transport_moment_objective_gradient_matches_finite_difference():
    powers = (0.0, 2.0, 4.0, 6.0)
    regularization = {"smoothness": 1.0e-7, "jac_roughness": 1.0e-7}

    def objective(log_length):
        grid = rational_tail_transport_grid(14, log_length)
        return transport_moment_objective(grid, powers=powers, regularization_weights=regularization)

    x0 = -0.1
    ad = float(jax.grad(lambda z: objective(z))(jnp.asarray(x0, dtype=jnp.float64)))
    fd = _finite_difference(lambda z: objective(jnp.asarray(z, dtype=jnp.float64)), x0)
    np.testing.assert_allclose(ad, fd, rtol=1.0e-6, atol=1.0e-6)


def test_brute_force_rational_tail_baseline_selects_best_candidate():
    values = jnp.linspace(-1.5, 1.0, 21)
    result = brute_force_rational_tail_moment_baseline(10, log_length_values=values, powers=(2.0, 4.0, 6.0))

    objectives = np.asarray(result["objectives"])
    best_idx = int(np.argmin(objectives))
    np.testing.assert_allclose(float(result["objective"]), float(objectives[best_idx]), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(float(result["log_length"]), float(np.asarray(values)[best_idx]), rtol=0.0, atol=0.0)
    assert float(result["objective"]) <= float(objectives[0])
    assert float(result["objective"]) <= float(objectives[-1])


def test_brute_force_rational_tail_default_scan_is_finite():
    result = brute_force_rational_tail_moment_baseline(6, powers=(2.0,))
    assert result["objectives"].shape == (81,)
    assert bool(jnp.all(jnp.isfinite(result["objectives"])))
    assert bool(jnp.isfinite(result["objective"]))


def test_transport_moment_input_validation():
    grid = rational_tail_transport_grid(8, log_length=0.0)
    with pytest.raises(ValueError, match="moment_weights"):
        transport_moment_report(grid, powers=(2.0, 4.0), moment_weights=(1.0,))
    with pytest.raises(KeyError, match="Unknown mapped-grid"):
        transport_moment_report(grid, regularization_weights={"not_a_metric": 1.0})
    with pytest.raises(ValueError, match="log_length_values"):
        brute_force_rational_tail_moment_baseline(8, log_length_values=jnp.ones((2, 2)))
