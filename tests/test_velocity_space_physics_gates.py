from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.operators.profile_full_system import _source_basis_constraint_scheme1_np
from sfincs_jax.problems.transport_matrix.diagnostics import f0_l0_v3_from_operator
from sfincs_jax.operators.profile_system import full_system_operator_from_namelist
from sfincs_jax.operators.profile_system import _source_basis_constraint_scheme_1
from sfincs_jax.discretization.xgrid import make_x_grid


@pytest.mark.parametrize("x_grid_k", [0.0, 2.0])
def test_x_grid_gaussian_quadrature_exact_for_weighted_monomials(x_grid_k: float) -> None:
    """The SFINCS speed grid must preserve Maxwellian moments exactly.

    The built-in grid is Gaussian quadrature for the weight
    ``exp(-x**2) * x**k`` on ``[0, infinity)``.  Exact low-order moments are a
    direct gate on density, flow, energy, and source constraints used by the
    kinetic equations.
    """

    n_x = 6
    x_grid = make_x_grid(n=n_x, k=x_grid_k, include_point_at_x0=False)

    for power in range(2 * n_x):
        expected = 0.5 * math.gamma((x_grid_k + power + 1.0) / 2.0)
        actual = float(np.sum(x_grid.gaussian_weights * x_grid.x**power))
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-12)

    np.testing.assert_allclose(
        x_grid.dx_weights() * np.exp(-(x_grid.x * x_grid.x)) * x_grid.x**x_grid_k,
        x_grid.gaussian_weights,
        rtol=0.0,
        atol=1.0e-14,
    )


@pytest.mark.parametrize("x_grid_k", [0.0, 2.0])
def test_x_grid_with_x0_gauss_radau_exact_for_weighted_monomials(x_grid_k: float) -> None:
    """The optional x=0 grid point must preserve low-order Maxwellian moments."""

    n_x = 6
    x_grid = make_x_grid(n=n_x, k=x_grid_k, include_point_at_x0=True)

    np.testing.assert_allclose(x_grid.x[0], 0.0, rtol=0.0, atol=0.0)
    for power in range(2 * n_x - 1):
        expected = 0.5 * math.gamma((x_grid_k + power + 1.0) / 2.0)
        actual = float(np.sum(x_grid.gaussian_weights * x_grid.x**power))
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=5.0e-12)


def test_constraint_scheme1_sources_are_density_energy_moment_biorthogonal() -> None:
    """ConstraintScheme=1 sources isolate density and energy moments.

    With the isotropic angular factor, source 1 has unit ``x**2`` moment and
    zero ``x**4`` moment, while source 2 has the opposite normalization.  This
    catches normalization regressions in the RHSMode=1 current/constraint
    equations without running a full transport solve.
    """

    x_grid = make_x_grid(n=8, k=0.0, include_point_at_x0=False)
    x = x_grid.x
    dx_weights = x_grid.dx_weights()

    source1, source2 = _source_basis_constraint_scheme_1(jnp.asarray(x, dtype=jnp.float64))
    source1_np, source2_np = _source_basis_constraint_scheme1_np(x)
    source1 = np.asarray(source1)
    source2 = np.asarray(source2)

    np.testing.assert_allclose(source1, source1_np, rtol=0.0, atol=1.0e-15)
    np.testing.assert_allclose(source2, source2_np, rtol=0.0, atol=1.0e-15)

    angular_factor = 4.0 * math.pi
    density_source1 = angular_factor * float(np.sum(dx_weights * x**2 * source1))
    density_source2 = angular_factor * float(np.sum(dx_weights * x**2 * source2))
    energy_source1 = angular_factor * float(np.sum(dx_weights * x**4 * source1))
    energy_source2 = angular_factor * float(np.sum(dx_weights * x**4 * source2))

    np.testing.assert_allclose(density_source1, 1.0, rtol=0.0, atol=2.0e-14)
    np.testing.assert_allclose(density_source2, 0.0, rtol=0.0, atol=2.0e-14)
    np.testing.assert_allclose(energy_source1, 0.0, rtol=0.0, atol=2.0e-14)
    np.testing.assert_allclose(energy_source2, 1.0, rtol=0.0, atol=2.0e-14)


def test_f0_l0_maxwellian_moments_normalize_density_and_pressure() -> None:
    """The background Maxwellian must reproduce density and pressure moments."""

    nml = read_sfincs_input("tests/ref/pas_1species_PAS_noEr_tiny.input.namelist")
    op = full_system_operator_from_namelist(nml=nml)
    f0_l0 = np.asarray(f0_l0_v3_from_operator(op))
    fs_factor = np.asarray(op.theta_weights)[:, None] * np.asarray(op.zeta_weights)[None, :] / np.asarray(op.d_hat)
    fs_sum = np.sum(fs_factor)

    fsavg_f0 = np.einsum("tz,sxtz->sx", fs_factor, f0_l0) / fs_sum
    density_moment = 4.0 * math.pi * np.einsum("x,x,sx->s", np.asarray(op.x_weights), np.asarray(op.x) ** 2, fsavg_f0)
    pressure_moment = (
        (8.0 * math.pi / 3.0)
        * np.einsum("x,x,sx->s", np.asarray(op.x_weights), np.asarray(op.x) ** 4, fsavg_f0)
    )

    np.testing.assert_allclose(density_moment, np.asarray(op.n_hat), rtol=0.0, atol=2.0e-12)
    np.testing.assert_allclose(pressure_moment, np.asarray(op.n_hat) * np.asarray(op.t_hat), rtol=0.0, atol=2.0e-12)
