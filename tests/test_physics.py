"""Proofs for the equilibrium Maxwellian that every moment is taken against.

plan.md section 7.4 asks for moments of manufactured distributions. ``f0`` is
the one distribution in the code whose exact form is known in closed form, and
it had no test: it is reached only indirectly, through solves whose answers
would shift but still look plausible if its normalization or its Phi1 factor
were wrong.

Each check here is against a closed form or an exactly known moment, so a
regression cannot be absorbed by refreshing a fixture.

What is deliberately *not* asserted: a physical density in SI or SFINCS-hat
units. Recovering ``n`` from ``f0`` requires the velocity-space Jacobian and
the moment normalization that ``vm_flux_moments`` applies, and pinning a
convention here that the moment routines might spell differently would be
inventing a contract rather than proving one. The shape-level invariants below
hold whatever that convention turns out to be.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from dkx.moments import SpeciesParams, maxwellian_f0_l0
from dkx.xgrid import make_x_grid

#: Speed grid with the k=2 weight the velocity moments use.
GRID = make_x_grid(n=10, k=2.0)
SPEEDS = np.asarray(GRID.x)
WEIGHTS = np.asarray(GRID.gaussian_weights)

#: (n_hat, t_hat, m_hat) triples: reference, hotter and denser, and a
#: deuteron-like heavy cold species.
PLASMAS = [(1.0, 1.0, 1.0), (3.0, 2.0, 1.0), (0.5, 0.25, 2.014)]


def one_species(n_hat: float, t_hat: float, m_hat: float, *, z_s: float = 1.0) -> SpeciesParams:
    return SpeciesParams(
        z_s=jnp.array([z_s]),
        m_hat=jnp.array([m_hat]),
        t_hat=jnp.array([t_hat]),
        n_hat=jnp.array([n_hat]),
    )


def f0_of(species: SpeciesParams, *, phi1=None, alpha: float = 1.0, n_theta: int = 3, n_zeta: int = 4):
    return np.asarray(
        maxwellian_f0_l0(
            species, jnp.asarray(SPEEDS), alpha=alpha, phi1_hat=phi1,
            n_theta=n_theta, n_zeta=n_zeta,
        )
    )


@pytest.mark.parametrize(("n_hat", "t_hat", "m_hat"), PLASMAS)
def test_f0_matches_its_documented_closed_form(n_hat: float, t_hat: float, m_hat: float) -> None:
    """``f0 = n (m/(pi T))^{3/2} exp(-x^2)``, to roundoff.

    The docstring states this formula; nothing checked it. The exponent 3/2 is
    the part that matters most -- writing it as m/(pi T) alone would leave every
    density right at T = m = 1 and wrong everywhere else, which no single-plasma
    test would catch.
    """
    f0 = f0_of(one_species(n_hat, t_hat, m_hat))
    expected = n_hat * (m_hat / (np.pi * t_hat)) ** 1.5 * np.exp(-SPEEDS**2)
    assert np.allclose(f0[0, :, 0, 0], expected, rtol=1e-14, atol=0.0)


@pytest.mark.parametrize(("n_hat", "t_hat", "m_hat"), PLASMAS)
def test_the_mean_square_speed_of_f0_is_three_halves(
    n_hat: float, t_hat: float, m_hat: float
) -> None:
    """``<x^2> = 3/2`` for a Maxwellian, whatever n, T and m are.

    This is the manufactured moment: x is the speed in thermal units, so the
    shape is universal and the ratio is exactly 3/2 by equipartition. It is
    independent of the normalization the previous test pins, so together they
    separate "right shape" from "right amplitude" -- a bug in either alone
    would leave the other passing.
    """
    profile = f0_of(one_species(n_hat, t_hat, m_hat))[0, :, 0, 0] / np.exp(-SPEEDS**2)
    mean_square = np.sum(WEIGHTS * profile * SPEEDS**2) / np.sum(WEIGHTS * profile)
    assert mean_square == pytest.approx(1.5, rel=1e-13)


def test_f0_is_uniform_over_the_flux_surface_without_phi1() -> None:
    """With no Phi1 the equilibrium Maxwellian cannot vary with theta or zeta.

    A leaked angular dependence here would be indistinguishable from physics in
    every downstream moment.
    """
    f0 = f0_of(one_species(2.0, 1.5, 1.0), n_theta=5, n_zeta=7)
    assert f0.shape == (1, SPEEDS.size, 5, 7)
    assert np.array_equal(np.broadcast_to(f0[:, :, :1, :1], f0.shape), f0)


def test_phi1_applies_the_boltzmann_factor_with_the_right_sign() -> None:
    """``f0`` picks up ``exp(-Z alpha Phi1 / T)``: positive charge is depleted where Phi1 > 0.

    The sign is the whole content. Reversing it accumulates ions in the
    potential hill instead of expelling them, which changes impurity transport
    qualitatively rather than quantitatively.
    """
    phi1 = jnp.asarray(np.array([[0.0, 0.5], [0.0, 0.5], [0.0, 0.5]]))
    ion = f0_of(one_species(1.0, 1.0, 1.0, z_s=1.0), phi1=phi1, n_zeta=2)
    assert ion[0, 0, 0, 1] / ion[0, 0, 0, 0] == pytest.approx(np.exp(-0.5), rel=1e-13)

    electron = f0_of(one_species(1.0, 1.0, 1.0, z_s=-1.0), phi1=phi1, n_zeta=2)
    assert electron[0, 0, 0, 1] / electron[0, 0, 0, 0] == pytest.approx(np.exp(0.5), rel=1e-13)


def test_the_boltzmann_factor_scales_with_alpha_and_temperature() -> None:
    """The exponent is ``Z alpha Phi1 / T``, so halving T doubles the depletion."""
    phi1 = jnp.asarray(np.array([[0.0, 0.4], [0.0, 0.4]]))
    hot = f0_of(one_species(1.0, 1.0, 1.0), phi1=phi1, n_theta=2, n_zeta=2)
    cold = f0_of(one_species(1.0, 0.5, 1.0), phi1=phi1, n_theta=2, n_zeta=2)
    assert hot[0, 0, 0, 1] / hot[0, 0, 0, 0] == pytest.approx(np.exp(-0.4), rel=1e-13)
    assert cold[0, 0, 0, 1] / cold[0, 0, 0, 0] == pytest.approx(np.exp(-0.8), rel=1e-13)

    doubled = f0_of(one_species(1.0, 1.0, 1.0), phi1=phi1, alpha=2.0, n_theta=2, n_zeta=2)
    assert doubled[0, 0, 0, 1] / doubled[0, 0, 0, 0] == pytest.approx(np.exp(-0.8), rel=1e-13)


def test_species_do_not_leak_into_one_another() -> None:
    """A multi-species call equals the single-species calls stacked.

    Every species shares one array here, so an indexing slip would mix a
    deuteron's mass into the electron row and still return finite, plausible
    numbers.
    """
    together = f0_of(
        SpeciesParams(
            z_s=jnp.array([1.0, -1.0]),
            m_hat=jnp.array([2.014, 5.45e-4]),
            t_hat=jnp.array([1.5, 0.8]),
            n_hat=jnp.array([3.0, 3.0]),
        )
    )
    deuteron = f0_of(one_species(3.0, 1.5, 2.014, z_s=1.0))
    electron = f0_of(one_species(3.0, 0.8, 5.45e-4, z_s=-1.0))
    assert np.allclose(together[0], deuteron[0], rtol=1e-14, atol=0.0)
    assert np.allclose(together[1], electron[0], rtol=1e-14, atol=0.0)


def test_f0_is_positive_everywhere() -> None:
    """A distribution function that goes negative is not a distribution.

    Cheap, and it catches a sign error in the Phi1 exponent that the ratio
    tests above would miss if both endpoints flipped together.
    """
    phi1 = jnp.asarray(np.array([[-1.0, 0.0, 2.0], [0.5, -0.5, 1.0]]))
    f0 = f0_of(one_species(1.0, 0.7, 1.0), phi1=phi1, n_theta=2, n_zeta=3)
    assert np.all(f0 > 0.0)
