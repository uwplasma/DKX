"""Physics and differentiability gates for the bounce-averaged 1/nu surrogate.

:mod:`sfincs_jax.bounce_averaged` computes the effective ripple ``epsilon_eff``
and the trapped-particle bounce integrals of the radial magnetic drift from the
``|B|`` Boozer spectrum.  These tests pin four things:

* analytic limits -- an axisymmetric tokamak and a single-helicity
  (quasisymmetric) field both have *zero* ``1/nu`` transport, and the surrogate
  reproduces ``Gamma_c = 0`` to machine precision (the finite ``1/nu`` transport
  of the reference helical deck is the symmetry-breaking beat between its
  toroidal and helical modes);
* the differentiable bounce-averaging kernel -- the second adiabatic invariant
  and the bounce-averaged radial drift match central finite differences of a
  Boozer amplitude to ``<= 1e-5`` (measured ``~1e-10``);
* jit/vmap safety and a flowing ``epsilon_eff`` gradient;
* low-collisionality cross-validation against the full drift-kinetic solve --
  the monoenergetic ``D11* nu*`` enters the ``1/nu`` regime and its
  geometry dependence converges to the surrogate's ``Gamma_c`` ratio as
  ``nu -> 0`` (numbers recorded 2026-07-13, float64, tier-1 direct solves).

Measured full-DKE ``D11* nu*`` on the tiny helical scheme-1 deck
(``epsilon_t = -0.07053``, ``epsilon_h`` scanned, ``iota = 0.4542``):

    nuPrime   base(eps_h)      2x(eps_h)        ratio
    1e-2      3.510e-2         --               --
    3e-3      1.503e-2         3.831e-2         2.549
    1e-3      8.898e-3         2.758e-2         3.100

The surrogate ``Gamma_c`` ratio is ``3.582`` (``nu -> 0`` limit); the DKE ratio
approaches it monotonically from below (2.549 -> 3.100 -> ...).
"""

from __future__ import annotations

import math
import re
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Importing the module enables float64 (required by the finite-difference gate).
from sfincs_jax.bounce_averaged import (
    BounceAveragedTransport,
    bounce_averaged_transport,
    deep_well_bounce_integrals,
    effective_ripple,
)
from sfincs_jax.magnetic_geometry import FluxSurfaceGeometry

HELICAL_DECK = Path(__file__).parent / "ref" / "monoenergetic_PAS_tiny_scheme1.input.namelist"

# Reference helical deck geometry constants.
_EPS_T = -0.07053
_EPS_H = 0.05067
_IOTA = 0.4542
_G_HAT = 3.7481
_N_PERIODS = 10
_R_EFF = 0.2645  # ~ |epsilon_t| R0, the large-aspect-ratio effective radius


def _scheme1(eps_t: float, eps_h: float, n: int = 49) -> FluxSurfaceGeometry:
    theta = jnp.linspace(0.0, 2.0 * math.pi, n, endpoint=False, dtype=jnp.float64)
    zeta = jnp.linspace(0.0, 2.0 * math.pi / _N_PERIODS, n, endpoint=False, dtype=jnp.float64)
    return FluxSurfaceGeometry.from_scheme(
        1, theta=theta, zeta=zeta, epsilon_t=eps_t, epsilon_h=eps_h, iota=_IOTA,
        g_hat=_G_HAT, i_hat=0.0, helicity_l=2, helicity_n=_N_PERIODS, b0_over_bbar=1.0,
    )  # fmt: skip


def _from_fourier(bmnc: jnp.ndarray, n: int = 49) -> FluxSurfaceGeometry:
    """Reference helical spectrum ``B = 1 + bmnc1 cos(theta) + bmnc2 cos(2 theta - 10 zeta)``."""
    theta = jnp.linspace(0.0, 2.0 * math.pi, n, endpoint=False, dtype=jnp.float64)
    zeta = jnp.linspace(0.0, 2.0 * math.pi / _N_PERIODS, n, endpoint=False, dtype=jnp.float64)
    return FluxSurfaceGeometry.from_fourier(
        theta=theta, zeta=zeta, bmnc=bmnc,
        m=jnp.array([0.0, 1.0, 2.0]), n=jnp.array([0.0, 0.0, 1.0]),
        n_periods=_N_PERIODS, iota=_IOTA, g_hat=_G_HAT, i_hat=0.0,
    )  # fmt: skip


_BMNC0 = jnp.array([1.0, _EPS_T, _EPS_H], dtype=jnp.float64)


# ---------------------------------------------------------------------------
# Analytic limits: quasisymmetry / axisymmetry => zero 1/nu transport
# ---------------------------------------------------------------------------


def test_axisymmetric_tokamak_has_no_ripple_transport() -> None:
    """An axisymmetric tokamak (``epsilon_h = 0``) has zero ``1/nu`` transport.

    The bounce-averaged radial drift of a symmetric banana orbit vanishes, so
    ``Gamma_c`` (hence ``epsilon_eff``) is zero to machine precision.
    """
    result = bounce_averaged_transport(_scheme1(_EPS_T, 0.0), r_eff=_R_EFF)
    assert float(result.gamma_c) < 1e-12
    assert float(result.epsilon_eff) < 1e-6


def test_single_helicity_field_is_quasisymmetric() -> None:
    """A single-helicity ripple (``epsilon_t = 0``) is quasisymmetric: no ``1/nu``.

    The finite transport of the reference deck therefore comes entirely from the
    symmetry-breaking beat between its toroidal and helical modes.
    """
    result = bounce_averaged_transport(_scheme1(0.0, _EPS_H), r_eff=_R_EFF)
    assert float(result.gamma_c) < 1e-8


def test_helical_deck_transport_is_finite_and_grid_independent() -> None:
    """The reference deck has finite ``Gamma_c`` and a band-limited (grid-exact) value."""
    values = [float(bounce_averaged_transport(_scheme1(_EPS_T, _EPS_H, n=g), r_eff=_R_EFF).gamma_c)
              for g in (33, 49, 65)]  # fmt: skip
    assert values[0] > 1e-5
    # spectral evaluation of a 3-mode field is exact on any resolving grid
    assert values[1] == pytest.approx(values[0], rel=1e-9)
    assert values[2] == pytest.approx(values[0], rel=1e-9)
    result = bounce_averaged_transport(_scheme1(_EPS_T, _EPS_H), r_eff=_R_EFF)
    assert isinstance(result, BounceAveragedTransport)
    # effective ripple comparable to the physical ripple amplitude
    assert 0.02 < float(result.epsilon_eff) < 0.12


# ---------------------------------------------------------------------------
# Differentiable bounce-averaging kernel (the crux): grad matches FD to 1e-5
# ---------------------------------------------------------------------------


def test_second_adiabatic_invariant_gradient_matches_finite_difference() -> None:
    """``jax.grad`` of the deepest-well second adiabatic invariant vs central FD.

    The sine-substitution bounce integral with differentiable bounce points
    (arXiv:2412.01724) is smooth: the gradient w.r.t. the helical Boozer
    amplitude matches central differences far inside the ``1e-5`` gate.
    """
    def j_inv(bmnc: jnp.ndarray) -> jnp.ndarray:
        return deep_well_bounce_integrals(_from_fourier(bmnc), 0.4)[0]

    grad = float(jax.grad(j_inv)(_BMNC0)[2])
    h = 1e-6
    fd = (float(j_inv(_BMNC0.at[2].add(h))) - float(j_inv(_BMNC0.at[2].add(-h)))) / (2.0 * h)
    assert abs(grad - fd) / abs(fd) < 1e-5


def test_bounce_averaged_radial_drift_gradient_matches_finite_difference() -> None:
    """``jax.grad`` of the bounce-averaged radial drift vs central FD (<= 1e-5)."""
    def drift(bmnc: jnp.ndarray) -> jnp.ndarray:
        return deep_well_bounce_integrals(_from_fourier(bmnc), 0.4)[1]

    grad = float(jax.grad(drift)(_BMNC0)[2])
    h = 1e-6
    fd = (float(drift(_BMNC0.at[2].add(h))) - float(drift(_BMNC0.at[2].add(-h)))) / (2.0 * h)
    assert abs(grad - fd) / abs(fd) < 1e-5


def test_effective_ripple_gradient_flows_with_correct_sign() -> None:
    """``jax.grad(epsilon_eff)`` through ``from_fourier`` is finite and physical.

    Assembling the pitch integral introduces integrable jump discontinuities at
    trapped-well bifurcations, so the assembled metric is not FD-exact (a
    documented deferral -- see the module docstring); the gradient nonetheless
    flows and is a valid descent direction (deepening the helical ripple raises
    ``epsilon_eff``).
    """
    def eps(bmnc: jnp.ndarray) -> jnp.ndarray:
        return effective_ripple(_from_fourier(bmnc), r_eff=_R_EFF)

    grad = np.asarray(jax.grad(eps)(_BMNC0))
    assert np.all(np.isfinite(grad))
    assert grad[2] > 0.0  # increasing helical ripple increases the effective ripple


# ---------------------------------------------------------------------------
# jit / vmap safety
# ---------------------------------------------------------------------------


def test_jit_and_vmap_over_geometries() -> None:
    """The surrogate is jit-compilable and vmaps over a batch of geometries."""
    from sfincs_jax.bounce_averaged import _gamma_c

    jitted = jax.jit(lambda bh: _gamma_c(bh, _G_HAT, 0.0, _IOTA, _N_PERIODS, 1.0,
                                         12, 8, 80, 48, 96, 14, 160, 1)[0])  # fmt: skip
    b_hats = jnp.stack([_scheme1(_EPS_T, e).b_hat for e in (0.04, _EPS_H, 0.06)])
    out = np.asarray(jax.vmap(jitted)(b_hats))
    assert out.shape == (3,)
    assert np.all(out > 0.0)
    # deeper ripple -> more 1/nu transport
    assert out[0] < out[1] < out[2]


# ---------------------------------------------------------------------------
# Low-collisionality cross-validation against the full drift-kinetic solve
# ---------------------------------------------------------------------------


def _d11_star_nu_star(tmp_path: Path, *, eps_h: float, nu: float, res: tuple[int, int, int]) -> float:
    from sfincs_jax.monoenergetic import monoenergetic_database

    text = HELICAL_DECK.read_text()

    def sub(key: str, value: str, source: str) -> str:
        return re.sub(rf"(?mi)^\s*{key}\s*=.*$", f"  {key} = {value}", source)

    text = sub("saveMatricesAndVectorsInBinary", ".false.", text)
    text = sub("epsilon_h", f"{eps_h:.6e}".replace("e", "d"), text)
    text = sub("Ntheta", str(res[0]), text)
    text = sub("Nzeta", str(res[1]), text)
    text = sub("Nxi", str(res[2]), text)
    case = tmp_path / f"eh{eps_h:.5f}_nu{nu:g}"
    case.mkdir()
    path = case / "input.namelist"
    path.write_text(text)
    db = monoenergetic_database(str(path), [nu], [0.0], tol=1e-11)
    return float(np.asarray(db.d11_star)[0, 0]) * float(np.asarray(db.nu_star)[0])


def test_low_collisionality_converges_to_bounce_averaged_prediction(tmp_path: Path) -> None:
    """Full-DKE ``D11* nu*`` enters the ``1/nu`` regime and its geometry
    dependence converges to the surrogate ``Gamma_c`` ratio as ``nu -> 0``.

    This is the credibility gate: the surrogate is the ``nu -> 0`` asymptote of
    the full drift-kinetic radial coefficient.  A finite-``nu`` scan approaches
    the limit (it does not equal it), so an honest measured envelope is
    asserted rather than equality.
    """
    # 1/nu regime: base D11* nu* decreases and saturates (ratios rise toward 1).
    base = {
        1e-2: _d11_star_nu_star(tmp_path, eps_h=_EPS_H, nu=1e-2, res=(21, 21, 40)),
        3e-3: _d11_star_nu_star(tmp_path, eps_h=_EPS_H, nu=3e-3, res=(25, 25, 60)),
        1e-3: _d11_star_nu_star(tmp_path, eps_h=_EPS_H, nu=1e-3, res=(31, 31, 90)),
    }
    ordered = [base[1e-2], base[3e-3], base[1e-3]]
    assert ordered[0] > ordered[1] > ordered[2] > 0.0  # decreasing (1/nu channel)
    step1 = ordered[1] / ordered[0]
    step2 = ordered[2] / ordered[1]
    assert step2 > step1  # decrements shrink: approaching the 1/nu saturation

    # geometry cross-validation: the 2x-ripple / base ratio approaches the
    # surrogate Gamma_c ratio (nu -> 0 limit) monotonically from below.
    two_3e3 = _d11_star_nu_star(tmp_path, eps_h=2 * _EPS_H, nu=3e-3, res=(25, 25, 60))
    two_1e3 = _d11_star_nu_star(tmp_path, eps_h=2 * _EPS_H, nu=1e-3, res=(31, 31, 90))
    dke_ratio_3e3 = two_3e3 / base[3e-3]
    dke_ratio_1e3 = two_1e3 / base[1e-3]

    surrogate_ratio = float(
        bounce_averaged_transport(_scheme1(_EPS_T, 2 * _EPS_H), r_eff=_R_EFF).gamma_c
        / bounce_averaged_transport(_scheme1(_EPS_T, _EPS_H), r_eff=_R_EFF).gamma_c
    )
    assert surrogate_ratio == pytest.approx(3.58, abs=0.1)
    # approaching the surrogate limit from below as nu decreases
    assert dke_ratio_3e3 < dke_ratio_1e3 < surrogate_ratio
    assert dke_ratio_1e3 == pytest.approx(3.10, abs=0.25)
    # within ~15% of the surrogate limit already at nuPrime = 1e-3
    assert dke_ratio_1e3 > 0.83 * surrogate_ratio


def test_surrogate_is_much_faster_than_the_full_dke_solve(tmp_path: Path) -> None:
    """The bounce-averaged evaluation is far cheaper than one full-DKE solve."""
    geom = _scheme1(_EPS_T, _EPS_H)
    bounce_averaged_transport(geom, r_eff=_R_EFF).gamma_c.block_until_ready()  # warm compile
    t0 = time.perf_counter()
    bounce_averaged_transport(geom, r_eff=_R_EFF).gamma_c.block_until_ready()
    t_surrogate = time.perf_counter() - t0

    t0 = time.perf_counter()
    _d11_star_nu_star(tmp_path, eps_h=_EPS_H, nu=1e-3, res=(31, 31, 90))
    t_dke = time.perf_counter() - t0

    assert t_surrogate < t_dke
    assert t_surrogate < 0.2 * t_dke  # at least a 5x surrogate speed-up
