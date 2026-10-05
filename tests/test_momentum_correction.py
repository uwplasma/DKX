"""Momentum correction of pitch-angle solves: physics anchors and AD gate.

Gates for :mod:`dkx.momentum_correction`, the moment method of H. Sugama and
S. Nishimura, Phys. Plasmas 9, 4637 (2002), as applied by H. Maassberg,
C. D. Beidler and Y. Turkin, Phys. Plasmas 16, 072504 (2009).  Every deck is a
small tokamak (geometryScheme 1) solved three ways on the same grid: plain
pitch-angle scattering (PAS), PAS with the correction, and the full linearized
Fokker-Planck operator (FP), which is the truth the correction is held to.

Measured 2026-10-04, float64, on the decks below:

=============================================  ======================
quantity                                       measured
=============================================  ======================
p+e tokamak ``<j.B>``: PAS / FP - 1            +0.084
p+e ``<j.B>`` corrected / FP - 1, 1..5 Sonine  +0.159 -0.113 -0.056 -0.048 -0.046
p+e ``sum_a Z_a Gamma_a / Gamma_i``            PAS 0.96, corrected -2.0e-3, FP 3.2e-3
p+e ``Gamma_i`` corrected / FP (3 Sonine)      0.968
H+C6 ``<j.B>``: PAS / FP - 1                   -0.843
H+C6 ``<j.B>`` corrected / FP - 1, 1..5 Sonine -0.027 -0.028 -0.031 -0.025 -0.024
=============================================  ======================
"""

from __future__ import annotations

import numpy as np
import pytest

_TWO_SPECIES_DECK = """\
&general
  RHSMode = 1
  saveMatricesAndVectorsInBinary = .false.
/
&geometryParameters
  geometryScheme = 1
  inputRadialCoordinate = 3
  rN_wish = 0.3
  B0OverBBar = 1.0d+0
  GHat = 1.0d+0
  IHat = 0.0d+0
  iota = 1.31d+0
  epsilon_t = 0.1d+0
  epsilon_h = 0.0d+0
  helicity_l = 1
  helicity_n = 1
  psiAHat = 0.045d+0
  aHat = 0.1
/
&speciesParameters
  Zs = 1 6
  mHats = 1.0d+0 6.0d+0
  nHats = 0.6d+0 0.0667d+0
  THats = 0.5d+0 0.5d+0
  dNHatdrHats = -6.0d+0 -0.5d+0
  dTHatdrHats = -3.0d+0 -3.0d+0
/
&physicsParameters
  Delta = 4.5694d-3
  alpha = 1.0d+0
  nu_n = 8.4774d-3
  Er = 0.0d+0
  collisionOperator = {collop}
  includeXDotTerm = .true.
  includeElectricFieldTermInXiDot = .true.
  includePhi1 = .false.
/
&resolutionParameters
  Ntheta = 9
  Nzeta = 1
  Nxi = 12
  NL = 4
  Nx = 6
  solverTolerance = 1d-9
/
&otherNumericalParameters
  Nxi_for_x_option = 0
  xGridScheme = 5
/
&preconditionerOptions
/
"""

_P_E = {"Zs = 1 6": "Zs = 1 -1", "mHats = 1.0d+0 6.0d+0": "mHats = 1.0d+0 5.446d-4",
        "nHats = 0.6d+0 0.0667d+0": "nHats = 0.6d+0 0.6d+0",
        "dNHatdrHats = -6.0d+0 -0.5d+0": "dNHatdrHats = -6.0d+0 -6.0d+0"}  # fmt: skip


def _operators(sub=None):
    from dkx.drift_kinetic import kinetic_operator_from_namelist
    from dkx.namelist import parse_sfincs_input_text

    def build(collop):
        text = _TWO_SPECIES_DECK.format(collop=collop)
        for old, new in (sub or {}).items():
            text = text.replace(old, new)
        return kinetic_operator_from_namelist(parse_sfincs_input_text(text))

    return build(1), build(0)


def _moments(op, state):
    from dkx.run import profile_moments_from_operator

    m = profile_moments_from_operator(op, state)
    flux = np.asarray(m["particleFlux_vm_psiHat"]).ravel()
    return float(np.asarray(m["FSABjHat"]).reshape(())), flux


@pytest.fixture(scope="module")
def electron_ion():
    from dkx.momentum_correction import momentum_corrected_solve
    from dkx.solve import solve

    pas, fp = _operators(_P_E)
    return pas, fp, solve(fp, fp.rhs(), emit=None).x, momentum_corrected_solve(pas, fp)


def test_electron_ion_bootstrap_matches_fokker_planck(electron_ion) -> None:
    """Three Sonine moments: ``<j.B>`` within 6% of FP (PAS: +8.4%)."""
    pas, fp, x_fp, result = electron_ion
    j_fp, _ = _moments(fp, x_fp)
    j_pas, _ = _moments(pas, result.uncorrected_state)
    j_corr, _ = _moments(pas, result.state)
    assert j_corr == pytest.approx(float(result.fsab_j), rel=1e-12)
    assert j_pas == pytest.approx(float(result.uncorrected_fsab_j), rel=1e-12)
    assert abs(j_corr / j_fp - 1.0) < 0.065  # measured -0.0557
    assert abs(j_corr / j_fp - 1.0) < 0.75 * abs(j_pas / j_fp - 1.0)


def test_corrected_fluxes_are_intrinsically_ambipolar(electron_ion) -> None:
    """In a tokamak, momentum conservation makes ``sum_a Z_a Gamma_a`` vanish.

    PAS breaks it; the corrected model operator conserves total parallel
    momentum exactly (the projection is weighted by ``nu_D``), so the
    back-substituted fluxes are ambipolar to the deck's discretization level,
    like FP's (measured 3.2e-3 FP, 2.0e-3 corrected, 0.96 PAS).
    """
    pas, fp, x_fp, result = electron_ion
    z = np.asarray(pas.z_s)
    _, g_fp = _moments(fp, x_fp)
    _, g_pas = _moments(pas, result.uncorrected_state)
    _, g_corr = _moments(pas, result.state)
    assert abs(z @ g_fp) < 1e-2 * abs(g_fp[0])
    assert abs(z @ g_pas) > 0.5 * abs(g_pas[0])
    assert abs(z @ g_corr) < 1e-2 * abs(g_corr[0])
    assert g_corr[0] == pytest.approx(g_fp[0], rel=0.05)


def test_impurity_friction_restores_the_bootstrap_current() -> None:
    """H + C6+: PAS lets the impurity slip (``<j.B>`` 84% low); friction locks it.

    The correction is within 4% of FP from one Sonine term on, with the friction
    evaluated once and passed in, as an optimization row would cache it.
    """
    from dkx.momentum_correction import friction_drives, momentum_corrected_solve
    from dkx.solve import solve

    pas, fp = _operators()
    j_fp, _ = _moments(fp, solve(fp, fp.rhs(), emit=None).x)
    for n in (1, 3):
        result = momentum_corrected_solve(pas, friction=friction_drives(pas, fp, n), n_sonine=n)
        assert float(result.uncorrected_fsab_j) / j_fp < 0.2
        assert abs(float(result.fsab_j) / j_fp - 1.0) < 0.04  # measured -0.027, -0.031


def test_gradient_matches_finite_differences(electron_ion) -> None:
    """``d<j.B>/d(dT/dpsi)`` through both PAS solves and the moment system."""
    import dataclasses

    import jax
    import jax.numpy as jnp

    from dkx.momentum_correction import friction_drives, momentum_corrected_solve

    pas, fp, _, _ = electron_ion
    friction = friction_drives(pas, fp)

    def current(scale):
        op = dataclasses.replace(pas, dt_hat_dpsi_hat=pas.dt_hat_dpsi_hat * scale)
        return momentum_corrected_solve(op, friction=friction).fsab_j

    one = jnp.asarray(1.0)
    eps = 1e-4
    fd = (current(one + eps) - current(one - eps)) / (2 * eps)
    np.testing.assert_allclose(float(jax.grad(current)(one)), float(fd), rtol=1e-6)


def test_retired_names_explain_themselves() -> None:
    import dkx.momentum_correction as mc
    from dkx.api import momentum_corrected_bootstrap
    from dkx.monoenergetic import MonoenergeticDatabase

    with pytest.raises(AttributeError, match="momentum_corrected_solve"):
        mc.parallel_viscosity  # noqa: B018
    with pytest.raises(AttributeError):
        mc.not_a_name  # noqa: B018
    with pytest.raises(TypeError, match="no longer takes a monoenergetic database"):
        momentum_corrected_bootstrap(MonoenergeticDatabase.__new__(MonoenergeticDatabase))
    with pytest.raises(ValueError, match="n_sonine"):
        mc._laguerre(np.zeros(3), 0)


def test_bootstrap_term_collision_model_option() -> None:
    from dkx.bootstrap import KineticBootstrapMismatch

    profiles = object()
    term = KineticBootstrapMismatch(profiles, collision_model="pas+momentum_correction")
    assert term.momentum_correction and term.collision_operator == 1
    assert KineticBootstrapMismatch(profiles, collision_model="fokker_planck").collision_operator == 0
    with pytest.raises(ValueError, match="collision_model"):
        KineticBootstrapMismatch(profiles, collision_model="sugama")
