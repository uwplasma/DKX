"""Canonical ambipolar radial-electric-field slice (:mod:`sfincs_jax.er`).

Pins the ``er.py`` slice against the legacy path it replaces:

- :func:`sfincs_jax.er.radial_current` at three ``E_r`` reproduces a direct
  ``run_profile`` particle-flux computation to machine precision;
- :func:`sfincs_jax.er.find_ambipolar_er` returns the same root as the legacy
  Fortran-parity Brent solver (``problems/ambipolar.py``, captured before its
  deletion and hard-coded here);
- warm starts / GCROT recycling reduce the total Krylov iteration count on a
  Fokker-Planck (tier-2) Er scan;
- ion / electron / unstable classification from the sign of ``dJr/dEr``;
- the differentiable :func:`sfincs_jax.er.ambipolar_er` gradient matches a
  central finite difference (implicit function theorem, not FD roots).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

# The legacy Fortran-parity Brent root of the two-species PAS deck below,
# captured from problems/ambipolar.brent_ambipolar_root (deleted in this slice)
# driven by er.radial_current on the identical physics.
LEGACY_BRENT_ROOT_ER = -0.4065766608975321


def _pas_deck(er: float = 0.0, *, collision_operator: int = 1,
              n_theta: int = 7, n_zeta: int = 7, n_xi: int = 8, n_x: int = 3) -> str:
    """A tiny non-axisymmetric two-species deck with an ambipolar root.

    Helical ripple (epsilon_h) makes the field non-intrinsically-ambipolar;
    ``inputRadialCoordinate=3`` selects the Er (coordinate 4) electric-field
    knob that ``ambipolarSolver.F90`` drives.
    """
    return f"""&general
/
&geometryParameters
  geometryScheme = 1
  inputRadialCoordinate = 3
  rN_wish = 0.3
  B0OverBBar = 1.0
  GHat = 1.0
  IHat = 0.0
  iota = 1.31
  epsilon_t = 0.1
  epsilon_h = 0.05
  helicity_l = 2
  helicity_n = 5
  psiAHat = 0.045
  aHat = 0.1
/
&speciesParameters
  Zs = 1 -1
  mHats = 1.0 0.000545509
  nHats = 1.0 1.0
  THats = 1.0 1.0
  dNHatdrHats = -0.5 -0.5
  dTHatdrHats = -1.0 -1.0
/
&physicsParameters
  Delta = 4.5694d-3
  alpha = 1.0
  nu_n = 8.4774d-3
  Er = {er}
  collisionOperator = {collision_operator}
  includeXDotTerm = .false.
  includeElectricFieldTermInXiDot = .false.
  useDKESExBDrift = .true.
  includePhi1 = .false.
/
&resolutionParameters
  Ntheta = {n_theta}
  Nzeta = {n_zeta}
  Nxi = {n_xi}
  NL = 4
  Nx = {n_x}
  solverTolerance = 1d-10
/
&otherNumericalParameters
  Nxi_for_x_option = 0
/
"""


def _write(tmp_path: Path, text: str, name: str = "input.namelist") -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


# ---------------------------------------------------------------------------
# 1. radial_current == a direct run_profile particle-flux computation
# ---------------------------------------------------------------------------


def test_radial_current_matches_run_profile(tmp_path: Path) -> None:
    from sfincs_jax import er as er_mod
    from sfincs_jax.run import run_profile

    prob = er_mod.prepare(_write(tmp_path, _pas_deck()), er_bracket=(-3.0, 1.0))

    for er_val in (-2.0, -0.5, 0.7):
        run = run_profile(_write(tmp_path, _pas_deck(er_val)), solve_method="auto", emit=None)
        gamma_rp = np.asarray(run.moments["particleFlux_vm_psiHat"], dtype=np.float64)
        z_s = np.asarray(run.operator.z_s, dtype=np.float64)
        jr_rp = float(np.dot(z_s, gamma_rp))

        j_r, gamma, state = er_mod.radial_current(prob, er_val)
        gamma = np.asarray(gamma, dtype=np.float64)

        np.testing.assert_allclose(gamma, gamma_rp, rtol=0.0, atol=1e-12)
        assert abs(float(j_r) - jr_rp) < 1e-12
        assert state.result.converged


# ---------------------------------------------------------------------------
# 2. find_ambipolar_er == legacy Fortran-parity Brent root
# ---------------------------------------------------------------------------


def test_find_ambipolar_er_matches_legacy_brent(tmp_path: Path) -> None:
    from sfincs_jax import er as er_mod

    result = er_mod.find_ambipolar_er(
        _write(tmp_path, _pas_deck()),
        er_bracket=(-3.0, 1.0),
        er_initial=0.0,
        max_iter=20,
        current_tol=1e-10,
        all_roots=False,
        emit=None,
    )
    assert result.converged
    assert result.er is not None
    assert abs(result.er - LEGACY_BRENT_ROOT_ER) < 1e-6
    # J_r is driven to (near) zero at the reported root.
    assert abs(result.radial_current) < 1e-9
    assert result.per_species_flux is not None
    assert result.per_species_flux.shape == (2,)


def test_brent_expands_bracket_and_finds_analytic_root() -> None:
    """The Fortran option-2 zbrent (bracket expansion + NR update) on a cubic.

    ``J(Er) = (Er + 0.4)(Er^2 + 1)`` has the single real root ``Er = -0.4``.
    The initial bracket [0, 1] does not contain it, exercising the
    ``ambipolarSolver.F90`` sign-change expansion loop.
    """
    from sfincs_jax.er import _brent

    def eval_jr(er: float, _stage: str) -> float:
        return (er + 0.4) * (er * er + 1.0)

    root, converged, status, _msg = _brent(
        eval_jr, er_min=0.0, er_max=1.0, er_initial=0.5,
        max_iter=80, current_tol=1e-12, max_expansions=50, emit=None,
    )
    assert converged and status == "converged"
    assert abs(root - (-0.4)) < 1e-6


# ---------------------------------------------------------------------------
# 3. warm starts / recycling reduce total Krylov iterations (tier-2 FP)
# ---------------------------------------------------------------------------


def test_warm_start_reduces_solver_iterations(tmp_path: Path) -> None:
    from sfincs_jax import er as er_mod

    # Fokker-Planck collisions route the auto policy to the tier-2 recycled
    # GCROT solver, where warm starts and recycling pay off.
    deck = _pas_deck(collision_operator=0, n_theta=5, n_zeta=5, n_xi=16, n_x=4)
    prob = er_mod.prepare(_write(tmp_path, deck), er_bracket=(-3.0, 1.0))
    er_seq = list(np.linspace(-0.6, -0.35, 5))

    def scan(warm: bool) -> int:
        total = 0
        state = None
        for er_val in er_seq:
            x0 = state.x if (warm and state is not None) else None
            recycle = state.recycle if (warm and state is not None) else None
            _j, _g, state = er_mod.radial_current(
                prob, float(er_val), x0=x0, recycle=recycle, solve_method="auto", tol=1e-9
            )
            assert state.result.converged
            total += int(state.result.iterations or 0)
        return total

    cold = scan(warm=False)
    warm = scan(warm=True)
    assert cold > 0
    assert warm < cold


# ---------------------------------------------------------------------------
# 4. ion / electron / unstable classification
# ---------------------------------------------------------------------------


def test_root_classification_ion(tmp_path: Path) -> None:
    from sfincs_jax import er as er_mod

    result = er_mod.find_ambipolar_er(
        _write(tmp_path, _pas_deck()), er_bracket=(-3.0, 1.0), all_roots=True, emit=None
    )
    # A single root at Er < 0 with dJr/dEr > 0 is the stable ion root.
    assert result.root_type == "ion"
    assert result.er is not None and result.er < 0.0
    assert len(result.roots) == 1
    assert result.roots[0].root_type == "ion"
    assert result.roots[0].slope > 0.0


def test_classify_unit_logic() -> None:
    from sfincs_jax.er import _classify

    assert _classify(-0.4, 1.0) == "ion"        # stable, Er < 0
    assert _classify(0.4, 1.0) == "electron"     # stable, Er > 0
    assert _classify(0.1, -1.0) == "unstable"    # dJr/dEr < 0 (middle branch)
    assert _classify(-0.1, -1.0) == "unstable"


# ---------------------------------------------------------------------------
# 5. differentiable ambipolar_er: jax.grad vs central finite difference
# ---------------------------------------------------------------------------


def test_ambipolar_er_grad_matches_finite_difference(tmp_path: Path) -> None:
    import jax

    jax.config.update("jax_enable_x64", True)

    from sfincs_jax import er as er_mod

    deck = _pas_deck()
    prob = er_mod.prepare(_write(tmp_path, deck), er_bracket=(-3.0, 1.0))

    # Seed the differentiable root near the true root (selects that branch).
    found = er_mod.find_ambipolar_er(
        _write(tmp_path, deck), er_bracket=(-3.0, 1.0), all_roots=False, emit=None
    )
    root = float(found.er)

    base_op = prob.operator
    base_dn = base_op.dn_hat_dpsi_hat

    def er_of_theta(theta):
        # theta scales both species' density gradient (drives the flux/root).
        op_theta = replace(base_op, dn_hat_dpsi_hat=base_dn * theta)
        return er_mod.ambipolar_er(
            op_theta, er0=root, dphi_per_er=prob.dphi_per_er, z_s=prob.z_s
        )

    # The differentiable dense solve reproduces the ambipolar root (the flat J_r
    # near the root makes the exact value tolerance-sensitive, hence the loose
    # bound; the implicit-function-theorem gradient below is the real check).
    assert abs(float(er_of_theta(1.0)) - root) < 1e-3

    grad = float(jax.grad(er_of_theta)(1.0))
    h = 1e-3
    fd = (float(er_of_theta(1.0 + h)) - float(er_of_theta(1.0 - h))) / (2.0 * h)

    assert np.isfinite(grad) and abs(fd) > 1e-6
    np.testing.assert_allclose(grad, fd, rtol=1e-4)
