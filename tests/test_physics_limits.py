"""Analytic limits: Spitzer-Harm conductivity and Onsager symmetry.

Spitzer-Harm conductivity
=========================

With the ions an immobile Maxwellian background of charge ``Z`` and density
``n_e / Z``, the electron response to a parallel electric field solves, in
the ``L = 1`` Legendre channel,

    (C_ee + C_ei) f_1 = x e^{-x^2}

and the current is the moment ``sum_x w_x x^3 f_1``.  Dropping the
electron-electron collisions (``C_ee``) leaves the Lorentz gas.  The ratio of
the two currents is Spitzer and Harm's ``gamma_E(Z)`` (Phys. Rev. 89, 977,
1953, Table III): 0.5816, 0.6833, 0.7849 and 0.9225 for ``Z = 1, 2, 4, 16``.

The ions are made immobile by a mass of ``1e8`` electron masses, and the
Lorentz gas by an electron density of ``1e-14`` in a second build, which
removes ``C_ee`` while keeping ``C_ei`` unchanged.

Measured (float64, ``k = 0`` speed grid): 0.58193, 0.68577, 0.78533, 0.92338
at ``Nx = 8``, unchanged to five digits at ``Nx = 12, 16, 24``.  The largest
deviation from the four-digit table is 0.36 % (``Z = 2``); the assertion is
0.5 %, which the tabulated values' own precision allows.

Onsager symmetry of the thermal transport matrix
================================================

``RHSMode = 2`` returns the 3x3 matrix ``L`` mapping the density-gradient,
temperature-gradient and inductive-field drives to the particle flux, heat
flux and parallel flow.  SFINCS normalizes it so that Onsager symmetry reads
``L_ij = L_ji`` at ``E_r = 0`` for a self-adjoint collision operator
(Landreman, Smith, Mollen & Helander, Phys. Plasmas 21, 042503, 2014;
Helander & Sigmar, *Collisional Transport in Magnetized Plasmas*, 2002).

The discretization does not preserve the continuum self-adjointness exactly,
so the symmetry is a statement about the converged answer: the asymmetry
``|L_ij - L_ji| / max(|L_ij|, |L_ji|)`` must fall under refinement.  The two
radial-flux entries ``L_12, L_21`` share the kinetic operator and a drive of
the same parity and agree to roundoff at every resolution.

Measured (float64, ``nu_n = 0.1``, ``(Ntheta, Nzeta, Nxi, Nx)``):

====================  ====================  =================  =================
case                  rungs                 ``L_13`` vs ``L_31``  ``L_23`` vs ``L_32``
====================  ====================  =================  =================
PAS tokamak           (5,1,6,4)->(9,1,12,6)   2.2e-2 -> 2.1e-3   1.4e-2 -> 1.3e-3
PAS stellarator       (5,11,6,4)->(9,15,12,6) 3.0e-2 -> 1.1e-3   1.6e-2 -> 4.3e-3
full FP tokamak       (9,1,12,6)->(13,1,20,8) 4.5e-4 -> 8.6e-5   2.7e-3 -> 6.9e-5
====================  ====================  =================  =================

``L_12 = L_21`` to <= 1.2e-13 on both PAS cases.  With full Fokker-Planck
collisions and one species in a tokamak, ``L_11``, ``L_12`` and ``L_21``
themselves vanish: like-particle collisions conserve momentum, so the radial
particle flux is intrinsically ambipolar and zero (Helander & Sigmar, ch. 8).
Measured ``|L_11| / |L_22|`` = 1.6e-8 at ``(9,1,12,6)`` and 8.4e-10 at
``(13,1,20,8)``, against 0.14 with pitch-angle scattering, which does not
conserve momentum.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dkx.collisions import make_fokker_planck_v3_operator
from dkx.phase_space import make_speed_grid, speed_grid_diff_matrices

#: Spitzer & Harm (1953), Table III: gamma_E(Z).
SPITZER_HARM_GAMMA_E = {1.0: 0.5816, 2.0: 0.6833, 4.0: 0.7849, 16.0: 0.9225}

ION_MASS = 1.0e8
LORENTZ_ELECTRON_DENSITY = 1.0e-14


def _electron_l1_block(z_ion: float, n_electron: float, n_x: int) -> tuple[np.ndarray, ...]:
    k = 0.0
    grid = make_speed_grid(n_x=n_x, k=k)
    x = np.asarray(grid.x, dtype=np.float64)
    weights = np.asarray(grid.dx_weights(k), dtype=np.float64)
    ddx, d2dx2 = speed_grid_diff_matrices(x, k=k)
    op = make_fokker_planck_v3_operator(
        x=x,
        x_weights=weights,
        ddx=ddx,
        d2dx2=d2dx2,
        x_grid_k=k,
        z_s=np.asarray([-1.0, z_ion]),
        m_hats=np.asarray([1.0, ION_MASS]),
        n_hats=np.asarray([n_electron, 1.0 / z_ion]),
        t_hats=np.asarray([1.0, 1.0]),
        nu_n=1.0,
        krook=0.0,
        n_xi=3,
        nl=3,
        n_xi_for_x=np.full((n_x,), 3, dtype=np.int32),
    )
    return x, weights, np.asarray(op.mat)[0, 0, 1]


def _current(z_ion: float, n_electron: float, n_x: int) -> float:
    x, weights, block = _electron_l1_block(z_ion, n_electron, n_x)
    response = np.linalg.solve(block, x * np.exp(-x * x))
    return float(np.sum(weights * x**3 * response))


@pytest.mark.parametrize("z_ion", sorted(SPITZER_HARM_GAMMA_E))
def test_full_fokker_planck_reproduces_the_spitzer_harm_conductivity(z_ion: float) -> None:
    n_x = 8
    ratio = _current(z_ion, 1.0, n_x) / _current(z_ion, LORENTZ_ELECTRON_DENSITY, n_x)
    expected = SPITZER_HARM_GAMMA_E[z_ion]
    assert abs(ratio - expected) / expected < 5e-3, (z_ion, ratio, expected)


# =============================================================================
# Onsager symmetry of the RHSMode = 2 thermal transport matrix
# =============================================================================

TRANSPORT_DECK = """\
&general
  RHSMode = 2
/
&geometryParameters
  geometryScheme = 1
  epsilon_t = -0.07d+0
  epsilon_h = {epsilon_h}d+0
  iota = 0.4542d+0
  GHat = 3.7481d+0
  IHat = 0d+0
  helicity_l = 2
  helicity_n = 10
  B0OverBBar = 1d+0
/
&speciesParameters
  Zs = 1
  mHats = 1
  nHats = 1.0d+0
  THats = 1.0d+0
/
&physicsParameters
  Delta = 4.5694d-3
  alpha = 1.0d+0
  nu_n = 0.1d+0
  Er = 0.0d+0
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
  solverTolerance = 1d-12
/
&otherNumericalParameters
  Nxi_for_x_option = 0
/
&preconditionerOptions
/
&export_f
  export_full_f = .false.
  export_delta_f = .false.
/
"""

#: (label, collisionOperator, epsilon_h, coarse rung, fine rung)
ONSAGER_CASES = [
    ("pas-tokamak", 1, 0.0, (5, 1, 6, 4), (9, 1, 12, 6)),
    ("pas-stellarator", 1, 0.05, (5, 11, 6, 4), (9, 15, 12, 6)),
    ("fokker-planck-tokamak", 0, 0.0, (9, 1, 12, 6), (13, 1, 20, 8)),
]


def _transport_matrix(
    tmp_path: Path, collision_operator: int, epsilon_h: float, rung: tuple[int, int, int, int]
) -> np.ndarray:
    from dkx.run import run_transport_matrix

    n_theta, n_zeta, n_xi, n_x = rung
    deck = tmp_path / f"onsager_{collision_operator}_{n_theta}_{n_zeta}_{n_xi}_{n_x}.namelist"
    deck.write_text(
        TRANSPORT_DECK.format(
            collision_operator=collision_operator,
            epsilon_h=epsilon_h,
            n_theta=n_theta,
            n_zeta=n_zeta,
            n_xi=n_xi,
            n_x=n_x,
        )
    )
    return np.asarray(run_transport_matrix(deck, emit=None).transport_matrix, dtype=float)


def _asymmetry(matrix: np.ndarray, i: int, j: int) -> float:
    return abs(matrix[i, j] - matrix[j, i]) / max(abs(matrix[i, j]), abs(matrix[j, i]))


@pytest.mark.parametrize(
    ("label", "collision_operator", "epsilon_h", "coarse", "fine"),
    ONSAGER_CASES,
    ids=[case[0] for case in ONSAGER_CASES],
)
def test_thermal_transport_matrix_becomes_onsager_symmetric_under_refinement(
    tmp_path: Path,
    label: str,
    collision_operator: int,
    epsilon_h: float,
    coarse: tuple[int, int, int, int],
    fine: tuple[int, int, int, int],
) -> None:
    """``L_13 = L_31`` and ``L_23 = L_32`` in the limit; each gap falls >= 3x.

    Asserted: the fine-rung asymmetry is at most a third of the coarse one
    and below 5e-3 (measured 2.7x to 39x reductions, fine values 6.9e-5 to
    4.3e-3; see the module table).  On the pitch-angle-scattering cases the
    radial block is symmetric to roundoff, asserted at 1e-10.
    """
    low = _transport_matrix(tmp_path, collision_operator, epsilon_h, coarse)
    high = _transport_matrix(tmp_path, collision_operator, epsilon_h, fine)
    for i, j in ((0, 2), (1, 2)):
        before, after = _asymmetry(low, i, j), _asymmetry(high, i, j)
        assert after < before / 3.0, (label, i, j, before, after)
        assert after < 5e-3, (label, i, j, after)
    if collision_operator == 1:
        assert _asymmetry(low, 0, 1) < 1e-10, label
        assert _asymmetry(high, 0, 1) < 1e-10, label


def test_like_particle_fokker_planck_collisions_give_no_tokamak_particle_flux(
    tmp_path: Path,
) -> None:
    """Momentum conservation makes the single-species tokamak particle flux vanish.

    With full Fokker-Planck collisions ``|L_11| / |L_22|`` is 1.6e-8 at
    ``(9,1,12,6)``; asserted below 1e-6.  Pitch-angle scattering on the same
    deck gives 0.14, asserted above 1e-2, which is what makes the first
    assertion a statement about momentum conservation and not about the deck.
    """
    rung = (9, 1, 12, 6)
    full = _transport_matrix(tmp_path, 0, 0.0, rung)
    pas = _transport_matrix(tmp_path, 1, 0.0, rung)
    assert abs(full[0, 0]) / abs(full[1, 1]) < 1e-6, full
    assert abs(pas[0, 0]) / abs(pas[1, 1]) > 1e-2, pas
