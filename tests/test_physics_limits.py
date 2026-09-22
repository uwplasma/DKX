"""Analytic limits of the full linearized Fokker-Planck operator.

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
"""

from __future__ import annotations

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
