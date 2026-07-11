"""Element-wise parity of the canonical tangential magnetic-drift term vs Fortran v3.

The tangential (poloidal+toroidal) magnetic drifts (``magneticDriftScheme=1``)
are consolidated into :class:`sfincs_jax.drift_kinetic.KineticOperator`
(``KineticOperator._magnetic_drifts``).  These tests isolate that term from the
canonical operator — the difference of ``apply_f`` with and without the drifts —
and compare it, element by element, against the frozen Fortran ``whichMatrix=1``
PETSc matrix for a tiny Boozer (geometryScheme 11) fixture.

Because ``Er=0`` and the collision frequency is zero in this fixture, the only
matrix entries coupling Legendre modes ``L`` to ``L±2`` come from the magnetic
drift, and the only ``ΔL=0`` f-block entries do too, so the filtered slices below
are unambiguously the magnetic-drift contribution:

- ``|ΔL|=2`` off-diagonal in theta  -> the d/dtheta drift term,
- ``|ΔL|=2`` off-diagonal in zeta   -> the d/dzeta drift term,
- ``|ΔL|=2`` diagonal in (theta,zeta) -> the diagonal parts of d/dtheta, d/dzeta
  and the non-standard d/dxi drift term.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from sfincs_jax.discretization.v3 import V3Indexing
from sfincs_jax.drift_kinetic import KineticOperator
from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.validation.fortran import read_petsc_mat_aij

_REF = Path(__file__).parent / "ref"
_INPUT = _REF / "magdrift_1species_tiny.input.namelist"
_MAT = _REF / "magdrift_1species_tiny.whichMatrix_1.petscbin"


@lru_cache(maxsize=1)
def _canonical() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(M_md, index_components, rect_index)`` for the canonical fixture.

    ``M_md`` is the isolated magnetic-drift f-block matrix (``apply_f`` with the
    drifts minus without), materialized in the rectangular ``(s,x,L,theta,zeta)``
    row-major layout.  ``index_components`` is the ``(5, n_f)`` array of
    ``(s,ix,L,itheta,izeta)`` for each packed Fortran dof, and ``rect_index`` maps
    each packed dof to its rectangular flat index.
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("SFINCS_JAX_EQUILIBRIA_DIRS", str(_REF))
        nml = read_sfincs_input(_INPUT)
        op = KineticOperator.from_namelist(nml)
    assert op.with_magnetic_drifts

    op_no = replace(op, with_magnetic_drifts=False)
    eye = jnp.eye(op.f_size, dtype=jnp.float64)

    def _mat(o: KineticOperator) -> np.ndarray:
        flat = lambda v: o.apply_f(v.reshape(o.f_shape)).reshape(-1)  # noqa: E731
        return np.asarray(jax.vmap(flat)(eye)).T  # columns = apply(e_j)

    m_md = _mat(op) - _mat(op_no)

    indexing = V3Indexing(
        n_species=op.n_species,
        n_x=op.n_x,
        n_theta=op.n_theta,
        n_zeta=op.n_zeta,
        n_xi_max=op.n_xi,
        n_xi_for_x=np.asarray(op.n_xi_for_x, dtype=int),
    )
    inv = indexing.build_inverse_f_map()
    comps = np.asarray(inv, dtype=int).T  # (5, n_f): s, ix, L, itheta, izeta
    s, ix, ell, it, iz = comps
    rect = (((s * op.n_x + ix) * op.n_xi + ell) * op.n_theta + it) * op.n_zeta + iz
    return m_md, comps, rect


def _fortran_fblock() -> np.ndarray:
    a = read_petsc_mat_aij(_MAT)
    return csr_matrix((a.data, a.col_ind, a.row_ptr), shape=a.shape).toarray()


def _assert_slice(keep_mask: np.ndarray, *, atol: float, require_nonzero: bool = True) -> None:
    """Assert the canonical magnetic drift matches Fortran on the kept (row,col) pairs."""
    m_md, _comps, rect = _canonical()
    a = _fortran_fblock()
    r_i, c_i = np.where(keep_mask)
    a_vals = a[r_i, c_i]  # Fortran is in packed order
    m_vals = m_md[rect[r_i], rect[c_i]]  # canonical is in rectangular order
    if require_nonzero:
        assert np.max(np.abs(a_vals)) > 0.0, "expected nonzero magnetic-drift entries"
    np.testing.assert_allclose(m_vals, a_vals, rtol=0.0, atol=atol)


def _pair_masks() -> dict[str, np.ndarray]:
    _m, comps, _rect = _canonical()
    s, ix, ell, it, iz = comps
    same_six = (s[:, None] == s[None, :]) & (ix[:, None] == ix[None, :])
    dl = np.abs(ell[:, None] - ell[None, :])
    dl2 = (dl == 2) & same_six
    same_t = it[:, None] == it[None, :]
    same_z = iz[:, None] == iz[None, :]
    return {
        # |ΔL|=2, same zeta, off-diagonal theta -> d/dtheta drift term
        "offdiag_theta": dl2 & same_z & ~same_t,
        # |ΔL|=2, same theta, off-diagonal zeta -> d/dzeta drift term
        "offdiag_zeta": dl2 & same_t & ~same_z,
        # |ΔL|=2, same theta and zeta -> diagonal parts + d/dxi drift term
        "diag_theta_zeta": dl2 & same_t & same_z,
        # ΔL=0 f-block entries -> magnetic-drift diagonal-in-L (Er=0, nu=0 deck)
        "diag_l": (dl == 0) & same_six,
    }


def test_magnetic_drift_theta_offdiag2_offdiag_theta_matches_fortran() -> None:
    """Parity for magnetic-drift d/dtheta term: |ΔL|=2, off-diagonal in theta."""
    _assert_slice(_pair_masks()["offdiag_theta"], atol=3e-12)


def test_magnetic_drift_zeta_offdiag2_offdiag_zeta_matches_fortran() -> None:
    """Parity for magnetic-drift d/dzeta term: |ΔL|=2, off-diagonal in zeta."""
    _assert_slice(_pair_masks()["offdiag_zeta"], atol=3e-12)


def test_magnetic_drift_diag_theta_zeta_offdiag2_matches_fortran() -> None:
    """Parity for the diagonal-in-(theta,zeta) part of the |ΔL|=2 contributions.

    This slice includes the diagonal-in-theta part of the d/dtheta term, the
    diagonal-in-zeta part of the d/dzeta term, and the (theta,zeta-diagonal)
    non-standard d/dxi term.
    """
    _assert_slice(_pair_masks()["diag_theta_zeta"], atol=3e-12)


def test_magnetic_drift_diagonal_in_l_matches_fortran() -> None:
    """Parity for the ΔL=0 magnetic-drift entries (diagonal-in-L drift coefficients)."""
    _assert_slice(_pair_masks()["diag_l"], atol=3e-12)
