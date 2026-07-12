"""Element-wise Fortran referees for the canonical drift-kinetic operator.

Consolidates the per-term legacy-operator parity suites (collisionless
streaming/mirror, ExB, Er xDot/xiDot, PAS and Fokker-Planck collisions,
full-system matvec/RHS/residual, monoenergetic and RHSMode=2 transport
matrices) into direct comparisons of :class:`sfincs_jax.drift_kinetic.
KineticOperator` against frozen Fortran v3 PETSc binaries:

- ``<base>.whichMatrix_3.petscbin`` (the RHSMode=1 solver matrix) with the
  frozen ``stateVector``: ``op.apply(x_ref)`` must equal the sparse matvec,
  and ``op.apply(x_ref) - op.rhs()`` must equal the frozen ``residual``.
- ``<base>.whichMatrix_1.petscbin`` (the RHSMode=2/3 and Er/ExB term decks):
  ``op.apply(v)`` must equal the sparse matvec on a seeded random state,
  pinning every term coefficient element-wise (streaming, mirror, ExB,
  Er xDot/xiDot, collisions, sources/constraints).

Magnetic-drift terms are pinned separately (and more finely) in
``tests/test_magnetic_drifts_parity.py``; Phi1 fixtures in ``tests/test_phi1.py``.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from sfincs_jax.drift_kinetic import kinetic_operator_from_namelist
from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.validation.fortran import read_petsc_mat_aij, read_petsc_vec

REF = Path(__file__).parent / "ref"

# RHSMode=1 fixtures with the full solver matrix + solved state (+ residual).
FULL_SYSTEM_BASES = (
    "pas_1species_PAS_noEr_tiny",
    "pas_1species_PAS_noEr_tiny_scheme1",
    "pas_1species_PAS_noEr_tiny_scheme5",
    "pas_1species_PAS_noEr_tiny_scheme11",
    "pas_1species_PAS_noEr_tiny_scheme12",
    "quick_2species_FPCollisions_noEr",
)

# Decks whose whichMatrix_1 pins individual physics terms (Er xDot / Er xiDot /
# ExB theta) and the RHSMode=2/3 transport-matrix operators.
WHICHMATRIX1_BASES = (
    "er_xdot_1species_tiny",
    "er_xidot_1species_tiny",
    "exb_theta_1species_tiny",
    "monoenergetic_PAS_tiny_scheme1",
    "monoenergetic_PAS_tiny_scheme11",
    "monoenergetic_PAS_tiny_scheme5_filtered",
    "transportMatrix_PAS_tiny_rhsMode2_scheme2",
    "transportMatrix_PAS_tiny_rhsMode2_scheme11",
    "transportMatrix_PAS_tiny_rhsMode2_scheme5_filtered",
)


def _csr(path: Path) -> csr_matrix:
    a = read_petsc_mat_aij(path)
    return csr_matrix((a.data, a.col_ind, a.row_ptr), shape=a.shape)


def _op(base: str):
    return kinetic_operator_from_namelist(read_sfincs_input(REF / f"{base}.input.namelist"))


def _assert_close(new: np.ndarray, ref: np.ndarray, *, rtol: float = 1e-12) -> None:
    scale = max(1.0, float(np.max(np.abs(ref))) if ref.size else 1.0)
    np.testing.assert_allclose(np.asarray(new), ref, rtol=rtol, atol=rtol * scale)


@pytest.mark.parametrize("base", FULL_SYSTEM_BASES)
def test_apply_matches_fortran_solver_matrix_at_solved_state(base: str) -> None:
    op = _op(base)
    a = _csr(REF / f"{base}.whichMatrix_3.petscbin")
    x_ref = read_petsc_vec(REF / f"{base}.stateVector.petscbin").values
    assert x_ref.size == op.total_size
    y = np.asarray(op.apply(jnp.asarray(x_ref)))
    _assert_close(y, a.dot(x_ref))


@pytest.mark.parametrize("base", FULL_SYSTEM_BASES)
def test_residual_matches_fortran_residual_at_solved_state(base: str) -> None:
    """r(x_ref) = A x_ref - b must equal the frozen Fortran residual (pins the RHS too)."""
    op = _op(base)
    x_ref = read_petsc_vec(REF / f"{base}.stateVector.petscbin").values
    r_ref = read_petsc_vec(REF / f"{base}.residual.petscbin").values
    r = np.asarray(op.apply(jnp.asarray(x_ref)) - op.rhs())
    scale = max(1.0, float(np.max(np.abs(read_petsc_vec(REF / f"{base}.stateVector.petscbin").values))))
    np.testing.assert_allclose(r, r_ref, rtol=0.0, atol=1e-11 * scale)


def test_apply_matches_fortran_solver_matrix_small_case() -> None:
    """The larger PAS deck (no residual fixture): matvec parity at the solved state."""
    base = "pas_1species_PAS_noEr_small"
    op = _op(base)
    a = _csr(REF / f"{base}.whichMatrix_3.petscbin")
    x_ref = read_petsc_vec(REF / f"{base}.stateVector.petscbin").values
    y = np.asarray(op.apply(jnp.asarray(x_ref)))
    _assert_close(y, a.dot(x_ref))


@pytest.mark.parametrize("base", WHICHMATRIX1_BASES)
def test_apply_matches_fortran_whichmatrix1_random_state(base: str) -> None:
    op = _op(base)
    a = _csr(REF / f"{base}.whichMatrix_1.petscbin")
    assert a.shape == (op.total_size, op.total_size)
    rng = np.random.default_rng(1234)
    v = rng.normal(size=(op.total_size,))
    y = np.asarray(op.apply(jnp.asarray(v)))
    _assert_close(y, a.dot(v))


@pytest.mark.parametrize(
    ("base", "n_rhs"),
    (
        ("monoenergetic_PAS_tiny_scheme1", 2),
        ("monoenergetic_PAS_tiny_scheme11", 2),
        ("monoenergetic_PAS_tiny_scheme5_filtered", 2),
        ("transportMatrix_PAS_tiny_rhsMode2_scheme2", 3),
        ("transportMatrix_PAS_tiny_rhsMode2_scheme11", 3),
        ("transportMatrix_PAS_tiny_rhsMode2_scheme5_filtered", 3),
    ),
)
def test_transport_rhs_columns_solve_to_fortran_state_vectors(base: str, n_rhs: int) -> None:
    """A x_ref(whichRHS) = rhs(whichRHS): the frozen v3 states satisfy the canonical system.

    The frozen states carry the Fortran Krylov solve's own convergence error,
    so the gate is a relative linear-system residual (1e-6 of the drive norm),
    not element-wise machine precision; matvec parity at machine precision is
    pinned separately above.
    """
    op = _op(base)
    a = _csr(REF / f"{base}.whichMatrix_1.petscbin")
    for which_rhs in range(1, n_rhs + 1):
        x_ref = read_petsc_vec(REF / f"{base}.whichRHS{which_rhs}.stateVector.petscbin").values
        b = np.asarray(op.rhs(which_rhs))
        ax = a.dot(x_ref)
        resid = float(np.linalg.norm(ax - b))
        assert resid <= 1e-6 * max(float(np.linalg.norm(b)), 1e-300), f"whichRHS={which_rhs}: {resid}"


@pytest.mark.parametrize(
    ("base", "n_rhs"),
    (
        ("monoenergetic_PAS_tiny_scheme1", 2),
        ("monoenergetic_PAS_tiny_scheme11", 2),
        ("monoenergetic_PAS_tiny_scheme5_filtered", 2),
        ("transportMatrix_PAS_tiny_rhsMode2_scheme2", 3),
        ("transportMatrix_PAS_tiny_rhsMode2_scheme11", 3),
        ("transportMatrix_PAS_tiny_rhsMode2_scheme5_filtered", 3),
    ),
)
def test_transport_matrix_assembly_from_frozen_fortran_states(base: str, n_rhs: int) -> None:
    """Canonical transport-matrix assembly at the FROZEN Fortran states.

    Inserting the recorded whichRHS state vectors isolates the flux-moment /
    transport-matrix assembly formulas (``diagnostics.F90``) from the linear
    solve, so the gate is much tighter than the re-solve referee in
    ``tests/test_run_transport.py``.
    """
    import h5py

    from sfincs_jax.moments import transport_matrix_from_state_vectors
    from sfincs_jax.writer import operator_containers

    op = _op(base)
    layout, vgrid, surface, species = operator_containers(op)
    states = np.stack(
        [
            read_petsc_vec(REF / f"{base}.whichRHS{w}.stateVector.petscbin").values
            for w in range(1, n_rhs + 1)
        ]
    )
    with h5py.File(REF / f"{base}.sfincsOutput.h5", "r") as f:
        tm_fortran = np.asarray(f["transportMatrix"][...], dtype=np.float64)
        g_hat = float(np.asarray(f["GHat"][...]))
        i_hat = float(np.asarray(f["IHat"][...]))
        iota = float(np.asarray(f["iota"][...]))
        b0_over_bbar = float(np.asarray(f["B0OverBBar"][...]))

    tm = np.asarray(
        transport_matrix_from_state_vectors(
            layout, vgrid, surface, species, jnp.asarray(states),
            rhs_mode=2 if n_rhs == 3 else 3, delta=op.delta, alpha=op.alpha,
            g_hat=g_hat, i_hat=i_hat, iota=iota, b0_over_bbar=b0_over_bbar,
        ),  # fmt: skip
        dtype=np.float64,
    )
    # Fortran stores the matrix column-major: golden reads back transposed.
    scale = float(np.max(np.abs(tm_fortran)))
    np.testing.assert_allclose(tm.T, tm_fortran, rtol=0.0, atol=5e-10 * max(1.0, scale))


def test_rhsmode1_moments_from_frozen_fortran_state_match_golden() -> None:
    """RHSMode=1 moment assembly at the FROZEN Fortran state vs the h5 golden.

    Inserting the recorded stateVector isolates the per-species moment
    integrals (``diagnostics.F90``) from the linear solve; the recorded
    ``sfincsOutput.h5`` referees the values much tighter than a re-solve.
    """
    import h5py

    from sfincs_jax.run import profile_moments_from_operator

    base = "pas_1species_PAS_noEr_tiny_scheme1"
    op = _op(base)
    x_ref = read_petsc_vec(REF / f"{base}.stateVector.petscbin").values
    table = profile_moments_from_operator(op, x_ref)

    with h5py.File(REF / f"{base}.sfincsOutput.h5", "r") as f:
        for key in (
            "particleFlux_vm_psiHat",
            "heatFlux_vm_psiHat",
            "momentumFlux_vm_psiHat",
            "FSABFlow",
            "FSABjHat",
            "FSADensityPerturbation",
            "FSAPressurePerturbation",
        ):
            golden = np.asarray(f[key][...], dtype=np.float64)
            if golden.ndim and golden.shape[-1] >= 1 and np.asarray(table[key]).ndim < golden.ndim:
                golden = golden[..., -1] if golden.ndim > np.asarray(table[key]).ndim else golden
            got = np.asarray(table[key], dtype=np.float64)
            golden = np.reshape(golden, got.shape)
            # Constrained-to-zero moments (FSADensityPerturbation etc.) sit at
            # summation roundoff (~1e-19); floor the gate at 1e-16 absolute.
            atol = max(1e-9 * float(np.max(np.abs(golden))), 1e-16)
            np.testing.assert_allclose(got, golden, rtol=0.0, atol=atol, err_msg=key)
