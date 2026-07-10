"""Referee tests for Phase 3.2 collisions-track: collisions == physics/collisions.py.

Gates:
1. Exact equivalence (1e-15; observed bitwise) with the old
   ``physics/collisions.py`` operators across (n_x, xGrid_k=0, NL, 1/2/3
   species) for both pitch-angle scattering and the full Fokker-Planck
   operator, including the strict-parity assembly branch.
2. Discrete conservation properties the old operator satisfies: the
   linearized Fokker-Planck blocks annihilate the Maxwellian null vectors
   (particle number, parallel momentum incl. interspecies exchange, energy).
3. Physics gates mirrored from ``test_collision_physics_gates.py`` (Lorentz
   L=0 null space, dissipativity, l(l+1)/2 eigenvalue ratios, Chandrasekhar
   small-x limit).
4. Golden check vs Fortran: the PAS diagonal matvec against the frozen v3
   PETSc matrix (mirrors ``test_pas_collision_operator_parity.py``).
5. ``CollisionMatrices``: matrix-free ``apply`` == per-operator applies, and
   ``blocks_for_l`` materializes exactly the operator that ``apply`` implements.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from sfincs_jax import collisions as c2
from sfincs_jax.constants import SQRT_PI_V3
from sfincs_jax.phase_space import make_speed_grid, n_xi_for_x_ramp, speed_grid_diff_matrices
from sfincs_jax.physics.collisions import (
    apply_fokker_planck_v3,
    apply_pitch_angle_scattering_v3,
    make_fokker_planck_v3_operator,
    make_pitch_angle_scattering_v3_operator,
)
from sfincs_jax.species import SpeciesSet

# Species parameter sets with 1, 2, and 3 species (as in test_species_constants.py).
SPECIES_CASES = {
    "1sp": dict(z=[1.0], m=[1.0], n=[1.0], t=[1.0]),
    "2sp": dict(z=[1.0, 6.0], m=[1.0, 6.0], n=[0.6, 0.009], t=[0.5, 0.8]),
    "3sp": dict(
        z=[1.0, 6.0, -1.0],
        m=[1.0, 6.0, 5.446170214e-4],
        n=[0.6, 0.009, 0.654],
        t=[0.5, 0.8, 0.7],
    ),
}

NU_N = 8.330e-3
X_GRID_K = 0.0


def _species_set(case: dict) -> SpeciesSet:
    s = len(case["z"])
    return SpeciesSet(
        z=jnp.asarray(case["z"], dtype=jnp.float64),
        m_hat=jnp.asarray(case["m"], dtype=jnp.float64),
        n_hat=jnp.asarray(case["n"], dtype=jnp.float64),
        t_hat=jnp.asarray(case["t"], dtype=jnp.float64),
        dn_hat_dpsi_hat=jnp.zeros((s,), dtype=jnp.float64),
        dt_hat_dpsi_hat=jnp.zeros((s,), dtype=jnp.float64),
    )


def _speed_grid_arrays(n_x: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sg = make_speed_grid(n_x=n_x, k=X_GRID_K)
    x = np.asarray(sg.x, dtype=np.float64)
    x_weights = np.asarray(sg.dx_weights(X_GRID_K), dtype=np.float64)
    ddx, d2dx2 = speed_grid_diff_matrices(x, k=X_GRID_K)
    return x, x_weights, ddx, d2dx2


def _random_f(rng, n_species: int, n_x: int, n_xi: int) -> np.ndarray:
    return rng.standard_normal((n_species, n_x, n_xi, 2, 1))


@pytest.mark.parametrize("case_name", sorted(SPECIES_CASES))
@pytest.mark.parametrize("nl", [2, 4])
@pytest.mark.parametrize("n_x", [4, 5, 8])
def test_pas_matches_old_operator_exactly(n_x: int, nl: int, case_name: str) -> None:
    case = SPECIES_CASES[case_name]
    species = _species_set(case)
    x, _, _, _ = _speed_grid_arrays(n_x)
    n_xi = max(nl, 5)
    n_xi_for_x = n_xi_for_x_ramp(x=x, n_xi=n_xi, n_l=nl, option=1)

    old = make_pitch_angle_scattering_v3_operator(
        x=jnp.asarray(x),
        z_s=species.z,
        m_hats=species.m_hat,
        n_hats=species.n_hat,
        t_hats=species.t_hat,
        nu_n=NU_N,
        krook=0.0,
        n_xi_for_x=jnp.asarray(n_xi_for_x, dtype=jnp.int32),
        n_xi=n_xi,
    )
    new = c2.make_pitch_angle_scattering(
        species=species,
        x=jnp.asarray(x),
        nu_n=NU_N,
        krook=0.0,
        n_xi_for_x=jnp.asarray(n_xi_for_x, dtype=jnp.int32),
        n_xi=n_xi,
    )

    np.testing.assert_allclose(np.asarray(new.nu_d_hat), np.asarray(old.nu_d_hat), rtol=1e-15, atol=0.0)
    np.testing.assert_allclose(np.asarray(new.coef), np.asarray(old.coef), rtol=1e-15, atol=0.0)
    np.testing.assert_array_equal(np.asarray(new.mask_xi), np.asarray(old.mask_xi))

    f = _random_f(np.random.default_rng(3 * n_x + nl), len(case["z"]), n_x, n_xi)
    out_new = np.asarray(c2.apply_pitch_angle_scattering(new, jnp.asarray(f)))
    out_old = np.asarray(apply_pitch_angle_scattering_v3(old, jnp.asarray(f)))
    np.testing.assert_allclose(out_new, out_old, rtol=1e-15, atol=0.0)


@pytest.mark.parametrize("case_name", sorted(SPECIES_CASES))
@pytest.mark.parametrize("nl", [2, 4])
@pytest.mark.parametrize("n_x", [4, 5, 8])
def test_fokker_planck_matches_old_operator_exactly(n_x: int, nl: int, case_name: str) -> None:
    case = SPECIES_CASES[case_name]
    species = _species_set(case)
    x, x_weights, ddx, d2dx2 = _speed_grid_arrays(n_x)
    n_xi = max(nl, 5)
    n_xi_for_x = n_xi_for_x_ramp(x=x, n_xi=n_xi, n_l=nl, option=1)

    old = make_fokker_planck_v3_operator(
        x=x,
        x_weights=x_weights,
        ddx=ddx,
        d2dx2=d2dx2,
        x_grid_k=X_GRID_K,
        z_s=np.asarray(case["z"]),
        m_hats=np.asarray(case["m"]),
        n_hats=np.asarray(case["n"]),
        t_hats=np.asarray(case["t"]),
        nu_n=NU_N,
        krook=0.0,
        n_xi=n_xi,
        nl=nl,
        n_xi_for_x=n_xi_for_x,
    )
    new = c2.make_fokker_planck(
        x=x,
        x_weights=x_weights,
        ddx=ddx,
        d2dx2=d2dx2,
        x_grid_k=X_GRID_K,
        species=species,
        nu_n=NU_N,
        krook=0.0,
        n_xi=n_xi,
        nl=nl,
        n_xi_for_x=n_xi_for_x,
    )

    np.testing.assert_allclose(np.asarray(new.mat), np.asarray(old.mat), rtol=1e-15, atol=0.0)
    np.testing.assert_array_equal(np.asarray(new.mask_xi), np.asarray(old.mask_xi))

    f = _random_f(np.random.default_rng(7 * n_x + nl), len(case["z"]), n_x, n_xi)
    out_new = np.asarray(c2.apply_fokker_planck(new, jnp.asarray(f)))
    out_old = np.asarray(apply_fokker_planck_v3(old, jnp.asarray(f)))
    np.testing.assert_allclose(out_new, out_old, rtol=1e-15, atol=0.0)


def test_fokker_planck_strict_parity_branch_matches_old_operator_exactly() -> None:
    case = SPECIES_CASES["2sp"]
    species = _species_set(case)
    x, x_weights, ddx, d2dx2 = _speed_grid_arrays(5)
    n_xi, nl = 5, 2
    n_xi_for_x = n_xi_for_x_ramp(x=x, n_xi=n_xi, n_l=nl, option=1)

    common = dict(
        x=x, x_weights=x_weights, ddx=ddx, d2dx2=d2dx2, x_grid_k=X_GRID_K,
        nu_n=NU_N, krook=0.1, n_xi=n_xi, nl=nl, n_xi_for_x=n_xi_for_x, strict_parity=True,
    )
    old = make_fokker_planck_v3_operator(
        z_s=np.asarray(case["z"]), m_hats=np.asarray(case["m"]),
        n_hats=np.asarray(case["n"]), t_hats=np.asarray(case["t"]), **common,
    )
    new = c2.make_fokker_planck(species=species, **common)
    np.testing.assert_allclose(np.asarray(new.mat), np.asarray(old.mat), rtol=1e-15, atol=0.0)


# ----------------------------------------------------------------------------
# Discrete conservation (what the operator actually satisfies at the
# collocation level; probed against the old implementation's matrices).
# ----------------------------------------------------------------------------


def _fp_blocks(species: SpeciesSet, n_x: int = 8, nl: int = 4, n_xi: int = 5) -> np.ndarray:
    x, x_weights, ddx, d2dx2 = _speed_grid_arrays(n_x)
    op = c2.make_fokker_planck(
        x=x,
        x_weights=x_weights,
        ddx=ddx,
        d2dx2=d2dx2,
        x_grid_k=X_GRID_K,
        species=species,
        nu_n=1.0,
        krook=0.0,
        n_xi=n_xi,
        nl=nl,
        n_xi_for_x=np.full((n_x,), n_xi, dtype=np.int32),
    )
    return np.asarray(op.mat)


def test_fokker_planck_annihilates_maxwellian_null_vectors_single_species() -> None:
    """C[F_M] = 0 at L=0 (density AND energy) and C[x F_M] = 0 at L=1 (momentum).

    The linearized self-collision operator annihilates the perturbations that
    correspond to shifting the background Maxwellian's density, temperature,
    and mean velocity; the discretization preserves this to machine precision.
    """
    species = _species_set(SPECIES_CASES["1sp"])
    mat = _fp_blocks(species)
    x = np.asarray(make_speed_grid(n_x=8, k=X_GRID_K).x)
    f_m = np.exp(-(x * x))
    scale = float(np.max(np.abs(mat[0, 0, :2])))

    assert np.max(np.abs(mat[0, 0, 0] @ f_m)) <= 1e-15 * scale  # particle number
    assert np.max(np.abs(mat[0, 0, 0] @ ((x * x - 1.5) * f_m))) <= 1e-15 * scale  # energy
    assert np.max(np.abs(mat[0, 0, 1] @ (x * f_m))) <= 1e-15 * scale  # momentum
    # L=2 has no conservation law: the same Maxwellian-weighted vector is NOT null.
    assert np.max(np.abs(mat[0, 0, 2] @ (x * x * f_m))) > 1e-6 * scale


def test_fokker_planck_interspecies_conservation_equal_temperature() -> None:
    """Cross-species null vectors: per-species density at L=0 and a common flow at L=1.

    For equal temperatures, C_ab[F_Ma, F_Mb] = 0, so per-species density
    perturbations (nHat_a e^{-x^2}) are annihilated at L=0.  A common mean
    velocity u gives f1_a = u (m_a v / T) F_Ma, i.e. collocation values
    proportional to nHat_a mHat_a^2 x e^{-x^2}; interspecies momentum exchange
    must cancel it at L=1.  Both hold to machine precision discretely.
    """
    species = SpeciesSet(
        z=jnp.asarray([1.0, 6.0]),
        m_hat=jnp.asarray([1.0, 6.0]),
        n_hat=jnp.asarray([0.6, 0.009]),
        t_hat=jnp.asarray([1.0, 1.0]),
        dn_hat_dpsi_hat=jnp.zeros((2,)),
        dt_hat_dpsi_hat=jnp.zeros((2,)),
    )
    mat = _fp_blocks(species)
    x = np.asarray(make_speed_grid(n_x=8, k=X_GRID_K).x)
    f_m = np.exp(-(x * x))
    n_hat = np.asarray(species.n_hat)
    m_hat = np.asarray(species.m_hat)
    scale = float(np.max(np.abs(mat[:, :, :2])))

    density = n_hat[:, None] * f_m[None, :]
    r0 = np.einsum("abij,bj->ai", mat[:, :, 0, :, :], density)
    assert np.max(np.abs(r0)) <= 1e-15 * scale

    flow = (n_hat * m_hat**2)[:, None] * (x * f_m)[None, :]
    r1 = np.einsum("abij,bj->ai", mat[:, :, 1, :, :], flow)
    assert np.max(np.abs(r1)) <= 1e-15 * scale


# ----------------------------------------------------------------------------
# Physics gates mirrored from test_collision_physics_gates.py
# ----------------------------------------------------------------------------


def _tiny_pas() -> c2.PitchAngleScattering:
    return c2.make_pitch_angle_scattering(
        species=_species_set(SPECIES_CASES["1sp"]),
        x=jnp.asarray([0.35, 0.9, 1.7], dtype=jnp.float64),
        nu_n=0.7,
        krook=0.0,
        n_xi_for_x=jnp.asarray([4, 3, 2], dtype=jnp.int32),
        n_xi=5,
    )


def test_pas_l0_is_null_and_inactive_legendre_slots_are_masked() -> None:
    op = _tiny_pas()
    f = np.ones((1, 3, 5, 2, 2), dtype=np.float64)
    out = np.asarray(c2.apply_pitch_angle_scattering(op, jnp.asarray(f)))
    np.testing.assert_allclose(out[:, :, 0, :, :], 0.0, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(out[:, 0, 4, :, :], 0.0, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(out[:, 1, 3:, :, :], 0.0, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(out[:, 2, 2:, :, :], 0.0, atol=0.0, rtol=0.0)
    assert np.all(out[:, :, 1, :, :] > 0.0)


def test_pas_anisotropic_modes_are_dissipative() -> None:
    op = _tiny_pas()
    f = np.arange(1, 1 + 1 * 3 * 5 * 2 * 2, dtype=np.float64).reshape(1, 3, 5, 2, 2) / 11.0
    f[:, :, 0, :, :] = 0.0
    out = np.asarray(c2.apply_pitch_angle_scattering(op, jnp.asarray(f)))
    coef = np.asarray(op.coef[0], dtype=np.float64)
    ell_mask = np.arange(5)[None, :] < np.asarray(op.n_xi_for_x)[:, None]
    expected = np.sum(coef[None, :, :, None, None] * ell_mask[None, :, :, None, None] * f * f)
    assert expected > 0.0
    np.testing.assert_allclose(np.sum(f * out), expected, rtol=0.0, atol=1.0e-12)


def test_pas_legendre_eigenvalues_follow_l_lplus1_over_two() -> None:
    coef = np.asarray(_tiny_pas().coef[0])
    for ix, n_l_active in enumerate([4, 3, 2]):
        base = coef[ix, 1]
        assert base > 0.0
        for ell in range(1, n_l_active):
            np.testing.assert_allclose(coef[ix, ell] / base, 0.5 * ell * (ell + 1.0), rtol=2e-15, atol=2e-15)


def test_chandrasekhar_matches_small_x_limit() -> None:
    x = jnp.asarray([1.0e-12, 1.0e-10, 1.0e-8], dtype=jnp.float64)
    psi = np.asarray(c2.chandrasekhar(x))
    np.testing.assert_allclose(psi / np.asarray(x), 2.0 / (3.0 * SQRT_PI_V3), rtol=1e-8, atol=1e-12)
    assert np.all(psi > 0.0)


def test_rosenbluth_potential_terms_shape_and_finiteness() -> None:
    species = _species_set(SPECIES_CASES["2sp"])
    sg = make_speed_grid(n_x=3, k=X_GRID_K)
    terms = c2.rosenbluth_potential_terms(
        x=np.asarray(sg.x),
        x_weights=np.asarray(sg.dx_weights(X_GRID_K)),
        x_grid_k=X_GRID_K,
        speed_grid=sg,
        species=species,
        nl=2,
    )
    assert terms.shape == (2, 2, 2, 3, 3)
    assert np.all(np.isfinite(terms))
    assert np.max(np.abs(terms)) > 0.0


# ----------------------------------------------------------------------------
# CollisionMatrices: one source of truth, multiple consumers
# ----------------------------------------------------------------------------


def _fp_operator_2sp() -> c2.FokkerPlanck:
    case = SPECIES_CASES["2sp"]
    x, x_weights, ddx, d2dx2 = _speed_grid_arrays(4)
    n_xi, nl = 5, 2
    return c2.make_fokker_planck(
        x=x,
        x_weights=x_weights,
        ddx=ddx,
        d2dx2=d2dx2,
        x_grid_k=X_GRID_K,
        species=_species_set(case),
        nu_n=NU_N,
        krook=0.0,
        n_xi=n_xi,
        nl=nl,
        n_xi_for_x=n_xi_for_x_ramp(x=x, n_xi=n_xi, n_l=nl, option=1),
    )


def test_collision_matrices_apply_matches_fokker_planck_apply_exactly() -> None:
    op = _fp_operator_2sp()
    cm = c2.collision_matrices_from_fokker_planck(op)
    f = _random_f(np.random.default_rng(11), 2, 4, 5)
    out_cm = np.asarray(cm.apply(jnp.asarray(f)))
    out_op = np.asarray(c2.apply_fokker_planck(op, jnp.asarray(f)))
    np.testing.assert_array_equal(out_cm, out_op)


def test_collision_matrices_from_pas_matches_pas_apply_exactly() -> None:
    op = _tiny_pas()
    cm = c2.collision_matrices_from_pitch_angle_scattering(op)
    f = _random_f(np.random.default_rng(13), 1, 3, 5)
    out_cm = np.asarray(cm.apply(jnp.asarray(f)))
    out_op = np.asarray(c2.apply_pitch_angle_scattering(op, jnp.asarray(f)))
    np.testing.assert_array_equal(out_cm, out_op)


def test_collision_matrices_blocks_for_l_materialize_the_apply_operator() -> None:
    """Structured-solver extractor == matrix-free apply, mode by mode (plan 2.2)."""
    cm = c2.collision_matrices_from_fokker_planck(_fp_operator_2sp())
    f = _random_f(np.random.default_rng(17), 2, 4, 5)
    out = np.asarray(cm.apply(jnp.asarray(f)))
    for ell in range(cm.n_xi):
        block = np.asarray(cm.blocks_for_l(ell))  # (S,S,X,X), rows masked
        expected = np.einsum("abij,bjtz->aitz", block, f[:, :, ell, :, :])
        np.testing.assert_allclose(out[:, :, ell, :, :], expected, rtol=0.0, atol=1e-13)


def test_collision_matrices_blocks_for_l_rejects_out_of_range_mode() -> None:
    cm = c2.collision_matrices_from_fokker_planck(_fp_operator_2sp())
    with pytest.raises(ValueError, match="ell must be in"):
        cm.blocks_for_l(cm.n_xi)
    with pytest.raises(ValueError, match="ell must be in"):
        cm.blocks_for_l(-1)


def test_apply_rejects_bad_shapes() -> None:
    op = _fp_operator_2sp()
    with pytest.raises(ValueError, match="f must have shape"):
        c2.apply_fokker_planck(op, jnp.ones((2, 4, 5, 1), dtype=jnp.float64))
    with pytest.raises(ValueError, match="op.mat has shape"):
        c2.apply_fokker_planck(op, jnp.ones((2, 4, 6, 1, 1), dtype=jnp.float64))
    cm = c2.collision_matrices_from_fokker_planck(op)
    with pytest.raises(ValueError, match="expected"):
        cm.apply(jnp.ones((2, 4, 6, 1, 1), dtype=jnp.float64))
    with pytest.raises(ValueError, match="f must have shape"):
        c2.apply_pitch_angle_scattering(_tiny_pas(), jnp.ones((1, 3, 5, 2), dtype=jnp.float64))


# ----------------------------------------------------------------------------
# Golden check vs Fortran (mirrors test_pas_collision_operator_parity.py)
# ----------------------------------------------------------------------------


def test_pas_diagonal_matvec_matches_fortran_matrix_golden() -> None:
    """PAS diagonal matvec vs the frozen Fortran v3 PETSc matrix (whichMatrix=3)."""
    from sfincs_jax.discretization.v3 import V3Indexing, grids_from_namelist
    from sfincs_jax.namelist import read_sfincs_input
    from sfincs_jax.validation.fortran import read_petsc_mat_aij, read_petsc_vec

    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "pas_1species_PAS_noEr_tiny.input.namelist")
    a = read_petsc_mat_aij(here / "ref" / "pas_1species_PAS_noEr_tiny.whichMatrix_3.petscbin")
    x_full = read_petsc_vec(here / "ref" / "pas_1species_PAS_noEr_tiny.stateVector.petscbin").values

    grids = grids_from_namelist(nml)
    species_params = nml.group("speciesParameters")
    phys = nml.group("physicsParameters")
    species = SpeciesSet(
        z=jnp.atleast_1d(jnp.asarray(species_params["ZS"], dtype=jnp.float64)),
        m_hat=jnp.atleast_1d(jnp.asarray(species_params["MHATS"], dtype=jnp.float64)),
        n_hat=jnp.atleast_1d(jnp.asarray(species_params["NHATS"], dtype=jnp.float64)),
        t_hat=jnp.atleast_1d(jnp.asarray(species_params["THATS"], dtype=jnp.float64)),
        dn_hat_dpsi_hat=jnp.zeros((1,), dtype=jnp.float64),
        dt_hat_dpsi_hat=jnp.zeros((1,), dtype=jnp.float64),
    )

    op = c2.make_pitch_angle_scattering(
        species=species,
        x=grids.x,
        nu_n=float(phys.get("NU_N", 0.0)),
        krook=float(phys.get("KROOK", 0.0)),
        n_xi_for_x=grids.n_xi_for_x,
        n_xi=int(grids.n_xi),
    )

    indexing = V3Indexing(
        n_species=species.n_species,
        n_x=int(grids.x.shape[0]),
        n_theta=int(grids.theta.shape[0]),
        n_zeta=int(grids.zeta.shape[0]),
        n_xi_max=int(grids.n_xi),
        n_xi_for_x=np.asarray(grids.n_xi_for_x, dtype=int),
    )
    inv = indexing.build_inverse_f_map()
    n_f = len(inv)

    f = np.zeros((indexing.n_species, indexing.n_x, indexing.n_xi_max, indexing.n_theta, indexing.n_zeta))
    for g, (s, ix, ell, it, iz) in enumerate(inv):
        f[s, ix, ell, it, iz] = x_full[g]

    y_jax = np.asarray(c2.apply_pitch_angle_scattering(op, jnp.asarray(f)))
    # The same result must come out of the consolidated container path.
    y_cm = np.asarray(c2.collision_matrices_from_pitch_angle_scattering(op).apply(jnp.asarray(f)))
    np.testing.assert_array_equal(y_cm, y_jax)

    # Fortran reference: diagonal-only contribution inside the f block (this
    # configuration has no other diagonal-in-L, diagonal-in-x terms).
    y_ref = np.zeros((n_f,), dtype=np.float64)
    for row in range(n_f):
        start, end = int(a.row_ptr[row]), int(a.row_ptr[row + 1])
        for c, v in zip(a.col_ind[start:end].tolist(), a.data[start:end].tolist()):
            if c == row:
                y_ref[row] = float(v) * float(x_full[row])
                break

    y_vec = np.zeros((n_f,), dtype=np.float64)
    for g, (s, ix, ell, it, iz) in enumerate(inv):
        y_vec[g] = y_jax[s, ix, ell, it, iz]

    np.testing.assert_allclose(y_vec, y_ref, rtol=0, atol=1e-12)
