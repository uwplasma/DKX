"""Physics-based validation of the improved Sugama model collision operator.

The improved Sugama linearized model collision operator (research extension
beyond SFINCS v3, ``collisionOperator = 3``) is defined by RETAINING the
conservation laws that bare pitch-angle scattering violates: particle number,
parallel momentum, and kinetic energy.  There is no Fortran golden for this
operator, so it is validated purely by physics:

1. **Conservation (the defining property).**  Assembled on a tiny grid, the
   operator's parallel-momentum and kinetic-energy moments vanish to machine
   precision for arbitrary perturbations, and particle number is conserved per
   species; bare PAS does NOT conserve momentum on the same grid (contrast).
2. **Analytic limit.**  For a single species the operator annihilates the
   shifted Maxwellian (Galilean invariance of like-particle collisions --
   Helander & Sigmar, *Collisional Transport in Magnetized Plasmas*), exactly;
   and the like-species momentum restoration enhances the parallel
   conductivity over the pure-Lorentz value, converging to it as ``Z -> inf``
   (the Spitzer momentum-restoring factor).
3. **Differentiability.**  ``jax.grad`` of a flow moment through a linear solve
   that uses the operator agrees with a central finite difference.

Primary references:
  H. Sugama, S. Matsuoka, S. Satake, M. Nunami, and T.-H. Watanabe,
  Phys. Plasmas 26, 102108 (2019) (the improved linearized model operator);
  B. J. Frei, S. Ernst, and P. Ricci, Phys. Plasmas 29, 093902 (2022),
  arXiv:2202.06293 (the moment-approach numerical implementation).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dkx.collisions import (
    ImprovedSugamaV3Operator,
    apply_improved_sugama_v3,
    apply_pitch_angle_scattering_v3,
    make_improved_sugama_v3_operator,
    make_pitch_angle_scattering_v3_operator,
)
from dkx.phase_space import make_speed_grid, speed_grid_diff_matrices

_K = 0.0


def _build(z, m, n, t, *, n_x=8, n_xi=6, nu_n=1.0, krook=0.0):
    sg = make_speed_grid(n_x=n_x, k=_K)
    x = np.asarray(sg.x, dtype=np.float64)
    w = np.asarray(sg.dx_weights(_K), dtype=np.float64)
    ddx, d2dx2 = speed_grid_diff_matrices(x, k=_K)
    op = make_improved_sugama_v3_operator(
        x=x,
        x_weights=w,
        ddx=ddx,
        d2dx2=d2dx2,
        z_s=np.asarray(z, dtype=np.float64),
        m_hats=np.asarray(m, dtype=np.float64),
        n_hats=np.asarray(n, dtype=np.float64),
        t_hats=np.asarray(t, dtype=np.float64),
        nu_n=nu_n,
        krook=krook,
        n_xi=n_xi,
        n_xi_for_x=np.full((n_x,), n_xi, dtype=np.int32),
    )
    return op, x, w, np.asarray(t, dtype=np.float64), np.asarray(m, dtype=np.float64)


def _moment_weights(x, w):
    return w * x**2, w * x**3, w * x**4  # particle, momentum, kinetic energy


# ---------------------------------------------------------------------------
# 1. Conservation laws (the defining property).
# ---------------------------------------------------------------------------

_SPECIES_CASES = [
    ("single H", [1.0], [1.0], [1.0], [1.0]),
    ("e-i equal T", [1.0, 1.0], [1.0 / 1836.0, 1.0], [1.0, 1.0], [1.0, 1.0]),
    ("H-C6 unequal n,T", [1.0, 6.0], [1.0, 6.0], [0.6, 0.05], [1.0, 1.3]),
    ("unequal T only", [1.0, 1.0], [1.0, 2.0], [1.0, 0.5], [1.0, 3.0]),
]


@pytest.mark.parametrize("tag,z,m,n,t", _SPECIES_CASES)
def test_improved_sugama_conserves_momentum_energy_particles(tag, z, m, n, t) -> None:
    """Momentum and energy moments vanish (~machine precision); particles per species."""
    op, x, w, t_arr, m_arr = _build(z, m, n, t)
    s = len(z)
    n_x, n_xi = x.size, int(op.mat.shape[2])
    w_part, w_mom, w_energy = _moment_weights(x, w)
    p_mom = t_arr**2 / m_arr  # ~ m_a v_th,a^4 (parallel-momentum species prefactor)
    p_energy = t_arr**2.5 / m_arr**1.5  # ~ m_a v_th,a^5 (energy species prefactor)

    rng = np.random.default_rng(20240711)
    for _ in range(4):
        f = rng.standard_normal((s, n_x, n_xi, 1, 1))
        out = np.asarray(apply_improved_sugama_v3(op, jnp.asarray(f)))[:, :, :, 0, 0]

        # Total parallel momentum (L=1) -- relative to the operator's own scale.
        mom = sum(p_mom[a] * (w_mom @ out[a, :, 1]) for a in range(s))
        mom_scale = sum(p_mom[a] * (np.abs(w_mom) @ np.abs(out[a, :, 1])) for a in range(s)) + 1e-300
        assert abs(mom) / mom_scale < 1e-12

        # Total kinetic energy (L=0).
        energy = sum(p_energy[a] * (w_energy @ out[a, :, 0]) for a in range(s))
        energy_scale = sum(p_energy[a] * (np.abs(w_energy) @ np.abs(out[a, :, 0])) for a in range(s)) + 1e-300
        assert abs(energy) / energy_scale < 1e-12

        # Particle number is conserved per species (L=0).
        for a in range(s):
            part = w_part @ out[a, :, 0]
            part_scale = np.abs(w_part) @ np.abs(out[a, :, 0]) + 1e-300
            assert abs(part) / part_scale < 1e-12


def test_bare_pas_does_not_conserve_momentum_but_improved_sugama_does() -> None:
    """Contrast: on the SAME grid, PAS leaks parallel momentum; improved Sugama does not."""
    z, m, n, t = [1.0, 6.0], [1.0, 6.0], [0.6, 0.05], [1.0, 1.3]
    op, x, w, t_arr, m_arr = _build(z, m, n, t)
    s = len(z)
    n_x, n_xi = x.size, int(op.mat.shape[2])
    _, w_mom, _ = _moment_weights(x, w)
    p_mom = t_arr**2 / m_arr

    pas = make_pitch_angle_scattering_v3_operator(
        x=jnp.asarray(x),
        z_s=jnp.asarray(z, dtype=jnp.float64),
        m_hats=jnp.asarray(m, dtype=jnp.float64),
        n_hats=jnp.asarray(n, dtype=jnp.float64),
        t_hats=jnp.asarray(t, dtype=jnp.float64),
        nu_n=1.0,
        krook=0.0,
        n_xi_for_x=jnp.asarray(np.full(n_x, n_xi), dtype=jnp.int32),
        n_xi=n_xi,
    )

    rng = np.random.default_rng(7)
    f = rng.standard_normal((s, n_x, n_xi, 1, 1))
    out_pas = np.asarray(apply_pitch_angle_scattering_v3(pas, jnp.asarray(f)))[:, :, :, 0, 0]
    out_imp = np.asarray(apply_improved_sugama_v3(op, jnp.asarray(f)))[:, :, :, 0, 0]

    mom_pas = sum(p_mom[a] * (w_mom @ out_pas[a, :, 1]) for a in range(s))
    mom_imp = sum(p_mom[a] * (w_mom @ out_imp[a, :, 1]) for a in range(s))
    mom_scale = sum(p_mom[a] * (np.abs(w_mom) @ np.abs(out_pas[a, :, 1])) for a in range(s))

    assert abs(mom_pas) / mom_scale > 1e-2  # PAS is momentum-deficient by O(1)
    assert abs(mom_imp) / mom_scale < 1e-12  # improved Sugama restores it exactly


# ---------------------------------------------------------------------------
# 2. Analytic limits.
# ---------------------------------------------------------------------------

def test_single_species_annihilates_shifted_maxwellian() -> None:
    """Like-particle collisions do not damp a rigid parallel flow (Galilean invariance).

    A shifted Maxwellian has L=1 speed profile ``x exp(-x^2)``; the momentum-
    conserving operator must annihilate it exactly (Helander & Sigmar).
    """
    op, x, w, _, _ = _build([1.0], [1.0], [1.0], [1.0], n_x=12, n_xi=5)
    n_x, n_xi = x.size, int(op.mat.shape[2])
    f = np.zeros((1, n_x, n_xi, 1, 1))
    f[0, :, 1, 0, 0] = x * np.exp(-(x * x))
    out = np.asarray(apply_improved_sugama_v3(op, jnp.asarray(f)))[0, :, 1, 0, 0]
    scale = float(np.max(np.abs(op.mat[0, 0, 1])))
    assert np.max(np.abs(out)) < 1e-11 * scale


def _test_only_l1_block(z, m, n, t, *, n_x):
    """The L=1 test-particle block (pitch-angle + energy diffusion), no field term.

    Uses the same per-pair velocity kernels as the operator builder, assembled
    without the momentum-restoring field term (i.e. the ``-nu_n(CE + pitch)``
    part only, ``nu_n = 1``).
    """
    from dkx.collisions import _improved_sugama_pair_kernels

    sg = make_speed_grid(n_x=n_x, k=_K)
    x = np.asarray(sg.x, dtype=np.float64)
    ddx, d2dx2 = speed_grid_diff_matrices(x, k=_K)
    nu_d_ab, ce_ab = _improved_sugama_pair_kernels(
        x=x, ddx=ddx, d2dx2=d2dx2,
        z_s=np.asarray(z, float), m_hats=np.asarray(m, float),
        n_hats=np.asarray(n, float), t_hats=np.asarray(t, float),
    )
    s = len(z)
    blk = np.zeros((n_x, n_x))
    for ib in range(s):
        blk += -1.0 * ce_ab[0, ib]
        blk[range(n_x), range(n_x)] += 1.0 * nu_d_ab[0, ib]  # L=1 pitch factor = 1
    return blk


def _spitzer_enhancement(z_i, *, n_x=16):
    """Parallel conductivity enhancement from like-species momentum restoration.

    Electrons on a fixed ion background (a parallel-momentum sink): compare the
    flow driven by an ``E``-field source with the e-e momentum-restoring field
    term ON (the full operator's L=1 block) vs OFF (the test-particle-only
    block).  The ratio is the Spitzer momentum-restoring factor; it exceeds 1
    (restoration enhances conductivity) and ``-> 1`` as the ion charge grows
    (Lorentz limit, where the fixed-ion sink dominates the e-e restoration).
    """
    op, x, w, _, _ = _build([1.0], [1.0], [1.0], [1.0], n_x=n_x, n_xi=4)
    p = w * x**3
    src = x * np.exp(-(x * x))
    nud_ei = (3.0 * np.sqrt(np.pi) / 4.0) * z_i**2 / x**3  # unrestored fixed-ion sink

    c_full = np.asarray(op.mat[0, 0, 1]) + np.diag(nud_ei)  # test + e-e field + sink
    c_test = _test_only_l1_block([1.0], [1.0], [1.0], [1.0], n_x=n_x) + np.diag(nud_ei)

    flow_full = p @ np.linalg.solve(c_full, src)
    flow_test = p @ np.linalg.solve(c_test, src)
    return flow_full / flow_test


@pytest.mark.parametrize("z_i", [1.0, 4.0, 16.0])
def test_spitzer_momentum_restoring_factor(z_i) -> None:
    """Momentum restoration enhances conductivity (>1) and -> 1 in the Lorentz limit."""
    ratio = _spitzer_enhancement(z_i)
    assert ratio > 1.0  # e-e momentum restoration always enhances the conductivity
    if z_i >= 16.0:
        assert ratio < 1.02  # Lorentz limit: e-e negligible, enhancement -> 1


def test_spitzer_factor_is_monotone_and_bounded() -> None:
    """The Z=1 enhancement is in the physical range and decreases with Z."""
    r1 = _spitzer_enhancement(1.0)
    r4 = _spitzer_enhancement(4.0)
    r16 = _spitzer_enhancement(16.0)
    assert 1.3 < r1 < 2.1  # rank-1 model restoration (exact Spitzer-Harm ~1.96)
    assert r1 > r4 > r16 > 1.0


# ---------------------------------------------------------------------------
# 3. Differentiability.
# ---------------------------------------------------------------------------

def test_grad_of_flow_through_operator_matches_finite_difference() -> None:
    """jax.grad of a flow moment through a solve that uses the operator vs central FD."""
    op, x, w, _, _ = _build([1.0, 1.0], [1.0, 2.0], [1.0, 0.5], [1.0, 1.3], n_x=8, n_xi=5)
    p = jnp.asarray(w * x**3)
    src = jnp.asarray(x * np.exp(-(x * x)))
    c_ee = op.mat[0, 0, 1]  # electron L=1 block (jnp leaf -> differentiable)
    nud_ei = jnp.asarray((3.0 * np.sqrt(np.pi) / 4.0) * 2.0 / x**3)

    def flow(scale):
        # A differentiable collisionality scaling entering the operator block.
        a = scale * c_ee + jnp.diag(nud_ei)
        f = jnp.linalg.solve(a, src)
        return jnp.sum(p * f)

    scale0 = 1.3
    ad = float(jax.grad(flow)(scale0))
    eps = 1e-6
    fd = float((flow(scale0 + eps) - flow(scale0 - eps)) / (2.0 * eps))
    assert abs(ad - fd) <= 1e-5 * (abs(fd) + 1.0)


# ---------------------------------------------------------------------------
# 4. Structure / API guards.
# ---------------------------------------------------------------------------

def test_improved_sugama_apply_matches_dense_block_matvec_and_masks_l() -> None:
    """apply reproduces the dense per-L block matvec and honours the Nxi_for_x mask."""
    from dkx.collisions import _mask_xi

    mat = np.zeros((1, 1, 3, 2, 2), dtype=np.float64)
    mat[0, 0, 0] = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    mat[0, 0, 1] = np.asarray([[0.5, -1.0], [2.0, 0.25]])
    mat[0, 0, 2] = np.asarray([[2.0, 0.0], [0.0, -1.0]])
    op = ImprovedSugamaV3Operator(
        mat=jnp.asarray(mat),
        n_xi_for_x=jnp.asarray([3, 1], dtype=jnp.int32),
        mask_xi=_mask_xi(jnp.asarray([3, 1], dtype=jnp.int32), 3),
    )
    f = np.arange(1 * 2 * 3 * 2 * 1, dtype=np.float64).reshape(1, 2, 3, 2, 1)
    out = np.asarray(apply_improved_sugama_v3(op, jnp.asarray(f)))
    expected = np.zeros_like(f)
    for ell in range(3):
        for itheta in range(2):
            expected[0, :, ell, itheta, 0] = mat[0, 0, ell] @ f[0, :, ell, itheta, 0]
    expected[0, 1, 1:, :, :] = 0.0  # x-node 1 keeps only L<1
    np.testing.assert_allclose(out, expected, rtol=0.0, atol=0.0)


def test_improved_sugama_apply_rejects_bad_shapes() -> None:
    op, *_ = _build([1.0], [1.0], [1.0], [1.0], n_x=4, n_xi=3)
    with pytest.raises(ValueError, match="f must have shape"):
        apply_improved_sugama_v3(op, jnp.ones((1, 4, 3, 2), dtype=jnp.float64))


def test_pytree_roundtrip_and_jit() -> None:
    op, *_ = _build([1.0, 1.0], [1.0, 2.0], [1.0, 0.5], [1.0, 1.3], n_x=6, n_xi=4)
    leaves, treedef = jax.tree_util.tree_flatten(op)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    f = jnp.asarray(np.cos(np.arange(2 * 6 * 4).reshape(2, 6, 4, 1, 1) / 5.0))
    from dkx.collisions import apply_improved_sugama_v3_jit

    np.testing.assert_allclose(
        np.asarray(apply_improved_sugama_v3_jit(rebuilt, f)),
        np.asarray(apply_improved_sugama_v3(op, f)),
        rtol=0.0, atol=0.0,
    )


# ---------------------------------------------------------------------------
# test-particle / field-particle split
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("z", "m", "n", "t"),
    (
        ([1.0], [1.0], [1.0], [1.0]),
        ([1.0, -1.0], [1.0, 5.446e-4], [1.0, 1.2], [1.0, 0.8]),
        ([1.0, 6.0, -1.0], [1.0, 12.0, 5.446e-4], [1.0, 0.05, 1.3], [1.0, 1.0, 0.9]),
    ),
)
def test_split_reassembles_the_operator_exactly(z, m, n, t) -> None:
    """``mat`` is the split, not an approximation of it.

    The whole value of the decomposition is that nothing is dropped: a
    preconditioner may relax on ``test`` alone, but the operator it is
    preconditioning is still exactly ``test`` plus the field back-reaction.
    """
    op, *_ = _build(z, m, n, t, n_x=6, n_xi=5)
    n_species = len(z)
    reassembled = np.einsum(
        "ab,aLij->abLij", np.eye(n_species), np.asarray(op.test)
    ) + np.einsum("aiLc,bjLc->abLij", np.asarray(op.columns), np.asarray(op.extraction))
    np.testing.assert_allclose(
        np.asarray(op.mat), reassembled, rtol=0.0, atol=0.0
    )


def test_field_part_is_low_rank_and_confined_to_the_conserved_moments() -> None:
    """Rank <= ``S (K + 2)`` at L=1 (momentum plus the flow correction), ``2S`` at L=0, nothing above.

    Only the particle (L=0), energy (L=0) and parallel-momentum (L=1) moments
    have a back-reaction, so every higher Legendre block is pure
    test-particle: species-diagonal and local in the pitch index.  That is the
    structural fact a velocity-space block smoother exploits.
    """
    z, m, n, t = [1.0, -1.0], [1.0, 5.446e-4], [1.0, 1.2], [1.0, 0.8]
    op, *_ = _build(z, m, n, t, n_x=6, n_xi=5)
    n_species, n_x, n_xi, _ = np.asarray(op.columns).shape[0], 6, 5, None

    columns = np.asarray(op.columns)
    extraction = np.asarray(op.extraction)
    for ell in range(n_xi):
        field = np.einsum("aic,bjc->aibj", columns[:, :, ell], extraction[:, :, ell])
        flat = field.reshape(n_species * n_x, n_species * n_x)
        singular = np.linalg.svd(flat, compute_uv=False)
        rank = int((singular > 1e-9 * max(singular[0], 1e-300)).sum())
        if ell == 0:
            assert 0 < rank <= n_species * n_species + n_species
        elif ell == 1:
            assert 0 < rank <= n_species * 7  # momentum term plus K = 5 flow moments per species
        else:
            assert rank == 0
            np.testing.assert_allclose(flat, 0.0, rtol=0.0, atol=0.0)


def test_test_particle_part_alone_does_not_conserve_momentum() -> None:
    """The split is meaningful: conservation lives entirely in the field part.

    Dropping the low-rank piece is exactly what a plain relaxation would do if
    it were handed the whole operator, and it costs parallel-momentum
    conservation -- the property the bootstrap current depends on.
    """
    z, m, n, t = [1.0, -1.0], [1.0, 5.446e-4], [1.0, 1.2], [1.0, 0.8]
    op, x, w, t_hats, m_hats = _build(z, m, n, t, n_x=8, n_xi=6)

    rng = np.random.default_rng(0)
    f = rng.standard_normal((len(z), 8))  # one L=1 profile per species
    weight = w * (x**3) * (t_hats**2 / m_hats)[:, None]

    full = np.einsum("abij,bj->ai", np.asarray(op.mat)[:, :, 1], f)
    test_only = np.einsum("aij,aj->ai", np.asarray(op.test)[:, 1], f)

    # The moment is a signed sum of O(1e6) contributions, so the statement is
    # about cancellation *relative to the terms being cancelled*, not about an
    # absolute size: the summation order alone moves an absolute residual by
    # orders of magnitude between architectures.
    scale = float(np.abs(weight * full).sum())
    assert abs(float((weight * full).sum())) / scale < 1e-14
    assert abs(float((weight * test_only).sum())) / scale > 1e-3


# ---------------------------------------------------------------------------
# 4. Unequal temperatures: l = 1 against the Landau operator, l = 0 against the exchange rate.
# ---------------------------------------------------------------------------


def _pair_kwargs(t_e: float, m_e: float, n_x: int) -> dict:
    from dkx.phase_space import make_speed_grid, speed_grid_diff_matrices

    grid = make_speed_grid(n_x=n_x, k=0.0)
    x = np.asarray(grid.x, dtype=np.float64)
    ddx, d2dx2 = speed_grid_diff_matrices(x, k=0.0)
    return dict(
        x=x, x_weights=np.asarray(grid.dx_weights(0.0), dtype=np.float64), ddx=ddx, d2dx2=d2dx2,
        z_s=np.array([-1.0, 1.0]), m_hats=np.array([m_e, 1.0]), n_hats=np.ones(2),
        t_hats=np.array([t_e, 1.0]), nu_n=1.0, krook=0.0, n_xi=3, n_xi_for_x=np.full(n_x, 3, np.int32),
    )


@pytest.mark.parametrize(("t_e", "m_e", "tol"), [(1.0, 1 / 1836, 2e-3), (10.0, 1 / 1836, 2e-3), (10.0, 1.0, 1.5e-2)])
def test_l1_block_matches_landau_flow_moments_at_unequal_temperature(t_e, m_e, tol) -> None:
    """Flow and heat-flow friction (Laguerre orders 0..3) against the full linearized Landau operator."""
    from scipy.special import eval_genlaguerre

    from dkx.collisions import make_fokker_planck_v3_operator

    kw = _pair_kwargs(t_e, m_e, 10)
    x, w = kw["x"], kw["x_weights"]
    fp = np.asarray(make_fokker_planck_v3_operator(x_grid_k=0.0, nl=3, rosenbluth_method="analytic", **kw).mat)[:, :, 1]
    shapes = [x * np.exp(-x * x) * eval_genlaguerre(j, 1.5, x * x) for j in range(4)]
    funcs = [w * x**3 * eval_genlaguerre(j, 1.5, x * x) for j in range(4)]

    def moments(mat):
        return np.array([[[[funcs[j] @ mat[a, b] @ shapes[k] for k in range(4)] for j in range(4)] for b in range(2)] for a in range(2)])

    ref = moments(fp)
    err = np.abs(moments(np.asarray(make_improved_sugama_v3_operator(**kw).mat)[:, :, 1]) - ref).max() / np.abs(ref).max()
    assert err < tol, err
    if t_e != 1.0:  # without the correction the unequal-temperature friction is far off
        bare = np.abs(moments(np.asarray(make_improved_sugama_v3_operator(flow_order=0, **kw).mat)[:, :, 1]) - ref).max()
        assert bare / np.abs(ref).max() > 1.0


@pytest.mark.parametrize(("t_e", "m_e"), [(10.0, 1.0), (3.0, 0.2)])
def test_energy_exchange_rate_matches_sugama_2019_eq_a1(t_e, m_e) -> None:
    """d(energy_a)/dT_a and d(energy_a)/dT_b of the l = 0 block against the derivatives of Eq. A1."""
    kw = _pair_kwargs(t_e, m_e, 24)
    x, w = kw["x"], kw["x_weights"]
    mat = np.asarray(make_improved_sugama_v3_operator(**kw).mat)[:, :, 0]
    h_t, e_en = (x * x - 1.5) * np.exp(-x * x), w * x**4

    def rate(ta, tb, ma, mb):  # Eq. A1 up to a constant: alpha^3 (Ta - Tb) / ((1 + alpha^2)^(3/2) Ta^(3/2))
        a2 = (ta / ma) / (tb / mb)
        return a2**1.5 * (ta - tb) / (1 + a2) ** 1.5 / ta**1.5

    h = 1e-6
    for a, b, (ta, tb, ma, mb) in ((0, 1, (t_e, 1.0, m_e, 1.0)), (1, 0, (1.0, t_e, 1.0, m_e))):
        d_a = ta * (rate(ta * (1 + h), tb, ma, mb) - rate(ta * (1 - h), tb, ma, mb)) / (2 * h * ta)
        d_b = tb * (rate(ta, tb * (1 + h), ma, mb) - rate(ta, tb * (1 - h), ma, mb)) / (2 * h * tb)
        scale = (np.sqrt(ta / ma) / np.sqrt(tb / mb)) ** 3  # species-dependent amplitude of the x-space f
        ratio = (e_en @ mat[a, b] @ h_t) / (e_en @ mat[a, a] @ h_t) * scale
        assert abs(ratio / (d_b / d_a) - 1.0) < 1e-3, (a, ratio, d_b / d_a)


# ---------------------------------------------------------------------------
# Adjointness (Sugama et al. 2019, Eqs. 23 and 29), including unequal temperatures.
# ---------------------------------------------------------------------------


def _asymmetry(mat, kw, ell: int) -> float:
    """Worst relative antisymmetric part of ``sum_ab T_a <g_a, C_ab f_b> / f_aM`` over smooth trial shapes.

    The pairing weights the speed quadrature by ``x^2 exp(+x^2)`` (the inverse Maxwellian) and each species
    by ``T_a vT_a^3 / n_a``; trial shapes are ``exp(-x^2) x^ell`` times a random quadratic in ``x^2``.
    """
    x, w, t, m, n = kw["x"], kw["x_weights"], kw["t_hats"], kw["m_hats"], kw["n_hats"]
    pref = t * (t / m) ** 1.5 / n
    rng = np.random.default_rng(0)

    def pair(g, f):
        return sum(
            pref[a] * np.sum(w * x**2 * np.exp(x * x) * g[a] * (mat[a, b] @ f[b]))
            for a in range(2)
            for b in range(2)
        )

    def shapes():
        return np.array([np.exp(-x * x) * x**ell * np.polyval(rng.normal(size=3), x * x) for _ in range(2)])

    worst = 0.0
    for _ in range(4):
        g, f = shapes(), shapes()
        worst = max(worst, abs(pair(g, f) - pair(f, g)) / np.sqrt(abs(pair(g, g) * pair(f, f))))
    return worst


@pytest.mark.parametrize(("t_e", "m_e"), [(1.0, 1.0), (1.0, 1 / 1836), (10.0, 1 / 1836), (10.0, 1.0), (3.0, 0.2)])
def test_l1_block_is_as_adjoint_as_the_landau_operator_at_unequal_temperature(t_e, m_e) -> None:
    """The L = 1 block, whose field part is the published one, has the Landau operator's own asymmetry.

    The full linearized Landau operator is exactly adjoint only for equal temperatures; for T_a != T_b both it
    and the improved Sugama operator carry the same finite defect (0.4 % to 41 % here, set by the
    temperature and mass ratios). The improved operator neither adds to it nor removes it.
    """
    from dkx.collisions import make_fokker_planck_v3_operator

    kw = _pair_kwargs(t_e, m_e, 16)
    kw["n_xi"] = 2
    kw["n_xi_for_x"] = np.full(16, 2, np.int32)
    sugama = np.asarray(make_improved_sugama_v3_operator(**kw).mat)[:, :, 1]
    landau = np.asarray(make_fokker_planck_v3_operator(x_grid_k=0.0, nl=2, rosenbluth_method="analytic", **kw).mat)[:, :, 1]
    a_sugama, a_landau = _asymmetry(sugama, kw, 1), _asymmetry(landau, kw, 1)
    assert abs(a_sugama - a_landau) < 1e-3 * max(a_landau, 1e-3), (a_sugama, a_landau)
    if t_e == 1.0 and m_e == 1.0:
        assert a_sugama < 1e-9


def test_l0_block_is_adjoint_for_like_species() -> None:
    """Open: Eq. 29 for the L = 0 field part (see docs/physics/collisions.md)."""
    kw = _pair_kwargs(1.0, 1.0, 16)
    kw["n_xi"] = 2
    kw["n_xi_for_x"] = np.full(16, 2, np.int32)
    assert _asymmetry(np.asarray(make_improved_sugama_v3_operator(**kw).mat)[:, :, 0], kw, 0) < 1e-6
