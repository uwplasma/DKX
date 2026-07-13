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

from sfincs_jax.collisions import (
    ImprovedSugamaV3Operator,
    apply_improved_sugama_v3,
    apply_pitch_angle_scattering_v3,
    make_improved_sugama_v3_operator,
    make_pitch_angle_scattering_v3_operator,
)
from sfincs_jax.phase_space import make_speed_grid, speed_grid_diff_matrices

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
    from sfincs_jax.collisions import _improved_sugama_pair_kernels

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
    from sfincs_jax.collisions import _mask_xi

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
    from sfincs_jax.collisions import apply_improved_sugama_v3_jit

    np.testing.assert_allclose(
        np.asarray(apply_improved_sugama_v3_jit(rebuilt, f)),
        np.asarray(apply_improved_sugama_v3(op, f)),
        rtol=0.0, atol=0.0,
    )
