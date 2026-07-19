"""Referee tests for ``dkx.solve`` — the plan-§2.3 three-tier auto-policy.

Tiny fixtures only (shared with ``tests/test_drift_kinetic.py``):

- tier 1 (analytic block-Thomas) must match the tier-3 dense solve to 1e-10
  on the monoenergetic (RHSMode=3) and PAS (RHSMode=1) fixtures, and match the
  recorded Fortran v3 ``stateVector`` fixtures on the RHSMode=3 transport
  columns (the referee formerly provided by the retired probing-based
  ``solvers/block_tridiagonal_transport`` POC);
- tier 2 (GCROT + coarse-operator preconditioner) must converge on the
  Fokker-Planck two-species fixture, match tier-3 dense to 1e-8, and need
  strictly fewer iterations than the unpreconditioned solve;
- the auto-policy must pick tier 1 for the PAS family and tier 2 for FP;
- recycling + warm start across an Er continuation must cut iterations;
- ``jax.grad`` through the differentiable tier-1 solve must match finite
  differences.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg as sla

from dkx.drift_kinetic import KineticOperator
from dkx.namelist import parse_sfincs_input_text, read_sfincs_input
from dkx.solve import (
    _resolve_subsystem_batch,
    build_coarse_preconditioner,
    materialize_dense,
    solve,
    tier1_available,
    tier1_full_band_bytes,
    tier1_peak_memory_bytes,
    tier1_truncated_peak_memory_bytes,
    tier1_truncated_subsystem_width,
)

REF = Path(__file__).parent / "ref"


def _load_op(name: str) -> KineticOperator:
    return KineticOperator.from_namelist(read_sfincs_input(REF / f"{name}.input.namelist"))


def _load_text(name: str) -> str:
    return (REF / f"{name}.input.namelist").read_text()


def _load_sugama_op() -> KineticOperator:
    """A ``collisionOperator=3`` (improved Sugama) deck from the FP fixture text.

    Reuses the two-species Fokker-Planck namelist with ``collisionOperator``
    switched to 3 (no new golden fixture): the improved Sugama operator has the
    same dense ``(species, x)`` block layout as Fokker-Planck and defaults to
    ``constraintScheme=1``.
    """
    txt = _load_text("quick_2species_FPCollisions_noEr").replace(
        "collisionOperator = 0", "collisionOperator = 3"
    )
    return KineticOperator.from_namelist(parse_sfincs_input_text(txt))


def _dense_solve(op: KineticOperator, rhs2d: np.ndarray) -> np.ndarray:
    return sla.solve(materialize_dense(op), rhs2d)


def _rel_err(x: np.ndarray, ref: np.ndarray) -> float:
    scale = max(1.0, float(np.max(np.abs(ref))))
    return float(np.max(np.abs(x - ref))) / scale


# ---------------------------------------------------------------------------
# Tier 1 == tier-3 dense on the structured-direct family
# ---------------------------------------------------------------------------


def test_tier1_matches_dense_monoenergetic_rhsmode3() -> None:
    op = _load_op("monoenergetic_PAS_tiny_scheme1")
    rhs = jnp.stack([op.rhs(1), op.rhs(2)], axis=1)  # both transport drives
    result = solve(op, rhs, method="block_tridiagonal")
    assert result.method == "block_tridiagonal"
    assert result.converged
    x_ref = _dense_solve(op, np.asarray(rhs))
    assert _rel_err(np.asarray(result.x), x_ref) < 1e-10


def test_tier1_matches_dense_pas_rhsmode1() -> None:
    op = _load_op("pas_1species_PAS_noEr_tiny_scheme1")
    rhs = op.rhs()
    result = solve(op, rhs, method="block_tridiagonal")
    assert result.converged
    x_ref = _dense_solve(op, np.asarray(rhs)[:, None])[:, 0]
    assert _rel_err(np.asarray(result.x), x_ref) < 1e-10
    # rhs was 1-D: the solution must come back 1-D.
    assert result.x.shape == rhs.shape


@pytest.mark.parametrize("base", ("monoenergetic_PAS_tiny_scheme1", "monoenergetic_PAS_tiny_scheme11"))
def test_tier1_matches_recorded_fortran_state_vectors_rhsmode3(base: str) -> None:
    """Tier 1 must reproduce the frozen v3 PETSc stateVector for both transport drives.

    This is the direct Fortran referee for the structured-direct tier on the
    RHSMode=3 transport columns; it replaces the equality test against the
    retired probing-based ``solvers/block_tridiagonal_transport`` POC.
    """
    from dkx.validation.fortran import read_petsc_vec

    op = _load_op(base)
    rhs = jnp.stack([op.rhs(1), op.rhs(2)], axis=1)
    result = solve(op, rhs, method="block_tridiagonal")
    assert result.converged
    x = np.asarray(result.x)
    for which_rhs in (1, 2):
        x_ref = read_petsc_vec(REF / f"{base}.whichRHS{which_rhs}.stateVector.petscbin").values
        assert _rel_err(x[:, which_rhs - 1], x_ref) < 1e-10, f"whichRHS={which_rhs}"


# ---------------------------------------------------------------------------
# Truncated tier 1: memory-driven routing, full parity, moments, gradients
# ---------------------------------------------------------------------------


def _vm_moments(op: KineticOperator, x_stack: np.ndarray) -> dict[str, np.ndarray]:
    """vm particle/heat fluxes + FSABFlow of solved states (moments module)."""
    from dkx.moments import (
        FluxSurface,
        SpeciesParams,
        StateLayout,
        VelocityGrid,
        vm_flux_moments_batch,
    )

    layout = StateLayout(
        n_species=op.n_species, n_x=op.n_x, n_xi=op.n_xi, n_theta=op.n_theta,
        n_zeta=op.n_zeta, include_phi1=False, constraint_scheme=op.constraint_scheme,
    )  # fmt: skip
    vgrid = VelocityGrid(x=op.x, x_weights=op.x_weights, n_xi_for_x=op.n_xi_for_x)
    surface = FluxSurface.from_operator(op)
    species = SpeciesParams.from_operator(op)
    m = vm_flux_moments_batch(
        layout, vgrid, surface, species, jnp.asarray(x_stack), delta=op.delta, alpha=op.alpha
    )
    return {
        "particleFlux": np.asarray(m.particle_flux_vm_psi_hat),
        "heatFlux": np.asarray(m.heat_flux_vm_psi_hat),
        "FSABFlow": np.asarray(m.fsab_flow),
    }


def test_tier1_memory_estimate_hand_computed() -> None:
    """The full-band byte formula must match a hand-computed value exactly.

    ``bytes = 3 * sum_x(Nxi_for_x) * n_species * (n_theta * n_zeta)**2 * 8`` —
    here n_theta=3, n_zeta=2 (m=6, m**2=36), n_xi=4, n_x=2, n_species=1 with
    uniform Nxi_for_x: 3 * (4+4) * 1 * 36 * 8 = 6912 bytes; the peak estimate
    is 2.5x that.  A ramped Nxi_for_x counts only the retained blocks.
    """
    fake = SimpleNamespace(
        n_theta=3, n_zeta=2, n_xi=4, n_x=2, n_species=1, n_xi_for_x=np.array([4, 4])
    )
    assert tier1_full_band_bytes(fake) == 3 * 4 * 1 * 2 * 36 * 8
    assert tier1_full_band_bytes(fake) == pytest.approx(6912.0)
    assert tier1_peak_memory_bytes(fake) == pytest.approx(2.5 * 6912.0)
    ramped = SimpleNamespace(
        n_theta=3, n_zeta=2, n_xi=4, n_x=2, n_species=1, n_xi_for_x=np.array([3, 4])
    )
    assert tier1_full_band_bytes(ramped) == 3 * (3 + 4) * 1 * 36 * 8


def test_auto_policy_selects_full_vs_truncated_by_budget() -> None:
    op = _load_op("pas_1species_PAS_noEr_tiny_scheme1")
    rhs = op.rhs()
    # Default (8 GB) budget dwarfs this tiny case: full factorization.
    r_full = solve(op, rhs, method="auto")
    assert r_full.method == "block_tridiagonal"
    # A deliberately tiny budget forces the truncated block-Thomas kernel.
    r_trunc = solve(op, rhs, method="auto", tier1_memory_budget_gb=1e-12)
    assert r_trunc.method == "block_tridiagonal_truncated"
    assert r_trunc.converged


def test_auto_policy_truncation_invalid_falls_through_to_tier2() -> None:
    # Tier-1-eligible operator, but the RHS carries Legendre support at l>=keep:
    # the truncated kernel would be inexact, so auto must fall back to tier 2.
    op = _load_op("pas_1species_PAS_noEr_tiny_scheme1")
    rhs = np.asarray(op.rhs())
    n_s, n_x, n_xi, n_t, n_z = op.f_shape
    f = rhs[: op.f_size].reshape(n_s, n_x, n_xi, n_t * n_z).copy()
    f[0, 0, 3, 0] = 1.0  # inject l=3 support (keep defaults to 3)
    rhs_bad = jnp.concatenate([jnp.asarray(f).reshape(-1), jnp.asarray(rhs[op.f_size :])])
    r = solve(op, rhs_bad, method="auto", tier1_memory_budget_gb=1e-12, tol=1e-9)
    assert r.method == "gcrot"
    assert r.converged


@pytest.mark.parametrize(
    "name,which",
    [
        ("pas_1species_PAS_noEr_tiny_scheme1", None),  # RHSMode=1 drive
        ("monoenergetic_PAS_tiny_scheme1", (1, 2)),  # RHSMode=3 transport drives
    ],
)
def test_truncated_matches_full_lowest_blocks_and_moments(name: str, which) -> None:
    op = _load_op(name)
    if which is None:
        rhs = op.rhs()
    else:
        rhs = jnp.stack([op.rhs(w) for w in which], axis=1)

    full = solve(op, rhs, method="block_tridiagonal")
    trunc = solve(op, rhs, method="block_tridiagonal_truncated")
    assert trunc.method == "block_tridiagonal_truncated"
    assert trunc.converged

    n_s, n_x, n_xi, n_t, n_z = op.f_shape
    n_tz = n_t * n_z
    xf, xt = np.asarray(full.x), np.asarray(trunc.x)
    ff = xf[: op.f_size].reshape(n_s, n_x, n_xi, n_tz, -1)
    ft = xt[: op.f_size].reshape(n_s, n_x, n_xi, n_tz, -1)
    # lowest-3 Legendre blocks are exact; blocks l>=3 are zero-padded.
    dl = np.linalg.norm(ft[:, :, :3] - ff[:, :, :3]) / np.linalg.norm(ff[:, :, :3])
    assert dl < 1e-10
    assert np.max(np.abs(ft[:, :, 3:])) == 0.0

    # Output moments (fluxes / FSABFlow) contract only l<=2, so they match.
    xf_stack = xf.T if xf.ndim == 2 else xf[None, :]
    xt_stack = xt.T if xt.ndim == 2 else xt[None, :]
    m_full, m_trunc = _vm_moments(op, xf_stack), _vm_moments(op, xt_stack)
    for key in ("particleFlux", "heatFlux", "FSABFlow"):
        a, b = m_full[key], m_trunc[key]
        rel = np.abs(a - b) / np.maximum(np.abs(a), 1e-300)
        assert rel.max() < 1e-10, f"{key}: {rel.max():.3e}"


def test_gradient_through_truncated_route_matches_finite_differences() -> None:
    op0 = _load_op("pas_1species_PAS_noEr_tiny_scheme1")

    def loss(t_hat_scalar: jnp.ndarray) -> jnp.ndarray:
        op = replace(op0, t_hat=jnp.reshape(t_hat_scalar, (1,)))
        # Tiny budget forces the truncated kernel; grad flows straight through
        # the block-Thomas sweeps (no full-operator IFT wrapper).
        result = solve(
            op, op.rhs(), method="auto", tier1_memory_budget_gb=1e-12, differentiable=True
        )
        return jnp.sum(result.x**2)

    t0 = float(op0.t_hat[0])
    g = float(jax.grad(loss)(jnp.asarray(t0)))
    eps = 1e-6
    fd = float((loss(jnp.asarray(t0 + eps)) - loss(jnp.asarray(t0 - eps))) / (2.0 * eps))
    assert np.isfinite(g) and np.isfinite(fd) and abs(fd) > 0.0
    np.testing.assert_allclose(g, fd, rtol=1e-4)


# ---------------------------------------------------------------------------
# Ramped (non-uniform Nxi_for_x) PAS decks: per-subsystem truncated tier 1
# ---------------------------------------------------------------------------


def _ramped_pas_op() -> KineticOperator:
    """The tiny PAS fixture rescaled so Nxi_for_x_option=1 gives a real ramp."""
    text = (
        _load_text("pas_1species_PAS_noEr_tiny_scheme1")
        .replace("Nxi = 4", "Nxi = 16")
        .replace("Nx = 3", "Nx = 5")
        .replace("Nxi_for_x_option = 0", "Nxi_for_x_option = 1")
    )
    op = KineticOperator.from_namelist(parse_sfincs_input_text(text))
    # The whole point: the production speed-dependent Legendre ramp.
    assert int(np.min(np.asarray(op.n_xi_for_x))) < op.n_xi
    return op


def test_ramped_pas_routes_truncated_and_matches_pinned_referees() -> None:
    """Auto must route ramped PAS decks to the per-subsystem truncated kernel.

    The solution (lowest-3 Legendre blocks and the vm flux/flow moments) must
    match both pinned referees — tier-2 GCROT and the dense pinned direct
    solve — to 1e-10; blocks l >= 3 are zero-padded (which covers every
    Nxi_for_x-truncated DOF, since keep <= min Nxi_for_x).
    """
    op = _ramped_pas_op()
    rhs = op.rhs()

    trunc = solve(op, rhs, method="auto")
    assert trunc.method == "block_tridiagonal_truncated"
    assert trunc.converged

    gcrot = solve(op, rhs, method="gmres", tol=1e-12)
    assert gcrot.converged
    dense = sla.solve(materialize_dense(op, pin_masked_dofs=True), np.asarray(rhs))

    n_s, n_x, n_xi, n_t, n_z = op.f_shape
    n_tz = n_t * n_z
    xt = np.asarray(trunc.x)
    ft = xt[: op.f_size].reshape(n_s, n_x, n_xi, n_tz)
    assert np.max(np.abs(ft[:, :, 3:])) == 0.0
    for x_ref in (np.asarray(gcrot.x), dense):
        f_ref = x_ref[: op.f_size].reshape(n_s, n_x, n_xi, n_tz)
        dl = np.linalg.norm(ft[:, :, :3] - f_ref[:, :, :3]) / np.linalg.norm(f_ref[:, :, :3])
        assert dl < 1e-10
        # The sources of this drive are numerically zero; agree absolutely.
        np.testing.assert_allclose(
            xt[op.f_size :], x_ref[op.f_size :], atol=1e-12 * np.linalg.norm(f_ref)
        )
        m_t = _vm_moments(op, xt[None, :])
        m_ref = _vm_moments(op, x_ref[None, :])
        for key in ("particleFlux", "heatFlux", "FSABFlow"):
            rel = np.abs(m_t[key] - m_ref[key]) / np.maximum(np.abs(m_ref[key]), 1e-300)
            assert rel.max() < 1e-10, f"{key}: {rel.max():.3e}"

    # The full-band factorization cannot carry the ramp and must refuse.
    with pytest.raises(NotImplementedError, match="uniform Nxi_for_x"):
        solve(op, rhs, method="block_tridiagonal")


def test_gradient_through_ramped_truncated_route_matches_finite_differences() -> None:
    op0 = _ramped_pas_op()

    def loss(t_hat_scalar: jnp.ndarray) -> jnp.ndarray:
        op = replace(op0, t_hat=jnp.reshape(t_hat_scalar, (1,)))
        # Auto routes the ramp to the truncated kernel; grad flows straight
        # through the per-subsystem block-Thomas sweeps.
        result = solve(op, op.rhs(), method="auto", differentiable=True)
        return jnp.sum(result.x**2)

    t0 = float(op0.t_hat[0])
    g = float(jax.grad(loss)(jnp.asarray(t0)))
    eps = 1e-6
    fd = float((loss(jnp.asarray(t0 + eps)) - loss(jnp.asarray(t0 - eps))) / (2.0 * eps))
    assert np.isfinite(g) and np.isfinite(fd) and abs(fd) > 0.0
    np.testing.assert_allclose(g, fd, rtol=1e-6)


# ---------------------------------------------------------------------------
# Subsystem batching: concurrent per-(species, x) elimination width
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("make_op", ["uniform", "ramped", "ramped_grouped"])
def test_truncated_subsystem_batch_any_width_is_bit_identical(make_op: str) -> None:
    """Every subsystem batch width must compute identical answers.

    The width only changes how many independent (species, x) eliminations run
    concurrently (grouped by equal ``Nxi_for_x`` block count on the ramped
    path); the per-subsystem arithmetic is unchanged, so the solutions must
    agree to the bit level.  ``ramped_grouped`` repeats ``Nxi_for_x`` values
    so the equal-``n_blocks`` groups hold more than one subsystem and the
    batched group path is genuinely exercised.
    """
    if make_op == "uniform":
        op = _load_op("pas_1species_PAS_noEr_tiny_scheme1")
    else:
        op = _ramped_pas_op()
        if make_op == "ramped_grouped":
            op = replace(
                op, n_xi_for_x=jnp.asarray([4, 4, 9, 16, 16], dtype=jnp.int32)
            )
    rhs = op.rhs()
    b = int(op.n_species) * int(op.n_x)
    x1 = np.asarray(
        solve(op, rhs, method="block_tridiagonal_truncated", subsystem_batch=1).x
    )
    for width in (2, b, "auto"):
        r = solve(op, rhs, method="block_tridiagonal_truncated", subsystem_batch=width)
        assert r.converged
        diff = np.max(np.abs(np.asarray(r.x) - x1))
        assert diff <= 1e-13, f"width={width}: max abs diff {diff:.3e}"


def test_truncated_subsystem_width_respects_memory_budget() -> None:
    op = _ramped_pas_op()
    b = int(op.n_species) * int(op.n_x)
    # A tiny budget forces the serial minimum-memory width.
    assert tier1_truncated_subsystem_width(op, memory_budget_gb=1e-12) == 1
    # An ample budget admits the full subsystem batch.
    assert tier1_truncated_subsystem_width(op, memory_budget_gb=1024.0) == b
    # The footprint model grows with the width it models (lockstep).
    lean = tier1_truncated_peak_memory_bytes(op, subsystem_batch=1)
    wide = tier1_truncated_peak_memory_bytes(op, subsystem_batch=b)
    assert wide > lean
    # "auto" is backend-aware: serial on CPU (measured best — XLA:CPU runs
    # batched LAPACK calls serially per element), memory-budgeted elsewhere.
    if jax.default_backend() == "cpu":
        assert _resolve_subsystem_batch(op, "auto", 3) == 1
    assert _resolve_subsystem_batch(op, 4, 3) == 4
    assert _resolve_subsystem_batch(op, 10**6, 3) == b
    with pytest.raises(ValueError):
        _resolve_subsystem_batch(op, 0, 3)
    with pytest.raises(ValueError):
        _resolve_subsystem_batch(op, "wide", 3)


# ---------------------------------------------------------------------------
# Tier 2 on the Fokker-Planck fixture: convergence, parity, preconditioning
# ---------------------------------------------------------------------------


def test_tier2_converges_and_matches_dense_fp() -> None:
    op = _load_op("quick_2species_FPCollisions_noEr")
    ok, _ = tier1_available(op)
    assert not ok  # FP couples (species, x): tier 1 must refuse
    rhs = op.rhs()
    tol = 1e-10
    result = solve(op, rhs, method="gmres", tol=tol)
    assert result.method == "gcrot"
    assert result.converged
    assert float(result.residual_norms[0]) < tol * float(jnp.linalg.norm(rhs))
    x_ref = _dense_solve(op, np.asarray(rhs)[:, None])[:, 0]
    assert _rel_err(np.asarray(result.x), x_ref) < 1e-8


def test_tier2_coarse_preconditioner_reduces_iterations() -> None:
    op = _load_op("quick_2species_FPCollisions_noEr")
    rhs = op.rhs()
    r_pc = solve(op, rhs, method="gmres", tol=1e-8)
    r_nopc = solve(
        op, rhs, method="gmres", tol=1e-8, use_preconditioner=False, max_restarts=60
    )
    assert r_pc.converged
    # The unpreconditioned Krylov stalls on this FP system (it hits its
    # iteration cap far from tolerance); the coarse-operator preconditioner
    # must converge in strictly fewer iterations than the cap it burned.
    assert not r_nopc.converged or r_pc.iterations < r_nopc.iterations
    assert r_pc.iterations < r_nopc.iterations


# ---------------------------------------------------------------------------
# Tier 2 on the improved Sugama fixture (collisionOperator=3): routing, parity,
# and differentiability — the momentum/energy-restoring field coupling is
# dropped from the coarse preconditioner but kept in the full operator.
# ---------------------------------------------------------------------------


def test_sugama_collisionop3_routes_tier2_matches_tier3_and_differentiates() -> None:
    op = _load_sugama_op()
    assert op.sugama is not None and op.fp is None  # collisionOperator=3 built
    assert op.constraint_scheme == 1  # {density, temperature} speed null space
    ok, _reason = tier1_available(op)
    assert not ok  # dense (species, x) collisions + constraintScheme=1: tier 1 refuses

    rhs = op.rhs()
    tol = 1e-10
    # The auto policy must pick the differentiable tier-2 GCROT, NOT the
    # non-differentiable tier-3 host direct fallback.
    r_auto = solve(op, rhs, method="auto", tol=tol)
    assert r_auto.method == "gcrot"
    assert r_auto.converged

    # Tier 2 == tier 3 (host SuperLU) element-wise.
    r_direct = solve(op, rhs, method="direct")
    assert r_direct.method == "direct"
    assert _rel_err(np.asarray(r_auto.x), np.asarray(r_direct.x)) < 1e-8

    # Gradient flows through the differentiable tier-2 solve.  The Sugama mat is
    # exactly proportional to nu_n, so a scalar multiplier on it models a nu_n
    # scan; AD (implicit function theorem) must match a central finite
    # difference on the differentiable segment.
    base_mat = op.sugama.mat
    g = jnp.asarray(np.random.default_rng(0).standard_normal(op.total_size))

    def moment(s: float) -> jnp.ndarray:
        op_s = replace(op, sugama=replace(op.sugama, mat=s * base_mat))
        sol = solve(op_s, rhs, method="gmres", tol=1e-11, differentiable=True)
        return jnp.dot(g, sol.x)

    # auto + differentiable stays on tier 2 (the production path).
    r_auto_diff = solve(
        replace(op, sugama=replace(op.sugama, mat=1.0 * base_mat)),
        rhs,
        method="auto",
        tol=1e-9,
        differentiable=True,
    )
    assert r_auto_diff.method == "gcrot"

    grad_ad = float(jax.grad(moment)(1.0))
    eps = 1e-4
    grad_fd = float((moment(1.0 + eps) - moment(1.0 - eps)) / (2 * eps))
    assert np.isfinite(grad_ad)
    assert abs(grad_ad - grad_fd) / max(1.0, abs(grad_fd)) < 1e-6


# ---------------------------------------------------------------------------
# Auto-policy selection
# ---------------------------------------------------------------------------


def test_auto_policy_selects_tier1_for_pas_and_tier2_for_fp() -> None:
    op_pas = _load_op("pas_1species_PAS_noEr_tiny_scheme1")
    r_pas = solve(op_pas, op_pas.rhs(), method="auto")
    assert r_pas.method == "block_tridiagonal"

    op_fp = _load_op("quick_2species_FPCollisions_noEr")
    r_fp = solve(op_fp, op_fp.rhs(), method="auto", tol=1e-8)
    assert r_fp.method == "gcrot"
    assert r_fp.converged


def test_tier1_refuses_er_xdot_l2_coupling() -> None:
    # Er xDot couples L±2: the analytic block extraction (and hence tier 1)
    # must refuse, leaving this family to the Krylov/direct tiers.
    op = _load_op("er_xdot_1species_tiny")
    ok, _reason = tier1_available(op)
    assert not ok


def test_auto_policy_tier3_fallback_on_iteration_cap() -> None:
    # Starve tier 2 (no preconditioner, tiny restart budget) on the FP
    # fixture: the auto policy must breach the cap loudly and land on the
    # tier-3 host direct solve, which still returns the right answer.
    op = _load_op("quick_2species_FPCollisions_noEr")
    rhs = op.rhs()
    result = solve(
        op, rhs, method="auto", tol=1e-10, use_preconditioner=False, max_restarts=2
    )
    assert result.method == "direct"
    assert result.converged
    x_ref = _dense_solve(op, np.asarray(rhs)[:, None])[:, 0]
    assert _rel_err(np.asarray(result.x), x_ref) < 1e-8


def test_explicit_tier1_request_raises_on_fp() -> None:
    op = _load_op("quick_2species_FPCollisions_noEr")
    with pytest.raises(NotImplementedError):
        solve(op, op.rhs(), method="block_tridiagonal")


# ---------------------------------------------------------------------------
# Tier 3 (host SuperLU) parity
# ---------------------------------------------------------------------------


def test_tier3_direct_solve_matches_dense() -> None:
    op = _load_op("pas_1species_PAS_noEr_tiny_scheme1")
    rhs = op.rhs()
    result = solve(op, rhs, method="direct")
    assert result.method == "direct"
    assert result.converged
    x_ref = _dense_solve(op, np.asarray(rhs)[:, None])[:, 0]
    assert _rel_err(np.asarray(result.x), x_ref) < 1e-10


# ---------------------------------------------------------------------------
# Recycling / warm start across an Er continuation
# ---------------------------------------------------------------------------


def _op_with_er(base_text: str, er: float) -> KineticOperator:
    assert "Er = 0" in base_text
    text = base_text.replace("Er = 0", f"Er = {er:.6f}")
    return KineticOperator.from_namelist(parse_sfincs_input_text(text))


def test_recycling_cuts_iterations_on_er_continuation() -> None:
    base = _load_text("quick_2species_FPCollisions_noEr")
    op1 = _op_with_er(base, 0.005)
    op2 = _op_with_er(base, 0.010)
    tol = 1e-9

    r1 = solve(op1, op1.rhs(), method="gmres", tol=tol)
    assert r1.converged

    cold = solve(op2, op2.rhs(), method="gmres", tol=tol)
    warm = solve(op2, op2.rhs(), method="gmres", tol=tol, x0=r1.x, recycle=r1.recycle)
    assert cold.converged and warm.converged
    assert warm.iterations < cold.iterations


# ---------------------------------------------------------------------------
# Differentiability: jax.grad through the tier-1 solve vs finite differences
# ---------------------------------------------------------------------------


def test_gradient_through_tier1_solve_matches_finite_differences() -> None:
    op0 = _load_op("pas_1species_PAS_noEr_tiny_scheme1")

    def loss(t_hat_scalar: jnp.ndarray) -> jnp.ndarray:
        # Thread the scalar through the operator pytree (streaming/mirror
        # coefficients and the RHS drive depend on THat); the PAS collision
        # matrices stay frozen, so finite differences see the same function.
        op = replace(op0, t_hat=jnp.reshape(t_hat_scalar, (1,)))
        result = solve(op, op.rhs(), method="block_tridiagonal", differentiable=True)
        return jnp.sum(result.x**2)

    t0 = float(op0.t_hat[0])
    g = float(jax.grad(loss)(jnp.asarray(t0)))
    eps = 1e-6
    fd = float((loss(jnp.asarray(t0 + eps)) - loss(jnp.asarray(t0 - eps))) / (2.0 * eps))
    assert np.isfinite(g) and np.isfinite(fd) and abs(fd) > 0.0
    np.testing.assert_allclose(g, fd, rtol=1e-6)


# ---------------------------------------------------------------------------
# FP + constraintScheme=1 with Nxi_for_x truncation: the rectangular state
# layout embeds the packed Fortran system with exact zero rows on the
# truncated (x, L) DOFs, so the raw embedding is structurally singular and
# the naive adjoint (transposed) solve is inconsistent -> silently wrong
# gradients.  solve() must pin those DOFs (identity rows/columns) and the
# implicit-function-theorem gradient must then match finite differences.
# ---------------------------------------------------------------------------

FP_CS1_TRUNCATED_TEXT = """
&general
/
&geometryParameters
  geometryScheme = 1
/
&speciesParameters
  Zs = 1 6
  mHats = 1 6
  nHats = 0.6d+0 0.009d+0
  THats = 0.5d+0 0.8d+0
  dNHatdrHats = -0.587199 -0.00195733
  dTHatdrHats = -0.587199 -0.391466
/
&physicsParameters
  Delta = 4.5694d-3
  alpha = 1.0d+0
  nu_n = 8.4774d-3
  Er = 0
  collisionOperator = 0
  includePhi1 = .false.
/
&resolutionParameters
  Ntheta = 7
  Nzeta = 5
  Nxi = 6
  Nx = 4
/
"""


def _fp_cs1_truncated_op() -> KineticOperator:
    op = KineticOperator.from_namelist(parse_sfincs_input_text(FP_CS1_TRUNCATED_TEXT))
    # The whole point of this fixture: the default Nxi_for_x truncates L at
    # low x (constraintScheme=1 Fokker-Planck), so the rectangular embedding
    # is structurally singular.
    assert op.constraint_scheme == 1 and op.fp is not None
    assert int(np.min(np.asarray(op.n_xi_for_x))) < op.n_xi
    return op


def test_fp_cs1_truncated_embedding_is_singular_and_pinning_fixes_it() -> None:
    op = _fp_cs1_truncated_op()
    mask = np.asarray(op.active_dof_mask())
    assert mask is not None and mask.min() == 0.0

    raw = materialize_dense(op)
    # Truncated DOFs are exact zero rows of the raw embedding (Fortran v3
    # never carries them: packed indexing in indices.F90).
    assert np.max(np.abs(raw[mask == 0.0, :])) == 0.0

    pinned = materialize_dense(op, pin_masked_dofs=True)
    # Pinned rows/columns are exactly the identity...
    n = op.total_size
    eye = np.eye(n)
    assert np.array_equal(pinned[mask == 0.0, :], eye[mask == 0.0, :])
    assert np.array_equal(pinned[:, mask == 0.0], eye[:, mask == 0.0])
    # ...and the active block is untouched and nonsingular.
    act = mask == 1.0
    assert np.array_equal(pinned[np.ix_(act, act)], raw[np.ix_(act, act)])
    s = np.linalg.svd(pinned, compute_uv=False)
    assert s[-1] > 1e-8

    # The tier-2 solve on the physical RHS matches the pinned dense solve.
    rhs = op.rhs()
    assert float(np.max(np.abs(np.asarray(rhs)[mask == 0.0]))) == 0.0
    result = solve(op, rhs, method="gmres", tol=1e-10)
    assert result.converged
    x_ref = sla.solve(pinned, np.asarray(rhs))
    assert _rel_err(np.asarray(result.x), x_ref) < 1e-8


def test_fp_cs1_gradients_match_fd() -> None:
    """jax.grad through the differentiable tier-2 solve vs central FD.

    Before the truncated-DOF pinning this returned catastrophically wrong
    gradients (the adjoint system was inconsistent) while the forward solve
    converged fine — the historical silent-wrong-gradient failure.
    """
    op0 = _fp_cs1_truncated_op()

    def loss(scale: jnp.ndarray, differentiable: bool = True) -> jnp.ndarray:
        # Thread the scalar through the operator pytree (streaming/mirror and
        # the RHS drive depend on THat); the Fokker-Planck matrices stay
        # frozen, so finite differences see the same function.
        op = replace(op0, t_hat=op0.t_hat * scale)
        result = solve(
            op, op.rhs(), method="gmres", tol=1e-10, differentiable=differentiable
        )
        return jnp.sum(result.x**2)

    g = float(jax.grad(loss)(jnp.asarray(1.0)))
    eps = 1e-4
    fd = float(
        (loss(jnp.asarray(1.0 + eps), differentiable=False)
         - loss(jnp.asarray(1.0 - eps), differentiable=False)) / (2.0 * eps)
    )
    assert np.isfinite(g) and np.isfinite(fd) and abs(fd) > 0.0
    np.testing.assert_allclose(g, fd, rtol=1e-4)


def test_differentiable_solve_aborts_loudly_on_genuinely_singular_operator() -> None:
    """check_adjoint (default on) must abort instead of silently corrupting grads.

    Dropping the constraint scheme leaves the Fokker-Planck f-block with its
    physical (Maxwellian) null space, which pinning cannot fix; the stalled
    forward/adjoint GCROT solve must raise.
    """
    op0 = _fp_cs1_truncated_op()
    op_singular = replace(op0, constraint_scheme=0)
    # A generic linear functional of the solution: its cotangent is not in
    # range(A^T) of the singular operator, so the adjoint solve must stall.
    w = jnp.asarray(
        np.random.default_rng(7).standard_normal(op_singular.total_size)
    )

    def loss(scale: jnp.ndarray) -> jnp.ndarray:
        op = replace(op_singular, t_hat=op_singular.t_hat * scale)
        result = solve(
            op, op.rhs(), method="gmres", tol=1e-10, differentiable=True,
            max_restarts=10,
        )
        return jnp.dot(w, result.x)

    with pytest.raises(Exception, match="GCROT solve failed to converge"):
        jax.grad(loss)(jnp.asarray(1.0))


# ---------------------------------------------------------------------------
# jit-safety: the Nxi_for_x truncation mask must not host-materialize, so the
# coarse preconditioner and the differentiable solve stay traceable and reuse
# their compilation.  Regression guard for the two former
# ``np.asarray(op._mask())`` host round-trips (solve.py / drift_kinetic.py) that
# blocked jit-over-operator-leaves, vmap batching, and cross-eval jit reuse.
# ---------------------------------------------------------------------------


def test_coarse_preconditioner_is_jit_safe_over_traced_operator_leaves() -> None:
    """``build_coarse_preconditioner`` must build when the operator *leaves* are
    tracers (regression for ``mask = np.asarray(op._mask())`` at its top).

    A ramped ``Nxi_for_x`` deck makes the truncation mask non-uniform; the
    preconditioner action under jit-over-leaves must match the eager build.
    """
    op = _ramped_pas_op()
    leaves, treedef = jax.tree_util.tree_flatten(op)
    v = jnp.asarray(np.linspace(-1.0, 1.0, op.total_size), dtype=jnp.float64)

    def precond_action(ls: list) -> jnp.ndarray:
        precond, _ = build_coarse_preconditioner(jax.tree_util.tree_unflatten(treedef, ls))
        return precond(v)

    jitted = jax.jit(precond_action)(leaves)  # compiles (was a Tracer error)
    ref, _ = build_coarse_preconditioner(op)
    np.testing.assert_allclose(np.asarray(jitted), np.asarray(ref(v)), rtol=1e-9, atol=1e-11)


def test_jit_value_and_grad_through_differentiable_solve_compiles_once() -> None:
    """The differentiable solve must be jittable end-to-end and *reuse* its
    compilation.

    ``jax.jit(value_and_grad(...))`` around the differentiable tier-1 solve must
    compile exactly once and NOT retrace when only the operator leaf VALUES
    change — the optimization inner loop that used to recompile per eval.  The
    jitted gradient still matches central finite differences.
    """
    op0 = _load_op("pas_1species_PAS_noEr_tiny_scheme1")
    traces = {"n": 0}

    def loss(t_vec: jnp.ndarray) -> jnp.ndarray:
        traces["n"] += 1  # increments once per trace (i.e. per compile)
        op = replace(op0, t_hat=t_vec)
        res = solve(op, op.rhs(), method="block_tridiagonal", differentiable=True)
        return jnp.sum(res.x**2)

    vg = jax.jit(jax.value_and_grad(loss))
    p0 = jnp.reshape(op0.t_hat[0], (1,))
    val0, g0 = vg(p0)
    n_after_first = traces["n"]
    val1, _ = vg(p0 * 1.3)  # different leaf VALUE, identical shape/dtype
    val2, _ = vg(p0 * 0.6)
    # compiled exactly once; new values reuse the executable (no retrace)
    assert n_after_first == 1
    assert traces["n"] == 1
    assert np.isfinite(float(g0[0]))
    assert float(val0) != float(val1) != float(val2)

    # the jitted AD gradient matches central finite differences
    def loss_plain(t: float) -> float:
        op = replace(op0, t_hat=jnp.reshape(jnp.asarray(t), (1,)))
        return float(jnp.sum(solve(op, op.rhs(), method="block_tridiagonal").x**2))

    t0 = float(op0.t_hat[0])
    eps = 1e-6
    fd = (loss_plain(t0 + eps) - loss_plain(t0 - eps)) / (2.0 * eps)
    np.testing.assert_allclose(float(g0[0]), fd, rtol=1e-5)


# ---------------------------------------------------------------------------
# Optional-dependency policy: solvax is optional until its PyPI release.
# ---------------------------------------------------------------------------


def test_solve_importable_without_solvax_and_fails_loudly_on_use() -> None:
    """``import dkx.solve`` must work without solvax; use must raise clearly.

    Runs in a subprocess (this session already imported solvax) and hides the
    package by poisoning ``sys.modules`` before the import.
    """
    import subprocess
    import sys

    code = "\n".join(
        [
            "import sys",
            "for m in ('solvax', 'solvax.direct', 'solvax.implicit', 'solvax.krylov',",
            "          'solvax.native', 'solvax.operators'):",
            "    sys.modules[m] = None  # poisoned: import raises ImportError",
            "import dkx.solve as solve_mod",
            "assert solve_mod._SOLVAX_IMPORT_ERROR is not None",
            "try:",
            "    solve_mod._require_solvax()",
            "except ImportError as exc:",
            "    assert 'solvax' in str(exc)",
            "else:",
            "    raise SystemExit('expected ImportError on solvax use')",
            "print('guarded-import-ok')",
        ]
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
    )
    assert proc.returncode == 0, proc.stderr
    assert "guarded-import-ok" in proc.stdout


# ---------------------------------------------------------------------------
# Size-aware device routing (solve(device=...))
# ---------------------------------------------------------------------------


def test_resolve_solve_device_semantics_on_cpu_host() -> None:
    """The routing knob on a CPU-default host: inert except explicit errors."""
    from dkx.solve import _resolve_solve_device

    op = _load_op("pas_1species_PAS_noEr_tiny_scheme1")
    assert jax.default_backend() == "cpu", "test assumes a CPU-default test host"
    # auto/default/cpu: no movement on a CPU host.
    assert _resolve_solve_device("auto", "block_tridiagonal", op, False) is None
    assert _resolve_solve_device("default", "gmres", op, False) is None
    assert _resolve_solve_device("cpu", "gmres", op, False) is None
    assert _resolve_solve_device(None, "gmres", op, False) is None
    # traced: always inert (jit-safety), whatever the request.
    cpu0 = jax.local_devices(backend="cpu")[0]
    assert _resolve_solve_device(cpu0, "gmres", op, True) is None
    # explicit device object passes through untraced.
    assert _resolve_solve_device(cpu0, "gmres", op, False) is cpu0
    # tier 3 is a host solve already: no movement.
    assert _resolve_solve_device(cpu0, "direct", op, False) is None
    # accelerator request on a CPU-only host is a loud error.
    with pytest.raises(ValueError, match="default JAX backend is CPU"):
        _resolve_solve_device("gpu", "gmres", op, False)
    with pytest.raises(ValueError, match="unknown solve device"):
        _resolve_solve_device("quantum", "gmres", op, False)


def test_solve_device_thresholds_read_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from dkx.solve import (
        _SOLVE_CPU_MAX_TIER1_DEFAULT,
        _SOLVE_CPU_MAX_TIER2_DEFAULT,
        _env_size,
    )

    # Auto-routing is OFF by default: the same-host measurements (see
    # docs/performance.rst) found the GPU faster at every practical size, so
    # a nonzero default is unsupported by data.  Users opt in via the envs.
    assert _SOLVE_CPU_MAX_TIER1_DEFAULT == 0
    assert _SOLVE_CPU_MAX_TIER2_DEFAULT == 0

    monkeypatch.setenv("DKX_SOLVE_CPU_MAX_SIZE_TIER1", "12345")
    assert _env_size("DKX_SOLVE_CPU_MAX_SIZE_TIER1", _SOLVE_CPU_MAX_TIER1_DEFAULT) == 12345
    monkeypatch.setenv("DKX_SOLVE_CPU_MAX_SIZE_TIER1", "not-a-number")
    assert (
        _env_size("DKX_SOLVE_CPU_MAX_SIZE_TIER1", _SOLVE_CPU_MAX_TIER1_DEFAULT)
        == _SOLVE_CPU_MAX_TIER1_DEFAULT
    )
    monkeypatch.delenv("DKX_SOLVE_CPU_MAX_SIZE_TIER1")
    assert (
        _env_size("DKX_SOLVE_CPU_MAX_SIZE_TIER2", _SOLVE_CPU_MAX_TIER2_DEFAULT)
        == _SOLVE_CPU_MAX_TIER2_DEFAULT
    )


def test_solve_on_explicit_second_cpu_device_matches_and_returns_home() -> None:
    """Real move/move-back exercise using two host CPU devices in a subprocess.

    ``--xla_force_host_platform_device_count=2`` must be set before jax import,
    so this runs in a fresh interpreter: solve on cpu:1 with inputs on cpu:0,
    check the answer matches the unrouted solve and comes back on cpu:0.
    """
    import subprocess
    import sys

    deck = REF / "pas_1species_PAS_noEr_tiny_scheme1.input.namelist"
    code = "\n".join(
        [
            "import os",
            "os.environ['XLA_FLAGS'] = (os.environ.get('XLA_FLAGS', '') +",
            "    ' --xla_force_host_platform_device_count=2').strip()",
            "import jax, numpy as np",
            "from dkx.drift_kinetic import KineticOperator",
            "from dkx.namelist import read_sfincs_input",
            "from dkx.solve import solve",
            f"op = KineticOperator.from_namelist(read_sfincs_input({str(deck)!r}))",
            "rhs = op.rhs()",
            "cpu0, cpu1 = jax.local_devices(backend='cpu')[:2]",
            "rhs0 = jax.device_put(rhs, cpu0)",
            "ref = solve(op, rhs0, device='default')",
            "routed = solve(op, rhs0, device=cpu1)",
            "assert routed.method == ref.method",
            "err = float(np.max(np.abs(np.asarray(routed.x) - np.asarray(ref.x))))",
            "assert err < 1e-12, f'routed answer differs: {err}'",
            "assert routed.x.devices() == {cpu0}, routed.x.devices()",
            "print('device-routing-ok')",
        ]
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
    )
    assert proc.returncode == 0, proc.stderr
    assert "device-routing-ok" in proc.stdout
