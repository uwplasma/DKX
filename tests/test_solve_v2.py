"""Referee tests for ``sfincs_jax.solve_v2`` — the plan-§2.3 three-tier auto-policy.

Tiny fixtures only (shared with ``tests/test_dke_v2.py``):

- tier 1 (analytic block-Thomas) must match the tier-3 dense solve to 1e-10
  on the monoenergetic (RHSMode=3) and PAS (RHSMode=1) fixtures, and match the
  committed probing-based ``solvers/block_tridiagonal_transport`` path on the
  RHSMode=3 transport columns;
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

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg as sla

from sfincs_jax.dke_v2 import KineticOperator
from sfincs_jax.namelist import parse_sfincs_input_text, read_sfincs_input
from sfincs_jax.solve_v2 import materialize_dense, solve, tier1_available

REF = Path(__file__).parent / "ref"


def _load_op(name: str) -> KineticOperator:
    return KineticOperator.from_namelist(read_sfincs_input(REF / f"{name}.input.namelist"))


def _load_text(name: str) -> str:
    return (REF / f"{name}.input.namelist").read_text()


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


def test_tier1_matches_committed_block_tridiagonal_transport_path() -> None:
    """The analytic tier-1 replaces the committed probing-based RHSMode=3 solver."""
    from sfincs_jax.operators.profile_system import full_system_operator_from_namelist
    from sfincs_jax.solvers.block_tridiagonal_transport import (
        solve_rhs3_block_tridiagonal,
    )

    nml = read_sfincs_input(REF / "monoenergetic_PAS_tiny_scheme1.input.namelist")
    op_new = KineticOperator.from_namelist(nml)
    op_old = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)

    rhs = jnp.stack([op_new.rhs(1), op_new.rhs(2)], axis=1)
    x_new = np.asarray(solve(op_new, rhs, method="block_tridiagonal").x)
    x_old = np.asarray(solve_rhs3_block_tridiagonal(op=op_old, rhs_columns=rhs))
    assert _rel_err(x_new, x_old) < 1e-10


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
