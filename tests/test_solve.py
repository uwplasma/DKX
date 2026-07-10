"""Referee tests for ``sfincs_jax.solve`` — the plan-§2.3 three-tier auto-policy.

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

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg as sla

from sfincs_jax.drift_kinetic import KineticOperator
from sfincs_jax.namelist import parse_sfincs_input_text, read_sfincs_input
from sfincs_jax.solve import materialize_dense, solve, tier1_available

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


@pytest.mark.parametrize("base", ("monoenergetic_PAS_tiny_scheme1", "monoenergetic_PAS_tiny_scheme11"))
def test_tier1_matches_recorded_fortran_state_vectors_rhsmode3(base: str) -> None:
    """Tier 1 must reproduce the frozen v3 PETSc stateVector for both transport drives.

    This is the direct Fortran referee for the structured-direct tier on the
    RHSMode=3 transport columns; it replaces the equality test against the
    retired probing-based ``solvers/block_tridiagonal_transport`` POC.
    """
    from sfincs_jax.validation.fortran import read_petsc_vec

    op = _load_op(base)
    rhs = jnp.stack([op.rhs(1), op.rhs(2)], axis=1)
    result = solve(op, rhs, method="block_tridiagonal")
    assert result.converged
    x = np.asarray(result.x)
    for which_rhs in (1, 2):
        x_ref = read_petsc_vec(REF / f"{base}.whichRHS{which_rhs}.stateVector.petscbin").values
        assert _rel_err(x[:, which_rhs - 1], x_ref) < 1e-10, f"whichRHS={which_rhs}"


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
# Optional-dependency policy: solvax is optional until its PyPI release.
# ---------------------------------------------------------------------------


def test_solve_importable_without_solvax_and_fails_loudly_on_use() -> None:
    """``import sfincs_jax.solve`` must work without solvax; use must raise clearly.

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
            "import sfincs_jax.solve as solve_mod",
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
