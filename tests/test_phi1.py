"""Canonical Phi1 / quasineutrality slice: parity, convergence, gradient, output.

Covers the ``includePhi1`` vertical slice consolidated into
:mod:`sfincs_jax.drift_kinetic` (the quasineutrality block, the ``<Phi1>=0``
lambda row, and the Phi1-in-kinetic-equation coupling for
``quasineutralityOption`` 1/2) and its nonlinear Newton solve in
:mod:`sfincs_jax.phi1`.

Parity oracle: ``operators.profile_system`` (the legacy ``V3FullSystemOperator``
residual/matvec) and the Fortran reference state vector.  This test conserves
the Fortran-parity assertion of the retired
``tests/test_full_system_newton_krylov.py`` (same fixture, atol 5e-8).

Section 6 conserves the ``includePhi1InCollisionOperator`` physics of the
retired legacy owner (the poloidally varying Fokker-Planck collision operator):
canonical residual/operator/Jacobian parity vs the legacy assembly and a Fortran
state-vector end-to-end check on the collision-Phi1 fixture.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# x64 is enabled on import of any sfincs_jax operator module (drift_kinetic).
from sfincs_jax.drift_kinetic import kinetic_operator_from_namelist
from sfincs_jax.moments import rhsmode1_moments
from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.operators.profile_system import (
    apply_v3_full_system_operator,
    full_system_operator_from_namelist,
    residual_v3_full_system,
    rhs_v3_full_system,
)
from sfincs_jax.phi1 import operator_from_input, phi1_state, solve_phi1
from sfincs_jax.validation.fortran import read_petsc_vec
from sfincs_jax.writer import operator_containers

_REF = Path(__file__).parent / "ref"
# The Fortran-parity fixture (quasineutralityOption=2, includePhi1InKineticEquation,
# adiabatic species) reused from the retired test_full_system_newton_krylov.py.
_FIXTURE = "pas_1species_PAS_noEr_tiny_withPhi1_inKinetic_linear"
_INPUT = _REF / f"{_FIXTURE}.input.namelist"
_STATEVEC = _REF / f"{_FIXTURE}.stateVector.petscbin"


def _legacy_op():
    return full_system_operator_from_namelist(nml=read_sfincs_input(_INPUT))


def _canonical_op():
    return kinetic_operator_from_namelist(read_sfincs_input(_INPUT))


def _fortran_state() -> np.ndarray:
    return np.asarray(read_petsc_vec(_STATEVEC).values, dtype=np.float64)


# ---------------------------------------------------------------------------
# 1. Operator / residual element-wise parity vs the legacy assembly (atol 1e-12)
# ---------------------------------------------------------------------------


def test_operator_layout_matches_legacy() -> None:
    op, leg = _canonical_op(), _legacy_op()
    assert op.include_phi1 and op.include_phi1_in_kinetic
    assert op.quasineutrality_option == int(leg.quasineutrality_option)
    assert int(op.total_size) == int(leg.total_size)
    assert int(op.f_size) == int(leg.f_size)
    assert int(op.phi1_size) == int(leg.phi1_size)
    assert int(op.extra_size) == int(leg.extra_size)


def test_residual_elementwise_parity_vs_legacy() -> None:
    op, leg = _canonical_op(), _legacy_op()
    rng = np.random.default_rng(0)
    for seed in range(3):
        x = jnp.asarray(rng.standard_normal(op.total_size))
        r_can = np.asarray(op.residual_phi1(x))
        r_leg = np.asarray(residual_v3_full_system(leg, x))
        np.testing.assert_allclose(r_can, r_leg, rtol=0.0, atol=1e-12)


def test_operator_and_rhs_elementwise_parity_vs_legacy() -> None:
    op, leg = _canonical_op(), _legacy_op()
    # rhs at the built (Phi1=0) linearization point.
    np.testing.assert_allclose(
        np.asarray(op.rhs_phi1()), np.asarray(rhs_v3_full_system(leg)), rtol=0.0, atol=1e-12
    )
    # Nonlinear operator action A(x) (include_jacobian_terms=False) at a random state.
    rng = np.random.default_rng(1)
    x = jnp.asarray(rng.standard_normal(op.total_size))
    phi1 = x[op.f_size : op.f_size + op.n_theta * op.n_zeta].reshape((op.n_theta, op.n_zeta))
    a_can = np.asarray(replace(op, phi1_hat_base=phi1)._apply_phi1_operator(x))
    a_leg = np.asarray(apply_v3_full_system_operator(replace(leg, phi1_hat_base=phi1), x, include_jacobian_terms=False))
    np.testing.assert_allclose(a_can, a_leg, rtol=0.0, atol=1e-12)


def test_jacobian_matvec_parity_vs_legacy_linearize() -> None:
    """``apply`` (the JVP used by solve.solve) matches the oracle's jax.linearize."""
    op, leg = _canonical_op(), _legacy_op()
    x0 = jnp.asarray(_fortran_state())
    op_lin = replace(op, phi1_lin_state=x0)
    rng = np.random.default_rng(2)
    v = jnp.asarray(rng.standard_normal(op.total_size))
    jvp_can = np.asarray(op_lin.apply(v))
    _, jvp_leg = jax.linearize(lambda xx: residual_v3_full_system(leg, xx), x0)
    np.testing.assert_allclose(jvp_can, np.asarray(jvp_leg(v)), rtol=0.0, atol=1e-11)


# ---------------------------------------------------------------------------
# 2. End-to-end Newton solve vs the Fortran reference and the legacy loop
# ---------------------------------------------------------------------------


def test_solve_phi1_matches_fortran_reference() -> None:
    """Conserves test_full_system_newton_krylov.py: state vs Fortran, atol 5e-8."""
    res = solve_phi1(_INPUT, tol=1e-9)
    assert res.converged
    assert float(res.residual_norm) < 1e-9
    np.testing.assert_allclose(np.asarray(res.x), _fortran_state(), rtol=0.0, atol=5e-8)


def test_solve_phi1_phi1hat_matches_fortran_reference() -> None:
    op = _canonical_op()
    res = solve_phi1(_INPUT, tol=1e-9)
    x_ref = _fortran_state()
    phi1_ref = x_ref[op.f_size : op.f_size + op.n_theta * op.n_zeta].reshape((op.n_theta, op.n_zeta))
    np.testing.assert_allclose(np.asarray(res.phi1_hat), phi1_ref, rtol=0.0, atol=5e-8)


# ---------------------------------------------------------------------------
# 3. Convergence + warm-start
# ---------------------------------------------------------------------------


def test_newton_converges_below_tol() -> None:
    res = solve_phi1(_INPUT, tol=1e-9)
    assert res.converged
    assert float(res.residual_norm) < 1e-9
    # Quadratic Newton convergence: the residual norms strictly decrease.
    norms = res.residual_norms
    assert all(norms[i + 1] < norms[i] for i in range(len(norms) - 1))


def test_warm_start_uses_fewer_iterations_than_cold() -> None:
    op = _canonical_op()
    cold = solve_phi1(op, tol=1e-9)
    assert cold.n_newton >= 2 and cold.inner_iterations_total > 0
    # Warm-starting from the solved state converges immediately: fewer Newton
    # iterations and fewer inner Krylov iterations.
    warm = solve_phi1(op, tol=1e-9, x0=cold.x)
    assert warm.n_newton < cold.n_newton
    assert (warm.inner_iterations_total or 0) < (cold.inner_iterations_total or 0)


# ---------------------------------------------------------------------------
# 4. Differentiability (implicit function theorem) vs finite differences
# ---------------------------------------------------------------------------


def test_phi1_output_gradient_matches_finite_difference() -> None:
    op0 = operator_from_input(_INPUT)
    dt0 = float(np.asarray(op0.dt_hat_dpsi_hat)[0])

    def fsabjhat(p: float) -> jnp.ndarray:
        op = replace(op0, dt_hat_dpsi_hat=jnp.asarray([p], dtype=jnp.float64))
        x = phi1_state(op, tol=1e-13)
        layout, vgrid, surface, species = operator_containers(op)
        table = rhsmode1_moments(
            layout, vgrid, surface, species, x,
            delta=op.delta, alpha=op.alpha, phi1_from_state=True,
        )  # fmt: skip
        return jnp.reshape(table["FSABjHat"], ())

    grad = float(jax.grad(fsabjhat)(dt0))
    h = 1e-4 * max(1.0, abs(dt0))
    fd = float((fsabjhat(dt0 + h) - fsabjhat(dt0 - h)) / (2.0 * h))
    np.testing.assert_allclose(grad, fd, rtol=1e-4, atol=1e-12)


# ---------------------------------------------------------------------------
# 5. run_profile routing + h5 output
# ---------------------------------------------------------------------------


def test_run_profile_routes_phi1_to_canonical() -> None:
    from sfincs_jax.run import run_profile

    run = run_profile(_INPUT, tol=1e-9, emit=None)
    assert run.solve_result.method == "phi1_newton_krylov"
    assert run.solve_result.converged
    np.testing.assert_allclose(run.state_vector, _fortran_state(), rtol=0.0, atol=5e-8)


# The canonical writer now emits every Phi1-dependent physics field the legacy
# writer produces: the electric-drift (vE/vE0) and total-drift (vd/vd1) flux
# families with their psiN/rHat/rN variants, the heat withoutPhi1 flux, the
# dPhi1Hat gradients, the lambda multiplier, the Phi1 scalar metadata
# (quasineutralityOption, readExternalPhi1, adiabatic{Z,NHat,THat,MHat}), and the
# Newton metadata (didNonlinearCalculationConverge, linearSolver{Method,
# RequestedMethod,ResidualNorm}).  The only residual fields are the legacy
# inner-linear-solver ACCEPTANCE diagnostics emitted solely on the legacy dense
# solve path (scheme5 deck): they describe the legacy GMRES/dense acceptance test
# (target = solverTol*||rhs||, ratio = ||r||/target, accepted/criterion flags),
# which has no analogue in the canonical Newton-Krylov solve.  The canonical
# solve reports only nonlinear convergence + the converged residual norm (both
# emitted); linearSolverResidualTargetRatio in particular cannot match (it is
# the legacy dense solve's ||r||/target).
_KNOWN_MISSING_PHI1_H5 = frozenset(
    {"linearSolverAccepted", "linearSolverAcceptanceCriterion", "linearSolverConverged",
     "linearSolverResidualTarget", "linearSolverResidualTargetRatio",
     "linearSolverTrueResidualConverged"}
)  # fmt: skip


def test_h5_phi1_fields_vs_legacy_writer(tmp_path: Path) -> None:
    from sfincs_jax.io import read_sfincs_h5, write_sfincs_jax_output_h5
    from sfincs_jax.run import run_profile

    leg_path = tmp_path / "legacy.h5"
    write_sfincs_jax_output_h5(input_namelist=_INPUT, output_path=leg_path, compute_solution=True, verbose=False)
    leg = read_sfincs_h5(leg_path)

    can_path = tmp_path / "canon.h5"
    run_profile(_INPUT, tol=1e-9, out_path=can_path, emit=None)
    can = read_sfincs_h5(can_path)

    # Phi1Hat is emitted and matches the legacy converged (last-iteration) field.
    assert "Phi1Hat" in can
    np.testing.assert_allclose(
        np.asarray(can["Phi1Hat"])[..., -1], np.asarray(leg["Phi1Hat"])[..., -1], rtol=0.0, atol=5e-8
    )

    # Every legacy field the canonical writer omits is a documented known-missing.
    missing = set(leg.keys()) - set(can.keys()) - {"Phi1Hat"}
    unexpected = missing - _KNOWN_MISSING_PHI1_H5
    assert not unexpected, f"canonical writer unexpectedly omits Phi1 fields: {sorted(unexpected)}"

    # Metadata whose value legitimately differs: the canonical Newton records
    # the converged state (NIterations=1); the legacy in-process writer stored
    # every accepted Newton iterate (here 3), and wall-clock timings differ.
    skip_value = {"NIterations", "elapsed time (s)"}

    # Shared numeric fields with matching shape agree (base/geometry/scalars and
    # the last-iteration slice of the RHSMode=1 moments).
    for key in sorted(set(leg.keys()) & set(can.keys())):
        if key in skip_value:
            continue
        lv, cv = leg[key], can[key]
        if isinstance(lv, (str, bytes)) or isinstance(cv, (str, bytes)):
            continue
        lv = np.asarray(lv, dtype=np.float64)
        cv = np.asarray(cv, dtype=np.float64)
        if lv.shape == cv.shape:
            np.testing.assert_allclose(cv, lv, rtol=0.0, atol=5e-8, err_msg=f"field {key}")
        elif lv.ndim == cv.ndim and lv.shape[:-1] == cv.shape[:-1] and cv.shape[-1] == 1:
            # Iteration axis: canonical stores the converged state (NIterations=1),
            # legacy stored the whole Newton history; compare the converged slice.
            np.testing.assert_allclose(cv[..., 0], lv[..., -1], rtol=0.0, atol=5e-8, err_msg=f"field {key}")


# Per-deck parity budgets for the Phi1-derived output families.  Each new family
# is an EXACT algebraic function of the converged (Phi1, f) state, so the
# canonical-vs-legacy parity is inherited from the two solvers' state parity:
#   * scheme5 is the LINEAR Phi1 deck (1 Newton step): state parity ~3e-16 here,
#     so both tiers assert tight.
#   * inKinetic / inCollision are NONLINEAR: state parity is the documented
#     Phi1Hat bound (measured ~5e-12 and ~1e-9 here).  The O(1e-8) flux families
#     land far below that (tight f-parity, small fluxes); the Phi1-scale
#     gradients (dPhi1Hat) and the near-zero multiplier (lambda) track Phi1Hat.
# ``state`` covers Phi1-scale/near-zero fields; ``flux`` covers the O(1e-8)
# electric-/total-drift flux families (compared in absolute terms because the
# vd/vd1/momentum components pass through zero, making relative error ill-posed).
_PHI1_H5_DECKS = {
    "pas_1species_PAS_noEr_tiny_scheme5_withPhi1_linear": {"state": 1e-12, "flux": 1e-13},
    "pas_1species_PAS_noEr_tiny_withPhi1_inKinetic_linear": {"state": 1e-9, "flux": 1e-12},
    "fp_1species_FPCollisions_noEr_tiny_withPhi1_inCollision": {"state": 5e-8, "flux": 1e-10},
}

_COORDS = ("psiHat", "psiN", "rHat", "rN")
_FLUX_FAMILY_KEYS = (
    [f"{fam}Flux_{drift}_{c}"
     for fam in ("particle", "heat", "momentum")
     for drift in ("vE", "vE0", "vd", "vd1") for c in _COORDS]
    + [f"heatFlux_withoutPhi1_{c}" for c in _COORDS]
    + [f"{fam}FluxBeforeSurfaceIntegral_{d}"
       for fam in ("particle", "heat", "momentum") for d in ("vE", "vE0")]
)  # fmt: skip
_STATE_FAMILY_KEYS = ("dPhi1Hatdtheta", "dPhi1Hatdzeta", "lambda")


def _converged_slice_pair(cv: object, lv: object) -> tuple[np.ndarray, np.ndarray]:
    """(canonical converged, legacy last-iteration) numeric slices for one field."""
    cva = np.asarray(cv, dtype=np.float64)
    lva = np.asarray(lv, dtype=np.float64)
    if cva.shape == lva.shape:
        return cva, lva
    assert cva.ndim == lva.ndim and cva.shape[:-1] == lva.shape[:-1] and cva.shape[-1] == 1
    return cva[..., 0], lva[..., -1]


@pytest.mark.parametrize("stem", sorted(_PHI1_H5_DECKS))
def test_h5_phi1_new_families_vs_legacy_writer(stem: str, tmp_path: Path) -> None:
    """Value parity of the newly-emitted Phi1 families vs legacy, all three decks.

    Covers the electric-drift (vE/vE0) and total-drift (vd/vd1) flux families and
    their psiN/rHat/rN variants, the heat withoutPhi1 flux, the vE/vE0
    BeforeSurfaceIntegral fields, the dPhi1Hat gradients, the lambda multiplier,
    the Phi1 scalar metadata, and the Newton metadata.
    """
    from sfincs_jax.io import read_sfincs_h5, write_sfincs_jax_output_h5
    from sfincs_jax.run import run_profile

    deck = _REF / f"{stem}.input.namelist"
    tol = _PHI1_H5_DECKS[stem]

    leg_path = tmp_path / "legacy.h5"
    write_sfincs_jax_output_h5(input_namelist=deck, output_path=leg_path, compute_solution=True, verbose=False)
    leg = read_sfincs_h5(leg_path)
    can_path = tmp_path / "canon.h5"
    run_profile(deck, tol=1e-9, out_path=can_path, emit=None)
    can = read_sfincs_h5(can_path)

    # The only legacy fields the canonical writer omits are documented.
    missing = set(leg.keys()) - set(can.keys()) - {"Phi1Hat"}
    assert missing <= _KNOWN_MISSING_PHI1_H5, f"unexpected missing: {sorted(missing - _KNOWN_MISSING_PHI1_H5)}"

    # Electric-/total-drift flux families + heat withoutPhi1 (O(1e-8) budget).
    seen_scale = 0.0
    for key in _FLUX_FAMILY_KEYS:
        assert key in can, f"{key} not emitted"
        assert key in leg, f"{key} absent from legacy for {stem}"
        cvs, lvs = _converged_slice_pair(can[key], leg[key])
        np.testing.assert_allclose(cvs, lvs, rtol=0.0, atol=tol["flux"], err_msg=f"{stem}:{key}")
        seen_scale = max(seen_scale, float(np.max(np.abs(lvs))))
    # Guard against a trivial all-zero pass: the particle/heat vE/vd fluxes are non-zero.
    assert seen_scale > 1e-12, f"{stem}: flux families are unexpectedly ~0 (scale={seen_scale:.2e})"

    # dPhi1Hat gradients and the <Phi1>=0 multiplier track the Phi1 state parity
    # (dPhi1Hat is the spectral theta/zeta derivative of the converged Phi1Hat).
    for key in _STATE_FAMILY_KEYS:
        assert key in can, f"{key} not emitted"
        assert key in leg, f"{key} absent from legacy for {stem}"
        cvs, lvs = _converged_slice_pair(can[key], leg[key])
        np.testing.assert_allclose(cvs, lvs, rtol=0.0, atol=tol["state"], err_msg=f"{stem}:{key}")

    # linearSolverResidualNorm is a converged-solve residual (legacy stores the
    # inner linear residual, canonical the Newton residual): both are ~0, well
    # below any physical scale.  Assert convergence, not tight value parity.
    assert "linearSolverResidualNorm" in can and "linearSolverResidualNorm" in leg
    assert float(np.asarray(can["linearSolverResidualNorm"])) < 1e-6
    assert float(np.asarray(leg["linearSolverResidualNorm"])) < 1e-6

    # Phi1 scalar metadata: exact integer/logical parity.
    assert int(np.asarray(can["quasineutralityOption"])) == int(np.asarray(leg["quasineutralityOption"]))
    assert int(np.asarray(can["readExternalPhi1"])) == int(np.asarray(leg["readExternalPhi1"]))
    assert int(np.asarray(can["didNonlinearCalculationConverge"])) == 1
    for key in ("adiabaticZ", "adiabaticNHat", "adiabaticTHat", "adiabaticMHat"):
        np.testing.assert_allclose(
            float(np.asarray(can[key])), float(np.asarray(leg[key])), rtol=0.0, atol=1e-12, err_msg=f"{stem}:{key}"
        )
    # Method-name metadata is present (string value is a solver label, not physics).
    for key in ("linearSolverMethod", "linearSolverRequestedMethod"):
        assert key in can and key in leg


# ---------------------------------------------------------------------------
# 6. includePhi1InCollisionOperator: canonical parity + Fortran end-to-end
#
# Conserves the collision-Phi1 physics of the retired legacy owner
# (profile_phi1_newton / FokkerPlanckV3Phi1Operator): the collisional densities
# become poloidally varying, n_pol = nHat*exp(-Z*alpha*Phi1Hat/THat).  The
# fixture activates collisionOperator=0 (full FP) + includePhi1 +
# includePhi1InKineticEquation + includePhi1InCollisionOperator, and ships a
# Fortran state-vector reference.
# ---------------------------------------------------------------------------

_COLL_FIXTURE = "fp_1species_FPCollisions_noEr_tiny_withPhi1_inCollision"
_COLL_INPUT = _REF / f"{_COLL_FIXTURE}.input.namelist"
_COLL_STATEVEC = _REF / f"{_COLL_FIXTURE}.stateVector.petscbin"


def _coll_canonical_op():
    return kinetic_operator_from_namelist(read_sfincs_input(_COLL_INPUT))


def _coll_legacy_op():
    return full_system_operator_from_namelist(nml=read_sfincs_input(_COLL_INPUT))


def _coll_fortran_state() -> np.ndarray:
    return np.asarray(read_petsc_vec(_COLL_STATEVEC).values, dtype=np.float64)


def test_collision_phi1_operator_is_canonical() -> None:
    op = _coll_canonical_op()
    assert op.include_phi1 and op.include_phi1_in_kinetic
    assert op.fp_phi1 is not None and op.fp is None


def test_collision_phi1_residual_elementwise_parity_vs_legacy() -> None:
    op, leg = _coll_canonical_op(), _coll_legacy_op()
    rng = np.random.default_rng(0)
    for _ in range(3):
        x = jnp.asarray(rng.standard_normal(op.total_size))
        r_can = np.asarray(op.residual_phi1(x))
        r_leg = np.asarray(residual_v3_full_system(leg, x))
        np.testing.assert_allclose(r_can, r_leg, rtol=0.0, atol=1e-12)


def test_collision_phi1_operator_and_rhs_elementwise_parity_vs_legacy() -> None:
    op, leg = _coll_canonical_op(), _coll_legacy_op()
    np.testing.assert_allclose(
        np.asarray(op.rhs_phi1()), np.asarray(rhs_v3_full_system(leg)), rtol=0.0, atol=1e-12
    )
    rng = np.random.default_rng(1)
    x = jnp.asarray(rng.standard_normal(op.total_size))
    phi1 = x[op.f_size : op.f_size + op.n_theta * op.n_zeta].reshape((op.n_theta, op.n_zeta))
    a_can = np.asarray(replace(op, phi1_hat_base=phi1)._apply_phi1_operator(x))
    a_leg = np.asarray(
        apply_v3_full_system_operator(replace(leg, phi1_hat_base=phi1), x, include_jacobian_terms=False)
    )
    np.testing.assert_allclose(a_can, a_leg, rtol=0.0, atol=1e-12)


def test_collision_phi1_jacobian_matvec_parity_vs_legacy_linearize() -> None:
    op, leg = _coll_canonical_op(), _coll_legacy_op()
    x0 = jnp.asarray(_coll_fortran_state())
    op_lin = replace(op, phi1_lin_state=x0)
    rng = np.random.default_rng(2)
    v = jnp.asarray(rng.standard_normal(op.total_size))
    jvp_can = np.asarray(op_lin.apply(v))
    _, jvp_leg = jax.linearize(lambda xx: residual_v3_full_system(leg, xx), x0)
    np.testing.assert_allclose(jvp_can, np.asarray(jvp_leg(v)), rtol=0.0, atol=1e-11)


def test_collision_phi1_solve_matches_fortran_reference() -> None:
    """Conserves the collision-Phi1 Fortran-parity end-to-end (atol 5e-8)."""
    res = solve_phi1(_COLL_INPUT, tol=1e-9)
    assert res.converged
    assert float(res.residual_norm) < 1e-9
    np.testing.assert_allclose(np.asarray(res.x), _coll_fortran_state(), rtol=0.0, atol=5e-8)


def test_collision_phi1_run_profile_routes_to_canonical() -> None:
    from sfincs_jax.run import run_profile

    run = run_profile(_COLL_INPUT, tol=1e-9, emit=None)
    assert run.solve_result.method == "phi1_newton_krylov"
    assert run.solve_result.converged
    np.testing.assert_allclose(run.state_vector, _coll_fortran_state(), rtol=0.0, atol=5e-8)


def test_collision_phi1_cli_predicate_routes_to_canonical() -> None:
    """The CLI dispatch predicate no longer defers includePhi1InCollisionOperator.

    The canonical writer emits the vE/vd flux families for Phi1 runs, so the
    ``write-output`` CLI routes the collision-Phi1 deck through the canonical
    stack (``deck_requires_legacy_pipeline`` returns ``None``) rather than the
    legacy outputs writer.
    """
    from sfincs_jax.cli import deck_requires_legacy_pipeline
    from sfincs_jax.inputs import read_sfincs_input

    assert deck_requires_legacy_pipeline(read_sfincs_input(_COLL_INPUT)) is None
