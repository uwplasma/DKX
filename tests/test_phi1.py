"""Canonical Phi1 / quasineutrality slice: parity, convergence, gradient, output.

Covers the ``includePhi1`` vertical slice consolidated into
:mod:`dkx.drift_kinetic` (the quasineutrality block, the ``<Phi1>=0``
lambda row, and the Phi1-in-kinetic-equation coupling for
``quasineutralityOption`` 1/2) and its nonlinear Newton solve in
:mod:`dkx.phi1`.

Parity oracles: the frozen Fortran reference fixtures (residual/stateVector
petscbin + whichMatrix_3 Jacobians for the linear decks, sfincsOutput.h5
goldens for the output families).  This conserves the Fortran-parity assertion
of the retired ``tests/test_full_system_newton_krylov.py`` (same fixture,
atol 5e-8).

Section 6 covers the ``includePhi1InCollisionOperator`` physics (the poloidally
varying Fokker-Planck collision operator): canonical residual parity vs the
frozen Fortran residual and a Fortran state-vector end-to-end check.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# x64 is enabled on import of any dkx operator module (drift_kinetic).
from dkx.drift_kinetic import kinetic_operator_from_namelist
from dkx.moments import rhsmode1_moments
from dkx.namelist import read_sfincs_input
from dkx.phi1 import operator_from_input, phi1_state, solve_phi1
from dkx.validation.fortran import read_petsc_vec
from dkx.writer import operator_containers

_REF = Path(__file__).parent / "ref"
# The Fortran-parity fixture (quasineutralityOption=2, includePhi1InKineticEquation,
# adiabatic species) reused from the retired test_full_system_newton_krylov.py.
_FIXTURE = "pas_1species_PAS_noEr_tiny_withPhi1_inKinetic_linear"
_INPUT = _REF / f"{_FIXTURE}.input.namelist"
_STATEVEC = _REF / f"{_FIXTURE}.stateVector.petscbin"


def _canonical_op():
    return kinetic_operator_from_namelist(read_sfincs_input(_INPUT))


def _fortran_state() -> np.ndarray:
    return np.asarray(read_petsc_vec(_STATEVEC).values, dtype=np.float64)


# ---------------------------------------------------------------------------
# 1. Operator / residual element-wise parity vs the frozen Fortran fixtures
# ---------------------------------------------------------------------------


def test_operator_layout_is_phi1() -> None:
    op = _canonical_op()
    assert op.include_phi1 and op.include_phi1_in_kinetic
    assert op.phi1_size == op.n_theta * op.n_zeta + 1
    assert op.total_size == op.f_size + op.phi1_size + op.extra_size


def test_residual_matches_fortran_at_reference_state() -> None:
    """residual_phi1 at the recorded Newton state equals the frozen Fortran residual."""
    op = _canonical_op()
    x0 = jnp.asarray(_fortran_state())
    r_ref = np.asarray(read_petsc_vec(_REF / f"{_FIXTURE}.residual.petscbin").values)
    r = np.asarray(op.residual_phi1(x0))
    np.testing.assert_allclose(r, r_ref, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize(
    "stem",
    (
        "pas_1species_PAS_noEr_tiny_scheme5_withPhi1_linear",
        "pas_1species_PAS_noEr_tiny_withPhi1_linear",
    ),
)
def test_jacobian_matvec_matches_fortran_whichmatrix3_linear_decks(stem: str) -> None:
    """The Phi1 Jacobian JVP equals the frozen Fortran whichMatrix=3 on linear decks."""
    from scipy.sparse import csr_matrix

    from dkx.validation.fortran import read_petsc_mat_aij

    op = kinetic_operator_from_namelist(read_sfincs_input(_REF / f"{stem}.input.namelist"))
    x0 = jnp.asarray(read_petsc_vec(_REF / f"{stem}.stateVector.petscbin").values)
    a = read_petsc_mat_aij(_REF / f"{stem}.whichMatrix_3.petscbin")
    mat = csr_matrix((a.data, a.col_ind, a.row_ptr), shape=a.shape)
    op_lin = replace(op, phi1_lin_state=x0)
    rng = np.random.default_rng(2)
    v = rng.standard_normal(op.total_size)
    np.testing.assert_allclose(
        np.asarray(op_lin.apply(jnp.asarray(v))), mat.dot(v), rtol=0.0, atol=1e-11
    )


def test_jacobian_matvec_matches_autodiff_linearize() -> None:
    """``apply`` (the hand-built JVP used by solve.solve) matches jax.linearize."""
    op = _canonical_op()
    x0 = jnp.asarray(_fortran_state())
    op_lin = replace(op, phi1_lin_state=x0)
    rng = np.random.default_rng(2)
    v = jnp.asarray(rng.standard_normal(op.total_size))
    jvp_can = np.asarray(op_lin.apply(v))
    _, jvp_ad = jax.linearize(op.residual_phi1, x0)
    np.testing.assert_allclose(jvp_can, np.asarray(jvp_ad(v)), rtol=0.0, atol=1e-11)


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


def test_preconditioner_reduces_inner_iterations_and_preserves_answer() -> None:
    """The Phi1-aware coarse preconditioner cuts inner Krylov iterations exactly.

    Enabling the bordered-Schur coarse preconditioner
    (:func:`dkx.solve.build_coarse_preconditioner`, which
    Schur-eliminates the Phi1 quasineutrality border in addition to the
    constraint border) must take strictly fewer total inner Krylov iterations
    than the unpreconditioned full-restart solve, and converge to the *same*
    state -- both solve the same Newton residual to the same tolerance, so the
    preconditioner only changes the Krylov path, never the answer.
    """
    op = _canonical_op()
    unprec = solve_phi1(op, tol=1e-9, use_preconditioner=False)
    prec = solve_phi1(op, tol=1e-9, use_preconditioner=True)
    assert unprec.converged and prec.converged
    assert unprec.inner_iterations_total and prec.inner_iterations_total
    # Preconditioned inner solve converges in far fewer Krylov iterations.
    assert prec.inner_iterations_total < unprec.inner_iterations_total
    # ... to the identical converged state (both hit tol on the same residual).
    np.testing.assert_allclose(np.asarray(prec.x), np.asarray(unprec.x), rtol=0.0, atol=1e-8)
    # ... which is still the Fortran reference.
    np.testing.assert_allclose(np.asarray(prec.x), _fortran_state(), rtol=0.0, atol=5e-8)


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
    from dkx.run import run_profile

    run = run_profile(_INPUT, tol=1e-9, emit=None)
    assert run.solve_result.method == "phi1_newton_krylov"
    assert run.solve_result.converged
    np.testing.assert_allclose(run.state_vector, _fortran_state(), rtol=0.0, atol=5e-8)


def test_h5_phi1_fields_vs_fortran_golden(tmp_path: Path) -> None:
    from dkx.compare import compare_sfincs_outputs
    from dkx.io import read_sfincs_h5
    from dkx.run import run_profile

    fortran_path = _REF / f"{_FIXTURE}.sfincsOutput.h5"
    fort = read_sfincs_h5(fortran_path)

    can_path = tmp_path / "canon.h5"
    run_profile(_INPUT, tol=1e-9, out_path=can_path, emit=None)
    can = read_sfincs_h5(can_path)

    # Phi1Hat is emitted and matches the Fortran converged (last-iteration) field.
    assert "Phi1Hat" in can
    np.testing.assert_allclose(
        np.asarray(can["Phi1Hat"])[..., -1], np.asarray(fort["Phi1Hat"])[..., -1], rtol=0.0, atol=5e-8
    )

    # Full physics-field parity via the release comparator against the golden.
    results = compare_sfincs_outputs(a_path=fortran_path, b_path=can_path, rtol=1e-8, atol=5e-8)
    failures = [r for r in results if not r.ok]
    assert not failures, f"canonical vs Fortran golden mismatches: {[f.key for f in failures]}"


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
    """(canonical, Fortran) converged-iterate slices for one per-iteration field.

    Every compared family carries a trailing ``NIterations`` axis; the value
    budgets below are calibrated for the CONVERGED state, so compare the last
    iterate on both sides (mid-Newton iterates are solver-trajectory detail,
    pinned separately at 5e-8 by the write-output end-to-end test).
    """
    cva = np.asarray(cv, dtype=np.float64)
    lva = np.asarray(lv, dtype=np.float64)
    assert cva.ndim == lva.ndim and cva.shape[:-1] == lva.shape[:-1]
    return cva[..., -1], lva[..., -1]


@pytest.mark.parametrize("stem", sorted(_PHI1_H5_DECKS))
def test_h5_phi1_new_families_vs_fortran_golden(stem: str, tmp_path: Path) -> None:
    """Value parity of the Phi1-derived output families vs the Fortran goldens.

    Covers the electric-drift (vE/vE0) and total-drift (vd/vd1) flux families and
    their psiN/rHat/rN variants, the heat withoutPhi1 flux, the vE/vE0
    BeforeSurfaceIntegral fields, the dPhi1Hat gradients, the lambda multiplier,
    the Phi1 scalar metadata, and the Newton metadata.
    """
    from dkx.io import read_sfincs_h5
    from dkx.run import run_profile

    deck = _REF / f"{stem}.input.namelist"
    tol = _PHI1_H5_DECKS[stem]

    fort = read_sfincs_h5(_REF / f"{stem}.sfincsOutput.h5")
    can_path = tmp_path / "canon.h5"
    run_profile(deck, tol=1e-9, out_path=can_path, emit=None)
    can = read_sfincs_h5(can_path)

    # Electric-/total-drift flux families + heat withoutPhi1 (O(1e-8) budget).
    seen_scale = 0.0
    for key in _FLUX_FAMILY_KEYS:
        assert key in can, f"{key} not emitted"
        if key not in fort:
            continue
        cvs, fvs = _converged_slice_pair(can[key], fort[key])
        np.testing.assert_allclose(cvs, fvs, rtol=0.0, atol=tol["flux"], err_msg=f"{stem}:{key}")
        seen_scale = max(seen_scale, float(np.max(np.abs(fvs))))
    # Guard against a trivial all-zero pass: the particle/heat vE/vd fluxes are non-zero.
    assert seen_scale > 1e-12, f"{stem}: flux families are unexpectedly ~0 (scale={seen_scale:.2e})"

    # dPhi1Hat gradients and the <Phi1>=0 multiplier track the Phi1 state parity
    # (dPhi1Hat is the spectral theta/zeta derivative of the converged Phi1Hat).
    for key in _STATE_FAMILY_KEYS:
        assert key in can, f"{key} not emitted"
        if key not in fort:
            continue
        cvs, fvs = _converged_slice_pair(can[key], fort[key])
        np.testing.assert_allclose(cvs, fvs, rtol=0.0, atol=tol["state"], err_msg=f"{stem}:{key}")

    # Canonical Newton metadata: converged, with a small recorded residual.
    assert "linearSolverResidualNorm" in can
    assert float(np.asarray(can["linearSolverResidualNorm"])) < 1e-6
    assert int(np.asarray(can["didNonlinearCalculationConverge"])) == 1

    # Phi1 scalar metadata: exact integer/logical parity with the Fortran golden.
    assert int(np.asarray(can["quasineutralityOption"])) == int(np.asarray(fort["quasineutralityOption"]))
    assert int(np.asarray(can["readExternalPhi1"])) == int(np.asarray(fort["readExternalPhi1"]))
    for key in ("adiabaticZ", "adiabaticNHat", "adiabaticTHat", "adiabaticMHat"):
        if key not in fort:
            continue
        np.testing.assert_allclose(
            float(np.asarray(can[key])), float(np.asarray(fort[key])), rtol=0.0, atol=1e-12, err_msg=f"{stem}:{key}"
        )
    # Method-name metadata is present (string value is a solver label, not physics).
    for key in ("linearSolverMethod", "linearSolverRequestedMethod"):
        assert key in can


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


def _coll_fortran_state() -> np.ndarray:
    return np.asarray(read_petsc_vec(_COLL_STATEVEC).values, dtype=np.float64)


def test_collision_phi1_operator_is_canonical() -> None:
    op = _coll_canonical_op()
    assert op.include_phi1 and op.include_phi1_in_kinetic
    assert op.fp_phi1 is not None and op.fp is None


def test_collision_phi1_residual_matches_fortran_at_reference_state() -> None:
    """residual_phi1 at the recorded Newton state equals the frozen Fortran residual."""
    op = _coll_canonical_op()
    x0 = jnp.asarray(_coll_fortran_state())
    r_ref = np.asarray(read_petsc_vec(_REF / f"{_COLL_FIXTURE}.residual.petscbin").values)
    np.testing.assert_allclose(np.asarray(op.residual_phi1(x0)), r_ref, rtol=0.0, atol=1e-12)


def test_collision_phi1_jacobian_matvec_matches_autodiff_linearize() -> None:
    op = _coll_canonical_op()
    x0 = jnp.asarray(_coll_fortran_state())
    op_lin = replace(op, phi1_lin_state=x0)
    rng = np.random.default_rng(2)
    v = jnp.asarray(rng.standard_normal(op.total_size))
    jvp_can = np.asarray(op_lin.apply(v))
    _, jvp_ad = jax.linearize(op.residual_phi1, x0)
    np.testing.assert_allclose(jvp_can, np.asarray(jvp_ad(v)), rtol=0.0, atol=1e-11)


def test_collision_phi1_solve_matches_fortran_reference() -> None:
    """Conserves the collision-Phi1 Fortran-parity end-to-end (atol 5e-8)."""
    res = solve_phi1(_COLL_INPUT, tol=1e-9)
    assert res.converged
    assert float(res.residual_norm) < 1e-9
    np.testing.assert_allclose(np.asarray(res.x), _coll_fortran_state(), rtol=0.0, atol=5e-8)


def test_collision_phi1_run_profile_routes_to_canonical() -> None:
    from dkx.run import run_profile

    run = run_profile(_COLL_INPUT, tol=1e-9, emit=None)
    assert run.solve_result.method == "phi1_newton_krylov"
    assert run.solve_result.converged
    np.testing.assert_allclose(run.state_vector, _coll_fortran_state(), rtol=0.0, atol=5e-8)


# ---------------------------------------------------------------------------
# 7. readExternalPhi1: FIXED external Phi1 field, LINEAR f-only solve
#
# With includePhi1=.true. and readExternalPhi1=.true. SFINCS reads a fixed
# Phi1(theta,zeta) from an external HDF5 file instead of solving quasineutrality:
# the DKE is LINEAR again (state = f only, no QN block, no lambda row), the fixed
# field enters the same Phi1-in-kinetic terms, and the solve is a single linear
# solve (no Newton).  The external field here is the sfincsOutput.h5 of the
# self-consistent inKinetic deck (same grid -> no interpolation), so the Fortran
# golden's Phi1Hat equals that field exactly.
# ---------------------------------------------------------------------------

_REXT = "pas_1species_PAS_noEr_tiny_readExternalPhi1"
_REXT_DECK = _REF / f"{_REXT}.input.namelist"
_REXT_FIELD = _REF / f"{_REXT}.externalPhi1.h5"
_REXT_GOLDEN = _REF / f"{_REXT}.sfincsOutput.h5"
_REXT_STATEVEC = _REF / f"{_REXT}.stateVector.petscbin"


def _rext_run_dir(tmp_path: Path) -> Path:
    """Deck + external field laid out as SFINCS runs it (field named externalPhi1.h5)."""
    import shutil

    shutil.copy(_REXT_DECK, tmp_path / "input.namelist")
    shutil.copy(_REXT_FIELD, tmp_path / "externalPhi1.h5")
    return tmp_path / "input.namelist"


def test_read_external_phi1_operator_is_linear_f_only(tmp_path: Path) -> None:
    """The operator keeps the f-only layout (no QN block / lambda) and stays linear."""
    op = kinetic_operator_from_namelist(read_sfincs_input(_rext_run_dir(tmp_path)))
    assert op.external_phi1_hat is not None
    assert not op.include_phi1  # no QN block, no Phi1 unknown, no lambda row
    assert op.include_phi1_in_kinetic
    assert op.phi1_size == 0
    # f-only + constraint sources (constraintScheme 2): f_size + n_species*n_x.
    assert int(op.total_size) == int(op.f_size) + op.n_species * op.n_x
    # apply is genuinely linear: apply(0) == 0.
    y0 = np.asarray(op.apply(jnp.zeros((op.total_size,), dtype=jnp.float64)))
    assert float(np.max(np.abs(y0))) == 0.0


def test_read_external_phi1_state_matches_fortran(tmp_path: Path) -> None:
    """Canonical linear solve state vs the Fortran readExternalPhi1 golden state."""
    from dkx.solve import solve

    op = kinetic_operator_from_namelist(read_sfincs_input(_rext_run_dir(tmp_path)))
    res = solve(op, op.rhs(), method="auto", tol=1e-11)
    assert res.converged
    x = np.asarray(res.x, dtype=np.float64).reshape((-1,))
    fort = np.asarray(read_petsc_vec(_REXT_STATEVEC).values, dtype=np.float64)
    worst = float(np.max(np.abs(x - fort)))
    assert worst < 5e-8, f"worst state |canonical-Fortran| = {worst:.3e}"


def test_read_external_phi1_run_profile_is_linear(tmp_path: Path) -> None:
    """run_profile solves readExternalPhi1 with a single LINEAR solve (no Newton)."""
    from dkx.run import run_profile

    deck = _rext_run_dir(tmp_path)
    run = run_profile(deck, tol=1e-11, emit=None)
    assert run.solve_result.method != "phi1_newton_krylov"
    assert run.solve_result.converged
    np.testing.assert_allclose(
        run.state_vector, np.asarray(read_petsc_vec(_REXT_STATEVEC).values, dtype=np.float64),
        rtol=0.0, atol=5e-8,
    )  # fmt: skip


def test_read_external_phi1_uses_external_field(tmp_path: Path) -> None:
    """The emitted Phi1Hat equals the external field's last-iteration slice."""
    from dkx.io import read_sfincs_h5
    from dkx.run import run_profile

    deck = _rext_run_dir(tmp_path)
    can_path = tmp_path / "canon.h5"
    run_profile(deck, tol=1e-11, out_path=can_path, emit=None)
    can = read_sfincs_h5(can_path)
    ext = read_sfincs_h5(_REXT_FIELD)
    assert "Phi1Hat" in can and "lambda" not in can  # fixed field, no lambda row
    np.testing.assert_allclose(
        np.asarray(can["Phi1Hat"])[..., -1], np.asarray(ext["Phi1Hat"])[..., -1],
        rtol=0.0, atol=1e-12,
    )  # fmt: skip


def test_read_external_phi1_output_matches_fortran(tmp_path: Path) -> None:
    """run_profile output h5 vs the Fortran golden: state, fluxes, moments, Phi1Hat.

    Every shared numeric field agrees to the sibling tiny-deck budget (atol 5e-8);
    the core neoclassical/flux/moment/Phi1 fields agree far tighter (~1e-10).
    """
    from dkx.io import read_sfincs_h5
    from dkx.run import run_profile

    deck = _rext_run_dir(tmp_path)
    can_path = tmp_path / "canon.h5"
    run_profile(deck, tol=1e-11, out_path=can_path, emit=None)
    can = read_sfincs_h5(can_path)
    gold = read_sfincs_h5(_REXT_GOLDEN)

    jax_only_metadata = {
        "linearSolverMethod",
        "linearSolverRequestedMethod",
        "linearSolverResidualNorm",
    }
    assert set(gold) <= set(can), f"golden-only={sorted(set(gold) - set(can))}"
    assert set(can) - set(gold) <= jax_only_metadata, (
        f"canonical-only={sorted(set(can) - set(gold) - jax_only_metadata)}"
    )

    skip_value = {"elapsed time (s)"}  # wall-clock, not a physics field
    worst_key, worst = "", 0.0
    for key in sorted(set(gold) & set(can)):
        if key in skip_value:
            continue
        gv, cv = gold[key], can[key]
        if isinstance(gv, (str, bytes)) or isinstance(cv, (str, bytes)):
            continue
        ga = np.asarray(gv, dtype=np.float64)
        ca = np.asarray(cv, dtype=np.float64)
        if ga.shape != ca.shape or ga.size == 0:
            continue
        np.testing.assert_allclose(ca, ga, rtol=0.0, atol=5e-8, err_msg=f"field {key}")
        d = float(np.max(np.abs(ca - ga)))
        if d > worst:
            worst_key, worst = key, d
    # The core physics families must agree far tighter than the 5e-8 field budget.
    for key in ("Phi1Hat", "particleFlux_vm_psiHat", "heatFlux_vm_psiHat",
                "particleFlux_vd_psiHat", "FSABFlow", "densityPerturbation"):  # fmt: skip
        np.testing.assert_allclose(
            np.asarray(can[key], dtype=np.float64), np.asarray(gold[key], dtype=np.float64),
            rtol=0.0, atol=1e-10, err_msg=f"core field {key}",
        )  # fmt: skip
    assert worst < 5e-8, f"worst shared-field diff {worst:.3e} at {worst_key!r}"
