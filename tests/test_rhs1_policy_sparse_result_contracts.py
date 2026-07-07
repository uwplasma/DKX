from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp

from sfincs_jax.operators.profile_full_system import (
    RHS1StructuredFullCSRSelection,
    RHS1StructuredFullCSRSolveResult,
)
from sfincs_jax.problems.profile_policies import (
    ActiveProjectedPreconditionerAutoPolicy,
    RHS1Constraint0PETScCompatConfig,
    RHS1FullSparseRescueSetupResult,
    RHS1PostMinresPolicy,
    RHS1SparseJAXConfig,
    RHS1SparseOperatorAdmission,
    RHS1SparsePreconditionerConfig,
    RHS1SparseRescueOrdering,
    RHS1SparseRescuePolicySetup,
    RHS1Stage2RetryAdmissionDecision,
    RHS1Stage2TriggerDecision,
    RHS1SubspaceCorrectionPolicy,
)
from sfincs_jax.problems.profile_sparse_solve import (
    RHS1FullSparseRetryStageResult,
    SparsePCAutoPreflightRetryStageResult,
    SparsePCDirectTailFactorSetupResult,
    SparsePCDirectTailRescuePolicySetupResult,
    SparsePCFactorPreflightRunResult,
    SparsePCGenericBranchSetupResult,
)


class _FakeFBlockSelection:
    def to_dict(self) -> dict[str, object]:
        return {"selected": True, "reason": "unit"}


def test_rhs1_policy_dataclasses_capture_solver_routing_contracts() -> None:
    post_minres = RHS1PostMinresPolicy(
        steps_requested=2,
        alpha_clip=0.5,
        min_improvement=0.1,
    )
    subspace = RHS1SubspaceCorrectionPolicy(
        steps_requested=1,
        max_directions=3,
        max_extra_units=2,
        fsavg_lmax=1,
        angular_lmax=2,
        include_angular_residual=True,
        include_raw=False,
        alpha_clip=0.25,
        rcond=1.0e-12,
        min_improvement=0.05,
    )
    active_auto = ActiveProjectedPreconditionerAutoPolicy(
        candidates=("active_xell", "active_diag_schur"),
        candidates_requested=("auto",),
        skipped_large_fallbacks=("active_diag_schur",),
        large_fallback_size=1024,
        large_default_used=True,
        log_progress=False,
    )
    petsc_compat = RHS1Constraint0PETScCompatConfig(
        drop_tol=1.0e-3,
        fill=8.0,
        diag_pivot=0.01,
        restart=80,
        maxiter=200,
    )
    ordering = RHS1SparseRescueOrdering(
        enabled=True,
        kind_use="direct_tail",
        xblock_rescue_active=True,
        reason_size_targeted=True,
    )
    policy_setup = RHS1SparseRescuePolicySetup(
        enabled=True,
        kind_use="direct_tail",
        ordering=ordering,
        sparse_jax_est_mb=12.5,
        sparse_jax_memory_disabled_message=None,
    )
    full_rescue = RHS1FullSparseRescueSetupResult(
        policy=policy_setup,
        ordering=ordering,
        enabled=True,
        kind_use="direct_tail",
        sparse_exact_direct=True,
        sparse_exact_lu=False,
        large_cpu_sparse_rescue=False,
    )
    sparse_jax = RHS1SparseJAXConfig(max_mb=64.0, sweeps=3, omega=0.8, reg=1.0e-8)
    sparse_config = RHS1SparsePreconditionerConfig(
        precond_mode="auto",
        precond_kind="direct_tail",
        allow_nondiff=True,
        use_matvec=False,
        operator_mode="structured",
        max_size=10_000,
        pas_sparse_min=500,
        drop_tol=1.0e-8,
        drop_rel=1.0e-6,
        ilu_drop_tol=1.0e-4,
        ilu_fill=5.0,
        ilu_dense_max=1024,
        dense_cache_max=256,
    )
    sparse_admission = RHS1SparseOperatorAdmission(
        use_sparse_operator=True,
        messages=((1, "structured sparse operator admitted"),),
    )
    trigger = RHS1Stage2TriggerDecision(
        stage2_trigger=True,
        fp_force_stage2=False,
        messages=((1, "stage-2 triggered by residual ratio"),),
    )
    retry = RHS1Stage2RetryAdmissionDecision(
        run_retry=True,
        messages=((1, "stage-2 retry admitted"),),
    )

    assert post_minres.steps_requested == 2
    assert subspace.include_post_coarse is True
    assert active_auto.skipped_large_fallbacks == ("active_diag_schur",)
    assert petsc_compat.restart == 80
    assert policy_setup.ordering is ordering
    assert full_rescue.policy is policy_setup
    assert sparse_jax.sweeps == 3
    assert sparse_config.precond_kind == "direct_tail"
    assert sparse_admission.messages[0][1].startswith("structured")
    assert trigger.stage2_trigger is True
    assert retry.run_retry is True


def test_sparse_pc_stage_results_preserve_orchestration_state() -> None:
    residual_vec = jnp.asarray([1.0, -0.5], dtype=jnp.float64)
    x0 = jnp.asarray([0.1, 0.2], dtype=jnp.float64)
    factor_bundle = SimpleNamespace(kind="unit_factor")
    operator_bundle = SimpleNamespace(kind="unit_operator")

    preflight = SparsePCFactorPreflightRunResult(
        residual_before=2.0,
        residual_after=0.5,
        residual_diagnostics={"relative_residual": 0.25},
        improvement_ratio=0.25,
        target_ratio=5.0,
        passed=True,
        seed_used=True,
        residual_vec=residual_vec,
        x0_seed=x0,
    )
    retry = SparsePCAutoPreflightRetryStageResult(
        selected=True,
        attempts=({"kind": "direct_tail", "accepted": True},),
        factor_bundle_pc=factor_bundle,
        direct_tail_structured_pc_selected=True,
        direct_tail_structured_pc_reason="accepted",
        direct_tail_structured_pc_metadata={"rank": 2},
        operator_bundle_pc=operator_bundle,
        pc_factor_s=0.2,
        setup_s=0.3,
        residual_vec_current=residual_vec,
        factor_preflight_residual_after=preflight.residual_after,
        factor_preflight_residual_diagnostics=preflight.residual_diagnostics,
        factor_preflight_improvement_ratio=preflight.improvement_ratio,
        factor_preflight_target_ratio=preflight.target_ratio,
        factor_preflight_passed=preflight.passed,
        factor_preflight_seed_used=preflight.seed_used,
        x0_sparse=x0,
    )
    assert preflight.passed is True
    assert retry.factor_bundle_pc is factor_bundle
    np.testing.assert_allclose(retry.residual_vec_current, residual_vec)


def test_sparse_pc_setup_results_preserve_sparse_policy_outputs() -> None:
    residual_vec = jnp.asarray([1.0, 0.0], dtype=jnp.float64)
    factor_bundle = SimpleNamespace(kind="factor")
    materialization = SimpleNamespace(kind="materialization")
    structured_admission = SimpleNamespace(accepted=True)
    factor_policy = SimpleNamespace(enabled=True)

    full_retry = RHS1FullSparseRetryStageResult(
        result="gmres-result",
        residual_vec=residual_vec,
        dense_matrix_cache=np.eye(2),
        host_sparse_direct_used=True,
    )
    generic = SparsePCGenericBranchSetupResult(
        active_idx_np=np.asarray([0, 1], dtype=np.int32),
        active_idx_jnp=jnp.asarray([0, 1], dtype=jnp.int32),
        full_to_active_jnp=jnp.asarray([1, 2], dtype=jnp.int32),
        rhs=residual_vec,
        linear_size=2,
        reduce_full=lambda x: x[:2],
        expand_reduced=lambda x: x,
        op_pc=SimpleNamespace(name="pc"),
        pattern_source_op=SimpleNamespace(name="pattern"),
        preconditioner_operator="structured",
        fortran_reduced_xblock_min_size=10,
        fortran_reduced_sparse_pc_backend="native",
        fortran_reduced_sparse_pc_backend_reason="configured",
        pattern=SimpleNamespace(nnz=2),
        sparse_pattern_scope="active",
        pattern_build_s=0.01,
        summary=SimpleNamespace(nnz=2),
        factor_policy=factor_policy,
    )
    direct_tail = SparsePCDirectTailFactorSetupResult(
        materialization=materialization,
        direct_tail_default=True,
        direct_tail_enabled=True,
        direct_tail_built=True,
        direct_tail_error=None,
        direct_tail_operator_bundle=SimpleNamespace(kind="operator"),
        direct_tail_structured_pc_requested="auto",
        direct_tail_structured_pc_selected=True,
        direct_tail_structured_pc_reason="admitted",
        direct_tail_structured_pc_metadata={"kind": "x_ell"},
        direct_tail_structured_pc_error=None,
        direct_tail_pc_env_early="auto",
        direct_tail_direct_reduced_pmat_requested=True,
        structured_admission=structured_admission,
        direct_tail_pc_env="auto",
        direct_tail_pc_auto_default=True,
        direct_tail_fail_closed_size=256,
        direct_tail_auto_large_fail_closed=False,
        direct_tail_structured_pc_required=True,
        structured_pc_ready=True,
        direct_tail_structured_layout=SimpleNamespace(name="layout"),
        direct_tail_structured_active_indices=np.asarray([0, 1], dtype=np.int32),
        direct_tail_structured_max_nbytes=4096,
        direct_tail_support_mode_preflight_requested=True,
        direct_tail_support_mode_preflight_selected=False,
        direct_tail_support_mode_preflight_metadata=None,
        direct_tail_support_mode_preflight_error=None,
        direct_tail_structured_pc_max_mb_auto=False,
        pc_max_mb=64.0,
        pc_reg=1.0e-8,
        operator_bundle_pc=SimpleNamespace(kind="operator_pc"),
        factor_bundle_pc=factor_bundle,
        pc_factor_s=0.2,
        setup_s=0.3,
    )
    rescue = SparsePCDirectTailRescuePolicySetupResult(
        factor_bundle_pc=factor_bundle,
        direct_tail_structured_pc_selected=True,
        direct_tail_structured_pc_reason="admitted",
        direct_tail_structured_pc_metadata={"kind": "x_ell"},
        direct_tail_support_mode_preflight_requested=True,
        direct_tail_support_mode_preflight_selected=False,
        direct_tail_support_mode_preflight_metadata=None,
        direct_tail_support_mode_preflight_error=None,
        factor_preflight_policy=factor_policy,
        factor_preflight_enabled=True,
        factor_preflight_required=True,
        factor_preflight_seed_enabled=True,
        structured_pc_preflight_required_min_size=128,
        direct_tail_structured_pc_requires_preflight=True,
        direct_tail_structured_pc_kind_for_preflight="x_ell",
        direct_tail_structured_pc_size_requires_preflight=True,
        structured_pc_preflight_required=True,
        factor_preflight_max_target_ratio=10.0,
        factor_preflight_residual_before=1.0,
        factor_preflight_residual_after=0.1,
        factor_preflight_improvement_ratio=0.1,
        factor_preflight_target_ratio=1.0,
        factor_preflight_residual_diagnostics={"relative_residual": 0.1},
        factor_preflight_seed_used=True,
        factor_preflight_passed=True,
        factor_preflight_error=None,
    )

    assert full_retry.host_sparse_direct_used is True
    assert generic.factor_policy is factor_policy
    assert direct_tail.structured_admission is structured_admission
    assert direct_tail.factor_bundle_pc is factor_bundle
    assert rescue.factor_preflight_policy is factor_policy
    np.testing.assert_allclose(generic.reduce_full(jnp.asarray([2.0, 3.0])), [2.0, 3.0])


def test_structured_full_csr_selection_and_solve_result_are_json_friendly() -> None:
    matrix = sp.csr_matrix(np.diag([2.0, 4.0]))
    selection = RHS1StructuredFullCSRSelection(
        fblock_selection=_FakeFBlockSelection(),
        matrix=matrix,
        selected=True,
        reason="built",
        cache_hit=False,
        build_s=0.01,
        metadata={"nnz": int(matrix.nnz)},
    )
    solve = RHS1StructuredFullCSRSolveResult(
        selection=selection,
        x=np.asarray([0.5, 0.25], dtype=np.float64),
        residual_norm=1.0e-13,
        residual_history=(1.0, 1.0e-3, 1.0e-13),
        info=0,
        converged=True,
        solve_s=0.02,
        metadata={"solver": "direct"},
    )
    rejected = RHS1StructuredFullCSRSelection(
        fblock_selection=_FakeFBlockSelection(),
        matrix=None,
        selected=False,
        reason="unsupported",
        cache_hit=False,
        build_s=0.0,
        metadata={},
    )

    np.testing.assert_allclose(selection.matvec(np.ones(2)), np.asarray([2.0, 4.0]))
    payload = solve.to_dict()
    assert payload["selected"] is True
    assert payload["residual_history"] == (1.0, 1.0e-3, 1.0e-13)
    assert payload["metadata"] == {"solver": "direct"}
    assert selection.to_dict()["fblock_selection"] == {"selected": True, "reason": "unit"}
    with pytest.raises(RuntimeError, match="structured full CSR operator was not selected"):
        rejected.matvec(np.ones(2))
