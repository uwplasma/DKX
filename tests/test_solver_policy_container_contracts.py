from __future__ import annotations

from dataclasses import fields, is_dataclass

import jax.numpy as jnp
import numpy as np

from sfincs_jax.problems.profile_dense import (
    ProfileLinearSolveDispatch,
    RHS1Constraint0PETScCompatSolveOutcome,
    RHS1DenseKSPFullSolveOutcome,
    RHS1DenseKSPReducedSolveOutcome,
    RHS1DenseProbeAdmission,
    RHS1DenseProbeShortcutDecision,
    RHS1DenseProbeStageResult,
    RHS1DenseShortcutSetup,
    RHS1EarlyDenseShortcutDecision,
    RHS1FullHostDenseShortcutResult,
    RHS1PostKrylovDenseShortcutDecision,
    RHS1ReducedHostDenseShortcutResult,
    RHS1ScipyRescueOutcome,
    RHS1ScipyRescueStageResult,
)
from sfincs_jax.problems.profile_setup import (
    RHS1ActiveDOFDecision,
    RHS1ActiveDOFState,
    RHS1ActiveProblemSetup,
    RHS1DKESAdjustmentSetup,
    RHS1DomainDecompositionSetup,
    RHS1GmresBudgetSetup,
    RHS1InitialRouteSetup,
    RHS1PhysicsFlagSetup,
    RHS1PostActiveSolvePolicySetup,
    RHS1PreconditionerOptionSetup,
    RHS1RecycleBasisSetup,
    RHS1ReducedModeShapeSetup,
    RHS1ToleranceSetup,
    SolveMethodRequestFlags,
)
from sfincs_jax.problems.transport_parallel_runtime import (
    CompiledShardedOperatorReuseGate,
    MultiGpuCaseThroughputAudit,
    ParallelScalingClaimScopeAudit,
    ShardedSolveAmortizationDiagnostics,
    ShardedSolveBalanceDiagnostics,
    ShardedSolveDeterministicOutputGate,
    ShardedSolveDeviceAssignment,
    ShardedSolveExecutionPlan,
    ShardedSolveScalingAudit,
    SingleCaseOperatorCoarseReusePlan,
    TransportParallelScalingAudit,
    format_transport_rhs_list,
)
from sfincs_jax.problems.transport_policies import (
    TransportActiveDOFDecision,
    TransportActiveDOFState,
    TransportDDConfig,
    TransportDensePolicy,
    TransportInitialSolvePolicy,
    TransportPerRHSLoopPolicy,
    TransportPolishConfig,
)
from sfincs_jax.problems.transport_setup import (
    TransportMaxiterSetup,
    TransportParallelRequest,
    TransportStateSetup,
    TransportWhichRHSSetup,
)
from sfincs_jax.solvers.native_block_factor import (
    NativeDenseBlockJacobi,
    NativePaddedIndexedBlockFactor,
    NativeTwoFieldSchurFactor,
    NativeXEllKineticFactor,
)
from sfincs_jax.solvers.preconditioner_symbolic_policy import (
    ActiveFortranV3ReducedFactorPolicy,
    ActiveSymbolicBlockSchurPolicy,
    ActiveSymbolicFrontalPolicy,
    ActiveSymbolicSuperblockPolicy,
)
from sfincs_jax.solvers.preconditioner_xblock_policy import (
    RHS1XBlockDeviceHostFallbackDecision,
    RHS1XBlockLocalSolveCandidate,
    RHS1XBlockLocalSolveTuning,
    RHS1XBlockLowerFillAcceptance,
    RHS1XBlockSideProbeControls,
    RHS1XBlockSparsePCPolicy,
)


def _value_for_field(name: str):
    if name in {"messages", "notes", "failures", "cap_reasons", "promotion_blockers"}:
        return ("unit",)
    if name.endswith("_messages"):
        return ("unit",)
    if name in {"available_device_ids", "device_counts", "worker_counts", "per_device_components", "compiled_components", "reused_components", "replicated_components", "required_runtime_gates", "permc_candidates"}:
        return (1, 2)
    if name in {"device_assignments"}:
        return ()
    if name == "balance_diagnostics":
        return None
    if name in {"active_idx_np", "nxi_for_x"}:
        return np.asarray([0, 1], dtype=np.int32)
    if name in {"active_idx_jnp", "full_to_active_jnp", "x0", "residual_vec", "replay_rhs", "replay_matvec", "rhs"}:
        return jnp.asarray([0.0, 1.0], dtype=jnp.float64)
    if name in {"x0_by_rhs", "state_x_by_rhs", "metadata"}:
        return {}
    if name in {"basis", "context", "result", "factor_preflight_policy"}:
        return object()
    if name in {"state_in_path", "state_out_path", "error", "direct_tail_structured_pc_reason", "factor_preflight_error"}:
        return None
    if name.endswith("_path"):
        return None
    if name.startswith("include_") or name.startswith("use_") or name.startswith("force_") or name.startswith("dense_") or name.startswith("stream_") or name.startswith("store_") or name.startswith("skip_") or name.startswith("requested") or name.endswith("_enabled") or name.endswith("_requested") or name.endswith("_selected") or name.endswith("_pass") or name.endswith("_used") or name.endswith("_allowed") or name.endswith("_supported") or name.endswith("_ready") or name.endswith("_valid") or name.endswith("_required") or name.endswith("_claim") or name in {"enabled", "selected", "capped", "passes", "plan_valid", "promotion_ready", "release_scaling_claim", "release_scaling_supported", "experimental_single_case_scaling", "ci_gate_pass", "compile_in_timed_region", "deterministic_output_check", "warm_run_amortization_pass", "output_digest_match", "backend_allowed", "maxiter_env_forced", "restart_env_forced", "fp_tightened", "pas_tightened", "tokamak_pas", "pas_large_bicgstab_fastpath", "has_reduced_modes", "full_preconditioner_requested", "pas_project_enabled", "use_pas_projection", "subset_mode"}:
        return True
    if "ratio" in name or "tol" in name or "mb" in name or name.endswith("_s") or "speedup" in name or "efficiency" in name or "fraction" in name or "units" in name or "latency" in name or "bytes" in name or "threshold" in name or "regularization" in name or "drop" in name or "fill" in name or "pivot" in name or "omega" in name or "alpha" in name or "clip" in name or "rcond" in name or "iota" in name:
        return 1.0
    if "kind" in name or "backend" in name or "method" in name or "scope" in name or "strategy" in name or "axis" in name or "solver" in name or "mode" in name or "reason" in name or "status" in name or "algorithm" in name or "digest" in name or "source" in name or "label" in name or name in {"precondition_side", "operator_build_scope", "operator_action_scope", "preconditioner_scope", "coarse_operator_scope", "coarse_solve_scope", "cache_required", "compile_cache_dir", "timing_semantics", "benchmark_kind", "schema_version", "artifact_kind", "claim_scope", "requested_kind", "active_symbolic_kind", "active_architecture", "ordering_kind", "factor_kind", "permc_requested", "permc_spec", "scale_norm", "solve_method", "solve_method_use"}:
        return "unit"
    return 1


def _construct(cls, **overrides):
    assert is_dataclass(cls)
    values = {field.name: _value_for_field(field.name) for field in fields(cls)}
    values.update(overrides)
    return cls(**values)


def test_profile_setup_policy_containers_capture_route_state() -> None:
    active_state = RHS1ActiveDOFState(
        active_idx_np=np.asarray([0, 2], dtype=np.int32),
        active_idx_jnp=jnp.asarray([0, 2], dtype=jnp.int32),
        full_to_active_jnp=jnp.asarray([1, 0, 2], dtype=jnp.int32),
        active_size=2,
    )
    active_problem = RHS1ActiveProblemSetup(
        tol=1.0e-9,
        restart=60,
        maxiter=120,
        messages=((1, "active setup"),),
        use_dkes=False,
        include_xdot_sparse_pc=True,
        include_electric_field_xi_sparse_pc=True,
        er_abs_sparse_pc=0.1,
        preconditioner_species=1,
        preconditioner_x=2,
        preconditioner_x_min_l=1,
        preconditioner_xi=3,
        full_preconditioner_requested=False,
        geom_scheme=5,
        use_pas_projection=True,
        use_active_dof_mode=True,
        active_idx_jnp=active_state.active_idx_jnp,
        full_to_active_jnp=active_state.full_to_active_jnp,
        active_size=active_state.active_size,
    )

    assert RHS1GmresBudgetSetup(60, 120, False, True).maxiter_env_forced is True
    assert RHS1ToleranceSetup(1e-8, 1e-9, 100, True, 1e-8, 1e-7, False, 1e-7).fp_tightened
    assert RHS1PhysicsFlagSetup(False, True, True, 0.1).include_xdot_sparse_pc
    assert RHS1DKESAdjustmentSetup(1e-8, 40, 80, ((1, "dkes"),)).messages
    assert RHS1PostActiveSolvePolicySetup(40, 80, "auto", False, True, 1000, ()).pas_large_bicgstab_fastpath
    assert SolveMethodRequestFlags("auto", False, False, True, False, True, False, False).sparse_pc_gmres_requested
    assert RHS1InitialRouteSetup(_construct(SolveMethodRequestFlags), False, 0.0, True, False).structured_auto_allowed
    assert RHS1RecycleBasisSetup(4, None).recycle_k == 4
    assert RHS1ReducedModeShapeSetup(np.asarray([2, 4], dtype=np.int32), 4, True).has_reduced_modes
    assert RHS1ActiveDOFDecision(True, "reduced").use_active_dof_mode
    assert active_problem.active_size == 2
    assert RHS1PreconditionerOptionSetup(1, 2, 1, 3, False, 5, "auto", True, True, True).pas_project_enabled
    assert RHS1DomainDecompositionSetup("theta", 100, 12, 4, 5, 1, 1, 16, 20).patch_dof_target == 100


def test_profile_dense_result_containers_preserve_shortcut_and_rescue_state() -> None:
    result = object()
    residual = jnp.asarray([1.0, 0.0], dtype=jnp.float64)
    outcome = RHS1ScipyRescueOutcome(
        result=result,
        residual_vec=residual,
        reported_residual=1.0e-8,
        history_len=3,
        preconditioned_residual=2.0e-8,
    )
    stage = RHS1ScipyRescueStageResult(result=result, residual_vec=residual, metadata={"solver": "lgmres"})
    compat = RHS1Constraint0PETScCompatSolveOutcome(
        result=result,
        replay_matvec=residual,
        replay_rhs=residual,
        true_residual=1.0e-10,
        preconditioned_residual=2.0e-10,
        rhs_pc_norm=3.0,
        drop_threshold=1.0e-8,
        regularization=1.0e-10,
        nnz=4,
    )
    full_ksp = RHS1DenseKSPFullSolveOutcome(result=result, replay_matvec=residual, replay_rhs=residual)
    reduced_ksp = RHS1DenseKSPReducedSolveOutcome(result=result, replay_matvec=residual, replay_rhs=residual)
    reduced_shortcut = RHS1ReducedHostDenseShortcutResult(result=result, early_dense_shortcut=True, probe_shortcut=False)
    full_shortcut = RHS1FullHostDenseShortcutResult(result=result, residual_vec=residual)
    probe_stage = RHS1DenseProbeStageResult(
        result=result,
        x0_reduced=residual,
        early_dense_shortcut=True,
        probe_shortcut=True,
    )

    assert ProfileLinearSolveDispatch(context={"kind": "unit"}).context["kind"] == "unit"
    assert outcome.history_len == 3
    assert stage.metadata == {"solver": "lgmres"}
    assert compat.nnz == 4
    assert full_ksp.replay_rhs is residual
    assert reduced_ksp.replay_matvec is residual
    assert reduced_shortcut.early_dense_shortcut is True
    assert full_shortcut.result is result
    assert RHS1DenseProbeAdmission(enabled=True).enabled
    assert RHS1DenseProbeShortcutDecision(True, False, ((1, "probe"),)).accept_shortcut
    assert probe_stage.probe_shortcut is True
    assert RHS1DenseShortcutSetup(10.0, 2, False, ((1, "dense"),)).dense_fallback_max == 2
    assert RHS1EarlyDenseShortcutDecision(True, ()).early_dense_shortcut
    assert RHS1PostKrylovDenseShortcutDecision(True, ()).dense_shortcut


def test_transport_policy_and_parallel_containers_preserve_release_gate_contracts() -> None:
    active = TransportActiveDOFState(
        active_idx_np=np.asarray([0], dtype=np.int32),
        active_idx_jnp=jnp.asarray([0], dtype=jnp.int32),
        full_to_active_jnp=jnp.asarray([1], dtype=jnp.int32),
        active_size=1,
    )
    assignment = ShardedSolveDeviceAssignment(
        device_index=0,
        device_id="gpu0",
        shard_start=0,
        shard_stop=4,
        work_units=4,
        workload_fraction=1.0,
    )
    balance = ShardedSolveBalanceDiagnostics(
        total_work_units=4,
        min_work_units=4,
        max_work_units=4,
        imbalance_units=0,
        max_to_mean_ratio=1.0,
        idle_device_count=0,
    )
    plan = ShardedSolveExecutionPlan(
        benchmark_kind="single_case",
        backend="gpu",
        rhs_mode=1,
        shard_axis="theta",
        task_count=1,
        requested_devices=1,
        active_devices=1,
        available_device_count=1,
        available_device_ids=("gpu0",),
        capped=False,
        cap_reasons=(),
        eligible_for_single_case_sharding=True,
        release_scaling_claim=False,
        release_scaling_supported=False,
        experimental_single_case_scaling=True,
        failures=(),
        notes=("unit",),
        device_assignments=(assignment,),
        balance_diagnostics=balance,
    )

    assert TransportPolishConfig(True, 1e-5, 10.0, 1e-10, 30, 80).enabled
    assert TransportActiveDOFDecision(True, "env", "dense", False).reason == "env"
    assert active.active_size == 1
    assert _construct(TransportDensePolicy, solve_method_use="dense").solve_method_use == "dense"
    assert _construct(TransportInitialSolvePolicy, geometry_scheme=5).geometry_scheme == 5
    assert _construct(TransportPerRHSLoopPolicy, rhs_mode=3).rhs_mode == 3
    assert TransportDDConfig(4, 1, 5, 1).block_theta == 4
    assert TransportMaxiterSetup(400, ((1, "maxiter"),)).maxiter == 400
    assert TransportStateSetup(None, None, None, {}, {}).x0_by_rhs == {}
    assert TransportWhichRHSSetup(3, 4, (1, 3), True).which_rhs_values == (1, 3)
    assert TransportParallelRequest(False, 2, "process").parallel_workers == 2
    assert _construct(TransportParallelScalingAudit, backend="cpu").backend == "cpu"
    assert _construct(ShardedSolveScalingAudit, backend="gpu").backend == "gpu"
    assert _construct(MultiGpuCaseThroughputAudit, backend="gpu").backend == "gpu"
    assert _construct(ParallelScalingClaimScopeAudit, claim_scope="throughput").claim_scope == "throughput"
    assert plan.balance_diagnostics.max_to_mean_ratio == 1.0
    assert _construct(ShardedSolveAmortizationDiagnostics, active_devices=2).active_devices == 2
    assert _construct(CompiledShardedOperatorReuseGate, strategy="persistent").strategy == "persistent"
    assert _construct(ShardedSolveDeterministicOutputGate, status="pass").status == "pass"
    assert _construct(SingleCaseOperatorCoarseReusePlan, active_devices=2).active_devices == 2
    assert format_transport_rhs_list([3, 1, 2]) == "[3, 1, 2]"


def test_xblock_symbolic_and_native_policy_contracts_are_json_ready() -> None:
    tuning = RHS1XBlockLocalSolveTuning(
        drop_tol=1.0e-8,
        drop_rel=1.0e-6,
        ilu_drop_tol=1.0e-4,
        fill_factor=5.0,
        row_nnz_cap=64,
        compact_row_nnz_cap=32,
    )
    candidate = RHS1XBlockLocalSolveCandidate(
        block_size=128,
        lu_max=256,
        mode="local",
        factorization="ilu",
        tuning=tuning,
        metadata_label="xblock_ilu",
        selection_reason="lower_fill",
        exact_lu=False,
        lower_fill=True,
        lower_fill_requested=True,
        ignored_lower_fill_env=False,
        lower_fill_max_block_size=1024,
        lower_fill_block_size_capped=False,
    )
    host = RHS1XBlockDeviceHostFallbackDecision("auto", True, "large_qi", True, "gmres", "lgmres", 100, True, False)
    side_probe = RHS1XBlockSideProbeControls(True, 20, 40, 10.0, True, True, 80, False, 5, 0.25)
    lower_fill = RHS1XBlockLowerFillAcceptance(True, "accepted", "lower", "base", 0.5, 2.0, 1.0, 10.0, 1.0, 1.0e8)

    assert RHS1XBlockSparsePCPolicy("right", True, "gmres", False, 80, False).precondition_side == "right"
    assert side_probe.should_switch(11.0) is True
    assert side_probe.should_switch(1.0) is False
    assert host.to_metadata()["non_autodiff"] is True
    assert candidate.to_metadata()["metadata_label"] == "xblock_ilu"
    assert lower_fill.to_metadata()["accepted"] is True
    assert ActiveSymbolicSuperblockPolicy(1, "rcm", 2, 3, 4, 5, 1, 0.1, 1e-8, 1.5, 2, 1e-6, 1.0).ordering_kind == "rcm"
    assert ActiveSymbolicBlockSchurPolicy(1, "rcm", 2, 3, 4, 1, 2, 1e-8, 1.5, 2, 1e-6, 1.0).separator_cols == 4
    assert ActiveSymbolicFrontalPolicy(
        requested_kind="auto",
        active_symbolic_kind="frontal",
        active_architecture="frontal",
        max_active_size=1,
        ordering_kind="rcm",
        block_size=2,
        max_permutation_size=3,
        separator_cols=4,
        max_superblock_size=5,
        max_superblock_blocks=6,
        boundary_width=1,
        high_degree_cols=2,
        min_cross_nnz=1,
        max_dense_rhs_entries=10,
        max_dense_rhs_cols_per_block=4,
        min_cross_separator_fraction=0.2,
        regularization_rel=1e-8,
        prefill_safety_factor=1.5,
        admission_probes=2,
        admission_max_relative_residual=1e-6,
        admission_min_improvement=1.0,
    ).active_architecture == "frontal"
    assert ActiveFortranV3ReducedFactorPolicy("auto", "ilu", True, 1000, False, 5.0, 1e-4, 0.01, "COLAMD", ("COLAMD",), "COLAMD", "row", 1e6, False, 2000, 1.5).factor_kind == "ilu"

    dense = NativeDenseBlockJacobi(
        block_inverses=jnp.eye(2, dtype=jnp.float64).reshape((1, 2, 2)),
        block_size=2,
        original_size=2,
        padded_size=2,
        regularization=1.0e-8,
    )
    schur = NativeTwoFieldSchurFactor(
        a_ff_inv=jnp.eye(1, dtype=jnp.float64),
        a_fc=jnp.ones((1, 1), dtype=jnp.float64),
        a_cf=jnp.ones((1, 1), dtype=jnp.float64),
        schur_inv=jnp.eye(1, dtype=jnp.float64),
        f_size=1,
        c_size=1,
        regularization=1.0e-8,
    )
    xell = NativeXEllKineticFactor(
        block_inverses=jnp.eye(2, dtype=jnp.float64).reshape((1, 2, 2)),
        block_indices=jnp.asarray([[0, 1]], dtype=jnp.int32),
        inv_tail=jnp.ones((0,), dtype=jnp.float64),
        f_size=2,
        total_size=2,
    )
    padded = NativePaddedIndexedBlockFactor(
        block_inverses=jnp.eye(2, dtype=jnp.float64).reshape((1, 2, 2)),
        block_indices=jnp.asarray([[0, 1]], dtype=jnp.int32),
        block_mask=jnp.asarray([[True, True]]),
        overlap_weights=jnp.ones((2,), dtype=jnp.float64),
        total_size=2,
        normalize_overlap=True,
        damping=1.0,
    )

    assert dense.block_size == 2
    assert schur.f_size == 1
    assert xell.total_size == 2
    assert padded.normalize_overlap is True
