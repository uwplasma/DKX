from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp

from sfincs_jax.compare import CompareResult, H5DatasetParity
from sfincs_jax.geometry.vmec_wout import VmecInterpolation
from sfincs_jax.operators.profile_device_sparse import (
    DeviceCSR,
    DeviceCSRMetadata,
    MatvecValidationResult,
)
from sfincs_jax.operators.profile_kinetic import (
    RHS1PartialFBlockAssembly,
    RHS1StructuredFBlockCSRSelection,
    RHS1StructuredFBlockSelection,
)
from sfincs_jax.operators.profile_layout import (
    RHS1BlockCOOBuilder,
    RHS1BlockLayout,
    RHS1BlockPreconditionerProbe,
    RHS1KineticIndices,
    RHS1MatrixFreeLeastSquaresResidualCorrection,
    build_rhs1_compressed_pitch_layout,
)
from sfincs_jax.operators.profile_sparse_pattern import V3SparsePatternSummary
from sfincs_jax.paths import ResolveResult
from sfincs_jax.problems.profile_phi1_newton import Phi1FrozenJacobianPolicy, Phi1LineSearchPolicy
from sfincs_jax.problems.profile_preconditioner_build import (
    RHS1FullBasePreconditionerSetupResult,
    RHS1FullStrongRetryStageResult,
    RHS1FullStrongPreconditionerSelection,
    RHS1PostPrimaryMinresCorrectionOutcome,
    RHS1ReducedPreconditionerBuildResult,
    RHS1ReducedStrongRetryStageResult,
    RHS1ReducedStrongPreconditionerSelection,
    RHS1StrongAutoSelection,
)
from sfincs_jax.problems.profile_residual import ProjectedResidualPolishOutcome
from sfincs_jax.problems.transport_finalize import TransportPostsolveDiagnostics, TransportRHSFinalizationResult
from sfincs_jax.solvers.diagnostics import compare_solver_profile_files
from sfincs_jax.solvers.memory_model import LinearSolveMemoryEstimate
from sfincs_jax.solvers.path_policy import SolverCandidateGate
from sfincs_jax.solvers.preconditioner_full_fp_csr import RHS1FullCSRKineticPreconditioner
from sfincs_jax.solvers.preconditioner_pas_composite import RHS1PasFamilyBuilders
from sfincs_jax.solvers.preconditioner_pas_matrix_free import PasRuntimeChunkPlan, Rhs1PasMatrixFreeResult
from sfincs_jax.solvers.preconditioner_pas_policy import (
    AdaptiveStationaryResult,
    ConstrainedPASBranchSummary,
    PasResidualTrend,
    PasSmootherDecision,
)
from sfincs_jax.solvers.preconditioner_schur_profile import (
    ActiveNativeFieldSplitSparseCoarsePolicy,
    ActiveNativeStackPolicy,
    ActiveSparseCoarseResidualPolicy,
)
from sfincs_jax.solvers.preconditioner_host_sparse import RHS1FullSystemMatrixFreeOperatorAdapter
from sfincs_jax.solvers.preconditioner_reduced_pmat import (
    RHS1ReducedPmatEliminationPlan,
    RHS1ReducedPmatGroup,
)
from sfincs_jax.validation.artifacts import (
    ARTIFACT_CLASS_RELEASE_BLOCKING,
    BenchmarkArtifactIndex,
    BenchmarkArtifactIndexEntry,
    BenchmarkArtifactPolicyError,
    CollisionalityLike,
    CollisionalityRecord,
    ErSweepRecord,
    PhaseRecord,
    SuiteCaseMetric,
    validate_benchmark_artifact_file,
)
from sfincs_jax.validation.fortran import PetscCSRMatrix, PetscVec, parse_fortran_v3_profile_file
from sfincs_jax.workflows.mapped_xgrid import (
    MappedTransportEvidenceReport,
    MappedTransportEvidenceRow,
    TransportMatrixError,
    TransportMomentReport,
    TransportSolveSummary,
)
from sfincs_jax.workflows import optimization as opt
from sfincs_jax.workflows.optimization import (
    AmbipolarRoot,
    AmbipolarRootSummary,
    CandidateScanPlan,
    NeoclassicalObjectiveWeights,
    PromotionEvidenceLane,
    PromotionEvidencePlan,
    ScanPromotionRun,
    ScanPromotionSummary,
    load_ladder_config,
    load_proxy_summary,
    run_fortran_er_scan,
    write_promotion_evidence_plan,
)


def _small_layout() -> RHS1BlockLayout:
    return RHS1BlockLayout(
        n_species=1,
        n_x=2,
        n_xi=2,
        n_theta=1,
        n_zeta=1,
        f_size=4,
        phi1_size=0,
        extra_size=1,
        total_size=5,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )


def _tiny_block_operator():
    builder = RHS1BlockCOOBuilder(shape=(2, 2), block_size=1, dtype=np.float64)
    builder.add_dense_block(0, 0, np.asarray([[1.0]]))
    builder.add_dense_block(1, 1, np.asarray([[2.0]]))
    return builder.build()


def _valid_pas_benchmark_payload() -> dict[str, object]:
    return {
        "schema_version": 2,
        "kind": "pas_tz_memory_fallback_benchmark",
        "plan": {
            "variants": ["zeta"],
            "variant_methods": [
                {"variant": "zeta", "realized_solve_method": "incremental"},
            ],
        },
        "results": [
            {
                "variant": "zeta",
                "status": "ok",
                "variant_provenance": {"variant": "zeta"},
                "solver_provenance": {
                    "requested_solve_method": "incremental",
                    "realized_solve_method": "incremental",
                },
                "phase_metadata": [{"name": "solve", "status": "ok", "elapsed_s": 0.1}],
                "tail_metadata": {"messages_tail_limit": 20, "messages_tail_count": 2},
            },
        ],
    }


def test_optimization_evidence_and_fortran_scan_contracts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = AmbipolarRoot(er=1.25, root_type="electron", bracket=(1.0, 1.5), slope=2.0)
    summary = AmbipolarRootSummary(
        roots=(root,),
        er_min=-1.0,
        er_max=2.0,
        radial_current_min=-0.5,
        radial_current_max=0.5,
        bracketed=True,
    )
    weights = NeoclassicalObjectiveWeights(bootstrap=0.5, electron_root=2.0)
    assert summary.has_electron_root
    assert summary.as_dict()["roots"][0]["root_type"] == "electron"
    assert weights.bootstrap == 0.5

    run = ScanPromotionRun(
        path=tmp_path / "sfincsOutput.h5",
        er=1.25,
        radial_current=0.0,
        bootstrap_current=0.01,
        particle_flux=(1.0, -0.2),
        heat_flux=(0.5, 0.6),
        residual_norm=1.0e-12,
        residual_target=1.0e-10,
        residual_gate={"status": "pass", "failures": []},
    )
    promotion = ScanPromotionSummary(
        scan_dir=tmp_path,
        runs=(run,),
        selected_root=root,
        bootstrap_objective=1.0e-4,
        flux_objective={"total": 0.2},
        electron_root_penalty=0.0,
        gate_status="pass",
        failures=(),
    )
    assert promotion.as_dict()["gate_status"] == "pass"

    proxy_path = tmp_path / "proxy.json"
    proxy_path.write_text(json.dumps({"workflow": "proxy", "objective_preset": "qa"}) + "\n")
    assert load_proxy_summary(proxy_path)["workflow"] == "proxy"

    candidate = CandidateScanPlan(
        proxy_summary=proxy_path,
        input_namelist=tmp_path / "input.namelist",
        out_dir=tmp_path / "scan",
        er_values=(-1.0, 0.0, 1.0),
        compute_solution=True,
        compute_transport_matrix=False,
        jobs=2,
        skip_existing=True,
        scan_command=("sfincs_jax", "scan-er"),
        promotion_command=("python", "promote.py"),
    )
    assert candidate.as_dict()["jobs"] == 2

    lane = PromotionEvidenceLane(
        label="cpu",
        backend="jax_cpu",
        scan_dir=tmp_path / "cpu_scan",
        promotion_dir=tmp_path / "cpu_promotion",
        scan_command=("sfincs_jax", "scan-er"),
        promotion_command=("python", "promote.py"),
        env={"JAX_PLATFORM_NAME": "cpu"},
    )
    evidence = PromotionEvidencePlan(
        input_namelist=tmp_path / "input.namelist",
        out_dir=tmp_path / "evidence",
        er_values=(-1.0, 1.0),
        lanes=(lane,),
        comparison_command=("python", "compare.py"),
    )
    evidence_path = write_promotion_evidence_plan(tmp_path / "evidence.json", evidence)
    payload = json.loads(evidence_path.read_text())
    assert payload["lanes"][0]["env"] == {"JAX_PLATFORM_NAME": "cpu"}

    ladder_path = tmp_path / "ladder.json"
    ladder_path.write_text(json.dumps({"surfaces": [0.25, 0.5]}) + "\n")
    assert load_ladder_config(ladder_path)["surfaces"] == [0.25, 0.5]

    input_path = tmp_path / "input.namelist"
    input_path.write_text(
        "&geometryParameters\n"
        " inputRadialCoordinateForGradients = 4\n"
        "/\n"
        "&physicsParameters\n"
        " Er = 0.0\n"
        "/\n"
    )

    def fake_run_sfincs_fortran(*, input_namelist, exe, workdir, timeout_s):
        del input_namelist, exe, timeout_s
        Path(workdir, "sfincsOutput.h5").write_bytes(b"placeholder")

    messages: list[str] = []
    monkeypatch.setattr(opt, "run_sfincs_fortran", fake_run_sfincs_fortran)
    result = run_fortran_er_scan(
        input_namelist=input_path,
        out_dir=tmp_path / "fortran_scan",
        values=(-1.0, 1.0),
        timeout_s=0.1,
        skip_existing=False,
        emit=lambda _level, message: messages.append(message),
    )
    assert result.variable == "Er"
    assert result.values == (1.0, -1.0)
    assert all(path.exists() for path in result.outputs)
    assert any("fortran-scan: progress" in message for message in messages)


def test_validation_artifact_and_profile_file_contracts(tmp_path: Path) -> None:
    assert CollisionalityLike.__name__ == "CollisionalityLike"
    collision = CollisionalityRecord(
        label="high-nu",
        nuprime=10.0,
        transport_matrix=np.eye(2),
    )
    sweep = ErSweepRecord(
        model="dkes",
        label="W7-X",
        er=0.0,
        er_over_eres=None,
        particle_flux_vm_psi_hat=1.0,
        heat_flux_vm_psi_hat=2.0,
        fsab_flow=0.1,
        fsab_jhat=0.2,
        output_path="sfincsOutput.h5",
    )
    phase = PhaseRecord(name="solve", elapsed_s=0.25, metadata={"rhs_mode": 1})
    assert collision.transport_matrix.shape == (2, 2)
    assert sweep.model == "dkes"
    assert phase.to_json()["metadata"] == {"rhs_mode": 1}

    metric = SuiteCaseMetric(
        case="qa_bootstrap",
        status="parity_ok",
        blocker_type="none",
        fortran_runtime_s=10.0,
        jax_runtime_s=5.0,
        jax_runtime_s_cold=6.0,
        jax_runtime_s_warm=4.0,
        jax_logged_elapsed_s=4.5,
        fortran_max_rss_mb=100.0,
        jax_max_rss_mb=50.0,
        jax_incremental_max_rss_mb=25.0,
        jax_rss_baseline_mb=20.0,
        jax_memory_metric_source="incremental",
        practical_mismatches=0,
        strict_mismatches=0,
    )
    assert metric.runtime_ratio == 0.5
    assert metric.warm_or_logged_runtime_source == "jax_runtime_s_warm"
    assert metric.active_memory_ratio == 0.25

    blocking_entry = BenchmarkArtifactIndexEntry(
        path=tmp_path / "bad.json",
        classification=ARTIFACT_CLASS_RELEASE_BLOCKING,
        errors=("missing schema_version",),
    )
    index = BenchmarkArtifactIndex(entries=(blocking_entry,))
    assert index.counts[ARTIFACT_CLASS_RELEASE_BLOCKING] == 1
    assert index.release_blocking == (blocking_entry,)

    valid_artifact = tmp_path / "pas_benchmark.json"
    valid_artifact.write_text(json.dumps(_valid_pas_benchmark_payload()) + "\n")
    validate_benchmark_artifact_file(valid_artifact)
    invalid_artifact = tmp_path / "invalid.json"
    invalid_artifact.write_text("{}\n")
    with pytest.raises(BenchmarkArtifactPolicyError, match="schema_version"):
        validate_benchmark_artifact_file(invalid_artifact)

    log_path = tmp_path / "fortran_profile.log"
    log_path.write_text(
        "Parallel job ( 2 processes) detected\n"
        "Ntheta = 25\n"
        "Nzeta = 51\n"
        "Nxi = 100\n"
        "NL = 4\n"
        "Nx = 4\n"
        "solverTolerance = 1.0e-9\n"
        "Solver package which will be used:\n"
        "mumps\n"
        "The matrix is 4 x 4 elements\n"
        "# of nonzeros in Jacobian matrix: 8, allocated: 16\n"
        "# of nonzeros in Jacobian preconditioner matrix: 6, allocated: 12\n"
        "Entering DMUMPS\n"
        "INFOG(21) = 33\n"
        "INFOG(22) = 44\n"
        "INFOG(29) = 55\n"
        " 0 KSP Residual norm 1.0e-3\n"
        " 1 KSP Residual norm 1.0e-9\n"
        "KSPConvergedReason = 2\n"
        "Elapsed time in analysis driver= 0.11\n"
        "Elapsed time in factorization driver= 0.22\n"
    )
    profile = parse_fortran_v3_profile_file(log_path)
    assert profile["solver_package"] == "mumps"
    assert profile["matrix_nnz"] == {"nnz": 8, "allocated_nnz": 16}
    assert profile["ksp"]["final_residual"] == 1.0e-9

    fortran_json = tmp_path / "fortran_profile.json"
    jax_json = tmp_path / "jax_profile.json"
    fortran_json.write_text(json.dumps(profile) + "\n")
    jax_json.write_text(
        json.dumps(
            {
                "residual_norm": 2.0e-9,
                "residual_target": 1.0e-8,
                "active_size": 4,
                "elapsed_s": 0.5,
                "solve_method": "structured_csr",
            }
        )
        + "\n"
    )
    comparison = compare_solver_profile_files(
        fortran_profile_path=fortran_json,
        jax_profile_path=jax_json,
    )
    assert comparison["comparison"]["same_active_size"] is True
    assert comparison["jax"]["residual_ratio"] == pytest.approx(0.2)

    vec = PetscVec(values=np.asarray([1.0, 2.0]))
    matrix = PetscCSRMatrix(
        shape=(2, 2),
        row_ptr=np.asarray([0, 1, 2]),
        col_ind=np.asarray([0, 1]),
        data=np.asarray([3.0, 4.0]),
    )
    assert vec.size == 2
    assert matrix.get(1, 1) == 4.0
    assert matrix.get(0, 1) == 0.0


def test_remaining_solver_policy_container_contracts() -> None:
    strong = RHS1StrongAutoSelection(kind="xblock", xblock_tz_lmax=2)
    reduced_selection = RHS1ReducedStrongPreconditionerSelection(
        kind="xblock",
        candidate_kind_before_skips="pas",
        xblock_tz_lmax=2,
        trigger="residual",
        skipped_weak_pas=True,
        skipped_guarded_pas_tz=False,
    )
    full_selection = RHS1FullStrongPreconditionerSelection(kind="full_csr", xblock_tz_lmax=3)
    post_primary = RHS1PostPrimaryMinresCorrectionOutcome(
        result="result",
        residual_vec=jnp.asarray([0.0]),
        residual_norm_true=1.0e-12,
        accepted_guarded=True,
        accepted_weak=False,
    )
    reduced_build = RHS1ReducedPreconditionerBuildResult(
        preconditioner=lambda x: x,
        rhs1_precond_kind="xblock",
        pas_precond_force_collision=False,
        bicgstab_preconditioner=None,
        pas_tz_guarded_fallback=False,
        pas_tz_guarded_axis=None,
    )
    full_base = RHS1FullBasePreconditionerSetupResult(
        preconditioner=lambda x: x,
        bicgstab_preconditioner=None,
    )
    full_retry = RHS1FullStrongRetryStageResult(
        result="accepted",
        residual_vec=jnp.asarray([1.0]),
        accepted=True,
        elapsed_s=0.1,
        selected_kind="full_csr",
        preconditioner=lambda x: x,
    )
    reduced_retry = RHS1ReducedStrongRetryStageResult(
        result="accepted",
        residual_vec=jnp.asarray([1.0]),
        accepted=True,
        elapsed_s=0.1,
        selected_kind="xblock",
        preconditioner=lambda x: x,
    )
    assert strong.kind == "xblock"
    assert reduced_selection.skipped_weak_pas
    assert full_selection.xblock_tz_lmax == 3
    assert post_primary.accepted_guarded
    assert reduced_build.rhs1_precond_kind == "xblock"
    assert full_base.bicgstab_preconditioner is None
    assert full_retry.accepted and reduced_retry.accepted

    trend = PasResidualTrend(
        history=(3.0, 2.0, 1.0),
        latest=1.0,
        previous=2.0,
        best_so_far=1.0,
        best_before_latest=2.0,
        worst_so_far=3.0,
        latest_ratio=0.5,
        best_before_latest_ratio=1.0,
        window_reference=2.0,
        window_ratio=0.5,
        window_log_slope=-1.0,
        consecutive_increases=0,
        has_nonfinite=False,
    )
    decision = PasSmootherDecision(accept=True, stop=False, reason="improved", trend=trend)
    adaptive = AdaptiveStationaryResult(
        x_best=jnp.asarray([1.0, 2.0]),
        best_residual_norm=1.0,
        residual_history=(2.0, 1.0),
        steps_completed=2,
        stop_reason="target",
        improved=True,
    )
    children, aux = adaptive.tree_flatten()
    rebuilt = AdaptiveStationaryResult.tree_unflatten(aux, children)
    summary = ConstrainedPASBranchSummary(
        reference_label="full",
        branch_sensitive=False,
        max_relative_spread=1.0e-4,
        weak_reference_labels=(),
        recommendation="converged_branch_consistent",
    )
    assert decision.trend.best_so_far == 1.0
    assert rebuilt.residual_history == (2.0, 1.0)
    assert not summary.has_reference_quality_blocker

    stack_policy = ActiveNativeStackPolicy(
        base_budget_fraction=0.5,
        base_budget_nbytes=1024,
        schwarz_requested=True,
        schwarz_max_size=16,
        max_coarse_size=8,
        coarse_solver_mode="dense",
    )
    field_policy = ActiveNativeFieldSplitSparseCoarsePolicy(
        requested_kind="active_native",
        requested_kind_normalized="active_native",
        output_kind="active_native_sparse_coarse",
        requested_base_kind="x_ell",
        is_multiline=False,
        is_angular_only=False,
        is_coupled_kinetic=True,
        max_coarse_size=8,
        coarse_solver_mode="dense",
        admission_probes=2,
        admission_max_relative_residual=0.25,
        admission_min_improvement=1.5,
    )
    coarse_policy = ActiveSparseCoarseResidualPolicy(
        requested_kind="active_sparse_coarse",
        requested_kind_normalized="active_sparse_coarse",
        base_kind="x_ell",
        output_kind="active_sparse_coarse",
        max_coarse_size=8,
        coarse_solver_mode="dense",
    )
    assert stack_policy.schwarz_requested
    assert field_policy.is_coupled_kinetic
    assert coarse_policy.base_kind == "x_ell"

    gate = SolverCandidateGate(
        accepted=False,
        reasons=("residual_gate",),
        residual_ratio=2.0,
        runtime_ratio=0.8,
        memory_ratio=0.9,
        memory_metric="active_rss_mb",
    )
    estimate = LinearSolveMemoryEstimate(
        unknowns=10,
        dtype="float64",
        dense_operator_nbytes=800,
        csr_operator_nbytes=120,
        gmres_basis_nbytes=80,
        bicgstab_work_nbytes=40,
        preconditioner_nbytes=20,
        compiled_temp_nbytes=10,
        device_count=2,
    )
    assert gate.reasons == ("residual_gate",)
    assert estimate.dense_total_nbytes == 910
    assert estimate.csr_per_device_nbytes == 115


def test_structured_operator_sparse_and_policy_contracts(tmp_path: Path) -> None:
    layout = _small_layout()
    operator = _tiny_block_operator()
    kinetic = RHS1KineticIndices(species=0, x=1, ell=1, theta=0, zeta=0)
    probe = RHS1BlockPreconditionerProbe(
        accepted=True,
        reason="residual_improved",
        residual_before_norm=2.0,
        residual_after_norm=0.5,
        improvement_ratio=4.0,
        target_residual_norm=1.0,
        target_ratio=0.5,
        x_candidate=jnp.asarray([1.0, 2.0]),
        factor_metadata={"kind": "block_jacobi"},
    )
    assert kinetic.x == 1
    assert probe.to_dict()["factor_metadata"] == {"kind": "block_jacobi"}

    correction = RHS1MatrixFreeLeastSquaresResidualCorrection.from_callbacks(
        operator=operator,
        prolong_fn=lambda coarse: jnp.asarray(coarse, dtype=jnp.float64),
        n_coarse=2,
        regularization=1.0e-12,
    )
    assert correction.n_coarse == 2
    assert correction.to_dict()["solver_kind"] == "precomputed_normal_inverse"

    metadata = DeviceCSRMetadata(
        shape=(2, 2),
        nnz=2,
        data_dtype="float64",
        index_dtype="int32",
        csr_nbytes=40,
        max_csr_nbytes=80,
        source="unit",
        all_arrays_same_device=True,
    )
    device_csr = DeviceCSR(
        data=jnp.asarray([1.0, 2.0]),
        indices=jnp.asarray([0, 1], dtype=jnp.int32),
        indptr=jnp.asarray([0, 1, 2], dtype=jnp.int32),
        shape=(2, 2),
        metadata=metadata,
    )
    matvec_result = np.asarray(device_csr.matvec(jnp.asarray([3.0, 4.0])))
    matvec_validation = MatvecValidationResult(
        samples=2,
        passed=True,
        max_abs_error=0.0,
        max_rel_error=0.0,
        rel_errors=(0.0, 0.0),
        rtol=1.0e-10,
        atol=1.0e-12,
        seed=1,
    )
    assert np.allclose(matvec_result, [3.0, 8.0])
    assert matvec_validation.to_dict()["samples"] == 2

    assembly = RHS1PartialFBlockAssembly(
        layout=layout,
        operator=operator,
        included_terms=("collisionless",),
        unsupported_terms=(),
        term_nnz_blocks={"collisionless": operator.nnz_blocks},
        term_data_nbytes={"collisionless": operator.data_nbytes},
    )
    selection = RHS1StructuredFBlockSelection(
        assembly=assembly,
        linear_operator=None,
        selected=False,
        reason="not_selected_in_contract_test",
    )
    csr_selection = RHS1StructuredFBlockCSRSelection(
        selection=selection,
        matrix=sp.eye(2, format="csr"),
        selected=True,
        reason="selected",
        cache_hit=False,
        build_s=0.01,
        metadata={"nnz": 2},
    )
    assert assembly.is_complete
    assert selection.to_dict()["selected"] is False
    assert np.allclose(csr_selection.matvec([2.0, 3.0]), [2.0, 3.0])

    compressed = build_rhs1_compressed_pitch_layout(layout)
    interior = RHS1ReducedPmatGroup("interior", "kinetic", np.asarray([0, 1]))
    separator = RHS1ReducedPmatGroup("separator", "kinetic", np.asarray([2]))
    tail = RHS1ReducedPmatGroup("tail", "tail", np.asarray([3]))
    root = RHS1ReducedPmatGroup("root", "constraint", np.asarray([4]))
    permutation = np.arange(compressed.reduced_size)
    symbolic_plan = RHS1ReducedPmatEliminationPlan(
        layout=compressed,
        interior_groups=(interior,),
        separator_group=separator,
        tail_group=tail,
        root_group=root,
        permutation=permutation,
        inverse_permutation=permutation,
        selected_separator_ells=(0,),
        selected_separator_x_indices=(0,),
        max_interior_group_size=2,
        max_separator_size=1,
    )
    assert interior.dense_lu_nbytes_estimate == 32
    assert symbolic_plan.metadata()["root_size"] == 1

    sparse_summary = V3SparsePatternSummary(
        shape=(5, 5),
        nnz=9,
        avg_row_nnz=1.8,
        max_row_nnz=3,
        include_phi1=False,
        constraint_scheme=1,
        has_fp=True,
        has_pas=False,
    )
    assert sparse_summary.to_dict()["nnz"] == 9

    phi1_policy = Phi1FrozenJacobianPolicy(mode="auto", use_cache=True, every=2)
    line_search = Phi1LineSearchPolicy(step_scale=1.0, factor=0.5, c1=1.0e-4, mode="armijo", maxiter=5)
    polish = ProjectedResidualPolishOutcome(
        result="same",
        accepted=False,
        full_residual_before=1.0,
        projected_residual_before=0.5,
        full_residual_after=0.9,
        projected_residual_after=0.4,
    )
    diagnostics = TransportPostsolveDiagnostics(
        transport_matrix=np.eye(2),
        particle_flux_vm_psi_hat=np.asarray([1.0]),
        heat_flux_vm_psi_hat=np.asarray([2.0]),
        fsab_flow=np.asarray([0.1]),
        transport_output_fields={"field": np.asarray([1.0])},
    )
    final = TransportRHSFinalizationResult(
        x_full=jnp.asarray([1.0]),
        ax_full=jnp.asarray([0.0]),
        residual_norm=1.0e-12,
    )
    assert phi1_policy.every == 2
    assert line_search.maxiter == 5
    assert polish.full_residual_after == 0.9
    assert diagnostics.transport_matrix.shape == (2, 2)
    assert final.residual_norm == 1.0e-12

    chunk_plan = PasRuntimeChunkPlan(
        element_count=100,
        itemsize=8,
        array_bytes=800,
        requested_block_size=32,
        block_size=16,
        live_array_count=3,
        estimated_live_array_bytes=2400,
        max_live_bytes=4096,
        live_byte_margin=1696,
        reduction_work_arrays=2,
        max_reduction_bytes=2048,
        estimated_reduction_bytes=1600,
        safe=True,
        reason="within_budget",
    )
    pas_result = Rhs1PasMatrixFreeResult(
        x=jnp.asarray([1.0]),
        residual_norm=0.1,
        initial_residual_norm=1.0,
        residual_history=(1.0, 0.1),
        accepted_steps=1,
        accepted=True,
        reason="improved",
        diagnostics={"chunk": chunk_plan.as_metadata()},
    )
    assert chunk_plan.as_metadata()["safe"] is True
    assert pas_result.accepted

    full_csr = RHS1FullCSRKineticPreconditioner(
        operator=None,
        native_factor=None,
        selected=False,
        kind="x_ell",
        reason="not_selected",
        setup_s=0.0,
        metadata={"budget": "low"},
    )
    with pytest.raises(RuntimeError, match="not selected"):
        full_csr.apply([1.0])
    assert full_csr.to_dict()["native_factor_available"] is False

    adapter = RHS1FullSystemMatrixFreeOperatorAdapter(op=SimpleNamespace(total_size=3))
    assert adapter.shape == (3, 3)
    assert np.asarray(adapter.blocks).shape == (1,)

    resolve_result = ResolveResult(path=tmp_path / "a.nc", tried=(tmp_path / "a.nc",))
    assert resolve_result.tried[0].name == "a.nc"


def test_mapped_grid_vmec_compare_and_pas_builder_contracts() -> None:
    moment = TransportMomentReport(
        objective=jnp.asarray(1.0),
        moment_loss=jnp.asarray(0.1),
        regularization_loss=jnp.asarray(0.01),
        powers=jnp.asarray([0.0, 2.0]),
        moments=jnp.asarray([1.0, 0.5]),
        references=jnp.asarray([1.0, 0.5]),
        relative_errors=jnp.asarray([0.0, 0.0]),
        regularization={"roughness": jnp.asarray(0.01)},
    )
    summary = TransportSolveSummary(
        max_residual_norm=1.0e-10,
        max_relative_residual_norm=1.0e-3,
        total_elapsed_time_s=0.2,
        total_size=10,
        active_size=8,
        active_fraction=0.8,
        n_x=4,
        use_active_dof_mode=True,
        solver_kinds=("gmres",),
        solve_methods=("structured_csr",),
    )
    matrix_error = TransportMatrixError(relative_frobenius=0.01, max_abs=0.001, reference_norm=1.0)
    row_fast = MappedTransportEvidenceRow(
        log_length=1.0,
        moment_objective=0.1,
        moment_loss=0.09,
        regularization_loss=0.01,
        matrix_relative_frobenius_error=0.02,
        matrix_max_abs_error=0.002,
        max_residual_norm=1.0e-10,
        max_relative_residual_norm=1.0e-3,
        total_elapsed_time_s=0.2,
        total_size=10,
        active_size=8,
        active_fraction=0.8,
        n_x=4,
        use_active_dof_mode=True,
        solver_kinds=("gmres",),
        solve_methods=("structured_csr",),
        min_dx=0.1,
        width_ratio=2.0,
        smoothness=0.5,
        jac_roughness=0.2,
        tail_mass_proxy=0.01,
    )
    row_accurate = MappedTransportEvidenceRow(
        **{**row_fast.__dict__, "moment_objective": 0.2, "matrix_relative_frobenius_error": 0.005}
    )
    report = MappedTransportEvidenceReport(reference_summary=summary, rows=(row_fast, row_accurate))
    assert float(moment.objective) == 1.0
    assert matrix_error.max_abs == 0.001
    assert report.best_by_moment is row_fast
    assert report.best_by_transport_error is row_accurate

    interpolation = VmecInterpolation(
        index_full=1,
        weight_full=0.25,
        index_half=0,
        weight_half=0.75,
        psi_n=0.5,
        psi_n_full=np.asarray([0.0, 1.0]),
        psi_n_half=np.asarray([0.5]),
    )
    compare = CompareResult(key="FSABjHat", max_abs=1.0e-9, max_rel=1.0e-8, ok=True)
    parity = H5DatasetParity(
        key="FSABjHat",
        status="ok",
        reference_shape=(1,),
        candidate_shape=(1,),
        max_abs=1.0e-9,
        max_rel=1.0e-8,
        atol=1.0e-8,
        rtol=1.0e-6,
    )
    assert interpolation.weight_full == 0.25
    assert compare.ok
    assert parity.to_json()["reference_shape"] == [1]

    def builder(**kwargs):
        return "built", kwargs

    families = RHS1PasFamilyBuilders(
        pas_tokamak_theta_applicable=lambda op: True,
        pas_tz_applicable=lambda op: True,
        pas_tz_memory_safe=lambda op: True,
        matvec_shard_axis=lambda op: None,
        device_count=lambda: 1,
        block_preconditioner_builder=builder,
        theta_schwarz_builder=builder,
        zeta_schwarz_builder=builder,
        theta_line_builder=builder,
        zeta_line_builder=builder,
        xblock_tz_lmax_builder=builder,
        xmg_builder=builder,
        xupwind_builder=builder,
        collision_builder=builder,
        tzfft_builder=builder,
        pas_hybrid_builder=builder,
    )
    composite = families.composite_builders()
    assert composite.pas_tz_applicable(object())
    assert composite.xmg_builder is builder
