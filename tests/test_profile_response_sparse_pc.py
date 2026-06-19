from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import SimpleNamespace

import numpy as np
import pytest
import jax.numpy as jnp
from scipy import sparse as scipy_sparse

import sfincs_jax.problems.profile_response.sparse_pc as sparse_pc_module
from sfincs_jax.problems.profile_response.active_projection import (
    expand_reduced_with_map,
    reduce_full_with_indices,
)
from sfincs_jax.problems.profile_response.diagnostics import (
    fortran_reduced_xblock_result_metadata,
)
from sfincs_jax.problems.profile_response.sparse_pc import (
    DirectTailMaterializationContext,
    DirectTailStructuredAdmissionContext,
    DirectTailStructuredBuildContext,
    DirectTailSupportModePreflightContext,
    DirectTailResidualRescuePolicy,
    DirectTailTrueActiveRescuePolicy,
    DirectTailCoupledCoarseRescuePolicy,
    SparsePCFactorPreflightPolicyContext,
    SparsePCFactorPreflightEvaluationContext,
    SparsePCResidualCandidateAcceptanceContext,
    SparsePCFactorDtypeRetryContext,
    SparsePCAutoPreflightRetrySelectionContext,
    SparsePCAutoPreflightRetryEvaluationContext,
    SparsePCGMRESControlPolicy,
    FortranReducedXBlockFactorBuildContext,
    FortranReducedXBlockGlobalCouplingStageContext,
    FortranReducedXBlockKrylovSetupContext,
    FortranReducedXBlockKrylovSolveContext,
    FortranReducedXBlockMomentSchurStageContext,
    MatvecCounter,
    SparsePCGMRESContext,
    SparsePCGMRESCompletionMessageContext,
    SparsePCGMRESFinalPayload,
    SparsePCMemoryBudgetPreflightContext,
    SparsePCPatternSetupContext,
    SparsePCPostMinresContext,
    SparsePCPostMinresUpdateContext,
    XBlockSubspaceCorrectionContext,
    SparseHostDirectPayload,
    SparseHostDirectFactorSolvePayload,
    SparseHostDirectPolishPayload,
    SparseHostDirectFallbackPayload,
    SparseHostOrILUFactorBuildContext,
    SparseILUPreconditionerBuildContext,
    SparseHostScipyPreconditionerBuildContext,
    ExplicitSparseOperatorBuildPolicy,
    ExplicitSparseOperatorBuildResult,
    SparseMinimumNormPayload,
    SparseMinimumNormPolicy,
    SparsePCGMRESResult,
    XBlockAssembledPreflightError,
    apply_fortran_reduced_xblock_global_coupling_stage,
    apply_fortran_reduced_xblock_initial_seed,
    apply_fortran_reduced_xblock_moment_schur_stage,
    apply_sparse_pc_post_minres,
    apply_sparse_pc_post_minres_if_needed,
    apply_sparse_pc_post_minres_from_driver_state,
    apply_xblock_subspace_correction_if_needed,
    build_fortran_reduced_xblock_factor_stage,
    build_explicit_sparse_operator_from_pattern,
    build_fortran_reduced_xblock_krylov_setup,
    build_sparse_pc_active_dof_setup,
    build_sparse_pc_pattern_setup,
    build_direct_tail_materialization_setup,
    build_direct_tail_structured_preconditioner_setup,
    build_xblock_assembled_equilibration_setup,
    build_xblock_assembled_device_setup,
    build_xblock_assembled_matvec_setup,
    build_xblock_assembled_operator_preflight_setup,
    build_xblock_krylov_matvec_setup,
    emit_sparse_pc_gmres_completion_from_driver_state,
    enforce_sparse_pc_memory_budget,
    evaluate_xblock_moment_schur_probe_result,
    evaluate_sparse_pc_factor_preflight,
    evaluate_sparse_pc_residual_candidate_acceptance,
    select_sparse_pc_auto_preflight_retry_candidates,
    evaluate_sparse_pc_auto_preflight_retry,
    evaluate_sparse_pc_factor_dtype_retry,
    explicit_sparse_pattern_progress_messages,
    resolve_sparse_pc_gmres_control_policy,
    failed_xblock_global_coupling_metadata,
    failed_xblock_two_level_metadata,
    failed_xblock_moment_schur_metadata,
    finalize_sparse_pc_gmres_from_driver_state,
    finalize_sparse_pc_gmres_with_dtype_retry_from_driver_state,
    finalize_xblock_global_coupling_metadata,
    finalize_xblock_two_level_metadata,
    finalize_xblock_moment_schur_metadata,
    prepare_fortran_reduced_xblock_initial_guess,
    prepare_xblock_initial_guess,
    resolve_fortran_reduced_sparse_pc_backend,
    resolve_fortran_reduced_xblock_factor_policy,
    resolve_fortran_reduced_xblock_global_coupling_policy,
    resolve_fortran_reduced_xblock_initial_seed_policy,
    resolve_fortran_reduced_xblock_krylov_policy,
    resolve_fortran_reduced_xblock_moment_schur_policy,
    resolve_sparse_pc_entry_policy,
    resolve_explicit_sparse_operator_build_policy,
    resolve_sparse_pc_factor_policy,
    resolve_sparse_minimum_norm_policy,
    sparse_pc_factor_dtype_retry_initial_guess,
    resolve_sparse_pc_factor_preflight_policy,
    resolve_direct_tail_structured_admission,
    resolve_direct_tail_residual_rescue_policy,
    resolve_direct_tail_true_active_rescue_policy,
    resolve_direct_tail_coupled_coarse_rescue_policy,
    run_direct_tail_support_mode_preflight,
    resolve_xblock_qi_device_admission_setup,
    resolve_xblock_qi_device_base_config_setup,
    resolve_xblock_qi_device_enrichment_config_setup,
    resolve_xblock_qi_device_multilevel_config_setup,
    resolve_xblock_qi_device_operator_reuse_setup,
    resolve_xblock_qi_galerkin_policy_setup,
    resolve_xblock_qi_seed_policy_setup,
    resolve_xblock_qi_two_level_policy_setup,
    resolve_xblock_global_coupling_policy_setup,
    resolve_xblock_moment_schur_policy_setup,
    resolve_xblock_seed_policy_setup,
    resolve_xblock_sparse_pc_setup,
    resolve_xblock_sparse_pc_side_policy_setup,
    resolve_xblock_two_level_policy_setup,
    fortran_reduced_xblock_final_payload_from_driver_state,
    run_fortran_reduced_xblock_krylov_solve,
    run_sparse_pc_gmres_once,
    retry_sparse_pc_factor_dtype_from_driver_state,
    retry_sparse_pc_factor_dtype_if_needed,
    sparse_pc_gmres_completion_message,
    sparse_pc_gmres_final_payload_from_driver_state,
    sparse_host_direct_solve_payload,
    sparse_host_direct_solve_from_pattern,
    solve_sparse_host_direct_from_available_factor,
    apply_sparse_host_direct_polish_if_needed,
    sparse_host_direct_fallback_payload,
    build_sparse_host_or_ilu_factor,
    build_sparse_ilu_preconditioner_from_cache,
    build_sparse_host_scipy_preconditioner,
    sparse_minimum_norm_solve_payload,
    sparse_minimum_norm_solve_from_pattern,
    sparse_minimum_norm_start_message,
    validate_explicit_sparse_host_request,
    finalize_xblock_assembled_operator_metadata,
    xblock_sparse_pc_final_metadata_from_driver_state,
    xblock_sparse_pc_final_payload_from_driver_state,
)


def _identity(v: jnp.ndarray) -> jnp.ndarray:
    return v


class _DefaultSparsePCDriverState(dict):
    def __missing__(self, key: str) -> object:
        self[key] = 1
        return self[key]


def _op(
    *, fp=False, pas=False, constraint_scheme=1, n_zeta=1, n_species=1
) -> SimpleNamespace:
    return SimpleNamespace(
        rhs_mode=1,
        constraint_scheme=constraint_scheme,
        include_phi1=False,
        n_zeta=n_zeta,
        n_species=n_species,
        point_at_x0=False,
        fblock=SimpleNamespace(
            fp=object() if fp else None,
            pas=object() if pas else None,
        ),
    )


def test_sparse_pc_active_dof_setup_disabled_uses_full_system_vectors() -> None:
    rhs = jnp.arange(6.0)
    setup = build_sparse_pc_active_dof_setup(
        op=SimpleNamespace(total_size=6),
        rhs=rhs,
        sparse_pc_use_active_dof=False,
        active_dof_indices=lambda _op: np.asarray([0, 2, 5]),
        reduce_full_with_indices=reduce_full_with_indices,
        expand_reduced_with_map=expand_reduced_with_map,
    )

    assert setup.active_idx_np is None
    assert setup.active_idx_jnp is None
    assert setup.full_to_active_jnp is None
    assert setup.linear_size == 6
    assert setup.messages == ()
    np.testing.assert_allclose(np.asarray(setup.rhs), np.asarray(rhs))
    np.testing.assert_allclose(np.asarray(setup.reduce_full(rhs + 10)), np.arange(6) + 10)
    np.testing.assert_allclose(np.asarray(setup.expand_reduced(rhs + 20)), np.arange(6) + 20)


def test_sparse_pc_active_dof_setup_builds_reduction_maps_and_message() -> None:
    rhs = jnp.arange(6.0)
    setup = build_sparse_pc_active_dof_setup(
        op=SimpleNamespace(total_size=6),
        rhs=rhs,
        sparse_pc_use_active_dof=True,
        active_dof_indices=lambda _op: np.asarray([0, 2, 5], dtype=np.int64),
        reduce_full_with_indices=reduce_full_with_indices,
        expand_reduced_with_map=expand_reduced_with_map,
    )

    assert setup.linear_size == 3
    assert setup.messages == (
        (
            1,
            "solve_v3_full_system_linear_gmres: sparse_pc_gmres active-DOF reduction "
            "enabled (size=3/6)",
        ),
    )
    np.testing.assert_array_equal(setup.active_idx_np, np.asarray([0, 2, 5]))
    np.testing.assert_array_equal(np.asarray(setup.active_idx_jnp), np.asarray([0, 2, 5]))
    np.testing.assert_array_equal(
        np.asarray(setup.full_to_active_jnp),
        np.asarray([1, 0, 2, 0, 0, 3], dtype=np.int32),
    )
    np.testing.assert_allclose(np.asarray(setup.rhs), np.asarray([0.0, 2.0, 5.0]))
    np.testing.assert_allclose(
        np.asarray(setup.reduce_full(jnp.arange(6.0) + 10.0)),
        np.asarray([10.0, 12.0, 15.0]),
    )
    np.testing.assert_allclose(
        np.asarray(setup.expand_reduced(jnp.asarray([1.0, 2.0, 3.0]))),
        np.asarray([1.0, 0.0, 2.0, 0.0, 0.0, 3.0]),
    )


@pytest.mark.parametrize(
    ("fortran_reduced", "active", "expected_scope", "expected_pattern"),
    (
        (False, False, "full", "generic_full"),
        (False, True, "active_dof", "generic_active"),
        (True, False, "fortran_reduced_full", "fortran_full"),
        (True, True, "fortran_reduced_active_dof", "fortran_active"),
    ),
)
def test_sparse_pc_pattern_setup_selects_scope_and_preserves_callbacks(
    fortran_reduced: bool,
    active: bool,
    expected_scope: str,
    expected_pattern: str,
) -> None:
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
    messages: list[tuple[int, str]] = []
    elapsed_values = iter((10.0, 10.25))

    def _record(name: str, pattern: str):
        def _inner(*args, **kwargs):
            calls.append((name, args, kwargs))
            return pattern

        return _inner

    result = build_sparse_pc_pattern_setup(
        SparsePCPatternSetupContext(
            op=SimpleNamespace(total_size=4),
            pattern_source_op="source-op",
            fortran_reduced_sparse_pc=fortran_reduced,
            sparse_pc_use_active_dof=active,
            active_idx_np=np.asarray([3, 1], dtype=np.int64) if active else None,
            preconditioner_x=2,
            preconditioner_xi=3,
            preconditioner_species=4,
            preconditioner_x_min_l=5,
            fp_dense_velocity_block=True,
            elapsed_s=lambda: next(elapsed_values),
            emit=lambda level, message: messages.append((level, message)),
            fortran_reduced_pattern_for_indices=_record(
                "fortran_active",
                "fortran_active",
            ),
            fortran_reduced_pattern=_record("fortran_full", "fortran_full"),
            conservative_pattern_for_indices=_record(
                "generic_active",
                "generic_active",
            ),
            conservative_pattern=_record("generic_full", "generic_full"),
            summarize_pattern=lambda _op, pattern: SimpleNamespace(
                nnz=7,
                avg_row_nnz=1.75,
                max_row_nnz=3,
                pattern=pattern,
            ),
        )
    )

    assert result.pattern == expected_pattern
    assert result.scope == expected_scope
    assert result.build_s == 0.25
    assert result.summary.nnz == 7
    assert calls[0][0] == expected_pattern
    if active:
        np.testing.assert_array_equal(calls[0][1][1], np.asarray([3, 1], dtype=np.int32))
    if fortran_reduced:
        assert calls[0][2] == {
            "preconditioner_x": 2,
            "preconditioner_xi": 3,
            "preconditioner_species": 4,
            "preconditioner_x_min_l": 5,
        }
    else:
        assert calls[0][2] == {"fp_dense_velocity_block": True}
    assert messages[0] == (
        1,
        "solve_v3_full_system_linear_gmres: sparse_pc_gmres building conservative pattern",
    )
    assert messages[1] == (
        1,
        "solve_v3_full_system_linear_gmres: sparse_pc_gmres pattern "
        f"scope={expected_scope} nnz=7 avg_row_nnz=1.75 max_row_nnz=3",
    )


def test_fortran_reduced_backend_policy_honors_explicit_backend_alias() -> None:
    setup = resolve_fortran_reduced_sparse_pc_backend(
        op=_op(fp=True, constraint_scheme=1, n_zeta=3),
        env={"SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND": "local-xblock"},
        fortran_reduced_sparse_pc=True,
        sparse_pc_linear_size=12,
    )

    assert setup.backend == "xblock"
    assert setup.reason == "env"
    assert setup.backend_raw == "local_xblock"
    assert setup.xblock_min_size == 100000
    assert setup.messages == ()


def test_fortran_reduced_backend_policy_auto_selects_large_full_fp_xblock() -> None:
    setup = resolve_fortran_reduced_sparse_pc_backend(
        op=_op(fp=True, constraint_scheme=1, n_zeta=3),
        env={"SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_MIN_SIZE": "10"},
        fortran_reduced_sparse_pc=True,
        sparse_pc_linear_size=12,
    )

    assert setup.backend == "xblock"
    assert setup.reason == "auto_large_full_fp_size>=10"
    assert not setup.backend_ignored_env


def test_fortran_reduced_backend_policy_direct_tail_required_forces_global() -> None:
    setup = resolve_fortran_reduced_sparse_pc_backend(
        op=_op(fp=True, constraint_scheme=1, n_zeta=3),
        env={
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_MIN_SIZE": "1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL": "1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PRECONDITIONER": (
                "active-fortran-v3-reduced-lu"
            ),
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_STRUCTURED_PC_REQUIRED": "1",
        },
        fortran_reduced_sparse_pc=True,
        sparse_pc_linear_size=12,
    )

    assert setup.backend == "global"
    assert setup.reason == "required_direct_tail_structured_pc"
    assert setup.direct_tail_pc_env == "active_fortran_v3_reduced_lu"
    assert setup.direct_tail_pc_explicit
    assert setup.direct_tail_structured_pc_required
    assert setup.direct_tail_structured_pc_forces_global


def test_fortran_reduced_backend_policy_ignored_env_reports_message() -> None:
    setup = resolve_fortran_reduced_sparse_pc_backend(
        op=_op(fp=False, constraint_scheme=1, n_zeta=1),
        env={"SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND": "unknown-backend"},
        fortran_reduced_sparse_pc=True,
        sparse_pc_linear_size=12,
    )

    assert setup.backend == "global"
    assert setup.reason == "auto_global"
    assert setup.backend_ignored_env
    assert setup.messages == (
        (
            1,
            "solve_v3_full_system_linear_gmres: ignoring unknown "
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND="
            "'unknown_backend'; using global",
        ),
    )


def test_sparse_pc_factor_policy_uses_large_fortran_reduced_defaults() -> None:
    setup = resolve_sparse_pc_factor_policy(
        env={},
        constrained_pas_pc=False,
        tokamak_fp_pc=False,
        fortran_reduced_sparse_pc=True,
        sparse_pc_linear_size=100000,
        pc_maxiter=120,
        default_permc_spec="MMD_ATA",
        host_sparse_factor_dtype=lambda **_kwargs: np.dtype(np.float32),
    )

    assert setup.pc_shift == 1.0e-8
    assert setup.factorization == "ilu"
    assert setup.default_factor_kind == "ilu"
    assert setup.default_ilu_fill_factor == 2.0
    assert setup.default_ilu_drop_tol == 1.0e-3
    assert setup.default_pattern_color_batch == 16
    assert setup.factor_dtype_initial == np.dtype(np.float64)
    assert setup.factor_dtype_used == np.dtype(np.float64)
    assert setup.factor_dtype_retry is None
    assert setup.default_permc_spec == "MMD_ATA"
    assert setup.permc_spec == "MMD_ATA"
    assert setup.fp32_probe_maxiter == 2
    assert setup.first_attempt_maxiter == 120


def test_sparse_pc_factor_policy_honors_env_overrides_and_fp32_probe() -> None:
    setup = resolve_sparse_pc_factor_policy(
        env={
            "SFINCS_JAX_RHSMODE1_SPARSE_PC_SHIFT": "2e-4",
            "SFINCS_JAX_EXPLICIT_SPARSE_FACTOR_KIND": "diagonal",
            "SFINCS_JAX_RHSMODE1_SPARSE_PC_FACTOR_DTYPE": "fp32",
            "SFINCS_JAX_EXPLICIT_SPARSE_PERMC_SPEC": "COLAMD",
            "SFINCS_JAX_RHSMODE1_SPARSE_PC_FP32_PROBE_MAXITER": "7",
        },
        constrained_pas_pc=True,
        tokamak_fp_pc=False,
        fortran_reduced_sparse_pc=False,
        sparse_pc_linear_size=9,
        pc_maxiter=20,
        default_permc_spec="MMD_AT_PLUS_A",
        host_sparse_factor_dtype=lambda **_kwargs: np.dtype(np.float64),
    )

    assert setup.pc_shift == 2.0e-4
    assert setup.factorization == "jacobi"
    assert setup.factor_dtype_initial == np.dtype(np.float32)
    assert setup.factor_dtype_used == np.dtype(np.float32)
    assert setup.permc_spec == "COLAMD"
    assert setup.fp32_probe_maxiter == 7
    assert setup.first_attempt_maxiter == 7


def test_sparse_pc_factor_policy_can_defer_dtype_to_host_policy() -> None:
    calls: list[dict[str, object]] = []

    def host_dtype(**kwargs):
        calls.append(kwargs)
        return np.dtype(np.float32)

    setup = resolve_sparse_pc_factor_policy(
        env={
            "SFINCS_JAX_HOST_SPARSE_FACTOR_DTYPE": "auto",
            "SFINCS_JAX_EXPLICIT_SPARSE_FACTOR_KIND": "ilu",
            "SFINCS_JAX_RHSMODE1_SPARSE_PC_SHIFT": "bad",
            "SFINCS_JAX_EXPLICIT_SPARSE_PERMC_SPEC": "bad",
            "SFINCS_JAX_RHSMODE1_SPARSE_PC_FP32_PROBE_MAXITER": "bad",
        },
        constrained_pas_pc=False,
        tokamak_fp_pc=True,
        fortran_reduced_sparse_pc=False,
        sparse_pc_linear_size=33,
        pc_maxiter=5,
        default_permc_spec="NATURAL",
        host_sparse_factor_dtype=host_dtype,
    )

    assert setup.pc_shift == 1.0e-8
    assert setup.factorization == "ilu"
    assert setup.factor_dtype_initial == np.dtype(np.float32)
    assert setup.permc_spec == "NATURAL"
    assert setup.fp32_probe_maxiter == 2
    assert setup.first_attempt_maxiter == 2
    assert calls == [
        {
            "size": 33,
            "factorization": "ilu",
            "use_implicit": False,
        }
    ]


def test_sparse_pc_factor_dtype_retry_promotes_failed_fp32_probe() -> None:
    decision = evaluate_sparse_pc_factor_dtype_retry(
        factor_dtype_used=np.dtype(np.float32),
        residual_norm=2.0,
        target=1.0,
    )

    assert decision.retry is True
    assert decision.factor_dtype_used == np.dtype(np.float64)
    assert decision.factor_dtype_retry == "float64"


def test_sparse_pc_factor_dtype_retry_promotes_nonfinite_fp32_probe() -> None:
    decision = evaluate_sparse_pc_factor_dtype_retry(
        factor_dtype_used=np.dtype(np.float32),
        residual_norm=float("nan"),
        target=1.0,
    )

    assert decision.retry is True
    assert decision.factor_dtype_used == np.dtype(np.float64)
    assert decision.factor_dtype_retry == "float64"


def test_sparse_pc_factor_dtype_retry_keeps_successful_or_fp64_probe() -> None:
    fp32_success = evaluate_sparse_pc_factor_dtype_retry(
        factor_dtype_used=np.dtype(np.float32),
        residual_norm=0.5,
        target=1.0,
    )
    fp64_failure = evaluate_sparse_pc_factor_dtype_retry(
        factor_dtype_used=np.dtype(np.float64),
        residual_norm=2.0,
        target=1.0,
    )

    assert fp32_success.retry is False
    assert fp32_success.factor_dtype_used == np.dtype(np.float32)
    assert fp32_success.factor_dtype_retry is None
    assert fp64_failure.retry is False
    assert fp64_failure.factor_dtype_used == np.dtype(np.float64)
    assert fp64_failure.factor_dtype_retry is None


def test_sparse_pc_factor_dtype_retry_initial_guess_uses_finite_candidate() -> None:
    fallback = jnp.asarray([9.0, 9.0])
    finite = sparse_pc_factor_dtype_retry_initial_guess(
        np.asarray([1.0, 2.0]),
        fallback,
    )
    nonfinite = sparse_pc_factor_dtype_retry_initial_guess(
        np.asarray([1.0, np.nan]),
        fallback,
    )

    np.testing.assert_allclose(np.asarray(finite), np.asarray([1.0, 2.0]))
    np.testing.assert_allclose(np.asarray(nonfinite), np.asarray([9.0, 9.0]))


def test_retry_sparse_pc_factor_dtype_if_needed_preserves_successful_probe_state() -> None:
    build_calls: list[np.dtype] = []
    run_calls: list[tuple[jnp.ndarray, int]] = []

    result = retry_sparse_pc_factor_dtype_if_needed(
        SparsePCFactorDtypeRetryContext(
            factor_dtype_used=np.dtype(np.float32),
            factor_dtype_retry=None,
            residual_norm=0.5,
            preconditioned_residual_norm=0.25,
            history=(1.0, 0.5),
            target=1.0,
            x=np.asarray([1.0, 2.0]),
            x0_fallback=jnp.asarray([0.0, 0.0]),
            solve_s=3.0,
            pc_maxiter=20,
            operator_bundle="operator0",
            factor_bundle="factor0",
            elapsed_s=lambda: 0.0,
            emit=None,
            build_factor=lambda dtype: build_calls.append(dtype) or ("operator1", "factor1"),
            run_gmres_once=lambda x0, maxiter: run_calls.append((x0, maxiter))
            or (np.zeros(2), 0.0, 0.0, (), 0.0),
        )
    )

    assert result.retried is False
    assert result.factor_dtype_used == np.dtype(np.float32)
    assert result.factor_dtype_retry is None
    assert result.operator_bundle == "operator0"
    assert result.factor_bundle == "factor0"
    assert result.factor_s_increment == 0.0
    assert result.setup_s is None
    np.testing.assert_allclose(result.x, np.asarray([1.0, 2.0]))
    assert result.residual_norm == 0.5
    assert result.preconditioned_residual_norm == 0.25
    assert result.history == (1.0, 0.5)
    assert result.solve_s == 3.0
    assert build_calls == []
    assert run_calls == []


def test_retry_sparse_pc_factor_dtype_if_needed_rebuilds_and_reruns_failed_fp32_probe() -> None:
    messages: list[str] = []
    times = iter((10.0, 10.4, 10.5))
    build_calls: list[np.dtype] = []
    run_calls: list[tuple[np.ndarray, int]] = []

    def build_factor(dtype: np.dtype):
        build_calls.append(np.dtype(dtype))
        return "operator64", "factor64"

    def run_gmres_once(x0: jnp.ndarray, maxiter: int):
        run_calls.append((np.asarray(x0), int(maxiter)))
        return np.asarray([3.0, 4.0]), 0.1, 0.05, (0.5, 0.1), 7.0

    result = retry_sparse_pc_factor_dtype_if_needed(
        SparsePCFactorDtypeRetryContext(
            factor_dtype_used=np.dtype(np.float32),
            factor_dtype_retry=None,
            residual_norm=2.0,
            preconditioned_residual_norm=1.0,
            history=(2.0,),
            target=1.0,
            x=np.asarray([1.0, 2.0]),
            x0_fallback=jnp.asarray([9.0, 9.0]),
            solve_s=3.0,
            pc_maxiter=20,
            operator_bundle="operator0",
            factor_bundle="factor0",
            elapsed_s=lambda: next(times),
            emit=lambda _level, msg: messages.append(msg),
            build_factor=build_factor,
            run_gmres_once=run_gmres_once,
        )
    )

    assert result.retried is True
    assert result.factor_dtype_used == np.dtype(np.float64)
    assert result.factor_dtype_retry == "float64"
    assert result.operator_bundle == "operator64"
    assert result.factor_bundle == "factor64"
    assert result.factor_s_increment == pytest.approx(0.4)
    assert result.setup_s == pytest.approx(10.5)
    np.testing.assert_allclose(result.x, np.asarray([3.0, 4.0]))
    assert result.residual_norm == 0.1
    assert result.preconditioned_residual_norm == 0.05
    assert result.history == (0.5, 0.1)
    assert result.solve_s == 10.0
    assert build_calls == [np.dtype(np.float64)]
    np.testing.assert_allclose(run_calls[0][0], np.asarray([1.0, 2.0]))
    assert run_calls[0][1] == 20
    assert any("factor_dtype=float64" in msg for msg in messages)


def test_retry_sparse_pc_factor_dtype_from_driver_state_forwards_build_policy() -> None:
    times = iter((2.0, 2.25, 2.5))
    build_kwargs: list[dict[str, object]] = []
    run_calls: list[tuple[np.ndarray, int]] = []

    def build_factor(**kwargs):
        build_kwargs.append(kwargs)
        return "operator64", "factor64"

    def run_gmres_once(x0: jnp.ndarray, *, maxiter_arg: int):
        run_calls.append((np.asarray(x0), int(maxiter_arg)))
        return np.asarray([4.0, 5.0]), 0.2, 0.1, (0.4, 0.2), 6.0

    state = {
        "_sparse_pc_factor_mv": "matvec",
        "sparse_pc_linear_size": 12,
        "rhs": jnp.ones(3),
        "pattern": "pattern",
        "emit": None,
        "constrained_pas_pc": False,
        "tokamak_fp_pc": True,
        "fortran_reduced_sparse_pc": False,
        "sparse_pc_default_permc_spec": "COLAMD",
        "sparse_pc_default_factor_kind": "ilu",
        "sparse_pc_default_ilu_fill_factor": 3.0,
        "sparse_pc_default_ilu_drop_tol": 1.0e-4,
        "sparse_pc_default_pattern_color_batch": 5,
        "sparse_pc_factor_dtype_used": np.dtype(np.float32),
        "sparse_pc_factor_dtype_retry": None,
        "residual_norm_sparse_pc": 2.0,
        "rn_pc": 1.0,
        "history": (2.0,),
        "target": 1.0,
        "x_np": np.asarray([1.0, 2.0]),
        "x0_sparse": jnp.asarray([9.0, 9.0]),
        "solve_s": 4.0,
        "pc_maxiter": 11,
        "_operator_bundle_pc": "operator0",
        "factor_bundle_pc": "factor0",
        "sparse_timer": SimpleNamespace(elapsed_s=lambda: next(times)),
    }

    result = retry_sparse_pc_factor_dtype_from_driver_state(
        state,
        build_host_sparse_direct_factor_from_matvec=build_factor,
        run_sparse_pc_gmres_once_callback=run_gmres_once,
    )

    assert result.retried is True
    assert result.factor_dtype_used == np.dtype(np.float64)
    assert result.factor_dtype_retry == "float64"
    assert build_kwargs[0]["matvec"] == "matvec"
    assert build_kwargs[0]["n"] == 12
    assert build_kwargs[0]["factor_dtype"] == np.dtype(np.float64)
    assert build_kwargs[0]["default_diag_pivot_thresh"] == 0.0
    assert build_kwargs[0]["default_permc_spec"] == "COLAMD"
    assert build_kwargs[0]["default_factor_kind"] == "ilu"
    assert build_kwargs[0]["default_pattern_color_batch"] == 5
    np.testing.assert_allclose(run_calls[0][0], np.asarray([1.0, 2.0]))
    assert run_calls[0][1] == 11
    assert result.factor_s_increment == pytest.approx(0.25)
    assert result.setup_s == pytest.approx(2.5)
    assert result.solve_s == pytest.approx(10.0)


def test_sparse_pc_memory_budget_preflight_is_noop_without_positive_budget() -> None:
    calls = 0

    def estimate(**_kwargs):
        nonlocal calls
        calls += 1
        return SimpleNamespace(csr_total_nbytes=10**9, dense_total_nbytes=0)

    enforce_sparse_pc_memory_budget(
        SparsePCMemoryBudgetPreflightContext(
            env={"SFINCS_JAX_RHSMODE1_SPARSE_PC_MAX_ESTIMATED_MB": "bad"},
            unknowns=10,
            gmres_restart=20,
            csr_nnz=30,
            dtype=np.dtype(np.float64),
            device_count=1,
            estimate_sparse_pc_memory=estimate,
        )
    )

    assert calls == 0


def test_sparse_pc_memory_budget_preflight_passes_estimator_inputs() -> None:
    calls: list[dict[str, object]] = []

    def estimate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(csr_total_nbytes=2_000_000, dense_total_nbytes=0)

    enforce_sparse_pc_memory_budget(
        SparsePCMemoryBudgetPreflightContext(
            env={
                "SFINCS_JAX_RHSMODE1_SPARSE_PC_MAX_ESTIMATED_MB": "3",
                "SFINCS_JAX_RHSMODE1_SPARSE_PC_FACTOR_FILL_ESTIMATE": "4.5",
            },
            unknowns=10,
            gmres_restart=20,
            csr_nnz=30,
            dtype=np.dtype(np.float32),
            device_count=0,
            estimate_sparse_pc_memory=estimate,
        )
    )

    assert calls == [
        {
            "unknowns": 10,
            "gmres_restart": 20,
            "csr_nnz": 30,
            "dtype": np.dtype(np.float32),
            "factor_fill_estimate": 4.5,
            "device_count": 1,
        }
    ]


def test_sparse_pc_memory_budget_preflight_raises_same_budget_error() -> None:
    def estimate(**_kwargs):
        return SimpleNamespace(csr_total_nbytes=0, dense_total_nbytes=5_500_000)

    with pytest.raises(MemoryError, match="estimated=5.500 MB budget=5.000 MB"):
        enforce_sparse_pc_memory_budget(
            SparsePCMemoryBudgetPreflightContext(
                env={
                    "SFINCS_JAX_RHSMODE1_SPARSE_PC_MAX_ESTIMATED_MB": "5",
                    "SFINCS_JAX_RHSMODE1_SPARSE_PC_FACTOR_FILL_ESTIMATE": "bad",
                },
                unknowns=11,
                gmres_restart=22,
                csr_nnz=33,
                dtype=np.dtype(np.float64),
                device_count=2,
                estimate_sparse_pc_memory=estimate,
            )
        )


def _direct_tail_context(
    *,
    env: dict[str, str] | None,
    sparse_pc_linear_size: int = 100000,
    active_indices: np.ndarray | None = None,
    build_direct_tail_bundle=None,
    elapsed_s=None,
    messages: list[tuple[int, str]] | None = None,
) -> DirectTailMaterializationContext:
    if build_direct_tail_bundle is None:
        def build_direct_tail_bundle(**_kwargs):
            return None

    if elapsed_s is None:
        def elapsed_s():
            return 0.0

    return DirectTailMaterializationContext(
        env=env or {},
        op=SimpleNamespace(rhs_mode=1, constraint_scheme=1, phi1_size=0),
        op_pc="op-pc",
        pattern="pattern",
        active_indices=active_indices,
        sparse_pc_use_active_dof=active_indices is not None,
        reduce_full=_identity,
        expand_reduced=_identity,
        pc_shift=1.0e-8,
        dtype=np.dtype(np.float64),
        factor_dtype=np.dtype(np.float64),
        sparse_pc_linear_size=sparse_pc_linear_size,
        default_pattern_color_batch=9,
        elapsed_s=elapsed_s,
        emit=None if messages is None else lambda level, msg: messages.append((level, msg)),
        is_direct_reduced_pmat_pc_kind=lambda kind: kind == "direct_pmat",
        build_direct_tail_bundle=build_direct_tail_bundle,
        build_structured_rhs1_full_csr_operator_bundle_callback=lambda **kwargs: kwargs,
    )


def test_direct_tail_materialization_respects_disabled_env_without_builder_call() -> None:
    calls = 0

    def builder(**_kwargs):
        nonlocal calls
        calls += 1
        return object()

    result = build_direct_tail_materialization_setup(
        _direct_tail_context(
            env={"SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL": "0"},
            build_direct_tail_bundle=builder,
        )
    )

    assert result.direct_tail_default is True
    assert result.enabled is False
    assert result.built is False
    assert result.operator_bundle is None
    assert result.error is None
    assert calls == 0


def test_direct_tail_materialization_skips_direct_reduced_pmat_request() -> None:
    messages: list[tuple[int, str]] = []
    result = build_direct_tail_materialization_setup(
        _direct_tail_context(
            env={
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL": "1",
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PRECONDITIONER": (
                    "direct-pmat"
                ),
            },
            build_direct_tail_bundle=lambda **_kwargs: pytest.fail("unexpected build"),
            messages=messages,
        )
    )

    assert result.enabled is True
    assert result.direct_reduced_pmat_requested is True
    assert result.built is False
    assert result.pc_env == "direct_pmat"
    assert "materialization skipped" in messages[0][1]


def test_direct_tail_materialization_forwards_builder_args_and_emits_complete() -> None:
    calls: list[dict[str, object]] = []
    bundle = SimpleNamespace(matrix="csr")
    elapsed_values = iter((1.0, 1.75))
    messages: list[tuple[int, str]] = []

    def builder(**kwargs):
        calls.append(kwargs)
        return bundle

    result = build_direct_tail_materialization_setup(
        _direct_tail_context(
            env={
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL": "1",
                "SFINCS_JAX_EXPLICIT_SPARSE_CSR_MAX_MB": "bad",
                "SFINCS_JAX_EXPLICIT_SPARSE_DROP_TOL": "bad",
                "SFINCS_JAX_EXPLICIT_SPARSE_PATTERN_COLOR_BATCH": "bad",
            },
            active_indices=np.asarray([2, 0], dtype=np.int64),
            build_direct_tail_bundle=builder,
            elapsed_s=lambda: next(elapsed_values),
            messages=messages,
        )
    )

    assert result.built is True
    assert result.operator_bundle is bundle
    assert result.error is None
    assert calls[0]["active_indices"].tolist() == [2, 0]
    assert calls[0]["csr_max_mb"] == 512.0
    assert calls[0]["drop_tol"] == 0.0
    assert calls[0]["color_batch"] == 9
    assert calls[0]["pc_shift"] == 1.0e-8
    assert "materialization start" in messages[0][1]
    assert "materialization complete elapsed_s=0.750" in messages[1][1]


def test_direct_tail_materialization_records_not_selected_and_exception() -> None:
    none_messages: list[tuple[int, str]] = []
    none_elapsed = iter((3.0, 3.5))
    none_result = build_direct_tail_materialization_setup(
        _direct_tail_context(
            env={"SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL": "1"},
            build_direct_tail_bundle=lambda **_kwargs: None,
            elapsed_s=lambda: next(none_elapsed),
            messages=none_messages,
        )
    )

    assert none_result.built is False
    assert none_result.error is None
    assert "materialization not selected elapsed_s=0.500" in none_messages[1][1]

    err_messages: list[tuple[int, str]] = []
    err_elapsed = iter((4.0, 4.25))

    def broken_builder(**_kwargs):
        raise RuntimeError("boom")

    err_result = build_direct_tail_materialization_setup(
        _direct_tail_context(
            env={"SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL": "1"},
            build_direct_tail_bundle=broken_builder,
            elapsed_s=lambda: next(err_elapsed),
            messages=err_messages,
        )
    )

    assert err_result.built is False
    assert err_result.operator_bundle is None
    assert err_result.error == "RuntimeError: boom"
    assert "materialization disabled after failure elapsed_s=0.250" in err_messages[1][1]


@dataclass(frozen=True)
class _FakeStructuredPreconditioner:
    operator: object | None = object()
    selected: bool = True
    kind: str = "fake_lu"
    reason: str = "ok"
    setup_s: float = 0.25
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "selected": bool(self.selected),
            "kind": str(self.kind),
            "reason": str(self.reason),
            "setup_s": float(self.setup_s),
            "metadata": dict(self.metadata),
        }


class _FakeLayout:
    total_size = 3
    f_size = 2

    def to_dict(self) -> dict[str, int]:
        return {"total_size": 3, "f_size": 2}


def _fake_cache_metadata(
    preconditioner: _FakeStructuredPreconditioner,
    *,
    cache_hit: bool,
    cache_key: tuple[object, ...],
) -> _FakeStructuredPreconditioner:
    metadata = dict(preconditioner.metadata)
    metadata["cache_hit"] = bool(cache_hit)
    metadata["cache_key"] = cache_key
    return replace(preconditioner, metadata=metadata)


def _fake_factor_bundle(**kwargs) -> SimpleNamespace:
    return SimpleNamespace(**kwargs)


def test_direct_tail_structured_admission_auto_defaults_when_large_bundle_exists() -> None:
    calls: list[dict[str, object]] = []

    def default_max_mb(**kwargs):
        calls.append(kwargs)
        return 768.0

    result = resolve_direct_tail_structured_admission(
        DirectTailStructuredAdmissionContext(
            env={},
            pc_env="",
            operator_bundle=object(),
            direct_reduced_pmat_requested=False,
            sparse_pc_linear_size=100000,
            default_max_mb=default_max_mb,
        )
    )

    assert result.pc_env == ""
    assert result.requested == "auto"
    assert result.auto_default is True
    assert result.required is True
    assert result.setup_allowed is True
    assert result.max_mb_auto is True
    assert result.max_mb == 768.0
    assert result.regularization == 1.0e-12
    assert calls == [{"requested_kind": "auto", "active_size": 100000}]


def test_direct_tail_structured_admission_explicit_kind_is_required_and_uses_env_caps() -> None:
    result = resolve_direct_tail_structured_admission(
        DirectTailStructuredAdmissionContext(
            env={
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PC_MAX_MB": "12.5",
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PC_REGULARIZATION": "4e-9",
            },
            pc_env="active-fortran-v3-reduced-lu",
            operator_bundle=object(),
            direct_reduced_pmat_requested=False,
            sparse_pc_linear_size=9,
            default_max_mb=lambda **_kwargs: pytest.fail("unexpected default cap"),
        )
    )

    assert result.requested == "active_fortran_v3_reduced_lu"
    assert result.required is True
    assert result.setup_allowed is True
    assert result.max_mb_auto is False
    assert result.max_mb == 12.5
    assert result.regularization == 4.0e-9


def test_direct_tail_structured_admission_fail_closed_and_overrides() -> None:
    result = resolve_direct_tail_structured_admission(
        DirectTailStructuredAdmissionContext(
            env={
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_STRUCTURED_PC_FAIL_CLOSED_SIZE": "10",
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_STRUCTURED_PC_REQUIRED": "0",
            },
            pc_env="auto",
            operator_bundle=object(),
            direct_reduced_pmat_requested=False,
            sparse_pc_linear_size=12,
            default_max_mb=lambda **_kwargs: 512.0,
        )
    )

    assert result.requested == "auto"
    assert result.fail_closed_size == 10
    assert result.auto_large_fail_closed is True
    assert result.required is False
    assert result.setup_allowed is True


def test_direct_tail_structured_admission_allows_direct_reduced_pmat_without_bundle() -> None:
    result = resolve_direct_tail_structured_admission(
        DirectTailStructuredAdmissionContext(
            env={
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PC_MAX_MB": "bad",
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PC_REGULARIZATION": "bad",
            },
            pc_env="direct-pmat",
            operator_bundle=None,
            direct_reduced_pmat_requested=True,
            sparse_pc_linear_size=5,
            default_max_mb=lambda **_kwargs: pytest.fail("unexpected default cap"),
        )
    )

    assert result.requested == "direct_pmat"
    assert result.setup_allowed is True
    assert result.required is True
    assert result.max_mb_auto is False
    assert result.max_mb == 512.0
    assert result.regularization == 1.0e-12


def test_direct_tail_structured_admission_no_bundle_blocks_default_setup() -> None:
    result = resolve_direct_tail_structured_admission(
        DirectTailStructuredAdmissionContext(
            env={},
            pc_env="",
            operator_bundle=None,
            direct_reduced_pmat_requested=False,
            sparse_pc_linear_size=100000,
            default_max_mb=lambda **_kwargs: pytest.fail("unexpected default cap"),
        )
    )

    assert result.auto_default is False
    assert result.requested is None
    assert result.setup_allowed is False
    assert result.max_mb_auto is False
    assert result.max_mb == 0.0


def test_direct_tail_structured_build_uses_direct_reduced_pmat_builder() -> None:
    calls: dict[str, object] = {}
    active_indices = np.array([0, 2], dtype=np.int64)

    def direct_builder(**kwargs) -> _FakeStructuredPreconditioner:
        calls.update(kwargs)
        return _FakeStructuredPreconditioner(metadata={"factor_nbytes_actual": 123})

    result = build_direct_tail_structured_preconditioner_setup(
        DirectTailStructuredBuildContext(
            env={},
            op=object(),
            operator_bundle=None,
            active_indices=active_indices,
            requested_kind="direct_reduced_pmat_lu",
            direct_reduced_pmat_requested=True,
            sparse_pc_linear_size=9,
            max_mb=2.0,
            regularization=1.0e-9,
            preconditioner_x=1,
            preconditioner_xi=2,
            preconditioner_species=3,
            preconditioner_x_min_l=4,
            layout_from_operator=lambda _op: _FakeLayout(),
            build_direct_active_preconditioner=direct_builder,
            build_active_projected_preconditioner=lambda **_kwargs: pytest.fail("unexpected active builder"),
            cache={},
            cache_key=lambda **_kwargs: pytest.fail("unexpected cache key"),
            with_cache_metadata=_fake_cache_metadata,
            factor_bundle=_fake_factor_bundle,
        )
    )

    assert result.ready is True
    assert result.selected is True
    assert result.error is None
    assert result.max_nbytes == 2 * 1024 * 1024
    assert result.cache_hit is False
    assert result.cache_key == ("direct_reduced_pmat_pc_cache_disabled", "direct_reduced_pmat_lu", 9, (1, 2, 3, 4))
    assert result.factor_bundle.factor_nbytes_estimate == 123
    assert calls["active_indices"] is active_indices
    assert calls["max_factor_nbytes"] == 2 * 1024 * 1024
    assert calls["max_csr_nbytes"] == 2 * 1024 * 1024
    assert calls["include_jacobian_terms"] is True


def test_direct_tail_structured_build_reuses_cached_active_preconditioner() -> None:
    cache: dict[tuple[object, ...], object] = {
        ("cached",): _FakeStructuredPreconditioner(metadata={"factor_nbytes_estimate": 77})
    }
    bundle = SimpleNamespace(matrix=scipy_sparse.eye(3, format="csr"))

    result = build_direct_tail_structured_preconditioner_setup(
        DirectTailStructuredBuildContext(
            env={},
            op=object(),
            operator_bundle=bundle,
            active_indices=None,
            requested_kind="active_fortran_v3_reduced_lu",
            direct_reduced_pmat_requested=False,
            sparse_pc_linear_size=3,
            max_mb=1.0,
            regularization=1.0e-12,
            preconditioner_x=0,
            preconditioner_xi=1,
            preconditioner_species=0,
            preconditioner_x_min_l=0,
            layout_from_operator=lambda _op: _FakeLayout(),
            build_direct_active_preconditioner=lambda **_kwargs: pytest.fail("unexpected direct builder"),
            build_active_projected_preconditioner=lambda **_kwargs: pytest.fail("unexpected active builder"),
            cache=cache,
            cache_key=lambda **_kwargs: ("cached",),
            with_cache_metadata=_fake_cache_metadata,
            factor_bundle=_fake_factor_bundle,
        )
    )

    assert result.ready is True
    assert result.cache_hit is True
    assert result.factor_bundle.operator is bundle
    assert result.factor_bundle.factor_nbytes_estimate == 77
    assert result.preconditioner.metadata["cache_hit"] is True


def test_direct_tail_structured_build_can_disable_active_cache() -> None:
    calls: dict[str, object] = {}
    cache: dict[tuple[object, ...], object] = {}
    bundle = SimpleNamespace(matrix=scipy_sparse.eye(3, format="csr"))

    def active_builder(**kwargs) -> _FakeStructuredPreconditioner:
        calls.update(kwargs)
        return _FakeStructuredPreconditioner(selected=False, operator=None, reason="memory_cap")

    result = build_direct_tail_structured_preconditioner_setup(
        DirectTailStructuredBuildContext(
            env={"SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PC_CACHE": "0"},
            op=object(),
            operator_bundle=bundle,
            active_indices=np.array([1], dtype=np.int64),
            requested_kind="active_fortran_v3_reduced_ilu",
            direct_reduced_pmat_requested=False,
            sparse_pc_linear_size=3,
            max_mb=0.5,
            regularization=2.0e-8,
            preconditioner_x=0,
            preconditioner_xi=1,
            preconditioner_species=0,
            preconditioner_x_min_l=0,
            layout_from_operator=lambda _op: _FakeLayout(),
            build_direct_active_preconditioner=lambda **_kwargs: pytest.fail("unexpected direct builder"),
            build_active_projected_preconditioner=active_builder,
            cache=cache,
            cache_key=lambda **_kwargs: pytest.fail("unexpected cache key"),
            with_cache_metadata=_fake_cache_metadata,
            factor_bundle=_fake_factor_bundle,
        )
    )

    assert result.ready is False
    assert result.selected is False
    assert result.reason == "memory_cap"
    assert result.error is None
    assert result.cache_key == ("direct_tail_structured_pc_cache_disabled", "active_fortran_v3_reduced_ilu", (0, 1, 0, 0))
    assert cache == {}
    assert calls["regularization"] == 2.0e-8


def test_direct_tail_structured_build_reports_missing_matrix_exception() -> None:
    result = build_direct_tail_structured_preconditioner_setup(
        DirectTailStructuredBuildContext(
            env={},
            op=object(),
            operator_bundle=None,
            active_indices=None,
            requested_kind="active_fortran_v3_reduced_lu",
            direct_reduced_pmat_requested=False,
            sparse_pc_linear_size=3,
            max_mb=1.0,
            regularization=1.0e-12,
            preconditioner_x=0,
            preconditioner_xi=0,
            preconditioner_species=0,
            preconditioner_x_min_l=0,
            layout_from_operator=lambda _op: _FakeLayout(),
            build_direct_active_preconditioner=lambda **_kwargs: pytest.fail("unexpected direct builder"),
            build_active_projected_preconditioner=lambda **_kwargs: pytest.fail("unexpected active builder"),
            cache={},
            cache_key=lambda **_kwargs: pytest.fail("unexpected cache key"),
            with_cache_metadata=_fake_cache_metadata,
            factor_bundle=_fake_factor_bundle,
        )
    )

    assert result.ready is False
    assert result.selected is False
    assert result.reason == "structured_pc_exception"
    assert result.error == "RuntimeError: direct-tail structured cache requested without a direct-tail matrix"


def test_direct_tail_support_mode_preflight_reports_not_applicable() -> None:
    result = run_direct_tail_support_mode_preflight(
        DirectTailSupportModePreflightContext(
            env={"SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_SUPPORT_MODE_PREFLIGHT": "1"},
            factor_kind="other",
            structured_pc_ready=True,
            operator_bundle=SimpleNamespace(matrix=scipy_sparse.eye(2, format="csr")),
            layout=_FakeLayout(),
            active_indices=None,
            max_nbytes=1024,
            regularization=1.0e-12,
            rhs=np.ones(2),
            true_matvec=lambda v: np.asarray(v),
            preconditioner_x=0,
            preconditioner_xi=0,
            preconditioner_species=0,
            preconditioner_x_min_l=0,
            selector=lambda **_kwargs: pytest.fail("unexpected selector"),
            factor_bundle=_fake_factor_bundle,
        )
    )

    assert result.requested is True
    assert result.applicable is False
    assert result.selected is False
    assert result.metadata == {
        "selected": False,
        "reason": "support_mode_preflight_not_applicable",
        "structured_pc_ready": True,
        "factor_kind": "other",
    }


def test_direct_tail_support_mode_preflight_selects_factor_bundle() -> None:
    calls: dict[str, object] = {}
    bundle = SimpleNamespace(matrix=scipy_sparse.eye(2, format="csr"))

    def selector(**kwargs):
        calls.update(kwargs)
        return (
            _FakeStructuredPreconditioner(
                kind="active_fortran_v3_reduced_lu",
                reason="support_mode",
                setup_s=0.75,
                metadata={"factor_nbytes_actual": 55},
            ),
            {
                "selected_candidate": "xmin_l2",
                "baseline_residual_after": 2.0,
                "best_residual_after": 1.0,
            },
        )

    result = run_direct_tail_support_mode_preflight(
        DirectTailSupportModePreflightContext(
            env={
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_SUPPORT_MODE_PREFLIGHT": "1",
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_SUPPORT_MODE_CANDIDATES": "current,xmin_l2",
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_SUPPORT_MODE_MAX_CANDIDATES": "2",
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_SUPPORT_MODE_MIN_IMPROVEMENT": "1.25",
            },
            factor_kind="active-fortran-v3-reduced-lu",
            structured_pc_ready=True,
            operator_bundle=bundle,
            layout=_FakeLayout(),
            active_indices=np.array([0, 1], dtype=np.int64),
            max_nbytes=2048,
            regularization=1.0e-9,
            rhs=np.ones(2),
            true_matvec=lambda v: np.asarray(v),
            preconditioner_x=1,
            preconditioner_xi=2,
            preconditioner_species=3,
            preconditioner_x_min_l=4,
            selector=selector,
            factor_bundle=_fake_factor_bundle,
        )
    )

    assert result.applicable is True
    assert result.selected is True
    assert result.factor_bundle.operator is bundle
    assert result.factor_bundle.factor_nbytes_estimate == 55
    assert result.metadata["selected_candidate"] == "xmin_l2"
    assert calls["requested_kind"] == "active_fortran_v3_reduced_lu"
    assert calls["candidates"] == "current,xmin_l2"
    assert calls["max_candidates"] == 2
    assert calls["min_improvement_ratio"] == 1.25


def test_direct_tail_support_mode_preflight_reports_selector_exception() -> None:
    def selector(**_kwargs):
        raise RuntimeError("selector failed")

    result = run_direct_tail_support_mode_preflight(
        DirectTailSupportModePreflightContext(
            env={"SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_SUPPORT_MODE_PREFLIGHT": "1"},
            factor_kind="active_fortran_v3_reduced_ilu",
            structured_pc_ready=True,
            operator_bundle=SimpleNamespace(matrix=scipy_sparse.eye(2, format="csr")),
            layout=_FakeLayout(),
            active_indices=None,
            max_nbytes=1024,
            regularization=1.0e-12,
            rhs=np.ones(2),
            true_matvec=lambda v: np.asarray(v),
            preconditioner_x=0,
            preconditioner_xi=0,
            preconditioner_species=0,
            preconditioner_x_min_l=0,
            selector=selector,
            factor_bundle=_fake_factor_bundle,
        )
    )

    assert result.applicable is True
    assert result.selected is False
    assert result.metadata is None
    assert result.error == "RuntimeError: selector failed"


def test_sparse_pc_factor_preflight_policy_uses_metadata_trigger() -> None:
    policy = resolve_sparse_pc_factor_preflight_policy(
        SparsePCFactorPreflightPolicyContext(
            env={},
            fortran_reduced_sparse_pc=True,
            structured_pc_ready=True,
            structured_pc_metadata={
                "kind": "",
                "metadata": {
                    "requested_kind": "active-fortran-v3-reduced-ilu",
                    "requires_preflight": True,
                },
            },
            sparse_pc_linear_size=10,
        )
    )

    assert policy.factor_preflight_enabled is True
    assert policy.factor_preflight_required is False
    assert policy.factor_preflight_seed_enabled is True
    assert policy.direct_tail_structured_pc_requires_preflight is True
    assert policy.direct_tail_structured_pc_kind_for_preflight == "active_fortran_v3_reduced_ilu"
    assert policy.direct_tail_structured_pc_size_requires_preflight is False
    assert policy.structured_pc_preflight_required is True
    assert policy.factor_preflight_max_target_ratio == 1.0e6


def test_sparse_pc_factor_preflight_policy_uses_size_trigger_and_overrides() -> None:
    policy = resolve_sparse_pc_factor_preflight_policy(
        SparsePCFactorPreflightPolicyContext(
            env={
                "SFINCS_JAX_RHSMODE1_SPARSE_PC_FACTOR_PREFLIGHT": "0",
                "SFINCS_JAX_RHSMODE1_SPARSE_PC_FACTOR_PREFLIGHT_REQUIRED": "1",
                "SFINCS_JAX_RHSMODE1_SPARSE_PC_FACTOR_PREFLIGHT_SEED": "0",
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_STRUCTURED_PC_PREFLIGHT_REQUIRED_MIN_SIZE": "20",
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_STRUCTURED_PC_PREFLIGHT_REQUIRED": "0",
                "SFINCS_JAX_RHSMODE1_SPARSE_PC_FACTOR_PREFLIGHT_MAX_TARGET_RATIO": "0.5",
            },
            fortran_reduced_sparse_pc=True,
            structured_pc_ready=True,
            structured_pc_metadata={
                "kind": "active-fortran-v3-reduced-ilu",
                "metadata": {},
            },
            sparse_pc_linear_size=25,
        )
    )

    assert policy.factor_preflight_enabled is False
    assert policy.factor_preflight_required is True
    assert policy.factor_preflight_seed_enabled is False
    assert policy.structured_pc_preflight_required_min_size == 20
    assert policy.direct_tail_structured_pc_kind_for_preflight == "active_fortran_v3_reduced_ilu"
    assert policy.direct_tail_structured_pc_size_requires_preflight is True
    assert policy.structured_pc_preflight_required is False
    assert policy.factor_preflight_max_target_ratio == 1.0


def test_sparse_pc_factor_preflight_evaluation_passes_and_seeds() -> None:
    diagnostics_calls: list[dict[str, object]] = []

    def diagnostics(**kwargs):
        diagnostics_calls.append(kwargs)
        return {"selected": True}

    result = evaluate_sparse_pc_factor_preflight(
        SparsePCFactorPreflightEvaluationContext(
            rhs=jnp.asarray([2.0]),
            rhs_norm=2.0,
            target=0.1,
            preconditioner=lambda _rhs: jnp.asarray([1.0]),
            matvec=lambda _x: jnp.asarray([1.5]),
            diagnostics=diagnostics,
            layout=_FakeLayout(),
            active_indices=np.array([0], dtype=np.int64),
            seed_enabled=True,
            max_target_ratio=10.0,
        )
    )

    assert result.residual_before == 2.0
    assert result.residual_after == 0.5
    assert result.improvement_ratio == 4.0
    assert result.target_ratio == 5.0
    assert result.passed is True
    assert result.seed_used is True
    np.testing.assert_allclose(np.asarray(result.x0_seed), np.asarray([1.0]))
    np.testing.assert_allclose(np.asarray(result.residual_vec), np.asarray([0.5]))
    assert result.diagnostics == {"selected": True}
    assert diagnostics_calls[0]["layout"].to_dict() == {"total_size": 3, "f_size": 2}
    np.testing.assert_array_equal(diagnostics_calls[0]["active_indices"], np.array([0]))


def test_sparse_pc_factor_preflight_evaluation_rejects_large_target_ratio() -> None:
    result = evaluate_sparse_pc_factor_preflight(
        SparsePCFactorPreflightEvaluationContext(
            rhs=jnp.asarray([2.0]),
            rhs_norm=2.0,
            target=0.1,
            preconditioner=lambda _rhs: jnp.asarray([1.0]),
            matvec=lambda _x: jnp.asarray([1.5]),
            diagnostics=lambda **_kwargs: {},
            layout=_FakeLayout(),
            active_indices=None,
            seed_enabled=False,
            max_target_ratio=2.0,
        )
    )

    assert result.residual_after == 0.5
    assert result.target_ratio == 5.0
    assert result.passed is False
    assert result.seed_used is False
    assert result.x0_seed is None


def test_sparse_pc_residual_candidate_acceptance_strict_passes_and_seeds() -> None:
    result = evaluate_sparse_pc_residual_candidate_acceptance(
        SparsePCResidualCandidateAcceptanceContext(
            candidate_residual_after=1.0,
            current_residual_after=2.0,
            original_residual_before=4.0,
            target=0.5,
            max_target_ratio=3.0,
            seed_enabled=True,
        )
    )

    assert result.finite_candidate is True
    assert result.improves_current_residual is True
    assert result.improves_original_residual is True
    assert result.strict_accept is True
    assert result.base_improvement_accept is False
    assert result.accepted is True
    assert result.base_improvement_override_used is False
    assert result.improvement_ratio == 4.0
    assert result.target_ratio == 2.0
    assert result.passed is True
    assert result.seed_used is True


def test_sparse_pc_residual_candidate_acceptance_current_only_can_select_without_pass() -> None:
    result = evaluate_sparse_pc_residual_candidate_acceptance(
        SparsePCResidualCandidateAcceptanceContext(
            candidate_residual_after=2.5,
            current_residual_after=3.0,
            original_residual_before=2.0,
            target=1.0,
            max_target_ratio=10.0,
            seed_enabled=True,
            require_original_improvement=False,
            current_min_improvement=0.1,
        )
    )

    assert result.improves_current_residual is True
    assert result.improves_original_residual is False
    assert result.strict_accept is True
    assert result.accepted is True
    assert result.improvement_ratio == pytest.approx(0.8)
    assert result.passed is False
    assert result.seed_used is False


def test_sparse_pc_residual_candidate_acceptance_base_improvement_override_sets_passed() -> None:
    result = evaluate_sparse_pc_residual_candidate_acceptance(
        SparsePCResidualCandidateAcceptanceContext(
            candidate_residual_after=2.5,
            current_residual_after=3.0,
            original_residual_before=2.0,
            target=1.0,
            max_target_ratio=1.0,
            seed_enabled=False,
            accept_base_improvement=True,
            base_improvement_requires_original_miss=False,
            base_improvement_sets_passed=True,
        )
    )

    assert result.strict_accept is False
    assert result.base_improvement_accept is True
    assert result.accepted is True
    assert result.base_improvement_override_used is True
    assert result.target_ratio == 2.5
    assert result.passed is True


def test_sparse_pc_residual_candidate_acceptance_rejects_nonfinite_candidate() -> None:
    result = evaluate_sparse_pc_residual_candidate_acceptance(
        SparsePCResidualCandidateAcceptanceContext(
            candidate_residual_after=float("nan"),
            current_residual_after=1.0,
            original_residual_before=2.0,
            target=0.1,
            max_target_ratio=10.0,
            seed_enabled=True,
        )
    )

    assert result.finite_candidate is False
    assert result.improves_current_residual is False
    assert result.accepted is False
    assert result.target_ratio == float("inf")
    assert result.passed is False
    assert result.seed_used is False


def test_sparse_pc_auto_preflight_retry_selection_filters_after_selected_kind() -> None:
    result = select_sparse_pc_auto_preflight_retry_candidates(
        SparsePCAutoPreflightRetrySelectionContext(
            metadata={
                "auto_selected_kind": "active-spilu",
                "auto_candidates": [
                    "active-spilu",
                    "active-ilu",
                    "active-fortran-v3-reduced-lu",
                    "structured",
                    "jacobi",
                ],
                "auto_rejected_candidates": [{"kind": "active_ilu"}],
            },
            current_kind="fallback",
            sparse_pc_linear_size=25,
            preflight_required_min_size=20,
            skip_large_kinds_raw="jacobi,diagonal",
            max_candidates=3,
        )
    )

    assert result.selected_kind == "active_spilu"
    assert result.auto_candidates == (
        "active_spilu",
        "active_ilu",
        "active_fortran_v3_reduced_lu",
        "structured",
        "jacobi",
    )
    assert result.rejected_kinds == frozenset({"active_ilu"})
    assert result.retry_candidates == ("active_fortran_v3_reduced_lu",)


def test_sparse_pc_auto_preflight_retry_selection_uses_current_kind_when_metadata_missing() -> None:
    result = select_sparse_pc_auto_preflight_retry_candidates(
        SparsePCAutoPreflightRetrySelectionContext(
            metadata={
                "auto_candidates": ["active_global_sparse_lu", "active_fortran_v3_reduced_lu"],
            },
            current_kind="active-global-sparse-lu",
            sparse_pc_linear_size=3,
            preflight_required_min_size=20,
            skip_large_kinds_raw="",
            max_candidates=1,
        )
    )

    assert result.selected_kind == "active_global_sparse_lu"
    assert result.retry_candidates == ("active_fortran_v3_reduced_lu",)


def test_sparse_pc_auto_preflight_retry_evaluation_required_candidate_must_pass_gate() -> None:
    result = evaluate_sparse_pc_auto_preflight_retry(
        SparsePCAutoPreflightRetryEvaluationContext(
            residual_after=0.5,
            target=0.25,
            max_target_ratio=3.0,
            residual_before=2.0,
            sparse_pc_linear_size=30,
            preflight_required_min_size=20,
            retry_kind="active-global-sparse-lu",
            retry_metadata={"requires_preflight": False},
        )
    )

    assert result.target_ratio == 2.0
    assert result.requires_metadata is False
    assert result.requires_size is True
    assert result.required is True
    assert result.preflight_passed is True
    assert result.policy_passed is True


def test_sparse_pc_auto_preflight_retry_evaluation_lu_can_pass_policy_without_required_preflight() -> None:
    result = evaluate_sparse_pc_auto_preflight_retry(
        SparsePCAutoPreflightRetryEvaluationContext(
            residual_after=10.0,
            target=1.0,
            max_target_ratio=1.0,
            residual_before=1.0,
            sparse_pc_linear_size=30,
            preflight_required_min_size=20,
            retry_kind="active-fortran-v3-reduced-lu",
            retry_metadata={},
        )
    )

    assert result.target_ratio == 10.0
    assert result.requires_metadata is False
    assert result.requires_size is False
    assert result.required is False
    assert result.preflight_passed is False
    assert result.policy_passed is True


def test_direct_tail_residual_rescue_policy_defaults() -> None:
    policy = resolve_direct_tail_residual_rescue_policy({})

    assert isinstance(policy, DirectTailResidualRescuePolicy)
    assert policy.residual_coarse_requested is False
    assert policy.residual_coarse_rank == 4
    assert policy.residual_coarse_max_mb == 512.0
    assert policy.residual_window_requested is False
    assert policy.residual_window_max_windows == 2
    assert policy.residual_window_coefficient_mode == "additive"
    assert policy.residual_window_combine_mode == "independent"
    assert policy.true_window_requested is False
    assert policy.true_window_max_windows == 1
    assert policy.true_window_column_batch == 4
    assert policy.true_window_include_tail is True
    assert policy.true_coupled_coarse_explicit_requested is False
    assert policy.true_coupled_coarse_auto_enabled is True
    assert policy.true_coupled_coarse_auto_native_enabled is False
    assert policy.true_coupled_coarse_auto_target_ratio == 10.0
    assert policy.true_coupled_coarse_auto_min_size == 300000


def test_direct_tail_residual_rescue_policy_normalizes_modes_and_clamps() -> None:
    policy = resolve_direct_tail_residual_rescue_policy(
        {
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_RESIDUAL_COARSE": "1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_RESIDUAL_COARSE_RANK": "0",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_RESIDUAL_COARSE_MAX_MB": "-5",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_RESIDUAL_WINDOW": "1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_RESIDUAL_WINDOW_MAX_WINDOWS": "0",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_RESIDUAL_WINDOW_X_RADIUS": "-1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_RESIDUAL_WINDOW_ELL_RADIUS": "-1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_RESIDUAL_WINDOW_COEFFICIENTS": "NORMAL-EQUATIONS",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_RESIDUAL_WINDOW_COMBINE": "graph-interface",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_WINDOW": "1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_WINDOW_DROP_TOL": "-1e-4",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_WINDOW_INCLUDE_TAIL": "0",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_WINDOW_DAMPING": "1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_WINDOW_BETA_MAX": "-2",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_COARSE": "1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_COARSE_AUTO": "0",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_AUTO_NATIVE": "1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_AUTO_TARGET_RATIO": "0",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_AUTO_MIN_SIZE": "0",
        }
    )

    assert policy.residual_coarse_requested is True
    assert policy.residual_coarse_rank == 1
    assert policy.residual_coarse_max_mb == 0.0
    assert policy.residual_window_requested is True
    assert policy.residual_window_max_windows == 1
    assert policy.residual_window_x_radius == 0
    assert policy.residual_window_ell_radius == 0
    assert policy.residual_window_coefficient_mode == "normal_equations"
    assert policy.residual_window_combine_mode == "graph_interface"
    assert policy.true_window_requested is True
    assert policy.true_window_drop_tol == 0.0
    assert policy.true_window_include_tail is False
    assert policy.true_window_damping is True
    assert policy.true_window_beta_max == 0.0
    assert policy.true_coupled_coarse_explicit_requested is True
    assert policy.true_coupled_coarse_auto_enabled is False
    assert policy.true_coupled_coarse_auto_native_enabled is True
    assert policy.true_coupled_coarse_auto_target_ratio == 1.0
    assert policy.true_coupled_coarse_auto_min_size == 1


def test_direct_tail_residual_rescue_policy_falls_back_for_bad_modes() -> None:
    policy = resolve_direct_tail_residual_rescue_policy(
        {
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_RESIDUAL_WINDOW_COEFFICIENTS": "bad",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_RESIDUAL_WINDOW_COMBINE": "bad",
        }
    )

    assert policy.residual_window_coefficient_mode == "additive"
    assert policy.residual_window_combine_mode == "independent"


def test_direct_tail_true_active_rescue_policy_defaults_and_inheritance() -> None:
    policy = resolve_direct_tail_true_active_rescue_policy({})

    assert isinstance(policy, DirectTailTrueActiveRescuePolicy)
    assert policy.active_block_requested is False
    assert policy.active_residual_block_requested is False
    assert policy.active_submatrix_requested is False
    assert policy.active_column_cache_requested is True
    assert policy.active_column_cache_max_mb == 512.0
    assert policy.active_block_x_count == 1
    assert policy.active_block_ell_count == 8
    assert policy.active_block_species_count is None
    assert policy.active_block_theta_stride == 1
    assert policy.active_block_zeta_stride == 1
    assert policy.active_block_max_mb == 1024.0
    assert policy.active_block_regularization == 1.0e-12
    assert policy.active_block_max_size == 4096
    assert policy.active_block_column_batch == 8
    assert policy.active_block_drop_tol == 1.0e-14
    assert policy.active_block_include_tail is True
    assert policy.active_block_max_tail == 512
    assert policy.active_block_damping is False
    assert policy.active_block_beta_max == 10.0
    assert policy.active_residual_block_max_mb == policy.active_block_max_mb
    assert policy.active_residual_block_regularization == policy.active_block_regularization
    assert policy.active_residual_block_max_size == policy.active_block_max_size
    assert policy.active_residual_block_column_batch == policy.active_block_column_batch
    assert policy.active_residual_block_drop_tol == policy.active_block_drop_tol
    assert policy.active_residual_block_include_tail == policy.active_block_include_tail
    assert policy.active_residual_block_max_tail == policy.active_block_max_tail
    assert policy.active_residual_block_damping == policy.active_block_damping
    assert policy.active_residual_block_beta_max == policy.active_block_beta_max
    assert policy.active_residual_block_kinetic_only is True
    assert policy.active_residual_block_min_improvement == 1.0e-6
    assert policy.active_residual_block_accept_base_improvement is False
    assert policy.active_submatrix_damping is True
    assert policy.active_submatrix_alpha_clip == 10.0
    assert policy.active_submatrix_min_improvement == 1.0e-6


def test_direct_tail_true_active_rescue_policy_clamps_and_overrides() -> None:
    policy = resolve_direct_tail_true_active_rescue_policy(
        {
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_BLOCK": "1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_RESIDUAL_BLOCK": "1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_SUBMATRIX": "1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_COLUMN_CACHE": "0",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_COLUMN_CACHE_MAX_MB": "-1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_BLOCK_X_COUNT": "-1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_BLOCK_ELL_COUNT": "-1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_BLOCK_SPECIES_COUNT": "3",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_BLOCK_THETA_STRIDE": "0",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_BLOCK_ZETA_STRIDE": "0",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_BLOCK_MAX_MB": "-5",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_BLOCK_REGULARIZATION": "-1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_BLOCK_MAX_SIZE": "0",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_BLOCK_COLUMN_BATCH": "0",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_BLOCK_DROP_TOL": "-1e-5",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_BLOCK_INCLUDE_TAIL": "0",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_BLOCK_MAX_TAIL": "-1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_BLOCK_DAMPING": "1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_BLOCK_BETA_MAX": "-10",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_RESIDUAL_BLOCK_MAX_MB": "9",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_RESIDUAL_BLOCK_REGULARIZATION": "2e-8",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_RESIDUAL_BLOCK_MAX_SIZE": "11",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_RESIDUAL_BLOCK_COLUMN_BATCH": "12",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_RESIDUAL_BLOCK_DROP_TOL": "3e-4",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_RESIDUAL_BLOCK_INCLUDE_TAIL": "1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_RESIDUAL_BLOCK_MAX_TAIL": "13",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_RESIDUAL_BLOCK_KINETIC_ONLY": "0",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_RESIDUAL_BLOCK_DAMPING": "0",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_RESIDUAL_BLOCK_BETA_MAX": "14",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_RESIDUAL_BLOCK_MIN_IMPROVEMENT": "-1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_RESIDUAL_BLOCK_ACCEPT_BASE_IMPROVEMENT": "1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_SUBMATRIX_DAMPING": "0",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_SUBMATRIX_ALPHA_CLIP": "-1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_SUBMATRIX_MIN_IMPROVEMENT": "-1",
        }
    )

    assert policy.active_block_requested is True
    assert policy.active_residual_block_requested is True
    assert policy.active_submatrix_requested is True
    assert policy.active_column_cache_requested is False
    assert policy.active_column_cache_max_mb == 0.0
    assert policy.active_block_x_count == 0
    assert policy.active_block_ell_count == 0
    assert policy.active_block_species_count == 3
    assert policy.active_block_theta_stride == 1
    assert policy.active_block_zeta_stride == 1
    assert policy.active_block_max_mb == 0.0
    assert policy.active_block_regularization == 0.0
    assert policy.active_block_max_size == 1
    assert policy.active_block_column_batch == 1
    assert policy.active_block_drop_tol == 0.0
    assert policy.active_block_include_tail is False
    assert policy.active_block_max_tail == 0
    assert policy.active_block_damping is True
    assert policy.active_block_beta_max == 0.0
    assert policy.active_residual_block_max_mb == 9.0
    assert policy.active_residual_block_regularization == 2.0e-8
    assert policy.active_residual_block_max_size == 11
    assert policy.active_residual_block_column_batch == 12
    assert policy.active_residual_block_drop_tol == 3.0e-4
    assert policy.active_residual_block_include_tail is True
    assert policy.active_residual_block_max_tail == 13
    assert policy.active_residual_block_kinetic_only is False
    assert policy.active_residual_block_damping is False
    assert policy.active_residual_block_beta_max == 14.0
    assert policy.active_residual_block_min_improvement == 0.0
    assert policy.active_residual_block_accept_base_improvement is True
    assert policy.active_submatrix_damping is False
    assert policy.active_submatrix_alpha_clip == 0.0
    assert policy.active_submatrix_min_improvement == 0.0


def test_direct_tail_true_active_rescue_policy_bad_species_count_is_none() -> None:
    policy = resolve_direct_tail_true_active_rescue_policy(
        {
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_BLOCK_SPECIES_COUNT": "bad",
        }
    )

    assert policy.active_block_species_count is None


def test_direct_tail_coupled_coarse_rescue_policy_defaults() -> None:
    policy = resolve_direct_tail_coupled_coarse_rescue_policy({})

    assert isinstance(policy, DirectTailCoupledCoarseRescuePolicy)
    assert policy.max_windows == 2
    assert policy.x_radius == 0
    assert policy.ell_radius == 1
    assert policy.max_mb == 512.0
    assert policy.regularization == 1.0e-12
    assert policy.max_size == 64
    assert policy.column_batch == 4
    assert policy.drop_tol == 1.0e-14
    assert policy.low_lmax == 3
    assert policy.profile_moment_count == 4
    assert policy.angular_lmax == 2
    assert policy.angular_mode_max == 1
    assert policy.max_tail_units == 16
    assert policy.include_tail is True
    assert policy.include_constraint_sources is True
    assert policy.include_fsavg is True
    assert policy.include_window_residual is True
    assert policy.include_profile_moments is True
    assert policy.include_angular_residual is True
    assert policy.include_angular_basis is False
    assert policy.include_preconditioned_loads is False
    assert policy.preconditioned_load_max_columns == 16
    assert policy.preconditioned_load_max_nnz == 50000
    assert policy.preconditioned_load_drop_tol == 1.0e-12
    assert policy.damping is False
    assert policy.beta_max == 10.0
    assert policy.accept_base_improvement is False


def test_direct_tail_coupled_coarse_rescue_policy_clamps_and_overrides() -> None:
    policy = resolve_direct_tail_coupled_coarse_rescue_policy(
        {
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_MAX_WINDOWS": "0",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_X_RADIUS": "-1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_ELL_RADIUS": "-1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_MAX_MB": "-1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_REGULARIZATION": "-1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_MAX_SIZE": "0",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_COLUMN_BATCH": "0",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_DROP_TOL": "-1e-3",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_LOW_LMAX": "-1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_PROFILE_MOMENT_COUNT": "-1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_ANGULAR_LMAX": "-1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_ANGULAR_MODE_MAX": "-1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_MAX_TAIL_UNITS": "-1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_INCLUDE_TAIL": "0",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_INCLUDE_CONSTRAINT_SOURCES": "0",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_INCLUDE_FSAVG": "0",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_INCLUDE_WINDOW_RESIDUAL": "0",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_INCLUDE_PROFILE_MOMENTS": "0",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_INCLUDE_ANGULAR_RESIDUAL": "0",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_INCLUDE_ANGULAR_BASIS": "1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_INCLUDE_PRECONDITIONED_LOADS": "1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_PRECONDITIONED_LOAD_MAX_COLUMNS": "-1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_PRECONDITIONED_LOAD_MAX_NNZ": "-1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_PRECONDITIONED_LOAD_DROP_TOL": "-1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_DAMPING": "1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_BETA_MAX": "-1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_ACCEPT_BASE_IMPROVEMENT": "1",
        }
    )

    assert policy.max_windows == 1
    assert policy.x_radius == 0
    assert policy.ell_radius == 0
    assert policy.max_mb == 0.0
    assert policy.regularization == 0.0
    assert policy.max_size == 1
    assert policy.column_batch == 1
    assert policy.drop_tol == 0.0
    assert policy.low_lmax == 0
    assert policy.profile_moment_count == 0
    assert policy.angular_lmax == 0
    assert policy.angular_mode_max == 0
    assert policy.max_tail_units == 0
    assert policy.include_tail is False
    assert policy.include_constraint_sources is False
    assert policy.include_fsavg is False
    assert policy.include_window_residual is False
    assert policy.include_profile_moments is False
    assert policy.include_angular_residual is False
    assert policy.include_angular_basis is True
    assert policy.include_preconditioned_loads is True
    assert policy.preconditioned_load_max_columns == 0
    assert policy.preconditioned_load_max_nnz == 0
    assert policy.preconditioned_load_drop_tol == 0.0
    assert policy.damping is True
    assert policy.beta_max == 0.0
    assert policy.accept_base_improvement is True


def test_fortran_reduced_xblock_factor_policy_uses_specific_env_before_generic() -> None:
    setup = resolve_fortran_reduced_xblock_factor_policy(
        env={
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_DROP_TOL": "9.0",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_PC_DROP_TOL": "1.5",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_PC_DROP_REL": "2.5e-7",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_PC_ILU_DROP_TOL": "bad",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_FILL_FACTOR": "4.0",
        },
        preconditioner_xi=2,
    )

    assert setup.drop_tol == 1.5
    assert setup.drop_rel == 2.5e-7
    assert setup.ilu_drop_tol == 1.0e-4
    assert setup.fill_factor == 4.0
    assert setup.preconditioner_xi == 2
    assert setup.promote_xi
    assert setup.messages == ()


def test_fortran_reduced_xblock_factor_policy_promotes_zero_xi_by_default() -> None:
    setup = resolve_fortran_reduced_xblock_factor_policy(
        env={},
        preconditioner_xi=0,
    )

    assert setup.preconditioner_xi == 1
    assert setup.promote_xi
    assert setup.messages == (
        (
            1,
            "solve_v3_full_system_linear_gmres: fortran_reduced_pc_gmres "
            "promoting x-block backend preconditioner_xi 0 -> 1 for stronger FP block factors",
        ),
    )


def test_fortran_reduced_xblock_factor_policy_can_disable_xi_promotion() -> None:
    setup = resolve_fortran_reduced_xblock_factor_policy(
        env={"SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_PROMOTE_XI": "off"},
        preconditioner_xi=0,
    )

    assert setup.preconditioner_xi == 0
    assert not setup.promote_xi
    assert setup.messages == ()


def test_fortran_reduced_xblock_factor_stage_builds_with_policy_and_timing() -> None:
    messages: list[str] = []
    times = iter([2.0, 2.75])
    calls: dict[str, object] = {}

    def assembled_allowed(**kwargs) -> bool:
        calls["assembled"] = kwargs
        return True

    def builder(**kwargs):
        calls["builder"] = kwargs
        return lambda v: 3.0 * v

    result = build_fortran_reduced_xblock_factor_stage(
        context=FortranReducedXBlockFactorBuildContext(
            op_pc=SimpleNamespace(),
            reduce_full=_identity,
            expand_reduced=_identity,
            preconditioner_species=1,
            preconditioner_xi=0,
            sparse_pc_linear_size=42,
            backend_reason="auto_large_full_fp",
            elapsed_s=lambda: next(times),
            emit=lambda _level, msg: messages.append(msg),
            env={
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_PC_DROP_REL": "2e-7",
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_PC_FILL_FACTOR": "6",
            },
            assembled_host_allowed=assembled_allowed,
            builder=builder,
        )
    )

    assert result.preconditioner_xi == 1
    assert result.drop_rel == pytest.approx(2.0e-7)
    assert result.fill_factor == pytest.approx(6.0)
    assert result.force_assembled_host_fp is True
    assert result.factor_s == pytest.approx(0.75)
    assert result.preconditioner(jnp.asarray([2.0])).tolist() == [6.0]
    assert calls["assembled"]["preconditioner_xi"] == 1
    assert calls["builder"]["preconditioner_species"] == 1
    assert calls["builder"]["force_assembled_host_fp"] is True
    assert any("promoting x-block backend preconditioner_xi 0 -> 1" in message for message in messages)
    assert any("using x-block backend instead of monolithic CSR factor" in message for message in messages)


def test_fortran_reduced_xblock_krylov_policy_defaults_and_counter() -> None:
    setup = resolve_fortran_reduced_xblock_krylov_policy(env={})

    assert setup.side_env == ""
    assert setup.precondition_side == "left"
    assert setup.pc_form == "scipy_left"
    assert setup.krylov_method == "gmres"
    assert setup.progress_every == 25
    assert int(setup.mv_count) == 0
    setup.mv_count.increment()
    assert int(setup.mv_count) == 1
    assert setup.messages == ()


def test_fortran_reduced_xblock_krylov_policy_normalizes_aliases() -> None:
    setup = resolve_fortran_reduced_xblock_krylov_policy(
        env={
            "SFINCS_JAX_GMRES_PRECONDITION_SIDE": "right",
            "SFINCS_JAX_RHSMODE1_SPARSE_PC_FORM": "explicit_left",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_KRYLOV": "gcrotmk-scipy",
            "SFINCS_JAX_SPARSE_PC_PROGRESS_EVERY": "7",
        }
    )

    assert setup.precondition_side == "right"
    assert setup.pc_form == "explicit_left"
    assert setup.krylov_method == "gcrotmk"
    assert setup.progress_every == 7
    assert setup.messages == ()


def test_fortran_reduced_xblock_krylov_policy_falls_back_invalid_values() -> None:
    setup = resolve_fortran_reduced_xblock_krylov_policy(
        env={
            "SFINCS_JAX_GMRES_PRECONDITION_SIDE": "bad-side",
            "SFINCS_JAX_RHSMODE1_SPARSE_PC_FORM": "bad-form",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_KRYLOV": "bad-method",
            "SFINCS_JAX_SPARSE_PC_PROGRESS_EVERY": "-5",
        }
    )

    assert setup.side_env == "bad-side"
    assert setup.precondition_side == "left"
    assert setup.pc_form == "scipy_left"
    assert setup.krylov_method == "gmres"
    assert setup.progress_every == 0
    assert setup.messages == (
        (
            1,
            "solve_v3_full_system_linear_gmres: fortran_reduced_pc_gmres xblock "
            "ignoring unknown SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_KRYLOV="
            "'bad_method'; using gmres",
        ),
    )


def test_fortran_reduced_xblock_krylov_setup_builds_active_matvec_and_wrapper() -> None:
    messages: list[str] = []
    setup = build_fortran_reduced_xblock_krylov_setup(
        context=FortranReducedXBlockKrylovSetupContext(
            op=SimpleNamespace(total_size=4),
            rhs=jnp.asarray([1.0, 2.0, 3.0, 4.0]),
            xblock_use_active_dof=True,
            active_idx=jnp.asarray([0, 2], dtype=jnp.int32),
            full_to_active=jnp.asarray([0, -1, 1, -1], dtype=jnp.int32),
            reduce_full_with_indices=lambda v, idx: v[idx],
            expand_reduced_with_map=lambda v, fmap: jnp.where(
                fmap >= 0,
                v[jnp.maximum(fmap, 0)],
                0.0,
            ),
            operator_matvec=lambda v: v + 10.0,
            base_preconditioner=lambda v: 2.0 * v,
            elapsed_s=lambda: 3.5,
            emit=lambda _level, msg: messages.append(msg),
            env={
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_KRYLOV": "bad-method",
                "SFINCS_JAX_SPARSE_PC_PROGRESS_EVERY": "2",
            },
        )
    )

    assert setup.precondition_side == "left"
    assert setup.pc_form == "scipy_left"
    assert setup.krylov_method == "gmres"
    assert setup.progress_every == 2
    assert setup.matvec_no_count(jnp.asarray([5.0, 7.0])).tolist() == [15.0, 17.0]
    assert setup.preconditioner(jnp.asarray([1.0, 3.0])).tolist() == [2.0, 6.0]
    assert int(setup.mv_count) == 0
    setup.matvec(jnp.asarray([0.0, 1.0]))
    setup.matvec(jnp.asarray([2.0, 3.0]))
    assert int(setup.mv_count) == 2
    assert any("using gmres" in message for message in messages)
    assert any("fortran_reduced_pc_gmres xblock matvecs=2" in message for message in messages)
    assert not any("active-DOF reduction" in message for message in messages)


def test_fortran_reduced_xblock_initial_seed_policy_parses_controls() -> None:
    setup = resolve_fortran_reduced_xblock_initial_seed_policy(
        env={
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_INITIAL_SEED": "off",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_SEED_REFINES": "bad",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_SEED_ACCEPT_RATIO": "-3",
        }
    )

    assert not setup.enabled
    assert setup.refine_steps == 2
    assert setup.accept_ratio == 0.0


def test_fortran_reduced_xblock_initial_seed_accepts_refined_seed() -> None:
    policy = resolve_fortran_reduced_xblock_initial_seed_policy(
        env={
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_SEED_REFINES": "2",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_SEED_ACCEPT_RATIO": "0.2",
        }
    )
    times = iter([5.0, 5.75])
    result = apply_fortran_reduced_xblock_initial_seed(
        policy=policy,
        rhs=jnp.asarray([4.0, 0.0]),
        rhs_norm=4.0,
        x0=None,
        preconditioner=lambda v: 0.25 * v,
        matvec_no_count=lambda v: 2.0 * v,
        elapsed_s=lambda: next(times),
    )

    assert result.used
    assert result.refines_performed == 2
    assert result.residual_norm == pytest.approx(0.5)
    assert result.improvement_ratio == pytest.approx(8.0)
    assert result.elapsed_s == pytest.approx(0.75)
    assert result.x0 is not None
    assert result.x0.tolist() == pytest.approx([1.75, 0.0])
    assert any("accepted=True" in message for _, message in result.messages)


def test_fortran_reduced_xblock_initial_seed_rejects_weak_seed() -> None:
    policy = resolve_fortran_reduced_xblock_initial_seed_policy(
        env={
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_SEED_REFINES": "3",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_SEED_ACCEPT_RATIO": "0.5",
        }
    )
    result = apply_fortran_reduced_xblock_initial_seed(
        policy=policy,
        rhs=jnp.asarray([2.0, 0.0]),
        rhs_norm=2.0,
        x0=None,
        preconditioner=lambda v: jnp.zeros_like(v),
        matvec_no_count=lambda v: v,
        elapsed_s=lambda: 0.0,
    )

    assert not result.used
    assert result.x0 is None
    assert result.refines_performed == 0
    assert result.residual_norm == pytest.approx(2.0)
    assert result.improvement_ratio == pytest.approx(1.0)
    assert any("accepted=False" in message for _, message in result.messages)


def test_fortran_reduced_xblock_krylov_solve_runs_gmres_and_true_residual() -> None:
    messages: list[str] = []
    counter = MatvecCounter(0)
    times = iter([1.0, 1.5, 2.0, 2.25])

    def matvec(v: jnp.ndarray) -> jnp.ndarray:
        counter.increment()
        return v

    def gmres_solver(**kwargs):
        assert kwargs["preconditioner"] is None
        kwargs["progress_callback"](2, 0.25)
        return np.asarray([1.0, 2.0]), 99.0, [0.5, 0.25]

    def unused_solver(**_kwargs):
        raise AssertionError("unused")

    result = run_fortran_reduced_xblock_krylov_solve(
        context=FortranReducedXBlockKrylovSolveContext(
            matvec=matvec,
            rhs=jnp.asarray([1.0, 2.0]),
            preconditioner=_identity,
            emit=lambda _level, msg: messages.append(msg),
            elapsed_s=lambda: next(times),
            method="gmres",
            pc_form="scipy_left",
            restart=8,
            maxiter=3,
            tol=1.0e-8,
            atol=1.0e-10,
            target=1.0e-9,
            precondition_side="none",
            progress_every=2,
            mv_count=counter,
            explicit_left_solver=unused_solver,
            gmres_solver=gmres_solver,
            lgmres_solver=unused_solver,
            gcrotmk_solver=unused_solver,
            bicgstab_solver=unused_solver,
        ),
        x0=None,
    )

    assert result.x.tolist() == [1.0, 2.0]
    assert result.residual_norm == pytest.approx(0.0)
    assert result.history == (0.5, 0.25)
    assert result.solve_s == pytest.approx(1.0)
    assert int(counter) == 1
    assert any("iters=2 ksp_residual=2.500000e-01" in message for message in messages)
    assert any("matvecs=1 residual=0.000000e+00" in message for message in messages)


def test_fortran_reduced_xblock_krylov_solve_explicit_left_reports_pc_residual() -> None:
    messages: list[str] = []
    counter = MatvecCounter(0)

    def matvec(v: jnp.ndarray) -> jnp.ndarray:
        counter.increment()
        return v

    def explicit_left_solver(**kwargs):
        assert kwargs["preconditioner"] is _identity
        kwargs["progress_callback"](1, 0.125)
        return np.asarray([3.0]), 4.0, 0.125, [0.125]

    def unused_solver(**_kwargs):
        raise AssertionError("unused")

    result = run_fortran_reduced_xblock_krylov_solve(
        context=FortranReducedXBlockKrylovSolveContext(
            matvec=matvec,
            rhs=jnp.asarray([3.0]),
            preconditioner=_identity,
            emit=lambda _level, msg: messages.append(msg),
            elapsed_s=lambda: 0.0,
            method="gmres",
            pc_form="explicit_left",
            restart=4,
            maxiter=2,
            tol=1.0e-8,
            atol=1.0e-10,
            target=1.0e-9,
            precondition_side="left",
            progress_every=1,
            mv_count=counter,
            explicit_left_solver=explicit_left_solver,
            gmres_solver=unused_solver,
            lgmres_solver=unused_solver,
            gcrotmk_solver=unused_solver,
            bicgstab_solver=unused_solver,
        ),
        x0=None,
    )

    assert result.residual_norm == pytest.approx(0.0)
    assert result.preconditioned_residual_norm == pytest.approx(0.125)
    assert any("preconditioned_residual=1.250000e-01" in message for message in messages)


def test_fortran_reduced_xblock_result_metadata_formats_branch_payload() -> None:
    state = {
        "op": SimpleNamespace(total_size=12),
        "fortran_reduced_xblock_accepted_converged": True,
        "history": (0.5, 0.25),
        "mv_count": MatvecCounter(7),
        "pc_restart": 8,
        "pc_maxiter": 3,
        "fortran_reduced_sparse_pc_backend_reason": "auto_large_full_fp",
        "fortran_reduced_xblock_min_size": 100,
        "preconditioner_x": 4,
        "preconditioner_x_min_l": 2,
        "preconditioner_xi": 1,
        "preconditioner_species": 0,
        "xblock_preconditioner_xi": 1,
        "force_assembled_host_fp": True,
        "xblock_krylov_method": "gmres",
        "seed_enabled": True,
        "seed_used": True,
        "seed_residual_norm": 1.0e-4,
        "seed_improvement_ratio": 10.0,
        "seed_accept_ratio": 1.0,
        "seed_refine_steps": 2,
        "seed_refines_performed": 1,
        "moment_schur_enabled": True,
        "moment_schur_built": True,
        "moment_schur_used": False,
        "moment_schur_reason": "probe_not_reduced",
        "moment_schur_metadata": {
            "mode": "additive",
            "rank": 3,
            "extra_size": 2,
            "setup_s": 0.25,
            "expected_size": 10,
            "rcond": 1.0e-12,
            "singular_value_proxy": (1.0, 0.1),
            "device_resident": False,
        },
        "moment_schur_probe_residual_before": 2.0,
        "moment_schur_probe_residual_after": 1.5,
        "moment_schur_probe_improvement_ratio": 0.75,
        "moment_schur_stats": {"applies": 4, "base_applies": 5},
        "global_coupling_enabled": True,
        "global_coupling_built": True,
        "global_coupling_metadata": {
            "mode": "multiplicative",
            "load_basis_size": 6,
            "basis_size": 5,
            "rank": 4,
            "setup_s": 0.5,
            "setup_budget_s": 1.0,
            "setup_budget_reached": False,
            "rcond": 1.0e-11,
            "smoother": "xblock",
            "basis_names": ("rhs", "fsavg"),
        },
        "global_coupling_stats": {"applies": 8, "coarse_applies": 9},
        "xblock_drop_tol": 0.0,
        "xblock_drop_rel": 1.0e-8,
        "xblock_ilu_drop_tol": 1.0e-4,
        "xblock_fill_factor": 10.0,
        "sparse_pc_use_active_dof": False,
        "sparse_pc_linear_size": 10,
        "sparse_pc_fp_dense_velocity_block": None,
        "setup_s": 0.75,
        "solve_s": 1.25,
        "sparse_timer": SimpleNamespace(elapsed_s=lambda: 2.5),
        "pc_factor_s": 0.5,
        "target": 0.2,
        "residual_norm_sparse_pc": 0.1,
        "fortran_reduced_xblock_factor_quality_rejected": False,
    }

    metadata = fortran_reduced_xblock_result_metadata(state)

    assert metadata["solver_kind"] == "fortran_reduced_pc_gmres"
    assert metadata["accepted_converged"] is True
    assert metadata["iterations"] == 2
    assert metadata["matvecs"] == 7
    assert metadata["sparse_pc_backend"] == "xblock"
    assert metadata["sparse_pc_xblock_initial_seed_used"] is True
    assert metadata["sparse_pc_xblock_moment_schur_rank"] == 3
    assert metadata["sparse_pc_xblock_global_coupling_basis_names"] == ("rhs", "fsavg")
    assert metadata["sparse_pc_residual_ratio_to_target"] == pytest.approx(0.5)
    assert metadata["sparse_pc_factor_quality_rejected"] is False


def test_fortran_reduced_xblock_final_payload_from_driver_state_sets_gates() -> None:
    state = _DefaultSparsePCDriverState(
        {
            "op": SimpleNamespace(total_size=4),
            "atol": 0.25,
            "tol": 0.0,
            "rhs_norm": 1.0,
            "target": 0.5,
            "mv_count": MatvecCounter(4),
            "pc_restart": 8,
            "pc_maxiter": 3,
            "fortran_reduced_sparse_pc_backend_reason": "auto_large_full_fp",
            "fortran_reduced_xblock_min_size": 100,
            "preconditioner_x": 4,
            "preconditioner_x_min_l": 2,
            "preconditioner_xi": 1,
            "preconditioner_species": 0,
            "xblock_preconditioner_xi": 1,
            "force_assembled_host_fp": True,
            "xblock_krylov_method": "gmres",
            "seed_enabled": True,
            "seed_used": False,
            "seed_residual_norm": None,
            "seed_improvement_ratio": None,
            "seed_accept_ratio": 1.0,
            "seed_refine_steps": 2,
            "seed_refines_performed": 0,
            "moment_schur_enabled": True,
            "moment_schur_built": False,
            "moment_schur_used": False,
            "moment_schur_reason": "disabled",
            "moment_schur_metadata": {},
            "moment_schur_probe_residual_before": None,
            "moment_schur_probe_residual_after": None,
            "moment_schur_probe_improvement_ratio": None,
            "moment_schur_stats": {},
            "global_coupling_enabled": True,
            "global_coupling_built": False,
            "global_coupling_metadata": {},
            "global_coupling_stats": {},
            "xblock_drop_tol": 0.0,
            "xblock_drop_rel": 1.0e-8,
            "xblock_ilu_drop_tol": 1.0e-4,
            "xblock_fill_factor": 10.0,
            "sparse_pc_use_active_dof": True,
            "sparse_pc_linear_size": 2,
            "sparse_pc_fp_dense_velocity_block": None,
            "setup_s": 0.75,
            "sparse_timer": SimpleNamespace(elapsed_s=lambda: 3.0),
            "pc_factor_s": 0.5,
        }
    )

    payload = fortran_reduced_xblock_final_payload_from_driver_state(
        state,
        result=SparsePCGMRESResult(
            x=np.asarray([1.0, 2.0]),
            residual_norm=0.2,
            preconditioned_residual_norm=np.nan,
            history=(1.0, 0.4, 0.2),
            solve_s=1.25,
        ),
        expand_reduced=lambda x: jnp.concatenate(
            [jnp.asarray([0.0], dtype=x.dtype), x]
        ),
    )

    assert isinstance(payload, SparsePCGMRESFinalPayload)
    np.testing.assert_allclose(np.asarray(payload.x), np.asarray([0.0, 1.0, 2.0]))
    assert float(payload.residual_norm) == pytest.approx(0.2)
    assert payload.metadata["solver_kind"] == "fortran_reduced_pc_gmres"
    assert payload.metadata["accepted_converged"] is True
    assert payload.metadata["iterations"] == 3
    assert payload.metadata["sparse_pc_factor_quality_rejected"] is False
    assert payload.metadata["sparse_pc_residual_ratio_to_target"] == pytest.approx(0.4)


def test_fortran_reduced_xblock_moment_schur_stage_accepts_probe() -> None:
    policy = resolve_fortran_reduced_xblock_moment_schur_policy(
        precondition_side="left",
        env={
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_MOMENT_SCHUR": "1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_MOMENT_SCHUR_PROBE": "1",
        },
    )
    messages: list[str] = []
    stats = {"applies": 2, "base_applies": 3}

    def builder(**_kwargs):
        return (lambda v: v), {"rank": 1}, stats

    result = apply_fortran_reduced_xblock_moment_schur_stage(
        context=FortranReducedXBlockMomentSchurStageContext(
            op=SimpleNamespace(),
            base_preconditioner=lambda v: jnp.zeros_like(v),
            reduce_full=None,
            expand_reduced=None,
            policy=policy,
            precondition_side="left",
            rhs=jnp.asarray([1.0]),
            matvec_no_count=lambda v: v,
            elapsed_s=lambda: 2.0,
            emit=lambda _level, msg: messages.append(msg),
            builder=builder,
        )
    )

    assert result.built
    assert result.used
    assert result.reason == "probe_reduced"
    assert result.metadata["setup_s"] == pytest.approx(0.0)
    assert result.probe_residual_after == pytest.approx(0.0)
    assert result.stats is stats
    stats["applies"] = 7
    assert result.stats["applies"] == 7
    assert result.preconditioner(jnp.asarray([4.0])).tolist() == [4.0]
    assert any("constraint1 moment-Schur accepted" in message for message in messages)


def test_fortran_reduced_xblock_moment_schur_stage_rejects_probe() -> None:
    policy = resolve_fortran_reduced_xblock_moment_schur_policy(
        precondition_side="left",
        env={
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_MOMENT_SCHUR": "1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_MOMENT_SCHUR_PROBE": "1",
        },
    )
    def base(v: jnp.ndarray) -> jnp.ndarray:
        return 0.5 * v

    def builder(**_kwargs):
        return (lambda v: jnp.zeros_like(v)), {}, {}

    result = apply_fortran_reduced_xblock_moment_schur_stage(
        context=FortranReducedXBlockMomentSchurStageContext(
            op=SimpleNamespace(),
            base_preconditioner=base,
            reduce_full=None,
            expand_reduced=None,
            policy=policy,
            precondition_side="left",
            rhs=jnp.asarray([1.0]),
            matvec_no_count=lambda v: v,
            elapsed_s=lambda: 0.0,
            emit=None,
            builder=builder,
        )
    )

    assert result.built
    assert not result.used
    assert result.reason == "probe_not_reduced"
    assert result.preconditioner is base
    assert result.probe_improvement_ratio == pytest.approx(1.0)


def test_fortran_reduced_xblock_moment_schur_stage_records_failure() -> None:
    policy = resolve_fortran_reduced_xblock_moment_schur_policy(
        precondition_side="left",
        env={"SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_MOMENT_SCHUR": "1"},
    )

    def builder(**_kwargs):
        raise RuntimeError("boom")

    result = apply_fortran_reduced_xblock_moment_schur_stage(
        context=FortranReducedXBlockMomentSchurStageContext(
            op=SimpleNamespace(),
            base_preconditioner=_identity,
            reduce_full=None,
            expand_reduced=None,
            policy=policy,
            precondition_side="left",
            rhs=jnp.asarray([1.0]),
            matvec_no_count=lambda v: v,
            elapsed_s=lambda: 0.0,
            emit=None,
            builder=builder,
        )
    )

    assert not result.built
    assert not result.used
    assert "RuntimeError: boom" in str(result.reason)
    assert result.metadata["error"] == "RuntimeError: boom"
    assert result.preconditioner is _identity


def test_fortran_reduced_xblock_global_coupling_stage_builds_and_records_stats() -> None:
    policy = resolve_fortran_reduced_xblock_global_coupling_policy(
        precondition_side="left",
        env={"SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_GLOBAL_COUPLING": "1"},
    )
    messages: list[str] = []
    stats = {"applies": 4, "coarse_applies": 5}

    def builder(**kwargs):
        assert kwargs["expected_size"] == 3
        assert kwargs["mode"] == "additive"
        return (lambda v: 2.0 * v), {"rank": 2}, stats

    result = apply_fortran_reduced_xblock_global_coupling_stage(
        context=FortranReducedXBlockGlobalCouplingStageContext(
            op=SimpleNamespace(),
            rhs=jnp.asarray([1.0, 2.0, 3.0]),
            matvec=lambda v: v,
            base_preconditioner=_identity,
            direction_projector=None,
            expected_size=3,
            policy=policy,
            elapsed_s=lambda: 1.0,
            emit=lambda _level, msg: messages.append(msg),
            builder=builder,
        )
    )

    assert result.built
    assert result.metadata["rank"] == 2
    assert result.metadata["setup_s"] == pytest.approx(0.0)
    assert result.stats is stats
    stats["applies"] = 6
    assert result.stats["applies"] == 6
    assert result.preconditioner(jnp.asarray([3.0])).tolist() == [6.0]
    assert any("global-coupling build start" in message for message in messages)


def test_fortran_reduced_xblock_global_coupling_stage_records_failure() -> None:
    policy = resolve_fortran_reduced_xblock_global_coupling_policy(
        precondition_side="left",
        env={"SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_GLOBAL_COUPLING": "1"},
    )

    def builder(**_kwargs):
        raise ValueError("bad basis")

    result = apply_fortran_reduced_xblock_global_coupling_stage(
        context=FortranReducedXBlockGlobalCouplingStageContext(
            op=SimpleNamespace(),
            rhs=jnp.asarray([1.0]),
            matvec=lambda v: v,
            base_preconditioner=_identity,
            direction_projector=None,
            expected_size=1,
            policy=policy,
            elapsed_s=lambda: 0.0,
            emit=None,
            builder=builder,
        )
    )

    assert not result.built
    assert result.preconditioner is _identity
    assert result.metadata["error"] == "ValueError: bad basis"
    assert result.stats == {"applies": 0, "coarse_applies": 0}


def test_fortran_reduced_xblock_moment_schur_policy_defaults_disabled() -> None:
    setup = resolve_fortran_reduced_xblock_moment_schur_policy(
        precondition_side="left",
        env={},
    )

    assert not setup.default_candidate
    assert not setup.default_blocked_by_compact_factors
    assert not setup.enabled
    assert setup.rcond == pytest.approx(1.0e-12)
    assert not setup.probe_enabled
    assert setup.probe_min_improvement == 0.0
    assert setup.messages == ()


def test_fortran_reduced_xblock_moment_schur_policy_uses_fortran_env_over_generic() -> (
    None
):
    setup = resolve_fortran_reduced_xblock_moment_schur_policy(
        precondition_side="right",
        env={
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_MOMENT_SCHUR": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_MOMENT_SCHUR_RCOND": "3e-9",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_MOMENT_SCHUR_RCOND": "2e-8",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_MOMENT_SCHUR_PROBE": "1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_MOMENT_SCHUR_MIN_IMPROVEMENT": "0.4",
        },
    )

    assert setup.enabled
    assert setup.rcond == pytest.approx(2.0e-8)
    assert setup.probe_enabled
    assert setup.probe_min_improvement == pytest.approx(0.4)
    assert any("moment-Schur build start" in message for _, message in setup.messages)


def test_fortran_reduced_xblock_moment_schur_policy_falls_back_to_generic_rcond() -> (
    None
):
    setup = resolve_fortran_reduced_xblock_moment_schur_policy(
        precondition_side="none",
        env={
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_MOMENT_SCHUR": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_MOMENT_SCHUR_RCOND": "3e-9",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_MOMENT_SCHUR_RCOND": "bad",
        },
    )

    assert setup.enabled
    assert setup.rcond == pytest.approx(3.0e-9)
    assert setup.messages == ()


def test_fortran_reduced_xblock_global_coupling_policy_defaults_off() -> None:
    setup = resolve_fortran_reduced_xblock_global_coupling_policy(
        precondition_side="right",
        env={},
    )

    assert not setup.enabled
    assert not setup.should_build
    assert not setup.use_device_builder
    assert setup.mode == "additive"
    assert setup.max_directions == 96
    assert setup.fsavg_lmax == 12
    assert setup.angular_lmax == 2
    assert setup.max_extra_units == 8
    assert setup.rcond == pytest.approx(1.0e-11)
    assert setup.include_rhs
    assert setup.setup_max_s == 0.0


def test_fortran_reduced_xblock_global_coupling_policy_parses_controls() -> None:
    setup = resolve_fortran_reduced_xblock_global_coupling_policy(
        precondition_side="left",
        env={
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_GLOBAL_COUPLING": "1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_GLOBAL_COUPLING_MODE": "multiplicative",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_GLOBAL_COUPLING_MAX_DIRECTIONS": "11",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_GLOBAL_COUPLING_FSAVG_LMAX": "4",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_GLOBAL_COUPLING_ANGULAR_LMAX": "5",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_GLOBAL_COUPLING_MAX_EXTRA_UNITS": "6",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_GLOBAL_COUPLING_RCOND": "3e-8",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_GLOBAL_COUPLING_INCLUDE_RHS": "0",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_GLOBAL_COUPLING_SETUP_MAX_S": "9.5",
        },
    )

    assert setup.enabled
    assert setup.should_build
    assert setup.mode == "multiplicative"
    assert setup.max_directions == 11
    assert setup.fsavg_lmax == 4
    assert setup.angular_lmax == 5
    assert setup.max_extra_units == 6
    assert setup.rcond == pytest.approx(3.0e-8)
    assert not setup.include_rhs
    assert setup.setup_max_s == pytest.approx(9.5)


def test_fortran_reduced_xblock_global_coupling_policy_generic_mode_and_no_side() -> (
    None
):
    setup = resolve_fortran_reduced_xblock_global_coupling_policy(
        precondition_side="none",
        env={
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_GLOBAL_COUPLING": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_MODE": "right_additive",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_GLOBAL_COUPLING_MAX_DIRECTIONS": "bad",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_GLOBAL_COUPLING_SETUP_MAX_S": "-1",
        },
    )

    assert setup.enabled
    assert not setup.should_build
    assert setup.mode == "right_additive"
    assert setup.max_directions == 96
    assert setup.setup_max_s == 0.0


def test_sparse_pc_entry_policy_classifies_pas_er_and_active_dof() -> None:
    def parse_config(**kwargs):
        assert kwargs["default_restart"] == 50
        assert kwargs["default_maxiter"] == 100
        return 50, 100

    setup = resolve_sparse_pc_entry_policy(
        op=_op(pas=True, constraint_scheme=2, n_zeta=1, n_species=1),
        solve_method_kind="sparse_pc_gmres",
        has_reduced_modes=True,
        use_active_dof_mode=False,
        xblock_active_dof_requested=False,
        active_maps_available=False,
        use_dkes=True,
        include_xdot_sparse_pc=False,
        include_electric_field_xi_sparse_pc=True,
        er_abs_sparse_pc=0.2,
        restart=50,
        maxiter=80,
        parse_polish_gmres_config=parse_config,
        sparse_pc_default_restart=lambda **kwargs: kwargs["requested_restart"] - 5,
        env={"SFINCS_JAX_RHSMODE1_SPARSE_PC_FP_DENSE_VELOCITY_BLOCK": "1"},
    )

    assert setup.constrained_pas_pc
    assert setup.tokamak_pas_er_pc
    assert not setup.tokamak_pas_noer_pc
    assert setup.sparse_pc_use_active_dof
    assert setup.sparse_pc_fp_dense_velocity_block is True
    assert setup.pc_restart == 45
    assert setup.pc_maxiter == 100


def test_sparse_pc_entry_policy_classifies_xblock_active_maps() -> None:
    setup = resolve_sparse_pc_entry_policy(
        op=_op(fp=True, constraint_scheme=1, n_zeta=3, n_species=2),
        solve_method_kind="xblock_sparse_pc_gmres",
        has_reduced_modes=True,
        use_active_dof_mode=True,
        xblock_active_dof_requested=True,
        active_maps_available=True,
        use_dkes=False,
        include_xdot_sparse_pc=False,
        include_electric_field_xi_sparse_pc=False,
        er_abs_sparse_pc=0.0,
        restart=10,
        maxiter=None,
        parse_polish_gmres_config=lambda **_kwargs: (20, 400),
        sparse_pc_default_restart=lambda **kwargs: kwargs["requested_restart"],
        env={},
    )

    assert setup.xblock_sparse_pc
    assert setup.xblock_use_active_dof
    assert not setup.sparse_pc_use_active_dof
    assert setup.pc_restart == 20
    assert setup.pc_maxiter == 400


def test_xblock_sparse_pc_setup_resolves_host_assembly_and_device_fallback() -> None:
    fallback_calls: list[dict[str, object]] = []

    def fallback_decision(**kwargs):
        fallback_calls.append(kwargs)
        return SimpleNamespace(
            used=True,
            ignored_env=False,
            mode="host",
            reason="forced",
            requested_method=kwargs["requested_krylov_method"],
            effective_krylov_env_value="auto",
            min_active_size=1,
            qi_like_full_fp_3d=False,
            non_autodiff=True,
        )

    setup = resolve_xblock_sparse_pc_setup(
        op=_op(fp=True, constraint_scheme=1, n_zeta=7, n_species=1),
        preconditioner_species=0,
        preconditioner_xi=0,
        active_size=1000,
        lower_fill_mode=lambda value: ("force", value == "bad"),
        species_decoupled_for_host_assembly=lambda **_kwargs: True,
        assembled_host_allowed=lambda **_kwargs: False,
        krylov_method=lambda value: (
            "gmres_jax" if value == "gmres_jax" else "gmres",
            False,
        ),
        device_host_fallback_decision=fallback_decision,
        env={
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_DROP_TOL": "1e-5",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_LOWER_FILL": "force",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV": "gmres_jax",
            "SFINCS_JAX_RHSMODE1_XBLOCK_DEVICE_HOST_FALLBACK": "host",
        },
    )

    assert setup.xblock_drop_tol == pytest.approx(1.0e-5)
    assert setup.xblock_lower_fill_mode == "force"
    assert setup.xblock_preconditioner_xi == 1
    assert setup.force_assembled_host_fp
    assert setup.xblock_assembled_host_fp
    assert setup.xblock_krylov_env_requested == "gmres_jax"
    assert setup.xblock_krylov_env == "auto"
    assert setup.xblock_krylov_requested == "gmres"
    assert not setup.xblock_device_krylov_requested
    assert setup.xblock_device_host_fallback_decision.used
    assert any(
        "non-autodiff host x-block fallback" in message for _, message in setup.messages
    )
    assert fallback_calls[0]["requested_krylov_method"] == "gmres_jax"


def test_xblock_sparse_pc_setup_disables_auto_host_fallback_for_qi_device_request() -> (
    None
):
    def fallback_decision(**kwargs):
        assert kwargs["env_value"] == "off"
        return SimpleNamespace(
            used=False,
            ignored_env=False,
            mode="disabled",
            reason="disabled",
            requested_method=kwargs["requested_krylov_method"],
            effective_krylov_env_value=kwargs["env_value"],
            min_active_size=1,
            qi_like_full_fp_3d=False,
            non_autodiff=False,
        )

    setup = resolve_xblock_sparse_pc_setup(
        op=_op(fp=True, constraint_scheme=1, n_zeta=5, n_species=1),
        preconditioner_species=1,
        preconditioner_xi=1,
        active_size=2000,
        lower_fill_mode=lambda _value: ("off", False),
        species_decoupled_for_host_assembly=lambda **_kwargs: False,
        assembled_host_allowed=lambda **_kwargs: False,
        krylov_method=lambda value: (
            "gmres_jax" if value == "gmres_jax" else "gmres",
            False,
        ),
        device_host_fallback_decision=fallback_decision,
        env={
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV": "gmres_jax",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_USE_IN_KRYLOV": "1",
        },
    )

    assert setup.xblock_device_krylov_requested
    assert setup.xblock_device_host_fallback_auto_disabled_by_qi_device
    assert setup.qi_device_preconditioner_requested_for_fallback
    assert any(
        "fallback disabled by explicit matrix-free" in message
        for _, message in setup.messages
    )


def test_xblock_sparse_pc_side_policy_parses_jax_factors_and_forces_fgmres_right_pc() -> (
    None
):
    def side_policy(**kwargs):
        assert kwargs["krylov_env_value"] == "fgmres_jax"
        assert kwargs["full_fp_3d_pc"] is True
        return SimpleNamespace(
            precondition_side="left",
            default_right_preconditioned=False,
            krylov_method="fgmres_jax",
            gmres_restart=33,
            restart_capped=True,
            ignored_krylov_env=True,
        )

    setup = resolve_xblock_sparse_pc_side_policy_setup(
        op=_op(fp=True, constraint_scheme=1, n_zeta=7, n_species=1),
        xblock_device_krylov_requested=True,
        xblock_device_host_fallback_decision=SimpleNamespace(used=False),
        xblock_krylov_env="fgmres_jax",
        pc_restart=50,
        pc_restart_env="50",
        tokamak_fp_er_pc=False,
        active_size=4000,
        use_dkes=False,
        include_xdot_sparse_pc=False,
        include_electric_field_xi_sparse_pc=False,
        resolve_xblock_policy=side_policy,
        env={
            "SFINCS_JAX_RHSMODE1_XBLOCK_SPARSE_JAX_FACTOR_FORMAT": "compact-csr",
            "SFINCS_JAX_RHSMODE1_XBLOCK_SPARSE_JAX_FACTOR_APPLY": "jacobi",
        },
    )

    assert setup.xblock_jax_factors
    assert not setup.xblock_jax_factors_requested
    assert setup.xblock_device_krylov_forced_jax_factors
    assert setup.xblock_jax_factor_format == "csr"
    assert setup.xblock_jax_factor_apply == "diagonal"
    assert setup.precondition_side == "right"
    assert setup.xblock_device_fgmres_forced_right_pc
    assert setup.pc_restart == 33
    assert setup.xblock_default_restart_capped
    assert any("ignoring unknown" in message for _, message in setup.messages)


def test_xblock_sparse_pc_side_policy_uses_host_factors_when_fallback_is_used() -> None:
    def side_policy(**kwargs):
        return SimpleNamespace(
            precondition_side="right",
            default_right_preconditioned=True,
            krylov_method=kwargs["krylov_env_value"],
            gmres_restart=kwargs["requested_restart"],
            restart_capped=False,
            ignored_krylov_env=False,
        )

    setup = resolve_xblock_sparse_pc_side_policy_setup(
        op=_op(fp=True, constraint_scheme=1, n_zeta=5, n_species=1),
        xblock_device_krylov_requested=True,
        xblock_device_host_fallback_decision=SimpleNamespace(used=True),
        xblock_krylov_env="gmres",
        pc_restart=20,
        pc_restart_env="",
        tokamak_fp_er_pc=False,
        active_size=2000,
        use_dkes=False,
        include_xdot_sparse_pc=False,
        include_electric_field_xi_sparse_pc=False,
        resolve_xblock_policy=side_policy,
        env={"SFINCS_JAX_RHSMODE1_XBLOCK_PC_JAX_FACTORS": "1"},
    )

    assert setup.xblock_jax_factors_requested
    assert not setup.xblock_jax_factors
    assert setup.xblock_jax_factor_format == "padded"
    assert setup.xblock_jax_factor_apply == "exact"
    assert any(
        "requires host sparse factors" in message for _, message in setup.messages
    )


def test_xblock_qi_device_operator_reuse_setup_skips_local_factors() -> None:
    calls: list[dict[str, object]] = []

    def reuse_decision(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(skip_xblock_factors=True)

    setup = resolve_xblock_qi_device_operator_reuse_setup(
        op=_op(fp=True, constraint_scheme=1, n_zeta=5, n_species=1),
        xblock_krylov_method="fgmres_jax",
        xblock_device_host_fallback_decision=SimpleNamespace(used=False),
        qi_device_preconditioner_requested=True,
        qi_device_matrix_free_requested=True,
        qi_device_use_in_krylov_requested=True,
        precondition_side="right",
        xblock_jax_factors=True,
        xblock_device_krylov_forced_jax_factors=True,
        xblock_preconditioner_xi=3,
        reuse_decision=reuse_decision,
        env={"SFINCS_JAX_RHSMODE1_XBLOCK_QI_DEVICE_OPERATOR_REUSE": "auto"},
    )

    assert setup.skip_xblock_factors
    assert not setup.xblock_jax_factors
    assert not setup.xblock_device_krylov_forced_jax_factors
    assert calls[0]["env_value"] == "auto"
    assert calls[0]["requested_krylov_method"] == "fgmres_jax"
    assert any(
        "skipping local x-block factors" in message for _, message in setup.messages
    )


def test_xblock_qi_device_operator_reuse_setup_reports_factor_build_route() -> None:
    def reuse_decision(**_kwargs):
        return SimpleNamespace(skip_xblock_factors=False)

    setup = resolve_xblock_qi_device_operator_reuse_setup(
        op=_op(fp=True, constraint_scheme=1, n_zeta=3, n_species=1),
        xblock_krylov_method="gmres_jax",
        xblock_device_host_fallback_decision=SimpleNamespace(used=False),
        qi_device_preconditioner_requested=False,
        qi_device_matrix_free_requested=False,
        qi_device_use_in_krylov_requested=False,
        precondition_side="right",
        xblock_jax_factors=True,
        xblock_device_krylov_forced_jax_factors=True,
        xblock_preconditioner_xi=1,
        reuse_decision=reuse_decision,
        env={},
    )

    assert not setup.skip_xblock_factors
    assert setup.xblock_jax_factors
    assert setup.factor_backend == "jax"
    assert setup.factor_reason == " device-krylov"
    assert any(
        "building jax x-block preconditioner" in message
        for _, message in setup.messages
    )


def test_xblock_krylov_matvec_setup_reduces_active_dofs_and_counts_progress() -> None:
    messages: list[str] = []
    active_idx = jnp.asarray([0, 2], dtype=jnp.int32)
    full_to_active = jnp.asarray([0, -1, 1, -1], dtype=jnp.int32)
    setup = build_xblock_krylov_matvec_setup(
        op=SimpleNamespace(total_size=4),
        rhs=jnp.asarray([1.0, 2.0, 3.0, 4.0]),
        xblock_use_active_dof=True,
        active_idx=active_idx,
        full_to_active=full_to_active,
        reduce_full_with_indices=lambda v, idx: v[idx],
        expand_reduced_with_map=lambda v, fmap: jnp.where(
            fmap >= 0, v[jnp.maximum(fmap, 0)], 0.0
        ),
        operator_matvec=lambda v: 2.0 * v,
        elapsed_s=lambda: 12.5,
        emit=lambda _level, msg: messages.append(msg),
        env={"SFINCS_JAX_SPARSE_PC_PROGRESS_EVERY": "2"},
    )

    assert setup.xblock_linear_size == 2
    assert setup.xblock_active_idx_np.tolist() == [0, 2]
    assert setup.xblock_rhs.tolist() == [1.0, 3.0]
    assert setup.matvec_no_count(jnp.asarray([5.0, 7.0])).tolist() == [10.0, 14.0]
    assert int(setup.mv_count) == 0
    assert setup.matvec(jnp.asarray([1.0, 2.0])).tolist() == [2.0, 4.0]
    assert int(setup.mv_count) == 1
    assert setup.matvec(jnp.asarray([2.0, 3.0])).tolist() == [4.0, 6.0]
    assert int(setup.mv_count) == 2
    assert any("active-DOF reduction" in message for _, message in setup.messages)
    assert any("matvecs=2" in message for message in messages)


def test_xblock_krylov_matvec_setup_full_space_is_identity_mapping() -> None:
    setup = build_xblock_krylov_matvec_setup(
        op=SimpleNamespace(total_size=3),
        rhs=jnp.asarray([1.0, 2.0, 3.0]),
        xblock_use_active_dof=False,
        active_idx=None,
        full_to_active=None,
        reduce_full_with_indices=lambda _v, _idx: (_ for _ in ()).throw(
            AssertionError("unused")
        ),
        expand_reduced_with_map=lambda _v, _idx: (_ for _ in ()).throw(
            AssertionError("unused")
        ),
        operator_matvec=lambda v: v + 1.0,
        elapsed_s=lambda: 0.0,
        emit=None,
        env={"SFINCS_JAX_SPARSE_PC_PROGRESS_EVERY": "bad"},
    )

    assert setup.progress_every == 25
    assert setup.xblock_linear_size == 3
    assert setup.xblock_active_idx_np is None
    assert setup.reduce_full(jnp.asarray([1.0, 2.0])).tolist() == [1.0, 2.0]
    assert setup.expand_reduced(jnp.asarray([1.0, 2.0])).tolist() == [1.0, 2.0]
    assert setup.matvec(jnp.asarray([1.0, 2.0, 3.0])).tolist() == [2.0, 3.0, 4.0]
    assert int(setup.mv_count) == 1


def test_xblock_krylov_matvec_setup_reuses_counter_and_custom_progress_label() -> None:
    messages: list[str] = []
    counter = MatvecCounter(3)
    setup = build_xblock_krylov_matvec_setup(
        op=SimpleNamespace(total_size=4),
        rhs=jnp.asarray([1.0, 2.0, 3.0, 4.0]),
        xblock_use_active_dof=True,
        active_idx=jnp.asarray([1, 3], dtype=jnp.int32),
        full_to_active=jnp.asarray([-1, 0, -1, 1], dtype=jnp.int32),
        reduce_full_with_indices=lambda v, idx: v[idx],
        expand_reduced_with_map=lambda v, fmap: jnp.where(
            fmap >= 0,
            v[jnp.maximum(fmap, 0)],
            0.0,
        ),
        operator_matvec=lambda v: 4.0 * v,
        elapsed_s=lambda: 8.25,
        emit=lambda _level, msg: messages.append(msg),
        progress_every=2,
        mv_count=counter,
        progress_label="fortran_reduced_pc_gmres xblock",
        emit_active_message=False,
    )

    assert setup.messages == ()
    assert setup.mv_count is counter
    assert int(counter) == 3
    assert setup.xblock_rhs.tolist() == [2.0, 4.0]
    assert setup.matvec(jnp.asarray([2.0, 3.0])).tolist() == [8.0, 12.0]
    assert int(counter) == 4
    assert any(
        "fortran_reduced_pc_gmres xblock matvecs=4" in message
        for message in messages
    )


def test_xblock_assembled_equilibration_setup_builds_row_scales() -> None:
    matrix = scipy_sparse.csr_matrix([[2.0, -1.0], [0.0, 4.0]])
    setup = build_xblock_assembled_equilibration_setup(
        assembled_matrix=matrix,
        xblock_linear_size=2,
        elapsed_s=lambda: 3.0,
        env={"SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_ROW_EQUILIBRATE": "1"},
    )

    assert setup.row_enabled
    assert setup.row_built
    assert not setup.col_enabled
    assert not setup.col_built
    assert setup.row_metadata["norm"] == "linf"
    assert np.asarray(setup.row_scale).tolist() == pytest.approx([0.5, 0.25])
    assert np.asarray(setup.inv_row_scale).tolist() == pytest.approx([2.0, 4.0])
    assert any(
        "assembled row equilibration built" in message for _, message in setup.messages
    )


def test_xblock_assembled_equilibration_setup_builds_row_and_column_scales() -> None:
    matrix = scipy_sparse.csr_matrix([[2.0, 0.0], [1.0, 4.0]])
    setup = build_xblock_assembled_equilibration_setup(
        assembled_matrix=matrix,
        xblock_linear_size=2,
        elapsed_s=lambda: 5.0,
        env={
            "SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_COL_EQUILIBRATE": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_ROW_EQUILIBRATE_NORM": "l1",
        },
    )

    assert setup.row_enabled
    assert setup.row_built
    assert setup.col_enabled
    assert setup.col_built
    assert setup.row_metadata["norm"] == "l1"
    assert setup.col_metadata["norm"] == "l1"
    assert setup.row_metadata["column_equilibration"] is True
    assert np.all(np.isfinite(np.asarray(setup.col_scale)))
    assert np.all(np.asarray(setup.col_scale) > 0.0)
    assert any(
        "assembled column equilibration built" in message
        for _, message in setup.messages
    )


def test_xblock_assembled_operator_preflight_uses_full_pattern_when_under_budget() -> (
    None
):
    full_pattern = object()
    full_summary = SimpleNamespace(nnz=4, shape=(2, 2), max_row_nnz=2, avg_row_nnz=2.0)

    setup = build_xblock_assembled_operator_preflight_setup(
        op=SimpleNamespace(),
        xblock_active_idx_np=None,
        sparse_pc_fp_dense_velocity_block=False,
        xblock_krylov_method="gmres_jax",
        estimate_summary=lambda *_args, **_kwargs: full_summary,
        full_pattern=lambda *_args, **_kwargs: full_pattern,
        active_pattern=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unused")
        ),
        summarize_pattern=lambda _op, pattern: full_summary
        if pattern is full_pattern
        else None,
        env={"SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_CSR_MAX_MB": "1"},
    )

    assert setup.pattern is full_pattern
    assert setup.summary is full_summary
    assert setup.device_enabled
    assert not setup.metadata["preflight_rejected"]
    assert setup.metadata["preflight_scope"] == "full"


def test_xblock_assembled_operator_preflight_uses_active_pattern_scope() -> None:
    full_summary = SimpleNamespace(
        nnz=1000, shape=(100, 100), max_row_nnz=20, avg_row_nnz=10.0
    )
    active_summary = SimpleNamespace(
        nnz=4, shape=(2, 2), max_row_nnz=2, avg_row_nnz=2.0
    )
    active_pattern = object()

    setup = build_xblock_assembled_operator_preflight_setup(
        op=SimpleNamespace(),
        xblock_active_idx_np=np.asarray([0, 2], dtype=np.int32),
        sparse_pc_fp_dense_velocity_block=False,
        xblock_krylov_method="gmres",
        estimate_summary=lambda *_args, **_kwargs: full_summary,
        full_pattern=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unused")
        ),
        active_pattern=lambda *_args, **_kwargs: active_pattern,
        summarize_pattern=lambda _op, pattern: active_summary
        if pattern is active_pattern
        else None,
        env={"SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_CSR_MAX_MB": "1"},
    )

    assert setup.pattern is active_pattern
    assert setup.summary is active_summary
    assert setup.metadata["preflight_scope"] == "active_dof"
    assert setup.metadata["preflight_active_pattern_nnz_estimate"] == 4
    assert not setup.device_enabled


def test_xblock_assembled_operator_preflight_rejection_carries_metadata() -> None:
    summary = SimpleNamespace(nnz=10, shape=(3, 3), max_row_nnz=4, avg_row_nnz=3.0)
    with pytest.raises(XBlockAssembledPreflightError) as excinfo:
        build_xblock_assembled_operator_preflight_setup(
            op=SimpleNamespace(),
            xblock_active_idx_np=None,
            sparse_pc_fp_dense_velocity_block=False,
            xblock_krylov_method="gmres",
            estimate_summary=lambda *_args, **_kwargs: summary,
            full_pattern=lambda *_args, **_kwargs: object(),
            active_pattern=lambda *_args, **_kwargs: object(),
            summarize_pattern=lambda _op, _pattern: summary,
            env={"SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_CSR_MAX_MB": "0"},
        )

    assert excinfo.value.metadata["preflight_rejected"] is True
    assert excinfo.value.metadata["preflight_pattern_nnz_estimate"] == 10
    assert "non-positive CSR memory budget" in str(excinfo.value)


def test_xblock_assembled_device_setup_builds_and_validates_operator() -> None:
    device_operator = SimpleNamespace(nnz=2, nbytes_estimate=64)

    setup = build_xblock_assembled_device_setup(
        assembled_matrix=object(),
        assembled_matvec=lambda x: x,
        csr_cap_nbytes=1024,
        device_enabled=True,
        device_required=False,
        validation_samples=2,
        validation_tol=1.0e-8,
        device_csr_from_matrix=lambda *_args, **_kwargs: device_operator,
        validate_device_csr_matvec=lambda *_args, **_kwargs: (0.0, 1.0e-12),
    )

    assert setup.device_operator is device_operator
    assert setup.device_resident
    assert setup.validation_errors == (0.0, 1.0e-12)
    assert setup.error is None


def test_xblock_assembled_device_setup_optional_failure_returns_message() -> None:
    setup = build_xblock_assembled_device_setup(
        assembled_matrix=object(),
        assembled_matvec=lambda x: x,
        csr_cap_nbytes=1,
        device_enabled=True,
        device_required=False,
        validation_samples=1,
        validation_tol=1.0e-8,
        device_csr_from_matrix=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            MemoryError("too large")
        ),
        validate_device_csr_matvec=lambda *_args, **_kwargs: (),
    )

    assert setup.device_operator is None
    assert not setup.device_resident
    assert "MemoryError" in str(setup.error)
    assert any(
        "disabled after build failure" in message for _, message in setup.messages
    )


def test_xblock_assembled_device_setup_required_failure_raises() -> None:
    with pytest.raises(RuntimeError, match="device CSR operator failed"):
        build_xblock_assembled_device_setup(
            assembled_matrix=object(),
            assembled_matvec=lambda x: x,
            csr_cap_nbytes=1,
            device_enabled=True,
            device_required=True,
            validation_samples=1,
            validation_tol=1.0e-8,
            device_csr_from_matrix=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                MemoryError("too large")
            ),
            validate_device_csr_matvec=lambda *_args, **_kwargs: (),
        )


def test_xblock_assembled_matvec_setup_host_counts_progress() -> None:
    messages: list[str] = []
    counter = build_xblock_krylov_matvec_setup(
        op=SimpleNamespace(total_size=2),
        rhs=jnp.asarray([0.0, 0.0]),
        xblock_use_active_dof=False,
        active_idx=None,
        full_to_active=None,
        reduce_full_with_indices=lambda _v, _idx: (_ for _ in ()).throw(
            AssertionError("unused")
        ),
        expand_reduced_with_map=lambda _v, _idx: (_ for _ in ()).throw(
            AssertionError("unused")
        ),
        operator_matvec=lambda v: v,
        elapsed_s=lambda: 0.0,
        emit=None,
        env={},
    ).mv_count
    setup = build_xblock_assembled_matvec_setup(
        assembled_matvec=lambda x: 3.0 * x,
        device_operator=None,
        mv_count=counter,
        progress_every=2,
        elapsed_s=lambda: 4.0,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert setup.location == "host"
    assert setup.matvec(jnp.asarray([1.0, 2.0])).tolist() == [3.0, 6.0]
    assert setup.matvec(jnp.asarray([2.0, 3.0])).tolist() == [6.0, 9.0]
    assert int(counter) == 2
    assert any("assembled_host_matvecs=2" in message for message in messages)


def test_xblock_assembled_matvec_setup_device_counts_progress() -> None:
    messages: list[str] = []
    counter = build_xblock_krylov_matvec_setup(
        op=SimpleNamespace(total_size=2),
        rhs=jnp.asarray([0.0, 0.0]),
        xblock_use_active_dof=False,
        active_idx=None,
        full_to_active=None,
        reduce_full_with_indices=lambda _v, _idx: (_ for _ in ()).throw(
            AssertionError("unused")
        ),
        expand_reduced_with_map=lambda _v, _idx: (_ for _ in ()).throw(
            AssertionError("unused")
        ),
        operator_matvec=lambda v: v,
        elapsed_s=lambda: 0.0,
        emit=None,
        env={},
    ).mv_count
    device_operator = SimpleNamespace(jitted_matvec=lambda: (lambda v: 5.0 * v))
    setup = build_xblock_assembled_matvec_setup(
        assembled_matvec=lambda _x: (_ for _ in ()).throw(AssertionError("unused")),
        device_operator=device_operator,
        mv_count=counter,
        progress_every=1,
        elapsed_s=lambda: 7.0,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert setup.location == "device"
    assert setup.matvec(jnp.asarray([1.0, 2.0])).tolist() == [5.0, 10.0]
    assert int(counter) == 1
    assert any("assembled_device_matvecs=1" in message for message in messages)


def test_finalize_xblock_assembled_operator_metadata_normalizes_fields() -> None:
    metadata = finalize_xblock_assembled_operator_metadata(
        metadata={"preflight_scope": "full"},
        setup_s=1.25,
        assembled_matrix=scipy_sparse.csr_matrix([[1.0, 0.0], [2.0, 3.0]]),
        assembled_summary=SimpleNamespace(nnz=3, avg_row_nnz=1.5, max_row_nnz=2),
        assembled_bundle_metadata=SimpleNamespace(
            storage_kind="csr",
            reason="materialized",
            csr_nbytes_estimate=128,
        ),
        max_colors=4,
        validation_errors=(1.0e-12,),
        device_enabled=True,
        device_required=False,
        device_resident=True,
        device_operator=SimpleNamespace(nnz=3, nbytes_estimate=96),
        device_validation_errors=(2.0e-12,),
        device_error=None,
    )

    assert metadata["preflight_scope"] == "full"
    assert metadata["matrix_nnz"] == 3
    assert metadata["pattern_avg_row_nnz"] == pytest.approx(1.5)
    assert metadata["device_nnz"] == 3
    assert metadata["device_validation_rel_errors"] == (2.0e-12,)


def test_xblock_moment_schur_policy_defaults_on_for_constraint1_device_krylov() -> None:
    setup = resolve_xblock_moment_schur_policy_setup(
        op=SimpleNamespace(rhs_mode=1, constraint_scheme=1, extra_size=2, phi1_size=0),
        xblock_krylov_method="gmres_jax",
        xblock_jax_factors=False,
        xblock_jax_factor_format="padded",
        precondition_side="right",
        env={},
    )

    assert setup.default_candidate
    assert setup.enabled
    assert not setup.default_blocked_by_compact_factors
    assert any("moment-Schur build start" in message for _, message in setup.messages)


def test_xblock_moment_schur_policy_blocks_compact_csr_default_but_allows_force() -> (
    None
):
    op = SimpleNamespace(rhs_mode=1, constraint_scheme=1, extra_size=2, phi1_size=0)
    blocked = resolve_xblock_moment_schur_policy_setup(
        op=op,
        xblock_krylov_method="gmres_jax",
        xblock_jax_factors=True,
        xblock_jax_factor_format="csr",
        precondition_side="right",
        env={},
    )
    forced = resolve_xblock_moment_schur_policy_setup(
        op=op,
        xblock_krylov_method="gmres_jax",
        xblock_jax_factors=True,
        xblock_jax_factor_format="csr",
        precondition_side="right",
        env={
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_MOMENT_SCHUR": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_MOMENT_SCHUR_RCOND": "1e-9",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_MOMENT_SCHUR_PROBE": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_MOMENT_SCHUR_MIN_IMPROVEMENT": "0.25",
        },
    )

    assert blocked.default_blocked_by_compact_factors
    assert not blocked.enabled
    assert any("default disabled" in message for _, message in blocked.messages)
    assert forced.enabled
    assert forced.rcond == pytest.approx(1.0e-9)
    assert forced.probe_enabled
    assert forced.probe_min_improvement == pytest.approx(0.25)


def test_xblock_moment_schur_policy_does_not_emit_build_for_no_preconditioner_side() -> (
    None
):
    setup = resolve_xblock_moment_schur_policy_setup(
        op=SimpleNamespace(rhs_mode=1, constraint_scheme=1, extra_size=2, phi1_size=0),
        xblock_krylov_method="gmres_jax",
        xblock_jax_factors=False,
        xblock_jax_factor_format="padded",
        precondition_side="none",
        env={},
    )

    assert setup.enabled
    assert not setup.messages


def test_xblock_moment_schur_probe_result_accepts_sufficient_reduction() -> None:
    result = evaluate_xblock_moment_schur_probe_result(
        residual_before=10.0,
        residual_after=7.0,
        min_improvement=0.2,
    )

    assert result.used
    assert result.reason == "probe_reduced"
    assert result.improvement_ratio == pytest.approx(0.7)
    assert any("accepted" in message for _, message in result.messages)


def test_xblock_moment_schur_probe_result_rejects_insufficient_reduction() -> None:
    result = evaluate_xblock_moment_schur_probe_result(
        residual_before=10.0,
        residual_after=9.0,
        min_improvement=0.2,
    )

    assert not result.used
    assert result.reason == "probe_not_reduced"
    assert result.improvement_ratio == pytest.approx(0.9)
    assert any("rejected" in message for _, message in result.messages)


def test_xblock_moment_schur_probe_result_handles_zero_rhs_norm() -> None:
    zero = evaluate_xblock_moment_schur_probe_result(
        residual_before=0.0,
        residual_after=0.0,
        min_improvement=0.5,
    )
    nonzero = evaluate_xblock_moment_schur_probe_result(
        residual_before=0.0,
        residual_after=1.0,
        min_improvement=0.5,
    )

    assert zero.used
    assert zero.improvement_ratio == 0.0
    assert not nonzero.used
    assert np.isinf(nonzero.improvement_ratio)


def test_xblock_moment_schur_metadata_helpers_normalize_success_and_failure() -> None:
    success = finalize_xblock_moment_schur_metadata(
        metadata={"rank": 3},
        setup_s=1.5,
    )
    failure = failed_xblock_moment_schur_metadata(
        exc=ValueError("bad factor"),
        setup_s=2.5,
    )

    assert success == {"rank": 3, "setup_s": 1.5}
    assert failure["setup_s"] == 2.5
    assert failure["error"] == "ValueError: bad factor"


def test_xblock_two_level_policy_defaults_off_and_honors_disabled_side() -> None:
    off = resolve_xblock_two_level_policy_setup(precondition_side="right", env={})
    no_side = resolve_xblock_two_level_policy_setup(
        precondition_side="none",
        env={"SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL": "1"},
    )

    assert not off.enabled
    assert not off.should_build
    assert off.mode == "additive"
    assert no_side.enabled
    assert not no_side.should_build


def test_xblock_two_level_policy_parses_build_parameters() -> None:
    setup = resolve_xblock_two_level_policy_setup(
        precondition_side="left",
        env={
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL_MODE": "multiplicative",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL_MAX_DIRECTIONS": "7",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL_FSAVG_LMAX": "3",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL_MAX_EXTRA_UNITS": "2",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL_RCOND": "1e-8",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL_INCLUDE_RHS": "0",
        },
    )

    assert setup.enabled
    assert setup.should_build
    assert setup.mode == "multiplicative"
    assert setup.max_directions == 7
    assert setup.fsavg_lmax == 3
    assert setup.max_extra_units == 2
    assert setup.rcond == pytest.approx(1.0e-8)
    assert not setup.include_rhs


def test_xblock_two_level_metadata_helpers_normalize_success_and_failure() -> None:
    success = finalize_xblock_two_level_metadata(
        metadata={"mode": "additive"}, setup_s=0.25
    )
    failure = failed_xblock_two_level_metadata(
        exc=RuntimeError("bad coarse"), setup_s=0.5
    )

    assert success == {"mode": "additive", "setup_s": 0.25}
    assert failure == {"error": "RuntimeError: bad coarse", "setup_s": 0.5}


def test_xblock_global_coupling_policy_defaults_off_and_selects_builder_defaults() -> (
    None
):
    off = resolve_xblock_global_coupling_policy_setup(
        precondition_side="right",
        xblock_krylov_method="gmres",
        env={},
    )
    device = resolve_xblock_global_coupling_policy_setup(
        precondition_side="right",
        xblock_krylov_method="gmres_jax",
        env={"SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING": "1"},
    )
    no_side = resolve_xblock_global_coupling_policy_setup(
        precondition_side="none",
        xblock_krylov_method="gmres_jax",
        env={"SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING": "1"},
    )

    assert not off.enabled
    assert not off.should_build
    assert not off.use_device_builder
    assert off.setup_max_s == 0.0
    assert device.enabled
    assert device.should_build
    assert device.use_device_builder
    assert device.setup_max_s == pytest.approx(180.0)
    assert no_side.enabled
    assert not no_side.should_build


def test_xblock_global_coupling_policy_parses_build_parameters() -> None:
    setup = resolve_xblock_global_coupling_policy_setup(
        precondition_side="left",
        xblock_krylov_method="bicgstab_jax",
        env={
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_MODE": "multiplicative",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_MAX_DIRECTIONS": "9",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_FSAVG_LMAX": "3",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_ANGULAR_LMAX": "4",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_MAX_EXTRA_UNITS": "5",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_RCOND": "1e-7",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_INCLUDE_RHS": "0",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_SETUP_MAX_S": "12.5",
        },
    )

    assert setup.enabled
    assert setup.should_build
    assert setup.use_device_builder
    assert setup.mode == "multiplicative"
    assert setup.max_directions == 9
    assert setup.fsavg_lmax == 3
    assert setup.angular_lmax == 4
    assert setup.max_extra_units == 5
    assert setup.rcond == pytest.approx(1.0e-7)
    assert not setup.include_rhs
    assert setup.setup_max_s == pytest.approx(12.5)


def test_xblock_global_coupling_metadata_helpers_normalize_success_and_failure() -> (
    None
):
    success = finalize_xblock_global_coupling_metadata(
        metadata={"mode": "additive"}, setup_s=0.75
    )
    failure = failed_xblock_global_coupling_metadata(
        exc=RuntimeError("timeout"), setup_s=1.5
    )

    assert success == {"mode": "additive", "setup_s": 0.75}
    assert failure == {"error": "RuntimeError: timeout", "setup_s": 1.5}


def test_prepare_xblock_initial_guess_accepts_reduced_and_full_active_shapes() -> None:
    reduced = jnp.asarray([1.0, 2.0])
    full = jnp.asarray([10.0, 11.0, 12.0, 13.0])
    rhs_reduced = jnp.zeros(2)
    rhs_full = jnp.zeros(4)

    reduced_result = prepare_xblock_initial_guess(
        x0=reduced,
        xblock_rhs=rhs_reduced,
        full_rhs=rhs_full,
        xblock_use_active_dof=True,
        reduce_full=lambda v: v[jnp.asarray([0, 2])],
    )
    full_result = prepare_xblock_initial_guess(
        x0=full,
        xblock_rhs=rhs_reduced,
        full_rhs=rhs_full,
        xblock_use_active_dof=True,
        reduce_full=lambda v: v[jnp.asarray([0, 2])],
    )

    assert reduced_result.messages == ()
    assert jnp.asarray(reduced_result.x0_full).tolist() == [1.0, 2.0]
    assert full_result.messages == ()
    assert jnp.asarray(full_result.x0_full).tolist() == [10.0, 12.0]


def test_prepare_xblock_initial_guess_rejects_incompatible_shape_with_message() -> None:
    result = prepare_xblock_initial_guess(
        x0=jnp.ones(3),
        xblock_rhs=jnp.zeros(2),
        full_rhs=jnp.zeros(4),
        xblock_use_active_dof=True,
        reduce_full=lambda v: v,
    )

    assert result.x0_full is None
    assert len(result.messages) == 1
    assert "ignoring incompatible x0 shape=(3,)" in result.messages[0][1]
    assert "expected=(2,) or (4,)" in result.messages[0][1]


def test_prepare_fortran_reduced_xblock_initial_guess_routes_reduced_and_full_shapes() -> None:
    reduced = jnp.asarray([1.0, 2.0])
    full = jnp.asarray([10.0, 11.0, 12.0, 13.0])
    rhs_reduced = jnp.zeros(2)
    rhs_full = jnp.zeros(4)

    reduced_result = prepare_fortran_reduced_xblock_initial_guess(
        x0=reduced,
        sparse_pc_rhs=rhs_reduced,
        full_rhs=rhs_full,
        reduce_full=lambda v: v[jnp.asarray([1, 3])],
    )
    full_result = prepare_fortran_reduced_xblock_initial_guess(
        x0=full,
        sparse_pc_rhs=rhs_reduced,
        full_rhs=rhs_full,
        reduce_full=lambda v: v[jnp.asarray([1, 3])],
    )

    assert reduced_result.messages == ()
    assert jnp.asarray(reduced_result.x0_full).tolist() == [1.0, 2.0]
    assert full_result.messages == ()
    assert jnp.asarray(full_result.x0_full).tolist() == [11.0, 13.0]


def test_prepare_fortran_reduced_xblock_initial_guess_handles_none_and_bad_shape() -> None:
    none_result = prepare_fortran_reduced_xblock_initial_guess(
        x0=None,
        sparse_pc_rhs=jnp.zeros(2),
        full_rhs=jnp.zeros(4),
        reduce_full=lambda v: v,
    )
    bad_result = prepare_fortran_reduced_xblock_initial_guess(
        x0=jnp.ones(3),
        sparse_pc_rhs=jnp.zeros(2),
        full_rhs=jnp.zeros(4),
        reduce_full=lambda v: v,
    )

    assert none_result.x0_full is None
    assert none_result.messages == ()
    assert bad_result.x0_full is None
    assert len(bad_result.messages) == 1
    assert "fortran_reduced_pc_gmres xblock ignoring incompatible x0 shape=(3,)" in (
        bad_result.messages[0][1]
    )
    assert "expected=(2,) or (4,)" in bad_result.messages[0][1]


def test_xblock_seed_policy_defaults_and_env_overrides() -> None:
    default = resolve_xblock_seed_policy_setup(moment_schur_used=True, env={})
    disabled = resolve_xblock_seed_policy_setup(
        moment_schur_used=True,
        env={
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_INITIAL_SEED": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_MOMENT_SCHUR_SEED": "0",
        },
    )

    assert not default.initial_seed_enabled
    assert default.moment_schur_seed_enabled
    assert disabled.initial_seed_enabled
    assert not disabled.moment_schur_seed_enabled


def test_xblock_qi_seed_policy_defaults_off_without_shared_basis() -> None:
    setup = resolve_xblock_qi_seed_policy_setup(env={})

    assert not setup.coarse_seed_enabled
    assert not setup.galerkin_preconditioner_enabled
    assert not setup.two_level_preconditioner_enabled
    assert not setup.device_preconditioner_enabled
    assert not setup.deflated_preconditioner_enabled
    assert not setup.shared_basis_required
    assert setup.max_rank == 0
    assert setup.basis_kind is None


def test_xblock_qi_seed_policy_deflated_only_does_not_parse_shared_basis() -> None:
    setup = resolve_xblock_qi_seed_policy_setup(
        env={
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEFLATED_PRECONDITIONER": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_MAX_RANK": "48",
        },
    )

    assert setup.deflated_preconditioner_enabled
    assert not setup.shared_basis_required
    assert setup.max_rank == 0
    assert setup.max_candidates == 0


def test_xblock_qi_seed_policy_parses_shared_basis_parameters() -> None:
    setup = resolve_xblock_qi_seed_policy_setup(
        env={
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_GALERKIN_PRECONDITIONER": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_MAX_RANK": "10",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_MAX_CANDIDATES": "24",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_MAX_ANGULAR_MODE": "4",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_RANK_RTOL": "1e-8",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_MIN_IMPROVEMENT": "0.15",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_RCOND": "1e-9",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_INCLUDE_ANGULAR": "0",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_INCLUDE_BLOCKS": "0",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_INCLUDE_RADIAL": "0",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_INCLUDE_RADIAL_ANGULAR": "0",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_INCLUDE_CONSTRAINT_MOMENTS": "0",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_INCLUDE_SCHUR": "0",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_BASIS": "Residual-Enriched",
        },
    )

    assert setup.coarse_seed_enabled
    assert setup.galerkin_preconditioner_enabled
    assert setup.two_level_preconditioner_enabled
    assert setup.device_preconditioner_enabled
    assert setup.shared_basis_required
    assert setup.max_rank == 10
    assert setup.max_candidates == 24
    assert setup.max_angular_mode == 4
    assert setup.rank_rtol == pytest.approx(1.0e-8)
    assert setup.min_improvement == pytest.approx(0.15)
    assert setup.rcond == pytest.approx(1.0e-9)
    assert not setup.include_angular
    assert not setup.include_blocks
    assert not setup.include_radial
    assert not setup.include_radial_angular
    assert not setup.include_constraint_moments
    assert not setup.include_schur
    assert setup.basis_kind == "residual_enriched"


def test_xblock_qi_galerkin_policy_handles_disabled_and_fallback_cases() -> None:
    def parse_modes(raw, *, default="auto"):
        return ("additive", "multiplicative") if (raw or default) == "auto" else (raw,)

    def parse_dampings(raw, *, default=1.0, auto_defaults=(1.0, 0.5, 0.25)):
        return (
            tuple(auto_defaults)
            if not raw
            else tuple(float(v) for v in str(raw).split(","))
        )

    off = resolve_xblock_qi_galerkin_policy_setup(
        enabled=False,
        host_fallback_used=False,
        precondition_side="right",
        parse_modes=parse_modes,
        parse_dampings=parse_dampings,
        env={},
    )
    fallback = resolve_xblock_qi_galerkin_policy_setup(
        enabled=True,
        host_fallback_used=True,
        precondition_side="right",
        parse_modes=parse_modes,
        parse_dampings=parse_dampings,
        env={},
    )

    assert not off.should_build
    assert off.reason is None
    assert fallback.enabled
    assert not fallback.should_build
    assert fallback.reason == "disabled_by_device_host_fallback"
    assert any(
        "device-host fallback" in message for _level, message in fallback.messages
    )


def test_xblock_qi_galerkin_policy_parses_build_parameters() -> None:
    def parse_modes(raw, *, default="auto"):
        return tuple(str(raw or default).split(","))

    def parse_dampings(raw, *, default=1.0, auto_defaults=(1.0, 0.5, 0.25)):
        return (
            tuple(auto_defaults)
            if not raw
            else tuple(float(v) for v in str(raw).split(","))
        )

    setup = resolve_xblock_qi_galerkin_policy_setup(
        enabled=True,
        host_fallback_used=False,
        precondition_side="right",
        parse_modes=parse_modes,
        parse_dampings=parse_dampings,
        env={
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_GALERKIN_PRECONDITIONER_MODE": "multiplicative",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_GALERKIN_PRECONDITIONER_RCOND": "1e-8",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_GALERKIN_PRECONDITIONER_DAMPING": "0.6",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_GALERKIN_PRECONDITIONER_DAMPINGS": "0.6,0.3",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_GALERKIN_PRECONDITIONER_PROBE": "0",
        },
    )

    assert setup.should_build
    assert setup.preconditioner_mode == "multiplicative"
    assert setup.candidate_modes == ("multiplicative",)
    assert setup.rcond == pytest.approx(1.0e-8)
    assert setup.damping == pytest.approx(0.6)
    assert setup.candidate_dampings == (0.6, 0.3)
    assert not setup.probe_enabled


def test_xblock_qi_two_level_policy_handles_disabled_and_side_none_cases() -> None:
    def parse_dampings(raw, *, default=1.0, auto_defaults=(1.0,)):
        return (
            tuple(auto_defaults)
            if not raw
            else tuple(float(v) for v in str(raw).split(","))
        )

    off = resolve_xblock_qi_two_level_policy_setup(
        enabled=False,
        host_fallback_used=False,
        precondition_side="right",
        seed_max_rank=8,
        parse_dampings=parse_dampings,
        env={},
    )
    side_none = resolve_xblock_qi_two_level_policy_setup(
        enabled=True,
        host_fallback_used=False,
        precondition_side="none",
        seed_max_rank=8,
        parse_dampings=parse_dampings,
        env={},
    )

    assert not off.should_build
    assert off.smoothed_load_max_rank == 8
    assert not side_none.should_build
    assert side_none.reason == "disabled_by_precondition_side_none"


def test_xblock_qi_two_level_policy_parses_build_parameters() -> None:
    def parse_dampings(raw, *, default=1.0, auto_defaults=(1.0,)):
        return (
            tuple(auto_defaults)
            if not raw
            else tuple(float(v) for v in str(raw).split(","))
        )

    setup = resolve_xblock_qi_two_level_policy_setup(
        enabled=True,
        host_fallback_used=False,
        precondition_side="right",
        seed_max_rank=8,
        parse_dampings=parse_dampings,
        env={
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_RCOND": "1e-9",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_DAMPING": "0.7",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_DAMPINGS": "0.7,0.35",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_MIN_IMPROVEMENT": "0.2",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_COARSE_SOLVER": "Action-Lstsq",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_RESIDUAL_AUGMENT": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_RESIDUAL_AUGMENT_MAX_EXTRA": "2",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_RESIDUAL_AUGMENT_STEPS": "3",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_RESIDUAL_AUGMENT_INCLUDE_RESIDUALS": "0",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_SMOOTHED_LOAD_BASIS": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_SMOOTHED_LOAD_BASIS_COMBINE": "0",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_SMOOTHED_LOAD_MAX_DIRECTIONS": "12",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_SMOOTHED_LOAD_MAX_RANK": "5",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_SMOOTHED_LOAD_FSAVG_LMAX": "2",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_SMOOTHED_LOAD_ANGULAR_LMAX": "3",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_SMOOTHED_LOAD_MAX_EXTRA_UNITS": "4",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_SMOOTHED_LOAD_INCLUDE_RHS": "0",
        },
    )

    assert setup.should_build
    assert setup.rcond == pytest.approx(1.0e-9)
    assert setup.damping == pytest.approx(0.7)
    assert setup.candidate_dampings == (0.7, 0.35)
    assert setup.min_improvement == pytest.approx(0.2)
    assert setup.coarse_solver == "action_lstsq"
    assert setup.residual_augment
    assert setup.residual_augment_max_extra == 2
    assert setup.residual_augment_steps == 3
    assert not setup.residual_augment_include_residuals
    assert setup.smoothed_load_basis
    assert not setup.smoothed_load_basis_combine
    assert setup.smoothed_load_max_directions == 12
    assert setup.smoothed_load_max_rank == 5
    assert setup.smoothed_load_fsavg_lmax == 2
    assert setup.smoothed_load_angular_lmax == 3
    assert setup.smoothed_load_max_extra_units == 4
    assert not setup.smoothed_load_include_rhs


def test_xblock_qi_device_admission_defaults_off_and_handles_host_fallback() -> None:
    off = resolve_xblock_qi_device_admission_setup(
        enabled=False,
        host_fallback_used=False,
        assembled_device_operator_available=False,
        assembled_operator_enabled=False,
        assembled_operator_built=False,
        assembled_operator_device_resident=False,
        assembled_operator_device_error=None,
        env={},
    )
    fallback = resolve_xblock_qi_device_admission_setup(
        enabled=True,
        host_fallback_used=True,
        assembled_device_operator_available=True,
        assembled_operator_enabled=True,
        assembled_operator_built=True,
        assembled_operator_device_resident=True,
        assembled_operator_device_error=None,
        env={"SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE": "1"},
    )

    assert not off.enabled
    assert not off.should_build
    assert fallback.enabled
    assert not fallback.should_build
    assert fallback.matrix_free_enabled
    assert fallback.reason == "disabled_by_device_host_fallback"
    assert any(
        "device-host fallback" in message for _level, message in fallback.messages
    )


def test_xblock_qi_device_admission_records_missing_device_metadata() -> None:
    setup = resolve_xblock_qi_device_admission_setup(
        enabled=True,
        host_fallback_used=False,
        assembled_device_operator_available=False,
        assembled_operator_enabled=True,
        assembled_operator_built=True,
        assembled_operator_device_resident=False,
        assembled_operator_device_error="validation failed",
        env={},
    )

    assert not setup.should_build
    assert setup.reason == "disabled_missing_assembled_device_operator"
    assert setup.metadata["assembled_operator_enabled"] is True
    assert setup.metadata["assembled_operator_built"] is True
    assert setup.metadata["assembled_operator_device_resident"] is False
    assert setup.metadata["assembled_operator_device_error"] == "validation failed"
    assert (
        "SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR=1" in setup.metadata["requires"]
    )
    assert any(
        "no assembled device CSR operator" in message
        for _level, message in setup.messages
    )


def test_xblock_qi_device_admission_allows_matrix_free_without_device_operator() -> (
    None
):
    setup = resolve_xblock_qi_device_admission_setup(
        enabled=True,
        host_fallback_used=False,
        assembled_device_operator_available=False,
        assembled_operator_enabled=False,
        assembled_operator_built=False,
        assembled_operator_device_resident=False,
        assembled_operator_device_error=None,
        env={"SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE": "1"},
    )

    assert setup.should_build
    assert setup.matrix_free_enabled
    assert setup.reason is None
    assert setup.metadata == {}


def test_xblock_qi_device_base_config_defaults_with_device_operator() -> None:
    setup = resolve_xblock_qi_device_base_config_setup(
        matrix_free_enabled=False,
        assembled_device_operator_available=True,
        precondition_side="right",
        probe_uses_minres_step=lambda: True,
        env={},
    )

    assert setup.rcond == pytest.approx(1.0e-12)
    assert setup.damping == pytest.approx(1.0)
    assert setup.jacobi_damping == pytest.approx(0.7)
    assert setup.jacobi_sweeps == 1
    assert setup.jacobi_floor == pytest.approx(1.0e-14)
    assert setup.jacobi_require_all_diagonal
    assert setup.local_smoother_kind == "auto"
    assert setup.matrix_free_smoother_sweeps == 1
    assert setup.matrix_free_smoother_damping == pytest.approx(1.0)
    assert setup.matrix_free_smoother_step_policy == "residual_minimizing"
    assert setup.matrix_free_block_smoother_max_groups == 32
    assert setup.matrix_free_block_smoother_include_tail
    assert setup.matrix_free_block_smoother_grouping == "contiguous"
    assert setup.jacobi_step_policy == "stationary"
    assert setup.coarse_solver == "action_lstsq"
    assert setup.min_improvement == pytest.approx(0.05)
    assert setup.cycles == 1
    assert not setup.augmented_seed_requested
    assert setup.augmented_seed_max_rank == 1
    assert setup.minres_step
    assert setup.alpha_clip == pytest.approx(10.0)
    assert setup.use_in_krylov_requested
    assert setup.use_in_krylov
    assert not setup.compose_with_base
    assert setup.compose_mode == "multiplicative"


def test_xblock_qi_device_base_config_parses_matrix_free_and_composition_settings() -> (
    None
):
    setup = resolve_xblock_qi_device_base_config_setup(
        matrix_free_enabled=True,
        assembled_device_operator_available=False,
        precondition_side="none",
        probe_uses_minres_step=lambda: False,
        env={
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RCOND": "1e-8",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_DAMPING": "0.6",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_JACOBI_DAMPING": "0.4",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_JACOBI_SWEEPS": "3",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_JACOBI_DIAGONAL_FLOOR": "1e-9",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_JACOBI_REQUIRE_ALL_DIAGONAL": "0",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_LOCAL_SMOOTHER": "matrix-free-block-minres",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE_SMOOTHER_SWEEPS": "4",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE_SMOOTHER_DAMPING": "0.75",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE_SMOOTHER_STEP_POLICY": "Fixed",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE_SMOOTHER_ALPHA_CLIP": "2.5",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE_BLOCK_SMOOTHER_MAX_GROUPS": "7",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE_BLOCK_SMOOTHER_INCLUDE_TAIL": "0",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE_BLOCK_SMOOTHER_RCOND": "1e-7",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE_BLOCK_SMOOTHER_GROUPING": "block-x-species",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_JACOBI_STEP_POLICY": "Residual-Minimizing",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_COARSE_SOLVER": "Galerkin",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MIN_IMPROVEMENT": "0.2",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_CYCLES": "5",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_AUGMENTED_SEED": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_AUGMENTED_SEED_MAX_RANK": "9",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_ALPHA_CLIP": "3.5",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_USE_IN_KRYLOV": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_COMPOSE_WITH_BASE": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_COMPOSE_MODE": "invalid",
        },
    )

    assert setup.rcond == pytest.approx(1.0e-8)
    assert setup.damping == pytest.approx(0.6)
    assert setup.jacobi_damping == pytest.approx(0.4)
    assert setup.jacobi_sweeps == 3
    assert setup.jacobi_floor == pytest.approx(1.0e-9)
    assert not setup.jacobi_require_all_diagonal
    assert setup.local_smoother_kind == "matrix_free_block_minres"
    assert setup.matrix_free_smoother_sweeps == 4
    assert setup.matrix_free_smoother_damping == pytest.approx(0.75)
    assert setup.matrix_free_smoother_step_policy == "fixed"
    assert setup.matrix_free_smoother_alpha_clip == pytest.approx(2.5)
    assert setup.matrix_free_block_smoother_max_groups == 7
    assert not setup.matrix_free_block_smoother_include_tail
    assert setup.matrix_free_block_smoother_rcond == pytest.approx(1.0e-7)
    assert setup.matrix_free_block_smoother_grouping == "block_x_species"
    assert setup.jacobi_step_policy == "residual_minimizing"
    assert setup.coarse_solver == "galerkin"
    assert setup.min_improvement == pytest.approx(0.2)
    assert setup.cycles == 5
    assert setup.augmented_seed_requested
    assert setup.augmented_seed_max_rank == 9
    assert not setup.minres_step
    assert setup.alpha_clip == pytest.approx(3.5)
    assert setup.use_in_krylov_requested
    assert not setup.use_in_krylov
    assert setup.compose_with_base
    assert setup.compose_mode == "multiplicative"


def test_xblock_qi_device_enrichment_config_defaults_follow_matrix_free() -> None:
    off = resolve_xblock_qi_device_enrichment_config_setup(
        matrix_free_enabled=False, env={}
    )
    matrix_free = resolve_xblock_qi_device_enrichment_config_setup(
        matrix_free_enabled=True, env={}
    )

    assert not off.residual_enrichment
    assert off.residual_enrichment_depth == 0
    assert matrix_free.residual_enrichment
    assert matrix_free.residual_enrichment_depth == 2
    assert matrix_free.residual_enrichment_include_residual
    assert not matrix_free.recycle_enrichment
    assert matrix_free.recycle_cycles == 0
    assert not matrix_free.operator_krylov_enrichment
    assert matrix_free.operator_krylov_depth == 0
    assert not matrix_free.adjoint_krylov_enrichment
    assert matrix_free.adjoint_krylov_depth == 0
    assert matrix_free.adjoint_krylov_transpose_source == "autodiff"
    assert not matrix_free.operator_action_enrichment
    assert matrix_free.operator_action_depth == 0


def test_xblock_qi_device_enrichment_config_parses_explicit_settings() -> None:
    setup = resolve_xblock_qi_device_enrichment_config_setup(
        matrix_free_enabled=False,
        env={
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_ENRICHMENT": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_ENRICHMENT_DEPTH": "5",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_ENRICHMENT_INCLUDE_RESIDUAL": "0",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RECYCLE_ENRICHMENT": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RECYCLE_CYCLES": "3",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_OPERATOR_KRYLOV_ENRICHMENT": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_OPERATOR_KRYLOV_DEPTH": "2",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_ADJOINT_KRYLOV_ENRICHMENT": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_ADJOINT_KRYLOV_DEPTH": "4",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_ADJOINT_KRYLOV_TRANSPOSE": "Finite-Difference",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_OPERATOR_ACTION_ENRICHMENT": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_OPERATOR_ACTION_DEPTH": "6",
        },
    )

    assert setup.residual_enrichment
    assert setup.residual_enrichment_depth == 5
    assert not setup.residual_enrichment_include_residual
    assert setup.recycle_enrichment
    assert setup.recycle_cycles == 3
    assert setup.operator_krylov_enrichment
    assert setup.operator_krylov_depth == 2
    assert setup.adjoint_krylov_enrichment
    assert setup.adjoint_krylov_depth == 4
    assert setup.adjoint_krylov_transpose_source == "finite_difference"
    assert setup.operator_action_enrichment
    assert setup.operator_action_depth == 6


def test_xblock_qi_device_multilevel_config_defaults_disabled() -> None:
    setup = resolve_xblock_qi_device_multilevel_config_setup(env={})

    assert not setup.multilevel_coarse
    assert setup.multilevel_max_levels == 1
    assert setup.multilevel_aggregate_factor == 2
    assert setup.multilevel_max_angular_mode == 1
    assert setup.multilevel_max_radial_degree == 2
    assert setup.multilevel_max_pitch_degree == 0
    assert not setup.multilevel_current_moments
    assert setup.multilevel_species_current_moments
    assert setup.multilevel_radial_current_moments
    assert setup.multilevel_tail_constraint_moments
    assert setup.multilevel_current_max_pitch_degree == 1
    assert not setup.multilevel_residual_equation
    assert setup.multilevel_residual_equation_max_level_rank == 16
    assert setup.multilevel_residual_equation_order == "coarse_to_fine"
    assert setup.multilevel_residual_equation_solver == "action_lstsq"
    assert setup.multilevel_residual_equation_include_global


def test_xblock_qi_device_multilevel_config_reuses_coarse_operator_alias() -> None:
    alias = resolve_xblock_qi_device_multilevel_config_setup(
        env={
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_REUSE_COARSE_OPERATOR": "1",
        }
    )
    explicit_off = resolve_xblock_qi_device_multilevel_config_setup(
        env={
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_REUSE_COARSE_OPERATOR": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_COARSE": "0",
        }
    )

    assert alias.multilevel_coarse
    assert alias.multilevel_max_levels == 3
    assert not explicit_off.multilevel_coarse
    assert explicit_off.multilevel_max_levels == 1


def test_xblock_qi_device_multilevel_config_parses_explicit_controls() -> None:
    setup = resolve_xblock_qi_device_multilevel_config_setup(
        env={
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_COARSE": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_MAX_LEVELS": "4",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_AGGREGATE_FACTOR": "5",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_MAX_ANGULAR_MODE": "3",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_MAX_RADIAL_DEGREE": "6",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_MAX_PITCH_DEGREE": "2",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_CURRENT_MOMENTS": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_SPECIES_CURRENT_MOMENTS": "0",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_RADIAL_CURRENT_MOMENTS": "0",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_TAIL_CONSTRAINT_MOMENTS": "0",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_CURRENT_MAX_PITCH_DEGREE": "4",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_RESIDUAL_EQUATION": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_RESIDUAL_EQUATION_MAX_LEVEL_RANK": "7",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_RESIDUAL_EQUATION_ORDER": "fine-to-coarse",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_RESIDUAL_EQUATION_SOLVER": "qtaq",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_RESIDUAL_EQUATION_INCLUDE_GLOBAL": "0",
        }
    )

    assert setup.multilevel_coarse
    assert setup.multilevel_max_levels == 4
    assert setup.multilevel_aggregate_factor == 5
    assert setup.multilevel_max_angular_mode == 3
    assert setup.multilevel_max_radial_degree == 6
    assert setup.multilevel_max_pitch_degree == 2
    assert setup.multilevel_current_moments
    assert not setup.multilevel_species_current_moments
    assert not setup.multilevel_radial_current_moments
    assert not setup.multilevel_tail_constraint_moments
    assert setup.multilevel_current_max_pitch_degree == 4
    assert setup.multilevel_residual_equation
    assert setup.multilevel_residual_equation_max_level_rank == 7
    assert setup.multilevel_residual_equation_order == "fine_to_coarse"
    assert setup.multilevel_residual_equation_solver == "galerkin"
    assert not setup.multilevel_residual_equation_include_global


def test_xblock_qi_device_multilevel_config_normalizes_invalid_residual_controls() -> (
    None
):
    setup = resolve_xblock_qi_device_multilevel_config_setup(
        env={
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_RESIDUAL_EQUATION_ORDER": "inside-out",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_RESIDUAL_EQUATION_SOLVER": "least-squares",
        }
    )
    invalid_solver = resolve_xblock_qi_device_multilevel_config_setup(
        env={
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_RESIDUAL_EQUATION_SOLVER": "unknown",
        }
    )

    assert setup.multilevel_residual_equation_order == "coarse_to_fine"
    assert setup.multilevel_residual_equation_solver == "action_lstsq"
    assert invalid_solver.multilevel_residual_equation_solver == "action_lstsq"


def test_sparse_pc_gmres_control_policy_defaults() -> None:
    policy = resolve_sparse_pc_gmres_control_policy({})

    assert isinstance(policy, SparsePCGMRESControlPolicy)
    assert policy.stagnation_abort is False
    assert policy.stagnation_min_iter == 500
    assert policy.stagnation_window == 500
    assert policy.stagnation_rel_improvement == pytest.approx(1.0e-3)
    assert policy.post_minres_steps == 0
    assert policy.post_minres_alpha_clip == pytest.approx(10.0)
    assert policy.post_minres_min_improvement == 0.0


def test_sparse_pc_gmres_control_policy_overrides_and_clamps() -> None:
    policy = resolve_sparse_pc_gmres_control_policy(
        {
            "SFINCS_JAX_RHSMODE1_SPARSE_PC_STAGNATION_ABORT": "1",
            "SFINCS_JAX_RHSMODE1_SPARSE_PC_STAGNATION_MIN_ITER": "0",
            "SFINCS_JAX_RHSMODE1_SPARSE_PC_STAGNATION_WINDOW": "-2",
            "SFINCS_JAX_RHSMODE1_SPARSE_PC_STAGNATION_REL_IMPROVEMENT": "-0.1",
            "SFINCS_JAX_RHSMODE1_SPARSE_PC_POST_MINRES_STEPS": "-3",
            "SFINCS_JAX_RHSMODE1_SPARSE_PC_POST_MINRES_ALPHA_CLIP": "-1.5",
            "SFINCS_JAX_RHSMODE1_SPARSE_PC_POST_MINRES_MIN_IMPROVEMENT": "0.25",
        }
    )

    assert policy.stagnation_abort is True
    assert policy.stagnation_min_iter == 1
    assert policy.stagnation_window == 1
    assert policy.stagnation_rel_improvement == 0.0
    assert policy.post_minres_steps == 0
    assert policy.post_minres_alpha_clip == 0.0
    assert policy.post_minres_min_improvement == pytest.approx(0.25)


def test_sparse_pc_gmres_once_explicit_left_recomputes_true_residual() -> None:
    messages: list[str] = []
    times = iter((0.0, 0.25, 0.5, 0.75))

    def explicit_left_solver(**kwargs):
        kwargs["progress_callback"](2, 4.0e-1)
        return np.asarray([0.25, 0.75]), 99.0, 0.5, (1.0, 0.4)

    result = run_sparse_pc_gmres_once(
        context=SparsePCGMRESContext(
            matvec=lambda x: 2.0 * x,
            rhs=jnp.asarray([1.0, 1.0]),
            preconditioner=_identity,
            emit=lambda _level, msg: messages.append(msg),
            elapsed_s=lambda: next(times),
            pc_form="explicit_left",
            restart=7,
            tol=1.0e-8,
            atol=0.0,
            precondition_side="left",
            factor_dtype=np.dtype(np.float32),
            progress_every=2,
            stagnation_abort=False,
            stagnation_min_iter=10,
            stagnation_window=10,
            stagnation_rel_improvement=1.0e-3,
            explicit_left_solver=explicit_left_solver,
            gmres_solver=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("wrong solver")
            ),
        ),
        x0=None,
        maxiter=3,
    )

    assert result.x.tolist() == [0.25, 0.75]
    assert result.preconditioned_residual_norm == pytest.approx(0.5)
    assert result.residual_norm == pytest.approx(np.linalg.norm([0.5, -0.5]))
    assert result.history == (1.0, 0.4)
    assert any("factor_dtype=float32" in msg for msg in messages)
    assert any("iters=2" in msg for msg in messages)


def test_sparse_pc_gmres_once_stagnation_guard_raises() -> None:
    def gmres_solver(**kwargs):
        progress = kwargs["progress_callback"]
        progress(1, 1.0)
        progress(2, 1.0)
        return np.ones(2), 1.0, (1.0,)

    with pytest.raises(RuntimeError, match="sparse_pc_gmres stagnation detected"):
        run_sparse_pc_gmres_once(
            context=SparsePCGMRESContext(
                matvec=_identity,
                rhs=jnp.ones(2),
                preconditioner=_identity,
                emit=None,
                elapsed_s=lambda: 0.0,
                pc_form="right",
                restart=5,
                tol=1.0e-8,
                atol=0.0,
                precondition_side="right",
                factor_dtype=np.dtype(np.float64),
                progress_every=0,
                stagnation_abort=True,
                stagnation_min_iter=2,
                stagnation_window=1,
                stagnation_rel_improvement=1.0e-3,
                explicit_left_solver=lambda **_kwargs: (_ for _ in ()).throw(
                    AssertionError("wrong solver")
                ),
                gmres_solver=gmres_solver,
            ),
            x0=None,
            maxiter=10,
        )


def test_sparse_pc_gmres_completion_message_includes_pc_and_ksp_residuals() -> None:
    message = sparse_pc_gmres_completion_message(
        SparsePCGMRESCompletionMessageContext(
            elapsed_s=12.3456,
            iterations=7,
            matvecs=13,
            residual_norm=1.25e-4,
            target=1.0e-6,
            preconditioned_residual_norm=2.5e-3,
            history=(1.0, 3.0e-3),
        )
    )

    assert message == (
        "solve_v3_full_system_linear_gmres: sparse_pc_gmres complete "
        "elapsed_s=12.346 iters=7 matvecs=13 residual=1.250000e-04 "
        "target=1.000000e-06 preconditioned_residual=2.500000e-03 "
        "ksp_residual=3.000000e-03"
    )


def test_sparse_pc_gmres_completion_message_omits_nonfinite_optional_residuals() -> None:
    message = sparse_pc_gmres_completion_message(
        SparsePCGMRESCompletionMessageContext(
            elapsed_s=1.0,
            iterations=0,
            matvecs=0,
            residual_norm=float("inf"),
            target=2.0,
            preconditioned_residual_norm=float("nan"),
            history=(),
        )
    )

    assert message == (
        "solve_v3_full_system_linear_gmres: sparse_pc_gmres complete "
        "elapsed_s=1.000 iters=0 matvecs=0 residual=inf target=2.000000e+00"
    )


def test_emit_sparse_pc_gmres_completion_from_driver_state_uses_current_state() -> None:
    messages: list[tuple[int, str]] = []

    emit_sparse_pc_gmres_completion_from_driver_state(
        {
            "emit": lambda level, msg: messages.append((level, msg)),
            "sparse_timer": SimpleNamespace(elapsed_s=lambda: 2.5),
            "history": (1.0, 0.25),
            "mv_count": 6,
            "residual_norm_sparse_pc": 0.125,
            "target": 0.01,
            "rn_pc": 0.5,
        }
    )
    emit_sparse_pc_gmres_completion_from_driver_state(
        {
            "emit": None,
            "sparse_timer": SimpleNamespace(elapsed_s=lambda: 0.0),
            "history": (),
            "mv_count": 0,
            "residual_norm_sparse_pc": 0.0,
            "target": 1.0,
            "rn_pc": float("nan"),
        }
    )

    assert messages == [
        (
            0,
            "solve_v3_full_system_linear_gmres: sparse_pc_gmres complete "
            "elapsed_s=2.500 iters=2 matvecs=6 residual=1.250000e-01 "
            "target=1.000000e-02 preconditioned_residual=5.000000e-01 "
            "ksp_residual=2.500000e-01",
        )
    ]


def test_sparse_pc_gmres_final_payload_from_driver_state_expands_result_and_metadata() -> None:
    state = _DefaultSparsePCDriverState(
        {
            "op": SimpleNamespace(total_size=np.int64(4)),
            "x_np": np.asarray([1.0, 2.0]),
            "residual_norm_sparse_pc": 0.25,
            "history": (1.0, 0.25),
            "mv_count": np.int64(3),
            "pc_restart": np.int64(4),
            "pc_maxiter": np.int64(5),
            "sparse_pc_first_attempt_maxiter": np.int64(2),
            "sparse_pc_post_minres_steps": np.int64(0),
            "sparse_pc_post_minres_alphas": (),
            "sparse_pc_post_minres_alpha_clip": 4.0,
            "sparse_pc_post_minres_min_improvement": 0.1,
            "sparse_pc_post_minres_residual_before": None,
            "sparse_pc_post_minres_residual_after": None,
            "sparse_pc_post_minres_history": (),
            "sparse_pc_post_minres_error": None,
            "pc_shift": 0.0,
            "sparse_pc_factor_dtype_used": np.dtype(np.float64),
            "sparse_pc_factor_dtype_initial": np.dtype(np.float64),
            "sparse_pc_factor_dtype_retry": None,
            "factor_preflight_enabled": False,
            "factor_preflight_required": False,
            "factor_preflight_seed_enabled": False,
            "factor_preflight_seed_used": False,
            "factor_preflight_passed": None,
            "factor_preflight_error": None,
            "factor_preflight_residual_before": None,
            "factor_preflight_residual_after": None,
            "factor_preflight_improvement_ratio": None,
            "factor_preflight_target_ratio": None,
            "factor_preflight_max_target_ratio": 8.0,
            "factor_preflight_residual_diagnostics": {},
            "fortran_reduced_sparse_pc": False,
            "sparse_pc_preconditioner_operator": "full",
            "sparse_pc_factorization": "lu",
            "sparse_pc_default_factor_kind": "ilu",
            "sparse_pc_default_ilu_fill_factor": 6.0,
            "sparse_pc_default_ilu_drop_tol": 1.0e-5,
            "sparse_pc_default_pattern_color_batch": np.int64(8),
            "preconditioner_x": np.int64(1),
            "preconditioner_x_min_l": np.int64(0),
            "preconditioner_xi": np.int64(1),
            "preconditioner_species": np.int64(1),
            "sparse_pc_permc_spec": "COLAMD",
            "sparse_pc_default_permc_spec": "COLAMD",
            "sparse_pc_use_active_dof": True,
            "sparse_pc_linear_size": np.int64(2),
            "sparse_pc_fp_dense_velocity_block": None,
            "setup_s": 0.5,
            "solve_s": 1.5,
            "sparse_timer": SimpleNamespace(elapsed_s=lambda: 3.0),
            "summary": SimpleNamespace(
                nnz=np.int64(7),
                avg_row_nnz=1.75,
                max_row_nnz=np.int64(3),
            ),
            "sparse_pattern_scope": "active_dof",
            "pattern_build_s": 0.125,
            "pc_factor_s": 0.25,
            "factor_bundle_pc": SimpleNamespace(
                factor_s=None,
                factor_nbytes_estimate=None,
                factor_nnz_estimate=None,
            ),
            "_operator_bundle_pc": None,
            "target": 0.5,
            "atol": 0.5,
            "tol": 0.0,
            "rhs_norm": 1.0,
            "direct_tail_operator_bundle": None,
            "direct_tail_structured_max_nbytes": None,
            "direct_tail_true_window_specs": (),
            "direct_tail_true_active_block_species_count": None,
            "direct_tail_structured_pc_metadata": {},
            "direct_tail_support_mode_preflight_metadata": {},
            "direct_tail_true_coupled_coarse_metadata": {},
            "direct_tail_residual_window_coefficient_mode": "normal",
            "direct_tail_residual_window_combine_mode": "additive",
            "direct_tail_error": None,
            "direct_tail_structured_pc_requested": "auto",
            "direct_tail_structured_pc_reason": "none",
            "direct_tail_structured_pc_error": None,
            "direct_tail_support_mode_preflight_error": None,
        }
    )

    payload = sparse_pc_gmres_final_payload_from_driver_state(
        state,
        expand_reduced=lambda x: jnp.concatenate(
            [jnp.asarray([0.0], dtype=x.dtype), x]
        ),
    )

    assert isinstance(payload, SparsePCGMRESFinalPayload)
    np.testing.assert_allclose(np.asarray(payload.x), np.asarray([0.0, 1.0, 2.0]))
    assert float(payload.residual_norm) == pytest.approx(0.25)
    assert payload.metadata["solver_kind"] == "sparse_pc_gmres"
    assert payload.metadata["accepted_converged"] is True
    assert payload.metadata["sparse_pc_residual_ratio_to_target"] == pytest.approx(0.5)
    assert payload.metadata["sparse_pc_linear_size"] == 2


def test_explicit_sparse_operator_policy_and_messages_are_stable() -> None:
    policy = resolve_explicit_sparse_operator_build_policy(
        {
            "SFINCS_JAX_EXPLICIT_SPARSE_CSR_MAX_MB": "bad",
            "SFINCS_JAX_EXPLICIT_SPARSE_DROP_TOL": "2e-4",
        }
    )
    messages = explicit_sparse_pattern_progress_messages(
        solver_label="sparse_host",
        summary=SimpleNamespace(nnz=np.int64(9), avg_row_nnz=2.25, max_row_nnz=np.int64(4)),
    )

    assert isinstance(policy, ExplicitSparseOperatorBuildPolicy)
    assert policy.csr_max_mb == pytest.approx(512.0)
    assert policy.drop_tol == pytest.approx(2.0e-4)
    assert messages == (
        (
            1,
            "solve_v3_full_system_linear_gmres: sparse_host building conservative pattern",
        ),
        (
            1,
            "solve_v3_full_system_linear_gmres: sparse_host pattern "
            "nnz=9 avg_row_nnz=2.25 max_row_nnz=4",
        ),
    )


def test_build_explicit_sparse_operator_from_pattern_forwards_policy_and_reports_storage() -> None:
    calls: list[dict[str, object]] = []
    bundle = SimpleNamespace(
        metadata=SimpleNamespace(storage_kind="csr", reason="materialized"),
        matrix=scipy_sparse.eye(2, format="csr"),
    )

    def builder(matvec_np, **kwargs):
        calls.append({"matvec_np": matvec_np, **kwargs})
        return bundle

    result = build_explicit_sparse_operator_from_pattern(
        matvec_np=lambda x: np.asarray(x, dtype=np.float64),
        pattern={"row": (0,)},
        dtype=np.float64,
        backend="cpu",
        env={
            "SFINCS_JAX_EXPLICIT_SPARSE_CSR_MAX_MB": "64",
            "SFINCS_JAX_EXPLICIT_SPARSE_DROP_TOL": "1e-5",
        },
        build_operator_from_pattern=builder,
        allow_operator_only=False,
    )

    assert isinstance(result, ExplicitSparseOperatorBuildResult)
    assert result.operator_bundle is bundle
    assert result.policy.csr_max_mb == pytest.approx(64.0)
    assert result.policy.drop_tol == pytest.approx(1.0e-5)
    assert calls[0]["pattern"] == {"row": (0,)}
    assert calls[0]["backend"] == "cpu"
    assert calls[0]["csr_max_mb"] == pytest.approx(64.0)
    assert calls[0]["drop_tol"] == pytest.approx(1.0e-5)
    assert calls[0]["allow_operator_only"] is False
    assert result.messages == (
        (1, "explicit_sparse: storage=csr reason=materialized"),
    )


def test_validate_explicit_sparse_host_request_preserves_user_facing_errors() -> None:
    validate_explicit_sparse_host_request(
        solve_method_label="sparse_host",
        differentiable=False,
        rhs_mode=1,
        use_active_dof=False,
        path_description="host sparse LU path",
    )

    with pytest.raises(ValueError, match="non-differentiable host sparse LU path"):
        validate_explicit_sparse_host_request(
            solve_method_label="sparse_host",
            differentiable=True,
            rhs_mode=1,
            use_active_dof=False,
            path_description="host sparse LU path",
        )
    with pytest.raises(NotImplementedError, match="RHSMode=1"):
        validate_explicit_sparse_host_request(
            solve_method_label="sparse_lsmr",
            differentiable=False,
            rhs_mode=2,
            use_active_dof=False,
            path_description="host sparse minimum-norm path",
        )
    with pytest.raises(NotImplementedError, match="ACTIVE_DOF=0"):
        validate_explicit_sparse_host_request(
            solve_method_label="sparse_lsmr",
            differentiable=False,
            rhs_mode=1,
            use_active_dof=True,
            path_description="host sparse minimum-norm path",
        )


def test_sparse_minimum_norm_policy_parses_env_and_preserves_defaults() -> None:
    default_policy = resolve_sparse_minimum_norm_policy(
        {},
        solve_method_kind="sparse_lsmr",
        tol=1.0e-8,
        maxiter=None,
        emit_enabled=False,
    )

    assert isinstance(default_policy, SparseMinimumNormPolicy)
    assert default_policy.solver_name == "lsmr"
    assert default_policy.atol == pytest.approx(1.0e-8)
    assert default_policy.btol == pytest.approx(1.0e-8)
    assert default_policy.conlim == pytest.approx(1.0e8)
    assert default_policy.damp == pytest.approx(0.0)
    assert default_policy.maxiter == 1000
    assert default_policy.show is False
    assert default_policy.petsc_compat_requested is False

    parsed_policy = resolve_sparse_minimum_norm_policy(
        {
            "SFINCS_JAX_SPARSE_LSMR_ATOL": "2e-7",
            "SFINCS_JAX_SPARSE_LSMR_BTOL": "bad",
            "SFINCS_JAX_SPARSE_LSMR_CONLIM": "3e5",
            "SFINCS_JAX_SPARSE_LSMR_DAMP": "4e-3",
            "SFINCS_JAX_SPARSE_LSMR_MAXITER": "12",
            "SFINCS_JAX_SPARSE_LSMR_SHOW": "yes",
        },
        solve_method_kind="sparse_lsqr",
        tol=1.0e-6,
        maxiter=7,
        emit_enabled=True,
    )

    assert parsed_policy.solver_name == "lsqr"
    assert parsed_policy.atol == pytest.approx(2.0e-7)
    assert parsed_policy.btol == pytest.approx(1.0e-6)
    assert parsed_policy.conlim == pytest.approx(3.0e5)
    assert parsed_policy.damp == pytest.approx(4.0e-3)
    assert parsed_policy.maxiter == 12
    assert parsed_policy.show is True
    assert sparse_minimum_norm_start_message(parsed_policy) == (
        "solve_v3_full_system_linear_gmres: sparse_lsmr solve start "
        "solver=lsqr atol=2.0e-07 btol=1.0e-06 damp=4.0e-03 "
        "conlim=3.0e+05 maxiter=12"
    )


def test_sparse_minimum_norm_solve_payload_solves_tiny_identity_system() -> None:
    policy = resolve_sparse_minimum_norm_policy(
        {},
        solve_method_kind="petsc_compat",
        tol=1.0e-12,
        maxiter=20,
        emit_enabled=False,
    )

    payload = sparse_minimum_norm_solve_payload(
        matrix=scipy_sparse.eye(2, format="csr"),
        rhs=jnp.asarray([1.0, -2.0], dtype=jnp.float64),
        policy=policy,
        atol=1.0e-12,
        tol=1.0e-12,
        rhs_norm=float(np.linalg.norm([1.0, -2.0])),
        elapsed_s=lambda: 1.25,
    )

    assert isinstance(payload, SparseMinimumNormPayload)
    np.testing.assert_allclose(np.asarray(payload.x), np.asarray([1.0, -2.0]), atol=1.0e-10)
    assert float(payload.residual_norm) < 1.0e-10
    assert payload.metadata["solver_kind"] == "sparse_lsmr"
    assert payload.metadata["petsc_compat_requested"] is True
    assert payload.metadata["accepted_converged"] is True
    assert payload.metadata["acceptance_criterion"] == "true_residual"
    assert "accepted=True criterion=true_residual" in payload.completion_message


def test_sparse_minimum_norm_solve_from_pattern_materializes_and_emits_messages() -> None:
    messages: list[tuple[int, str]] = []
    calls: list[dict[str, object]] = []
    bundle = SimpleNamespace(
        metadata=SimpleNamespace(storage_kind="csr", reason="materialized"),
        matrix=scipy_sparse.eye(2, format="csr"),
    )

    def builder(matvec_np, **kwargs):
        calls.append({"matvec_np": matvec_np, **kwargs})
        return bundle

    payload = sparse_minimum_norm_solve_from_pattern(
        matvec_np=lambda x: np.asarray(x, dtype=np.float64),
        pattern={"rows": (0, 1)},
        summary=SimpleNamespace(nnz=2, avg_row_nnz=1.0, max_row_nnz=1),
        rhs=jnp.asarray([1.0, -2.0], dtype=jnp.float64),
        solve_method_kind="sparse_lsmr",
        tol=1.0e-12,
        atol=1.0e-12,
        maxiter=20,
        rhs_norm=float(np.linalg.norm([1.0, -2.0])),
        elapsed_s=lambda: 2.5,
        backend="cpu",
        env={"SFINCS_JAX_EXPLICIT_SPARSE_CSR_MAX_MB": "32"},
        emit=lambda level, message: messages.append((level, message)),
        build_operator_from_pattern=builder,
    )

    assert isinstance(payload, SparseMinimumNormPayload)
    np.testing.assert_allclose(np.asarray(payload.x), np.asarray([1.0, -2.0]), atol=1.0e-10)
    assert float(payload.residual_norm) < 1.0e-10
    assert calls[0]["pattern"] == {"rows": (0, 1)}
    assert calls[0]["backend"] == "cpu"
    assert calls[0]["csr_max_mb"] == pytest.approx(32.0)
    assert calls[0]["allow_operator_only"] is False
    assert messages[0] == (
        1,
        "solve_v3_full_system_linear_gmres: sparse_lsmr building conservative pattern",
    )
    assert messages[1] == (
        1,
        "solve_v3_full_system_linear_gmres: sparse_lsmr pattern "
        "nnz=2 avg_row_nnz=1 max_row_nnz=1",
    )
    assert messages[2] == (1, "explicit_sparse: storage=csr reason=materialized")
    assert messages[3][1].startswith("solve_v3_full_system_linear_gmres: sparse_lsmr solve start")
    assert messages[4][1].startswith("solve_v3_full_system_linear_gmres: sparse_lsmr complete")


def test_sparse_minimum_norm_solve_from_pattern_requires_materialized_matrix() -> None:
    bundle = SimpleNamespace(
        metadata=SimpleNamespace(storage_kind="operator_only", reason="too_large"),
        matrix=None,
    )

    with pytest.raises(RuntimeError, match="requires a materialized sparse matrix"):
        sparse_minimum_norm_solve_from_pattern(
            matvec_np=lambda x: np.asarray(x, dtype=np.float64),
            pattern={"rows": (0,)},
            summary=SimpleNamespace(nnz=1, avg_row_nnz=1.0, max_row_nnz=1),
            rhs=jnp.asarray([1.0], dtype=jnp.float64),
            solve_method_kind="sparse_lsmr",
            tol=1.0e-12,
            atol=1.0e-12,
            maxiter=10,
            rhs_norm=1.0,
            elapsed_s=lambda: 0.0,
            backend="cpu",
            env={},
            emit=None,
            build_operator_from_pattern=lambda *_args, **_kwargs: bundle,
        )


def test_sparse_host_direct_solve_payload_recomputes_true_residual_and_metadata() -> None:
    calls: list[dict[str, object]] = []

    def direct_solve_with_refinement(**kwargs):
        calls.append(kwargs)
        return np.asarray([3.0, -1.0]), 99.0

    payload = sparse_host_direct_solve_payload(
        factor_solve=lambda rhs: rhs,
        operator_matrix=scipy_sparse.eye(2, format="csr"),
        rhs=jnp.asarray([3.0, -1.0], dtype=jnp.float64),
        factor_dtype=np.dtype(np.float64),
        refine_steps=2,
        matvec=lambda x: jnp.asarray(x, dtype=jnp.float64),
        atol=1.0e-12,
        tol=1.0e-12,
        rhs_norm=float(np.linalg.norm([3.0, -1.0])),
        elapsed_s=lambda: 0.75,
        direct_solve_with_refinement=direct_solve_with_refinement,
    )

    assert isinstance(payload, SparseHostDirectPayload)
    assert calls[0]["refine_steps"] == 2
    np.testing.assert_allclose(np.asarray(payload.x), np.asarray([3.0, -1.0]))
    assert float(payload.residual_norm) == pytest.approx(0.0)
    assert payload.metadata == {
        "solver_kind": "sparse_host",
        "residual_kind": "true_residual",
        "accepted_converged": True,
        "acceptance_criterion": "true_residual",
    }
    assert payload.completion_message == (
        "solve_v3_full_system_linear_gmres: sparse_host complete "
        "elapsed_s=0.750 residual=0.000000e+00"
    )


def test_sparse_host_direct_solve_from_pattern_builds_factor_and_emits_messages() -> None:
    messages: list[tuple[int, str]] = []
    build_calls: list[dict[str, object]] = []
    direct_calls: list[dict[str, object]] = []
    operator_bundle = SimpleNamespace(matrix=scipy_sparse.diags([2.0, 4.0], format="csr"))
    factor_bundle = SimpleNamespace(solve=lambda rhs: rhs)

    def build_factor(**kwargs):
        build_calls.append(kwargs)
        return operator_bundle, factor_bundle

    def direct_solve_with_refinement(**kwargs):
        direct_calls.append(kwargs)
        return np.asarray([1.0, 2.0]), 9.0

    payload = sparse_host_direct_solve_from_pattern(
        matvec=lambda x: jnp.asarray([2.0 * x[0], 4.0 * x[1]], dtype=jnp.float64),
        pattern={"rows": (0, 1)},
        summary=SimpleNamespace(nnz=2, avg_row_nnz=1.0, max_row_nnz=1),
        n=2,
        dtype=jnp.float64,
        rhs=jnp.asarray([2.0, 8.0], dtype=jnp.float64),
        factor_dtype=np.dtype(np.float64),
        refine_steps=3,
        atol=1.0e-12,
        tol=1.0e-12,
        rhs_norm=float(np.linalg.norm([2.0, 8.0])),
        elapsed_s=lambda: 1.5,
        emit=lambda level, message: messages.append((level, message)),
        build_host_sparse_direct_factor_from_matvec=build_factor,
        direct_solve_with_refinement=direct_solve_with_refinement,
    )

    assert isinstance(payload, SparseHostDirectPayload)
    assert build_calls[0]["pattern"] == {"rows": (0, 1)}
    assert build_calls[0]["n"] == 2
    assert build_calls[0]["factor_dtype"] == np.dtype(np.float64)
    assert build_calls[0]["emit"] is not None
    assert direct_calls[0]["factor_solve"] is factor_bundle.solve
    assert direct_calls[0]["operator_matrix"] is operator_bundle.matrix
    assert direct_calls[0]["refine_steps"] == 3
    np.testing.assert_allclose(np.asarray(payload.x), np.asarray([1.0, 2.0]))
    assert float(payload.residual_norm) == pytest.approx(0.0)
    assert payload.metadata["solver_kind"] == "sparse_host"
    assert payload.metadata["accepted_converged"] is True
    assert messages == [
        (
            1,
            "solve_v3_full_system_linear_gmres: sparse_host building conservative pattern",
        ),
        (
            1,
            "solve_v3_full_system_linear_gmres: sparse_host pattern "
            "nnz=2 avg_row_nnz=1 max_row_nnz=1",
        ),
        (
            0,
            "solve_v3_full_system_linear_gmres: sparse_host complete "
            "elapsed_s=1.500 residual=0.000000e+00",
        ),
    ]


def test_solve_sparse_host_direct_from_available_factor_prefers_explicit_factor() -> None:
    calls: list[str] = []
    explicit_factor = SimpleNamespace(solve=lambda rhs: rhs)
    explicit_operator = SimpleNamespace(matrix=scipy_sparse.eye(2, format="csr"))

    def direct_solve(**kwargs):
        calls.append("direct")
        assert kwargs["factor_solve"] is explicit_factor.solve
        assert kwargs["operator_matrix"] is explicit_operator.matrix
        assert kwargs["refine_steps"] == 3
        return np.asarray([1.0, 2.0]), 0.125

    def ilu_solve(**_kwargs):
        calls.append("ilu")
        return np.asarray([9.0, 9.0]), 9.0

    payload = solve_sparse_host_direct_from_available_factor(
        explicit_sparse_factor=explicit_factor,
        explicit_sparse_operator=explicit_operator,
        ilu=object(),
        a_csr_full=object(),
        rhs=jnp.asarray([1.0, 2.0], dtype=jnp.float64),
        factor_dtype=np.dtype(np.float64),
        refine_steps=3,
        direct_solve_with_refinement=direct_solve,
        ilu_solve_with_refinement=ilu_solve,
    )

    assert isinstance(payload, SparseHostDirectFactorSolvePayload)
    assert calls == ["direct"]
    np.testing.assert_allclose(payload.x, np.asarray([1.0, 2.0]))
    assert payload.residual_norm == pytest.approx(0.125)
    assert payload.used_explicit_factor is True


def test_solve_sparse_host_direct_from_available_factor_uses_ilu_without_explicit_factor() -> None:
    calls: list[str] = []
    ilu = object()
    matrix = scipy_sparse.eye(2, format="csr")

    def direct_solve(**_kwargs):
        calls.append("direct")
        return np.asarray([9.0, 9.0]), 9.0

    def ilu_solve(**kwargs):
        calls.append("ilu")
        assert kwargs["ilu"] is ilu
        assert kwargs["a_csr_full"] is matrix
        assert kwargs["refine_steps"] == 1
        return np.asarray([-1.0, 3.0]), 0.25

    payload = solve_sparse_host_direct_from_available_factor(
        explicit_sparse_factor=None,
        explicit_sparse_operator=None,
        ilu=ilu,
        a_csr_full=matrix,
        rhs=jnp.asarray([-1.0, 3.0], dtype=jnp.float64),
        factor_dtype=np.dtype(np.float32),
        refine_steps=1,
        direct_solve_with_refinement=direct_solve,
        ilu_solve_with_refinement=ilu_solve,
    )

    assert calls == ["ilu"]
    np.testing.assert_allclose(payload.x, np.asarray([-1.0, 3.0]))
    assert payload.residual_norm == pytest.approx(0.25)
    assert payload.used_explicit_factor is False


def test_apply_sparse_host_direct_polish_skips_non_float32_or_converged_result() -> None:
    payload = apply_sparse_host_direct_polish_if_needed(
        x=np.asarray([1.0]),
        residual_norm=10.0,
        factor_dtype=np.dtype(np.float64),
        target=1.0,
        matvec=_identity,
        rhs=jnp.asarray([1.0], dtype=jnp.float64),
        ilu=object(),
        tol=1.0e-8,
        atol=1.0e-8,
        restart=80,
        maxiter=100,
        precondition_side="right",
        emit=lambda *_args: pytest.fail("unexpected emit"),
        polish_enabled=lambda **_kwargs: pytest.fail("unexpected policy"),
        parse_polish_gmres_config=lambda **_kwargs: pytest.fail("unexpected parse"),
        host_sparse_direct_polish=lambda **_kwargs: pytest.fail("unexpected polish"),
    )

    assert isinstance(payload, SparseHostDirectPolishPayload)
    assert payload.attempted is False
    assert payload.accepted is False
    assert payload.restart is None
    assert float(payload.residual_norm) == pytest.approx(10.0)


def test_apply_sparse_host_direct_polish_accepts_improved_float32_result() -> None:
    messages: list[tuple[int, str]] = []
    parse_calls: list[dict[str, object]] = []
    polish_calls: list[dict[str, object]] = []

    def parse_config(**kwargs):
        parse_calls.append(kwargs)
        return 17, 33

    def polish(**kwargs):
        polish_calls.append(kwargs)
        return np.asarray([0.5, -0.5]), 0.25

    payload = apply_sparse_host_direct_polish_if_needed(
        x=np.asarray([1.0, -1.0]),
        residual_norm=2.0,
        factor_dtype=np.dtype(np.float32),
        target=1.0,
        matvec=_identity,
        rhs=jnp.asarray([1.0, -1.0], dtype=jnp.float64),
        ilu=object(),
        tol=1.0e-8,
        atol=1.0e-8,
        restart=80,
        maxiter=100,
        precondition_side="left",
        emit=lambda level, message: messages.append((level, message)),
        polish_enabled=lambda **kwargs: kwargs["env_name"] == "SFINCS_JAX_RHSMODE1_SPARSE_DIRECT_POLISH",
        parse_polish_gmres_config=parse_config,
        host_sparse_direct_polish=polish,
    )

    assert payload.attempted is True
    assert payload.accepted is True
    assert payload.restart == 17
    assert payload.maxiter == 33
    np.testing.assert_allclose(np.asarray(payload.x), np.asarray([0.5, -0.5]))
    assert float(payload.residual_norm) == pytest.approx(0.25)
    assert parse_calls[0]["default_restart"] == 40
    assert parse_calls[0]["default_maxiter"] == 100
    assert polish_calls[0]["precondition_side"] == "left"
    assert messages == [
        (
            0,
            "solve_v3_full_system_linear_gmres: host sparse direct polish "
            "restart=17 maxiter=33",
        )
    ]


def test_apply_sparse_host_direct_polish_rejects_nonimproving_result() -> None:
    payload = apply_sparse_host_direct_polish_if_needed(
        x=np.asarray([2.0]),
        residual_norm=2.0,
        factor_dtype=np.dtype(np.float32),
        target=1.0,
        matvec=_identity,
        rhs=jnp.asarray([2.0], dtype=jnp.float64),
        ilu=object(),
        tol=1.0e-8,
        atol=1.0e-8,
        restart=20,
        maxiter=None,
        precondition_side="right",
        emit=None,
        polish_enabled=lambda **_kwargs: True,
        parse_polish_gmres_config=lambda **_kwargs: (5, 6),
        host_sparse_direct_polish=lambda **_kwargs: (np.asarray([0.0]), 3.0),
    )

    assert payload.attempted is True
    assert payload.accepted is False
    np.testing.assert_allclose(np.asarray(payload.x), np.asarray([2.0]))
    assert float(payload.residual_norm) == pytest.approx(2.0)


def test_sparse_host_direct_fallback_payload_polishes_and_recomputes_residual_vector() -> None:
    messages: list[tuple[int, str]] = []
    explicit_factor = SimpleNamespace(solve=lambda rhs: rhs)
    explicit_operator = SimpleNamespace(matrix=scipy_sparse.diags([2.0, 4.0], format="csr"))

    def direct_solve(**kwargs):
        assert kwargs["factor_solve"] is explicit_factor.solve
        assert kwargs["operator_matrix"] is explicit_operator.matrix
        return np.asarray([1.0, 1.0]), 8.0

    def ilu_solve(**_kwargs):
        pytest.fail("explicit factor should be preferred")

    def polish(**kwargs):
        assert kwargs["precondition_side"] == "left"
        return np.asarray([1.0, 2.0]), 0.125

    payload = sparse_host_direct_fallback_payload(
        explicit_sparse_factor=explicit_factor,
        explicit_sparse_operator=explicit_operator,
        ilu=object(),
        a_csr_full=object(),
        rhs=jnp.asarray([3.0, 9.0], dtype=jnp.float64),
        factor_dtype=np.dtype(np.float32),
        refine_steps=2,
        matvec=lambda x: jnp.asarray([2.0 * x[0], 4.0 * x[1]], dtype=jnp.float64),
        target=1.0,
        tol=1.0e-8,
        atol=1.0e-8,
        restart=80,
        maxiter=120,
        precondition_side="left",
        emit=lambda level, msg: messages.append((level, msg)),
        backend_name="cpu",
        polish_enabled=lambda **_kwargs: True,
        parse_polish_gmres_config=lambda **_kwargs: (11, 22),
        direct_solve_with_refinement=direct_solve,
        ilu_solve_with_refinement=ilu_solve,
        host_sparse_direct_polish=polish,
    )

    assert isinstance(payload, SparseHostDirectFallbackPayload)
    assert payload.used_explicit_factor is True
    assert payload.polish_attempted is True
    assert payload.polish_accepted is True
    assert payload.polish_restart == 11
    assert payload.polish_maxiter == 22
    np.testing.assert_allclose(np.asarray(payload.x), np.asarray([1.0, 2.0]))
    assert float(payload.residual_norm) == pytest.approx(0.125)
    np.testing.assert_allclose(np.asarray(payload.residual_vec), np.asarray([1.0, 1.0]))
    assert messages == [
        (
            0,
            "solve_v3_full_system_linear_gmres: host sparse LU direct fallback "
            "on backend=cpu",
        ),
        (
            0,
            "solve_v3_full_system_linear_gmres: host sparse direct polish "
            "restart=11 maxiter=22",
        )
    ]


def test_build_sparse_host_or_ilu_factor_prefers_explicit_sparse_when_allowed() -> None:
    calls: list[dict[str, object]] = []
    explicit_operator = SimpleNamespace(matrix="csr")
    explicit_factor = SimpleNamespace(factor="lu")

    def build_host(**kwargs):
        calls.append(kwargs)
        return explicit_operator, explicit_factor

    result = build_sparse_host_or_ilu_factor(
        SparseHostOrILUFactorBuildContext(
            matvec=_identity,
            n=3,
            dtype=np.float64,
            cache_key="cache",
            factor_dtype=np.dtype(np.float64),
            drop_tol=0.0,
            drop_rel=0.0,
            ilu_drop_tol=1.0e-6,
            fill_factor=10.0,
            build_dense_factors=False,
            build_jax_factors=False,
            store_dense=False,
            factorization="lu",
            emit=None,
            host_sparse_direct_wanted=True,
            explicit_sparse_allowed=True,
            explicit_sparse_pattern="pattern",
            build_host_sparse_direct_factor_from_matvec=build_host,
            build_sparse_ilu_from_matvec=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("ILU should not be built")
            ),
        )
    )

    assert result.used_explicit_sparse
    assert result.explicit_sparse_operator is explicit_operator
    assert result.explicit_sparse_factor is explicit_factor
    assert result.a_csr_full == "csr"
    assert result.ilu == "lu"
    assert calls[0]["pattern"] == "pattern"
    assert calls[0]["n"] == 3


def test_build_sparse_host_or_ilu_factor_uses_ilu_when_explicit_not_allowed() -> None:
    calls: list[dict[str, object]] = []

    def build_ilu(**kwargs):
        calls.append(kwargs)
        return "csr", "drop", "ilu", "dense", "l", "u", True

    result = build_sparse_host_or_ilu_factor(
        SparseHostOrILUFactorBuildContext(
            matvec=_identity,
            n=4,
            dtype=np.float64,
            cache_key="cache",
            factor_dtype=np.dtype(np.float32),
            drop_tol=1.0e-4,
            drop_rel=1.0e-5,
            ilu_drop_tol=1.0e-6,
            fill_factor=8.0,
            build_dense_factors=True,
            build_jax_factors=True,
            store_dense=True,
            factorization="ilu",
            emit=None,
            host_sparse_direct_wanted=True,
            explicit_sparse_allowed=False,
            build_host_sparse_direct_factor_from_matvec=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("explicit host factor should not be built")
            ),
            build_sparse_ilu_from_matvec=build_ilu,
        )
    )

    assert not result.used_explicit_sparse
    assert result.explicit_sparse_operator is None
    assert result.explicit_sparse_factor is None
    assert result.a_csr_full == "csr"
    assert result.a_csr_drop == "drop"
    assert result.ilu == "ilu"
    assert result.a_dense_cache == "dense"
    assert result.l_dense == "l"
    assert result.u_dense == "u"
    assert result.l_unit_diag
    assert calls[0]["cache_key"] == "cache"
    assert calls[0]["build_jax_factors"] is True
    assert calls[0]["factorization"] == "ilu"


def test_build_sparse_ilu_preconditioner_from_cache_uses_dense_triangular() -> None:
    cache = SimpleNamespace(
        perm_r=jnp.asarray([0, 1], dtype=jnp.int32),
        inv_perm_c=jnp.asarray([0, 1], dtype=jnp.int32),
        lower_idx=None,
        lower_val=None,
        lower_diag=None,
        upper_idx=None,
        upper_val=None,
        upper_diag=None,
    )
    lower = np.asarray([[1.0, 0.0], [2.0, 1.0]])
    upper = np.asarray([[3.0, 1.0], [0.0, 4.0]])

    result = build_sparse_ilu_preconditioner_from_cache(
        SparseILUPreconditionerBuildContext(
            cache_entry=cache,
            l_dense=lower,
            u_dense=upper,
            l_unit_diag=True,
        )
    )

    assert result.preconditioner is not None
    assert result.used_dense_triangular
    assert not result.used_padded_triangular
    rhs = np.asarray([7.0, 10.0])
    expected = np.linalg.solve(upper, np.linalg.solve(lower, rhs))
    np.testing.assert_allclose(np.asarray(result.preconditioner(jnp.asarray(rhs))), expected)


def test_build_sparse_ilu_preconditioner_from_cache_uses_padded_triangular() -> None:
    cache = SimpleNamespace(
        perm_r=jnp.asarray([0, 1], dtype=jnp.int32),
        inv_perm_c=jnp.asarray([0, 1], dtype=jnp.int32),
        lower_idx=jnp.asarray([[-1], [0]], dtype=jnp.int32),
        lower_val=jnp.asarray([[0.0], [2.0]], dtype=jnp.float64),
        lower_diag=jnp.ones(2, dtype=jnp.float64),
        upper_idx=jnp.asarray([[1], [-1]], dtype=jnp.int32),
        upper_val=jnp.asarray([[1.0], [0.0]], dtype=jnp.float64),
        upper_diag=jnp.asarray([3.0, 4.0], dtype=jnp.float64),
    )

    result = build_sparse_ilu_preconditioner_from_cache(
        SparseILUPreconditionerBuildContext(
            cache_entry=cache,
            l_dense=None,
            u_dense=None,
            l_unit_diag=True,
        )
    )

    assert result.preconditioner is not None
    assert not result.used_dense_triangular
    assert result.used_padded_triangular
    lower = np.asarray([[1.0, 0.0], [2.0, 1.0]])
    upper = np.asarray([[3.0, 1.0], [0.0, 4.0]])
    rhs = np.asarray([7.0, 10.0])
    expected = np.linalg.solve(upper, np.linalg.solve(lower, rhs))
    np.testing.assert_allclose(np.asarray(result.preconditioner(jnp.asarray(rhs))), expected)


def test_build_sparse_ilu_preconditioner_from_cache_reports_unavailable() -> None:
    result = build_sparse_ilu_preconditioner_from_cache(
        SparseILUPreconditionerBuildContext(
            cache_entry=None,
            l_dense=None,
            u_dense=None,
            l_unit_diag=False,
        )
    )

    assert result.preconditioner is None
    assert not result.used_dense_triangular
    assert not result.used_padded_triangular


def test_build_sparse_host_scipy_preconditioner_uses_explicit_matrix_matvec() -> None:
    factor = SimpleNamespace(solve=lambda rhs: 0.25 * rhs)
    matrix = np.asarray([[2.0, 0.0], [0.0, 3.0]])

    result = build_sparse_host_scipy_preconditioner(
        SparseHostScipyPreconditionerBuildContext(
            ilu=factor,
            a_csr_full=matrix,
            base_matvec=lambda v: 10.0 * v,
            sparse_use_matvec=True,
        )
    )

    np.testing.assert_allclose(
        np.asarray(result.preconditioner(jnp.asarray([4.0, 8.0]))),
        np.asarray([1.0, 2.0]),
    )
    np.testing.assert_allclose(
        np.asarray(result.matvec(jnp.asarray([5.0, 7.0]))),
        np.asarray([10.0, 21.0]),
    )


def test_build_sparse_host_scipy_preconditioner_can_reuse_base_matvec() -> None:
    factor = SimpleNamespace(solve=lambda rhs: rhs)

    result = build_sparse_host_scipy_preconditioner(
        SparseHostScipyPreconditionerBuildContext(
            ilu=factor,
            a_csr_full=np.eye(2),
            base_matvec=lambda v: 3.0 * v,
            sparse_use_matvec=False,
        )
    )

    np.testing.assert_allclose(
        np.asarray(result.matvec(jnp.asarray([2.0, 4.0]))),
        np.asarray([6.0, 12.0]),
    )


def test_build_sparse_host_scipy_preconditioner_raises_when_factor_missing() -> None:
    with pytest.raises(RuntimeError, match="missing"):
        build_sparse_host_scipy_preconditioner(
            SparseHostScipyPreconditionerBuildContext(
                ilu=None,
                a_csr_full=np.eye(2),
                base_matvec=_identity,
                sparse_use_matvec=True,
                unavailable_message="missing",
            )
        )


def test_sparse_pc_post_minres_accepts_improved_residual_and_recomputes_pc_norm() -> (
    None
):
    messages: list[str] = []
    times = iter((1.0, 1.4))

    def minres_correction(**_kwargs):
        return (
            jnp.asarray([0.5, 0.5]),
            jnp.asarray([0.1, 0.2]),
            (0.9, 0.25),
            (0.75,),
        )

    result = apply_sparse_pc_post_minres(
        context=SparsePCPostMinresContext(
            matvec=_identity,
            rhs=jnp.zeros(2),
            preconditioner=lambda v: 0.5 * v,
            emit=lambda _level, msg: messages.append(msg),
            elapsed_s=lambda: next(times),
            pc_form="explicit_left",
            steps=2,
            alpha_clip=10.0,
            min_improvement=0.0,
            minres_correction=minres_correction,
        ),
        x=np.zeros(2),
        residual_norm=1.0,
        preconditioned_residual_norm=float("nan"),
    )

    assert result.x.tolist() == [0.5, 0.5]
    assert result.residual_norm == pytest.approx(np.linalg.norm([0.1, 0.2]))
    assert result.preconditioned_residual_norm == pytest.approx(
        np.linalg.norm([-0.25, -0.25])
    )
    assert result.history == (0.9, 0.25)
    assert result.alphas == (0.75,)
    assert result.error is None
    assert result.solve_s == pytest.approx(0.4)
    assert any("post-minres improved residual" in msg for msg in messages)


def test_sparse_pc_post_minres_uses_custom_solver_label() -> None:
    messages: list[str] = []
    times = iter((1.0, 1.1))

    def minres_correction(**_kwargs):
        return (
            jnp.asarray([0.5, 0.5]),
            jnp.asarray([0.1, 0.2]),
            (0.25,),
            (0.75,),
        )

    apply_sparse_pc_post_minres(
        context=SparsePCPostMinresContext(
            matvec=_identity,
            rhs=jnp.zeros(2),
            preconditioner=_identity,
            emit=lambda _level, msg: messages.append(msg),
            elapsed_s=lambda: next(times),
            pc_form="none",
            steps=1,
            alpha_clip=10.0,
            min_improvement=0.0,
            minres_correction=minres_correction,
            solver_label="xblock_sparse_pc_gmres",
        ),
        x=np.zeros(2),
        residual_norm=1.0,
        preconditioned_residual_norm=1.0,
    )

    assert any(
        "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres post-minres improved"
        in msg
        for msg in messages
    )


def test_sparse_pc_post_minres_if_needed_preserves_state_when_disabled_or_converged() -> None:
    def minres_correction(**_kwargs):
        raise AssertionError("post-minres should not run")

    disabled = apply_sparse_pc_post_minres_if_needed(
        SparsePCPostMinresUpdateContext(
            matvec=_identity,
            rhs=jnp.zeros(2),
            preconditioner=_identity,
            emit=None,
            elapsed_s=lambda: 0.0,
            pc_form="right",
            steps=0,
            alpha_clip=10.0,
            min_improvement=0.0,
            minres_correction=minres_correction,
            x=np.asarray([1.0, 2.0]),
            residual_norm=2.0,
            preconditioned_residual_norm=1.0,
            solve_s=3.0,
            target=1.0,
        )
    )
    converged = apply_sparse_pc_post_minres_if_needed(
        SparsePCPostMinresUpdateContext(
            matvec=_identity,
            rhs=jnp.zeros(2),
            preconditioner=_identity,
            emit=None,
            elapsed_s=lambda: 0.0,
            pc_form="right",
            steps=2,
            alpha_clip=10.0,
            min_improvement=0.0,
            minres_correction=minres_correction,
            x=np.asarray([1.0, 2.0]),
            residual_norm=0.5,
            preconditioned_residual_norm=0.25,
            solve_s=3.0,
            target=1.0,
        )
    )

    np.testing.assert_allclose(disabled.x, np.asarray([1.0, 2.0]))
    assert disabled.residual_norm == 2.0
    assert disabled.preconditioned_residual_norm == 1.0
    assert disabled.history == ()
    assert disabled.alphas == ()
    assert disabled.residual_before is None
    assert disabled.residual_after is None
    assert disabled.error is None
    assert disabled.solve_s == 3.0
    assert converged.residual_norm == 0.5
    assert converged.solve_s == 3.0


def test_xblock_subspace_correction_accepts_improved_residual() -> None:
    messages: list[str] = []
    times = iter((2.0, 2.25))

    def direction_builder(residual_vec):
        return (("fsavg", residual_vec),)

    def correction(**kwargs):
        assert kwargs["steps"] == 2
        assert kwargs["max_directions"] == 4
        assert kwargs["cached_labels"] == ("qi",)
        return (
            jnp.asarray([0.5, 0.5]),
            jnp.asarray([0.1, 0.2]),
            (0.5, 0.25),
            (1, 2),
            ("fsavg", "angular"),
        )

    result = apply_xblock_subspace_correction_if_needed(
        XBlockSubspaceCorrectionContext(
            matvec=_identity,
            rhs=jnp.zeros(2),
            x=np.zeros(2),
            residual_norm=1.0,
            target=1.0e-6,
            direction_builder=direction_builder,
            steps=2,
            max_directions=4,
            alpha_clip=10.0,
            rcond=1.0e-12,
            min_improvement=0.0,
            emit=lambda _level, msg: messages.append(msg),
            elapsed_s=lambda: next(times),
            correction=correction,
            correction_kwargs={"cached_labels": ("qi",)},
            correction_label="post-residual-equation",
            diagnostic_suffix=" cached_qi=1",
        )
    )

    assert result.x.tolist() == [0.5, 0.5]
    assert result.residual_norm == pytest.approx(np.linalg.norm([0.1, 0.2]))
    assert result.history == (0.5, 0.25)
    assert result.direction_counts == (1, 2)
    assert result.direction_names == ("fsavg", "angular")
    assert result.residual_before == 1.0
    assert result.solve_s == pytest.approx(0.25)
    assert any(
        "xblock_sparse_pc_gmres post-residual-equation improved" in msg
        and "cached_qi=1" in msg
        for msg in messages
    )


def test_xblock_subspace_correction_rejects_nonimproving_residual() -> None:
    messages: list[str] = []
    times = iter((2.0, 2.25))

    def correction(**_kwargs):
        return (
            jnp.asarray([2.0, 2.0]),
            jnp.asarray([2.0, 0.0]),
            (2.0,),
            (1,),
            ("fsavg",),
        )

    result = apply_xblock_subspace_correction_if_needed(
        XBlockSubspaceCorrectionContext(
            matvec=_identity,
            rhs=jnp.zeros(2),
            x=np.asarray([1.0, 1.0]),
            residual_norm=1.0,
            target=1.0e-6,
            direction_builder=lambda residual_vec: (("fsavg", residual_vec),),
            steps=1,
            max_directions=4,
            alpha_clip=10.0,
            rcond=1.0e-12,
            min_improvement=0.0,
            emit=lambda _level, msg: messages.append(msg),
            elapsed_s=lambda: next(times),
            correction=correction,
        )
    )

    assert result.x.tolist() == [1.0, 1.0]
    assert result.residual_norm == 1.0
    assert result.residual_after == 2.0
    assert any("xblock_sparse_pc_gmres post-coarse rejected" in msg for msg in messages)


def test_sparse_pc_post_minres_from_driver_state_updates_solve_state() -> None:
    times = iter((4.0, 4.6))

    def minres_correction(**_kwargs):
        return (
            jnp.asarray([0.25, 0.75]),
            jnp.asarray([0.1, 0.0]),
            (1.5, 0.1),
            (0.5,),
        )

    state = {
        "_mv_true": _identity,
        "sparse_pc_rhs": jnp.zeros(2),
        "_precond_sparse": lambda v: 2.0 * v,
        "emit": None,
        "sparse_timer": SimpleNamespace(elapsed_s=lambda: next(times)),
        "pc_form": "explicit_left",
        "sparse_pc_post_minres_steps": 2,
        "sparse_pc_post_minres_alpha_clip": 10.0,
        "sparse_pc_post_minres_min_improvement": 0.0,
        "x_np": np.zeros(2),
        "residual_norm_sparse_pc": 1.0,
        "rn_pc": float("nan"),
        "solve_s": 7.0,
        "target": 0.1,
    }

    result = apply_sparse_pc_post_minres_from_driver_state(
        state,
        minres_correction=minres_correction,
    )

    np.testing.assert_allclose(result.x, np.asarray([0.25, 0.75]))
    assert result.residual_norm == pytest.approx(0.1)
    assert result.preconditioned_residual_norm == pytest.approx(
        np.linalg.norm([-0.5, -1.5])
    )
    assert result.history == (1.5, 0.1)
    assert result.alphas == (0.5,)
    assert result.residual_before == 1.0
    assert result.residual_after == pytest.approx(0.1)
    assert result.error is None
    assert result.solve_s == pytest.approx(7.6)


def test_finalize_sparse_pc_gmres_from_driver_state_applies_polish_and_payload() -> None:
    messages: list[tuple[int, str]] = []

    def minres_correction(**_kwargs):
        return (
            jnp.asarray([3.0, 4.0]),
            jnp.asarray([0.25, 0.0]),
            (1.0, 0.25),
            (0.5,),
        )

    state = _DefaultSparsePCDriverState(
        {
            "op": SimpleNamespace(total_size=np.int64(3)),
            "_mv_true": _identity,
            "sparse_pc_rhs": jnp.zeros(2),
            "_precond_sparse": _identity,
            "emit": lambda level, msg: messages.append((level, msg)),
            "sparse_timer": SimpleNamespace(elapsed_s=lambda: 2.5),
            "pc_form": "right",
            "x_np": np.asarray([1.0, 2.0]),
            "residual_norm_sparse_pc": 1.0,
            "rn_pc": 0.5,
            "solve_s": 7.0,
            "target": 0.5,
            "atol": 0.5,
            "tol": 0.0,
            "rhs_norm": 1.0,
            "history": (1.0,),
            "mv_count": np.int64(3),
            "pc_restart": np.int64(4),
            "pc_maxiter": np.int64(5),
            "sparse_pc_first_attempt_maxiter": np.int64(2),
            "sparse_pc_post_minres_steps": np.int64(2),
            "sparse_pc_post_minres_alpha_clip": 4.0,
            "sparse_pc_post_minres_min_improvement": 0.0,
            "sparse_pc_post_minres_alphas": (),
            "sparse_pc_post_minres_residual_before": None,
            "sparse_pc_post_minres_residual_after": None,
            "sparse_pc_post_minres_history": (),
            "sparse_pc_post_minres_error": None,
            "pc_shift": 0.0,
            "sparse_pc_factor_dtype_used": np.dtype(np.float64),
            "sparse_pc_factor_dtype_initial": np.dtype(np.float64),
            "sparse_pc_factor_dtype_retry": None,
            "factor_preflight_enabled": False,
            "factor_preflight_required": False,
            "factor_preflight_seed_enabled": False,
            "factor_preflight_seed_used": False,
            "factor_preflight_passed": None,
            "factor_preflight_error": None,
            "factor_preflight_residual_before": None,
            "factor_preflight_residual_after": None,
            "factor_preflight_improvement_ratio": None,
            "factor_preflight_target_ratio": None,
            "factor_preflight_max_target_ratio": 8.0,
            "factor_preflight_residual_diagnostics": {},
            "fortran_reduced_sparse_pc": False,
            "sparse_pc_preconditioner_operator": "full",
            "sparse_pc_factorization": "lu",
            "sparse_pc_default_factor_kind": "ilu",
            "sparse_pc_default_ilu_fill_factor": 6.0,
            "sparse_pc_default_ilu_drop_tol": 1.0e-5,
            "sparse_pc_default_pattern_color_batch": np.int64(8),
            "preconditioner_x": np.int64(1),
            "preconditioner_x_min_l": np.int64(0),
            "preconditioner_xi": np.int64(1),
            "preconditioner_species": np.int64(1),
            "sparse_pc_permc_spec": "COLAMD",
            "sparse_pc_default_permc_spec": "COLAMD",
            "sparse_pc_use_active_dof": True,
            "sparse_pc_linear_size": np.int64(2),
            "sparse_pc_fp_dense_velocity_block": None,
            "setup_s": 0.5,
            "summary": SimpleNamespace(
                nnz=np.int64(7),
                avg_row_nnz=1.75,
                max_row_nnz=np.int64(3),
            ),
            "sparse_pattern_scope": "active_dof",
            "pattern_build_s": 0.125,
            "pc_factor_s": 0.25,
            "factor_bundle_pc": SimpleNamespace(
                factor_s=None,
                factor_nbytes_estimate=None,
                factor_nnz_estimate=None,
            ),
            "_operator_bundle_pc": None,
            "direct_tail_operator_bundle": None,
            "direct_tail_structured_max_nbytes": None,
            "direct_tail_true_active_block_species_count": None,
            "direct_tail_true_window_specs": (),
        }
    )

    payload = finalize_sparse_pc_gmres_from_driver_state(
        state,
        minres_correction=minres_correction,
        expand_reduced=lambda x: jnp.concatenate(
            [jnp.asarray([0.0], dtype=x.dtype), x]
        ),
    )

    assert isinstance(payload, SparsePCGMRESFinalPayload)
    np.testing.assert_allclose(np.asarray(payload.x), np.asarray([0.0, 3.0, 4.0]))
    assert float(payload.residual_norm) == pytest.approx(0.25)
    assert payload.metadata["accepted_converged"] is True
    assert payload.metadata["sparse_pc_post_minres_steps_accepted"] == 1
    assert payload.metadata["sparse_pc_post_minres_residual_after"] == pytest.approx(
        0.25
    )
    assert state["x_np"].tolist() == [1.0, 2.0]
    assert any("post-minres improved" in message for _, message in messages)
    assert any("sparse_pc_gmres complete" in message for _, message in messages)


def test_finalize_sparse_pc_gmres_with_dtype_retry_updates_copied_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}
    state = {
        "pc_factor_s": 2.0,
        "x_np": np.asarray([1.0, 2.0]),
    }

    def fake_retry(arg_state, **kwargs):
        calls["retry_state"] = arg_state
        calls["retry_kwargs"] = kwargs
        return sparse_pc_module.SparsePCFactorDtypeRetryResult(
            retried=True,
            factor_dtype_used=np.dtype(np.float64),
            factor_dtype_retry="float64",
            operator_bundle="operator64",
            factor_bundle="factor64",
            factor_s_increment=0.75,
            setup_s=4.0,
            x=np.asarray([3.0, 4.0]),
            residual_norm=0.25,
            preconditioned_residual_norm=0.125,
            history=(1.0, 0.25),
            solve_s=5.0,
        )

    def fake_finalize(arg_state, **kwargs):
        calls["final_state"] = arg_state
        calls["final_kwargs"] = kwargs
        return SparsePCGMRESFinalPayload(
            x=jnp.asarray([0.0, 3.0, 4.0]),
            residual_norm=jnp.asarray(0.25),
            metadata={"accepted_converged": True},
        )

    monkeypatch.setattr(
        sparse_pc_module,
        "retry_sparse_pc_factor_dtype_from_driver_state",
        fake_retry,
    )
    monkeypatch.setattr(
        sparse_pc_module,
        "finalize_sparse_pc_gmres_from_driver_state",
        fake_finalize,
    )

    payload = finalize_sparse_pc_gmres_with_dtype_retry_from_driver_state(
        state,
        build_host_sparse_direct_factor_from_matvec=lambda **_kwargs: None,
        run_sparse_pc_gmres_once_callback=lambda *_args, **_kwargs: None,
        minres_correction=lambda **_kwargs: None,
        expand_reduced=lambda x: x,
    )

    assert isinstance(payload, SparsePCGMRESFinalPayload)
    assert calls["retry_state"] is state
    assert calls["final_state"] is not state
    final_state = calls["final_state"]
    assert final_state["sparse_pc_factor_dtype_used"] == np.dtype(np.float64)
    assert final_state["sparse_pc_factor_dtype_retry"] == "float64"
    assert final_state["_operator_bundle_pc"] == "operator64"
    assert final_state["factor_bundle_pc"] == "factor64"
    assert final_state["pc_factor_s"] == pytest.approx(2.75)
    assert final_state["setup_s"] == pytest.approx(4.0)
    np.testing.assert_allclose(final_state["x_np"], np.asarray([3.0, 4.0]))
    assert final_state["residual_norm_sparse_pc"] == pytest.approx(0.25)
    assert final_state["rn_pc"] == pytest.approx(0.125)
    assert final_state["history"] == (1.0, 0.25)
    assert final_state["solve_s"] == pytest.approx(5.0)
    assert state["x_np"].tolist() == [1.0, 2.0]


def test_xblock_sparse_pc_final_metadata_from_driver_state_merges_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}
    state = {"token": object()}

    def fake_result_metadata(arg_state, *, full_size):
        calls["result_state"] = arg_state
        calls["full_size"] = full_size
        return {"core": 1, "shared": "core"}

    def fake_correction_metadata(arg_state):
        calls["correction_state"] = arg_state
        return {"correction": 2, "shared": "correction"}

    monkeypatch.setattr(
        sparse_pc_module,
        "xblock_sparse_pc_result_diagnostics_from_driver_state",
        fake_result_metadata,
    )
    monkeypatch.setattr(
        sparse_pc_module,
        "build_rhs1_xblock_correction_metadata_from_driver_state",
        fake_correction_metadata,
    )

    metadata = xblock_sparse_pc_final_metadata_from_driver_state(
        state,
        full_size=123,
    )

    assert calls == {
        "result_state": state,
        "full_size": 123,
        "correction_state": state,
    }
    assert metadata == {"core": 1, "correction": 2, "shared": "correction"}


def test_xblock_sparse_pc_final_payload_from_driver_state_sets_gate_and_expands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    def fake_result_metadata(arg_state, *, full_size):
        calls["accepted"] = arg_state["accepted_converged_xblock"]
        calls["full_size"] = full_size
        return {"core": 1}

    def fake_correction_metadata(_arg_state):
        return {"correction": 2}

    monkeypatch.setattr(
        sparse_pc_module,
        "xblock_sparse_pc_result_diagnostics_from_driver_state",
        fake_result_metadata,
    )
    monkeypatch.setattr(
        sparse_pc_module,
        "build_rhs1_xblock_correction_metadata_from_driver_state",
        fake_correction_metadata,
    )

    payload = xblock_sparse_pc_final_payload_from_driver_state(
        {
            "op": SimpleNamespace(total_size=7),
            "x_np": np.asarray([3.0, 4.0]),
            "residual_norm_xblock_pc": 0.25,
            "target_xblock": 0.5,
        },
        expand_reduced=lambda x: jnp.concatenate(
            [jnp.asarray([0.0], dtype=x.dtype), x]
        ),
    )

    assert isinstance(payload, SparsePCGMRESFinalPayload)
    np.testing.assert_allclose(np.asarray(payload.x), np.asarray([0.0, 3.0, 4.0]))
    assert float(payload.residual_norm) == pytest.approx(0.25)
    assert calls == {"accepted": True, "full_size": 7}
    assert payload.metadata == {"core": 1, "correction": 2}
