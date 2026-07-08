"""Diagnostics metadata builders for profile-response sparse-PC solves."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


def record_structured_fblock_preconditioner_metadata(
    *,
    target: dict[str, object],
    preconditioner: object,
) -> None:
    """Append structured f-block preconditioner diagnostics when available."""

    metadata = getattr(preconditioner, "_sfincs_jax_structured_fblock_metadata", None)
    if not isinstance(metadata, dict):
        return
    assembly = metadata.get("assembly", {})
    if not isinstance(assembly, dict):
        assembly = {}
    target.update(
        {
            "structured_fblock_preconditioner_enabled": True,
            "structured_fblock_preconditioner_selected": bool(
                metadata.get("selected", False)
            ),
            "structured_fblock_preconditioner_reason": str(metadata.get("reason", "")),
            "structured_fblock_preconditioner_nnz_blocks": int(
                assembly.get("nnz_blocks", 0) or 0
            ),
            "structured_fblock_preconditioner_data_nbytes": int(
                assembly.get("data_nbytes", 0) or 0
            ),
            "structured_fblock_preconditioner_metadata": metadata,
        }
    )


@dataclass(frozen=True)
class SparseRescueTailMetadataContext:
    """Explicit inputs for final RHSMode=1 sparse-rescue tail metadata."""

    sparse_xblock_rescue_active: object
    sparse_xblock_rescue_attempted: object
    sparse_xblock_rescue_built: object
    sparse_xblock_rescue_error: object
    sparse_xblock_rescue_reason: object
    sparse_xblock_rescue_assembled_host_fp: object
    sparse_xblock_rescue_preconditioner_xi: object
    sparse_xblock_rescue_seed_residual: object
    sparse_xblock_rescue_seed_improvement_ratio: object
    sparse_xblock_rescue_seed_accept_ratio: object
    sparse_xblock_rescue_seed_refine_steps: object
    sparse_xblock_rescue_seed_refines_performed: object
    sparse_xblock_rescue_candidate_residual: object
    sparse_xblock_rescue_candidate_accepted: object


def sparse_xblock_rescue_metadata(scope: Mapping[str, object]) -> dict[str, object]:
    """Return stable diagnostics for the sparse x-block rescue tail."""

    return {
        "sparse_xblock_rescue_active": bool(scope["sparse_xblock_rescue_active"]),
        "sparse_xblock_rescue_attempted": bool(scope["sparse_xblock_rescue_attempted"]),
        "sparse_xblock_rescue_built": bool(scope["sparse_xblock_rescue_built"]),
        "sparse_xblock_rescue_error": scope["sparse_xblock_rescue_error"],
        "sparse_xblock_rescue_reason": str(scope["sparse_xblock_rescue_reason"]),
        "sparse_xblock_rescue_assembled_host_fp": bool(
            scope["sparse_xblock_rescue_assembled_host_fp"]
        ),
        "sparse_xblock_rescue_preconditioner_xi": scope[
            "sparse_xblock_rescue_preconditioner_xi"
        ],
        "sparse_xblock_rescue_seed_residual": scope[
            "sparse_xblock_rescue_seed_residual"
        ],
        "sparse_xblock_rescue_seed_improvement_ratio": scope[
            "sparse_xblock_rescue_seed_improvement_ratio"
        ],
        "sparse_xblock_rescue_seed_accept_ratio": scope[
            "sparse_xblock_rescue_seed_accept_ratio"
        ],
        "sparse_xblock_rescue_seed_refine_steps": scope[
            "sparse_xblock_rescue_seed_refine_steps"
        ],
        "sparse_xblock_rescue_seed_refines_performed": scope[
            "sparse_xblock_rescue_seed_refines_performed"
        ],
        "sparse_xblock_rescue_candidate_residual": scope[
            "sparse_xblock_rescue_candidate_residual"
        ],
        "sparse_xblock_rescue_candidate_accepted": bool(
            scope["sparse_xblock_rescue_candidate_accepted"]
        ),
    }


def sparse_rescue_tail_metadata_from_context(
    context: SparseRescueTailMetadataContext,
) -> dict[str, object]:
    """Return the combined sparse-rescue tail diagnostics for final metadata."""

    return sparse_rescue_tail_metadata(context.__dict__)


def sparse_rescue_tail_metadata(scope: Mapping[str, object]) -> dict[str, object]:
    """Return the combined sparse-rescue tail diagnostics for final metadata."""

    return {
        **sparse_xblock_rescue_metadata(scope),
    }


_DIRECT_TAIL_BOOL_SUFFIXES: tuple[str, ...] = ()
_DIRECT_TAIL_INT_SUFFIXES: tuple[str, ...] = ()
_DIRECT_TAIL_FLOAT_SUFFIXES: tuple[str, ...] = ()
_DIRECT_TAIL_STR_SUFFIXES: tuple[str, ...] = ()
_DIRECT_TAIL_OBJECT_SUFFIXES: tuple[str, ...] = ()

_DIRECT_TAIL_SUFFIXES = (
    *_DIRECT_TAIL_BOOL_SUFFIXES,
    *_DIRECT_TAIL_INT_SUFFIXES,
    *_DIRECT_TAIL_FLOAT_SUFFIXES,
    *_DIRECT_TAIL_STR_SUFFIXES,
    *_DIRECT_TAIL_OBJECT_SUFFIXES,
)


@dataclass(frozen=True)
class SparsePCDirectTailMetadataContext:
    """Explicit direct-tail diagnostics consumed by sparse-PC metadata."""

    structured_pc_preflight_required: object
    structured_pc_preflight_required_min_size: object
    suffix_values: Mapping[str, object]
    operator_bundle: object
    structured_max_nbytes: object
    enabled: object
    direct_reduced_pmat_requested: object
    built: object
    error: object
    structured_pc_requested: object
    structured_pc_required: object
    structured_pc_selected: object
    structured_pc_reason: object
    structured_pc_error: object
    structured_pc_max_mb_auto: object
    structured_pc_metadata: object


def _direct_tail_suffix_values_from_state(
    state: Mapping[str, object],
) -> dict[str, object]:
    return {
        suffix: state[f"direct_tail_{suffix}"]
        for suffix in _DIRECT_TAIL_SUFFIXES
        if f"direct_tail_{suffix}" in state
    }


def _copy_direct_tail_suffix_values(
    metadata: dict[str, object],
    values: Mapping[str, object],
    suffixes: tuple[str, ...],
    coerce: object,
) -> None:
    for suffix in suffixes:
        metadata[f"sparse_pc_direct_tail_{suffix}"] = coerce(values[suffix])


def sparse_pc_direct_tail_result_metadata_from_context(
    context: SparsePCDirectTailMetadataContext,
) -> dict[str, object]:
    """Return direct-tail sparse-PC diagnostics for RHSMode=1 solve metadata.

    The direct-tail production path has many setup knobs and admission results.
    The metadata keys intentionally mirror the driver variable names so reports
    generated before the refactor remain byte-for-byte stable at the key level.
    """

    direct_tail_operator_bundle = context.operator_bundle
    direct_tail_structured_max_nbytes = context.structured_max_nbytes
    operator_metadata = (
        None
        if direct_tail_operator_bundle is None
        else direct_tail_operator_bundle.metadata
    )

    metadata: dict[str, object] = {
        "sparse_pc_direct_tail_structured_pc_preflight_required": bool(
            context.structured_pc_preflight_required
        ),
        "sparse_pc_direct_tail_structured_pc_preflight_required_min_size": int(
            context.structured_pc_preflight_required_min_size
        ),
    }

    _copy_direct_tail_suffix_values(
        metadata,
        context.suffix_values,
        _DIRECT_TAIL_BOOL_SUFFIXES,
        bool,
    )
    _copy_direct_tail_suffix_values(
        metadata,
        context.suffix_values,
        _DIRECT_TAIL_INT_SUFFIXES,
        int,
    )
    _copy_direct_tail_suffix_values(
        metadata,
        context.suffix_values,
        _DIRECT_TAIL_FLOAT_SUFFIXES,
        float,
    )
    _copy_direct_tail_suffix_values(
        metadata,
        context.suffix_values,
        _DIRECT_TAIL_STR_SUFFIXES,
        str,
    )
    _copy_direct_tail_suffix_values(
        metadata,
        context.suffix_values,
        _DIRECT_TAIL_OBJECT_SUFFIXES,
        lambda value: value,
    )

    metadata.update(
        {
            "sparse_pc_fortran_reduced_direct_tail_enabled": bool(
                context.enabled
            ),
            "sparse_pc_fortran_reduced_direct_pmat_requested": bool(
                context.direct_reduced_pmat_requested
            ),
            "sparse_pc_fortran_reduced_direct_tail_built": bool(context.built),
            "sparse_pc_fortran_reduced_direct_tail_error": context.error,
            "sparse_pc_fortran_reduced_direct_tail_operator_reason": (
                None if operator_metadata is None else str(operator_metadata.reason)
            ),
            "sparse_pc_fortran_reduced_direct_tail_nnz": (
                None if operator_metadata is None else operator_metadata.nnz_estimate
            ),
            "sparse_pc_fortran_reduced_direct_tail_csr_nbytes_estimate": (
                None
                if operator_metadata is None
                else int(operator_metadata.csr_nbytes_estimate)
            ),
            "sparse_pc_fortran_reduced_direct_tail_structured_pc_requested": (
                context.structured_pc_requested
            ),
            "sparse_pc_fortran_reduced_direct_tail_structured_pc_required": bool(
                context.structured_pc_required
            ),
            "sparse_pc_fortran_reduced_direct_tail_structured_pc_selected": bool(
                context.structured_pc_selected
            ),
            "sparse_pc_fortran_reduced_direct_tail_structured_pc_reason": (
                context.structured_pc_reason
            ),
            "sparse_pc_fortran_reduced_direct_tail_structured_pc_error": (
                context.structured_pc_error
            ),
            "sparse_pc_fortran_reduced_direct_tail_structured_pc_max_mb": (
                None
                if direct_tail_structured_max_nbytes is None
                else float(direct_tail_structured_max_nbytes) / (1024.0 * 1024.0)
            ),
            "sparse_pc_fortran_reduced_direct_tail_structured_pc_max_mb_auto": bool(
                context.structured_pc_max_mb_auto
            ),
            "sparse_pc_fortran_reduced_direct_tail_structured_pc_metadata": (
                context.structured_pc_metadata
            ),
        }
    )
    return metadata


def sparse_pc_direct_tail_result_metadata(
    state: Mapping[str, object],
) -> dict[str, object]:
    """Return direct-tail sparse-PC diagnostics from solve-state keys."""

    return sparse_pc_direct_tail_result_metadata_from_context(
        SparsePCDirectTailMetadataContext(
            structured_pc_preflight_required=state[
                "structured_pc_preflight_required"
            ],
            structured_pc_preflight_required_min_size=state[
                "structured_pc_preflight_required_min_size"
            ],
            suffix_values=_direct_tail_suffix_values_from_state(state),
            operator_bundle=state["direct_tail_operator_bundle"],
            structured_max_nbytes=state["direct_tail_structured_max_nbytes"],
            enabled=state["direct_tail_enabled"],
            direct_reduced_pmat_requested=state[
                "direct_tail_direct_reduced_pmat_requested"
            ],
            built=state["direct_tail_built"],
            error=state["direct_tail_error"],
            structured_pc_requested=state["direct_tail_structured_pc_requested"],
            structured_pc_required=state["direct_tail_structured_pc_required"],
            structured_pc_selected=state["direct_tail_structured_pc_selected"],
            structured_pc_reason=state["direct_tail_structured_pc_reason"],
            structured_pc_error=state["direct_tail_structured_pc_error"],
            structured_pc_max_mb_auto=state[
                "direct_tail_structured_pc_max_mb_auto"
            ],
            structured_pc_metadata=state["direct_tail_structured_pc_metadata"],
        )
    )


def _dtype_name(value: object) -> str:
    return str(getattr(value, "name", value))


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


@dataclass(frozen=True)
class SparsePCFactorPreflightMetadataContext:
    """Explicit factor-preflight diagnostics consumed by sparse-PC metadata."""

    enabled: object
    required: object
    seed_enabled: object
    seed_used: object
    passed: object
    error: object
    residual_before: object
    residual_after: object
    improvement_ratio: object
    target_ratio: object
    max_target_ratio: object
    residual_diagnostics: object


def sparse_pc_factor_preflight_result_metadata_from_context(
    context: SparsePCFactorPreflightMetadataContext,
) -> dict[str, object]:
    """Return sparse-PC factor-preflight diagnostics for final metadata."""

    return {
        "sparse_pc_factor_preflight_enabled": bool(context.enabled),
        "sparse_pc_factor_preflight_required": bool(context.required),
        "sparse_pc_factor_preflight_seed_enabled": bool(context.seed_enabled),
        "sparse_pc_factor_preflight_seed_used": bool(context.seed_used),
        "sparse_pc_factor_preflight_passed": context.passed,
        "sparse_pc_factor_preflight_error": context.error,
        "sparse_pc_factor_preflight_residual_before": context.residual_before,
        "sparse_pc_factor_preflight_residual_after": context.residual_after,
        "sparse_pc_factor_preflight_improvement_ratio": context.improvement_ratio,
        "sparse_pc_factor_preflight_target_ratio": context.target_ratio,
        "sparse_pc_factor_preflight_max_target_ratio": float(
            context.max_target_ratio
        ),
        "sparse_pc_factor_preflight_residual_diagnostics": (
            context.residual_diagnostics
        ),
    }


def sparse_pc_factor_preflight_result_metadata(
    state: Mapping[str, object],
) -> dict[str, object]:
    """Return factor-preflight diagnostics from stored solver metadata names."""

    return sparse_pc_factor_preflight_result_metadata_from_context(
        SparsePCFactorPreflightMetadataContext(
            enabled=state["factor_preflight_enabled"],
            required=state["factor_preflight_required"],
            seed_enabled=state["factor_preflight_seed_enabled"],
            seed_used=state["factor_preflight_seed_used"],
            passed=state["factor_preflight_passed"],
            error=state["factor_preflight_error"],
            residual_before=state["factor_preflight_residual_before"],
            residual_after=state["factor_preflight_residual_after"],
            improvement_ratio=state["factor_preflight_improvement_ratio"],
            target_ratio=state["factor_preflight_target_ratio"],
            max_target_ratio=state["factor_preflight_max_target_ratio"],
            residual_diagnostics=state["factor_preflight_residual_diagnostics"],
        )
    )


@dataclass(frozen=True)
class SparsePCPatternMetadataContext:
    """Sparse pattern diagnostics consumed by generic sparse-PC metadata."""

    summary: object
    scope: object
    build_s: object


@dataclass(frozen=True)
class SparsePCGMRESStaticMetadataContext:
    """Static generic sparse-PC metadata known before final solve polishing."""

    op: object
    fortran_reduced_sparse_pc: object
    fortran_reduced_sparse_pc_backend: object
    fortran_reduced_sparse_pc_backend_reason: object
    fortran_reduced_xblock_min_size: object
    pc_restart: object
    pc_maxiter: object
    sparse_pc_first_attempt_maxiter: object
    pc_shift: object
    sparse_pc_factor_dtype_initial: object
    sparse_pc_preconditioner_operator: object
    sparse_pc_factorization: object
    sparse_pc_default_factor_kind: object
    sparse_pc_default_ilu_fill_factor: object
    sparse_pc_default_ilu_drop_tol: object
    sparse_pc_default_pattern_color_batch: object
    preconditioner_x: object
    preconditioner_x_min_l: object
    preconditioner_xi: object
    preconditioner_species: object
    sparse_pc_permc_spec: object
    sparse_pc_default_permc_spec: object
    sparse_pc_use_active_dof: object
    sparse_pc_linear_size: object
    sparse_pc_fp_dense_velocity_block: object


def sparse_pc_gmres_static_metadata_from_context(
    context: SparsePCGMRESStaticMetadataContext,
) -> dict[str, object]:
    """Return static generic sparse-PC diagnostics for final metadata."""

    fortran_reduced = bool(context.fortran_reduced_sparse_pc)
    fp_dense_velocity_block = context.sparse_pc_fp_dense_velocity_block
    return {
        "solver_kind": (
            "fortran_reduced_pc_gmres" if fortran_reduced else "sparse_pc_gmres"
        ),
        "gmres_restart": int(context.pc_restart),
        "gmres_maxiter": int(context.pc_maxiter),
        "sparse_pc_first_attempt_maxiter": int(
            context.sparse_pc_first_attempt_maxiter
        ),
        "sparse_pc_shift": float(context.pc_shift),
        "sparse_pc_initial_factor_dtype": _dtype_name(
            context.sparse_pc_factor_dtype_initial
        ),
        "sparse_pc_backend": (
            str(context.fortran_reduced_sparse_pc_backend)
            if fortran_reduced
            else "global"
        ),
        "sparse_pc_backend_reason": (
            str(context.fortran_reduced_sparse_pc_backend_reason)
            if fortran_reduced
            else "not_fortran_reduced"
        ),
        "sparse_pc_xblock_min_size": (
            int(context.fortran_reduced_xblock_min_size)
            if fortran_reduced
            else None
        ),
        "sparse_pc_preconditioner_operator": (
            context.sparse_pc_preconditioner_operator
        ),
        "sparse_pc_factorization": context.sparse_pc_factorization,
        "sparse_pc_default_factorization": context.sparse_pc_default_factor_kind,
        "sparse_pc_default_ilu_fill_factor": float(
            context.sparse_pc_default_ilu_fill_factor
        ),
        "sparse_pc_default_ilu_drop_tol": float(
            context.sparse_pc_default_ilu_drop_tol
        ),
        "sparse_pc_default_pattern_color_batch": int(
            context.sparse_pc_default_pattern_color_batch
        ),
        "sparse_pc_fortran_reduced": fortran_reduced,
        "sparse_pc_fortran_reduced_keeps_theta_zeta": fortran_reduced,
        "sparse_pc_fortran_reduced_preconditioner_x": int(
            context.preconditioner_x
        ),
        "sparse_pc_fortran_reduced_preconditioner_x_min_L": int(
            context.preconditioner_x_min_l
        ),
        "sparse_pc_fortran_reduced_preconditioner_xi": int(
            context.preconditioner_xi
        ),
        "sparse_pc_fortran_reduced_preconditioner_species": int(
            context.preconditioner_species
        ),
        "sparse_pc_permc_spec": context.sparse_pc_permc_spec,
        "sparse_pc_default_permc_spec": context.sparse_pc_default_permc_spec,
        "sparse_pc_active_dof": bool(context.sparse_pc_use_active_dof),
        "sparse_pc_linear_size": int(context.sparse_pc_linear_size),
        "sparse_pc_full_size": int(getattr(context.op, "total_size")),
        "sparse_pc_fp_dense_velocity_block": (
            None if fp_dense_velocity_block is None else bool(fp_dense_velocity_block)
        ),
    }


def sparse_pc_gmres_static_metadata(
    state: Mapping[str, object],
) -> dict[str, object]:
    """Return static generic sparse-PC diagnostics from driver-style names."""

    return sparse_pc_gmres_static_metadata_from_context(
        SparsePCGMRESStaticMetadataContext(
            op=state["op"],
            fortran_reduced_sparse_pc=state["fortran_reduced_sparse_pc"],
            fortran_reduced_sparse_pc_backend=state[
                "fortran_reduced_sparse_pc_backend"
            ],
            fortran_reduced_sparse_pc_backend_reason=state[
                "fortran_reduced_sparse_pc_backend_reason"
            ],
            fortran_reduced_xblock_min_size=state["fortran_reduced_xblock_min_size"],
            pc_restart=state["pc_restart"],
            pc_maxiter=state["pc_maxiter"],
            sparse_pc_first_attempt_maxiter=state["sparse_pc_first_attempt_maxiter"],
            pc_shift=state["pc_shift"],
            sparse_pc_factor_dtype_initial=state["sparse_pc_factor_dtype_initial"],
            sparse_pc_preconditioner_operator=state[
                "sparse_pc_preconditioner_operator"
            ],
            sparse_pc_factorization=state["sparse_pc_factorization"],
            sparse_pc_default_factor_kind=state["sparse_pc_default_factor_kind"],
            sparse_pc_default_ilu_fill_factor=state[
                "sparse_pc_default_ilu_fill_factor"
            ],
            sparse_pc_default_ilu_drop_tol=state["sparse_pc_default_ilu_drop_tol"],
            sparse_pc_default_pattern_color_batch=state[
                "sparse_pc_default_pattern_color_batch"
            ],
            preconditioner_x=state["preconditioner_x"],
            preconditioner_x_min_l=state["preconditioner_x_min_l"],
            preconditioner_xi=state["preconditioner_xi"],
            preconditioner_species=state["preconditioner_species"],
            sparse_pc_permc_spec=state["sparse_pc_permc_spec"],
            sparse_pc_default_permc_spec=state["sparse_pc_default_permc_spec"],
            sparse_pc_use_active_dof=state["sparse_pc_use_active_dof"],
            sparse_pc_linear_size=state["sparse_pc_linear_size"],
            sparse_pc_fp_dense_velocity_block=state[
                "sparse_pc_fp_dense_velocity_block"
            ],
        )
    )


def sparse_pc_pattern_result_metadata_from_context(
    context: SparsePCPatternMetadataContext,
) -> dict[str, object]:
    """Return sparse-PC pattern diagnostics for final metadata."""

    summary = context.summary
    return {
        "sparse_pattern_nnz": int(summary.nnz),
        "sparse_pattern_avg_row_nnz": float(summary.avg_row_nnz),
        "sparse_pattern_max_row_nnz": int(summary.max_row_nnz),
        "sparse_pattern_scope": context.scope,
        "sparse_pattern_build_s": float(context.build_s),
    }


def sparse_pc_pattern_result_metadata(
    state: Mapping[str, object],
) -> dict[str, object]:
    """Return sparse pattern diagnostics from stored solver metadata names."""

    return sparse_pc_pattern_result_metadata_from_context(
        SparsePCPatternMetadataContext(
            summary=state["summary"],
            scope=state["sparse_pattern_scope"],
            build_s=state["pattern_build_s"],
        )
    )


def sparse_pc_gmres_result_metadata(
    state: Mapping[str, object],
) -> dict[str, object]:
    """Return final metadata for generic sparse-PC GMRES solves.

    This helper is deliberately state-mapping based to preserve stored metadata
    key names while keeping schema conversion out of the solve loop.
    """

    target = float(state["target"])
    residual_norm = float(state["residual_norm_sparse_pc"])
    history = state["history"] or ()
    elapsed_s = state.get("sparse_pc_elapsed_s")
    if elapsed_s is None:
        elapsed_s = state["sparse_timer"].elapsed_s()
    factor_bundle_pc = state["factor_bundle_pc"]
    operator_bundle = state["_operator_bundle_pc"]
    operator_metadata = None if operator_bundle is None else getattr(operator_bundle, "metadata", None)
    operator_nnz_estimate = None if operator_metadata is None else getattr(operator_metadata, "nnz_estimate", None)
    operator_csr_nbytes_estimate = (
        None if operator_metadata is None else getattr(operator_metadata, "csr_nbytes_estimate", None)
    )
    factor_nbytes_estimate = getattr(factor_bundle_pc, "factor_nbytes_estimate", None)
    factor_nnz_estimate = getattr(factor_bundle_pc, "factor_nnz_estimate", None)
    factor_elapsed_s = getattr(factor_bundle_pc, "factor_s", None)
    direct_tail_metadata = state.get("sparse_pc_direct_tail_metadata")
    if direct_tail_metadata is None:
        direct_tail_metadata = sparse_pc_direct_tail_result_metadata(state)
    factor_preflight_metadata = state.get("sparse_pc_factor_preflight_metadata")
    if factor_preflight_metadata is None:
        factor_preflight_metadata = sparse_pc_factor_preflight_result_metadata(state)
    pattern_metadata = state.get("sparse_pc_pattern_metadata")
    if pattern_metadata is None:
        pattern_metadata = sparse_pc_pattern_result_metadata(state)
    static_metadata = state.get("sparse_pc_static_metadata")
    if static_metadata is None:
        static_metadata = sparse_pc_gmres_static_metadata(state)

    metadata: dict[str, object] = {
        **static_metadata,
        "residual_kind": "true_residual",
        "accepted_converged": bool(state["sparse_pc_accepted_converged"]),
        "acceptance_criterion": "true_residual",
        "iterations": int(len(history)),
        "matvecs": int(state["mv_count"]),
        "sparse_pc_post_minres_steps_requested": int(state["sparse_pc_post_minres_steps"]),
        "sparse_pc_post_minres_steps_accepted": int(len(state["sparse_pc_post_minres_alphas"])),
        "sparse_pc_post_minres_alpha_clip": float(state["sparse_pc_post_minres_alpha_clip"]),
        "sparse_pc_post_minres_min_improvement": float(state["sparse_pc_post_minres_min_improvement"]),
        "sparse_pc_post_minres_residual_before": state["sparse_pc_post_minres_residual_before"],
        "sparse_pc_post_minres_residual_after": state["sparse_pc_post_minres_residual_after"],
        "sparse_pc_post_minres_history": tuple(float(v) for v in state["sparse_pc_post_minres_history"]),
        "sparse_pc_post_minres_alphas": tuple(float(v) for v in state["sparse_pc_post_minres_alphas"]),
        "sparse_pc_post_minres_error": state["sparse_pc_post_minres_error"],
        "sparse_pc_factor_dtype": _dtype_name(state["sparse_pc_factor_dtype_used"]),
        "sparse_pc_factor_dtype_retry": state["sparse_pc_factor_dtype_retry"],
        **factor_preflight_metadata,
        **direct_tail_metadata,
        "setup_s": float(state["setup_s"]),
        "solve_s": float(state["solve_s"]),
        "elapsed_s": float(elapsed_s),
        **pattern_metadata,
        "sparse_pc_factor_s": float(state["pc_factor_s"]),
        "sparse_pc_factor_elapsed_s": _optional_float(factor_elapsed_s),
        "sparse_pc_factor_nbytes_estimate": _optional_int(factor_nbytes_estimate),
        "sparse_pc_factor_nnz_estimate": _optional_int(factor_nnz_estimate),
        "sparse_pc_operator_nnz_estimate": operator_nnz_estimate,
        "sparse_pc_operator_csr_nbytes_estimate": _optional_int(operator_csr_nbytes_estimate),
        "sparse_pc_residual_target": float(target),
        "sparse_pc_residual_ratio_to_target": (
            residual_norm / float(target) if float(target) > 0.0 else float("inf")
        ),
        "sparse_pc_factor_quality_rejected": bool(state["sparse_pc_factor_quality_rejected"]),
    }
    return metadata


def fortran_reduced_xblock_result_metadata(
    state: Mapping[str, object],
) -> dict[str, object]:
    """Return final metadata for the fortran-reduced x-block sparse-PC solve."""

    moment_metadata = state["moment_schur_metadata"]
    moment_stats = state["moment_schur_stats"]
    global_metadata = state["global_coupling_metadata"]
    global_stats = state["global_coupling_stats"]
    if not isinstance(moment_metadata, Mapping):
        raise TypeError("moment_schur_metadata must be a mapping")
    if not isinstance(moment_stats, Mapping):
        raise TypeError("moment_schur_stats must be a mapping")
    if not isinstance(global_metadata, Mapping):
        raise TypeError("global_coupling_metadata must be a mapping")
    if not isinstance(global_stats, Mapping):
        raise TypeError("global_coupling_stats must be a mapping")

    op = state["op"]
    sparse_pc_fp_dense_velocity_block = state["sparse_pc_fp_dense_velocity_block"]
    target = float(state["target"])
    residual_norm = float(state["residual_norm_sparse_pc"])

    return {
        "solver_kind": "fortran_reduced_pc_gmres",
        "residual_kind": "true_residual",
        "accepted_converged": bool(
            state["fortran_reduced_xblock_accepted_converged"]
        ),
        "acceptance_criterion": "true_residual",
        "iterations": int(len(state["history"] or ())),
        "matvecs": int(state["mv_count"]),
        "gmres_restart": int(state["pc_restart"]),
        "gmres_maxiter": int(state["pc_maxiter"]),
        "sparse_pc_backend": "xblock",
        "sparse_pc_backend_reason": str(
            state["fortran_reduced_sparse_pc_backend_reason"]
        ),
        "sparse_pc_xblock_min_size": int(state["fortran_reduced_xblock_min_size"]),
        "sparse_pc_preconditioner_operator": "fortran_reduced_xblock",
        "sparse_pc_factorization": "xblock_host_sparse",
        "sparse_pc_default_factorization": "xblock_host_sparse",
        "sparse_pc_fortran_reduced": True,
        "sparse_pc_fortran_reduced_keeps_theta_zeta": True,
        "sparse_pc_fortran_reduced_preconditioner_x": int(
            state["preconditioner_x"]
        ),
        "sparse_pc_fortran_reduced_preconditioner_x_min_L": int(
            state["preconditioner_x_min_l"]
        ),
        "sparse_pc_fortran_reduced_preconditioner_xi": int(
            state["preconditioner_xi"]
        ),
        "sparse_pc_fortran_reduced_preconditioner_species": int(
            state["preconditioner_species"]
        ),
        "sparse_pc_xblock_preconditioner_xi": int(
            state["xblock_preconditioner_xi"]
        ),
        "sparse_pc_xblock_assembled_host_fp": bool(state["force_assembled_host_fp"]),
        "sparse_pc_xblock_krylov_method": str(state["xblock_krylov_method"]),
        "sparse_pc_xblock_initial_seed_enabled": bool(state["seed_enabled"]),
        "sparse_pc_xblock_initial_seed_used": bool(state["seed_used"]),
        "sparse_pc_xblock_initial_seed_residual_norm": state[
            "seed_residual_norm"
        ],
        "sparse_pc_xblock_initial_seed_improvement_ratio": state[
            "seed_improvement_ratio"
        ],
        "sparse_pc_xblock_initial_seed_accept_ratio": float(
            state["seed_accept_ratio"]
        ),
        "sparse_pc_xblock_initial_seed_refine_steps": int(
            state["seed_refine_steps"]
        ),
        "sparse_pc_xblock_initial_seed_refines_performed": int(
            state["seed_refines_performed"]
        ),
        "sparse_pc_xblock_moment_schur_enabled": bool(
            state["moment_schur_enabled"]
        ),
        "sparse_pc_xblock_moment_schur_built": bool(state["moment_schur_built"]),
        "sparse_pc_xblock_moment_schur_used": bool(state["moment_schur_used"]),
        "sparse_pc_xblock_moment_schur_reason": state["moment_schur_reason"],
        "sparse_pc_xblock_moment_schur_mode": moment_metadata.get("mode"),
        "sparse_pc_xblock_moment_schur_rank": moment_metadata.get("rank"),
        "sparse_pc_xblock_moment_schur_extra_size": moment_metadata.get(
            "extra_size"
        ),
        "sparse_pc_xblock_moment_schur_setup_s": moment_metadata.get("setup_s"),
        "sparse_pc_xblock_moment_schur_expected_size": moment_metadata.get(
            "expected_size"
        ),
        "sparse_pc_xblock_moment_schur_rcond": moment_metadata.get("rcond"),
        "sparse_pc_xblock_moment_schur_singular_value_proxy": moment_metadata.get(
            "singular_value_proxy",
            (),
        ),
        "sparse_pc_xblock_moment_schur_device_resident": bool(
            moment_metadata.get("device_resident", False)
        ),
        "sparse_pc_xblock_moment_schur_probe_residual_before": state[
            "moment_schur_probe_residual_before"
        ],
        "sparse_pc_xblock_moment_schur_probe_residual_after": state[
            "moment_schur_probe_residual_after"
        ],
        "sparse_pc_xblock_moment_schur_probe_improvement_ratio": state[
            "moment_schur_probe_improvement_ratio"
        ],
        "sparse_pc_xblock_moment_schur_error": moment_metadata.get("error"),
        "sparse_pc_xblock_moment_schur_applies": int(
            moment_stats.get("applies", 0)
        ),
        "sparse_pc_xblock_moment_schur_base_applies": int(
            moment_stats.get("base_applies", 0)
        ),
        "sparse_pc_xblock_global_coupling_enabled": bool(
            state["global_coupling_enabled"]
        ),
        "sparse_pc_xblock_global_coupling_built": bool(
            state["global_coupling_built"]
        ),
        "sparse_pc_xblock_global_coupling_mode": global_metadata.get("mode"),
        "sparse_pc_xblock_global_coupling_load_basis_size": global_metadata.get(
            "load_basis_size"
        ),
        "sparse_pc_xblock_global_coupling_basis_size": global_metadata.get(
            "basis_size"
        ),
        "sparse_pc_xblock_global_coupling_rank": global_metadata.get("rank"),
        "sparse_pc_xblock_global_coupling_setup_s": global_metadata.get("setup_s"),
        "sparse_pc_xblock_global_coupling_setup_budget_s": global_metadata.get(
            "setup_budget_s"
        ),
        "sparse_pc_xblock_global_coupling_setup_budget_reached": bool(
            global_metadata.get("setup_budget_reached", False)
        ),
        "sparse_pc_xblock_global_coupling_rcond": global_metadata.get("rcond"),
        "sparse_pc_xblock_global_coupling_smoother": global_metadata.get("smoother"),
        "sparse_pc_xblock_global_coupling_basis_names": global_metadata.get(
            "basis_names",
            (),
        ),
        "sparse_pc_xblock_global_coupling_error": global_metadata.get("error"),
        "sparse_pc_xblock_global_coupling_applies": int(
            global_stats.get("applies", 0)
        ),
        "sparse_pc_xblock_global_coupling_coarse_applies": int(
            global_stats.get("coarse_applies", 0)
        ),
        "sparse_pc_xblock_drop_tol": float(state["xblock_drop_tol"]),
        "sparse_pc_xblock_drop_rel": float(state["xblock_drop_rel"]),
        "sparse_pc_xblock_ilu_drop_tol": float(state["xblock_ilu_drop_tol"]),
        "sparse_pc_xblock_fill_factor": float(state["xblock_fill_factor"]),
        "sparse_pc_active_dof": bool(state["sparse_pc_use_active_dof"]),
        "sparse_pc_linear_size": int(state["sparse_pc_linear_size"]),
        "sparse_pc_full_size": int(getattr(op, "total_size")),
        "sparse_pc_fp_dense_velocity_block": (
            None
            if sparse_pc_fp_dense_velocity_block is None
            else bool(sparse_pc_fp_dense_velocity_block)
        ),
        "setup_s": float(state["setup_s"]),
        "solve_s": float(state["solve_s"]),
        "elapsed_s": float(state["sparse_timer"].elapsed_s()),
        "sparse_pattern_nnz": 0,
        "sparse_pattern_avg_row_nnz": 0.0,
        "sparse_pattern_max_row_nnz": 0,
        "sparse_pattern_scope": "fortran_reduced_xblock_no_global_pattern",
        "sparse_pattern_build_s": 0.0,
        "sparse_pc_factor_s": float(state["pc_factor_s"]),
        "sparse_pc_factor_elapsed_s": float(state["pc_factor_s"]),
        "sparse_pc_factor_nbytes_estimate": None,
        "sparse_pc_factor_nnz_estimate": None,
        "sparse_pc_residual_target": float(target),
        "sparse_pc_residual_ratio_to_target": (
            residual_norm / float(target) if float(target) > 0.0 else float("inf")
        ),
        "sparse_pc_factor_quality_rejected": bool(
            state["fortran_reduced_xblock_factor_quality_rejected"]
        ),
    }


@dataclass(frozen=True, slots=True)
class XBlockSideProbeDiagnosticsContext:
    """Explicit inputs for side-probe and LGMRES-rescue diagnostics."""

    enabled: object
    used: object
    switched: object
    switch_suppressed_by_global_coupling: object
    switch_suppressed_by_explicit_side: object
    physical_seed_preserved_after_switch: object
    seed_used: object
    seed_residual_norm: object
    initial_side: object
    selected_side: object
    initial_method: object
    selected_method: object
    lgmres_rescue: object
    lgmres_rescue_maxiter_capped: object
    lgmres_rescue_outer_k: object
    residual_norm: object
    residual_ratio: object
    iterations: object
    matvecs: object
    elapsed_s: object


def xblock_side_probe_diagnostics(
    context: XBlockSideProbeDiagnosticsContext,
) -> dict[str, object]:
    """Return side-probe and LGMRES-rescue diagnostics for x-block solves."""

    return {
        "xblock_side_probe_enabled": bool(context.enabled),
        "xblock_side_probe_used": bool(context.used),
        "xblock_side_probe_switched": bool(context.switched),
        "xblock_side_probe_switch_suppressed_by_global_coupling": bool(
            context.switch_suppressed_by_global_coupling
        ),
        "xblock_side_probe_switch_suppressed_by_explicit_side": bool(
            context.switch_suppressed_by_explicit_side
        ),
        "xblock_side_probe_physical_seed_preserved_after_switch": bool(
            context.physical_seed_preserved_after_switch
        ),
        "xblock_side_probe_seed_used": bool(context.seed_used),
        "xblock_side_probe_seed_residual_norm": context.seed_residual_norm,
        "xblock_side_probe_initial_side": context.initial_side,
        "xblock_side_probe_selected_side": context.selected_side,
        "xblock_side_probe_initial_method": context.initial_method,
        "xblock_side_probe_selected_method": context.selected_method,
        "xblock_side_probe_lgmres_rescue": bool(context.lgmres_rescue),
        "xblock_lgmres_rescue_maxiter_capped": bool(
            context.lgmres_rescue_maxiter_capped
        ),
        "xblock_lgmres_rescue_outer_k": context.lgmres_rescue_outer_k,
        "xblock_side_probe_residual_norm": context.residual_norm,
        "xblock_side_probe_residual_ratio": context.residual_ratio,
        "xblock_side_probe_iterations": int(context.iterations),
        "xblock_side_probe_matvecs": int(context.matvecs),
        "xblock_side_probe_s": float(context.elapsed_s),
    }


@dataclass(frozen=True, slots=True)
class XBlockAssembledOperatorDiagnosticsContext:
    """Explicit inputs for assembled-operator and equilibration diagnostics."""

    enabled: object
    built: object
    metadata: Mapping[str, object]
    row_equilibration_enabled: object
    row_equilibration_built: object
    row_equilibration_metadata: Mapping[str, object]
    col_equilibration_enabled: object
    col_equilibration_built: object
    col_equilibration_metadata: Mapping[str, object]


def xblock_assembled_operator_diagnostics(
    context: XBlockAssembledOperatorDiagnosticsContext,
) -> dict[str, object]:
    """Return assembled-operator and equilibration diagnostics for x-block solves."""

    metadata = context.metadata
    row_metadata = context.row_equilibration_metadata
    col_metadata = context.col_equilibration_metadata
    if not isinstance(metadata, Mapping):
        raise TypeError("assembled_operator_metadata must be a mapping")
    if not isinstance(row_metadata, Mapping):
        raise TypeError("xblock_row_equilibration_metadata must be a mapping")
    if not isinstance(col_metadata, Mapping):
        raise TypeError("xblock_col_equilibration_metadata must be a mapping")

    return {
        "xblock_assembled_operator_enabled": bool(context.enabled),
        "xblock_assembled_operator_built": bool(context.built),
        "xblock_assembled_operator_active_dof": metadata.get("active_dof", False),
        "xblock_assembled_operator_preflight_scope": metadata.get("preflight_scope"),
        "xblock_assembled_operator_setup_s": metadata.get("setup_s"),
        "xblock_assembled_operator_preflight_rejected": metadata.get(
            "preflight_rejected",
            False,
        ),
        "xblock_assembled_operator_preflight_pattern_nnz_estimate": metadata.get(
            "preflight_pattern_nnz_estimate"
        ),
        "xblock_assembled_operator_preflight_peak_nbytes_estimate": metadata.get(
            "preflight_peak_nbytes_estimate"
        ),
        "xblock_assembled_operator_preflight_full_csr_nbytes_estimate": metadata.get(
            "preflight_full_csr_nbytes_estimate"
        ),
        "xblock_assembled_operator_preflight_active_csr_nbytes_estimate": metadata.get(
            "preflight_active_csr_nbytes_estimate"
        ),
        "xblock_assembled_operator_pattern_nnz": metadata.get("pattern_nnz"),
        "xblock_assembled_operator_matrix_nnz": metadata.get("matrix_nnz"),
        "xblock_assembled_operator_csr_nbytes_estimate": metadata.get(
            "csr_nbytes_estimate"
        ),
        "xblock_assembled_operator_device_enabled": bool(
            metadata.get("device_enabled", False)
        ),
        "xblock_assembled_operator_device_required": bool(
            metadata.get("device_required", False)
        ),
        "xblock_assembled_operator_device_resident": bool(
            metadata.get("device_resident", False)
        ),
        "xblock_assembled_operator_device_nnz": metadata.get("device_nnz"),
        "xblock_assembled_operator_device_csr_nbytes_estimate": metadata.get(
            "device_csr_nbytes_estimate"
        ),
        "xblock_assembled_operator_device_validation_rel_errors": metadata.get(
            "device_validation_rel_errors",
            (),
        ),
        "xblock_assembled_operator_device_error": metadata.get("device_error"),
        "xblock_assembled_operator_row_equilibration_enabled": bool(
            context.row_equilibration_enabled
        ),
        "xblock_assembled_operator_row_equilibration_built": bool(
            context.row_equilibration_built
        ),
        "xblock_assembled_operator_row_equilibration_norm": row_metadata.get("norm"),
        "xblock_assembled_operator_row_equilibration_setup_s": row_metadata.get(
            "setup_s"
        ),
        "xblock_assembled_operator_row_equilibration_zero_or_tiny_rows": (
            row_metadata.get("zero_or_tiny_rows")
        ),
        "xblock_assembled_operator_row_equilibration_row_norm_min": row_metadata.get(
            "row_norm_min"
        ),
        "xblock_assembled_operator_row_equilibration_row_norm_max": row_metadata.get(
            "row_norm_max"
        ),
        "xblock_assembled_operator_row_equilibration_scale_min": row_metadata.get(
            "row_scale_min"
        ),
        "xblock_assembled_operator_row_equilibration_scale_max": row_metadata.get(
            "row_scale_max"
        ),
        "xblock_assembled_operator_col_equilibration_enabled": bool(
            context.col_equilibration_enabled
        ),
        "xblock_assembled_operator_col_equilibration_built": bool(
            context.col_equilibration_built
        ),
        "xblock_assembled_operator_col_equilibration_norm": col_metadata.get("norm"),
        "xblock_assembled_operator_col_equilibration_setup_s": col_metadata.get(
            "setup_s"
        ),
        "xblock_assembled_operator_col_equilibration_zero_or_tiny_columns": (
            col_metadata.get("zero_or_tiny_columns")
        ),
        "xblock_assembled_operator_col_equilibration_col_norm_min": col_metadata.get(
            "col_norm_min"
        ),
        "xblock_assembled_operator_col_equilibration_col_norm_max": col_metadata.get(
            "col_norm_max"
        ),
        "xblock_assembled_operator_col_equilibration_scale_min": col_metadata.get(
            "col_scale_min"
        ),
        "xblock_assembled_operator_col_equilibration_scale_max": col_metadata.get(
            "col_scale_max"
        ),
        "xblock_assembled_operator_max_colors": metadata.get("max_colors"),
        "xblock_assembled_operator_validation_rel_errors": metadata.get(
            "validation_rel_errors",
            (),
        ),
        "xblock_assembled_operator_error": metadata.get("error"),
    }


@dataclass(frozen=True)
class XBlockCoarseCorrectionDiagnosticsContext:
    """Explicit moment-Schur, two-level, and global-coupling diagnostics."""

    moment_schur_enabled: object
    moment_schur_built: object
    moment_schur_used: object
    moment_schur_reason: object
    moment_schur_default_blocked_by_compact_factors: object
    moment_schur_probe_residual_before: object
    moment_schur_probe_residual_after: object
    moment_schur_probe_improvement_ratio: object
    moment_schur_metadata: object
    moment_schur_stats: object
    two_level_enabled: object
    two_level_built: object
    two_level_metadata: object
    two_level_stats: object
    global_coupling_enabled: object
    global_coupling_built: object
    global_coupling_metadata: object
    global_coupling_stats: object


def xblock_coarse_correction_diagnostics_from_context(
    context: XBlockCoarseCorrectionDiagnosticsContext,
) -> dict[str, object]:
    """Return moment-Schur, two-level, and global-coupling diagnostics."""

    moment_metadata = context.moment_schur_metadata
    moment_stats = context.moment_schur_stats
    two_level_metadata = context.two_level_metadata
    two_level_stats = context.two_level_stats
    global_metadata = context.global_coupling_metadata
    global_stats = context.global_coupling_stats
    for name, value in (
        ("moment_schur_metadata", moment_metadata),
        ("moment_schur_stats", moment_stats),
        ("two_level_metadata", two_level_metadata),
        ("two_level_stats", two_level_stats),
        ("global_coupling_metadata", global_metadata),
        ("global_coupling_stats", global_stats),
    ):
        if not isinstance(value, Mapping):
            raise TypeError(f"{name} must be a mapping")

    return {
        "xblock_moment_schur_enabled": bool(context.moment_schur_enabled),
        "xblock_moment_schur_built": bool(context.moment_schur_built),
        "xblock_moment_schur_used": bool(context.moment_schur_used),
        "xblock_moment_schur_reason": context.moment_schur_reason,
        "xblock_moment_schur_default_blocked_by_compact_factors": bool(
            context.moment_schur_default_blocked_by_compact_factors
        ),
        "xblock_moment_schur_mode": moment_metadata.get("mode"),
        "xblock_moment_schur_rank": moment_metadata.get("rank"),
        "xblock_moment_schur_extra_size": moment_metadata.get("extra_size"),
        "xblock_moment_schur_setup_s": moment_metadata.get("setup_s"),
        "xblock_moment_schur_expected_size": moment_metadata.get("expected_size"),
        "xblock_moment_schur_rcond": moment_metadata.get("rcond"),
        "xblock_moment_schur_singular_value_proxy": moment_metadata.get(
            "singular_value_proxy",
            (),
        ),
        "xblock_moment_schur_device_resident": bool(
            moment_metadata.get("device_resident", False)
        ),
        "xblock_moment_schur_probe_residual_before": (
            context.moment_schur_probe_residual_before
        ),
        "xblock_moment_schur_probe_residual_after": (
            context.moment_schur_probe_residual_after
        ),
        "xblock_moment_schur_probe_improvement_ratio": (
            context.moment_schur_probe_improvement_ratio
        ),
        "xblock_moment_schur_error": moment_metadata.get("error"),
        "xblock_moment_schur_applies": int(moment_stats.get("applies", 0)),
        "xblock_moment_schur_base_applies": int(moment_stats.get("base_applies", 0)),
        "xblock_two_level_enabled": bool(context.two_level_enabled),
        "xblock_two_level_built": bool(context.two_level_built),
        "xblock_two_level_mode": two_level_metadata.get("mode"),
        "xblock_two_level_basis_size": two_level_metadata.get("basis_size"),
        "xblock_two_level_rank": two_level_metadata.get("rank"),
        "xblock_two_level_setup_s": two_level_metadata.get("setup_s"),
        "xblock_two_level_rcond": two_level_metadata.get("rcond"),
        "xblock_two_level_basis_names": two_level_metadata.get("basis_names", ()),
        "xblock_two_level_active_projected": bool(
            two_level_metadata.get("active_projected", False)
        ),
        "xblock_two_level_expected_size": two_level_metadata.get("expected_size"),
        "xblock_two_level_error": two_level_metadata.get("error"),
        "xblock_two_level_applies": int(two_level_stats.get("applies", 0)),
        "xblock_two_level_coarse_applies": int(
            two_level_stats.get("coarse_applies", 0)
        ),
        "xblock_global_coupling_enabled": bool(context.global_coupling_enabled),
        "xblock_global_coupling_built": bool(context.global_coupling_built),
        "xblock_global_coupling_mode": global_metadata.get("mode"),
        "xblock_global_coupling_load_basis_size": global_metadata.get(
            "load_basis_size"
        ),
        "xblock_global_coupling_basis_size": global_metadata.get("basis_size"),
        "xblock_global_coupling_rank": global_metadata.get("rank"),
        "xblock_global_coupling_setup_s": global_metadata.get("setup_s"),
        "xblock_global_coupling_setup_budget_s": global_metadata.get("setup_budget_s"),
        "xblock_global_coupling_setup_budget_reached": bool(
            global_metadata.get("setup_budget_reached", False)
        ),
        "xblock_global_coupling_rcond": global_metadata.get("rcond"),
        "xblock_global_coupling_coarse_solver": global_metadata.get("coarse_solver"),
        "xblock_global_coupling_smoother": global_metadata.get("smoother"),
        "xblock_global_coupling_ridge": global_metadata.get("ridge"),
        "xblock_global_coupling_singular_values": global_metadata.get(
            "singular_values",
            (),
        ),
        "xblock_global_coupling_device_resident": bool(
            global_metadata.get("device_resident", False)
        ),
        "xblock_global_coupling_fsavg_lmax": global_metadata.get("fsavg_lmax"),
        "xblock_global_coupling_angular_lmax": global_metadata.get("angular_lmax"),
        "xblock_global_coupling_basis_names": global_metadata.get("basis_names", ()),
        "xblock_global_coupling_error": global_metadata.get("error"),
        "xblock_global_coupling_applies": int(global_stats.get("applies", 0)),
        "xblock_global_coupling_coarse_applies": int(
            global_stats.get("coarse_applies", 0)
        ),
    }


def xblock_coarse_correction_diagnostics(
    scope: Mapping[str, object],
) -> dict[str, object]:
    """Return coarse-correction diagnostics from stored solver metadata names."""

    return xblock_coarse_correction_diagnostics_from_context(
        XBlockCoarseCorrectionDiagnosticsContext(
            moment_schur_enabled=scope["moment_schur_enabled"],
            moment_schur_built=scope["moment_schur_built"],
            moment_schur_used=scope["moment_schur_used"],
            moment_schur_reason=scope["moment_schur_reason"],
            moment_schur_default_blocked_by_compact_factors=scope[
                "moment_schur_default_blocked_by_compact_factors"
            ],
            moment_schur_probe_residual_before=scope[
                "moment_schur_probe_residual_before"
            ],
            moment_schur_probe_residual_after=scope[
                "moment_schur_probe_residual_after"
            ],
            moment_schur_probe_improvement_ratio=scope[
                "moment_schur_probe_improvement_ratio"
            ],
            moment_schur_metadata=scope["moment_schur_metadata"],
            moment_schur_stats=scope["moment_schur_stats"],
            two_level_enabled=scope["two_level_enabled"],
            two_level_built=scope["two_level_built"],
            two_level_metadata=scope["two_level_metadata"],
            two_level_stats=scope["two_level_stats"],
            global_coupling_enabled=scope["global_coupling_enabled"],
            global_coupling_built=scope["global_coupling_built"],
            global_coupling_metadata=scope["global_coupling_metadata"],
            global_coupling_stats=scope["global_coupling_stats"],
        )
    )


def xblock_device_krylov_diagnostics(scope: Mapping[str, object]) -> dict[str, object]:
    """Return device-Krylov, host-fallback, and transfer-free diagnostics."""

    method = str(scope["xblock_krylov_method"])
    device_methods = scope["xblock_device_krylov_methods"]
    global_metadata = scope["global_coupling_metadata"]
    if not isinstance(global_metadata, Mapping):
        raise TypeError("global_coupling_metadata must be a mapping")

    host_transfer_free_base = bool(
        method in device_methods
        and bool(scope["xblock_jax_factors"])
        and (
            not bool(scope["assembled_operator_built"])
            or bool(scope["assembled_operator_device_resident"])
        )
        and not bool(scope["two_level_built"])
        and (
            not bool(scope["global_coupling_built"])
            or bool(global_metadata.get("device_resident", False))
        )
    )
    fallback_decision = scope["xblock_device_host_fallback_decision"]
    fgmres_jit_active = bool(
        method in {"fgmres_jax", "gmres_jax"}
        and bool(scope["xblock_device_fgmres_jit"])
    )

    return {
        "xblock_device_krylov_method": method if method in device_methods else None,
        "xblock_device_host_fallback_mode": str(fallback_decision.mode),
        "xblock_device_host_fallback_used": bool(fallback_decision.used),
        "xblock_device_host_fallback_reason": str(fallback_decision.reason),
        "xblock_device_host_fallback_requested_method": str(
            fallback_decision.requested_method
        ),
        "xblock_device_host_fallback_requested_env": str(
            scope["xblock_krylov_env_requested"]
        ),
        "xblock_device_host_fallback_effective_krylov_env_value": str(
            fallback_decision.effective_krylov_env_value
        ),
        "xblock_device_host_fallback_min_active_size": int(
            fallback_decision.min_active_size
        ),
        "xblock_device_host_fallback_large_full_fp_3d": bool(
            fallback_decision.large_full_fp_3d
        ),
        "xblock_device_host_fallback_ignored_env": bool(fallback_decision.ignored_env),
        "xblock_device_host_fallback_non_autodiff": bool(
            fallback_decision.non_autodiff
        ),
        "xblock_device_gmres_enabled": bool(method == "gmres_jax"),
        "xblock_device_fgmres_enabled": bool(method == "fgmres_jax"),
        "xblock_device_fgmres_jit_enabled": fgmres_jit_active,
        "xblock_device_fgmres_jit_mode": (
            scope["xblock_device_fgmres_jit_mode"] if fgmres_jit_active else None
        ),
        "xblock_device_fgmres_jit_outer_k": (
            int(scope["xblock_device_fgmres_jit_outer_k"])
            if fgmres_jit_active and scope["xblock_device_fgmres_jit_mode"] == "cycle"
            else 0
        ),
        "xblock_device_bicgstab_enabled": bool(method == "bicgstab_jax"),
        "xblock_device_tfqmr_enabled": bool(method == "tfqmr_jax"),
        "xblock_device_tfqmr_replacement_interval": int(
            scope["tfqmr_replacement_interval"]
        ),
        "xblock_device_krylov_forced_jax_factors": bool(
            scope["xblock_device_krylov_forced_jax_factors"]
        ),
        "xblock_device_fgmres_forced_jax_factors": bool(
            scope["xblock_device_krylov_forced_jax_factors"]
        ),
        "xblock_device_fgmres_forced_right_pc": bool(
            scope["xblock_device_fgmres_forced_right_pc"]
        ),
        "xblock_device_fgmres_block_between_cycles": bool(
            scope["fgmres_block_between_cycles"]
        ),
        "xblock_estimated_gmres_basis_nbytes": int(
            scope["xblock_estimated_gmres_basis_nbytes"]
        ),
        "xblock_estimated_bicgstab_work_nbytes": int(
            scope["xblock_estimated_bicgstab_work_nbytes"]
        ),
        "xblock_estimated_tfqmr_work_nbytes": int(
            scope["xblock_estimated_tfqmr_work_nbytes"]
        ),
        "xblock_device_krylov_host_transfer_free": host_transfer_free_base,
        "xblock_device_fgmres_host_transfer_free": bool(
            method == "fgmres_jax" and host_transfer_free_base
        ),
        "xblock_device_bicgstab_host_transfer_free": bool(
            method == "bicgstab_jax" and host_transfer_free_base
        ),
        "xblock_device_tfqmr_host_transfer_free": bool(
            method == "tfqmr_jax" and host_transfer_free_base
        ),
    }


@dataclass(frozen=True, slots=True)
class XBlockSparsePCCoreDiagnosticsContext:
    """Explicit inputs for top-level x-block sparse-PC diagnostics."""

    solver_kind: object
    accepted_converged: object
    reported_iterations: object
    reported_matvecs: object
    python_matvecs: object
    device_cycle_estimated_matvecs: object
    krylov_method: object
    candidate_krylov_method: object
    candidate_iterations: object
    candidate_matvecs: object
    candidate_residual_norm: object
    fallback_started_from_candidate: object
    fallback_candidate_improved_rhs: object
    precondition_side: object
    default_right_preconditioned: object
    default_short_restart_capped: object
    gmres_restart: object
    gmres_maxiter: object
    setup_s: object
    solve_s: object
    elapsed_s: object
    sparse_pc_factor_s: object
    preconditioner_xi: object
    preconditioner_built: object
    assembled_host: object
    jax_factors: object
    jax_factor_format: object
    jax_factor_apply: object
    lower_fill_mode: object
    lower_fill_ignored_env: object


def xblock_sparse_pc_core_diagnostics(
    context: XBlockSparsePCCoreDiagnosticsContext,
) -> dict[str, object]:
    """Return top-level x-block sparse-PC solve diagnostics."""

    method = str(context.krylov_method)
    candidate_method = str(context.candidate_krylov_method)
    device_estimated_matvecs = context.device_cycle_estimated_matvecs
    jax_factors = bool(context.jax_factors)
    lower_fill_mode = str(context.lower_fill_mode)

    return {
        "solver_kind": context.solver_kind,
        "residual_kind": "true_residual",
        "accepted_converged": bool(context.accepted_converged),
        "acceptance_criterion": "true_residual",
        "iterations": int(context.reported_iterations),
        "matvecs": int(context.reported_matvecs),
        "python_matvecs": int(context.python_matvecs),
        "device_cycle_estimated_matvecs": (
            None if device_estimated_matvecs is None else int(device_estimated_matvecs)
        ),
        "krylov_method": method,
        "candidate_krylov_method": candidate_method,
        "candidate_iterations": int(context.candidate_iterations),
        "candidate_matvecs": int(context.candidate_matvecs),
        "candidate_residual_norm": float(context.candidate_residual_norm),
        "fallback_from_krylov_method": (
            candidate_method if candidate_method != method else None
        ),
        "fallback_started_from_candidate": bool(
            context.fallback_started_from_candidate
        ),
        "fallback_candidate_improved_rhs": bool(
            context.fallback_candidate_improved_rhs
        ),
        "precondition_side": str(context.precondition_side),
        "default_right_preconditioned": bool(context.default_right_preconditioned),
        "default_short_restart_capped": bool(context.default_short_restart_capped),
        "gmres_restart": int(context.gmres_restart),
        "gmres_maxiter": int(context.gmres_maxiter),
        "setup_s": float(context.setup_s),
        "solve_s": float(context.solve_s),
        "elapsed_s": float(context.elapsed_s),
        "sparse_pc_factor_s": float(context.sparse_pc_factor_s),
        "sparse_pc_xblock_preconditioner_xi": int(context.preconditioner_xi),
        "sparse_pc_xblock_preconditioner_built": bool(context.preconditioner_built),
        "sparse_pc_xblock_assembled_host": bool(context.assembled_host),
        "sparse_pc_xblock_jax_factors": jax_factors,
        "sparse_pc_xblock_jax_factor_format": (
            str(context.jax_factor_format) if jax_factors else None
        ),
        "sparse_pc_xblock_jax_factor_apply": (
            str(context.jax_factor_apply) if jax_factors else None
        ),
        "xblock_lower_fill_mode": lower_fill_mode,
        "xblock_lower_fill_requested": lower_fill_mode in {"probe", "force"},
        "xblock_lower_fill_ignored_env": bool(context.lower_fill_ignored_env),
    }


def xblock_sparse_pc_result_diagnostics_from_solve_state(
    state: Mapping[str, object],
    *,
    full_size: object,
) -> dict[str, object]:
    """Build final x-block sparse-PC diagnostics from the solve state.

    The driver now passes precomputed coarse, device, and side-probe
    payloads through typed contexts. This helper keeps the stable public
    metadata keys used by downstream reports and tests.
    """

    assembled_operator_metadata = state.get("xblock_assembled_operator_result_metadata")
    if assembled_operator_metadata is None:
        assembled_operator_metadata = xblock_assembled_operator_diagnostics(
            XBlockAssembledOperatorDiagnosticsContext(
                enabled=state["assembled_operator_enabled"],
                built=state["assembled_operator_built"],
                metadata=state["assembled_operator_metadata"],
                row_equilibration_enabled=state[
                    "xblock_row_equilibration_enabled"
                ],
                row_equilibration_built=state["xblock_row_equilibration_built"],
                row_equilibration_metadata=state[
                    "xblock_row_equilibration_metadata"
                ],
                col_equilibration_enabled=state[
                    "xblock_col_equilibration_enabled"
                ],
                col_equilibration_built=state["xblock_col_equilibration_built"],
                col_equilibration_metadata=state[
                    "xblock_col_equilibration_metadata"
                ],
            )
        )
    coarse_correction_metadata = state.get("xblock_coarse_correction_metadata")
    if coarse_correction_metadata is None:
        coarse_correction_metadata = xblock_coarse_correction_diagnostics(state)
    side_probe_metadata = state.get("xblock_side_probe_metadata")
    if side_probe_metadata is None:
        side_probe_metadata = xblock_side_probe_diagnostics(
            XBlockSideProbeDiagnosticsContext(
                enabled=state["xblock_side_probe_enabled"],
                used=state["xblock_side_probe_used"],
                switched=state["xblock_side_probe_switched"],
                switch_suppressed_by_global_coupling=state[
                    "xblock_side_probe_switch_suppressed_by_global_coupling"
                ],
                switch_suppressed_by_explicit_side=state[
                    "xblock_side_probe_switch_suppressed_by_explicit_side"
                ],
                physical_seed_preserved_after_switch=state[
                    "xblock_side_probe_physical_seed_preserved_after_switch"
                ],
                seed_used=state["xblock_side_probe_seed_used"],
                seed_residual_norm=state["xblock_side_probe_seed_residual_norm"],
                initial_side=state["xblock_side_probe_initial_side"],
                selected_side=state["xblock_side_probe_selected_side"],
                initial_method=state["xblock_side_probe_initial_method"],
                selected_method=state["xblock_side_probe_selected_method"],
                lgmres_rescue=state["xblock_side_probe_lgmres_rescue"],
                lgmres_rescue_maxiter_capped=state[
                    "xblock_lgmres_rescue_maxiter_capped"
                ],
                lgmres_rescue_outer_k=state["xblock_lgmres_rescue_outer_k"],
                residual_norm=state["xblock_side_probe_residual_norm"],
                residual_ratio=state["xblock_side_probe_residual_ratio"],
                iterations=state["xblock_side_probe_iterations"],
                matvecs=state["xblock_side_probe_matvecs"],
                elapsed_s=state["xblock_side_probe_s"],
            )
        )

    return {
        **xblock_sparse_pc_core_diagnostics(
            XBlockSparsePCCoreDiagnosticsContext(
                solver_kind=state["xblock_solver_kind"],
                accepted_converged=state["accepted_converged_xblock"],
                reported_iterations=state["reported_iterations"],
                reported_matvecs=state["reported_matvecs"],
                python_matvecs=state["mv_count"],
                device_cycle_estimated_matvecs=state[
                    "device_krylov_estimated_matvecs"
                ],
                krylov_method=state["xblock_krylov_method"],
                candidate_krylov_method=state["candidate_krylov_method"],
                candidate_iterations=state["candidate_iterations"],
                candidate_matvecs=state["candidate_matvecs"],
                candidate_residual_norm=state["candidate_residual_norm"],
                fallback_started_from_candidate=state[
                    "fallback_started_from_candidate"
                ],
                fallback_candidate_improved_rhs=state[
                    "fallback_candidate_improved_rhs"
                ],
                precondition_side=state["precondition_side"],
                default_right_preconditioned=state["xblock_default_right_pc"],
                default_short_restart_capped=state["xblock_default_restart_capped"],
                gmres_restart=state["pc_restart"],
                gmres_maxiter=state["pc_maxiter"],
                setup_s=state["setup_s"],
                solve_s=state["solve_s"],
                elapsed_s=state["sparse_timer"].elapsed_s(),
                sparse_pc_factor_s=state["pc_factor_s"],
                preconditioner_xi=state["xblock_preconditioner_xi"],
                preconditioner_built=state["xblock_preconditioner_built"],
                assembled_host=state["xblock_assembled_host_fp"],
                jax_factors=state["xblock_jax_factors"],
                jax_factor_format=state["xblock_jax_factor_format"],
                jax_factor_apply=state["xblock_jax_factor_apply"],
                lower_fill_mode=state["xblock_lower_fill_mode"],
                lower_fill_ignored_env=state["xblock_lower_fill_ignored_env"],
            )
        ),
        **xblock_device_krylov_diagnostics(state),
        "xblock_active_dof": bool(state["xblock_use_active_dof"]),
        "xblock_linear_size": int(state["xblock_linear_size"]),
        "xblock_full_size": int(full_size),
        "xblock_initial_seed_used": bool(
            state.get("xblock_initial_seed_used", False)
        ),
        "xblock_initial_seed_residual_norm": state.get(
            "xblock_initial_seed_residual_norm"
        ),
        "xblock_initial_seed_residual_ratio": state.get(
            "xblock_initial_seed_residual_ratio"
        ),
        "xblock_moment_schur_seed_enabled": bool(
            state.get("moment_schur_seed_enabled", False)
        ),
        "xblock_moment_schur_seed_used": bool(
            state.get("moment_schur_seed_used", False)
        ),
        "xblock_moment_schur_seed_residual_norm": state.get(
            "moment_schur_seed_residual_norm"
        ),
        "xblock_moment_schur_seed_residual_ratio": state.get(
            "moment_schur_seed_residual_ratio"
        ),
        **assembled_operator_metadata,
        **coarse_correction_metadata,
        **side_probe_metadata,
    }
