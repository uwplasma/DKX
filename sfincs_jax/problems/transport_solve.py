"""Top-level RHSMode=2/3 transport-matrix solve entry points.

This module owns the public transport solve orchestration that historically
lived in ``sfincs_jax.v3_driver``. The implementation is moved mechanically so
Iteration 4 changes ownership without changing numerical algorithms.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
import math
import os
import time
from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
import numpy as np

from sfincs_jax.profiling import Timer
from sfincs_jax.solvers.explicit_sparse import build_operator_from_matvec, estimate_csr_nbytes, factorize_host_sparse_operator
from sfincs_jax.solver import (
    GMRESSolveResult,
    assemble_dense_matrix_from_matvec,
    dense_solve_from_matrix,
    bicgstab_solve_with_history_scipy,
    explicit_left_preconditioned_gmres_scipy,
    gmres_solve_with_history_scipy,
    recycled_initial_guess,
)
from sfincs_jax.problems.transport_finalize import (
    V3TransportMatrixSolveResult,
    compute_transport_postsolve_diagnostics,
)
from sfincs_jax.operators.profile_sparse_pattern import (
    summarize_v3_sparse_pattern,
    v3_full_system_conservative_sparsity_pattern,
    v3_full_system_conservative_sparsity_pattern_for_indices,
)
from sfincs_jax.solvers.diagnostics import transport_progress_message
from sfincs_jax.problems.transport_policies import (
    transport_host_gmres_accepts_preconditioned_residual,
    transport_residual_gate_failure,
    transport_residual_gate_thresholds_from_env,
)
from sfincs_jax.outputs.transport import TransportStreamingOutputAccumulator
from sfincs_jax.operators.profile_system import (
    V3FullSystemOperator,
    _operator_signature_cached,
    apply_v3_full_system_operator_cached,
)
from sfincs_jax.problems.transport_linear_system import (
    TransportDenseBatchContext,
    TransportLinearSolveCallbacks,
    TransportLinearSolveContext,
    _dense_dtype,
    _emit_rhs_residual,
    _store_dense_batch_result,
    dense_preconditioner_for_matvec,
    dense_solver_for_matvec,
    solve_transport_dense_batch,
    solve_transport_linear,
    solve_transport_linear_with_residual,
    transport_host_gmres_solve,
    transport_restart_for_method,
    transport_solver_kind,
)


_transport_solver_kind = transport_solver_kind
_transport_restart_for_method = transport_restart_for_method
_dense_preconditioner_for_matvec = dense_preconditioner_for_matvec
_dense_solver_for_matvec = dense_solver_for_matvec
_transport_host_gmres_solve = transport_host_gmres_solve
_solve_transport_dense_batch = solve_transport_dense_batch

from importlib import import_module as _import_module

_PROFILE_SOLVE = _import_module("sfincs_jax.problems.profile_solve")
for _name, _value in vars(_PROFILE_SOLVE).items():
    if not _name.startswith("__"):
        globals()[_name] = _value
from sfincs_jax.problems.profile_preconditioner_build import (
    _build_rhsmode23_theta_dd_preconditioner,
    _build_rhsmode23_theta_schwarz_preconditioner,
    _build_rhsmode23_zeta_dd_preconditioner,
    _build_rhsmode23_zeta_schwarz_preconditioner,
)

def _transport_parallel_worker(payload: dict[str, object]) -> dict[str, object]:
    """Worker entry point for parallel whichRHS transport solves."""
    return _solve_transport_parallel_payload(
        payload,
        read_input=read_sfincs_input,
        solve_transport=solve_v3_transport_matrix_linear_gmres,
    )


def solve_v3_transport_matrix_linear_gmres(
    *,
    nml: Namelist,
    x0: jnp.ndarray | None = None,
    x0_by_rhs: dict[int, jnp.ndarray] | None = None,
    tol: float = 1e-10,
    atol: float = 0.0,
    restart: int = 80,
    maxiter: int | None = 400,
    solve_method: str = "auto",
    identity_shift: float = 0.0,
    phi1_hat_base: jnp.ndarray | None = None,
    differentiable: bool | None = None,
    emit: Callable[[int, str], None] | None = None,
    input_namelist: Path | None = None,
    which_rhs_values: Sequence[int] | None = None,
    force_stream_diagnostics: bool | None = None,
    force_store_state: bool | None = None,
    collect_transport_output_fields: bool = True,
    parallel_workers: int | None = None,
) -> V3TransportMatrixSolveResult:
    """Compute a RHSMode=2/3 transport matrix by running all `whichRHS` solves matrix-free in JAX.

    Notes
    -----
    This mirrors the v3 `solver.F90` RHSMode=2/3 path:
    - Loop `whichRHS`
    - Overwrite (dnHatdpsiHats, dTHatdpsiHats, EParallelHat)
    - Build the RHS via `evaluateResidual(f=0)`
    - Solve `A x = rhs`
    - Use `diagnostics.F90` formulas to fill `transportMatrix`
    """
    t_all = Timer()

    maxiter_setup = resolve_transport_maxiter_setup(maxiter)
    maxiter = maxiter_setup.maxiter
    if emit is not None:
        for level, message in maxiter_setup.notes:
            emit(int(level), message)

    if emit is not None:
        emit(0, "solve_v3_transport_matrix_linear_gmres: starting whichRHS loop")
    op0 = full_system_operator_from_namelist(nml=nml, identity_shift=identity_shift, phi1_hat_base=phi1_hat_base)
    _set_precond_size_hint(int(op0.total_size))
    _set_precond_policy_hints(
        has_pas=getattr(op0.fblock, "pas", None) is not None,
        has_fp=getattr(op0.fblock, "fp", None) is not None,
        include_phi1=bool(op0.include_phi1),
        rhs_mode=int(op0.rhs_mode),
    )
    state_setup = resolve_transport_state_setup(op=op0, x0=x0, x0_by_rhs=x0_by_rhs)
    state_in_env = state_setup.state_in_path
    state_out_env = state_setup.state_out_path
    x0 = state_setup.x0
    x0_by_rhs = state_setup.x0_by_rhs
    state_x_by_rhs = state_setup.state_x_by_rhs
    rhs_setup = resolve_transport_which_rhs_setup(rhs_mode=int(op0.rhs_mode), which_rhs_values=which_rhs_values)
    rhs_mode = int(rhs_setup.rhs_mode)
    n = int(rhs_setup.n_rhs)
    which_rhs_values = rhs_setup.which_rhs_values
    subset_mode = bool(rhs_setup.subset_mode)
    parallel_request = resolve_transport_parallel_request(
        which_rhs_count=len(which_rhs_values),
        n_rhs=int(n),
        parallel_workers=parallel_workers,
        parallel_backend=_transport_parallel_backend(),
        visible_gpu_ids=_transport_parallel_visible_gpu_ids,
    )
    parallel_child = bool(parallel_request.parallel_child)
    parallel_workers = int(parallel_request.parallel_workers)
    parallel_backend = str(parallel_request.parallel_backend)

    parallel_result = maybe_run_transport_parallel_solve(
        nml=nml,
        op0=op0,
        rhs_mode=int(rhs_mode),
        n_rhs=int(n),
        which_rhs_values=which_rhs_values,
        parallel_child=bool(parallel_child),
        parallel_workers=int(parallel_workers),
        parallel_backend=parallel_backend,
        input_namelist=input_namelist,
        tol=float(tol),
        atol=float(atol),
        restart=int(restart),
        maxiter=maxiter,
        solve_method=solve_method,
        identity_shift=float(identity_shift),
        collect_transport_output_fields=bool(collect_transport_output_fields),
        phi1_hat_base=phi1_hat_base,
        differentiable=differentiable,
        runtime=TransportParallelSolveRuntime(
            run_gpu_subprocesses=_run_transport_parallel_gpu_subprocesses,
            persistent_pool_enabled=_transport_parallel_persistent_pool_enabled(),
            get_pool=_get_transport_parallel_pool,
            shutdown_pool=_shutdown_transport_parallel_pool,
            worker=_transport_parallel_worker,
            worker_env=_transport_parallel_worker_env,
            executor_class=_transport_parallel_process_pool_executor,
            executor_kwargs=_transport_parallel_pool_executor_kwargs,
            elapsed_s=t_all.elapsed_s,
        ),
        emit=emit,
    )
    if parallel_result is not None:
        return parallel_result
    if emit is not None:
        emit(1, f"solve_v3_transport_matrix_linear_gmres: rhs_mode={rhs_mode} whichRHS_count={n} total_size={int(op0.total_size)}")
        emit(
            0,
            "solve_v3_transport_matrix_linear_gmres: ETA becomes available after the first completed whichRHS solve. "
            "The first solve may include one-time JIT compilation, so later solves can be faster.",
        )

    transport_geom_scheme = transport_geometry_scheme_from_namelist(nml)
    active_dense_setup = resolve_transport_active_dense_setup(
        op=op0,
        rhs_mode=int(rhs_mode),
        n_rhs=int(n),
        solve_method=str(solve_method),
        restart=int(restart),
        maxiter=maxiter,
        backend=jax.default_backend(),
        geometry_scheme=int(transport_geom_scheme),
        dense_accelerator_auto_allowed=_transport_dense_accelerator_auto_allowed(
            op0,
            geometry_scheme=int(transport_geom_scheme),
        ),
        dense_backend_policy_allowed=_transport_dense_backend_allowed(),
        state_out_requested=bool(state_out_env),
        force_stream_diagnostics=force_stream_diagnostics,
        force_store_state=force_store_state,
        subset_mode=bool(subset_mode),
        active_dof_indices=_transport_active_dof_indices,
    )
    if emit is not None:
        for level, message in active_dense_setup.initial_notes:
            emit(int(level), message)
    low_memory_outputs = bool(active_dense_setup.low_memory_outputs)
    stream_diagnostics = bool(active_dense_setup.stream_diagnostics)
    store_state_vectors = bool(active_dense_setup.store_state_vectors)
    solve_method_use = str(active_dense_setup.solve_method_use)
    dense_retry_max = int(active_dense_setup.dense_retry_max)
    dense_mem_block = bool(active_dense_setup.dense_mem_block)
    dense_use_mixed = bool(active_dense_setup.dense_use_mixed)
    dense_backend_allowed = bool(active_dense_setup.dense_backend_allowed)
    gmres_restart = int(active_dense_setup.gmres_restart)
    maxiter = active_dense_setup.maxiter

    use_implicit = _resolve_use_implicit(differentiable=differentiable)
    transport_precondition_side = _transport_precondition_side(op=op0, use_implicit=bool(use_implicit))
    if emit is not None and transport_precondition_side != "left":
        emit(
            1,
            "solve_v3_transport_matrix_linear_gmres: transport preconditioner side="
            f"{transport_precondition_side}",
        )
    distributed_axis = _resolve_distributed_gmres_axis(op=op0, emit=emit)

    use_solver_jit = _use_solver_jit(int(op0.total_size))
    transport_linear_context = TransportLinearSolveContext(
        rhs_mode=int(rhs_mode),
        size_hint=int(op0.total_size),
        use_implicit=bool(use_implicit),
        use_solver_jit=bool(use_solver_jit),
        distributed_axis=distributed_axis,
    )
    transport_linear_callbacks = TransportLinearSolveCallbacks(context=transport_linear_context)

    def _dense_dtype(dtype_in: jnp.dtype) -> jnp.dtype:
        return jnp.float32 if dense_use_mixed else dtype_in

    def _solver_kind(method: str) -> tuple[str, str]:
        return _transport_solver_kind(method, rhs_mode=int(rhs_mode))

    def _restart_for_method(method: str) -> int:
        return _transport_restart_for_method(
            method,
            rhs_mode=int(rhs_mode),
            gmres_restart=int(gmres_restart),
            restart=int(restart),
        )

    _solve_linear = transport_linear_callbacks.solve
    _solve_linear_with_residual = transport_linear_callbacks.solve_with_residual

    if emit is not None:
        for level, message in (*active_dense_setup.active_notes, *active_dense_setup.dense_notes):
            emit(int(level), message)
    use_active_dof_mode = bool(active_dense_setup.use_active_dof_mode)
    active_idx_np = active_dense_setup.active_idx_np
    active_idx_jnp = active_dense_setup.active_idx_jnp
    full_to_active_jnp = active_dense_setup.full_to_active_jnp
    active_size = int(active_dense_setup.active_size)
    dense_mem_block = bool(active_dense_setup.dense_mem_block)
    dense_use_mixed = bool(active_dense_setup.dense_use_mixed)
    solve_method_use = str(active_dense_setup.solve_method_use)
    dense_precond_enabled = bool(active_dense_setup.dense_precond_enabled)
    dense_precond_cache_full: dict[tuple[object, int], Callable[[jnp.ndarray], jnp.ndarray]] = {}
    dense_precond_cache_reduced: dict[tuple[object, int], Callable[[jnp.ndarray], jnp.ndarray]] = {}
    dense_solver_cache_full: dict[tuple[object, int], Callable[[jnp.ndarray], jnp.ndarray]] = {}
    dense_solver_cache_reduced: dict[tuple[object, int], Callable[[jnp.ndarray], jnp.ndarray]] = {}

    reduce_full = None
    expand_reduced = None
    if use_active_dof_mode:
        assert active_idx_jnp is not None
        assert full_to_active_jnp is not None

        def reduce_full(v_full: jnp.ndarray) -> jnp.ndarray:
            return v_full[active_idx_jnp]

        def expand_reduced(v_reduced: jnp.ndarray) -> jnp.ndarray:
            z0 = jnp.zeros((1,), dtype=v_reduced.dtype)
            padded = jnp.concatenate([z0, v_reduced], axis=0)
            return padded[full_to_active_jnp]

    transport_precond_kind = normalize_transport_preconditioner_kind(
        env_value=os.environ.get("SFINCS_JAX_TRANSPORT_PRECOND", "")
    )
    preconditioner_full = None
    preconditioner_reduced = None
    strong_precond_kind: str | None = None
    default_solver_kind = _solver_kind(solve_method_use)[0]
    precond_kind_used: str | None = None
    sparse_jax_config = transport_sparse_jax_config_from_env()
    dd_config = transport_dd_config_from_env(op=op0)
    transport_precond_context = TransportPreconditionerContext(
        op=op0,
        active_size=int(active_size),
        use_active_dof_mode=bool(use_active_dof_mode),
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
        active_indices_np=active_idx_np,
        emit=emit,
    )
    transport_precond_builders = TransportPreconditionerDispatchBuilders(
        collision_builder=_build_rhsmode23_collision_preconditioner,
        sxblock_builder=_build_rhsmode23_sxblock_preconditioner,
        block_builder=_build_rhsmode23_block_preconditioner,
        xmg_builder=_build_rhsmode23_xmg_preconditioner,
        theta_dd_builder=_build_rhsmode23_theta_dd_preconditioner,
        theta_schwarz_builder=_build_rhsmode23_theta_schwarz_preconditioner,
        zeta_dd_builder=_build_rhsmode23_zeta_dd_preconditioner,
        zeta_schwarz_builder=_build_rhsmode23_zeta_schwarz_preconditioner,
        tzfft_builder=_build_rhsmode23_tzfft_preconditioner,
        sparse_jax_builder=_build_sparse_jax_preconditioner_from_matvec,
        sparse_jax_cache_key=_transport_precond_cache_key,
        apply_operator_cached=apply_v3_full_system_operator_cached,
        precond_dtype=_precond_dtype,
        fp_tzfft_builder=_build_rhsmode23_fp_tzfft_preconditioner,
        fp_tzfft_line_builder=_build_rhsmode23_fp_tzfft_line_preconditioner,
        fp_tzfft_line_schur_builder=_build_rhsmode23_fp_tzfft_line_schur_preconditioner,
        fp_local_geom_line_builder=_build_rhsmode23_fp_local_geom_line_preconditioner,
        fp_xblock_tz_lu_builder=_build_rhsmode23_fp_xblock_tz_lu_preconditioner,
        fp_xblock_tz_lu_schur_builder=_build_rhsmode23_fp_xblock_tz_lu_schur_preconditioner,
        fp_structured_fblock_lu_builder=_build_rhsmode23_fp_structured_fblock_lu_preconditioner,
        fp_fortran_reduced_lu_builder=_build_rhsmode23_fp_fortran_reduced_lu_preconditioner,
        fp_direct_active_block_schur_builder=_build_rhsmode23_fp_direct_active_block_schur_preconditioner,
    )
    structured_tzfft_size = int(active_size) if use_active_dof_mode else int(op0.total_size)
    structured_tzfft_first_auto = _transport_tzfft_structured_first_attempt_allowed(
        op0,
        size=int(structured_tzfft_size),
        use_implicit=bool(use_implicit),
    )
    tzfft_backend_allowed = (
        _transport_tzfft_backend_allowed()
        or _transport_tzfft_accelerator_auto_allowed(op0)
        or bool(structured_tzfft_first_auto)
    )
    if structured_tzfft_first_auto and emit is not None:
        method_tz, restart_tz, maxiter_tz = _transport_tzfft_first_attempt_budget(
            restart=int(gmres_restart),
            maxiter=maxiter,
        )
        emit(
            1,
            "solve_v3_transport_matrix_linear_gmres: structured tzfft first attempt enabled "
            f"(size={int(structured_tzfft_size)} method={method_tz} "
            f"restart={int(restart_tz)} maxiter={int(maxiter_tz)})",
        )
    if transport_precond_kind is not None and int(rhs_mode) in {2, 3}:
        precond_kind_used, strong_precond_kind = resolve_transport_preconditioner_choice(
            op=op0,
            transport_precond_kind=transport_precond_kind,
            default_solver_kind=default_solver_kind,
            parallel_workers=int(parallel_workers),
            dense_mem_block=bool(dense_mem_block),
            tzfft_backend_allowed=bool(tzfft_backend_allowed),
            shard_axis=_matvec_shard_axis(op0),
            backend=jax.default_backend(),
            emit=emit,
        )
        transport_precondition_side, side_changed = resolve_transport_precondition_side_for_kind(
            kind=precond_kind_used,
            requested_side=transport_precondition_side,
        )
        if side_changed and emit is not None:
            emit(
                1,
                "solve_v3_transport_matrix_linear_gmres: FP line-factor preconditioner uses left "
                "preconditioning; overriding requested right preconditioning",
            )
        if precond_kind_used is not None:
            preconditioner_full = build_transport_preconditioner_from_kind(
                kind=precond_kind_used,
                context=transport_precond_context,
                builders=transport_precond_builders,
                dd_config=dd_config,
                sparse_jax_config=sparse_jax_config,
                use_reduced=False,
            )
            if use_active_dof_mode and reduce_full is not None and expand_reduced is not None:
                preconditioner_reduced = build_transport_preconditioner_from_kind(
                    kind=precond_kind_used,
                    context=transport_precond_context,
                    builders=transport_precond_builders,
                    dd_config=dd_config,
                    sparse_jax_config=sparse_jax_config,
                    use_reduced=True,
                )
        if emit is not None and precond_kind_used is not None:
            emit(
                1,
                "solve_v3_transport_matrix_linear_gmres: preconditioner="
                f"{precond_kind_used} strong={strong_precond_kind}",
            )

    strong_preconditioner_cache = TransportStrongPreconditionerCache(
        kind=strong_precond_kind,
        precond_kind_used=precond_kind_used,
        preconditioner_full=preconditioner_full,
        preconditioner_reduced=preconditioner_reduced,
        context=transport_precond_context,
        builders=transport_precond_builders,
        dd_config=dd_config,
        sparse_jax_config=sparse_jax_config,
    )

    def _get_strong_preconditioner(use_reduced: bool) -> Callable[[jnp.ndarray], jnp.ndarray] | None:
        return strong_preconditioner_cache.get(use_reduced=bool(use_reduced))

    # RHSMode=2/3 transport reuses the same active operator for multiple drives,
    # so keep sparse-helper factors scoped to this solve and reuse them across RHS.
    transport_sparse_direct_context = _transport_sparse_direct_context_from_env(
        op=op0,
        emit=emit,
        sparse_factor_cache_key=_sparse_factor_cache_key,
        hash_numpy_array_for_cache=_hash_numpy_array_for_cache,
        build_host_sparse_direct_factor_from_matvec=_build_host_sparse_direct_factor_from_matvec,
        build_sparse_ilu_from_matvec=_build_sparse_ilu_from_matvec,
        try_build_direct_active_operator_bundle=_try_build_rhsmode23_fp_direct_active_operator_bundle,
        host_sparse_direct_solve_with_refinement=_host_sparse_direct_solve_with_refinement,
        host_sparse_direct_refine_steps=_host_sparse_direct_refine_steps,
        host_sparse_direct_polish=_host_sparse_direct_polish,
        sparse_factor_dtype=_transport_sparse_factor_dtype,
        sparse_direct_use_explicit_helper=_transport_sparse_direct_use_explicit_helper,
        sparse_direct_needs_float64_retry=_transport_sparse_direct_needs_float64_retry,
    )

    # Geometry scalars needed for the transport-matrix formulas.
    grids = grids_from_namelist(nml)
    geom = geometry_from_namelist(nml=nml, grids=grids)

    state_vectors: dict[int, jnp.ndarray] = {}
    residual_norms: dict[int, jnp.ndarray] = {}
    solver_kinds_by_rhs: dict[int, str] = {}
    solve_methods_by_rhs: dict[int, str] = {}
    elapsed_s = np.zeros((n,), dtype=np.float64)
    op_rhs_by_index = [with_transport_rhs_settings(op0, which_rhs=which_rhs) for which_rhs in which_rhs_values]
    rhs_by_index = [rhs_v3_full_system_jit(op_rhs) for op_rhs in op_rhs_by_index]
    rhs_norms: dict[int, jnp.ndarray] = {
        int(which_rhs): jnp.linalg.norm(rhs_by_index[idx])
        for idx, which_rhs in enumerate(which_rhs_values)
    }
    abort_max_residual, abort_max_relative_residual = transport_residual_gate_thresholds_from_env()
    transport_loop_progress = TransportLoopProgress(
        which_rhs_values=which_rhs_values,
        rhs_norms=rhs_norms,
        residual_norms=residual_norms,
        elapsed_s=elapsed_s,
        abort_max_residual=float(abort_max_residual),
        abort_max_relative_residual=float(abort_max_relative_residual),
        emit=emit,
    )

    use_op_rhs_in_matvec = bool(op0.include_phi1_in_kinetic)
    env_transport_matvec = os.environ.get("SFINCS_JAX_TRANSPORT_MATVEC_MODE", "").strip().lower()
    if env_transport_matvec == "rhs":
        use_op_rhs_in_matvec = True
    elif env_transport_matvec == "base":
        use_op_rhs_in_matvec = False
    op_matvec_by_index = [op_rhs if use_op_rhs_in_matvec else op0 for op_rhs in op_rhs_by_index]

    env_diag_op = os.environ.get("SFINCS_JAX_TRANSPORT_DIAG_OP", "").strip().lower()
    use_diag_op0 = env_diag_op != "rhs"
    diag_op_by_index = op_rhs_by_index if not use_diag_op0 else None

    transport_output_fields: dict[str, np.ndarray] | None = None
    collect_full_transport_outputs = bool(collect_transport_output_fields)
    streaming_outputs: TransportStreamingOutputAccumulator | None = None
    if stream_diagnostics:
        streaming_outputs = TransportStreamingOutputAccumulator.create(
            nml=nml,
            grids=grids,
            geom=geom,
            op0=op0,
            n_rhs=n,
            collect_full_output_fields=collect_full_transport_outputs,
        )

        def _collect_transport_outputs(which_rhs: int, x_full: jnp.ndarray) -> None:
            """Populate streaming diagnostics for a single whichRHS solve."""
            assert streaming_outputs is not None
            streaming_outputs.collect(int(which_rhs), x_full)

    transport_matvec_cache = TransportMatvecCache(
        use_active_dof_mode=bool(use_active_dof_mode),
        active_size=int(active_size),
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
    )
    _get_full_matvec = transport_matvec_cache.get_full
    _get_reduced_matvec = transport_matvec_cache.get_reduced

    recycle_k = resolve_transport_recycle_k(
        op=op0,
        use_implicit=bool(use_implicit),
        op_matvec_by_index=op_matvec_by_index,
        disable_auto_recycle=_transport_disable_auto_recycle,
        emit=emit,
    )
    recycle_state = TransportRecycleState(k=int(recycle_k))
    state_recycle_env = os.environ.get("SFINCS_JAX_TRANSPORT_RECYCLE_STATE", "").strip().lower()
    state_recycle_enabled = state_recycle_env not in {"0", "false", "no", "off"}
    if recycle_k > 0 and state_recycle_enabled and state_x_by_rhs:
        recycle_state.seed_from_state(
            state_x_by_rhs=state_x_by_rhs,
            total_size=int(op0.total_size),
            active_size=int(active_size),
            matvec_cache=transport_matvec_cache,
            op_ref=op_matvec_by_index[0],
        )

    def _residual_value(res: GMRESSolveResult) -> float:
        return transport_residual_value(res)

    def _needs_retry(res: GMRESSolveResult, target: float) -> bool:
        return transport_result_needs_retry(
            res,
            float(target),
            result_is_finite=_gmres_result_is_finite,
        )

    per_rhs_loop_policy = resolve_transport_per_rhs_loop_policy(op=op0, rhs_mode=int(rhs_mode))

    constraint_projector = TransportConstraintNullspaceProjector(op=op0, policy=per_rhs_loop_policy)
    _maybe_project_constraint_nullspace = constraint_projector.project

    dense_batch_done = False
    dense_batch_fallback_enabled = bool(per_rhs_loop_policy.dense_batch_fallback_enabled)
    transport_rhs_finalization_context = TransportRHSFinalizationContext(
        state_vectors=state_vectors,
        residual_norms=residual_norms,
        solver_kinds_by_rhs=solver_kinds_by_rhs,
        solve_methods_by_rhs=solve_methods_by_rhs,
        store_state_vectors=bool(store_state_vectors),
        stream_diagnostics=bool(stream_diagnostics),
        collect_transport_outputs=_collect_transport_outputs if stream_diagnostics else None,
        recycle_state=recycle_state if recycle_k > 0 else None,
        apply_operator=apply_v3_full_system_operator_cached,
        emit_iteration_stats=_emit_transport_ksp_iteration_stats,
        emit=emit,
        iter_stats_enabled=bool(per_rhs_loop_policy.iter_stats_enabled),
        iter_stats_max_size=per_rhs_loop_policy.iter_stats_max_size,
        atol=float(atol), maxiter=maxiter, precond_side=transport_precondition_side,
    )

    def _dense_batch_solve_all(*, op_probe_ref: V3FullSystemOperator, reason: str) -> bool:
        dense_batch_context = TransportDenseBatchContext(
            dense_backend_allowed=bool(dense_backend_allowed),
            dense_use_mixed=bool(dense_use_mixed),
            use_active_dof_mode=bool(use_active_dof_mode),
            active_size=int(active_size),
            op0=op0,
            op_matvec_by_index=op_matvec_by_index,
            rhs_by_index=rhs_by_index,
            which_rhs_values=which_rhs_values,
            rhs_norms=rhs_norms,
            residual_norms=residual_norms,
            solver_kinds_by_rhs=solver_kinds_by_rhs,
            solve_methods_by_rhs=solve_methods_by_rhs,
            elapsed_s=elapsed_s,
            state_vectors=state_vectors,
            store_state_vectors=bool(store_state_vectors),
            stream_diagnostics=bool(stream_diagnostics),
            rhs3_krylov_flags=per_rhs_loop_policy.rhs3_krylov_flags,
            maybe_project_constraint_nullspace=_maybe_project_constraint_nullspace,
            collect_transport_outputs=_collect_transport_outputs if stream_diagnostics else None,
            reduce_full=reduce_full,
            expand_reduced=expand_reduced,
            emit=emit,
        )
        return _solve_transport_dense_batch(
            context=dense_batch_context,
            op_probe_ref=op_probe_ref,
            reason=reason,
        )

    if str(solve_method_use).lower() == "dense":
        op_probe_ref = op_matvec_by_index[0]
        if _dense_batch_solve_all(op_probe_ref=op_probe_ref, reason="auto dense"):
            dense_batch_done = True

    if not dense_batch_done:
        for idx, which_rhs in enumerate(which_rhs_values):
            t_rhs = Timer()
            op_rhs = op_rhs_by_index[idx]
            rhs = rhs_by_index[idx]
            op_matvec = op_matvec_by_index[idx]
            if emit is not None:
                emit(0, f"whichRHS={which_rhs}/{n}: assembling+solving (rhs_norm={float(jnp.linalg.norm(rhs)):.6e})")
                emit(1, f"whichRHS={which_rhs}/{n}: evaluateJacobian called (matrix-free)")

            use_loose_epar_krylov, force_epar_krylov = per_rhs_loop_policy.rhs3_krylov_flags(which_rhs)
            solve_method_rhs = solve_method_use
            tol_rhs = tol
            if force_epar_krylov or use_loose_epar_krylov:
                solve_method_rhs = "incremental"
                if use_loose_epar_krylov:
                    epar_tol_env = os.environ.get("SFINCS_JAX_TRANSPORT_EPAR_TOL", "").strip()
                    try:
                        epar_tol = float(epar_tol_env) if epar_tol_env else 1e-8
                    except ValueError:
                        epar_tol = 1e-8
                    tol_rhs = max(float(tol), float(epar_tol))

            if use_active_dof_mode:
                assert active_idx_jnp is not None
                assert full_to_active_jnp is not None
                assert reduce_full is not None
                assert expand_reduced is not None
                mv_reduced = _get_reduced_matvec(op_matvec)

                rhs_reduced = reduce_full(rhs)
                preconditioner_use = preconditioner_reduced
                if dense_precond_enabled:
                    sig = _operator_signature_cached(op_matvec)
                    preconditioner_use = _dense_preconditioner_for_matvec(
                        matvec_fn=mv_reduced,
                        n=active_size,
                        dtype=_dense_dtype(rhs_reduced.dtype),
                        cache=dense_precond_cache_reduced,
                        key=(sig, int(active_size)),
                    )
                x0_reduced = None
                x0_local = x0_by_rhs.get(int(which_rhs)) if x0_by_rhs else x0
                if x0_local is not None:
                    x0_arr = jnp.asarray(x0_local)
                    if x0_arr.shape == (active_size,):
                        x0_reduced = x0_arr
                    elif x0_arr.shape == (op0.total_size,):
                        x0_reduced = reduce_full(x0_arr)
                if recycle_k > 0:
                    x0_recycled = recycle_state.candidate_reduced(rhs_reduced)
                    if x0_reduced is None and x0_recycled is not None:
                        x0_reduced = x0_recycled

                solver_kind_used = _solver_kind(solve_method_rhs)[0]
                solve_method_used = solve_method_rhs
                restart_used = _restart_for_method(solve_method_rhs)
                preconditioner_used = preconditioner_use
                x0_used = x0_reduced
                dense_used = False
                structured_tzfft_first_attempt = False
                initial_solve_method_rhs = solve_method_rhs
                initial_restart_used = _restart_for_method(solve_method_rhs)
                initial_maxiter = maxiter
                if (
                    structured_tzfft_first_auto
                    and precond_kind_used == "tzfft"
                    and preconditioner_use is not None
                    and str(solve_method_rhs).strip().lower()
                    in {"auto", "default", "batched", "bicgstab", "bicgstab_jax", "incremental"}
                ):
                    structured_tzfft_first_attempt = True
                    initial_solve_method_rhs, initial_restart_used, initial_maxiter = (
                        _transport_tzfft_first_attempt_budget(
                            restart=int(gmres_restart),
                            maxiter=maxiter,
                        )
                    )
                    solver_kind_used = "gmres"
                    solve_method_used = initial_solve_method_rhs
                    restart_used = int(initial_restart_used)
                    if emit is not None:
                        emit(
                            1,
                            "solve_v3_transport_matrix_linear_gmres: structured tzfft first attempt "
                            f"whichRHS={int(which_rhs)} size={int(active_size)} "
                            f"restart={int(initial_restart_used)} maxiter={int(initial_maxiter)}",
                        )
                target_rhs = max(float(atol), float(tol_rhs) * float(jnp.linalg.norm(rhs_reduced)))
                host_gmres_first_attempt = _transport_host_gmres_first_attempt_allowed(
                    op=op0,
                    size=int(active_size),
                    use_implicit=bool(use_implicit),
                )
                sparse_direct_first_attempt = _transport_sparse_direct_first_attempt_allowed(
                    op=op0,
                    size=int(active_size),
                    use_implicit=bool(use_implicit),
                )
                if host_gmres_first_attempt:
                    if emit is not None:
                        emit(
                            1,
                            "solve_v3_transport_matrix_linear_gmres: host SciPy GMRES first attempt "
                            f"(size={int(active_size)} backend={jax.default_backend()})",
                        )
                    try:
                        res_reduced, residual_vec = _transport_host_gmres_solve(
                            op=op0,
                            matvec_fn=mv_reduced,
                            b_vec=rhs_reduced,
                            x0_vec=x0_reduced,
                            preconditioner_fn=preconditioner_use,
                            tol_val=tol_rhs,
                            atol_val=atol,
                            restart_val=initial_restart_used,
                            maxiter_val=initial_maxiter,
                            precondition_side_val=transport_precondition_side,
                            emit=emit,
                            which_rhs=int(which_rhs),
                            progress_every=_transport_host_gmres_progress_every(),
                        )
                        solver_kind_used = "gmres_scipy"
                        solve_method_used = "incremental"
                        restart_used = initial_restart_used
                    except Exception as exc:  # noqa: BLE001
                        if emit is not None:
                            emit(
                                1,
                                "solve_v3_transport_matrix_linear_gmres: host SciPy GMRES first attempt failed "
                                f"({type(exc).__name__}: {exc})",
                            )
                        res_reduced, residual_vec = _solve_linear_with_residual(
                            matvec_fn=mv_reduced,
                            b_vec=rhs_reduced,
                            x0_vec=x0_reduced,
                            tol_val=tol_rhs,
                            atol_val=atol,
                            restart_val=initial_restart_used,
                            maxiter_val=initial_maxiter,
                            solve_method_val=initial_solve_method_rhs,
                            preconditioner_val=preconditioner_use,
                            precondition_side_val=transport_precondition_side,
                        )
                elif sparse_direct_first_attempt:
                    if emit is not None:
                        emit(
                            1,
                            "solve_v3_transport_matrix_linear_gmres: host sparse LU first attempt "
                            f"(size={int(active_size)} backend={jax.default_backend()})",
                        )
                    try:
                        sig = _operator_signature_cached(op_matvec)
                        res_reduced = transport_sparse_direct_context.solve(
                            matvec_fn=mv_reduced,
                            b_vec=rhs_reduced,
                            n=int(active_size),
                            dtype=rhs_reduced.dtype,
                            cache_key=("transport_sparse_lu", sig, int(active_size), "active"),
                            active_indices_np=active_idx_np,
                            tol_val=tol_rhs,
                            atol_val=atol,
                            restart_val=initial_restart_used,
                            maxiter_val=initial_maxiter,
                            precondition_side_val=transport_precondition_side,
                        )
                        solver_kind_used = "sparse_lu"
                        solve_method_used = "sparse_lu"
                        restart_used = 0
                        preconditioner_used = None
                        x0_used = None
                    except Exception as exc:  # noqa: BLE001
                        if emit is not None:
                            emit(
                                1,
                                "solve_v3_transport_matrix_linear_gmres: host sparse LU first attempt failed "
                                f"({type(exc).__name__}: {exc})",
                            )
                        res_reduced = _solve_linear(
                            matvec_fn=mv_reduced,
                            b_vec=rhs_reduced,
                            x0_vec=x0_reduced,
                            tol_val=tol_rhs,
                            atol_val=atol,
                            restart_val=initial_restart_used,
                            maxiter_val=initial_maxiter,
                            solve_method_val=initial_solve_method_rhs,
                            preconditioner_val=preconditioner_use,
                            precondition_side_val=transport_precondition_side,
                        )
                else:
                    res_reduced = _solve_linear(
                        matvec_fn=mv_reduced,
                        b_vec=rhs_reduced,
                        x0_vec=x0_reduced,
                        tol_val=tol_rhs,
                        atol_val=atol,
                        restart_val=initial_restart_used,
                        maxiter_val=initial_maxiter,
                        solve_method_val=initial_solve_method_rhs,
                        preconditioner_val=preconditioner_use,
                        precondition_side_val=transport_precondition_side,
                    )
                solver_kind = _solver_kind(initial_solve_method_rhs)[0]
                if solver_kind == "bicgstab" and (not _gmres_result_is_finite(res_reduced) or float(res_reduced.residual_norm) > target_rhs):
                    if emit is not None:
                        emit(
                            0,
                            "solve_v3_transport_matrix_linear_gmres: BiCGStab fallback to GMRES "
                            f"(residual={float(res_reduced.residual_norm):.3e} > target={target_rhs:.3e})",
                        )
                    res_reduced = _solve_linear(
                        matvec_fn=mv_reduced,
                        b_vec=rhs_reduced,
                        x0_vec=x0_reduced,
                        tol_val=tol_rhs,
                        atol_val=atol,
                        restart_val=gmres_restart,
                        maxiter_val=maxiter,
                        solve_method_val="incremental",
                        preconditioner_val=preconditioner_use,
                        precondition_side_val=transport_precondition_side,
                    )
                    solver_kind_used = "gmres"
                    solve_method_used = "incremental"
                    restart_used = gmres_restart
                sparse_direct_rescue = _transport_sparse_direct_rescue_allowed(
                    op=op0,
                    size=int(active_size),
                    residual_norm=float(res_reduced.residual_norm),
                    target=float(target_rhs),
                    use_implicit=bool(use_implicit),
                )
                if structured_tzfft_first_attempt and _needs_retry(res_reduced, target_rhs):
                    sparse_direct_rescue = sparse_direct_rescue or _transport_sparse_direct_rescue_allowed(
                        op=op0,
                        size=int(active_size),
                        residual_norm=float("nan"),
                        target=float(target_rhs),
                        use_implicit=bool(use_implicit),
                    )
                sparse_direct_rescue_first = _transport_sparse_direct_rescue_first(
                    sparse_direct_rescue=sparse_direct_rescue,
                )
                if sparse_direct_rescue_first and emit is not None:
                    emit(
                        1,
                        "solve_v3_transport_matrix_linear_gmres: sparse LU rescue-first "
                        "auto mode -> defer transport retry branches",
                    )
                if _needs_retry(res_reduced, target_rhs) and preconditioner_use is not None and (not sparse_direct_rescue_first):
                    if emit is not None:
                        emit(
                            0,
                            "solve_v3_transport_matrix_linear_gmres: retry without preconditioner "
                            f"(residual={float(res_reduced.residual_norm):.3e} > target={target_rhs:.3e})",
                        )
                    res_retry = _solve_linear(
                        matvec_fn=mv_reduced,
                        b_vec=rhs_reduced,
                        x0_vec=x0_reduced,
                        tol_val=tol_rhs,
                        atol_val=atol,
                        restart_val=_restart_for_method(solve_method_rhs),
                        maxiter_val=maxiter,
                        solve_method_val=solve_method_rhs,
                        preconditioner_val=None,
                        precondition_side_val=transport_precondition_side,
                    )
                    if _residual_value(res_retry) < _residual_value(res_reduced):
                        res_reduced = res_retry
                        preconditioner_use = None
                        preconditioner_used = None
                if _needs_retry(res_reduced, target_rhs) and (not sparse_direct_rescue_first):
                    strong_precond = _get_strong_preconditioner(True)
                    if strong_precond is not None and strong_precond is not preconditioner_use:
                        if emit is not None:
                            emit(
                                0,
                                "solve_v3_transport_matrix_linear_gmres: retry with strong preconditioner "
                                f"(residual={float(res_reduced.residual_norm):.3e} > target={target_rhs:.3e})",
                            )
                        res_strong = _solve_linear(
                            matvec_fn=mv_reduced,
                            b_vec=rhs_reduced,
                            x0_vec=res_reduced.x,
                            tol_val=tol_rhs,
                            atol_val=atol,
                            restart_val=gmres_restart,
                            maxiter_val=maxiter,
                            solve_method_val="incremental",
                            preconditioner_val=strong_precond,
                            precondition_side_val=transport_precondition_side,
                        )
                        if _residual_value(res_strong) < _residual_value(res_reduced):
                            res_reduced = res_strong
                            preconditioner_use = strong_precond
                            preconditioner_used = strong_precond
                            solver_kind_used = "gmres"
                            solve_method_used = "incremental"
                            restart_used = gmres_restart
                if _needs_retry(res_reduced, target_rhs) and sparse_direct_rescue:
                    if emit is not None:
                        emit(
                            0,
                            "solve_v3_transport_matrix_linear_gmres: sparse LU direct rescue "
                            f"(size={int(active_size)} residual={float(res_reduced.residual_norm):.3e} > target={target_rhs:.3e})",
                        )
                    try:
                        sig = _operator_signature_cached(op_matvec)
                        res_sparse = transport_sparse_direct_context.solve(
                            matvec_fn=mv_reduced,
                            b_vec=rhs_reduced,
                            n=int(active_size),
                            dtype=rhs_reduced.dtype,
                            cache_key=("transport_sparse_lu", sig, int(active_size), "active"),
                            active_indices_np=active_idx_np,
                            tol_val=tol_rhs,
                            atol_val=atol,
                            restart_val=_restart_for_method(solve_method_rhs),
                            maxiter_val=maxiter,
                            precondition_side_val=transport_precondition_side,
                        )
                        if _residual_value(res_sparse) < _residual_value(res_reduced):
                            res_reduced = res_sparse
                            preconditioner_use = None
                            preconditioner_used = None
                            solver_kind_used = "sparse_lu"
                            solve_method_used = "sparse_lu"
                    except Exception as exc:  # noqa: BLE001
                        if emit is not None:
                            emit(
                                1,
                                "solve_v3_transport_matrix_linear_gmres: sparse LU direct rescue failed "
                                f"({type(exc).__name__}: {exc})",
                            )
                if _needs_retry(res_reduced, target_rhs) and dense_retry_max > 0 and int(active_size) <= int(dense_retry_max):
                    if emit is not None:
                        emit(
                            0,
                            "solve_v3_transport_matrix_linear_gmres: dense fallback "
                            f"(size={int(active_size)} residual={float(res_reduced.residual_norm):.3e} > target={target_rhs:.3e})",
                        )
                    try:
                        sig = _operator_signature_cached(op_matvec)
                        dense_solver = _dense_solver_for_matvec(
                            matvec_fn=mv_reduced,
                            n=int(active_size),
                            dtype=_dense_dtype(rhs_reduced.dtype),
                            cache=dense_solver_cache_reduced,
                            key=(sig, int(active_size), str(_dense_dtype(rhs_reduced.dtype))),
                        )
                        rhs_dense = jnp.asarray(rhs_reduced, dtype=_dense_dtype(rhs_reduced.dtype))
                        x_dense = dense_solver(rhs_dense)
                        if dense_use_mixed:
                            r_dense0 = rhs_reduced - mv_reduced(jnp.asarray(x_dense, dtype=rhs_reduced.dtype))
                            dx = dense_solver(jnp.asarray(r_dense0, dtype=_dense_dtype(rhs_reduced.dtype)))
                            x_dense = jnp.asarray(x_dense, dtype=rhs_reduced.dtype) + jnp.asarray(dx, dtype=rhs_reduced.dtype)
                        r_dense = rhs_reduced - mv_reduced(x_dense)
                        res_dense = GMRESSolveResult(x=x_dense, residual_norm=jnp.linalg.norm(r_dense))
                        if _residual_value(res_dense) < _residual_value(res_reduced):
                            res_reduced = res_dense
                            dense_used = True
                            solver_kind_used = "dense"
                            solve_method_used = "dense"
                    except Exception as exc:  # noqa: BLE001
                        if emit is not None:
                            emit(
                                1,
                                "solve_v3_transport_matrix_linear_gmres: dense fallback failed "
                                f"({type(exc).__name__}: {exc})",
                            )
                polish_config = transport_polish_config_from_env(
                    rhs_mode=int(rhs_mode),
                    residual_norm=_residual_value(res_reduced),
                    target=float(target_rhs),
                    gmres_restart=int(gmres_restart),
                    maxiter=maxiter,
                )
                if _needs_retry(res_reduced, target_rhs) and polish_config.enabled:
                    polish_precond = _get_strong_preconditioner(True)
                    if polish_precond is None:
                        polish_precond = preconditioner_use
                    if emit is not None:
                        emit(
                            0,
                            "solve_v3_transport_matrix_linear_gmres: polish solve for RHSMode=3 "
                            f"(residual={float(res_reduced.residual_norm):.3e} > "
                            f"max({polish_config.ratio:.1f}x target, {polish_config.abs_tol:.1e}), "
                            f"restart={polish_config.restart} maxiter={polish_config.maxiter})",
                        )
                    res_polish = _solve_linear(
                        matvec_fn=mv_reduced,
                        b_vec=rhs_reduced,
                        x0_vec=res_reduced.x,
                        tol_val=tol_rhs,
                        atol_val=atol,
                        restart_val=int(polish_config.restart),
                        maxiter_val=int(polish_config.maxiter),
                        solve_method_val="incremental",
                        preconditioner_val=polish_precond,
                        precondition_side_val=transport_precondition_side,
                    )
                    if transport_candidate_is_better(candidate=res_polish, current=res_reduced):
                        res_reduced = res_polish
                        preconditioner_used = polish_precond
                        solver_kind_used = "gmres"
                        solve_method_used = "incremental"
                        restart_used = int(polish_config.restart)
                x_full = expand_reduced(res_reduced.x)
                x_full = _maybe_project_constraint_nullspace(
                    x_full, which_rhs=int(which_rhs), op_matvec=op_matvec, rhs_vec=rhs
                )
                ax_full = apply_v3_full_system_operator_cached(op_matvec, x_full)
                res_norm_full = jnp.linalg.norm(ax_full - rhs)
                if (not dense_used) and dense_retry_max > 0 and int(active_size) <= int(dense_retry_max):
                    target_full = max(float(atol), float(tol_rhs) * float(jnp.linalg.norm(rhs)))
                    if float(res_norm_full) > target_full:
                        if emit is not None:
                            emit(
                                0,
                                "solve_v3_transport_matrix_linear_gmres: dense fallback (true residual) "
                                f"(size={int(active_size)} residual={float(res_norm_full):.3e} > target={target_full:.3e})",
                            )
                        try:
                            sig = _operator_signature_cached(op_matvec)
                            dense_solver = _dense_solver_for_matvec(
                                matvec_fn=mv_reduced,
                                n=int(active_size),
                                dtype=_dense_dtype(rhs_reduced.dtype),
                                cache=dense_solver_cache_reduced,
                                key=(sig, int(active_size), str(_dense_dtype(rhs_reduced.dtype))),
                            )
                            rhs_dense = jnp.asarray(rhs_reduced, dtype=_dense_dtype(rhs_reduced.dtype))
                            x_dense = dense_solver(rhs_dense)
                            if dense_use_mixed:
                                r_dense0 = rhs_reduced - mv_reduced(jnp.asarray(x_dense, dtype=rhs_reduced.dtype))
                                dx = dense_solver(jnp.asarray(r_dense0, dtype=_dense_dtype(rhs_reduced.dtype)))
                                x_dense = jnp.asarray(x_dense, dtype=rhs_reduced.dtype) + jnp.asarray(dx, dtype=rhs_reduced.dtype)
                            x_full_dense = expand_reduced(x_dense)
                            x_full_dense = _maybe_project_constraint_nullspace(
                                x_full_dense, which_rhs=int(which_rhs), op_matvec=op_matvec, rhs_vec=rhs
                            )
                            ax_dense = apply_v3_full_system_operator_cached(op_matvec, x_full_dense)
                            res_dense_norm = jnp.linalg.norm(ax_dense - rhs)
                            if float(res_dense_norm) < float(res_norm_full):
                                x_full = x_full_dense
                                ax_full = ax_dense
                                res_norm_full = res_dense_norm
                                dense_used = True
                                solver_kind_used = "dense"
                                solve_method_used = "dense"
                        except Exception as exc:  # noqa: BLE001
                            if emit is not None:
                                emit(
                                    1,
                                    "solve_v3_transport_matrix_linear_gmres: dense fallback failed "
                                    f"({type(exc).__name__}: {exc})",
                                )
                if (
                    dense_used
                    and dense_batch_fallback_enabled
                    and (not dense_batch_done)
                    and dense_retry_max > 0
                    and int(active_size) <= int(dense_retry_max)
                ):
                    if _dense_batch_solve_all(op_probe_ref=op_matvec_by_index[0], reason="dense fallback"):
                        dense_batch_done = True
                        break
                finalize_reduced_transport_rhs(
                    context=transport_rhs_finalization_context,
                    which_rhs=int(which_rhs),
                    result=res_reduced,
                    rhs_full=rhs,
                    op_matvec=op_matvec,
                    solver_kind=str(solver_kind_used),
                    solve_method=str(solve_method_used),
                    dense_used=bool(dense_used),
                    expand_reduced=expand_reduced,
                    reduce_full=reduce_full,
                    maybe_project_constraint_nullspace=_maybe_project_constraint_nullspace,
                    ksp_request=transport_rhs_finalization_context.ksp_request(
                        mv_reduced,
                        rhs_reduced,
                        preconditioner_used,
                        x0_used,
                        tol_val=float(tol_rhs),
                        restart_val=int(restart_used),
                        solver_kind=str(solver_kind_used),
                    ),
                    accepted_x_full=x_full,
                    accepted_ax_full=ax_full,
                    accepted_residual_norm=res_norm_full,
                )
            else:
                mv = _get_full_matvec(op_matvec)

                preconditioner_use = preconditioner_full
                if dense_precond_enabled:
                    sig = _operator_signature_cached(op_matvec)
                    preconditioner_use = _dense_preconditioner_for_matvec(
                        matvec_fn=mv,
                        n=int(op0.total_size),
                        dtype=_dense_dtype(rhs.dtype),
                        cache=dense_precond_cache_full,
                        key=(sig, int(op0.total_size)),
                    )
                x0_full = x0_by_rhs.get(int(which_rhs)) if x0_by_rhs else x0
                if recycle_k > 0:
                    x0_recycled = recycle_state.candidate_full(rhs)
                    if x0_full is None and x0_recycled is not None:
                        x0_full = x0_recycled

                solver_kind_used = _solver_kind(solve_method_rhs)[0]
                solve_method_used = solve_method_rhs
                restart_used = _restart_for_method(solve_method_rhs)
                preconditioner_used = preconditioner_use
                x0_used = x0_full
                dense_used = False
                structured_tzfft_first_attempt = False
                initial_solve_method_rhs = solve_method_rhs
                initial_restart_used = _restart_for_method(solve_method_rhs)
                initial_maxiter = maxiter
                if (
                    structured_tzfft_first_auto
                    and precond_kind_used == "tzfft"
                    and preconditioner_use is not None
                    and str(solve_method_rhs).strip().lower()
                    in {"auto", "default", "batched", "bicgstab", "bicgstab_jax", "incremental"}
                ):
                    structured_tzfft_first_attempt = True
                    initial_solve_method_rhs, initial_restart_used, initial_maxiter = (
                        _transport_tzfft_first_attempt_budget(
                            restart=int(gmres_restart),
                            maxiter=maxiter,
                        )
                    )
                    solver_kind_used = "gmres"
                    solve_method_used = initial_solve_method_rhs
                    restart_used = int(initial_restart_used)
                    if emit is not None:
                        emit(
                            1,
                            "solve_v3_transport_matrix_linear_gmres: structured tzfft first attempt "
                            f"whichRHS={int(which_rhs)} size={int(op0.total_size)} "
                            f"restart={int(initial_restart_used)} maxiter={int(initial_maxiter)}",
                        )
                target_rhs = max(float(atol), float(tol_rhs) * float(jnp.linalg.norm(rhs)))
                host_gmres_first_attempt = _transport_host_gmres_first_attempt_allowed(
                    op=op0,
                    size=int(op0.total_size),
                    use_implicit=bool(use_implicit),
                )
                sparse_direct_first_attempt = _transport_sparse_direct_first_attempt_allowed(
                    op=op0,
                    size=int(op0.total_size),
                    use_implicit=bool(use_implicit),
                )
                if host_gmres_first_attempt:
                    if emit is not None:
                        emit(
                            1,
                            "solve_v3_transport_matrix_linear_gmres: host SciPy GMRES first attempt "
                            f"(size={int(op0.total_size)} backend={jax.default_backend()})",
                        )
                    try:
                        res, residual_vec = _transport_host_gmres_solve(
                            op=op0,
                            matvec_fn=mv,
                            b_vec=rhs,
                            x0_vec=x0_full,
                            preconditioner_fn=preconditioner_use,
                            tol_val=tol_rhs,
                            atol_val=atol,
                            restart_val=initial_restart_used,
                            maxiter_val=initial_maxiter,
                            precondition_side_val=transport_precondition_side,
                            emit=emit,
                            which_rhs=int(which_rhs),
                            progress_every=_transport_host_gmres_progress_every(),
                        )
                        solver_kind_used = "gmres_scipy"
                        solve_method_used = "incremental"
                        restart_used = initial_restart_used
                    except Exception as exc:  # noqa: BLE001
                        if emit is not None:
                            emit(
                                1,
                                "solve_v3_transport_matrix_linear_gmres: host SciPy GMRES first attempt failed "
                                f"({type(exc).__name__}: {exc})",
                            )
                        res, residual_vec = _solve_linear_with_residual(
                            matvec_fn=mv,
                            b_vec=rhs,
                            x0_vec=x0_full,
                            tol_val=tol_rhs,
                            atol_val=atol,
                            restart_val=initial_restart_used,
                            maxiter_val=initial_maxiter,
                            solve_method_val=initial_solve_method_rhs,
                            preconditioner_val=preconditioner_use,
                            precondition_side_val=transport_precondition_side,
                        )
                elif sparse_direct_first_attempt:
                    if emit is not None:
                        emit(
                            1,
                            "solve_v3_transport_matrix_linear_gmres: host sparse LU first attempt "
                            f"(size={int(op0.total_size)} backend={jax.default_backend()})",
                        )
                    try:
                        sig = _operator_signature_cached(op_matvec)
                        res = transport_sparse_direct_context.solve(
                            matvec_fn=mv,
                            b_vec=rhs,
                            n=int(op0.total_size),
                            dtype=rhs.dtype,
                            cache_key=("transport_sparse_lu", sig, int(op0.total_size), "full"),
                            active_indices_np=None,
                            tol_val=tol_rhs,
                            atol_val=atol,
                            restart_val=initial_restart_used,
                            maxiter_val=initial_maxiter,
                            precondition_side_val=transport_precondition_side,
                        )
                        residual_vec = None
                        solver_kind_used = "sparse_lu"
                        solve_method_used = "sparse_lu"
                        restart_used = 0
                        preconditioner_used = None
                        x0_used = None
                    except Exception as exc:  # noqa: BLE001
                        if emit is not None:
                            emit(
                                1,
                                "solve_v3_transport_matrix_linear_gmres: host sparse LU first attempt failed "
                                f"({type(exc).__name__}: {exc})",
                            )
                        res, residual_vec = _solve_linear_with_residual(
                            matvec_fn=mv,
                            b_vec=rhs,
                            x0_vec=x0_full,
                            tol_val=tol_rhs,
                            atol_val=atol,
                            restart_val=initial_restart_used,
                            maxiter_val=initial_maxiter,
                            solve_method_val=initial_solve_method_rhs,
                            preconditioner_val=preconditioner_use,
                            precondition_side_val=transport_precondition_side,
                        )
                else:
                    res, residual_vec = _solve_linear_with_residual(
                        matvec_fn=mv,
                        b_vec=rhs,
                        x0_vec=x0_full,
                        tol_val=tol_rhs,
                        atol_val=atol,
                        restart_val=initial_restart_used,
                        maxiter_val=initial_maxiter,
                        solve_method_val=initial_solve_method_rhs,
                        preconditioner_val=preconditioner_use,
                        precondition_side_val=transport_precondition_side,
                    )
                solver_kind = _solver_kind(initial_solve_method_rhs)[0]
                if solver_kind == "bicgstab" and (not _gmres_result_is_finite(res) or float(res.residual_norm) > target_rhs):
                    if emit is not None:
                        emit(
                            0,
                            "solve_v3_transport_matrix_linear_gmres: BiCGStab fallback to GMRES "
                            f"(residual={float(res.residual_norm):.3e} > target={target_rhs:.3e})",
                        )
                    res, residual_vec = _solve_linear_with_residual(
                        matvec_fn=mv,
                        b_vec=rhs,
                        x0_vec=x0_full,
                        tol_val=tol_rhs,
                        atol_val=atol,
                        restart_val=gmres_restart,
                        maxiter_val=maxiter,
                        solve_method_val="incremental",
                        preconditioner_val=preconditioner_use,
                        precondition_side_val=transport_precondition_side,
                    )
                    solver_kind_used = "gmres"
                    solve_method_used = "incremental"
                    restart_used = gmres_restart
                sparse_direct_rescue = _transport_sparse_direct_rescue_allowed(
                    op=op0,
                    size=int(op0.total_size),
                    residual_norm=float(res.residual_norm),
                    target=float(target_rhs),
                    use_implicit=bool(use_implicit),
                )
                if structured_tzfft_first_attempt and _needs_retry(res, target_rhs):
                    sparse_direct_rescue = sparse_direct_rescue or _transport_sparse_direct_rescue_allowed(
                        op=op0,
                        size=int(op0.total_size),
                        residual_norm=float("nan"),
                        target=float(target_rhs),
                        use_implicit=bool(use_implicit),
                    )
                sparse_direct_rescue_first = _transport_sparse_direct_rescue_first(
                    sparse_direct_rescue=sparse_direct_rescue,
                )
                if sparse_direct_rescue_first and emit is not None:
                    emit(
                        1,
                        "solve_v3_transport_matrix_linear_gmres: sparse LU rescue-first "
                        "auto mode -> defer transport retry branches",
                    )
                if _needs_retry(res, target_rhs) and preconditioner_use is not None and (not sparse_direct_rescue_first):
                    if emit is not None:
                        emit(
                            0,
                            "solve_v3_transport_matrix_linear_gmres: retry without preconditioner "
                            f"(residual={float(res.residual_norm):.3e} > target={target_rhs:.3e})",
                        )
                    res_retry, residual_retry = _solve_linear_with_residual(
                        matvec_fn=mv,
                        b_vec=rhs,
                        x0_vec=x0_full,
                        tol_val=tol_rhs,
                        atol_val=atol,
                        restart_val=_restart_for_method(solve_method_rhs),
                        maxiter_val=maxiter,
                        solve_method_val=solve_method_rhs,
                        preconditioner_val=None,
                        precondition_side_val=transport_precondition_side,
                    )
                    if _residual_value(res_retry) < _residual_value(res):
                        res = res_retry
                        residual_vec = residual_retry
                        preconditioner_use = None
                        preconditioner_used = None
                if _needs_retry(res, target_rhs) and (not sparse_direct_rescue_first):
                    strong_precond = _get_strong_preconditioner(False)
                    if strong_precond is not None and strong_precond is not preconditioner_use:
                        if emit is not None:
                            emit(
                                0,
                                "solve_v3_transport_matrix_linear_gmres: retry with strong preconditioner "
                                f"(residual={float(res.residual_norm):.3e} > target={target_rhs:.3e})",
                            )
                        res_strong, residual_vec_strong = _solve_linear_with_residual(
                            matvec_fn=mv,
                            b_vec=rhs,
                            x0_vec=res.x,
                            tol_val=tol_rhs,
                            atol_val=atol,
                            restart_val=gmres_restart,
                            maxiter_val=maxiter,
                            solve_method_val="incremental",
                            preconditioner_val=strong_precond,
                            precondition_side_val=transport_precondition_side,
                        )
                        if _residual_value(res_strong) < _residual_value(res):
                            res = res_strong
                            residual_vec = residual_vec_strong
                            preconditioner_use = strong_precond
                            preconditioner_used = strong_precond
                            solver_kind_used = "gmres"
                            solve_method_used = "incremental"
                            restart_used = gmres_restart
                if _needs_retry(res, target_rhs) and sparse_direct_rescue:
                    if emit is not None:
                        emit(
                            0,
                            "solve_v3_transport_matrix_linear_gmres: sparse LU direct rescue "
                            f"(size={int(op0.total_size)} residual={float(res.residual_norm):.3e} > target={target_rhs:.3e})",
                        )
                    try:
                        sig = _operator_signature_cached(op_matvec)
                        res_sparse = transport_sparse_direct_context.solve(
                            matvec_fn=mv,
                            b_vec=rhs,
                            n=int(op0.total_size),
                            dtype=rhs.dtype,
                            cache_key=("transport_sparse_lu", sig, int(op0.total_size), "full"),
                            active_indices_np=None,
                            tol_val=tol_rhs,
                            atol_val=atol,
                            restart_val=_restart_for_method(solve_method_rhs),
                            maxiter_val=maxiter,
                            precondition_side_val=transport_precondition_side,
                        )
                        if _residual_value(res_sparse) < _residual_value(res):
                            res = res_sparse
                            residual_vec = None
                            preconditioner_use = None
                            preconditioner_used = None
                            solver_kind_used = "sparse_lu"
                            solve_method_used = "sparse_lu"
                    except Exception as exc:  # noqa: BLE001
                        if emit is not None:
                            emit(
                                1,
                                "solve_v3_transport_matrix_linear_gmres: sparse LU direct rescue failed "
                                f"({type(exc).__name__}: {exc})",
                            )
                if _needs_retry(res, target_rhs) and dense_retry_max > 0 and int(op0.total_size) <= int(dense_retry_max):
                    if emit is not None:
                        emit(
                            0,
                            "solve_v3_transport_matrix_linear_gmres: dense fallback "
                            f"(size={int(op0.total_size)} residual={float(res.residual_norm):.3e} > target={target_rhs:.3e})",
                        )
                    try:
                        sig = _operator_signature_cached(op_matvec)
                        dense_solver = _dense_solver_for_matvec(
                            matvec_fn=mv,
                            n=int(op0.total_size),
                            dtype=_dense_dtype(rhs.dtype),
                            cache=dense_solver_cache_full,
                            key=(sig, int(op0.total_size), str(_dense_dtype(rhs.dtype))),
                        )
                        rhs_dense = jnp.asarray(rhs, dtype=_dense_dtype(rhs.dtype))
                        x_dense = dense_solver(rhs_dense)
                        if dense_use_mixed:
                            r_dense0 = rhs - mv(jnp.asarray(x_dense, dtype=rhs.dtype))
                            dx = dense_solver(jnp.asarray(r_dense0, dtype=_dense_dtype(rhs.dtype)))
                            x_dense = jnp.asarray(x_dense, dtype=rhs.dtype) + jnp.asarray(dx, dtype=rhs.dtype)
                        residual_dense = rhs - mv(x_dense)
                        res_dense = GMRESSolveResult(x=x_dense, residual_norm=jnp.linalg.norm(residual_dense))
                        if _residual_value(res_dense) < _residual_value(res):
                            res = res_dense
                            residual_vec = residual_dense
                            dense_used = True
                            solver_kind_used = "dense"
                            solve_method_used = "dense"
                    except Exception as exc:  # noqa: BLE001
                        if emit is not None:
                            emit(
                                1,
                                "solve_v3_transport_matrix_linear_gmres: dense fallback failed "
                                f"({type(exc).__name__}: {exc})",
                            )
                polish_config = transport_polish_config_from_env(
                    rhs_mode=int(rhs_mode),
                    residual_norm=_residual_value(res),
                    target=float(target_rhs),
                    gmres_restart=int(gmres_restart),
                    maxiter=maxiter,
                )
                if _needs_retry(res, target_rhs) and polish_config.enabled:
                    polish_precond = _get_strong_preconditioner(False)
                    if polish_precond is None:
                        polish_precond = preconditioner_use
                    if emit is not None:
                        emit(
                            0,
                            "solve_v3_transport_matrix_linear_gmres: polish solve for RHSMode=3 "
                            f"(residual={float(res.residual_norm):.3e} > "
                            f"max({polish_config.ratio:.1f}x target, {polish_config.abs_tol:.1e}), "
                            f"restart={polish_config.restart} maxiter={polish_config.maxiter})",
                        )
                    res_polish, residual_polish = _solve_linear_with_residual(
                        matvec_fn=mv,
                        b_vec=rhs,
                        x0_vec=res.x,
                        tol_val=tol_rhs,
                        atol_val=atol,
                        restart_val=int(polish_config.restart),
                        maxiter_val=int(polish_config.maxiter),
                        solve_method_val="incremental",
                        preconditioner_val=polish_precond,
                        precondition_side_val=transport_precondition_side,
                    )
                    if transport_candidate_is_better(candidate=res_polish, current=res):
                        res = res_polish
                        residual_vec = residual_polish
                        preconditioner_used = polish_precond
                        solver_kind_used = "gmres"
                        solve_method_used = "incremental"
                        restart_used = int(polish_config.restart)
                if (
                    dense_used
                    and dense_batch_fallback_enabled
                    and (not dense_batch_done)
                    and dense_retry_max > 0
                    and int(op0.total_size) <= int(dense_retry_max)
                ):
                    if _dense_batch_solve_all(op_probe_ref=op_matvec_by_index[0], reason="dense fallback"):
                        dense_batch_done = True
                        break
                projection_needed = per_rhs_loop_policy.projection_needed(which_rhs)
                finalize_full_transport_rhs(
                    context=transport_rhs_finalization_context,
                    which_rhs=int(which_rhs),
                    result=res,
                    rhs_full=rhs,
                    op_matvec=op_matvec,
                    solver_kind=str(solver_kind_used),
                    solve_method=str(solve_method_used),
                    dense_used=bool(dense_used),
                    projection_needed=bool(projection_needed),
                    residual_vec=residual_vec,
                    maybe_project_constraint_nullspace=_maybe_project_constraint_nullspace,
                    ksp_request=transport_rhs_finalization_context.ksp_request(
                        mv,
                        rhs,
                        preconditioner_used,
                        x0_used,
                        tol_val=float(tol_rhs),
                        restart_val=int(restart_used),
                        solver_kind=str(solver_kind_used),
                    ),
                )
            transport_loop_progress.finish_rhs(
                which_rhs=int(which_rhs),
                rhs_elapsed_s=float(t_rhs.elapsed_s()),
                total_elapsed_s=float(t_all.elapsed_s()),
            )

    if emit is not None:
        emit(0, "solve_v3_transport_matrix_linear_gmres: computing whichRHS diagnostics (batched)")
    postsolve_diagnostics = compute_transport_postsolve_diagnostics(
        op0=op0,
        geom=geom,
        state_vectors=state_vectors,
        which_rhs_values=which_rhs_values,
        stream_diagnostics=bool(stream_diagnostics),
        streaming_outputs=streaming_outputs,
        use_diag_op0=bool(use_diag_op0),
        diag_op_by_index=diag_op_by_index,
        emit=None,
    )
    tm = postsolve_diagnostics.transport_matrix
    diag_pf_jnp = postsolve_diagnostics.particle_flux_vm_psi_hat
    diag_hf_jnp = postsolve_diagnostics.heat_flux_vm_psi_hat
    diag_flow_jnp = postsolve_diagnostics.fsab_flow
    transport_output_fields = postsolve_diagnostics.transport_output_fields
    if state_out_env:
        try:
            from sfincs_jax.solvers.diagnostics import save_krylov_state  # noqa: PLC0415

            save_krylov_state(path=state_out_env, op=op0, x_by_rhs=state_vectors)
        except Exception:
            if emit is not None:
                emit(1, f"solve_v3_transport_matrix_linear_gmres: failed to write state {state_out_env}")
    if emit is not None:
        emit(0, "solve_v3_transport_matrix_linear_gmres: done")
        emit(1, f"solve_v3_transport_matrix_linear_gmres: elapsed_s={t_all.elapsed_s():.3f}")
    return V3TransportMatrixSolveResult(
        op0=op0,
        transport_matrix=tm,
        state_vectors_by_rhs=state_vectors,
        residual_norms_by_rhs=residual_norms,
        fsab_flow=diag_flow_jnp,
        particle_flux_vm_psi_hat=diag_pf_jnp,
        heat_flux_vm_psi_hat=diag_hf_jnp,
        elapsed_time_s=jnp.asarray(elapsed_s, dtype=jnp.float64),
        transport_output_fields=transport_output_fields,
        rhs_norms_by_rhs=rhs_norms,
        active_size=int(active_size),
        use_active_dof_mode=bool(use_active_dof_mode),
        solver_kinds_by_rhs=solver_kinds_by_rhs,
        solve_methods_by_rhs=solve_methods_by_rhs,
        preconditioner_kind=precond_kind_used,
        strong_preconditioner_kind=strong_precond_kind,
    )

# Optional host-side Krylov iteration diagnostics.
EmitFn = Callable[[int, str], None]


def emit_transport_ksp_iteration_stats(
    *,
    which_rhs: int,
    matvec_fn: Callable[[jnp.ndarray], jnp.ndarray],
    b_vec: jnp.ndarray,
    precond_fn: Callable[[jnp.ndarray], jnp.ndarray] | None,
    x0_vec: jnp.ndarray | None,
    tol_val: float,
    atol_val: float,
    restart_val: int,
    maxiter_val: int | None,
    precond_side: str,
    solver_kind: str,
    emit: EmitFn | None,
    enabled: bool,
    max_size: int | None,
) -> None:
    """Emit optional SciPy KSP iteration counts without affecting the solve.

    The diagnostics re-run the requested Krylov method on the host for small
    systems only.  Any diagnostic failure is reported and swallowed so that the
    production transport solve remains the source of truth.
    """
    if emit is None or not enabled:
        return
    size = int(b_vec.size)
    if max_size is not None and size > int(max_size):
        emit(1, f"whichRHS={which_rhs} ksp_iterations skipped (size={size} > max={int(max_size)})")
        return
    solver_kind_l = str(solver_kind).strip().lower()
    try:
        history = _solve_history(
            solver_kind=solver_kind_l,
            matvec_fn=matvec_fn,
            b_vec=b_vec,
            precond_fn=precond_fn,
            x0_vec=x0_vec,
            tol_val=tol_val,
            atol_val=atol_val,
            restart_val=restart_val,
            maxiter_val=maxiter_val,
            precond_side=precond_side,
        )
    except Exception as exc:  # noqa: BLE001
        emit(1, f"whichRHS={which_rhs} ksp_iterations unavailable ({type(exc).__name__}: {exc})")
        return
    if history is None:
        return
    emit(0, f"whichRHS={which_rhs} ksp_iterations={len(history)} solver={solver_kind_l}")


def _solve_history(
    *,
    solver_kind: str,
    matvec_fn: Callable[[jnp.ndarray], jnp.ndarray],
    b_vec: jnp.ndarray,
    precond_fn: Callable[[jnp.ndarray], jnp.ndarray] | None,
    x0_vec: jnp.ndarray | None,
    tol_val: float,
    atol_val: float,
    restart_val: int,
    maxiter_val: int | None,
    precond_side: str,
) -> list[Any] | None:
    if solver_kind == "gmres":
        _x_hist, _rn, history = gmres_solve_with_history_scipy(
            matvec=matvec_fn,
            b=b_vec,
            preconditioner=precond_fn,
            x0=x0_vec,
            tol=tol_val,
            atol=atol_val,
            restart=restart_val,
            maxiter=maxiter_val,
            precondition_side=precond_side,
        )
        return history
    if solver_kind == "bicgstab":
        _x_hist, _rn, history = bicgstab_solve_with_history_scipy(
            matvec=matvec_fn,
            b=b_vec,
            preconditioner=precond_fn,
            x0=x0_vec,
            tol=tol_val,
            atol=atol_val,
            maxiter=maxiter_val,
            precondition_side=precond_side,
        )
        return history
    return None


_emit_transport_ksp_iteration_stats = emit_transport_ksp_iteration_stats

# Loop-local matvec caching, recycle bases, and progress bookkeeping.
MatvecFn = Callable[[jnp.ndarray], jnp.ndarray]


@dataclass
class TransportMatvecCache:
    """Cache full and active-DOF transport matvec closures by operator signature."""

    use_active_dof_mode: bool
    active_size: int
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None
    apply_operator: Callable[[Any, jnp.ndarray], jnp.ndarray] = apply_v3_full_system_operator_cached
    operator_signature: Callable[[Any], tuple[object, ...]] = _operator_signature_cached
    full_cache: dict[tuple[object, ...], MatvecFn] = field(default_factory=dict)
    reduced_cache: dict[tuple[object, ...], MatvecFn] = field(default_factory=dict)

    def get_full(self, op_matvec: Any) -> MatvecFn:
        """Return a cached full-space matvec for ``op_matvec``."""
        signature = self.operator_signature(op_matvec)
        fn = self.full_cache.get(signature)
        if fn is None:

            def mv(x: jnp.ndarray, op=op_matvec) -> jnp.ndarray:
                return self.apply_operator(op, x)

            self.full_cache[signature] = mv
            fn = mv
        return fn

    def get_reduced(self, op_matvec: Any) -> MatvecFn:
        """Return a cached active-DOF matvec, or the full matvec when inactive."""
        if not self.use_active_dof_mode or self.reduce_full is None or self.expand_reduced is None:
            return self.get_full(op_matvec)
        signature = self.operator_signature(op_matvec)
        key = (signature, int(self.active_size))
        fn = self.reduced_cache.get(key)
        if fn is None:

            def mv(x_reduced: jnp.ndarray, op=op_matvec) -> jnp.ndarray:
                y_full = self.apply_operator(op, self.expand_reduced(x_reduced))
                return self.reduce_full(y_full)

            self.reduced_cache[key] = mv
            fn = mv
        return fn


def recycled_transport_initial_guess(
    rhs_vec: jnp.ndarray,
    basis: Sequence[jnp.ndarray],
    basis_au: Sequence[jnp.ndarray],
) -> jnp.ndarray | None:
    """Return a residual-minimizing recycled initial guess for one transport RHS."""
    return recycled_initial_guess(rhs_vec, basis, basis_au)


@dataclass
class TransportRecycleState:
    """Bounded recycled Krylov bases for full and active-DOF transport solves."""

    k: int
    full_basis: list[jnp.ndarray] = field(default_factory=list)
    full_basis_au: list[jnp.ndarray] = field(default_factory=list)
    reduced_basis: list[jnp.ndarray] = field(default_factory=list)
    reduced_basis_au: list[jnp.ndarray] = field(default_factory=list)

    @property
    def enabled(self) -> bool:
        """Whether recycle candidates should be used."""
        return int(self.k) > 0

    def candidate_full(self, rhs_vec: jnp.ndarray) -> jnp.ndarray | None:
        """Return a recycled full-space initial guess, if available."""
        if not self.enabled:
            return None
        return recycled_transport_initial_guess(rhs_vec, self.full_basis[-int(self.k) :], self.full_basis_au[-int(self.k) :])

    def candidate_reduced(self, rhs_vec: jnp.ndarray) -> jnp.ndarray | None:
        """Return a recycled active-DOF initial guess, if available."""
        if not self.enabled:
            return None
        return recycled_transport_initial_guess(
            rhs_vec,
            self.reduced_basis[-int(self.k) :],
            self.reduced_basis_au[-int(self.k) :],
        )

    def append_full(self, x_full: jnp.ndarray, ax_full: jnp.ndarray) -> None:
        """Append and trim one full-space recycle vector."""
        if not self.enabled:
            return
        self.full_basis.append(x_full)
        self.full_basis_au.append(ax_full)
        self._trim()

    def append_reduced(
        self,
        x_reduced: jnp.ndarray,
        ax_reduced: jnp.ndarray,
        *,
        x_full: jnp.ndarray | None = None,
        ax_full: jnp.ndarray | None = None,
    ) -> None:
        """Append and trim one reduced recycle vector and optional full vector."""
        if not self.enabled:
            return
        self.reduced_basis.append(x_reduced)
        self.reduced_basis_au.append(ax_reduced)
        if x_full is not None and ax_full is not None:
            self.full_basis.append(x_full)
            self.full_basis_au.append(ax_full)
        self._trim()

    def seed_from_state(
        self,
        *,
        state_x_by_rhs: Mapping[int, jnp.ndarray],
        total_size: int,
        active_size: int,
        matvec_cache: TransportMatvecCache,
        op_ref: Any,
    ) -> None:
        """Seed recycle bases from stored transport Krylov state vectors."""
        if not self.enabled:
            return
        mv_ref_full = matvec_cache.get_full(op_ref)
        mv_ref_reduced = matvec_cache.get_reduced(op_ref)
        for which_rhs in sorted(state_x_by_rhs.keys()):
            x_arr = jnp.asarray(state_x_by_rhs[int(which_rhs)])
            if x_arr.shape == (int(total_size),):
                self.full_basis.append(x_arr)
                self.full_basis_au.append(mv_ref_full(x_arr))
                if matvec_cache.use_active_dof_mode and matvec_cache.reduce_full is not None:
                    x_reduced = matvec_cache.reduce_full(x_arr)
                    self.reduced_basis.append(x_reduced)
                    self.reduced_basis_au.append(mv_ref_reduced(x_reduced))
            elif (
                matvec_cache.use_active_dof_mode
                and x_arr.shape == (int(active_size),)
                and matvec_cache.reduce_full is not None
            ):
                self.reduced_basis.append(x_arr)
                self.reduced_basis_au.append(mv_ref_reduced(x_arr))
        self._trim()

    def _trim(self) -> None:
        k = max(0, int(self.k))
        if k <= 0:
            self.full_basis.clear()
            self.full_basis_au.clear()
            self.reduced_basis.clear()
            self.reduced_basis_au.clear()
            return
        if len(self.full_basis) > k:
            self.full_basis = self.full_basis[-k:]
            self.full_basis_au = self.full_basis_au[-k:]
        if len(self.reduced_basis) > k:
            self.reduced_basis = self.reduced_basis[-k:]
            self.reduced_basis_au = self.reduced_basis_au[-k:]


@dataclass
class TransportLoopProgress:
    """Residual-gate and ETA bookkeeping for sequential transport RHS solves."""

    which_rhs_values: Sequence[int]
    rhs_norms: Mapping[int, Any]
    residual_norms: Mapping[int, Any]
    elapsed_s: Any
    abort_max_residual: float
    abort_max_relative_residual: float
    emit: EmitFn | None = None
    progress_message: Callable[..., str] = transport_progress_message
    residual_gate_failure: Callable[..., str | None] = transport_residual_gate_failure
    elapsed_history: list[float] = field(default_factory=list)

    def relative_residual(self, which_rhs: int) -> float:
        """Return the RHS-normalized residual for one completed transport drive."""
        rhs_norm_val = float(self.rhs_norms[int(which_rhs)])
        residual_norm_val = float(self.residual_norms[int(which_rhs)])
        if math.isfinite(rhs_norm_val) and rhs_norm_val > 0.0:
            return residual_norm_val / rhs_norm_val
        return float("nan")

    def residual_failure(self, which_rhs: int) -> str | None:
        """Return the configured residual-gate failure string, if any."""
        if self.abort_max_residual <= 0.0 and self.abort_max_relative_residual <= 0.0:
            return None
        return self.residual_gate_failure(
            which_rhs=int(which_rhs),
            residual_norm=float(self.residual_norms[int(which_rhs)]),
            rhs_norm=float(self.rhs_norms[int(which_rhs)]),
            max_abs=float(self.abort_max_residual),
            max_relative=float(self.abort_max_relative_residual),
        )

    def finish_rhs(self, *, which_rhs: int, rhs_elapsed_s: float, total_elapsed_s: float) -> None:
        """Record, gate, and report one completed sequential transport RHS solve."""
        which_rhs_i = int(which_rhs)
        elapsed = float(rhs_elapsed_s)
        if self.emit is not None:
            rhs_norm_val = float(self.rhs_norms[which_rhs_i])
            residual_norm_val = float(self.residual_norms[which_rhs_i])
            self.emit(
                0,
                f"whichRHS={which_rhs_i}: residual_norm={residual_norm_val:.6e} "
                f"rhs_norm={rhs_norm_val:.6e} relative_residual={self.relative_residual(which_rhs_i):.6e} "
                f"elapsed_s={elapsed:.3f}",
            )
        self.elapsed_s[which_rhs_i - 1] = elapsed
        failure = self.residual_failure(which_rhs_i)
        if failure is not None:
            if self.emit is not None:
                self.emit(
                    1,
                    "solve_v3_transport_matrix_linear_gmres: transport residual gate failed; "
                    f"aborting remaining whichRHS solves ({failure})",
                )
            raise RuntimeError(f"transport residual gate failed: {failure}")
        self.elapsed_history.append(elapsed)
        if self.emit is not None:
            completed_rhs = len(self.elapsed_history)
            avg_rhs_s = float(sum(self.elapsed_history) / max(1, completed_rhs))
            self.emit(
                0,
                self.progress_message(
                    completed=completed_rhs,
                    total=len(self.which_rhs_values),
                    avg_rhs_s=avg_rhs_s,
                    elapsed_s=float(total_elapsed_s),
                ),
            )


def resolve_transport_recycle_k(
    *,
    op: Any,
    use_implicit: bool,
    op_matvec_by_index: Sequence[Any],
    disable_auto_recycle: Callable[..., bool],
    emit: EmitFn | None,
    operator_signature: Callable[[Any], tuple[object, ...]] = _operator_signature_cached,
) -> int:
    """Resolve the bounded recycle-basis size for the transport solve loop."""
    recycle_k_env = os.environ.get("SFINCS_JAX_TRANSPORT_RECYCLE_K", "").strip()
    try:
        recycle_k = int(recycle_k_env) if recycle_k_env else 4
    except ValueError:
        recycle_k = 4
    recycle_k = max(0, int(recycle_k))
    if recycle_k > 0 and disable_auto_recycle(op=op, use_implicit=bool(use_implicit)):
        recycle_k = 0
        if emit is not None:
            emit(
                1,
                "solve_v3_transport_matrix_linear_gmres: auto recycle disabled "
                "for branch-sensitive explicit mono transport",
            )
    if recycle_k > 0 and op_matvec_by_index:
        signature_ref = operator_signature(op_matvec_by_index[0])
        for op_probe in op_matvec_by_index[1:]:
            if operator_signature(op_probe) != signature_ref:
                recycle_k = 0
                if emit is not None:
                    emit(
                        1,
                        "solve_v3_transport_matrix_linear_gmres: recycle disabled "
                        "(matvec operator varies across whichRHS)",
                    )
                break
    return int(recycle_k)


# Sparse-direct rescue helpers for RHSMode=2/3 transport solves.
@dataclass
class TransportSparseDirectContext:
    """Driver-owned state needed by the transport sparse-direct rescue path."""

    op: Any
    factor_cache: MutableMapping[tuple[object, ...], tuple[object, object, str, str]]
    pattern_cache: MutableMapping[tuple[object, ...], object]
    sparse_drop_tol: float
    sparse_drop_rel: float
    emit: EmitFn | None
    sparse_factor_cache_key: Callable[[tuple[object, ...], np.dtype], tuple[object, ...]]
    hash_numpy_array_for_cache: Callable[[np.ndarray], object]
    build_host_sparse_direct_factor_from_matvec: Callable[..., tuple[Any, Any]]
    build_sparse_ilu_from_matvec: Callable[..., tuple[Any, Any, Any, Any, Any, Any, Any]]
    try_build_direct_active_operator_bundle: Callable[..., tuple[Any, Any] | None]
    host_sparse_direct_solve_with_refinement: Callable[..., tuple[np.ndarray, float]]
    host_sparse_direct_refine_steps: Callable[..., int]
    host_sparse_direct_polish: Callable[..., tuple[np.ndarray, float]]
    sparse_factor_dtype: Callable[..., np.dtype]
    sparse_direct_use_explicit_helper: Callable[..., bool]
    sparse_direct_needs_float64_retry: Callable[..., bool]

    def pattern_for_solve(
        self,
        *,
        n: int,
        active_indices_np: np.ndarray | None,
    ) -> object | None:
        """Return the conservative sparse pattern admitted by this context."""
        return transport_sparse_direct_pattern_for_solve(
            context=self,
            n=int(n),
            active_indices_np=active_indices_np,
        )

    def solve(
        self,
        *,
        matvec_fn: Callable[[jnp.ndarray], jnp.ndarray],
        b_vec: jnp.ndarray,
        n: int,
        dtype: jnp.dtype,
        cache_key: tuple[object, ...],
        active_indices_np: np.ndarray | None,
        tol_val: float,
        atol_val: float,
        restart_val: int,
        maxiter_val: int | None,
        precondition_side_val: str,
    ) -> GMRESSolveResult:
        """Run this context's sparse-direct rescue path."""
        return transport_sparse_direct_solve(
            context=self,
            matvec_fn=matvec_fn,
            b_vec=b_vec,
            n=int(n),
            dtype=dtype,
            cache_key=cache_key,
            active_indices_np=active_indices_np,
            tol_val=float(tol_val),
            atol_val=float(atol_val),
            restart_val=int(restart_val),
            maxiter_val=maxiter_val,
            precondition_side_val=str(precondition_side_val),
        )


def transport_sparse_direct_context_from_env(
    *,
    op: Any,
    emit: EmitFn | None,
    sparse_factor_cache_key: Callable[[tuple[object, ...], np.dtype], tuple[object, ...]],
    hash_numpy_array_for_cache: Callable[[np.ndarray], object],
    build_host_sparse_direct_factor_from_matvec: Callable[..., tuple[Any, Any]],
    build_sparse_ilu_from_matvec: Callable[..., tuple[Any, Any, Any, Any, Any, Any, Any]],
    try_build_direct_active_operator_bundle: Callable[..., tuple[Any, Any] | None],
    host_sparse_direct_solve_with_refinement: Callable[..., tuple[np.ndarray, float]],
    host_sparse_direct_refine_steps: Callable[..., int],
    host_sparse_direct_polish: Callable[..., tuple[np.ndarray, float]],
    sparse_factor_dtype: Callable[..., np.dtype],
    sparse_direct_use_explicit_helper: Callable[..., bool],
    sparse_direct_needs_float64_retry: Callable[..., bool],
) -> TransportSparseDirectContext:
    """Create the per-solve sparse-direct context and caches from env policy."""
    return TransportSparseDirectContext(
        op=op,
        factor_cache={},
        pattern_cache={},
        sparse_drop_tol=_read_float_env("SFINCS_JAX_TRANSPORT_SPARSE_DROP_TOL", default=0.0),
        sparse_drop_rel=_read_float_env("SFINCS_JAX_TRANSPORT_SPARSE_DROP_REL", default=0.0),
        emit=emit,
        sparse_factor_cache_key=sparse_factor_cache_key,
        hash_numpy_array_for_cache=hash_numpy_array_for_cache,
        build_host_sparse_direct_factor_from_matvec=build_host_sparse_direct_factor_from_matvec,
        build_sparse_ilu_from_matvec=build_sparse_ilu_from_matvec,
        try_build_direct_active_operator_bundle=try_build_direct_active_operator_bundle,
        host_sparse_direct_solve_with_refinement=host_sparse_direct_solve_with_refinement,
        host_sparse_direct_refine_steps=host_sparse_direct_refine_steps,
        host_sparse_direct_polish=host_sparse_direct_polish,
        sparse_factor_dtype=sparse_factor_dtype,
        sparse_direct_use_explicit_helper=sparse_direct_use_explicit_helper,
        sparse_direct_needs_float64_retry=sparse_direct_needs_float64_retry,
    )


def transport_sparse_direct_pattern_for_solve(
    *,
    context: TransportSparseDirectContext,
    n: int,
    active_indices_np: np.ndarray | None,
) -> object | None:
    """Return the conservative transport sparse pattern when policy admits it."""
    op = context.op
    raw = os.environ.get("SFINCS_JAX_TRANSPORT_SPARSE_PATTERN", "").strip().lower()
    if raw in {"0", "false", "no", "off", "dense", "matvec"}:
        return None
    force_pattern = raw in {"1", "true", "yes", "on", "pattern", "probe", "color_probe"}
    mono_pas_transport = (
        int(op.rhs_mode) == 3
        and not bool(op.include_phi1)
        and getattr(op.fblock, "fp", None) is None
        and int(getattr(op, "n_x", 0) or 0) <= 2
    )
    if not (force_pattern or mono_pas_transport):
        return None
    active_key = "full" if active_indices_np is None else context.hash_numpy_array_for_cache(active_indices_np)
    cache_key = ("transport_sparse_pattern", int(n), active_key)
    cached_pattern = context.pattern_cache.get(cache_key)
    if cached_pattern is not None:
        return cached_pattern
    if active_indices_np is None:
        if int(n) != int(op.total_size):
            return None
        pattern = v3_full_system_conservative_sparsity_pattern(op)
    else:
        active_np = np.asarray(active_indices_np, dtype=np.int32).reshape((-1,))
        if int(n) != int(active_np.size):
            return None
        pattern = v3_full_system_conservative_sparsity_pattern_for_indices(op, active_np)
    summary = summarize_v3_sparse_pattern(op, pattern)
    csr_estimate_mb = float(estimate_csr_nbytes(summary.shape, summary.nnz)) / 1.0e6
    max_mb_env = os.environ.get("SFINCS_JAX_TRANSPORT_SPARSE_PATTERN_CSR_MAX_MB", "").strip()
    try:
        max_mb = float(max_mb_env) if max_mb_env else 512.0
    except ValueError:
        max_mb = 512.0
    if csr_estimate_mb > max(0.0, float(max_mb)):
        message = (
            "transport sparse-pattern assembly exceeds CSR budget "
            f"({csr_estimate_mb:.3f} MB > {float(max_mb):.3f} MB, nnz={summary.nnz})"
        )
        if force_pattern:
            raise MemoryError(message)
        if context.emit is not None:
            context.emit(1, f"solve_v3_transport_matrix_linear_gmres: {message}; using matvec probing")
        return None
    context.pattern_cache[cache_key] = pattern
    if context.emit is not None:
        context.emit(
            1,
            "solve_v3_transport_matrix_linear_gmres: transport sparse pattern selected "
            f"shape={summary.shape} nnz={summary.nnz} avg_row_nnz={summary.avg_row_nnz:.3f} "
            f"max_row_nnz={summary.max_row_nnz} csr_estimate_mb={csr_estimate_mb:.3f}",
        )
    return pattern


def transport_sparse_direct_solve(
    *,
    context: TransportSparseDirectContext,
    matvec_fn: Callable[[jnp.ndarray], jnp.ndarray],
    b_vec: jnp.ndarray,
    n: int,
    dtype: jnp.dtype,
    cache_key: tuple[object, ...],
    active_indices_np: np.ndarray | None,
    tol_val: float,
    atol_val: float,
    restart_val: int,
    maxiter_val: int | None,
    precondition_side_val: str,
) -> GMRESSolveResult:
    """Run the RHSMode=2/3 sparse-direct rescue with true-residual checks."""
    factor_dtype = context.sparse_factor_dtype(size=int(n), use_implicit=False)
    if context.emit is not None:
        context.emit(
            1,
            "solve_v3_transport_matrix_linear_gmres: sparse LU factor_dtype="
            f"{np.dtype(factor_dtype).name}",
        )
    target_true = max(float(atol_val), float(tol_val) * float(jnp.linalg.norm(b_vec)))
    x_np, residual_norm, ilu_for_polish = _solve_with_factor_dtype(
        context=context,
        matvec_fn=matvec_fn,
        b_vec=b_vec,
        n=int(n),
        dtype=dtype,
        cache_key=cache_key,
        active_indices_np=active_indices_np,
        factor_dtype_use=np.dtype(factor_dtype),
    )

    def true_residual_norm(x_arr: np.ndarray) -> float:
        ax = matvec_fn(jnp.asarray(x_arr, dtype=dtype))
        residual = np.asarray(ax - b_vec, dtype=np.float64).reshape((-1,))
        return float(np.linalg.norm(residual))

    true_residual = true_residual_norm(x_np)
    if np.isfinite(true_residual) and (
        (not np.isfinite(float(residual_norm))) or float(true_residual) > float(residual_norm)
    ):
        residual_norm = float(true_residual)
    if np.dtype(factor_dtype) == np.dtype(np.float32) and residual_norm > target_true:
        x_np, residual_norm = _maybe_polish_float32_factor(
            context=context,
            matvec_fn=matvec_fn,
            b_vec=b_vec,
            x_np=x_np,
            residual_norm=float(residual_norm),
            ilu_for_polish=ilu_for_polish,
            factor_dtype=np.dtype(factor_dtype),
            tol_val=float(tol_val),
            atol_val=float(atol_val),
            restart_val=int(restart_val),
            maxiter_val=maxiter_val,
            precondition_side_val=str(precondition_side_val),
            true_residual_norm=true_residual_norm,
        )
    if context.sparse_direct_needs_float64_retry(
        factor_dtype=np.dtype(factor_dtype),
        residual_norm=float(residual_norm),
        target_true=float(target_true),
    ):
        if context.emit is not None:
            context.emit(
                1,
                "solve_v3_transport_matrix_linear_gmres: retrying sparse LU with float64 factors "
                f"(residual={float(residual_norm):.6e}, target={float(target_true):.6e})",
            )
        x64_np, residual64, _ilu64 = _solve_with_factor_dtype(
            context=context,
            matvec_fn=matvec_fn,
            b_vec=b_vec,
            n=int(n),
            dtype=dtype,
            cache_key=cache_key,
            active_indices_np=active_indices_np,
            factor_dtype_use=np.dtype(np.float64),
        )
        if np.isfinite(residual64) and (
            not np.isfinite(float(residual_norm)) or float(residual64) < float(residual_norm)
        ):
            x_np = x64_np
            residual_norm = residual64
    return GMRESSolveResult(
        x=jnp.asarray(x_np, dtype=jnp.float64),
        residual_norm=jnp.asarray(residual_norm, dtype=jnp.float64),
    )


def _solve_with_factor_dtype(
    *,
    context: TransportSparseDirectContext,
    matvec_fn: Callable[[jnp.ndarray], jnp.ndarray],
    b_vec: jnp.ndarray,
    n: int,
    dtype: jnp.dtype,
    cache_key: tuple[object, ...],
    active_indices_np: np.ndarray | None,
    factor_dtype_use: np.dtype,
) -> tuple[np.ndarray, float, object]:
    direct_true_attempted, a_csr_full, ilu = _maybe_build_direct_active_true_factor(
        context=context,
        active_indices_np=active_indices_np,
        n=int(n),
        cache_key=cache_key,
        factor_dtype_use=np.dtype(factor_dtype_use),
    )
    if not direct_true_attempted:
        pattern = transport_sparse_direct_pattern_for_solve(
            context=context,
            n=int(n),
            active_indices_np=active_indices_np,
        )
    else:
        pattern = None
    if (not direct_true_attempted) and pattern is not None:
        a_csr_full, ilu = _build_pattern_factor(
            context=context,
            matvec_fn=matvec_fn,
            n=int(n),
            dtype=dtype,
            cache_key=cache_key,
            factor_dtype_use=np.dtype(factor_dtype_use),
            pattern=pattern,
        )
    elif (not direct_true_attempted) and context.sparse_direct_use_explicit_helper(size=int(n)):
        a_csr_full, ilu = _build_explicit_helper_factor(
            context=context,
            matvec_fn=matvec_fn,
            n=int(n),
            dtype=dtype,
            cache_key=cache_key,
            factor_dtype_use=np.dtype(factor_dtype_use),
        )
    elif not direct_true_attempted:
        cache_key_use = context.sparse_factor_cache_key(cache_key, np.dtype(factor_dtype_use))
        a_csr_full, _a_csr_drop, ilu, _a_dense, _l_dense, _u_dense, _l_unit = context.build_sparse_ilu_from_matvec(
            matvec=matvec_fn,
            n=int(n),
            dtype=dtype,
            cache_key=cache_key_use,
            factor_dtype=np.dtype(factor_dtype_use),
            drop_tol=float(context.sparse_drop_tol),
            drop_rel=float(context.sparse_drop_rel),
            ilu_drop_tol=0.0,
            fill_factor=1.0,
            build_dense_factors=False,
            build_jax_factors=False,
            build_ilu=True,
            store_dense=False,
            factorization="lu",
            emit=context.emit,
        )
        if ilu is None:
            raise RuntimeError("transport sparse_lu: factors unavailable")
    x_local, residual_local = context.host_sparse_direct_solve_with_refinement(
        ilu=ilu,
        a_csr_full=a_csr_full,
        rhs_vec=b_vec,
        factor_dtype=np.dtype(factor_dtype_use),
        refine_steps=context.host_sparse_direct_refine_steps(
            "SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_REFINE",
            default=2,
        ),
    )
    return x_local, float(residual_local), ilu


def _maybe_build_direct_active_true_factor(
    *,
    context: TransportSparseDirectContext,
    active_indices_np: np.ndarray | None,
    n: int,
    cache_key: tuple[object, ...],
    factor_dtype_use: np.dtype,
) -> tuple[bool, object | None, object | None]:
    op = context.op
    direct_true_enabled_env = os.environ.get("SFINCS_JAX_TRANSPORT_FP_DIRECT_ACTIVE_OPERATOR", "").strip().lower()
    direct_true_enabled = direct_true_enabled_env not in {"0", "false", "no", "off"}
    if not (
        bool(direct_true_enabled)
        and int(op.rhs_mode) in {2, 3}
        and getattr(op.fblock, "fp", None) is not None
        and not bool(op.include_phi1)
        and active_indices_np is not None
        and int(n) == int(np.asarray(active_indices_np).size)
    ):
        return False, None, None

    direct_factor_kind, direct_ilu_fill, direct_ilu_drop = _direct_active_factor_options(n=int(n))
    active_hash_direct = context.hash_numpy_array_for_cache(np.asarray(active_indices_np, dtype=np.int64))
    factor_cache_key = (
        *context.sparse_factor_cache_key(cache_key, np.dtype(factor_dtype_use)),
        "direct_active_true_fp_operator",
        str(jax.default_backend()),
        str(active_hash_direct),
        str(direct_factor_kind),
        float(direct_ilu_fill),
        float(direct_ilu_drop),
    )
    cached_factor = context.factor_cache.get(factor_cache_key)
    if cached_factor is not None:
        a_csr_full, ilu, storage_kind, reason = cached_factor
        if context.emit is not None:
            context.emit(
                1,
                "solve_v3_transport_matrix_linear_gmres: reusing direct active true FP operator "
                f"storage={storage_kind} reason={reason}",
            )
        return True, a_csr_full, ilu

    direct_result = context.try_build_direct_active_operator_bundle(
        op=op,
        active_indices=np.asarray(active_indices_np, dtype=np.int64),
        factor_dtype=np.dtype(factor_dtype_use),
        emit=context.emit,
    )
    if direct_result is None:
        return False, None, None
    operator_bundle, _direct_metadata = direct_result
    factor_bundle = factorize_host_sparse_operator(
        operator_bundle,
        kind=direct_factor_kind,
        drop_tol=float(direct_ilu_drop) if direct_factor_kind == "ilu" else 0.0,
        fill_factor=float(direct_ilu_fill) if direct_factor_kind == "ilu" else 1.0,
        permc_spec="MMD_AT_PLUS_A",
        diag_pivot_thresh=0.0,
    )
    a_csr_full = factor_bundle.operator.matrix
    ilu = factor_bundle.factor
    context.factor_cache[factor_cache_key] = (
        a_csr_full,
        ilu,
        str(operator_bundle.metadata.storage_kind),
        str(operator_bundle.metadata.reason),
    )
    if context.emit is not None:
        context.emit(
            1,
            "solve_v3_transport_matrix_linear_gmres: direct active true FP operator "
            f"factorization complete factor_kind={factor_bundle.kind} "
            f"factor_s={float(factor_bundle.factor_s or 0.0):.3f} "
            f"factor_mb={float(factor_bundle.factor_nbytes_estimate or 0) / 1.0e6:.3f}",
        )
    return True, a_csr_full, ilu


def _direct_active_factor_options(*, n: int) -> tuple[str, float, float]:
    factor_kind_env = os.environ.get("SFINCS_JAX_TRANSPORT_FP_DIRECT_ACTIVE_OPERATOR_FACTOR", "").strip().lower()
    if factor_kind_env in {"lu", "exact"}:
        direct_factor_kind = "lu"
    elif factor_kind_env in {"ilu", "spilu", "incomplete"}:
        direct_factor_kind = "ilu"
    else:
        direct_factor_kind = "lu" if int(n) <= 50_000 else "ilu"
    fill_env = os.environ.get("SFINCS_JAX_TRANSPORT_FP_DIRECT_ACTIVE_OPERATOR_ILU_FILL", "").strip()
    drop_env = os.environ.get("SFINCS_JAX_TRANSPORT_FP_DIRECT_ACTIVE_OPERATOR_ILU_DROP_TOL", "").strip()
    try:
        direct_ilu_fill = float(fill_env) if fill_env else 6.0
    except ValueError:
        direct_ilu_fill = 6.0
    try:
        direct_ilu_drop = float(drop_env) if drop_env else 1.0e-4
    except ValueError:
        direct_ilu_drop = 1.0e-4
    return direct_factor_kind, direct_ilu_fill, direct_ilu_drop


def _build_pattern_factor(
    *,
    context: TransportSparseDirectContext,
    matvec_fn: Callable[[jnp.ndarray], jnp.ndarray],
    n: int,
    dtype: jnp.dtype,
    cache_key: tuple[object, ...],
    factor_dtype_use: np.dtype,
    pattern: object,
) -> tuple[object, object]:
    color_batch_env = os.environ.get("SFINCS_JAX_TRANSPORT_SPARSE_PATTERN_COLOR_BATCH", "").strip()
    try:
        color_batch = int(color_batch_env) if color_batch_env else 8
    except ValueError:
        color_batch = 8
    color_batch = max(1, int(color_batch))
    factor_cache_key = (
        *context.sparse_factor_cache_key(cache_key, np.dtype(factor_dtype_use)),
        "pattern_probe",
        int(getattr(pattern, "nnz", 0)),
        int(color_batch),
        str(jax.default_backend()),
    )
    cached_factor = context.factor_cache.get(factor_cache_key)
    if cached_factor is not None:
        a_csr_full, ilu, storage_kind, reason = cached_factor
        if context.emit is not None:
            context.emit(
                1,
                "solve_v3_transport_matrix_linear_gmres: reusing pattern sparse helper "
                f"storage={storage_kind} reason={reason}",
            )
        return a_csr_full, ilu
    operator_bundle, factor_bundle = context.build_host_sparse_direct_factor_from_matvec(
        matvec=matvec_fn,
        n=int(n),
        dtype=dtype,
        factor_dtype=np.dtype(factor_dtype_use),
        pattern=pattern,
        emit=context.emit,
        default_factor_kind="lu",
        default_pattern_color_batch=int(color_batch),
    )
    a_csr_full = factor_bundle.operator.matrix
    ilu = factor_bundle.factor
    metadata = getattr(operator_bundle, "metadata", None)
    storage_kind = str(getattr(metadata, "storage_kind", "unknown"))
    reason = str(getattr(metadata, "reason", "unknown"))
    context.factor_cache[factor_cache_key] = (a_csr_full, ilu, storage_kind, reason)
    return a_csr_full, ilu


def _build_explicit_helper_factor(
    *,
    context: TransportSparseDirectContext,
    matvec_fn: Callable[[jnp.ndarray], jnp.ndarray],
    n: int,
    dtype: jnp.dtype,
    cache_key: tuple[object, ...],
    factor_dtype_use: np.dtype,
) -> tuple[object, object]:
    block_cols = _read_int_env("SFINCS_JAX_TRANSPORT_SPARSE_HELPER_BLOCK_COLS", default=32)
    dense_max_mb = _read_float_env("SFINCS_JAX_TRANSPORT_SPARSE_HELPER_DENSE_MAX_MB", default=128.0)
    csr_max_mb = _read_float_env("SFINCS_JAX_TRANSPORT_SPARSE_HELPER_CSR_MAX_MB", default=512.0)

    def matvec_host(x_np: np.ndarray) -> np.ndarray:
        return np.asarray(
            matvec_fn(jnp.asarray(x_np, dtype=dtype)),
            dtype=np.dtype(factor_dtype_use),
            copy=True,
        )

    def matmat_host(cols_np: np.ndarray) -> np.ndarray:
        cols = jnp.asarray(cols_np, dtype=dtype)
        out = jax.vmap(matvec_fn, in_axes=1, out_axes=1)(cols)
        return np.asarray(out, dtype=np.dtype(factor_dtype_use), copy=True)

    force_sparse = bool(jax.default_backend() != "cpu")
    factor_cache_key = (
        *context.sparse_factor_cache_key(cache_key, np.dtype(factor_dtype_use)),
        "explicit_helper",
        str(jax.default_backend()),
        int(max(1, int(block_cols))),
        float(dense_max_mb),
        float(csr_max_mb),
        int(force_sparse),
    )
    cached_factor = context.factor_cache.get(factor_cache_key)
    if cached_factor is not None:
        a_csr_full, ilu, storage_kind, reason = cached_factor
        if context.emit is not None:
            context.emit(
                1,
                "solve_v3_transport_matrix_linear_gmres: reusing explicit sparse helper "
                f"storage={storage_kind} reason={reason}",
            )
        return a_csr_full, ilu
    operator_bundle = build_operator_from_matvec(
        matvec_host,
        n=int(n),
        dtype=np.dtype(factor_dtype_use),
        backend=jax.default_backend(),
        block_cols=max(1, int(block_cols)),
        dense_max_mb=float(dense_max_mb),
        csr_max_mb=float(csr_max_mb),
        prefer_sparse_on_gpu=True,
        force_sparse=force_sparse,
        drop_tol=0.0,
        matmat=matmat_host,
        allow_operator_only=False,
    )
    if context.emit is not None:
        context.emit(
            1,
            "solve_v3_transport_matrix_linear_gmres: explicit sparse helper "
            f"storage={operator_bundle.metadata.storage_kind} "
            f"reason={operator_bundle.metadata.reason}",
        )
    factor_bundle = factorize_host_sparse_operator(
        operator_bundle,
        kind="lu",
        drop_tol=0.0,
        fill_factor=1.0,
    )
    a_csr_full = factor_bundle.operator.matrix
    ilu = factor_bundle.factor
    context.factor_cache[factor_cache_key] = (
        a_csr_full,
        ilu,
        str(operator_bundle.metadata.storage_kind),
        str(operator_bundle.metadata.reason),
    )
    return a_csr_full, ilu


def _maybe_polish_float32_factor(
    *,
    context: TransportSparseDirectContext,
    matvec_fn: Callable[[jnp.ndarray], jnp.ndarray],
    b_vec: jnp.ndarray,
    x_np: np.ndarray,
    residual_norm: float,
    ilu_for_polish: object,
    factor_dtype: np.dtype,
    tol_val: float,
    atol_val: float,
    restart_val: int,
    maxiter_val: int | None,
    precondition_side_val: str,
    true_residual_norm: Callable[[np.ndarray], float],
) -> tuple[np.ndarray, float]:
    polish_env = os.environ.get("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_POLISH", "").strip().lower()
    if polish_env in {"0", "false", "no", "off"}:
        return x_np, float(residual_norm)
    polish_restart = _read_int_env(
        "SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_POLISH_RESTART",
        default=min(int(restart_val), 40),
    )
    polish_maxiter = _read_int_env(
        "SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_POLISH_MAXITER",
        default=min(max(40, int(maxiter_val or 120)), 120),
    )
    polish_restart = max(5, int(polish_restart))
    polish_maxiter = max(5, int(polish_maxiter))
    x_polish, residual_norm_polish = context.host_sparse_direct_polish(
        matvec_fn=matvec_fn,
        rhs_vec=b_vec,
        x0_np=x_np,
        ilu=ilu_for_polish,
        factor_dtype=np.dtype(factor_dtype),
        tol=float(tol_val),
        atol=float(atol_val),
        restart=polish_restart,
        maxiter=polish_maxiter,
        precondition_side=precondition_side_val,
    )
    if np.isfinite(residual_norm_polish) and residual_norm_polish < residual_norm:
        x_np = x_polish
        residual_norm = residual_norm_polish
        true_residual = true_residual_norm(x_np)
        if np.isfinite(true_residual):
            residual_norm = float(true_residual)
    return x_np, float(residual_norm)


def _read_int_env(name: str, *, default: int) -> int:
    env = os.environ.get(name, "").strip()
    try:
        return int(env) if env else int(default)
    except ValueError:
        return int(default)


def _read_float_env(name: str, *, default: float) -> float:
    env = os.environ.get(name, "").strip()
    try:
        return float(env) if env else float(default)
    except ValueError:
        return float(default)

_transport_sparse_direct_context_from_env = transport_sparse_direct_context_from_env
