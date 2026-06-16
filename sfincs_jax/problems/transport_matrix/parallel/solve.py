"""Parent-side orchestration for parallel RHSMode=2/3 transport solves."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from sfincs_jax.namelist import Namelist
from sfincs_jax.transport_matrix import (
    v3_transport_matrix_from_flux_arrays,
    v3_transport_output_fields_vm_only,
)
from sfincs_jax.problems.transport_matrix.parallel.execution import (
    build_transport_parallel_payloads,
    run_transport_parallel_payloads,
    should_run_transport_parallel,
)
from sfincs_jax.problems.transport_matrix.parallel.runtime import (
    merge_transport_parallel_results,
    partition_transport_rhs,
)
from sfincs_jax.v3 import geometry_from_namelist, grids_from_namelist
from sfincs_jax.v3_results import V3TransportMatrixSolveResult
from sfincs_jax.v3_system import V3FullSystemOperator


@dataclass(frozen=True)
class TransportParallelSolveRuntime:
    """Injected runtime hooks needed to launch and merge transport workers."""

    run_gpu_subprocesses: Callable[..., list[dict[str, object]]]
    persistent_pool_enabled: bool
    get_pool: Callable[..., object]
    shutdown_pool: Callable[[], None]
    worker: Callable[[dict[str, object]], dict[str, object]]
    worker_env: Callable[[int], object]
    executor_class: Any
    executor_kwargs: Callable[..., dict[str, object]]
    elapsed_s: Callable[[], float]


def maybe_run_transport_parallel_solve(
    *,
    nml: Namelist,
    op0: V3FullSystemOperator,
    rhs_mode: int,
    n_rhs: int,
    which_rhs_values: Sequence[int],
    parallel_child: bool,
    parallel_workers: int,
    parallel_backend: str,
    input_namelist: Path | None,
    tol: float,
    atol: float,
    restart: int,
    maxiter: int | None,
    solve_method: str,
    identity_shift: float,
    collect_transport_output_fields: bool,
    phi1_hat_base: jnp.ndarray | None,
    differentiable: bool | None,
    runtime: TransportParallelSolveRuntime,
    emit: Callable[[int, str], None] | None = None,
) -> V3TransportMatrixSolveResult | None:
    """Run the parent-side parallel whichRHS branch, or return ``None``.

    The worker payload format and process/GPU execution helpers live in the
    transport-parallel modules. This function owns the parent orchestration that
    was historically embedded in ``v3_driver.py``: partitioning, worker launch,
    result merge, transport diagnostic assembly, and early result construction.
    """

    if not should_run_transport_parallel(
        parallel_child=bool(parallel_child),
        parallel_workers=int(parallel_workers),
        which_rhs_values=which_rhs_values,
        input_namelist=input_namelist,
    ):
        return None

    if input_namelist is None:
        raise RuntimeError("parallel transport solve requires input_namelist")

    if emit is not None:
        emit(
            0,
            "solve_v3_transport_matrix_linear_gmres: parallel whichRHS "
            f"(backend={parallel_backend} workers={int(parallel_workers)} "
            f"rhs_count={len(which_rhs_values)}/{int(n_rhs)})",
        )

    chunks = partition_transport_rhs(list(which_rhs_values), int(parallel_workers))
    payloads = build_transport_parallel_payloads(
        chunks=chunks,
        input_namelist=input_namelist,
        tol=float(tol),
        atol=float(atol),
        restart=int(restart),
        maxiter=maxiter,
        solve_method=str(solve_method),
        identity_shift=float(identity_shift),
        collect_transport_output_fields=bool(collect_transport_output_fields),
        phi1_hat_base=phi1_hat_base,
        differentiable=differentiable,
    )

    results = run_transport_parallel_payloads(
        payloads=payloads,
        parallel_workers=int(parallel_workers),
        parallel_backend=str(parallel_backend),
        run_gpu_subprocesses=runtime.run_gpu_subprocesses,
        persistent_pool_enabled=bool(runtime.persistent_pool_enabled),
        get_pool=runtime.get_pool,
        shutdown_pool=runtime.shutdown_pool,
        worker=runtime.worker,
        worker_env=runtime.worker_env,
        executor_class=runtime.executor_class,
        executor_kwargs=runtime.executor_kwargs,
        emit=emit,
    )

    state_vectors_np, residual_norms_np, rhs_norms_np, elapsed_s = merge_transport_parallel_results(
        n_rhs=int(n_rhs),
        results=results,
    )
    state_vectors = {
        int(k): jnp.asarray(v, dtype=jnp.float64)
        for k, v in state_vectors_np.items()
    }
    residual_norms = {
        int(k): jnp.asarray(v, dtype=jnp.float64)
        for k, v in residual_norms_np.items()
    }
    rhs_norms = {
        int(k): jnp.asarray(v, dtype=jnp.float64)
        for k, v in rhs_norms_np.items()
    }

    missing_rhs = [which_rhs for which_rhs in range(1, int(n_rhs) + 1) if which_rhs not in state_vectors]
    if missing_rhs:
        raise RuntimeError(f"parallel transport solve missing state vectors for whichRHS={missing_rhs}")

    if emit is not None:
        for which_rhs in range(1, int(n_rhs) + 1):
            rn = float(np.asarray(residual_norms.get(which_rhs, np.nan), dtype=np.float64))
            rhsn = float(np.asarray(rhs_norms.get(which_rhs, np.nan), dtype=np.float64))
            rel = rn / rhsn if np.isfinite(rhsn) and rhsn > 0.0 else float("nan")
            emit(
                0,
                f"whichRHS={which_rhs}: residual_norm={rn:.6e} rhs_norm={rhsn:.6e} "
                f"relative_residual={rel:.6e} elapsed_s={float(elapsed_s[which_rhs - 1]):.3f}",
            )
        emit(0, "solve_v3_transport_matrix_linear_gmres: computing whichRHS diagnostics (batched)")

    transport_fields_full = v3_transport_output_fields_vm_only(
        op0=op0,
        state_vectors_by_rhs=state_vectors,
    )
    diag_pf = jnp.asarray(transport_fields_full["particleFlux_vm_psiHat"], dtype=jnp.float64)
    diag_hf = jnp.asarray(transport_fields_full["heatFlux_vm_psiHat"], dtype=jnp.float64)
    diag_flow = jnp.asarray(transport_fields_full["FSABFlow"], dtype=jnp.float64)
    geom = geometry_from_namelist(nml=nml, grids=grids_from_namelist(nml))
    tm = v3_transport_matrix_from_flux_arrays(
        op=op0,
        geom=geom,
        particle_flux_vm_psi_hat=diag_pf,
        heat_flux_vm_psi_hat=diag_hf,
        fsab_flow=diag_flow,
    )
    transport_output_fields = transport_fields_full if bool(collect_transport_output_fields) else None
    if emit is not None:
        emit(0, "solve_v3_transport_matrix_linear_gmres: done")
        emit(1, f"solve_v3_transport_matrix_linear_gmres: elapsed_s={runtime.elapsed_s():.3f}")
    return V3TransportMatrixSolveResult(
        op0=op0,
        transport_matrix=tm,
        state_vectors_by_rhs=state_vectors,
        residual_norms_by_rhs=residual_norms,
        fsab_flow=diag_flow,
        particle_flux_vm_psi_hat=diag_pf,
        heat_flux_vm_psi_hat=diag_hf,
        elapsed_time_s=jnp.asarray(elapsed_s, dtype=jnp.float64),
        transport_output_fields=transport_output_fields,
        rhs_norms_by_rhs=rhs_norms,
    )


__all__ = ["TransportParallelSolveRuntime", "maybe_run_transport_parallel_solve"]
