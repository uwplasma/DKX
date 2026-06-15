"""Post-solve diagnostics for RHSMode=2/3 transport solves."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any
import os

from jax import tree_util as jtu
import jax.numpy as jnp
import numpy as np

from sfincs_jax.transport_matrix import (
    v3_transport_diagnostics_vm_only_batch_jit,
    v3_transport_diagnostics_vm_only_batch_op0_jit,
    v3_transport_diagnostics_vm_only_batch_op0_precomputed_jit,
    v3_transport_diagnostics_vm_only_batch_op0_precomputed_remat_jit,
    v3_transport_diagnostics_vm_only_batch_op0_remat_jit,
    v3_transport_diagnostics_vm_only_batch_remat_jit,
    v3_transport_diagnostics_vm_only_precompute,
    v3_transport_matrix_from_flux_arrays,
)
from sfincs_jax.problems.transport_matrix.streaming_outputs import TransportStreamingOutputAccumulator
from sfincs_jax.v3_system import V3FullSystemOperator


@dataclass(frozen=True)
class TransportPostsolveDiagnostics:
    """Flux arrays, optional output fields, and matrix assembled after transport solves."""

    transport_matrix: jnp.ndarray
    particle_flux_vm_psi_hat: jnp.ndarray
    heat_flux_vm_psi_hat: jnp.ndarray
    fsab_flow: jnp.ndarray
    transport_output_fields: dict[str, np.ndarray] | None


def compute_transport_postsolve_diagnostics(
    *,
    op0: V3FullSystemOperator,
    geom: Any,
    state_vectors: Mapping[int, jnp.ndarray],
    which_rhs_values: Sequence[int],
    stream_diagnostics: bool,
    streaming_outputs: TransportStreamingOutputAccumulator | None,
    use_diag_op0: bool,
    diag_op_by_index: Sequence[V3FullSystemOperator] | None,
    emit: Callable[[int, str], None] | None = None,
) -> TransportPostsolveDiagnostics:
    """Compute batched or streamed transport diagnostics after all whichRHS solves.

    The heavy solve loop owns state-vector generation. This helper owns only the
    memory policy for diagnostics: streamed buffers when available, otherwise
    batched or chunked JAX diagnostics with optional rematerialization and
    precomputed fixed-operator geometry factors.
    """

    if emit is not None:
        emit(0, "solve_v3_transport_matrix_linear_gmres: computing whichRHS diagnostics (batched)")

    n_rhs = int(len(which_rhs_values))
    if stream_diagnostics:
        if streaming_outputs is None:
            raise RuntimeError("streaming transport diagnostics requested without an accumulator")
        diag_pf_jnp, diag_hf_jnp, diag_flow_jnp = streaming_outputs.diagnostic_flux_arrays()
        transport_output_fields = streaming_outputs.output_fields()
    else:
        remat_env = os.environ.get("SFINCS_JAX_REMAT_TRANSPORT_DIAGNOSTICS", "").strip().lower()
        if remat_env in {"1", "true", "yes", "on"}:
            use_remat_diag = True
        elif remat_env in {"0", "false", "no", "off"}:
            use_remat_diag = False
        else:
            remat_min_env = os.environ.get("SFINCS_JAX_REMAT_TRANSPORT_DIAGNOSTICS_MIN", "").strip()
            try:
                remat_min = int(remat_min_env) if remat_min_env else 20000
            except ValueError:
                remat_min = 20000
            use_remat_diag = int(op0.total_size) * int(n_rhs) >= remat_min

        diag_chunk_env = os.environ.get("SFINCS_JAX_TRANSPORT_DIAG_CHUNK", "").strip()
        try:
            diag_chunk = int(diag_chunk_env) if diag_chunk_env else None
        except ValueError:
            diag_chunk = None
        if diag_chunk is None or int(diag_chunk) <= 0:
            diag_chunk = 0
        if diag_chunk == 0 and int(op0.total_size) * int(n_rhs) >= 200_000:
            diag_chunk = 4

        if use_diag_op0:
            precompute_env = os.environ.get("SFINCS_JAX_TRANSPORT_DIAG_PRECOMPUTE", "").strip().lower()
            use_precompute = precompute_env not in {"0", "false", "no", "off"}
            if use_precompute:
                precomputed = v3_transport_diagnostics_vm_only_precompute(op0)
                diag_fn = (
                    v3_transport_diagnostics_vm_only_batch_op0_precomputed_remat_jit
                    if use_remat_diag
                    else v3_transport_diagnostics_vm_only_batch_op0_precomputed_jit
                )
            else:
                precomputed = None
                diag_fn = (
                    v3_transport_diagnostics_vm_only_batch_op0_remat_jit
                    if use_remat_diag
                    else v3_transport_diagnostics_vm_only_batch_op0_jit
                )
        else:
            if diag_op_by_index is None:
                raise RuntimeError("transport diagnostics with RHS operators require diag_op_by_index")
            diag_op_stack = jtu.tree_map(lambda *xs: jnp.stack(xs, axis=0), *diag_op_by_index)
            diag_fn = (
                v3_transport_diagnostics_vm_only_batch_remat_jit
                if use_remat_diag
                else v3_transport_diagnostics_vm_only_batch_jit
            )

        if diag_chunk <= 0 or int(diag_chunk) >= int(n_rhs):
            x_stack = jnp.stack([state_vectors[int(which_rhs)] for which_rhs in which_rhs_values], axis=0)
            if use_diag_op0:
                if use_precompute:
                    diag_stack = diag_fn(op0=op0, precomputed=precomputed, x_full_stack=x_stack)
                else:
                    diag_stack = diag_fn(op0=op0, x_full_stack=x_stack)
            else:
                diag_stack = diag_fn(op_stack=diag_op_stack, x_full_stack=x_stack)
            diag_pf_jnp = jnp.transpose(diag_stack.particle_flux_vm_psi_hat, (1, 0))
            diag_hf_jnp = jnp.transpose(diag_stack.heat_flux_vm_psi_hat, (1, 0))
            diag_flow_jnp = jnp.transpose(diag_stack.fsab_flow, (1, 0))
        else:
            n_species = int(op0.n_species)
            diag_pf_arr = np.zeros((n_species, n_rhs), dtype=np.float64)
            diag_hf_arr = np.zeros((n_species, n_rhs), dtype=np.float64)
            diag_flow_arr = np.zeros((n_species, n_rhs), dtype=np.float64)
            for start in range(0, n_rhs, int(diag_chunk)):
                end = min(n_rhs, start + int(diag_chunk))
                rhs_chunk = which_rhs_values[start:end]
                x_stack_chunk = jnp.stack([state_vectors[int(which_rhs)] for which_rhs in rhs_chunk], axis=0)
                if use_diag_op0:
                    if use_precompute:
                        diag_stack = diag_fn(op0=op0, precomputed=precomputed, x_full_stack=x_stack_chunk)
                    else:
                        diag_stack = diag_fn(op0=op0, x_full_stack=x_stack_chunk)
                else:
                    op_chunk = jtu.tree_map(lambda arr: arr[start:end], diag_op_stack)
                    diag_stack = diag_fn(op_stack=op_chunk, x_full_stack=x_stack_chunk)
                diag_pf_arr[:, start:end] = np.asarray(
                    jnp.transpose(diag_stack.particle_flux_vm_psi_hat, (1, 0))
                )
                diag_hf_arr[:, start:end] = np.asarray(jnp.transpose(diag_stack.heat_flux_vm_psi_hat, (1, 0)))
                diag_flow_arr[:, start:end] = np.asarray(jnp.transpose(diag_stack.fsab_flow, (1, 0)))
            diag_pf_jnp = jnp.asarray(diag_pf_arr, dtype=jnp.float64)
            diag_hf_jnp = jnp.asarray(diag_hf_arr, dtype=jnp.float64)
            diag_flow_jnp = jnp.asarray(diag_flow_arr, dtype=jnp.float64)
        transport_output_fields = None

    tm = v3_transport_matrix_from_flux_arrays(
        op=op0,
        geom=geom,
        particle_flux_vm_psi_hat=diag_pf_jnp,
        heat_flux_vm_psi_hat=diag_hf_jnp,
        fsab_flow=diag_flow_jnp,
    )
    return TransportPostsolveDiagnostics(
        transport_matrix=tm,
        particle_flux_vm_psi_hat=diag_pf_jnp,
        heat_flux_vm_psi_hat=diag_hf_jnp,
        fsab_flow=diag_flow_jnp,
        transport_output_fields=transport_output_fields,
    )


__all__ = ["TransportPostsolveDiagnostics", "compute_transport_postsolve_diagnostics"]
