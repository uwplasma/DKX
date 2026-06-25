from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from sfincs_jax.problems.transport_matrix import postsolve_diagnostics as postsolve


@dataclass(frozen=True)
class _FakeDiagnostics:
    particle_flux_vm_psi_hat: jnp.ndarray
    heat_flux_vm_psi_hat: jnp.ndarray
    fsab_flow: jnp.ndarray


class _FakeStreamingOutputs:
    def diagnostic_flux_arrays(self):
        return (
            jnp.asarray([[1.0, 2.0]], dtype=jnp.float64),
            jnp.asarray([[3.0, 4.0]], dtype=jnp.float64),
            jnp.asarray([[5.0, 6.0]], dtype=jnp.float64),
        )

    def output_fields(self):
        return {"streamed": np.asarray([1.0])}


def test_postsolve_diagnostics_uses_streaming_accumulator(monkeypatch) -> None:
    """Streaming mode should reuse already-collected flux arrays and output fields."""

    calls: list[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]] = []

    def fake_transport_matrix_from_flux_arrays(*, op, geom, particle_flux_vm_psi_hat, heat_flux_vm_psi_hat, fsab_flow):
        del op, geom
        calls.append((particle_flux_vm_psi_hat, heat_flux_vm_psi_hat, fsab_flow))
        return jnp.asarray([[42.0]], dtype=jnp.float64)

    monkeypatch.setattr(postsolve, "v3_transport_matrix_from_flux_arrays", fake_transport_matrix_from_flux_arrays)

    result = postsolve.compute_transport_postsolve_diagnostics(
        op0=SimpleNamespace(total_size=2, n_species=1),
        geom=object(),
        state_vectors={},
        which_rhs_values=(1, 2),
        stream_diagnostics=True,
        streaming_outputs=_FakeStreamingOutputs(),
        use_diag_op0=True,
        diag_op_by_index=None,
    )

    assert np.asarray(result.transport_matrix).tolist() == [[42.0]]
    assert result.transport_output_fields is not None
    assert result.transport_output_fields["streamed"].tolist() == [1.0]
    assert np.asarray(calls[0][0]).tolist() == [[1.0, 2.0]]
    assert np.asarray(result.heat_flux_vm_psi_hat).tolist() == [[3.0, 4.0]]
    assert np.asarray(result.fsab_flow).tolist() == [[5.0, 6.0]]


def test_postsolve_diagnostics_chunks_fixed_operator_path(monkeypatch) -> None:
    """Chunked fixed-operator diagnostics should assemble species-by-RHS arrays."""

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_DIAG_CHUNK", "1")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_DIAG_PRECOMPUTE", "0")
    monkeypatch.setenv("SFINCS_JAX_REMAT_TRANSPORT_DIAGNOSTICS", "0")
    seen_chunks: list[tuple[int, int]] = []

    def fake_diag_fn(*, op0, x_full_stack):
        del op0
        n_chunk = int(x_full_stack.shape[0])
        seen_chunks.append((n_chunk, int(x_full_stack.shape[1])))
        base = jnp.sum(x_full_stack, axis=1)
        data = jnp.stack([base, base + 10.0], axis=1)
        return _FakeDiagnostics(
            particle_flux_vm_psi_hat=data,
            heat_flux_vm_psi_hat=data + 100.0,
            fsab_flow=data + 200.0,
        )

    def fake_transport_matrix_from_flux_arrays(*, op, geom, particle_flux_vm_psi_hat, heat_flux_vm_psi_hat, fsab_flow):
        del op, geom, heat_flux_vm_psi_hat, fsab_flow
        return particle_flux_vm_psi_hat + 1.0

    monkeypatch.setattr(postsolve, "v3_transport_diagnostics_vm_only_batch_op0_jit", fake_diag_fn)
    monkeypatch.setattr(postsolve, "v3_transport_matrix_from_flux_arrays", fake_transport_matrix_from_flux_arrays)

    result = postsolve.compute_transport_postsolve_diagnostics(
        op0=SimpleNamespace(total_size=3, n_species=2),
        geom=object(),
        state_vectors={
            1: jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float64),
            2: jnp.asarray([4.0, 5.0, 6.0], dtype=jnp.float64),
        },
        which_rhs_values=(1, 2),
        stream_diagnostics=False,
        streaming_outputs=None,
        use_diag_op0=True,
        diag_op_by_index=None,
    )

    assert seen_chunks == [(1, 3), (1, 3)]
    assert np.asarray(result.particle_flux_vm_psi_hat).tolist() == [[6.0, 15.0], [16.0, 25.0]]
    assert np.asarray(result.heat_flux_vm_psi_hat).tolist() == [[106.0, 115.0], [116.0, 125.0]]
    assert np.asarray(result.fsab_flow).tolist() == [[206.0, 215.0], [216.0, 225.0]]
    assert np.asarray(result.transport_matrix).tolist() == [[7.0, 16.0], [17.0, 26.0]]
