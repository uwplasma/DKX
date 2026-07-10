from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

import sfincs_jax.problems.transport_parallel_runtime as parallel_solve


def _runtime(**overrides):
    values = {
        "run_gpu_subprocesses": lambda **_kwargs: [],
        "persistent_pool_enabled": False,
        "get_pool": lambda **_kwargs: object(),
        "shutdown_pool": lambda: None,
        "worker": lambda _payload: {},
        "worker_env": lambda _workers: nullcontext(),
        "executor_class": object,
        "executor_kwargs": lambda **_kwargs: {},
        "elapsed_s": lambda: 12.5,
    }
    values.update(overrides)
    return parallel_solve.TransportParallelSolveRuntime(**values)


def test_parallel_parent_solve_inactive_without_workers() -> None:
    """The extracted parent solve should no-op when parallel policy is inactive."""

    result = parallel_solve.maybe_run_transport_parallel_solve(
        nml=SimpleNamespace(),
        op0=SimpleNamespace(),
        rhs_mode=2,
        n_rhs=2,
        which_rhs_values=(1, 2),
        parallel_child=False,
        parallel_workers=1,
        parallel_backend="cpu",
        input_namelist=Path("input.namelist"),
        tol=1e-10,
        atol=0.0,
        restart=40,
        maxiter=100,
        solve_method="auto",
        identity_shift=0.0,
        collect_transport_output_fields=False,
        phi1_hat_base=None,
        differentiable=None,
        runtime=_runtime(),
    )

    assert result is None


def test_parallel_parent_solve_merges_workers_and_builds_result(monkeypatch) -> None:
    """Parallel parent orchestration should merge worker payloads and assemble diagnostics."""

    calls: dict[str, object] = {}

    def fake_run_transport_parallel_payloads(**kwargs):
        calls["payloads"] = kwargs["payloads"]
        calls["parallel_backend"] = kwargs["parallel_backend"]
        return [
            {
                "which_rhs_values": [1],
                "state_vectors_by_rhs": {1: np.asarray([1.0, 2.0])},
                "residual_norms_by_rhs": {1: 1.0e-11},
                "rhs_norms_by_rhs": {1: 10.0},
                "elapsed_time_s": np.asarray([0.25]),
            },
            {
                "which_rhs_values": [2],
                "state_vectors_by_rhs": {2: np.asarray([3.0, 4.0])},
                "residual_norms_by_rhs": {2: 2.0e-11},
                "rhs_norms_by_rhs": {2: 20.0},
                "elapsed_time_s": np.asarray([0.5]),
            },
        ]

    def fake_output_fields(*, op0, state_vectors_by_rhs):
        del op0
        assert sorted(state_vectors_by_rhs) == [1, 2]
        return {
            "particleFlux_vm_psiHat": np.asarray([[1.0, 2.0]]),
            "heatFlux_vm_psiHat": np.asarray([[3.0, 4.0]]),
            "FSABFlow": np.asarray([[5.0, 6.0]]),
            "diagnostic": np.asarray([7.0]),
        }

    def fake_transport_matrix_from_flux_arrays(*, op, geom, particle_flux_vm_psi_hat, heat_flux_vm_psi_hat, fsab_flow):
        del op, geom, heat_flux_vm_psi_hat, fsab_flow
        return particle_flux_vm_psi_hat + 10.0

    monkeypatch.setattr(parallel_solve, "run_transport_parallel_payloads", fake_run_transport_parallel_payloads)
    monkeypatch.setattr(parallel_solve, "v3_transport_output_fields_vm_only", fake_output_fields)
    monkeypatch.setattr(parallel_solve, "grids_from_namelist", lambda _nml: object())
    monkeypatch.setattr(parallel_solve, "geometry_from_namelist", lambda **_kwargs: object())
    monkeypatch.setattr(parallel_solve, "v3_transport_matrix_from_flux_arrays", fake_transport_matrix_from_flux_arrays)

    messages: list[str] = []
    result = parallel_solve.maybe_run_transport_parallel_solve(
        nml=SimpleNamespace(),
        op0=SimpleNamespace(),
        rhs_mode=2,
        n_rhs=2,
        which_rhs_values=(1, 2),
        parallel_child=False,
        parallel_workers=2,
        parallel_backend="gpu",
        input_namelist=Path("input.namelist"),
        tol=1e-10,
        atol=0.0,
        restart=40,
        maxiter=100,
        solve_method="auto",
        identity_shift=0.0,
        collect_transport_output_fields=True,
        phi1_hat_base=jnp.asarray([0.0]),
        differentiable=False,
        runtime=_runtime(),
        emit=lambda _level, msg: messages.append(msg),
    )

    assert result is not None
    assert calls["parallel_backend"] == "gpu"
    assert [payload["which_rhs_values"] for payload in calls["payloads"]] == [[1], [2]]
    assert np.asarray(result.transport_matrix).tolist() == [[11.0, 12.0]]
    assert np.asarray(result.elapsed_time_s).tolist() == [0.25, 0.5]
    assert np.asarray(result.rhs_norms_by_rhs[2]).item() == 20.0
    assert result.transport_output_fields is not None
    assert result.transport_output_fields["diagnostic"].tolist() == [7.0]
    assert any("parallel whichRHS" in msg for msg in messages)


def test_parallel_parent_solve_rejects_missing_worker_state_vectors(monkeypatch) -> None:
    """Parallel orchestration should fail before diagnostics if a worker drops an RHS."""

    def fake_run_transport_parallel_payloads(**_kwargs):
        return [
            {
                "which_rhs_values": [1],
                "state_vectors_by_rhs": {1: np.asarray([1.0, 2.0])},
                "residual_norms_by_rhs": {1: 1.0e-11},
                "rhs_norms_by_rhs": {1: 10.0},
                "elapsed_time_s": np.asarray([0.25]),
            },
        ]

    monkeypatch.setattr(parallel_solve, "run_transport_parallel_payloads", fake_run_transport_parallel_payloads)

    try:
        parallel_solve.maybe_run_transport_parallel_solve(
            nml=SimpleNamespace(),
            op0=SimpleNamespace(),
            rhs_mode=2,
            n_rhs=2,
            which_rhs_values=(1, 2),
            parallel_child=False,
            parallel_workers=2,
            parallel_backend="cpu",
            input_namelist=Path("input.namelist"),
            tol=1e-10,
            atol=0.0,
            restart=40,
            maxiter=100,
            solve_method="auto",
            identity_shift=0.0,
            collect_transport_output_fields=False,
            phi1_hat_base=None,
            differentiable=False,
            runtime=_runtime(),
        )
    except RuntimeError as exc:
        assert "missing state vectors for whichRHS=[2]" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected missing-state-vector failure")
