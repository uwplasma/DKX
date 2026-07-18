from __future__ import annotations

import h5py
import pytest

from dkx.solver_trace import (
    SolverTrace,
    SolverTraceCandidate,
    read_solver_trace_h5,
    read_solver_trace_json,
    write_solver_trace_h5,
    write_solver_trace_json,
)


def _example_trace() -> SolverTrace:
    return SolverTrace(
        backend="gpu",
        rhs_mode=1,
        selected_path="dense",
        solve_method="direct",
        preconditioner="none",
        geometry_scheme=5,
        collision_operator="full-fp",
        total_size=2048,
        active_size=1536,
        device_count=1,
        cold_jit=False,
        residual_norm=2.0e-12,
        residual_target=1.0e-9,
        elapsed_s=2.5,
        setup_s=0.4,
        solve_s=2.1,
        peak_rss_mb=950.0,
        active_rss_mb=320.0,
        device_peak_mb=410.0,
        compiled_temp_mb=96.0,
        estimated_dense_nbytes=2048 * 2048 * 8,
        estimated_csr_nbytes=2_400_000,
        estimated_gmres_basis_nbytes=2048 * 36 * 8,
        matvec_count=42,
        candidate_decisions=(
            SolverTraceCandidate(
                name="dense",
                accepted=True,
                residual_ratio=2.0e-3,
                runtime_ratio=0.2,
                memory_ratio=0.5,
                memory_metric="active_rss_mb",
                active_rss_mb=320.0,
                device_peak_mb=410.0,
                compiled_temp_mb=96.0,
                candidate_setup_s=0.4,
                candidate_solve_s=2.1,
            ),
            SolverTraceCandidate(
                name="pas_lite",
                accepted=False,
                reasons=("residual_not_clean", "runtime_regression"),
                residual_ratio=1.0e6,
                runtime_ratio=10.0,
                memory_ratio=1.4,
                memory_metric="active_rss_mb",
                active_rss_mb=512.0,
                candidate_setup_s=0.8,
                candidate_solve_s=10.0,
            ),
        ),
        metadata={"case": "finite_beta_profile_current"},
    )


def test_solver_trace_json_roundtrip_preserves_decisions(tmp_path) -> None:
    trace = _example_trace()
    path = tmp_path / "solver_trace.json"

    write_solver_trace_json(path, trace)
    loaded = read_solver_trace_json(path)

    assert loaded == trace
    assert loaded.candidate_decisions[1].reasons == ("residual_not_clean", "runtime_regression")
    assert loaded.active_rss_mb == 320.0
    assert loaded.device_peak_mb == 410.0
    assert loaded.estimated_csr_nbytes == 2_400_000
    assert loaded.candidate_decisions[0].memory_metric == "active_rss_mb"
    assert loaded.candidate_decisions[0].candidate_solve_s == 2.1
    assert loaded.metadata["case"] == "finite_beta_profile_current"


def test_solver_trace_hdf5_roundtrip_preserves_schema(tmp_path) -> None:
    trace = _example_trace()
    path = tmp_path / "output.h5"

    with h5py.File(path, "w") as h5:
        write_solver_trace_h5(h5, trace)

    with h5py.File(path, "r") as h5:
        loaded = read_solver_trace_h5(h5)
        assert h5["solver_trace"].attrs["schema_version"] == 1

    assert loaded == trace


def test_solver_trace_rejects_unknown_schema_version() -> None:
    trace = _example_trace().to_dict()
    trace["schema_version"] = 999

    with pytest.raises(ValueError, match="Unsupported solver trace schema_version=999"):
        SolverTrace.from_dict(trace)


def test_solver_trace_minimal_payload_roundtrip() -> None:
    trace = SolverTrace(backend="cpu", rhs_mode=3, selected_path="transport_dense")

    loaded = SolverTrace.from_json(trace.to_json())

    assert loaded.backend == "cpu"
    assert loaded.rhs_mode == 3
    assert loaded.selected_path == "transport_dense"
    assert loaded.candidate_decisions == ()
