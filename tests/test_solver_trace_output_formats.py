from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from sfincs_jax.io import write_sfincs_output_file
from sfincs_jax.outputs.rhsmode1 import (
    _profile_memory_summary,
    _solver_trace_memory_estimate,
)
from sfincs_jax.solvers.trace import (
    SolverTrace,
    SolverTraceCandidate,
    read_solver_trace_h5,
)


def _trace() -> SolverTrace:
    return SolverTrace(
        backend="gpu",
        rhs_mode=1,
        selected_path="dense",
        solve_method="direct",
        preconditioner="none",
        geometry_scheme=5,
        collision_operator="full-fp",
        total_size=4096,
        active_size=3072,
        device_count=1,
        cold_jit=False,
        residual_norm=4.0e-12,
        residual_target=1.0e-9,
        elapsed_s=3.4,
        setup_s=0.9,
        solve_s=2.5,
        peak_rss_mb=875.0,
        active_rss_mb=275.0,
        device_peak_mb=350.0,
        compiled_temp_mb=80.0,
        estimated_dense_nbytes=4096 * 4096 * 8,
        estimated_csr_nbytes=4_000_000,
        estimated_gmres_basis_nbytes=4096 * 24 * 8,
        matvec_count=24,
        candidate_decisions=(
            SolverTraceCandidate(
                name="dense",
                accepted=True,
                residual_ratio=4.0e-3,
                memory_metric="device_peak_mb",
                device_peak_mb=350.0,
                candidate_setup_s=0.9,
                candidate_solve_s=2.5,
            ),
        ),
        metadata={"case": "solver-trace-output-format"},
    )


@pytest.mark.parametrize("suffix", [".h5", ".nc", ".npz"])
def test_write_sfincs_output_file_can_attach_solver_trace(tmp_path: Path, suffix: str) -> None:
    out = tmp_path / f"sfincsOutput{suffix}"
    trace = _trace()

    write_sfincs_output_file(
        path=out,
        data={"scalar": np.asarray(1.0), "vector": np.asarray([1.0, 2.0])},
        fortran_layout=False,
        solver_trace=trace,
    )

    if suffix == ".h5":
        with h5py.File(out, "r") as h5:
            loaded = read_solver_trace_h5(h5)
    elif suffix == ".nc":
        netcdf4 = pytest.importorskip("netCDF4")
        with netcdf4.Dataset(out, "r") as ds:
            loaded = SolverTrace.from_json(ds.getncattr("sfincs_jax_solver_trace_json"))
    else:
        with np.load(out, allow_pickle=False) as npz:
            loaded = SolverTrace.from_json(str(npz["sfincs_jax_solver_trace_json"].reshape(()).item()))

    assert loaded == trace


def test_solver_trace_memory_estimate_uses_sparse_metadata() -> None:
    estimate = _solver_trace_memory_estimate(
        total_size=100,
        active_size=80,
        solver_metadata={
            "sparse_pattern_nnz": 1_200,
            "gmres_restart": 30,
            "sparse_pc_factor_nbytes_estimate": 50_000,
        },
        device_count=2,
    )

    assert estimate is not None
    assert estimate["dense_operator_nbytes"] == 100 * 100 * 8
    assert estimate["csr_operator_nbytes"] == 1_200 * (8 + 4) + 101 * 4
    assert estimate["gmres_basis_nbytes"] == 100 * (30 + 1 + 4) * 8
    assert estimate["preconditioner_nbytes"] == 50_000
    assert estimate["csr_per_device_nbytes"] > 0


def test_profile_memory_summary_prefers_active_and_device_peaks() -> None:
    profiler = SimpleNamespace(
        entries=[
            {"rss_mb": 100.0, "peak_rss_mb": 120.0, "dpeak_rss_mb": 10.0, "device_mb": 30.0},
            {"rss_mb": 110.0, "peak_rss_mb": 150.0, "dpeak_rss_mb": 42.0, "device_mb": 55.0},
        ]
    )

    active_rss_mb, device_peak_mb, peak_rss_mb = _profile_memory_summary(profiler)

    assert active_rss_mb == 42.0
    assert device_peak_mb == 55.0
    assert peak_rss_mb == 150.0
