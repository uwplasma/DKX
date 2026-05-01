from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from sfincs_jax.io import write_sfincs_output_file
from sfincs_jax.solver_trace import (
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
        candidate_decisions=(
            SolverTraceCandidate(name="dense", accepted=True, residual_ratio=4.0e-3),
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
