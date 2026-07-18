from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dkx import io


def test_output_file_format_suffixes_and_invalid_suffix() -> None:
    assert io.output_file_format(Path("sfincsOutput.h5")) == "h5"
    assert io.output_file_format(Path("sfincsOutput.hdf5")) == "h5"
    assert io.output_file_format(Path("sfincsOutput")) == "h5"
    assert io.output_file_format(Path("sfincsOutput.nc")) == "netcdf"
    assert io.output_file_format(Path("sfincsOutput.netcdf")) == "netcdf"
    assert io.output_file_format(Path("sfincsOutput.npz")) == "npz"
    with pytest.raises(ValueError, match="Unsupported dkx output suffix"):
        io.output_file_format(Path("sfincsOutput.txt"))


def test_netcdf_safe_name_preserves_uniqueness() -> None:
    used: set[str] = set()

    assert io.netcdf_safe_name("matrix with spaces", used) == "matrix_with_spaces"
    assert io.netcdf_safe_name("matrix-with-spaces", used) == "matrix_with_spaces_2"
    assert io.netcdf_safe_name("123", used) == "v_123"
    assert io.netcdf_safe_name("!!!", used) == "dataset"


def test_h5_roundtrip_decodes_nested_bytes_and_fortran_layout(tmp_path: Path) -> None:
    out = tmp_path / "mini.h5"
    arr = np.arange(24.0).reshape(2, 3, 4)

    io.write_sfincs_h5(
        path=out,
        data={"cube": arr, "label": np.asarray(b"abc")},
        fortran_layout=True,
    )
    loaded = io.read_sfincs_h5(out)

    np.testing.assert_allclose(loaded["cube"], np.transpose(arr, axes=(2, 1, 0)))
    assert loaded["label"] == "abc"
    with pytest.raises(FileExistsError):
        io.write_sfincs_h5(path=out, data={"x": np.asarray(1.0)}, overwrite=False)


@pytest.mark.parametrize("suffix", [".npz", ".nc"])
def test_output_file_roundtrip_preserves_names_strings_and_bools(tmp_path: Path, suffix: str) -> None:
    if suffix == ".nc":
        pytest.importorskip("netCDF4")
    out = tmp_path / f"sfincsOutput{suffix}"
    data = {
        "scalar": np.asarray(3.0),
        "matrix with spaces": np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        "logical flag": np.asarray(True),
        "input.namelist": "example = true",
    }

    io.write_sfincs_output_file(path=out, data=data, fortran_layout=False)
    loaded = io.read_sfincs_output_file(out)

    np.testing.assert_allclose(loaded["scalar"], data["scalar"])
    np.testing.assert_allclose(loaded["matrix with spaces"], data["matrix with spaces"])
    assert bool(np.asarray(loaded["logical flag"])) is True
    assert str(loaded["input.namelist"]) == "example = true"


def _trace():
    from dkx.solver_trace import SolverTrace, SolverTraceCandidate

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
    import h5py

    from dkx.solver_trace import SolverTrace, read_solver_trace_h5

    out = tmp_path / f"sfincsOutput{suffix}"
    trace = _trace()

    io.write_sfincs_output_file(
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
            loaded = SolverTrace.from_json(ds.getncattr("dkx_solver_trace_json"))
    else:
        with np.load(out, allow_pickle=False) as npz:
            loaded = SolverTrace.from_json(str(npz["dkx_solver_trace_json"].reshape(()).item()))

    assert loaded == trace
