from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import sfincs_jax.io as io
from sfincs_jax.outputs import formats


def test_io_legacy_output_format_aliases_point_to_new_owner() -> None:
    """Keep existing ``sfincs_jax.io`` imports stable during the I/O split."""

    assert io.read_sfincs_h5 is formats.read_sfincs_h5
    assert io.write_sfincs_h5 is formats.write_sfincs_h5
    assert io.read_sfincs_output_file is formats.read_sfincs_output_file
    assert io.write_sfincs_output_file is formats.write_sfincs_output_file
    assert io._fortran_h5_layout is formats.fortran_h5_layout
    assert io._output_file_format is formats.output_file_format


def test_output_file_format_suffixes_and_invalid_suffix() -> None:
    assert formats.output_file_format(Path("sfincsOutput.h5")) == "h5"
    assert formats.output_file_format(Path("sfincsOutput.hdf5")) == "h5"
    assert formats.output_file_format(Path("sfincsOutput")) == "h5"
    assert formats.output_file_format(Path("sfincsOutput.nc")) == "netcdf"
    assert formats.output_file_format(Path("sfincsOutput.netcdf")) == "netcdf"
    assert formats.output_file_format(Path("sfincsOutput.npz")) == "npz"
    with pytest.raises(ValueError, match="Unsupported sfincs_jax output suffix"):
        formats.output_file_format(Path("sfincsOutput.txt"))


def test_netcdf_safe_name_preserves_uniqueness() -> None:
    used: set[str] = set()

    assert formats.netcdf_safe_name("matrix with spaces", used) == "matrix_with_spaces"
    assert formats.netcdf_safe_name("matrix-with-spaces", used) == "matrix_with_spaces_2"
    assert formats.netcdf_safe_name("123", used) == "v_123"
    assert formats.netcdf_safe_name("!!!", used) == "dataset"


def test_h5_roundtrip_decodes_nested_bytes_and_fortran_layout(tmp_path: Path) -> None:
    out = tmp_path / "mini.h5"
    arr = np.arange(24.0).reshape(2, 3, 4)

    formats.write_sfincs_h5(
        path=out,
        data={"cube": arr, "label": np.asarray(b"abc")},
        fortran_layout=True,
    )
    loaded = formats.read_sfincs_h5(out)

    np.testing.assert_allclose(loaded["cube"], np.transpose(arr, axes=(2, 1, 0)))
    assert loaded["label"] == "abc"
    with pytest.raises(FileExistsError):
        formats.write_sfincs_h5(path=out, data={"x": np.asarray(1.0)}, overwrite=False)


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

    formats.write_sfincs_output_file(path=out, data=data, fortran_layout=False)
    loaded = formats.read_sfincs_output_file(out)

    np.testing.assert_allclose(loaded["scalar"], data["scalar"])
    np.testing.assert_allclose(loaded["matrix with spaces"], data["matrix with spaces"])
    assert bool(np.asarray(loaded["logical flag"])) is True
    assert str(loaded["input.namelist"]) == "example = true"
