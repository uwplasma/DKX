from pathlib import Path

import numpy as np
import pytest

from sfincs_jax.io import read_sfincs_h5, write_sfincs_jax_output_h5


def test_write_output_return_results(tmp_path: Path) -> None:
    input_path = Path(__file__).parent / "ref" / "output_scheme4_1species_tiny.input.namelist"
    out_path = tmp_path / "sfincsOutput.h5"

    resolved, results = write_sfincs_jax_output_h5(
        input_namelist=input_path,
        output_path=out_path,
        return_results=True,
    )

    assert resolved == out_path.resolve()
    assert resolved.exists()
    assert isinstance(results, dict)
    assert "Ntheta" in results
    assert int(np.asarray(results["Ntheta"]).reshape(())) > 0


def test_write_output_wout_path_override_resolves_missing_scheme5_input(tmp_path: Path) -> None:
    source_input = Path(__file__).parent / "ref" / "output_scheme5_1species_tiny.input.namelist"
    patched_input = tmp_path / "input.namelist"
    patched_input.write_text(
        source_input.read_text().replace("wout_w7x_standardConfig.nc", "missing_wout.nc"),
        encoding="utf-8",
    )
    out_path = tmp_path / "sfincsOutput.h5"
    actual_wout = Path(__file__).parent / "ref" / "wout_w7x_standardConfig.nc"

    with pytest.raises(FileNotFoundError):
        write_sfincs_jax_output_h5(
            input_namelist=patched_input,
            output_path=out_path,
        )

    write_sfincs_jax_output_h5(
        input_namelist=patched_input,
        output_path=out_path,
        wout_path=actual_wout,
    )

    data = read_sfincs_h5(out_path)
    input_text = str(data["input.namelist"])
    assert "missing_wout.nc" not in input_text
    assert str(actual_wout) in input_text
