from __future__ import annotations

from pathlib import Path

from sfincs_jax.namelist import read_sfincs_input


def test_parse_input_namelist_quick_example() -> None:
    input_path = Path(__file__).parent / "ref" / "quick_2species_FPCollisions_noEr.input.namelist"
    nml = read_sfincs_input(input_path)

    geom = nml.group("geometryParameters")
    assert geom["GEOMETRYSCHEME"] == 4

    species = nml.group("speciesParameters")
    assert species["ZS"] == [1, 6]
    assert species["MHATS"] == [1, 6]

    physics = nml.group("physicsParameters")
    assert abs(float(physics["DELTA"]) - 4.5694e-3) < 1e-12
    assert physics["INCLUDEXDOTTERM"] is True
    assert physics["INCLUDEPHI1"] is False


def test_parse_double_quoted_string_and_comment_marker(tmp_path: Path) -> None:
    input_path = tmp_path / "input.namelist"
    input_path.write_text(
        "&geometryParameters\n"
        '  equilibriumFile = "archive/path!with_marker/wout.nc" ! external VMEC file\n'
        "/\n",
        encoding="utf-8",
    )

    nml = read_sfincs_input(input_path)

    assert nml.group("geometryParameters")["EQUILIBRIUMFILE"] == "archive/path!with_marker/wout.nc"
