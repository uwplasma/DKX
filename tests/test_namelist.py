from __future__ import annotations

from pathlib import Path

import pytest

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


def test_parse_fortran_scalars_vectors_and_indexed_assignments(tmp_path: Path) -> None:
    input_path = tmp_path / "input.namelist"
    input_path.write_text(
        "&speciesParameters\n"
        "  Zs = 1, +6\n"
        "  mHats(1) = 1d0\n"
        "  mHats(2) = 6.0D+0\n"
        "/\n"
        "&physicsParameters\n"
        "  includeXDotTerm = T\n"
        "  includeElectricFieldTermInXiDot = .FALSE.\n"
        "  solverTolerance = 1d-12\n"
        "  quoted = 'value with ! marker'\n"
        "/\n",
        encoding="utf-8",
    )

    nml = read_sfincs_input(input_path)

    species = nml.group("speciesParameters")
    physics = nml.group("physicsParameters")
    assert species["ZS"] == [1, 6]
    assert nml.indexed["speciesparameters"]["MHATS"][(1,)] == 1.0
    assert nml.indexed["speciesparameters"]["MHATS"][(2,)] == 6.0
    assert physics["INCLUDEXDOTTERM"] is True
    assert physics["INCLUDEELECTRICFIELDTERMINXIDOT"] is False
    assert physics["SOLVERTOLERANCE"] == 1.0e-12
    assert physics["QUOTED"] == "value with ! marker"
    assert nml.group("missing") == {}


def test_parse_multidimensional_indexed_assignment(tmp_path: Path) -> None:
    input_path = tmp_path / "input.namelist"
    input_path.write_text(
        "&geometryParameters\n"
        "  table(2, 3) = -4\n"
        "/\n",
        encoding="utf-8",
    )

    nml = read_sfincs_input(input_path)

    assert nml.indexed["geometryparameters"]["TABLE"][(2, 3)] == -4


def test_namelist_parser_rejects_nested_or_unterminated_groups(tmp_path: Path) -> None:
    nested = tmp_path / "nested.namelist"
    nested.write_text(
        "&geometryParameters\n"
        "&physicsParameters\n"
        "/\n",
        encoding="utf-8",
    )
    unterminated = tmp_path / "unterminated.namelist"
    unterminated.write_text("&geometryParameters\n  geometryScheme = 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Nested namelist group"):
        read_sfincs_input(nested)
    with pytest.raises(ValueError, match="not terminated"):
        read_sfincs_input(unterminated)


def test_namelist_parser_rejects_multi_value_indexed_assignment(tmp_path: Path) -> None:
    input_path = tmp_path / "input.namelist"
    input_path.write_text(
        "&speciesParameters\n"
        "  Zs(1) = 1, 2\n"
        "/\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Indexed assignment Zs\\(1\\) has multiple values"):
        read_sfincs_input(input_path)
