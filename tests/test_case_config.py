from __future__ import annotations

from dataclasses import FrozenInstanceError, is_dataclass
import json
from pathlib import Path

import pytest

import dkx
from dkx import cli
from dkx.config import (
    COMMENTED_TOML_EXAMPLE,
    Case,
    CaseValidationError,
    ScanConfig,
    GeometryConfig,
    SpeciesConfig,
    case_json_schema,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = REPO_ROOT / "examples" / "native" / "w7x_ambipolar_profile.toml"


def _mapping() -> dict:
    return {
        "schema": 1,
        "name": "ordered-independent",
        "run": {"workflow": "profile"},
        "geometry": {"format": "vmec", "file": "wout.nc", "surfaces": [0.25, 0.75]},
        "species": [
            {
                "name": "ion",
                "charge": 1,
                "mass_amu": 2,
                "density_m3": [1e19, 8e18],
                "temperature_keV": [1.0, 0.8],
            },
            {
                "name": "electron",
                "charge": -1,
                "mass_amu": 5.485799e-4,
                "density_m3": [1e19, 8e18],
                "temperature_keV": [1.5, 1.0],
            },
        ],
        "physics": {},
        "electric_field": {"mode": "prescribed", "value_kV_m": 0},
        "resolution": {"theta": 15, "zeta": 17, "pitch": 8, "speed": 4},
        "solver": {},
    }


def test_representative_toml_is_a_frozen_typed_case() -> None:
    case = Case.from_file(EXAMPLE)

    assert is_dataclass(case)
    assert case.schema == 1
    assert case.run.workflow == "ambipolar_profile"
    assert case.geometry.surfaces == (0.2, 0.35, 0.5, 0.65, 0.8)
    assert case.geometry_path == EXAMPLE.parent / "wout_w7x.nc"
    assert len(case.species) == 2
    assert len(case.case_id) == 64
    with pytest.raises(FrozenInstanceError):
        case.name = "changed"  # type: ignore[misc]


def test_json_and_reordered_mapping_have_same_semantic_id(tmp_path: Path) -> None:
    first = _mapping()
    second = dict(reversed(list(first.items())))
    json_path = tmp_path / "case.json"
    json_path.write_text(json.dumps(second, indent=2), encoding="utf-8")

    a = Case.from_mapping(first, source_path=tmp_path / "a.toml")
    b = Case.from_file(json_path)

    assert a.case_id == b.case_id
    assert a.to_dict() == b.to_dict()
    assert a.source_path != b.source_path


def test_default_pitch_speed_ramp_preserves_case_id_and_nondefault_is_semantic() -> (
    None
):
    implicit = _mapping()
    explicit = _mapping()
    explicit["resolution"]["pitch_speed_ramp"] = 1
    uniform = _mapping()
    uniform["resolution"]["pitch_speed_ramp"] = 0

    default_case = Case.from_mapping(implicit)
    explicit_case = Case.from_mapping(explicit)
    uniform_case = Case.from_mapping(uniform)

    assert default_case.resolution.pitch_speed_ramp == 1
    assert default_case.case_id == explicit_case.case_id
    assert "pitch_speed_ramp" not in default_case.to_dict()["resolution"]
    assert uniform_case.case_id != default_case.case_id
    assert uniform_case.to_dict()["resolution"]["pitch_speed_ramp"] == 0


def test_default_tail_retention_preserves_case_id_and_opt_in_is_semantic() -> None:
    implicit = Case.from_mapping(_mapping())
    explicit_false_mapping = _mapping()
    explicit_false_mapping["convergence"] = {"retain_legendre_tail": False}
    enabled_mapping = _mapping()
    enabled_mapping["convergence"] = {"retain_legendre_tail": True}

    explicit_false = Case.from_mapping(explicit_false_mapping)
    enabled = Case.from_mapping(enabled_mapping)

    assert implicit.case_id == explicit_false.case_id
    assert "retain_legendre_tail" not in implicit.to_dict()["convergence"]
    assert enabled.case_id != implicit.case_id
    assert enabled.to_dict()["convergence"]["retain_legendre_tail"] is True


def test_explicit_pitch_modes_are_immutable_semantic_content() -> None:
    default_case = Case.from_mapping(_mapping())
    explicit = _mapping()
    explicit["resolution"]["pitch"] = 8
    explicit["resolution"]["speed"] = 4
    explicit["resolution"]["pitch_modes_by_speed"] = [4, 5, 7, 8]

    case = Case.from_mapping(explicit)

    assert case.resolution.pitch_modes_by_speed == (4, 5, 7, 8)
    assert case.to_dict()["resolution"]["pitch_modes_by_speed"] == [4, 5, 7, 8]
    assert case.case_id != default_case.case_id


def test_direct_construction_freezes_nested_sequences_and_normalizes_paths() -> None:
    surfaces = [0.25, 0.75]
    densities = [1e19, 8e18]
    geometry = GeometryConfig(
        format="vmec", file="geometry/../wout.nc", surfaces=surfaces
    )  # type: ignore[arg-type]
    species = SpeciesConfig(
        name="ion",
        charge=1,
        mass_amu=2,
        density_m3=densities,
        temperature_keV=[1.0, 0.8],  # type: ignore[arg-type]
    )
    surfaces.append(0.9)
    densities[0] = 0

    assert geometry.file == Path("wout.nc")
    assert geometry.surfaces == (0.25, 0.75)
    assert species.density_m3 == (1e19, 8e18)


@pytest.mark.parametrize(
    ("mutation", "path", "correction"),
    [
        (
            lambda data: data["geometry"].update(surfaces=[1.2]),
            "geometry.surfaces[0]",
            "normalized radial surface",
        ),
        (
            lambda data: data["species"][0].update(density_m3=[1e19]),
            "species[0].density_m3",
            "match geometry.surfaces",
        ),
        (
            lambda data: data["solver"].update(method="tier_1"),
            "solver.method",
            "structured_direct",
        ),
        (
            lambda data: data["electric_field"].update(search_points=2),
            "electric_field.search_points",
            "three coarse electric-field samples",
        ),
        (
            lambda data: data["electric_field"].update(root_tolerance_kV_m=0.0),
            "electric_field.root_tolerance_kV_m",
            "stated physical or numerical range",
        ),
        (
            lambda data: data["electric_field"].update(max_root_iterations=0),
            "electric_field.max_root_iterations",
            "one bracket-refinement iteration",
        ),
        (
            lambda data: data["resolution"].update(pitch_speed_ramp=3),
            "resolution.pitch_speed_ramp",
            "one of",
        ),
        (
            lambda data: data["resolution"].update(
                pitch=8, speed=4, pitch_modes_by_speed=[4, 6, 8]
            ),
            "resolution.pitch_modes_by_speed",
            "one active pitch-mode count",
        ),
        (
            lambda data: data["resolution"].update(
                pitch=8, speed=4, pitch_modes_by_speed=[4, 7, 6, 8]
            ),
            "resolution.pitch_modes_by_speed",
            "Increase or retain",
        ),
        (
            lambda data: data["resolution"].update(
                pitch=8, speed=4, pitch_modes_by_speed=[4, 5, 6, 7]
            ),
            "resolution.pitch_modes_by_speed[-1]",
            "Retain the declared maximum",
        ),
        (
            lambda data: data["resolution"].update(
                pitch=8,
                speed=4,
                pitch_speed_ramp=0,
                pitch_modes_by_speed=[4, 5, 7, 8],
            ),
            "resolution.pitch_speed_ramp",
            "Remove pitch_speed_ramp",
        ),
        (
            lambda data: data["run"].update(workflo="profile"),
            "run.workflo",
            "correct its spelling",
        ),
    ],
)
def test_validation_errors_name_path_value_expectation_and_correction(
    mutation, path, correction
) -> None:
    data = _mapping()
    mutation(data)
    with pytest.raises(CaseValidationError) as caught:
        Case.from_mapping(data)
    message = str(caught.value)
    assert path in message
    assert "supplied" in message
    assert "expected" in message
    assert correction in message


def test_scan_preflight_counts_cartesian_and_bounds_launch() -> None:
    data = _mapping()
    data["scan"] = {
        "combine": "cartesian",
        "max_cases": 6,
        "axis": [
            {"path": "electric_field.value_kV_m", "values": [-1, 0, 1]},
            {"path": "species[ion].density_scale", "values": [0.5, 1.0]},
        ],
    }
    case = Case.from_mapping(data)
    assert isinstance(case.scan, ScanConfig)
    assert case.scan.case_count == 6

    data["scan"]["max_cases"] = 5
    with pytest.raises(
        CaseValidationError, match=r"scan\.axis.*at most scan\.max_cases"
    ):
        Case.from_mapping(data)


def test_zipped_scan_requires_equal_axis_lengths() -> None:
    data = _mapping()
    data["scan"] = {
        "combine": "zipped",
        "axis": [
            {"path": "electric_field.value_kV_m", "values": [-1, 0, 1]},
            {"path": "species[ion].density_scale", "values": [0.5, 1.0]},
        ],
    }
    with pytest.raises(CaseValidationError, match="equal value counts"):
        Case.from_mapping(data)


def test_scan_rejects_unknown_paths_and_unknown_species() -> None:
    data = _mapping()
    data["scan"] = {"axis": [{"path": "electric_field.typo", "values": [0]}]}
    with pytest.raises(CaseValidationError, match="supported numeric Case path"):
        Case.from_mapping(data)

    data["scan"]["axis"][0]["path"] = "species[missing].density_scale"
    with pytest.raises(CaseValidationError, match="supported numeric Case path"):
        Case.from_mapping(data)


def test_schema_outputs_are_complete_and_machine_readable(capsys) -> None:
    schema = case_json_schema()
    assert schema["properties"]["schema"] == {"const": 1}
    assert "species" in schema["required"]
    assert schema["properties"]["scan"]["properties"]["axis"]["minItems"] == 1
    assert (
        schema["properties"]["electric_field"]["properties"]["search_points"]["minimum"]
        == 3
    )
    assert "[[species]]" in COMMENTED_TOML_EXAMPLE

    assert cli.main(["schema", "--format", "json", "--quiet"]) == 0
    rendered = json.loads(capsys.readouterr().out)
    assert rendered["$id"].endswith("case-v1.json")


def test_validate_cli_reports_case_id_and_scan_preflight(
    tmp_path: Path, capsys
) -> None:
    data = _mapping()
    data["scan"] = {
        "axis": [{"path": "electric_field.value_kV_m", "values": [-1, 0, 1]}]
    }
    path = tmp_path / "case.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    assert cli.main(["validate", str(path), "--quiet"]) == 0
    output = capsys.readouterr().out
    assert "valid DKX case: ordered-independent" in output
    assert "case_id:" in output
    assert "scan: 3 cases" in output


def test_validate_cli_returns_two_for_precise_error(tmp_path: Path, capsys) -> None:
    data = _mapping()
    data["schema"] = 2
    path = tmp_path / "future.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    assert cli.main(["validate", str(path), "--quiet"]) == 2
    assert "schema: supplied 2; expected integer 1" in capsys.readouterr().err


def test_native_case_is_reexported_from_top_level() -> None:
    assert dkx.Case is Case
    assert "Case" in dkx.__all__
    assert dkx.case_json_schema is case_json_schema
