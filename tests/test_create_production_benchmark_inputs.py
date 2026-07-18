from __future__ import annotations

import json
from pathlib import Path

from dkx.validation import release as bench_inputs


REPO_ROOT = Path(__file__).resolve().parents[1]
MIN_3D = {"NTHETA": 25, "NZETA": 51, "NX": 4, "NXI": 100}
MIN_TOKAMAK = {"NTHETA": 33, "NX": 12, "NXI": 140}
MIN_TOKAMAK_PAS_NOER = {"NTHETA": 89, "NX": 24, "NXI": 300}
POLICY_TEXT = (
    "preserve nominal grid, but enforce 3D >= 25x51x4x100 "
    "and tokamak >= 33x1x12x140 "
    "(RHSMode=1 PAS/no-Er tokamak >= 89x1x24x300); "
    "production timing rows target Fortran v3 >= 10 s"
)


def _write_input(
    path: Path,
    *,
    ntheta: int,
    nzeta: int,
    nx: int,
    nxi: int,
    equilibrium_file: Path | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    geometry = ""
    if equilibrium_file is not None:
        geometry = f'\n&geometryParameters\n  equilibriumFile = "{equilibrium_file}"\n/\n'
    path.write_text(
        (
            "&resolutionParameters\n"
            f"  NTHETA = {ntheta}\n"
            f"  NZETA = {nzeta}\n"
            f"  NX = {nx}\n"
            f"  NXI = {nxi}\n"
            "/\n"
            f"{geometry}"
        ),
        encoding="utf-8",
    )


def test_production_manifest_generator_defaults_to_untracked_outputs_tree() -> None:
    assert (
        bench_inputs.DEFAULT_OUT_ROOT
        == REPO_ROOT / "outputs" / "benchmarks" / "production_resolution_inputs_2026-05-04"
    )


def _assert_floor(resolution: dict[str, int], floor: dict[str, int]) -> None:
    for key, value in floor.items():
        assert int(resolution[key]) >= int(value)


def test_generator_enforces_research_baseline_on_examples_and_external_inputs(tmp_path: Path) -> None:
    examples_root = tmp_path / "examples"
    _write_input(examples_root / "tokamak_demo" / "input.namelist", ntheta=7, nzeta=1, nx=1, nxi=8)
    _write_input(examples_root / "stellarator_demo" / "input.namelist", ntheta=9, nzeta=11, nx=2, nxi=10)

    source_wout = tmp_path / "source_geometry" / "wout_test.nc"
    source_wout.parent.mkdir(parents=True)
    source_wout.write_bytes(b"mock-netcdf")
    external_input = tmp_path / "external" / "finite_beta" / "rho_0p5" / "input.namelist"
    _write_input(external_input, ntheta=13, nzeta=15, nx=5, nxi=8, equilibrium_file=source_wout)

    out_root = tmp_path / "production_inputs"
    assert (
        bench_inputs.production_inputs_main(
            [
                "--examples-root",
                str(examples_root),
                "--additional-input",
                str(tmp_path / "missing_input.namelist"),
                "--out-root",
                str(out_root),
                "--external-input",
                str(external_input),
                "--clean",
            ]
        )
        == 0
    )

    manifest = json.loads((out_root / "manifest.json").read_text(encoding="utf-8"))
    cases = {case["case"]: case for case in manifest["cases"]}

    assert manifest["minimum_3d_resolution"] == MIN_3D
    assert manifest["minimum_tokamak_resolution"] == MIN_TOKAMAK
    assert manifest["minimum_tokamak_pas_noer_resolution"] == MIN_TOKAMAK_PAS_NOER
    assert manifest["target_fortran_min_runtime_s"] == 10.0

    assert cases["tokamak_demo"]["benchmark_resolution"] == {"NTHETA": 33, "NZETA": 1, "NX": 12, "NXI": 140}
    assert cases["stellarator_demo"]["benchmark_resolution"] == MIN_3D
    assert cases["tokamak_demo"]["size_estimate"]["total_unknowns_estimate"] == 55442
    assert cases["tokamak_demo"]["size_estimate"]["run_recommendation"] == "remote_or_cluster_only"
    tokamak_text = (out_root / cases["tokamak_demo"]["input"]).read_text(encoding="utf-8")
    assert "preconditioner_x = 1" in tokamak_text
    assert "whichParallelSolverToFactorPreconditioner = 1" in tokamak_text
    assert "useIterativeLinearSolver = .false." not in tokamak_text

    external_case_name = next(name for name in cases if name.startswith("external_"))
    assert cases[external_case_name]["benchmark_resolution"] == {"NTHETA": 25, "NZETA": 51, "NX": 5, "NXI": 100}
    _assert_floor(cases[external_case_name]["benchmark_resolution"], MIN_3D)
    assert cases[external_case_name]["resolution_policy"] == POLICY_TEXT

    localized_input = out_root / cases[external_case_name]["input"]
    localized_text = localized_input.read_text(encoding="utf-8")
    assert f'"{source_wout}"' not in localized_text
    assert 'equilibriumFile = "wout_test.nc"' in localized_text
    assert (localized_input.parent / "wout_test.nc").read_bytes() == b"mock-netcdf"


def test_generator_can_preserve_external_resolution_for_reproduction(tmp_path: Path) -> None:
    examples_root = tmp_path / "examples"
    _write_input(examples_root / "stellarator_demo" / "input.namelist", ntheta=25, nzeta=51, nx=4, nxi=100)
    external_input = tmp_path / "external" / "finite_beta" / "rho_0p5" / "input.namelist"
    _write_input(external_input, ntheta=13, nzeta=15, nx=5, nxi=8)

    out_root = tmp_path / "production_inputs"
    assert (
        bench_inputs.production_inputs_main(
            [
                "--examples-root",
                str(examples_root),
                "--additional-input",
                str(tmp_path / "missing_input.namelist"),
                "--out-root",
                str(out_root),
                "--external-input",
                str(external_input),
                "--preserve-external-resolution",
                "--clean",
            ]
        )
        == 0
    )

    manifest = json.loads((out_root / "manifest.json").read_text(encoding="utf-8"))
    cases = {case["case"]: case for case in manifest["cases"]}
    external_case_name = next(name for name in cases if name.startswith("external_"))
    assert cases[external_case_name]["benchmark_resolution"] == {"NTHETA": 13, "NZETA": 15, "NX": 5, "NXI": 8}
    assert cases[external_case_name]["resolution_policy"] == "preserve authored external resolution"


def test_generator_relabels_historic_external_deck_with_benchmark_resolution(tmp_path: Path) -> None:
    examples_root = tmp_path / "examples"
    _write_input(examples_root / "stellarator_demo" / "input.namelist", ntheta=25, nzeta=51, nx=4, nxi=100)
    external_input = (
        tmp_path
        / "external"
        / "outputs"
        / "dkx_rhsmode1_profile_current_profiling"
        / "cpu_17x21x12_deck"
        / "finite_beta"
        / "input.namelist"
    )
    _write_input(external_input, ntheta=17, nzeta=21, nx=5, nxi=12)

    out_root = tmp_path / "production_inputs"
    assert (
        bench_inputs.production_inputs_main(
            [
                "--examples-root",
                str(examples_root),
                "--additional-input",
                str(tmp_path / "missing_input.namelist"),
                "--out-root",
                str(out_root),
                "--external-input",
                str(external_input),
                "--clean",
            ]
        )
        == 0
    )

    manifest = json.loads((out_root / "manifest.json").read_text(encoding="utf-8"))
    cases = {case["case"]: case for case in manifest["cases"]}
    assert any("cpu_25x51x5x100_deck" in name for name in cases)
    assert not any("cpu_17x21x12_deck" in name for name in cases)


def test_generator_uses_calibrated_floor_for_tokamak_pas_noer(tmp_path: Path) -> None:
    examples_root = tmp_path / "examples"
    noer_input = examples_root / "tokamak_pas_noer" / "input.namelist"
    wither_input = examples_root / "tokamak_pas_wither" / "input.namelist"
    _write_input(noer_input, ntheta=7, nzeta=1, nx=2, nxi=10)
    _write_input(wither_input, ntheta=7, nzeta=1, nx=2, nxi=10)
    with noer_input.open("a", encoding="utf-8") as stream:
        stream.write(
            "\n&physicsParameters\n"
            "  RHSMode = 1\n"
            "  collisionOperator = 1\n"
            "  includeXDotTerm = .true.\n"
            "  dPhiHatdrN = 0.0\n"
            "/\n"
        )
    with wither_input.open("a", encoding="utf-8") as stream:
        stream.write(
            "\n&physicsParameters\n"
            "  RHSMode = 1\n"
            "  collisionOperator = 1\n"
            "  includeXDotTerm = .true.\n"
            "  dPhiHatdrN = 1.0\n"
            "/\n"
        )

    out_root = tmp_path / "production_inputs"
    assert (
        bench_inputs.production_inputs_main(
            [
                "--examples-root",
                str(examples_root),
                "--additional-input",
                str(tmp_path / "missing_input.namelist"),
                "--out-root",
                str(out_root),
                "--clean",
            ]
        )
        == 0
    )

    manifest = json.loads((out_root / "manifest.json").read_text(encoding="utf-8"))
    cases = {case["case"]: case for case in manifest["cases"]}

    assert cases["tokamak_pas_noer"]["benchmark_resolution"] == {
        "NTHETA": 89,
        "NZETA": 1,
        "NX": 24,
        "NXI": 300,
    }
    assert cases["tokamak_pas_wither"]["benchmark_resolution"] == {
        "NTHETA": 33,
        "NZETA": 1,
        "NX": 12,
        "NXI": 140,
    }


def test_generator_estimates_large_pas_xdot_case_as_remote_only(tmp_path: Path) -> None:
    examples_root = tmp_path / "examples"
    input_path = examples_root / "stellarator_xdot" / "input.namelist"
    _write_input(input_path, ntheta=17, nzeta=21, nx=5, nxi=12)
    with input_path.open("a", encoding="utf-8") as stream:
        stream.write(
            "\n&speciesParameters\n"
            "  Zs = 1.0 -1.0\n"
            "/\n"
            "\n&physicsParameters\n"
            "  collisionOperator = 1\n"
            "  includeXDotTerm = .true.\n"
            "  dPhiHatdrN = 1.0\n"
            "/\n"
        )

    out_root = tmp_path / "production_inputs"
    assert (
        bench_inputs.production_inputs_main(
            [
                "--examples-root",
                str(examples_root),
                "--additional-input",
                str(tmp_path / "missing_input.namelist"),
                "--out-root",
                str(out_root),
                "--clean",
            ]
        )
        == 0
    )

    manifest = json.loads((out_root / "manifest.json").read_text(encoding="utf-8"))
    case = manifest["cases"][0]
    size = case["size_estimate"]

    assert case["benchmark_resolution"] == {"NTHETA": 25, "NZETA": 51, "NX": 5, "NXI": 100}
    _assert_floor(case["benchmark_resolution"], MIN_3D)
    assert size["species_count"] == 2
    assert size["collision_operator"] == 1
    assert size["include_xdot_requested"] is True
    assert size["include_xdot"] is True
    assert size["include_xdot_effective"] is True
    assert size["total_unknowns_estimate"] == 1275010
    assert size["dense_matrix_nbytes_estimate"] > 1_000_000_000_000
    assert size["conservative_csr_nbytes_estimate"] > 10_000_000_000
    assert size["run_recommendation"] == "remote_or_cluster_only"


def test_generator_does_not_overestimate_zero_er_xdot_case(tmp_path: Path) -> None:
    examples_root = tmp_path / "examples"
    input_path = examples_root / "stellarator_zero_er_xdot" / "input.namelist"
    _write_input(input_path, ntheta=17, nzeta=21, nx=5, nxi=12)
    with input_path.open("a", encoding="utf-8") as stream:
        stream.write(
            "\n&speciesParameters\n"
            "  Zs = 1.0 -1.0\n"
            "/\n"
            "\n&physicsParameters\n"
            "  collisionOperator = 1\n"
            "  includeXDotTerm = .true.\n"
            "  dPhiHatdrN = 0.0\n"
            "/\n"
        )

    out_root = tmp_path / "production_inputs"
    assert (
        bench_inputs.production_inputs_main(
            [
                "--examples-root",
                str(examples_root),
                "--additional-input",
                str(tmp_path / "missing_input.namelist"),
                "--out-root",
                str(out_root),
                "--clean",
            ]
        )
        == 0
    )

    manifest = json.loads((out_root / "manifest.json").read_text(encoding="utf-8"))
    size = manifest["cases"][0]["size_estimate"]

    assert size["include_xdot_requested"] is True
    assert size["include_xdot"] is False
    assert size["include_xdot_effective"] is False
    assert size["conservative_csr_nbytes_estimate"] < 10_000_000_000
    assert size["run_recommendation"] == "remote_or_cluster_only"


def test_generator_default_floor_keeps_hsx_fp_at_authored_large_resolution(tmp_path: Path) -> None:
    examples_root = tmp_path / "examples"
    _write_input(
        examples_root / "HSX_FPCollisions_fullTrajectories" / "input.namelist",
        ntheta=25,
        nzeta=115,
        nx=5,
        nxi=149,
    )
    _write_input(examples_root / "tokamak_demo" / "input.namelist", ntheta=5, nzeta=1, nx=1, nxi=4)

    out_root = tmp_path / "production_inputs"
    assert (
        bench_inputs.production_inputs_main(
            [
                "--examples-root",
                str(examples_root),
                "--additional-input",
                str(tmp_path / "missing_input.namelist"),
                "--out-root",
                str(out_root),
                "--clean",
            ]
        )
        == 0
    )

    manifest = json.loads((out_root / "manifest.json").read_text(encoding="utf-8"))
    cases = {case["case"]: case for case in manifest["cases"]}
    hsx_res = cases["HSX_FPCollisions_fullTrajectories"]["benchmark_resolution"]
    tok_res = cases["tokamak_demo"]["benchmark_resolution"]

    assert hsx_res == {"NTHETA": 25, "NZETA": 115, "NX": 5, "NXI": 149}
    _assert_floor(hsx_res, MIN_3D)
    assert tok_res == {"NTHETA": 33, "NZETA": 1, "NX": 12, "NXI": 140}
    _assert_floor(tok_res, MIN_TOKAMAK)
