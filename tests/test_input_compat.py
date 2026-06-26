from __future__ import annotations

from pathlib import Path

import numpy as np

from sfincs_jax.input_compat import (
    bool_config_values,
    canonical_equilibrium_override,
    config_bool,
    config_float,
    config_int,
    effective_equilibrium_file,
    effective_psi_n_wish,
    effective_r_n_wish,
    effective_use_iterative_linear_solver,
    first_config_value,
    infer_phi_input_radial_coordinate_for_gradients,
    infer_input_radial_coordinate_for_gradients,
    infer_species_input_radial_coordinate_for_gradients,
    lookup_config_value,
    with_equilibrium_override,
)
from sfincs_jax.io import _resolve_equilibrium_file_from_namelist, localize_equilibrium_file_in_place, sfincs_jax_output_dict
from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.v3 import grids_from_namelist
from sfincs_jax.operators.profile_response.fblock import _dphi_hat_dpsi_hat_from_er
from sfincs_jax.v3_system import full_system_operator_from_namelist


def test_shared_config_lookup_handles_namelists_and_nested_mappings() -> None:
    input_path = Path(__file__).parent / "ref" / "pas_1species_PAS_noEr_tiny.input.namelist"
    nml = read_sfincs_input(input_path)
    nested = {
        "general": {"rhsmode": 4},
        "physicsParameters": {"includePhi1": False, "EParallelHat": 1.25},
        "adjointOptions": {"adjointParticleFluxOption": [True, False]},
    }

    assert lookup_config_value(nml, ("general",), "RHSMode", 1) == 1
    assert lookup_config_value(nested, ("general",), "RHSMode") == 4
    assert first_config_value([3, 4], default=0) == 3
    assert bool_config_values([1, 0, True]) == (True, False, True)
    assert config_int(nested, ("general",), "RHSMode") == 4
    assert config_bool(nested, ("physicsParameters",), "includePhi1", True) is False
    assert config_float(nested, ("physicsParameters",), "EParallelHat") == 1.25


def test_infer_input_radial_coordinate_for_gradients_legacy_multispecies_psin() -> None:
    input_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "upstream"
        / "fortran_multispecies"
        / "quick_2species_FPCollisions_noEr"
        / "input.namelist"
    )
    nml = read_sfincs_input(input_path)
    assert (
        infer_input_radial_coordinate_for_gradients(
            geom_params=nml.group("geometryParameters"),
            species_params=nml.group("speciesParameters"),
            phys_params=nml.group("physicsParameters"),
            default=4,
    )
        == 1
    )


def test_infer_gradient_coordinates_legacy_mixed_species_and_er() -> None:
    input_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "sfincs_examples"
        / "geometryScheme5_3species_loRes"
        / "input.namelist"
    )
    nml = read_sfincs_input(input_path)
    geom = nml.group("geometryParameters")
    species = nml.group("speciesParameters")
    phys = nml.group("physicsParameters")
    assert infer_species_input_radial_coordinate_for_gradients(geom_params=geom, species_params=species, default=4) == 2
    assert infer_phi_input_radial_coordinate_for_gradients(geom_params=geom, phys_params=phys, default=4) == 4
    assert (
        infer_input_radial_coordinate_for_gradients(
            geom_params=geom,
            species_params=species,
            phys_params=phys,
            default=4,
        )
        == 4
    )


def test_infer_gradient_coordinates_prefers_v3_default_when_mixed_fields_are_present() -> None:
    input_path = (
        Path(__file__).resolve().parent
        / "ref"
        / "pas_1species_PAS_noEr_tiny_withPhi1_inKinetic_linear.input.namelist"
    )
    nml = read_sfincs_input(input_path)
    geom = nml.group("geometryParameters")
    species = nml.group("speciesParameters")
    phys = nml.group("physicsParameters")
    assert infer_species_input_radial_coordinate_for_gradients(geom_params=geom, species_params=species, default=4) == 2
    assert infer_phi_input_radial_coordinate_for_gradients(geom_params=geom, phys_params=phys, default=4) == 4
    assert (
        infer_input_radial_coordinate_for_gradients(
            geom_params=geom,
            species_params=species,
            phys_params=phys,
            default=4,
        )
        == 4
    )


def test_effective_equilibrium_file_supports_legacy_jgboozer_alias() -> None:
    input_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "upstream"
        / "fortran_multispecies"
        / "HSX_FPCollisions_DKESTrajectories"
        / "input.namelist"
    )
    nml = read_sfincs_input(input_path)
    equilibrium_file = effective_equilibrium_file(geom_params=nml.group("geometryParameters"))
    assert str(equilibrium_file).strip('"').strip("'").endswith("hsx3free.bc")


def test_canonical_equilibrium_override_accepts_matching_wout_alias() -> None:
    assert (
        canonical_equilibrium_override(
            equilibrium_file="wout_test.nc",
            wout_path="wout_test.nc",
        )
        == "wout_test.nc"
    )


def test_canonical_equilibrium_override_rejects_conflicting_values() -> None:
    try:
        canonical_equilibrium_override(
            equilibrium_file="a.nc",
            wout_path="b.nc",
        )
    except ValueError as exc:
        assert "conflicting equilibrium overrides" in str(exc)
    else:
        raise AssertionError("expected conflicting override values to raise")


def test_with_equilibrium_override_preserves_source_path_and_updates_text() -> None:
    input_path = Path(__file__).parent / "ref" / "output_scheme5_1species_tiny.input.namelist"
    nml = read_sfincs_input(input_path)
    updated = with_equilibrium_override(nml=nml, wout_path="override_wout.nc")
    equilibrium_file = effective_equilibrium_file(geom_params=updated.group("geometryParameters"))
    assert equilibrium_file == "override_wout.nc"
    assert updated.source_path == nml.source_path
    assert updated.source_text is not None
    assert 'equilibriumFile = "override_wout.nc"' in updated.source_text


def test_effective_r_n_wish_supports_legacy_normradius_alias() -> None:
    input_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "upstream"
        / "fortran_multispecies"
        / "HSX_FPCollisions_DKESTrajectories"
        / "input.namelist"
    )
    nml = read_sfincs_input(input_path)
    assert effective_r_n_wish(geom_params=nml.group("geometryParameters"), default=0.5) == 0.22


def test_effective_psi_n_wish_respects_declared_input_radial_coordinate() -> None:
    assert effective_psi_n_wish(geom_params={"INPUTRADIALCOORDINATE": 1, "PSIN_WISH": 0.2}) == 0.2
    assert np.isclose(effective_psi_n_wish(geom_params={"INPUTRADIALCOORDINATE": 3, "RN_WISH": 0.4}), 0.16)
    assert np.isclose(
        effective_psi_n_wish(
            geom_params={"INPUTRADIALCOORDINATE": 2, "RHAT_WISH": 0.3},
            a_hat=0.6,
        ),
        0.25,
    )
    assert np.isclose(
        effective_psi_n_wish(
            geom_params={"INPUTRADIALCOORDINATE": 0, "PSIHAT_WISH": 0.04},
            psi_a_hat=0.2,
        ),
        0.2,
    )


def test_er_conversion_supports_psin_selected_surface(tmp_path: Path) -> None:
    input_path = tmp_path / "input.namelist"
    input_path.write_text(
        "&geometryParameters\n"
        "  geometryScheme = 1\n"
        "  inputRadialCoordinate = 1\n"
        "  psiN_wish = 0.25\n"
        "  inputRadialCoordinateForGradients = 4\n"
        "  psiAHat = 2.0\n"
        "  aHat = 4.0\n"
        "/\n"
        "&physicsParameters\n"
        "  Er = -3.0\n"
        "/\n",
        encoding="utf-8",
    )
    nml = read_sfincs_input(input_path)
    assert np.isclose(_dphi_hat_dpsi_hat_from_er(nml=nml, er=-3.0), 6.0)


def test_effective_use_iterative_linear_solver_supports_legacy_alias() -> None:
    input_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "upstream"
        / "fortran_multispecies"
        / "inductiveE_noEr"
        / "input.namelist"
    )
    nml = read_sfincs_input(input_path)
    assert effective_use_iterative_linear_solver(other_params=nml.group("otherNumericalParameters"), default=0) == 1


def test_sfincs_output_dict_uses_legacy_gradient_coordinate_inference() -> None:
    input_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "upstream"
        / "fortran_multispecies"
        / "inductiveE_noEr"
        / "input.namelist"
    )
    nml = read_sfincs_input(input_path)
    grids = grids_from_namelist(nml)
    data = sfincs_jax_output_dict(nml=nml, grids=grids)
    assert int(np.asarray(data["inputRadialCoordinateForGradients"]).reshape(-1)[0]) == 1


def test_sfincs_output_dict_uses_legacy_normradius_wish_for_bc_geometry() -> None:
    input_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "upstream"
        / "fortran_multispecies"
        / "HSX_FPCollisions_DKESTrajectories"
        / "input.namelist"
    )
    nml = read_sfincs_input(input_path)
    grids = grids_from_namelist(nml)
    data = sfincs_jax_output_dict(nml=nml, grids=grids)
    assert np.isclose(float(np.asarray(data["rN"]).reshape(-1)[0]), 0.22703830459418076)


def test_localize_equilibrium_file_in_place_patches_legacy_boozer_key(tmp_path: Path) -> None:
    source_input = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "upstream"
        / "fortran_multispecies"
        / "HSX_FPCollisions_DKESTrajectories"
        / "input.namelist"
    )
    dst_input = tmp_path / "input.namelist"
    dst_input.write_text(source_input.read_text(encoding="utf-8"), encoding="utf-8")
    localized = localize_equilibrium_file_in_place(input_namelist=dst_input, overwrite=True)
    assert localized is not None
    patched = dst_input.read_text(encoding="utf-8")
    assert f'JGboozer_file = "{localized.name}"' in patched


def test_localize_equilibrium_file_in_place_patches_nonstelsym_boozer_key(tmp_path: Path) -> None:
    src_bc = Path(__file__).resolve().parents[1] / "tests" / "ref" / "nonStelSym_tiny_geometryScheme12.bc"
    input_path = tmp_path / "input.namelist"
    input_path.write_text(
        "&geometryParameters\n"
        "  geometryScheme = 12\n"
        f'  JGboozer_file_NonStelSym = "{src_bc}"\n'
        "/\n",
        encoding="utf-8",
    )
    localized = localize_equilibrium_file_in_place(input_namelist=input_path, overwrite=True)
    assert localized is not None
    patched = input_path.read_text(encoding="utf-8")
    assert f'JGboozer_file_NonStelSym = "{localized.name}"' in patched


def test_localize_equilibrium_file_in_place_handles_no_equilibrium_file(tmp_path: Path) -> None:
    input_path = tmp_path / "input.namelist"
    input_path.write_text(
        "&geometryParameters\n"
        "  geometryScheme = 1\n"
        "/\n",
        encoding="utf-8",
    )

    assert localize_equilibrium_file_in_place(input_namelist=input_path, overwrite=True) is None
    assert "equilibriumFile" not in input_path.read_text(encoding="utf-8")


def test_localize_equilibrium_file_in_place_patches_unquoted_legacy_key(tmp_path: Path) -> None:
    source_dir = tmp_path / "equilibria"
    source_dir.mkdir()
    source_bc = source_dir / "source.bc"
    source_bc.write_text("placeholder", encoding="utf-8")
    input_path = tmp_path / "input.namelist"
    input_path.write_text(
        "&geometryParameters\n"
        "  geometryScheme = 10\n"
        f"  fort996boozer_file = {source_bc}\n"
        "/\n",
        encoding="utf-8",
    )

    localized = localize_equilibrium_file_in_place(input_namelist=input_path, overwrite=True)

    assert localized == (tmp_path / source_bc.name).resolve()
    assert localized.read_text(encoding="utf-8") == "placeholder"
    patched = input_path.read_text(encoding="utf-8")
    assert f'fort996boozer_file = "{source_bc.name}"' in patched


def test_resolve_equilibrium_prefers_vmec_netcdf_sibling(tmp_path: Path) -> None:
    txt = tmp_path / "eq.txt"
    nc = tmp_path / "eq.nc"
    txt.write_text("ascii placeholder", encoding="utf-8")
    nc.write_text("netcdf placeholder", encoding="utf-8")
    input_path = tmp_path / "input.namelist"
    input_path.write_text(
        "&geometryParameters\n"
        "  geometryScheme = 5\n"
        '  equilibriumFile = "eq.txt"\n'
        "/\n",
        encoding="utf-8",
    )
    nml = read_sfincs_input(input_path)
    resolved = _resolve_equilibrium_file_from_namelist(nml=nml)
    assert resolved == nc.resolve()


def test_sfincs_output_dict_preserves_legacy_er_coordinate_and_value() -> None:
    input_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "sfincs_examples"
        / "geometryScheme5_3species_loRes"
        / "input.namelist"
    )
    nml = read_sfincs_input(input_path)
    grids = grids_from_namelist(nml)
    data = sfincs_jax_output_dict(nml=nml, grids=grids)
    psi_a_hat = float(np.asarray(data["psiAHat"]).reshape(-1)[0])
    a_hat = float(np.asarray(data["aHat"]).reshape(-1)[0])
    r_n = float(np.asarray(data["rN"]).reshape(-1)[0])
    er = float(np.asarray(data["Er"]).reshape(-1)[0])
    ddrhat2ddpsihat = a_hat / (2.0 * psi_a_hat * r_n)
    assert int(np.asarray(data["inputRadialCoordinateForGradients"]).reshape(-1)[0]) == 4
    assert np.isclose(er, -8.5897)
    assert np.isclose(float(np.asarray(data["dPhiHatdpsiHat"]).reshape(-1)[0]), ddrhat2ddpsihat * (-er))


def test_dphi_hat_dpsi_hat_from_er_supports_geometry_scheme1_with_er() -> None:
    input_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "sfincs_examples"
        / "tokamak_1species_FPCollisions_withEr_DKESTrajectories"
        / "input.namelist"
    )
    nml = read_sfincs_input(input_path)
    er = float(nml.group("physicsParameters").get("Er", nml.group("physicsParameters").get("ER", 0.0)))
    dphi = _dphi_hat_dpsi_hat_from_er(nml=nml, er=er)
    assert np.isfinite(dphi)
    assert dphi != 0.0


def test_full_system_operator_uses_split_legacy_gradient_coordinates() -> None:
    input_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "sfincs_examples"
        / "geometryScheme5_3species_loRes"
        / "input.namelist"
    )
    nml = read_sfincs_input(input_path)
    op = full_system_operator_from_namelist(nml=nml)
    grids = grids_from_namelist(nml)
    data = sfincs_jax_output_dict(nml=nml, grids=grids)
    psi_a_hat = float(np.asarray(data["psiAHat"]).reshape(-1)[0])
    a_hat = float(np.asarray(data["aHat"]).reshape(-1)[0])
    r_n = float(np.asarray(data["rN"]).reshape(-1)[0])
    ddrhat2ddpsihat = a_hat / (2.0 * psi_a_hat * r_n)
    assert np.isclose(float(op.dphi_hat_dpsi_hat), ddrhat2ddpsihat * 8.5897)
    assert np.allclose(
        np.asarray(op.dn_hat_dpsi_hat),
        ddrhat2ddpsihat * np.asarray([-15.0, -15.5, -0.025]),
    )


def test_full_system_operator_prefers_v3_default_gradients_for_mixed_phi1_fixture() -> None:
    input_path = Path(__file__).resolve().parent / "ref" / "pas_1species_PAS_noEr_tiny_withPhi1_inKinetic_linear.input.namelist"
    nml = read_sfincs_input(input_path)
    op = full_system_operator_from_namelist(nml=nml)
    psi_a_hat = -0.384935
    a_hat = 0.5109
    r_n = 0.5
    ddrhat2ddpsihat = a_hat / (2.0 * psi_a_hat * r_n)
    assert np.allclose(np.asarray(op.dn_hat_dpsi_hat), ddrhat2ddpsihat * np.asarray([-0.5]))
    assert np.allclose(np.asarray(op.dt_hat_dpsi_hat), ddrhat2ddpsihat * np.asarray([-2.0]))
    assert np.isclose(float(op.dphi_hat_dpsi_hat), 0.0)
