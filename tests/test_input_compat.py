from __future__ import annotations

from pathlib import Path

import numpy as np

from sfincs_jax.input_compat import (
    _resolve_equilibrium_file_from_namelist,
    bool_config_values,
    canonical_equilibrium_override,
    config_bool,
    config_float,
    config_int,
    effective_equilibrium_file,
    effective_psi_a_hat,
    effective_psi_n_wish,
    effective_r_n_wish,
    effective_use_iterative_linear_solver,
    first_config_value,
    infer_phi_input_radial_coordinate_for_gradients,
    infer_input_radial_coordinate_for_gradients,
    infer_species_input_radial_coordinate_for_gradients,
    localize_equilibrium_file_in_place,
    lookup_config_value,
    render_input_with_equilibrium_override,
    with_equilibrium_override,
)
from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.drift_kinetic import kinetic_operator_from_namelist


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


def test_config_lookup_covers_top_level_case_insensitive_and_empty_values() -> None:
    flat = {"rhsMode": [3], "includePhi1": (), "eparallelhat": 0.75}

    assert lookup_config_value(object(), ("missing",), "RHSMode", "fallback") == "fallback"
    assert lookup_config_value(flat, ("unused",), "RHSMODE") == [3]
    assert config_int(flat, ("unused",), "rhsMode") == 3
    assert config_bool(flat, ("unused",), "includePhi1", default=True) is True
    assert config_float(flat, ("unused",), "EParallelHat") == 0.75
    assert first_config_value([], default="fallback") == "fallback"
    assert bool_config_values(None) == ()
    assert bool_config_values(0) == (False,)


def test_infer_input_radial_coordinate_for_gradients_legacy_multispecies_psin() -> None:
    input_path = (
        Path(__file__).resolve().parent / "ref" / "multispecies_quick_2species_FPCollisions_noEr.input.namelist"
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
        Path(__file__).resolve().parent / "ref" / "multispecies_HSX_FPCollisions_DKESTrajectories.input.namelist"
    )
    nml = read_sfincs_input(input_path)
    equilibrium_file = effective_equilibrium_file(geom_params=nml.group("geometryParameters"))
    assert str(equilibrium_file).strip('"').strip("'").endswith("hsx3free.bc")


def test_effective_equilibrium_file_covers_legacy_scheme_aliases() -> None:
    assert effective_equilibrium_file(geom_params={"GEOMETRYSCHEME": 10, "FORT996BOOZER_FILE": "fort.bc"}) == "fort.bc"
    assert effective_equilibrium_file(geom_params={"GEOMETRYSCHEME": 12, "JGBOOZER_FILE_NONSTELSYM": "non.bc"}) == "non.bc"
    assert effective_equilibrium_file(geom_params={"GEOMETRYSCHEME": 5}) is None


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


def test_equilibrium_override_rendering_inserts_only_when_geometry_group_exists() -> None:
    inserted = render_input_with_equilibrium_override(
        source_text="&geometryParameters\n  geometryScheme = 5\n/\n",
        equilibrium_override="wout_inserted.nc",
    )
    unchanged = render_input_with_equilibrium_override(
        source_text="&physicsParameters\n/\n",
        equilibrium_override="unused.nc",
    )
    replaced = render_input_with_equilibrium_override(
        source_text="&geometryParameters\n  equilibriumFile = 'old.nc'\n/\n",
        equilibrium_override="new.nc",
    )

    assert 'equilibriumFile = "wout_inserted.nc"' in inserted
    assert unchanged == "&physicsParameters\n/\n"
    assert 'equilibriumFile = "new.nc"' in replaced


def test_with_equilibrium_override_returns_original_when_no_override() -> None:
    input_path = Path(__file__).parent / "ref" / "output_scheme5_1species_tiny.input.namelist"
    nml = read_sfincs_input(input_path)

    assert with_equilibrium_override(nml=nml) is nml


def test_effective_r_n_wish_supports_legacy_normradius_alias() -> None:
    input_path = (
        Path(__file__).resolve().parent / "ref" / "multispecies_HSX_FPCollisions_DKESTrajectories.input.namelist"
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


def test_effective_psi_n_wish_reports_missing_normalizations_and_bad_coordinate() -> None:
    try:
        effective_psi_n_wish(geom_params={"INPUTRADIALCOORDINATE": 0, "PSIHAT_WISH": 0.1})
    except ValueError as exc:
        assert "psi_a_hat is required" in str(exc)
    else:
        raise AssertionError("expected psiHat conversion without psi_a_hat to fail")

    try:
        effective_psi_n_wish(geom_params={"INPUTRADIALCOORDINATE": 2, "RHAT_WISH": 0.1})
    except ValueError as exc:
        assert "a_hat is required" in str(exc)
    else:
        raise AssertionError("expected rHat conversion without a_hat to fail")

    try:
        effective_psi_n_wish(geom_params={"INPUTRADIALCOORDINATE": 99})
    except ValueError as exc:
        assert "Invalid inputRadialCoordinate=99" in str(exc)
    else:
        raise AssertionError("expected invalid inputRadialCoordinate to fail")


def test_effective_psi_a_hat_prefers_geometry_then_physics_default() -> None:
    assert effective_psi_a_hat(geom_params={"PSIAHAT": 3.0}, phys_params={"PSIAHAT": 4.0}, default=1.0) == 3.0
    assert effective_psi_a_hat(geom_params={}, phys_params={"PSIAHAT": 4.0}, default=1.0) == 4.0
    assert effective_psi_a_hat(geom_params={}, phys_params={}, default=1.0) == 1.0


def test_gradient_coordinate_inference_honors_explicit_override() -> None:
    geom = {"INPUTRADIALCOORDINATEFORGRADIENTS": 2}
    assert infer_species_input_radial_coordinate_for_gradients(geom_params=geom, species_params={}, default=4) == 2
    assert infer_phi_input_radial_coordinate_for_gradients(geom_params=geom, phys_params={}, default=4) == 2
    assert (
        infer_input_radial_coordinate_for_gradients(
            geom_params=geom,
            species_params={},
            phys_params={},
            default=4,
        )
        == 2
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
    op = kinetic_operator_from_namelist(nml)
    assert np.isclose(float(op.dphi_hat_dpsi_hat_kinetic), 6.0)


def test_effective_use_iterative_linear_solver_supports_legacy_alias() -> None:
    input_path = (
        Path(__file__).resolve().parent / "ref" / "multispecies_inductiveE_noEr.input.namelist"
    )
    nml = read_sfincs_input(input_path)
    assert effective_use_iterative_linear_solver(other_params=nml.group("otherNumericalParameters"), default=0) == 1


def test_gradient_coordinate_inference_on_inductive_deck() -> None:
    input_path = (
        Path(__file__).resolve().parent / "ref" / "multispecies_inductiveE_noEr.input.namelist"
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


def test_normradius_wish_alias_resolves_bc_geometry_surface() -> None:
    from sfincs_jax.magnetic_geometry import selected_r_n_from_bc

    input_path = (
        Path(__file__).resolve().parent / "ref" / "multispecies_HSX_FPCollisions_DKESTrajectories.input.namelist"
    )
    nml = read_sfincs_input(input_path)
    geom_params = nml.group("geometryParameters")
    # The legacy ``normradius_wish`` alias feeds the surface selection.
    r_n_wish = effective_r_n_wish(geom_params=geom_params)
    assert np.isclose(float(r_n_wish), 0.22)
    bc_path = _resolve_equilibrium_file_from_namelist(nml=nml)
    assert bc_path is not None
    r_n = selected_r_n_from_bc(
        path=bc_path, geometry_scheme=11, r_n_wish=float(r_n_wish), vmec_radial_option=1
    )
    assert np.isclose(float(r_n), 0.22703830459418076)


def test_localize_equilibrium_file_in_place_patches_legacy_boozer_key(tmp_path: Path) -> None:
    source_input = (
        Path(__file__).resolve().parent / "ref" / "multispecies_HSX_FPCollisions_DKESTrajectories.input.namelist"
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


def test_er_coordinate_and_split_gradient_coordinates_scheme5_example() -> None:
    from sfincs_jax.drift_kinetic import _geometry_and_radial
    from sfincs_jax.inputs import load_sfincs_input
    from sfincs_jax.run import _grids_from_input

    input_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "sfincs_examples"
        / "geometryScheme5_3species_loRes"
        / "input.namelist"
    )
    inp = load_sfincs_input(input_path)
    raw = inp.raw
    grids = _grids_from_input(inp, raw)
    _geom, radial = _geometry_and_radial(nml=raw, grids=grids)
    op = kinetic_operator_from_namelist(raw)

    er = float(raw.group("physicsParameters").get("ER", 0.0))
    assert np.isclose(er, -8.5897)
    ddrhat2ddpsihat = float(radial.a_hat) / (2.0 * float(radial.psi_a_hat) * float(radial.r_n))
    assert np.isclose(float(op.dphi_hat_dpsi_hat), ddrhat2ddpsihat * 8.5897)
    assert np.isclose(float(op.dphi_hat_dpsi_hat_kinetic), ddrhat2ddpsihat * (-er))
    assert np.allclose(
        np.asarray(op.dn_hat_dpsi_hat),
        ddrhat2ddpsihat * np.asarray([-15.0, -15.5, -0.025]),
    )


def test_dphi_hat_dpsi_hat_from_er_supports_geometry_scheme1_with_er() -> None:
    input_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "sfincs_examples"
        / "tokamak_1species_FPCollisions_withEr_DKESTrajectories"
        / "input.namelist"
    )
    nml = read_sfincs_input(input_path)
    op = kinetic_operator_from_namelist(nml)
    dphi = float(op.dphi_hat_dpsi_hat_kinetic)
    assert np.isfinite(dphi)
    assert dphi != 0.0


def test_operator_prefers_v3_default_gradients_for_mixed_phi1_fixture() -> None:
    input_path = Path(__file__).resolve().parent / "ref" / "pas_1species_PAS_noEr_tiny_withPhi1_inKinetic_linear.input.namelist"
    nml = read_sfincs_input(input_path)
    op = kinetic_operator_from_namelist(nml)
    psi_a_hat = -0.384935
    a_hat = 0.5109
    r_n = 0.5
    ddrhat2ddpsihat = a_hat / (2.0 * psi_a_hat * r_n)
    assert np.allclose(np.asarray(op.dn_hat_dpsi_hat), ddrhat2ddpsihat * np.asarray([-0.5]))
    assert np.allclose(np.asarray(op.dt_hat_dpsi_hat), ddrhat2ddpsihat * np.asarray([-2.0]))
    assert np.isclose(float(op.dphi_hat_dpsi_hat), 0.0)
