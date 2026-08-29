"""The first native Case -> solve -> Result route never passes through a deck."""

from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest

import dkx
from dkx.constants import RadialCoordinates
from dkx.units import HEAT_FLUX, PARALLEL_CURRENT, PARTICLE_FLUX


def _case():
    return dkx.Case.from_mapping(
        {
            "schema": 1,
            "name": "native_tokamak_profile",
            "run": {"workflow": "profile", "progress": False},
            "geometry": {
                "format": "analytic",
                "file": "tokamak",
                "surfaces": [0.09, 0.16],
            },
            "species": [
                {
                    "name": "deuterium",
                    "charge": 1,
                    "mass_amu": 2.014,
                    "density_m3": [8.0e19, 7.0e19],
                    "temperature_keV": [1.0, 0.8],
                }
            ],
            "physics": {
                "collisions": "pitch_angle_scattering",
                "magnetic_drifts": "dkes",
                "phi1": "off",
            },
            "electric_field": {"mode": "prescribed", "value_kV_m": 0.0},
            "resolution": {"theta": 9, "zeta": 1, "pitch": 8, "speed": 4},
            "solver": {"method": "auto", "relative_tolerance": 1.0e-8},
            "output": {"file": "native-result.nc", "plots": False},
        }
    )


def _vmec_case():
    base = _case()
    return replace(
        base,
        name="native_vmec_profile",
        geometry=replace(
            base.geometry,
            format="vmec",
            file=Path("ref/wout_up_down_asymmetric_tokamak.nc"),
            surfaces=(0.16, 0.25),
        ),
        source_path=Path(__file__).resolve(),
    )


def _ambipolar_case():
    base = _case()
    return replace(
        base,
        name="native_ambipolar_profile",
        run=replace(base.run, workflow="ambipolar_profile"),
        electric_field=replace(
            base.electric_field,
            mode="ambipolar",
            value_kV_m=None,
            search_kV_m=(-5.0, 5.0),
            search_points=5,
            root_tolerance_kV_m=0.05,
            max_root_iterations=8,
        ),
    )


def test_native_case_solves_without_namelist_conversion(monkeypatch, tmp_path) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("native execution serialized or parsed a SFINCS namelist")

    monkeypatch.setattr("dkx.inputs.SfincsInput.to_namelist", forbidden)
    monkeypatch.setattr("dkx.run.parse_sfincs_input_text", forbidden)
    path = tmp_path / "result.nc"
    result = dkx.run(_case(), out=path)

    assert isinstance(result, dkx.Result)
    assert path.is_file()
    assert result.metadata["converged"] is True
    assert result.particle_flux_m2_s.shape == (2, 1)
    assert np.all(np.isfinite(result.particle_flux_m2_s))
    assert np.any(result.particle_flux_m2_s != 0.0)
    assert np.max(result.primal_residual) < 1.0e-8
    assert result.dimensions["particle_flux_m2_s"] == ("surface", "species")
    assert result.certificate()["case_id"] == _case().case_id

    loaded = dkx.Result.load(path)
    assert loaded.case_id == result.case_id
    np.testing.assert_array_equal(loaded.species, ["deuterium"])
    np.testing.assert_allclose(loaded.particle_flux_m2_s, result.particle_flux_m2_s)
    assert result.plot(tmp_path / "profile.png").is_file()


def test_native_ambipolar_result_preserves_scan_roots_and_selection(
    monkeypatch, tmp_path
) -> None:
    from dkx.workflows.ambipolar_native import (
        NativeAmbipolarRoot,
        NativeAmbipolarSurface,
        RootEvaluation,
    )

    base = _case()
    case = replace(
        base,
        name="native_ambipolar_profile",
        run=replace(base.run, workflow="ambipolar_profile"),
        electric_field=replace(
            base.electric_field,
            mode="ambipolar",
            value_kV_m=None,
            search_kV_m=(-4.0, 4.0),
        ),
    )
    calls = []

    def fake_surface(_problem, *, previous_root_kv_m, **_controls):
        surface_index = len(calls)
        root_field = -1.5 + 0.25 * surface_index
        evaluation = RootEvaluation(
            electric_field_kv_m=root_field,
            radial_current_a_m2=1.0e-10,
            particle_flux_m2_s=np.asarray([2.0 + surface_index]),
            heat_flux_w_m2=np.asarray([3.0 + surface_index]),
            parallel_current_a_t_m2=4.0 + surface_index,
            residual_norm=1.0e-12,
            stage="root_refinement",
        )
        coarse = replace(
            evaluation,
            electric_field_kv_m=-4.0,
            radial_current_a_m2=-2.0,
            stage="coarse_scan",
        )
        root = NativeAmbipolarRoot(
            electric_field_kv_m=root_field,
            radial_current_a_m2=evaluation.radial_current_a_m2,
            slope_a_m2_per_kv_m=2.0,
            root_type="ion",
            bracket_kv_m=(-2.0, 0.0),
            evaluation=evaluation,
        )
        calls.append(previous_root_kv_m)
        return NativeAmbipolarSurface(
            evaluations=(coarse, evaluation),
            roots=(root,),
            selected_root=0,
            selected=evaluation,
            status="bracketed_root",
            solve_seconds=0.01,
            batch_chunk_size=2,
            batch_chunks=1,
        )

    monkeypatch.setattr(
        "dkx.workflows.ambipolar_native.solve_native_ambipolar_surface",
        fake_surface,
    )
    result = dkx.run(case, out=tmp_path / "ambipolar.nc")

    assert calls == [None, -1.5]
    np.testing.assert_allclose(result.electric_field_kV_m, [-1.5, -1.25])
    np.testing.assert_allclose(result.particle_flux_m2_s[:, 0], [2.0, 3.0])
    np.testing.assert_array_equal(result.ambipolar_root_count, [1, 1])
    np.testing.assert_array_equal(result.ambipolar_status, ["bracketed_root"] * 2)
    assert result.dimensions["radial_current_A_m2"] == ("surface", "evaluation")
    assert result.metadata["ambipolar_all_surfaces_bracketed"] is True
    loaded = dkx.Result.load(tmp_path / "ambipolar.nc")
    np.testing.assert_allclose(loaded.ambipolar_root_kV_m[:, 0], [-1.5, -1.25])


def test_native_ambipolar_real_solver_brackets_and_roundtrips(tmp_path, capsys) -> None:
    result = dkx.run(_ambipolar_case(), out=tmp_path / "real-ambipolar.nc")

    assert "[dkx.solve]" not in capsys.readouterr().out
    np.testing.assert_array_equal(result.ambipolar_root_count, [1, 1])
    np.testing.assert_array_equal(result.ambipolar_status, ["bracketed_root"] * 2)
    assert np.all(np.isfinite(result.electric_field_kV_m))
    assert np.all(np.isfinite(result.particle_flux_m2_s))
    assert np.all(np.isfinite(result.heat_flux_W_m2))
    assert np.all(result.selected_ambipolar_root == 0)
    scan_scale = np.nanmax(np.abs(result.radial_current_A_m2), axis=1)
    root_residual = np.abs(result.ambipolar_root_current_A_m2[:, 0])
    assert np.all(root_residual < 0.02 * scan_scale)
    loaded = dkx.Result.load(tmp_path / "real-ambipolar.nc")
    np.testing.assert_allclose(loaded.electric_field_kV_m, result.electric_field_kV_m)


def test_result_arrays_and_contract_are_immutable() -> None:
    result = dkx.Result(
        case_id="a" * 64,
        case_name="small",
        workflow="profile",
        arrays={"surface": [0.25], "flux": [[1.0]]},
        dimensions={"surface": ("surface",), "flux": ("surface", "species")},
        metadata={"converged": True},
    )
    with pytest.raises(ValueError):
        result.flux[0, 0] = 2.0
    with pytest.raises(TypeError):
        result.metadata["converged"] = False
    with pytest.raises(FrozenInstanceError):
        result.case_name = "changed"

    nested = dkx.Result(
        case_id="b" * 64,
        case_name="nested",
        workflow="profile",
        arrays={"surface": [0.25]},
        dimensions={"surface": ("surface",)},
        metadata={"timings_s": {"total": 1.0}},
    )
    with pytest.raises(TypeError):
        nested.metadata["timings_s"]["total"] = 2.0


def test_physical_electric_field_normalization_is_explicit() -> None:
    """1 kV/m maps to ErHat=1 only because TBar=1 keV and RBar=1 m."""

    from dkx.execution import _electric_field_kv_m_to_er_hat

    assert _electric_field_kv_m_to_er_hat(1.0) == pytest.approx(1.0)
    assert _electric_field_kv_m_to_er_hat(-3.25) == pytest.approx(-3.25)


def test_native_normalization_matches_the_accepted_kernel_path() -> None:
    """The new boundary changes names/units, not the numerical answer."""

    case = _case()
    native = dkx.run(case)
    r_hat = 0.5585 * np.sqrt(np.asarray(case.geometry.surfaces))
    n_hat = np.asarray(case.species[0].density_m3) / 1.0e20
    t_hat = np.asarray(case.species[0].temperature_keV)
    dn_dr_hat = np.gradient(n_hat, r_hat)[-1]
    dt_dr_hat = np.gradient(t_hat, r_hat)[-1]
    mass_hat = case.species[0].mass_amu * 1.66053906892e-27 / 1.67262192369e-27
    legacy = dkx.run(
        geometryScheme=1,
        inputRadialCoordinate=3,
        rN_wish=0.4,
        Zs=[1.0],
        mHats=[mass_hat],
        nHats=[n_hat[-1]],
        THats=[t_hat[-1]],
        dNHatdrHats=[dn_dr_hat],
        dTHatdrHats=[dt_dr_hat],
        Ntheta=9,
        Nzeta=1,
        Nxi=8,
        NL=4,
        Nx=4,
        collisionOperator=1,
        useDKESExBDrift=True,
        Nxi_for_x_option=1,
        xGridScheme=5,
        solverTolerance=1.0e-8,
    )
    radial = RadialCoordinates(psi_a_hat=0.15596, a_hat=0.5585, r_n=0.4)
    factor = radial.d_dpsi_hat_to_d_dr_hat
    np.testing.assert_allclose(
        native.particle_flux_m2_s[-1],
        np.asarray(legacy.moments["particleFlux_vm_psiHat"]) * factor * PARTICLE_FLUX,
        rtol=2.0e-12,
    )
    np.testing.assert_allclose(
        native.heat_flux_W_m2[-1],
        np.asarray(legacy.moments["heatFlux_vm_psiHat"]) * factor * HEAT_FLUX,
        rtol=2.0e-12,
    )
    np.testing.assert_allclose(
        native.parallel_current_A_T_m2[-1],
        np.asarray(legacy.moments["FSABjHat"]) * PARALLEL_CURRENT,
        rtol=2.0e-12,
    )


def test_native_vmec_reuses_file_and_grids_and_matches_scheme5(
    monkeypatch,
) -> None:
    """A profile reads VMEC and constructs its shape-stable grids exactly once."""

    import dkx.magnetic_geometry as magnetic_geometry
    import dkx.phase_space as phase_space

    case = _vmec_case()
    original_read = magnetic_geometry.read_vmec_wout
    original_make_grids = phase_space.make_grids
    original_to_namelist = dkx.inputs.SfincsInput.to_namelist
    original_parse = dkx.run.parse_sfincs_input_text
    calls = {"read": 0, "grids": 0}

    def counted_read(*args, **kwargs):
        calls["read"] += 1
        return original_read(*args, **kwargs)

    def counted_make_grids(*args, **kwargs):
        calls["grids"] += 1
        return original_make_grids(*args, **kwargs)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("native VMEC execution used a SFINCS namelist adapter")

    monkeypatch.setattr(magnetic_geometry, "read_vmec_wout", counted_read)
    monkeypatch.setattr(phase_space, "make_grids", counted_make_grids)
    monkeypatch.setattr("dkx.inputs.SfincsInput.to_namelist", forbidden)
    monkeypatch.setattr("dkx.run.parse_sfincs_input_text", forbidden)
    native = dkx.run(case)

    assert calls == {"read": 1, "grids": 1}
    assert (
        native.metadata["geometry_sha256"]
        == hashlib.sha256(case.geometry_path.read_bytes()).hexdigest()
    )
    assert native.metadata["normalization"]["a_hat"] > 0.0
    assert native.metadata["normalization"]["psi_a_hat"] != 0.0

    # Restore the readers before exercising the established compatibility path.
    monkeypatch.setattr(magnetic_geometry, "read_vmec_wout", original_read)
    monkeypatch.setattr(phase_space, "make_grids", original_make_grids)
    monkeypatch.setattr(dkx.inputs.SfincsInput, "to_namelist", original_to_namelist)
    monkeypatch.setattr(dkx.run, "parse_sfincs_input_text", original_parse)
    r_n = np.sqrt(np.asarray(case.geometry.surfaces))
    r_hat = native.metadata["normalization"]["a_hat"] * r_n
    n_hat = np.asarray(case.species[0].density_m3) / 1.0e20
    t_hat = np.asarray(case.species[0].temperature_keV)
    mass_hat = case.species[0].mass_amu * 1.66053906892e-27 / 1.67262192369e-27
    legacy = dkx.run(
        geometryScheme=5,
        equilibriumFile=str(case.geometry_path),
        inputRadialCoordinate=3,
        rN_wish=float(r_n[-1]),
        VMECRadialOption=0,
        VMEC_Nyquist_option=1,
        Zs=[1.0],
        mHats=[mass_hat],
        nHats=[n_hat[-1]],
        THats=[t_hat[-1]],
        dNHatdrHats=[np.gradient(n_hat, r_hat)[-1]],
        dTHatdrHats=[np.gradient(t_hat, r_hat)[-1]],
        Ntheta=case.resolution.theta,
        Nzeta=case.resolution.zeta,
        Nxi=case.resolution.pitch,
        NL=min(4, case.resolution.pitch),
        Nx=case.resolution.speed,
        collisionOperator=1,
        useDKESExBDrift=True,
        Nxi_for_x_option=1,
        xGridScheme=5,
        solverTolerance=case.solver.relative_tolerance,
    )
    radial = RadialCoordinates(
        psi_a_hat=native.metadata["normalization"]["psi_a_hat"],
        a_hat=native.metadata["normalization"]["a_hat"],
        r_n=float(r_n[-1]),
    )
    np.testing.assert_allclose(
        native.particle_flux_m2_s[-1],
        np.asarray(legacy.moments["particleFlux_vm_psiHat"])
        * radial.d_dpsi_hat_to_d_dr_hat
        * PARTICLE_FLUX,
        rtol=2.0e-12,
    )
    np.testing.assert_allclose(
        native.heat_flux_W_m2[-1],
        np.asarray(legacy.moments["heatFlux_vm_psiHat"])
        * radial.d_dpsi_hat_to_d_dr_hat
        * HEAT_FLUX,
        rtol=2.0e-12,
    )
    np.testing.assert_allclose(
        native.parallel_current_A_T_m2[-1],
        np.asarray(legacy.moments["FSABjHat"]) * PARALLEL_CURRENT,
        rtol=2.0e-12,
    )


def test_unsupported_native_route_names_the_field_and_correction() -> None:
    case = _case()
    case = replace(case, physics=replace(case.physics, magnetic_drifts="full"))
    with pytest.raises(dkx.CaseValidationError) as excinfo:
        dkx.run(case)
    message = str(excinfo.value)
    assert "physics.magnetic_drifts" in message
    assert "dkes" in message


def test_boozer_native_route_is_explicitly_unsupported() -> None:
    case = _case()
    case = replace(case, geometry=replace(case.geometry, format="boozer"))
    with pytest.raises(dkx.CaseValidationError, match="dedicated reader"):
        dkx.run(case)
