"""The first native Case -> solve -> Result route never passes through a deck."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

import dkx
from dkx.constants import RadialCoordinates
from dkx.units import HEAT_FLUX, PARALLEL_CURRENT, PARTICLE_FLUX


def _case():
    return dkx.Case.from_mapping({
        "schema": 1,
        "name": "native_tokamak_profile",
        "run": {"workflow": "profile", "progress": False},
        "geometry": {
            "format": "analytic", "file": "tokamak", "surfaces": [0.09, 0.16],
        },
        "species": [{
            "name": "deuterium", "charge": 1, "mass_amu": 2.014,
            "density_m3": [8.0e19, 7.0e19], "temperature_keV": [1.0, 0.8],
        }],
        "physics": {
            "collisions": "pitch_angle_scattering",
            "magnetic_drifts": "dkes", "phi1": "off",
        },
        "electric_field": {"mode": "prescribed", "value_kV_m": 0.0},
        "resolution": {"theta": 9, "zeta": 1, "pitch": 8, "speed": 4},
        "solver": {"method": "auto", "relative_tolerance": 1.0e-8},
        "output": {"file": "native-result.nc", "plots": False},
    })


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


def test_result_arrays_and_contract_are_immutable() -> None:
    result = dkx.Result(
        case_id="a" * 64, case_name="small", workflow="profile",
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
        case_id="b" * 64, case_name="nested", workflow="profile",
        arrays={"surface": [0.25]}, dimensions={"surface": ("surface",)},
        metadata={"timings_s": {"total": 1.0}},
    )
    with pytest.raises(TypeError):
        nested.metadata["timings_s"]["total"] = 2.0


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
        geometryScheme=1, inputRadialCoordinate=3, rN_wish=0.4,
        Zs=[1.0], mHats=[mass_hat], nHats=[n_hat[-1]], THats=[t_hat[-1]],
        dNHatdrHats=[dn_dr_hat], dTHatdrHats=[dt_dr_hat],
        Ntheta=9, Nzeta=1, Nxi=8, NL=4, Nx=4,
        collisionOperator=1, useDKESExBDrift=True,
        Nxi_for_x_option=1, xGridScheme=5, solverTolerance=1.0e-8,
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


def test_unsupported_native_route_names_the_field_and_correction() -> None:
    case = _case()
    case = replace(case, physics=replace(case.physics, magnetic_drifts="full"))
    with pytest.raises(dkx.CaseValidationError) as excinfo:
        dkx.run(case)
    message = str(excinfo.value)
    assert "physics.magnetic_drifts" in message
    assert "dkes" in message
