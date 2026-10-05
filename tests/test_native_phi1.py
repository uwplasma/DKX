"""Native ``physics.phi1 = "kinetic"`` route: coupled kinetic + QN + gauge solve."""

from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np
import pytest

from dkx.config import Case
from dkx.execution import run_case

_CASE = Path(__file__).resolve().parents[1] / "examples" / "01_tokamak_profile" / "case.toml"


def _case(tmp_path, phi1, **physics):
    data = tomllib.loads(_CASE.read_text())
    data["geometry"]["surfaces"] = [0.16, 0.25]
    ion = data["species"][0]
    ion["density_m3"], ion["temperature_keV"] = [8.0e19, 6.0e19], [1.0, 0.7]
    data["species"].append(
        {"name": "electron", "charge": -1, "mass_amu": 5.48579909e-4,
         "density_m3": [8.0e19, 6.0e19], "temperature_keV": [1.0, 0.7]}
    )
    data["physics"].update(phi1=phi1, **physics)
    data["electric_field"]["value_kV_m"] = 2.0
    data["output"].update(file=f"{phi1}.nc", plots=False)
    return Case.from_mapping(data, source_path=tmp_path / "case.toml")


def test_native_phi1_kinetic_solves_the_coupled_residual(tmp_path):
    off = run_case(_case(tmp_path, "off"))
    on = run_case(_case(tmp_path, "kinetic"))
    # Accepted by the original nonlinear residual F(x) = A(x) - b(x).
    assert np.all(on.residuals <= 1.0e-8 * on.rhs_norms)
    flux_off, flux_on = np.asarray(off.particle_flux), np.asarray(on.particle_flux)
    assert np.all(np.isfinite(flux_on))
    # Phi1 changes the solution, but only as a correction on this weak-E_r case.
    assert not np.allclose(flux_on, flux_off, rtol=1e-12, atol=0.0)
    assert np.allclose(flux_on, flux_off, rtol=0.5, atol=0.0)


@pytest.mark.parametrize(("phi1", "workflow"), [("full", "profile"), ("kinetic", "ambipolar_profile")])
def test_native_phi1_refuses_unsupported_routes(tmp_path, phi1, workflow):
    case = _case(tmp_path, phi1)
    object.__setattr__(case.run, "workflow", workflow)
    with pytest.raises(Exception, match="physics.phi1"):
        run_case(case)
