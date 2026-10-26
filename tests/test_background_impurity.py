"""A collision-only impurity scatters the ions exactly as a kinetic one does under pitch-angle scattering.

With ``pitch_angle_scattering`` the operator is diagonal in species: a kinetic
impurity reaches the ions only through the partner sum in ``nu_D``, which is
all a background carries. The ion fluxes must therefore agree to solver
tolerance, with a smaller state.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import dkx
from dkx.config import CaseValidationError
from dkx.execution import run_case

_IMPURITY = {"name": "carbon", "charge": 6, "mass_amu": 12.0,
             "density_m3": [4.0e18, 3.0e18], "temperature_keV": [1.0, 0.8]}


def _case(*, kinetic: bool):
    ion = {"name": "deuterium", "charge": 1, "mass_amu": 2.014,
           "density_m3": [8.0e19, 7.0e19], "temperature_keV": [1.0, 0.8]}
    return dkx.Case.from_mapping({
        "name": "background_impurity",
        "run": {"workflow": "profile", "progress": False},
        "geometry": {"format": "analytic", "file": "tokamak", "surfaces": [0.09, 0.16]},
        "species": [ion, _IMPURITY] if kinetic else [ion],
        **({} if kinetic else {"background": [_IMPURITY]}),
        "physics": {"collisions": "pitch_angle_scattering", "magnetic_drifts": "dkes", "phi1": "off"},
        "electric_field": {"mode": "prescribed", "value_kV_m": 0.5},
        "resolution": {"theta": 9, "zeta": 1, "pitch": 8, "speed": 4},
        "solver": {"method": "auto", "relative_tolerance": 1.0e-10},
        "output": {"file": "out.nc", "plots": False},
    })  # fmt: skip


def test_background_impurity_matches_the_kinetic_one_for_the_ions():
    kinetic = run_case(_case(kinetic=True))
    background = run_case(_case(kinetic=False))
    for name in ("particle_flux_m2_s", "heat_flux_W_m2"):
        assert background.arrays[name].shape[1] == 1  # no impurity unknowns
        np.testing.assert_allclose(background.arrays[name][:, 0], kinetic.arrays[name][:, 0], rtol=1e-8)
    # the impurity does change the ions: without it the flux differs
    bare = run_case(replace(_case(kinetic=False), background=()))
    assert not np.allclose(bare.arrays["particle_flux_m2_s"], background.arrays["particle_flux_m2_s"], rtol=1e-3)


def test_background_needs_pitch_angle_scattering():
    case = _case(kinetic=False)
    with pytest.raises(CaseValidationError, match="pitch_angle_scattering"):
        replace(case, physics=replace(case.physics, collisions="linearized_fokker_planck"))
