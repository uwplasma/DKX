"""``run_case`` solves the surfaces of a profile as one batch and matches the surface loop."""

from __future__ import annotations

import numpy as np

import dkx
from dkx import execution


def _case(surfaces):
    n = len(surfaces)
    return dkx.Case.from_mapping({
        "name": "batched_surfaces",
        "run": {"workflow": "profile", "progress": False},
        "geometry": {"format": "analytic", "file": "tokamak", "surfaces": surfaces},
        "species": [{"name": "deuterium", "charge": 1, "mass_amu": 2.014,
                     "density_m3": list(np.linspace(8e19, 6e19, n)),
                     "temperature_keV": list(np.linspace(1.0, 0.6, n))}],
        "physics": {"collisions": "pitch_angle_scattering", "magnetic_drifts": "dkes", "phi1": "off"},
        "electric_field": {"mode": "prescribed", "value_kV_m": 1.0},
        "resolution": {"theta": 9, "zeta": 1, "pitch": 8, "speed": 4},
        "solver": {"method": "auto", "relative_tolerance": 1.0e-10},
        "output": {"file": "out.nc", "plots": False},
    })  # fmt: skip


def test_surfaces_are_one_batch_and_match_the_loop(monkeypatch, tmp_path):
    case = _case([0.09, 0.16, 0.25, 0.36])
    seen = []
    real = execution._batched_surfaces
    monkeypatch.setattr(execution, "_batched_surfaces",
                        lambda *a: seen.append(real(*a)) or seen[-1])
    batched = execution.run_case(case, out=tmp_path / "b.nc")
    assert seen[0] is not None and np.asarray(seen[0].algebraic_converged).all()
    monkeypatch.setattr(execution, "_batched_surfaces", lambda *a: None)
    looped = execution.run_case(case, out=tmp_path / "l.nc")
    for name in ("particle_flux_m2_s", "heat_flux_W_m2", "parallel_current_A_T_m2"):
        np.testing.assert_allclose(batched.arrays[name], looped.arrays[name], rtol=1e-6, atol=1e-30)
    assert np.ptp(batched.arrays["parallel_current_A_T_m2"]) > 0  # the surfaces really differ
    assert np.all(batched.arrays["primal_residual"] <= 1e-10 * batched.arrays["primal_rhs_norm"])
