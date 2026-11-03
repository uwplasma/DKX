"""An ``E_r`` scan runs as one compiled batch per surface and matches the point-by-point scan."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

import dkx
from dkx.workflows import scan as sc

_VALUES = (-2.0, 0.0, 1.5)


def _case():
    return dkx.Case.from_mapping({
        "name": "batched_er_scan",
        "run": {"workflow": "profile", "progress": False},
        "geometry": {"format": "analytic", "file": "tokamak", "surfaces": [0.09, 0.16]},
        "species": [{"name": "deuterium", "charge": 1, "mass_amu": 2.014,
                     "density_m3": [8.0e19, 7.0e19], "temperature_keV": [1.0, 0.8]}],
        "physics": {"collisions": "pitch_angle_scattering", "magnetic_drifts": "dkes", "phi1": "off"},
        "electric_field": {"mode": "prescribed", "value_kV_m": 0.0},
        "resolution": {"theta": 9, "zeta": 1, "pitch": 8, "speed": 4},
        "solver": {"method": "auto", "relative_tolerance": 1.0e-10},
        "output": {"file": "out.nc", "plots": False},
        "scan": {"axis": [{"path": "electric_field.value_kV_m", "values": list(_VALUES)}]},
    })  # fmt: skip


def test_batched_er_scan_matches_the_point_by_point_scan(tmp_path, monkeypatch):
    case = _case()
    batched, failures = sc.run_scan(case, out=tmp_path / "batched.nc", resume=False)
    assert failures == 0

    def forbidden(*args, **kwargs):
        raise AssertionError("the batched scan must not run points one by one")

    import dkx.execution as execution

    original = execution.run_case
    monkeypatch.setattr(execution, "run_case", forbidden)
    sc.run_scan(case, out=tmp_path / "again.nc", resume=False)  # still batched
    monkeypatch.setattr(execution, "run_case", original)
    monkeypatch.setattr(sc, "_batched_er_rows", lambda *a, **k: {})  # force the loop
    looped, _ = sc.run_scan(case, out=tmp_path / "looped.nc", resume=False)
    for name in sc.SCAN_OBSERVABLES[:3]:
        np.testing.assert_allclose(batched.arrays[name], looped.arrays[name], rtol=1e-6)
    assert np.ptp(batched.arrays["particle_flux_m2_s"]) > 0  # E_r actually varies the answer


def test_other_scans_fall_back_to_the_loop():
    case = _case()
    other = replace(case, scan=replace(case.scan, axes=(replace(case.scan.axes[0], path="resolution.theta", values=(9, 11)),)))
    assert sc._batched_er_rows(other, other.scan, [(0, 9), (1, 11)]) == {}
