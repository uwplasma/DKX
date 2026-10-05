"""Bracketed J_r roots are refined, not left at a 10-20 kV/m linear interpolation."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def test_refine_root_converges_on_a_curved_fake_jr():
    from dkx.representative import ER_ROOT_MAX_SOLVES, _refine_root

    jr = lambda e: np.tanh((e + 33.0) / 10.0) + 0.02 * (e + 33.0)  # noqa: E731
    er = np.array([-60.0, -40.0, -20.0, 0.0])
    lin = -40.0 + 20.0 * jr(-40.0) / (jr(-40.0) - jr(-20.0))
    assert abs(lin + 33.0) > 0.5  # the linear bracket root is off by ~1 kV/m
    root, evals = _refine_root(lambda e: (float(jr(e)), None), er, jr(er), lin)
    assert root == pytest.approx(-33.0, abs=1e-3) and 0 < len(evals) <= ER_ROOT_MAX_SOLVES


def test_radial_profiles_read_moments_at_the_refined_root(monkeypatch):
    import dkx.api
    import dkx.representative as rep

    def fake(deck, er):
        er = np.asarray(er, float)
        return SimpleNamespace(radial_current=np.tanh((er + 7.3) / 4.0),
                               moments={"FSABjHatOverRootFSAB2": er, "FSABjHat": er})

    monkeypatch.setattr(dkx.api, "batched_er_scan", fake)
    monkeypatch.setattr(rep, "resolve_plasma", lambda eq: (dict(rep.FALLBACK_PLASMA), "x"))
    monkeypatch.setattr(rep, "equilibrium_scalars", lambda eq: {"a_hat": 1.0, "psi_a_hat": 0.1})
    monkeypatch.setattr(rep, "plasma_parameters", lambda eq, radius=0.5: dict(rep.FALLBACK_PLASMA))
    (p,) = rep.radial_profiles(Path("w.nc"), surfaces=(0.5,), er_values=(-12.0, -4.0, 4.0), emit=None)
    assert p["er_ambipolar"] == pytest.approx(-7.3, abs=1e-2) == p["roots"][0]
    assert p["bootstrap"] == pytest.approx(p["er_ambipolar"])  # moment evaluated at the root
    assert 0 < p["er_root_refine_solves"] <= rep.ER_ROOT_MAX_SOLVES and len(p["er_scan"]) > 3
