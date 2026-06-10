from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from scripts.audit_qs_paper_terms import (
    _replace_namelist_scalar,
    fsabjhat_assembly_audit,
    gradient_conversion_audit,
    max_ok_relative_error,
    relative_metrics,
)


def test_relative_metrics_reports_roundoff_agreement() -> None:
    metrics = relative_metrics(np.asarray([1.0, 2.0]), np.asarray([1.0, 2.0 + 1.0e-14]))

    assert metrics["status"] == "ok"
    assert metrics["max_rel"] < 1.0e-13


def test_replace_namelist_scalar_preserves_comments() -> None:
    text = "  Ntheta = 25   ! angular grid\n  solverTolerance = 1.0e-9\n"

    patched = _replace_namelist_scalar(text, "Ntheta", 13)
    patched = _replace_namelist_scalar(patched, "solverTolerance", 1.0e-6)

    assert "Ntheta = 13   ! angular grid" in patched
    assert "solverTolerance = 1e-06" in patched


def test_max_ok_relative_error_filters_status_and_keys() -> None:
    report = {
        "FSABjHat": {"status": "ok", "max_rel": 0.02},
        "FSABFlow": {"status": "ok", "max_rel": 0.2},
        "missing": {"status": "missing_jax", "max_rel": 1.0},
    }

    assert max_ok_relative_error(report, ("FSABjHat", "missing")) == 0.02
    assert max_ok_relative_error(report) == 0.2


def test_gradient_conversion_audit_matches_fortran_v3_rhat_er_contract(tmp_path: Path) -> None:
    path = tmp_path / "gradients.h5"
    psi_a_hat = 5.0
    a_hat = 2.0
    r_n = 0.5
    ddrhat2ddpsihat = a_hat / (2.0 * psi_a_hat * r_n)
    ddpsihat2ddrhat = 1.0 / ddrhat2ddpsihat
    with h5py.File(path, "w") as h5:
        h5["psiAHat"] = np.asarray(psi_a_hat)
        h5["aHat"] = np.asarray(a_hat)
        h5["rN"] = np.asarray(r_n)
        h5["inputRadialCoordinateForGradients"] = np.asarray(4, dtype=np.int32)
        h5["Er"] = np.asarray(0.25)
        h5["dPhiHatdpsiHat"] = np.asarray(ddrhat2ddpsihat * -0.25)
        h5["dPhiHatdpsiN"] = np.asarray(psi_a_hat * ddrhat2ddpsihat * -0.25)
        h5["dPhiHatdrHat"] = np.asarray(-0.25)
        h5["dPhiHatdrN"] = np.asarray((2.0 * psi_a_hat * r_n) * ddrhat2ddpsihat * -0.25)
        h5["dnHatdrHat"] = np.asarray([-2.0, -3.0])
        h5["dTHatdrHat"] = np.asarray([-4.0, -5.0])
        h5["dnHatdpsiHat"] = ddrhat2ddpsihat * np.asarray([-2.0, -3.0])
        h5["dTHatdpsiHat"] = ddrhat2ddpsihat * np.asarray([-4.0, -5.0])
        h5["dnHatdpsiN"] = psi_a_hat * h5["dnHatdpsiHat"][()]
        h5["dTHatdpsiN"] = psi_a_hat * h5["dTHatdpsiHat"][()]
        h5["dnHatdrN"] = (2.0 * psi_a_hat * r_n) * h5["dnHatdpsiHat"][()]
        h5["dTHatdrN"] = (2.0 * psi_a_hat * r_n) * h5["dTHatdpsiHat"][()]
        assert np.isclose(ddpsihat2ddrhat * h5["dPhiHatdpsiHat"][()], h5["dPhiHatdrHat"][()])

    audit = gradient_conversion_audit(path)

    assert audit["status"] == "ok"
    assert audit["density"]["max_rel"] == 0.0
    assert audit["temperature"]["max_rel"] == 0.0
    assert audit["potential"]["max_rel"] == 0.0


def test_fsabjhat_assembly_audit_checks_species_dot_and_normalizations(tmp_path: Path) -> None:
    path = tmp_path / "current.h5"
    z_s = np.asarray([1.0, -1.0])
    flow = np.asarray([[3.0], [1.25]])
    current = np.einsum("s,sn->n", z_s, flow)
    with h5py.File(path, "w") as h5:
        h5["Zs"] = z_s
        h5["FSABFlow"] = flow
        h5["FSABFlow_vs_x"] = np.asarray([0.25 * flow, 0.75 * flow])
        h5["FSABjHat"] = current
        h5["B0OverBBar"] = np.asarray(2.0)
        h5["FSABHat2"] = np.asarray(4.0)
        h5["FSABjHatOverB0"] = current / 2.0
        h5["FSABjHatOverRootFSAB2"] = current / 2.0

    audit = fsabjhat_assembly_audit(path)

    assert audit["status"] == "ok"
    assert audit["FSABjHat_from_Zs_dot_FSABFlow"]["max_rel"] == 0.0
    assert audit["FSABjHatOverB0"]["max_rel"] == 0.0
    assert audit["FSABjHatOverRootFSAB2"]["max_rel"] == 0.0
    assert audit["FSABFlow_vs_x_sum"]["max_rel"] == 0.0
