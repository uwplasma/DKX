from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from dkx.validation.independent import (
    coefficient_relative_errors,
    dkes_to_beidler,
    nu_prime_for_nu_over_v,
)


_ROOT = Path(__file__).resolve().parents[1]
_ARTIFACT = _ROOT / "validation" / "independent_cross_code_v1.json"


def test_nu_over_v_map_includes_the_applied_deflection_shape() -> None:
    value = nu_prime_for_nu_over_v(
        0.01,
        g_hat=2.3204100000000003,
        i_hat=-0.0033862,
        iota=-0.468945,
        b0_over_bbar=1.5451308075,
        nu_d_hat_x0=0.8360276804879032,
    )
    assert value == pytest.approx(0.017975290661104672, rel=2e-15)


def test_dkes_conversion_uses_local_radius_and_orientation() -> None:
    converted = dkes_to_beidler(
        d11=0.09420723079216414,
        d31=0.5671247552808146,
        d13=-0.5689076170400662,
        d33=60.24173801533913,
        nu_over_v=0.01,
        g_hat=2.3204100000000003,
        iota=-0.468945,
        b0_over_bbar=1.5451308075,
        r_hat=0.1600004874677725,
        raw_b0_over_bbar=1.5451292699958306,
        raw_fsab_b2=0.9930199354487245 * 1.5451292699958306**2,
    )
    assert converted["D11_star"] == pytest.approx(0.40334459550951945)
    assert converted["D31_star"] == pytest.approx(-0.13780489737481735)
    assert converted["D13_star"] == pytest.approx(0.13823811260564514)
    assert converted["D33_star"] == pytest.approx(0.9099777738316578)
    assert converted["eps_t"] == pytest.approx(0.10654224141486761)


def test_conversion_rejects_ambiguous_d33_scale() -> None:
    common = dict(
        d11=1.0,
        d31=1.0,
        d13=-1.0,
        d33=1.0,
        nu_over_v=0.1,
        g_hat=1.0,
        iota=0.5,
        b0_over_bbar=1.0,
        r_hat=0.2,
    )
    with pytest.raises(ValueError, match="exactly one"):
        dkes_to_beidler(**common)
    with pytest.raises(ValueError, match="exactly one"):
        dkes_to_beidler(**common, raw_fsab_b2=1.0, d33_spitzer=1.0)


def test_checked_artifact_recomputes_every_normalization_and_gate() -> None:
    payload = json.loads(_ARTIFACT.read_text(encoding="utf-8"))
    assert payload["schema"] == "dkx.independent_cross_code.v1"
    assert payload["claim_scope"] == "matched_zero_field_mdke_pas_dkes_only"
    assert {case["family"] for case in payload["cases"]} == {
        "axisymmetric_tokamak",
        "ncsx_stellarator",
        "w7x_eim_stellarator",
    }

    for case in payload["cases"]:
        raw = case["reference"]["raw_dkes"]
        norm = case["normalization"]
        d33_kwargs = (
            {"d33_spitzer": raw["D33_spitzer"]}
            if "D33_spitzer" in raw
            else {"raw_fsab_b2": raw["fsab_B2"]}
        )
        converted = dkes_to_beidler(
            d11=raw["D11"],
            d31=raw["D31"],
            d13=raw["D13"],
            d33=raw["D33"],
            nu_over_v=case["equations"]["nu_over_v_per_m"],
            g_hat=norm["g_hat_T_m"],
            iota=norm["iota"],
            b0_over_bbar=norm["B0_T"],
            r_hat=norm["local_r_m"],
            raw_b0_over_bbar=norm["raw_B0_T"],
            cross_orientation=norm["cross_orientation"],
            **d33_kwargs,
        )
        for key, expected in case["reference"]["beidler"].items():
            assert converted[key] == pytest.approx(expected, rel=2e-14)

        errors = coefficient_relative_errors(case["dkx"]["beidler"], converted)
        assert errors == pytest.approx(case["comparison"]["relative_error"], rel=2e-13)
        assert max(errors.values()) <= case["comparison"]["relative_tolerance"]


def test_checked_audit_script_passes_without_external_checkout() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "tools/paper_benchmarks/audit_independent_cross_code_validation.py",
        ],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    assert report["pass"] is True
    assert report["errors"] == []
    assert report["external_inputs_verified"] is False


def test_artifact_does_not_overclaim_full_kinetic_or_ambipolar_validation() -> None:
    text = _ARTIFACT.read_text(encoding="utf-8").lower()
    for phrase in (
        "not_full_fokker_planck",
        "not_ambipolar_profile_validation",
        "not_experimental_validation",
    ):
        assert phrase in text
