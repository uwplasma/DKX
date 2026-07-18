from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import dkx.workflows.geometry_adapters as jga
from dkx.workflows.geometry_adapters import (
    boozer_bhat_from_spectrum,
    boozer_spectrum_geometry_proxy_objective,
    boozer_spectrum_proxy_transport_objective,
    geometry_proxy_no_solve_provenance_gate,
    geometry_proxy_workflow_contract,
    geometry_proxy_workflow_summary,
    kinetic_transport_scalar_no_overclaim_gate,
    optional_jax_geometry_backend_report,
    optional_jax_geometry_backend_status,
    vmec_boozer_kinetic_transport_scalar_contract,
    vmec_wout_from_wout_like,
)
from dkx.magnetic_geometry import FluxSurfaceGeometry, read_vmec_wout


def _wout_like(*, radius_mode_order: bool = True) -> SimpleNamespace:
    ns = 4
    mnmax = 3
    mnmax_nyq = 5

    def arr(modes: int, radius: int = ns) -> np.ndarray:
        data = np.arange(modes * radius, dtype=np.float64).reshape((modes, radius))
        return data.T if radius_mode_order else data

    return SimpleNamespace(
        path="vmex_in_memory",
        ns=ns,
        mpol=2,
        ntor=1,
        nfp=5,
        lasym=False,
        mnmax=mnmax,
        mnmax_nyq=mnmax_nyq,
        Aminor_p=0.3,
        phi=np.linspace(0.0, 1.0, ns),
        xm=np.arange(mnmax),
        xn=np.arange(mnmax) * 5,
        xm_nyq=np.arange(mnmax_nyq),
        xn_nyq=np.arange(mnmax_nyq) * 5,
        bmnc=arr(mnmax_nyq),
        gmnc=arr(mnmax_nyq) + 10.0,
        bsubumnc=arr(mnmax_nyq) + 20.0,
        bsubvmnc=arr(mnmax_nyq) + 30.0,
        bsubsmns=arr(mnmax_nyq) + 40.0,
        bsupumnc=arr(mnmax_nyq) + 50.0,
        bsupvmnc=arr(mnmax_nyq) + 60.0,
        rmnc=arr(mnmax),
        zmns=arr(mnmax) + 70.0,
        lmns=arr(mnmax, ns - 1),
        iotas=np.linspace(0.0, 0.8, ns),
        presf=np.linspace(1.0, 0.0, ns),
    )


def test_optional_jax_geometry_backend_status_is_structural() -> None:
    status = optional_jax_geometry_backend_status()
    assert set(status) == {"vmex", "booz_xform_jax"}
    assert all(isinstance(value, bool) for value in status.values())


def test_optional_jax_geometry_backend_status_uses_shallow_find_spec(monkeypatch) -> None:
    calls: list[str] = []

    def fake_find_spec(name: str):
        calls.append(name)
        return object() if name == "vmex" else None

    monkeypatch.setattr(jga, "find_spec", fake_find_spec)

    assert optional_jax_geometry_backend_status() == {
        "vmex": True,
        "booz_xform_jax": False,
    }
    assert calls == ["vmex", "booz_xform_jax"]


def test_optional_jax_geometry_backend_report_marks_gradient_boundary() -> None:
    report = optional_jax_geometry_backend_report()

    assert report["backends"] == optional_jax_geometry_backend_status()
    assert report["workflow_contract"]["workflow"] == "vmex_to_boozer_sfincs_geometry_proxy"
    assert report["gradient_availability"]["spectral_scale_to_boozer_proxy"] == (
        "available_when_optional_backends_installed"
    )
    assert (
        report["gradient_availability"]["sfincs_kinetic_transport_solve"]
        == "not_covered_by_this_lane"
    )
    assert "booz_xform_jax" in report["differentiated_graph"]
    assert "SFINCS kinetic transport solve" in report["outside_differentiated_graph"]
    assert report["no_overclaim_gate"]["full_transport_gradients_claimed"] is False
    assert report["kinetic_transport_scalar_contract"]["no_overclaim_gate"]["status"] == "pass"
    assert report["claim"] == "geometry_proxy_gradient_gate_not_full_transport_gradient"


def test_geometry_proxy_workflow_contract_records_ci_and_differentiability_policy() -> None:
    contract = geometry_proxy_workflow_contract(
        backend_status={"vmex": False, "booz_xform_jax": True}
    )

    assert contract["contract_version"] >= 1
    assert contract["optional_backends"] == {"vmex": False, "booz_xform_jax": True}
    assert contract["ci_dependency_policy"]["default_ci_requires_vmex"] is False
    assert contract["ci_dependency_policy"]["default_ci_requires_booz_xform_jax"] is False
    assert contract["ci_dependency_policy"]["backend_check_imports_optional_packages"] is False
    assert contract["differentiability_labels"]["differentiated"].startswith("covered by JAX")
    assert "SFINCS kinetic transport solve" in contract["outside_differentiated_graph"]
    gate = contract["no_overclaim_gate"]
    assert gate["status"] == "pass"
    assert gate["claim_scope"] == "geometry_proxy_gradient_only"
    assert gate["full_transport_gradients_claimed"] is False
    assert gate["forbidden_gradient_claim"] == "full VMEC-boundary-to-SFINCS kinetic transport gradients"
    assert gate["kinetic_gradient_status"] == "deferred_not_covered_by_this_lane"
    assert gate["kinetic_transport_scalar_contract_gate"]["status"] == "pass"


def test_vmec_boozer_kinetic_transport_scalar_contract_is_machine_readable() -> None:
    contract = vmec_boozer_kinetic_transport_scalar_contract(
        backend_status={"vmex": False, "booz_xform_jax": False}
    )

    assert contract["scalar_target"] == "future_vmec_boozer_to_sfincs_kinetic_transport_scalar"
    assert contract["ci_dependency_policy"]["default_ci_requires_vmex"] is False
    assert contract["ci_dependency_policy"]["default_ci_requires_booz_xform_jax"] is False
    assert contract["current_public_scalar"]["kinetic_transport_scalar_claimed"] is False
    assert contract["current_public_scalar"]["kinetic_solve_executed"] is False
    assert contract["current_public_scalar"]["scalar_kind"] == "boozer_spectrum_proxy_not_kinetic"
    assert contract["no_overclaim_gate"]["status"] == "pass"

    stage_names = [stage["name"] for stage in contract["required_stages"]]
    allowed_boundaries = {
        "differentiated",
        "differentiated_for_geometry_proxy_gate",
        "setup_only_not_differentiated",
        "not_claimed_not_covered_by_this_lane",
    }
    assert stage_names == [
        "vmec_source",
        "vmec_equilibrium_or_wout",
        "boozer_transform",
        "sfincs_geometry_adapter",
        "kinetic_operator_assembly",
        "linear_kinetic_solve",
        "transport_scalar_reduction",
        "gradient_validation",
    ]
    for stage in contract["required_stages"]:
        assert stage["required_for_future_kinetic_scalar"] is True
        assert stage["required_evidence"]
        assert stage["differentiability_boundary"] in allowed_boundaries


def test_kinetic_transport_scalar_no_overclaim_gate_rejects_false_promotion() -> None:
    contract = vmec_boozer_kinetic_transport_scalar_contract(
        backend_status={"vmex": True, "booz_xform_jax": True}
    )
    contract["current_public_scalar"]["kinetic_transport_scalar_claimed"] = True
    contract["current_public_scalar"]["kinetic_solve_executed"] = True
    contract["promotion_requirements"]["full_kinetic_scalar_promoted"] = True

    gate = kinetic_transport_scalar_no_overclaim_gate(contract)

    assert gate["status"] == "fail"
    assert "kinetic_transport_scalar_claimed" in gate["forbidden_claims"]
    assert "kinetic_solve_executed" in gate["forbidden_claims"]
    assert "kinetic_operator_assembly" in gate["promotion_violations"]


def test_geometry_proxy_workflow_summary_records_stage_claims_and_gate() -> None:
    summary = geometry_proxy_workflow_summary(
        provenance="unit-test wout",
        requested_surface=0.5,
        selected_surface=0.49,
        boozer_resolution={"mboz": 3, "nboz": 3},
        grid_shape={"n_theta": 8, "n_zeta": 6},
        scale=1.0,
        proxy_objective=0.125,
        autodiff_gradient=2.0,
        finite_difference_gradient=2.000001,
        finite_difference_step=1.0e-4,
        backend_status={"vmex": True, "booz_xform_jax": False},
    )

    assert summary["workflow"] == "vmex_to_boozer_sfincs_geometry_proxy"
    assert summary["workflow_contract"]["ci_dependency_policy"]["default_ci_requires_vmex"] is False
    assert summary["kinetic_transport_scalar_contract"]["no_overclaim_gate"]["status"] == "pass"
    assert summary["provenance"]["source"] == "unit-test wout"
    assert summary["required_optional_dependencies"]["vmex"]["importable"] is True
    assert summary["required_optional_dependencies"]["booz_xform_jax"]["importable"] is False
    assert summary["numerical_gradient_gate"]["status"] == "pass"
    assert summary["results"]["proxy_objective"] == pytest.approx(0.125)
    kinetic_stage = next(
        stage for stage in summary["stages"] if stage["name"] == "sfincs_kinetic_transport_solve"
    )
    assert kinetic_stage["differentiability"] == "not_claimed_not_covered_by_this_lane"
    assert summary["claims"]["not_claimed"] == (
        "full VMEC-boundary-to-SFINCS kinetic transport gradients"
    )
    assert summary["no_overclaim_gate"]["full_transport_gradients_claimed"] is False
    assert summary["no_overclaim_gate"]["kinetic_transport_scalar_contract_gate"]["status"] == "pass"


def test_geometry_proxy_no_solve_gate_validates_file_provenance_and_scalar_contract() -> None:
    summary = geometry_proxy_workflow_summary(
        provenance="unit-test wout",
        requested_surface=0.5,
        selected_surface=0.49,
        boozer_resolution={"mboz": 3, "nboz": 3},
        grid_shape={"n_theta": 8, "n_zeta": 6},
        scale=1.0,
        proxy_objective=0.125,
        autodiff_gradient=2.0,
        finite_difference_gradient=2.000001,
        finite_difference_step=1.0e-4,
        backend_status={"vmex": True, "booz_xform_jax": True},
    )

    gate = geometry_proxy_no_solve_provenance_gate(
        summary,
        require_file_provenance=True,
    )

    assert gate["status"] == "pass"
    assert gate["kinetic_solve_executed"] is False
    assert gate["kinetic_transport_scalar_claimed"] is False
    assert gate["current_public_scalar_kind"] == "boozer_spectrum_proxy_not_kinetic"
    assert gate["missing_file_provenance_fields"] == []
    assert set(gate["present_file_provenance_fields"]) == {
        "source",
        "selected_surface",
        "boozer_resolution",
        "grid_shape",
        "scale",
    }
    assert "linear_kinetic_solve" in gate["required_kinetic_transport_scalar_stages"]
    assert "kinetic_operator_assembly" in gate["differentiability_boundary"][
        "not_covered_stage_names"
    ]

    missing_provenance = geometry_proxy_no_solve_provenance_gate(
        geometry_proxy_workflow_summary(),
        require_file_provenance=True,
    )
    assert missing_provenance["status"] == "fail"
    assert missing_provenance["missing_file_provenance_fields"] == [
        "source",
        "selected_surface",
        "boozer_resolution",
        "grid_shape",
        "scale",
    ]

    tampered = json.loads(json.dumps(summary))
    tampered["kinetic_transport_scalar_contract"]["current_public_scalar"][
        "kinetic_transport_scalar_claimed"
    ] = True
    tampered_gate = geometry_proxy_no_solve_provenance_gate(
        tampered,
        require_file_provenance=True,
    )
    assert tampered_gate["status"] == "fail"
    assert tampered_gate["kinetic_transport_scalar_claimed"] is True


def test_geometry_proxy_workflow_summary_marks_failed_gradient_gate() -> None:
    summary = geometry_proxy_workflow_summary(
        autodiff_gradient=1.0,
        finite_difference_gradient=2.0,
        finite_difference_step=1.0e-4,
        gradient_rtol=1.0e-6,
        gradient_atol=1.0e-9,
    )

    assert summary["numerical_gradient_gate"]["status"] == "fail"
    assert summary["numerical_gradient_gate"]["absolute_error"] == pytest.approx(1.0)


def test_geometry_proxy_workflow_summary_labels_all_stage_claims() -> None:
    summary = geometry_proxy_workflow_summary(
        autodiff_gradient=0.25,
        finite_difference_gradient=0.25,
        backend_status={"vmex": False, "booz_xform_jax": False},
    )
    labels = set(summary["differentiability_labels"])

    assert {stage["differentiability"] for stage in summary["stages"]} <= labels
    assert summary["numerical_gradient_gate"]["claim"] == "geometry_proxy_gradient_gate_only"
    assert "transport" not in summary["claims"]["differentiable"].lower()
    assert summary["no_overclaim_gate"]["forbidden_gradient_claim"] == (
        "full VMEC-boundary-to-SFINCS kinetic transport gradients"
    )


def test_vmec_wout_from_wout_like_transposes_vmex_radius_mode_arrays() -> None:
    converted = vmec_wout_from_wout_like(_wout_like(radius_mode_order=True))
    assert converted.nfp == 5
    assert converted.ns == 4
    assert converted.aminor_p == pytest.approx(0.3)
    assert converted.bmnc.shape == (5, 4)
    assert converted.rmnc.shape == (3, 4)
    assert converted.lmns.shape == (3, 3)
    assert converted.bmnc[2, 3] == pytest.approx(11.0)
    assert converted.bsupvmnc[4, 2] == pytest.approx(78.0)


def test_vmec_wout_from_wout_like_accepts_sfincs_mode_radius_arrays() -> None:
    converted = vmec_wout_from_wout_like(_wout_like(radius_mode_order=False))
    assert converted.gmnc.shape == (5, 4)
    assert converted.zmns.shape == (3, 4)
    assert converted.gmnc[1, 2] == pytest.approx(16.0)


def test_vmec_wout_from_wout_like_path_override_is_metadata_only() -> None:
    converted = vmec_wout_from_wout_like(_wout_like(radius_mode_order=True), path="custom/in_memory_wout.nc")
    assert converted.path == Path("custom/in_memory_wout.nc")
    assert converted.bmnc.shape == (5, 4)


def test_vmec_wout_from_wout_like_zero_fills_optional_field_tables() -> None:
    minimal = _wout_like(radius_mode_order=False)
    del minimal.bsubumnc
    del minimal.bsubvmnc
    del minimal.bsubsmns
    del minimal.bsupumnc
    del minimal.bsupvmnc
    del minimal.presf

    converted = vmec_wout_from_wout_like(minimal)

    np.testing.assert_allclose(converted.bsubumnc, np.zeros_like(converted.gmnc))
    np.testing.assert_allclose(converted.bsubvmnc, np.zeros_like(converted.gmnc))
    np.testing.assert_allclose(converted.bsubsmns, np.zeros_like(converted.gmnc))
    np.testing.assert_allclose(converted.bsupumnc, np.zeros_like(converted.gmnc))
    np.testing.assert_allclose(converted.bsupvmnc, np.zeros_like(converted.gmnc))
    np.testing.assert_allclose(converted.presf, np.zeros((converted.ns,), dtype=np.float64))


def test_vmec_wout_from_wout_like_rejects_missing_required_tables() -> None:
    bad = _wout_like(radius_mode_order=False)
    del bad.bmnc
    with pytest.raises(AttributeError, match="bmnc"):
        vmec_wout_from_wout_like(bad)


def test_vmec_wout_from_wout_like_rejects_bad_shapes() -> None:
    bad = _wout_like(radius_mode_order=True)
    bad.bmnc = np.zeros((2, 2, 2))
    with pytest.raises(ValueError, match="bmnc must be 2D"):
        vmec_wout_from_wout_like(bad)


def test_boozer_bhat_from_spectrum_matches_manual_cosine_sum() -> None:
    theta = jnp.asarray([0.0, 0.5 * jnp.pi])
    zeta = jnp.asarray([0.0, 0.1, 0.2])
    bmnc_b = jnp.asarray([2.0, 0.4, 0.2])
    ixm_b = jnp.asarray([0, 1, 0])
    ixn_b = jnp.asarray([0, 0, 5])

    bhat = boozer_bhat_from_spectrum(
        theta,
        zeta,
        bmnc_b=bmnc_b,
        ixm_b=ixm_b,
        ixn_b=ixn_b,
    )

    expected = (
        2.0
        + 0.4 * np.cos(np.asarray(theta))[:, None]
        + 0.2 * np.cos(-5.0 * np.asarray(zeta))[None, :]
    ) / 2.0
    np.testing.assert_allclose(np.asarray(bhat), expected, rtol=1.0e-14, atol=1.0e-14)


def test_boozer_spectrum_proxy_gradient_matches_centered_difference() -> None:
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, 9, endpoint=False)
    zeta = jnp.linspace(0.0, 2.0 * jnp.pi / 5.0, 7, endpoint=False)
    bmnc_b = jnp.asarray([2.0, 0.2, -0.1, 0.05])
    direction = jnp.asarray([0.0, 0.7, -0.3, 0.2])
    ixm_b = jnp.asarray([0, 1, 1, 2])
    ixn_b = jnp.asarray([0, 0, 5, -5])

    def objective(alpha: jnp.ndarray) -> jnp.ndarray:
        return boozer_spectrum_geometry_proxy_objective(
            bmnc_b + alpha * direction,
            ixm_b,
            ixn_b,
            theta=theta,
            zeta=zeta,
        )

    step = 1.0e-5
    autodiff = float(jax.grad(objective)(0.0))
    finite_difference = float((objective(step) - objective(-step)) / (2.0 * step))

    assert autodiff == pytest.approx(finite_difference, rel=5.0e-6, abs=1.0e-9)


def test_boozer_proxy_transport_normalized_invariants_are_no_solve_gates() -> None:
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, 11, endpoint=False)
    zeta = jnp.linspace(0.0, 2.0 * jnp.pi / 5.0, 9, endpoint=False)
    ixm_b = jnp.asarray([0, 1, 1, 2])
    ixn_b = jnp.asarray([0, 0, 5, -5])
    bmnc_b = jnp.asarray([2.0, 0.25, -0.12, 0.07])

    def objective(coeff: jnp.ndarray) -> jnp.ndarray:
        return boozer_spectrum_proxy_transport_objective(
            coeff,
            ixm_b,
            ixn_b,
            theta=theta,
            zeta=zeta,
        )

    base = objective(bmnc_b)
    scaled = objective(3.5 * bmnc_b)
    scale_gradient = jax.grad(lambda scale: objective(scale * bmnc_b))(3.5)

    np.testing.assert_allclose(np.asarray(scaled), np.asarray(base), rtol=1.0e-6, atol=1.0e-9)
    assert float(scale_gradient) == pytest.approx(0.0, abs=1.0e-8)

    constant_spectrum = jnp.asarray([2.0, 0.0, 0.0, 0.0])
    constant_value, constant_gradient = jax.value_and_grad(objective)(constant_spectrum)

    assert float(constant_value) == pytest.approx(0.0, abs=1.0e-12)
    np.testing.assert_allclose(np.asarray(constant_gradient), np.zeros(4), rtol=0.0, atol=1.0e-10)


def test_public_vmex_boozer_example_backend_check_is_runnable() -> None:
    script = (
        Path(__file__).parents[1]
        / "examples"
        / "autodiff"
        / "vmex_to_boozer_sfincs_pipeline.py"
    )
    result = subprocess.run(
        [sys.executable, str(script), "--check-backends"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Optional JAX geometry backend status:" in result.stdout
    assert "vmex:" in result.stdout
    assert "booz_xform_jax:" in result.stdout
    assert "file-backed/setup only:" in result.stdout
    assert "not claimed: full VMEC-boundary-to-SFINCS-transport gradients" in result.stdout
    assert "Public workflow contract:" in result.stdout
    assert "default CI requires vmex: false" in result.stdout
    assert "default CI requires booz_xform_jax: false" in result.stdout
    assert "no-overclaim gate: pass" in result.stdout
    assert "kinetic scalar contract gate: pass" in result.stdout
    assert "no-solve provenance gate: pass" in result.stdout
    assert "numerical gradient gate: not_run" in result.stdout
    assert "pass --json with --check-backends" in result.stdout
    assert "pass --summary-json PATH" in result.stdout


def test_public_vmex_boozer_example_backend_check_json_is_runnable() -> None:
    script = (
        Path(__file__).parents[1]
        / "examples"
        / "autodiff"
        / "vmex_to_boozer_sfincs_pipeline.py"
    )
    result = subprocess.run(
        [sys.executable, str(script), "--check-backends", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(result.stdout)
    assert set(report["backends"]) == {"vmex", "booz_xform_jax"}
    assert report["workflow_contract"]["ci_dependency_policy"]["default_ci_requires_vmex"] is False
    assert report["gradient_availability"]["vmec_file_io"] == "setup_only_not_differentiated"
    assert report["gradient_availability"]["sfincs_kinetic_transport_solve"] == "not_covered_by_this_lane"
    assert report["no_overclaim_gate"]["full_transport_gradients_claimed"] is False
    assert report["no_solve_provenance_gate"]["status"] == "pass"
    assert report["no_solve_provenance_gate"]["kinetic_solve_executed"] is False
    assert report["no_solve_provenance_gate"]["requires_file_provenance"] is False
    assert report["kinetic_transport_scalar_contract"]["no_overclaim_gate"]["status"] == "pass"
    assert report["no_solve_provenance_gate"]["kinetic_transport_scalar_contract_gate"]["status"] == "pass"
    assert "linear_kinetic_solve" in report["no_solve_provenance_gate"][
        "required_kinetic_transport_scalar_stages"
    ]


def test_public_vmex_boozer_example_backend_check_writes_summary_json(tmp_path: Path) -> None:
    script = (
        Path(__file__).parents[1]
        / "examples"
        / "autodiff"
        / "vmex_to_boozer_sfincs_pipeline.py"
    )
    summary_path = tmp_path / "workflow-summary.json"
    result = subprocess.run(
        [sys.executable, str(script), "--check-backends", "--summary-json", str(summary_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "pass --summary-json PATH" in result.stdout
    assert summary["workflow"] == "vmex_to_boozer_sfincs_geometry_proxy"
    assert summary["workflow_contract"]["contract_version"] >= 1
    assert summary["numerical_gradient_gate"]["status"] == "not_run"
    assert summary["no_solve_provenance_gate"]["status"] == "pass"
    assert summary["no_solve_provenance_gate"]["kinetic_solve_executed"] is False
    assert summary["no_solve_provenance_gate"]["requires_file_provenance"] is False
    assert summary["kinetic_transport_scalar_contract"]["no_overclaim_gate"]["status"] == "pass"
    assert summary["no_solve_provenance_gate"]["kinetic_transport_scalar_contract_gate"]["status"] == "pass"
    assert summary["claims"]["not_claimed"] == (
        "full VMEC-boundary-to-SFINCS kinetic transport gradients"
    )
    assert summary["no_overclaim_gate"]["kinetic_gradient_status"] == "deferred_not_covered_by_this_lane"


def _load_finite_beta_example_module() -> ModuleType:
    script = (
        Path(__file__).parents[1]
        / "examples"
        / "vmex_finite_beta"
        / "finite_beta_vmec_to_sfincs.py"
    )
    spec = importlib.util.spec_from_file_location("finite_beta_vmec_to_sfincs_contract_test", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_finite_beta_summary_records_radial_profile_provenance(tmp_path: Path) -> None:
    mod = _load_finite_beta_example_module()
    records = [
        mod.RunRecord(
            r_n=0.5,
            er=-1.0,
            radial_current=-0.2,
            bootstrap_current=-0.03,
            ion_particle_flux_rhat=0.0,
            electron_particle_flux_rhat=0.0,
            ion_heat_flux_rhat=0.0,
            electron_heat_flux_rhat=0.0,
            output_h5=str(tmp_path / "minus.h5"),
        ),
        mod.RunRecord(
            r_n=0.5,
            er=1.0,
            radial_current=0.2,
            bootstrap_current=-0.031,
            ion_particle_flux_rhat=0.0,
            electron_particle_flux_rhat=0.0,
            ion_heat_flux_rhat=0.0,
            electron_heat_flux_rhat=0.0,
            output_h5=str(tmp_path / "plus.h5"),
        ),
    ]
    profile = [
        mod.SurfaceProfileRecord(
            r_n=0.25,
            psi_n=0.0625,
            roots_er=[-0.5],
            bootstrap_current_at_roots=[-0.02],
            selected_ambipolar_er=-0.5,
            selected_bootstrap_current=-0.02,
            scan_dir=str(tmp_path / "rN0p25"),
        ),
        mod.SurfaceProfileRecord(
            r_n=0.5,
            psi_n=0.25,
            roots_er=[0.1],
            bootstrap_current_at_roots=[-0.03],
            selected_ambipolar_er=0.1,
            selected_bootstrap_current=-0.03,
            scan_dir=str(tmp_path / "rN0p5"),
        ),
    ]
    convergence_profile = [
        mod.SurfaceProfileRecord(
            r_n=0.5,
            psi_n=0.25,
            roots_er=[0.11],
            bootstrap_current_at_roots=[-0.0301],
            selected_ambipolar_er=0.11,
            selected_bootstrap_current=-0.0301,
            scan_dir=str(tmp_path / "conv_rN0p5"),
        )
    ]

    summary = mod.build_summary(
        vmec_summary={"fsq_total": 1.0e-6},
        records=records,
        roots=[],
        profile=profile,
        convergence_profile=convergence_profile,
        accuracy={
            "surfaces_checked": 1,
            "max_abs_er": 0.01,
            "max_abs_bootstrap": 1.0e-4,
            "passed": True,
        },
        figure_png=tmp_path / "figure.png",
        figure_pdf=tmp_path / "figure.pdf",
        scan_dir=tmp_path / "scan",
        representative_r_n=0.5,
    )

    metadata = summary["metadata"]
    contract = metadata["workflow_contract"]
    provenance = metadata["radial_profile_provenance"]
    assert contract["no_overclaim_gate"]["full_transport_gradients_claimed"] is False
    assert contract["differentiability"]["sfincs_kinetic_transport_solve"] == (
        "primal_solve_only_not_differentiated"
    )
    assert provenance["radial_coordinate_input"] == "r_N"
    assert provenance["radial_profile_axis"] == "normalized toroidal flux psi_N"
    assert provenance["all_bracketed_roots_preserved"] is True
    assert [surface["psi_n"] for surface in provenance["surfaces"]] == [0.0625, 0.25]
    assert provenance["surfaces"][1]["selected_ambipolar_er_recorded"] is True
    assert provenance["convergence"]["profile_present"] is True
    assert provenance["convergence"]["surfaces_checked"] == 1
    assert provenance["convergence"]["passed"] is True


def _optional_vmex_wout_fixture(vmex_module) -> Path | None:
    candidates: list[Path] = []
    env_text = os.environ.get("DKX_VMEX_WOUT", "").strip()
    if env_text:
        candidates.append(Path(env_text))
    candidates.append(
        Path(vmex_module.__file__).resolve().parents[1]
        / "examples"
        / "data"
        / "wout_circular_tokamak.nc"
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # The package is installed, so the integration gate must run: fall back to
    # the release-asset equilibrium cache rather than skipping on local paths.
    from dkx.validation.data_fetch import resolve_external_equilibrium

    return resolve_external_equilibrium(Path("wout_w7x_standardConfig.nc"))


def test_vmex_boozer_spectrum_proxy_gradient_matches_fd_on_optional_backends() -> None:
    vmex = pytest.importorskip("vmex")
    pytest.importorskip("booz_xform_jax")
    from booz_xform_jax import Booz_xform
    from booz_xform_jax.jax_api import booz_xform_jax
    read_vmex_wout = vmex.read_wout

    fixture = _optional_vmex_wout_fixture(vmex)
    if fixture is None:
        pytest.skip("optional vmex wout fixture not found")

    wout_like = read_vmex_wout(fixture)
    bx = Booz_xform()
    try:
        bx.read_wout_data(wout_like)
    except AttributeError:
        bx.read_wout(str(fixture))
    bx.mboz = 3
    bx.nboz = 3

    surface_index = int(np.argmin(np.abs(np.asarray(bx.s_in) - 0.5)))
    rmnc = jnp.asarray(np.asarray(bx.rmnc).T)
    zmns = jnp.asarray(np.asarray(bx.zmns).T)
    lmns = jnp.asarray(np.asarray(bx.lmns).T)
    bmnc0 = jnp.asarray(np.asarray(bx.bmnc).T)
    bsubumnc = jnp.asarray(np.asarray(bx.bsubumnc).T)
    bsubvmnc = jnp.asarray(np.asarray(bx.bsubvmnc).T)
    iota = jnp.asarray(np.asarray(bx.iota))
    xm_nyq = jnp.asarray(np.asarray(bx.xm_nyq))
    xn_nyq = jnp.asarray(np.asarray(bx.xn_nyq))
    non_axis = (xm_nyq != 0) | (xn_nyq != 0)

    theta = jnp.linspace(0.0, 2.0 * jnp.pi, 8, endpoint=False)
    zeta = jnp.linspace(0.0, 2.0 * jnp.pi / float(bx.nfp), 6, endpoint=False)

    def objective(scale: jnp.ndarray) -> jnp.ndarray:
        bmnc = jnp.where(non_axis[None, :], bmnc0 * scale, bmnc0)
        out = booz_xform_jax(
            rmnc=rmnc,
            zmns=zmns,
            lmns=lmns,
            bmnc=bmnc,
            bsubumnc=bsubumnc,
            bsubvmnc=bsubvmnc,
            iota=iota,
            xm=bx.xm,
            xn=bx.xn,
            xm_nyq=bx.xm_nyq,
            xn_nyq=bx.xn_nyq,
            nfp=int(bx.nfp),
            mboz=int(bx.mboz),
            nboz=int(bx.nboz),
            asym=bool(bx.asym),
            surface_indices=[surface_index],
        )
        return boozer_spectrum_geometry_proxy_objective(
            out["bmnc_b"][0],
            out["ixm_b"],
            out["ixn_b"],
            theta=theta,
            zeta=zeta,
        )

    scale0 = 1.0
    step = 1.0e-4
    value = float(objective(scale0))
    autodiff = float(jax.grad(objective)(scale0))
    finite_difference = float((objective(scale0 + step) - objective(scale0 - step)) / (2.0 * step))

    assert np.isfinite(value)
    assert np.isfinite(autodiff)
    assert autodiff == pytest.approx(finite_difference, rel=5.0e-3, abs=1.0e-7)


def test_vmex_woutdata_adapter_matches_file_reader_on_optional_fixture() -> None:
    vmex = pytest.importorskip("vmex")
    pytest.importorskip("netCDF4")
    read_vmex_wout = vmex.read_wout

    fixture = _optional_vmex_wout_fixture(vmex)
    if fixture is None:
        pytest.skip("optional vmex fixture not found")

    sfincs_file = read_vmec_wout(fixture)
    sfincs_from_vmex = vmec_wout_from_wout_like(read_vmex_wout(fixture))

    for name in (
        "bmnc",
        "gmnc",
        "bsubumnc",
        "bsubvmnc",
        "bsubsmns",
        "bsupumnc",
        "bsupvmnc",
        "rmnc",
        "zmns",
        "lmns",
    ):
        np.testing.assert_allclose(
            getattr(sfincs_from_vmex, name),
            getattr(sfincs_file, name),
            rtol=0.0,
            atol=0.0,
        )

    theta = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    zeta = np.linspace(0.0, 2.0 * np.pi / sfincs_file.nfp, 5, endpoint=False)
    geom_file = FluxSurfaceGeometry.from_vmec(sfincs_file, theta=theta, zeta=zeta, psi_n_wish=0.25)
    geom_vmex = FluxSurfaceGeometry.from_vmec(
        sfincs_from_vmex,
        theta=theta,
        zeta=zeta,
        psi_n_wish=0.25,
    )

    np.testing.assert_allclose(np.asarray(geom_vmex.b_hat), np.asarray(geom_file.b_hat), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        np.asarray(geom_vmex.db_hat_dtheta),
        np.asarray(geom_file.db_hat_dtheta),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(np.asarray(geom_vmex.d_hat), np.asarray(geom_file.d_hat), rtol=0.0, atol=0.0)
