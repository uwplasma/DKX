from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from sfincs_jax.jax_geometry_adapters import (
    boozer_bhat_from_spectrum,
    boozer_spectrum_geometry_proxy_objective,
    geometry_proxy_workflow_summary,
    optional_jax_geometry_backend_report,
    optional_jax_geometry_backend_status,
    vmec_wout_from_wout_like,
)
from sfincs_jax.vmec_geometry import vmec_geometry_from_wout
from sfincs_jax.vmec_wout import read_vmec_wout


def _wout_like(*, radius_mode_order: bool = True) -> SimpleNamespace:
    ns = 4
    mnmax = 3
    mnmax_nyq = 5

    def arr(modes: int, radius: int = ns) -> np.ndarray:
        data = np.arange(modes * radius, dtype=np.float64).reshape((modes, radius))
        return data.T if radius_mode_order else data

    return SimpleNamespace(
        path="vmec_jax_in_memory",
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
    assert set(status) == {"vmec_jax", "booz_xform_jax"}
    assert all(isinstance(value, bool) for value in status.values())


def test_optional_jax_geometry_backend_report_marks_gradient_boundary() -> None:
    report = optional_jax_geometry_backend_report()

    assert report["backends"] == optional_jax_geometry_backend_status()
    assert report["gradient_availability"]["spectral_scale_to_boozer_proxy"] == (
        "available_when_optional_backends_installed"
    )
    assert (
        report["gradient_availability"]["sfincs_kinetic_transport_solve"]
        == "not_covered_by_this_lane"
    )
    assert "booz_xform_jax" in report["differentiated_graph"]
    assert "SFINCS kinetic transport solve" in report["outside_differentiated_graph"]
    assert report["claim"] == "geometry_proxy_gradient_gate_not_full_transport_gradient"


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
        backend_status={"vmec_jax": True, "booz_xform_jax": False},
    )

    assert summary["workflow"] == "vmec_jax_to_boozer_sfincs_geometry_proxy"
    assert summary["provenance"]["source"] == "unit-test wout"
    assert summary["required_optional_dependencies"]["vmec_jax"]["importable"] is True
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


def test_vmec_wout_from_wout_like_transposes_vmec_jax_radius_mode_arrays() -> None:
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


def test_public_vmec_jax_boozer_example_backend_check_is_runnable() -> None:
    script = (
        Path(__file__).parents[1]
        / "examples"
        / "autodiff"
        / "vmec_jax_to_boozer_sfincs_pipeline.py"
    )
    result = subprocess.run(
        [sys.executable, str(script), "--check-backends"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Optional JAX geometry backend status:" in result.stdout
    assert "vmec_jax:" in result.stdout
    assert "booz_xform_jax:" in result.stdout
    assert "file-backed/setup only:" in result.stdout
    assert "not claimed: full VMEC-boundary-to-SFINCS-transport gradients" in result.stdout
    assert "numerical gradient gate: not_run" in result.stdout
    assert "pass --json with --check-backends" in result.stdout
    assert "pass --summary-json PATH" in result.stdout


def test_public_vmec_jax_boozer_example_backend_check_json_is_runnable() -> None:
    script = (
        Path(__file__).parents[1]
        / "examples"
        / "autodiff"
        / "vmec_jax_to_boozer_sfincs_pipeline.py"
    )
    result = subprocess.run(
        [sys.executable, str(script), "--check-backends", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(result.stdout)
    assert set(report["backends"]) == {"vmec_jax", "booz_xform_jax"}
    assert report["gradient_availability"]["vmec_file_io"] == "setup_only_not_differentiated"
    assert report["gradient_availability"]["sfincs_kinetic_transport_solve"] == "not_covered_by_this_lane"


def test_public_vmec_jax_boozer_example_backend_check_writes_summary_json(tmp_path: Path) -> None:
    script = (
        Path(__file__).parents[1]
        / "examples"
        / "autodiff"
        / "vmec_jax_to_boozer_sfincs_pipeline.py"
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
    assert summary["workflow"] == "vmec_jax_to_boozer_sfincs_geometry_proxy"
    assert summary["numerical_gradient_gate"]["status"] == "not_run"
    assert summary["claims"]["not_claimed"] == (
        "full VMEC-boundary-to-SFINCS kinetic transport gradients"
    )


def _optional_vmec_jax_wout_fixture(vmec_jax_module) -> Path | None:
    candidates: list[Path] = []
    env_text = os.environ.get("SFINCS_JAX_VMEC_JAX_WOUT", "").strip()
    if env_text:
        candidates.append(Path(env_text))
    candidates.extend(
        [
            Path(vmec_jax_module.__file__).resolve().parents[1]
            / "examples"
            / "data"
            / "wout_circular_tokamak.nc",
            Path("/Users/rogeriojorge/local/vmec_jax/examples/data/wout_circular_tokamak.nc"),
            Path.cwd() / "tests" / "ref" / "wout_w7x_standardConfig.nc",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def test_vmec_jax_boozer_spectrum_proxy_gradient_matches_fd_on_optional_backends() -> None:
    vmec_jax = pytest.importorskip("vmec_jax")
    pytest.importorskip("booz_xform_jax")
    from booz_xform_jax import Booz_xform
    from booz_xform_jax.jax_api import booz_xform_jax
    from vmec_jax.wout import read_wout as read_vmec_jax_wout

    fixture = _optional_vmec_jax_wout_fixture(vmec_jax)
    if fixture is None:
        pytest.skip("optional vmec_jax wout fixture not found")

    wout_like = read_vmec_jax_wout(fixture)
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


def test_vmec_jax_woutdata_adapter_matches_file_reader_on_optional_fixture() -> None:
    vmec_jax = pytest.importorskip("vmec_jax")
    pytest.importorskip("netCDF4")
    from vmec_jax.wout import read_wout as read_vmec_jax_wout

    fixture = _optional_vmec_jax_wout_fixture(vmec_jax)
    if fixture is None:
        pytest.skip("optional vmec_jax fixture not found")

    sfincs_file = read_vmec_wout(fixture)
    sfincs_from_vmec_jax = vmec_wout_from_wout_like(read_vmec_jax_wout(fixture))

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
            getattr(sfincs_from_vmec_jax, name),
            getattr(sfincs_file, name),
            rtol=0.0,
            atol=0.0,
        )

    theta = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    zeta = np.linspace(0.0, 2.0 * np.pi / sfincs_file.nfp, 5, endpoint=False)
    geom_file = vmec_geometry_from_wout(w=sfincs_file, theta=theta, zeta=zeta, psi_n_wish=0.25)
    geom_vmec_jax = vmec_geometry_from_wout(
        w=sfincs_from_vmec_jax,
        theta=theta,
        zeta=zeta,
        psi_n_wish=0.25,
    )

    np.testing.assert_allclose(np.asarray(geom_vmec_jax.b_hat), np.asarray(geom_file.b_hat), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        np.asarray(geom_vmec_jax.db_hat_dtheta),
        np.asarray(geom_file.db_hat_dtheta),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(np.asarray(geom_vmec_jax.d_hat), np.asarray(geom_file.d_hat), rtol=0.0, atol=0.0)
