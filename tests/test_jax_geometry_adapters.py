from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sfincs_jax.jax_geometry_adapters import optional_jax_geometry_backend_status, vmec_wout_from_wout_like
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


def test_vmec_jax_woutdata_adapter_matches_file_reader_on_optional_fixture() -> None:
    vmec_jax = pytest.importorskip("vmec_jax")
    pytest.importorskip("netCDF4")
    from vmec_jax.wout import read_wout as read_vmec_jax_wout

    fixture = Path(vmec_jax.__file__).resolve().parents[1] / "examples" / "data" / "wout_circular_tokamak.nc"
    if not fixture.exists():
        pytest.skip(f"optional vmec_jax fixture not found: {fixture}")

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
