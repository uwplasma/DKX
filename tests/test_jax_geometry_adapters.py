from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from sfincs_jax.jax_geometry_adapters import optional_jax_geometry_backend_status, vmec_wout_from_wout_like


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


def test_vmec_wout_from_wout_like_rejects_bad_shapes() -> None:
    bad = _wout_like(radius_mode_order=True)
    bad.bmnc = np.zeros((2, 2, 2))
    with pytest.raises(ValueError, match="bmnc must be 2D"):
        vmec_wout_from_wout_like(bad)
