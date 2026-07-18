from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import numpy as _np

from dkx.magnetic_geometry import (
    FluxSurfaceGeometry,
    VmecWout,
    _scale_factors,
    psi_a_hat_from_wout,
    read_vmec_wout,
    vmec_radial_interpolation,
)


def vmec_interpolation(*, w: VmecWout, psi_n_wish: float, vmec_radial_option: int):
    return vmec_radial_interpolation(w=w, psi_n_wish=psi_n_wish, vmec_radial_option=vmec_radial_option)


def _set_scale_factor(*, n: int, m: int, helicity_n: int, helicity_l: int, ripple_scale: float) -> float:
    """Scalar view of the canonical vectorized ``_scale_factors`` helper."""
    out = _scale_factors(
        m=_np.asarray([m], dtype=_np.float64),
        n_over_nfp=_np.asarray([n], dtype=_np.float64),
        helicity_n=helicity_n,
        helicity_l=helicity_l,
        ripple_scale=ripple_scale,
    )
    return float(out[0])


def _minimal_wout(*, ns: int = 5) -> VmecWout:
    mnmax = 2
    mnmax_nyq = 3
    return VmecWout(
        path=Path("synthetic_wout.nc"),
        nfp=5,
        ns=ns,
        mpol=2,
        ntor=1,
        mnmax=mnmax,
        mnmax_nyq=mnmax_nyq,
        lasym=False,
        aminor_p=0.5,
        phi=np.linspace(0.0, 2.0 * np.pi, ns),
        xm=np.asarray([0, 1], dtype=np.int32),
        xn=np.asarray([0, 5], dtype=np.int32),
        xm_nyq=np.asarray([0, 1, 2], dtype=np.int32),
        xn_nyq=np.asarray([0, 5, 10], dtype=np.int32),
        bmnc=np.ones((mnmax_nyq, ns), dtype=np.float64),
        gmnc=np.ones((mnmax_nyq, ns), dtype=np.float64),
        bsubumnc=np.zeros((mnmax_nyq, ns), dtype=np.float64),
        bsubvmnc=np.zeros((mnmax_nyq, ns), dtype=np.float64),
        bsubsmns=np.zeros((mnmax_nyq, ns), dtype=np.float64),
        bsupumnc=np.zeros((mnmax_nyq, ns), dtype=np.float64),
        bsupvmnc=np.zeros((mnmax_nyq, ns), dtype=np.float64),
        rmnc=np.zeros((mnmax, ns), dtype=np.float64),
        zmns=np.zeros((mnmax, ns), dtype=np.float64),
        lmns=np.zeros((mnmax, ns - 1), dtype=np.float64),
        iotas=np.linspace(0.4, 0.6, ns),
        presf=np.zeros((ns,), dtype=np.float64),
    )


def _write_minimal_wout_file(
    path: Path,
    *,
    lasym: int = 0,
    xm0: int = 0,
    xn0: int = 0,
    xm_nyq0: int = 0,
    xn_nyq0: int = 0,
    omit: str | None = None,
) -> None:
    from scipy.io import netcdf_file

    ns = 5
    mnmax = 2
    mnmax_nyq = 3

    def maybe_write(f, name: str, dtype: str, dims: tuple[str, ...], data) -> None:
        if name == omit:
            return
        var = f.createVariable(name, dtype, dims)
        var[...] = data

    with netcdf_file(path, "w") as f:
        f.createDimension("radius", ns)
        f.createDimension("radius_lmns", ns - 1)
        f.createDimension("mnmax", mnmax)
        f.createDimension("mnmax_nyq", mnmax_nyq)

        maybe_write(f, "nfp", "i", (), 5)
        maybe_write(f, "ns", "i", (), ns)
        maybe_write(f, "mpol", "i", (), 2)
        maybe_write(f, "ntor", "i", (), 1)
        maybe_write(f, "mnmax", "i", (), mnmax)
        maybe_write(f, "mnmax_nyq", "i", (), mnmax_nyq)
        maybe_write(f, "lasym__logical__", "i", (), lasym)
        maybe_write(f, "Aminor_p", "d", (), 0.5)
        maybe_write(f, "phi", "d", ("radius",), np.linspace(0.0, 2.0 * np.pi, ns))

        maybe_write(f, "xm", "i", ("mnmax",), np.asarray([xm0, 1], dtype=np.int32))
        maybe_write(f, "xn", "i", ("mnmax",), np.asarray([xn0, 5], dtype=np.int32))
        maybe_write(f, "xm_nyq", "i", ("mnmax_nyq",), np.asarray([xm_nyq0, 1, 2], dtype=np.int32))
        maybe_write(f, "xn_nyq", "i", ("mnmax_nyq",), np.asarray([xn_nyq0, 5, 10], dtype=np.int32))

        nyq_table = np.arange(ns * mnmax_nyq, dtype=np.float64).reshape(ns, mnmax_nyq)
        full_table = np.arange(ns * mnmax, dtype=np.float64).reshape(ns, mnmax)
        lmns_table = np.arange((ns - 1) * mnmax, dtype=np.float64).reshape(ns - 1, mnmax)
        for name in ("bmnc", "gmnc", "bsubumnc", "bsubvmnc", "bsubsmns", "bsupumnc", "bsupvmnc"):
            maybe_write(f, name, "d", ("radius", "mnmax_nyq"), nyq_table)
        maybe_write(f, "rmnc", "d", ("radius", "mnmax"), full_table)
        maybe_write(f, "zmns", "d", ("radius", "mnmax"), full_table + 100.0)
        maybe_write(f, "lmns", "d", ("radius_lmns", "mnmax"), lmns_table)
        maybe_write(f, "iotas", "d", ("radius",), np.linspace(0.4, 0.6, ns))
        maybe_write(f, "presf", "d", ("radius",), np.linspace(1.0, 0.0, ns))


def test_psi_a_hat_uses_last_vmec_phi_over_two_pi() -> None:
    assert psi_a_hat_from_wout(_minimal_wout()) == pytest.approx(1.0)


def test_vmec_interpolation_matches_full_and_half_mesh_conventions() -> None:
    interp = vmec_interpolation(w=_minimal_wout(), psi_n_wish=0.25, vmec_radial_option=0)

    np.testing.assert_allclose(interp.psi_n_full, np.asarray([0.0, 0.25, 0.5, 0.75, 1.0]))
    np.testing.assert_allclose(interp.psi_n_half, np.asarray([0.125, 0.375, 0.625, 0.875]))
    assert interp.psi_n == pytest.approx(0.25)
    assert interp.index_full == (1, 2)
    assert interp.weight_full == pytest.approx((1.0, 0.0))
    assert interp.index_half == (1, 2)
    assert interp.weight_half == pytest.approx((0.5, 0.5))


def test_vmec_radial_options_snap_to_half_or_full_mesh() -> None:
    w = _minimal_wout()

    nearest_half = vmec_interpolation(w=w, psi_n_wish=0.31, vmec_radial_option=1)
    assert nearest_half.psi_n == pytest.approx(0.375)
    assert nearest_half.index_full == (1, 2)
    assert nearest_half.weight_full == pytest.approx((0.5, 0.5))
    assert nearest_half.index_half == (2, 3)
    assert nearest_half.weight_half == pytest.approx((1.0, 0.0))

    nearest_full = vmec_interpolation(w=w, psi_n_wish=0.31, vmec_radial_option=2)
    assert nearest_full.psi_n == pytest.approx(0.25)
    assert nearest_full.index_full == (1, 2)
    assert nearest_full.weight_full == pytest.approx((1.0, 0.0))


def test_vmec_interpolation_endpoint_and_invalid_inputs() -> None:
    edge = vmec_interpolation(w=_minimal_wout(), psi_n_wish=1.0, vmec_radial_option=0)
    assert edge.index_full == (3, 4)
    assert edge.weight_full == pytest.approx((0.0, 1.0))
    assert edge.index_half == (3, 4)
    assert edge.weight_half == pytest.approx((-0.5, 1.5))

    with pytest.raises(ValueError, match="psiN_wish"):
        vmec_interpolation(w=_minimal_wout(), psi_n_wish=-0.1, vmec_radial_option=0)
    with pytest.raises(ValueError, match="Invalid VMECRadialOption"):
        vmec_interpolation(w=_minimal_wout(), psi_n_wish=0.5, vmec_radial_option=99)


def test_vmec_interpolation_covers_inner_and_exact_outer_half_mesh() -> None:
    inner = vmec_interpolation(w=_minimal_wout(), psi_n_wish=0.0, vmec_radial_option=0)
    assert inner.index_full == (0, 1)
    assert inner.weight_full == pytest.approx((1.0, 0.0))
    assert inner.index_half == (1, 2)
    assert inner.weight_half == pytest.approx((1.5, -0.5))

    outer_half = vmec_interpolation(w=_minimal_wout(), psi_n_wish=0.875, vmec_radial_option=0)
    assert outer_half.index_full == (3, 4)
    assert outer_half.weight_full == pytest.approx((0.5, 0.5))
    assert outer_half.index_half == (3, 4)
    assert outer_half.weight_half == pytest.approx((0.0, 1.0))


def test_vmec_ripple_scale_factor_matches_helicity_selection_rules() -> None:
    assert _set_scale_factor(n=0, m=3, helicity_n=0, helicity_l=1, ripple_scale=0.2) == pytest.approx(1.0)
    assert _set_scale_factor(n=1, m=3, helicity_n=0, helicity_l=1, ripple_scale=0.2) == pytest.approx(0.2)

    assert _set_scale_factor(n=2, m=1, helicity_n=2, helicity_l=1, ripple_scale=0.3) == pytest.approx(1.0)
    assert _set_scale_factor(n=2, m=2, helicity_n=2, helicity_l=1, ripple_scale=0.3) == pytest.approx(0.3)
    assert _set_scale_factor(n=0, m=1, helicity_n=2, helicity_l=1, ripple_scale=0.3) == pytest.approx(0.3)


def test_gpsipsi_from_vmec_reconstructs_finite_metric_and_fail_closed() -> None:
    ns = 5
    base = _minimal_wout(ns=ns)
    w = replace(
        base,
        nfp=1,
        ntor=0,
        xm=np.asarray([0, 1], dtype=np.int32),
        xn=np.asarray([0, 0], dtype=np.int32),
        xm_nyq=np.asarray([0, 1], dtype=np.int32),
        xn_nyq=np.asarray([0, 0], dtype=np.int32),
        bmnc=np.asarray(
            [[1.0, 1.0, 1.0, 1.0, 1.0], [0.05, 0.05, 0.05, 0.05, 0.05]],
            dtype=np.float64,
        ),
        rmnc=np.asarray(
            [[1.5, 1.6, 1.7, 1.8, 1.9], [0.10, 0.12, 0.14, 0.16, 0.18]],
            dtype=np.float64,
        ),
        zmns=np.asarray(
            [[0.0, 0.0, 0.0, 0.0, 0.0], [0.08, 0.10, 0.12, 0.14, 0.16]],
            dtype=np.float64,
        ),
    )
    theta = np.asarray([0.0, np.pi / 2.0, np.pi], dtype=np.float64)
    zeta = np.asarray([0.0, np.pi], dtype=np.float64)
    geom = FluxSurfaceGeometry.from_vmec(
        w,
        theta=theta,
        zeta=zeta,
        psi_n_wish=0.5,
        vmec_radial_option=0,
        min_bmn_to_load=0.02,
        ripple_scale=0.5,
        vmec_nyquist_option=1,
        compute_gpsipsi=True,
    )
    metric = np.asarray(geom.gpsipsi)

    assert metric.shape == (3, 2)
    assert np.all(np.isfinite(metric))
    assert np.all(metric > 0.0)

    with pytest.raises(ValueError, match="VMEC_Nyquist_option"):
        FluxSurfaceGeometry.from_vmec(
            w,
            theta=theta,
            zeta=zeta,
            psi_n_wish=0.5,
            vmec_radial_option=0,
            vmec_nyquist_option=99,
        )

    with pytest.raises(ValueError, match="No VMEC modes"):
        FluxSurfaceGeometry.from_vmec(
            w,
            theta=theta,
            zeta=zeta,
            psi_n_wish=0.5,
            vmec_radial_option=0,
            min_bmn_to_load=2.0,
        )

    zero_b00 = replace(w, bmnc=np.zeros_like(w.bmnc))
    with pytest.raises(ValueError, match=r"bmnc\(0,0\)"):
        FluxSurfaceGeometry.from_vmec(
            zero_b00,
            theta=theta,
            zeta=zeta,
            psi_n_wish=0.5,
            vmec_radial_option=0,
            min_bmn_to_load=0.02,
        )


def test_read_vmec_wout_transposes_radius_mode_tables(tmp_path: Path) -> None:
    nc_path = tmp_path / "wout_synthetic.nc"
    _write_minimal_wout_file(nc_path)

    wout = read_vmec_wout(nc_path)

    assert wout.path == nc_path.resolve()
    assert wout.nfp == 5
    assert wout.ns == 5
    assert wout.mnmax == 2
    assert wout.mnmax_nyq == 3
    assert not wout.lasym
    np.testing.assert_allclose(wout.phi, np.linspace(0.0, 2.0 * np.pi, 5))
    np.testing.assert_array_equal(wout.xm, np.asarray([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(wout.xn_nyq, np.asarray([0, 5, 10], dtype=np.int32))

    expected_nyq = np.arange(15, dtype=np.float64).reshape(5, 3).T
    expected_full = np.arange(10, dtype=np.float64).reshape(5, 2).T
    expected_lmns = np.arange(8, dtype=np.float64).reshape(4, 2).T
    np.testing.assert_allclose(wout.bmnc, expected_nyq)
    np.testing.assert_allclose(wout.bsupvmnc, expected_nyq)
    np.testing.assert_allclose(wout.rmnc, expected_full)
    np.testing.assert_allclose(wout.zmns, expected_full + 100.0)
    np.testing.assert_allclose(wout.lmns, expected_lmns)


def test_read_vmec_wout_rejects_missing_file_and_required_variables(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_vmec_wout(tmp_path / "does_not_exist.nc")

    missing_var = tmp_path / "wout_missing_var.nc"
    _write_minimal_wout_file(missing_var, omit="presf")
    with pytest.raises(KeyError, match="presf"):
        read_vmec_wout(missing_var)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"xm0": 1},
        {"xn0": 5},
        {"xm_nyq0": 1},
        {"xn_nyq0": 5},
    ],
)
def test_read_vmec_wout_rejects_invalid_first_mode_metadata(
    tmp_path: Path,
    kwargs: dict[str, int],
) -> None:
    path = tmp_path / "wout_bad_metadata.nc"
    _write_minimal_wout_file(path, **kwargs)

    with pytest.raises(ValueError, match=r"first VMEC mode to be \(0,0\)"):
        read_vmec_wout(path)
