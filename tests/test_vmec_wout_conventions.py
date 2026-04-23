from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sfincs_jax.vmec_wout import VmecWout, _set_scale_factor, psi_a_hat_from_wout, vmec_interpolation


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


def test_vmec_ripple_scale_factor_matches_helicity_selection_rules() -> None:
    assert _set_scale_factor(n=0, m=3, helicity_n=0, helicity_l=1, ripple_scale=0.2) == pytest.approx(1.0)
    assert _set_scale_factor(n=1, m=3, helicity_n=0, helicity_l=1, ripple_scale=0.2) == pytest.approx(0.2)

    assert _set_scale_factor(n=2, m=1, helicity_n=2, helicity_l=1, ripple_scale=0.3) == pytest.approx(1.0)
    assert _set_scale_factor(n=2, m=2, helicity_n=2, helicity_l=1, ripple_scale=0.3) == pytest.approx(0.3)
    assert _set_scale_factor(n=0, m=1, helicity_n=2, helicity_l=1, ripple_scale=0.3) == pytest.approx(0.3)
