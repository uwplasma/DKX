from __future__ import annotations

import numpy as np

from sfincs_jax.namelist import Namelist
from sfincs_jax.discretization.v3 import grids_from_namelist


def _nml(*, other: dict | None = None, rhs_mode: int = 1) -> Namelist:
    return Namelist(
        groups={
            "resolutionparameters": {
                "NTHETA": 5,
                "NZETA": 1,
                "NX": 6,
                "NXI": 8,
                "NL": 4,
            },
            "othernumericalparameters": {
                "XGRIDSCHEME": 50,
                "NXI_FOR_X_OPTION": 0,
                **(other or {}),
            },
            "geometryparameters": {
                "GEOMETRYSCHEME": 1,
            },
            "general": {
                "RHSMODE": rhs_mode,
            },
        },
        indexed={},
        source_path=None,
        source_text=None,
    )


def test_v3_grids_support_opt_in_rational_tail_mapped_xgrid():
    nml = _nml(
        other={
            "MAPPEDXGRIDFAMILY": "rational_tail",
            "MAPPEDXGRIDLOGLENGTH": 0.0,
            "MAPPEDXGRIDEPS": 1.0e-4,
        }
    )
    grids = grids_from_namelist(nml)
    x = np.asarray(grids.x)
    weights = np.asarray(grids.x_weights)

    assert x.shape == (6,)
    assert weights.shape == (6,)
    assert np.all(np.diff(x) > 0.0)
    assert np.all(weights > 0.0)
    assert grids.ddx.shape == (6, 6)
    assert grids.d2dx2.shape == (6, 6)
    assert np.asarray(grids.n_xi_for_x).tolist() == [8] * 6

    poly = x**3 - 2.0 * x + 1.0
    dpoly = 3.0 * x**2 - 2.0
    np.testing.assert_allclose(np.asarray(grids.ddx) @ poly, dpoly, rtol=1.0e-8, atol=1.0e-8)


def test_v3_grids_support_opt_in_softplus_cell_mapped_xgrid():
    nml = _nml(
        other={
            "MAPPEDXGRIDFAMILY": "softplus_cell",
            "MAPPEDXGRIDETAKIND": "uniform",
            "MAPPEDXGRIDXMAX": 3.0,
            "MAPPEDXGRIDPARAM": 0.0,
        }
    )
    grids = grids_from_namelist(nml)
    x = np.asarray(grids.x)
    weights = np.asarray(grids.x_weights)

    assert np.all(np.diff(x) > 0.0)
    np.testing.assert_allclose(np.sum(weights), 3.0, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(x, np.linspace(0.25, 2.75, 6), rtol=1.0e-13, atol=1.0e-13)


def test_v3_grids_rhs_mode_3_keeps_monoenergetic_x_override():
    nml = _nml(
        rhs_mode=3,
        other={
            "MAPPEDXGRIDFAMILY": "rational_tail",
            "MAPPEDXGRIDLOGLENGTH": 3.0,
        },
    )
    grids = grids_from_namelist(nml)
    np.testing.assert_allclose(np.asarray(grids.x), np.asarray([1.0]), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(grids.x_weights), np.asarray([np.e]), rtol=1.0e-15, atol=1.0e-15)


def test_v3_grids_reject_bad_softplus_param_length():
    nml = _nml(
        other={
            "MAPPEDXGRIDFAMILY": "softplus_cell",
            "MAPPEDXGRIDPARAMS": [0.0, 1.0],
        }
    )
    try:
        grids_from_namelist(nml)
    except ValueError as exc:
        assert "mappedXGridParams" in str(exc)
    else:
        raise AssertionError("expected a ValueError for wrong mappedXGridParams length")
