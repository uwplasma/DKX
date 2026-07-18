"""Canonical ``export_f`` distribution-function output parity.

The canonical writer (:mod:`dkx.writer`) now computes the ``full_f`` /
``delta_f`` distribution-function export on the ``export_f`` user grids from the
solved state, so an ``export_full_f``/``export_delta_f`` deck no longer falls
back to any non-canonical pipeline.

The fixture ``quick_2species_FPCollisions_noEr`` (2-species Fokker-Planck,
geometryScheme=4) requests ``export_full_f``/``export_delta_f`` with
``export_f_x_option=1`` (barycentric speed interpolation) and
``export_f_xi_option=1`` (Legendre pitch reconstruction); its Fortran reference
``output_scheme4_2species_quick.sfincsOutput.h5`` carries the ``full_f`` /
``delta_f`` datasets.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dkx.api import write_output
from dkx.io import read_sfincs_h5
from dkx.run import run_profile

REF = Path(__file__).parent / "ref"

_EXPORT_F_DATA = ("full_f", "delta_f")
_EXPORT_F_GRIDS = ("export_f_theta", "export_f_zeta", "export_f_x", "export_f_xi")
_EXPORT_F_META = (
    "export_f_theta_option",
    "export_f_zeta_option",
    "export_f_x_option",
    "export_f_xi_option",
    "N_export_f_theta",
    "N_export_f_zeta",
    "N_export_f_x",
    "N_export_f_xi",
    "export_full_f",
    "export_delta_f",
)


def test_canonical_export_f_matches_fortran_reference(tmp_path: Path) -> None:
    input_path = REF / "quick_2species_FPCollisions_noEr.input.namelist"
    ref = read_sfincs_h5(REF / "output_scheme4_2species_quick.sfincsOutput.h5")

    out_path = tmp_path / "canonical.h5"
    run = run_profile(input_path, out_path=out_path, emit=None)
    out = read_sfincs_h5(run.output_path)

    for key in _EXPORT_F_DATA + _EXPORT_F_GRIDS + _EXPORT_F_META:
        assert key in out, f"canonical writer missing export_f dataset {key!r}"
        assert key in ref, f"reference missing export_f dataset {key!r}"

    # full_f/delta_f distribution-function parity at ~1e-10.
    for key in _EXPORT_F_DATA:
        a = np.asarray(out[key], dtype=np.float64)
        b = np.asarray(ref[key], dtype=np.float64)
        assert a.shape == b.shape, f"{key}: {a.shape} != {b.shape}"
        np.testing.assert_allclose(a, b, rtol=0.0, atol=1e-9, err_msg=key)

    # Export grids and option/count metadata match exactly.
    for key in _EXPORT_F_GRIDS:
        np.testing.assert_allclose(
            np.asarray(out[key], dtype=np.float64),
            np.asarray(ref[key], dtype=np.float64),
            rtol=0.0,
            atol=1e-12,
            err_msg=key,
        )
    for key in _EXPORT_F_META:
        assert int(np.asarray(out[key])) == int(np.asarray(ref[key])), key


def test_public_write_output_facade_matches_run_profile(tmp_path: Path) -> None:
    """``api.write_output`` routes the deck through the same canonical run.

    ``run_profile`` is called here with its default tolerance (1e-10) while
    ``write_output`` honors the deck's ``solverTolerance``, so the two solves
    agree only to the looser solver tolerance, not machine precision.
    """
    input_path = REF / "quick_2species_FPCollisions_noEr.input.namelist"

    canonical = tmp_path / "canonical.h5"
    facade = tmp_path / "facade.h5"
    run_profile(input_path, out_path=canonical, emit=None)
    write_output(input_path, facade)

    a = read_sfincs_h5(canonical)
    b = read_sfincs_h5(facade)
    for key in _EXPORT_F_DATA + _EXPORT_F_GRIDS:
        av = np.asarray(a[key], dtype=np.float64)
        bv = np.asarray(b[key], dtype=np.float64)
        assert av.shape == bv.shape, key
        assert av.dtype == np.asarray(a[key]).dtype
        np.testing.assert_allclose(av, bv, rtol=0.0, atol=1e-8, err_msg=key)
