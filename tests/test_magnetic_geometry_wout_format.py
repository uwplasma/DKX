"""``read_vmec_wout`` names the ASCII wout format instead of failing opaquely.

A VMEC ``wout`` ships in two forms and upstream reads both (LIBSTELL
``read_wout_file``, ``geometry.F90:96``).  ``dkx`` reads only NetCDF.  Without
the format check the ASCII form surfaces as a scipy "not a valid NetCDF 3 file"
``TypeError`` from inside the parser, which reads like a corrupt file rather
than a missing capability -- and that is how it went unnoticed across several
parity campaigns.  See uwplasma/DKX#30.
"""

from __future__ import annotations

import pytest

from dkx.magnetic_geometry import read_vmec_wout


def test_ascii_wout_is_refused_by_name(tmp_path):
    """The LIBSTELL text format is recognised and named, not mis-parsed."""
    ascii_wout = tmp_path / "wout_example.txt"
    ascii_wout.write_text(
        "VMEC VERSION = 8.00\n"
        " 3.117986435492261 5.032045665837008E-13 0.000000000000000E+00\n"
        " 5 99 12 12 288 99 10000 0 0 0\n"
    )
    with pytest.raises(NotImplementedError, match="ASCII wout"):
        read_vmec_wout(ascii_wout)


def test_refusal_points_at_the_netcdf_form(tmp_path):
    """The message has to say what to do, not only what went wrong."""
    ascii_wout = tmp_path / "wout_example.txt"
    ascii_wout.write_text("VMEC VERSION = 8.00\n")
    with pytest.raises(NotImplementedError) as excinfo:
        read_vmec_wout(ascii_wout)
    assert "wout_*.nc" in str(excinfo.value)


def test_an_unrelated_binary_file_is_also_refused(tmp_path):
    """Anything without a NetCDF/HDF5 signature is refused the same way."""
    junk = tmp_path / "wout_junk.nc"
    junk.write_bytes(b"\x00\x01\x02\x03rubbish")
    with pytest.raises(NotImplementedError, match="not a NetCDF VMEC wout"):
        read_vmec_wout(junk)
