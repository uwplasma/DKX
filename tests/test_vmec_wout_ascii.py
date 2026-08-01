"""The ASCII VMEC ``wout`` reader recovers what the writer put in.

``geometryScheme = 5`` takes a VMEC ``wout`` and upstream reads either form:
LIBSTELL's ``read_wout_file`` dispatches on the file and ``read_wout_text``
handles the text branch (``read_wout_mod.F``).  :func:`read_vmec_wout_ascii`
mirrors that branch for ``VMEC VERSION <= 8.0``.

Two levels are checked.  A synthesized file with known values pins the record
layout exactly and runs anywhere; the equilibrium the upstream suite ships in
*both* forms pins the reader against an independent implementation, and is
skipped when that file is not present.  See uwplasma/DKX#30.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from dkx.magnetic_geometry import read_vmec_wout, read_vmec_wout_ascii

# Small enough to write out by hand, big enough to have a half/full mesh
# distinction and more than one Fourier mode.
NS, MNMAX, NFP = 4, 3, 5
MODES = ((0, 0), (1, 0), (1, NFP))

#: The eleven symmetric values per (mn, surface) record, in file order.  The
#: eleventh is ``currvmnc``, which dkx does not carry.
RECORD = (
    "rmnc", "zmns", "lmns", "bmnc", "gmnc", "bsubumnc",
    "bsubvmnc", "bsubsmns", "bsupumnc", "bsupvmnc", "currvmnc",
)


def _value(field: str, mn: int, js: int) -> float:
    """A distinct, exactly representable value per (field, mode, surface)."""
    return (RECORD.index(field) + 1) * 100.0 + mn * 10.0 + js * 0.5


def _synth_wout(path: Path) -> dict:
    """Write a VMEC 8.00 text ``wout`` and return what it should read back as."""
    out: list[str] = ["VMEC VERSION = 8.00"]
    out.append(" ".join(["1.0"] * 7))  # wb wp gamma pfac rmax rmin zmax
    out.append(f"{NFP} {NS} 2 1 {MNMAX} 0 0 0 0 0")  # ... iasym=0 -> stell-sym
    out.append("-1 0 0 0 0 1")  # imse itse nbsets nobd nextcur nstore_seq
    out.append("none")  # mgrid_file

    for js in range(NS):
        for mn in range(MNMAX):
            if js == 0:
                out.append(f"{MODES[mn][0]} {MODES[mn][1]}")
            out.append(" ".join(repr(_value(f, mn, js)) for f in RECORD))

    # Full mesh: iotaf presf phipf phi jcuru jcurv
    presf = np.array([1.0 + js for js in range(NS)])
    phi = np.array([10.0 + js for js in range(NS)])
    for js in range(NS):
        out.append(f"0.0 {float(presf[js])!r} 0.0 {float(phi[js])!r} 0.0 0.0")

    # Half mesh, surfaces 2..ns: iotas mass pres beta_vol phip buco bvco vp overr specw
    iotas = np.zeros(NS)
    for js in range(1, NS):
        iotas[js] = 0.25 + js
        out.append(f"{float(iotas[js])!r} " + " ".join(["0.0"] * 9))

    out.append(" ".join(["0.0"] * 6))  # aspect betatot betapol betator betaxis b0
    out.append("1")  # isigng
    out.append("synthetic")  # input_extension
    aminor = 0.4321
    # IonLarmor VolAvgB RBtor0 RBtor Itor Aminor Rmajor Volume
    out.append(f"0.0 0.0 0.0 0.0 0.0 {aminor!r} 0.0 0.0")

    path.write_text("\n".join(out) + "\n")
    return {"presf": presf, "phi": phi, "iotas": iotas, "aminor_p": aminor}


def test_scalars_and_modes_round_trip(tmp_path):
    """Header counts, symmetry flag and the mode table come back unchanged."""
    path = tmp_path / "wout_synth.txt"
    _synth_wout(path)
    w = read_vmec_wout_ascii(path)
    assert (w.nfp, w.ns, w.mnmax, w.mnmax_nyq) == (NFP, NS, MNMAX, MNMAX)
    assert w.lasym is False
    assert list(zip(w.xm.tolist(), w.xn.tolist())) == list(MODES)
    # Version <= 8.0 writes no separate Nyquist grid, so it equals the base one.
    assert np.array_equal(w.xm_nyq, w.xm) and np.array_equal(w.xn_nyq, w.xn)


def test_every_fourier_table_lands_in_its_own_column(tmp_path):
    """The record layout is pinned value by value, not just by shape.

    A transposed or shifted column would still produce arrays of the right
    shape and plausible magnitude, which is exactly the failure a shape-only
    assertion misses.
    """
    path = tmp_path / "wout_synth.txt"
    _synth_wout(path)
    w = read_vmec_wout_ascii(path)
    for field in RECORD[:-1]:  # currvmnc is not carried
        got = np.asarray(getattr(w, field), dtype=float)
        want = np.array([[_value(field, mn, js) for js in range(NS)]
                         for mn in range(MNMAX)])
        assert np.array_equal(got, want), field


def test_profiles_keep_the_half_mesh_dummy(tmp_path):
    """Half-mesh arrays start at index 1, as the NetCDF form also does."""
    path = tmp_path / "wout_synth.txt"
    expected = _synth_wout(path)
    w = read_vmec_wout_ascii(path)
    assert w.iotas[0] == 0.0
    assert np.array_equal(w.iotas, expected["iotas"])
    assert np.array_equal(w.presf, expected["presf"])
    assert np.array_equal(w.phi, expected["phi"])
    assert w.aminor_p == pytest.approx(expected["aminor_p"])


def test_dispatch_picks_the_reader_from_the_file(tmp_path):
    """``read_vmec_wout`` routes by signature, so callers need not care."""
    path = tmp_path / "wout_synth.txt"
    _synth_wout(path)
    assert read_vmec_wout(path).nfp == NFP


def test_unsupported_versions_are_refused_by_name(tmp_path):
    """Above 8.0 the Nyquist tables move to their own records.

    Parsing such a file with the <= 8.0 layout would silently mis-assign every
    field, so it is refused rather than attempted.
    """
    path = tmp_path / "wout_new.txt"
    _synth_wout(path)
    path.write_text(path.read_text().replace("VMEC VERSION = 8.00",
                                             "VMEC VERSION = 9.00"))
    with pytest.raises(NotImplementedError, match="ASCII reader covers"):
        read_vmec_wout_ascii(path)


def test_animec_variant_is_refused(tmp_path):
    """ANIMEC writes extra columns per record; mis-parsing it would be silent."""
    path = tmp_path / "wout_animec.txt"
    _synth_wout(path)
    path.write_text(path.read_text().replace("VMEC VERSION = 8.00",
                                             "VMEC VERSION = 8.00_ANIMEC"))
    with pytest.raises(NotImplementedError, match="ANIMEC"):
        read_vmec_wout_ascii(path)


def test_a_truncated_file_says_where_it_ran_out(tmp_path):
    """Silent short reads are the failure mode this format invites."""
    path = tmp_path / "wout_short.txt"
    _synth_wout(path)
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[: len(lines) // 2]) + "\n")
    with pytest.raises(ValueError, match="ended early"):
        read_vmec_wout_ascii(path)


def _equilibrium(name: str) -> Path | None:
    for root in (os.environ.get("DKX_EQUILIBRIA_DIRS") or "").split(os.pathsep):
        if root and (path := Path(root) / name).exists():
            return path
    return None


def test_ascii_and_netcdf_agree_on_the_same_equilibrium():
    """The two readers on the two forms upstream ships of one equilibrium.

    The ASCII file is VMEC 8.0, which writes no separate Nyquist grid, so its
    ``mnmax_nyq`` tables carry a subset of the NetCDF file's modes.  That is a
    property of the files, not of the readers: compare the shared modes.
    """
    ascii_path = _equilibrium("wout_w7x_standardConfig.txt")
    netcdf_path = _equilibrium("wout_w7x_standardConfig.nc")
    if ascii_path is None or netcdf_path is None:
        pytest.skip("W7-X wout pair not present (set DKX_EQUILIBRIA_DIRS)")

    a = read_vmec_wout(ascii_path)
    b = read_vmec_wout(netcdf_path)
    assert (a.nfp, a.ns, a.mnmax, a.lasym) == (b.nfp, b.ns, b.mnmax, b.lasym)
    assert a.aminor_p == pytest.approx(b.aminor_p, rel=1e-12)

    for field in ("phi", "xm", "xn", "rmnc", "zmns", "lmns", "iotas", "presf"):
        x = np.asarray(getattr(a, field), dtype=float)
        y = np.asarray(getattr(b, field), dtype=float)
        scale = max(float(np.abs(y).max()), 1e-30)
        assert np.abs(x - y).max() / scale < 1e-12, field

    index = {(int(m), int(n)): i for i, (m, n) in enumerate(zip(b.xm_nyq, b.xn_nyq))}
    shared = [index[(int(m), int(n))] for m, n in zip(a.xm_nyq, a.xn_nyq)]
    for field in ("bmnc", "gmnc", "bsubumnc", "bsubvmnc", "bsubsmns",
                  "bsupumnc", "bsupvmnc"):
        x = np.asarray(getattr(a, field), dtype=float)
        y = np.asarray(getattr(b, field), dtype=float)[shared, :]
        scale = max(float(np.abs(y).max()), 1e-30)
        assert np.abs(x - y).max() / scale < 1e-12, field
