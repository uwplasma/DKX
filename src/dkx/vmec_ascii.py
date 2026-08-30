"""The LIBSTELL text form of a VMEC ``wout``.

A VMEC ``wout`` ships as NetCDF or as LIBSTELL text, and upstream reads either:
``read_wout_file`` dispatches on the file and ``read_wout_text`` handles the
text branch (``read_wout_mod.F``).  :func:`dkx.magnetic_geometry.read_vmec_wout`
routes here by file signature, so callers never choose.

This lives outside :mod:`dkx.magnetic_geometry` because it is a file-format
parser rather than geometry: it shares only the :class:`VmecWout` container, and
the geometry module carries the schemes, the Boozer readers and the
differentiable Fourier path already.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dkx.magnetic_geometry import VmecWout, resolve_existing_path

__all__ = ["read_vmec_wout_ascii"]


class _TokenStream:
    """Whitespace-separated token cursor over the free-format part of a wout.

    For ``VMEC VERSION <= 8.0`` every record after the version line is a
    list-directed Fortran read (``READ(iunit,*)``), so record boundaries carry
    no information and the file is one flat token stream.  The two exceptions
    are read as whole lines by the reference implementation and handled by the
    caller.
    """

    def __init__(self, text: str) -> None:
        self._tokens = text.split()
        self._i = 0

    def take(self, count: int) -> list[str]:
        if self._i + count > len(self._tokens):
            raise ValueError(
                f"wout file ended early: wanted {count} more values at token "
                f"{self._i} of {len(self._tokens)}"
            )
        chunk = self._tokens[self._i : self._i + count]
        self._i += count
        return chunk

    def floats(self, count: int) -> np.ndarray:
        return np.array([float(t) for t in self.take(count)], dtype=np.float64)

    def ints(self, count: int) -> list[int]:
        return [int(float(t)) for t in self.take(count)]


def read_vmec_wout_ascii(path: str | Path) -> VmecWout:
    """Read a LIBSTELL text-format VMEC ``wout``.

    Upstream accepts either form for ``geometryScheme = 5`` -- LIBSTELL's
    ``read_wout_file`` dispatches on the file, and ``read_wout_text`` is the
    text branch this mirrors.  Supported here for ``VMEC VERSION <= 8.0``,
    where the Fourier records carry the Nyquist tables inline
    (``mnmax_nyq = mnmax``) and every record after the version line is
    list-directed.  Later versions split the Nyquist block out and are
    refused by name rather than mis-parsed.

    ``ANIMEC`` and ``FLOW`` variants write extra columns per record; their
    version strings are recognised and refused for the same reason.

    Returns the same :class:`VmecWout` layout as :func:`read_vmec_wout`, so
    the two readers are interchangeable -- which is what
    ``tests/test_vmec_wout_ascii_parity.py`` checks against the NetCDF twin of
    one equilibrium.
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        p = resolve_existing_path(path).path.resolve()

    lines = p.read_text(errors="replace").splitlines()
    header = lines[0]
    for variant in ("_ANIMEC", "_FLOW"):
        if variant in header:
            raise NotImplementedError(
                f"{p.name} is a {variant.lstrip('_')} wout; its Fourier records carry "
                "extra columns that this reader does not parse."
            )
    marker = header.find("=")
    version = float(header[marker + 1 :].split()[0]) if marker >= 0 else -1.0
    if not 6.54 <= version <= 8.0 + 1e-4:
        raise NotImplementedError(
            f"{p.name} declares VMEC version {version}; the ASCII reader covers "
            "6.54 through 8.0, where mnmax_nyq == mnmax and the Fourier records "
            "are self-contained. Use the NetCDF form (wout_*.nc) for other versions."
        )

    stream = _TokenStream("\n".join(lines[1:]))
    stream.floats(7)  # wb, wp, gamma, pfac, rmax_surf, rmin_surf, zmax_surf
    nfp, ns, mpol, ntor, mnmax, _itfsq, _niter, iasym, _irecon, _ierr = stream.ints(10)
    mnmax_nyq = mnmax  # version <= 8.0 writes the Nyquist tables inline
    lasym = iasym > 0
    _imse, _itse, nbsets, _nobd, _nextcur, _nstore = stream.ints(6)
    if nbsets > 0:
        stream.ints(nbsets)
    # ``mgrid_file`` is the one whole-line record inside the free-format region;
    # it is a filename, so consuming it as a single token is right unless it
    # contains spaces, which VMEC does not write.
    stream.take(1)

    shape_full = (mnmax, ns)
    shape_nyq = (mnmax_nyq, ns)
    xm = np.zeros(mnmax, dtype=np.int32)
    xn = np.zeros(mnmax, dtype=np.int32)
    sym = {name: np.zeros(shape_full) for name in ("rmnc", "zmns", "lmns")}
    sym.update({
        name: np.zeros(shape_nyq)
        for name in ("bmnc", "gmnc", "bsubumnc", "bsubvmnc", "bsubsmns",
                     "bsupumnc", "bsupvmnc")
    })
    asym = {
        name: (np.zeros(shape_full if name in {"rmns", "zmnc", "lmnc"} else shape_nyq)
               if lasym else None)
        for name in ("rmns", "zmnc", "lmnc", "bmns", "gmns", "bsubumns",
                     "bsubvmns", "bsubsmnc", "bsupumns", "bsupvmns")
    }

    # Record order per (js, mn), from read_wout_mod.F: the eleventh symmetric
    # value is currvmnc, which dkx does not carry.
    order_sym = ("rmnc", "zmns", "lmns", "bmnc", "gmnc", "bsubumnc",
                 "bsubvmnc", "bsubsmns", "bsupumnc", "bsupvmnc")
    order_asym = ("rmns", "zmnc", "lmnc", "bmns", "gmns", "bsubumns",
                  "bsubvmns", "bsubsmnc", "bsupumns", "bsupvmns")
    for js in range(ns):
        for mn in range(mnmax):
            if js == 0:
                m, n = stream.ints(2)
                xm[mn], xn[mn] = m, n
            values = stream.floats(11)  # ... + currvmnc, dropped
            for name, value in zip(order_sym, values[:10]):
                sym[name][mn, js] = value
            if lasym:
                values = stream.floats(10)
                for name, value in zip(order_asym, values):
                    asym[name][mn, js] = value

    # Full-mesh profiles: (iotaf, presf, phipf, phi, jcuru, jcurv) per surface.
    full = stream.floats(6 * ns).reshape(ns, 6)
    presf, phi = full[:, 1].copy(), full[:, 3].copy()

    # Half-mesh profiles run js = 2..ns in Fortran, so index 0 stays the dummy
    # that the NetCDF form also carries.
    half = stream.floats(10 * (ns - 1)).reshape(ns - 1, 10)
    iotas = np.zeros(ns)
    iotas[1:] = half[:, 0]

    stream.floats(6)  # aspect, betatot, betapol, betator, betaxis, b0
    stream.ints(1)  # isigng
    stream.take(1)  # input_extension
    # IonLarmor, VolAvgB, RBtor0, RBtor, Itor, Aminor, Rmajor, Volume
    aminor_p = float(stream.floats(8)[5])

    out = VmecWout(
        path=p, nfp=nfp, ns=ns, mpol=mpol, ntor=ntor, mnmax=mnmax,
        mnmax_nyq=mnmax_nyq, lasym=lasym, aminor_p=aminor_p,
        phi=phi, xm=xm, xn=xn, xm_nyq=xm.copy(), xn_nyq=xn.copy(),
        iotas=iotas, presf=presf,
        **sym,
        **({k: v for k, v in asym.items()} if lasym else {}),
    )
    if out.xm[0] != 0 or out.xn[0] != 0:
        raise ValueError("Expected the first VMEC mode to be (0,0).")
    return out


