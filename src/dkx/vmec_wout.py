"""VMEC ``wout`` files (geometryScheme 5): the reader and radial interpolation.

``read_vmec_wout`` routes NetCDF or LIBSTELL text by file signature; the
half/full-mesh interpolation follows ``computeBHat_VMEC``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

# The JAX backend is imported below; dkx/runtime.py explains why this is here.
from .runtime import configure as _configure_runtime

_configure_runtime()

import numpy as np  # noqa: E402

from .paths import (  # noqa: E402
    resolve_existing_path,
)

# =============================================================================
# VMEC wout files (geometryScheme 5): reader and radial interpolation
# =============================================================================

@dataclass(frozen=True)
class VmecWout:
    """VMEC ``wout`` data normalized to the internal scheme-5 layout.

    Fourier tables use the ``(mode, radius)`` convention (netCDF files store
    ``(radius, mode)``; the reader transposes).  Half-mesh arrays keep VMEC's
    dummy element at radius index 0, matching v3's indexing rules.
    """

    path: Path
    nfp: int
    ns: int
    mpol: int
    ntor: int
    mnmax: int
    mnmax_nyq: int
    lasym: bool
    aminor_p: float
    phi: np.ndarray  # (ns,)
    xm: np.ndarray  # (mnmax,)
    xn: np.ndarray  # (mnmax,)
    xm_nyq: np.ndarray  # (mnmax_nyq,)
    xn_nyq: np.ndarray  # (mnmax_nyq,) = n * nfp
    bmnc: np.ndarray  # (mnmax_nyq, ns) half mesh
    gmnc: np.ndarray  # (mnmax_nyq, ns) half mesh
    bsubumnc: np.ndarray  # (mnmax_nyq, ns) half mesh
    bsubvmnc: np.ndarray  # (mnmax_nyq, ns) half mesh
    bsubsmns: np.ndarray  # (mnmax_nyq, ns) full mesh
    bsupumnc: np.ndarray  # (mnmax_nyq, ns) half mesh
    bsupvmnc: np.ndarray  # (mnmax_nyq, ns) half mesh
    rmnc: np.ndarray  # (mnmax, ns) full mesh
    zmns: np.ndarray  # (mnmax, ns) full mesh
    lmns: np.ndarray  # (mnmax, ns) half mesh
    iotas: np.ndarray  # (ns,) half mesh
    presf: np.ndarray  # (ns,) full mesh
    # Stellarator-asymmetric (lasym=T) complementary-parity tables.  ``None`` for
    # stellarator-symmetric equilibria, so the stell-sym build is unaffected.  The
    # parity of each is flipped relative to its symmetric counterpart above: the
    # cosine tables gain sine partners (bmns/gmns/bsub{u,v}mns/bsup{u,v}mns/rmns),
    # and the sine tables gain cosine partners (bsubsmnc, zmnc).
    bmns: np.ndarray | None = None  # (mnmax_nyq, ns) half mesh
    gmns: np.ndarray | None = None  # (mnmax_nyq, ns) half mesh
    bsubumns: np.ndarray | None = None  # (mnmax_nyq, ns) half mesh
    bsubvmns: np.ndarray | None = None  # (mnmax_nyq, ns) half mesh
    bsubsmnc: np.ndarray | None = None  # (mnmax_nyq, ns) full mesh
    bsupumns: np.ndarray | None = None  # (mnmax_nyq, ns) half mesh
    bsupvmns: np.ndarray | None = None  # (mnmax_nyq, ns) half mesh
    rmns: np.ndarray | None = None  # (mnmax, ns) full mesh
    zmnc: np.ndarray | None = None  # (mnmax, ns) full mesh
    lmnc: np.ndarray | None = None  # (mnmax, ns) half mesh

def read_vmec_wout(path: str | Path) -> VmecWout:
    """Read a VMEC ``wout_*.nc`` file.

    Pure function returning plain NumPy arrays; not differentiable.  The
    returned tables feed :meth:`FluxSurfaceGeometry.from_vmec`.  For
    stellarator-asymmetric equilibria (``lasym=T``) the complementary-parity
    tables (``bmns``, ``gmns``, ``bsub{u,v}mns``, ``bsubsmnc``, ``bsup{u,v}mns``,
    ``rmns``, ``zmnc``, ``lmnc``) are also loaded; they stay ``None`` for
    stellarator-symmetric files so that path is unchanged.
    """
    from scipy.io import netcdf_file  # noqa: PLC0415 - keep scipy import lazy

    p = Path(path).expanduser().resolve()
    if not p.exists():
        p = resolve_existing_path(path).path.resolve()

    # A VMEC ``wout`` comes in two forms and upstream reads both: LIBSTELL's
    # ``read_wout_file`` (``geometry.F90:96``) auto-detects ASCII or NetCDF.
    # This reader handles NetCDF only, and without this check the ASCII form
    # surfaces as a scipy "not a valid NetCDF 3 file" ``TypeError`` from inside
    # the parser, which reads like a corrupt file rather than a missing
    # capability.  See uwplasma/DKX#30.
    magic = p.open("rb").read(4)
    if not (magic.startswith(b"CDF") or magic == b"\x89HDF"):
        # LIBSTELL text form.  Imported lazily: dkx.vmec_ascii imports VmecWout
        # from here, so a module-level import would be circular.
        from dkx.vmec_ascii import read_vmec_wout_ascii  # noqa: PLC0415

        return read_vmec_wout_ascii(p)

    with netcdf_file(p, "r", mmap=False) as f:
        def var(name: str) -> np.ndarray:
            if name not in f.variables:
                raise KeyError(f"Missing variable {name!r} in wout file.")
            return np.array(f.variables[name].data)

        def opt_var(name: str) -> np.ndarray | None:
            if name not in f.variables:
                return None
            return np.array(f.variables[name].data).astype(np.float64).T

        nfp = int(var("nfp"))
        ns = int(var("ns"))
        lasym = bool(int(np.asarray(var("lasym__logical__")).reshape(())))
        out = VmecWout(
            path=p,
            nfp=nfp,
            ns=ns,
            mpol=int(var("mpol")),
            ntor=int(var("ntor")),
            mnmax=int(var("mnmax")),
            mnmax_nyq=int(var("mnmax_nyq")),
            lasym=lasym,
            aminor_p=float(np.asarray(var("Aminor_p")).reshape(())),
            phi=var("phi").astype(np.float64),
            xm=var("xm").astype(np.int32),
            xn=var("xn").astype(np.int32),
            xm_nyq=var("xm_nyq").astype(np.int32),
            xn_nyq=var("xn_nyq").astype(np.int32),
            bmnc=var("bmnc").astype(np.float64).T,
            gmnc=var("gmnc").astype(np.float64).T,
            bsubumnc=var("bsubumnc").astype(np.float64).T,
            bsubvmnc=var("bsubvmnc").astype(np.float64).T,
            bsubsmns=var("bsubsmns").astype(np.float64).T,
            bsupumnc=var("bsupumnc").astype(np.float64).T,
            bsupvmnc=var("bsupvmnc").astype(np.float64).T,
            rmnc=var("rmnc").astype(np.float64).T,
            zmns=var("zmns").astype(np.float64).T,
            lmns=var("lmns").astype(np.float64).T,
            iotas=var("iotas").astype(np.float64),
            presf=var("presf").astype(np.float64),
            # Complementary-parity tables (present only when lasym=T).
            bmns=opt_var("bmns") if lasym else None,
            gmns=opt_var("gmns") if lasym else None,
            bsubumns=opt_var("bsubumns") if lasym else None,
            bsubvmns=opt_var("bsubvmns") if lasym else None,
            bsubsmnc=opt_var("bsubsmnc") if lasym else None,
            bsupumns=opt_var("bsupumns") if lasym else None,
            bsupvmns=opt_var("bsupvmns") if lasym else None,
            rmns=opt_var("rmns") if lasym else None,
            zmnc=opt_var("zmnc") if lasym else None,
            lmnc=opt_var("lmnc") if lasym else None,
        )
    if out.xm[0] != 0 or out.xn[0] != 0 or out.xm_nyq[0] != 0 or out.xn_nyq[0] != 0:
        raise ValueError("Expected the first VMEC mode to be (0,0).")
    return out

def psi_a_hat_from_wout(w: VmecWout) -> float:
    """``psiAHat = phi(ns) / (2 pi)`` as in v3."""
    return float(w.phi[-1]) / (2.0 * math.pi)

@dataclass(frozen=True)
class VmecRadialInterpolation:
    """Resolved full- and half-mesh interpolation state for one VMEC radius."""

    index_full: tuple[int, int]  # 0-based
    weight_full: tuple[float, float]
    index_half: tuple[int, int]  # 0-based; half arrays have a dummy 0 at index 0
    weight_half: tuple[float, float]
    psi_n: float
    psi_n_full: np.ndarray  # (ns,)
    psi_n_half: np.ndarray  # (ns-1,)

def vmec_radial_interpolation(*, w: VmecWout, psi_n_wish: float, vmec_radial_option: int) -> VmecRadialInterpolation:
    """v3's radius selection and interpolation index/weight logic (scheme 5)."""
    psi_n_full = np.asarray(w.phi, dtype=np.float64) / float(w.phi[-1])
    psi_n_half = 0.5 * (psi_n_full[:-1] + psi_n_full[1:])
    if not (0.0 <= float(psi_n_wish) <= 1.0):
        raise ValueError("psiN_wish must be in [0,1].")

    if int(vmec_radial_option) == 0:
        psi_n = float(psi_n_wish)
    elif int(vmec_radial_option) == 1:
        psi_n = float(psi_n_half[int(np.argmin((psi_n_half - float(psi_n_wish)) ** 2))])
    elif int(vmec_radial_option) == 2:
        psi_n = float(psi_n_full[int(np.argmin((psi_n_full - float(psi_n_wish)) ** 2))])
    else:
        raise ValueError(f"Invalid VMECRadialOption={vmec_radial_option}")

    ns = int(w.ns)
    if psi_n == 1.0:
        i0, i1, w0 = ns - 2, ns - 1, 0.0
    else:
        i0 = int(math.floor(psi_n * (ns - 1)))
        i1 = i0 + 1
        w0 = float(i0 + 1) - float(psi_n) * float(ns - 1)

    # Half-mesh: file arrays have length ns with a dummy 0 at index 0, so real
    # half indices start at python index 1 (Fortran index 2).
    if float(psi_n) < float(psi_n_half[0]):
        j0, j1 = 1, 2
        wh0 = (float(psi_n_half[1]) - float(psi_n)) / (float(psi_n_half[1]) - float(psi_n_half[0]))
    elif float(psi_n) > float(psi_n_half[-1]):
        j0, j1 = ns - 2, ns - 1
        wh0 = (float(psi_n_half[-1]) - float(psi_n)) / (float(psi_n_half[-1]) - float(psi_n_half[-2]))
    elif float(psi_n) == float(psi_n_half[-1]):
        j0, j1, wh0 = ns - 2, ns - 1, 0.0
    else:
        j_for = max(int(math.floor(float(psi_n) * float(ns - 1) + 0.5)) + 1, 2)  # v3, 1-based
        j0 = j_for - 1
        j1 = j0 + 1
        wh0 = float(j_for) - float(psi_n) * float(ns - 1) - 0.5

    return VmecRadialInterpolation(
        index_full=(i0, i1),
        weight_full=(w0, 1.0 - w0),
        index_half=(j0, j1),
        weight_half=(float(wh0), float(1.0 - wh0)),
        psi_n=float(psi_n),
        psi_n_full=psi_n_full,
        psi_n_half=psi_n_half,
    )

def _scale_factors(*, m: np.ndarray, n_over_nfp: np.ndarray, helicity_n: int, helicity_l: int, ripple_scale: float) -> np.ndarray:
    """v3 ``setScaleFactor``: keep the quasisymmetric family, scale the rest by rippleScale."""
    n = np.asarray(n_over_nfp, dtype=np.int64)
    m = np.asarray(m, dtype=np.int64)
    if int(helicity_n) == 0:
        scaled = n != 0
    else:
        scaled = ((n != 0) & (n * int(helicity_l) != m * int(helicity_n))) | (n == 0)
    return np.where(scaled, float(ripple_scale), 1.0)

def _vmec_included_modes(
    *,
    w: VmecWout,
    interp: VmecRadialInterpolation,
    vmec_nyquist_option: int,
    min_bmn_to_load: float,
    ripple_scale: float,
    helicity_n: int,
    helicity_l: int,
    amplitude_table: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Nyquist-mode selection shared by ``from_vmec`` and the gpsipsi metric.

    Returns ``(idx, scale_all)``: the included indices into the Nyquist mode
    tables and the per-mode rippleScale factors.  Mirrors v3's
    ``min_Bmn_to_load`` filter (applied after scaling) and the
    ``VMEC_Nyquist_option=1`` truncation to ``|m| < mpol``, ``|n| <= ntor``.

    ``amplitude_table`` selects which ``|B|`` Fourier table drives the
    amplitude filter: ``bmnc`` (default) for the stellarator-symmetric modes and
    ``bmns`` for the stellarator-asymmetric modes.  The ``(0,0)`` reference
    ``b00`` is always taken from ``bmnc`` (v3's ``geometry.F90``).
    """
    (j0, j1) = interp.index_half
    (wh0, wh1) = interp.weight_half
    b00 = float(w.bmnc[0, j0] * wh0 + w.bmnc[0, j1] * wh1)
    if b00 == 0.0:
        raise ValueError("VMEC bmnc(0,0) is zero; cannot apply min_Bmn_to_load filter.")
    xm = np.asarray(w.xm_nyq, dtype=np.float64)
    xn = np.asarray(w.xn_nyq, dtype=np.float64)
    n_over_nfp = np.round(xn / float(w.nfp)).astype(np.int64)
    scale_all = _scale_factors(
        m=np.round(xm).astype(np.int64),
        n_over_nfp=n_over_nfp,
        helicity_n=int(helicity_n),
        helicity_l=int(helicity_l),
        ripple_scale=float(ripple_scale),
    )
    table = w.bmnc if amplitude_table is None else amplitude_table
    b_mode = table[:, j0] * wh0 + table[:, j1] * wh1
    include = np.abs((b_mode * scale_all) / b00) >= float(min_bmn_to_load)

    option = int(vmec_nyquist_option)
    if option == 0:
        option = 1  # early-prototype compatibility: treat 0 as the v3 default
    if option not in {1, 2}:
        raise ValueError("VMEC_Nyquist_option must be 1 (skip Nyquist) or 2 (include Nyquist).")
    if option == 1:
        include = include & (np.abs(xm) < float(w.mpol)) & (np.abs(xn / float(w.nfp)) <= float(w.ntor))
    idx = np.nonzero(include)[0].astype(np.int32)
    if idx.size == 0:
        raise ValueError("No VMEC modes were included (min_Bmn_to_load too large?).")
    return idx, scale_all
