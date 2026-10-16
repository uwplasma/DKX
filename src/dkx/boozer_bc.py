"""Boozer ``.bc`` equilibrium files (geometryScheme 11 and 12).

Plain-NumPy readers: header and surface parsing, a parse cache, scheme
detection, and the bracketing surfaces around a requested radius.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import IO
from typing import Iterator
import math

# The JAX backend is imported below; dkx/runtime.py explains why this is here.
from .runtime import configure as _configure_runtime

_configure_runtime()

import io  # noqa: E402
import numpy as np  # noqa: E402

from .paths import (  # noqa: E402
    resolve_existing_path,
)

_MU0 = 4.0 * math.pi * 1e-7

# =============================================================================
# Boozer .bc files (geometryScheme 11/12): reader
# =============================================================================

@dataclass(frozen=True)
class BoozerBcHeader:
    """Header of a ``.bc`` Boozer file, with v3 sign conventions applied."""

    n_periods: int
    psi_a_hat: float
    a_hat: float
    turkin_sign: int

@dataclass(frozen=True)
class BoozerBcSurface:
    """One flux surface of a ``.bc`` file: scalars plus the Fourier tables."""

    r_n: float
    iota: float
    g_hat: float
    i_hat: float
    p_prime_hat: float
    b0_over_bbar: float
    r0: float
    m: np.ndarray  # (H,) int32
    n: np.ndarray  # (H,) int32
    parity: np.ndarray  # (H,) bool, True=cos, False=sin
    b_amp: np.ndarray  # (H,) float64
    r_amp: np.ndarray  # (H,) float64
    z_amp: np.ndarray  # (H,) float64
    dz_amp: np.ndarray  # (H,) float64

    def __post_init__(self) -> None:
        # Instances are shared by a memoizing cache; read-only arrays stop one
        # caller's in-place edit from silently changing every later result.
        for value in vars(self).values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)

@dataclass(frozen=True)
class BoozerBcData:
    """A parsed Boozer file plus its detected v3 column convention."""

    geometry_scheme: int
    header: BoozerBcHeader
    surfaces: tuple[BoozerBcSurface, ...]

def _parse_bc_header_line(line: str) -> tuple[list[int], list[float]]:
    parts = line.split()
    if len(parts) < 6:
        raise ValueError(f"Unexpected .bc header line (too short): {line!r}")
    return [int(x) for x in parts[:4]], [float(x.replace("D", "E").replace("d", "E")) for x in parts[4:]]

def _try_parse_floats(tokens: list[str], count: int) -> list[float] | None:
    if len(tokens) < count:
        return None
    out: list[float] = []
    for t in tokens[:count]:
        try:
            out.append(float(t.replace("D", "E").replace("d", "E")))
        except ValueError:
            return None
    return out

def _try_parse_ints(tokens: list[str], count: int) -> list[int] | None:
    if len(tokens) < count:
        return None
    out: list[int] = []
    for t in tokens[:count]:
        try:
            out.append(int(t))
        except ValueError:
            return None
    return out

def _iter_bc_surfaces(
    *, fh: IO[str], geometry_scheme: int, n_periods: int, psi_a_hat: float
) -> Iterator[BoozerBcSurface]:
    """Yield surfaces in order, following v3's list-directed read patterns."""
    _ = fh.readline()  # v3 reads and discards one line after the header.
    while True:
        line = fh.readline()
        if not line:
            return
        if "s" not in line:  # scan to the next surface marker line
            continue
        surf_header: list[float] | None = None
        while surf_header is None:
            line = fh.readline()
            if not line:
                return
            surf_header = _try_parse_floats(line.split(), 5)
        s, iota, g_raw, i_raw, pprime_raw = surf_header[:5]
        # G and I pick up a minus sign from Ampere's law in the left-handed
        # (r, pol, tor) system; amperes -> normalized units via mu0/(2 pi).
        g_hat = -float(g_raw) * float(n_periods) / (2.0 * math.pi) * _MU0
        i_hat = -float(i_raw) / (2.0 * math.pi) * _MU0
        p_prime_hat = float(pprime_raw) / float(psi_a_hat) * _MU0

        _ = fh.readline()  # units line
        m_list: list[int] = []
        n_list: list[int] = []
        parity_list: list[bool] = []
        b_list: list[float] = []
        r_list: list[float] = []
        z_list: list[float] = []
        dz_list: list[float] = []
        b0_over_bbar = 0.0
        r0 = 0.0
        found_b00 = False

        while True:
            pos = fh.tell()
            line = fh.readline()
            if not line:
                break
            if "s" in line:
                fh.seek(pos)  # push back for the next surface
                break
            tokens = line.split()
            ij = _try_parse_ints(tokens, 2)
            if ij is None:
                continue
            m, n = int(ij[0]), int(ij[1])
            if geometry_scheme == 11:
                vals = _try_parse_floats(tokens[2:], 4)
                if vals is None:
                    continue
                rmn, zmn, dzmn, bmn = vals
                if m == 0 and n == 0:
                    b0_over_bbar, r0, found_b00 = float(bmn), float(rmn), True
                else:
                    m_list.append(m)
                    n_list.append(n)
                    parity_list.append(True)  # cosine-only format
                    r_list.append(float(rmn))
                    z_list.append(float(zmn))
                    dz_list.append(float(dzmn))
                    b_list.append(float(bmn))
            else:
                vals8 = _try_parse_floats(tokens[2:], 8)
                if vals8 is None:
                    continue
                if m == 0 and n == 0:
                    b0_over_bbar, r0, found_b00 = float(vals8[6]), float(vals8[0]), True
                else:
                    # Non-stellarator-symmetric format: each (m,n) expands into
                    # a cosine entry and a sine entry (v3 geometryScheme=12).
                    rcos, rsin, zcos, zsin, dzcos, dzsin, bcos, bsin = vals8
                    m_list += [m, m]
                    n_list += [n, n]
                    parity_list += [True, False]
                    r_list += [float(rcos), float(rsin)]
                    z_list += [float(zsin), float(zcos)]
                    dz_list += [float(dzsin), float(dzcos)]
                    b_list += [float(bcos), float(bsin)]

        if not found_b00:
            raise ValueError("No (0,0) mode found in Boozer .bc file surface block.")
        yield BoozerBcSurface(
            r_n=math.sqrt(float(s)),
            iota=float(iota),
            g_hat=g_hat,
            i_hat=i_hat,
            p_prime_hat=p_prime_hat,
            b0_over_bbar=float(b0_over_bbar),
            r0=float(r0),
            m=np.asarray(m_list, dtype=np.int32),
            n=np.asarray(n_list, dtype=np.int32),
            parity=np.asarray(parity_list, dtype=bool),
            b_amp=np.asarray(b_list, dtype=np.float64),
            r_amp=np.asarray(r_list, dtype=np.float64),
            z_amp=np.asarray(z_list, dtype=np.float64),
            dz_amp=np.asarray(dz_list, dtype=np.float64),
        )

# --- Vectorized .bc parsing --------------------------------------------------
#
# ``read_boozer_bc`` reads the whole file once: the Fortran ``d``/``D``
# exponent rewrite is applied to the full text with one ``str.translate``,
# line boundaries and the ``"s"`` surface-marker lines are located with one
# NumPy pass over the ASCII bytes, and each surface's numeric block is parsed
# by ``np.loadtxt`` (strict integer ``m``/``n`` columns, correctly-rounded
# float conversion, so values are bit-identical to ``float()``).  Any block
# that is not a plain uniform numeric table falls back to ``_bc_block_slow``
# (the same per-line rules as ``_iter_bc_surfaces``), and non-ASCII files fall
# back to ``_parse_bc_stream`` (the reference readline-based parser), so
# accepted inputs, results, and error behavior are unchanged.

_BC_D_TO_E = bytes.maketrans(b"Dd", b"EE")

# Bytes allowed in a fast-path numeric block once ``d/D -> E`` is applied.
# Anything else (letters, ``#``, ``_``, ...) routes the block to the
# reference parser, whose token-level accept/skip rules then decide.
_BC_NUMERIC_LUT = np.zeros(256, dtype=bool)

_BC_ROW_DTYPE = {
    11: np.dtype([("m", "i8"), ("n", "i8")] + [(f"c{k}", "f8") for k in range(4)]),
    12: np.dtype([("m", "i8"), ("n", "i8")] + [(f"c{k}", "f8") for k in range(8)]),
}

# One surface block: (found_b00, b0_over_bbar, r0, m, n, parity, b, r, z, dz).
_BcBlock = tuple[
    bool, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]

def _bc_block_slow(lines: list[str], *, geometry_scheme: int) -> _BcBlock:
    """Per-line parse of one surface's numeric block (v3 list-directed rules)."""
    m_list: list[int] = []
    n_list: list[int] = []
    parity_list: list[bool] = []
    b_list: list[float] = []
    r_list: list[float] = []
    z_list: list[float] = []
    dz_list: list[float] = []
    b0_over_bbar = 0.0
    r0 = 0.0
    found_b00 = False
    for line in lines:
        tokens = line.split()
        ij = _try_parse_ints(tokens, 2)
        if ij is None:
            continue
        m, n = int(ij[0]), int(ij[1])
        if geometry_scheme == 11:
            vals = _try_parse_floats(tokens[2:], 4)
            if vals is None:
                continue
            rmn, zmn, dzmn, bmn = vals
            if m == 0 and n == 0:
                b0_over_bbar, r0, found_b00 = float(bmn), float(rmn), True
            else:
                m_list.append(m)
                n_list.append(n)
                parity_list.append(True)  # cosine-only format
                r_list.append(float(rmn))
                z_list.append(float(zmn))
                dz_list.append(float(dzmn))
                b_list.append(float(bmn))
        else:
            vals8 = _try_parse_floats(tokens[2:], 8)
            if vals8 is None:
                continue
            if m == 0 and n == 0:
                b0_over_bbar, r0, found_b00 = float(vals8[6]), float(vals8[0]), True
            else:
                # Non-stellarator-symmetric format: each (m,n) expands into
                # a cosine entry and a sine entry (v3 geometryScheme=12).
                rcos, rsin, zcos, zsin, dzcos, dzsin, bcos, bsin = vals8
                m_list += [m, m]
                n_list += [n, n]
                parity_list += [True, False]
                r_list += [float(rcos), float(rsin)]
                z_list += [float(zsin), float(zcos)]
                dz_list += [float(dzsin), float(dzcos)]
                b_list += [float(bcos), float(bsin)]
    return (
        found_b00,
        b0_over_bbar,
        r0,
        np.asarray(m_list, dtype=np.int32),
        np.asarray(n_list, dtype=np.int32),
        np.asarray(parity_list, dtype=bool),
        np.asarray(b_list, dtype=np.float64),
        np.asarray(r_list, dtype=np.float64),
        np.asarray(z_list, dtype=np.float64),
        np.asarray(dz_list, dtype=np.float64),
    )

def _bc_block_from_rows(rows: np.ndarray, *, geometry_scheme: int) -> _BcBlock:
    """Vectorized equivalent of ``_bc_block_slow`` for a uniform numeric table."""
    m = rows["m"]
    n = rows["n"]
    is00 = (m == 0) & (n == 0)
    idx00 = np.flatnonzero(is00)
    found_b00 = idx00.size > 0
    keep = ~is00
    if geometry_scheme == 11:
        r, z, dz, b = rows["c0"], rows["c1"], rows["c2"], rows["c3"]
        b0_over_bbar, r0 = (float(b[idx00[-1]]), float(r[idx00[-1]])) if found_b00 else (0.0, 0.0)
        return (
            found_b00,
            b0_over_bbar,
            r0,
            m[keep].astype(np.int32),
            n[keep].astype(np.int32),
            np.ones(int(keep.sum()), dtype=bool),  # cosine-only format
            b[keep],
            r[keep],
            z[keep],
            dz[keep],
        )
    rcos, rsin, zcos, zsin = rows["c0"], rows["c1"], rows["c2"], rows["c3"]
    dzcos, dzsin, bcos, bsin = rows["c4"], rows["c5"], rows["c6"], rows["c7"]
    b0_over_bbar, r0 = (float(bcos[idx00[-1]]), float(rcos[idx00[-1]])) if found_b00 else (0.0, 0.0)
    # Non-stellarator-symmetric format: each (m,n) expands into a cosine and a
    # sine entry, with the z / dz sine coefficient listed first (see
    # ``_bc_block_slow``; v3 geometryScheme=12).
    return (
        found_b00,
        b0_over_bbar,
        r0,
        np.repeat(m[keep], 2).astype(np.int32),
        np.repeat(n[keep], 2).astype(np.int32),
        np.tile(np.array([True, False]), int(keep.sum())),
        np.column_stack([bcos[keep], bsin[keep]]).ravel(),
        np.column_stack([rcos[keep], rsin[keep]]).ravel(),
        np.column_stack([zsin[keep], zcos[keep]]).ravel(),
        np.column_stack([dzsin[keep], dzcos[keep]]).ravel(),
    )

def _parse_bc_block(
    text: str,
    arr: np.ndarray,
    line_starts: np.ndarray,
    line_ends: np.ndarray,
    i0: int,
    i1: int,
    *,
    geometry_scheme: int,
) -> _BcBlock:
    """Parse the numeric block spanning lines ``[i0, i1)`` of ``text``."""
    if i0 < i1:
        lo, hi = int(line_starts[i0]), int(line_ends[i1 - 1])
        seg = arr[lo:hi]
        # The all-whitespace check keeps ``np.loadtxt`` off empty blocks
        # (which it reports with a warning instead of an exception).
        if bool(_BC_NUMERIC_LUT[seg].all()) and bool((seg > 32).any()):
            try:
                rows = np.loadtxt(
                    io.StringIO(text[lo:hi]),
                    dtype=_BC_ROW_DTYPE[geometry_scheme],
                    comments=None,
                    ndmin=1,
                )
            except Exception:  # noqa: BLE001 - ragged/short rows: defer to the reference rules
                rows = None
            if rows is not None:
                return _bc_block_from_rows(rows, geometry_scheme=geometry_scheme)
    lines = [text[int(line_starts[j]) : int(line_ends[j])] for j in range(i0, i1)]
    return _bc_block_slow(lines, geometry_scheme=geometry_scheme)

def _parse_bc_text(
    text: str, buf: bytes, *, geometry_scheme: int, path: Path
) -> tuple[BoozerBcHeader, tuple[BoozerBcSurface, ...]]:
    """Vectorized parse of a whole ``.bc`` file (``d/D -> E`` already applied)."""
    arr = np.frombuffer(buf, dtype=np.uint8)
    newline_pos = np.flatnonzero(arr == 0x0A)
    line_starts = np.empty(newline_pos.size + 1, dtype=np.int64)
    line_starts[0] = 0
    line_starts[1:] = newline_pos + 1
    line_ends = np.empty(newline_pos.size + 1, dtype=np.int64)
    line_ends[: newline_pos.size] = newline_pos
    line_ends[newline_pos.size] = arr.size
    n_lines = newline_pos.size + 1
    if line_starts[n_lines - 1] == arr.size:
        n_lines -= 1  # file ends with a newline: no trailing partial line
    # Lines containing "s" are the surface markers v3 scans for.  ``bytes.find``
    # (memchr) beats a full-array comparison here: markers are rare.
    s_pos = []
    j = buf.find(b"s")
    while j != -1:
        s_pos.append(j)
        j = buf.find(b"s", j + 1)
    s_lines = np.unique(newline_pos.searchsorted(np.asarray(s_pos, dtype=np.int64), side="left"))

    turkin_sign = 1
    i = 0
    while True:
        if i >= n_lines:
            raise ValueError(f"Unexpected EOF while reading {str(path)!r}")
        line = text[line_starts[i] : line_ends[i]]
        i += 1
        if line.startswith("CC"):
            # v3: this substring marks files saved by Yu. Turkin's
            # CStconfig, which use an extra sign convention.
            if geometry_scheme == 11 and "CStconfig" in line:
                turkin_sign = -1
            continue
        try:
            header_ints, header_reals = _parse_bc_header_line(line)
        except Exception:  # noqa: BLE001 - column-name lines precede the numeric header
            continue
        n_periods = int(header_ints[3])
        psi_a_hat = float(header_reals[0]) / (2.0 * math.pi)
        a_hat = float(header_reals[1])
        break
    # Left-handed -> right-handed (radial, poloidal, toroidal):
    if geometry_scheme == 11:
        psi_a_hat = psi_a_hat * (-1.0) * float(turkin_sign)
    else:
        psi_a_hat = psi_a_hat * (-1.0)
    header = BoozerBcHeader(n_periods=n_periods, psi_a_hat=psi_a_hat, a_hat=a_hat, turkin_sign=turkin_sign)

    surfaces: list[BoozerBcSurface] = []
    i += 1  # v3 reads and discards one line after the header.
    while True:
        k = int(np.searchsorted(s_lines, i, side="left"))  # scan to the next surface marker line
        if k >= s_lines.size:
            break
        i = int(s_lines[k]) + 1
        surf_header: list[float] | None = None
        while surf_header is None:
            if i >= n_lines:
                return header, tuple(surfaces)
            surf_header = _try_parse_floats(text[line_starts[i] : line_ends[i]].split(), 5)
            i += 1
        s, iota, g_raw, i_raw, pprime_raw = surf_header[:5]
        # G and I pick up a minus sign from Ampere's law in the left-handed
        # (r, pol, tor) system; amperes -> normalized units via mu0/(2 pi).
        g_hat = -float(g_raw) * float(n_periods) / (2.0 * math.pi) * _MU0
        i_hat = -float(i_raw) / (2.0 * math.pi) * _MU0
        p_prime_hat = float(pprime_raw) / float(psi_a_hat) * _MU0
        i += 1  # units line
        k = int(np.searchsorted(s_lines, i, side="left"))
        block_end = int(s_lines[k]) if k < s_lines.size else n_lines
        found_b00, b0_over_bbar, r0, m_arr, n_arr, parity, b_amp, r_amp, z_amp, dz_amp = _parse_bc_block(
            text, arr, line_starts, line_ends, i, block_end, geometry_scheme=geometry_scheme
        )
        if not found_b00:
            raise ValueError("No (0,0) mode found in Boozer .bc file surface block.")
        surfaces.append(
            BoozerBcSurface(
                r_n=math.sqrt(float(s)),
                iota=float(iota),
                g_hat=g_hat,
                i_hat=i_hat,
                p_prime_hat=p_prime_hat,
                b0_over_bbar=float(b0_over_bbar),
                r0=float(r0),
                m=m_arr,
                n=n_arr,
                parity=parity,
                b_amp=b_amp,
                r_amp=r_amp,
                z_amp=z_amp,
                dz_amp=dz_amp,
            )
        )
        i = block_end
    return header, tuple(surfaces)

def _parse_bc_stream(
    fh: IO[str], *, geometry_scheme: int, path: Path
) -> tuple[BoozerBcHeader, tuple[BoozerBcSurface, ...]]:
    """Reference readline-based parse, kept for non-ASCII ``.bc`` files."""
    turkin_sign = 1
    while True:
        line = fh.readline()
        if not line:
            raise ValueError(f"Unexpected EOF while reading {str(path)!r}")
        if line.startswith("CC"):
            # v3: this substring marks files saved by Yu. Turkin's
            # CStconfig, which use an extra sign convention.
            if geometry_scheme == 11 and "CStconfig" in line:
                turkin_sign = -1
            continue
        try:
            header_ints, header_reals = _parse_bc_header_line(line)
        except Exception:  # noqa: BLE001 - column-name lines precede the numeric header
            continue
        n_periods = int(header_ints[3])
        psi_a_hat = float(header_reals[0]) / (2.0 * math.pi)
        a_hat = float(header_reals[1])
        break
    # Left-handed -> right-handed (radial, poloidal, toroidal):
    if geometry_scheme == 11:
        psi_a_hat = psi_a_hat * (-1.0) * float(turkin_sign)
    else:
        psi_a_hat = psi_a_hat * (-1.0)
    header = BoozerBcHeader(n_periods=n_periods, psi_a_hat=psi_a_hat, a_hat=a_hat, turkin_sign=turkin_sign)
    surfaces = tuple(
        _iter_bc_surfaces(fh=fh, geometry_scheme=geometry_scheme, n_periods=n_periods, psi_a_hat=psi_a_hat)
    )
    return header, surfaces

# Memoized .bc parses keyed by file identity (device/inode/mtime/size) and
# geometry scheme.  Parsing a large .bc text file is a leading host-side cost
# of a cold start, and the same equilibrium is read by grid construction,
# geometry selection, and the writer metric derivation within one run (and
# across surfaces in scans), so the parsed header/surface tables are shared.  Entries are treated as immutable
# by every consumer; the file-identity key invalidates edited files.
_BC_PARSE_CACHE: dict[
    tuple[int, int, int, int, int], tuple[BoozerBcHeader, tuple[BoozerBcSurface, ...]]
] = {}

_BC_PARSE_CACHE_MAX = 4

def read_boozer_bc(
    path: str | Path, *, geometry_scheme: int
) -> tuple[BoozerBcHeader, tuple[BoozerBcSurface, ...]]:
    """Read a v3 ``.bc`` Boozer equilibrium file (geometryScheme 11/12).

    Pure function returning plain NumPy arrays; the sign switches applied in
    v3 ``geometry.F90`` (left- to right-handed system, and the ``CStconfig``
    files that need an extra flip for scheme 11) are folded into the header
    ``psi_a_hat``.  Not differentiable; downstream construction is.  Parses are
    memoized on file identity (device/inode/mtime/size), so repeated reads of
    an unchanged equilibrium return the shared parsed tables; treat the
    returned header/surfaces as immutable.
    """
    if geometry_scheme not in {11, 12}:
        raise ValueError(f"geometry_scheme must be 11 or 12, got {geometry_scheme}")
    p = Path(path).expanduser()
    if not p.exists():
        p = resolve_existing_path(p).path

    st = p.stat()
    cache_key = (
        int(st.st_dev), int(st.st_ino), int(st.st_mtime_ns), int(st.st_size), int(geometry_scheme),
    )  # fmt: skip
    cached = _BC_PARSE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        raw = p.read_bytes()
    except OSError as exc:
        raise ValueError(f"Could not read Boozer equilibrium {str(p)!r}: {exc}") from exc
    header, surfaces = _parse_boozer_bytes(
        raw, geometry_scheme=geometry_scheme, path=p
    )
    while len(_BC_PARSE_CACHE) >= _BC_PARSE_CACHE_MAX:
        _BC_PARSE_CACHE.pop(next(iter(_BC_PARSE_CACHE)))
    _BC_PARSE_CACHE[cache_key] = (header, surfaces)
    return header, surfaces

def _parse_boozer_bytes(
    raw: bytes, *, geometry_scheme: int, path: Path
) -> tuple[BoozerBcHeader, tuple[BoozerBcSurface, ...]]:
    """Parse bytes already read from a Boozer file without reopening it."""
    if b"\r" in raw:  # universal-newline translation, as text-mode reads apply
        raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    if b"d" in raw or b"D" in raw:
        raw = raw.translate(_BC_D_TO_E)  # Fortran d/D exponents -> E, once for the whole file
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError:
        # Non-ASCII comments do not affect the list-directed numeric grammar.
        # Preserve the reference stream parser without reopening the file.
        text = raw.decode("utf-8", errors="replace")
        return _parse_bc_stream(
            io.StringIO(text), geometry_scheme=geometry_scheme, path=path
        )
    return _parse_bc_text(text, raw, geometry_scheme=geometry_scheme, path=path)

def _detect_boozer_geometry_scheme(text: str, *, path: Path) -> int:
    """Detect cosine-only scheme 11 versus asymmetric scheme 12 columns."""
    header_seen = False
    surface_seen = False
    surface_header_seen = False
    skip_units = False
    for line in text.splitlines():
        tokens = line.split()
        if not header_seen:
            try:
                _parse_bc_header_line(line)
            except Exception:  # noqa: BLE001 - headings precede the numeric header
                continue
            header_seen = True
            continue
        if not surface_seen:
            if "s" in line:
                surface_seen = True
            continue
        if not surface_header_seen:
            if _try_parse_floats(tokens, 5) is not None:
                surface_header_seen = True
                skip_units = True
            continue
        if skip_units:
            skip_units = False
            continue
        if _try_parse_ints(tokens, 2) is None:
            continue
        values = tokens[2:]
        if len(values) >= 8 and _try_parse_floats(values, 8) is not None:
            return 12
        if len(values) >= 4 and _try_parse_floats(values, 4) is not None:
            return 11
    raise ValueError(
        f"Could not detect Boozer .bc column convention in {str(path)!r}; "
        "expected a scheme-11 six-column or scheme-12 ten-column Fourier row."
    )

def read_native_boozer(path: str | Path) -> BoozerBcData:
    """Read a native Boozer source with its column convention auto-detected.

    Native ``Case`` inputs name a physical format rather than a SFINCS geometry
    scheme. Detection is therefore confined to this file boundary; the parsed
    immutable tables are reused for every requested surface.
    """
    resolved = Path(path).expanduser()
    if not resolved.exists():
        resolved = resolve_existing_path(resolved).path
    try:
        raw = resolved.read_bytes()
    except OSError as exc:
        raise ValueError(
            f"Could not read Boozer equilibrium {str(resolved)!r}: {exc}"
        ) from exc
    text = raw.decode("ascii", errors="replace")
    geometry_scheme = _detect_boozer_geometry_scheme(text, path=resolved)
    header, surfaces = _parse_boozer_bytes(
        raw, geometry_scheme=geometry_scheme, path=resolved
    )
    if len(surfaces) < 2:
        raise ValueError(
            f"Boozer equilibrium {str(resolved)!r} has {len(surfaces)} surface(s); "
            "at least two are required for radial derivatives."
        )
    return BoozerBcData(
        geometry_scheme=geometry_scheme, header=header, surfaces=surfaces
    )

def _bracketing_surfaces(
    surfaces: tuple[BoozerBcSurface, ...], r_n_wish: float
) -> tuple[BoozerBcSurface, BoozerBcSurface]:
    """Return the pair of surfaces bracketing ``r_n_wish`` (v3 selection order)."""
    old: BoozerBcSurface | None = None
    new: BoozerBcSurface | None = None
    for s in surfaces:
        if new is None:
            new = s
        if new.r_n < float(r_n_wish):
            old = new
            new = None
            continue
        if old is None:
            old = s
        new = s
        break
    if old is None or new is None:
        raise ValueError(f"Failed to locate surfaces bracketing rN_wish={r_n_wish}")
    return old, new

def selected_r_n_from_bc(
    *, path: str | Path, geometry_scheme: int, r_n_wish: float, vmec_radial_option: int = 1
) -> float:
    """Effective radius used by v3 for geometryScheme 11/12 output metadata.

    ``vmec_radial_option=1`` snaps to the nearest surface (v3 tie-breaking);
    otherwise v3 interpolates linearly in ``s = rN^2`` and this returns the
    corresponding effective ``rN``.
    """
    _header, surfaces = read_boozer_bc(path, geometry_scheme=int(geometry_scheme))
    surf_old, surf_new = _bracketing_surfaces(surfaces, float(r_n_wish))
    r_old, r_new = float(surf_old.r_n), float(surf_new.r_n)
    if r_new == r_old:
        return r_old
    if int(vmec_radial_option) == 1:
        return r_old if abs(r_old - float(r_n_wish)) < abs(r_new - float(r_n_wish)) else r_new
    s_old, s_new, s_wish = r_old * r_old, r_new * r_new, float(r_n_wish) ** 2
    radial_weight = (s_new - s_wish) / (s_new - s_old)
    return float(math.sqrt(max(radial_weight * s_old + (1.0 - radial_weight) * s_new, 0.0)))
