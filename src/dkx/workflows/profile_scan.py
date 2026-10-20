"""Radius x E_r scans of a SFINCS deck driven by a profile file.

This is the namelist-route equivalent of upstream ``sfincsScan`` with
``scanType = 5``: the deck's ``!ss`` comment lines choose the radii, a profile
file gives ``n(rN)``, ``T(rN)`` and the ``E_r`` range, and every radius is an
ordinary E_r scan over its ``E_r`` values. :func:`scan` dispatches a deck to the
E_r or the radius x E_r scan from its ``!ss`` lines.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np

from .scans import EmitFn, _patch_scalar_in_group, run_er_scan

_SS_LINE = re.compile(r"^\s*!ss\s+(\w+)\s*=\s*(.*?)\s*$", re.IGNORECASE)
_STRING_KEYS = {"scanvariable", "scanvariablescale"}


class ScanInputError(ValueError):
    """A deck, ``!ss`` line or profile file that cannot be read; the message names the line."""


@dataclass(frozen=True)
class ProfileScanResult:
    scan_dir: Path
    radii: tuple[float, ...]
    n_points: int
    failures: tuple[tuple[float, float, str], ...] = ()


def _number(text: str) -> float:
    """A Fortran real: ``d`` exponents, a trailing ``!`` comment and a Fortran-style ``.`` are fine."""
    return float(text.split("!")[0].strip().lower().replace("d", "e"))


def parse_ss_lines(text: str, *, source: str = "input.namelist") -> dict[str, float | str]:
    """The ``!ss key = value`` lines of a deck, keyed by lower-case name.

    A numeric value that does not parse raises ``ValueError`` naming the file,
    line number and the line itself.
    """
    out: dict[str, float | str] = {}
    for lineno, line in enumerate(text.splitlines(), start=1):
        match = _SS_LINE.match(line)
        if match is None:
            continue
        key, value = match.group(1).lower(), match.group(2)
        if key in _STRING_KEYS:
            out[key] = value.split("!")[0].strip()
            continue
        try:
            out[key] = _number(value)
        except ValueError:
            raise ScanInputError(
                f"{source}:{lineno}: cannot read a number from {line.strip()!r}; "
                f"{value!r} is not a Fortran real (for example 20.0d+0)."
            ) from None
    return out


def read_profiles(path: Path) -> dict[str, np.ndarray]:
    """Read a ``profile_*`` file: ``rN, nErs, Ermin, Ermax, n_i, T_i, n_e, T_e`` per radius.

    The first data line is the radial coordinate (``1`` psiN, ``3`` rN); ``#`` lines
    are comments. ``n`` is in 1e20 m^-3 and ``T`` in keV.
    """
    rows: list[list[float]] = []
    coordinate: int | None = None
    for lineno, line in enumerate(Path(path).read_text().splitlines(), start=1):
        body = line.split("#")[0].strip()
        if not body:
            continue
        try:
            values = [_number(token) for token in body.split()]
        except ValueError:
            raise ScanInputError(f"{path}:{lineno}: cannot read numbers from {line.strip()!r}") from None
        if coordinate is None:
            if len(values) != 1 or values[0] not in (1, 3):
                raise ScanInputError(
                    f"{path}:{lineno}: the first data line must be the radial coordinate, "
                    f"1 (psiN) or 3 (rN); got {line.strip()!r}"
                )
            coordinate = int(values[0])
        elif len(values) != 8:
            raise ScanInputError(f"{path}:{lineno}: expected 8 columns (rN nErs Ermin Ermax n_i T_i n_e T_e), got {len(values)}")
        else:
            rows.append(values)
    if coordinate is None or len(rows) < 2:
        raise ScanInputError(f"{path}: need a radial coordinate line and at least two radii")
    table = np.asarray(rows, dtype=np.float64)
    r_n = np.sqrt(table[:, 0]) if coordinate == 1 else table[:, 0]
    if np.any(np.diff(r_n) <= 0):
        raise ScanInputError(f"{path}: the radial coordinate must be strictly increasing")
    names = ("n_er", "er_min", "er_max", "n_i", "t_i", "n_e", "t_e")
    return {"r_n": r_n, **{name: table[:, 1 + i] for i, name in enumerate(names)}}


def _radius_deck(txt: str, profiles: dict[str, np.ndarray], r_n: float, a_hat: float, zs: Sequence[float]) -> str:
    """The deck at ``rN = r_n`` with ``n``, ``T`` and their rHat gradients from the profiles."""
    grid = profiles["r_n"]
    out = _patch_scalar_in_group(txt=txt, group="geometryParameters", key="rN_wish", value=r_n)
    values: dict[str, list[float]] = {"nHats": [], "THats": [], "dNHatdrHats": [], "dTHatdrHats": []}
    for z in zs:
        ion = z > 0
        for key, name in (("nHats", "n_i" if ion else "n_e"), ("THats", "t_i" if ion else "t_e")):
            column = profiles[name]
            values[key].append(float(np.interp(r_n, grid, column)))
            d_key = "dNHatdrHats" if key == "nHats" else "dTHatdrHats"
            values[d_key].append(float(np.interp(r_n, grid, np.gradient(column, grid))) / a_hat)
    for key, vals in values.items():
        out = _patch_scalar_in_group(txt=out, group="speciesParameters", key=key, value=" ".join(f"{v:.16g}" for v in vals))
    return out


def run_profile_scan(
    *,
    input_namelist: Path,
    out_dir: Path,
    profiles: Path | None = None,
    radii: Sequence[float] | None = None,
    er_range: tuple[float, float, int] | None = None,
    skip_existing: bool = False,
    solve_method: str = "auto",
    jobs: int | None = None,
    emit: EmitFn | None = None,
    scan_er: Callable = run_er_scan,
) -> ProfileScanResult:
    """Solve every ``(rN, E_r)`` point; a failed point is recorded and the scan goes on.

    ``radii`` defaults to the deck's ``!ss Nradius / rN_min / rN_max``, and the ``E_r``
    values at each radius to the profile file's ``nErs / Ermin / Ermax`` columns
    interpolated there; ``er_range = (min, max, n)`` replaces them.
    """
    from ..drift_kinetic import kinetic_operator_build_from_namelist  # noqa: PLC0415
    from ..inputs import parse_sfincs_input_text  # noqa: PLC0415
    from ..namelist import read_sfincs_input  # noqa: PLC0415

    input_namelist = Path(input_namelist).resolve()
    out_dir = Path(out_dir).resolve()
    txt = input_namelist.read_text()
    ss = parse_ss_lines(txt, source=str(input_namelist))
    single = ss.get("scantype", 5) == 4
    if ss.get("scantype", 5) not in (4, 5):
        raise ScanInputError(f"{input_namelist}: !ss scanType = {ss['scantype']:g}; the radius scans run scanType 4 and 5")
    nml = read_sfincs_input(input_namelist)
    geometry, species = nml.group("geometryParameters"), nml.group("speciesParameters")
    if int(geometry.get("INPUTRADIALCOORDINATE", 3)) != 3 or "DNHATDRHATS" not in species or "DTHATDRHATS" not in species:
        raise ScanInputError(
            f"{input_namelist}: scan-profile needs inputRadialCoordinate = 3 and species gradients "
            "given as dNHatdrHats and dTHatdrHats (with Er); it replaces rN_wish, nHats, THats and those gradients."
        )
    if profiles is None:
        found = sorted(input_namelist.parent.glob("profile*"))
        if len(found) != 1:
            raise ScanInputError(f"pass --profiles: found {len(found)} files named profile* beside {input_namelist}")
        profiles = found[0]
    table = read_profiles(Path(profiles))
    if radii is None:
        radii = (
            np.linspace(ss["rn_min"], ss["rn_max"], int(ss["nradius"]))
            if {"nradius", "rn_min", "rn_max"} <= ss.keys()
            else [float(geometry["RN_WISH"])]
        )
    outside = [float(r) for r in radii if not table["r_n"][0] <= r <= table["r_n"][-1]]
    if outside:  # np.interp would hold the edge value, and the gradient with it
        raise ScanInputError(
            f"radii {outside} lie outside the profile file's rN range [{table['r_n'][0]:g}, {table['r_n'][-1]:g}]"
        )
    zs = [float(z) for z in np.atleast_1d(species["ZS"])]
    if sum(z > 0 for z in zs) > 1 or sum(z < 0 for z in zs) > 1:
        raise ScanInputError("scan-profile maps one ion and one electron profile; the deck has several ions or electrons")
    a_hat = float(kinetic_operator_build_from_namelist(parse_sfincs_input_text(txt, source_path=input_namelist)).radial.a_hat)

    failures: list[tuple[float, float, str]] = []
    n_points = 0
    for r_n in map(float, radii):
        if er_range is None:
            n_er = max(2, round(np.interp(r_n, table["r_n"], table["n_er"])))
            er_min, er_max = (float(np.interp(r_n, table["r_n"], table[k])) for k in ("er_min", "er_max"))
        else:
            er_min, er_max, n_er = er_range
        values = np.linspace(er_min, er_max, int(n_er))
        run_dir = _radius_dir(out_dir, r_n)
        run_dir.mkdir(parents=True, exist_ok=True)
        deck = run_dir / "input.namelist"
        deck.write_text(_radius_deck(txt, table, r_n, a_hat, zs))
        if single:  # scanType 4: one solve per radius at the deck's own E_r
            n_points += 1
            point_failures = _solve_points(out_dir, [(run_dir.name, deck.read_text())], skip_existing, solve_method, emit)
            failures += [(r_n, float("nan"), why) for _, why in point_failures]
            continue
        if emit is not None:
            emit(0, f"scan-profile: rN={r_n:.4g} {len(values)} E_r points in [{er_min:.4g}, {er_max:.4g}]")
        result = scan_er(
            input_namelist=deck, out_dir=run_dir, values=[float(v) for v in values], compute_solution=True,
            skip_existing=skip_existing, solve_method=solve_method, jobs=jobs, emit=emit,
        )
        n_points += len(values)
        failures += [(r_n, v, reason) for v, reason in result.failures]
    report = out_dir / "scan_failures.json"
    if failures:
        rows = [{"rN": r, **({} if np.isnan(v) else {"Er": v}), "reason": why} for r, v, why in failures]
        report.write_text(json.dumps(rows, indent=2) + "\n")
    else:
        report.unlink(missing_ok=True)
    return ProfileScanResult(out_dir, tuple(map(float, radii)), n_points, tuple(failures))


def _radius_dir(out_dir: Path, r_n: float) -> Path:
    """``rN_<r>`` as upstream ``sfincsScan`` names it; an older ``rN<r>`` folder is reused."""
    legacy = out_dir / f"rN{r_n:.4g}"
    return legacy if legacy.is_dir() else out_dir / f"rN_{r_n:.4g}"


def _solve_points(out_dir, points, skip_existing, solve_method, emit):
    """Solve ``(folder, deck text)`` points; a failure is recorded, not raised. Returns ``(folder, reason)`` failures."""
    from ..io import localize_equilibrium_file_in_place  # noqa: PLC0415
    from ..run import run_from_namelist  # noqa: PLC0415

    failures = []
    for i, (folder, text) in enumerate(points, start=1):
        run_dir = out_dir / folder
        run_dir.mkdir(parents=True, exist_ok=True)
        out_path = run_dir / "sfincsOutput.h5"
        if skip_existing and out_path.exists():
            continue
        if emit is not None:
            emit(0, f"scan: [{i}/{len(points)}] {folder}")
        deck = run_dir / "input.namelist"
        deck.write_text(text)
        try:
            localize_equilibrium_file_in_place(input_namelist=deck, overwrite=False)
            run_from_namelist(deck, out_path=out_path, overwrite=True, solve_method=solve_method)
        except Exception as exc:  # noqa: BLE001 - one bad point must not discard the others
            failures.append((folder, f"{type(exc).__name__}: {exc}"))
            if emit is not None:
                emit(0, f"scan: {folder} FAILED: {failures[-1][1]}")
    return failures


def _set_in_deck(txt: str, key: str, value) -> str:
    """Set ``key`` in whichever namelist group already holds it (``physicsParameters`` otherwise)."""
    for match in re.finditer(r"(?im)^\s*&(\w+)\s*$(.*?)^\s*/\s*$", txt, re.DOTALL | re.MULTILINE):
        if re.search(rf"(?im)^[ \t]*{re.escape(key)}[ \t]*=", match.group(2)):
            return _patch_scalar_in_group(txt=txt, group=match.group(1), key=key, value=value)
    return _patch_scalar_in_group(txt=txt, group="physicsParameters", key=key, value=value)


def _space(low: float, high: float, n: int, log: bool) -> list[float]:
    if n < 1:
        return []
    if n == 1:
        return [low]
    return list(np.geomspace(low, high, n) if log else np.linspace(low, high, n))


# upstream sfincsScan_1: key, default, integer, odd
_RESOLUTION = (("Ntheta", 15, True, True), ("Nzeta", 15, True, True), ("Nxi", 16, True, False), ("NL", 4, True, False),
               ("Nx", 5, True, False), ("NxPotentialsPerVth", 40.0, False, False), ("xMax", 5.0, False, False),
               ("solverTolerance", 1e-6, False, False))


def run_convergence_scan(txt: str, ss: dict, base: dict, **run) -> list[tuple[str, str]]:
    """``!ss scanType = 1``: ``baseCase`` plus one folder per ``<key>MinFactor/MaxFactor/NumRuns`` value, one key at a time."""
    points = [("baseCase", txt)]
    for key, default, integer, odd in _RESOLUTION:
        n = int(ss.get(f"{key.lower()}numruns", 0))
        if n < 1:
            continue
        lo, hi = ss.get(f"{key.lower()}minfactor"), ss.get(f"{key.lower()}maxfactor")
        if lo is None or hi is None or key.lower() not in base:
            raise ScanInputError(f"scanning {key} needs its base value and !ss {key}MinFactor, {key}MaxFactor and {key}NumRuns")
        b = float(base[key.lower()])
        values = _space(lo * b, hi * b, n, log=True)
        if integer:  # upstream: odd grids truncate and round up to odd; others round; duplicates drop
            values = [(int(v) | 1) if odd else round(v) for v in values]
            values = list(dict.fromkeys(values))
        points += [
            (f"{key}{v:.4g}", _set_in_deck(txt, key, v))
            for v in values if abs(v - b) / abs(v) >= 1e-6
        ]
    return points


def run_variable_scan(txt: str, ss: dict) -> list[tuple[str, str]]:
    """``!ss scanType = 3``: ``scanVariable`` over ``scanVariableMin/Max/N`` (``scanVariableScale`` linear or log), folders ``<name>_<value>``."""
    name = str(ss.get("scanvariable", "")).strip("'\" ")
    scale = str(ss.get("scanvariablescale", "")).strip("'\" ").lower()
    if not name or scale not in ("lin", "linear", "log", "logarithmic"):
        raise ScanInputError("scanType 3 needs !ss scanVariable and scanVariableScale = linear or log")
    try:
        low, high, n = ss["scanvariablemin"], ss["scanvariablemax"], int(ss["scanvariablen"])
    except KeyError as exc:
        raise ScanInputError(f"scanType 3 needs !ss scanVariableMin, scanVariableMax and scanVariableN (missing {exc})") from None
    log = scale.startswith("log")
    if log and low * high < 0:
        raise ScanInputError("scanVariableMin and scanVariableMax must have the same sign for a logarithmic scan")
    values = _space(abs(low), abs(high), n, log) if log else _space(low, high, n, False)
    values = [-v for v in values] if log and low < 0 else values
    if not re.search(rf"(?im)^[ \t]*{re.escape(name)}[ \t]*=", txt):
        raise ScanInputError(f"scanVariable {name} must be assigned in the namelist, even though the value is ignored")
    if re.search(rf"(?im)^[ \t]*{re.escape(name)}[ \t]*=[ \t]*[+-]?\d+[ \t]*(!.*)?$", txt):
        values = list(dict.fromkeys(int(v) for v in values))  # an integer variable
    return [(f"{name}_{v:.4g}", _set_in_deck(txt, name, v)) for v in values]


def run_deck_scan(namelist: Path, out_dir: Path, kind: int, *, skip_existing: bool = False, solve_method: str = "auto", emit=None):
    """Convergence (1) and generic-variable (3) scans of one deck."""
    from ..namelist import read_sfincs_input  # noqa: PLC0415

    txt = Path(namelist).read_text()
    ss = parse_ss_lines(txt, source=str(namelist))
    if kind == 3:
        points = run_variable_scan(txt, ss)
    else:
        nml = read_sfincs_input(namelist)
        base = {k.lower(): v for group in ("resolutionParameters", "physicsParameters", "general") for k, v in nml.group(group).items()}
        points = run_convergence_scan(txt, ss, base)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    failures = _solve_points(out_dir, points, skip_existing, solve_method, emit)
    report = out_dir / "scan_failures.json"
    if failures:
        report.write_text(json.dumps([{"point": f, "reason": why} for f, why in failures], indent=2) + "\n")
    else:
        report.unlink(missing_ok=True)
    return ProfileScanResult(out_dir, (), len(points), tuple((float("nan"), float("nan"), f"{f}: {why}") for f, why in failures))


def scan(
    namelist: Path | str = "input.namelist",
    *,
    out_dir: Path | str | None = None,
    profiles: Path | str | None = None,
    er: tuple[float, float, int] | None = None,
    radii: Sequence[float] | None = None,
    **kwargs,
):
    """Run the scan a deck's ``!ss`` lines describe and write the ``sfincsScan`` folder tree.

    ``!ss scanType = 2`` solves ``NErs`` values between ``ErMin`` and ``ErMax`` (or the
    ``dPhiHatd*`` bounds the deck's gradient coordinate selects) into ``Er<value>/``
    folders; ``scanType = 5``, or any ``profiles`` file, solves ``rN_<radius>/Er<value>/``
    from the profile file. ``er = (min, max, n)`` and ``radii`` override the deck.
    ``kwargs`` are ``skip_existing``, ``solve_method``, ``jobs`` and ``emit(level, msg)``.
    Returns a :class:`ProfileScanResult` or a ``ScanResult``; both carry ``failures``.
    ``scanType = 1`` (convergence, ``baseCase`` plus ``<key><value>/``), ``3`` (one variable,
    ``<name>_<value>/``) and ``4`` (``rN_<radius>/``, one solve each) follow upstream's folders.
    """
    from ..namelist import read_sfincs_input  # noqa: PLC0415
    from .scans import _er_scan_var_name  # noqa: PLC0415

    namelist = Path(namelist).resolve()
    out_dir = namelist.parent if out_dir is None else Path(out_dir)
    ss = parse_ss_lines(namelist.read_text(), source=str(namelist))
    kind = ss.get("scantype", 5 if profiles is not None else 2)
    if kind in (4, 5):
        return run_profile_scan(
            input_namelist=namelist, out_dir=out_dir, profiles=None if profiles is None else Path(profiles),
            radii=radii, er_range=er, **kwargs,
        )
    if kind in (1, 3):
        return run_deck_scan(namelist, out_dir, int(kind), **kwargs)
    if kind != 2:
        raise ScanInputError(f"{namelist}: !ss scanType = {kind:g} is not an sfincsScan type (1 to 5)")
    var = _er_scan_var_name(nml=read_sfincs_input(namelist)).lower()
    try:
        low, high, n = er if er is not None else (ss[f"{var}min"], ss[f"{var}max"], int(ss["ners"]))
    except KeyError as exc:
        raise ScanInputError(f"{namelist}: an E_r scan needs the !ss lines NErs, {var}Min and {var}Max (missing {exc})") from None
    return run_er_scan(
        input_namelist=namelist, out_dir=out_dir, values=np.linspace(low, high, int(n)).tolist(),
        compute_solution=True, **kwargs,
    )
