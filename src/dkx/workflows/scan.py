"""Declarative parameter scans over a case.

``[scan]`` has been in the case schema since the native path existed --
parsed, validated, and then refused by ``execution.run_case`` with a message
pointing at a ``dkx.scan`` that did not exist. This is that module.

A scan expands the axes into derived cases, runs each one, and writes a single
``Result`` whose arrays carry a leading ``case`` dimension. The axis values are
written alongside the observables so the output is self-describing: reading it
back tells you what was varied without also needing the case file.

Two behaviours are deliberate and are pinned by tests.

A case that fails does not abandon the ones that already succeeded. The run
records the failure for that point, continues, and reports a non-zero exit at
the end. Losing forty completed solves because the forty-first hit an
unsupported combination is the wrong trade for a scan.

``resume`` compares deterministic ``case_id`` values, not positions. An
interrupted scan resumed after the case file was edited must not silently
graft new physics onto old rows, so a changed case simply misses the cache and
runs again.
"""

from __future__ import annotations

from dataclasses import replace
import itertools
from pathlib import Path
from typing import Any, Callable

import numpy as np

#: Observables carried through from each case, in the order they are reported.
SCAN_OBSERVABLES: tuple[str, ...] = (
    "particle_flux_m2_s",
    "heat_flux_W_m2",
    "parallel_current_A_T_m2",
)


def _axis_points(scan) -> list[tuple[float, ...]]:
    """One tuple of axis values per derived case.

    ``cartesian`` is every combination; ``zipped`` walks the axes together.
    The schema already guarantees at least one axis and, for ``zipped``,
    that the lengths agree.
    """
    values = [axis.values for axis in scan.axes]
    if scan.combine == "cartesian":
        return [tuple(point) for point in itertools.product(*values)]
    return [tuple(column) for column in zip(*values, strict=True)]


def _apply_axis(case, path: str, value: float):
    """Return ``case`` with the single schema path ``path`` set to ``value``.

    Only the paths ``config._is_supported_scan_path`` admits are reachable
    here; anything else has already been refused at validation time, so an
    unrecognised path is a programming error rather than user input.
    """
    if path == "electric_field.value_kV_m":
        return replace(case, electric_field=replace(case.electric_field, value_kV_m=float(value)))
    if path.startswith("resolution."):
        field = path.split(".", 1)[1]
        return replace(case, resolution=replace(case.resolution, **{field: int(value)}))
    if path.startswith("solver."):
        field = path.split(".", 1)[1]
        return replace(case, solver=replace(case.solver, **{field: float(value)}))
    if path.startswith("species["):
        name, _, attribute = path.partition("].")
        name = name[len("species["):]
        scaled = []
        for species in case.species:
            if species.name != name:
                scaled.append(species)
            elif attribute == "density_scale":
                scaled.append(replace(
                    species,
                    density_m3=tuple(float(d) * float(value) for d in species.density_m3),
                ))
            else:
                scaled.append(replace(
                    species,
                    temperature_keV=tuple(float(t) * float(value) for t in species.temperature_keV),
                ))
        return replace(case, species=tuple(scaled))
    raise AssertionError(f"unsupported scan path reached execution: {path!r}")


def expand_scan(case) -> list[tuple[tuple[float, ...], Any]]:
    """Expand ``case.scan`` into ``(axis values, derived case)`` pairs.

    The derived cases carry no ``[scan]`` of their own, so each is an ordinary
    single run and gets its own deterministic ``case_id``.
    """
    scan = case.scan
    if scan is None:
        raise ValueError("this case has no [scan] table to expand")
    points = _axis_points(scan)
    if len(points) > scan.max_cases:
        raise ValueError(
            f"scan expands to {len(points)} cases, above the {scan.max_cases} limit; "
            "raise scan.max_cases deliberately or shorten an axis"
        )
    expanded = []
    for point in points:
        derived = replace(case, scan=None)
        for axis, value in zip(scan.axes, point, strict=True):
            derived = _apply_axis(derived, axis.path, value)
        expanded.append((point, derived))
    return expanded


def _text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _existing_rows(path: Path) -> dict[str, dict[str, Any]]:
    """Rows already recorded in a scan output, keyed by ``case_id``.

    Returned as rows rather than as a set of ids because a resumed scan has to
    *rewrite* the whole output: the cached rows must be carried into the new
    file. Returning only the ids made the resume path write a file containing
    just the freshly-run cases, so resuming a finished scan replaced its
    results with an empty table -- destroying precisely what resume exists to
    preserve.
    """
    if not path.exists():
        return {}
    try:
        from ..result import Result  # noqa: PLC0415

        previous = Result.load(path)
    except (OSError, ValueError, KeyError):
        # An unreadable or half-written output is not a cache. Rerun rather
        # than trust it; a truncated file is exactly what an interrupted scan
        # leaves behind.
        return {}
    ids = previous.arrays.get("case_id")
    if ids is None:
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(np.asarray(ids).ravel()):
        case_id = _text(raw)
        row: dict[str, Any] = {"case_id": case_id, "status": "ok"}
        status = previous.arrays.get("status")
        if status is not None:
            row["status"] = _text(np.asarray(status).ravel()[index])
        for observable in SCAN_OBSERVABLES:
            values = previous.arrays.get(observable)
            if values is not None:
                row[observable] = float(np.asarray(values).ravel()[index])
        rows[case_id] = row
    return rows


def run_scan(
    case,
    *,
    out: Path | None = None,
    emit: Callable[[str], None] | None = None,
    resume: bool | None = None,
):
    """Run every case in ``case.scan`` and write one combined Result."""
    from ..execution import run_case  # noqa: PLC0415
    from ..result import Result  # noqa: PLC0415

    scan = case.scan
    if scan is None:
        raise ValueError("this case has no [scan] table to run")
    destination = Path(out) if out is not None else Path(scan.output)
    should_resume = scan.resume if resume is None else bool(resume)
    cached = _existing_rows(destination) if should_resume else {}

    def _say(message: str) -> None:
        if emit is not None:
            emit(message)

    expanded = expand_scan(case)
    _say(f"scan: {len(expanded)} cases over {len(scan.axes)} axis/axes ({scan.combine})")

    rows: list[dict[str, Any]] = []
    failures = 0
    for index, (point, derived) in enumerate(expanded, start=1):
        label = ", ".join(
            f"{axis.path}={value:g}" for axis, value in zip(scan.axes, point, strict=True)
        )
        if derived.case_id in cached:
            _say(f"[{index}/{len(expanded)}] cached  {label}")
            rows.append({**cached[derived.case_id], "point": point})
            continue
        _say(f"[{index}/{len(expanded)}] {label}")
        row: dict[str, Any] = {"point": point, "case_id": derived.case_id, "status": "ok"}
        try:
            result = run_case(derived)
        except Exception as exc:  # noqa: BLE001 - the failure is the datum
            failures += 1
            row["status"] = f"failed: {type(exc).__name__}: {exc}"
            _say(f"    failed: {exc}")
        else:
            for name in SCAN_OBSERVABLES:
                if name in result.arrays:
                    values = np.asarray(result.arrays[name], dtype=float)
                    row[name] = float(np.nanmax(np.abs(values))) if values.size else float("nan")
        rows.append(row)

    result = _assemble(case, scan, rows, destination)
    result.save(destination)
    _say(f"wrote {destination}")
    return result, failures


def _assemble(case, scan, rows, destination: Path):
    """Build the combined Result from the per-case rows."""
    from ..result import Result  # noqa: PLC0415

    n = len(rows)
    arrays: dict[str, np.ndarray] = {
        "case_id": np.array([row["case_id"] for row in rows], dtype=object),
        "status": np.array([row["status"] for row in rows], dtype=object),
    }
    dimensions: dict[str, tuple[str, ...]] = {"case_id": ("case",), "status": ("case",)}

    for position, axis in enumerate(scan.axes):
        name = f"axis_{axis.path.replace('.', '_').replace('[', '_').replace(']', '')}"
        arrays[name] = np.array([row["point"][position] for row in rows], dtype=float)
        dimensions[name] = ("case",)

    for observable in SCAN_OBSERVABLES:
        if any(observable in row for row in rows):
            arrays[observable] = np.array(
                [float(row.get(observable, np.nan)) for row in rows], dtype=float
            )
            dimensions[observable] = ("case",)

    ok = sum(1 for row in rows if row["status"] == "ok")
    return Result(
        case_id=case.case_id,
        case_name=case.name,
        workflow=f"scan:{case.run.workflow}",
        arrays=arrays,
        dimensions=dimensions,
        metadata={
            "scan_combine": scan.combine,
            "scan_axes": [axis.path for axis in scan.axes],
            "scan_cases": n,
            "scan_succeeded": ok,
            "scan_failed": n - ok,
            "converged": n > 0 and ok == n,
        },
    )
