from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import numpy as np


def _now() -> float:
    return time.perf_counter()

def _emit(msg: str, *, level: int, args: argparse.Namespace) -> None:
    """Simple structured stdout logging for the CLI.

    We intentionally avoid the stdlib `logging` module here to keep CLI output
    deterministic across platforms and to make it easy to compare with upstream
    SFINCS logs.
    """
    verbose = int(getattr(args, "verbose", 0) or 0)
    quiet = bool(getattr(args, "quiet", False))
    if quiet:
        return
    if verbose >= level:
        print(msg, flush=True)


def _cmd_converge(args: argparse.Namespace) -> int:
    """Refine each phase-space axis of a case and report observable convergence.

    Reports the joint refinement alongside the per-axis table because the two
    can disagree: axes that each look settled on their own are not evidence
    that the case is converged when they couple.
    """
    from rich.console import Console  # noqa: PLC0415
    from rich.table import Table  # noqa: PLC0415

    from .config import Case, CaseValidationError  # noqa: PLC0415
    from .workflows.converge import (  # noqa: PLC0415
        converge_case,
        converge_sfincs_input,
    )

    try:
        namelist = Path(args.case).suffix.lower() == ".namelist"
        case = args.case if namelist else Case.from_file(args.case)
    except (CaseValidationError, OSError) as exc:
        print(f"dkx converge failed: {exc}", file=sys.stderr)
        return 2

    progress = Console(stderr=True)
    emit = None if args.quiet else (lambda message: progress.print(message, highlight=False))
    # The solver kernels still write progress to stdout rather than emitting
    # events (plan.md section 5.6). For --format json that would interleave with
    # the report and make it unparseable, so stdout is borrowed for the duration
    # of the solves and the JSON is the only thing written to the real one.
    quiet_stdout = (
        contextlib.redirect_stdout(sys.stderr)
        if args.format == "json"
        else contextlib.nullcontext()
    )
    try:
        with quiet_stdout:
            report = (converge_sfincs_input if namelist else converge_case)(
                case,
                axes=tuple(args.axes),
                factor=args.factor,
                tolerance=args.tolerance,
                joint=not args.no_joint,
                emit=emit,
            )
    except (CaseValidationError, NotImplementedError, ValueError, OSError) as exc:
        print(f"dkx converge failed: {exc}", file=sys.stderr)
        return 2

    rows = [*report.refinements] + ([report.joint] if report.joint is not None else [])
    if args.format == "json":
        def _json_finite(value):
            if isinstance(value, dict):
                return {key: _json_finite(item) for key, item in value.items()}
            if isinstance(value, np.ndarray):
                return _json_finite(value.tolist())
            if isinstance(value, (list, tuple)):
                return [_json_finite(item) for item in value]
            if isinstance(value, (float, np.floating)):
                return float(value) if np.isfinite(value) else None
            if isinstance(value, np.integer):
                return int(value)
            if isinstance(value, np.bool_):
                return bool(value)
            return value

        def _rung_json(r):
            return {
                "label": r.label,
                "resolution": r.resolution,
                "changes": _json_finite(r.changes),
                "worst": _json_finite(r.worst),
                "seconds": r.seconds,
                "observables": _json_finite(r.observables),
                "original_residuals": _json_finite(r.original_residuals),
                "original_residual_norm": r.original_residual_norm,
                "original_residual_tolerance": r.original_residual_tolerance,
                "original_residual_complete_state": r.original_residual_complete_state,
                "original_residual_status": r.residual_evidence_status,
                "status": r.status,
                "refusal": r.refusal,
                "third_rung": (_rung_json(r.third_rung)
                               if r.third_rung is not None else None),
            }

        print(json.dumps({
            "baseline": report.baseline,
            "baseline_evidence": {
                "observables": _json_finite(report.baseline_observables),
                "original_residuals": _json_finite(report.baseline_original_residuals),
                "original_residual_norm": report.baseline_original_residual_norm,
                "original_residual_tolerance": report.baseline_original_residual_tolerance,
                "original_residual_complete_state": (
                    report.baseline_original_residual_complete_state
                ),
                "original_residual_status": report.baseline_residual_evidence_status,
                "status": "accepted",
            },
            "tolerance": report.tolerance,
            "converged": bool(report.converged),
            "original_equations_accepted": bool(report.original_equations_accepted),
            "axes_understate_the_joint_change": bool(report.axes_understate_the_joint_change),
            "refinements": [_rung_json(r) for r in rows],
        }, indent=2, sort_keys=True, allow_nan=False))
    else:
        console = Console()
        console.print(f"baseline {report.baseline}", highlight=False)
        table = Table(show_lines=False)
        table.add_column("refined", style="bold")
        table.add_column("resolution")
        table.add_column("worst relative change", justify="right")
        table.add_column("", justify="left")
        for row in rows:
            if row.status != "accepted":
                table.add_row(
                    row.label,
                    ", ".join(f"{k}={v}" for k, v in row.resolution.items()),
                    row.status,
                    f"[yellow]{row.refusal}[/yellow]",
                )
                continue
            inside = row.worst < report.tolerance
            table.add_row(
                row.label,
                ", ".join(f"{k}={v}" for k, v in row.resolution.items()),
                f"{row.worst:.3e}",
                "[green]within[/green]" if inside else "[yellow]above[/yellow]",
            )
        console.print(table)
        if report.axes_understate_the_joint_change:
            console.print(
                "[yellow]The joint refinement moved the outputs more than twice as far as any "
                "single axis. The per-axis rows are not a safe summary of this case: refine "
                "the axes together before treating it as converged.[/yellow]"
            )
        console.print(
            f"grid changes within tolerance {report.tolerance:g}: "
            + ("[green]yes[/green]" if report.converged else "[red]no[/red]")
        )
        console.print(
            "original equations accepted: "
            + (
                "[green]yes[/green]"
                if report.original_equations_accepted
                else "[yellow]no[/yellow]"
            )
        )
    return 0 if report.converged and report.original_equations_accepted else 1


def _cmd_roots(args: argparse.Namespace) -> int:
    """Print the ambipolar root table stored in a Result.

    plan.md section 5.6 asks Rich to own "root tables and branch events". The
    ambipolar workflow already records every root's field, current, slope,
    classification, bracket and branch, plus the events where a branch
    appears or vanishes; until now nothing surfaced them except reading the
    NetCDF by hand.

    A branch event marked nonsmooth is reported rather than smoothed over: the
    output is not differentiable across it, so a gradient taken through that
    interval is meaningless even though ``jax.grad`` will return a number.
    """
    from rich.console import Console  # noqa: PLC0415
    from rich.table import Table  # noqa: PLC0415

    from .result import Result  # noqa: PLC0415

    try:
        result = Result.load(args.result)
    except (OSError, ValueError, KeyError) as exc:
        print(f"dkx roots failed: {exc}", file=sys.stderr)
        return 2

    arrays = result.arrays
    if "ambipolar_root_kV_m" not in arrays:
        print(
            f"dkx roots failed: {result.case_name} carries no ambipolar roots. "
            "Roots come from workflow = 'ambipolar_profile'; this result used "
            f"workflow = '{result.workflow}'.",
            file=sys.stderr,
        )
        return 2

    fields = np.asarray(arrays["ambipolar_root_kV_m"], dtype=float)
    counts = np.asarray(arrays["ambipolar_root_count"]).astype(int).ravel()
    currents = np.asarray(arrays["ambipolar_root_current_A_m2"], dtype=float)
    slopes = np.asarray(arrays["ambipolar_root_slope_A_m2_per_kV_m"], dtype=float)
    kinds = np.asarray(arrays["ambipolar_root_type"])
    widths = np.asarray(arrays["ambipolar_root_final_bracket_width_kV_m"], dtype=float)
    selected = np.asarray(arrays.get("selected_ambipolar_root", [])).ravel()

    def _text(value: object) -> str:
        return value.decode() if isinstance(value, bytes) else str(value)

    rows: list[dict[str, object]] = []
    for surface in range(fields.shape[0]):
        for index in range(int(counts[surface])):
            rows.append({
                "surface": surface,
                "root": index,
                "field_kV_m": float(fields[surface, index]),
                "current_A_m2": float(currents[surface, index]),
                "slope": float(slopes[surface, index]),
                "type": _text(kinds[surface, index]),
                "bracket_width_kV_m": float(widths[surface, index]),
                "selected": bool(selected.size > surface and selected[surface] == index),
            })

    nonsmooth = arrays.get("ambipolar_nonsmooth_event")
    nonsmooth_surfaces = (
        [int(i) for i, flag in enumerate(np.asarray(nonsmooth).ravel()) if flag]
        if nonsmooth is not None
        else []
    )

    if args.format == "json":
        print(json.dumps({
            "case": result.case_name,
            "roots": rows,
            "nonsmooth_event_surfaces": nonsmooth_surfaces,
        }, indent=2, sort_keys=True))
        return 0

    console = Console()
    console.print(f"{result.case_name} ({result.case_id[:12]})", markup=False, highlight=False)
    if not rows:
        console.print(
            "No roots were admitted. A sign-sampled scan cannot see a tangential "
            "root or an even number of crossings between samples, so this is not "
            "evidence that none exist."
        )
        return 0

    table = Table(show_lines=False)
    for column, justify in (
        ("surface", "right"), ("root", "right"), ("E_r [kV/m]", "right"),
        ("J_r [A/m^2]", "right"), ("type", "left"), ("bracket [kV/m]", "right"),
        ("selected", "left"),
    ):
        table.add_column(column, justify=justify)
    for row in rows:
        table.add_row(
            str(row["surface"]), str(row["root"]),
            f"{row['field_kV_m']:.6g}", f"{row['current_A_m2']:.3e}",
            str(row["type"]), f"{row['bracket_width_kV_m']:.2e}",
            "yes" if row["selected"] else "",
        )
    console.print(table)

    if nonsmooth_surfaces:
        console.print(
            f"[yellow]Nonsmooth branch event on surface(s) {nonsmooth_surfaces}. "
            "A root appears or vanishes there, so the selected-root output is not "
            "differentiable across it; a gradient through that interval is not "
            "meaningful even though jax.grad returns one.[/yellow]"
        )
    return 0


def _looks_like_sfincs_h5(path: Path) -> bool:
    """True for a SFINCS HDF5 output, false for a dkx NetCDF Result.

    Decided by extension rather than by sniffing the file: NetCDF4 *is* HDF5,
    so an h5py open succeeds on both and would misroute every dkx Result into
    the SFINCS comparison.
    """
    return path.suffix.lower() in {".h5", ".hdf5"}


#: Arrays that record how a solve went rather than what it computed. They are
#: reported but never decide the verdict: wall-clock time differs between any
#: two runs of the same case, so counting it would make `dkx compare` exit
#: non-zero on a bit-identical re-run and train the reader to ignore the exit
#: status. Iteration counts move with warm starts for the same reason.
_COMPARE_INFORMATIONAL: frozenset[str] = frozenset({"solve_time_s", "solver_iterations"})


def _compare_result_arrays(a, b, *, rtol: float, atol: float):
    """Compare two dkx Results array by array.

    Returns ``(rows, only_a, only_b)``. Keys present in one result and not the
    other are reported separately rather than skipped: a comparison that
    silently ignores them would call two runs equal when one stopped producing
    an output entirely.

    Rows in :data:`_COMPARE_INFORMATIONAL` carry ``status="informational"``
    when they differ, so they appear in the table without failing the run.
    """
    shared = sorted(set(a.arrays) & set(b.arrays))
    only_a = sorted(set(a.arrays) - set(b.arrays))
    only_b = sorted(set(b.arrays) - set(a.arrays))

    rows = []
    for key in shared:
        left = np.asarray(a.arrays[key])
        right = np.asarray(b.arrays[key])
        if left.shape != right.shape:
            rows.append({"key": key, "status": "shape",
                         "detail": f"{left.shape} vs {right.shape}",
                         "max_abs": float("nan"), "max_rel": float("nan")})
            continue
        if not (np.issubdtype(left.dtype, np.number) and np.issubdtype(right.dtype, np.number)):
            same = bool(np.array_equal(left, right))
            rows.append({"key": key, "status": "ok" if same else "differs",
                         "detail": "non-numeric", "max_abs": 0.0 if same else float("nan"),
                         "max_rel": 0.0 if same else float("nan")})
            continue
        lf = left.astype(float, copy=False)
        rf = right.astype(float, copy=False)
        diff = np.abs(lf - rf)
        # NaN in the same place on both sides is agreement, not a difference.
        both_nan = np.isnan(lf) & np.isnan(rf)
        diff = np.where(both_nan, 0.0, diff)
        scale = np.maximum(np.abs(lf), np.abs(rf))
        max_abs = float(np.nanmax(diff)) if diff.size else 0.0
        with np.errstate(invalid="ignore", divide="ignore"):
            rel = np.where(scale > 0.0, diff / scale, 0.0)
        max_rel = float(np.nanmax(rel)) if rel.size else 0.0
        ok = bool(np.allclose(lf, rf, rtol=rtol, atol=atol, equal_nan=True))
        status = "ok" if ok else ("informational" if key in _COMPARE_INFORMATIONAL else "differs")
        rows.append({"key": key, "status": status,
                     "detail": "", "max_abs": max_abs, "max_rel": max_rel})
    return rows, only_a, only_b


def _cmd_compare(args: argparse.Namespace) -> int:
    """Compare two results, dkx-native or SFINCS, and exit non-zero on a difference.

    plan.md section 5.6 lists one ``compare`` rather than a per-format command.
    Dispatch is by extension: ``.h5`` pairs go to the SFINCS comparison, which
    already carries the upstream per-dataset tolerances, and everything else is
    read back as a dkx Result. A mixed pair is refused -- the two carry
    different variable names, so "nothing matched" would look like agreement.
    """
    from rich.console import Console  # noqa: PLC0415
    from rich.table import Table  # noqa: PLC0415

    a_path, b_path = Path(args.a), Path(args.b)
    a_h5, b_h5 = _looks_like_sfincs_h5(a_path), _looks_like_sfincs_h5(b_path)
    if a_h5 != b_h5:
        print(
            f"dkx compare failed: cannot compare a SFINCS HDF5 output against a dkx "
            f"Result ({a_path.name} vs {b_path.name}). They use different variable "
            "names, so every key would be unmatched and the result would read as "
            "agreement. Convert one first.",
            file=sys.stderr,
        )
        return 2

    if a_h5:
        args.tolerances_json = getattr(args, "tolerances_json", None)
        args.show_all = args.verbose_keys
        from .cli import _cmd_compare_h5  # noqa: PLC0415

        return _cmd_compare_h5(args)

    from .result import Result  # noqa: PLC0415

    try:
        left = Result.load(a_path)
        right = Result.load(b_path)
    except (OSError, ValueError, KeyError) as exc:
        print(f"dkx compare failed: {exc}", file=sys.stderr)
        return 2

    rows, only_a, only_b = _compare_result_arrays(
        left, right, rtol=float(args.rtol), atol=float(args.atol)
    )
    # Anything not "ok" and not informational fails, so a status added later
    # (a shape mismatch, a non-numeric difference) counts by default rather
    # than passing silently because it was not listed here.
    bad = [r for r in rows if r["status"] not in {"ok", "informational"}]
    noted = [r for r in rows if r["status"] == "informational"]

    if args.format == "json":
        print(json.dumps({"a": str(a_path), "b": str(b_path), "rows": rows,
                          "only_in_a": only_a, "only_in_b": only_b,
                          "agree": not bad and not only_a and not only_b},
                         indent=2, sort_keys=True))
        return 0 if not bad and not only_a and not only_b else 2

    console = Console()
    shown = rows if args.verbose_keys else (bad + noted)
    if shown:
        table = Table(show_lines=False)
        table.add_column("array", style="bold")
        table.add_column("status")
        table.add_column("max abs", justify="right")
        table.add_column("max rel", justify="right")
        for row in shown[:60]:
            table.add_row(row["key"], row["status"] or row["detail"],
                          f"{row['max_abs']:.3e}", f"{row['max_rel']:.3e}")
        console.print(table)
        if len(shown) > 60:
            console.print(f"... {len(shown) - 60} more rows not shown")
    for label, keys in (("only in A", only_a), ("only in B", only_b)):
        if keys:
            console.print(f"[yellow]{label} ({len(keys)}): {', '.join(keys[:12])}"
                          + (" ..." if len(keys) > 12 else "") + "[/yellow]")
    if not bad and not only_a and not only_b:
        console.print(f"[green]{len(rows)} arrays agree within rtol={args.rtol:g} "
                      f"atol={args.atol:g}[/green]")
        return 0
    console.print(f"[red]{len(bad)} of {len(rows)} arrays differ[/red]")
    return 2


def _cmd_plot(args: argparse.Namespace) -> int:
    """Plot a result, dkx-native or SFINCS, dispatching on the file extension.

    plan.md section 5.6 lists one ``plot``. The SFINCS path is the existing
    diagnostics panel; the dkx path is a radial-profile panel that did not
    exist before -- ``OutputConfig.plots`` was in the case schema but nothing
    read it, so a native Result could only be looked at by writing a script.
    """
    from .result import Result  # noqa: PLC0415

    source = Path(args.result)
    out_path = Path(args.out) if args.out else source.with_suffix(".png")

    if _looks_like_sfincs_h5(source):
        if args.kind not in ("auto", "summary"):
            print(
                f"dkx plot failed: --kind {args.kind} applies to a dkx Result; "
                "a SFINCS HDF5 file has only the summary panel.",
                file=sys.stderr,
            )
            return 2
        from .plotting import plot_sfincs_output_summary  # noqa: PLC0415

        written = plot_sfincs_output_summary(input_h5=source, output_png=out_path)
    else:
        from .plotting import (  # noqa: PLC0415
            plot_ambipolar_search,
            plot_result_summary,
            plot_scan_summary,
        )

        try:
            result = Result.load(source)
        except (OSError, ValueError, KeyError) as exc:
            print(f"dkx plot failed: {exc}", file=sys.stderr)
            return 2

        kind = args.kind
        if kind == "auto":
            # A scan Result has no r_N to plot against, so the profile panel
            # would refuse it. Dispatch on what the file actually holds rather
            # than making the user know which panel their result supports.
            kind = "scan" if any(k.startswith("axis_") for k in result.arrays) else "summary"
        draw = {
            "summary": plot_result_summary,
            "search": plot_ambipolar_search,
            "scan": plot_scan_summary,
        }[kind]
        try:
            written = draw(result=result, output_path=out_path)
        except ValueError as exc:
            print(f"dkx plot failed: {exc}", file=sys.stderr)
            return 2

    if not args.quiet:
        print(f"wrote {written}")
    return 0


def _cmd_convert(args: argparse.Namespace) -> int:
    """Convert a SFINCS ``input.namelist`` into a native case file.

    This is the migration path for SFINCS users, and the direction that matters:
    a deck is dimensionless and single-surface, a case is SI and profile-shaped,
    so the conversion is a real translation rather than a rename. Anything the
    case schema cannot carry refuses here, naming the namelist key, rather than
    producing a case that runs and answers a different question (plan.md
    operating rule 11).
    """
    from rich.console import Console  # noqa: PLC0415
    from rich.table import Table  # noqa: PLC0415

    from .config import CaseValidationError  # noqa: PLC0415
    from .input_compat import convert_sfincs_namelist  # noqa: PLC0415

    try:
        case, written = convert_sfincs_namelist(
            args.source, args.destination, name=args.name, overwrite=args.force
        )
    except (CaseValidationError, OSError, ValueError) as exc:
        print(f"dkx convert failed: {exc}", file=sys.stderr)
        return 2

    if args.quiet:
        return 0
    console = Console(stderr=False)
    console.print(f"{case.name} ({case.case_id[:12]})", markup=False, highlight=False)
    table = Table(show_lines=False)
    table.add_column("quantity", style="bold")
    table.add_column("value")
    table.add_row("source", str(args.source))
    table.add_row("case", str(written))
    table.add_row("workflow", case.run.workflow)
    table.add_row("geometry", f"{case.geometry.format} ({case.geometry.file})")
    table.add_row(
        "surfaces",
        ", ".join(f"{value:.6g}" for value in case.geometry.surfaces),
    )
    table.add_row("species", ", ".join(item.name for item in case.species))
    table.add_row(
        "electric field",
        case.electric_field.mode
        + (
            f" at {case.electric_field.value_kV_m:.6g} kV/m"
            if case.electric_field.value_kV_m is not None
            else f" over {list(case.electric_field.search_kV_m or ())} kV/m"
        ),
    )
    console.print(table)
    # The deck names one surface; the case states a profile. Saying so here is
    # the difference between a reader trusting the extra surfaces and wondering
    # where they came from.
    console.print(
        f"The deck's single surface became {len(case.geometry.surfaces)} surfaces "
        "carrying a profile linear in rHat, so the deck's prescribed gradients are "
        "recovered exactly where it asked for them.",
        highlight=False,
    )
    return 0


def _is_namelist(path: str | None) -> bool:
    """True for anything that is not a native ``.toml``/``.json`` case (``None`` is ``./input.namelist``)."""
    return path is None or Path(path).suffix.lower() not in {".toml", ".json"}


def _scan_namelist(args: argparse.Namespace) -> int:
    from .workflows.profile_scan import ScanInputError, scan  # noqa: PLC0415

    er = None if args.er is None else (float(args.er[0]), float(args.er[1]), int(args.er[2]))
    try:
        result = scan(
            args.case or "input.namelist", out_dir=args.out_dir, profiles=args.profiles, er=er,
            radii=None if args.radii is None else [float(r) for r in args.radii],
            skip_existing=args.skip_existing, solve_method=args.solve_method, jobs=args.jobs,
            emit=None if args.quiet else (lambda level, msg: print(msg, flush=True)),
        )
    except (ScanInputError, OSError) as exc:
        print(f"dkx scan failed: {exc}", file=sys.stderr)
        return 2
    if result.failures:
        print(f"dkx scan: {len(result.failures)} points failed (see scan_failures.json)", file=sys.stderr)
        return 1
    return 0


def _cmd_scan(args: argparse.Namespace) -> int:
    """Expand a case's ``[scan]`` axes, run every point, and write one Result.

    Exits non-zero when any point failed. The output is still written: a scan
    is run because each point is expensive, so losing the completed ones
    because a later one failed is the wrong trade.
    """
    from rich.console import Console  # noqa: PLC0415

    from .config import Case, CaseValidationError  # noqa: PLC0415
    from .workflows.converge import AXES  # noqa: PLC0415
    from .workflows.scan import run_scan  # noqa: PLC0415

    if _is_namelist(args.case):
        return _scan_namelist(args)
    try:
        case = Case.from_file(args.case)
    except (CaseValidationError, OSError) as exc:
        print(f"dkx scan failed: {exc}", file=sys.stderr)
        return 2
    if case.scan is None:
        print(
            f"dkx scan failed: {args.case} has no [scan] table. Add one with at least "
            "one [[scan.axis]], or use `dkx run` for a single case.",
            file=sys.stderr,
        )
        return 2

    progress = Console(stderr=True)
    emit = None if args.quiet else (lambda message: progress.print(message, highlight=False))
    try:
        result, failures = run_scan(
            case,
            out=Path(args.out) if args.out else None,
            emit=emit,
            resume=False if args.no_resume else None,
            check_grid=None if args.check_grid is None else tuple(args.check_grid) or AXES,
        )
    except (CaseValidationError, ValueError) as exc:
        print(f"dkx scan failed: {exc}", file=sys.stderr)
        return 2

    total = int(result.metadata.get("scan_cases", 0))
    if failures:
        print(
            f"dkx scan: {failures} of {total} cases failed; the rest were written",
            file=sys.stderr,
        )
        return 1
    if not args.quiet:
        Console().print(f"[green]{total} cases completed[/green]")
    return 0


def _cmd_run_case(args: argparse.Namespace) -> int:
    """Execute a case and write its Result.

    This is the command the case API had no CLI path to. Before it, `dkx`
    could validate a Case and print its schema but not run one: every
    executing subcommand took a SFINCS namelist, so the case workflow was
    Python-only.
    """
    from rich.console import Console  # noqa: PLC0415
    from rich.table import Table  # noqa: PLC0415

    from .config import Case, CaseValidationError  # noqa: PLC0415

    if _is_namelist(args.case):
        # The SFINCS executable's contract: input.namelist in, sfincsOutput.h5 out.
        from .cli import main as _main  # noqa: PLC0415

        deck = args.case or "input.namelist"
        if not Path(deck).is_file():
            print(f"dkx run failed: {deck} not found", file=sys.stderr)
            return 2
        return _main(["write-output", "--input", deck, "--out", args.out or "sfincsOutput.h5"] + (["--quiet"] if args.quiet else []))
    console = Console(stderr=False)
    try:
        case = Case.from_file(args.case)
    except (CaseValidationError, OSError) as exc:
        print(f"dkx run failed: {exc}", file=sys.stderr)
        return 2

    # A case file names where its result goes, and `dkx run` honours it.
    # It did not: `[output].file` was parsed, validated, resolved and recorded
    # on the Result, and then the save was skipped unless --out was given, so a
    # case that declared an output path ran to convergence and wrote nothing
    # without saying so. --out still wins when both are given.
    out_path = (
        Path(args.out)
        if args.out
        else (case.base_directory / case.output.file).resolve()
    )
    from .execution import run_case  # noqa: PLC0415
    from .workflows.converge import AXES  # noqa: PLC0415

    # Progress goes to stderr so `dkx run ... --out -` style piping of the
    # summary stays clean, and so a redirected log keeps the two streams apart.
    progress = Console(stderr=True)
    emit = None if args.quiet else (lambda message: progress.print(message, highlight=False))

    started = time.perf_counter()
    try:
        result = run_case(
            case, out=out_path, emit=emit,
            check_grid=None if args.check_grid is None else tuple(args.check_grid) or AXES,
        )
    except (CaseValidationError, NotImplementedError, ValueError) as exc:
        # A model the case route does not implement must say so precisely
        # rather than fall back to something adjacent (plan.md operating rule 11).
        print(f"dkx run failed: {exc}", file=sys.stderr)
        return 2
    elapsed = time.perf_counter() - started

    # `[output].plots` asks for the summary figure next to the Result.
    plot_path = None
    if case.output.plots:
        try:
            plot_path = result.plot(out_path.with_suffix(".png"))
        except (ImportError, ValueError) as exc:
            print(f"dkx run: plot skipped: {exc}", file=sys.stderr)

    console.print(
        f"{result.case_name} ({result.case_id[:12]})", markup=False, highlight=False
    )
    table = Table(show_lines=False)
    table.add_column("quantity", style="bold")
    table.add_column("value")
    table.add_row("workflow", str(result.workflow))
    table.add_row("surfaces", str(len(case.geometry.surfaces)))
    table.add_row("species", str(len(case.species)))
    table.add_row("converged", "yes" if result.metadata.get("converged") else "no")
    residual = result.metadata.get("residual_norm")
    table.add_row("true residual", "not measured" if residual is None else f"{residual:.3e}")
    table.add_row("solver route", str(result.metadata.get("solver_route", "unknown")))
    table.add_row("wall time", f"{elapsed:.2f} s")
    certificate = result.metadata.get("resolution_certificate")
    if certificate:
        table.add_row(
            "grid check",
            ("resolved" if certificate["all_resolved"] else "NOT RESOLVED")
            + f" (worst change {certificate['worst_change']:.1%} under {', '.join(certificate['axes'])} x{certificate['factor']:g})",
        )
    table.add_row("result", str(out_path))
    if plot_path is not None:
        table.add_row("plot", str(plot_path))
    console.print(table)

    if not args.quiet:
        result.print_summary()
    return 0


def _cmd_inspect_result(args: argparse.Namespace) -> int:
    """Print what a saved Result contains, without recomputing it."""
    from rich.console import Console  # noqa: PLC0415
    from rich.table import Table  # noqa: PLC0415

    from .result import Result  # noqa: PLC0415

    try:
        result = Result.load(args.result)
    except (OSError, ValueError, KeyError) as exc:
        print(f"dkx inspect failed: {exc}", file=sys.stderr)
        return 2

    console = Console()
    console.print(
        f"{result.case_name} ({result.case_id[:12]})", markup=False, highlight=False
    )
    header = Table()
    header.add_column("quantity", style="bold")
    header.add_column("value")
    header.add_row("workflow", str(result.workflow))
    header.add_row("result format", str(result.schema_version))
    header.add_row("converged", "yes" if result.metadata.get("converged") else "no")
    console.print(header)

    # No units column: a Result carries no per-variable units metadata
    # yet. plan.md section 5.5 requires it, and until it exists an empty column
    # would imply the metadata is present and blank rather than absent. Names
    # carry the unit by convention (heat_flux_W_m2), which is what a reader has.
    arrays = Table(title="arrays")
    arrays.add_column("name", style="bold")
    arrays.add_column("shape")
    arrays.add_column("dtype")
    for name in sorted(result.arrays):
        value = np.asarray(result.arrays[name])
        arrays.add_row(name, str(value.shape), str(value.dtype))
    console.print(arrays)
    return 0


