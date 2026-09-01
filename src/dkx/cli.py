from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

from . import runtime as _runtime
from .input_compat import with_equilibrium_override
from .namelist import read_sfincs_input

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

def _emit_namelist_summary(*, nml, args: argparse.Namespace) -> None:
    geom = nml.group("geometryParameters")
    phys = nml.group("physicsParameters")
    res = nml.group("resolutionParameters")
    general = nml.group("general")

    def _g(group: dict, key: str, default=None):
        return group.get(key.upper(), default)

    _emit("----------------------------------------------------------------", level=0, args=args)
    _emit(" input.namelist summary", level=0, args=args)
    _emit(f" geometryScheme={_g(geom, 'geometryScheme', '?')}", level=0, args=args)
    _emit(f" RHSMode={_g(general, 'RHSMode', '?')}", level=0, args=args)
    _emit(f" collisionOperator={_g(phys, 'collisionOperator', '?')}", level=0, args=args)
    _emit(f" includePhi1={bool(_g(phys, 'includePhi1', False))}", level=0, args=args)
    _emit(f" includePhi1InKineticEquation={bool(_g(phys, 'includePhi1InKineticEquation', False))}", level=2, args=args)
    _emit(f" includePhi1InCollisionOperator={bool(_g(phys, 'includePhi1InCollisionOperator', False))}", level=2, args=args)
    _emit(f" useDKESExBDrift={bool(_g(phys, 'useDKESExBDrift', False))}", level=2, args=args)
    _emit(
        " resolution:"
        f" Ntheta={_g(res, 'Ntheta', '?')}"
        f" Nzeta={_g(res, 'Nzeta', '?')}"
        f" Nxi={_g(res, 'Nxi', '?')}"
        f" NL={_g(res, 'NL', '?')}"
        f" Nx={_g(res, 'Nx', '?')}",
        level=0,
        args=args,
    )
    _emit(f" solverTolerance={_g(res, 'solverTolerance', '?')}", level=2, args=args)

def _emit_runtime_info(*, args: argparse.Namespace) -> None:
    """Emit basic runtime info helpful for benchmarking and bug reports."""
    try:
        import jax  # noqa: PLC0415
        import jax.numpy as _jnp  # noqa: PLC0415

        _emit(f" jax={jax.__version__} backend={jax.default_backend()} devices={jax.devices()}", level=2, args=args)
        _emit(f" jax_enable_x64={bool(_jnp.array(0.0).dtype == _jnp.float64)}", level=3, args=args)
    except Exception:  # noqa: BLE001
        return


def _cmd_validate_case(args: argparse.Namespace) -> int:
    """Validate a case without touching JAX kernels or external files."""
    from .config import Case, CaseValidationError  # noqa: PLC0415

    try:
        case = Case.from_file(args.case)
    except (CaseValidationError, OSError) as exc:
        print(f"dkx validate failed: {exc}", file=sys.stderr)
        return 2
    print(f"valid DKX case: {case.name}")
    print(f"case_id: {case.case_id}")
    print(
        f"workflow: {case.run.workflow}; "
        f"surfaces: {len(case.geometry.surfaces)}; species: {len(case.species)}"
    )
    if case.scan is not None:
        print(
            f"scan: {case.scan.case_count} cases "
            f"(limit {case.scan.max_cases}, resume={str(case.scan.resume).lower()})"
        )
    if case.run.workflow == "ambipolar_profile":
        from .workflows.ambipolar_native import (  # noqa: PLC0415
            preflight_ambipolar_case,
        )

        try:
            preflight = preflight_ambipolar_case(case)
        except ValueError as exc:
            print(f"dkx validate failed: {exc}", file=sys.stderr)
            return 2
        print(
            "ambipolar preflight: "
            f"hierarchy_points={preflight.hierarchy_points}; "
            f"max_evaluations_per_surface={preflight.evaluations_per_surface}; "
            f"max_profile_evaluations={preflight.profile_evaluations}"
        )
        print(
            "retained evidence upper bound: "
            f"{preflight.retained_profile_bytes} B profile "
            f"({preflight.retained_bytes_per_surface} B/surface); "
            "runtime not estimated"
        )
        if preflight.search_strategy == "seeded_brackets":
            print(
                "seeded bracket scope: "
                f"endpoint_counts={list(preflight.search_points_by_surface)}; "
                "explicit intervals only; unsampled crossings not excluded"
            )
    return 0


def _cmd_schema(args: argparse.Namespace) -> int:
    """Print the complete human or machine-readable case schema."""
    from .config import COMMENTED_TOML_EXAMPLE, case_json_schema  # noqa: PLC0415

    if args.format == "toml":
        print(COMMENTED_TOML_EXAMPLE, end="")
    else:
        print(json.dumps(case_json_schema(), indent=2, sort_keys=True))
    return 0

def _doctor_checks() -> list[tuple[str, str, str]]:
    """Collect one ``(status, name, detail)`` row per environment check.

    Every row reports what this process *observed*, not what it was asked for.
    The distinction matters: ``JAX_ENABLE_X64`` being set in the environment is
    not evidence that float64 is active, because a backend already initialized
    by an earlier import ignores it. So the check below allocates an array and
    reads its dtype. The same rule applies to the accelerator row, which lists
    the devices JAX actually enumerates rather than the platform requested.

    Status is one of ``ok``, ``warn`` or ``fail``. Only ``fail`` means the
    install cannot run correctly; ``warn`` marks something absent that limits
    what is available without breaking the core solver.
    """
    from importlib.metadata import PackageNotFoundError, version  # noqa: PLC0415

    rows: list[tuple[str, str, str]] = []

    py = ".".join(str(n) for n in sys.version_info[:3])
    rows.append(
        ("ok", "python", py) if sys.version_info >= (3, 11)
        else ("fail", "python", f"{py} is below the 3.11 floor")
    )

    try:
        rows.append(("ok", "dkx", version("dkx")))
    except PackageNotFoundError:
        rows.append(("warn", "dkx", "not installed as a distribution (running from a checkout)"))

    # solvax carries the solver routes, and a version below the declared floor
    # fails deep inside a solve rather than at import, so it is checked here.
    floor = (0, 19, 0)
    try:
        raw = version("solvax")
        parsed = tuple(int(part) for part in raw.split(".")[:3])
        rows.append(
            ("ok", "solvax", raw) if parsed >= floor
            else ("fail", "solvax", f"{raw} is below the {'.'.join(map(str, floor))} floor")
        )
    except PackageNotFoundError:
        rows.append(("fail", "solvax", "missing; every canonical solve needs it"))
    except ValueError:
        rows.append(("warn", "solvax", f"{raw} could not be compared against the floor"))

    for name, required in (
        ("jax", True), ("jaxlib", True), ("numpy", True), ("scipy", True),
        ("h5py", False), ("netCDF4", False), ("matplotlib", False), ("rich", True),
    ):
        try:
            rows.append(("ok", name, version(name)))
        except PackageNotFoundError:
            rows.append(("fail" if required else "warn", name, "missing"))

    # Observed float64, not the environment variable that requests it.
    try:
        import warnings  # noqa: PLC0415

        import jax.numpy as jnp  # noqa: PLC0415

        # JAX warns when it truncates a requested float64 to float32. Probing
        # for exactly that is the point here, so the warning is the finding,
        # not a problem to surface twice.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            dtype = str(jnp.zeros(1, dtype=jnp.float64).dtype)
        rows.append(
            ("ok", "float64", "active") if dtype == "float64"
            else ("fail", "float64", f"arrays materialize as {dtype}; results will be wrong")
        )
    except Exception as exc:  # noqa: BLE001 - any backend failure is the finding
        rows.append(("fail", "float64", f"could not allocate a JAX array: {exc}"))

    # Observed devices, not the requested platform.
    try:
        import jax  # noqa: PLC0415

        devices = jax.devices()
        kinds = sorted({d.platform for d in devices})
        rows.append(("ok", "devices", f"{len(devices)} x {','.join(kinds)}"))
    except Exception as exc:  # noqa: BLE001
        rows.append(("fail", "devices", f"JAX enumerated no devices: {exc}"))

    return rows


def _cmd_doctor(args: argparse.Namespace) -> int:
    """Report whether this install can run, and exit non-zero when it cannot.

    Exists because the failure modes users actually hit are environmental, not
    physical: a conda numpy 1.x that dies inside ``ml_dtypes``, a solvax below
    the floor that fails mid-solve, or float64 silently off so every number is
    plausible and wrong. Each of those produces a traceback that names neither
    dkx nor the real cause.
    """
    from rich.console import Console  # noqa: PLC0415
    from rich.table import Table  # noqa: PLC0415

    rows = _doctor_checks()

    if args.format == "json":
        print(json.dumps(
            {name: {"status": status, "detail": detail} for status, name, detail in rows},
            indent=2, sort_keys=True,
        ))
    else:
        console = Console()
        table = Table(show_lines=False)
        table.add_column("")
        table.add_column("check", style="bold")
        table.add_column("observed")
        marks = {"ok": "[green]ok[/green]", "warn": "[yellow]warn[/yellow]", "fail": "[red]fail[/red]"}
        for status, name, detail in rows:
            table.add_row(marks[status], name, detail)
        console.print(table)

    failures = [name for status, name, _ in rows if status == "fail"]
    if failures:
        print(f"dkx doctor: {len(failures)} blocking problem(s): {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


def _cmd_converge(args: argparse.Namespace) -> int:
    """Refine each phase-space axis of a case and report observable convergence.

    Reports the joint refinement alongside the per-axis table because the two
    can disagree: axes that each look settled on their own are not evidence
    that the case is converged when they couple.
    """
    from rich.console import Console  # noqa: PLC0415
    from rich.table import Table  # noqa: PLC0415

    from .config import Case, CaseValidationError  # noqa: PLC0415
    from .workflows.converge import converge_case  # noqa: PLC0415

    try:
        case = Case.from_file(args.case)
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
            report = converge_case(
                case,
                axes=tuple(args.axes),
                factor=args.factor,
                tolerance=args.tolerance,
                joint=not args.no_joint,
                emit=emit,
            )
    except (CaseValidationError, NotImplementedError, ValueError) as exc:
        print(f"dkx converge failed: {exc}", file=sys.stderr)
        return 2

    rows = [*report.refinements] + ([report.joint] if report.joint is not None else [])
    if args.format == "json":
        print(json.dumps({
            "baseline": report.baseline,
            "tolerance": report.tolerance,
            "converged": report.converged,
            "axes_understate_the_joint_change": report.axes_understate_the_joint_change,
            "refinements": [
                {
                    "label": r.label,
                    "resolution": r.resolution,
                    "changes": r.changes,
                    "worst": r.worst,
                    "seconds": r.seconds,
                }
                for r in rows
            ],
        }, indent=2, sort_keys=True))
    else:
        console = Console()
        console.print(f"baseline {report.baseline}", highlight=False)
        table = Table(show_lines=False)
        table.add_column("refined", style="bold")
        table.add_column("resolution")
        table.add_column("worst relative change", justify="right")
        table.add_column("", justify="left")
        for row in rows:
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
            f"converged at tolerance {report.tolerance:g}: "
            + ("[green]yes[/green]" if report.converged else "[red]no[/red]")
        )
    return 0 if report.converged else 1


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
        from .plotting import plot_sfincs_output_summary  # noqa: PLC0415

        written = plot_sfincs_output_summary(input_h5=source, output_png=out_path)
    else:
        from .plotting import plot_result_summary  # noqa: PLC0415

        try:
            result = Result.load(source)
        except (OSError, ValueError, KeyError) as exc:
            print(f"dkx plot failed: {exc}", file=sys.stderr)
            return 2
        try:
            written = plot_result_summary(result=result, output_path=out_path)
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


def _cmd_scan(args: argparse.Namespace) -> int:
    """Expand a case's ``[scan]`` axes, run every point, and write one Result.

    Exits non-zero when any point failed. The output is still written: a scan
    is run because each point is expensive, so losing the completed ones
    because a later one failed is the wrong trade.
    """
    from rich.console import Console  # noqa: PLC0415

    from .config import Case, CaseValidationError  # noqa: PLC0415
    from .workflows.scan import run_scan  # noqa: PLC0415

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

    console = Console(stderr=False)
    try:
        case = Case.from_file(args.case)
    except (CaseValidationError, OSError) as exc:
        print(f"dkx run failed: {exc}", file=sys.stderr)
        return 2

    out_path = Path(args.out) if args.out else None
    from .execution import run_case  # noqa: PLC0415

    # Progress goes to stderr so `dkx run ... --out -` style piping of the
    # summary stays clean, and so a redirected log keeps the two streams apart.
    progress = Console(stderr=True)
    emit = None if args.quiet else (lambda message: progress.print(message, highlight=False))

    started = time.perf_counter()
    try:
        result = run_case(case, out=out_path, emit=emit)
    except (CaseValidationError, NotImplementedError, ValueError) as exc:
        # A model the case route does not implement must say so precisely
        # rather than fall back to something adjacent (plan.md operating rule 11).
        print(f"dkx run failed: {exc}", file=sys.stderr)
        return 2
    elapsed = time.perf_counter() - started

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
    if out_path is not None:
        table.add_row("result", str(out_path))
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
    header.add_row("schema", str(result.schema_version))
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


def _emit_parallel_runtime_info(*, args: argparse.Namespace) -> None:
    def _env(name: str, default: str = "") -> str:
        return os.environ.get(name, default).strip()

    cores = _env("DKX_CORES")
    threads = _env("NPROC")
    cpu_devices = _env("DKX_CPU_DEVICES")
    transport_parallel = _env("DKX_TRANSPORT_PARALLEL", "off") or "off"
    transport_workers = _env("DKX_TRANSPORT_PARALLEL_WORKERS", "1") or "1"
    distributed = _env("DKX_DISTRIBUTED")

    if not any(
        (
            cores,
            cpu_devices,
            transport_parallel not in {"", "off"},
            distributed,
        )
    ):
        return

    _emit(
        " parallel:"
        f" cores={cores or '-'}"
        f" threads={threads or '-'}"
        f" cpu_devices={cpu_devices or '-'}",
        level=1,
        args=args,
    )
    _emit(
        f" transport_parallel: mode={transport_parallel} workers={transport_workers}",
        level=1,
        args=args,
    )
    if distributed in {"1", "true", "yes", "on"}:
        _emit(
            " multi_host:"
            " enabled=1"
            f" process_id={_env('DKX_PROCESS_ID', '-') or '-'}"
            f" process_count={_env('DKX_PROCESS_COUNT', '-') or '-'}"
            f" coordinator={_env('DKX_COORDINATOR_ADDRESS', '-') or '-'}"
            f" port={_env('DKX_COORDINATOR_PORT', '-') or '-'}",
            level=1,
            args=args,
        )

def _nml_with_cli_equilibrium_override(nml, args: argparse.Namespace):
    return with_equilibrium_override(
        nml=nml,
        equilibrium_file=getattr(args, "equilibrium_file", None),
        wout_path=getattr(args, "wout_path", None),
    )

@contextlib.contextmanager
def _canonical_namelist_path(*, nml, input_path: Path, args: argparse.Namespace):
    """Yield a namelist path for the canonical driver, honoring CLI equilibrium overrides.

    The canonical :func:`dkx.run.run_transport_matrix` reads the input
    file itself, so a ``--equilibrium-file``/``--wout-path`` override is
    materialized as a sibling temporary namelist (same directory, so any other
    relative paths keep resolving) and removed afterwards.
    """
    from .input_compat import canonical_equilibrium_override  # noqa: PLC0415

    override = canonical_equilibrium_override(
        equilibrium_file=getattr(args, "equilibrium_file", None),
        wout_path=getattr(args, "wout_path", None),
    )
    if override is None:
        yield input_path
        return
    if nml.source_text is None:
        raise ValueError(
            "--equilibrium-file/--wout-path require a readable input.namelist source text."
        )
    tmp = tempfile.NamedTemporaryFile(
        "w",
        dir=str(input_path.parent),
        prefix=f".{input_path.stem}.override.",
        suffix=".namelist",
        delete=False,
        encoding="utf-8",
    )
    try:
        tmp.write(nml.source_text)
        tmp.close()
        yield Path(tmp.name)
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

def _add_equilibrium_override_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--equilibrium-file",
        default=None,
        help="Override geometryParameters.equilibriumFile without editing input.namelist.",
    )
    parser.add_argument(
        "--wout-path",
        default=None,
        help="Compatibility alias for --equilibrium-file, commonly used for geometryScheme=5 VMEC runs.",
    )

def _add_common_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=1,
        help="Increase verbosity (repeatable).",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Reduce output to a minimum.")
    parser.add_argument(
        "--cores",
        type=int,
        default=None,
        help=(
            "Solver CPU threads (sets DKX_CORES; pins the XLA host threadpool "
            "via NPROC plus the OpenMP/OpenBLAS pools before JAX initializes). "
            "0 lets XLA size the threadpool itself; when omitted the threadpool "
            "is clamped to min(8, cpu_count) — the measured optimum is 4-8 "
            "threads, and a full-width pool on a many-core host is slower."
        ),
    )
    parser.add_argument(
        "--fortran-stdout",
        dest="fortran_stdout",
        action="store_true",
        help="Mirror upstream v3 stdout line-for-line (including KSP/SNES iteration lines).",
    )
    parser.add_argument(
        "--no-fortran-stdout",
        dest="fortran_stdout",
        action="store_false",
        help="Disable strict Fortran-style stdout mirroring.",
    )
    parser.set_defaults(fortran_stdout=None)

def _add_parallel_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--transport-workers",
        type=int,
        default=None,
        help="Parallel worker processes for independent transport-RHS solves.",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable JAX multi-host distributed initialization for this CLI run.",
    )
    parser.add_argument("--process-id", type=int, default=None, help="Multi-host JAX process id.")
    parser.add_argument("--process-count", type=int, default=None, help="Multi-host JAX process count.")
    parser.add_argument(
        "--coordinator-address",
        default=None,
        help="Multi-host JAX coordinator host or host:port.",
    )
    parser.add_argument(
        "--coordinator-port",
        type=int,
        default=None,
        help="Coordinator port when --coordinator-address omits it.",
    )

def _cmd_solve_v3(args: argparse.Namespace) -> int:
    t0 = _now()
    nml = _nml_with_cli_equilibrium_override(read_sfincs_input(Path(args.input)), args)
    rhs_mode = int(nml.group("general").get("RHSMODE", 1))
    _emit("################################################################", level=0, args=args)
    _emit(" dkx solve-v3", level=0, args=args)
    _emit(f" input={Path(args.input).resolve()}", level=0, args=args)
    _emit_namelist_summary(nml=nml, args=args)
    _emit_runtime_info(args=args)
    _emit_parallel_runtime_info(args=args)
    _emit(f" tol={args.tol} atol={args.atol} restart={args.restart} maxiter={args.maxiter} solve_method={args.solve_method}", level=1, args=args)
    if args.which_rhs is not None:
        _emit(f" whichRHS={args.which_rhs}", level=0, args=args)

    out_state = Path(args.out_state)

    # The canonical stack (dkx.run) owns every supported deck.  Invalid
    # namelist values (RHSMode outside 1-3, out-of-range option values) surface
    # as load-time validation errors.
    quiet = bool(getattr(args, "quiet", False))
    emit_line = None if quiet else (lambda line: _emit(line, level=0, args=args))
    try:
        with _canonical_namelist_path(nml=nml, input_path=Path(args.input), args=args) as namelist_path:
            if rhs_mode == 1:
                from .run import run_profile  # noqa: PLC0415

                run = run_profile(
                    namelist_path,
                    solve_method=str(args.solve_method),
                    tol=float(args.tol),
                    emit=emit_line,
                )
                state = np.asarray(run.state_vector)
                residual = float(np.atleast_1d(np.asarray(run.solve_result.residual_norms, dtype=np.float64))[0])
            else:
                from .run import run_transport_matrix  # noqa: PLC0415

                run = run_transport_matrix(
                    namelist_path,
                    solve_method=str(args.solve_method),
                    tol=float(args.tol),
                    emit=emit_line,
                )
                state_vectors = np.asarray(run.state_vectors)
                residual_norms = np.atleast_1d(np.asarray(run.solve_result.residual_norms, dtype=np.float64))
                col = (int(args.which_rhs) - 1) if args.which_rhs is not None else 0
                if not (0 <= col < state_vectors.shape[0]):
                    raise ValueError(
                        f"whichRHS={args.which_rhs} is out of range for the "
                        f"{state_vectors.shape[0]} transport-matrix RHS columns"
                    )
                state = state_vectors[col]
                residual = float(residual_norms[col])
    except (NotImplementedError, ValueError, RuntimeError) as exc:
        if os.environ.get("DKX_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}:
            raise
        print(f"dkx solve-v3 failed: {exc}", file=sys.stderr, flush=True)
        return 2
    out_state.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_state, state)
    _emit(f" wrote stateVector -> {out_state.resolve()}", level=0, args=args)
    _emit(f" residual_norm={residual:.6e}", level=0, args=args)
    _emit(f" elapsed_s={_now()-t0:.3f}", level=1, args=args)
    return 0

def _cmd_run_fortran(args: argparse.Namespace) -> int:
    t0 = _now()
    from .validation.fortran import run_sfincs_fortran  # noqa: PLC0415

    _emit("################################################################", level=0, args=args)
    _emit(" dkx run-fortran", level=0, args=args)
    _emit(f" input={Path(args.input).resolve()}", level=0, args=args)
    output_path = run_sfincs_fortran(
        input_namelist=Path(args.input),
        exe=Path(args.exe) if args.exe else None,
        workdir=Path(args.workdir) if args.workdir else None,
    )
    _emit(f" wrote sfincsOutput.h5 -> {output_path}", level=0, args=args)
    _emit(f" elapsed_s={_now()-t0:.3f}", level=1, args=args)
    return 0

def _cmd_write_output(args: argparse.Namespace) -> int:
    t0 = _now()
    from .writer import output_format_from_suffix  # noqa: PLC0415

    nml = _nml_with_cli_equilibrium_override(read_sfincs_input(Path(args.input)), args)
    rhs_mode = int(nml.group("general").get("RHSMODE", 1))
    try:
        output_format = output_format_from_suffix(Path(args.out))
    except ValueError as exc:
        print(f"dkx write-output failed: {exc}", file=sys.stderr, flush=True)
        return 2
    _emit("################################################################", level=0, args=args)
    _emit(" dkx write-output", level=0, args=args)
    _emit(f" input={Path(args.input).resolve()}", level=0, args=args)
    _emit(f" output={Path(args.out).resolve()} format={output_format}", level=0, args=args)
    _emit_namelist_summary(nml=nml, args=args)
    _emit_runtime_info(args=args)
    _emit_parallel_runtime_info(args=args)

    # Default to upstream v3 behavior: full solve/write appropriate to RHSMode
    # (--geometry-only skips the solve), all on the canonical stack
    # (dkx.run).  Invalid namelist values (RHSMode outside 1-3,
    # out-of-range option values) surface as load-time validation errors.
    geometry_only = bool(getattr(args, "geometry_only", False))
    res_group = nml.group("resolutionParameters")
    try:
        solver_tol = float(res_group.get("SOLVERTOLERANCE", 1e-10))
    except (TypeError, ValueError):
        solver_tol = 1e-10
    quiet = bool(getattr(args, "quiet", False))
    emit_line = None if quiet else (lambda line: _emit(line, level=0, args=args))
    solver_trace_path = Path(args.solver_trace) if getattr(args, "solver_trace", None) else None
    overwrite = bool(args.overwrite)
    fortran_layout = bool(args.fortran_layout)
    try:
        with _canonical_namelist_path(nml=nml, input_path=Path(args.input), args=args) as namelist_path:
            if geometry_only:
                from .run import run_geometry  # noqa: PLC0415

                run = run_geometry(
                    namelist_path,
                    out_path=Path(args.out),
                    overwrite=overwrite,
                    fortran_layout=fortran_layout,
                    solver_trace_path=solver_trace_path,
                    emit=emit_line,
                )
            elif rhs_mode == 1:
                from .run import run_profile  # noqa: PLC0415

                run = run_profile(
                    namelist_path,
                    solve_method=str(getattr(args, "solve_method", "auto")),
                    tol=solver_tol,
                    out_path=Path(args.out),
                    overwrite=overwrite,
                    fortran_layout=fortran_layout,
                    solver_trace_path=solver_trace_path,
                    emit=emit_line,
                )
            else:
                from .run import run_transport_matrix  # noqa: PLC0415

                run = run_transport_matrix(
                    namelist_path,
                    solve_method=str(getattr(args, "solve_method", "auto")),
                    tol=solver_tol,
                    out_path=Path(args.out),
                    overwrite=overwrite,
                    fortran_layout=fortran_layout,
                    solver_trace_path=solver_trace_path,
                    emit=emit_line,
                )
    except (NotImplementedError, ValueError, FileExistsError, RuntimeError) as exc:
        if os.environ.get("DKX_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}:
            raise
        print(f"dkx write-output failed: {exc}", file=sys.stderr, flush=True)
        return 2
    _emit(f" wrote output -> {run.output_path}", level=0, args=args)
    if solver_trace_path is not None:
        _emit(f" wrote solver trace -> {solver_trace_path.resolve()}", level=0, args=args)
    _emit(f" elapsed_s={_now()-t0:.3f}", level=1, args=args)
    return 0

def _cmd_transport_matrix_v3(args: argparse.Namespace) -> int:
    """RHSMode=2/3 transport-matrix runs on the canonical stack (:mod:`dkx.run`)."""
    t0 = _now()
    from .run import run_transport_matrix  # noqa: PLC0415

    input_path = Path(args.input)
    nml = _nml_with_cli_equilibrium_override(read_sfincs_input(input_path), args)
    _emit("################################################################", level=0, args=args)
    _emit(" dkx transport-matrix-v3", level=0, args=args)
    _emit(f" input={input_path.resolve()}", level=0, args=args)
    _emit_namelist_summary(nml=nml, args=args)
    _emit_runtime_info(args=args)
    _emit_parallel_runtime_info(args=args)
    _emit(f" tol={args.tol} solve_method={args.solve_method}", level=1, args=args)
    quiet = bool(getattr(args, "quiet", False))
    with _canonical_namelist_path(nml=nml, input_path=input_path, args=args) as namelist_path:
        run = run_transport_matrix(
            namelist_path,
            solve_method=str(args.solve_method),
            tol=float(args.tol),
            out_path=Path(args.out) if getattr(args, "out", None) else None,
            emit=None if quiet else (lambda line: _emit(line, level=0, args=args)),
        )

    out_tm = Path(args.out_matrix)
    out_tm.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_tm, np.asarray(run.transport_matrix))
    _emit(f" wrote transportMatrix -> {out_tm.resolve()}", level=0, args=args)
    if run.output_path is not None:
        _emit(f" wrote output -> {Path(run.output_path).resolve()}", level=0, args=args)

    if args.out_state_prefix is not None:
        pref = Path(args.out_state_prefix)
        pref.parent.mkdir(parents=True, exist_ok=True)
        for idx, x in enumerate(np.asarray(run.state_vectors), start=1):
            p = pref.with_name(f"{pref.name}.whichRHS{idx}.npy")
            np.save(p, np.asarray(x))
            _emit(f" wrote stateVector(whichRHS={idx}) -> {p.resolve()}", level=1, args=args)

    residual_norms = np.atleast_1d(np.asarray(run.solve_result.residual_norms, dtype=np.float64))
    for idx, rn in enumerate(residual_norms, start=1):
        _emit(f" whichRHS={idx} residual_norm={float(rn):.6e}", level=0, args=args)
    _emit(f" elapsed_s={_now()-t0:.3f}", level=1, args=args)
    return 0

def _cmd_monoenergetic_database(args: argparse.Namespace) -> int:
    """Scan (nuPrime, EStar) and write the monoenergetic-coefficient database."""
    t0 = _now()
    from .monoenergetic import monoenergetic_database, save_database  # noqa: PLC0415

    input_path = Path(args.input)
    nu_values = [float(v) for v in args.nu_prime]
    er_values = [float(v) for v in args.e_star]
    _emit("################################################################", level=0, args=args)
    _emit(" dkx monoenergetic-database", level=0, args=args)
    _emit(f" input={input_path.resolve()}", level=0, args=args)
    _emit(f" nuPrime grid ({len(nu_values)}): {nu_values}", level=0, args=args)
    _emit(f" EStar grid ({len(er_values)}): {er_values}", level=0, args=args)
    quiet = bool(getattr(args, "quiet", False))
    db = monoenergetic_database(
        input_path,
        nu_values,
        er_values,
        solve_method=str(args.solve_method),
        tol=float(args.tol),
        emit=None if quiet else (lambda line: _emit(line, level=0, args=args)),
    )
    out = save_database(Path(args.out), db)
    _emit(f" wrote database -> {out.resolve()}", level=0, args=args)
    header = f" {'nuPrime':>12} {'EStar':>12} {'nu_star':>12} {'D11*':>13} {'D31*':>13} {'D13*':>13} {'D33*':>13}"
    _emit(header, level=0, args=args)
    nu_star = np.asarray(db.nu_star)
    for i, nu in enumerate(np.asarray(db.nu_prime)):
        for j, er in enumerate(np.asarray(db.e_star)):
            _emit(
                f" {nu:12.5e} {er:12.5e} {nu_star[i]:12.5e}"
                f" {float(np.asarray(db.d11_star)[i, j]):13.6e}"
                f" {float(np.asarray(db.d31_star)[i, j]):13.6e}"
                f" {float(np.asarray(db.d13_star)[i, j]):13.6e}"
                f" {float(np.asarray(db.d33_star)[i, j]):13.6e}",
                level=0,
                args=args,
            )
    _emit(f" elapsed_s={_now()-t0:.3f}", level=1, args=args)
    return 0

def _cmd_dump_h5(args: argparse.Namespace) -> int:
    from .io import read_sfincs_h5  # noqa: PLC0415

    data = read_sfincs_h5(Path(args.sfincs_output))
    if args.keys_only:
        for k in sorted(data.keys()):
            print(k)
        return 0
    out = {k: v.tolist() if hasattr(v, "tolist") else v for k, v in data.items()}
    Path(args.out_json).write_text(json.dumps(out, indent=2, sort_keys=True))
    return 0

def _default_plot_output_path(input_h5: Path) -> Path:
    input_h5 = Path(input_h5)
    stem = input_h5.stem
    if stem.endswith(".sfincsOutput"):
        stem = stem[: -len(".sfincsOutput")]
    return input_h5.with_name(f"{stem}_summary.pdf")

def _cmd_plot_output(args: argparse.Namespace) -> int:
    t0 = _now()
    from .plotting import plot_sfincs_output_summary  # noqa: PLC0415

    input_h5 = Path(args.input_h5)
    out_path = Path(args.out) if args.out else _default_plot_output_path(input_h5)
    _emit("################################################################", level=0, args=args)
    _emit(" dkx plot-output", level=0, args=args)
    _emit(f" input={input_h5.resolve()}", level=0, args=args)
    _emit(f" out={out_path.resolve()}", level=0, args=args)
    plot_path = plot_sfincs_output_summary(input_h5=input_h5, output_png=out_path)
    _emit(f" wrote plot -> {plot_path}", level=0, args=args)
    _emit(f" elapsed_s={_now()-t0:.3f}", level=1, args=args)
    return 0

def _cmd_compare_h5(args: argparse.Namespace) -> int:
    from .compare import compare_sfincs_outputs  # noqa: PLC0415

    tolerances = None
    if args.tolerances_json:
        with open(args.tolerances_json, "r", encoding="utf-8") as f:
            tolerances = json.load(f)
    results = compare_sfincs_outputs(
        a_path=Path(args.a),
        b_path=Path(args.b),
        rtol=float(args.rtol),
        atol=float(args.atol),
        tolerances=tolerances,
    )
    bad = [r for r in results if not r.ok]
    if args.show_all:
        for r in results:
            status = "OK" if r.ok else "FAIL"
            print(f"{status} {r.key}: max_abs={r.max_abs:.3e} max_rel={r.max_rel:.3e}")
    else:
        for r in bad[:50]:
            print(f"FAIL {r.key}: max_abs={r.max_abs:.3e} max_rel={r.max_rel:.3e}")
        if len(bad) > 50:
            print(f"... {len(bad) - 50} more failing keys omitted")
    return 0 if not bad else 2

def _cmd_scan_er(args: argparse.Namespace) -> int:
    t0 = _now()
    from .workflows.scans import linspace_including_endpoints, run_er_scan  # noqa: PLC0415

    _emit("################################################################", level=0, args=args)
    _emit(" dkx scan-er", level=0, args=args)
    _emit(f" input={Path(args.input).resolve()}", level=0, args=args)
    _emit(f" out-dir={Path(args.out_dir).resolve()}", level=0, args=args)
    _emit_runtime_info(args=args)
    _emit_parallel_runtime_info(args=args)

    if args.values is not None:
        values = [float(x) for x in args.values]
    else:
        values = list(linspace_including_endpoints(float(args.min), float(args.max), int(args.n)))

    run_er_scan(
        input_namelist=Path(args.input),
        out_dir=Path(args.out_dir),
        values=values,
        compute_transport_matrix=bool(args.compute_transport_matrix),
        compute_solution=bool(getattr(args, "compute_solution", False)),
        skip_existing=bool(getattr(args, "skip_existing", False)),
        solve_method=str(getattr(args, "solve_method", "auto")),
        differentiable=False,
        jobs=int(args.jobs) if getattr(args, "jobs", None) is not None else None,
        index=int(args.index) if getattr(args, "index", None) is not None else None,
        stride=int(args.stride) if getattr(args, "stride", None) is not None else None,
        emit=lambda level, msg: _emit(msg, level=level, args=args),
    )
    _emit(f" elapsed_s={_now()-t0:.3f}", level=1, args=args)
    return 0

def _cmd_ambipolar_solve(args: argparse.Namespace) -> int:
    t0 = _now()
    from .ambipolar import solve_ambipolar_from_scan_dir  # noqa: PLC0415

    _emit("################################################################", level=0, args=args)
    _emit(" dkx ambipolar-solve", level=0, args=args)
    _emit(f" scan-dir={Path(args.scan_dir).resolve()}", level=0, args=args)
    _emit_runtime_info(args=args)
    _emit_parallel_runtime_info(args=args)

    res = solve_ambipolar_from_scan_dir(
        scan_dir=Path(args.scan_dir),
        write_pickle=True,
        write_json=True,
        n_fine=int(args.n_fine),
    )

    if res.roots_er.size == 0:
        _emit(" ambipolar-solve: no sign change found (no roots).", level=0, args=args)
    else:
        for i, (rv, re, rt) in enumerate(zip(res.roots_var, res.roots_er, res.root_types, strict=False), start=1):
            _emit(f" root[{i}] {res.var_name}={float(rv):.16g} Er={float(re):.16g} type={rt}", level=0, args=args)

    _emit(f" wrote {Path(args.scan_dir).resolve() / 'ambipolarSolutions.dat'}", level=1, args=args)
    _emit(f" wrote {Path(args.scan_dir).resolve() / 'ambipolarSolutions.json'}", level=2, args=args)
    _emit(f" elapsed_s={_now()-t0:.3f}", level=1, args=args)
    return 0

def _cmd_ambipolar(args: argparse.Namespace) -> int:
    t0 = _now()
    from .er import find_ambipolar_er  # noqa: PLC0415

    _emit("################################################################", level=0, args=args)
    _emit(" dkx ambipolar", level=0, args=args)
    _emit(f" input={Path(args.input).resolve()}", level=0, args=args)
    _emit(f" out-dir={Path(args.out_dir).resolve()}", level=0, args=args)
    _emit_runtime_info(args=args)
    _emit_parallel_runtime_info(args=args)
    _emit(
        " ambipolar:"
        f" method=brent er_min={float(args.er_min):.16g}"
        f" er_max={float(args.er_max):.16g}"
        f" er_initial={float(args.er_initial):.16g}"
        f" max_evaluations={int(args.max_evaluations)}"
        f" current_tolerance={float(args.current_tolerance):.3e}",
        level=0,
        args=args,
    )

    result = find_ambipolar_er(
        Path(args.input),
        er_bracket=(float(args.er_min), float(args.er_max)),
        er_initial=float(args.er_initial),
        max_iter=int(args.max_evaluations),
        current_tol=float(args.current_tolerance),
        solve_method=str(args.solve_method),
        emit=lambda msg: _emit(msg, level=1, args=args),
    )

    summary_path = Path(args.summary_json) if args.summary_json else Path(args.out_dir) / "ambipolar_result.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "converged": bool(result.converged),
        "method": result.method,
        "status": result.status,
        "message": result.message,
        "root_er": result.er,
        "root_radial_current": result.radial_current,
        "root_type": result.root_type,
        "iterations": [
            {
                "index": item.index,
                "er": item.er,
                "radial_current": item.radial_current,
                "stage": item.stage,
            }
            for item in result.iterations
        ],
        "roots": [
            {
                "er": root.er,
                "radial_current": root.radial_current,
                "slope": root.slope,
                "root_type": root.root_type,
            }
            for root in result.roots
        ],
        "elapsed_s": float(_now() - t0),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    if result.converged and result.er is not None:
        _emit(
            f" ambipolar root: Er={float(result.er):.16g} "
            f"radial_current={float(result.radial_current):.6e} type={result.root_type}",
            level=0,
            args=args,
        )
    else:
        _emit(f" ambipolar status={result.status}: {result.message}", level=0, args=args)
    _emit(f" wrote summary -> {summary_path.resolve()}", level=0, args=args)
    _emit(f" elapsed_s={_now()-t0:.3f}", level=1, args=args)
    return 0 if result.converged else 2

def _apply_cores_setting(cores: int | None) -> None:
    """Record the requested solver thread count in the process environment.

    ``cores > 0`` pins the XLA host threadpool (``NPROC`` — the variable XLA
    actually reads when its CPU backend initializes) and defaults the host BLAS
    pools (``OMP_NUM_THREADS``/``OPENBLAS_NUM_THREADS``) to match; ``cores ==
    0`` requests XLA's own full-width sizing (``DKX_CORES=0`` suppresses the
    package default clamp of ``min(8, cpu_count)``).  Thread counts only take
    effect before JAX initializes, so the CLI re-execs itself with
    ``DKX_CORES`` exported (:func:`_maybe_reexec_for_early_runtime`) and the
    package applies the variables at import; the assignments here keep child
    processes (transport workers, spawned tools) consistent.
    """
    if cores is None:
        return
    try:
        cores_val = int(cores)
    except (TypeError, ValueError):
        return
    if cores_val < 0:
        return
    os.environ["DKX_CORES"] = str(cores_val)
    if cores_val > 0:
        os.environ["NPROC"] = str(cores_val)
        os.environ.setdefault("OMP_NUM_THREADS", str(cores_val))
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(cores_val))

def _apply_runtime_env_defaults() -> None:
    # Avoid large eager GPU preallocation in CLI workflows so solver/benchmark
    # runs coexist better with other accelerator jobs by default.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

def _apply_parallel_runtime_settings(args: argparse.Namespace) -> None:
    transport_workers = getattr(args, "transport_workers", None)
    if transport_workers is not None:
        workers_val = max(1, int(transport_workers))
        os.environ["DKX_TRANSPORT_PARALLEL"] = "process" if workers_val > 1 else "off"
        os.environ["DKX_TRANSPORT_PARALLEL_WORKERS"] = str(workers_val)

    if bool(getattr(args, "distributed", False)):
        from . import initialize_distributed_runtime_from_env  # noqa: PLC0415

        os.environ["DKX_DISTRIBUTED"] = "1"
        process_id = getattr(args, "process_id", None)
        process_count = getattr(args, "process_count", None)
        coordinator_address = getattr(args, "coordinator_address", None)
        coordinator_port = getattr(args, "coordinator_port", None)
        if process_id is not None:
            os.environ["DKX_PROCESS_ID"] = str(int(process_id))
        if process_count is not None:
            os.environ["DKX_PROCESS_COUNT"] = str(int(process_count))
        if coordinator_address:
            os.environ["DKX_COORDINATOR_ADDRESS"] = str(coordinator_address)
        if coordinator_port is not None:
            os.environ["DKX_COORDINATOR_PORT"] = str(int(coordinator_port))
        initialize_distributed_runtime_from_env()

def _maybe_handle_plot(argv: list[str]) -> int | None:
    """``dkx --plot PATH`` / ``dkx PATH`` --- the vmex-style front door.

    Handled before subcommand dispatch, so the common cases need no subcommand:
    an output file renders its panels, an equilibrium is solved first and then
    rendered.  Returns ``None`` when this invocation is not one of those, so the
    existing subcommands are untouched.
    """
    from pathlib import Path as _Path  # noqa: PLC0415

    target: str | None = None
    if "--plot" in argv:
        i = argv.index("--plot")
        if i + 1 < len(argv) and not argv[i + 1].startswith("-"):
            target = argv[i + 1]
        else:  # `--plot` with the path as the sole positional
            rest = [a for a in argv if a != "--plot" and not a.startswith("-")]
            target = rest[0] if rest else None
        if target is None:
            print("dkx --plot needs a path: an output file, or an equilibrium.")
            return 2
    else:
        # Find a positional .nc/.h5 anywhere in argv, not just as a lone token:
        # `dkx wout.nc --out fig.png` must work, and requiring len(argv)==1 sent
        # it to write-output, which reads the netCDF as a namelist and dies on
        # a UnicodeDecodeError.  Values belonging to option flags are skipped.
        _OPTS_WITH_VALUE = {"--out", "--equilibrium-file", "--wout", "--cores"}
        skip_next = False
        for tok in argv:
            if skip_next:
                skip_next = False
                continue
            if tok.startswith("-"):
                skip_next = tok in _OPTS_WITH_VALUE
                continue
            cand = _Path(tok)
            if cand.is_file() and cand.suffix.lower() in {".nc", ".h5"}:
                target = tok
                break
            break  # a positional that is not an equilibrium: leave argv alone
    if target is None:
        return None

    out = None
    if "--out" in argv:
        j = argv.index("--out")
        if j + 1 < len(argv):
            out = argv[j + 1]
    path = _Path(target)
    if not path.exists():
        print(f"dkx --plot: no such file: {path}")
        return 2
    from dkx.representative import plot_output_file, run_representative  # noqa: PLC0415

    # Dispatch on CONTENT, not extension: ".nc" is a SFINCS output *or* a VMEC
    # wout, and guessing from the suffix breaks `dkx --plot sfincsOutput.nc`,
    # which predates this entry point.
    def _is_solver_output(candidate) -> bool:
        try:
            from dkx.io import read_sfincs_output_file  # noqa: PLC0415

            data = read_sfincs_output_file(candidate)
        except Exception:
            return False
        return any(k in data for k in ("RHSMode", "FSABFlow", "transportMatrix"))

    if _is_solver_output(path):
        print(f" dkx --plot {path.name} (solver output)")
        print(f" wrote {plot_output_file(path, out)}")
    else:
        quick = "--quick" in argv
        label = " (quick)" if quick else ""
        print(f" dkx {path.name} — representative run{label}")
        print(f" wrote {run_representative(path, out_path=out, full='--full' in argv, quick=quick)}")
    return 0


#: Refinement axes, mirrored from dkx.workflows.converge.AXES so building the
#: parser does not import the execution stack. The test suite pins them equal.
_CONVERGE_AXES: tuple[str, ...] = ("theta", "zeta", "pitch", "speed")


#: The commands `dkx --help` advertises, in the order plan.md section 5.6 lists
#: them. The SFINCS commands are reachable as `dkx sfincs <command>` and as
#: hidden top-level aliases, but are deliberately absent here: listing 21
#: choices is what the compatibility group exists to avoid.
_USER_COMMANDS: tuple[str, ...] = (
    "doctor", "schema", "validate", "run", "roots", "converge", "inspect",
    "compare", "plot", "scan", "convert", "sfincs",
)


#: Every registered subcommand name. ``main`` does not read this -- it passes
#: the parser's own ``sub.choices`` -- so the runtime behaviour cannot drift
#: from the registered set. It exists for direct callers and as documentation.
_KNOWN_COMMANDS: frozenset[str] = frozenset({
    "validate", "doctor", "converge", "roots", "compare", "plot", "scan", "convert", "schema", "run", "inspect", "solve-v3", "ambipolar",
    "scan-er", "ambipolar-solve", "run-fortran", "write-output",
    "transport-matrix-v3", "monoenergetic-database", "dump-h5", "plot-output",
    "compare-h5", "postprocess-upstream",
})


def _normalize_default_argv(
    argv: list[str], known_cmds: frozenset[str] | set[str] = _KNOWN_COMMANDS
) -> list[str]:
    """Insert the implicit ``write-output`` command when the user named none.

    ``main`` passes the built parser's own command set rather than the constant
    above. That coupling is deliberate: the set was previously a literal here,
    and adding a subcommand without mirroring it in meant the new name was not
    recognised as a command, fell through to the positional namelist path, and
    surfaced as a ``FileNotFoundError`` naming a file the user never typed.
    """
    if not argv:
        return argv
    if any(tok in known_cmds for tok in argv):
        return argv
    global_opts_with_val = {
        "--cores",
        "--transport-workers",
        "--process-id",
        "--process-count",
        "--coordinator-address",
        "--coordinator-port",
    }
    global_opts_no_val = {
        "-v",
        "--verbose",
        "-q",
        "--quiet",
        "--fortran-stdout",
        "--no-fortran-stdout",
        "--distributed",
    }
    if "--plot" in argv:
        global_args: list[str] = []
        rest: list[str] = []
        input_h5: str | None = None
        idx = 0
        while idx < len(argv):
            tok = argv[idx]
            if tok in global_opts_with_val:
                if idx + 1 < len(argv):
                    global_args.extend([tok, argv[idx + 1]])
                    idx += 2
                    continue
            if tok.startswith("--cores="):
                global_args.append(tok)
                idx += 1
                continue
            if tok in global_opts_no_val:
                global_args.append(tok)
                idx += 1
                continue
            if tok == "--plot":
                if idx + 1 < len(argv):
                    input_h5 = argv[idx + 1]
                    idx += 2
                    continue
            rest.append(tok)
            idx += 1
        if input_h5 is not None:
            return [*global_args, "plot-output", "--input-h5", input_h5, *rest]
    global_args: list[str] = []
    rest: list[str] = []
    input_path: str | None = None
    idx = 0
    while idx < len(argv):
        tok = argv[idx]
        if tok in global_opts_with_val:
            if idx + 1 < len(argv):
                global_args.extend([tok, argv[idx + 1]])
                idx += 2
                continue
        if tok.startswith("--cores="):
            global_args.append(tok)
            idx += 1
            continue
        if tok in global_opts_no_val:
            global_args.append(tok)
            idx += 1
            continue
        if tok.startswith("-"):
            rest.append(tok)
            idx += 1
            continue
        if input_path is None:
            input_path = tok
        else:
            rest.append(tok)
        idx += 1
    if input_path is None:
        return argv
    return [*global_args, "write-output", "--input", input_path, *rest]

def _maybe_reexec_for_early_runtime(argv: list[str]) -> None:
    """Re-exec with early runtime env so thread pinning/bootstrap take effect.

    The CLI is imported after the package, so JAX may already be imported before
    flags like `--cores` or `--distributed` are parsed. When those flags would
    change pre-import runtime state (the XLA threadpool is sized once, at CPU
    backend initialization), restart the process once with the relevant env
    vars set before package import.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--cores", type=int, default=None)
    pre.add_argument("--distributed", action="store_true")
    pre.add_argument("--process-id", type=int, default=None)
    pre.add_argument("--process-count", type=int, default=None)
    pre.add_argument("--coordinator-address", default=None)
    pre.add_argument("--coordinator-port", type=int, default=None)
    args, _ = pre.parse_known_args(argv)

    desired: dict[str, str] = {}
    if args.cores is not None and int(args.cores) >= 0:
        desired["DKX_CORES"] = str(int(args.cores))
    if bool(args.distributed):
        desired["DKX_DISTRIBUTED"] = "1"
        if args.process_id is not None:
            desired["DKX_PROCESS_ID"] = str(int(args.process_id))
        if args.process_count is not None:
            desired["DKX_PROCESS_COUNT"] = str(int(args.process_count))
        if args.coordinator_address is not None:
            desired["DKX_COORDINATOR_ADDRESS"] = str(args.coordinator_address)
        if args.coordinator_port is not None:
            desired["DKX_COORDINATOR_PORT"] = str(int(args.coordinator_port))

    if not desired:
        return

    if all(os.environ.get(key, "") == value for key, value in desired.items()):
        return

    env = os.environ.copy()
    env.update(desired)
    env["DKX_CLI_BOOTSTRAPPED"] = "1"
    os.execvpe(sys.executable, [sys.executable, "-m", "dkx", *argv], env)

def _merge_global_cli_args(argv: list[str], args: argparse.Namespace) -> argparse.Namespace:
    """Preserve global CLI flags regardless of whether they appear before or after the subcommand.

    Argparse defaults on both the root parser and subparsers can otherwise cause
    root-level values to be overwritten by subparser defaults when a flag is
    supplied before the subcommand. Parse the shared global options once more
    from the full argv and reapply them onto the final namespace.
    """
    pre = argparse.ArgumentParser(add_help=False)
    _add_common_cli_args(pre)
    _add_parallel_cli_args(pre)
    pre_args, _ = pre.parse_known_args(argv)
    for name in (
        "verbose",
        "quiet",
        "cores",
        "fortran_stdout",
        "transport_workers",
        "distributed",
        "process_id",
        "process_count",
        "coordinator_address",
        "coordinator_port",
    ):
        setattr(args, name, getattr(pre_args, name))
    return args

class _HiddenAliases:
    """Registers subcommands without listing them in ``--help``.

    The SFINCS commands are registered twice: once under ``dkx sfincs``, where
    they are documented, and once at the top level so existing scripts keep
    working. Only the first set should appear in ``dkx --help`` -- otherwise
    the compatibility group would make the help output longer rather than
    shorter, which is the opposite of what plan.md section 5.6 asks for.
    """

    def __init__(self, sub) -> None:
        self._sub = sub

    def add_parser(self, name: str, **kwargs):
        kwargs["help"] = argparse.SUPPRESS
        return self._sub.add_parser(name, **kwargs)


def _add_compat_parsers(sub) -> None:
    """Register the SFINCS-compatibility commands on a subparsers object.

    plan.md section 5.6 keeps SFINCS-specific operations in a compatibility
    group rather than as unrelated top-level commands. This is called twice:
    once to build ``dkx sfincs <command>``, and once at the top level, where
    every command is registered with a suppressed help string so the old
    spellings keep working for existing scripts without appearing in ``dkx
    --help``.

    Registering the same commands twice creates two independent parser
    objects, which is what argparse requires; they share the handlers.
    """
    p_solve = sub.add_parser("solve-v3", help="Solve a supported v3 linear problem matrix-free and write stateVector.npy.")
    _add_common_cli_args(p_solve)
    _add_parallel_cli_args(p_solve)
    p_solve.add_argument("--input", required=True, help="Path to input.namelist")
    p_solve.add_argument("--out-state", default="stateVector.npy", help="Where to write the solution vector (NumPy .npy)")
    p_solve.add_argument("--tol", default="1e-10", help="GMRES relative tolerance")
    p_solve.add_argument("--atol", default="0.0", help="GMRES absolute tolerance")
    p_solve.add_argument("--restart", default="80", help="GMRES restart")
    p_solve.add_argument("--maxiter", default=None, help="GMRES maxiter (default: library default)")
    p_solve.add_argument(
        "--solve-method",
        default="auto",
        help="Advanced solver override. Default 'auto' is recommended for normal runs; see docs/usage.rst.",
    )
    p_solve.add_argument(
        "--which-rhs",
        default=None,
        help="For RHSMode=2/3 transport-matrix runs, select whichRHS (v3 loops over multiple RHS).",
    )
    _add_equilibrium_override_args(p_solve)
    p_solve.set_defaults(func=_cmd_solve_v3)

    p_scan = sub.add_parser(
        "scan-er",
        help="Run an Er (or dPhiHatd*) scan by writing sfincsOutput.h5 in multiple run directories.",
    )
    _add_common_cli_args(p_scan)
    _add_parallel_cli_args(p_scan)
    p_scan.add_argument("--input", required=True, help="Path to input.namelist (template).")
    p_scan.add_argument("--out-dir", required=True, help="Directory to create scan subdirectories inside.")
    p_scan.add_argument(
        "--compute-transport-matrix",
        action="store_true",
        help="Also compute RHSMode=2/3 transport-matrix outputs (slow).",
    )
    p_scan.add_argument(
        "--compute-solution",
        action="store_true",
        help="For RHSMode=1 runs, also solve and write solution-derived fields (may be slow).",
    )
    p_scan.add_argument("--min", default="-1.0", help="Minimum value (ignored if --values is provided).")
    p_scan.add_argument("--max", default="1.0", help="Maximum value (ignored if --values is provided).")
    p_scan.add_argument("--n", default="5", help="Number of points (ignored if --values is provided).")
    p_scan.add_argument("--values", default=None, nargs="+", help="Explicit list of values to use.")
    p_scan.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing sfincsOutput.h5 files and only solve missing scan points.",
    )
    p_scan.add_argument("--jobs", type=int, default=1, help="Parallel worker processes for scan points.")
    p_scan.add_argument("--index", type=int, default=None, help="Optional job-array index (0-based).")
    p_scan.add_argument("--stride", type=int, default=1, help="Stride for job-array slicing.")
    p_scan.add_argument(
        "--solve-method",
        default="auto",
        help="Advanced RHSMode=1 solution solve override for --compute-solution scan points.",
    )
    p_scan.set_defaults(func=_cmd_scan_er)

    p_ambi = sub.add_parser(
        "ambipolar-solve",
        help="Given an existing scan-er directory, solve for ambipolar Er roots and write ambipolarSolutions.dat.",
    )
    _add_common_cli_args(p_ambi)
    _add_parallel_cli_args(p_ambi)
    p_ambi.add_argument("--scan-dir", required=True, help="Scan directory produced by `dkx scan-er`.")
    p_ambi.add_argument("--n-fine", default="500", help="Number of fine-grid points for bracketing (default: 500).")
    p_ambi.set_defaults(func=_cmd_ambipolar_solve)

    p_ambi_direct = sub.add_parser(
        "ambipolar",
        help="Run an in-process Brent ambipolar Er solve from input.namelist.",
    )
    _add_common_cli_args(p_ambi_direct)
    _add_parallel_cli_args(p_ambi_direct)
    p_ambi_direct.add_argument("--input", required=True, help="Path to input.namelist.")
    p_ambi_direct.add_argument(
        "--out-dir",
        default="ambipolar_run",
        help="Directory for per-evaluation inputs, outputs, traces, and the summary JSON.",
    )
    p_ambi_direct.add_argument("--er-min", default="-100.0", help="Lower Er bracket.")
    p_ambi_direct.add_argument("--er-max", default="100.0", help="Upper Er bracket.")
    p_ambi_direct.add_argument("--er-initial", default="0.0", help="Initial Er evaluation.")
    p_ambi_direct.add_argument("--max-evaluations", default="20", help="Maximum Brent radial-current evaluations.")
    p_ambi_direct.add_argument("--current-tolerance", default="1e-10", help="Radial-current convergence tolerance.")
    p_ambi_direct.add_argument("--step-tolerance", default="1e-8", help="Reserved Er-step tolerance for Newton-compatible APIs.")
    p_ambi_direct.add_argument(
        "--solve-method",
        default="auto",
        help="Advanced RHSMode=1 solver override for each radial-current evaluation.",
    )
    p_ambi_direct.add_argument(
        "--summary-json",
        default=None,
        help="Optional summary JSON path. Default: <out-dir>/ambipolar_result.json.",
    )
    p_ambi_direct.add_argument(
        "--no-output-cache",
        action="store_true",
        help="Disable the per-run geometry/output cache used across Er evaluations.",
    )
    p_ambi_direct.add_argument(
        "--no-solver-state",
        action="store_true",
        help="Disable shape-checked Krylov state reuse across Er evaluations.",
    )
    p_ambi_direct.set_defaults(func=_cmd_ambipolar)

    p_run = sub.add_parser("run-fortran", help="Run the compiled Fortran SFINCS v3 executable.")
    _add_common_cli_args(p_run)
    _add_parallel_cli_args(p_run)
    p_run.add_argument("--input", required=True, help="Path to input.namelist")
    p_run.add_argument("--exe", default=None, help="Path to Fortran v3 sfincs executable")
    p_run.add_argument("--workdir", default=None, help="Directory to run in (default: temp dir)")
    p_run.set_defaults(func=_cmd_run_fortran)

    p_out = sub.add_parser(
        "write-output",
        help="Write a SFINCS output file; the --out suffix selects HDF5, NetCDF4, or NPZ.",
    )
    _add_common_cli_args(p_out)
    _add_parallel_cli_args(p_out)
    p_out.add_argument("--input", required=True, help="Path to input.namelist")
    p_out.add_argument(
        "--out",
        default="sfincsOutput.h5",
        help="Output path. Suffix selects format: .h5/.hdf5, .nc/.netcdf, or .npz.",
    )
    p_out.add_argument(
        "--no-fortran-layout",
        dest="fortran_layout",
        action="store_false",
        default=True,
        help="Disable Fortran-compatible array layout (not recommended for parity)",
    )
    p_out.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        default=True,
        help="Fail if output already exists",
    )
    p_out.add_argument(
        "--compute-transport-matrix",
        action="store_true",
        help="Force transport-matrix solves for RHSMode=2/3 (default: enabled when RHSMode=2/3).",
    )
    p_out.add_argument(
        "--compute-solution",
        action="store_true",
        help="Force RHSMode=1 solves (default: enabled when RHSMode=1).",
    )
    p_out.add_argument(
        "--geometry-only",
        action="store_true",
        help="Only write geometry/grid outputs (skip RHSMode=1 solve and RHSMode=2/3 transport-matrix loop).",
    )
    p_out.add_argument(
        "--solver-trace",
        default=None,
        help="Optional JSON sidecar path for solver/backend/timing metadata.",
    )
    p_out.add_argument(
        "--solve-method",
        default="auto",
        help="Advanced RHSMode=1 solver override. Default 'auto' is recommended for normal runs; see docs/usage.rst.",
    )
    _add_equilibrium_override_args(p_out)
    p_out.set_defaults(func=_cmd_write_output)

    p_tm = sub.add_parser(
        "transport-matrix-v3",
        help="Solve RHSMode=2/3 transport-matrix systems on the canonical stack and write transportMatrix.npy.",
    )
    _add_common_cli_args(p_tm)
    _add_parallel_cli_args(p_tm)
    p_tm.add_argument("--input", required=True, help="Path to input.namelist (must have RHSMode=2 or 3)")
    p_tm.add_argument("--out-matrix", default="transportMatrix.npy", help="Where to write the transport matrix (NumPy .npy)")
    p_tm.add_argument(
        "--out",
        default=None,
        help="Optional sfincsOutput file (.h5/.hdf5 or .nc/.netcdf) written by the canonical writer.",
    )
    p_tm.add_argument(
        "--out-state-prefix",
        default=None,
        help="Optional prefix for saving solution vectors as <prefix>.whichRHS{k}.npy",
    )
    p_tm.add_argument("--tol", default="1e-10", help="Relative residual tolerance per whichRHS column")
    p_tm.add_argument(
        "--solve-method",
        default="auto",
        help="Advanced solver-route override (dkx.solve). Default 'auto' is recommended; see docs/usage.rst.",
    )
    _add_equilibrium_override_args(p_tm)
    p_tm.set_defaults(func=_cmd_transport_matrix_v3)

    p_mono = sub.add_parser(
        "monoenergetic-database",
        help="Scan (nuPrime, EStar) monoenergetic transport coefficients and write a .npz database.",
    )
    _add_common_cli_args(p_mono)
    p_mono.add_argument("--input", required=True, help="Path to input.namelist (geometry/resolution deck)")
    p_mono.add_argument(
        "--nu-prime", required=True, nargs="+", help="nuPrime scan values (nonzero)", metavar="NU"
    )
    p_mono.add_argument(
        "--e-star", nargs="+", default=["0.0"], help="EStar scan values (default: 0.0)", metavar="ESTAR"
    )
    p_mono.add_argument("--out", default="monoenergeticDatabase.npz", help="Output .npz database path")
    p_mono.add_argument("--tol", default="1e-10", help="Relative residual tolerance per whichRHS column")
    p_mono.add_argument(
        "--solve-method",
        default="auto",
        help="Advanced solver-route override (dkx.solve). Default 'auto' is recommended.",
    )
    p_mono.set_defaults(func=_cmd_monoenergetic_database)

    p_dump = sub.add_parser("dump-h5", help="Dump SFINCS HDF5 output to JSON (small files only).")
    _add_common_cli_args(p_dump)
    _add_parallel_cli_args(p_dump)
    p_dump.add_argument("--sfincs-output", required=True, help="Path to sfincsOutput.h5")
    p_dump.add_argument("--out-json", required=True, help="Where to write JSON")
    p_dump.add_argument("--keys-only", action="store_true", help="Only print dataset names")
    p_dump.set_defaults(func=_cmd_dump_h5)

    p_plot = sub.add_parser("plot-output", help="Write a diagnostics PDF/figure panel from a SFINCS output file.")
    _add_common_cli_args(p_plot)
    _add_parallel_cli_args(p_plot)
    p_plot.add_argument("--input-h5", required=True, help="Path to sfincsOutput.h5/.nc/.npz")
    p_plot.add_argument(
        "--out",
        default=None,
        help="Where to write the diagnostics panel (default: <input>_summary.pdf next to the output file).",
    )
    p_plot.set_defaults(func=_cmd_plot_output)

    p_cmp = sub.add_parser("compare-h5", help="Compare two SFINCS HDF5 output files.")
    _add_common_cli_args(p_cmp)
    _add_parallel_cli_args(p_cmp)
    p_cmp.add_argument("--a", required=True, help="First sfincsOutput.h5")
    p_cmp.add_argument("--b", required=True, help="Second sfincsOutput.h5")
    p_cmp.add_argument("--rtol", default="1e-12")
    p_cmp.add_argument("--atol", default="1e-12")
    p_cmp.add_argument("--tolerances-json", default=None, help="Optional JSON file of per-key tolerances")
    p_cmp.add_argument("--show-all", action="store_true", help="Print all keys (not just failures)")
    p_cmp.set_defaults(func=_cmd_compare_h5)

    p_pp = sub.add_parser(
        "postprocess-upstream",
        help="Run a vendored upstream v3 utils/ postprocessing script (best-effort, requires sfincsOutput.h5).",
    )
    _add_common_cli_args(p_pp)
    _add_parallel_cli_args(p_pp)
    p_pp.add_argument("--case-dir", required=True, help="Directory containing sfincsOutput.h5")
    p_pp.add_argument("--util", required=True, help="Upstream util script name (e.g. sfincsScanPlot_1)")
    p_pp.add_argument("--utils-dir", default=None, help="Override utils/ directory (else auto-detect / env var)")
    p_pp.add_argument("--interactive", action="store_true", help="Do not override input() (may hang in CI)")
    p_pp.add_argument(
        "util_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the upstream script (e.g. 'pdf'). Prefix with '--' to separate args.",
    )

    def _cmd_postprocess_upstream(args: argparse.Namespace) -> int:
        t0 = _now()
        from .workflows.scans import run_upstream_util  # noqa: PLC0415

        _emit("################################################################", level=0, args=args)
        _emit(" dkx postprocess-upstream", level=0, args=args)
        _emit(f" case_dir={Path(args.case_dir).resolve()}", level=0, args=args)
        _emit(f" util={args.util}", level=0, args=args)
        if args.utils_dir is not None:
            _emit(f" utils_dir={Path(args.utils_dir).resolve()}", level=1, args=args)
        util_args = list(args.util_args or [])
        if util_args and util_args[0] == "--":
            util_args = util_args[1:]
        run_upstream_util(
            util=str(args.util),
            case_dir=Path(args.case_dir),
            args=util_args,
            utils_dir=Path(args.utils_dir) if args.utils_dir is not None else None,
            noninteractive=not bool(args.interactive),
            emit=lambda level, msg: _emit(msg, level=level, args=args),
        )
        _emit(f" elapsed_s={_now()-t0:.3f}", level=1, args=args)
        return 0

    p_pp.set_defaults(func=_cmd_postprocess_upstream)


def main(argv: list[str] | None = None) -> int:
    """Run the dkx command-line interface."""
    # The CLI bootstrap, as plan.md section 6.4 requires: the runtime is applied
    # here, once, before anything imports the JAX backend. Doing it first is what
    # lets --cores reach XLA's threadpool, which reads NPROC when the CPU backend
    # initialises and never again.
    _runtime.configure()
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    rc = _maybe_handle_plot(argv)
    if rc is not None:
        return rc
    parser = argparse.ArgumentParser(prog="dkx")
    _add_common_cli_args(parser)
    _add_parallel_cli_args(parser)
    # metavar hides the compatibility aliases from the choice list. Without it
    # argparse prints all 21 names even though their help is suppressed.
    sub = parser.add_subparsers(
        dest="cmd", required=True, metavar="{" + ",".join(_USER_COMMANDS) + "}"
    )

    p_validate = sub.add_parser(
        "validate",
        help="Validate a versioned case file and print its deterministic ID.",
    )
    _add_common_cli_args(p_validate)
    _add_parallel_cli_args(p_validate)
    p_validate.add_argument("case", help="Path to a .toml or .json case file.")
    p_validate.set_defaults(func=_cmd_validate_case)

    p_doctor = sub.add_parser(
        "doctor",
        help="Check that this install can run, and say what is wrong when it cannot.",
    )
    _add_common_cli_args(p_doctor)
    _add_parallel_cli_args(p_doctor)
    p_doctor.add_argument("--format", choices=("table", "json"), default="table")
    p_doctor.set_defaults(func=_cmd_doctor)

    p_converge = sub.add_parser(
        "converge",
        help="Refine each phase-space axis of a case and report observable convergence.",
    )
    _add_common_cli_args(p_converge)
    _add_parallel_cli_args(p_converge)
    p_converge.add_argument("case", help="Path to a .toml or .json case file.")
    p_converge.add_argument(
        "--axes", nargs="+", default=list(_CONVERGE_AXES),
        choices=list(_CONVERGE_AXES),
        help="Phase-space axes to refine (default: all).",
    )
    p_converge.add_argument("--factor", type=float, default=1.5, help="Refinement factor per axis.")
    p_converge.add_argument("--tolerance", type=float, default=0.02, help="Relative-change tolerance.")
    p_converge.add_argument(
        "--no-joint", action="store_true",
        help="Skip the all-axes-together run. Faster, and unable to detect axis coupling.",
    )
    p_converge.add_argument("--format", choices=("table", "json"), default="table")
    p_converge.set_defaults(func=_cmd_converge)

    p_roots = sub.add_parser(
        "roots",
        help="Print the ambipolar root table and branch events stored in a result.",
    )
    _add_common_cli_args(p_roots)
    _add_parallel_cli_args(p_roots)
    p_roots.add_argument("result", help="Path to a NetCDF Result written by dkx run.")
    p_roots.add_argument("--format", choices=("table", "json"), default="table")
    p_roots.set_defaults(func=_cmd_roots)

    p_compare = sub.add_parser(
        "compare",
        help="Compare two results (dkx NetCDF or SFINCS HDF5) and exit non-zero if they differ.",
    )
    _add_common_cli_args(p_compare)
    _add_parallel_cli_args(p_compare)
    p_compare.add_argument("a")
    p_compare.add_argument("b")
    p_compare.add_argument("--rtol", type=float, default=1e-9)
    p_compare.add_argument("--atol", type=float, default=0.0)
    p_compare.add_argument("--verbose-keys", action="store_true",
                           help="List every array, not only the differing ones.")
    p_compare.add_argument("--format", choices=("table", "json"), default="table")
    p_compare.set_defaults(func=_cmd_compare)

    p_plot_result = sub.add_parser(
        "plot",
        help="Plot a result (dkx NetCDF radial profiles, or a SFINCS HDF5 panel).",
    )
    _add_common_cli_args(p_plot_result)
    _add_parallel_cli_args(p_plot_result)
    p_plot_result.add_argument("result")
    p_plot_result.add_argument("--out", default=None, help="Output image path (default: alongside the input).")
    p_plot_result.set_defaults(func=_cmd_plot)

    p_scan_case = sub.add_parser(
        "scan",
        help="Expand a case's [scan] axes, run every point, and write one Result.",
    )
    _add_common_cli_args(p_scan_case)
    _add_parallel_cli_args(p_scan_case)
    p_scan_case.add_argument("case", help="Path to a .toml or .json case file with a [scan] table.")
    p_scan_case.add_argument("--out", default=None, help="Output NetCDF path (default: scan.output).")
    p_scan_case.add_argument(
        "--no-resume", action="store_true",
        help="Rerun every point even if the output already holds its result.",
    )
    p_scan_case.set_defaults(func=_cmd_scan)

    p_schema = sub.add_parser(
        "schema",
        help="Print the complete case example or machine-readable JSON Schema.",
    )
    _add_common_cli_args(p_schema)
    _add_parallel_cli_args(p_schema)
    p_schema.add_argument("--format", choices=("toml", "json"), default="toml")
    p_schema.set_defaults(func=_cmd_schema)

    p_run = sub.add_parser(
        "run",
        help="Execute a case file and write its NetCDF Result.",
    )
    _add_common_cli_args(p_run)
    _add_parallel_cli_args(p_run)
    p_run.add_argument("case", help="Path to a .toml or .json case file.")
    p_run.add_argument(
        "--out",
        default=None,
        help="Write the NetCDF Result here. Omitted, nothing is saved.",
    )
    p_run.set_defaults(func=_cmd_run_case)

    p_inspect = sub.add_parser(
        "inspect",
        help="Print what a saved NetCDF Result contains.",
    )
    _add_common_cli_args(p_inspect)
    _add_parallel_cli_args(p_inspect)
    p_inspect.add_argument("result", help="Path to a DKX .nc result.")
    p_inspect.set_defaults(func=_cmd_inspect_result)

    p_convert = sub.add_parser(
        "convert",
        help="Convert a SFINCS input.namelist into a native .toml or .json case.",
    )
    _add_common_cli_args(p_convert)
    _add_parallel_cli_args(p_convert)
    p_convert.add_argument("source", help="Path to a SFINCS v3 input.namelist.")
    p_convert.add_argument(
        "destination",
        help="Case file to write; the .toml or .json extension picks the format.",
    )
    p_convert.add_argument(
        "--name", default=None, help="Case name (default: derived from SOURCE)."
    )
    p_convert.add_argument(
        "--force", action="store_true", help="Overwrite DESTINATION if it exists."
    )
    p_convert.set_defaults(func=_cmd_convert)

    # SFINCS-compatibility commands: visible under `dkx sfincs`, and still
    # accepted at the top level as hidden aliases so existing scripts run.
    p_sfincs = sub.add_parser(
        "sfincs",
        help="SFINCS v3 compatibility commands (namelist and HDF5 workflows).",
    )
    _add_compat_parsers(
        p_sfincs.add_subparsers(dest="sfincs_cmd", required=True)
    )
    _add_compat_parsers(_HiddenAliases(sub))


    argv = _normalize_default_argv(argv, set(sub.choices))
    _maybe_reexec_for_early_runtime(argv)
    args = parser.parse_args(argv)
    args = _merge_global_cli_args(argv, args)
    _apply_runtime_env_defaults()
    # No CLI-side default core count: when --cores/DKX_CORES is absent the
    # package import already clamped the threadpool to min(8, cpu_count).
    _apply_cores_setting(args.cores)
    _apply_parallel_runtime_settings(args)
    if args.fortran_stdout is True:
        os.environ["DKX_FORTRAN_STDOUT"] = "1"
    elif args.fortran_stdout is False:
        os.environ["DKX_FORTRAN_STDOUT"] = "0"
    else:
        os.environ.setdefault("DKX_FORTRAN_STDOUT", "1" if not getattr(args, "quiet", False) else "0")
    return int(args.func(args))

if __name__ == "__main__":
    raise SystemExit(main())
