from __future__ import annotations

import argparse
from pathlib import Path


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
    from . import cli as _cli  # noqa: PLC0415

    p_solve = sub.add_parser("solve-v3", help="Solve a supported v3 linear problem matrix-free and write stateVector.npy.")
    _add_common_cli_args(p_solve)
    _add_parallel_cli_args(p_solve)
    p_solve.add_argument("--input", required=True, help="Path to input.namelist")
    p_solve.add_argument("--out-state", default="stateVector.npy", help="Where to write the solution vector (NumPy .npy)")
    p_solve.add_argument("--tol", default="1e-10", help="GMRES relative tolerance")
    p_solve.add_argument("--atol", default="0.0", help="GMRES absolute tolerance")
    p_solve.add_argument(
        "--restart",
        default="auto",
        help="accepted for compatibility and not applied: the recycled Krylov "
        "route uses the library's memory-aware restart policy",
    )
    p_solve.add_argument("--maxiter", default=None, help="GMRES maxiter (default: library default)")
    p_solve.add_argument(
        "--solve-method",
        default="auto",
        help="Advanced solver override. Default 'auto' is recommended for normal runs; see docs/numerics/solver_routes.md.",
    )
    p_solve.add_argument(
        "--which-rhs",
        default=None,
        help="For RHSMode=2/3 transport-matrix runs, select whichRHS (v3 loops over multiple RHS).",
    )
    _add_equilibrium_override_args(p_solve)
    p_solve.set_defaults(func=_cli._cmd_solve_v3)

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
    p_scan.set_defaults(func=_cli._cmd_scan_er)

    p_prof = sub.add_parser(
        "scan-profile",
        help="Radius x Er scan of a deck from a profile file (upstream sfincsScan scanType 5).",
    )
    _add_common_cli_args(p_prof)
    _add_parallel_cli_args(p_prof)
    p_prof.add_argument("--input", required=True, help="Path to input.namelist; its '!ss' lines give Nradius, rN_min, rN_max.")
    p_prof.add_argument("--out-dir", required=True, help="Directory for the rN<radius>/Er<value> run directories.")
    p_prof.add_argument("--profiles", default=None, help="Profile file (default: the one profile* file beside the deck).")
    p_prof.add_argument("--radii", default=None, nargs="+", help="Explicit rN values (default: the '!ss' radii).")
    p_prof.add_argument("--er-min", default=None, help="Er range minimum (default: the profile file's Ermin).")
    p_prof.add_argument("--er-max", default=None, help="Er range maximum (default: the profile file's Ermax).")
    p_prof.add_argument("--n-er", default=None, help="Er points per radius (default: the profile file's nErs).")
    p_prof.add_argument("--skip-existing", action="store_true", help="Reuse existing sfincsOutput.h5 files.")
    p_prof.add_argument("--jobs", type=int, default=1, help="Parallel worker processes per radius.")
    p_prof.add_argument("--solve-method", default="auto", help="Solve method for every point.")
    p_prof.set_defaults(func=_cli._cmd_scan_profile)

    p_ambi = sub.add_parser(
        "ambipolar-solve",
        help="Given an existing scan-er directory, solve for ambipolar Er roots and write ambipolarSolutions.dat.",
    )
    _add_common_cli_args(p_ambi)
    _add_parallel_cli_args(p_ambi)
    p_ambi.add_argument("--scan-dir", required=True, help="Scan directory produced by `dkx scan-er`.")
    p_ambi.add_argument("--n-fine", default="500", help="Number of fine-grid points for bracketing (default: 500).")
    p_ambi.set_defaults(func=_cli._cmd_ambipolar_solve)

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
    p_ambi_direct.add_argument("--current-tolerance", default="1e-6", help="Radial-current tolerance, relative to the largest |J_r| seen while bracketing.")
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
    p_ambi_direct.set_defaults(func=_cli._cmd_ambipolar)

    p_run = sub.add_parser("run-fortran", help="Run the compiled Fortran SFINCS v3 executable.")
    _add_common_cli_args(p_run)
    _add_parallel_cli_args(p_run)
    p_run.add_argument("--input", required=True, help="Path to input.namelist")
    p_run.add_argument("--exe", default=None, help="Path to Fortran v3 sfincs executable")
    p_run.add_argument("--workdir", default=None, help="Directory to run in (default: temp dir)")
    p_run.set_defaults(func=_cli._cmd_run_fortran)

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
        help="Advanced RHSMode=1 solver override. Default 'auto' is recommended for normal runs; see docs/numerics/solver_routes.md.",
    )
    _add_equilibrium_override_args(p_out)
    p_out.set_defaults(func=_cli._cmd_write_output)

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
        help="Advanced solver-route override (dkx.solve). Default 'auto' is recommended; see docs/numerics/solver_routes.md.",
    )
    _add_equilibrium_override_args(p_tm)
    p_tm.set_defaults(func=_cli._cmd_transport_matrix_v3)

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
    p_mono.set_defaults(func=_cli._cmd_monoenergetic_database)

    p_dump = sub.add_parser("dump-h5", help="Dump SFINCS HDF5 output to JSON (small files only).")
    _add_common_cli_args(p_dump)
    _add_parallel_cli_args(p_dump)
    p_dump.add_argument("--sfincs-output", required=True, help="Path to sfincsOutput.h5")
    p_dump.add_argument("--out-json", required=True, help="Where to write JSON")
    p_dump.add_argument("--keys-only", action="store_true", help="Only print dataset names")
    p_dump.set_defaults(func=_cli._cmd_dump_h5)

    p_plot = sub.add_parser("plot-output", help="Write a diagnostics PDF/figure panel from a SFINCS output file.")
    _add_common_cli_args(p_plot)
    _add_parallel_cli_args(p_plot)
    p_plot.add_argument("--input-h5", required=True, help="Path to sfincsOutput.h5/.nc/.npz")
    p_plot.add_argument(
        "--out",
        default=None,
        help="Where to write the diagnostics panel (default: <input>_summary.pdf next to the output file).",
    )
    p_plot.set_defaults(func=_cli._cmd_plot_output)

    p_cmp = sub.add_parser("compare-h5", help="Compare two SFINCS HDF5 output files.")
    _add_common_cli_args(p_cmp)
    _add_parallel_cli_args(p_cmp)
    p_cmp.add_argument("--a", required=True, help="First sfincsOutput.h5")
    p_cmp.add_argument("--b", required=True, help="Second sfincsOutput.h5")
    p_cmp.add_argument("--rtol", default="1e-12")
    p_cmp.add_argument("--atol", default="1e-12")
    p_cmp.add_argument("--tolerances-json", default=None, help="Optional JSON file of per-key tolerances")
    p_cmp.add_argument("--show-all", action="store_true", help="Print all keys (not just failures)")
    p_cmp.set_defaults(func=_cli._cmd_compare_h5)

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
        t0 = _cli._now()
        from .workflows.scans import run_upstream_util  # noqa: PLC0415

        _cli._emit("################################################################", level=0, args=args)
        _cli._emit(" dkx postprocess-upstream", level=0, args=args)
        _cli._emit(f" case_dir={Path(args.case_dir).resolve()}", level=0, args=args)
        _cli._emit(f" util={args.util}", level=0, args=args)
        if args.utils_dir is not None:
            _cli._emit(f" utils_dir={Path(args.utils_dir).resolve()}", level=1, args=args)
        util_args = list(args.util_args or [])
        if util_args and util_args[0] == "--":
            util_args = util_args[1:]
        run_upstream_util(
            util=str(args.util),
            case_dir=Path(args.case_dir),
            args=util_args,
            utils_dir=Path(args.utils_dir) if args.utils_dir is not None else None,
            noninteractive=not bool(args.interactive),
            emit=lambda level, msg: _cli._emit(msg, level=level, args=args),
        )
        _cli._emit(f" elapsed_s={_cli._now()-t0:.3f}", level=1, args=args)
        return 0

    p_pp.set_defaults(func=_cmd_postprocess_upstream)


