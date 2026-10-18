from __future__ import annotations

import argparse
import os
import sys

from .cli_parsers import _add_common_cli_args, _add_parallel_cli_args


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
        # --density-m3 pins the on-axis density instead of scaling it from the
        # equilibrium's own pressure. A flag rather than an environment
        # variable: the plasma a run assumed should be visible in the command
        # that produced it, not in the shell that happened to launch it.
        from dkx.representative import set_axis_density_override  # noqa: PLC0415

        try:
            set_axis_density_override(_density_override(argv))
        except ValueError as exc:
            print(f"dkx: {exc}", file=sys.stderr)
            return 2
        print(f" dkx {path.name} — representative run{label}")
        print(f" wrote {run_representative(path, out_path=out, full='--full' in argv, quick=quick)}")
    return 0


def _density_override(argv: list[str]) -> float | None:
    """``--density-m3 VALUE`` from the raw argv, or ``None`` when absent.

    Read here rather than through argparse because the representative run is
    reached before the subcommand parser exists -- ``dkx wout.nc`` names a file,
    not a command.
    """
    for form in ("--density-m3", "--density"):
        if form in argv:
            index = argv.index(form)
            if index + 1 >= len(argv):
                raise ValueError(f"{form} needs a value in m^-3, e.g. {form} 2.38e20")
            raw = argv[index + 1]
            try:
                return float(raw)
            except ValueError:
                raise ValueError(f"{form}={raw!r} is not a number; expected m^-3") from None
        prefix = f"{form}="
        for token in argv:
            if token.startswith(prefix):
                raw = token[len(prefix):]
                try:
                    return float(raw)
                except ValueError:
                    raise ValueError(f"{form}={raw!r} is not a number; expected m^-3") from None
    return None


#: Refinement axes, mirrored from dkx.workflows.converge.AXES so building the
#: parser does not import the execution stack. The test suite pins them equal.
_CONVERGE_AXES: tuple[str, ...] = ("theta", "zeta", "pitch", "speed")


#: The commands `dkx --help` advertises, in the order plan.md section 5.6 lists
#: them. The SFINCS commands are reachable as `dkx sfincs <command>` and as
#: hidden top-level aliases, but are deliberately absent here: listing 21
#: choices is what the compatibility group exists to avoid.
_USER_COMMANDS: tuple[str, ...] = (
    "doctor", "template", "validate", "run", "roots", "converge", "inspect",
    "compare", "plot", "scan", "convert", "sfincs",
)


#: Every registered subcommand name. ``main`` does not read this -- it passes
#: the parser's own ``sub.choices`` -- so the runtime behaviour cannot drift
#: from the registered set. It exists for direct callers and as documentation.
_KNOWN_COMMANDS: frozenset[str] = frozenset({
    "validate", "doctor", "converge", "roots", "compare", "plot", "scan", "convert", "template", "schema", "run", "inspect", "solve-v3", "ambipolar",
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

