#!/usr/bin/env python

"""Run dkx and write a SFINCS-style sfincsOutput.h5.

This module is the shared execution backend for the utils scripts. It defaults
to reading ``input.namelist`` in the current directory and writing a local
``sfincsOutput.h5``. Use ``--input`` and ``--out`` to target other paths, or
import ``run_dkx`` from Python to embed it in custom workflows.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dkx.api import write_output  # noqa: E402
from dkx.io import localize_equilibrium_file_in_place, _resolve_equilibrium_file_from_namelist  # noqa: E402
from dkx.input_compat import effective_equilibrium_file  # noqa: E402
from dkx.namelist import read_sfincs_input  # noqa: E402


def output_is_complete(path: Path) -> bool:
    """Execution/retry gate, not a residual or resolution certificate."""
    import h5py
    import numpy as np

    try:
        with h5py.File(path, "r") as f:
            mode = int(f["RHSMode"][()].item())
            true = int(f["integerToRepresentTrue"][()].item())
            if true == 0 or int(f["finished"][()].item()) != true:
                return False
            if "includePhi1" in f and int(f["includePhi1"][()].item()) == true:
                if "didNonlinearCalculationConverge" not in f:
                    return False
            if "didNonlinearCalculationConverge" in f:
                if np.asarray(f["didNonlinearCalculationConverge"]).ravel()[-1] != true:
                    return False
            keys = (("FSABFlow", "FSABjHat", "particleFlux_vm_psiHat", "heatFlux_vm_psiHat")
                    if mode == 1 else ("transportMatrix",) if mode in (2, 3) else ())
            if not keys:
                return False
            for key in keys:
                values = np.asarray(f[key])
                if not values.size or not np.isfinite(values).all():
                    return False
                if key == "transportMatrix" and values.shape != ((3, 3) if mode == 2 else (2, 2)):
                    return False
            return True
    except (OSError, KeyError, TypeError, ValueError, IndexError, AttributeError):
        return False


def output_matches_input(path: Path, requested_text: str, source_path: Path) -> bool:
    """Compare the output's embedded deck, not a possibly edited input file."""
    import hashlib
    import h5py
    from dkx.namelist import parse_sfincs_input_text
    from dkx.input_compat import effective_equilibrium_file, _resolve_equilibrium_file_from_namelist

    try:
        with h5py.File(path, "r") as f:
            saved_text = f["input.namelist"].asstr()[()]
        old = parse_sfincs_input_text(saved_text, source_path=path.parent / "input.namelist")
        new = parse_sfincs_input_text(requested_text, source_path=source_path)
        for nml in (old, new):
            geom = nml.group("geometryParameters")
            if effective_equilibrium_file(geom_params=geom) is not None:
                equilibrium = _resolve_equilibrium_file_from_namelist(nml=nml)
                digest = hashlib.sha256(equilibrium.read_bytes()).hexdigest()
                for key in ("EQUILIBRIUMFILE", "FORT996BOOZER_FILE", "JGBOOZER_FILE", "JGBOOZER_FILE_NONSTELSYM"):
                    if key in geom:
                        geom[key] = digest
        return old.groups == new.groups and old.indexed == new.indexed
    except (OSError, KeyError, TypeError, ValueError, AttributeError):
        return False


def run_dkx(
    *,
    input_namelist: Path,
    output_path: Optional[Path] = None,
    compute_transport_matrix: Optional[bool] = None,
    compute_solution: Optional[bool] = None,
    overwrite: bool = True,
    verbose: bool = True,
    ensure_equilibrium: bool = True,
    differentiable: bool = False,
) -> Path:
    input_namelist = Path(input_namelist).resolve()
    if output_path is None:
        output_path = input_namelist.parent / "sfincsOutput.h5"

    nml = read_sfincs_input(input_namelist)
    rhs_mode = int(nml.group("general").get("RHSMODE", 1))

    if compute_transport_matrix is None:
        compute_transport_matrix = rhs_mode in {2, 3}
    if compute_solution is None:
        compute_solution = rhs_mode == 1

    if ensure_equilibrium:
        if effective_equilibrium_file(geom_params=nml.group("geometryParameters")) is not None:
            source = _resolve_equilibrium_file_from_namelist(nml=nml)
            local = input_namelist.parent / source.name
            if local.exists() and local.resolve() != source.resolve() and local.read_bytes() != source.read_bytes():
                raise RuntimeError(
                    f"Conflicting localized equilibrium {local}; preserve it and use a fresh run directory."
                )
        localize_equilibrium_file_in_place(input_namelist=input_namelist, overwrite=False)

    if verbose:
        print(
            "dkx_driver: start "
            f"input={input_namelist.name} output={Path(output_path).name} "
            f"rhs_mode={rhs_mode} compute_solution={bool(compute_solution)} "
            f"compute_transport_matrix={bool(compute_transport_matrix)} "
            f"differentiable={bool(differentiable)}",
            flush=True,
        )
    t0 = time.perf_counter()
    out = write_output(
        input_namelist,
        output_path,
        overwrite=overwrite,
        emit=print if verbose else None,
    )
    if not output_is_complete(Path(out)):
        raise RuntimeError(f"Incomplete or unsuccessful scan output: {out}")
    if verbose:
        print(
            f"dkx_driver: done output={Path(out).name} elapsed_s={time.perf_counter() - t0:.3f}",
            flush=True,
        )
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="dkx_driver",
        description="Run dkx to write a SFINCS-style sfincsOutput.h5.",
    )
    parser.add_argument("--input", default="input.namelist", help="Path to input.namelist.")
    parser.add_argument("--out", default="sfincsOutput.h5", help="Output sfincsOutput.h5 path.")
    parser.add_argument(
        "--transport",
        action="store_true",
        help="Force transport-matrix solve (RHSMode=2/3).",
    )
    parser.add_argument(
        "--solution",
        action="store_true",
        help="Force RHSMode=1 solve and diagnostics.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Do not overwrite existing output file.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose sfincs-style logging.",
    )
    parser.add_argument(
        "--differentiable",
        action="store_true",
        help=(
            "Use the implicit/differentiable linear-solve path. The default utility "
            "path is explicit and performance-oriented for scans and parity runs."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input)
    output_path = Path(args.out)

    compute_transport = True if args.transport else None
    compute_solution = True if args.solution else None

    run_dkx(
        input_namelist=input_path,
        output_path=output_path,
        compute_transport_matrix=compute_transport,
        compute_solution=compute_solution,
        overwrite=not args.no_overwrite,
        verbose=not args.quiet,
        differentiable=bool(args.differentiable),
    )


if __name__ == "__main__":
    main()
