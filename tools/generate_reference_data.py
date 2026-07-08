#!/usr/bin/env python3
"""Generate golden reference data from the Fortran SFINCS v3 binary.

Runs a matrix of SFINCS example cases (optionally with resolution overrides),
capturing for each run:

* ``stdout.log``     — the full program output (print-parity reference),
* ``sfincsOutput.h5``— the HDF5 output (field-parity reference),
* ``input.namelist`` — the exact input used (after any overrides),
* timing and peak-RSS metadata in a global ``manifest.json``.

The result directory is meant to be tarballed and uploaded as a GitHub release
asset (e.g. ``reference-data-v2``); nothing produced here should be committed.

Usage (from the sfincs_jax repo root, with the Fortran binary built)::

    python tools/generate_reference_data.py \
        --binary  ~/local/sfincs/fortran/version3/sfincs \
        --examples-dir ~/local/sfincs/fortran/version3/examples \
        --out-dir /tmp/reference-data-v2 \
        --mpi-n 1

Resolution variants: for cases listed in ``SCALED_CASES`` the script also runs
"med" and "high" variants with Ntheta/Nzeta/Nxi/Nx scaled by the factors in
``RESOLUTION_VARIANTS``, in addition to the as-shipped "low" variant.
"""

from __future__ import annotations

import argparse
import json
import re
import resource
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

# Cases whose as-shipped resolution is deliberately tiny; we add scaled
# variants to probe convergence and solver behaviour at realistic grids.
SCALED_CASES = [
    "quick_2species_FPCollisions_noEr",
    "filteredW7XNetCDF_2species_noEr",
    "tokamak_1species_FPCollisions_noEr",
    "HSX_FPCollisions_DKESTrajectories",
    "geometryScheme4_2species_noEr",
]

# name -> multiplier applied to Ntheta, Nzeta, Nxi (Nx gets the sqrt of it,
# rounded, since speed resolution converges much faster).
RESOLUTION_VARIANTS = {"low": 1.0, "med": 2.0, "high": 4.0}

_RES_KEYS = ("Ntheta", "Nzeta", "Nxi", "Nx")


def scale_namelist(text: str, factor: float) -> str:
    """Scale the resolution parameters in a namelist by ``factor``.

    Only touches whole-line assignments like ``Ntheta = 15`` inside the file;
    keeps everything else (comments, !ss directives) byte-identical.
    """
    if factor == 1.0:
        return text

    def repl(match: re.Match) -> str:
        key, value = match.group(1), int(match.group(2))
        if key == "Nzeta" and value == 1:
            return match.group(0)  # axisymmetric: Nzeta=1 is a special case
        f = factor ** 0.5 if key == "Nx" else factor
        new = max(value + 1, int(round(value * f)))
        if key in ("Ntheta", "Nzeta") and new % 2 == 0:
            new += 1  # forceOddNthetaAndNzeta would bump these anyway
        return f"{match.group(0).split('=')[0]}= {new}"

    pattern = re.compile(
        r"^\s*(%s)\s*=\s*(\d+)\s*$" % "|".join(_RES_KEYS), re.MULTILINE
    )
    return pattern.sub(repl, text)


def run_case(
    binary: Path,
    case_dir: Path,
    out_dir: Path,
    variant: str,
    factor: float,
    mpi_n: int,
    extra_args: list[str],
    timeout: float,
) -> dict:
    """Run one (case, variant) and return its manifest entry."""
    name = case_dir.name if variant == "low" else f"{case_dir.name}__{variant}"
    work = out_dir / name
    work.mkdir(parents=True, exist_ok=True)

    namelist = scale_namelist((case_dir / "input.namelist").read_text(), factor)
    # Examples reference equilibria relative to their own directory; the run
    # happens elsewhere, so rewrite those paths to absolute ones.
    namelist = re.sub(
        r'(equilibriumFile\s*=\s*")([^"]+)(")',
        lambda m: m.group(1)
        + str((case_dir / m.group(2)).resolve())
        + m.group(3),
        namelist,
    )
    (work / "input.namelist").write_text(namelist)

    cmd = ["mpiexec", "-n", str(mpi_n), str(binary), *extra_args]
    entry = {"case": case_dir.name, "variant": variant, "command": " ".join(cmd)}

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd, cwd=work, capture_output=True, text=True, timeout=timeout
        )
        entry["returncode"] = proc.returncode
        (work / "stdout.log").write_text(proc.stdout)
        if proc.stderr:
            (work / "stderr.log").write_text(proc.stderr)
    except subprocess.TimeoutExpired:
        entry["returncode"] = None
        entry["error"] = f"timeout after {timeout}s"
    entry["wall_seconds"] = round(time.perf_counter() - t0, 3)
    # Peak RSS of the largest child seen so far (cumulative across cases, so
    # only meaningful per-case when cases run in increasing size order).
    entry["ru_maxrss_children"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss

    h5 = work / "sfincsOutput.h5"
    entry["has_h5"] = h5.is_file()
    if entry["has_h5"]:
        entry["h5_bytes"] = h5.stat().st_size
    return entry


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--examples-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--mpi-n", type=int, default=1)
    ap.add_argument("--cases", nargs="*", default=None,
                    help="subset of example names (default: all with tests.py)")
    ap.add_argument("--skip-scaled", action="store_true",
                    help="only run as-shipped resolutions")
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--tar", action="store_true",
                    help="write <out-dir>.tar.gz when done")
    ap.add_argument("--extra-args", nargs="*", default=[],
                    help="extra CLI args passed to sfincs (e.g. -log_view)")
    args = ap.parse_args()

    binary = args.binary.resolve()
    if not binary.is_file():
        sys.exit(f"error: sfincs binary not found at {binary}")

    case_dirs = sorted(
        d for d in args.examples_dir.iterdir()
        if d.is_dir() and (d / "tests.py").is_file() and (d / "input.namelist").is_file()
    )
    if args.cases:
        wanted = set(args.cases)
        case_dirs = [d for d in case_dirs if d.name in wanted]
        missing = wanted - {d.name for d in case_dirs}
        if missing:
            sys.exit(f"error: unknown cases: {sorted(missing)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "binary": str(binary),
        "mpi_n": args.mpi_n,
        "generated_unix_time": time.time(),
        "runs": [],
    }

    for case_dir in case_dirs:
        variants = (
            RESOLUTION_VARIANTS
            if (case_dir.name in SCALED_CASES and not args.skip_scaled)
            else {"low": 1.0}
        )
        for variant, factor in variants.items():
            print(f"=== {case_dir.name} [{variant}] ===", flush=True)
            entry = run_case(
                binary, case_dir, args.out_dir, variant, factor,
                args.mpi_n, args.extra_args, args.timeout,
            )
            manifest["runs"].append(entry)
            status = "ok" if entry.get("returncode") == 0 and entry["has_h5"] else "FAILED"
            print(f"    {status}  {entry['wall_seconds']}s", flush=True)

    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    n_ok = sum(1 for r in manifest["runs"] if r.get("returncode") == 0 and r["has_h5"])
    print(f"\n{n_ok}/{len(manifest['runs'])} runs succeeded -> {args.out_dir}")

    if args.tar:
        tar_path = args.out_dir.with_suffix(".tar.gz")
        with tarfile.open(tar_path, "w:gz") as tf:
            tf.add(args.out_dir, arcname=args.out_dir.name)
        print(f"wrote {tar_path} ({tar_path.stat().st_size / 1e6:.1f} MB)")

    return 0 if n_ok == len(manifest["runs"]) else 1


if __name__ == "__main__":
    sys.exit(main())
