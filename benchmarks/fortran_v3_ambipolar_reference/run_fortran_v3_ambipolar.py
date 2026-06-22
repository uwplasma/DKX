#!/usr/bin/env python3
"""Run and summarize SFINCS Fortran v3 ambipolar reference decks.

The script keeps verbose Fortran/MUMPS logs in a scratch directory and writes a
compact JSON summary suitable for plan/docs updates. It is intentionally
stdlib-only so it can run in a clean developer checkout.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FORTRAN = Path("/Users/rogeriojorge/local/sfincs/fortran/version3/sfincs")
NAMELIST_DIR = Path(__file__).resolve().parent / "namelists"


def _parse_vector_after(label: str, text: str) -> list[float]:
    idx = text.rfind(label)
    if idx < 0:
        return []
    tail = text[idx + len(label) :].splitlines()
    values: list[float] = []
    for line in tail:
        if not line.strip():
            continue
        if re.match(r"^\s*[-+0-9.Eed\s]+$", line):
            values.extend(float(tok.replace("d", "e").replace("D", "E")) for tok in line.split())
            if values:
                return values
        elif values:
            return values
    return values


def parse_fortran_log(path: Path) -> dict[str, object]:
    text = path.read_text(errors="replace")
    solve_times = [
        float(v.replace("D", "E"))
        for v in re.findall(r"Done with the main solve\.\s+Time to solve:\s*([-+0-9.EeDd]+)", text)
    ]
    adjoint_times = [float(v.replace("D", "E")) for v in re.findall(r"Done with the adjoint solve\.\s+Time to solve:\s*([-+0-9.EeDd]+)", text)]
    pmat_nnz = [int(v) for v in re.findall(r"# of nonzeros in Jacobian preconditioner matrix:\s*(\d+)", text)]
    jac_nnz = [int(v) for v in re.findall(r"# of nonzeros in Jacobian matrix:\s*(\d+)", text)]
    residual_iters = [int(v) for v in re.findall(r"\s+(\d+) KSP Residual norm", text)]
    wall_match = re.search(r"^real\s+([-+0-9.Ee]+)", text, flags=re.MULTILINE)
    rss_match = re.search(r"^\s*(\d+)\s+maximum resident set size", text, flags=re.MULTILINE)
    ambi_match = re.search(r"Time for ambipolar solve:\s*([-+0-9.EeDd]+)", text)
    return {
        "success_markers": {
            "brent_successful": "Brent algorithm successful." in text,
            "newton_successful": "Newton ambipolar solve was successful." in text,
            "goodbye": "Goodbye!" in text,
            "mpi_finalize_error": "MPI_Finalize failed" in text or "Fatal error in internal_Finalize" in text,
        },
        "solver_packages": sorted(set(re.findall(r"Solver package which will be used:\s*(\S+)", text))),
        "er_values": _parse_vector_after("Here are the Ers we used:", text),
        "radial_currents": _parse_vector_after("Here are the radial currents:", text),
        "internal_ambipolar_time_s": None if ambi_match is None else float(ambi_match.group(1).replace("D", "E")),
        "wall_time_s": None if wall_match is None else float(wall_match.group(1)),
        "max_rss_bytes": None if rss_match is None else int(rss_match.group(1)),
        "main_solve_times_s": solve_times,
        "adjoint_solve_times_s": adjoint_times,
        "jacobian_nnz": jac_nnz,
        "preconditioner_nnz": pmat_nnz,
        "max_ksp_iteration_index": max(residual_iters) if residual_iters else None,
        "mumps_jobs": {
            "analysis": text.count("JOB, N, NNZ =   1"),
            "factor": text.count("JOB, N, NNZ =   2"),
            "solve": text.count("JOB, N, NNZ =   3"),
            "destroy": text.count("JOB =  -2"),
        },
    }


def selected_namelists(tier: str, cases: list[str] | None) -> list[Path]:
    all_cases = sorted(NAMELIST_DIR.glob("*.namelist"))
    if tier != "all":
        all_cases = [p for p in all_cases if f"_{tier}_" in p.name]
    if cases:
        requested = set(cases)
        all_cases = [p for p in all_cases if p.stem in requested or p.name in requested]
    return all_cases


def run_case(*, namelist: Path, executable: Path, scratch: Path, timeout_s: float, extra_args: list[str]) -> dict[str, object]:
    case_dir = scratch / namelist.stem
    case_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(namelist, case_dir / "input.namelist")
    log_path = case_dir / "fortran_stdout.log"
    time_exe = Path("/usr/bin/time")
    if time_exe.exists():
        cmd = [str(time_exe), "-lp", str(executable), *extra_args]
    else:
        cmd = [str(executable), *extra_args]
    t0 = time.perf_counter()
    with log_path.open("w") as out:
        try:
            proc = subprocess.run(
                cmd,
                cwd=case_dir,
                stdout=out,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=timeout_s,
            )
            timed_out = False
            return_code = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            return_code = None
    elapsed = time.perf_counter() - t0
    parsed = parse_fortran_log(log_path) if log_path.exists() else {}
    return {
        "case": namelist.stem,
        "namelist": str(namelist.relative_to(REPO_ROOT)),
        "scratch_dir": str(case_dir),
        "return_code": return_code,
        "timed_out": timed_out,
        "elapsed_wall_s": elapsed,
        "parsed": parsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=["small", "production", "all"], default="small")
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--fortran-exe", type=Path, default=DEFAULT_FORTRAN)
    parser.add_argument("--scratch", type=Path, default=Path("/tmp/sfincs_v3_ambipolar_reference"))
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument(
        "--quiet-mumps",
        action="store_true",
        help="Pass the PETSc/MUMPS option used in the probe campaign to reduce verbosity when honored.",
    )
    args = parser.parse_args()

    namelists = selected_namelists(args.tier, args.cases)
    if not namelists:
        raise SystemExit("No matching namelists.")
    if not args.fortran_exe.exists():
        raise SystemExit(f"Missing Fortran v3 executable: {args.fortran_exe}")

    extra_args = ["-mat_mumps_icntl_4", "0"] if args.quiet_mumps else []
    args.scratch.mkdir(parents=True, exist_ok=True)
    results = [
        run_case(
            namelist=namelist,
            executable=args.fortran_exe,
            scratch=args.scratch,
            timeout_s=float(args.timeout_s),
            extra_args=extra_args,
        )
        for namelist in namelists
    ]
    payload = {
        "kind": "sfincs_fortran_v3_ambipolar_reference",
        "fortran_executable": str(args.fortran_exe),
        "tier": args.tier,
        "cases": results,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
