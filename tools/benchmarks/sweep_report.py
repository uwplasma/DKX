#!/usr/bin/env python
"""Turn a parity_performance_matrix JSONL into a readable sweep report.

Three questions, three sections:

1. **Parity** -- does dkx still agree with SFINCS Fortran v3, broken out by the
   feature axes that matter (geometry scheme, RHS mode, collision operator,
   Phi1, electric field, species count)? A regression hides in an axis, not in
   an average.
2. **Performance** -- cold and warm, against the Fortran wall time on the same
   machine, with the route dkx chose. Cold is the honest number for one run in
   a terminal; warm is the honest number for a scan.
3. **Where the Fortran time goes** -- from PETSc's own event log. This is the
   section that says what is worth attacking: on these decks the Krylov
   iteration and even the LU are minor, and the dominant cost is SFINCS's own
   matrix assembly, which PETSc does not instrument and which dkx's structured
   direct route never performs.

Usage:
  python tools/benchmarks/sweep_report.py results.jsonl [--markdown]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

AXES = (
    ("geometryScheme", "geometry scheme"),
    ("RHSMode", "RHS mode"),
    ("collisionOperator", "collision operator"),
    ("includePhi1", "Phi1"),
    ("n_species", "species"),
)

MOMENTS = ("FSABFlow", "FSABjHat", "particleFlux_vm_psiHat", "heatFlux_vm_psiHat")


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# A moment this far below the case's largest compared moment is a physical
# zero, and a relative difference taken against it says nothing. The clearest
# example: on an axisymmetric single-species deck at Er = 0 the particle flux
# is intrinsically ambipolar and lands at ~5e-12 against a bootstrap current of
# ~3e-2, so its 1.4e-2 "disagreement" is an absolute difference of 7e-14.
NEGLIGIBLE_FRACTION = 1.0e-6


def _split_parity(record: dict) -> tuple[float | None, list[str]]:
    """Worst meaningful relative difference, and the keys judged negligible.

    Older records store ``parity`` as a flat ``{key: difference}`` mapping with
    no magnitudes; those cannot be classified, so every key counts and the
    caller is warned once.
    """
    parity = record.get("parity") or {}
    differences = parity.get("difference")
    magnitudes = parity.get("magnitude") or {}
    if differences is None:  # legacy flat schema
        values = [abs(v) for v in parity.values() if isinstance(v, (int, float))]
        return (max(values) if values else None), []

    scale = max(
        (abs(v) for v in magnitudes.values() if isinstance(v, (int, float))),
        default=0.0,
    )
    worst, negligible = None, []
    for key, value in differences.items():
        if not isinstance(value, (int, float)):
            continue
        magnitude = magnitudes.get(key)
        if (
            scale > 0.0
            and isinstance(magnitude, (int, float))
            and abs(magnitude) < NEGLIGIBLE_FRACTION * scale
        ):
            negligible.append(key)
            continue
        worst = abs(value) if worst is None else max(worst, abs(value))
    return worst, negligible


def _worst_parity(record: dict) -> float | None:
    return _split_parity(record)[0]


def section_parity(rows: list[dict]) -> None:
    print("\n== Parity against SFINCS Fortran v3, by feature axis ==")
    print("(worst scaled difference over " + ", ".join(MOMENTS) + ")\n")
    for key, label in AXES:
        buckets: dict = {}
        for record in rows:
            if key not in record:
                continue
            worst = _worst_parity(record)
            if worst is None:
                continue
            buckets.setdefault(record[key], []).append(worst)
        if not buckets:
            continue
        print(f"  {label}:")
        for value in sorted(buckets, key=str):
            values = buckets[value]
            print(
                f"    {value!s:<10s} n={len(values):<3d} worst={max(values):.2e}  median={sorted(values)[len(values) // 2]:.2e}"
            )


def section_performance(rows: list[dict]) -> None:
    print("\n== Runtime, same machine, 1 MPI rank ==\n")
    print(
        f"  {'case':<46s} {'dof':>7s} {'fortran':>8s} {'cold':>8s} {'warm':>8s} {'route':>28s}"
    )
    faster_cold = faster_warm = comparable = 0
    for record in sorted(rows, key=lambda r: r.get("dof", 0)):
        fortran = (record.get("fortran") or {}).get("1") or {}
        native = record.get("dkx") or {}
        f_s = fortran.get("wall_s")
        cold, warm = native.get("cold_s"), native.get("warm_s")
        if f_s is None or cold is None:
            continue
        comparable += 1
        faster_cold += cold < f_s
        faster_warm += warm is not None and warm < f_s
        print(
            f"  {record['case'][:46]:<46s} {record.get('dof', 0):7d} "
            f"{f_s:8.2f} {cold:8.2f} {(warm if warm is not None else float('nan')):8.2f} "
            f"{str(native.get('method'))[:28]:>28s}"
        )
    if comparable:
        print(
            f"\n  dkx faster cold: {faster_cold}/{comparable}   warm: {faster_warm}/{comparable}"
        )


def section_profile(rows: list[dict]) -> None:
    print("\n== Where the Fortran run spends its time (PETSc -log_view) ==\n")
    print(
        f"  {'case':<46s} {'petsc_tot':>10s} {'LUnum%':>7s} {'LUsym%':>7s} {'MatSolve%':>10s} {'KSP%':>6s}"
    )
    totals: dict = {}
    for record in sorted(rows, key=lambda r: r.get("dof", 0)):
        events = ((record.get("fortran") or {}).get("1") or {}).get(
            "petsc_events"
        ) or {}
        if not events:
            continue

        def pct(name: str, events: dict = events) -> float:
            entry = events.get(name)
            return entry["percent_time"] if isinstance(entry, dict) else 0.0

        for name in (
            "MatLUFactorNum",
            "MatLUFactorSym",
            "MatSolve",
            "KSPSolve",
            "MatAssemblyEnd",
        ):
            totals.setdefault(name, []).append(pct(name))
        print(
            f"  {record['case'][:46]:<46s} {events.get('total_time_s', float('nan')):10.3f} "
            f"{pct('MatLUFactorNum'):7.0f} {pct('MatLUFactorSym'):7.0f} "
            f"{pct('MatSolve'):10.0f} {pct('KSPSolve'):6.0f}"
        )
    if totals:
        print("\n  median share of PETSc-instrumented time:")
        for name, values in totals.items():
            ordered = sorted(values)
            print(f"    {name:<18s} {ordered[len(ordered) // 2]:5.0f}%")
        print(
            "\n  Note: these percentages are of PETSc's own accounted time, which is a "
            "\n  fraction of the wall clock. The remainder is SFINCS's Fortran matrix "
            "\n  pre-assembly, which PETSc does not instrument -- on the decks measured "
            "\n  here that is the dominant cost, and it is the step dkx's structured "
            "\n  direct route does not perform at all."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("jsonl", type=Path)
    args = parser.parse_args()
    rows = load(args.jsonl)
    print(f"cases: {len(rows)}")
    section_parity(rows)
    section_performance(rows)
    section_profile(rows)


if __name__ == "__main__":
    main()
