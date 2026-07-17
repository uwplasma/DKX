"""CPU-vs-GPU per-phase profile ladder and crossover scan (GPU performance lane).

One in-process case = one backend (pick it with ``JAX_PLATFORMS=cpu`` or
``JAX_PLATFORMS=cuda`` in the environment), one deck, per-phase wall times:

* ``operator_build`` — namelist -> :class:`KineticOperator` (host + device setup)
* ``rhs`` — drive assembly
* ``solve_cold`` — first :func:`sfincs_jax.solve.solve` (includes JIT compile,
  or the persistent-cache load when the cache is warm)
* ``solve_warm`` — second identical solve (cached executable; pure execution)
* ``moments`` — output moments from the solution

Compilation-cache state is controlled by the caller: point
``SFINCS_JAX_COMPILATION_CACHE_DIR`` at a fresh directory for a true cold
start, reuse it for a cache-warm cold start.

Usage (from the repo root)::

    # phase profile, built-in ladder size
    JAX_PLATFORMS=cpu python tools/benchmarks/gpu_cpu_ladder.py --size small
    JAX_PLATFORMS=cuda python tools/benchmarks/gpu_cpu_ladder.py --size mid

    # phase profile, explicit deck (e.g. the production namelist)
    python tools/benchmarks/gpu_cpu_ladder.py --deck /path/to/input.namelist

    # tier-1 crossover scan (warm auto-route solve per size, one backend)
    python tools/benchmarks/gpu_cpu_ladder.py --scan tier1

    # tier-2 GCROT crossover scan (FP deck family, method="gmres")
    python tools/benchmarks/gpu_cpu_ladder.py --scan tier2

Each invocation prints one JSON object per case on stdout (last line(s)), so
an orchestrator over ssh can collect results with ``tail``/``json.loads``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from profile_production import patch_namelist  # noqa: E402

_HSX = "tests/reduced_inputs/HSX_PASCollisions_DKESTrajectories.input.namelist"
_S4_2SP = "tests/ref/quick_2species_FPCollisions_noEr.input.namelist"

# The three-rung ladder: the reduced HSX PAS/DKES deck at its shipped
# resolution (small), the profile_production mid resolution, and the full
# production resolution (identical to the fortran_scaling_baseline deck's
# resolutionParameters; pass --deck for the real production namelist).
SIZES: dict[str, list[tuple[str, dict[str, str]]]] = {
    "small": [],
    "mid": [("resolutionParameters", {"Ntheta": "17", "Nzeta": "33", "Nxi": "60", "Nx": "5"})],
    "prod": [("resolutionParameters", {"Ntheta": "25", "Nzeta": "51", "Nxi": "100", "Nx": "5"})],
}

# Crossover scans: resolutions chosen to sweep total_size over ~2 decades.
TIER1_SCAN: list[tuple[str, str, str, str]] = [
    ("9", "9", "10", "4"),
    ("12", "12", "10", "12"),
    ("13", "25", "24", "5"),
    ("17", "33", "40", "5"),
    ("17", "33", "60", "5"),
    ("21", "41", "80", "5"),
    ("25", "51", "100", "5"),
]
TIER2_SCAN: list[tuple[str, str, str, str]] = [
    ("5", "7", "8", "5"),
    ("9", "11", "12", "5"),
    ("13", "21", "24", "6"),
    ("15", "27", "32", "6"),
    ("17", "33", "48", "7"),
]


def _write_deck(base_text: str, patches: list[tuple[str, dict[str, str]]], workdir: Path) -> Path:
    text = base_text
    for group, settings in patches:
        text = patch_namelist(text, group, settings)
    deck = workdir / "input.namelist"
    deck.write_text(text)
    return deck


def _phase_profile(deck: Path, *, method: str, tol: float, label: str) -> dict:
    import jax

    from sfincs_jax.drift_kinetic import kinetic_operator_from_namelist
    from sfincs_jax.namelist import read_sfincs_input
    from sfincs_jax.run import profile_moments_from_operator
    from sfincs_jax.solve import solve

    report: dict = {"case": label, "backend": jax.default_backend(), "method_requested": method}
    t0 = time.perf_counter()
    op = kinetic_operator_from_namelist(read_sfincs_input(deck))
    jax.block_until_ready(jax.tree_util.tree_leaves(op))
    t1 = time.perf_counter()
    rhs = op.rhs()
    rhs.block_until_ready()
    t2 = time.perf_counter()
    res = solve(op, rhs, method=method, tol=tol)
    res.x.block_until_ready()
    t3 = time.perf_counter()
    res2 = solve(op, rhs, method=method, tol=tol)
    res2.x.block_until_ready()
    t4 = time.perf_counter()
    moments = profile_moments_from_operator(op, res.x)
    jax.block_until_ready(list(moments.values()))
    t5 = time.perf_counter()
    report.update(
        n_dofs=int(op.total_size),
        method=res.method,
        iterations=None if res.iterations is None else int(res.iterations),
        residual=float(max(res.residual_norms)),
        operator_build_s=t1 - t0,
        rhs_s=t2 - t1,
        solve_cold_s=t3 - t2,
        solve_warm_s=t4 - t3,
        moments_s=t5 - t4,
        total_s=t5 - t0,
    )
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", choices=sorted(SIZES), help="built-in HSX PAS/DKES ladder size")
    ap.add_argument("--deck", type=Path, help="explicit input.namelist (overrides --size)")
    ap.add_argument("--scan", choices=["tier1", "tier2"], help="crossover scan instead of one profile")
    ap.add_argument("--method", default="auto", help="solve method (default auto)")
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--repeat-warm", type=int, default=1, help="extra warm solves averaged into solve_warm_s")
    args = ap.parse_args()

    import tempfile

    if args.scan:
        base = REPO_ROOT / (_HSX if args.scan == "tier1" else _S4_2SP)
        grid = TIER1_SCAN if args.scan == "tier1" else TIER2_SCAN
        method = "auto" if args.scan == "tier1" else "gmres"
        text = base.read_text()
        if args.scan == "tier2":
            text = patch_namelist(text, "physicsParameters", {"collisionOperator": "0"})
        for ntheta, nzeta, nxi, nx in grid:
            patches = [("resolutionParameters", {"Ntheta": ntheta, "Nzeta": nzeta, "Nxi": nxi, "Nx": nx})]
            with tempfile.TemporaryDirectory() as td:
                deck = _write_deck(text, patches, Path(td))
                label = f"{args.scan}_{ntheta}x{nzeta}x{nxi}x{nx}"
                rep = _phase_profile(deck, method=method, tol=args.tol, label=label)
            print(json.dumps(rep), flush=True)
        return 0

    if args.deck:
        deck = args.deck.resolve()
        label = str(deck)
        rep = _phase_profile(deck, method=args.method, tol=args.tol, label=label)
    else:
        size = args.size or "small"
        base = (REPO_ROOT / _HSX).read_text()
        with tempfile.TemporaryDirectory() as td:
            deck = _write_deck(base, SIZES[size], Path(td))
            rep = _phase_profile(deck, method=args.method, tol=args.tol, label=f"hsx_pas_dkes_{size}")
    print(json.dumps(rep), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
