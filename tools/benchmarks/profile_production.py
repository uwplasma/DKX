"""Production profiling battery: stage-level runtime + memory across use cases.

Runs each case in a fresh subprocess (clean peak-RSS and JIT state) and
records per-stage wall time and memory via
:class:`dkx.profiling.SimpleProfiler`:

* RHSMode=1 cases: ``operator_build``, ``rhs``, ``solve_cold`` (includes
  compile), ``solve_warm`` (cached executable), ``moments``, plus an
  end-to-end ``run_profile`` total (adds writer/console/IO).
* RHSMode=2/3: ``run_transport_matrix`` end-to-end total.
* Phi1: Newton ``solve_phi1`` cold + warm-started re-solve.
* Gradient: ``jax.value_and_grad`` of ``FSABjHat`` w.r.t. ``THat`` through the
  implicit solve (cold + warm).
* Ambipolar: ``find_ambipolar_er`` total with root-solve iteration count.

Usage::

    python tools/benchmarks/profile_production.py --all --out profile.json
    python tools/benchmarks/profile_production.py --all --big --out profile.json
    python tools/benchmarks/profile_production.py --case hsx_pas_dkes_mid

The battery is portable CPU/GPU: the report records ``jax.default_backend()``
and device memory on GPU. Cases are sized for a 24 GB laptop by default; the
``--big`` flag adds the full HSX production resolution (744,610 unknowns, the
Fortran head-to-head case of ``tier1_hsx_head_to_head.py``).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Namelist patching (same convention as tier1_hsx_head_to_head.patch_namelist)
# ---------------------------------------------------------------------------


def patch_namelist(text: str, group: str, settings: dict[str, str]) -> str:
    """Insert or override ``key = value`` lines inside one namelist group."""
    lines = text.splitlines()
    out: list[str] = []
    in_group = False
    pending = dict(settings)
    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("&" + group.lower()):
            in_group = True
            out.append(line)
            continue
        if in_group and stripped == "/":
            for key, value in pending.items():
                out.append(f"  {key} = {value}")
            pending = {}
            in_group = False
            out.append(line)
            continue
        if in_group:
            m = re.match(r"\s*([A-Za-z_]\w*)\s*=", line)
            if m and m.group(1).lower() in {k.lower() for k in pending}:
                key = next(k for k in pending if k.lower() == m.group(1).lower())
                out.append(f"  {key} = {pending.pop(key)}")
                continue
        out.append(line)
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Case definitions: (base deck, [(group, {key: value}), ...], runner)
# ---------------------------------------------------------------------------

_HSX = "tests/reduced_inputs/HSX_PASCollisions_DKESTrajectories.input.namelist"
_S4_2SP = "tests/ref/quick_2species_FPCollisions_noEr.input.namelist"
_MONO = "tests/ref/monoenergetic_PAS_tiny_scheme1.input.namelist"
_PHI1 = "tests/ref/pas_1species_PAS_noEr_tiny_withPhi1_inKinetic_linear.input.namelist"
_PAS1 = "tests/ref/pas_1species_PAS_noEr_tiny_scheme1.input.namelist"

CASES: dict[str, dict] = {
    "hsx_pas_dkes_mid": {
        "base": _HSX,
        "patches": [("resolutionParameters", {"Ntheta": "17", "Nzeta": "33", "Nxi": "60", "Nx": "5"})],
        "runner": "rhs1",
    },
    "hsx_pas_dkes_prod": {
        "base": _HSX,
        "patches": [("resolutionParameters", {"Ntheta": "25", "Nzeta": "51", "Nxi": "100", "Nx": "5"})],
        "runner": "rhs1",
        "big": True,
    },
    "w7x_fp_2species": {
        "base": _S4_2SP,
        "patches": [
            ("resolutionParameters", {"Ntheta": "13", "Nzeta": "21", "Nxi": "24", "Nx": "6"}),
            ("physicsParameters", {"collisionOperator": "0"}),
        ],
        "runner": "rhs1",
    },
    # RHSMode=2 is validated to a single species (inputs._validate, Fortran parity).
    "rhs2_pas_scheme1": {
        "base": _PAS1,
        "patches": [
            ("general", {"RHSMode": "2"}),
            ("resolutionParameters", {"Ntheta": "15", "Nzeta": "25", "Nxi": "32", "Nx": "6"}),
        ],
        "runner": "transport",
    },
    "mono_rhs3_scheme1": {
        "base": _MONO,
        "patches": [("resolutionParameters", {"Ntheta": "15", "Nzeta": "27", "Nxi": "64"})],
        "runner": "transport",
    },
    # solve_phi1's inner Krylov is unpreconditioned and refuses total_size > 6000
    # (the documented Phi1-aware tier-2 preconditioner follow-up), so this case
    # is sized just under that cap — the largest canonical Phi1 solve possible.
    "phi1_newton_mid": {
        "base": _PHI1,
        "patches": [("resolutionParameters", {"Ntheta": "7", "Nzeta": "7", "Nxi": "16", "Nx": "5"})],
        "runner": "phi1",
    },
    "grad_pas_scheme1": {
        "base": _PAS1,
        "patches": [
            ("resolutionParameters", {"Ntheta": "13", "Nzeta": "21", "Nxi": "24", "Nx": "6"}),
        ],
        "runner": "gradient",
    },
    "ambipolar_er_2species": {
        "base": _S4_2SP,
        "patches": [
            ("resolutionParameters", {"Ntheta": "11", "Nzeta": "15", "Nxi": "16", "Nx": "5"}),
            ("physicsParameters", {"collisionOperator": "1"}),
        ],
        "runner": "ambipolar",
    },
}


def _write_deck(case: dict, workdir: Path) -> Path:
    text = (REPO_ROOT / case["base"]).read_text()
    for group, settings in case["patches"]:
        text = patch_namelist(text, group, settings)
    deck = workdir / "input.namelist"
    deck.write_text(text)
    return deck


# ---------------------------------------------------------------------------
# In-process runners (executed inside the per-case subprocess)
# ---------------------------------------------------------------------------


def _run_case(name: str) -> dict:
    import jax

    from dkx.profiling import SimpleProfiler

    case = CASES[name]
    report: dict = {"case": name, "backend": jax.default_backend()}
    prof = SimpleProfiler(sample_device_mem=jax.default_backend() != "cpu")

    with tempfile.TemporaryDirectory() as td:
        deck = _write_deck(case, Path(td))
        runner = case["runner"]

        if runner == "rhs1":
            from dkx.drift_kinetic import kinetic_operator_from_namelist
            from dkx.namelist import read_sfincs_input
            from dkx.run import profile_moments_from_operator, run_profile
            from dkx.solve import solve

            op = kinetic_operator_from_namelist(read_sfincs_input(deck))
            report["n_dofs"] = int(op.total_size)
            prof.mark("operator_build")
            rhs = op.rhs()
            rhs.block_until_ready()
            prof.mark("rhs")
            res = solve(op, rhs, tol=1e-10)
            res.x.block_until_ready()
            prof.mark("solve_cold")
            report["method"] = res.method
            report["iterations"] = None if res.iterations is None else int(res.iterations)
            report["residual"] = float(max(res.residual_norms))
            res2 = solve(op, rhs, tol=1e-10)
            res2.x.block_until_ready()
            prof.mark("solve_warm")
            moments = profile_moments_from_operator(op, res.x)
            _ = {k: v for k, v in moments.items()}
            prof.mark("moments")
            t0 = time.perf_counter()
            run_profile(deck, tol=1e-10, out_path=Path(td) / "out.h5", emit=None)
            report["e2e_run_profile_s"] = time.perf_counter() - t0

        elif runner == "transport":
            from dkx.run import run_transport_matrix

            t0 = time.perf_counter()
            run = run_transport_matrix(deck, tol=1e-10, emit=None)
            report["e2e_run_transport_s"] = time.perf_counter() - t0
            report["method"] = run.solve_result.method
            prof.mark("run_transport_matrix")

        elif runner == "phi1":
            from dkx.phi1 import solve_phi1

            res = solve_phi1(deck, tol=1e-9)
            prof.mark("solve_phi1_cold")
            report["n_newton"] = int(res.n_newton)
            report["inner_iterations"] = int(res.inner_iterations_total or 0)
            report["residual"] = float(res.residual_norm)
            solve_phi1(deck, tol=1e-9, x0=res.x)
            prof.mark("solve_phi1_warm")

        elif runner == "gradient":
            from dataclasses import replace

            import jax.numpy as jnp

            from dkx.drift_kinetic import kinetic_operator_from_namelist
            from dkx.namelist import read_sfincs_input
            from dkx.run import profile_moments_from_operator
            from dkx.solve import solve

            op0 = kinetic_operator_from_namelist(read_sfincs_input(deck))
            report["n_dofs"] = int(op0.total_size)
            prof.mark("operator_build")

            def fsabjhat(t_hat):
                op = replace(op0, t_hat=jnp.asarray([t_hat], dtype=jnp.float64))
                res = solve(op, op.rhs(), method="block_tridiagonal", differentiable=True)
                return profile_moments_from_operator(op, res.x)["FSABjHat"]

            t0 = float(op0.t_hat[0])
            val, grad = jax.value_and_grad(fsabjhat)(t0)
            jax.block_until_ready((val, grad))
            prof.mark("value_and_grad_cold")
            report["fsabjhat"] = float(val)
            report["dfsabjhat_dthat"] = float(grad)
            val2, grad2 = jax.value_and_grad(fsabjhat)(t0 * 1.01)
            jax.block_until_ready((val2, grad2))
            prof.mark("value_and_grad_warm")

        elif runner == "ambipolar":
            from dkx.er import find_ambipolar_er

            res = find_ambipolar_er(deck, er_bracket=(-2.0, 2.0), tol=1e-8)
            prof.mark("find_ambipolar_er")
            report["er_root"] = float(res.er)
            report["n_evaluations"] = int(getattr(res, "n_evaluations", 0) or 0)

        else:  # pragma: no cover - config error
            raise ValueError(f"unknown runner {runner!r}")

    report["stages"] = prof.entries
    report["total_s"] = prof.entries[-1]["total_s"] if prof.entries else None
    report["peak_rss_mb"] = prof.entries[-1]["peak_rss_mb"] if prof.entries else None
    return report


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", help="run one case in-process and print its JSON report")
    ap.add_argument("--all", action="store_true", help="run every (non-big) case, each in a subprocess")
    ap.add_argument("--big", action="store_true", help="include the full HSX production case")
    ap.add_argument("--out", type=Path, default=None, help="write the aggregate JSON report here")
    args = ap.parse_args()

    if args.case:
        print(json.dumps(_run_case(args.case)))
        return 0

    if not getattr(args, "all"):
        ap.error("pass --case NAME or --all")

    reports = []
    for name, case in CASES.items():
        if case.get("big") and not args.big:
            continue
        t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--case", name],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        wall = time.perf_counter() - t0
        if proc.returncode != 0:
            reports.append({"case": name, "error": proc.stderr.strip()[-2000:], "subprocess_wall_s": wall})
            print(f"[FAIL] {name}: rc={proc.returncode}", flush=True)
            continue
        rep = json.loads(proc.stdout.strip().splitlines()[-1])
        rep["subprocess_wall_s"] = wall
        reports.append(rep)
        peak = rep.get("peak_rss_mb")
        peak_txt = f"{peak:.0f}" if isinstance(peak, (int, float)) else "na"
        print(
            f"[ok] {name}: total={rep.get('total_s', 0):.1f}s wall={wall:.1f}s "
            f"peak_rss={peak_txt}MB method={rep.get('method', '-')}",
            flush=True,
        )

    if args.out:
        args.out.write_text(json.dumps(reports, indent=1))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
