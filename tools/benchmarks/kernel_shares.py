"""Kernel-share profile of a warm DKX solve on CPU or GPU.

    python tools/benchmarks/kernel_shares.py --case mono_w7x --data DIR --out shares.json
    JAX_PLATFORMS=cuda python tools/benchmarks/kernel_shares.py --case ncsx_fp --data DIR

Each case is built, solved once untraced (compile), timed warm, then solved once
more under ``jax.profiler.trace``. The Perfetto trace is parsed and every device
or XLA-thunk event is assigned to one class: batched LU (getrf), triangular
solves (trsm / getrs), dense matmul (gemm / dot), FFT, Krylov reductions,
host-device copies, and the rest (elementwise fusions, loop control). Shares are
fractions of the summed kernel time; ``wall`` is the traced solve's wall time.
``--data`` holds boozmn_wout_w7x_eim.nc, boozmn_HSX_QHS_vac.nc and wout_NCSX.nc.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "benchmarks"))

CLASSES = [
    ("lu (getrf)", r"getrf|lu_pivots|lu\b|lu_factor|_lu|getf2|laswp"),
    ("triangular (trsm/getrs)", r"trsm|getrs|triangular|trsv"),
    ("matmul (gemm/dot)", r"ynn|gemm|dot|cutlass|matmul|sm\d+_xmma|ampere_|cublas"),
    ("fft", r"fft|cufft|ducc"),
    ("copies h<->d", r"memcpy|copy-start|copy-done|transfer|MemcpyH|MemcpyD"),
    ("reduce (Krylov dots/norms)", r"reduce|norm"),
]


def _classify(name: str) -> str:
    low = name.lower()
    for label, pat in CLASSES:
        if re.search(pat, low):
            return label
    return "other (fusions, loops)"


def build(case: str, data: Path, tmp: Path):
    """Return ``fn()`` running one solve and a description."""
    import numpy as np

    from dkx.solve import solve

    if case in ("mono_w7x", "mono_hsx"):
        from cross_code_speed import DEVICES, NUD_X0, dkx_mono_deck, spectrum
        from dkx.monoenergetic import monoenergetic_database

        dev = "W7-X" if case == "mono_w7x" else "HSX"
        booz, s = DEVICES[dev][:2]
        sp = spectrum(data / booz, s)
        dkx_mono_deck(sp, s, (19, 41, 64), tmp / "input.namelist")
        b0 = next(v for m, n, v in sp["modes"] if m == n == 0)
        nup = abs(sp["G"] + sp["iota"] * sp["I"]) / b0 * 1e-2 / NUD_X0

        def fn():
            db = monoenergetic_database(tmp / "input.namelist", [nup], [0.0], tol=1e-6)
            return np.ravel(db.d11_star)

        return fn, "W7-X/HSX monoenergetic (19, 41, 64), nu*=0.067"
    if case == "bootstrap_row":
        import dkx
        from dkx.workflows.geometry_adapters import kinetic_operator_on_boozer_surface

        sys.path.insert(0, str(REPO / "tests"))
        fix = json.loads((REPO / "tests/ref/boozer_route_sign_vmex_seed_nfp2.json").read_text())
        S = float(fix["s"])
        n, t = 1e19 / 1e20 * (1 - S**5), 2.0 * (1 - S)
        plasma = dict(
            inputRadialCoordinate=1, inputRadialCoordinateForGradients=1, psiN_wish=S,
            Zs=[1.0, -1.0], mHats=[1.0, 5.446170214e-4], nHats=[n, n], THats=[t, t],
            dNHatdpsiNs=[-5e-1 * S**4] * 2, dTHatdpsiNs=[-2.0] * 2,
            Ntheta=13, Nzeta=13, Nxi=48, NL=4, Nx=5,
            collisionOperator=1, Delta=4.5694e-3, alpha=1.0, nu_n=8.330e-3, Er=0.0,
        )  # fmt: skip
        tpl = dkx.run(geometryScheme=1, psiAHat=0.05, aHat=0.1, **plasma, emit=None).operator
        op = kinetic_operator_on_boozer_surface(
            tpl, bmnc_b=np.asarray(fix["bmnc_b"]), ixm_b=fix["ixm_b"], ixn_b=fix["ixn_b"],
            nfp=fix["nfp"], iota=fix["iota_b"], g_hat=fix["bvco_b"], i_hat=fix["buco_b"],
        )  # fmt: skip
        rhs = op.rhs()

        def fn():
            return solve(op, rhs, method="auto", tol=1e-10, tier1_keep_lowest=op.n_xi, emit=None).x

        return fn, f"bootstrap row, nfp2 Boozer 13x13, Nxi 48, Nx 5, {op.total_size} unknowns"
    from dkx.drift_kinetic import kinetic_operator_from_namelist
    from dkx.namelist import read_sfincs_input

    if case == "ncsx_fp":
        from cross_code_speed import dkx_dke_deck

        dkx_dke_deck(data / "wout_NCSX.nc", (15, 31, 61, 6), tmp / "input.namelist")
        deck = tmp / "input.namelist"
    elif case == "hsx_fp":
        from profile_production import patch_namelist

        base = (REPO / "tests/reduced_inputs/HSX_FPCollisions_DKESTrajectories.input.namelist").read_text()
        res = {"Ntheta": "25", "Nzeta": "51", "Nxi": "100", "Nx": "5"}
        (tmp / "input.namelist").write_text(patch_namelist(base, "resolutionParameters", res))
        deck = tmp / "input.namelist"
    else:
        raise SystemExit(f"unknown case {case}")
    op = kinetic_operator_from_namelist(read_sfincs_input(deck))
    rhs = op.rhs()

    def fn():
        r = solve(op, rhs, method="auto", tol=1e-6, emit=None)
        fn.info = dict(method=r.method, iterations=None if r.iterations is None else int(r.iterations))
        return r.x

    return fn, f"{case} {op.total_size} unknowns"


def shares(trace_dir: Path) -> dict:
    path = next(trace_dir.rglob("*.trace.json.gz"), None) or next(trace_dir.rglob("perfetto_trace.json.gz"))
    ev = json.loads(gzip.open(path).read())
    events = ev["traceEvents"] if isinstance(ev, dict) else ev
    pids = {e["pid"]: e["args"]["name"] for e in events if e.get("ph") == "M" and e.get("name") == "process_name"}
    tot, names = defaultdict(float), defaultdict(float)
    for e in events:
        if e.get("ph") != "X" or "dur" not in e:
            continue
        pname = pids.get(e["pid"], "")
        dev = "/device:GPU" in pname
        # On GPU count kernels and memcpys of the device streams; on CPU the XLA thunks.
        if dev or ("/host:CPU" in pname and _is_thunk(e)):
            nm = e["name"]
            if re.match(r"(while|conditional|call)\b", nm.lower()):
                continue  # control thunks enclose their bodies; counting them double-counts
            tot[_classify(nm)] += e["dur"] * 1e-6
            names[nm[:60]] += e["dur"] * 1e-6
    s = sum(tot.values()) or 1.0
    top = sorted(names.items(), key=lambda kv: -kv[1])[:25]
    return dict(kernel_s=s, shares={k: v / s for k, v in sorted(tot.items(), key=lambda kv: -kv[1])},
                top=[(k, round(v, 4)) for k, v in top])


def _is_thunk(e: dict) -> bool:
    a = e.get("args", {})
    return "hlo_op" in a or "hlo_module" in a or "long_name" in a


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--trace-dir", type=Path)
    a = ap.parse_args()
    import jax

    jax.config.update("jax_enable_x64", True)
    tmp = Path(tempfile.mkdtemp())
    t0 = time.perf_counter()
    fn, desc = build(a.case, a.data, tmp)
    t1 = time.perf_counter()
    jax.block_until_ready(fn())
    t2 = time.perf_counter()
    jax.block_until_ready(fn())
    t3 = time.perf_counter()
    td = a.trace_dir or tmp / "trace"
    with jax.profiler.trace(str(td), create_perfetto_trace=True):
        jax.block_until_ready(fn())
    t4 = time.perf_counter()
    rep = dict(case=a.case, desc=desc, backend=jax.default_backend(), build_s=t1 - t0, cold_s=t2 - t1,
               warm_s=t3 - t2, traced_s=t4 - t3, info=getattr(fn, "info", None), **shares(td))
    print(json.dumps(rep, indent=1))
    if a.out:
        a.out.write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
