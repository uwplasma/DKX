"""Matched speed and memory benchmark: DKX against MONKES and YANCC.

    python tools/benchmarks/cross_code_speed.py            # table + figure from the stored JSON
    python tools/benchmarks/cross_code_speed.py measure --monkes BIN --yancc-python PY \
        --data DIR --ncores 4 [--gpu 0]                     # re-measure (about two hours)

``--data`` holds boozmn_wout_w7x_eim.nc and wout_NCSX.nc (YANCC tests/data) and
boozmn_HSX_QHS_vac.nc. Every timed run is a fresh process pinned with taskset to the
four least busy physical cores of the (shared) host at launch, with OPENBLAS_NUM_THREADS
equal to the core count and an XLA pool of NPROC=16 threads; the codes alternate order
per repeat. Cold is the first solve in the process (JAX tracing and
compilation included, no persistent cache); warm is the identical second solve. MONKES
has no compile step: cold is its process wall time, warm its own reported solve time.
Peak RSS comes from wait4.
The ``cached`` stage re-times DKX with its persistent compilation cache on (the
default outside this harness), after one untimed process has filled it. Accuracy is the largest relative change of D11, D31, D33
(monoenergetic) or of the flow and fluxes (full DKE) against the same code's finest rung.
"""

import json, os, subprocess, sys, tempfile, threading, time
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "docs/_static/figures/benchmarks"
# (booz_xform file, s, nu/v [1/m], E_r/v [V s/m^2]): both devices at the same
# nu* = nu R0/(v iota) = 0.672, 0.0672, 0.00672 and v_E = E_r/(v B00) = 0, 4.1e-4.
DEVICES = {"W7-X": ("boozmn_wout_w7x_eim.nc", 0.2, (1e-1, 1e-2, 1e-3), (0.0, 1e-3)),
           "HSX": ("boozmn_HSX_QHS_vac.nc", 0.25, (0.4955, 0.04955, 0.004955), (0.0, 4.15e-4))}
LADDER = [(11, 23, 32), (15, 31, 48), (19, 41, 64), (25, 51, 96), (31, 63, 128)]
DKE_LADDER = [(9, 19, 31, 4), (13, 25, 41, 5), (15, 31, 61, 6), (19, 41, 81, 7), (25, 51, 101, 8)]
DKE_ER = -3.0  # kV/m, row 10 of YANCC's SFINCS NCSX one-species table
TARGET, DKE_TARGET, NUD_X0 = 0.03, 0.10, 0.8360276804879032  # DKE: particle flux nearly cancels


def spectrum(booz, s, cutoff=1e-5):
    """|B| spectrum and flux functions of a booz_xform file, mapped as YANCC maps them."""
    import netCDF4
    import numpy as np

    f = netCDF4.Dataset(booz)
    g = lambda k: np.asarray(f[k][:])  # noqa: E731
    ns, nfp = int(g("ns_b")), int(g("nfp_b"))
    sh = (np.arange(ns - 1) + 0.5) / (ns - 1)
    bm = np.array([np.interp(s, sh[g("jlist") - 2], c) for c in g("bmnc_b").T])
    it, i_, g_ = (np.interp(s, sh, g(k)[1:]) for k in ("iota_b", "buco_b", "bvco_b"))
    sg = np.sign(g_ - it * i_)
    keep = np.abs(bm) > cutoff * np.abs(bm).max()
    modes = [(int(m), int(-n // nfp), v) for m, n, v, k in zip(g("ixm_b"), g("ixn_b"), bm, keep) if k]
    psi = abs(g("phi_b")[-1])
    vol = np.sum(g("gmn_b")[:, 0] * psi / (2 * np.pi * (ns - 1))) * 4 * np.pi**2
    a = (abs(vol) / (2 * np.pi**2 * float(g("aspect_b")))) ** (1 / 3)
    return dict(nfp=nfp, iota=-it, G=sg * g_, I=-sg * i_, psiA=psi / (2 * np.pi), a=a, modes=modes)


def dkx_mono_deck(sp, s, res, path):
    lines = "\n".join(f"  boozer_bmnc({m},{n}) = {v:.16e}" for m, n, v in sp["modes"])
    path.write_text(f"""&general\n  RHSMode = 3\n/\n&geometryParameters\n  geometryScheme = 13
  inputRadialCoordinate = 3\n  rN_wish = {s ** 0.5}\n  aHat = {sp['a']:.16e}\n  Nperiods = {sp['nfp']}\n  iota = {sp['iota']:.16e}
  GHat = {sp['G']:.16e}\n  IHat = {sp['I']:.16e}\n  psiAHat = {sp['psiA']:.16e}\n{lines}\n/
&speciesParameters\n/\n&physicsParameters\n  collisionOperator = 1\n/
&resolutionParameters\n  Ntheta = {res[0]}\n  Nzeta = {res[1]}\n  Nxi = {res[2]}\n  Nx = 1\n  solverTolerance = 1d-6\n/
&otherNumericalParameters\n/\n&preconditionerOptions\n/\n""")


def dkx_dke_deck(wout, res, path):
    path.write_text(f"""&general\n  RHSMode = 1\n/\n&geometryParameters\n  geometryScheme = 5
  inputRadialCoordinate = 3\n  inputRadialCoordinateForGradients = 4\n  rN_wish = 0.5
  equilibriumFile = "{wout}"\n/\n&speciesParameters\n  Zs = 1\n  mHats = 1\n  nHats = 1.5\n  THats = 0.8
  dNHatdrHats = -0.4\n  dTHatdrHats = -2.0\n/\n&physicsParameters\n  nu_n = 8.330d-3\n  Er = {DKE_ER}
  collisionOperator = 0\n  includeXDotTerm = .true.\n  includeElectricFieldTermInXiDot = .true.
  useDKESExBDrift = .false.\n/\n&resolutionParameters\n  Ntheta = {res[0]}\n  Nzeta = {res[1]}
  Nxi = {res[2]}\n  Nx = {res[3]}\n  solverTolerance = 1d-6\n/\n&otherNumericalParameters\n/\n&preconditionerOptions\n/\n""")


def _gpu_peak():
    import jax

    d = jax.devices()[0]
    return (d.memory_stats() or {}).get("peak_bytes_in_use") if d.platform == "gpu" else None


def _timed(fn, n):
    out = []
    for _ in range(n):
        t = time.perf_counter()
        out.append((fn(), time.perf_counter() - t))
    return out


def worker(code, case, res, points, data, repeats):
    """One process: solve each point ``repeats`` times; return values and times."""
    import numpy as np

    data, res = Path(data), tuple(res)
    if code == "dkx":
        import jax

        jax.config.update("jax_enable_x64", True)
        from dkx.monoenergetic import monoenergetic_database
        from dkx.api import read_output
        from dkx.run import run_profile

        tmp = Path(tempfile.mkdtemp())
        if case == "DKE":
            dkx_dke_deck(data / "wout_NCSX.nc", res, tmp / "input.namelist")

            def solve(_):
                run_profile(tmp / "input.namelist", tol=1e-6, emit=None, out_path=tmp / "out.npz")
                m = read_output(tmp / "out.npz")
                return [float(np.ravel(m[k])[0]) for k in ("FSABFlow", "particleFlux_vm_rHat", "heatFlux_vm_rHat")]
        else:
            booz, s = DEVICES[case][:2]
            sp = spectrum(data / booz, s)
            dkx_mono_deck(sp, s, res, tmp / "input.namelist")
            b0 = next(v for m, n, v in sp["modes"] if m == n == 0)
            r0, r = abs(sp["G"]) / b0, np.sqrt(2 * s * sp["psiA"] / b0)

            def solve(p):
                nup = abs(sp["G"] + sp["iota"] * sp["I"]) / b0 * p[0] / NUD_X0
                es = p[1] / b0 / (abs(sp["iota"]) * r / r0)
                db = monoenergetic_database(tmp / "input.namelist", [nup], [es], tol=1e-6)
                return [float(np.ravel(getattr(db, k))[0]) for k in ("d11_star", "d31_star", "d33_star")]
    else:
        import jax

        jax.config.update("jax_enable_x64", True)
        from yancc.field import Field
        from yancc.velocity_grids import UniformPitchAngleGrid

        if case == "DKE":
            import yancc
            from yancc.solve import solve_dke
            from yancc.species import LocalMaxwellian
            from yancc.velocity_grids import MaxwellSpeedGrid

            field = Field.from_vmec(str(data / "wout_NCSX.nc"), 0.5, res[0], res[1])
            a = float(field.a_minor)
            sp = [LocalMaxwellian(yancc.species.Hydrogen, 0.8e3, 1.5e20, -2e3 * a, -0.4e20 * a)]
            grids = UniformPitchAngleGrid(res[2]), MaxwellSpeedGrid(res[3])

            def solve(_):
                sol, _i = solve_dke(field, *grids, sp, Erho=DKE_ER * a * 1000, coulomb_log=17, rtol=1e-6,
                                    multigrid_options={"max_grids": 3, "coarse_N": 2000})
                v = [sol.get(k) for k in ("FSABFlow_sfincs", "particleFlux_vm_rN_sfincs", "heatFlux_vm_rN_sfincs")]
                return [float(np.ravel(v[0])[0]), float(np.ravel(v[1])[0]) * a, float(np.ravel(v[2])[0]) * a]
        else:
            from yancc.solve import solve_mdke

            booz, s = DEVICES[case][:2]
            field = Field.from_booz_xform(str(data / booz), np.sqrt(s), res[0], res[1], cutoff=1e-5)
            pitch = UniformPitchAngleGrid(res[2])

            def solve(p):
                sol, _i = solve_mdke(field, pitch, p[1] * field.a_minor, p[0], rtol=1e-6,
                                     multigrid_options={"coarse_N": 1000})
                d = np.asarray(jax.block_until_ready(sol.get("Dij_DKES")))
                return [float(d[0, 0]), float(d[2, 0]), float(d[2, 2])]
    out = []
    for p in points:
        runs = _timed(lambda: solve(p), repeats)
        out.append(dict(point=p, values=runs[-1][0], times=[t for _, t in runs]))
    return dict(results=out, gpu_peak_bytes=_gpu_peak())


def run_monkes(binary, booz, s, res, points, workdir):
    wd = Path(tempfile.mkdtemp(dir=workdir))
    (wd / "boozmn.nc").symlink_to(booz)
    (wd / "monkes_input.surface").write_text(f"&surface\ns={s}\n/\n")
    nus = sorted({p[0] for p in points})
    ers = sorted({p[1] for p in points})
    (wd / "monkes_input.parameters").write_text(
        f"&parameters\nN_theta={res[0]}\nN_zeta={res[1]}\nN_xi={res[2]}\n"
        f"nu={','.join(map(str, nus))}\nE_r={','.join(map(str, ers))}\n/\n")
    return wd, [str(binary)]


def parse_monkes(wd):
    rows = [list(map(float, ln.split())) for ln in (wd / "monkes_Monoenergetic_Database.dat").read_text().splitlines()
            if ln.strip() and not ln.lstrip().startswith("nu")]
    return [dict(point=[r[0], r[1]], values=[r[5], r[6], r[8]], times=[r[10]]) for r in rows]


def idle_cores(n, physical):
    """The ``n`` least busy physical cores over a one-second sample (the host is shared)."""
    def ticks():
        rows = [ln.split() for ln in open("/proc/stat") if ln[:3] == "cpu" and ln[3] != " "]
        return {int(r[0][3:]): (sum(map(int, r[1:])), int(r[4]) + int(r[5])) for r in rows}

    a = ticks()
    time.sleep(1.0)
    b = ticks()
    busy = {c: 1 - (b[c][1] - a[c][1]) / max(b[c][0] - a[c][0], 1) for c in a}
    load = {c: busy[c] + busy.get(c + physical, 0) for c in range(physical)}  # core plus its HT sibling
    return ",".join(map(str, sorted(sorted(load, key=load.get)[:n])))


def launch(cmd, env, cores, timeout, cwd=None):
    """Run pinned; return (stdout, wall seconds, peak RSS bytes, status)."""
    t = time.perf_counter()
    with tempfile.TemporaryFile() as f:
        p = subprocess.Popen(["taskset", "-c", cores, *cmd], stdout=f, stderr=subprocess.DEVNULL, env=env, cwd=cwd)
        timer = threading.Timer(timeout, p.kill)
        timer.start()
        _, status, ru = os.wait4(p.pid, 0)
        timer.cancel()
        p.returncode = status
        f.seek(0)
        return f.read().decode(), time.perf_counter() - t, ru.ru_maxrss * 1024, status


def measure(args):
    ncores = args.ncores
    # XLA's pool is NPROC threads on the pinned cores; YANCC deadlocks in its preconditioner
    # setup when the pool has <= 4 threads, so both JAX codes get --xla-threads (16).
    base = dict(os.environ, NPROC=str(args.xla_threads), OPENBLAS_NUM_THREADS=str(ncores),
                OMP_NUM_THREADS=str(ncores), DKX_DISABLE_COMPILATION_CACHE="1", JAX_ENABLE_X64="1")
    base.pop("DKX_CORES", None)
    py = {"dkx": sys.executable, "yancc": args.yancc_python}
    data, work = Path(args.data).resolve(), Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    log = work / "raw.jsonl"
    recs = [json.loads(ln) for ln in log.read_text().splitlines()] if log.exists() else []

    def one(stage, code, case, res, points, repeats=1, gpu=None):
        env = dict(base, JAX_PLATFORMS="cuda" if gpu is not None else "cpu")
        if code == "dkx" and args.dkx_xla_threads:  # DKX's LAPACK calls share the cores with XLA's pool
            env["NPROC"] = str(args.dkx_xla_threads)
        if stage in ("cached", "fill"):
            env.pop("DKX_DISABLE_COMPILATION_CACHE")
            env["DKX_COMPILATION_CACHE_DIR"] = str(work / "jax_cache")
        done = [r for r in recs if (r["stage"], r["code"], r["case"], r["res"], r["status"]) ==
                (stage, code, case, list(res), 0) and sorted(q["point"] for q in r["results"]) == sorted(points)]
        if stage == "ladder" and done:
            return done[0]
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        if code == "monkes":
            booz, s = DEVICES[case][:2]
            wd, cmd = run_monkes(args.monkes, data / booz, s, res, points, work)
            cores = idle_cores(ncores, args.physical)
            stdout, wall, rss, st = launch(cmd, env, cores, args.timeout, cwd=wd)
            results = parse_monkes(wd) if st == 0 else []
            rec = dict(results=results, gpu_peak_bytes=None)
        else:
            payload = json.dumps([code, case, list(res), points, str(data), repeats])
            cmd = [py[code], __file__, "worker", payload]
            cores = idle_cores(ncores, args.physical)
            stdout, wall, rss, st = launch(cmd, env, cores, args.timeout)
            rec = json.loads(stdout.strip().splitlines()[-1]) if st == 0 else dict(results=[])
        rec.update(stage=stage, code=code, case=case, res=list(res), process_wall=wall, peak_rss=rss, status=st,
                   device="gpu" if gpu is not None else "cpu", cores=cores, load=os.getloadavg()[0])
        with log.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(code, case, res, f"{wall:.1f}s rss={rss / 2**30:.2f}G st={st}", flush=True)
        return rec

    stages = args.stages.split(",")
    pts = {c: [[n, e] for n in d[2] for e in d[3]] for c, d in DEVICES.items()} | {"DKE": [[0, 0]]}
    if "ladder" in stages:
        for case in DEVICES:
            for res in LADDER:
                for code in ("monkes", "dkx", "yancc"):
                    one("ladder", code, case, res, pts[case])
        for res in DKE_LADDER:
            for code in ("dkx", "yancc"):
                one("ladder", code, "DKE", res, [[0, 0]])
    if {"time", "gpu", "cached"} & set(stages):
        recs = [json.loads(ln) for ln in log.read_text().splitlines()]
        pick = select(recs)
        for rep in range(args.repeats if "time" in stages else 0):
            for case, codes in pick.items():
                if args.cases and case not in args.cases.split(","):
                    continue
                order = list(codes)[rep % len(codes):] + list(codes)[: rep % len(codes)]
                for p in pts[case]:
                    for code in order:
                        one("time", code, case, codes[code], [p], repeats=2)
        for rep in range(args.repeats + 1 if "cached" in stages else 0):  # the first fills the cache
            for case, codes in pick.items():
                if "dkx" in codes and (not args.cases or case in args.cases.split(",")):
                    for p in pts[case]:
                        one("cached" if rep else "fill", "dkx", case, codes["dkx"], [p], repeats=2)
        if "gpu" in stages:
            for case, codes in pick.items():
                if args.cases and case not in args.cases.split(","):
                    continue
                for p in pts[case]:
                    for code in ("dkx", "yancc"):
                        if code in codes:
                            one("gpu", code, case, codes[code], [p], repeats=2, gpu=args.gpu)


def _err(vals, ref):
    return max(abs(v - r) / max(abs(r), 1e-300) for v, r in zip(vals, ref))


def convergence(recs):
    """{case: {code: [(res, max error over points vs finest rung)]}}."""
    out = {}
    for r in recs:
        if r["stage"] == "ladder" and r["status"] == 0 and r["results"]:
            out.setdefault(r["case"], {}).setdefault(r["code"], {})[tuple(r["res"])] = r["results"]
    conv = {}
    for case, codes in out.items():
        for code, rungs in codes.items():
            ladder = [tuple(x) for x in (DKE_LADDER if case == "DKE" else LADDER) if tuple(x) in rungs]
            ref = {tuple(q["point"]): q["values"] for q in rungs[ladder[-1]]}
            conv.setdefault(case, {})[code] = [
                (res, max(_err(q["values"], ref[tuple(q["point"])]) for q in rungs[res])) for res in ladder[:-1]]
    return conv


def select(recs):
    """Cheapest rung within TARGET of the code's finest rung, per case and code."""
    pick = {}
    for case, codes in convergence(recs).items():
        for code, rows in codes.items():
            ok = [res for res, e in rows if e <= (DKE_TARGET if case == "DKE" else TARGET)]
            if ok:
                pick.setdefault(case, {})[code] = ok[0]
    return pick


def summary(recs):
    """Headline rows: per case and code, the matched rung's median cold/warm time and peak memory."""
    import numpy as np

    conv, pick, rows = convergence(recs), select(recs), []
    for case, codes in pick.items():
        for code, res in codes.items():
            err = dict(conv[case][code])[tuple(res)]
            row = dict(case=case, code=code, res=res, error=err)
            for dev in ("cpu", "gpu"):
                rs = [r for r in recs if r["stage"] == ("time" if dev == "cpu" else "gpu") and r["case"] == case
                      and r["code"] == code and r["status"] == 0 and r["results"]]
                if not rs:
                    continue
                t = np.array([[r["process_wall"], r["results"][0]["times"][-1]] for r in rs])
                row[dev] = dict(cold=float(np.median(t[:, 0] if code == "monkes" else
                                                     [r["results"][0]["times"][0] for r in rs])),
                                warm=float(np.median(t[:, 1])), rss=max(r["peak_rss"] for r in rs) / 2**30,
                                gpu_mem=max((r["gpu_peak_bytes"] or 0) for r in rs) / 2**30, n=len(rs))
            rs = [r for r in recs if r["stage"] == "cached" and r["case"] == case and r["code"] == code and r["results"]]
            if rs and "cpu" in row:
                row["cpu"]["cold_cached"] = float(np.median([r["results"][0]["times"][0] for r in rs]))
            rows.append(row)
    return rows


def report(path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = json.loads(path.read_text())["records"]
    rows = summary(recs)
    print("| Case | Code | Resolution | Error vs finest | CPU cold [s] | CPU cold, cached [s] | CPU warm [s] | Peak RSS [GB] | GPU warm [s] | GPU peak [GB] |")
    print("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in rows:
        c, g = r.get("cpu", {}), r.get("gpu", {})
        f = lambda d, k, fmt: format(d[k], fmt) if k in d else "-"  # noqa: E731
        print(f"| {r['case']} | {r['code']} | {'x'.join(map(str, r['res']))} | {r['error']:.2%} | {f(c, 'cold', '.2f')} | {f(c, 'cold_cached', '.2f')} | "
              f"{f(c, 'warm', '.2f')} | {f(c, 'rss', '.2f')} | {f(g, 'warm', '.2f')} | {f(g, 'gpu_mem', '.2f')} |")
    print("\nConvergence (largest relative change against the finest rung):")
    for case, codes in convergence(recs).items():
        for code, rs in codes.items():
            print(f"  {case:5s} {code:6s} " + "  ".join(f"{'x'.join(map(str, q))}:{e:.1e}" for q, e in rs))
    cases = list(dict.fromkeys(r["case"] for r in rows))
    codes = ("monkes", "dkx", "yancc")
    colors = dict(monkes="#7a5195", dkx="#1f77b4", yancc="#ef8c2e")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for j, code in enumerate(codes):
        sel = {r["case"]: r for r in rows if r["code"] == code and "cpu" in r}
        x = [i + 0.27 * (j - 1) for i, c in enumerate(cases) if c in sel]
        ax[0].bar(x, [sel[c]["cpu"]["cold"] for c in cases if c in sel], 0.25, color=colors[code], alpha=0.35)
        ax[0].bar(x, [sel[c]["cpu"]["warm"] for c in cases if c in sel], 0.25, color=colors[code], label=code.upper())
        ax[1].bar(x, [sel[c]["cpu"]["rss"] for c in cases if c in sel], 0.25, color=colors[code])
    for a, lab in zip(ax, ("wall time per solve [s] (light: cold)", "peak RSS [GB]")):
        a.set_xticks(range(len(cases)), [c if c != "DKE" else "NCSX full DKE" for c in cases])
        a.set_yscale("log"), a.set_ylabel(lab)
    ax[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "cross_code_speed.png", dpi=150)


def main():
    import argparse

    if len(sys.argv) > 2 and sys.argv[1] == "worker":
        print(json.dumps(worker(*json.loads(sys.argv[2]))))
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="report", choices=["report", "measure"])
    ap.add_argument("--monkes"), ap.add_argument("--yancc-python"), ap.add_argument("--data")
    ap.add_argument("--ncores", type=int, default=4), ap.add_argument("--physical", type=int, default=18)
    ap.add_argument("--gpu", type=int, default=0), ap.add_argument("--xla-threads", type=int, default=16)
    ap.add_argument("--cases", default=""), ap.add_argument("--dkx-xla-threads", type=int, default=0)
    ap.add_argument("--work", default="cross_code_speed_work"), ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--repeats", type=int, default=3), ap.add_argument("--stages", default="ladder,time,gpu")
    args = ap.parse_args()
    if args.mode == "measure":
        measure(args)
        recs = [json.loads(ln) for ln in (Path(args.work) / "raw.jsonl").read_text().splitlines()]
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "cross_code_speed.json").write_text(json.dumps({"records": recs}, indent=1) + "\n")
    report(OUT / "cross_code_speed.json")


if __name__ == "__main__":
    main()
