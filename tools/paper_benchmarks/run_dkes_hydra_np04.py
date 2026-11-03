#!/usr/bin/env python3
"""Run DKX's monoenergetic route at the DKES points of the HELIAS hydra_Np04_20190108 reactor.

Neither the Boozer file (J. Geiger, IPP) nor the DKES results are in the repository:
the DKES data cannot be redistributed. Point ``DKX_HYDRA_BOOZER`` at the Boozer file and
``DKX_DKES_HYDRA_DIR`` at the directory with ``DKES_rho_<rho>.h5``. ``run`` writes one JSON
per (rho, efield) column (DKX D* at every DKES ``cmul``); ``record`` joins those with the
DKES points into ``validation/dkes_hydra_np04_v1.json``, which stores DKX values and
agreement statistics per regime, never the DKES values. ``audit_dkes_hydra_np04.py`` checks
the gates on the record, and recomputes the statistics when the DKES directory is set.

    python tools/paper_benchmarks/run_dkes_hydra_np04.py run raw.json --grid 17 41 96 \
        --rho 0.25 0.5 --efield 0.0 0.01 0.03
    python tools/paper_benchmarks/run_dkes_hydra_np04.py record raw_a.json raw_b.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from tools.release.artifacts import dkes_transport_to_beidler  # noqa: E402

RECORD = ROOT / "validation" / "dkes_hydra_np04_v1.json"
FIRST_SURFACE_RN = 0.1224745  # sqrt(0.015): DKX needs a bracketing pair, so rho = 0.12247 is nudged up
DECK = """&general
  RHSMode = 3
/
&geometryParameters
  geometryScheme = 11
  equilibriumFile = "{boozer}"
  inputRadialCoordinate = 3
  rN_wish = {rho}
  VMECRadialOption = 0
/
&speciesParameters
/
&physicsParameters
  collisionOperator = 1
  includeXDotTerm = .false.
  includeElectricFieldTermInXiDot = .false.
  useDKESExBDrift = .true.
  includePhi1 = .false.
/
&resolutionParameters
  Ntheta = {nt}
  Nzeta = {nz}
  Nxi = {nxi}
  Nx = 1
  solverTolerance = 1d-8
/
"""


def dkes_dir(path: Path | str | None = None) -> Path | None:
    """Directory of the private ``DKES_rho_<rho>.h5`` files (``DKX_DKES_HYDRA_DIR``), or None."""
    value = path or os.environ.get("DKX_DKES_HYDRA_DIR")
    return Path(value) if value and Path(value).is_dir() else None


def reference_rows(directory: Path | str | None = None) -> list[dict[str, float | str]]:
    """The DKES points of the private ``.h5`` files: ``cmul = nuprime/nuprimefac``, ``|efield| = |Erstar/Estarfac|``."""
    import h5py

    root = dkes_dir(directory)
    if root is None:
        raise FileNotFoundError("DKX_DKES_HYDRA_DIR (directory with DKES_rho_<rho>.h5) is not set")
    rows: list[dict[str, float | str]] = []
    for path in sorted(root.glob("DKES_rho_*.h5")):
        rho = path.stem.removeprefix("DKES_rho_")
        with h5py.File(path, "r") as f:
            cmul = np.asarray(f["nuprime"]) / float(f["nuprimefac"][()])
            efield = np.abs(np.asarray(f["Erstar"]) / float(f["Estarfac"][()]))
            data = {k: np.asarray(f[k]) for k in ("D11_S", "D13_S", "D31_S", "D33_S")}
        for i in range(len(cmul)):
            rows.append({
                "rho": rho, "cmul": float(f"{cmul[i]:.6g}"), "efield": float(f"{efield[i]:.6g}"),
                **{k: float(v[i]) for k, v in data.items()},
            })
    return rows


REGIMES = (("collisional", 0.1, 10.0), ("plateau", 1e-3, 3e-2), ("low_nu", 0.0, 3e-4))


def agreement(columns: list[dict], reference: list[dict]) -> list[dict]:
    """Per (rho, E class, regime) statistics of DKX against DKES, rounded to 0.1% (D31*: 1e-3).

    ``columns`` are the record's columns (geometry and DKX ``D*`` per ``cmul``), ``reference`` the
    DKES points. Only these statistics are stored: they do not reveal a DKES value.
    """
    out: list[dict] = []
    for rho in dict.fromkeys(c["rho"] for c in columns):
        for e_class in ("zero", "finite"):
            for name, lo, hi in REGIMES:
                d11, d33, d31 = [], [], []
                for col in columns:
                    if col["rho"] != rho or (col["efield"] == 0.0) != (e_class == "zero"):
                        continue
                    for row in col["rows"]:
                        if not lo <= row["cmul"] <= hi:
                            continue
                        ref = next((r for r in reference if r["rho"] == rho and abs(r["cmul"] / row["cmul"] - 1) < 1e-3
                                    and abs(r["efield"] - col["efield"]) <= 1e-9 + 1e-3 * col["efield"]), None)
                        if ref is None:
                            continue
                        b = reference_beidler(ref, col["geometry"])
                        d11.append(abs(row["dkx"]["D11_star"] / b["D11_star"] - 1.0))
                        d33.append(abs(row["dkx"]["D33_star"] / b["D33_star"] - 1.0))
                        d31.append(abs(row["dkx"]["D31_star"] - b["D31_star"]))
                if not d11:
                    continue
                out.append({
                    "rho": rho, "efield": e_class, "regime": name, "points": len(d11),
                    "d11_median_pct": round(100 * float(np.median(d11)), 1), "d11_max_pct": round(100 * max(d11), 1),
                    "d33_max_pct": round(100 * max(d33), 1),
                    "d31_max_abs": None if e_class == "zero" else round(max(d31), 3),
                })
    return out


def deck_text(boozer: Path, rho: float, grid: tuple[int, int, int]) -> str:
    rn = max(float(rho), FIRST_SURFACE_RN + 1e-7)
    return DECK.format(boozer=boozer, rho=rn, nt=grid[0], nz=grid[1], nxi=grid[2])


def surface_geometry(boozer: Path, rho: float) -> dict[str, float]:
    """DKX's flux-surface constants at ``rho`` from a minimal solve."""
    from dkx.monoenergetic import monoenergetic_database

    with tempfile.TemporaryDirectory() as tmp:
        deck = Path(tmp) / "probe.namelist"
        deck.write_text(deck_text(boozer, rho, (5, 5, 4)))
        db = monoenergetic_database(deck, [1.0], [0.0], tol=1e-6)
    return {
        "g": float(db.g_hat), "I": float(db.i_hat), "iota": float(db.iota), "b0": float(db.b0_over_bbar),
        "r": float(db.r_hat), "nud": float(db.nu_d_hat_x0), "x0": float(db.x0), "w0": float(db.w0),
        "delta": float(db.delta), "fsab2": float(np.ravel(db.fsab_hat2)[0]),
    }


def run(boozer: Path, out: Path, grid: tuple[int, int, int], rhos, efields, cmuls) -> None:
    """DKX D* at ``cmul x efield`` for each rho; ``efield = E_r/(v B0)``, ``cmul = nu_D/v`` [1/m]."""
    from dkx.monoenergetic import monoenergetic_database

    results = json.loads(out.read_text()) if out.exists() else {}
    for rho in rhos:
        geometry = surface_geometry(boozer, rho)
        r_major = abs(geometry["g"]) / geometry["b0"]
        nu_prime_per_cmul = r_major / geometry["nud"]
        for efield in efields:
            key = f"{rho}|{efield}"
            if key in results:
                continue
            e_star = efield * r_major / (abs(geometry["iota"]) * geometry["r"])  # v_E = efield
            with tempfile.TemporaryDirectory() as tmp:
                deck = Path(tmp) / "case.namelist"
                deck.write_text(deck_text(boozer, rho, grid))
                start = time.time()
                db = monoenergetic_database(deck, [c * nu_prime_per_cmul for c in cmuls], [e_star], tol=1e-8)
            results[key] = {
                "geometry": geometry, "grid": list(grid), "efield": efield, "cmul": list(cmuls),
                "seconds": time.time() - start,
                **{n: np.ravel(getattr(db, f"{n}_star")).tolist() for n in ("d11", "d13", "d31", "d33")},
            }
            out.write_text(json.dumps(results))
            print(key, f"{results[key]['seconds']:.0f} s", flush=True)


GATES = {"cmul_min": 1e-3, "d11_rel": 0.25, "d33_rel": 0.01, "d31_abs": 0.003}
EXCLUDED = [{"rho": "0.875", "efield": 0.0, "coefficient": "D11_star", "reason": "the stored D11_S at efield = 0 is inconsistent with the .dk output of the same run, which DKX tracks at every radius to a constant factor; reported to the file's owner"}]
EXCLUSIONS = [
    "cmul below 1e-3 is reported, not gated: the DKES variational bounds are wide there and DKX's 17 x 41 x 96 grid is not converged (see convergence)",
    "D31* at efield = 0 is not compared: DKES returns noise of either sign at its roundoff floor",
    "rho = 0.5 at efield = 0 has no DKES reference in the supplied files",
    "rho = 0.12247 is run at the first surface of the Boozer file plus 1e-7 in rN, because DKX interpolates between two surfaces",
    "the DKES values cannot be redistributed: the record stores agreement statistics per regime, and the audit recomputes them only when DKX_DKES_HYDRA_DIR is set",
    "DKES reports one value per point; no lower/upper variational bounds are in the files, so the comparison is against the single stored value",
    "NTX is compared in its own repository (uwplasma/NTX#40), not in this record",
]


def reference_beidler(row: dict, geometry: dict) -> dict[str, float]:
    tm = [[row["D11_S"], row["D13_S"]], [row["D31_S"], row["D33_S"]]]
    return dkes_transport_to_beidler(tm, cmul=row["cmul"], geometry=geometry)


def build_record(raw_files: list[Path], *, environment: dict, source: dict, boozer: Path, convergence: dict) -> dict:
    """Join the DKX columns with the DKES reference; the audit recomputes everything stored here."""
    import hashlib

    raw: dict = {}
    for path in raw_files:
        raw.update(json.loads(path.read_text()))
    reference = reference_rows()
    columns = []  # DKX values and geometry only: the DKES points are not stored
    for key in sorted(raw, key=lambda k: (float(k.split("|")[0]), float(k.split("|")[1]))):
        rho, efield = key.split("|")
        col = raw[key]
        rows = []
        for ref in reference:
            if ref["rho"] != rho or abs(ref["efield"] - float(efield)) > 1e-9 + 1e-3 * float(efield):
                continue
            idx = [i for i, c in enumerate(col["cmul"]) if abs(c / ref["cmul"] - 1.0) < 1e-3]
            if not idx:
                continue
            i = idx[0]
            rows.append({
                "cmul": ref["cmul"],
                "dkx": {"D11_star": col["d11"][i], "D31_star": col["d31"][i], "D33_star": col["d33"][i]},
            })
        if rows:
            columns.append({"rho": rho, "efield": float(efield), "geometry": col.get("geometry") or col["geom"], "grid": col["grid"],
                            "rows": sorted(rows, key=lambda r: -r["cmul"])})
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()  # noqa: E731
    return {
        "schema": "dkx.dkes_hydra_np04.v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "claim_scope": "hydra_np04_dkes_monoenergetic_pas_dkes_trajectories_seven_radii",
        "source": source,
        "environment": environment,
        "reference": {
            "provenance": "DKES, IPP, hydra_Np04_20190108; supplied by the repository owner as DKES_rho_<rho>.h5, not redistributable: only agreement statistics are stored",
            "dkes_dir_env": "DKX_DKES_HYDRA_DIR", "boozer_env": "DKX_HYDRA_BOOZER",
            "boozer_file": boozer.name, "boozer_sha256": sha(boozer), "boozer_bytes": boozer.stat().st_size,
        },
        "normalization": {
            "nu_prime": "nuPrime = cmul |G + iota I| / (B0 nuD(x0)), nuD(1) = 0.836028 (DKX applies nuPrime B0 nuD / |G + iota I| = cmul)",
            "e_star": "EStar = efield |G| / (B0 |iota| r_hat), so v_E = |E_r|/(v B0) = efield",
            "d_s": "D_S are SFINCS transportMatrix entries; D* by dkes_transport_to_beidler (closed form of monoenergetic_dstar_from_transport_matrix)",
            "d33": "D33_S = -D33_dkes 2/(sqrt(pi) R0), R0 = |G|/B0: D33* = 1 at cmul = 10 to 1e-4",
        },
        "gates": GATES,
        "excluded": EXCLUDED,
        "columns": columns,
        "agreement": agreement(columns, reference),
        "convergence": convergence,
        "exclusions": EXCLUSIONS,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("out", type=Path)
    p_run.add_argument("--boozer", type=Path, default=os.environ.get("DKX_HYDRA_BOOZER"))
    p_run.add_argument("--grid", type=int, nargs=3, default=(17, 41, 96))
    p_run.add_argument("--rho", nargs="+", default=["0.12247", "0.25", "0.375", "0.5", "0.625", "0.75", "0.875"])
    p_run.add_argument("--efield", type=float, nargs="+", default=[0.0, 0.01, 0.03])
    p_run.add_argument("--cmul", type=float, nargs="+", default=[m * 10.0**k for k in range(-6, 1) for m in (1, 3)][:-1] + [10.0])
    p_rec = sub.add_parser("record")
    p_rec.add_argument("raw", type=Path, nargs="+")
    p_rec.add_argument("--boozer", type=Path, default=os.environ.get("DKX_HYDRA_BOOZER"))
    p_rec.add_argument("--meta", type=Path, required=True, help="JSON with source, environment, convergence")
    args = parser.parse_args()
    if args.cmd == "record":
        meta = json.loads(args.meta.read_text())
        RECORD.write_text(json.dumps(build_record(args.raw, environment=meta["environment"], source=meta["source"],
                                                  boozer=Path(args.boozer), convergence=meta["convergence"]), indent=1) + "\n")
        return 0
    if args.cmd == "run":
        if args.boozer is None:
            parser.error("--boozer or DKX_HYDRA_BOOZER is required")
        run(Path(args.boozer), args.out, tuple(args.grid), [float(r) for r in args.rho], args.efield, sorted(set(args.cmul)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
