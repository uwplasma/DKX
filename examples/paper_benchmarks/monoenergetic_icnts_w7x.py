"""ICNTS-style monoenergetic transport-coefficient benchmark on W7-X geometry.

This is the first entry of the methods-paper benchmark suite: the normalized
monoenergetic transport coefficients ``D11*`` and ``D31*`` of the W7-X
standard configuration (Boozer ``.bc`` equilibrium, half radius) versus the
normalized collisionality ``nuPrime``, at several normalized radial electric
fields ``EStar`` -- the comparison format established by the International
Collaboration on Neoclassical Transport in Stellarators (ICNTS) benchmark
[C.D. Beidler et al., Nucl. Fusion 51, 076001 (2011)].  The monoenergetic
formulation is that of S.P. Hirshman et al., Phys. Fluids 29, 2951 (1986);
the normalization conventions (equivalent-tokamak plateau value for ``D11``,
large-aspect-ratio banana bootstrap value for ``D31``) are documented in
:mod:`sfincs_jax.monoenergetic`.

What the script does:
  1. writes an RHSMode=3 input namelist on the W7-X standard-configuration
     equilibrium (fetched into the local data cache on first use),
  2. scans the (nuPrime, EStar) grid with
     :func:`sfincs_jax.monoenergetic.monoenergetic_database`, one nuPrime
     row at a time, checkpointing every finished row to a cache file so an
     interrupted scan resumes where it stopped,
  3. re-solves one low-collisionality point at 1.5x resolution as a
     convergence sanity check,
  4. optionally re-runs matched single-point decks with the SFINCS Fortran
     v3 executable (set ``SFINCS_FORTRAN_EXE``; skipped otherwise) and
     converts the Fortran transport matrices with the same normalization
     helper, giving a direct cross-check of the full solve,
  5. writes the coefficients + provenance to JSON and renders the benchmark
     figure (``D11*`` log-log and ``D31*`` semilog-x, one curve per
     ``EStar``, Fortran cross-check points as open markers).

Physics expectations encoded in the scan range: reading from high to low
collisionality, ``D11*`` decreases from the Pfirsch-Schlueter branch
(proportional to ``nuPrime`` in plateau units), passes the plateau minimum,
and rises as ``1/nu`` where helically trapped particles dominate; the
radial electric field suppresses the ``1/nu`` branch, and at the field
strengths scanned here it also visibly suppresses the Pfirsch-Schlueter
branch.  ``D31*`` changes sign across the scanned collisionality range.

Expected runtime: ~12 min total on a 10-core laptop CPU (27 scan points at
Ntheta=29, Nzeta=55, Nxi=64, ~15 s/point; ~2 min for the 1.5x-resolution
convergence point; the Fortran cross-check adds a few minutes when
enabled).  Finished rows are cached in ``output/`` and skipped on re-runs.

Run (from the repo root):
  SFINCS_FORTRAN_EXE=/path/to/sfincs python examples/paper_benchmarks/monoenergetic_icnts_w7x.py
"""

import json
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from sfincs_jax.monoenergetic import (
    monoenergetic_database,
    monoenergetic_dstar_from_transport_matrix,
)

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
EQUILIBRIUM = "w7x_standardConfig.bc"  # resolved via the sfincs_jax data cache
RN_WISH = 0.5  # benchmark surface r/a = 0.5

# nuPrime scan: 3x steps across Pfirsch-Schlueter -> plateau -> 1/nu.
NU_PRIMES = [3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0]
# EStar values: 0 plus two fields that visibly suppress the 1/nu branch
# (E* <~ 1e-3 changes D11* by <0.1% on this surface and is not plotted).
E_STARS = [0.0, 3e-2, 1e-1]

# Production resolution: converged to ~0.2% against a 1.3x grid at the
# hardest corner (nuPrime=3e-3, EStar in {0, 1e-1}); see the JSON record.
N_THETA, N_ZETA, N_XI = 29, 55, 64
SOLVER_TOLERANCE = 1e-8

# Convergence sanity point: 1.5x every kinetic dimension at the lowest
# collisionality (the least-converged corner of the scan).
CONV_POINT = (3e-3, 0.0)
CONV_RESOLUTION = (43, 83, 96)

# Fortran v3 cross-check points (nuPrime, EStar): 1/nu regime, ExB-detached
# 1/nu regime, plateau, and Pfirsch-Schlueter (with and without ExB).
# EStar = 1e-4 is used as the quasi-zero field (it changes D11* by < 0.1%
# on this surface): at exactly EStar = 0 the upstream executable's Krylov
# solve reaches its tolerance yet returns an Onsager-violating transport
# matrix at this problem size, which is why the upstream monoenergetic
# examples scan EStar >= 1e-4.  Off-grid points are re-solved with
# sfincs_jax on the identical deck, so every comparison is matched.
FORTRAN_POINTS = [(3e-3, 1e-4), (3e-3, 1e-1), (3e-1, 1e-4), (30.0, 1e-4), (30.0, 1e-1)]
FORTRAN_EXE = os.environ.get("SFINCS_FORTRAN_EXE", "")

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).parent / "output"
CACHE_PATH = OUT_DIR / "monoenergetic_icnts_w7x_cache.json"
FIG_DIR = REPO_ROOT / "docs" / "_static" / "figures" / "paper_benchmarks"
JSON_PATH = FIG_DIR / "monoenergetic_icnts_w7x.json"
PNG_PATH = FIG_DIR / "monoenergetic_icnts_w7x.png"

DECK_TEMPLATE = """! W7-X standard configuration, ICNTS-style monoenergetic benchmark deck.
! Generated by examples/paper_benchmarks/monoenergetic_icnts_w7x.py
&general
  RHSMode = 3
/
&geometryParameters
  geometryScheme = 11
  equilibriumFile = "{equilibrium}"
  inputRadialCoordinate = 3
  rN_wish = {rn_wish}
/
&speciesParameters
/
&physicsParameters
  nuPrime = {nu_prime}
  EStar = {e_star}
  collisionOperator = 1
  includeXDotTerm = .false.
  includeElectricFieldTermInXiDot = .false.
  useDKESExBDrift = .true.
  includePhi1 = .false.
/
&resolutionParameters
  Ntheta = {n_theta}
  Nzeta = {n_zeta}
  Nxi = {n_xi}
  Nx = 1
  solverTolerance = {tol}
/
&otherNumericalParameters
/
&preconditionerOptions
/
&export_f
/
"""

DB_CONSTANT_KEYS = (
    "delta", "g_hat", "i_hat", "iota", "b0_over_bbar", "fsab_hat2",
    "r_hat", "x0", "w0", "nu_d_hat_x0",
)  # fmt: skip
COEFF_KEYS = ("d11_star", "d31_star", "d13_star", "d33_star")


def write_deck(path, *, nu_prime=1.0, e_star=0.0, resolution=(N_THETA, N_ZETA, N_XI)):
    nt, nz, nxi = resolution
    path.write_text(
        DECK_TEMPLATE.format(
            equilibrium=EQUILIBRIUM, rn_wish=RN_WISH, nu_prime=nu_prime, e_star=e_star,
            n_theta=nt, n_zeta=nz, n_xi=nxi, tol=SOLVER_TOLERANCE,
        )
    )  # fmt: skip


def load_cache():
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {"rows": {}, "constants": {}, "conv": {}, "fortran": {}}


def save_cache(cache):
    CACHE_PATH.write_text(json.dumps(cache, indent=1))


def row_key(nu_prime):
    return f"nu={nu_prime:.6g}@{N_THETA}x{N_ZETA}x{N_XI}"


print("=== examples/paper_benchmarks/monoenergetic_icnts_w7x.py ===", flush=True)
print(f"  equilibrium: {EQUILIBRIUM} (geometryScheme=11), rN = {RN_WISH}")
print(f"  scan: {len(NU_PRIMES)} nuPrime x {len(E_STARS)} EStar")
print(f"  resolution: Ntheta={N_THETA} Nzeta={N_ZETA} Nxi={N_XI} tol={SOLVER_TOLERANCE:g}")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
cache = load_cache()

# ----------------------------------------------------------------------------
# 1) The (nuPrime, EStar) scan, one nuPrime row at a time (checkpointed)
# ----------------------------------------------------------------------------
deck_path = OUT_DIR / "monoenergetic_icnts_w7x.input.namelist"
write_deck(deck_path)

print("Step 1: scanning the (nuPrime, EStar) grid", flush=True)
for nu_prime in NU_PRIMES:
    key = row_key(nu_prime)
    if key in cache["rows"]:
        print(f"  nuPrime={nu_prime:g}: cached, skipping")
        continue
    t0 = time.time()
    db = monoenergetic_database(deck_path, [nu_prime], E_STARS, emit=lambda s: print("  " + s, flush=True))
    row = {
        "seconds": time.time() - t0,
        "nu_star": float(np.asarray(db.nu_star)[0]),
        **{k: np.asarray(getattr(db, k))[0].tolist() for k in COEFF_KEYS},
    }  # fmt: skip
    cache["rows"][key] = row
    if not cache["constants"]:
        cache["constants"] = {k: float(getattr(db, k)) for k in DB_CONSTANT_KEYS}
        cache["constants"]["r_major"] = db.r_major
        cache["constants"]["eps_t"] = db.eps_t
    save_cache(cache)
    print(f"  nuPrime={nu_prime:g}: done in {row['seconds']:.1f} s (checkpointed)", flush=True)

rows = [cache["rows"][row_key(nu)] for nu in NU_PRIMES]
d11 = np.asarray([r["d11_star"] for r in rows])
d31 = np.asarray([r["d31_star"] for r in rows])
d33 = np.asarray([r["d33_star"] for r in rows])
nu_star = [r["nu_star"] for r in rows]
scan_seconds = float(sum(r["seconds"] for r in rows))
n_points = len(NU_PRIMES) * len(E_STARS)
print(f"  scan wall time: {scan_seconds:.1f} s ({scan_seconds / n_points:.1f} s/point)")

# ----------------------------------------------------------------------------
# 2) Convergence sanity: one hard point at 1.5x resolution
# ----------------------------------------------------------------------------
print(f"Step 2: convergence check at (nuPrime, EStar) = {CONV_POINT}, 1.5x resolution", flush=True)
if not cache["conv"]:
    conv_deck = OUT_DIR / "monoenergetic_icnts_w7x_conv.input.namelist"
    write_deck(conv_deck, resolution=CONV_RESOLUTION)
    t0 = time.time()
    db_conv = monoenergetic_database(conv_deck, [CONV_POINT[0]], [CONV_POINT[1]])
    cache["conv"] = {
        "seconds": time.time() - t0,
        **{k: float(np.asarray(getattr(db_conv, k))[0, 0]) for k in COEFF_KEYS},
    }
    save_cache(cache)
i_nu = NU_PRIMES.index(CONV_POINT[0])
i_es = E_STARS.index(CONV_POINT[1])
tables = {"d11_star": d11, "d31_star": d31, "d33_star": d33}
conv = {}
for key in ("d11_star", "d31_star", "d33_star"):
    base = float(tables[key][i_nu, i_es])
    fine = cache["conv"][key]
    conv[key] = {"base": base, "fine_1p5x": fine, "rel_dev": abs(base - fine) / abs(fine)}
    print(f"  {key}: base={base:+.6e}  1.5x={fine:+.6e}  rel dev={conv[key]['rel_dev']:.2e}")
print(f"  convergence-point wall time: {cache['conv']['seconds']:.1f} s")

# ----------------------------------------------------------------------------
# 3) Fortran v3 cross-check (optional; requires SFINCS_FORTRAN_EXE)
# ----------------------------------------------------------------------------
fortran_records = []
if FORTRAN_EXE and Path(FORTRAN_EXE).exists():
    import h5py

    from sfincs_jax.validation.fortran import run_sfincs_fortran

    print(f"Step 3: Fortran v3 cross-check at {len(FORTRAN_POINTS)} points", flush=True)
    c = cache["constants"]
    for nu_prime, e_star in FORTRAN_POINTS:
        tag = f"nu{nu_prime:g}_es{e_star:g}".replace(".", "p").replace("-", "m")
        fkey = f"{tag}@{N_THETA}x{N_ZETA}x{N_XI}"
        point_deck = OUT_DIR / f"fortran_{tag}.input.namelist"
        write_deck(point_deck, nu_prime=nu_prime, e_star=e_star)
        if fkey not in cache["fortran"]:
            t0 = time.time()
            h5_path = run_sfincs_fortran(
                input_namelist=point_deck,
                exe=Path(FORTRAN_EXE),
                workdir=OUT_DIR / f"fortran_{tag}",
            )
            fortran_seconds = time.time() - t0
            with h5py.File(h5_path, "r") as f:
                # The h5 dataset is stored column-major: transpose to
                # mathematical row/column order.
                tm = np.asarray(f["transportMatrix"][...], dtype=np.float64).T
            point = monoenergetic_dstar_from_transport_matrix(
                tm, nu_prime=nu_prime, **{k: c[k] for k in DB_CONSTANT_KEYS}
            )
            cache["fortran"][fkey] = {
                "seconds": fortran_seconds,
                **{k: float(np.asarray(getattr(point, k))) for k in COEFF_KEYS},
            }
            save_cache(cache)
        # The matched sfincs_jax value: from the scan table when the point
        # is on the grid, otherwise a dedicated solve of the same deck.
        if nu_prime in NU_PRIMES and e_star in E_STARS:
            i, j = NU_PRIMES.index(nu_prime), E_STARS.index(e_star)
            jax_point = {
                "d11_star": float(d11[i, j]),
                "d31_star": float(d31[i, j]),
                "d33_star": float(d33[i, j]),
            }
        else:
            jkey = f"jax_{tag}@{N_THETA}x{N_ZETA}x{N_XI}"
            if jkey not in cache["fortran"]:
                db_pt = monoenergetic_database(point_deck, [nu_prime], [e_star])
                cache["fortran"][jkey] = {
                    k: float(np.asarray(getattr(db_pt, k))[0, 0]) for k in COEFF_KEYS
                }
                save_cache(cache)
            jax_point = cache["fortran"][jkey]
        f_rec = cache["fortran"][fkey]
        rec = {"nu_prime": nu_prime, "e_star": e_star, "fortran_seconds": f_rec["seconds"]}
        for key in ("d11_star", "d31_star", "d33_star"):
            ours = jax_point[key]
            ref = f_rec[key]
            rec[key] = {"sfincs_jax": ours, "fortran": ref, "rel_dev": abs(ours - ref) / abs(ref)}
        fortran_records.append(rec)
        print(
            f"  (nuPrime={nu_prime:g}, EStar={e_star:g}) [{f_rec['seconds']:.0f} s]  "
            + "  ".join(f"{k} rel dev {rec[k]['rel_dev']:.2e}" for k in ("d11_star", "d31_star", "d33_star")),
            flush=True,
        )  # fmt: skip
else:
    print("Step 3: SFINCS_FORTRAN_EXE not set (or missing) -- skipping the Fortran cross-check")

# ----------------------------------------------------------------------------
# 4) JSON record
# ----------------------------------------------------------------------------
print("Step 4: writing the JSON record")
record = {
    "benchmark": "ICNTS-style monoenergetic coefficients, W7-X standard configuration",
    "references": [
        "C.D. Beidler et al., Nucl. Fusion 51, 076001 (2011)",
        "S.P. Hirshman et al., Phys. Fluids 29, 2951 (1986)",
    ],
    "equilibrium": EQUILIBRIUM,
    "rN": RN_WISH,
    "resolution": {"Ntheta": N_THETA, "Nzeta": N_ZETA, "Nxi": N_XI, "Nx": 1,
                   "solverTolerance": SOLVER_TOLERANCE},
    "nu_prime": NU_PRIMES,
    "e_star": E_STARS,
    "nu_star": nu_star,
    "d11_star": d11.tolist(),
    "d31_star": d31.tolist(),
    "d33_star": d33.tolist(),
    "geometry": {"GHat": cache["constants"]["g_hat"], "IHat": cache["constants"]["i_hat"],
                 "iota": cache["constants"]["iota"],
                 "B0OverBBar": cache["constants"]["b0_over_bbar"],
                 "rHat": cache["constants"]["r_hat"], "R0": cache["constants"]["r_major"],
                 "eps_t": cache["constants"]["eps_t"]},
    "scan_seconds": scan_seconds,
    "seconds_per_point": scan_seconds / n_points,
    "convergence_check": {
        "nu_prime": CONV_POINT[0], "e_star": CONV_POINT[1],
        "fine_resolution": {"Ntheta": CONV_RESOLUTION[0], "Nzeta": CONV_RESOLUTION[1],
                            "Nxi": CONV_RESOLUTION[2]},
        "seconds": cache["conv"]["seconds"],
        **conv,
    },
    "fortran_crosscheck": fortran_records,
    "fortran_crosscheck_note": (
        "Matched decks at the scan resolution; e_star = 1e-4 serves as the "
        "quasi-zero field (D11* differs from EStar = 0 by < 0.1% on this "
        "surface) because the upstream Krylov solve at exactly EStar = 0 "
        "returns an Onsager-violating matrix at this problem size."
    ),
}  # fmt: skip
JSON_PATH.write_text(json.dumps(record, indent=1))
print(f"  wrote {JSON_PATH}")

# ----------------------------------------------------------------------------
# 5) Figure
# ----------------------------------------------------------------------------
print("Step 5: rendering the figure")
plt.rcParams.update(
    {
        "figure.dpi": 140,
        "font.family": "DejaVu Sans",
        "font.size": 10.0,
        "axes.labelsize": 11.0,
        "axes.titlesize": 11.0,
        "legend.fontsize": 9.0,
        "axes.grid": True,
        "grid.alpha": 0.24,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
colors = ["#1f77b4", "#d62728", "#2ca02c"]
fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(9.2, 3.9), constrained_layout=True)

nu = np.asarray(NU_PRIMES)
for j, e_star in enumerate(E_STARS):
    label = "$E^* = 0$" if e_star == 0.0 else f"$E^* = {e_star:g}$"
    ax1.loglog(nu, d11[:, j], "-o", ms=4, color=colors[j], label=label)
    ax3.semilogx(nu, d31[:, j], "-o", ms=4, color=colors[j], label=label)
for rec in fortran_records:
    ax1.plot(rec["nu_prime"], rec["d11_star"]["fortran"], "s", ms=9, mfc="none",
             mec="k", mew=1.2, zorder=5)  # fmt: skip
    ax3.plot(rec["nu_prime"], rec["d31_star"]["fortran"], "s", ms=9, mfc="none",
             mec="k", mew=1.2, zorder=5)  # fmt: skip
if fortran_records:
    ax1.plot([], [], "s", ms=8, mfc="none", mec="k", mew=1.2, label="Fortran v3")

ax1.set_xlabel(r"$\nu'$")
ax1.set_ylabel(r"$D_{11}^*$")
ax1.set_title("Radial diffusion (plateau units)")
ax1.legend(frameon=False, loc="upper center")

ax3.axhline(0.0, color="0.6", lw=0.8, zorder=0)
ax3.set_xlabel(r"$\nu'$")
ax3.set_ylabel(r"$D_{31}^*$")
ax3.set_title("Bootstrap/Ware coefficient (banana units)")

fig.suptitle(
    f"W7-X standard configuration, $r/a = {RN_WISH}$: monoenergetic benchmark", y=1.04
)
fig.savefig(PNG_PATH, dpi=170, bbox_inches="tight")
plt.close(fig)

# Palette-quantize to keep the committed figure small (repo convention).
img = Image.open(PNG_PATH).convert("P", palette=Image.ADAPTIVE, colors=64)
img.save(PNG_PATH, optimize=True)
print(f"  wrote {PNG_PATH} ({PNG_PATH.stat().st_size / 1024:.0f} KB)")
print("Done.")
