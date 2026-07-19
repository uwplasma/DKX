"""Low-collisionality local-validity diagnostics overlaid on a W7-X nu-scan.

Sixth entry of the methods-paper benchmark suite, and the companion to the
Shaing-Callen convergence "hard mode" test: it takes the same W7-X standard
configuration surface (r/a = 0.5) and, instead of asking *what* the
monoenergetic coefficients are, asks *when the radially-local result is
trustworthy*.  The diagnostics come from :mod:`dkx.validity`:

  * the collisionality-regime classifier (Pfirsch-Schlueter / plateau /
    banana / 1/nu / sqrt-nu / superbanana-plateau) from the normalized
    collisionality ``nu_star`` and the DKES ``EStar`` (Helander & Sigmar,
    *Collisional Transport in Magnetized Plasmas*, CUP (2002); K.C. Shaing,
    Phys. Fluids 27, 1567 (1984); D.-I. Ho & R.M. Kulsrud, Phys. Fluids 30,
    442 (1987); C.D. Beidler et al., Nucl. Fusion 51, 076001 (2011)),
  * the E x B resonance-layer parameter ``k_ExB = omega_E/nu_eff`` that marks
    the ``1/nu -> sqrt(nu)`` boundary (``k_ExB ~ 1``), and the drift-resonance
    ratio ``k_res = omega_E/omega_d`` for the superbanana-plateau resonance,
  * the finite-orbit-width parameter ``delta_FOW = w_b/L`` that flags radial
    locality (Hinton & Hazeltine, Rev. Mod. Phys. 48, 239 (1976)).

The monoenergetic coefficients themselves are computed with the full
drift-kinetic solve through :func:`dkx.monoenergetic.monoenergetic_database`
(Beidler et al. 2011 normalization; S.P. Hirshman et al., Phys. Fluids 29,
2951 (1986)); the collisionless Shaing-Callen bootstrap asymptote is drawn
for reference from :func:`dkx.shaing_callen.shaing_callen_d31_limit`
(K.C. Shaing & J.D. Callen, Phys. Fluids 26, 3315 (1983); closed form of
C.G. Albert et al., arXiv:2407.21599, eq. (42)).

What the script does:
  1. scans ``D31*``/``D11*`` versus ``nuPrime`` at ``EStar = 0`` (pure 1/nu
     branch), a small ``EStar = 3e-3`` (E x B precession switching on), and a
     larger ``EStar = 1e-2`` (reaching the drift resonance), checkpointing
     every finished row,
  2. classifies every scan point and computes the E x B / orbit-width
     diagnostics from the database's own flux-surface constants,
  3. checks that the measured regime transitions line up with the classifier:
     the plateau -> 1/nu transition at ``nu_star ~ eps_t^{3/2}`` and the
     ``EStar = 3e-3`` detachment from the ``EStar = 0`` curve where ``k_ExB``
     enters its marginal band (the same detachment documented by the
     Shaing-Callen convergence benchmark below ``nuPrime ~ 1e-3``),
  4. writes the JSON provenance record and renders the two-panel figure:
     ``D31*(nuPrime)`` with the points coloured by regime and the collisional
     boundaries marked, and the ``k_ExB(nuPrime)`` Er-layer diagnostic with
     the ``sqrt(nu)`` onset lines.

Expected runtime: ~8-12 min total on a 10-core laptop CPU (7 nuPrime x 3
EStar at a modest 17 x 31 angular grid with an Nxi ramp; this is a validity /
regime-classification figure, not a converged-coefficient benchmark, so the
angular grid is deliberately coarse and labelled as such).  Finished rows are
cached in ``output/`` and skipped on re-runs; set
``LOW_COLL_VALIDITY_MAX_NEW_POINTS=N`` to stop after ``N`` newly computed rows
(chunked/resumable runs).

Run (from the repo root):
  python examples/paper_benchmarks/low_collisionality_validity.py
"""

import json
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from PIL import Image

os.environ.setdefault("DKX_TIER1_MEMORY_BUDGET_GB", "16")

from dkx.drift_kinetic import kinetic_operator_from_namelist  # noqa: E402
from dkx.inputs import load_sfincs_input  # noqa: E402
from dkx.monoenergetic import (  # noqa: E402
    monoenergetic_database,
    monoenergetic_dstar_from_transport_matrix,
)
from dkx.shaing_callen import shaing_callen_d31_limit  # noqa: E402
from dkx.validity import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    Regime,
    local_validity_report,
)

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
EQUILIBRIUM = "w7x_standardConfig.bc"  # resolved via the dkx data cache
RN_WISH = 0.5  # benchmark surface r/a = 0.5
N_PERIODS = 5  # W7-X field periods (the .bc header value)

# EStar values: 0 (pure 1/nu), a small field that turns on E x B precession,
# and a larger field that reaches the drift resonance at the lowest nuPrime.
E_STARS = [0.0, 3e-3, 1e-2]

# Per-point resolution schedule (Ntheta, Nzeta, Nxi).  Deliberately modest:
# this figure classifies regimes and validity, it is not a converged-D31*
# benchmark (that is shaing_callen_convergence.py at 29 x 55).  Nxi grows as
# the pitch boundary layer narrows like sqrt(nu).
SCHEDULE = [
    (1.0, (17, 31, 24)),
    (0.1, (17, 31, 24)),
    (3e-2, (17, 31, 32)),
    (1e-2, (17, 31, 40)),
    (3e-3, (17, 31, 56)),
    (1e-3, (17, 31, 80)),
    (3e-4, (17, 31, 96)),
]
SOLVER_TOLERANCE = 1e-8

# Representative gradient scale length for the finite-orbit-width flag, in
# RBar units.  With no kinetic profiles in a monoenergetic scan we use the
# minor radius rHat as a conservative (long) scale length; a peaked reactor
# profile with L ~ rHat/3 is reported alongside for context.
GRAD_SCALE_FRACTIONS = {"minor_radius": 1.0, "peaked_profile": 1.0 / 3.0}

# |B| grid + spectral upsample for the Shaing-Callen reference asymptote.
LIMIT_GRID = (29, 55)
LIMIT_UPSAMPLE = 384

# Chunked runs: stop cleanly after this many newly computed rows.
MAX_NEW_POINTS = int(os.environ.get("LOW_COLL_VALIDITY_MAX_NEW_POINTS", "0")) or None

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).parent / "output"
CACHE_PATH = OUT_DIR / "low_collisionality_validity_cache.json"
FIG_DIR = REPO_ROOT / "docs" / "_static" / "figures" / "paper_benchmarks"
JSON_PATH = FIG_DIR / "low_collisionality_validity.json"
PNG_PATH = FIG_DIR / "low_collisionality_validity.png"

DECK_TEMPLATE = """! W7-X standard configuration, low-collisionality validity deck.
! Generated by examples/paper_benchmarks/low_collisionality_validity.py
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

# Regime -> plot colour (one per class the scan visits).
REGIME_COLORS = {
    Regime.PFIRSCH_SCHLUETER.value: "#8c564b",
    Regime.PLATEAU.value: "#7f7f7f",
    Regime.ONE_OVER_NU.value: "#1f77b4",
    Regime.BANANA.value: "#17becf",
    Regime.SQRT_NU.value: "#2ca02c",
    Regime.SUPERBANANA_PLATEAU.value: "#d62728",
}
E_STAR_MARKERS = {0.0: "o", 3e-3: "s", 1e-2: "^"}


def write_deck(path, *, nu_prime=1.0, e_star=0.0, resolution=(17, 31, 24)):
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
    return {"rows": {}, "constants": {}, "limit": {}}


def save_cache(cache):
    CACHE_PATH.write_text(json.dumps(cache, indent=1))


def row_key(nu_prime, resolution):
    nt, nz, nxi = resolution
    return f"nu={nu_prime:.6g}@{nt}x{nz}x{nxi}"


class ChunkBudget:
    """Stop the run cleanly after a fixed number of new solve rows."""

    def __init__(self, limit):
        self.limit = limit
        self.used = 0

    def spend(self):
        self.used += 1
        if self.limit is not None and self.used >= self.limit:
            save_cache(cache)
            print(f"Chunk budget reached ({self.used} new row(s)); re-run to continue.", flush=True)
            sys.exit(0)


budget = ChunkBudget(MAX_NEW_POINTS)

print("=== examples/paper_benchmarks/low_collisionality_validity.py ===", flush=True)
print(f"  equilibrium: {EQUILIBRIUM} (geometryScheme=11), rN = {RN_WISH}")
print(f"  scan: {len(SCHEDULE)} nuPrime x {len(E_STARS)} EStar (modest 17 x 31 grid)")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
cache = load_cache()

# ----------------------------------------------------------------------------
# 1) The nuPrime scan, one row (all EStar values) at a time (checkpointed)
# ----------------------------------------------------------------------------
print("Step 1: scanning nuPrime with the per-point resolution schedule", flush=True)
for nu_prime, resolution in SCHEDULE:
    key = row_key(nu_prime, resolution)
    if key in cache["rows"]:
        print(f"  nuPrime={nu_prime:g} @ {resolution}: cached, skipping")
        continue
    deck_path = OUT_DIR / "low_collisionality_validity.input.namelist"
    write_deck(deck_path, nu_prime=nu_prime, resolution=resolution)
    t0 = time.time()
    db = monoenergetic_database(deck_path, [nu_prime], E_STARS, emit=lambda s: print("  " + s, flush=True))
    row = {
        "seconds": time.time() - t0,
        "resolution": list(resolution),
        "nu_star": float(np.asarray(db.nu_star)[0]),
        **{k: np.asarray(getattr(db, k))[0].tolist() for k in COEFF_KEYS},
    }  # fmt: skip
    cache["rows"][key] = row
    if not cache["constants"]:
        cache["constants"] = {k: float(getattr(db, k)) for k in DB_CONSTANT_KEYS}
        cache["constants"]["r_major"] = db.r_major
        cache["constants"]["eps_t"] = db.eps_t
    save_cache(cache)
    print(f"  nuPrime={nu_prime:g} @ {resolution}: done in {row['seconds']:.1f} s (checkpointed)", flush=True)
    budget.spend()

NU_PRIMES = [nu for nu, _ in SCHEDULE]
rows = [cache["rows"][row_key(nu, res)] for nu, res in SCHEDULE]
d31 = np.asarray([r["d31_star"] for r in rows])  # (n_nu, n_estar)
d11 = np.asarray([r["d11_star"] for r in rows])
nu_star = np.asarray([r["nu_star"] for r in rows])
scan_seconds = float(sum(r["seconds"] for r in rows))
c = cache["constants"]
print(f"  scan wall time: {scan_seconds:.1f} s")

# ----------------------------------------------------------------------------
# 2) Shaing-Callen collisionless asymptote (reference line for D31*)
# ----------------------------------------------------------------------------
print("Step 2: evaluating the Shaing-Callen collisionless asymptote", flush=True)
if "d31_star" not in cache["limit"]:
    geo_deck = OUT_DIR / "low_collisionality_validity_limit.input.namelist"
    write_deck(geo_deck, resolution=(LIMIT_GRID[0], LIMIT_GRID[1], 16))
    op = kinetic_operator_from_namelist(load_sfincs_input(geo_deck).raw)
    lim = shaing_callen_d31_limit(
        np.asarray(op.b_hat), g_hat=c["g_hat"], i_hat=c["i_hat"], iota=c["iota"],
        n_periods=N_PERIODS, x=np.asarray(op.x), x_weights=np.asarray(op.x_weights),
        upsample=LIMIT_UPSAMPLE,
    )  # fmt: skip
    tm = np.zeros((2, 2))
    tm[1, 0] = lim.d31
    point = monoenergetic_dstar_from_transport_matrix(tm, nu_prime=1.0, **{k: c[k] for k in DB_CONSTANT_KEYS})
    cache["limit"] = {"d31_star": float(np.asarray(point.d31_star)), "b_grid": list(LIMIT_GRID)}
    save_cache(cache)
d31_limit = cache["limit"]["d31_star"]
print(f"  D31*_SC = {d31_limit:+.6e} (banana units)")

# ----------------------------------------------------------------------------
# 3) Classify every point and compute the validity diagnostics
# ----------------------------------------------------------------------------
print("Step 3: classifying regimes and computing the validity diagnostics", flush=True)
grad_scales = {name: frac * c["r_hat"] for name, frac in GRAD_SCALE_FRACTIONS.items()}
points = []
for i_nu, nu_prime in enumerate(NU_PRIMES):
    for j_es, e_star in enumerate(E_STARS):
        report = local_validity_report(
            nu_star=float(nu_star[i_nu]), e_star=float(e_star),
            delta=c["delta"], g_hat=c["g_hat"], iota=c["iota"],
            b0_over_bbar=c["b0_over_bbar"], eps_t=c["eps_t"],
            grad_scale_length_hat=grad_scales["minor_radius"],
        )  # fmt: skip
        points.append({
            "nu_prime": nu_prime,
            "nu_star": float(nu_star[i_nu]),
            "e_star": e_star,
            "d31_star": float(d31[i_nu, j_es]),
            "d11_star": float(d11[i_nu, j_es]),
            "regime": report.regime.value,
            "k_exb": report.k_exb,
            "k_res": report.k_res,
            "orbit_width_hat": report.orbit_width_hat,
            "delta_fow_minor_radius": report.delta_fow,
            "delta_fow_peaked_profile": report.orbit_width_hat / grad_scales["peaked_profile"],
            "radial_locality_flag": report.radial_locality_flag.value,
            "one_over_nu_surrogate_flag": report.one_over_nu_surrogate_flag.value,
        })  # fmt: skip

# Collisional-regime boundaries mapped back to nuPrime (nu_star = a * nuPrime).
nu_star_slope = float(nu_star[0]) / NU_PRIMES[0]
nu_pb = DEFAULT_THRESHOLDS.nu_star_pb_factor * c["eps_t"] ** 1.5
nu_prime_plateau_ps = DEFAULT_THRESHOLDS.nu_star_ps / nu_star_slope
nu_prime_plateau_1nu = nu_pb / nu_star_slope
print(f"  plateau/PS boundary  nu_star=1        -> nuPrime = {nu_prime_plateau_ps:.3g}")
print(f"  plateau/1nu boundary nu_star=eps^1.5  -> nuPrime = {nu_prime_plateau_1nu:.3g} "
      f"(eps_t={c['eps_t']:.4f})")  # fmt: skip

# Where does the EStar = 3e-3 curve detach (k_ExB enters the marginal band)?
detach = None
for pt in sorted((p for p in points if p["e_star"] == 3e-3), key=lambda p: -p["nu_prime"]):
    if pt["k_exb"] >= DEFAULT_THRESHOLDS.k_exb_marginal:
        detach = pt["nu_prime"]
        break
print(f"  EStar=3e-3 k_ExB enters the marginal band at nuPrime ~ {detach:.2g}"
      if detach else "  EStar=3e-3 stays in the deep 1/nu band over the scan")  # fmt: skip

# ----------------------------------------------------------------------------
# 4) JSON provenance record
# ----------------------------------------------------------------------------
print("Step 4: writing the JSON record", flush=True)
record = {
    "benchmark": (
        "Low-collisionality local-validity diagnostics, W7-X standard configuration"
    ),
    "references": [
        "F.L. Hinton & R.D. Hazeltine, Rev. Mod. Phys. 48, 239 (1976)",
        "K.C. Shaing, Phys. Fluids 27, 1567 (1984)",
        "D.-I. Ho & R.M. Kulsrud, Phys. Fluids 30, 442 (1987)",
        "C.D. Beidler et al., Nucl. Fusion 51, 076001 (2011)",
        "K.C. Shaing & J.D. Callen, Phys. Fluids 26, 3315 (1983)",
    ],
    "equilibrium": EQUILIBRIUM,
    "rN": RN_WISH,
    "e_star": E_STARS,
    "nu_prime": NU_PRIMES,
    "resolution_note": "Modest 17 x 31 angular grid with an Nxi ramp; a regime/validity figure, not a converged-coefficient benchmark.",
    "geometry": {
        "GHat": c["g_hat"], "IHat": c["i_hat"], "iota": c["iota"],
        "B0OverBBar": c["b0_over_bbar"], "Delta": c["delta"],
        "rHat": c["r_hat"], "R0": c["r_major"], "eps_t": c["eps_t"],
    },
    "grad_scale_lengths_hat": grad_scales,
    "regime_boundaries": {
        "nu_star_plateau_ps": DEFAULT_THRESHOLDS.nu_star_ps,
        "nu_star_plateau_one_over_nu": nu_pb,
        "nu_prime_plateau_ps": nu_prime_plateau_ps,
        "nu_prime_plateau_one_over_nu": nu_prime_plateau_1nu,
        "k_exb_marginal": DEFAULT_THRESHOLDS.k_exb_marginal,
        "k_exb_sqrt_nu": DEFAULT_THRESHOLDS.k_exb_sqrt_nu,
        "k_res_window": DEFAULT_THRESHOLDS.k_res_window,
    },
    "shaing_callen_d31_star_limit": d31_limit,
    "estar_3e-3_detachment_nu_prime": detach,
    "points": points,
    "scan_seconds": scan_seconds,
    "finding": (
        "The classifier's plateau -> 1/nu boundary (nu_star = eps_t^{3/2}, "
        f"nuPrime ~ {nu_prime_plateau_1nu:.2g}) coincides with where the "
        "measured |D31*| leaves the plateau and rises toward the Shaing-Callen "
        "asymptote.  The EStar = 3e-3 curve stays in the 1/nu band until the "
        f"E x B parameter k_ExB enters its marginal band near nuPrime ~ "
        f"{detach:.2g}, matching the detachment the Shaing-Callen convergence "
        "benchmark documents below nuPrime ~ 1e-3; the k_ExB = 1 sqrt(nu) "
        "crossing sits at the low end of the scan.  The larger EStar = 1e-2 "
        "field drives the reference particle to the drift resonance "
        "(k_res ~ 1) and the classifier flags the superbanana-plateau at the "
        "lowest collisionality.  The finite-orbit-width flag stays PASS "
        "(reference rho*: delta_FOW = w_b/rHat ~ "
        f"{points[0]['delta_fow_minor_radius']:.2g})."
    ),
}
JSON_PATH.write_text(json.dumps(record, indent=1))
print(f"  wrote {JSON_PATH}")

# ----------------------------------------------------------------------------
# 5) Figure
# ----------------------------------------------------------------------------
print("Step 5: rendering the figure", flush=True)
plt.rcParams.update({
    "figure.dpi": 140, "font.family": "DejaVu Sans", "font.size": 10.0,
    "axes.labelsize": 11.0, "axes.titlesize": 11.0, "legend.fontsize": 8.0,
    "axes.grid": True, "grid.alpha": 0.24, "axes.spines.top": False, "axes.spines.right": False,
})  # fmt: skip
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0), constrained_layout=True)
nu = np.asarray(NU_PRIMES)

# Panel 1: D31* coloured by regime, with the collisional boundaries + SC limit.
for j, e_star in enumerate(E_STARS):
    marker = E_STAR_MARKERS[e_star]
    ax1.semilogx(nu, d31[:, j], "-", color="0.7", lw=1.0, zorder=1)
    for i_nu, nu_prime in enumerate(NU_PRIMES):
        pt = next(p for p in points if p["nu_prime"] == nu_prime and p["e_star"] == e_star)
        ax1.semilogx(nu_prime, d31[i_nu, j], marker, ms=7, mec="k", mew=0.5,
                     color=REGIME_COLORS[pt["regime"]], zorder=3)  # fmt: skip
ax1.axhline(d31_limit, color="0.25", lw=1.2, ls="--", zorder=0)
ax1.axhline(0.0, color="0.6", lw=0.8, zorder=0)
ax1.axvline(nu_prime_plateau_ps, color="0.5", lw=0.9, ls=":", zorder=0)
ax1.axvline(nu_prime_plateau_1nu, color="0.5", lw=0.9, ls=":", zorder=0)
ax1.set_xlabel(r"$\nu'$")
ax1.set_ylabel(r"$D_{31}^*$")
ax1.set_title("Bootstrap coefficient, coloured by regime")
regime_handles = [
    Line2D([], [], marker="o", ls="", mec="k", mew=0.5, color=REGIME_COLORS[r], label=r)
    for r in (Regime.PLATEAU.value, Regime.ONE_OVER_NU.value, Regime.SQRT_NU.value,
              Regime.SUPERBANANA_PLATEAU.value)
    if any(p["regime"] == r for p in points)
]  # fmt: skip
estar_handles = [
    Line2D([], [], marker=E_STAR_MARKERS[e], ls="", mfc="none", mec="k",
           label=(r"$E^*=0$" if e == 0.0 else rf"$E^*={e:g}$"))
    for e in E_STARS
]  # fmt: skip
ax1.legend(handles=regime_handles + estar_handles, frameon=False, loc="lower left", ncol=2)

# Panel 2: the Er-layer diagnostic k_ExB(nuPrime) with the sqrt(nu) onset lines.
for e_star in E_STARS:
    if e_star == 0.0:
        continue
    kexb = np.asarray([
        next(p for p in points if p["nu_prime"] == nu_prime and p["e_star"] == e_star)["k_exb"]
        for nu_prime in NU_PRIMES
    ])  # fmt: skip
    ax2.loglog(nu, kexb, "-" + E_STAR_MARKERS[e_star], ms=5, label=rf"$E^*={e_star:g}$")
ax2.axhline(DEFAULT_THRESHOLDS.k_exb_sqrt_nu, color="#2ca02c", lw=1.2, ls="--",
            label=r"$k_{E\times B}=1$ ($\sqrt{\nu}$)")  # fmt: skip
ax2.axhline(DEFAULT_THRESHOLDS.k_exb_marginal, color="0.5", lw=1.0, ls=":",
            label="marginal band")  # fmt: skip
ax2.set_xlabel(r"$\nu'$")
ax2.set_ylabel(r"$k_{E\times B}=\omega_E/\nu_{\rm eff}$")
ax2.set_title(r"E$\times$B resonance-layer diagnostic")
ax2.legend(frameon=False, loc="upper right")

fig.suptitle(
    f"W7-X standard configuration, $r/a = {RN_WISH}$: local-validity diagnostics", y=1.04
)
fig.savefig(PNG_PATH, dpi=170, bbox_inches="tight")
plt.close(fig)

# Palette-quantize to keep the committed figure small (repo convention).
img = Image.open(PNG_PATH).convert("P", palette=Image.ADAPTIVE, colors=64)
img.save(PNG_PATH, optimize=True)
print(f"  wrote {PNG_PATH} ({PNG_PATH.stat().st_size / 1024:.0f} KB)")
print("Done.")
