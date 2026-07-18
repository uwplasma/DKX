"""ICNTS-style monoenergetic transport-coefficient benchmark on TJ-II geometry.

Second entry of the methods-paper benchmark suite: the normalized
monoenergetic transport coefficients ``D11*`` and ``D31*`` of the TJ-II
standard configuration near mid radius (s = 0.493) versus the normalized
collisionality ``nuPrime``, at several normalized radial electric fields
``EStar`` -- the comparison format established by the International
Collaboration on Neoclassical Transport in Stellarators (ICNTS) benchmark
[C.D. Beidler et al., Nucl. Fusion 51, 076001 (2011)], which includes TJ-II
as its strongest-ripple, lowest-symmetry configuration.  The monoenergetic
formulation is that of S.P. Hirshman et al., Phys. Fluids 29, 2951 (1986);
the normalization conventions are documented in
:mod:`dkx.monoenergetic`.

Geometry: the flux surface is supplied as a Boozer ``|B|`` spectrum through
``geometryScheme = 13`` (namelist ``boozer_bmnc(m,n)`` amplitudes), the one
input path shared verbatim by dkx and the Fortran v3 executable for
spectra that are not stored in a ``.bc`` file.  The 121 cosine amplitudes
below (all modes with ``|B_mn / B_00| >= 1e-4``; max m = 7, max |n| = 20
field periods) were extracted from the TJ-II mid-radius Boozer-spectrum file
``TJII-midradius_example_s_0493_fort.996`` (fort.996 format; s = 0.493
surface: iota = -1.59235, G = 1.69047 T m, I ~ 0, B_00 = 0.979469 T,
NPeriods = 4).  ``psiAHat`` is the surface-linear toroidal-flux estimate
``phip = -1.52789e-2`` stored in the same file; ``aHat = R0/aspect``.
Note the mixed stored orientation (``GHat > 0``, ``iota < 0``), which
complements the W7-X case (``GHat < 0``, ``iota < 0``) in exercising the
orientation-robust normalization.  Set ``DKX_TJII_FORT996`` to a fort.996
file path to regenerate the spectrum block (printed to stdout).

What the script does mirrors ``monoenergetic_icnts_w7x.py``: a checkpointed
(nuPrime, EStar) scan, a finer-grid convergence point, optional matched-deck
Fortran v3 cross-checks, a JSON record, and the palette-quantized figure.

Physics expectations: TJ-II's large helical ripple gives the largest
normalized ``D11*`` of the benchmark configurations (plateau minimum ~ 6
in equivalent-tokamak plateau units) and a deep 1/nu rise that the radial
electric field suppresses by an order of magnitude at ``E* = 0.1``.  The scan stops at ``nuPrime = 1e-2``: below that the
bootstrap coefficient ``D31*`` of this configuration is not resolution
converged on a laptop-sized grid (the JSON records the drift across a
resolution ladder), a known-hard regime for this machine in the benchmark
literature.

Solver notes recorded in the JSON:
  - the scan sets a 14 GB tier-1 direct-factorization budget
    (``DKX_TIER1_MEMORY_BUDGET_GB``) so every point uses the full
    banded factorization on a 24 GB machine;
  - the Fortran v3 cross-check runs pass
    ``-mat_mumps_cntl_1 0.1 -mat_mumps_icntl_10 10``: with the default
    pivot threshold the upstream executable's factorized preconditioner is
    inaccurate on this deck and its Krylov solve reports convergence of the
    preconditioned residual while the true residual stagnates at the
    10-percent level (verified against an exact factorization of the
    executable's own assembled matrix and right-hand side, which reproduces
    the dkx coefficients to ~3e-7).

Expected runtime: ~12 min total on a 10-core laptop CPU (24 scan points at
Ntheta=29, Nzeta=63, Nxi=70, ~20 s/point; ~2 min for the finer convergence
point; the Fortran cross-check adds a few minutes when enabled).  Finished
rows are cached in ``output/`` and skipped on re-runs.

Run (from the repo root):
  SFINCS_FORTRAN_EXE=/path/to/sfincs python examples/paper_benchmarks/monoenergetic_icnts_tjii.py
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

# A 24 GB laptop fits the full tier-1 factorization of this deck at ~13 GB
# peak; opt out by presetting the environment variable.
os.environ.setdefault("DKX_TIER1_MEMORY_BUDGET_GB", "14")

from dkx.monoenergetic import (  # noqa: E402
    monoenergetic_database,
    monoenergetic_dstar_from_transport_matrix,
)

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
RN_WISH = 0.70213959  # sqrt(0.493): the mid-radius benchmark surface

# nuPrime scan: 3x steps from the low-collisionality 1/nu regime through
# plateau to Pfirsch-Schlueter.  nuPrime < 1e-2 is intentionally excluded
# (D31* convergence caveat; see the module docstring and the JSON record).
NU_PRIMES = [1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0]
E_STARS = [0.0, 3e-2, 1e-1]

# Production resolution: all 121 spectrum modes are grid-representable
# (m <= 14 = Ntheta/2, |n| <= 31 = Nzeta/2); D11*/D33* converged to ~1-3%
# against the finer grids below, D31* still drifting at the lowest nuPrime
# (recorded in the JSON).
N_THETA, N_ZETA, N_XI = 29, 63, 70
SOLVER_TOLERANCE = 1e-8

# Convergence ladder at the least-converged corner (nuPrime=1e-2, EStar=0):
# the base grid of the W7-X-sized deck plus a 1.2x step above production.
CONV_POINT = (1e-2, 0.0)
CONV_RESOLUTIONS = {"coarse_23x49x70": (23, 49, 70), "fine_35x77x84": (35, 77, 84)}

# Fortran v3 cross-check points (nuPrime, EStar): 1/nu regime, ExB-suppressed
# 1/nu regime, and Pfirsch-Schlueter.  EStar = 1e-4 is the quasi-zero field
# (same convention as the W7-X case: upstream monoenergetic scans avoid
# exactly EStar = 0).
FORTRAN_POINTS = [(1e-2, 1e-4), (1e-2, 1e-1), (30.0, 1e-4)]
FORTRAN_EXE = os.environ.get("SFINCS_FORTRAN_EXE", "")
# MUMPS pivot-threshold + iterative-refinement options (module docstring).
FORTRAN_PETSC_OPTIONS = "-mat_mumps_cntl_1 0.1 -mat_mumps_icntl_10 10"

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).parent / "output"
CACHE_PATH = OUT_DIR / "monoenergetic_icnts_tjii_cache.json"
FIG_DIR = REPO_ROOT / "docs" / "_static" / "figures" / "paper_benchmarks"
JSON_PATH = FIG_DIR / "monoenergetic_icnts_tjii.json"
PNG_PATH = FIG_DIR / "monoenergetic_icnts_tjii.png"

# TJ-II mid-radius Boozer |B| cosine spectrum (see module docstring).
TJII_BOOZER_BMNC = """\
boozer_bmnc(0,0) = 9.794690e-01
boozer_bmnc(0,1) = 1.552950e-02
boozer_bmnc(0,2) = 4.160050e-04
boozer_bmnc(0,3) = -6.796890e-03
boozer_bmnc(0,4) = -3.009440e-03
boozer_bmnc(0,5) = -8.005090e-04
boozer_bmnc(0,6) = -3.262050e-03
boozer_bmnc(0,7) = 1.187050e-02
boozer_bmnc(0,8) = -2.710270e-02
boozer_bmnc(0,9) = 2.657340e-03
boozer_bmnc(0,10) = 1.361340e-04
boozer_bmnc(0,13) = -2.049340e-04
boozer_bmnc(0,15) = 3.964450e-04
boozer_bmnc(0,16) = -3.927800e-04
boozer_bmnc(1,-18) = 2.032000e-04
boozer_bmnc(1,-17) = -3.780690e-04
boozer_bmnc(1,-16) = -1.890610e-04
boozer_bmnc(1,-15) = 1.309300e-04
boozer_bmnc(1,-13) = 1.138450e-04
boozer_bmnc(1,-12) = 3.639640e-04
boozer_bmnc(1,-11) = -7.713280e-04
boozer_bmnc(1,-10) = 4.980210e-03
boozer_bmnc(1,-9) = -8.076400e-03
boozer_bmnc(1,-8) = -4.888500e-03
boozer_bmnc(1,-7) = 8.104670e-04
boozer_bmnc(1,-6) = -1.847640e-03
boozer_bmnc(1,-5) = -1.596240e-03
boozer_bmnc(1,-4) = 1.016350e-03
boozer_bmnc(1,-3) = -9.695690e-03
boozer_bmnc(1,-2) = 5.632790e-02
boozer_bmnc(1,-1) = -9.227370e-02
boozer_bmnc(1,0) = -8.405740e-02
boozer_bmnc(1,1) = 7.337940e-03
boozer_bmnc(1,2) = -3.345120e-03
boozer_bmnc(1,4) = 1.343720e-03
boozer_bmnc(1,5) = -1.053100e-03
boozer_bmnc(1,6) = 2.096090e-03
boozer_bmnc(1,7) = 1.008890e-02
boozer_bmnc(1,8) = -5.334000e-03
boozer_bmnc(1,9) = 4.811210e-04
boozer_bmnc(1,10) = -3.836620e-04
boozer_bmnc(1,12) = 1.409210e-04
boozer_bmnc(1,15) = 3.412130e-04
boozer_bmnc(2,-19) = -1.040360e-04
boozer_bmnc(2,-18) = 4.103460e-04
boozer_bmnc(2,-17) = -4.282460e-04
boozer_bmnc(2,-16) = -1.589010e-04
boozer_bmnc(2,-15) = 2.806740e-04
boozer_bmnc(2,-14) = 2.517600e-04
boozer_bmnc(2,-13) = 1.733720e-04
boozer_bmnc(2,-12) = 1.656020e-04
boozer_bmnc(2,-11) = -2.037190e-03
boozer_bmnc(2,-10) = 1.131550e-02
boozer_bmnc(2,-9) = -3.340850e-03
boozer_bmnc(2,-7) = -4.866370e-04
boozer_bmnc(2,-6) = -1.189530e-04
boozer_bmnc(2,-5) = -1.503160e-03
boozer_bmnc(2,-4) = 3.345480e-03
boozer_bmnc(2,-3) = -1.698580e-02
boozer_bmnc(2,-2) = 6.126030e-03
boozer_bmnc(2,-1) = -2.461620e-03
boozer_bmnc(2,0) = 1.216770e-03
boozer_bmnc(2,1) = -1.838430e-03
boozer_bmnc(2,2) = 3.631440e-04
boozer_bmnc(2,3) = -3.624010e-04
boozer_bmnc(2,4) = 2.335950e-03
boozer_bmnc(2,5) = -6.387840e-03
boozer_bmnc(2,6) = 1.133270e-02
boozer_bmnc(2,7) = -8.217630e-04
boozer_bmnc(2,8) = 3.849230e-04
boozer_bmnc(2,9) = -2.038320e-04
boozer_bmnc(3,-20) = -2.007260e-04
boozer_bmnc(3,-19) = 3.355360e-04
boozer_bmnc(3,-17) = -1.737410e-04
boozer_bmnc(3,-15) = 1.962530e-04
boozer_bmnc(3,-13) = 1.213010e-04
boozer_bmnc(3,-12) = -6.297800e-04
boozer_bmnc(3,-11) = 2.679530e-04
boozer_bmnc(3,-10) = 2.127360e-03
boozer_bmnc(3,-8) = -2.019070e-04
boozer_bmnc(3,-7) = -1.137600e-04
boozer_bmnc(3,-6) = 5.859690e-04
boozer_bmnc(3,-5) = -1.844160e-03
boozer_bmnc(3,-4) = 6.675060e-03
boozer_bmnc(3,-3) = -4.618040e-03
boozer_bmnc(3,-2) = -1.367820e-03
boozer_bmnc(3,-1) = -1.215840e-03
boozer_bmnc(3,0) = -2.716020e-04
boozer_bmnc(3,2) = 1.475580e-04
boozer_bmnc(3,3) = -4.823930e-04
boozer_bmnc(3,4) = 1.289800e-03
boozer_bmnc(3,5) = -3.362520e-03
boozer_bmnc(3,6) = 8.306680e-04
boozer_bmnc(3,7) = 2.929350e-04
boozer_bmnc(3,12) = -1.858580e-04
boozer_bmnc(4,-19) = 1.098770e-04
boozer_bmnc(4,-13) = 3.374610e-04
boozer_bmnc(4,-12) = -1.128420e-03
boozer_bmnc(4,-11) = 1.517490e-04
boozer_bmnc(4,-10) = 1.010650e-04
boozer_bmnc(4,-7) = -2.779610e-04
boozer_bmnc(4,-6) = 9.409780e-04
boozer_bmnc(4,-5) = -2.765140e-03
boozer_bmnc(4,-4) = 2.237860e-03
boozer_bmnc(4,-3) = 5.807540e-04
boozer_bmnc(4,-2) = 1.196360e-04
boozer_bmnc(4,4) = -1.645030e-04
boozer_bmnc(4,6) = -2.328760e-04
boozer_bmnc(5,-13) = 1.979210e-04
boozer_bmnc(5,-12) = -1.190000e-04
boozer_bmnc(5,-8) = 1.484780e-04
boozer_bmnc(5,-7) = -4.060240e-04
boozer_bmnc(5,-6) = 1.039560e-03
boozer_bmnc(5,-5) = -7.615250e-04
boozer_bmnc(5,-4) = -2.811750e-04
boozer_bmnc(5,3) = 1.144510e-04
boozer_bmnc(6,-8) = 1.743900e-04
boozer_bmnc(6,-7) = -3.793840e-04
boozer_bmnc(6,-6) = 2.171330e-04
boozer_bmnc(6,-5) = 1.448010e-04
boozer_bmnc(7,-8) = 1.474720e-04
"""

DECK_TEMPLATE = """! TJ-II standard configuration (s = 0.493), ICNTS-style monoenergetic deck.
! Generated by examples/paper_benchmarks/monoenergetic_icnts_tjii.py
&general
  RHSMode = 3
/
&geometryParameters
  geometryScheme = 13
  Nperiods = 4
  psiAHat = -0.0152789
  aHat = 0.19329318
  iota = -1.59235
  GHat = 1.69047
  IHat = 0.0
  inputRadialCoordinate = 3
  rN_wish = {rn_wish}
{spectrum}
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


def regenerate_spectrum(fort996_path: Path, threshold: float = 1e-4) -> str:
    """Re-extract the ``boozer_bmnc`` block from a fort.996 Boozer-spectrum file.

    Format (single surface): line 1 ``nfp ns``; line 2 ``aspect rmax rmin
    betaxis``; line 3 ``mboz nboz mnboz``; line 4 ``iota pres beta phip phi
    bvco buco``; then ``mnboz`` rows ``m n bmn rmnc zmns pmns gmn`` with the
    toroidal mode number ``n`` in absolute (not per-period) units.
    """
    tok = fort996_path.read_text().split()
    nfp = int(tok[0])
    arr = np.array(tok[16:], dtype=float).reshape(-1, 7)
    m, n, b = arr[:, 0].astype(int), arr[:, 1].astype(int), arr[:, 2]
    b00 = b[(m == 0) & (n == 0)][0]
    keep = np.abs(b) / b00 >= threshold
    order = np.lexsort((n[keep] // nfp, m[keep]))
    return "\n".join(
        f"  boozer_bmnc({mm},{nn}) = {bb:.6e}"
        for mm, nn, bb in zip(m[keep][order], n[keep][order] // nfp, b[keep][order])
    )


def write_deck(path, *, nu_prime=1.0, e_star=0.0, resolution=(N_THETA, N_ZETA, N_XI)):
    nt, nz, nxi = resolution
    spectrum = "\n".join("  " + line.strip() for line in TJII_BOOZER_BMNC.strip().splitlines())
    path.write_text(
        DECK_TEMPLATE.format(
            rn_wish=RN_WISH, spectrum=spectrum, nu_prime=nu_prime, e_star=e_star,
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


if os.environ.get("DKX_TJII_FORT996", ""):
    print("Regenerated boozer_bmnc block from DKX_TJII_FORT996:")
    print(regenerate_spectrum(Path(os.environ["DKX_TJII_FORT996"])))

print("=== examples/paper_benchmarks/monoenergetic_icnts_tjii.py ===", flush=True)
print(f"  TJ-II standard configuration (geometryScheme=13), rN = {RN_WISH}")
print(f"  scan: {len(NU_PRIMES)} nuPrime x {len(E_STARS)} EStar")
print(f"  resolution: Ntheta={N_THETA} Nzeta={N_ZETA} Nxi={N_XI} tol={SOLVER_TOLERANCE:g}")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
cache = load_cache()

# ----------------------------------------------------------------------------
# 1) The (nuPrime, EStar) scan, one nuPrime row at a time (checkpointed)
# ----------------------------------------------------------------------------
deck_path = OUT_DIR / "monoenergetic_icnts_tjii.input.namelist"
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
# 2) Convergence ladder at (nuPrime, EStar) = (1e-2, 0)
# ----------------------------------------------------------------------------
print(f"Step 2: convergence ladder at (nuPrime, EStar) = {CONV_POINT}", flush=True)
for tag, resolution in CONV_RESOLUTIONS.items():
    if tag in cache["conv"]:
        continue
    conv_deck = OUT_DIR / f"monoenergetic_icnts_tjii_conv_{tag}.input.namelist"
    write_deck(conv_deck, resolution=resolution)
    t0 = time.time()
    db_conv = monoenergetic_database(conv_deck, [CONV_POINT[0]], [CONV_POINT[1]])
    cache["conv"][tag] = {
        "resolution": list(resolution),
        "seconds": time.time() - t0,
        **{k: float(np.asarray(getattr(db_conv, k))[0, 0]) for k in COEFF_KEYS},
    }
    save_cache(cache)
i_nu = NU_PRIMES.index(CONV_POINT[0])
i_es = E_STARS.index(CONV_POINT[1])
tables = {"d11_star": d11, "d31_star": d31, "d33_star": d33}
conv = {"ladder": {}}
for tag in CONV_RESOLUTIONS:
    rec = cache["conv"][tag]
    entry = {"resolution": rec["resolution"], "seconds": rec["seconds"]}
    for key in ("d11_star", "d31_star", "d33_star"):
        base = float(tables[key][i_nu, i_es])
        entry[key] = {"production": base, "value": rec[key],
                      "rel_dev": abs(base - rec[key]) / abs(rec[key])}  # fmt: skip
    conv["ladder"][tag] = entry
    print(
        f"  {tag}: " + "  ".join(
            f"{k} {entry[k]['value']:+.4e} (prod dev {entry[k]['rel_dev']:.1e})"
            for k in ("d11_star", "d31_star", "d33_star")
        )
    )  # fmt: skip

# ----------------------------------------------------------------------------
# 3) Fortran v3 cross-check (optional; requires SFINCS_FORTRAN_EXE)
# ----------------------------------------------------------------------------
fortran_records = []
if FORTRAN_EXE and Path(FORTRAN_EXE).exists():
    import h5py

    from dkx.validation.fortran import run_sfincs_fortran

    print(f"Step 3: Fortran v3 cross-check at {len(FORTRAN_POINTS)} points", flush=True)
    fortran_env = {
        "PETSC_OPTIONS": (os.environ.get("PETSC_OPTIONS", "") + " " + FORTRAN_PETSC_OPTIONS).strip()
    }
    c = cache["constants"]
    for nu_prime, e_star in FORTRAN_POINTS:
        tag = f"nu{nu_prime:g}_es{e_star:g}".replace(".", "p").replace("-", "m")
        fkey = f"{tag}@{N_THETA}x{N_ZETA}x{N_XI}"
        point_deck = OUT_DIR / f"fortran_tjii_{tag}.input.namelist"
        write_deck(point_deck, nu_prime=nu_prime, e_star=e_star)
        if fkey not in cache["fortran"]:
            t0 = time.time()
            h5_path = run_sfincs_fortran(
                input_namelist=point_deck,
                exe=Path(FORTRAN_EXE),
                workdir=OUT_DIR / f"fortran_tjii_{tag}",
                env=fortran_env,
            )
            fortran_seconds = time.time() - t0
            with h5py.File(h5_path, "r") as f:
                # Column-major on disk: transpose to mathematical order.
                tm = np.asarray(f["transportMatrix"][...], dtype=np.float64).T
            point = monoenergetic_dstar_from_transport_matrix(
                tm, nu_prime=nu_prime, **{k: c[k] for k in DB_CONSTANT_KEYS}
            )
            cache["fortran"][fkey] = {
                "seconds": fortran_seconds,
                **{k: float(np.asarray(getattr(point, k))) for k in COEFF_KEYS},
            }
            save_cache(cache)
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
            rec[key] = {"dkx": ours, "fortran": ref, "rel_dev": abs(ours - ref) / abs(ref)}
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
    "benchmark": "ICNTS-style monoenergetic coefficients, TJ-II standard configuration",
    "references": [
        "C.D. Beidler et al., Nucl. Fusion 51, 076001 (2011)",
        "S.P. Hirshman et al., Phys. Fluids 29, 2951 (1986)",
    ],
    "equilibrium": (
        "TJ-II mid-radius Boozer |B| spectrum (s = 0.493, fort.996 format), "
        "121 cosine modes with |B_mn/B_00| >= 1e-4, geometryScheme = 13"
    ),
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
        **conv,
    },
    "convergence_note": (
        "D11* and D33* are grid converged to ~1-3% at the production "
        "resolution; the small bootstrap coefficient D31* still drifts "
        "across the resolution ladder at nuPrime = 1e-2 (a known-hard "
        "low-collisionality regime for this strong-ripple configuration), "
        "which is why the scan does not extend below nuPrime = 1e-2."
    ),
    "fortran_crosscheck": fortran_records,
    "fortran_crosscheck_note": (
        "Matched decks at the scan resolution; e_star = 1e-4 serves as the "
        "quasi-zero field (same convention as the W7-X case).  The runs "
        f"pass PETSc options '{FORTRAN_PETSC_OPTIONS}': with the default "
        "pivot threshold the upstream factorized preconditioner is "
        "inaccurate on this deck and the Krylov solve reports preconditioned"
        "-residual convergence while the true residual stagnates at the "
        "10-percent level."
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
ax1.legend(frameon=False, loc="lower right")

ax3.axhline(0.0, color="0.6", lw=0.8, zorder=0)
ax3.set_xlabel(r"$\nu'$")
ax3.set_ylabel(r"$D_{31}^*$")
ax3.set_title("Bootstrap/Ware coefficient (banana units)")

fig.suptitle(
    "TJ-II standard configuration, $s = 0.493$: monoenergetic benchmark", y=1.04
)
fig.savefig(PNG_PATH, dpi=170, bbox_inches="tight")
plt.close(fig)

# Palette-quantize to keep the committed figure small (repo convention).
img = Image.open(PNG_PATH).convert("P", palette=Image.ADAPTIVE, colors=64)
img.save(PNG_PATH, optimize=True)
print(f"  wrote {PNG_PATH} ({PNG_PATH.stat().st_size / 1024:.0f} KB)")
print("Done.")
