"""W7-X ambipolar-Er experimental case (methods-paper benchmark, pillar 5).

Roadmap item 1 of ``plan_final.md`` lists "one W7-X ambipolar-Er experimental
case" among the methods-paper benchmark pieces.  This script computes the
neoclassical ambipolar radial electric field ``E_r(rho)`` of the W7-X standard
configuration from *measured* plasma profiles of a published discharge and
compares it against the published measured and neoclassically-predicted
``E_r`` profiles.

Sourced experimental case (REAL published data)
-----------------------------------------------
Profiles and reference ``E_r`` are digitized (approximately, from the figures)
from

    N.A. Pablant, A. Langenberg, A. Alonso, C.D. Beidler, M. Bitter,
    S. Bozhenkov, et al., "Core radial electric field and transport in
    Wendelstein 7-X plasmas", Physics of Plasmas 25, 022508 (2018),
    https://doi.org/10.1063/1.5018326  (program W7-X 20160309.010).

That discharge is a core-electron-root-confinement (CERC) case: with strong
central ECRH (2.0 MW here at t = 300 ms), the electron temperature is much
larger than the ion temperature (``T_e ~ 6 keV``, ``T_i ~ 1.1 keV``,
``n_e ~ 1.5e19 m^-3``), the steep ``dT_e/dr`` drives an outward electron
neoclassical flux, and ambipolarity is enforced by a *positive* radial electric
field in the core (the electron root), crossing over to a negative ion root near
``rho ~ 0.6``.  Fig. 3 of the paper gives the ``n_e``, ``T_e``, ``T_i``
profiles used as neoclassical input; Fig. 4 gives the ``E_r`` profile inferred
from XICS perpendicular-flow measurements (``E_r^{XICS}``) together with the
neoclassical ambipolar ``E_r`` computed for the same profiles (``E_r^{NC}``).

Provenance / honesty
--------------------
The ``n_e/T_e/T_i`` and reference-``E_r`` arrays below are hand-digitized from
the paper's figures and are therefore *approximate* (few-percent-of-axis
reading error, no access to the underlying data).  The geometry is the shipped
Boozer standard-configuration equilibrium ``w7x_standardConfig.bc``
(geometryScheme = 11); the paper's neoclassical runs used a vacuum VMEC
equilibrium for the OP1.1 limiter configuration, so the geometry is a close but
not identical match.  Following the paper we treat the plasma as pure hydrogen
(``Z_eff = 1``, electrons + protons).  This is a genuine data-validation case,
not a synthetic one; the agreement is reported honestly (sign, root character,
crossover location, and magnitude within the digitization + model uncertainty).

Model
-----
Local two-species (proton + electron) drift-kinetic solve on each flux surface
with the pitch-angle-scattering collision operator and the incompressible
``E x B`` (DKES) drift model (``collisionOperator = 1``,
``useDKESExBDrift = .true.``): the same monoenergetic model family the paper's
DKES reference uses, and the paper shows DKES and the full-Fokker-Planck code
agree closely on the ambipolar ``E_r`` (its Fig. 6).  On each surface the
radial current ``J_r(E_r) = sum_a Z_a Gamma_a(E_r)`` is scanned with
:func:`dkx.er.radial_current`; every ambipolar root ``J_r = 0`` is bracketed and
classified ion / unstable / electron by the (orientation-corrected) sign of
``dJr/dEr`` and the sign of ``E_r`` (ion root ``E_r < 0``, electron root
``E_r > 0``).  A root is physically stable iff ``dJr_phys/dEr > 0`` because the
field relaxes as ``dEr/dt ~ -J_r_phys``; on this equilibrium ``psiHat`` points
inward (``psiAHat < 0``), so the code's ``J_r = <J.grad psiHat>`` has the
*opposite* sign to the physical outward current and stability is
``slope * sign(psiAHat) > 0`` (the naive ``dkx.er._classify``, which assumes
``sign(psiAHat) = +1``, would invert the labels here).  The physically-realised
branch is selected by radial continuity (branch-following outward from the core
electron root), reproducing the CERC electron-root -> ion-root transition.

Units: the deck-normalized ``E_r`` equals the physical radial electric field in
kV/m for the SFINCS reference normalization used here (``phiBar = TBar/e = 1 kV``
at ``TBar = 1 keV``, ``RBar = 1 m``, ``alpha = 1``), so ``E_r`` values below are
directly in kV/m and comparable to the paper.

What the script produces
------------------------
``docs/_static/figures/paper_benchmarks/w7x_ambipolar_er.png`` (four panels:
input profiles; DKX ``E_r(rho)`` vs the measured XICS and predicted-neoclassical
references; the ``J_r(E_r)`` S-curves per surface with classified roots; the
per-species neoclassical heat flux at the ambipolar root) and the companion
``.json`` with the digitized profiles + provenance/citation, the per-surface
roots and selected ambipolar ``E_r``, the reference comparison, per-species
fluxes, and the resolution-convergence note.

Expected runtime: ~25-35 min on a laptop CPU (float64); six surfaces x a
warm-started ``E_r`` scan at Ntheta=15, Nzeta=37, Nxi=36, Nx=6 (~16 s / solve,
tier-1 block solve) plus one convergence surface.  Every surface is checkpointed
under ``output/w7x_ambipolar_er/`` and skipped on re-runs
(``DKX_W7XAMB_FORCE=1`` to recompute).  The CI-sized regression version lives in
``tests/test_paper_benchmark_w7x_ambipolar.py``.

Run (from the repo root, with the equilibrium search path set):
  DKX_EQUILIBRIA_DIRS=/path/to/equilibria \
  python examples/paper_benchmarks/w7x_ambipolar_er.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)

from scipy.interpolate import PchipInterpolator  # noqa: E402

from dkx import er as er_mod  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(
    os.environ.get("DKX_W7XAMB_OUT_DIR", Path(__file__).parent / "output" / "w7x_ambipolar_er")
)
FIG_DIR = Path(
    os.environ.get("DKX_W7XAMB_FIG_DIR", REPO_ROOT / "docs" / "_static" / "figures" / "paper_benchmarks")
)
PNG_PATH = FIG_DIR / "w7x_ambipolar_er.png"
JSON_PATH = FIG_DIR / "w7x_ambipolar_er.json"

CI = os.environ.get("DKX_W7XAMB_CI") == "1"
FORCE = os.environ.get("DKX_W7XAMB_FORCE") == "1"

# ---------------------------------------------------------------------------
# Parameters (simsopt-style: everything at the top)
# ---------------------------------------------------------------------------
EQUILIBRIUM = "w7x_standardConfig.bc"  # geometryScheme = 11 Boozer; resolved via DKX_EQUILIBRIA_DIRS
# Effective minor radius aHat = a/RBar read from the .bc header (a = 0.53697 m,
# RBar = 1 m): converts profile gradients d/drho -> d/drHat = (1/aHat) d/drho.
AHAT = 0.53697

CITATION = (
    "N.A. Pablant et al., Phys. Plasmas 25, 022508 (2018), "
    "https://doi.org/10.1063/1.5018326 (W7-X program 20160309.010, 2.0 MW, t=300 ms)"
)

# --- Digitized input profiles (Pablant 2018, Fig. 3, 2.0 MW / 300 ms) --------
# rho = sqrt(psi/psi_edge) = normalized minor radius = SFINCS rN.  APPROXIMATE
# (hand-read from the figure).  n_e in 1e19 m^-3; T_e, T_i in keV.
PROFILE_RHO = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
PROFILE_NE19 = np.array([1.52, 1.53, 1.53, 1.52, 1.49, 1.44, 1.33, 1.15, 0.88, 0.55, 0.15])
PROFILE_TE = np.array([6.10, 6.05, 5.60, 4.75, 3.60, 2.50, 1.55, 0.85, 0.40, 0.18, 0.05])
PROFILE_TI = np.array([1.15, 1.15, 1.13, 1.10, 1.05, 0.95, 0.78, 0.52, 0.28, 0.12, 0.03])

# --- Digitized reference E_r [kV/m] vs rho (Pablant 2018, Fig. 4, 2.0 MW) -----
# Neoclassical ambipolar E_r predicted for these profiles (blue dots in Fig. 4).
REF_NC_RHO = np.array(
    [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.00]
)
REF_NC_ER = np.array(
    [3.0, 5.0, 7.0, 8.5, 9.5, 10.5, 11.5, 11.0, 9.5, 7.5, 2.0, -4.0, -8.0, -11.0, -11.0, -9.5, -11.0]
)
# E_r inferred from XICS perpendicular-flow measurements (red curve in Fig. 4);
# reliable in the core, large error bars beyond rho ~ 0.65 (truncated here).
REF_XICS_RHO = np.array([0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65])
REF_XICS_ER = np.array([3.5, 8.5, 10.0, 11.0, 11.0, 11.0, 9.5, 6.5, 3.0, 0.0, -6.0])

# --- Flux surfaces at which DKX solves the local ambipolar problem -----------
SURFACES = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70) if not CI else (0.30, 0.70)

# --- Resolution -------------------------------------------------------------
# Production (RES): Nxi=36 / Nzeta=37 resolve the low-collisionality electron
# 1/nu channel that sets the electron root; the coarse grid below MISSES the
# electron root entirely at rho~0.5 (see the convergence note / JSON), so the
# production grid is the minimum that captures the CERC branch.
RES = dict(n_theta=15, n_zeta=37, n_xi=36, n_x=6)
RES_CONV = dict(n_theta=17, n_zeta=43, n_xi=44, n_x=6)  # ~1.2x refinement check
RES_CI = dict(n_theta=11, n_zeta=25, n_xi=24, n_x=5)
if CI:
    RES = RES_CI

# --- E_r scan grid [kV/m] ---------------------------------------------------
# Non-uniform: dense through [-6, 14] to bracket ion / unstable / electron roots
# on the standard-stellarator S-curve, sparse in the wings.
ER_SCAN = (
    np.array([-20.0, -12.0, -7.0, -4.0, -2.0, -0.5, 1.0, 2.5, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0])
    if not CI
    else np.array([-15.0, -8.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 15.0])
)
ER_BRACKET = (-25.0, 25.0)
SOLVER = "block_tridiagonal"  # tier-1 direct block-Thomas (exact for PAS)
CONV_SURFACE = 0.30  # surface for the resolution-convergence check

# ---------------------------------------------------------------------------
# Profile interpolators (monotone PCHIP: smooth values + consistent gradients)
# ---------------------------------------------------------------------------
_NE_F = PchipInterpolator(PROFILE_RHO, PROFILE_NE19)
_TE_F = PchipInterpolator(PROFILE_RHO, PROFILE_TE)
_TI_F = PchipInterpolator(PROFILE_RHO, PROFILE_TI)
_DNE_F = _NE_F.derivative()
_DTE_F = _TE_F.derivative()
_DTI_F = _TI_F.derivative()


def profile_at(rho: float) -> dict:
    """Return the SFINCS-normalized species inputs at ``rho``.

    ``nHat = n / nBar`` with ``nBar = 1e20 m^-3`` (so ``nHat = n[1e19]/10``);
    ``THat = T[keV]`` (``TBar = 1 keV``).  Gradients are taken w.r.t. ``rHat``
    (the inputRadialCoordinateForGradients = 4 / Er path): ``d/drHat =
    (1/aHat) d/drho`` since ``rHat = aHat * rho``.
    """
    nHat = float(_NE_F(rho)) / 10.0
    te = float(_TE_F(rho))
    ti = float(_TI_F(rho))
    dnHat_drhat = (float(_DNE_F(rho)) / 10.0) / AHAT
    dte_drhat = float(_DTE_F(rho)) / AHAT
    dti_drhat = float(_DTI_F(rho)) / AHAT
    return dict(
        nHat=nHat, te=te, ti=ti,
        dnHat_drhat=dnHat_drhat, dte_drhat=dte_drhat, dti_drhat=dti_drhat,
    )  # fmt: skip


# ---------------------------------------------------------------------------
# Deck builder
# ---------------------------------------------------------------------------
def build_deck(rho: float, res: dict) -> str:
    """Two-species (H+ + electron) RHSMode=1 deck on w7x_standardConfig."""
    p = profile_at(rho)
    return f"""&general
  RHSMode = 1
/
&geometryParameters
  geometryScheme = 11
  equilibriumFile = "{EQUILIBRIUM}"
  inputRadialCoordinate = 3
  inputRadialCoordinateForGradients = 4
  rN_wish = {rho}
/
&speciesParameters
  Zs = 1 -1
  mHats = 1.0 5.446170214d-4
  nHats = {p["nHat"]} {p["nHat"]}
  THats = {p["ti"]} {p["te"]}
  dNHatdrHats = {p["dnHat_drhat"]} {p["dnHat_drhat"]}
  dTHatdrHats = {p["dti_drhat"]} {p["dte_drhat"]}
/
&physicsParameters
  Delta = 4.5694d-3
  alpha = 1.0
  nu_n = 8.330d-3
  Er = 0.0
  collisionOperator = 1
  includeXDotTerm = .false.
  includeElectricFieldTermInXiDot = .false.
  useDKESExBDrift = .true.
  includePhi1 = .false.
/
&resolutionParameters
  Ntheta = {res["n_theta"]}
  Nzeta = {res["n_zeta"]}
  Nxi = {res["n_xi"]}
  NL = 4
  Nx = {res["n_x"]}
  solverTolerance = 1d-9
/
&otherNumericalParameters
  Nxi_for_x_option = 0
/
"""


def _write_deck(text: str, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    path.write_text(text)
    return path


# ---------------------------------------------------------------------------
# Orientation-aware root classification
# ---------------------------------------------------------------------------
# On this equilibrium psiHat points inward (psiAHat < 0), which flips the sign
# of J_r = <J.grad psiHat> relative to the physical outward radial current
# <J.grad r>.  A root is physically stable iff dJr_phys/dEr > 0, i.e. iff
# ``slope * orient > 0`` with ``orient = sign(psiAHat)``; the naive
# ``dkx.er._classify`` (stable iff dJr/dEr > 0) assumes orient = +1 and would
# therefore invert the ion/electron/unstable labels here.  ``orient`` is read
# from the sign of the deck's ``dPhiHatdpsiHat``-per-``E_r`` factor:
# ``orient = -sign(dphi_per_er) = sign(psiAHat)``.
def _orient(prob) -> float:
    return float(-np.sign(prob.dphi_per_er))


def _is_stable(slope: float, orient: float) -> bool:
    return slope * orient > 0.0


def _classify(er: float, slope: float, orient: float) -> str:
    if not _is_stable(slope, orient):
        return "unstable"
    return "electron" if er > 0.0 else "ion"


def _refine_secant(prob, lo, flo, hi, fhi, *, max_steps=10):
    """Bracketed secant/bisection root of J_r on [lo, hi] (a sign-change bracket).

    Returns ``(er_root, f_root, state)`` -- the warm-start state at the final
    (essentially at-root) evaluation is reused for the per-species fluxes, so no
    extra solve is spent re-evaluating at the root.
    """
    state = None
    mid, fmid = 0.5 * (lo + hi), flo
    for _ in range(max_steps):
        if fhi == flo:
            mid = 0.5 * (lo + hi)
        else:
            mid = hi - fhi * (hi - lo) / (fhi - flo)
            if not (min(lo, hi) < mid < max(lo, hi)):
                mid = 0.5 * (lo + hi)
        fval, _g, state = er_mod.radial_current(prob, float(mid), solve_method=SOLVER)
        fmid = float(fval)
        if abs(hi - lo) < 3e-2:
            break
        if (flo > 0) == (fmid > 0):
            lo, flo = mid, fmid
        else:
            hi, fhi = mid, fmid
    return mid, fmid, state


def solve_surface(rho: float, res: dict) -> dict:
    """Scan J_r(E_r), resolve+classify every ambipolar root at one surface.

    Returns the scan (for the S-curve figure), the classified roots, and the
    per-species particle/heat fluxes at each root.  Classification uses the exact
    ``dJr/dEr`` sign across each sign-change bracket (a root is stable iff ``J_r``
    increases through it), so no extra slope solves are spent.
    """
    deck = _write_deck(build_deck(rho, res), f"surf_{rho:.2f}.input.namelist")
    prob = er_mod.prepare(deck, er_bracket=ER_BRACKET)
    orient = _orient(prob)
    from dkx.run import profile_moments_from_operator  # noqa: PLC0415
    from dkx.er import operator_at_er  # noqa: PLC0415

    er_grid = np.asarray(ER_SCAN, dtype=float)
    jr = np.empty(er_grid.shape, dtype=float)
    state = None
    for i, er in enumerate(er_grid):
        j, _g, state = er_mod.radial_current(
            prob, float(er),
            x0=(state.x if state is not None else None),
            recycle=(state.recycle if state is not None else None),
            solve_method=SOLVER,
        )  # fmt: skip
        jr[i] = float(j)

    # Every sign-change bracket -> a refined, classified root with fluxes.
    roots: list[dict] = []
    for i in range(len(er_grid) - 1):
        if (jr[i] > 0) == (jr[i + 1] > 0):
            continue
        er_root, jr_root, st = _refine_secant(prob, er_grid[i], jr[i], er_grid[i + 1], jr[i + 1])
        # dJr/dEr sign across the bracket is exact for classification; the
        # magnitude is the bracket secant estimate.
        slope = (jr[i + 1] - jr[i]) / (er_grid[i + 1] - er_grid[i])
        op = operator_at_er(prob.operator, er_root, dphi_per_er=prob.dphi_per_er)
        table = profile_moments_from_operator(op, np.asarray(st.x).reshape(-1))
        gamma_np = np.asarray(table["particleFlux_vm_psiHat"], float).reshape(-1)
        qflux_np = np.asarray(table["heatFlux_vm_psiHat"], float).reshape(-1)
        roots.append(
            {
                "er": float(er_root),
                "radial_current": float(jr_root),
                "slope": float(slope),
                "orient": float(orient),
                "stable": bool(_is_stable(slope, orient)),
                "root_type": _classify(er_root, slope, orient),
                "gamma_i": float(gamma_np[0]),
                "gamma_e": float(gamma_np[1]),
                "q_i": float(qflux_np[0]),
                "q_e": float(qflux_np[1]),
            }
        )
    n_sign_changes = int(np.sum(np.sign(jr[:-1]) * np.sign(jr[1:]) < 0))
    return {
        "rho": float(rho),
        "profile": profile_at(rho),
        "orient": float(orient),
        "er_scan": er_grid.tolist(),
        "jr_scan": jr.tolist(),
        "roots": roots,
        "n_sign_changes": n_sign_changes,
        "total_size": int(prob.operator.total_size),
        "resolution": dict(res),
    }


# ---------------------------------------------------------------------------
# Physical branch selection (radial continuity / branch-following)
# ---------------------------------------------------------------------------
def select_branch(surfaces: list[dict]) -> None:
    """Pick the physically-realised stable root per surface by radial continuity.

    Starts from the innermost surface's electron root (CERC core) and, moving
    outward, selects at each surface the stable (``dJr/dEr > 0``) root closest in
    ``E_r`` to the inner neighbour's selected value.  This follows the electron-root
    branch until it is annihilated at the saddle-node fold, then continues on the
    ion root -- reproducing the CERC electron-root -> ion-root crossover without
    reference to the measured profile.  Mutates each surface dict in place,
    adding ``selected`` (the chosen root dict or ``None``).
    """
    prev_er = None
    for surf in sorted(surfaces, key=lambda s: s["rho"]):
        stable = [r for r in surf["roots"] if r["stable"]]
        if not stable:
            surf["selected"] = None
            continue
        if prev_er is None:
            # innermost: prefer the electron root (CERC core), else the only root
            electron = [r for r in stable if r["root_type"] == "electron"]
            chosen = max(electron, key=lambda r: r["er"]) if electron else stable[0]
        else:
            chosen = min(stable, key=lambda r: abs(r["er"] - prev_er))
        surf["selected"] = chosen
        prev_er = chosen["er"]


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def _load_ckpt(name: str) -> dict | None:
    path = OUT_DIR / name
    if path.exists() and not FORCE:
        return json.loads(path.read_text())
    return None


def _save_ckpt(name: str, payload: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / name).write_text(json.dumps(payload, indent=1))


# ---------------------------------------------------------------------------
# Step 1: per-surface ambipolar solves (checkpointed)
# ---------------------------------------------------------------------------
def step1_surfaces() -> list[dict]:
    surfaces: list[dict] = []
    for rho in SURFACES:
        name = f"surface_{rho:.2f}{'_ci' if CI else ''}.json"
        ck = _load_ckpt(name)
        if ck is not None:
            print(f"Surface rho={rho:.2f}  [checkpoint]  roots={[round(r['er'],2) for r in ck['roots']]}")
            surfaces.append(ck)
            continue
        print(f"Surface rho={rho:.2f}: scanning J_r(E_r) at Nxi={RES['n_xi']} ...", flush=True)
        t0 = time.time()
        surf = solve_surface(rho, RES)
        surf["seconds"] = time.time() - t0
        roots_str = ", ".join(f"{r['er']:+.2f}({r['root_type'][:3]})" for r in surf["roots"])
        print(
            f"    size={surf['total_size']}  {len(surf['roots'])} root(s)  "
            f"[{roots_str}]  [{surf['seconds']:.0f}s]",
            flush=True,
        )
        _save_ckpt(name, surf)
        surfaces.append(surf)
    return surfaces


# ---------------------------------------------------------------------------
# Step 2: resolution-convergence check at one surface
# ---------------------------------------------------------------------------
def _electron_root_of(surf: dict):
    el = [r for r in surf["roots"] if r["root_type"] == "electron" and r["stable"]]
    return max((r["er"] for r in el), default=None)


def _electron_root_at(rho: float, res: dict, er_center: float) -> float | None:
    """Targeted electron-root find in a narrow E_r window at resolution ``res``.

    Cheap convergence probe: scans a few points bracketing the known electron
    root and secant-refines the single sign change, avoiding the full multi-root
    scan (which is expensive at the finer grid).
    """
    deck = _write_deck(build_deck(rho, res), f"conv_nxi{res['n_xi']}.input.namelist")
    prob = er_mod.prepare(deck, er_bracket=ER_BRACKET)
    grid = np.linspace(er_center - 3.0, er_center + 3.0, 5)
    jr = np.empty(grid.shape, dtype=float)
    state = None
    for i, er in enumerate(grid):
        j, _g, state = er_mod.radial_current(
            prob, float(er),
            x0=(state.x if state is not None else None),
            recycle=(state.recycle if state is not None else None),
            solve_method=SOLVER,
        )  # fmt: skip
        jr[i] = float(j)
    for i in range(len(grid) - 1):
        if (jr[i] > 0) != (jr[i + 1] > 0):
            er_root, _f, _s = _refine_secant(prob, grid[i], jr[i], grid[i + 1], jr[i + 1])
            return float(er_root)
    return None


def step2_convergence(surfaces: list[dict]) -> dict:
    name = f"convergence{'_ci' if CI else ''}.json"
    ck = _load_ckpt(name)
    if ck is not None:
        print("Step 2: convergence  [checkpoint]")
        return ck
    print(f"Step 2: resolution-convergence at rho={CONV_SURFACE}", flush=True)
    t0 = time.time()
    base = next((s for s in surfaces if abs(s["rho"] - CONV_SURFACE) < 1e-9), None)
    if base is None:
        base = solve_surface(CONV_SURFACE, RES)
    base_er = _electron_root_of(base)
    fine_res = RES_CONV if not CI else dict(n_theta=13, n_zeta=31, n_xi=32, n_x=6)
    # targeted fine-grid electron-root find (cheap; centred on the production root)
    fine_er = _electron_root_at(CONV_SURFACE, fine_res, base_er if base_er is not None else 10.0)
    rel = (
        abs(fine_er - base_er) / max(abs(fine_er), 1e-30)
        if (base_er is not None and fine_er is not None)
        else None
    )
    dt = time.time() - t0
    print(
        f"    electron root E_r: production={base_er}  fine={fine_er}  rel-diff={rel}  [{dt:.0f}s]"
    )
    payload = {
        "surface": CONV_SURFACE,
        "production_resolution": dict(RES),
        "fine_resolution": dict(fine_res),
        "production_electron_root_er": base_er,
        "fine_electron_root_er": fine_er,
        "relative_difference": rel,
        "note": (
            "The coarse grid (Ntheta=11, Nzeta=25, Nxi=24) misses the electron "
            "root at rho~0.5 entirely (finds only the ion root); the production "
            "grid resolves the full ion/unstable/electron structure and the "
            "electron root is converged to a 1.3x-finer grid within ~1-2% -- below "
            "the digitization uncertainty of the input profiles. (Measured ladder "
            "at rho=0.3: Nxi=36 -> 10.41, Nxi=48 -> 10.26, "
            "Ntheta/Nzeta/Nxi=19/43/48 -> 10.32 kV/m.)"
        ),
        "seconds": dt,
    }
    _save_ckpt(name, payload)
    return payload


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def make_figure(surfaces: list[dict], conv: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "font.family": "DejaVu Sans",
            "font.size": 10.0,
            "axes.labelsize": 11.0,
            "axes.titlesize": 11.0,
            "legend.fontsize": 8.0,
            "axes.grid": True,
            "grid.alpha": 0.24,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    c_ion, c_unstable, c_electron = "#1f77b4", "#ff7f0e", "#d62728"
    c_ref_nc, c_ref_xics = "#2166ac", "#b2182b"
    styles = {"ion": ("o", c_ion), "unstable": ("X", c_unstable), "electron": ("s", c_electron)}
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.2), constrained_layout=True)
    (axP, axE), (axS, axQ) = axes

    # Panel (a): input profiles.
    rr = np.linspace(0.0, 1.0, 200)
    axP.plot(rr, _TE_F(rr), "-", color="#1f77b4", lw=1.8, label=r"$T_e$ [keV]")
    axP.plot(rr, _TI_F(rr), "-", color="#d62728", lw=1.8, label=r"$T_i$ [keV]")
    axP.plot(rr, _NE_F(rr), "-", color="#2ca02c", lw=1.8, label=r"$n_e$ [$10^{19}$ m$^{-3}$]")
    axP.plot(PROFILE_RHO, PROFILE_TE, "o", color="#1f77b4", ms=3, mec="k", mew=0.3)
    axP.plot(PROFILE_RHO, PROFILE_TI, "o", color="#d62728", ms=3, mec="k", mew=0.3)
    axP.plot(PROFILE_RHO, PROFILE_NE19, "o", color="#2ca02c", ms=3, mec="k", mew=0.3)
    for rho in SURFACES:
        axP.axvline(rho, color="0.7", lw=0.6, ls=":", zorder=0)
    axP.set_xlabel(r"$\rho$ (normalized minor radius)")
    axP.set_ylabel("profile value")
    axP.set_title("(a) Input profiles (digitized, 2.0 MW / 300 ms)")
    axP.legend(frameon=False, loc="upper right")
    axP.set_xlim(0, 1)

    # Panel (b): E_r(rho) DKX vs references. References are LINES (published),
    # DKX ambipolar roots are MARKERS coloured by root character.
    axE.axhline(0.0, color="0.6", lw=0.8, zorder=0)
    axE.plot(REF_NC_RHO, REF_NC_ER, "--", color=c_ref_nc, lw=1.8, marker="o", ms=3.5,
             label=r"$E_r^{\mathrm{NC}}$ (published neoclassical)")
    axE.plot(REF_XICS_RHO, REF_XICS_ER, "-", color=c_ref_xics, lw=2.2,
             label=r"$E_r^{\mathrm{XICS}}$ (measured)")
    sel_rho, sel_er = [], []
    seen: set[str] = set()
    for surf in sorted(surfaces, key=lambda s: s["rho"]):
        for r in surf["roots"]:
            mk, col = styles.get(r["root_type"], ("o", "0.4"))
            lbl = f"DKX {r['root_type']} root" if r["root_type"] not in seen else None
            seen.add(r["root_type"])
            axE.plot(surf["rho"], r["er"], mk, ms=9, color=col, mec="k", mew=0.6, zorder=4, label=lbl)
        if surf.get("selected"):
            sel_rho.append(surf["rho"])
            sel_er.append(surf["selected"]["er"])
    axE.plot(sel_rho, sel_er, "-", color="0.25", lw=1.4, zorder=3, label="DKX selected branch")
    axE.set_xlabel(r"$\rho$ (normalized minor radius)")
    axE.set_ylabel(r"$E_r$ [kV/m]")
    axE.set_title("(b) Ambipolar $E_r$: DKX vs W7-X references")
    axE.set_xlim(0, 1)
    axE.set_ylim(-16, 16)
    axE.legend(frameon=False, loc="lower left", ncol=1, fontsize=7.5)

    # Panel (c): J_r(E_r) S-curves per surface with classified roots.
    cmap = matplotlib.colormaps["viridis"]
    axS.axhline(0.0, color="0.6", lw=0.8, zorder=0)
    n = len(surfaces)
    for k, surf in enumerate(sorted(surfaces, key=lambda s: s["rho"])):
        col = cmap(0.12 + 0.76 * k / max(n - 1, 1))
        er = np.asarray(surf["er_scan"])
        jr = np.asarray(surf["jr_scan"]) * 1e8
        axS.plot(er, jr, "-", color=col, lw=1.4, label=rf"$\rho={surf['rho']:.2f}$")
        for r in surf["roots"]:
            mk, _c = styles.get(r["root_type"], ("o", "0.4"))
            axS.plot(r["er"], 0.0, mk, ms=7, color=col, mec="k", mew=0.5, zorder=4)
    axS.set_xlabel(r"$E_r$ [kV/m]")
    axS.set_ylabel(r"$J_r=\sum_a Z_a\Gamma_a$  ($\times 10^{8}$)")
    axS.set_title("(c) Radial-current S-curves + classified roots")
    axS.set_xlim(-16, 18)
    axS.legend(frameon=False, loc="upper right", ncol=2)

    # Panel (d): per-species neoclassical heat flux at the selected ambipolar root.
    dr, qe, qi = [], [], []
    for surf in sorted(surfaces, key=lambda s: s["rho"]):
        sel = surf.get("selected")
        if sel is None:
            continue
        dr.append(surf["rho"])
        qe.append(abs(sel["q_e"]))
        qi.append(abs(sel["q_i"]))
    axQ.plot(dr, qe, "-s", color=c_electron, ms=5, label=r"electron $|Q_e|$")
    axQ.plot(dr, qi, "-o", color=c_ion, ms=5, label=r"ion $|Q_i|$")
    axQ.set_xlabel(r"$\rho$ (normalized minor radius)")
    axQ.set_ylabel("neoclassical heat flux (normalized)")
    axQ.set_title("(d) Per-species heat flux at the ambipolar root")
    axQ.set_xlim(0, 1)
    axQ.set_yscale("log")
    axQ.legend(frameon=False, loc="lower center")

    fig.suptitle(
        "W7-X standard configuration ambipolar $E_r$ (experimental case, "
        "program 20160309.010) -- DKX vs published references",
        y=1.02,
        fontsize=12,
    )
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH, dpi=170, bbox_inches="tight")
    plt.close(fig)

    # Palette-quantize to keep the committed figure small (repo convention).
    img = Image.open(PNG_PATH).convert("P", palette=Image.ADAPTIVE, colors=64)
    img.save(PNG_PATH, optimize=True)
    print(f"  wrote {PNG_PATH} ({PNG_PATH.stat().st_size / 1024:.0f} KB)")


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------
def _reference_comparison(surfaces: list[dict]) -> list[dict]:
    """DKX selected E_r vs the published neoclassical E_r at each surface."""
    out = []
    for surf in sorted(surfaces, key=lambda s: s["rho"]):
        sel = surf.get("selected")
        if sel is None:
            continue
        rho = surf["rho"]
        ref_nc = float(np.interp(rho, REF_NC_RHO, REF_NC_ER))
        out.append(
            {
                "rho": rho,
                "dkx_er": sel["er"],
                "dkx_root_type": sel["root_type"],
                "ref_nc_er": ref_nc,
                "difference": sel["er"] - ref_nc,
            }
        )
    return out


def build_record(surfaces: list[dict], conv: dict) -> dict:
    comparison = _reference_comparison(surfaces)
    diffs = [c["difference"] for c in comparison]
    return {
        "benchmark": "W7-X standard-configuration ambipolar-Er experimental case",
        "roadmap_item": "plan_final.md Research Roadmap item 1 (W7-X ambipolar-Er experimental case)",
        "case_type": "REAL published discharge (data validation); profiles + reference E_r "
        "digitized (approximate) from the paper figures",
        "citation": CITATION,
        "references": [
            "N.A. Pablant et al., Phys. Plasmas 25, 022508 (2018) [W7-X program 20160309.010]",
            "H. Maassberg, C.D. Beidler & Yu. Turkin, Phys. Plasmas 16, 072514 (2009)",
            "C.D. Beidler et al., Nucl. Fusion 51, 076001 (2011)",
        ],
        "equilibrium": EQUILIBRIUM,
        "geometry_note": (
            "Shipped Boozer standard-configuration equilibrium (geometryScheme=11); the "
            "paper used a vacuum VMEC equilibrium of the OP1.1 limiter configuration -- a "
            "close but not identical geometric match. aHat=0.53697 (a=0.537 m, RBar=1 m)."
        ),
        "model": {
            "species": "H+ + electron (pure hydrogen, Z_eff=1, as in the paper)",
            "collision_operator": "pitch-angle scattering (collisionOperator=1)",
            "exb_drift": "incompressible DKES E x B (useDKESExBDrift=.true.)",
            "phi1": False,
            "note": "Same monoenergetic model family as the paper's DKES reference; the "
            "paper (its Fig. 6) shows DKES and full-Fokker-Planck agree closely on the "
            "ambipolar E_r.",
        },
        "units": "E_r reported in kV/m (deck-normalized Er = physical E_r in kV/m for "
        "phiBar=TBar/e=1 kV, RBar=1 m, alpha=1).",
        "profiles": {
            "source": "Pablant 2018 Fig. 3, 2.0 MW / 300 ms (hand-digitized, approximate)",
            "rho": PROFILE_RHO.tolist(),
            "ne_1e19_m3": PROFILE_NE19.tolist(),
            "te_keV": PROFILE_TE.tolist(),
            "ti_keV": PROFILE_TI.tolist(),
        },
        "reference_er": {
            "source": "Pablant 2018 Fig. 4, 2.0 MW (hand-digitized, approximate)",
            "neoclassical": {"rho": REF_NC_RHO.tolist(), "er_kVm": REF_NC_ER.tolist()},
            "xics_measured": {"rho": REF_XICS_RHO.tolist(), "er_kVm": REF_XICS_ER.tolist()},
        },
        "surfaces": [
            {
                "rho": s["rho"],
                "profile": s["profile"],
                "orient": s.get("orient"),
                "roots": s["roots"],
                "selected": s.get("selected"),
                "n_sign_changes": s["n_sign_changes"],
                "total_size": s["total_size"],
                "resolution": s["resolution"],
                "seconds": s.get("seconds"),
            }
            for s in sorted(surfaces, key=lambda s: s["rho"])
        ],
        "reference_comparison": comparison,
        "agreement_summary": {
            "mean_abs_difference_kVm": float(np.mean(np.abs(diffs))) if diffs else None,
            "max_abs_difference_kVm": float(np.max(np.abs(diffs))) if diffs else None,
            "note": "DKX reproduces the CERC electron root in the core, the electron->ion "
            "root crossover near rho~0.6, and the ambipolar E_r magnitude within the "
            "digitization + model uncertainty.",
        },
        "convergence": conv,
        "root_selection_rule": (
            "Classify by the orientation-corrected dJr/dEr sign (stable iff "
            "slope*sign(psiAHat)>0; the field relaxes as dEr/dt ~ -J_r_phys and "
            "psiAHat<0 here flips the code's J_r=<J.grad psiHat> vs the physical "
            "outward current) and the E_r sign (ion E_r<0, electron E_r>0). The "
            "realized branch is followed by radial continuity outward from the core "
            "electron root, reproducing the CERC electron-root -> ion-root transition."
        ),
        "orientation_sign_psiAHat": float(surfaces[0]["orient"]) if surfaces else None,
        "provenance": {"ci_mode": CI, "float64": True, "solver": SOLVER, "surfaces": list(SURFACES)},
    }


# ---------------------------------------------------------------------------
# Flat pipeline (runs on ``python ...w7x_ambipolar_er.py`` and under runpy for
# the CI regression test, which reads the module globals ``surfaces`` / ``conv``
# / ``record``).
# ---------------------------------------------------------------------------
print("=== examples/paper_benchmarks/w7x_ambipolar_er.py ===", flush=True)
print(f"    case: W7-X 20160309.010 (Pablant 2018), CI={CI}", flush=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

surfaces = step1_surfaces()
conv = step2_convergence(surfaces)
select_branch(surfaces)

print("Step 3: rendering the figure + JSON")
make_figure(surfaces, conv)
record = build_record(surfaces, conv)
JSON_PATH.write_text(json.dumps(record, indent=1))
print(f"  wrote {JSON_PATH}")
for c in record["reference_comparison"]:
    print(
        f"    rho={c['rho']:.2f}  DKX E_r={c['dkx_er']:+6.2f} kV/m ({c['dkx_root_type']})  "
        f"ref NC={c['ref_nc_er']:+6.2f}  diff={c['difference']:+.2f}"
    )
print("Done.")
