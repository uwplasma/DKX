"""Differentiable ambipolar-Er / electron-root optimization workflow.

Roadmap item 3 of ``plan_final.md``: productize ``d(ambipolar Er)/d(shape or
profile)`` through the implicit ambipolar root, with robust multi-root
resolution, ion / electron branch classification, and graceful handling of the
branch transition (near-degenerate roots) at the electron-root onset.

Physics
-------
A stellarator radial electric field is set by ambipolarity
``J_r(E_r) = sum_a Z_a Gamma_a(E_r) = 0``.  When the field is
non-intrinsically-ambipolar and the plasma is at low collisionality, the
species fluxes ``Gamma_a(E_r)`` are non-monotonic and ``J_r(E_r)`` can be an
S-curve with **three** roots: a stable **ion root** (``E_r < 0``,
``dJr/dEr > 0``), an **unstable** middle branch (``dJr/dEr < 0``) and a stable
**electron root** (``E_r > 0``).  Strong electron heating (large ``T_e/T_i``,
steep ``dT_e/dr`` -- the ECH / core-electron-root-confinement regime) drives the
transition from the ion root to the electron root
[H. Maassberg, C.D. Beidler & E.E. Simmet, Plasma Phys. Control. Fusion 41,
1135 (1999); Yu. Turkin et al., Phys. Plasmas 18, 022505 (2011);
D.A. Spong, Phys. Plasmas 12, 056114 (2005)].  The stable/unstable split is the
sign of ``dJr/dEr`` because the field relaxes as ``dEr/dt ~ -J_r``
[the ambipolarity/neoclassical-Er references cited in ``dkx/er.py``].

Modeling honesty
----------------
On the tiny analytic single-helicity Boozer field used here, the physical
electron/ion mass ratio ``m_e/m_i = 1/1836`` geometrically suppresses the
electron neoclassical channel by ``(rho_e/rho_i)^2`` and places its
superbanana resonance outside the ``E_r`` window resolvable at laptop
resolution -- a wide profile scan on that deck yields only the ion root
(documented in the JSON ``modeling_note``).  To *exhibit and differentiate* the
full three-branch structure and the ion->electron transition on a laptop, the
demonstration uses a **reduced ion/electron mass ratio** (``mHat_e = 0.25``)
that brings the electron channel and its resonance into the same ``E_r`` window
as the ion channel.  The mathematical branch structure (ion root / unstable
branch / electron root, the saddle-node transition, the implicit-function-
theorem sensitivity) is identical; only the resonance *scale* is rescaled.

What the script produces
------------------------
1. ``J_r(E_r)`` S-curve at the transition ``T_e/T_i`` with all roots resolved
   and classified ion / unstable / electron (moderate resolution; the tier-1
   block solve, no autodiff).
2. A ``T_e/T_i`` scan bracketing the electron-root onset: the ambipolar-Er
   bifurcation diagram (ion-root branch, unstable branch, electron-root branch)
   and the near-degenerate saddle-node where the electron/unstable pair is born.
3. The differentiable objective ``d(selected ambipolar Er)/d(profile scale)``
   through :func:`dkx.er.ambipolar_er` (implicit-function-theorem root solve,
   ``jax.lax.custom_root`` via solvax -- no finite differences in the AD path),
   verified against a central finite difference; and a small gradient-descent
   optimization that drives the ambipolar ``E_r`` toward a target by tuning a
   profile parameter, reporting the objective decreasing.  The differentiable
   root uses an exact dense solve, feasible only at reduced resolution, so it is
   evaluated on a small deck and differentiates the robust ion root.

Outputs: ``docs/_static/figures/paper_benchmarks/electron_root_optimization.png``
(three panels: S-curve, Te/Ti bifurcation, optimization) and the companion
``.json`` with roots, gradients, the optimization history, and provenance.
Intermediate results are checkpointed under ``output/electron_root/`` and reused
on re-runs (set ``DKX_EROOT_FORCE=1`` to recompute).

Expected runtime: ~3-4 min on a laptop CPU (float64); the CI regression version
lives in ``tests/test_paper_benchmark_electron_root.py``.

Run (from the repo root):
  python examples/paper_benchmarks/electron_root_optimization.py
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)

from dkx import er as er_mod  # noqa: E402
from dkx.workflows.optimization import find_ambipolar_roots  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
# Output/figure directories are overridable so the CI regression test can
# redirect them to temp dirs without clobbering the committed artifacts.
OUT_DIR = Path(os.environ.get("DKX_EROOT_OUT_DIR", Path(__file__).parent / "output" / "electron_root"))
FIG_DIR = Path(os.environ.get("DKX_EROOT_FIG_DIR", REPO_ROOT / "docs" / "_static" / "figures" / "paper_benchmarks"))
PNG_PATH = FIG_DIR / "electron_root_optimization.png"
JSON_PATH = FIG_DIR / "electron_root_optimization.json"

CI = os.environ.get("DKX_EROOT_CI") == "1"
FORCE = os.environ.get("DKX_EROOT_FORCE") == "1"

# ---------------------------------------------------------------------------
# Parameters (simsopt-style: everything at the top)
# ---------------------------------------------------------------------------

# Reduced ion/electron mass ratio: brings the electron channel's resonance into
# the resolvable E_r window (see the module "Modeling honesty" note).
MHAT_ELECTRON = 0.25
# Helically rippled analytic Boozer field (non-intrinsically-ambipolar).
GEOM = dict(iota=1.31, eps_t=0.10, eps_h=0.08, helicity_l=2, helicity_n=5)
# Reactor-relevant low-collisionality regime and profiles.
NU_N = 1.0e-3
DN = -0.5  # d n / d r (both species)
DTI = -1.0  # d T_ion / d r (steep ion temperature gradient)
DTE = -3.0  # d T_electron / d r (ECH-like electron heating)

# Te/Ti transition ladder (electron-heating knob) bracketing the onset.
# 0.70 is the saddle-node fold (near-degenerate electron/unstable pair); 0.72
# is the interior three-root point (ion / unstable / electron well separated).
TE_RATIOS = (
    (0.30, 0.50, 0.70, 0.72, 0.80, 0.90, 1.10, 1.50) if not CI else (0.30, 0.72, 1.50)
)
# The Te/Ti where the S-curve exhibits three well-separated roots.
TE_TRANSITION = 0.72

# Moderate resolution for the (autodiff-free) S-curve scan: tier-1 block solve.
# The electron channel's superbanana resonance needs Nxi >~ 28 to converge
# (below that the ion/electron trend is unresolved); the tier-1 block solve
# keeps each J_r(E_r) evaluation to a fraction of a second even at Nxi=32.
RES_SCAN = dict(n_theta=9, n_zeta=9, n_xi=32, n_x=6) if not CI else dict(
    n_theta=7, n_zeta=7, n_xi=24, n_x=5
)
# Small resolution for the differentiable root (exact dense solve is O(N^2)).
RES_GRAD = dict(n_theta=7, n_zeta=7, n_xi=8, n_x=3)

ER_BRACKET = (-6.0, 8.0)
N_ER_SCAN = 43 if not CI else 15
N_ER_FINE = 121 if not CI else 41

# Optimization: drive the (differentiable) ambipolar Er toward a target by
# scaling the electron temperature gradient (the ECH control knob).
ER_TARGET = -0.15
OPT_STEPS = 8 if not CI else 3
OPT_LR = 6.0

FD_STEP = 1.0e-3


# ---------------------------------------------------------------------------
# Deck builder
# ---------------------------------------------------------------------------
def build_deck(*, te_ratio: float, dte: float, res: dict) -> str:
    """A two-species helically rippled deck with a reduced e/i mass ratio."""
    return f"""&general
/
&geometryParameters
  geometryScheme = 1
  inputRadialCoordinate = 3
  rN_wish = 0.3
  B0OverBBar = 1.0
  GHat = 1.0
  IHat = 0.0
  iota = {GEOM["iota"]}
  epsilon_t = {GEOM["eps_t"]}
  epsilon_h = {GEOM["eps_h"]}
  helicity_l = {GEOM["helicity_l"]}
  helicity_n = {GEOM["helicity_n"]}
  psiAHat = 0.045
  aHat = 0.1
/
&speciesParameters
  Zs = 1 -1
  mHats = 1.0 {MHAT_ELECTRON}
  nHats = 1.0 1.0
  THats = 1.0 {te_ratio}
  dNHatdrHats = {DN} {DN}
  dTHatdrHats = {DTI} {dte}
/
&physicsParameters
  Delta = 4.5694d-3
  alpha = 1.0
  nu_n = {NU_N}
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
  solverTolerance = 1d-10
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


def _classify(er: float, slope: float) -> str:
    """ion / electron / unstable from the E_r sign and the dJr/dEr sign.

    Mirrors :func:`dkx.er._classify`: a root is stable iff ``dJr/dEr > 0``
    (the field relaxes as ``dEr/dt ~ -J_r``); the stable outer roots are the
    ion root (``E_r < 0``) and electron root (``E_r > 0``).
    """
    if slope < 0.0:
        return "unstable"
    return "electron" if er > 0.0 else "ion"


def scan_radial_current(deck_path: Path, er_values: np.ndarray) -> np.ndarray:
    """J_r at each E_r via the canonical solve (warm-started, tier-1 block)."""
    prob = er_mod.prepare(deck_path, er_bracket=ER_BRACKET)
    jr = np.empty(er_values.shape, dtype=np.float64)
    state = None
    for i, er in enumerate(er_values):
        j, _gamma, state = er_mod.radial_current(
            prob, float(er),
            x0=(state.x if state is not None else None),
            recycle=(state.recycle if state is not None else None),
            solve_method="block_tridiagonal",
        )
        jr[i] = float(j)
    return jr


def resolve_roots(er_values: np.ndarray, jr: np.ndarray) -> list[dict]:
    """Resolve and classify EVERY ambipolar root on the scanned S-curve.

    Reuses :func:`dkx.workflows.optimization.find_ambipolar_roots` (linear
    bracketing) for the root E_r values and their local ``dJr/dEr`` slopes, then
    applies the stable/unstable + ion/electron classification.
    """
    summary = find_ambipolar_roots(er_values, jr)
    roots = []
    for r in summary.roots:
        roots.append(
            {
                "er": float(r.er),
                "slope": float(r.slope),
                "root_type": _classify(float(r.er), float(r.slope)),
                "bracket": [float(r.bracket[0]), float(r.bracket[1])],
            }
        )
    return roots


def flag_near_degenerate(roots: list[dict], *, tol: float = 0.25) -> list[list[int]]:
    """Group roots that are near-degenerate (a saddle-node/fold transition).

    Two ambipolar roots whose E_r values are within ``tol`` are the electron
    root and the unstable branch being born (or the ion and unstable branch
    annihilating).  Near the fold ``dJr/dEr -> 0`` for both, so the implicit-
    function-theorem sensitivity ``dEr/dp = -(dJr/dEr)^{-1} dJr/dp`` is stiff;
    the selection rule must pick a well-separated stable root there.
    """
    ers = sorted(range(len(roots)), key=lambda i: roots[i]["er"])
    groups: list[list[int]] = []
    for i in ers:
        if groups and abs(roots[i]["er"] - roots[groups[-1][-1]]["er"]) <= tol:
            groups[-1].append(i)
        else:
            groups.append([i])
    return [g for g in groups if len(g) > 1]


def select_root(roots: list[dict], *, prefer: str) -> dict | None:
    """Documented physical selection rule.

    Pick the requested stable branch: the electron root is the most positive
    stable (``dJr/dEr > 0``) root; the ion root is the most negative stable
    root.  Near-degenerate (fold) roots (``|dJr/dEr|`` tiny) are skipped -- a
    fold root is not a robust operating point.  Falls back to the other stable
    branch, then to any root.
    """
    stable = [r for r in roots if r["slope"] > 1e-14]
    ion = sorted((r for r in stable if r["root_type"] == "ion"), key=lambda r: r["er"])
    electron = sorted(
        (r for r in stable if r["root_type"] == "electron"), key=lambda r: r["er"]
    )
    if prefer == "electron" and electron:
        return electron[-1]
    if prefer == "ion" and ion:
        return ion[0]
    if electron:
        return electron[-1]
    if ion:
        return ion[0]
    return roots[0] if roots else None


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
# Step 1: the S-curve at the transition Te/Ti
# ---------------------------------------------------------------------------
def step1_scurve() -> dict:
    ck = _load_ckpt("scurve.json")
    if ck is not None:
        print("Step 1: S-curve  [checkpoint]")
        return ck
    print(f"Step 1: J_r(E_r) S-curve at Te/Ti = {TE_TRANSITION} (resolving all roots)")
    t0 = time.time()
    deck = _write_deck(
        build_deck(te_ratio=TE_TRANSITION, dte=DTE, res=RES_SCAN), "scurve.input.namelist"
    )
    er_fine = np.linspace(ER_BRACKET[0], ER_BRACKET[1], N_ER_FINE)
    jr = scan_radial_current(deck, er_fine)
    roots = resolve_roots(er_fine, jr)
    folds = flag_near_degenerate(roots)
    n_sign_changes = int(np.sum(np.sign(jr[:-1]) * np.sign(jr[1:]) < 0))
    for r in roots:
        print(f"    root  E_r = {r['er']:+.4f}  dJr/dEr = {r['slope']:+.3e}  ({r['root_type']})")
    if folds:
        print(f"    near-degenerate (saddle-node) branch groups: {folds}")
    # Apply the documented physical selection rule to both operating regimes.
    selected = {
        "ion": select_root(roots, prefer="ion"),
        "electron": select_root(roots, prefer="electron"),
    }
    for regime, r in selected.items():
        if r is not None:
            print(f"    selected {regime}-regime root: E_r = {r['er']:+.4f}")
    dt = time.time() - t0
    print(f"    {len(roots)} roots, {n_sign_changes} sign changes  [{dt:.1f}s]")
    payload = {
        "te_ratio": TE_TRANSITION,
        "er": er_fine.tolist(),
        "radial_current": jr.tolist(),
        "roots": roots,
        "n_sign_changes": n_sign_changes,
        "near_degenerate_groups": folds,
        "selected_roots": selected,
        "seconds": dt,
    }
    _save_ckpt("scurve.json", payload)
    return payload


# ---------------------------------------------------------------------------
# Step 2: Te/Ti transition (bifurcation diagram)
# ---------------------------------------------------------------------------
def step2_bifurcation() -> dict:
    ck = _load_ckpt("bifurcation.json")
    if ck is not None:
        print("Step 2: Te/Ti bifurcation  [checkpoint]")
        return ck
    print("Step 2: Te/Ti scan bracketing the electron-root onset")
    t0 = time.time()
    er_scan = np.linspace(ER_BRACKET[0], ER_BRACKET[1], N_ER_SCAN)
    entries = []
    for te in TE_RATIOS:
        deck = _write_deck(
            build_deck(te_ratio=te, dte=DTE, res=RES_SCAN), f"bif_te{te:.2f}.input.namelist"
        )
        jr = scan_radial_current(deck, er_scan)
        roots = resolve_roots(er_scan, jr)
        jr0 = float(np.interp(0.0, er_scan, jr))
        types = sorted({r["root_type"] for r in roots})
        folds = flag_near_degenerate(roots)
        entries.append(
            {"te_ratio": float(te), "roots": roots, "jr_at_zero": jr0,
             "root_types": types, "near_degenerate_groups": folds}
        )
        fold_note = "  [near-degenerate fold]" if folds else ""
        root_str = ", ".join(
            "{:+.2f}({})".format(r["er"], r["root_type"][:3]) for r in roots
        )
        print(
            f"    Te/Ti = {te:.2f}:  {len(roots)} root(s)  "
            f"[{root_str}]  J_r(0) = {jr0:+.2e}{fold_note}"
        )
    dt = time.time() - t0
    # locate the onset: first Te/Ti where a stable electron root appears.
    onset = next(
        (e["te_ratio"] for e in entries
         if any(r["root_type"] == "electron" and r["slope"] > 0 for r in e["roots"])),
        None,
    )
    print(f"    electron-root onset near Te/Ti = {onset}  [{dt:.1f}s]")
    payload = {"entries": entries, "electron_root_onset": onset, "seconds": dt,
               "er_scan": er_scan.tolist()}
    _save_ckpt("bifurcation.json", payload)
    return payload


# ---------------------------------------------------------------------------
# Step 3 + 4: differentiable ambipolar Er + optimization (small deck)
# ---------------------------------------------------------------------------
def _grad_problem():
    """Prepare the small-deck ErProblem and the base electron dT/dr leaf."""
    deck = _write_deck(
        build_deck(te_ratio=1.5, dte=DTE, res=RES_GRAD), "grad.input.namelist"
    )
    prob = er_mod.prepare(deck, er_bracket=ER_BRACKET)
    found = er_mod.find_ambipolar_er(deck, er_bracket=ER_BRACKET, all_roots=True, emit=None)
    return deck, prob, found


def _er_of_scale(prob, base_op, base_dt):
    """Return ``p -> ambipolar_er`` scaling the electron dT/dr by ``p``.

    ``p`` is the ECH control knob (electron temperature-gradient scale); the
    ambipolar Er is a differentiable function of it through the implicit root.
    """

    def er_of(p, er0):
        dt_scaled = base_dt.at[1].multiply(p)  # electron component only
        op_p = replace(base_op, dt_hat_dpsi_hat=dt_scaled)
        return er_mod.ambipolar_er(
            op_p, er0=er0, dphi_per_er=prob.dphi_per_er, z_s=prob.z_s
        )

    return er_of


def step34_differentiable() -> dict:
    ck = _load_ckpt("differentiable.json")
    if ck is not None:
        print("Step 3-4: differentiable Er + optimization  [checkpoint]")
        return ck
    print("Step 3: differentiable ambipolar Er  d(Er)/d(electron dT/dr scale)")
    t0 = time.time()
    _deck, prob, found = _grad_problem()
    root0 = float(found.er)
    base_op = prob.operator
    base_dt = base_op.dt_hat_dpsi_hat
    sel = _classify(root0, float(found.roots[0].slope) if found.roots else 1.0)
    print(f"    base ambipolar root: E_r = {root0:+.5f}  ({sel} root, small deck)")

    er_of = _er_of_scale(prob, base_op, base_dt)

    def er_p(p):
        return er_of(p, root0)

    fwd = float(er_p(1.0))
    grad_ad = float(jax.grad(er_p)(1.0))
    fd = (float(er_p(1.0 + FD_STEP)) - float(er_p(1.0 - FD_STEP))) / (2.0 * FD_STEP)
    rel_dev = abs(grad_ad - fd) / max(abs(fd), 1e-30)
    print(f"    Er(p=1) = {fwd:+.6f}")
    print(f"    AD = {grad_ad:+.6e}   central FD = {fd:+.6e}   rel dev = {rel_dev:.2e}")

    # Step 4: gradient-descent optimization driving Er toward a target.
    print(f"Step 4: drive the ambipolar Er toward target E_r = {ER_TARGET:+.3f}")

    def objective(p):
        return 0.5 * (er_p(p) - ER_TARGET) ** 2

    val_and_grad = jax.value_and_grad(objective)
    p = 1.0
    history = []
    for it in range(OPT_STEPS + 1):
        obj, g = val_and_grad(p)
        er_here = float(er_p(p))
        history.append(
            {"iter": it, "p": float(p), "objective": float(obj),
             "grad": float(g), "er": er_here}
        )
        print(
            f"    iter {it:2d}:  p = {p:.5f}  Er = {er_here:+.5f}  "
            f"obj = {float(obj):.4e}  |grad| = {abs(float(g)):.3e}"
        )
        if it < OPT_STEPS:
            p = float(p - OPT_LR * float(g))
    dt = time.time() - t0
    obj0 = history[0]["objective"]
    objN = history[-1]["objective"]
    print(f"    objective {obj0:.4e} -> {objN:.4e}  ({objN / obj0:.2e} x)  [{dt:.1f}s]")
    payload = {
        "resolution": RES_GRAD,
        "base_root_er": root0,
        "base_root_type": sel,
        "sensitivity": {
            "parameter": "electron dT/dr scale (ECH control knob)",
            "er_at_p1": fwd,
            "ad": grad_ad,
            "fd": fd,
            "fd_step": FD_STEP,
            "rel_dev": rel_dev,
        },
        "optimization": {
            "target_er": ER_TARGET,
            "learning_rate": OPT_LR,
            "history": history,
            "objective_start": obj0,
            "objective_end": objN,
        },
        "seconds": dt,
    }
    _save_ckpt("differentiable.json", payload)
    return payload


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def make_figure(scurve: dict, bif: dict, diff: dict) -> None:
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
            "legend.fontsize": 8.5,
            "axes.grid": True,
            "grid.alpha": 0.24,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    c_ion, c_unstable, c_electron = "#1f77b4", "#ff7f0e", "#d62728"
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.4, 3.9), constrained_layout=True)

    # Panel (a): the S-curve with classified roots.
    er = np.asarray(scurve["er"])
    jr = np.asarray(scurve["radial_current"]) * 1e6  # scale for readability
    ax1.axhline(0.0, color="0.6", lw=0.8, zorder=0)
    ax1.plot(er, jr, "-", color="0.25", lw=1.6, zorder=2)
    styles = {"ion": ("o", c_ion), "unstable": ("X", c_unstable), "electron": ("s", c_electron)}
    seen = set()
    for r in scurve["roots"]:
        mk, col = styles.get(r["root_type"], ("o", "0.4"))
        lbl = r["root_type"] if r["root_type"] not in seen else None
        seen.add(r["root_type"])
        ax1.plot(r["er"], 0.0, mk, ms=9, color=col, mec="k", mew=0.6, zorder=4, label=lbl)
    ax1.set_xlabel(r"$E_r$  (deck-normalized)")
    ax1.set_ylabel(r"$J_r = \sum_a Z_a\Gamma_a$  ($\times 10^{6}$)")
    ax1.set_title(f"Ambipolar S-curve  ($T_e/T_i = {scurve['te_ratio']}$)")
    ax1.legend(frameon=False, loc="upper left")

    # Panel (b): the Te/Ti bifurcation diagram.
    for entry in bif["entries"]:
        te = entry["te_ratio"]
        for r in entry["roots"]:
            _mk, col = styles.get(r["root_type"], ("o", "0.4"))
            ax2.plot(te, r["er"], "o", ms=6, color=col, mec="k", mew=0.5, zorder=3)
    ax2.axhline(0.0, color="0.6", lw=0.8, zorder=0)
    for name, col in (("ion", c_ion), ("unstable", c_unstable), ("electron", c_electron)):
        ax2.plot([], [], "o", ms=6, color=col, mec="k", mew=0.5, label=f"{name} root")
    ax2.set_xlabel(r"$T_e/T_i$  (electron-heating knob)")
    ax2.set_ylabel(r"ambipolar $E_r$ roots")
    ax2.set_title(r"Ion-root $\to$ electron-root transition")
    ax2.legend(frameon=False, loc="upper left")
    onset = bif.get("electron_root_onset")
    if onset is not None:
        ax2.axvline(onset, color="0.5", ls="--", lw=1.0, zorder=1)
        y_lo, y_hi = ax2.get_ylim()
        ax2.text(onset + 0.02, y_lo + 0.06 * (y_hi - y_lo), "electron-root\nonset",
                 fontsize=8, va="bottom", ha="left", color="0.35")

    # Panel (c): the optimization objective + |grad| vs iteration.
    hist = diff["optimization"]["history"]
    its = [h["iter"] for h in hist]
    objs = [h["objective"] for h in hist]
    grads = [abs(h["grad"]) for h in hist]
    ax3.semilogy(its, objs, "-o", ms=5, color=c_ion,
                 label=r"objective $\frac{1}{2}(E_r - E_r^{\rm target})^2$")
    ax3.semilogy(its, grads, "-s", ms=5, color=c_electron,
                 label=r"$|\mathrm{d\,obj}/\mathrm{d}p|$")
    ax3.set_xlabel("iteration")
    ax3.set_ylabel("objective  /  |gradient|")
    ax3.set_title("Drive $E_r$ to target (implicit-diff grad)")
    ax3.legend(frameon=False, loc="lower left")

    sens = diff["sensitivity"]
    fig.suptitle(
        "Differentiable ambipolar-Er / electron-root workflow  "
        f"(reduced $m_e/m_i={MHAT_ELECTRON}$; AD-vs-FD rel dev {sens['rel_dev']:.1e})",
        y=1.05,
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
def build_record(scurve: dict, bif: dict, diff: dict) -> dict:
    return {
        "benchmark": "differentiable ambipolar-Er / electron-root optimization workflow",
        "roadmap_item": "plan_final.md Research Roadmap item 3",
        "references": [
            "H. Maassberg, C.D. Beidler & E.E. Simmet, Plasma Phys. Control. Fusion 41, 1135 (1999)",
            "Yu. Turkin et al., Phys. Plasmas 18, 022505 (2011)",
            "D.A. Spong, Phys. Plasmas 12, 056114 (2005)",
        ],
        "geometry": GEOM,
        "physics": {"nu_n": NU_N, "dn": DN, "dti": DTI, "dte": DTE,
                    "mHat_electron": MHAT_ELECTRON},
        "scurve": {
            "te_ratio": scurve["te_ratio"],
            "roots": scurve["roots"],
            "n_sign_changes": scurve["n_sign_changes"],
            "near_degenerate_groups": scurve["near_degenerate_groups"],
            "selected_roots": scurve.get("selected_roots"),
        },
        "bifurcation": {
            "te_ratios": [e["te_ratio"] for e in bif["entries"]],
            "root_counts": [len(e["roots"]) for e in bif["entries"]],
            "electron_root_onset": bif["electron_root_onset"],
            "entries": bif["entries"],
        },
        "differentiable": diff,
        "root_selection_rule": (
            "Classify by dJr/dEr sign (stable iff dJr/dEr > 0; the field relaxes "
            "as dEr/dt ~ -J_r) and by E_r sign (ion root E_r<0, electron root "
            "E_r>0). Select the requested stable branch (most positive stable = "
            "electron; most negative stable = ion); skip near-degenerate fold "
            "roots (|dJr/dEr| tiny) as non-robust operating points."
        ),
        "modeling_note": (
            "The physical m_e/m_i=1/1836 geometrically suppresses the electron "
            "neoclassical channel by (rho_e/rho_i)^2 on this analytic single-"
            "helicity field and pushes its superbanana resonance outside the "
            "laptop-resolvable E_r window, so that deck yields only the ion root "
            "across a wide profile scan. A reduced mass ratio mHat_e=0.25 brings "
            "the electron channel and its resonance into the same E_r window as "
            "the ion channel; the branch structure (ion/unstable/electron, the "
            "saddle-node transition, the implicit-diff sensitivity) is identical. "
            "The differentiable root uses an exact dense solve (feasible only at "
            "reduced resolution) and differentiates the robust ion root; the "
            "electron-root S-curve is resolved by the block solve at converged "
            "resolution."
        ),
        "provenance": {
            "resolution_scan": RES_SCAN,
            "resolution_grad": RES_GRAD,
            "er_bracket": list(ER_BRACKET),
            "ci_mode": CI,
            "float64": True,
        },
    }


# ---------------------------------------------------------------------------
# Flat pipeline (runs on ``python ...electron_root_optimization.py`` and also
# under runpy for the CI regression test, which reads the module globals
# ``scurve`` / ``bif`` / ``diff``).
# ---------------------------------------------------------------------------
print("=== examples/paper_benchmarks/electron_root_optimization.py ===", flush=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

scurve = step1_scurve()
bif = step2_bifurcation()
diff = step34_differentiable()

print("Step 5: rendering the figure + JSON")
make_figure(scurve, bif, diff)
record = build_record(scurve, bif, diff)
JSON_PATH.write_text(json.dumps(record, indent=1))
print(f"  wrote {JSON_PATH}")
print("Done.")
