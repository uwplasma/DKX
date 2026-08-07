"""Before/after physics of the QH low-bootstrap optimization.

Companion to ``examples/optimization/optimize_QH_bootstrap.py``.  That script
optimizes and writes ``output/input.optimize_QH_bootstrap_optimized``; this one
reloads the seed and that boundary and shows what changed, so the figure is
reproducible in minutes without repeating the ~80-minute optimization.

Three panels per configuration, which is what a reader needs to judge whether a
low-bootstrap QH configuration is still a QH configuration:

* the **boundary in 3D**, coloured by field strength, for the shape itself;
* **|B| on the last closed flux surface in Boozer angles**, where quasi-helical
  symmetry is visible directly --- straight contours in ``theta_B - nfp zeta_B``
  are what "QH" means, and their breakup is what a bootstrap optimizer will buy
  if the quasisymmetry penalty lets it;
* the **bootstrap-current profile** ``<j.B>/sqrt(<B^2>)(s)``, a kinetic solve per
  surface with the momentum-conserving Fokker-Planck operator, because a single
  mid-radius number hides whether a reduction is local or global.

The bootstrap profile is the expensive part: one tier-2 GCROT solve per surface
per configuration.  ``DKX_QH_PROFILE_SURFACES`` sets how many (default 5), and
``DKX_CI=1`` shrinks the resolution.

Read the profile panel and the summary table as answering different questions.
The table reports the optimizer's own figure of merit --- one surface, at the
resolution it optimized --- while the profile is recomputed here on its own
surfaces and, under ``DKX_CI=1``, at reduced resolution.  The two therefore
differ in magnitude (about 9% against about 1.5% on the CI profile) and neither
is wrong; run without ``DKX_CI`` for a profile at the optimization's own
resolution before quoting either.

Run:
  python examples/optimization/qh_bootstrap_before_after.py
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

jax.config.update("jax_enable_x64", True)
matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402  (registers 3d projection)

try:
    import vmex as _vmex_pkg
    from vmex.core import implicit as vmec_implicit
    from vmex.core.boozer_tables import boozer_input_tables
    from vmex.core.input import VmecInput
    from booz_xform_jax.jax_api import booz_xform_jax as booz_transform
except ImportError as exc:  # pragma: no cover - optional companions
    raise SystemExit(
        "This example needs vmex (new core API, with core.boozer_tables) and "
        "booz_xform_jax. Install with `pip install -e /path/to/vmex "
        "/path/to/booz_xform_jax`."
    ) from exc

import objectives as ob  # noqa: E402
from dkx.drift_kinetic import kinetic_operator_from_namelist  # noqa: E402
from dkx.inputs import parse_sfincs_input_text  # noqa: E402
from dkx.phase_space import make_grids  # noqa: E402

CI = os.environ.get("DKX_CI") == "1"
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "output"
OPTIMIZED = OUT_DIR / "input.optimize_QH_bootstrap_optimized"
SEED = Path(
    os.environ.get(
        "DKX_QH_VMEC_INPUT",
        Path(_vmex_pkg.__file__).resolve().parents[1]
        / "examples" / "data" / "input.LandremanPaul2021_QH_reactorScale_lowres",
    )
)

NFP = 4
NS = 7 if CI else 13
VMEC_FTOL = 1e-11 if CI else 1e-13
VMEC_MAX_ITER = 5000
MBOZ, NBOZ = (2, 2) if CI else (3, 3)
KIN = (9, 7, 8, 4, 4) if CI else (13, 11, 16, 4, 6)
COLLISION_OPERATOR = 0
DELTA, NU_N = 4.5694e-3, 0.00831565
ER, KIN_TOL = 1.0, 1e-9
A_MINOR, N_AXIS, T_AXIS = 1.70442623, 4.13, 12.0
N_SURFACES = int(os.environ.get("DKX_QH_PROFILE_SURFACES", "3" if CI else "5"))

_TEMPLATE = """&general
  RHSMode = 1
/
&geometryParameters
  geometryScheme = 1
  helicity_n = {nfp}
  psiAHat = {psi_a:.10g}
  aHat = {amin:.10g}
  inputRadialCoordinate = 1
  psiN_wish = {s:.10g}
/
&speciesParameters
  Zs = 1.0d+0 -1.0d+0
  mHats = 1.0d+0 5.446170214d-4
  nHats = {nhat:.10g} {nhat:.10g}
  THats = {that:.10g} {that:.10g}
  dNHatdrHats = {dn:.10g} {dn:.10g}
  dTHatdrHats = {dt:.10g} {dt:.10g}
/
&physicsParameters
  Delta = {delta:.10g}
  alpha = 1.0d+0
  nu_n = {nu}
  Er = {er}
  collisionOperator = {coll}
  includeXDotTerm = .true.
  includeElectricFieldTermInXiDot = .true.
/
&resolutionParameters
  Ntheta = {nt}
  Nzeta = {nz}
  Nxi = {nxi}
  NL = {nl}
  Nx = {nx}
  solverTolerance = {tol}
/
&otherNumericalParameters
  xGridScheme = 5
  Nxi_for_x_option = 0
/
&preconditionerOptions
/
"""


def _profiles(s: float) -> tuple[float, float, float, float]:
    """Reactor-core n and T and their radial derivatives at ``s`` [arXiv:2205.02914]."""
    ds_drhat = 2.0 * np.sqrt(s) / A_MINOR
    return (
        N_AXIS * (1.0 - s**5),
        T_AXIS * (1.0 - s),
        -5.0 * N_AXIS * s**4 * ds_drhat,
        -T_AXIS * ds_drhat,
    )


#: Boozer output mode numbers, in booz_xform ordering, fixed by MBOZ/NBOZ/NFP.
def _boozer_modes() -> tuple[np.ndarray, np.ndarray]:
    bm, bn = [], []
    for m in range(MBOZ):
        for n in (range(0, NBOZ + 1) if m == 0 else range(-NBOZ, NBOZ + 1)):
            bm.append(m)
            bn.append(n * NFP)
    return np.asarray(bm), np.asarray(bn)


BOOZ_XM, BOOZ_XN = _boozer_modes()


def _operator_template(s: float, psi_a: float):
    nhat, that, dn, dt = _profiles(s)
    text = _TEMPLATE.format(
        s=s, nfp=NFP, psi_a=psi_a, amin=A_MINOR, nhat=nhat, that=that, dn=dn, dt=dt,
        delta=DELTA, nu=NU_N, er=ER, coll=COLLISION_OPERATOR, nt=KIN[0], nz=KIN[1],
        nxi=KIN[2], nl=KIN[3], nx=KIN[4], tol=KIN_TOL,
    )  # fmt: skip
    return kinetic_operator_from_namelist(parse_sfincs_input_text(text))


def equilibrium(path: Path):
    """Converged VMEC state plus its runtime, for one boundary input file."""
    inp = VmecInput.from_file(str(path))
    inp = dataclasses.replace(
        inp, ns_array=np.asarray([NS]), ftol_array=np.asarray([VMEC_FTOL]),
        niter_array=np.asarray([VMEC_MAX_ITER]),
    )  # fmt: skip
    cfg = vmec_implicit.make_config(inp)
    params = vmec_implicit.params_from_input(inp)
    state = vmec_implicit.solve_implicit(params, cfg)
    psi_a = abs(float(inp.phiedge)) / (2.0 * np.pi)
    return state, vmec_implicit.runtime_from_params(params, cfg), psi_a


def boozer_at(state, rt, row: int):
    """Boozer spectrum on one half-mesh surface."""
    tabs = boozer_input_tables(state, rt, row)
    booz = booz_transform(
        rmnc=tabs["rmnc"][None, :], zmns=tabs["zmns"][None, :], lmns=tabs["lmns"][None, :],
        bmnc=tabs["bmnc"][None, :], bsubumnc=tabs["bsubumnc"][None, :],
        bsubvmnc=tabs["bsubvmnc"][None, :], iota=tabs["iota"][None],
        xm=tabs["xm"], xn=tabs["xn"], xm_nyq=tabs["xm"], xn_nyq=tabs["xn"],
        nfp=NFP, mboz=MBOZ, nboz=NBOZ, asym=False,
    )  # fmt: skip
    return tabs, booz


def bootstrap_at(state, rt, row: int, s: float, psi_a: float) -> float:
    """``<j.B>/sqrt(<B^2>)`` on one surface: a full tier-2 kinetic solve."""
    _tabs, booz = boozer_at(state, rt, row)
    template = _operator_template(s, psi_a)
    grids = make_grids(
        n_theta=template.n_theta, n_zeta=template.n_zeta, n_xi=template.n_xi,
        n_x=template.n_x, n_l=KIN[3], n_periods=NFP, x_grid_scheme=5,
    )  # fmt: skip
    op = ob.operator_with_boozer_geometry(
        template, bmnc=booz["bmnc_b"][0], m=jnp.asarray(BOOZ_XM),
        n=jnp.asarray(BOOZ_XN // NFP), nfp=NFP, iota=booz["iota_b"][0],
        g_hat=booz["bvco_b"][0], i_hat=booz["buco_b"][0],
        theta=grids.theta, zeta=grids.zeta,
        theta_weights=grids.theta_weights, zeta_weights=grids.zeta_weights,
    )  # fmt: skip
    mom, _ = ob.solve_and_moments(op, tol=KIN_TOL)
    return float(ob.bootstrap_current(mom))


def surface_geometry(state, rt, row: int, n_theta: int = 80, n_zeta: int = 160):
    """``(x, y, z, |B|)`` of one flux surface, for the 3D panel."""
    tabs = boozer_input_tables(state, rt, row)
    xm, xn = np.asarray(tabs["xm"]), np.asarray(tabs["xn"])
    rmnc, zmns, bmnc = (np.asarray(tabs[k]) for k in ("rmnc", "zmns", "bmnc"))
    theta = np.linspace(0.0, 2 * np.pi, n_theta)
    zeta = np.linspace(0.0, 2 * np.pi, n_zeta)
    th, ze = np.meshgrid(theta, zeta, indexing="ij")
    ang = xm[None, None, :] * th[:, :, None] - xn[None, None, :] * ze[:, :, None]
    r = np.einsum("m,tzm->tz", rmnc, np.cos(ang))
    z = np.einsum("m,tzm->tz", zmns, np.sin(ang))
    b = np.einsum("m,tzm->tz", bmnc, np.cos(ang))
    return r * np.cos(ze), r * np.sin(ze), z, b


def boozer_field(booz, n_theta: int = 96, n_zeta: int = 96):
    """``|B|(theta_B, zeta_B)`` over one field period."""
    xm, xn = BOOZ_XM, BOOZ_XN
    bmnc = np.asarray(booz["bmnc_b"][0])
    th = np.linspace(0.0, 2 * np.pi, n_theta)
    ze = np.linspace(0.0, 2 * np.pi / NFP, n_zeta)
    ang = th[:, None, None] * xm[None, None, :] - ze[None, :, None] * xn[None, None, :]
    return th, ze, np.einsum("m,tzm->tz", bmnc, np.cos(ang))


def main() -> int:
    if not OPTIMIZED.exists():
        raise SystemExit(
            f"{OPTIMIZED} not found -- run optimize_QH_bootstrap.py first; this "
            "script reloads its saved boundary rather than re-optimizing."
        )
    cases = {"seed (precise QH)": SEED, "optimized": OPTIMIZED}
    lcfs_row = NS - 1
    surfaces = np.linspace(0.15, 0.85, N_SURFACES)
    rows = [max(1, min(NS - 1, int(round(s * (NS - 1))))) for s in surfaces]

    data = {}
    for label, path in cases.items():
        t0 = time.perf_counter()
        print(f"[{label}] equilibrium from {path.name}")
        state, rt, psi_a = equilibrium(path)
        _tabs, booz = boozer_at(state, rt, lcfs_row)
        print(f"[{label}] bootstrap profile over {N_SURFACES} surfaces")
        profile = [
            bootstrap_at(state, rt, row, float(s), psi_a)
            for row, s in zip(rows, surfaces)
        ]
        data[label] = {
            "surface": surface_geometry(state, rt, lcfs_row),
            "boozer": boozer_field(booz),
            "profile": profile,
        }
        print(f"[{label}] done in {time.perf_counter() - t0:.0f} s")

    fig = plt.figure(figsize=(13.5, 8.4))
    labels = list(cases)
    bmax = max(float(np.max(data[k]["surface"][3])) for k in labels)
    bmin = min(float(np.min(data[k]["surface"][3])) for k in labels)

    for col, label in enumerate(labels):
        x, y, z, b = data[label]["surface"]
        ax = fig.add_subplot(2, 3, col + 1, projection="3d")
        norm = plt.Normalize(bmin, bmax)
        ax.plot_surface(
            x, y, z, facecolors=plt.cm.viridis(norm(b)), rstride=2, cstride=2,
            linewidth=0, antialiased=False, shade=False,
        )  # fmt: skip
        ax.set_title(f"boundary, |B| [T] ({label})", fontsize=10)
        ax.set_box_aspect((np.ptp(x), np.ptp(y), max(np.ptp(z), 1e-9)))
        ax.set_axis_off()

        th, ze, bmag = data[label]["boozer"]
        axb = fig.add_subplot(2, 3, col + 4)
        im = axb.contourf(ze, th, bmag, levels=26, cmap="viridis")
        axb.set_title(f"|B| on the LCFS, Boozer angles ({label})", fontsize=10)
        axb.set_xlabel(r"$\zeta_B$")
        axb.set_ylabel(r"$\theta_B$")
        fig.colorbar(im, ax=axb, fraction=0.046)

    axm = fig.add_subplot(2, 3, 3)
    for label, marker in zip(labels, ("o-", "s--")):
        axm.plot(surfaces, data[label]["profile"], marker, label=label)
    axm.axhline(0.0, color="0.6", lw=0.8)
    axm.set_xlabel("s (normalized toroidal flux)")
    axm.set_ylabel(r"$\langle j\cdot B\rangle/\sqrt{\langle B^2\rangle}$")
    axm.set_title("bootstrap-current profile", fontsize=10)
    axm.legend(fontsize=8)
    axm.grid(alpha=0.3)

    axt = fig.add_subplot(2, 3, 6)
    axt.axis("off")
    history = OUT_DIR / "optimize_QH_bootstrap_history.json"
    if history.exists():
        h = json.loads(history.read_text())
        rows_txt = [
            f"{'':<10}{'seed':>12}{'optimized':>12}",
            f"{'<j.B>':<10}{h['initial']['jbs']:>12.3e}{h['final']['jbs']:>12.3e}",
            f"{'QS':<10}{h['initial']['qs']:>12.3e}{h['final']['qs']:>12.3e}",
            f"{'aspect':<10}{h['initial']['aspect']:>12.3f}{h['final']['aspect']:>12.3f}",
            f"{'iota':<10}{h['initial']['iota']:>12.4f}{h['final']['iota']:>12.4f}",
        ]
        axt.text(0.0, 0.95, "\n".join(rows_txt), family="monospace", fontsize=9,
                 va="top", transform=axt.transAxes)  # fmt: skip

    fig.suptitle(
        "QH low bootstrap current: what the optimizer changed, and what it kept",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "qh_bootstrap_before_after.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)

    payload = {
        "surfaces": surfaces.tolist(),
        "profiles": {k: data[k]["profile"] for k in labels},
    }
    (OUT_DIR / "qh_bootstrap_before_after.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Saved plot: {out}")
    for label in labels:
        print(f"  {label:20s} <j.B> profile {[f'{v:+.3e}' for v in data[label]['profile']]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
