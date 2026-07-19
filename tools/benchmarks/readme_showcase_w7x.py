"""Generate the README hero figure: a W7-X standard-configuration showcase.

Three panels, all W7-X standard configuration:

``(a)`` the 3-D plasma boundary colored by ``|B|``, evaluated from the Fourier
surface of ``wout_w7x_standardConfig.nc`` (resolved through
:func:`dkx.paths.resolve_existing_path`, i.e. ``DKX_EQUILIBRIA_DIRS`` or the
fetched data cache).

``(b)`` the bootstrap current profile ``<j.B>(rho)``: one two-species
(H+ + electron) drift-kinetic solve per flux surface at that surface's
ambipolar radial electric field, with the per-species ``Z_s FSABFlow_s``
contributions.  The species profiles, resolution, physics switches, and the
ambipolar ``E_r`` per surface come from the committed benchmark record
``docs/_static/figures/paper_benchmarks/w7x_ambipolar_er.json`` (the Pablant
2018 CERC discharge case, ``examples/paper_benchmarks/w7x_ambipolar_er.py``),
so panel (b) is the bootstrap-current output of exactly that validated setup.
The SI conversion ``<j.B> = FSABjHat * vBar * nBar * e`` (kA T m^-2 with
``vBar = 437695 m/s``, ``nBar = 1e20 m^-3``) is the one documented in
``docs/examples.rst``.

``(c)`` the ambipolar ``E_r(rho)`` with every root classified ion/electron,
replotted directly from the same committed JSON (no recompute) next to the
published neoclassical and XICS-measured references.

The kinetic solves in (b) are checkpointed to ``w7x_showcase.json`` next to
the PNG and skipped on re-runs (``DKX_SHOWCASE_FORCE=1`` recomputes).  The PNG
is palette-quantized to stay under the size budget.

Run (from the repo root, with the equilibrium search path set):
  DKX_EQUILIBRIA_DIRS=/path/to/equilibria \
  python tools/benchmarks/readme_showcase_w7x.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "docs" / "_static" / "figures" / "readme"
PNG_PATH = OUT_DIR / "w7x_showcase.png"
JSON_PATH = OUT_DIR / "w7x_showcase.json"
AMBIPOLAR_JSON = (
    REPO_ROOT / "docs" / "_static" / "figures" / "paper_benchmarks" / "w7x_ambipolar_er.json"
)

FORCE = os.environ.get("DKX_SHOWCASE_FORCE") == "1"
SIZE_BUDGET_KB = 200.0
QUANTIZE_COLORS = 64

# --- Case (matches the committed w7x_ambipolar_er benchmark record) ---------
EQUILIBRIUM = "w7x_standardConfig.bc"  # geometryScheme = 11 Boozer spectrum
WOUT_FILE = "wout_w7x_standardConfig.nc"  # VMEC surface for the 3-D panel
SOLVER = "block_tridiagonal"  # tier-1 direct block-Thomas solve
# SI conversion for the bootstrap current, documented in docs/examples.rst:
# <j.B> = FSABjHat * vBar * nBar * e  with vBar = 437695 m/s (TBar = 1 keV,
# proton mBar), nBar = 1e20 m^-3.  In kA T m^-2:
JBOOT_KA_PER_HAT = 437695.0 * 1e20 * 1.602176634e-19 / 1e3

# --- 3-D boundary panel -----------------------------------------------------
# The fine toroidal striping on the inboard side is genuine |B| structure of
# the W7-X boundary (high-n harmonics of the Nyquist spectrum), not a mesh
# artifact; it persists on interior surfaces.
N_THETA_3D = 240
N_ZETA_3D = 601
VIEW_ELEV = 22.0
VIEW_AZIM = -56.0
CMAP_3D = "plasma"

# --- Palette (shared with readme_figures.py; color follows the code) --------
BLUE = "#2a78d6"  # dkx
ORANGE = "#eb6834"  # references / ion species
GREEN = "#2e9e6b"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"


def _style_axes(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8.5)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.7)


def _build_deck(rho: float, profile: dict, er: float, resolution: dict) -> str:
    """Two-species RHSMode=1 deck: the w7x_ambipolar_er deck at fixed E_r."""
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
  nHats = {profile["nHat"]} {profile["nHat"]}
  THats = {profile["ti"]} {profile["te"]}
  dNHatdrHats = {profile["dnHat_drhat"]} {profile["dnHat_drhat"]}
  dTHatdrHats = {profile["dti_drhat"]} {profile["dte_drhat"]}
/
&physicsParameters
  Delta = 4.5694d-3
  alpha = 1.0
  nu_n = 8.330d-3
  Er = {er}
  collisionOperator = 1
  includeXDotTerm = .false.
  includeElectricFieldTermInXiDot = .false.
  useDKESExBDrift = .true.
  includePhi1 = .false.
/
&resolutionParameters
  Ntheta = {resolution["n_theta"]}
  Nzeta = {resolution["n_zeta"]}
  Nxi = {resolution["n_xi"]}
  NL = 4
  Nx = {resolution["n_x"]}
  solverTolerance = 1d-9
/
&otherNumericalParameters
  Nxi_for_x_option = 0
/
"""


def compute_bootstrap_profile(record: dict) -> dict:
    """One kinetic solve per surface at its ambipolar E_r; returns <j.B> data."""
    import jax

    jax.config.update("jax_enable_x64", True)
    from dkx.run import run_profile

    rows = []
    for surface in record["surfaces"]:
        rho = float(surface["rho"])
        er = float(surface["selected"]["er"])
        root_type = str(surface["selected"]["root_type"])
        deck = _build_deck(rho, surface["profile"], er, surface["resolution"])
        deck_path = OUT_DIR / "_w7x_showcase_deck.namelist"
        deck_path.write_text(deck)
        t0 = time.time()
        run = run_profile(deck_path, solve_method=SOLVER, emit=lambda _s: None)
        seconds = time.time() - t0
        deck_path.unlink()
        fsab_flow = np.asarray(run.moments["FSABFlow"]).ravel()
        zs = np.array([1.0, -1.0])
        row = dict(
            rho=rho,
            er=er,
            root_type=root_type,
            fsab_jhat=float(np.asarray(run.moments["FSABjHat"]).ravel()[0]),
            ion_z_fsab_flow=float(zs[0] * fsab_flow[0]),
            electron_z_fsab_flow=float(zs[1] * fsab_flow[1]),
            seconds=seconds,
        )
        rows.append(row)
        print(
            f"  rho={rho:.2f}  Er={er:+7.2f} kV/m ({root_type} root)  "
            f"FSABjHat={row['fsab_jhat']:+.6e}  [{seconds:.1f} s]"
        )
    return dict(
        case="W7-X standard configuration, CERC profiles (Pablant 2018)",
        source_record=str(AMBIPOLAR_JSON.relative_to(REPO_ROOT)),
        model=(
            "two-species H+ + electron, pitch-angle scattering, DKES ExB, "
            f"RHSMode=1, solver={SOLVER}, one solve per surface at the "
            "ambipolar E_r of the committed benchmark record"
        ),
        jboot_ka_per_hat=JBOOT_KA_PER_HAT,
        surfaces=rows,
    )


def _boundary_from_wout() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(X, Y, Z, |B|) on the outermost VMEC surface of the wout file."""
    from netCDF4 import Dataset

    from dkx.paths import resolve_existing_path

    wout = resolve_existing_path(WOUT_FILE).path
    with Dataset(wout) as data:
        xm = np.asarray(data["xm"][:], dtype=float)
        xn = np.asarray(data["xn"][:], dtype=float)
        rmnc = np.asarray(data["rmnc"][-1], dtype=float)
        zmns = np.asarray(data["zmns"][-1], dtype=float)
        xm_nyq = np.asarray(data["xm_nyq"][:], dtype=float)
        xn_nyq = np.asarray(data["xn_nyq"][:], dtype=float)
        bmnc = np.asarray(data["bmnc"][-1], dtype=float)

    theta = np.linspace(0.0, 2.0 * np.pi, N_THETA_3D)
    zeta = np.linspace(0.0, 2.0 * np.pi, N_ZETA_3D)
    th, ze = np.meshgrid(theta, zeta, indexing="ij")
    angle = xm[:, None, None] * th[None] - xn[:, None, None] * ze[None]
    rr = np.tensordot(rmnc, np.cos(angle), axes=(0, 0))
    zz = np.tensordot(zmns, np.sin(angle), axes=(0, 0))
    angle_nyq = xm_nyq[:, None, None] * th[None] - xn_nyq[:, None, None] * ze[None]
    bmag = np.abs(np.tensordot(bmnc, np.cos(angle_nyq), axes=(0, 0)))
    return rr * np.cos(ze), rr * np.sin(ze), zz, bmag


def _panel_boundary(fig: plt.Figure, gs) -> None:
    xx, yy, zz, bmag = _boundary_from_wout()
    ax = fig.add_subplot(gs, projection="3d")
    norm = plt.Normalize(float(bmag.min()), float(bmag.max()))
    cmap = plt.get_cmap(CMAP_3D)
    ax.plot_surface(
        xx,
        yy,
        zz,
        facecolors=cmap(norm(bmag)),
        rcount=N_THETA_3D,
        ccount=N_ZETA_3D,
        linewidth=0,
        antialiased=False,
        shade=False,
    )
    ax.set_box_aspect((np.ptp(xx), np.ptp(yy), np.ptp(zz)))
    ax.set_axis_off()
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
    box = ax.get_position()
    ax.set_position(
        [
            box.x0 - 0.36 * box.width,
            box.y0 - 0.26 * box.height,
            1.72 * box.width,
            1.66 * box.height,
        ]
    )
    fig.text(
        box.x0 + 0.42 * box.width,
        0.895,
        "(a) plasma boundary, colored by $|B|$",
        fontsize=10.5,
        color=INK,
        ha="center",
    )
    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cax = fig.add_axes([box.x0 + 0.13 * box.width, 0.075, 0.60 * box.width, 0.022])
    cbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    cbar.set_label("$|B|$ [T]", fontsize=9, color=INK_2)
    cbar.ax.tick_params(labelsize=8, colors=INK_2)
    cbar.outline.set_edgecolor(GRID)


def _panel_bootstrap(ax: plt.Axes, showcase: dict) -> None:
    rows = showcase["surfaces"]
    rho = np.array([row["rho"] for row in rows])
    jboot = np.array([row["fsab_jhat"] for row in rows]) * JBOOT_KA_PER_HAT
    ion = np.array([row["ion_z_fsab_flow"] for row in rows]) * JBOOT_KA_PER_HAT
    electron = np.array([row["electron_z_fsab_flow"] for row in rows]) * JBOOT_KA_PER_HAT
    ax.axhline(0.0, color=INK_2, linewidth=0.8, alpha=0.6)
    ax.plot(rho, ion, "s--", color=ORANGE, linewidth=1.2, markersize=4, label="ion $Z_i\\,\\langle B\\,V_{\\|i}\\rangle$")
    ax.plot(rho, electron, "o--", color=GREEN, linewidth=1.2, markersize=4, label="electron $Z_e\\,\\langle B\\,V_{\\|e}\\rangle$")
    ax.plot(rho, jboot, "o-", color=BLUE, linewidth=2.4, markersize=6, label="total $\\langle\\, j_{\\|}\\, B\\,\\rangle$")
    ax.set_ylim(float(jboot.min()) - 22.0, max(float(ion.max()), 0.0) + 6.0)
    ax.set_xlabel("$\\rho$ (normalized minor radius)", fontsize=9.5)
    ax.set_ylabel("$\\langle\\, j_{\\|}\\, B\\,\\rangle$  [kA T m$^{-2}$]", fontsize=9.5)
    ax.set_title(
        "(b) bootstrap current, one kinetic solve per surface at the ambipolar $E_r$",
        fontsize=10.5,
        color=INK,
    )
    ax.legend(fontsize=8, frameon=False, loc="lower center", ncols=3)
    _style_axes(ax)


def _panel_ambipolar_er(ax: plt.Axes, record: dict) -> None:
    ref = record["reference_er"]
    ax.axhline(0.0, color=INK_2, linewidth=0.8, alpha=0.6)
    ax.plot(
        ref["neoclassical"]["rho"],
        ref["neoclassical"]["er_kVm"],
        "--",
        color=INK_2,
        linewidth=1.2,
        label="published neoclassical",
    )
    ax.plot(
        ref["xics_measured"]["rho"],
        ref["xics_measured"]["er_kVm"],
        "-",
        color=ORANGE,
        linewidth=1.4,
        alpha=0.9,
        label="XICS measurement",
    )
    comparison = record["reference_comparison"]
    rho = np.array([row["rho"] for row in comparison])
    er = np.array([row["dkx_er"] for row in comparison])
    kind = [row["dkx_root_type"] for row in comparison]
    ax.plot(rho, er, "-", color=BLUE, linewidth=1.6, alpha=0.8)
    is_electron = np.array([k == "electron" for k in kind])
    ax.plot(
        rho[is_electron],
        er[is_electron],
        "s",
        color=BLUE,
        markersize=7,
        label="dkx electron root",
    )
    ax.plot(
        rho[~is_electron],
        er[~is_electron],
        "o",
        markerfacecolor="white",
        markeredgecolor=BLUE,
        markersize=7,
        label="dkx ion root",
    )
    ax.set_xlabel("$\\rho$ (normalized minor radius)", fontsize=9.5)
    ax.set_ylabel("$E_r$  [kV/m]", fontsize=9.5)
    ax.set_title(
        "(c) ambipolar $E_r$: electron-root core, crossover near $\\rho \\sim 0.6$",
        fontsize=10.5,
        color=INK,
    )
    ax.legend(fontsize=8, frameon=False, loc="lower left", ncols=2)
    _style_axes(ax)


def _quantize(path: Path) -> None:
    from PIL import Image

    with Image.open(path) as image:
        image.convert("RGB").convert(
            "P", palette=Image.Palette.ADAPTIVE, colors=QUANTIZE_COLORS
        ).save(path, optimize=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    record = json.loads(AMBIPOLAR_JSON.read_text())

    if JSON_PATH.exists() and not FORCE:
        showcase = json.loads(JSON_PATH.read_text())
        print(f"reusing checkpointed solves from {JSON_PATH.relative_to(REPO_ROOT)}")
    else:
        print("bootstrap-current solves (one per surface, ambipolar E_r):")
        showcase = compute_bootstrap_profile(record)
        JSON_PATH.write_text(json.dumps(showcase, indent=1) + "\n")

    fig = plt.figure(figsize=(15.0, 7.2), dpi=120)
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.08, 1.0),
        left=0.035,
        right=0.985,
        top=0.865,
        bottom=0.085,
        hspace=0.42,
        wspace=0.14,
    )
    _panel_boundary(fig, gs[:, 0])
    ax_boot = fig.add_subplot(gs[0, 1])
    _panel_bootstrap(ax_boot, showcase)
    ax_er = fig.add_subplot(gs[1, 1])
    _panel_ambipolar_er(ax_er, record)
    fig.suptitle(
        "DKX on the W7-X standard configuration: drift-kinetic neoclassical transport, "
        "differentiable end to end",
        fontsize=13,
        color=INK,
        y=0.965,
    )
    fig.savefig(PNG_PATH, dpi=120)
    plt.close(fig)

    _quantize(PNG_PATH)
    size_kb = PNG_PATH.stat().st_size / 1024.0
    print(f"wrote {PNG_PATH.relative_to(REPO_ROOT)} ({size_kb:.1f} kB)")
    if size_kb > SIZE_BUDGET_KB:
        raise SystemExit(f"figure exceeds the {SIZE_BUDGET_KB:.0f} kB budget")


if __name__ == "__main__":
    main()
