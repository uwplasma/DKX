"""Generate the README hero figure: a W7-X standard-configuration showcase.

Four panels in a 2x2 layout, all W7-X standard configuration.  Left column,
two stacked 3-D plasma boundaries reconstructed from the Boozer ``.bc``
equilibrium (``w7x_standardConfig.bc``, geometryScheme 11, the same file the
drift-kinetic solves use), drawn with the ``jet`` colormap:

``(a)`` the boundary colored by ``|B|``, evaluated from the full Boozer ``bmn``
spectrum of the surface nearest ``rho = 0.7``.

``(b)`` the boundary colored by the LOCAL parallel current density
``j_par(theta, zeta)`` — the genuine theta,zeta-resolved parallel current
``jHat = sum_s Z_s flow_s`` (:func:`dkx.moments.rhsmode1_moments`, the
``"jHat"`` table entry) from one two-species drift-kinetic solve on that
surface.  This field carries the true Pfirsch-Schlüter + bootstrap ``(m, n)``
structure: it is not proportional to ``1/|B|`` (its correlation with ``1/|B|``
on the solver grid is ``~0.1``).  Its flux-surface average ``<j_par B>`` is the
bootstrap current of panel (c).  Both toruses share the same Boozer geometry,
so ``|B|`` and ``j_par`` live on identical angle coordinates.

Right column, two line plots:

``(c)`` the bootstrap current profile ``<j.B>(rho)``: one two-species
(H+ + electron) drift-kinetic solve per flux surface at that surface's
ambipolar radial electric field, with the per-species ``Z_s FSABFlow_s``
contributions.  The species profiles, resolution, physics switches, and the
ambipolar ``E_r`` per surface come from the committed benchmark record
``docs/_static/figures/paper_benchmarks/w7x_ambipolar_er.json`` (the Pablant
2018 CERC discharge case, ``examples/paper_benchmarks/w7x_ambipolar_er.py``),
so panel (c) is the bootstrap-current output of exactly that validated setup.
The SI conversion ``<j.B> = FSABjHat * vBar * nBar * e`` (kA T m^-2 with
``vBar = 437695 m/s``, ``nBar = 1e20 m^-3``) is the one documented in
``docs/examples.rst``.  The local ``j_par`` of panel (b) uses the same
``vBar * nBar * e`` normalization without the extra ``B`` weighting, so it is
in ``kA m^-2``.

``(d)`` the ambipolar ``E_r(rho)`` with every root classified ion/electron,
replotted directly from the same committed JSON (no recompute) next to the
published neoclassical and XICS-measured references.

The kinetic solves are checkpointed to ``w7x_showcase.json`` next to the PNG
and skipped on re-runs (``DKX_SHOWCASE_FORCE=1`` recomputes): the per-surface
bootstrap solves under ``surfaces``, and the near-edge ``jHat(theta, zeta)``
field under ``edge_jhat``.  The PNG is palette-quantized to stay under budget.

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
from matplotlib.ticker import MaxNLocator

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "docs" / "_static" / "figures" / "readme"
PNG_PATH = OUT_DIR / "w7x_showcase.png"
JSON_PATH = OUT_DIR / "w7x_showcase.json"
AMBIPOLAR_JSON = (
    REPO_ROOT / "docs" / "_static" / "figures" / "paper_benchmarks" / "w7x_ambipolar_er.json"
)

FORCE = os.environ.get("DKX_SHOWCASE_FORCE") == "1"
SIZE_BUDGET_KB = 290.0
QUANTIZE_COLORS = 64

# --- Case (matches the committed w7x_ambipolar_er benchmark record) ---------
EQUILIBRIUM = "w7x_standardConfig.bc"  # geometryScheme = 11 Boozer spectrum
SOLVER = "block_tridiagonal"  # tier-1 direct block-Thomas solve
N_PERIODS = 5  # W7-X field periods
# SI conversion, documented in docs/examples.rst: <j.B> = FSABjHat * vBar *
# nBar * e with vBar = 437695 m/s (TBar = 1 keV, proton mBar), nBar = 1e20
# m^-3.  In kA T m^-2 (bootstrap) or kA m^-2 (local jHat, no B weighting):
JBOOT_KA_PER_HAT = 437695.0 * 1e20 * 1.602176634e-19 / 1e3

# --- 3-D boundary panel -----------------------------------------------------
# The fine toroidal striping on the inboard side is genuine |B| structure of
# the W7-X boundary (high-n harmonics of the Boozer spectrum), not a mesh
# artifact.  ``nsign = -1`` is the toroidal-mode sign convention that makes the
# reconstructed |B| match the drift-kinetic operator's b_hat (verified).
N_THETA_3D = 200
N_ZETA_3D = 561
VIEW_ELEV = 31.0
VIEW_AZIM = -58.0
CMAP_3D = "jet"
BOX_ZOOM = 1.25  # fit the full W7-X torus width with margin, no side clipping

# --- Left-column 3-D panel geometry (figure fraction) -----------------------
# Oversized axes with a title just above and a vertical colorbar snug to the
# right of each torus; the toruses fill most of the left half of the canvas.
TORUS_TOP_RECT = (-0.035, 0.490, 0.500, 0.410)
TORUS_TOP_TITLE_XY = (0.230, 0.918)
TORUS_TOP_CBAR_RECT = (0.452, 0.545, 0.016, 0.300)
TORUS_BOT_RECT = (-0.035, 0.030, 0.500, 0.410)
TORUS_BOT_TITLE_XY = (0.230, 0.458)
TORUS_BOT_CBAR_RECT = (0.452, 0.085, 0.016, 0.300)

# --- Fonts (README-legible) -------------------------------------------------
FS_SUPTITLE = 19.0
FS_PANEL_TITLE = 15.0
FS_AXIS_LABEL = 15.0
FS_TICK = 12.5
FS_CBAR_LABEL = 15.0
FS_CBAR_TICK = 12.0
FS_LEGEND = 12.5

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
    ax.tick_params(colors=INK_2, labelsize=FS_TICK)
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


def _solve_edge_surface(record: dict):
    """One RHSMode=1 solve on the near-edge surface; returns ``(run, edge)``."""
    import jax

    jax.config.update("jax_enable_x64", True)
    from dkx.run import run_profile

    edge = record["surfaces"][-1]
    rho = float(edge["rho"])
    er = float(edge["selected"]["er"])
    deck = _build_deck(rho, edge["profile"], er, edge["resolution"])
    deck_path = OUT_DIR / "_w7x_showcase_deck.namelist"
    deck_path.write_text(deck)
    try:
        run = run_profile(deck_path, solve_method=SOLVER, emit=lambda _s: None)
    finally:
        deck_path.unlink(missing_ok=True)
    return run, edge


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


def compute_edge_jhat(record: dict) -> dict:
    """One kinetic solve; checkpoint the local ``jHat(theta, zeta)`` field.

    Extracts the genuine theta,zeta-resolved parallel current
    ``jHat = sum_s Z_s flow_s`` (``run.moments["jHat"]``), records its
    ``1/|B|`` decorrelation, and stores it for the panel-(b) render.
    """
    t0 = time.time()
    run, edge = _solve_edge_surface(record)
    seconds = time.time() - t0

    jhat = np.asarray(run.moments["jHat"], dtype=np.float64)  # (T, Z)
    b_hat = np.asarray(run.operator.b_hat, dtype=np.float64)  # (T, Z)
    fsab_jhat = float(np.asarray(run.moments["FSABjHat"]).ravel()[0])

    inv_b = (1.0 / b_hat).ravel()
    jflat = jhat.ravel()
    jc = jflat - jflat.mean()
    ic = inv_b - inv_b.mean()
    corr = float((jc * ic).sum() / np.sqrt((jc * jc).sum() * (ic * ic).sum()))
    print(
        f"  edge jHat solve: rho={float(edge['rho']):.2f}  "
        f"FSABjHat={fsab_jhat:+.6e}  corr(jHat,1/|B|)={corr:+.3f}  [{seconds:.1f} s]"
    )
    return dict(
        rho=float(edge["rho"]),
        er=float(edge["selected"]["er"]),
        root_type=str(edge["selected"]["root_type"]),
        n_periods=N_PERIODS,
        n_theta=int(jhat.shape[0]),
        n_zeta=int(jhat.shape[1]),
        nsign=-1,
        fsab_jhat=fsab_jhat,
        corr_jhat_invB=corr,
        jhat_tz=jhat.tolist(),
        seconds=seconds,
    )


def _nearest_bc_surface(rho: float):
    """The Boozer ``.bc`` surface nearest ``rho`` (v3 snap-to-nearest selection)."""
    from dkx.magnetic_geometry import _bracketing_surfaces, read_boozer_bc
    from dkx.paths import resolve_existing_path

    path = resolve_existing_path(EQUILIBRIUM).path
    _header, surfaces = read_boozer_bc(path, geometry_scheme=11)
    old, new = _bracketing_surfaces(surfaces, float(rho))
    return old if abs(old.r_n - rho) < abs(new.r_n - rho) else new


def _eval_boozer_field(
    amp: np.ndarray,
    m: np.ndarray,
    n: np.ndarray,
    theta: np.ndarray,
    zeta: np.ndarray,
    *,
    cosine: bool,
    base: float = 0.0,
    chunk: int = 320,
) -> np.ndarray:
    """``base + sum_h amp_h {cos|sin}(m_h theta - NPeriods n_h zeta)``.

    ``nsign = -1`` (folded into ``n`` by the caller) matches the operator's
    b_hat.  Chunked over harmonics so the full Boozer spectrum (no truncation)
    fits in memory on the dense render grid.
    """
    out = np.full((theta.shape[0], zeta.shape[0]), float(base), dtype=np.float64)
    th = theta[None, :, None]
    ze = zeta[None, None, :]
    for i in range(0, m.shape[0], chunk):
        mm = m[i : i + chunk].astype(np.float64)[:, None, None]
        nn = n[i : i + chunk].astype(np.float64)[:, None, None]
        aa = amp[i : i + chunk].astype(np.float64)
        angle = mm * th - float(N_PERIODS) * nn * ze
        basis = np.cos(angle) if cosine else np.sin(angle)
        out = out + np.tensordot(aa, basis, axes=(0, 0))
    return out


def _boundary_from_boozer(rho: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(X, Y, Z, |B|) on the W7-X boundary from the Boozer ``.bc`` surface near ``rho``.

    Real-space geometry (stellarator-symmetric: R cosine, Z sine, and the
    Boozer->cylindrical toroidal-angle correction sine) and ``|B|`` are
    reconstructed on a full-torus grid from the same surface, with the toroidal
    mode sign (``nsign = -1``) that matches the drift-kinetic operator.
    """
    surf = _nearest_bc_surface(rho)
    m = np.asarray(surf.m, dtype=np.float64)
    n = -np.asarray(surf.n, dtype=np.float64)  # nsign = -1, folded into n

    theta = np.linspace(0.0, 2.0 * np.pi, N_THETA_3D)
    zeta = np.linspace(0.0, 2.0 * np.pi, N_ZETA_3D)  # full-torus Boozer angle
    rr = _eval_boozer_field(surf.r_amp, m, n, theta, zeta, cosine=True, base=float(surf.r0))
    zz = _eval_boozer_field(surf.z_amp, m, n, theta, zeta, cosine=False)
    pcorr = _eval_boozer_field(surf.dz_amp, m, n, theta, zeta, cosine=False)
    bmag = _eval_boozer_field(surf.b_amp, m, n, theta, zeta, cosine=True, base=float(surf.b0_over_bbar))
    phi_cyl = zeta[None, :] - (2.0 * np.pi / float(N_PERIODS)) * pcorr
    return rr * np.cos(phi_cyl), rr * np.sin(phi_cyl), zz, bmag


def _jpar_on_boundary(edge_jhat: dict) -> np.ndarray:
    """Local ``j_par(theta, zeta)`` [kA m^-2] on the full-torus render grid.

    Fourier-interpolates the checkpointed ``jHat(theta, zeta)`` (solver grid,
    one field period in zeta) onto the (N_THETA_3D, N_ZETA_3D) full-torus grid
    and converts to ``kA m^-2`` with the panel-(c) normalization.
    """
    jhat = np.asarray(edge_jhat["jhat_tz"], dtype=np.float64)  # (Ts, Zs)
    n_ts, n_zs = jhat.shape
    coeff = np.fft.fft2(jhat) / (n_ts * n_zs)
    k_th = np.fft.fftfreq(n_ts, d=1.0) * n_ts
    k_ze = np.fft.fftfreq(n_zs, d=1.0) * n_zs
    theta = np.linspace(0.0, 2.0 * np.pi, N_THETA_3D)
    zeta = np.linspace(0.0, 2.0 * np.pi, N_ZETA_3D)
    e_th = np.exp(1j * k_th[None, :] * theta[:, None])  # (T, Ts)
    e_ze = np.exp(1j * k_ze[None, :] * float(N_PERIODS) * zeta[:, None])  # (Z, Zs)
    jhat_dense = (e_th @ coeff @ e_ze.T).real
    return jhat_dense * JBOOT_KA_PER_HAT


def _add_torus(
    fig: plt.Figure,
    rect: tuple[float, float, float, float],
    xx: np.ndarray,
    yy: np.ndarray,
    zz: np.ndarray,
    field: np.ndarray,
    title: str,
    title_xy: tuple[float, float],
) -> plt.Normalize:
    """Draw one W7-X boundary torus colored by ``field``; return its color norm."""
    ax = fig.add_axes(rect, projection="3d")
    norm = plt.Normalize(float(field.min()), float(field.max()))
    cmap = plt.get_cmap(CMAP_3D)
    ax.plot_surface(
        xx,
        yy,
        zz,
        facecolors=cmap(norm(field)),
        rcount=N_THETA_3D,
        ccount=N_ZETA_3D,
        linewidth=0,
        antialiased=False,
        shade=False,
    )
    # Tight limits + zoom so the flat W7-X torus nearly fills its 3-D axes.
    ax.set_xlim(float(xx.min()), float(xx.max()))
    ax.set_ylim(float(yy.min()), float(yy.max()))
    ax.set_zlim(float(zz.min()), float(zz.max()))
    ax.set_box_aspect((np.ptp(xx), np.ptp(yy), np.ptp(zz)), zoom=BOX_ZOOM)
    ax.set_axis_off()
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
    fig.text(title_xy[0], title_xy[1], title, fontsize=FS_PANEL_TITLE, color=INK, ha="center")
    return norm


def _add_torus_colorbar(
    fig: plt.Figure,
    norm: plt.Normalize,
    cbar_label: str,
    cbar_rect: tuple[float, float, float, float],
) -> None:
    """Vertical colorbar snug to the right of a torus panel."""
    mappable = plt.cm.ScalarMappable(cmap=plt.get_cmap(CMAP_3D), norm=norm)
    cax = fig.add_axes(list(cbar_rect))
    cbar = fig.colorbar(mappable, cax=cax, orientation="vertical")
    cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    cbar.set_label(cbar_label, fontsize=FS_CBAR_LABEL, color=INK_2)
    cbar.ax.tick_params(labelsize=FS_CBAR_TICK, colors=INK_2)
    cbar.outline.set_edgecolor(GRID)


def _panel_bootstrap(ax: plt.Axes, showcase: dict) -> None:
    rows = showcase["surfaces"]
    rho = np.array([row["rho"] for row in rows])
    jboot = np.array([row["fsab_jhat"] for row in rows]) * JBOOT_KA_PER_HAT
    ion = np.array([row["ion_z_fsab_flow"] for row in rows]) * JBOOT_KA_PER_HAT
    electron = np.array([row["electron_z_fsab_flow"] for row in rows]) * JBOOT_KA_PER_HAT
    ax.axhline(0.0, color=INK_2, linewidth=0.8, alpha=0.6)
    ax.plot(rho, ion, "s--", color=ORANGE, linewidth=1.4, markersize=5, label="ion $Z_i\\,\\langle B\\,V_{\\|i}\\rangle$")
    ax.plot(rho, electron, "o--", color=GREEN, linewidth=1.4, markersize=5, label="electron $Z_e\\,\\langle B\\,V_{\\|e}\\rangle$")
    ax.plot(rho, jboot, "o-", color=BLUE, linewidth=2.6, markersize=7, label="total $\\langle\\, j_{\\|}\\, B\\,\\rangle$")
    ax.set_ylim(float(jboot.min()) - 22.0, max(float(ion.max()), 0.0) + 6.0)
    ax.set_xlabel("$\\rho$ (normalized minor radius)", fontsize=FS_AXIS_LABEL)
    ax.set_ylabel("$\\langle\\, j_{\\|}\\, B\\,\\rangle$  [kA T m$^{-2}$]", fontsize=FS_AXIS_LABEL)
    ax.set_title(
        "(c) bootstrap current at the ambipolar $E_r$",
        fontsize=FS_PANEL_TITLE,
        color=INK,
    )
    ax.legend(fontsize=FS_LEGEND, frameon=False, loc="lower center", ncols=3)
    _style_axes(ax)


def _panel_ambipolar_er(ax: plt.Axes, record: dict) -> None:
    ref = record["reference_er"]
    ax.axhline(0.0, color=INK_2, linewidth=0.8, alpha=0.6)
    ax.plot(
        ref["neoclassical"]["rho"],
        ref["neoclassical"]["er_kVm"],
        "--",
        color=INK_2,
        linewidth=1.4,
        label="published neoclassical",
    )
    ax.plot(
        ref["xics_measured"]["rho"],
        ref["xics_measured"]["er_kVm"],
        "-",
        color=ORANGE,
        linewidth=1.6,
        alpha=0.9,
        label="XICS measurement",
    )
    comparison = record["reference_comparison"]
    rho = np.array([row["rho"] for row in comparison])
    er = np.array([row["dkx_er"] for row in comparison])
    kind = [row["dkx_root_type"] for row in comparison]
    ax.plot(rho, er, "-", color=BLUE, linewidth=1.8, alpha=0.8)
    is_electron = np.array([k == "electron" for k in kind])
    ax.plot(
        rho[is_electron],
        er[is_electron],
        "s",
        color=BLUE,
        markersize=8,
        label="dkx electron root",
    )
    ax.plot(
        rho[~is_electron],
        er[~is_electron],
        "o",
        markerfacecolor="white",
        markeredgecolor=BLUE,
        markersize=8,
        label="dkx ion root",
    )
    ax.set_xlabel("$\\rho$ (normalized minor radius)", fontsize=FS_AXIS_LABEL)
    ax.set_ylabel("$E_r$  [kV/m]", fontsize=FS_AXIS_LABEL)
    ax.set_title(
        "(d) ambipolar $E_r$: electron-root core to ion-root edge",
        fontsize=FS_PANEL_TITLE,
        color=INK,
    )
    ax.legend(fontsize=FS_LEGEND, frameon=False, loc="lower left", ncols=2)
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

    showcase: dict
    if JSON_PATH.exists() and not FORCE:
        showcase = json.loads(JSON_PATH.read_text())
        print(f"reusing checkpointed solves from {JSON_PATH.relative_to(REPO_ROOT)}")
    else:
        print("bootstrap-current solves (one per surface, ambipolar E_r):")
        showcase = compute_bootstrap_profile(record)
        JSON_PATH.write_text(json.dumps(showcase, indent=1) + "\n")

    if "edge_jhat" not in showcase or FORCE:
        print("near-edge jHat(theta,zeta) solve:")
        showcase["edge_jhat"] = compute_edge_jhat(record)
        JSON_PATH.write_text(json.dumps(showcase, indent=1) + "\n")
    else:
        print("reusing checkpointed edge jHat(theta,zeta) field")

    plt.rcParams.update({"font.size": 13.0, "axes.titlesize": FS_PANEL_TITLE})
    fig = plt.figure(figsize=(14.2, 8.6), dpi=120)

    # Left column: two stacked Boozer boundaries (|B| over local j_par), jet.
    edge_rho = float(showcase["edge_jhat"]["rho"])
    xx, yy, zz, bmag = _boundary_from_boozer(edge_rho)
    jpar = _jpar_on_boundary(showcase["edge_jhat"])
    norm_b = _add_torus(
        fig,
        TORUS_TOP_RECT,
        xx,
        yy,
        zz,
        bmag,
        "(a) plasma boundary, colored by $|B|$",
        TORUS_TOP_TITLE_XY,
    )
    norm_j = _add_torus(
        fig,
        TORUS_BOT_RECT,
        xx,
        yy,
        zz,
        jpar,
        "(b) local parallel current $j_{\\|}(\\theta,\\zeta)$",
        TORUS_BOT_TITLE_XY,
    )
    # Colorbars after both toruses so neither 3-D axes overpaints their labels.
    _add_torus_colorbar(fig, norm_b, "$|B|$ [T]", TORUS_TOP_CBAR_RECT)
    _add_torus_colorbar(fig, norm_j, "$j_{\\|}$ [kA m$^{-2}$]", TORUS_BOT_CBAR_RECT)

    # Right column: bootstrap profile over ambipolar E_r (unchanged content).
    gs_r = fig.add_gridspec(
        2,
        1,
        left=0.600,
        right=0.980,
        top=0.895,
        bottom=0.090,
        hspace=0.34,
    )
    ax_boot = fig.add_subplot(gs_r[0])
    _panel_bootstrap(ax_boot, showcase)
    ax_er = fig.add_subplot(gs_r[1])
    _panel_ambipolar_er(ax_er, record)
    fig.suptitle(
        "DKX on the W7-X standard configuration: drift-kinetic neoclassical transport, "
        "differentiable end to end",
        fontsize=FS_SUPTITLE,
        color=INK,
        y=0.970,
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
