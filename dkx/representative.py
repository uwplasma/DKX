"""One-command representative run: an equilibrium in, publication panels out.

``dkx wout_XXX.nc`` solves the small set of cases a neoclassical paper actually
reports and renders them as one figure, in tens of seconds rather than hours.
``dkx --plot FILE`` renders the same panels from an existing output file --- and
because :mod:`dkx.writer` emits ``sfincsOutput.h5`` in SFINCS's own layout, that
works on Fortran SFINCS output too, with no separate reader.

Panels
------
* **monoenergetic** ``D11*``, ``D31*``, ``D33*`` against ``nuPrime``, one curve
  per ``EStar``.  The standard cross-code benchmark figure.
* **bootstrap** ``<j.B>/sqrt(<B^2>)``, the parallel-momentum moment.
* **fluxes** particle and heat flux per species.
* **|B|** on the flux surface, for context on what device produced the numbers.

Default resolution
------------------
``Ntheta=25, Nzeta=41, Nxi=20`` (20,500 DOF), chosen from a measured convergence
scan against a ``41x71x80`` reference rather than by judgement:

===========  =====================  ===================
axis         error at the low end   at the high end
===========  =====================  ===================
``Nzeta``    3.59e-02 (15)          1.55e-05 (61)
``Ntheta``   1.03e-02 (11)          4.00e-04 (31)
``Nxi``      1.10e-07 (12)          5.59e-10 (64)
===========  =====================  ===================

``Nxi`` is converged to 1e-07 at the *lowest* value tested, so spending
resolution there buys nothing; ``Nzeta`` is the expensive axis.  That is the
opposite of the drift-dominated decks, where ``Nxi`` binds hardest, and it is
why the default is measured rather than assumed.  At the default the transport
matrix is within 2.3e-04 of the reference, a solve costs 1.85 s cold and 1.11 s
warm, and a 5x3 scan fits in about 19 s.

Monoenergetic runs take ``nuPrime``/``EStar``, **not** ``nu_n``/``Er`` --- the
upstream decks say so in a comment, and varying ``nu_n`` here changes nothing at
all while looking like it worked.
"""

from __future__ import annotations

import dataclasses
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

#: Convergence-tested default, see the module docstring.
#: ``n_xi`` must be at least ``n_zeta``: at low collisionality the pitch-angle
#: resolution is what limits the answer, and the convergence scan that set the
#: earlier 25/41/20 was run at a single mid collisionality, where it does not
#: show.  25/25/41 on the collaborator's advice from a full-range test.
DEFAULT_RESOLUTION = {"n_theta": 25, "n_zeta": 25, "n_xi": 41}

#: Resolution for the single RHSMode=1 solve behind the bootstrap/flux panels.
#: Smaller than the monoenergetic grid because it runs once, not 21 times.
DEFAULT_RESOLUTION_PROFILE = {"n_theta": 21, "n_zeta": 31, "n_xi": 24, "n_x": 5}

#: Generic fallback plasma, used only when the equilibrium carries no pressure.
FALLBACK_PLASMA = {"n_hat": 1.0, "t_hat": 1.0, "dn_dr": -0.5, "dt_dr": -1.0}

#: A deuterium/electron pair at modest collisionality, with the density and
#: temperature gradients that make the bootstrap current nonzero.  The gradient
#: keys must match ``inputRadialCoordinateForGradients``: naming ``dNHatdrHats``
#: while asking for coordinate 3 leaves the gradients at ZERO and the whole
#: solve returns ~1e-20 -- a run that completes and drives nothing.  Follow the
#: upstream decks, which omit the key and use ``dNHatdrHats``.  Deliberately
#: generic: this panel says "here is what this equilibrium does", not "here are
#: your machine's parameters", and the header records that.
_PROFILE_TEMPLATE = """&general
  RHSMode = 1
/
&geometryParameters
  geometryScheme = 5
  equilibriumFile = "{equilibrium}"
  VMECRadialOption = 0
  inputRadialCoordinate = 3
  rN_wish = 0.5
/
&speciesParameters
  Zs = 1.0d+0 -1.0d+0
  mHats = 1.0d+0 5.446170214d-4
  nHats = {n_hat:.6g} {n_hat:.6g}
  THats = {t_hat:.6g} {t_hat:.6g}
  dNHatdrHats = {dn_dr:.6g} {dn_dr:.6g}
  dTHatdrHats = {dt_dr:.6g} {dt_dr:.6g}
/
&physicsParameters
  Delta = 4.5694d-3
  alpha = 1.0d+0
  nu_n = 8.4774d-3
  Er = 0.0d+0
  collisionOperator = 1
/
&resolutionParameters
  Ntheta = {n_theta}
  Nzeta = {n_zeta}
  Nxi = {n_xi}
  NL = 4
  Nx = {n_x}
  solverTolerance = 1d-8
/
&otherNumericalParameters
/
&preconditionerOptions
/
"""

#: Monoenergetic scan grid, spanning the three collisionality regimes a
#: neoclassical paper reports: the ``1/nu`` branch at low collisionality, the
#: plateau, and Pfirsch-Schlueter at high.  Two decades show none of that -- the
#: upstream decks themselves sit at ``nuPrime`` 1.2e-3 to 1.0, so a grid that
#: starts at 1e-2 misses the physics that makes a stellarator interesting.
DEFAULT_NU_PRIME = (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0e0, 1.0e1, 1.0e2)
#: ``EStar`` values.  The intermediate point belongs at 1e-3, not 0.1: between
#: zero field and 0.1 the D11 curves are nearly indistinguishable, so a grid of
#: 0/0.1/0.3 spends two of its three curves in the same regime.
DEFAULT_E_STAR = (0.0, 1.0e-3, 1.0e-1)


def _quiet(fn: Callable[[], Any]) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn()


def monoenergetic_scan(
    namelist: Any,
    *,
    nu_prime: Sequence[float] = DEFAULT_NU_PRIME,
    e_star: Sequence[float] = DEFAULT_E_STAR,
    resolution: dict[str, int] | None = None,
    emit: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Solve the ``(nuPrime, EStar)`` grid and return one record per point.

    Each record carries the full 2x2 transport matrix, so a caller can plot
    ``D11``/``D31``/``D33`` without re-running anything.
    """
    from dkx.inputs import read_sfincs_input, sfincs_input_from_raw  # noqa: PLC0415
    from dkx.run import run_transport_matrix  # noqa: PLC0415

    base = namelist
    if not hasattr(base, "resolution"):
        base = sfincs_input_from_raw(read_sfincs_input(namelist))
    res = dict(DEFAULT_RESOLUTION if resolution is None else resolution)

    records: list[dict[str, Any]] = []
    for e in e_star:
        for nu in nu_prime:
            inp = dataclasses.replace(
                base,
                resolution=dataclasses.replace(base.resolution, **res),
                physics=dataclasses.replace(base.physics, nu_prime=float(nu), e_star=float(e)),
            )
            t0 = time.perf_counter()
            run = _quiet(lambda: run_transport_matrix(inp, out_path=None, emit=None))
            matrix = np.asarray(run.transport_matrix)
            records.append({
                "nu_prime": float(nu), "e_star": float(e),
                "transport_matrix": matrix.tolist(),
                "D11": float(matrix[0, 0]), "D31": float(matrix[0, 1]),
                "D33": float(matrix[1, 1]),
                "wall_s": round(time.perf_counter() - t0, 3),
                "converged": bool(run.solve_result.converged),
            })  # fmt: skip
            if emit:
                emit(f"    nuPrime={nu:<8.3g} EStar={e:<5.3g} D11={matrix[0, 0]:+.4e}"
                     f"  ({records[-1]['wall_s']:.2f}s)")  # fmt: skip
    return records


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------
def _import_matplotlib():
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # noqa: PLC0415

    return plt


def _panel_monoenergetic(ax_d11, ax_d31, ax_d33, records: list[dict[str, Any]]) -> bool:
    """D11*, D31*, D33* against nuPrime, one curve per EStar."""
    if not records:
        return False
    for e in sorted({r["e_star"] for r in records}):
        pts = sorted((r for r in records if r["e_star"] == e), key=lambda r: r["nu_prime"])
        nu = [r["nu_prime"] for r in pts]
        for ax, key in ((ax_d11, "D11"), (ax_d31, "D31"), (ax_d33, "D33")):
            vals = [r[key] if key == "D31" else abs(r[key]) for r in pts]
            ax.plot(nu, vals, "o-", ms=3, label=f"$E^*$={e:g}")
    for ax, key in ((ax_d11, "D_{11}"), (ax_d31, "D_{31}"), (ax_d33, "D_{33}")):
        ax.set_xscale("log")
        ax.set_xlabel(r"$\nu'$")
        ax.grid(alpha=0.3, which="both")
        if ax is ax_d31:
            # D31 is conventionally shown on a LINEAR axis: it changes sign,
            # and |D31| on a log axis hides the zero crossing entirely -- the
            # one feature of that coefficient a reader looks for.
            ax.set_ylabel(rf"${key}$")
            ax.axhline(0.0, color="0.7", lw=0.8)
        else:
            ax.set_yscale("log")
            ax.set_ylabel(rf"$|{key}|$")
        # Every panel carries its own legend: a reader looking at D33 should not
        # have to find the key two panels away.
        ax.legend(fontsize=7)
    return True


def _panel_modB(ax, data: dict[str, Any]) -> bool:
    """|B| over the flux surface -- context for which device produced the rest."""
    b = data.get("BHat")
    if b is None:
        return False
    b = np.asarray(b)
    while b.ndim > 2:
        b = b[..., 0] if b.shape[-1] != b.shape[0] else b[0]
    if b.ndim != 2:
        return False
    theta = np.asarray(data.get("theta", np.arange(b.shape[0]))).ravel()
    zeta = np.asarray(data.get("zeta", np.arange(b.shape[1]))).ravel()
    if min(b.shape) < 2:
        # Axisymmetric (Nzeta=1, i.e. a tokamak): |B| is a curve in theta, and
        # contourf needs at least (2, 2).  Draw the curve rather than skipping
        # the panel -- the field strength is still what the reader wants.
        line = b.ravel()
        # Pick the coordinate whose length matches the surviving axis rather
        # than guessing from the array order: BHat is stored (zeta, theta), so
        # an Nzeta=1 tokamak leaves a curve in THETA even though the degenerate
        # axis is the first one.  Labelling that as zeta would be wrong on every
        # axisymmetric figure.
        if theta.size == line.size:
            angle, label = theta, r"$\theta$"
        elif zeta.size == line.size:
            angle, label = zeta, r"$\zeta$"
        else:
            angle, label = np.arange(line.size), "index"
        ax.plot(angle, line, "-", color="tab:blue")
        ax.set_xlabel(label)
        ax.set_ylabel(r"$|B|$")
        ax.set_title(r"$|B|$ (axisymmetric)", fontsize=9)
        ax.grid(alpha=0.3)
        return True
    # BHat is stored (zeta, theta).  Transpose when that is what the coordinate
    # lengths say, rather than assuming an order and silently plotting against
    # arange -- which is how the axisymmetric label came out as zeta.
    if theta.size == b.shape[1] and zeta.size == b.shape[0]:
        b = b.T
    elif theta.size != b.shape[0] or zeta.size != b.shape[1]:
        theta, zeta = np.arange(b.shape[0]), np.arange(b.shape[1])
    im = ax.contourf(zeta, theta, b, levels=24, cmap="viridis")
    ax.set_xlabel(r"$\zeta$")
    ax.set_ylabel(r"$\theta$")
    ax.set_title(r"$|B|$ on the flux surface", fontsize=9)
    ax.figure.colorbar(im, ax=ax, fraction=0.046)
    return True


def _scalar(data: dict[str, Any], *names: str):
    """First present, finite scalar among ``names`` -- output layouts vary."""
    for n in names:
        if n in data:
            v = np.asarray(data[n]).ravel()
            v = v[np.isfinite(v)]
            if v.size:
                return v
    return None


def _panel_bootstrap(ax, data: dict[str, Any]) -> bool:
    """<j.B>, the parallel-momentum moment a bootstrap study reports."""
    v = _scalar(data, "FSABjHatOverRootFSAB2", "FSABjHat", "FSABjHatOverB0")
    if v is None:
        return False
    if len(v) == 1:
        # One converged value: a full-width bar says nothing a number does not.
        ax.axhline(0.0, color="0.5", lw=0.8)
        ax.plot([0], v, "o", ms=9, color="tab:blue")
        ax.set_xlim(-1, 1)
        ax.set_xticks([])
        ax.text(0.02, 0.06, f"{v[-1]:+.4e}", transform=ax.transAxes, fontsize=11)
    else:
        ax.plot(range(len(v)), v, "o-", ms=4, color="tab:blue")
        ax.axhline(0.0, color="0.5", lw=0.8)
        ax.set_xlabel("Newton iteration")
    ax.set_ylabel(r"$\langle j_\parallel B\rangle/\sqrt{\langle B^2\rangle}$")
    ax.set_title("bootstrap current", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    return True


def _panel_fluxes(ax, data: dict[str, Any]) -> bool:
    """Particle and heat flux per species."""
    pf = _scalar(data, "particleFlux_vm_psiHat", "particleFlux_vm_psiN")
    hf = _scalar(data, "heatFlux_vm_psiHat", "heatFlux_vm_psiN")
    if pf is None and hf is None:
        return False
    n = max(len(pf) if pf is not None else 0, len(hf) if hf is not None else 0)
    idx = np.arange(n)
    w = 0.38
    if pf is not None:
        ax.bar(idx[: len(pf)] - w / 2, np.abs(pf), w, label=r"$|\Gamma_s|$")
    if hf is not None:
        ax.bar(idx[: len(hf)] + w / 2, np.abs(hf), w, label=r"$|Q_s|$")
    ax.set_yscale("log")
    ax.set_xlabel("species / entry")
    ax.set_title("particle and heat flux", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")
    return True


def plot_representative(
    out_path: str | Path,
    *,
    data: dict[str, Any] | None = None,
    scan: list[dict[str, Any]] | None = None,
    ambipolar: list[dict[str, Any]] | None = None,
    profiles: list[dict[str, Any]] | None = None,
    plasma: dict[str, float] | None = None,
    resolutions: dict[str, dict] | None = None,
    title: str = "DKX representative run",
) -> Path:
    """Assemble whichever panels the inputs can support, and say which are missing.

    Panels are rendered on a best-effort basis: a single output file has no
    ``(nuPrime, EStar)`` scan in it, and a monoenergetic file carries no species
    fluxes.  Rather than fail or, worse, draw an empty axis that reads as "zero",
    an unsupported panel is annotated with why it is absent.
    """
    plt = _import_matplotlib()
    data = data or {}
    rows = 3 if ambipolar else 2
    fig = plt.figure(figsize=(13.5, 4.0 * rows), constrained_layout=True)
    gs = fig.add_gridspec(rows, 3)
    axes = {
        "d11": fig.add_subplot(gs[0, 0]), "d31": fig.add_subplot(gs[0, 1]),
        "d33": fig.add_subplot(gs[0, 2]), "modb": fig.add_subplot(gs[1, 0]),
        "boot": fig.add_subplot(gs[1, 1]), "flux": fig.add_subplot(gs[1, 2]),
    }  # fmt: skip
    if ambipolar:
        axes["ambi"] = fig.add_subplot(gs[2, 0])

    drawn = {}
    drawn["monoenergetic"] = _panel_monoenergetic(
        axes["d11"], axes["d31"], axes["d33"], scan or []
    )
    drawn["modB"] = _panel_modB(axes["modb"], data)
    # Radial profiles win where present: a single point at one surface and a
    # bar chart indexed by species number are strictly less informative.
    drawn["bootstrap"] = (
        _panel_radial_bootstrap(axes["boot"], profiles) if profiles
        else _panel_bootstrap(axes["boot"], data)
    )  # fmt: skip
    drawn["fluxes"] = (
        _panel_radial_fluxes(axes["flux"], profiles) if profiles
        else _panel_fluxes(axes["flux"], data)
    )  # fmt: skip
    if ambipolar:
        drawn["ambipolarity"] = _panel_ambipolarity(axes["ambi"], ambipolar)

    if not drawn["monoenergetic"]:
        for key in ("d11", "d31", "d33"):
            axes[key].text(0.5, 0.5, "no (nuPrime, EStar) scan\nin this input",
                           ha="center", va="center", fontsize=8, color="0.4",
                           transform=axes[key].transAxes)  # fmt: skip
            axes[key].set_xticks([]); axes[key].set_yticks([])
    for key, name in (("modb", "modB"), ("boot", "bootstrap"), ("flux", "fluxes")):
        if not drawn[name]:
            axes[key].text(0.5, 0.5, f"{name}: not present\nin this output",
                           ha="center", va="center", fontsize=8, color="0.4",
                           transform=axes[key].transAxes)  # fmt: skip
            axes[key].set_xticks([]); axes[key].set_yticks([])

    # State the plasma and the resolutions on the figure: a reader cannot judge
    # a transport number without knowing the n, T and grid behind it, and the
    # split of pressure into n and T is an assumption, not a measurement.
    if plasma or resolutions:
        bits = []
        if plasma:
            src = "from p(s)" if "p_pa" in plasma else "generic"
            bits.append(
                f"n={plasma['n_hat']:.3g}e20 m$^{{-3}}$, T$_i$=T$_e$={plasma['t_hat']:.3g} keV "
                f"({src}, T constant), dn/dr={plasma['dn_dr']:+.3g}"
            )
        for name, res in (resolutions or {}).items():
            bits.append(f"{name}: " + "x".join(str(v) for v in res.values()))
        fig.text(0.5, 0.005, "   |   ".join(bits), ha="center", fontsize=7.5, color="0.35")
    fig.suptitle(title, fontsize=12)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path.resolve()


def plot_output_file(path: str | Path, out_path: str | Path | None = None) -> Path:
    """Panels from an existing output file --- DKX's or Fortran SFINCS's.

    Both are ``sfincsOutput.h5`` in the same layout (:mod:`dkx.writer`), so no
    format sniffing is needed.
    """
    from dkx.io import read_sfincs_output_file  # noqa: PLC0415

    path = Path(path)
    data = read_sfincs_output_file(path)
    out = Path(out_path) if out_path else path.with_suffix(".panels.png")
    return plot_representative(out, data=data, title=f"DKX panels — {path.name}")


def run_representative(
    equilibrium: str | Path,
    *,
    out_path: str | Path | None = None,
    full: bool = False,
    emit: Callable[[str], None] | None = print,
) -> Path:
    """Solve the representative set for one equilibrium and plot it.

    ``full`` widens the monoenergetic grid; the default keeps the whole run
    inside the tens-of-seconds budget the module docstring quantifies.
    """
    from dkx.inputs import read_sfincs_input, sfincs_input_from_raw  # noqa: PLC0415

    equilibrium = Path(equilibrium)
    template = Path(__file__).parent / "data" / "representative.namelist"
    if not template.exists():  # fall back to an upstream monoenergetic deck
        template = (
            Path(__file__).resolve().parents[1]
            / "examples" / "sfincs_examples" / "monoenergetic_geometryScheme5_netCDF"
            / "input.namelist"
        )  # fmt: skip
    base = sfincs_input_from_raw(read_sfincs_input(template))
    base = dataclasses.replace(
        base, geometry=dataclasses.replace(base.geometry, equilibrium_file=str(equilibrium))
    )
    if emit:
        emit(f"  monoenergetic scan at {DEFAULT_RESOLUTION}")
    nu = DEFAULT_NU_PRIME if not full else tuple(np.logspace(-5, 2, 15))
    scan = monoenergetic_scan(base, nu_prime=nu, emit=emit)

    # The monoenergetic scan alone leaves the whole second row empty: RHSMode=3
    # produces a transport matrix and nothing else.  |B| comes free from the
    # geometry, and one RHSMode=1 solve on the same equilibrium supplies the
    # bootstrap current and the species fluxes.  Without this the figure renders
    # three "not present in this output" boxes, which is not a representative
    # run of anything.
    if emit:
        emit("  profile solve for |B|")
    data = _profile_data(equilibrium, emit=emit)
    if emit:
        emit("  radial scan: ambipolar Er, bootstrap and fluxes at the root")
    profiles = radial_profiles(equilibrium, emit=emit)

    out = Path(out_path) if out_path else Path(f"{equilibrium.stem}.panels.png")
    figure = plot_representative(
        out, data=data, scan=scan, profiles=profiles,
        plasma=plasma_parameters(equilibrium) or dict(FALLBACK_PLASMA),
        resolutions={"monoenergetic": DEFAULT_RESOLUTION,
                     "profiles": DEFAULT_RESOLUTION_PROFILE},
        title=f"DKX representative run — {equilibrium.name}",
    )  # fmt: skip
    # Always leave the numbers behind, not only the picture: a figure cannot be
    # re-plotted, re-scaled or checked against another code.
    written = write_representative_output(
        out.with_suffix(".h5"), scan=scan, profiles=profiles, data=data,
        equilibrium=equilibrium,
    )  # fmt: skip
    if emit:
        emit(f" wrote {written}")
    return figure


#: Reference temperature (keV) used to split the equilibrium's pressure into a
#: density and a temperature.  p = n_i T_i + n_e T_e with quasineutrality and
#: T_i = T_e = T leaves p = 2 n T: one equation, two unknowns.  Fixing T on axis
#: and letting n carry the profile is the conventional choice, and it is stated
#: on the figure because it is an assumption, not a measurement.
DEFAULT_T_AXIS_KEV = 2.0


def _plasma_keys(plasma: dict) -> dict:
    """Only the keys the namelist template interpolates."""
    return {k: plasma[k] for k in ("n_hat", "t_hat", "dn_dr", "dt_dr")}


def plasma_parameters(equilibrium: Path, radius: float = 0.5) -> dict[str, float]:
    """Density and temperature at ``r/a`` from the equilibrium's own pressure.

    The wout carries ``presf`` in Pascals; a drift-kinetic run needs ``n`` and
    ``T`` separately, and pressure alone does not determine them.  The stated
    assumption is quasineutral hydrogen with ``T_i = T_e = T``, ``T`` fixed on
    axis at :data:`DEFAULT_T_AXIS_KEV` and constant, so ``n(s) = p(s) / (2 T)``.

    Returns ``{}`` when the file carries no pressure, in which case the caller
    keeps the generic reference profile and says so.  Reporting numbers derived
    from an equilibrium the user supplied, without saying how, would be worse
    than reporting generic ones.
    """
    try:
        import netCDF4  # noqa: PLC0415

        with netCDF4.Dataset(str(equilibrium)) as handle:
            if "presf" not in handle.variables:
                return {}
            pres = np.asarray(handle.variables["presf"][:], dtype=float)
    except Exception:
        return {}
    if pres.size < 2 or not np.any(pres > 0.0):
        return {}
    s_grid = np.linspace(0.0, 1.0, pres.size)
    p_pa = float(np.interp(radius**2, s_grid, pres))          # r/a = sqrt(s)
    t_kev = DEFAULT_T_AXIS_KEV
    # n [1e20 m^-3] = p / (2 T) with T in Joules.
    n_20 = p_pa / (2.0 * t_kev * 1.0e3 * 1.602176634e-19) / 1.0e20
    # Logarithmic gradients from the pressure profile itself, at fixed T.
    eps = 1.0e-3
    lo = float(np.interp(max(radius - eps, 0.0) ** 2, s_grid, pres))
    hi = float(np.interp(min(radius + eps, 1.0) ** 2, s_grid, pres))
    dn_dr = (hi - lo) / (2.0 * eps) / (2.0 * t_kev * 1.0e3 * 1.602176634e-19) / 1.0e20
    return {"n_hat": n_20, "t_hat": t_kev, "dn_dr": dn_dr, "dt_dr": 0.0,
            "p_pa": p_pa, "radius": radius}  # fmt: skip


def _field_periods(equilibrium: Path) -> int:
    """Field-period count, read from the wout itself.

    For ``geometryScheme=5`` the namelist's ``NPeriods`` stays at its default of
    0 and the real value comes from the equilibrium, so neither the input object
    nor the operator carries it.  Falling back to 1 draws zeta over a full turn
    on an nfp-period device --- an axis wrong by exactly that factor.
    """
    try:
        import netCDF4  # noqa: PLC0415

        with netCDF4.Dataset(str(equilibrium)) as handle:
            return max(1, int(handle.variables["nfp"][...]))
    except Exception:
        return 1


def _profile_data(equilibrium: Path, *, emit: Callable[[str], None] | None = None) -> dict:
    """Geometry plus one RHSMode=1 solve, for the panels the scan cannot fill.

    Returns ``{}`` rather than raising if the profile solve does not apply to
    this equilibrium: a missing panel that says so is better than no figure.
    """
    from dkx.inputs import parse_sfincs_input_text, sfincs_input_from_raw  # noqa: PLC0415
    from dkx.run import run_profile  # noqa: PLC0415

    plasma = plasma_parameters(equilibrium) or dict(FALLBACK_PLASMA)
    derived = "p(s) from the equilibrium" if "p_pa" in plasma else "generic reference"
    if emit:
        emit(f"    plasma ({derived}): n={plasma['n_hat']:.3g}e20 m^-3, "
             f"T={plasma['t_hat']:.3g} keV, dn/dr={plasma['dn_dr']:+.3g}")  # fmt: skip
        emit(f"    profile resolution: {DEFAULT_RESOLUTION_PROFILE}")
    text = _PROFILE_TEMPLATE.format(
        equilibrium=str(equilibrium), **DEFAULT_RESOLUTION_PROFILE, **_plasma_keys(plasma)
    )
    try:
        # run_profile takes a path or an SfincsInput; parse_sfincs_input_text
        # returns a RawNamelist, which it silently mistakes for a path.
        inp = sfincs_input_from_raw(parse_sfincs_input_text(text))
        run = _quiet(lambda: run_profile(inp, out_path=None, emit=None))
    except Exception as exc:  # pragma: no cover - geometry-dependent
        # An out-of-memory here costs three panels, so retry once at half the
        # angular resolution rather than giving up: |B| and a bootstrap number
        # at reduced resolution beat three boxes saying "not present".  The
        # reduced resolution is reported, because a panel at a resolution the
        # caller did not choose must say so.
        reduced = {"n_theta": 15, "n_zeta": 15, "n_xi": 25, "n_x": 5}
        if emit:
            emit(f"    profile solve failed ({type(exc).__name__}); "
                 f"retrying at {reduced}")  # fmt: skip
        try:
            inp = sfincs_input_from_raw(parse_sfincs_input_text(
                _PROFILE_TEMPLATE.format(equilibrium=str(equilibrium), **reduced,
                                         **_plasma_keys(plasma))
            ))  # fmt: skip
            run = _quiet(lambda: run_profile(inp, out_path=None, emit=None))
            resolution = reduced
        except Exception as exc2:
            if emit:
                emit(f"    still unavailable ({type(exc2).__name__}); "
                     f"|B|/bootstrap/flux panels will say so")  # fmt: skip
            return {}
    else:
        resolution = dict(DEFAULT_RESOLUTION_PROFILE)
    op, mom = run.operator, run.moments
    # The operator carries no angle grids, and b_hat here is (n_theta, n_zeta)
    # -- the OPPOSITE order from the output-file layout the other panel path
    # sees.  Build the uniform grids explicitly rather than falling back to
    # arange, which silently labels a publication figure in index units.
    nfp = _field_periods(equilibrium)
    data: dict[str, Any] = {
        "resolution": resolution,
        "BHat": np.asarray(op.b_hat),
        "theta": np.linspace(0.0, 2.0 * np.pi, op.n_theta, endpoint=False),
        "zeta": np.linspace(0.0, 2.0 * np.pi / nfp, op.n_zeta, endpoint=False),
    }
    for key in ("FSABjHatOverRootFSAB2", "FSABjHat", "FSABFlow",
                "particleFlux_vm_psiHat", "heatFlux_vm_psiHat"):  # fmt: skip
        if key in mom:
            data[key] = np.asarray(mom[key])
    if emit:
        boot = data.get("FSABjHatOverRootFSAB2", data.get("FSABjHat"))
        if boot is not None:
            emit(f"    bootstrap <j.B>/sqrt(<B^2>) = {np.asarray(boot).ravel()[-1]:+.4e}")
    return data


# ---------------------------------------------------------------------------
# Panels that need more than one solve (opt-in via --full)
# ---------------------------------------------------------------------------
#: E_r values (kV/m) for the ambipolarity panel.  Spans the ion root and, on
#: electron-root devices, the positive branch; a root-find would be cheaper but
#: a scan is what shows a reader whether the root is unique.
DEFAULT_ER_SCAN = (-8.0, -5.0, -3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 4.0)


def ambipolarity_scan(
    namelist: Any,
    *,
    er_values: Sequence[float] = DEFAULT_ER_SCAN,
    emit: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Radial current against ``E_r``: the ambipolar condition is ``J_r = 0``.

    One batched ``vmap`` solve over the whole ``E_r`` vector on a shared
    geometry, so this costs far less than the point count suggests.
    """
    from dkx.api import batched_er_scan  # noqa: PLC0415

    result = _quiet(lambda: batched_er_scan(namelist, np.asarray(er_values, dtype=float)))
    # BatchedSolveResult names it `radial_current`; guessing "J_r" silently
    # yields an empty scan and an empty panel rather than an error.
    j_r = np.asarray(result.radial_current, dtype=float).ravel()
    records = [
        {"er": float(e), "J_r": float(j)}
        for e, j in zip(np.asarray(er_values, dtype=float), j_r)
    ]
    if emit:
        for r in records:
            emit(f"    Er={r['er']:+6.2f} kV/m   J_r={r['J_r']:+.4e}")
    return records


def _ambipolar_roots(records: list[dict[str, Any]]) -> list[float]:
    """Sign changes of ``J_r(E_r)``, linearly interpolated.

    Reported as *bracketed* roots rather than solved ones: the panel's job is to
    show how many there are (one ion root, or the ion/unstable/electron triplet),
    which a Brent solve started from a single guess would hide.
    """
    roots: list[float] = []
    for a, b in zip(records, records[1:]):
        ja, jb = a["J_r"], b["J_r"]
        if np.isfinite(ja) and np.isfinite(jb) and ja * jb < 0.0:
            roots.append(a["er"] + (b["er"] - a["er"]) * ja / (ja - jb))
    return roots


def _panel_ambipolarity(ax, records: list[dict[str, Any]]) -> bool:
    if not records:
        return False
    er = [r["er"] for r in records]
    jr = [r["J_r"] for r in records]
    ax.plot(er, jr, "o-", ms=3, color="tab:purple")
    ax.axhline(0.0, color="0.5", lw=0.8)
    for root in _ambipolar_roots(records):
        ax.axvline(root, color="tab:red", ls="--", lw=0.9)
        ax.annotate(f"{root:+.2f}", (root, 0.0), fontsize=7, color="tab:red",
                    xytext=(2, 6), textcoords="offset points")  # fmt: skip
    ax.set_xlabel(r"$E_r$ [kV/m]")
    ax.set_ylabel(r"$J_r$")
    ax.set_title("ambipolarity: roots of $J_r(E_r)$", fontsize=9)
    ax.grid(alpha=0.3)
    return True


# ---------------------------------------------------------------------------
# Radial profiles at the ambipolar root
# ---------------------------------------------------------------------------
#: Flux-surface labels for the radial scan.  Five is enough to show a profile's
#: shape; the cost is one batched E_r scan per surface.
DEFAULT_SURFACES = (0.25, 0.4, 0.55, 0.7, 0.85)

#: E_r values (kV/m) bracketing the ion root at every surface.
DEFAULT_ER_BRACKET = (-8.0, -4.0, -2.0, -1.0, -0.4, 0.4, 1.0, 2.0)


def _interp_at_root(er: np.ndarray, values: np.ndarray, root: float) -> float:
    """Linear interpolation of a scanned quantity onto the ambipolar root."""
    order = np.argsort(er)
    return float(np.interp(root, np.asarray(er)[order], np.asarray(values)[order]))


def radial_profiles(
    equilibrium: Path,
    *,
    surfaces: Sequence[float] = DEFAULT_SURFACES,
    er_values: Sequence[float] = DEFAULT_ER_BRACKET,
    emit: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Ambipolar ``E_r`` and the moments evaluated *at that root*, per surface.

    One batched ``E_r`` scan per surface, because that single call returns the
    radial current *and* every moment for every ``E_r``.  Evaluating the
    bootstrap current and the fluxes at the ambipolar root is the physically
    meaningful thing to report: at ``E_r = 0`` they are not what the device
    would actually do.
    """
    import tempfile  # noqa: PLC0415

    from dkx.api import batched_er_scan  # noqa: PLC0415

    er = np.asarray(er_values, dtype=float)
    if emit:
        emit(f"    radial-scan resolution: {DEFAULT_RESOLUTION_PROFILE}; "
             f"{len(er)} Er points per surface")  # fmt: skip
    plasma = plasma_parameters(equilibrium) or dict(FALLBACK_PLASMA)
    template = _PROFILE_TEMPLATE.format(
        equilibrium=str(equilibrium), **DEFAULT_RESOLUTION_PROFILE, **_plasma_keys(plasma)
    ).replace("rN_wish = 0.5", "rN_wish = {radius}")  # fmt: skip

    out: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as work:
        for radius in surfaces:
            deck = Path(work) / f"in_{radius}.namelist"
            deck.write_text(template.format(radius=radius))
            try:
                scan = _quiet(lambda d=deck: batched_er_scan(d, er))
            except Exception as exc:  # pragma: no cover - geometry-dependent
                if emit:
                    emit(f"    r/a={radius:.2f}: unavailable ({type(exc).__name__})")
                continue
            j_r = np.asarray(scan.radial_current, dtype=float).ravel()
            roots = _ambipolar_roots([{"er": float(e), "J_r": float(j)}
                                      for e, j in zip(er, j_r)])  # fmt: skip
            record: dict[str, Any] = {"r": float(radius), "er_scan": er.tolist(),
                                      "J_r": j_r.tolist(), "roots": roots}  # fmt: skip
            if roots:
                # The ion root is the most negative crossing; a device in the
                # electron-root regime has a positive one too, and reporting
                # only the first found would hide that.
                root = min(roots)
                record["er_ambipolar"] = root
                mom = scan.moments
                boot = mom.get("FSABjHatOverRootFSAB2", mom.get("FSABjHat"))
                if boot is not None:
                    record["bootstrap"] = _interp_at_root(er, np.asarray(boot).ravel(), root)
                for key, name in (("particleFlux_vm_psiHat", "particle_flux"),
                                  ("heatFlux_vm_psiHat", "heat_flux")):  # fmt: skip
                    if key in mom:
                        arr = np.asarray(mom[key])
                        record[name] = [
                            _interp_at_root(er, arr[:, s], root) for s in range(arr.shape[1])
                        ]
            if emit:
                e_txt = f"{record.get('er_ambipolar', float('nan')):+.3f}"
                b_txt = f"{record.get('bootstrap', float('nan')):+.4e}"
                emit(f"    r/a={radius:.2f}  Er_ambipolar={e_txt} kV/m  <j.B>={b_txt}")
            out.append(record)
    return out


def _species_labels(n: int, charges: Sequence[float] | None = None) -> list[str]:
    """Name the species, because "0" and "1" tell a reader nothing.

    Sign of the charge is what distinguishes them; the template's ``Zs`` are
    ``+1`` and ``-1``, so the electron is the negative one.
    """
    if charges is not None and len(charges) == n:
        return ["electrons" if z < 0 else ("ions" if n <= 2 else f"ion Z={z:g}")
                for z in charges]  # fmt: skip
    return ["ions", "electrons"][:n] if n <= 2 else [f"species {i}" for i in range(n)]


def _panel_radial_bootstrap(ax, profiles: list[dict[str, Any]]) -> bool:
    """Bootstrap current and ambipolar E_r against radius, on twin axes.

    Both are radial profiles a reader wants together: the bootstrap current is
    evaluated *at* the ambipolar root, so plotting the root beside it says which
    field the current belongs to.
    """
    pts = [p for p in profiles if "bootstrap" in p]
    if not pts:
        return False
    r = [p["r"] for p in pts]
    ax.plot(r, [p["bootstrap"] for p in pts], "o-", ms=4, color="tab:blue",
            label=r"$\langle j_\parallel B\rangle/\sqrt{\langle B^2\rangle}$")  # fmt: skip
    ax.axhline(0.0, color="0.7", lw=0.8)
    ax.set_xlabel("$r/a$")
    ax.set_ylabel(r"$\langle j_\parallel B\rangle/\sqrt{\langle B^2\rangle}$", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax.grid(alpha=0.3)

    twin = ax.twinx()
    er = [p.get("er_ambipolar", float("nan")) for p in pts]
    twin.plot(r, er, "s--", ms=4, color="tab:red", label=r"$E_r$ (ambipolar)")
    twin.set_ylabel(r"$E_r$ [kV/m]", color="tab:red")
    twin.tick_params(axis="y", labelcolor="tab:red")
    ax.set_title("bootstrap current and ambipolar $E_r$", fontsize=9)
    handles = ax.get_lines()[:1] + twin.get_lines()[:1]
    ax.legend(handles, [h.get_label() for h in handles], fontsize=7, loc="best")
    return True


def _panel_radial_fluxes(ax, profiles: list[dict[str, Any]],
                         charges: Sequence[float] | None = None) -> bool:  # fmt: skip
    """Particle and heat flux per species against radius.

    Named species and a radial axis, rather than a bar chart indexed by species
    number: the convention every neoclassical paper uses, and the only form in
    which "which one is the electron" is answerable from the figure.  Particle
    and heat flux differ by orders of magnitude, so the heat flux takes a twin
    axis rather than being flattened onto one log scale with the other.
    """
    pts = [p for p in profiles if "particle_flux" in p or "heat_flux" in p]
    if not pts:
        return False
    r = [p["r"] for p in pts]
    n = max(len(p.get("particle_flux", p.get("heat_flux", []))) for p in pts)
    names = _species_labels(n, charges)
    styles = ("o-", "s-", "^-")

    # At the ambipolar root sum_s Z_s Gamma_s = 0, so for a Z=+-1 pair the two
    # particle fluxes are EQUAL by construction -- drawing both puts one line
    # exactly on top of the other, which reads as a rendering fault rather than
    # as the physics it is.  Draw one, and say why.
    gammas = [[p["particle_flux"][s] for p in pts if "particle_flux" in p] for s in range(n)]
    ambipolar_pair = (
        n == 2 and gammas[0] and gammas[1]
        and np.allclose(gammas[0], gammas[1], rtol=1e-6, atol=0.0)
    )  # fmt: skip
    if ambipolar_pair:
        ax.plot(r[: len(gammas[0])], gammas[0], "o-", ms=4, color="C0",
                label=r"$\Gamma_i=\Gamma_e$ (ambipolar)")  # fmt: skip
    else:
        for s in range(n):
            if gammas[s]:
                ax.plot(r[: len(gammas[s])], gammas[s], styles[s % 3], ms=4,
                        color=f"C{s}", label=rf"$\Gamma$ {names[s]}")  # fmt: skip
    ax.set_xlabel("$r/a$")
    ax.set_ylabel(r"$\Gamma_s$  [particle flux]")
    ax.grid(alpha=0.3)

    twin = ax.twinx()
    for s in range(n):
        q = [p["heat_flux"][s] for p in pts if "heat_flux" in p]
        if q:
            # marker only in the fmt: passing "o-" together with ls="--" makes
            # matplotlib warn that the linestyle is defined twice.
            twin.plot(r[: len(q)], q, marker=styles[s % 3][0], ms=4, ls="--",
                      alpha=0.75, color=f"C{s + n}", label=rf"$Q$ {names[s]}")  # fmt: skip
    twin.set_ylabel(r"$Q_s$  [heat flux]")
    ax.set_title("particle and heat flux at the ambipolar root", fontsize=9)
    handles = ax.get_lines() + twin.get_lines()
    ax.legend(handles, [h.get_label() for h in handles], fontsize=6, loc="best", ncol=2)
    return True


def write_representative_output(
    path: str | Path,
    *,
    scan: list[dict[str, Any]] | None = None,
    profiles: list[dict[str, Any]] | None = None,
    data: dict[str, Any] | None = None,
    equilibrium: str | Path | None = None,
) -> Path:
    """Persist the numbers behind the figure.

    A run that leaves only a PNG cannot be re-plotted, re-scaled, or checked
    against another code, so ``dkx wout_*.nc`` always writes this alongside it.
    HDF5 when ``h5py`` is available, JSON otherwise --- the point is that the
    data survives, not the container.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "equilibrium": str(equilibrium) if equilibrium else "",
        "resolution_monoenergetic": DEFAULT_RESOLUTION,
        "resolution_profile": DEFAULT_RESOLUTION_PROFILE,
    }
    if scan:
        for key in ("nu_prime", "e_star", "D11", "D31", "D33"):
            payload[f"monoenergetic/{key}"] = [r[key] for r in scan]
    if profiles:
        payload["profiles/r"] = [p["r"] for p in profiles]
        for key in ("er_ambipolar", "bootstrap"):
            payload[f"profiles/{key}"] = [p.get(key, float("nan")) for p in profiles]
        for key in ("particle_flux", "heat_flux"):
            rows = [p.get(key) for p in profiles if p.get(key) is not None]
            if rows:
                payload[f"profiles/{key}"] = rows
    if data and "BHat" in data:
        for key in ("BHat", "theta", "zeta"):
            if key in data:
                payload[f"geometry/{key}"] = np.asarray(data[key]).tolist()

    try:
        import h5py  # noqa: PLC0415

        with h5py.File(path, "w") as handle:
            for key, value in payload.items():
                if isinstance(value, str):
                    handle.attrs[key] = value
                elif isinstance(value, dict):
                    for sub, v in value.items():
                        handle.attrs[f"{key}/{sub}"] = v
                else:
                    handle.create_dataset(key, data=np.asarray(value, dtype=float))
        return path.resolve()
    except Exception:
        import json  # noqa: PLC0415

        out = path.with_suffix(".json")
        out.write_text(json.dumps(payload, indent=1) + "\n")
        return out.resolve()
