"""One-command representative run: an equilibrium in, publication panels out.

``dkx wout_XXX.nc`` solves the small set of cases a neoclassical paper actually
reports and renders them as one figure, in about a minute rather than hours.
``dkx --plot FILE`` renders the same panels from an existing output file --- and
because :mod:`dkx.writer` emits ``sfincsOutput.h5`` in SFINCS's own layout, that
works on Fortran SFINCS output too, with no separate reader.

Panels
------
* **monoenergetic** ``D11*``, ``D31*``, ``D33*`` against ``nuPrime``, one curve
  per ``EStar``.  The standard cross-code benchmark figure.
* **bootstrap** ``<j.B>/sqrt(<B^2>)`` in kA/m^2 against radius, evaluated at
  the ambipolar root, with the VMEC equilibrium's own current beside it and the
  ambipolar ``E_r`` on a twin axis.
* **fluxes** ``<Gamma.grad r>`` and ``<Q.grad r>`` per species against radius,
  in SI units (:mod:`dkx.units`), at the ambipolar root.
* **|B|** on the flux surface, for context on what device produced the numbers.

Default resolution
------------------
Monoenergetic ``Ntheta=25, Nzeta=25, Nxi=41``; radial profiles
``Ntheta=21, Nzeta=31, Nxi=24, Nx=5``.  A convergence scan against a
``41x71x80`` reference, run at moderate collisionality, gives:

===========  =====================  ===================
axis         error at the low end   at the high end
===========  =====================  ===================
``Nzeta``    3.59e-02 (15)          1.55e-05 (61)
``Ntheta``   1.03e-02 (11)          4.00e-04 (31)
``Nxi``      1.10e-07 (12)          5.59e-10 (64)
===========  =====================  ===================

By that table ``Nxi`` looks free and ``Nzeta`` expensive.  The default
nevertheless carries ``Nxi >= Nzeta``, because the Legendre resolution is what
binds at the *low-collisionality* end of :data:`DEFAULT_NU_PRIME`, which the
scan above did not probe --- it was run at one moderate ``nuPrime``.  Treat the
table as a floor on the spatial axes, not as permission to cut ``Nxi``.

Cost
----
Measured on W7-X standard configuration (10-core M4, float64), the three stages
are **18.5 s** for the 21-point monoenergetic scan (0.8 s per solve after a
1.8 s cold compile), **11.1 s** for the ``|B|`` profile solve, and **33.2 s**
for the radial scan (5 surfaces x 8 ``E_r`` points): **63.5 s** in total.

A *vacuum* equilibrium used to take 23 minutes instead, because VMEC writes
``presf`` of order 1e-6 Pa rather than zero and the pressure split turned that
into a collisionless 1e9 m^-3 plasma.  :data:`VACUUM_DENSITY_FLOOR` now catches
that and :func:`resolve_plasma` says so on the figure.

Monoenergetic runs take ``nuPrime``/``EStar``, **not** ``nu_n``/``Er`` --- the
upstream decks say so in a comment, and varying ``nu_n`` here changes nothing at
all while looking like it worked.
"""

from __future__ import annotations

import dataclasses
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

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
FALLBACK_PLASMA = {"n_hat": 1.0, "t_hat": 1.0, "dn_ds": -0.5, "dt_ds": -1.0}

#: A deuterium/electron pair at modest collisionality, with the density and
#: temperature gradients that make the bootstrap current nonzero.  The gradient
#: keys must match ``inputRadialCoordinateForGradients``, and the key lives in
#: ``&geometryParameters``: a mismatch leaves the gradients at ZERO and the
#: whole solve returns ~1e-20 -- a run that completes and drives nothing.
#: Coordinate 1 is ``psiN = s``, which is what the wout tabulates ``presf`` in,
#: so the profile derivative needs no chain rule and cannot pick up a stray
#: factor of ``aHat``.
#:
#: ``collisionOperator = 0`` is the full linearized Fokker-Planck operator.
#: Pitch-angle scattering is cheaper and fine for ``D11``, but the bootstrap
#: current is the parallel-momentum moment and PAS has no momentum-restoring
#: term: measured against Redl on a finite-beta precise-QA equilibrium it runs
#: 35-47% high, where Fokker-Planck lands within 2-7%
#: (:data:`dkx.bootstrap.DEFAULT_COLLISION_OPERATOR`).
_PROFILE_TEMPLATE = """&general
  RHSMode = 1
/
&geometryParameters
  geometryScheme = 5
  equilibriumFile = "{equilibrium}"
  VMECRadialOption = 0
  inputRadialCoordinate = 3
  inputRadialCoordinateForGradients = 1
  rN_wish = 0.5
/
&speciesParameters
  Zs = 1.0d+0 -1.0d+0
  mHats = 1.0d+0 5.446170214d-4
  nHats = {n_hat:.6g} {n_hat:.6g}
  THats = {t_hat:.6g} {t_hat:.6g}
  dNHatdpsiNs = {dn_ds:.6g} {dn_ds:.6g}
  dTHatdpsiNs = {dt_ds:.6g} {dt_ds:.6g}
/
&physicsParameters
  Delta = 4.5694d-3
  alpha = 1.0d+0
  nu_n = 8.4774d-3
  Er = 0.0d+0
  collisionOperator = 0
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
    plasma_source: str = "",
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
            bits.append(
                f"n={plasma['n_hat']:.3g}e20 m$^{{-3}}$, T$_i$=T$_e$={plasma['t_hat']:.3g} keV "
                f"dn/ds={plasma['dn_ds']:+.3g}, dT/ds={plasma['dt_ds']:+.3g}"
            )
        for name, res in (resolutions or {}).items():
            bits.append(f"{name} " + "x".join(str(v) for v in res.values()))
        caption = "   |   ".join(bits)
        if plasma:
            src = plasma_source or (
                "p(s) from the equilibrium" if "p_pa" in plasma else "generic reference")
            caption += f"\nplasma source: {src}"
        # Reserve the strip rather than drawing into it: constrained_layout owns
        # the whole canvas, so a bare fig.text lands on top of the bottom row's
        # x-labels.  Older matplotlib has no layout engine to ask; there the
        # caption goes under the figure and the axes keep their room.
        band = 0.018 * caption.count("\n") + 0.030
        engine = getattr(fig, "get_layout_engine", lambda: None)()
        if engine is not None:
            # Reserve the title strip as well: rect hands the engine the whole
            # remaining canvas, and suptitle is not one of the artists it packs.
            engine.set(rect=(0.0, band, 1.0, 1.0 - _TITLE_BAND / rows))
        fig.text(0.5, band * 0.5, caption, ha="center", va="center",
                 fontsize=7.5, color="0.35", linespacing=1.5)  # fmt: skip
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
    inside the one-minute budget the module docstring measures.
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
    # Each stage is timed and the time is printed.  The three stages differ by
    # more than an order of magnitude in cost, and on a large device the radial
    # scan dominates, so a reader deciding whether to trim DEFAULT_SURFACES
    # needs the split rather than one total at the end.
    started = time.perf_counter()

    def stage(label: str) -> Callable[[], None]:
        mark = time.perf_counter()
        if emit:
            emit(label)
        return lambda: (emit(f"    ({time.perf_counter() - mark:.1f} s)") if emit else None)

    done = stage(f"  monoenergetic scan at {DEFAULT_RESOLUTION}")
    nu = DEFAULT_NU_PRIME if not full else tuple(np.logspace(-5, 2, 15))
    scan = monoenergetic_scan(base, nu_prime=nu, emit=emit)
    done()

    # The monoenergetic scan alone leaves the whole second row empty: RHSMode=3
    # produces a transport matrix and nothing else.  |B| comes free from the
    # geometry, and one RHSMode=1 solve on the same equilibrium supplies the
    # bootstrap current and the species fluxes.  Without this the figure renders
    # three "not present in this output" boxes, which is not a representative
    # run of anything.
    done = stage("  profile solve for |B|")
    data = _profile_data(equilibrium, emit=emit)
    done()
    done = stage("  radial scan: ambipolar Er, bootstrap and fluxes at the root")
    profiles = radial_profiles(equilibrium, emit=emit)
    done()
    plasma, plasma_source = resolve_plasma(equilibrium)

    out = Path(out_path) if out_path else Path(f"{equilibrium.stem}.panels.png")
    figure = plot_representative(
        out, data=data, scan=scan, profiles=profiles,
        plasma=plasma, plasma_source=plasma_source,
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
        emit(f" total {time.perf_counter() - started:.1f} s")
    return figure


#: Reference temperature (keV) used to split the equilibrium's pressure into a
#: density and a temperature.  p = n_i T_i + n_e T_e with quasineutrality and
#: T_i = T_e = T leaves p = 2 n T: one equation, two unknowns.  Fixing T on axis
#: and letting n carry the profile is the conventional choice, and it is stated
#: on the figure because it is an assumption, not a measurement.
DEFAULT_T_AXIS_KEV = 2.0

#: Exponent splitting the pressure between temperature and density:
#: ``T ~ p^TEMPERATURE_PRESSURE_EXPONENT`` and therefore ``n ~ p^(1-e)``.
#: One third gives ``n ~ p^(2/3)``, the conventional mild split.  The exponent
#: form is what keeps the closure well behaved: both profiles then fall wherever
#: the pressure falls.  A temperature falling *linearly* against a flat-topped
#: pressure makes ``n = p/(2T)`` rise off-axis, i.e. invents a hollow density.
TEMPERATURE_PRESSURE_EXPONENT = 1.0 / 3.0


def _temperature(p_pa: float, p_axis: float) -> float:
    """``T(s) = T_0 (p(s)/p(0))^e`` in keV --- the assumed half of the closure."""
    if p_axis <= 0.0:
        return DEFAULT_T_AXIS_KEV
    ratio = max(float(p_pa) / float(p_axis), 0.0)
    return DEFAULT_T_AXIS_KEV * ratio**TEMPERATURE_PRESSURE_EXPONENT


def _plasma_keys(plasma: dict) -> dict:
    """Only the keys the namelist template interpolates."""
    return {k: plasma[k] for k in ("n_hat", "t_hat", "dn_ds", "dt_ds")}


#: Smallest on-axis density (in 1e20 m^-3) the pressure split may produce.
#: VMEC writes ``presf`` of order 1e-6 Pa for a *vacuum* run rather than exactly
#: zero, and the split turns that into n ~ 1e9 m^-3: a collisionless deck that
#: grinds for tens of minutes and reports transport for a plasma that is not
#: there.  1e-3 is 1e17 m^-3, the same floor ``vmex`` clamps its Redl profiles
#: to, and eleven orders of magnitude below any fusion plasma.
VACUUM_DENSITY_FLOOR = 1.0e-3


def plasma_parameters(equilibrium: Path, radius: float = 0.5) -> dict[str, float]:
    r"""Density and temperature at ``r/a`` from the equilibrium's own pressure.

    The wout carries ``presf`` in Pascals; a drift-kinetic run needs ``n`` and
    ``T`` separately, and pressure alone does not determine them.  One equation,
    two unknowns, so the closure is stated rather than hidden: quasineutral
    hydrogen with ``T_i = T_e = T`` and the pressure split by a power law,

    .. math:: T(s) = T_0 (p(s)/p(0))^{e},
              \qquad n(s) = p(s) / (2 T(s)) \propto p^{1-e},

    with ``T_0 =`` :data:`DEFAULT_T_AXIS_KEV` and ``e =``
    :data:`TEMPERATURE_PRESSURE_EXPONENT`.  Only ``T_0`` is free, and the figure
    states it.

    The temperature must carry a gradient.  A *constant* ``T`` is the simpler
    closure and it is wrong for this figure: the bootstrap current is driven
    mostly by ``dT/dr``, so ``dT/ds = 0`` reports a device with almost no
    bootstrap current and puts the kinetic curve an order of magnitude under
    the equilibrium's own.

    Gradients are returned as ``d/d psiN`` with ``psiN = s``, matching
    ``inputRadialCoordinateForGradients = 1``.  That is the coordinate
    ``presf`` is tabulated in, so no chain rule and no stray ``aHat`` --- one
    factor of ``aHat`` (0.17 on a compact device) is a six-fold error in the
    drive.

    Returns ``{}`` when the file carries no usable pressure --- absent, or below
    :data:`VACUUM_DENSITY_FLOOR` --- in which case the caller keeps the generic
    reference profile and says so.  Reporting numbers derived from an
    equilibrium the user supplied, without saying how, would be worse than
    reporting generic ones.
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
    p_axis = float(pres[0])

    def profiles(s: float) -> tuple[float, float]:
        """``(n [1e20 m^-3], T [keV])`` implied by ``p(s)`` under the closure."""
        p_pa = float(np.interp(s, s_grid, pres))
        t_kev = _temperature(p_pa, p_axis)
        if t_kev <= 0.0:
            return 0.0, 0.0
        return p_pa / (2.0 * t_kev * 1.0e3 * 1.602176634e-19) / 1.0e20, t_kev

    s0 = float(np.clip(radius, 0.0, 1.0)) ** 2  # r/a = sqrt(s)
    n_20, t_kev = profiles(s0)
    if not np.isfinite(n_20) or n_20 < VACUUM_DENSITY_FLOOR:
        return {}
    eps = 1.0e-3
    lo, hi = max(s0 - eps, 0.0), min(s0 + eps, 1.0)
    n_lo, t_lo = profiles(lo)
    n_hi, t_hi = profiles(hi)
    return {
        "n_hat": n_20, "t_hat": t_kev,
        "dn_ds": (n_hi - n_lo) / (hi - lo), "dt_ds": (t_hi - t_lo) / (hi - lo),
        "p_pa": float(np.interp(s0, s_grid, pres)), "radius": float(radius),
    }  # fmt: skip


def equilibrium_scalars(equilibrium: Path) -> dict[str, Any]:
    """``psiAHat``, ``aHat`` and the VMEC ``<J.B>`` profile, read from the wout.

    ``psiAHat = phi(ns)/(2 pi)`` and ``aHat = Aminor_p`` are how
    ``geometry.F90:130`` and :func:`dkx.magnetic_geometry.psi_a_hat_from_wout`
    define the two flux-surface constants for ``geometryScheme=5``; they set the
    ``psiHat -> rHat`` factor every dimensional flux needs.  ``jdotb`` is VMEC's
    own equilibrium parallel current, in A T/m^2 --- the same quantity DKX
    computes kinetically, so the two can be drawn on one axis.

    Returns ``{}`` if the file cannot be read; every caller treats the
    dimensional overlay as optional.
    """
    try:
        import netCDF4  # noqa: PLC0415

        with netCDF4.Dataset(str(equilibrium)) as handle:
            out: dict[str, Any] = {}
            if "phi" in handle.variables:
                phi = np.asarray(handle.variables["phi"][:], dtype=float)
                out["psi_a_hat"] = float(phi[-1]) / (2.0 * np.pi)
            if "Aminor_p" in handle.variables:
                out["a_hat"] = float(np.asarray(handle.variables["Aminor_p"][...]).reshape(()))
            if "jdotb" in handle.variables:
                jdotb = np.asarray(handle.variables["jdotb"][:], dtype=float)
                out["jdotb"] = jdotb
                out["jdotb_s"] = np.linspace(0.0, 1.0, jdotb.size)
            return out
    except Exception:
        return {}


def resolve_plasma(equilibrium: Path) -> tuple[dict[str, float], str]:
    """The plasma to run, and a phrase saying where it came from.

    Three cases, and the caller must be able to tell them apart on the figure:
    the equilibrium carries a real pressure profile; it carries none; or it
    carries a *vacuum* one, which is the trap.  A vacuum wout's ``presf`` is
    ~1e-6 Pa rather than zero, so the split silently yields n ~ 1e9 m^-3 and a
    collisionless deck --- a 23-minute run on W7-X whose every transport number
    describes a plasma that is not there.
    """
    plasma = plasma_parameters(equilibrium)
    if plasma:
        return plasma, "p(s) from the equilibrium"
    return dict(FALLBACK_PLASMA), f"generic reference; {_no_pressure_reason(equilibrium)}"


def _no_pressure_reason(equilibrium: Path) -> str:
    """Why the equilibrium's own pressure was not usable, in the reader's terms."""
    try:
        import netCDF4  # noqa: PLC0415

        with netCDF4.Dataset(str(equilibrium)) as handle:
            if "presf" not in handle.variables:
                return "the equilibrium carries no pressure profile"
            peak = float(np.max(np.asarray(handle.variables["presf"][:], dtype=float)))
    except Exception:
        return "the equilibrium's pressure could not be read"
    return (f"the equilibrium is a vacuum field, p(0)={peak:.3g} Pa"
            if peak < 1.0 else f"p(0)={peak:.3g} Pa is below the plasma floor")  # fmt: skip


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

    plasma, derived = resolve_plasma(equilibrium)
    if emit:
        emit(f"    plasma ({derived}): n={plasma['n_hat']:.3g}e20 m^-3, "
             f"T={plasma['t_hat']:.3g} keV, dn/ds={plasma['dn_ds']:+.3g}, "
             f"dT/ds={plasma['dt_ds']:+.3g}")  # fmt: skip
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
    show how many there are (one ion root, or the ion/unstable/electron
    triplet), which a Brent solve started from a single guess would hide.

    The interpolation is linear on purpose, and a smarter one is a regression.
    ``J_r = sum_s Z_s Gamma_s`` is a *linear* combination of the moments, and
    linear interpolation is the only scheme that commutes with it: interpolate
    every moment linearly onto the linearly-interpolated root and
    ``sum_s Z_s Gamma_s = 0`` holds exactly there, which is why a Z=+-1 pair
    reports two identical particle fluxes.  A monotone cubic moves the root by
    under 0.01 kV/m on the decks tried and breaks that identity, so the panel
    starts drawing two nearly-coincident Gamma curves that differ only by
    interpolation error.
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
#: Fraction of one panel row reserved for the figure title.
_TITLE_BAND = 0.10

DEFAULT_SURFACES = (0.25, 0.4, 0.55, 0.7, 0.85)

#: E_r values (kV/m) bracketing the ion root at every surface.
DEFAULT_ER_BRACKET = (-8.0, -4.0, -2.0, -1.0, -0.4, 0.4, 1.0, 2.0)


def _interp_at_root(er: np.ndarray, values: np.ndarray, root: float) -> float:
    """Linear interpolation of a scanned quantity onto the ambipolar root."""
    order = np.argsort(er)
    return float(np.interp(root, np.asarray(er)[order], np.asarray(values)[order]))


def _root_fsab2(moments: Mapping[str, Any] | dict[str, Any]) -> float | None:
    """``sqrt(<B^2>)/BBar`` for this surface, from two moments the scan already has.

    ``FSABjHat / FSABjHatOverRootFSAB2`` is exactly ``sqrt(FSABHat2)`` by the
    definitions in documentation eq. (194) and (196), and it is a property of
    the geometry alone, so it is the same at every scanned ``E_r``.  Taking the
    least-squares slope over the whole scan rather than one ratio keeps the
    estimate finite when an individual current passes through zero.
    """
    num = moments.get("FSABjHat")
    den = moments.get("FSABjHatOverRootFSAB2")
    if num is None or den is None:
        return None
    num = np.asarray(num, dtype=float).ravel()
    den = np.asarray(den, dtype=float).ravel()
    denominator = float(np.dot(den, den))
    if not np.isfinite(denominator) or denominator <= 0.0:
        return None
    slope = float(np.dot(num, den)) / denominator
    return slope if np.isfinite(slope) and slope > 0.0 else None


def _vmec_current_kA_m2(geometry: dict[str, Any], radius: float,
                        root_fsab2: float) -> float | None:  # fmt: skip
    """VMEC's own ``<J.B>`` at ``r/a = radius``, in the DKX panel's units.

    The wout ``jdotb`` is ``<J.B>`` in A T/m^2 on the full mesh in ``s``; the
    DKX curve is ``<j.B>/sqrt(<B^2>)`` in kA/m^2, so dividing by the same
    ``sqrt(<B^2>)`` puts the equilibrium and the kinetic current on one axis.
    """
    if "jdotb" not in geometry or root_fsab2 <= 0.0:
        return None
    jdotb = np.asarray(geometry["jdotb"], dtype=float)
    s_grid = np.asarray(geometry["jdotb_s"], dtype=float)
    if jdotb.size < 2:
        return None
    return float(np.interp(radius**2, s_grid, jdotb)) / root_fsab2 / 1.0e3


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
    from dkx.units import CURRENT_DENSITY, HEAT_FLUX, PARTICLE_FLUX  # noqa: PLC0415
    from dkx.units import flux_psi_hat_to_r_hat  # noqa: PLC0415

    er = np.asarray(er_values, dtype=float)
    if emit:
        emit(f"    radial-scan resolution: {DEFAULT_RESOLUTION_PROFILE}; "
             f"{len(er)} Er points per surface")  # fmt: skip
    plasma, _derived = resolve_plasma(equilibrium)
    geometry = equilibrium_scalars(equilibrium)
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
                    value = _interp_at_root(er, np.asarray(boot).ravel(), root)
                    record["bootstrap"] = value
                    # <j.B>/sqrt(<B^2>) carries e nBar vBar (documentation
                    # eq. 196), so this is the same current in kA/m^2.
                    record["bootstrap_kA_m2"] = value * CURRENT_DENSITY / 1.0e3
                root_fsab2 = _root_fsab2(mom)
                if root_fsab2 is not None:
                    record["root_fsab2"] = root_fsab2
                    vmec = _vmec_current_kA_m2(geometry, radius, root_fsab2)
                    if vmec is not None:
                        record["jdotb_vmec_kA_m2"] = vmec
                # Fluxes are computed in the psiHat coordinate; the radial flux
                # density a reader wants is the rHat one (eq. 175), and the RBar
                # in its normalization then cancels (see dkx.units).
                to_r_hat = None
                if "psi_a_hat" in geometry and "a_hat" in geometry:
                    to_r_hat = flux_psi_hat_to_r_hat(
                        psi_a_hat=geometry["psi_a_hat"], a_hat=geometry["a_hat"], r_n=radius
                    )
                for key, name, unit in (
                    ("particleFlux_vm_psiHat", "particle_flux", PARTICLE_FLUX),
                    ("heatFlux_vm_psiHat", "heat_flux", HEAT_FLUX / 1.0e3),
                ):  # fmt: skip
                    if key in mom:
                        arr = np.asarray(mom[key])
                        values = [
                            _interp_at_root(er, arr[:, s], root) for s in range(arr.shape[1])
                        ]
                        record[name] = values
                        if to_r_hat is not None:
                            record[f"{name}_si"] = [v * to_r_hat * unit for v in values]
            if emit:
                e_txt = f"{record.get('er_ambipolar', float('nan')):+.3f}"
                b_txt = f"{record.get('bootstrap_kA_m2', float('nan')):+.4g}"
                emit(f"    r/a={radius:.2f}  Er_ambipolar={e_txt} kV/m  "
                     f"<j.B>/sqrt(<B^2>)={b_txt} kA/m^2")  # fmt: skip
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

    Three curves a reader wants together.  The kinetic
    ``<j.B>/sqrt(<B^2>)`` DKX computes is drawn in kA/m^2 (:mod:`dkx.units`);
    the equilibrium's own current, from the VMEC ``jdotb`` profile divided by
    the same ``sqrt(<B^2>)``, is drawn beside it, because the difference
    between the prescribed and the kinetic current is the whole point of
    running a drift-kinetic code on a finite-beta equilibrium.  The bootstrap
    current is evaluated *at* the ambipolar root, so the root belongs on the
    same panel: it says which radial electric field the current is for.
    """
    pts = [p for p in profiles if "bootstrap" in p or "bootstrap_kA_m2" in p]
    if not pts:
        return False
    dimensional = all("bootstrap_kA_m2" in p for p in pts)
    key = "bootstrap_kA_m2" if dimensional else "bootstrap"
    label = (r"$\langle j_\parallel B\rangle/\sqrt{\langle B^2\rangle}$"
             + (" [kA/m$^2$]" if dimensional else " [SFINCS units]"))  # fmt: skip
    r = [p["r"] for p in pts]
    ax.plot(r, [p[key] for p in pts], "o-", ms=4, color="tab:blue", label="DKX (kinetic)")
    vmec = [(p["r"], p["jdotb_vmec_kA_m2"]) for p in pts if "jdotb_vmec_kA_m2" in p]
    if dimensional and len(vmec) >= 2:
        ax.plot([v[0] for v in vmec], [v[1] for v in vmec], "^:", ms=4,
                color="tab:green", label="VMEC equilibrium")  # fmt: skip
    ax.axhline(0.0, color="0.7", lw=0.8)
    ax.set_xlabel("$r/a$")
    ax.set_ylabel(label, color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax.grid(alpha=0.3)

    twin = ax.twinx()
    er = [p.get("er_ambipolar", float("nan")) for p in pts]
    twin.plot(r, er, "s--", ms=4, color="tab:red", label=r"$E_r$ (ambipolar)")
    twin.set_ylabel(r"$E_r$ [kV/m]", color="tab:red")
    twin.tick_params(axis="y", labelcolor="tab:red")
    # Autoscale would fill the axis with a 6% variation and draw it as a
    # dramatic zigzag.  Floor the span so a nearly-flat root looks nearly flat;
    # a genuinely varying one still fills the axis.
    finite = [v for v in er if np.isfinite(v)]
    if finite:
        centre = 0.5 * (max(finite) + min(finite))
        span = max(max(finite) - min(finite), 0.25 * abs(centre), 0.5)
        twin.set_ylim(centre - 0.6 * span, centre + 0.6 * span)
    ax.set_title("bootstrap current and ambipolar $E_r$", fontsize=9)
    handles = ax.get_lines()[: 2 if (dimensional and len(vmec) >= 2) else 1] + twin.get_lines()[:1]
    # "best" put the box on the Er curve on every device tried; the three curves
    # here all trend downward, so the upper-left corner is the reliable gap.
    ax.legend(handles, [h.get_label() for h in handles], fontsize=7,
              loc="upper left", framealpha=0.9)  # fmt: skip
    return True


def _panel_radial_fluxes(ax, profiles: list[dict[str, Any]],
                         charges: Sequence[float] | None = None) -> bool:  # fmt: skip
    """Particle and heat flux per species against radius, in SI units.

    Named species and a radial axis, rather than a bar chart indexed by species
    number: the convention every neoclassical paper uses, and the only form in
    which "which one is the electron" is answerable from the figure.  The
    values are the radial flux densities ``<Gamma.grad r>`` and ``<Q.grad r>``
    (:mod:`dkx.units`); they differ by orders of magnitude, so the heat flux
    takes a twin axis rather than being flattened onto one log scale.
    """
    # Fall back to SFINCS units unless every surface that has fluxes also has
    # the SI conversion: mixing the two on one axis is worse than either alone.
    with_flux = [p for p in profiles if "particle_flux" in p or "heat_flux" in p]
    dimensional = bool(with_flux) and all(
        "particle_flux_si" in p or "heat_flux_si" in p for p in with_flux
    )
    g_key, q_key = ("particle_flux_si", "heat_flux_si") if dimensional else (
        "particle_flux", "heat_flux")  # fmt: skip
    pts = [p for p in profiles if g_key in p or q_key in p]
    if not pts:
        return False
    r = [p["r"] for p in pts]
    n = max(len(p.get(g_key, p.get(q_key, []))) for p in pts)
    names = _species_labels(n, charges)
    styles = ("o-", "s-", "^-")

    # At the ambipolar root sum_s Z_s Gamma_s = 0, so for a Z=+-1 pair the two
    # particle fluxes are EQUAL by construction -- drawing both puts one line
    # exactly on top of the other, which reads as a rendering fault rather than
    # as the physics it is.  Draw one, and say why.
    gammas = [[p[g_key][s] for p in pts if g_key in p] for s in range(n)]
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
    ax.set_ylabel(r"$\langle\Gamma_s\cdot\nabla r\rangle$"
                  + (" [m$^{-2}$s$^{-1}$]" if dimensional else "  [SFINCS units]"))  # fmt: skip
    ax.grid(alpha=0.3)

    twin = ax.twinx()
    for s in range(n):
        q = [p[q_key][s] for p in pts if q_key in p]
        if q:
            # marker only in the fmt: passing "o-" together with ls="--" makes
            # matplotlib warn that the linestyle is defined twice.
            twin.plot(r[: len(q)], q, marker=styles[s % 3][0], ms=4, ls="--",
                      alpha=0.75, color=f"C{s + n}", label=rf"$Q$ {names[s]}")  # fmt: skip
    twin.set_ylabel(r"$\langle Q_s\cdot\nabla r\rangle$"
                    + (" [kW/m$^2$]" if dimensional else "  [SFINCS units]"))  # fmt: skip
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
    from dkx import units  # noqa: PLC0415

    payload: dict[str, Any] = {
        "equilibrium": str(equilibrium) if equilibrium else "",
        "resolution_monoenergetic": DEFAULT_RESOLUTION,
        "resolution_profile": DEFAULT_RESOLUTION_PROFILE,
        # The SI factors behind every "_kA_m2" / "_si" dataset, so the file can
        # be converted back to SFINCS units without consulting the source.
        "units": {"current_density_A_per_m2": units.CURRENT_DENSITY,
                  "particle_flux_per_m2_s": units.PARTICLE_FLUX,
                  "heat_flux_W_per_m2": units.HEAT_FLUX},
    }
    if scan:
        for key in ("nu_prime", "e_star", "D11", "D31", "D33"):
            payload[f"monoenergetic/{key}"] = [r[key] for r in scan]
    if profiles:
        payload["profiles/r"] = [p["r"] for p in profiles]
        for key in ("er_ambipolar", "bootstrap", "bootstrap_kA_m2",
                    "jdotb_vmec_kA_m2", "root_fsab2"):  # fmt: skip
            payload[f"profiles/{key}"] = [p.get(key, float("nan")) for p in profiles]
        for key in ("particle_flux", "heat_flux", "particle_flux_si", "heat_flux_si"):
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
