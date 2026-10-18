"""Figure panels and plotting entry points for the representative run."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

#: Fraction of one panel row reserved for the figure title.
_TITLE_BAND = 0.10


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
        pts = sorted(
            (r for r in records if r["e_star"] == e), key=lambda r: r["nu_prime"]
        )
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


def figure_caption(plasma: dict | None, plasma_source: str = "",
                   resolutions: dict[str, dict] | None = None) -> str:  # fmt: skip
    """The two-line footer: what plasma, at what resolution, and how it was got.

    A reader cannot judge a transport number without the ``n``, ``T`` and grid
    behind it, and the split of pressure into ``n`` and ``T`` is an assumption
    rather than a measurement --- so the caption says so.  Without that clause
    the bootstrap panel reads as DKX disagreeing with VMEC, when on the
    precise-QA reference the same solver gives DKX/VMEC = 1.04-1.23 with that
    equilibrium's own profiles and 0.42 with this figure's closure.
    """
    from .representative import plasma_summary  # noqa: PLC0415

    bits = []
    if plasma:
        bits.append(plasma_summary(plasma))
    for name, res in (resolutions or {}).items():
        bits.append(f"{name} " + "x".join(str(v) for v in res.values()))
    caption = "   |   ".join(bits)
    if plasma:
        source = plasma_source or (
            "p(s) from the equilibrium" if "p_pa" in plasma else "generic reference"
        )
        caption += f"\nplasma source: {source}"
        if not source.startswith("explicit"):
            caption += ("   |   n and T are an assumed split of p,"
                        " so the kinetic and equilibrium currents need not coincide")  # fmt: skip
    return caption


def _profile_panel_absence(profiles: list[dict[str, Any]], name: str) -> str:
    """Explain why a physical radial panel could not be drawn.

    The categories are deliberately stable output vocabulary: unavailable
    physical inputs, an unreadable/malformed VMEC profile, a kinetic solve
    failure, and a completed scan with no observed bracket are materially
    different outcomes and must not collapse into ``not present``.
    """
    if not profiles:
        return f"{name}: no radial-profile evidence"

    parser_failures = [
        p for p in profiles if p.get("profile_input_status") == "parser_failure"
    ]
    if parser_failures:
        detail = str(
            parser_failures[0].get("profile_input_detail", "VMEC profile unreadable")
        )
        return f"{name}: VMEC parser failure\n{detail}"

    unavailable = [
        p for p in profiles if p.get("profile_input_status") == "physics_unavailable"
    ]
    if unavailable:
        detail = str(
            unavailable[0].get("profile_input_detail", "physical profiles unavailable")
        )
        return f"{name}: physical profiles unavailable\n{detail}"

    solve_failures = [
        p for p in profiles if p.get("evaluation_status") == "solve_failure"
    ]
    if solve_failures:
        kinds = sorted({str(p.get("failure_type", "unknown")) for p in solve_failures})
        return f"{name}: solve failure\n" + ", ".join(kinds)

    if any(p.get("evaluation_status") == "no_bracketed_root" for p in profiles):
        return f"{name}: no bracket observed\nand no evaluated moments were retained"
    return f"{name}: requested observable absent\nfrom completed solves"


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
        axes["ambi"] = fig.add_subplot(gs[2, :])  # one panel, the whole row

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
            axes[key].set_xticks([])
            axes[key].set_yticks([])
    for key, name in (("modb", "modB"), ("boot", "bootstrap"), ("flux", "fluxes")):
        if not drawn[name]:
            message = (
                _profile_panel_absence(profiles or [], name)
                if key in {"boot", "flux"} and profiles
                else f"{name}: not present\nin this output"
            )
            axes[key].text(0.5, 0.5, message,
                           ha="center", va="center", fontsize=8, color="0.4",
                           transform=axes[key].transAxes)  # fmt: skip
            axes[key].set_xticks([])
            axes[key].set_yticks([])

    # State the plasma and the resolutions on the figure: a reader cannot judge
    # a transport number without knowing the n, T and grid behind it, and the
    # split of pressure into n and T is an assumption, not a measurement.
    if plasma or resolutions:
        import textwrap  # noqa: PLC0415

        # Wrap rather than let a long provenance line run off both canvas edges.
        caption = "\n".join(textwrap.fill(line, 170) for line in
                            figure_caption(plasma, plasma_source, resolutions).split("\n"))  # fmt: skip
        # Reserve the strip rather than drawing into it: constrained_layout owns
        # the whole canvas, so a bare fig.text lands on top of the bottom row's
        # x-labels.  Older matplotlib has no layout engine to ask; there the
        # caption goes under the figure and the axes keep their room.
        band = 0.018 * caption.count("\n") + 0.030
        engine = getattr(fig, "get_layout_engine", lambda: None)()
        if engine is not None:
            # Reserve the title strip as well: rect hands the engine the whole
            # remaining canvas, and suptitle is not one of the artists it packs.
            engine.set(rect=(0.0, band, 1.0, 1.0 - band - _TITLE_BAND / rows))
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
    # A single solve carries no (nuPrime, EStar) scan, so the representative
    # layout's whole top row would say "not present" three times.  Give that
    # space to what a single run does have instead.
    if not any(key in data for key in ("transportMatrix", "nuPrime")):
        return plot_single_run(out, data, title=f"DKX panels — {path.name}")
    return plot_representative(out, data=data, title=f"DKX panels — {path.name}")


def plot_single_run(out_path: str | Path, data: dict[str, Any],
                    title: str = "DKX run") -> Path:  # fmt: skip
    """Six panels for one solved deck: geometry, speed profiles, moments.

    The counterpart of :func:`plot_representative` for a single output file.
    Same six-panel shape, but the top row carries the run's own summary and its
    speed-resolved profiles rather than a monoenergetic scan it does not have.
    """
    plt = _import_matplotlib()
    fig = plt.figure(figsize=(13.5, 8.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    axes = [fig.add_subplot(gs[r, c]) for r in (0, 1) for c in (0, 1, 2)]

    _panel_run_summary(axes[0], data)
    _panel_vs_x(axes[1], data, "FSABFlow_vs_x", r"$\langle B V_\parallel\rangle$")
    _panel_vs_x(
        axes[2], data, "particleFlux_vm_psiHat_vs_x", r"$\Gamma$ (magnetic drift)"
    )
    if not _panel_modB(axes[3], data):
        axes[3].text(0.5, 0.5, "no |B| in this output", ha="center", va="center",
                     fontsize=8, color="0.4", transform=axes[3].transAxes)  # fmt: skip
        axes[3].set_xticks([])
        axes[3].set_yticks([])
    _panel_vs_x(axes[4], data, "heatFlux_vm_psiHat_vs_x", r"$Q$ (magnetic drift)")
    _panel_bootstrap(axes[5], data)

    fig.suptitle(title, fontsize=12)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path.resolve()


def _panel_run_summary(ax, data: dict[str, Any]) -> bool:
    """The scalars a reader checks first: which case, which grid, did it solve."""
    rows = []
    for key, label in (("geometryScheme", "geometryScheme"), ("Ntheta", "Ntheta"),
                       ("Nzeta", "Nzeta"), ("Nxi", "Nxi"), ("Nx", "Nx"),
                       ("Delta", "Delta"), ("nu_n", "nu_n"),
                       ("VPrimeHat", "VPrimeHat"), ("FSABHat2", "FSABHat2"),
                       ("linearSolverResidualNorm", "residual")):  # fmt: skip
        if key not in data:
            continue
        value = np.asarray(data[key]).ravel()
        if value.size:
            item = value[-1]
            rows.append(f"{label:<16s} {item:.6g}" if abs(item) < 1e5 else
                        f"{label:<16s} {item:.4e}")  # fmt: skip
    ax.axis("off")
    ax.text(0.0, 1.0, "\n".join(rows) or "no scalars in this output",
            va="top", ha="left", family="monospace", fontsize=9,
            transform=ax.transAxes)  # fmt: skip
    ax.set_title("run summary", fontsize=9)
    return bool(rows)


def _panel_vs_x(ax, data: dict[str, Any], key: str, label: str) -> bool:
    """One speed-resolved profile, one curve per species."""
    if key not in data or "x" not in data:
        ax.text(0.5, 0.5, f"{key}\nnot in this output", ha="center", va="center",
                fontsize=8, color="0.4", transform=ax.transAxes)  # fmt: skip
        ax.set_xticks([])
        ax.set_yticks([])
        return False
    x = np.asarray(data["x"], dtype=float).ravel()
    arr = np.asarray(data[key], dtype=float)
    arr = arr.reshape(arr.shape[0], -1) if arr.ndim > 1 else arr.reshape(-1, 1)
    names = _species_labels(arr.shape[1])
    for column in range(arr.shape[1]):
        ax.plot(x[: arr.shape[0]], arr[:, column], "o-", ms=3, label=names[column])
    ax.set_xlabel("$x = v / v_{th}$")
    ax.set_ylabel(label)
    ax.grid(alpha=0.3)
    if arr.shape[1] > 1:
        ax.legend(fontsize=7)
    ax.set_title(key, fontsize=9)
    return True


def _panel_ambipolarity(ax, records: list[dict[str, Any]]) -> bool:
    """``J_r(E_r)`` with each root labelled ion / unstable / electron.

    ``records`` is either one scan (``{"er", "J_r"}`` rows) or the radial
    profiles, which carry ``er_scan``/``J_r``/``root_types`` per surface; the
    latter are drawn normalized to each surface's ``max|J_r|``.
    """
    if records and "er_scan" in records[0]:
        curves = [(p["er_scan"], p["J_r"], p.get("root_types"), f"r/a={p['r']:.2f}")
                  for p in records if len(p.get("er_scan", ())) > 1]  # fmt: skip
        ylabel = r"$J_r / \max|J_r|$"
    else:
        curves = [([r["er"] for r in records], [r["J_r"] for r in records], None, None)]
        ylabel = r"$J_r$"
    if not records or not curves:
        return False
    from .representative import _ambipolar_roots, _root_types  # noqa: PLC0415

    ax.axhline(0.0, color="0.5", lw=0.8)
    for er, jr, types, label in curves:
        er, jr = np.asarray(er, dtype=float), np.asarray(jr, dtype=float)
        order = np.argsort(er)
        er, jr = er[order], jr[order]
        if label is not None:
            jr = jr / max(float(np.nanmax(np.abs(jr))), np.finfo(float).tiny)
        (line,) = ax.plot(er, jr, "o-", ms=3, label=label)
        roots = _ambipolar_roots([{"er": e, "J_r": j} for e, j in zip(er, jr)])
        types = types if types and len(types) == len(roots) else _root_types(er, jr)
        for root, kind in zip(roots, types):
            ax.axvline(root, color=line.get_color(), ls="--", lw=0.8)
            ax.annotate(f"{kind} {root:+.1f}", (root, 0.0), fontsize=7,
                        color=line.get_color(), rotation=90, xytext=(2, 6),
                        textcoords="offset points")  # fmt: skip
    if curves[0][3] is not None:
        ax.legend(fontsize=7)
    ax.set_xlabel(r"$E_r$ [kV/m]")
    ax.set_ylabel(ylabel)
    ax.set_title("ambipolarity: roots of $J_r(E_r)$", fontsize=9)
    ax.grid(alpha=0.3)
    return True


# ---------------------------------------------------------------------------
# Radial profiles at the ambipolar root
# ---------------------------------------------------------------------------
#: Flux-surface labels for the radial scan.  Five is enough to show a profile's
#: shape; the cost is one batched E_r scan per surface.
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

    **The two curves are not expected to coincide, and the reason is stated on
    the figure.**  DKX runs the assumed ``n``/``T`` split from
    :func:`plasma_parameters`, not the profiles the equilibrium was actually
    built with, and the bootstrap current is roughly linear in the temperature.
    Measured on the precise-QA finite-beta reference at ``s = 0.25/0.5/0.75``:
    with that equilibrium's own Landreman-Buller-Drevlak profiles
    (``T_0 = 5`` keV) DKX/VMEC is 1.04, 1.07, 1.23; with this figure's closure
    (``T_0 = 2`` keV, ``T ~ p^(1/3)``) it is 0.42, 0.41, 0.42.  A factor of 2.7
    from the assumed temperature alone, with the solver unchanged --- so a gap
    of that size here is the closure, and a gap at *matched* profiles would be
    the physics.
    """
    pts = [p for p in profiles if "bootstrap" in p or "bootstrap_kA_m2" in p]
    if not pts:
        return False
    dimensional = all("bootstrap_kA_m2" in p for p in pts)
    key = "bootstrap_kA_m2" if dimensional else "bootstrap"
    label = (r"$\langle j_\parallel B\rangle/\sqrt{\langle B^2\rangle}$"
             + (" [kA/m$^2$]" if dimensional else " [SFINCS units]"))  # fmt: skip
    r = [p["r"] for p in pts]
    ax.plot(
        r, [p[key] for p in pts], "o-", ms=4, color="tab:blue", label="DKX (kinetic)"
    )
    vmec = [(p["r"], p["jdotb_vmec_kA_m2"]) for p in pts if "jdotb_vmec_kA_m2" in p]
    if dimensional and len(vmec) >= 2:
        ax.plot([v[0] for v in vmec], [v[1] for v in vmec], "^:", ms=4,
                color="tab:green", label="VMEC equilibrium")  # fmt: skip
    redl = [(p["r"], p["jdotb_redl_kA_m2"]) for p in pts if "jdotb_redl_kA_m2" in p]
    if dimensional and len(redl) >= 2:
        ax.plot([v[0] for v in redl], [v[1] for v in redl], "v-.", ms=4,
                color="tab:orange", label="Redl (same profiles)")  # fmt: skip
    ax.axhline(0.0, color="0.7", lw=0.8)
    ax.set_xlabel("$r/a$")
    ax.set_ylabel(label, color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax.grid(alpha=0.3)

    twin = ax.twinx()
    prescribed = all("er_prescribed" in p for p in pts)
    all_roots = prescribed or all(
        bool(p.get("evaluation_is_root", "er_ambipolar" in p)) for p in pts)
    er = [p.get("er_ambipolar", p.get("er_evaluated", float("nan"))) for p in pts]
    er_label = (r"$E_r$ (prescribed)" if prescribed else r"$E_r$ (ambipolar)" if all_roots
                else r"$E_r$ (root / closest scanned)")  # fmt: skip
    twin.plot(r, er, "s--", ms=4, color="tab:red", label=er_label)
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
    title = (
        "bootstrap current and ambipolar $E_r$"
        if all_roots
        else ("bootstrap current; open squares use closest scanned $E_r$")
    )
    ax.set_title(title, fontsize=9)
    if not all_roots:
        twin.scatter(
            [p["r"] for p in pts if not p.get("evaluation_is_root", True)],
            [
                p.get("er_evaluated", float("nan"))
                for p in pts
                if not p.get("evaluation_is_root", True)
            ],
            marker="s",
            facecolors="none",
            edgecolors="tab:red",
            zorder=4,
        )
    handles = [h for h in ax.get_lines() if not h.get_label().startswith("_")]
    handles += twin.get_lines()[:1]
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
    all_roots = all(bool(p.get("evaluation_is_root", True)) for p in pts)
    ambipolar_pair = (
        all_roots and n == 2 and gammas[0] and gammas[1]
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
    ax.set_title(
        "particle and heat flux at the ambipolar root"
        if all_roots
        else "fluxes at roots / explicitly flagged closest scanned $E_r$",
        fontsize=9,
    )
    handles = ax.get_lines() + twin.get_lines()
    ax.legend(handles, [h.get_label() for h in handles], fontsize=6, loc="best", ncol=2)
    return True

