from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from .io import read_sfincs_output_file


def _select_x_profile(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr)
    while out.ndim > 1:
        out = out[..., 0]
    return np.asarray(out)


def _surface(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64)
    while out.ndim > 2:
        out = out[..., 0]
    if out.ndim == 1:
        out = out[None, :]
    return out


def _matrix(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64)
    if out.ndim == 0:
        return out.reshape(1, 1)
    if out.ndim == 1:
        return out.reshape(-1, 1)
    while out.ndim > 2:
        out = out[..., 0]
    return out


def _add_heatmap(ax, data: dict[str, object], key: str, *, title: str | None = None) -> bool:
    if key not in data:
        ax.axis("off")
        ax.set_title(title or key)
        ax.text(0.5, 0.5, "not written", ha="center", va="center", color="0.35")
        return False
    arr = _surface(np.asarray(data[key]))
    im = ax.imshow(arr, aspect="auto", origin="lower")
    ax.set_title(title or key)
    ax.set_xlabel("grid index")
    ax.set_ylabel("grid index")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return True



def _theta_zeta_oriented(b_hat: np.ndarray, data: dict[str, object]) -> np.ndarray:
    """Return ``BHat`` as ``(theta, zeta)`` whatever order it was stored in.

    The output file stores it ``(zeta, theta)`` (Fortran layout) while the
    operator hands it over ``(theta, zeta)``, and ``imshow`` puts axis 0 on the
    vertical.  Labelling without checking is how every axisymmetric figure came
    out with its variation on the axis marked zeta: the tokamak's Nzeta=1 strip
    was drawn horizontally and called zeta when it was theta.

    Orientation is decided by matching the array's dimensions against the run's
    own ``Ntheta``/``Nzeta`` rather than by assuming an order.  When the two are
    equal there is nothing to disambiguate, and the stored layout is kept.
    """
    b_hat = np.asarray(b_hat)
    if b_hat.ndim != 2:
        return b_hat
    n_theta = int(np.asarray(data.get("Ntheta", 0)).reshape(-1)[0]) if "Ntheta" in data else 0
    n_zeta = int(np.asarray(data.get("Nzeta", 0)).reshape(-1)[0]) if "Nzeta" in data else 0
    if n_theta and n_zeta and n_theta != n_zeta and b_hat.shape == (n_zeta, n_theta):
        return b_hat.T
    return b_hat

def _add_profile(ax, data: dict[str, object], x: np.ndarray, key: str, *, title: str | None = None) -> bool:
    if key not in data:
        return False
    y = _select_x_profile(np.asarray(data[key], dtype=np.float64)).ravel()
    xx = x[: y.size] if x.size >= y.size else np.arange(y.size, dtype=np.float64)
    ax.plot(xx, y, "o-", lw=1.8, label=key)
    ax.set_title(title or key)
    ax.set_xlabel("x" if x.size >= y.size else "index")
    ax.grid(True, alpha=0.25)
    return True


def _summary_text(data: dict[str, object], input_path: Path) -> str:
    keys = (
        "geometryScheme",
        "RHSMode",
        "collisionOperator",
        "Ntheta",
        "Nzeta",
        "Nx",
        "Nxi",
        "NL",
        "Er",
        "dPhiHatdpsiHat",
        "solverTolerance",
        "VPrimeHat",
        "FSABHat2",
        "NIterations",
    )
    lines = [f"file = {input_path.name}"]
    for key in keys:
        if key not in data:
            continue
        arr = np.asarray(data[key])
        if arr.size != 1:
            continue
        value = arr.reshape(-1)[0]
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float):
            lines.append(f"{key} = {value:.8g}")
        else:
            lines.append(f"{key} = {value}")
    lines.append(f"datasets = {len(data)}")
    return "\n".join(lines)


def _geometry_page(data: dict[str, object], input_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), constrained_layout=True)
    axes = axes.ravel()
    axes[0].axis("off")
    axes[0].text(0.0, 1.0, _summary_text(data, input_path), va="top", ha="left", family="monospace", fontsize=9)
    axes[0].set_title("Run summary")
    _add_heatmap(axes[1], data, "BHat", title=r"$\hat B(\theta,\zeta)$")
    _add_heatmap(axes[2], data, "uHat", title=r"$\hat u(\theta,\zeta)$")
    _add_heatmap(axes[3], data, "dBHatdtheta", title=r"$\partial_\theta \hat B$")
    fig.suptitle("Geometry and normalization diagnostics")
    return fig


def _radial_page(data: dict[str, object]):
    x = np.asarray(data.get("x", np.arange(1.0)), dtype=np.float64).ravel()
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), constrained_layout=True)
    entries = (
        ("FSABFlow_vs_x", r"$\langle B V_\parallel\rangle$ vs $x$"),
        ("particleFlux_vm_psiHat_vs_x", r"magnetic-drift particle flux vs $x$"),
        ("heatFlux_vm_psiHat_vs_x", r"magnetic-drift heat flux vs $x$"),
        ("transportMatrix", "transport matrix entries"),
    )
    for ax, (key, title) in zip(axes.ravel(), entries, strict=False):
        if key == "transportMatrix" and key in data:
            im = ax.imshow(_matrix(np.asarray(data[key])), aspect="auto", origin="lower")
            ax.set_title(title)
            ax.set_xlabel("drive index")
            ax.set_ylabel("flux index")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            continue
        ok = _add_profile(ax, data, x, key, title=title)
        if not ok:
            ax.axis("off")
            ax.set_title(title)
            ax.text(0.5, 0.5, "not written", ha="center", va="center", color="0.35")
    fig.suptitle("Transport diagnostics")
    return fig


def _flux_page(data: dict[str, object]):
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), constrained_layout=True)
    entries = (
        ("particleFlux_vm_psiHat", r"$\Gamma_{vm}$"),
        ("heatFlux_vm_psiHat", r"$Q_{vm}$"),
        ("momentumFlux_vm_psiHat", r"$\Pi_{vm}$"),
        ("NTV", "NTV"),
    )
    for ax, (key, title) in zip(axes.ravel(), entries, strict=False):
        if key not in data:
            ax.axis("off")
            ax.set_title(title)
            ax.text(0.5, 0.5, "not written", ha="center", va="center", color="0.35")
            continue
        arr = _matrix(np.asarray(data[key]))
        im = ax.imshow(arr, aspect="auto", origin="lower")
        ax.set_title(title)
        ax.set_xlabel("RHS / iteration")
        ax.set_ylabel("species")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Flux, momentum, and neoclassical-toroidal-viscosity outputs")
    return fig


def _moment_page(data: dict[str, object]):
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.6), constrained_layout=True)
    entries = (
        ("densityPerturbation", r"$n_1$"),
        ("pressurePerturbation", r"$p_1$"),
        ("pressureAnisotropy", "pressure anisotropy"),
        ("flow", "parallel flow"),
        ("MachUsingFSAThermalSpeed", "Mach number"),
        ("jHat", r"$\hat j_\parallel$"),
    )
    for ax, (key, title) in zip(axes.ravel(), entries, strict=False):
        _add_heatmap(ax, data, key, title=title)
    fig.suptitle("Distribution-function moments")
    return fig


def plot_sfincs_output_summary(
    *,
    input_h5: Path,
    output_png: Path,
) -> Path:
    """Write summary plots for a SFINCS output file.

    If ``output_png`` ends in ``.pdf``, a multi-page diagnostics panel is written.
    Otherwise a compact single-page raster/vector summary is written.
    """
    input_h5 = Path(input_h5)
    data = read_sfincs_output_file(input_h5)
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    if output_png.suffix.lower() == ".pdf":
        with PdfPages(output_png) as pdf:
            for page in (
                _geometry_page(data, input_h5),
                _radial_page(data),
                _flux_page(data),
                _moment_page(data),
            ):
                pdf.savefig(page, bbox_inches="tight")
                plt.close(page)
        return output_png.resolve()

    x = np.asarray(data["x"]).ravel()
    zeta = np.asarray(data["zeta"]).ravel()
    b_hat = np.asarray(data["BHat"])
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), constrained_layout=True)

    if "FSABFlow_vs_x" in data:
        flow_vs_x = _select_x_profile(np.asarray(data["FSABFlow_vs_x"]))
        axes[0].plot(x, np.asarray(flow_vs_x).ravel(), "o-", lw=1.8)
        axes[0].set_title("Flow profile vs x")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("FSABFlow_vs_x")
        axes[0].grid(True, alpha=0.25)
    else:
        oriented = _theta_zeta_oriented(b_hat, data)
        theta = np.asarray(data.get("theta", np.arange(oriented.shape[0]))).ravel()
        axes[0].plot(theta, oriented[:, 0], lw=1.8)
        axes[0].set_title("BHat(theta, zeta=0)")
        axes[0].set_xlabel("theta")
        axes[0].set_ylabel("BHat")
        axes[0].grid(True, alpha=0.25)

    if "heatFlux_vm_psiHat_vs_x" in data:
        heat_vs_x = _select_x_profile(np.asarray(data["heatFlux_vm_psiHat_vs_x"]))
        axes[1].plot(x, np.asarray(heat_vs_x).ravel(), "o-", lw=1.8, color="#b45309")
        axes[1].set_title("Heat-flux profile vs x")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("heatFlux_vm_psiHat_vs_x")
        axes[1].grid(True, alpha=0.25)
    else:
        info_lines = []
        for key in ("geometryScheme", "VPrimeHat", "FSABHat2", "Ntheta", "Nzeta", "Nx"):
            if key not in data:
                continue
            value = np.asarray(data[key]).reshape(-1)[0]
            info_lines.append(f"{key} = {value}")
        axes[1].axis("off")
        axes[1].text(
            0.02,
            0.98,
            "\n".join(info_lines) if info_lines else "Geometry-only output",
            va="top",
            ha="left",
            family="monospace",
        )
        axes[1].set_title("Run summary")

    oriented = _theta_zeta_oriented(b_hat, data)
    im = axes[2].imshow(oriented, aspect="auto", origin="lower")
    axes[2].set_title(r"$\hat B(\theta, \zeta)$")
    axes[2].set_xlabel(r"$\zeta$ index")
    axes[2].set_ylabel(r"$\theta$ index")
    if zeta.size and oriented.shape[-1] == zeta.size:
        axes[2].set_xticks([0, max(0, oriented.shape[-1] - 1)])
        axes[2].set_xticklabels(["0", f"{float(zeta[-1]):.2f}"])
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    # y=1.03 put the title outside the figure, where constrained_layout does
    # not reserve room for it, so it landed on the subplot titles.  Let the
    # layout engine place it.
    fig.suptitle(f"SFINCS output summary: {Path(input_h5).name}")
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_png.resolve()



def _shown(written: Path) -> Path:
    """Display an already-written figure, for IDE and notebook users.

    The panel builders save and close their own figure, so there is nothing
    left to ``plt.show()``.  Re-reading the PNG and showing it is what puts it
    in Spyder's plots pane, and it costs one imread.
    """
    try:
        image = plt.imread(written)
    except Exception:  # pragma: no cover - unreadable/exotic format
        return written
    figure = plt.figure(figsize=(13.5, 8.0))
    axis = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    axis.imshow(image)
    axis.axis("off")
    # block=False so a script run from a terminal is not held open by a
    # window nobody is there to close; Spyder and Jupyter render regardless.
    plt.show(block=False)
    return written


def plot(source, out="dkx_panels.png", *, style="panels", show=False):
    """Plot a run or an output file.  One call, one figure, returns its path.

    ``source`` is whatever you already have:

    - the object :func:`dkx.run` returned, when the run was given ``out=``;
    - a path to an ``sfincsOutput`` file, DKX's or Fortran SFINCS's --- the
      layout is the same, so both work;
    - a directory containing one.

    The figure is the same six-panel one ``dkx --plot`` writes: the
    monoenergetic coefficients, ``|B|`` on the surface, the bootstrap current
    and the species fluxes.  ``out`` picks the format by suffix.

    ``style="summary"`` selects the compact three-panel page instead.  Only
    that style expands with a ``.pdf`` suffix, into a four-page diagnostics
    book; ``style="panels"`` writes its one figure whatever the suffix.

    For a figure this does not draw, read the numbers off ``run.moments`` and
    use matplotlib directly --- ``examples/1_basics/plot_custom.py`` shows
    that, and it is not a fallback so much as the normal way to make a figure
    for a paper.

    Args:
        source: a run object, an output-file path, or a directory.
        out: destination path; the suffix selects png or pdf.  A ``.pdf``
            is multi-page only under ``style="summary"``.
        style: ``"panels"`` (default) for the ``dkx --plot`` figure, or
            ``"summary"`` for the compact three-panel page.
        show: also display the figure.  For Spyder, Jupyter or any IDE where
            you want it in the plots pane rather than only on disk; a no-op
            under a headless backend.

    Returns:
        The :class:`~pathlib.Path` actually written.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    path = getattr(source, "output_path", None) or (
        source if isinstance(source, (str, Path)) else None)
    if path is not None:
        path = Path(path)
        if path.is_dir():
            candidates = sorted(
                p for p in path.iterdir()
                if p.suffix in {".h5", ".nc", ".npz"} and "sfincsOutput" in p.name
            )
            if not candidates:
                raise FileNotFoundError(f"no sfincsOutput file in {path}")
            path = candidates[0]
        if style == "summary":
            written = plot_sfincs_output_summary(input_h5=path, output_png=out)
            return _shown(written) if show else written
        if style != "panels":
            raise ValueError(f"style must be 'panels' or 'summary'; got {style!r}")
        from dkx.representative import plot_output_file  # noqa: PLC0415

        written = plot_output_file(path, out)
        return _shown(written) if show else written

    if hasattr(source, "moments"):
        raise ValueError(
            "this run has no output file to plot: pass out=... to dkx.run so the "
            "solve writes one, e.g.\n"
            '    run = dkx.run(case, out="sfincsOutput.h5")\n'
            "    dkx.plot(run)\n"
            "Plotting cannot reuse the in-memory result because the panels read "
            "the output file's full dataset, not just the moments."
        )
    raise TypeError(
        "plot() takes the object dkx.run() returned, a path to an sfincsOutput "
        f"file, or a directory holding one; got {type(source).__name__}"
    )


def plot_result_summary(*, result, output_path: Path) -> Path:
    """Write a radial-profile panel for a dkx Result.

    One row per admitted observable against ``r_N``, with one line per species
    where the array carries a species axis. The panel is deliberately plain:
    it exists so ``dkx run`` output can be looked at without writing a script,
    not to be a publication figure.

    A Result from the ambipolar workflow also gets its admitted roots drawn
    against ``r_N``, marked by branch. Unstable roots are drawn hollow --
    they solve ``J_r = 0`` but a plasma does not sit on them, and a filled
    marker would imply otherwise.
    """
    arrays = result.arrays
    radius = np.asarray(arrays.get("r_N", []), dtype=float).ravel()
    if radius.size == 0:
        raise ValueError(
            f"{getattr(result, 'case_name', 'result')} carries no r_N axis, so there "
            "is no radial coordinate to plot against"
        )

    panels = [
        ("particle_flux_m2_s", r"particle flux  [m$^{-2}$s$^{-1}$]"),
        ("heat_flux_W_m2", r"heat flux  [W m$^{-2}$]"),
        ("parallel_current_A_T_m2", r"$\langle j_\parallel B\rangle$  [A T m$^{-2}$]"),
    ]
    present = [(key, label) for key, label in panels if key in arrays]
    has_roots = "ambipolar_root_kV_m" in arrays
    rows = len(present) + (1 if has_roots else 0)
    if rows == 0:
        raise ValueError(
            f"{getattr(result, 'case_name', 'result')} carries none of "
            f"{[k for k, _ in panels]}, so there is nothing to plot"
        )

    species = [str(s) for s in np.asarray(arrays.get("species", [])).ravel()]
    fig, axes = plt.subplots(rows, 1, figsize=(7.0, 2.4 * rows), sharex=True)
    axes = np.atleast_1d(axes)

    for ax, (key, label) in zip(axes, present):
        values = np.asarray(arrays[key], dtype=float)
        if values.ndim == 2 and values.shape[0] == radius.size:
            for index in range(values.shape[1]):
                name = species[index] if index < len(species) else f"species {index}"
                ax.plot(radius, values[:, index], marker="o", ms=3, label=name)
            ax.legend(fontsize="small", frameon=False)
        else:
            ax.plot(radius, values.ravel()[: radius.size], marker="o", ms=3)
        ax.set_ylabel(label, fontsize="small")
        ax.grid(alpha=0.3)

    if has_roots:
        ax = axes[len(present)]
        fields = np.asarray(arrays["ambipolar_root_kV_m"], dtype=float)
        counts = np.asarray(arrays["ambipolar_root_count"]).astype(int).ravel()
        kinds = np.asarray(arrays["ambipolar_root_type"])
        any_unstable = False
        for surface in range(min(fields.shape[0], radius.size)):
            for index in range(int(counts[surface])):
                kind = kinds[surface, index]
                kind = kind.decode() if isinstance(kind, bytes) else str(kind)
                unstable = kind == "unstable"
                any_unstable = any_unstable or unstable
                ax.plot(
                    radius[surface], fields[surface, index],
                    marker="o", ms=6,
                    markerfacecolor="none" if unstable else None,
                    color={"ion": "tab:blue", "electron": "tab:red"}.get(kind, "tab:grey"),
                )
        ax.axhline(0.0, color="k", lw=0.6, alpha=0.5)
        ax.set_ylabel(r"$E_r$ roots  [kV m$^{-1}$]", fontsize="small")
        ax.grid(alpha=0.3)
        if any_unstable:
            # Only when one is drawn: a legend for a marker style that does not
            # appear reads as a missing feature rather than an absent branch.
            ax.text(
                0.99, 0.97, "hollow = unstable branch", transform=ax.transAxes,
                ha="right", va="top", fontsize="x-small", alpha=0.7,
            )

    axes[-1].set_xlabel(r"$r_N$")
    fig.suptitle(getattr(result, "case_name", "dkx result"), fontsize="medium")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


#: Elementary charge, for assembling the radial current from species fluxes.
_ELEMENTARY_CHARGE = 1.602176634e-19

#: Marker colour per admitted root classification.
_ROOT_COLOURS = {"ion": "tab:blue", "electron": "tab:red", "unstable": "tab:grey"}


def plot_ambipolar_search(*, result, output_path: Path) -> Path:
    r"""Plot the radial current against the electric field, one panel per surface.

    plan.md section 10.1 asks for "radial current versus electric field with
    every evaluation, bracket, root type, selected branch, and failed
    interval". This is that family, and it exists because DKX's admitted
    ambipolar results are scoped to explicit sampled intervals: the claim is
    only honest if a reader can see where the scan actually looked.

    The radial current is assembled as :math:`J_r = \sum_s Z_s e \Gamma_s`
    from the per-evaluation species fluxes, because the result stores
    ``radial_current_A_m2`` only at the accepted answer, not along the search.

    Two contract points from section 10.3 are load-bearing:

    * an evaluation that did not produce a number is drawn on a rug at the
      bottom rather than dropped or plotted as zero -- a failed solve at a
      given field is information about the search, and rendering it as
      :math:`J_r = 0` would put a fake root there;
    * a surface with no admitted roots says so on the panel, together with the
      reason a null result is not proof, rather than showing an empty axis.
    """
    arrays = result.arrays
    required = ("evaluation_electric_field_kV_m", "evaluation_particle_flux_m2_s")
    missing = [name for name in required if name not in arrays]
    if missing:
        raise ValueError(
            f"{getattr(result, 'case_name', 'result')} carries no ambipolar search "
            f"(missing {missing}). This plot needs workflow = 'ambipolar_profile'."
        )

    fields = np.asarray(arrays["evaluation_electric_field_kV_m"], dtype=float)
    fluxes = np.asarray(arrays["evaluation_particle_flux_m2_s"], dtype=float)
    charges = np.asarray(arrays.get("charge_e", []), dtype=float)
    if charges.size == 0:
        raise ValueError("result carries no species charges; cannot form a radial current")

    current = np.einsum("sek,k->se", fluxes, charges) * _ELEMENTARY_CHARGE

    roots = np.asarray(arrays.get("ambipolar_root_kV_m", np.zeros((fields.shape[0], 0))))
    counts = np.asarray(arrays.get("ambipolar_root_count", np.zeros(fields.shape[0]))).astype(int).ravel()
    kinds = np.asarray(arrays.get("ambipolar_root_type", np.empty(roots.shape, dtype=object)))
    brackets = arrays.get("ambipolar_root_bracket_kV_m")
    selected = np.asarray(arrays.get("selected_ambipolar_root", [])).ravel()
    radius = np.asarray(arrays.get("r_N", []), dtype=float).ravel()

    n_surface = fields.shape[0]
    fig, axes = plt.subplots(n_surface, 1, figsize=(7.0, 3.0 * n_surface), squeeze=False)
    axes = axes[:, 0]

    for surface, ax in enumerate(axes):
        x = fields[surface]
        y = current[surface]
        finite = np.isfinite(x) & np.isfinite(y)
        failed = np.isfinite(x) & ~np.isfinite(y)

        order = np.argsort(x[finite])
        ax.plot(x[finite][order], y[finite][order], "-o", ms=3, lw=1.0,
                color="0.35", label=r"$J_r$ evaluations", zorder=2)
        ax.axhline(0.0, color="k", lw=0.7, alpha=0.6, zorder=1)

        if brackets is not None:
            band = np.asarray(brackets, dtype=float)
            for index in range(int(counts[surface])):
                lo, hi = band[surface, index]
                if np.isfinite(lo) and np.isfinite(hi):
                    ax.axvspan(lo, hi, color="tab:orange", alpha=0.18, zorder=0,
                               label="bracket" if index == 0 else None)

        for index in range(int(counts[surface])):
            kind = kinds[surface, index]
            kind = kind.decode() if isinstance(kind, bytes) else str(kind)
            is_selected = selected.size > surface and selected[surface] == index
            ax.plot(
                roots[surface, index], 0.0, marker="o", ms=10 if is_selected else 7,
                markerfacecolor="none" if kind == "unstable" else _ROOT_COLOURS.get(kind, "tab:green"),
                markeredgecolor=_ROOT_COLOURS.get(kind, "tab:green"),
                markeredgewidth=2.0 if is_selected else 1.2, linestyle="none", zorder=4,
                label=f"{kind} root" if index == 0 else None,
            )

        if failed.any():
            # A rug, not a y value: a failed solve has no current to place.
            ax.plot(x[failed], np.full(failed.sum(), ax.get_ylim()[0]), marker="|",
                    linestyle="none", color="tab:red", ms=10,
                    label="evaluation failed", zorder=3)

        if int(counts[surface]) == 0:
            ax.text(
                0.5, 0.5,
                "no root admitted in the sampled interval\n"
                "(sign sampling cannot see a tangential root,\n"
                "or an even number of crossings between samples)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize="small", color="tab:red", alpha=0.85,
            )

        label = f"surface {surface}"
        if radius.size > surface:
            label += rf"  ($r_N = {radius[surface]:.3g}$)"
        ax.set_title(label, fontsize="small")
        ax.set_ylabel(r"$J_r$  [A m$^{-2}$]", fontsize="small")
        ax.grid(alpha=0.3)
        ax.legend(fontsize="x-small", frameon=False, loc="best")

    axes[-1].set_xlabel(r"$E_r$  [kV m$^{-1}$]")
    fig.suptitle(f"{getattr(result, 'case_name', 'dkx result')} — ambipolar search",
                 fontsize="medium")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
