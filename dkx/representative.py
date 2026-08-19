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
DEFAULT_RESOLUTION = {"n_theta": 25, "n_zeta": 41, "n_xi": 20}

#: Monoenergetic scan grid.  Five collisionalities over two decades is what a
#: D11* figure needs to show the plateau and the 1/nu branch; three ``EStar``
#: values show the electric-field dependence without tripling the runtime.
DEFAULT_NU_PRIME = (1.0e-2, 3.0e-2, 1.0e-1, 3.0e-1, 1.0e0)
DEFAULT_E_STAR = (0.0, 0.1, 0.3)


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
            ax.plot(nu, [abs(r[key]) for r in pts], "o-", ms=3, label=f"$E^*$={e:g}")
    for ax, key in ((ax_d11, "D_{11}"), (ax_d31, "D_{31}"), (ax_d33, "D_{33}")):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\nu'$")
        ax.set_ylabel(rf"$|{key}|$")
        ax.grid(alpha=0.3, which="both")
    ax_d11.legend(fontsize=7)
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
    ax.bar(range(len(v)), v, color="tab:blue")
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.set_xlabel("iteration" if len(v) > 1 else "")
    ax.set_ylabel(r"$\langle j_\parallel B\rangle/\sqrt{\langle B^2\rangle}$")
    ax.set_title(f"bootstrap current = {v[-1]:+.4e}", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    if len(v) == 1:
        ax.set_xticks([])
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
    fig = plt.figure(figsize=(13.0, 7.6))
    gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.30)
    axes = {
        "d11": fig.add_subplot(gs[0, 0]), "d31": fig.add_subplot(gs[0, 1]),
        "d33": fig.add_subplot(gs[0, 2]), "modb": fig.add_subplot(gs[1, 0]),
        "boot": fig.add_subplot(gs[1, 1]), "flux": fig.add_subplot(gs[1, 2]),
    }  # fmt: skip

    drawn = {}
    drawn["monoenergetic"] = _panel_monoenergetic(
        axes["d11"], axes["d31"], axes["d33"], scan or []
    )
    drawn["modB"] = _panel_modB(axes["modb"], data)
    drawn["bootstrap"] = _panel_bootstrap(axes["boot"], data)
    drawn["fluxes"] = _panel_fluxes(axes["flux"], data)

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

    fig.suptitle(title, fontsize=12)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
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
    nu = DEFAULT_NU_PRIME if not full else tuple(np.logspace(-2.5, 0.5, 9))
    scan = monoenergetic_scan(base, nu_prime=nu, emit=emit)
    out = Path(out_path) if out_path else Path(f"{equilibrium.stem}.panels.png")
    return plot_representative(out, scan=scan, title=f"DKX representative run — {equilibrium.name}")
