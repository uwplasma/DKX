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
        theta = np.asarray(data.get("theta", np.arange(b_hat.shape[0]))).ravel()
        axes[0].plot(theta, b_hat[:, 0], lw=1.8)
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

    im = axes[2].imshow(b_hat, aspect="auto", origin="lower")
    axes[2].set_title("BHat(theta, zeta)")
    axes[2].set_xlabel("zeta index")
    axes[2].set_ylabel("theta index")
    axes[2].set_xticks([0, max(0, b_hat.shape[-1] - 1)])
    axes[2].set_xticklabels(["0", f"{float(zeta[-1]):.2f}"])
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(f"SFINCS output summary: {Path(input_h5).name}", y=1.03)
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_png.resolve()
