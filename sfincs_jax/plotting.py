from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from .io import read_sfincs_h5


def _select_x_profile(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr)
    while out.ndim > 1:
        out = out[..., 0]
    return np.asarray(out)


def plot_sfincs_output_summary(
    *,
    input_h5: Path,
    output_png: Path,
) -> Path:
    """Write a compact summary plot for a `sfincsOutput.h5` file."""
    data = read_sfincs_h5(Path(input_h5))
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)

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
