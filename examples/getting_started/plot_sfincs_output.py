"""Plot a few diagnostics from ``sfincsOutput.h5``.

If ``--input-h5`` is omitted, the script uses a frozen tiny output fixture.
This keeps the example fast while still demonstrating how to read and visualize
SFINCS-style output files.

Run:
  python examples/getting_started/plot_sfincs_output.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


def _read_array(group: h5py.File, key: str) -> np.ndarray:
    return np.asarray(group[key])


def _select_species_row(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 0:
        return np.asarray([float(arr)])
    if arr.ndim == 1:
        return arr
    return np.asarray(arr[0])


def _select_x_profile(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr)
    while out.ndim > 1:
        out = out[..., 0]
    return np.asarray(out)


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-h5",
        type=Path,
        default=repo_root / "tests" / "ref" / "output_scheme4_2species_quick.sfincsOutput.h5",
        help="Path to sfincsOutput.h5 to visualize.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).with_suffix("").parent / "output" / "sfincsOutput_summary.png",
        help="Output PNG path.",
    )
    args = parser.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.input_h5, "r") as h5:
        x = _read_array(h5, "x")
        zeta = _read_array(h5, "zeta")
        b_hat = _read_array(h5, "BHat")
        flow_vs_x = _select_x_profile(_read_array(h5, "FSABFlow_vs_x"))
        heat_vs_x = _select_x_profile(_read_array(h5, "heatFlux_vm_psiHat_vs_x"))

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), constrained_layout=True)

    x_1d = np.asarray(x).ravel()
    axes[0].plot(x_1d, np.asarray(flow_vs_x).ravel(), "o-", lw=1.8)
    axes[0].set_title("Flow profile vs x")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("FSABFlow_vs_x")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(x_1d, np.asarray(heat_vs_x).ravel(), "o-", lw=1.8, color="#b45309")
    axes[1].set_title("Heat-flux profile vs x")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("heatFlux_vm_psiHat_vs_x")
    axes[1].grid(True, alpha=0.25)

    im = axes[2].imshow(np.asarray(b_hat), aspect="auto", origin="lower")
    axes[2].set_title("BHat(theta, zeta)")
    axes[2].set_xlabel("zeta index")
    axes[2].set_ylabel("theta index")
    axes[2].set_xticks([0, max(0, b_hat.shape[-1] - 1)])
    axes[2].set_xticklabels(["0", f"{float(np.asarray(zeta).ravel()[-1]):.2f}"])
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(f"SFINCS output summary: {args.input_h5.name}", y=1.03)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
