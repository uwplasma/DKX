from __future__ import annotations

import json
from pathlib import Path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _speedups(results: list[dict], key: str) -> list[float]:
    base = None
    for row in results:
        if int(row[key]) == 1 and row.get("status", "ok") == "ok":
            base = float(row["mean_s"])
            break
    if base is None or base <= 0.0:
        return [float("nan")] * len(results)
    out: list[float] = []
    for row in results:
        if row.get("status", "ok") != "ok":
            out.append(float("nan"))
        else:
            out.append(base / float(row["mean_s"]))
    return out


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    cpu_json = repo / "examples" / "performance" / "output" / "sharded_scaling_large_cpu" / "sharded_solve_scaling.json"
    fortran_json = repo / "examples" / "performance" / "output" / "fortran_mpi_rhsmode1_sharded_scaling" / "fortran_mpi_scaling.json"
    transport_json = repo / "examples" / "performance" / "output" / "transport_parallel_2min_cpu" / "transport_parallel_scaling.json"
    gpu_json = repo / "examples" / "performance" / "output" / "gpu_sharded_scaling_snapshot.json"

    cpu = _load_json(cpu_json)
    fortran = _load_json(fortran_json)
    transport = _load_json(transport_json)
    gpu = _load_json(gpu_json)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "figure.titlesize": 18,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.8,
        }
    )

    colors = {
        "jax_cpu": "#0F766E",
        "fortran": "#B45309",
        "transport": "#1D4ED8",
        "gpu_medium": "#7C3AED",
        "gpu_large": "#DC2626",
        "ideal": "#6B7280",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.0), constrained_layout=True)

    # Panel A: matched CPU scaling vs Fortran MPI on the same input.
    ax = axes[0, 0]
    cpu_x = np.array([int(r["devices"]) for r in cpu["results"]], dtype=int)
    cpu_t = np.array([float(r["mean_s"]) for r in cpu["results"]], dtype=float)
    cpu_s = np.array([float(r["speedup"]) for r in cpu["results"]], dtype=float)
    f_x = np.array([int(r["ranks"]) for r in fortran], dtype=int)
    f_t = np.array([float(r["elapsed_s"]) for r in fortran], dtype=float)
    f_s = np.array([f_t[0] / t for t in f_t], dtype=float)

    ax.plot(cpu_x, cpu_s, "o-", lw=2.4, ms=7, color=colors["jax_cpu"], label="sfincs_jax CPU sharding")
    ax.plot(f_x, f_s, "s-", lw=2.4, ms=7, color=colors["fortran"], label="SFINCS v3 MPI")
    ax.plot(cpu_x, cpu_x, "--", lw=1.5, color=colors["ideal"], label="Ideal")
    for x, t, s in zip(cpu_x, cpu_t, cpu_s, strict=False):
        ax.annotate(f"{t:.2f}s", (x, s), xytext=(0, 8), textcoords="offset points", ha="center", color=colors["jax_cpu"])
    for x, t, s in zip(f_x, f_t, f_s, strict=False):
        ax.annotate(f"{t:.2f}s", (x, s), xytext=(0, -14), textcoords="offset points", ha="center", color=colors["fortran"])
    ax.set_title("Matched CPU Strong Scaling")
    ax.set_xlabel("Devices / MPI ranks")
    ax.set_ylabel("Speedup vs 1 device")
    ax.set_xticks([1, 2, 4])
    ax.legend(frameon=False, loc="upper left")

    # Panel B: transport-worker scaling on CPU.
    ax = axes[0, 1]
    t_x = np.array([int(r["workers"]) for r in transport["results"]], dtype=int)
    t_speedup = np.array([float(r["speedup"]) for r in transport["results"]], dtype=float)
    t_time = np.array([float(r["mean_s"]) for r in transport["results"]], dtype=float)
    ax.bar(t_x, t_speedup, color=colors["transport"], width=0.58)
    ax.plot(t_x, t_x, "--", lw=1.5, color=colors["ideal"], label="Ideal")
    for x, t, s in zip(t_x, t_time, t_speedup, strict=False):
        ax.annotate(f"{t:.2f}s", (x, s), xytext=(0, 8), textcoords="offset points", ha="center")
    ax.set_title("Transport Worker Scaling")
    ax.set_xlabel("Worker processes")
    ax.set_ylabel("Speedup vs 1 worker")
    ax.set_xticks([1, 2, 4])
    ax.legend(frameon=False, loc="upper right")

    # Panel C: GPU sharded scaling medium case.
    ax = axes[1, 0]
    medium = next(case for case in gpu["cases"] if case["case"] == "rhsmode1_sharded_scaling")
    large = next(case for case in gpu["cases"] if case["case"] == "rhsmode1_sharded")
    m_x = np.array([int(r["devices"]) for r in medium["results"]], dtype=int)
    m_t = np.array([float(r["mean_s"]) for r in medium["results"]], dtype=float)
    m_s = np.array(_speedups(medium["results"], "devices"), dtype=float)
    ax.plot(m_x, m_s, "o-", lw=2.4, ms=7, color=colors["gpu_medium"], label="Medium-large case")
    ax.plot(m_x, m_x, "--", lw=1.5, color=colors["ideal"], label="Ideal")
    for x, t, s in zip(m_x, m_t, m_s, strict=False):
        ax.annotate(f"{t:.1f}s", (x, s), xytext=(0, 8), textcoords="offset points", ha="center", color=colors["gpu_medium"])
    ax.set_title("Office GPU Strong Scaling")
    ax.set_xlabel("GPUs")
    ax.set_ylabel("Speedup vs 1 GPU")
    ax.set_xticks([1, 2])
    ax.legend(frameon=False, loc="upper right")

    # Panel D: large-case runtime status with timeout annotation.
    ax = axes[1, 1]
    l_x = np.array([int(r["devices"]) for r in large["results"]], dtype=int)
    l_t = np.array([float(r["mean_s"]) for r in large["results"]], dtype=float)
    bars = ax.bar(l_x, l_t, color=[colors["gpu_large"], "#FCA5A5"], width=0.58)
    ax.axhline(l_t[0], color=colors["ideal"], linestyle="--", linewidth=1.5, label="1 GPU baseline")
    for bar, row in zip(bars, large["results"], strict=False):
        label = row.get("label", f"{float(row['mean_s']):.1f}s")
        ax.annotate(label, (bar.get_x() + bar.get_width() / 2.0, bar.get_height()), xytext=(0, 8), textcoords="offset points", ha="center")
    ax.set_title("Large PAS Case Runtime Status")
    ax.set_xlabel("GPUs")
    ax.set_ylabel("Wall time (s)")
    ax.set_xticks([1, 2])
    ax.legend(frameon=False, loc="upper left")

    out_dir = repo / "docs" / "_static" / "figures" / "parallel"
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "strong_scaling_snapshot.png"
    pdf = out_dir / "strong_scaling_snapshot.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"Saved {png}")
    print(f"Saved {pdf}")


if __name__ == "__main__":
    main()
