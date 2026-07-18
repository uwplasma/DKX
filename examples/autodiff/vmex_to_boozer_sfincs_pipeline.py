"""Differentiable ``vmex`` -> ``booz_xform_jax`` -> ``dkx`` geometry workflow.

What this example teaches:
  - how a VMEC ``wout`` (read through ``vmex``) is transformed on one flux
    surface by ``booz_xform_jax`` and fed to a differentiable ``dkx`` Boozer
    spectrum proxy transport objective,
  - how ``jax.value_and_grad`` differentiates that objective w.r.t. an in-memory
    spectral scale, validated against a centered finite difference, and how a
    few scalar gradient-descent steps move it,
  - the explicit differentiability boundary: only the spectral scaling ->
    ``booz_xform_jax`` -> ``dkx`` proxy objective is differentiated; VMEC file
    I/O, fixed-boundary setup, and the full SFINCS kinetic transport solve are
    outside the differentiated graph (and are an explicit non-claim).

Physics context: the optimized scalar is a bounded geometry/transport proxy on
the Boozer |B| spectrum, the public interface point for fully JAX-native
geometry workflows while the VMEC-boundary-to-kinetic-transport objective
remains a larger research lane [M. Landreman et al., Phys. Plasmas 21, 042503
(2014); SFINCS technical documentation, https://github.com/landreman/sfincs].
The dependency-free local gradient gate always runs; the full pipeline runs
when ``vmex`` and ``booz_xform_jax`` are installed and a VMEC ``wout`` resolves.

Run:
  python examples/autodiff/vmex_to_boozer_sfincs_pipeline.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# Keep this documentation example readable: favor clean output over the
# persistent compilation cache the full CLI enables.
os.environ.setdefault("DKX_DISABLE_COMPILATION_CACHE", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dkx.paths import resolve_existing_path  # noqa: E402
from dkx.workflows.geometry_adapters import (  # noqa: E402
    boozer_spectrum_proxy_transport_gradient_gate,
    boozer_spectrum_proxy_transport_objective,
    geometry_proxy_no_solve_provenance_gate,
    geometry_proxy_workflow_summary,
    optional_jax_geometry_backend_report,
    optional_jax_geometry_backend_status,
)

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]

# VMEC wout override; None auto-resolves a public fixture (or set DKX_VMEX_WOUT).
WOUT_PATH: Path | None = None

# Boozer transform + on-surface grid.
SURFACE = 0.5  # normalized toroidal flux s of the surface to transform
MBOZ = 3  # poloidal Boozer resolution
NBOZ = 3  # toroidal Boozer resolution
N_THETA = 12
N_ZETA = 10

# Differentiable spectral-scale objective + optimization.
SCALE0 = 1.0  # initial spectral scale
FD_STEP = 1e-4  # centered finite-difference step for the gradient check
STEPS = 3  # scalar gradient-descent steps
LEARNING_RATE = 5.0

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "vmex_to_boozer_sfincs_pipeline"
SUMMARY_JSON = OUTPUT_DIR / "vmex_to_boozer_sfincs_pipeline_summary.json"
PLOT_PATH = OUTPUT_DIR / "vmex_to_boozer_sfincs_pipeline.png"


def find_wout() -> Path | None:
    """Resolve a VMEC ``wout`` fixture from the override, env, or data cache."""

    candidates: list[Path] = []
    if WOUT_PATH is not None:
        candidates.append(Path(WOUT_PATH))
    env_path = os.environ.get("DKX_VMEX_WOUT", "").strip()
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(REPO_ROOT / "dkx" / "data" / "equilibria" / "wout_w7x_standardConfig.nc")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    try:
        return resolve_existing_path("wout_w7x_standardConfig.nc").path
    except FileNotFoundError:
        return None


def surface_first_arrays(bx) -> dict[str, jnp.ndarray]:
    """Collect the surface-major spectral arrays booz_xform_jax needs."""

    return {
        "rmnc": jnp.asarray(np.asarray(bx.rmnc).T),
        "zmns": jnp.asarray(np.asarray(bx.zmns).T),
        "lmns": jnp.asarray(np.asarray(bx.lmns).T),
        "bmnc": jnp.asarray(np.asarray(bx.bmnc).T),
        "bsubumnc": jnp.asarray(np.asarray(bx.bsubumnc).T),
        "bsubvmnc": jnp.asarray(np.asarray(bx.bsubvmnc).T),
        "iota": jnp.asarray(np.asarray(bx.iota)),
        "xm_nyq": jnp.asarray(np.asarray(bx.xm_nyq)),
        "xn_nyq": jnp.asarray(np.asarray(bx.xn_nyq)),
    }


# ----------------------------------------------------------------------------
# 1) Optional backend status and the dependency-free gradient gate
# ----------------------------------------------------------------------------
print("=== examples/autodiff/vmex_to_boozer_sfincs_pipeline.py ===")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
report = optional_jax_geometry_backend_report()
status = optional_jax_geometry_backend_status()
print("Step 1: optional backend status and the local gradient gate")
for name in ("vmex", "booz_xform_jax"):
    print(f"  {name}: {'available' if status[name] else 'missing'}")
readiness = boozer_spectrum_proxy_transport_gradient_gate()
print(
    f"  local Boozer-proxy gradient gate: {readiness['status']} "
    f"(max_grad_error={readiness['max_gradient_abs_error']:.3e}, "
    f"tol={readiness['gradient_tolerance']:.3e})"
)
print("  differentiated graph: scaled spectral arrays -> booz_xform_jax -> dkx proxy transport objective")
print("  not claimed: full VMEC-boundary-to-SFINCS kinetic transport gradients")

# ----------------------------------------------------------------------------
# 2) Full pipeline when both backends and a wout are available
# ----------------------------------------------------------------------------
wout_path = find_wout() if (status["vmex"] and status["booz_xform_jax"]) else None
if wout_path is None:
    print("Step 2: optional backends or a VMEC wout are unavailable -- skipping the full pipeline")
    summary = geometry_proxy_workflow_summary()
    summary["backend_readiness_gate"] = readiness
    summary["no_solve_provenance_gate"] = geometry_proxy_no_solve_provenance_gate(
        summary, require_file_provenance=False
    )
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("=== Final results ===")
    print(f"  backend report contract version: {report['workflow_contract']['contract_version']}")
    print(f"  Wrote summary JSON: {SUMMARY_JSON.name}")
    print("Done: examples/autodiff/vmex_to_boozer_sfincs_pipeline.py")
else:
    from booz_xform_jax import Booz_xform
    from booz_xform_jax.jax_api import booz_xform_jax
    from vmex import read_wout as read_vmex_wout

    print(f"Step 2: reading the VMEC wout and building the Boozer context ({wout_path.name})")
    wout_like = read_vmex_wout(wout_path)
    provenance = f"vmex.read_wout('{wout_path}')"
    bx = Booz_xform()
    bx.read_wout_data(wout_like)
    bx.mboz = int(MBOZ)
    bx.nboz = int(NBOZ)
    s_in = np.asarray(bx.s_in, dtype=float)
    surface_index = int(np.argmin(np.abs(s_in - float(SURFACE))))
    selected_surface = float(s_in[surface_index])

    arrays = surface_first_arrays(bx)
    non_axis = (arrays["xm_nyq"] != 0) | (arrays["xn_nyq"] != 0)
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, N_THETA, endpoint=False)
    zeta = jnp.linspace(0.0, 2.0 * jnp.pi / float(bx.nfp), N_ZETA, endpoint=False)

    def objective(scale: jnp.ndarray) -> jnp.ndarray:
        scaled_bmnc = jnp.where(non_axis[None, :], arrays["bmnc"] * scale, arrays["bmnc"])
        out = booz_xform_jax(
            rmnc=arrays["rmnc"], zmns=arrays["zmns"], lmns=arrays["lmns"],
            bmnc=scaled_bmnc, bsubumnc=arrays["bsubumnc"], bsubvmnc=arrays["bsubvmnc"],
            iota=arrays["iota"], xm=bx.xm, xn=bx.xn, xm_nyq=bx.xm_nyq, xn_nyq=bx.xn_nyq,
            nfp=int(bx.nfp), mboz=int(bx.mboz), nboz=int(bx.nboz), asym=bool(bx.asym),
            surface_indices=[surface_index],
        )  # fmt: skip
        return boozer_spectrum_proxy_transport_objective(
            out["bmnc_b"][0], out["ixm_b"], out["ixn_b"], theta=theta, zeta=zeta
        )

    print("Step 3: differentiating the proxy objective and taking a few descent steps")
    scale0 = jnp.asarray(float(SCALE0))
    value, gradient = jax.value_and_grad(objective)(scale0)
    finite_difference = float(
        (objective(scale0 + FD_STEP) - objective(scale0 - FD_STEP)) / (2.0 * FD_STEP)
    )
    gradient_error = abs(float(gradient) - finite_difference)

    scale = scale0
    trajectory = [(0, float(value), float(scale))]
    for k in range(max(0, int(STEPS))):
        loss, grad = jax.value_and_grad(objective)(scale)
        scale = scale - float(LEARNING_RATE) * grad
        trajectory.append((k + 1, float(loss), float(scale)))
        print(f"  step {k + 1:02d}: proxy={float(loss):.9e}, scale={float(scale):.9f}")

    summary = geometry_proxy_workflow_summary(
        provenance=provenance,
        requested_surface=float(SURFACE),
        selected_surface=selected_surface,
        boozer_resolution={"mboz": int(bx.mboz), "nboz": int(bx.nboz)},
        grid_shape={"n_theta": int(N_THETA), "n_zeta": int(N_ZETA)},
        scale=float(scale0),
        proxy_objective=float(value),
        autodiff_gradient=float(gradient),
        finite_difference_gradient=finite_difference,
        finite_difference_step=float(FD_STEP),
    )
    summary["backend_readiness_gate"] = readiness
    summary["no_solve_provenance_gate"] = geometry_proxy_no_solve_provenance_gate(
        summary, require_file_provenance=True
    )
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    steps = [t[0] for t in trajectory]
    proxies = [t[1] for t in trajectory]
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    ax.plot(steps, proxies, "o-", color="tab:blue")
    ax.set_xlabel("gradient-descent step")
    ax.set_ylabel("proxy transport objective")
    ax.set_title("vmex -> Boozer -> dkx proxy: descent trajectory")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=140)
    plt.close(fig)

    print("=== Final results ===")
    print(f"  VMEC provenance: {provenance}")
    print(f"  Boozer surface: requested s={SURFACE:.3f}, selected s={selected_surface:.6f}")
    print(f"  proxy objective(scale={float(scale0):.6g}) = {float(value):.12e}")
    print(f"  d objective / d scale (JAX)         = {float(gradient):.12e}")
    print(f"  d objective / d scale (centered FD) = {finite_difference:.12e}")
    print(f"  abs gradient error = {gradient_error:.3e}")
    print(f"  numerical gradient gate: {summary['numerical_gradient_gate']['status']}")
    print(f"  no-solve provenance gate: {summary['no_solve_provenance_gate']['status']}")
    print(f"  Saved plot: {PLOT_PATH.name}")
    print(f"  Wrote summary JSON: {SUMMARY_JSON.name}")
    print("Done: examples/autodiff/vmex_to_boozer_sfincs_pipeline.py")
