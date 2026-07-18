"""RHSMode=2 transport matrices for Boozer ``.bc`` and filtered VMEC geometries.

What this example teaches:
  - how ``dkx.api.write_output`` runs the SFINCS v3 ``whichRHS = 1..3`` loop for
    a 3x3 RHSMode=2 transport matrix on two geometry backends:
    ``geometryScheme=11`` (Boozer ``.bc``) and ``geometryScheme=5`` (filtered
    VMEC ``wout_*.nc``),
  - how ``dkx.compare.compare_sfincs_outputs`` checks the JAX output against the
    frozen SFINCS Fortran v3 reference fixtures in ``tests/ref/``,
  - how the classical (collisional) transport fluxes are reported per whichRHS.

Physics context: the transport matrix summarizes the neoclassical response on a
flux surface; the classical fluxes add the collisional (finite-Larmor-radius)
piece the v3 ``classicalTransport`` module computes [M. Landreman, H. M. Smith,
A. Mollen and P. Helander, Phys. Plasmas 21, 042503 (2014); SFINCS technical
documentation, https://github.com/landreman/sfincs].  Reproducing the frozen
Fortran matrices to ~1e-8 shows the JAX driver matches the reference code on
realistic Boozer and VMEC geometry.

Run:
  python examples/transport/transport_matrix_rhsmode2_scheme11_and_scheme5.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dkx.api import write_output  # noqa: E402
from dkx.compare import compare_sfincs_outputs  # noqa: E402
from dkx.io import read_sfincs_h5  # noqa: E402

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]

# The two frozen RHSMode=2 reference cases (Boozer .bc and filtered VMEC).
CASES = (
    "transportMatrix_PAS_tiny_rhsMode2_scheme11",
    "transportMatrix_PAS_tiny_rhsMode2_scheme5_filtered",
)

# Parity tolerances against the frozen Fortran v3 fixtures.
COMPARE_RTOL = 1e-12
COMPARE_ATOL = 5e-8

# Datasets to echo from the parity comparison.
PARITY_KEYS = {
    "transportMatrix",
    "classicalHeatFlux_psiHat",
    "classicalParticleFlux_psiHat",
    "particleFlux_vm_psiHat",
    "heatFlux_vm_psiHat",
    "FSABFlow",
}

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "transport_matrix_rhsmode2_scheme11_and_scheme5"

# ----------------------------------------------------------------------------
# 1) Solve each case, compare to the frozen Fortran reference, collect matrices
# ----------------------------------------------------------------------------
print("=== examples/transport/transport_matrix_rhsmode2_scheme11_and_scheme5.py ===")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
matrices: dict[str, np.ndarray] = {}
for base in CASES:
    print(f"Step 1[{base}]: solving RHSMode=2 and comparing to the frozen fixture")
    input_path = REPO_ROOT / "tests" / "ref" / f"{base}.input.namelist"
    ref_path = REPO_ROOT / "tests" / "ref" / f"{base}.sfincsOutput.h5"
    out_path = OUTPUT_DIR / f"{base}.sfincsOutput_jax.h5"

    write_output(input_path, out_path, overwrite=True)

    results = compare_sfincs_outputs(a_path=ref_path, b_path=out_path, rtol=COMPARE_RTOL, atol=COMPARE_ATOL)
    for r in sorted((r for r in results if r.key in PARITY_KEYS), key=lambda x: x.key):
        status = "OK" if r.ok else "FAIL"
        print(f"    {status:4s} {r.key:28s} max_abs={r.max_abs:.3e} max_rel={r.max_rel:.3e}")

    # A stronger direct check on the transport matrix itself.
    ref = read_sfincs_h5(ref_path)
    got = read_sfincs_h5(out_path)
    tm_ref = np.asarray(ref["transportMatrix"], dtype=np.float64)
    tm_got = np.asarray(got["transportMatrix"], dtype=np.float64)
    np.testing.assert_allclose(tm_ref, tm_got, rtol=0.0, atol=COMPARE_ATOL)
    matrices[base] = tm_got
    print(f"    transportMatrix matches the fixture within atol={COMPARE_ATOL:g}; wrote {out_path.name}")

# ----------------------------------------------------------------------------
# 2) Plot the JAX transport matrices
# ----------------------------------------------------------------------------
print("Step 2: plotting the transport matrices")
fig, axes = plt.subplots(1, len(CASES), figsize=(9.4, 3.9), constrained_layout=True)
for ax, base in zip(np.atleast_1d(axes), CASES):
    tm = matrices[base]
    im = ax.imshow(tm, cmap="coolwarm", interpolation="nearest")
    ax.set_title(base.replace("transportMatrix_PAS_tiny_rhsMode2_", ""), fontsize=9)
    ax.set_xlabel("column (whichRHS)")
    ax.set_ylabel("row")
    fig.colorbar(im, ax=ax, shrink=0.85)
PLOT_PATH = OUTPUT_DIR / "transport_matrix_rhsmode2_scheme11_and_scheme5.png"
fig.savefig(PLOT_PATH, dpi=150)
plt.close(fig)

# ----------------------------------------------------------------------------
# 3) Results
# ----------------------------------------------------------------------------
print("=== Final results ===")
for base in CASES:
    print(f"  {base}: L11 = {float(matrices[base][0, 0]):.6e}")
print(f"  Saved plot: {PLOT_PATH.name}")
print("Done: examples/transport/transport_matrix_rhsmode2_scheme11_and_scheme5.py")
