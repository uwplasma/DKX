"""Scan the radial electric field -> find the ambipolar root -> plot both.

The radial electric field is not an input; it is set by ambipolarity,
``J_r(E_r) = sum_s Z_s Gamma_s = 0``.  ``dkx.batched_er_scan`` does the scan
in one batched call and the crossing is the root -- plotting it, rather than
solving from a guess, shows whether you have one root or a stellarator's three.

Ambipolarity is *charge-weighted*, not an equality of fluxes.  Here (hydrogen
Z=1, carbon Z=6) the root is where ``Gamma_H = -6 Gamma_C``, so the curves sit
nowhere near each other.  Only a Z = +-1 pair gives equal fluxes; taking that
as the general rule is the mistake this example prevents.

Physics: hydrogen with a carbon impurity, from the bundled SFINCS examples.
The root is negative -- an ion root -- and the impurity flux runs opposite in
sign to the bulk flux.

Expected runtime: ~30 s on a laptop CPU for the 13-point scan.

Achieved: one root near -0.97 kV/m, where the script asserts the
charge-weighted sum rather than claiming it: -1.0e-24 against terms of 1e-09.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import dkx

# --------------------------- parameters -------------------------------------
CI = os.environ.get("DKX_CI") == "1"
OUT_DIR = Path(__file__).resolve().parent / "output"
REPO = Path(__file__).resolve().parents[2]
DECK = REPO / "examples/sfincs_examples/quick_2species_FPCollisions_noEr/input.namelist"

ER_VALUES = np.linspace(-8.0, 8.0, 7 if CI else 13)
# Widen if the scan shows no sign change -- a steeper plasma pushes the root
# further out, and a bracket that misses it reports "no root" rather than
# guessing:
# ER_VALUES = np.linspace(-30.0, 10.0, 21)
# ----------------------------- end of parameters ----------------------------

OUT_DIR.mkdir(exist_ok=True)

scan = dkx.batched_er_scan(DECK, ER_VALUES)
j_r = np.asarray(scan.radial_current, dtype=float).ravel()
gamma = np.asarray(scan.moments["particleFlux_vm_psiHat"], dtype=float)

print(f"{'Er [kV/m]':>11} {'J_r':>14}   sign")
for value, current in zip(ER_VALUES, j_r):
    print(f"{value:>11.3f} {current:>14.5e}   {'+' if current > 0 else '-'}")

# The root is where J_r crosses zero.  Linear interpolation across the
# bracketing pair is deliberate: J_r is a linear combination of the species
# fluxes, so interpolating every moment the same way keeps sum_s Z_s Gamma_s
# exactly zero at the reported root.
roots = [
    float(a + (b - a) * ja / (ja - jb))
    for a, b, ja, jb in zip(ER_VALUES, ER_VALUES[1:], j_r, j_r[1:])
    if ja * jb < 0.0
]
print(f"\nroots found: {[f'{r:+.3f}' for r in roots] or 'none in this bracket'}")

figure, (left, right) = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)

left.plot(ER_VALUES, j_r, "o-", color="tab:green")
left.axhline(0.0, color="0.4", ls=":", lw=1.0)
for root in roots:
    left.plot([root], [0.0], "x", color="tab:red", ms=11, mew=2.5,
              label=f"root {root:+.3f} kV/m")  # fmt: skip
left.set(xlabel=r"$E_r$ [kV/m]", ylabel=r"$J_r=\sum_s Z_s\Gamma_s$")
left.set_title("ambipolarity: the root is where this crosses zero", fontsize=10)
left.grid(alpha=0.3)
if roots:
    left.legend(fontsize=8)

# Label the species by their actual charge rather than assuming ions/electrons:
# this deck is hydrogen plus a carbon impurity, and "electrons" would be a lie.
from dkx.inputs import read_sfincs_input, sfincs_input_from_raw  # noqa: E402

species = sfincs_input_from_raw(read_sfincs_input(DECK)).species
names = [f"Z={z:g}" for z in species.z_s]
for column, name in enumerate(names):
    right.plot(ER_VALUES, gamma[:, column], "o-", ms=4, label=rf"$\Gamma$ {name}")
right.set(xlabel=r"$E_r$ [kV/m]", ylabel=r"$\Gamma_s$")
right.set_title("species particle flux across the scan", fontsize=10)
right.grid(alpha=0.3)
right.legend(fontsize=8)

# Self-check: sum_s Z_s Gamma_s must vanish at the root.  With Z = 1 and 6 that
# is Gamma_H = -6 Gamma_C, not Gamma_H = Gamma_C.
if roots:
    root = roots[0]
    at_root = [float(np.interp(root, ER_VALUES, gamma[:, c])) for c in range(gamma.shape[1])]
    weighted = sum(float(z) * g for z, g in zip(species.z_s, at_root))
    scale = max(abs(float(z) * g) for z, g in zip(species.z_s, at_root))
    for (z, g) in zip(species.z_s, at_root):
        print(f"  Z={z:g}  Gamma = {g:+.5e}")
    print(f"  sum_s Z_s Gamma_s = {weighted:+.3e}  ({weighted / scale:.1%} of the largest term)")
    assert abs(weighted) < 1e-2 * scale, "ambipolarity violated at the reported root"

figure.suptitle("Ambipolar radial electric field")
out = OUT_DIR / "plot_ambipolar_er.png"
figure.savefig(out, dpi=150)
plt.show(block=False)
print(f"wrote {out}")
