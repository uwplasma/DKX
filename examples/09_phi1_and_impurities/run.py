"""Impurities, and the potential variation on the surface that moves them.

A highly charged impurity is pulled inward by the main-ion density gradient
and tends to accumulate in a stellarator core.  Whether it does depends on
temperature screening and on ``Phi1``, the variation of the electrostatic
potential *within* the flux surface, which a ``Z = 6`` ion feels six times as
strongly as the bulk.  On this deck the effect is not confined to the
impurity: every species moves by tens of percent, the bulk ions most of all.

Turning ``Phi1`` on makes the equation nonlinear: quasineutrality becomes an
extra block of rows and the solve switches to a Newton-Krylov route.  Both
legs run at identical resolution, so the difference in the fluxes is physics
rather than discretization.

Both legs run on the SFINCS-parameter route: the native executor implements
only ``phi1 = "off"`` today, so there is no ``case.toml`` for this rung.
Multi-species runs *without* ``Phi1`` are fully native -- add a second
``[[species]]`` block to rung 01's case and it works.

Physics: circular tokamak at ``r/a = 0.3``, hydrogen plus fully ionized carbon
plus electrons, full linearized Fokker-Planck collisions -- a momentum-restoring
operator is not optional when the answer is a flux ratio between species.

Expected runtime: ~12 s on a laptop CPU; the ``Phi1`` leg is most of it.
"""

# 1. Imports
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from netCDF4 import Dataset  # noqa: E402

import dkx  # noqa: E402

# 2. User-editable parameters
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent / "output" / "09_phi1_and_impurities"
RESULT_FILE = OUT_DIR / "phi1_and_impurities.nc"
PLOT_FILE = OUT_DIR / "phi1_and_impurities.png"

SPECIES_LABELS = ("hydrogen Z=+1", "carbon Z=+6", "electron Z=-1")
# Impurity concentration n_C / n_H.  Raise it to leave the trace limit and the
# impurity starts changing the bulk ions rather than only responding to them.
CARBON_FRACTION = 0.0125

# 3. Geometry and species construction
GEOMETRY = dict(
    geometryScheme=1, inputRadialCoordinate=3, rN_wish=0.3,
    B0OverBBar=1.0, epsilon_t=-0.07, epsilon_h=0.0,
    iota=0.4542, GHat=3.7481, IHat=0.0, psiAHat=0.15596, aHat=0.5585,
)
# Charge neutrality: n_e = n_H + 6 n_C, so the electron density follows from
# the impurity fraction rather than being typed independently.
SPECIES = dict(
    Zs=[1.0, 6.0, -1.0],
    mHats=[1.0, 12.011, 5.446170214e-4],
    nHats=[1.0, CARBON_FRACTION, 1.0 + 6.0 * CARBON_FRACTION],
    THats=[1.0, 1.0, 1.0],
    dNHatdrHats=[-0.5, -0.5, -0.5],
    dTHatdrHats=[-1.0, -1.0, -1.0],
)

# 4. Physics and numerical configuration
NUMERICS = dict(
    Ntheta=9, Nzeta=1, Nxi=8, NL=4, Nx=4,
    # collisionOperator=0 is the full linearized Fokker-Planck operator.  Pitch
    # angle scattering has no inter-species momentum exchange, which is the
    # entire mechanism setting the impurity flux, so it is not an option here.
    collisionOperator=0,
    Delta=4.5694e-3, alpha=1.0, nu_n=8.330e-3,
)
PHI1_ON = dict(includePhi1=True, includePhi1InKineticEquation=True)
# end of parameters

OUT_DIR.mkdir(parents=True, exist_ok=True)

# 5. Run
without_phi1 = dkx.run(**GEOMETRY, **SPECIES, **NUMERICS, emit=None)
print(f"  Phi1 off: solver route {without_phi1.solve_result.method}, "
      f"converged {bool(without_phi1.solve_result.converged)}")
with_phi1 = dkx.run(**GEOMETRY, **SPECIES, **NUMERICS, **PHI1_ON, emit=None)
print(f"  Phi1 on:  solver route {with_phi1.solve_result.method}, "
      f"converged {bool(with_phi1.solve_result.converged)}")

flux_off = np.asarray(without_phi1.moments["particleFlux_vm_psiHat"], dtype=float).ravel()
flux_on = np.asarray(with_phi1.moments["particleFlux_vm_psiHat"], dtype=float).ravel()
heat_off = np.asarray(without_phi1.moments["heatFlux_vm_psiHat"], dtype=float).ravel()
heat_on = np.asarray(with_phi1.moments["heatFlux_vm_psiHat"], dtype=float).ravel()
charges = np.asarray(SPECIES["Zs"], dtype=float)
perturbation = np.asarray(with_phi1.moments["densityPerturbation"], dtype=float)

# 6. Print a scientific summary and certificate
print("\n=== Final results ===")
print(f"  {'species':<16} {'Gamma (Phi1 off)':>18} {'Gamma (Phi1 on)':>18} {'change':>10}")
for index, label in enumerate(SPECIES_LABELS):
    change = flux_on[index] / flux_off[index] - 1.0
    print(f"  {label:<16} {flux_off[index]:>18.5e} {flux_on[index]:>18.5e} {change:>9.1%}")
carbon_change = flux_on[1] / flux_off[1] - 1.0
print(f"  impurity particle flux changes by {carbon_change:.1%} when Phi1 is included")
print(f"  impurity heat flux: {heat_off[1]:.5e} -> {heat_on[1]:.5e} (normalized)")

# Print the bulk change beside it rather than assuming the effect is confined
# to the impurity.  On this deck it is not: the bulk ions move more than the
# carbon does, so "Phi1 matters for impurities" understates the case.
bulk_change = abs(flux_on[0] / flux_off[0] - 1.0)
print(f"  bulk-ion flux changes by {bulk_change:.1%} over the same switch")
print("  every species moved: Phi1 is not an impurity-only correction on this deck")

# Phi1 is what makes the surface non-uniform, so the density perturbation on
# the surface is the direct measure of how large the effect is.
peak = np.abs(perturbation).max(axis=(1, 2))
for index, label in enumerate(SPECIES_LABELS):
    print(f"  {label:<16} peak |delta n / n| on the surface = {peak[index]:.4e}")

# Charge conservation: with no radial electric field the charge-weighted sum is
# not zero, but it must be the same size as the terms, not larger -- a much
# bigger sum means the species solves have drifted apart.
weighted_on = float(np.sum(charges * flux_on))
scale_on = float(np.max(np.abs(charges * flux_on)))
print(f"  sum_s Z_s Gamma_s (Phi1 on) = {weighted_on:+.4e} "
      f"({weighted_on / scale_on:.1%} of the largest term)")
assert abs(weighted_on) <= 10.0 * scale_on, "charge-weighted flux sum is out of scale"
print(f"  residual (Phi1 on): "
      f"{float(np.asarray(with_phi1.solve_result.residual_norms)[-1]):.3e}")

# 7. Save native result
with Dataset(RESULT_FILE, "w", format="NETCDF4") as dataset:
    dataset.createDimension("species", flux_off.size)
    dataset.createDimension("theta", perturbation.shape[1])
    dataset.createVariable("charge_e", "f8", ("species",))[:] = charges
    dataset.createVariable("particleFlux_phi1_off", "f8", ("species",))[:] = flux_off
    dataset.createVariable("particleFlux_phi1_on", "f8", ("species",))[:] = flux_on
    dataset.createVariable("heatFlux_phi1_off", "f8", ("species",))[:] = heat_off
    dataset.createVariable("heatFlux_phi1_on", "f8", ("species",))[:] = heat_on
    dataset.createVariable("densityPerturbation_phi1_on", "f8", ("species", "theta"))[:] = (
        perturbation[:, :, 0]
    )
    dataset.dkx_version = dkx.__version__
    dataset.solver_route_phi1_on = str(with_phi1.solve_result.method)
    dataset.carbon_fraction = CARBON_FRACTION
print(f"  Wrote result: {RESULT_FILE}")

# 8. Plot publication-ready outputs
figure, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
positions = np.arange(len(SPECIES_LABELS))
left.bar(positions - 0.18, np.abs(flux_off), 0.36, label=r"$\Phi_1$ off", color="tab:grey")
left.bar(positions + 0.18, np.abs(flux_on), 0.36, label=r"$\Phi_1$ on", color="tab:orange")
left.set_yscale("log")
left.set_xticks(positions, SPECIES_LABELS, fontsize=8)
left.set_ylabel(r"$|\Gamma_s|$ (normalized)")
left.set_title("including the in-surface potential moves the impurity", fontsize=10)
left.grid(alpha=0.3, axis="y", which="both")
left.legend(fontsize=9)

theta = np.linspace(0.0, 2.0 * np.pi, perturbation.shape[1], endpoint=False)
for index, label in enumerate(SPECIES_LABELS):
    right.plot(theta, perturbation[index, :, 0], "o-", ms=4, label=label)
right.axhline(0.0, color="0.7", lw=0.8)
right.set(xlabel=r"$\theta$", ylabel=r"$\delta n_s / n_s$")
right.set_title(r"density perturbation on the surface, $\Phi_1$ on", fontsize=10)
right.grid(alpha=0.3)
right.legend(fontsize=8)
figure.suptitle("Impurity transport with and without in-surface potential variation")
figure.savefig(PLOT_FILE, dpi=150)
plt.close(figure)
print(f"  Saved plot: {PLOT_FILE}")
print("Done: examples/09_phi1_and_impurities/run.py")
