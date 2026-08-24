"""Particle flux, heat flux and bootstrap current across the minor radius.

One surface is a number; a profile is a result.  This is the loop that turns
DKX into the thing you actually want -- neoclassical transport as a function
of radius -- and it is a ``for`` loop over ``dkx.run``, nothing more.

Two details are what make it correct rather than merely plausible:

  - the fluxes come out in ``psiHat``, and a flux per unit toroidal flux is
    not a flux per unit length.  ``dkx.units`` converts, and the conversion
    depends on radius, so it belongs inside the loop.
  - ``geometryScheme=1`` is an analytic model, so the *surface* is an input
    too.  The inverse aspect ratio has to grow with radius; leaving
    ``epsilon_t`` at its on-axis value would model a torus whose cross-section
    never widens, and the trapped fraction -- which is what neoclassical
    transport is about -- would be wrong everywhere but one surface.

Physics: a single-species tokamak with parabolic n and T, no radial electric
field.  Fluxes are outward-positive.  The n and T curves are the same function
here, so they genuinely do lie on top of each other in the first panel -- they
are drawn with different markers and dash patterns so that reads as one shared
profile rather than as a curve that failed to draw.

Expected runtime: ~40 s on a laptop CPU for seven surfaces.

Achieved: all three peak at mid-radius rather than at the edge -- Q near
r/a = 0.3, Gamma near 0.45, the bootstrap current near 0.6 -- and fall away
on both sides.  Two effects fight: the gradient that drives the transport
grows linearly outward, while the n and T it multiplies fall to 35% of their
axis value by r/a = 0.9.  The product turns over in between.  Guessing
"largest at the edge, where the gradient is largest" gets this wrong, which
is the reason to plot a profile instead of extrapolating from one surface.

The turnover radii are a property of these parabolic profiles, not of
neoclassical theory; change PEAKING and they move.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import dkx
from dkx import units

# --------------------------- parameters -------------------------------------
A_HAT = 0.5585          # minor radius / RBar
G_HAT = 3.7481          # R0 * B0, so R0 = G_HAT with B0OverBBar = 1
PSI_A_HAT = 0.15596     # toroidal flux at the edge / (2 pi)
RADII = (0.15, 0.3, 0.45, 0.6, 0.75, 0.9)   # rN = r / a
PEAKING = 0.8           # n, T fall to (1 - PEAKING) of their axis value

RESOLUTION = dict(Ntheta=15, Nzeta=1, Nxi=24, NL=4, Nx=6)
OUT = Path(__file__).resolve().parent / "output" / "plot_flux_profile.png"
# ----------------------------- end of parameters ----------------------------


def profiles(r_n):
    """Parabolic n and T, and their gradients with respect to rHat."""
    shape = 1.0 - PEAKING * r_n**2
    # d/drHat = (1 / aHat) d/drN, because rHat = aHat * rN.
    slope = -2.0 * PEAKING * r_n / A_HAT
    return shape, slope


records = []
print(f"{'rN':>6} {'Gamma [1/m2/s]':>16} {'Q [W/m2]':>14} {'<j.B> [A/m2 T]':>16}")
for r_n in RADII:
    n_hat, dn = profiles(r_n)
    t_hat, dt = profiles(r_n)
    run = dkx.run(
        geometryScheme=1, inputRadialCoordinate=3, rN_wish=r_n,
        aHat=A_HAT, GHat=G_HAT, IHat=0.0, psiAHat=PSI_A_HAT,
        B0OverBBar=1.0, iota=0.4542, epsilon_h=0.0,
        # The surface must widen with radius: epsilon_t = r / R0.
        epsilon_t=-(A_HAT / G_HAT) * r_n,
        Zs=[1.0], mHats=[1.0], nHats=[n_hat], THats=[t_hat],
        dNHatdrHats=[dn], dTHatdrHats=[dt],
        collisionOperator=1, Delta=4.5694e-3, alpha=1.0, nu_n=8.330e-3,
        **RESOLUTION,
    )

    # psiHat -> rHat, then Hat -> SI.  Both steps are needed and neither is
    # a constant: the first depends on radius.
    to_r_hat = units.flux_psi_hat_to_r_hat(psi_a_hat=PSI_A_HAT, a_hat=A_HAT, r_n=r_n)
    # The flux moments carry a species axis even when there is one species,
    # so index it rather than flattening -- on a multi-species deck the
    # flattened first entry would silently be whichever species came first.
    ion = 0
    gamma = float(run.moments["particleFlux_vm_psiHat"][ion]) * to_r_hat * units.PARTICLE_FLUX
    heat = float(run.moments["heatFlux_vm_psiHat"][ion]) * to_r_hat * units.HEAT_FLUX
    current = float(np.asarray(run.moments["FSABjHat"]).ravel()[0]) * units.PARALLEL_CURRENT

    records.append((r_n, n_hat, t_hat, gamma, heat, current))
    print(f"{r_n:>6.2f} {gamma:>16.4e} {heat:>14.4e} {current:>16.4e}")

r_n, n_hat, t_hat, gamma, heat, current = (np.array(c) for c in zip(*records))

figure, axes = plt.subplots(2, 2, figsize=(9.5, 7.0), constrained_layout=True)

axes[0, 0].plot(r_n, n_hat, "o-", label=r"$\hat n$")
axes[0, 0].plot(r_n, t_hat, "s--", label=r"$\hat T$")
axes[0, 0].set_ylabel("normalised profile")
axes[0, 0].set_title("what drives the transport")
axes[0, 0].legend(frameon=False)

axes[0, 1].plot(r_n, gamma, "o-", color="C2")
axes[0, 1].set_ylabel(r"$\Gamma$  [m$^{-2}$ s$^{-1}$]")
axes[0, 1].set_title("particle flux (outward positive)")

axes[1, 0].plot(r_n, heat, "o-", color="C3")
axes[1, 0].set_ylabel(r"$Q$  [W m$^{-2}$]")
axes[1, 0].set_title("heat flux")

axes[1, 1].plot(r_n, current, "o-", color="C4")
axes[1, 1].set_ylabel(r"$\langle j_\parallel B\rangle$  [A m$^{-2}$ T]")
axes[1, 1].set_title("bootstrap current")

for axis in axes.ravel():
    axis.set_xlabel(r"$r/a$")
    axis.axhline(0.0, lw=0.6, color="0.7", zorder=0)
    axis.grid(alpha=0.25)

figure.suptitle("Neoclassical transport across the minor radius")
OUT.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(OUT, dpi=150)
print(f"\nwrote {OUT}")
