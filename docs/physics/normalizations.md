# Normalizations and units

DKX keeps the SFINCS v3 normalization. Every quantity inside the solver is a dimensionless
"hat" quantity: a ratio to a fixed reference ("bar") value. The native case format states
SI inputs (m$^{-3}$, keV, amu, kV/m) and `dkx.units` converts both ways through the same
pinned reference set.

## Reference set

The reference values are fixed (`src/dkx/units.py`, SFINCS `globalVariables.F90`):

| Symbol | Name | Value |
| --- | --- | --- |
| $\bar n$ | `N_BAR` | $10^{20}$ m$^{-3}$ |
| $\bar T$ | `T_BAR` | 1 keV |
| $\bar m$ | `M_BAR` | proton mass |
| $\bar B$ | `B_BAR` | 1 T |
| $\bar R$ | `R_BAR` | 1 m |
| $\ln\Lambda$ | `COULOMB_LOGARITHM` | 17 |
| $\bar v = \sqrt{2\bar T/\bar m}$ | `V_BAR` | $4.3769\times10^{5}$ m/s |

Hatted fields are ratios to these: $\hat n = n/\bar n$, $\hat T = T/\bar T$,
$\hat m = m/\bar m$, $\hat B = B/\bar B$, $\hat r = r/\bar R$,
$\hat\psi = \psi/(\bar B\bar R^2)$, $\hat\Phi = e\Phi/\bar T$ (with $\alpha = 1$). The
Boozer flux functions are $\hat G = \langle B_\zeta\rangle/(\bar B\bar R)$ (`GHat`) and
$\hat I = \langle B_\theta\rangle/(\bar B\bar R)$ (`IHat`). The distribution function is
normalized as $f_s = (\bar n/\bar v^3)\hat f_s$ (v3 technical documentation, eq. 98).

A case declares masses in atomic mass units while $\hat m$ is a ratio to the proton mass;
the two differ by 0.14 % and `dkx.units.m_hat_from_mass_amu` holds the conversion in one
place.

## Ordering parameters

Three scalars multiply the operator terms:

$$
\Delta = \frac{\bar m\,\bar v}{e\,\bar B\,\bar R},
\qquad
\alpha = \frac{e\,\bar\Phi}{\bar T},
\qquad
\nu_n = \frac{\bar\nu\,\bar R}{\bar v},
\qquad
\bar\nu = \frac{4\sqrt{2\pi}\,\bar n\,e^4\ln\Lambda}{3\,(4\pi\epsilon_0)^2\sqrt{\bar m}\,\bar T^{3/2}} .
$$

- $\Delta$ (`Delta`) is the reference gyroradius over $\bar R$. It multiplies every drift
  term and every radial flux.
- $\alpha$ (`alpha`) converts normalized potential to normalized energy. It appears in the
  $E\times B$ drift, the $\Phi_1$ Boltzmann factor and quasineutrality.
- $\nu_n$ (`nu_n`) multiplies the collision operator.

At the reference set $\Delta = 4.5694\times10^{-3}$ and $\nu_n = 8.330\times10^{-3}$
(`DEFAULT_DELTA`, `DEFAULT_NU_N` in `src/dkx/constants.py`).
`dkx.units.reference_delta` and `reference_nu_n` recompute both from the SI constants, and
`tests/test_units.py` pins them to the defaults.

The native executor uses these defaults, with $\nu_n$ scaled by the case's Coulomb
logarithm: $\nu_n = 8.330\times10^{-3}\,(\ln\Lambda/17)$ (`physics.coulomb_logarithm`,
default 17, accepted range 5-30). A SFINCS namelist sets `Delta`, `alpha` and `nu_n`
directly, and all three are written to the output.

## Angle and velocity coordinates

The equation is solved on $(\theta,\zeta,x,\xi)$:

- $\theta$, $\zeta$: Boozer angles, or VMEC angles for `geometryScheme = 5`
  ({doc}`geometry`);
- $x_s = v/v_{\mathrm{th},s}$ with $v_{\mathrm{th},s} = \sqrt{2T_s/m_s}$, so in hat units
  the thermal speed factor is $\sqrt{\hat T_s/\hat m_s}$;
- $\xi = v_\parallel/v$, expanded in Legendre modes.

## Flux-surface average

Constraints and diagnostics use

$$
\langle g\rangle
= \frac{\int d\theta\,d\zeta\; g\,\hat D^{-1}}{\int d\theta\,d\zeta\;\hat D^{-1}},
\qquad
\hat V' = \sum_{ij} \frac{w^\theta_i w^\zeta_j}{\hat D_{ij}},
$$

with the grid quadrature weights `thetaWeights`, `zetaWeights` and the Jacobian factor
`DHat`. $\hat V'$ is `VPrimeHat`; $\langle\hat B^2\rangle$ is `FSABHat2`.

## Radial coordinates

Four radial labels are supported, selected by `inputRadialCoordinate` (the surface) and
`inputRadialCoordinateForGradients` (the profile gradients):

| Code | Label |
| --- | --- |
| 0 | $\hat\psi$ |
| 1 | $\psi_N = \psi/\psi_a$ |
| 2 | $\hat r$ |
| 3 | $r_N = r/a$, with $r_N = \sqrt{\psi_N}$ |
| 4 (gradients only) | $\hat r$ for $n$ and $T$, and $E_r$ in place of $d\Phi/dr$ |

Radial fluxes are computed against $\nabla\hat\psi$ and written in all four variants
(`*_psiHat`, `*_psiN`, `*_rHat`, `*_rN`). A `*_psiHat` flux times

$$
\texttt{ddrHat2ddpsiHat} = \frac{\hat a}{2\hat\psi_a\sqrt{\psi_N}}
$$

is the `*_rHat` flux (v3 documentation eq. 175, `dkx.units.flux_psi_hat_to_r_hat`). This factor carries the sign
of $\hat\psi_a$, which follows the toroidal-flux orientation of the equilibrium file
($+0.083$ for the precise-QA reference, $-0.385$ for W7-X standard configuration, from the
`flux_psi_hat_to_r_hat` docstring). A flux against $\hat\psi$ can therefore change sign
with a file convention; the $\hat r$ flux is outward-positive in both cases.

## Radial electric field

$E_r = -d\Phi_0/dr$. The operator uses $d\hat\Phi/d\hat\psi$; the inputs `Er`,
`dPhiHatdpsiHat`, `dPhiHatdrHat` and the others are converted through the radial factors,
with $d\hat\Phi/d\hat\psi = \texttt{ddrHat2ddpsiHat}\cdot(-E_r)$. Because
$\hat\Phi = e\Phi/\bar T$ with $\bar T = 1$ keV and $\hat r = r/\bar R$ with
$\bar R = 1$ m, the normalized `Er` is numerically equal to $E_r$ in kV/m
(`dkx.units.er_hat_from_electric_field_kv_m`). The native case states
`electric_field.value_kV_m`.

## Monoenergetic parameters

For `RHSMode = 3` the collisionality and field are given as `nuPrime` and `EStar`:

$$
\nu' = \frac{(\hat G + \iota\hat I)\,\hat\nu}{v\,\hat B_0},
\qquad
E^* = \frac{\hat G}{\iota\,v\,\hat B_0}\frac{d\hat\Phi}{d\hat\psi},
$$

with $\hat B_0$ the $(0,0)$ Fourier mode of $\hat B$. The code overwrites `nu_n` and
`dPhiHatdpsiHat` from these (v3 `sfincs_main.F90`), and the speed grid collapses to a
single node at $x = 1$. Defaults are `nuPrime = 1`, `EStar = 0`.

## Conversion to SI

The output fields convert to SI with the factors in `dkx.units` (v3 documentation
eqs. 194-221):

| Output | Multiply by | SI unit | Factor |
| --- | --- | --- | --- |
| `FSABjHat` | `PARALLEL_CURRENT` $= e\bar n\bar v\bar B$ | A T m$^{-2}$ ($\langle\mathbf{j}\cdot\mathbf{B}\rangle$, the VMEC `jdotb` unit) | $7.0126\times10^{6}$ |
| `FSABjHatOverRootFSAB2` | `CURRENT_DENSITY` $= e\bar n\bar v$ | A m$^{-2}$ | $7.0126\times10^{6}$ |
| `particleFlux_*_rHat` | `PARTICLE_FLUX` $= \bar n\bar v$ | m$^{-2}$ s$^{-1}$ | $4.3769\times10^{25}$ |
| `heatFlux_*_rHat` | `HEAT_FLUX` $= \bar n\bar m\bar v^3$ | W m$^{-2}$ | $1.4025\times10^{10}$ |

The factors are those in the `dkx.units` source comments. The $\bar R$ in the flux
normalization cancels against $\hat r = r/\bar R$, so the `*_rHat` flux times the factor is
the SI flux density through the surface.
