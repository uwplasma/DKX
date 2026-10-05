# Reduced models

Several modules reuse the same `KineticOperator`, geometry and normalization to give
cheaper or more specialized answers than a full multi-species solve. They are
namelist/Python tools; the native case executor does not call them.

| Module | Model | Inputs |
| --- | --- | --- |
| `dkx.monoenergetic` | $(\nu', E^*)$ database of monoenergetic coefficients and energy convolution | `RHSMode = 3` namelist |
| `dkx.variational` | Upper and lower bounds on monoenergetic $D_{11}$ | solved `RHSMode = 3` state |
| `dkx.shaing_callen` | Collisionless limit of the bootstrap coefficient | $\hat B(\theta,\zeta)$ and flux functions |
| `dkx.bounce_averaged` | Differentiable $1/\nu$ effective ripple | `FluxSurfaceGeometry` |
| `dkx.momentum_correction` | Sugama-Nishimura parallel-momentum correction of monoenergetic flows | monoenergetic database, species |
| `dkx.validity` | Regime and radial-locality flags | operator, geometry |
| `dkx.bootstrap` | Kinetic bootstrap current as an objective on a VMEX equilibrium | VMEX state or `wout` |

## Monoenergetic database

At fixed speed the pitch-angle-scattering equation with DKES trajectories depends only on
the normalized collisionality $\nu'$ and field $E^*$ ({doc}`normalizations`). The
monoenergetic formulation is that of Hirshman, Shaing, van Rij, Beasley & Crume, Phys.
Fluids 29, 2951 (1986). `monoenergetic_database` scans a $(\nu', E^*)$ grid with
`RHSMode = 3` solves and stores $D^*_{11}$, $D^*_{13}$, $D^*_{31}$, $D^*_{33}$ in the ICNTS
normalization (Beidler et al., Nucl. Fusion 51, 076001, 2011): $D_{11}$ against the
equivalent-tokamak plateau value, $D_{31}$ and $D_{13}$ against the banana-regime bootstrap
value, and $D^*_{33} \to 1$ in the collisional limit. `energy_convolution` integrates the
database over a Maxwellian to give the thermal transport matrix per species.
`save_database` and `load_database` use a portable `.npz` with a schema tag. The CLI
equivalent is `dkx monoenergetic-database`.

Measured checks (`tests/test_monoenergetic_database.py`, module docstring):

| Check | Measured |
| --- | --- |
| Database point vs direct `run_transport_matrix` | $\le 10^{-15}$ relative |
| $\lvert 1 - D^*_{33}\rvert$ at $\nu' = 10$ | $3.0\times10^{-4}$ |
| Onsager $\lvert D^*_{13} + D^*_{31}\rvert/\lvert D^*_{31}\rvert$ at $\nu' = 1.2\times10^{-3}$ | $7.9\times10^{-4}$ |
| Convolution vs full `RHSMode = 2` $3\times3$ solve, database at exact nodes | $5.8\times10^{-14}$ max relative (asserted $< 10^{-10}$) |
| `jax.grad` of convolved $L_{11}$ w.r.t. a Boozer amplitude vs finite differences | $5.5\times10^{-10}$ relative |

The convolution is exact only for the model in the database: pitch-angle scattering, DKES
trajectories. It omits momentum conservation (see the correction below) and energy
scattering.

## Variational bounds on $D_{11}$

With pitch-angle scattering the monoenergetic operator splits as $M = V + P$, where the
collision part $P$ is symmetric positive semidefinite and the streaming, mirror and
$E\times B$ part $V$ is antisymmetric in the entropy inner product. Two quadratic
functionals then bound $D_{11}$ from below and above for any trial function and coincide
at the exact solution (Hirshman et al. 1986; van Rij & Hirshman, Phys. Fluids B 1, 563,
1989). Evaluated on the even and odd Legendre-parity parts of the discrete solution, they
bracket the computed $D_{11}$ at zero radial electric field.

`monoenergetic_d11_bounds(op, state, ...)` returns `lower`, `d11`, `upper` and
`gap = |upper - lower| / |d11|`; `d11_bounds_supported(op)` checks the preconditions
(`RHSMode = 3`, pitch-angle scattering, monoenergetic trajectories). The gap is a discrete
entropy-structure diagnostic from one run: a gap that shrinks on a refinement ladder is
evidence of improved resolution, not an enclosure of the continuum discretization error,
which needs an observable-specific resolution study. `tests/test_variational_bounds.py`
checks that the bounds bracket $D_{11}$, that the gap shrinks with resolution, and that it
is tight at high collisionality. The strict bound holds for purely parity-flipping
trajectories ($E^* = 0$); at finite $E^*$ the gap is a consistency diagnostic only.

## Shaing-Callen collisionless limit

As $\nu' \to 0$ the monoenergetic bootstrap coefficient (`transportMatrix[1][0]` in
`RHSMode = 3`) approaches a collisionality-independent value set by the geometry alone
(Shaing & Callen, Phys. Fluids 26, 3315, 1983). `shaing_callen_d31_limit` evaluates it in
the closed form of Albert, Beidler, Kapper, Kasilov & Kernbichler (arXiv:2407.21599,
2024), solving the geodesic-curvature magnetic differential equations spectrally on a
Fourier-upsampled grid. It returns `d31` and the geometric factor `lambda_bb`.

For an axisymmetric field the factor reduces to $\lambda_{bB} = (G/\iota)\,f_t$, with
$f_t$ the trapped-particle fraction (Boozer & Gardner, Phys. Fluids B 2, 2408, 1990),
exposed as `trapped_fraction`. `tests/test_shaing_callen.py` checks that closed form and
that a $\nu'$ scan of the full monoenergetic solve approaches the limit in the
axisymmetric case (and behaves qualitatively in a helical case).

The limit is a collisionless geometric reference, not a value every finite-collisionality
scan must reach. Albert et al. show that at zero radial electric field in the $1/\nu$
regime the offset from it can oscillate rather than vanish as collisionality decreases;
convergence needs additional conditions such as significant orbit precession, for example
from a finite radial electric field. The scan in {doc}`../benchmarks/analytic_limits` uses
$E^* = 0.003$, a finite radial electric field.

## Bounce-averaged $1/\nu$ transport

`bounce_averaged_transport(geometry, ...)` evaluates the effective ripple, the geometry
factor of $1/\nu$ transport (Velasco et al., J. Comput. Phys. 418, 109512, 2020; Nucl.
Fusion 61, 116059, 2021):

$$
\epsilon_{\mathrm{eff}}^{3/2} = \frac{\pi R_0^2}{8\sqrt2\,\langle|\nabla\psi|\rangle^2}\,\Gamma_c,
\qquad
\Gamma_c = \left\langle\Bigl(\int\frac{dl}{B}\Bigr)^{-1}
  \int_{B_{\min}}^{B_{\max}} d\rho\sum_{\text{wells}}\frac{H_j^2}{I_j}\right\rangle_{\text{field lines}},
$$

with the well integrals

$$
I_j = \int_{\text{well}}\sqrt{1 - \frac{B}{B_0\rho}}\,\frac{dl}{B},
\qquad
H_j = \rho^{-3/2}\int_{\text{well}}\sqrt{1 - \frac{B}{B_0\rho}}
      \left(\frac{4B_0\rho}{B} - 1\right)|\nabla\psi|\,\kappa_G\,\frac{dl}{B},
$$

and, in Boozer coordinates,
$|\nabla\psi|\kappa_G = -(G\,\partial_\theta B - I\,\partial_\zeta B)/(G + \iota I)$,
$R_0 = G/B_0$. Bounce points are found by smooth threshold crossings and a differentiable
Newton refinement on the spectral $|B|$, and each well integral uses a sine substitution
whose Jacobian cancels the square-root singularity (following arXiv:2412.01724), so
`jax.grad` flows through the whole pipeline. `effective_ripple` returns
$\epsilon_{\mathrm{eff}}$ alone.

For analytic schemes $\langle|\nabla\psi|\rangle$ uses the large-aspect-ratio circular
model; file geometries use the metric when available.

Checks in `tests/test_bounce_averaged.py`: $\Gamma_c < 10^{-12}$ for an axisymmetric field;
$\Gamma_c < 10^{-8}$ for a single-helicity (quasisymmetric) field; gradients match finite
differences to $10^{-5}$ relative. Against the full drift-kinetic solve on scheme 1, the
ratio of $D^*_{11}\nu^*$ between a doubled-ripple and a base field approaches the
surrogate's $\Gamma_c$ ratio ($3.58 \pm 0.1$) from below as $\nu'$ drops: $3.10 \pm 0.25$ at
$\nu' = 10^{-3}$, within 17 % of the limit. The surrogate is the $\nu \to 0$ asymptote,
not a finite-$\nu$ prediction, and it has no $\sqrt\nu$ or $E_r$ physics.

## Parallel-momentum correction

Pitch-angle scattering (PAS) conserves particles but not parallel momentum, so its flows
and bootstrap current are biased. `dkx.momentum_correction` keeps the cheap structured PAS
solve and restores momentum by the moment method of Sugama & Nishimura (Phys. Plasmas 9,
4637, 2002), the method Maassberg, Beidler & Turkin (Phys. Plasmas 16, 072504, 2009) apply
to stellarators. With $K_{\mathrm{PAS}}$ and $K_{\mathrm{FP}}$ the kinetic operators of one
deck under PAS and under the full linearized Fokker–Planck operator, the model equation is

$$
K_{\mathrm{PAS}}\,f = S + D\,P f, \qquad D = K_{\mathrm{PAS}} - K_{\mathrm{FP}},
$$

the full operator on the Sonine part $Pf$ of the $\ell = 1$ distribution and PAS on the
rest. $Pf = \sum_k \alpha_{ak}(\theta,\zeta)\, x L_k^{(3/2)}(x^2) e^{-x^2}$ carries the
parallel particle flow ($k = 0$), heat flow ($k = 1$) and, by default, the next Sonine
moment ($k = 2$, Maassberg's $j_x = 2$); two terms are the 13-moment approximation.
$D$ applied to a unit Sonine flow holds the Hirshman–Sigmar friction coefficients
$l^{ab}_{ij}$ and the field-particle momentum restoration, evaluated from DKX's own
Rosenbluth-potential operator (`friction_drives`; velocity space only, so it is computed
once and cached). The coefficients $\alpha$ are matched on the $\nu_D$-weighted moments
$\sum_x w\,\nu_D x^3 L_k f$, which leaves the remainder $f - Pf$ without PAS momentum: the
model conserves total parallel momentum exactly, so fluxes in a symmetric field are
intrinsically ambipolar. The local flow is a surface part $A_{ak}B/\langle B^2\rangle$ plus
a Pfirsch–Schlüter part set by the gradients alone, which the corrected state shares with
the plain PAS solution $g_0$; so

$$
f = g_0 + p + \sum_{bk} A_{bk} h_{bk}, \qquad
(I - W)A = \langle B\alpha(g_0 + p)\rangle, \quad W_{ij} = \langle B\alpha_i(h_j)\rangle,
$$

with $h_{bk}$ the PAS responses to the unit drives $D\phi_{bk}B/\langle B^2\rangle$ (their
surface flows are the energy-convolved viscosity coefficients, the monoenergetic
convolution done on DKX's speed grid) and $p$ the response to the friction of $g_0$'s
Pfirsch–Schlüter flow. $\langle j\cdot B\rangle$, the flows and the radial fluxes are then
read off $f$ with the ordinary moment table, so the fluxes carry their back-substituted
correction. Cost: one PAS elimination with $1 + NS$ right-hand sides and one more solve,
for $N$ Sonine terms and $S$ species, plus an $NS \times NS$ dense solve; everything is
traceable and differentiable (`momentum_corrected_solve`; in a VMEX objective,
`KineticBootstrapMismatch(..., collision_model="pas+momentum_correction")`).

Measured against full Fokker–Planck on the same grid (`tests/test_momentum_correction.py`
and the validation in the pull request that introduced it):

| Deck | PAS / FP − 1 | corrected / FP − 1 |
| --- | --- | --- |
| p + e tokamak, test deck (3 Sonine) | +8.4 % | −5.6 % |
| H + C$^{6+}$ tokamak, test deck (1–5 Sonine) | −84 % | −2.4 to −3.1 % |
| p + e tokamak, $\epsilon_t = 0.05$, Maassberg gradients, $15\times1\times48\times10$; $\nu_n = 8.3\times10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}$ (×8.33) | +55, +45, +64, +146 % | −9.1, −7.2, −5.3, +2.6 % |
| W7-X standard (geometryScheme 4), same plasma, $15\times25\times48\times8$; $\nu_n = 8.3\times10^{-5}, 10^{-4}, 10^{-3}$ (×8.33) | +9.2, +39, +264 % | −9.4, −2.7, +18 % |
| precise QA, $\beta = 2.5\,\%$ (VMEC), $21\times31\times32\times5$, $s = 0.25, 0.75$ | +85, +28 % | −11, −7.0 % |

The W7-X row at $\nu_n = 8.3\times10^{-3}$ sits next to the sign change of the current, so
its relative error is of a small number. On the QA equilibrium the same grid puts full
Fokker–Planck at 0.895 and 0.915 of Redl; the $s = 0.5$ surface is not resolved on it
(Fokker–Planck at 2.1 times Redl) and is left out. The high- and low-mirror W7-X
configurations of Maassberg et al. are not among DKX's decks, so their 88/104 kA and
19.5/28.4 kA totals were not reproduced. The published admission is 5 % in the banana
and plateau regimes; the tokamak and QA banana rows miss it by a few percent, so full
Fokker–Planck stays the default for bootstrap rows. Nor is the corrected route cheaper on
these grids: on four shared host cores it took 47–101 s per surface on the QA deck and
46–53 s on the W7-X deck, against 22–37 s for the Fokker–Planck Krylov solve, because
the structured elimination of $1 + NS$ columns over dense $(\theta,\zeta)$ blocks dominates.

Fluxes: on the p + e test deck $\sum_a Z_a\Gamma_a/\Gamma_i$ is 0.96 for PAS and
$-2\times10^{-3}$ corrected, at the deck's discretization level ($3\times10^{-3}$ for FP);
the corrected ion flux is within 3.2 % of FP's. Limits: the radial electric field enters
only through the PAS operator, so $E^* \ll 1$ is assumed (Maassberg et al.); PAS on the
$\ell \ne 1$ part leaves a few-percent residual in the tokamak banana regime that more
Sonine terms do not remove; and Mollén et al. (Phys. Plasmas 22, 112508, 2015) report a
low-collisionality inter-species coefficient at $E_r = 0$ that no moment correction
reproduces.

## Local validity flags

`dkx.validity.local_validity_report` turns the orderings behind a local, monoenergetic
result into pass/marginal/fail flags:

- Radial locality: $\delta_{\mathrm{FOW}} = w_b/L$ with orbit width
  $w_b = \hat\rho/(|\iota|\sqrt{\epsilon_t})$ and $L$ the shortest profile gradient
  length (Hinton & Hazeltine, Rev. Mod. Phys. 48, 239, 1976). Pass below 0.1, marginal
  below 0.3, fail above.
- Collisionality regime (Pfirsch-Schlüter, plateau, banana or $1/\nu$, $\sqrt\nu$,
  superbanana-plateau) from $\nu^*$, the ratio $k_{E\times B} = \omega_E/\nu_{\mathrm{eff}}$
  ($\sqrt\nu$ onset at 1, marginal from 0.3) and $k_{\mathrm{res}} = \omega_E/\omega_d$
  (superbanana-plateau resonance within a factor 3 of 1) (Shaing, Phys. Fluids 27, 1567,
  1984; Ho & Kulsrud, Phys. Fluids 30, 442, 1987).

Thresholds are the defaults of `RegimeThresholds`; the collisionality boundaries are the
standard orderings and the others are deliberate order-one bands.

## Kinetic bootstrap current for equilibrium optimization

`dkx.bootstrap` exposes the drift-kinetic $\langle j_\parallel B\rangle$ as an objective
term for `vmex.optimize`, the kinetic counterpart of the Redl analytic formula (Redl et al.,
Phys. Plasmas 28, 022502, 2021). `KineticBootstrapMismatch` is traced through the
equilibrium and differentiated with the VMEX implicit Jacobian;
`KineticBootstrapCurrent` evaluates a written `wout`, optionally at the ambipolar root.
The default grid is $N_\theta = 21$, $N_\zeta = 31$, $N_\xi = 32$, $N_x = 5$
(`DEFAULT_RESOLUTION`). See {doc}`../tutorials/vmex_optimization`.
