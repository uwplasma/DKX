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

Pitch-angle scattering conserves particles but not parallel momentum, so flows and
bootstrap currents from a monoenergetic database are momentum-deficient.
`dkx.momentum_correction` applies the moment method of Sugama & Nishimura (Phys. Plasmas 9,
4637, 2002; 15, 042502, 2008) as used by Maassberg, Beidler & Turkin (Phys. Plasmas 16,
072504, 2009). For species parallel flows $V_a = \langle BV_{\parallel a}\rangle$:

$$
\bigl(\operatorname{diag}(M_a) + \Lambda\bigr)\,V = \operatorname{diag}(M_a^{(0)})\,V^{\mathrm{unc}},
\qquad
\Lambda_{ab} = \delta_{ab}\sum_c\gamma_{ac} - \gamma_{ab},
$$

where $M_a$ is the parallel viscosity from an energy convolution of $D_{33}$, $M_a^{(0)}$
the same without the like-particle momentum restoration, and $\gamma_{ab} = \gamma_{ba}$
the parallel friction coefficients. The corrected bootstrap current is
$\langle Bj_\parallel\rangle = \sum_a Z_a V_a$ (`momentum_corrected_bootstrap`).

Measured in `tests/test_momentum_correction.py` (module docstring): single-species
restoring factor $M^{(0)}/M = 0.9310$ on the scheme 1 test deck; friction-matrix momentum
conservation $\le 10^{-18}$; for a two-species H + C$^{6+}$ deck the bootstrap difference
from a full Fokker-Planck solve drops from $3.81\times10^{-2}$ (uncorrected pitch-angle
scattering) to $6.9\times10^{-3}$ (corrected). The correction is the single-moment
restoration and does not capture the energy-scattering difference between the operators, so
it does not reach the Fokker-Planck value.

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
