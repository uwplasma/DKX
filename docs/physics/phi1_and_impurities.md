# $\Phi_1$, quasineutrality and impurities

Heavy impurities respond strongly to small variations of the electrostatic potential on a
flux surface, because the Boltzmann factor $e^{-Z e\Phi_1/T}$ grows with charge. SFINCS
includes this variation as an extra unknown closed by quasineutrality (Mollén et al.,
Plasma Phys. Control. Fusion 60, 084001, 2018). DKX implements the same model on the
namelist route. It also provides the classical (friction-driven) impurity flux, which is
algebraic and needs no kinetic solve.

## Potential split

The potential is

$$
\Phi(\psi,\theta,\zeta) = \Phi_0(\psi) + \Phi_1(\theta,\zeta),
$$

and the leading-order distribution carries the Boltzmann factor

$$
f_{s0} = f_{sM}\,\exp\!\left(-\frac{Z_s\alpha\,\hat\Phi_1}{\hat T_s}\right).
$$

$\Phi_1$ enters in three places:

- the background $f_{s0}$ and therefore the density, which is no longer a flux function;
- the kinetic equation, through the parallel electric field $-\nabla_\parallel\Phi_1$ and
  the $E\times B$ drift of $\Phi_1$ (`includePhi1InKineticEquation`, default `.true.`);
- the collision operator, through the poloidally varying densities
  $\hat n_s e^{-Z_s\alpha\hat\Phi_1/\hat T_s}$ (`includePhi1InCollisionOperator`, which
  requires `collisionOperator = 0`).

## Quasineutrality closure

With `includePhi1 = .true.` the state vector grows by $N_\theta N_\zeta$ values of
$\Phi_1$ and one scalar $\lambda$, laid out `[f | Phi1 | lambda | sources]`. The added rows
are

$$
\lambda + \sum_s Z_s\int d^3v\,(f_{s0} + f_{s1}) = 0 \quad\text{at each }(\theta,\zeta),
\qquad
\langle\Phi_1\rangle = 0 .
$$

The second row fixes the gauge; $\lambda$ is its Lagrange multiplier
(`KineticOperator._quasineutrality_rows`). Two closures are implemented:

| `quasineutralityOption` | Charge density |
| --- | --- |
| 1 (full, default) | all kinetic species, with the nonlinear Boltzmann response of $f_{s0}$ |
| 2 (EUTERPE form) | the first kinetic species only, linearized, plus an optional adiabatic species (`withAdiabatic`) as a diagonal term in $\Phi_1$ |

Other values are refused. The adiabatic species defaults to $Z = -1$,
$\hat m = 5.446\times10^{-4}$ (the electron-proton mass ratio), $\hat n = \hat T = 1$
(`src/dkx/constants.py`). A neutral-beam species (`withNBIspec`) combined with a solved
$\Phi_1$ is refused, because DKX does not add the beam charge to quasineutrality
(`src/dkx/inputs.py`).

## Nonlinear solve

Because $f_{s0}$ depends exponentially on $\Phi_1$, the coupled system is nonlinear.
`dkx.phi1.solve_phi1` is a Newton iteration, the counterpart of the SFINCS PETSc `SNES`
loop: each step linearizes the residual `KineticOperator.residual_phi1`, solves
$J\,\delta x = -r$ with the Krylov route on the matrix-free Jacobian-vector product, and
warm-starts across iterations. `dkx.phi1.phi1_state` is the differentiable version: the
fixed point $F(x) = 0$ is wrapped in `solvax.implicit.root_solve`, so gradients of any
moment follow $dx/dp = -(\partial F/\partial x)^{-1}\partial F/\partial p$.

`readExternalPhi1 = .true.` reads a fixed $\Phi_1(\theta,\zeta)$ from file instead. There
is then no quasineutrality block, no $\Phi_1$ unknown and no $\lambda$ row; the system is
linear again, with the external field entering the same $\Phi_1$ terms.

## Fluxes with $\Phi_1$

With $\Phi_1$ the radial flux has an $E\times B$ part in addition to the magnetic-drift
part. DKX writes both families (`particleFlux_vm_*`, `particleFlux_vE_*`, and their sums);
the $E\times B$ geometric factor
$(\hat B_\theta\,\partial_\zeta\hat\Phi_1 - \hat B_\zeta\,\partial_\theta\hat\Phi_1)/\hat B^2$
is in `dkx.moments.electric_drift_flux_moments` ({doc}`drives_and_rhs_modes`).

## Scope

- The native case accepts `physics.phi1 = "off"` only; `"kinetic"` and `"full"` validate
  in the case format but the executor refuses them. $\Phi_1$ runs go through a SFINCS namelist
  ({doc}`../user_guide/sfincs_namelist`).
- Full $\Phi_1$ coupling with the collision operator is verified as a full-system matvec
  and residual against the v3 fixture `fp_1species_FPCollisions_noEr_tiny_withPhi1_inCollision`
  (see {doc}`../benchmarks/sfincs`); broader $\Phi_1$ configurations have less coverage.

## Classical impurity transport

The classical flux (gyro-motion plus inter-species friction, without the
Pfirsch-Schlüter part) is local and algebraic. `dkx.impurity` evaluates it from the
Braginskii friction moments (Braginskii, Rev. Plasma Phys. 1, 205, 1965) with the
Maxwellian background:

$$
\Gamma_a^{\mathrm{cl}} = Z_a\,\frac{\Delta^2\nu_n\sqrt{\hat m_a}}{2\hat T_a^{3/2}}\;\mathcal{G}
\sum_b Z_b^2\,\hat n_a\hat n_b\Bigl[M^{ab}_{00}(u^a_n - u^b_n)
  + (M^{ab}_{00} - M^{ab}_{01})\,u^a_T
  - (M^{ab}_{00} - x_{ab}^2 M^{ab}_{01})\,u^b_T\Bigr],
$$

with $u^a_n = \hat T_a\hat n_a'/(\hat n_a Z_a)$, $u^a_T = \hat T_a'/Z_a$ (primes are
$d/d\hat\psi$) and $x_{ab}^2 = \hat m_a\hat T_b/(\hat m_b\hat T_a)$. All geometry enters
through one scalar,

$$
\mathcal{G} = \left\langle\frac{|\nabla\hat\psi|^2}{\hat B^2}\right\rangle
$$

(`classical_geometry_factor`). The metric $|\nabla\hat\psi|^2$ is available for VMEC and
Boozer `.bc` geometries; the analytic schemes carry a zero placeholder, so classical fluxes
are zero there.

For a trace impurity the flux takes the textbook form (Rutherford, Phys. Fluids 17, 1782,
1974; Helander & Sigmar 2002)

$$
\frac{\Gamma_z}{n_z} = -D_z^{\mathrm{cl}}\left[g_{n_z} - Z_z g_{n_i} + H Z_z g_{T_i} + g_{T_z}\right],
\qquad g_x = \frac{1}{x}\frac{dx}{d\hat\psi},
$$

where $-Z_z$ is the density-peaking coefficient and $H$ the temperature-screening
coefficient. In the collisional heavy-impurity limit $H \to 1/2$ (Wenzel & Sigmar, Nucl.
Fusion 30, 1117, 1990). A peaked ion temperature then opposes the inward pinch.

| Function | Returns |
| --- | --- |
| `build_impurity_plasma` | bulk plasma plus a trace or non-trace impurity as a `SpeciesSet` |
| `classical_impurity_flux` | impurity flux split by drive |
| `classical_diffusion_coefficient` | $D_z^{\mathrm{cl}}$ |
| `temperature_screening_diagnostic` | $H$, the $-Z_z$ coefficient, and whether screening opposes the pinch |
| `classical_impurity_flux_over_charge_states` | the flux vectorized over charge states |

The multi-species algebra reproduces the $\Phi_1 = 0$ branch of SFINCS
`classicalTransport.F90:calculateClassicalFlux`, and `classical_species_fluxes` agrees with
`dkx.moments.classical_fluxes` (module docstring of `dkx.impurity`). All functions are
pure JAX and differentiable.
