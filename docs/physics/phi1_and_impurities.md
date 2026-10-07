# $\Phi_1$, quasineutrality and impurities

Heavy impurities respond strongly to small variations of the electrostatic potential on a
flux surface, because the Boltzmann factor $e^{-Z e\Phi_1/T}$ grows with charge. SFINCS
includes this variation as an extra unknown closed by quasineutrality (Mollén et al.,
Plasma Phys. Control. Fusion 60, 084001, 2018). DKX implements the same model on the
namelist route, and its kinetic-equation form natively (`physics.phi1 = "kinetic"`).
Its validation status is `validated_limited`: it matches Fortran SFINCS v3 to 2.0e-6
on one geometry with Fokker–Planck collisions and zero $E_r$ (the benchmark below);
`stable_candidate` is withheld until more geometries, collision operators and finite
$E_r$ are covered. It also provides the classical (friction-driven) impurity flux, which is
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

- The native case accepts `physics.phi1 = "kinetic"` on a `profile` workflow: $\Phi_1$ in
  the kinetic equation, quasineutrality option 1, no adiabatic species. It runs the same
  Newton–Krylov solve as the namelist route and is accepted by the nonlinear residual.
  `dkx.phi1.phi1_solution` differentiates through the converged state with a
  matrix-free implicit adjoint that works with Legendre truncation.
  `"full"`, option 2, adiabatic species and `ambipolar_profile` go through a SFINCS namelist
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

## Fortran SFINCS benchmark

DKX against pinned Fortran SFINCS v3 on the geometryScheme 4 Phi1 example deck (`main`, ions and electrons) and a
trace-C6+ variant (`imp`). Fluxes are `particleFlux_vm_psiHat`. The rung written in bold is the deck's own resolution. Data, settings and limits:
`validation/phi1_sfincs_benchmark_v1.json`.

| case | Ntheta, Nzeta, Nxi, Nx | DKX fluxes (i, e[, C]) | DKX Phi1 rms | max rel. diff. vs SFINCS |
|---|---|---|---|---|
| main_A | 9, 13, 32, 4 | -1.06913e-07, 2.13643e-08 | 4.44803e-04 | 1.2e-07 |
| main_B | **13, 19, 48, 5** | -1.13440e-07, 2.16511e-08 | 4.51407e-04 | 2.7e-07 |
| main_C | 17, 25, 64, 6 | -1.13954e-07, 2.15501e-08 | 4.50735e-04 | 5.7e-07 |
| imp_A | 9, 13, 32, 4 | -9.74765e-08, 1.40626e-08, 2.53423e-11 | 3.07182e-04 | 2.8e-07 |
| imp_B | **13, 19, 48, 5** | -1.02856e-07, 1.38749e-08, 2.17061e-11 | 3.05901e-04 | 8.2e-07 |
| imp_C | 17, 25, 64, 6 | -1.03030e-07, 1.40006e-08, 2.58225e-11 | 3.06048e-04 | 2.0e-06 |
| imp_Bx | 13, 19, 48, 6 | -1.02953e-07, 1.39673e-08, 2.55758e-11 | 3.06031e-04 | 2.0e-06 |
| imp_Bx7 | 13, 19, 48, 7 | -1.02887e-07, 1.40382e-08, 2.43575e-11 | 3.06249e-04 | 1.2e-06 |
| imp_Bx8 | 13, 19, 48, 8 | -1.02927e-07, 1.39502e-08, 2.32806e-11 | 3.06392e-04 | SFINCS not run |
| imp_Bx10 | 13, 19, 48, 10 | -1.02895e-07, 1.40064e-08, 2.48247e-11 | 3.06138e-04 | SFINCS not run |
| imp_Bx12 | 13, 19, 48, 12 | -1.02865e-07, 1.40410e-08, 2.40797e-11 | 3.07501e-04 | SFINCS not run |

Main-deck fluxes and Phi1 move under 0.5% from the deck's resolution to the next rung. The trace
carbon flux, about 4e-4 of the ion flux, oscillates in Nx: it moves 18% from Nx 5 to 6, then 3% from Nx 10 to 12.
Both codes show the same oscillation at every rung, so this is a speed-grid resolution limit, not a
discrepancy between the codes. Run SFINCS with `-mat_mumps_icntl_7 2`, because the default METIS ordering hangs.
