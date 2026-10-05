# Collision operators

The collision term $\sum_b C^{\mathrm{lin}}_{sb}[f_{s1}, f_{b1}]$ is linearized about the
Maxwellians of all species. DKX implements the two SFINCS v3 operators and one extension,
selected by `collisionOperator` in a namelist or `physics.collisions` in a native case:

| Operator | Namelist | Native case | Conserves |
| --- | --- | --- | --- |
| Full linearized Fokker-Planck | `collisionOperator = 0` (default) | `linearized_fokker_planck` (default) | particles, momentum, energy |
| Pitch-angle scattering | `collisionOperator = 1` | `pitch_angle_scattering` | particles, energy (not momentum) |
| Improved Sugama model operator | `collisionOperator = 3` | not available | particles, momentum, energy |

Other `collisionOperator` values raise `NotImplementedError` in
`kinetic_operator_from_namelist`. All operators are built in `dkx.collisions` and applied
inside `KineticOperator.apply`. Every block is diagonal in $(\theta,\zeta)$ and in the
Legendre index $L$.

## Pitch-angle scattering

The Lorentz operator is diagonal in $(\theta,\zeta,x,L)$ with the $L(L+1)$ eigenvalues of
the Legendre polynomials:

$$
\left(C^{\mathrm{PAS}} f\right)_{sL} = -\,\nu_n\,\hat\nu_D^s(x)\,\frac{L(L+1)}{2}\,f_{sL}.
$$

The deflection frequency sums over collision partners $b$
(`nu_d_hat_pitch_angle_scattering_v3`):

$$
\hat\nu_D^s(x) = \frac{3\sqrt\pi}{4}\,\frac{Z_s^2}{\hat T_s\sqrt{\hat T_s\hat m_s}}
\sum_b Z_b^2\,\hat n_b\,\frac{\operatorname{erf}(x_b) - \Psi(x_b)}{x^3},
\qquad
x_b = x\sqrt{\frac{\hat T_s\hat m_b}{\hat T_b\hat m_s}},
$$

with the Chandrasekhar function

$$
\Psi(x) = \frac{\operatorname{erf}(x) - \frac{2}{\sqrt\pi}\,x\,e^{-x^2}}{2x^2}.
$$

The $x^3$ in the denominator is on the base grid of species $s$, as in the Fortran
(`populateMatrix.F90`). Because the operator is diagonal in $L$, the pitch-angle-scattering
system is block-tridiagonal in $L$ and admits the structured direct solve
({doc}`../numerics/solver_routes`).

Pitch-angle scattering does not conserve parallel momentum. Flows and bootstrap currents
computed with it are momentum-deficient; {doc}`reduced_models` describes the
Sugama-Nishimura correction in `dkx.momentum_correction`, which keeps the structured
pitch-angle solve and restores momentum through the projection of
$C_{\mathrm{FP}} - C_{\mathrm{PAS}}$ onto the Sonine flow moments of every species.

## Full linearized Fokker-Planck

The linearized Landau operator adds energy scattering and the field-particle
back-reaction to pitch-angle deflection. For each species pair $(a,b)$ and Legendre mode
$L$ it is a dense matrix on the speed grid. Schematically (the individual kernels and their
signs are those of the v3 technical documentation and `populateMatrix.F90`):

$$
\left(C^{\mathrm{FP}} f\right)_{a,L}
= \nu_n\sum_b\Bigl(
    \underbrace{C^{E}_{ab}}_{\text{energy scattering}}
  + \underbrace{C^{D}_{ab}}_{\text{drag}}
  + \underbrace{R^{L}_{ab}}_{\text{field particle}}\Bigr) f
  - \nu_n\,\hat\nu_D^a\,\frac{L(L+1)}{2}\,f_{a,L} .
$$

The test-particle part ($C^E$, $C^D$, $\hat\nu_D$) acts on $f_a$; the field-particle part
$R^L$ acts on $f_b$ through the Rosenbluth potentials $H$ and $G$ of the perturbation. For
each $L$ the code builds the speed integrals that give $H$, $dH/dx$ and $d^2G/dx^2$ and maps
them back to the collocation grid:

$$
H_L(x) \propto \frac{1}{2L+1}\left(
  \frac{1}{x^{L+1}}\int_0^x t^{L+2}\,\hat f_L(t)\,dt
+ x^L\int_x^\infty t^{1-L}\,\hat f_L(t)\,dt\right),
$$

and similarly for $G$. The construction follows Landreman & Ernst, J. Comput. Phys. 243,
130 (2013), and the SFINCS implementation of Mollén et al., arXiv:1504.04810 (2015): the
map from collocation values to potentials is a transform to a modal basis followed by
integral-evaluation matrices. Conservation of momentum and energy is structural: it follows
from the exact linearized field term, with no moment-restoring projection (an optional Krook
term exists but is zero by default, `krook=0.0` in the native builder).

Species coupling happens only through the field term and the cross-species $\hat\nu_D$, so
the Fokker-Planck block is dense in (species, $x$) but independent of $(\theta,\zeta)$. That
dense block is the main cost of a multi-species run. The Rosenbluth response matrices are
evaluated with adaptive quadrature (`scipy.integrate.quad`, QUADPACK as in v3).

Limits:

- `collisionOperator = 0` with the uniform or Chebyshev speed grids (`xGridScheme` 3, 4, 7,
  8) is refused; the Fortran interpolation matrices for those grids are not ported
  (`src/dkx/drift_kinetic.py` module docstring).
- `includePhi1InCollisionOperator = .true.` requires `collisionOperator = 0`; the operator
  then uses the poloidally varying densities $\hat n_s e^{-Z_s\alpha\Phi_1/\hat T_s}$
  ({doc}`phi1_and_impurities`).

## Improved Sugama model operator

`collisionOperator = 3` is a DKX extension beyond SFINCS v3: the momentum- and
energy-conserving improved linearized model operator of Sugama, Matsuoka, Satake, Nunami &
Watanabe, Phys. Plasmas 26, 102108 (2019), with the moment-based field-particle
construction of Frei, Ernst & Ricci, Phys. Plasmas 29, 093902 (2022). The test-particle
part reuses the Fokker-Planck deflection and energy-diffusion kernels. The field-particle
part is a low-rank moment term in $L=0$ (particle and energy) and $L=1$ (parallel
momentum) whose coefficients cancel the test-particle moment functionals algebraically, so
conservation is exact on the collocation grid. It assembles into the same per-$L$ dense
speed blocks and uses the same matvec (`apply_fokker_planck_v3`). Its speed nullspace is
{density, temperature} per species, so the default constraint is `constraintScheme = 1`.

This operator is available from a namelist only; the native case accepts
`linearized_fokker_planck` and `pitch_angle_scattering`.

## Temperature equilibration is not included

DKX builds only the part of the collision operator that acts on $f_{s1}$. The SFINCS option
`includeTemperatureEquilibrationTerm = .true.`, which adds $C_{ab}[f_{aM}, f_{bM}]$ (the heat
exchange between species at unequal temperatures) to the drive, is refused at deck-read
time for `RHSMode` 1 and 2 (`src/dkx/inputs.py`). For `RHSMode = 3` SFINCS disables it
itself, so it is accepted there and ignored.

## Choosing an operator

- Multi-species fluxes, flows and bootstrap current: Fokker-Planck. It is the native
  default.
- Monoenergetic coefficients, DKES-type benchmarks and ICNTS comparisons (Beidler et al.,
  Nucl. Fusion 51, 076001, 2011): pitch-angle scattering, which is what those benchmarks
  define.
- High collisionality: the Sugama model operator is designed to stay accurate there
  (Sugama et al. 2019).
