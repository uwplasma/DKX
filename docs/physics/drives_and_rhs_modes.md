# Drives, RHS modes and moments

The right-hand side $S_s$ of the drift-kinetic equation ({doc}`drift_kinetic_equation`)
holds the free-energy sources: radial gradients of density, temperature and potential, and
the inductive parallel field. `RHSMode` selects whether the code solves once with the
physical drives or several times with unit drives to build a transport matrix. The drive is
`KineticOperator.rhs`, which reproduces the v3 `evaluateResidual.F90` evaluated at
$f = 0$.

## The drives

The radial drift of the Maxwellian gives the thermodynamic drive (sign conventions as in the
v3 technical documentation),

$$
S_s^{\mathrm{grad}} \propto (\mathbf{v}_m + \mathbf{v}_E)\cdot\nabla\psi\;
\left[\frac{1}{n_s}\frac{dn_s}{d\psi}
    + \frac{Z_s e}{T_s}\frac{d\Phi_0}{d\psi}
    + \left(x_s^2 - \frac32\right)\frac{1}{T_s}\frac{dT_s}{d\psi}\right] f_{s0}.
$$

In normalized form (`KineticOperator._rhs`) its $x$ and angle dependence is

$$
S_s^{\mathrm{grad}} = \frac{\Delta\,\hat n_s\,\hat m_s^{3/2}}{2\pi^{3/2} Z_s\sqrt{\hat T_s}}\;
\hat g_2(\theta,\zeta)\; x^2 e^{-x^2}
\left[\frac{1}{\hat n_s}\frac{d\hat n_s}{d\hat\psi}
    + \frac{Z_s\alpha}{\hat T_s}\frac{d\hat\Phi}{d\hat\psi}
    + \left(x^2 - \frac32\right)\frac{1}{\hat T_s}\frac{d\hat T_s}{d\hat\psi}\right],
\qquad
\hat g_2 = \frac{\hat D\,(\hat B_\zeta\,\partial_\theta\hat B - \hat B_\theta\,\partial_\zeta\hat B)}{\hat B^3},
$$

projected onto Legendre mode $L=0$ with weight $4/3$ and $L=2$ with weight $2/3$ (the
$v_\parallel^2 + v_\perp^2/2$ structure of the magnetic drift). The inductive drive acts on
$L = 1$:

$$
S_s^{\mathrm{ind}} \propto Z_s\alpha\,x\,e^{-x^2}\,
\frac{\hat n_s\hat m_s}{\hat T_s^2\,\langle\hat B^2\rangle}\,\hat B\,\hat E_\parallel,
$$

with $\hat E_\parallel$ = `EParallelHat` plus the per-species `EParallelHatSpec`, applied as
a flux-surface constant. A spatially varying `EParallelHatSpec_bcdatFile` is refused
(`src/dkx/inputs.py`).

The native case executor sets $\hat E_\parallel = 0$ and computes the density and
temperature gradients from the profile values at neighbouring surfaces, which is why a
native case needs at least two surfaces (`_validate_native_slice`).

## `RHSMode`

| `RHSMode` | Solves | Output | Route |
| --- | --- | --- | --- |
| 1 | one RHS with the physical drives | fluxes, flows, current per species | namelist and native case |
| 2 | 3 RHS columns (`whichRHS` 1-3) | $3\times3$ thermal transport matrix | namelist |
| 3 | 2 RHS columns (`whichRHS` 1-2), single speed $x=1$ | $2\times2$ monoenergetic matrix | namelist |

For `RHSMode` 2 and 3 the code overwrites the gradients and $\hat E_\parallel$ internally
before building each column (`KineticOperator._with_rhs_settings`, v3 `solver.F90`), and
drops the $d\hat\Phi/d\hat\psi$ term from the drive so each column isolates one force:

| Mode | `whichRHS` | $d\hat n/d\hat\psi$ | $d\hat T/d\hat\psi$ | $\hat E_\parallel$ |
| --- | --- | --- | --- | --- |
| 2 | 1 | 1 | 0 | 0 |
| 2 | 2 | $\tfrac32\hat n\hat T$ (zero pressure drive) | 1 | 0 |
| 2 | 3 | 0 | 0 | 1 |
| 3 | 1 | 1 | 0 | 0 |
| 3 | 2 | 0 | 0 | 1 |

The $E\times B$ drift itself stays in the operator, so the matrix is evaluated at the given
$E_r$. `RHSMode = 3` uses the first species only; SFINCS and DKX warn and drop the others
(`src/dkx/inputs.py`). Its collisionality and field are `nuPrime` and `EStar`
({doc}`normalizations`).

## Moments

All outputs are moments of the solved $f_{s1}$ (`dkx.moments`). Radial fluxes are formed
against $\nabla\hat\psi$ and converted to the other radial coordinates afterwards.

Particle and heat flux
: The magnetic-drift fluxes use the geometric factor
  $\hat g_{vm} = (\hat B_\theta\,\partial_\zeta\hat B - \hat B_\zeta\,\partial_\theta\hat B)/\hat B^3$
  and the Legendre weights $8/3$ ($L=0$) and $4/15$ ($L=2$):

  $$
  \Gamma_s = \left\langle\int d^3v\; f_{s1}\,\mathbf{v}_{m}\cdot\nabla\psi\right\rangle,
  \qquad
  Q_s = \left\langle\int d^3v\; \frac{m_s v^2}{2}\,f_{s1}\,\mathbf{v}_{m}\cdot\nabla\psi\right\rangle .
  $$

  These are `particleFlux_vm_psiHat` and `heatFlux_vm_psiHat` (`vm_flux_moments`). With
  $\Phi_1$ there are also $E\times B$ fluxes (`_vE`, `electric_drift_flux_moments`) whose
  factor $(\hat B_\theta\,\partial_\zeta\Phi_1 - \hat B_\zeta\,\partial_\theta\Phi_1)/\hat B^2$
  has $\hat B^2$ in the denominator, and `_vm0` variants from $f_{s0}$ alone.

Parallel flow and bootstrap current
: The $L = 1$ moment gives the parallel flow; its charge-weighted sum is the bootstrap
  current:

  $$
  \mathrm{FSABFlow}_s = \langle\hat B\,\hat n_s\hat V_{\parallel s}\rangle,
  \qquad
  \mathrm{FSABjHat} = \langle\hat{\mathbf{j}}\cdot\hat{\mathbf{B}}\rangle
                    = \sum_s Z_s\,\mathrm{FSABFlow}_s .
  $$

  `jHat` is the parallel current density on the $(\theta,\zeta)$ grid. SI factors are in
  {doc}`normalizations`.

Radial current
: $J_r = \sum_s Z_s\Gamma_s$, the function whose zero is the ambipolar field
  ({doc}`electric_field`).

Neoclassical toroidal viscosity
: `ntv_moments` evaluates the NTV torque moment for non-axisymmetric geometry.

Classical fluxes
: The classical (gyroradius, friction-driven) particle and heat fluxes, written as
  `classicalParticleFlux*` and `classicalHeatFlux*`, need the metric
  $|\nabla\hat\psi|^2$ and are produced for geometries that supply it (VMEC and Boozer
  `.bc`). See {doc}`phi1_and_impurities`.

## Transport matrix

`transport_matrix_from_flux_arrays` assembles the `transportMatrix` entries of v3
`diagnostics.F90` from the per-column fluxes and flows of the first species. The response
columns are the radial particle flux, radial heat flux (mode 2 only) and parallel flow; the
forces are the columns listed above. The entries carry the factors
$\hat G + \iota\hat I$, $\hat B_0/\hat G^2$ and $\Delta^{-2}$; the particle-flux entries
are

$$
L_{1j} = \frac{4}{\Delta^2}\sqrt{\frac{\hat T}{\hat m}}\;Z^2\,
\frac{(\hat G + \iota\hat I)\,\hat B_0}{c_j\,\hat G^2}\;\Gamma^{(j)},
$$

where $\Gamma^{(j)}$ is `particleFlux_vm_psiHat` from column $j$ and $c_j$ is a
column-dependent factor (`_pf_gradient_entry` in `src/dkx/moments.py`). Heat-flux entries
carry $8/\Delta^2$ in place of $4/\Delta^2$.

For `RHSMode = 3` the $2\times2$ matrix is the monoenergetic (DKES-normalized) one, related
to the $D_{11}$, $D_{31}$, $D_{13}$, $D_{33}$ coefficients of Beidler et al., Nucl. Fusion
51, 076001 (2011), and Hirshman et al., Phys. Fluids 29, 2951 (1986). With the
normalization above the matrix is Onsager-symmetric, $L_{12} = L_{21}$;
`tests/test_transport_limits.py` measures the asymmetry under refinement.

```{figure} ../_static/figures/docs/transport_coeff_vs_collisionality.png
:alt: Monoenergetic transport coefficients versus normalized collisionality for a three-helicity model field.
:width: 78%

Monoenergetic (`RHSMode = 3`) transport-matrix entries versus $\nu'$ for the
three-helicity model field (`geometryScheme = 1`). $|L_{11}|$ ($D_{11}$-like) rises as
$\nu'$ drops, the $1/\nu$ regime of a non-axisymmetric field; $|L_{21}|$ ($D_{31}$-like) is
the bootstrap coefficient. Generated by `docs/figures/generate_docs_figures.py`.
```

The monoenergetic database, its energy convolution into thermal coefficients, and the
variational bounds on $D_{11}$ are in {doc}`reduced_models`.
