# The drift-kinetic equation

DKX solves for the first-order distribution $f_{s1}$ of every kinetic species $s$ on a
single flux surface. Radial profiles and their gradients are held fixed at that surface
(the radially local approximation), time derivatives are dropped (steady state), and the
equation is linearized about a Maxwellian. This is the model of SFINCS version 3
(Landreman, Smith, Mollén & Helander, Phys. Plasmas 21, 042503, 2014); the textbook
derivation is in Helander & Sigmar, *Collisional Transport in Magnetized Plasmas* (2002).

The whole operator is one class, `dkx.drift_kinetic.KineticOperator`. Its matrix-free
action is `KineticOperator.apply`, the drives are `KineticOperator.rhs`, and a SFINCS
namelist is turned into an operator by `dkx.drift_kinetic.kinetic_operator_from_namelist`.

## The $f_0 + f_1$ split

The distribution is split into a leading-order part and a first-order correction,

$$
f_s = f_{s0} + f_{s1},
\qquad
f_{s0} = n_s\left(\frac{m_s}{2\pi T_s}\right)^{3/2} e^{-x_s^2}\,
         \exp\!\left(-\frac{Z_s e\,\Phi_1}{T_s}\right),
$$

where the Boltzmann factor is present only when the flux-surface potential variation
$\Phi_1(\theta,\zeta)$ is enabled ({doc}`phi1_and_impurities`); otherwise $f_{s0}$ is the
flux-function Maxwellian $f_{sM}$. The velocity coordinates are

$$
x_s = \frac{v}{\sqrt{2T_s/m_s}},
\qquad
\xi = \frac{v_\parallel}{v},
\qquad
\mu = \frac{v_\perp^2}{2B}.
$$

$x_s$ is discretized on a speed grid over $0 \le x \le x_{\max}$ and $\xi$ is expanded in
Legendre polynomials, $f_{s1} = \sum_L f_{sL}(\theta,\zeta,x)\,P_L(\xi)$
(see {doc}`../numerics/discretization`).

## The linearized equation

The steady linearized drift-kinetic equation is

$$
\underbrace{v_\parallel \mathbf{b}\cdot\nabla f_{s1}
  + \mathcal{M}[f_{s1}]}_{\text{streaming + mirror}}
+ \underbrace{\mathbf{v}_E\cdot\nabla f_{s1}}_{E\times B}
+ \underbrace{\mathbf{v}_m\cdot\nabla f_{s1}}_{\text{magnetic drift}}
+ \underbrace{\dot x\,\partial_x f_{s1} + \dot\xi\,\partial_\xi f_{s1}}_{E_r\ \text{energy and pitch drifts}}
- \sum_b C^{\mathrm{lin}}_{sb}[f_{s1}, f_{b1}]
= S_s .
$$

The source $S_s$ holds the radial thermodynamic drives and the inductive field
({doc}`drives_and_rhs_modes`); the collision operator is described in {doc}`collisions`.
The guiding-center drifts, with signed $\Omega_s = Z_s e B/m_s$, are

$$
\mathbf{v}_m = \frac{v_\parallel^2}{\Omega_s}\,\mathbf{b}\times(\mathbf{b}\cdot\nabla\mathbf{b})
             + \frac{\mu}{\Omega_s}\,\mathbf{b}\times\nabla B,
\qquad
\mathbf{v}_E = \frac{\mathbf{E}\times\mathbf{B}}{B^2}.
$$

Only their components tangential to the flux surface act on $f_{s1}$; the radial
components appear in the drive and in the flux moments. This is what makes the model
radially local.

```{list-table} Terms of the operator and their Legendre coupling (`KineticOperator` docstring, `src/dkx/drift_kinetic.py`)
:header-rows: 1

* - Term
  - $L$ coupling
  - Enabling input (namelist)
* - Streaming
  - $L\pm1$
  - always on
* - Mirror force
  - $L\pm1$
  - always on
* - $E\times B$ ($\partial_\theta$, $\partial_\zeta$)
  - diagonal
  - nonzero `dPhiHatdpsiHat` / `Er`; form set by `useDKESExBDrift`
* - Tangential magnetic drift
  - $L$, $L\pm2$
  - `magneticDriftScheme` 1-9
* - $E_r$ term in $\dot\xi$
  - $L$, $L\pm2$
  - `includeElectricFieldTermInXiDot`
* - $E_r$ term in $\dot x$
  - $L$, $L\pm2$, dense in $x$
  - `includeXDotTerm`
* - Pitch-angle scattering
  - diagonal
  - `collisionOperator = 1`
* - Fokker-Planck
  - diagonal in $L$, dense in (species, $x$)
  - `collisionOperator = 0`
* - Sources and constraints
  - bordered rows and columns
  - `constraintScheme`
```

## Streaming and mirror

The parallel gradient in flux coordinates is

$$
\mathbf{b}\cdot\nabla = \frac{\hat B^\theta\,\partial_\theta + \hat B^\zeta\,\partial_\zeta}{\hat B}.
$$

With the recursion $\xi P_L = \frac{L+1}{2L+3}P_{L+1} + \frac{L}{2L-1}P_{L-1}$, written for
the Legendre coefficients, streaming couples each mode to its neighbours:

$$
\left(v_\parallel\mathbf{b}\cdot\nabla f\right)_L
= x\sqrt{\frac{\hat T_s}{\hat m_s}}\left[
    \frac{L+1}{2L+3}\,\mathbf{b}\cdot\nabla f_{L+1}
  + \frac{L}{2L-1}\,\mathbf{b}\cdot\nabla f_{L-1}\right].
$$

The coupling factors are `dkx.phase_space.legendre_coupling_upper` and
`legendre_coupling_lower`. The mirror force has the same $L\pm1$ structure with geometric
prefactor

$$
\mathcal{M} \propto -\,x\sqrt{\frac{\hat T_s}{\hat m_s}}\,
\frac{\mathbf{b}\cdot\nabla\hat B}{2\hat B},
$$

and upper and lower couplings scaled by $(L+2)$ and $-(L-1)$. Both terms are always
present (`KineticOperator._streaming_mirror`).

## $E\times B$ drift and the trajectory model

The equilibrium radial field $E_r = -d\Phi_0/dr$ advects $f_{s1}$ in the angles with
prefactors that are diagonal in $L$:

$$
F_{E\times B,\theta} = \frac{\alpha\Delta}{2}\frac{d\hat\Phi}{d\hat\psi}
                       \frac{\hat D\,\hat B_\zeta}{\mathcal{B}},
\qquad
F_{E\times B,\zeta} = -\frac{\alpha\Delta}{2}\frac{d\hat\Phi}{d\hat\psi}
                      \frac{\hat D\,\hat B_\theta}{\mathcal{B}},
$$

where $\hat D$ is the Jacobian factor and $\hat B_\theta$, $\hat B_\zeta$ are covariant
components. The denominator selects the trajectory model:

- full trajectories: $\mathcal{B} = \hat B^2$;
- DKES trajectories (`useDKESExBDrift = .true.`): $\mathcal{B} = \langle\hat B^2\rangle$,
  the incompressible $E\times B$ drift of the DKES code.

The full trajectory model adds two more $E_r$ terms (`KineticOperator._er_xidot` and
`_er_xdot`):

$$
F_\xi = \frac{\alpha\Delta}{4\hat B^3}\frac{d\hat\Phi}{d\hat\psi}\,\hat D\,
        \bigl(\hat B_\zeta\,\partial_\theta\hat B - \hat B_\theta\,\partial_\zeta\hat B\bigr),
\qquad
F_x = -\frac{\alpha\Delta}{4}\frac{d\hat\Phi}{d\hat\psi}\frac{\hat D}{\hat B^3}
      \bigl(\hat B_\theta\,\partial_\zeta\hat B - \hat B_\zeta\,\partial_\theta\hat B\bigr).
$$

$F_\xi$ carries the diagonal Legendre weight $\frac{L(L+1)}{(2L-1)(2L+3)}$ and $L\pm2$
couplings; $F_x$ acts through the dense speed operator $x\,\partial_x$. Both vanish at
$d\hat\Phi/d\hat\psi = 0$. Landreman et al. (2014) compare these trajectory models and show
that they differ most at low collisionality and finite $E_r$.

The native case executor builds the DKES form only: `use_dkes_exb=True`,
`with_er_xidot=False`, `with_er_xdot=False` (`_make_operator` in `src/dkx/execution.py`).

## Tangential magnetic drifts

`magneticDriftScheme` 1-9 adds the $\partial_\theta$, $\partial_\zeta$ magnetic-drift
advection (upwinded, controlled by `magneticDriftDerivativeScheme`) and a non-standard
$\partial_\xi$ term, coupling $L$ and $L\pm2$. They are transcriptions of the three
`select case (magneticDriftScheme)` blocks of the SFINCS v3 `populateMatrix.F90`:

| Scheme | Content |
| --- | --- |
| 1 | Reference form: $\partial_\theta$, $\partial_\zeta$ and $\partial_\xi$ terms |
| 2 | Scheme 1 plus `geometricFactor3` terms built from $\mathbf{B}\cdot\nabla\times\mathbf{B}$ |
| 3, 4 | Poloidal drift folded into $\partial_\zeta + \iota^{-1}\partial_\theta$; no $\partial_\xi$ term; 4 adds a shear piece and is limited to `geometryScheme` 11/12 |
| 5, 6 | Sugama forms from `gradpsidotgradB_overgpsipsi`; 5 regularized, 6 the $2\,\hat p'/\hat B$ pressure form |
| 7 | Scheme 1 without the $\partial_\xi$ term |
| 8 | Scheme 1 with the magnetic-shear correction in the $\partial_\zeta$ factor |
| 9 | Identical to scheme 1 (the Fortran shear term reads out-of-bounds indices and contributes zero in the reference build) |

Drift schemes need radial derivatives of $\hat B$, so they require a geometry that carries
them (`geometryScheme` 5, 11 or 12); `kinetic_operator_from_namelist` raises otherwise.
Because they couple $L\pm2$, the block-tridiagonal direct route does not apply and these
decks go to the Krylov route ({doc}`../numerics/solver_routes`). The native case executor
does not build magnetic drifts.

## Sources, constraints and nullspaces

The linearized collision operator conserves particles and energy (and momentum for the
Fokker-Planck operator), so the homogeneous operator has a nullspace: adding a perturbed
Maxwellian to $f_{s1}$ changes nothing. SFINCS closes the system by appending source
unknowns and constraint rows, and DKX uses the same bordered layout
`[f | Phi1 | lambda | sources]` (v3 `indices.F90`):

| `constraintScheme` | Sources | Constraints |
| --- | --- | --- |
| 0 | none | none; the system is rank-deficient |
| 1 (default for `collisionOperator` 0 and 3) | particle and energy source per species | $\langle\int d^3v\, f_{s1}\rangle = 0$ and $\langle\int d^3v\, v^2 f_{s1}\rangle = 0$ |
| 2 (default for `collisionOperator` 1) | one $L=0$ source per species and $x$ | $\langle f_{s1,L=0}\rangle = 0$ at each $x$ |
| 3, 4 | variants of the scheme 1 source shapes | as scheme 1 |

With `constraintScheme = 0` different solvers can return different nullspace components;
DKX treats density- and pressure-like diagnostics as gauge-dependent in that case
(`dkx.compare`). The native executor picks scheme 2 for pitch-angle scattering and scheme 1
for Fokker-Planck.

## Limits of the model

- One surface, fixed profiles: finite-orbit-width effects are absent. `dkx.validity`
  reports the orbit-width parameter so a user can see when the ordering is marginal
  ({doc}`reduced_models`).
- Linear in $f_{s1}$ except through $\Phi_1$ ({doc}`phi1_and_impurities`).
- No turbulence, no time dependence, no radial coupling between surfaces.
