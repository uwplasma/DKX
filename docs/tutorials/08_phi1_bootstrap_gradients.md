# 8. $\Phi_1$, the bootstrap current and gradients

Three further uses of the same solve, each briefly: the in-surface potential,
a check of the bootstrap current against the Redl formula, and derivatives for
optimization.

## The in-surface potential

The potential is not exactly a flux function. Its variation $\Phi_1(\theta,\zeta)$
is small, $e\Phi_1/T\sim\Delta$, but it adds a parallel electric field, a
tangential $E\times B$ drift and a Boltzmann factor to every density. It is
fixed by quasineutrality on the surface,

$$
\sum_s Z_s\Big[n_{s0}\,e^{-Z_s e\Phi_1/T_s} + \int d^3v\, f_{s1}\Big] = 0 ,
$$

which is nonlinear in $\Phi_1$; DKX solves the coupled system by
Newton-Krylov ({doc}`../physics/phi1_and_impurities`). It matters most for
high-$Z$ impurities. Natively it is `[physics] phi1 = "kinetic"` on a profile
case. `11_phi1.py` passes SFINCS keywords to `dkx.run` (hydrogen, 1.25% carbon,
electrons, Fokker-Planck, `includePhi1` off then on; 31 s):

```python
runs[phi1] = dkx.run(**PARAMETERS, includePhi1=phi1, includePhi1InKineticEquation=phi1)
```

```text
  Phi1 off: solver route block_tridiagonal, converged True
  Phi1 on : solver route phi1_newton_krylov, converged True
  hydrogen Z=+1   Gamma off=-1.0008e-06  on=-1.6454e-06  change=+64.4%  peak |dn/n|=2.59e-02
  carbon Z=+6     Gamma off=+1.9442e-07  on=+1.6222e-07  change=-16.6%  peak |dn/n|=6.42e-03
  electron Z=-1   Gamma off=+1.7719e-07  on=-6.6438e-07  change=-475.0%  peak |dn/n|=9.83e-04
```

```{figure} ../_static/figures/tutorials/phi1.png
:alt: Density perturbation on the flux surface for each species and the change in flux from Phi1.
:width: 70%

Density variation on the surface and the flux change.
```

The route changes from one linear solve to Newton-Krylov. On this coarse grid
$\Phi_1$ changes the hydrogen flux by 64% and reverses the small electron flux,
while the density varies on the surface by at most 2.6%. Status: $\Phi_1$ is
`validated_limited`: one geometry, FP collisions and zero $E_r$ match SFINCS v3;
other options are refused natively and go through a namelist.

## Bootstrap current against Redl

The kinetic bootstrap current is the moment $\langle j_\parallel B\rangle$ of
page 2. Equilibrium codes often use the fitted Sauter-Redl formula instead,

$$
\langle j_\parallel B\rangle_{\mathrm{Redl}} = -I(\psi)\,p\Big[\mathcal L_{31}\frac{\partial\ln p}{\partial\psi}
+ \mathcal L_{32}\frac{\partial\ln T_e}{\partial\psi}
+ \mathcal L_{34}\,\alpha\,\frac{1-R_{pe}}{R_{pe}}\frac{\partial\ln T_i}{\partial\psi}\Big],
$$

with coefficients fitted to tokamak solutions as functions of trapped fraction,
collisionality and $Z_\mathrm{eff}$. For a quasi-symmetric stellarator it is
applied through an equivalent tokamak; for other fields it is an extrapolation.
`12_bootstrap_vs_redl.py` evaluates both on one surface of the two-field-period
VMEC seed of page 3, $T_e=T_i$ (15 s):

```text
  Redl fit               <j.B> = +1.5465e+06 A T m^-2
  DKX Fokker-Planck          <j.B> = +9.9108e+05 A T m^-2  ratio to Redl = 0.641
  DKX pitch-angle scattering <j.B> = +1.4226e+06 A T m^-2  ratio to Redl = 0.920
```

```{figure} ../_static/figures/tutorials/bootstrap.png
:alt: Bootstrap current from the Redl fit, DKX Fokker-Planck and DKX pitch-angle scattering.
:width: 50%

Kinetic against fitted current on the same plasma.
```

The seed is not quasi-symmetric, so Redl is an extrapolation and the gap is not
a DKX error. PAS sits closer to Redl only because it overestimates the current
(page 4). The FP value is the reference, at this grid still resolution-limited.

## Gradients and optimization

The kinetic equation is linear, $K(p)f=b(p)$, so one transposed solve gives
the derivative of an output $Q$ with respect to every input:

$$
\frac{dQ}{dp} = \frac{\partial Q}{\partial p}
- \lambda^{\mathsf T}\Big(\frac{\partial K}{\partial p}f - \frac{\partial b}{\partial p}\Big),
\qquad K^{\mathsf T}\lambda = \frac{\partial Q}{\partial f}.
$$

`jax.grad` of a function calling `solve(..., differentiable=True)` returns it.
`14_gradients.py` differentiates the PAS bootstrap current with respect to the
temperature and compares to central differences (9 s):

```python
gradient = jax.grad(bootstrap_current)(1.0)   # operator built once outside JAX; see the script
```

```text
  <j.B> at THat=1.000: +3.67978755e-02
  jax.grad            d<j.B>/dTHat = +3.2840365846e-02
  central differences d<j.B>/dTHat = [0.0328403659 0.0328403658 0.0328403659]
  max relative difference          = 2.55e-09
```

```{figure} ../_static/figures/tutorials/gradients.png
:alt: Gradient from jax.grad compared with central differences over a temperature sweep.
:width: 70%

`jax.grad` against central differences.
```

`15_optimization.py` differentiates with respect to the geometry: the helical
$|B|$ harmonic $\epsilon_h$ of a three-harmonic surface enters through
`FluxSurfaceGeometry.from_fourier`, and eight gradient steps lower the particle
flux (13 s):

```text
  eps_h: 0.05000 -> 0.02136   Gamma: 1.65121e-05 -> 5.55778e-06  (66.3% lower)
  first gradient: jax.grad +6.37301608e-04  central difference +6.37301608e-04  relative difference 8.4e-11
```

```{figure} ../_static/figures/tutorials/optimization.png
:alt: Particle flux and helical harmonic along the gradient-descent steps.
:width: 70%

Gradient descent on $\epsilon_h$.
```

Removing helical ripple lowers the $1/\nu$ flux, which is what quasi-symmetric
design does. The same chain runs inside a VMEX optimization through
`dkx.bootstrap.KineticBootstrapMismatch`, and `vmex --neoclassical` surveys a
finished equilibrium; both are in `examples/advanced/` and
{doc}`../numerics/differentiation`. The host runner `dkx.run(case)` itself is
not differentiated; the derivative path is the operator-level function.

```{figure} ../_static/figures/finite_beta_vmex_sfincs_bootstrap_er.png
:alt: Ambipolar radial electric field and bootstrap current profiles for the finite-beta QA equilibrium.
:width: 70%

`examples/advanced/finite_beta_vmec_to_sfincs.py`: ambipolar $E_r$ and bootstrap
current on a finite-beta QA equilibrium.
```

## Where to go from here

- Every case field: {doc}`../user_guide/inputs`; every output: {doc}`../user_guide/outputs`.
- How the solver chooses: {doc}`../numerics/solver_routes`.
- What has been checked, and against what: {doc}`../benchmarks/index`.
