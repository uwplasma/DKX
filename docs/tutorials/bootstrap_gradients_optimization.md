# 4. Bootstrap current, gradients and optimization

The last part uses DKX the way stellarator design does: as a bootstrap-current
model checked against the Redl formula, as a function that can be
differentiated, and as a term inside an equilibrium optimization.

Scripts: `examples/tutorials/12_bootstrap_vs_redl.py`, `14_gradients.py`,
`15_optimization.py`, and `examples/advanced/` for the VMEX optimizations.

## The bootstrap current and the Redl formula

The bootstrap current is the parallel current driven by the radial pressure
gradient through trapped-particle friction, with no inductive field. DKX
computes it directly as the moment

$$
\langle j_\parallel B\rangle = \sum_s Z_s e\,\Bigl\langle B \int d^3v\; v_\parallel\, f_{s1}\Bigr\rangle ,
$$

reported as `parallel_current_A_T_m2` (native) or `FSABjHat` (SFINCS units).
Equilibrium codes usually take it instead from a fitted formula. The
Sauter–Redl form is

$$
\langle j_\parallel B\rangle_{\mathrm{Redl}}
= -\,I(\psi)\,p\left[\mathcal{L}_{31}\,\frac{\partial\ln p}{\partial\psi}
+ \mathcal{L}_{32}\,\frac{\partial\ln T_e}{\partial\psi}
+ \mathcal{L}_{34}\,\alpha\,\frac{1-R_{pe}}{R_{pe}}\,\frac{\partial\ln T_i}{\partial\psi}\right],
\qquad R_{pe} = \frac{p_e}{p},
$$

with coefficients fitted to tokamak drift-kinetic solutions as functions of
the trapped fraction, collisionality and $Z_{\mathrm{eff}}$ (Sauter et al.
1999; Redl et al. 2021). For a quasi-symmetric stellarator it is applied
through an equivalent tokamak, which Landreman, Buller & Drevlak (2022) found
accurate for quasi-axisymmetric and quasi-helical fields; for a
quasi-isodynamic field it is an extrapolation. DKX has no Redl implementation of its own;
`12_bootstrap_vs_redl.py` evaluates the formula beside DKX's kinetic current on
the same plasma and plots both, so the comparison isolates the model, not the
inputs.

What to expect: the kinetic current is the reference wherever the two
disagree, provided it is converged and uses a momentum-conserving operator.
Pitch-angle scattering lacks momentum restoration and
overestimates the current: in the VMEX QA optimization below it carries
1.5–1.6 times Redl's current. `pas+momentum_correction` closes most of that gap
at PAS cost ({doc}`../physics/reduced_models`). Converge `pitch` before
comparing: at low collisionality the bootstrap current needs far more Legendre
modes than the fluxes (`KineticBootstrapMismatch` defaults to 48).

## Gradients through the solve

The kinetic equation is linear in $f$: $K(p)\,f = b(p)$ for inputs $p$
(temperatures, gradients, geometry amplitudes). A solved output $Q(f, p)$ is
therefore an implicit function of $p$, and one transposed (adjoint) solve gives
its derivative with respect to every input at once:

$$
\frac{dQ}{dp} = \frac{\partial Q}{\partial p}
- \lambda^{\mathsf T}\left(\frac{\partial K}{\partial p}\,f - \frac{\partial b}{\partial p}\right),
\qquad K^{\mathsf T}\lambda = \frac{\partial Q}{\partial f}.
$$

On a direct route the adjoint reuses the forward factorization; compiled, a
gradient costs 1.00–1.11× its primal on a 16,230-unknown deck
({doc}`../numerics/differentiation`). `jax.grad` of a function that calls
`solve(..., differentiable=True)` returns exactly this derivative.

`14_gradients.py` differentiates the PAS bootstrap current of a circular
tokamak with respect to the species temperature, rebuilding the collision
operator in JAX at each temperature. The pattern: build the operator once
outside JAX, then write the parameter dependence as a pure function that
replaces fields of the operator pytree, solves and reduces to a scalar.

```python
operator = dkx.run(**CASE, **NUMERICS, emit=None).operator   # build once, outside JAX

@jax.jit
def bootstrap_current(t_hat):
    pas = make_pitch_angle_scattering_v3_operator(...)          # collisions at this temperature
    perturbed = replace(operator, t_hat=jnp.reshape(t_hat, (1,)), pas=pas)
    solved = solve(perturbed, perturbed.rhs(), method="auto", differentiable=True)
    return profile_moments_from_operator(perturbed, solved.x)["FSABjHat"]

gradient = jax.grad(bootstrap_current)(1.0)
```

```text
  <j.B> at THat = 1.0000      = +3.28188908e-02 (normalized)
  jax.grad           d<j.B>/dTHat = +2.1672520046e-02
  central difference d<j.B>/dTHat = +2.1672520011e-02
  relative difference             = 6.518e-09
  all gradients verified against central finite differences
```

Every derivative example compares against central differences at several step
sizes: a central difference has truncation error $O(h^2)$ and round-off
$O(\epsilon/h)$, and agreement across steps shows the comparison sits where both
are small.

```{figure} ../_static/figures/paper/dkx_autodiff_gradient_check.png
:alt: Gradient parity against finite differences, finite-difference step sweep, primal and adjoint residuals, and solve counts against parameter count.
:width: 92%

Gradient parity, step sweep, primal and adjoint residuals and solve counts
(`tools/publication_figures/generate_autodiff_sensitivity_validation.py`).
```

```{figure} ../_static/figures/paper_benchmarks/gradient_cost_scaling.png
:alt: Gradient wall time against parameter count, adjoint against finite differences.
:width: 80%

The adjoint costs one transposed solve whatever the number of parameters;
finite differences cost two solves per parameter
(`tools/paper_benchmarks/gradient_cost_scaling.py`).
```

**Other differentiable outputs.**

- *Over $E_r$ and profiles:* `dkx.prepare_er_scan` and
  `dkx.batched_er_scan(..., differentiable=True)` give
  `jax.value_and_grad` over a set of fields in one compiled call; with
  `differentiable_profiles=True`, `problem.with_profiles(...)` refreshes drives
  and collisions inside JAX, so derivatives with respect to $n$ and $T$ follow.
- *The ambipolar root:* `dkx.ambipolar_er` is differentiable through the
  implicit-function theorem, $dE_r/dp = -(\partial J_r/\partial p)/(\partial J_r/\partial E_r)$.
- *$\Phi_1$:* `dkx.phi1.phi1_solution` returns the converged nonlinear state
  with a matrix-free implicit adjoint: GMRES on the transposed Jacobian of the
  coupled residual, on the Legendre-truncated subspace. On a native
  `phi1 = "kinetic"` case its gradient matches central differences to 1.2e-9
  and the Taylor remainder falls as $h^2$.

The host runner `dkx.run(case)` itself is not differentiated; the derivative
path is the operator-level function above. On the recycled Krylov route the
adjoint is an iterative solve whose accuracy is bounded by the Krylov
tolerance.

## A shape derivative and a small optimization

`15_optimization.py` differentiates the kinetic solve with respect to the
geometry. The design variable $\epsilon_h$ is the helical $|B|$ harmonic of a
three-harmonic surface with $N = 5$ field periods. The traced chain is
$\epsilon_h \to$ Boozer spectrum $\to$ `FluxSurfaceGeometry.from_fourier`
$\to K(\epsilon_h) \to f \to \Gamma$, and gradient descent lowers the particle
flux:

```text
  step 0: epsilon_h = 0.050000  Gamma = 1.505231e-05  dGamma/d(epsilon_h) = +6.048958e-04
  step 1: epsilon_h = 0.043951  Gamma = 1.169691e-05  dGamma/d(epsilon_h) = +5.048168e-04
  ...
  step 5: epsilon_h = 0.027852  Gamma = 5.507189e-06  dGamma/d(epsilon_h) = +2.732785e-04
  reduction: 63.4% in 5 gradient steps
  jax.grad           dGamma/d(epsilon_h) = +6.0489577987e-04
  central difference dGamma/d(epsilon_h) = +6.0489577985e-04
  physics check: removing helical ripple lowered the neoclassical flux
```

Removing helical ripple reduces the $1/\nu$ transport, which is what quasi-
symmetric design does. Only geometry leaves of the operator are replaced, so
this is a pure shape derivative; the script does not import `vmex`.

## Inside a VMEX optimization

In a design loop the $|B|$ spectrum comes from an equilibrium: boundary →
VMEX → `booz_xform_jax` → DKX → scalar. `dkx.bootstrap.KineticBootstrapMismatch`
is the mismatch between the equilibrium's $\langle j\cdot B\rangle$ and DKX's
kinetic one, traced through that whole chain so VMEX's implicit Jacobian
carries it. Adding it to a VMEX optimization is one import and one tuple:

```python
from dkx.bootstrap import KineticBootstrapMismatch

kinetic = KineticBootstrapMismatch(profiles, surfaces=[0.25, 0.5, 0.75])
objective_function_terms.append((kinetic, 0.0, 1.0))
```

`profiles` is the `vmex.core.bootstrap.KineticProfiles` the Redl term takes, so
both models describe one plasma. `collision_model="pas+momentum_correction"`
replaces PAS with the momentum-corrected solve. It needs the optional `vmex`
and `booz_xform_jax >= 0.4`.

```{figure} ../_static/figures/readme/QA_optimization_bootstrap_dkx.png
:alt: Bootstrap current profiles and objective history of the VMEX QA optimization with a DKX kinetic row.
:width: 85%

`examples/advanced/QA_optimization_bootstrap_dkx.py`. At the committed
parameters (three kinetic surfaces, $11\times11\times16\times4$), one run on four
laptop CPU threads took 30 minutes and 4.6 GB. The objective fell from 1.78 to
0.0061 and the DKX mismatch from 1.2e-3 to 1.2e-4; finite differences agree with
the traced Jacobian of the row to 5.7e-5–1.8e-3. The kinetic grid is small and
the operator is PAS (1.5–1.6× Redl's current), so the current is not a
converged value.
```

`QH_optimization_bootstrap_dkx.py` (nfp = 4) and
`QI_optimization_bootstrap_dkx.py` (nfp = 2, where Redl is an extrapolation)
use the host-side `KineticBootstrapCurrent` on the written `wout`, under a
finite-difference Jacobian; with `ambipolar=True` it refines the ion root by
Brent's method and reads $\langle j\cdot B\rangle$ at the refined root.

**`vmex --neoclassical`.** VMEX's own command line runs a DKX neoclassical
survey of a finished equilibrium by calling
`dkx.representative.run_representative`: on each surface it scans $E_r$, takes
the most negative stable ambipolar root, refines it by Brent's method, and
reads fluxes and the bootstrap current there, with the plasma's own profiles
(`ne_coeffs`, `Te_coeffs`, `Ti_coeffs`), a choice of `fp`, `pas` or
`pas+momentum_correction`, and an optional external Redl curve drawn beside
DKX's. Without VMEX, `dkx wout_XXX.nc` runs the same survey from a `wout` with
assumed profiles ({doc}`../user_guide/cli`). It is a first look; for results,
write a case with the real profiles, then `dkx validate` and `dkx converge` it.

## Where to go from here

- Every case field: {doc}`../user_guide/inputs`; every output: {doc}`../user_guide/outputs`.
- How the solver chooses and why: {doc}`../numerics/solver_routes`.
- What has been checked, and against what: {doc}`../benchmarks/index`.
