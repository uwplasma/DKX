# Gradients of kinetic outputs

The drift-kinetic equation is linear in the distribution function. A solved
output $Q$ (a flux, a current, a transport coefficient) is therefore an
implicit function of the inputs $p$ through $K(p)\,f = b(p)$:

$$
\frac{dQ}{dp} = \frac{\partial Q}{\partial p} - \lambda^{\mathsf T}\left(\frac{\partial K}{\partial p} f - \frac{\partial b}{\partial p}\right),
\qquad K^{\mathsf T}\lambda = \frac{\partial Q}{\partial f}.
$$

One transposed solve gives the derivative with respect to every parameter at
once, and on a direct route that solve reuses the forward factorization. DKX
wraps its solve in this implicit differentiation, so `jax.grad` of a function
that calls `solve(..., differentiable=True)` returns this derivative. The
derivation and the routes that support it are in
{doc}`../numerics/differentiation`.

This tutorial follows `examples/07_gradients`.

## Differentiate a solved output

The example computes the pitch-angle-scattering bootstrap current
$\langle j\cdot B\rangle$ of a circular tokamak as a function of the species
temperature, with the collision operator rebuilt in JAX at each temperature:

```python
from dataclasses import replace
import jax, jax.numpy as jnp
import dkx
from dkx.run import profile_moments_from_operator
from dkx.solve import solve

operator = dkx.run(**CASE, **NUMERICS, emit=None).operator   # build once, outside JAX

@jax.jit
def bootstrap_current(t_hat):
    temperature = jnp.reshape(t_hat, (1,))
    pas = make_pitch_angle_scattering_v3_operator(...)          # collisions at this temperature
    perturbed = replace(operator, t_hat=temperature, pas=pas)
    solved = solve(perturbed, perturbed.rhs(), method="auto", differentiable=True)
    return profile_moments_from_operator(perturbed, solved.x)["FSABjHat"]

gradient = jax.grad(bootstrap_current)(t_hat_0)
```

The pattern is general: build the operator once outside any JAX
transformation, then write the parameter dependence as a pure function that
replaces fields of the operator pytree, solves with `differentiable=True`, and
reduces the state to a scalar. `examples/07_gradients/run.py` gives the full
script, including the arguments elided above.

## Check it

Every derivative example compares `jax.grad` with central differences. The
example uses three step sizes, $h$, $3h$ and $10h$, and asserts a relative
difference below 1e-5:

```console
python examples/07_gradients/run.py
```

A step sweep matters because a central difference has truncation error
$O(h^2)$ and round-off error $O(\epsilon/h)$; agreement across steps shows the
comparison is inside the window where both are small.

Three derivatives through three different paths, from
`tools/paper_benchmarks/gradient_verification.py`:

| Objective | Parameter | Relative AD–FD deviation |
| --- | --- | ---: |
| $D_{11}^*$ at $\nu' = 0.3$, $E^* = 0$ | Boozer amplitude $B_{10}$ | 4.3e-10 |
| `FSABjHat`, two-species RHSMode=1 solve | normalized collision frequency `nu_n` | 4.2e-9 |
| ambipolar $E_r$ (ion root), two-species PAS deck | scale on both species' density gradients | 7.9e-11 |

```{figure} ../_static/figures/paper/dkx_autodiff_gradient_check.png
:alt: Gradient parity against finite differences, finite-difference step sweep, primal and adjoint residuals, and solve counts against parameter count.
:width: 92%

Autodiff validation dashboard: gradient parity, the finite-difference step
sweep, primal and adjoint residuals, and solve counts
(`tools/publication_figures/generate_autodiff_sensitivity_validation.py`).
```

## What it costs

A central-difference gradient costs two solves per parameter; the adjoint costs
one transposed solve whatever the number of parameters.

```{figure} ../_static/figures/paper_benchmarks/gradient_cost_scaling.png
:alt: Gradient wall time against parameter count, adjoint against finite differences.
:width: 80%

Gradient wall time against parameter count (`tools/paper_benchmarks/gradient_cost_scaling.py`).
```

Compiled, a gradient costs 1.00–1.11× its primal on a 16,230-unknown deck
({doc}`../numerics/differentiation`). The first call also pays compilation of
the adjoint.

## Gradients over electric-field scans

For a function of several $E_r$ values on one surface, prepare the problem once
and scan it in one compiled call:

```python
import jax, jax.numpy as jnp, dkx

case = dkx.Case.from_file("examples/05_ambipolar_profile/case.toml")
problem = dkx.prepare_er_scan(case, surface_index=1)   # geometry, grids, collisions once

def bootstrap_current(er_kv_m):
    scan = dkx.batched_er_scan(problem, er_kv_m, differentiable=True, retain_full_state=True)
    return jnp.sum(scan.moments["FSABjHat"])

j, dj_der = jax.jit(jax.value_and_grad(bootstrap_current))(jnp.array([-0.2, 0.0, 0.2]))
```

With `prepare_er_scan(..., differentiable_profiles=True)`,
`problem.with_profiles(density_m3=..., temperature_keV=...)` refreshes the
profiles, drives and collisions inside JAX transformations, so derivatives with
respect to the plasma profiles follow the same pattern. The differentiable
ambipolar root is `dkx.ambipolar_er` ({doc}`ambipolar_er`).

## Limits

- The native `dkx.run(case)` host runner is not itself differentiated; the
  derivative path is the operator-level function above.
- `examples/07_gradients` differentiates pitch-angle scattering. Full
  Fokker–Planck temperature derivatives are not claimed by that example.
- On the recycled Krylov route the adjoint is an iterative solve, so its
  accuracy is bounded by the Krylov tolerance, and its cost is not one
  back-substitution ({doc}`../numerics/differentiation`).
- `examples/autodiff/` holds further patterns: a matrix-free residual and JVP,
  implicit differentiation through a Krylov solve, geometry gradients, and
  `gradients_tour.py`, which differentiates with respect to temperature and the
  $E_r$-like `dPhiHatdpsiHat` scalar (about 40 s on a laptop CPU).
