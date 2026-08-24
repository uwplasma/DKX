# 3. Gradients — differentiating through the solve

The solve is differentiable, so `jax.grad` of any output with respect to any
input costs one transposed solve regardless of how many parameters there are —
against 2N solves for finite differences, with a step size to choose.

| script | what it shows |
|---|---|
| `gradient_of_bootstrap.py` | `d<j.B>/dTHat` by autodiff, checked against central differences |

The pattern is the same every time:

```python
operator = dkx.run(**case).operator          # build it once
perturbed = replace(operator, t_hat=...)     # swap the leaf you differentiate
solved = solve(perturbed, perturbed.rhs(), method="auto", differentiable=True)
```

`KineticOperator` is a registered JAX pytree, so replacing one leaf traces
everything downstream: drive, matrix apply, solve, moment integrals.

Some scripts here need the optional companions `vmex` and `booz_xform_jax`;
each says so at the top and skips cleanly when they are absent.
