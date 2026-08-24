# 3. Gradients — differentiating through the solve

The solve is differentiable, so `jax.grad` of any output with respect to any
input costs one transposed solve regardless of how many parameters there are.
These examples put that inside an optimization loop.

Some scripts here need the optional companions `vmex` and `booz_xform_jax`;
each says so at the top and skips cleanly when they are absent.
