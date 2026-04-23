from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from sfincs_jax.geometry import boozer_geometry_scheme4


def test_scheme4_geometry_scalar_gradient_matches_finite_difference() -> None:
    """Check a cheap geometry-only autodiff gate against central differences.

    This is intentionally smaller than a full transport solve.  It validates that
    the public scheme-4 harmonic hook stays differentiable through the normalized
    magnetic-field and Jacobian-like arrays that later feed the kinetic operator.
    """

    theta = jnp.linspace(0.0, 2.0 * jnp.pi, 10, endpoint=False)
    zeta = jnp.linspace(0.0, 2.0 * jnp.pi / 5.0, 8, endpoint=False)
    amp0 = jnp.asarray([0.08, -0.03, 0.02], dtype=jnp.float64)

    def objective(a: jnp.ndarray) -> jnp.ndarray:
        geom = boozer_geometry_scheme4(theta=theta, zeta=zeta, harmonics_amp0=a)
        return jnp.mean(geom.b_hat**2) + 0.1 * jnp.mean(geom.d_hat)

    grad_jax = np.asarray(jax.grad(objective)(amp0))

    eps = 1.0e-6
    grad_fd = []
    for i in range(int(amp0.size)):
        direction = jnp.zeros_like(amp0).at[i].set(1.0)
        plus = objective(amp0 + eps * direction)
        minus = objective(amp0 - eps * direction)
        grad_fd.append(float((plus - minus) / (2.0 * eps)))

    np.testing.assert_allclose(grad_jax, np.asarray(grad_fd), rtol=2e-6, atol=2e-8)
