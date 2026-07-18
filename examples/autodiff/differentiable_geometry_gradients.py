"""Differentiate a simple geometry-based scalar with respect to harmonic amplitudes.

Uses the canonical differentiable geometry constructor
`FluxSurfaceGeometry.from_fourier` (the `geometryScheme=13` |B|-spectrum path used
by optimization loops) with the W7-X standard harmonics of `geometryScheme=4`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dkx.magnetic_geometry import FluxSurfaceGeometry  # noqa: E402

# W7-X standard (geometryScheme=4): (m, n) modes and flux functions.
_B0_OVER_BBAR = 3.089
_M = jnp.asarray([0.0, 0.0, 1.0, 1.0], dtype=jnp.float64)
_N = jnp.asarray([0.0, 1.0, 1.0, 0.0], dtype=jnp.float64)
_N_PERIODS = 5
_IOTA = 0.8700
_G_HAT = -17.885
_I_HAT = 0.0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-theta", type=int, default=31)
    p.add_argument("--n-zeta", type=int, default=31)
    args = p.parse_args()

    theta = jnp.linspace(0.0, 2 * jnp.pi, args.n_theta, endpoint=False, dtype=jnp.float64)
    zeta = jnp.linspace(0.0, 2 * jnp.pi / _N_PERIODS, args.n_zeta, endpoint=False, dtype=jnp.float64)

    def objective(harmonics_amp0: jnp.ndarray) -> jnp.ndarray:
        # bmnc holds the (0,0) mode (B0OverBBar) followed by the ripple harmonics.
        bmnc = jnp.concatenate([jnp.asarray([_B0_OVER_BBAR]), harmonics_amp0 * _B0_OVER_BBAR])
        geom = FluxSurfaceGeometry.from_fourier(
            theta=theta, zeta=zeta, bmnc=bmnc, m=_M, n=_N,
            n_periods=_N_PERIODS, iota=_IOTA, g_hat=_G_HAT, i_hat=_I_HAT,
        )  # fmt: skip
        # A made-up scalar objective: mean(BHat^2).
        return jnp.mean(geom.b_hat**2)

    amps0 = jnp.asarray([0.04645, -0.04351, -0.01902], dtype=jnp.float64)
    g = jax.grad(objective)(amps0)
    g_jit = jax.jit(jax.grad(objective))(amps0)

    print("amps0 =", amps0)
    print("grad  =", g)
    print("grad(jit) =", g_jit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
