"""Eager against compiled primal and gradient of a DKX observable, interleaved.

The harness behind ``docs/experiments/2026-09-22-derivative-cost.md``. The
objective is ``FSABjHat`` of a pitch-angle-scattering tokamak-like deck, as a
function of multiplicative scalings of ``THat``, ``nHat`` and ``dTHat/dpsiHat``
threaded through the operator. Every arm is called once to compile, then all
arms run in interleaved rounds so load drift hits them alike; the medians and
minima of wall and process CPU time are printed, with the ratios the record
admits on.

Usage::

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=1 \\
        python tools/benchmarks/derivative_cost.py --resolution 13 13 16 6 --repeats 15

Pin the process to its cores (``taskset``) and record the host load; the
ratios are only meaningful on a quiet host.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

import dkx
from dkx.run import profile_moments_from_operator
from dkx.solve import solve

DECK = dict(
    geometryScheme=1, inputRadialCoordinate=3, rN_wish=0.3,
    B0OverBBar=1.0, epsilon_t=-0.07, epsilon_h=0.02, iota=0.4542,
    GHat=3.7481, IHat=0.0, psiAHat=0.15596, aHat=0.5585,
    Zs=[1.0], mHats=[1.0], nHats=[1.0], THats=[1.0],
    dNHatdrHats=[-0.5], dTHatdrHats=[-1.0],
    Delta=4.5694e-3, alpha=1.0, nu_n=0.01, Nxi_for_x_option=0, RHSMode=1,
    collisionOperator=1,
)  # fmt: skip


def objective(op0, *, differentiable: bool):
    """``FSABjHat`` as a function of the three profile scalings."""

    def loss(p):
        op = replace(
            op0, t_hat=p[0] * op0.t_hat, n_hat=p[1] * op0.n_hat,
            dt_hat_dpsi_hat=p[2] * op0.dt_hat_dpsi_hat,
        )  # fmt: skip
        x = solve(
            op, op.rhs(), method="block_tridiagonal", tol=1e-10,
            differentiable=differentiable, emit=None,
        ).x  # fmt: skip
        return profile_moments_from_operator(op, jnp.reshape(x, (-1,)))["FSABjHat"]

    return loss


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--resolution", type=int, nargs=4, default=(13, 13, 16, 6),
        metavar=("NTHETA", "NZETA", "NXI", "NX"),
    )  # fmt: skip
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args(argv)

    n_theta, n_zeta, n_xi, n_x = args.resolution
    op0 = dkx.run(**DECK, Ntheta=n_theta, Nzeta=n_zeta, Nxi=n_xi, NL=4, Nx=n_x, emit=None).operator
    loss = objective(op0, differentiable=True)
    arms = {
        "eager primal": loss,
        "eager grad": jax.grad(loss),
        "jit primal": jax.jit(loss),
        "jit grad": jax.jit(jax.grad(loss)),
        "jit primal, differentiable=False": jax.jit(objective(op0, differentiable=False)),
    }
    p = jnp.ones(3)
    print(f"unknowns {op0.total_size}, load {os.getloadavg()}")
    values = {}
    for name, fn in arms.items():
        start = time.perf_counter()
        values[name] = np.asarray(jax.block_until_ready(fn(p)))
        print(f"first call  {name:34s} {time.perf_counter() - start:8.3f} s")

    wall = {name: [] for name in arms}
    cpu = {name: [] for name in arms}
    for _ in range(args.repeats):
        for name, fn in arms.items():
            c0, t0 = time.process_time(), time.perf_counter()
            jax.block_until_ready(fn(p))
            wall[name].append(time.perf_counter() - t0)
            cpu[name].append(time.process_time() - c0)
    for name in arms:
        print(
            f"{name:46s} wall {np.median(wall[name]):.4f} (min {np.min(wall[name]):.4f})"
            f"  cpu {np.median(cpu[name]):.4f} (min {np.min(cpu[name]):.4f})"
        )
    for a, b in (("eager primal", "jit primal"), ("eager grad", "jit grad"), ("jit grad", "jit primal")):
        print(
            f"{a} / {b}: cpu {np.median(cpu[a]) / np.median(cpu[b]):.2f} (median),"
            f" wall {np.median(wall[a]) / np.median(wall[b]):.2f} (median)"
        )
    spread = np.max(np.abs(values["eager grad"] - values["jit grad"]))
    print(f"primal {float(values['jit primal']):.15g}; eager and jit gradients differ by {spread:.1e}")
    print(f"load at end {os.getloadavg()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
