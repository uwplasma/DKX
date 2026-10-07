"""Micro-benchmark: block Thomas (SOLVAX, serial in the block index) against block
cyclic reduction (log2 depth, batched LUs) on DKX-shaped block-tridiagonal chains.

    python tools/benchmarks/block_cr_vs_thomas.py --out cr.json
    JAX_PLATFORMS=cuda python tools/benchmarks/block_cr_vs_thomas.py --out cr_gpu.json

Shapes (L blocks, m = Ntheta*Nzeta, B independent chains = species x speeds):
the bootstrap row (48, 169, 10), monoenergetic W7-X (64, 779, 1), the mid HSX
deck (60, 561, 5) and HSX production (100, 1275, 5). Matrices are random,
block diagonally dominant. Reports warm factor+solve time, achieved GFLOP/s of
the Thomas count (L * 14/3 m^3 per chain) and the relative residual.
"""

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp
from jax.scipy.linalg import lu_factor, lu_solve

jax.config.update("jax_enable_x64", True)

SHAPES = {"bootstrap_row": (48, 169, 10), "mono_w7x": (64, 779, 1), "hsx_mid": (60, 561, 5), "hsx_prod": (100, 1275, 5)}


def make(key, n, m, dtype):
    k1, k2, k3 = jax.random.split(key, 3)
    lo = jax.random.normal(k1, (n, m, m), dtype) / m**0.5
    up = jax.random.normal(k2, (n, m, m), dtype) / m**0.5
    d = jax.random.normal(k3, (n, m, m), dtype) / m**0.5 + 4.0 * jnp.eye(m, dtype=dtype)
    return lo.at[0].set(0.0), d, up.at[-1].set(0.0)


def thomas(lo, d, up, b):
    from solvax.direct import block_thomas_factor, block_thomas_solve

    return block_thomas_solve(block_thomas_factor(lo, d, up), b)


def cyclic_reduction(lo, d, up, b):
    """Block cyclic reduction, one right-hand side ``b`` of shape ``(n, m)``.

    Each level factors every odd row's diagonal block at once (one batched LU),
    folds them into the even rows and recurses on half the rows. Padded to a
    power of two with identity rows.
    """
    n, m = b.shape
    p = 1 << (n - 1).bit_length()
    eye = jnp.eye(m, dtype=d.dtype)
    if p > n:
        z = jnp.zeros((p - n, m, m), d.dtype)
        lo, up = jnp.concatenate([lo, z]), jnp.concatenate([up, z])
        d = jnp.concatenate([d, jnp.broadcast_to(eye, (p - n, m, m))])
        b = jnp.concatenate([b, jnp.zeros((p - n, m), b.dtype)])
    return _cr(lo, d, up, b)[:n]


def _cr(lo, d, up, b):
    n = d.shape[0]
    if n == 1:
        return lu_solve(lu_factor(d[0]), b[0])[None]
    de, do = d[0::2], d[1::2]
    le, lo_ = lo[0::2], lo[1::2]
    ue, uo = up[0::2], up[1::2]
    be, bo = b[0::2], b[1::2]
    f = lu_factor(do)  # batched over the n/2 odd rows
    # D_odd^-1 [L_odd, U_odd, b_odd] in one batched solve
    rhs = jnp.concatenate([lo_, uo, bo[..., None]], axis=-1)
    w = jax.vmap(lu_solve)(f, rhs)
    m = d.shape[-1]
    wl, wu, wb = w[..., :m], w[..., m : 2 * m], w[..., 2 * m]
    # even row i couples to odd i-1 (via le) and odd i (via ue)
    sh = lambda a: jnp.concatenate([jnp.zeros_like(a[:1]), a[:-1]])  # noqa: E731  odd i-1
    nd = de - ue @ wl - le @ sh(wu)
    nl = -(le @ sh(wl))
    nu = -(ue @ wu)
    nb = be - jnp.einsum("kab,kb->ka", ue, wb) - jnp.einsum("kab,kb->ka", le, sh(wb))
    xe = _cr(nl, nd, nu, nb)
    # odd row i: x_i = wb_i - wl_i x_(i-1, even) - wu_i x_(i+1, even)
    xn = jnp.concatenate([xe[1:], jnp.zeros_like(xe[:1])])
    xo = wb - jnp.einsum("kab,kb->ka", wl, xe) - jnp.einsum("kab,kb->ka", wu, xn)
    return jnp.stack([xe, xo], 1).reshape(n, -1)


def resid(lo, d, up, x, b):
    from solvax.direct import block_tridiag_matvec

    r = block_tridiag_matvec(lo, d, up, x) - b
    return float(jnp.linalg.norm(r) / jnp.linalg.norm(b))


def timeit(f, *a, rep=3):
    jax.block_until_ready(f(*a))
    ts = []
    for _ in range(rep):
        t = time.perf_counter()
        jax.block_until_ready(f(*a))
        ts.append(time.perf_counter() - t)
    return min(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--dtypes", default="float64,float32")
    a = ap.parse_args()
    rows = []
    for name in a.shapes.split(","):
        n, m, bsz = SHAPES[name]
        for dt in a.dtypes.split(","):
            dtype = jnp.dtype(dt)
            lo, d, up = jax.vmap(lambda k: make(k, n, m, dtype))(jax.random.split(jax.random.PRNGKey(0), bsz))
            b = jax.random.normal(jax.random.PRNGKey(1), (bsz, n, m), dtype)
            rec = dict(shape=name, L=n, m=m, B=bsz, dtype=dt, backend=jax.default_backend())
            for label, fn in (("thomas", thomas), ("cr", cyclic_reduction)):
                f = jax.jit(jax.vmap(fn))
                try:
                    t = timeit(f, lo, d, up, b)
                    x = f(lo, d, up, b)
                    rec[label + "_s"] = t
                    rec[label + "_resid"] = resid(lo[0], d[0], up[0], x[0], b[0])
                except Exception as e:  # noqa: BLE001  out of memory on the big shapes
                    rec[label + "_s"] = None
                    rec[label + "_err"] = str(e)[:120]
            flops = bsz * n * 14 / 3 * m**3
            rec["thomas_gflops"] = flops / rec["thomas_s"] / 1e9 if rec.get("thomas_s") else None
            print(json.dumps(rec), flush=True)
            rows.append(rec)
            del lo, d, up, b
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
