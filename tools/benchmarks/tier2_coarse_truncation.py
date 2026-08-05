"""Can the tier-2 coarse preconditioner be truncated in Legendre index?

The tier-1 kernel keeps only the lowest ``K`` Legendre blocks and that is what
makes the 744k HSX case fit in 0.3 GB where the full band factorization wants
91 GB.  The obvious next move is to truncate the *coarse preconditioner* the
same way: factor the leading ``K`` blocks once, reuse those factors on every
Krylov application, and pay ``O(K m^2)`` storage instead of ``O(Nxi m^2)`` with
``m = Ntheta*Nzeta``.  A preconditioner is allowed to be approximate, so the
only question is what the approximation costs in GCROT iterations.

This script answers that question, and it answers it the same way whatever else
the machine is doing: **the iteration count is deterministic and independent of
load**, unlike any wall time.

Two studies, both against the same reference — the unmodified coarse
preconditioner of :func:`dkx.solve.build_coarse_preconditioner`:

``truncation``
    A ladder over ``keep``.  Blocks ``l >= keep`` are inverted *exactly*, block
    by block, so the ladder isolates the cost of severing the ``L+-1`` coupling
    at ``l = keep`` and nothing else.  That exact tail is deliberately not
    memory-lean: making the ladder as favourable as possible to truncation is
    the point, because the result is negative and a negative result has to be
    measured against the best case, not a convenient one.

``precision``
    The Schur LU factors — the one part of the stored state that is not
    reconstructible from a handful of coefficient arrays — in float32 instead of
    float64, through ``solvax.direct.block_thomas_factor``'s ``factor_dtype``.

Reproduce (from the repo root):

  python tools/benchmarks/tier2_coarse_truncation.py \\
      --examples /path/to/sfincs/fortran/version3/examples \\
      --deck geometryScheme4_2species_noEr

Wall times are reported for orientation only and are meaningless on a busy
machine; the iteration and residual columns are the measurement.
"""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp

import dkx.solve as solve_module
from dkx import require_float64
from dkx.drift_kinetic import KineticOperator
from dkx.namelist import read_sfincs_input
from solvax.direct import block_thomas_factor, block_thomas_solve
from solvax.krylov import gcrot
from solvax.operators import schur_projected_precond


def pinned_rows(op: KineticOperator):
    """The per-``(species, x)`` pinned row generators the coarse route builds.

    Same simplification and the same three regularizations as
    :func:`dkx.solve.build_coarse_preconditioner`: the self-species x-diagonal
    collision term, the ``1e-8`` invertibility floor, identity rows on the
    ``(x, l)`` pairs ``Nxi_for_x`` truncates, and the rank-one ``l = 0`` pin.
    """
    stripped = replace(
        op, fp=None, sugama=None, fp_phi1=None, with_er_xidot=False, with_er_xdot=False,
        with_magnetic_drifts=False, external_phi1_hat=None, include_phi1=False,
        include_phi1_in_kinetic=False,
    )
    mask = op._mask()
    collision = None
    for dense in (op.fp, op.sugama):
        if dense is not None:
            collision = solve_module._dense_collision_diagonal(dense.mat) * mask[None, :, :]
    if op.fp_phi1 is not None:
        collision = solve_module._collision_phi1_diagonal(op) * mask[None, :, :]
    stripped._check_block_extraction_supported()
    coef = solve_module._truncated_coefficients(stripped)
    c0 = op._fs_average_factor().reshape(-1)
    data = solve_module._coarse_generated_block_data(op, coef, mask, collision, c0, False)
    return solve_module._coarse_pinned_block_fns(coef, op.n_xi, *data, False)


def coarse_preconditioner(op: KineticOperator, *, keep: int, factor_dtype=None):
    """``(precond, precond_t)`` over the leading ``keep`` Legendre blocks.

    ``keep = op.n_xi`` with float64 factors reproduces the dense arm of
    :func:`dkx.solve.build_coarse_preconditioner` exactly.
    """
    n_s, n_x, n_xi, n_t, n_z = op.f_shape
    n_tz, batch = n_t * n_z, n_s * n_x
    rows = pinned_rows(op)
    head = [jax.lax.map(row, jnp.arange(keep, dtype=jnp.int32)) for row in rows]
    factors = jax.vmap(lambda lo, di, up: block_thomas_factor(lo, di, up, factor_dtype))(
        *(jnp.stack([blocks[i] for blocks in head]) for i in range(3))
    )
    tail = None
    if keep < n_xi:
        indices = jnp.arange(keep, n_xi, dtype=jnp.int32)
        tail = jax.vmap(jax.vmap(jax.scipy.linalg.lu_factor))(
            jnp.stack([jax.lax.map(lambda j, r=row: r(j)[1], indices) for row in rows])
        )

    def inverse(transpose: bool):
        trans = 1 if transpose else 0

        def apply(v):
            g = v.reshape(batch, n_xi, n_tz)
            solved = jax.vmap(lambda f, r: block_thomas_solve(f, r, transpose=transpose))(
                factors, g[:, :keep]
            )
            if tail is None:
                return solved.reshape(v.shape)
            rest = jax.vmap(
                jax.vmap(lambda lu, r: jax.scipy.linalg.lu_solve(lu, r, trans=trans))
            )(tail, g[:, keep:])
            return jnp.concatenate([solved, rest], axis=1).reshape(v.shape)

        # Compiled once and reused, so an application is the elimination rather
        # than a re-trace of the generated block rows.
        return jax.jit(apply)

    a_inv, a_inv_t = inverse(False), inverse(True)
    if op.include_phi1:
        b_cols, c_rows, d_block = solve_module._materialize_full_border(op)
        return (
            schur_projected_precond(a_inv, b_cols, c_rows, d_block=d_block),
            schur_projected_precond(a_inv_t, c_rows.T, b_cols.T, d_block=d_block.T),
        )
    if op.extra_size == 0:
        return a_inv, a_inv_t
    b_cols, c_rows = solve_module._materialize_borders(op)
    return (
        schur_projected_precond(a_inv, b_cols, c_rows),
        schur_projected_precond(a_inv_t, c_rows.T, b_cols.T),
    )


def measure(op, matvec, rhs, build, *, tol: float, max_restarts: int) -> dict:
    """Iterations, convergence and true relative residual for one preconditioner."""
    start = time.perf_counter()
    precond, _precond_t = build()
    jax.block_until_ready(precond(jnp.zeros((op.total_size,), dtype=jnp.float64)))
    build_s = time.perf_counter() - start
    start = time.perf_counter()
    solution = gcrot(
        matvec, rhs, precond=precond, m=30, k=8, rtol=tol, atol=0.0,
        max_restarts=max_restarts,
    )
    x = jax.block_until_ready(solution.x)
    solve_s = time.perf_counter() - start
    residual = jnp.linalg.norm(matvec(x) - rhs) / jnp.linalg.norm(rhs)
    return {
        "iterations": int(solution.iterations),
        "converged": bool(solution.converged),
        "relative_residual": float(residual),
        "build_s": round(build_s, 3),
        "solve_s": round(solve_s, 3),
    }


def head_factor_bytes(op: KineticOperator, keep: int, factor_dtype) -> float:
    """Bytes the leading ``keep`` blocks' reusable factors occupy.

    Three ``(Ntheta*Nzeta)`` blocks per ``(species, x, l < keep)``: the Schur LU
    factors plus the two off-diagonal bands ``block_thomas_solve`` sweeps with.
    The ladder's exact ``l >= keep`` tail is *not* counted, and is not small --
    it exists to make the comparison favourable to truncation, not to be a
    proposal.
    """
    n_s, n_x, _n_xi, n_t, n_z = op.f_shape
    width = 4.0 if factor_dtype is jnp.float32 else 8.0
    return n_s * n_x * keep * (n_t * n_z) ** 2 * (width + 16.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--deck", default="geometryScheme4_2species_noEr")
    parser.add_argument("--keeps", default="3,8,16,24,36,44,47")
    parser.add_argument("--study", default="both", choices=("truncation", "precision", "both"))
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--max-restarts", type=int, default=10)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    require_float64()

    op = KineticOperator.from_namelist(
        read_sfincs_input(args.examples / args.deck / "input.namelist")
    )
    n_s, n_x, n_xi, n_t, n_z = op.f_shape
    print(
        f"{args.deck}: Nxi={n_xi} Ntheta*Nzeta={n_t * n_z} subsystems={n_s * n_x} "
        f"unknowns={op.total_size}\n  full coarse bands "
        f"{solve_module.coarse_preconditioner_band_bytes(op) / 2**30:.2f} GB",
        flush=True,
    )
    rhs = op.rhs()
    matvec, _matvec_t = solve_module._pinned_matvecs(op)
    records = []

    def run(label: str, build, **extra):
        record = {"deck": args.deck, "n_xi": n_xi, "case": label, **extra}
        record.update(measure(op, matvec, rhs, build, tol=args.tol,
                              max_restarts=args.max_restarts))
        records.append(record)
        print(
            f"  {label:<28} iterations {record['iterations']:>5}  "
            f"converged {str(record['converged']):<5}  "
            f"relative residual {record['relative_residual']:.3e}  "
            f"[{record['build_s']:.1f} s build, {record['solve_s']:.1f} s solve]",
            flush=True,
        )

    run("reference (dkx.solve)", lambda: solve_module.build_coarse_preconditioner(op),
        keep=n_xi, factors="float64",
        head_factor_bytes=head_factor_bytes(op, n_xi, None))
    if args.study in ("truncation", "both"):
        for token in args.keeps.split(","):
            keep = min(int(token), n_xi)
            run(f"keep={keep}", lambda k=keep: coarse_preconditioner(op, keep=k),
                keep=keep, factors="float64",
                head_factor_bytes=head_factor_bytes(op, keep, None))
    if args.study in ("precision", "both"):
        # The whole chain from the local builder, at both precisions.  The
        # float64 row must reproduce the reference above; that is what makes
        # the float32 row a measurement of the precision and nothing else.
        for name, dtype in (("float64", None), ("float32", jnp.float32)):
            run(
                f"whole chain, {name} factors",
                lambda d=dtype: coarse_preconditioner(op, keep=n_xi, factor_dtype=d),
                keep=n_xi, factors=name,
                head_factor_bytes=head_factor_bytes(op, n_xi, dtype),
            )

    if args.out:
        args.out.write_text(json.dumps(records, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
