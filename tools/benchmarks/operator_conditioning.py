#!/usr/bin/env python
"""Condition number of the kinetic operator, raw and equilibrated.

Why this exists: three unrelated-looking observations turned out to be one
fact, and this script is how that was established.

  * the Boozer round-trip's FSABjHat moved 2.7e-4 across solver routes while
    every route reported a 1e-15 residual;
  * warm-starting one surface from another moved a flux by 2.46% while
    ``primal_residual < 1e-10`` passed;
  * FSABjHat changed sign as Nxi was refined, with every solve converged.

All three follow from ``||dx||/||x|| <= cond(A) * ||r||/||b||``. Measured here
by dense SVD, which bounds the decks to a few thousand unknowns -- that is the
point: these are the *small* decks, and they already reach 1e18.

Recorded on 2026-09-02 (see plan.md):

    deck                     n     cond(raw)   cond(equil)   gain
    pas_scheme12           303      3.00e12       6.58e9     456x
    fp_cs3                 402      2.96e07       1.58e5     187x
    er_xdot                604      3.36e18       1.24e18      3x
    magdrift               604      6.43e07       2.46e6      26x

The ``er_xdot`` row is the one that matters. At cond ~ 3e18 the system is
numerically singular in float64 (``cond * eps`` ~ 750), equilibration barely
moves it, and the term responsible -- the ``Er`` ``xDot`` coupling -- is
present in exactly the runs where the anomalies above appeared.

Run:
  python tools/benchmarks/operator_conditioning.py DECK.input.namelist [...]
"""

from __future__ import annotations

import sys

import numpy as np

from dkx.run import run_profile
from dkx.solve import materialize_csr


def operator_for(namelist: str):
    """Build the operator a deck solves, without keeping the solve.

    ``run_profile`` owns operator construction and there is no public seam that
    returns the operator alone, so this wraps ``dkx.solve.solve`` to capture the
    first operator handed to it. A deck that fails to converge still yields its
    operator, which is the interesting case here.
    """
    import dkx.run as run_mod
    import dkx.solve as solve_mod

    captured: dict = {}
    original = solve_mod.solve

    def spy(op, rhs, **kwargs):
        captured.setdefault("op", op)
        return original(op, rhs, **kwargs)

    solve_mod.solve = spy
    run_mod.solve = spy
    try:
        run_profile(namelist, emit=None, tol=1e-9)
    except Exception as exc:  # noqa: BLE001 - the operator is what we came for
        # A deck that stalls is exactly the interesting case here, so the
        # failure is reported and discarded rather than raised.
        print(f"  (solve raised {type(exc).__name__}; using the operator anyway)")
    finally:
        solve_mod.solve = original
        run_mod.solve = original
    return captured.get("op")


def ruiz(matrix: np.ndarray, iterations: int = 20) -> np.ndarray:
    """Ruiz row/column equilibration: scale both to unit max magnitude."""
    scaled = matrix.copy()
    for _ in range(iterations):
        rows = np.sqrt(np.maximum(np.abs(scaled).max(axis=1), 1e-300))
        cols = np.sqrt(np.maximum(np.abs(scaled).max(axis=0), 1e-300))
        scaled = scaled / rows[:, None] / cols[None, :]
    return scaled


def main(decks: list[str], max_size: int = 4000) -> None:
    print(
        f"{'deck':52s} {'n':>6s} {'cond(raw)':>11s} {'cond(equil)':>12s} {'gain':>7s}"
    )
    for deck in decks:
        name = deck.split("/")[-1][:52]
        op = operator_for(deck)
        if op is None:
            print(f"{name:52s} {'-':>6s} {'no operator':>11s}")
            continue
        if op.total_size > max_size:
            print(f"{name:52s} {op.total_size:6d} {'too large':>11s}")
            continue
        dense = materialize_csr(op, pin_masked_dofs=True).toarray()
        raw = np.linalg.svd(dense, compute_uv=False)
        equil = np.linalg.svd(ruiz(dense), compute_uv=False)
        c_raw = raw[0] / raw[-1]
        c_eq = equil[0] / equil[-1]
        print(
            f"{name:52s} {op.total_size:6d} {c_raw:11.2e} {c_eq:12.2e} {c_raw / c_eq:6.0f}x",
            flush=True,
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
