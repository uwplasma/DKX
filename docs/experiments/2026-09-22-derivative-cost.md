# Derivative cost: where the 2x of a gradient goes

**Status: step 7 implemented (#279).** The profile below found the excess to be
eager dispatch; the structured route now compiles itself, and the eager primal
and gradient are within 1.2 to 1.4 of their compiled times on a loaded host.
Sections marked *open* are not measured yet.

## Hypothesis

`2026-09-20-one-factorization-many-solves.md` measured `jax.grad` through DKX at
2.0 to 2.6 times its primal on every differentiable route, and showed the
factorization is already shared: one factorization per gradient, adjoint on the
primal's `Tier1Solver`. It attributed the remainder to the reverse pass over
the operator's coefficient construction. The adjoint pattern (Paul, Abel,
Landreman and Dorland, JPP 2019) says a gradient of a scalar should cost one
transposed solve with the primal's factors plus one vector-Jacobian product of
the residual `A(p) x - b(p)` at the converged `x`, i.e. 1.0 to 1.3 primals.
Before implementing that, measure what the extra cost actually is.

## Admission test

- `jax.grad` / primal at most 1.3 on the structured direct deck of the
  2026-09-20 record, back to back in one process, load recorded.
- Finite-difference or Taylor agreement of the gradient to about `1e-6`
  relative on at least three parameters.
- No change to any primal answer beyond round-off; focused solve and adjoint
  tests pass.

## Result (partial)

Deck: analytic `geometryScheme=1`, pitch-angle scattering, RHSMode 1,
`(Ntheta, Nzeta, Nxi, Nx) = (13, 13, 16, 6)`, 16,230 unknowns, tolerance
`1e-10`, route `block_tridiagonal`. Objective: `FSABjHat` from
`profile_moments_from_operator`. Parameters: multiplicative scalings of `THat`,
`nHat` and `dTHat/dpsiHat` threaded through `dataclasses.replace` of the
operator. Host: office workstation (Xeon W-2295), process pinned to four cores
with one XLA/BLAS thread, one-minute load 7 to 10 on 36 hardware threads.
JAX/jaxlib 0.9.2, SOLVAX 0.25.0, float64. Nine repeats after a warm-up, median
and minimum seconds. Harness: `tools/benchmarks/derivative_cost.py` (the
component and operation-count profiles below were one-off scripts and are not
kept).

### The gradient is not 2x once it is compiled

| Arm | eager median | eager min | `jax.jit` median | `jax.jit` min |
| --- | ---: | ---: | ---: | ---: |
| primal, `differentiable=False` | 0.968 | 0.877 | 0.528 | 0.458 |
| primal, `differentiable=True` | 1.293 | 1.202 | 0.373 | 0.304 |
| `jax.grad` | 2.280 | 2.152 | 0.373 | 0.338 |
| `jax.value_and_grad` | -- | -- | 0.366 | 0.316 |
| `jax.grad`, `check_adjoint=False` | -- | -- | 0.379 | 0.346 |

Under `jax.jit` the gradient costs 1.00 (median) to 1.11 (minimum) of the
differentiable primal, inside the 1.3 gate with no change to DKX. Executed
op by op, the same gradient costs 1.76 to 1.79, and the eager primal itself is
3.5 to 4 times slower than the compiled one.

The cause is dispatch count. The primal's jaxpr holds 5,783 primitives with
`differentiable=False`, 8,460 with it on (the residual guard and the implicit
solve wrapper), and the gradient's 13,698. Eager execution dispatches each one
separately; the block-Thomas substitution alone is a statically unrolled
recurrence over the 16 Legendre blocks (about 250 triangular solves per
factor-and-solve).

### Where a compiled gradient spends its time

| Component, compiled | median s | min s |
| --- | ---: | ---: |
| `to_block_tridiagonal` | 0.0345 | 0.0313 |
| `build_tier1_solver` (bands and factorization) | 0.3055 | 0.2643 |
| one block-Thomas substitution | 0.0300 | 0.0243 |
| `op.apply` | 0.0002 | 0.0002 |
| transposed apply | 0.0002 | 0.0002 |
| VJP of `A(p) x - b(p)` in `p` at fixed `x` | 0.0002 | 0.0002 |
| moments, value and gradient | 0.0001 | 0.0001 |

The factorization is about 85 percent of a compiled primal. A compiled gradient
adds one transposed substitution with refinement and the residual VJP, which
is where the 1.0 to 1.1 comes from: this is the adjoint pattern's cost already,
because `custom_linear_solve` implements it and XLA removes the unused tangent
of the factorization.

### Where an eager gradient spends its time

| Component, eager | median s | min s |
| --- | ---: | ---: |
| `build_tier1_solver` | 0.635 | 0.591 |
| `to_block_tridiagonal` (inside the above) | 0.121 | 0.112 |
| forward substitution | 0.129 | 0.123 |
| transposed substitution | 0.122 | 0.118 |
| `op.apply` | 0.011 | 0.011 |
| transposed apply, per application | 0.008 | 0.007 |
| building the transposed apply (`linear_transpose`) | 0.030 | 0.030 |
| two-probe operator norm estimate (residual guard) | 0.023 | 0.022 |
| moments | 0.023 | 0.021 |

An eager substitution costs 0.12 s against 0.03 s compiled, and a primal with
refinement runs two of them, a gradient four. The eager factorization is 2.2
times the compiled one.

### Open

- Explicit adjoint solve timed separately from `jax.grad` (*open*).
- The compiled `differentiable=False` primal is slower than the
  `differentiable=True` one (0.53 against 0.37 median). Explained in the
  step 7 section below: load noise, not extra work.
- Recycled Krylov route and the ambipolar root's implicit gradient (*open*).
- Transposed sparse direct floor at `3.18e-8` on the 1,204-unknown assembly deck
  (*open*).

## Decision (provisional)

The 2.0 to 2.6 of the 2026-09-20 record is eager-mode dispatch, not the
formulation: compiled, the gradient already meets the gate on this deck. The
work item is to make the solve path compile itself so that an eager caller
(`jax.value_and_grad(objective)` in the optimization workflow) reaches compiled
kernels: jitted factorization and substitution, jitted operator applications
inside the implicit solve, and a stop-gradient on the factorization input so
eager linearization does not carry the factorization's tangent.

## Follow-up

- Implement the compiled solve path, keep `custom_linear_solve` so forward mode
  keeps working, and measure eager and compiled gradients again.
- Measure first-call compile cost, since an internally compiled route trades a
  one-time compile for every later call.

## Step 7: the structured route compiles itself

What changed in `dkx.solve`, `dkx.drift_kinetic` and `dkx.run`:

- The band assembly and block-Thomas elimination run as one compiled program
  per operator structure; `Tier1Solver` is a pytree, so the physical
  coefficients are arguments and a neighbouring operator (an optimizer step, an
  `Er` scan point) reuses the executable.
- A differentiable structured solve is one compiled program: a single
  `custom_linear_solve` over every right-hand-side column, with the refined
  substitution, the original-residual guard and the residual norms inside it.
  The guard reaches its tolerances and diagnostics sink through an integer
  token, so one executable serves every call. Forward mode still works.
- The factorization and the substitutions take a gradient-free copy of the
  operator; the gradient flows through the operator's action and its transpose
  at the converged state, as the adjoint pattern prescribes.
- The refined substitution is fenced by optimization barriers, so the plain
  solve and the forward pass of the differentiable one compile it identically
  and return the same bits.
- `KineticOperator.rhs` and `profile_moments_from_operator` are compiled, and
  the output writer computes its moment table through the latter, so a file a
  run writes and one written from the same state hold the same numbers.
- Anything other than a `Tier1Solver` in the `factors` slot keeps the
  uncompiled reference path.

Measured on the office host (Xeon W-2295), cores 4 to 7, one XLA/BLAS thread,
CPU, float64, same 16,230-unknown deck, 15 interleaved repeats in one process
per version. **The host was saturated by other jobs throughout (one-minute load
55 to 100 on 36 hardware threads), so these ratios are indicative, not an
admission.** Process-CPU-time median (minimum in parentheses):

| Ratio | before (main) | after, run 1 | after, run 2 |
| --- | ---: | ---: | ---: |
| eager primal / compiled primal | 5.03 (4.74) | 1.30 (1.17) | 1.18 (1.21) |
| eager gradient / compiled gradient | 7.33 (6.83) | 1.36 (1.27) | 1.30 (1.24) |
| compiled gradient / compiled primal | 1.17 (1.17) | 1.19 (1.18) | 1.20 (1.22) |

- Absolute eager times: primal 3.2 s CPU before, 0.83 to 0.93 s after;
  gradient 5.5 s before, 1.09 to 1.16 s after.
- Fixed eager overhead, on a 5×5×6×4 deck where compute is negligible: primal
  1.33 s before, 0.017 s after; gradient 2.5 s before, 0.10 s after.
- Primal values agree with main to round-off (`FSABjHat` 0.032789784558823 in
  every arm).
- One-time compile cost of the first call, at the same load: eager primal 3.4
  to 13.6 s and eager gradient 5 to 25 s after the change, against 3.7 to 9.6 s
  and 3.8 to 16 s before. The first `jax.jit` of the whole objective is 12 to
  22 s either way. Not separable from the load at this spread (*open*).
- The anomaly: the compiled programs with `differentiable=False` and `True`
  contain the same 2 LU call sites and 188 triangular solves; the first lacks
  only the guard callback. Three later campaigns at the same load measured them
  equal within noise (0.435 against 0.438 s median on main), so the 0.53 against
  0.37 s above was load. The compiled gradient has 374 triangular solves and
  still 2 LU call sites: one factorization.

A second measurement after rebasing onto current main, with the fenced
substitution: an Apple-silicon laptop, one process at a time on one XLA/BLAS
thread (macOS cannot pin cores), main `95aa5286` and the branch alternated
A/B/A/B, 15 interleaved repeats each. **One-minute load 62 to 108 on 14 cores
throughout, so again indicative only.** Process-CPU-time medians:

| Ratio | main, round 1 | branch, round 1 | main, round 2 | branch, round 2 |
| --- | ---: | ---: | ---: | ---: |
| eager primal / compiled primal | 5.69 | 1.27 | 5.91 | 1.27 |
| eager gradient / compiled gradient | 9.57 | 1.38 | 7.94 | 1.27 |
| compiled gradient / compiled primal | 1.00 | 1.14 | 1.20 | 1.13 |

Eager primal 1.09 to 1.11 s CPU on main against 0.18 to 0.20 s on the branch;
eager gradient 1.80 to 1.85 s against 0.22 to 0.23 s. `FSABjHat` is
0.0327897845588232 on main and 0.0327897845588231 on the branch, and eager and
compiled gradients agree to `4e-16`.

### Decision

Adopted. The eager primal is within 1.3 of compiled on all four loaded runs and
the eager gradient sits at 1.24 to 1.38, on the edge of the gate at that load; the
remaining eager cost is about 0.1 s per gradient, spread across linearizing and
transposing some nine compiled calls. A quiet-host rerun decides the gate. If it
misses, the next reduction folds the factorization into the differentiable
program.

### Still open

- A quiet-host rerun of the ratios and of the first-call compile cost.
- The recycled Krylov route (`method="auto"` on the Fokker-Planck deck) and the
  ambipolar root under `jit`. A laptop smoke run of the reduced W7-X root at
  load about 50 (not admissible) took 29 radial-current evaluations and 266 s
  with eager Brent, against 1.54 s for the compiled secant (root `-3.69254`
  against Brent's `-3.69258`).
- A finite-difference or Taylor check of the gradient on `THat`,
  `dnHat/dpsiHat` and `dTHat/dpsiHat`; the small deck agrees to about `1e-9`.
- The transposed sparse direct floor at `3.18e-8`.
