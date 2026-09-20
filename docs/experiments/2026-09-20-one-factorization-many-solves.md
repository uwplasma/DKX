# One factorization, many solves

## Hypothesis

Phase 2, step 4. Production use is never one solve: a transport matrix is three
right-hand sides of one operator, a gradient is one transposed solve, an
ambipolar root is five to ten operators differing in `E_r` alone, and a `Phi1`
Newton iteration or an optimizer line search is a sequence of neighbours. The
reuse contract of §5.1 says right-hand sides and adjoints share factors exactly
and neighbours are preconditioned by factors frozen at the first. This record
measures where each of the step's three admission items stands, implements the
part of the contract that the measurement says is missing, and measures again.
Owner: rogeriojorge.

## Admission test

Each item kills independently.

- **A.** An ambipolar root on the W7-X deck in at most 1.5x the wall time of its
  first solve.
- **B.** A gradient in at most 1.3x the primal, on every route.
- **C.** A transport matrix in at most 1.2x one right-hand side on the
  structured direct route.

Decks are the repository's own, at the smallest resolution that still exercises
the mechanism, because factor reuse is a structural property and does not need a
research grid to show. Item A uses
`tests/reduced_inputs/filteredW7XNetCDF_2species_magneticDrifts_withEr.input.namelist`
(two species, `(Ntheta, Nzeta, Nxi, Nx) = (5, 6, 6, 5)`, 2,104 unknowns,
`magneticDriftScheme=1`, full trajectories, against the public
`wout_w7x_standardConfig.nc`). Items B and C use an analytic `geometryScheme=1`
deck at `(13, 13, 16, 6)` with pitch-angle scattering, 16,230 unknowns, for the
structured direct route and at `(7, 7, 8, 5)` with full Fokker-Planck collisions,
1,962 unknowns, for the sparse direct one. Tolerance `1e-10` relative
throughout, and every solve's original residual is checked before its time is
counted.

Timing protocol: one process per campaign, every arm interleaved and repeated
nine times, median and minimum both reported, and a ratio is only accepted when
both sides of it pass. **The host was shared and loaded throughout** (one-minute
load average 15 to 45, from other work on the same machine), so the wall times
below are not publication benchmarks and the ratios carry that uncertainty. Each
campaign therefore also counts **factorizations**, which contention cannot move,
and that count is the decisive evidence for the reuse claim.

## Result

DKX on branch `perf/one-factorization-many-solves` from `6bee31ab`, SOLVAX
0.24.0, JAX/jaxlib 0.10.2, float64, local M3 Max CPU, Python 3.11.14.

### Where the three items stood before

| Item | Route | Baseline | Admission |
| --- | --- | ---: | ---: |
| C, transport matrix | structured direct | 0.76 | <= 1.2 |
| C, transport matrix | sparse direct | 0.88 | (not named) |
| C, transport matrix | recycled Krylov | 2.99 | (not named) |
| B, `jax.grad` / primal | structured direct | 2.35 | <= 1.3 |
| B, `jax.grad` / primal | recycled Krylov | 3.07 | <= 1.3 |
| B, `jax.grad` / primal | sparse direct | refused | <= 1.3 |
| A, root / first solve | W7-X, recycled Krylov | 6.58 | <= 1.5 |

**Item C already passed**, on both direct routes, and that is a result: a
multi-column `rhs` reaches one elimination. The structured route has presolved
its border columns for the forward and the transposed system from one set of
block-Thomas factors since it was written, and #256's assembly plus one SuperLU
factorization serves every column of `rhs2d` on the sparse route. The recycled
Krylov route's 2.99 is the expected one: it shares a preconditioner across
columns and nothing else, so three right-hand sides are three Krylov solves.

What was missing was everything that does **not** arrive as one array at one
moment. Three right-hand sides delivered as three calls cost three
factorizations; the adjoint had no way to reach the primal's factors at all;
and the sparse direct route refused a gradient outright.

### What was implemented

The two direct routes now return the factorization they built in
`SolveResult.factors` and take it back through `solve(..., factors=...)`, with
`solve(..., transpose=True)` solving `A^T x = b` from those same factors. On the
structured route the object is the existing `Tier1Solver`; on the sparse route a
new `DirectFactors` holds the SuperLU factors beside the Ruiz diagonals, and the
transpose is `A_s^T z = D_c g` with `y = D_r z`, which is a substitution through
the stored `L` and `U` in the other order.

Reuse across a *different* operator is permitted and is not a gamble. The stored
inverse is applied, the defect is measured against the operator actually passed
in, and the original residual decides: a solve that misses its tolerance
factorizes once and repeats. That is the cheap staleness test and the bounded
fallback the contract asks for, and it is bounded by construction, because the
recovery is a factorization rather than more iterations. A preconditioner frozen
at `Er = 0` has been measured to leave a Krylov solve unconverged at `Er = 80`
after 6,000 iterations at a residual of 0.35 against `1e-10`
(`2026-09-07-recycling-and-preconditioner-reuse.md`); a route that never
iterates cannot repeat that.

Two applications of the operator were also removed from every solve. The sparse
route recomputed the residual it had just measured as the last correction's
defect. More consequentially, `_transposed_apply` called
`jax.linear_transpose` **inside** the returned callable, so the whole operator
was re-traced on every application of the transpose rather than once per solve;
the transposition is now taken where it is built. The focused solve suites run
in 247 s where they took 406 s.

### Where the three items stand now

Nine interleaved repeats, load 18 to 20, median and minimum seconds.

Structured direct, 16,230 unknowns:

| Arm | median s | min s | factorizations |
| --- | ---: | ---: | ---: |
| one right-hand side | 0.3355 | 0.2406 | 1 |
| three right-hand sides, one call | 0.3279 | 0.2739 | 1 |
| three right-hand sides, three calls | 0.9593 | 0.7954 | 3 |
| three right-hand sides, three calls, stored factors | 0.2822 | 0.2347 | 0 |
| primal + adjoint, stored factors | 0.4406 | 0.3703 | 1 |

Sparse direct, 1,962 unknowns:

| Arm | median s | min s | factorizations |
| --- | ---: | ---: | ---: |
| one right-hand side | 0.3557 | 0.3305 | 1 |
| three right-hand sides, one call | 0.3540 | 0.3343 | 1 |
| three right-hand sides, three calls | 1.1931 | 0.9689 | 3 |
| three right-hand sides, three calls, stored factors | 0.2779 | 0.2130 | 0 |
| primal + adjoint, stored factors | 0.4090 | 0.3637 | 1 |

| Item | Route | Median | Min | Admission | Verdict |
| --- | --- | ---: | ---: | ---: | --- |
| C, one call | structured direct | 0.978 | 1.139 | 1.2 | pass |
| C, three calls, stored factors | structured direct | 0.841 | 0.976 | 1.2 | pass |
| C, one call | sparse direct | 0.995 | 1.011 | 1.2 | pass |
| C, three calls, stored factors | sparse direct | 0.781 | 0.644 | 1.2 | pass |
| B, primal + adjoint | sparse direct | 1.150 | 1.100 | 1.3 | pass |
| B, primal + adjoint | structured direct | 1.313 | 1.539 | 1.3 | **miss** |
| B, `jax.grad` / primal | structured direct | 2.571 | 2.565 | 1.3 | **miss** |
| B, `jax.grad` / primal | recycled Krylov | 2.064 | 2.078 | 1.3 | **miss** |
| A, root / first solve | W7-X, recycled Krylov | 9.72 | -- | 1.5 | **miss** |

Item C is admitted on the structured direct route, which is the route it names,
and holds on the sparse one. The three-call arm costs **zero** factorizations,
which is the claim the timings are only evidence for.

### Item B: what the remaining cost is, and what it is not

The sparse direct route passes, and the change there is categorical rather than
numerical: it had no adjoint at all, and the transposed solve now costs 0.15 of
a primal.

The structured direct route's explicit adjoint lands at 1.31 median and 1.54
minimum, missing the 1.3 gate on a shared host where the same arm has read
between 1.11 and 1.51 across campaigns. It is not admitted. What the arm does
is settled regardless of the seconds: one factorization serves both solves.

`jax.grad` misses on every differentiable route, and the reason is not the
factorization. Counted directly, `jax.grad` through the structured direct route
performs **exactly one** factorization, the same as its primal: the adjoint
already runs on the primal's `Tier1Solver`, which is what the contract asks for.
Reverse mode re-executes the forward pass and then runs the backward one, so a
ratio below 2 requires the factorization to dominate a solve it does not
dominate at these sizes, and the residue is the reverse pass over the operator's
own coefficient construction, not linear algebra that reuse could remove.
Disabling `check_adjoint` does not move it (2.900 against 2.990 seconds in the
baseline probe), so the residual audit is not the cost either; nor is the
transposed application, measured at 0.65 and 0.63 of the forward one on the two
decks. **Factor reuse cannot close item B on the AD routes because factor reuse
is already complete there.** Closing it would mean reducing what the tape
carries around the solve, which is a different piece of work.

### Item A: the item kills on evaluation count, not on factors

The W7-X root takes nine radial-current evaluations to bracket and refine, at
4.730 s against a 0.487 s first solve, i.e. 9.72x, or 1.08x per evaluation. The
admission asks for 1.5x, so it asks nine converged solves of nine different
operators to cost one and a half. The best reuse saving this programme has
measured on an `Er` sequence is 2.4x on CPU
(`2026-09-12-reuse-admission.md`), which would put a nine-evaluation root at
3.75x. **No factor or preconditioner policy reaches 1.5x while the root spends
nine evaluations**; only a continuation that spends two or three can, which is
the safeguarded Newton alternative §5.2 names against the existing Brent search.

The existing warm-start path also measured *worse* than cold on this deck, twice:
5.558 s against 4.730 s, and 9.197 s against 7.650 s in an earlier campaign at
higher load. The bracket the deck supplies runs from its `ErMin` to its `ErMax`,
so consecutive points are not neighbours, and threading the previous state and
preconditioner through costs more than it saves. This is consistent with the
recorded negative pilot on the 73,444-unknown case, where bounded reuse took
33.14 s against 26.89 s cold and needed four cold retries. Do not promote warm
start to a default on root finding.

### Reuse across neighbours is narrow on the direct routes

Scaling `THat` by `1 + eps` and solving with factors built at `eps = 0`, at
`1e-10`: the sparse direct route reuses through `eps = 1e-3` and refactorizes
from `eps = 5e-2`; the structured route reuses through `eps = 1e-6` and
refactorizes from `eps = 1e-3`. Every case converged and every distant case
recovered in one factorization. The value of the contract on these routes is
therefore in right-hand sides and the adjoint of the *same* operator, which is
exact, and not in neighbours, which at a production tolerance mostly refactorize.
That is the opposite of the Krylov route, where a frozen preconditioner is
useful precisely because it need only be approximate.

## Decision

Land the reuse contract on the two direct routes: factors returned and accepted,
the adjoint from the same factors, and a bounded staleness recovery. Item C is
admitted. Item B is admitted on the sparse direct route and is **not** admitted
elsewhere; item A is **not** admitted.

Item B on the AD routes and item A are not closed by anything in this step's
mechanism, and the measurements say why in each case, so neither should be
retried as a reuse problem. Item A's next measurement is a safeguarded Newton or
secant continuation against Brent, charged by evaluation count. Item B's next
measurement is the reverse pass around the solve rather than inside it.

No default changed. Reuse is passed in by the caller and nothing refreshes on its
own. The recycled Krylov route keeps its own reusable object, the preconditioner,
which is a different thing with different validity rules, and `solve` now refuses
to confuse the two.

## Follow-up

- Thread the stored factors through `dkx.er` and `dkx.batch` so a scan that does
  hold its operator fixed across right-hand sides reaches them without the caller
  wiring it.
- Replay forward, reverse and permuted `Er` sequences against independent cold
  solves, which §5.1 requires and this record does not do.
- Qualify the structured route's explicit adjoint on an idle host before calling
  1.3 either way.
- GPU behaviour is untouched here; every number above is CPU.
