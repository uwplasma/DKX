# Factor reuse

Production use is rarely one solve. A transport matrix is three right-hand sides
of one operator, a gradient is one transposed solve, an ambipolar root is a
sequence of operators that differ only in $E_r$, and a $\Phi_1$ Newton iteration
or an optimizer line search is a sequence of neighbours. The two direct routes
therefore return what they factored and accept it back; the recycled Krylov
route reuses its preconditioner and recycle space instead.

## The contract

```python
from dkx.solve import solve

first = solve(op, rhs_1, method="direct")
second = solve(op, rhs_2, method="direct", factors=first.factors)
adjoint = solve(op, cotangent, method="direct", factors=first.factors, transpose=True)
```

- `SolveResult.factors` holds a `Tier1Solver` on the structured direct route and
  a `DirectFactors` (the SuperLU or MUMPS factors and the two Ruiz diagonals) on
  the sparse direct route.
- `solve(..., factors=...)` makes the next solve a pair of triangular
  substitutions. Stored factors keep their backend; asking for a different one
  raises.
- `solve(..., transpose=True)` solves $A^\mathsf{T}u = b$ from the same factors.
  On the sparse route that is $A_s^\mathsf{T} z = D_c\, g$ with $u = D_r z$, a
  substitution through the stored $L$ and $U$ in the other order.
- A multi-column `rhs` in one call reaches one elimination on both direct
  routes, and each column is accepted against its own norm.
- `transpose=True` excludes `differentiable=True`, which builds its own
  transposed solve ({doc}`differentiation`).
- Nothing refreshes on its own. Reuse is passed in by the caller, and `solve`
  refuses to confuse a Krylov preconditioner (`precond=`) with direct factors
  (`factors=`).

## Measured cost

```{figure} ../_static/figures/readme/factor_reuse.png
:alt: Wall time and factorization count for one, three, and adjoint solves on the structured direct and sparse direct routes.
:width: 95%

Median of nine interleaved repeats, CPU float64, tolerance `1e-10`, every
solve's original residual checked before its time counted. Left: structured
direct, analytic `geometryScheme=1` deck at `(13, 13, 16, 6)` with pitch-angle
scattering, 16,230 unknowns. Right: sparse direct, `(7, 7, 8, 5)` with full
Fokker–Planck collisions, 1,962 unknowns. Regenerate with
`tools/publication_figures/generate_readme_showcase.py`.
```

```{list-table}
:header-rows: 1

* - Arm
  - structured direct, 16,230 unknowns
  - sparse direct, 1,962 unknowns
  - factorizations
* - one right-hand side
  - 0.34 s
  - 0.36 s
  - 1
* - three right-hand sides, one call
  - 0.33 s
  - 0.35 s
  - 1
* - three right-hand sides, three calls
  - 0.96 s
  - 1.19 s
  - 3
* - three right-hand sides, three calls, stored factors
  - 0.28 s
  - 0.28 s
  - 0
* - primal plus adjoint, stored factors
  - 0.44 s
  - 0.41 s
  - 1
```

The host was shared (load average 18 to 20), so the factorization count, which
contention cannot move, is the decisive column. A sparse transposed solve costs
0.15 of a primal. Measured on an Apple M3 Max CPU with SOLVAX 0.24.0 and JAX
0.10.2.

## Neighbouring operators

Factors of a neighbouring operator are accepted as an approximate inverse. The
stored inverse is applied, the residual is recomputed against the operator that
was passed in, and a solve that misses its tolerance refactorizes once and
repeats. The fallback is bounded by construction: the recovery is one
factorization, not more iterations.

On the direct routes the useful range is narrow. Scaling `THat` by $1+\epsilon$
and solving with factors built at $\epsilon = 0$, at tolerance `1e-10`:

```{list-table}
:header-rows: 1

* - Route
  - reused through
  - refactorized from
* - sparse direct
  - $\epsilon = 10^{-3}$
  - $\epsilon = 5\times10^{-2}$
* - structured direct
  - $\epsilon = 10^{-6}$
  - $\epsilon = 10^{-3}$
```

Every case converged, and every distant case recovered in one factorization.
The value of direct-factor reuse is therefore in right-hand sides and adjoints
of the same operator, where it is exact. The Krylov route is the opposite case:
a frozen preconditioner is useful precisely because it only needs to be
approximate.

## Krylov preconditioners and recycle spaces

The recycled Krylov route returns its preconditioner (`SolveResult.precond`) and
recycle pair (`SolveResult.recycle`), which the next solve accepts through
`precond=` and `recycle=`, with the last solution as `x0=`. The limits are
measured:

- A preconditioner frozen at `Er = 0` left a solve at `Er = 80` unconverged after
  6,000 iterations at a residual of 0.35 against `1e-10`.
- On the reduced W7-X magnetic-drift deck
  (`tests/reduced_inputs/filteredW7XNetCDF_2species_magneticDrifts_withEr.input.namelist`,
  2,104 unknowns), threading the last state and preconditioner through an
  ambipolar root search was slower than cold solves in two campaigns: 5.56 s
  against 4.73 s, and 9.20 s against 7.65 s. The bracket runs from `ErMin` to
  `ErMax`, so consecutive points are not neighbours. On a 73,444-unknown case
  bounded reuse took 33.14 s against 26.89 s cold and needed four cold retries.
  Warm starts are not a default for root finding.
- The same root took nine radial-current evaluations, 9.7 times its first solve.
  No factor or preconditioner policy brings a nine-evaluation root near the cost
  of one solve; fewer evaluations is the lever. The differentiable
  `dkx.er.ambipolar_er` seeds an unbracketed secant near the desired root for
  that reason.

## Limitation: the sparse transposed solve

The sparse transposed solve can stop above its tolerance for a general
cotangent. On the `DECK` of `tests/test_operator_assembly.py` (1,204 unknowns)
with `numpy.random.default_rng(0).standard_normal(1204)` it reaches `3.18e-8`
against the forward solve's `1e-10`, and further refinement sweeps do not move
it. It reports `converged=False` rather than a value. The 1,962-unknown full
Fokker–Planck grid reaches `2e-13`, so the floor depends on the operator.

## Scope

Stored factors belong to one operator and one process. They are not serialized
for restart, and they are not refreshed across changing profile or geometry
inputs. `dkx.er` and `dkx.batch` do not thread stored direct factors through
scans on their own; a caller that holds an operator fixed across right-hand sides
passes them explicitly. All measurements on this page are CPU.
