# DKX research qualification plan

**Authoritative plan as of 2026-09-21.** This file defines the active order of
work and its scientific acceptance criteria. Release history belongs in
[`CHANGELOG.md`](CHANGELOG.md), implementation decisions in
[`docs/adr/`](docs/adr/), and bounded experiments in
[`docs/experiments/`](docs/experiments/README.md). Nothing here authorizes a
release.

## Mission and evidence standard

DKX aims to provide research-grade, differentiable neoclassical transport for
stellarators and tokamaks: fluxes, flows, bootstrap current, transport matrices,
and regular stellarator ambipolar roots from the radially local linearized
drift-kinetic equation. The target includes the scientifically supported SFINCS
Fortran v3 model, CPU and GPU workflows, verified derivatives, and one real
equilibrium-boundary optimization. Native Phi1 follows that optimization.

An advertised result must state the model, geometry, profiles, resolution,
precision, solver route, original-equation residual, observable uncertainty,
hardware, software versions, and cold/warm timing boundary where timing is
reported. The evidence tiers are:

1. **Code verification:** analytic identities, conservation, Onsager checks,
   transpose identities, and finite-difference or Taylor derivative tests.
2. **Solution verification:** separate and joint refinement of every relevant
   grid axis, per observable. A refused ladder produces no certified value.
3. **Cross-code verification:** matched equations and discretizations against
   SFINCS, and independent-model comparisons against MONKES, NTX, or yancc.
4. **Validation:** comparison with experiment, including experimental and model
   uncertainty. Cross-code agreement alone is not validation.

A small algebraic residual does not establish grid convergence. A successful
SFINCS process does not admit a reference unless its original equation, complete
state, requested tolerance, solver, and norm pass the same checks. Sparse
transpose support is **not** automatic differentiation: `A^T x = b` verifies a
linear algebra operation; gradients additionally require a correct pullback for
operator construction, observables, roots, and every active input.

DKX owns physics, discretization, admission, and physics-dependent solver policy.
[SOLVAX](https://github.com/uwplasma/SOLVAX) owns reusable linear algebra and
generic differentiation primitives. The supported dependency floor is
`solvax>=0.24.0`; raise it only when DKX consumes a released capability.

### Error and root contracts

For every published observable `Q`, define a physical scale and an application
tolerance `atol_Q + rtol_Q |Q|`. Budget algebraic, grid/quadrature, geometry,
and root errors separately; use a physical absolute tolerance for near-zero
currents and fluxes. Do not replace these budgets with one universal residual.
For linear `Q=c^T x`, `A^T lambda=c`, and `r=b-Ax`, `lambda^T r` estimates the
discrete algebraic error. It is not a rigorous bound unless its assumptions hold,
and an approximate adjoint needs its own allowance. Nonlinear observables need a
linearization remainder or conservative refinement evidence.

The variational-structure gap alone does not certify continuum discretization
error. Acceptance requires its stated assumptions and an independent,
observable-specific refinement study.

For a simple ambipolar root, propagate current uncertainty through
`|dJ_r/dE_r|` and include geometry uncertainty. A slope that is zero or uncertain,
or an error interval that overlaps another branch, is marginal rather than
stable. Sign samples cannot exclude an even crossing or tangency. Axisymmetric,
local, momentum-conserving neoclassical theory is intrinsically ambipolar and
does not select a unique tokamak `E_r`; prescribe it or name an additional
closure.

### Capability-family contract

[`validation/capabilities.toml`](validation/capabilities.toml) is the status
source. Each SFINCS-v3 family must distinguish four claims: native `Case`
execution, expert/compatibility execution, derivative support, and validated
evidence. A test that an option runs is not a validation checkmark.

| Family | Native | Expert/compatibility | Derivative | Evidence status |
| --- | --- | --- | --- | --- |
| Collisions/backgrounds | PAS and full-FP subset | broader species and collision controls | prepared-input subset | limited |
| Trajectories/`E_r` | qualified DKES/no-Phi1 envelope | broader drifts and trajectory terms | selected profile/`E_r` inputs | limited |
| Phi1 | not exposed as a coupled native workflow | kinetic, quasineutrality and gauge path | coupled derivative unqualified | compatibility only |
| Geometry | analytic, VMEC and Boozer profiles | schemes, asymmetry, signs and radial controls | selected prepared geometry inputs | limited |
| Drives/observables | profiles, moments and regular roots | broader RHS, sources, constraints, matrices and exports | named moments and roots | limited |
| Numerics/execution | guarded automatic routes | grids, solver controls, restart and complete output | route-dependent | per capability |

### Reuse contract

Compiled structure may be reused only for matching shapes, dtype, static model,
layout, and device. Numerical factors are exact only for the same operator,
border, and constraints; RHS-only changes may share them. For a neighboring
operator, factors are merely a preconditioner until that equation passes its own
original residual, or the route refactorizes. Initial states and recycle spaces
are guesses, never accepted results by identity. Refresh on changed constraints
or layout, failed admission, measured extra work, or memory pressure.

Compare fixed-matrix repeated RHS and transpose use separately from neighboring
operator reuse. Every promoted neighbor policy must replay forward, reverse, and
permuted sequences against independent cold solves, include failed or rejected
states, bound stale-state recovery, and show that accepted answers do not depend
on history. Cache and refresh decisions stay outside the differentiated physical
equations.

## Ranked queue

Work proceeds in this order. A lower item may run only when it does not compete
for the same reviewer, machine, or scientific decision. Each item ends with an
admitted result, an explicit refusal, or its written kill criterion.

### 1. Repair admission and qualify adjoints

Make reference and result admission uniform before producing more performance or
optimization claims.

- Require the original unscaled residual, complete-state status, requested norm,
  and per-column acceptance for forward and transposed solves. Preserve this
  typed evidence through native Result and convergence reporting; a successful
  grid-change diagnostic without it is not a research certificate. Never infer
  observable correctness from a solver success flag. Preserve independent batch
  rejection through root selection and recovery, including state-finiteness and
  underflow-safe checks; a zero norm cannot override a rejected state.
  Host optimization objectives must also reject nonfinite surface currents and
  independent scan failures. Imported scans must contain every required observable;
  absent bootstrap-current datasets cannot become a zero objective. Replacing
  either missing or failed currents with zero is inadmissible.
- Qualify active magnetic-upwind support in both the assembly pattern and column
  grouping against column sampling before larger direct timing.
  [Correction and public W7-X reproduction #268](https://github.com/uwplasma/DKX/pull/268)
  retain the unchanged matrix/operator verification threshold.
- Qualify sparse-direct transposes on independent general and physical moment
  cotangents. The reduced public W7-X case in #268 has reproducible general and
  ion-flow failures at requested `1e-10`; bootstrap current passes. Rebuilding,
  removing Ruiz scaling, and 50-digit refinement followed by a float64 cast did
  not meet the original-equation request. Keep Ruiz and the explicit refusals;
  do not weaken tolerance or claim a universal float64 floor. Require a bounded
  reproducer before any new arithmetic policy, and verify the actual precision
  of diagnostic dtypes. The unrecovered historical `3e-8` transpose observation
  is retained in the PR context, not treated as a reproducible regression.
- Qualify observable adjoints independently on PAS and full-Fokker–Planck decks.
  Compare the adjoint identity, finite differences over a step window, and a
  second-order Taylor remainder. Record branch and active-state assumptions.
- Keep timing descriptive. The universal “gradient ≤1.3× primal” gate is
  retired: [the factor-reuse experiment](docs/experiments/2026-09-20-one-factorization-many-solves.md)
  showed that reverse mode can reuse one factorization and still exceed that
  ratio because work remains around the solve.

**Acceptance:** every admitted derivative names its inputs and outputs, passes
the original forward and transpose equations, and passes finite-difference or
Taylor checks at a stated grid. Unsupported combinations are explicit in
[`validation/capabilities.toml`](validation/capabilities.toml).

### 2. Finish NCSX and W7-X uncertainty benchmarks

Complete the two public, reproducible benchmark families before extending the
physics envelope.

- Finish separate and joint `Ntheta`, `Nzeta`, `Nxi`, and `Nx` ladders for
  particle flux, heat flux, flow, and bootstrap current.
- Carry algebraic and discretization uncertainty separately. Test current and
  fluxes separately; convergence of one does not admit the others.
- Publish NCSX and W7-X tables with all attempted rungs, including refusals and
  non-monotone ladders. Preserve branch-search scope for W7-X roots.
- Before repeating a budget-limited large rung, capture compilation,
  preconditioner-build and Krylov phase costs separately. Memory admission
  alone does not establish affordable completion.

**Acceptance:** each quoted NCSX and W7-X observable has a reproducible relative
uncertainty of at most 1%, or the workflow refuses to quote it. The existing NCSX
record is a partial result, not completion:
[`2026-09-14-ncsx-refinement-ladder.md`](docs/experiments/2026-09-14-ncsx-refinement-ladder.md).

### 3. Add an identical-matrix optional backend in SOLVAX and a memory guard

The assembled sparse-direct route needs a production-class optional factorizer
for repeated right-hand sides and transposes of the **same matrix**. Put the
backend abstraction in SOLVAX; DKX supplies the assembled operator and admission.
Keep the portable fallback.

- Select an available optional multifrontal or supernodal backend at runtime,
  report its identity, and use one factorization for forward and transposed
  substitutions where the backend supports both.
- Estimate factorization memory from measured symbolic information or a
  conservative backend query. Refuse before allocation when the configured
  budget is exceeded. Do not use unknown count as a memory proxy. The current
  [MUMPS preflight #267](https://github.com/uwplasma/DKX/pull/267) accounts for
  transient RHS buffers and rechecks retained-factor headroom, but runs after
  grouped assembly. Bound assembly memory separately before claiming the entire
  direct route fits a process budget; conservative estimates are not OS limits.
- Compare identical matrices, permutations, scaling, tolerances, and residual
  definitions. Include fallback and missing-backend behavior.

**Acceptance:** the existing public 66,004-unknown benchmark uses the checked-in
full-FP fixture
`tests/reduced_inputs/filteredW7XNetCDF_2species_magneticDrifts_withEr.input.namelist`
at the recorded `(Ntheta, Nzeta, Nxi, Nx) = (11, 15, 20, 10)` resolution. It
passes forward and general-cotangent transpose admission. Attempt a larger public
rung only when the guard admits it within the declared budget. Report measured
peak memory; no backend-specific speed ratio is a universal release gate.

### 4. Measure root continuation and profile autodiff

The W7-X root timing gate failed because Brent used nine different operators;
factor reuse cannot make nine solves cost one. Compare safeguarded Newton/secant
continuation with the existing bracketed method using evaluation count and
branch evidence. Separately profile the reverse pass around the solve to locate
operator-construction, observable, and saved-state costs.

**Acceptance:** continuation returns the same admitted regular root and branch
classification as the bracketed reference, never weakens bracketing fallback,
and reduces accepted field evaluations on the benchmark family. Profile AD by
component and memory, then optimize only a measured bottleneck. Derivative tests,
not a fixed AD/primal timing ratio, decide correctness.

### 5. Run one bounded hard-regime probe

Spend at most three working days on one public or redistributable hard-regime
deck. Probe transient Krylov behavior with a field-of-values or pseudospectral
diagnostic. Earlier interventions did not improve the accepted solve; that does
not prove the associated mechanisms are absent or causal.

**Acceptance:** use the diagnostic to choose one bounded adjustment, then measure
it at two `Nx` values against the unchanged baseline. Retain it only for at least
1.5x fewer iterations or 20% lower end-to-end time or peak memory at the same
original-equation and observable acceptance. The diagnostic need not explain the
whole residual history. If no measured improvement appears, record the negative
result and stop; do not open another preconditioner branch. Non-redistributable
history may guide parameter choice but is not public benchmark evidence.

### 6. Qualify CPU and GPU workflows

Run installed-artifact workflows at fixed accepted physics and resolution:
single CPU, bounded CPU parallelism, one GPU, and independent GPU batches.
Measure compile, cold, warm, end-to-end time, host RSS, device memory, rejected
points, placement, and result agreement. Logical JAX CPU devices are not extra
physical CPUs; independent batching is not state decomposition.

**Acceptance:** CPU and GPU produce the same admitted observables and derivative
checks within stated tolerances; requested parallel execution is visible in
provenance; memory is bounded; cold fallback works; and any performance claim is
scoped to its exact workflow and hardware.

### 7. Deliver real VMEX optimization, then native Phi1

First replace the analytic geometry descent with an installed VMEX
equilibrium-boundary optimization. Use explicit profiles and prescribed `E_r`
for the first admitted chain; add a regular stellarator root only after item 4.
Differentiate boundary coefficients through equilibrium preparation and DKX,
with finite-difference and Taylor checks, geometric constraints, rejected-step
reporting, and a cold finer-grid recomputation.

**Acceptance:** objective improvement exceeds numerical uncertainty, constraints
and equilibrium residual pass, full-chain derivative checks pass on a smooth
branch, and an independent transport/current reference confirms the final design.
The existing example 08 remains an analytic proxy until this result exists.

The coordinate default remains the existing Boozer/general surface-angle geometry
with Legendre pitch; the VMEC path may use its native surface angles where the
same equilibrium, moments, Jacobian, and shape derivative agree. A coordinate
rewrite is deferred unless measured coupling after the ranked solver work
justifies it; reduced bounce coordinates may screen only inside their stated
validity range.

Then expose native Phi1 with the coupled kinetic, quasineutrality, and gauge
residual. Qualify linearized quasineutrality first, reproduce the W7-X impurity
comparison in [Mollén et al. (2018)](https://doi.org/10.1088/1361-6587/aac700),
and only then run warm impurity scans. A frozen-Phi1 calculation must be labeled
as such and cannot satisfy native coupled acceptance.

## Scientific deliverables retained

The queue supports four broad results:

- certified NCSX and W7-X transport/current tables with algebraic and
  discretization uncertainty;
- verified profile, root, and boundary derivatives on regular branches;
- a closure-accuracy map comparing full kinetics with monoenergetic and analytic
  bootstrap models across collisionality and finite `E_r`;
- a real equilibrium-boundary optimization, followed by a native Phi1 W7-X
  impurity result.

After those admitted interfaces, NEOPAX integration uses a small in-memory
protocol for species order, radial centers/faces, SI fluxes and Jacobians,
boundary conditions, validity, and refresh. It must check conservation and
lagged-response error before claiming a transport simulation. ESSOS follows an
accepted plasma target and must constrain field error, coil length, curvature,
distance, and current. Joint plasma/coil and open-field mirror optimization stay
deferred.

Low-collisionality studies use finite `E_r` and test the expected behavior rather
than assuming a universal asymptote; see
[Albert et al. (2024)](https://arxiv.org/abs/2407.21599). Reduced objectives may
screen designs, but their validity range remains explicit; see the
[differentiable bounce-averaging study](https://arxiv.org/abs/2412.01724) and
[direct neoclassical optimization](https://arxiv.org/abs/2406.04147).

## Deferred and closed directions

Deferred until an admitted user calculation requires them: a new fine-grid
coordinate/discretization backend; state decomposition or multi-host execution;
mixed/half-precision factors; differentiating through Krylov iterations; cuDSS
FFI; learned, Nyström, tensor, or PINN preconditioners; a second database
framework; joint plasma/coil optimization; mirror optimization; and neural
surrogates.

Do not reactivate these measured negative directions without a new hypothesis
and kill criterion:

- exact or factored cross-species collision retention
  ([record](docs/experiments/2026-09-19-collision-coupling-is-cross-species.md));
- diagonal balancing as the cure for hard-regime Krylov growth
  ([record](docs/experiments/2026-09-20-the-balanced-solve.md));
- float32 factors as a general memory route
  ([record](docs/experiments/2026-09-07-float32-factor-scope.md));
- a drop-tolerance ILU, geometric multigrid, and constant-coefficient or
  low-rank tensor preconditioners
  ([experiment index](docs/experiments/README.md)).

The speed triangle remains an optional measured route; it is not a claim that
the retained block predicts Krylov convergence. The unresolved historical
hard-regime ladder is evidence for refusal and bounded diagnosis, not an active
public performance result.

## Working method

- One coherent idea per change; tests and documentation travel with it.
- Record a hypothesis, budget, acceptance, kill criterion, and result for every
  algorithm experiment. Stop when the kill criterion fires.
- Benchmark pinned source and inputs with absolute times, compilation separated,
  synchronized repeats, original residuals, memory definitions, and all losses.
  Follow [Hoefler and Belli](https://htor.inf.ethz.ch/publications/img/hoefler-scientific-benchmarking.pdf)
  and the [JAX benchmarking guide](https://docs.jax.dev/en/latest/benchmarking.html).
- Keep compact public inputs, checksums, commands, and summaries in Git. Keep raw
  states, traces, build trees, machine paths, and non-redistributable inputs out.
- Every publication figure has a tracked input manifest, a generator under
  `tools/publication_figures/`, and an exact reproduction command in the evidence
  page. The command records source revision, dependencies, x64 state, device,
  case identity, and acceptance; papers cite a release tag rather than `main`.
- Keep fresh full clone, wheel, sdist, and installed DKX-owned files as separate
  size categories, each targeting less than 20 MiB. Report dependencies as setup
  cost and never change a measurement definition to clear the target.
- Maintain 95% line and branch coverage of stable reachable code as a release
  target and ratchet. Do not manufacture tests or delay urgent correctness work
  merely to move coverage.
- Commits and published repository text use maintainer-only authorship. Do not add
  assistant coauthor trailers.

## History

Earlier plans and execution detail remain available without duplicating them:

- [three-deliverable plan, #189](https://github.com/uwplasma/DKX/pull/189)
- [independent plan reconciliation, #190](https://github.com/uwplasma/DKX/pull/190)
- [production solver handoff, #253](https://github.com/uwplasma/DKX/pull/253)
- [factor reuse and transpose support, #261](https://github.com/uwplasma/DKX/pull/261)
- [v2.5.0 scope and retractions, #262](https://github.com/uwplasma/DKX/pull/262)
- [all experiment decisions](docs/experiments/README.md) and
  [release history](CHANGELOG.md)

Useful background remains in the public literature: the
[SFINCS model comparison](https://arxiv.org/abs/1312.6058),
[MONKES](https://arxiv.org/abs/2312.12248),
[yancc](https://arxiv.org/abs/2607.20861),
[bootstrap-consistent equilibrium optimization](https://arxiv.org/abs/2205.02914),
and the [VMEX references](https://vmex.readthedocs.io/en/latest/project/references.html).
