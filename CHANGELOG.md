# Changelog

## Unreleased

- Default CPU threads: with neither `DKX_CORES` nor `NPROC` set, DKX splits the cores it may run on (taskset aware, not `os.cpu_count()`) between XLA's pool, `min(8, cores // 2)`, and BLAS, the remaining cores per XLA thread, instead of 8 XLA threads and single-threaded BLAS. Explicit `DKX_CORES`, `NPROC` and BLAS variables are unchanged. On four pinned Xeon cores, warm W7-X monoenergetic goes from 2.58 to 1.53 s, NCSX full-FP (Krylov, coupled preconditioner) from 5.7 to 5.1 s, sparse direct from 25.2 to 24.4 s; unpinned on 36 threads 2.73 to 1.96 s and 6.13 to 5.58 s; on a 14-core laptop all within noise.

- `tools/benchmarks/cross_code_speed.py --dkx-xla-threads N` sets DKX's XLA pool apart from YANCC's. On four pinned x86 cores, XLA's pool and OpenBLAS compete inside every LAPACK call, and one XLA thread makes DKX's warm monoenergetic solve 2–3× faster (W7-X 3.4 to 1.6 s, HSX 19.5 to 6.8 s; MONKES 0.32 s and 1.5 s).

- `monoenergetic_database` now uses spectral (Fourier) angular derivatives when the deck does not set `thetaDerivativeScheme`/`zetaDerivativeScheme`. A deck that sets them keeps its choice. On the matched W7-X/HSX benchmark this is the discretization MONKES uses, and it converges like MONKES: HSX reaches 3% at 19x41x64 instead of 25x51x96, and W7-X is at 4e-5 one rung above 15x31x48 (finite differences: 0.6% at 25x51x96). Values at a fixed grid change within discretization error. To pair a database with a full-kinetic solve of the same deck, set the schemes in the deck.
- The full-band structured direct route (`build_tier1_solver`) builds its Legendre rows one at a time inside the elimination scan, stores only the Schur LUs, and regenerates the off-diagonal blocks in the substitution sweeps. Answers are unchanged (1e-12). On W7-X 15x31x48 the peak RSS falls from 1.95 GB to 0.94 GB and the first solve without a compilation cache from 8.2 s to 3.7 s; the warm solve is 8% slower. With the default persistent cache the first solve takes 1.4 s.
- `tools/benchmarks/cross_code_speed.py` has a `cached` stage that times DKX's first solve with the persistent compilation cache on.

- Phi1 is promoted from `compatibility_only` to `validated_limited`. On the geometryScheme 4 Phi1 example deck and a trace-carbon variant, DKX matches Fortran SFINCS v3 to 2.0e-6 over an eight-rung resolution ladder (`validation/phi1_sfincs_benchmark_v1.json`). `stable_candidate` is withheld: there is one geometry, Fokker-Planck collisions only, zero Er, and the trace-impurity flux is resolved only to about 3% in Nx.
- `dkx.phi1.phi1_solution` gives the converged Phi1 state with a matrix-free implicit adjoint: GMRES on the transposed Jacobian (a VJP of the coupled residual) on the active Legendre-truncated subspace, preconditioned by the last Newton step's transpose preconditioner. On a native `phi1 = "kinetic"` case the gradient matches central finite differences to 1.2e-9 and the Taylor remainder falls as h^2 (ratios 4.02, 4.01).
- Case inputs no longer carry a `schema` key. Older files with `schema = 1`
  still load (the key is ignored); other values are refused. Case IDs are
  unchanged.
- `dkx schema` is renamed `dkx template` (`schema` stays a hidden alias for one
  release). `Case` no longer has a `schema` attribute and `SCHEMA_VERSION` is
  no longer exported; case IDs are unchanged.

### Solvers

- Full Fokker–Planck and improved Sugama decks with DKES trajectories, no
  tangential drifts and no Phi1 now have an exact structured direct route: a
  block elimination along the Legendre index whose blocks couple all species
  and speeds (`dkx.structured_direct.build_coupled_solver`). It eliminates from
  the highest Legendre mode down, lifts the momentum null space of the `L = 1`
  block with a rank-`Nspecies` term removed again by Woodbury, and handles every
  `constraintScheme` border. `method="auto"` takes it within the structured
  memory budget and 150 GFlop per right-hand side; on the upstream suite it
  solves `transportMatrix_geometryScheme2` in 2.8 s against 33 s (1,181 GCROT
  iterations) and `transportMatrix_geometryScheme11` in 9.9 s against 48 s, and
  leaves `geometryScheme4_2species_noEr` and the W7-X/HSX decks, where Krylov is
  faster or the blocks do not fit, on recycled Krylov. A speed-triangular sweep
  was measured first and is not exact in this speed basis (lower triangle 1-3%
  of the diagonal, electron-ion block as large as its diagonal).
- `preconditioner="coupled"`: the speed-coupled elimination as a Krylov
  preconditioner, exact for Legendre blocks `L < 2` and `(species, x)`-diagonal
  (float32) in the tail; `E_r` and drift `L ± 2` terms are dropped from the
  factored operator. Iterations fall from 46 to 17 on a 396k-unknown HSX FP
  deck at equal wall time; opt-in, `auto` is unchanged.
  It keeps the `L`-diagonal magnetic drifts at `L <= 2` (23 to 17 iterations on
  the reduced W7-X drift deck) and stores tail bands as per-species blocks times
  the speed scale, `Nx` times smaller.
  The tail is now eliminated and swept with `lax.scan` (padded with identity
  blocks where `Nxi_for_x` truncates), its bands are kept as two scalars per `L`
  over shared float32 streaming and mirror blocks, and the dense `L < 2` Schur
  step is formed one column pair at a time: the compiled HSX FP production
  factorization drops from 11.5 GB output + 26.6 GB temporaries to
  11.7 GB + 4.4 GB.
  Its tail and dense `L < 2` blocks are now factored in float32: on the
  production HSX FP deck (1.88M unknowns, 8 cores) 17 iterations, 508 s and
  16.6 GB, against 1,100 s and 15.0 GB for the coarse preconditioner.
- `method="auto"` now preconditions Fokker–Planck and Sugama Krylov solves with
  `coupled` when its estimated peak fits the Krylov memory budget (default: half
  the available memory), else `coarse`.
- `dkx.solve` keeps its API; the structured route's applicability tests and
  memory model moved to `dkx.structured_direct`. Structured solves refine
  against the pinned operator (identity rows on truncated `Nxi_for_x` DOFs).
- Native `[physics] phi1 = "kinetic"` runs the coupled kinetic + quasineutrality + gauge Newton–Krylov solve on `profile` cases and is accepted by the nonlinear residual; `dkx convert` maps kinetic-only Phi1 decks to it. `"full"`, adiabatic species, quasineutrality option 2 and the ambipolar workflow stay refused. Status remains `compatibility_only`.

## v2.8.0 — 2026-10-05

A converged kinetic bootstrap row compiled as one program, a momentum-conserving
correction for pitch-angle-scattering solves, refined ambipolar roots, and the
options `vmex --neoclassical` uses.

### Documentation and repository

- Documentation rebuilt (#292): getting started, user guide (every case field,
  outputs, CLI, Python API), tutorials, how DKX works, physics, numerics,
  benchmarks, examples, features and design decisions. The README defines the
  case schema before using it.
- Historical development files removed (#293, #298); dead doc citations
  repointed; `dkx run --out` help, the `dkx.run` method docstring and the
  analytic tokamak wording corrected; `[output].plots` wired.
- Ambipolar root stability is classified from the outward current (#301); the
  optional Boozer backend is declared as the `booz` extra (#302).

### Bootstrap objective

- `KineticBootstrapMismatch` defaults to `Nxi = 48` (was 16). Pitch-angle
  convergence study (plan.md 16.7): on the VMEX QA beta = 2.5% deck with its
  published profiles, `<j.B>` at `Nxi = 48` is within 1.1% of `Nxi = 96-128`
  for |E_r| >= 5 kV/m and within 3% at |E_r| <= 1 kV/m; `Nxi = 16` was 16-44%
  low. A realistic E_r does not converge at far lower `Nxi` (one step of the
  ladder at most), so the row keeps `E_r = 0` by default. Rows built with an
  explicit `resolution` are unchanged.

- `KineticBootstrapCurrent(ambipolar=True)` refines its bracketed ion root
  with the same Brent refinement as the representative radial scan (at most
  4 more single-field solves) and reads `<j.B>` from the solve at the refined
  root instead of interpolating across the scan bracket.

### Representative run

- The ambipolar root of each surface is refined after bracketing: Brent's
  method on the bracket, at most 4 more single-field solves, stopping at
  |J_r| <= 1e-6 max|J_r| or a 0.01 kV/m bracket. Moments are read at the
  refined root instead of interpolated across a 10-20 kV/m bracket. On the
  VMEX QA beta = 2.5% bootstrap deck with its published profiles (default
  grid) the roots move from -11.8, -25.8, -49.0, -39.8, -43.3 to -11.9, -26.1,
  -46.5, -43.4, -39.0 kV/m, and <j.B> at r/a = 0.55 from -4755 to -3192
  kA/m^2, for 18 extra solves (radial-scan wall time 59 s to 117 s). The
  outer roots stay non-monotone: there J_r is within 1-5% of zero over tens
  of kV/m and its crossings move by tens of kV/m on the `full` grid, so that
  spread is a resolution effect, not a bracket artifact.
- The ambipolarity panel spans its row; the two empty slots beside it are gone.
- `run_representative` accepts `surfaces`, `er` (ambipolar root or a
  prescribed kV/m), `collision_operator` (`fp`, `pas`,
  `pas+momentum_correction`), explicit `profiles` (`ne_coeffs`, `Te_coeffs`,
  `Ti_coeffs` polynomials in s, as VMEX's `KineticProfiles`; T_i may differ
  from T_e) and `redl_jdotb`, an external Redl curve drawn on the bootstrap
  panel. Defaults are unchanged.
- Fix: the radial scan solved every surface with the n, T and gradients of
  r/a = 0.5. Each surface now uses its own plasma. On the VMEX QA beta = 2.5%
  bootstrap deck this removes the r/a = 0.55 outlier (-46 kV/m between -31
  and -28). The Er profile is now monotone: -13, -27, -50, -60, -61 kV/m at
  r/a = 0.25 to 0.85. A dense scan confirmed that the root itself was a real
  crossing for the plasma it was given, not a root-pick or bracket error.
- The evaluated root is the most negative *stable* root under the
  outward-current rule of #301. The ambipolarity panel is now drawn for every
  surface, with each root labelled ion, unstable or electron.
- Fix: the caption no longer clips at the canvas edges. Long lines wrap, and
  the layout rect's height no longer runs past the top of the figure.

### Physics

- Rewrite `dkx.momentum_correction` as the Sugama–Nishimura moment method on the
  structured pitch-angle solve (Phys. Plasmas 9, 4637 (2002); Maassberg, Beidler &
  Turkin, Phys. Plasmas 16, 072504 (2009)). The parallel particle flow, heat flow and
  next Sonine moment of every species are coupled through the full Fokker–Planck
  operator (friction `l^ab_ij` and field-particle restoration from DKX's own
  Rosenbluth-potential operator), the projection is weighted by `nu_D` so the model
  conserves total parallel momentum exactly, and `<j.B>`, flows and back-substituted
  radial fluxes are read off one corrected state. `momentum_corrected_solve(pas, fp)`
  is traceable and differentiable; `friction_drives` caches the velocity-space part.
  `KineticBootstrapMismatch(collision_model="pas+momentum_correction")` uses it as a
  VMEX objective row. Accuracy against full Fokker–Planck on the same grid is in
  `docs/physics/reduced_models.md`.

### Removed

- The single-moment, database-based correction (`ParallelViscosity`,
  `parallel_viscosity`, `parallel_friction_matrix`, `solve_corrected_flows` and the
  database signature of `momentum_corrected_bootstrap`). It carried the parallel
  particle flow only, so the temperature-gradient bootstrap drive was lost, and closed
  the system with an ad hoc `M0 V_unc` drive. Accessing the old names raises an error
  that names the replacement; `dkx.api.momentum_corrected_bootstrap` now takes the
  pitch-angle operator and its Fokker–Planck twin.

### Performance

- Retain the Fokker-Planck speed triangle in the sparse Krylov preconditioner
  (#303). `solve(preconditioner="sparse_triangle")` keeps the self-species
  upper speed triangle that `"sparse"` drops, reusing the same sparse LU
  factors by back-substitution over speed. The stalled-solve escalation now
  tries it in place of `"sparse"` on Fokker-Planck and Sugama decks; an
  explicit `preconditioner="sparse"` is unchanged. On the HSX Fokker-Planck
  DKES deck it cut GCROT iterations from 46-55 to 16-18 across a four-rung
  ladder (54,564 to 786,244 unknowns) and on the W7-X paper deck from 22-26
  to 10-11, with currents and fluxes equal to the solver tolerance.
- Compile the kinetic bootstrap-current row of a VMEX optimization as one
  program. `KineticBootstrapMismatch` traced one Boozer transform and one
  kinetic solve per surface as an unrolled loop, was dispatched op by op when
  VMEX evaluated it eagerly at set-up, and was differentiated forward through
  the kinetic solve once per boundary dof by VMEX's block Jacobian. The
  surfaces are now one batch axis of one compiled program, an eager call
  compiles once per runtime, and the forward derivative applies the
  reverse-mode Jacobian (one adjoint solve per surface) to every tangent
  (`batched_with_reverse_jvp`). The Boozer plan is also built under an outer
  `jit`, which used to raise. Values and Jacobian rows are unchanged to
  3e-14 and 1.4e-12 relative (8.6e-6 for the Fokker-Planck Krylov route,
  whose tangent solves become adjoint solves at the same tolerance). On a
  shared 36-core host (8 pinned cores, load 8-33), the row's added cost over
  the same VMEX problem without it fell from 92-122 s to 56-74 s for set-up
  plus one residual and one Jacobian (three surfaces, default grid); an eager evaluation from 68-80 s cold and
  6-7 s warm to 20-26 s and 0.6 s; the example's CI smoke pass from 358 s
  to 298 s.

## v2.7.0 — 2026-09-27

Faster by default and ready for optimization: a memory-aware Krylov restart
that removes the high-`Nx` stall, a self-compiling structured route for
callers that do not `jit`, and a drift-kinetic bootstrap-current objective for
VMEX optimization.

### Correctness

- Refuse failed bootstrap-current evidence (#274). `KineticBootstrapCurrent`
  replaced a nonfinite surface current with zero, so an unsuccessful solve
  looked like an optimal one; nonfinite currents, residuals and squared terms
  and a non-positive reference current are now refused, and the ambipolar
  interpolation requires every scan point to be accepted and finite. The
  scan-promotion audit likewise substituted zero when an output had neither
  `FSABjHatOverRootFSAB2` nor `FSABjHat`, passing the gate with a zero
  objective; a missing or nonfinite bootstrap current is now refused.

- Convert the handedness of the Boozer route in one place
  (`boozer_route_psi_a_hat`, and its traced form
  `convert_boozer_route_handedness`): `psiAHat = |phi_edge|/(2 pi) signgs
  sign(G + iota I)`. A VMEX equilibrium has `signgs = -1` and `booz_xform`
  returns `G > 0`, so taking `psiAHat = +|phi_edge|/(2 pi)` flipped every flux
  and `<j.B>` on that route against the VMEC-file route and Redl.
  `optimize_QA_bootstrap.py`, `optimize_QH_bootstrap.py` and the gradient hook
  of `bootstrap_consistency_kinetic_loop.py` printed the flipped sign; their
  objectives were squares or magnitudes, so the optimizations were unaffected.

### Performance

- Retain only the active pitch rows in reusable coarse factors (#273). The
  generated reusable preconditioner factored every subsystem through all `Nxi`
  rows, including the masked identity rows past `Nxi_for_x`; it now factors
  `max(1, Nxi_for_x)` rows and applies the uncoupled tails as
  `r / (1 + floor)` at factor precision. On the recorded NCSX layout that is
  744 rows instead of 1,111 (predicted float64 LU and pivots 14.5 GiB instead of
  21.7 GiB); a traced `Nxi_for_x` keeps the rectangular layout. The operator,
  masks, pinning and residual acceptance are unchanged.

- Memory-aware Krylov restart by default. `solve(restart=None)` and
  `SolverOptions(restart=None)`, the new defaults, run five FGMRES cycles of
  30 and two of 100, then continue from the iterate at the longest restart, at
  most 1,000, whose basis (`2 * restart * unknowns * 8` bytes, since flexible
  GMRES stores `V` and `Z`) fits `krylov_memory_budget_gb`
  (`DKX_KRYLOV_MEMORY_BUDGET_GB`, else a quarter of available memory). The
  previous policy widened only to 100 and only while two such bases fit in
  256 MiB, so decks above 167,772 unknowns never widened. On the HSX-like
  `Nx` ladder the iteration growth from `Nx = 10` to 16 was restart
  stagnation: 2,788 iterations at restart 200 against 357 at 1,000, and 7,167
  against 390 at `(Nxi, Nx) = (40, 16)`
  (`docs/experiments/2026-09-23-restart-and-direct-reach.md`). On the public
  tokamak full-Fokker-Planck deck at `(Nx, Nxi) = (24, 40)`, 87,362 unknowns,
  the default converges in 1,736 iterations where the previous one stalled
  and failed every rung of the stall ladder (`docs/performance.rst`, "Krylov
  restart length"). Decks that converged within 350 iterations follow the
  same path as before. The stall
  ladder's larger-budget rung spends its budget at the wide restart. An integer
  `restart` fixes the cycle size as before; explicit `method="gmres"`,
  differentiable and traced solves, and caller-supplied preconditioners keep a
  fixed size of 30 unless given one.
- `dkx solve-v3 --restart` was printed but never applied; its default reads
  `auto` and its help says so.

### Optimization

- `dkx.bootstrap.KineticBootstrapMismatch`: the drift-kinetic bootstrap
  current as a traced VMEX objective term, with the interface and normalized
  residual of VMEX's `RedlBootstrapMismatch`. On each kinetic surface the chain
  VMEX state -> `boozer_input_tables` -> `booz_xform_jax` -> DKX structured
  solve -> `<j.B>` is traced, so VMEX's implicit Jacobian differentiates it;
  `mismatch=False` gives a pure `<j.B>` row. Pitch-angle scattering is the
  default collision operator (cheap, no momentum restoration).
- `examples/optimization/QA_optimization_bootstrap_dkx.py` is VMEX's
  `QA_optimization_bootstrap.py` with the DKX row in place of Redl's
  (`BOOTSTRAP_MODEL = "dkx" | "redl" | "both"`), replacing the
  finite-difference version; the seed deck ships in `examples/data`.
- The structured direct route compiles itself for callers that do not `jit`
  (#279). The factorization, the refined substitution, the whole
  differentiable solve (one `custom_linear_solve` with the residual guard
  inside it), `KineticOperator.rhs` and `profile_moments_from_operator` each
  run as one cached executable per operator structure, so an eager
  `jax.value_and_grad` of an objective no longer dispatches, linearizes and
  transposes thousands of small operations one at a time. On the
  16,230-unknown structured deck, on heavily loaded hosts, the eager primal
  went from 5.0–5.9× to 1.18–1.30× of its compiled time and the eager
  gradient from 7.3–9.6× to 1.24–1.38×; the fixed eager overhead fell from
  1.33 s to 0.017 s (primal) and from 2.5 s to 0.10 s (gradient). Answers are
  unchanged to round-off, and the plain and differentiable solves return the
  same bits. The first call of each new operator structure pays a one-time
  compile.

## v2.6.0 — 2026-09-21

Acceptance tightened where it could pass a wrong answer, MUMPS as an explicit
direct backend, and convergence and optimization reports that no longer claim
more than their evidence supports.

### Correctness

- Admit each right-hand side against its own norm (#265). A multi-column solve
  admitted every column against the *largest* right-hand side's norm, so with
  norms `[1, 1e6]`, residuals `[1e-5, 0]` and a tolerance of `1e-10` the small
  column passed while missing its own target by a factor of 100,000. Each
  column now meets `max(atol, tol * ||b_j||)`, reused factors included; zero
  columns use the requested absolute tolerance, and nonfinite inputs, negative
  residual norms and invalid tolerances are refused.
- Include the magnetic drift's upwind support in the sparse assembly (#268).
  Streaming uses centred derivatives and the tangential magnetic drift separate
  upwind ones, whose toroidal radius is three points rather than two on the
  public reduced W7-X case; the pattern and the column groups now use the union
  of the active supports. The assembly's own `1e-10` check against the operator
  refused the old pattern, so no wrong matrix was ever factored; decks without
  magnetic drifts assemble exactly as before.
- Reject invalid optimization evidence and require a genuine refinement before
  promotion (#264): nonfinite or negative residuals, a single baseline grid
  labelled as a ladder, and a refinement that changes radius or species count
  are refused, and an absent backend comparison is reported as `untested`.

### Execution

- Select MUMPS explicitly as the sparse direct backend,
  `SolverOptions(method="direct", direct_backend="mumps", memory_budget_gb=...)`
  (#267). SuperLU stays the default and the default path is unchanged. The
  budget is checked after assembly and before factorization, against measured
  process memory, the COO storage, the right-hand-side buffers and the host's
  currently available memory, and again before a retained factorization is
  applied to a wider right-hand side. An explicit MUMPS request that cannot be
  met is refused and never silently falls back to SuperLU. It needs the MUMPS
  adapter of SOLVAX 0.25.0 and PyMUMPS; with SOLVAX 0.24 it raises an
  `ImportError` naming what is missing.

### Reporting

- Report grid convergence and original-equation acceptance as separate
  verdicts (#269). `dkx converge` keeps `converged` for the observable changes
  and adds `original_equations_accepted`, which requires complete, finite,
  typed residual evidence within the requested tolerance on every rung; the
  exit status is zero only when both pass. Native runs reapply the operator to
  the returned state and reject a false solver success.
- Compare bootstrap current in `dkx converge`, keep earlier rungs when a later
  one fails, and record failed and refused refinements without claiming
  convergence (#266). The NCSX record gains an executable reconstruction from
  pinned public sources with four SHA-256 checks.
- Report memory in one unit (#271). Traces mixed decimal MB with MiB and could
  present device capacity or a historical peak as current usage; profiling now
  uses MiB throughout and keeps current RSS, peak RSS and current device usage
  distinct.

### Development

- Let a cancelled CI run stop before the final gate (#270).

### Known limitation

- The sparse direct route's transposed solve can stall above its tolerance for
  a general cotangent. On the operator of
  `tests/test_operator_assembly.py::DECK` (1,204 unknowns) with the cotangent
  `numpy.random.default_rng(0).standard_normal(1204)`, it reaches a relative
  residual of `3.18e-8` against the forward solve's `1e-10` on the same
  factors, in 2.5.0 and here alike; further refinement sweeps do not move it.
  The route reports `converged=False` rather than a value, so this is an
  accuracy limit that is reported, not a wrong answer. The 1,962-unknown
  full-FP grid of #265 reaches `2e-13`, so the limit depends on the operator.

## v2.5.0 — 2026-09-20

The production solver program of #253: the operator assembled from products
with it, the sparse direct route rebuilt on that assembly and made reusable,
and the gap deck's iteration growth traced to the speed resolution with six
candidate causes measured and excluded.

### Execution

- Return the factorization the two direct routes build, in
  `SolveResult.factors`, and take it back through `solve(..., factors=...)`;
  `solve(..., transpose=True)` solves `A^T x = b` from those same factors. A
  transport matrix delivered as three separate calls cost three factorizations
  and now costs none: 0.96 s against 0.28 s on a 16,230-unknown structured
  deck, 1.19 s against 0.28 s on a 1,962-unknown sparse one. The sparse direct
  route had no adjoint at all; a transposed solve is now 0.15 of a primal,
  reached by substituting through the stored `L` and `U` in the other order.
  Reuse across a *neighbouring* operator is permitted and checked rather than
  assumed: the defect is measured against the operator actually passed in, and
  a solve that misses its tolerance refactorizes once, so recovery is bounded
  by a factorization instead of an unbounded Krylov wait.
- Take the operator's transposition once per solve rather than once per
  application. `_transposed_apply` called `jax.linear_transpose` inside the
  callable it returned, so every adjoint application re-traced the whole
  operator. The focused solve suites run in 247 s where they took 406 s.
- Assemble the sparse direct route's matrix in fewer products by dropping the
  requirement that the angular separation divide the grid. The grouping was
  already optimal for its own conflict structure -- every group held exactly 72
  columns and `633,600 / 72 = 8,800` -- but at `Ntheta = 11`, a prime, the only
  divisor above twice the stencil radius is 11 itself, so every theta became
  its own class. Checking the wrap-around gap directly instead takes the
  633,604-unknown deck from 8,800 products to 4,800, each product being one
  operator application; the recovered matrix is unchanged entrywise.

- Build the sparse direct route's matrix from products with the operator rather
  than one column at a time, scale it before factoring, and correct the defect
  once. Sampling costs one operator application per column, which is what
  `max_dense_size` bounds and why the route refused every production deck; the
  assembly costs one per group of columns that share no row, 5,508 for the
  66,004-unknown collaborator grid. That deck now solves in 846 s to a relative
  residual of `1.3e-14`, where the route previously refused it.
- Apply the exact speed triangle (`preconditioner="coarse_triangle"`) by
  back-substitution over `x`, rather than reaching the same inverse as a
  nilpotent series of `n_x` sweeps of the whole band. The map, the factors and
  the memory are unchanged, and one application at NCSX `(25, 37, 61, 8)` costs
  3.7x less: 2.3x a `coarse` application against 8.5x.
- Continue every escalation rung from the stalled iterate, rather than from the
  original guess, when that iterate is finite and improved on the right-hand
  side it started from. On a forced NCSX stall the rung that converges takes 18
  iterations instead of 27, for the same answer. A diverged iterate is refused,
  so a rung never starts further away than the original guess.
- Escalate a stalled recycled-Krylov solve to that triangle before `sparse` and
  `multigrid`. Those three are inverses of one simplified operator, so none of
  them answers a stall caused by the Fokker-Planck speed coupling that operator
  drops; retaining its upper triangle changes the operator being inverted, cuts
  NCSX iterations 2.3-4.1x, and reuses the factors the coarse route already
  built. A deck with no dense collision operator skips the rung, because there
  the triangle is the stalled preconditioner under another name.

### Research records

- The speed-triangle back-substitution measurement, and the batched LU dispatch
  measurement on CPU and GPU, in `docs/experiments/`.
- Why the gap deck is slow, in four records. The dropped collision coupling is
  not the cause although it is 0.9999997 of `A - M`; retaining it exactly costs
  17% *more* iterations. The cost is carried by `Nx` and not `Nxi`, and is
  independent of the tolerance. The preconditioned operator is strongly
  non-normal and far more so with `Nx`, and a diagonal similarity that removes
  1.5e10 of that eigenvector conditioning moves the iteration count by at most
  1.05x, so the balanced solve is killed on its own criterion. Six candidate
  causes are now measured and excluded.
- What one factorization serving many solves does and does not buy: the
  transport matrix admitted on both direct routes, the gradient admitted on the
  sparse one, and `jax.grad` measured to miss its gate for a reason factor
  reuse cannot address, since reverse mode re-executes the forward pass and the
  adjoint already runs on the primal's factors.

## v2.4.0 — 2026-09-15

The integrated #169–#191 stack, the reconciled #190 plan and the work from the
2026-09-13 independent review (#229–#246).

### Correctness and differentiation

- Report stalled solves without attributing an unmeasured physical cause or
  recommending changes to the requested electric field and accuracy.
- Add the opt-in `SfincsMatrixThreshold` parity switch, which reproduces SFINCS
  v3's `1d-12` matrix sparsification on the Fokker-Planck operator. It accounts
  for a 12-19% bootstrap-current gap on a hot-electron HSX-like deck; the
  default keeps every entry.
- Add `dkx.validity.normalized_radial_electric_field` and its operator-build
  form: the per-species `E_*` of Landreman et al. (2014), whose magnitude above
  about 1/3 marks the E x B resonance regime where trajectory models separate and
  resolution must follow `E_r`.

- Add opt-in bounded GMRES reuse to the host ambipolar root, with one cold retry
  after failed admission and independent final acceptance. Keep retry policy out
  of root differentiation.
- Correct the ambipolar teaching case to quasineutral analytic W7-X with full-FP
  collisions. JIT the geometry-optimization value/gradient, check its original
  kinetic residual and retain a three-step finite-difference window.
- Report monotone refinement ladders that converge faster than `max_order`, as
  spectral pitch and angle directions do, with three times the last difference
  as the grid uncertainty instead of refusing them. Diverging and oscillating
  ladders are still refused.

- Keep discrete pitch layouts static under JIT; refresh full-FP density kernels
  and opt-in temperature-dependent coefficients for prepared profile derivatives
  (#173, #174, #178, #186).
- Prepare immutable native profile/field scans with per-input original-equation
  status and complete kinetic-state recovery. Differentiate supported regular
  ambipolar roots with original primal/transpose admission (#182, #184, #187, #188).
- Reject invalid kinetic states, roots and adjoint references; qualify physical
  profile sensitivities with independent cold solves and Taylor checks
  (#180, #184, #188).

### Execution

- Bound the first automatic Krylov attempt and reuse its factors with a wider
  restart window when the basis-size guard permits. Preserve explicit solver
  settings and the differentiated path.

- Propagate nested legacy scan failures, distinguish failed progress from success,
  and retry incomplete Er points while preserving previous attempt files.
- Default the host BLAS pools to one thread. XLA's threadpool already runs
  batched CPU LAPACK in parallel; four BLAS threads per call made the NCSX
  `(21,37,61,8)` coarse factorization about 3x slower. Explicit settings win.

- Build the transposed coarse preconditioner on its first application, and
  synchronize coarse builds on their factors instead of a discarded zero-vector
  application. Non-differentiable solves no longer pay two transposed coarse
  applications per build; the preconditioner maps are unchanged.
- Factor the dense coarse preconditioner straight from its pinned row generator
  inside one compiled computation, so no band is materialized before
  elimination, and store only the rows each subsystem's `Nxi_for_x` keeps. The
  elimination and both substitution sweeps run one Legendre row at a time,
  batched over the subsystems active at that row, so the kernels launched per
  application scale with the longest chain. The truncated rows are an uncoupled
  `(1 + floor) I` and are applied as such. The map is unchanged: the f-block
  inverse agrees with the reusable Schur-LU route to rounding, GCROT iteration
  counts are identical, and the band guard is left as it was, conservative for
  ramped decks (`tests/test_coarse_ragged_chains.py`).

- Preserve independent-device batch sharding through JIT and gradients, including
  uneven batches (#179). Each complete system still resides on one device.
- Reuse generated SOLVAX Schur factors within supported PAS solve executions for
  multiple right-hand sides, transpose solves and refinement, subject to memory
  policy (#188). Persistent reuse across changed operators is still future work.
- Require SOLVAX 0.21.0, whose bordered Krylov preconditioner applies the coarse
  inverse once per iteration instead of twice, matching the release used for
  recent DKX timings.

### Benchmarking and development

- Supervise benchmark process groups and cancellation; reject invalid reference
  outputs and stale resumed results; retain original-equation checks and evidence
  (#170, #183, #185).
- Preserve requested PETSc options and observed backends; make source inventories
  reproducible; correct sparse reference matrix interpretation (#169, #176, #177).
- Preserve the tested package tree when GitHub merge refs move (#181), and balance
  thirteen coverage shards using measured test durations (#191).
- Adopt one authoritative figure-first plan, a concise workflow-oriented README,
  grouped documentation and citation metadata (#189, #190). Establish decision
  records and an experiment template in Phase 0.
- Revise the plan from the 2026-09-13 independent review (#230). Take
  finite-difference gradient references at `1e-12`, bound the sparse-map `pas`
  check at `1e-7`, and regenerate `.test_durations` from one complete run on one
  host so the thirteen coverage shards fit their time cap (#238, #241).

### Research records

- NCSX `(21,37,61,8)` refinement ladder with baseline adjoint estimates, its
  `Nx = 11, 12` speed rungs, and the cause of GCROT iteration growth with
  `Ntheta`: the dropped Fokker-Planck speed coupling (#237, #240, #242).
- SFINCS v3's `1e-12` sparsification threshold explains the 12-19% current gap
  on the HSX-like deck, confirmed from the SFINCS side and reported upstream as
  landreman/sfincs#27 (#243).
- SOLVAX operator couplings in the coarse preconditioner halve its memory but
  run 3x slower, so the dense route stays; explicit-inverse storage is rejected
  as not backward stable (#244).
- The HSX-like deck at resonant `E_*` is resolved in pitch by `Nxi = 120` but
  not in speed for the ion channel; no bootstrap current or ion flux at its two
  points is admitted (#245, #246).

These changes do not certify converged production performance, general persistent
restart reuse, full native Phi1 support or complete equilibrium-boundary optimization.
Historical measurements and their limits remain in the performance documentation.
