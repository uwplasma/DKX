# Reuse economics and one redundant inverse

## Hypothesis

Phase 2, step 3: saved preconditioner setup can outweigh extra Krylov work
beyond the tiny field sweeps. Measure existing interfaces before adding a
refresh policy. Owner: rogeriojorge. This record also follows the application
cost the audit exposed into its generic owner, SOLVAX.

## Admission test

DKX `9e9536d9`, SOLVAX 0.20.0; start from
`tests/reduced_inputs/geometryScheme4_2species_noEr.input.namelist` and set
`(Ntheta, Nzeta, Nxi, Nx) = (15, 17, 24, 6)`, tolerance `1e-10`.
The full-Fokker–Planck, full-trajectory, no-Phi1 W7-X case has two species and
73,444 rectangular unknowns. Fields are `0, 0.2, 0.4, 0.6` in deck units.
The input SHA256 is
`f9b12309ca821471d9676cb2393a7dee4345f43435311020117147c0eb7261f4`.
This is a scaling diagnostic, **not a converged research grid**.

Both arms pass the previous distribution as `x0`. One rebuilds the
preconditioner per point; one holds the first preconditioner. Both pass
`recycle=None`, `restart=30`, `recycle_dim=8`, `max_restarts=12` and explicit
`method="gmres"`, avoiding automatic escalation. This disables transferring
recycle vectors **between solves**, not recycling within GCROT; DKX rejects
`recycle_dim=0`. Recover the complete state, check the original equation at
every point and stop on failure. Bound each process at ten minutes.

Discard the first sequence for timing; alternate arm order over five repeats.
`wall_s` covers field/operator/RHS preparation and synchronized solver work,
including preconditioner construction. It **excludes initial problem preparation
and the subsequent moment/residual audit**, so the table is not end-to-end
certified-sweep performance. Both arms are warm in `x0`, not independent cold
solutions. These omissions must be closed for the step's full admission.

## Result

Local M3 Max, Python 3.11.14, JAX/jaxlib 0.9.2; office A4000 GPU 1,
JAX/jaxlib 0.10.2. Both use float64. Median and min–max below sum the four
solve-path times in each repeated sequence; compare arms within each backend,
not hardware with differing software environments.

| Backend | Rebuild, seconds | Fixed, seconds | Iterations, rebuild → fixed |
| --- | ---: | ---: | ---: |
| CPU | 18.31 (17.39–20.97) | 14.64 (14.12–16.18) | 177 → 191 |
| GPU 1 | 25.53 (24.74–26.14) | 22.71 (21.04–23.35) | 181 → 195 |

All 48 solves in each campaign satisfy the original `1e-10` tolerance.
The host was shared; GPU process sampling at 0.5-second intervals found no
competing compute PID during the GPU 1 campaign. This does not establish the
plan's fully idle publication protocol. GPU live peak was 2,164,308,992 bytes;
allocator-pool peak was 4,292,870,144 bytes, a different metric.

An exploratory GPU 0 run had inconsistent times and competing processes;
its timings are excluded. The guarded GPU 0 attempt stopped before solving.
A separate profiled run recorded eight `jit(while)` compilations/cache loads
of 0.32–0.43 seconds, with persistent-cache hits. First invocation is therefore
not synonymous with fresh compilation. The Chrome export covers only part of
the run (1,000,049 events); it cannot support full kernel attribution.
A smaller follow-up disables Python-call tracing but has **not run**.
The complete XPlane is retained outside Git.

### Implementation found by the audit

SOLVAX forms `W = a_inv(B)` during setup, discards it, and applies `a_inv`
twice per residual. For its documented linear inverse,
`a_inv(r_x - B y) = a_inv(r_x) - W y`. Retaining W in place of B removes
one inverse call. The retained array's shape is unchanged; mixed dtypes can
change bytes. This is the same [bordered factorization](https://doi.org/10.1017/S0962492904000212),
with different floating-point evaluation order, not a new physics model.

[SOLVAX #101](https://github.com/uwplasma/SOLVAX/pull/101), source
`bd52aea19b1e6545a9da01fcd683ddbc543a0a46`, implements the rearrangement.
Its suite had 749 passes and six optional-backend skips; 31 operator tests
include inverse-call count, nonsymmetric real/complex references, nonzero
border blocks and derivatives through setup. Ruff and Sphinx `-W` passed.
Sixteen DKX GPU solves completed with maximum original residual `9.334e-11`;
each reported species flux/flow agrees with the baseline within `4e-9` relative.
The timing campaign stopped on a competing GPU PID and is **not admitted**.
The original broader integration run was interrupted and is not counted.

### Continuation: completed integration and measured application cost

A fresh single-worker local run passed **234 DKX integration tests** in 414.88 s
(solve, Phi1, ambipolar, coarse constraints and full-FP/Phi1 profile derivatives).
Pytest reported an unraisable JAX garbage-collection KeyboardInterrupt warning
at teardown; the run returned zero with a complete passing summary.
SOLVAX #101's CI, including combined coverage, passed.

Fresh CPU baseline/candidate campaigns each completed 48 accepted solves. Median
four-field times **including moment/original-equation audits** were 16.41 → 11.75 s
with rebuilt factors and 13.25 → 8.56 s with fixed factors. Largest per-species
flux/flow change was 1.04e-10 relative. Initial problem preparation is reported
separately; these remain shared-host, cache-warm scaling diagnostics, not complete
root/optimization or publication benchmarks. Baseline and candidate processes
ran sequentially; only reuse-arm order alternated within each process.

The candidate GPU 1 campaign completed all 48 solves, with no foreign compute
PID sampled: audited medians 13.86 s rebuilt and 11.72 s fixed. Its solve-path
medians were 13.53 and 11.39 s. A fresh baseline stopped after 20 accepted solves
when another compute job started; all its times are excluded. Against the earlier
complete GPU baseline, the matched **solve-path** medians were 25.53 → 13.53 s
and 22.71 → 11.39 s, respectively; these are separate campaigns, not interleaved
A/B trials. Candidate live peak was 2,166,612,992 bytes; pool peak remained
4,292,870,144 bytes. No memory-saving claim follows. A shorter GPU 0 profile
also stopped on contention, before any solve; full kernel attribution remains open.

Bounded recovery is now implemented as an opt-in host control, rather than an
automatic economic refresh heuristic: explicit GMRES, a smaller reused-factor
restart budget, one cold retry with the full budget on failed original-equation
or finite-current/flux admission, then refusal. Failed state never enters
continuation; the final root remains independently cold. Runtime/resource
exceptions propagate. The final ambipolar regression run passed 124 tests,
including bounded failure recovery and full-FP agreement with independent cold
roots; 27 documentation/example/size contracts and Sphinx `-W` also passed. See `find_ambipolar_er` and `docs/usage.rst`. Full step-3
admission still requires representative root costs, histories and grid errors.
The associated tutorial repair uses quasineutral full-FP analytic W7-X instead
of an axisymmetric ambipolar example. The JIT-compiled geometry tutorial lowers
its analytic objective by 63.4% in five steps, with maximum original residual
1.44e-13 and three-step central-difference disagreement 2.72e-9. Neither is a
converged research-grid or VMEX-boundary optimization result.

A separate 904-unknown full-FP root diagnostic on implementation `e98506e9`
completed twelve roots (one initial pair, five alternating repetitions). Whole
root-call medians were 2.062 s fully cold versus 1.551 s bounded reuse, including
seven admitted current evaluations, independent final acceptance and slope
classification; initial prepare was 0.164 s separately. Both modes retained the
ion root near Er=-0.356279 with current about 1.46e-11 against current_tol=1e-10.
No retry was needed in this benign sequence. This tiny-grid result does not
qualify large-root economics, an adversarial recovery cost or discretization.

On the 73,444-unknown case, a 300-second-capped CPU pilot completed four
whole-root calls with identical accepted fields/fluxes. The initial/repeated
cold calls took 28.55/26.89 s; bounded reuse took 33.14/31.67 s and required
four cold retries in each call. There are not five repetitions here: this is
a negative pilot, not an admitted speed factor. The Er≈-3.42701 root was
classified unstable with current -1.80e-11, so it is not an optimization-qualified
branch either. **Do not promote the two-cycle budget or automatic refresh**:
its retry cost loses on this wider field sequence. The GPU root follow-up
also stopped on a foreign PID before solving. Keep the successful local
bounded-recovery evidence separate from failed GPU and performance admission.

## Decision and handoff

Keep explicit preconditioner reuse, but do not promote a refresh default from
these measurements. Use #101's qualified application cost and complete step 3's root,
single-species, cold-equivalence,
profile/history and bounded stale-recovery admission. Charge audits, pilots and
retries in end-to-end timing. Preserve Phase 1's separate grid/observable-error
requirements. No release or phase completion is claimed.

Source-bound inputs, drivers, logs and integrity manifests are retained privately.
Checksums establish integrity, not scientific validity. Recheck available host
memory and GPU occupancy before any follow-up; shared workloads can change
mid-run. Keep large traces outside Git and preserve failed attempts.

### Merged-source profile and dense application follow-up

DKX `20542772` and SOLVAX `9357191` passed 28 focused merged-source CPU
checks, including full recovery, Phi1, direct transpose and derivatives. The
NCSX `(21,37,61,8)` baseline then completed a bounded office CPU profile in
187.27 s with 28.03 GiB sampled peak RSS, 53 iterations and original residual
8.54e-11. Signed moments match the archived reference. Operator construction
and dense band extraction took 0.91/9.15 s; fused band assembly/factorization
57.48 s; projected-border setup dispatch 12.20/13.40 s; the synchronized
first preconditioner probe 8.83 s; GCROT compilation plus execution 70.32 s.
Nested timings are not additive, and synchronization/logging affect runtime.
This qualifies a baseline state, not grid error bars or a speed ranking.

The dense application now shares a module-level JIT helper with factors passed
as runtime arguments. It removes eager per-block dispatch and reuses helper
compilation for matching shapes/dtypes, following [JAX caching guidance](https://docs.jax.dev/en/latest/jit-compilation.html#jit-and-caching). The physical equation, factorization,
SOLVAX recurrence and transpose semantics are unchanged. It also replaces the
speed-triangle path's duplicated inverse application. Production code shrinks
by six lines, with no new source file. Direct linear-transpose, real/complex
setup/RHS derivatives, batching and changed-factor checks passed on JAX 0.9.2.

A two-subsystem, 61-block, 8-by-8 diagnostic gave five-pair median warm
application times around 131 ms eagerly and 0.42–0.53 ms with the helper, across
real/complex forward/transposed solves; original residuals were below 1e-12.
Both arms were warmed, factors excluded, and order alternated. This measures
application dispatch outside an enclosing JIT, not whole GCROT or optimization
speed. The full NCSX candidate comparison remains pending its 43.7 GiB host
availability gate; no memory-saving or GPU performance claim follows. The
separate GPU correctness attempt stopped on a foreign process after two cases;
it does not qualify the complete GPU selection.

The complete-solve helper ablation on reduced NCSX `(11,13,25,6)` completed
an initial pair plus five alternating pairs on local CPU. Rebuilding operator,
factors and state every call, audited medians were 2.252 s eager versus 1.986 s
jitted (about 12% lower). All 12 original residuals were at most 7.684e-11;
41 iterations, all 45 moment fields and complete states agreed (largest state
relative difference 2.14e-15). Compilation caches persisted; this compares the
helper with its eager body in one candidate checkout, not independent releases
or optimization runs. Campaign peak RSS was 2.24 GiB, not a per-arm memory
comparison. The research-grid and GPU qualifications remain separate.
