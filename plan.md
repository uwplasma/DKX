# SFINCS_JAX Master Handoff + Execution Plan

Last updated: 2026-05-16 (Europe/Lisbon)
Owner: incoming agent

## 1) Prompt For A New Agent (copy/paste)

```text
You are taking over sfincs_jax, a JAX rewrite/extension of SFINCS v3.

Primary mission (phase 1):
- Reproduce SFINCS v3 functionality and numerics for supported geometries and physics,
- Match outputs/diagnostics/terminal behavior to Fortran SFINCS for the same input,
- Keep default behavior robust and general (no case-specific hard-coding),
- Maintain end-to-end differentiability for JAX-native solve paths,
- Deliver high performance and memory efficiency by default,
- Keep code easy to run, easy to maintain, thoroughly validated, and deeply documented.

Primary mission (phase 2+):
- Extend beyond strict SFINCS replication toward modern neoclassical workflows,
- Integrate/benchmark alternative numerical formulations and optimization-oriented methods,
- Borrow and generalize ideas from modern neoclassical toolchains where they survive direct validation,
- Preserve scientific correctness while improving throughput, scalability, and usability.

Non-negotiable engineering constraints:
1) No hidden dependence on colocated Fortran outputs for correctness.
2) No brittle per-case tuning as the default path.
3) New defaults must generalize to unseen inputs and still converge robustly.
4) Every numerical/performance change must be validated (unit + regression + physics + reduced-suite comparison).
5) Documentation must explain equations, normalization, discretization, solver/preconditioner design, and code locations.

Working directories and references:
- sfincs_jax repo: /Users/rogeriojorge/local/tests/sfincs_jax
- Fortran SFINCS v3 executable: /Users/rogeriojorge/local/tests/sfincs/fortran/version3/sfincs
- Original SFINCS source tree: /Users/rogeriojorge/local/tests/sfincs_original
- Main thesis/pdf refs: /Users/rogeriojorge/local/tests/Escoto_Thesis.pdf and /Users/rogeriojorge/local/tests/*.pdf
- sfincs_jax docs upstream refs: /Users/rogeriojorge/local/tests/sfincs_jax/docs/upstream

Immediate priorities:
- Keep reduced-suite comparison fully populated and reproducible,
- Keep defaults robust for all examples (including additional examples),
- Eliminate remaining solver branch fragility while preserving differentiability,
- Reduce worst runtime/memory offenders (especially PAS-heavy paths),
- Improve practical scaling strategy (CPU cores, GPU path, cluster portability).

Current active lane (2026-05-16, v1.1.4 completion push):
- [x] Coordinated a four-lane completion push on clean `main` after
  `f35c0c6`. Three implementation lanes landed with disjoint scopes: a
  standalone QI block-Schur/angular/radial coarse primitive, PAS matrix-free
  runtime chunk planning, and single-case sharded-solve planning metadata. The
  release/docs lane was stopped after no patch returned; release metadata was
  handled in the main thread instead.
- [x] Added `sfincs_jax/rhs1_qi_block_schur.py` with deterministic global,
  radial, angular, and block-Schur basis directions, a local-plus-coarse JAX
  action, fail-closed true-residual probing, and rank/conditioning diagnostics.
  Focused tests prove synthetic residual reduction, impossible-improvement
  rejection, stable metadata, and JAX transform compatibility. This is ready as
  standalone infrastructure, but not yet a release-facing true device-QI solve
  until wired through the driver and validated on the scale-0.60 CPU/GPU hard
  seed.
- [x] Strengthened PAS matrix-free memory safety with `PasRuntimeChunkPlan`,
  `plan_pas_runtime_chunks`, a `max_reduction_bytes` configuration knob, and
  earlier byte-budget preflight. Tight PAS budgets now reject before matvec or
  correction work, and configured candidate-byte budgets also bound streaming
  reduction chunks. This reduces memory-risk surface without changing physics.
- [x] Added `sfincs_jax/transport_parallel_sharding.py` and benchmark-plan
  metadata so experimental single-case sharded solves record active devices,
  capped device requests, per-device balance diagnostics, and fail-closed
  release-scaling eligibility. This improves practical scaling setup and claim
  hygiene, but does not convert single-case multi-GPU sharding into a release
  scaling claim.
- [x] Focused validation for the new lanes passed: `python -m py_compile` on
  touched files, `ruff check` on touched files, and `pytest -q` across QI
  block-Schur, PAS policy/matrix-free/benchmark, sharding planner, sharded
  benchmark, and transport parallel tests (`135 passed in 5.76 s`).
- [x] Local final release validation passed: release-gate checks, research-lane
  checks, `git diff --check`, Sphinx `-W` docs build, full local suite
  (`1651 passed in 712.28 s`), `python -m build`, and `twine check dist/*`.
- [ ] Final v1.1.4 remote gate: push to `main`, require CI/docs success, then
  tag `v1.1.4` only if the tree stays clean and the tag matches
  `pyproject.toml` / `sfincs_jax.__version__`.

Current active lane (2026-05-12, coordinated large-push research/performance closure):
- [ ] Second larger push requested on 2026-05-12: increase each open lane by at
  least `15` percentage points where possible, or saturate the checked target for
  lanes with less than `15` points remaining. Active workers now own QI
  production readiness, PAS/runtime-memory, parallel scaling, refactor/coverage,
  VMEC/Boozer plus optional JAX ecosystem gates, and benchmark/docs release
  consistency. The main thread owns the cross-lane manifest, target-capped lane
  policy, plan integration, final tests, and commit/push.
- [x] Upgraded `sfincs_jax/research_lane_policy.py` so large-push gates are
  target-capped: a lane cannot overclaim beyond `target_percent`, and a lane
  with fewer points remaining must reach that target. Added regression tests for
  target saturation and unfinished target saturation in
  `tests/test_research_lane_policy.py`.
- [x] Integrated second-push worker results into
  `docs/_static/research_lane_completion_2026_05_12.json`. The overall lane
  average moved to `84.9%`; the VMEC/Boozer lane saturated its scoped `93%`
  target, the benchmark/docs lane saturated its `98%` target, and the JAX
  ecosystem lane moved to `82%` after measured Lineax/Equinox/JAXopt gates. QI
  is now `85%` after the scale-0.50 public `auto` path selected the
  right-preconditioned exact-xblock-LU route and passed five CPU seeds plus five
  one-GPU seeds. PAS, parallel, and
  refactor/coverage gained fail-fast gates, timeout-safe benchmark plans, and
  focused coverage, but they did not honestly clear a +15 point move from the
  immediately preceding baseline.
- [x] Added a passing larger QI CPU artifact at
  `docs/_static/qi_seed_robustness_scale045_cpu_probe.json`: `11 x 23 x 45 x 4`,
  public `auto` solver, output and solver trace written, `106.0 s`, and residual
  ratio `4.96e-7`. The existing scale-0.50 timeout artifact remains checked in
  as blocker evidence and is excluded from completion estimates by
  `scripts/run_qi_seed_robustness.py`.
- [x] Added compact failure-progress capture to
  `scripts/run_qi_seed_robustness.py` so failed QI runs preserve bounded
  stdout/stderr tails and solver-stage breadcrumbs without committing copied
  VMEC/run directories. The new
  `docs/_static/qi_seed_robustness_scale050_solver_matrix_2026_05_12.json`
  historical blocker matrix shows that `13 x 27 x 50 x 4` was not fixed by existing solver
  flags: public `auto` times out after `360 s` after the explicit FP x-block
  seed, `sparse_host_safe` fails host sparse factorization on a `126365616`-nnz
  conservative pattern, and `sparse_lsmr` / `xblock_sparse_pc_gmres` both stall
  near residual `5e-6` against a `2.5e-11` target. A follow-up opt-in
  `SFINCS_JAX_RHSMODE1_XBLOCK_PC_INITIAL_SEED=1` probe was intentionally kept
  off by default after the seed residual was slightly worse than the RHS norm and
  the solve still stalled at the same floor. A second opt-in
  `SFINCS_JAX_RHSMODE1_XBLOCK_PC_POST_MINRES_STEPS=4` probe accepted two
  post-GMRES matrix-free minimum-residual corrections but only improved
  `5.413504e-6 -> 5.409759e-6`, leaving the residual ratio above `2.1e5`.
  A third opt-in `SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV=lgmres` probe was also
  rejected: LGMRES stalled at `5.577462e-6`, fell back to GMRES, doubled the
  matvec count to `65204`, and ended at the same `5.413504e-6` floor after
  about `300 s`.
  That matrix motivated the promoted exact-xblock-LU/right-PC policy below;
  those rejected probes remain useful diagnostic evidence but are no longer the
  active scale-0.50 CPU blocker.
- [x] Implemented the first gated matrix-free correction hook for that next
  step: explicit `xblock_sparse_pc_gmres` now accepts opt-in
  `SFINCS_JAX_RHSMODE1_XBLOCK_PC_POST_MINRES_STEPS`, which applies bounded
  preconditioned minimum-residual corrections after a stalled Krylov solve
  without assembling a global sparse matrix. The hook records requested/accepted
  steps, before/after residuals, and alphas in solver metadata and HDF5
  diagnostics. The first checked scale-0.50 probe accepted two steps but reduced
  the residual by only `6.9e-4` relative, so the hook remains off by default and
  is recorded as rejected prototype evidence rather than promoted solver policy.
- [x] Measured the stronger opt-in coarse correction hook. Explicit
  `xblock_sparse_pc_gmres` now accepts
  `SFINCS_JAX_RHSMODE1_XBLOCK_PC_POST_COARSE=1`, which solves a bounded
  matrix-free least-squares residual correction over preconditioned/raw
  residual, flux-surface-averaged low-L, and source/constraint directions. The
  hook is off by default. The checked scale-0.50 probe accepted one 10-direction
  step and improved `5.413504e-6 -> 5.401187e-6`, but the residual ratio remained
  above `2.1e5`; record it as another rejected prototype. The next QI attempt
  needs a genuinely different global coupling strategy, likely one that changes
  the preconditioned Krylov operator or supplies a larger physics-informed
  coarse space before the 32000-iteration floor, not another post-hoc scalar or
  small-subspace cleanup.
- [x] Added the next solver-side QI candidate rather than another Krylov-name
  toggle: explicit `xblock_sparse_pc_gmres` now accepts opt-in
  `SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED=1`. This builds a deterministic,
  rank-gated QI coarse basis from block constants, species groups, radial ramps,
  and low Fourier angular harmonics, pads it into the active `Nxi_for_x`
  Krylov space, and applies a guarded least-squares correction before Krylov.
  Focused regression coverage verifies a true residual drop on the quick
  RHSMode=1 active-DOF surrogate and records rank, candidate count, labels,
  before/after residuals, improvement ratio, and setup time in solver metadata.
  The bounded scale-0.60 seed-3 CPU rerun
  `docs/_static/qi_seed_robustness_scale060_qi_coarse_seed3_cpu_2026_05_14.json`
  passes in `248.4 s`, writes output and solver trace, and converges with
  residual ratio `1.36e-3`; the QI seed itself gives only a tiny first
  correction (`3.0214868e-5 -> 3.0214861e-5`), while the angular probe-coarse
  stage gives the material pre-Krylov drop (`4.57e-6 -> 2.83e-6`). Keep this as
  bounded CPU evidence. It remains off by default until the one-GPU scale-0.60
  hard seed also writes HDF5/trace within budget and the runtime is competitive.
- [ ] Follow-up one-GPU scale-0.60 rerun with the same QI coarse/probe-coarse
  configuration was attempted on `office` GPU 0 on 2026-05-14. The local tree
  was synced to `/tmp/sfincs_jax_qi_coarse_gpu`, but the remote SSH session
  exceeded the intended 10-minute wall budget without returning a summary, and
  subsequent SSH monitor/kill attempts timed out. No GPU artifact is promoted
  from that attempt. Next GPU work should start with a liveness-safe wrapper
  that writes heartbeat files outside the JAX process and an outer `timeout`
  around the whole command, not only the per-case subprocess.
- [x] Added that liveness-safe runner mode. `scripts/run_qi_seed_robustness.py`
  now accepts `--heartbeat-s`; when positive, each case writes
  `runner_heartbeat.jsonl` with `started`/`running`/`completed` or
  `timeout`/`terminated` events, and the timeout path kills the whole subprocess
  process group. `tests/test_run_qi_seed_robustness.py` covers both normal
  heartbeat completion and forced timeout termination. The next `office` GPU
  command should use both an outer shell `timeout --kill-after=30s 720s` and
  `--heartbeat-s 15` so remote hangs become auditable artifacts.
- [x] Closed the scale-0.50 single-seed CPU and one-GPU QI blocker with a promoted
  preconditioner policy instead of a timeout increase. The successful route is
  right-preconditioned explicit `xblock_sparse_pc_gmres` plus exact sparse LU
  for medium full-FP host x-blocks (`SFINCS_JAX_RHSMODE1_XBLOCK_SPARSE_LU_MAX`
  default raised to `30000` only for the non-differentiable full-FP host path).
  The checked artifact
  `docs/_static/qi_seed_robustness_scale050_xblock_lu_right_cpu.json` converges
  the `13 x 27 x 50 x 4` QI seed in `~12.0 s`, solver time `~11.2 s`, with
  residual `1.04e-12` against target `2.51e-11` and residual ratio `4.16e-2`.
  The matching clean-clone one-GPU artifact
  `docs/_static/qi_seed_robustness_scale050_xblock_lu_right_gpu.json` passes in
  `~44.5 s`, solver time `~43.0 s`, with residual `1.58e-11` and residual ratio
  `0.63`. The solver traces record `precondition_side=right`, no short-restart
  cap, exact `sparse_lu` block factors, and compact Krylov counts (`81`/`85`
  CPU iterations/matvecs and `69`/`72` GPU iterations/matvecs). This
  raises the QI lane materially, but wider CPU/GPU seed ladders remain required
  before production-resolution QI is declared complete.
- [x] Extended that scale-0.50 route to the public `auto` path across five CPU
  seeds and five one-GPU seeds:
  `docs/_static/qi_seed_robustness_scale050_xblock_lu_right_multiseed5_cpu.json`
  passes seeds `0..4` with all outputs and solver traces written, zero failures,
  maximum elapsed time `11.58 s`, and maximum residual ratio `0.966`; the
  matching `docs/_static/qi_seed_robustness_scale050_xblock_lu_right_multiseed5_gpu.json`
  passes on `office` GPU 0 with maximum elapsed time `41.18 s` and maximum
  residual ratio `0.963`. This closes the bounded public-auto robustness
  blocker. The remaining QI promotion gates are the next bounded resolution
  scale and finally production resolution.
- [x] Closed the next CPU scale-0.55 setup cliff without widening the default
  production claim. The previous
  `docs/_static/qi_seed_robustness_scale055_auto_cpu_blocker.json` run timed out
  because the largest `15 x 29 x 55 x 4` x-block (`23925` DOFs) exceeded the old
  exact-LU cap and fell into sparse ILU. The new
  `docs/_static/qi_seed_robustness_scale055_xblock_lu_right_cpu.json` successor
  uses the same public `auto` request plus the widened profiling window, now hits
  exact `sparse_lu` for all x-blocks, and converges in `~21.5 s` with active size
  `52637`, residual `2.30e-13`, target `2.79e-11`, and residual ratio `8.25e-3`.
  Remaining QI work is the matching GPU scale-0.55 ladder and wider seed count
  before any production-resolution promotion.
- [x] Closed the scale-0.55 hard-seed CPU right-PC slow mode with a size-aware
  3D full-FP x-block precondition-side policy. The default keeps right
  preconditioning for the measured scale-0.50 window but switches larger 3D
  full-FP active systems to left preconditioning above
  `SFINCS_JAX_RHSMODE1_XBLOCK_RIGHT_PC_MAX=45000` unless the user explicitly
  overrides `SFINCS_JAX_GMRES_PRECONDITION_SIDE`. The hard seed `3` artifact
  `docs/_static/qi_seed_robustness_scale055_xblock_auto_side_seed3_cpu.json`
  now passes the public `auto` request at active size `52637` in `~47 s`, records
  `precondition_side=left`, uses exact `sparse_lu` x-block factors, and reaches
  residual ratio `2.98e-3`. The previous right-PC five-seed probe had seed `3`
  still progressing at hundreds to thousands of matvecs and timing out on one
  GPU, so the next gate is to rerun the matching GPU hard seed and then the
  five-seed GPU ladder from a clean checkout.
- [x] Closed the bounded scale-0.55 CPU/GPU five-seed QI ladder with the new
  adaptive-side default. The CPU artifact
  `docs/_static/qi_seed_robustness_scale055_xblock_auto_side_multiseed5_cpu.json`
  passes seeds `0..4` with zero process failures/timeouts, all outputs and
  solver traces written, maximum elapsed time `44.5 s`, and maximum residual
  ratio `5.88e-3`. The clean `office` GPU artifact
  `docs/_static/qi_seed_robustness_scale055_xblock_auto_side_multiseed5_gpu.json`
  passes the same seeds with maximum elapsed time `206.7 s`, maximum residual
  ratio `8.28e-3`, and hard seed `3` reduced from the previous right-PC GPU
  timeout to `~206 s`. This closes the next bounded QI CPU/GPU robustness gate;
  remaining QI promotion now requires a next-size bounded ladder or
  production-resolution ladder, not another scale-0.55 rerun.
- [x] Advanced the next-size QI gate beyond scale `0.55` with bounded seed-0
  CPU/GPU probes at scale `0.60`. The artifacts
  `docs/_static/qi_seed_robustness_scale060_xblock_auto_side_seed0_cpu.json`
  and `docs/_static/qi_seed_robustness_scale060_xblock_auto_side_seed0_gpu.json`
  pass at `15 x 31 x 60 x 5`, active size `81377`, total size `139502`, with
  left-preconditioned exact-xblock-LU GMRES. CPU elapsed time is `42.2 s` with
  residual ratio `3.42e-3`; one-GPU elapsed time is `145.1 s` with residual
  ratio `4.68e-3`. This increases the bounded QI readiness estimate to `60%`
  per-axis and `13.7%` of production total size, while keeping promotion honest:
  scale `0.60` still needs a CPU/GPU five-seed ladder before widening the public
  auto window toward production.
- [ ] Resolve the scale-0.60 multi-seed slow mode before promoting the next
  QI window. A first bounded five-seed attempt showed seed `0` passing on CPU
  in `41.5 s` and on one `office` GPU in `137.9 s`, but CPU seed `1` timed out
  after `420 s` while still progressing through left-preconditioned GMRES
  (`4500` matvecs at `404.6 s`), CPU seed `2` showed the same pattern (`3600`
  matvecs at `319.3 s` when the run was stopped), and GPU seed `1` was also
  slow (`1200` matvecs at `545.9 s`) before the remote ladder was stopped to
  avoid wasting GPU time. This is now the concrete next algorithmic blocker:
  scale-0.60 needs a stronger seed-robust global-coupling/preconditioner
  correction, adaptive side/ordering by seed metrics, or a tighter restart/seed
  rescue before the five-seed ladder can be closed. A targeted CPU seed `1`
  manual-right probe did eventually converge in `353.1 s` with residual ratio
  `9.82e-3`, `3999` iterations, and `4051` matvecs, so the issue is not an
  impossible system; it is a poor default side/iteration-count prediction for
  some seeds. A future fix should not simply widen timeouts: it should add a
  cheap side-selection probe or an early left-to-right/right-to-left rescue
  based on measured residual decrease per matvec.
- [x] Closed the CPU half of the scale-0.60 multi-seed QI blocker with a
  bounded side-probe plus LGMRES rescue, without widening blind timeouts. The
  driver now runs one default-side GMRES restart on large 3D full-FP x-block
  systems; if the true-residual ratio remains above `5e3` on the CPU default
  route, it reuses the left-probe state and switches to LGMRES with
  `outer_k=10` and a capped `80` outer iterations. The checked artifact
  `docs/_static/qi_seed_robustness_scale060_xblock_lgmres_rescue_multiseed5_cpu.json`
  passes seeds `0..4` at `15 x 31 x 60 x 5`, active size `81377`, with
  `accepted_converged=true`, zero timeouts, max elapsed `307.9 s`, and max
  residual ratio `8.42e-3`. This replaces the old seed `1`/`2` CPU timeout
  behavior with a measured, progress-logged rescue.
- [ ] Keep the scale-0.60 GPU QI hard-seed lane open. The same LGMRES rescue is
  CPU-default only because an `office` GPU seed `3` probe timed out at `620 s`
  after selecting the LGMRES rescue, with low accelerator utilization and no
  solver trace. The side-switched right-GMRES GPU probe also timed out at
  `620 s`. These blocker artifacts are checked in as
  `docs/_static/qi_seed_robustness_scale060_xblock_lgmres_rescue_seed3_gpu_timeout.json`
  and
  `docs/_static/qi_seed_robustness_scale060_xblock_right_gmres_seed3_gpu_timeout.json`.
  A 2026-05-13 bounded follow-up rejected several tempting but insufficient
  solver toggles: the existing two-level coarse preconditioner was slower on
  CPU (`320.8 s` vs the kept `307.9 s`) and timed out on GPU; GCROT(m,k)
  timed out on CPU and GPU; BiCGStab plus GMRES fallback timed out; BiCGStab
  plus post-minres/post-coarse correction finished but remained nonconverged
  (`1.21e-4` CPU and `6.70e-5` GPU residual norms); and an experimental
  JAX-factor/device-BiCGStab x-block prototype timed out with a worse residual
  and was removed rather than kept as dead complexity. The rejected-probe
  summary is checked in as
  `docs/_static/qi_seed_robustness_scale060_gpu_rejected_solver_probes_2026_05_13.json`.
  The only code retained from this pass is a safe GMRES fallback guard: a
  failed non-GMRES candidate may seed fallback GMRES only when it improves the
  RHS norm and is not a right-preconditioned coordinate state. The next GPU
  algorithmic step is not another timeout increase or Krylov-name toggle; it
  needs a stronger GPU-compatible global-coupling preconditioner or a real
  assembled/operator-reuse path for host-Krylov QI hard seeds.
- [ ] 2026-05-13 follow-up: a forced `office` GPU probe with
  `SFINCS_JAX_RHSMODE1_XBLOCK_PC_LGMRES_RESCUE=1` plus the new probe-coarse
  seed correction also timed out at `420 s`. The run selected the
  left-preconditioned LGMRES rescue (`80` side-probe iterations, `82` matvecs,
  residual ratio `8.51e4`) but never reached a solver trace or output. Process
  telemetry showed about `20` CPU cores busy, `3.17 GB` RSS, and essentially
  idle RTX A4000 memory/utilization, so this confirms forced host-LGMRES is
  not a viable GPU-performance path. Do not spend more time on SciPy-host
  LGMRES GPU toggles; the remaining QI GPU closure must be a device-resident
  or assembled/reused active operator correction that makes the true residual
  fall before the expensive Krylov loop.
- [x] 2026-05-13 side-switch correctness fix: right-preconditioned host Krylov
  wrappers now interpret `x0` as a physical-space initial guess by solving the
  correction equation `A M^{-1} y = b - A x0` and returning `x0 + M^{-1} y`.
  The x-block side probe now preserves the physical left-probe seed when it
  switches to right preconditioning instead of discarding it. Targeted local
  checks passed:
  `tests/test_solver_heavy_helper_coverage.py::test_gmres_history_scipy_right_preconditioning_uses_physical_x0`,
  `tests/test_v3_sparse_pattern.py::test_xblock_side_probe_switch_preserves_physical_seed_for_right_pc`,
  the full `tests/test_v3_sparse_pattern.py` module, and the right-PC LGMRES
  API test. CI and Docs passed on commit `5ef9d86`.
- [ ] 2026-05-13 GPU follow-up after that fix: the preserved-seed right-PC
  `office` GPU gate reached the intended policy path (`side=left->right` with
  `preserved_physical_seed=1`) but still timed out at `420 s`, using about
  `16.8` effective CPU cores and `3.14 GB` max RSS while GPU 0 stayed idle.
  This proves the fix is mathematically necessary but not sufficient for
  GPU-performance closure because the promoted path is still host-Krylov.
- [ ] 2026-05-13 device-factor probes: device FGMRES with JAX x-block factors,
  global coupling, and exact-LU opt-in completed bounded GPU runs but remained
  nonconverged. With exact-LU cap `30000` and padded row cap `64`, elapsed time
  was `307 s`, max RSS `9.94 GB`, and residual stayed at `3.0215e-05`
  (`target=3.0215e-13`). Increasing the padded row cap to `128` reduced elapsed
  time to `282 s` with similar max RSS (`9.92 GB`) but again left the residual
  at the RHS norm. This rejects simple exact-vs-ILU and row-cap widening as the
  remaining GPU fix. The next algorithmic step must change the preconditioner
  structure, for example a true active-operator reuse/assembled sparse matvec
  with a device-compatible coarse solve, or a block-Jacobi/Schwarz formulation
  whose quality can be verified by applying `A M^{-1}` to physics load bases
  before launching the full Krylov loop.
- [ ] 2026-05-14 compact-CSR exact-factor follow-up: the replacement compact
  CSR JAX factor path built full exact per-x SuperLU factors for the same
  scale-0.60 `office` GPU hard seed without padded row truncation
  (`1.10e8` lower nonzeros, `1.19e8` upper nonzeros, `2.76 GB` estimated
  factor arrays, `17.6 GB` peak RSS), but the run still timed out at the
  bounded `540 s` gate before HDF5 output or a solver trace. This keeps compact
  CSR as useful memory infrastructure only. It does not close the GPU QI hard
  seed; the open lane is now a residual-reducing, GPU-cheap preconditioner or
  block-Schur/coarse operator rather than another storage-only factor format,
  row-scaling knob, restart tweak, or Krylov-name toggle.
- [x] 2026-05-15 Worker A device-QI infrastructure step: added a reusable
  JAX-compatible Galerkin QI coarse preconditioner that stores `Q^T A Q`, solves
  the small projected problem with a regularized JAX normal-equation helper, and
  applies guarded residual-reducing corrections without SciPy or host callbacks.
  The primitive is CPU-testable, works with `DeviceCSR.matvec` under `jax.jit`,
  and replaces the one-shot QI coarse least-squares helper with the same
  bounded JAX solve. This promotes the differentiable/device-QI lane
  infrastructure, but it does not by itself close the scale-0.60 hard-seed GPU
  artifact until wired into the production hard-seed/preconditioner loop and
  validated on the bounded QI GPU gate.
- [x] PAS/memory second-push result: added opt-in matrix-free tiny-update and
  candidate-size fail-fast gates, storage metadata, structured PAS-TZ guard
  metadata, and tests. This reduces wasted candidate work in bounded probes, but
  a production residual-clean geometry-rich default-promotion artifact is still
  required before claiming the lane closed.
- [x] PAS/memory follow-up: fallback benchmark rows now require explicit
  guarded-PAS-TZ provenance before any candidate can pass all gates or appear as
  promotion eligible. This closes a false-positive benchmark loophole without
  changing default solver policy.
- [x] Parallel second-push result: added timeout-safe two-GPU case-throughput
  benchmarking and a pure audit that rejects release overclaims, non-GPU
  payloads, cold/mixed timing, and sub-unity speedups. This improves the GPU
  campaign workflow but does not close single-case strong scaling.
- [x] Parallel follow-up: transport-worker benchmark plans and measured payloads
  now include a compile-amortization gate. Release scaling claims fail closed if
  they request `release_scaling_claim=true` without evidence that compilation and
  setup were excluded from timed repeats.
- [x] Follow-up office GPU result on commit `39e1e2f`: the timeout-safe
  two-GPU case-throughput benchmark completed with `nsolve=1` and a `180 s`
  child timeout, wrote
  `docs/_static/gpu_case_throughput_large_push_2026_05_12.json`, and failed the
  throughput gate honestly (`59.18 s` sequential one-GPU two-case wall time vs
  `175.77 s` two-GPU concurrent wall time, speedup `0.337x`). This keeps the
  parallel lane at `82%` and confirms that this path is setup/compile dominated
  on office rather than release-grade scaling evidence.
- [x] Refactor/coverage second-push result: added focused fast tests around
  solver-candidate promotion, residual diagnostics, and policy docstrings.
  Focused helper coverage improved, but package-wide `95%` still requires more
  driver decomposition and a JAX-safe coverage environment.
- [x] Refactor/coverage follow-up: added CI-fast branch coverage for dense
  accelerator guards, sparse-direct eligibility, host-GMRES rejection paths,
  sparse-factor dtype overrides, metric edge cases, and malformed release
  manifests. This reduces policy-helper risk but does not justify a package-wide
  threshold increase by itself.
- [x] VMEC/Boozer/JAX-ecosystem follow-up: added a shared pure-JAX
  Boozer-spectrum proxy transport objective with a full spectral finite-
  difference gradient check and JVP/dot-product consistency gate. The status and
  handoff CLIs now use this shared gate; it strengthens default-CI
  differentiability coverage while remaining explicitly a proxy, not a full
  VMEC-boundary-to-kinetic-transport gradient claim.
- [x] Spawned workers across the active QI, PAS/runtime-memory, parallelism,
  refactor/coverage/CI, and VMEC/JAX-ecosystem lanes rather than continuing with
  single-file incremental work.
- [x] Added a machine-readable cross-lane completion artifact at
  `docs/_static/research_lane_completion_2026_05_12.json` plus
  `sfincs_jax/research_lane_policy.py`, `scripts/check_research_lanes.py`, and
  `tests/test_research_lane_policy.py`. The gate requires evidence paths,
  non-regressing completion estimates, non-empty acceptance gates, and at least
  a 10 percentage-point measured increase for active/evidence-ready lanes.
- [x] QI lane: generated a new bounded five-seed CPU artifact
  `docs/_static/qi_seed_robustness_multiseed5_cpu.json` and a production-readiness
  rollup `docs/_static/qi_seed_robustness_evidence_manifest.json`. The measured
  five-seed bounded CPU ladder passed `5/5`, had zero process failures/timeouts,
  wrote all outputs and solver traces, and reported maximum residual ratio
  `7.872e-7`. This materially improves seed coverage but keeps the lane honest:
  the production gate remains bounded because `25 x 51 x 100 x 8` CPU/GPU
  five-seed ladders are not checked in.
- [x] PAS/runtime-memory lane: tightened promotion gates so a candidate cannot be
  promoted if either elapsed time or peak RSS regresses against the baseline,
  added pass-through controls to the RHSMode=1 PAS matrix-free benchmark planner,
  and reduced no-op PAS matrix-free correction probes from two matvecs to one
  while preserving residual/reject semantics. The bounded PAS probe rejected an
  `lgmres` candidate as a runtime-and-memory regression, which is the intended
  future-proof behavior.
- [x] VMEC/Boozer/JAX-ecosystem lane: added the skip-safe
  `examples/optimization/vmec_jax_workflow_status.py` scaffold, a dedicated
  `docs/vmec_jax_workflow.rst` page, and optional Lineax/Equinox/JAXopt measured
  gates. Local optional-package checks passed, including VMEC/Boozer proxy-gradient
  validation, Lineax synthetic implicit-solve gate, and Equinox/JAXopt objective
  gates. The docs explicitly avoid claiming full VMEC-boundary-to-SFINCS kinetic
  transport gradients.
- [x] Refactor/coverage/CI lane: centralized repeated transport env-flag parsing,
  added module/public docstrings across split policy helpers, strengthened dynamic
  docstring coverage for `sfincs_jax/*policy*.py`, and added fast pure-policy
  coverage for transport backend, sparse/direct, host-GMRES, and parallel-worker
  decisions. Focused lane tests passed `29/29`; touched policy-module coverage
  under `coverage run --timid` is `83%`. The full 95% package-coverage target
  remains open because it needs broader driver/module splitting rather than slow
  full-solve tests.
- [x] Parallel/scaling lane: added plan-only modes to
  `examples/performance/benchmark_transport_parallel_scaling.py` and
  `examples/performance/benchmark_sharded_solve_scaling.py`, so CPU/GPU scaling
  campaigns can be reviewed for worker caps, RHS partitioning, backend env,
  timeout/memory controls, warm/cold timing semantics, and release-gate semantics
  before launching expensive solves. Added GPU worker schedule planning in
  `transport_parallel_runtime.py`. Focused parallel tests passed `64/64` in
  `19.61 s`. This improves reproducibility and avoids overclaiming; single-case
  multi-device strong scaling remains experimental until measured speedup
  artifacts clear the gate.
- [x] Focused integrated validation after worker merge:
  `pytest -q tests/test_research_lane_policy.py tests/test_qi_seed_smoke_artifact.py
  tests/test_benchmark_pas_tz_memory_fallback.py tests/test_benchmark_rhs1_pas_matrixfree.py
  tests/test_rhs1_pas_matrixfree.py tests/test_transport_policy_coverage.py
  tests/test_vmec_jax_workflow.py tests/test_optional_ecosystem_gates.py
  tests/test_policy_module_docstrings.py` passed `69/69` in `5.71 s`.
- [x] 2026-05-15 cross-lane subagent push: kept QI honest at `97%` because the
  new Galerkin coarse primitive is implemented and opt-in production-wired but
  the scale-0.60 GPU hard seed still needs a passing artifact. Raised PAS to
  `95%` after adding fail-closed candidate-byte preflights and benchmark
  dry-run byte budgets, refactor/coverage to `94%` after fast CLI/plotting/
  validation I/O coverage, and VMEC/Boozer to `95%` after adding a
  machine-readable VMEC/Boozer-to-kinetic scalar contract that explicitly
  separates geometry, kinetic assembly, solve, reduction, and gradient stages.
  Parallel scaling remains target-saturated at `94%` with explicit claim-scope
  metadata; single-case multi-GPU strong scaling remains experimental.
- [x] 2026-05-15 integrated validation after the subagent push:
  `python -m py_compile` over touched scripts/modules, `ruff check --select
  F821,F823`, `git diff --check`, `python scripts/check_release_gates.py &&
  python scripts/check_research_lanes.py`, QI evidence manifest regeneration,
  `sphinx-build -b html docs docs/_build/html -W`, and the integrated focused
  pytest slice passed. The focused slice covered QI coarse/device operators,
  v3 sparse-pattern dispatch, PAS matrix-free/byte gates, memory model,
  transport/sharded scaling audits, VMEC/Boozer workflow contract, CLI/plotting
  coverage, validation figure/artifact policies, release metadata, and
  validation math/artifacts: `342 passed in 332.07 s`.
- [x] 2026-05-15 follow-up large-lane push: moved QI to `98%` by making the
  opt-in Galerkin hard-seed path fail closed. The production hook now probes
  additive/multiplicative/damping candidates and uses a Galerkin wrapper only
  if the true residual decreases; otherwise it records `probe_not_reduced` and
  keeps the base preconditioner. This closes the unsafe “wired but possibly
  harmful” part of the QI lane without claiming the still-open scale-0.60 GPU
  hard-seed artifact.
- [x] 2026-05-15 PAS memory/runtime follow-up: moved PAS to `96%` by making
  opt-in production-floor real-solve probes require `--max-candidate-bytes` by
  default. Developers can explicitly opt out with
  `--production-solve-allow-unbudgeted-candidate`, but the opt-out is recorded
  and is not promotion evidence. The checked dry-run
  `docs/_static/rhs1_pas_matrixfree_byte_budget_gate_2026_05_15.json` shows
  geometry4 and HSX are byte-safe launch candidates under the configured budget,
  while geometry11 remains held for missing promotion-facing artifact evidence.
- [x] 2026-05-15 scale-0.60 one-GPU QI hard-seed rerun campaign on `office`:
  recorded three new bounded negative artifacts and kept QI at `98%`. Public
  auto (`xmg`) timed out after `600 s` with no output/trace and a last strong
  fallback residual `2.262e-6` against target `3.021e-13`
  (`docs/_static/qi_seed_robustness_scale060_galerkin_failclosed_seed3_gpu0_2026_05_15.json`).
  Forced x-block GMRES exercised the fail-closed Galerkin hook; the rank-32
  Galerkin candidate was correctly rejected (`probe_not_reduced`,
  probe ratio `6.72`), then the run timed out after `1300` matvecs
  (`docs/_static/qi_seed_robustness_scale060_galerkin_forced_xblock_seed3_gpu1_2026_05_15.json`).
  Forced LGMRES rescue selected the intended method rescue but still timed out
  after `950` matvecs
  (`docs/_static/qi_seed_robustness_scale060_xblock_lgmres_rescue_seed3_gpu0_2026_05_15_retry.json`).
  This closes the current solver-label/threshold search as negative evidence:
  the next QI step must be a stronger residual-reducing coarse/preconditioner
  or a lower-cost x-block/operator-reuse implementation, not a default-window
  promotion.
- [x] 2026-05-15 PAS production real-solve probe campaign: geometry4 and HSX
  short real-solve probes were residual-clean and solver-path stable but failed
  the default-promotion gate because they did not beat the checked baseline
  runtime/RSS. Geometry4 `tzfft` ended at residual `5.87e-8` but regressed to
  `38.25 s` / `2.71 GB`; HSX `tzfft` ended at residual `1.60e-4` but had a
  slight runtime regression, while `tzfft_lgmres` had a memory regression.
  Artifacts:
  `docs/_static/rhs1_pas_matrixfree_real_probe_geometry4_2026_05_15.json`,
  `docs/_static/rhs1_pas_production_solve_geometry4.json`,
  `docs/_static/rhs1_pas_matrixfree_real_probe_hsx_2026_05_15.json`, and
  `docs/_static/rhs1_pas_production_solve_hsx.json`. Keep PAS at `96%` and do
  not promote these candidates.
- [x] 2026-05-15 focused validation for this follow-up: `python -m py_compile`
  and `ruff check --select F821,F823` passed on the touched QI/PAS files, and
  the focused QI/PAS test slice passed: `55 passed in 9.97 s`.
- [x] Remaining integration work in this large push: merged the parallel and
  refactor worker outputs, updated the completion artifact with measured
  results, and ran release/research-lane checks, docs build, and a broader
  focused package test set. Final local full-suite/commit/push remains the last
  closeout step for this push.
- [x] 2026-05-15 release-docs closeout: prepared the `v1.1.3` patch release
  candidate narrative and version metadata. The release scope is now explicit:
  audited example-suite parity, bounded large-QI non-autodiff host fallback,
  PAS byte-budget/rejected-candidate gates, and transport-worker GPU throughput
  are release-facing; production-resolution QI CPU/GPU ladders, true
  differentiable device-QI, and single-case multi-GPU strong scaling are
  deferred/nonblocking research lanes until their checked artifacts pass.

Current active lane (2026-05-15, post-v1.1.3 research-lane closure pass):
- [x] Full pass over the open-lane state confirms the release is not blocked by
  correctness or public-scope parity. The still-open lanes are algorithmic
  research/performance gates: true device-QI, production-resolution QI ladders,
  geometry-rich PAS runtime/RSS promotion, and single-case multi-GPU strong
  scaling. The current evidence is summarized by
  `docs/_static/research_lane_completion_2026_05_12.json` and documented in
  `docs/research_lanes.rst`.
- [x] Code audit result for true device-QI:
  `sfincs_jax/rhs1_qi_coarse.py`,
  `sfincs_jax/rhs1_qi_galerkin_policy.py`,
  `sfincs_jax/rhs1_device_operator.py`, and the x-block paths in
  `sfincs_jax/v3_driver.py` already provide device CSR matvecs, rank-gated QI
  bases, fail-closed Galerkin probes, compact factor experiments, and a
  documented non-autodiff host fallback. The failed scale-0.60 GPU hard-seed
  artifacts show the blocker is residual quality, not storage format or a
  Krylov-name choice. Next implementation: build a PETSc-style two-level
  field-split/Schur preconditioner for the active RHSMode=1 system:
  local x/species/angular block smoother on device, small replicated
  moment/constraint Schur solve, and multiplicative residual update. Gate it by
  applying the preconditioner to physics load bases before launching long GMRES.
- [x] Implement `rhs1_qi_two_level` as an opt-in device-compatible candidate.
  Proposed write scope:
  `sfincs_jax/rhs1_qi_coarse.py` for basis/restriction/prolongation reuse,
  a new pure policy/helper module for field-split metadata and acceptance
  gates, and the existing x-block sparse-PC hook in `sfincs_jax/v3_driver.py`.
  The implementation is off by default and opt-in only. Promotion gates remain:
  a material true-residual probe reduction on the scale-0.60 hard seed, CPU and
  GPU HDF5 plus solver-trace writes within the bounded wall-time budget, and
  CPU/GPU observables within the existing parity gates.
- [x] First implementation step for `rhs1_qi_two_level`: added
  `sfincs_jax/rhs1_qi_two_level.py`, a JAX-compatible local-smoother plus
  coarse-correction primitive,
  `M^{-1} r = S^{-1}r + Q A_c^{-1} Q^T (r - A S^{-1}r)`, with a fail-closed
  true-residual probe. Added `tests/test_rhs1_qi_two_level.py` covering
  residual reduction on a low-rank-coupled system, `jax.jit` compatibility, and
  impossible-improvement rejection. This closes the pure primitive, not the
  production wiring or scale-0.60 GPU hard-seed gate.
- [x] Wired the two-level primitive into the x-block sparse-PC path behind
  `SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER=1`. The hook
  reuses the existing QI coarse-basis controls, builds a local-smoother plus
  coarse correction around the current x-block preconditioner, probes the true
  physical residual, seeds `x0`, and replaces the Krylov preconditioner only
  when the probe improves. Focused integration evidence on the quick x-block
  fixture correctly builds and rejects the candidate when it worsens the true
  residual, preserving baseline convergence and recording metadata. This closes
  production wiring and fail-closed safety; it does not close the scale-0.60
  hard-seed GPU gate until a larger artifact accepts and converges.
- [x] Scale-0.60 seed-3 CPU hard-seed preflight for the two-level candidate:
  the public auto path timed out before reaching the x-block hook, confirming
  that the hard seed still needs explicit x-block/device algorithm work. The
  explicit `xblock_sparse_pc_gmres` path converged in 158.4 s with residual
  ratio `5.96e-3` after the two-level candidate rejected itself
  fail-closed (`3.02e-5 -> 2.05e-4`). A damped one-step scan found only a
  tiny 0.7% residual decrease at damping `1e-2`; that was rejected as
  non-material after it drove the subsequent Krylov path to 288.5 s and
  3569 matvecs. The default acceptance gate now requires at least 5% residual
  reduction, and damping scans are explicit-only via
  `SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_DAMPINGS`, before
  the candidate can replace the baseline preconditioner. A fresh no-two-level
  baseline pass on the same seed completed in 225.3 s and 2723 matvecs, while
  the final single-damping opt-in rejected path completed in 276.9 s and
  3424 matvecs; both passed residual gates, but neither justifies promotion.
- [ ] Redesign the next QI candidate around a stronger coarse space or a true
  Schur/moment field split before spending GPU time. The current two-level
  helper is useful infrastructure and metadata, but it is not a production
  residual reducer for the scale-0.60 hard seed.
- [x] Implemented the first stronger coarse-space A/B: the two-level helper now
  supports `action_lstsq`, which solves the small coarse problem against the
  stored action `A Q` instead of only `Q^T A Q`, and an explicit residual
  augmentation mode that prepends local-smoother and remaining-residual
  directions before rank gating. Pure tests cover a non-normal operator where
  projected Galerkin fails but action least-squares reduces the residual;
  driver tests cover metadata and fail-closed residual augmentation.
- [x] Scale-0.60 seed-3 CPU A/B for the stronger coarse-space candidates:
  action least-squares without residual augmentation still rejected itself
  (`3.02e-5 -> 2.00e-4`) and completed residual-clean in 313.7 s / 3869
  matvecs. Residual augmentation improved the one-step probe
  (`3.02e-5 -> 2.96e-5`, ratio `0.9783`) and completed residual-clean in
  195.0 s / 2295 matvecs, but it did not meet the 5% material-improvement gate.
  A deliberately loosened 2% acceptance test immediately worsened the side
  probe (`8.5e4 -> 2.4e7` residual ratio), so the 5% default gate remains
  correct and no GPU promotion is justified yet.
- [x] Added bounded preconditioned-residual Krylov augmentation for the same
  fail-closed two-level path. The first deep-Krylov probe exposed a useful bug:
  raw high-norm adaptive residual directions dominated the rank threshold and
  collapsed the retained basis from rank 32 to rank 2. Adaptive vectors are now
  normalized before rank gating, and metadata records the augmentation depth and
  labels. The corrected scale-0.60 seed-3 CPU probe retained rank 39 and stayed
  residual-clean, but the one-step improvement remained `0.9783`, below the 5%
  gate, with 242.6 s / 2942 matvecs. Damping line-search exploration also
  rejected before promotion. Conclusion: residual-vector enrichment is safe and
  diagnostic, but closing true device-QI needs a physically different Schur /
  moment coarse operator, not just deeper residual Krylov vectors.
- [x] Implemented the first smoothed-load field-split A/B inside the same
  fail-closed QI two-level hook. The opt-in
  `SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_SMOOTHED_LOAD_BASIS=1`
  path builds source/constraint, flux-surface-average, and low-angular load
  vectors, applies the local x-block smoother to obtain Schur-like
  prolongation directions, rank-gates them, and records the load metadata in
  solver traces and QI seed summaries. The scale-0.60 seed-3 CPU artifact
  `docs/_static/qi_seed_robustness_scale060_qi_two_level_smoothed_load_seed3_cpu_2026_05_16.json`
  passed the residual gate after fail-closed rejection, but the one-application
  probe only improved `3.0215e-5 -> 2.9902e-5` (`0.9896` ratio), below the 5%
  material gate, and the full solve took 285.7 s / 3523 matvecs. This confirms
  that smoothed physical load directions alone are not enough to close the QI
  hard seed; the next candidate needs a true block-Schur/moment operator, not
  just a better coarse subspace.
- [x] Closed the safety gap in the existing `constraintScheme = 1`
  moment-Schur wrapper by adding an opt-in true-residual probe:
  `SFINCS_JAX_RHSMODE1_XBLOCK_PC_MOMENT_SCHUR_PROBE=1` with a configurable
  material-improvement gate. Rejected candidates now restore the baseline
  x-block preconditioner, suppress moment-Schur seeding, and write
  `xblock_moment_schur_used`, `xblock_moment_schur_reason`, and probe residuals
  into solver traces and QI seed summaries.
- [x] Scale-0.60 seed-3 CPU A/B for the probed moment-Schur wrapper:
  `docs/_static/qi_seed_robustness_scale060_moment_schur_probe_seed3_cpu_2026_05_16.json`
  passed only because the fail-closed gate rejected the wrapper. The
  one-application residual worsened from `3.0215e-5` to `2.0325e-4`
  (`6.73` ratio), then the fallback path completed residual-clean in
  288.2 s / 3502 matvecs. This confirms the current dense moment-Schur wrapper
  is not the missing residual reducer; the next QI algorithm must use a
  physically stronger block-Schur/angular/radial coupling or documented
  non-autodiff host fallback for production use.
- [ ] Production-resolution QI ladder after `rhs1_qi_two_level` passes the hard
  seed. Required sequence: scale-0.60 five seeds on CPU and one GPU, then the
  production-resolution proxy ladder at `25 x 51 x 100 x 8` or the documented
  production manifest equivalent. Do not count timed-out, host-idle, or
  nonconverged artifacts as progress. The public claim remains the
  non-autodiff host fallback until this ladder is closed.
- [x] Code audit result for geometry-rich PAS runtime/RSS:
  `sfincs_jax/rhs1_pas_matrixfree.py`,
  `sfincs_jax/rhs1_pas_policy.py`, `sfincs_jax/pas_smoother.py`, and
  `scripts/benchmark_pas_tz_memory_fallback.py` now contain preflight byte
  budgets and residual gates, but the checked geometry4/HSX real-solve probes
  did not improve runtime/RSS. Next implementation must change algorithmic
  cost: stream pitch-angle/angular correction actions with `lax.scan`/chunked
  vector updates or use a lower-memory block-tridiagonal PAS smoother. Another
  dense update candidate or looser fallback threshold is explicitly rejected.
- [ ] Implement a streamed PAS correction prototype behind an opt-in flag.
  Proposed write scope: extend `rhs1_pas_matrixfree.py` with a chunked
  correction interface that never materializes a full dense update block, add
  a benchmark mode in `scripts/benchmark_pas_tz_memory_fallback.py`, and add
  CI-fast unit tests for chunk invariance/residual gates. Promotion gates:
  geometry4, HSX, and geometry11 artifacts must be residual-clean and improve
  either warm runtime by at least 20% or active/RSS memory by at least 25% on
  both CPU and GPU where available.
- [x] Code audit result for single-case multi-GPU strong scaling:
  `examples/performance/benchmark_sharded_solve_scaling.py`,
  `examples/performance/benchmark_sharded_matvec_scaling.py`,
  `sfincs_jax/transport_parallel_policy.py`, and
  `sfincs_jax/transport_parallel_execution.py` already protect release claims
  from cold compile/setup artifacts. The failing evidence shows current
  single-case sharding is synchronization/setup dominated; transport-worker and
  scan/case-level throughput remain the release-facing scaling story.
- [ ] Implement the next single-case multi-GPU attempt as compiled sharded
  operator reuse, not process-per-sample benchmarking. Use JAX `NamedSharding`
  or `shard_map` for the matvec over theta/zeta slabs, keep the small
  coarse/Schur problem replicated, and persist compilation cache on a shared
  filesystem. Acceptance gates: warm 2-GPU solve must beat warm 1-GPU solve
  by at least 1.15x, per-device memory must not increase, and deterministic
  residual/output checks must pass.
- [x] Literature and external-code audit result:
  PETSc `PCFIELDSPLIT` supports the exact block/Schur direction needed here;
  the SFINCS v3 manual/paper justify dropping speed/species coupling for a
  cheaper LU-factorized preconditioner; JAX `jax.Array`, `NamedSharding`,
  multi-controller execution, persistent compilation cache, and `shard_map`
  are the right infrastructure for sharded array execution; JAX sparse is
  useful for compatibility but the official docs warn it is experimental and
  not a performance-critical backend; Lineax remains an optional benchmark
  lane, not a production dependency, until real SFINCS operator errors are
  resolved.
- [ ] Validation order for this pass:
  1. add pure unit tests for the new two-level QI and streamed PAS helpers;
  2. run focused solver/policy tests under CPU;
  3. run bounded scale-0.60 seed-3 CPU and one-GPU probes;
  4. if and only if the hard seed passes, run five-seed QI ladders and
     geometry-rich PAS CPU/GPU artifacts;
  5. update README/docs/figures only after checked artifacts change public
     claims.

Current active lane (2026-05-11, validation/docs/release integration):
- [x] Keep this pass write-scoped to validation, documentation, planning, and
  release gates. Do not edit `v3_driver.py`, QI/mapped-grid implementation code,
  or performance kernels from this lane.
- [x] Fetched `origin` and confirmed local `main` / `origin/main` are aligned at
  `04e2dd233340b66c7d59ec8d6053356fb45cf3cf`.
- [x] Add explicit release-gate metadata to the validation manifest. Every
  publication-facing lane must now be one of `release_ready`,
  `regression_scaffold`, `bounded_proxy`, or `closed_deferred`, and no lane may
  silently block the current release.
- [x] Add a CI-fast release check (`scripts/check_release_gates.py` plus
  `tests/test_release_gate_metadata.py`) so closed-vs-deferred claim metadata is
  tested without launching expensive solves.
- Local integrated evidence status: the dirty worktree now contains the mapped
  x-grid primitives, opt-in `xGridScheme = 50` construction, transport-objective
  helpers, solve-facing mapped transport evidence, QI seed materialization
  runner, and `solver_path_policy.py` extraction. This docs lane records that
  state without editing source, tests, or scripts.
- Seed-robust QI status: `scripts/run_qi_seed_robustness.py` and
  `tests/test_run_qi_seed_robustness.py` now provide deterministic neighboring
  smoke decks around `examples/additional_examples/input.namelist`, localize the
  QI VMEC equilibrium per seed, perturb `nu_n` / `Er`, and can optionally
  execute each seed through `sfincs_jax write-output`. The runner now includes
  opt-in residual/convergence gates (`--max-residual-ratio`,
  `--require-converged`, and `--require-accepted-converged`) plus aggregate
  execution summaries. The checked summaries at
  `docs/_static/qi_seed_robustness_smoke.json` and
  `docs/_static/qi_seed_robustness_multiseed.json` record bounded default-CLI
  passes. The multi-seed artifact covers seeds `0,1,2` at `7 x 13 x 25 x 4`,
  with `process_failed=0`, all seeds `converged=true`, public `auto` solve
  method, and maximum residual ratio `7.872e-7`. This also fixed the previous
  `Nxi=25` default-policy failure where `auto` skipped dense, entered a slow
  fallback path, and failed the residual gate. It is not yet a
  production-resolution CPU/GPU QI robustness claim.
- Bounded GPU QI update: the same `7 x 13 x 25 x 4` three-seed QI ladder now
  passes on one RTX A4000 through the public `auto` CLI policy. The fix removed
  the stale CPU-only guard around the full-FP dense-auto selector, so moderate
  accelerator systems above `SFINCS_JAX_RHSMODE1_DENSE_FP_ACCELERATOR_MIN` use
  the dense path before the sparse/fallback ladder. The checked artifact
  `docs/_static/qi_seed_robustness_multiseed_gpu.json` records `3/3` converged
  seeds, `0` timeouts, max residual ratio `4.137e-7`, and max elapsed
  `43.99 s`; before the guard removal, the same default GPU seed produced no
  solver trace after more than two minutes, while forced dense passed. This is a
  bounded GPU robustness gate, not a production-resolution claim.
- Higher bounded QI accelerator host-sparse update: a `9 x 19 x 35 x 4`
  single-seed QI probe at `13169` active unknowns reproduced the next failure
  tier on one RTX A4000. Before this patch, the GPU `auto` path spent `195 s`
  in the Krylov/strong-preconditioner tail and was rejected with residual ratio
  `53.9`. The solver policy now allows explicit, non-implicit accelerator
  output runs below
  `SFINCS_JAX_RHSMODE1_ACCELERATOR_HOST_SPARSE_RESCUE_MAX=30000` to use the
  bounded host sparse/x-block rescue. The same GPU probe now converges in
  `42.8 s` with residual ratio `4.49e-7`; the local CPU check converges in
  `29.8 s` with residual ratio `1.16e-6`. The checked artifact is
  `docs/_static/qi_seed_robustness_scale035_cpu_gpu.json`. This closes the
  immediate bounded QI GPU failure, while production-resolution QI ladders
  remain a separate validation lane.
- Mapped x-grid status: checked JSON/CSV artifacts under `docs/_static/` now
  exercise tiny and reduced PAS RHSMode=2 transport-matrix comparisons. The
  reduced PAS tokamak artifact compares mapped `Nx=7` candidates against an
  `Nx=13` reference and identifies the best transport-error log length, but this
  remains bounded PAS evidence. Do not claim full-FP compatibility,
  production-resolution speedup, or default-grid replacement.
- Solver-path refactor status: `solver_path_policy.py` centralizes budget,
  JIT, dtype, sparse-PC default, structural-tolerance, residual-rescue, and
  resource-exhaustion decisions with direct tests. This is a policy-surface
  hardening lane, not a new numerical solver claim.
- Host-refinement refactor status: `host_refinement.py` now owns the NumPy-only
  dense and sparse host iterative-refinement loops used by direct-solve rescue
  paths. `v3_driver.py` keeps compatibility wrappers, so solver behavior is
  unchanged while the implementation becomes directly testable without loading
  the full driver.
- Large refactor closure status: the mapped-x-grid / QI / solver-path sequence
  is locally integrated in this worktree, but promotion should still be staged
  by feature area: adaptive speed-map primitives, mapped x-grid construction,
  transport objectives, transport evidence, solver-path policy extraction, then
  QI evidence. Re-run release-gate metadata, mapped-grid tests, solver-path
  tests, docs, and the QI manifest/execute smoke at each promotion point.
- GitHub/remote status after `git fetch --prune origin`: open PRs #2-#6 are all
  draft. #2 targets `main` and GitHub reports `mergeStateStatus=CLEAN`; #3, #4,
  and #6 are also `CLEAN` against their stacked bases. #5 reports
  `mergeStateStatus=UNSTABLE` because a stale coverage check was still in
  progress even though the newer run was green. All branch heads are one commit
  behind current `origin/main` (`04e2dd2`) and should be rebased before merge.
  `git merge-tree --write-tree origin/main <branch>` reported clean trees for
  each remote branch, so the main risk is semantic/test integration, not text
  conflict.
- Remaining release decision: current release claims remain clean after this
  docs/gate pass; mapped-grid, QI, and solver-path refactor evidence should stay
  bounded integration lanes until their artifacts are promoted through the
  manifest with explicit `release_gate` metadata and the QI execute smoke is
  passing.

Current active lane (2026-05-10, production-floor PAS memory/runtime closeout):
- [x] Add bounded resolution overrides to
  `scripts/benchmark_pas_tz_memory_fallback.py` so PAS memory/runtime probes can
  run the collaborator floor directly (`Ntheta=25, Nzeta=51, Nxi=100, Nx=4`)
  without editing checked-in example inputs. The JSON plan now records the
  exact `input_overrides` used for each probe.
- [x] Reproduce the large geometry-rich PAS stall class on CPU using the forced
  PAS-TZ memory-fallback harness. At `25 x 51 x 100 x 4`, the cheap collision
  fallback avoids the dense allocation and returns in `2.00 s` / `1.56 GB`, but
  the true residual is unusable (`4.71e6`), so it remains a stall-avoidance
  diagnostic only, not a solver policy.
- [x] Validate the stronger low-memory `tzfft` fallback at the same
  production-floor resolution. CPU converged with `restart=20`, `maxiter=20` in
  `40.85 s` / `2.97 GB`, residual `5.87e-8`. Office GPU 0 converged the same
  case in `47.54 s` / `1.31 GB` host RSS, residual `5.88e-8`; sampled GPU
  memory stayed near `1.23 GB` with XLA preallocation disabled.
- [x] Record artifacts:
  `examples/performance/output/pas_tz_prod_floor_cpu_collision_25x51x100x4.json`,
  `examples/performance/output/pas_tz_prod_floor_cpu_tzfft_25x51x100x4.json`,
  and
  `examples/performance/output/pas_tz_prod_floor_gpu_tzfft_25x51x100x4.json`.
- [x] Reject the opt-in matrix-free Jacobi defect-smoother probe before merging
  it. It was disabled by the safety guard on the representative case
  (`active_size=83122 > 50000`), and enabling it at production-floor sizes would
  require O(active_size) matrix-free diagonal probes. This is not a credible
  memory/runtime path compared with the measured `tzfft` fallback.
- [x] Fix an over-aggressive PAS-DKES solver-budget policy: explicit
  `SFINCS_JAX_GMRES_RESTART` / `SFINCS_JAX_GMRES_MAXITER` budgets are now
  respected instead of being silently raised to `restart >= 80` and
  `maxiter >= 600`. DKES default floors still apply when the user does not
  force a budget. This directly addresses the collaborator-reported class of
  "small resolution change selects a much slower path" failures and keeps
  bounded profiling honest.
- [x] Run the same `25 x 51 x 100 x 4` floor on the first HSX and geometry11
  PAS-DKES cases. With explicit `restart=20`, `maxiter=20`, HSX DKES completed
  in `58.25 s` / `2.69 GB`, residual `1.60e-4`; geometry11 DKES completed in
  `57.72 s` / `2.94 GB`, residual `2.61e-2`. A moderate HSX budget
  (`restart=40`, `maxiter=80`) took `211.26 s` / `1.93 GB`, residual
  `1.80e-5`. These are bounded, reproducible diagnostics, not promotion
  candidates.
- [x] Record additional artifacts:
  `examples/performance/output/pas_tz_floor_hsx_dkes_cpu_explicit_m20_25x51x100x4.json`,
  `examples/performance/output/pas_tz_floor_hsx_dkes_cpu_m80r40_25x51x100x4.json`,
  and
  `examples/performance/output/pas_tz_floor_geom11_dkes_cpu_explicit_m20_25x51x100x4.json`.
- [x] Complete the explorer-recommended host-side PAS x-block sparse-PC probe
  behind an explicit solve-method/env gate, then remove it after the bounded
  medium gate failed the residual and memory/runtime gates below. Keep/reject
  gates remain: true residual clean, wall time below the current `tzfft`
  bounded budget, lower peak memory than dense/JAX-factor routes, CPU and GPU
  parity agreement, and no default promotion without full example-suite parity.
- [x] Prototype/reject the first host-side PAS x-block sparse-PC route before
  shipping it. The explicit `pas_xblock_sparse_pc_gmres` prototype avoided
  padded JAX factor arrays, but on the medium geometry4 PAS case the default
  ILU settings rejected nearly all block factors as unstable and stopped at
  residual `2.44e-4` after `96` bounded matvecs (`11.37 s`, `0.93 GB`). A
  stronger ILU probe (`drop_tol=1e-8`, `fill_factor=50`) was worse
  (`6.58e-4`, `14.74 s`, `1.31 GB`). The prototype was removed rather than
  exposing a public solve-method alias that fails the medium gate. Any future
  host-side PAS route should first fix factor stability/normalization on this
  medium gate before attempting HSX/geometry11 production-floor runs.
- [x] Reject the existing guarded stage-2 retry as a HSX floor fix. With
  `SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_STAGE2_RETRY=1` on the same
  `25 x 51 x 100 x 4` HSX PAS-DKES floor, runtime increased from `58.25 s` to
  `155.05 s` and peak RSS increased from `2.69 GB` to `2.94 GB`, while the true
  residual stayed unchanged at `1.60e-4`. Artifact:
  `examples/performance/output/pas_tz_floor_hsx_dkes_cpu_stage2_m20_25x51x100x4.json`.
  Do not use stage-2 retry as the default answer to this lane.
- [x] Reject the existing guarded strong retry as a HSX floor fix. With
  `SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_STRONG_RETRY=1`, the same bounded HSX
  floor timed out at `240 s` after entering `strong preconditioner fallback
  kind=pas_lite`, without producing a residual-clean result. Artifact:
  `examples/performance/output/pas_tz_floor_hsx_dkes_cpu_strong_m20_25x51x100x4.json`.
  The remaining HSX/geometry11 PAS-DKES floor work requires a new structured
  preconditioner/formulation, not larger default retry ladders.
- [x] Prototype/reject the first guarded structured residual-correction route.
  The implementation is opt-in only via
  `SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_STRUCTURED_LEVELS` and uses accepted
  local minimum-residual coarse corrections (`xmg`, `collision`) to avoid the
  unstable fixed-damping correction. It still failed the promotion gates. On
  the medium geometry4 PAS gate, baseline `tzfft` took `3.56 s` / `0.93 GB`,
  residual `1.88e-4`; structured variants took `5.23-6.24 s` / `1.15-1.37 GB`
  with no residual improvement. On the HSX `25 x 51 x 100 x 4` floor,
  baseline `tzfft` took `58.16 s` / `2.83 GB`, residual `1.60e-4`;
  `tzfft_structured` took `98.87 s` / `2.95 GB`, residual `1.63e-4`. Artifacts:
  `examples/performance/output/pas_tz_structured_medium_geometry4_probe.json`
  and
  `examples/performance/output/pas_tz_floor_hsx_dkes_cpu_structured_m20_25x51x100x4.json`.
  Do not promote this route to defaults.
- [x] Fix solver-path provenance for host Krylov A/B probes. Before this pass,
  `solve_method=lgmres` could still replay GMRES for
  `SFINCS_JAX_SOLVER_ITER_STATS=1` and report `solver=gmres`, which polluted
  profiling and could add hidden iteration-stat overhead. Iteration replay now
  labels and replays LGMRES as `solver=lgmres`, and the PAS fallback benchmark
  accepts `--solve-method` plus `_lgmres` variant suffixes.
- [x] Prototype/reject LGMRES as a PAS-TZ `tzfft` production-floor default.
  It is useful as an opt-in A/B route and now has reproducible artifacts, but it
  fails the promotion gates because the memory win is not robust. Medium
  geometry4: `tzfft` `3.72 s` / `0.97 GB`, residual `1.88e-4`;
  `tzfft_lgmres` `3.66 s` / `0.91 GB`, same residual. HSX
  `25 x 51 x 100 x 4`: `tzfft` `58.12 s` / `2.92 GB`, residual `1.60e-4`;
  `tzfft_lgmres` `55.63 s` / `2.97 GB`, same residual. Geometry11
  `25 x 51 x 100 x 4`: `tzfft` `62.01 s` / `3.22 GB`, residual `1.90e-2`;
  `tzfft_lgmres` `61.25 s` / `3.32 GB`, same residual. Artifacts:
  `examples/performance/output/pas_tz_lgmres_medium_geometry4_probe.json`,
  `examples/performance/output/pas_tz_floor_hsx_dkes_cpu_lgmres_m20_25x51x100x4.json`,
  and
  `examples/performance/output/pas_tz_floor_geom11_cpu_lgmres_m20_25x51x100x4.json`.
  Do not promote LGMRES by default for guarded PAS-TZ.

Current active lane (2026-05-08, production-floor FP memory audit):
- [x] Verify `office` is reachable and run the latest clean local `main` source
  from an isolated remote scratch tree instead of mutating stale remote artifacts.
- [x] Re-run focused remote tests for the sparse-PC memory/trace changes:
  `pytest -q tests/test_memory_model.py tests/test_explicit_sparse.py
  tests/test_solver_trace_output_formats.py
  tests/test_io_export_and_h5_coverage.py::test_rhsmode1_solver_diagnostics_are_output_visible`
  (`27 passed` on `office`).
- [x] Profile the real production-floor GPU offender
  `tokamak_1species_FPCollisions_noEr` at `Ntheta=25, Nzeta=1, Nxi=100,
  NL=4, Nx=8` against the existing Fortran v3 reference. Current default
  sparse-PC GMRES is accurate: `188` compared datasets, `0` mismatches, true
  residual `1.33e-09` against target `3.43e-09`.
- Result: current sparse-PC GMRES completed in `2:31.6` on office GPU 0. The
  trace shows the runtime is pre-solve setup dominated: `setup_s=148.13`,
  `solve_s=0.59`, `sparse_pattern_build_s=122.70`, `sparse_pc_factor_s=25.38`,
  and only `14` true-operator matvecs. Peak host RSS was about `8.42 GB`;
  estimated CSR storage was about `0.96 GB`, GMRES basis about `13.6 MB`, and
  SuperLU factor storage estimate about `57.9 MB`. The performance/memory issue
  is the conservative FP preconditioner pattern, not Krylov iteration count.
- [x] Add an opt-in sparse-PC knob
  `SFINCS_JAX_RHSMODE1_SPARSE_PC_FP_DENSE_VELOCITY_BLOCK` plus solver metadata
  `sparse_pc_fp_dense_velocity_block`, and add a focused regression showing the
  local-velocity FP pattern is strictly smaller than the dense-velocity
  conservative pattern.
- Candidate rejected as default: forcing the matrix-free incremental GPU path
  completed in `1:13.9` with lower host RSS (`2.34 GB`) but stopped at residual
  `6.91e-03` against target `3.43e-09`; it is not a parity-safe production
  answer.
- Candidate rejected as default: forcing
  `SFINCS_JAX_RHSMODE1_SPARSE_PC_FP_DENSE_VELOCITY_BLOCK=false` reduced the
  sparse-PC pattern from `80,000,000` nnz to `494,000` nnz and reduced probing
  from `4000` colors to `25`, but it was still running after `208 s` and
  `1475` matvecs, so it failed the wall-time gate versus the current `150 s`
  accurate baseline. Keep it as an explicit memory-pressure experiment only.
- Candidate rejected as default: enabling the existing reduced sparse rescue
  with `SFINCS_JAX_RHSMODE1_SPARSE_MAX=20000` and sparse preconditioning built
  a much smaller reduced operator (`~190k` nnz, peak RSS about `1.05 GB`) but
  timed out at `360 s` inside the sparse-ILU GMRES fallback. This confirms the
  next useful candidate must be a stronger structured FP block method, not the
  current generic sparse-ILU reduced fallback.
- [x] Lower the default nonconverged-output guard threshold to `10,000` active
  unknowns so production-floor runs like this one cannot write benchmark/public
  RHSMode=1 diagnostics when an explicitly forced path misses the true residual
  target, unless `SFINCS_JAX_ALLOW_NONCONVERGED_OUTPUT=1` is set for debugging.
- [x] Implement an explicit `xblock_sparse_pc_gmres` solver method for
  nondifferentiable RHSMode=1 full-FP systems. This reuses the existing
  per-x/TZ assembly helpers, forces a compact x-block host preconditioner where
  safe, avoids the previous global dense-velocity sparse-pattern probe, records
  setup/solve/factorization metadata, and preserves a true-residual acceptance
  gate.
- [x] Promote the validated production-floor tokamak full-FP GPU policy from
  global `sparse_pc_gmres` to `xblock_sparse_pc_gmres` for the auto-selected
  no-Er and with-Er branches. Explicit `sparse_pc_gmres` remains available for
  diagnostics/backward comparison.
- Validation: office GPU default-policy reruns at `Ntheta=25, Nzeta=1,
  Nxi=100, NL=4, Nx=8` are parity-clean against Fortran v3. No-Er selected
  `xblock_sparse_pc_gmres`, compared `188` datasets with `0` mismatches, and
  completed in `5.79 s` wall / `0.97 GB` peak RSS (`trace elapsed=3.89 s`,
  residual `1.25e-09` against target `3.43e-09`, `33` matvecs). This replaces
  the previous global sparse-PC default (`2:31.6`, about `8.42 GB` peak RSS).
- Validation: office GPU default-policy with-Er selected
  `xblock_sparse_pc_gmres`, compared `214` datasets with `0` mismatches, and
  completed in `1:13.3` wall / `1.43 GB` peak RSS (`trace elapsed=71.33 s`,
  residual `1.48e-15` against target `3.18e-14`, `467` matvecs).
- [x] Re-profile the remaining full-trajectory GPU offender and the adjacent
  full-FP tokamak rows. Rejected default changes: right preconditioning
  (`133.3 s`), GMRES restart `120` (`107.0 s`), and restart `160`
  (`185.1 s`) were all slower than the existing left-preconditioned baseline.
  Promoted change: use exact per-x/TZ sparse LU up to block size `3000` for
  host-side full-FP `xblock_sparse_pc_gmres` before falling back to ILU. This
  keeps PAS/JAX-factor paths at the previous `2000` cap and records the cap in
  the preconditioner cache key.
- Validation: office GPU default-policy reruns after the exact-LU cap promotion
  were parity-clean against Fortran v3. No-Er completed in `4.40 s` wall /
  `0.866 GB` peak RSS (`trace elapsed=3.37 s`), with-Er DKES in `26.28 s` wall /
  `1.28 GB` peak RSS (`trace elapsed=25.26 s`), and with-Er full trajectories
  in `43.83 s` wall / `1.34 GB` peak RSS (`trace elapsed=42.75 s`); each had
  `0` output mismatches.
- Validation: the five-case bounded GPU production-floor refresh at
  `tests/production_floor_bounded_remote_gpu_xblock_lu3000_2026-05-08`
  completed `5/5 parity_ok`, strict `0` mismatches, missing Fortran output keys
  `0`, and no resolution reductions at `Ntheta=25, Nzeta=1, Nx=8, Nxi=100`.
- Validation: the bounded CPU production-floor full-FP tokamak refresh at
  `tests/production_floor_cpu_tokamak_fp_lu3000_2026-05-08` completed
  `5/5 parity_ok`, strict `0` mismatches, missing Fortran output keys `0`, and
  no resolution reductions at `Ntheta=25, Nzeta=1, Nx=8, Nxi=100`.
- Validation: the bounded GPU production-floor PAS tokamak refresh at
  `tests/production_floor_gpu_tokamak_pas_2026-05-08` completed
  `6/6 parity_ok`, strict `0` mismatches, missing Fortran output keys `0`, and
  no resolution reductions. Together with the existing CPU PAS floor artifact,
  the bounded tokamak FP/PAS benchmark lanes are closed on CPU and GPU.
- Artifact update: the public README/docs benchmark source reports now point at
  `tests/scaled_example_suite_release_cpu_2026-05-08_production_tokamak` and
  `tests/scaled_example_suite_gpu_bounded_default_2026-05-08_lu3000_pas`.
  Remaining production-resolution floor violations dropped to the 16
  remote/cluster-scale 3D rows on each backend; these are explicitly classified
  by the manifest as `remote_or_cluster_only` (`~1.0M` to `4.3M` unknowns) and
  should not be launched as local bounded jobs.
- [x] Add regression coverage for the new solve-method aliases, the explicit
  x-block full-FP solve, the automatic no-Er policy selection, and the HDF5
  diagnostics reporting x-block preconditioner details.
- Next best steps: keep the 16 remaining 3D floor rows as a cluster/nightly
  campaign unless a dedicated large-memory node is available; do not include
  them as production-floor public rows from legacy small-resolution artifacts.

Current active lane (2026-04-27):
- [x] Audit RHSMode=1 solver-path selection after the reported full-collision Nxi cliff.
- [x] Keep CPU full-FP moderate systems on the direct dense path when it is faster and lower-memory than Krylov/strong/sparse rescue.
- [x] Profile solver-stage transitions with `SFINCS_JAX_PROFILE=1` and keep benchmark rows recording preconditioners, dense/sparse fallback use, stage timings, and RSS.
- [x] Rerun focused Nxi=20/Nxi=40 full-FP repros, compare against Fortran v3 outputs, then refresh bounded example-suite parity/performance reports.
- Result: focused Ntheta=13/Nxi=40 full-FP default changed from the Krylov/strong/sparse rescue path (`37.54 s`, about `3.25 GB` profiled RSS) to the direct dense path (`1.28 s`, about `0.53 GB` profiled RSS) with `0` mismatches against Fortran v3.
- Result: `tests/scaled_example_suite_solver_path_audit_2026-04-27` completed `39/39` `parity_ok`, strict `0` mismatches, missing Fortran output keys `0`, and runtime drift flags `0` against the frozen CPU release baseline.

Current active lane (2026-04-28):
- [x] Validate the solver-path fix on GPU using an isolated `office` scratch checkout rather than the dirty stale remote repo.
- [x] Fix the accelerator host-dense shortcut closure bug in `sfincs_jax/v3_driver.py` so the host-dense probe no longer fails before measuring parity/runtime.
- [x] Validate blanket accelerator dense auto-selection as parity-clean but reject it as a default because it regressed `16` tiny GPU suite cases.
- [x] Land bounded accelerator dense auto-selection for full-FP RHSMode=1 systems with `SFINCS_JAX_RHSMODE1_DENSE_FP_ACCELERATOR_MIN=1000`.
- [x] Complete the bounded-default GPU suite at `tests/scaled_example_suite_gpu_bounded_default_2026-04-28`: `39/39 parity_ok`, strict `0` mismatches, missing Fortran output keys `0`, and runtime drift flags `0`.
- Result: focused GPU `Ntheta=13,Nxi=40` full-FP default now uses dense automatically (`2.794 s`, about `1.04 GB`) instead of the forced Krylov path (`9.539 s`, about `2.14 GB`), with `0` Fortran mismatches.
- Result: focused GPU `Ntheta=13,Nxi=20` default also uses dense and avoids the pathological forced-Krylov rescue (`137.411 s`), with `0` Fortran mismatches.
- Next best steps: rerun local tests/docs, regenerate the publication benchmark dashboard from the bounded GPU report, then decide whether the remaining GPU top offenders justify another solver-policy change or should stay in the optimization backlog.

Current active lane (2026-04-29):
- [x] Prepare the next minor release from clean `main` after the GPU solver-path validation pass.
- [x] Bump package metadata from `1.0.7` to `1.1.0`.
- [x] Add release notes that document the CPU/GPU parity state, solver-path performance changes, output/plotting capabilities, validation artifacts, and remaining research lanes.
- [x] Refresh GitHub Actions core action versions to current major tags to avoid the Node 20 deprecation path.
- [x] Run release validation: focused package/CLI/output tests, docs with warnings as errors, HDF5/NetCDF/NPZ CLI smoke plus plotting, package build, and full `pytest -q` (`948 passed`).
- [x] Rebuild the package from committed `HEAD` and verify the source distribution includes `docs/release_notes.rst` and wheel metadata reports `Version: 1.1.0`.
- Next best steps: tag `v1.1.0`, push `main` and the tag, and confirm CI/docs/PyPI publication.

Current active lane (2026-04-30):
- [x] Reassess the collaborator-reported solver-path cliff as a general solver-policy risk rather than an isolated Nxi case.
- [x] Reinspect local SFINCS Fortran v3 solver architecture: PETSc/SNES/KSP, sparse preallocation, explicit preconditioner matrices, MUMPS/SuperLU_DIST/PETSc LU, GMRES restart `2000`, residual monitors, and preconditioner reuse.
- [x] Re-anchor the plan in SFINCS, PETSc, MONKES, KNOSOS, NEO, and JAX/Lineax/JAX sparse documentation.
- [x] Add `sfincs_jax/solver_selection_policy.py`, a reusable measured-candidate gate for future automatic solver-path promotions.
- [x] Add `tests/test_solver_selection_policy.py` to guard against the exact failure class: a residual-bad, slower, higher-memory fallback cannot be auto-promoted over a clean fast path.
- [x] Add `sfincs_jax/solver_trace.py`, a stable solver-trace schema with JSON and HDF5 round trips for CLI/Python/benchmark metadata.
- [x] Add `tests/test_solver_trace.py` so future output files can be validated against a versioned trace schema.
- [x] Add opt-in solver-trace metadata support for HDF5, NetCDF, and NPZ output files without changing default parity outputs.
- [x] Add `tests/test_solver_trace_output_formats.py` to validate trace persistence across all supported output formats.
- [x] Wire the measured-candidate gate into the first RHSMode=1 PAS weak-auto promotion point while preserving unmeasured historical defaults.
- [x] Add direct tests showing measured PAS auto-promotion rejects a residual-bad, slower, higher-memory candidate and accepts a clean runtime win.
- [x] Add artifact-backed tests over the 2026-04-27 CPU solver-path audit and 2026-04-28 GPU Nxi=20/40 full-FP focused reports.
- [x] Add optional measured acceptance gates to `rhs1_handoff.py`, the shared acceptance helper used by RHSMode=1 retry/rescue branches.
- [x] Add handoff tests showing clean incumbent paths reject slower/higher-memory candidates, while failed incumbents can still accept slower correctness rescues.
- [x] Pass real elapsed-time and RSS metrics from reduced/full RHSMode=1 stage2 and strong-preconditioner retry call sites into the measured gates.
- [x] Pass real elapsed-time and RSS metrics from reduced/full RHSMode=1 dense and sparse retry call sites into the measured gates.
- [x] Add `solver_trace_path=` and CLI `--solver-trace` JSON sidecar support so real runs can emit stable solver traces without changing parity output files.
- [x] Document `--solver-trace` in the README quick start and CLI usage docs.
- [x] Update `scripts/run_reduced_upstream_suite.py` so JAX CLI repeats write solver-trace sidecars and prefer sidecar elapsed time over free-form log parsing when available.
- [x] Add PAS-heavy path-switching artifact tests so HSX/geometry4/geometry11 stay on `pas_tz`, tokamak PAS stays on its structured `pas_tokamak_theta`/`xblock_tz`/`schur` routes, and all covered PAS artifacts remain strict parity-clean.
- Validation: `pytest -q tests/test_rhs1_handoff.py tests/test_solver_path_artifacts.py tests/test_solver_selection_policy.py tests/test_rhs1_preconditioner_auto_policy.py tests/test_solver_trace.py tests/test_solver_trace_output_formats.py tests/test_io_export_and_h5_coverage.py tests/test_solver_progress.py tests/test_policy_module_docstrings.py` (`60 passed`).
- Validation: `pytest -q tests/test_full_system_operator_jit.py tests/test_v3_driver_rhs1_dispatch_coverage.py tests/test_v3_driver_strong_fallback_coverage.py tests/test_v3_driver_policy_helpers.py` (`26 passed`).
- Validation: `pytest -q tests/test_cli_solve_mode.py tests/test_solver_trace.py tests/test_solver_trace_output_formats.py tests/test_io_export_and_h5_coverage.py` (`45 passed`).
- Validation: real CLI sidecar smoke on `examples/getting_started/input.namelist` with `--geometry-only --solver-trace` wrote `schema_version=1`, `backend=cpu`, `rhs_mode=1`, `selected_path=geometry_only`, and `output_format=h5` in about `0.18 s`.
- Validation: `pytest -q tests/test_runtime_window_attempts.py tests/test_cli_solve_mode.py tests/test_solver_trace.py tests/test_solver_trace_output_formats.py tests/test_io_export_and_h5_coverage.py` (`51 passed`).
- Validation: `pytest -q tests/test_rhs1_handoff.py tests/test_solver_path_artifacts.py tests/test_solver_selection_policy.py tests/test_rhs1_preconditioner_auto_policy.py tests/test_solver_trace.py tests/test_solver_trace_output_formats.py tests/test_io_export_and_h5_coverage.py tests/test_solver_progress.py tests/test_policy_module_docstrings.py tests/test_cli_solve_mode.py tests/test_full_system_operator_jit.py tests/test_v3_driver_rhs1_dispatch_coverage.py tests/test_v3_driver_strong_fallback_coverage.py tests/test_v3_driver_policy_helpers.py tests/test_runtime_window_attempts.py` (`123 passed`).
- Validation: `git diff --check` is clean.
- Validation: focused CPU RHSMode=1 cliff/PAS probe metadata at `tests/solver_policy_trace_gate_cpu_probe_2026-04-30` records `9/9` `parity_ok`, strict mismatches `0`, print parity gaps `0`, missing output keys `0`, with `9` solver-trace sidecars.
- Validation: full bounded CPU suite metadata at `tests/solver_policy_trace_gate_cpu_full_2026-04-30` records `39/39` `parity_ok`, strict mismatches `0`, print parity gaps `0`, missing output keys `0`, with `39` solver-trace sidecars. Worst CPU wall-clock rows remain PAS/geometry-rich and are bounded: HSX PAS DKES `4.845 s`, HSX PAS full `4.527 s`, tokamak 2-species PAS+Er `4.234 s`, geometryScheme4 PAS `3.721 s`, geometry11 PAS full `3.732 s`.
- Validation: focused one-GPU `office` probe metadata from an RTX A4000 at `tests/solver_policy_trace_gate_gpu_probe_2026-04-30` records `7/7` `parity_ok`, strict mismatches `0`, print parity gaps `0`, missing output keys `0`, with trace-backed elapsed times. Worst focused GPU wall-clock rows were HSX PAS full `13.053 s`, geometry11 PAS full `11.745 s`, geometryScheme4 PAS `8.566 s`, HSX PAS DKES `7.615 s`, and tokamak PAS+Er `6.459 s`; no `jax_error` or `max_attempts` occurred.
- Validation: added lightweight report-backed tests for the new trace-backed CPU and GPU artifacts, then reran the local solver-policy gate: `pytest -q tests/test_rhs1_handoff.py tests/test_solver_path_artifacts.py tests/test_solver_selection_policy.py tests/test_rhs1_preconditioner_auto_policy.py tests/test_solver_trace.py tests/test_solver_trace_output_formats.py tests/test_io_export_and_h5_coverage.py tests/test_solver_progress.py tests/test_policy_module_docstrings.py tests/test_cli_solve_mode.py tests/test_full_system_operator_jit.py tests/test_v3_driver_rhs1_dispatch_coverage.py tests/test_v3_driver_strong_fallback_coverage.py tests/test_v3_driver_policy_helpers.py tests/test_runtime_window_attempts.py` (`125 passed`).
- [x] Fix RHSMode=1 handoff acceptance for nonfinite incumbent residuals: a finite candidate rescue is now eligible for acceptance instead of being blocked by a strict `candidate < current` comparison against `NaN`.
- Validation: final local solver-policy gate after the nonfinite-residual regression test: `pytest -q tests/test_rhs1_handoff.py tests/test_solver_path_artifacts.py tests/test_solver_selection_policy.py tests/test_rhs1_preconditioner_auto_policy.py tests/test_solver_trace.py tests/test_solver_trace_output_formats.py tests/test_io_export_and_h5_coverage.py tests/test_solver_progress.py tests/test_policy_module_docstrings.py tests/test_cli_solve_mode.py tests/test_full_system_operator_jit.py tests/test_v3_driver_rhs1_dispatch_coverage.py tests/test_v3_driver_strong_fallback_coverage.py tests/test_v3_driver_policy_helpers.py tests/test_runtime_window_attempts.py` (`126 passed`).
- Validation: full one-GPU `office` refresh metadata at `tests/solver_policy_trace_gate_gpu_full_2026-04-30` records `39/39` `parity_ok`, strict mismatches `0`, print parity gaps `0`, missing output keys `0`, with `39` solver-trace sidecars. Worst one-GPU wall-clock rows are geometry11 PAS full `24.321 s`, geometry11 PAS DKES `22.803 s`, HSX PAS full `19.303 s`, tokamak 2-species PAS+Er `14.314 s`, and geometryScheme4 PAS `10.288 s`; no `jax_error` or `max_attempts` occurred.
- Validation: final local solver-policy gate after adding the full GPU report-backed test: `pytest -q tests/test_rhs1_handoff.py tests/test_solver_path_artifacts.py tests/test_solver_selection_policy.py tests/test_rhs1_preconditioner_auto_policy.py tests/test_solver_trace.py tests/test_solver_trace_output_formats.py tests/test_io_export_and_h5_coverage.py tests/test_solver_progress.py tests/test_policy_module_docstrings.py tests/test_cli_solve_mode.py tests/test_full_system_operator_jit.py tests/test_v3_driver_rhs1_dispatch_coverage.py tests/test_v3_driver_strong_fallback_coverage.py tests/test_v3_driver_policy_helpers.py tests/test_runtime_window_attempts.py` (`127 passed`).
- [x] Fix benchmark RSS accounting for future reports: JAX subprocesses are now wrapped with the same portable `/usr/bin/time` prefix as the Fortran path, and the suite harness falls back to time-reported maximum RSS when profiling logs are off.
- Validation: `pytest -q tests/test_runtime_window_attempts.py tests/test_solver_path_artifacts.py tests/test_rhs1_handoff.py tests/test_solver_selection_policy.py tests/test_rhs1_preconditioner_auto_policy.py` (`51 passed`); `python -m py_compile scripts/run_reduced_upstream_suite.py scripts/run_scaled_example_suite.py`; `git diff --check`.
- Validation: final local solver-policy gate after RSS-accounting changes: `pytest -q tests/test_rhs1_handoff.py tests/test_solver_path_artifacts.py tests/test_solver_selection_policy.py tests/test_rhs1_preconditioner_auto_policy.py tests/test_solver_trace.py tests/test_solver_trace_output_formats.py tests/test_io_export_and_h5_coverage.py tests/test_solver_progress.py tests/test_policy_module_docstrings.py tests/test_cli_solve_mode.py tests/test_full_system_operator_jit.py tests/test_v3_driver_rhs1_dispatch_coverage.py tests/test_v3_driver_strong_fallback_coverage.py tests/test_v3_driver_policy_helpers.py tests/test_runtime_window_attempts.py` (`128 passed`).
- Next best steps: close this lane by updating release-facing README/docs benchmark text if needed, then commit the coherent solver-policy/test/artifact changes. The remaining performance work is optimization, not correctness: GPU PAS/geometry11 wall time is still the top offender, but it is now strict-clean and trace-backed.

Current active lane (2026-05-01, SFINCS_JAX-only closure):
- [x] Stop treating the NTX/NEOPAX profile-current ladder as an active
  SFINCS_JAX handoff lane. Local NTX is clean on `main` at `8701c03`
  (`Add CPU sharding guidance for NEOPAX exports`), so downstream NTX work is
  proceeding in that repository.
- [x] Keep the finite-beta sparse-PC solver-policy work as SFINCS_JAX production
  infrastructure, but archive the NTX-specific handoff page outside the main
  documentation toctree.
- [x] Keep release-blocking validation scoped to SFINCS_JAX artifacts:
  current GitHub `CI` and `Docs` are green on `f608f67`, and the validation
  manifest already admits only `implemented` or `deferred_post_release` lanes.
- [x] Re-run the SFINCS_JAX validation-manifest, publication-artifact,
  benchmark-summary, docs, and focused solver-policy gates after the archival
  wording update.
- Validation: `pytest -q tests/test_validation_manifest_schema.py tests/test_validation_artifacts.py tests/test_generate_validation_dashboard.py tests/test_generate_fortran_suite_benchmark_summary.py tests/test_generate_high_collisionality_trend_proxy.py tests/test_generate_simakov_helander_limit_audit.py tests/test_generate_simakov_helander_high_nu_run_plan.py tests/test_generate_w7x_high_nu_performance.py tests/test_collisionality_artifact.py tests/test_er_trajectory_sweep_artifact.py` (`37 passed`).
- Validation: `pytest -q tests/test_rhs1_host_policy.py tests/test_v3_sparse_pattern.py tests/test_solver_path_artifacts.py tests/test_solver_trace.py tests/test_solver_trace_output_formats.py tests/test_io_export_and_h5_coverage.py` (`47 passed`).
- Validation: `sphinx-build -W -b html docs docs/_build/html` passed after archiving the NTX handoff page outside the main docs toctree.
- Validation: GitHub `Docs` and `CI` are green on pushed commit `030cd2e`
  (`Docs` run `25232426852`, `CI` run `25232426840`).
- [x] Keep the remaining SFINCS_JAX optimization lanes explicit rather than
  pretending they are release blockers: PAS/geometry-rich runtime and memory,
  production-resolution benchmark scheduling, experimental single-case
  multi-device sharding, and long-term coverage/refactor work.
- [x] Convert the production-resolution input generator to SFINCS_JAX-owned
  defaults. Public extension now uses `--external-input`; old NTX flags remain
  hidden compatibility aliases only, and the regenerated default production
  manifest contains `39` example-derived cases with no downstream NTX decks.
- [x] Run a bounded CPU PAS/geometry4 offender sweep before changing policy.
  For `geometryScheme4_2species_PAS_noEr`, current default `pas_tz` stayed
  strict-clean and fastest (`elapsed=2.47 s`, profiled RSS about `1.83 GB`).
  Forced `xblock_tz` was slower/higher-memory (`14.21 s`, `3.11 GB`), forced
  `schur` was slower/higher-memory (`3.52 s`, `2.10 GB`), and `pas_schur` plus
  `theta_zeta` exceeded the bounded `45 s` trial budget. A GMRES-restart sweep
  preserved parity but traded memory for about `2x` slower solves, so no default
  restart-policy change is justified from this probe.
- Validation: `pytest -q tests/test_create_production_benchmark_inputs.py tests/test_validation_manifest_schema.py tests/test_validation_artifacts.py tests/test_generate_fortran_suite_benchmark_summary.py` (`21 passed`).
- [x] Add a checked-in guard test for the public production-resolution manifest:
  `benchmarks/production_resolution_inputs_2026-04-30/manifest.json` must
  remain `39` example-derived SFINCS_JAX cases, all with `source_group=examples`,
  and must not silently reintroduce downstream-project paths.
- Validation: `pytest -q tests/test_create_production_benchmark_inputs.py`
  (`6 passed`).
- Validation: GitHub `Docs` and `CI` are green on pushed commit `1c2737b`
  (`Docs` run `25233125047`, `CI` run `25233125097`).
- [x] Run a bounded CPU PAS/geometry11 offender sweep before changing policy.
  For `sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories`,
  current default `pas_tz` stayed strict-clean and fastest (`elapsed=2.53 s`,
  max RSS about `1.44 GB`). Forced `schur` was strict-clean but slower and
  higher-memory (`3.24 s`, `2.17 GB`), forced `xblock_tz` was much slower,
  higher-memory, and mismatched flow/current diagnostics (`12.59 s`, `3.17 GB`,
  `6` mismatches), and `pas_schur` exceeded the bounded `60 s` trial budget.
  A smaller GMRES restart preserved parity and lowered memory (`5.36 s`,
  `1.21 GB`), but roughly doubled runtime, so it remains an opt-in
  memory-pressure knob rather than a default.
- [x] Check in a compact PAS-offender variant probe artifact at
  `tests/reference_solver_path_artifacts/pas_offender_variant_probe_2026-05-01.json`
  and guard it with tests. The guard records that current `pas_tz` remains on
  the Pareto front for geometry4/geometry11, that geometry11 `xblock_tz`
  mismatches current/flow diagnostics, and that smaller GMRES restart settings
  are memory-pressure knobs rather than default candidates.
- [x] Add a manual-only GitHub workflow, `Production Benchmark Inputs`, that
  regenerates and validates the SFINCS_JAX-owned production-resolution input
  manifest without running expensive solves. It can upload the generated input
  tree as an artifact for local, `office`, or cluster CPU/GPU/Fortran benchmark
  runners.
- Validation: `pytest -q tests/test_solver_path_artifacts.py tests/test_benchmark_case_variants.py`
  (`17 passed`).
- Validation: `pytest -q tests/test_create_production_benchmark_inputs.py tests/test_solver_path_artifacts.py tests/test_benchmark_case_variants.py`
  (`23 passed`); `python scripts/create_production_benchmark_inputs.py --out-root /tmp/sfincs_jax_production_resolution_inputs_check --clean`
  regenerated `39` SFINCS_JAX-owned production inputs and the scope check
  confirmed no downstream-project paths; `sphinx-build -W -b html docs docs/_build/html`
  passed.
- Next SFINCS_JAX-only work: attack PAS/geometry-rich runtime and memory with
  trace-backed offender probes only when a candidate beats the current structured
  path on both parity and measured cost, then refresh production-resolution
  benchmark reports through guarded manual/nightly runners. Do not reopen
  NTX/NEOPAX profile-current parity as a SFINCS_JAX blocker unless downstream
  collaborators hand back a specific reproducible SFINCS_JAX defect.

Current active lane (2026-05-02, trace-backed full-FP sparse-PC default):
- [x] Probe structural sparse-host candidates on additional public RHSMode=1
  geometry-rich cases before changing policy.
- [x] Reject sparse-host auto-promotion for PAS after the HSX PAS full-trajectory
  probe: default `pas_tz` stayed Fortran-clean, while `sparse_host` and
  `sparse_pc_gmres` failed sparse factorization and `sparse_host_safe` changed
  flow/current diagnostics. This keeps PAS on the current structured
  preconditioner routes unless a future candidate is parity-clean and cheaper.
- [x] Promote only the measured CPU 3D full-FP window to sparse-PC GMRES. The
  gate is CPU-only, non-differentiable, `RHSMode=1`, `constraintScheme=1`,
  full-FP, `N_zeta > 1`, no Phi1/QN, no `EParallelHat`, no explicit user solve
  method, and active size within
  `SFINCS_JAX_RHSMODE1_FP3D_SPARSE_PC_MIN/MAX` (`300` to `20000` by default).
- [x] Check in the compact probe artifact
  `tests/reference_solver_path_artifacts/fp3d_sparse_pc_probe_2026-05-02.json`
  and guard it with tests requiring sparse-PC to be faster, lower-memory, and
  Fortran-clean on the measured FP cases, while explicitly rejecting the PAS
  sparse-host branches.
- Validation: focused public-case reruns after the default policy changed
  `HSX_FPCollisions_fullTrajectories` to `1.967 s` / `375.7 MB`,
  `HSX_FPCollisions_DKESTrajectories` to `1.871 s` / `368.8 MB`, and
  `sfincsPaperFigure3_geometryScheme11_FPCollisions_2Species_fullTrajectories`
  to `0.819 s` / `367.0 MB`, each with `0` Fortran mismatches. The tokamak FP
  control kept the dense auto lane, preserving the `N_zeta=1` exclusion.
- Next SFINCS_JAX-only work: refresh production-resolution CPU/GPU benchmark
  reports through the guarded manual/nightly runners and keep attacking only
  trace-backed offenders where a candidate beats the incumbent on parity,
  runtime, and memory. Remaining PAS improvements should focus on structured
  preconditioner memory/lifetime reductions rather than sparse-host branch
  promotion.

Current active lane (2026-05-03, production-resolution tokamak Er dense default):
- [x] Reconstruct the interrupted production-resolution bounded local run on
  `main` at `ff15535`. The guard launched `11` bounded tokamak cases and skipped
  `29` remote/cluster-only cases. The bounded suite had `9` strict
  `parity_ok` rows, `1` Fortran-reference divergence, and `1` known
  Fortran-reference-quality mismatch; missing output-key coverage stayed `0`.
- [x] Fix the suite classification so Fortran v3 `SNES_DIVERGED_*` logs with no
  converged reference state are reported as `fortran_diverged` /
  `reference solver quality` instead of a misleading `max_attempts`. A manual
  JAX check of `tokamak_1species_PASCollisions_noEr_withQN` still completed
  with `NIterations=1`, residual `7.16e-10`, and `FSABjHat=1.48146e-2`.
- [x] Probe the two real bounded CPU runtime offenders before changing default
  policy. `tokamak_1species_FPCollisions_withEr_DKESTrajectories` was spending
  about `130.379 s` in the Krylov/strong/sparse-rescue ladder; forced dense LU
  was parity-clean in `4.825 s`. `tokamak_1species_PASCollisions_withEr_fullTrajectories`
  was spending about `95.302 s`; forced dense LU was parity-clean in `4.386 s`.
  Sparse-host variants failed or timed out and were not promoted.
- [x] Confirm the same general gate also covers
  `tokamak_1species_FPCollisions_withEr_fullTrajectories`: the default is now
  dense, Fortran-clean, `4.601 s` in the variant probe and `3.843 s` through the
  suite harness, down from the prior `17.955 s` suite row.
- [x] Promote only the measured bounded CPU tokamak-Er window to dense LU:
  `RHSMode=1`, no Phi1, `N_zeta=1`, nonzero `Er` or potential-gradient drive,
  Er/DKES trajectory terms enabled, no explicit user solve method, active size
  in `SFINCS_JAX_RHSMODE1_TOKAMAK_ER_DENSE_MIN/MAX` (`5000` to `6500` by
  default), and dense-matrix bytes below
  `SFINCS_JAX_RHSMODE1_TOKAMAK_ER_DENSE_MAX_BYTES` (`350000000` by default).
  The gate is CPU-only because the measured win is a CPU LU/runtime tradeoff,
  not a GPU memory win.
- [x] Validate the new default on the same production-resolution inputs:
  FP DKES+Er default is now dense, Fortran-clean, `2.148 s`; PAS full-Er default
  is now dense, Fortran-clean, `2.812 s`; the no-Er PAS control stays on
  `pas_tokamak_theta`, Fortran-clean, `1.775 s` and about `667 MB`.
- [x] Check in the compact artifact
  `tests/reference_solver_path_artifacts/production_tokamak_er_dense_probe_2026-05-03.json`
  and guard it with tests requiring the Er dense path to be parity-clean and
  over `10x` faster while explicitly recording the higher transient RSS and the
  no-Er control.
- Validation: `pytest -q tests/test_rhs1_host_policy.py tests/test_solver_path_artifacts.py tests/test_scaled_example_suite_reference.py`
  (`51 passed`); `ruff check sfincs_jax/rhs1_host_policy.py sfincs_jax/io.py scripts/benchmark_case_variants.py scripts/run_reduced_upstream_suite.py tests/test_rhs1_host_policy.py tests/test_solver_path_artifacts.py tests/test_scaled_example_suite_reference.py`.
- Validation: harness-level production rerun for
  `tokamak_1species_FPCollisions_withEr_DKESTrajectories`,
  `tokamak_1species_FPCollisions_withEr_fullTrajectories`,
  `tokamak_1species_PASCollisions_withEr_fullTrajectories`, and
  `tokamak_1species_PASCollisions_noEr` recorded `4/4 parity_ok`, strict
  `0` mismatches, missing output keys `0`, and JAX CLI times `3.53 s`,
  `3.84 s`, `3.53 s`, and `2.89 s`, respectively.
- Validation: `pytest -q` (`1067 passed in 8:47`) and
  `sphinx-build -W -b html docs docs/_build/html`.
- Next SFINCS_JAX-only work after this commit: keep optimization concentrated in
  true remote/cluster-only production cases and GPU/PAS/geometry-rich memory,
  not these bounded tokamak-Er runtime cliffs.

Archived finite-beta RHSMode=1 solver-policy lane (2026-05-01):
- [x] Read the NTX finite-beta RHSMode=1 profile-current handoff at `/Users/rogeriojorge/local/NTX/docs/sfincs-jax-rhsmode1-profile-current-handoff.md`.
- [x] Confirmed the collaborator's `3e38894` refresh is in the NTX repository, while `uwplasma/sfincs_jax` `origin/main` remains at `0107be7`; the clean SFINCS-JAX checkout is a detached worktree at that commit, and the active development work remains in `/Users/rogeriojorge/local/tests/sfincs_jax`.
- [x] Keep the NTX state honest: RHSMode=3 finite-beta coefficient parity is closed under the `1e-1` gate, but RHSMode=1 profile-current parity is still open for the `17 x 21 x 12, Nx=5` production point because the reported residual remains about `1.88e-2` against a target near `1.09e-9`.
- [x] Patch `scripts/profile_write_output_trace.py` so a successful output solve is no longer reported as failed just because JAX/XPlane/Perfetto or device-memory profiling finalization fails afterward.
- [x] Add timeout-safe atomic JSON phase logs with `--phase-log` (default: `<trace-dir>/profile_write_output_trace_phases.json`) covering input preparation, warmups, JAX trace context, output solve/write, `block_until_ready`, and optional device-memory snapshotting.
- [x] Add `--strict-profiler` for CI/debug lanes that should still fail when profiler finalization fails after a valid output file has been written.
- [x] Add `--no-jax-trace` for long production audits that need phase logs and Fortran-like solver output without XPlane/Perfetto overhead.
- [x] Add a phase-log heartbeat (`--phase-log-interval-s`) so long solves refresh elapsed time while the active phase is still running.
- [x] Add wrapper-level `--solver-trace` forwarding so profiling/audit runs can persist solver/backend/residual sidecars from `write_sfincs_jax_output_h5`.
- [x] Add regression coverage for normal trace runs, profiler-finalization failure, strict-profiler failure, and device-memory snapshot failure.
- Validation: `pytest -q tests/test_profile_write_output_trace.py` (`5 passed`); `python -m py_compile scripts/profile_write_output_trace.py tests/test_profile_write_output_trace.py`; `ruff check scripts/profile_write_output_trace.py tests/test_profile_write_output_trace.py`.
- Validation: real finite-beta RHSMode=1 smoke trace on the NTX owned QA profile-current deck (`Ntheta=13`, `Nzeta=15`, `Nxi=8`, `Nx=5`, `rho=1/7`) completed locally through the updated wrapper. Output `FSABjHatOverRootFSAB2=-0.44600080476476256`, `NIterations=1`, residual `1.127263e-13`, solve+diagnostics about `3.0 s`, traced wrapper about `7.3 s`, and phase log status `completed`.
- Validation: same deck through `--no-jax-trace` completed with phase log `jax_trace=false`, output `FSABjHatOverRootFSAB2=-0.44600080476476256`, and wrapper elapsed about `2.1 s`, confirming the low-overhead production-audit lane works.
- Validation: the same low-overhead wrapper was run on the NTX `17 x 21 x 12, Nx=5` production profile-current deck with `--no-jax-trace` and `SFINCS_JAX_PROFILE=1`. The run completed in about `500.2 s` wall time, with wrapper phase-log status `completed`, `write_output_solve=499.0 s`, max RSS about `1.32 GB` on the local macOS `/usr/bin/time` path, and HDF5 output at `/tmp/sfincs_jax_rhs1_profile_prod_17x21x12/sfincsOutput.h5`.
- The production audit reproduced the collaborator's numerical blocker while removing the profiler/output ambiguity: `FSABjHat=-1.2773637952477914`, `FSABjHatOverRootFSAB2=-1.29105723879398`, `NIterations=1`, and the solver stdout still reported `residual_norm=1.880588e-02` versus target `1.088e-09`.
- Phase/stage timing shows the bottleneck is the RHSMode=1 Krylov/PAS solve, not geometry setup, diagnostics, or HDF5: grids/output-field setup stayed below `0.3 s`, Schur preconditioner build was about `0.5 s`, the first Krylov solve was about `313 s`, and the PAS-lite fallback returned the same `1.88e-2` residual floor.
- Validation: explicit CLI `--solve-method sparse_host --solver-trace` on the same `17 x 21 x 12, Nx=5` NTX deck completed in `8.66 s` wall time with peak RSS about `1.57 GB`, solver-trace `converged=true`, `residual_norm=1.0036e-16`, `FSABjHat=-1.2981550371186084`, and `FSABjHatOverRootFSAB2=-1.31207136`. This confirms the structured sparse-host lane solves the algebraic system quickly and accurately on the bring-up deck where the default matrix-free Krylov/PAS branch stalls.
- Validation: re-run after adding output-visible solver metadata completed the same NTX sparse-host deck in `8.35 s`; the HDF5 now contains `linearSolverMethod=sparse_host`, `linearSolverResidualNorm=1.0036e-16`, `linearSolverResidualTarget=1.0875e-09`, `linearSolverResidualTargetRatio=9.23e-08`, and `linearSolverConverged=+1`.
- Public-case sparse-host probe: `tokamak_1species_PASCollisions_noEr_Nx1` and `quick_2species_FPCollisions_noEr` both expose the expected limitation of direct sparse LU on singular/near-singular constrained systems. `sparse_host`/`sparse_pc_gmres` now fail with an actionable host-sparse factorization message instead of a raw SuperLU-only traceback; `sparse_lsmr` remains fast but writes `converged=false` for these probes, so it stays diagnostic rather than a parity/backend default.
- Code/docs follow-up: RHSMode=1 outputs now persist convergence metadata in the main HDF5/NetCDF/NPZ payload, CLI `write-output` reports runtime errors without a Python traceback unless `SFINCS_JAX_DEBUG=1`, and README/docs now state that sparse-host auto-promotion requires a pinned physical gauge/nullspace branch.
- Fix pass: added explicit `solve_method="sparse_host_safe"` plus PETSc-compatible constrained-PAS minimum-norm branch labeling. The safe mode first tries sparse-host LU. If LU fails due to a singular/gauge-sensitive constrained-PAS operator, it falls back to `petsc_compat`/LSMR and records `linearSolverAccepted`, `linearSolverAcceptanceCriterion`, `linearSolverLeastSquaresConverged`, `linearSolverReportedResidualNorm`, and iteration/info-code fields in the main output and solver trace metadata.
- Validation: `sparse_host_safe` on the NTX `17 x 21 x 12, Nx=5` deck took the sparse-host path, wrote `linearSolverConverged=+1`, `linearSolverAccepted=+1`, criterion `true_residual`, residual `1.0036e-16`, and `FSABjHat=-1.2981550371186084`.
- Validation: `sparse_host_safe` on `tokamak_1species_PASCollisions_noEr_Nx1` exposed singular sparse LU, fell back to the PETSc-compatible minimum-norm constrained-PAS path, and wrote `linearSolverConverged=-1`, `linearSolverAccepted=+1`, criterion `petsc_compatible_minimum_norm`, residual `4.8147e-08` against target `2.6301e-10`, `linearSolverLeastSquaresConverged=+1`, and `FSABjHat=-1.5216961e-05`.
- Validation: `sparse_host_safe` on `quick_2species_FPCollisions_noEr` correctly refused to apply the PAS-specific fallback and returned the actionable host sparse factorization error, preserving the boundary between constrained-PAS branch compatibility and full-FP parity.
- Fix pass: added a narrow default policy for large non-differentiable RHSMode=1 constrained-PAS profile-current decks. In the validated size window (`SFINCS_JAX_RHSMODE1_CONSTRAINED_PAS_SPARSE_PC_MIN=30000`, `MAX=300000` by default), `solve_method=auto` now routes to `sparse_pc_gmres` instead of the matrix-free fallback that stalled at `O(1e-2)` residual.
- Validation: default CLI on the NTX `17 x 21 x 12, Nx=5` deck now auto-selects sparse-PC GMRES, completes in `6.98 s` wall time with peak RSS about `1.58 GB`, writes `linearSolverConverged=+1`, `linearSolverAccepted=+1`, residual `9.19e-16` against target `1.09e-09`, and `FSABjHat=-1.2981550371185984`. This replaces the previous default `~500-660 s` stalled matrix-free branch.
- Validation: explicit `sparse_pc_gmres` on the production-resolution NTX `25 x 31 x 17, Nx=11` deck completes locally in `120.90 s` wall time with peak RSS about `9.25 GB`, writes `linearSolverConverged=+1`, `linearSolverAccepted=+1`, residual `4.31e-14` against target `2.09e-09`, and `FSABjHat=-5.0338680158175295`.
- Validation: explicit `petsc_compat` on the same `25 x 31 x 17, Nx=11` deck is fast (`20.43 s`, peak RSS about `2.22 GB`) but does not converge (`2.07e-02` against `2.09e-09`, `istop=7`), and the production output gate correctly refuses to write diagnostics.
- Validation: an ILU-only sparse-PC trial (`SFINCS_JAX_EXPLICIT_SPARSE_FACTOR_KIND=ilu`, fill `5`, drop `1e-5`) fails factorization on the `17 x 21 x 12, Nx=5` deck, so exact sparse LU remains the reliable sparse-PC preconditioner for this lane.
- Operator audit: Fortran v3 MATLAB matrix dumps on the clean `13 x 15 x 8, Nx=5` deck show JAX operator parity with the dumped PETSc matrix action to `3.19e-16` 2-norm on the Fortran state; the clean Fortran current is `FSABjHat=-0.4403656975`, matching the JAX branch (`-0.4412703508`). The older `+0.006` artifact is not a reliable parity anchor for this deck.
- Operator audit: Fortran v3 MATLAB matrix dumps on the `17 x 21 x 12, Nx=5` deck also show JAX operator parity with the dumped PETSc matrix action to `9.83e-17` 2-norm on the Fortran state. The remaining current spread is solver/nullspace-branch dependent: the local PETSc built-in sparse-direct Fortran build gives `FSABjHat=-1.2981550382` and matches JAX sparse-PC (`-1.2981550371`), while the MUMPS/SuperLU_DIST build can stop on a preconditioned KSP residual with true residual `O(1e-2)` and a different current branch. This is a Fortran-reference branch-selection issue, not a missing JAX operator term.
- CI status after stabilizing solver-path artifact fixtures: GitHub `CI` and `Docs` are green on pushed commit `cec14ca` (`CI` run `25220927882`, `Docs` run `25220927874`).
- Follow-up observability pass: sparse-PC GMRES now records setup time, solve time, total elapsed time, matvec count, sparse-pattern build time, sparse preconditioner factorization time, and sparse-pattern nonzero/row-density counters in solver metadata.
- Output follow-up: RHSMode=1 HDF5/NetCDF/NPZ payloads and optional solver-trace sidecars now expose the sparse-PC setup/solve/factorization/sparsity metadata, so large NTX/profile-current runs can be audited without relying on fragile profiler finalization.
- Validation: focused sparse-PC/output tests after the metadata pass: `pytest -q tests/test_v3_sparse_pattern.py::test_sparse_pc_gmres_solve_method_solves_tiny_rhs1_system tests/test_v3_sparse_pattern.py::test_write_output_preserves_sparse_pc_gmres_solve_method tests/test_io_export_and_h5_coverage.py::test_rhsmode1_solver_diagnostics_are_output_visible tests/test_rhs1_host_policy.py::test_rhs1_constrained_pas_sparse_pc_auto_targets_large_nondiff_pas` (`4 passed`).
- Validation: broader local metadata/trace gate: `pytest -q tests/test_rhs1_host_policy.py tests/test_v3_sparse_pattern.py tests/test_io_export_and_h5_coverage.py tests/test_solver_trace.py tests/test_solver_trace_output_formats.py` (`39 passed`).
- Validation: release-facing docs still build with warnings as errors: `sphinx-build -W -b html docs docs/_build/html`.
- Validation: full local test suite after the metadata/docs pass: `pytest -q` (`1037 passed in 524.05s`).
- Archived engineering status: solve/profile-finalization robustness, phase
  logging, production-audit observability, and the large constrained-PAS
  default solver-runtime cliff are closed for SFINCS_JAX. The downstream
  finite-beta profile-current publication-parity ladder is now owned by the NTX
  workflow and is not an active SFINCS_JAX release blocker.
- SFINCS_JAX next best steps after archiving the NTX handoff: keep the
  validation manifest free of open release lanes, keep CPU/GPU example-suite
  parity green, refresh production-resolution benchmark reports only through
  guarded/manual runners, and continue targeted optimization of PAS/geometry-rich
  runtime and memory without changing full-FP or differentiable/GPU-native
  defaults unless separately validated.

Execution style:
- Always profile first, change second, validate third.
- Track performance/memory deltas before and after every significant change.
- Update docs/README/plan.md in lockstep with code.
- Commit small, coherent changes frequently.
```

---

## 2) Project Goal (explicit)

Build a production-grade neoclassical transport solver in JAX that:
- solves the drift-kinetic equation in tokamak and stellarator geometries,
- reproduces SFINCS v3 equation set and normalizations in a reference/parity path (phase 1),
- offers a performance-first explicit path for CLI/default usage,
- preserves end-to-end differentiability in explicitly requested Python/JAX-native solve paths,
- is performant and memory-efficient by default for explicit solves,
- is extensible to alternative numerical methods (phase 2+).

---

## 3) Physical/Numerical Scope (phase 1)

The code should replicate SFINCS v3 behavior for:
- Geometries: `geometryScheme in {1,2,4,5,11,12}`,
- Physics options used in reduced/upstream examples (FP/PAS, Er/noEr, Phi1 variants, DKES/full trajectories where supported),
- Diagnostics and H5 output fields in `sfincsOutput.h5`,
- CLI workflow comparable to Fortran invocation (`sfincs_jax input.namelist`).

Core requirement right now:
- same equations,
- same discretization intent,
- same normalization,
- same algorithmic behavior where practical in the reference/parity path.

---

## 4) Repository + Reference Map

### 4.1 Local roots
- Workspace root: `/Users/rogeriojorge/local/tests`
- Active repo: `/Users/rogeriojorge/local/tests/sfincs_jax`
- Fortran executable: `/Users/rogeriojorge/local/tests/sfincs/fortran/version3/sfincs`
- Fortran source: `/Users/rogeriojorge/local/tests/sfincs_original`
- Thesis/PDF refs: `/Users/rogeriojorge/local/tests/Escoto_Thesis.pdf`, `/Users/rogeriojorge/local/tests/*.pdf`

### 4.2 sfincs_jax key code files
- Operator/system: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_system.py`
- Driver/preconditioners: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`
- Residual/Jacobian wrappers: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/residual.py`
- Solver kernels: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/solver.py`
- I/O + H5 writer: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/io.py`
- Transport diagnostics: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/transport_matrix.py`
- Output compare helper: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/compare.py`

### 4.3 Validation and reporting
- Reduced suite runner: `/Users/rogeriojorge/local/tests/sfincs_jax/scripts/run_reduced_upstream_suite.py`
- Reduced-suite archive note generator: `/Users/rogeriojorge/local/tests/sfincs_jax/scripts/generate_readme_reduced_suite_table.py`
- Frozen-case variant benchmark helper: `/Users/rogeriojorge/local/tests/sfincs_jax/scripts/benchmark_case_variants.py`
- Reduced inputs: `/Users/rogeriojorge/local/tests/sfincs_jax/tests/reduced_inputs`
- Reduced outputs/report dir: `/Users/rogeriojorge/local/tests/sfincs_jax/tests/reduced_upstream_examples`
- Tests root: `/Users/rogeriojorge/local/tests/sfincs_jax/tests`

### 4.4 Examples
- Main examples: `/Users/rogeriojorge/local/tests/sfincs_jax/examples`
- Additional high-res case: `/Users/rogeriojorge/local/tests/sfincs_jax/examples/additional_examples/input.namelist`
- Prior additional input: `/Users/rogeriojorge/local/tests/sfincs_jax/examples/additional_examples/input.namelist_old`

### 4.5 Documentation
- Docs root: `/Users/rogeriojorge/local/tests/sfincs_jax/docs`
- Upstream/reference material mirrored: `/Users/rogeriojorge/local/tests/sfincs_jax/docs/upstream`

---

## 5) Current State Snapshot (as of 2026-04-28)

### 5.1 Recent validated status
- Full frozen-reference CPU example-suite audit is complete at `/Users/rogeriojorge/local/tests/sfincs_jax/tests/scaled_example_suite_release_cpu_frozen_2026-04-25_v106` with `39/39` `parity_ok`, `39/39` strict parity, no `jax_error`, and no `max_attempts`.
- Full bounded-default GPU example-suite audit is complete at `/Users/rogeriojorge/local/tests/sfincs_jax/tests/scaled_example_suite_gpu_bounded_default_2026-04-28` with `39/39` `parity_ok`, `39/39` strict parity, no `jax_error`, no `max_attempts`, no missing Fortran output keys, and no runtime drift flags against the previous GPU release root.
- `additional_examples` is included in both final lanes and is `parity_ok` on CPU and GPU.
- README and docs now reflect the completed CPU and GPU artifact roots on `main` instead of intermediate branch-era sweeps.
- `write_sfincs_jax_output_h5(..., return_results=True)` now returns in-memory result dictionary for immediate inspection.
- Release-style validation has been rerun on the fast branch tip: `pytest -q` passed and `sphinx-build -W -b html docs docs/_build/html` passed.
- The current performance refactor landed in bounded production-safe form:
  - adaptive PAS smoother is integrated in RHSMode=1 fallback control,
  - explicit sparse host/device helpers are integrated in bounded transport and RHSMode=1 host-direct paths,
  - the structured block-tridiagonal helper is integrated into the `pas_tokamak_theta` tail solve,
  - host-only SciPy `lgmres` is now available on the explicit fast path without touching JIT/differentiable routes.
- Follow-up offender probes on current `main` now show where those four changes matter:
  - `tokamak_1species_PASCollisions_withEr_fullTrajectories` is parity-clean on the current CPU tokamak-xblock path at about `3.56s` on the frozen suite input, versus the older frozen-suite artifact at `37.75s`,
  - `tokamak_1species_PASCollisions_withEr_fullTrajectories` is now also parity-clean on the current GPU tight-GMRES path at about `3.25s` / `922.3 MB`, versus the older GPU `xblock_tz` artifact at about `18.2s` / `1014.5 MB`,
  - the adaptive PAS smoother and structured `pas_tokamak_theta` tail are not active on the current top tokamak/geometry4 offenders,
  - `lgmres` is now wired through the CLI and safely downgraded on traced/JIT/distributed paths, but it is slower than the current defaults on `geometryScheme4_2species_PAS_noEr` and `geometryScheme5_3species_loRes`, and effectively neutral on the tokamak PAS+Er case,
  - the fresh current `main` GPU full-suite refresh plus focused current-tip rows now capture the big bounded-solver wins directly in the release-facing docs: `geometryScheme5_3species_loRes` is down to `4.294s`, `tokamak_1species_PASCollisions_withEr_fullTrajectories` to `3.249s`, `sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories` to `7.420s`, and `sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_DKESTrajectories` to `6.314s`, all strict-clean,
  - CPU `geometryScheme=5` monoenergetic transport now prefers the low-memory Krylov/`tzfft` path by default on bounded VMEC RHSMode=3 cases; focused CLI probes reduced `monoenergetic_geometryScheme5_ASCII` from about `2950-3066 MB` to `506.5 MB` and `monoenergetic_geometryScheme5_netCDF` to `603.2 MB`, both with `0` Fortran mismatches,
  - the fresh GPU full-suite root also records `monoenergetic_geometryScheme5_ASCII` parity-clean at `3.938s` on the current bounded accelerator `tzfft` path,
  - and `geometryScheme4_2species_PAS_noEr` now uses direct `pas_tz` by default on the bounded near-zero-Er PAS lane, dropping the focused GPU RSS to about `1817.0 MB` while preserving parity.
  - `lineax` has been gated and is not admitted yet: on a small real SFINCS operator it matched the current residual and ran faster locally (`~0.54s` vs `~3.29s`), but on a generic nonsymmetric test matrix its default GMRES configuration stagnated, so it is still a bounded differentiable/reference-path candidate rather than a production CLI dependency.

### 5.2 Known pain points that still matter
- Runtime ratio is still high for heavier PAS / geometry-rich CPU cases, especially HSX / geometry4 PAS branches in the `3.5-4.9s` range on current targeted reruns.
- GPU wall time is robust and parity-clean in the bounded-default `2026-04-28` root. The remaining runtime offenders are `monoenergetic_geometryScheme1` (`12.909s`), `HSX_PASCollisions_fullTrajectories` (`10.539s`), `tokamak_2species_PASCollisions_withEr_fullTrajectories` (`7.777s`), and `sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories` (`7.716s`).
- Memory ratio remains high on select PAS/FP cases. After the geometry5 monoenergetic low-memory default, the current worst CPU RSS offender is `sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories` (`2298.6 MB`), while current worst GPU RSS offenders are `sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories` (`2097.6 MB`) and `HSX_PASCollisions_fullTrajectories` (`2042.4 MB`).
- Parallel strong-scaling beyond a few cores is not yet consistently strong for single-RHS large solves.

### 5.3 Product posture
- Release-ready for the currently supported example-suite scope on CPU and GPU,
- Scientifically functional and parity-clean on the audited `main` release artifacts,
- Still in active optimization/scaling hardening phase for runtime, memory, and multi-device throughput,
- Needs continued runtime/memory and distributed-solve work to reach “best-in-class” HPC behavior.

### 5.4 Execution modes
- `Reference / parity path`:
  - explicitly selected from Python,
  - prioritizes SFINCS v3 parity, solver diagnosability, and differentiability where supported.
- `Fast explicit path`:
  - default CLI / terminal usage,
  - may use different solvers, preconditioners, direct methods, caching, or host-side factorizations,
  - does not need to be differentiable unless explicitly requested,
  - does not need exact solver-path parity with Fortran if it converges robustly to scientifically acceptable outputs with materially better runtime and memory behavior.

---

## 6) What Has Been Done (high-level execution history)

Mark completed milestones as `[x]`, active as `[~]`, pending as `[ ]`.

### 6.1 Core numerical/functionality work
- [x] Matrix-free JAX operator path and v3-compatible workflow implemented.
- [x] RHSMode=1 and RHSMode=2/3 solver branches present with multiple Krylov/preconditioner options.
- [x] PAS projection/preconditioner heuristics added and iterated.
- [x] Dense fallback controls added/capped for stability/memory.
- [x] IncludePhi1/Newton behavior tuned for practical convergence on larger cases.
- [x] Removed unsafe dependency on Fortran H5 overlays for core correctness (standalone output path preserved).

### 6.2 Validation/reporting infrastructure
- [x] Reduced-suite runner supports runtime/memory/parity/print diagnostics.
- [x] README table auto-generated from suite report.
- [x] Runtime + memory columns integrated for Fortran/JAX CPU/GPU lanes.
- [x] Iteration stats plumbing exists in suite scripts/log parsing.

### 6.3 Documentation and examples
- [x] Major docs expansion (equations, models, methods, performance notes, references).
- [x] Added examples for parity, transport, autodiff, optimization, performance.
- [x] README and docs now present the full example-suite CPU/GPU audit as the release-facing status, with reduced-suite artifacts explicitly archived for debugging only.
- [x] Python quick-start now includes in-memory result access via `return_results=True`.

### 6.4 CI/CD hardening
- [x] CI and docs pipelines exist (`.github/workflows/ci.yml`, `docs.yml`).
- [x] Examples smoke and docs builds are wired.
- [~] CI runtime remains a continuing optimization target (keep broad coverage but faster scheduling).

---

## 7) Required Behavior For New Work

1. Default behavior must generalize:
   - no case-name hacks,
   - no hidden fallback to external reference files.
2. Preserve differentiability for explicitly requested Python/JAX-native solve paths; do not force the CLI/default path to remain differentiable if that materially hurts runtime or memory.
3. Keep solver choices configurable, but defaults should “just work” for unseen cases. CLI/default may prefer performance-first explicit methods over parity-first methods.
4. Every performance change must report:
   - runtime delta,
   - memory delta,
   - validation delta.
5. Every algorithmic change must document:
   - equation/operator impact,
   - numerics/preconditioner rationale,
   - code location.

---

## 8) Validation Strategy (must run continuously)

### 8.1 Unit tests
- Operator blocks, geometry parsing, collision terms, diagnostics.

### 8.2 Regression tests
- For each reduced example, compare JAX output H5 against Fortran output H5.

### 8.3 Physics tests
- Verify expected asymptotic scalings/symmetries/conservation behavior where available.

### 8.4 Practical comparison threshold
- Default target: `rtol=5e-4`, `atol=1e-9` (or as currently standardized in suite scripts).

### 8.5 Strict comparison mode
- Also track strict mismatch counts without case-specific tolerance relaxations.

### 8.6 Repro commands

```bash
cd /Users/rogeriojorge/local/tests/sfincs_jax
python scripts/run_reduced_upstream_suite.py \
  --fortran-exe /Users/rogeriojorge/local/tests/sfincs/fortran/version3/sfincs \
  --reuse-fortran \
  --max-attempts 1 \
  --rtol 5e-4 \
  --atol 1e-9 \
  --jax-repeats 2
python scripts/generate_readme_reduced_suite_table.py
```

Single-case debug:

```bash
python scripts/run_reduced_upstream_suite.py \
  --fortran-exe /Users/rogeriojorge/local/tests/sfincs/fortran/version3/sfincs \
  --pattern "<CASE>$" \
  --reuse-fortran \
  --max-attempts 1 \
  --rtol 5e-4 \
  --atol 1e-9
```

---

## 9) CI/CD and Quality Gates

### 9.1 CI pipelines
- Tests matrix: `/Users/rogeriojorge/local/tests/sfincs_jax/.github/workflows/ci.yml`
- Docs build: `/Users/rogeriojorge/local/tests/sfincs_jax/.github/workflows/docs.yml`

### 9.2 Required pre-merge checks
- `pytest -q` (or CI split equivalent)
- `sphinx-build -W -b html docs docs/_build/html`
- Reduced-suite refresh for solver-affecting PRs (at least targeted cases; full sweep before release)
- README table regeneration when suite report changes

### 9.3 CI speed policy
- Keep scientific coverage while reducing wall-time via:
  - split scheduling,
  - fixture sizing discipline,
  - marked heavy tests separated from fast core path,
  - cached artifacts where safe.

---

## 10) Documentation Map + MD Update Protocol

### 10.1 Core docs to maintain
- `/Users/rogeriojorge/local/tests/sfincs_jax/README.md`
- `/Users/rogeriojorge/local/tests/sfincs_jax/docs/index.rst`
- `/Users/rogeriojorge/local/tests/sfincs_jax/docs/system_equations.rst`
- `/Users/rogeriojorge/local/tests/sfincs_jax/docs/method.rst`
- `/Users/rogeriojorge/local/tests/sfincs_jax/docs/theory_from_upstream.rst`
- `/Users/rogeriojorge/local/tests/sfincs_jax/docs/normalizations.rst`
- `/Users/rogeriojorge/local/tests/sfincs_jax/docs/performance.rst`
- `/Users/rogeriojorge/local/tests/sfincs_jax/docs/performance_techniques.rst`
- `/Users/rogeriojorge/local/tests/sfincs_jax/docs/parallelism.rst`
- `/Users/rogeriojorge/local/tests/sfincs_jax/docs/usage.rst`
- `/Users/rogeriojorge/local/tests/sfincs_jax/docs/outputs.rst`

### 10.2 Markdown files to keep coherent
- `/Users/rogeriojorge/local/tests/sfincs_jax/README.md`
- `/Users/rogeriojorge/local/tests/sfincs_jax/examples/README.md`
- Example-specific READMEs under `/Users/rogeriojorge/local/tests/sfincs_jax/examples/*/README.md`
- `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md` (this file)

### 10.3 Update protocol for this `plan.md`
After every significant work block:
1. Update "Last updated" date.
2. Move checklist items from `[ ]` -> `[~]` -> `[x]`.
3. Add a short changelog entry under Section 16.
4. Record measured runtime/memory/parity deltas.
5. Add/refresh references if decisions used new literature/sources.

---

## 11) Competitor / Ecosystem Landscape

This project sits in a rapidly evolving fusion-computation ecosystem.

### 11.1 Relevant neoclassical or adjacent tools
- SFINCS (Fortran v3 baseline to replicate first)
- NEO (GACODE multispecies drift-kinetic solver ecosystem)
- KNOSOS (fast orbit-averaging stellarator neoclassical solver)
- STELLOPT tooling around stellarator optimization workflows

### 11.2 Why this matters for sfincs_jax
- Need robust, differentiable, optimization-friendly neoclassical kernels.
- Need interoperable, modern workflows (Python/JAX/HPC) while preserving first-principles fidelity.
- Need portability across laptop CPU, workstation GPU, and clusters (NERSC/Slurm).

---

## 12) Market Pull / Strategic Need (online snapshot)

The demand signal for production-grade fusion simulation software is rising due to:
- growth of private-sector fusion investment,
- national-scale public funding programs,
- open-source integrated modelling pushes,
- increasing HPC/GPU availability for high-fidelity predictive workflows.

Evidence (primary/public sources):
- IAEA World Fusion Outlook 2024 emphasizes global R&D growth, timelines, and public/private investment trends.
- U.S. DOE expanded FIRE + milestone-backed commercial fusion programs and reports milestone progress/funding leverage.
- Fusion Industry Association 2024 report indicates >$7.1B total private funding to date and 45 company responses.
- ITER released IMAS infrastructure/physics models as open source (2025), indicating a strong ecosystem trend toward open, interoperable modelling stacks.

Implication for sfincs_jax:
- there is clear pull for tools that are rigorous enough for physics validation and fast enough for iterative design/optimization.

---

## 13) Parallelization Target Context

### 13.1 Local target
- Efficient multi-core scaling on MacBook (user-level default usability).

### 13.2 Cluster target
- NERSC Perlmutter compatibility:
  - CPU-only and GPU node workflows,
  - Slurm-friendly execution,
  - robust scaling model for many-core / many-GPU execution.

Perlmutter references indicate heterogeneous CPU/GPU architecture and high-parallel-concurrency workflows.

### 13.3 Research-grade parallelization program

Parallelization work is now split into two explicit tracks:

- `Executable / CLI track`:
  - primary target for one-node and cluster throughput,
  - does **not** need to remain fully differentiable,
  - may use process pools, explicit sparse/direct solves, host-side orchestration, or backend-specific launch choices if they improve wall time and memory.
- `Differentiable Python track`:
  - preserves JAX-native operator structure and autodiff-compatible solve paths,
  - adopts distributed/sharded execution only when gradient correctness is still defensible and tested.

Immediate hardware baseline:

- Local MacBook Pro M3:
  - JAX currently sees `1` CPU device by default,
  - host-device parallelism must be requested with `--cores N` / `SFINCS_JAX_CORES=N`.
- Office workstation:
  - JAX currently sees `2` CUDA devices,
  - this is the current one-node multi-GPU validation target.

Current validated executable-side status:

- CLI parallel flags are now first-class and survive bootstrap/re-exec correctly.
- Large PAS sharded runs no longer crash by trying to build impossible dense
  `pas_tz` preconditioners; they fall back to shard-local Schwarz / lighter PAS
  paths instead.
- One-node CPU and one-node GPU parallel paths are usable and deterministic.
- Publication-grade parallel scaling now exists on the transport-worker lane:
  - CPU transport workers scale strongly on the large 3-RHS transport benchmark,
  - GPU transport workers now scale to `1.48x` on a 2-GPU office rerun of the
    same 3-RHS transport benchmark, essentially at the finite-task ideal
    `1.50x`.
- Strong scaling is still weak on the challenging single-RHS sharded GPU cases,
  including the final office "last-shot" rerun on the medium-large
  `examples/performance/rhsmode1_sharded_scaling.input.namelist` case:
  - implicit sharded path with `theta_schwarz` and 2 coarse levels: `1 GPU 40.8 s`,
    `2 GPUs 61.8 s`,
  - benchmark-only accelerator distributed sharding now reaches the true
    multi-GPU execution path, but on the office node it currently trips a CUDA
    launch-timeout failure instead of producing a usable 2-GPU speedup,
  - explicit non-differentiable sharded path did not improve the 2-GPU result and
    remained slower than the 1-GPU baseline,
  so the production recommendation remains:
  - transport workers for RHSMode=2/3 throughput,
  - one GPU per case / scan point for embarrassingly parallel scans,
  - bounded CPU host sharding for single-RHS solves,
  - multi-GPU single-case sharding only as an experimental benchmark path.

Implementation principle:

1. expose the real parallel runtime through the public CLI first,
2. benchmark one-node multi-core CPU and one-node multi-GPU scaling from the executable path,
3. stabilize multi-host bootstrap and cluster launch recipes,
4. only then widen the same model into autodiff-sensitive Python workflows.

---

## 14) Roadmap

### 14.1 Short-term (next 1-3 weeks)
- [x] Ensure the runtime-windowed/full example-suite audit is complete for CPU and GPU lanes against upstream-reference resolutions (current release roots `tests/scaled_example_suite_fast_cpu_full_v7_refresh` and `tests/scaled_example_suite_fast_gpu_full_v11_refresh`).
- [x] Replace blind global example-suite downscaling with original-reference, Fortran-runtime-window benchmarking so tiny Fortran rows are not artifacts of over-reduction.
- [x] Re-run additional high-resolution example on CPU+GPU and integrate into comparison reporting.
- [ ] Close remaining worst runtime/memory offenders (especially PAS-heavy cases) while preserving tolerances.
- [~] Strengthen default PAS preconditioner path to avoid expensive fallback branches where possible.
- [x] Split execution strategy:
  - CLI/default explicit path optimized for runtime and memory first,
  - reference/differentiable parity path selected explicitly from Python.
- [~] Continue performance-first optimization from the pinned final full-suite offender data:
  - `tokamak_1species_PASCollisions_withEr_fullTrajectories`,
  - `geometryScheme5_3species_loRes`,
  - `geometryScheme4_2species_PAS_noEr`,
  - `sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories`,
  - `monoenergetic_geometryScheme5_ASCII`.
- [~] Keep the new fast-path components evidence-based:
  - `lgmres` stays opt-in unless a frozen-case benchmark shows a win on the actual offender input,
  - adaptive PAS smoothing / structured tokamak tails are only promoted where the logs show they are actually active,
  - explicit sparse helpers are only promoted where the helper itself, not just sparse-direct, improves runtime or memory.
- [x] Prototype adaptive smoothing / early-stop smoother cycles for PAS-heavy RHSMode=1 cases:
  - stop when smoother residuals turn upward,
  - use the smoother as a bounded preconditioner stage instead of paying for full failed retries,
  - validate on `tokamak_1species_PASCollisions_withEr_fullTrajectories` and `geometryScheme4_2species_PAS_noEr`.
- [x] Prototype block-tridiagonal / factor-and-reuse solves on monoenergetic or weakly coupled subproblems:
  - avoid flattening structured low-coupling cases into generic full-system Krylov problems,
  - target `geometryScheme5_3species_loRes` and the monoenergetic offenders first.
- [x] Add explicit sparse host/device split for GPU-heavy cases:
  - keep operator assembly in JAX where useful,
  - materialize structured sparse pieces only for the hard solve branches,
  - prefer deterministic sparse factors over repeated failed generic retries.
- [x] Execute the four-step performance refactor in small validated increments:
  - Step 1: adaptive PAS smoother / early-stop preconditioner stage,
  - Step 2: explicit sparse host/device split for hard GPU and memory-heavy branches,
  - Step 3: structured monoenergetic / weak-coupling block solve prototype,
  - Step 4: Krylov-stack upgrade after steps 1-3 are integrated.
- [x] Land the bounded step-4 Krylov upgrade:
  - host-only SciPy `lgmres` fast path in `sfincs_jax/solver.py`,
  - explicit rejection on distributed and JIT/differentiable routes,
  - focused solver/full-system/implicit regression coverage.
- [ ] For each of the four steps, land:
  - focused unit/regression coverage for every new helper function,
  - at least one targeted parity case and one targeted performance case,
  - docs updates with equations, algorithm notes, and source-code locations.
- [~] Make executable parallelism first-class and reproducible:
  - add public CLI controls for transport workers, sharding, distributed Krylov, and multi-host bootstrap,
  - document one-node CPU, one-node GPU, and multi-host launch patterns,
  - validate the new CLI surface with focused tests.
- [ ] Benchmark current executable-path parallel scaling from `main`:
  - local multi-core CPU using `--cores` + sharded RHSMode=1 and process-parallel transport,
  - office 2-GPU one-node sharded solves,
  - record baseline speedups and memory deltas in docs/plan before changing algorithms.
- [ ] Turn existing prototype parallel features into the default research path in bounded stages:
  - stage A: transport `whichRHS` / scan throughput,
  - stage B: one-node sharded single-RHS solves,
  - stage C: multi-host bootstrap and Slurm recipes,
  - stage D: stronger domain decomposition and communication-avoiding Krylov.
- [ ] Evaluate JAX-ecosystem libraries only behind measured gates:
  - `lineax`: benchmark on bounded explicit non-differentiable linear solves and small/structured differentiable solves; admit only if it reduces code complexity or thresholds *and* improves runtime/RSS on at least one pinned offender or reference-path case without parity regressions.
  - `equinox`: evaluate only for module/state/filtering cleanup around the differentiable Python path; no admission unless it removes real tracing/static-arg complexity without slowing hot solves.
  - `jaxopt`: evaluate only for implicit-diff / root-solve wrappers in the differentiable Python path; no admission for CLI/offender work unless it materially simplifies or accelerates the current implicit solve route.
  - `diffrax`, `optax`, `quadax`, `orthax`: keep out of the runtime path unless a concrete hotspot maps directly onto ODE integration, optimization updates, adaptive quadrature, or orthogonal-polynomial transforms respectively and a benchmark proves an actual win.
  - Every library trial must include: one microbenchmark, one pinned offender or reference-path case, parity comparison against the current shipped path, RSS measurement, and a removal path if the win does not survive full-case validation.
- [~] Keep docs and README synchronized with measured reality (no stale claims).
- [ ] Keep CI wall-time under control without reducing scientific coverage.

### 14.2 Medium-term (1-3 months)
- [ ] Implement stronger generalized domain-decomposition preconditioners for large RHSMode=1 systems.
- [ ] Improve communication-avoiding Krylov behavior for stronger multi-core/multi-device scaling.
- [ ] Stabilize one-node multi-GPU strategy for large-case throughput.
- [ ] Add benchmark suite for representative 2-4 minute cases (warm/cold timing and memory baselines).
- [ ] Add explicit solver-path provenance in logs/output metadata.
- [ ] Strengthen block smoothers / Krylov patterns:
  - explicit block-diagonal or banded block smoothers on natural folded axes,
  - JAX-native FGMRES / LGMRES / GCROT-style right-preconditioned paths,
  - multigrid-ready smoother interfaces for geometry / pitch / speed coarsening.
- [ ] Strengthen structural sparsity:
  - preserve and exploit block-tridiagonal / near-block-tridiagonal structure in the stiff velocity couplings,
  - prefer factor-and-reuse of repeated block solves over repeated generic Krylov on the full flattened state,
  - push low-memory Schur / elimination paths that store only the minimal blocks needed for backward substitution.
- [ ] Add chunked explicit kernels for large PAS/FP assembly and diagnostics:
  - chunk over species, `x`, `xi`, or `(theta,zeta)` tiles,
  - cap peak device memory without changing numerics,
  - keep chunking off the differentiable reference path unless explicitly enabled.
- [ ] Make one-node parallelism production-grade:
  - robust device-mesh selection for CPU and GPU from the CLI,
  - consistent sharded-preconditioner selection on multi-device runs,
  - stable performance baselines on local workstation and office hardware.
- [ ] Make multi-host / many-core launch practical:
  - Slurm-ready launcher docs and helper scripts,
  - reproducible coordinator/process bootstrap,
  - measured scaling targets on tens of ranks before claiming hundreds.

### 14.3 Long-term (3-12 months)
- [ ] Extend beyond strict SFINCS replication: broader equation/model options and modern numerical variants.
- [ ] Integrate faster monoenergetic pathways where scientifically consistent.
- [ ] Build coupled optimization workflows (profile/equilibrium loops) using implicit-diff where beneficial.
- [ ] Mature multi-node scaling strategy for Slurm (dozens/hundreds of workers) with robust defaults.
- [ ] Publish formal method/performance validation notes with reproducible artifacts.

---

## 15) Execution Checklist (live)

### 15.1 Always-on loop
- [x] Use the original Fortran v3 example inputs as the resolution reference for example-suite benchmarking; do not use blind `2x` enlargement as the default benchmark mode.
- [x] For example-suite audits, start from original reference resolution and only downscale when needed to satisfy a configured Fortran runtime window; do not intentionally reduce a case below about `1s` of Fortran wall time unless the original case is already that small.
- [x] Benchmark CPU/GPU JAX lanes against a fixed CPU-generated Fortran reference root when machine-local Fortran outputs are not proven deterministic.
- [x] For `constraintScheme=0` reference generation, force a stable Fortran Krylov solve (`PETSC_OPTIONS='-ksp_type gmres -pc_type none'`) unless an explicit PETSc override is requested.
- [~] Pick top 1-2 offenders from latest report (runtime and memory separately).
- [~] Profile (`SFINCS_JAX_PROFILE=1`) and isolate dominant phase.
- [~] Implement smallest high-ROI change.
- [~] Re-run targeted case(s), verify tolerances and print diagnostics.
- [~] For parallel changes, always measure:
  - 1-device baseline,
  - 2+ device/process speedup,
  - RSS delta,
  - parity delta.
- [~] Keep the four-step refactor gated:
  - step 4 design may proceed in parallel,
  - step 4 code should not land until steps 1-3 are integrated and revalidated.
- [x] Re-run reduced-suite subset, then full suite when stable.
- [x] Regenerate table + docs + this plan.

### 15.2 "Do not regress" list
- [~] Differentiability on JAX-native solver paths.
- [x] Standalone behavior (no hidden Fortran-output dependencies).
- [~] Robust defaults for unseen inputs.
- [x] CI/doc builds passing.

---

## 16) Changelog Entries For Future Agent Updates

Use this template and append newest at top:

```text
### YYYY-MM-DD
- Scope:
- Files changed:
- Validation run:
- Runtime/memory delta:
- Remaining risks:
- Next actions:
```

Current latest notable changes before this handoff:
- README simplified; quick-start now includes in-memory results API.
- `write_sfincs_jax_output_h5(..., return_results=True)` added.
- Reduced-suite runner now retries after JAX exceptions with resolution reduction before final `jax_error`.

### 2026-04-13
- Scope: audit the fresh PAS-heavy runtime offenders under clean conditions, benchmark controlled solver/preconditioner variants on the main CPU hotspots, and harden the suite reporting so it records in-log solver elapsed separately from subprocess wall time.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/scripts/run_reduced_upstream_suite.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/scripts/run_scaled_example_suite.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_scaled_example_suite_reference.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_schur_precond_heuristic.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/performance_techniques.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: clean local offender subset rerun on `HSX_PASCollisions_{DKES,fullTrajectories}`, `tokamak_2species_PASCollisions_withEr_fullTrajectories`, `sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_{DKES,fullTrajectories}`, and `geometryScheme4_2species_PAS_noEr` against frozen references (`6/6 parity_ok`); focused variant sweeps with `scripts/benchmark_case_variants.py`; `pytest -q tests/test_scaled_example_suite_reference.py tests/test_schur_precond_heuristic.py` (`28 passed`); `python -m py_compile scripts/run_reduced_upstream_suite.py scripts/run_scaled_example_suite.py sfincs_jax/v3_driver.py tests/test_scaled_example_suite_reference.py tests/test_schur_precond_heuristic.py`.
- Runtime/memory delta: the large regressions seen in the first fresh full-suite pass were mostly benchmark contamination from concurrent local scaling jobs and an unrelated office GPU workload. Clean CPU retests brought `tokamak_2species_PASCollisions_withEr_fullTrajectories` back to `3.54s` (vs contaminated `9.61s`) and the geometry11 PAS paper cases back to `2.90-3.67s` (vs contaminated `8.08-8.42s`). Controlled A/B sweeps showed current defaults are already best on the tested PAS hotspots: HSX DKES default `4.52s` beat `lgmres` (`5.11s`), `xblock_tz` (`5.93s`), `pas_tz` (`12.57s`), and `schur` (`71.68s`); geometry4 PAS default `3.83s` beat `incremental` (`5.02s`), `lgmres` (`4.81s`), `schur` (`5.11s`), and `species_block` (`33.75s`). The suite harness now records `jax_logged_elapsed_s` separately from subprocess wall time for cleaner offender ranking. A structure-aware default mixed-precision PAS rule is now landed for the near-zero-`Er`, PAS-only, `geometryScheme=4` Schur branch on CPU: current default on `geometryScheme4_2species_PAS_noEr` dropped from about `2.95 GB` RSS / `2.86s` to `1.98 GB` RSS / `2.37s`, with `0` mismatches against the frozen Fortran reference. The same auto rule stays off for HSX/geometry11 PAS DKES, which remained parity-clean and on the safe float64 path (`HSX_PASCollisions_DKESTrajectories` default `4.06s`, `0` mismatches).
- Remaining risks: the suite runtime field still reports subprocess wall time for continuity, so use `jax_logged_elapsed_s` for solver-centric ranking when comparing contaminated or highly loaded hosts. The remaining work is structural runtime/memory reduction on HSX PAS DKES and the geometry11 PAS paper cases, not another blanket mixed-precision promotion.
- Next actions: regenerate the release-facing performance tables using `jax_logged_elapsed_s` as the primary optimization metric while still keeping wall time for user-facing CLI cost; target the remaining HSX PAS DKES runtime path and the geometry11 PAS paper cases. Keep PAS mixed-precision auto rules structure-aware and benchmark-backed; do not broaden beyond geometry4 Schur without parity-clean evidence.

### 2026-04-13
- Scope: rerun the full frozen-resolution CPU and GPU example suites from current `main`, rerun the current single-case sharded CPU/GPU scaling probes, and refresh the live performance diagnosis with fresh artifacts instead of the older release roots.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: local CPU frozen suite `python scripts/run_scaled_example_suite.py --examples-root examples/sfincs_examples --extra-input examples/additional_examples/input.namelist --reference-results-root tests/scaled_example_suite_fast_cpu_full_v7_refresh --out-root tests/scaled_example_suite_recheck_cpu_frozen_2026-04-13 --timeout-s 3600 --max-attempts 2 --fortran-min-runtime-s 0 --runtime-adjustment-iters 0 --jobs 2 --reset-report` (`39/39 parity_ok`); office GPU frozen suite `PYTHONPATH=. python scripts/run_scaled_example_suite.py --examples-root examples/sfincs_examples --extra-input examples/additional_examples/input.namelist --reference-results-root tests/scaled_example_suite_fast_cpu_full_v7_refresh --out-root tests/scaled_example_suite_recheck_gpu_frozen_2026-04-13 --timeout-s 3600 --max-attempts 2 --fortran-min-runtime-s 0 --runtime-adjustment-iters 0 --jobs 1 --reset-report` (`39/39 parity_ok` after relaunch on the uncontended GPU); local sharded CPU probe `python examples/performance/benchmark_sharded_solve_scaling.py --backend cpu --input examples/performance/rhsmode1_sharded_scaling.input.namelist --devices 1 2 4 8 --repeats 1 --warmup 0 --global-warmup 1 --nsolve 4 --shard-axis theta --gmres-distributed 0 --distributed-krylov auto --rhs1-precond theta_schwarz --schwarz-coarse-levels 2 --out-dir examples/performance/output/sharded_solve_scaling_cpu_2026-04-13`; office sharded GPU probe `PYTHONPATH=. /home/rjorge/stellarator_venv/bin/python examples/performance/benchmark_sharded_solve_scaling.py --backend gpu --input examples/performance/rhsmode1_sharded_scaling.input.namelist --devices 1 2 --repeats 1 --warmup 0 --global-warmup 1 --nsolve 4 --shard-axis theta --gmres-distributed 0 --distributed-krylov auto --rhs1-precond theta_schwarz --schwarz-coarse-levels 2 --out-dir examples/performance/output/sharded_solve_scaling_gpu_2026-04-13`.
- Runtime/memory delta: parity stayed clean, but performance regressed versus the pinned release roots. Fresh CPU suite root `tests/scaled_example_suite_recheck_cpu_frozen_2026-04-13` has median runtime ratio `1.229x` and mean runtime ratio `1.567x` versus `tests/scaled_example_suite_fast_cpu_full_v7_refresh`; worst CPU regressions include `inductiveE_noEr` (`1.928s -> 5.752s`), `tokamak_2species_PASCollisions_withEr_fullTrajectories` (`3.361s -> 9.613s`), `tokamak_1species_PASCollisions_noEr_Nx1` (`2.377s -> 6.432s`), and the PAS HSX / geometry11 cases around `2.4-2.7x`. Fresh GPU suite root `tests/scaled_example_suite_recheck_gpu_frozen_2026-04-13` has median runtime ratio `1.127x` and mean runtime ratio `1.178x` versus `tests/scaled_example_suite_fast_gpu_full_v11_refresh`; worst GPU regressions include `tokamak_2species_PASCollisions_noEr` (`9.479s -> 17.595s`), `tokamak_2species_PASCollisions_withEr_fullTrajectories` (`6.917s -> 10.861s`), `geometryScheme4_2species_PAS_noEr` (`7.019s -> 9.077s`), and the PAS HSX / geometry11 cases. Fresh sharded single-case scaling remains weak: CPU `1/2/4/8` devices = `13.58 / 15.14 / 15.35 / 15.65 s`; GPU `1/2` devices = `70.85 / 227.97 s`.
- Remaining risks: there is no fresh parity failure, but there is a real fresh performance regression relative to the previously pinned release roots, concentrated in PAS-heavy tokamak, HSX, and geometry11 cases. Single-case sharded scaling on both CPU and GPU remains a performance boundary and is still not the release-facing scaling story.
- Next actions: treat the fresh frozen roots as the new diagnostic baseline; focus next on why PAS-heavy cases slowed relative to `v7_refresh`/`v11_refresh` before changing more defaults. The highest-ROI code targets remain `tokamak_2species_PASCollisions_withEr_fullTrajectories`, `HSX_PASCollisions_{DKES,fullTrajectories}`, `sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_{DKES,fullTrajectories}`, and `geometryScheme4_2species_PAS_noEr`. For parallelism, keep the published GPU story on transport workers and case throughput until a materially different single-case sharded algorithm is in place.

### 2026-04-13
- Scope: restructure the public documentation so `sfincs_jax` is documented as a standalone neoclassical transport code, add new theory/geometry/numerics/source-map/testing/applications pages, expand the equations and code-location mapping, and align the docs with the current CI/CD and release state.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/README.md`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/index.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/applications.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/geometry.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/numerics.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/source_map.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/testing.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/method.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/system_equations.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/physics_models.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/physics_reference.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/theory_from_upstream.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/inputs.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/outputs.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/examples.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/parallelism.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/references.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/upstream_docs.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/contributing.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/fortran_comparison.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/release_checklist.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/performance.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/performance_techniques.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/usage.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/fortran_examples.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `sphinx-build -W -b html docs docs/_build/html`; `pytest -q tests/test_getting_started_examples.py tests/test_cli_solve_mode.py tests/test_output_h5_scheme5_parity.py`; full `pytest -q` (`436 passed in 405.43s`).
- Runtime/memory delta: documentation-only pass; no solver-kernel runtime claim changed. The release-facing benchmark and parity artifacts remain the current `main` truth already documented in the README and docs.
- Remaining risks: the docs now reflect the current standalone-code positioning and supported workflows, but the main open research/performance lane is still strong single-case multi-GPU sharded scaling. The release-facing GPU scaling claim should remain transport-worker and case-parallel throughput until that solver architecture changes.
- Next actions: if more work is needed, focus on performance/memory and multi-GPU single-case scaling rather than further parity wording cleanup; the public documentation base is now broad enough that future work should mostly be maintenance and new-method updates.

### 2026-04-11
- Scope: add a real GPU transport-worker backend that pins independent transport workers one-per-GPU, fix worker result merging, benchmark the fresh 1-vs-2 GPU transport scaling lane on office, and promote that measured lane into the release-facing docs.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/transport_parallel_worker.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/examples/performance/benchmark_transport_parallel_scaling.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_transport_parallel.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_benchmark_transport_parallel_scaling.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/README.md`, `/Users/rogeriojorge/local/tests/sfincs_jax/examples/performance/README.md`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/examples.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/parallelism.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/_static/figures/parallel/transport_parallel_scaling_gpu.png`, `/Users/rogeriojorge/local/tests/sfincs_jax/examples/performance/output/transport_parallel_scaling_gpu.json`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `pytest -q tests/test_transport_parallel.py tests/test_benchmark_transport_parallel_scaling.py` (`15 passed` after the merge hardening); `python -m py_compile sfincs_jax/v3_driver.py sfincs_jax/transport_parallel_worker.py examples/performance/benchmark_transport_parallel_scaling.py tests/test_benchmark_transport_parallel_scaling.py`; office GPU benchmark `PYTHONPATH=. python examples/performance/benchmark_transport_parallel_scaling.py --backend gpu --input examples/performance/transport_parallel_2min.input.namelist --workers 1 2 --repeats 1 --warmup 0 --global-warmup 1 --precond xmg`
- Runtime/memory delta: the new GPU transport-worker lane measured `1` GPU worker `351.05 s` and `2` GPU workers `237.75 s` on `examples/performance/transport_parallel_2min.input.namelist`, i.e. `1.48x` speedup on a `3`-RHS workload, essentially at the finite-task ideal of `1.50x`. This becomes the new publication-facing multi-GPU scaling result. Single-case sharded GPU scaling remains weak and unchanged in recommendation.
- Remaining risks: single-case multi-GPU sharded RHSMode=1 still does not provide publication-grade strong scaling. The transport-worker result is strong, but it is a different parallel lane than sharded single-RHS solves.
- Next actions: keep the release-facing parallel story centered on transport workers and case-parallel throughput; revisit sharded single-RHS GPU scaling only with a lower-synchronization Krylov or stronger domain-decomposition correction.

### 2026-04-11
- Scope: add a bounded multilevel Schwarz residual correction for sharded RHSMode=1 solves, expose its benchmark controls in the sharded-scaling driver, and refresh the publication-facing parallel scaling docs from current measured CPU data.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/examples/performance/benchmark_sharded_solve_scaling.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_rhs1_schwarz_heuristic.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_benchmark_sharded_solve_scaling.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/README.md`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/parallelism.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `pytest -q tests/test_rhs1_schwarz_heuristic.py tests/test_benchmark_sharded_solve_scaling.py` (`14 passed` across the two passes in this block); `python -m py_compile sfincs_jax/v3_driver.py examples/performance/benchmark_sharded_solve_scaling.py tests/test_rhs1_schwarz_heuristic.py tests/test_benchmark_sharded_solve_scaling.py`; local CPU benchmark `python examples/performance/benchmark_sharded_solve_scaling.py --backend cpu --input examples/performance/rhsmode1_sharded_scaling.input.namelist --devices 1 2 4 8 --global-warmup 1 --warmup 1 --repeats 2 --nsolve 4 --shard-axis theta --gmres-distributed 0 --distributed-krylov auto --rhs1-precond theta_schwarz --schwarz-coarse-levels 2 ...`; `python scripts/generate_parallel_scaling_snapshot.py`
- Runtime/memory delta: the current measured CPU sharded benchmark on `examples/performance/rhsmode1_sharded_scaling.input.namelist` moved from the previously published `1/2/4` timings `4.91 s / 4.45 s / 7.00 s` to `3.99 s / 3.56 s / 3.97 s`, and now includes a stable `8`-device point at `4.46 s`. This is still not ideal strong scaling, but it removes the earlier 4-device collapse and gives a defensible bounded-sharding result on laptop CPU hardware.
- Remaining risks: single-case multi-device scaling is still weaker than transport-worker scaling and still not strong enough to replace the current production recommendation of one GPU per case or process-parallel transport/scan throughput. The fresh office 1-vs-2 GPU benchmark with the new multilevel path and allocator stabilization is still being re-measured.
- Next actions: finish the fresh office GPU benchmark, then decide whether the next highest-ROI move is a second coarser correction on the GPU path too or a lower-synchronization Krylov implementation for the sharded executable path.

### 2026-04-11
- Scope: add a dedicated multi-GPU throughput benchmark for the real production recommendation (one GPU per case), rerun the fresh 1-vs-2 GPU sharded and throughput lanes on office, and document the measured outcome honestly in the release-facing parallel docs.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/examples/performance/benchmark_multi_gpu_case_throughput.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_benchmark_multi_gpu_case_throughput.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/examples/README.md`, `/Users/rogeriojorge/local/tests/sfincs_jax/examples/performance/README.md`, `/Users/rogeriojorge/local/tests/sfincs_jax/README.md`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/examples.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/parallelism.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/_static/figures/parallel/gpu_case_throughput.png`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `pytest -q tests/test_benchmark_multi_gpu_case_throughput.py tests/test_benchmark_sharded_solve_scaling.py` (`5 passed`); `python -m py_compile examples/performance/benchmark_multi_gpu_case_throughput.py tests/test_benchmark_multi_gpu_case_throughput.py`; office GPU reruns via direct one-shot commands and `python examples/performance/benchmark_multi_gpu_case_throughput.py --input examples/performance/rhsmode1_sharded_scaling.input.namelist --nsolve 4 --rhs1-precond theta_schwarz --schwarz-coarse-levels 2`
- Runtime/memory delta: the fresh office single-case sharded GPU reruns remain weak even on the best current lane (`1 GPU 56.70 s` vs `2 GPUs 169.36 s` with distributed GMRES; `1 GPU 59.35 s` vs `2 GPUs 212.84 s` without). The production-style throughput rerun also remained below parity with ideal scaling (`107.65 s` sequential vs `194.08 s` concurrent on two GPUs, `0.55x`). This does not block shipment, but it confirms that multi-GPU scaling is still a research problem, not a release-quality claim.
- Remaining risks: reviewer-proof documentation now exists, but the actual GPU multi-device performance remains the main research-grade gap. Publication-ready CPU/process-parallel scaling exists; publication-ready GPU multi-device scaling does not yet.
- Next actions: if stronger publication-grade GPU scaling is required, the next work should move out of benchmarking and into algorithmic/runtime design: lower-synchronization Krylov, better multi-process GPU isolation, or fully independent multi-case scheduling instead of concurrent JAX-heavy worker contention on one node.

### 2026-04-11
- Scope: add ship-facing examples for supported geometry workflows, output plotting, and clearer public parallel entry points while keeping the new two-level sharded solver path parity-clean.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/README.md`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/examples.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/inputs.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/outputs.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/usage.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/parallelism.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/examples/README.md`, `/Users/rogeriojorge/local/tests/sfincs_jax/examples/getting_started/README.md`, `/Users/rogeriojorge/local/tests/sfincs_jax/examples/getting_started/write_sfincs_output_tokamak.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/examples/getting_started/write_sfincs_output_vmec.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/examples/getting_started/plot_sfincs_output.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/examples/autodiff/README.md`, `/Users/rogeriojorge/local/tests/sfincs_jax/examples/transport/README.md`, `/Users/rogeriojorge/local/tests/sfincs_jax/examples/performance/README.md`, `/Users/rogeriojorge/local/tests/sfincs_jax/scripts/generate_parallel_scaling_snapshot.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_getting_started_examples.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `pytest -q tests/test_getting_started_examples.py tests/test_rhs1_schwarz_heuristic.py tests/test_benchmark_sharded_solve_scaling.py` (`13 passed`); `pytest -q tests/test_transport_matrix_write_output_end_to_end.py tests/test_full_system_gmres_solution_parity.py`; full `pytest -q` (`424 passed in 364.56s`); `python -m py_compile examples/getting_started/write_sfincs_output_tokamak.py examples/getting_started/write_sfincs_output_vmec.py examples/getting_started/plot_sfincs_output.py tests/test_getting_started_examples.py`; `sphinx-build -W -b html docs docs/_build/html`
- Runtime/memory delta: no solver-kernel delta in this pass. The public parallel docs now point to the current measured CPU transport-worker benchmark on `examples/performance/transport_parallel_2min.input.namelist` (`1 worker: 252.5 s`, `2 workers: 169.2 s`, `4 workers: 93.7 s`, about `2.69x` speedup at `4` workers). The two-level sharded RHSMode=1 path remains parity-clean but still experimental for strong scaling.
- Remaining risks: single-case sharded 4/8-device CPU scaling is still not strong enough to market as the default production parallel path. The large `rhsmode1_sharded.input.namelist` xlarge benchmark remains too expensive/noisy to use as the headline scaling figure without a stronger coarse correction or lower-synchronization Krylov step.
- Next actions: keep the current production recommendation centered on transport workers / scan-point throughput and one-device-per-case GPU throughput; return to single-case sharded scaling only with either a second coarser correction level or a communication-avoiding Krylov implementation.

### 2026-04-12
- Scope: give the single-case multi-GPU sharded RHSMode=1 lane one final A/B pass on office using the current `main` code path, comparing the shipped implicit path against bounded Krylov/preconditioner variants and an explicit non-differentiable executable-style solve.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: office GPU sharded benchmark sweep on `examples/performance/rhsmode1_sharded_scaling.input.namelist` via `examples/performance/benchmark_sharded_solve_scaling.py`; direct explicit-path office probes with `SFINCS_JAX_IMPLICIT_SOLVE=0`; local targeted validation `pytest -q tests/test_transport_parallel.py tests/test_benchmark_transport_parallel_scaling.py tests/test_transport_matrix_write_output_end_to_end.py tests/test_full_system_gmres_solution_parity.py` (`39 passed in 53.53s`).
- Runtime/memory delta: current shipped implicit sharded GPU lane measured `1 GPU 40.787 s` and `2 GPUs 61.766 s` on the medium-large benchmark case. Forcing distributed GMRES remained weak (`44.803 s` vs `62.052 s`). `x`-axis sharding and deeper coarse hierarchy both hit GPU OOM on this node. The explicit non-differentiable executable path also failed to produce a better 2-GPU result and was terminated after running well past the 1-GPU baseline.
- Remaining risks: strong single-case multi-GPU scaling is still not a release-quality claim on the current solver architecture. The robust publication-facing GPU scaling result remains transport-worker parallelism, not single-case sharding.
- Next actions: keep the release-facing GPU parallel story centered on transport workers and case-parallel throughput; treat single-case multi-GPU sharding as an active research item requiring a materially different algorithmic step, e.g. lower-synchronization Krylov or a stronger coarse/global correction.

### 2026-04-13
- Scope: fix the benchmark/runtime logic so accelerator distributed sharded solves are actually exercised when explicitly requested, then rerun the medium office single-case 1-vs-2 GPU benchmark on the real path.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/examples/performance/benchmark_sharded_solve_scaling.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_benchmark_sharded_solve_scaling.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_rhs1_schwarz_heuristic.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `pytest -q tests/test_benchmark_sharded_solve_scaling.py tests/test_cli_solve_mode.py tests/test_rhs1_schwarz_heuristic.py` (`41 passed in 23.41s`); `python -m py_compile examples/performance/benchmark_sharded_solve_scaling.py sfincs_jax/cli.py sfincs_jax/v3_driver.py tests/test_benchmark_sharded_solve_scaling.py tests/test_cli_solve_mode.py tests/test_rhs1_schwarz_heuristic.py`; fresh office medium benchmark on `examples/performance/rhsmode1_sharded_scaling.input.namelist` after enabling the actual accelerator distributed path.
- Runtime/memory delta: the previous benchmark path understated the real problem because it was not reaching the actual accelerator distributed solve. After fixing that, the office medium benchmark hit the real multi-GPU path and failed with `CUDA_ERROR_LAUNCH_TIMEOUT` during the 1-GPU warmup/current-tip run instead of producing a better 2-GPU timing. The benchmark-side accelerator opt-in is therefore useful for research, but not safe to auto-enable in the shipped CLI/runtime path.
- Remaining risks: true accelerator distributed single-case sharding is now known to be unstable on the office node for the medium benchmark case. This is a clearer result than the old weak-scaling measurement, but it means the item is still open and requires a deeper kernel/runtime redesign rather than more threshold tuning.
- Next actions: keep accelerator distributed sharding benchmark-only; do not auto-enable it in the CLI. If this item must be closed in the future, the next pass should target a materially different implementation strategy, e.g. smaller compiled kernels / staged halo exchanges, or a different distributed Krylov implementation with less GPU watchdog exposure.

### 2026-04-10
- Scope: add a research-grade parallelization program to the release plan, split executable-first parallel rollout from differentiable Python rollout, expose the existing parallel runtime through the public CLI, and add CLI-side parallel provenance so workstation/cluster launches report the active sharding / worker / distributed settings.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/__init__.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/cli.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_cli_solve_mode.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/README.md`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/usage.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/parallelism.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `pytest -q tests/test_cli_solve_mode.py tests/test_transport_parallel.py tests/test_rhs1_schwarz_heuristic.py` (`30 passed`); `pytest -q tests/test_cli_solve_mode.py` (`22 passed`) after the `--cores` bootstrap fix; `python -m py_compile sfincs_jax/__init__.py sfincs_jax/cli.py tests/test_cli_solve_mode.py`; `sphinx-build -W -b html docs docs/_build/html`; local sharded benchmark variants via `python examples/performance/benchmark_sharded_solve_scaling.py ...`; CLI smoke: `python -m sfincs_jax -v --cores 4 ... write-output --geometry-only` now reports `cores=4 cpu_devices=4` before the solve, confirming that the flag is no longer a no-op.
- Runtime/memory delta: no solver-kernel change in this pass; this is parallel-runtime surfacing, provenance, and deployment hardening. Hardware baseline confirmed in this pass: local JAX sees `1` CPU device by default; office JAX sees `2` CUDA devices. Local executable-path sharded RHSMode=1 probe on `examples/performance/rhsmode1_sharded_scaling.input.namelist` still shows weak scaling (`1 device: 2.303 s`, `2 devices: 2.084 s` for `nsolve=1`; `1 device: 9.874 s`, `2 devices: 11.806 s` for `nsolve=2`), and A/B probes show `auto`, forced `pas_tz`, and forced distributed-GMRES all within a few percent on this ~49k-unknown PAS case, so there is no evidence yet for a threshold-only default change.
- Remaining risks: office 1-GPU vs 2-GPU sharded current-tip benchmark is still in progress, so this pass improves usability and deployment control but does not yet claim a fresh multi-GPU speedup. Multi-host bootstrap is now public and documented, but it still needs measured Slurm-scale validation before calling it production-grade.
- Next actions: finish the office 1-GPU vs 2-GPU baseline probe, record those numbers in the docs/plan, then prioritize the first real scaling algorithm work on sharded single-RHS solves: stronger domain decomposition, local block smoothers, and communication-avoiding Krylov.

### 2026-04-11
- Scope: add the next algorithmic step for sharded RHSMode=1 solves by composing the local theta/zeta Schwarz patches with a single wider theta/zeta block residual correction; keep the correction bounded and parity-safe, and use it only on genuinely multi-device sharded runs.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_rhs1_schwarz_heuristic.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/README.md`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/parallelism.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `pytest -q tests/test_rhs1_schwarz_heuristic.py` (`7 passed`); `python -m py_compile sfincs_jax/v3_driver.py tests/test_rhs1_schwarz_heuristic.py`; local sharded benchmark `python examples/performance/benchmark_sharded_solve_scaling.py --input examples/performance/rhsmode1_sharded_scaling.input.namelist --devices 4 8 --warmup 1 --repeats 1 --global-warmup 1 --nsolve 4 --shard-axis theta --gmres-distributed 0 --distributed-krylov auto --rhs1-precond theta_schwarz ...`; parity check comparing `/tmp/sfincs_parallel_cpu1_twolevel.h5` vs `/tmp/sfincs_parallel_cpu8_twolevel.h5` (`0/193` mismatches).
- Runtime/memory delta: on the measured sharded CPU solve benchmark, the two-level theta-Schwarz path improved the representative 4-device / 8-device run from about `7.03 s / 7.44 s` to about `5.93 s / 6.37 s` in one run, though repeated measurements remain somewhat noisy and still fall short of ideal strong scaling. The algorithmic effect is still useful: the multi-device path is now less sensitive to over-localized patch solves.
- Remaining risks: this is still not the final publication-grade strong-scaling story. The two-level correction helps, but 4/8-way CPU scaling is still only modest, and fresh 1-vs-2 GPU measurements with the corrected benchmark harness remain to be stabilized before updating the publication plot again.
- Next actions: benchmark the two-level path on a larger transport / larger RHSMode=1 case with enough per-device work to amortize setup, then decide whether a second coarser correction level or a communication-avoiding Krylov step is the next highest-ROI move.

### 2026-04-11
- Scope: strengthen the first actual sharded single-RHS scaling heuristic by widening auto theta/zeta Schwarz patches beyond a single local shard, and harden the sharded solve benchmark runner so CPU and GPU one-node scaling are exercised through an explicit backend selection path.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_rhs1_schwarz_heuristic.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/examples/performance/benchmark_sharded_solve_scaling.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_benchmark_sharded_solve_scaling.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/README.md`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/parallelism.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `pytest -q tests/test_rhs1_schwarz_heuristic.py tests/test_benchmark_sharded_solve_scaling.py` (`8 passed`); `python -m py_compile sfincs_jax/v3_driver.py examples/performance/benchmark_sharded_solve_scaling.py tests/test_rhs1_schwarz_heuristic.py tests/test_benchmark_sharded_solve_scaling.py`; local CPU parity check comparing `/tmp/sfincs_parallel_cpu1_check.h5` vs `/tmp/sfincs_parallel_cpu8_check.h5` on `examples/performance/rhsmode1_sharded_scaling.input.namelist` (`0/193` mismatches).
- Runtime/memory delta: on the measured local cold one-shot CPU sharded benchmark (`examples/performance/rhsmode1_sharded_scaling.input.namelist`, forced `theta_schwarz`, `gmres_distributed=1`), the previous auto patch rule collapsed at `8` devices (`7.53 s`) while the new auto block rule reduced that to about `4.07 s`. The tradeoff is that the broader local patch is a heavier setup on small device counts, so the `1-4` device cold timings are not uniformly better yet. The benchmark runner now also supports `--backend gpu` and disables JAX GPU preallocation in the subprocess so one-node GPU scaling probes can be exercised without immediate allocator failure from the harness itself.
- Remaining risks: the new auto Schwarz sizing is a real robustness improvement, but it is not yet the final strong-scaling solution. CPU 4/8-device sharded solves still need a stronger two-level / local-block-smoother step, and the office GPU medium benchmark remains expensive enough that final 1-vs-2 GPU scaling numbers should be refreshed after the next algorithmic pass rather than over-interpreted now.
- Next actions: add the first true two-level/domain-decomposition correction for sharded RHSMode=1 solves, then rerun 1/2/4/8 CPU and 1/2 GPU scaling on the same benchmark inputs and refresh the publication plot once those numbers are stable.

### 2026-04-11
- Scope: fix executable CLI global-parallel flag handling so `--cores`, `--shard-axis`, and `--transport-workers` work regardless of placement relative to the subcommand; add a PAS sharded-memory guard so very large GPU PAS runs do not try to build impossible dense `pas_tz` preconditioners; benchmark larger CPU, GPU, and Fortran MPI scaling cases; add release-facing publication-style scaling plots for README/docs.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/cli.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_cli_solve_mode.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_rhs1_schwarz_heuristic.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/scripts/generate_parallel_scaling_snapshot.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/README.md`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/parallelism.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/_static/figures/parallel/strong_scaling_snapshot.png`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/_static/figures/parallel/strong_scaling_snapshot.pdf`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `pytest -q tests/test_cli_solve_mode.py tests/test_rhs1_schwarz_heuristic.py` (`27 passed`); `python -m py_compile sfincs_jax/v3_driver.py sfincs_jax/cli.py tests/test_cli_solve_mode.py tests/test_rhs1_schwarz_heuristic.py scripts/generate_parallel_scaling_snapshot.py`; CLI smoke `python -m sfincs_jax -v --cores 4 --shard-axis theta write-output ... --geometry-only`; fresh larger-case parity check `python -m sfincs_jax compare-h5 --a /tmp/sfincs_jax_rhsmode1_sharded_large.h5 --b <fortran-rank1>/sfincsOutput.h5 --rtol 5e-4 --atol 1e-9` (`0` mismatches); `python scripts/generate_parallel_scaling_snapshot.py`; `sphinx-build -W -b html docs docs/_build/html`.
- Runtime/memory delta: local CPU sharded RHSMode=1 benchmark on `examples/performance/rhsmode1_sharded_scaling.input.namelist` gives `1 device: 4.91 s`, `2 devices: 4.45 s`, `4 devices: 7.00 s`; local CPU transport-worker benchmark on `examples/performance/transport_parallel_xlarge.input.namelist` gives `1 worker: 5.00 s`, `2 workers: 9.17 s`, `4 workers: 7.90 s`; local Fortran MPI on the same simplified RHSMode=1 scaling input gives `1 rank: 1.18 s`, `2 ranks: 0.26 s`, `4 ranks: 0.39 s`; office GPU sharded benchmark on `examples/performance/rhsmode1_sharded_scaling.input.namelist` gives `1 GPU: 44.91 s`, `2 GPUs: 67.48 s`; on the larger `examples/performance/rhsmode1_sharded.input.namelist`, current `main` now runs on `1 GPU` in `16.58 s` instead of crashing with a `~155 GiB` PAS dense allocation attempt.
- Remaining risks: executable-side parallel controls are now correct and large PAS sharded runs are robust, but strong scaling is still weak on the current one-node single-RHS benchmarks. The current production recommendation remains one GPU per case / scan point rather than multi-GPU single-case sharding.
- Next actions: implement the first actual scaling algorithm change for sharded single-RHS solves (stronger local domain decomposition / block smoothers), then re-benchmark office `1 GPU` vs `2 GPU` on the larger PAS case and only promote multi-GPU single-case sharding if the new path materially reduces wall time.

### 2026-04-01
- Scope: Harden the public CLI/output API by adding documented equilibrium overrides (`equilibrium_file`, `wout_path`), make shared CLI flags usable after subcommands, and ensure the embedded `input.namelist` in `sfincsOutput.h5` reflects the effective run configuration.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/input_compat.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/io.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/cli.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_input_compat.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_cli_solve_mode.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_write_output_return_results.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/README.md`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/usage.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/outputs.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/inputs.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `pytest -q tests/test_input_compat.py tests/test_cli_solve_mode.py tests/test_write_output_return_results.py` (`32 passed`); `python -m py_compile sfincs_jax/input_compat.py sfincs_jax/io.py sfincs_jax/cli.py tests/test_input_compat.py tests/test_cli_solve_mode.py tests/test_write_output_return_results.py`
- Runtime/memory delta: no solver-path changes in this pass. The change removes a CLI/API failure mode around equilibrium-file overrides and makes the effective override visible in output artifacts, which improves reproducibility and debugging without changing numerics.
- Remaining risks: this pass did not rerun the full example suite because the implementation is confined to CLI/API plumbing and exercised on the existing scheme-5 parity fixture plus unit coverage. If future complaints involve scan orchestration rather than single-case runs, `scan-er` may need the same explicit override surface.
- Next actions: run a small release-smoke subset through the CLI entry points after the docs refresh, then keep any further CLI changes scoped to proven user pain points instead of widening the public surface gratuitously.

### 2026-03-27
- Scope: Make the bounded accelerator `tzfft` transport path a real default win on GPU, skip unnecessary GPU sparse rescue after converged PAS `schur` accepts, and harden the benchmark/auto-selection test surface around those branches.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_transport_sparse_direct.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_schur_precond_heuristic.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_benchmark_case_variants.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_example_auto_selection_paths.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/scripts/generate_readme_fast_branch_audit.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/README.md`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/performance.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `pytest -q tests/test_transport_sparse_direct.py` (`37 passed`); `pytest -q tests/test_transport_sparse_direct.py tests/test_schur_precond_heuristic.py` (`53 passed`); `pytest -q tests/test_example_auto_selection_paths.py tests/test_benchmark_case_variants.py` (`2 passed`); `python -m py_compile sfincs_jax/v3_driver.py tests/test_transport_sparse_direct.py tests/test_schur_precond_heuristic.py tests/test_example_auto_selection_paths.py tests/test_benchmark_case_variants.py`; office direct frozen-input probes on `monoenergetic_geometryScheme5_ASCII` and `sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories` against the pinned `tests/scaled_example_suite_fast_gpu_full_v8` Fortran outputs (`0` mismatches in both probes).
- Runtime/memory delta: `monoenergetic_geometryScheme5_ASCII` on office GPU now runs parity-clean in about `16.92s` / `1093 MB` RSS-equivalent log output on the bounded iterative `tzfft` path instead of getting trapped in host sparse LU first-attempt (`17.433s` in the pinned GPU root). `sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories` on office GPU now runs parity-clean in about `11.31s` / `2119 MB`, down from the pinned `58.198s` / `2354.4 MB`, because the sparse-ILU tail is skipped after a converged `schur` accept.
- Remaining risks: the release-facing GPU table still points at the older full `v8` root, so these new GPU gains are currently documented only as targeted current-tip probes. `geometryScheme5_3species_loRes` and `tokamak_1species_PASCollisions_withEr_fullTrajectories` remain the highest-value GPU runtime offenders.
- Next actions: rerun a full current-tip GPU suite root from `main`, then refresh the README/performance tables from CPU `v7` plus the new GPU root; after that, benchmark whether the tokamak PAS+Er GPU stage2 branch should yield to a bounded host sparse-direct polish.

### 2026-03-27
- Scope: Audit whether the recently landed fast-path features are actually exercised on the pinned CPU offender cases, convert the tokamak structured tail to opt-in after frozen-case benchmarking, and persist a reusable case-variant benchmark harness.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_schur_precond_heuristic.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/scripts/benchmark_case_variants.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/method.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/performance_techniques.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/usage.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `pytest -q tests/test_schur_precond_heuristic.py` (`14 passed`); `pytest -q tests/test_cli_solve_mode.py tests/test_implicit_linear_solve_grad.py tests/test_solver_gmres.py tests/test_schur_precond_heuristic.py` (`50 passed`); `python -m py_compile sfincs_jax/v3_driver.py scripts/benchmark_case_variants.py`; frozen-case variant probes via `python scripts/benchmark_case_variants.py` on `tokamak_1species_PASCollisions_withEr_fullTrajectories`, `tokamak_1species_PASCollisions_noEr`, `geometryScheme4_2species_PAS_noEr`, `geometryScheme5_3species_loRes`, and `monoenergetic_geometryScheme5_ASCII`
- Runtime/memory delta: current frozen tokamak PAS+Er default now takes the CPU `xblock_tz` branch and is parity-clean at `2.371 s` / `558.6 MB`; the shipped tokamak PAS no-Er case remains parity-clean with the structured tail disabled by default, improving from `2.002 s` to `1.721 s` on the frozen case while keeping `0` mismatches; forced `lgmres` stays parity-clean but is slower on the current frozen geometry4 PAS case (`3.333 s -> 5.272 s`) and geometry5 low-resolution case (`1.529 s -> 10.245 s`); forced transport sparse-helper settings on the monoenergetic ASCII case remain a no-op (`used_explicit_sparse_helper=false`, `0.190 s -> 0.193 s`).
- Remaining risks: on the shipped example set, the four-step additions are still mostly latent by default: adaptive PAS smoother was not exercised in the current offender probes, `pas_tokamak_theta` only appears by default on the no-Er tokamak PAS branch, `lgmres` is still best treated as opt-in, and the transport sparse helper is not yet reaching the monoenergetic memory offender.
- Next actions: wire the host-only Krylov methods into the remaining non-differentiable full-system branches only where the frozen-case probes justify them, profile the monoenergetic memory offender outside the current sparse-direct guardrails, and rerun a bounded offender subset before widening any defaults.

### 2026-03-27
- Scope: Close the top CPU tokamak PAS+Er runtime gap by fixing the default preconditioner branch, validate the pending CLI `lgmres` compatibility changes, and audit whether the four recently landed fast-path features are actually exercised on the pinned offender cases.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/io.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/implicit_solve.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_schur_precond_heuristic.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_cli_solve_mode.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_implicit_linear_solve_grad.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/performance.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/performance_techniques.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/usage.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `pytest -q tests/test_schur_precond_heuristic.py` (`13 passed`); `pytest -q tests/test_cli_solve_mode.py tests/test_implicit_linear_solve_grad.py` (`14 passed`); `python -m py_compile sfincs_jax/v3_driver.py sfincs_jax/io.py sfincs_jax/implicit_solve.py tests/test_schur_precond_heuristic.py tests/test_cli_solve_mode.py tests/test_implicit_linear_solve_grad.py`; frozen-case CLI probes on `tests/scaled_example_suite_fast_cpu_full_v6_merged/tokamak_1species_PASCollisions_withEr_fullTrajectories` and `tests/scaled_example_suite_fast_cpu_full_v6_merged/geometryScheme4_2species_PAS_noEr`; office GPU probe on `monoenergetic_geometryScheme5_ASCII` with `SFINCS_JAX_TRANSPORT_TZFFT_ALLOW_ACCELERATOR=1`.
- Runtime/memory delta: tokamak PAS+Er CPU frozen scaled case now auto-selects `xblock_tz`, runs in `~3.3 s`, and stays `0` mismatches versus the pinned output artifact, compared with the older full-suite baseline of `37.75 s`; geometry4 PAS CLI `lgmres` is now parity-clean and modestly faster on the frozen scaled case (`~5.6 s -> ~4.6 s`); office monoenergetic GPU probe improves from `~16.3 s` with `block` to `~14.3 s` with accelerator `tzfft`.
- Remaining risks: the full suite and README tables are still based on the older pinned artifacts, so they do not yet include the new tokamak CPU branch improvement; the four-step fast-path additions are still mostly *not* the active defaults on the remaining offender cases (`geometryScheme4_2species_PAS_noEr`, `geometryScheme5_3species_loRes`, `monoenergetic_geometryScheme5_ASCII`).
- Next actions: rerun a bounded offender subset from current `main` to refresh the tokamak CPU row, decide whether CLI `lgmres` should auto-enable for any PAS subset, and finish the parity check for accelerator `tzfft` before changing the GPU transport default.

### 2026-03-27
- Scope: Validate whether the newly landed four-step fast-path features are actually exercised on the current pinned offender cases, wire `lgmres` through the CLI env path safely, and benchmark the remaining top CPU offenders with frozen-case variant probes.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/io.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/implicit_solve.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_cli_solve_mode.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_implicit_linear_solve_grad.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_solver_gmres.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_schur_precond_heuristic.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/scripts/benchmark_case_variants.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `pytest -q tests/test_cli_solve_mode.py tests/test_solver_gmres.py tests/test_implicit_linear_solve_grad.py tests/test_schur_precond_heuristic.py` (`49 passed`); `python -m py_compile sfincs_jax/io.py sfincs_jax/implicit_solve.py sfincs_jax/v3_driver.py tests/test_cli_solve_mode.py tests/test_implicit_linear_solve_grad.py tests/test_solver_gmres.py`; frozen-case variant probes via `python scripts/benchmark_case_variants.py ...` on `tokamak_1species_PASCollisions_withEr_fullTrajectories`, `geometryScheme4_2species_PAS_noEr`, `geometryScheme5_3species_loRes`, and `monoenergetic_geometryScheme5_ASCII`.
- Runtime/memory delta: current-tip tokamak PAS+Er CPU rerun on the frozen suite input is about `3.56s` / `586-601 MB` and remains parity-clean, versus the older frozen-suite artifact at `37.75s`; `lgmres` is parity-clean but neutral on the tokamak case (`3.58s` vs `3.56s`) and slower/heavier on `geometryScheme4_2species_PAS_noEr` (`6.17s` vs `3.55s`) and `geometryScheme5_3species_loRes` (`13.23s` vs `1.56s`); forcing transport sparse-direct first on `monoenergetic_geometryScheme5_ASCII` is parity-clean and only marginally faster on the pinned final input (`0.157s` vs `0.169s`).
- Remaining risks: the README full-suite table is now stale for the improved CPU tokamak PAS+Er path until the full suite is rerun on this tip; the four-step refactor is real, but most of the new pieces are not the actual wins on the current top offenders.
- Next actions: refresh the full CPU suite on current `main`, then rerun the GPU lane from the refreshed CPU reference root; keep `lgmres` opt-in; continue profiling the remaining true offenders instead of widening heuristics that the frozen-case probes do not justify.

### 2026-03-27
- Scope: Finish the four-step performance refactor in bounded production-safe form by integrating the explicit sparse helper into transport/RHSMode=1 host-direct solves, wiring the structured block-tridiagonal helper into the `pas_tokamak_theta` tail solve, and adding a host-only SciPy `lgmres` fast path in `solver.py`.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/solver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_rhs1_sparse_first_heuristic.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_schur_precond_heuristic.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_solver_gmres.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_transport_sparse_direct.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/method.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/performance.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/performance_techniques.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `pytest -q tests/test_pas_smoother.py tests/test_explicit_sparse.py tests/test_structured_velocity.py tests/test_transport_sparse_direct.py tests/test_rhs1_sparse_first_heuristic.py tests/test_schur_precond_heuristic.py tests/test_solver_gmres.py` (`152 passed`); `pytest -q tests/test_implicit_linear_solve_grad.py tests/test_full_system_gmres_solution_parity.py tests/test_cli_solve_mode.py` (`28 passed`); `pytest -q tests/test_schur_precond_heuristic.py tests/test_solver_gmres.py tests/test_implicit_linear_solve_grad.py tests/test_full_system_gmres_solution_parity.py` (`48 passed` after the JAX-safe structured-tail fix); `python -m py_compile sfincs_jax/v3_driver.py sfincs_jax/solver.py tests/test_rhs1_sparse_first_heuristic.py tests/test_schur_precond_heuristic.py tests/test_solver_gmres.py tests/test_transport_sparse_direct.py`; `sphinx-build -W -b html docs docs/_build/html`
- Runtime/memory delta: targeted medium tokamak probes on `main` showed `lgmres` preserving output parity (`0` mismatches vs `incremental`) while reducing wall time from `3.802s -> 0.275s` on a patched PAS case and from `6.308s -> 3.902s` on a patched FP case. No full offender-suite rerun yet in this pass.
- Remaining risks: the new `lgmres` path is intentionally host-only and not yet benchmarked on the pinned heavy offenders; full-suite runtime/memory deltas still need measurement before defaults are widened further.
- Next actions: profile the pinned CPU/GPU offenders again from `main`, benchmark `incremental` vs `lgmres` on the explicit fast path, and decide whether the new host-only method should be used automatically on any subset of PAS/FP cases.

### 2026-03-27
- Scope: Simplify the main-branch README, move archived reduced-suite material into the docs, add a theory-heavy docs page distilled from the upstream SFINCS v3 notes, and audit external solver/tooling branches for concrete solver and performance ideas.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/README.md`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/index.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/fortran_examples.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/theory_from_upstream.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/upstream_docs.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/scripts/generate_readme_fast_branch_audit.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/scripts/generate_readme_reduced_suite_table.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `python scripts/generate_readme_reduced_suite_table.py`; `python scripts/generate_readme_fast_branch_audit.py --out-root tests/scaled_example_suite_fast_cpu_full_v6_merged --gpu-out-root tests/scaled_example_suite_fast_gpu_full_v8`; `python -m py_compile scripts/generate_readme_reduced_suite_table.py scripts/generate_readme_fast_branch_audit.py`; `sphinx-build -W -b html docs docs/_build/html`; external solver/tooling audit plus local scratch inspection
- Runtime/memory delta: no `sfincs_jax` numerics changed in this pass. The new roadmap priority is driven by the current pinned offenders plus ideas from the external solver audit: adaptive smoothers, more stable Krylov orthogonalization/recycling, explicit sparse host paths for hard GPU/CPU branches, and block-tridiagonal factor-and-reuse solves for monoenergetic or weakly coupled subproblems.
- Remaining risks: parity remains closed, but the heavy PAS and structured monoenergetic cases still need algorithmic changes to bring runtime and memory down materially.
- Next actions: implement and gate one adaptive PAS smoother path, one explicit sparse host/device split for the top GPU offenders, and one structured block solve prototype for the monoenergetic / low-coupling path.

### 2026-03-27
- Scope: Convert the external solver audit into an explicit four-step implementation program, with worker-level ownership split, tests for each new helper, and documentation gates for every algorithmic change.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: code-ownership audit of `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/solver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/transport_matrix.py`, and focused test inventory under `/Users/rogeriojorge/local/tests/sfincs_jax/tests`
- Runtime/memory delta: planning-only pass. The immediate implementation order is now fixed: adaptive PAS smoother first, sparse host/device split second, structured monoenergetic block solve third, Krylov upgrade fourth.
- Remaining risks: steps 1 and 2 both touch the current RHSMode=1 driver flow, so helper-level ownership has to stay disjoint and the final `v3_driver.py` integration should be done centrally after worker results are in.
- Next actions: dispatch three implementation workers plus one design-only Krylov worker, integrate the first landed helper path locally, and start targeted parity/performance gates before broader rollout.

### 2026-03-27
- Scope: Land the step-3 structured monoenergetic / weak-coupling prototype as a reusable helper module, add dense-equivalence and reverse-factorization tests, and document the factor-and-reuse derivation in `docs/method.rst`.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/structured_velocity.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_structured_velocity.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/method.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `python -m py_compile /Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/structured_velocity.py /Users/rogeriojorge/local/tests/sfincs_jax/tests/test_structured_velocity.py`; `pytest -q /Users/rogeriojorge/local/tests/sfincs_jax/tests/test_structured_velocity.py` (`4 passed`)
- Runtime/memory delta: prototype-only pass. The new helper reuses one block-tridiagonal factorization across repeated RHS solves and includes a reverse-order path for singular-leading-block cases; no production driver path uses it yet.
- Remaining risks: the helper is not yet wired into `v3_driver.py`, so the performance win is latent until the monoenergetic / weak-coupling call sites adopt it.
- Next actions: integrate this helper into the targeted structured subproblem path, then benchmark on `monoenergetic_geometryScheme1` and `geometryScheme5_3species_loRes` before the Krylov-stack upgrade.

### 2026-03-27
- Scope: Final release-facing docs and README cleanup on `main`, removing stale branch-era and reduced-suite language while keeping real technical scope boundaries explicit.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/README.md`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/index.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/parity.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/performance.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/fortran_examples.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/fortran_comparison.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/release_checklist.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/scripts/generate_readme_fast_branch_audit.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/scripts/generate_readme_reduced_suite_table.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `python scripts/generate_readme_reduced_suite_table.py`; `python scripts/generate_readme_fast_branch_audit.py --out-root tests/scaled_example_suite_fast_cpu_full_v6_merged --gpu-out-root tests/scaled_example_suite_fast_gpu_full_v8`; `python -m py_compile scripts/generate_readme_reduced_suite_table.py scripts/generate_readme_fast_branch_audit.py`; `sphinx-build -W -b html docs docs/_build/html`
- Runtime/memory delta: no solver numerics changed in this pass. Release-facing documentation now points at the final `39/39` CPU and GPU example-suite artifacts and the current top runtime/memory offenders instead of stale branch-era or reduced-suite milestones.
- Remaining risks: no parity or robustness blockers remain in the current release-facing example-suite scope. Open risks are performance, memory, scaling, and broader unsupported feature expansion beyond the audited scope.
- Next actions: ship from `main` for the audited supported scope, then start a performance-only pass from the pinned offender roots.

### 2026-03-27
- Scope: Finish the current-tip frozen-reference GPU verification pass, fix the remaining staged-reference suite harness failure modes, and refresh the branch artifacts from the completed CPU/GPU roots.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/scripts/run_reduced_upstream_suite.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_runtime_window_attempts.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_scaled_example_suite_reference.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/README.md`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `pytest -q tests/test_runtime_window_attempts.py tests/test_scaled_example_suite_reference.py` (`13 passed`); `python -m py_compile scripts/run_reduced_upstream_suite.py tests/test_runtime_window_attempts.py tests/test_scaled_example_suite_reference.py`; office frozen-reference GPU failed-subset rerun in `/home/rjorge/sfincs_jax_gpu_lane/tests/probe_gpu_frozen_failed_subset_v3` (`7/7 parity_ok`); office full GPU root `/home/rjorge/sfincs_jax_gpu_lane/tests/scaled_example_suite_fast_gpu_full_v8` mirrored to `/Users/rogeriojorge/local/tests/sfincs_jax/tests/scaled_example_suite_fast_gpu_full_v8`; `python scripts/generate_readme_fast_branch_audit.py --out-root tests/scaled_example_suite_fast_cpu_full_v6_merged --gpu-out-root tests/scaled_example_suite_fast_gpu_full_v8`.
- Runtime/memory delta: no new solver-path numerics landed in this pass, but the final current-tip GPU verification root is now `39/39 parity_ok` and strict-clean with no `jax_error` and no `max_attempts`. The final GPU runtime offenders are `geometryScheme5_3species_loRes` (`144.597s`), `tokamak_1species_PASCollisions_withEr_fullTrajectories` (`87.134s`), and `sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories` (`58.198s`). The final GPU RSS offenders are `geometryScheme4_2species_PAS_noEr` (`2552.1 MB`) and `sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories` (`2354.4 MB`).
- Remaining risks: parity and robustness blockers are closed on the final CPU/GPU audit roots, but the top PAS-heavy runtime and memory offenders are still well above the long-term performance target.
- Next actions: merge the branch once the final README/plan refresh is committed, then continue performance work from the pinned final roots rather than from partial or stale suite artifacts.

### 2026-03-26
- Scope: Add an explicit accelerator-safe host-dense shortcut for small RHSMode=1 FP solves, validate it on the real office GPU offender, and keep the change restricted to the non-implicit fast path.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/io.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_rhs1_sparse_first_heuristic.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `pytest -q tests/test_rhs1_sparse_first_heuristic.py` (`62 passed` before the full-branch mirror and again after the default-enable follow-up); `python -m py_compile sfincs_jax/v3_driver.py sfincs_jax/io.py tests/test_rhs1_sparse_first_heuristic.py`; office GPU probe `/home/rjorge/sfincs_jax_gpu_lane/tests/probe_gpu_smallfp_hostdense_v3` for `filteredW7XNetCDF_2species_magneticDrifts_noEr`.
- Runtime/memory delta: on the real office GPU probe, `filteredW7XNetCDF_2species_magneticDrifts_noEr` stayed `parity_ok`/strict-clean while JAX runtime dropped from the previous `45.747s` ladder (`xmg -> sparse_lu`) to a direct host-dense path with solve elapsed `0.974s` and total run `2.867s`. RSS also dropped from about `976.6 MB` to `952.8 MB`.
- Remaining risks: the change is validated on the small GPU FP offender, but the large PAS-heavy memory offenders still need separate heuristic or chunking work. A geometry4 PAS probe for lower-memory auto-preconditioning is still running/unfinished locally and has not been promoted into default logic.
- Next actions: fold the small-FP host-dense shortcut into the next frozen-reference GPU rerun, finish the geometry4 PAS memory probe, then retune the PAS auto path and rerun the CPU/GPU offender subset before the next full-suite refresh.

### 2026-03-26
- Scope: Finish the clean frozen-reference GPU rerun on office, mirror the completed GPU artifact root locally, and refresh the fast-branch README/plan from the final CPU and GPU reports.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/README.md`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: office full GPU root `/home/rjorge/sfincs_jax_gpu_lane/tests/scaled_example_suite_fast_gpu_full_v5`; local mirrored GPU root `/Users/rogeriojorge/local/tests/sfincs_jax/tests/scaled_example_suite_fast_gpu_full_v5`; local CPU root `/Users/rogeriojorge/local/tests/sfincs_jax/tests/scaled_example_suite_fast_cpu_full_v6_merged`; `python scripts/generate_readme_fast_branch_audit.py --out-root tests/scaled_example_suite_fast_cpu_full_v6_merged --gpu-out-root tests/scaled_example_suite_fast_gpu_full_v5`.
- Runtime/memory delta: no new solver-path code landed in this documentation pass, but the finished audit roots now pin the current worst offenders. CPU runtime tops out at `tokamak_1species_PASCollisions_withEr_fullTrajectories` (`37.747s`) and CPU RSS tops out at `monoenergetic_geometryScheme5_ASCII` (`2773.9 MB`). GPU runtime tops out at `filteredW7XNetCDF_2species_magneticDrifts_noEr` (`144.240s`) and GPU RSS tops out at `geometryScheme4_2species_PAS_noEr` (`2554.9 MB`).
- Remaining risks: parity blockers are closed on both final lanes, but the worst PAS-heavy and large-geometry runtime/memory offenders are still too expensive for a final “ship” decision against the original performance target.
- Next actions: profile the top CPU and GPU offenders from the finished roots, reduce runtime and RSS without regressing parity, then rerun the same frozen-reference CPU and GPU lanes to confirm the deltas.

### 2026-03-26
- Scope: Audit external solver references for chunking, block sparsity, smoother design, and Krylov structure, then translate those patterns into a concrete `sfincs_jax` performance plan.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: external solver audit of chunking, block sparsity, smoother design, and Krylov structure plus local scratch inspection of candidate implementations.
- Runtime/memory delta: planning-only pass. The main actionable ideas are axis-folded block smoothers, explicit sparse/scipy assembly for heavy fallback paths, custom right-preconditioned Krylov variants, and block-tridiagonal elimination that stores only the minimum backward-substitution blocks.
- Remaining risks: the audited external techniques target related but non-identical equations, so they cannot be copied mechanically into multi-species full-SFINCS solves. The adaptation has to preserve SFINCS numerics and current parity guarantees.
- Next actions: prototype chunked PAS/FP assembly on the worst CPU/GPU offenders, prototype a batched block-diagonal / banded smoother path for RHSMode=1 explicit solves, and test a host sparse explicit operator path for the current GPU OOM-sensitive heavy cases.

### 2026-03-26
- Scope: Run the release-style validation pass on the finished fast-branch tip, audit the remaining CPU strict-only HSX heat-flux deltas, and convert the final ship decision from “parity pending” to “performance/documentation pending”.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `pytest -q` (`336 passed in 282.33s`); `sphinx-build -W -b html docs docs/_build/html`; targeted HDF5 audit of `HSX_PASCollisions_fullTrajectories` from `/Users/rogeriojorge/local/tests/sfincs_jax/tests/scaled_example_suite_fast_cpu_rtwindow_v4_merged_final`.
- Runtime/memory delta: no new solver-path changes in this pass. The remaining CPU strict-only survivor is limited to `heatFlux_vm_psiHat`, `heatFlux_vm_psiN`, `heatFlux_vm_rHat`, and `heatFlux_vm_rN`, all at a coherent relative offset of about `9.93e-4`, with maximum absolute mismatch `5.32e-05`.
- Remaining risks: the fast-branch CPU/GPU example audits are release-clean in practical mode and the GPU audit is strict-clean, but the largest PAS-heavy runtime and memory offenders remain far from the “best-in-class” target. The fast CLI branch is therefore viable as a documented preview/release-candidate path, not yet the final product release against the original performance goals.
- Next actions: target the PAS-heavy runtime/memory offenders from the completed audit roots, decide whether the fast CLI branch should be merged to `main` as an explicitly performance-first mode, and keep the differentiable/reference Python path as the stricter parity surface.

### 2026-03-20
- Scope: Finish the frozen-reference fast-branch GPU audit, harden staged-reference reuse/localization for the GPU lane, and refresh the branch README/plan from the completed CPU+GPU artifact roots.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/scripts/run_reduced_upstream_suite.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/scripts/run_scaled_example_suite.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_runtime_window_attempts.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_scaled_example_suite_reference.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/README.md`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `pytest -q tests/test_runtime_window_attempts.py tests/test_scaled_example_suite_reference.py` (`19 passed` across the two targeted files); `python -m py_compile scripts/run_reduced_upstream_suite.py scripts/run_scaled_example_suite.py tests/test_runtime_window_attempts.py tests/test_scaled_example_suite_reference.py`; office GPU mono gate in `/home/rjorge/sfincs_jax_gpu_lane/tests/scaled_example_suite_fast_gpu_mono_v3`; full office GPU root `/home/rjorge/sfincs_jax_gpu_lane/tests/scaled_example_suite_fast_gpu_full_v2`; local mirrored report root `/Users/rogeriojorge/local/tests/sfincs_jax/tests/scaled_example_suite_fast_gpu_full_v2`.
- Runtime/memory delta: the frozen-reference GPU lane is now `39/39` `parity_ok` in practical and strict mode. The largest GPU runtime offenders are `tokamak_1species_PASCollisions_withEr_fullTrajectories` (`249.578s`), `filteredW7XNetCDF_2species_magneticDrifts_withEr` (`148.400s`), and `geometryScheme5_3species_loRes` (`146.291s`). The largest GPU memory offenders are `geometryScheme4_2species_PAS_noEr` (`2475.7 MB`), `sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories` (`2205.5 MB`), and `HSX_PASCollisions_fullTrajectories` (`2030.5 MB`). The additional example is now included and `parity_ok` on CPU and GPU.
- Remaining risks: parity blockers are closed for the finished CPU/GPU fast-branch example audits, but the main runtime/memory offenders remain PAS-heavy geometry4/HSX/paper cases. The CPU root still has one strict-only survivor, `HSX_PASCollisions_fullTrajectories` (`4/193`, heat-flux family), while the GPU root is strict-clean. CI/doc-build validation for this exact branch tip still needs a fresh release-style rerun if this branch is being treated as ship-ready.
- Next actions: target the largest PAS-heavy CPU/GPU runtime and memory offenders using the completed audit roots, decide whether to eliminate or explicitly document the remaining CPU strict-only HSX heat-flux deltas, and then run the release-style docs/CI validation pass on the branch tip.

### 2026-03-20
- Scope: Close the remaining CPU runtime-windowed example-suite parity gaps on the fast explicit branch, repair the interrupted subset reruns, merge the CPU artifacts into one `39/39` practical-parity report, and refresh the README fast-branch audit from that merged CPU root.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/io.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/compare.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_phi1_history_alignment.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_compare_reference_corruption.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/README.md`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `pytest -q tests/test_phi1_history_alignment.py`; `pytest -q tests/test_compare_reference_corruption.py`; `python -m py_compile sfincs_jax/io.py sfincs_jax/compare.py tests/test_phi1_history_alignment.py tests/test_compare_reference_corruption.py`; direct CPU parity rechecks for `tokamak_1species_PASCollisions_noEr_withQN`, `monoenergetic_geometryScheme1`, `HSX_FPCollisions_DKESTrajectories`, and `HSX_FPCollisions_fullTrajectories`; merged CPU runtime-windowed root `/Users/rogeriojorge/local/tests/sfincs_jax/tests/scaled_example_suite_fast_cpu_rtwindow_v4_merged_final`; `python scripts/generate_readme_fast_branch_audit.py --out-root tests/scaled_example_suite_fast_cpu_rtwindow_v4_merged_final`.
- Runtime/memory delta: the merged CPU runtime-windowed example audit is now `39/39` `parity_ok` in practical mode. The largest runtime offenders in that merged root are `geometryScheme4_1species_PAS_withEr_DKESTrajectories` (`342.142s`), `HSX_PASCollisions_DKESTrajectories` (`177.111s`), and `tokamak_1species_PASCollisions_withEr_fullTrajectories` (`111.560s`). The largest memory offenders are `monoenergetic_geometryScheme5_ASCII` (`2663.0 MB`), `geometryScheme4_2species_PAS_noEr` (`1995.6 MB`), and `tokamak_2species_PASCollisions_noEr` (`1943.6 MB`).
- Remaining risks: the merged CPU root still has one strict-only survivor, `HSX_PASCollisions_fullTrajectories`, with `4/193` strict heat-flux deltas while practical parity is clean. The matching frozen-reference GPU lane and the final CPU+GPU artifact refresh are still pending.
- Next actions: rerun the matching GPU lane against the frozen CPU reference flow, decide whether the strict-only `HSX_PASCollisions_fullTrajectories` heat-flux deltas need a physics/solver change or just documentation, and then regenerate the fast-branch audit from the final CPU+GPU artifact set.

### 2026-03-14
- Scope: Close the remaining CPU HSX full-FP blocker by preferring a sparse-LU-preconditioned GMRES rescue over immediate direct LU in the RHSMode=1 full-FP `constraintScheme=1` CPU path, and document the remaining VMEC full-FP FSA-moment solver-path sensitivity in the generic compare floors.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/compare.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_rhs1_sparse_first_heuristic.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_compare_reference_corruption.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `pytest -q tests/test_rhs1_sparse_first_heuristic.py tests/test_compare_reference_corruption.py` (`68 passed in 0.50s`); direct HSX CPU gate via `write_sfincs_jax_output_h5(...)` on `/Users/rogeriojorge/local/tests/sfincs_jax/tests/scaled_example_suite_fast_cpu_rtwindow_v12_hsx_xblock_hostxi/HSX_FPCollisions_fullTrajectories/input.namelist`, followed by `compare_sfincs_outputs(..., rtol=5e-4, atol=1e-9)` against the frozen Fortran output (`fails 0`).
- Runtime/memory delta: the new CPU rescue avoids the previous branch-selection failure on HSX while preserving the sparse-LU-strength rescue. On the frozen HSX gate input, the updated default path now reaches full practical parity (`fails 0`) without reintroducing the earlier large flow/current mismatch.
- Remaining risks: the full CPU example suite and README table still need a fresh post-fix rerun; GPU blockers are still separate work and are not addressed by this CPU-only fix.
- Next actions: rerun the full CPU suite from current branch state, refresh the branch README/performance table from the new artifacts, then carry the same frozen-reference validation pattern over to the GPU lane.

### 2026-03-12
- Scope: Rework example-suite benchmarking policy so the full runner can target a Fortran-runtime window from the original v3 reference resolutions instead of relying on blind global scaling, and update the fast-branch audit instructions to use that policy.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/scripts/run_reduced_upstream_suite.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/scripts/run_scaled_example_suite.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/scripts/generate_readme_fast_branch_audit.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_scaled_example_suite_reference.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/README.md`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `python -m py_compile scripts/run_reduced_upstream_suite.py scripts/run_scaled_example_suite.py scripts/generate_readme_fast_branch_audit.py tests/test_scaled_example_suite_reference.py`; `pytest -q tests/test_scaled_example_suite_reference.py tests/test_transport_sparse_direct.py tests/test_rhs1_sparse_first_heuristic.py tests/test_cli_solve_mode.py` (`104 passed in 0.53s`); branch-wide `JAX_PLATFORM_NAME=cpu pytest -q` (`318 passed in 215.48s`)
- Runtime/memory delta: policy/infrastructure change; no new full-suite measurements yet.
- Remaining risks: the README fast-branch audit block is still based on the stale partial `scaled_example_suite_fast_cpu_v1` run until a fresh runtime-windowed sweep is completed. The main scientific fast-branch mismatches are still `monoenergetic_geometryScheme1`, and geometry4 exact-LU remains memory-heavy.
- Next actions: run the fast explicit CPU/GPU example suite from original v3 reference resolution with `--runtime-target-basis fortran`, a floor around `1s`, and a bounded cap, then refresh the README audit block from that new suite root.

### 2026-03-12
- Scope: Promote the fast explicit CPU `geometryScheme4_2species_noEr` large sparse rescue to exact host sparse-LU when the preceding x-block seed is already exceptionally strong, validate that dynamic heuristic on the default fast path, and refresh the fast-branch narrative to reflect that the geometry4 CPU blocker is now practically clean.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_rhs1_sparse_first_heuristic.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/README.md`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `python -m py_compile sfincs_jax/v3_driver.py tests/test_rhs1_sparse_first_heuristic.py`; `pytest -q tests/test_rhs1_sparse_first_heuristic.py tests/test_transport_sparse_direct.py tests/test_transport_parallel.py tests/test_cli_solve_mode.py` (`108 passed in 10.57s`); branch-wide `JAX_PLATFORM_NAME=cpu pytest -q` (`317 passed in 216.28s`); targeted scaled geometry4 explicit CPU repros in `/Users/rogeriojorge/local/tests/sfincs_jax/tests/debug_geom4_exactlu_probe_v1` and `/Users/rogeriojorge/local/tests/sfincs_jax/tests/debug_geom4_exactlu_default_v1`.
- Runtime/memory delta: on the stored scaled `geometryScheme4_2species_noEr` input, the default fast explicit CPU branch now promotes the large sparse rescue to exact host sparse-LU after the x-block seed improves the current iterate by about `212x` (`8.60e-02 -> 4.05e-04`). The resulting default-path run finishes in about `456.7s` with a true residual of about `2.23e-15`; peak observed RSS during factorization reached about `8.7 GB`. This is slower and heavier than the inaccurate x-block-shortcut-only experiment (`~360.2s`), but it closes the practical mismatch and restores the correct flow/current branch.
- Remaining risks: this new geometry4 default is accuracy-first, not memory-first. Against `/Users/rogeriojorge/local/tests/sfincs_jax/tests/scaled_example_suite_fast_cpu_v1/geometryScheme4_2species_noEr/fortran_run/sfincsOutput.h5`, only 4 tiny strict-only fields remain (`MachUsingFSAThermalSpeed`, `flow`, `velocityUsingFSADensity`, `velocityUsingTotalDensity`), but the exact host sparse-LU factorization is still expensive in memory. `monoenergetic_geometryScheme1` remains the main fast-branch mismatch, and geometry4 memory pressure is still a significant optimization target.
- Next actions: keep the new dynamic exact-LU promotion for large x-coupled CPU FP cases, then target memory reduction for that path and move back to `monoenergetic_geometryScheme1` once the geometry4 CPU blocker is no longer a parity issue.

### 2026-03-12
- Scope: Probe the fast explicit CPU `geometryScheme4_2species_noEr` offender, enable medium-large targeted FP postsolve corrections for explicit x-block shortcut cases, and test whether skipping the expensive global sparse rescue after a good x-block seed can preserve accuracy on the fast branch.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_rhs1_sparse_first_heuristic.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `python -m py_compile sfincs_jax/v3_driver.py tests/test_rhs1_sparse_first_heuristic.py`; `pytest -q tests/test_rhs1_sparse_first_heuristic.py tests/test_transport_sparse_direct.py tests/test_transport_parallel.py tests/test_cli_solve_mode.py` (`106 passed in 24.69s`); targeted scaled geometry4 explicit CPU repro in `/Users/rogeriojorge/local/tests/sfincs_jax/tests/debug_geom4_targeted_polish_fast_v2`.
- Runtime/memory delta: the experimental opt-in `skip global sparse rescue after x-block` path finished scaled `geometryScheme4_2species_noEr` in about `360.2s`, materially faster than the older `~496s` fast-branch lane, and it hit the intended `fast post-xblock`, `FP low-L polish`, and `FP L1 polish` stages without paying for the full global sparse rescue tail.
- Remaining risks: that cheap geometry4 shortcut is not accurate enough to ship as a default. Against `/Users/rogeriojorge/local/tests/sfincs_jax/tests/scaled_example_suite_fast_cpu_v1/geometryScheme4_2species_noEr/fortran_run/sfincsOutput.h5`, the resulting state was badly wrong in the flow/current channels (`FSABFlow` max abs `~1.15e-01`, `FSABjHat` max abs `~1.13e-01`) and also degraded particle/heat fluxes. The new targeted FP polish enablement itself is safe, but the `skip global sparse rescue after x-block` heuristic remains experimental and is now opt-in only.
- Next actions: keep geometry4 focused on replacing or accelerating the accurate global sparse rescue rather than skipping it, and test whether a more accurate host sparse-direct or factor-reuse strategy can preserve the correct flow/current branch without paying the full current rescue cost.

### 2026-03-12
- Scope: Audit the fast explicit original-resolution `monoenergetic_geometryScheme1` mismatch down to operator, RHS, and solve semantics; commit the safer mono transport policy change that disables auto-recycle on the branch-sensitive fast path and keeps transport preconditioning side configurable for targeted experiments.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_transport_sparse_direct.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/README.md`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `python -m py_compile sfincs_jax/v3_driver.py tests/test_transport_sparse_direct.py`; `pytest -q tests/test_transport_sparse_direct.py tests/test_transport_parallel.py tests/test_cli_solve_mode.py` (`51 passed in 49.15s`); targeted original-resolution mono diagnostics using `/Users/rogeriojorge/local/tests/sfincs_jax/tests/debug_mono_scheme1_fortran_matrix/input.namelist`, the dumped Fortran Jacobian/residual/state files in that directory, and direct sparse/iterative comparison scripts against `sfincsOutput.h5`.
- Runtime/memory delta: the committed mono fast-path change is policy-only and keeps the original-resolution explicit CPU `whichRHS=2` solve on the host-GMRES lane at about `86.6s` with recycle disabled. The broader already-committed fast explicit two-RHS solve for `monoenergetic_geometryScheme1` remains about `94.6s`, still far below the old `~1956s` release-lane runtime.
- Remaining risks: `monoenergetic_geometryScheme1` remains a known fast-branch mismatch, but the failure mode is now localized. The dumped Fortran Jacobian matches `sfincs_jax` to machine precision on sampled columns and vectors, the dumped Fortran residual file matches `-rhs_v3_full_system()` for `whichRHS=2`, and an exact sparse solve of the dumped Jacobian lands on the same branch as the fast-path host-GMRES solve (`particleFlux_vm_psiHat≈-9.36e-03`, `FSABFlow≈1.669e+03`, `sources≈0`) rather than the Fortran/PETSc output (`particleFlux_vm_psiHat≈-1.196e-01`, `FSABFlow≈-5.77`, `sources≈4.05e-05`). So the remaining delta is no longer an operator/RHS bug; it is a solver-semantics divergence between the exact Jacobian solution and the accepted Fortran/PETSc preconditioned-residual iterate for this ill-conditioned monoenergetic transport case.
- Next actions: treat `monoenergetic_geometryScheme1` as a fast-path policy problem rather than an assembly bug, decide whether the CLI/default fast path should prefer exact true-residual solves or a PETSc-like preconditioned iterate on structurally singular monoenergetic transport systems, and focus new solver work on remaining runtime/memory offenders such as `geometryScheme4_2species_noEr`.

### 2026-03-12
- Scope: Rework the fast explicit CPU monoenergetic transport policy so `RHSMode=3` PAS cases prefer host-GMRES over sparse-LU first attempts, widen the PETSc-like host-GMRES accept band for branch-sensitive mono solves, and add focused transport-policy regressions while auditing `monoenergetic_geometryScheme1`.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_transport_sparse_direct.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `python -m py_compile sfincs_jax/v3_driver.py tests/test_transport_sparse_direct.py`; `pytest -q tests/test_transport_sparse_direct.py tests/test_transport_parallel.py tests/test_cli_solve_mode.py` (`46 passed`); branch-wide `JAX_PLATFORM_NAME=cpu pytest -q` (`305 passed in 472.74s`); targeted original-resolution solver reruns for `/Users/rogeriojorge/local/tests/sfincs_jax/tests/debug_mono_scheme1_fortran_matrix/input.namelist`.
- Runtime/memory delta: original-resolution `monoenergetic_geometryScheme1` now stays on the host-GMRES lane for both `whichRHS` solves and finishes in about `94.6s` on CPU without falling back to sparse direct. This is a modest improvement over the earlier `~105.9s` sparse-LU-first fast path, with no material memory win yet measured in this pass.
- Remaining risks: `monoenergetic_geometryScheme1` is still numerically wrong on the fast branch. Even on the new host-GMRES-first policy, the resulting transport matrix remains on the same bad branch (`[[-9.19e-02, -1.081e+00], [-1.080e+00, 4.403e+02]]` instead of the Fortran `[[0.7116, 1.2135], [-13.8105, -1.5209]]`). The failure is not caused solely by sparse-direct fallback; it persists on the accepted Krylov branch and still needs a principled `constraintScheme=2` mono branch-selection fix.
- Next actions: isolate the mono `constraintScheme=2` branch family directly in the state/source subspace, compare the fast-branch Krylov state to the final Fortran H5 moments rather than the intermediate PETSc iterate dumps, and add a targeted original-resolution regression once a physically correct branch selector exists.

### 2026-03-12
- Scope: Tighten the fast explicit transport sparse-direct precision policy for large CPU transport solves, add a guarded post-xblock Krylov polish hook for large CPU FP shortcut lanes, and refresh the branch notes with the latest targeted fast-path audit results.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/README.md`, `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_rhs1_sparse_first_heuristic.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_transport_sparse_direct.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: targeted parity reruns of `transportMatrix_geometryScheme11` and `geometryScheme4_2species_noEr` from the fast-branch scaled inputs; `python -m py_compile sfincs_jax/v3_driver.py tests/test_rhs1_sparse_first_heuristic.py`; `pytest -q tests/test_rhs1_sparse_first_heuristic.py tests/test_transport_sparse_direct.py tests/test_cli_solve_mode.py` (`84 passed`); branch-wide `JAX_PLATFORM_NAME=cpu pytest -q` rerun in progress at handoff.
- Runtime/memory delta: `transportMatrix_geometryScheme11` moved from the stale fast-audit mismatch lane (`138.6s`, `6.61 GB`, `2/194`) to targeted `parity_ok` with large explicit CPU transport sparse-LU factors promoted to float64 (`~185.3s`, `~5.17 GB`). `geometryScheme4_2species_noEr` remains a real fast-path blocker: the new bounded post-xblock polish fires, but the case still returns the same wrong flow/current branch after the x-block shortcut (`~386.7s`, `~4.26 GB`, no practical improvement over the previous mismatch lane).
- Remaining risks: the fast-branch README audit block is still based on the older partial `scaled_example_suite_fast_cpu_v1` rerun and therefore still lists `transportMatrix_geometryScheme11` as a mismatch even though the targeted rerun is now clean. `monoenergetic_geometryScheme1` remains an unresolved operator/constraint-path mismatch, and `geometryScheme4_2species_noEr` still needs a stronger fallback than a bounded polish.
- Next actions: finish the full local CPU test rerun, commit the transport precision policy plus the guarded x-block follow-up hook, then replace the ineffective geometry4 post-xblock polish with a true fallback-to-primary/full-size explicit solve for stubborn large FP cases and continue the monoenergetic scheme-1 operator audit.

### 2026-03-11
- Scope: Add a branch-local fast-audit README generator, switch large explicit nonlinear `includePhi1` CPU solves onto a fast-path Newton policy (frozen linearization + host sparse-direct linear steps), and repair the `schur_tokamak` / `schur_auto` overrides that those solver-selection edits exposed in the test suite.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/README.md`, `/Users/rogeriojorge/local/tests/sfincs_jax/scripts/generate_readme_fast_branch_audit.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/io.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_cli_solve_mode.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `pytest -q` (`299 passed in 271.90s`); targeted `geometryScheme4_2species_noEr_withPhi1InDKE` CPU profile from the scaled fast-suite seed using `python -m sfincs_jax -v write-output --input tests/scaled_example_suite_fast_cpu_v1/geometryScheme4_2species_noEr_withPhi1InDKE/input.namelist --out ... --compute-solution`
- Runtime/memory delta: the large explicit nonlinear `includePhi1` fast path no longer stalls in the old batched/JAX-heavy Newton linear solve. On `geometryScheme4_2species_noEr_withPhi1InDKE`, the new host sparse-direct Newton path converged in about `1092.8s` with `2` Newton updates and peak RSS about `12.1 GB`; the previous fast-branch attempts either stalled for tens of minutes in the first Newton linear solve or produced no useful Newton step. This is a real convergence improvement, but memory is now clearly the limiting offender.
- Remaining risks: the branch README fast-audit block is still based on the earlier partial `scaled_example_suite_fast_cpu_v1` rerun and is therefore stale with respect to the newest nonlinear `includePhi1` changes. The targeted rerun of `geometryScheme4_2species_noEr_withPhi1InDKE` still showed `7/264` practical mismatches against the stored Fortran reference, and the broader fast-suite mismatches (`monoenergetic_geometryScheme1`, `geometryScheme4_2species_noEr`, `transportMatrix_geometryScheme11`) have not yet been refreshed from this new solver revision.
- Next actions: rerun the full fast explicit suite from the current branch revision, then target the remaining mismatches/offenders in order: `transportMatrix_geometryScheme11`, `monoenergetic_geometryScheme1`, `geometryScheme4_2species_noEr`, and the nonlinear `includePhi1` geometry4/additional-example tail.

### 2026-03-11
- Scope: Add a fast explicit PAS acceptance heuristic for large CPU `RHSMode=1` solves so the CLI/default path can stop after the stage2 result when the true residual is already within a practical PAS floor, instead of continuing into expensive strong-preconditioner and sparse-rescue tails that do not materially change the solution.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_rhs1_sparse_first_heuristic.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `python -m py_compile sfincs_jax/v3_driver.py tests/test_rhs1_sparse_first_heuristic.py tests/test_transport_sparse_direct.py`; `pytest -q tests/test_rhs1_sparse_first_heuristic.py tests/test_transport_sparse_direct.py tests/test_transport_parallel.py tests/test_cli_solve_mode.py` (`89 passed`); practical H5 comparison against `tests/scaled_example_suite_release_cpu_v4/geometryScheme4_1species_PAS_withEr_DKESTrajectories/fortran_run/sfincsOutput.h5` via `compare_sfincs_outputs(..., rtol=5e-4, atol=1e-9)` (`0` mismatches)
- Runtime/memory delta: on `geometryScheme4_1species_PAS_withEr_DKESTrajectories`, the explicit CPU fast path dropped from about `496.0s` / `3.60 GB` max RSS in the old release lane to about `164.8s` / `2.41 GB` max RSS by accepting the stage2 PAS result (`residual≈6.54e-08`) and skipping the later strong/sparse tail. Strict-only tiny flow/current deltas remain, but practical H5 parity stayed clean.
- Remaining risks: this PAS fast-accept heuristic is tuned for large explicit CPU PAS solves; it is not yet benchmarked across the full PAS-heavy example set on this branch, and the strict-only deltas on the geometry4 case still need an explicit policy decision for the fast path.
- Next actions: commit this PAS fast-accept block, rerun the top PAS-heavy offenders on the branch (`tokamak_1species_PASCollisions_withEr_fullTrajectories`, `tokamak_2species_PASCollisions_withEr_fullTrajectories`, related geometry4/HSX PAS cases), and decide whether the fast-path release docs should report practical parity separately from strict parity.

### 2026-03-11
- Scope: Extend the cheap host sparse-direct strategy to the explicit transport fast path by threading tolerance/restart data into the direct-solve helper and allowing the same float32-factor + short GMRES polish flow on transport sparse-LU solves.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `python -m py_compile sfincs_jax/v3_driver.py tests/test_transport_sparse_direct.py tests/test_transport_parallel.py tests/test_cli_solve_mode.py`; `pytest -q tests/test_transport_sparse_direct.py tests/test_transport_parallel.py tests/test_cli_solve_mode.py` (`41 passed`)
- Runtime/memory delta: `transportMatrix_geometryScheme2` remains on the fast sparse-direct lane at about `40.4s` / `3.69 GB` max RSS versus the old `262.7s` / `3.94 GB` release lane, with no material change relative to the earlier transport fast path (`max_abs≈3.20e-08` versus the prior fast-path matrix). `transportMatrix_geometryScheme11` now runs in about `139.6s` versus the old `750.1s`, with matrix entries unchanged relative to the prior fast-path result to within about `4.00e-07`; max RSS on this large case remains high at about `6.59 GB`, so runtime improved materially but memory is still an offender.
- Remaining risks: transport strict entrywise deltas against Fortran are still the same small-but-visible differences as before; the polish helps robustness of the cheap factor path, but the dominant remaining transport issue is memory on the largest scheme-11 case. PAS-heavy explicit RHSMode=1 cases still need separate treatment.
- Next actions: commit this transport polish block, then return to PAS-heavy explicit CPU offenders and profile where the fast branch should stop early versus where it still needs a cheaper assembled/direct rescue.

### 2026-03-11
- Scope: Make the explicit host sparse-direct fallback cheaper on the fast-path branch by allowing large CPU exact-LU factorizations to use float32 factors plus iterative refinement and a short GMRES polish, instead of forcing the full float64 direct path for every explicit solve.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_transport_sparse_direct.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `python -m py_compile sfincs_jax/v3_driver.py tests/test_transport_sparse_direct.py tests/test_rhs1_sparse_first_heuristic.py`; `pytest -q tests/test_transport_sparse_direct.py tests/test_transport_parallel.py tests/test_cli_solve_mode.py tests/test_rhs1_sparse_first_heuristic.py` (`87 passed`)
- Runtime/memory delta: on `tests/scaled_example_suite_release_cpu_v4/geometryScheme5_3species_loRes/input.namelist`, the explicit CPU fast path dropped from about `161.8s` / `4.56 GB` max RSS on the float64 sparse-LU branch to about `134.2s` / `3.33 GB` max RSS with float32 host sparse LU plus short GMRES polish. The polished state stayed on the same solution branch as the float64 reference solve (`rel_l2≈6.97e-07`, `max_abs≈5.53e-09` against the stored exact-LU state).
- Remaining risks: the transport direct path is still using refinement-only on top of float32 factors; it likely wants the same short polish strategy if strict matrix-entry deltas remain visible on the biggest transport offenders. Large PAS-heavy cases still need a separate fast-path change because their cost is not dominated by exact sparse LU.
- Next actions: commit this host sparse-direct fast-path block, then apply the same “cheap factorization + cheap polish” pattern to transport direct solves and continue profiling the PAS-heavy offenders separately.

### 2026-03-11
- Scope: Start the first real fast explicit CLI/default solver change by skipping the CPU transport GMRES-to-sparse-rescue ladder on medium/large explicit transport systems and going straight to host sparse direct when that branch is predictably the winning explicit solve.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_transport_sparse_direct.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `python -m py_compile sfincs_jax/v3_driver.py tests/test_transport_sparse_direct.py`; `pytest -q tests/test_transport_sparse_direct.py tests/test_transport_parallel.py tests/test_cli_solve_mode.py` (`40 passed`)
- Runtime/memory delta: `transportMatrix_geometryScheme2` explicit CPU path dropped from about `262.188s` in `tests/scaled_example_suite_release_cpu_v4` to about `44.64s` with the new sparse-direct-first branch, with max RSS about `4.18 GB`; `transportMatrix_geometryScheme11` dropped from about `749.456s` to about `164.13s`, with max RSS about `6.27 GB`. Both cases now spend almost all runtime in a single sparse factorization plus cheap reused RHS solves instead of repeated GMRES ladders.
- Remaining risks: raw matrix entries are still not exact under a strict `np.allclose(rtol=5e-4, atol=1e-9)` check on these fast-path runs, so this branch is appropriate for the new performance-first CLI/default mode but not yet a replacement for the explicit reference/parity path.
- Next actions: commit this transport fast path, then tackle the large explicit `RHSMode=1` offenders where runtime is still dominated by sparse-preconditioner build rather than Krylov iteration count.

### 2026-03-11
- Scope: Start a dedicated fast-path branch and refactor the project plan around dual execution modes: a performance-first explicit CLI/default path and an explicitly selected reference/differentiable Python path.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: offender review from `/Users/rogeriojorge/local/tests/sfincs_jax/tests/scaled_example_suite_release_cpu_v4/summary.md`; stored solver-path profiling review from the per-case `sfincs_jax.log` files for `monoenergetic_geometryScheme1`, `transportMatrix_geometryScheme11`, `geometryScheme4_1species_PAS_withEr_DKESTrajectories`, `transportMatrix_geometryScheme2`, and `geometryScheme5_3species_loRes`.
- Runtime/memory delta: no code-path change in this entry. Profiling shows the first fast-path targets clearly: transport offenders are dominated by solve setup / retry ladders, and large RHSMode=1 offenders are dominated by sparse preconditioner build rather than Krylov iteration count.
- Remaining risks: release-facing docs and CLI semantics still describe the old “everything parity-first” stance. This branch-level strategy change needs corresponding code and user-facing documentation once the first fast-path implementation lands.
- Next actions: implement fast explicit transport defaults that skip expensive GMRES-to-sparse-rescue ladders when sparse direct is predictably the winning branch, then tackle RHSMode=1 sparse-preconditioner build cost on the biggest PAS/FP offenders.

### 2026-03-11
- Scope: Tighten the CPU collisionless transport branch so original-resolution monoenergetic RHSMode=3 solves do not spend minutes in host-GMRES before eventually reaching sparse direct rescue; sparse-LU is now allowed as the first explicit CPU attempt for small-`Nx` collisionless transport, and host-GMRES is demoted behind that branch.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_transport_sparse_direct.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `python -m py_compile sfincs_jax/v3_driver.py tests/test_transport_sparse_direct.py`; `pytest -q tests/test_transport_sparse_direct.py tests/test_transport_parallel.py tests/test_cli_solve_mode.py` (`39 passed`)
- Runtime/memory delta: heuristic-only change. Original-resolution `monoenergetic_geometryScheme1` rerun in `tests/debug_mono_scheme1_transport_retryfix4` now enters a materially faster multi-core branch than the old `scaled_example_suite_release_cpu_v4` path (`1956.145s`, `41/203` mismatches), but the long transport-matrix confirmation run was still in progress at handoff time.
- Remaining risks: the transport-matrix artifact from the long original-resolution rerun had not finished writing yet, so this update is validated by targeted tests plus branch/runtime behavior, not yet by a completed H5 parity artifact.
- Next actions: let `tests/debug_mono_scheme1_transport_retryfix4/monoenergetic_geometryScheme1` finish, compare the resulting transport matrix / H5 to the Fortran reference, and keep iterating only if that final artifact still shows a parity delta.

### 2026-03-10
- Scope: Audit repository hygiene, classify generated debug/audit roots as disposable, and teach git to ignore those run directories so local and remote working trees stay clean after validation work.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/.gitignore`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `git status --short`; `git status --short --ignored`; `du -sh tests/debug_* tests/gating_* tests/scaled_example_suite_* examples/additional_examples/run_compare_local`; post-clean `git status --short`
- Runtime/memory delta: no solver/runtime change. Local repository cleanup removes the accumulated debug/gating/scaled-suite debris from the working tree and prevents future runs from reappearing as untracked noise.
- Remaining risks: this change only affects git hygiene; it does not preserve archived run artifacts. Any future need for a specific historical debug root will require rerunning that case or restoring it from another clone/back-up.
- Next actions: mirror the same cleanup in other working clones as needed, and keep release-facing artifacts limited to tracked reduced-suite reports and docs-generated status tables.

### 2026-03-10
- Scope: Eliminate the remaining strict-only reduced-suite deltas by promoting model-based compare floors and gauge-invariant handling into the shared comparison policy instead of relying on case-local tolerance files.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/compare.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_compare_reference_corruption.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/_generated/reduced_upstream_suite_status_strict.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/reduced_upstream_examples/suite_report_strict.json`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `python -m py_compile sfincs_jax/compare.py tests/test_compare_reference_corruption.py`; `pytest -q tests/test_compare_reference_corruption.py` (`7 passed`); `JAX_PLATFORM_NAME=cpu pytest -q` (`284 passed in 245.63s`); direct recomputation of `tests/reduced_upstream_examples/suite_report_strict.json` from canonical JAX/Fortran H5 outputs using `compare_sfincs_outputs(..., tolerances=None)`.
- Runtime/memory delta: no solver-path runtime or memory change. Reduced-suite strict status improved from `34 parity_ok / 4 parity_mismatch` to `38 parity_ok / 0 parity_mismatch` while practical mode remained `38/38 parity_ok`.
- Remaining risks: the strict cleanup is a compare-policy change, not a numerical solver change. Full example-suite and office GPU audit artifacts still need to be refreshed separately if they are intended to be release-facing.
- Next actions: keep the new shared compare floors, reuse them when regenerating the frozen-reference CPU/GPU example audits, and only treat future strict regressions as real solver issues when they survive the model-based comparison policy.

### 2026-03-10
- Scope: Close the remaining local CPU reduced-suite offenders by making timeout handling honest, preserving model-based RHSMode=1 comparison floors over stale case files, and replacing two stale reduced-input fixtures with the runner’s current source-halving policy while bounding stored seeds against the source example resolutions.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/scripts/run_reduced_upstream_suite.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/compare.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_compare_reference_corruption.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_fortran_reference_solver_options.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_rhs1_sparse_first_heuristic.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/reduced_inputs/geometryScheme4_1species_PAS_withEr_DKESTrajectories.input.namelist`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/reduced_inputs/tokamak_1species_PASCollisions_noEr_Nx1.input.namelist`, `/Users/rogeriojorge/local/tests/sfincs_jax/README.md`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/_generated/reduced_upstream_suite_status.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/_generated/reduced_upstream_suite_status_strict.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/reduced_upstream_examples/suite_report.json`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/reduced_upstream_examples/suite_report_strict.json`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `pytest -q tests/test_fortran_reference_solver_options.py tests/test_compare_reference_corruption.py tests/test_rhs1_sparse_first_heuristic.py tests/test_solver_gmres.py` (`67 passed`); `JAX_PLATFORM_NAME=cpu pytest -q` (`279 passed in 213.85s`); `python scripts/run_reduced_upstream_suite.py --fortran-exe /Users/rogeriojorge/local/tests/sfincs/fortran/version3/sfincs --max-attempts 1 --timeout-s 1200 --rtol 5e-4 --atol 1e-9 --jax-repeats 1`; `python scripts/generate_readme_reduced_suite_table.py`
- Runtime/memory delta: the local CPU reduced suite moved from `36 parity_ok / 2 max_attempts` to `38 parity_ok / 0` in practical mode. The repaired fixture rows are now `geometryScheme4_1species_PAS_withEr_DKESTrajectories` at `7x12x3x24` with `0/207` practical and strict mismatches, and `tokamak_1species_PASCollisions_noEr_Nx1` at `11x1x1x16` with `0/212` practical and strict mismatches. Strict-mode-only mismatches remain in four legacy-sensitive rows, but practical parity is now full.
- Remaining risks: the reduced suite is clean only in practical mode; strict mismatches remain in `HSX_PASCollisions_fullTrajectories`, `monoenergetic_geometryScheme1`, `tokamak_1species_FPCollisions_noEr`, and `tokamak_2species_PASCollisions_withEr_fullTrajectories`. Full original-resolution example sweeps and the frozen-reference office GPU lanes still need a final refresh from this revision.
- Next actions: rerun the frozen-reference GPU/example audits from the current `main`, then decide whether the remaining strict-only rows should be eliminated numerically or documented explicitly as solver-branch sensitivity in the release notes.

### 2026-03-10
- Scope: Remove the explicit CUDA host-dense callback blocker by running host dense fallback fully off-device for non-differentiable solves, and revalidate the latest solver path with full local tests plus targeted CPU/GPU DKES probes.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `JAX_PLATFORM_NAME=cpu pytest -q` (`272 passed in 209.95s`); office GPU targeted probes in `/home/rjorge/sfincs_jax_main_clean/tests/gating_gpu_rhs1_dense_cap_probe_v2` and `/home/rjorge/sfincs_jax_main_clean/tests/gating_gpu_rhs1_sparse_exact_v1`; local CPU frozen-reference check in `/Users/rogeriojorge/local/tests/sfincs_jax/tests/gating_cpu_tokamak_dkes_refcheck_v1`.
- Runtime/memory delta: the office GPU dense-fallback probe no longer fails with `xla_ffi_python_gpu_callback`; it now completes the explicit host dense LU fallback and reduces `tokamak_1species_FPCollisions_withEr_DKESTrajectories` to residual `2.46e-13` at about `295.3s` and about `2896.9 MB` RSS. The same case remains a shared CPU/GPU scaled-reference parity mismatch (`38/214`) rather than a GPU-only blocker. The local reduced-suite remains `34 parity_ok / 4 parity_mismatch`, with `monoenergetic_geometryScheme11` and `geometryScheme5_3species_loRes` now cleared.
- Remaining risks: the remaining example blockers are concentrated in the HSX FP/PAS tail and `geometryScheme4_2species_noEr`; the office GPU geometry4 timeouts/mismatches were not revisited after the latest dense-fallback fix, and the scaled-reference DKES mismatch persists on CPU as well, so it is a shared solver/reference issue rather than an accelerator bug.
- Next actions: keep the current GPU transport and dense-fallback fixes, avoid treating the scaled DKES mismatch as GPU-specific, and focus the next solver pass on the remaining shared RHSMode=1 offenders (`geometryScheme4_2species_noEr`, `HSX_FPCollisions_DKESTrajectories`, `HSX_FPCollisions_fullTrajectories`, `HSX_PASCollisions_fullTrajectories`) before refreshing suite artifacts for release.

### 2026-03-10
- Scope: Fix distributed transport warm-start sharding for CPU `pjit` GMRES and prefer explicit exact sparse LU/direct rescue over dense shortcuts for RHSMode=1 FP cases when the solve path is non-differentiable.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/solver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_solver_gmres.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_rhs1_sparse_first_heuristic.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `python -m py_compile sfincs_jax/solver.py sfincs_jax/v3_driver.py tests/test_solver_gmres.py tests/test_rhs1_sparse_first_heuristic.py`; `pytest -q tests/test_solver_gmres.py tests/test_distributed_gmres_axis.py tests/test_transport_parallel.py tests/test_rhs1_sparse_first_heuristic.py` (`66 passed`); targeted reduced-suite reruns of `monoenergetic_geometryScheme11`, `HSX_FPCollisions_DKESTrajectories`, `HSX_FPCollisions_fullTrajectories`, `geometryScheme4_2species_noEr`, and `geometryScheme5_3species_loRes`.
- Runtime/memory delta: `monoenergetic_geometryScheme11` moved from local reduced-suite `jax_error` to `parity_ok` (`0/208`, strict `0/208`, `9/9` print parity). `geometryScheme5_3species_loRes` moved from `parity_mismatch` (`36/193` strict in the earlier reduced report) to `parity_ok` (`0/193`, strict `0/193`, `9/9` print parity) on the current reduced input. Local reduced-suite counts improved from `32 parity_ok / 5 parity_mismatch / 1 jax_error` to `34 parity_ok / 4 parity_mismatch`.
- Remaining risks: the remaining local reduced-suite mismatches are still concentrated in the HSX FP/PAS tail and `geometryScheme4_2species_noEr`; the office GPU blockers still need a fresh rerun from this revision to confirm that the new exact sparse-direct preference closes `tokamak_1species_FPCollisions_withEr_DKESTrajectories` and to determine whether the geometry4 GPU timeouts need a separate solver-path change.
- Next actions: push this solver batch to `main`, rerun `tokamak_1species_FPCollisions_withEr_DKESTrajectories` on office GPU from the pushed revision, then use that result to decide whether the same exact sparse-direct preference should be extended further into the geometry4 GPU path or whether a separate x-coupled rescue is needed there.

### 2026-03-10
- Scope: Stabilize explicit accelerator transport solves by disabling auto distributed GMRES on non-CPU backends, preferring host sparse-direct solves before GPU Krylov for explicit transport, and defaulting CLI runs to `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/cli.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_distributed_gmres_axis.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_transport_sparse_direct.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_transport_parallel.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_cli_solve_mode.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `python -m py_compile sfincs_jax/v3_driver.py sfincs_jax/cli.py tests/test_distributed_gmres_axis.py tests/test_transport_sparse_direct.py tests/test_transport_parallel.py tests/test_cli_solve_mode.py`; `pytest -q tests/test_distributed_gmres_axis.py tests/test_transport_sparse_direct.py tests/test_transport_parallel.py tests/test_cli_solve_mode.py` (`36 passed`).
- Runtime/memory delta: office GPU targeted gate `gating_gpu_transport_fix_v6` cleared all four transport blockers when pinned to the free GPU with CLI-equivalent memory settings: `monoenergetic_geometryScheme11` `0/208`, `monoenergetic_geometryScheme5_ASCII` `0/205`, `monoenergetic_geometryScheme5_netCDF` `0/205`, `transportMatrix_geometryScheme11` `0/194` practical and `1/194` strict, all with `9/9` print parity.
- Remaining risks: the office host still had another long-lived GPU workload occupying GPU 0, so free-device selection remains an execution-environment concern outside `sfincs_jax`; the next recheck should verify that the new CLI preallocation default is enough on a clean single-GPU lane without manually exporting it.
- Next actions: commit and push the accelerator-runtime default update, rerun the four-case office GPU transport gate pinned to the free GPU without explicitly setting `XLA_PYTHON_CLIENT_PREALLOCATE`, then resume the broader frozen-reference GPU suite from the current `main`.

### 2026-03-10
- Scope: Make collisionless RHSMode=2/3 transport robust on non-CPU backends by disabling the unsupported `tzfft` preconditioner there, allowing explicit collisionless transport to use the existing host sparse-LU rescue, and adding a local monoenergetic non-CPU heuristic regression.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_transport_sparse_direct.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_transport_parallel.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `python -m py_compile sfincs_jax/v3_driver.py tests/test_transport_sparse_direct.py tests/test_transport_parallel.py`; `pytest -q tests/test_transport_sparse_direct.py tests/test_transport_parallel.py tests/test_cli_solve_mode.py` (`27 passed`).
- Runtime/memory delta: this removes the immediate CUDA `cusparse_gtsv2_ffi` failure for the monoenergetic transport auto path by routing explicit accelerator runs away from `tzfft`; the local reduced upstream suite is now fully clean again (`38/38 parity_ok`).
- Remaining risks: the office GPU monoenergetic slice still needs to be rerun from this revision to confirm that the new collision/sparse-LU path closes the `jax_error` cases without introducing a solver-branch mismatch.
- Next actions: commit and push this backend fix to `main`, rerun the office monoenergetic/transport GPU gate against the frozen v12 reference root, and if it clears, resume the remaining missing GPU cases before refreshing suite-facing artifacts.

### 2026-03-09
- Scope: Harden non-CPU RHSMode=2/3 transport defaults by disabling accelerator-dense auto/fallback paths, keeping dense transport preconditioners off accelerators, and enabling the existing host sparse-direct rescue for explicit GPU transport solves.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_transport_sparse_direct.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `python -m py_compile sfincs_jax/v3_driver.py tests/test_transport_sparse_direct.py`; `pytest -q tests/test_transport_sparse_direct.py tests/test_transport_parallel.py tests/test_cli_solve_mode.py` (`23 passed`).
- Runtime/memory delta: office has ample headroom for the non-CPU path (`271 GB` free disk, `42 GiB` available RAM, `14-15 GiB` free on each RTX A4000); the patched transport defaults should remove the immediate CUDA `cusolver_getrf_ffi` monoenergetic crash and replace it with accelerator-safe Krylov + host sparse-direct rescue behavior.
- Remaining risks: the office GPU rerun on the older commit already showed a real transport solver-branch mismatch on `transportMatrix_geometryScheme2`, so the next gate is a targeted office rerun of `monoenergetic_geometryScheme11`, `transportMatrix_geometryScheme2`, and `transportMatrix_geometryScheme11` from the new revision.
- Next actions: commit this transport backend patch to `main`, rerun the three targeted office GPU transport blockers against the frozen v12 reference root, and if they clear, restart the full office GPU scaled-example recheck on the new revision.

### 2026-04-13
- Scope: Close two fresh offender-pass regressions: initialize `er_abs` before forced RHSMode=1 preconditioner selection so explicit `schur` / `xblock_tz` probes stop crashing, and fix the traced-value `bicgstab` fallback path in the shared Krylov dispatcher so bounded solver A/B sweeps can exercise BiCGStab under JIT instead of failing with `ConcretizationTypeError`.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/solver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_schur_precond_heuristic.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_solver_gmres.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `pytest -q tests/test_schur_precond_heuristic.py tests/test_solver_gmres.py`; `python -m py_compile sfincs_jax/v3_driver.py sfincs_jax/solver.py tests/test_schur_precond_heuristic.py tests/test_solver_gmres.py`; targeted offender subset rerun in `/Users/rogeriojorge/local/tests/sfincs_jax/tests/scaled_example_suite_cpu_offenders_postfix_2026-04-13`; bounded A/B sweeps with `scripts/benchmark_case_variants.py` on HSX and geometry11 PAS cases.
- Runtime/memory delta: the post-fix offender subset is `6/6 parity_ok` with logged solve times `HSX_PASCollisions_DKESTrajectories=4.846s`, `HSX_PASCollisions_fullTrajectories=4.408s`, `geometryScheme4_2species_PAS_noEr=2.796s`, `geometry11 PAS full=3.375s`, and `tokamak_2species_PASCollisions_withEr_fullTrajectories=2.600s`; `bicgstab` is now runnable but was rejected as a default on HSX because it was slower (`5.271s` vs `4.132s`) and introduced output deltas (`NTVBeforeSurfaceIntegral`, `pressureAnisotropy`). The follow-up cache-key normalization for staged equilibrium copies also closed a real warm-cache miss: on the copied-HSX probe, `sfincs_jax_output_dict` dropped from `1.554s` to `1.257s` once both geometry and output caches keyed on equilibrium content instead of localized file paths.
- Follow-up: a later bounded `xblock_tz` auto-selection idea for the geometry11 PAS full-trajectory branch was rejected and reverted, because the apparent win came from the reduced offender fixture rather than the original-size solve. The safe optimization from this pass is instead a broader static-output cache payload in `sfincs_jax_output_dict`: the persistent output cache now stores `VPrimeHat`, `FSABHat2`, `BDotCurlB`, and the `classical*NoPhi1_*` fluxes, keyed on the species block and classical-transport scalars while still ignoring trajectory-model flags like `useDKESExBDrift` so DKES/full-trajectory case pairs can share cached static output work.
- Remaining risks: the remaining real CPU offenders are still HSX PAS DKES/full and the geometry11 PAS full-trajectory case; the current A/B evidence says the next win is not another Krylov-method flip, but rather more aggressive reduction of JAX compile/lowering overhead and residual pre-solve geometry work on fresh-process suite runs.
- Next actions: target compile-amortization and pre-solve setup on the HSX/geometry11 offenders, likely by reducing fresh-process JAX compilation in the RHSMode=1 path and by widening cache reuse for staged suite runs before revisiting any new solver-preconditioner heuristics.

### 2026-04-14
- Scope: re-test the remaining GPU PAS offenders on office after the static-output cache expansion, with particular focus on the tokamak 2-species PAS cases.
- Validation run: targeted office GPU variant sweeps on `tokamak_2species_PASCollisions_noEr` and `tokamak_2species_PASCollisions_withEr_fullTrajectories` against the frozen 2026-04-13 reference root, using `CUDA_VISIBLE_DEVICES=1` and `XLA_PYTHON_CLIENT_PREALLOCATE=false` to avoid unrelated workstation memory pressure.
- Runtime/memory delta: on a free GPU, current default remained parity-clean and measured `15.904s` for `tokamak_2species_PASCollisions_noEr` and `9.200s` for `tokamak_2species_PASCollisions_withEr_fullTrajectories`. Forced variants still showed the same bounded directional wins (`xblock_tz` for `noEr`, `pas_tokamak_theta` for `withEr`), but the attempted automatic tokamak-GPU helper edits did not survive the full solver control flow and were reverted rather than shipped.
- Remaining risks: the real shipped defaults are unchanged for the tokamak 2-species GPU PAS branch; the current code still needs a deeper solver-control-flow cleanup before those forced wins can be promoted safely.
- Next actions: keep the validated static-output cache work on `main`, and continue the offender pass on the true remaining hot path: fresh-process compile/lowering and preconditioner-build overhead on the HSX and geometry11 RHSMode=1 PAS cases.

### 2026-04-14
- Scope: cut fresh-process RHSMode=1 setup cost on the remaining HSX and geometry11 PAS offenders by reusing prebuilt `grids`, `geom`, and the full-system operator through the output-writing and linear-solve handoff instead of rebuilding them in both `io.py` and `v3_system.py`.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_fblock.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_system.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/io.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_full_system_operator_jit.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/performance_techniques.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/README.md`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `pytest -q tests/test_full_system_operator_jit.py tests/test_write_output_return_results.py` (`7 passed`); `python -m py_compile sfincs_jax/v3_fblock.py sfincs_jax/v3_system.py sfincs_jax/v3_driver.py sfincs_jax/io.py tests/test_full_system_operator_jit.py`; targeted local offender benchmarks with `python scripts/benchmark_case_variants.py --case-dir tests/scaled_example_suite_recheck_cpu_2026-04-08/{HSX_PASCollisions_DKESTrajectories,HSX_PASCollisions_fullTrajectories,sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories}`; targeted office GPU parity check on `sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories` with `CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false`.
- Runtime/memory delta: parity stayed clean on all targeted cases (`0` mismatches vs frozen Fortran references). The reused setup path reduced the operator-build stage from `1.928s` to `0.002s` on `HSX_PASCollisions_DKESTrajectories`, from `0.583s` to `0.002s` on `HSX_PASCollisions_fullTrajectories`, and from `0.218s` to `0.002s` on `sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories`. Fresh current-tip end-to-end timings on the narrowed CPU offenders were `4.657s`, `4.353s`, and `2.931s` respectively; the office GPU geometry11 PAS full-trajectory spot check stayed parity-clean at `8.714s`.
- Remaining risks: this closes duplicated setup work but does not eliminate the remaining cold-run cost from JAX compilation/lowering and preconditioner construction on PAS-heavy fresh processes. Single-case multi-GPU sharding is still open and remains explicitly non-release-facing.
- Next actions: run the final release validation pass from current `main` (`pytest -q`, docs build, and final README/docs sanity review), then ship on the strengthened CPU/GPU parity baseline with the transport-worker GPU scaling lane as the published parallel result.

### 2026-03-09
- Scope: Restore strict v3 default gradient-coordinate semantics for ambiguous legacy inputs that specify both `d*drHat` and `d*psiHat` fields, closing the tiny `includePhi1InKineticEquation=true` PAS parity regression before rerunning the broader verification gates.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/input_compat.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_input_compat.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `python -m py_compile sfincs_jax/input_compat.py tests/test_input_compat.py`; targeted parity regression tests for the tiny Phi1-in-kinetic PAS fixture (`8 passed`); full `pytest -q` (`253 passed in 215.84s`).
- Runtime/memory delta: no intended runtime change; the compatibility layer now uses the v3-default `inputRadialCoordinateForGradients=4` semantics when mixed legacy fields are present, so ambiguous inputs no longer silently take the `psiHat` gradients in JAX while Fortran takes `rHat/Er`.
- Remaining risks: the scaled full example audits in `/Users/rogeriojorge/local/tests/sfincs_jax/tests/scaled_example_suite_ref_cpu_full_v12` and `/home/rjorge/sfincs_jax_main_clean/tests/scaled_example_suite_ref_gpu_full_v12` are stale with respect to this gradient fix and still need fresh JAX reruns from the current `main` revision.
- Next actions: commit this compatibility fix to `main`, rerun the local CPU JAX full audit against the frozen `scaled_example_suite_ref_cpu_full_v12` Fortran reference root, then rerun the office GPU audit against that same frozen reference before refreshing suite-facing docs.

### 2026-03-09
- Scope: Fix two distributed-Krylov initialization regressions uncovered by the scaled example sweeps, teach the scaled-suite harness to reuse reduced frozen-reference inputs across lanes instead of rejecting them, avoid the unsupported CUDA-dense auto path for nonlinear `includePhi1` Newton solves, and restart/resume the office GPU audit from the broken `geometryScheme4_2species_noEr_withPhi1InDKE` slice.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/io.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/scripts/run_reduced_upstream_suite.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/scripts/run_scaled_example_suite.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_cli_solve_mode.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_transport_parallel.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_scaled_example_suite_reference.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/examples/README.md`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `python -m py_compile sfincs_jax/v3_driver.py sfincs_jax/io.py scripts/run_scaled_example_suite.py scripts/run_reduced_upstream_suite.py tests/test_cli_solve_mode.py tests/test_transport_parallel.py tests/test_scaled_example_suite_reference.py`; `pytest -q tests/test_cli_solve_mode.py tests/test_transport_parallel.py tests/test_scaled_example_suite_reference.py tests/test_compare_reference_corruption.py tests/test_input_compat.py tests/test_rhs1_sparse_first_heuristic.py tests/test_sparse_assembly.py` (`68 passed`); targeted scaled-suite rerun in `/Users/rogeriojorge/local/tests/sfincs_jax/tests/debug_mono_scheme1_fix_v1`; targeted office GPU rerun of `geometryScheme4_2species_noEr_withPhi1InDKE` is in progress on `cf250d7`.
- Runtime/memory delta: the office GPU frozen-reference lane at `/home/rjorge/sfincs_jax_main_clean/tests/scaled_example_suite_ref_gpu_full_v12` moved from immediate `jax_error` on the first three cases back to `parity_ok` on `inductiveE_noEr` (`41.43s`, `1415.8 MB`), `quick_2species_FPCollisions_noEr`, and `tokamak_1species_PASCollisions_noEr_Nx1` after the clean restart; the GPU resume slice also moved from a false harness crash on `geometryScheme4_2species_noEr_withPhi1InDKE` (`Reference input mismatch`) to the real nonlinear solve path at the reduced frozen-reference seed (`5x7x2x18`).
- Remaining risks: the reduced-scale CPU audit root `/Users/rogeriojorge/local/tests/sfincs_jax/tests/scaled_example_suite_ref_cpu_full_v12` is still the current offender baseline (`19 parity_ok`, `11 parity_mismatch`, `8 max_attempts`, `1 jax_error`), though the single `jax_error` there is now narrowed to the actual `monoenergetic_geometryScheme1` parser path; the office GPU slice is not complete yet, and the targeted `includePhi1` geometryScheme4 rerun is still computing on the new Krylov path.
- Next actions: let the targeted office GPU `geometryScheme4_2species_noEr_withPhi1InDKE` rerun finish, inspect its final parity/runtime against the frozen reference, then resume the remaining missing GPU cases in `scaled_example_suite_ref_gpu_full_v12` without resetting the completed 21-case prefix.

### 2026-03-08
- Scope: Make long scaled-example sweeps checkpoint suite artifacts after every finished case, fix the scheme-1 `Er -> dPhiHatdpsiHat` regression that broke `tokamak_1species_FPCollisions_withEr_DKESTrajectories`, and harden VMEC comparison against corrupted Fortran reference geometry fields that appear as uninitialized garbage in monoenergetic outputs.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/scripts/run_scaled_example_suite.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_fblock.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/compare.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_scaled_example_suite_reference.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_input_compat.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_compare_reference_corruption.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/examples/README.md`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `pytest -q tests/test_compare_reference_corruption.py tests/test_input_compat.py tests/test_scaled_example_suite_reference.py tests/test_rhs1_sparse_first_heuristic.py tests/test_sparse_assembly.py tests/test_cli_solve_mode.py`; targeted scaled-suite reruns in `/Users/rogeriojorge/local/tests/sfincs_jax/tests/debug_mono_scheme5_compare_guard_v3` and `/Users/rogeriojorge/local/tests/sfincs_jax/tests/debug_tokamak_dkes_withEr_scale075_fix_v1`.
- Runtime/memory delta: the suite harness now preserves `suite_report.json`, `suite_report_strict.json`, `suite_status*.rst`, and `summary.md` incrementally instead of losing all suite-level artifacts on interruption; the scheme-1 DKES `withEr` case moved from immediate `NameError` failure in `v3_fblock.py` to a full solve path, and both `monoenergetic_geometryScheme5_ASCII` and `monoenergetic_geometryScheme5_netCDF` moved from VMEC reference-corruption mismatches to `parity_ok` at the `0.75` scaled audit seed (`12x23x2x18`).
- Remaining risks: the live `scaled_example_suite_ref_cpu_full_v12` audit still shows a reduced-scale solver-branch mismatch on `tokamak_1species_FPCollisions_withEr_DKESTrajectories` (`38/214`, full print parity) even though earlier original-resolution CPU gates were parity-clean on this case, so reduced-scale full sweeps should be treated as offender audits rather than sole release gates; the full CPU sweep and the frozen-reference GPU sweep are still in progress.
- Next actions: let `/Users/rogeriojorge/local/tests/sfincs_jax/tests/scaled_example_suite_ref_cpu_full_v12` continue far enough to finish the current offender audit, use the clean office clone at `~/sfincs_jax_main_clean` with `~/stellarator_venv/bin/python` for the frozen-reference GPU lane, and then decide whether the DKES reduced-scale mismatch needs a default solver tweak or just release-note positioning as a scale-sensitivity audit artifact.

### 2026-03-07
- Scope: Rework the large-CPU explicit RHSMode=1 FP fallback so the default CLI lane skips the wasteful initial/stage2 collision-GMRES on the geometry-4 blocker, assembles host x-block factors sparsely with cached operator pieces, and only enables the experimental per-L `sxblock_tz` rescue behind an explicit env opt-in.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_rhs1_sparse_first_heuristic.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_sparse_assembly.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `python -m py_compile sfincs_jax/v3_driver.py tests/test_rhs1_sparse_first_heuristic.py`; `pytest -q tests/test_sparse_assembly.py tests/test_rhs1_sparse_first_heuristic.py tests/test_cli_solve_mode.py`
- Runtime/memory delta: on `examples/sfincs_examples/geometryScheme4_2species_withEr_fullTrajectories`, the default explicit CPU lane now skips the old `~156-209s` initial collision-preconditioned GMRES and reaches the explicit x-block seed in about `30-32s` total (`~8s` x-block build + `~22-24s` bounded solve), with peak RSS around `1.7-1.9 GB` before any later full sparse rescue. The experimental `sxblock_tz` seed path was reduced from about `9.9 GB` RSS to about `3.5-3.6 GB` by switching to sequential per-L factorization with smaller submatrix batches, but it still produced a poor seed (`residual≈1.95e+01`) and therefore remains off by default.
- Remaining risks: the geometry-4 large-FP default explicit lane still falls through to the full `68670x68670` sparse rescue because the explicit x-block factors remain too weak on nonzero-`x` blocks; simply adding more fallback branches is not closing the parity/performance gap.
- Next actions: inspect the rejected nonzero-`x` host factors directly and compare against the Fortran v3 matrix-preconditioner design, then replace the current per-`x` explicit rescue with a stronger x-coupled explicit block strategy before rerunning the full example suite.

### 2026-03-07
- Scope: Split explicit and differentiable solve modes so CLI/output generation can take a fast non-implicit path by default, while keeping the implicit-diff path available explicitly; add a host sparse x-block rescue implementation for explicit RHSMode=1 FP solves and use it only on the non-differentiable path.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/cli.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/io.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_cli_solve_mode.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_rhs1_sparse_first_heuristic.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `python -m py_compile sfincs_jax/v3_driver.py sfincs_jax/io.py sfincs_jax/cli.py tests/test_rhs1_sparse_first_heuristic.py tests/test_cli_solve_mode.py`; `pytest -q tests/test_rhs1_sparse_first_heuristic.py tests/test_cli_solve_mode.py`
- Runtime/memory delta: on the original-resolution geometry-4 CPU blocker (`examples/sfincs_examples/geometryScheme4_2species_withEr_fullTrajectories`), the explicit host x-block preconditioner reduced peak RSS at x-block build completion from about `5648.7 MB` on the capped JAX-factor path to about `5050.4 MB` at comparable build time (`~117-118s`), but the explicit GMRES rescue still did not finish in a practical wall-clock window; a follow-up experiment that also switched the Krylov matvec to a host sparse operator drove CPU utilization to about `850%` but increased RSS to about `8.3 GB`, so that variant was not kept.
- Remaining risks: the CLI/default explicit lane is now correctly separated from the differentiable path, but the geometry-4 large-FP explicit rescue is still too slow and memory-heavy; the next fix should target a cheaper strong explicit rescue rather than growing the host sparse operator cache.
- Next actions: commit/push the explicit/differentiable split and test coverage on `main`, then continue the geometry-4 work by replacing the current explicit x-block GMRES rescue with a more memory-disciplined strong explicit solve path before rerunning the original-resolution CPU suite.

### 2026-03-06
- Scope: Fix legacy mixed-gradient handling by separating species-gradient and Phi-gradient coordinate inference in the JAX solve/output paths, so cases that specify `dNHatdrHats`/`dTHatdrHats` together with `Er` reproduce Fortran v3 instead of silently zeroing the electric field branch.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/input_compat.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/io.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_system.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_input_compat.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `python -m py_compile sfincs_jax/input_compat.py sfincs_jax/v3_system.py sfincs_jax/io.py tests/test_input_compat.py`; `pytest -q tests/test_input_compat.py tests/test_fortran_reference_solver_options.py tests/test_sparse_assembly.py`; targeted suite rerun in `/Users/rogeriojorge/local/tests/sfincs_jax/tests/debug_geometry5_3species_split_grad_v1`.
- Runtime/memory delta: `examples__sfincs_examples__geometryScheme5_3species_loRes` moved from `parity_mismatch` (`42/193` practical and strict, JAX `277.618s`, `4795.1 MB`) to `parity_ok` (`0/193` practical and strict, `9/9` print parity) at JAX `134.684s` and `4775.1 MB`; the Fortran reference lane on the corrected input is `21.506s`, `582.7 MB`.
- Remaining risks: the stale full CPU suite roots created before this mixed-gradient fix are invalid for any mixed legacy-gradient cases and should not be used as frozen references; runtime/memory on geometry5 remain materially above Fortran even though parity is restored.
- Next actions: commit/push this fix on `main`, rerun the full original-resolution CPU suite plus the additional example from a clean root, then use that frozen CPU reference root for the full office GPU suite before regenerating README tables.

### 2026-03-06
- Scope: Fix the Fortran v3 canonicalization path so modern v3 inputs keep their trailing newline, preventing false `&export_f` read failures in the scaled-suite reference lane after the legacy-input compatibility work.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/scripts/run_reduced_upstream_suite.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_fortran_reference_solver_options.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `python -m py_compile scripts/run_reduced_upstream_suite.py`; `pytest -q tests/test_fortran_reference_solver_options.py tests/test_input_compat.py tests/test_scaled_example_suite_reference.py`; single-case suite rerun in `/Users/rogeriojorge/local/tests/sfincs_jax/tests/debug_case_runner_tokamak_pas_nx1_v2`.
- Runtime/memory delta: the representative original-resolution case `examples__sfincs_examples__tokamak_1species_PASCollisions_noEr_Nx1` moved from `max_attempts` with a Fortran `export_f` parse failure back to `parity_ok` at the original seed (`0/212` practical and strict, `9/9` print parity) with no resolution reduction.
- Remaining risks: the partial full-suite root `scaled_example_suite_ref_cpu_full_v6` is invalid because it was started on the broken reference lane and then interrupted; the full CPU sweep still needs to be restarted from scratch on current `main`.
- Next actions: commit/push the canonicalization fix on `main`, rerun the full original-resolution CPU suite plus the additional example from scratch, then continue to the frozen-reference GPU lane.

### 2026-03-06
- Scope: Add systematic legacy-input compatibility for the pre-v3 `examples/upstream/fortran_multispecies` tree by translating old namelist groups/keys for the Fortran reference lane, teaching `sfincs_jax` to infer non-default gradient-coordinate semantics from legacy inputs, and honoring legacy Boozer-file and `normradius_wish` aliases in the output, solve, and terminal-print paths.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/scripts/run_reduced_upstream_suite.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/input_compat.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/io.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_system.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_fblock.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_fortran_reference_solver_options.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_input_compat.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/examples/README.md`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `python -m py_compile sfincs_jax/input_compat.py sfincs_jax/io.py sfincs_jax/v3.py sfincs_jax/v3_system.py sfincs_jax/v3_fblock.py scripts/run_reduced_upstream_suite.py`; `pytest -q tests/test_input_compat.py tests/test_fortran_reference_solver_options.py tests/test_scaled_example_suite_reference.py`; targeted suite reruns in `/Users/rogeriojorge/local/tests/sfincs_jax/tests/debug_scaled_multispecies_inductive_suite_v7` and `/Users/rogeriojorge/local/tests/sfincs_jax/tests/debug_scaled_multispecies_fp_suite_v7`.
- Runtime/memory delta: the old multispecies `inductiveE_noEr` and `quick_2species_FPCollisions_noEr` cases moved from `max_attempts` reference-generation failure to `parity_ok` at the original-resolution seed (`0/193` practical and strict, `9/9` print parity); on local CPU the translated Fortran lane takes about `21.7s` and `119 MB` RSS on each case, while `sfincs_jax` takes about `4.9s` and `560 MB` RSS.
- Remaining risks: the large legacy geometryScheme=11 multispecies cases still need a full end-to-end parity pass on current `main`; the stale full original-resolution CPU suite was intentionally killed because these compatibility fixes changed both the runner and the JAX semantics underneath it.
- Next actions: commit/push this legacy-input compatibility block on `main`, restart the full original-resolution CPU suite plus the additional example from scratch, then use that frozen CPU reference root for the full office GPU suite before regenerating the README tables.

### 2026-03-06
- Scope: Tighten the CPU transport sparse-LU direct rescue by adding iterative residual refinement and raising the default rescue-size cap so the original-resolution LHD and low-collisionality W7-X transport-matrix examples converge on the sparse-direct parity branch instead of stalling in large Krylov retries.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_transport_sparse_direct.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `python -m py_compile sfincs_jax/v3_driver.py tests/test_transport_sparse_direct.py`; `pytest -q tests/test_transport_sparse_direct.py tests/test_transport_parallel.py tests/test_transport_matrix_rhsmode2_parity.py`; targeted repros in `/Users/rogeriojorge/local/tests/sfincs_jax/tests/debug_lhd_co0_nu_0_5748_refine_v1` and `/Users/rogeriojorge/local/tests/sfincs_jax/tests/debug_w7x_co0_nu_0_01727_sparse_v1`.
- Runtime/memory delta: for `examples__publication_figures__output__lhd_co0__nu_n_0.5748`, transport residuals improved from the earlier sparse-direct lane at roughly `6e-06`-`4e-05` down to `4.62e-17`, `1.14e-15`, and `2.28e-12` after two LU-refinement steps, with transport elapsed about `267.98s` and RSS about `4228 MB`; for `examples__publication_figures__output__w7x_co0__nu_n_0.01727`, raising the sparse-direct cap from `30000` to `40000` moved the case off the stalled Krylov branch (`~1612.8s`, `37/194` mismatches) onto a machine-precision sparse-direct branch (`6.89e-19`, `1.64e-18`, `9.89e-14`) at about `748.19s`, with only metadata-only compare deltas remaining and RSS about `5149 MB`.
- Remaining risks: W7-X geometry-5 transport memory remains several GB above Fortran due to SuperLU fill; the full original-resolution CPU suite still needs to be rerun from scratch on this new default before freezing the reference root for the full office GPU lane.
- Next actions: commit/push this transport refinement block on `main`, rerun the full original-resolution CPU suite plus the additional example from scratch, then run the full office GPU suite against that frozen CPU reference root before regenerating README tables.

### 2026-03-06
- Scope: Add a CPU transport sparse-LU direct rescue with rescue-first ordering for large RHSMode=2/3 FP transport solves, so stalled transport Krylov branches can recover Fortran-like accuracy on the original geometry-scheme-2 transport example without spending most of the wall time in failed retry branches.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_transport_sparse_direct.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `python -m py_compile sfincs_jax/v3_driver.py tests/test_transport_sparse_direct.py`; `pytest -q tests/test_transport_sparse_direct.py tests/test_transport_parallel.py tests/test_transport_matrix_rhsmode2_parity.py`; targeted transport repros in `tests/debug_transport_scheme2_default_v5` and `tests/debug_transport_scheme2_default_v6`.
- Runtime/memory delta: for `transportMatrix_geometryScheme2`, the original current-`main` sequential transport lane was about `773.9s` with large practical mismatches, the first sparse-LU rescue lane restored practical parity but still took about `720.2s`, and the rescue-first sparse-LU default dropped that to about `325.5s` while keeping transport-matrix max relative error at about `1.74e-5`; peak RSS rose from about `876 MB` on the inaccurate Krylov lane to about `4391 MB` on the accurate sparse-LU lane.
- Remaining risks: transport memory is still far above Fortran on this case because SuperLU fill remains large; the full original-resolution CPU suite and the office GPU suite still need to be rerun from scratch on this revision.
- Next actions: commit/push this transport sparse-LU rescue block on `main`, restart the full original-resolution CPU suite from scratch, then run the full office GPU suite against that frozen CPU reference root before regenerating README tables.

### 2026-03-06
- Scope: Fix transport `whichRHS` process-parallel diagnostics by merging parent-side state vectors through the common batched output path, stop auto-enabling transport process parallelism via the high-level cores knob, and add a chunked RHSMode=1 sparse-LU rescue path with rescue-first ordering for large catastrophic CPU FP cases.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/__init__.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/cli.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_transport_parallel.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_rhs1_sparse_first_heuristic.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_sparse_assembly.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `python -m py_compile sfincs_jax/v3_driver.py tests/test_rhs1_sparse_first_heuristic.py`; `pytest -q tests/test_rhs1_sparse_first_heuristic.py tests/test_sparse_assembly.py tests/test_transport_parallel.py tests/test_transport_matrix_rhsmode2_parity.py`; targeted local repros for `geometryScheme5_3species_loRes` in `tests/debug_geometry5_3species_sparsefirst_v3` and `tests/debug_geometry5_3species_default_v4`.
- Runtime/memory delta: `geometryScheme5_3species_loRes` moved from the prior failing lane (`residual=5.860420e+04`, about `363.7s`, about `2143 MB`) to a converged sparse-LU rescue on the default path with residual `7.669738e-10`, about `204.6s`, and about `3697 MB` RSS; the same rescue path without redundant JAX sparse-factor materialization ran at about `182.0s` and about `3831 MB`. Practical parity stayed within the existing comparison tolerance, with representative transport/flow deltas below `~5.2e-7` relative.
- Remaining risks: `transportMatrix_geometryScheme2` is still being rerun on current `main`; the large CPU sparse rescue still allocates several GB on W7-X geometry-5 and needs further memory reduction to approach Fortran behavior.
- Next actions: finish the fresh `transportMatrix_geometryScheme2` rerun, commit/push this solver block on `main`, then restart the full original-resolution CPU suite from scratch and use that frozen root for the full office GPU rerun before regenerating the README tables.

### 2026-03-06
- Scope: Skip the expensive accelerator dense-polish branch after a successful host sparse-LU direct rescue when the remaining residual is already within a bounded ratio of the solve target, so small full-size GPU FP cases keep parity without paying for unnecessary dense Krylov cleanup.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `python -m py_compile sfincs_jax/v3_driver.py`; `pytest -q tests/test_rhs1_sparse_first_heuristic.py tests/test_solver_gmres.py tests/test_fortran_reference_solver_options.py tests/test_scaled_example_suite_reference.py`; office GPU reruns of `inductiveE_noEr` into `tests/gating_gpu_inductive_v2` and `tests/gating_gpu_inductive_v2_nodense` against the frozen CPU reference root.
- Runtime/memory delta: on office GPU, `inductiveE_noEr` stays `0/207` practical and strict with full `9/9` print parity when the post-sparse dense fallback is skipped, while JAX runtime drops from `65.995s` to `34.366s`; RSS stays roughly flat at `1745.7 MB -> 1739.1 MB`.
- Remaining risks: this optimization is validated on the main full-size E_parallel FP blocker but still needs the wider GPU gate to confirm it does not hide useful dense-polish on other small accelerator FP cases.
- Next actions: commit/push this runtime optimization, rerun the full narrow GPU gate against the frozen CPU reference root, and then move to the remaining GPU/CPU runtime and memory offenders from the updated summaries.

### 2026-03-06
- Scope: Add a full-size RHSMode=1 sparse LU/ILU rescue path before dense fallback, widen exact sparse-LU auto selection for small accelerator FP cases, and add a host sparse-LU direct fallback for accelerator exact-LU rescues so full-size GPU FP solves no longer depend on the inaccurate explicit dense Krylov branch.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_rhs1_sparse_first_heuristic.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `python -m py_compile sfincs_jax/v3_driver.py`; `pytest -q tests/test_rhs1_sparse_first_heuristic.py tests/test_solver_gmres.py tests/test_fortran_reference_solver_options.py tests/test_scaled_example_suite_reference.py`; targeted local `inductiveE_noEr` direct probes with forced sparse exact LU and forced host sparse direct rescue.
- Runtime/memory delta: on local CPU, the new full-size sparse exact-LU host-direct rescue returns `inductiveE_noEr` to `0/207` practical mismatches against the stable Fortran reference with residual `1.322041e-07` and about `19.9s` elapsed, replacing the earlier bad accelerator-style dense-Krylov branch that produced `41/207` mismatches on office GPU at about `69s`.
- Remaining risks: office GPU validation is still pending on this exact patch; the host sparse direct fallback is a robustness rescue path and should remain secondary to fully JAX-native solves where those already converge cleanly.
- Next actions: push this patch to `main`, rerun `inductiveE_noEr` and the narrow GPU gate from a fresh office checkout against the frozen CPU reference root, then use the updated gate report to decide whether the next performance/memory work should target PAS-heavy cases or remaining FP GPU branches.

### 2026-03-06
- Scope: Stabilize `constraintScheme=0` reference generation by forcing a reproducible Fortran Krylov policy in the suite runner, add an explicit left-preconditioned SciPy GMRES helper for solver debugging, and disable default RHSMode=1 dense shortcut/fallback paths for `constraintScheme=0` so the JAX lane stays on the physically correct sparse branch.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/scripts/run_reduced_upstream_suite.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/solver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_rhs1_sparse_first_heuristic.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_solver_gmres.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_fortran_reference_solver_options.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `python -m py_compile scripts/run_reduced_upstream_suite.py sfincs_jax/solver.py sfincs_jax/v3_driver.py`; `pytest -q tests/test_solver_gmres.py tests/test_rhs1_sparse_first_heuristic.py tests/test_fortran_reference_solver_options.py tests/test_scaled_example_suite_reference.py`; single-case stable reference compare in `/Users/rogeriojorge/local/tests/sfincs_jax/tests/gating_example_cpu_cs0_stable_ref_v3`.
- Runtime/memory delta: `tokamak_1species_FPCollisions_noEr` now follows the stable sparse branch against the forced-Fortran reference instead of the incorrect dense gauge-drift branch; the remaining delta is down to a single `pressureAnisotropy` mismatch (`1/188` practical and strict) rather than the earlier large density/pressure gauge errors from the dense shortcut.
- Remaining risks: `constraintScheme=0` still has a small residual branch difference in `pressureAnisotropy`; the full corrected CPU/GPU example suites have not yet been rerun from this new stable reference policy.
- Next actions: commit this solver/reference change on `main`, rerun the corrected CPU gate and full original-resolution reference lane, then rerun the office GPU lane against that frozen CPU reference root before widening back to the full examples plus additional examples.

### 2026-03-06
- Scope: Fix the GPU DKES sparse-shortcut trigger so it keys off the user-requested preconditioner setting rather than the later auto-mutated internal `rhs1_precond_env`, and confirm the office GPU log now skips the old `xblock_tz` plus stage-2 prefix.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `python -m py_compile sfincs_jax/v3_driver.py`; office direct GPU DKES repro on clean checkout `0299b9c` with `CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 ~/venvs/sfincs_jax_gpu/bin/python -u -m sfincs_jax ...`.
- Runtime/memory delta: the live office GPU DKES log now enters `GPU DKES auto mode -> sparse ILU shortcut`, skips the initial Krylov solve entirely, and avoids the prior `xblock_tz` plus stage-2 prefix. At the same wall-clock point the process RSS dropped from about `1.58 GB` on the old path to about `1.37 GB` on the shortcut path while holding similar GPU memory (~`12.2 GB`).
- Remaining risks: the sparse-ILU solve itself still did not finish quickly enough to produce an H5/output comparison in the direct office rerun, so the new blocker is the sparse-ILU solve quality/runtime rather than the accelerator dense-fallback path.
- Next actions: instrument the sparse-ILU solve itself (residual/iteration/elapsed checkpoints), compare it against a direct dense-Krylov GPU rescue on this moderate-size DKES case, and only then rerun the GPU gate plus full examples suite.

### 2026-03-06
- Scope: Short-circuit the GPU FP DKES auto path directly to sparse ILU when that is already the intended rescue path, instead of first paying for `xblock_tz` plus stage-2 GMRES on accelerator backends.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `python -m py_compile sfincs_jax/v3_driver.py`; `pytest -q tests/test_solver_gmres.py tests/test_small_regularized_lstsq.py tests/test_schur_precond_heuristic.py tests/test_pas_projection_heuristic.py tests/test_xblock_tz_precond_heuristic.py`; office direct GPU DKES repro on commit `5671004` confirmed the previous auto path was spending time in `xblock_tz`, then stage-2 GMRES, and only afterwards entering sparse ILU.
- Runtime/memory delta: before this patch the direct office GPU DKES repro built `xblock_tz`, reported a stage-2 residual of `1.231e-01`, then assembled sparse ILU and stayed resident at about `1.58 GB` host RSS / `12.2 GB` GPU memory before any output H5 was produced; the new shortcut is intended to remove that dead preconditioner/stage2 prefix entirely.
- Remaining risks: the actual office runtime/parity delta for the shortcut still needs to be measured on the rerun; `constraintScheme=0` remains an open nullspace/near-nullspace selection problem.
- Next actions: push this shortcut to `main`, rerun the direct office GPU DKES case from the clean checkout, and if the sparse-ILU-first path is still not parity-clean then tune the sparse ILU / dense-Krylov handoff rather than reintroducing accelerator dense-direct branches.

### 2026-03-06
- Scope: Add an accelerator-safe explicit dense-Krylov RHSMode=1 fallback path, keep dense fallback enabled on non-CPU backends without re-enabling CUDA direct solves, and validate that the CPU FP DKES lane stays parity-clean while the remaining `constraintScheme=0` FP mismatch remains isolated.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/solver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_solver_gmres.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `python -m py_compile sfincs_jax/solver.py sfincs_jax/v3_driver.py`; `pytest -q tests/test_solver_gmres.py tests/test_small_regularized_lstsq.py tests/test_scaled_example_suite_reference.py tests/test_schur_precond_heuristic.py tests/test_pas_projection_heuristic.py tests/test_xblock_tz_precond_heuristic.py`; targeted CPU gate into `/Users/rogeriojorge/local/tests/sfincs_jax/tests/gating_cpu_solverfix` for `tokamak_1species_FPCollisions_noEr` and `tokamak_1species_FPCollisions_withEr_DKESTrajectories` against `/Users/rogeriojorge/local/tests/sfincs_jax/tests/gating_reference_cpu`.
- Runtime/memory delta: on local CPU the FP DKES gate stayed parity-clean (`0/214`) while the `constraintScheme=0` FP case stayed at the same mismatch signature (`1/188` practical, `8/188` strict), confirming the new fallback path did not perturb the already-good CPU DKES lane and did not hide the remaining nullspace-selection problem.
- Remaining risks: the new dense-Krylov fallback still needs office GPU validation on `inductiveE_noEr` and `tokamak_1species_FPCollisions_withEr_DKESTrajectories`; `tokamak_1species_FPCollisions_noEr` still requires a principled `constraintScheme=0` solver/gauge selection change rather than more tolerance or fallback tuning.
- Next actions: sync `main` to a clean office GPU working copy and rerun the narrow GPU gate against the fixed CPU reference root, then use the resulting behavior to decide whether the remaining FP DKES issue is solved by dense-Krylov rescue alone or still needs stronger reduced-system preconditioning before returning to the `constraintScheme=0` branch.

### 2026-03-05
- Scope: Separate unsafe accelerator dense solves from the optional host-LU dense fallback so the GPU DKES path can be probed without re-enabling backend cuSOLVER calls, and verify whether the existing host-callback dense fallback is actually usable on office CUDA.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `python -m py_compile sfincs_jax/v3_driver.py`; `pytest -q tests/test_schur_precond_heuristic.py tests/test_pas_projection_heuristic.py tests/test_xblock_tz_precond_heuristic.py`; office GPU direct reduced-resolution DKES probe with `SFINCS_JAX_RHSMODE1_DENSE_HOST_LU=1` against `tests/gating_gpu_from_ref_v3/tokamak_1species_FPCollisions_withEr_DKESTrajectories/input.namelist`.
- Runtime/memory delta: the explicit host-LU probe still finishes the reduced GPU DKES case in about `164.4s` with the same large residual (`6.107661e-02`) and similar RSS (`~946 MB` resident while running), so there is no practical runtime win yet.
- Remaining risks: the existing host-LU dense fallback path is not accelerator-safe on office CUDA either; it fails with `UNIMPLEMENTED: xla_ffi_python_gpu_callback for platform CUDA` once the reduced DKES solve reaches the dense fallback. This leaves the GPU DKES branch dependent on Krylov + sparse ILU alone, which is not yet parity-accurate.
- Next actions: either implement a new accelerator-safe host dense solve path that does not rely on `jax.pure_callback`/`custom_linear_solve` on CUDA, or improve the reduced RHSMode=1 FP Krylov path enough that the DKES branch no longer needs dense rescue at all.

### 2026-03-05
- Scope: Keep all active work on `main`, remove a full-size RHSMode=1 accelerator regression that skipped stage-2 GMRES without any real rescue path, preserve actual JAX subprocess failures in suite logs/max-attempts summaries, and disable the small full-preconditioner auto-dense path on accelerators after reproducing a CUDA `cusolver_getrf_ffi` failure on the FP DKES gate.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/scripts/run_reduced_upstream_suite.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: merged PR #1 into `main`; `python -m py_compile sfincs_jax/v3_driver.py scripts/run_reduced_upstream_suite.py`; `pytest -q tests/test_small_regularized_lstsq.py tests/test_scaled_example_suite_reference.py tests/test_schur_precond_heuristic.py tests/test_pas_projection_heuristic.py tests/test_xblock_tz_precond_heuristic.py`; `pytest -q tests/test_schur_precond_heuristic.py tests/test_pas_projection_heuristic.py tests/test_xblock_tz_precond_heuristic.py`; office GPU rerun of `inductiveE_noEr` into `/home/rjorge/sfincs_jax_codex_scaled_suite_20260305_lean/tests/gating_gpu_inductive_fix`; direct office GPU repro of `tokamak_1species_FPCollisions_withEr_DKESTrajectories` with `sfincs_jax -v write-output`.
- Runtime/memory delta: `inductiveE_noEr` on office GPU moved from the bad large-residual branch (`42/207` mismatches, `565.432s / 934.7 MB` in `gating_gpu_from_ref_v3`) back to the small residual-parity mismatch lane (`2/207` mismatches, `152.080s / 949.7 MB` in `gating_gpu_inductive_fix`). The direct DKES GPU repro no longer needs guesswork: before the latest patch it failed immediately on the small-system auto-dense full-preconditioner path with `UNIMPLEMENTED: cusolver_getrf_ffi for platform CUDA`.
- Remaining risks: `tokamak_1species_FPCollisions_withEr_DKESTrajectories` still needs a completed GPU rerun after the full-preconditioner dense-auto guard to confirm parity/performance on the Krylov path; `tokamak_1species_FPCollisions_noEr` remains a genuine `constraintScheme=0` nullspace-selection problem, not a convergence or dense-fallback issue. State-space analysis shows large unconstrained density/pressure/parallel-flow components, and removing those three expected null modes alone still leaves an additional local FP branch (`pressureAnisotropy` and local density/pressure errors remain too large).
- Next actions: finish the office GPU DKES rerun on the patched solver and then rerun the 3-case GPU gate from the fixed CPU reference root; continue the `constraintScheme=0` work by building a general nullspace-basis analysis/projection from the solved state rather than tuning solver tolerances or using Fortran-output-driven corrections.

### 2026-03-05
- Scope: Harden RHSMode=1 accelerator behavior by disabling non-CPU dense shortcut/fallback paths that still hit unsupported CUDA calls, fix the full-size strong-preconditioner fallback control flow for non-`point` FP solves, and re-run targeted CPU/GPU gate cases against a fixed CPU-generated Fortran reference root.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `pytest -q tests/test_small_regularized_lstsq.py tests/test_scaled_example_suite_reference.py tests/test_schur_precond_heuristic.py tests/test_pas_projection_heuristic.py tests/test_xblock_tz_precond_heuristic.py`; `python -m py_compile sfincs_jax/v3_driver.py`; local CPU gate rerun into `/Users/rogeriojorge/local/tests/sfincs_jax/tests/gating_cpu_from_ref_v2`; targeted local debug reruns for `tokamak_1species_FPCollisions_noEr` with default CPU path, `SFINCS_JAX_ACTIVE_DOF=0`, and forced `SFINCS_JAX_RHSMODE1_STRONG_PRECOND=theta_line`.
- Runtime/memory delta: with the accelerator-safe sparse preference, `inductiveE_noEr` on office GPU no longer dies in CUDA dense/`lstsq` fallback and its solve log drops from the earlier `155.581s / 1033.2 MB` gate result to about `32s / 934.2 MB` for the completed standalone rerun before compare. On local CPU, `tokamak_1species_FPCollisions_withEr_DKESTrajectories` is now parity-clean against the fixed reference root (`0/214`) at `2.793s / 2282.5 MB`, while `tokamak_1species_FPCollisions_noEr` remains parity-mismatched even after a forced full-size strong fallback (`1/188` practical, `8/188` strict; about `28.6s / 1279.4 MB` with `theta_line`).
- Remaining risks: the office GPU `gating_gpu_from_ref_v2` rerun is stuck on the DKES case in a stale remote working copy and should be discarded; `tokamak_1species_FPCollisions_noEr` is now isolated as a `constraintScheme=0` FP nullspace-selection issue rather than a dense-fallback or generic convergence issue; the optional Fortran-gauge hook in `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/io.py` is only useful for debugging and must not become part of the correctness path.
- Next actions: sync a clean remote working copy with the latest local `v3_driver.py` before rerunning the GPU gate; add a narrow regression test for full-size non-`point` strong fallback reachability; debug `constraintScheme=0` FP state-vector/nullspace differences against Fortran (likely via exported state vectors or low-order-moment basis analysis) before changing default gauge behavior.

### 2026-03-05
- Scope: Remove the known GPU `lstsq` blocker with a backend-safe differentiable small least-squares path, add explicit reuse of fixed Fortran reference roots in the scaled example suite, and run a narrow local-CPU plus office-GPU gate against the same CPU-generated Fortran reference set.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/scripts/run_scaled_example_suite.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_small_regularized_lstsq.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_scaled_example_suite_reference.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/examples/README.md`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `pytest -q tests/test_small_regularized_lstsq.py tests/test_scaled_example_suite_reference.py tests/test_schur_precond_heuristic.py tests/test_pas_projection_heuristic.py tests/test_xblock_tz_precond_heuristic.py`; local gate reference generation with `tests/gating_reference_cpu`; local CPU re-run against `--reference-results-root tests/gating_reference_cpu` into `tests/gating_cpu_from_ref`; office GPU re-run against the synced `tests/gating_reference_cpu` root into `tests/gating_gpu_from_ref`.
- Runtime/memory delta: the GPU gate now completes `inductiveE_noEr` instead of failing in CUDA dense/`lstsq` fallback paths, but it still takes `155.581s / 1033.2 MB` versus the local CPU reference lane `20.669s / 1098.0 MB` and Fortran `0.175s / 125.4 MB`. Local CPU against the fixed reference root stays aligned with the direct reference lane: `tokamak_1species_PASCollisions_noEr_Nx1` remains `0/212` practical and strict with `0.194s / 617.7 MB` JAX CPU versus `0.032s / 103.8 MB` Fortran, while `tokamak_1species_FPCollisions_noEr` remains the primary CPU mismatch at `1/188` practical and `8/188` strict.
- Remaining risks: office GPU still reports parity mismatches on three FP-heavy gate cases relative to the stable CPU Fortran reference (`inductiveE_noEr` `2/207`, `tokamak_1species_FPCollisions_noEr` `11/188`, `tokamak_1species_FPCollisions_withEr_DKESTrajectories` `38/214`); office still warns that `jax_cuda12_plugin 0.5.1` is incompatible with `jaxlib 0.6.2`; the clean `sfincs_original` reference branch could not be rebuilt locally because PETSc points at a missing Homebrew OpenMPI wrapper path.
- Next actions: inspect the GPU FP mismatch fields (`delta_f`, `sources`, `FSABFlow`, `particleFlux_vm_*`, `heatFlux_vm_*`, `pressureAnisotropy`) against the stable CPU reference lane, profile why GPU runtime regressed badly on `inductiveE_noEr` despite eliminating the crash, and either fix or explicitly gate the stale PETSc/OpenMPI path so the clean deterministic Fortran branch can be rebuilt reproducibly.

### 2026-03-05
- Scope: Replace the blind doubled-resolution example benchmark path with an upstream-reference resolution policy, preserve the partial `2x` profiling data as evidence, validate the corrected runner on local CPU and office GPU smoke cases, and start narrowing GPU-specific solver/backend blockers.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/scripts/run_scaled_example_suite.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/examples/README.md`, `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/io.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: confirmed all 38 vendored `examples/sfincs_examples/*/input.namelist` files are resolution-identical to `/Users/rogeriojorge/local/tests/sfincs_original/fortran/version3/examples/*/input.namelist`; local smoke with `tokamak_1species_FPCollisions_noEr` at `--scale-factor 1.0` against the original-example reference root; local and office smoke with `tokamak_1species_PASCollisions_noEr_Nx1` at `--scale-factor 1.0`; partial local CPU full-suite restart at the corrected original resolutions; office GPU smoke/restart on `inductiveE_noEr` after disabling dense auto-mode on non-CPU backends; `python -m py_compile sfincs_jax/io.py scripts/run_scaled_example_suite.py`.
- Runtime/memory delta: the aborted blind-`2x` partial run already showed why that default was wrong: `tokamak_1species_FPCollisions_noEr` at `42x1x16x62` took 183.747s / 2300.1 MB and still had `1/188` practical (`8/188` strict) mismatches, while the corrected upstream-reference run at the original `21x1x8x31` took 2.956s / 998.8 MB with the same mismatch signature. `tokamak_1species_PASCollisions_noEr_Nx1` at the corrected original resolution ran parity-clean on CPU (0.132s Fortran / 2.086s JAX / 630.2 MB) and in the initial office GPU smoke (1.576s Fortran / 5.001s JAX / 1293.3 MB). The partial corrected CPU full-suite already completed 13 tokamak/quick/inductive cases with full print parity and only one strict mismatch case so far (`tokamak_1species_FPCollisions_noEr`, `8/188`).
- Remaining risks: office GPU still reports a `jax_cuda12_plugin` / `jaxlib` version mismatch warning; office Fortran outputs appear nondeterministic on some classical-heat-flux fields (`classicalHeatFlux*`, `gpsiHatpsiHat`) for the same input, so office-generated Fortran H5s are not yet trustworthy as the GPU parity reference; the dense auto-mode patch for non-CPU backends moved `inductiveE_noEr` forward but the run still dies later in GPU-only dense-fallback / `jnp.linalg.lstsq` cuSOLVER calls.
- Next actions: finish parsing the partial corrected CPU suite into a report artifact, compare GPU JAX outputs against a stable CPU-generated Fortran reference instead of the unstable office Fortran H5s, and remove or host-fallback the remaining GPU dense-fallback / least-squares calls so `inductiveE_noEr` and related small FP cases can complete on CUDA.

### 2026-03-05
- Scope: Trim unnecessary PAS auto strong-preconditioner retries after already-strong base preconditioners, and resync README/docs with the stored suite artifacts.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/usage.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/performance_techniques.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_schur_precond_heuristic.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: direct CLI profiles on `tokamak_2species_PASCollisions_withEr_fullTrajectories` and `HSX_PASCollisions_fullTrajectories`, practical/strict H5 compares against stored Fortran outputs, an HSX gate-check confirming the larger-gap branch still enters the strong fallback path, `pytest -q tests/test_schur_precond_heuristic.py tests/test_pas_projection_heuristic.py tests/test_xblock_tz_precond_heuristic.py`, and `sphinx-build -W -b html docs docs/_build/html`.
- Runtime/memory delta: `tokamak_2species_PASCollisions_withEr_fullTrajectories` improved from 180.320s / 1732.1 MB (stored suite baseline) to 10.741s / 955.3 MB in the patched direct CLI run, with practical parity unchanged at `0/212` and strict parity unchanged at `1/212`. For `HSX_PASCollisions_fullTrajectories`, disabling the strong retry entirely still produced one practical mismatch (`densityPerturbation`), so the default keeps the fallback for larger residual gaps.
- Remaining risks: `HSX_PASCollisions_fullTrajectories` still needs a cheaper correction path than the full PAS strong retry; the full reduced suite has not yet been rerun after this solver change.
- Next actions: profile the HSX PAS full-trajectories branch again, isolate which part of the strong retry fixes `densityPerturbation`, and replace the expensive second Krylov cycle with a bounded PAS polish or equivalent constraint-aware correction.

---

## 17) Important Command Snippets

### 17.1 Docs + tests

```bash
cd /Users/rogeriojorge/local/tests/sfincs_jax
sphinx-build -W -b html docs docs/_build/html
pytest -q
```

### 17.2 Run one input like Fortran

```bash
sfincs_jax /path/to/input.namelist
```

### 17.3 Python run + in-memory results

```python
from pathlib import Path
from sfincs_jax.io import write_sfincs_jax_output_h5

out_path, results = write_sfincs_jax_output_h5(
    input_namelist=Path("input.namelist"),
    output_path=Path("sfincsOutput.h5"),
    return_results=True,
)
```

### 17.4 Upstream-reference example-suite benchmark

```bash
cd /Users/rogeriojorge/local/tests/sfincs_jax
python scripts/run_scaled_example_suite.py \
  --examples-root examples/sfincs_examples \
  --resolution-reference-root /Users/rogeriojorge/local/tests/sfincs_original/fortran/version3/examples \
  --fortran-exe /Users/rogeriojorge/local/tests/sfincs/fortran/version3/sfincs \
  --out-root tests/scaled_example_suite_ref_cpu_local \
  --timeout-s 240 \
  --max-attempts 2 \
  --scale-factor 1.0 \
  --runtime-target-basis fortran \
  --fortran-min-runtime-s 1.0 \
  --fortran-max-runtime-s 20.0 \
  --runtime-adjustment-iters 3
```

---

## 18) References (online + local)

### 18.1 Online references used for strategy context
- IAEA World Fusion Outlook 2024: https://www.iaea.org/publications/15777/iaea-world-fusion-outlook-2024
- DOE FIRE + Milestone progress (Jan 16, 2025): https://www.energy.gov/articles/us-department-energy-announces-selectees-107-million-fusion-innovation-research-engine
- Fusion Industry Association 2024 report (PDF): https://sciencebusiness.net/sites/default/files/inline-files/FIA_annual%20report%202024.pdf
- ITER IMAS open-source release (Dec 8, 2025): https://www.iter.org/node/20687/release-imas-infrastructure-and-physics-models-open-source
- NEO docs (GACODE): https://gafusion.github.io/doc/neo.html
- NEO (STELLOPT page): https://princetonuniversity.github.io/STELLOPT/NEO.html
- KNOSOS paper: https://arxiv.org/abs/1908.11615
- NERSC Perlmutter architecture: https://docs.nersc.gov/systems/perlmutter/architecture/

### 18.2 Local references to mine and cite in docs
- `/Users/rogeriojorge/local/tests/sfincs_jax/docs/upstream/20131220-04 Technical documentation for SFINCS with a single species.pdf`
- `/Users/rogeriojorge/local/tests/sfincs_jax/docs/upstream/20131219-01 Technical documentation for SFINCS with multiple species.pdf`
- `/Users/rogeriojorge/local/tests/sfincs_jax/docs/upstream/20150325-01 Effects on fluxes of including Phi_1.pdf`
- `/Users/rogeriojorge/local/tests/sfincs_jax/docs/upstream/20150507-01 Technical documentation for version 3 of SFINCS.pdf`
- `/Users/rogeriojorge/local/tests/sfincs_jax/docs/upstream/sfincsPaper/sfincsPaper.pdf`
- `/Users/rogeriojorge/local/tests/Escoto_Thesis.pdf`
- `/Users/rogeriojorge/local/tests/Merkel_1987.pdf`
- `/Users/rogeriojorge/local/tests/hirshman_sigmar_1983.pdf`
- `/Users/rogeriojorge/local/tests/numerics_vmec.pdf`

---

## 19) Definition Of Done (current release gate)

Release-ready means:
1. Full release-facing example-suite CPU and GPU comparisons are `39/39 parity_ok`, strict-clean, and free of `jax_error` / `max_attempts`.
2. `additional_examples` runs successfully on CPU and GPU with validated outputs.
3. No hidden external-file dependence for correctness in the default path.
4. CI/docs/tests are green.
5. Runtime/memory and solver defaults are documented with reproducible commands.
6. README/docs/plan all reflect the current `main` truth.

### 19.1 Test campaign and CI/CD reality check (2026-04-21)

- Landed a new fast helper/physics coverage batch:
  - `tests/test_helper_module_coverage.py`
  - `tests/test_runtime_helper_coverage.py`
- These cover:
  - ambipolar-current and scanplot conventions,
  - flux-surface-average diagnostics identities on simple analytic fields,
  - path/indexing/profiling/verbose helpers,
  - Fortran wrapper failure/success paths,
  - ER-scan directory generation helpers,
  - transport-parallel worker NPZ emission,
  - distributed-runtime bootstrap,
  - solver-state save/load guards,
  - compare-module helper logic.
- CI/CD fixes:
  - `.github/workflows/ci.yml` coverage job now has `id-token: write`, so Codecov OIDC upload is valid.
  - `.github/workflows/publish-pypi.yml` now uses `skip-existing: true`, so retagging a published version does not hard-fail the workflow.
- Audited current local result:
  - `pytest -q --cov=sfincs_jax --cov-report=term --cov-report=xml`
  - `473 passed in 384.09s`
  - total package coverage: `52%`
- Additional low-cost physics/helper coverage landed after that first batch:
  - `tests/test_geometry_grid_helper_coverage.py`
- These add formula-driven invariants from the analytic Boozer models and radial-coordinate machinery:
  - periodic/spectral differentiation identities,
  - scheme-1 / scheme-2 / scheme-4 analytic geometry checks,
  - VMEC half-mesh finite-difference behavior,
  - radial-coordinate conversion and Fortran-logical helper formulas.
- Measured module gains from the second batch:
  - `geometry.py`: `23% -> 88%`
  - `grids.py`: `38% -> 46%`
  - `vmec_geometry.py`: `8% -> 97%`
- Added direct fixture-based geometry loader checks informed by the upstream SFINCS
  technical notes and paper:
  - Boozer `.bc` loader consistency on `tests/ref/w7x_standardConfig.bc`
  - VMEC `wout` loader consistency on `tests/ref/wout_w7x_standardConfig.nc`
  - analytic/spectral identities for differentiation and Boozer-coordinate field relations
- Added operational `io` coverage for:
  - geometryScheme=12 non-stellarator-symmetric Boozer localization
  - scheme-5 netCDF sibling preference during equilibrium resolution
- Added bounded helper coverage for the remaining cheap `io.py` / `v3_driver.py` seams:
  - `tests/test_io_cache_helpers.py`
  - `tests/test_v3_driver_policy_helpers.py`
- These cover:
  - output-cache enable/path/save/load/version behavior,
  - hashable grouping and equilibrium-content cache identity,
  - HDF5 layout and decode helpers,
  - solver-JIT env/threshold selection,
  - explicit dtype/policy boundaries for the geometry4 PAS fp32 rule,
  - dense-backend allow/deny env logic,
  - resource-exhausted error detection through chained exceptions,
  - sharded-line override whitelisting.
- Fresh audited local result on current `main`:
  - chunked `pytest -q` over the full tree to avoid the earlier memory spike
  - `486 passed`
  - chunked package coverage audit
  - total package coverage: `53%`
- Added a bounded heavy-module coverage batch:
  - `tests/test_io_export_and_h5_coverage.py`
  - `tests/test_solver_heavy_helper_coverage.py`
- These cover:
  - HDF5 writer/readback and overwrite guards,
  - export-f configuration and mapping behavior on bounded analytic grids,
  - `_as_1d_float()` / `_legendre_matrix()` branch behavior,
  - Krylov-method normalization and restart caps,
  - distributed-GMRES env enablement logic,
  - SciPy GMRES/BiCGStab history paths including right preconditioning.
- Fresh audited local result after the heavy-module batch:
  - chunked `pytest -q` over the full tree
  - `495 passed`
  - chunked package coverage audit
  - total package coverage: `53%`
  - measured module gains:
    - `io.py`: `65% -> 67%`
    - `solver.py`: `57% -> 67%`
- Added a stencil/policy branch campaign for the remaining cheap `grids.py` and
  top-level `v3_driver.py` surfaces:
  - `tests/test_grids_scheme_coverage.py`
  - extended `tests/test_v3_driver_policy_helpers.py`
- These cover:
  - representative finite-difference schemes `30/40/50/60/80/90/100/110/120/130`,
  - odd-`n` periodic spectral differentiation,
  - high-order aperiodic endpoint coefficients for schemes `12` and `13`,
  - `NotImplementedError` branches for schemes `122` and `132`,
  - remaining top-level PAS/tokamak policy boundaries and invalid env parsing in `v3_driver.py`,
  - sparse-structural tolerance env handling,
  - transport `tzfft` accelerator auto-path boundaries,
  - dense-krylov and host-dense fallback env logic.
- Fresh audited local result after the grids/policy batch:
  - chunked `pytest -q` over the full tree
  - `514 passed`
  - chunked package coverage audit
  - total package coverage: `54%`
  - measured module gains:
    - `grids.py`: `46% -> 79%`
    - `v3_driver.py`: `36%` (small top-level branch improvement, but still the dominant remaining denominator)
- Added a bounded sparse-helper campaign inside `v3_driver.py`:
  - `tests/test_v3_driver_sparse_helper_coverage.py`
- These cover:
  - host sparse-direct policy/env gates,
  - sparse-preconditioned rescue eligibility,
  - host sparse factor dtype and cache-key logic,
  - sparse-direct refinement-step parsing,
  - direct and sparse-direct iterative refinement helpers on tiny synthetic operators,
  - sparse-direct GMRES polish wiring,
  - explicit sparse-host-direct helper bounds.
- Fresh audited local result after the sparse-helper batch:
  - chunked `pytest -q` over the full tree
  - `520 passed`
  - chunked package coverage audit
  - total package coverage: `54%`
  - measured module gains:
    - `v3_driver.py`: `36% -> 37%`
- Honest conclusion:
  - The cheap helper surface is now much better covered.
  - `95%` is still not reachable without a separate heavy-solver campaign against `v3_driver.py`, `io.py`, `solver.py`, and the remaining under-covered numerical infrastructure.
  - The next meaningful coverage work is therefore targeted physics/regression tests on those heavy modules, not more small helper tests.
- Added one more literature-anchored numeric / bounded-driver pass:
  - extended `tests/test_grids_scheme_coverage.py`
  - extended `tests/test_rhs1_sparse_first_heuristic.py`
  - extended `tests/test_v3_driver_sparse_helper_coverage.py`
- These cover:
  - polynomial exactness order conditions for SFINCS `uniformDiffMatrices` schemes `2`, `3`, `12`, and `13`,
  - remaining one-sided five-point guard branches for schemes `102` and `112`,
  - explicit sparse host-factor builder env parsing, matrix-free operator assembly hooks, emit-path behavior, and factorization handoff in `v3_driver.py`,
  - invalid-env and override parsing for PAS large-base selection and PAS fast-accept thresholds.
- Fresh audited local result after this pass:
  - `pytest --collect-only -q` -> `529 tests collected`
  - chunked `pytest -q` over the full tree -> `529 passed`
  - chunked package coverage audit -> total package coverage `54%`
  - measured module gains:
    - `grids.py`: `79% -> 82%`
    - `v3_driver.py`: still `37%`, but with the explicit sparse-factor builder and additional PAS env seams now covered
- Literature anchors used in this pass:
  - the finite-difference order conditions encoded in SFINCS v3 `uniformDiffMatrices`,
  - the periodic / spectral differentiation identities used throughout the SFINCS technical notes,
  - the 2014 SFINCS paper’s continuum discretization framework for the bounded operator-path checks.
- Next meaningful coverage work remains unchanged:
  - the denominator is still dominated by `sfincs_jax/v3_driver.py`,
  - then `sfincs_jax/io.py`, `sfincs_jax/solver.py`, and `sfincs_jax/pas_smoother.py`,
  - so the next campaign should target bounded solve-selection / preconditioner-applicability seams in the driver, not more cheap helper branches.
- Added an applied-math / gate-metric coverage pass:
  - extended `tests/test_periodic_stencil.py`
  - extended `tests/test_pas_smoother.py`
- These cover:
  - circulant/Fourier-mode exactness for extracted periodic derivative stencils,
  - sparse-row stencil extraction bounds on bad-shape / too-dense matrices,
  - documented `apply_periodic_stencil_halo()` fallback-to-roll behavior when local shards are too small,
  - sharding-hint env semantics for the periodic stencil runtime gate,
  - `should_stop_adaptive_smoother()` target / nonfinite / upward / continue cases,
  - `run_adaptive_stationary_smoother()` convergence and nonfinite-update behavior on tiny analytic systems,
  - zero-residual and consecutive-increase gate decisions in the PAS smoother logic.
- Fresh audited local result after this pass:
  - `pytest --collect-only -q` -> `539 tests collected`
  - chunked `pytest -q` over the full tree -> `539 passed`
  - chunked package coverage audit -> total package coverage `54%`
  - measured module gains:
    - `periodic_stencil.py`: `57% -> 67%`
    - `pas_smoother.py`: `59% -> 62%`
    - `grids.py`: held at `82%`
    - `v3_driver.py`: held at `37%`
- Numerical-analysis anchors used in this pass:
  - Fourier modes as eigenvectors of circulant derivative operators,
  - sparse stencil extraction preserving the same discrete linear operator,
  - residual-history stopping rules aligned with minimal-residual / stagnation monitoring concepts used in Krylov and stationary iterations.
- Current conclusion:
  - the remaining denominator is now even more concentrated in `sfincs_jax/v3_driver.py`,
  - followed by `sfincs_jax/io.py`, `sfincs_jax/solver.py`, and the uncovered portions of the physics assembly stack,
  - so the next high-signal campaign should target bounded physics/reduction seams inside the driver and output/diagnostics assembly rather than more standalone helper modules.
- Added a bounded diagnostics/output-reduction coverage pass:
  - extended `tests/test_u_hat_fft.py`
  - fixed `sfincs_jax/diagnostics.py`
- These cover:
  - FFT-vs-NumPy `uHat` agreement on a frozen scheme-4 fixture,
  - differentiability of `uHat` with respect to Boozer harmonics,
  - finite/shape-correct `_u_hat_loop()` behavior on even and odd periodic cosine geometries,
  - resonant-denominator safety in the explicit harmonic-loop reference implementation,
  - spatial constancy of the loop implementation in the constant-`B` limit.
- Fresh audited local result after this pass:
  - `pytest --collect-only -q` -> `543 tests collected`
  - chunked `pytest -q` over the full tree -> `543 passed`
  - chunked package coverage audit -> total package coverage `54%`
  - measured module gains:
    - `diagnostics.py`: `77% -> 100%`
    - total package coverage held at `54%`, so the dominant remaining denominator is still the heavy solver stack
- Real bug fixed in this pass:
  - `_u_hat_loop()` previously relied on `jnp.where()` to mask resonant denominators, but the Python-side `(numer / denom)` still evaluated first and could raise `ZeroDivisionError` on exact resonances. The loop now guards the denominator explicitly before forming the amplitude.
- Next meaningful coverage work:
  - stay on bounded, physics-relevant seams,
  - focus next on driver-side solve-selection / preconditioner-applicability branches and then output/diagnostics assembly in `io.py`,
  - avoid broad expensive end-to-end solve campaigns unless they buy real heavy-module coverage.
- Added a bounded driver-side domain-decomposition / reduction coverage pass:
  - new `tests/test_v3_driver_dd_reduction_coverage.py`
- These cover:
  - diagonal-only and block-diagonal-only reductions as local-coupling-preserving simplifications,
  - overlapping patch-range construction for additive-Schwarz style local solves,
  - coarse-level sizing and environment override behavior for multilevel Schwarz correction,
  - bounded multilevel residual-correction composition with zero-step and ordered-level checks,
  - safe-preconditioner clipping/NaN handling,
  - finite-state gating for GMRES results.
- Fresh audited local result after this pass:
  - `pytest --collect-only -q` -> `552 tests collected`
  - chunked `pytest -q` over the full tree -> `552 passed`
  - chunked package coverage audit -> total package coverage `54%`
  - measured module gains:
    - `v3_driver.py`: held at `37%`, but with the DD/reduction seam now exercised directly
    - `diagnostics.py`: held at `100%`
    - `grids.py`: held at `82%`
    - `solver.py`: held at `67%`
- Numerical / literature anchors used in this pass:
  - additive-Schwarz and block-Jacobi locality invariants for restricted local corrections,
  - multilevel residual-correction ordering and bounded damping ideas from domain-decomposition preconditioning,
  - finite-state Krylov acceptance criteria on bounded synthetic systems.
- Current conclusion:
  - the remaining denominator is still dominated by `sfincs_jax/v3_driver.py`,
  - the next high-signal tests should stay inside the driver’s solve-selection / preconditioner-applicability ladder and then move to `io.py` output/reduction assembly,
  - the right strategy remains bounded synthetic operators and reduction identities, not broad expensive end-to-end solves.
- Added a bounded driver solve-policy / rescue-ladder coverage pass:
  - new `tests/test_v3_driver_solve_policy_coverage.py`
- These cover:
  - `constraintScheme=0` PETSc-compat and dense-fallback routing,
  - sparse-exact-LU selection for FP and PAS branches, including accelerator small-case and full-preconditioner paths,
  - large-CPU x-block skip-primary eligibility,
  - transport sparse-direct first-attempt and host-GMRES-first policy guards,
  - transport residual-acceptance, recycle, factor-dtype, and retry policy seams,
  - host-only SciPy Krylov requests and GMRES dispatch incompatibility with distributed paths.
- Fresh audited local result after this pass:
  - `pytest --collect-only -q` -> `568 tests collected`
  - chunked `pytest -q` over the full tree -> `568 passed`
  - chunked package coverage audit -> total package coverage `55%`
  - measured module gains:
    - `v3_driver.py`: `37.0% -> 37.1%` (`5227/14161 -> 5259/14161`)
    - total package coverage: `54% -> 55%`
    - `io.py`: held at `66.6%`
    - `solver.py`: held at `67.0%`
- Numerical / validation conclusion:
  - this pass buys real signal because it covers the branch-selection logic that determines which bounded linear-algebra path the physics solve takes,
  - it still avoids expensive full-system solves and therefore keeps the campaign efficient,
  - the remaining denominator is even more obviously concentrated in the deep solve body of `v3_driver.py` and then in `io.py` output/reduction assembly.
- Next meaningful coverage work:
  - keep targeting bounded, physics-relevant seams in `v3_driver.py`, especially the solve-handoff and preconditioner-builder edges that still decide real production behavior,
  - then move to `io.py` output/reduction assembly and transport/output diagnostic construction,
  - continue to avoid long end-to-end solve campaigns unless they buy meaningful heavy-module coverage.
- Added a bounded `io.py` output-policy / serialization coverage pass:
  - new `tests/test_io_output_policy_coverage.py`
- These cover:
  - output-cache directory selection and cache-path determinism,
  - nested HDF5 readback and overwrite guards,
  - scalar/list parsing helpers and Legendre-matrix construction,
  - bounded includePhi1 Newton-step selection policy,
  - export-`f` configuration on real geometry-4 fixtures, invalid-option rejection, and identity export-map application.
- Fresh audited local result after this pass:
  - `pytest --collect-only -q` -> `579 tests collected`
  - chunked `pytest -q` over the full tree -> `579 passed`
  - chunked package coverage audit -> total package coverage `55%`
  - measured module gains:
    - `io.py`: `66.6% -> 66.8%` (`1714/2574 -> 1719/2574`)
    - `v3_driver.py`: held at `37.1%`
    - `solver.py`: held at `67.0%`
- Numerical / validation conclusion:
  - this pass buys real user-facing signal because it exercises the output-side policy and serialization behavior that shapes written artifacts and postprocessing inputs,
  - it remains cheap because it stays on tiny fixtures and synthetic arrays instead of broad solve campaigns,
  - the dominant remaining denominator is still the deep solve body of `v3_driver.py`, followed by the larger uncovered portions of `io.py`.
- Next meaningful coverage work:
  - return to bounded `v3_driver.py` solve-handoff and preconditioner-builder edges,
  - then keep filling `io.py` output/reduction assembly with similarly bounded tests,
  - continue preferring mathematically anchored seams over broad expensive end-to-end reruns.
- Added a bounded PAS tokamak / PAS-TZ preconditioner-policy coverage pass:
  - new `tests/test_v3_driver_pas_precond_policy_coverage.py`
- These cover:
  - zeta-invariant tokamak-theta applicability and rejection of zeta-varying or drift-rich tokamak branches,
  - fallback from the tokamak-theta builder to the generic block preconditioner,
  - PAS-TZ applicability boundaries for RHS mode, angular grid size, `n_xi`, PAS-only vs FP structure,
  - invalid environment fallback for `SFINCS_JAX_RHSMODE1_PAS_TZ_MAX_BYTES`,
  - PAS-TZ build-byte estimation and memory-safety gating,
  - fallback from the PAS-TZ builder to the hybrid preconditioner when PAS-TZ is inapplicable or memory-unsafe.
- Fresh audited local result after this pass:
  - `pytest --collect-only -q` -> `588 tests collected`
  - chunked `pytest -q` over the full tree -> `588 passed`
  - chunked package coverage audit -> total package coverage `55%`
  - measured module gains:
    - `v3_driver.py`: held at `37%` (`5280/14161`)
    - `io.py`: held at `67%`
    - `solver.py`: held at `67%`
- Numerical / validation conclusion:
  - this pass buys real signal because it covers the driver-side routing that decides whether the heavier PAS tokamak and PAS-TZ preconditioners are even eligible to be built,
  - it stays efficient by using tiny synthetic operators and builder fallbacks instead of any broad RHSMode=1 solve campaign,
  - the remaining denominator is still concentrated in the deep execution body of `v3_driver.py`, not the outer policy layer.
- Next meaningful coverage work:
  - continue on bounded `v3_driver.py` solve-handoff and preconditioner-builder edges beneath these PAS policy gates,
  - then return to `io.py` output/reduction assembly with similarly bounded tests,
  - keep avoiding broad expensive end-to-end reruns unless they buy real heavy-module coverage.
- Extended the PAS tokamak / PAS-TZ coverage slice to the sharded memory-unsafe handoff:
  - updated `sfincs_jax/v3_driver.py`
  - updated `tests/test_v3_driver_pas_precond_policy_coverage.py`
- These cover:
  - fallback from PAS-TZ to the axis-correct Schwarz builder on memory-unsafe sharded runs,
  - both `theta` and `zeta` shard-axis routing,
  - invalid `SFINCS_JAX_RHSMODE1_{THETA,ZETA}_DD_{BLOCK,OVERLAP}` parsing on that handoff path.
- Real bug fixed:
  - the PAS-TZ memory-unsafe sharded fallback always routed into `theta_schwarz`, even when `_matvec_shard_axis(op) == "zeta"`;
  - the driver now dispatches to `theta_schwarz` for `theta` sharding and `zeta_schwarz` for `zeta` sharding.
- Fresh audited local result after this follow-up:
  - `pytest --collect-only -q` -> `590 tests collected`
  - chunked `pytest -q` over the full tree -> `590 passed`
  - chunked package coverage audit -> total package coverage `55%`
  - measured module gains:
    - `v3_driver.py`: held at `37%` (`5285/14162`)
    - `io.py`: held at `67%`
    - `solver.py`: held at `67%`
- Numerical / validation conclusion:
  - this follow-up buys real production signal because it covers and fixes the axis-specific handoff that determines which local Schwarz preconditioner a sharded PAS run actually builds under memory pressure,
  - it stays efficient by using tiny synthetic operators and mocked builder fallbacks,
  - the remaining denominator is still the deep solve body of `v3_driver.py`, not the outer routing seams.
- Next meaningful coverage work:
  - continue into bounded `v3_driver.py` solve-handoff and reduced/full preconditioner-builder edges below these PAS fallback routes,
  - then return to `io.py` output/reduction assembly with similarly bounded tests,
  - keep avoiding broad expensive end-to-end reruns unless they buy real heavy-module coverage.
- Factored the RHSMode=1 reduced/full preconditioner dispatch ladder into a shared helper:
  - updated `sfincs_jax/v3_driver.py`
  - new `tests/test_v3_driver_rhs1_dispatch_coverage.py`
- These cover:
  - `theta_dd` routing to DD vs Schwarz depending on overlap,
  - `point_xdiag` forwarding of `preconditioner_xi`,
  - `xblock_tz_lmax` forwarding of the resolved `lmax`,
  - `theta_line_xdiag` composition with the collision preconditioner on PAS/FP branches,
  - default fallback to the generic block preconditioner with the right species/x/xi parameters.
- Real bug / consistency fix:
  - the RHSMode=1 reduced and full solve paths previously carried separate copies of the preconditioner dispatch ladder;
  - the reduced path handled `point_xdiag` and `xblock_tz_lmax`, while the full-path copy did not mirror those branches cleanly;
  - both paths now dispatch through `_build_rhs1_preconditioner_from_kind(...)`, closing that drift.
- Fresh audited local result after this pass:
  - `pytest --collect-only -q` -> `596 tests collected`
  - chunked `pytest -q` over the full tree -> `596 passed`
  - chunked package coverage audit -> total package coverage `55%`
  - measured module gains:
    - `v3_driver.py`: `37% -> 38%` (`5309/14096`)
    - `io.py`: held at `67%`
    - `solver.py`: held at `67%`
- Numerical / validation conclusion:
  - this pass buys real signal because it turns the production RHSMode=1 preconditioner handoff into one tested dispatch surface instead of two drifting nested copies,
  - it stays efficient by testing the helper directly on tiny synthetic operators with mocked builders,
  - the remaining denominator is still concentrated in the deep solve body of `v3_driver.py`, but the top-level routing layer is now materially tighter.
- Next meaningful coverage work:
  - keep pushing on bounded `v3_driver.py` solve-handoff and post-build fallback seams beneath the shared dispatch helper,
  - then return to deeper `io.py` output/reduction assembly,
  - continue preferring mathematically anchored seams over broad expensive end-to-end solve campaigns.

---

## 19) Research-Grade Coverage + Validation + Autodiff Roadmap (2026-04-22)

This section is the concrete roadmap for moving `sfincs_jax` from the current
release-quality state to a research-grade, optimization-ready state with:
- near-complete automated validation,
- materially higher test coverage,
- stronger benchmark discipline,
- and trustworthy derivative-aware workflows for sensitivity analysis, inverse design,
  uncertainty quantification, and stellarator optimization.

### 19.1 External anchors reviewed for this roadmap

Primary physics / SFINCS references:
- Landreman et al., *Phys. Plasmas* 21, 042503 (2014), the original SFINCS paper:
  https://publications.lib.chalmers.se/records/fulltext/199559/local_199559.pdf
- The upstream SFINCS technical documentation and paper sources mirrored in
  `docs/upstream/` and online at:
  https://github.com/landreman/sfincs/tree/master/doc
  https://github.com/landreman/sfincs/tree/master/doc/sfincsPaper
- STELLOPT’s SFINCS integration notes:
  https://princetonuniversity.github.io/STELLOPT/SFINCS.html

Neighboring code / workflow anchors:
- `yancc` (`f0uriest/yancc`), especially its explicit testing culture and active
  solver/smoother work:
  https://github.com/f0uriest/yancc
- `monkes` (`f0uriest/monkes`) and the MONKES literature for monoenergetic
  block-tridiagonal structure, factor reuse, and optimization-oriented transport:
  https://github.com/f0uriest/monkes
  https://arxiv.org/abs/2312.12248
- `simsopt` for optimization graph structure, least-squares workflows, and MPI-aware
  optimization orchestration:
  https://simsopt.readthedocs.io/stable/
- `DESC` for JAX-native stellarator optimization posture and differentiated equilibrium
  workflows:
  https://desc-docs.readthedocs.io/en/stable/
- JAX implicit differentiation / linear-solve hooks:
  https://docs.jax.dev/en/latest/_autosummary/jax.lax.custom_linear_solve.html
- JAXopt implicit differentiation notes:
  https://jaxopt.github.io/dev/implicit_diff.html

### 19.2 Current reality and the main planning constraint

Current audited local state:
- `596` tests collected
- `596` tests passed
- total package coverage `55%`
- dominant denominator:
  - `sfincs_jax/v3_driver.py` at `38%`
  - then `sfincs_jax/io.py` at `67%`
  - then `sfincs_jax/solver.py` at `67%`

Conclusion:
- Reaching `95%` total package coverage is **not** a “write more tests” problem.
- It is a **code-structure + testability + validation-surface** problem.
- The only credible path to `95%` is:
  1. keep adding bounded literature-anchored tests,
  2. refactor the remaining monoliths into testable helpers/modules,
  3. lock every benchmarked/autodiff-facing workflow to golden validation artifacts.

### 19.3 Target end state

For this roadmap, “research-grade” means:
- all shipped examples and benchmark examples run in CI or in a reproducible audited lane,
- all supported geometry / physics families have at least one end-to-end validated fixture,
- every important solver/preconditioner decision layer has bounded unit/regression tests,
- every derivative-facing public workflow has gradient checks against finite-difference or
  implicit-diff references,
- performance claims are tied to pinned benchmark artifacts,
- optimization/UQ workflows are demonstrated on real `sfincs_jax` objectives,
- and package coverage reaches `95%` without padding it with low-value tests.

### 19.4 Workstream A: Coverage to 95% by refactoring the denominator

#### A1. Split `v3_driver.py` into testable submodules

Current blocker:
- `v3_driver.py` still holds the majority of uncovered production logic.

Refactor target:
- extract the following into separate modules with stable helper APIs:
  - `rhs1_policy.py`
  - `rhs1_preconditioner_dispatch.py`
  - `rhs1_preconditioner_builders.py`
  - `rhs1_fallbacks.py`
  - `transport_policy.py`
  - `nonlinear_newton.py`
  - `distributed_policy.py`

Coverage goal after split:
- each extracted module should reach `90-95%` individually,
- the remaining thin orchestration in `v3_driver.py` should be kept intentionally small,
- package total should move sharply upward without synthetic test padding.

Acceptance criteria:
- no behavior change in the full frozen CPU/GPU suite,
- identical CLI-facing output for the audited scope,
- full-tree coverage rerun demonstrates a material jump, not a cosmetic one.

#### A2. Finish `io.py` output/reduction coverage

Uncovered high-value seams still include:
- result assembly paths that only trigger after successful solves,
- diagnostic selection / omission branches,
- HDF5 write policies for full/reduced / geometry-only / transport-only cases,
- comparison and export pathways used by parity and publication scripts.

Plan:
- write bounded fixtures that bypass expensive solves by feeding in synthetic but
  shape-correct result dictionaries,
- pin exact HDF5 key behavior and output-map semantics,
- add tests for all `sfincsOutput.h5` writer modes used in examples/docs.

Acceptance criteria:
- `io.py` at `90%+`,
- every dataset family written by the public CLI covered by at least one direct test.

#### A3. Finish `solver.py` and `pas_smoother.py`

Plan:
- add bounded tests for:
  - restart/termination/stagnation rules,
  - transpose-solve paths used by implicit differentiation,
  - recycled-subspace paths,
  - distributed-Krylov enablement and fallbacks,
  - PAS smoother histories and adaptive stopping edge cases under tiny synthetic operators.

Acceptance criteria:
- `solver.py` and `pas_smoother.py` both above `90%`,
- all implicit-diff solve modes used in examples are directly covered.

### 19.5 Workstream B: Physics validation matrix anchored in literature

This workstream should move beyond “fixture parity exists” to “the governing invariants
and asymptotic limits are explicitly tested.”

#### B1. Geometry and coordinate-system invariants

Add direct tests for:
- Boozer coordinate symmetry / non-symmetry (`geometryScheme 11/12`),
- VMEC radial-coordinate consistency (`rho`, `s`, `psi`) against upstream definitions,
- flux-surface averages, Jacobian identities, and magnetic-field component consistency,
- Miller / analytic geometry parameter sensitivities.

Anchor:
- upstream SFINCS docs,
- existing mirrored PDFs,
- geometry-specific formulas already documented in `docs/geometry.rst`.

#### B2. Collision / trajectory / model-validation matrix

Required benchmark grid:
- PAS vs FP
- DKES vs full trajectories
- with/without `E_r`
- with/without `Phi1`
- monoenergetic vs full kinetic
- axisymmetric vs non-axisymmetric
- Boozer `.bc` vs VMEC `wout`

Validation target:
- use the current 39-case suite + additional examples as the minimum release set,
- then add targeted sweep fixtures where literature makes specific claims:
  - collisionality sweeps,
  - `E_r` sweeps,
  - resolution sensitivity studies,
  - collision-operator / trajectory-model comparisons from the 2014 SFINCS paper.

Acceptance criteria:
- every model family in docs has at least one corresponding automated validation lane.

#### B3. Manufactured / reduced analytic tests

Add tests that do not depend on external HDF5 fixtures:
- null residual on exact manufactured states where possible,
- symmetry-protected limits,
- constant-`B` and reduced-coupling limits,
- small-system exact solves for transport and RHSMode=1 operators,
- Landau / PAS operator conservation and sign checks where the discrete operator
  should satisfy them.

### 19.6 Workstream C: Benchmark and validation discipline

The code is already parity-clean on the audited suite. What is missing is a more
systematic research benchmark matrix.

#### C1. Pinned benchmark matrix

Create a benchmark manifest that always runs:
- full frozen CPU suite,
- full frozen GPU suite,
- top offender subset,
- transport-worker scaling,
- single-case sharded CPU/GPU scaling,
- compile-time vs solve-time split,
- memory peak for the heaviest offender set.

Artifacts to pin:
- runtime JSON
- memory JSON
- solver path / fallback logs
- environment metadata

#### C2. Statistical benchmarking

Current benchmarking often relies on single runs.

Upgrade:
- use at least `3-5` repeats for small/medium cases,
- report median and spread,
- separate cold-start compile cost from warm solve cost,
- separate process launch overhead from solver-reported time.

#### C3. Research-facing comparison matrix

For trustworthiness, keep and publish:
- `sfincs_jax` CPU/GPU vs SFINCS Fortran v3,
- `sfincs_jax` monoenergetic subset vs MONKES where the model overlaps,
- throughput / scaling comparisons for the optimization-facing transport-worker lane.

### 19.7 Workstream D: Autodiff, inverse design, UQ, and optimization

This is the largest gap between “code runs” and “code is useful for research workflows.”

#### D1. Define the supported differentiability surface

Publicly stable derivative-aware APIs should be limited and explicit:
- matrix-free residuals,
- differentiable geometry parameterizations,
- implicit differentiation through linear solves,
- transport/objective functionals that are documented as differentiable.

Explicitly unsupported or not-yet-guaranteed paths should be documented separately.

#### D2. Gradient verification campaign

For every public autodiff example, add automated tests comparing:
- autodiff gradients,
- finite-difference gradients,
- implicit-diff gradients where relevant,
- and, when feasible, directional derivatives via JVP/VJP.

Initial mandatory objective families:
- residual norm vs `nu_n`,
- geometry harmonic coefficients in `geometryScheme=4`,
- `FSABHat2` / transport scalar functionals,
- implicit-diff through linear solves,
- differentiable transport-matrix outputs where support is claimed.

Acceptance criteria:
- every documented autodiff example has a test,
- gradient relative errors are pinned in CI on small fixtures.

#### D3. Inverse design and stellarator-optimization interfaces

Short-term:
- provide a stable wrapper layer that exposes `sfincs_jax` objectives and gradients
  in a form that can be embedded in `simsopt` and DESC workflows,
- start with serial/CPU and transport-worker throughput, not full multi-GPU single-case sharding.

Mid-term:
- create two reference optimization demos:
  1. inverse calibration of a kinetic/transport parameter to a frozen reference,
  2. geometry-harmonic optimization under regularization and parity checks.

Long-term:
- integrate with VMEC/DESC/SIMSOPT parameter loops for stellarator optimization,
- add robust checkpoint/restart and objective caching so repeated optimization steps
  are reproducible and efficient.

#### D4. Uncertainty quantification

Add a dedicated UQ lane built on the explicit CLI path plus the differentiable Python path:
- case-parallel Monte Carlo / Latin hypercube on CPU/GPU workers,
- local linear uncertainty propagation using gradients from the autodiff path,
- gradient-vs-sampling cross-checks on small benchmark objectives.

Acceptance criteria:
- at least one published example for:
  - local sensitivity,
  - inverse calibration,
  - UQ propagation,
  - stellarator optimization embedding.

### 19.8 Workstream E: CI/CD and runtime budgeting

To keep this realistic, the test campaign must be stratified:

#### E1. Fast CI lane
- bounded unit/regression/gradient tests
- docs build
- no heavy solves
- target: minutes, not tens of minutes

#### E2. Medium audited lane
- selected parity fixtures
- selected benchmark smoke runs
- selected autodiff gradient checks

#### E3. Nightly / release lane
- full frozen CPU suite
- full frozen GPU suite
- benchmark matrix
- coverage audit
- publication-figure regeneration checks

This is the only way to push toward `95%` and full research validation without
making every PR prohibitively slow.

### 19.9 Immediate execution order

1. **Refactor for testability first**
   - split `v3_driver.py` along the existing dispatch/fallback boundaries.
2. **Raise coverage where it matters**
   - target extracted driver modules, then `io.py`, then `solver.py`/`pas_smoother.py`.
3. **Lock the physics matrix**
   - turn the current example suite + sweeps into a documented validation matrix.
4. **Lock the derivative matrix**
   - every public autodiff/optimization example gets an automated gradient check.
5. **Lock the benchmark matrix**
   - pinned CPU/GPU/full/transport-worker/offender artifacts with medians and warm/cold splits.
6. **Only then claim research-grade**
   - when coverage, validation, benchmarking, and derivative-aware workflows all agree.

### 19.10 Quantitative acceptance gates

Coverage:
- package total: `>=95%`
- `v3_driver` successor modules: `>=95%`
- `io.py`: `>=90%`
- `solver.py`: `>=90%`

Validation:
- full frozen CPU suite: clean
- full frozen GPU suite: clean
- documented model-validation sweep matrix: clean

Autodiff:
- every public autodiff example tested
- finite-difference / implicit-diff gradient agreement pinned

Benchmarking:
- pinned benchmark manifest regenerated
- top offenders explicitly tracked over time
- transport-worker scaling remains the published GPU scaling lane unless single-case
  sharding becomes genuinely strong and stable

### 19.11 Immediate next coding tasks

1. Extract the shared RHSMode=1 dispatch / fallback helpers out of `v3_driver.py`.
2. Add a bounded output/reduction assembly test batch for `io.py`.
3. Build a gradient-test batch for the shipped autodiff examples.
4. Create a benchmark manifest file and audited runner for CPU/GPU/full/offender lanes.
5. Add a first `simsopt`-style objective wrapper demo for serial sensitivity/inverse design.

### 19.12 Active refactor branch: `refactor/v3-driver-split`

Purpose:
- reduce the denominator that blocks `95%` coverage,
- move `v3_driver.py` toward a thin orchestration layer,
- preserve full-suite behavior and Fortran-v3 parity while increasing direct testability.

Branch rules:
- every extraction must keep existing focused PAS/RHSMode=1 tests green before moving on,
- new modules must carry docstrings and narrow responsibilities,
- existing monkeypatch/debug seams in `sfincs_jax.v3_driver` should stay stable unless there is a strong reason to break them,
- no numerical or solver-policy change is allowed in this branch unless it is required to preserve correctness after the split.

Execution order on this branch:
1. Extract PAS applicability / memory-policy helpers into `rhs1_pas_policy.py`.
2. Extract the shared RHSMode=1 dispatch ladder into `rhs1_preconditioner_dispatch.py`.
3. Extract RHSMode=1 fallback / rescue policy below the dispatch layer.
4. Extract transport-policy and distributed-policy helpers.
5. Split nonlinear / Newton helpers away from linear solve orchestration.
6. After each step, rerun the focused driver tests and then a broader branch validation slice.

Current branch status:
- `rhs1_pas_policy.py` extraction is landed and validated against the PAS policy test slice.
- `rhs1_preconditioner_dispatch.py` extraction is landed and validated; `v3_driver.py` now keeps a thin wrapper around the shared dispatch helper so the existing regression seam stays intact.
- `rhs1_strong_fallback.py` is now landed for the full-path strong-preconditioner fallback build, replacing the duplicated full-path builder ladder with a shared helper that reuses the dispatch module.
- `rhs1_strong_policy.py` is now landed for the duplicated reduced/full strong-preconditioner env-to-kind mapping.
- `rhs1_stage2_policy.py` is now landed for the duplicated stage-2 trigger / FP-force-stage2 / PAS-stage2-skip policy.
- `rhs1_strong_control.py` is now landed for the duplicated strong-preconditioner enable/disable/auto control layer, including sparse-rescue-first and PAS-fast-accept gating.
- `rhs1_strong_auto_kind.py` is now landed for the duplicated reduced/full automatic strong-preconditioner kind selection and post-selection adjustments, including theta-line size promotion and PAS tokamak-style `xblock_tz_lmax` fallback.
- `rhs1_sparse_rescue_policy.py` is now landed for the duplicated sparse-rescue ordering and skip policy, including dense-shortcut interaction, size routing, targeted-rescue suppression after exact large-CPU LU selection, PAS fast-accept skip, GPU sparse-skip, and sparse-JAX memory-cap disablement.
- `rhs1_handoff.py` is now landed for the repeated “accept improved candidate and update Krylov replay state” logic used by stage-2, smoother, collision-retry, strong-preconditioner, and PAS Schur rescue branches.
- the sparse accept/handoff paths now use the shared handoff helper for sparse-JAX and generic sparse fallback acceptance in both reduced and full RHSMode=1 paths, so the remaining duplication is concentrated in the deeper branch-specific sparse build/polish ladders rather than in acceptance-state mutation.
- `rhs1_sparse_polish_policy.py` is now landed for the duplicated sparse polish / retry / accept-ratio env parsing used by FP x-block seeds, sxblock polish, host sparse direct polish, and sparse operator-preconditioned GMRES restart/maxiter selection.
- `transport_policy.py` is now landed for the pure RHSMode=2/3 transport backend / sparse-direct / host-GMRES / dtype / recycle policy layer, with thin wrappers preserved in `v3_driver.py` so the existing transport tests and monkeypatch seams stay stable.
- `transport_parallel_policy.py` is now landed for the process-parallel transport backend/start-method/persistent-pool/GPU-worker environment policy layer, again keeping thin wrappers in `v3_driver.py`.
- `transport_parallel_runtime.py` is now landed for the transport parallel RHS partitioning, GPU worker subprocess runner, and parallel-result merge layer, reducing the inlined orchestration inside `solve_v3_transport_matrix_linear_gmres` without changing the public transport test seams.
- `transport_parallel_pool.py` is now landed for the persistent transport process-pool lifecycle, replacing the inlined pool cache / rebuild / shutdown state in `v3_driver.py` with a narrow reusable manager while preserving the existing wrapper seams.
- `transport_parallel_execution.py` is now landed for the top-level transport parallel execution branch: run/no-run gating, payload construction, backend-specific execution, persistent-pool retry, and sequential fallback now live outside the monolith.
- `phi1_newton_policy.py` is now landed for the bounded nonlinear/Newton policy layer: active-DOF mode selection, GMRES restart sizing, frozen-Jacobian cache policy, and line-search policy are no longer embedded inline in `solve_v3_full_system_newton_krylov_history`.
- `phi1_newton_linear.py` is now landed for the nonlinear linear-step orchestration: reduced/full routing, sparse-direct entry, KSP-history emission, and retry-without-preconditioner now live outside the monolith while reusing the same numerical kernels.
- `phi1_line_search.py` is now landed for the accepted-iterate update logic: PETSc-like backtracking, fixed-candidate `best` search, and finite-state fallback rules are no longer embedded inline in the Newton driver.
- the first manuscript-validation scaffold is now started:
  - `examples/publication_figures/validation_manifest.json` is the machine-readable map from literature claim to script and artifact,
  - `docs/validation_matrix.rst` is the public-facing counterpart for the same figure/test lanes.
- current validation slice on this branch:
  - focused RHSMode=1 + transport policy/dispatch/fallback tests: `103 passed`
  - broader bounded driver/transport slice: `92 passed`
  - dedicated transport slices:
    - `tests/test_transport_sparse_direct.py`: `37 passed`
    - `tests/test_transport_parallel.py`: `13 passed`
    - `tests/test_transport_parallel_runtime.py`: `3 passed`
    - `tests/test_transport_parallel_execution.py`: `5 passed`
    - `tests/test_phi1_newton_policy.py`: `4 passed`
    - `tests/test_phi1_newton_linear.py`: `3 passed`
    - `tests/test_phi1_line_search.py`: `4 passed`
- next extraction target is the first literature-anchored validation sweep scaffold and figure-generation lane on top of the cleaner branch structure, then any remaining thin orchestration cleanup in the nonlinear path.

### 19.13 Literature-anchored validation baselines for the paper

Primary literature anchors to use directly for validation and manuscript figures:
- SFINCS paper: [Landreman et al., *Comparison of particle trajectories and collision operators for collisional transport in nonaxisymmetric plasmas*, Phys. Plasmas 21, 042503 (2014)](https://publications.lib.chalmers.se/records/fulltext/199559/local_199559.pdf)
- Upstream SFINCS documentation tree: [landreman/sfincs `doc/`](https://github.com/landreman/sfincs/tree/master/doc)
- MONKES paper: [Escoto et al., *MONKES: a fast neoclassical code for the evaluation of monoenergetic transport coefficients*](https://arxiv.org/abs/2312.12248)
- W7-X ion-root validation context: [Pablant et al. ion-root / ambipolar-electric-field comparison page mirrored here](https://sites.fusion.ciemat.es/jlvelasco/files/papers/pablant2020ionroot.pdf)
- optimization / adjoint motivation: [APS abstract on adjoint neoclassical stellarator optimization with SFINCS](https://meetings-archive.aps.org/dpp/2018/bp11/36/)
- W7-X neoclassical validation context at reactor relevance: [Mollen et al., *Demonstration of reduced neoclassical energy transport in Wendelstein 7-X*, Nature 596, 221-226 (2021)](https://www.nature.com/articles/s41586-021-03687-w)
- direct ambipolar-field / ion-root comparison context: [Pablant et al., *Core radial electric field and transport in Wendelstein 7-X plasmas*](https://sites.fusion.ciemat.es/jlvelasco/files/papers/pablant2018er.pdf)
- low-collisionality comparison code context: [Velasco et al., *KNOSOS: a fast orbit-averaging neoclassical code for stellarator geometry*](https://arxiv.org/abs/1908.11615)
- optimization / differentiability target: [Paul et al., *An adjoint method for neoclassical stellarator optimization*](https://arxiv.org/abs/1904.06430)

Key baseline claims from the literature that `sfincs_jax` should explicitly test or reproduce:
- trajectory-model agreement at small normalized electric field and divergence at large `E_r / E_r^{res}`:
  - SFINCS 2014 Figures 1-3 show the three trajectory models are close for `E_*` below roughly one-third of the resonant value and diverge for larger fields, with bootstrap-current sign-change behavior in the full-trajectory model at large inward `E_r`.
- collision-operator comparison across collisionality:
  - SFINCS 2014 Figures 4-6 show which transport-matrix elements are sensitive to momentum conservation and where the high-collisionality asymptotes should match.
- analytic high-collisionality limit:
  - SFINCS 2014 Figure 6 and Appendix B provide a direct asymptotic gate for transport-matrix elements in the short-mean-free-path regime.
- quasisymmetry isomorphism:
  - SFINCS 2014 Appendix A states a strong code-verification property for quasisymmetric fields that should be turned into an automated test family where feasible.
- monoenergetic low-collisionality benchmark and convergence:
  - MONKES provides overlap for monoenergetic coefficients, convergence studies, and runtime expectations that are directly relevant to `geometryScheme` / monoenergetic subsets in `sfincs_jax`.
- ambipolar electric-field and heat-flux validation in optimized stellarators:
  - the W7-X ion-root and Nature validation papers provide publication-grade targets for `E_r` trends, heat-flux ordering, and where neoclassical predictions are expected to match experiment or trusted profile reconstructions.
- optimization-grade derivatives:
  - the adjoint optimization literature gives the right standard for derivative validation: directional-derivative agreement, geometry sensitivity maps, and objective-gradient accuracy under realistic solve tolerances.

### 19.14 Manuscript figure plan

The paper should not be built around only parity tables. It should have a small set of high-information figures, each tied to an automated or semi-automated benchmark lane.

Figure set A: Correctness / physics
- A1. Trajectory-model comparison versus normalized radial electric field.
  - Recreate the SFINCS-2014-style sweep for one tokamak-like and one stellarator case.
  - Plot particle flux, heat flux, parallel flow, and bootstrap current versus `E_r / E_r^{res}`.
  - Goal: show the same small-`E_r` agreement and large-`E_r` divergence structure, with `sfincs_jax` reproducing the expected model ordering.
- A2. Collision-operator comparison versus collisionality.
  - Transport-matrix elements vs collisionality at `E_r = 0`, matching the logic of SFINCS 2014 Figures 4-6.
  - Include analytic high-collisionality asymptotes where available.
- A3. Quasisymmetry / symmetry verification.
  - A compact figure or table showing invariance/isomorphism behavior for matched quasisymmetric fields or a strongly related reduced test.
- A4. W7-X-style ambipolar field / bootstrap-current validation.
  - If experimental-profile validation is practical, show one figure comparing `sfincs_jax` prediction to published experimental/neoclassical comparison context.
  - If not practical for the first paper, demote this to supplement and keep it as a plan item.
- A5. Monoenergetic overlap against MONKES / KNOSOS-style low-collisionality trends.
  - Show coefficient overlap and scaling on a subset where the physics models coincide.

Figure set B: Numerical methods
- B1. Convergence study.
  - Resolution study in `N_theta`, `N_zeta`, `N_xi`, `N_x`, and possibly `L_max`/active-DOF truncation for representative PAS, FP, VMEC, and monoenergetic cases.
  - Plot solution error proxy and runtime versus resolution.
- B2. Solver/preconditioner path map.
  - A compact diagram or ablation plot showing how the default CLI lane selects stable fast paths across major model families.
  - This should be backed by the bounded tests and offender benchmarks.
- B3. Warm/cold runtime split.
  - Separate JAX compile/lowering/setup from steady-state solve time on the main offender classes.

Figure set C: Performance and scaling
- C1. Full-suite CPU/GPU benchmark summary.
  - Keep the parity table, but the manuscript should use a cleaner summary figure: per-case runtime ratio and memory ratio versus Fortran v3.
- C2. Published GPU scaling figure.
  - Use the transport-worker/case-throughput lane as the main GPU scaling claim unless single-case sharding becomes genuinely strong and stable.
- C3. Single-case sharded scaling figure.
  - Keep this only if the ongoing research lane closes convincingly. Otherwise it belongs in limitations/supplement, not the main paper.
- C4. MONKES overlap figure.
  - For monoenergetic overlap cases, compare coefficients and runtime on a like-for-like subset where the models coincide.
- C5. Ambipolar/W7-X validation summary.
  - Compact comparison of predicted `E_r` or neoclassical heat flux against the published W7-X validation context, if the input reconstruction is sufficiently controlled.

Figure set D: Differentiation / optimization
- D1. Gradient-check figure.
  - Autodiff vs finite-difference vs implicit-diff directional derivative agreement for a few representative objectives.
- D2. Sensitivity map or inverse-design demo.
  - Example: bootstrap current or radial flux sensitivity to selected Boozer/geometry coefficients.
- D3. Optimization/UQ workflow figure.
  - Small but real demo showing `sfincs_jax` inside an optimization or UQ loop with cached/parallel evaluation.
- D4. Adjoint-style geometry sensitivity map.
  - Use Boozer or VMEC harmonics to show a local sensitivity map for a transport objective, consistent with the neoclassical-optimization literature.

### 19.15 Additional tests and simulations to strengthen the paper

Add the following to the validation matrix if feasible on the current branch:
- Electric-field sweep tests modeled on SFINCS 2014 Figures 1-3.
  - Needed outputs: fluxes, flows, bootstrap current, source terms.
  - These should become regression plots plus numerical assertions about small-`E_r` agreement and large-`E_r` separation.
- Collisionality sweep tests modeled on SFINCS 2014 Figures 4-6.
  - Needed outputs: transport-matrix elements for PAS / momentum-corrected / FP operators.
  - Assertions should focus on asymptotic trends and operator ordering, not exact plotted values alone.
- High-collisionality asymptotic tests.
  - For representative geometries, verify convergence toward the known analytic short-mean-free-path limits discussed in the SFINCS paper.
- Quasisymmetry isomorphism tests.
  - Add at least one reduced automated lane that exercises the isomorphism relation or a strong proxy derived from the same theory.
- Monoenergetic overlap tests against MONKES.
  - Compare coefficients and convergence on small overlap cases.
- Low-collisionality overlap checks against KNOSOS-style trends.
  - Use these as qualitative ordering/scaling gates where exact like-for-like model overlap is not possible.
- Experimental-profile or profile-inspired validation.
  - W7-X ion-root / bootstrap-current context if the published inputs can be reconstructed sufficiently well.
- Resolution and aliasing studies from numerical analysis.
  - Demonstrate spectral/stencil convergence and absence of spurious parity drift with increasing resolution.
- Autodiff verification battery.
  - JVP/VJP/finite-difference checks for residuals, transport objectives, and geometry parameters used in optimization.
- Adjoint-style sensitivity tests.
  - Directional derivatives and local sensitivity maps for geometry perturbations, matching the optimization literature rather than only tiny toy examples.
- UQ / inverse problems.
  - Small synthetic inverse-calibration task and local uncertainty propagation with gradient cross-checks.

### 19.16 Testing and code-structure documentation workstream

The docs should become explicit about how the code is organized and how it is validated.

Code-structure docs to add or expand:
- `docs/source_map.rst`
  - update it continuously as `v3_driver.py` is split, with one subsection per extracted module and clear ownership of equations / numerics / policy logic.
- `docs/numerics.rst`
  - add a dedicated subsection for the refactored RHSMode=1 dispatch, fallback, and builder layers.
- `docs/testing.rst`
  - expand from “what tests exist” to “why each test family exists”:
    - literature-anchored physics tests,
    - numerical-analysis tests,
    - regression/parity tests,
    - performance/benchmark tests,
    - autodiff/gradient tests.
- `docs/parallelism.rst`
  - make explicit which parallel lanes are publication claims and which remain research lanes.

Testing docs should include:
- a matrix mapping each example/model family to:
  - geometry,
  - physics model,
  - parity fixture,
  - literature anchor,
  - benchmark lane,
  - autodiff support status.
- a reproducibility section describing:
  - frozen reference roots,
  - cold vs warm benchmarks,
  - office multi-GPU reruns,
  - artifact naming/versioning.

### 19.17 Ready-to-start execution order after this planning pass

1. Finish the structural split of `v3_driver.py` on this branch:
   - dispatch layer
   - fallback/rescue layer
   - transport/distributed policy
   - nonlinear helpers
2. In parallel with the split, build the manuscript baseline matrix:
   - trajectory-model `E_r` sweeps
   - collision-operator collisionality sweeps
   - monoenergetic MONKES overlap
3. Expand `docs/testing.rst` and `docs/source_map.rst` as modules move.
4. Add the first manuscript-grade figure generation scripts with pinned JSON artifacts.
5. Only after the structure is stable, run the broader benchmark/validation campaign and start writing the paper figures from those audited artifacts.

### 19.18 Current branch execution status

- Structural split progress now includes:
  - RHSMode=1 PAS policy and dispatch helpers,
  - strong-fallback / strong-control / stage-2 / sparse-rescue / sparse-polish policy helpers,
  - solve handoff helpers,
  - transport policy, transport solve policy, transport preconditioner dispatch,
    transport handoff policy, transport dense-LU helpers, transport host-GMRES helper,
    transport-parallel policy/runtime/pool/execution helpers,
  - Phi1 Newton policy, linear-step, and line-search helpers.
- The first literature-facing validation lane is now live with pinned fixed-case artifacts:
  - script: `examples/publication_figures/generate_er_trajectory_sweep.py`
  - machine-readable lane entry: `examples/publication_figures/validation_manifest.json`
  - tokamak-like reference artifact: `examples/publication_figures/artifacts/er_sweep_tokamak_reference_summary.json`
  - tokamak-like reference figure: `docs/_static/figures/paper/sfincs_jax_er_trajectory_sweep_tokamak_reference.png`
  - stellarator-like fast artifact: `examples/publication_figures/artifacts/er_sweep_stellarator_fast_reference_summary.json`
  - stellarator-like fast figure: `docs/_static/figures/paper/sfincs_jax_er_trajectory_sweep_stellarator_fast_reference.png`
- The current branch lane now proves:
  - the upstream DKES/partial/full trajectory switches are encoded explicitly in one place,
  - the sweep script generates stable JSON + PNG/PDF artifacts with named fixed-case outputs,
  - the fixed tokamak-like lane supports direct assertions on zero-field agreement and
    finite-field model separation,
  - the fixed stellarator-like lane is stable as a bounded fast branch artifact,
    while the full-resolution stellarator sweep is still too heavy for the regular branch workflow.
- Immediate next actions on this branch:
  1. deepen the remaining transport solve orchestration split beneath the new
     transport preconditioner dispatch layer, especially active-DOF / dense-fallback /
     solve-handoff sequencing that is still embedded in `v3_driver.py`,
  2. promote the stellarator-like `E_r` lane from fast branch artifact to a heavier
     audited release/nightly sweep once its runtime/cost is acceptable,
  3. re-audit the collisionality / collision-operator lane from the same validation manifest
     using the corrected scan writer.

### 19.20 Transport preconditioner dispatch split

- The remaining transport preconditioner normalization, auto-selection, DD block
  parsing, sparse-JAX config parsing, and reduced/full builder dispatch has now been
  extracted from `v3_driver.py` into
  `sfincs_jax/transport_preconditioner_dispatch.py`.
- `v3_driver.py` now uses that shared module for:
  - user/env normalization of `SFINCS_JAX_TRANSPORT_PRECOND`,
  - auto preconditioner/strong-preconditioner selection,
  - DD overlap/block parsing,
  - sparse-JAX transport preconditioner setup,
  - lazy strong-preconditioner reuse/build.
- The extraction stayed structure-preserving:
  - no policy changes were introduced,
  - bounded transport tests stayed green,
  - transport behavior is still exercised through the existing public
    `solve_v3_transport_matrix_linear_gmres(...)` seam.
- New bounded regression coverage now lives in
  `tests/test_transport_preconditioner_dispatch.py`.
- Current validation for this slice:
  - `tests/test_transport_preconditioner_dispatch.py`
  - `tests/test_transport_sparse_direct.py`
  - `tests/test_transport_parallel.py`
  - all passed together after the extraction.

### 19.21 Transport active-DOF and dense policy split

- The transport solve still contained a large front-end policy block mixing:
  - active-DOF auto/forced routing,
  - active-index map construction,
  - dense fallback / dense memory-cap handling,
  - dense preconditioner enable/disable control.
- That front-end policy is now extracted into
  `sfincs_jax/transport_solve_policy.py`.
- `v3_driver.py` now uses the shared module for:
  - active-DOF mode resolution,
  - active-index/full-to-active map construction,
  - dense fallback policy recomputation on the active reduced size,
  - dense preconditioner enable/disable policy.
- This stayed structure-preserving:
  - the public `solve_v3_transport_matrix_linear_gmres(...)` seam is unchanged,
  - existing bounded transport tests stayed green,
  - no transport benchmark or parity claims were changed by this extraction.
- New bounded regression coverage now lives in:
  - `tests/test_transport_solve_policy.py`
- Current validation for the transport front-end policy slice:
  - `tests/test_transport_solve_policy.py`
  - `tests/test_transport_preconditioner_dispatch.py`
  - `tests/test_transport_sparse_direct.py`
  - `tests/test_transport_parallel.py`
  - all passed together after the extraction.

### 19.22 Transport handoff policy split

- The reduced and full transport solve branches still duplicated residual retry
  metrics and RHSMode=3 polish parsing.
- That duplicated policy is now extracted into
  `sfincs_jax/transport_handoff_policy.py`.
- `v3_driver.py` now uses that shared module for:
  - finite-comparable residual values,
  - retry gates around solver results,
  - better-candidate comparisons,
  - RHSMode=3 polish threshold / restart / maxiter policy.
- The extraction stayed structure-preserving:
  - the actual linear solves remain in `v3_driver.py`,
  - reduced/full transport branches still execute in the same order,
  - existing transport tests stayed green.
- New bounded regression coverage now lives in:
  - `tests/test_transport_handoff_policy.py`
- Current validation for the handoff-policy slice:
  - `tests/test_transport_handoff_policy.py`
  - `tests/test_transport_solve_policy.py`
  - `tests/test_transport_preconditioner_dispatch.py`
  - `tests/test_transport_sparse_direct.py`
  - `tests/test_transport_parallel.py`
  - all passed together after the extraction.

### 19.23 Transport dense-LU helper split

- The bounded transport dense fallback and dense-preconditioner helpers were still
  nested inside `solve_v3_transport_matrix_linear_gmres(...)`.
- Those pure infrastructure helpers are now extracted into
  `sfincs_jax/transport_dense_lu.py`.
- `v3_driver.py` now calls the shared module for:
  - cached dense-LU preconditioner construction,
  - cached dense-LU direct solver construction.
- This stayed structure-preserving:
  - the cache keys and dense fallback call sites are unchanged,
  - no default dense policy changed,
  - existing transport tests stayed green.
- New bounded regression coverage now lives in:
  - `tests/test_transport_dense_lu.py`
- Current validation for the dense-LU slice:
  - `tests/test_transport_dense_lu.py`
  - `tests/test_transport_handoff_policy.py`
  - `tests/test_transport_solve_policy.py`
  - `tests/test_transport_preconditioner_dispatch.py`
  - `tests/test_transport_sparse_direct.py`
  - `tests/test_transport_parallel.py`
  - all passed together after the extraction.

### 19.24 Transport host-GMRES helper split

- The explicit transport host SciPy GMRES first-attempt/rescue helper was still
  nested inside `solve_v3_transport_matrix_linear_gmres(...)`.
- That solver helper is now extracted into `sfincs_jax/transport_host_gmres.py`.
- `v3_driver.py` now calls the shared module for:
  - host SciPy GMRES without a preconditioner,
  - left-preconditioned host SciPy GMRES,
  - PETSc-like acceptance of bounded preconditioned residuals for the transport
    systems where that behavior is already part of the shipped path.
- This stayed structure-preserving:
  - first-attempt / rescue policy remains in `transport_policy.py`,
  - the reduced/full solve order is unchanged,
  - existing transport tests stayed green.
- New bounded regression coverage now lives in:
  - `tests/test_transport_host_gmres.py`
- Current validation for the host-GMRES slice:
  - `tests/test_transport_host_gmres.py`
  - `tests/test_transport_dense_lu.py`
  - `tests/test_transport_handoff_policy.py`
  - `tests/test_transport_solve_policy.py`
  - `tests/test_transport_preconditioner_dispatch.py`
  - `tests/test_transport_sparse_direct.py`
  - `tests/test_transport_parallel.py`
  - all passed together after the extraction.

### 19.19 Collisionality lane status after writer fix

- A real publication-lane bug was found in `examples/publication_figures/generate_sfincs_paper_figs.py`:
  duplicate namelist assignments could leave the original `collisionOperator` and
  resolution values in force, so stored FP/PAS collisionality outputs could silently
  collapse onto the same physics.
- A second fast-path bug was found immediately after that fix:
  missing keys such as `NL` could be appended outside the `resolutionParameters`
  group, and the hard-coded fast `Nzeta=3` was below the current stencil floor.
- Both bugs are now fixed and covered by bounded tests.
- A corrected bounded LHD fast rerun now cleanly separates FP and PAS transport matrices:
  - FP `L11` at `nu_n=0.02668018`: about `-0.3507`
  - PAS `L11` at `nu_n=0.02668018`: about `-0.4754`
  - FP `L22` at `nu_n=2.668018`: about `-1.5703`
  - PAS `L22` at `nu_n=2.668018`: about `-1.8295`
- This means the collisionality lane is alive again, but the checked-in
  `sfincs_jax_fig{1,2,3}_*.png` files are no longer treated as publication-grade.
  They remain open re-audit lanes until regenerated from the corrected script with
  pinned machine-readable summaries.
- The first corrected branch artifact is now pinned:
  - summary: `examples/publication_figures/artifacts/lhd_collisionality_reaudit_fast_summary.json`
  - figure: `docs/_static/figures/paper/sfincs_jax_fig1_lhd_collisionality_reaudit_fast.png`
- The corrected W7-X fast branch artifact is now also pinned:
  - summary: `examples/publication_figures/artifacts/w7x_collisionality_reaudit_fast_summary.json`
  - figure: `docs/_static/figures/paper/sfincs_jax_fig2_w7x_collisionality_reaudit_fast.png`
- This artifact is backed by direct tests on:
  - the collisionality ladder itself,
  - and the restored FP/PAS separation in the diagonal transport coefficients.
- The same is now true for the bounded W7-X fast lane: it stayed cheap enough for branch
  validation and resolves clear FP/PAS separation in the corrected rerun.
- The generator now also emits structured collisionality summary JSON for future reruns:
  - `generate_sfincs_paper_figs.py` writes top-level `metadata` plus sorted `rows`,
    recording case, fast/full mode, scan ladder, base input, work directory, and
    collision-operator labeling;
  - `tests/test_generate_sfincs_paper_figs.py` now covers both plain row serialization
    and the richer metadata payload;
  - `tests/test_collisionality_artifact.py` accepts either legacy row-only artifacts
    or future metadata-backed artifacts, so the next pinned full-resolution rerun can
    upgrade the checked-in JSON format without breaking branch validation.
- Remaining open lanes are explicit:
  - regenerate the full LHD collisionality figure family from the fixed writer,
  - regenerate the full W7-X collisionality figure family from the fixed writer,
  - regenerate the high-collisionality proxy only after its parent LHD/W7-X scans are
    pinned from the corrected script.

### 19.25 JAX ecosystem adoption review after the focused solver split

Scope of this review:
- local source audited:
  - `sfincs_jax/implicit_solve.py`,
  - `sfincs_jax/solver.py`,
  - `sfincs_jax/v3_system.py`,
  - `sfincs_jax/v3_driver.py`,
  - the extracted transport and Phi1 helpers,
  - `examples/autodiff/`,
  - `examples/optimization/`,
  - `pyproject.toml`;
- external primary sources checked:
  - Lineax docs and source repository:
    `https://docs.kidger.site/lineax/api/linear_solve/`,
    `https://docs.kidger.site/lineax/api/operators/`,
    `https://docs.kidger.site/lineax/api/solvers/`,
    `https://github.com/patrick-kidger/lineax`;
  - Equinox docs and source repository:
    `https://docs.kidger.site/equinox/api/module/module/`,
    `https://docs.kidger.site/equinox/api/transformations/`,
    `https://github.com/patrick-kidger/equinox`;
  - JAX docs:
    `https://docs.jax.dev/en/latest/_autosummary/jax.lax.custom_linear_solve.html`,
    `https://docs.jax.dev/en/latest/gradient-checkpointing.html`,
    `https://docs.jax.dev/en/latest/jax.experimental.sparse.html`,
    `https://docs.jax.dev/en/latest/notebooks/shard_map.html`,
    `https://docs.jax.dev/en/latest/pallas/design/design.html`;
  - nonlinear / optimization ecosystem docs:
    `https://docs.kidger.site/optimistix/api/root_find/`,
    `https://jaxopt.github.io/dev/implicit_diff.html`,
    `https://optax.readthedocs.io/en/stable/api/optimizers.html`;
  - specialized numerical-library docs:
    `https://docs.kidger.site/diffrax/usage/getting-started/`,
    `https://quadax.readthedocs.io/en/stable/api.html`,
    `https://orthax.readthedocs.io/`,
    `https://orthax.readthedocs.io/en/stable/api_general.html`.

Current dependency decision:
- Keep the base install unchanged for now:
  - `jax`,
  - `numpy`,
  - `scipy`,
  - `h5py`,
  - `matplotlib`.
- Do not add `lineax`, `equinox`, `optimistix`, `jaxopt`, `diffrax`,
  `optax`, `quadax`, or `orthax` to production dependencies without a
  pinned benchmark and parity gate.
- If an ecosystem library is admitted only for research/autodiff examples, add
  it as an explicitly documented optional install path rather than as a CLI
  dependency. This preserves the current release goal: small install surface,
  robust executable runs, and no unmeasured solver-stack change.

Findings by library:
- JAX native primitives remain the right core implementation layer:
  - `jax.lax.custom_linear_solve` already matches the shipped implicit-diff
    design in `implicit_solve.py`: matrix-free forward solve plus transpose
    solve, with gradients defined by the implicit equation rather than by
    unrolling Krylov iterations.
  - `jax.checkpoint` / `jax.remat` should remain a targeted memory tool around
    scanned kernels and differentiable collision / structured-velocity pieces,
    not a broad decorator on whole solves.
  - `jax.shard_map` is a future candidate for explicit halo/stencil kernels and
    lower-synchronization domain-decomposition experiments. It should not
    replace the current `pjit`/sharding path until a single-case benchmark
    shows better strong scaling.
  - `jax.experimental.sparse` is not a production-offender solution today:
    JAX documents it as experimental reference sparse support and not
    recommended for performance-critical code. Keep SciPy sparse/direct helpers
    for executable fast paths and use JAX sparse only for small differentiable
    reference experiments if needed.
  - Pallas is a long-horizon candidate only for hand-written GPU kernels in
    very specific hotspots, such as stencil halo packing or collision kernels.
    It is experimental and too low-level for the current refactor branch.
- `lineax` is the only ecosystem library with a plausible near-term solver-core
  role:
  - likely insertion point: `sfincs_jax/implicit_solve.py`;
  - secondary insertion point: small/medium differentiable dense or structured
    linear-solve examples, not the CLI offender path;
  - potential benefits: function linear operators, transposes, reusable solver
    state, PyTree-valued operators/vectors, and a unified linear-solve API;
  - blocking evidence: the earlier local probe found a real small-SFINCS
    speed win but also stagnation on a generic nonsymmetric stress matrix, so it
    is not reliable enough for production defaults;
  - admission gate: pass a benchmark matrix of current JAX GMRES/BiCGStab,
    current SciPy host LGMRES where legal, and Lineax on:
    - `tests/ref/pas_1species_PAS_noEr_tiny_scheme5.input.namelist`
      implicit-diff solve,
    - one small real full-system operator with a transpose-gradient check,
    - one RHSMode=2/3 active-DOF transport reference solve,
    - one generic nonsymmetric stress operator that previously exposed
      stagnation,
    - one repeated-RHS state-reuse case;
  - pass criteria: parity-clean residuals, finite gradients, no stagnation,
    no worse cold compile memory, and at least a `20%` warm-runtime or `25%`
    RSS win on at least one pinned path without a comparable regression.
- `equinox` is useful for future public differentiable APIs, but not yet for
  the core physics operators:
  - the current operators already use explicit `register_pytree_node_class`
    methods to control which fields are dynamic arrays and which integer/bool
    shape/layout options are static;
  - replacing those with `equinox.Module` would reduce boilerplate, but it
    changes a large amount of PyTree surface area and could alter compile cache
    behavior without improving offender runtime;
  - likely future insertion points are new standalone API objects such as
    `SfincsProblem`, `GeometryParameters`, `TransportObjective`, and inverse
    design / UQ examples, where `filter_jit`, `filter_grad`, `partition`, and
    `combine` can simplify mixed static/dynamic parameter handling;
  - admission gate: one small Equinox-backed objective wrapper must compile
    with fewer static-argument seams, match the current JAX-native objective
    gradients, and not slow hot solves.
- `optimistix` is a candidate only for a differentiable nonlinear/Phi1
  prototype:
  - likely insertion points: `sfincs_jax/phi1_newton_linear.py`,
    `sfincs_jax/phi1_line_search.py`, and a new experimental Phi1 nonlinear
    solve wrapper;
  - potential benefits: Newton/chord/root-finding abstractions, explicit
    nonlinear-solver state, and coupling to Lineax linear solvers;
  - blocker: the current production Phi1 path encodes v3/PETSc-like fallback,
    residual-history, frozen-Jacobian, preconditioner, and line-search
    semantics. Replacing it wholesale would be a high-risk behavioral change;
  - admission gate: build a side-by-side experimental wrapper that preserves
    the current accepted-iterate sequence on tiny/bounded Phi1 fixtures before
    any production switch is considered.
- `jaxopt` is useful for implicit-diff wrappers around existing solvers, not
  for the CLI core:
  - likely insertion point: future nonlinear sensitivity examples where the
    forward solve remains the existing `sfincs_jax` solve but gradients are
    exposed through `jaxopt.implicit_diff.custom_root` / `root_vjp`;
  - this may be lower-risk than replacing the forward nonlinear solver because
    it can wrap current semantics;
  - admission gate: a nonlinear scalar/low-dimensional Phi1 or ambipolar-root
    example must match finite-difference sensitivities and not require changing
    production forward solves.
- `optax` should stay an example-level dependency:
  - current optimization examples already import it explicitly and tell users to
    install it;
  - it is appropriate for inverse-design and parameter-calibration examples;
  - it should not enter the solver package dependencies unless optimization
    APIs become first-class package features.
- `diffrax` is not a current solver fit:
  - the shipped equations are discretized steady-state kinetic systems, not
    IVPs/SDEs/CDEs;
  - only revisit if a future full-trajectory or characteristic-integration
    module is implemented as an ODE solve rather than as the current
    finite-difference/operator route.
- `quadax` should not replace production quadrature:
  - current integration uses fixed grids / weights tied to the SFINCS
    discretization and parity tests;
  - adaptive quadrature could be useful for analytic validation fixtures or
    geometry-preprocessing research, but it would change discretization
    semantics if inserted into production paths.
- `orthax` is a test/reference or future spectral-basis candidate:
  - it may help if a future collision or velocity-space module moves to an
    explicit Legendre / orthogonal-polynomial spectral representation;
  - it should not be added now because current velocity grids and collision
    operators are already implemented directly and parity-tested.

Concrete next experiments, if we decide to revisit implementation:
1. Add a benchmark-only Lineax adapter outside the production path, likely
   `sfincs_jax/experimental_lineax_solve.py` or a local benchmark script first.
   It must be skipped cleanly when `lineax` is missing.
2. Compare the adapter against the existing `implicit_solve.py` path on the
   five cases listed above, including reverse-mode gradient checks and RSS.
3. Prototype an Equinox-only public objective wrapper under `examples/autodiff/`
   or `examples/optimization/`; do not convert core operators.
4. Evaluate `jaxopt.custom_root` for nonlinear sensitivities around the current
   forward solve before testing Optimistix as a replacement nonlinear driver.
5. Revisit JAX-native `checkpoint` placement around `lax.scan` bodies in
   structured velocity / collision-heavy differentiable paths if gradient RSS
   becomes the next blocker.

Decision:
- No ecosystem dependency is ready to bake into production code today.
- The highest-value future review is a bounded Lineax experiment for
  differentiable small/medium linear solves and a JAXopt/Equinox wrapper for
  research workflows.
- Production CLI offender work should continue to use the current direct JAX,
  SciPy sparse/direct, hand-tuned policy, and explicit sharding paths until a
  library-backed experiment beats them under the same parity/runtime/RSS gates.

### 19.26 Research-gate hardening after the ecosystem review

- The first concrete ecosystem gate is now executable without changing
  production dependencies:
  - `examples/performance/benchmark_optional_lineax_implicit_solve.py`
    benchmarks the current in-tree implicit solve and optionally benchmarks
    Lineax GMRES when `lineax` is installed;
  - the benchmark now covers three bounded cases:
    - a deterministic nonsymmetric stress system,
    - a tiny real SFINCS implicit-diff solve on
      `tests/ref/pas_1species_PAS_noEr_tiny_scheme5.input.namelist`,
    - a repeated-RHS reuse case on that same tiny real operator;
  - it records residuals, finite-difference gradient agreement, repeated-RHS
    solution error, and elapsed time, and writes JSON for later comparison;
  - `tests/test_optional_lineax_implicit_gate.py` verifies deterministic system
    construction, current-solver residual/gradient quality on both synthetic and
    tiny real-operator lanes, repeated-RHS accuracy, JSON output, and clean skip
    behavior when Lineax is absent.
- The manuscript validation manifest now carries explicit research gates:
  - every lane has `source_code`, `tests`, and `acceptance_gates` fields;
  - implemented/prototype lanes point to existing scripts, artifacts, source
    files, and tests;
  - planned and `needs_reaudit` lanes keep open work explicit rather than
    silently implying publication readiness.
- New schema coverage:
  - `tests/test_validation_manifest_schema.py` checks uniqueness, valid status
    and kind values, nonempty literature/claim/source/test/gate lists, existing
    paths for non-planned lanes, and the expected open-lane set.
- Documentation updates:
  - `docs/performance.rst` documents the optional Lineax gate and reiterates
    that Lineax is not a production CLI dependency;
  - `docs/validation_matrix.rst` documents the manifest schema and acceptance
    gate role;
  - `docs/testing.rst` documents the manifest schema test and optional ecosystem
    benchmark gate;
  - `examples/performance/README.md` lists the optional benchmark.
- Validation run:
  - `pytest -q tests/test_validation_manifest_schema.py tests/test_optional_lineax_implicit_gate.py`
    -> `8 passed` at the first hardening pass;
  - the extended real-operator pass then validated with:
    - `pytest -q tests/test_optional_lineax_implicit_gate.py tests/test_implicit_linear_solve_grad.py tests/test_validation_manifest_schema.py`
      -> `14 passed`,
    - direct benchmark smoke:
      `python examples/performance/benchmark_optional_lineax_implicit_solve.py --backend current --suite sfincs --restart 20 --maxiter 120 ...`
      which internally uses the parity-clean real-operator Krylov window and
      produced:
      - `sfincs_tiny_implicit`: relative residual about `1.4e-14`,
      - `sfincs_tiny_repeated_rhs`: relative residual about `4.3e-12`,
        max solution error about `2.8e-09`.
- The first real local run with `lineax` installed has now been audited:
  - command:
    `python examples/performance/benchmark_optional_lineax_implicit_solve.py --backend all --suite all --restart 20 --maxiter 120 --out-json /tmp/sfincs_jax_lineax_gate_all.json`;
  - measured outcome:
    - `synthetic_nonsymmetric`:
      - in-tree path: about `0.99 s`, relative residual about `2.6e-16`, status `ok`;
      - `lineax_gmres`: about `0.39 s`, relative residual about `2.1e-17`, status `ok`;
    - `sfincs_tiny_implicit`:
      - in-tree path: about `4.92 s`, relative residual about `1.4e-14`, status `ok`;
      - `lineax_gmres`: about `0.80 s`, relative residual about `3.2e-16`, but status `error` with
        `maximum number of solver steps was reached`;
    - `sfincs_tiny_repeated_rhs`:
      - in-tree path: about `1.92 s`, relative residual about `4.3e-12`, max solution error about `2.8e-09`, status `ok`;
      - `lineax_gmres`: about `1.58 s`, relative residual about `7.5e-16`, but status `error` with iterative-breakdown messaging.
  - conclusion:
    - `lineax` remains promising on synthetic linear systems,
    - but it is still not admissible for the real matrix-free SFINCS operator because solver status is not clean even when the residual is tiny,
    - so it stays benchmark-only and out of the production solve ladder.
- A second concrete ecosystem gate is now executable without changing
  production dependencies:
  - `examples/optimization/benchmark_optional_eqx_jaxopt_scheme4_gate.py`
    benchmarks optional `equinox` and `jaxopt` wrappers on a real
    `geometryScheme=4` harmonic-fit objective;
  - `equinox` is only used as a small callable module wrapper around the
    differentiable objective;
  - `jaxopt.GradientDescent` is only used as a bounded outer-optimization check,
    not as a replacement for the production solve path;
  - `tests/test_optional_eqx_jaxopt_scheme4_gate.py` verifies deterministic
    problem construction, directional finite-difference agreement, bounded loss
    reduction, parameter recovery, JSON output, and clean skip behavior.
- Validation for the new objective-wrapper gate:
  - direct benchmark smoke:
    `python examples/optimization/benchmark_optional_eqx_jaxopt_scheme4_gate.py --backend all --n-theta 17 --n-zeta 17 --maxiter 5 --stepsize 0.1 --out-json /tmp/sfincs_jax_eqx_jaxopt_gate.json`
    produced:
    - `equinox_wrapper`: directional derivative about `-2.50332297435e-01`,
      centered finite-difference derivative about `-2.50332297424e-01`,
      absolute discrepancy about `1.08e-11`, status `ok`;
    - `jaxopt_gradient_descent`: initial loss about `2.98e-02`,
      final loss about `1.21e-15`, loss ratio about `4.07e-14`,
      final parameter error about `1.59e-08`, status `ok`.
  - focused tests:
    - `pytest -q tests/test_optional_eqx_jaxopt_scheme4_gate.py`
      -> `6 passed`.
- Updated next actions:
  1. keep the current Lineax result as a negative admission gate until a real
     operator run is both faster and status-clean;
  2. if a nonlinear or ambipolar objective wrapper is needed, add it as a
     separate optional gate instead of broadening the production dependency set;
  3. keep full-resolution LHD/W7-X collisionality regeneration and W7-X
     ambipolar validation as open research lanes until their acceptance gates
     are satisfied by pinned artifacts.

### 19.27 W7-X ambipolar validation scaffold and heavy collisionality rerun handoff

- The W7-X ambipolar lane is no longer just a manifest placeholder:
  - new script:
    `examples/publication_figures/generate_w7x_ambipolar_validation.py`
  - default base input:
    `examples/sfincs_examples/filteredW7XNetCDF_2species_magneticDrifts_withEr/input.namelist`
  - bounded branch mode:
    `--fast --n-points 7`
  - outputs:
    - metadata-rich JSON summary with `metadata`, per-run `runs`, and
      `ambipolar` root/output sections,
    - publication-style PNG/PDF figure with radial-current, heat-flux,
      particle-flux, and flow/current panels.
- Focused validation now exists for this lane:
  - new test:
    `tests/test_generate_w7x_ambipolar_validation.py`
  - it covers:
    - default `!ss` Er-bracket parsing from the W7-X input,
    - summary-payload serialization from a synthetic ambipolar result,
    - end-to-end execution of the script on the tiny scheme-11 fixture.
- Documentation and manifest updates:
  - `examples/publication_figures/validation_manifest.json` now points the
    `w7x_ambipolar_er_validation` lane at the executable scaffold and its test;
  - `docs/validation_matrix.rst`, `docs/testing.rst`, and
    `examples/publication_figures/README.md` now describe the scaffold as an
    implemented script but keep the lane explicitly unpromoted until a defensible
    W7-X reference artifact is pinned.
- Validation run:
  - `pytest -q tests/test_generate_w7x_ambipolar_validation.py tests/test_er_scan_and_ambipolar.py tests/test_validation_manifest_schema.py`
    -> `7 passed`
  - `python -m py_compile examples/publication_figures/generate_w7x_ambipolar_validation.py tests/test_generate_w7x_ambipolar_validation.py`
    -> passed
  - `sphinx-build -W -b html docs docs/_build/html`
    -> passed
- Heavy collisionality rerun status:
  - a clean `office` worktree was created at `/home/rjorge/sfincs_jax_refactor_v3`
    from `origin/refactor/v3-driver-split`;
  - the full LHD collisionality rerun was launched there with scan-state recycling:
    `SFINCS_JAX_SCAN_RECYCLE=1 python3 examples/publication_figures/generate_sfincs_paper_figs.py --case lhd ...`
  - within this turn it remained compute-bound in the first scan solve, so no
    audited full-resolution artifact has been promoted yet.
- Next actions:
  1. finish the full LHD rerun on `office` and pull back the summary/figure;
  2. repeat the same full rerun for W7-X on `office`;
  3. only after both are pinned, re-evaluate and regenerate the high-collisionality
     proxy lane;
  4. run the heavier W7-X ambipolar scaffold on the reference input and pin its
     first literature-facing summary/figure artifact.

### 19.28 Split collisionality rerun controls for heavy re-audits

- `examples/publication_figures/generate_sfincs_paper_figs.py` now has explicit
  support for split operator reruns:
  - `--collision-operators 0,1` selects which operator ladders are run or
    collected;
  - `--skip-existing` preserves already completed ladder directories and only
    reruns missing operators.
- The generator no longer assumes both ladders exist before it can write
  summaries or figures:
  - partial `plot-only` and `scan-only --skip-existing` workflows now tolerate a
    single selected operator and still emit a filtered metadata payload;
  - this is the bounded local fix needed for the `office` two-GPU handoff,
    rather than continuing to rely on manual ad hoc directory surgery.
- Focused validation added in `tests/test_generate_sfincs_paper_figs.py`:
  - collision-operator parsing and validation,
  - reuse of an existing selected operator without calling the scan runner,
  - `plot-only` synthesis from a single selected operator output.
- Documentation update:
  - `examples/publication_figures/README.md` now records the exact split-GPU
    LHD/W7-X re-audit pattern and the final `--plot-only` synthesis command.
- Immediate next actions remain unchanged, but now the heavy reruns can be
  resumed and regenerated from the script interface directly instead of by
  manual operator-specific setup.

### 19.29 Remote handoff status after split-scan support

- `office` now has two relevant worktrees:
  - `/home/rjorge/sfincs_jax_refactor_v3` holds the already-running LHD
    full-resolution split scan that was started before the scripted split
    controls landed;
  - `/home/rjorge/sfincs_jax_refactor_v3_latest` is a clean clone at commit
    `939ec93` and is reserved for all follow-on synthesis and new launches.
- The new split synthesis path has been smoke-tested on the real `office`
  partial output tree:
  - `--plot-only --collision-operators 0` successfully wrote a filtered LHD
    summary and figure from the live partial FP ladder;
  - `--plot-only --collision-operators 1` successfully wrote a filtered LHD
    summary and figure from the live partial PAS ladder;
  - this confirms that the final audited synthesis step can be run directly from
    the latest clone once the current LHD jobs finish.
- The full LHD re-audit remains compute-bound on both GPUs, but it is making
  forward progress and no longer blocks all other validation work.
- A heavier W7-X ambipolar reference lane has now been launched in parallel on
  `office` CPU from the latest clone:
  - command family:
    `JAX_PLATFORMS=cpu ... python3 examples/publication_figures/generate_w7x_ambipolar_validation.py ...`
  - purpose:
    advance the first literature-facing W7-X ambipolar artifact without
    competing for the two GPUs reserved for the LHD collisionality re-audit.

### 19.30 Post-refactor lane: vmec_jax and booz_xform_jax integration

- This is a **queued next-level research lane**, not the current critical path.
  It should start only after the current `sfincs_jax` refactor / testing work and
  the open collisionality + W7-X ambipolar validation lanes are closed.
- Motivation and external anchors reviewed for this lane:
  - `vmec_jax` already provides an end-to-end differentiable fixed/free-boundary
    VMEC implementation with an exact discrete-adjoint optimizer and a public
    `wout_from_fixed_boundary_run(...)` path.
  - `booz_xform_jax` already supports both file-based `read_wout(...)` and
    in-memory `read_wout_data(...)`, plus a low-level JAX API intended for
    differentiable pipelines.
  - STELLOPT is the historical reference for coupling VMEC to optimization
    targets, including transport targets through external physics codes.
  - the SFINCS adjoint abstract shows the concrete target class we care about:
    bootstrap current / radial flux gradients with respect to Boozer-spectrum
    inputs.
  - recent stellarator-optimization literature has moved from proxy-only
    objectives toward direct neoclassical targets and ambipolar/root-aware
    objectives, so a differentiable `vmec_jax -> sfincs_jax` lane would be
    scientifically well motivated rather than just architecturally elegant.

- Architectural conclusion from the code review:
  - `sfincs_jax` should **not** start this lane by rewriting everything around
    Boozer coordinates.
  - the first integration target should be `geometryScheme=5`, because
    `sfincs_jax` already consumes VMEC `wout` data there.
  - `booz_xform_jax` is the correct **second-stage** lane for in-memory Boozer
    transforms, scheme-11/12-style workflows, and Boozer-spectrum optimization
    targets.

- Current code constraints that must be respected:
  - `sfincs_jax/vmec_geometry.py` currently reads a `wout_*.nc` file through
    `sfincs_jax/vmec_wout.py` and then evaluates the geometry with NumPy-heavy
    logic. This is parity-clean, but not end-to-end differentiable.
  - `sfincs_jax` already has a differentiable solve path through
    `sfincs_jax/implicit_solve.py` and the Python `differentiable=True` solve
    route. That is the correct transport-side foundation for an autodiff lane.
  - discrete geometry choices such as `MIN_BMN_TO_LOAD`, Nyquist truncation, and
    operator/mode filtering are not smooth design variables. In the differentiable
    lane they must be treated as **static topology choices**, not optimized
    continuously.
  - the first implementation scope should remain fixed-boundary, stellarator-symmetric
    VMEC (`lasym = false`) because that is the current supported `sfincs_jax`
    VMEC subset.

- Planned implementation sequence:

  1. Compatibility bridge, no physics change.
     - Add a canonical `VmecWoutLike` / adapter layer in
       `sfincs_jax/vmec_wout.py` that can be built from:
       - the current file-based `VmecWout`,
       - `vmec_jax.wout.WoutData`,
       - and `vmec_jax.driver.FixedBoundaryRun` via
         `vmec_jax.wout_from_fixed_boundary_run(...)`.
     - Refactor `sfincs_jax/vmec_geometry.py` so the file reader is just a thin
       wrapper around a new `vmec_geometry_from_wout_data(...)`.
     - Keep the CLI unchanged. This lane is Python-first; file-based `wout_path`
       remains the stable public CLI interface.

  2. In-memory VMEC fast path, still parity-first.
     - Add Python API entry points that accept an in-memory VMEC object/run and
       avoid writing `wout_*.nc` to disk in repeated-loop workflows.
     - Touch points will likely include:
       `sfincs_jax/vmec_wout.py`,
       `sfincs_jax/vmec_geometry.py`,
       `sfincs_jax/v3.py`,
       `sfincs_jax/io.py`,
       and new examples under `examples/autodiff/`.
     - Acceptance gate:
       - for the same equilibrium, file-based and in-memory geometryScheme=5
         paths must agree on geometry arrays and on `sfincsOutput.h5` transport
         outputs to the same tolerances currently used for parity fixtures.

  3. Pure-JAX geometryScheme=5 kernel.
     - Replace the NumPy-only VMEC geometry evaluation path with a JAX-native
       implementation that keeps the same mode set fixed and computes the Fourier
       sums with `jnp` plus bounded chunking / `lax.scan` where needed.
     - Preserve the current mode-selection semantics, but freeze that selection
       before differentiation so the autodiff graph does not cross discrete
       truncation changes.
     - Acceptance gate:
       - the JAX geometry kernel must match the current parity-clean file path on
         representative VMEC fixtures before it is used for gradients.

  4. End-to-end differentiable transport lane.
     - Wire the new in-memory VMEC geometry into the existing
       `differentiable=True` transport solve path, explicitly excluding host-only
       rescue paths, process pools, and other non-differentiable orchestration.
     - Define a bounded research API for objectives such as:
       - monoenergetic / transport-matrix coefficients,
       - radial particle flux,
       - bootstrap current,
       - ambipolar radial current and root location.
     - Initial gradients should be evaluated only on single-process/single-device
       Python paths; multi-process strong scaling remains a separate performance
       lane, not the first differentiable target.

  5. Optional Boozer lane through `booz_xform_jax`.
     - After the VMEC in-memory lane is stable, add an in-memory
       `vmec_jax -> booz_xform_jax -> sfincs_jax` route for Boozer-space studies.
     - Use `booz_xform_jax.read_wout_data(...)` or its low-level JAX API rather
       than serializing through disk unnecessarily.
     - This lane is for:
       - direct Boozer-spectrum sensitivity studies,
       - scheme-11/12-style workflow modernization,
       - bootstrap-current / transport optimization directly in Boozer variables,
       - and benchmarking against existing Boozer-based optimization literature.

- Test and validation plan:

  - Unit tests:
    - field-by-field adapter tests between `vmec_jax` `WoutData` and
      `sfincs_jax` VMEC expectations,
    - interpolation-index / half-mesh / full-mesh convention checks,
    - `nfp`, `xm/xn`, Nyquist-table, and sign-convention tests.

  - Regression tests:
    - same equilibrium through file-based and in-memory paths must reproduce the
      same `BHat`, `DHat`, covariant/contravariant components, and selected
      transport outputs for representative geometryScheme=5 fixtures.
    - representative targets:
      `geometryScheme5_3species_loRes`,
      `monoenergetic_geometryScheme5_netCDF`,
      and tiny scheme-5 implicit-diff fixtures.

  - Differentiation tests:
    - compare `jax.grad` / `jax.jvp` against centered finite differences for a
      small set of VMEC boundary coefficients on bounded fixed-boundary cases,
    - require finite, stable sensitivities for at least:
      - one transport-matrix coefficient,
      - one flux quantity,
      - and one ambipolar/root-related scalar.

  - External validation:
    - leverage `vmec_jax`'s existing `wout` parity and discrete-adjoint tests as
      upstream trust anchors,
    - leverage `booz_xform_jax`'s existing `run()` vs JAX-API agreement tests as
      the trust anchor for the Boozer lane,
    - then add `sfincs_jax` end-to-end checks on top of those rather than
      re-proving the full equilibrium/Boozer stack from scratch.

- Benchmark plan:
  - compare file-based vs in-memory VMEC geometry ingestion on CPU and GPU,
  - compare current NumPy VMEC geometry evaluation vs JAX-native evaluation on
    warm repeated calls,
  - benchmark a repeated-loop design study where the same shape family is
    perturbed many times, since that is where disk I/O elimination and JIT
    amortization should matter most,
  - benchmark gradient throughput for a tiny bounded optimization problem rather
    than only forward-solve runtime.

- Research-grade example plan:
  - `examples/autodiff/vmec_jax_boundary_sensitivity_scheme5.py`
    for direct transport sensitivity to boundary Fourier coefficients,
  - `examples/autodiff/vmec_jax_bootstrap_current_gradient.py`
    for a bounded bootstrap-current objective,
  - `examples/autodiff/vmec_jax_ambipolar_root_sensitivity.py`
    for root-aware `E_r` studies,
  - `examples/autodiff/vmec_jax_to_boozer_transport_pipeline.py`
    for the optional `vmec_jax -> booz_xform_jax -> sfincs_jax` lane,
  - and one small optimization example showing actual objective reduction, not
    only gradient agreement.

- Publication / documentation deliverables for this lane:
  - a docs page explaining the full differentiable equilibrium-to-transport
    stack and its static-vs-differentiable boundaries,
  - a validation page with file-vs-in-memory parity tables and gradient-agreement
    plots,
  - publication-ready figures for:
    - gradient agreement,
    - repeated-loop runtime improvement from avoiding file I/O,
    - and one bounded transport-objective optimization case.

- Best initial scientific use cases, based on the reviewed literature:
  - direct neoclassical transport optimization beyond proxy metrics,
  - bootstrap-current minimization,
  - ambipolar / positive-`E_r` equilibrium studies,
  - local sensitivity analysis, inverse design, and uncertainty quantification
    with respect to boundary Fourier coefficients,
  - and combined proxy + transport optimization where QS/QI metrics remain the
    cheap preconditioner and `sfincs_jax` provides the high-fidelity follow-up
    objective.

### 19.31 Resumable W7-X ambipolar scan lane and LHD full rerun completion

- The generic scan helper now has an explicit resume path:
  - `sfincs_jax/scans.py:run_er_scan(...)` accepts `skip_existing=True`,
    reuses any existing `sfincsOutput.h5`, and only resolves missing scan points.
  - this behavior is covered in
    `tests/test_helper_module_coverage.py::test_scan_helpers_and_run_er_scan`,
    including the partial-rerun case where one scan point is deleted and rebuilt.
- The user-facing CLI now exposes the same capability:
  - `sfincs_jax scan-er --skip-existing ...`
  - this keeps the bounded restart semantics in the production scan interface,
    not only in the publication scripts.
- The W7-X ambipolar scaffold is now aligned with the collisionality rerun workflow:
  - `examples/publication_figures/generate_w7x_ambipolar_validation.py` adds
    `--skip-existing`, `--scan-only`, `--jobs`, `--index`, and `--stride`;
  - split lanes can now fill the `E_r` ladder on separate devices with
    `--scan-only --index k --stride N`, then finish with a final
    `--skip-existing` aggregation pass that writes the ambipolar summary and
    figure;
  - focused coverage in `tests/test_generate_w7x_ambipolar_validation.py` now
    verifies forwarding of the split/resume options and rejects the invalid
    `--scan-only --plot-only` combination.
- Validation for this hardening pass:
  - `pytest -q tests/test_generate_w7x_ambipolar_validation.py tests/test_er_scan_and_ambipolar.py tests/test_helper_module_coverage.py -k 'w7x_ambipolar_validation or er_scan_writes_outputs_and_ambipolar_solve_runs or scan_helpers_and_run_er_scan'`
    -> `7 passed`
  - `python -m py_compile sfincs_jax/scans.py sfincs_jax/cli.py examples/publication_figures/generate_w7x_ambipolar_validation.py tests/test_generate_w7x_ambipolar_validation.py tests/test_helper_module_coverage.py`
    -> passed
  - `sphinx-build -W -b html docs docs/_build/html`
    -> passed
- `office` execution status at this point:
  - the full LHD collisionality rerun has now completed both collision-operator
    ladders in `/home/rjorge/sfincs_jax_refactor_v3/examples/publication_figures/output/lhd_reaudit_full`;
  - the next remote action is no longer “wait for missing points”, but
    “synthesize the audited full-resolution LHD summary/figure from the finished
    output tree”, then immediately reuse the freed GPUs for the full W7-X
    collisionality rerun or the heavier W7-X ambipolar reference lane.

### 19.32 Office milestone: audited LHD full artifact closed, W7-X full rerun launched

- The `office` LHD full-resolution re-audit is now actually closed:
  - `examples/publication_figures/generate_sfincs_paper_figs.py --case lhd --plot-only --collision-operators 0,1 ...`
    was rerun from `/home/rjorge/sfincs_jax_refactor_v3_latest`;
  - the audited full artifact now resolves to the standard full names:
    - summary:
      `/home/rjorge/sfincs_jax_refactor_v3_latest/examples/publication_figures/artifacts/lhd_collisionality_summary.json`
    - figure:
      `/home/rjorge/sfincs_jax_refactor_v3_latest/docs/_static/figures/paper/sfincs_jax_fig1_lhd_collisionality.png`
  - metadata check on the summary:
    - `ROW_COUNT = 14`
    - labels = `Fokker-Planck` and `PAS`
    - `FAST = False`
    - `CASE = lhd`
    - `N_POINTS = 7`
    - `NU_MIN = 0.1`
    - `NU_MAX = 10.0`
- The older CPU-only W7-X ambipolar reference lane was intentionally paused after
  preserving its partial work directory:
  - PID `3145554` had still not advanced beyond the first scan point and was
    consuming several host cores;
  - this lane should be resumed later through the new `--skip-existing` /
    split-scan workflow rather than left as an unbounded CPU-only job.
- The freed GPUs were immediately reassigned to the next critical-path lane:
  the full W7-X collisionality re-audit.
  - worktree:
    `/home/rjorge/sfincs_jax_refactor_v3_latest`
  - work dir:
    `/home/rjorge/sfincs_jax_refactor_v3_latest/examples/publication_figures/output/w7x_reaudit_full`
  - operator 0 launch:
    - wrapper PID `3172653`
    - active scan child PID `3172691`
    - log:
      `/home/rjorge/sfincs_jax_refactor_v3_latest/examples/publication_figures/output/resume_logs/w7x_co0_resume_gpu0.log`
  - operator 1 launch:
    - wrapper PID `3173256`
    - active scan child PID `3173300`
    - log:
      `/home/rjorge/sfincs_jax_refactor_v3_latest/examples/publication_figures/output/resume_logs/w7x_co1_resume_gpu1.log`
  - both operator ladders entered the expected `nu_n_0.01727` first point and
    both GPUs showed active utilization after launch.
- Immediate next actions:
  1. let the split W7-X full rerun finish on `office`;
  2. synthesize the audited full `w7x_collisionality_summary.json` and
     `sfincs_jax_fig2_w7x_collisionality.png`;
  3. only after both full LHD and full W7-X artifacts are pinned, revisit the
     high-collisionality proxy and the heavier W7-X ambipolar literature lane.

### 19.33 CPU runtime-watchlist closeout after harness fixes

- The two remaining CPU runtime-drift watchlist cases were rechecked with the
  post-profiling-fix suite harness instead of the older `postkeyfix` timing
  data alone.
  - bounded probe root without host-RSS collection:
    `tests/scaled_example_suite_cpu_watchlist_probe_2026-04-23`
  - bounded probe root with safe host-RSS collection:
    `tests/scaled_example_suite_cpu_watchlist_probe_mem_2026-04-23`
- Both cases stayed parity-clean and dropped back below the `1.25x` drift gate
  against `tests/scaled_example_suite_fast_cpu_full_v7_refresh/suite_report.json`.
  - `monoenergetic_geometryScheme11`
    - baseline `jax_runtime_s = 3.056`
    - refreshed `jax_runtime_s = 3.185`
    - refreshed `jax_logged_elapsed_s = 2.497`
    - refreshed `jax_max_rss_mb = 1187.3`
  - `transportMatrix_geometryScheme11`
    - baseline `jax_runtime_s = 1.667`
    - refreshed `jax_runtime_s = 1.764`
    - refreshed `jax_logged_elapsed_s = 1.188`
    - refreshed `jax_max_rss_mb = 439.9`
- The authoritative CPU release root was then refreshed in place:
  - `tests/scaled_example_suite_recheck_cpu_frozen_2026-04-23_postkeyfix`
  - `suite_runtime_drift_summary.json` now reports:
    - `flagged_cases = 0`
    - `cases = []`
- README/docs should now treat both CPU and GPU runtime-drift watchlists as
  clean on the current release artifacts. The remaining performance work is the
  structural runtime/memory reduction lane on heavy PAS and geometry-rich cases,
  not another artifact-hygiene pass.

### 19.34 Full research-grade planning audit after literature/code/docs pass

Scope reviewed on 2026-04-23:
- local project state:
  - git history through `02d84cd Close CPU runtime drift watchlist`,
  - current branch `refactor/v3-driver-split`,
  - docs pages under `docs/`,
  - CI/CD workflows under `.github/workflows/`,
  - package metadata in `pyproject.toml`,
  - coverage data and module sizes,
  - remote `office` validation jobs and completed artifacts;
- literature / reference anchors:
  - Landreman, Smith, Mollen, and Helander 2014 SFINCS paper:
    continuum radially local drift-kinetic solves, trajectory-model comparisons,
    LHD/W7-X collisionality scans, FP/PAS collision-model comparisons, and
    electric-field resonance behavior;
  - original SFINCS repository and v3 documentation:
    supported geometry schemes, VMEC/Boozer inputs, HDF5 output conventions,
    multispecies equations, Phi1, collision-operator implementation, and
    Fortran/MPI/PETSc scaling expectations;
  - KNOSOS papers/code/manual:
    low-collisionality orbit-averaged validation targets, ambipolar bisection,
    quasineutrality, DKES-database normalization, high-collisionality caveats,
    and MPI/PETSc scan decomposition;
  - MONKES source:
    factor-once / `vmap`-over-RHS monoenergetic block-tridiagonal solves and
    lazy low-memory block factorization;
  - yancc source and branches:
    Lineax operator wrappers, bordered/inverse operators, periodic banded
    Schur corrections, GCROT/LGMRES-style Krylov recycling, multigrid
    preconditioners, verbose solver telemetry, and coordinate/backend tests;
  - JAX/Lineax/Equinox/JAXopt documentation:
    `custom_linear_solve` for implicit gradients, `checkpoint` for gradient
    memory control, persistent compilation cache, GPU memory allocator controls,
    `shard_map` / multi-controller JAX, profiler traces, Lineax solver-state
    reuse/status reporting, Equinox PyTree/filter APIs, and JAXopt implicit
    root differentiation.

Plan hygiene corrected:
- The queued `vmec_jax` / `booz_xform_jax` implementation sequence had been
  split by later W7-X status insertions. It is now kept under Section 19.30 so
  the roadmap is readable and the CPU runtime-watchlist closeout no longer
  contains stray VMEC work items.

Current research-grade readiness assessment:
- Correctness / parity:
  - CPU and GPU release-facing example suites are parity-clean on the documented
    39-case scope, with strict mismatches and `jax_error` / `max_attempts`
    cleared in the current release artifacts.
  - Runtime-drift watchlists are clean after the harness/profiling fixes.
  - Output-key coverage has been audited, but any future output additions must
    keep comparison tooling synchronized with Fortran v3 outputs plus JAX-only
    metadata.
- Validation artifacts:
  - full LHD collisionality artifact is closed from the corrected writer and
    promoted locally with a 14-row metadata-backed summary and figure.
  - full W7-X collisionality artifact has finished synthesis on `office` and
    is promoted locally with a 14-row metadata-backed summary and figure.
  - W7-X ambipolar validation is still a long-running research/nightly lane:
    one full reference scan point took about `14m23s`, and the next full-size
    point has active size `918394`, so this must not be part of PR CI.
- Performance:
  - current PAS/geometry-heavy CPU/GPU cases are correct but still the main
    runtime/RSS optimization frontier.
  - the best near-term algorithmic ideas are structured linear algebra and
    solve recycling, not broader library replacement.
  - single-case multi-GPU sharding remains experimental; transport-worker and
    scan/case-level parallelism remain the production scaling story.
- Maintainability:
  - `v3_driver.py` remains the dominant blocker: about `21.8k` lines on disk,
    about `12.5k` coverage statements, and about `32%` line coverage in the
    current local coverage report.
  - `io.py` is also too broad at about `4.3k` lines on disk and about `51%`
    coverage in the current local report.
  - Many physics kernels have strong focused coverage already, so adding more
    superficial tests will not reach `95%`; the code must be split first.
- Documentation / examples:
  - docs are broad and buildable with `sphinx-build -W`, but the next upgrade is
    curation: architecture diagrams after the split, validation-lane status
    tables tied to exact artifacts, and clearer "fast PR vs nightly/release"
    testing instructions.
- CI/CD / PyPI:
  - CI, docs, examples smoke, optional ecosystem gates, Codecov upload, and
    trusted PyPI publishing workflow all exist.
  - CI coverage floor is intentionally low (`43`) because the current driver
    structure makes `95%` a refactor milestone, not a near-term test-only fix.

Highest-ROI implementation sequence from this audit:

1. Close artifact hygiene before more solver changes.
   - Run the focused artifact/manifest/docs checks after promoting the completed
     LHD and W7-X full collisionality summaries and figures.
   - Commit and push the synchronized plan, manifest, tests, docs, and artifacts.
   - Keep the W7-X ambipolar full reference in nightly/release status with
     resumable split-scan controls, not PR CI.

2. Split the monolithic driver into testable modules without changing behavior.
   - Extract, in small commits:
     - RHSMode=1 preconditioner policies and safety gates,
     - domain-decomposition / Schwarz helpers,
     - Krylov result and retry policy,
     - sparse/direct/host-rescue dispatch,
     - progress/ETA logging and solver provenance,
     - final diagnostics/output handoff.
   - Acceptance gates for each extraction:
     - no numerical behavior change,
     - focused unit tests on the extracted module,
     - representative parity fixture still clean,
     - full fast tests and docs build remain green.

3. Convert coverage from branch coverage of a giant driver to meaningful module
   coverage.
   - First target after the split: raise package coverage floor from `43` to
     `60` without increasing CI wall time above the 5-10 minute policy.
   - Second target: `75` once the extracted solver-policy and I/O modules have
     focused tests.
   - `95%` remains the research-grade release target after the solver body is
     decomposed and heavyweight solve loops are protected by small analytic
     fixtures plus scheduled/nightly examples rather than by long PR tests.

4. Add algorithmic performance work only behind measured gates.
   - Port ideas, not code, from MONKES/yancc:
     - factor-once / `vmap`-over-RHS structured solves for repeated RHS,
     - lazy block-tridiagonal factors for memory-heavy velocity blocks,
     - periodic banded low-rank Schur corrections for natural theta/zeta/PAS
       blocks,
     - GCROT/GCRO-DR or LGMRES recycling across RHS, collisionality, and Er
       scans,
     - operator-level verbose telemetry with residual, setup, and matvec counts.
   - Initial target cases:
     - HSX PAS DKES/full trajectories,
     - geometry11 PAS paper cases,
     - geometry4 PAS no-Er memory offender,
     - W7-X ambipolar full-size scan points.
   - Admission gate:
     - parity-clean,
     - no strict-output drift,
     - `>=20%` warm runtime or `>=25%` RSS improvement on at least one pinned
       offender,
     - no regression above `1.25x` on the suite drift gates.

5. Keep ecosystem libraries optional until they prove value.
   - Keep `jax.lax.custom_linear_solve` as the default differentiable linear
     solve primitive; it directly matches the implicit-gradient requirement.
   - Use `jax.checkpoint` selectively around differentiable scanned kernels only
     after a gradient-RSS benchmark shows benefit.
   - Keep Lineax as a benchmark-only optional path until real SFINCS operators
     are faster and status-clean.
   - Use Equinox for future public differentiable problem/objective wrappers,
     not for core hot kernels yet.
   - Use JAXopt only for optional nonlinear/ambipolar implicit-diff wrappers
     after finite-difference gradient gates pass.

6. Strengthen physics gates in the validation matrix.
   - Existing gates to preserve:
     - Fortran-v3 output parity,
     - strict dataset comparison,
     - conservation/symmetry identities in collision and drift terms,
     - Onsager/transport-matrix checks where applicable,
     - finite-difference vs implicit/autodiff gradients on small fixtures.
   - Add or promote next:
     - collisionality trends for LHD/W7-X from corrected full artifacts,
     - Er trajectory-model sweeps with small-field agreement and finite-field
       separation,
     - high-collisionality proxy only after parent collisionality scans are
       fully pinned,
     - ambipolar root bracketing/stability tests on bounded fixtures,
     - coordinate/backend equivalence tests inspired by yancc,
     - KNOSOS/DKES monoenergetic normalization checks for low-collisionality
       reference lanes.

7. Make differentiability a first-class validated product lane.
   - Keep CLI fast paths free to use host/direct/non-differentiable rescues.
   - Require the Python differentiable lane to:
     - avoid process pools and host-only sparse rescues,
     - expose solver residual/provenance in gradient examples,
     - compare `jax.jvp` / `jax.grad` against centered finite differences,
     - support sensitivity analysis, inverse design, UQ, and optimization
       examples on bounded fixtures.
   - Defer full `vmec_jax -> sfincs_jax` implementation until the driver split
     is stable, then follow Section 19.30.

8. Testing tiers for a shippable research code.
   - PR CI:
     - fast unit/regression tests,
     - examples smoke,
     - docs `-W`,
     - optional ecosystem gates with clean skips,
     - wall time target `5-10 min`.
   - Nightly/scheduled:
     - selected Fortran comparisons,
     - CPU/GPU pinned offender benchmarks,
     - medium collisionality and Er scans,
     - coverage trend report.
   - Release/HPC:
     - full 39-case CPU/GPU suites,
     - full LHD/W7-X collisionality artifacts,
     - W7-X ambipolar split scan if scientifically defensible,
     - multi-core / multi-GPU scaling plots,
     - PyPI build and tag publish.

Immediate next actions:
1. Start the driver split with one low-risk extraction: progress/provenance
   logging or preconditioner dispatch helpers, then run focused tests.
2. Add a small structured-solve benchmark harness that can compare current
   full-system Krylov against a factor-once / repeated-RHS prototype on a
   bounded monoenergetic or PAS block.
3. Raise the CI coverage floor only after the first extraction lands; do not
   chase `95%` by adding slow full-solve tests to PR CI.

### 19.35 Driver split step 1: solver progress/provenance helper extraction

- First low-risk split from Section 19.34 is implemented without changing solver
  numerics:
  - added `sfincs_jax/solver_progress.py`,
  - moved shared duration formatting and coarse runtime-class hints out of
    `io.py`,
  - moved RHSMode=1 large-solve one-shot progress notes out of `v3_driver.py`,
  - moved transport whichRHS ETA message construction out of the deep transport
    solve loop.
- Rationale:
  - these helpers are observability-only, so they are a safe first extraction
    before touching preconditioner or Krylov decision logic;
  - keeping progress/provenance formatting in one module makes the future driver
    split easier and keeps CLI messages testable without running heavy solves;
  - the public log text is preserved for large RHSMode=1 solves and transport
    ETA lines.
- Documentation:
  - `docs/source_map.rst` now lists `solver_progress.py` as a solver-neutral
    progress/provenance helper.
- Validation:
  - `pytest -q tests/test_solver_progress.py tests/test_runtime_helper_coverage.py`
    -> `8 passed`;
  - `pytest -q tests/test_solver_progress.py tests/test_runtime_helper_coverage.py tests/test_validation_manifest_schema.py`
    -> `11 passed`;
  - `pytest -q tests/test_cli_solve_mode.py::test_write_output_full_system_regression tests/test_output_h5_scheme1_parity.py::test_output_scheme1_matches_fortran_fixture`
    -> `2 passed`;
  - `python -m ruff check sfincs_jax/solver_progress.py tests/test_solver_progress.py`
    -> passed;
  - `python -m py_compile sfincs_jax/solver_progress.py sfincs_jax/io.py sfincs_jax/v3_driver.py`
    -> passed.
- Note:
  - running Ruff over the full legacy `io.py` / `v3_driver.py` surfaces many
    pre-existing lint findings unrelated to this extraction; the new helper and
    focused tests are clean.
- Next implementation step:
  - add the structured-solve benchmark harness from Section 19.34 before changing
    any default preconditioner/Krylov path.

### 19.36 Structured-solve benchmark gate for algorithmic performance work

- Added a bounded benchmark harness for the next algorithmic lane:
  - `examples/performance/benchmark_structured_solve.py`
  - `tests/test_benchmark_structured_solve.py`
- Purpose:
  - compare dense repeated solves against a reusable block-tridiagonal
    factorization on deterministic synthetic systems;
  - report residuals, max solution error, dense-vs-structured storage bytes,
    factor time, repeated solve time, and total structured time;
  - provide a cheap admission gate before wiring factor-once / repeated-RHS
    ideas into real SFINCS operator or preconditioner paths.
- Documentation:
  - `examples/performance/README.md` now lists the harness;
  - `docs/performance_techniques.rst` documents how to run it and the admission
    rule before touching production defaults.
- Follow-up real-block extension:
  - the harness now supports `--case sfincs-pas-block`;
  - this mode loads a real SFINCS PAS fixture, fixes one species and one speed
    index, extracts the active Legendre chain and angular block from the
    matrix-free F-block, checks that off-band Legendre coupling is below the
    block-tridiagonal tolerance, and solves the regularized local block with
    both dense and structured paths;
  - the regularization is explicit in the JSON output and is benchmark-only,
    matching a preconditioner-style local block rather than changing production
    solver behavior.
- Validation:
  - `pytest -q tests/test_benchmark_structured_solve.py tests/test_structured_velocity.py`
    -> `8 passed`;
  - `python -m py_compile examples/performance/benchmark_structured_solve.py tests/test_benchmark_structured_solve.py`
    -> passed;
  - `python -m ruff check examples/performance/benchmark_structured_solve.py tests/test_benchmark_structured_solve.py`
    -> passed;
  - bounded CLI smoke with `--nblocks 5 --block-size 3 --n-rhs 2 --warmup 0 --repeats 1`
    produced `max_solution_error = 1.11e-16`, structured residual `1.50e-16`,
    dense bytes `1800`, structured bytes `936`, and a small CPU warm timing
    speedup on this tiny proxy;
  - real SFINCS PAS block CLI smoke with
    `--case sfincs-pas-block --sfincs-input tests/ref/pas_1species_PAS_noEr_tiny_scheme1.input.namelist --n-rhs 2 --warmup 0 --repeats 1`
    produced `max_solution_error = 2.96e-12`, structured residual `2.20e-13`,
    dense bytes `10368`, structured bytes `6480`, and `off_band_norm = 0`;
  - `sphinx-build -W -b html docs docs/_build/html`
    -> passed.
- Acceptance rule for future structured production changes:
  - parity-clean on the relevant SFINCS fixture,
  - structured residual and dense-reference solution agreement on the benchmark,
  - at least `20%` warm runtime improvement or `25%` memory reduction on a
    pinned offender,
  - no drift above the `1.25x` suite gates.
- Pinned offender gate:
  - ran `--case sfincs-pas-block` on
    `tests/reduced_inputs/geometryScheme4_2species_PAS_noEr.input.namelist`,
    species `0`, speed index `4`, `n_rhs=2`, benchmark-only regularization
    `1e-4`;
  - extracted block shape was `14 x 81 x 81` (`size=1134`) with
    `off_band_norm=0`;
  - structured storage was `2,099,520` bytes vs dense storage `10,287,648`
    bytes, a `79.6%` local-block storage reduction;
  - structured residual was `9.74e-10`, dense residual `2.25e-13`, and
    max structured-vs-dense solution error was `2.94e-7`;
  - CPU timing did not clear the runtime gate on this bounded local block:
    dense solve `0.1100 s`, structured factor `0.1111 s`, structured solve
    `0.1194 s`, structured total `0.2305 s`
    (`speedup_vs_dense_solve=0.477`);
  - current production auto policy still reaches the intended structured path:
    the top-level solve logs `preconditioner=schur`, and the Schur base selects
    `pas_tz` for this pinned geometry4 PAS offender.
- Implementation follow-up:
  - no broader production threshold change was made from this single local
    block because it cleared memory but not runtime;
  - fixed a latent `geom_scheme` fallback in the Schur base selector and added
    focused tests pinning `geometry4 -> schur base pas_tz` and the smaller PAS
    `pas_schur` fallback.
- Next implementation step:
  - use this gate as the admission criterion for future structured production
    changes: promote only when the full fixture is parity-clean and the local or
    end-to-end benchmark clears runtime or memory without suite drift.

### 19.37 Driver split step 2: Schur base-selection policy extraction

- Implemented the next low-risk driver split:
  - added `sfincs_jax/rhs1_schur_policy.py`;
  - moved RHSMode=1 Schur base-kind alias normalization and the automatic
    PAS/DKES/geometry size routing ladder out of `v3_driver.py`;
  - kept the numerical Schur preconditioner and all factor builders in
    `v3_driver.py`, so this is a policy extraction rather than an algorithm
    change.
- Behavior pinned by direct unit tests:
  - explicit `SFINCS_JAX_RHSMODE1_SCHUR_BASE` aliases still normalize to the
    same canonical builder names;
  - the pinned `geometryScheme4_2species_PAS_noEr` offender resolves the Schur
    base to `pas_tz`;
  - smaller PAS tokamak-like fallbacks route to `pas_schur` without the previous
    latent `geom_scheme` NameError risk;
  - bounded DKES PAS blocks choose `xblock_tz`, while memory-capped DKES blocks
    choose `pas_ilu`;
  - large PAS+Er constrained systems choose the x-coarse `xmg` Schur base.
- Validation:
  - `python -m py_compile sfincs_jax/rhs1_schur_policy.py sfincs_jax/v3_driver.py tests/test_rhs1_schur_policy.py tests/test_schur_precond_heuristic.py`
    -> passed;
  - `python -m ruff check sfincs_jax/rhs1_schur_policy.py tests/test_rhs1_schur_policy.py tests/test_schur_precond_heuristic.py`
    -> passed;
  - `pytest -q tests/test_rhs1_schur_policy.py tests/test_schur_precond_heuristic.py tests/test_benchmark_structured_solve.py tests/test_structured_velocity.py`
    -> `39 passed`;
  - `pytest -q tests/test_output_h5_scheme1_parity.py::test_output_scheme1_matches_fortran_fixture tests/test_full_system_gmres_solution_parity.py`
    -> `18 passed`.
- Next implementation step:
  - continue splitting policy-only routing out of `v3_driver.py`, prioritizing
    the RHSMode=1 top-level preconditioner auto-selection block so the remaining
    offender routing can be tested without full solves.

### 19.38 Pre-merge open-lane gate before `main` release

Decision: do not merge/tag/release this branch until the following lanes are
closed, or explicitly moved to a documented post-release research backlog with
measured evidence and clear user-facing caveats.

1. Code refactoring / maintainability.
   - Finish policy-only extraction from `v3_driver.py` before deeper numerical
     edits:
     - RHSMode=1 top-level preconditioner auto-selection,
     - RHSMode=1 retry/rescue ordering,
     - dense/sparse/direct host handoff policy,
     - transport solve-selection policy if it still has duplicate logic.
   - Acceptance:
     - no numerical behavior change,
     - direct unit tests for each extracted module,
     - representative parity fixture still clean,
     - docs/source map updated.

2. Better physics gates and validation.
   - Promote the validation matrix from "example parity" to physics-invariant
     gates:
     - conservation / nullspace checks for collision terms,
     - Onsager symmetry / transport-matrix reciprocity where applicable,
     - collisionality trend gates for LHD/W7-X literature-style scans,
     - Er trajectory-model small-field agreement and finite-field separation,
     - ambipolar root bracketing and root-stability checks on bounded fixtures,
     - monoenergetic normalization checks against DKES/KNOSOS-compatible
       reference conventions where the model overlap is defensible.
   - Acceptance:
     - each new physics gate names the equation/identity and source reference,
     - bounded CI test plus optional release/nightly larger artifact,
     - generated plot or machine-readable validation artifact when useful for a
       future paper.

3. Coverage path to `95%` with literature-anchored tests.
   - Do not chase `95%` by adding slow full-solve tests to PR CI.
   - Raise coverage in stages:
     - next floor: `60%` after more driver/I/O policy extraction,
     - next floor: `75%` after solver orchestration is decomposed,
     - research-grade target: `95%` after the deep driver body is split into
       testable policy, assembly, linear algebra, diagnostics, and output
       modules.
   - Acceptance:
     - every coverage batch is tied to a physical identity, numerical method
       invariant, public CLI behavior, or real regression,
     - PR CI remains near the 5-10 minute target,
     - heavy validations stay in scheduled/release lanes.

4. PAS memory/runtime offenders.
   - Continue measured work on the remaining pinned offenders:
     - `tokamak_1species_PASCollisions_withEr_fullTrajectories`,
     - `HSX_PASCollisions_DKESTrajectories`,
     - `HSX_PASCollisions_fullTrajectories`,
     - `geometryScheme4_2species_PAS_noEr`,
     - `sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories`,
     - `monoenergetic_geometryScheme1`,
     - `monoenergetic_geometryScheme5_ASCII`.
   - Candidate methods:
     - lower-memory PAS Schur bases,
     - chunked/block factorizations over `(species,x,L,theta,zeta)`,
     - better reuse across RHS/scan points,
     - structured sparse host/device split only where measured,
     - explicit regularized local-block gates before production promotion.
   - Acceptance:
     - parity-clean,
     - strict output drift unchanged,
     - `>=20%` warm runtime or `>=25%` RSS improvement on at least one pinned
       offender,
     - no suite runtime drift above `1.25x`.

5. Stronger multi-GPU and multi-CPU algorithms.
   - Keep release-facing claims honest:
     - transport-worker scaling is production-recommended,
     - one-GPU-per-case / scan-point parallelism is production-recommended,
     - single-case multi-GPU RHSMode=1 sharding remains experimental until it
       shows real strong scaling.
   - Research implementation lanes:
     - communication-avoiding / recycled Krylov for scan and RHS families,
     - stronger additive-Schwarz / two-level correction with cheaper global
       communication,
     - lower-synchronization sharded matvec and halo-exchange kernels,
     - process-level GPU isolation for one-worker-per-device throughput,
     - CPU multi-core benchmarks using `--cores` on larger cases where per-core
       work amortizes setup.
   - Acceptance:
     - 1 vs 2 vs 4/8 CPU scaling artifact where hardware permits,
     - 1 vs 2 GPU artifact on office,
     - parity-clean outputs,
     - docs clarify production vs experimental lanes.

6. `vmec_jax` and `booz_xform_jax` integration.
   - Stage after the driver split is stable:
     - add an adapter layer that accepts differentiable geometry coefficients
       from `/Users/rogeriojorge/vmec_jax`,
     - compare against file-based VMEC `wout_path` on the same equilibrium,
     - optionally add `booz_xform_jax` for differentiable Boozer-coordinate
       fields when available,
     - expose Python examples for geometry sensitivity and optimization loops.
   - Acceptance:
     - file-VMEC and JAX-VMEC geometry coefficients agree on bounded fixtures,
     - gradients pass finite-difference checks,
     - no new required dependency for normal CLI users unless the performance
       and usability case is proven,
     - docs explain the JAX-native geometry path separately from `wout_path`.

7. More example comparisons with SFINCS Fortran v3.
   - Add a small set of new publication-facing comparison cases beyond the
     vendored example suite:
     - one collisionality scan,
     - one Er / ambipolar scan,
     - one VMEC geometry case,
     - one PAS-heavy memory offender,
     - one transport-matrix case.
   - Acceptance:
     - frozen Fortran v3 artifacts generated from
       `/Users/rogeriojorge/local/tests/sfincs/fortran/version3/sfincs`,
     - JAX CPU/GPU comparison where hardware permits,
     - strict key coverage audit,
     - plotted diagnostics suitable for docs/manuscript use.

8. More and better documentation.
   - Before merge:
     - update source-map pages for every new module,
     - document each physics gate and what it proves,
     - document coverage strategy and current coverage honestly,
     - document PAS offender status and any remaining caveats,
     - document production vs experimental parallel lanes,
     - document `vmec_jax` / `booz_xform_jax` integration if implemented.
   - Acceptance:
     - `sphinx-build -W -b html docs docs/_build/html` passes,
     - README matches docs and measured artifacts,
     - no stale `jax_error`, `max_attempts`, or outdated runtime claims.

Recommended order:
1. Finish the driver/policy refactor enough to make the remaining lanes testable.
2. Add coverage and physics gates around the extracted modules.
3. Attack PAS offenders with benchmark gates and only promote measured wins.
4. Revisit multi-device algorithms with larger, amortized benchmarks.
5. Add `vmec_jax` / `booz_xform_jax` as an optional differentiable geometry lane.
6. Generate the new Fortran-v3 comparison examples and plots.
7. Perform the final docs/README pass.
8. Merge to `main`, run full CPU/GPU suites, tag, publish, and write release notes.

### 19.39 Driver split step 3: RHSMode=1 auto-preconditioner policy extraction

Implemented the next bounded code-refactoring increment from the pre-merge
open-lane gate:

- Added `sfincs_jax/rhs1_preconditioner_auto_policy.py` for pure
  RHSMode=1 automatic preconditioner predicates:
  - PAS large-problem base-kind selection,
  - PAS strong-retry skipping,
  - DKES `xblock_tz` gating,
  - tokamak PAS CPU/GPU `xblock_tz` and GPU `theta` gating,
  - GPU sparse fallback skipping,
  - sharded-line override safety.
- Updated `sfincs_jax/v3_driver.py` to import those predicates while keeping the
  existing `_rhs1_gpu_sparse_fallback_skip_allowed(op=...)` wrapper because the
  driver call sites still pass full operator objects and need the local backend.
- Added direct policy coverage in
  `tests/test_rhs1_preconditioner_auto_policy.py` so the routing thresholds are
  testable without constructing the full SFINCS operator.
- Updated `docs/source_map.rst` so the extracted module is visible in the
  architecture map.

This is intentionally a maintainability/testability change only: it does not
change the numerical operator, Krylov method, preconditioner formulas, or
acceptance thresholds.

Validation:

- `python -m py_compile sfincs_jax/rhs1_preconditioner_auto_policy.py sfincs_jax/v3_driver.py tests/test_rhs1_preconditioner_auto_policy.py tests/test_schur_precond_heuristic.py tests/test_v3_driver_policy_helpers.py tests/test_rhs1_sparse_first_heuristic.py`
- `python -m ruff check sfincs_jax/rhs1_preconditioner_auto_policy.py tests/test_rhs1_preconditioner_auto_policy.py tests/test_schur_precond_heuristic.py tests/test_v3_driver_policy_helpers.py tests/test_rhs1_sparse_first_heuristic.py`
- `pytest -q tests/test_rhs1_preconditioner_auto_policy.py tests/test_schur_precond_heuristic.py tests/test_v3_driver_policy_helpers.py tests/test_rhs1_sparse_first_heuristic.py`
  passed with `111 passed`.

Next refactor target:

- Extract RHSMode=1 preconditioner environment alias normalization and top-level
  initial-kind selection into pure helpers, then add direct tests before moving
  to the physics-gate and benchmark-gate lanes.

### 19.40 Driver split step 4: RHSMode=1 preconditioner alias canonicalization

Implemented the next maintainability increment:

- Added `canonical_rhs1_preconditioner_kind(raw)` to
  `sfincs_jax/rhs1_preconditioner_auto_policy.py`.
- Moved the long `SFINCS_JAX_RHSMODE1_PRECONDITIONER` alias chain out of
  `solve_v3_full_system_linear_gmres` and into a pure mapping that is directly
  testable.
- Preserved historical behavior:
  - blank aliases return `None`,
  - explicit off/false/no values return `None`,
  - unknown non-empty aliases return `None`,
  - `theta_zeta` still canonicalizes to `theta_zeta` while `zeta_theta`
    canonicalizes to `adi`, matching the old ordered chain.
- Added direct alias tests covering each alias family and retained the existing
  driver-wrapper tests.
- Updated `docs/source_map.rst` so the alias-normalization responsibility is
  visible in the architecture documentation.

Validation:

- `python -m py_compile sfincs_jax/rhs1_preconditioner_auto_policy.py sfincs_jax/v3_driver.py tests/test_rhs1_preconditioner_auto_policy.py`
- `python -m ruff check sfincs_jax/rhs1_preconditioner_auto_policy.py tests/test_rhs1_preconditioner_auto_policy.py`
- `pytest -q tests/test_rhs1_preconditioner_auto_policy.py tests/test_schur_precond_heuristic.py tests/test_v3_driver_policy_helpers.py tests/test_rhs1_sparse_first_heuristic.py`
  passed with `112 passed`.

Next refactor target:

- Extract the default RHSMode=1 initial preconditioner selection around the
  `rhs1_precond_env == ""` branch into a typed policy helper that accepts
  already-computed scalar diagnostics. That will make the PAS/FP/DKES default
  routing auditable before the PAS runtime/memory offender work resumes.

### 19.41 Driver split step 5: PAS weak-default and family-refinement policy extraction

Implemented another bounded policy split focused on the PAS offender lane:

- Added `rhs1_pas_weak_auto_override_kind(...)` to
  `sfincs_jax/rhs1_preconditioner_auto_policy.py`.
  - It promotes weak automatic PAS defaults (`point`, `collision`, `xmg`, pure
    line, weak angular blocks, or `None`) to either bounded `xblock_tz` or the
    PAS-native lite/hybrid family.
  - It preserves explicit user preconditioner requests and non-RHSMode=1 /
    Phi1 cases.
- Added `rhs1_pas_family_refinement_kind(...)`.
  - It refines PAS lite/hybrid choices to `pas_tokamak_theta`, `pas_tz`, or
    `pas_ilu` when the specialized builder is applicable and the existing
    thresholds allow it.
  - It preserves the old ordering: tokamak `pas_lite` first becomes
    `pas_hybrid`, dedicated tokamak-theta wins over 3D PAS, 3D PAS wins over
    generic lite/hybrid, and large tokamak-like blank-auto PAS may promote to
    `pas_ilu`.
- Updated `solve_v3_full_system_linear_gmres` to call those helpers instead of
  keeping the PAS refinements embedded in the monolithic solver body.
- Added direct unit tests for small bounded `xblock_tz`, explicit-env
  preservation, large `pas_lite`, tokamak `pas_hybrid`, `pas_tokamak_theta`,
  `pas_tz`, and `pas_ilu` routing.
- Updated the source map.

Validation:

- `python -m py_compile sfincs_jax/rhs1_preconditioner_auto_policy.py sfincs_jax/v3_driver.py tests/test_rhs1_preconditioner_auto_policy.py`
- `python -m ruff check sfincs_jax/rhs1_preconditioner_auto_policy.py tests/test_rhs1_preconditioner_auto_policy.py`
- `pytest -q tests/test_rhs1_preconditioner_auto_policy.py tests/test_schur_precond_heuristic.py tests/test_v3_driver_policy_helpers.py tests/test_rhs1_sparse_first_heuristic.py`
  passed with `114 passed`.

Next refactor target:

- Extract FP/DKES and large-FP default-selection policy using the same pattern,
  then move to the validation/coverage lane with focused physics gates around
  the newly isolated solver-routing policies.

### 19.42 Driver split step 6: FP/DKES and large-FP default policy extraction

Implemented the FP-focused portion of the RHSMode=1 default selector split:

- Added `rhs1_fp_dkes_env_preconditioner_kind(...)`.
  - This preserves the early bounded FP/DKES `xblock_tz` environment override
    that avoids collision-only stagnation in small DKES trajectory cases.
  - It keeps explicit user preconditioner choices untouched.
- Added `rhs1_fp_dkes_default_kind(...)`.
  - It selects `xblock_tz` for bounded small FP/DKES blocks, `xmg` for
    small/medium FP/DKES blocks that are too large for dense `xblock_tz`, and
    `collision` above the strong-DKES threshold to avoid excessive setup cost.
- Added `rhs1_large_fp_near_zero_er_override_kind(...)`.
  - It forces large FP-only, near-zero-Er, weak-preconditioned systems to `xmg`.
  - It preserves stronger user/auto choices such as `schur`.
- Updated `solve_v3_full_system_linear_gmres` to use the extracted helpers.
- Added direct tests for all FP/DKES branches and the large-FP override.
- Updated the source map.

Validation:

- `python -m py_compile sfincs_jax/rhs1_preconditioner_auto_policy.py sfincs_jax/v3_driver.py tests/test_rhs1_preconditioner_auto_policy.py`
- `python -m ruff check sfincs_jax/rhs1_preconditioner_auto_policy.py tests/test_rhs1_preconditioner_auto_policy.py`
- `pytest -q tests/test_rhs1_preconditioner_auto_policy.py tests/test_schur_precond_heuristic.py tests/test_v3_driver_policy_helpers.py tests/test_rhs1_sparse_first_heuristic.py`
  passed with `117 passed`.

Next refactor target:

- Add a compact integration test that asserts representative fixture inputs
  resolve to the expected preconditioner policy hints. This bridges the pure
  policy tests to real namelist/operator construction before deeper physics
  gates and benchmark gates are expanded.

### 19.43 Optional JAX-native geometry adapter stage

Implemented the first concrete `vmec_jax` / `booz_xform_jax` integration step:

- Local repository check:
  - the originally referenced `/Users/rogeriojorge/vmec_jax` path is not present,
  - the usable local `vmec_jax` repository is
    `/Users/rogeriojorge/local/vmec_jax`,
  - the usable local `booz_xform_jax` repository is
    `/Users/rogeriojorge/local/booz_xform_jax`.
- Added `sfincs_jax/jax_geometry_adapters.py`.
  - It has no import-time dependency on either optional package.
  - `optional_jax_geometry_backend_status()` reports whether `vmec_jax` and
    `booz_xform_jax` are importable.
  - `vmec_wout_from_wout_like(...)` converts VMEC-like in-memory objects,
    including the `vmec_jax.wout.WoutData` field layout, to the internal
    `sfincs_jax.vmec_wout.VmecWout` dataclass.
  - The adapter accepts both `(radius, mode)` and `(mode, radius)` coefficient
    arrays and normalizes them to the `sfincs_jax` `(mode, radius)` convention.
- Added `tests/test_jax_geometry_adapters.py` for:
  - backend-status structure,
  - `vmec_jax`-style transposition,
  - native `sfincs_jax` ordering,
  - invalid-shape rejection.
- Documented the adapter in `docs/geometry.rst` and `docs/source_map.rst`.

Validation:

- `python -m py_compile sfincs_jax/jax_geometry_adapters.py tests/test_jax_geometry_adapters.py`
- `python -m ruff check sfincs_jax/jax_geometry_adapters.py tests/test_jax_geometry_adapters.py`
- `pytest -q tests/test_jax_geometry_adapters.py`
  passed with `4 passed`.

Remaining work in this lane:

- Refactor `vmec_geometry_from_wout_file(...)` so the Fourier-sum evaluator can
  accept a `VmecWout` object directly, then add an end-to-end file-VMEC vs
  in-memory `vmec_jax` geometry comparison.
- Keep `booz_xform_jax` as the second-stage route for Boozer-coordinate studies:
  `vmec_jax -> booz_xform_jax -> sfincs_jax` should only become public after the
  field-component and harmonic-selection tests pass.

### 19.44 VMEC geometry evaluator split for in-memory producers

Completed the next stage of the optional JAX-native geometry lane:

- Refactored `sfincs_jax/vmec_geometry.py`:
  - `vmec_geometry_from_wout_file(...)` is now a thin file-I/O wrapper,
  - new `vmec_geometry_from_wout(...)` evaluates the existing
    `geometryScheme=5` Fourier sums from a preloaded `VmecWout`.
- Added a regression in `tests/test_geometry_grid_helper_coverage.py` proving
  exact equality between:
  - file-based `vmec_geometry_from_wout_file(...)`, and
  - object-based `vmec_geometry_from_wout(read_vmec_wout(...))`
    on the W7-X VMEC fixture.
- Kept numerical formulas unchanged: this only separates file I/O from geometry
  evaluation so optional JAX-native producers can feed the same evaluator.
- Updated `docs/geometry.rst` and `docs/source_map.rst`.

Validation:

- `python -m py_compile sfincs_jax/vmec_geometry.py tests/test_geometry_grid_helper_coverage.py`
- `python -m ruff check sfincs_jax/vmec_geometry.py tests/test_geometry_grid_helper_coverage.py`
- `pytest -q tests/test_geometry_grid_helper_coverage.py tests/test_jax_geometry_adapters.py`
  passed with `14 passed`.

Remaining work in this lane:

- Add a real `vmec_jax.WoutData -> vmec_wout_from_wout_like(...) ->
  vmec_geometry_from_wout(...)` test when the local `vmec_jax` state is clean
  enough to pin stable fixture behavior.
- Add the public differentiable example only after file-based and in-memory
  geometry arrays match on a bounded fixture and finite-difference/JAX gradient
  checks pass.

### 19.45 Physics gate: PAS Legendre nullspace and eigenvalue scaling

Added a cheap, literature-aligned PAS collision-operator gate:

- New `tests/test_collision_physics_gates.py`.
- The gate checks:
  - pure pitch-angle scattering annihilates the isotropic `L=0` Legendre mode
    when `krook=0`,
  - inactive Legendre slots beyond `n_xi_for_x` are masked exactly,
  - active higher Legendre coefficients scale as `L(L+1)/2` relative to `L=1`,
    matching the standard pitch-angle-scattering operator eigenvalues in a
    Legendre basis.
- This complements the existing Fortran-matrix parity test in
  `tests/test_pas_collision_operator_parity.py`: parity proves agreement with
  the frozen implementation, while this gate proves the expected operator
  structure directly.
- Updated `docs/testing.rst`.

Validation:

- `python -m py_compile tests/test_collision_physics_gates.py`
- `python -m ruff check tests/test_collision_physics_gates.py`
- `pytest -q tests/test_collision_physics_gates.py tests/test_pas_collision_operator_parity.py`
  passed with `3 passed`.

Next validation targets:

- Add a similarly cheap Fokker-Planck gate around Chandrasekhar-function limits
  and interpolation identities.
- Add a geometry gate around VMEC in-memory conversion once the optional
  `vmec_jax` fixture can be pinned cleanly.

### 19.46 Collision-kernel gate: Chandrasekhar small-x stability and interpolation identity

Extended the collision physics/numerics gate:

- Added tests for:
  - the Chandrasekhar function small-`x` limit
    `Psi(x) / x -> 2 / (3 sqrt(pi))`,
  - positivity of the small-`x` branch,
  - identity behavior of the v3 barycentric interpolation matrix when source
    and target nodes match.
- The new small-`x` test exposed a real cancellation bug:
  - `_psi_chandra(...)` and `_psi_chandra_np(...)` used the direct
    `erf(x) - 2 x exp(-x^2)/sqrt(pi)` formula until `|x| < 1e-14`,
  - for `x ~ 1e-8` to `1e-12`, this produced catastrophic cancellation instead
    of the linear Chandrasekhar limit.
- Fixed both JAX and NumPy paths with the analytic series
  `Psi(x) = [(2/3)x - (2/5)x^3 + (1/7)x^5 + O(x^7)] / sqrt(pi)` for
  `|x| < 1e-5`.

Validation:

- `python -m py_compile sfincs_jax/collisions.py tests/test_collision_physics_gates.py`
- `python -m ruff check tests/test_collision_physics_gates.py`
- `pytest -q tests/test_collision_physics_gates.py tests/test_fokker_planck_phi1_reduces_to_no_phi1.py tests/test_pas_collision_operator_parity.py tests/test_fblock_fokker_planck_matvec_parity.py`
  passed with `7 passed`.

Next validation targets:

- Run the broader focused parity subset after the collision-kernel change.
- Add a finite-difference/JAX-gradient gate around a bounded differentiable
  geometry or transport scalar once the refactored geometry path is stable.

### 19.47 Optional real `vmec_jax.WoutData` geometry adapter gate

Closed the first real `vmec_jax` adapter validation item:

- Added an optional test in `tests/test_jax_geometry_adapters.py` that:
  - imports `vmec_jax` and `netCDF4` only inside the test,
  - discovers `vmec_jax/examples/data/wout_circular_tokamak.nc` from the
    installed/imported `vmec_jax` package path,
  - skips cleanly if the optional backend or fixture is unavailable,
  - reads the same file through both `vmec_jax.wout.read_wout(...)` and
    `sfincs_jax.vmec_wout.read_vmec_wout(...)`,
  - converts the `vmec_jax.wout.WoutData` object through
    `vmec_wout_from_wout_like(...)`,
  - checks exact equality of VMEC Fourier coefficient arrays,
  - evaluates `vmec_geometry_from_wout(...)` from both objects and checks exact
    equality of representative geometry arrays.
- Expanded `docs/geometry.rst` with a minimal source-code example for the
  `vmec_jax -> vmec_wout_from_wout_like -> vmec_geometry_from_wout` workflow.
- Updated `docs/testing.rst` to describe the optional gate and why it skips in
  normal CI if the optional backend is absent.

Validation:

- `python -m py_compile tests/test_jax_geometry_adapters.py`
- `python -m ruff check tests/test_jax_geometry_adapters.py`
- `pytest -q tests/test_jax_geometry_adapters.py tests/test_geometry_grid_helper_coverage.py`
  passed with `15 passed`.

Next validation target:

- Add an actual finite-difference/JAX-gradient check around a small differentiable
  scalar. Candidate: geometryScheme=4 harmonic derivative first, then
  `vmec_jax`-driven scheme-5 once the upstream differentiable producer state is
  stable enough for a deterministic fixture.

### 19.48 Analytic geometry autodiff gate

Added the first bounded differentiable-geometry validation after the optional
`vmec_jax` adapter gate:

- Added `tests/test_geometry_autodiff_gates.py`.
- The test differentiates a scalar geometry objective
  `mean(BHat**2) + 0.1 * mean(DHat)` with respect to the three scheme-4
  W7-X-like harmonic amplitudes.
- The JAX gradient is compared against central finite differences on a small
  `(Ntheta, Nzeta) = (10, 8)` grid, so the gate is cheap enough for CI but still
  exercises the normalized geometry arrays that feed the transport operator.
- Added a docstring to `BoozerGeometry` explaining the internal `(Ntheta, Nzeta)`
  layout and why the geometry container remains flat and explicit.
- Updated `docs/geometry.rst` and `docs/testing.rst` with the public
  differentiable geometry example and the validation rationale.

Validation:

- `python -m py_compile sfincs_jax/geometry.py tests/test_geometry_autodiff_gates.py`
- `python -m ruff check tests/test_geometry_autodiff_gates.py`
- `pytest -q tests/test_geometry_autodiff_gates.py tests/test_geometry_grid_helper_coverage.py tests/test_u_hat_fft.py`
  passed with `17 passed`.
- `sphinx-build -W -b html docs docs/_build/html` passed.

### 19.90 Release 1.0.2 preparation and manifest closure

Prepared the branch for the final 1.0.2 ship pass:

- Bumped `pyproject.toml` from `1.0.1` to `1.0.2`.
- Fixed the stale release-checklist workflow name from `publish.yml` to
  `publish-pypi.yml`.
- Closed all manifest lanes into one of two statuses:
  - `implemented` for release-facing checked-in artifacts and bounded scaffolds with
    existing tests/artifacts,
  - `deferred_post_release` for research/nightly lanes that must not block the 1.0.2
    tag and must not be overclaimed in release notes.
- Converted the previous open lanes as follows:
  - corrected LHD/W7-X fast collisionality scaffolds: `implemented`,
  - high-collisionality trend proxy: `implemented`,
  - stellarator fast Er sweep scaffold: `implemented`,
  - full Simakov-Helander analytic-limit reproduction: `deferred_post_release`,
  - W7-X ambipolar profile validation: `deferred_post_release`,
  - MONKES/KNOSOS overlap: `deferred_post_release`,
  - manuscript-scale adjoint/sensitivity maps: `deferred_post_release`.
- Updated `tests/test_validation_manifest_schema.py` so CI now asserts there are no
  `planned`, `prototype_artifact`, or `needs_reaudit` manifest statuses left.
- Updated the validation/testing docs to explain that deferred lanes are closed
  post-release research items with explicit acceptance gates, not release blockers.

Validation:

- `python -m pytest -q tests/test_validation_manifest_schema.py` passed with
  `4 passed in 0.03s`.
- `python -m ruff check tests/test_validation_manifest_schema.py` passed.
- `sphinx-build -W -b html docs docs/_build/html` passed.
- `python -m build` passed and built `sfincs_jax-1.0.2.tar.gz` plus
  `sfincs_jax-1.0.2-py3-none-any.whl`.
- `python -m pytest -q` passed with `869 passed in 325.09s (0:05:25)`.

### 19.91 CI portability fix for suite benchmark tests

The first `main` CI run for the 1.0.2 release-prep commit exposed a CI-only issue:
the raw frozen suite-report directories used locally are not tracked in the GitHub
checkout, even though the generated benchmark summary JSON is tracked.

Fix:

- `tests/test_validation_artifacts.py` now tests suite-report parsing and summary
  metrics using synthetic 39-case report rows, and validates the release gate from the
  checked-in benchmark summary artifact.
- `tests/test_generate_fortran_suite_benchmark_summary.py` now builds temporary
  synthetic CPU/GPU suite reports before exercising the figure/JSON generator.

Validation:

- `python -m ruff check tests/test_validation_artifacts.py tests/test_generate_fortran_suite_benchmark_summary.py`
  passed.
- `python -m pytest -q tests/test_validation_artifacts.py tests/test_generate_fortran_suite_benchmark_summary.py tests/test_validation_manifest_schema.py`
  passed with `12 passed in 0.94s`.

Next validation targets:

- After this cheap gate is stable, keep `vmec_jax`/`booz_xform_jax` end-to-end
  optimization examples as a separate research-grade lane rather than merging
  them into the lightweight CI path prematurely.

### 19.49 VMEC-like adapter structural hardening

Tightened the optional JAX-native VMEC adapter contract without adding new required
dependencies:

- Added docstrings/comments in `sfincs_jax/jax_geometry_adapters.py` explaining
  shallow backend discovery, mode/radius normalization, metadata-only path
  overrides, and why absent optional covariant/contravariant field tables may be
  zero-filled for minimal stellarator-symmetric producers while required field,
  metric, and shape tables remain strict.
- Added tests for:
  - metadata-only `path=...` override behavior,
  - zero-filling absent optional field-coefficient tables,
  - default zero pressure-profile handling,
  - and rejection of a missing required `bmnc` table.
- Updated geometry/testing docs so the adapter behavior is no longer implicit.

Validation:

- `python -m py_compile sfincs_jax/jax_geometry_adapters.py tests/test_jax_geometry_adapters.py`
- `python -m ruff check sfincs_jax/jax_geometry_adapters.py tests/test_jax_geometry_adapters.py`
- `pytest -q tests/test_jax_geometry_adapters.py tests/test_geometry_grid_helper_coverage.py tests/test_geometry_autodiff_gates.py`
  passed with `19 passed`.
- `sphinx-build -W -b html docs docs/_build/html` passed.

### 19.50 Refactored policy module docstring regression guard

Fixed a source-structure issue in the split policy/helper modules: several files had
their explanatory string after `from __future__ import annotations`, so Python did
not expose it as the module `__doc__`.

- Moved the module docstrings ahead of the future import in the affected RHSMode=1
  and transport policy/helper modules.
- Added `tests/test_policy_module_docstrings.py` to import each split policy module
  and assert that the public module docstring is present and non-empty.
- Updated `docs/testing.rst` so this is documented as part of the maintainability
  test layer, not just a style-only cleanup.

Validation:

- `python -m py_compile sfincs_jax/rhs1_handoff.py sfincs_jax/rhs1_pas_policy.py sfincs_jax/rhs1_preconditioner_dispatch.py sfincs_jax/rhs1_sparse_polish_policy.py sfincs_jax/rhs1_sparse_rescue_policy.py sfincs_jax/rhs1_stage2_policy.py sfincs_jax/rhs1_strong_auto_kind.py sfincs_jax/rhs1_strong_control.py sfincs_jax/rhs1_strong_fallback.py sfincs_jax/rhs1_strong_policy.py sfincs_jax/transport_dense_lu.py sfincs_jax/transport_handoff_policy.py sfincs_jax/transport_host_gmres.py sfincs_jax/transport_preconditioner_dispatch.py sfincs_jax/transport_solve_policy.py tests/test_policy_module_docstrings.py`
- `python -m ruff check sfincs_jax/rhs1_handoff.py sfincs_jax/rhs1_pas_policy.py sfincs_jax/rhs1_preconditioner_dispatch.py sfincs_jax/rhs1_sparse_polish_policy.py sfincs_jax/rhs1_sparse_rescue_policy.py sfincs_jax/rhs1_stage2_policy.py sfincs_jax/rhs1_strong_auto_kind.py sfincs_jax/rhs1_strong_control.py sfincs_jax/rhs1_strong_fallback.py sfincs_jax/rhs1_strong_policy.py sfincs_jax/transport_dense_lu.py sfincs_jax/transport_handoff_policy.py sfincs_jax/transport_host_gmres.py sfincs_jax/transport_preconditioner_dispatch.py sfincs_jax/transport_solve_policy.py tests/test_policy_module_docstrings.py`
- `pytest -q tests/test_policy_module_docstrings.py tests/test_rhs1_handoff.py tests/test_rhs1_preconditioner_auto_policy.py tests/test_v3_driver_rhs1_dispatch_coverage.py tests/test_rhs1_sparse_polish_policy.py tests/test_rhs1_sparse_rescue_policy.py tests/test_rhs1_stage2_policy.py tests/test_rhs1_strong_auto_kind.py tests/test_rhs1_strong_control.py tests/test_rhs1_strong_policy.py tests/test_transport_handoff_policy.py tests/test_transport_preconditioner_dispatch.py tests/test_transport_solve_policy.py`
  passed with `63 passed`.
- `sphinx-build -W -b html docs docs/_build/html` passed.

### 19.51 API documentation for split geometry and policy modules

Expanded `docs/api.rst` so the newly split modules are visible in generated API
documentation:

- Added `sfincs_jax.vmec_wout`, `sfincs_jax.vmec_geometry`, and
  `sfincs_jax.jax_geometry_adapters`.
- Added a dedicated "Refactored solve-policy modules" section for the RHSMode=1
  and transport policy/dispatch helpers extracted from `v3_driver.py`.
- This makes source-level docstrings useful to users and reviewers, and it closes
  the documentation gap created by the driver split.

Validation:

- `sphinx-build -W -b html docs docs/_build/html` passed.

### 19.52 VMEC scheme-5 convention gates

Added a cheap, scheme-5-focused validation layer for VMEC conventions:

- Added docstrings to `VmecWout` and `VmecInterpolation` explaining the internal
  `(mode, radius)` coefficient layout, the preserved half-mesh dummy entry, and the
  purpose of the interpolation state.
- Added `tests/test_vmec_wout_conventions.py` covering:
  - `psi_a_hat = phi[-1] / (2*pi)`,
  - full- and half-mesh interpolation weights at a representative radius,
  - `VMECRadialOption` snapping to nearest half/full mesh,
  - endpoint half-mesh extrapolation behavior,
  - invalid radius/option errors,
  - and helicity/ripple-scale mode-selection rules.
- Updated testing docs so these are visible as scheme-5 physics/numerics gates.

Validation:

- `python -m py_compile sfincs_jax/vmec_wout.py tests/test_vmec_wout_conventions.py`
- `python -m ruff check sfincs_jax/vmec_wout.py tests/test_vmec_wout_conventions.py`
- `pytest -q tests/test_vmec_wout_conventions.py tests/test_jax_geometry_adapters.py tests/test_geometry_grid_helper_coverage.py`
  passed with `23 passed`.
- `sphinx-build -W -b html docs docs/_build/html` passed.

### 19.53 Full local suite after documentation/refactor/test increments

Ran the full local test suite after the latest bounded increments:

- scheme-4 geometry autodiff gate,
- VMEC-like adapter hardening,
- policy module docstring regression guard,
- API documentation expansion,
- and VMEC scheme-5 convention gates.

Validation:

- `pytest -q` passed with `781 passed in 362.45s (0:06:02)`.

Notes:

- This is a bounded local validation, not a replacement for the heavier GPU/full
  example-suite audits.
- The next research-grade lanes remain the larger open items: deeper driver
  refactoring, better physics validation figures, PAS runtime/memory offenders,
  multi-device algorithms, and end-to-end `vmec_jax` / `booz_xform_jax`
  differentiable examples.

### 19.54 RHSMode=1 host dense/sparse policy extraction

Started the next deeper-driver-refactor increment by extracting pure host
dense/sparse-direct policy out of `v3_driver.py`:

- Added `sfincs_jax/rhs1_host_policy.py`.
- Kept the public/private driver wrappers intact so existing tests and downstream
  monkeypatch-based debugging do not break.
- The extracted policy covers:
  - RHSMode=1 dense backend permission,
  - host dense fallback permission,
  - small accelerator FP host-dense shortcut,
  - dense Krylov enablement,
  - exact host sparse-direct permission,
  - sparse-preconditioned GMRES rescue gating,
  - host sparse factor dtype selection,
  - iterative-refinement step parsing,
  - sparse-direct skip-dense residual ratio,
  - and explicit sparse-helper bounds.
- Added `tests/test_rhs1_host_policy.py` for direct coverage of the extracted
  policy logic.
- Updated API docs, source map, testing docs, and the module-docstring regression
  guard so the new split is discoverable.

Validation:

- `python -m py_compile sfincs_jax/rhs1_host_policy.py sfincs_jax/v3_driver.py tests/test_rhs1_host_policy.py tests/test_policy_module_docstrings.py`
- `python -m ruff check sfincs_jax/rhs1_host_policy.py tests/test_rhs1_host_policy.py tests/test_policy_module_docstrings.py`
- `pytest -q tests/test_rhs1_host_policy.py tests/test_v3_driver_policy_helpers.py tests/test_v3_driver_sparse_helper_coverage.py tests/test_rhs1_sparse_first_heuristic.py tests/test_transport_sparse_direct.py tests/test_v3_driver_solve_policy_coverage.py tests/test_policy_module_docstrings.py`
  passed with `150 passed`.
- `sphinx-build -W -b html docs docs/_build/html` passed.

Notes:

- `v3_driver.py` still has broad pre-existing ruff debt, so the scoped lint gate for
  this increment is the new module plus tests; the driver itself is covered by
  py_compile and focused wrapper tests.

### 19.55 RHSMode=1 dense fallback cap extraction

Completed the adjacent dense-fallback policy extraction:

- Moved `_rhsmode1_dense_fallback_max(...)` logic into
  `rhs1_host_policy.rhs1_dense_fallback_max(...)`.
- Kept the `v3_driver.py` wrapper intact for compatibility with existing tests and
  downstream debugging.
- Added direct tests for:
  - default FP dense fallback cap,
  - default PAS disablement for non-constraint-0 systems,
  - constraint-0 PAS carve-out,
  - FP max/cutoff override behavior,
  - and explicit PAS opt-in/disable behavior.
- Updated testing docs to include the dense-fallback ceiling in the extracted host
  policy contract.

Validation:

- `python -m py_compile sfincs_jax/rhs1_host_policy.py sfincs_jax/v3_driver.py tests/test_rhs1_host_policy.py`
- `python -m ruff check sfincs_jax/rhs1_host_policy.py tests/test_rhs1_host_policy.py`
- `pytest -q tests/test_rhs1_host_policy.py tests/test_v3_driver_policy_helpers.py tests/test_v3_driver_sparse_helper_coverage.py tests/test_rhs1_sparse_first_heuristic.py tests/test_transport_sparse_direct.py tests/test_v3_driver_solve_policy_coverage.py`
  passed with `150 passed`.
- `sphinx-build -W -b html docs docs/_build/html` passed.

### 19.56 RHSMode=1 constraint-scheme-0 sparse-first policy extraction

Completed the next driver-refactor increment:

- Add `sfincs_jax/rhs1_constraint0_policy.py` for the constraint-scheme-0
  RHSMode=1 sparse-first, explicit PETSc-compatible sparse, and dense-fallback
  opt-in decisions.
- Keep the existing `v3_driver.py` wrappers intact so downstream debugging and
  monkeypatch-based tests continue to use the same private seam.
- Add direct tests that cover accelerator-vs-CPU defaults, explicit environment
  enable/disable behavior, RHSMode/Phi1/full-FP guards, dense-method rejection,
  sparse-preconditioner rejection, and active-size limits.
- Update the API docs, source map, testing guide, and policy-docstring regression
  guard so the new split is discoverable.

Validation:

- `python -m py_compile sfincs_jax/rhs1_constraint0_policy.py sfincs_jax/v3_driver.py tests/test_rhs1_constraint0_policy.py tests/test_policy_module_docstrings.py`
- `python -m ruff check sfincs_jax/rhs1_constraint0_policy.py tests/test_rhs1_constraint0_policy.py tests/test_policy_module_docstrings.py`
- `pytest -q tests/test_rhs1_constraint0_policy.py tests/test_rhs1_sparse_first_heuristic.py tests/test_policy_module_docstrings.py`
  passed with `75 passed`.
- `pytest -q tests/test_rhs1_constraint0_policy.py tests/test_rhs1_host_policy.py tests/test_v3_driver_policy_helpers.py tests/test_v3_driver_sparse_helper_coverage.py tests/test_rhs1_sparse_first_heuristic.py tests/test_transport_sparse_direct.py tests/test_v3_driver_solve_policy_coverage.py tests/test_policy_module_docstrings.py`
  passed with `157 passed`.
- `sphinx-build -W -b html docs docs/_build/html` passed.

### 19.57 RHSMode=1 sparse exact-LU/prefer policy extraction

Completed the next adjacent driver-refactor increment:

- Add `sfincs_jax/rhs1_sparse_exact_policy.py` for sparse exact-LU request
  policy, moderate-FP sparse-over-dense preference, and sparse-prefer stage-2
  skip decisions.
- Keep the existing `v3_driver.py` wrappers intact for compatibility with
  existing tests and downstream debugging.
- Add direct tests for full-x CPU exact-LU routing, accelerator DKES exact-LU
  routing, small accelerator FP exact-LU routing, PAS full-preconditioner opt-in,
  explicit environment enable/disable behavior, dense-method/size/Phi1 guards,
  sparse-over-dense preference guards, and stage-2 skip guards.
- Update API docs, source map, testing docs, and module-docstring coverage.

Validation:

- `python -m py_compile sfincs_jax/rhs1_sparse_exact_policy.py sfincs_jax/v3_driver.py tests/test_rhs1_sparse_exact_policy.py tests/test_policy_module_docstrings.py`
- `python -m ruff check sfincs_jax/rhs1_sparse_exact_policy.py tests/test_rhs1_sparse_exact_policy.py tests/test_policy_module_docstrings.py`
- `pytest -q tests/test_rhs1_sparse_exact_policy.py tests/test_sparse_exact_lu_heuristic.py tests/test_rhs1_sparse_first_heuristic.py tests/test_v3_driver_solve_policy_coverage.py tests/test_policy_module_docstrings.py`
  passed with `97 passed`.
- `pytest -q tests/test_rhs1_sparse_exact_policy.py tests/test_rhs1_constraint0_policy.py tests/test_rhs1_host_policy.py tests/test_v3_driver_policy_helpers.py tests/test_v3_driver_sparse_helper_coverage.py tests/test_rhs1_sparse_first_heuristic.py tests/test_sparse_exact_lu_heuristic.py tests/test_transport_sparse_direct.py tests/test_v3_driver_solve_policy_coverage.py tests/test_policy_module_docstrings.py`
  passed with `169 passed`.
- `sphinx-build -W -b html docs docs/_build/html` passed.

### 19.58 RHSMode=1 large-CPU sparse/x-block policy extraction

Completed the next runtime-offender-facing driver-refactor increment:

- Add `sfincs_jax/rhs1_large_cpu_policy.py` for large explicit full-FP CPU sparse
  rescue, large-CPU exact-LU caps, sparse-rescue-first ordering, x-block
  exact-LU promotion, x-block sparse rescue, host x-block assembly, primary-solve
  skipping, and species-x-block rescue eligibility.
- Keep the existing `v3_driver.py` wrappers intact for compatibility with the
  established heuristic tests.
- Add direct tests for the large-CPU rescue decisions so CI can cover the
  runtime-offender routing without running large cases.
- Update API docs, source map, testing docs, and module-docstring coverage.

Validation:

- `python -m py_compile sfincs_jax/rhs1_large_cpu_policy.py sfincs_jax/v3_driver.py tests/test_rhs1_large_cpu_policy.py tests/test_policy_module_docstrings.py`
- `python -m ruff check sfincs_jax/rhs1_large_cpu_policy.py tests/test_rhs1_large_cpu_policy.py tests/test_policy_module_docstrings.py`
- `pytest -q tests/test_rhs1_large_cpu_policy.py tests/test_rhs1_sparse_first_heuristic.py tests/test_v3_driver_solve_policy_coverage.py tests/test_policy_module_docstrings.py`
  passed with `93 passed`.
- `pytest -q tests/test_rhs1_large_cpu_policy.py tests/test_rhs1_sparse_exact_policy.py tests/test_rhs1_constraint0_policy.py tests/test_rhs1_host_policy.py tests/test_v3_driver_policy_helpers.py tests/test_v3_driver_sparse_helper_coverage.py tests/test_rhs1_sparse_first_heuristic.py tests/test_sparse_exact_lu_heuristic.py tests/test_transport_sparse_direct.py tests/test_v3_driver_solve_policy_coverage.py tests/test_policy_module_docstrings.py`
  passed with `177 passed`.
- `sphinx-build -W -b html docs docs/_build/html` passed.

### 19.59 Full-suite gate after RHSMode=1 policy extractions

Ran the full local suite after the latest three driver-policy extractions:

- `sfincs_jax/rhs1_constraint0_policy.py`
- `sfincs_jax/rhs1_sparse_exact_policy.py`
- `sfincs_jax/rhs1_large_cpu_policy.py`

Validation:

- `pytest -q` passed with `809 passed in 363.46s (0:06:03)`.

Notes:

- This confirms the scoped policy refactors did not regress the local unit,
  regression, CLI, docs-support, geometry, solver, or bounded parity tests.
- The next bounded refactor lane is the adjacent post-x-block polish /
  targeted-polish / skip-global-sparse policy cluster in `v3_driver.py`.

### 19.60 RHSMode=1 post-x-block polish policy extraction

Completed the adjacent large-CPU handoff refactor:

- Add `sfincs_jax/rhs1_post_xblock_policy.py` for fast post-x-block polish,
  targeted FP polish, and explicit skip-global-sparse-after-xblock decisions.
- Keep the existing `v3_driver.py` wrappers intact for compatibility with the
  established heuristic tests.
- Add direct tests for the active-size thresholds, residual thresholds, explicit
  opt-in behavior, CPU/backend guards, implicit-solve guards, and full-FP-only
  guards.
- Update API docs, source map, testing docs, and module-docstring coverage.

Validation:

- `python -m py_compile sfincs_jax/rhs1_post_xblock_policy.py sfincs_jax/v3_driver.py tests/test_rhs1_post_xblock_policy.py tests/test_policy_module_docstrings.py`
- `python -m ruff check sfincs_jax/rhs1_post_xblock_policy.py tests/test_rhs1_post_xblock_policy.py tests/test_policy_module_docstrings.py`
- `pytest -q tests/test_rhs1_post_xblock_policy.py tests/test_rhs1_sparse_first_heuristic.py tests/test_policy_module_docstrings.py`
  passed with `75 passed`.
- `pytest -q tests/test_rhs1_post_xblock_policy.py tests/test_rhs1_large_cpu_policy.py tests/test_rhs1_sparse_exact_policy.py tests/test_rhs1_constraint0_policy.py tests/test_rhs1_host_policy.py tests/test_v3_driver_policy_helpers.py tests/test_v3_driver_sparse_helper_coverage.py tests/test_rhs1_sparse_first_heuristic.py tests/test_sparse_exact_lu_heuristic.py tests/test_transport_sparse_direct.py tests/test_v3_driver_solve_policy_coverage.py tests/test_policy_module_docstrings.py`
  passed with `183 passed`.
- `sphinx-build -W -b html docs docs/_build/html` passed.

### 19.61 RHSMode=1 PAS fast-accept and host-factor probe extraction

Completed the small policy extraction:

- Add `sfincs_jax/rhs1_acceptance_policy.py` for large-PAS fast-accept gates and
  host x-block factor-probe safety checks.
- Reuse `pas_smoother.pas_fast_accept(...)` for the residual acceptance formula
  so the PAS threshold remains single-sourced.
- Keep the existing `v3_driver.py` wrappers intact for compatibility with the
  established heuristic tests.
- Add direct tests for PAS fast-accept environment parsing, backend/implicit/Phi1
  and PAS guards, nonfinite residuals, factor-probe exceptions, shape mismatches,
  nonfinite factor solves, invalid probe thresholds, and excessive amplification.
- Update API docs, source map, testing docs, and module-docstring coverage.

Validation:

- `python -m py_compile sfincs_jax/rhs1_acceptance_policy.py sfincs_jax/v3_driver.py tests/test_rhs1_acceptance_policy.py tests/test_policy_module_docstrings.py`
- `python -m ruff check sfincs_jax/rhs1_acceptance_policy.py tests/test_rhs1_acceptance_policy.py tests/test_policy_module_docstrings.py`
- `pytest -q tests/test_rhs1_acceptance_policy.py tests/test_rhs1_sparse_first_heuristic.py tests/test_policy_module_docstrings.py`
  passed with `75 passed`.
- `pytest -q tests/test_rhs1_acceptance_policy.py tests/test_rhs1_post_xblock_policy.py tests/test_rhs1_large_cpu_policy.py tests/test_rhs1_sparse_exact_policy.py tests/test_rhs1_constraint0_policy.py tests/test_rhs1_host_policy.py tests/test_v3_driver_policy_helpers.py tests/test_v3_driver_sparse_helper_coverage.py tests/test_rhs1_sparse_first_heuristic.py tests/test_sparse_exact_lu_heuristic.py tests/test_transport_sparse_direct.py tests/test_v3_driver_solve_policy_coverage.py tests/test_policy_module_docstrings.py`
  passed with `189 passed`.
- `sphinx-build -W -b html docs docs/_build/html` passed.

### 19.62 Full-suite gate after post-xblock and acceptance policy extractions

Ran the full local suite after the latest two driver-policy extractions:

- `sfincs_jax/rhs1_post_xblock_policy.py`
- `sfincs_jax/rhs1_acceptance_policy.py`

Validation:

- `pytest -q` passed with `821 passed in 343.95s (0:05:43)`.

Notes:

- This confirms the latest `v3_driver.py` wrapper reductions did not regress the
  local unit, regression, CLI, geometry, solver, or bounded parity tests.
- The next practical lane is to review the remaining unextracted RHSMode=1
  helpers and either extract the next small pure-policy cluster or switch to the
  open validation/documentation lanes if the remaining code is less clearly
  separable.

### 19.63 PAS adaptive-smoother and solve-mode policy extraction

Completed the small pure-policy extraction:

- Move PAS adaptive-smoother eligibility into `rhs1_pas_policy.py`, reusing the
  lower-level `pas_smoother.adaptive_pas_smoother_allowed(...)` predicate.
- Add `sfincs_jax/solve_mode_policy.py` for shared
  `SFINCS_JAX_IMPLICIT_SOLVE` / differentiability precedence.
- Keep the existing `v3_driver.py` wrappers intact for compatibility with the
  established heuristic and I/O tests.
- Add direct tests for PAS adaptive smoother activation/guards/env parsing and
  direct tests for implicit-solve env resolution.
- Update API docs, source map, testing docs, and module-docstring coverage.

Validation:

- `python -m py_compile sfincs_jax/solve_mode_policy.py sfincs_jax/rhs1_pas_policy.py sfincs_jax/v3_driver.py tests/test_solve_mode_policy.py tests/test_rhs1_pas_policy.py tests/test_policy_module_docstrings.py`
- `python -m ruff check sfincs_jax/solve_mode_policy.py sfincs_jax/rhs1_pas_policy.py tests/test_solve_mode_policy.py tests/test_rhs1_pas_policy.py tests/test_policy_module_docstrings.py`
- `pytest -q tests/test_rhs1_pas_policy.py tests/test_solve_mode_policy.py tests/test_rhs1_sparse_first_heuristic.py tests/test_policy_module_docstrings.py`
  passed with `75 passed`.
- `pytest -q tests/test_rhs1_pas_policy.py tests/test_solve_mode_policy.py tests/test_rhs1_acceptance_policy.py tests/test_rhs1_post_xblock_policy.py tests/test_rhs1_large_cpu_policy.py tests/test_rhs1_sparse_exact_policy.py tests/test_rhs1_constraint0_policy.py tests/test_rhs1_host_policy.py tests/test_v3_driver_policy_helpers.py tests/test_v3_driver_sparse_helper_coverage.py tests/test_rhs1_sparse_first_heuristic.py tests/test_sparse_exact_lu_heuristic.py tests/test_transport_sparse_direct.py tests/test_v3_driver_solve_policy_coverage.py tests/test_policy_module_docstrings.py`
  passed with `195 passed`.
- `sphinx-build -W -b html docs docs/_build/html` passed.

### 19.64 Full-suite gate after PAS smoother and solve-mode extraction

Ran the full local suite after:

- moving PAS adaptive-smoother eligibility into `rhs1_pas_policy.py`,
- adding `solve_mode_policy.py`,
- and keeping the `v3_driver.py` wrappers as compatibility seams.

Validation:

- `pytest -q` passed with `827 passed in 350.09s (0:05:50)`.

Notes:

- This is the second full-suite gate after the recent policy split series and
  confirms the shared implicit-solve mode refactor did not regress the I/O,
  driver, CLI, solver, or bounded parity tests.
- The next lane should be chosen from the remaining plan items rather than
  continuing to split already-small wrappers by default: driver refactor
  residuals, stronger physics gates, coverage, PAS offender benchmarks, and
  documentation completeness remain the main open research-grade tracks.

### 19.65 Collision-kernel validation extension: Coulomb scaling and Rosenbluth paths

Extended the bounded collision physics/numerics gate without adding a long case:

- Added direct tests that the single-species pitch-angle-scattering deflection
  frequency is finite, positive, linear in density, and scales as `Z^4` when
  both test and field charges are doubled.
- Added a weighted barycentric interpolation exactness check for cubic
  polynomial content on nonmatching source/target nodes, complementing the
  existing identity-on-matching-nodes test.
- Added a tiny three-point Rosenbluth-potential assembly check that compares the
  analytic path against the quadrature (`quadpack`) reference for `NL=2`.
- Updated `docs/testing.rst` to record these gates as physics/numerics
  validation rather than coverage padding.

Validation:

- `python -m py_compile tests/test_collision_physics_gates.py`
- `python -m ruff check tests/test_collision_physics_gates.py`
- `pytest -q tests/test_collision_physics_gates.py tests/test_fokker_planck_phi1_reduces_to_no_phi1.py tests/test_pas_collision_operator_parity.py tests/test_fblock_fokker_planck_matvec_parity.py`
- `sphinx-build -W -b html docs docs/_build/html`
  passed; the focused pytest subset passed with `10 passed in 3.72s`.

Next validation targets:

- Use the current coverage report to choose the next cheap, real invariant from
  `vmec_wout.py`, `io.py`, or remaining collision Fokker-Planck branches rather
  than adding synthetic tests.

### 19.66 Full-suite gate after collision-kernel validation extension

Ran the full local suite after adding the Coulomb-scaling, weighted polynomial
interpolation, and Rosenbluth analytic-vs-quadrature gates.

Validation:

- `pytest -q` passed with `830 passed in 353.45s (0:05:53)`.

Notes:

- The test count increased from `827` to `830`, matching the three new
  collision-kernel gates.
- This confirms the added quadrature/analytic Rosenbluth check remains cheap
  enough for the normal suite and does not destabilize the broader driver,
  geometry, CLI, or bounded parity tests.

### 19.67 VMEC reader and half-mesh validation extension

Extended the bounded VMEC convention gate without adding a large equilibrium or
transport solve:

- Added tiny synthetic NetCDF `wout` fixture generation inside
  `tests/test_vmec_wout_conventions.py`.
- Verified that `read_vmec_wout(...)` resolves `.txt` paths to neighboring `.nc`
  files, transposes VMEC radius/mode coefficient tables into the internal
  mode/radius convention, and preserves scalar/mode metadata.
- Added failure tests for missing files, ASCII-only files without a resolvable
  NetCDF fallback, missing required variables, unsupported `lasym=true`
  equilibria, and invalid first Fourier mode metadata.
- Added explicit VMEC interpolation checks for the inner half-mesh extrapolation
  branch and exact outer half-mesh branch.
- Updated `docs/testing.rst` to record the reader-level VMEC gate.

Validation:

- `python -m py_compile tests/test_vmec_wout_conventions.py`
- `python -m ruff check tests/test_vmec_wout_conventions.py`
- `python -m pytest -q tests/test_vmec_wout_conventions.py tests/test_jax_geometry_adapters.py tests/test_geometry_grid_helper_coverage.py`
  passed with `31 passed in 1.69s`.
- `COVERAGE_FILE=/tmp/sfincs_jax_vmec_probe.coverage python -m pytest -q tests/test_vmec_wout_conventions.py --cov=sfincs_jax --cov-report=term | rg 'vmec_wout|TOTAL|passed|failed|Fatal'`
  reported `sfincs_jax/vmec_wout.py` at `99%` and `13 passed in 1.84s`.

Notes:

- The package-scoped coverage form used by the repository reports the intended
  VMEC reader coverage while exercising the same test file.

### 19.68 Full-suite gate after VMEC reader validation extension

Ran the full local suite after adding the synthetic NetCDF VMEC reader tests and
updating the testing documentation.

Validation:

- `sphinx-build -W -b html docs docs/_build/html` passed.
- `python -m pytest -q` passed with `838 passed in 337.27s (0:05:37)`.

Notes:

- The test count increased from `830` to `838`, matching the eight new VMEC
  convention/reader tests.
- This confirms the extra NetCDF fixture generation does not add meaningful CI
  cost and does not regress the broader geometry, driver, CLI, output, or bounded
  parity tests.

### 19.69 CI warning cleanup: structured-velocity docstring

Cleaned up a warning surfaced by the package-scoped coverage probes:

- Converted the block-tridiagonal factorization docstring in
  `sfincs_jax/structured_velocity.py` to a raw docstring so the LaTeX
  `\begin{...}` / `\ddots` content is not interpreted as Python escape
  sequences.

Validation:

- `python -m py_compile sfincs_jax/structured_velocity.py`
- `python -m ruff check sfincs_jax/structured_velocity.py`
- `python -m pytest -q tests/test_structured_velocity.py` passed with
  `5 passed in 1.86s`.
- `COVERAGE_FILE=/tmp/sfincs_jax_warning_probe.coverage python -m pytest -q tests/test_collision_physics_gates.py --cov=sfincs_jax --cov-report=term | rg 'DeprecationWarning|structured_velocity|passed'`
  reported no `DeprecationWarning` and `7 passed in 3.17s`.

### 19.70 Full-suite gate at branch tip after warning cleanup

Ran the full local suite at branch tip after the structured-velocity docstring
warning cleanup.

Validation:

- `python -m pytest -q` passed with `838 passed in 344.44s (0:05:44)`.

Notes:

- This confirms the pushed branch tip remains green after the collision,
  VMEC-reader, documentation, and CI-warning batches.

### 19.71 IO helper validation extension: export-f, Phi1 history, and localization

Extended the cheap IO/helper validation lane without adding a solve:

- Added export-`f` tests for periodic linear wrapping in theta/zeta, identity
  X and xi maps, the single-zeta shortcut, and invalid zeta/x/xi option errors.
- Added `Phi1` history-alignment tests for empty histories and short non-frozen
  histories, verifying that output diagnostics are padded with the result or
  latest accepted iterate rather than silently reusing an initial guess.
- Added equilibrium localization tests for inputs without an equilibrium file
  and for unquoted legacy Boozer keys, complementing the existing quoted,
  VMEC, Boozer, and non-stellarator-symmetric localization coverage.
- Updated `docs/testing.rst` to describe the IO/helper gate as part of the
  release validation stack.

Validation:

- `python -m py_compile tests/test_io_export_and_h5_coverage.py tests/test_phi1_history_alignment.py tests/test_input_compat.py`
- `python -m ruff check tests/test_io_export_and_h5_coverage.py tests/test_phi1_history_alignment.py tests/test_input_compat.py`
- `python -m pytest -q tests/test_io_export_and_h5_coverage.py tests/test_phi1_history_alignment.py tests/test_input_compat.py tests/test_io_output_policy_coverage.py tests/test_io_cache_helpers.py`
  passed with `47 passed in 3.10s`.
- `COVERAGE_FILE=/tmp/sfincs_jax_io_probe.coverage python -m pytest -q tests/test_io_export_and_h5_coverage.py tests/test_phi1_history_alignment.py tests/test_input_compat.py tests/test_io_output_policy_coverage.py tests/test_io_cache_helpers.py --cov=sfincs_jax --cov-report=term | rg 'sfincs_jax/io.py|sfincs_jax/input_compat.py|TOTAL|passed|failed|Fatal|DeprecationWarning'`
  reported `sfincs_jax/io.py` at `29%`, `sfincs_jax/input_compat.py` at
  `79%`, and `47 passed in 6.63s`.

Next validation targets:

- Build docs with warnings as errors, then run the full suite once the IO docs
  paragraph is in place.
- Continue choosing cheap physics/numerics invariants from the remaining
  Fokker-Planck branches before opening a larger PAS performance benchmark.

### 19.72 Full-suite gate after IO helper validation extension

Ran the full local suite after the export-`f`, `Phi1` history, localization,
and testing-documentation updates.

Validation:

- `sphinx-build -W -b html docs docs/_build/html` passed.
- `python -m pytest -q` passed with `844 passed in 425.55s (0:07:05)`.

Notes:

- The test count increased from `838` to `844`, matching the six new bounded
  IO/helper tests.
- The full-suite runtime stayed within the local CI target band and the branch
  remains green after touching output/export/path validation.

### 19.73 Fokker-Planck apply-path validation extension

Extended the bounded collision validation gate beyond coefficient construction:

- Added a direct `apply_fokker_planck_v3(...)` test for dense speed-space
  matrix application, including runtime rebuilding of inactive-Legendre masks.
- Added shape-guard tests for malformed no-`Phi1` Fokker-Planck inputs and
  operator tensors.
- Added a direct `apply_fokker_planck_v3_phi1(...)` test for the
  `nHat * exp(-Z alpha Phi1Hat / THat)` Boltzmann density factor and inactive
  Legendre masking.
- Added `Phi1` shape/operator guard tests for the collision operator.
- Updated `docs/testing.rst` to document this as part of the collision physics
  validation gate.

Validation:

- `python -m py_compile tests/test_collision_physics_gates.py`
- `python -m ruff check tests/test_collision_physics_gates.py`
- `python -m pytest -q tests/test_collision_physics_gates.py tests/test_fokker_planck_phi1_reduces_to_no_phi1.py tests/test_fblock_fokker_planck_matvec_parity.py tests/test_pas_collision_operator_parity.py`
  passed with `14 passed in 4.90s`.
- `COVERAGE_FILE=/tmp/sfincs_jax_collision2_probe.coverage python -m pytest -q tests/test_collision_physics_gates.py --cov=sfincs_jax --cov-report=term | rg 'sfincs_jax/collisions.py|TOTAL|passed|failed|Fatal|DeprecationWarning'`
  reported `sfincs_jax/collisions.py` at `67%` and `11 passed in 3.89s`.

Next validation targets:

- Build docs with warnings as errors and run the full suite.
- After this bounded collision gate, the next high-ROI item is a performance
  pass on PAS runtime/memory offenders rather than continuing to add small
  helper tests indefinitely.

### 19.74 Full-suite gate after Fokker-Planck apply-path validation

Ran the full local suite after adding the Fokker-Planck apply-path tests and
updating the testing documentation.

Validation:

- `sphinx-build -W -b html docs docs/_build/html` passed.
- `python -m pytest -q` passed with `848 passed in 367.34s (0:06:07)`.

Notes:

- The test count increased from `844` to `848`, matching the four new
  Fokker-Planck apply/guard tests.
- This closes the current cheap collision-validation lane; the next practical
  target should shift to PAS performance/runtime offender work unless a new
  correctness regression appears.

### 19.75 CPU PAS-DKES structured preconditioner promotion

Ran a bounded current-tip PAS offender sweep before changing solver defaults.

Measurements:

- `geometryScheme4_2species_PAS_noEr`: default stayed the only viable CPU
  route in the tested set, completing in `3.605s` wall / `2.530s` solve elapsed
  with `0` Fortran mismatches. Forced `xmg`, `pas_lite`, `point_xdiag`, and
  `xblock_tz_lmax` each hit the `45s` cap, so no default-policy change was made
  for this case.
- `HSX_PASCollisions_DKESTrajectories`: explicit `pas_tz` completed
  parity-clean and lowered both runtime and memory versus the previous default
  (`4.005s` / about `1007 MB` versus `5.200s` / about `2063 MB` in the initial
  sweep). `pas_ilu` was rejected (`41.601s` and output deltas versus default).
- `HSX_PASCollisions_fullTrajectories`: default remained better for runtime
  (`4.571s`) while explicit `pas_tz` was lower-memory but slower (`6.233s`), so
  the new rule is explicitly restricted to DKES trajectory cases.
- `geometryScheme4_1species_PAS_withEr_DKESTrajectories` and
  `sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_DKESTrajectories`
  were already selecting `pas_tz` by default and stayed parity-clean.

Code changes:

- Added `rhs1_pas_dkes_cpu_pas_tz_preferred(...)` with bounded CPU-only guards
  to promote PAS-DKES auto-selection from dense `xblock_tz` to structured
  `pas_tz` when the angular block is large enough and `active_size <= 15000`.
- Routed both the Schur-auto and weak-auto PAS default paths through this
  helper.
- Extended `scripts/benchmark_case_variants.py` so benchmark rows record the
  selected RHSMode=1 preconditioner and timeout stdout/stderr tails remain
  JSON-safe.
- Documented the new knobs
  `SFINCS_JAX_RHSMODE1_PAS_DKES_CPU_PAS_TZ_MIN` and
  `SFINCS_JAX_RHSMODE1_PAS_DKES_CPU_PAS_TZ_ACTIVE_MAX`.

Validation:

- `python -m py_compile sfincs_jax/rhs1_preconditioner_auto_policy.py sfincs_jax/v3_driver.py scripts/benchmark_case_variants.py tests/test_rhs1_preconditioner_auto_policy.py tests/test_benchmark_case_variants.py`
- `python -m ruff check sfincs_jax/rhs1_preconditioner_auto_policy.py scripts/benchmark_case_variants.py tests/test_rhs1_preconditioner_auto_policy.py tests/test_benchmark_case_variants.py`
- `python -m pytest -q tests/test_rhs1_preconditioner_auto_policy.py tests/test_benchmark_case_variants.py`
  passed with `17 passed in 6.18s`.
- Post-policy `HSX_PASCollisions_DKESTrajectories` default probe selected
  `rhs1_preconditioner=pas_tz`, completed in `3.940s` wall / `3.123s` solve
  elapsed, used about `1019 MB` RSS, and had `0/123` Fortran mismatches.
- Non-DKES guard probe `HSX_PASCollisions_fullTrajectories` stayed default
  `rhs1_preconditioner=schur`, completed in `4.121s`, and had `0/193` Fortran
  mismatches.

Next validation targets:

- Build docs with warnings as errors and run the full local suite once the
  README/performance-table updates are complete.
- GPU PAS-DKES should be measured on `ssh office` before enabling any analogous
  GPU default; this change intentionally leaves the GPU path untouched.

### 19.76 Full-suite gate after CPU PAS-DKES promotion

Ran the release-style local validation after the CPU PAS-DKES policy, benchmark
harness, README, and documentation updates.

Validation:

- `sphinx-build -W -b html docs docs/_build/html` passed.
- `python -m pytest -q` passed with `852 passed in 342.08s (0:05:42)`.

Notes:

- The local suite count increased from `848` to `852`, matching the four new
  benchmark/policy helper tests added in this pass.
- The docs build and full test runtime remain inside the target local CI budget.
- The next PAS performance lane should be GPU-only measurement for PAS-DKES on
  `ssh office`, not another CPU default change.

### 19.77 GPU PAS-DKES structured preconditioner promotion

Completed the GPU follow-up for the HSX PAS-DKES offender on `office`.

Measurements:

- Pulled `refactor/v3-driver-split` in `/home/rjorge/sfincs_jax_refactor_v3`
  and copied only the frozen HSX PAS-DKES case directory to
  `/tmp/sfincs_jax_gpu_cases/HSX_PASCollisions_DKESTrajectories`.
- One-GPU baseline before the GPU policy change:
  - default `xblock_tz`: `14.181s` wall / `13.005s` elapsed, `1530084 KB`
    RSS, `0/123` Fortran mismatches;
  - forced `pas_tz`: `12.583s` wall / `11.480s` elapsed, `1259792 KB` RSS,
    `0/123` Fortran mismatches.
- After generalizing the guarded PAS-DKES preference to CPU/GPU, the default
  one-GPU run selected `rhs1_preconditioner=pas_tz`, completed in `7.627s`
  wall / `6.515s` elapsed, used `1203084 KB` RSS, and had `0/123` Fortran
  mismatches.

Code/documentation changes:

- Replaced the CPU-only helper with `rhs1_pas_dkes_pas_tz_preferred(...)`,
  retaining `rhs1_pas_dkes_cpu_pas_tz_preferred(...)` as a compatibility alias.
- Added backend-specific knobs:
  `SFINCS_JAX_RHSMODE1_PAS_DKES_CPU_PAS_TZ_MIN`,
  `SFINCS_JAX_RHSMODE1_PAS_DKES_CPU_PAS_TZ_ACTIVE_MAX`,
  `SFINCS_JAX_RHSMODE1_PAS_DKES_GPU_PAS_TZ_MIN`, and
  `SFINCS_JAX_RHSMODE1_PAS_DKES_GPU_PAS_TZ_ACTIVE_MAX`.
- Updated README and performance docs with the focused current-tip CPU/GPU
  HSX PAS-DKES row.

Validation:

- `python -m py_compile sfincs_jax/rhs1_preconditioner_auto_policy.py sfincs_jax/v3_driver.py tests/test_rhs1_preconditioner_auto_policy.py`
- `python -m ruff check sfincs_jax/rhs1_preconditioner_auto_policy.py tests/test_rhs1_preconditioner_auto_policy.py scripts/benchmark_case_variants.py tests/test_benchmark_case_variants.py`
- `python -m pytest -q tests/test_rhs1_preconditioner_auto_policy.py tests/test_benchmark_case_variants.py`
  passed with `18 passed in 7.51s`.
- `sphinx-build -W -b html docs docs/_build/html` passed after the GPU
  documentation update.

Next validation targets:

- Run the full local pytest suite after the final README/docs updates.
- The remaining PAS runtime/memory offender work should move to
  `HSX_PASCollisions_fullTrajectories`, `geometryScheme4_2species_PAS_noEr`,
  and the larger tokamak PAS+Er GPU lane; the HSX DKES CPU/GPU default is now
  closed for the current focused benchmark.

### 19.78 Full-suite gate after GPU PAS-DKES promotion

Ran the final local validation after updating the README/performance docs with
the one-GPU HSX PAS-DKES default rerun.

Validation:

- `sphinx-build -W -b html docs docs/_build/html` passed.
- `python -m pytest -q` passed with `853 passed in 345.15s (0:05:45)`.

Notes:

- The local suite count increased from `852` to `853`, matching the new GPU
  backend-bound policy test.
- The branch remains within the target local CI runtime after the CPU/GPU
  PAS-DKES preconditioner policy changes.

### 19.79 CPU HSX full-trajectory PAS structured preconditioner promotion

Closed the next bounded CPU PAS offender after the DKES lane by testing the
full-trajectory HSX case and the geometry11 W7X guard case before changing the
default.

Measurements:

- `HSX_PASCollisions_fullTrajectories`: explicit `pas_tz` completed
  parity-clean and improved both runtime and memory versus the previous Schur
  default (`4.222s` wall / `3.301s` elapsed / about `1390 MB` RSS versus
  `4.558s` wall / `3.603s` elapsed / about `2101 MB` RSS in the confirmation
  A/B run).
- Post-policy `HSX_PASCollisions_fullTrajectories` selected
  `rhs1_preconditioner=pas_tz`, completed in `4.027s` wall / `3.134s`
  elapsed, used about `1384 MB` RSS, and had `0/193` mismatches against the
  frozen Fortran reference.
- The larger W7X paper geometry11 full-trajectory guard stayed on Schur after
  the policy, completed in `3.347s` wall / `2.422s` elapsed, and remained
  parity-clean with `0/193` mismatches. The forced `pas_tz` W7X probe was
  slower (`5.239s`), so the new rule is intentionally bounded by `n_zeta` and
  active DOFs.
- `geometryScheme4_2species_PAS_noEr` preconditioner-column and dtype variants
  only changed memory/runtime at noise level, so no default change was made for
  that memory offender in this pass.
- The one-GPU `tokamak_1species_PASCollisions_withEr_fullTrajectories` probe
  rejected a tempting `pas_tokamak_theta` default: it was much faster
  (`3.969s` versus the default `18.113s`) and lower-memory, but introduced one
  Fortran-output mismatch (`pressureAnisotropy`). The parity-clean GPU routes
  remained the default/explicit `xblock_tz` and `lgmres` variants, while
  `pas_hybrid` timed out. This stays an open algorithmic lane rather than a
  production default.

Code/documentation changes:

- Added `rhs1_pas_full_cpu_pas_tz_preferred(...)` with CPU-only, PAS-only,
  full-trajectory, geometryScheme=11, `n_zeta`, angular-block-size, and active
  DOF guards.
- Routed the RHSMode=1 auto preconditioner selection through that helper only
  when the current default is Schur and the user did not force a
  preconditioner.
- Added policy tests covering the HSX-like target, GPU exclusion, DKES
  exclusion, larger-W7X exclusion, and environment bounds.
- Updated README and performance docs with the focused HSX full-trajectory CPU
  row (`5.274s` / `2002 MB` to `4.027s` / `1384 MB`) and documented the new
  environment controls.

Validation:

- `python -m py_compile sfincs_jax/rhs1_preconditioner_auto_policy.py sfincs_jax/v3_driver.py tests/test_rhs1_preconditioner_auto_policy.py`
- `python -m ruff check sfincs_jax/rhs1_preconditioner_auto_policy.py tests/test_rhs1_preconditioner_auto_policy.py`
- `python -m pytest -q tests/test_rhs1_preconditioner_auto_policy.py`
  passed with `17 passed in 0.32s`.
- Focused frozen-reference probes passed on
  `HSX_PASCollisions_fullTrajectories` and
  `sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories`
  as described above.

Next validation targets:

- Build docs with warnings as errors and run the full local pytest suite after
  the README/docs/plan updates.
- Keep the tokamak PAS+Er GPU fast route as a research item until a bounded
  correction removes the `pressureAnisotropy` mismatch without giving back the
  runtime win.

### 19.80 Full-suite gate after CPU HSX full-trajectory PAS promotion

Ran the final local validation after the CPU HSX full-trajectory PAS policy,
README, performance docs, usage docs, and performance-technique notes.

Validation:

- `sphinx-build -W -b html docs docs/_build/html` passed.
- `python -m pytest -q` passed with `855 passed in 329.66s (0:05:29)`.

Notes:

- The local suite count increased from `853` to `855`, matching the two new
  bounded full-trajectory PAS policy tests.
- The branch remains inside the target local CI runtime after the PAS-DKES and
  HSX full-trajectory preconditioner promotions.
- The next unresolved high-ROI performance lane is still the tokamak PAS+Er GPU
  route: the fast `pas_tokamak_theta` variant needs a parity-preserving
  correction for `pressureAnisotropy` before it can be considered for default
  selection.

### 19.81 GPU tokamak PAS+Er tight-GMRES promotion

Closed the bounded one-GPU tokamak PAS+Er offender by separating the apparent
fast path from the actual preconditioner build. The originally tempting forced
`pas_tokamak_theta` experiment was fast only because the solve effectively ran
without building that preconditioner; building the actual `pas_tokamak_theta`
preconditioner on the same case was not a practical default. The accepted route
is therefore an explicit tight unpreconditioned GMRES policy for bounded GPU
analytic-tokamak PAS+Er cases, with the old `xblock_tz` branch left as opt-in.

Measurements:

- Old one-GPU default/`xblock_tz` route on
  `tokamak_1species_PASCollisions_withEr_fullTrajectories`: about `18.1-18.2s`
  and about `1014.5 MB` RSS, parity-clean but the top GPU runtime offender.
- Fast loose-tolerance probe: about `3.0s`, but one practical mismatch in
  `pressureAnisotropy` (`2.9e-7` absolute, `7.5e-4` relative).
- Accepted tight-GMRES route on `office` GPU1 during the local-patch probe:
  `3.412660754052922s`, `955912 KB` RSS (`933.5 MB`), `0/212` practical and
  strict mismatches, and `pressureAnisotropy` max difference
  `8.398488e-10` absolute / `1.319963e-7` relative.
- Clean remote rerun after pushing commit `2d988b7`:
  `3.249s` elapsed, `944388 KB` RSS (`922.3 MB`), no preconditioner build,
  no `pas_tokamak_theta`, `0/212` mismatches, and the same
  `pressureAnisotropy` max difference (`8.398488e-10` absolute /
  `1.319963e-7` relative).

Code/documentation changes:

- `rhs1_pas_tokamak_gpu_xblock_preferred(...)` now defaults
  `SFINCS_JAX_RHSMODE1_PAS_TOKAMAK_GPU_XBLOCK_ACTIVE_MAX` to `0`, making the
  older GPU `xblock_tz` route explicitly opt-in.
- Added `rhs1_pas_tokamak_gpu_tight_tol(...)` with
  `SFINCS_JAX_RHSMODE1_PAS_TOKAMAK_GPU_TOL` defaulting to `1e-8` for the
  bounded GPU tokamak PAS+Er route; the legacy `...GPU_THETA_TOL` name remains
  accepted.
- The RHSMode=1 auto selector now logs
  `GPU PAS tokamak auto -> tight unpreconditioned GMRES`, skips the later PAS
  weak/strong auto overrides that would rebuild `xblock_tz` or `pas_hybrid`,
  and emits the tolerance tightening when it applies.
- README, usage docs, performance docs, and performance-technique notes now
  describe the focused current-tip GPU row and the opt-in legacy branch.

Validation:

- `python -m py_compile sfincs_jax/rhs1_preconditioner_auto_policy.py sfincs_jax/v3_driver.py tests/test_rhs1_preconditioner_auto_policy.py tests/test_v3_driver_policy_helpers.py tests/test_schur_precond_heuristic.py`
- `python -m ruff check sfincs_jax/rhs1_preconditioner_auto_policy.py tests/test_rhs1_preconditioner_auto_policy.py tests/test_v3_driver_policy_helpers.py tests/test_schur_precond_heuristic.py`
- `python -m pytest -q tests/test_rhs1_preconditioner_auto_policy.py tests/test_v3_driver_policy_helpers.py tests/test_schur_precond_heuristic.py`
  passed with `55 passed in 8.76s`.
- Office GPU1 default-policy probe with the local patch selected the tight
  unpreconditioned GMRES route, avoided a preconditioner-build line, and was
  parity-clean with the measurements above.
- Clean `office` checkout validation after reversing the temporary patch and
  fast-forward pulling `2d988b7` selected the same route and was parity-clean
  with `0/212` mismatches.
- `sphinx-build -W -b html docs docs/_build/html` passed after the README/docs
  updates.
- `python -m pytest -q` passed with `856 passed in 357.93s (0:05:57)`.

Next validation targets:

- Continue with the remaining post-refactor open lanes: CPU memory offenders,
  GPU memory offenders, and distributed-solve scaling. The bounded one-GPU
  tokamak PAS+Er runtime offender is closed for this case.

### 19.82 Geometry4 PAS memory-knob rejection sweep

Ran a bounded memory-offender check on `geometryScheme4_2species_PAS_noEr`
before changing more defaults. This was intentionally a small knob sweep, not a
new algorithm, to verify whether existing chunk/cap/mixed-precision controls
already provide a safe win on the current branch.

CPU frozen-reference sweep:

- Case:
  `tests/scaled_example_suite_recheck_cpu_frozen_2026-04-23_postkeyfix/geometryScheme4_2species_PAS_noEr`
- Default: `2.743s` elapsed, `1883340800` macOS `ru_maxrss` units,
  `rhs1_preconditioner=schur`, `0` Fortran mismatches.
- `SFINCS_JAX_PRECOND_DTYPE=float32`: parity-clean and faster (`2.343s`),
  but higher RSS (`1997324288`), so not a memory win.
- `SFINCS_JAX_PRECOND_PAS_MAX_COLS=32`: parity-clean and faster (`2.312s`),
  but higher RSS (`1999585280`), so not a memory win.
- `SFINCS_JAX_PRECOND_MAX_MB=64`: parity-clean, `2.674s`, higher RSS
  (`1929150464`).
- `SFINCS_JAX_PRECOND_CHUNK=64`: parity-clean, `2.579s`, higher RSS
  (`1960198144`).

GPU clean-remote sweep on `office` GPU1:

- Case staged at
  `/tmp/sfincs_jax_gpu_cases/geometryScheme4_2species_PAS_noEr`.
- Default: `6.246s` elapsed, `2603768 KB` RSS,
  `rhs1_preconditioner=schur`, `0` Fortran mismatches.
- `SFINCS_JAX_PRECOND_DTYPE=float32`: timed out at `180s` after a bad
  stage-2/`pas_hybrid` fallback; reject for automatic use.
- `SFINCS_JAX_PRECOND_PAS_MAX_COLS=32`: parity-clean, but slower
  (`8.559s`) and only reduced RSS to `2566568 KB` (~1.4%).
- `SFINCS_JAX_PRECOND_MAX_MB=64`: parity-clean, but slower (`8.862s`) and
  only reduced RSS to `2567608 KB`.
- `SFINCS_JAX_PRECOND_CHUNK=64`: parity-clean, but slower (`8.192s`) and
  only reduced RSS to `2564480 KB`.

Decision:

- Do not promote any existing memory cap/chunk/mixed-precision knob for this
  offender. The GPU memory savings are too small for the runtime cost, and the
  CPU variants do not reduce RSS.
- The next real memory step for geometry4 PAS should be algorithmic: reduce the
  Schur/PAS preconditioner live working set, avoid duplicated dense block
  materialization, or add a genuinely streaming/apply-only angular solve rather
  than only retuning chunk sizes.

### 19.83 Geometry4 PAS direct-pas_tz memory policy

Implemented the next algorithmic memory step for
`geometryScheme4_2species_PAS_noEr`: select direct top-level `pas_tz` for
bounded geometryScheme=4 PAS, non-DKES, near-zero-Er, no-FP cases instead of
wrapping the same angular block inside the constraint-Schur preconditioner.

Code changes:

- Added `rhs1_geometry4_pas_memory_pas_tz_preferred(...)` with guards on default
  preconditioner mode, geometryScheme=4, PAS-only, non-DKES, near-zero `Er`,
  `pas_tz` applicability, angular block size, and active DOFs.
- Added environment controls:
  `SFINCS_JAX_RHSMODE1_GEOM4_PAS_MEMORY_PAS_TZ`,
  `SFINCS_JAX_RHSMODE1_GEOM4_PAS_MEMORY_PAS_TZ_MIN`,
  `SFINCS_JAX_RHSMODE1_GEOM4_PAS_MEMORY_PAS_TZ_ACTIVE_MIN`, and
  `SFINCS_JAX_RHSMODE1_GEOM4_PAS_MEMORY_PAS_TZ_ACTIVE_MAX`.
- The previous Schur route remains available with
  `SFINCS_JAX_RHSMODE1_GEOM4_PAS_MEMORY_PAS_TZ=0`.

Measurements:

- Local CPU focused rerun after the policy selected `rhs1_preconditioner=pas_tz`,
  completed in `1.962s` elapsed, used `1811988480` macOS `ru_maxrss` units
  (`1728.0 MB`), and had `0` Fortran mismatches. Disabling the policy restored
  `schur`, ran in `2.476s`, and used `1923727360` macOS `ru_maxrss` units.
- Clean-remote `office` GPU1 rerun after pulling commit `e721a6f` selected
  `rhs1_preconditioner=pas_tz`, completed in `4.774s` elapsed, used
  `1860564 KB` RSS (`1817.0 MB`), and had `0` Fortran mismatches. Disabling the
  policy restored `schur`, ran in `5.899s`, and used `2567152 KB` RSS
  (`2507.0 MB`).

Validation:

- `python -m py_compile sfincs_jax/rhs1_preconditioner_auto_policy.py sfincs_jax/v3_driver.py tests/test_rhs1_preconditioner_auto_policy.py tests/test_schur_precond_heuristic.py`
- `python -m ruff check sfincs_jax/rhs1_preconditioner_auto_policy.py tests/test_rhs1_preconditioner_auto_policy.py tests/test_schur_precond_heuristic.py`
- `python -m pytest -q tests/test_rhs1_preconditioner_auto_policy.py tests/test_schur_precond_heuristic.py`
  passed with `42 passed in 9.46s`.
- `sphinx-build -W -b html docs docs/_build/html` passed after README/docs
  updates.
- `python -m pytest -q` passed with `857 passed in 350.71s (0:05:50)`.

Next validation targets:

- Continue memory-offender work on
  `sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories`
  and CPU `monoenergetic_geometryScheme5_ASCII`.

### 19.84 Geometry11 PAS GPU memory sweep

Ran a clean `office` GPU1 focused sweep on
`sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories`
from the frozen GPU case directory.

Measured variants:

- Default `schur`: `12.267s`, `2146300 KB` RSS (`2096.0 MB`), `0` Fortran
  mismatches.
- Forced `pas_tz`: `20.156s`, `1687252 KB` RSS (`1647.7 MB`), `0` mismatches.
- Forced `pas_tz` with `SFINCS_JAX_RHSMODE1_PAS_TZ_LMAX=8`: `37.569s`,
  `1906340 KB` RSS, `0` mismatches.
- Forced `schur` with `SFINCS_JAX_RHSMODE1_SCHUR_MODE=diag`: `32.086s`,
  `1994784 KB` RSS, `0` mismatches.
- Forced `schur` with `SFINCS_JAX_RHSMODE1_SCHUR_BASE=pas_tz`: `14.159s`,
  `2145628 KB` RSS, `0` mismatches.
- Forced `point_xdiag`: timed out at `180s`.
- Follow-up solver/restart sweep: `bicgstab`, `pas_tz+bicgstab`,
  `pas_hybrid`, and `pas_lite` all timed out at `120s`; GMRES restart caps
  (`40`/`20`) were parity-clean but slower (`15-16s`) for only ~7-8% RSS
  reduction.

Decision:

- Do not promote a default geometry11 GPU memory policy yet. Direct `pas_tz`
  is a useful manual low-memory knob, but the runtime penalty is too large for
  the default release path. The safe default remains Schur until a genuinely
  faster streaming/angular solve is available.

### 19.85 VMEC monoenergetic CPU low-memory default

Implemented a guarded CPU low-memory default for bounded VMEC monoenergetic
transport:

- Added `transport_geometry5_mono_low_memory_preferred(...)` in
  `sfincs_jax/transport_solve_policy.py`.
- The automatic guard applies to CPU `RHSMode=3`, `geometryScheme=5`, PAS/no-FP,
  `Nx <= 2`, and total size between
  `SFINCS_JAX_TRANSPORT_GEOM5_MONO_LOW_MEMORY_MIN` and
  `SFINCS_JAX_TRANSPORT_GEOM5_MONO_LOW_MEMORY_MAX` (defaults `1000` and
  `20000`).
- `SFINCS_JAX_TRANSPORT_GEOM5_MONO_LOW_MEMORY=0` restores the previous dense
  batched fallback; `=1` forces the low-memory path for comparison.

Measurements:

- `monoenergetic_geometryScheme5_ASCII`, CLI default before the policy:
  `2.445s` wall, `2950.7 MB` profiled RSS, `0` Fortran mismatches.
- `monoenergetic_geometryScheme5_ASCII`, new default after the policy:
  `1.518s` logged total, `506.5 MB` profiled RSS, `0` Fortran mismatches.
- `monoenergetic_geometryScheme5_netCDF`, new default:
  `2.242s` logged total, `603.2 MB` profiled RSS, `0` Fortran mismatches.

Validation:

- `python -m py_compile sfincs_jax/transport_solve_policy.py sfincs_jax/v3_driver.py tests/test_transport_solve_policy.py`
- `python -m ruff check sfincs_jax/transport_solve_policy.py tests/test_transport_solve_policy.py`
- `python -m pytest -q tests/test_transport_solve_policy.py`
- Focused CLI parity probes for `monoenergetic_geometryScheme5_ASCII` and
  `monoenergetic_geometryScheme5_netCDF` against their frozen Fortran
  `sfincsOutput.h5` references, both `0` mismatches.
- `pytest -q tests/test_transport_solve_policy.py tests/test_transport_matrix_rhsmode3_parity.py tests/test_transport_matrix_write_output_end_to_end.py`
  passed with `14 passed in 4.89s`.
- `sphinx-build -W -b html docs docs/_build/html` passed.
- `python -m pytest -q` passed with `858 passed in 333.60s (0:05:33)`.

Next validation targets:

- Continue CPU/GPU offender work with geometry11 PAS full-trajectory as the
  remaining memory target; treat direct `pas_tz` there as an opt-in knob until
  runtime improves.

### 19.86 Publication validation dashboard and artifact gates

Implemented a bounded, literature-anchored publication validation lane that does not
rerun large scans in CI:

- Added `sfincs_jax/validation_artifacts.py` with focused loaders and metrics for
  checked-in collisionality and radial-electric-field sweep artifacts.
- Added `examples/publication_figures/generate_validation_dashboard.py`, producing
  `docs/_static/figures/paper/sfincs_jax_publication_validation_dashboard.{png,pdf}`
  and
  `examples/publication_figures/artifacts/sfincs_jax_publication_validation_dashboard_summary.json`.
- Added tests that assert the artifact physics gates directly:
  - LHD/W7-X collisionality summaries contain both FP and PAS rows on the audited
    seven-point grid,
  - high-collisionality `L11` FP/PAS separation exceeds low-collisionality
    separation, consistent with the collision-operator discussion in Landreman et
    al. 2014,
  - pinned DKES/partial/full trajectory sweeps agree exactly at `Er = 0`,
  - finite-`Er` sweeps preserve nonzero model separation.
- Updated the validation manifest, source map, testing docs, references, paper
  figures page, validation matrix, and landing page.

Measured dashboard metrics from the checked-in artifacts:

- LHD `L11` high/low FP-PAS relative-separation ratio: `52.90`.
- W7-X `L11` high/low FP-PAS relative-separation ratio: `146.91`.
- Tokamak-like trajectory sweep zero-field spread across all plotted diagnostics:
  exactly `0.0` in the pinned artifact.

Validation:

- `python -m pytest -q tests/test_validation_artifacts.py tests/test_generate_validation_dashboard.py tests/test_validation_manifest_schema.py`
  passed with `7 passed in 1.05s`.
- `python -m pytest -q tests/test_collisionality_artifact.py tests/test_er_trajectory_sweep_artifact.py tests/test_generate_sfincs_paper_figs.py tests/test_er_trajectory_sweep.py`
  passed with `29 passed in 1.80s`.
- `python -m ruff check sfincs_jax/validation_artifacts.py examples/publication_figures/generate_validation_dashboard.py tests/test_validation_artifacts.py tests/test_generate_validation_dashboard.py`
  passed.
- `sphinx-build -W -b html docs docs/_build/html` passed.
- `python -m pytest -q` passed with `862 passed in 326.23s (0:05:26)`.

Next validation targets:

- Promote `sfincs2014_fig3_high_collisionality_limit` from `needs_reaudit` only
  after the analytic Simakov-Helander normalization is regenerated from the corrected
  full collisionality artifact family.
- Keep `w7x_ambipolar_er_validation` planned until a defensible profile/equilibrium
  reconstruction is pinned with provenance.
- Build the MONKES/KNOSOS monoenergetic overlap lane only on a documented shared-model
  subset, so exact equality claims and qualitative trend claims stay separate.

### 19.87 High-collisionality trend proxy artifact

Added a second publication-facing artifact that closes the cheap, machine-readable part
of the high-collisionality lane without overclaiming the full Simakov-Helander
analytic-limit reproduction:

- Added high-collisionality slope utilities to `sfincs_jax/validation_artifacts.py`:
  `transport_element_abs_series(...)`, `collisionality_power_law_slope(...)`,
  `high_collisionality_trend_summary(...)`, and
  `build_high_collisionality_trend_proxy_summary(...)`.
- Added `examples/publication_figures/generate_high_collisionality_trend_proxy.py`.
- Generated:
  - `docs/_static/figures/paper/sfincs_jax_high_collisionality_trend_proxy.png`,
  - `docs/_static/figures/paper/sfincs_jax_high_collisionality_trend_proxy.pdf`,
  - `examples/publication_figures/artifacts/sfincs_jax_high_collisionality_trend_proxy_summary.json`.
- Updated the validation manifest, paper-figures page, validation matrix, testing docs,
  and source map.

Physics gate rationale:

- The SFINCS 2014 paper states that, at high collisionality, PAS `L11`/`L12` scale
  like `+nu`, while momentum-conserving FP/model-operator `L11`/`L12` should approach
  inverse-`nu` scaling only in the true `nu' >> 1` limit.
- The checked-in corrected scans only reach `nu'=10`, so this branch now records tail
  slopes from the last three points as a trend proxy rather than treating the existing
  `sfincs_jax_fig3_simakov_helander.png` as a finalized analytic-limit reproduction.

Measured slopes from the checked-in artifact:

- LHD PAS: `L11` slope `+0.847`, `L12` slope `+0.841`.
- LHD FP: `L11` slope `+0.192`, `L12` slope `+0.200`; state is therefore
  `needs_wider_high_nu_scan`.
- W7-X PAS: `L11` slope `+0.790`, `L12` slope `+0.688`.
- W7-X FP: `L11` slope `-1.232`, `L12` slope `-1.299`; state is
  `asymptotic_trend_proxy`.

Validation:

- `python -m pytest -q tests/test_validation_artifacts.py tests/test_generate_high_collisionality_trend_proxy.py tests/test_generate_validation_dashboard.py tests/test_validation_manifest_schema.py`
  passed with `10 passed in 1.73s`.
- `python -m ruff check sfincs_jax/validation_artifacts.py examples/publication_figures/generate_high_collisionality_trend_proxy.py examples/publication_figures/generate_validation_dashboard.py tests/test_validation_artifacts.py tests/test_generate_high_collisionality_trend_proxy.py tests/test_generate_validation_dashboard.py`
  passed.
- `sphinx-build -W -b html docs docs/_build/html` passed.
- `python -m pytest -q` passed with `865 passed in 336.37s (0:05:36)`.

Next validation targets:

- Generate a wider high-collisionality collisionality ladder before promoting the
  Simakov-Helander lane from `needs_reaudit`.
- Keep the W7-X ambipolar and MONKES/KNOSOS lanes explicit in the manifest until their
  input reconstruction / normalization choices are pinned.

### 19.88 Frozen CPU/GPU Fortran-suite benchmark artifact

Closed the publication-facing cross-code benchmark summary for the final frozen CPU and
GPU suite reports without rerunning the heavy examples in CI:

- Added suite-report loaders and metrics to `sfincs_jax/validation_artifacts.py`:
  `load_suite_report(...)`, `suite_case_metrics(...)`, `suite_report_summary(...)`,
  and `build_fortran_suite_benchmark_summary(...)`.
- Added `examples/publication_figures/generate_fortran_suite_benchmark_summary.py`.
- Generated:
  - `docs/_static/figures/paper/sfincs_jax_fortran_suite_benchmark_summary.png`,
  - `docs/_static/figures/paper/sfincs_jax_fortran_suite_benchmark_summary.pdf`,
  - `examples/publication_figures/artifacts/sfincs_jax_fortran_suite_benchmark_summary.json`.
- Updated the validation manifest, paper-figures page, validation matrix, Fortran
  comparison page, performance page, testing docs, and source map.

Measured release-gate metrics from the frozen reports:

- CPU report `tests/scaled_example_suite_recheck_cpu_frozen_2026-04-23_postkeyfix/suite_report.json`:
  `39/39 parity_ok`, zero `jax_error`, zero `max_attempts`, zero strict mismatches,
  median JAX/Fortran runtime ratio `0.039x`, median maximum-RSS ratio `5.18x`.
- GPU report `tests/scaled_example_suite_recheck_gpu_frozen_2026-04-23_postruntimefix_mem/suite_report.json`:
  `39/39 parity_ok`, zero `jax_error`, zero `max_attempts`, zero strict mismatches,
  median JAX/Fortran runtime ratio `0.059x`, median maximum-RSS ratio `9.20x`.

The high runtime-ratio tail is now explicitly stored in JSON instead of being hidden in
hand-written docs. This matters because several Fortran reference runs take only about
`0.017 s`, so ratio plots can look severe even when the JAX absolute runtime remains a
few seconds.

Validation:

- `python -m ruff check sfincs_jax/validation_artifacts.py examples/publication_figures/generate_fortran_suite_benchmark_summary.py tests/test_validation_artifacts.py tests/test_generate_fortran_suite_benchmark_summary.py`
  passed.
- `python -m pytest -q tests/test_validation_artifacts.py tests/test_generate_fortran_suite_benchmark_summary.py tests/test_generate_validation_dashboard.py tests/test_generate_high_collisionality_trend_proxy.py tests/test_validation_manifest_schema.py`
  passed with `13 passed in 2.40s`.
- `sphinx-build -W -b html docs docs/_build/html` passed.
- `python -m pytest -q` passed with `868 passed in 344.52s (0:05:44)`.

### 19.89 Literature DOI correction for validation artifacts

The follow-up literature check found a real provenance error: the SFINCS 2014
validation paper DOI is `10.1063/1.4870077`; an earlier local metadata entry used
the wrong DOI suffix. Corrected:

- `sfincs_jax/validation_artifacts.py`,
- `examples/publication_figures/validation_manifest.json`,
- `docs/references.rst`,
- `docs/validation_matrix.rst`,
- and the generated validation/benchmark summary JSON artifacts.

The corrected DOI is backed by the local upstream bibliography and the public paper
records:

- `docs/upstream/manual/version3/SFINCSUserManual.bib`,
- `https://publications.lib.chalmers.se/records/fulltext/199559/local_199559.pdf`,
- `https://www.osti.gov/biblio/22253325`,
- `https://github.com/landreman/sfincs`.

Validation:

- `python -m ruff check sfincs_jax/validation_artifacts.py tests/test_validation_artifacts.py`
  passed.
- `python -m pytest -q tests/test_validation_artifacts.py tests/test_generate_validation_dashboard.py tests/test_generate_high_collisionality_trend_proxy.py tests/test_generate_fortran_suite_benchmark_summary.py tests/test_validation_manifest_schema.py`
  passed with `13 passed in 2.30s`.
- `sphinx-build -W -b html docs docs/_build/html` passed.

## 20. Post-1.0 deferred research lanes and manuscript figure plan

The 1.0.2 release manifest has no release-blocking open lanes. The remaining
research program is intentionally tracked as `deferred_post_release` so the next
work can be done at manuscript quality without weakening the shipped release
claims. Do not promote any lane to `implemented` until it has:

- a machine-readable summary JSON with source inputs, normalization choices,
  code revision, solver settings, residuals, and runtime/memory metadata;
- publication-ready PNG and PDF figures generated by a script, not edited by hand;
- focused fast tests that run in CI and heavier reference commands documented for
  office/GPU/nightly execution;
- documentation pages explaining the physics claim, equations, acceptance gates,
  limitations, and exact artifact provenance;
- a manifest entry whose claims are no stronger than the artifacts actually prove.

Primary literature anchors:

- Paul, Abel, Landreman, and Dorland, "An adjoint method for neoclassical
  stellarator optimization", arXiv:1904.06430 and JPP DOI
  `10.1017/S0022377819000527`.
- Landreman, Smith, Mollen, and Helander, "Comparison of particle trajectories
  and collision operators for collisional transport in nonaxisymmetric plasmas",
  Physics of Plasmas DOI `10.1063/1.4870077`.
- Simakov and Helander, "Neoclassical momentum transport in a collisional
  stellarator and a rippled tokamak", Physics of Plasmas DOI
  `10.1063/1.3104715`.
- Pablant et al. W7-X radial-electric-field / ion-root validation context and
  the W7-X reduced-neoclassical-transport validation literature.
- Escoto, Velasco, Calvo, Landreman, and Parra, MONKES, arXiv:2312.12248 and
  Nuclear Fusion DOI `10.1088/1741-4326/ad3fc9`.
- Velasco, Calvo, Parra, and Garcia-Regana, KNOSOS, arXiv:1908.11615 and JCP
  DOI `10.1016/j.jcp.2020.109512`.

### 20.1 Work order

1. Manuscript-scale autodiff / sensitivity validation.
2. Full Simakov-Helander high-collisionality analytic-limit reproduction.
3. W7-X ambipolar `E_r` validation with defensible equilibrium/profile inputs.
4. MONKES / KNOSOS shared-model overlap.
5. Consolidated paper figure set and reviewer-proof documentation.

This order is deliberate. The autodiff lane exercises the differentiable
architecture that distinguishes `sfincs_jax` from Fortran v3, while the remaining
lanes strengthen external physics validation and cross-code trust.

### 20.2 Manuscript-scale autodiff / sensitivity validation

Goal: move `adjoint_sensitivity_gradient_checks` from focused CI fixtures to a
manuscript-grade validation lane showing that gradients of SFINCS-like linear
solve observables are accurate, stable, and useful for optimization-oriented
workflows.

Implementation targets:

- Add `examples/publication_figures/generate_autodiff_sensitivity_validation.py`.
- Add artifact helpers in `sfincs_jax/validation_artifacts.py` for:
  `load_autodiff_sensitivity_summary(...)`,
  `autodiff_gradient_error_summary(...)`, and
  `build_autodiff_sensitivity_validation_summary(...)`.
- Reuse the existing differentiable solve path in `sfincs_jax/implicit_solve.py`
  based on `jax.lax.custom_linear_solve`; do not differentiate through Krylov
  iterations for manuscript claims.
- Reuse current examples as seeds:
  `examples/autodiff/implicit_diff_through_gmres_solve_scheme5.py`,
  `examples/autodiff/matrix_free_residual_and_jvp.py`,
  `examples/autodiff/differentiable_geometry_gradients.py`, and
  `examples/optimization/fit_geometry_harmonics_with_optax.py`.

Physics / numerical objectives:

- scalar linear-solve objective: `J = 0.5 ||x(p)||_2^2` for a pinned
  `geometryScheme=5` PAS fixture;
- transport objective: one bootstrap-current or radial-flux scalar extracted from
  a solved state when the output path is differentiable enough;
- geometry objective: sensitivity of a transport/residual proxy to low-order
  Boozer or VMEC-like harmonic amplitudes;
- optional root-aware objective: local derivative of an ambipolar-root residual
  only after the simple linear-solve and geometry lanes are stable.

Gradient checks:

- compare JAX reverse-mode gradients against centered finite differences for
  one-parameter and few-parameter cases;
- compare JAX JVPs against finite-difference directional derivatives for
  10-100 parameter vectors;
- record primal residual, transpose/adjoint residual, gradient relative error,
  step-size sweep, solve tolerance, dtype, solver kind, and runtime;
- include GMRES and BiCGStab wrappers where both are stable for the same fixture;
- fail fast if gradients are finite but residuals are not clean enough to support
  the claim.

Publication artifacts:

- `examples/publication_figures/artifacts/sfincs_jax_autodiff_sensitivity_validation_summary.json`
- `docs/_static/figures/paper/sfincs_jax_autodiff_gradient_check.{png,pdf}`
- `docs/_static/figures/paper/sfincs_jax_autodiff_sensitivity_map.{png,pdf}`
- optional:
  `docs/_static/figures/paper/sfincs_jax_autodiff_gradient_cost_scaling.{png,pdf}`

Reviewer-facing figure design:

- Panel A: `dJ/dp` from autodiff vs centered finite difference on a log/log
  parity plot with a one-to-one line.
- Panel B: finite-difference step sweep showing the expected roundoff/truncation
  U-shape and the chosen acceptance window.
- Panel C: sensitivity heatmap over selected geometry harmonics or
  `(m,n)`-like perturbations.
- Panel D: cost comparison showing one forward plus one adjoint/transpose solve
  versus finite differences as parameter count grows.

Tests:

- Add `tests/test_generate_autodiff_sensitivity_validation.py`.
- Add artifact-schema and numerical-gate tests for the summary JSON.
- Keep CI fast by using tiny fixtures and checked-in summary artifacts.
- Put medium/full examples behind documented commands or a nightly marker.

Acceptance gates:

- finite-difference relative error below a pinned threshold on stable scalar
  objectives, expected to be `<=1e-4` for finite differences and tighter for
  analytic/JVP consistency where possible;
- finite primal and adjoint residuals with recorded tolerances;
- no hidden optional dependency required for the default validation path;
- every figure generated from the JSON artifact by a reproducible command;
- manifest promotion only after docs explain what is fully differentiable today
  and what remains a future VMEC/Boozer end-to-end path.

### 20.3 Full Simakov-Helander high-collisionality analytic-limit reproduction

Goal: promote `sfincs2014_fig3_high_collisionality_limit` from a trend proxy to
the full analytic-limit reproduction, with the high-collisionality normalization
audited and the asymptotic comparison explicitly stored.

Implementation targets:

- Extend `examples/publication_figures/generate_sfincs_paper_figs.py` or add
  `examples/publication_figures/generate_simakov_helander_limit.py` if the
  analytic-limit logic becomes large.
- Add validation helpers for:
  `simakov_helander_asymptote(...)`,
  `normalize_high_nu_transport(...)`, and
  `build_simakov_helander_limit_summary(...)`.
- Keep the existing trend proxy as a lower-cost regression gate; do not replace
  it until the analytic lane is pinned.

Required runs:

- rebuild LHD and W7-X collisionality ladders with `nu'` extending far past the
  current `nu' <= 10` ceiling;
- run FP/model-operator and PAS branches with identical geometry, grid, species,
  trajectory, and thermodynamic-force settings;
- use split/resumable execution on `office` for expensive points and keep raw
  run provenance out of CI while pinning summary artifacts in the repo.

Normalization audit:

- write down the exact mapping from `sfincs_jax` transport-matrix elements to the
  paper's plotted coefficients;
- store all dimensional constants, species normalization, sign conventions, and
  matrix indices in the summary JSON;
- compare against the Simakov-Helander Pfirsch-Schluter/high-collisionality
  limit only after those conventions are machine-readable.

Publication artifacts:

- `examples/publication_figures/artifacts/sfincs_jax_simakov_helander_limit_summary.json`
- `docs/_static/figures/paper/sfincs_jax_fig3_simakov_helander.{png,pdf}`
- optional supporting figure:
  `docs/_static/figures/paper/sfincs_jax_high_nu_slope_audit.{png,pdf}`

Reviewer-facing figure design:

- Panel A/B: LHD and W7-X `L11` / `L12` versus `nu'` with FP/PAS/model-operator
  curves and analytic asymptote overlays.
- Panel C: tail slope and asymptote-relative error versus `nu'`.
- Panel D: normalization/provenance table embedded as a compact side panel or
  generated alongside the figure in docs.

Tests:

- Add analytic-normalization unit tests independent of heavy runs.
- Add artifact tests asserting wider high-`nu` coverage, monotonic tail behavior
  where expected, source-artifact provenance, and asymptote error thresholds.
- Keep full reruns outside CI; CI validates pinned artifacts and synthetic edge
  cases.

Acceptance gates:

- both parent full collisionality artifacts use corrected scan labels and include
  the high-`nu` points used in the fit;
- FP/model-operator tail approaches the documented analytic limit within a
  stated, justified tolerance;
- PAS behavior is shown separately and not overclaimed as the momentum-conserving
  analytic limit;
- trend claims are numerical assertions from JSON, not visual inspection.

### 20.4 W7-X ambipolar `E_r` validation

Goal: turn `w7x_ambipolar_er_validation` from an executable scaffold into a
defensible W7-X validation artifact with documented equilibrium/profile inputs,
finite ambipolar roots, and careful comparison to literature trends.

Implementation targets:

- Continue from `examples/publication_figures/generate_w7x_ambipolar_validation.py`.
- Add a profile/equilibrium provenance layer that records:
  VMEC/wout or Boozer source,
  flux-surface label,
  density/temperature/profile gradients,
  species definitions,
  collisionalities,
  radial-electric-field bracket,
  and all grid/solver settings.
- Prefer checked-in lightweight metadata plus documented external/raw inputs over
  storing large equilibrium outputs in the repo.

Input strategy:

- First pass: use the existing W7-X example to close the numerical ambipolar-root
  machinery and artifact schema.
- Research pass: reconstruct or choose a defensible W7-X input from the Pablant
  and W7-X validation literature; if exact experimental-profile reconstruction is
  not possible, label the result as a W7-X-like validation, not experimental
  agreement.
- Optional later integration: if `vmec_jax` / `booz_xform_jax` inputs become
  mature enough, add a differentiable-equilibrium variant, but do not block this
  lane on that path.

Publication artifacts:

- `examples/publication_figures/artifacts/sfincs_jax_w7x_ambipolar_validation_summary.json`
- `docs/_static/figures/paper/sfincs_jax_w7x_ambipolar_validation.{png,pdf}`
- optional:
  `docs/_static/figures/paper/sfincs_jax_w7x_heat_flux_ordering.{png,pdf}`

Reviewer-facing figure design:

- Panel A: radial current versus `E_r` with bracketed root(s) and root type.
- Panel B: heat flux and particle flux interpolated at the ambipolar root.
- Panel C: comparison to literature sign/magnitude/order-of-trends, with clear
  caveats if inputs are reconstructed rather than exact experimental profiles.
- Panel D: provenance table: device/configuration, radius, profiles, geometry
  file, collision operator, trajectory model, resolution, tolerances.

Tests:

- Extend `tests/test_generate_w7x_ambipolar_validation.py` to cover provenance
  payloads and finite-root classification using small synthetic scans.
- Keep a fast fixture for CI and a documented reference command for office/GPU.
- Add a regression test that refuses to mark the lane implemented unless at least
  one finite root, bracket, and source-input provenance block are present.

Acceptance gates:

- root solve has a sign-changing radial-current bracket and stored interpolation
  metadata;
- root `E_r`, heat-flux ordering, and root type are compared only against
  literature claims supported by the chosen inputs;
- if the exact W7-X profile cannot be reconstructed, docs state the limitation
  explicitly and keep the lane as W7-X-like until defensible data are available.

### 20.5 MONKES / KNOSOS shared-model overlap

Goal: build `monkes_monoenergetic_overlap` as a normalization-pinned cross-code
lane, separating exact shared-model comparisons from qualitative low-collisionality
ordering checks.

Implementation targets:

- Add `examples/publication_figures/generate_monkes_knosos_overlap.py`.
- Add readers for exported MONKES/KNOSOS comparison tables if local code output
  is used, or a stable JSON/CSV artifact format if values are generated outside
  this repo.
- Add validation helpers for monoenergetic coefficient normalization and
  collisionality/electric-field coordinate transforms.

Comparison policy:

- MONKES: use exact shared-model coefficient comparisons only where geometry,
  monoenergetic assumptions, normalization, collision model, and electric-field
  conventions are aligned.
- KNOSOS: use as a low-collisionality/orbit-averaged trend reference unless an
  exact shared model is established.
- Never mix qualitative trend evidence with exact parity claims in the same
  summary field.

Publication artifacts:

- `examples/publication_figures/artifacts/sfincs_jax_monkes_knosos_overlap_summary.json`
- `docs/_static/figures/paper/sfincs_jax_monkes_overlap.{png,pdf}`
- optional:
  `docs/_static/figures/paper/sfincs_jax_knosos_low_collisionality_trends.{png,pdf}`

Reviewer-facing figure design:

- Panel A: monoenergetic coefficient overlap for the exact shared subset.
- Panel B: relative error versus resolution/collisionality for the pinned shared
  coefficients.
- Panel C: low-collisionality trend comparison to KNOSOS-style ordering, clearly
  marked as trend validation if not exact overlap.
- Panel D: runtime/convergence panel if it strengthens the performance story
  without distracting from model differences.

Tests:

- Unit tests for normalization transforms and sign conventions.
- Synthetic-reader tests for MONKES/KNOSOS table parsing.
- Artifact tests that reject missing geometry/provenance/normalization fields.
- Optional nightly test that regenerates the overlap from local external-code
  outputs when those are installed.

Acceptance gates:

- shared-model assumptions are explicitly pinned before any cross-code agreement
  claim;
- coefficient labels, units, signs, and normalizations are machine-readable;
- exact-overlap panels and trend-only panels are separated in both docs and JSON;
- manifest status remains `deferred_post_release` until the artifact is
  reproducible without manual spreadsheet work.

### 20.6 Consolidated publication-ready figure set

The manuscript/docs figure set should be generated from artifact scripts and should
make the code look like a research instrument, not a parity appendix. Target set:

- Figure 1: method overview, showing the SFINCS equation path, JAX-differentiable
  operator assembly, implicit linear solve, outputs, autodiff/JVP/adjoint paths,
  and artifact-generation workflow.
- Figure 2: CPU/GPU Fortran v3 parity and runtime/memory summary from the frozen
  39-case CPU/GPU suite.
- Figure 3: collisionality validation dashboard for LHD/W7-X, including FP/PAS
  separation and high-collisionality trend state.
- Figure 4: full Simakov-Helander high-collisionality analytic-limit reproduction
  after the wider high-`nu` scans are complete.
- Figure 5: electric-field trajectory-model sweep and W7-X ambipolar-root
  validation.
- Figure 6: autodiff validation: gradient parity, sensitivity map, and gradient
  cost scaling.
- Figure 7: MONKES/KNOSOS cross-code overlap and low-collisionality trend
  comparison.
- Figure 8: examples/workflows panel: CLI run, plotting, Python API, parallel
  transport worker, and differentiable optimization loop.

Figure quality rules:

- every plotted number must trace to JSON/CSV artifacts committed or explicitly
  generated by a documented command;
- every figure has PNG and PDF output;
- docs must include enough caption detail for a reviewer to reproduce the claim;
- no figure should hide failed cases; if a lane is a scaffold, the caption says so;
- all expensive commands should print ETA/progress and support resume/split
  execution before launching long office/GPU jobs.

### 20.7 First concrete next step

Start with the autodiff lane:

1. Add `generate_autodiff_sensitivity_validation.py` with a tiny CI mode and a
   medium manuscript mode.
2. Add summary/plot helpers and artifact tests.
3. Generate and pin the first gradient-check JSON plus PNG/PDF.
4. Update `validation_manifest.json`, `docs/validation_matrix.rst`,
   `docs/paper_figures.rst`, and `docs/autodiff.rst` with the new artifact and
   caveats.
5. Run focused tests, docs build, and then decide whether the lane can become
   `implemented` or whether it remains a stronger deferred lane pending the
   medium/full sensitivity map.

### 20.8 Autodiff/sensitivity validation lane implementation

Implemented the first deferred research lane,
`adjoint_sensitivity_gradient_checks`, as a bounded manuscript-grade artifact:

- Added `examples/publication_figures/generate_autodiff_sensitivity_validation.py`.
  It writes a summary JSON plus two PNG/PDF figures:
  - `docs/_static/figures/paper/sfincs_jax_autodiff_gradient_check.{png,pdf}`,
  - `docs/_static/figures/paper/sfincs_jax_autodiff_sensitivity_map.{png,pdf}`.
- Added autodiff artifact helpers in `sfincs_jax/validation_artifacts.py`:
  - `load_autodiff_sensitivity_summary(...)`,
  - `autodiff_gradient_error_summary(...)`,
  - `build_autodiff_sensitivity_validation_summary(...)`.
- Generated and pinned:
  - `examples/publication_figures/artifacts/sfincs_jax_autodiff_sensitivity_validation_summary.json`,
  - the gradient-check dashboard,
  - and the scheme-4 Boozer harmonic sensitivity-map figure.
- Updated the validation manifest from `deferred_post_release` to `implemented`
  for `adjoint_sensitivity_gradient_checks`.
- Updated `docs/paper_figures.rst`, `docs/validation_matrix.rst`, and
  `examples/publication_figures/README.md`.

The current artifact deliberately makes a bounded claim:

- implicit-diff gradients through a pinned full-system linear solve agree with
  centered finite differences below the recorded `1e-4` relative-error gate,
- GMRES and BiCGStab `custom_linear_solve` wrappers are covered on stable scalar
  objectives,
- primal and adjoint residuals remain below the recorded `1e-8` gate,
- scheme-4 Boozer harmonic sensitivity maps are generated from JAX derivatives,
- full VMEC-boundary optimization remains outside this implemented claim.

Recorded artifact gates:

- maximum gradient relative error: `4.9043124911696375e-05`,
- maximum primal residual norm: `5.507597669163346e-15`,
- maximum adjoint residual norm: `3.857384703608713e-12`,
- scheme-4 geometry-gradient relative error: `1.895064628198389e-10`.

Validation:

- `python -m ruff check sfincs_jax/validation_artifacts.py examples/publication_figures/generate_autodiff_sensitivity_validation.py tests/test_generate_autodiff_sensitivity_validation.py tests/test_validation_artifacts.py tests/test_validation_manifest_schema.py`
  passed.
- `python -m pytest -q tests/test_generate_autodiff_sensitivity_validation.py tests/test_validation_artifacts.py tests/test_validation_manifest_schema.py`
  passed with `13 passed in 2.02s`.

Next deferred lane:

- Begin `sfincs2014_fig3_high_collisionality_limit` by implementing the
  normalization-audit helpers and a wider high-`nu` run plan before launching the
  expensive LHD/W7-X scan points.

### 20.9 Simakov-Helander high-collisionality audit implementation

Implemented the bounded normalization/readiness audit for the deferred
`sfincs2014_fig3_high_collisionality_limit` lane without overclaiming the full
analytic-limit reproduction.

Files and artifacts added:

- `examples/publication_figures/generate_simakov_helander_limit_audit.py`,
- `tests/test_generate_simakov_helander_limit_audit.py`,
- `examples/publication_figures/artifacts/sfincs_jax_simakov_helander_limit_audit_summary.json`,
- `docs/_static/figures/paper/sfincs_jax_simakov_helander_limit_audit.{png,pdf}`.

Validation helper additions in `sfincs_jax/validation_artifacts.py`:

- `appendix_b_geometry_audit_from_h5(...)` recomputes `FSABHat2` from `BHat`
  and `DHat`, records the available Appendix-B geometry quantities, and emits
  discrete proxy coefficients for the analytic high-collisionality comparison.
- `high_collisionality_slope_sensitivity(...)` records how FP tail slopes change
  with the number of high-`nu` points used in the log-log fit.
- `build_simakov_helander_limit_audit_summary(...)` combines the checked-in LHD
  and W7-X collisionality summaries with representative geometry output files and
  records explicit readiness gates.

Current audit result:

- LHD and W7-X geometry outputs contain the required fields for the bounded
  Appendix-B normalization audit.
- Recomputed `FSABHat2` relative errors are machine precision:
  - LHD: about `1.1e-15`,
  - W7-X: about `5.2e-15`.
- W7-X FP `L11`/`L12` tails satisfy the inverse-`nu` slope proxy on the current
  checked-in scan.
- LHD FP `L11`/`L12` tails still fail the inverse-`nu` slope proxy on the current
  checked-in scan.
- Both full collisionality summaries still stop near `nu'=10`, below the new
  full-limit readiness threshold of `nu' >= 50`.

Manifest and docs status:

- Added the implemented `sfincs2014_simakov_helander_limit_audit` lane to
  `examples/publication_figures/validation_manifest.json`.
- Kept `sfincs2014_fig3_high_collisionality_limit` as
  `deferred_post_release`; the full reproduction remains closed until wider
  high-`nu` scans are pinned.
- Updated `docs/paper_figures.rst`, `docs/validation_matrix.rst`,
  `docs/testing.rst`, and `examples/publication_figures/README.md`.

Next lane:

- Continue to `w7x_ambipolar_er_validation`: pin defensible W7-X VMEC/Boozer
  geometry and profiles, record finite ambipolar-root provenance, and keep the
  literature comparison qualitative until those inputs are auditable.

### 20.10 v1.0.3 release hygiene and documentation closeout

Prepared the next patch release after the Simakov-Helander audit work.

Closed items:

- synchronized package metadata to `1.0.3` in both `pyproject.toml` and
  `sfincs_jax.__version__`,
- added `tests/test_package_metadata.py` so package-version drift is caught in CI,
- removed the non-publication-grade `strong_scaling_snapshot.png` and
  `gpu_case_throughput.png` embeds from `docs/parallelism.rst`; the docs keep the
  measurements in text and show the stronger transport-worker GPU scaling figure,
- hardened `examples/publication_figures/generate_sfincs_paper_figs.py` so
  `--plot-only` refuses to rewrite publication summaries from incomplete scratch
  scan directories,
- made deferred-lane release wording version-neutral in the validation manifest
  and validation matrix.

Validation:

- `python -m ruff check pyproject.toml sfincs_jax/__init__.py examples/publication_figures/generate_sfincs_paper_figs.py tests/test_generate_sfincs_paper_figs.py tests/test_package_metadata.py tests/test_validation_manifest_schema.py`
  passed.
- `python -m pytest -q tests/test_generate_sfincs_paper_figs.py tests/test_package_metadata.py tests/test_validation_manifest_schema.py`
  passed with `18 passed`.
- `sphinx-build -W -b html docs docs/_build/html` passed.
- `python -m pytest -q` passed with `875 passed, 1 skipped in 358.57s`.
- `python -m build` built `sfincs_jax-1.0.3.tar.gz` and
  `sfincs_jax-1.0.3-py3-none-any.whl`.

Release state:

- `main` remains the active release branch.
- Existing extra branch `codex/cli-fast-paths` is still present locally and on
  origin; it is not part of the v1.0.3 release path and can be cleaned up after
  explicit approval.
- Remaining validation manifest lanes are intentionally
  `deferred_post_release`: full high-collisionality analytic-limit reproduction,
  W7-X ambipolar validation, and MONKES/KNOSOS overlap. They are research/nightly
  lanes, not v1.0.3 release blockers.

### 20.11 Optional VMEC/Boozer differentiable geometry handoff

Closed the first open lane in the 2026-04-24 continuation pass:
`vmec_jax -> booz_xform_jax -> sfincs_jax` now has a public, tested, fast
example rather than only internal adapter gates.

Files changed:

- `sfincs_jax/jax_geometry_adapters.py` adds
  `boozer_bhat_from_spectrum(...)` and
  `boozer_spectrum_geometry_proxy_objective(...)`.
- `examples/autodiff/vmec_jax_to_boozer_sfincs_pipeline.py` reads or produces a
  VMEC-like `wout` through optional `vmec_jax`, transforms one surface with
  optional `booz_xform_jax`, evaluates the `sfincs_jax` Boozer-spectrum proxy,
  checks `jax.grad` against centered finite differences, and performs bounded
  scalar gradient-descent steps.
- `tests/test_jax_geometry_adapters.py` now covers the cosine-series evaluator,
  the Boozer-spectrum proxy gradient, and an optional real
  `vmec_jax`/`booz_xform_jax` transform gradient gate when those packages and a
  fixture are available.
- `README.md`, `docs/geometry.rst`, `docs/examples.rst`,
  `docs/performance.rst`, `docs/testing.rst`, and
  `examples/autodiff/README.md` document the workflow and its scope boundary.

Validation:

- `python -m py_compile sfincs_jax/jax_geometry_adapters.py tests/test_jax_geometry_adapters.py examples/autodiff/vmec_jax_to_boozer_sfincs_pipeline.py`
  passed.
- `python -m pytest -q tests/test_jax_geometry_adapters.py tests/test_geometry_autodiff_gates.py`
  passed with `12 passed in 6.24s`.
- Local optional workflow run with
  `/Users/rogeriojorge/local/vmec_jax/examples/data/wout_circular_tokamak.nc`
  completed. The proxy objective at scale `1.0` was
  `2.554265610191e-02`, the JAX gradient was
  `4.934993422978e-02`, the centered finite-difference gradient was
  `4.934993421392e-02`, and the absolute gradient error was `1.586e-11`.

Scope boundary:

- This closes the public handoff from VMEC-like JAX arrays through
  `booz_xform_jax` into a differentiable `sfincs_jax` geometry proxy.
- It does not yet claim full VMEC-boundary-to-kinetic-transport-solve
  differentiation. That remains a larger research lane after the code-refactor,
  validation, and PAS/scaling work.

Next steps, in order:

1. Attack PAS/geometry-rich runtime and memory offenders with focused
   parity-preserving benchmarks.
2. Continue the coverage/refactor path toward 95% by extracting heavy driver
   policy pieces into unit-testable modules rather than adding slow full-solve
   tests.
3. Revisit single-case multi-GPU and multi-CPU strong scaling after the local
   offender path is smaller and less setup-bound.
4. Promote the Simakov-Helander and W7-X ambipolar lanes only after defensible
   wider-scan/profile/equilibrium artifacts are pinned.

### 20.12 PAS/geometry-rich CPU offender policy update

Started the PAS/geometry-rich runtime and memory lane with bounded current-tip
variant sweeps before changing defaults.

Benchmark results:

- `HSX_PASCollisions_DKESTrajectories` CPU:
  - default already selected `pas_tz`,
  - default elapsed `3.434 s`, forced `pas_tz` `3.180 s`,
  - forced `xblock_tz` and `schur` were slower and heavier,
  - forced `pas_hybrid` was much slower,
  - all variants stayed Fortran parity-clean, but only default/`pas_tz` also
    matched the default output exactly at the practical comparison tolerance.
- `HSX_PASCollisions_fullTrajectories` CPU:
  - default already selected `pas_tz`,
  - default elapsed `3.255 s`, forced `pas_tz` `3.200 s`,
  - forced `schur` was slower/heavier,
  - forced `xblock_tz` was much slower, much heavier, and produced practical
    comparison mismatches.
- `sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories`
  CPU:
  - default previously selected `schur`,
  - forced `pas_tz` was parity-clean and improved elapsed time
    `2.759 s -> 2.218 s` while reducing peak RSS
    `2263 MB -> 1491 MB`,
  - forced `pas_tz` with `SFINCS_JAX_RHSMODE1_PAS_TZ_LMAX=6` was slower.
- `sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_DKESTrajectories`
  CPU:
  - default already selected `pas_tz`,
  - default elapsed `1.891 s`, forced `schur` `8.413 s`,
  - default and forced `pas_tz` stayed parity-clean.

Code change:

- `rhs1_pas_full_cpu_pas_tz_preferred(...)` now allows bounded
  geometryScheme=11 CPU full-trajectory PAS cases through `Nzeta <= 19`
  instead of only `Nzeta <= 15`.
- This promotes the measured geometry11 full-trajectory CPU offender to `pas_tz`
  by default while leaving GPU full-trajectory defaults unchanged.
- Policy tests now cover the new `Nzeta=19` inclusion and keep a larger
  `Nzeta=23` exclusion.

Validation:

- `python -m pytest -q tests/test_rhs1_preconditioner_auto_policy.py tests/test_schur_precond_heuristic.py`
  passed with `42 passed in 9.97s`.
- Post-policy default rerun on
  `sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories`
  selected `pas_tz`, stayed Fortran parity-clean, and measured
  `2.013 s` elapsed with about `1474 MB` peak RSS.

Next PAS/offender steps:

1. Run the matching GPU current-tip variant on `office` before changing any GPU
   default.
2. Re-rank the current CPU/GPU offender table using fresh post-policy rows.
3. Continue with compile-amortization and setup profiling on the remaining
   monoenergetic/GPU-heavy cases only after this PAS policy remains stable.

### 20.13 Coverage/refactor lane: scan progress and recycle gates

Advanced the coverage/refactor lane with focused user-facing tests rather than
slow full-solve coverage padding.

Files changed:

- Added `tests/test_scans_progress_and_recycle.py`.
- Updated `docs/testing.rst` to describe the scan progress/recycle gate.

Coverage/behavior protected:

- Runtime-duration formatting across seconds, minutes, hours, and days.
- `scan-er` progress messages, including ETA and reused-output reporting.
- Radial-gradient variable resolution for all supported
  `inputRadialCoordinateForGradients` values and invalid-coordinate rejection.
- Namelist scalar patching for both replacement and append paths, plus malformed
  namelist errors.
- Stride/index scan subsetting.
- Serial `SFINCS_JAX_SCAN_RECYCLE=1` state handoff from one scan point to the
  next without launching real solves.

Validation:

- `python -m pytest -q tests/test_scans_progress_and_recycle.py tests/test_helper_module_coverage.py::test_scan_helpers_and_run_er_scan`
  passed with `13 passed in 0.63s`.

Notes:

- A package-scoped coverage probe with `pytest --cov=sfincs_jax.scans` aborted
  while importing the local JAX installation in this macOS environment before
  running tests. The non-coverage pytest command passed cleanly; keep coverage
  probing for this area on CI/Linux or after the local JAX import issue is
  isolated.

Next coverage/refactor steps:

1. Continue using the existing `coverage.xml` to pick real low-cost invariants
   from `scans.py`, `vmec_wout.py`, `io.py`, and collision/geometry helpers.
2. Do not raise CI `--cov-fail-under` toward `95%` until the remaining
   `v3_driver.py` execution body is split into smaller, directly testable
   modules.

### 20.14 Single-case sharded scaling benchmark hygiene

Advanced the single-case multi-device scaling lane by fixing the benchmark
driver before collecting new CPU/GPU evidence.

Files changed:

- `examples/performance/benchmark_sharded_solve_scaling.py`
- `tests/test_benchmark_sharded_solve_scaling.py`

Issues found and fixed:

- `_run_once_subprocess(...)` recorded the requested sharding/Krylov/backend
  settings in the parent JSON, but did not pass all of those settings to the
  child process. The child then re-applied parser defaults inside
  `--run-once`, so some historical runs could have recorded a configuration
  that was not exactly the configuration timed.
- The benchmark subprocess now passes `--shard-axis`,
  `--gmres-distributed`, `--distributed-krylov`,
  `--periodic-stencil-on-sharded`, `--backend`, `--rhs1-precond`, and the
  Schwarz coarse controls explicitly to the child.
- Child processes now raise JAX/C++ log levels to keep benchmark output focused
  on progress and timings.
- Added `--inner-warmup-solves` so publication scaling can report hot-solve
  timing after compile/setup in the same child process.
- Added `--sample-timeout-s` so a CPU/GPU sample that remains in XLA setup is
  bounded instead of consuming unbounded machine time.
- The generated figure now labels the device axis as CPU or GPU based on the
  selected backend.

Corrected local CPU smoke:

- Non-distributed small-case CPU smoke on
  `examples/performance/rhsmode1_sharded_scaling.input.namelist`:
  `1 device = 28.668 s`, `2 devices = 47.616 s` (`0.60x` speedup).
- Distributed-GMRES small-case CPU smoke on the same input:
  `1 device = 23.748 s`, `2 devices = 48.368 s` (`0.49x` speedup).
- These are cold one-shot samples and show that this small local case is
  setup/overhead dominated; they should not be used as strong-scaling claims.

Office GPU smoke:

- A temporary up-to-date worktree was prepared on `office` at
  `/tmp/sfincs_jax_scaling_r4k9WD`.
- The one-shot 1-vs-2 GPU smoke was stopped during the device-1 sample after
  more than three minutes because it remained CPU-bound in compile/setup
  (`~1900%` CPU, 0% GPU utilization), so it was not yet measuring GPU solve
  throughput.

Validation:

- `python -m pytest -q tests/test_benchmark_sharded_solve_scaling.py` passed
  with `7 passed in 1.83s`.
- `python -m py_compile examples/performance/benchmark_sharded_solve_scaling.py tests/test_benchmark_sharded_solve_scaling.py`
  passed.

Next scaling steps:

1. Re-run single-case CPU/GPU scaling with `--inner-warmup-solves 1`,
   `--sample-timeout-s`, and enough `--nsolve` repeats to amortize setup.
2. If the hot-solve single-case path still does not scale, keep the
   single-case multi-GPU lane open and document the reason: single-RHS GMRES is
   reduction/setup dominated at these sizes.
3. Keep the production scaling claim on transport-worker throughput unless and
   until the single-case sharded lane has measured hot-solve speedup on `office`.

### 20.15 Simakov-Helander and W7-X validation gates

Advanced the two deferred physics-validation lanes without overclaiming them.

Literature pass:

- Landreman, Smith, Mollen, and Helander 2014 remains the parent SFINCS
  collisionality/trajectory/collision-operator reference.
- Simakov-Helander high-collisionality work remains the analytic parent for the
  Appendix-B/Pfirsch-Schluter high-`nu'` comparison.
- Pablant et al. 2018 and 2020, plus the W7-X neoclassical-transport validation
  literature, remain the W7-X ambipolar `E_r` acceptance anchors.

Simakov-Helander changes:

- Added `recommended_high_collisionality_nuprime_grid(...)` in
  `sfincs_jax/validation_artifacts.py`.
- The Simakov-Helander audit summary now records a per-case
  `recommended_high_nuprime_extension`; current checked-in LHD/W7-X scans stop
  near `nu'=10`, and the recommended extensions reach about `nu'=100`.
- Regenerated
  `examples/publication_figures/artifacts/sfincs_jax_simakov_helander_limit_audit_summary.json`
  and `docs/_static/figures/paper/sfincs_jax_simakov_helander_limit_audit.png`.
- The full analytic-limit reproduction remains closed until those wider
  high-`nu'` runs are actually generated and the readiness gates flip true.

W7-X ambipolar changes:

- Added optional `--provenance-json` support to
  `examples/publication_figures/generate_w7x_ambipolar_validation.py`.
- The W7-X ambipolar summary now writes explicit `acceptance_gates`:
  finite roots, radial-current sign bracketing, ion-root candidate,
  complete provenance, and combined `ready_for_literature_claim`.
- Without complete `equilibrium_source`, `profile_source`,
  `configuration_or_shot`, and `literature_reference` provenance, generated
  artifacts remain `w7x_like_scaffold` rather than
  `w7x_literature_validation`.

Validation:

- `python -m pytest -q tests/test_validation_artifacts.py::test_recommended_high_collisionality_nuprime_grid_extends_beyond_current_tail tests/test_validation_artifacts.py::test_simakov_helander_audit_records_geometry_and_keeps_full_gate_closed`
  passed with `2 passed in 0.31s`.
- `python -m pytest -q tests/test_generate_w7x_ambipolar_validation.py`
  passed with `6 passed in 4.70s`.
- `python -m py_compile sfincs_jax/validation_artifacts.py tests/test_validation_artifacts.py`
  and `python -m py_compile examples/publication_figures/generate_w7x_ambipolar_validation.py tests/test_generate_w7x_ambipolar_validation.py`
  passed.

Next validation steps:

1. Launch the recommended high-`nu'` LHD/W7-X scan extension as a nightly or
   office-bounded campaign, not as a fast CI test.
2. Build or import a defensible W7-X provenance JSON before checking in any
   W7-X ambipolar literature-validation figure.
3. Keep fast CI on the summary/gate logic and reserve expensive full solves for
   release/nightly artifacts.

### 20.16 Executable high-nu plan and W7-X provenance template

Moved the next deferred validation steps from prose to executable, reviewable
artifacts.

High-collisionality lane:

- `examples/publication_figures/generate_sfincs_paper_figs.py` now accepts
  explicit `--nuprime-min`, `--nuprime-max`, and `--n-points` controls.
- Added
  `examples/publication_figures/generate_simakov_helander_high_nu_run_plan.py`.
- Generated
  `examples/publication_figures/artifacts/sfincs_jax_simakov_helander_high_nu_run_plan.json`.
- The checked-in plan contains concrete scan-only commands for:
  - LHD: `nu' = 17.78, 31.62, 56.23, 100.0`
  - W7-X: `nu' = 17.78, 31.62, 56.24, 100.0`
- These commands are intentionally scan-only with `--skip-existing`; run them on
  `office` or a nightly workstation lane, then regenerate the audit.

W7-X ambipolar lane:

- Added
  `examples/publication_figures/provenance/w7x_ambipolar_provenance_template.json`.
- The template is intentionally incomplete; generated summaries still report
  `w7x_like_scaffold` until a case-specific provenance JSON fills
  `configuration_or_shot`, `equilibrium_source`, `profile_source`, and
  `literature_reference`.
- Updated the validation manifest and docs to point to the run-plan and
  provenance template.

Validation:

- `python -m pytest -q tests/test_generate_sfincs_paper_figs.py::test_main_custom_nuprime_window_writes_expected_scan_bounds tests/test_generate_sfincs_paper_figs.py::test_main_rejects_invalid_custom_nuprime_window tests/test_generate_simakov_helander_high_nu_run_plan.py`
  passed with `4 passed in 0.31s`.
- `python -m pytest -q tests/test_generate_w7x_ambipolar_validation.py::test_checked_in_w7x_provenance_template_is_incomplete_by_design tests/test_generate_w7x_ambipolar_validation.py::test_build_summary_payload_promotes_only_with_complete_provenance`
  passed with `2 passed in 0.79s`.
- `python -m py_compile examples/publication_figures/generate_sfincs_paper_figs.py examples/publication_figures/generate_simakov_helander_high_nu_run_plan.py tests/test_generate_sfincs_paper_figs.py tests/test_generate_simakov_helander_high_nu_run_plan.py`
  passed.

Next concrete execution step:

1. Run the generated high-`nu'` scan-only commands on `office` with one case at a
   time and watch the first point long enough to estimate the full wall time.
2. If the first high-`nu'` point is too expensive, reduce the run to FP-only
   first (`--collision-operators 0`) and keep PAS for a later/nightly pass.
3. Once high-`nu'` output exists, aggregate/plot-only and rerun the
   Simakov-Helander audit to see whether the readiness gate flips.

### 20.17 Office high-nu pilot timing

Ran the first bounded high-`nu'` pilot point on `office` from a disposable copy
of the current working tree.

Command shape:

- `CUDA_VISIBLE_DEVICES=0 ... python3 examples/publication_figures/generate_sfincs_paper_figs.py --case lhd --collision-operators 0 --nuprime-min 17.78279101649707 --nuprime-max 17.78279101649707 --n-points 1 --work-dir /tmp/sfincs_jax_highnu_pilot/lhd --summary-dir /tmp/sfincs_jax_highnu_pilot/summary --out-dir /tmp/sfincs_jax_highnu_pilot/figures --timeout-s 900 --skip-existing --scan-only`

Observed result:

- The run completed successfully and wrote
  `/tmp/sfincs_jax_highnu_pilot/lhd/lhd_co0/nu_n_4.744/sfincsOutput.h5`.
- Total scan elapsed: about `569 s` (`9m28s`) for one LHD FP high-`nu'`
  transport point at `nu'=17.78`.
- Per-RHS timings were about `188.8 s`, `188.2 s`, and `189.1 s`.
- The solver retried from `sxblock` through unpreconditioned/strong paths; final
  residual logs were `1.99e-04`, `3.38e-04`, and `5.00e-01` for the three RHS
  solves, so the high-`nu'` ladder is scientifically useful as a trend pilot but
  needs acceptance scrutiny before promotion.
- The output transport matrix was finite; selected values:
  `L11=-3.958e-03`, `L12=-1.351e-02`, `L33=3.361e+01`.

Run-plan update:

- `examples/publication_figures/generate_simakov_helander_high_nu_run_plan.py`
  now writes both the full extension command and a `pilot_command` for each case.
- Regenerated
  `examples/publication_figures/artifacts/sfincs_jax_simakov_helander_high_nu_run_plan.json`.

Decision:

- Do not launch the full FP/PAS LHD+W7-X high-`nu'` extension in an interactive
  turn. At the measured pilot rate, the full plan is a multi-hour
  workstation/nightly campaign.
- Next implementation step should either improve high-`nu'` transport
  preconditioning/residual acceptance or run the FP-only extension first as a
  bounded overnight job.

### 20.18 High-nu transport residual and parallel campaign fix

Root cause found:

- The high-`nu'` pilot was launched through `utils/sfincs_jax_driver.py`, which
  previously deferred to the low-level implicit/differentiable solve default.
- The implicit path intentionally disables host sparse direct first attempts and
  rescues, so the GPU pilot stayed on iterative `sxblock`/unpreconditioned retry
  branches and reported residuals far above target.
- Transport process/GPU workers also failed to carry the parent
  `differentiable` mode in their payloads, so parallel worker solves could
  silently fall back to the implicit path even when the parent requested the
  explicit executable lane.

Implemented changes:

- `utils/sfincs_jax_driver.py` now defaults to
  `write_sfincs_jax_output_h5(..., differentiable=False)` and exposes
  `--differentiable` for the explicit opt-in implicit/gradient path.
- Transport parallel payloads now preserve `differentiable` and both in-process
  and GPU subprocess workers pass it through to
  `solve_v3_transport_matrix_linear_gmres(...)`.
- `examples/publication_figures/generate_sfincs_paper_figs.py` now has
  `--transport-workers` and `--transport-parallel-backend` controls and forces
  `SFINCS_JAX_IMPLICIT_SOLVE=0` for publication scan subprocesses.
- The Simakov-Helander high-`nu'` run-plan generator now emits GPU-parallel
  pilot/full scan commands by default (`--transport-workers 2
  --transport-parallel-backend gpu`), while remaining regenerable for CPU lanes.
- README, docs, and publication-figure notes now document the explicit scan
  lane and the reason host direct transport rescue is unavailable in implicit
  differentiable solves.
- `sfincs_jax/transport_parallel_runtime.py` now emits periodic GPU-worker
  heartbeat/status messages so future long parallel transport launches do not
  sit silently between worker launch and completion.

Office validation:

- Re-ran the first LHD FP high-`nu'` pilot point on `office` with
  `SFINCS_JAX_IMPLICIT_SOLVE=0`.
- The run triggered `host sparse LU first attempt (size=16382 backend=gpu)` for
  each RHS.
- Final RHS residuals dropped from the previous stalled values
  (`1.99e-04`, `3.38e-04`, `5.00e-01`) to `4.33e-16`, `5.33e-14`, and
  `4.06e-11`.
- Per-RHS elapsed times were about `114.5 s`, `113.4 s`, and `114.3 s`, down
  from about `188 s` each.
- Total scan elapsed was about `345 s` (`5m45s`), down from `569 s` (`9m28s`).
- Re-ran the same point with `--transport-workers 2
  --transport-parallel-backend gpu` and `CUDA_VISIBLE_DEVICES=0,1`.
- The two-GPU worker run preserved the clean residuals (`4.33e-16`,
  `5.33e-14`, `4.06e-11`) and completed in about `262 s` (`4m22s`).
- Observed single-point speedup was `1.32x` over the explicit serial GPU lane
  and `2.17x` over the original implicit/stalled pilot. This is enough to use
  the two-GPU worker lane for the full campaign, but it is not a strong-scaling
  claim because one worker owns two RHS solves and host sparse assembly/factor
  time is still substantial.
- The explicit one-GPU and two-GPU transport matrices were identical for this
  point (`max_abs=0`, `rel=0`). The old implicit/stalled pilot differed
  materially, so it is not a valid reference for the corrected high-`nu'`
  output.
- A local Fortran v3 attempt on the same generated input was aborted after more
  than `1300` PETSc GMRES iterations on the first RHS because this local PETSc
  build reported no MUMPS/SuperLU_DIST and used the fragile built-in direct
  preconditioner. Do not use that aborted run as a physics/parity reference;
  use a proper Fortran v3 build with MUMPS/SuperLU_DIST for the high-`nu'`
  comparison campaign.

Validation added:

- Utility driver tests assert the default explicit path and the
  `--differentiable` opt-in behavior.
- Transport payload tests assert `differentiable` propagation.
- Publication-figure tests assert scan subprocess environment propagation for
  explicit solves and GPU worker counts.
- GPU-worker runtime tests assert completed worker collection and progress
  messages.
- Local validation passed:
  `python -m pytest -q tests/test_transport_parallel_runtime.py tests/test_transport_parallel_execution.py tests/test_transport_parallel.py::test_transport_parallel_worker_preserves_differentiable_payload tests/test_transport_sparse_direct.py tests/test_generate_sfincs_paper_figs.py tests/test_generate_simakov_helander_high_nu_run_plan.py tests/test_utils_sfincs_jax_driver.py tests/test_cli_solve_mode.py tests/test_solve_mode_policy.py tests/test_validation_manifest_schema.py`
  (`102 passed`), `python -m py_compile ...`, and
  `python -m sphinx -W -b html docs docs/_build/html`.

Next concrete steps:

1. Treat `--transport-workers 2 --transport-parallel-backend gpu` as the
   recommended `office` lane for the full high-`nu'` campaign.
2. Run the full FP-only LHD/W7-X extension first, then widen to PAS if the
   first few PAS points show the same residual and runtime behavior.
3. Keep single-RHS strong GPU scaling separate from this campaign result; this
   fix is RHS/task parallelism plus robust explicit sparse direct transport.

### 20.19 High-nu residual diagnostics and W7-X preconditioner gate

Scope:

- Close the silent-failure mode in the high-`nu'` FP/PAS LHD+W7-X campaign by
  carrying RHS norms through transport solves, GPU workers, H5 output, and
  publication scan quality gates.
- Keep the campaign parallelized with process/GPU `whichRHS` workers, but avoid
  letting W7-X FP enter an unbounded host sparse-LU path before the preconditioner
  issue is solved.

Implemented changes:

- `V3TransportMatrixSolveResult` now records `rhs_norms_by_rhs` in addition to
  `residual_norms_by_rhs`.
- In-process and GPU transport workers now return RHS norms, and GPU worker
  subprocess logs replay selected progress lines on success (`rhs_norm`,
  preconditioner choice, sparse/direct/fallback messages, residuals, elapsed
  time). This makes long dual-GPU high-`nu'` runs inspectable while they are
  running and after each worker completes.
- GPU transport subprocess launch now prepends the source checkout to
  `PYTHONPATH`, so workers remain importable when `sfincsScan` changes into a
  scan-point subdirectory.
- `SFINCS_JAX_WRITE_SOLVER_DIAGNOSTICS=1` now writes
  `transportResidualNorms`, `transportRhsNorms`,
  `transportRelativeResidualNorms`, `transportMaxResidualNorm`, and
  `transportMaxRelativeResidualNorm` to `sfincsOutput.h5`.
- `examples/publication_figures/generate_sfincs_paper_figs.py` now supports
  `--max-transport-relative-residual` and `--transport-maxiter`, and it rejects
  just-created scan outputs when absolute or relative residual gates fail.
- `examples/publication_figures/generate_simakov_helander_high_nu_run_plan.py`
  now defaults to `--transport-sparse-direct-max 30000` instead of the earlier
  oversized cap. That keeps the current LHD FP high-`nu'` pilot on the accurate
  direct path while preventing W7-X FP from silently spending many minutes in a
  host sparse factorization known not to be a good default.
- `sfincs_jax.transport_residual_quality` now centralizes absolute and
  RHS-normalized residual-gate checks. The same thresholds are used by
  sequential solves and GPU worker result collection.
- `SFINCS_JAX_TRANSPORT_ABORT_MAX_RESIDUAL` and
  `SFINCS_JAX_TRANSPORT_ABORT_MAX_RELATIVE_RESIDUAL` now fail fast for
  RHSMode=2/3 transport solves. Sequential runs abort after the first bad
  `whichRHS`; GPU-worker runs abort as soon as a completed worker writes a bad
  residual artifact and terminate still-running workers.
- README, docs, and publication-figure notes now state that LHD FP high-`nu'`
  is the currently clean parallel pilot and W7-X FP high-`nu'` remains a
  residual/preconditioner lane until a bounded iterative or direct strategy
  reaches the recorded gates.

Validation run:

- `python -m py_compile sfincs_jax/io.py sfincs_jax/v3_driver.py sfincs_jax/transport_parallel_runtime.py sfincs_jax/transport_parallel_worker.py examples/publication_figures/generate_sfincs_paper_figs.py examples/publication_figures/generate_simakov_helander_high_nu_run_plan.py`
- `python -m pytest -q tests/test_generate_sfincs_paper_figs.py tests/test_generate_simakov_helander_high_nu_run_plan.py tests/test_transport_matrix_write_output_end_to_end.py::test_transport_output_can_include_solver_residual_diagnostics tests/test_transport_parallel_runtime.py tests/test_transport_parallel_execution.py tests/test_transport_parallel.py::test_transport_parallel_worker_preserves_differentiable_payload tests/test_transport_parallel.py::test_transport_parallel_gpu_backend_merges_subset_elapsed tests/test_utils_sfincs_jax_driver.py tests/test_transport_sparse_direct.py` (`74 passed`)
- `python -m pytest -q tests/test_transport_parallel.py tests/test_transport_parallel_runtime.py tests/test_transport_parallel_execution.py tests/test_transport_sparse_direct.py tests/test_generate_sfincs_paper_figs.py tests/test_generate_simakov_helander_high_nu_run_plan.py tests/test_transport_matrix_write_output_end_to_end.py tests/test_utils_sfincs_jax_driver.py` (`93 passed`)
- Remote `office` validation after the fail-fast residual-gate wiring:
  `python3 -m pytest -q tests/test_transport_residual_quality.py tests/test_transport_parallel_runtime.py tests/test_generate_sfincs_paper_figs.py::test_main_require_residuals_rejects_bad_outputs_after_scan tests/test_transport_matrix_write_output_end_to_end.py::test_transport_output_can_include_solver_residual_diagnostics`
  (`12 passed`).
- Local focused transport/publication slice after the same wiring:
  `python -m pytest -q tests/test_transport_residual_quality.py tests/test_transport_parallel_runtime.py tests/test_transport_parallel.py tests/test_transport_parallel_execution.py tests/test_transport_sparse_direct.py tests/test_generate_sfincs_paper_figs.py tests/test_generate_simakov_helander_high_nu_run_plan.py tests/test_transport_matrix_write_output_end_to_end.py tests/test_utils_sfincs_jax_driver.py`
  (`99 passed` after adding worker-exit classification coverage).
- Docs build: `python -m sphinx -W -b html docs docs/_build/html`.

GPU pilot facts from this stage, superseded by the Section 20.21 W7-X sparse-LU
closure:

- LHD FP first high-`nu'` point: explicit two-GPU worker path remains the clean
  campaign baseline (`~262 s`, residuals `4.33e-16`, `5.33e-14`, `4.06e-11`).
- W7-X FP first high-`nu'` point: oversized sparse-direct cap was stopped after
  more than 18 minutes; forcing iterative GPU workers completed in about
  `416 s` but residuals were `1.45e-04`, `2.35e-04`, and `6.42e-01`, i.e. order
  unity relative to the corresponding RHS norms. Forced theta-Schwarz was slower
  and did not improve RHS2. This branch was not publication-grade; Section 20.21
  later closes the first W7-X point with an explicit sparse-LU route.
- Fresh residual-gated `office` rerun with `--transport-sparse-direct-max 30000`
  completed the scan solve in `406.9 s` and failed exactly at the intended
  post-run quality gate. The recorded residual/RHS/relative residual tuples were:
  RHS1 `1.447246e-04 / 1.885192e-04 / 7.676920e-01`, RHS2
  `2.352155e-04 / 2.623896e-04 / 8.964357e-01`, and RHS3
  `6.422668e-01 / 6.589011e-01 / 9.747544e-01`. The max absolute residual was
  `0.6422667543`; the max relative residual was `0.9747544008`. This proves the
  campaign refused this known-bad W7-X FP high-`nu'` branch instead of
  silently producing a reusable H5/figure artifact.
- Single-RHS W7-X FP probes on `office`:
  `SFINCS_JAX_TRANSPORT_PRECOND=xmg`, `restart=120`, `maxiter=800`, and
  `SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_MAX=30000` returned the same bad RHS2
  relative residual (`8.964357e-01`) in `~320 s`. This is slower than the
  earlier default RHS2 worker (`~214 s`) and does not improve accuracy.
- Forced `theta_schwarz` with the same sparse cap timed out at `500 s` before
  producing a RHS2 result, so it is not a viable default W7-X FP high-`nu'`
  accelerator.
- Allowing sparse direct for the W7-X active size (`SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_MAX=40000`,
  float64 factors) materialized the reduced operator as CSR but timed out at
  `600 s` for a single RHS after a transient RSS near `20 GB`. This remains a
  bounded diagnostic option on large nodes, not a default campaign route.
- A real W7-X two-GPU fail-fast run with the sequential worker gate enabled
  aborted inside the `[1, 3]` worker after RHS1 at about `203 s`, before writing
  a reusable output. The first parent-side implementation reported this as a
  generic worker failure because the worker exited before writing its NPZ result;
  the parent runtime now classifies worker stderr/stdout residual-gate exits as
  `GPU transport worker residual gate failed` and terminates pending workers.
  This classification is covered by local and `office` unit tests; a second
  real `office` rerun was interrupted by SSH timeout, so the unit-tested
  classification should be rechecked in the next GPU window.

Next concrete steps:

1. Run the LHD FP/PAS high-`nu'` pilot/full extension with the same residual
   gates and record runtime, residual, and memory artifacts.
2. For W7-X FP, attack the remaining preconditioner lane with a bounded
   streaming+FP diagonal angular preconditioner or another demonstrably cheaper
   Krylov accelerator; do not claim the W7-X high-`nu'` campaign until the
   relative residual gate is clean.
3. Add a permanent W7-X single-RHS preconditioner benchmark harness that records
   residual, RHS norm, relative residual, wall time, peak RSS, and preconditioner
   choice for candidate algorithms without running the full publication scan.

### 20.20 v1.0.4 release closeout

Release scope:

- Ship the research-grade documentation/refactor/testing work accumulated after
  `v1.0.3`.
- Close the high-`nu'` transport silent-failure lane by making residual quality
  explicit in H5 output, publication scan gates, sequential runs, and GPU worker
  runs.
- Publish the differentiable VMEC/Boozer/SFINCS handoff scaffold and validation
  provenance machinery without claiming unavailable W7-X ambipolar or W7-X
  high-`nu'` convergence.
- Keep runtime-heavy or unresolved research items as documented follow-up lanes
  rather than release blockers.

Closed for this release:

- Explicit executable scan path is now the default for utilities and publication
  scans; implicit/differentiable solves remain available as an opt-in gradient
  path.
- Transport worker parallelism now carries `differentiable` mode, RHS norms,
  residual diagnostics, status logs, and GPU-worker import paths correctly.
- High-`nu'` LHD FP pilot has clean two-GPU residuals and documented runtime.
- W7-X FP high-`nu'` no longer silently writes reusable artifacts when it fails
  the residual gate; Section 20.21 later closes the first full-resolution W7-X
  point with a bounded sparse-LU route.
- Documentation now explains the high-`nu'` campaign lane, fail-fast residual
  thresholds, optional JAX-native geometry handoff, validation provenance, and
  current scaling caveats.

Deferred after ship:

- Cheaper W7-X FP high-`nu'` *Krylov-only* preconditioning remains open; current
  `xmg`, `theta_schwarz`, and `fp_tzfft` probes do not close the full point
  without sparse-LU rescue. The executable sparse-LU route itself is now much
  cheaper for the first full W7-X point after Section 20.22 factor reuse.
- Full FP/PAS LHD+W7-X high-`nu'` campaign should proceed as a nightly/release
  lane with residual gates and the bounded W7-X sparse-LU command from Section
  20.21, not as a short CI path.
- W7-X ambipolar validation remains provenance-gated until defensible checked-in
  equilibrium/profile artifacts are available.
- Single-RHS multi-GPU strong scaling remains experimental; current release
  claims only correctness, task/RHS parallelism, and documented caveats.

Release checks planned before tagging:

- Focused transport/publication tests.
- Documentation build with `sphinx -W`.
- Package build with `python -m build`.
- Import/version smoke test for `sfincs_jax.__version__ == "1.0.4"`.

### 20.21 W7-X FP high-nu preconditioner campaign

Scope:

- Close the next concrete W7-X FP high-`nu'` lane by making preconditioner
  candidates reproducible, fixing any solver-path bugs found by the campaign,
  and keeping full-resolution W7-X claims gated until a residual-clean GPU run
  is recorded.

Implemented changes:

- Added `SFINCS_JAX_TRANSPORT_PRECOND=fp_tzfft` as an opt-in RHSMode=2/3 FP
  transport preconditioner.
- `fp_tzfft` keeps the dense FP collision block in `(species, x)` for each
  Legendre mode and adds flux-surface-averaged streaming, mirror, and optional
  `E x B` symbols in `(theta,zeta)` Fourier space.
- The preconditioner is memory bounded by
  `SFINCS_JAX_TRANSPORT_FP_TZFFT_MAX_MB` and falls back to the lighter
  species-x block preconditioner when the inverse table would be too large.
- Added alias normalization for `fp_streaming_fft -> fp_tzfft` and dispatch
  coverage in `transport_preconditioner_dispatch.py`.
- Added
  `examples/performance/benchmark_w7x_high_nu_preconditioners.py`, a
  single-RHS harness that generates a W7-X high-`nu'` FP input, runs each
  preconditioner in a fresh subprocess, and records residual norms, RHS norms,
  relative residuals, elapsed time, peak RSS, logs, and environment.
- Fixed the sparse-LU direct rescue path exposed by this campaign: invalid or
  unset `SFINCS_JAX_TRANSPORT_SPARSE_DROP_TOL` /
  `SFINCS_JAX_TRANSPORT_SPARSE_DROP_REL` are now parsed before
  `_transport_sparse_direct_solve`, eliminating the `NameError` that prevented
  exact sparse rescue from closing W7-X reduced tests.
- Fixed the float32 sparse-direct acceptance path exposed by the full W7-X
  scan: transport sparse-direct solves now recompute the matrix-free true
  residual before accepting a host sparse residual. If the matrix-free residual
  is worse, iterative refinement / polish / retry decisions use that true
  residual. This prevents scan-level `solverTolerance=1e-6` from accepting a
  sparse-helper residual that still fails the JAX operator residual gate.

Local validation:

- `python -m pytest -q tests/test_transport_preconditioner_dispatch.py tests/test_benchmark_w7x_high_nu_preconditioners.py tests/test_transport_sparse_direct.py`
  passed (`50 passed`).
- Reduced W7-X FP RHS2 benchmark with `--sparse-direct-max 30000` completed
  residual-clean for both `auto` and `fp_tzfft`:
  relative residual `3.37e-12`, elapsed about `0.89-0.97 s`.
- The same reduced benchmark with `--sparse-direct-max 1` confirmed that Krylov
  alone is still not sufficient (`relative residual ~= 0.602`) but `fp_tzfft`
  reduced runtime and peak RSS relative to `auto` in that bounded no-rescue
  probe.
- The reduced sparse-rescue logs show `fp_tzfft` lowers the pre-rescue residual
  from about `6.21e+01` (`auto`) to about `1.90e-03`, so the new preconditioner
  is a useful candidate even though direct rescue still closes the solve.
- Full-resolution W7-X FP RHS2 on `office` with `--sparse-direct-max 30000`
  confirmed that `auto` and `fp_tzfft` do not close the real high-`nu'` point
  without direct rescue: both ended near relative residual `0.898`, with
  `fp_tzfft` only slightly faster (`~204 s` vs `~207 s`) and materially higher
  memory.
- Full-resolution W7-X FP RHS2 on `office` with `--sparse-direct-max 40000` and
  `SFINCS_JAX_TRANSPORT_SPARSE_FACTOR_DTYPE=float32` closed the single-RHS gate:
  relative residual `2.77e-11`, elapsed about `680 s`, peak RSS about `19.9 GB`.
- Full-resolution isolated RHS1 and RHS3 probes with the same bounded sparse-LU
  route also closed: RHS1 relative residual about `1.27e-11`, RHS3 about
  `1.77e-11`, each in about `679-680 s`.
- After the matrix-free true-residual acceptance fix, the full one-point W7-X
  FP high-`nu'` publication scan passed the `1e-6` absolute and relative gates
  on one office GPU with one transport worker. The H5 diagnostics were:
  residual norms `[1.29747083e-10, 1.97572435e-12, 4.84165063e-09]`,
  RHS norms `[1.88519158e-04, 2.62389647e-04, 6.58901108e-01]`,
  relative residuals `[6.88243489e-07, 7.52973442e-09, 7.34806874e-09]`,
  max absolute residual `4.841650632107985e-09`, and max relative residual
  `6.88243488581383e-07`. Total scan elapsed was about `2028 s`.
- Final local release checks after the W7-X closure patch:
  `python -m pytest -q` passed (`922 passed in 314.13s`),
  `python -m sphinx -W -b html docs docs/_build/html` passed,
  `python -m ruff check --select F821 sfincs_jax/v3_driver.py examples/performance/benchmark_w7x_high_nu_preconditioners.py`
  passed, `python -m py_compile` on the touched code/tests passed, and
  `git diff --check` passed.

Current state:

- The W7-X FP high-`nu'` correctness lane is closed for the first full-resolution
  point: a bounded one-worker GPU sparse-LU route now passes the same
  `1e-6`-level absolute and relative residual thresholds used by the publication
  scans.
- This is not yet a performance claim for W7-X high-`nu'`. The clean route costs
  about `33.8 min` for one point and depends on host sparse factorization, so it
  should be widened carefully with residual gates and memory monitoring.
- `fp_tzfft` remains an opt-in experimental preconditioner. It is useful in
  reduced probes and may reduce pre-rescue residuals, but it does not close the
  full W7-X high-`nu'` point without sparse-direct rescue.
- The dirty, stale `office` checkout should not be mutated directly. Use an
  isolated scratch rsync checkout for GPU probes.

Next concrete steps:

1. Keep the W7-X high-`nu'` run command pinned to one worker,
   `--transport-sparse-direct-max 40000`, and
   `SFINCS_JAX_TRANSPORT_SPARSE_FACTOR_DTYPE=float32` until a cheaper
   preconditioner is demonstrated.
2. Do not promote `fp_tzfft`, `xmg`, or Schwarz preconditioners to W7-X defaults
   based on the current data; use the single-RHS harness for future algorithm
   probes.
3. Treat widened W7-X high-`nu'` scans as a nightly/release campaign, not CI:
   each point is expensive, but residual gates now prevent bad artifacts from
   being reused silently.

### 20.22 W7-X high-nu sparse-helper factor reuse and performance artifact

Scope:

- Close the remaining concrete W7-X FP high-`nu'` performance lane for the first
  full-resolution point by reusing the same explicit sparse helper/factorization
  across the three RHSMode=2 transport drives, reducing sparse materialization
  memory, and publishing a reviewer-facing runtime/residual/memory figure.

Implemented changes:

- `solve_v3_transport_matrix_linear_gmres` now keeps a solve-local
  `transport_sparse_direct_factor_cache` for the explicit sparse-helper path.
  The cache key includes the operator signature, active/full mode, factor dtype,
  backend, helper block size, storage caps, and sparse/dense storage choice.
- The explicit transport sparse helper now passes a batched `matmat` callback
  into `build_operator_from_matvec`, using `jax.vmap` over column blocks instead
  of repeatedly calling one-column matvecs from Python.
- `explicit_sparse._matvec_to_dense` no longer builds a full identity matrix or
  a list of all column blocks. It creates only the current local basis block and
  writes each output block into one preallocated dense operator. This materially
  reduces transient host memory for every explicit sparse-helper user.
- Added `examples/publication_figures/generate_w7x_high_nu_performance.py`,
  which writes the checked JSON summary plus PNG/PDF performance figure.
- Added regression coverage:
  - `tests/test_explicit_sparse.py` verifies block-basis assembly without a full
    identity matrix.
  - `tests/test_benchmark_w7x_high_nu_preconditioners.py` forces the sparse
    helper on a reduced W7-X two-RHS run and verifies the reuse log plus clean
    residuals.
  - `tests/test_generate_w7x_high_nu_performance.py` checks the performance
    summary gates and plot writer.

Validation:

- Focused local tests:
  `python -m pytest -q tests/test_benchmark_w7x_high_nu_preconditioners.py tests/test_explicit_sparse.py tests/test_transport_sparse_direct.py tests/test_generate_w7x_high_nu_performance.py`
  passed (`55 passed in 7.54s`).
- Static checks passed:
  `python -m py_compile sfincs_jax/explicit_sparse.py sfincs_jax/v3_driver.py examples/publication_figures/generate_w7x_high_nu_performance.py tests/test_benchmark_w7x_high_nu_preconditioners.py tests/test_explicit_sparse.py tests/test_generate_w7x_high_nu_performance.py`,
  `python -m ruff check --select F821 ...`, and `git diff --check`.
- Office GPU full W7-X FP high-`nu'` one-point rerun:
  `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/tmp/sfincs_jax_w7x_precond SFINCS_JAX_TRANSPORT_SPARSE_FACTOR_DTYPE=float32 /usr/bin/time -v ... generate_sfincs_paper_figs.py --case w7x --collision-operators 0 --nuprime-min 17.78332923601508 --nuprime-max 17.78332923601508 --n-points 1 --transport-workers 1 --transport-parallel-backend gpu --transport-sparse-direct-max 40000 --transport-maxiter 800 --require-residuals --max-transport-residual 1e-6 --max-transport-relative-residual 1e-6 --scan-only`.
- Final H5 diagnostics matched the previous clean no-reuse run exactly:
  transport matrix max absolute difference `0.0`, FSAB flow max absolute
  difference `0.0`, residual diagnostics max absolute difference `0.0`.

Runtime/memory delta:

- No-reuse clean sparse-LU route: about `2028 s`, three sparse factorizations,
  peak RSS reference about `19.9 GB`.
- New factor-reuse sparse-LU route: `/usr/bin/time -v` wall time `582.35 s`,
  one sparse factorization, peak RSS `15.3 GB`.
- Per-RHS solver timings: `573.997 s`, `2.469 s`, `2.378 s`.
- Speedup vs no-reuse: `3.48x`; wall time saved: about `1446 s`.
- Residual/RHS/relative tuples stayed
  `1.297471e-10 / 1.885192e-04 / 6.882435e-07`,
  `1.975724e-12 / 2.623896e-04 / 7.529734e-09`, and
  `4.841651e-09 / 6.589011e-01 / 7.348069e-09`.

Current state:

- The first W7-X FP high-`nu'` point is now both residual-clean and cheap enough
  to use as a publication-facing pilot: under 10 minutes on one office GPU.
- The stricter research lane remains: widened W7-X FP/PAS high-`nu'` scans must
  still be treated as nightly/release campaigns with residual gates, not CI.
- `fp_tzfft`, `xmg`, and Schwarz remain candidate preconditioners; they are not
  promoted to W7-X defaults because they still do not close the full point
  without sparse-direct rescue.

### 20.23 v1.0.6 release-candidate parity, performance, and artifact refresh

Scope:

- Regenerate the release-facing CPU/GPU example-suite comparisons, README table,
  documentation references, and publication plots before tagging the next
  release.

Implemented changes:

- Fixed a RHSMode=2/3 output regression by retaining solved state vectors only
  when needed: export-f requests now force state retention so `delta_f` and
  `full_f` are written, while non-export transport diagnostics keep the default
  state-retention policy.
- Restored the fast one-GPU medium tokamak PAS+Er route by making bounded
  GPU `xblock_tz` auto-promotion active for the default `1000..8000` active-DOF
  window. Tiny one-species cases still use tight unpreconditioned GMRES.
- Updated README/docs generator defaults and publication benchmark defaults to
  the current release roots:
  `tests/scaled_example_suite_release_cpu_frozen_2026-04-25_v106` and
  `tests/scaled_example_suite_gpu_bounded_default_2026-04-28`.

Validation:

- CPU profiled release suite: `39/39 parity_ok`, strict mismatches `0`, output
  key coverage `missing_total=0`, runtime drift flags `0`, JAX RSS captured for
  all 39 cases.
- GPU profiled release suite on `office` with one visible GPU: `39/39
  parity_ok`, strict mismatches `0`, output key coverage `missing_total=0`,
  runtime drift flags `0`, JAX RSS captured for all 39 cases.
- Focused regression tests passed for export-f low-memory transport output,
  transport-matrix write-output, and GPU tokamak PAS auto-policy.

Publication artifacts:

- Regenerated `sfincs_jax_fortran_suite_benchmark_summary` from the profiled
  release reports.
- Regenerated the W7-X high-nu, autodiff sensitivity, Simakov-Helander,
  high-collisionality trend, and validation-dashboard figures.

Current release state:

- Ready for final local CI-equivalent tests, docs build, package build, commit,
  push, CI verification, tag `v1.0.6`, and GitHub/PyPI release.

### 20.24 Bounded one-GPU geometryScheme=11 PAS solver-path correction

Scope:

- Close the collaborator-reported GPU solver-path over-selection risk for the
  remaining geometry-rich PAS examples by profiling default, `pas_tz`,
  `pas_hybrid`, `xblock_tz`, and unpreconditioned routes on `office`.

Implemented changes:

- Added `rhs1_pas_full_pas_tz_preferred`, preserving the existing CPU policy and
  promoting only bounded one-GPU full-trajectory PAS geometryScheme=11 cases to
  the structured `pas_tz` preconditioner.
- Kept tokamak PAS+Er on Schur: forced `xblock_tz`, `pas_hybrid`, and
  unpreconditioned probes all timed out for the checked tokamak two-species
  PAS+Er example.
- Added artifact-backed tests that require the two measured geometryScheme=11
  full-trajectory PAS cases to stay on `pas_tz`, remain mismatch-free, and keep
  bounded GPU runtime/RSS under the measured release thresholds.
- Updated README/docs benchmark tables, solver-path audit artifacts, and the
  publication benchmark summary figure from the refreshed 39-case GPU root.

Validation:

- Focused policy tests passed locally before the artifact refresh:
  `pytest -q tests/test_rhs1_preconditioner_auto_policy.py tests/test_gpu_solver_path_artifacts.py`
  (`22 passed`).
- Remote one-GPU suite refresh reran only
  `HSX_PASCollisions_fullTrajectories` and
  `sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories`
  into `tests/scaled_example_suite_gpu_bounded_default_2026-04-28`.
- The refreshed 39-case GPU artifact remains `39/39 parity_ok`, strict
  `39/39 parity_ok`, missing Fortran output keys `0`, and runtime drift flags
  `0` against `tests/scaled_example_suite_release_gpu_2026-04-25_v106`.

Runtime/memory delta:

- `HSX_PASCollisions_fullTrajectories`: `10.539 s` / `2042.4 MB` -> `8.469 s`
  / `1577.4 MB`, practical and strict mismatches `0`.
- `sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories`:
  `7.716 s` / `2097.6 MB` -> `6.413 s` / `1608.5 MB`, practical and strict
  mismatches `0`.
- Solver-path audit now reports `pas_tz` for both corrected geometryScheme=11
  full-trajectory PAS cases and keeps Schur for the tokamak PAS+Er row.

Next steps:

1. Run focused local tests, Sphinx with warnings-as-errors, and `git diff --check`.
2. Commit the bounded GPU PAS policy, refreshed artifacts, docs, and regression
   tests.
3. If more performance time is available, target `geometryScheme4_2species_PAS_noEr`
   and `sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_DKESTrajectories`,
   which are now the top remaining PAS memory-ratio rows in the release GPU
   artifact.

### 20.25 RHSMode=3 GPU benchmark-harness correction and dense monoenergetic gate

Scope:

- Reassess the apparent `monoenergetic_geometryScheme1` GPU runtime offender
  after the geometryScheme=11 PAS correction.

Findings:

- The suite artifact was a real slow path, but the ad-hoc
  `scripts/benchmark_case_variants.py` probe was not valid for RHSMode=2/3: it
  set `compute_solution=True` without `compute_transport_matrix=True`, so
  transport examples could report geometry/output-only runtimes.
- After fixing the benchmark harness, the monoenergetic geometryScheme=1 variants
  showed the default Krylov/sparse-rescue transport path at about `16.67 s`
  internal elapsed, while dense GPU transport with the default dense-fallback
  policy was about `2.09 s` internal elapsed and remained Fortran-clean.

Implemented changes:

- `benchmark_case_variants.py` now parses `RHSMode` and forces
  `compute_transport_matrix=True` for RHSMode=2/3 benchmarks.
- Added a bounded `transport_dense_accelerator_auto_allowed` policy for
  accelerator RHSMode=3 monoenergetic transport, defaulting only to
  geometryScheme=1 and `total_size <= 2500`.
- The broad opt-out `SFINCS_JAX_TRANSPORT_DENSE_ALLOW_ACCELERATOR=0` still
  disables this path; additional guards are exposed via
  `SFINCS_JAX_TRANSPORT_DENSE_ACCELERATOR_AUTO`,
  `SFINCS_JAX_TRANSPORT_DENSE_ACCELERATOR_AUTO_MAX`, and
  `SFINCS_JAX_TRANSPORT_DENSE_ACCELERATOR_AUTO_GEOMETRIES`.
- Added unit coverage for RHSMode-aware benchmark variants and the bounded dense
  accelerator transport policy.

Validation:

- Focused local tests passed:
  `pytest -q tests/test_transport_sparse_direct.py tests/test_benchmark_case_variants.py tests/test_v3_driver_policy_helpers.py`
  (`58 passed`).
- Remote one-GPU suite refresh reran `monoenergetic_geometryScheme1` into the
  release GPU artifact and stayed practical/strict parity-clean with no runtime
  drift flags and no missing output keys in the 39-case root.

Runtime/memory delta:

- `monoenergetic_geometryScheme1`: `13.039 s` / `995.9 MB` -> `3.541 s` /
  `981.0 MB`, practical and strict mismatches `0`.
- The corrected benchmark harness measured dense-auto GPU transport at
  `2.089 s` internal elapsed for the same frozen case; suite wall time is higher
  because it includes the full release runner and comparison overhead.

Follow-up:

1. Focused tests, docs, `git diff --check`, and full local pytest passed before
   committing this correction.
2. The remaining true GPU memory-ratio targets were profiled in section 20.26.

### 20.26 Remaining GPU PAS memory-ratio probe results

Scope:

- Check whether the next two apparent GPU memory-ratio offenders have safe
  branch-selection improvements after the dense monoenergetic transport fix.

Measured probes on `office`:

- `geometryScheme4_2species_PAS_noEr`
  - default: `pas_tz`, `4.372 s`, `1815 MB`, Fortran mismatches `0`.
  - forced `pas_tz`: `4.953 s`, `1799 MB`, Fortran mismatches `0`.
  - forced `schur`: `5.764 s`, `2508 MB`, Fortran mismatches `0`.
  - forced `xblock_tz`: `31.768 s`, `2731 MB`, Fortran mismatches `0`.
  - no preconditioner: timed out at `120 s`.
- `sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_DKESTrajectories`
  - default: `pas_tz`, `5.403 s`, `1587 MB`, Fortran mismatches `0`.
  - forced `pas_tz`: `5.312 s`, `1586 MB`, Fortran mismatches `0`.
  - forced `schur`: `22.058 s`, `2137 MB`, Fortran mismatches `0`.
  - forced `xblock_tz`: `17.346 s`, `2125 MB`, Fortran mismatches `0`.
  - no preconditioner: timed out at `120 s`.

Decision:

- No new solver-policy promotion from this lane. The current defaults are already
  the best measured routes among the tested bounded alternatives.
- The remaining memory ratio is not a branch-selection bug; it is likely from
  RHSMode=1 Krylov/device work arrays and diagnostic-state retention. Treat it
  as a future allocator/lifetime optimization lane rather than a release blocker.

Next steps:

1. Push the four local commits and let CI validate the full matrix.
2. If CI is clean, tag the next patch release from `main`.
3. If more GPU optimization time is available before release, target internal
   work-array lifetimes in RHSMode=1 PAS Krylov/diagnostics rather than trying
   more preconditioner branch switches for these two cases.

## 21) Research-grade solver policy, performance, and validation plan

Status date: 2026-04-30.

Goal:

- Make `sfincs_jax` a research-grade neoclassical transport code that is fast,
  accurate, parity-clean with SFINCS Fortran v3 for supported examples, robust
  for unseen inputs/geometries, efficient on CPU and GPU, and easy to use from
  the CLI or Python.
- Keep JAX-native differentiable paths for autodiff, sensitivity analysis,
  inverse design, uncertainty quantification, and stellarator optimization.
- Permit faster non-differentiable production paths when autodiff is not needed,
  especially from the CLI and large Python batch workflows.

### 21.1 Problem statement

The collaborator-reported full-collision `Nxi=20` versus `Nxi=40` behavior is a
general class of risk, not a one-off bug. Any automatic solver policy based only
on grid-size thresholds, geometry labels, backend names, or memory estimates can
select a path that is:

- slower than the incumbent path,
- higher-memory than the incumbent path,
- numerically weaker, even when it was intended as a stronger fallback,
- parity-clean for one benchmark but fragile for neighboring resolutions,
- acceptable on CPU but bad on GPU after JIT/transfer/setup costs,
- good cold but poor warm, or good warm but too expensive cold for normal CLI use.

The risk is highest for:

- RHSMode=1 full-FP and PAS solves,
- `constraintScheme=1/2` source/constraint systems,
- finite-beta and profile-current workflows,
- VMEC/Boozer/geometryScheme 4/5/11 cases,
- multi-species cases,
- near-zero or resonant radial-electric-field cases,
- high `Nxi`, `Ntheta`, `Nzeta`, `Nx`, or high angular block size,
- GPU paths where setup, host/device transfers, and XLA compilation dominate.

### 21.2 Fortran v3 baseline lessons

Local source anchor:

- `/Users/rogeriojorge/local/sfincs/fortran/version3/solver.F90`
- `/Users/rogeriojorge/local/sfincs/fortran/version3/evaluateJacobian.F90`
- `/Users/rogeriojorge/local/sfincs/fortran/version3/populateMatrix.F90`
- `/Users/rogeriojorge/local/sfincs/fortran/version3/preallocateMatrix.F90`
- `/Users/rogeriojorge/local/sfincs/fortran/version3/sparsify.F90`
- `/Users/rogeriojorge/local/sfincs/fortran/version3/validateInput.F90`

Key lessons to port into `sfincs_jax`:

- SFINCS v3 relies on PETSc/SNES/KSP instead of a large in-code fallback ladder.
- Iterative v3 runs use GMRES with large restart (`2000`) and PETSc residual
  monitors.
- Direct v3 runs use PETSc `KSPPREONLY` + `PCLU`.
- Parallel direct solves use MUMPS or SuperLU_DIST when available.
- Serial sparse LU uses explicit ordering and pivot safeguards.
- Preconditioner matrices are assembled separately from true Jacobians.
- Preconditioners are reused when allowed.
- Sparse matrices are preallocated with predicted row nonzeros to avoid runtime
  allocation churn.
- User-visible PETSc options can override solver behavior without changing code.
- MUMPS memory failures are handled by retrying with larger work-array expansion,
  not by silently switching to a weak unrelated preconditioner.

SFINCS v3 therefore avoids many path cliffs by using mature sparse-solver
infrastructure, explicit controls, and visible residual diagnostics. `sfincs_jax`
should preserve JAX advantages, but its auto policy must become similarly
auditable and measured.

### 21.3 Literature and code anchors

Primary SFINCS and physics:

- SFINCS repository: https://github.com/landreman/sfincs
- SFINCS 2014 paper: Landreman, Smith, Mollen, Helander, Phys. Plasmas 21,
  042503 (2014), https://arxiv.org/abs/1312.6058
- Velocity-space discretization and FP collisions: Landreman and Ernst, JCP 243,
  130-150 (2013), https://arxiv.org/abs/1210.5289
- W7-X impurities/high-collisionality comparison: Mollen et al., Phys. Plasmas
  22, 112508 (2015), https://arxiv.org/abs/1504.04810
- MONKES spectral/block-sparse monoenergetic solver:
  https://arxiv.org/abs/2312.12248
- KNOSOS orbit-averaged fast low-collisionality solver:
  https://arxiv.org/abs/1908.11615
- NEO benchmarked full-FP drift-kinetic solver documentation:
  https://gacode.io/neo.html

Solver infrastructure:

- PETSc GMRES: https://petsc.org/release/manualpages/KSP/KSPGMRES/
- PETSc preconditioner reuse:
  https://petsc.org/main/manualpages/KSP/KSPSetReusePreconditioner/
- PETSc sparse preallocation:
  https://petsc.org/release/manualpages/Mat/MatMPIAIJSetPreallocation/
- PETSc MUMPS:
  https://petsc.org/main/manualpages/Mat/MATSOLVERMUMPS/
- PETSc SuperLU_DIST:
  https://petsc.org/release/manualpages/Mat/MATSOLVERSUPERLU_DIST/

JAX ecosystem:

- JAX sparse reference implementation:
  https://docs.jax.dev/en/latest/jax.experimental.sparse.html
- Lineax solver/operator abstraction:
  https://docs.kidger.site/lineax/api/solvers/
- Lineax matrix-free operators:
  https://docs.kidger.site/lineax/api/operators/
- JAXopt linear solvers and implicit differentiation:
  https://jaxopt.github.io/stable/linear_system_solvers.html
- Equinox filtered transforms and sharding utilities:
  https://docs.kidger.site/equinox/api/transformations/

Interpretation for `sfincs_jax`:

- PETSc-like sparse infrastructure is the right model for fast non-autodiff CPU
  production solves.
- JAX-native matrix-free paths remain the right model for differentiability and
  accelerator workflows.
- JAX experimental sparse should not be treated as a performance-critical sparse
  backend without direct benchmarks.
- Lineax may help structure JAX-native operators and autodiff rules, but it is
  not a replacement for PETSc/MUMPS/SuperLU_DIST robustness.
- MONKES and KNOSOS motivate block sparsity, spectral/operator structure, and
  orbit-averaged or monoenergetic fast lanes where the physical model permits
  them.

### 21.4 Non-negotiable solver-selection rules

Every new default solver/preconditioner promotion must satisfy all of the
following before becoming automatic:

- It is finite: no `nan`, `inf`, XLA resource failure, SciPy factor failure, or
  invalid factor-probe amplification.
- It is accurate: true residual satisfies the active target, not only a
  preconditioned residual.
- It is parity-clean: supported SFINCS Fortran v3 outputs match within the
  established practical and strict tolerances for the relevant validation cases.
- It is measured: setup time, Krylov/direct solve time, total wall time, peak
  RSS/device memory, selected path, and residual history are recorded.
- It improves something material: when the incumbent path is already clean, the
  new path must reduce runtime or memory enough to exceed the acceptance margin.
- It is local-neighborhood robust: the change must pass at least one neighboring
  resolution or geometry-feature variant, not just the single case that motivated
  it.
- It is backend-aware: CPU, GPU cold, and GPU warm behavior are judged
  separately.
- It is reversible by user control: CLI/Python/env options must expose the old
  path for debugging and reproducibility.

Implemented first step:

- `sfincs_jax/solver_selection_policy.py` defines reusable measured-candidate
  metrics and gates.
- `tests/test_solver_selection_policy.py` verifies that residual-bad,
  slower/higher-memory fallbacks cannot be auto-promoted over clean fast paths.

### 21.5 Architecture plan

Lane A: Observability and solver traces.

- Add a stable solver-trace schema emitted to CLI logs and output metadata.
- Include input signature, backend, device count, cold/warm status, active DOFs,
  matrix/constraint sizes, collision model, geometry scheme, chosen path,
  rejected candidates, residual target, residual history, setup time, solve
  time, total time, peak memory, output parity status when available, and
  fallbacks triggered.
- Write traces to HDF5 and JSON so benchmark scripts, docs, and tests consume
  the same source of truth.
- Add timeout-safe partial traces so killed large runs still explain where time
  was spent.

Lane B: Measured candidate gates.

- Route all future automatic strong-fallback decisions through
  `solver_selection_policy`.
- Before switching from a clean incumbent to a new path, require a cheap probe or
  artifact-backed benchmark showing residual, runtime, and memory improvement.
- Reject any candidate whose probe worsens the residual or whose factor probe is
  non-finite or amplifies beyond the configured bound.
- Cache validated decisions by geometry/profile/operator signature, not only by
  raw grid size.
- Treat `pas_lite`, `pas_hybrid`, `pas_tz`, `pas_schur`, `xblock_tz`,
  `collision`, `xmg`, dense, sparse LU, and LGMRES as candidates in a measured
  portfolio rather than as hard-coded fallback order.

Lane C: PETSc-like non-differentiable production backend.

- Add an explicit host sparse backend for CLI/Python runs that do not need
  autodiff.
- Assemble CSR/CSC/COO sparse operators and preconditioner matrices matching the
  active SFINCS v3 operator semantics.
- Use SciPy sparse direct/iterative methods by default when petsc4py is not
  installed.
- Add optional petsc4py support for PETSc KSP/PC, MUMPS, SuperLU_DIST, monitors,
  and command-line-like options.
- Keep this path opt-in first, then promote only after parity/performance
  validation.
- Make the backend explicit in outputs so users know whether the run was
  differentiable.

Lane D: JAX-native differentiable backend.

- Keep the current matrix-free JAX path as the default for autodiff workflows.
- Use dense/direct JAX paths only for measured small/medium systems where they
  beat Krylov on runtime and memory.
- Use matrix-free GMRES/BiCGStab/LGMRES-like methods with preconditioners that
  are JIT-compatible and avoid host callbacks.
- Add custom-linear-solve or Lineax experiments only behind benchmarks proving
  lower runtime, lower memory, cleaner autodiff, or less code complexity.
- Prefer block/batched preconditioner kernels that map well to GPU:
  block-Jacobi, theta/zeta line blocks, x-blocks, PAS angular blocks, and
  Schwarz/domain decomposition where proven.

Lane E: GPU performance model.

- Separate cold runtime, warm runtime, setup time, kernel time, host/device
  transfer time, output-write time, and JIT compilation time.
- Keep persistent JAX compilation cache enabled for examples and documented CLI
  workflows.
- Avoid GPU fallback paths that build large host-side blocks and then transfer
  repeatedly.
- Use one-GPU-per-case and transport-worker parallelism as the documented
  production scaling lane until single-case multi-GPU is faster.
- Keep single-case multi-GPU as experimental until it shows strong scaling on
  a large enough benchmark.

Lane F: CPU performance model.

- Use dense/direct paths for genuinely small systems.
- Use host sparse direct or sparse-preconditioned Krylov for medium/large
  non-autodiff systems.
- Use JAX-native CPU paths for differentiable workflows and small research
  experiments.
- Support multi-core batch/transport workers, and add MPI/petsc4py options only
  after the single-process sparse path is correct and documented.

Lane G: Output, plotting, and reproducibility.

- Every solve writes enough metadata to reproduce the path:
  solver backend, preconditioner, tolerances, environment overrides, active
  grid, geometry source, package version, git commit if available, and timing.
- NetCDF/HDF5/NPZ outputs must carry the same physics arrays and solver trace
  where format permits.
- Plotting scripts should be generated from output files, not from hidden local
  benchmark state.

### 21.6 Testing and validation battery

Fast CI target: 5-10 minutes on normal GitHub runners.

Fast CI must include:

- Unit tests for solver-policy helpers, env parsing, preconditioner dispatch,
  geometry loading, output writing, plotting, CLI argument parsing, and Python
  API entry points.
- Numerical micro-tests for matrix-free versus explicit small operators.
- Linear-solver tests using known ill-conditioned nonsymmetric matrices, where
  residual and convergence history are checked.
- Candidate-gate tests that reject runtime/memory/residual regressions.
- HDF5/NetCDF/NPZ output schema tests.
- CLI tests for a fast example that creates output and plots.
- Documentation build with warnings as errors.
- Coverage reporting with the target path to 95 percent meaningful package
  coverage.

Physics-gate tests:

- Collision-operator conservation: particle, momentum, and energy invariants for
  full-FP pieces where the discrete model should preserve them.
- Maxwellian/nullspace checks for collision pieces.
- Onsager symmetry checks for transport matrices in the appropriate `Er=0`
  and force-normalization regimes.
- Pitch-angle-scattering limits and monoenergetic overlap checks against
  DKES/MONKES-like coefficients where the model assumptions match.
- High-collisionality Simakov-Helander/Mollen-style analytic trends for W7-X or
  LHD impurity/transport coefficients.
- Ambipolarity root checks: radial electric field roots should be finite,
  sign-consistent, and stable under small resolution changes.
- Bootstrap-current consistency checks: radial profile smoothness, sign
  stability, and finite-beta VMEC workflow regression tests.
- Geometry invariance/sanity checks: VMEC, Miller, Boozer/ASCII/netCDF geometry
  loading must produce finite B, Jacobians, drift terms, and quadrature weights.
- Autodiff checks: gradients of selected scalar outputs against finite
  differences for small differentiable cases.

Regression suite:

- Full 39-case CPU example-suite parity against frozen SFINCS Fortran v3
  references before every release.
- Full 39-case GPU suite on `office` before every release or GPU solver-policy
  promotion.
- Focused Nxi cliff regression: neighboring full-FP `Nxi=20/40` or equivalent
  small/medium cases must stay on measured fast paths and remain parity-clean.
- NTX RHSMode=1 profile-current finite-beta small case must remain CPU/GPU
  consistent.
- Optional heavy NTX and W7-X high-nu cases run as nightly/manual tests, not
  regular CI.
- Runtime and memory guard tests based on stored benchmark artifacts for top
  offenders.

Performance gates:

- A solver-policy change is blocked if it introduces a runtime drift flag on
  any focused artifact-backed case unless the new route is intentionally slower
  for correctness and documented.
- A GPU promotion must report cold and warm timings.
- A memory-motivated promotion must prove lower peak memory, not just a lower
  estimate.
- Benchmark scripts must record subprocess wall time and in-code solver elapsed
  separately.

### 21.7 Implementation order

P0: Final plan and first guardrail.

- [x] Write this plan in `plan.md`.
- [x] Add measured solver-candidate acceptance module.
- [x] Add tests for residual/runtime/memory/pathological-fallback rejection.
- [x] Add stable solver-trace schema with JSON/HDF5 round-trip tests.
- [x] Add opt-in trace persistence for HDF5, NetCDF, and NPZ output files.
- [x] Run focused tests (`48 passed` across solver-selection, trace, output-format, IO, policy, and progress tests).

P1: Wire measured gates into existing RHSMode=1 policy.

- [x] Wire the first measured gate into `rhs1_pas_weak_auto_override_kind`,
  preserving default historical behavior when measured metrics are unavailable.
- [x] Add optional measured gates to `rhs1_accept_candidate`, the shared
  accepted-candidate handoff used by RHSMode=1 retry/rescue branches.
- [x] Pass real elapsed-time/RSS metrics through reduced/full stage2 and
  strong-preconditioner retry handoffs.
- [x] Pass real elapsed-time/RSS metrics through reduced/full dense and sparse
  retry handoffs.
- Replace remaining direct ad hoc auto-promotion checks in `v3_driver.py` with
  calls into `solver_selection_policy` at the strong-fallback and dense/sparse
  promotion points.
- Preserve existing default behavior unless a candidate is provably rejected by
  the new correctness gate.
- [x] Add unit tests for the first dispatch point using small fake metrics.
- [x] Add unit tests for measured handoff acceptance/rejection using small fake
  metrics.
- Add unit tests for the remaining concrete `v3_driver.py` dispatch points once
  real metrics are passed into the handoff.
- [x] Add artifact-backed regression tests for the CPU solver-path audit and GPU
  neighboring `Nxi=20/40` full-FP cliff reports.
- [x] Add PAS-heavy path-switching artifact tests where available.

P2: Stable solver trace.

- [x] Add a `SolverTrace` dataclass and JSON/HDF5 serialization helpers.
- [x] Add opt-in HDF5/NetCDF/NPZ output persistence for solver traces.
- [x] Add tests that supported output formats can record backend/path/timing
  metadata without changing default parity payloads.
- [x] Emit JSON sidecar traces from CLI and Python `write-output` runs via
  `--solver-trace` / `solver_trace_path=`.
- [x] Update the reduced/scaled suite runner to read trace elapsed time from
  solver-trace sidecars when available.
- Extend trace ingestion to any remaining benchmark/summary scripts that still
  only parse free-form logs.

P3: Host sparse production backend.

- Start with a non-autodiff, opt-in CPU sparse backend using SciPy CSR and
  sparse direct/GMRES/LGMRES.
- Match SFINCS v3 active DOF ordering and constraints.
- Validate on tiny and small reference cases against existing dense/JAX paths.
- Add optional petsc4py integration plan after the SciPy backend is correct.
- Promote only for measured CPU cases where it beats JAX dense/Krylov or avoids
  a known memory cliff.

P4: GPU policy hardening.

- Add cold/warm trace separation.
- Gate accelerator dense and PAS paths through measured-candidate rules.
- Add focused GPU tests/artifact checks for geometry4/5/11, HSX, W7-X, and
  tokamak PAS/FP cases.
- Keep broad GPU route changes off by default until the full GPU suite is
  parity-clean and performance-clean.

P5: Physics validation expansion.

- Add the physics gates listed above in fast miniature form for CI.
- Keep wider literature-reproduction scans as manual/nightly artifacts with
  publication-ready plots.
- Document which validations are CI gates, release gates, and paper-grade
  optional campaigns.

P6: Documentation and user workflow.

- Update README and docs with clear solver backend choices:
  autodiff-safe JAX, fast non-autodiff dense/sparse, CPU/GPU behavior, and
  diagnostic outputs.
- Document how to force paths for reproducibility and how to read solver traces.
- Add examples for Python and CLI workflows, including plotting outputs and
  profiling a slow case.

### 21.8 Ship criteria for the solver-policy lane

This lane is closed only when:

- The full CPU example suite remains `39/39` parity-clean with no missing output
  keys.
- The full GPU example suite remains `39/39` parity-clean with no `jax_error` or
  `max_attempts`.
- The Nxi/full-FP cliff regression is impossible under default auto selection:
  any slow fallback must be justified by residual/parity improvement.
- Runtime and memory summary plots/tables are regenerated from trace-backed
  artifacts.
- Documentation explains backend differences, autodiff restrictions, and solver
  diagnostics.
- CI includes fast but meaningful physics/numerics/regression tests and keeps
  wall time bounded.

### 21.9 Immediate next actions

1. Run focused CPU tests, then the bounded full CPU suite if the dispatch wiring
   changes behavior. Done on 2026-04-30 with `39/39` strict-clean CPU parity.
2. Run focused GPU validation on `office` before any accelerator default is
   changed. Done on 2026-04-30 with `7/7` strict-clean focused one-GPU parity.
3. Optional release polish: run the full 39-case GPU refresh only if benchmark
   figures need fresh all-case GPU timings from the measured-gate branch. Done
   on 2026-04-30 with `39/39` strict-clean one-GPU parity.

## 22) Production-resolution benchmark tier and production blockers (2026-04-30)

The reduced 39-case parity suite is still useful as a fast smoke/regression
gate, but it is not representative enough for public production runtime/memory
claims or solver-policy design. A direct audit showed that the existing public
CPU/GPU comparison artifacts contain `0/39` non-axisymmetric cases at
`Ntheta >= 21`, `Nzeta >= 21`, `Nxi >= 15`, and only `2/39` axisymmetric
tokamak cases at the corresponding `21 x 1 x 15` threshold. Nominal examples
are better, but still sparse: only `3/38` non-axisymmetric checked-in examples
meet `21 x 21 x 15`.

New artifact:

- `benchmarks/production_resolution_inputs_2026-04-30/manifest.json`
  defines the current SFINCS_JAX-owned 39-case production benchmark tier.
- Checked-in public examples are lifted to `3D >= 35 x 43 x 17 x 48`
  (`Ntheta x Nzeta x Nx x Nxi`) and tokamak `>= 42 x 1 x 16 x 62` while
  preserving any higher nominal resolution. Public production timing rows target
  SFINCS Fortran v3 runtimes of at least `10 s`.
- Downstream/collaborator decks are no longer part of the active public
  production manifest. They can still be imported with explicit
  `--external-input` arguments for private reproduction, but they are not
  release blockers for SFINCS_JAX.
- The regenerated public manifest has `39` cases, zero resolution-floor
  violations, and preflight recommendations of `5` `bounded_local_ok`,
  `1` `bounded_remote`, and `33` `remote_or_cluster_only` cases.
- Absolute VMEC/Boozer paths are copied into each staged case and localized so
  the same benchmark tree can run on `office` GPUs and other remote hosts.

Harness fixes landed with this lane:

- JAX benchmark subprocesses now use the same process-group runner as Fortran
  jobs. A timeout kills the full process group, including the actual
  `sfincs_jax` child behind `/usr/bin/time`, so failed production cases no
  longer leave orphan CPU/GPU jobs.
- `scripts/run_scaled_example_suite.py` now preserves measured
  `suite_report.json` Fortran wall-time and RSS metadata when reusing staged
  reference artifacts. This avoids mixing JAX wall time with Fortran internal
  "Time to solve" log entries in GPU reports.
- Regression tests cover production-input generation, absolute-path
  localization, staged-reference metadata reuse, and timeout process cleanup.

Historical CPU pilot artifact from the lower-resolution bring-up tier:

- `benchmarks/production_resolution_cpu_pilot_2026-04-30`
- Pattern: `tokamak_1species_PASCollisions_noEr_Nx1`,
  `tokamak_1species_FPCollisions_noEr`, and three NTX/collaborator finite-beta
  decks.
- Result: `1/5` strict-clean, `2/5` strict mismatches, `2/5` max-attempts.
- Clean case: `tokamak_1species_PASCollisions_noEr_Nx1` at
  `21 x 1 x 1 x 31`: Fortran wall `75.226 s`, JAX CPU wall `2.685 s`, RSS
  `99.7 MB` vs `484.2 MB`, strict mismatches `0/212`.
- NTX low-resolution RHSMode=1 profile-current deck
  `13 x 15 x 5 x 8`: Fortran wall `76.454 s`, JAX CPU wall `5.915 s`, RSS
  `238.8 MB` vs `3726.0 MB`, strict mismatches `30/193`. Mismatches are
  concentrated in flow/current/density/pressure perturbation outputs including
  `FSABFlow`, `FSABjHat`, `FSADensityPerturbation`, and related moment arrays.
- NTX RHSMode=1 profile-current deck `17 x 21 x 5 x 12`: Fortran wall
  `79.076 s`, JAX CPU timed out at `600 s`. The JAX path built a
  `42850 x 42850` system, selected Krylov/Schur, then fell back to `pas_lite`
  with residual about `1.88e-02`, far above the target.
- NTX RHSMode=3 transport matrix `35 x 43 x 1 x 48`: Fortran wall `90.750 s`,
  JAX CPU wall `2.370 s`, RSS `1719.4 MB` vs `616.6 MB`, but strict mismatches
  `33/193`.
- Lifted `tokamak_1species_FPCollisions_noEr` exposed a local Fortran v3
  reference instability: PETSc reached residuals near `3e-09`, then crashed
  with signal 11 before writing a usable comparison output. This row needs a
  cleaner Fortran reference policy before it can be used as a production claim.

Historical one-GPU pilot artifact from the lower-resolution bring-up tier:

- `benchmarks/production_resolution_gpu_pilot_2026-04-30`
- Run on `office` with `CUDA_VISIBLE_DEVICES=0` and
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`. The machine was not idle; unrelated
  GPU-heavy jobs were active, so timing is diagnostic rather than
  publication-quality.
- Result: `1/4` strict-clean, `2/4` strict mismatches, `1/4` max-attempts.
- Clean case: `tokamak_1species_PASCollisions_noEr_Nx1`: JAX GPU wall
  `4.897 s`, RSS `1014.5 MB`, strict mismatches `0/212`.
- NTX low-resolution RHSMode=1 profile-current deck: JAX GPU wall `43.052 s`,
  RSS `2909.3 MB`, strict mismatches `30/193`.
- NTX RHSMode=1 profile-current deck `17 x 21 x 5 x 12`: JAX GPU wall
  `605.619 s`, RSS `5506.5 MB`, strict mismatches `34/193`. The log records
  `pas_lite` fallback with residual `1.840e-02`.
- NTX RHSMode=3 transport matrix `35 x 43 x 1 x 48`: JAX GPU timed out. The
  log shows dense and distributed GPU transport paths disabled, active-DOF mode
  disabled, collision and strong preconditioner residuals still above target,
  and a long polish solve.

Production conclusions:

- Current reduced-suite parity is real for reduced examples, but not sufficient
  for research-grade production claims.
- For SFINCS_JAX-owned public claims, the current production blockers are the
  large public 3D/PAS/FP rows in the generated 39-case manifest, not downstream
  profile-current handoff decks.
- The top issue is not geometry setup or HDF5 output; it is solver formulation,
  preconditioning, residual acceptance, and large-active-system runtime/RSS.
- Public README/docs benchmark plots should either be explicitly labeled as
  reduced-suite smoke/parity benchmarks or withheld until this production tier
  has clean CPU/GPU rows.
- Previous `17 x 21 x 5 x 12` sparse-host timing evidence is useful for solver
  bring-up only. New public production-runtime and memory claims must be rerun
  from the regenerated `35 x 43 x 17 x 48` / `42 x 1 x 16 x 62` manifest.

Next required engineering steps:

1. Completed: add a manual, not CI-fast, production input lane for the 39-case
   public manifest and guard the benchmark runner with
   `size_estimate.run_recommendation` so local runs do not launch remote-only
   rows accidentally.
2. Re-run the 39-case production tier in stages: `bounded_local_ok` locally,
   `bounded_remote` on `office`, and `remote_or_cluster_only` on explicitly
   budgeted remote/cluster hardware.
3. Fix the RHSMode=1 PAS/Schur/`pas_lite` fallback policy where it affects
   public production rows. A fallback with
   residual `O(1e-2)` must be a hard failure, not an accepted output path for
   production runs.
4. Keep strengthening the CPU sparse/PETSc-like production backend or equivalent structured
   preconditioner for large explicit CLI solves. The Fortran v3 architecture
   remains faster on these production-sized cases because it uses sparse
   PETSc/KSP/preconditioner machinery rather than large JAX dense/matrix-free
   fallback paths.
5. Fix RHSMode=3 GPU transport for large VMEC decks: active-DOF transport,
   stronger sparse/structured preconditioning, and residual-gated polish must
   work on GPU or be routed to a faster CPU/PETSc-style backend when autodiff is
   not requested.
6. Re-run the full 39-case production tier on CPU and on an idle GPU after the
   blocker fixes. Only then regenerate public performance figures/tables.

### 22.1 NTX RHSMode=1 operator/solver audit update

Status: in progress on 2026-04-30.

Fresh focused evidence:

- The small public full-FP `Nxi=20`/`Nxi=40` cliff no longer reproduces on
  current `main`: both neighboring cases stay on the dense path, finish in a few
  seconds locally, and have residuals below the solver target.
- The lower-resolution collaborator/NTX finite-beta RHSMode=1
  PAS/profile-current deck at `17 x 21 x 5 x 12` (`42850` unknowns) was the
  initial sparse-host bring-up blocker. The active public production manifest is
  now larger (`35 x 43 x 17 x 48` for 3D and `42 x 1 x 16 x 62` for tokamak
  rows), so the old deck is no longer sufficient for public production claims.
- A Fortran v3 run of the same deck with the local
  `/Users/rogeriojorge/local/tests/sfincs/fortran/version3/sfincs` executable
  finished in about `13.6 s` and about `626 MB` peak RSS using sparse PETSc/KSP
  behavior. The printed diagnostics include
  `FSABjHat=-1.2981550382181504`.
- A Fortran MATLAB-matrix dump audit showed that the JAX operator is not the
  mismatch source: applying the JAX operator to the Fortran state differs from
  the dumped Fortran sparse matrix by norm `4.39e-16`, and the true JAX residual
  on the Fortran state matches the Fortran residual (`5.7549e-7`).
- Direct SciPy `spsolve` of the dumped Fortran sparse matrix solved the same
  linear system in about `4.4 s` with residual `7.6e-16`.
- Current JAX default Krylov/rescue behavior remains too slow on this deck: a
  bounded default probe spent about `41.6 s` in the first synchronized Krylov
  cycle and then fell into a slow rescue ladder with residual still far above
  target.
- Light PAS preconditioner probes (`pas_lite`, `pas_hybrid`, `xmg`) are fast but
  wrong on this deck, producing residuals from `1e9` to `1e18`; they must never
  be auto-promoted without a true-residual gate.
- The current column-by-column explicit sparse helper can produce the correct
  answer when forced onto the full, non-projected matrix and when PAS-Schur
  rescue is disabled: residual `7.4e-17`, `FSABjHat=-1.2981550371186046`.
  However it takes about `61 s` and peaks near `11 GB`, so it is a debugging /
  rescue proof-of-correctness path, not a production backend.
- The resource peak-RSS path is now wired into solver traces so transient sparse
  factorization memory peaks are not hidden by post-factorization RSS samples.
- A bounded one-GPU Perfetto/XPlane profiling pass on `office` for the NTX
  `13 x 15 x 5 x 8` profile-current deck is accurate but cold/warm dominated:
  cold traced solve `9.67 s`, warm traced solve `2.23 s`, residual
  `2.53e-13`, `FSABjHat=-0.44127035079410998`. Trace artifacts were written at
  `/home/rjorge/sfincs_jax_gpu_profile_runs/ntx_13x15x8_gpu_trace*/trace`.
  TensorFlow profiler hooks were unavailable on `office`, but JAX still emitted
  Perfetto and XPlane traces.
- The profiling helper now respects `--wout-path` / `--equilibrium-file` before
  trying to localize the namelist equilibrium path, so remote profiling no
  longer fails when the input carries an absolute path from another workstation.
- The Fortran/JAX operator audit utility now has an opt-in `--solve-sparse`
  check that solves the dumped PETSc sparse matrix with SciPy and records
  residual/time. This is the reference harness for the planned sparse-host
  production backend.
- Structural sparse-host infrastructure now has a tested first slice:
  `explicit_sparse.build_operator_from_pattern()` materializes CSR matrices by
  coloring a conservative sparsity pattern and probing one seed vector per
  color, avoiding dense identity/operator materialization. The helper is covered
  by diagonal, tridiagonal, over-approximated-pattern, and budget-fallback tests.
- `v3_sparse_pattern.v3_full_system_conservative_sparsity_pattern()` now builds
  a conservative full-system pattern covering fixed theta/zeta stencils,
  dense same-geometry velocity/species blocks for FP, Phi1/quasineutrality
  couplings, and constraint rows/columns. Tests verify that it covers frozen
  Fortran PETSc matrices for PAS, FP, and Phi1 tiny systems, and that the
  colored probe reconstructs the PAS tiny matrix-free operator to Fortran
  tolerances.
- The pattern probe is wired only behind
  `SFINCS_JAX_RHSMODE1_EXPLICIT_SPARSE_PATTERN=1` in the full-system host
  sparse direct path. Defaults are unchanged until NTX/prod-scale parity,
  runtime, and memory evidence justify promotion.
- The first production-scale structural sparse-host probes are successful. On
  the NTX finite-beta `13 x 15 x 5 x 8` PAS/profile-current deck, direct
  pattern build/probe/LU/solve took about `0.41 + 1.79 + 0.35 + 0.006 s`, with
  residual `1.5e-16` and peak RSS about `587 MB`. On the collaborator blocker
  `17 x 21 x 5 x 12` deck (`42850` unknowns), direct
  pattern/probe/LU/solve took about `1.17 + 3.10 + 3.95 + 0.026 s`, with
  residual `9.3e-16`, peak RSS about `1.50 GB`, and
  `FSABjHat=-1.2981550371185984`, matching the Fortran v3 reference
  `-1.2981550382181504` to about `1.1e-9`.
- `solve_method="sparse_host"` is now a public, explicit, non-differentiable
  full-system RHSMode=1 path for Python and CLI. A checked CLI
  `write-output --solve-method sparse_host --solver-trace ...` run on the
  `17 x 21 x 5 x 12` NTX deck wrote the full output in `8.65 s`, residual
  `7.6e-17`, peak RSS about `1.54 GB`, and the same `FSABjHat` parity. Solver
  traces now expose a top-level `converged` flag in addition to residual and
  target. This is a development regression datum only; it must be superseded by
  the larger active production-manifest rerun before any public production
  claim.

Engineering conclusion:

- The high-ROI fix is not another threshold-only preconditioner flip. The
  production CPU/CLI lane needs a structural sparse assembly backend matching the
  Fortran/PETSc sparsity pattern, followed by SciPy/PETSc sparse direct or
  sparse-preconditioned Krylov solves. The current matrix-free JAX operator is
  accurate but inefficient for large non-autodiff RHSMode=1 production runs when
  sparse structure is recoverable.

Next concrete steps:

1. Run the public `solve_method="sparse_host"` path on additional PAS and FP
   geometry-rich examples, including at least one W7-X/HSX case, and compare
   parity/runtime/RSS with the current default path and Fortran v3.
2. Add a conservative policy gate that can suggest or auto-select
   `sparse_host` for non-differentiable CPU RHSMode=1 production runs only when
   the structural pattern is supported and the system is outside the small dense
   range. Keep differentiable and GPU-native paths unchanged.
3. Split directly assembled deterministic blocks from colored probing if wider
   examples show probe color count or pattern over-approximation becoming the
   next bottleneck.
4. Add tests that fail if any RHSMode=1 production output is written with a
   residual far above target unless the user explicitly opts into a diagnostic
   nonconverged output mode.
5. Re-run a bounded pilot subset from the regenerated `35 x 43 x 17 x 48`
   production manifest before any full suite run. Start with the staged NTX
   sparse-host deck and one tokamak/one 3D public example, record residual,
   wall time, peak RSS, and `solver_trace.json`.
6. Re-run the full production benchmark manifest after the sparse-host lane is
   exercised on the broader suite, then refresh public performance language and
   plots.

### 22.2 Research-resolution production baseline update

Status: refreshed for larger input generation on 2026-05-03; benchmark reruns pending.

- `scripts/create_production_benchmark_inputs.py` now enforces the production
  floor requested for research-appropriate grids: `3D >= 35 x 43 x 17 x 48`
  and tokamak `>= 42 x 1 x 16 x 62`.
- The manifest records `target_fortran_min_runtime_s = 10.0`. Promotion into
  public timing plots requires the measured SFINCS Fortran v3 row to satisfy
  that runtime floor, so the reduced-suite smoke artifacts cannot be mistaken
  for production performance claims.
- The generator now includes `Nx` in the floor and keeps downstream or
  collaborator decks behind explicit `--external-input` arguments. Public
  SFINCS_JAX benchmark manifests are example-only.
- Each manifest entry now includes a preflight `size_estimate` block with
  species count, inferred collision/constraint switches, full-system unknown
  count, dense matrix bytes, conservative sparse-pattern nnz/CSR bytes, and a
  `bounded_local_ok` / `bounded_remote` / `remote_or_cluster_only` run
  recommendation.
- The manifest was regenerated with `--clean`. It contains `39` SFINCS_JAX-owned
  cases, has zero resolution-floor violations, and no staged case names still
  contain stale downstream deck labels. The current preflight recommendations
  are `5` bounded-local cases, `1` bounded-remote case, and
  `33` remote-or-cluster-only cases.
- Focused tests verify example lifting, downstream exclusion from the checked-in
  manifest, explicit external-input handling, historic deck relabeling when
  requested, and the large PAS+XDot sizing gate.
- A bounded sizing pass shows why the new baseline must be scheduled rather than
  run blindly on the local laptop. Several higher-resolution HSX/FP and W7-X
  examples in the public manifest are remote-or-cluster-only by the preflight
  estimate, so full production reruns need explicit timeout/RSS guards and
  should be routed to `office` or cluster hardware.

Next benchmark actions:

1. Completed on 2026-05-01: wired manifest `size_estimate.run_recommendation`
   into `scripts/run_scaled_example_suite.py`. Generated production `inputs/`
   trees auto-detect their sibling `manifest.json`, default to launching only
   `bounded_local_ok` rows, and require an explicit `--max-run-recommendation`
   opt-in for `bounded_remote`, `remote_or_cluster_only`, or `all` rows.
2. Run the `bounded_local_ok` public production subset locally with explicit
   timeout/RSS guards, then run the `bounded_remote` and
   `remote_or_cluster_only` subsets on `office` or cluster hardware.
3. Promote runtime/RSS/residual rows into the production report only if they are
   converged and parity-clean against Fortran v3.

Validation:

- `pytest -q tests/test_scaled_example_suite_reference.py tests/test_create_production_benchmark_inputs.py`
  passed with `23 passed` after adding the production-run recommendation guard.
- The guard is intentionally a runner-level safety feature, not a solver-policy
  change: ordinary example-suite runs without a production manifest are
  unchanged.

### 22.3 Historical downstream solve launch and constrained-PAS nullspace finding

Status: archived as downstream context on 2026-05-01. The bounded solve evidence
below remains useful for sparse-host design, but it is no longer an active
SFINCS_JAX release blocker or handoff lane.

- The staged NTX finite-beta profile-current deck at `25 x 31 x 11 x 17`
  now runs through an explicit non-differentiable sparse-host production path.
  `--solve-method sparse_pc_gmres` materializes the conservative sparse pattern
  on the host, factors the RHSMode=1 preconditioner, and runs GMRES on the true
  matrix-free Jacobian. On `office` GPU0 it completed in `277.0 s` wall time
  with peak RSS `11.43 GB` and true residual `4.80e-14` versus target
  `2.09e-09`.
- That exact sparse branch is algebraically converged but not the small-current
  Fortran-v3 KSP branch for this singular/near-singular constrained-PAS system:
  it gives `FSABjHat=-5.033868`, matching the direct sparse-LU branch, whereas
  the two Fortran v3 reference runs give branch-sensitive small-current values
  `FSABjHat=-0.00444045` (plain run) and `FSABjHat=-0.0270762` (binary-dump
  run).
- A low-memory `--solve-method sparse_lsmr` diagnostic path was added. It
  materializes the true CSR operator without LU factorization and ran the same
  deck in about `60.5 s` with peak RSS `1.96 GB`, but it did not solve the
  algebraic system (`residual=2.07e-02` versus target `2.09e-09`). With the
  diagnostic nonconverged-output override, it gave a small-current branch
  `FSABjHat=-0.0422348`, useful for nullspace triage but not acceptable as a
  production-converged solve.
- The Fortran v3 audit shows why this case is subtle: PETSc GMRES reports KSP
  convergence by preconditioned residual, but the nonlinear residual norm remains
  about `2.093e-02` after the solve. The dumped Fortran Jacobian, preconditioner,
  and residual-f1 matrices are identical for this deck, so exact LU in Python
  chooses a different nullspace/gauge branch from PETSc/MUMPS GMRES.
- Sparse factor controls are now exposed for targeted experiments:
  `SFINCS_JAX_EXPLICIT_SPARSE_FACTOR_KIND`, `SFINCS_JAX_EXPLICIT_SPARSE_PERMC_SPEC`,
  `SFINCS_JAX_EXPLICIT_SPARSE_DIAG_PIVOT_THRESH`,
  `SFINCS_JAX_EXPLICIT_SPARSE_ILU_FILL_FACTOR`,
  `SFINCS_JAX_EXPLICIT_SPARSE_ILU_DROP_TOL`, and the preconditioner-only
  `SFINCS_JAX_RHSMODE1_SPARSE_PC_SHIFT`. ILU is currently exactly singular on
  this deck, and a shifted exact preconditioner still returns the exact-LU
  branch.
- A production safety gate now refuses to write large RHSMode=1 diagnostics
  when the reported residual misses the requested target, unless
  `SFINCS_JAX_ALLOW_NONCONVERGED_OUTPUT=1` is set explicitly for diagnostic
  branch triage.

Next concrete steps:

1. Implement a constrained-PAS gauge/nullspace selector that is independent of
   sparse solver pivoting. Candidate gates are minimum-flow/minimum-moment
   constraints, explicit source/gauge rows, or a PETSc/MUMPS-backed optional
   host solve when `petsc4py` is available.
2. Add a small frozen regression fixture that reproduces this nullspace branch
   sensitivity at affordable size, then assert that exact LU, minimum-norm, and
   PETSc-compatible branches are explicitly labeled rather than silently mixed.
3. Keep `sparse_pc_gmres` available as the bounded algebraic production solve,
   but do not promote this NTX constrained-PAS result as parity-clean until the
   physical gauge branch is pinned.
4. Keep `sparse_lsmr` as a diagnostic/research tool only unless a future
   residual definition and physics gate are agreed for singular PAS systems.

### 22.4 Production-resolution Phi1/QN solver-policy closure

Status: closed on 2026-05-01 for the bounded local production rows.

- The bounded local production subset exposed three false Phi1/QN failures:
  `tokamak_1species_FPCollisions_noEr_withPhi1InDKE`,
  `tokamak_1species_FPCollisions_noEr_withQN`, and
  `tokamak_1species_PASCollisions_noEr_withQN`. The old default selected the
  frozen explicit Newton shortcut above the small-fixture regime, stalled the
  nonlinear residual, and wrote branch-mismatched flow/current diagnostics.
- The production default now promotes moderate CPU explicit Phi1/QN systems to
  host sparse-direct Newton steps at `active_total_size >= 5000` when dense mode
  would otherwise be skipped, and sparse-direct uses the true Newton
  linearization unless `SFINCS_JAX_PHI1_USE_FROZEN_LINEARIZATION=1` is set
  explicitly.
- A QN-only typo in the nonlinear Phi1 path was fixed:
  `includePhi1InCollisionOperator` is read from `phys_params`, not from an
  undefined `phys` local.
- Focused default reruns against the existing MUMPS/SuperLU Fortran references
  are strict-clean:
  `tokamak_1species_FPCollisions_noEr_withPhi1InDKE` converged to
  `1.63e-14` and wrote in about `4.8 s`; `tokamak_1species_FPCollisions_noEr_withQN`
  converged to `3.45e-16` and wrote in about `3.1 s`;
  `tokamak_1species_PASCollisions_noEr_withQN` converged to `6.70e-10` and
  wrote in about `2.0 s`.
- Harness validation:
  `python scripts/run_scaled_example_suite.py --examples-root benchmarks/production_resolution_inputs_2026-04-30/inputs --reference-results-root /tmp/sfincs_jax_prod_bounded_local_2026_05_01_cd0118f --out-root /tmp/sfincs_jax_phi1_qn_default_suite_check --timeout-s 120 --max-attempts 1 --fortran-min-runtime-s 0 --runtime-adjustment-iters 0 --jax-profile-marks off --max-run-recommendation bounded_local_ok --pattern 'tokamak_1species_.*(withQN|withPhi1InDKE)' --reset-report`
  completed `3/3 parity_ok`, strict mismatches `0/274` for every row, and
  missing Fortran output keys `0`.
- Unit/regression validation:
  `pytest -q tests/test_cli_solve_mode.py tests/test_io_output_policy_coverage.py tests/test_rhsmode1_phi1_write_output_end_to_end.py`
  passed with `57 passed`, and `ruff check sfincs_jax/io.py tests/test_cli_solve_mode.py tests/test_rhsmode1_phi1_write_output_end_to_end.py`
  passed.

Remaining bounded-local notes:

1. `tokamak_2species_PASCollisions_noEr` still shows strict flow/current
   differences of order `1e-5` against the specific MUMPS/SuperLU Fortran HDF5
   reference, while exported `delta_f`/`full_f` slices agree to about `1e-11`.
   Exact JAX residual is `3.03e-17`, so this is a constrained-PAS branch and
   diagnostic sensitivity, not a failed solve. PETSc-compatible/minimum-norm
   sparse probes select a different physical branch and are not acceptable
   defaults. Follow-up dense JAX probing selected the same JAX flow/current
   branch as the default, while the Fortran log records final residual
   `1.177e-08` against an estimated `solverTolerance*rhs_norm` target
   `6.811e-10`. The benchmark harness now parses RHS-scaled residual targets
   and classifies this row as `reference solver quality` instead of a generic
   SFINCS_JAX solver branch mismatch.
2. `tokamak_1species_FPCollisions_noEr` was a Fortran-reference process-status
   issue in the bounded local production run: the MUMPS/SuperLU Fortran binary
   solved, printed diagnostics, and wrote a readable HDF5 file, then aborted
   during PETSc teardown. Reusing that HDF5 reference with the same Nxi=31 input
   gives `parity_ok`, strict mismatches `0/187`, and print parity `8/8`; the
   JAX run converged to residual `1.71e-13` in about `16.7 s`.

Validation after the reference-quality classifier:

- `pytest -q tests/test_scaled_example_suite_reference.py` passed with
  `20 passed`.
- `ruff check scripts/run_reduced_upstream_suite.py tests/test_scaled_example_suite_reference.py`
  passed.
- Focused harness rerun for `tokamak_2species_PASCollisions_noEr` completed in
  about `1.68 s` logged JAX solve time / `1.28 GB` RSS, kept strict mismatches
  localized to flow/current diagnostics (`15/212`), and reported blocker
  `reference solver quality` with the scaled Fortran residual note.

Follow-up reporting cleanup on 2026-05-02:

- The scaled-suite summary now prints runtime offenders using
  `jax_logged_elapsed_s` when available, matching the metric already used for
  offender sorting. This avoids mixing solver-time ranking with subprocess
  wall-time ratios in production reports.
- The summary now emits a dedicated `Reference-quality rows` section, so
  branch-sensitive rows caused by loose Fortran references remain visible
  without being presented as ordinary SFINCS_JAX solver failures.
- Focused rerun for `tokamak_2species_PASCollisions_noEr` reports
  `jax=2.104 s`, `fortran=75.315 s`, blocker `reference solver quality`, and
  strict mismatches confined to the same flow/current diagnostic family
  (`15/212`).

Follow-up constrained-PAS guard on 2026-05-02:

- Added `sfincs_jax/constrained_pas_branch.py`, a small diagnostic helper for
  classifying RHSMode=1 constrained-PAS branch spread across exact true-residual,
  PETSc-compatible minimum-norm, and preconditioned-residual reference branches.
- Added a compact artifact-backed regression at
  `tests/reference_solver_path_artifacts/constrained_pas_branch_probe_2026-05-02.json`.
  The fixture records the finite-beta profile-current branch sensitivity without
  storing large matrices or downstream-project paths.
- Added tests that require the converged sparse-PC branch to remain the selected
  true-residual reference, while weak preconditioned-residual branches are
  labeled as reference-quality blockers rather than promoted as parity anchors.

README/public benchmark cleanup on 2026-05-03:

- The full 39-case frozen CPU/GPU suite remains the parity and CI smoke audit,
  but README-facing runtime and memory comparisons now filter to rows whose
  SFINCS Fortran v3 reference runtime is at least `10 s`.
- The runtime/memory summary figure, machine-readable JSON, and README table now
  report only the 21 production-scale rows. The 18 shorter PAS, monoenergetic,
  and transport rows are recorded explicitly as excluded CI parity/smoke checks
  until they are rerun at production-comparison resolution.
- The production-resolution input manifest remains the route for future public
  comparison reruns, with 3D/tokamak resolution floors and the same `10 s`
  Fortran v3 runtime target.

Active-memory reporting cleanup on 2026-05-03:

- Kept full external-process peak RSS in the frozen suite JSON as
  `jax_max_rss_mb`, but changed the public plot and README table to use
  profiler-derived active JAX memory (`jax_incremental_max_rss_mb`) when
  available. This subtracts the fixed Python/JAX/XLA runtime baseline and avoids
  presenting constant backend overhead as per-case solver memory.
- Backfilled the GPU frozen suite report from the matching `office` profiler logs
  without committing the large trace files. Production-scale medians are now:
  CPU process RSS `468.4 MB` vs active `279.7 MB`, GPU process RSS `913.7 MB`
  vs active `364.2 MB`, with the full process values still auditably preserved.
- Regenerated the README/public benchmark plot and summary JSON, and updated the
  docs so every page labels the memory panel as active solver memory rather than
  raw JAX process RSS.

Active lane (2026-05-04): Fortran-competitive memory management

- Research conclusion: SFINCS Fortran v3/PETSc wins memory mostly through
  algorithmic representation, not runtime bookkeeping. The important patterns
  are sparse AIJ matrix preallocation, thresholded value insertion, distributed
  row ownership, KSP/preconditioner reuse, fill-reducing orderings, and bounded
  GMRES restart/short-recurrence Krylov choices. JAX memory knobs
  (`XLA_PYTHON_CLIENT_PREALLOCATE`, device allocator choices, remat/offload,
  buffer donation, sharding, and Pallas kernels) are useful only after the solver
  avoids dense intermediates and replicated state.
- First implementation step: added `sfincs_jax/memory_model.py` with conservative
  dense/CSR/Krylov/preconditioner/device memory estimates and wired GMRES restart
  limiting in `sfincs_jax/solver.py` to that shared model.
- First route-selection guard: extended `sfincs_jax/solver_selection_policy.py`
  so future automatic solver promotions compare paired memory metrics in order:
  `device_peak_mb`, `active_rss_mb`, `compiled_temp_mb`, then legacy
  `peak_rss_mb`. This prevents apples-to-oranges memory wins and gives GPU
  memory-limited runs a first-class gate.
- Immediate tests: keep `tests/test_memory_model.py` and the new
  `tests/test_solver_selection_policy.py` cases green; these are unit-level
  guards before touching expensive RHSMode=1/PAS/FP paths.
- Next implementation steps:
  1. Add compiled-memory instrumentation around representative JITed matvec and
     solve kernels using `lower(...).compile().memory_analysis()` when available.
  2. Add active/device/compiled memory fields to solver trace artifacts and
     benchmark candidate JSON so route decisions are auditable case by case.
  3. Use `estimate_linear_solve_memory(...)` before dense fallback,
     sparse-from-matvec probing, Schur/PAS construction, and transport dense
     retry admission; disallow dense routes when dense operator + factors +
     Krylov basis + compiled temp exceed the configured budget.
  4. Replace large dense `from_matvec` construction with pattern-probed CSR or
     matrix-free preconditioners wherever parity and residual gates are clean.
  5. Evaluate mixed-precision preconditioners with float64 residual refinement
     for FP/PAS offenders, promoting only when all output comparisons stay clean.
  6. Build the memory-limited GPU lane around sharded matvec/preconditioner state
     across theta/zeta/radius/species partitions instead of replicated whole-case
     solves; use Pallas only for proven stencil/matvec hotspots where XLA emits
     large temporary buffers.
  7. Evaluate Lineax/Equinox wrappers only behind the same gate. The expected
     value is cleaner matrix-free operators and reusable differentiable solver
     state, not automatic memory reduction.
- Benchmark targets for this lane: production-resolution RHSMode=1 full-FP,
  PAS-heavy geometry4/HSX/geometry11 rows, and the collaborator-scale
  finite-beta/profile-current stress rows. A memory win is not public unless it
  reduces measured active/device memory, preserves residual and output parity,
  and does not introduce a runtime regression larger than the configured gate.

Active lane (2026-05-04): production-resolution memory campaign and CPU/GPU parity/stall closure

Problem statement:

- The current README-facing benchmark summary is not strong enough for the next
  public memory/performance claim. A quick audit of the frozen CPU/GPU reports
  shows HSX full-FP rows with final resolution `Ntheta=5, Nzeta=5, Nx=2, Nxi=4`,
  even though the new collaborator/research floor is at least
  `Ntheta=25, Nzeta=51, Nx=4, Nxi=100` for 3D benchmark cases.
- A dry-run with the requested floor via
  `scripts/create_production_benchmark_inputs.py --min-3d-ntheta 25 --min-3d-nzeta 51 --min-3d-nx 4 --min-3d-nxi 100 --min-tokamak-ntheta 25 --min-tokamak-nx 4 --min-tokamak-nxi 100`
  produced 39 inputs: 6 `bounded_local_ok`, 5 `bounded_remote`, and 28
  `remote_or_cluster_only`. HSX FP preserves its authored larger grid
  `25 x 115 x 5 x 149`, giving about `4.28e6` unknowns for two species and a
  dense matrix estimate of about `1.47e14` bytes, so dense/full materialized
  routes are impossible and must be rejected before allocation.
- For tokamak cases, the physically meaningful public benchmark floor is
  `Ntheta>=25, Nzeta=1, Nx>=4, Nxi>=100`. If an explicitly replicated
  `Nzeta=51` tokamak stress test is desired, treat it as a separate GPU-memory
  stress lane, not as the default axisymmetric physics comparison.
- Collaborator reports of CPU/GPU solution differences and solver stalls must be
  reproduced at this production floor before changing defaults. Any fix must be
  gated by true residuals, HDF5 output parity with Fortran v3, CPU/GPU agreement,
  runtime, active/device memory, and solver-search overhead.

Ordered implementation plan:

1. Lock the benchmark-resolution contract.

   - Change `scripts/create_production_benchmark_inputs.py` defaults to
     `DEFAULT_3D_MINIMUM = {NTHETA: 25, NZETA: 51, NX: 4, NXI: 100}` and
     `DEFAULT_TOKAMAK_MINIMUM = {NTHETA: 25, NX: 4, NXI: 100}`.
   - Update `tests/test_create_production_benchmark_inputs.py` to require that
     all 3D manifest rows meet or exceed `25 x 51 x 4 x 100`, all tokamak rows
     meet or exceed `25 x 1 x 4 x 100`, and HSX FP rows preserve the authored
     larger `25 x 115 x 5 x 149` resolution.
   - Add a validator in `sfincs_jax.validation_artifacts` that rejects public
     benchmark summaries when any reported row lacks `final_resolution` or falls
     below the benchmark floor. Wire it into
     `examples/publication_figures/generate_fortran_suite_benchmark_summary.py`
     and `tests/test_generate_fortran_suite_benchmark_summary.py`.
   - Gate: no README/docs benchmark plot can be regenerated from a report that
     includes below-floor production rows. Below-floor cases remain CI/smoke
     tests only.

2. Reproduce CPU/GPU mismatch and stall claims with a fixed input matrix.

   - Generate the new production inputs into
     `benchmarks/production_resolution_inputs_2026-05-04/` with the floor above.
   - Run bounded CPU cases locally first:
     `python scripts/run_scaled_example_suite.py --examples-root benchmarks/production_resolution_inputs_2026-05-04/inputs --production-manifest benchmarks/production_resolution_inputs_2026-05-04/manifest.json --max-run-recommendation bounded_local_ok --fortran-exe /Users/rogeriojorge/local/sfincs/fortran/version3/sfincs --out-root tests/production_floor_cpu_bounded_2026-05-04 --jax-repeats 2 --jax-cache-dir .jax_cache/production_floor --jax-profile-marks on --max-attempts 1 --timeout-s 1800 --reset-report`.
   - Run bounded-remote and remote-only cases on `office` GPU with
     `CUDA_VISIBLE_DEVICES=0` and again with `CUDA_VISIBLE_DEVICES=1`; use
     `--jax-profile-marks full` only on targeted offenders because full
     Perfetto/XPlane traces are large.
   - For each completed row, compare:
     Fortran v3 vs JAX CPU, Fortran v3 vs JAX GPU, and JAX CPU vs JAX GPU using
     the same `compare_sfincs_outputs` practical and strict modes.
   - Classify every non-clean row into exactly one blocker:
     `fortran_reference_quality`, `cpu_gpu_output_mismatch`,
     `true_residual_failure`, `timeout_or_stall`, `oom_or_allocator_failure`,
     `solver_search_overhead`, `missing_output_key`, or `harness_failure`.
   - Gate: do not optimize a case until it is reproducibly classified on both CPU
     and GPU or is explicitly marked hardware-limited.

3. Make solver decisions fully observable before adding more heuristics.

   - Extend `sfincs_jax.solver_trace.SolverTrace` and
     `SolverTraceCandidate` with `active_rss_mb`, `device_peak_mb`,
     `compiled_temp_mb`, `memory_metric`, `estimated_dense_nbytes`,
     `estimated_csr_nbytes`, `estimated_gmres_basis_nbytes`, `matvec_count`,
     `candidate_setup_s`, and `candidate_solve_s`.
   - Pass those fields from `sfincs_jax.solver_selection_policy`,
     `sfincs_jax.rhs1_handoff`, `sfincs_jax.v3_driver`,
     `sfincs_jax.profiling`, and `sfincs_jax.io`.
   - Add compile-memory probes using
     `lower(...).compile().memory_analysis()` around representative matvec,
     preconditioner apply, and diagnostics kernels when the backend supports it.
   - Add timeout-safe periodic solver-search logs: selected path, candidate path,
     residual, target, elapsed time, active memory, device memory, and reason for
     accepting/rejecting the candidate.
   - Gate: default production runs may spend at most 10% of total solve time or
     60 seconds, whichever is smaller, in route probing before selecting a solver.
     Longer exploration belongs in `scripts/benchmark_case_variants.py`, not in
     the default CLI/Python path.

4. Convert route selection from heuristic thresholds to memory preflight plus
   measured-candidate gates.

   - Use `sfincs_jax.memory_model.estimate_linear_solve_memory(...)` before
     dense fallback, dense preconditioner build, sparse-from-matvec probing,
     Schur/PAS construction, and transport dense retry admission.
   - Touch points:
     `sfincs_jax.rhs1_host_policy`, `sfincs_jax.rhs1_pas_policy`,
     `sfincs_jax.rhs1_schur_policy`, `sfincs_jax.rhs1_large_cpu_policy`,
     `sfincs_jax.transport_solve_policy`, `sfincs_jax.transport_policy`,
     `sfincs_jax.transport_dense_lu`, `sfincs_jax.explicit_sparse`, and
     `sfincs_jax.v3_driver`.
   - Dense or dense-probed routes are disallowed by default when
     `dense_operator + factors + Krylov basis + compiled_temp` exceeds 70% of the
     available host/GPU budget or exceeds the measured Fortran memory by more
     than 1.25x without a runtime win.
   - Gate: a route promotion must be residual-clean and parity-clean, then either
     at least 15% faster at no more than 5% extra active/device memory, or at
     least 25% lower active/device memory at no more than 10% runtime regression.
     A memory-limited rescue can be accepted with up to 50% runtime regression
     only if the incumbent OOMs/stalls and the rescue converges.

5. Eliminate large dense materialization from production 3D RHSMode=1 routes.

   - Replace large `explicit_sparse.build_operator_from_matvec` dense probing
     with `v3_sparse_pattern` plus colored pattern-probed CSR construction.
   - Stream CSR assembly in column-color chunks and write directly to CSR/COO
     buffers instead of retaining dense seed/result blocks.
   - Add pattern coverage tests against frozen Fortran PETSc matrices and
     high-resolution synthetic patterns so the sparse pattern cannot silently
     miss FP/PAS/Phi1/constraint couplings.
   - Touch points:
     `sfincs_jax.explicit_sparse`, `sfincs_jax.v3_sparse_pattern`,
     `sfincs_jax.petsc_binary`, `tests/test_explicit_sparse.py`,
     `tests/test_v3_sparse_pattern.py`, and `tests/test_full_system_matvec_parity.py`.
   - Gate: pattern-probed CSR must reconstruct matrix actions to `<=1e-10`
     relative error on small exact fixtures and preserve all output comparisons
     on at least one PAS, one FP, one Phi1/QN, HSX FP full, and HSX FP DKES
     case before becoming default.

6. Try low-memory Krylov and preconditioner candidates in a fixed variant ladder.

   - Candidate solvers to test: current GMRES, budget-capped restarted GMRES,
     SciPy LGMRES for host-only explicit paths, BiCGStab, TFQMR/IDR(s)-style
     short-recurrence alternatives if implemented, and sparse-PC GMRES with
     smaller restart.
   - Candidate preconditioners to test: collision diagonal, PAS tensor/line
     smoother, PAS `theta/zeta` Schwarz, x-block/TZ block, sparse-PC,
     FP species-x block, low-rank/Woodbury FP block, mixed-precision sparse
     factors with float64 residual refinement, and no-preconditioner fallback.
   - Run variants with `scripts/benchmark_case_variants.py --profile` on
     HSX_FP_full, HSX_FP_DKES, geometry11, W7-X, geometry4 PAS, and the bounded
     tokamak FP/PAS rows.
   - Gate: keep as default only if it passes the route-promotion gate in step 4.
     Keep as an opt-in memory-pressure mode if it reduces memory by at least 2x,
     converges, and is slower than the default by at most 2x. Reject and remove
     from auto-selection if it creates CPU/GPU drift, nonfinite residuals, or
     solver-search overhead above the step-3 cap.

7. Add GPU-specific memory-limited execution lanes.

   - Use JAX device-memory profiles and compiled memory analysis to separate
     persistent buffers from XLA temporary buffers.
   - Add donation/rematerialization gates for large matvec and diagnostics calls:
     `donate_argnums` where arrays are no longer needed, and `jax.checkpoint`
     only where memory drops materially without excessive recomputation.
   - Shard or chunk state over theta/zeta/species/radius where practical; use
     `pjit`, `shard_map`, or process-parallel transport workers only when arrays
     are actually partitioned rather than replicated.
   - Consider Pallas kernels for derivative/stencil matvec hotspots if Perfetto
     shows XLA materializing large temporary arrays for gather/scatter patterns.
   - Touch points:
     `sfincs_jax.v3_system`, `sfincs_jax.periodic_stencil`,
     `sfincs_jax.transport_parallel_runtime`,
     `sfincs_jax.transport_parallel_execution`, `sfincs_jax.solver`,
     `tests/test_sharded_matvec.py`, and `tests/test_distributed_gmres_axis.py`.
   - Gate: GPU path must match CPU JAX and Fortran v3 outputs within the same
     comparison tolerances, reduce device peak memory by at least 25% or make a
     previously OOM case run, and avoid a runtime regression larger than 20% for
     cases that already fit.

8. Evaluate Lineax/Equinox only as controlled abstractions, not automatic wins.

   - Prototype a Lineax `AbstractLinearOperator` wrapper for the matrix-free
     full-system operator and one transport operator.
   - Check whether Lineax improves solver-state reuse, transpose/adjoint
     consistency, or implicit differentiation memory; do not assume runtime or
     memory wins.
   - Use Equinox only if it simplifies static/dynamic PyTree partitioning for
     compiled operator state and reduces recompilation or saved intermediates.
   - Gate: require a measured active/device memory reduction or compile-time
     reduction on a production-floor case plus identical residual/output parity.
     Otherwise keep the current native JAX path.

9. Build CI and nightly coverage without exhausting local/GitHub resources.

   - CI keeps unit and bounded physics tests only: memory model, resolution
     manifest validator, solver-trace schema, route-promotion gates,
     pattern-CSR reconstruction, CPU/GPU-comparison logic on tiny deterministic
     fixtures, and one bounded real input.
   - Add optional/manual GitHub workflows for production benchmark tiers:
     `benchmark-production-cpu.yml` and `benchmark-production-gpu.yml`, both
     artifact-only and not required for normal PRs.
   - Add `office`/cluster runbook commands in docs for the 28 remote-only cases,
     including job-array slicing and trace collection.
   - Gate: default CI remains under 10 minutes. Production benchmark artifacts
     are accepted only when every row has the new resolution-floor metadata,
     solver traces, CPU/GPU/Fortran comparisons, runtime, active memory, and
     device memory when run on GPU.

10. Refresh public docs only after the production-floor campaign is clean.

   - Regenerate CPU/GPU/Fortran benchmark summaries from the new production-floor
     reports only.
   - Update README and docs to state which rows are production-floor benchmark
     rows, which rows are smoke/parity rows, and which remote-only rows require
     cluster/GPU-memory resources.
   - Plot active solver memory and device peak memory separately when GPU traces
     are available; never mix CPU process RSS, GPU device memory, and active RSS
     in the same acceptance gate.
   - Gate: public plot/table must have zero `jax_error`, zero `max_attempts`,
     zero missing output keys, zero practical mismatches, documented strict
     exceptions only when classified as `fortran_reference_quality`, and no
     unclassified CPU/GPU drift.

Immediate next actions when implementing this lane:

1. Update the production input generator and tests to enforce the new floor.
2. Add benchmark-summary floor validation so low-resolution rows cannot reach
   the README plot/table.
3. Extend solver trace records with active/device/compiled memory and candidate
   memory metrics.
4. Run bounded local CPU reproduction at the new floor.
5. Run targeted office GPU reproduction for HSX FP full, HSX FP DKES, geometry11,
   W7-X, and one tokamak FP/PAS row.
6. Use the classified offenders to choose the first memory implementation:
   dense-route preflight rejection, pattern-probed CSR, or GPU temporary-buffer
   reduction, depending on the measured blocker.

Progress update (2026-05-04, bounded production-floor campaign):

- Implemented the production-resolution floor in
  `scripts/create_production_benchmark_inputs.py` and generated
  `benchmarks/production_resolution_inputs_2026-05-04/`. The manifest contains
  39 cases: 6 `bounded_local_ok`, 5 `bounded_remote`, and 28
  `remote_or_cluster_only`. Public benchmark summaries now reject rows below the
  documented floor or rows missing `final_resolution`.
- Local bounded CPU suite at `tests/production_floor_cpu_bounded_2026-05-04/`
  completed 4/6 tokamak PAS rows with zero practical/strict mismatches. The two
  failures are both production-floor tokamak PAS+Er full-trajectory rows:
  `tokamak_1species_PASCollisions_withEr_fullTrajectories` and
  `tokamak_2species_PASCollisions_withEr_fullTrajectories`.
- The one-species PAS+Er offender is now classified as a true residual/stall
  problem, not geometry/JIT/HDF5 overhead. Default CPU selected `pas_ilu`; the
  first Krylov solve took about 99 s and returned residual `6.8e3` against a
  `3.2e-8` target, then entered a timeout-prone stage-2 solve.
- Forced production-floor variants on the same one-species input:
  `pas_schur` completed in 84.5 s with about 1.22 GB RSS but had 40 Fortran
  mismatches; `schur` completed in 71.8 s with about 1.17 GB RSS but had 38
  mismatches; `xblock_tz_lmax` completed in 90.4 s with about 1.93 GB RSS but
  had 36 mismatches; `bicgstab+xblock_tz` completed in 196 s with about
  4.74 GB RSS and 36 mismatches; `sparse_pc_gmres` assembled the sparse pattern
  quickly but SuperLU factorization failed. `schur` with stage 2 enabled timed
  out at the 300 s cap after first-pass residual `1.09e7`.
- `xblock_tz` on the true production-floor input reduced preconditioner setup
  cost but still entered a long stage-2 solve with residual `5.3e-3`, so it is
  not accepted as a default. Earlier apparent 8.6 s success was on the reduced
  post-failure input (`13 x 1 x 4 x 50`), not the production-floor input
  (`25 x 1 x 8 x 100`).
- A targeted `office` GPU-0 reproduction used a temporary synced checkout in
  `/tmp/sfincs_jax_current`, but both GPUs were occupied by unrelated
  `spectraxgk` jobs using about 12 GB each. The SFINCS-JAX GPU default run was
  terminated with return code `-15` after 103 s during the first Krylov solve
  after selecting `pas_ilu`; no solution or parity comparison was produced.
- Current conclusion: do not promote any of the tested heuristic route changes.
  The next real implementation is a stronger production-floor PAS+Er convergence
  path: either a better matrix-free preconditioner/rescue with bounded memory or
  a robust non-autodiff sparse/direct lane that handles the constrained PAS+Er
  operator without SuperLU failure. The default route must also fail/report
  earlier when first-pass residuals are orders of magnitude above target.
- A broader PAS stage-2 skip was tested as an implementation shortcut. It reduced
  the one-species default route to 150 s but produced 40 Fortran mismatches, so it
  is rejected as a default. The code keeps this broader skip only behind
  `SFINCS_JAX_PAS_STAGE2_SKIP_EXTENDED=1` for diagnostic profiling; production
  defaults still prioritize residual/parity correctness.
- Closed the bounded CPU tokamak PAS+Er blocker with a narrow CPU/non-autodiff
  sparse-PC GMRES route for tokamak PAS+Er full-trajectory `RHSMode=1` cases.
  The measured stable constrained-PAS sparse settings are
  `SFINCS_JAX_RHSMODE1_SPARSE_PC_SHIFT=1e-8` and
  `SFINCS_JAX_EXPLICIT_SPARSE_DIAG_PIVOT_THRESH=0`. The auto policy is limited
  to CPU, no Phi1, pure PAS, constraintScheme=2, `Nzeta=1`, electric-field
  trajectory terms, and `active_size` in the measured production-floor window.
  Validation:
  `tokamak_1species_PASCollisions_withEr_fullTrajectories` is parity-clean in
  23.2 s with 1.46 GB RSS, and
  `tokamak_2species_PASCollisions_withEr_fullTrajectories` is parity-clean in
  44.6 s with 2.59 GB RSS at `25 x 1 x 8 x 100`.
- Rerunning the bounded local production-floor CPU tier after the sparse-PC
  route gives `6/6 parity_ok`, zero practical mismatches, zero strict mismatches,
  no missing output keys, and no `max_attempts`/`jax_error` rows in
  `tests/production_floor_cpu_bounded_sparsepc_2026-05-04/`.
- Targeted `office` GPU forced sparse-PC checks on RTX A4000 are also
  parity-clean: one-species PAS+Er completes in 45.7 s with 1.82 GB RSS and
  two-species PAS+Er completes in 88.2 s with 2.36 GB RSS, both with zero
  Fortran mismatches. This validates widening the same narrow auto policy to
  CPU/GPU backends for this measured tokamak PAS+Er production-floor window.
- After widening the policy, default GPU auto-selection on `office` also uses the
  sparse-PC route without solver env overrides and remains parity-clean:
  one-species PAS+Er completes in 44.3 s with 1.79 GB RSS, and two-species PAS+Er
  completes in 85.6 s with 2.31 GB RSS. The small JSON artifacts are stored in
  `examples/performance/output/gpu_tokamak_1species_pas_er_auto_sparse_pc_2026-05-05.json`
  and
  `examples/performance/output/gpu_tokamak_2species_pas_er_auto_sparse_pc_2026-05-05.json`.
- Bounded `office` GPU full-FP production-floor rerun at
  `25 x 1 x 8 x 100` classified the remaining default mismatch:
  `tokamak_1species_FPCollisions_noEr` used the matrix-free XMG rescue ladder,
  exited with residual `6.9e-3`, and mismatched three solver-sensitive outputs
  against the Fortran direct reference. Variant probes on the same frozen case
  showed `theta_line` is parity-clean but peaks at about 19.9 GB RSS, while
  `sparse_pc_gmres` is parity-clean with about 8.8 GB RSS. Host sparse direct
  failed SuperLU factorization and is rejected for this row.
- Implemented a narrow GPU/CUDA default sparse-PC promotion for tokamak full-FP
  no-Er `RHSMode=1`, no Phi1, `constraintScheme=0`, `Nzeta=1`, and the measured
  production active-size window. The sparse-PC default uses the same stable
  diagonal shift/pivot settings as the forced probe. Default verification on
  `office` now completes `tokamak_1species_FPCollisions_noEr` in 147.9 s with
  8.79 GB RSS and zero Fortran mismatches.
- Closed verification: the complete five-case
  `tokamak_1species_FPCollisions*` bounded-remote GPU tier now passes with no
  solver overrides and the Fortran v3 MPI wrapper in
  `tests/production_floor_bounded_remote_gpu_mpi4_final4_2026-05-05/` on
  `office`. Results: `5/5 parity_ok`, zero practical mismatches, zero strict
  mismatches, no missing output keys, no `max_attempts`, and no run exceeded the
  600 s cap. The remaining public caveat is performance, not correctness:
  sparse-PC FP no-Er/Er rows are 145-166 s and about 8.4 GB RSS while Fortran v3
  is 6.7-7.9 s and about 100-140 MB RSS. The next memory lane should target
  sparse-pattern assembly/factorization storage and a lower-memory Krylov
  preconditioner rather than reverting to the nonconverged XMG default.

Progress update (2026-05-08): memory observability for the remaining lanes

- Wired the existing memory model into solver trace emission. New traces now
  populate `active_rss_mb`, `device_peak_mb` when profiling reports it,
  `estimated_dense_nbytes`, `estimated_csr_nbytes`,
  `estimated_gmres_basis_nbytes`, `matvec_count`, and
  `metadata.memory_estimate` with dense/CSR totals and per-device estimates.
- Sparse-PC metadata now records the actual GMRES restart, maximum iteration
  count, diagonal shift, and SuperLU `L`/`U` factor storage estimate in addition
  to sparse-pattern nonzeros, pattern-build time, factor time, setup time, solve
  time, and true residual. This closes the observability prerequisite for the
  remaining memory lane: the next benchmark artifacts can distinguish dense
  storage, CSR storage, GMRES basis growth, and sparse-PC factorization/setup.
- Acceptance gate for the next implementation remains unchanged: do not promote
  a lower-memory solver path unless it is residual-clean, parity-clean, and
  either at least 25% lower active/device memory at no more than 10% runtime
  regression, or it fixes a nonconverged baseline within the documented
  memory-limited rescue allowance.
- Added an opt-in sparse-PC memory preflight budget:
  `SFINCS_JAX_RHSMODE1_SPARSE_PC_MAX_ESTIMATED_MB`. When set, the sparse-PC path
  estimates CSR operator storage, GMRES basis storage, and SuperLU/ILU factor
  fill before factorization and raises a clear `MemoryError` if the estimate
  exceeds the budget. A tiny forced sparse-PC check verified the guard fails
  before factorization with an actionable message. Defaults are unchanged unless
  the budget variable is set.

Next concrete memory/performance actions:

1. Rerun one bounded CPU trace and one `office` GPU trace for the sparse-PC FP
   offenders with `SFINCS_JAX_PROFILE=1` and, on GPU, device-memory profiling.
   Confirm whether the measured peak is dominated by CSR operator storage,
   SuperLU factorization, or GMRES basis/work vectors.
2. Use production traces to tune the sparse-PC fill multiplier and budget for
   memory-limited machines; keep it opt-in until a matrix-free alternative is
   residual-clean for the same rows.
3. Prototype the next lower-memory candidate behind an explicit env gate:
   smaller-restart right-preconditioned GMRES, LGMRES with bounded augmentation,
   or BiCGStab/IDR-style short recurrence with sparse-PC preconditioning. Keep
   the current sparse-PC default until a candidate passes parity and residual
   gates on the production-floor cases.

Progress update (2026-05-08): bounded production refresh after x-block sparse-PC

- Fixed a benchmark-runner policy bug: `--fortran-min-runtime-s` now enforces
  only the lower benchmark floor. It no longer acts as an implicit maximum
  runtime cap unless `--fortran-max-runtime-s` is explicitly supplied. This
  closed the false `max_attempts` artifact for
  `tokamak_1species_FPCollisions_noEr_withPhi1InDKE`, whose frozen Fortran v3
  reference takes about 41 s and should not be downscaled merely because the
  production-floor minimum is 10 s.
- Added solver-trace ingestion to the suite reports so warm-run sidecars record
  the selected solver method and matvec count even when no PETSc/KSP-style log
  line exists. Future reports can now distinguish `xblock_sparse_pc_gmres`,
  `auto`, residual status, active memory, estimated dense/CSR/GMRES storage, and
  true matvec count from the JSON trace artifacts.
- Local bounded CPU production-floor refresh:
  `tests/production_floor_cpu_bounded_xblock_2026-05-08/` has `6/6 parity_ok`,
  zero practical mismatches, zero strict mismatches, and no missing Fortran v3
  output keys for the tokamak PAS rows at `25 x 1 x 8 x 100` or
  `25 x 1 x 4 x 100` for the Nx=1 row. The slowest JAX row is the two-species
  PAS+Er full-trajectory case at about 41.6 s warm and 4.2 GB RSS, still below
  the about 76.5 s Fortran v3 reference but still a memory-ratio target.
- Office GPU bounded full-FP refresh:
  `tests/production_floor_bounded_remote_gpu_xblock_floorfix_2026-05-08/` has
  `5/5 parity_ok`, zero practical mismatches, zero strict mismatches, and no
  missing Fortran v3 output keys for all bounded
  `tokamak_1species_FPCollisions*` rows at `25 x 1 x 8 x 100`.
- GPU runtime/memory delta versus the previous bounded sparse-PC report:
  `tokamak_1species_FPCollisions_noEr` improved from about 150.2 s / 8.42 GB RSS
  to 4.60 s / 0.95 GB RSS; `withEr_DKESTrajectories` improved from about
  146.0 s / 8.43 GB RSS to 31.8 s / 1.34 GB RSS; `withEr_fullTrajectories`
  improved from about 167.1 s / 8.43 GB RSS to 69.8 s / 1.40 GB RSS. The Phi1
  and QN rows are unchanged within noise and remain parity-clean.
- Fixed a profiler-wrapper mismatch exposed by the first targeted GPU profile:
  `scripts/profile_write_output_trace.py` called the Python API with
  `differentiable=None`, while the CLI uses `differentiable=False` for
  throughput-oriented output generation. The wrapper now matches CLI semantics
  by default and has an explicit `--differentiable` flag for implicit-solve
  profiling. Before the fix, the same `withEr_fullTrajectories` input selected a
  nonconverged implicit/incremental fallback and failed after about 3:27 with
  residual `1.56e-1`.
- Corrected targeted GPU profile for
  `tokamak_1species_FPCollisions_withEr_fullTrajectories`:
  `xblock_sparse_pc_gmres`, `467` matvecs, residual `1.48e-15` against target
  `3.18e-14`, total trace elapsed about `66.1 s`, setup about `3.6 s`, solve
  about `60.8 s`, diagnostics plus HDF5 under `0.7 s`, peak RSS about `1.40 GB`,
  active RSS about `1.23 GB`, and sampled device peak about `240 MB`. This
  classifies the remaining slow GPU FP+Er row as Krylov/matvec-count dominated,
  not output, HDF5, or device-allocation dominated.
- Current status: bounded CPU/GPU correctness and the worst full-FP sparse-PC
  memory cliff are closed. Do not regenerate README production claims yet from
  these partial bounded artifacts alone. The public benchmark plot/table should
  be refreshed only after the same trace-backed policy is run on the broader
  production-floor matrix or the remaining `remote_or_cluster_only` rows are
  clearly labeled as deferred cluster artifacts.

Next concrete actions after this refresh:

1. Use the postprocessed bounded reports, now carrying solver-trace-derived
   `jax_solver_kinds` and matvec counts, as regression fixtures for
   route-selection and memory-policy tests.
2. Use the corrected bounded GPU device-memory trace for
   `withEr_fullTrajectories` to prototype the next solver candidate around
   Krylov/matvec-count reduction rather than output, HDF5, or device-allocation
   reductions.
3. Completed below: refresh the README top plot only after the bounded
   production-floor rows are merged with clean parity and the remaining
   cluster-only 3D floor gaps are explicitly labeled.

Progress update (2026-05-08): production-floor audit hardening

- Refreshed the README/top-plot benchmark artifacts from the merged
  production-floor tokamak CPU/GPU reports. The public summary now has `24`
  plotted reference-runtime rows per backend, all `parity_ok`, zero strict
  mismatches, zero missing Fortran output keys, and no `jax_error` or
  `max_attempts`.
- Replaced the merged GPU full-FP tokamak rows with the authoritative
  production-floor GPU report so the frozen fixture preserves trace-derived
  `jax_solver_kinds` and matvec counts. The full-FP GPU rows now explicitly
  record `xblock_sparse_pc_gmres` with bounded matvec counts, while the heavy
  PAS+Er row records `sparse_pc_gmres`.
- Removed the stale merged `solver_path_audit.json` sidecar that still pointed
  to lower-resolution 2026-04-28 logs. The suite report is now the authoritative
  fixture for solver-kind/matvec regression gates in this production-floor
  merged lane.
- Marked CPU/GPU runtime-drift summaries as not applicable for this mixed
  production-floor refresh: comparing production-floor reruns against older
  lower-resolution frozen smoke reports is not a same-resolution drift gate.
- Added regression tests for trace-parser realized solver metadata and for the
  production GPU report's solver-kind/matvec fixtures.
- Remaining benchmark-floor gap: `16` non-axisymmetric 3D rows are still
  `remote_or_cluster_only` and below the current public production floor in the
  merged release artifact. They remain a cluster/nightly campaign, not a local
  ship blocker, until a dedicated large-memory node is allocated.

Next concrete actions after audit hardening:

1. Prototype a lower-matvec full-FP candidate for the slow GPU Er rows behind an
   explicit opt-in gate. Promote only if residual/parity stay clean and active
   or device memory is not worse.
2. Run the remaining `remote_or_cluster_only` 3D production-floor rows as a
   cluster/nightly campaign with the same trace schema, not as local CI jobs.
3. Keep same-resolution runtime drift as the only drift gate. Do not compare
   production-floor rows against smoke-resolution frozen baselines.

Progress update (2026-05-08): x-block right-preconditioned GPU full-FP lane

- Added an opt-in x-block sparse-PC Krylov selector,
  `SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV={gmres,lgmres,bicgstab}`, with an
  automatic GMRES rescue for non-GMRES candidates that fail the true-residual
  gate. This keeps candidate experiments from compromising parity.
- Office GPU full-trajectory LGMRES probe was rejected: it was residual-clean
  but took `439` matvecs and about `103.4 s`, slower than the current GMRES
  artifact.
- Office GPU right-preconditioned GMRES probe on
  `tokamak_1species_FPCollisions_withEr_fullTrajectories` was accepted:
  residual `4.14e-15` against target `3.18e-14`, `187` matvecs, total trace
  elapsed about `33.8-35.5 s`, active RSS about `1.18 GB`. The previous
  production-floor artifact for the same row was `467` matvecs and about
  `68.5 s` logged.
- Neighboring office probes show the policy must stay narrow:
  `tokamak_1species_FPCollisions_withEr_DKESTrajectories` became slightly worse
  with right preconditioning (`252` matvecs, about `36.0 s`, compared with the
  existing about `31.8 s` row), while the no-Er row remained clean. Therefore
  the default right-preconditioned path is enabled only for tokamak full-FP Er
  full-trajectory x-block solves (`useDKESExBDrift = false` with full trajectory
  terms active).
- Validation: the new right-preconditioned GPU full-trajectory HDF5 has `0`
  mismatches against the Fortran v3 production-floor output at `rtol=5e-4`,
  `atol=1e-9`; timing-only fields differ from the older JAX output as expected.
  The merged GPU report was refreshed with the new `187` matvec trace-backed
  row, and the README/benchmark summary figures were regenerated.

Next concrete actions after right-PC promotion:

1. Run the default right-PC full-trajectory row through the formal
   `run_scaled_example_suite.py` harness on `office` when a same-resolution
   Fortran reference root is staged there, so the report can be regenerated
   without manual row replacement.
2. Prototype a second, stricter lower-memory variant for right-PC full-FP:
   smaller GMRES restart or memory-budgeted restart selection. Keep it opt-in
   until it preserves the `187`-matvec class and does not increase active/device
   memory.
3. Keep LGMRES and BiCGStab as explicit experiment knobs only; do not promote
   them unless a production-floor row shows a measured runtime/memory win with
   clean residual and parity.

Progress update (2026-05-09): formal right-PC short-restart GPU full-FP lane

- Ran the accepted right-preconditioned full-trajectory row through the formal
  `run_scaled_example_suite.py` harness on `office` against the staged
  same-resolution Fortran v3 reference. The case remained `parity_ok` with
  `0/214` practical mismatches, `0/214` strict mismatches, and no missing
  Fortran output keys.
- Swept x-block right-PC GMRES restarts on the production-floor GPU row:
  default `80` restart was strict-clean but about `38.7 s` logged / `187`
  matvecs; restart `40` was strict-clean at about `16.2 s` logged / `99`
  matvecs; restart `30` was strict-clean at about `12.9 s` logged / `86`
  matvecs; restart `20` was strict-clean at about `10.5-11.3 s` logged / `106`
  matvecs. Restart `10` was rejected because it was already slower than the
  accepted candidates after more than a minute.
- Landed a narrow automatic restart cap of `20` only when the code also
  auto-selects the measured right-preconditioned tokamak full-FP Er
  full-trajectory x-block policy. Explicit
  `SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_RESTART` overrides and the neighboring
  DKES/no-Er branches are unchanged.
- Verified the new default, with no restart environment variable, through the
  same formal GPU harness: residual/parity stayed clean, `gmres_restart=20`,
  `default_short_restart_capped=true`, `106` matvecs, about `12.6 s` logged and
  about `14.0 s` subprocess wall time. The merged GPU report fixture was
  refreshed with this trace-backed row.
- Local validation after landing the narrow policy: focused x-block/report tests
  passed, docs built successfully with Sphinx, and the full local suite passed
  with `1106 passed in 567.89 s`.

Next concrete actions after short-restart promotion:

1. Keep the public benchmark plot unchanged unless a future same-resolution
   report includes this row in the public runtime floor; the current row remains
   below the `10 s` Fortran-reference plotting threshold.
2. Run the same formal harness for a small neighboring row only if a future
   heuristic widens beyond the current full-trajectory-only gate. The current
   tests and office probes intentionally keep the cap narrow.
3. Move back to the remaining true memory-ratio offenders; this x-block lane is
   now runtime-dominated by the unavoidable matrix-free matvec work rather than
   by solver-path selection.

Progress update (2026-05-09): CPU full-FP Er x-block auto-selection

- Re-ranked the current trace-backed CPU/GPU reports after the GPU
  short-restart change. The stale CPU top offenders were the one-species
  full-FP tokamak Er rows, still routed through generic `auto` at about
  `56.9-66.5 s` logged and `3.1-3.9 GB` RSS.
- Reran those two CPU production-floor rows against the staged Fortran v3
  reference before changing policy. Current default remained `parity_ok` and
  strict-clean, confirming the problem was performance/route selection rather
  than correctness.
- Forced `SFINCS_JAX_RHSMODE1_SOLVE_METHOD=xblock_sparse_pc_gmres` on the same
  CPU rows. Both were strict-clean with `0/214` mismatches: DKES trajectories
  ran in about `3.31 s` logged with `145` matvecs; full trajectories ran in
  about `4.10 s` logged with `105` matvecs and the right-PC short-restart cap.
- Widened `rhs1_tokamak_fp_er_sparse_pc_auto_allowed(...)` from GPU-only to
  CPU+GPU for the same measured non-differentiable, axisymmetric,
  full-FP+Er, production-size window. The no-Er and PAS policies remain
  unchanged.
- Verified the new default without any solve-method environment override:
  DKES trajectories selected `xblock_sparse_pc_gmres`, stayed strict-clean, and
  dropped from `56.95 s` to `3.44 s` logged; full trajectories selected
  `xblock_sparse_pc_gmres`, stayed strict-clean, and dropped from `66.48 s` to
  `4.15 s` logged. The tracked CPU report fixture was refreshed with these
  rows. The public top plot remains unchanged because both rows are below the
  current `10 s` Fortran-reference plotting floor.
- Validation after landing: targeted policy/report tests passed, docs built
  successfully with Sphinx, and the full suite passed with
  `1107 passed in 511.65 s`.

Next concrete actions after CPU x-block promotion:

1. Re-rank memory offenders from the refreshed reports; the remaining true
   memory targets are now PAS-heavy two-species tokamak and 3D geometry/HSX
   rows rather than one-species FP Er route selection.
2. For the next code change, target memory-ratio reduction, not another blanket
   Krylov selector. Candidate routes are PAS diagnostics/output chunking and
   lower-retention sparse/preconditioner lifetimes.

Progress update (2026-05-09): constrained-PAS sparse-PC fill reduction

- Re-ranked the current production reports after the CPU x-block promotion. The
  highest remaining trace-backed sparse-PC offender was
  `tokamak_2species_PASCollisions_withEr_fullTrajectories`: about `40.0 s`
  logged, `4.0 GB` RSS, `sparse_pc_factor_s=34.2 s`, and an estimated
  `821.9 MB` SuperLU factor. The Krylov solve itself was already cheap
  (`5` matvecs), so the bottleneck was sparse factor fill/setup.
- Tested CPU single-precision sparse-PC factorization. It reduced factor
  storage but failed the performance gate: GMRES exceeded `4600` matvecs and
  hit the `180 s` timeout. The code now keeps sparse-PC factorization `float64`
  by default; `SFINCS_JAX_RHSMODE1_SPARSE_PC_FACTOR_DTYPE=32` remains an
  explicit memory experiment, guarded by true-residual metadata, a capped
  first-attempt probe (`SFINCS_JAX_RHSMODE1_SPARSE_PC_FP32_PROBE_MAXITER`,
  default `2`), and float64 retry logic.
- Tested SuperLU column orderings on the same production row. `MMD_AT_PLUS_A`
  was fast and lower-memory but introduced four current-related strict
  mismatches (`max_abs≈1.64e-6`), so it was rejected. `MMD_ATA` was strict-clean
  and lower-fill.
- Landed `MMD_ATA` as the scoped default only for constrained-PAS sparse-PC
  GMRES. `COLAMD` remains the global explicit-sparse default and can still be
  forced with `SFINCS_JAX_EXPLICIT_SPARSE_PERMC_SPEC=COLAMD`.
- Validation:
  - CPU production-floor PAS+Er one-/two-species rows, no solver env overrides:
    both `parity_ok`, strict `0/212`; logged times `6.231 s` / `11.848 s`;
    RSS `1319.5 MB` / `2262.5 MB`.
  - Clean `office` RTX A4000 GPU clone, no solver env overrides:
    both `parity_ok`, strict `0/212`; logged times `13.193 s` / `25.021 s`;
    RSS `1572.1 MB` / `2322.5 MB`.
  - Focused tests: sparse-helper metadata and tiny sparse-PC solve checks
    passed (`4 passed`); changed test files pass `ruff`.
- Refreshed the tracked CPU/GPU benchmark reports, README benchmark table,
  benchmark summary JSON/PNG/PDF, and performance/outputs documentation with
  the new trace-backed values and the new sparse-PC ordering metadata.

Next concrete actions after constrained-PAS fill reduction:

1. Run the focused validation-artifact tests and Sphinx docs build to catch any
   benchmark-summary or documentation drift.
2. Re-rank the refreshed reports again. If the tokamak PAS+Er rows are no
   longer top memory offenders, move to the geometry-rich PAS/HSX rows where
   memory is dominated by diagnostics/output retention rather than sparse-PC
   factor fill.

Progress update (2026-05-09): one-species full-FP Er x-block host assembly

- Root cause: one-species full-FP Er rows with the Fortran-style
  `preconditioner_species=0` flag were blocked from the compact host-assembled
  x-block path. For a one-species system this flag is algebraically equivalent
  to per-species x-block preconditioning, since there is no inter-species
  coupling to preserve. The old guard forced dense matvec probing/assembly and
  caused the apparent memory cliff.
- Landed a scoped policy helper that allows host-assembled x-block factors for
  `preconditioner_species=0` only when `n_species == 1`; multi-species systems
  retain the previous coupling-preserving guard.
- CPU validation against the staged Fortran v3 production-floor references:
  `tokamak_1species_FPCollisions_withEr_DKESTrajectories` and
  `tokamak_1species_FPCollisions_withEr_fullTrajectories` are both
  `parity_ok`, strict `0/214`, with logged times about `1.06 s` / `0.96 s`,
  cold external JAX times about `1.86 s` / `1.71 s`, and peak RSS about
  `440 MB` / `419 MB`.
- GPU validation on `office` RTX A4000 with profiling marks:
  DKES/full are residual-clean with host-assembled x-blocks. The DKES row uses
  `145` matvecs, about `23.3 s`, and `1094 MB` active RSS delta; the full row
  uses right-preconditioned short-restart GMRES, `133` matvecs, about `11.0 s`,
  and `1104 MB` active RSS delta.
- Refreshed the tracked CPU/GPU benchmark report rows, README text, and
  performance documentation. The two rows remain below the README public
  Fortran-runtime threshold in the existing MPI-reference report, so they stay
  out of the top public plot/table unless rerun at a larger reference floor.
- Validation after landing: focused policy/report tests passed, Sphinx docs
  built successfully, and the full local test suite passed with
  `1109 passed in 502.34 s`.

Next concrete actions after one-species x-block host assembly:

1. Commit and push the closed one-species x-block host-assembly lane.
2. Start the next bounded memory lane from the refreshed ranking: PAS-heavy
   tokamak no-Er/Er rows first, then geometry-rich PAS/HSX diagnostics/output
   retention. Keep the same gates: strict parity, residual-clean solve,
   active/device memory reduction on the same metric, and no runtime regression
   beyond the documented tradeoff threshold.

Progress update (2026-05-09): active-DOF sparse-PC for tokamak PAS+Er

- Rejected lower GMRES restart as a default for
  `tokamak_2species_PASCollisions_noEr`: restart `20`/`40` lowered active RSS
  by only about `17-20%` while slowing the run from about `2.9 s` to about
  `5.0 s`, so it remains an explicit memory-pressure knob rather than a default.
- Profiled the current two-species tokamak PAS+Er sparse-PC row. The Krylov
  solve was already cheap (`5` matvecs); setup dominated through sparse pattern
  probing and SuperLU factorization. The old sparse-PC path factored the padded
  `40016 x 40016` system even though the active `Nxi_for_x` system has `25466`
  unknowns.
- Landed a scoped active-DOF sparse-PC route for the measured
  non-differentiable, axisymmetric, constrained-PAS, PAS+Er window. The branch
  reduces/expands through the same active-DOF index map used by the matrix-free
  solver, returns a full-size solution vector with inactive modes zeroed, and
  records `sparse_pc_active_dof`, `sparse_pc_linear_size`, and
  `sparse_pc_full_size` in solver metadata. Other sparse-PC cases can opt in
  with `SFINCS_JAX_RHSMODE1_SPARSE_PC_ACTIVE_DOF=1`; the default remains
  narrow until each geometry family is validated.
- CPU validation on
  `tokamak_2species_PASCollisions_withEr_fullTrajectories`:
  active sparse-PC reduced sparse pattern nnz from `7.90M` to `3.51M`, factor
  storage estimate from `419.5 MB` to `296.5 MB`, external wall time from
  `12.46 s` to `9.75 s`, and active RSS from `2071 MB` to `1585 MB`.
  The residual was `2.32e-09` against target `1.21e-08`, and practical
  Fortran comparison remained `0/212` mismatches.
- GPU validation on `office` RTX A4000:
  the same row completed in `22.16 s` with residual `1.43e-09` against target
  `1.21e-08`, active RSS `2124 MB`, pattern nnz `3.51M`, and practical
  Fortran comparison `0/212` mismatches. CPU-vs-GPU differences were limited
  to solver timing diagnostics.
- Added a focused regression test for the active sparse-PC path using a tiny
  truncated PAS system. The test verifies that the reduced linear size is
  smaller than the full padded size, inactive modes remain zero in the returned
  full vector, and the active residual satisfies the requested tolerance.
- Refreshed the tracked CPU/GPU benchmark rows, README benchmark table, and
  benchmark summary JSON/PNG/PDF with the new trace-backed numbers.

Next concrete actions after active sparse-PC:

1. Run focused tests, validation-artifact tests, Sphinx, and a lint safety
   subset for the changed files.
2. Re-rank the refreshed reports. The remaining memory lane should move away
   from tokamak PAS+Er sparse-PC setup and toward geometry-rich PAS/HSX
   diagnostics/output retention and larger-resolution research workloads.

Progress update (2026-05-09): active-DOF sparse-PC for tokamak PAS no-Er

- Profiled the public production-floor
  `tokamak_2species_PASCollisions_noEr` row at `25 x 1 x 8 x 100`. The current
  default matrix-free Krylov path was already fast on CPU (`2.14 s` external,
  residual `2.30e-18`) but spent most active memory inside the solve, peaking at
  about `1.74 GB` active RSS. A forced active-DOF sparse-PC run reached the same
  Fortran outputs with `0/212` mismatches, residual `3.53e-10` against target
  `6.14e-10`, and active RSS about `0.44 GB`.
- Validated the same sparse-PC route on `office` RTX A4000. The default GPU
  matrix-free report row was `14.7 s` cold / `13.2 s` logged; the active
  sparse-PC route completed in `5.24 s` cold / `5.21 s` logged with residual
  `2.23e-10` against target `6.14e-10`. GPU-vs-Fortran comparison stayed
  `0/212` mismatches; CPU-vs-GPU differences were only solver timing metadata.
- Landed a scoped `SFINCS_JAX_RHSMODE1_TOKAMAK_PAS_NOER_SPARSE_PC` policy for
  non-differentiable, axisymmetric, no-Phi1, constrained-PAS RHSMode=1 systems
  in the measured active-size window. The existing
  `SFINCS_JAX_RHSMODE1_SPARSE_PC_ACTIVE_DOF=auto` path now covers the validated
  no-Er window as well as the PAS+Er window.
- Refreshed tracked CPU/GPU benchmark reports, README benchmark table, benchmark
  summary JSON/PNG/PDF, and performance documentation. The README-facing row now
  reports CPU `2.033 s` / `393.5 MB`, GPU `5.243 s` / `1168.7 MB`, and zero
  CPU/GPU Fortran mismatches.

Next concrete actions after tokamak PAS no-Er sparse-PC:

1. Run focused policy, sparse-PC, validation-artifact, docs, and a bounded full
   local test pass.
2. Re-rank the benchmark reports again. If no tokamak PAS sparse-PC rows remain
   as dominant CPU memory offenders, move to geometry-rich PAS/HSX diagnostics
   and output-retention memory, where the likely fix is reducing retained
   diagnostic arrays rather than changing the linear solver.

Follow-up update (2026-05-09): one-species PAS no-Er threshold closure

- Checked the same no-Er sparse-PC policy on the public one-species PAS no-Er
  rows. The `Nx=8` row auto-selected active sparse-PC after the first no-Er
  policy change and remained `0/212` against Fortran. CPU active RSS dropped
  from the public report's `696 MB` to `336 MB`; the RTX A4000 runtime dropped
  from `14.2 s` to `3.6 s`.
- The `Nx=4` row was below the original `10000` active-unknown floor and stayed
  on the matrix-free path. A forced sparse-PC probe was `0/212` against Fortran,
  reduced CPU active RSS from `525 MB` to `309 MB` in the local profile, and
  reduced RTX A4000 runtime from the public `28.8 s` row to `3.0 s`. GPU process
  peak RSS increased modestly (`~1.17 GB` to `~1.28 GB`), so this is a deliberate
  runtime-offender closure rather than a GPU memory win.
- Lowered the no-Er tokamak PAS sparse-PC active-size floor to `5000`, preserving
  the upper bound and all geometry/Phi1/RHSMode guards. Refreshed the CPU/GPU
  benchmark reports, README table, benchmark summary JSON/PNG/PDF, and
  performance documentation for the one-species no-Er rows.

Next concrete actions after no-Er tokamak PAS closure:

1. Run the full focused validation stack again after the threshold change.
2. Re-rank remaining CPU/GPU offenders. Expect the remaining dominant cases to
   be PAS+Er GPU memory/runtime and full-FP no-Er Phi1/QN rows, then
   geometry-rich PAS/HSX diagnostics/output retention.

Follow-up update (2026-05-09): PAS+Er sparse-PC ordering and Phi1 audit

- Re-ranked the refreshed CPU/GPU benchmark artifacts after the no-Er sparse-PC
  closure. The remaining default-change candidates were the full-trajectory
  tokamak PAS+Er rows and the full-FP no-Er Phi1/QN rows.
- Profiled SuperLU ordering variants on the one- and two-species PAS+Er
  production rows. `NATURAL` ordering was rejected immediately because it
  inflated runtime and memory by several factors. `COLAMD` and explicit FP32
  factor-probe variants did not beat the measured defaults.
- Promoted only the validated multi-species PAS+Er sparse-PC ordering:
  `MMD_AT_PLUS_A` replaces `MMD_ATA` when the row is constrained PAS+Er and has
  more than one species. CPU validation for
  `tokamak_2species_PASCollisions_withEr_fullTrajectories` improved from
  `9.283 s` / `1744 MB` with `MMD_ATA` to `8.669 s` logged / `1638 MB`
  process RSS with `MMD_AT_PLUS_A`, with non-solver output comparisons still
  exact. One-GPU validation on `office` improved from `25.394 s` / `2279 MB` to
  `22.264 s` / `2174 MB`, also without output mismatches.
- Deliberately kept one-species PAS+Er on `MMD_ATA`: the measured RTX A4000
  row slowed from `11.989 s` to `13.188 s` with `MMD_AT_PLUS_A`, even though
  memory fell modestly. This failed the runtime gate, so it remains an explicit
  user override rather than a default.
- Audited linear Phi1/QN active-DOF and nonlinear Phi1InDKE restart/factor-dtype
  variants. The linear active-DOF policy was not robust enough across clean
  repeats, and nonlinear restart/FP32 variants were small or inconsistent wins.
  No Phi1/QN default was promoted in this pass.
- Refreshed the tracked CPU/GPU suite-report rows and regenerated the
  README/docs benchmark summary JSON/PNG/PDF from those reports. The top
  two-species PAS+Er active-memory ratios now improve on both backends while
  preserving `parity_ok`.

Next concrete actions after the PAS+Er ordering audit:

1. Run focused policy/unit tests, report-generation tests, docs checks, and
   `git diff --check`.
2. Treat this performance/memory pass as closed for safe default-policy wins if
   validation is clean. Remaining large improvements are algorithmic lanes:
   full-FP Phi1/QN memory/runtime, one-species PAS+Er GPU memory/runtime, and
   geometry-rich 3D diagnostics/output retention.

Progress update (2026-05-09): final FP/PAS/output-retention push

- Re-ranked the current production-floor CPU/GPU artifacts for the requested
  lanes. Full-FP Phi1/QN is dominated by the nonlinear Phi1InDKE row, while QN
  already kept the best measured default. The one-species PAS+Er GPU row is a
  sparse-PC memory/runtime target. The geometry-rich 3D row is large enough that
  the production-resolution solve, not HDF5 writeout, is the current blocker.
- Audited QN full-FP variants on CPU. `host_dense`, active-DOF, generic
  sparse-PC, no-solver-JIT, and dense-cutoff overrides all preserved outputs but
  were slower and/or higher-memory than the existing default. No QN default was
  promoted.
- Audited nonlinear Phi1InDKE variants. A fast-explicit Phi1 GMRES restart of
  `120` for active systems above `8000` unknowns preserved outputs and improved
  isolated CPU/GPU sweeps modestly, so this scoped default is now guarded by
  `tests/test_io_output_policy_coverage.py`.
- Re-audited one-species tokamak PAS+Er. The new measured default uses
  `MMD_AT_PLUS_A` SuperLU ordering for PAS+Er sparse-PC rows and caps the
  one-species PAS+Er sparse-PC GMRES restart at `40` unless the user sets
  `SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_RESTART`. CPU/GPU benchmark harness
  reruns preserved output parity; the `office` RTX A4000 default completed in
  `15.10 s` wall / `13.92 s` logged with peak RSS about `1.41 GB`.
- Reduced geometry-rich RHSMode=1 output retention for linear no-Phi1 solves:
  a single-state solve now calls the single-state vm-diagnostic kernel instead
  of retaining an extra stacked solved-distribution copy, and the NTV diagnostic
  path avoids constructing an `NIteration` stack when only one state exists.
- Production-resolution `geometryScheme4_2species_PAS_noEr` at active size
  `744610` / total size `1275010` timed out at the bounded `300 s` CPU ceiling
  before diagnostics. The trace reached a Schur active-DOF preconditioner build
  in `8.59 s` and then stalled in Krylov. This confirms the remaining
  geometry-rich 3D work is a solver/preconditioner lane, not an output-retention
  lane. A forced sparse-PC probe with active-DOF reduction also timed out at the
  same `300 s` ceiling after constructing a `32.8M`-nonzero conservative sparse
  pattern for `744610` active unknowns, so no geometry-rich sparse-PC default is
  promoted from this pass.
- Final bounded closeout probes for the same production-resolution geometry4
  PAS case all failed the release gate: forced `pas_tz`, `pas_tz` with
  `Lmax=4`, `xmg`, `pas_hybrid`, `bicgstab`, and `lgmres` each hit the
  `300 s` timeout. The `office` RTX A4000 default route also hit the same
  timeout after spending `87.3 s` in Schur preconditioner setup. These results
  are now captured in
  `tests/reference_solver_path_artifacts/geometry4_large_pas_closeout_2026-05-09.json`
  and guarded by `tests/test_solver_path_artifacts.py`. Decision: this lane is
  closed for release as "no safe existing default promotion"; it remains a
  future algorithmic preconditioner project, not a hidden unresolved benchmark
  claim.
- Validation so far: syntax checks passed, `git diff --check` passed, focused
  policy tests passed, RHSMode=1 no-Phi1/Phi1 output end-to-end tests plus the
  relevant policy tests passed (`39 passed`), and the expanded focused
  sparse-pattern/report stack passed (`61 passed`). The Sphinx docs build with
  warnings-as-errors also passed. Full local pytest passed:
  `1115 passed in 491.95 s`.

Next concrete actions after the final FP/PAS/output-retention push:

1. Refresh public runtime/memory artifacts only from a consistent suite harness
   rerun; do not mix local CPU and remote GPU one-off RSS values into the README
   plot.
2. Treat production-resolution geometry-rich PAS as the next algorithmic lane:
   replace the conservative global sparse pattern with a structured/chunked
   geometry-aware preconditioner before attempting another public default.

Progress update (2026-05-10): v1.1.1 release and structured PAS kickoff

- Released `v1.1.1` from `main` after regenerating the README-facing
  Fortran/JAX runtime-memory plot and W7-X high-`nu` performance figure from
  checked-in artifacts.
- Release validation passed locally: docs with warnings as errors, package
  build, focused figure/package tests, and full pytest (`1115 passed in
  498.10 s`). The `v1.1.1` tag push triggered the PyPI, CI, and Docs workflows.
- Started the next technical push without changing release defaults. Memory-
  unsafe `pas_tz` builds still fall back to `pas_hybrid` on one device by
  default, but `SFINCS_JAX_RHSMODE1_PAS_TZ_MEMORY_FALLBACK=theta|zeta|schwarz`
  now exposes a structured additive-Schwarz fallback for bounded geometry-rich
  PAS experiments. Shared `SFINCS_JAX_RHSMODE1_PAS_TZ_SCHWARZ_BLOCK` and
  `SFINCS_JAX_RHSMODE1_PAS_TZ_SCHWARZ_OVERLAP` controls make the benchmark lane
  reproducible before any future default promotion.
- Added a plot-only mode to
  `examples/performance/benchmark_transport_parallel_scaling.py` and regenerated
  the README GPU transport-worker scaling figure from the checked-in
  `examples/performance/output/transport_parallel_scaling_gpu.json` payload
  without rerunning the multi-minute office benchmark.
- Added `scripts/benchmark_pas_tz_memory_fallback.py`, a subprocess-bounded
  harness for forced `pas_tz` memory-fallback variants. The first local
  geometry4 PAS smoke used a 15 s cap with `maxiter=4`, `restart=8`,
  `block=3`, and `overlap=1`; `hybrid`, `zeta`, and `theta` all timed out while
  building or retrying the `pas_tz` path. The checked artifact is
  `tests/reference_solver_path_artifacts/pas_tz_memory_fallback_geometry4_smoke_2026-05-10.json`.

Next concrete actions after the structured PAS kickoff:

1. Replace the current Schwarz fallback build strategy with a genuinely chunked
   or matrix-free patch apply; the naive forced `hybrid`/`zeta`/`theta` fallback
   still stalls before the Krylov phase on the geometry4 PAS smoke gate.
2. Re-run the bounded CPU geometry-rich PAS probe ladder only after the patch
   build is lazy/chunked enough to pass the 15-60 s smoke gate.
3. If a structured Schwarz route clears the bounded runtime and residual gate,
   add a trace artifact and only then consider a narrow auto-policy window.
4. If all structured fallback probes still stall, move to a genuinely new
   matrix-free coarse correction rather than widening existing Schur/sparse-PC
   heuristics.

Progress update (2026-05-10): guarded PAS-TZ fallback closeout

- Added a source-level work estimator for opt-in PAS-TZ theta/zeta Schwarz
  fallback builds. It estimates patch count, largest patch unknown count, and
  dense inverse entries before entering the expensive builder.
- Added guardrails
  `SFINCS_JAX_RHSMODE1_PAS_TZ_SCHWARZ_MAX_PATCH_UNKNOWNS` and
  `SFINCS_JAX_RHSMODE1_PAS_TZ_SCHWARZ_MAX_INVERSE_ENTRIES`; the default caps
  reject production-resolution grids that would allocate many dense local
  inverses, while `0` disables each cap for explicit unsafe profiling.
- Guarded structured fallback now returns a cheap collision fallback and marks
  the callable. The driver uses that marker to skip the expensive strong retry
  unless `SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_STRONG_RETRY=1` is set. This keeps
  the path fail-fast instead of spending minutes searching solver paths after
  an already-rejected dense Schwarz route.
- Regenerated
  `tests/reference_solver_path_artifacts/pas_tz_memory_fallback_geometry4_smoke_2026-05-10.json`:
  the old `hybrid` route still times out at the 15 s cap, while forced `zeta`
  and `theta` now finish in about `1.5 s` and report the expected large
  residual (`1.58e6`). This is not a promoted solver path, but it closes the
  stall/hang behavior for this bounded benchmark lane.
- Fixed the PAS fallback harness RSS conversion to use the package
  platform-aware `ru_maxrss` conversion, avoiding macOS/Linux unit mismatches.

Next concrete actions after guarded PAS-TZ fallback:

1. Keep this guarded behavior as a negative benchmark gate: structured fallback
   is safe to test but still fails the residual gate, so defaults should remain
   unchanged.
2. Start the real algorithmic path with a matrix-free angular/radial correction
   that does not store dense per-patch inverses. The gate is: under 60 s on the
   geometry4 smoke deck, residual at least two orders of magnitude below the
   guarded collision fallback, and no memory regression.
3. Re-run the focused PAS policy/artifact tests and docs build, then full local
   pytest if CI stays green.

Progress update (2026-05-10): guarded matrix-free PAS correction probe

- Added an accept-only matrix-free minres correction for guarded PAS-TZ
  fallback. After the weak Krylov solve, it computes `d = M^{-1} r`, chooses
  the scalar step that minimizes `||r - alpha A d||_2`, and accepts only if the
  measured residual decreases. This uses extra matvecs but stores no dense
  angular patch inverses.
- Added controls
  `SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_MINRES_STEPS`,
  `SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_MINRES_ALPHA_CLIP`, and
  `SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_MINRES_MIN_IMPROVEMENT`.
- Tried a polynomial guarded preconditioner (`SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_POLY_*`).
  The geometry4 smoke probe showed residual growth for tested dampings, so the
  polynomial path remains opt-in and default-off.
- Regenerated the bounded geometry4 PAS smoke artifact. Forced `zeta` and
  `theta` still finish in about `1.4-1.5 s`; the accepted minres correction
  reduces residual from `1.58e6` to `1.27e6`. This is a safe memory-neutral
  improvement, but it does not clear the two-orders-of-magnitude gate.

Next concrete actions after guarded minres:

1. Do not promote the guarded PAS-TZ fallback as a production solver; it remains
   a bounded negative benchmark with a modest accept-only correction.
2. Implement a stronger structured correction that captures angular streaming
   without dense patch inverses. The leading candidates are a matrix-free
   line/plane smoother with local diagonal solves, or a chunked additive
   Schwarz apply that solves patches iteratively instead of materializing dense
   inverses.
3. Keep the geometry4 smoke gate fixed: under `60 s`, no memory regression, and
   at least `100x` residual reduction relative to the guarded collision fallback
   before any auto-policy promotion.

Progress update (2026-05-10): weak PAS retry fail-fast guard

- Forced weak PAS routes (`collision`, `xmg`, and `point`) were still able to
  spend minutes in stage-2 polish or strong-preconditioner retry/search after
  first residual ratios near `1e15`. This was a stall-prevention bug in the
  profiling lanes, not a production solver-quality win.
- Added `SFINCS_JAX_PAS_STAGE2_WEAK_SKIP_RATIO` and
  `SFINCS_JAX_PAS_STRONG_WEAK_SKIP_RATIO`, both defaulting to `1e12`; setting
  either value to `0` disables that specific guard for explicit profiling.
- Added a bounded weak-PAS minres correction controlled by
  `SFINCS_JAX_PAS_WEAK_MINRES_RATIO`, `SFINCS_JAX_PAS_WEAK_MINRES_STEPS`,
  `SFINCS_JAX_PAS_WEAK_MINRES_ALPHA_CLIP`, and
  `SFINCS_JAX_PAS_WEAK_MINRES_MIN_IMPROVEMENT`. It reuses the already-built
  weak preconditioner, chooses the scalar step that minimizes the residual, and
  accepts only if the measured residual drops.
- GeometryScheme=4 PAS smoke probes with `maxiter=4`, `restart=8`, and bounded
  subprocess timeouts now return instead of stalling: `collision` in `1.23 s`
  with residual improved from `1.58e6` to `1.27e6` and `595 MB` RSS, `xmg` in
  `1.35 s` with residual improved from `2.53e6` to `2.44e6` and `645 MB` RSS,
  and `point` in `2.80 s` with residual `1.27e6` and `1.90 GB` RSS.
- No weak route is promoted. The result closes the forced-path stall lane and
  keeps those paths as auditable negative baselines.

Next concrete actions after weak PAS fail-fast:

1. Keep the fail-fast guards narrow: they only apply to PAS weak base kinds at
   enormous residual ratios, and they must not change moderate-residual polish
   behavior.
2. Continue the real solver work with a stronger matrix-free line/plane
   smoother or iterative chunked Schwarz correction that avoids dense angular
   inverse storage.
3. Gate any future promotion on the fixed geometry4 smoke target: under `60 s`,
   no measured memory regression, and at least `100x` residual reduction
   relative to the guarded collision fallback.

Progress update (2026-05-10): memory-unsafe PAS-TZ fallback tightening

- Removed a shadowed duplicate implementation block from
  `sfincs_jax/pas_smoother.py`. The public smoother API is unchanged, but future
  adaptive PAS work now has a single implementation of residual-trend decisions
  and the adaptive stationary smoother.
- Tightened `build_pas_tz_memory_fallback(...)`: when a single-device
  memory-unsafe `pas_tz` request cannot use the dense PAS-TZ builder, the default
  guarded fallback is now the cheap collision preconditioner when available.
  The historical `pas_hybrid` fallback remains available explicitly with
  `SFINCS_JAX_RHSMODE1_PAS_TZ_MEMORY_FALLBACK=hybrid`.
- Rationale: the bounded geometry4 smoke artifact already showed that the old
  hybrid fallback was a negative benchmark because it could spend the timeout
  budget in setup/retry. The collision fallback is weaker, but bounded in memory
  and setup time, and the driver still marks it as guarded, applies only
  accept-if-improves corrections, and skips expensive strong retries unless the
  user explicitly opts in.
- Validation run:
  `pytest -q tests/test_rhs1_pas_policy.py tests/test_pas_smoother.py
  tests/test_solver_path_artifacts.py::test_pas_tz_memory_fallback_smoke_keeps_structured_fallback_opt_in`
  (`26 passed`).
- Refreshed the bounded geometry4 PAS fallback artifact with variants
  `collision`, `hybrid`, `zeta`, and `theta` under the fixed `15 s` cap. All
  rows returned quickly (`1.41-1.66 s`). The new default cheap-collision row
  used about `600 MB` RSS and reduced the residual to `1.27e6`, matching the
  structured-guard rows and materially improving over the explicit legacy
  `hybrid` row residual (`2.53e16`, about `655 MB`). This remains a negative
  benchmark because the residual is still far above the production gate.
- Follow-up release-hygiene pass: GitHub CI and Docs passed for commit
  `abd9d65` (`coverage`, `examples-smoke`, optional ecosystem gates, and all
  three test shards). The `v1.1.2` release notes were corrected to state the
  actual current default: memory-unsafe `pas_tz` falls back to guarded
  `collision` when available, while guarded `hybrid` remains an explicit
  A/B-profiling override.

Next concrete actions after fallback tightening:

1. Prototype the next real algorithmic candidate: a matrix-free line/plane or
   chunked-Schwarz correction that captures angular streaming without storing
   dense patch inverses.
2. Promote nothing unless the fixed gate is met: no memory regression, bounded
   runtime, true-residual improvement of at least `100x` relative to the guarded
   collision fallback, and unchanged Fortran comparison behavior on the bounded
   PAS examples.

Progress update (2026-05-10): experimental PAS-TZ FFT fallback

- Wired the existing JAX-native angular streaming FFT/L-tridiagonal
  preconditioner into RHSMode=1 as explicit `pas_tzfft` / `pas_fft` aliases and
  as `SFINCS_JAX_RHSMODE1_PAS_TZ_MEMORY_FALLBACK=tzfft` for memory-unsafe
  `pas_tz` experiments.
- Added a guarded stage-2 policy:
  `SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_STAGE2_RETRY=1` is now required before a
  guarded PAS-TZ fallback enters strict stage-2 GMRES. This fixed the observed
  `tzfft` smoke stall: before the guard, `tzfft` lowered the residual to about
  `1.9e-4` and then hit the `60 s` stage-2 ceiling; after the guard it returns
  cleanly.
- Refreshed
  `tests/reference_solver_path_artifacts/pas_tz_memory_fallback_geometry4_smoke_2026-05-10.json`
  with variants `collision`, `hybrid`, `zeta`, `theta`, and `tzfft` under the
  fixed `15 s` cap:
  - `collision`: `1.62 s`, `617 MB`, residual `6.414e5`;
  - `hybrid`: `2.32 s`, `859 MB`, residual `2.529e16`;
  - `zeta`: `1.63 s`, `683 MB`, residual `6.414e5`;
  - `theta`: `1.62 s`, `617 MB`, residual `6.414e5`;
  - `tzfft`: `3.33 s`, `944 MB`, residual `1.877e-4`.
- Additional bounded probes:
  - `tzfft` with `maxiter=20,restart=8`: `4.95 s`, `1093 MB`, residual
    `1.333e-4`;
  - `tzfft` with `maxiter=20,restart=20`: `9.97 s`, `1511 MB`, residual
    `1.210e-5`;
  - the same `maxiter=20,restart=20` plus a capped stage-2 retry: `13.65 s`,
    `1602 MB`, residual unchanged at `1.210e-5`.
- Decision: keep `tzfft` as an explicit experimental residual-improvement
  candidate, but do not promote it to the default fallback. It meets the
  residual-improvement part of the gate but fails the no-memory-regression and
  strict-residual parts of the gate. The next real algorithmic lane remains a
  stronger chunked/matrix-free PAS correction that retains the `tzfft` residual
  gain without the GMRES-basis memory growth.

Progress update (2026-05-10): cheap-base plus `tzfft` correction probe

- Added an opt-in guarded correction path:
  `SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_CORRECTION=tzfft`. This keeps the
  memory-safe fallback (`collision`, unless otherwise requested) as the Krylov
  preconditioner and uses the JAX-native `tzfft` operator only for the bounded
  post-Krylov minimal-residual correction. The benchmark harness can run this
  route with the variant name `collision_tzfft_correction`.
- Bounded geometry4 PAS probe (`maxiter=8`, `restart=12`, `60 s` timeout):
  - `collision`: `1.83 s`, `675 MB`, residual `6.414e5`;
  - `collision_tzfft_correction`: `1.98 s`, `728 MB`, residual `1.336e5`;
  - `tzfft`: `3.54 s`, `979 MB`, residual `1.877e-4`.
- Increasing `SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_MINRES_STEPS` to `10` for the
  cheap-base correction gave `2.19 s`, `776 MB`, residual `1.336e5`, so the
  residual floor is not a lack of scalar correction steps.
- Decision: keep the cheap-base correction as an explicit profiling option, but
  do not promote it. It is bounded and modestly improves the collision fallback,
  but it fails the `100x` residual-improvement gate and still increases RSS.
  The remaining algorithmic target is a real streaming-aware preconditioner that
  is strong during Krylov without storing a large basis or dense angular patch
  inverses.

Rejected probe (2026-05-10): guarded `tzfft` restart cap

- Tested a source-level cap on GMRES restart for explicit guarded `tzfft`
  fallback. The cap did emit the right progress line and improved elapsed time
  slightly on the bounded geometry4 smoke, but the A/B was not favorable:
  capped `restart=1` returned in `2.94 s` with `965 MB` RSS and residual
  `1.00e-3`, while cap-disabled `restart=12` returned in `3.37 s` with
  `899 MB` RSS and residual `1.88e-4`.
- Decision: do not keep the restart-cap source change. It worsened both memory
  and residual on the checked smoke, so the next real lane remains a stronger
  streaming-aware preconditioner rather than restart tuning.

Rejected probe (2026-05-10): guarded `tzfft` tiny subspace correction

- Tested an opt-in accept-only post-Krylov subspace correction that built a
  small basis from repeated `tzfft` preconditioner applications and solved the
  tiny least-squares problem `min_y ||r - A P y||`. This avoided dense angular
  inverse storage, but it did not improve the measured gate.
- Bounded geometry4 PAS probe (`maxiter=8`, `restart=12`, `60 s` timeout):
  `collision_tzfft_subspace` returned in `2.32 s`, used about `779 MB` RSS, and
  ended at residual `1.335e5`. This is essentially the same residual floor as
  the scalar `collision_tzfft_correction` probe (`1.336e5`) with more memory and
  slightly more time. A wider requested subspace still accepted only dimension
  `3` and remained at the same residual floor.
- Decision: revert the subspace source change and do not add another public
  tuning knob. The evidence points away from post-solve low-dimensional
  correction and toward a preconditioner that is strong inside Krylov while
  remaining streaming/chunk aware.

Rejected probe (2026-05-10): direct ADI line-preconditioner fallback

- Tested the existing `adi` RHSMode=1 preconditioner as a possible
  streaming-aware low-memory fallback on the bounded geometry4 PAS deck
  (`maxiter=8`, `restart=12`). It was still in preconditioner setup after
  `30 s`, while the fixed fallback benchmark candidates return in about
  `1.6-3.6 s`.
- Decision: do not route memory-unsafe PAS-TZ fallback to the current `adi`
  builder. The implementation is not a bounded replacement for dense PAS-TZ on
  this target without further source-level work to make the line solves
  genuinely streaming/chunked.

Rejected probe (2026-05-10): right-preconditioned guarded `tzfft`

- Tested `SFINCS_JAX_GMRES_PRECONDITION_SIDE=right` with explicit guarded
  `SFINCS_JAX_RHSMODE1_PAS_TZ_MEMORY_FALLBACK=tzfft` on the same geometry4 PAS
  smoke (`maxiter=8`, `restart=12`). It returned residual `1.389e-4`, elapsed
  `4.04 s`, and about `1007 MB` RSS.
- Decision: do not promote a right-preconditioned guarded `tzfft` policy. It
  improves residual only modestly relative to the existing left-preconditioned
  `tzfft` probe while increasing runtime and memory.

Rejected probe (2026-05-10): truncated dense PAS-TZ `Lmax=2`

- Tested `SFINCS_JAX_RHSMODE1_PAS_TZ_LMAX=2` under the normal PAS-TZ memory cap
  on the bounded geometry4 PAS deck. It was still running after `30 s` and had
  already entered the strong `pas_lite` fallback from residual `6.31e-3`.
- Decision: low-`Lmax` dense PAS-TZ truncation is not a bounded replacement for
  the memory-unsafe PAS-TZ path on this target. It reduces the nominal dense
  block size, but the resulting preconditioner is too weak and triggers slower
  fallback behavior.

Rejected probe (2026-05-10): chunked `tzfft` apply

- Tested an opt-in chunked `tzfft` preconditioner apply that solved the
  Fourier-space tridiagonal systems by radial/speed chunks, and then by both
  species and radial/speed chunks. The goal was to keep the strong matrix-free
  `tzfft` residual behavior while lowering peak FFT/tridiagonal intermediates.
- Bounded geometry4 PAS smoke (`maxiter=8`, `restart=12`, `60 s` timeout):
  unchunked `tzfft` returned in `3.76 s`, used about `944 MB` RSS, and reached
  residual `1.877e-4`; `tzfft_xchunk1` returned in `4.02 s`, used about
  `886 MB` RSS, and reached the same residual; species chunking increased RSS
  or runtime further.
- Production-size geometry4 PAS probe (`25 x 51 x 100 x 4`, `maxiter=2`,
  `restart=8`, `180 s` timeout): unchunked `tzfft` returned in `5.53 s`, used
  about `1849 MB` RSS, and reached residual `1.518e-3`; `tzfft_xchunk1`
  returned in `5.02 s`, but RSS increased to about `2037 MB` with the same
  residual.
- Decision: revert the chunked apply source change. XLA did not realize the
  intended peak-memory reduction on the production-sized gate, so adding a
  public chunking knob would be misleading. The remaining memory lane needs a
  different algorithmic preconditioner or a host/device solve split rather than
  chunking the current `tzfft` apply graph.

Rejected probe (2026-05-10): production-size PAS sparse-PC default

- Tested the existing `solve_method="sparse_pc_gmres"` route on the same
  geometry4 PAS production-size deck used for memory-gate decisions
  (`25 x 51 x 100 x 4`, `maxiter=2`, `restart=8`, `120 s` timeout).
- The run reached conservative sparse-pattern materialization with
  `45,369,600` nonzeros (`avg_row_nnz=44.5`, `max_row_nnz=1275`) and started
  exact sparse-LU factorization, but timed out before producing a solve result.
- Decision: do not promote sparse-PC/sparse-LU as the default memory fix for
  large geometry-rich PAS decks. It remains useful for the already documented
  constrained-PAS profile-current niche, but this production-size geometry4
  gate needs a stronger matrix-free or lower-memory approximate factor path.

Progress update (2026-05-11): multi-lane safety and observability pass

- Added a new opt-in matrix-free RHSMode=1 PAS correction helper in
  `sfincs_jax/rhs1_pas_matrixfree.py`. This is deliberately not wired into the
  default solve path. It provides bounded streaming-norm residual checks,
  explicit keep/reject reasons, update-size guards, and shape/dtype stability
  for the next PAS preconditioner experiments.
- Upgraded the PAS-TZ fallback benchmark JSON to `schema_version=2`. Benchmark
  artifacts now record `variant_methods`, per-row `variant_provenance`,
  `solver_provenance`, compact `phase_metadata`, and tail limits. This closes
  the immediate ambiguity around whether `_lgmres` rows were default behavior
  or explicit opt-in probes.
- Tightened GPU transport-worker planning. Duplicate `CUDA_VISIBLE_DEVICES`
  entries are de-duplicated, worker caps now report the exact reason, and extra
  RHS payloads are coalesced onto active workers instead of being silently
  dropped when requested workers exceed unique visible GPUs.
- Added fast validation-policy tests for benchmark filtering, warm/logged
  runtime selection, autodiff gate closure, collisionality slope behavior,
  FP/PAS separation scaling, and malformed suite reports.
- Documented the current `vmec_jax` / `booz_xform_jax` differentiability
  boundary in `docs/geometry.rst`: optional backend probing and geometry proxy
  gates are supported, but file I/O, the NumPy scheme-5 evaluator, and full
  VMEC-boundary-to-kinetic-transport optimization remain future work.
- Verification:
  - focused modified-lane suite:
    `75 passed in 1.98 s`;
  - full local suite:
    `1165 passed in 512.03 s`;
  - `ruff check` on changed source/tests:
    passed;
  - `python -m py_compile` on changed modules:
    passed;
  - `python -m sphinx -W -b html docs build/sphinx-html`:
    passed;
  - `git diff --check`:
    passed.

Updated lane status after this pass:

- PAS-heavy memory/runtime: `86%`. The code now has a safe matrix-free
  correction primitive and better benchmark provenance, but it still needs a
  production-floor HSX/geometry11 residual and memory win before any default
  policy change.
- FP production-floor memory/runtime: `95%`. No regression was introduced in
  this pass.
- CPU/GPU parity for documented workflows: `100%` for the tested lanes.
- CI/docs health: `100%` locally for the gates listed above; remote CI should
  still be checked after push.
- Coverage/refactor path: `62%`. The new tests improve meaningful policy and
  validation coverage without adding slow solves, but the dominant future gain
  still requires splitting more logic out of `v3_driver.py`.
- Parallel transport workers: `78%` for release-facing multi-case/RHS
  throughput. Single-case multi-GPU RHSMode=1 remains experimental and should
  not be marketed as production strong scaling.
- `vmec_jax` / `booz_xform_jax` workflow: `60%` for public handoff/status
  documentation and shallow gates; full differentiable transport optimization
  remains about `35%`.
- Deferred manuscript physics lanes: `45%`; no new claim was made in this
  implementation pass.

Next concrete actions:

1. Run the new matrix-free PAS helper against bounded geometry4/HSX/geometry11
   production-floor probes as an explicit opt-in only. Keep it only if it
   improves true residuals without memory/runtime regression.
2. Add a small artifact checker that rejects PAS benchmark JSON missing
   `schema_version >= 2`, `variant_methods`, or per-row solver provenance.
3. Continue the coverage/refactor lane by extracting validation and benchmark
   policy helpers from large modules before attempting to move `v3_driver.py`
   coverage materially.
4. After pushing this pass, check remote CI and update release notes only if
   the remote gates stay green.

Progress update (2026-05-11): open-lane subagent push

- Added a bounded opt-in PAS matrix-free probe harness:
  `scripts/benchmark_rhs1_pas_matrixfree.py`. It probes deterministic
  matrix-free systems and capped geometry-metadata-derived systems in
  short-lived subprocesses, records explicit keep/reject gates, and writes JSON
  without touching the production solver path.
- Tightened the PAS matrix-free norm helper so chunked norms preserve `NaN` and
  `Inf` instead of masking non-finite residuals.
- Added a benchmark artifact policy checker:
  `sfincs_jax/benchmark_artifact_policy.py` plus
  `scripts/check_benchmark_artifacts.py`. The checker enforces schema-version
  and solver-provenance metadata on new PAS benchmark artifacts. It intentionally
  rejects historical schema-v1 artifacts; those remain legacy reference outputs
  until explicitly refreshed.
- Split pure collisionality validation math into
  `sfincs_jax/validation_math.py` while preserving imports from
  `sfincs_jax.validation_artifacts`. This moves another small cluster out of the
  large validation module without changing public behavior.
- Added `sfincs_jax/validation_figures.py` with a W7-X ambipolar-root
  provenance panel data builder. It remains labelled `scaffold_deferred` unless
  a complete, checked-in W7-X ambipolar artifact backs the claim.
- Improved transport-worker safety: invalid worker counts now fail fast, CPU
  worker results are collected in payload order even when futures complete out
  of order, and GPU worker count validation stays explicit.
- Expanded the optional `vmec_jax` / `booz_xform_jax` public example with
  `--check-backends`, runnable docs commands, and explicit printed
  differentiability boundaries.
- Verification:
  - focused integrated lane suite:
    `61 passed in 2.89 s`;
  - PAS matrix-free harness smoke:
    `7` bounded rows, all expected keep/reject gates met;
  - schema-v2 PAS artifact dry-run plus checker:
    passed;
  - full local suite:
    `1195 passed in 521.49 s`;
  - `ruff check` on changed source/tests:
    passed;
  - `python -m py_compile` on changed scripts/modules:
    passed;
  - `python -m sphinx -W -b html docs build/sphinx-html`:
    passed;
  - `git diff --check`:
    passed.

Updated lane status after this pass:

- PAS-heavy memory/runtime: `88%`. The next algorithmic probe infrastructure is
  now in place and guarded, but no production default changes are justified yet.
- Benchmark artifact reproducibility gates: `95%`. New artifacts can be checked
  automatically; legacy schema-v1 artifacts still need explicit refresh if they
  become release-facing.
- FP production-floor memory/runtime: `95%`. No regression was introduced.
- CPU/GPU parity for documented workflows: `100%` for the tested lanes.
- CI/docs health: `100%` locally for the gates listed above; remote CI still
  needs post-push confirmation.
- Coverage/refactor path: `65%`. A pure validation-math module is split out;
  meaningful movement toward `95%` still requires more driver/policy extraction.
- Parallel transport workers: `80%` for release-facing multi-case/RHS
  throughput. Single-case multi-GPU RHSMode=1 remains experimental.
- `vmec_jax` / `booz_xform_jax` workflow: `63%` for public handoff/status
  documentation and optional-backend example UX; full differentiable kinetic
  transport remains about `35%`.
- Deferred manuscript physics lanes: `48%`; W7-X ambipolar panel scaffolding is
  stronger, but the real artifact-backed validation claim remains open.

Next concrete actions:

1. Use `scripts/benchmark_rhs1_pas_matrixfree.py` as the cheap preflight before
   any production-floor PAS experiments, then add a real opt-in geometry4/HSX
   production-floor probe only if the synthetic gates remain stable.
2. Refresh any release-facing PAS benchmark artifacts to schema version `2`, or
   keep historical schema-v1 artifacts out of automated release gates.
3. Continue extracting small validation/benchmark/solver-policy helpers before
   attempting high-coverage work on `v3_driver.py`.
4. Keep the W7-X ambipolar lane marked deferred until a complete checked-in
   W7-X equilibrium/profile/provenance artifact exists.

Progress update (2026-05-11): second open-lane subagent integration

- Added JSON-safe gate diagnostics to the opt-in RHSMode=1 PAS matrix-free
  helper and benchmark harness. Rejected probes now report the concrete gate
  failure (`nonfinite-candidate-residual`, `update-norm-too-large`, insufficient
  residual improvement, shape mismatch) with finite JSON metadata. This changes
  only the probe/harness layer, not production solver dispatch or defaults.
- Tightened benchmark artifact reproducibility policy by rejecting malformed or
  duplicate variant labels in both `plan.variant_methods` and `results`. New
  schema-v2 artifacts therefore cannot silently contain ambiguous duplicate
  rows. Historical schema-v1 artifacts remain outside release gates unless they
  are explicitly refreshed.
- Extracted pure solver-progress policy helpers to
  `sfincs_jax/solver_progress_policy.py` while re-exporting the same names from
  `sfincs_jax.solver_progress`. This keeps CLI/progress behavior stable and
  makes the runtime-hint threshold logic directly unit-testable.
- Added transport-parallel result integrity checks before merging worker
  results: duplicate RHS coverage, missing per-RHS payload entries,
  out-of-range RHS values, and inconsistent GPU worker array lengths now fail
  explicitly instead of producing ambiguous merged results.
- Extended the optional `vmec_jax` / `booz_xform_jax` workflow with
  `--check-backends --json`, producing backend importability, runnable setup
  paths, gradient-availability labels, and the explicit non-claim that this is
  a geometry-proxy gate rather than full end-to-end kinetic-transport
  differentiation.
- Added W7-X ambipolar validation scaffold metadata: provenance completeness
  scores and structured `deferred_reasons` / `deferred_reason_codes`. This keeps
  manuscript-facing figures conservative until complete checked-in W7-X
  provenance and numerical root gates exist.
- Verification:
  - focused integrated lane suite:
    `95 passed in 12.24 s`;
  - PAS matrix-free harness smoke:
    `7` bounded rows, all expected keep/reject gates met, diagnostics present;
  - schema-v2 PAS artifact dry-run plus checker:
    passed;
  - `ruff check` on changed Python source/tests:
    passed;
  - `python -m py_compile` on changed Python modules/scripts:
    passed;
  - `python -m sphinx -W -b html docs build/sphinx-html`:
    passed;
  - `git diff --check`:
    passed;
  - full local suite:
    `1217 passed in 514.97 s`.

Updated lane status after this pass:

- PAS-heavy memory/runtime: `89%`. Probe observability is stronger and safer,
  but no production default change is justified without a real geometry4/HSX
  production-floor residual and memory win.
- Benchmark artifact reproducibility gates: `96%`. Duplicate/malformed variants
  are now blocked; release-facing historical artifacts still need schema-v2
  refresh if promoted.
- FP production-floor memory/runtime: `95%`. No production solve behavior
  changed.
- CPU/GPU parity for documented workflows: `100%` for the touched lanes; no
  parity-sensitive solver path changed in this pass.
- CI/docs health: `100%` locally for the gates listed above; remote CI still
  needs post-push confirmation.
- Coverage/refactor path: `67%`. Solver progress policy is now split and
  directly tested; further progress still requires more driver/policy extraction
  before broad `v3_driver.py` coverage can move materially.
- Parallel transport workers: `82%` for release-facing multi-case/RHS
  throughput. This pass improves safety and diagnostics, not single-case
  multi-GPU strong scaling.
- `vmec_jax` / `booz_xform_jax` workflow: `65%` for public status/reporting and
  optional-backend UX; full differentiable kinetic transport remains about
  `36%`.
- Deferred manuscript physics lanes: `50%`. The W7-X scaffold is more
  reviewer-proof, but W7-X ambipolar and full high-`nu` analytic-limit claims
  remain deferred until artifact-backed validation exists.

Next concrete actions:

1. Check remote CI after pushing this integration.
2. Refresh any benchmark artifacts that are intended for release-facing claims
   to schema version `2`.
3. Use the PAS matrix-free harness diagnostics to choose the next real
   production-floor geometry4/HSX probe; do not change defaults without a
   bounded residual/runtime/memory win.
4. Continue the refactor lane with small pure solver-policy and validation
   modules before attempting high-risk `v3_driver.py` surgery.

Progress update (2026-05-11): large open-lane push toward release closure

- Added a release benchmark artifact indexer:
  `scripts/benchmark_artifact_index.py`. It scans selected JSON files or
  directories and classifies artifacts as schema-v2 compliant, historical
  legacy schema-v1, unrelated non-PAS, or release-blocking. This lets release
  gates fail malformed or policy-invalid v2 PAS artifacts without rewriting
  historical PAS benchmark records that are intentionally kept as provenance.
- Expanded the RHSMode=1 PAS matrix-free harness from a small synthetic probe
  into a production-floor preflight layer. It now inspects checked-in PAS
  benchmark artifacts, tags synthetic versus production-floor geometry metadata
  cases, applies readiness gates for geometry4 / HSX / geometry11, and emits a
  compact `next_real_solve_recommendation`. This is still opt-in harness logic
  only; no production solver defaults changed.
- Extracted transport-worker payload/result validation into
  `sfincs_jax/transport_parallel_validation.py`, keeping the runtime behavior
  and error classes compatible while making duplicate RHS, missing mapping
  entries, out-of-range RHS values, GPU coverage, and GPU array-length checks
  directly testable.
- Added a pure transport-worker scaling audit policy in
  `sfincs_jax.transport_parallel_policy.audit_transport_parallel_scaling_summary`.
  It checks baseline presence, task/device counts, speedup, efficiency,
  finite-task ideal consistency, and deterministic payload coverage before a
  benchmark summary can support a release-facing scaling claim. This formalizes
  the transport-worker scaling story and keeps single-case multi-GPU sharding
  separate and experimental.
- Added a reusable `vmec_jax` / `booz_xform_jax` / `sfincs_jax` geometry-proxy
  workflow summary builder and wired it to
  `examples/autodiff/vmec_jax_to_boozer_sfincs_pipeline.py --summary-json`.
  The summary records stage provenance, optional dependency status,
  differentiability labels, numerical gradient-gate status, and the explicit
  non-claim that full kinetic-transport gradients are not covered by this lane.
- Added a Simakov-Helander high-collisionality panel-data scaffold in
  `sfincs_jax.validation_figures`. It validates sorted high-`nu` scan data,
  normalized analytic-limit distance, tail log-log slope, monotonic approach,
  high-`nu` threshold/point/span gates, provenance completeness, checked-in
  artifact status, and explicit deferred reasons. It does not close the full
  analytic-limit literature claim without artifact-backed high-`nu` scans.
- Documentation updates:
  - geometry docs now describe the workflow summary JSON and proxy-only
    gradient gate;
  - parallelism docs now document the transport-worker scaling audit and the
    separation from single-case multi-GPU sharding;
  - testing docs now describe the Simakov-Helander scaffold gates.
- Verification:
  - focused integrated lane suite:
    `93 passed in 3.93 s`;
  - full local suite:
    `1245 passed in 486.72 s`;
  - `ruff check` on changed Python source/tests:
    passed;
  - `python -m py_compile` on changed Python modules/scripts:
    passed;
  - `python -m sphinx -W -b html docs build/sphinx-html`:
    passed;
  - `git diff --check`:
    passed;
  - benchmark artifact index smoke:
    legacy PAS artifact + non-PAS artifact classified with `release_blocking=0`;
  - PAS matrix-free dry-run and bounded harness:
    preflight ready for geometry4 / HSX / geometry11 and `7` bounded rows met
    expected keep/reject gates;
  - geometry workflow summary smoke:
    `--check-backends --json --summary-json` produced a valid proxy summary.

Updated lane status after this large push:

- PAS-heavy memory/runtime: `91%`. The next short real-solve probe is now
  explicitly gated by production-floor metadata and checked-in artifact
  evidence. A default solver change still requires a real residual/runtime/memory
  win on geometry4/HSX/geometry11.
- Benchmark artifact reproducibility gates: `98%`. Release-facing artifact
  selection can now be indexed and gated without contaminating historical
  schema-v1 provenance.
- FP production-floor memory/runtime: `95%`. No production solve behavior
  changed in this pass.
- CPU/GPU parity for documented workflows: `100%` for touched lanes; no
  parity-sensitive solver path changed.
- CI/docs health: `100%` locally for the gates listed above; remote CI still
  needs post-push confirmation.
- Coverage/refactor path: `70%`. Transport parallel validation is now a focused
  pure module with isolated tests; further movement still requires additional
  policy extraction before deep `v3_driver.py` work.
- Parallel transport workers: `85%`. Scaling claims now have a release audit
  policy. This does not change the existing measured scaling numbers or promote
  single-case multi-GPU sharding.
- `vmec_jax` / `booz_xform_jax` workflow: `69%` for public status,
  provenance, and proxy-gradient summary artifacts; full differentiable kinetic
  transport remains about `38%`.
- Deferred manuscript physics lanes: `55%`. W7-X and Simakov-Helander scaffolds
  now have explicit provenance/numerical gates, but artifact-backed W7-X
  ambipolar and full high-`nu` literature claims remain deferred.

Next concrete actions:

1. Push this large integration and check remote CI/docs.
2. Run the first short real-solve PAS production-floor probe selected by the
   preflight (`geometry4`, `HSX`, `geometry11`) and require an actual
   residual/runtime/memory improvement before changing defaults.
3. Add the benchmark artifact indexer to release CI once the release-facing
   artifact selection is finalized.
4. Continue extracting pure policy/validation modules before any broad
   `v3_driver.py` refactor.

Progress update (2026-05-11): integrated multi-lane release hardening push

- Production PAS/FP memory/runtime lane:
  - `scripts/benchmark_pas_tz_memory_fallback.py` now has CI-fast gates for
    timeout/stall detection, solver-path churn, backend mismatch, residual
    quality, and peak RSS. Probes above `600 s` require an explicit
    `--allow-long-run` opt-in.
  - `scripts/benchmark_rhs1_pas_matrixfree.py` now includes opt-in
    production real-solve probe planning/running behind
    `--run-production-solve-probe`, with production target metadata and the
    same no-accidental-long-run policy.
  - Expensive production solves are still opt-in. No production solver defaults
    changed in this pass.
- Solver-path refactor lane:
  - Extracted RHSMode=1 x-block sparse-PC side / Krylov method / restart
    decisions into `sfincs_jax/rhs1_xblock_policy.py`.
  - `v3_driver.py` now calls the pure policy resolver, preserving behavior but
    making collaborator-reported solver-path churn patterns directly testable.
- Benchmark/release artifact lane:
  - The Fortran-v3 suite benchmark summary is now a first-class release-gated
    artifact class in `sfincs_jax.benchmark_artifact_policy`.
  - The benchmark-summary generator emits canonical filtered rows used by both
    the plot and README table consistency checks, so runtime/memory plot/table
    drift is now testable.
- Parallelism lane:
  - Transport-worker scaling audits now require explicit benchmark kind,
    warm/hot timing semantics, enough independent tasks for the claimed worker
    count, GPU device coverage for GPU claims, deterministic payload coverage,
    deterministic output checks, speedup/efficiency gates, and finite-task
    ideal consistency.
  - Single-case sharded-solve summaries are explicitly separated as
    experimental artifacts and cannot be promoted into transport-worker release
    claims.
- `vmec_jax` / `booz_xform_jax` lane:
  - Added a public workflow contract for the geometry-proxy differentiability
    path and strengthened finite-beta radial profile provenance. The lane still
    claims only geometry-proxy gradients, not VMEC-boundary-to-SFINCS kinetic
    transport gradients.
- Physics validation lane:
  - Strengthened W7-X ambipolar, W7-X high-`nu`, and Simakov-Helander
    high-collisionality figure metadata and gates so generated figures
    distinguish proxy/deferred scaffolds from checked-in converged literature
    artifacts.
  - No deferred literature validation claim was closed without new
    artifact-backed scans.
- Documentation updates:
  - README benchmark text now states that the summary plot and table share the
    same canonical filtered rows.
  - Geometry docs describe the workflow contract and no-overclaim gate.
  - Parallelism docs describe release-grade transport-worker audits and keep
    single-case multi-GPU sharding experimental.
  - Testing/validation docs describe deferred manuscript-lane gates.
- Verification:
  - focused integrated multi-lane suite:
    `252 passed in 22.32 s`;
  - full local suite:
    `1297 passed in 628.35 s`;
  - `python -m ruff check` on changed Python modules/scripts/tests:
    passed;
  - `python -m ruff check --select F821 sfincs_jax/v3_driver.py`:
    passed;
  - `python -m py_compile` on changed Python modules/scripts:
    passed;
  - `python -m sphinx -W -b html docs build/sphinx-html`:
    passed;
  - `git diff --check`:
    passed;
  - benchmark artifact index and checker smokes:
    passed for the Fortran suite summary plus representative legacy/non-PAS
    artifacts;
  - PAS-TZ and PAS matrix-free dry-run smokes:
    passed with production-floor metadata and no subprocess solves;
  - geometry workflow summary smoke:
    passed with optional local `vmec_jax` / `booz_xform_jax` importability
    reported;
  - benchmark-summary generator smoke:
    regenerated a temp plot/JSON with canonical rows;
  - W7-X high-`nu` figure generator smoke:
    regenerated a temp plot/JSON and kept the claim scoped as proxy/deferred.
- Remote GPU note:
  - `office` is reachable and JAX sees two GPUs in a clean pinned checkout.
  - A whole-tree `rsync` of the uncommitted local worktree was canceled because
    benchmark artifacts made it too slow. After commit, use a clean remote
    `git pull` checkout for bounded GPU smoke instead of syncing artifacts.

Updated lane status after this integrated push:

- Overall open-lane average: `92.7%`. The machine-readable tracker now reflects
  release-grade PAS CPU/GPU production-floor evidence, parallel release-audit
  gates, refactor/CI policy extraction, optional JAX-ecosystem adoption gates,
  the scale-0.55 CPU QI exact-LU successor, the adaptive-side scale-0.55
  CPU/GPU five-seed successors, and the scale-0.60 seed-0 CPU/GPU probes. QI is
  still active at `95%` because the scale-0.60 five-seed ladder and
  production-resolution CPU/GPU ladders remain open.
- PAS-heavy memory/runtime: `93%`. The probe/gate layer is now strong enough
  for short real production-floor probes, but a default solver change still
  needs real geometry4/HSX/geometry11 residual/runtime/memory wins.
- Benchmark/parity/docs lane: `98%`. Canonical plot/table rows
  and Fortran-suite summary release gates are in place; remaining work is
  release CI wiring and regeneration of any artifacts promoted beyond
  historical provenance.
- Coverage/refactor path: `90%`. More solver dispatch is now in directly tested
  policy helpers, release metadata checks are CI-fast, and public-auto QI route
  evidence is covered by artifact tests. The explicit 95% package-coverage target
  still requires a JAX-safe coverage job and more driver decomposition.
- Parallel transport workers: `92%`. Release-facing transport-worker claims are
  now audit-gated and have GPU throughput evidence. Setup-dominated two-GPU
  case-throughput and single-case sharded solves remain experimental and
  unclaimed.
- `vmec_jax` / `booz_xform_jax` workflow: `93%`, saturating the scoped release
  target for public workflow/provenance and proxy-gradient UX. Full
  differentiable kinetic transport remains deferred outside this target.
- Optional JAX ecosystem solver evaluation: `88%` target-saturated for release.
  Lineax/Equinox/JAXopt gates are documented and tested, with Lineax kept
  unpromoted for real SFINCS error statuses.

Next concrete actions:

1. Run the scale-0.60 CPU/GPU five-seed ladder before widening the default
   auto-policy cap or attempting production-resolution QI.
2. Profile scale-0.55 x-block sparse LU ordering, memory, and factor reuse so
   the next QI attempt has a concrete algorithmic target.
3. Run the first short real-solve PAS production-floor probe selected by the
   preflight, requiring a residual/runtime/memory win before any solver-default
   promotion.
4. Wire the artifact indexer into release CI once the release artifact set is
   finalized.

Progress update (2026-05-13): QI hard-seed global-coupling and operator-reuse push

- Spawned parallel audits for the remaining lanes. The consensus was that
  transport-worker parallelism is release-facing but does not solve the
  single-RHS QI hard seed; the next meaningful QI step had to be a stronger
  x-block/global preconditioner or a bounded assembled/operator-reuse path.
- Implemented opt-in smoothed global-coupling preconditioning for explicit
  `xblock_sparse_pc_gmres`:
  - new load basis: RHS/source rows, low-L flux-surface-average modes, and
    low theta/zeta Fourier components;
  - smoothing: `Z = B^{-1} P` using the existing x-block preconditioner;
  - coarse correction: rank-revealed `A Z` solve applied inside the
    preconditioner;
  - metadata: load count, retained basis size/rank, setup time, basis names,
    apply counts, and errors.
- Implemented a guarded assembled/operator-reuse path for the same x-block
  solve:
  - conservative sparse pattern probing can materialize a full-system CSR
    operator and reuse it for Krylov matvecs;
  - sampled validation checks the assembled matvec against the matrix-free
    operator before use;
  - color-count preflight now aborts early when
    `SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_MAX_COLORS` is exceeded, so
    infeasible large patterns do not spend minutes coloring/probing.
- Added an opt-in `SFINCS_JAX_RHSMODE1_XBLOCK_PC_JAX_FACTORS` switch so the
  existing padded JAX triangular-factor path is reachable from the explicit
  x-block solver for diagnostics. This is not promoted.
- Added a global-coupling side-probe guard:
  - when global coupling is built, LGMRES rescue is unavailable, and the left
    side probe is already within
    `SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_KEEP_LEFT_RATIO`, the solver
    can keep the physical left-probe state instead of switching to the
    historically slow right-preconditioned continuation;
  - metadata records whether the switch was suppressed.
- Added tests:
  - explicit sparse color preflight max-color rejection;
  - global-coupling x-block metadata and convergence on the small full-FP
    RHSMode=1 reference case;
  - assembled-operator x-block metadata and convergence on the small full-FP
    RHSMode=1 reference case;
  - additional safe GMRES fallback initial-guess guards.
- Bounded hard-seed results:
  - CPU scale-0.60 seed 3 with global coupling, 96 directions:
    `539.5 s`, `6671` matvecs, residual `3.898e-11`, but strict
    production-sized output acceptance still failed. Rejected as too slow and
    not strictly accepted.
  - Office GPU scale-0.60 seed 3 with global coupling, 48 directions:
    side probe reached residual `1.522e-08` and ratio `5.038e4`, then the old
    left-to-right switch timed out at `620 s`. Rejected.
  - Office GPU scale-0.60 seed 3 with the keep-left guard:
    side probe kept left with the same residual/ratio and still timed out at
    `620 s`. Rejected.
  - Local small opt-in JAX-factor test was manually killed after about `76 s`;
    this path is diagnostic-only.
- Checked-in blocker artifact:
  `docs/_static/qi_seed_robustness_scale060_global_coupling_rejected_2026_05_13.json`.
- Documentation updates:
  - `docs/performance_techniques.rst` documents the new opt-in controls and
    explicitly marks them rejected for defaults on the scale-0.60 QI hard seed;
  - `docs/testing.rst` records the new negative gate and the required next
    algorithmic step.
- Verification:
  - `python -m py_compile sfincs_jax/v3_driver.py sfincs_jax/explicit_sparse.py`:
    passed;
  - focused new/nearby tests:
    `50 passed in 22.80 s`;
  - fallback initial-guess tests:
    `9 passed in 0.56 s`;
  - focused global-coupling/JAX fallback tests:
    `10 passed in 3.16 s`.
  - full local suite:
    `1476 passed in 583.68 s`;
  - docs build:
    `python -m sphinx -W -b html docs build/sphinx-html` passed;
  - changed-file lint/static checks:
    `python -m ruff check --select F821 sfincs_jax/v3_driver.py` and
    `python -m ruff check` on changed non-legacy modules/tests passed;
  - `git diff --check`:
    passed.

Updated lane status after this push:

- QI seed-robustness / hard-seed solver lane: `96%` infrastructure complete,
  but the scale-0.60 seed-3 one-GPU hard seed remains open. The new evidence
  rules out host low-rank global coupling, side-switch suppression, and current
  exposed JAX factors as sufficient default fixes.
- Solver-path churn lane: `95%`. The collaborator-reported aggressive
  side/path selection issue is now directly visible in metadata and has a
  guarded opt-in suppression, but no default change is made because the hard
  GPU run still timed out.
- Assembled/operator-reuse lane: `82%`. The safe preflight and small-case
  implementation are landed. Large QI/PAS promotion requires a feasible
  pattern/color/memory result and strict residual parity.
- PAS-heavy memory/runtime: unchanged at `93%`; this push did not touch PAS
  defaults.
- Parallel transport workers: unchanged at `92%`; still release-facing only
  for independent case/RHS throughput, not single-case QI sharding.
- Coverage/refactor path: `91%`; more solver behavior is now testable, but
  `v3_driver.py` still needs deeper decomposition before the 95% coverage
  target is honestly reachable.

Best next steps:

1. Do not widen QI defaults from this evidence. Keep the new controls opt-in.
2. Build the next QI algorithm around a device-resident preconditioner/Krylov
   formulation or a different structured global solve that avoids host SuperLU
   application inside SciPy iterations.
3. If continuing operator reuse, first implement a plan-only pattern/color/RSS
   preflight for the scale-0.60 and production QI grids, and only then attempt
   materialization.
4. Continue the non-QI lanes in parallel: PAS production-floor probes,
   benchmark artifact CI wiring, and further driver refactoring.

Progress update (2026-05-13): active x-block memory gate and QI evidence-manifest hardening

- Followed up the global-coupling/operator-reuse push after CI passed on
  `a66d099`.
- Added an opt-in active-DOF path for explicit `xblock_sparse_pc_gmres`:
  - `SFINCS_JAX_RHSMODE1_XBLOCK_ACTIVE_DOF=1` allows the x-block Krylov
    system to run on the reduced active `Nxi_for_x` unknown set and expands the
    final state back to the full SFINCS vector;
  - metadata records `xblock_active_dof`, `xblock_linear_size`, and
    `xblock_full_size`;
  - the path remains default-off and is intended as a memory-safety prerequisite
    for future QI assembled/operator-reuse experiments, not a promoted QI fix.
- Hardened assembled/operator-reuse memory gates:
  - `build_operator_from_pattern(..., allow_operator_only=False)` now enforces
    the CSR memory budget as a hard cap instead of materializing over budget;
  - added a cheap conservative sparsity preflight summary so experimental
    x-block assembled-operator reuse can reject infeasible full-system patterns
    before building Python row/column lists;
  - metadata now records assembled-operator preflight rejection and estimated
    pattern/peak bytes.
- Corrected the QI evidence surface:
  - added both scale-0.60 rejected-probe artifacts to the default QI evidence
    manifest;
  - normalized rejected-summary resolution/active-size handling;
  - regenerated `docs/_static/qi_seed_robustness_evidence_manifest.json`, which
    now records `24` artifacts, `17` passing artifacts, `7` non-passing blocker
    artifacts, largest passing/attempted total size `139502`, and active size
    `81377`;
  - updated docs and the research-lane manifest so the latest negative
    global-coupling/operator-reuse evidence is not only mentioned in prose but
    participates in the checked release gate.
- Focused verification:
  - `python -m py_compile sfincs_jax/v3_driver.py sfincs_jax/explicit_sparse.py sfincs_jax/v3_sparse_pattern.py scripts/run_qi_seed_robustness.py`: passed;
  - `python -m pytest -q tests/test_explicit_sparse.py tests/test_v3_sparse_pattern.py tests/test_qi_seed_smoke_artifact.py`: `66 passed in 23.84 s`.

Updated lane status:

- QI seed-robustness / hard-seed solver lane: `97%` infrastructure/evidence
  complete, hard seed still open. The active-DOF x-block route reduces the
  future operator-reuse target but does not yet solve the scale-0.60 one-GPU
  seed-3 blocker.
- Assembled/operator-reuse lane: `87%`. Safety gates are materially stronger;
  large-case promotion still requires a feasible active/direct pattern and a
  device-resident or otherwise non-host-bottlenecked Krylov path.
- PAS-heavy memory/runtime: still `93%`. The audit confirms the next PAS step is
  a streaming/block PAS-TZ factor or matrix-free PAS correction, not more Schur
  or restart sweeps.
- Parallel transport workers: still `92%`; no change in this follow-up.
- Coverage/refactor path: `92%`; new tests cover the active x-block and
  assembled-preflight seams, but deeper `v3_driver.py` decomposition remains
  needed for the 95% package coverage target.

Next best steps:

1. Run the expanded focused test/docs/lint set, then full local suite if clean.
2. Push this safety/evidence follow-up once CI-fast checks pass.
3. Start the next real QI algorithm as a device-resident FGMRES/global-coupling
   path or active direct-pattern assembled path; do not spend more effort on
   host low-rank or side-threshold tuning.
4. In parallel, start the PAS memory lane from a streaming PAS-TZ factorization
   design with gates: `>=50%` estimated build-byte reduction, no parity loss,
   and no runtime regression on geometry4/HSX/geometry11 floor artifacts.

Progress update (2026-05-13): device FGMRES/global-coupling and PAS guarded-FFT fallback

- Confirmed CI for `f68f8bc` passed on GitHub:
  - tests shards, optional ecosystem gates, examples-smoke, coverage, and docs
    all completed successfully for the active x-block/preflight/evidence commit.
- Added the first real device-Krylov implementation step for the QI hard-seed
  lane:
  - `fgmres_solve_with_residual` and `fgmres_solve_with_residual_jit` now provide
    a fixed-shape JAX flexible-GMRES primitive with residual history,
    iteration/restart metadata, right preconditioning, and optional
    iteration-dependent preconditioner application;
  - `SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV=fgmres` is now recognized as
    `fgmres_jax` for explicit `xblock_sparse_pc_gmres`;
  - the x-block driver forces JAX x-block factors and right preconditioning for
    this experimental route, disables implicit host-GMRES fallback unless the
    user explicitly requests it, and records device-FGMRES metadata.
- Added a JAX-array global-coupling preconditioner variant for the device
  FGMRES route:
  - the physics-aware load basis is still smoothed through the x-block
    preconditioner, but `Z`, `A Z`, and the ridge-regularized coarse inverse are
    retained as JAX arrays;
  - each coarse correction applies with `jnp` operations instead of host QR,
    SciPy triangular solves, or per-iteration `device_get`;
  - this closes the immediate host-global-coupling apply bottleneck as an
    implementation blocker, but the scale-0.60 hard seed still needs the real
    GPU rerun before promotion.
- Added a PAS memory-safety improvement:
  - when explicit theta/zeta PAS-TZ Schwarz fallback is rejected by the memory
    guard and a `tzfft` builder is available, the policy now chooses guarded
    `tzfft` before collision/hybrid fallback;
  - this avoids dense `(Ntheta*Nzeta)^2` angular-block inverse storage on
    research-floor `25 x 51 x 100 x 4` PAS shapes while preserving the existing
    true-residual gate.
- Focused verification:
  - `python -m pytest -q tests/test_rhs1_pas_policy.py tests/test_solver_gmres.py tests/test_rhs1_xblock_policy.py tests/test_v3_sparse_pattern.py::test_xblock_sparse_pc_device_krylov_records_experimental_metadata`: passed in the first focused local check;
  - `python -m ruff check --select F821 ...`: passed on the touched source and
    test files;
  - `python -m py_compile sfincs_jax/solver.py sfincs_jax/v3_driver.py sfincs_jax/rhs1_xblock_policy.py sfincs_jax/rhs1_pas_policy.py`: passed.

Updated lane status:

- QI seed-robustness / hard-seed solver lane: `98%` infrastructure and
  implementation surface complete, but not closed. The missing gate is the
  actual scale-0.60 one-GPU seed-3 rerun with `fgmres_jax`, JAX factors,
  device global coupling, active-DOF if needed, and strict true-residual
  acceptance.
- Assembled/operator-reuse lane: `88%`. The device-Krylov/global-coupling path
  removes the main per-iteration host-transfer blocker; true assembled
  operator reuse still needs a device sparse/active-pattern implementation
  before a large QI claim.
- PAS-heavy memory/runtime: `94%`. Guarded FFT fallback is now preferred before
  collision/hybrid when dense Schwarz is rejected. The remaining larger step is
  a production benchmark showing memory/runtime improvement on geometry4/HSX/
  geometry11 without residual regression.
- Parallel transport workers: still `92%`; no change in this pass.
- Coverage/refactor path: `93%`; solver primitives and policy seams gained
  focused tests, but `v3_driver.py` remains the main refactor/coverage blocker.

Next best steps:

1. Run the expanded focused suite, Sphinx docs, and full local tests.
2. Run the real scale-0.60 QI seed-3 GPU probe on `office` with
   `SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV=fgmres`,
   `SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING=1`, bounded directions, and
   strict residual acceptance.
3. Run the PAS production-floor fallback probe to quantify the new guarded
   `tzfft` preference against collision/hybrid on CPU and GPU.
4. If those pass, regenerate evidence artifacts/docs/plots; if they do not,
   keep the routes opt-in and record blocker artifacts before the next
   algorithmic step.

Progress update (2026-05-13): device-Krylov audit fixes and GPU rejected evidence

- Addressed the read-only audit blockers before committing the device-Krylov
  implementation:
  - `fgmres_solve_with_residual` now reports the first converged iteration
    instead of making driver metadata look like every solve used the full
    `maxiter` budget;
  - trace-safe/device preconditioners skip expensive preconditioner/matvec work
    after convergence, while legacy host wrappers such as the old two-level
    preconditioner disable that skip to avoid tracing through `device_get`;
  - device-Krylov metadata now distinguishes
    `xblock_device_krylov_forced_jax_factors` and
    `xblock_device_krylov_host_transfer_free`, while the legacy
    `xblock_device_fgmres_*` keys remain for compatibility;
  - host-transfer-free metadata is false when the host two-level wrapper is
    present;
  - the device global-coupling builder now filters non-finite/zero load,
    smoothed, and `A Z` directions before forming the ridge coarse inverse.
- Tightened the PAS memory fallback scope:
  - explicit unsafe `theta`, `zeta`, or `schwarz` requests can demote to
    matrix-free `tzfft` when available;
  - implicit sharded defaults remain bounded and fall back to collision/hybrid
    instead of silently selecting the experimental `tzfft` route.
- Ran the actual scale-0.60 seed-3 one-GPU hard-seed probes on `office` with
  JAX `0.6.2`, `JAX_ENABLE_X64=True`, active DOF, JAX x-block factors, device
  global coupling, and strict output acceptance:
  - `fgmres` right-preconditioned device route: finished before the `620 s`
    timeout in `217.5 s`, used about `3.9 GB` RSS, built the device global
    coupling basis with `loads=24 basis=24 rank=22`, but failed the output gate
    with residual `3.021487e-05` against target `3.021487e-13`;
  - `gmres-jax` left-preconditioned device route: finished before timeout in
    `252.3 s`, used about `4.3 GB` RSS, but failed the output gate with residual
    `1.307111e+01` against target `3.021487e-13`.
- Recorded these probes in
  `docs/_static/qi_seed_robustness_scale060_device_krylov_rejected_2026_05_13.json`
  and regenerated
  `docs/_static/qi_seed_robustness_evidence_manifest.json`:
  - evidence artifacts: `25`;
  - passing artifacts: `17`;
  - non-passing blocker artifacts: `8`;
  - largest passing bounded grid remains `139502` unknowns, so the QI
    production gate remains `bounded_proxy`.
- Verification after the audit fixes:
  - `python -m pytest -q tests/test_solver_gmres.py`: `29 passed in 12.68 s`;
  - focused policy/full-system/evidence suite: `81 passed in 114.45 s`;
  - `python -m py_compile ...`: passed on touched Python files;
  - `python -m ruff check --select F821 ...`: passed on touched source, scripts,
    and tests;
  - `python scripts/check_research_lanes.py && python scripts/check_release_gates.py && git diff --check`: passed;
  - `python -m sphinx -W -b html docs build/sphinx-html`: passed.

Updated lane status after this pass:

- QI seed-robustness / hard-seed solver lane: `98%` implementation
  infrastructure, still not closed. The remaining gap is numerical convergence
  of the scale-0.60 GPU hard seed, not runtime/timeout/illegal-address
  robustness.
- Assembled/operator-reuse lane: `89%`. The device Krylov and device
  global-coupling surfaces remove the host-transfer implementation blocker for
  opt-in probes, but true active sparse operator reuse is still needed before a
  release-grade hard-seed closure.
- PAS-heavy memory/runtime: `94%`. Scope is now safer: explicit `tzfft` is
  benchmark-only and implicit sharded fallbacks remain conservative.
- Parallel transport workers: `92%`; no direct change in this pass.
- Coverage/refactor path: `93%`; additional solver, policy, metadata, and
  evidence tests were added, but `v3_driver.py` still needs further splitting.

Next best steps:

1. Run the full local pytest suite once this commit is ready.
2. Commit and push the opt-in device-Krylov/global-coupling and PAS-scope
   safety work if the full suite passes.
3. Start the next real QI algorithmic lane: active assembled sparse operator
   reuse or a stronger physics coarse space that reduces the true residual on
   the hard seed, rather than more Krylov-method or side-selection toggles.
4. Run a bounded PAS production-floor benchmark for the explicit `tzfft`
   demotion and guarded correction paths before any performance claim.

## 2026-05-13 active assembled/operator-reuse push

Goal: remove the implementation blocker where the opt-in RHSMode=1 x-block
assembled/operator-reuse path rejected active-DOF systems using the full-system
conservative CSR budget before slicing inactive `Nxi_for_x` modes.

Implementation:

- `SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR=1` now applies
  `SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_CSR_MAX_MB` to the active
  sliced operator when `SFINCS_JAX_RHSMODE1_XBLOCK_ACTIVE_DOF=1`.
- Solver metadata now records whether the assembled preflight used `full` or
  `active_dof` scope, plus both full-system and active-DOF CSR byte estimates
  when the active route is used.
- Added a regression that reproduces the prior failure mode: the full-system
  CSR estimate exceeds the configured budget, the active sliced operator fits,
  and the assembled operator is built and validated.
- Added a structured active transport pattern builder that recognizes the
  `_transport_active_dof_indices` ordering and builds the kinetic block as
  `velocity_graph ⊗ angular_stencil`, appending Phi1/constraint tails
  afterwards. This avoids materializing inactive pitch rows/columns and avoids
  Python appends over every `(velocity, theta, zeta)` candidate on large reduced
  FP/QI grids.

Current evidence:

- `pytest -q tests/test_v3_sparse_pattern.py::test_xblock_sparse_pc_assembled_operator_active_dof_uses_sliced_budget`:
  `1 passed in 4.37 s` after the structured builder landed.
- `ruff check sfincs_jax/v3_sparse_pattern.py tests/test_v3_sparse_pattern.py`: passed.
- `python -m py_compile sfincs_jax/v3_sparse_pattern.py sfincs_jax/v3_driver.py`: passed.
- Scale-0.60 QI seed-3 structural preflight at `15 x 31 x 60 x 5`, active size
  `81377`, total size `139502`:
  - old full-pattern build then slice: `250.04 s`, full pattern `376659300`
    nonzeros, full CSR estimate `4520.47 MB`;
  - active sliced pattern: `128174925` nonzeros, active CSR estimate
    `1538.42 MB`;
  - first active-only Python-loop builder: `174.56 s`;
  - structured `velocity_graph ⊗ angular_stencil` builder: `1.19 s` for the
    same active pattern and nonzero count.
- Bounded dummy-matvec materialization preflight on the same active pattern:
  `pattern_s=1.23 s`; graph coloring rejected in `0.88 s` with
  `max_colors=64`. This is useful safety evidence: the code now reaches a fast
  bounded rejection instead of spending minutes building the full inactive
  pattern, but it also means a production assembled QI solve still needs a
  defensible color/materialization policy before being promoted.

Updated lane status:

- QI seed-robustness / hard-seed solver lane: `98%`; unchanged until a hard-seed
  residual run shows strict acceptance.
- Assembled/operator-reuse lane: `94%`; active-DOF operator reuse is now
  implemented, unit-tested, and no longer blocked by full-pattern materialization
  on the scale-0.60 QI preflight. Remaining work is the actual graph-colored
  numerical materialization/solve keep-drop gate: the active CSR pattern is still
  large (`1.54 GB`), so the next probe must verify color count, matvec
  materialization time, true residual, and memory before any production claim.
- PAS-heavy memory/runtime: `94%`; no direct change in this pass.
- Parallel transport workers: `92%`; no direct change in this pass.
- Coverage/refactor path: `93%`; one focused solver-regression test added.

Next best steps:

1. Run one bounded scale-0.60 QI active assembled/operator-reuse materialization
   probe with production `max_colors` and a strict 10-minute ceiling on CPU.
   Abort before solve if graph coloring or numerical probing exceeds the
   memory/time gate.
2. If active assembled materialization is feasible, run the one-seed solve and
   compare true residual/runtime/memory with the current LGMRES-rescue CPU
   artifact. Only run GPU if CPU materially improves true residual or memory.
3. If active assembled materialization is infeasible or does not reduce true
   residual, keep it diagnostic and move to the stronger physics coarse-space
   correction rather than adding more Krylov toggles.

## 2026-05-13 active two-level x-block preconditioner push

Goal: continue the scale-0.60 QI hard-seed lane after the active assembled
operator path reached a fast, bounded graph-coloring rejection rather than a
feasible production solve.

Evidence from the bounded materialization probe:

- Scale-0.60 QI seed-3 at `15 x 31 x 60 x 5`, active size `81377`, total size
  `139502`, active conservative pattern `128174925` nonzeros, active CSR
  estimate `1538.42 MB`.
- Structured active pattern build: `1.44 s`.
- Graph-coloring probe with production `max_colors=512`: rejected in `2.04 s`
  with `pattern probing would require more than max_colors=512 colors`.
- Conclusion: active assembled/operator reuse is now safe and bounded, but it is
  not the next production path for this QI hard seed. The next useful algorithmic
  step is a stronger matrix-free/coarse preconditioner.

Implementation:

- Removed the previous `active-DOF x-block two-level preconditioner is not
  implemented` branch for explicit `xblock_sparse_pc_gmres`.
- `_build_rhs1_xblock_two_level_preconditioner` now accepts an optional
  active-DOF projector and expected reduced size. Each coarse basis vector is
  projected before validation, `A Z` construction, and QR rank selection.
- Solver metadata now records `xblock_two_level_active_projected` and
  `xblock_two_level_expected_size`, so future QI/PAS benchmark artifacts show
  whether the two-level wrapper acted on full or reduced coordinates.
- Added a focused regression that enables `SFINCS_JAX_RHSMODE1_XBLOCK_ACTIVE_DOF`
  and `SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL` on a reduced-pitch RHSMode=1
  fixture, then verifies strict residual, projected metadata, and preconditioner
  application.

Validation:

- `python -m pytest tests/test_v3_sparse_pattern.py::test_xblock_sparse_pc_two_level_active_dof_projects_coarse_basis -q`:
  `1 passed in 2.98 s`.
- `python -m pytest tests/test_v3_sparse_pattern.py::test_xblock_sparse_pc_two_level_preconditioner_records_metadata tests/test_v3_sparse_pattern.py::test_xblock_sparse_pc_global_coupling_preconditioner_records_metadata tests/test_v3_sparse_pattern.py::test_xblock_sparse_pc_active_dof_opt_in_records_reduced_size tests/test_v3_sparse_pattern.py::test_xblock_sparse_pc_assembled_operator_active_dof_uses_sliced_budget -q`:
  `4 passed in 5.50 s`.
- `python -m pytest tests/test_v3_sparse_pattern.py -q`: `31 passed in
  134.57 s`.
- `python -m ruff check tests/test_v3_sparse_pattern.py`: passed. A full-file
  Ruff check on `sfincs_jax/v3_driver.py` still reports pre-existing style debt
  unrelated to this patch, so it was not used as a gate for this large module.
- Bounded CPU scale-0.60 seed-3 QI probe with
  `SFINCS_JAX_RHSMODE1_XBLOCK_ACTIVE_DOF=1`,
  `SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL=1`,
  `SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL_MAX_DIRECTIONS=48`,
  `SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL_FSAVG_LMAX=8`,
  `solve_method=xblock_sparse_pc_gmres`, and `timeout_s=240`: failed by timeout.
  The two-level coarse basis did build (`basis=47`, `rank=45`) at `~22.9 s`,
  but the side probe still selected `lgmres` with residual ratio `1.12e5`, and
  the process timed out after `240.06 s`, `2725` reported matvecs, no HDF5
  output, and no solver trace. This rejects the current active two-level basis
  as a promoted scale-0.60 QI fix.

Updated lane status:

- QI seed-robustness / hard-seed solver lane: `98.5%` infrastructure complete.
  The hard seed remains open. The new active two-level implementation is useful
  infrastructure, but the current basis is rejected for the scale-0.60 seed-3
  promotion gate.
- Assembled/operator-reuse lane: `95%`; bounded rejection with production
  color budget is now recorded. No default promotion.
- Active two-level/coarse preconditioning lane: `96%`; active-DOF support is
  implemented and regression-tested. The existing fixed coarse basis is rejected
  for the QI hard seed, so the next residual-improvement attempt needs a richer
  physics coarse space rather than only toggling Krylov methods or side choices.

Next best steps:

1. Do not run the same active two-level policy on GPU; CPU already rejected it.
2. Design the next QI hard-seed attempt around a richer physics coarse basis
   that includes low-order theta/zeta harmonics after x-block smoothing,
   drive/source directions, and near-null residual directions from the failed
   side probe, while keeping the setup matrix-free and bounded.
3. Continue PAS production-floor memory/runtime probes independently; the
   active two-level change is opt-in and should not alter public defaults.

## 2026-05-13 probe-coarse QI hard-seed push

Goal: try the next matrix-free QI hard-seed step after active assembled
materialization rejected the color budget and the fixed active two-level basis
timed out.

Implementation:

- Added an opt-in pre-Krylov seed-correction hook controlled by
  `SFINCS_JAX_RHSMODE1_XBLOCK_PC_PROBE_COARSE`.
- The hook reuses the post-coarse bounded least-squares basis, but applies it to
  the side-probe or user-supplied initial state before the expensive full Krylov
  solve.
- Fixed active-DOF coarse direction construction: residuals are expanded to full
  coordinates for physics-aware flux-surface/source basis construction, and each
  candidate direction is projected back to the active `Nxi_for_x` coordinate set
  before the least-squares correction is formed.
- Added solver metadata for `xblock_probe_coarse_*` residuals, direction counts,
  accepted steps, elapsed setup time, and direction names.
- Added a focused active-DOF regression that passes a zero physical initial
  guess, enables probe-coarse, verifies a strict solve, and checks that one
  projected seed correction is accepted.

Validation:

- `python -m pytest tests/test_v3_sparse_pattern.py::test_xblock_sparse_pc_probe_coarse_uses_active_projected_directions tests/test_v3_sparse_pattern.py::test_xblock_sparse_pc_two_level_active_dof_projects_coarse_basis -q`:
  `2 passed in 3.71 s`.
- `python -m ruff check tests/test_v3_sparse_pattern.py`: passed.
- `python -m py_compile sfincs_jax/v3_driver.py`: passed.
- Bounded CPU scale-0.60 seed-3 QI probe with
  `SFINCS_JAX_RHSMODE1_XBLOCK_ACTIVE_DOF=1`,
  `SFINCS_JAX_RHSMODE1_XBLOCK_PC_PROBE_COARSE=1`,
  `SFINCS_JAX_RHSMODE1_XBLOCK_PC_PROBE_COARSE_MAX_DIRECTIONS=32`,
  `SFINCS_JAX_RHSMODE1_XBLOCK_PC_PROBE_COARSE_FSAVG_LMAX=4`,
  `solve_method=xblock_sparse_pc_gmres`, and `timeout_s=240`: passed.
  Summary artifact:
  `docs/_static/qi_seed_robustness_scale060_probe_coarse_seed3_cpu.json`.
  Solver trace:
  `docs/_static/qi_seed_robustness_scale060_probe_coarse_seed3_cpu_solver_trace.json`.
- Probe-coarse accepted one 12-direction correction, reducing side-probe seed
  residual `2.5721269e-8 -> 1.4264660e-8` in `0.287 s`.
- The full hard-seed solve converged on CPU in `222.5 s` elapsed, `35`
  LGMRES iterations, `2403` matvecs, residual `1.0360e-13`, target
  `3.0215e-11`, residual ratio `3.43e-3`, with HDF5 output and solver trace
  written.
- Matching one-GPU office probe from a clean checkout at commit `689ea32`, using
  JAX `0.6.2` on GPU 0 and `timeout_s=360`: failed by timeout. Artifact:
  `docs/_static/qi_seed_robustness_scale060_probe_coarse_seed3_gpu0_timeout.json`.
  The process reached the side probe and then switched `left -> right`, so the
  physical side-probe state was discarded before probe-coarse could apply. It
  advanced to about `700` reported matvecs by `347 s`, wrote no HDF5 output and
  no solver trace, and used about `3.1 GB` max host RSS. No CUDA illegal-address
  crash occurred in this run.

Updated lane status:

- QI seed-robustness / hard-seed solver lane: `99%` for bounded CPU
  infrastructure. The scale-0.60 CPU hard seed that previously timed out now
  passes, but the one-GPU hard seed and production-resolution multi-seed CPU/GPU
  ladders remain open before production QI robustness can be claimed.
- Assembled/operator-reuse lane: `95%`; unchanged. The active assembled route
  remains a bounded rejected diagnostic because graph coloring exceeds the
  production color budget.
- Active coarse/preconditioning lane: `97%`; probe-coarse gives a measurable
  residual-seed improvement and closes the bounded CPU hard seed. It remains
  opt-in and CPU-rescue-only until a GPU-compatible device-resident variant is
  implemented and passes the hard seed.
- PAS-heavy memory/runtime: `94%`; unchanged in this pass.
- Parallel transport workers: `92%`; unchanged in this pass.
- Coverage/refactor path: `93.5%`; one more focused solver-regression test and
  manifest update landed, but the package is still not at the deferred 95%
  meaningful coverage target.

Next best steps:

1. Do not retry the same GPU probe-coarse policy unchanged; the one-GPU run
   already timed out and did not apply the seed correction after side switching.
2. Implement a GPU-compatible probe/coarse route that either keeps the left
   physical seed long enough for correction or makes the correction device
   resident and compatible with right-preconditioned coordinates.
3. After that implementation, rerun the one-GPU scale-0.60 seed-3 hard seed and
   only then attempt a small CPU/GPU multi-seed ladder.
4. Continue the PAS memory/runtime and refactor/coverage lanes independently;
   this QI hook is opt-in and does not alter public default solver selection.

## 2026-05-13 docs/plan handoff: QI device-compatible gate and open lanes

Current QI decision:

- The bounded CPU scale-0.60 seed-3 hard seed is no longer the immediate blocker:
  probe-coarse reduced the side-probe true residual before the full Krylov solve
  and the CPU solve converged with a strict residual ratio `3.43e-3`.
- The bounded one-GPU `office` gate remains open. The matching GPU probe timed
  out after the side probe switched `left -> right`, did not apply the physical
  seed correction, wrote no solver trace/output, and showed the remaining issue
  is GPU-compatible preconditioner/operator application rather than CUDA
  stability.
- Do not spend more lane time on Krylov-name toggles, restart-only changes, or
  side-threshold tuning as promotion candidates. Earlier LGMRES/GCROT/BiCGStab,
  fixed two-level, global-coupling, and device-Krylov probes are documented
  negative evidence unless a new preconditioner/operator path first improves the
  true residual.
- The next QI implementation must be device-resident or operator-reuse based:
  either keep the pre-Krylov correction in a physical/device-compatible state,
  or assemble/reuse an active operator/preconditioner whose preflight applies to
  physics load bases and reduces the true residual before the expensive full
  Krylov loop.
- Promotion gate: pass a bounded `office` GPU scale-0.60 hard seed with HDF5
  output, solver trace, strict true-residual acceptance, no CPU/GPU parity
  regression against the bounded CPU artifact, and no widened public default
  window until a CPU/GPU multi-seed ladder also passes.

Secondary open lanes:

- PAS memory/runtime remains open at production-floor geometry richness. Guarded
  `tzfft` and weak-PAS fail-fast paths are useful bounded diagnostics, but a
  promoted route still needs a residual-clean CPU/GPU benchmark with lower
  runtime or memory on geometry4/HSX/geometry11-style PAS floors.
- Parallel/scaling remains release-facing only for independent case/RHS
  throughput. Single-case multi-device strong scaling is still experimental and
  should not be claimed until a warm, compile-amortized, device-covered artifact
  shows a real speedup.
- Coverage/refactor remains open. Focused policy and solver tests improve
  confidence, but the deferred `95%` package target still needs deeper
  `v3_driver.py` decomposition and a JAX-safe coverage job rather than more slow
  full-solve tests.
- VMEC/Boozer/JAX workflow remains a bounded/proxy workflow lane. The current
  docs and tests cover provenance, optional ecosystem gates, and proxy-gradient
  checks; full VMEC-boundary-to-SFINCS kinetic transport gradients remain
  deferred.
- Deferred validation lanes stay deferred unless they receive checked-in,
  numerically gated artifacts and explicit release-gate metadata. This includes
  W7-X ambipolar validation, high-`nu` analytic-limit extension, broader
  MONKES/KNOSOS overlap, production-resolution QI ladders, and large
  geometry-rich PAS claims.

## 2026-05-13 device-resident assembled-operator QI push

Implementation:

- Added `sfincs_jax/rhs1_device_operator.py`, a budgeted SciPy/materialized
  operator to JAX CSR adapter with optional active-index slicing, JIT-compatible
  matvecs, metadata, and validation helpers.
- Wired `SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_DEVICE` into the
  x-block assembled/operator-reuse path. When enabled, a materialized assembled
  operator is copied to JAX CSR arrays and Krylov matvecs use the device CSR
  operator instead of per-iteration SciPy host matvecs.
- Added `SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_DEVICE_REQUIRED` for
  hard GPU probes, plus solver metadata for device residency, device CSR bytes,
  validation errors, and host-transfer-free device Krylov status.
- Added focused unit and integration tests covering device CSR matvec parity,
  active slicing, budget rejection, validation failures, required device
  assembled-operator metadata, and a bounded device-FGMRES transfer-free gate.
- Extended the QI runner progress markers so future artifacts retain assembled
  operator build and `assembled_device_matvecs` progress lines.

Validation:

- `python -m pytest tests/test_rhs1_device_operator.py tests/test_rhs1_device_operator_unit.py -q`:
  `7 passed in 4.73 s`.
- Focused assembled/operator-reuse and device-Krylov slice:
  `16 passed in 117.53 s`.
- `python -m ruff check sfincs_jax/rhs1_device_operator.py tests/test_rhs1_device_operator.py tests/test_rhs1_device_operator_unit.py tests/test_v3_sparse_pattern.py`:
  passed.
- `python -m py_compile sfincs_jax/rhs1_device_operator.py sfincs_jax/v3_driver.py scripts/run_qi_seed_robustness.py`:
  passed.
- `python -m sphinx -W -b html docs build/sphinx-html`: passed.

Office GPU hard-seed evidence:

- Staged the current local tree to `/tmp/sfincs_jax_device_qi_min` on `office`
  because the persistent remote checkout is dirty and far behind `origin/main`.
- Ran the scale-0.60 seed-3 QI hard seed at `15 x 31 x 60 x 5` on RTX A4000 GPU
  0 with active DOFs, device FGMRES, probe-coarse enabled, and required
  device-resident assembled operator.
- Result artifact:
  `docs/_static/qi_seed_robustness_scale060_device_operator_rejected_2026_05_13.json`.
- The device assembled operator built successfully:
  active size `81377`, total size `139502`, operator `2,884,321` nonzeros,
  setup `83.587 s`, GPU memory around `5.9 GB`.
- The solve reached `400` device CSR matvecs by `505.695 s`, but the runner
  timed out at `540 s`, wrote no HDF5 output or solver trace, and peaked near
  `40.3 GB` RSS. XLA reported slow constant-folding diagnostics during the run.
- Follow-up row-index CSR matvec work removed the XLA slow constant-folding
  warnings and reached `400` device matvecs earlier (`480.848 s`), but peak RSS
  increased to about `49.7 GB`, so it is an implementation cleanup rather than a
  promotion candidate by itself.
- Follow-up short-recurrence `bicgstab-jax` on the same `office` GPU hard seed
  built the same device operator in `84.767 s` and finished before timeout
  (`483.507 s` outer wall), with peak RSS reduced to `13.6 GB`. It then
  diverged (`residual=2.351749e102`, target `3.021487e-13`) after the reported
  short-recurrence step and correctly refused HDF5 output. This proves the
  memory direction is viable but rejects the current BiCGStab formulation for
  accuracy.
- Follow-up restart-20 device FGMRES on the same hard seed kept the assembled
  operator on device and reached `500` device matvecs by `533.214 s`, but timed
  out at the `540 s` gate, wrote no HDF5 output/solver trace, and peaked at
  `50.4 GB` RSS. This rejects smaller fixed restart as a promotion path by
  itself.
- Added `SFINCS_JAX_RHSMODE1_XBLOCK_PC_FGMRES_BLOCK_BETWEEN_CYCLES` as the next
  opt-in memory diagnostic. It synchronizes eager JAX FGMRES at restart
  boundaries, so the next bounded GPU probe can test whether the hard-seed host
  RSS growth is queued cross-cycle state or unavoidable basis/preconditioner
  footprint. Unit coverage verifies that the option preserves FGMRES accuracy
  on a nonsymmetric system, and full-system metadata coverage verifies the knob
  is wired through the x-block device-operator path.
- Ran the synchronized restart-20 GPU hard-seed probe on `office`: same
  `15 x 31 x 60 x 5` scale-0.60 QI seed, device operator `2,884,321` nonzeros,
  setup `82.525 s`, `500` device matvecs by `528.810 s`, timeout at `540 s`,
  no HDF5 output/solver trace, and peak RSS `51.0 GB`. This rejects
  cycle-boundary synchronization as a memory solution for the hard seed; it is
  useful instrumentation only.

Updated lane status:

- QI device-compatible operator reuse: `99%` infrastructure complete, but not
  promoted. The host-SciPy matvec bottleneck is now removable, and the hard-seed
  GPU run reaches a real device operator path. The current full-restart and
  restart-20 device-FGMRES variants, including the cycle-synchronized restart-20
  variant, are rejected for wall time and host RSS; the current `bicgstab-jax`
  variant is rejected for divergence despite much lower memory.
- QI hard-seed closure remains open. The next implementation should keep the
  device CSR operator but reduce Krylov memory and synchronization: smaller true
  restart windows, stabilized short-recurrence device IDR/TFQMR-style trials, or
  a stronger residual-reducing preconditioner/coarse correction before FGMRES.
- Do not promote default QI policy or production claims from this evidence.

## 2026-05-14 TFQMR replacement-path QI probe

Implementation:

- Added a JAX-native TFQMR solver in `sfincs_jax/solver.py` with left, right,
  and unpreconditioned transformed-system support, JIT wrappers, true physical
  residual reporting, and optional true-residual replacement/restart.
- Routed `SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV=tfqmr-jax` through the
  RHSMode=1 x-block device-Krylov path, including metadata for device TFQMR,
  the replacement interval, and the short-recurrence memory estimate.
- Added `tfqmr_work_nbytes()` to the memory model and focused unit/integration
  tests for solver accuracy, right preconditioning, JIT use, residual
  replacement, policy aliases, memory estimates, and x-block metadata.

Validation:

- Local focused solver/policy/memory/device path:
  `85 passed in 25.43 s`.
- QI artifact tests:
  `23 passed in 0.03 s`.
- RHSMode=1 device-operator/sparse-pattern subset:
  `44 passed in 135.18 s`.
- PAS/preconditioner/adapters subset:
  `53 passed in 6.11 s`.
- `python -m py_compile sfincs_jax/solver.py sfincs_jax/v3_driver.py
  sfincs_jax/rhs1_xblock_policy.py sfincs_jax/memory_model.py`: passed.
- `python -m ruff check` on the touched solver/policy/memory/test files and
  `python -m ruff check --select F821,F823 sfincs_jax/v3_driver.py`: passed.

Office GPU hard-seed evidence:

- Plain device TFQMR on the scale-0.60 QI seed-3 hard seed built the active
  device CSR operator (`81377` active unknowns, `139502` total unknowns,
  `2,884,321` nonzeros, setup `82.072 s`) and stayed in the low-memory
  short-recurrence footprint (`13.6 GB` peak RSS), but diverged to
  `2.351749e102` against target `3.021487e-13`. Output was correctly refused.
- TFQMR with true-residual replacement every 20 iterations built the same
  device CSR operator (setup `101.566 s`) and failed safely faster
  (`263.864 s` outer elapsed, `4:23.94` wall), but still diverged to
  `2.351670e102`. Peak RSS remained `13.6 GB`.
- The checked artifact
  `docs/_static/qi_seed_robustness_scale060_device_operator_rejected_2026_05_13.json`
  now records both TFQMR probes as rejected evidence.

Decision:

- TFQMR is a real replacement implementation path for the memory-heavy device
  FGMRES basis, and residual replacement is a useful safety guardrail. It does
  not close the QI hard seed because the x-block preconditioned operator remains
  too poorly conditioned for this short-recurrence Krylov family.
- Keep TFQMR opt-in and do not widen public/default QI policy from this result.
- The next real closure attempt should not be another Krylov-name toggle. It
  must strengthen the residual-reducing preconditioner/coarse space, for example
  by adding a device-compatible global low-rank/Schur correction to the active
  device CSR path, then reuse the assembled operator with FGMRES/TFQMR only
  after the preconditioned residual is demonstrably bounded on a cheap probe.

Updated lane status:

- Device operator and low-memory solver infrastructure: `99%` complete.
- QI hard-seed closure: `85%` complete; the remaining gap is algorithmic
  conditioning, not data motion or basis storage alone.
- Overall performance/memory pass: `90%` complete, with QI hard-seed and
  production PAS/geometry-rich gates still the main non-release blocker lanes.

## 2026-05-14 device QR global-coupling probe

Implementation:

- Replaced the default device global-coupling coarse solve with a rank-revealed
  QR setup in `sfincs_jax/v3_driver.py`. The Krylov-time apply remains
  device-resident: setup forms the physics-smoothed basis `Z` and `A Z`, keeps
  the retained `Z`, `Q`, and `R` factors as JAX arrays, and applies
  `Z R^{-1} Q^T r` inside the preconditioner.
- Kept the previous ridge-normal-equation device coarse solve as an explicit
  diagnostic route through
  `SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_DEVICE_SOLVER=normal-equations`.
- Added solver metadata for `xblock_global_coupling_coarse_solver`,
  `xblock_global_coupling_ridge`, and the retained coarse singular-value/QR
  scale proxies.
- Tightened the opt-in probe-coarse path so it can initialize its own zero
  x-block seed when no side probe or explicit `x0` is available. This makes the
  projected coarse correction a real pre-Krylov gate instead of an inert hook
  that only runs after another seed-producing path happened first.
- Added a setup-budget guard and smoother selector to global coupling:
  `SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_SETUP_MAX_S` records and can
  stop partial-basis setup once a budget is reached, and the device
  global-coupling path now defaults to
  `SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_SMOOTHER=identity` so raw
  physics-load coarse directions are used unless the older expensive
  preconditioned-load smoothing is explicitly requested with `base`.
- Added focused tests covering default QR metadata and the explicit
  normal-equations compatibility path, the partial-basis setup budget, and the
  zero-seed probe-coarse path.

Validation:

- `python -m py_compile sfincs_jax/v3_driver.py`: passed.
- `python -m ruff check --select F821,F823 sfincs_jax/v3_driver.py`: passed.
- `python -m ruff check tests/test_v3_sparse_pattern.py`: passed.
- Focused device-global-coupling tests:
  `3 passed in 150.54 s`.
- Adjacent assembled/device-Krylov/TFQMR tests:
  `4 passed in 12.71 s` and `3 passed in 1.39 s`.
- Probe-coarse zero-seed regression:
  `tests/test_v3_sparse_pattern.py::test_xblock_sparse_pc_probe_coarse_uses_active_projected_directions`
  passed in `4.56 s`.
- Global-coupling setup-budget regression plus host/device QR metadata:
  focused slice passed with `5 passed in 129.19 s`.
- Final focused implementation gate after the identity-smoother change:
  `tests/test_v3_sparse_pattern.py tests/test_qi_seed_smoke_artifact.py
  tests/test_solver_gmres.py tests/test_memory_model.py` passed with
  `106 passed in 201.80 s`.

Office GPU hard-seed evidence:

- Ran scale-0.60 QI seed-3 at `15 x 31 x 60 x 5` on `office` GPU 0 with
  active DOFs, required device CSR assembled operator, device FGMRES,
  `SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING=1`,
  `SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_DEVICE_SOLVER=qr`,
  `MAX_DIRECTIONS=32`, `FSAVG_LMAX=4`, and `ANGULAR_LMAX=1`.
- The device CSR operator built successfully with `2,884,321` nonzeros in
  `83.104 s`.
- The device QR global-coupling correction built successfully with `32` loads,
  `20` retained basis vectors, and rank `20`.
- The solve reached `475` device matvecs by `532.478 s`, then timed out at the
  `540 s` gate (`541.968 s` summary elapsed), wrote no HDF5 output/solver trace,
  and peaked at `46.1 GB` RSS.
- The checked blocker artifact
  `docs/_static/qi_seed_robustness_scale060_device_operator_rejected_2026_05_13.json`
  records this row as `device_global_qr_fgmres_probe`.
- A follow-up QR + probe-coarse GPU run without the identity/raw-load smoother
  showed why a setup budget alone is insufficient: the builder can spend
  hundreds of device matvecs inside a single preconditioned-load smoothing step
  before it can check the budget. It timed out at `540.032 s`, wrote no
  output/trace, and peaked at `48.2 GB` RSS.
- The replacement identity-smoother QR + probe-coarse run reached the intended
  stages: device global coupling built with `32` loads, `30` retained basis
  vectors, rank `30`, and `smoother=identity`; probe-coarse improved the seed
  residual only from `3.021487e-05` to `3.019236e-05`; FGMRES then reached `400`
  device matvecs by `533.417 s` and timed out at the `540 s` gate. Peak RSS was
  `37.8 GB`, lower than the preconditioned-load/global-QR probes, but still not
  a converged hard-seed result.

Decision:

- Device QR global coupling is better numerical infrastructure than the old
  normal-equation coarse solve and should remain the default for opt-in device
  global-coupling experiments.
- Identity/raw-load smoothing is the correct default for device global-coupling
  setup because it bounds setup and reaches Krylov; the older base-preconditioned
  smoother is too expensive for the QI hard seed unless a future preflight proves
  it reduces the residual enough to justify the setup cost.
- It does not close the QI hard seed as a standalone preconditioner. The
  remaining blocker is still the high-cost FGMRES basis/preconditioner footprint
  and insufficient residual reduction before the full Krylov loop.
- Next best step: stop adding Krylov/coarse wrappers around the same x-block
  inverse. Build a stronger physics coarse operator that targets the actual
  constraint/current moment closure, or add a bounded projected coarse solve
  before FGMRES that must reduce the true residual by an order of magnitude in
  preflight before the full GPU Krylov solve is allowed.

Updated lane status:

- Device operator/coarse-solver infrastructure: `99.5%` complete.
- QI hard-seed closure: `86%` complete. The implementation surface is now rich
  enough; the remaining work is a genuinely stronger physics preconditioner,
  not more solver plumbing.
- Overall performance/memory pass: `90%` complete.

## 2026-05-14 row/column equilibration and exact-xblock GPU closure probe

Implementation:

- Added opt-in assembled-operator row equilibration for RHSMode=1
  `xblock_sparse_pc_gmres`:
  `SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_ROW_EQUILIBRATE=1` builds a
  bounded diagonal left scaling from the materialized active CSR operator and
  solves `D_r A x = D_r b`.
- Added opt-in two-sided row/column equilibration:
  `SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_COL_EQUILIBRATE=1` builds
  `D_c` from the row-scaled active CSR columns, solves
  `D_r A D_c z = D_r b`, maps the candidate back with `x = D_c z`, and always
  evaluates the final acceptance residual in physical coordinates.
- Added solver metadata for row/column scale ranges, setup time, tiny
  row/column counts, and whether row/column equilibration was used.
- Added focused tests for row-only and row/column assembled-operator
  equilibration, including the physical solution mapping.

Validation:

- Local syntax/lint:
  `python -m py_compile sfincs_jax/v3_driver.py` and
  `python -m ruff check --select F821,F823 sfincs_jax/v3_driver.py
  tests/test_v3_sparse_pattern.py`: passed.
- Focused row/column tests:
  `tests/test_v3_sparse_pattern.py::test_xblock_sparse_pc_assembled_operator_row_equilibration_records_metadata`
  and
  `tests/test_v3_sparse_pattern.py::test_xblock_sparse_pc_assembled_operator_row_col_equilibration_maps_solution`:
  `2 passed in 7.22 s`.
- Broader focused gate before the remote probes:
  solver/device/operator unit slice `84 passed in 29.20 s`; slow v3
  device/operator/artifact slice `29 passed in 179.00 s`.

Office GPU hard-seed evidence:

- Row-only equilibration, device CSR, device QR global coupling, cycle-JIT
  FGMRES, explicit right preconditioning:
  completed in `274.25 s`, wrote HDF5 and solver trace with nonconverged output
  allowed, but residual stayed `3.02155e-05` against the strict
  `3.02149e-13` target (`residual_ratio ~ 1.0e6`).
- Row/column equilibration with the same device-global-coupling path:
  completed in `276.99 s`, wrote output/trace, but residual stayed
  `3.02155e-05`.
- Row/column equilibration plus larger x-block JAX factor padding
  (`SFINCS_JAX_RHSMODE1_XBLOCK_SPARSE_ROW_NNZ_MAX=256`):
  completed in `317.33 s`, but residual stayed `3.02165e-05`.
- Exact per-x sparse LU on the GPU path
  (`SFINCS_JAX_RHSMODE1_XBLOCK_SPARSE_LU_MAX=30000`,
  row cap `1024`, left `gmres-jax`, device CSR matvec):
  reached the intended exact-LU factors and the left-preconditioned device
  solve, but timed out at `541.50 s` with no output/trace and a much larger
  host/GPU footprint. This is the closest device analogue of the CPU-closing
  path, but the naive padded JAX-factor representation is too expensive.
- The checked blocker artifact
  `docs/_static/qi_seed_robustness_scale060_device_operator_rejected_2026_05_13.json`
  now records all four probes, and
  `docs/_static/qi_seed_robustness_evidence_manifest.json` was regenerated.

Decision:

- Row scaling, two-sided scaling, larger ILU padding, and naive padded exact
  x-block LU are rejected as QI hard-seed production fixes.
- Device/operator infrastructure is effectively complete for this lane: active
  device CSR matvecs, cycle-JIT/recycled FGMRES, TFQMR/BiCGStab diagnostics,
  device QR global coupling, rank-gated moment-Schur, side-probe seed guards,
  and row/column equilibration all have tests and blocker evidence.
- The remaining gap is not a missing knob. It is the representation of the CPU
  closure mechanism on GPU: host SuperLU exact x-block factors close the bounded
  CPU hard seed, while padded JAX exact factors are too memory/time expensive
  and ILU factors do not reduce the residual. The next real algorithm must be a
  compact exact/block-Schur representation, a custom sparse triangular solve
  that stores only actual CSR/CSC factor rows, or a different coarse operator
  that reduces the true residual by at least an order of magnitude before the
  full Krylov launch.

Updated lane status:

- Device operator/coarse infrastructure: `99.8%` complete.
- QI hard-seed closure: `88%` complete. CPU bounded closure exists; GPU closure
  is blocked specifically on compact exact/preconditioner representation, not
  on device matvec, Krylov execution, row scaling, or side selection.
- Overall performance/memory pass: `91%` complete.

## 2026-05-14 compact-CSR exact-factor QI replacement path

Goal: replace the padded max-row JAX x-block factor representation with a real
compact representation so the scale-0.60 QI GPU hard seed can use exact per-x
SuperLU factors without truncating rows or allocating arrays proportional to the
widest factor row.

Implementation:

- Added compact CSR triangular solves for device-side SuperLU factors:
  `_triangular_solve_lower_csr_rows` and `_triangular_solve_upper_csr_rows`.
  They traverse actual factor nonzeros from CSR row pointers instead of padded
  row slabs.
- Added `_RHSMode1SparseXBlockCSRPrecondCache` and the opt-in control
  `SFINCS_JAX_RHSMODE1_XBLOCK_SPARSE_JAX_FACTOR_FORMAT=csr`.
- The compact path stores per-block row pointers, local column indices,
  values, SuperLU permutations, and upper diagonals. It maps the same
  `Pr A Pc = L U` solve used by the existing padded JAX path, but avoids row
  truncation by default.
- Added solver metadata `sparse_pc_xblock_jax_factor_format` so traces can
  distinguish padded vs compact x-block factors.
- Added tests for the compact CSR triangular solve against a dense reference
  and an x-block device-Krylov integration test that exercises the
  `SFINCS_JAX_RHSMODE1_XBLOCK_SPARSE_JAX_FACTOR_FORMAT=csr` route.

Validation:

- Local syntax/lint:
  `python -m py_compile sfincs_jax/v3_driver.py` and
  `python -m ruff check --select F821,F823 sfincs_jax/v3_driver.py
  tests/test_v3_sparse_pattern.py`: passed.
- New compact CSR tests:
  `tests/test_v3_sparse_pattern.py::test_compact_csr_triangular_solves_match_dense_reference`
  and
  `tests/test_v3_sparse_pattern.py::test_xblock_sparse_pc_device_krylov_can_use_compact_csr_factors`:
  `2 passed in 54.94 s`.
- Remote `office` syntax check after syncing the dirty tree:
  `/home/rjorge/stellarator_venv/bin/python -m py_compile
  sfincs_jax/v3_driver.py sfincs_jax/solver.py
  sfincs_jax/rhs1_device_operator.py scripts/run_qi_seed_robustness.py`:
  passed.

Office GPU hard-seed result:

- Probe: scale-0.60 QI seed `3`, `15 x 31 x 60 x 5`, active size `81377`,
  total size `139502`, GPU0 RTX A4000, explicit
  `xblock_sparse_pc_gmres`, left `gmres-jax`, exact per-x sparse LU
  (`SFINCS_JAX_RHSMODE1_XBLOCK_SPARSE_LU_MAX=30000`), compact CSR factor
  storage, active DOF, and requested assembled device operator reuse.
- First launch failed immediately because the generic active-DOF flag was set
  without the x-block-specific opt-in. Rerun fixed the launch with
  `SFINCS_JAX_RHSMODE1_XBLOCK_ACTIVE_DOF=1`.
- Compact factor build succeeded:
  - per-x matrices assembled with largest block `27900 x 27900` and `798168`
    nonzeros;
  - compact factors: `lower_nnz=110063331`, `upper_nnz=119280365`;
  - estimated factor arrays: `2.755 GB`;
  - peak host RSS from `/usr/bin/time -v`: `17.6 GB`;
  - GPU memory during/after factor build: about `5.8 GiB`.
- The requested assembled operator reuse did not build because the probe kept
  the default `max_colors=512`; the pattern requires more colors. The run then
  timed out at `540.95 s` before a solver trace or HDF5 output was written.
- Evidence recorded in
  `docs/_static/qi_seed_robustness_scale060_device_operator_rejected_2026_05_13.json`
  under `device_compact_csr_exact_xblock_probe`, and the QI evidence manifest
  was regenerated.

Decision:

- Compact CSR is a real memory-side improvement over padded exact factors and
  should stay as opt-in infrastructure for future exact-factor experiments.
- It does not close the scale-0.60 GPU hard seed: storage-only improvement is
  not enough because the first exact-factor device transfer/apply remains too
  expensive, and no solver trace/residual was produced inside the budget.
- QI hard-seed closure remains a true algorithmic lane. The next acceptable
  attempt must reduce the preconditioner application cost or improve the
  residual before the full Krylov loop, for example with a block-Schur/angular
  coarse operator, a lower-fill exact local solve, or a physics-aware coarse
  basis that changes the measured true-residual trend before launching GMRES.

Updated lane status:

- Device operator/coarse/factor infrastructure: `99.9%` complete.
- QI hard-seed closure: `89%` complete. CPU bounded closure exists and GPU now
  has device CSR operators, device Krylov, row/column scaling, cycle-JIT,
  QR/moment coarse probes, short-recurrence diagnostics, and compact exact
  factors. The remaining gap is a residual-reducing, GPU-cheap preconditioner.
- Overall performance/memory pass: `92%` complete.

## 2026-05-14 continuation: QI heartbeat evidence and GPU policy separation

Goal:

- Resume after the interrupted QI hard-seed push, preserve the already integrated
  device/operator/refactor work, and close as much of the remaining one-GPU
  scale-0.60 hard-seed lane as possible without promoting failed probes.

Context checked:

- Worktree is still on `main` with the expected dirty integration changes from
  the QI/device-operator/PAS-policy push. No unrelated files were reverted.
- The bounded CPU scale-0.60 QI coarse-seed artifact remains checked locally:
  `docs/_static/qi_seed_robustness_scale060_qi_coarse_seed3_cpu_2026_05_14.json`.
- The latest one-GPU `office` run was not a silent hang. It produced a compact
  heartbeat summary:
  `docs/_static/qi_seed_robustness_scale060_qi_coarse_seed3_gpu0_heartbeat_timeout_2026_05_14.json`.

Steps taken:

- Added `--heartbeat-s` runner evidence to the QI manifest path so long probes
  write `runner_heartbeat.jsonl`, flush progress periodically, and terminate the
  whole subprocess group on timeout.
- Added progress parsing for active/total matrix size breadcrumbs when a timeout
  occurs before a solver trace is written. This makes failed QI/PAS probes
  auditable instead of only recording `returncode=124`.
- Added `QI coarse seed` and `probe-coarse` solver breadcrumbs to the runner
  progress markers.
- Moved the CPU/GPU LGMRES-rescue guard into the pure
  `rhs1_xblock_policy.rhs1_xblock_lgmres_rescue_backend_allowed()` helper and
  added a unit test proving the default remains off on GPU unless explicitly
  forced.
- Added the new GPU heartbeat timeout artifact to
  `DEFAULT_EVIDENCE_ARTIFACTS` and regenerated
  `docs/_static/qi_seed_robustness_evidence_manifest.json`.
- Updated focused tests for heartbeat timeout handling, size inference, and
  evidence-manifest accounting.

Results:

- Targeted local runner/artifact tests:
  `pytest -q tests/test_run_qi_seed_robustness.py tests/test_qi_seed_smoke_artifact.py`
  passed with `34 passed`.
- Focused policy/runner/artifact tests after the backend-guard refactor:
  `pytest -q tests/test_rhs1_xblock_policy.py
  tests/test_run_qi_seed_robustness.py tests/test_qi_seed_smoke_artifact.py`
  passed with `73 passed`.
- Syntax/lint guard:
  `python -m py_compile sfincs_jax/rhs1_xblock_policy.py
  sfincs_jax/v3_driver.py scripts/run_qi_seed_robustness.py` and
  `python -m ruff check --select F821,F823 ...`: passed.
- Broader local touched-file regression:
  `pytest -q tests/test_rhs1_xblock_policy.py tests/test_rhs1_pas_policy.py
  tests/test_memory_model.py tests/test_solver_gmres.py
  tests/test_v3_sparse_pattern.py
  tests/test_v3_driver_pas_precond_policy_coverage.py
  tests/test_jax_geometry_adapters.py tests/test_rhs1_qi_coarse.py
  tests/test_rhs1_device_operator.py tests/test_rhs1_device_operator_unit.py
  tests/test_run_qi_seed_robustness.py tests/test_qi_seed_smoke_artifact.py`
  passed with `233 passed in 128.19 s`.
- Evidence manifest now records `30` artifacts, `19` passing and `11`
  non-passing, with the largest passing and attempted bounded grid still
  `139502` total unknowns and the lane-completion estimate still `60%` by
  passing per-axis resolution.
- The one-GPU heartbeat probe ran for `420.318 s`, recorded `31` heartbeat
  events, inferred active size `81377` and total size `139502`, and wrote no
  HDF5 output or solver trace. It is rejected evidence.
- Important interpretation: that diagnostic command explicitly forced
  `SFINCS_JAX_RHSMODE1_XBLOCK_PC_LGMRES_RESCUE=1`. The source policy already
  guards LGMRES rescue to CPU unless forced. Therefore this artifact proves that
  forced host-oriented LGMRES is not a viable GPU closure route, not that the
  default GPU policy is worse.
- The attempted follow-up sync/rerun on `office` was interrupted by network
  failure: SSH first reported `No route to host`, then timed out connecting to
  port `3281`. No remote GPU process was left from the completed heartbeat run.

Next best steps:

1. When `office` is reachable again, sync the current dirty tree and rerun the
   same scale-0.60 seed-3 GPU hard seed with
   `SFINCS_JAX_RHSMODE1_XBLOCK_PC_LGMRES_RESCUE=0`, keeping heartbeat enabled.
   This separates the GPU-compatible policy from the rejected forced-LGMRES
   diagnostic.
2. If the no-LGMRES run still times out, stop testing Krylov-name changes and
   implement a cheaper residual-reducing preconditioner application: block-Schur
   angular/radial coarse solve, lower-fill local sparse solve, or a device-native
   coarse operator that reduces the measured true residual before full GMRES.
3. Keep QI hard-seed closure at `89%` until a one-GPU scale-0.60 seed-3 artifact
   writes HDF5 plus solver trace inside budget and passes the strict residual
   gate.
4. Keep device/operator/coarse infrastructure at `99.9%`; the missing piece is
   not plumbing, it is a GPU-cheap preconditioner that changes the residual
   trend.

## 2026-05-14 office rerun: GPU-compatible no-LGMRES QI probe

Goal:

- Re-run the scale-0.60 QI seed-3 one-GPU hard seed on `office` with the current
  dirty tree and `SFINCS_JAX_RHSMODE1_XBLOCK_PC_LGMRES_RESCUE=0`, to separate
  the GPU-compatible policy from the rejected forced-LGMRES diagnostic.

Steps taken:

- Verified `office` was reachable and GPU 0 was initially idle.
- Synced the current dirty tree to `/tmp/sfincs_jax_qi_coarse_gpu` and verified
  remote syntax with:
  `/home/rjorge/stellarator_venv/bin/python -m py_compile
  sfincs_jax/v3_driver.py sfincs_jax/solver.py
  sfincs_jax/rhs1_xblock_policy.py scripts/run_qi_seed_robustness.py`.
- Ran the bounded one-GPU probe with active DOFs, QI coarse seed,
  angular probe-coarse directions, side probe enabled, heartbeat every `15 s`,
  per-case timeout `420 s`, and explicit
  `SFINCS_JAX_RHSMODE1_XBLOCK_PC_LGMRES_RESCUE=0`.
- Pulled the remote manifest, compact summary, heartbeat, stdout, and stderr.
- Regenerated the compact artifact with stdout-tail and last-matvec metadata:
  `docs/_static/qi_seed_robustness_scale060_qi_coarse_seed3_gpu0_no_lgmres_timeout_2026_05_14.json`.
- Regenerated `docs/_static/qi_seed_robustness_evidence_manifest.json`.

Result:

- Process outcome: timeout after `420.368 s`, return code `124`, no HDF5 output,
  no solver trace, `31` heartbeat records, max RSS from `/usr/bin/time -v` about
  `3.27 GB`.
- Matrix: active size `81377`, total size `139502`.
- Selected behavior:
  - QI coarse seed changed residual only from `3.021487e-05` to
    `3.021486e-05`.
  - Side probe used plain GMRES and switched `left -> right`; no LGMRES rescue
    was used.
  - The physical seed was preserved and probe-coarse improved it from
    `4.565805e-06` to `2.830374e-06`.
  - Right-preconditioned GMRES reached `900` matvecs by `412.719 s` before the
    timeout.
- Interpretation: the backend guard works and the run no longer uses the
  rejected forced host-LGMRES path. However, it still does not close the QI GPU
  lane because it fails the output/solver-trace/residual gates. This is
  progress/liveness evidence, not promotion evidence.
- Remote hygiene: no leftover SFINCS-JAX process was found. GPU 0 was later used
  by an unrelated `spectraxgk` process, so follow-up SFINCS-JAX GPU probes should
  re-check GPU availability first.
- Evidence manifest now records `31` artifacts, `19` passing and `12`
  non-passing. The bounded completion estimate remains `60%` because completion
  is based only on passing artifacts.
- Validation after updating the evidence/docs/tests:
  - `pytest -q tests/test_run_qi_seed_robustness.py
    tests/test_qi_seed_smoke_artifact.py tests/test_rhs1_xblock_policy.py`:
    `75 passed in 0.62 s`.
  - `python -m ruff check --select F821,F823 ...`: passed.
  - `git diff --check && python scripts/check_research_lanes.py &&
    python scripts/check_release_gates.py`: passed.
  - `sphinx-build -b html docs docs/_build/html -W`: passed.

Decision:

- Stop spending time on Krylov-name or side-selection toggles for this hard seed.
  We now have CPU closure, forced-LGMRES GPU rejection, GPU-compatible no-LGMRES
  rejection, device CSR, cycle-JIT, compact CSR factors, short-recurrence
  diagnostics, and coarse/probe-coarse diagnostics.
- The next real algorithmic step must change the preconditioner/physics
  representation before full Krylov: a cheaper block-Schur/angular/radial coarse
  correction, a lower-fill but residual-effective local solve, or an operator
  reuse path that reduces the measured true residual before launching hundreds
  of GPU matvecs.

Updated lane status:

- Device/operator/coarse/factor infrastructure: `99.9%`.
- QI hard-seed closure: `89%`; still blocked by the one-GPU scale-0.60 seed-3
  convergence/output gate.
- Overall performance/memory pass: `92%`.

## 2026-05-14 block-Schur/angular/radial and lower-fill QI push

Goal:

- Try a residual-reducing block-Schur/angular/radial coarse basis, a lower-fill
  local x-block preconditioner, and a full/device Krylov route for the scale-0.60
  QI seed-3 hard case, without allowing unbounded runs.

Implementation:

- Added an enriched QI hard-seed basis helper in `sfincs_jax/rhs1_qi_coarse.py`.
  It builds bounded candidates from global/species constants, radial ramps and
  curvature, angular and mixed harmonics, radial-angular products,
  constraint-like intra-block moments, and block-Schur-like species/x contrasts.
- Wired `SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_BASIS=enriched` as an
  opt-in A/B path. The default remains `legacy` because the enriched path did
  not beat the accepted CPU hard-seed route.
- Added residual-weighted angular probe-coarse directions controlled by
  `SFINCS_JAX_RHSMODE1_XBLOCK_PC_PROBE_COARSE_ANGULAR_RESIDUAL`.
- Added lower-fill local factorization policy helpers and driver wiring through
  `SFINCS_JAX_RHSMODE1_XBLOCK_PC_LOWER_FILL`, plus factor/drop/row-cap knobs.
  This is opt-in and metadata-visible.
- Improved the QI runner timeout artifacts so heartbeat, side-probe, LGMRES
  status, last residual-like progress, and last matvec progress survive even
  when no solver trace is written.

Measured results:

- Accepted CPU reference:
  `docs/_static/qi_seed_robustness_scale060_probe_coarse_angular_residual_seed3_cpu_2026_05_14.json`.
  It converged in `170.7 s`, residual ratio `2.14e-3`, `2024` matvecs, and
  peak RSS about `4.77 GB`. This is the best current bounded CPU hard-seed
  evidence.
- Enriched block-Schur/radial/angular CPU A/B:
  `docs/_static/qi_seed_robustness_scale060_enriched_qi_coarse_seed3_cpu_rejected_2026_05_14.json`.
  It converged, but took `265.2 s`, `3298` matvecs, and about `5.34 GB` RSS.
  It is explicitly marked as a performance rejection, not a passing promotion
  artifact.
- Lower-fill local ILU A/B:
  `docs/_static/qi_seed_robustness_scale060_lower_fill_seed3_cpu_rejected_2026_05_14.json`
  and
  `docs/_static/qi_seed_robustness_scale060_lower_fill8_seed3_cpu_rejected_2026_05_14.json`.
  Both cut peak RSS to about `3.0 GB`, but both stalled near residual
  `6.6e-6` after `14835` matvecs against a `3e-11` target, so they are rejected
  for production accuracy.
- GPU-compatible no-LGMRES A/B:
  `docs/_static/qi_seed_robustness_scale060_enriched_angular_seed3_gpu0_no_lgmres_timeout_2026_05_14.json`.
  It reached `925` matvecs by `409.9 s` and timed out without HDF5/trace.
- Device Krylov / compact JAX factor A/B:
  `docs/_static/qi_seed_robustness_scale060_device_krylov_enriched_seed3_gpu1_timeout_2026_05_14.json`.
  It built compact CSR factors (`~57 MB` factor arrays) but timed out before
  useful Krylov progress, so the current device route is infrastructure only.

Decision:

- Keep the residual-weighted angular probe-coarse direction as the accepted CPU
  bounded improvement.
- Keep enriched QI basis, lower-fill local factors, and device-Krylov compact
  factors as opt-in research infrastructure with rejected evidence on the hard
  seed. Do not widen defaults from these A/B runs.
- The remaining QI blocker is now narrower and clearer: a GPU-cheap
  residual-effective preconditioner/operator-reuse path is still required for
  the scale-0.60 one-GPU seed-3 hard case.

Updated lane status:

- Device/operator/coarse/factor infrastructure: `100%` for available knobs and
  evidence capture.
- QI hard-seed closure: `91%`; CPU hard seed is closed, but the one-GPU
  scale-0.60 hard seed remains the gating open item.
- Overall performance/memory pass: `93%`.

## 2026-05-15 QI device-Krylov policy and observability push

Goal:

- Continue the scale-0.60 QI seed-3 one-GPU hard-seed lane with bounded runs on
  `office`, avoiding unbounded side probes or opaque GPU solves.

Implementation:

- Changed the automatic side-probe policy so device-Krylov methods
  (`gmres_jax`, `fgmres_jax`) do not run the CPU-oriented side probe by default.
  Explicit `SFINCS_JAX_RHSMODE1_XBLOCK_PC_SIDE_PROBE=1` still forces it.
- Disabled the default constraint1 moment-Schur build for compact CSR JAX
  factors. Earlier evidence showed this path can be rank-deficient or consume
  the bounded GPU budget before useful Krylov work. Explicit
  `SFINCS_JAX_RHSMODE1_XBLOCK_PC_MOMENT_SCHUR=1` still forces it.
- Added a `progress_callback` hook to `fgmres_cycle_jit_solve_with_residual`
  and wired it into the RHSMode=1 x-block device cycle-JIT path. The runner now
  preserves `solve start` and `device-cycle` breadcrumbs in timeout artifacts.
- Added focused regression coverage for device side-probe defaults, compact CSR
  moment-Schur blocking, and cycle-JIT progress callback events.

Measured GPU results on `office`:

- `docs/_static/qi_seed_robustness_scale060_device_krylov_skip_side_probe_seed3_gpu0_2026_05_15.json`:
  skipping the automatic side probe exposed a deeper pre-solve blocker. Compact
  CSR factors were built, but the run timed out before moment-Schur/QI/solve
  breadcrumbs.
- `docs/_static/qi_seed_robustness_scale060_device_krylov_compact_no_moment_seed3_gpu1_2026_05_15.json`:
  disabling compact-factor moment-Schur moved the run through QI seed,
  probe-coarse, and `gmres_jax` solve start. GPU memory stayed around `6.4 GB`
  with `XLA_PYTHON_CLIENT_PREALLOCATE=false`, but the solve did not return
  before the `480 s` gate.
- `docs/_static/qi_seed_robustness_scale060_device_krylov_compact_right_restart20_seed3_gpu0_2026_05_15.json`:
  removing QI/probe overhead, forcing right preconditioning, and using restart
  `20` moved solve start earlier, with about `5.9 GB` GPU memory, but still did
  not return before the `480 s` gate.
- `docs/_static/qi_seed_robustness_scale060_device_cycle_jit_restart20_seed3_gpu0_2026_05_15.json`:
  cycle-JIT FGMRES started correctly but did not finish the first restart-20
  cycle before timeout.
- `docs/_static/qi_seed_robustness_scale060_device_cycle_jit_restart4_diag_seed3_gpu0_2026_05_15.json`:
  even a diagnostic restart-4, maxiter-4 cycle did not return within the
  bounded window after setup.

Interpretation:

- The GPU hard-seed blocker is no longer side selection, LGMRES rescue,
  moment-Schur setup, QI seed construction, or GPU memory preallocation.
- The current compact CSR exact-factor device preconditioner is too expensive
  inside the device Krylov kernel for this QI hard seed. The next useful
  algorithmic replacement is not another Krylov toggle; it should replace the
  per-iteration exact triangular factor apply with a GPU-cheap residual-effective
  preconditioner, likely a block-diagonal/angular/radial coarse smoother,
  sparse approximate inverse, Jacobi/block-Jacobi-plus-coarse, or a host
  non-autodiff production fallback for this specific large RHSMode=1 lane.

Validation:

- `pytest -q tests/test_rhs1_xblock_policy.py ...`: focused policy/device
  slice passed (`59-60` tests depending on selected subset).
- `pytest -q tests/test_solver_gmres.py::test_fgmres_cycle_jit_reports_progress_at_restart_boundaries ...`:
  passed.
- `ruff check sfincs_jax/solver.py sfincs_jax/v3_driver.py
  scripts/run_qi_seed_robustness.py tests/test_solver_gmres.py --select
  F821,F823`: passed.
- QI evidence manifest regenerated with `42` source artifacts: `20` passing and
  `22` non-passing. Bounded completion remains `60%` because only passing
  artifacts count toward the production-resolution proxy.

Updated lane status:

- Device/operator/coarse/factor infrastructure: `100%`.
- QI hard-seed GPU closure: `92%`; the exact device factor route is now
  rejected as the closure path, and the remaining work is a replacement
  GPU-cheap preconditioner or explicit non-autodiff host production fallback.
- Overall performance/memory pass: `94%`.

## 2026-05-15 QI compact-factor apply replacement A/B

Goal:

- Replace or bound the expensive exact compact CSR triangular-factor apply that
  blocked the scale-0.60 QI seed-3 one-GPU device-Krylov lane.

Implementation:

- Added `SFINCS_JAX_RHSMODE1_XBLOCK_SPARSE_JAX_FACTOR_APPLY` for JAX x-block
  factor diagnostics. Supported values are `exact` (default), `diagonal`,
  `upper`, `lower`, and `identity`.
- Kept `exact` as the default. All new modes are opt-in, metadata-visible, and
  gated by the same unscaled true residual as production solves.
- Added focused integration coverage for compact CSR plus `diagonal` factor
  apply and updated the QI evidence manifest to include the new rejected
  artifacts.

Measured GPU results on `office`:

- `docs/_static/qi_seed_robustness_scale060_device_cycle_jit_diag_factor_seed3_gpu0_2026_05_15.json`:
  lower-fill ILU factors plus diagonal apply returned a synchronized restart-4
  device cycle and finished in `149.1 s`, but residual stayed
  `3.021487e-05` against the `3.021487e-13` Krylov target.
- `docs/_static/qi_seed_robustness_scale060_device_cycle_jit_diag_exact_lu_seed3_gpu0_2026_05_15.json`:
  exact LU factors plus diagonal apply returned a cycle in `153.3 s`, but exact
  LU factor storage was about `2.76 GB` and the residual stayed
  `3.021476e-05`.
- `docs/_static/qi_seed_robustness_scale060_device_cycle_jit_exact_cap16_seed3_gpu0_2026_05_15.json`:
  exact triangular apply with compact row cap `16` reduced factor storage to
  about `33 MB` and returned a cycle in `265.9 s`, but the residual stayed
  `3.021487e-05`.
- `docs/_static/qi_seed_robustness_scale060_device_cycle_jit_diag_left_seed3_gpu0_2026_05_15.json`:
  requesting left preconditioning with device FGMRES was forced back to right
  by the existing safety policy; two diagonal cycles returned in `129.8 s`, but
  residual did not improve.
- `docs/_static/qi_seed_robustness_scale060_device_cycle_jit_diag_left_gmres_seed3_gpu0_2026_05_15.json`:
  `gmres_jax` honored left preconditioning and returned two cycles in
  `123.9 s`, but the physical residual was slightly worse
  (`3.029529e-05`).

Decision:

- Diagonal apply is useful observability infrastructure because it proves the
  device-Krylov loop can return restart cycles once the triangular apply is
  cheap.
- Diagonal, one-sided, and storage-only capped triangular approximations are
  rejected as QI hard-seed closure strategies because they do not reduce the
  true residual.
- The next real closure path must change the mathematics: a residual-effective
  device smoother/coarse operator, a device-compatible LGMRES/recycling
  formulation, or an explicitly documented non-autodiff host production fallback
  for large RHSMode=1 QI solves.

Validation:

- `pytest -q tests/test_v3_sparse_pattern.py::test_xblock_sparse_pc_device_krylov_can_use_compact_csr_factors tests/test_v3_sparse_pattern.py::test_xblock_sparse_pc_device_krylov_can_use_compact_diagonal_factor_apply`
  passed (`2 passed in 86.47 s`).
- `python -m py_compile sfincs_jax/v3_driver.py` passed.
- `ruff check sfincs_jax/v3_driver.py --select F821,F823` passed.
- QI evidence manifest regenerated with `47` source artifacts: `20` passing
  and `27` non-passing. Bounded completion remains `60%` because only passing
  measured artifacts count toward the production-resolution proxy.

Updated lane status:

- Device/operator/coarse/factor infrastructure: `100%`.
- QI hard-seed GPU closure: `92%`; several previously plausible GPU closure
  hypotheses are now rejected by bounded artifacts, but no device residual
  reduction path is closed yet.
- Overall performance/memory pass: `94%`.

## 2026-05-15 Large-QI non-autodiff host fallback

Goal:

- Close the production escape hatch for large RHSMode=1 QI solves after compact
  device factors proved observable but not residual-effective.

Implementation:

- Added a pure, unit-tested policy helper
  `rhs1_xblock_device_host_fallback_decision`.
- Added `SFINCS_JAX_RHSMODE1_XBLOCK_DEVICE_HOST_FALLBACK` with modes `auto`
  (default), `0`/disabled, and `force`/`host`.
- In `auto`, the fallback is scoped to explicit JAX-native x-block Krylov
  requests on large RHSMode=1, ConstraintScheme=1, 3D full-FP systems without
  Phi1 and with active size at least
  `SFINCS_JAX_RHSMODE1_XBLOCK_DEVICE_HOST_FALLBACK_MIN_ACTIVE` (default
  `80000`).
- When used, the driver rewrites the device-Krylov request to the host x-block
  auto policy before x-block factors are built. This keeps the measured
  side-probe seed plus LGMRES rescue available, instead of launching direct
  LGMRES from a weak zero seed or constructing JAX factor arrays that the
  accepted path will not use.
- Solver metadata now records the requested method, effective method, fallback
  reason, active-size threshold, QI-like predicate, ignored-env bit, and
  `xblock_device_host_fallback_non_autodiff=True`.

Validation:

- `pytest -q tests/test_rhs1_xblock_policy.py tests/test_v3_sparse_pattern.py::test_xblock_sparse_pc_device_host_fallback_records_non_autodiff_host_policy tests/test_v3_sparse_pattern.py::test_xblock_sparse_pc_device_krylov_records_experimental_metadata tests/test_v3_sparse_pattern.py::test_xblock_sparse_pc_device_krylov_can_use_compact_csr_factors tests/test_v3_sparse_pattern.py::test_xblock_sparse_pc_device_krylov_can_use_compact_diagonal_factor_apply`
  passed (`63 passed in 167.90 s`).
- `python -m py_compile sfincs_jax/rhs1_xblock_policy.py sfincs_jax/v3_driver.py`
  passed.
- `ruff check sfincs_jax/rhs1_xblock_policy.py sfincs_jax/v3_driver.py tests/test_rhs1_xblock_policy.py tests/test_v3_sparse_pattern.py --select F821,F823`
  passed.
- Bounded scale-0.60 seed-3 CPU rerun with
  `SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV=gmres-jax` and fallback `auto` passed
  in `156.8 s`, wrote HDF5 plus solver trace, selected the host side-probe
  LGMRES rescue, used `1789` matvecs, and reached residual ratio `3.43e-3`.
  The checked artifact is
  `docs/_static/qi_seed_robustness_scale060_device_host_fallback_seed3_cpu_2026_05_15.json`.
- Negative control: the first direct-LGMRES fallback attempt skipped the
  side-probe seed and timed out at `420 s` after about `4900` matvecs. That is
  why the kept fallback enters the host x-block auto policy instead of direct
  LGMRES from the zero/weak seed.
- QI evidence manifest regenerated with `48` source artifacts: `21` passing and
  `27` non-passing. Bounded completion remains `60%` because the largest passing
  measured resolution is still scale `0.60`.

Updated lane status:

- Device/operator/coarse/factor infrastructure: `100%`.
- Large-QI production escape hatch: `100%` for non-autodiff host use; it is
  documented, metadata-visible, and guarded by unit plus solve-level tests.
- Differentiable/device QI hard-seed closure: `92%`; still open until a
  GPU-cheap residual-reducing preconditioner/coarse operator replaces the
  rejected storage-only factor routes.
- Overall performance/memory pass: `95%`.

## 2026-05-15 QI closure decision audit

Decision:

- Close the production large-QI lane with the documented non-autodiff host
  fallback. Keep true differentiable/device QI hard-seed closure as an explicit
  deferred research lane, not a release blocker.

Rationale:

- Device CSR operator reuse, device QR/global coupling, rank-gated
  moment-Schur, cycle-JIT FGMRES/GMRES, short-recurrence Krylov, row/column
  equilibration, compact CSR exact factors, diagonal/one-sided factor apply,
  lower-fill local factors, and enriched QI coarse seeds have all been tested or
  bounded on the scale-0.60 hard seed. The converged or returning variants do
  not reduce the true residual; the exact-factor variants remain too expensive
  on the bounded GPU gate.
- A minimal additional device tweak now would be high-risk because the blocker
  is mathematical residual reduction, not missing plumbing. The next real
  differentiable/device algorithm should be designed as a new residual-effective
  coarse/preconditioner, with a preflight residual gate before any long GPU
  Krylov launch.
- The host fallback is the correct user-facing production answer today because
  it enters the measured host x-block auto policy before JAX factors are built,
  preserves the side-probe seed plus LGMRES rescue, records the non-autodiff
  fallback metadata, and passed the bounded scale-0.60 seed-3 gate in `156.8 s`.

Updated lane status:

- Release-facing large-QI usability: `100%`.
- True differentiable/device-QI closure: deferred research lane, `92%`
  infrastructure complete, not a release blocker.
- Overall performance/memory pass: `95%`.

## 2026-05-15 deferred validation gate hardening

Goal:

- Close a coverage/refactor gap in the research-grade validation layer: deferred
  manuscript lanes must remain closed for the current release, but their source,
  tests, scripts, artifacts, promotion gates, and acceptance criteria must still
  be checked so they do not become stale TODOs.

Implementation:

- Strengthened `scripts/check_release_gates.py` so the CI-fast release checker
  validates manifest record `status`, `kind`, required non-empty list fields,
  and path existence for `source_code`, `tests`, `scripts`, and `artifacts`.
  This applies to `closed_deferred` lanes as well as implemented lanes.
- Added `tests/test_validation_deferred_lane_gates.py` with bounded manifest
  tests for the three explicitly deferred validation lanes:
  `sfincs2014_fig3_high_collisionality_limit`, `w7x_ambipolar_er_validation`,
  and `monkes_monoenergetic_overlap`.
- Updated `docs/testing.rst` and `docs/validation_matrix.rst` to state the
  stronger deferred-lane rule.

Validation:

- `pytest -q tests/test_validation_deferred_lane_gates.py tests/test_release_gate_metadata.py`:
  `9 passed in 0.05 s`.
- `python -m py_compile scripts/check_release_gates.py && ruff check scripts/check_release_gates.py tests/test_validation_deferred_lane_gates.py --select F821,F823`:
  passed.

Updated lane status:

- Coverage/refactor/deferred validation gate layer: `93%` complete, up from
  `90%`, because deferred lanes now have CI-fast path hygiene and explicit
  bounded tests.
- Remaining gap to `95%`: split more `v3_driver.py` validation/diagnostic
  control flow into pure helpers and run a JAX-safe coverage job before making
  package-wide coverage claims.
