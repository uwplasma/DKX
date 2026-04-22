# SFINCS_JAX Master Handoff + Execution Plan

Last updated: 2026-04-11 (America/Chicago)
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

## 5) Current State Snapshot (as of 2026-03-27)

### 5.1 Recent validated status
- Full fast explicit CPU example-suite audit is complete at `/Users/rogeriojorge/local/tests/sfincs_jax/tests/scaled_example_suite_fast_cpu_full_v7_refresh` with `39/39` `parity_ok`, `39/39` strict parity, no `jax_error`, and no `max_attempts`.
- Full frozen-reference GPU example-suite audit is complete at `/Users/rogeriojorge/local/tests/sfincs_jax/tests/scaled_example_suite_fast_gpu_full_v11_refresh` with `39/39` `parity_ok`, `39/39` strict parity, no `jax_error`, and no `max_attempts`.
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
  - the adaptive PAS smoother and structured `pas_tokamak_theta` tail are not active on the current top tokamak/geometry4 offenders,
  - `lgmres` is now wired through the CLI and safely downgraded on traced/JIT/distributed paths, but it is slower than the current defaults on `geometryScheme4_2species_PAS_noEr` and `geometryScheme5_3species_loRes`, and effectively neutral on the tokamak PAS+Er case,
  - the fresh current `main` GPU full-suite refresh now captures the big bounded-solver wins directly in the release artifact root: `geometryScheme5_3species_loRes` is down to `4.294s`, `tokamak_1species_PASCollisions_withEr_fullTrajectories` to `18.300s`, `sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories` to `7.420s`, and `sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_DKESTrajectories` to `6.314s`, all strict-clean,
  - forcing transport sparse-direct first on `monoenergetic_geometryScheme5_ASCII` is parity-clean but only marginally faster on the pinned final CPU input, so it is not yet a compelling default change,
  - the fresh GPU full-suite root also records `monoenergetic_geometryScheme5_ASCII` parity-clean at `3.938s` on the current bounded accelerator `tzfft` path,
  - and `geometryScheme4_2species_PAS_noEr` remains parity-clean at `7.019s` but is still the worst GPU RSS case at `2477.1 MB`.
  - `lineax` has been gated and is not admitted yet: on a small real SFINCS operator it matched the current residual and ran faster locally (`~0.54s` vs `~3.29s`), but on a generic nonsymmetric test matrix its default GMRES configuration stagnated, so it is still a bounded differentiable/reference-path candidate rather than a production CLI dependency.

### 5.2 Known pain points that still matter
- The pinned full-suite CPU root still records a stale pre-optimization `tokamak_1species_PASCollisions_withEr_fullTrajectories` artifact (`37.747s` JAX CPU vs `0.017s` Fortran), but current-tip frozen-case reruns on the same input are now down to about `3.56s` with parity preserved. A full-suite refresh is still needed before README tables can claim that improvement.
- Runtime ratio is still high for the heavier PAS / geometry-rich CPU cases, especially HSX / geometry4 PAS branches in the `3.5-4.9s` range on current targeted reruns.
- GPU wall time is now robust and parity-clean in the refreshed `v11` root. The remaining runtime offenders are `tokamak_1species_PASCollisions_withEr_fullTrajectories` (`18.300s`), `monoenergetic_geometryScheme1` (`14.621s`), `HSX_PASCollisions_DKESTrajectories` (`11.142s`), and `HSX_PASCollisions_fullTrajectories` (`9.584s`).
- Memory ratio remains high on select PAS/FP cases. Current worst CPU RSS offenders are `monoenergetic_geometryScheme5_ASCII` (`2773.9 MB`) and `geometryScheme4_2species_PAS_noEr` (`2623.4 MB`), while current worst GPU RSS offenders are `geometryScheme4_2species_PAS_noEr` (`2477.1 MB`) and `sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories` (`2070.4 MB`).
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
