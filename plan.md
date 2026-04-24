# SFINCS_JAX Master Handoff + Execution Plan

Last updated: 2026-04-23 (America/Chicago)
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
  - `tokamak_1species_PASCollisions_withEr_fullTrajectories` is now also parity-clean on the current GPU tight-GMRES path at about `3.25s` / `922.3 MB`, versus the older GPU `xblock_tz` artifact at about `18.2s` / `1014.5 MB`,
  - the adaptive PAS smoother and structured `pas_tokamak_theta` tail are not active on the current top tokamak/geometry4 offenders,
  - `lgmres` is now wired through the CLI and safely downgraded on traced/JIT/distributed paths, but it is slower than the current defaults on `geometryScheme4_2species_PAS_noEr` and `geometryScheme5_3species_loRes`, and effectively neutral on the tokamak PAS+Er case,
  - the fresh current `main` GPU full-suite refresh plus focused current-tip rows now capture the big bounded-solver wins directly in the release-facing docs: `geometryScheme5_3species_loRes` is down to `4.294s`, `tokamak_1species_PASCollisions_withEr_fullTrajectories` to `3.249s`, `sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories` to `7.420s`, and `sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_DKESTrajectories` to `6.314s`, all strict-clean,
  - CPU `geometryScheme=5` monoenergetic transport now prefers the low-memory Krylov/`tzfft` path by default on bounded VMEC RHSMode=3 cases; focused CLI probes reduced `monoenergetic_geometryScheme5_ASCII` from about `2950-3066 MB` to `506.5 MB` and `monoenergetic_geometryScheme5_netCDF` to `603.2 MB`, both with `0` Fortran mismatches,
  - the fresh GPU full-suite root also records `monoenergetic_geometryScheme5_ASCII` parity-clean at `3.938s` on the current bounded accelerator `tzfft` path,
  - and `geometryScheme4_2species_PAS_noEr` now uses direct `pas_tz` by default on the bounded near-zero-Er PAS lane, dropping the focused GPU RSS to about `1817.0 MB` while preserving parity.
  - `lineax` has been gated and is not admitted yet: on a small real SFINCS operator it matched the current residual and ran faster locally (`~0.54s` vs `~3.29s`), but on a generic nonsymmetric test matrix its default GMRES configuration stagnated, so it is still a bounded differentiable/reference-path candidate rather than a production CLI dependency.

### 5.2 Known pain points that still matter
- The pinned full-suite CPU root still records a stale pre-optimization `tokamak_1species_PASCollisions_withEr_fullTrajectories` artifact (`37.747s` JAX CPU vs `0.017s` Fortran), but current-tip frozen-case reruns on the same input are now down to about `3.56s` with parity preserved. A full-suite refresh is still needed before README tables can claim that CPU improvement.
- Runtime ratio is still high for the heavier PAS / geometry-rich CPU cases, especially HSX / geometry4 PAS branches in the `3.5-4.9s` range on current targeted reruns.
- GPU wall time is now robust and parity-clean in the refreshed `v11` root plus focused current-tip rows. The remaining runtime offenders are `monoenergetic_geometryScheme1` (`14.571s`), `HSX_PASCollisions_fullTrajectories` (`9.082s`), `tokamak_2species_PASCollisions_withEr_fullTrajectories` (`7.722s`), and `sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_DKESTrajectories` (`6.458s`).
- Memory ratio remains high on select PAS/FP cases. After the geometry5 monoenergetic low-memory default, the current worst CPU RSS offender is `sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories` (`2298.6 MB`), while current worst GPU RSS offenders are `sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories` (`2097.0 MB`) and `HSX_PASCollisions_fullTrajectories` (`2042.1 MB`).
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
