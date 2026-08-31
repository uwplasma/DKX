# DKX 3 Research-Grade Consolidation and Release Plan

**Status:** authoritative replacement plan and new-agent handoff  
**Audit date:** 2026-08-30  
**Audited repository:** `uwplasma/DKX`  
**Audited `main`:** `c958a947505ba31f4e6f80c6fde983ab1db05b71`  
**Audited release version:** `2.3.1`  
**Open pull requests at audit:** none found  
**Open issues at audit:** none found  
**Latest hosted state:** 21 successful checks on the audited commit  
**This file replaces:** the previous 2200-line `plan.md` and its chronological execution diary

---

## 0. Read this first

This file is the only active implementation plan for DKX 3. Git history, old pull-request descriptions, release notes, validation artifacts, and the previous `plan.md` are evidence, not instructions. When they conflict with this file, follow this file.

The previous plan was useful, and much of it has been completed. It is replaced because it still describes the August 25 baseline, marks completed work as future work, and ends with a next action that was already merged in PR #106. Its execution ledger grew into a detailed research notebook and now obscures the product, architecture, testing, and release work that remains.

A new agent must begin by running the following checks locally and updating the short **Current checkpoint** section near the end of this file if `main` has moved:

```bash
git fetch --all --tags --prune
git checkout main
git pull --ff-only
git status --short
git log -20 --oneline --decorate
python --version
python -m pip --version
```

Then inspect:

```bash
find dkx tests examples docs validation tools -type f | sort
python -m pytest --collect-only -q
python -m build
python -m twine check dist/*
```

Do not immediately launch a new production-resolution W7-X run. First verify the repository state, close architectural duplication, measure exact coverage and size, and establish the next coherent pull request.

### 0.1 Operating rules

1. **One authoritative plan.** Do not create `plan_final.md`, an agent-specific master plan, or another competing roadmap.
2. **One coherent pull request at a time.** A PR should change one public contract or one vertical implementation slice. Do not combine source migration, physics changes, numerical changes, documentation rewrites, and benchmark claims in one PR.
3. **Delete while extracting.** Moving code without deleting the superseded owner is not a refactor.
4. **No microtranches.** Do not open a PR for one private helper, one renamed file, or one additional evidence JSON unless it completes a larger acceptance gate.
5. **No evidence-file proliferation.** The recent W7-X campaign produced valuable results but too many narrowly named scripts, tests, and JSON summaries. Future evidence must enter a common registry and common runner.
6. **Claims fail closed.** A capability is stable only when its declared equations, domain of validity, tests, convergence evidence, independent comparison, API, examples, and documentation all pass its promotion gate.
7. **Performance is part of correctness.** A route that is accurate but cannot finish within its declared memory and runtime envelope is not production-ready.
8. **True residuals are mandatory.** Solver-internal or preconditioned residual estimates are diagnostics, not acceptance certificates.
9. **Keep generic numerics in SOLVAX.** DKX owns neoclassical physics, discretization choices, workflow policy, and physics-aware preconditioners. Reusable generic linear algebra belongs in `solvax`.
10. **Preserve SFINCS compatibility as an adapter.** The SFINCS namelist and HDF5 layouts remain supported permanently, but they must not define the native DKX API or internal architecture.
11. **No silent downgrades.** Unsupported native physics must raise a precise error. Never silently switch collision models, trajectories, resolution, root scope, or output meaning.
12. **Document negative results.** A failed hypothesis that rules out a numerical route belongs in the compact evidence registry or this plan's short log, not in stable source code.
13. **Do not optimize unprofiled code.** Record compile time, warm runtime, peak host/device memory, transfers, iterations, and accuracy before changing a performance-critical path.
14. **Examples are user programs.** They contain editable inputs at the top, direct object construction, execution, printing, saving, and plotting. They have no `argparse`, no `main()`, and no hidden framework.
15. **Tests are scientific evidence.** A test that only executes a line to raise coverage is not acceptable.

---

## 1. Mission

DKX 3 will be a small, understandable, fast, differentiable, research-grade neoclassical transport code for tokamaks and stellarators. It will provide one coherent workflow for:

- radially local multispecies neoclassical particle, heat, and momentum transport;
- parallel flow and bootstrap current;
- prescribed and ambipolar radial electric fields;
- complete retained root evidence and radial root-branch continuation;
- monoenergetic coefficients and thermal convolution;
- full speed-dependent Fokker-Planck calculations;
- convergence and physical-validity certificates;
- CPU and GPU execution, repeated-solve reuse, batching, and sharding;
- differentiable sensitivities and optimization;
- coupling to VMEX and other equilibrium, optimization, and transport codes;
- permanent SFINCS input/output compatibility without a SFINCS-shaped native API.

The intended public identity is:

> **DKX is the verified, differentiable neoclassical workflow for tokamaks and stellarators: physical-unit inputs, native profile and root calculations, quantified numerical and physical credibility, and fast coupling to equilibrium and optimization codes.**

DKX must not compete by accumulating the largest feature table. It should compete by making the full calculation—from equilibrium and profiles to trustworthy transport, roots, current, diagnostics, plots, and reusable output—simple, fast, inspectable, and difficult to misuse.

---

## 2. Locked product decisions

These decisions were made by the maintainers and are not open implementation questions unless new evidence makes one untenable.

| Topic | Decision |
| --- | --- |
| Major version | DKX 3 may break the DKX 2 Python API. |
| Legacy input | SFINCS `input.namelist` support remains permanently as an adapter. |
| Legacy output | SFINCS-compatible HDF5 remains permanently as an adapter. |
| Native input | TOML is primary; JSON shares the same schema. |
| Native units | User-facing values are physical or engineering units and are normalized once at the execution boundary. |
| Native output | Versioned NetCDF is canonical. |
| Python | Python 3.11 is the minimum supported version for DKX 3. |
| CLI | Standard-library `argparse` plus Rich tables, progress, diagnostics, and errors. |
| Package layout | `src/dkx` with shallow, physics- or numerics-named domain packages. |
| First flagship | Whole-profile ambipolar roots, transport, and bootstrap current. |
| Stable physics | SFINCS-v3-overlap features are stable candidates when the matched implementation and comparison gates pass. JAX-only features require their own promotion evidence. |
| External codes | SFINCS, MONKES, YANCC, KNOSOS, NEO/NEO-2, BOOTSJ, PENTA, and other codes may be installed locally for validation but are not runtime or ordinary-CI dependencies. |
| Official performance | Laptop CPU is the release baseline; the office NVIDIA GPU is the maintained accelerator platform. Exact models must be recorded. |
| Size | Fresh full clone, wheel, source distribution, and installed DKX-owned files each target less than 20 MiB. Third-party dependencies are not counted as DKX-owned files. |
| Documentation | Sphinx + MyST + Furo, organized by tutorials, how-to guides, explanation, and reference. |
| Governance | `plan.md` is merged into `main`; implementation proceeds through small feature branches. |

---

## 3. Current repository audit

### 3.1 What has been completed since the old baseline

The prior audit used commit `0d5606ce` from DKX 2.3.0. Forty commits and a large number of focused pull requests have since landed. The important completed work is summarized below; the individual PR descriptions remain in Git history.

| Area | Completed work |
| --- | --- |
| Planning and release contracts | PRs #70-#71 established an authoritative plan, evidence registries, isolated wheel/sdist execution, post-PyPI smoke logic, workflow linting, and DKX-owned artifact-size checks. |
| Native configuration | PR #72 added immutable schema-v1 `Case`, TOML/JSON readers, deterministic semantic IDs, validation, schema output, and scan preflight. |
| Native results | PR #73 added direct analytic `Case -> KineticOperator -> Result` execution and versioned NetCDF without a namelist round trip. |
| Repeated scans | PR #74 reused geometry/operator setup across compatible electric-field scans and materially reduced runtime and memory. |
| User-visible status | PR #75 separated bracketed ambipolar roots from closest-scanned fallbacks and replaced opaque solver-tier terminology with physical route names. |
| Native geometry | PR #76 added direct VMEC profile execution; PR #82 added direct Boozer profile execution. |
| Native ambipolar workflow | PR #77 added retained physical-unit all-root scans and refinement; PRs #79-#80 added deterministic refinement evidence, radial branch identity, selection reasons, and discrete branch-event records. |
| Independent validation | PR #83 added matched monoenergetic comparisons with live MONKES and YANCC; PRs #84-#86 added full-kinetic SFINCS comparisons for tokamak and W7-X cases. |
| Robust solver fallback | PR #87 added bounded scalar recovery for failed batched native points without hiding failed attempts. |
| W7-X root evidence | PRs #88-#106 built increasingly strict W7-X root, phase-space, cross-code, normalization, fixed-field, preflight, discovery, and admitted-grid seeded certificates. |
| Physical-unit correction | PR #101 corrected the native physical radial-flux conversion and superseded earlier physical flux/current values while retaining the observed root topology. |
| Fixed-field admission | PR #102 admitted a W7-X transport grid for particle/heat flux under its stated tolerance, while explicitly withholding high-zeta parallel-current admission. |
| Operational diagnosis | PR #104 demonstrated that the broad production-grid uniform all-root route was operationally unacceptable and added explicit interval-scoped seeded promotion. |
| Current admitted roots | PR #106 certified two admitted-grid W7-X seeded roots near `12.681640625` and `11.533203125 kV/m`; the result remains explicitly interval-scoped, not a proof of all roots. |

The latest audited commit has green hosted checks. There were no open pull requests or issues found at audit time.

### 3.2 Scientific status at the audited commit

The following statements are safe:

- The SFINCS-compatible stack covers a broad set of v3 models and has extensive direct comparison evidence.
- Native analytic, VMEC, and Boozer prescribed-field profile execution exists.
- Native ambipolar execution retains solved fields, currents, fluxes, true residuals, brackets, refinement information, branch IDs, and event metadata.
- Independent monoenergetic and full-kinetic cross-code comparisons exist on a bounded set of matched cases.
- The low-resolution W7-X search and admitted-grid seeded replay demonstrate real roots at the declared points.
- The broad admitted-grid uniform all-root search is not a viable production route in its present form.
- Particle and heat transport have stronger admitted-grid evidence than parallel current on the audited W7-X case.

The following statements are not yet justified:

- that the seeded W7-X intervals contain every ambipolar root on each surface;
- that a finite sampled grid excludes an even number of unresolved crossings;
- that the admitted W7-X parallel current or bootstrap profile is converged;
- that native Phi1, full tangential-drift, full-FP ambipolar, or distributed execution is production-ready;
- that the current native API is complete or is the only public path;
- that current CI enforces the final intended coverage standard;
- that the complete fresh clone satisfies the 20 MiB requirement.

### 3.3 Engineering strengths

- Isolated installed-wheel and source-distribution smoke tests now exercise finite, nontrivial science outside the checkout.
- Package-owned wheel, sdist, and installed-file sizes are well below 20 MiB.
- Solver routes report true residuals and expose forward/adjoint diagnostics.
- Structured block elimination, bounded generated-block elimination, recycled Krylov solves, and reusable preconditioners provide a strong numerical foundation.
- The native `Case` and `Result` are immutable and preserve explicit evidence rather than reducing a run to a few scalars.
- Examples already avoid `argparse` and `if __name__ == "__main__"`.
- Documentation, package, examples, external-data, and workflow-lint jobs are integrated into hosted CI.

### 3.4 Engineering gaps that now dominate

#### Packaging and governance

- `pyproject.toml` still supports Python 3.10 and includes a Python-3.10 floor job, despite the accepted Python 3.11 decision.
- Rich is not a dependency and the CLI still uses hand-written `print()` formatting.
- Documentation still uses `sphinx-rtd-theme` with an `alabaster` fallback rather than MyST/Furo.
- Source still lives in repository-root `dkx/`, not `src/dkx`.
- The full fresh clone remains above 20 MiB. Earlier measurements were about 36 MiB, while the current checked-out tree and package artifacts individually fit. A coordinated history rewrite is required only after the current tree has adequate margin.
- Green checks do not replace an audit of branch-protection required checks. Verify and enforce them.

#### Source ownership

The current package has gained native owners without retiring enough old public and orchestration paths. Major current files include approximately:

- `drift_kinetic.py`: 129 kB;
- `magnetic_geometry.py`: 100 kB;
- `multigrid.py`: 89 kB;
- `workflows/optimization.py`: 72 kB;
- `collisions.py`: 65 kB;
- `moments.py`: 63 kB;
- `cli.py`: 59 kB;
- `coarse_precond.py`: 57 kB;
- `config.py`: 56 kB;
- `execution.py`: 56 kB;
- `workflows/ambipolar_native.py`: 52 kB;
- `collocation.py`: 49 kB.

Large files are not automatically wrong, especially for equation-dense code, but several mix schema, policy, construction, execution, evidence, formatting, and compatibility responsibilities. The native stack was added to a legacy-oriented public surface rather than fully replacing it.

`dkx/__init__.py` still performs process-wide work during import: NumPy validation, environment mutation, distributed initialization, thread settings, XLA flags, cache-directory creation, JAX import/configuration, and x64 activation. `import dkx` must become inert.

`dkx/api.py` exposes generic result/request dataclasses and SFINCS-oriented execution alongside the new `Case`/`Result`. `dkx.run` is both a public concept and a module/lazy export. DKX 3 must have one obvious high-level API.

`dkx/config.py` declares workflow names that native execution does not yet implement. Scan axes are largely numeric and do not yet form a complete general workflow model. `execution.py` still rejects native scans, native transport matrices/monoenergetic workflows, Phi1, full tangential drifts, and explicit sharding. This is acceptable only while the schema and documentation state the limitation precisely.

#### Tests and evidence

- The hosted coverage threshold is still `80`, not the accepted research-grade target.
- Coverage is divided across 13 shards, indicating that the pull-request suite has grown rather than becoming simpler.
- `tests/` contains many narrowly scoped files and dated artifact directories.
- Root-level `validation/` contains many similarly named W7-X JSON summaries, each with a companion audit script and test.
- Several tests protect exact file inventories, source shapes, or historical organization rather than scientific behavior.
- The project has strong regression evidence but does not yet present a compact, explicit proof matrix for every mathematical, physical, and numerical component.

#### Documentation and examples

- The documentation landing page is still namelist-first and describes DKX mainly as SFINCS in JAX.
- Public navigation includes roadmaps, research lanes, release checklists, old upstream pages, and implementation history beside user documentation.
- The API reference mixes native public contracts with internal modules and legacy pathways.
- The README is long, benchmark-heavy, and still uses the old `sfincs-jax` documentation identity.
- Examples are spread across overlapping trees: numbered basics, getting started, tutorials, transport, autodiff, optimization, VMEX, SFINCS examples, and native TOMLs.
- Native TOML examples are not consistently paired with equally clear native Python examples.
- Some introductory Python examples still construct legacy SFINCS keyword dictionaries instead of the DKX 3 native objects.

#### Plotting

- Native `Result.plot()` is a good start but supports only a small set of automatic panels and a fixed simple layout.
- The SFINCS output plotting module uses index axes for several fields, ad hoc labels, fixed page layouts, and scattered plot construction.
- Plotting lacks a common units/labels registry, uncertainty/convergence overlays, root-branch/event visualization standards, accessible style checks, and semantic figure validation.
- Publication-facing figures and user-facing plots are generated by several unrelated modules and scripts.

---

## 4. Research-grade standard

Research-grade DKX requires more than passing regression tests or matching SFINCS on a few outputs. Each stable capability must satisfy four layers.

### 4.1 Code verification: are the implemented equations solved correctly?

Verification evidence includes:

- exact algebraic identities;
- manufactured solutions and observed convergence order;
- conservation laws and null spaces;
- discrete symmetry and adjoint identities;
- true primal and transpose residuals;
- solver-route invariance;
- grid-refinement studies with quantified error;
- finite-difference, complex-step, or independent automatic-differentiation checks of derivatives;
- CPU/GPU and eager/JIT equivalence at matched precision.

The method of manufactured solutions should be used wherever a full operator or differential stencil can be supplied with a known constructed answer. This catches implementation errors that a frozen-output regression can preserve indefinitely.

### 4.2 Model validation: are the equations appropriate and are results consistent with independent knowledge?

Validation evidence includes:

- analytic neoclassical limits;
- published asymptotes and device cases;
- matched comparisons to independent codes;
- comparison to experimental or independently digitized data when appropriate;
- documented distinctions among trajectory, collision, locality, Phi1, and drift models.

SFINCS is a crucial reference but is not an independent oracle for code paths intentionally copied from it. DKX needs a mixture of SFINCS, MONKES, YANCC, KNOSOS, NEO/NEO-2, BOOTSJ/PENTA, analytic theory, and experimental evidence selected by capability.

### 4.3 Numerical uncertainty and validity

Every production result should be able to report:

- requested and achieved true residual;
- discretization settings;
- convergence-rung changes in requested observables;
- variational gap where available;
- root bracket, slope, separation, search scope, and branch-selection reason;
- local-model validity indicators, including finite-orbit-width and resonance warnings;
- model choices and omitted terms;
- runtime, memory, and route provenance;
- DKX/JAX/SOLVAX versions, commit, platform, and geometry/input hashes.

### 4.4 Reproducibility and software credibility

A stable release must provide:

- a clean wheel and sdist that reproduce the documented examples;
- a versioned input and output schema;
- deterministic compact evidence artifacts and checksums;
- source, test, documentation, benchmark, and environment provenance;
- `CITATION.cff`, DOI archive, release notes, and machine-readable capability status;
- no claim whose generating command, input, reference, and acceptance rule cannot be identified.

---

## 5. Target public API

DKX 3 must have one obvious high-level path.

### 5.1 Canonical Python workflow

```python
from pathlib import Path

import dkx

case = dkx.Case.from_file("w7x_profile.toml")
result = dkx.run(case)

result.print_summary()
result.save(Path("output/w7x_profile.nc"))
result.plot(Path("output/w7x_profile.pdf"))
```

Programmatic construction must be equally direct:

```python
case = dkx.Case(
    name="tokamak-profile",
    geometry=dkx.Geometry.vmec("wout_tokamak.nc", surfaces=[0.25, 0.5, 0.75]),
    species=[
        dkx.Species.deuterium(...),
        dkx.Species.electron(...),
    ],
    physics=dkx.Physics.full_local(
        collisions="fokker_planck",
        electric_field="ambipolar",
    ),
    numerics=dkx.Numerics(...),
)
result = dkx.run(case)
```

Exact class names may be adjusted during implementation, but the number of concepts a normal user must memorize must remain small.

### 5.2 API levels

1. **Stable high-level API**
   - `Case`, `Result`, `run`, `scan`, `converge`, `plot`, `read_result`;
   - physical units and native names;
   - all normal workflows.

2. **Stable expert API**
   - geometry, species, grids, operators, solvers, moments, and certificates;
   - typed objects with documented normalization and shape contracts;
   - suitable for VMEX and other code coupling.

3. **Compatibility API**
   - `dkx.compat.sfincs.read_namelist`;
   - `dkx.compat.sfincs.run`;
   - `dkx.compat.sfincs.write_hdf5`;
   - SFINCS names and normalizations stay confined to this namespace and output adapter.

4. **Internal API**
   - private implementation helpers;
   - no compatibility guarantee.

### 5.3 Required API corrections

- Remove the ambiguity between a `dkx.run` function and `dkx.run` module.
- Retire or merge `SolveInputs`, `TransportResult`, and other facade types that duplicate `Case` and `Result`.
- Make workflow-specific `Case` validation exact: a declared workflow must either execute natively or fail during validation, not deep inside execution.
- Add typed native support for:
  - prescribed-field surface/profile;
  - ambipolar profile;
  - monoenergetic database;
  - transport matrix;
  - convergence study;
  - declarative scan;
  - differentiable objective evaluation.
- Allow scan values to be numeric, boolean, categorical, or quantity-like where the schema permits.
- Separate immutable case semantics from runtime state and reuse state.
- Introduce a documented `ReuseState` or equivalent that can hold geometry transforms, collision data, factors, preconditioners, initial states, compiled-shape metadata, and recycled Krylov spaces without being serialized as part of the physical input.
- Replace kernel printing with structured progress events consumed by the CLI, notebooks, or callbacks.
- Make `import dkx` inert. Runtime configuration is explicit through `dkx.runtime.configure(...)` and the CLI bootstrap.
- Keep all public objects type-checkable and documented.

### 5.4 Native input contract

TOML is the human format. JSON is machine-friendly and uses the same schema. The schema must include:

- geometry source and surfaces;
- species names, charges, masses, profiles, gradients, and kinetic/background role;
- model terms and trajectory/collision choices;
- prescribed, scanned, or ambipolar electric field;
- resolution and convergence ladders;
- solver and memory policy;
- CPU/GPU/batching/sharding policy;
- requested outputs and retained evidence;
- declarative scans and resume behavior.

Physical units should be explicit. Prefer a small fixed canonical-unit schema (`T`, `m`, `m^-3`, `eV`/`keV`, `kV/m`, `W/m^2`, etc.) and clear field names over a large unit-package dependency. If string quantities are accepted, implement a deliberately small parser with proof tests or use a well-justified optional dependency.

Every case has:

- `schema_version`;
- deterministic semantic `case_id` independent of formatting and irrelevant defaults;
- source-file-relative path resolution;
- a complete normalized echo stored in the output;
- a preflight report for work, retained evidence, unsupported combinations, and likely recompilation families.

### 5.5 Native result contract

NetCDF is canonical. A `Result` must contain or reference:

- complete normalized input;
- physical coordinates and per-variable `units`, `long_name`, dimensions, and normalization metadata;
- species-resolved fluxes, flows, currents, moments, and transport matrices;
- every root and every retained root-search evaluation when requested;
- selected branch, branch IDs, event intervals, and selection reason;
- convergence and validity certificates;
- solver attempts, true residuals, route, iterations, setup/solve timing, and memory;
- software versions, commit, backend, platform, precision, and hashes;
- optional distribution-function data behind an explicit output choice.

The SFINCS HDF5 adapter translates this result or the compatibility run into the established SFINCS keys without making those keys the native in-memory representation.

### 5.6 CLI contract

The final normal-user CLI should be small:

```text
dkx doctor
dkx schema
dkx validate CASE
dkx run CASE
dkx scan CASE
dkx roots RESULT-or-CASE
dkx converge CASE
dkx plot RESULT
dkx inspect RESULT
dkx convert SOURCE DESTINATION
dkx compare A B
```

SFINCS-specific operations belong under a compatibility group or explicit flags rather than many unrelated top-level commands.

Rich is used for:

- case summaries;
- phase and scan progress;
- solver iteration summaries at a controlled cadence;
- root tables and branch events;
- warnings and unsupported-model explanations;
- final output paths, runtime, memory, and certificate status.

The core emits structured events. Rich formatting remains in the CLI layer.

---

## 6. Target source architecture and size discipline

### 6.1 Intended package layout

Use one domain level under `src/dkx`; do not build a deep framework.

```text
src/dkx/
├── __init__.py
├── _version.py
├── api.py
├── case.py
├── result.py
├── runtime.py
├── cli.py
├── geometry/
│   ├── model.py
│   └── readers.py
├── kinetics/
│   ├── grids.py
│   ├── collisions.py
│   ├── operator.py
│   └── moments.py
├── solvers/
│   ├── linear.py
│   ├── structured.py
│   ├── krylov.py
│   ├── preconditioners.py
│   └── implicit.py
├── workflows/
│   ├── surface.py
│   ├── profile.py
│   ├── ambipolar.py
│   ├── monoenergetic.py
│   ├── convergence.py
│   └── optimization.py
├── io/
│   ├── native.py
│   ├── sfincs.py
│   └── plotting.py
├── physics/
│   ├── units.py
│   ├── validity.py
│   └── reduced.py
└── validation/
    ├── registry.py
    └── compare.py
```

This is a target, not an instruction to mechanically create every file. Merge adjacent owners when one clear file is better. Do not create empty abstraction layers.

### 6.2 Source budgets

Budgets are review triggers, not invitations to game line counts.

- Target no more than about **35 production Python files**.
- Target no more than about **45,000 production lines**, including useful docstrings but excluding generated version data.
- A file above **1,200 lines** requires a clear single-responsibility justification.
- A file above **1,800 lines** must be split or receive an explicit exception in this plan.
- No stable attempt-, date-, device-, or campaign-named source files.
- No duplicate public orchestration paths.
- No generic linear-algebra implementation in DKX when SOLVAX can own it.
- No large checked-in raw output, profiler trace, equilibrium archive, movie, or benchmark state.

### 6.3 Refactor method

Refactor by vertical slice:

1. establish a behavior/evidence baseline;
2. introduce the new owner;
3. route one complete public workflow through it;
4. compare results, true residuals, runtime, and memory;
5. delete the old owner and old tests in the same PR;
6. update the API, source map, docs, and examples;
7. lower file/line budgets rather than raising a permanent ratchet.

Do not split equation code merely to reduce line count. Separate policy, data models, construction, and formatting from the equation kernel first.

### 6.4 Import and runtime policy

`import dkx` must:

- import no JAX backend;
- mutate no environment variables;
- initialize no distributed runtime;
- create no cache directory;
- choose no CPU thread count;
- suppress no process-wide warnings;
- finish quickly and deterministically.

The CLI calls `dkx.runtime.configure()` before importing heavy execution modules. Embedded users configure JAX themselves or call the same explicit function.

---

## 7. Test, proof, and validation architecture

### 7.1 Coverage contract

The current CI gate of 80% is temporary and insufficient.

The intended DKX 3 end state is:

- **100% line and branch coverage of stable reachable Python code**;
- exclusions only for generated version data, platform-unavailable external callbacks, or defensive impossibilities, each with a documented reason;
- no stable physics branch excluded merely because it is expensive;
- performance and large external comparisons separated from the coverage denominator when their logic is already exercised by bounded fixtures.

Raise the gate through measured ratchets while deleting dead code:

```text
80 -> 90 -> 95 -> 98 -> 100
```

Do not add hundreds of shallow tests to reach the number. First remove unreachable, duplicate, experimental, and compatibility-only code from the stable denominator.

### 7.2 Compact test tree

Converge toward roughly 8-12 substantive test modules, heavily parametrized:

```text
tests/
├── conftest.py
├── test_case_api.py
├── test_math.py
├── test_physics.py
├── test_numerics.py
├── test_solvers.py
├── test_workflows.py
├── test_io_cli_plotting.py
├── test_validation.py
├── test_performance_contracts.py
└── test_packaging.py
```

A few focused files may be justified, but one test file per campaign, PR, or artifact is not.

Remove:

- dated result directories;
- exact source-file inventories that block legitimate refactoring;
- exact line ceilings;
- tests for historical roadmap wording;
- duplicate frozen-output tests that assert the same behavior;
- large tracked fixtures that can be generated analytically or fetched from checksum-pinned release assets.

### 7.3 Mathematical proof tests

The following properties must be tested directly.

#### Grids, interpolation, and derivatives

- Fourier differentiation is exact to roundoff for every representable Fourier mode.
- Periodic derivative matrices annihilate constants and satisfy expected skew/symmetry identities.
- Finite-difference and upwind stencils recover their declared order on manufactured periodic functions.
- Speed-grid quadrature integrates the polynomial/Maxwellian moments used by the collision and moment operators to the expected order or exactness.
- Interpolation reproduces constants and representable modes and obeys conservative/periodic boundary conventions.
- VMEC half/full-mesh interpolation and Boozer Fourier conventions are checked against independently constructed fields.

#### Linear algebra and solver structure

- Block-tridiagonal extraction reconstructs the operator exactly on admitted models.
- Structured forward and transposed solves match dense and sparse referees on bounded systems.
- Multi-RHS factor reuse returns the same answers as separate solves.
- Generated/checkpointed and materialized routes are algebraically equivalent.
- Preconditioners do not change the converged answer.
- Warm starts, recycled subspaces, and cached factors do not change the answer.
- True residual acceptance is independent of solver-internal residual estimates.
- Mixed-precision factors are admitted only with double-precision defect correction and final true-residual/observable agreement.

#### Root and continuation algorithms

Use analytic current functions with known roots to prove:

- single and multiple sign-changing roots;
- exact-grid roots;
- closely spaced roots;
- no-root cases;
- tangential/double roots that do not change sign;
- branch creation, loss, merger, crossing, and classification transitions;
- deterministic branch IDs and selection;
- seeded-interval scope versus global-search scope;
- failure to exclude hidden even-numbered crossings under finite sign sampling.

### 7.4 Physics proof tests

#### Collision operators

For each declared collision model and species coupling:

- Maxwellian equilibrium/null modes;
- particle conservation;
- total momentum conservation where the model claims it;
- total energy conservation where the model claims it;
- interspecies exchange antisymmetry/balance;
- self-adjointness or weighted symmetry when applicable;
- nonnegative entropy production/H-theorem form where applicable;
- correct Lorentz/PAS limit;
- Rosenbluth-potential equations and boundary behavior;
- exact behavior under species relabeling and identical-species limits.

#### Drift-kinetic operator

- constant and symmetry-preserving null modes;
- streaming/mirror/drift term parity and periodicity;
- phase-space conservation/compressibility identities for each trajectory model;
- equivalence of trajectory models at `E_r = 0` where theory requires it;
- zero-drive solutions and fluxes;
- sign and coordinate transformation identities;
- manufactured solutions for separately enabled operator terms and selected full combinations;
- bordered constraints remove only the intended null space.

#### Moments and transport

- moments of manufactured distributions;
- radial-coordinate chain rules;
- dimensional/normalized round trips;
- Onsager symmetry in regimes where it applies;
- positive-semidefinite entropy-production transport form where it applies;
- variational lower/upper ordering and gap closure under refinement;
- Spitzer conductivity and analytic tokamak limits;
- Shaing-Callen/Boozer-Gardner collisionless bootstrap limit;
- Pfirsch-Schlüter/high-collisionality limits;
- classical flux and impurity-screening identities;
- monoenergetic-to-thermal convolution against analytic manufactured coefficient functions.

### 7.5 Numerical convergence tests

For each stable discretization and workflow:

- observed order on manufactured solutions;
- convergence of requested observables, not only state-vector norms;
- independent refinement of theta, zeta, pitch, speed, collision-potential order, and root search;
- joint-tail checks where separated one-axis convergence can be misleading;
- residual-versus-discretization error separation;
- deterministic failure when the requested tolerance is below the achievable roundoff floor;
- retained uncertainty estimate in the `Result` certificate.

### 7.6 Derivative tests

- JVP/VJP adjoint identities for the operator and moment maps;
- implicit linear-solve gradients against finite differences or complex step on bounded cases;
- ambipolar-root implicit gradients on a fixed simple branch;
- branch-event intervals explicitly marked nonsmooth rather than differentiated through as if smooth;
- gradient independence from Krylov iteration history;
- CPU/GPU derivative agreement at matched precision;
- Taylor tests for every stable optimization objective.

### 7.7 Independent validation

Create one local/release runner that can discover installed external codes and execute matched cases. It must record exact external commits/builds and never be required for ordinary users or pull-request CI.

The runner must compare equations before numbers:

- geometry and radial coordinate;
- normalization and units;
- trajectory model;
- collision model and conservation properties;
- species and gradients;
- electric-field definition;
- resolution and tolerance;
- output conversion.

Use external references by domain:

- SFINCS for full local multispecies parity and compatibility;
- MONKES for Legendre/block monoenergetic coefficients;
- YANCC for independent full/mono JAX discretization and GPU behavior;
- KNOSOS for low-collisionality bounce-averaged/tangential-drift/Phi1 regimes;
- DKES/NEO/NEO-2 for established monoenergetic and stellarator transport coefficients;
- BOOTSJ and analytic formulas for bootstrap-current regimes;
- PENTA for ambipolar scan/root workflow conventions, while retaining real DKX solves at every promoted point;
- published experimental or profile-prediction cases only with explicit uncertainty and model limitations.

### 7.8 Validation registry

Replace the forest of campaign JSON files and audit scripts with:

```text
validation/
├── registry.toml
├── cases/
├── summaries/
└── README.md
```

Each registry entry contains:

```toml
[id]
capability = "ambipolar_profile"
status = "validated_limited"
claim = "..."
model = "..."
domain = "..."
input = "..."
reference = "..."
command = "..."
observables = ["..."]
tolerances = { ... }
source_tests = ["..."]
artifact = "..."
external_commit = "..."
limitations = ["..."]
```

One generic audit command validates checksums, schema, status, paths, tolerances, and claims. Large raw outputs live in GitHub release assets. Compact summaries remain in Git only when necessary for CI.

### 7.9 CI lanes

#### Pull request, target under 10 minutes

- Ruff/format/type checks;
- line and branch coverage on the installed wheel;
- mathematical and bounded physics proof tests;
- API/CLI/IO/plot semantic tests;
- warning-clean docs, doctests, and links where network-independent;
- wheel/sdist isolated execution;
- package/repository current-tree size;
- one CPU backend.

Consolidation should reduce the current 13 coverage shards to the smallest reliable number, preferably 4-6, without increasing wall time.

#### Nightly

- medium convergence ladders;
- CPU/JIT/eager equivalence;
- GPU tests on the office host;
- repeated-solve and memory-leak tests;
- current external-code matched cases where available;
- mutation testing of critical pure logic and solver policy.

#### Release

- complete declared capability registry;
- official laptop benchmark suite;
- office NVIDIA GPU suite;
- external-code campaigns;
- documentation examples from installed artifacts;
- PyPI artifact smoke;
- DOI/release evidence archive.

---

## 8. Documentation plan

### 8.1 Tooling

Move to:

- Sphinx;
- MyST Markdown for narrative pages;
- Furo theme;
- autodoc/autosummary for API;
- MathJax;
- a small code-copy extension if useful;
- no notebook dependency unless a tutorial genuinely needs interactive execution.

Build with warnings as errors. Add link checking, spelling/terminology checks for key vocabulary, and code/example execution from an installed wheel.

### 8.2 Information architecture

Use four user needs rather than project history.

```text
docs/
├── index.md
├── getting-started/
│   ├── install.md
│   ├── quickstart.md
│   └── first-profile.md
├── tutorials/
│   ├── tokamak.md
│   ├── stellarator.md
│   ├── ambipolar-roots.md
│   ├── monoenergetic.md
│   ├── convergence.md
│   └── optimization.md
├── how-to/
│   ├── vmec-boozer.md
│   ├── scans.md
│   ├── gpu.md
│   ├── resume.md
│   ├── compare-sfincs.md
│   └── troubleshoot.md
├── theory/
│   ├── ordering-and-model.md
│   ├── coordinates-and-normalization.md
│   ├── trajectories.md
│   ├── collisions.md
│   ├── discretization.md
│   ├── solvers.md
│   ├── ambipolarity.md
│   ├── bootstrap-current.md
│   ├── monoenergetic-convolution.md
│   ├── validity.md
│   └── differentiability.md
├── reference/
│   ├── case-schema.md
│   ├── result-schema.md
│   ├── cli.md
│   ├── api.md
│   ├── outputs.md
│   ├── compatibility.md
│   └── capability-matrix.md
├── validation/
│   ├── philosophy.md
│   ├── analytic-limits.md
│   ├── cross-code.md
│   ├── performance.md
│   ├── known-limitations.md
│   └── reproducibility.md
└── development/
    ├── architecture.md
    ├── testing.md
    ├── benchmarking.md
    ├── contributing.md
    └── releasing.md
```

Roadmaps, campaign diaries, and agent logs do not appear in normal user navigation.

### 8.3 Physics documentation standard

Each model page must state:

1. the physical question;
2. assumptions and ordering;
3. variables and units;
4. governing equations;
5. coordinate and normalization definitions;
6. boundary, gauge, and constraint conditions;
7. derivation or a transparent route to it;
8. discretization and solver;
9. inputs and outputs;
10. domain of validity and failure modes;
11. validation evidence;
12. primary sources;
13. exact source-code owner.

The original SFINCS notes and manuals are source material, not pages to copy verbatim. Rewrite them in consistent DKX notation, preserve attribution, explain transitions, and link each implemented term to source and tests.

### 8.4 README

Hard cap 250 lines, enforced by `tests/test_benchmark_doc_claims.py`. That
ceiling was raised three times before the cap was restored -- 327, 345, 347 --
each with a comment justifying that particular addition, which is how a budget
becomes a rubber stamp. Raising it again requires deleting something else.

The README answers, in this order: what is this, how do I install it, how do I
run it, how fast is it, how accurate is it, what can it do, what can it not do.
Anything else belongs in the documentation. Benchmark narratives, optimization
demonstrations, and implementation notes are not README material.

Target no more than about 250 lines:

1. badges;
2. one-sentence purpose;
3. one honest representative figure;
4. installation;
5. native TOML quickstart;
6. native Python quickstart;
7. core capabilities and declared limitations;
8. documentation and citation links;
9. license.

Move detailed benchmarks, implementation notes, and long scientific claims to the documentation.

### 8.5 Prose quality

The failure mode is specific and recurring: a paragraph that packs six facts
into one sentence, joined by em-dashes and subordinate clauses, so a reader
looking for one number has to parse all six. Two real examples, both removed:

> ``dkx wout_XXX.nc`` writes ``<name>.panels.png`` and ``<name>.panels.h5``:
> monoenergetic D11/D31/D33 vs nuPrime across the 1/nu, plateau and
> Pfirsch-Schlueter regimes (a curve per EStar), |B| on the surface, and --- at
> the ambipolar root, against radius --- the bootstrap current in kA/m^2 beside
> the VMEC equilibrium's own, and species fluxes in SI units. [...]

> With the matched Nxi-for-x ramp discretization, DKX solves in 27.2 s at
> 0.93 GB --- 17x faster than 1-rank Fortran (463.6 s, 3.98 GB) and 8.4x faster
> than Fortran's best measured parallel floor (229.5 s / 2.86 GB at 2 ranks),
> at roughly 30% of the memory. [...]

Both are tables, not paragraphs. A comparison of measured numbers is always a
table. Rules that follow from this:

- put the result first; a reader must not read prose to reach a number;
- a sentence carries one fact, and a comparison of numbers is a table;
- headings name the content, not the act of thinking about it. "Interpreting a
  difference" and "Understanding the results" are not headings;
- delete throat-clearing: "It is worth noting", "Importantly", "In practice",
  "The practical point is", "This is not specific to";
- do not restate a claim in the next paragraph in different words;
- an em-dash aside that carries a fact should be its own sentence;
- a section that needs three paragraphs of setup belongs in the reference
  documentation, with a link from where the user actually is;
- prefer a figure to a description of a figure.

Apply concrete no-slop rules:

- lead with the result or user action;
- remove generic scene-setting and repeated summaries;
- avoid inflated adjectives and unsupported superlatives;
- define every specialist term before use;
- do not say a feature is “robust”, “production”, “exact”, or “validated” without naming the gate;
- prefer examples and measured quantities over promotional prose;
- keep paragraphs focused on one idea;
- use active, direct language;
- retain uncertainty and limitations beside the claim, not on a distant page.

---

## 9. Examples plan

### 9.1 Canonical example ladder

Replace the overlapping current folders with no more than ten user examples. Each directory contains a Python script and, when relevant, an equivalent TOML file.

```text
examples/
├── 01_tokamak_profile/
│   ├── run.py
│   └── case.toml
├── 02_vmec_stellarator/
│   ├── run.py
│   └── case.toml
├── 03_boozer_stellarator/
│   ├── run.py
│   └── case.toml
├── 04_monoenergetic_scan/
│   ├── run.py
│   └── case.toml
├── 05_ambipolar_profile/
│   ├── run.py
│   └── case.toml
├── 06_convergence_certificate/
│   ├── run.py
│   └── case.toml
├── 07_gradients/
│   └── run.py
├── 08_vmex_optimization/
│   └── run.py
├── 09_phi1_and_impurities/
│   ├── run.py
│   └── case.toml
└── README.md
```

Do not add an example until its public API is stable enough for users to copy.

### 9.2 Example style contract

Every script follows this visible sequence:

```python
# 1. Imports
# 2. User-editable parameters
# 3. Geometry and species construction
# 4. Physics and numerical configuration
# 5. Run
# 6. Print a scientific summary and certificate
# 7. Save native result
# 8. Plot publication-ready outputs
```

Requirements:

- no `argparse`;
- no `main()` or `if __name__ == "__main__"`;
- no hidden helper functions that belong in the library;
- editable parameters at the top;
- clear expected runtime and physical regime in the module docstring;
- direct display of convergence/residual/validity status;
- deterministic output directory;
- at least one printed physical result with units;
- saved NetCDF;
- saved polished plot;
- equivalent CLI command when a TOML exists;
- no environment-dependent scientific changes inside the user example.

CI should run intrinsically small examples. Heavy examples run nightly; do not mutate user code through `CI=True` branches that change the scientific case.

### 9.3 Compatibility examples

Keep a compact set of SFINCS decks under validation or `examples/compatibility/`, clearly separated from the native teaching path. Do not expose hundreds of upstream inputs as the main example gallery.

---

## 10. Plotting and visual-output standard

Consolidate all public plotting under one `dkx.io.plotting` owner with reusable semantic plot functions.

### 10.1 Required plot families

- geometry and `|B|` in physical/Boozer coordinates;
- species particle and heat flux profiles with units;
- bootstrap/parallel-current profiles;
- radial current versus electric field with every evaluation, bracket, root type, selected branch, and failed interval;
- branch continuation versus radius with creation/loss/crossing/event intervals;
- monoenergetic coefficients versus collisionality and electric field;
- convergence plots with accepted tolerances and uncertainty bands;
- solver residual and route diagnostics;
- model-validity indicators;
- comparison plots with ratio/difference panels and reference uncertainty;
- optimization histories and before/after physics.

### 10.2 Plot contract

- axes use physical coordinates and units, never unexplained array indices when coordinates are available;
- titles state the case and quantity, not internal key names;
- legends identify species and model;
- accessible palettes and distinguishable line styles;
- sensible scientific notation and zero lines;
- no clipped labels or overlapping colorbars;
- vector PDF/SVG for line art, compressed WebP/PNG for raster content;
- WebM for short movies;
- metadata or adjacent JSON records the generating case and commit;
- every documentation figure has a generating example or registered validation command.

### 10.3 Figure tests

Avoid fragile pixel-only tests. Test semantics:

- expected axes, labels, units, lines, roots, and annotations;
- plotted data equal the result arrays;
- missing data are explained, not rendered as zero;
- no-root and interval-scoped status appear visibly;
- output opens and has a bounded file size;
- a small curated visual review gallery may be checked manually at release.

---

## 11. Performance and parallelism plan

### 11.1 Measurement contract

Every performance claim records separately:

- input and exact model;
- grid and total unknowns;
- requested and achieved true residual;
- observable difference from the accuracy reference;
- process startup/import;
- geometry and operator construction;
- compilation;
- preconditioner/factor setup;
- solve;
- postprocessing/output;
- total cold wall time;
- warm repeated wall time;
- peak process RSS;
- peak device memory;
- host-device transfer time/volume when available;
- iterations and matrix-vector products;
- factor/preconditioner/compile reuse;
- hardware, software versions, precision, and thread/device topology.

JAX timings must synchronize with `.block_until_ready()`. CPU/GPU and cross-code comparisons use matched precision and matched numerical accuracy.

### 11.2 Solver routes

#### Route S1: structured block elimination

Use when the Legendre/block structure is exact enough for the declared model. Preserve:

- multi-RHS factor reuse;
- selected-low-mode output where mathematically sufficient;
- generated/checkpointed forms when full bands do not fit;
- transposed factor reuse for adjoints;
- iterative refinement and final true residual.

This route follows the strongest idea in MONKES and the existing DKX/SOLVAX implementation.

#### Route S2: matrix-free flexible recycled Krylov

Use for general full local operators. Require:

- GCROT/FGMRES or measured alternative;
- physics-aware right preconditioning;
- true-residual refresh and final certification;
- bounded Krylov memory;
- warm initial states and recycled spaces across scans/surfaces;
- exact transpose action and certified adjoint solve.

#### Route S3: physics-aware coarse preconditioner

The preconditioner may simplify collisions, drifts, resolution, or coupling only when:

- the approximation is explicitly documented;
- it preserves the important null-space pins and constraints;
- its application is independently verified;
- it reduces complete solve time and/or memory on representative hard cases;
- reuse policy is based on measured operator changes.

#### Route S4: sparse-direct referee

Use host sparse direct factorization for bounded systems, debugging, and independent solver verification. Study PETSc/MUMPS/SuperLU_DIST ordering, symbolic reuse, factor reuse, diagonal handling, iterative refinement, and distributed algorithms, but do not add them as normal DKX dependencies.

#### Route S5: experimental multigrid/collocation

Do not promote because a small case runs. Promotion requires:

- a clear discretization/model contract;
- convergence over a representative regime matrix;
- complete runtime and memory improvement;
- CPU and GPU evidence;
- derivative support if claimed;
- fewer total lines/complexity than the benefit justifies.

### 11.3 Reuse model

Separate each solve into:

1. analysis;
2. geometry/grid/collision construction;
3. operator construction;
4. factor/preconditioner construction;
5. solve;
6. certification;
7. observable extraction.

A reusable state should support:

- many right-hand sides;
- electric-field scans;
- root refinement;
- adjacent radial surfaces;
- collisionality scans;
- optimization iterations;
- forward and adjoint solves.

Each reuse decision must record why it is mathematically valid and when it is invalidated.

### 11.4 JAX standards

- JIT the outer stable computation, not hundreds of tiny helpers.
- Keep hot functions pure and arguments as stable PyTrees.
- Keep shapes static across a scan family when practical.
- Do host-side file parsing and policy selection outside JIT.
- Avoid closing over large arrays that become XLA constants.
- Use `lax.scan`/`lax.fori_loop` for compiled loops and `lax.map` for memory-bounded batches.
- Use `vmap` only after measuring its memory footprint.
- Use explicit sharding for independent surfaces, electric-field points, collisionalities, and geometries before attempting to shard one Krylov solve.
- Keep arrays correctly sharded before calls to avoid reshard copies.
- Use buffer donation only when the caller's ownership contract makes reuse impossible.
- Profile with XProf/Perfetto and device-memory snapshots before kernel work.
- Use Pallas/custom kernels only after a profile identifies a dominant unfused operation that XLA cannot optimize adequately.
- Preserve double precision for final scientific results. Mixed precision is restricted to preconditioners/factors with double-precision refinement and certification.
- Provide a bounded, documented compilation-cache maintenance policy rather than unbounded silent growth.

### 11.5 Parallelism order

1. Independent validation/optimization cases.
2. Radial surfaces.
3. Electric-field/collisionality scan points.
4. Multiple right-hand sides with one factorization.
5. Preconditioner construction and independent line/subsystem batches.
6. Multi-device single-solve sharding only when a profile shows enough arithmetic and acceptable communication.
7. Multi-host single-solve scaling only after the previous levels and with explicit communication analysis.

The first GPU target is correctness and complete profiling. The second is near-linear throughput for independent work. A distributed-array demo without end-to-end speedup is not a milestone.

### 11.6 Official benchmark suite

Record the exact laptop and GPU models, then maintain a small benchmark matrix:

- small tokamak prescribed field;
- medium stellarator monoenergetic;
- large structured HSX-like case;
- full-FP two-species tokamak;
- full-FP W7-X prescribed field;
- two-surface ambipolar replay;
- five-surface profile after admission;
- gradient/adjoint case;
- repeated optimization-like sequence.

For every PR touching performance-critical code:

- no unexplained >10% median warm-runtime regression;
- no unexplained >10% peak-memory regression;
- no accuracy or residual regression;
- compilation and total-workflow effects reported, not only a kernel microbenchmark.

### 11.7 Immediate hard-case conclusion

The broad fixed-grid all-root W7-X route is currently a known operational no-go. Do not optimize it by adding more one-off campaign scripts. The production strategy should combine:

- cheap bounded discovery models/grids;
- branch continuity and previous-surface information;
- explicit candidate intervals;
- real admitted-grid endpoint and refinement solves;
- independent checks for tangential roots or missed branches;
- honest interval-scoped claims when global completeness is not established.

---

## 12. Capability promotion matrix

Use the machine-readable registry as the source of truth. The table below gives the current direction.

| Capability | Current assessment | Required next gate |
| --- | --- | --- |
| SFINCS namelist/HDF5 compatibility | Stable candidate | Consolidate under compatibility namespace; preserve installed-artifact tests. |
| Analytic prescribed-field native profile | Stable candidate | Complete high-level API/docs/example and proof matrix. |
| VMEC prescribed-field native profile | Stable candidate | Convergence certificate, public example, Result metadata audit. |
| Boozer prescribed-field native profile | Stable candidate | Asymmetric conventions, independent geometry tests, public example. |
| Monoenergetic coefficients | SFINCS-compatible stable candidate; native workflow incomplete | Native Case/Result/CLI, variational certificate, MONKES/YANCC matrix. |
| Full local Fokker-Planck | Compatibility path strongly validated; native workflow incomplete | Native API, conservation proof suite, broader independent regime matrix. |
| Native ambipolar roots | `validated_limited` | Complete API, tangential-root strategy, second device family, full-FP campaign. |
| Radial branch continuation/events | `validated_limited` | Manufactured bifurcation tests and continuous event localization policy. |
| W7-X admitted roots | Explicit seeded intervals only | Extend surface-local evidence carefully; never relabel as global all-root without proof. |
| Bootstrap/parallel current | Compatibility path stable candidate; admitted W7-X profile not converged | Separate theta/zeta/pitch/speed current convergence and independent formula/code comparison. |
| Phi1/quasineutrality | Compatibility path candidate | Native workflow, proof/convergence suite, KNOSOS/SFINCS comparisons. |
| Tangential magnetic drifts | Compatibility path candidate | Native workflow, resonance/validity tests, KNOSOS/SFINCS comparisons. |
| Momentum correction | Experimental/validated limited | Conservation and independent bootstrap/flow comparisons. |
| Impurity transport | Experimental/validated limited | Multispecies full-FP cases, screening/asymptote and independent comparisons. |
| Bounce-averaged reduced model | Experimental surrogate | KNOSOS/literature matrix, validity selector, optimization demonstration. |
| Variational bounds | Research capability | Mathematical proof tests and user-facing convergence certificate. |
| Differentiable optimization | Validated limited | Stable objective API, Taylor tests, VMEX end-to-end benchmark. |
| CPU batching/sharding | Partial | Public policy and scaling evidence. |
| GPU execution | Partial | Maintained office-GPU gate and memory profiles. |
| Multi-host/distributed | Experimental | End-to-end scaling and failure/restart policy; no import-time initialization. |
| Collocation/multigrid alternate route | Experimental | Decide promote or remove after measured regime matrix. |

---

## 13. Ordered implementation phases

Each phase ends with a reviewable repository state. Do not skip acceptance gates because a later phase seems more scientifically interesting.

### Phase A — Replace the plan and freeze the current state

**Goal:** make this document and an exact current inventory the handoff ground truth.

Tasks:

- replace the old `plan.md` with this file;
- record current commit, open PR/issues, branch protection, CI jobs, exact coverage, test count, source/test/example/doc file counts, production LOC, package sizes, and full-clone size;
- identify every current validation artifact and its generating script/test;
- map every public symbol and CLI command to native, compatibility, expert, experimental, or internal status;
- record exact laptop and GPU hardware;
- run the complete current tests and warning-clean docs without changing tolerances;
- add only a compact current-state machine-readable inventory if useful.

Acceptance:

- no runtime or physics change;
- one authoritative plan;
- current metrics recorded;
- every known failing/flaky test classified;
- hosted checks green.

### Phase B — Consolidate evidence and tests before adding campaigns

**Goal:** stop evidence and test-file growth while preserving all useful conclusions.

Tasks:

- create the common validation registry and generic audit runner;
- merge the many W7-X campaign summaries into a small number of capability summaries with raw evidence in release assets;
- merge one-purpose audit scripts into parameterized runners;
- merge one-purpose test files into behavior suites;
- remove dated test directories and historical source-shape policy tests;
- preserve checksums, commands, external commits, negative results, and limitations;
- establish coverage by package/module and an uncovered-lines report.

Acceptance:

- fewer validation files, test files, and test LOC than before;
- no lost scientific assertion;
- identical registry gate results;
- PR CI no slower;
- coverage does not decrease.

### Phase C — Align packaging, runtime, and source layout with DKX 3

**Goal:** establish a clean installed-package boundary before further API work.

Tasks:

- raise Python floor to 3.11 and remove `tomli`/3.10 CI;
- move to `src/dkx`;
- make `import dkx` inert;
- introduce explicit runtime configuration;
- retain isolated wheel/sdist and external-data tests;
- add Rich and MyST/Furo as appropriate core/docs dependencies;
- audit dependency lower bounds and remove no-op extras;
- ensure coverage runs against the installed artifact;
- enforce current-tree, wheel, sdist, and installed-file size gates.

Acceptance:

- `import dkx` does not import JAX or mutate the environment;
- wheel/sdist work from an empty directory;
- all public imports resolve from `site-packages`;
- no scientific output change;
- package files/LOC do not grow without deletion;
- Python 3.11 is the only floor job.

### Phase D — Complete and simplify the native API

**Goal:** one native high-level API and a confined compatibility layer.

Tasks:

- resolve `run` naming/module ambiguity;
- make `Case` and `Result` the only normal high-level contracts;
- move SFINCS-facing entry points under compatibility namespace;
- implement or remove unsupported declared workflow values;
- add native monoenergetic, transport-matrix, scan, convergence, and objective workflows;
- introduce reusable runtime state and structured progress events;
- complete physical-unit metadata and result schema;
- simplify CLI to the declared command set with Rich output;
- deprecate/delete DKX 2 facade types and duplicate paths.

Acceptance:

- all canonical examples use native API;
- a user can perform each stable workflow from Python and TOML;
- legacy namelist/HDF5 round trips still pass;
- fewer public symbols and fewer orchestration LOC;
- API docs and type checks pass.

### Phase E — Build the full proof-oriented verification suite

**Goal:** make every stable mathematical, physical, and numerical component directly defensible.

Tasks:

- implement the proof matrix in section 7;
- add manufactured solutions for derivative/operator combinations;
- add collision conservation/entropy tests;
- add root/bifurcation/tangency tests;
- add solver/adjoint/route invariance tests;
- add dimensional/normalization round trips;
- consolidate cross-code runners;
- raise coverage gates through `90 -> 95 -> 98 -> 100` while deleting dead code;
- add mutation testing for pure critical logic in nightly CI.

Acceptance:

- 100% line/branch coverage of stable reachable code, or an explicitly approved temporary gap with owner and deadline;
- every stable capability linked to proof, convergence, and validation tests;
- no coverage-only tests;
- PR suite remains under the wall-time contract.

### Phase F — Rebuild documentation, examples, and plotting

**Goal:** make the native product understandable and attractive without reading source or project history.

Tasks:

- migrate narrative docs to MyST/Furo and the declared information architecture;
- rewrite landing page, installation, and first profile around native TOML/Python;
- rewrite theory from first principles with citations and source/test links;
- generate schema, CLI, and API reference;
- build the nine-example ladder;
- consolidate plotting and implement the required figure families;
- replace or compress repository media;
- move large reproducibility artifacts to releases;
- shorten README and correct documentation identity/links.

Acceptance:

- a clean installed wheel executes every documented fast example;
- all equations, inputs, outputs, models, algorithms, and limitations are documented;
- no stale SFINCS-first quickstart;
- examples satisfy the no-argparse/no-main contract;
- plots pass semantic tests and release visual review;
- warning-clean, link-clean docs.

### Phase G — Concentrated performance and sharding campaign

**Goal:** reduce complete-workflow runtime and memory at matched accuracy.

Tasks:

- establish official CPU/GPU baselines with the common schema;
- profile hard routes and remove recompilation/captured constants/transfers;
- formalize reusable analysis/factor/preconditioner state;
- batch RHS and root/profile workloads;
- shard independent surfaces/fields/collisionalities across devices;
- tune Krylov memory, true-residual refresh, recycling, and preconditioner lag/reuse;
- evaluate structured selected-tail and mixed-precision refinement routes;
- decide the future of multigrid/collocation based on production evidence;
- investigate single-solve sharding only after independent-work scaling.

Acceptance:

- every promoted optimization has an end-to-end benchmark;
- no accuracy/residual regression;
- official benchmark artifacts reproduce;
- memory guard predicts observed peaks conservatively;
- GPU route demonstrates real complete-workflow value;
- experimental routes that fail promotion are removed or moved out of stable source.

### Phase H — Promote the flagship scientific workflows

**Goal:** deliver a credible whole-profile neoclassical product, not only infrastructure.

Ordered campaigns:

1. complete a five-surface interval-scoped W7-X profile with independently justified surface-local intervals;
2. perform a separate bootstrap/parallel-current convergence campaign;
3. add full-FP ambipolar roots and compare to SFINCS/YANCC as appropriate;
4. add native Phi1 and tangential-drift profile/root cases with KNOSOS/SFINCS comparisons;
5. validate a second stellarator family and a tokamak profile;
6. add an experimental/published-device comparison with uncertainty;
7. promote monoenergetic database and reduced-model selection;
8. complete impurity and momentum-correction evidence.

Acceptance for each campaign:

- registered model/domain and limitation;
- complete proof/convergence/cross-code chain;
- native API/TOML/example/docs/plot;
- runtime and memory within declared envelope;
- no unqualified global-root or current-convergence claim.

### Phase I — Coupling and DKX 3 release

**Goal:** make DKX usable as a dependable component of the UW plasma ecosystem and broader community.

Tasks:

- stabilize VMEX geometry/profile/objective adapter;
- define small protocols for geometry and profile providers;
- demonstrate gradient-verified kinetic optimization;
- add resume/checkpoint/provenance behavior for long workflows;
- publish PyPI artifact and conda-forge feedstock;
- add DOI/Zenodo archive and `CITATION.cff`;
- enforce branch protection and release checklist;
- coordinate a history rewrite only after current-tree margin makes the full clone safely below 20 MiB;
- publish the capability/evidence bundle with the release.

Acceptance:

- clean install and first result from PyPI;
- official laptop and GPU benchmarks;
- full capability matrix with no ambiguous statuses;
- fresh full clone, wheel, sdist, and installed DKX-owned files each below 20 MiB;
- documentation and examples match the released API;
- no known high-severity correctness, packaging, or provenance gap.

---

## 14. Recommended pull-request sequence

Do not implement all phases in one branch. The following sequence is deliberately ordered to reduce risk and repository size.

1. **`plan/current-state-handoff`**  
   Replace `plan.md`; record exact current inventory and hardware; no runtime changes.

2. **`validation/consolidate-registry`**  
   Merge campaign registries/runners/tests; reduce files and LOC; preserve evidence.

3. **`package/python311-src-runtime`**  
   Python 3.11, `src/` layout, inert import, explicit runtime configuration, installed coverage.

4. **`api/native-contract`**  
   Unify `Case`/`Result`/`run`; isolate SFINCS compatibility; delete duplicate facade types.

5. **`api/native-workflows`**  
   Native monoenergetic, transport matrix, scans, convergence, and structured progress.

6. **`tests/math-numerics-proof`**  
   Derivative/quadrature/block/root/MMS proof suites; first coverage ratchet.

7. **`tests/physics-proof`**  
   Collision invariants, operator identities, moments, limits, units, derivatives; next coverage ratchet.

8. **`docs/native-rebuild`**  
   MyST/Furo structure, native quickstart, schema/API generation, README contraction.

9. **`examples/native-ladder`**  
   Canonical scripts/TOMLs; remove overlapping example trees.

10. **`plotting/semantic-publication`**  
    Consolidated plotting, root/branch/convergence figures, semantic tests.

11. **`performance/reuse-batching`**  
    Reuse state, multi-RHS/root/profile batching, compilation and memory improvements.

12. **`performance/sharding-gpu`**  
    Independent-work sharding and maintained GPU evidence.

13. **`science/profile-bootstrap`**  
    Five-surface interval profile plus separate current convergence.

14. **`science/fullfp-phi1-drifts`**  
    Full-FP ambipolar, Phi1, tangential drifts, second device family.

15. **`integration/vmex-dkx3-release`**  
    Stable coupling, release artifacts, DOI, conda-forge, final clone-size action.

A PR may be split when reviewability demands it, but do not create a chain of tiny PRs that leaves two active owners for weeks.

---

## 15. Definition of done

DKX 3 is complete only when all of the following are true.

### Code and architecture

- one native high-level API;
- permanent compatibility namespace;
- inert import;
- shallow `src/dkx` layout;
- no duplicate execution stack;
- generic numerics in SOLVAX;
- approximately 35 or fewer production Python files and 45k or fewer production lines unless explicitly justified;
- concise module/function docstrings documenting units, assumptions, shapes, and references;
- comments explain non-obvious physics/numerics, not restate code.

### Tests and scientific credibility

- 100% line/branch coverage of stable reachable code;
- proof tests for all mathematical kernels;
- conservation, symmetry, entropy, limit, and manufactured-solution tests for stable physics;
- convergence and uncertainty certificates;
- independent code/literature validation by capability;
- derivative and transpose-solve certification;
- CPU/GPU and solver-route equivalence;
- every fixed bug has a minimal regression test.

### User experience

- `pip install dkx` and one TOML command produce a meaningful result;
- Rich progress prevents apparent hangs;
- physical units throughout the native interface;
- NetCDF result contains complete provenance and evidence;
- nine or fewer clear examples, each printing, saving, and plotting;
- polished publication-ready plot families;
- complete tutorials, how-to guides, theory, reference, validation, and limitations.

### Performance

- official laptop and GPU benchmark suite;
- compile/cold/warm/total timing separated;
- bounded memory with conservative preflight;
- reuse across scans, roots, profiles, and adjoints;
- efficient independent-work CPU/GPU sharding;
- no promoted route with an unexplained runtime/memory regression;
- hard cases have an honest operational scope.

### Distribution and sustainability

- wheel, sdist, installed DKX files, and fresh full clone below 20 MiB;
- PyPI, conda-forge, DOI, citation metadata;
- branch protection and release gates enforced;
- FAIR software metadata and persistent identifiers;
- capability registry and reproducibility bundle released.

---

## 16. Current checkpoint and immediate next action

### Current checkpoint

Recorded on 2026-08-30 at `main = c958a947505ba31f4e6f80c6fde983ab1db05b71`, release version `2.3.1`. Every number below is measured; the machine-readable form is `validation/baseline.toml` and the measurement host is `official-laptop-cpu` in `validation/hardware.toml`.

**Hosted state.** 21 checks, all green. CI run `33292524627` completes in 525 s; the docs build in 54 s. No open pull requests and no open issues at audit. Branch protection was advisory at audit and is now enforced: `main` takes changes only through a pull request, one approving review is required, and eight checks are required. Admin bypass and force push are deliberately retained so the sole maintainer can merge without self-approval.

**Size.** Fresh clone from origin 36.46 MiB (Git object store 22.35 MiB, working tree 14.11 MiB); wheel 0.56 MiB; sdist 0.91 MiB; installed DKX-owned files 4.67 MiB; `twine check` passes. Only the full clone misses the 20 MiB target, and its Git object store alone already exceeds it. Cloning the local development checkout instead of the remote reports about 88 MiB; that number is an artifact of extra local refs and is not the contract figure.

**Source.** 63 production Python files, 51,230 lines at audit: 28 files and 6,230 lines above the section 6.2 budgets. 15 files exceed 1,200 lines and 7 exceed 1,800. Source now lives in `src/dkx`, so a test run from the repository root imports the installed package rather than the working-directory copy. `import dkx` was importing the JAX backend, mutating seven environment variables, creating `~/.cache/dkx/jax_compilation_cache`, and costing 0.46 s against 0.04 s for a bare interpreter; it is now inert and costs 0.09 s, importing numpy and nothing heavier. The runtime moved to `dkx.runtime.configure()`, which the CLI bootstrap and every module that imports the JAX backend call, so the numerics are unchanged. numpy is the one deliberate exception: `dkx.runtime` runs the numpy-version guard at its own import, because deferring it would let the package import cleanly on numpy 1.x and then die inside ml_dtypes with a traceback naming neither DKX nor numpy.

**Public surface.** 46 names in `__all__`, 56 resolvable public names. `Result` and `RESULT_SCHEMA_VERSION` are reachable but absent from `__all__`; `run` is in `__all__` but resolves lazily to a callable module (`dkx.run._CallableModule`); ten internal names leak, including `os` and `tempfile`. The canonical Python workflow of section 5.1 does execute: `Case.from_file(...) -> dkx.run(case) -> Result` with `print_summary`, `save`, and `plot`. Every other public `run_*` entry point takes a SFINCS namelist. The CLI exposes 16 subcommands. `run` and `inspect` joined `validate` and `schema` as native commands, so `dkx run CASE --out RESULT.nc` executes a native `Case` and `dkx inspect RESULT.nc` reads one back; the other twelve still take a SFINCS namelist. `scan`, `roots`, `converge`, `plot`, `convert`, `compare`, and `doctor` from the section 5.6 target set do not exist yet.

**Tests and coverage.** 1,721 tests collected; the pull-request gate selects 1,700 and defers 21 `slow` tests. A clean-environment local run is green end to end: the gate selection gives 1,675 passed, 27 skipped, 0 failed in 578 s at `-n 4`, and the deferred `slow` selection gives 17 passed, 6 skipped, 0 failed in 160 s. The two failures the 2026-08-28 baseline recorded on the Mac mini do not reproduce here; one of them is a wall-clock threshold and remains a flakiness risk on slower hosts. Hosted CI measures lines only and reports **90%** against an **80%** gate across 13 shards. Measuring branches locally on the same selection gives **90.47% line, 78.28% branch, 87.64% combined** over 20,266 statements and 6,124 branches. The branch number, not the hosted line number, is the real distance to the section 7.1 target. The lowest-covered modules are `dkx/bootstrap.py` (56.4%), `dkx/validation/release.py` (63.8%), and `dkx/representative.py` (76.0%).

**Evidence.** 20 campaign summaries under `validation/`. As of the first Phase B slice they are indexed by `validation/registry.toml` and checked by one runner, `dkx.validation.registry`, from one test module, `tests/test_validation.py`; the nineteen per-campaign modules are gone and every entry names a reproducible command. What is not yet done: the 20 summaries are still 20 files rather than a few capability summaries, 17 of the 30 `tools/paper_benchmarks` scripts still exist to audit exactly one of them, the capability registry still carries the 2026-08-28 baseline commit, and `tests/` still holds three dated campaign directories that `dkx/validation/release.py` reads.

**Docs and examples.** 172 documentation files, 39 top-level navigation entries including roadmap, research-lane, release-checklist, and upstream pages; `sphinx-rtd-theme` with an `alabaster` fallback; the landing page is still namelist-first and the documentation URL still says `sfincs-jax`. The docs build is warning-clean under `-W`. 202 example files across 12 directories, 113 of them SFINCS decks; 5 native TOML cases and **0** paired native Python scripts.

**Dependencies.** The floor is now `>=3.11`, the `tomli` marker and the Python-3.10 CI job are gone, `rich` is a core dependency, `myst-parser` and `furo` replace `sphinx-rtd-theme` in the docs extra, and the no-op `structured` extra is removed. Phase C is complete: the coverage shards install the package non-editable and measure what a user gets.

**Science.** Unchanged by this audit: two admitted-grid W7-X seeded roots near `12.681640625` and `11.533203125 kV/m`, explicit interval-only scope; high-zeta parallel-current convergence is not admitted; the broad uniform all-root route remains an operational no-go.

**Principal engineering blockers.** Duplicated native/legacy public surfaces, import-time runtime mutation, an 80% line-only coverage gate with no branch gate, legacy-first docs and examples, no native CLI run path, and a full-clone size that cannot pass without a history rewrite. Evidence fragmentation is partly addressed: the registry and its runner exist, the summary and audit-script counts are not yet reduced.

### Immediate next action

`plan/current-state-handoff` (PR #107) replaced `plan.md` and recorded this inventory; `validation/consolidate-registry` is the first Phase B slice and adds the common registry and runner. The next two steps, in order:

1. complete the section 7.4 physics proofs that are genuinely missing: moments of manufactured distributions, the Spitzer conductivity limit, and the Shaing-Callen collisionless bootstrap limit. Survey what exists before writing: section 3.2 understated the current proof coverage, and three separate attempts to add proofs found them already present;
2. finish Phase B by moving the dated `tests/` campaign directories and raw suite outputs to release assets, rewiring `dkx/validation/release.py` to read them from the registry, and reducing the 20 summaries and 17 single-purpose audit scripts;
3. complete the section 5.6 command set: `scan`, `roots`, `converge`, `plot`, `convert`, `compare`, `doctor`.

A GPU lane is blocked on a decision, not on work. The office host is reachable and its inventory is recorded, but the only interpreter on it is the system Python 3.10 and DKX now requires 3.11. Standing the lane up means installing a 3.11+ toolchain into the user home on a shared lab workstation, which the maintainer should agree to first.

Do not begin another root-resolution campaign before step 1 lands.

The first implementation success metric is not another W7-X number. It is a smaller repository with fewer test/evidence files, preserved scientific assertions, a measured coverage baseline, and a clear route to the DKX 3 installed-package/API refactor.

### Compact execution log

Keep only one row per merged PR or consequential failed hypothesis.

| Date | Change | Evidence | Next |
| --- | --- | --- | --- |
| 2026-08-30 | Full current-state audit and replacement handoff plan prepared at `c958a947`. | Reviewed current and historical commits/PRs, plan ledger, source owners, native API, CI, docs, examples, tests/evidence, SFINCS/PETSc/MUMPS/SuperLU_DIST/MONKES/YANCC/KNOSOS methods, JAX guidance, and research-software V&V practices. | Merge plan-only current-state PR, then consolidate evidence/tests. |
| 2026-08-30 | Phase A executed: `plan.md` replaced and the exact current-state inventory recorded at `c958a947`. | `validation/baseline.toml` (repository, branch protection, source, import side effects, tests, coverage, public API, CLI, examples, docs, dependencies, package size, CI, and the 20 validation artifacts with their generators); `validation/hardware.toml` (named laptop observed, office GPU unreachable). Local run green end to end: 1692 passed / 33 skipped / 0 failed; warning-clean `sphinx -W` build. | Merge, enforce required status checks, then start Phase B. |
| 2026-08-30 | Branch protection enforced on `main`: one required approving review, eight required status checks, pull requests only. Admin bypass and force push are retained so the maintainer can merge without self-approval. | GitHub branch-protection API; `validation/baseline.toml` `[repository.branch_protection]`. | Closes GAP-A-002. |
| 2026-08-30 | Phase E started, and a survey corrected the audit. Speed-grid quadrature and derivative matrices proved in `tests/test_math.py`; bracket search, root classification and angular stencils proved in `tests/test_numerics.py`. | 60 proofs against closed forms rather than recorded outputs. The bracket proofs include the two negative results that entitle every admitted W7-X claim to its interval-only scope: a tangential root and an even number of crossings between samples are both invisible to sign sampling. | Section 3.2 understated existing proof coverage. Before writing anything, these were found already proven and were **not** duplicated: structured-solver identities in `test_solve.py` and `test_drift_kinetic.py` (dense referee, truncation parity, batch bit-identity, block-extraction reconstruction, preconditioner, gradient-versus-finite-difference); collision invariants in `test_collision_physics_gates.py` (PAS eigenvalues, Fokker-Planck Maxwellian null, interspecies conservation, Rosenbluth versus quadpack); Onsager symmetry `D13* = -D31*` in `test_monoenergetic_database.py` and `test_paper_benchmark_monoenergetic.py`. Remaining section 7.4 gaps are moments of manufactured distributions, the Spitzer conductivity limit, and the Shaing-Callen collisionless bootstrap limit. |
| 2026-08-30 | Frozen source-shape policy retired per section 7.2: `test_source_tree_consolidation.py` 712 -> 304 lines, 33 -> 8 tests. | `test_package_tree_has_no_tracked_generated_or_large_runtime_outputs` had been vacuously passing for three PRs because `git ls-files dkx` returns nothing under the src layout; it checks 67 files again. `core_slim_inventory.json` was deleted and restored: a dated research-lane record cites it as evidence, and deleting a refactor veto is not the same act as deleting cited evidence. | Section 5.6 still owes `scan`, `roots`, `converge`, `plot`, `convert`, `compare`, `doctor`. |
| 2026-08-30 | Phase C closed: the coverage shards install the package non-editable, so coverage measures the installed artifact. | Running the suite against a wheel first produced 8 failures and 4 collection errors, all from shipped code assuming it lives in a checkout. `dkx.validation.release` fell back to `parents[3]`, which resolves to site-packages and sent `git ls-files` into a directory that is not a repository; it now falls back to the working directory. `tests/test_validation.py` asked the package to locate the checkout instead of supplying its own root. 1773 passed / 27 skipped / 0 failed against the installed wheel, the same count as the editable run. | Phase B remainder: collapse the four hand-synced path indexes. Then the rest of the section 5.6 CLI. |
| 2026-08-30 | Phase D, first slice: `dkx run CASE` and `dkx inspect RESULT` give the native `Case` a CLI path, with Rich output. | 1773 passed / 27 skipped / 0 failed; line coverage 90.59% and branch 78.49%. `inspect` deliberately prints no units column: a native Result carries no per-variable units metadata, and an empty column would read as present-and-blank rather than absent. | Remaining section 5.6 commands: `scan`, `roots`, `converge`, `plot`, `convert`, `compare`, `doctor`. Solver kernels still print to stdout instead of emitting progress events. |
| 2026-08-30 | Phase C, second slice: the package moved to `src/dkx`. | Wheel unchanged in shape -- top level is still `dkx/`, no `src/` prefix leaks in, package data still ships. The layout is what stops a test run from the repository root importing the working-directory copy instead of the installed package. | Phase C remainder: measure coverage against the installed artifact. Then Phase D, starting with the native `Case` execution path the CLI still lacks. |
| 2026-08-30 | Phase C, first slice: Python floor raised to 3.11, `dkx.runtime.configure()` introduced, and `import dkx` made inert. | Import loads numpy only -- no JAX backend -- mutates no environment variable, creates no directory, and costs 0.09 s against 0.46 s before; `import dkx.run` still applies all seven variables and float64, so numerics are unchanged. 1762 passed / 27 skipped / 0 failed; line coverage 90.56% and branch 78.48% against 90.54% and 78.48% on `main`. `tomli`, the 3.10 CI job, and the no-op `structured` extra are gone; `rich` is core and MyST/Furo replace `sphinx-rtd-theme`. | Phase C second slice: `src/dkx` layout. |
| 2026-08-30 | Phase B, first slice: `validation/registry.toml` plus the one generic runner `dkx.validation.registry` replace nineteen per-campaign test modules with `tests/test_validation.py`. Added the missing `audit_w7x_admitted_grid_uniform_probe_no_go.py` so all 20 entries name a command. | 20/20 registry entries pass, including 8 declared corruption probes; test modules 182 -> 163; test lines 42,713 -> 42,284; the new runner is 98% covered. | Second Phase B slice: move the dated `tests/` campaign directories and raw suite outputs to release assets, then reduce the summary count itself. |

Future agents append concise rows here and update **Current checkpoint**. Detailed command logs belong in PR descriptions or registered artifacts, not in this file.

---

## 17. Primary references for implementation decisions

### DKX and comparison-code sources

- DKX repository and merged PR history: <https://github.com/uwplasma/DKX>
- SFINCS source, manual, and technical notes: <https://github.com/landreman/sfincs>
- MONKES source: <https://github.com/JavierEscoto/MONKES>
- YANCC source: <https://github.com/f0uriest/yancc>
- KNOSOS source: <https://github.com/joseluisvelasco/KNOSOS>
- PETSc KSP manual: <https://petsc.org/release/manual/ksp/>
- MUMPS documentation: <https://mumps-solver.org/>
- SuperLU_DIST: <https://github.com/xiaoyeli/superlu_dist>

### Core neoclassical references

- Landreman, Smith, Mollén, and Helander, *Physics of Plasmas* **21**, 042503 (2014).
- Landreman and Ernst, *Journal of Computational Physics* **243**, 130 (2013).
- Hirshman et al., *Physics of Fluids* **29**, 2951 (1986).
- van Rij and Hirshman, *Physics of Fluids B* **1**, 563 (1989).
- Shaing and Callen, *Physics of Fluids* **26**, 3315 (1983).
- Beidler et al., *Nuclear Fusion* **51**, 076001 (2011).
- Escoto et al., “MONKES: a fast neoclassical code for the evaluation of monoenergetic transport coefficients,” arXiv:2312.12248.
- Velasco et al., “KNOSOS: A fast orbit-averaging neoclassical code for stellarator geometry,” *Journal of Computational Physics* **418**, 109512 (2020).
- Conlin and Landreman, “yancc: A GPU-accelerated, differentiable solver for neoclassical transport in tokamaks and stellarators,” arXiv:2607.20861.

### Verification, performance, and software practice

- Salari and Knupp, *Code Verification by the Method of Manufactured Solutions*, SAND2000-1444.
- NASA-STD-7009B, *Standard for Models and Simulations*.
- FAIR4RS Working Group, *FAIR Principles for Research Software*, DOI 10.15497/RDA00068.
- JAX benchmarking and profiling: <https://docs.jax.dev/en/latest/benchmarking.html> and <https://docs.jax.dev/en/latest/profiling.html>
- JAX persistent compilation cache: <https://docs.jax.dev/en/latest/persistent_compilation_cache.html>
- JAX buffer donation and sharding documentation: <https://docs.jax.dev/en/latest/buffer_donation.html> and <https://docs.jax.dev/en/latest/parallel.html>
- Diátaxis documentation system: <https://diataxis.fr/>
- MyST Parser: <https://myst-parser.readthedocs.io/>
- Furo: <https://pradyunsg.me/furo/>

