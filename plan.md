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

- **95% line and branch coverage of stable reachable Python code** (maintainer decision, 2026-08-31, revised down from 100%);
- exclusions only for generated version data, platform-unavailable external callbacks, or defensive impossibilities, each with a documented reason;
- no stable physics branch excluded merely because it is expensive;
- performance and large external comparisons separated from the coverage denominator when their logic is already exercised by bounded fixtures.

Raise the gate through measured ratchets while deleting dead code:

```text
80 -> 90 -> 95
```

Do not add hundreds of shallow tests to reach the number. First remove unreachable, duplicate, experimental, and compatibility-only code from the stable denominator.

The end state was 100% until 2026-08-31, when the maintainer set it to 95%.
The reason to record: the last few percent of branch coverage in this codebase
is concentrated in defensive impossibilities, platform-unavailable callbacks,
and compatibility paths that only a second Fortran build would exercise.
Chasing those to 100% buys shallow tests, which the paragraph above forbids,
and pressure to delete error handling that exists for good reasons. 95% with
the deletion discipline intact is the stronger target.

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

Measured 2026-08-31 on W7-X standard, `geometryScheme=11`, `r_N=0.5`, PAS with
DKES-like trajectories, GCROT to a true relative residual of `1e-10`
(`validation/benchmarks/collocation_multigrid_h_independence_2026-08-31.json`):

| unknowns | iterations | cycle rate | peak GB |
| ---: | ---: | ---: | ---: |
| 48,384 | 94 | 0.487 | 0.37 |
| 114,688 | 137 | 0.537 | 0.51 |
| 224,000 | 162 | 0.512 | 0.87 |
| 387,072 | 179 | 0.504 | 0.87 |
| 614,656 | 230 | 0.493 | 1.08 |

The result is split, and the split is the point. The V-cycle achieves what it
was built for: the residual reduction per cycle is flat to 10% across a 12.7x
range in unknowns. The outer iteration count is not flat -- 94 to 230, a 2.45x
growth -- so by the criterion the benchmark's own docstring sets ("flat counts
*and* a resolution-independent cycle rate"), the route is not h-independent.
The growth is sublinear, not proportional.

What is real is memory. Every rung converged, including 614,656 unknowns at
1.08 GB, past the ~488,000 where the classical route is recorded as exhausting
memory. So the honest position is neither promote nor remove: the route
reaches sizes the structured direct route cannot, and does not scale the way
the experiment set out to show.

Do not promote because a small case runs, and do not remove code that reaches
sizes nothing else does. Promotion still requires:

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
- raise coverage gates through `90 -> 95` while deleting dead code;
- add mutation testing for pure critical logic in nightly CI.

Acceptance:

- 95% line/branch coverage of stable reachable code, or an explicitly approved temporary gap with owner and deadline;
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

- 95% line/branch coverage of stable reachable code;
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

Recorded on 2026-09-01 at `main = 038fbd6a756c401994282f2073fa64302b529cb3`, release version `2.3.1`. This replaces the 2026-08-30 checkpoint, which had gone stale on almost every line -- it recorded 16 CLI subcommands and listed `scan`, `roots`, `converge`, `plot`, `compare` and `doctor` as not existing, all of which now do.

**Hosted state.** No open issues. GitHub's runner queue has been the binding constraint on CI turnaround rather than the suite itself, so recent merges were made on full local verification with CI following. `main` history was rewritten on 2026-08-31: superseded binary blobs stripped, all 2,537 commits and 30 tags preserved, tip tree byte-identical. Anyone holding an older clone must re-clone or reset.

**Size.** Fresh clone 15 MiB, Git pack 13.94 MiB -- **inside** the 20 MiB target for the first time, where the previous checkpoint recorded 36.46 MiB with the object store alone over budget.

**Source.** 60 production Python files, 46,827 lines: 25 files and 1,827 lines above the section 6.2 budgets, against 28 files and 6,230 lines at the last checkpoint. 14 files exceed 1,200 lines and 7 exceed 1,800. The line budget is now close; the file budget is not, and will not be reached by merging small modules -- `units.py` into `constants.py` saves one file of the 25 for nineteen files of import churn. It needs either genuine dead-code deletion or the explicit justification section 15 allows.

**Public surface.** 50 names in `dkx.__all__`. `dkx.workflows` exports `converge`, `geometry_adapters`, `optimization` and `scan`. The CLI advertises **11** commands and registers 22: the twelve SFINCS commands moved under `dkx sfincs` and are kept as hidden top-level aliases so existing scripts keep working. Of the section 5.6 target set, ten of eleven exist -- `doctor`, `schema`, `validate`, `run`, `roots`, `converge`, `inspect`, `compare`, `plot`, `scan` -- and only `convert` is missing.

**Tests and coverage.** 2,067 tests collected across 176 files, up from 1,721 at the last checkpoint. The last full measurement was **90.6% line, 78.5% branch**; the target is now 95% for both, not 100%. A fresh measurement is in progress and this line should be replaced with it. Note that the branch figure, not the hosted line figure, is the real distance to the target.

**Evidence.** The registry and single runner are in place. `validation/benchmarks/` now carries the collocation/multigrid h-independence ladder, which closed a promote-or-remove question Route S5 had left open since this plan was written. What is still not done: the 20 summaries are still 20 files rather than a few capability summaries, and `tests/` still holds dated campaign directories.

**Docs and examples.** 172 documentation files, 41 `.rst` pages, building warning-clean under `-W`. The tier vocabulary is retired from prose everywhere, surviving only in identifiers with one note in `numerics.rst` mapping the two. 239 example files, 5 native TOML cases -- still **0** paired native Python scripts, unchanged from the last checkpoint and a Phase F gap.

**Science.** Unchanged: two admitted-grid W7-X seeded roots near `12.681640625` and `11.533203125 kV/m`, interval-only scope; high-zeta parallel-current convergence not admitted; the broad uniform all-root route remains an operational no-go. New this cycle: the shipped `analytic_tokamak_profile` deck is measurably unconverged (`theta` 9 to 14 moves the particle flux 1187%), and axis coupling makes single-axis refinement misleading on it -- `theta` looks settled to 0.2% at `pitch = 8` and moves 74% at `pitch = 40`.

### Immediate next action

Phases A and C are closed. B is substantially done. D and E are in flight. F through I are barely started, and they are the bulk of the remaining work.

The next steps, in order:

1. **Coverage is measured and deferred.** 2026-09-01: **91.55% line, 79.77% branch** over 18,074 statements and 5,234 branches (1,994 passed, 1 failed, 65 minutes sequential on the office host). The maintainer has deferred the 95% gate as not urgent.

   Record why the strategy that preceded this measurement was wrong, so it is not retried: this entry previously said the uncovered-line report would drive a *dead-code sweep* serving both the coverage gate and the file budget. The report says otherwise. The worst modules are all live -- `magnetic_geometry` (180 missing), `representative` (131), `cli` (118), `solve` (87), `bootstrap` (60, and the worst percentage at 57.4%). `bootstrap.py` was checked directly: it is a live optimization objective used by three examples and two test modules, and its gaps are real paths needing `vmex` and equilibria, clustered in only three runs of six lines or more. The one unpromoted module in that list is `multigrid.py`, already kept on the Route S5 gate evidence.

   So closing 79.8% to 95% branch means **writing tests for live code**, not deleting it, and it does nothing for the file budget. The two goals are independent. Treat them separately from here.

2. **Decide the file budget honestly.** 60 files against a budget of about 35. Merging small cohesive modules does not close that gap -- the largest single saving available is one file for nineteen files of import churn. Either the dead-code sweep in step 1 produces real deletions, or section 15 needs the explicit justification it already allows. Do not manufacture 2,000-line grab-bag modules to hit a number.

3. **Finish Phase B**: move the dated `tests/` campaign directories and raw suite outputs to release assets, rewire the release reader to the registry, and reduce the 20 summaries and the single-purpose audit scripts.

4. **`dkx convert`**, the last of the section 5.6 set, and then Phase D's remaining native workflows (monoenergetic, transport-matrix, objective) and the physical-unit metadata that `dkx inspect` still cannot print.

The GPU lane is live: `~/venvs/dkx-gpu` on the office host runs jax 0.10.2 and reports both CUDA devices.

Do not begin another root-resolution campaign before the coverage sweep lands.

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
| 2026-09-01 | Phase F opens with a command-line reference. Four of the eleven commands -- `doctor`, `roots`, `convert` and the `sfincs` group -- were documented on **no page at all**, having been added faster than the docs followed. | `docs/cli.rst` covers all eleven plus the compatibility group, organised by what a user is trying to do rather than alphabetically, and states the exit-status contract: 0 success, 1 the command ran and the answer is no (a scan point failed, a case is not converged, two results differ), 2 the command could not run at all. It documents the behaviours that are not visible from `--help`: that `compare` dispatches on extension because NetCDF4 *is* HDF5, that timings are reported but do not decide its verdict, that `scan` resume keys on `case_id` rather than position, and that zero admitted roots is not proof that none exist. | The documentation-identity task in Phase F turns out **not** to be a repository edit. `README.md` and `pyproject.toml` point at `sfincs-jax.readthedocs.io`, which looks like stale naming -- but that URL is live and already serves a site titled "DKX documentation", while `dkx.readthedocs.io` returns 404. Changing the links in the repository would break working ones. The rename has to happen on Read the Docs first, which is a maintainer account action; only then does the repository follow. Recorded rather than done. |
| 2026-09-01 | `physics.coulomb_logarithm` added to the case schema, acting on the gap `dkx convert` had just exposed. Convertible checked-in decks go from **2 of 102 to 12**. | The case schema had no collisionality field at all: `execution.py` hard-coded `DEFAULT_NU_N = 8.330e-3`, the pinned value at ln-Lambda 17, so a native case could not express a different Coulomb logarithm and 28 decks were unconvertible for that reason alone. `nu_n` is proportional to ln-Lambda (it enters through `nuBar` in `units.reference_nu_n`), so the field scales the pinned literal rather than recomputing the collisionality. | Scaling rather than recomputing is the load-bearing choice. Deriving `nu_n` from `reference_nu_n()` would be more principled in isolation but gives 8.330406e-3 at the default, shifting **every existing result** by the 5e-5 the Fortran's rounded 8.330e-3 literal carries. Scaling makes the default a bit-for-bit no-op, confirmed across the rhsmode1, transport, drift-kinetic and README suites. The converter inverts that exact expression, so a deck's `nu_n` survives the round trip exactly rather than to 5e-5: 8.4774e-3 maps to ln-Lambda 17.300816 and back to 8.4774e-3. | The field is in the JSON schema and the commented TOML template, without which it would exist but be undiscoverable. Remaining convert blockers, in order: 30 decks need overridable scheme-1 analytic parameters (`execution.py` calls `from_scheme(1)` with no arguments), 24 need `VMECRadialOption`, 15 need `RHSMode` 2/3. |
| 2026-09-01 | `dkx convert` completes the section 5.6 set at eleven of eleven -- and its acceptance test found something more important than the command. **Not one pre-existing checked-in deck converts.** | Verified independently of the implementation: of 102 decks under `tests/ref/` and `examples/sfincs_examples/`, exactly 2 convert, and both are the fixtures written for the acceptance test. Every refusal is a real limit of the native route rather than a converter defect: 30 decks set scheme-1 `epsilon_t` (which `execution.py` cannot override -- it calls `from_scheme(1)` with no arguments), 28 override `nu_n`, 24 use `VMECRadialOption=1`, 15 use `RHSMode` 2 or 3. | The `nu_n` blocker is the sharpest and is a **schema gap, not a missing guard**: the case schema has no collisionality field at all, and `execution.py` hard-codes `DEFAULT_NU_N = 8.330e-3` from the pinned set at ln-Lambda 17. Decks carry 8.4774e-3 or 8.31565e-3, so a native case cannot express a different Coulomb logarithm, and the ~2% flux shift that follows is far above any honest round-trip tolerance. Adding one collisionality field is the highest-value schema addition available to Phase D. Round-trip agreement where conversion *is* possible is 0.0 to 9e-11 across PAS/W7-X, Fokker-Planck/LHD with finite `Er`, and a Boozer scheme-12 fixture -- the 1e-11 floor is the iterative solve at `solverTolerance=1e-10`, not a physics difference. | A deck is one surface with prescribed gradients; a case is a profile that `execution.py` differentiates with `np.gradient`. The converter therefore emits three surfaces carrying a profile linear in `rHat`, exact for `np.gradient` at every node including the edges, with the stencil half-width shrunk to respect the equilibrium's stored span, positivity, and locality. |
| 2026-09-01 | `dkx scan` lands. `[scan]` had been parsed and validated by the case schema from the start, then refused by `run_case` with a message naming a `dkx.scan` module that did not exist -- the same dangling-reference pattern `dkx.converge` had. Section 5.6 is now ten of eleven; only `convert` remains. | The workflow expands the axes (cartesian or zipped), runs each derived case, and writes one Result with a leading `case` dimension. Axis values are written beside the observables so the output says what was varied without also needing the case file. A failing point does not discard the ones that already succeeded: the failure is recorded in that row, the run continues, and the command exits 1. A scan is run because each point is expensive, so losing forty completed solves to the forty-first is the wrong trade. | Building it surfaced a **data-loss bug in resume**, caught by running it twice. Resume compared `case_id` values but carried only the ids forward, so the rewritten output contained just the freshly-run cases -- none, when everything was cached. Resuming a *finished* scan therefore replaced its results with an empty table, destroying precisely what resume exists to preserve. It now carries the cached rows into the new file, pinned by a regression test. Resume also keys on the deterministic `case_id` rather than position, so a scan resumed after the case file was edited misses the cache and reruns instead of grafting new physics onto old rows. |
| 2026-08-31 | `dkx plot` lands, and dkx Results are visible without writing a script for the first time. Section 5.6 now has nine of eleven; `scan` and `convert` remain. | `OutputConfig.plots` had been in the case schema since the native path existed, but nothing read it -- a Result could only be looked at through matplotlib by hand. The native panel is radial profiles against `r_N`, one line per species, plus an `E_r` root panel for ambipolar Results. Deliberately plain: it exists so `dkx run` output can be looked at, not to be a publication figure. | Two decisions worth recording. The root panel draws only `ambipolar_root_count` entries, not the full rectangular row, or every surface with fewer roots than the widest would gain a spurious marker sitting exactly on the zero line. And the "hollow = unstable" caption is drawn only when an unstable root is actually present: a legend for a marker style that does not appear reads as a missing feature rather than an absent branch. |
| 2026-08-31 | Silently-ignored SFINCS namelist options now refuse instead of solving a different equation (operating rule 11), and a real default mismatch was found and fixed. | Every v3 namelist variable in `readInput.F90` was classified as honoured / partially honoured / never reaching the physics. Five keys can change the answer and are guarded at deck-read time, each error naming the key, the value found, what DKX assumes, and the Fortran source line for what differs: `includeTemperatureEquilibrationTerm`, `withNBIspec` (only when `includePhi1` is on and Phi1 is not read externally, matching upstream's own "no impact" rule), `ExBDerivativeSchemeTheta`/`Zeta`, `EParallelHatSpec_bcdatFile`, and `force0RadialCurrentInEquilibrium`. | Two corrections to the premise this started from. `force0RadialCurrentInEquilibrium` is **not a namelist variable at all** -- it is a compile-time global in `globalVariables.F90:65`, so no valid deck can set it and the exposure was overstated; the guard is kept because DKX's tolerant parser would absorb it silently. And the actual live bug was elsewhere: `drift_kinetic.py` defaulted `includeXDotTerm` and `includeElectricFieldTermInXiDot` to `False` where `globalVariables.F90:144-145` defaults both to `.true.`, so a compact deck omitting them with finite `Er` silently lost both E_r trajectory terms. Now matched to upstream. Safe because both terms are gated on `has_er` and all three checked-in decks that omit the key have `Er = 0`; 194 parity tests confirm nothing moved. `useDKESExBDrift` really is `.false.` upstream and is pinned so a later sweep does not "fix" it to match. | Four keys were deliberately **not** guarded, the sharpest being `include_fDivVE_term`: its Fortran block is entirely commented out, so SFINCS ignores it too and DKX ignoring it is exact parity -- and three checked-in decks name it, so guarding would have been a false refusal on live fixtures. |
| 2026-08-31 | Phase D: the twelve SFINCS commands move under `dkx sfincs`, and `dkx --help` drops from 21 entries to 9. | Registered twice from one function -- once under the group, where they are documented, and once at the top level through a `_HiddenAliases` proxy that forces `help=SUPPRESS`. The old spellings still run, because they appear in existing scripts, CI jobs and the upstream comparison harness; removing them to tidy the help output would break work unrelated to the rename. | Suppressing the help was not enough on its own: argparse still prints every choice in the metavar, so all 21 names appeared in the usage line with only their descriptions hidden. The metavar is now built from `_USER_COMMANDS`. That constant is hand-kept, so a test asserts each entry is really dispatchable **against the parser `main` builds** -- the first version of that test rebuilt only the compatibility half and therefore asserted nothing about the user commands, which is exactly the drift that once made `dkx doctor` parse as a filename. |
| 2026-08-31 | Coverage target set to **95%**, not 100% (maintainer decision), and the tier vocabulary retired from the last surfaces that still carried it. | The 100% end state is replaced in all five places it was stated, with the reasoning recorded next to the standard rather than left as a slipped number: the last few percent of branch coverage here is concentrated in defensive impossibilities, platform-unavailable callbacks, and compatibility paths only a second Fortran build would exercise. Chasing those buys shallow tests, which section 7 already forbids, and creates pressure to delete error handling that exists for good reasons. The ratchet is now `80 -> 90 -> 95`. | The vocabulary work was recovered from a background session that ended without landing anything: 79 files, package docstrings and every `tools/` and `examples/` surface. Verified that no identifier moved -- `tier1_memory_budget_gb`, `tier1_keep_lowest`, `_solve_tier2`, `tier1_available`, `DKX_TIER1_MEMORY_BUDGET_GB`, `tier1_peak_memory_bytes` and `tier1_adjoint_window` all still resolve. One real trap found and documented: `workflows/optimization.py` uses "tier" in an unrelated sense -- the resolution levels of a convergence ladder -- and `"tiers"` there is a key in both the user's ladder config and the emitted summary JSON. Renaming it during a vocabulary sweep would break user configs, so the module docstring now says so explicitly. |
| 2026-08-31 | `dkx compare` lands, one command across both formats as section 5.6 specifies. Section 5.6 now has **eight** of eleven -- an earlier row in this log said ten, which was a miscount; `scan`, `plot` and `convert` are the remainder. | Dispatch is by **extension, not by sniffing the file**: NetCDF4 *is* HDF5, so an `h5py` open succeeds on both and would route every dkx Result into the SFINCS comparison, where none of the variable names exist. A mixed pair is refused outright rather than compared -- every key would be unmatched, and "nothing differed" reads as agreement. | Wall-clock time and iteration counts are reported but do not decide the verdict. Two runs of one case always differ in timing, so counting it would make `compare` exit non-zero on a bit-identical re-run and train the reader to ignore the exit status, which is the one thing the command exists to make trustworthy. Writing the tests found a real defect: the verdict counted only `status == "differs"`, so a **shape mismatch passed silently**. It now fails closed on any status that is not `ok` or informational, so a status added later counts by default instead of slipping through. |
| 2026-08-31 | The remaining three audited SOLVAX duplications were each checked against both sides, and **none is removable**. The audit overstated all three. | `multigrid._separable_transfer`: the loop body is byte-for-byte the same math as `solvax.transfer._AxisTransfer.__call__`, so the duplication is real -- but `_AxisTransfer` is private and the public `grid_transfer` builds its *own* matrices from scheme names (`full_weighting`, `linear`, `injection`). DKX supplies matrices built for its odd-periodic constraint, which exists so the centered first-derivative matrix does not annihilate the Nyquist mode into a coarse-grid null vector. No public entry point accepts caller-supplied matrices, so this cannot be deleted. | `sparse_precond` Sherman-Morrison: the identity is the same one `solvax.precond.low_rank_corrected` implements, but that takes a JAX `MatVec` and DKX applies the correction to scipy SuperLU factors inside a single `pure_callback`, batched over every subsystem and every right-hand-side column at once. Routing it through a JAX MatVec would break exactly that batching. Same mathematics, incompatible layer. `sparse_precond` SuperLU loop: the claimed ~20 lines are two -- `splu(m.tocsc())` and `lu.solve(rhs, trans=)`. `SpluFactorization.solve` returns a `jax.Array`, which is the wrong type to return from inside a `pure_callback`, and its numpy path `_solve_numpy` is private. Wrapping two lines of direct scipy in a class whose public API is the wrong layer is not a simplification. **Asks for SOLVAX**, in priority order: (1) make separable application public -- something like `separable_apply(matrices: Mapping[int, Array]) -> MatVec` -- which alone would let DKX delete its copy; (2) a documented numpy-returning solve on `SpluFactorization` for host-callback callers; (3) a rank-one, host-side variant of `low_rank_corrected` that composes with batched factor solves. Until (1) lands, DKX keeps all three. |
| 2026-08-31 | First SOLVAX slice: the operator materializer stops building a dense intermediate, and the differentiable tier-2 path stops discarding its recycle pair. **The audited "2x cost" claim is refuted.** | `materialize_dense` built an `O(total_size^2)` dense array and then converted it to CSR; `solvax.native_eigen.sparse_operator_matrix` emits CSR directly from the same vmapped column sampling. The kinetic operator is a stencil -- roughly 9 of 1121 entries per angular row -- so CSR is what the matrix actually is. `materialize_dense` stays as a wrapper; `test_solve.py` calls it in four places. The `max_dense_size` refusal does **not** relax: it guards the sampling cost, one operator application per column, which the change does not remove. That corrects the hypothesis this slice started from. | The audit claimed `_solve_tier2` runs `gcrot`, discards the recycle pair, then runs it again inside the differentiation wrapper -- "a ~2x cost duplication on every differentiable solve". Measured, the ratio was **1.44**, not 2, and the second solve is the adjoint, which reverse mode requires. The waste was the discarded recycle pair, now threaded out through `has_aux`. Measured on the Sugama `collisionOperator=3` deck: `jax.grad` 2.983 s -> **2.094 s** (-30%), plain solve 2.069 s -> **1.570 s** (-24%), ratio 1.44 -> 1.33. Medians of five runs. Not done in this slice: the Sherman-Morrison and SuperLU duplicates in `sparse_precond.py`, and `_separable_transfer` in `multigrid.py`. |
| 2026-08-31 | The collocation/multigrid promote-or-remove question now has its measured regime matrix, and the answer is neither. | Five rungs on W7-X standard, 48,384 to 614,656 unknowns, all converged to a true relative residual of `1e-10`. The V-cycle **is** resolution-independent: residual reduction per cycle stays between 0.487 and 0.537 across a 12.7x range, flat to 10%. The outer Krylov count is **not**: 94 to 230, a 2.45x growth. The benchmark's own docstring demands both, so the route fails its stated scaling criterion -- but sublinearly, not proportionally. | The measured advantage is memory: 614,656 unknowns at 1.08 GB, past the ~488,000 where the classical route is recorded as exhausting memory. That is why the 3,005 lines in `multigrid.py` and `collocation.py` are not being deleted despite the file-count budget: they reach sizes the structured direct route cannot. Promotion still fails on four criteria -- one geometry, one collisionality, one `Er`, CPU only, derivatives never exercised, and no runtime comparison against the classical route on the same host. Evidence checked in at `validation/benchmarks/collocation_multigrid_h_independence_2026-08-31.json` so the next person deciding this starts from numbers rather than from the blank this was. |
| 2026-08-31 | `dkx roots` lands. Section 5.6 now has seven of its eleven commands; `plot`, `convert` and `compare` remain, and the SFINCS-specific commands are still top-level rather than grouped. | The ambipolar workflow already recorded every root's field, current, slope, classification, bracket and branch, plus the events where a branch appears or vanishes -- nothing surfaced them except reading the NetCDF by hand. Two reporting decisions carry the weight. The rectangular arrays are padded, so the command reads `ambipolar_root_count` rather than the array shape: reading the shape would report a fabricated root at exactly `E_r = 0` with zero current, which looks entirely plausible. And a run that admitted no roots says why that is not proof -- sign sampling cannot see a tangential root or an even number of crossings between samples, both proved in `tests/test_numerics.py` -- so the command cannot silently convert "we did not find one" into "there is none". | A nonsmooth branch event is reported rather than smoothed over, as section 7.6 requires: `jax.grad` returns a number across such an interval whether or not it means anything, so an unmarked event makes a meaningless gradient indistinguishable from a good one. |
| 2026-08-31 | Seven documentation contradictions resolved against the source, and one of them was not a contradiction. | `docs/optimization.rst` quoted two different windows for the x-block policy, `n_active <= 60,000 / Nxi <= 14` and `<= 100,000 / Nxi <= 16`. Both are real: they are successive `policy_after_probe` states as the probe ladder widened, recorded in the `21x25x14` and `25x31x16` audit artifacts. The page now says so instead of asserting each as *the* default. | The larger finding is that **no code implements this policy at all**. `multispecies` appears nowhere in `src/dkx`, and `solve(method="auto")` selects its route from a memory budget (`tier1_peak_memory_bytes` against `DKX_TIER1_MEMORY_BUDGET_GB`), not from an `n_active` or `Nxi` bound. The window is the range the probes cover, so the page now calls it a measured envelope and warns that nothing raises outside it. `docs/differentiability.rst` was genuinely wrong rather than merely unclear: its table listed the truncated kernel's reverse-mode route as a transposed block-Thomas solve, but `solve.py:2334` documents `tier1_adjoint_window=None` as the default and says it keeps the taped gradient. The bounded custom-VJP path is opt-in. | Also: the examples printed "solver tier used", the last user-visible survivor of the retired vocabulary, now "solver route used" with the pinned strings updated; a doubled backslash that rendered `\nu_n` as literal text; and a 2014/2015 citation conflation. Next: the collocation/multigrid promote-or-remove decision still needs its measured regime matrix, and the SOLVAX extraction has not started. |
| 2026-08-31 | Section 7.4 closed: the thermal Lorentz conductivity, the monoenergetic-to-thermal convolution, and the high-collisionality order are proved against closed forms in `tests/test_transport_limits.py`. | The conductivity factor in DKX's units is `8/sqrt(pi)`, not the `32/(3*pi)` this plan reached for: the two are the same result, `32/(3*pi)` being it expressed in Braginskii's `tau_e`. Derived from the code's own definitions rather than assumed -- the inductive drive prefactor in `drift_kinetic.rhs_phi1` and the `L(L+1)/2` collision eigenvalue were each re-derived independently and match. On a uniform-|B| deck streaming and mirror annihilate the L=1 response, so the closed form is geometry-free; reproduced to **2.2e-16** across four decks varying `B0`, `iota`, `GHat`, `IHat`, `dPhiHatdpsiHat`, temperature, mass and collisionality. The convolution is checked against manufactured coefficient tables whose Maxwellian integrals are Gamma functions: 9/9 transport entries to 1.5e-15. Pfirsch-Schlueter is covered in the conductivity channel only -- the leading correction is second order in both `nu_n` and `epsilon_t` (measured orders 1.99), and streaming can only reduce the conductivity, which follows from a variational identity. A sharp general-geometry `D11` closed form needs a magnetic differential equation and was deliberately not attempted. | The assertions were mutation-tested: perturbing the L=1 collision eigenvalue by 1e-6, or the `x^2` convolution weight by 1e-7, fails 8 of the 9 tests. That is the standard the rest of section 7.4 should be held to -- a proof that survives a deliberate small error in the thing it claims to prove is not a proof. |
| 2026-08-31 | A release manifest asserted the opposite of its own evidence, and now cannot again. | `tools/publication_figures/validation_manifest.json` claimed W7-X meets the high-collisionality inverse-nu proxy while LHD fails it. The artifact data says the reverse -- `fp_l11_l12_inverse_like` is true for LHD (Fokker-Planck L11/L12 slopes -0.64 / -0.62) and false for W7-X (-0.53 / -0.43) -- and `docs/paper_figures.rst` agreed with the data. Four claim strings were reversed and are corrected. | The suite already recomputed this gate from the raw scans and pinned both verdicts, so the numbers were never wrong; what was missing is that nothing compared the manifest's English to them. `test_the_manifest_prose_names_the_same_device_the_data_does` closes that, and is written against whichever device currently passes rather than the literal names, so it keeps holding when a wider scan promotes W7-X. Verified by reverting the manifest and watching it fail. The general lesson for section 8: prose claims in checked-in evidence need the same pinning as the numbers, because a reversed sentence passes every numeric test. |
| 2026-08-31 | Phase D: `dkx converge` and `dkx doctor` land, and `converge` immediately found that the shipped example is not converged. | `doctor` reports observed state, not requested state: the float64 row allocates an array and reads its dtype rather than trusting `JAX_ENABLE_X64`, which an already-initialised backend ignores. Adding it exposed a live dispatch bug -- the implicit-namelist path matched against a hand-kept command list, so `dkx doctor` parsed as `dkx write-output doctor` and failed on a file the user never named; `main` now passes the parser's own `sub.choices`. `converge` refines theta, zeta, pitch and speed separately *and* jointly, and reports observables rather than a state-vector norm. | The first real run is the finding. `examples/cases/analytic_tokamak_profile.toml`, which the README quickstart mirrors, moves its particle flux by **1187%** when theta goes 9 -> 14. Worse, section 7.5's warning about separated convergence is not hypothetical here: at `pitch = 8` the theta axis looks settled to 0.2%, and at `pitch = 40` the same theta refinement moves the outputs by **74%** -- the apparent convergence was an artifact of pitch being too coarse to expose the error. The solver route was checked and is identical at both resolutions (`block_tridiagonal_truncated`/`gcrot`, `keep_lowest=3`), so this is discretization, not a route switch. The example and the README now say so and point at `dkx converge`; the README no longer prints a bootstrap current from an unconverged solve. Raising the example to a converged resolution was rejected: it would cost minutes and defeat a smoke test. |
| 2026-08-31 | Phase E survey, fourth pass: the section 7.4 gap list was itself wrong. `tests/test_shaing_callen.py` already proves the Shaing-Callen collisionless bootstrap limit and has since before this plan was written. | The file proves `lambda_bB` against the trapped fraction, runs a `nuPrime` scan from 3e-1 to 1e-4 on axisymmetric and helical decks, and pins the *rate*: the deficit `1 - D31/D31_limit` falls like `sqrt(nuPrime)`, observed 1.73 / 1.86 / 1.77 against the predicted 1.83. The monoenergetic half of the Spitzer limit is also proved -- `D33* -> 1` geometry-free to 3.0e-4 in `test_normalization_physics_gates` -- as is the e-e momentum-restoration factor, though only inside a 1.3-2.1 envelope because the rank-1 restoring model is approximate, not because the test is weak. | The genuinely open section 7.4 items are now just two: the *thermal* Lorentz conductivity closed form, and the monoenergetic-to-thermal convolution against analytic manufactured coefficients. Four surveys have now each found claimed gaps already closed; the lesson is that this plan's inventory sections decay faster than the code, so survey before writing remains a standing rule rather than a one-time correction. |
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

