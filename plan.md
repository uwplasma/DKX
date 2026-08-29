# DKX 3 Authoritative Modernization and Performance Plan

Status: P2/P8 native profile, adaptive all-root, branch-event, VMEC plotting, and Boozer-reader vertical slices in progress

Prepared: 2026-08-25

Repository: `uwplasma/DKX`

Audited baseline: `main` at commit `0d5606ce97e93a7d84f068c751abf2e7b2af6af3`

P0 branch base: `main` at commit `c003ba8e0332509bc07b3a20fddf2b8db2c05ad9`

Target release line: DKX 3.x

Proposed planning branch: `rj/dkx3-plan`

Planning pull request: [#70, `RFC: make plan.md the authoritative DKX 3 roadmap`](https://github.com/uwplasma/DKX/pull/70)

## 0. Authority, scope, and use of this file

This file is the single authoritative roadmap for DKX 3. It defines the product, scientific scope, performance architecture, source ownership, user contracts, validation requirements, work sequence, acceptance gates, and execution state. A contributor or coding agent starting with no prior context should be able to read this file, inspect the repository at the recorded commit, and continue the work without relying on a private conversation.

The repository audit found no open pull request whose purpose is a current DKX modernization plan. The previous comprehensive plan was `plan_final.md` in the already merged and closed pull request #8, `Refactor v3 driver architecture`. Its durable requirements are incorporated here: vertical-slice migration, one canonical owner per behavior, strict production admission gates, reusable solver and preconditioner state, bounded-memory block elimination, measured CPU/GPU scaling, and a prohibition on tiny disconnected refactor pull requests. The old plan and its branch history are historical evidence, not a second authority.

When this file is merged:

- `plan.md` is the only controlling plan in `main`;
- any documentation that describes another roadmap must be deleted, reduced to historical release notes, or point here;
- tests that reject `plan.md`, pin old branch names, or enforce an obsolete source inventory must be removed or rewritten;
- no `plan_final.md`, `roadmap_v2.md`, dated campaign plan, or agent-specific checklist may coexist as another source of truth;
- an unindexed or private planning pull request discovered later must have its unique requirements reconciled into this file and then be closed or explicitly marked superseded;
- completed work remains summarized in the execution ledger rather than spawning a new master plan.

### 0.1 Pull-request rules

Every implementation pull request must:

1. cite one or more work-item IDs from this file;
2. implement the smallest coherent **vertical slice**, not the smallest possible diff;
3. preserve or deliberately revise one documented scientific or product contract;
4. include the tests, validation evidence, performance evidence, documentation, and migration notes required by that slice;
5. update this file before merge with the result, measurements, unresolved risks, and next action;
6. avoid mixing a source move, physics change, numerical-method change, and public-API redesign unless they are mechanically inseparable;
7. leave one canonical owner for the changed behavior and remove the superseded owner in the same pull-request series;
8. never promote a route because it is new, elegant, or JAX-native; promotion requires the admission gates in this file.

A pull request that only renames a few files, adds an unused abstraction, or moves code without deleting an old path is too small. A pull request that rewrites several physics families and every user interface at once is too large. The normal unit is one end-to-end behavior: input to model to solver to result to tests and documentation.

### 0.2 Execution ledger rules

This file is not a command transcript. Each merged pull request or consequential failed experiment gets one compact ledger row containing:

- date and work-item IDs;
- branch, pull request, and commit;
- behavior changed;
- tests and validation run;
- cold and warm performance measurements when relevant;
- accepted limitations or failed hypotheses;
- next exact action.

Large benchmark tables, profiler traces, generated outputs, and external-code builds belong in release assets, CI artifacts, or a versioned evidence bundle. The ledger links to them by stable identifier.

### 0.3 Performance is a cross-cutting release requirement

Performance is not postponed until a late optimization phase. Every phase must preserve or improve the relevant compilation count, runtime, memory use, data transfer, and parallel scalability. A new API, data model, validation layer, or documentation example is incomplete if it creates an avoidable copy, recompilation, synchronization, or duplicate solve in a flagship workflow.

## 1. Mission

Build DKX into the dependable, fast, differentiable neoclassical workflow used by tokamak and stellarator researchers for:

- radially local, non-monoenergetic drift-kinetic calculations;
- monoenergetic coefficients and Maxwellian thermal convolutions;
- particle and heat fluxes, parallel flows, conductivity, momentum transport, and bootstrap current;
- all ambipolar radial-electric-field roots and their continuation across a radial profile;
- multispecies, impurity, and momentum-conserving collision calculations;
- reduced-to-full model comparisons, convergence certification, and validity warnings;
- low-runtime and bounded-memory calculations on laptop CPUs, workstations, and GPUs;
- parallel parameter scans, radial-profile calculations, and optimization loops;
- exact or certified sensitivities through linear solves, nonlinear solves, and branch-local ambipolar roots;
- in-memory coupling to VMEX and other equilibrium, optimization, and transport tools;
- permanent SFINCS namelist and output compatibility for established workflows.

DKX should be easy enough that a new user can obtain a scientifically interpretable result with one TOML file, yet powerful enough that an expert can control the equation, discretization, solver, sharding, and differentiation policy without bypassing the supported architecture.

The product is not defined by the number of options. It is defined by a small set of complete workflows that are fast, reliable, transparent, reproducible, and independently verified.

## 2. Accepted product decisions

The following decisions are settled for DKX 3. Changing one requires an explicit maintainer decision recorded in this file.

| ID | Decision | Accepted direction |
| --- | --- | --- |
| D001 | Major version | DKX 3 may break the DKX 2 Python API. Preserve high-value behavior and file compatibility, not accidental imports or signatures. |
| D002 | SFINCS compatibility | SFINCS namelist reading and SFINCS-compatible HDF5 output remain permanent adapters. Native DKX models do not depend on namelist objects. |
| D003 | Native units | User-facing TOML, JSON, and high-level Python APIs use physical or engineering units. Normalize exactly once at the canonical model boundary. Store physical and normalized metadata. |
| D004 | Native result | Versioned NetCDF is canonical. SFINCS-compatible HDF5 is a permanent adapter. Compact NPZ or JSON artifacts are internal or evidence formats only. |
| D005 | Python floor | DKX 3 requires Python 3.11 or newer. |
| D006 | CLI | Keep `argparse`; add Rich for progress, tables, diagnostics, and readable errors. Do not add Typer or Click without a measured maintenance benefit. |
| D007 | Package layout | Use `src/dkx` and shallow domain packages. Avoid both a forty-file flat root and deeply nested framework-style packages. |
| D008 | First flagship | Whole-profile ambipolar roots, neoclassical transport, and bootstrap current are the first flagship workflow. Monoenergetic throughput, optimization, Phi1, and impurity workflows remain high priorities. |
| D009 | Capability status | A feature that reproduces a matched, converged SFINCS v3 capability and passes DKX release gates is a stable candidate. JAX-only and research features require additional independent validation, documentation, and examples before stable promotion. |
| D010 | External codes | Codex may install and run SFINCS, MONKES, YANCC, KNOSOS, NEO, BOOTSJ, and PENTA locally. They are validation and benchmark tools, not DKX runtime or CI dependencies. |
| D011 | Benchmark hardware | The official release baseline is a named laptop CPU. A named NVIDIA GPU reached through the office SSH host provides the maintained accelerator lane. |
| D012 | Size | A fresh full DKX clone, DKX wheel, DKX source distribution, and installed DKX-owned package files must each be below 20 MiB. Third-party dependencies such as JAX are excluded from the package-footprint number and must not be represented as if they fit within 20 MiB. |
| D013 | Planning workflow | Merge this plan into `main`, then execute through small coherent feature branches and pull requests. No long-lived competing modernization branch. |
| D014 | Documentation stack | Use MyST Markdown, Sphinx, and Furo, with minimal custom CSS and a path to a later shared UW Plasma theme. |
| D015 | Plan retention | Preserve decisions, evidence IDs, failed hypotheses that affect future work, migrations, and current state. Summarize ordinary completed steps at releases so the plan remains readable. |
| D016 | Performance priority | Runtime, memory, compile behavior, reuse, and parallel scaling are release contracts. A scientifically correct route that is unusably slow or memory-heavy is not production-ready. |

### 2.1 Interpretation of the 20 MiB installation target

`pip install dkx` necessarily installs or resolves third-party scientific packages whose combined size is far above 20 MiB. The enforceable target is therefore DKX-owned content:

- downloaded DKX wheel: below 20 MiB;
- downloaded DKX source distribution: below 20 MiB;
- installed files under the DKX distribution: below 20 MiB;
- fresh full Git clone including `.git` and working tree: below 20 MiB.

The release report must show all four measurements separately. It must not imply that a complete JAX environment is below 20 MiB.

## 3. Product position

### 3.1 Competitive landscape

The main codes occupy different portions of neoclassical physics and numerical design:

- SFINCS remains the broad reference for a radially local four-dimensional drift-kinetic equation, full linearized Fokker-Planck collisions, multiple trajectory models, Phi1, and established stellarator and tokamak workflows.
- YANCC is a modern JAX, GPU, differentiable implementation of full and monoenergetic drift-kinetic equations with flexible recycled Krylov methods and multigrid preconditioning.
- DKES remains a reference for monoenergetic coefficients and variational upper and lower bounds.
- MONKES exploits Fourier and Legendre structure with block-tridiagonal elimination for fast monoenergetic calculations.
- KNOSOS is specialized for fast bounce-averaged low-collisionality calculations, including effects important in that regime.
- NEO, NEO-2, BOOTSJ, and PENTA provide established reduced, tokamak, bootstrap, and ambipolar workflows.
- FORTEC-3D and other global approaches address finite-orbit-width and radially nonlocal physics at much higher cost.

DKX cannot be positioned only as “SFINCS in JAX.” JAX, GPU execution, differentiability, full collisions, monoenergetic calculations, and multigrid are no longer unique claims. DKX should own the combined position:

> **DKX is the verified, high-performance, differentiable neoclassical workflow that unifies reduced and full local models, follows all ambipolar branches over complete profiles, reports numerical and physical validity certificates, and couples directly to equilibrium and optimization codes.**

### 3.2 Advantages to preserve and strengthen

The current repository already contains assets that should be consolidated rather than rewritten without cause:

- broad SFINCS v3 input, output, and field-level parity;
- pitch-angle-scattering and full Fokker-Planck collision paths;
- profile-gradient and transport-matrix solves;
- ambipolar scans, root solving, and implicit root derivatives;
- monoenergetic databases and thermal convolution;
- variational transport bounds;
- a Shaing-Callen collisionless-limit evaluator;
- exact linear-solve differentiation through Solvax;
- structured block elimination, recycled Krylov, and physics preconditioners;
- CPU and GPU execution;
- VMEX-facing geometry and optimization paths;
- extensive reference, benchmark, and solver-trace artifacts.

The modernization is an in-place scientific consolidation. Validated kernels move behind stable contracts one domain at a time. A rewrite is justified only when the current representation prevents correctness, performance, composability, or maintainability and the replacement passes all admission gates.

## 4. Audited baseline and capability policy

### 4.1 Current strengths

- Tests already contain genuine mathematical and physical gates: conservation and nullspaces, ambipolar-root replay, finite-difference checks of implicit derivatives, variational bounds, resolution trends, operator parity, and complete SFINCS-output comparisons.
- The repository has PyPI publishing, warning-clean documentation builds, CPU/GPU code paths, solver traces, and recent work on repository size and installed-wheel failures.
- The current implementation has measured cases in which structured elimination is substantially faster and lighter than a global sparse solve, as well as documented hard cases in which general Krylov routes remain expensive.
- The project records limitations and failed performance routes instead of presenting only favorable cases.

### 4.2 Problems to correct

1. **Installed artifacts are not yet the primary test subject.** Source-checkout CI has allowed broken wheel workflows to ship.
2. **Coverage policy is below the requested standard.** The active gate is lower than 95 percent and several large risk-bearing owners remain under-tested.
3. **The source tree is shallow but not simple.** Several very large modules combine physics, discretization, solver policy, compatibility, and orchestration.
4. **Importing DKX mutates process-wide runtime state.** An embeddable library must not initialize distributed JAX, set thread counts, change XLA flags, or create cache directories on import.
5. **SFINCS compatibility still shapes native internals.** Compatibility is valuable, but legacy names and `RHSMode` should not define the DKX 3 user or domain model.
6. **Examples and tools expose project history rather than one teaching path.** Multiple overlapping taxonomies make it hard to identify canonical workflows.
7. **Documentation mixes end-user guidance with campaign management and source-history notes.**
8. **Performance is uneven.** Structured cases are strong, while some full-physics Krylov cases require large memory, many iterations, or long wall times; GPU execution does not automatically imply GPU speedup.
9. **Repeated-workflow reuse is not yet a first-class product contract.** Profile, scan, root, and optimization calculations must reuse compilation, geometry, factors, preconditioners, warm states, and Krylov information deliberately.
10. **Current JAX runtime and cache policy is too implicit.** Compile families, transfer behavior, memory ownership, and sharding need explicit models and evidence.
11. **Repository size is at the requested ceiling.** Media, fixtures, generated artifacts, and history require automated budgets.
12. **The product identity overlaps YANCC.** DKX needs to differentiate through complete profile workflows, evidence, compatibility, and coupling rather than a longer solver feature list.

### 4.3 Capability status policy

The planning pull request must generate `validation/capabilities.toml` or an equivalent machine-readable registry. Each capability is one of:

- `stable`: supported in the public API, documented, tested in the installed artifact, and backed by release evidence;
- `stable_candidate`: matches a supported SFINCS v3 capability with converged field-level comparison but still needs DKX 3 API, packaging, or documentation gates;
- `validated_limited`: scientifically verified only for stated geometry, model, resolution, device, or parameter ranges;
- `experimental`: callable for research but excluded from stable defaults and headline claims;
- `compatibility_only`: retained to read or reproduce a legacy workflow but not recommended for new native cases;
- `deprecated`: scheduled for removal with a migration path.

Default classification rule:

- SFINCS-overlap features that have matched equations, converged reference runs, field-by-field comparison, and installed-artifact tests begin as `stable_candidate` and become `stable` after DKX 3 documentation and release gates.
- JAX-only features, new collision models, new preconditioners, bounce-averaged surrogates, variational certificates, distributed routes, and optimization chains remain `validated_limited` or `experimental` until independent evidence is complete.
- A GPU or sharded execution path has its own status. A stable CPU model does not automatically make its accelerator implementation stable.
- A feature is not promoted by README prose, same-code regression, or one visually plausible plot.

## 5. DKX 3 success criteria

A stable DKX 3 release is complete only when the user, scientific, software, and performance criteria below all pass.

### 5.1 User success

A new user can:

- install DKX from PyPI with one normal command;
- run `dkx doctor` and receive a useful device, precision, cache, thread, memory, and dependency report;
- run an included tokamak or stellarator case from TOML;
- see stages, surfaces, scan progress, root brackets, convergence rungs, solver iterations, and elapsed time without debug output;
- obtain a versioned NetCDF result containing physical units, normalized values, provenance, convergence data, and plots;
- run a declarative scan, resume it, and find all ambipolar roots without writing Python;
- run the same workflow from a short Python script using `Case`, `run`, and `Result`;
- convert and execute a supported SFINCS namelist and export a SFINCS-compatible HDF5 result;
- find the governing equations, normalization, algorithm, limitations, performance expectations, and validation evidence in the documentation.

### 5.2 Scientific success

- Every stable observable has at least one analytic, mathematical, manufactured, literature, or independent-code validation gate.
- Full and reduced models state their assumptions and validity range.
- Ambipolar roots include bracket, residual, local slope, stability classification, branch identity, resolution evidence, and transition information.
- Bootstrap-current, flux, flow, and conductivity outputs use documented signs, radial coordinates, flux-surface averages, and unit conventions.
- Primal and differentiated solves report true residual certificates, not only recursive or preconditioned residuals.
- Cross-code comparisons match equations, collision operators, trajectory models, geometry, normalization, resolution, tolerances, and convergence criteria.
- A performance improvement is rejected if it changes stable observables, conservation defects, variational bounds, branch selection, or derivatives outside accepted tolerance.

### 5.3 Software success

- The package uses `src/` layout and CI tests the built wheel and source distribution outside the checkout.
- `import dkx` is inert with respect to JAX precision, distributed initialization, environment variables, threads, warning policy, and cache creation.
- Installed-package line and branch coverage are each at least 95 percent, with narrow hardware-only exclusions documented.
- Pull-request, nightly, local external-code, GPU, and release lanes have separate purposes.
- Core runtime dependencies are limited to ordinary documented workflows.
- The package is on PyPI and, after the API stabilizes, conda-forge.
- Read the Docs builds warning-free with all getting-started and tutorial snippets executed in CI.
- The full clone, wheel, source distribution, and installed DKX-owned files each remain below 20 MiB.

### 5.4 Performance success

Performance success is assessed by workload family, not one favorable deck.

- Benchmarks separate import, input/geometry setup, tracing, compilation, preconditioner setup, factorization, solve, post-processing, output, and complete wall time.
- Cold compile, first execution, warm execution, repeated-solve throughput, peak host memory, peak device memory, transfer volume, iterations, and true residual are reported.
- Official release results are measured on the named laptop CPU; accelerator results are measured separately on the named office NVIDIA GPU.
- Same-family scans and radial profiles reuse compiled executables. Compilation is amortized rather than repeated per surface, electric field, species drive, or optimizer iteration.
- Geometry transforms, collision tables, operator coefficients, factorization analysis, factors, preconditioners, warm states, and recycle spaces are reused when validity checks permit.
- Exact structured cases use the fastest validated bounded-memory route rather than the general solver by default.
- General cases have a robust preconditioned Krylov route whose memory model prevents swapping and out-of-memory termination.
- Independent scan and profile work can be sharded over available devices without code changes in user examples.
- Multi-device claims include strong- or weak-scaling efficiency and communication costs. “Runs on multiple devices” is not enough.
- Pull requests fail on unexplained median runtime regressions above 10 percent or peak-memory regressions above 10 percent in the fast benchmark set. Larger deliberate changes require an approved evidence note.
- Release candidates include all cases that failed, timed out, exceeded memory, or lost to a reference code; summaries may not filter them away.
- Custom kernels, mixed precision, low-rank compression, pipelined Krylov methods, and single-solve sharding become stable only when they improve an end-to-end flagship case at matched accuracy.

## 6. Non-goals

The first stable DKX 3 release will not:

- rewrite every validated physics kernel at once;
- reproduce every radially global capability of FORTEC-3D or global gyrokinetic tools;
- promise smooth derivatives through ambipolar branch creation, loss, or switching;
- require native users to understand SFINCS option numbers;
- add a dependency merely to shorten ordinary Python;
- preserve accidental DKX 2 internals or every old import path;
- make PETSc, MUMPS, SuperLU_DIST, SFINCS, MONKES, or YANCC runtime dependencies;
- claim that a full `pip install` environment including JAX is below 20 MiB;
- introduce Pallas, FFI, custom CUDA kernels, low-rank factorization, or multi-host execution before profiling identifies a stable bottleneck and a simpler route is insufficient;
- claim universal superiority from unmatched hardware, equations, tolerances, or accuracy;
- commit generated solver outputs, profiler traces, compiled binaries, large equilibria, or paper-scale datasets to Git.

## 7. Product contracts

### 7.1 Stable high-level Python workflow

The canonical user path is deliberately small:

```python
from pathlib import Path

import dkx

case = dkx.Case.from_file("w7x_profile.toml")
result = dkx.run(case)

result.print_summary()
result.plot(Path("outputs/w7x_profile.webp"))
result.save(Path("outputs/w7x_profile.nc"))
```

The high-level API should expose no more than these concepts:

- `Case`: immutable, validated physical, numerical, execution, and output input;
- `Result`: immutable scientific result with coordinates, units, certificates, provenance, plotting, and saving;
- `run`: dispatch the workflow declared by a case;
- `scan`: execute or resume declarative scans;
- `load`: read native NetCDF or supported compatible output;
- `convert`: translate supported SFINCS inputs and outputs;
- `doctor`: programmatic runtime and environment diagnostics;
- `dkx.advanced`: a clearly marked namespace for operators, solver plans, differentiable objectives, and research controls.

DKX 2 public functions may be removed, renamed, or wrapped. A compatibility shim is retained only when it is low-cost and supports a real downstream workflow. There is no goal of preserving complete DKX 2 Python API compatibility.

### 7.2 Native `Case` model

`Case` is the only canonical high-level input model. It must not be a decorated SFINCS namelist. It contains typed submodels for:

- geometry and radial surfaces;
- species and profiles;
- physical model and collision operator;
- radial electric field or ambipolar policy;
- phase-space resolution;
- nonlinear and linear solver policy;
- convergence policy;
- execution, batching, sharding, precision, and memory policy;
- output and plotting policy;
- declarative scans.

Normalization occurs once when the validated physical case is transformed into a canonical internal problem. The normalizer must return both the internal arrays and a versioned record of reference quantities, signs, radial-coordinate derivatives, and conversion factors. No later module may silently renormalize a value.

### 7.3 Native TOML and JSON

TOML is the primary human-authored format. JSON uses the same schema for generated inputs. The schema is versioned from its first release and has an explicit migration function.

A representative profile input is:

```toml
schema = 1
name = "w7x_ambipolar_profile"

[run]
workflow = "ambipolar_profile"
precision = "float64"
device = "auto"
progress = true

[geometry]
format = "vmec"
file = "wout_w7x.nc"
surfaces = [0.20, 0.35, 0.50, 0.65, 0.80]

[[species]]
name = "deuterium"
charge = 1
mass_amu = 2.014
density_m3 = [8.0e19, 7.3e19, 6.4e19, 5.2e19, 3.8e19]
temperature_keV = [1.2, 1.1, 0.95, 0.75, 0.50]

[[species]]
name = "electron"
charge = -1
mass_amu = 5.485799e-4
density_m3 = [8.0e19, 7.3e19, 6.4e19, 5.2e19, 3.8e19]
temperature_keV = [2.5, 2.3, 2.0, 1.6, 1.0]

[physics]
model = "full_local"
collisions = "linearized_fokker_planck"
magnetic_drifts = "full"
phi1 = "off"

[electric_field]
mode = "ambipolar"
search_kV_m = [-40.0, 40.0]
find_all_roots = true
continue_branches = true

[resolution]
theta = 31
zeta = 31
pitch = 24
speed = 8

[solver]
method = "auto"
relative_tolerance = 1.0e-10
memory_fraction = 0.75
reuse = "auto"

[parallel]
strategy = "auto"
shard = ["surface", "electric_field"]

[convergence]
enabled = true
observables = ["particle_flux", "heat_flux", "bootstrap_current", "electric_field"]
relative_tolerance = 0.02
max_refinements = 3

[output]
file = "outputs/w7x_ambipolar.nc"
plots = true
```

Declarative scans use paths into the same schema:

```toml
[scan]
combine = "cartesian"
resume = true
output = "outputs/er_density_scan.nc"

[[scan.axis]]
path = "electric_field.value_kV_m"
values = [-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0]

[[scan.axis]]
path = "species[deuterium].density_scale"
values = [0.5, 1.0, 1.5]
```

Required schema behavior:

- descriptive physical names instead of Fortran option numbers;
- explicit engineering units in field names or schema metadata without a runtime unit-package dependency;
- precise validation errors with the input path, supplied value, expected form, and a correction;
- Cartesian, zipped, linear, logarithmic, and explicit-list scans;
- deterministic case IDs based on normalized semantic content rather than file ordering;
- bounded case-count and memory estimates before launching a scan;
- resumable, append-safe output;
- optional reusable profile blocks only when the resulting file remains readable;
- round-trip conversion for the supported SFINCS subset;
- a `dkx schema` command that prints a complete commented example and machine-readable JSON schema.

### 7.4 Permanent SFINCS adapters

Required compatibility commands:

```console
dkx run input.namelist
dkx convert input.namelist case.toml
dkx convert result.nc sfincsOutput.h5
dkx validate input.namelist
```

The adapter maps a namelist once into `Case` and records:

- source keys and values;
- aliases applied;
- SFINCS defaults applied;
- values that cannot be represented exactly in native DKX;
- compatibility-only options;
- unsupported options and the reason they are rejected.

Native code must not convert `Case` back into a namelist to build geometry, grids, operators, or solvers. SFINCS names may remain in the compatibility result, but native `Result` uses documented DKX names and dimensions.

### 7.5 Native `Result` and NetCDF schema

The canonical output is versioned NetCDF organized around named dimensions and coordinates and readable through `netCDF4` without xarray. The layout should also be natural for xarray users without requiring xarray as a core dependency.

Every native result must contain, as applicable:

- complete canonical input and original input text;
- physical and normalized coordinates and conversion factors;
- species, radial surface, scan, force, response, speed, pitch, angle, root, branch, convergence-rung, and iteration coordinates;
- particle, heat, momentum, and current observables;
- transport matrices, monoenergetic coefficients, distribution diagnostics, and classical contributions;
- all ambipolar roots, root classes, slopes, brackets, branch IDs, and branch events;
- primal and transpose residuals, conservation defects, reciprocity defects, variational gaps, discretization estimates, and model-validity flags;
- selected solver, preconditioner, route-reason codes, iteration counts, rebuilds, warm-start source, and recycle-space use;
- import, setup, compile, factorization, solve, post-process, output, and total timings when measurement is enabled;
- peak host and device memory, estimated memory, transfer diagnostics, and sharding layout when measured;
- DKX, Python, JAX, jaxlib, Solvax, platform, precision, device, commit, and schema versions;
- geometry source, checksums, profile checksums, external-reference IDs, and validation-manifest IDs;
- warnings that are structured fields, not only console strings.

`Result` provides:

- `print_summary()`;
- `save(path)`;
- `plot(path=None, panels="auto")`;
- `certificate()`;
- `to_dict()` for small metadata;
- direct array access by documented names;
- explicit migration when an older native schema is loaded.

SFINCS-compatible HDF5 export must preserve its expected keys and conventions. It is tested as an adapter and must not constrain the native schema.

### 7.6 CLI contract

The supported command surface is:

```console
dkx doctor
dkx validate case.toml
dkx run case.toml
dkx scan case.toml
dkx converge case.toml
dkx roots case.toml
dkx plot result.nc
dkx inspect result.nc
dkx convert input.namelist case.toml
dkx schema --format toml
```

The CLI uses `argparse` and Rich. It must support plain-text and noninteractive output for logs, schedulers, tests, and screen readers.

Progress is emitted through structured host-side events. Numerical kernels do not print and do not call back to the host merely to update a progress bar. Typical events include:

- input and schema validation;
- geometry loading and interpolation;
- compilation-family selection;
- memory and route estimate;
- operator and preconditioner setup;
- compile start and finish;
- surface, electric-field, collisionality, convergence-rung, and optimizer progress;
- Krylov or nonlinear residual updates at a bounded cadence;
- root brackets and branch continuation;
- output and plotting;
- final timing, memory, residual, and certificate summary.

The library event stream can be silenced, collected, or rendered by another application. Benchmark mode disables presentation while preserving timing events.

User-facing output names solver routes by what they do: `structured direct`,
`recycled Krylov`, and `sparse direct referee`. Historical “tier 1/2/3”
labels may appear in migration notes and compatibility metadata, but they are
not the primary language in commands, progress, plots, errors, or tutorials.

## 8. Target source organization and ownership

Use `src/` layout so tests and examples exercise installed code. Keep packages one domain level deep unless a compelling owner boundary emerges.

```text
src/dkx/
  __init__.py
  _version.py
  config.py
  runtime.py
  execution.py

  geometry/
    model.py
    analytic.py
    readers.py
    interpolation.py

  kinetics/
    grids.py
    trajectories.py
    collisions.py
    operator.py
    constraints.py
    moments.py

  solvers/
    structured.py
    krylov.py
    preconditioners.py
    multigrid.py
    direct.py
    implicit.py
    diagnostics.py

  workflows/
    surface.py
    transport.py
    monoenergetic.py
    ambipolar.py
    profiles.py
    scans.py
    convergence.py
    optimization.py

  io/
    native.py
    sfincs.py
    results.py
    plotting.py

  cli/
    main.py
    commands.py
    rendering.py

  validation/
    registry.py
    references.py
    certificates.py
```

### 8.1 Ownership rules

- `config.py` owns canonical case models, validation, normalization requests, and schema migration.
- `runtime.py` owns explicit JAX precision, cache, thread, distributed, and device configuration. Importing DKX does not invoke it.
- `execution.py` owns compilation-family IDs, execution plans, batching, sharding, memory budgets, and reusable state descriptors.
- `geometry/` owns flux-surface geometry, readers, Fourier representations, derivatives, and interpolation.
- `kinetics/` owns the equation, discretization, collision physics, constraints, and moments. It contains no CLI or file-format logic.
- `solvers/` owns DKX-specific solver policy and physics-aware preconditioners. Generic algorithms remain in Solvax when they are reusable outside DKX.
- `workflows/` owns orchestration across surfaces, roots, scans, convergence, and optimization. It does not duplicate operators or moments.
- `io/` owns serialization, compatibility, and plotting; it does not compute missing physics.
- `validation/` owns capability status, evidence loading, and certificate construction; it does not become a second workflow stack.
- Compatibility code depends on canonical DKX models. Canonical code never imports SFINCS adapters.

### 8.2 Vertical-slice migration

Migration proceeds by complete owner:

1. freeze behavior and evidence;
2. introduce the new canonical owner;
3. route one supported public workflow through it;
4. compare outputs, derivatives, runtime, and memory against the baseline;
5. migrate remaining callers;
6. remove the old owner and obsolete tests;
7. update source map, docs, examples, and this plan.

Temporary shims are allowed only during an active slice. No “new architecture” may remain unused beside the old stack.

### 8.3 Source quality rules

- Functions are small enough to state one responsibility but not split merely to satisfy a line count.
- Module names describe stable physics or numerics, not attempts, dates, authors, or versions.
- Docstrings explain contracts, units, shapes, assumptions, and return values. Derivations belong in documentation, not hundred-line docstrings.
- Comments explain non-obvious reasoning, invariants, numerical traps, or source correspondence; they do not narrate every line.
- Public functions are typed. Array shapes and axis order are documented near their owning model.
- No environment-variable-only production route. Stable controls are case, API, or CLI options with provenance.
- No exact module-inventory or line-count tests. Use import, dependency-direction, API, and complexity review instead.
- Delete before adding when behavior is duplicated.

## 9. Performance architecture

### 9.1 Performance principles

DKX performance work follows this order:

1. remove avoidable Python, import, I/O, normalization, and host-device overhead;
2. stabilize shapes and compile families so work is compiled once and reused;
3. exploit mathematical structure before applying a generic solver;
4. separate exact operator application from a cheaper physics-aware preconditioner;
5. reuse analysis, factors, preconditioners, warm states, and recycled subspaces across related solves;
6. batch and shard independent work before distributing a tightly coupled single solve;
7. control Krylov and factor memory explicitly;
8. introduce mixed precision, low-rank approximations, pipelining, or custom kernels only after an end-to-end profile identifies the need;
9. preserve true-residual, conservation, and observable accuracy at every step.

A lower kernel time that causes more compilation, more iterations, more transfers, or more memory is not necessarily an improvement. The governing metric is complete time and peak memory for a scientifically matched workflow.

### 9.2 Lessons to carry forward from established solvers

The methods below are design evidence, not permission to copy source. Code with incompatible licensing must not be copied. DKX implementations should be derived from published algorithms and independently tested.

| Source | Relevant design lesson for DKX | Required DKX action |
| --- | --- | --- |
| SFINCS v3 and PETSc | Keep the exact operator and the preconditioning operator distinct; permit direct and iterative routes; expose runtime solver policy; reuse preconditioners for successive systems; support transpose solves; inspect the true residual rather than trusting only the default preconditioned norm. | Preserve a matrix-free exact operator, build a deliberately simpler physics preconditioner, record route decisions, add reuse and transpose contracts, and make true residuals release gates. |
| MONKES | Fourier-angle and Legendre-pitch structure yields a block-tridiagonal system; Schur-complement elimination can factor each angular block once, reuse it for multiple right-hand sides, sweep all pitch modes, and retain only low modes needed for moments to reduce storage. | Keep and improve the structured path; batch right-hand sides; reuse factors; offer full-factor, Schur-only, streamed, and checkpointed storage policies; retain only scientifically required solution modes when the user does not request the full distribution. |
| YANCC | Flexible recycled GCROT, warm starts, reusable preconditioners and source/constraint data, lower-order geometric multigrid, static-shape scan reuse, periodic true-residual stabilization, and avoiding storage of a flexible preconditioned basis when the preconditioner is linear. | Benchmark GCROT/FGMRES variants, classify preconditioners as fixed-linear or variable, reduce Krylov-basis memory for fixed-linear routes, refresh true residuals, and make repeated-solve state explicit. |
| MUMPS | Separate analysis, factorization, and solve; estimate memory during analysis; use ordering, scaling, pivoting, iterative refinement, single-precision factors with double-precision solutions, out-of-core fallback, tree parallelism, and optional block-low-rank compression. | Give DKX factor routes explicit analysis/factor/solve phases and memory estimates; evaluate mixed-precision preconditioning with double residual correction; treat out-of-core and low-rank ideas as optional research routes, not defaults. |
| SuperLU_DIST | Distributed sparse direct solution uses OpenMP within a rank, MPI between ranks, GPU numerical kernels, and newer 3-D communication-avoiding algorithms for stronger scaling than older 2-D layouts. | Use SuperLU_DIST as a local reference and optional benchmark backend; learn from communication-avoiding layouts before attempting single-solve multi-host DKX; do not add it to core dependencies. |
| JAX/XLA | JIT and static shapes enable fusion and executable reuse; explicit sharding exposes distributed layout; asynchronous dispatch requires synchronization for timing; persistent caches reduce repeated compile cost; buffer donation can reduce peak memory; profiler and device-memory tools are required to verify improvements. | Define compilation families, inspect lowered programs and profiler traces, synchronize every measurement, use explicit sharding plans, and admit donation or custom kernels only with measured memory/runtime gains. |

### 9.3 Workload hierarchy and parallelism order

DKX has several levels of parallelism. They should be used in descending order of benefit-to-complexity.

#### Level A: independent cases

Independent scan points, equilibria, profile scenarios, and optimization samples require no collectives. This is the first multi-device and multi-process target.

- shard complete cases across devices or worker processes;
- preserve deterministic result ordering and case IDs;
- balance work dynamically when iteration counts differ;
- avoid replicating large immutable data unnecessarily;
- write independent chunks and merge metadata safely;
- expect near-linear scaling when the batch is larger than the device count and each case is sufficiently large.

#### Level B: surfaces and electric-field points in one workflow

Radial surfaces and electric-field brackets often share geometry families, resolution, species count, and operator shapes.

- group surfaces into compilation families;
- `vmap` or chunk over equal-shaped work where it reduces launch overhead and increases accelerator occupancy;
- use `lax.map` or bounded chunks when full batching exceeds memory;
- shard the batch axis across devices;
- warm-start neighboring electric-field points and radial surfaces only when the branch and model make that safe;
- order work to maximize reuse without biasing all-root discovery.

#### Level C: multiple right-hand sides

Transport matrices, drives, adjoints, and sensitivity batches share the same operator.

- factor once and solve multiple right-hand sides together on structured/direct routes;
- compare block Krylov against independent recycled solves on iterative routes;
- choose batch width from memory and backend throughput;
- never rebuild geometry, collision tables, constraints, or preconditioners per right-hand side.

#### Level D: one large solve across devices

Single-solve sharding is more communication-intensive and is not the first accelerator milestone.

Candidate decompositions include:

- angular or operator-row sharding with collective dot products and reductions;
- species-speed subsystem sharding where coupling permits;
- distributed preconditioner blocks with replicated small coarse information;
- parallel block-tridiagonal algorithms such as cyclic reduction or recursive doubling when the sequential Legendre sweep becomes the measured bottleneck;
- communication-avoiding or pipelined Krylov variants when global reductions dominate.

A single-solve sharded route remains experimental until it beats the best one-device route end to end, reports communication time, and passes identical residual and observable gates.

#### Level E: multi-host execution

Multi-host JAX is a later research lane. It requires explicit startup, fault behavior, topology-aware meshes, distributed output, and cluster benchmark ownership. It must never initialize during import.

### 9.4 Compilation families

A `CompilationFamily` is defined by values that change array shapes, control-flow structure, or generated kernels. It normally includes:

- number of species;
- angular, pitch, and speed resolutions;
- geometry representation and symmetry class when shape-changing;
- collision and trajectory model;
- constraint family;
- Phi1/nonlinear structure;
- solver algorithm and preconditioner structure;
- batch shape and sharding layout;
- precision policy.

Continuous values such as density, temperature, gradients, electric field, collisionality, Fourier coefficients, and objective weights are dynamic arrays unless they truly change shape or algorithm.

Every workflow must report:

- compilation-family ID;
- number of traces and compilations;
- cache hit/miss status where available;
- compile time and executable-reuse count;
- reason when a new family is created.

The normal profile and scan paths must not recompile once per surface or parameter value. A new family is acceptable when resolution or discrete physics genuinely changes; accidental recompilation is a bug.

### 9.5 Data layout and operator fusion

Before changing algorithms, profile array layout and memory traffic.

Required work:

- document the canonical axis order for state, geometry, coefficients, right-hand sides, and moments;
- benchmark layouts on laptop CPU and NVIDIA GPU because the best contiguous axis may differ;
- avoid repeated transposes between operator terms, preconditioners, moments, and output;
- precompute immutable geometry and collision coefficients once per family or surface;
- fuse pointwise coefficient application, masks, and adjacent stencil operations when XLA can do so without increasing live memory;
- inspect lowered HLO or StableHLO and profiler traces rather than assuming `jit` fused a path;
- avoid materializing dense broadcasted coefficient tensors when a separable representation suffices;
- use batched matrix-matrix operations for dense angular blocks instead of many small Python-dispatched matrix-vector operations;
- keep global sparse matrices out of the primary exact operator unless measurement shows a backend benefits from them;
- make masks, truncated pitch counts, and constraints shape-stable when possible.

Fusion is a means to reduce memory traffic and launch overhead. A fused computation that creates a very large live intermediate or prevents reuse may be slower and must be rejected by measurement.

### 9.6 Solver hierarchy

DKX uses a documented solver hierarchy rather than one method for every equation.

#### Route S1: exact structured block elimination

Use when the discrete operator is block tridiagonal in the chosen pitch/modal ordering and the route passes its memory estimate.

Required properties:

- exact or tolerance-controlled block-Thomas/Schur elimination;
- factor reuse for multiple right-hand sides and transpose solves;
- full-factor, Schur-LU-only, generated-off-diagonal, checkpointed, and streamed storage options selected by memory model;
- low-mode-only output option when moments do not require the full distribution;
- batched dense block operations;
- singularity and pivot diagnostics;
- double-precision true-residual certification even if internal preconditioner factors use lower precision;
- automatic fallback when the structured factorization is ill-conditioned or its assumptions are violated.

The sequential Legendre sweep should first be optimized through fused/generated blocks, batched dense kernels, memory layout, and factor reuse. Parallel cyclic reduction is evaluated only if the sweep remains the dominant GPU or multi-device bottleneck.

#### Route S2: matrix-free flexible recycled Krylov

Use for full physics that breaks the exact structured form or when S1 is inadmissible.

The default candidate is right-preconditioned GCROT or FGMRES because the preconditioner may be approximate, variable, or rebuilt. The implementation must support:

- nonzero initial guesses;
- recycle spaces across related systems;
- fixed-linear and variable-preconditioner modes;
- memory-aware restart and recycle dimensions;
- CGS2 or another tested stable orthogonalization;
- periodic recomputation of the genuine residual to prevent recursive-residual drift;
- block or multiple-right-hand-side variants only when measured;
- explicit failure and fallback rather than returning a plausible unconverged state;
- transpose/adjoint solves with the same convergence contract.

If the preconditioner is proven fixed and linear, DKX should evaluate a non-flexible storage mode that avoids retaining a separate preconditioned basis, reducing Krylov memory from an additional `O(Nm)` term. If it is variable or nonlinear, flexible storage is required.

Pipelined FGMRES/GCR is a candidate for multi-device runs when collective reductions dominate. It is not a default merely because it is communication-hiding; numerical stability, true residual, shift selection, and actual overlap must be measured.

#### Route S3: physics-aware coarse preconditioner

The exact operator remains matrix-free while the preconditioner deliberately omits or coarsens expensive couplings in the spirit of SFINCS/PETSc.

Candidate simplifications include:

- lower angular, pitch, or speed resolution;
- limited Legendre coupling;
- simplified collisions that retain conservation-critical structure;
- selected magnetic-drift diagonal terms;
- frozen Phi1 or electric-field coefficients;
- block-diagonal species or speed approximations plus a small coupling correction;
- Schur or block factorizations for constraints and moments;
- multigrid or domain-decomposition cycles on a discretization for which smoothing is actually effective.

Every omitted term must have a named policy and a regression test. The preconditioner may change iteration count, runtime, and memory, never the converged solution.

#### Route S4: sparse direct referee

A host sparse LU route remains available for tiny and moderate cases, validation, debugging, and route comparison. It is not the production default for large JAX workflows.

- use an explicit sparse assembly only in this route;
- separate symbolic analysis, numerical factorization, and solve;
- record ordering, scaling, pivoting, fill, factor memory, and residual;
- reuse symbolic and numeric factors where valid;
- support transpose solves;
- refuse cases whose estimated fill exceeds the memory budget.

SciPy/SuperLU may provide the local referee. PETSc with MUMPS or SuperLU_DIST may be used in local benchmark campaigns but is not a DKX runtime dependency.

#### Route S5: experimental multigrid or alternate discretization

Current and future multigrid or pitch-collocation routes remain experimental until they converge over the production physics matrix and outperform S2. A smoother that looks effective on a toy operator is insufficient. Promotion requires all model families claimed, matched residuals, CPU/GPU results, and no hidden host callback bottleneck.

### 9.7 Analysis, factorization, solve, and reuse state

Every solver route exposes explicit phases:

1. `analyze`: shapes, sparsity/structure, ordering, memory, route admissibility, and sharding;
2. `build`: operator coefficients and reusable data;
3. `factor` or `precondition`: numerical setup;
4. `solve`: one or multiple right-hand sides;
5. `certify`: genuine residual and scientific checks;
6. `update`: decide what can be reused for the next related system.

A versioned `ReuseState` may contain:

- geometry and Boozer/Fourier transforms;
- grids and derivative operators;
- collision tables and Rosenbluth data;
- source and constraint data;
- structured factors or generated-block state;
- sparse symbolic ordering and numerical factors;
- coarse operators and preconditioner factors;
- initial distribution state;
- Krylov recycle vectors;
- root brackets and branch continuation information;
- compilation-family and executable identifiers.

Reuse is never implicit guesswork. Each state records a validity signature. The planner distinguishes:

- same operator, new right-hand side: reuse factors and preconditioner;
- same structure, changed coefficients: reuse symbolic analysis; decide whether to refactor or lag preconditioner;
- nearby coefficients: warm-start and optionally reuse recycle space;
- changed shapes or discrete physics: invalidate the affected state;
- changed branch: preserve evidence but do not blindly warm-start across a discontinuity.

Preconditioner lagging uses measured criteria such as parameter distance, iteration growth, residual stagnation, or maximum reuse count. Rebuilding too often wastes setup; never rebuilding can explode iterations.

### 9.8 Mixed precision and numerical accuracy

Stable scientific outputs default to float64. Lower precision is allowed only in a controlled component:

- float32 or another lower precision for preconditioner factors, coarse solves, smoothers, or selected dense updates;
- float64 exact operator, right-hand side, solution accumulation, moments, and final residual;
- iterative refinement or residual correction when a lower-precision direct factor is used;
- precision policy recorded in `Case`, `Result`, and benchmark metadata;
- automatic fallback to float64 when convergence, pivot, or residual criteria fail.

GPU tensor-core or TF32 behavior must not be enabled silently. The plan should evaluate whether lower-precision preconditioners offset weak FP64 throughput on the maintained NVIDIA GPU, but promotion requires matched observables and reliable convergence over the full benchmark family.

### 9.9 Memory architecture

Memory is designed, estimated, and gated before execution.

Required memory model components:

- immutable geometry and coefficient state;
- exact operator working set;
- right-hand sides and outputs;
- structured bands, Schur factors, pivots, and checkpoints;
- coarse-preconditioner state;
- Krylov bases, Hessenberg data, recycle spaces, and temporary vectors;
- batch replication and sharding;
- compilation and backend overhead measured separately from modeled arrays;
- output buffers and distribution-function retention.

Rules:

- route selection uses **available** host or device memory, not installed memory;
- default admission budget is conservative and configurable in the case;
- a route predicted to exceed its budget is rejected before allocation;
- generated or streamed blocks replace stored bands where profitable;
- checkpointing trades recomputation for memory only when the total workflow improves or the alternative cannot run;
- Krylov restart and recycle sizes are automatically bounded by memory;
- batch size is selected by a memory estimator and can shrink dynamically between compilation families;
- buffer donation is used at safe ownership boundaries and tested for actual peak-memory reduction;
- reverse-mode differentiation does not retain iteration histories of implicit solves;
- `jax.checkpoint`/`remat` is considered only for explicitly differentiated non-implicit work with measured memory benefit;
- full distribution functions are optional outputs; moments-only workflows do not retain unnecessary high-mode state;
- out-of-core factors or host offloading are research fallbacks, not a substitute for a sound in-core algorithm.

Every memory failure becomes a small regression test for admission or fallback logic.

### 9.10 CPU execution

The named laptop CPU is the release reference. The CPU path must be first-class, not a GPU fallback.

CPU requirements:

- one explicit `cores` setting controls JAX/XLA and any owned thread pools before backend initialization;
- avoid oversubscription between XLA, BLAS, Python workers, and external validation codes;
- benchmark 1, 2, 4, and available physical-core counts for each production family;
- prefer batched BLAS-3 work in structured block solves;
- compare process-level independent-case parallelism against JAX sharding and vectorization;
- keep small cases fast by avoiding unnecessary accelerator-style batching, compilation families, and large runtime setup;
- report import and JAX runtime floor separately from DKX allocations;
- use deterministic core affinity or document when the operating system prevents it;
- choose the release default from the laptop benchmark matrix rather than `os.cpu_count()` alone.

The CLI may recommend a core count based on measured policy, but it reports the choice and allows override.

### 9.11 GPU and multi-device execution

The maintained NVIDIA GPU lane has three milestones.

#### Milestone G1: one-device throughput

- keep geometry, operator state, solve, and moments on device;
- eliminate repeated host transfers and callbacks;
- batch enough surfaces, electric-field points, or right-hand sides to reach useful occupancy;
- separate compile and transfer time from solve time;
- compare float64 and controlled mixed-precision preconditioners;
- profile kernel launch overhead, memory bandwidth, dense-block throughput, and sequential scans;
- use persistent compilation cache for repeated CLI and optimization workflows.

#### Milestone G2: independent-work sharding

- create a named JAX device mesh;
- shard case, surface, electric-field, collisionality, or right-hand-side batch axes explicitly;
- replicate only small immutable state;
- record array sharding in the result;
- use deterministic gathering and output ordering;
- require high parallel efficiency for sufficiently large independent batches before calling the route stable.

#### Milestone G3: single-solve sharding

- shard operator rows or another mathematically justified axis;
- quantify all-reduce and all-gather cost in Krylov orthogonalization, moments, and constraints;
- evaluate communication-avoiding/pipelined Krylov or distributed block algorithms;
- retain a one-device fallback;
- require an end-to-end speedup and per-device memory reduction at matched accuracy.

A multi-GPU implementation that only divides a small batch or adds communication overhead is not a release feature.

### 9.12 JAX implementation standards

- `import dkx` does not initialize JAX runtime policy.
- `runtime.configure()` or CLI startup sets x64, cache, thread, device, and distributed options before first backend use.
- Hot functions are pure and operate on immutable PyTrees with arrays as dynamic leaves.
- Python objects, strings, paths, and open files do not cross JIT boundaries.
- `lax.scan`, `fori_loop`, `while_loop`, `vmap`, and bounded `lax.map` replace hot Python loops when measurement supports it.
- Static arguments are minimal and hash-stable.
- Closed-over large arrays are passed as arguments, not captured as compile-time constants.
- Every benchmark calls `.block_until_ready()` or an equivalent synchronization before stopping the timer.
- Transfer guard is enabled in development runs that claim device residency.
- Persistent cache configuration is explicit and its disk use is observable and cleanable.
- Device-memory profiles and XProf traces are captured for production bottlenecks, not on every ordinary run.
- `jax.lax.custom_linear_solve` or Solvax implicit primitives define derivatives through linear solves; nonlinear roots use implicit-function rules with branch-local contracts.
- Buffer donation, host offload, rematerialization, Pallas, CuTe, Triton, or FFI require a fallback and evidence.
- XLA flags are not hidden stable APIs. Any required flag is versioned, documented, and tested on the maintained JAX range.

### 9.13 Automatic route planning

`method = "auto"` is a transparent planner, not an opaque heuristic.

The planner consumes:

- equation and coupling structure;
- problem shape and batch axes;
- device and precision capabilities;
- available memory;
- requested outputs and whether the full distribution is needed;
- number of right-hand sides and related solves;
- differentiation requirements;
- benchmark-calibrated route models.

It returns an `ExecutionPlan` containing:

- compilation family;
- exact operator representation;
- solver and preconditioner;
- factor-storage policy;
- batch size and sharding;
- precision policy;
- reusable-state policy;
- predicted setup, solve, and memory ranges;
- rejected alternatives and reasons;
- fallback sequence.

The result records the plan. Users can inspect it with `dkx inspect --execution case.toml` before running and can override stable controls. Auto-routing policy changes are versioned and benchmarked.

### 9.14 Production admission gates for a numerical route

A solver, preconditioner, mixed-precision mode, sharding strategy, or custom kernel cannot become a default until it passes:

1. true primal residual on the full benchmark family;
2. true transpose/adjoint residual where differentiation is supported;
3. field-level or observable parity against the current accepted route;
4. conservation, reciprocity, variational, and branch checks applicable to the model;
5. cold compile, warm solve, repeated-workflow, and complete wall time;
6. peak host and device memory and admission-model accuracy;
7. CPU and GPU comparison where the route claims both;
8. derivative comparison where the route is differentiable;
9. deterministic failure and fallback behavior;
10. documentation of supported and unsupported model combinations.

A route may remain available as `experimental` after failing performance admission, but it is not advertised as a production improvement.

## 10. Testing, validation, and scientific evidence

### 10.1 Test organization

Keep `tests/` shallow and group by stable scientific owner:

```text
tests/
  conftest.py
  data/
    manifest.toml
    compact generated and independent references
  test_config.py
  test_geometry.py
  test_grids.py
  test_collisions.py
  test_operator.py
  test_solvers.py
  test_observables.py
  test_transport.py
  test_monoenergetic.py
  test_ambipolar.py
  test_autodiff.py
  test_parallel.py
  test_workflows.py
  test_cli_package.py
  test_validation.py
```

The exact number of files is not a target. Rules:

- parameterize repeated models, geometries, devices, and resolutions;
- share small readable builders rather than copying long decks;
- generate mathematical fixtures when possible;
- keep independent reference data compact, checksum-pinned, and attributable;
- do not create dated artifact directories inside `tests/`;
- do not test project-management prose, exact filenames, exact module counts, old branch names, or private campaign percentages;
- every fixed bug gets the smallest regression test that reproduces it;
- every orchestration bug also gets an installed-artifact end-to-end test when source-tree context contributed to the failure;
- no test exists solely to execute a line for coverage.

### 10.2 Coverage contract

- Measure line and branch coverage on the installed wheel, not an editable source checkout.
- Raise the gate in evidence-backed steps until both are at least 95 percent.
- Hardware-unavailable branches may be excluded only by narrow documented pragmas and must run in the maintained GPU lane.
- Generated compatibility code is not excluded merely because it is tedious.
- Remove dead, unreachable, duplicate, or compatibility-expired code before writing tests for it.
- Report coverage by domain so a high aggregate cannot hide an untested solver or normalization owner.

### 10.3 Evidence classes

Every stable capability has one or more of:

1. `identity`: algebraic, conservation, nullspace, symmetry, adjoint, or reciprocity property;
2. `manufactured`: known solution/source for discretization and convergence;
3. `asymptotic`: collisionality, axisymmetry, electric-field, mass-ratio, or geometry limit;
4. `literature`: published table, curve, scaling, or formula reproduced with matched conventions;
5. `independent_code`: result from a separately implemented code at a pinned commit;
6. `experiment`: measured data with explicit equilibrium and profile assumptions;
7. `performance`: time, memory, scaling, and failure result with matched accuracy and hardware provenance.

A frozen DKX output is a regression reference, not independent validation.

### 10.4 Required scientific gates

Where applicable, stable paths must test:

- collision-operator particle, momentum, and energy conservation;
- Maxwellian and collisional-invariant nullspaces;
- nonnegative entropy production and operator symmetry properties;
- streaming antisymmetry under the correct discrete inner product;
- operator linearity and transpose identities;
- bordered constraints and nullspace removal;
- Onsager symmetry in the model where it applies;
- DKES variational upper and lower bounds and convergence of their gap;
- Shaing-Callen, axisymmetric, Spitzer, trapped-fraction, and high-collisionality limits;
- banana, plateau, Pfirsch-Schlüter, `1/nu`, `sqrt(nu)`, and superbanana-related trends where the local model supports them;
- monoenergetic-to-thermal convolution consistency;
- convergence in angular, pitch, speed, and radial-surface interpolation resolution;
- all-root discovery, stability classification, branch continuation, branch creation/loss, and transition handling;
- bootstrap-current, conductivity, particle-flux, heat-flux, and flow benchmarks;
- impurity and interspecies momentum exchange;
- Phi1 and tangential-drift combinations within their supported scope;
- CPU/GPU, batched/unbatched, sharded/unsharded, and structured/Krylov equivalence;
- JVP/VJP dot products, transpose solves, finite differences, and branch-local root derivatives;
- physical/normalized unit round trips and radial-coordinate chain rules;
- native NetCDF and SFINCS HDF5 schema integrity.

### 10.5 Cross-code validation matrix

Local validation campaigns may build the following codes at pinned commits:

- SFINCS: matched full local and monoenergetic physics, output keys, Phi1, trajectory models, collisions, and bootstrap current;
- MONKES: monoenergetic coefficients and structured-solver performance in its supported equation;
- YANCC: matched full and monoenergetic equations, modern JAX solver behavior, GPU throughput, and memory;
- KNOSOS: low-collisionality bounce-averaged domain and tangential-drift behavior;
- NEO: axisymmetric transport and bootstrap quantities;
- BOOTSJ: reduced stellarator bootstrap current;
- PENTA: ambipolar-root and flow workflows from monoenergetic data;
- DKES where accessible: monoenergetic coefficients and variational conventions.

These executables are **not** CI requirements. The local campaign produces a compact evidence bundle containing:

- code URL, commit, build options, compiler, libraries, and patches;
- original and normalized inputs;
- stdout/stderr and convergence status;
- selected compact outputs;
- matching/conversion script;
- comparison tolerances and rationale;
- hardware and timing metadata;
- license and redistribution notes.

CI consumes only redistributable compact references and regenerates comparisons. A scheduled or pre-release local campaign refreshes them.

### 10.6 Capability and evidence registry

A machine-readable registry maps each public claim to:

- capability ID and status;
- equations and assumptions;
- source modules and public APIs;
- supported input combinations;
- mathematical and physical tests;
- external references and pinned commits;
- benchmark cases;
- accuracy and performance gates;
- documentation and examples;
- known limitations;
- last passing DKX commit and evidence bundle.

Documentation capability tables and release checklists are generated from this registry. They are not manually maintained in several places.

### 10.7 CI and validation lanes

#### Pull-request lane

Target: bounded, normally under ten minutes per required job.

- formatting and linting;
- type checks for public and risk-bearing internal contracts;
- built wheel and source-distribution installation outside the checkout;
- line and branch coverage;
- unit, mathematical, compact physics, CLI, and schema tests;
- warning-clean documentation build;
- fast synchronized performance and memory regression set;
- full-clone and artifact-size gates;
- workflow YAML linting.

#### Nightly lane

- larger convergence ladders;
- full canonical examples;
- repeated-workflow and cache benchmarks;
- profile and ambipolar branch tests;
- mutation/property tests where useful;
- current/minimum dependency matrices;
- performance trend recording.

#### Maintained GPU lane

Run on the office NVIDIA GPU:

- real full-physics solves;
- CPU/GPU and one-device/sharded equivalence;
- device-memory profiles and transfer guards;
- compile and warm throughput;
- mixed-precision preconditioner experiments;
- multi-device scaling if more than one GPU is available;
- gradients and transpose solves.

#### Local external-code lane

Run before major scientific promotion and release. It builds and executes the available comparison codes and writes a versioned evidence bundle. It is reproducible through scripts but not required for ordinary contributors.

#### Release lane

- all stable capability gates;
- frozen benchmark campaign on the named laptop CPU;
- maintained NVIDIA GPU campaign;
- current external-code evidence replay;
- wheel, source distribution, clean install, and published-PyPI smoke;
- all documentation examples;
- repository and package size;
- migration guide, changelog, citation, DOI, and evidence archive.

## 11. Performance and benchmark contract

### 11.1 Official hardware

The benchmark registry records exact hardware rather than labels such as “laptop” or “GPU.” At first execution, Codex must fill:

```text
CPU_RELEASE_MACHINE:
  model:
  physical_cores:
  logical_cores:
  memory:
  operating_system:
  BLAS:

GPU_RELEASE_MACHINE:
  model:
  memory:
  driver:
  CUDA:
  host_CPU:
  interconnect:
```

The laptop CPU is the source of headline release numbers. GPU results are a separate maintained table and must not replace the CPU baseline.

### 11.2 Benchmark families

The stable registry should include at least:

1. import and tiny analytic tokamak: runtime floor, compile count, package overhead;
2. axisymmetric pitch-angle-scattering surface: structured path and analytic limits;
3. axisymmetric full-Fokker-Planck multispecies surface;
4. three-dimensional monoenergetic surface over collisionality and electric field;
5. W7-X or equivalent pitch-angle structured production case;
6. W7-X or equivalent full-collision, magnetic-drift, finite-electric-field hard case;
7. Phi1 nonlinear case;
8. all-root electric-field scan;
9. complete radial profile with bootstrap current and branch continuation;
10. monoenergetic database and thermal convolution;
11. forward-response and reverse-adjoint gradients over increasing parameter count;
12. VMEX-coupled repeated objective evaluation;
13. one memory-stress case that requires generated, streamed, or checkpointed factors;
14. one batch large enough for GPU and multi-device throughput.
15. the established SFINCS scan-5 pattern over electric field and radii, with
    separate process startup, input discovery, geometry setup, compilation, and
    first-result latency so multi-minute pre-solve stalls cannot hide inside
    complete runtime.

Each family has `smoke`, `representative`, and `production` sizes. Smoke belongs in PR CI; representative in nightly; production in release/local campaigns.

### 11.3 Required metrics

Every recorded run includes:

- status: success, nonconvergence, timeout, memory refusal, OOM, crash, or unsupported;
- scientific model and equation options;
- input checksum and output checksum;
- resolution and number of unknowns;
- true residual and requested tolerance;
- selected observables and error against reference;
- import, parse, geometry, normalization, trace, compile, setup, factor, solve, moments, output, plot, and total time;
- first-run and warm-run time;
- number of traces, compilations, cache hits, operator applications, iterations, restarts, preconditioner builds, and factor rebuilds;
- peak process RSS, modeled host arrays, peak device memory, and estimated memory;
- host-device transfer count/volume when available;
- core/thread count, device mesh, sharding, batch size, and precision;
- software and hardware versions;
- at least three repetitions for stable short cases and robust summary statistics.

### 11.4 Timing protocol

- Start a fresh process for cold import and compile measurements.
- Separate compilation from first execution using lowering/compilation APIs where practical.
- Synchronize device work before every stopped timer.
- Warm up before measuring steady-state execution.
- Control CPU core count and avoid oversubscribed BLAS.
- Disable interactive progress and profiler collection in ordinary timing runs.
- Keep persistent-cache state explicit: empty, warm, or disabled.
- Record background-load limitations; repeat noisy results rather than presenting a single favorable run.
- Report median and dispersion, not only the minimum.
- Never compare a warm DKX run to a cold reference run without labeling both.

### 11.5 Fair comparison protocol

Cross-code performance comparisons must match:

- physical equation and local/global approximation;
- collision operator and momentum conservation;
- trajectory model and electric field;
- geometry and surface;
- species, profiles, and normalization;
- resolution or demonstrated observable convergence;
- solver tolerance and **true** residual;
- requested outputs;
- hardware allocation, core count, and precision;
- cold/warm status and repeated-solve context.

If exact matching is impossible, present the difference before the timing and do not compute a misleading speedup.

### 11.6 Performance regression gates

The PR benchmark set uses a baseline window rather than a single noisy value.

- median complete runtime regression greater than 10 percent fails unless the pull request contains an approved reason and compensating scientific or memory benefit;
- peak-memory regression greater than 10 percent fails unless required by a documented capability and still within the admission budget;
- new compilations or transfers fail when the semantic shape family is unchanged;
- iteration growth greater than 20 percent fails unless setup time decreases enough to improve complete workflow time;
- any observable, residual, or certificate regression fails regardless of speed;
- reductions greater than 10 percent are recorded but must reproduce before becoming a public claim.

Thresholds may be adjusted after enough historical data exist, with the decision recorded here.

### 11.7 Repeated-workflow gates

For scans, profiles, roots, and optimization, report both one solve and the complete sequence.

Required acceptance:

- no per-case recompilation within one compilation family;
- geometry and normalization are not repeated unnecessarily;
- factor/preconditioner reuse is visible in metadata;
- warm starts do not cause missed roots or wrong branches;
- the sequence reaches the same answers as independent cold solves;
- throughput improves over naïve repeated calls at matched peak memory;
- resuming a partial scan does not recompute completed cases.

### 11.8 Parallel scaling gates

For independent-work sharding, measure 1, 2, and available devices with enough work per device. A stable claim normally requires:

- at least 80 percent parallel efficiency on two devices for a sufficiently large independent batch;
- at least 65 percent efficiency on four devices when four are maintained;
- no unexplained result differences;
- bounded per-device memory and deterministic output.

These are admission targets, not promises for tiny workloads.

For single-solve sharding, do not set a generic efficiency target before implementation. Promotion requires an end-to-end speedup of at least 1.3 times over the best one-device route on a production case, a lower per-device memory footprint, and a communication breakdown. If it does not meet that bar, keep it experimental and prefer workload sharding.

### 11.9 Benchmark artifacts

Benchmark scripts write compact JSON or NetCDF records with a schema. Figures and tables are generated from records, never hand-entered. Production raw outputs, traces, and external-code builds live in release assets or an archival evidence bundle. The repository stores only small summaries needed for regression and documentation.

## 12. Documentation, examples, and media

### 12.1 Documentation architecture

Use MyST Markdown, Sphinx, and Furo. Keep extensions limited to citations, API extraction, code copying, tabs/admonitions, and required mathematical support.

Organize by reader need:

```text
Get started
  What DKX computes
  Installation
  First tokamak case
  First stellarator case
  Reading a result

Tutorials
  Monoenergetic coefficients
  Full multispecies transport
  Ambipolar roots
  Whole radial profile
  Bootstrap current
  Convergence and certificates
  Gradients
  VMEX coupling

How-to guides
  Convert a SFINCS namelist
  Build a TOML scan
  Resume a scan
  Choose resolution
  Control CPU/GPU execution
  Inspect performance
  Export SFINCS HDF5
  Add a geometry/profile adapter

Physics and mathematics
  Ordering and drift-kinetic equation
  Coordinates and normalization
  Trajectory models
  Collision operators
  Constraints and nullspaces
  Fluxes, flows, conductivity, and bootstrap current
  Monoenergetic coefficients and thermal convolution
  Ambipolarity and root stability
  Phi1 and impurities
  Model validity and finite-orbit-width limits

Numerical methods
  Phase-space discretization
  Structured block elimination
  Krylov and recycled Krylov methods
  Physics preconditioners and multigrid
  Nonlinear and root solves
  Implicit differentiation
  Batching, sharding, and memory
  Convergence and certificates

Reference
  TOML/JSON schema
  CLI
  Python API
  NetCDF result schema
  SFINCS compatibility
  Capability matrix
  Validation registry
  Performance tables
  Limitations

Developer guide
  Source ownership
  Testing and evidence
  Benchmark protocol
  Release process
  Contributing a model or adapter
```

The original SFINCS notes and source correspondence are retained as cited source material, not pasted as an archaic parallel manual. Rewrite derivations in DKX notation, connect equations to implementation owners, and state assumptions and supported options.

### 12.2 Writing rules

- Start pages with the user question or physical purpose, not project history.
- Define symbols before use and keep notation consistent.
- State the equation, approximation, normalization, discretization, solver, outputs, and limitations.
- Cite primary papers and code documentation close to the claim.
- Use measured numbers with hardware, commit, resolution, tolerance, and evidence IDs.
- Avoid generic superlatives, filler summaries, fake quotations, repetitive “robust/flexible/powerful” language, and mechanical section conclusions.
- Do not describe a research route as complete because code exists.
- Keep internal branch names, dated campaigns, and agent instructions out of user navigation.
- Prefer physical route names over historical tier numbers in user-facing text.
- Every code block in getting-started and tutorials is executed in CI.
- Every figure has a generating script or evidence record.

### 12.3 Canonical examples

Keep no more than ten primary examples. Initial set:

1. circular tokamak surface in physical units;
2. stellarator surface from VMEC;
3. monoenergetic collisionality/electric-field scan;
4. full multispecies transport and bootstrap current;
5. all-root ambipolar electric-field scan;
6. complete radial profile with branch continuation;
7. convergence and validity certificate;
8. forward and reverse derivative verification;
9. VMEX-coupled bootstrap objective;
10. one advanced Phi1 or impurity workflow after stable promotion.

Every example:

- has editable inputs near the top;
- uses the same public API or one TOML file;
- prints structured progress and a concise summary;
- saves one native result;
- produces one scientifically useful plot;
- states expected laptop runtime and approximate memory;
- identifies the physical regime and validation source;
- avoids `main()`, custom argument parsing, and helper-function forests;
- is small enough to understand but does not hide all inputs behind a convenience call.

Reference-code parity drivers, paper-scale benchmarks, and profiler tools belong under `tools/` or the evidence bundle, not `examples/`.

### 12.4 Media and repository policy

- Prefer SVG for diagrams and line art and compressed WebP for raster figures.
- Use compressed WebM for movies; provide a static fallback.
- Strip unnecessary metadata and whitespace.
- Keep generated scientific outputs out of Git.
- Put large public equilibria and benchmark bundles in release assets with checksums and a fetch command.
- Maintain `docs/media_manifest.toml` containing source script, output file, dimensions, byte size, and compression settings.
- CI enforces per-file and aggregate budgets.
- Never commit compiled binaries, object files, module files, virtual environments, JAX caches, or external-code builds.

The README release figure is an evidence-backed, reviewer-readable overview,
not a decorative collage. It must show the native TOML, short Python, and
SFINCS-compatibility workflows; a matched DKX/SFINCS result; cold and warm
runtime plus peak memory; and a reproducible code-footprint comparison (files,
source lines, input lines, and output-adapter lines) whose counting rules,
commits, exclusions, and generating script are visible.

## 13. Packaging and dependency policy

### 13.1 Packaging

- move to `src/dkx`;
- single-source the version;
- require Python 3.11+;
- build wheel and source distribution;
- install and test each artifact in an isolated environment outside the checkout;
- verify the imported module resolves inside `site-packages`;
- verify every declared package-data file is present and no undeclared data are required;
- run a scientifically nontrivial CLI smoke from the installed artifact;
- use current SPDX license metadata, project URLs, and `CITATION.cff`;
- keep trusted PyPI publishing;
- smoke-test the actual published artifact;
- add conda-forge after the DKX 3 API and dependency set stabilize;
- publish changelog, migration guide, release checklist, DOI, and evidence bundle.

### 13.2 Size gates

Release CI measures:

```text
fresh_full_clone_bytes
working_tree_bytes
git_directory_bytes
sdist_bytes
wheel_bytes
installed_dkx_distribution_bytes
tracked_docs_media_bytes
largest_tracked_files
```

Each of the first five DKX-owned deliverables is below 20 MiB. The planning phase must determine whether another history rewrite is required; do not rewrite history repeatedly for small gains. Remove generated and binary content first, compress media, and move large references to release assets.

### 13.3 Runtime dependencies

Provisional core dependencies:

- `jax`;
- `numpy`;
- `scipy`;
- `solvax`;
- `h5py`;
- `netCDF4`;
- `matplotlib`;
- `rich`.

Rules:

- use an unversioned dependency when DKX relies only on long-stable behavior;
- retain a minimum version when a required API or correctness fix first appears there;
- do not add speculative upper bounds;
- store exact reproducible environments in CI constraints or evidence manifests, not runtime metadata;
- test the declared minimum compatible set and a current set before release;
- keep external codes, PETSc, MUMPS, SuperLU_DIST, xarray, pandas, Dask, Ray, Equinox, Lineax, and optimizer frameworks out of core unless a future accepted contract requires them;
- avoid proliferating user installation extras. Core examples use the normal install; integration documentation names external packages separately.

The dependency budget includes import time, wheel size, transitive complexity, maintenance burden, and security updates, not only package count.

## 14. Scientific workflow roadmap

### 14.1 Flagship A: whole-profile ambipolar transport and bootstrap current

This is the first DKX 3 flagship and primary product differentiator.

Required capabilities:

- read VMEX/VMEC/Boozer geometry and physical species profiles;
- choose and interpolate radial surfaces with documented conventions;
- group surfaces into compilation families;
- reuse geometry, operator state, factors, preconditioners, and recycle spaces;
- compute particle and heat fluxes, flows, conductivity, classical terms, and bootstrap current;
- find all ambipolar roots on every surface;
- continue ion, electron, and unstable branches radially;
- detect root creation, loss, merger, crossing, and physical branch transitions;
- report the branch-selection policy and alternatives;
- couple resolution ladders to uncertainty in roots and observables;
- shard surfaces and electric-field points over available devices;
- save one resumable profile result and produce a coherent radial summary figure;
- expose branch-local derivatives and nonsmooth transition warnings.
- accept VMEC pressure/density/temperature profiles represented by `sum_atan`
  and other supported profile families without silently dropping electric
  field or flux panels; plots must distinguish unavailable physics from parser
  or solve failure.

### 14.2 Flagship B: monoenergetic coefficients and databases

- standard DKES/ICNTS-style coefficient conventions;
- collisionality and electric-field scans;
- variational upper/lower bounds and gap;
- fast structured solver with factor reuse and low-mode retention;
- thermal convolution to species transport matrices;
- compact portable database schema;
- comparisons with DKES, MONKES, SFINCS, and YANCC;
- high-throughput CPU/GPU batching and sharding;
- use as a fast model and as input to PENTA-compatible workflows.

### 14.3 Flagship C: full multispecies local kinetics

- pitch-angle and full linearized Fokker-Planck collisions;
- momentum, particle, and energy conservation;
- multiple trajectory and magnetic-drift models;
- Phi1 within validated combinations;
- impurities and trace-species diagnostics;
- full transport matrices, flows, conductivity, and bootstrap current;
- robust general solver and bounded-memory preconditioner;
- clear comparison to reduced and monoenergetic models.

### 14.4 Flagship D: convergence and validity certificates

Every result should be able to explain why it is or is not trustworthy through:

- true solver and transpose residuals;
- phase-space refinement trends;
- variational gap where available;
- conservation and reciprocity defects;
- derivative certificate;
- root separation, slope, and branch events;
- finite-orbit-width and radial-locality indicators;
- collisionality regime;
- electric-field resonance indicators;
- tangential-drift and Phi1 warnings;
- collision-model and impurity warnings;
- comparison to a reduced or analytic model where meaningful.

### 14.5 Flagship E: optimization and code coupling

Define small in-memory protocols:

```text
GeometryProvider -> FluxSurfaceGeometry
ProfileProvider -> SpeciesProfiles
NeoclassicalModel -> Result
Objective -> value/residuals plus derivative and certificate
```

First-class targets:

- VMEX geometry and boundary parameters;
- Boozer transformations without intermediate files on the primary path;
- bootstrap-consistent equilibrium iteration;
- optimization of transport, bootstrap current, ambipolar-root behavior, electron-root robustness, and certificate margins;
- profile-evolution codes through batched flux interfaces;
- STELLOPT, SIMSOPT, DESC, and IMAS-style adapters when maintainers and users justify them.

The core remains optimizer-neutral. DKX supplies values, residuals, JVPs, VJPs, and certificates; external frameworks own optimization loops.

### 14.6 Longer-term research lanes

Not DKX 3 release blockers:

- radially global or finite-orbit-width corrections;
- certified bounce-averaged fast models;
- advanced collision operators beyond the validated SFINCS-equivalent set;
- uncertainty propagation and Hessian-vector products;
- surrogate models trained on versioned DKX data;
- time-dependent radial-electric-field or transport coupling;
- multi-host single-solve JAX;
- block-low-rank or out-of-core production factorization;
- custom accelerator kernels.

Each begins `experimental` and must satisfy the same scientific and performance admission gates before appearing in the stable feature list.

## 15. Implementation phases and work items

Performance, testing, documentation, and evidence are part of every phase. The dedicated performance phases deepen those contracts; they do not permit earlier phases to regress them.

### Phase P0: establish the single authority and freeze DKX 2 evidence

Status: ready for planning pull request

Work items:

- `P0.1` Create `plan.md` on `rj/dkx3-plan` with this content.
- `P0.2` Record that PR #8 is merged/closed and import its durable `plan_final.md` requirements into this file.
- `P0.3` Search open and closed pull requests, branches, docs, issues, and tests for competing plans; reconcile any unique requirement and mark the old source superseded.
- `P0.4` Remove the test that prohibits `plan.md` and replace obsolete roadmap prose with a pointer to this file or historical release notes.
- `P0.5` Freeze a DKX 2.3 scientific baseline: representative inputs, native and SFINCS-compatible outputs, solver traces, derivatives, runtime, memory, and package sizes.
- `P0.6` Create the capability registry with initial status and evidence gaps.
- `P0.7` Create the benchmark schema and identify the exact laptop and NVIDIA GPU hardware.
- `P0.8` Record source ownership, public API, file/dependency size, compile-family count, and current CI duration.

Acceptance criteria:

- only `plan.md` is authoritative in `main`;
- no open planning pull request remains competing with it;
- PR #8 and old roadmap content are referenced only as history;
- baseline artifacts are reproducible and checksum-pinned;
- stable-candidate and experimental capabilities are explicit;
- no physics behavior changes in this phase;
- full test and documentation suites pass.

### Phase P1: make releases and size trustworthy

Status: completed for ordinary release artifacts; accepted D012 full-history rewrite remains a coordinated-maintainer action

Work items:

- `P1.1` Land or supersede PR #67 with a clean-wheel test that runs outside the checkout and checks nontrivial scientific output.
- `P1.2` Land or supersede PR #66 so `main` is green before modernization begins.
- `P1.3` Build and test wheel and source distribution in isolated environments.
- `P1.4` Add package-data completeness and undeclared-data gates.
- `P1.5` Add a post-PyPI smoke test of the actual published artifact.
- `P1.6` Add action/workflow linting and required-job aggregation.
- `P1.7` Add full-clone, wheel, source-distribution, installed-package, and media size measurements.
- `P1.8` Remove or release-host large fixtures and media until all DKX-owned deliverables are below 20 MiB.
- `P1.9` Single-source version and release metadata.

Acceptance criteria:

- source checkout cannot mask a broken wheel;
- canonical CLI and Python workflows run from clean installed artifacts;
- package data needed at runtime are present in both wheel and source distribution;
- the PyPI artifact is tested after publication;
- all accepted size gates pass;
- no external validation code is needed to install or test DKX.

### Phase P2: introduce native `Case`, `Result`, and CLI contracts

Status: in progress (`P2.1`, `P2.2`, the deterministic-ID/bounded-count portion of `P2.3`, and the analytic and VMEC `P2.4`/`P2.7` native profile slices are merged; native ambipolar profile execution is in progress)

Work items:

- `P2.1` Implement immutable canonical case models and one normalization boundary.
- `P2.2` Implement versioned TOML and JSON readers, validation, and schema generation.
- `P2.3` Implement declarative scans, deterministic case IDs, resume metadata, and preflight case/memory estimates.
- `P2.4` Implement native `Result` and versioned NetCDF writer/reader.
- `P2.5` Implement permanent SFINCS namelist and HDF5 adapters.
- `P2.6` Implement the argparse/Rich command surface and structured progress events.
- `P2.7` Route one tokamak, one stellarator, one transport-matrix, and one ambipolar workflow through the new contracts without changing the accepted numerical kernels.
- `P2.8` Measure copies, compile families, setup time, and complete runtime so the new contracts do not add hidden overhead.

Acceptance criteria:

- native workflows do not pass through a namelist internally;
- normalization is tested in physical and normalized units;
- NetCDF round trip and SFINCS HDF5 compatibility pass;
- scans need no auxiliary Python file;
- progress does not introduce hot-kernel host callbacks;
- baseline observables, residuals, derivatives, runtime, and memory remain within accepted bounds.

### Phase P3: make runtime explicit and migrate source ownership

Status: planned

Work items:

- `P3.1` Move to `src/dkx` and make installed-artifact testing the default.
- `P3.2` Make `import dkx` inert.
- `P3.3` Implement explicit runtime configuration for x64, threads, cache, devices, and distributed startup.
- `P3.4` Introduce `CompilationFamily`, `ExecutionPlan`, and `ReuseState` without changing numerical answers.
- `P3.5` Migrate configuration/normalization, geometry, grids, collisions, operator, moments, solvers, workflows, and I/O one vertical slice at a time.
- `P3.6` Remove old owners, transitional packages, import shims, and structural tests as each slice completes.
- `P3.7` Move generic reusable algorithms to Solvax only when another code can use them and DKX-specific physics does not leak into the API.

Acceptance criteria:

- importing DKX has no process-wide effect;
- every supported workflow uses one canonical owner per behavior;
- no new and old stack coexist indefinitely;
- compilation-family and reuse metadata are visible;
- source migration does not regress scientific, runtime, memory, or packaging gates.

### Phase P4: consolidate testing and reach 95 percent meaningful coverage

Status: planned

Work items:

- `P4.1` Replace inventory, branch-name, dated-artifact, and plan-prohibition tests with scientific and contract tests.
- `P4.2` Consolidate fixtures and remove duplicate or generated bulk.
- `P4.3` Add conservation, nullspace, reciprocity, manufactured, convergence, and unit/normalization gates.
- `P4.4` Add installed-wheel API, CLI, NetCDF, HDF5, and resume tests.
- `P4.5` Add property or metamorphic tests for schema, scan, route, and cache logic where exact oracles are difficult.
- `P4.6` Raise installed-package line and branch gates through 85, 90, and 95 percent.
- `P4.7` Generate documentation capability tables from the evidence registry.

Acceptance criteria:

- line and branch coverage are at least 95 percent;
- test count and fixture bytes are lower or deliberately justified;
- every stable capability has non-regression evidence beyond same-code golden files;
- CI remains within its wall-time and memory budgets;
- no production route is untested because it is difficult.

### Phase P5: optimize data layout, compilation, and the exact structured path

Status: planned

Work items:

- `P5.1` Profile operator, collision, moment, and structured-solver data layouts on the laptop CPU and NVIDIA GPU.
- `P5.2` Remove accidental recompilation, large captured constants, repeated geometry construction, and host transfers.
- `P5.3` Inspect XLA fusion and live buffers; fuse profitable coefficient/stencil operations without increasing peak memory.
- `P5.4` Batch dense angular-block operations and multiple right-hand sides.
- `P5.5` Implement and benchmark factor-storage policies: full bands, Schur-only, generated off-diagonals, streamed, and checkpointed.
- `P5.6` Retain only requested distribution modes in moments-only workflows.
- `P5.7` Add safe buffer donation and bounded batch selection where measured.
- `P5.8` Evaluate lower-precision structured/preconditioner factors with double-precision residual correction.
- `P5.9` Evaluate parallel cyclic reduction only if the sequential block sweep remains a measured production bottleneck.
- `P5.10` Profile and remove scan-5 startup latency across electric-field and
  radial batches, including repeated input discovery, geometry transforms,
  compilation-family misses, and per-point process/JAX initialization.

Acceptance criteria:

- exact structured cases preserve true residual and observables;
- best storage route is selected by an accurate memory model;
- complete structured workflows improve or are no worse on both official machines;
- factor and compilation reuse is visible in repeated scans;
- no custom kernel or parallel block algorithm is promoted without end-to-end benefit.

### Phase P6: optimize the general solver and bounded-memory preconditioning

Status: planned

Work items:

- `P6.1` Define exact-operator and coarse-preconditioner contracts term by term.
- `P6.2` Benchmark GCROT, FGMRES, fixed-linear storage, orthogonalization, restart, recycle, and true-residual refresh policies.
- `P6.3` Improve the physics preconditioner using measured SFINCS/PETSc and YANCC lessons.
- `P6.4` Separate analysis, setup/factorization, solve, certify, and update phases.
- `P6.5` Implement preconditioner lagging/rebuild criteria for nearby systems.
- `P6.6` Add memory-aware restart/recycle and fail-before-OOM routing.
- `P6.7` Maintain a sparse direct referee and local PETSc/MUMPS/SuperLU_DIST comparison path.
- `P6.8` Evaluate mixed-precision preconditioners, iterative refinement, block Krylov, pipelined Krylov, and low-rank approximations as experiments.
- `P6.9` Certify forward and transpose solves and implicit derivatives.

Acceptance criteria:

- all stable full-physics benchmark cases complete or fail early with an actionable reason;
- true residual and field-level parity pass;
- peak memory remains within the admission budget;
- the selected default minimizes complete workflow time, not only iteration count;
- failed and slower routes remain in the benchmark record;
- experimental methods stay out of stable defaults until admission gates pass.

### Phase P7: batch, shard, and scale CPU/GPU workloads

Status: planned

Work items:

- `P7.1` Implement independent-case and surface/electric-field batch planning.
- `P7.2` Implement explicit device meshes and sharding for independent batch axes.
- `P7.3` Add bounded chunking and load balancing for heterogeneous iteration counts.
- `P7.4` Measure CPU process parallelism, CPU-device sharding, and vectorized batching without oversubscription.
- `P7.5` Establish one-GPU throughput and residency, then independent-work multi-device scaling.
- `P7.6` Measure communication and evaluate single-solve row/domain sharding only after workload sharding is mature.
- `P7.7` Add CPU/GPU, sharded/unsharded, and gradient equivalence tests.
- `P7.8` Add strong/weak scaling records and route recommendations to `doctor` and `inspect`.

Acceptance criteria:

- users request `device = "auto"` or a simple strategy rather than rewriting examples;
- independent-work sharding meets the efficiency gates on maintained hardware;
- batch size obeys the memory budget;
- GPU and sharded results match accepted precision-aware tolerances;
- single-solve sharding is stable only if it beats the best one-device route and reduces per-device memory.

### Phase P8: deliver the whole-profile flagship

Status: planned

Work items:

- `P8.1` Implement physical profile ingestion and radial surface scheduling.
- `P8.2` Implement robust all-root search with adaptive brackets and batched evaluations.
- `P8.3` Implement radial branch continuation, classification, and branch-event detection.
- `P8.4` Couple convergence ladders to root and observable uncertainty.
- `P8.5` Produce flux, flow, conductivity, bootstrap, electric-field, and certificate profiles.
- `P8.6` Make the workflow resumable and shard surfaces/electric-field points.
- `P8.7` Validate at least one tokamak and two stellarator families against independent references.
- `P8.8` Expose branch-local derivatives and nonsmooth-event warnings.
- `P8.9` Add regression cases for `dkx wout_*.nc` with `sum_atan` and each
  supported VMEC profile representation, requiring electric-field and flux
  panels when the physical inputs and solves are available.

Acceptance criteria:

- one TOML file produces a complete, resumable profile;
- all roots and branch decisions are visible;
- root and observable convergence are quantified;
- factor, compilation, warm-state, and recycle reuse are reported;
- the profile result feeds VMEX or a transport code without parsing console output;
- runtime and memory are competitive with a naïve collection of independent surface solves.

### Phase P9: rebuild documentation, examples, and release UX

Status: planned

Work items:

- `P9.1` Establish MyST/Sphinx/Furo and the reader-oriented navigation.
- `P9.2` Rewrite installation, first run, inputs, outputs, CLI, and troubleshooting.
- `P9.3` Rewrite physics and numerical documentation from the original SFINCS notes, primary papers, and DKX implementation.
- `P9.4` Build the canonical example ladder and execute it in CI.
- `P9.5` Add performance, memory, sharding, convergence, and validity guides.
- `P9.6` Generate API, schema, capability, and validation reference from canonical sources.
- `P9.7` Compress media and enforce the manifest and size budgets.
- `P9.8` Add migration documentation from DKX 2 and SFINCS decks.
- `P9.9` Replace primary tier-number wording with physical solver-route names
  and add the regenerable README workflow/result/performance/code-footprint
  comparison figure.

Acceptance criteria:

- a new user completes the golden workflows from documentation;
- every example states expected laptop runtime and memory;
- equations, algorithms, inputs, outputs, validity, and evidence are findable;
- no internal campaign clutter appears in user navigation;
- all links, citations, snippets, media, and size gates pass.

### Phase P10: coupling, advanced workflows, and DKX 3 release

Status: planned

Work items:

- `P10.1` Stabilize VMEX in-memory geometry, profile, and objective protocols.
- `P10.2` Demonstrate bootstrap-consistent equilibrium iteration.
- `P10.3` Demonstrate transport, bootstrap, and ambipolar optimization with certified gradients.
- `P10.3a` Run downstream VMEX optimization gates for low bootstrap current
  and QI/electron-root robustness, recording primal, gradient, cold/warm JIT,
  peak-memory, and missed-branch checks.
- `P10.4` Promote validated monoenergetic, Phi1, impurity, and certificate workflows.
- `P10.5` Add small adapters for priority external ecosystems without core dependency growth.
- `P10.6` Publish PyPI and conda-forge packages, DOI, citation, migration guide, and evidence bundle.
- `P10.7` Establish governance, support expectations, and contribution routes for new validation cases and devices.

Acceptance criteria:

- primary VMEX coupling needs no intermediate file;
- values, residuals, derivatives, and certificates have stable contracts;
- adapters remain isolated from core dependencies;
- PyPI and conda installations run the same canonical examples;
- all headline scientific and performance claims are reproducible from the release evidence bundle.

## 16. Recommended pull-request sequence

After the planning PR, the expected sequence is:

1. `P0`: authoritative plan, prior-plan retirement, baseline registry;
2. `P1`: clean wheel/source distribution and size gates;
3. `P2`: native `Case` and normalization boundary;
4. `P2`: native `Result`/NetCDF and SFINCS adapters;
5. `P2`: CLI, progress events, declarative scans, and resume;
6. `P3`: inert import and explicit runtime configuration;
7. `P3`: `src/` migration and compilation/execution state;
8. `P4`: test consolidation and 95-percent installed-package gates;
9. `P5`: data layout, compilation reuse, and structured-solver optimization;
10. `P6`: general Krylov/preconditioner and bounded-memory routing;
11. `P7`: one-GPU batching and independent-work sharding;
12. `P8`: whole-profile ambipolar workflow;
13. `P9`: complete documentation/example rebuild;
14. `P10`: VMEX coupling, advanced promotion, and DKX 3 release.

Some items may split into a short series when a mechanical migration is too large, but each pull request must leave a useful coherent state. Do not create dozens of micro-PRs that each add scaffolding without retiring behavior.

## 17. Codex operating contract

Codex receives this file as its primary instruction. It should proceed as follows.

### 17.1 At the start of every work session

1. Read all of `plan.md`, especially current status and ledger.
2. Fetch `main`, open pull requests, and the work-item branch.
3. Confirm the recorded baseline and note repository drift.
4. Inspect the relevant current source, tests, docs, and recent commits before editing.
5. Identify the smallest coherent vertical slice and its acceptance gates.
6. Record a concise start entry in the branch copy of the ledger.

### 17.2 External-code use

Codex may clone, build, and run SFINCS, MONKES, YANCC, KNOSOS, NEO, BOOTSJ, PENTA, DKES, PETSc, MUMPS, and SuperLU_DIST locally when useful.

Rules:

- pin exact commits and record build configuration;
- do not modify upstream repositories except in disposable local branches;
- do not copy source with incompatible licensing into DKX;
- derive algorithms from papers and documentation and implement them independently;
- keep external executables out of DKX package metadata and ordinary CI;
- store only compact redistributable reference results in DKX;
- state when an equation or normalization cannot be matched exactly;
- include unsuccessful runs and convergence failures in the evidence record.

### 17.3 Hardware use

- Laptop CPU measurements define release baselines.
- Use the `ssh office` NVIDIA machine for maintained GPU work.
- Record hardware and software automatically in benchmark artifacts.
- Do not compare different machines as if they differ only by code.
- Do not occupy the GPU with uncontrolled long jobs; use bounded cases, progress, timeouts, and resumable artifacts.

### 17.4 Implementation behavior

Codex must:

- prefer deleting duplication over adding an adapter stack;
- preserve stable scientific outputs unless the work item explicitly corrects them;
- add tests before or with behavior changes;
- run focused tests during development and the required full gates before opening a pull request;
- measure cold and warm performance when a hot path, dependency, data model, or execution boundary changes;
- use true residuals and matched observables to validate solvers;
- inspect JAX traces, compilation counts, transfers, and memory rather than inferring them;
- keep user-facing names physical and clear;
- update documentation and examples in the same pull request as public behavior;
- update the capability registry and plan ledger before completion;
- open a pull request with goals, equations/algorithms changed, evidence, performance, limitations, and follow-up work.

Codex must not:

- declare success because tests compile or a toy case runs;
- silently loosen tolerances to improve speed;
- hide failed benchmark rows;
- add large fixtures or generated output to Git;
- add environment-variable-only production features;
- create a new authoritative plan;
- leave disabled, duplicate, or dead implementations “for later” without a deprecation decision;
- merge its own pull request unless explicitly authorized.

### 17.5 Stop and escalation conditions

Stop the current slice and update the ledger when:

- baseline evidence is contradictory or cannot be reproduced;
- a proposed migration changes physics unexpectedly;
- an external reference cannot be legally redistributed;
- a performance result depends on an undocumented flag or unmatched accuracy;
- the route exceeds the memory/time budget and no bounded fallback exists;
- a design decision conflicts with this plan and cannot be resolved by the accepted contracts.

The ledger entry must state the smallest reproducible failure and the decision required.

## 18. Risks and controls

| Risk | Consequence | Control |
| --- | --- | --- |
| Broad rewrite destroys validated behavior | Lost credibility and long regressions | Vertical slices, frozen baseline, independent identities, no duplicate owners |
| Performance is postponed until after redesign | New architecture becomes slower and harder to fix | Performance gates in every phase, benchmark before/after each boundary |
| SFINCS compatibility dominates internals | Native DKX remains archaic | Permanent adapter depends on canonical `Case`/`Result`, never the reverse |
| JAX compilation dominates short runs | Poor laptop usability | Compilation families, cache, shape stability, separate tiny-case policy |
| GPU path is slower than CPU | Misleading accelerator claim | End-to-end matched benchmarks, batch-first GPU strategy, honest failures |
| Sequential block sweep limits GPU | Structured route underuses accelerator | Optimize data layout/batched blocks first; evaluate cyclic reduction only if measured |
| Krylov basis exceeds memory | OOM or swapping | Memory-aware restart/recycle, fixed-linear storage, admission guards |
| Preconditioner reuse becomes stale | Iteration explosion or failure | Validity signatures, lagging criteria, residual/iteration-triggered rebuild |
| Mixed precision corrupts results | Fast but wrong transport | Float64 final residual and moments, refinement, fallback, status separation |
| Sharding adds collective overhead | More devices make runs slower | Workload sharding first, communication accounting, admission speedup gates |
| Root continuation misses branches | Wrong radial electric field | All-root scans, independent cold checks, branch-event detection |
| Coverage is gamed | High number, weak science | Branch coverage, scientific oracles, installed artifact, dead-code removal |
| External-code comparisons are irreproducible | Unverifiable claims | Pinned commits/builds, compact evidence bundles, matched conventions |
| Package footprint target is misrepresented | Impossible or misleading promise | Measure DKX-owned bytes separately from dependencies |
| Media and caches regrow repository | Slow clones and repeated history rewrites | CI budgets, release assets, manifest, no generated binaries |
| `plan.md` becomes a diary | Agents cannot find current state | Compact ledger and release summaries |
| Multiple plans reappear | Conflicting agent instructions | CI check for authoritative-plan references and explicit supersession policy |

## 19. Closed decision register

| ID | Status | Decision |
| --- | --- | --- |
| D001 | accepted 2026-08-25 | DKX 3 may break DKX 2 Python API. |
| D002 | accepted 2026-08-25 | Permanent SFINCS namelist and HDF5 adapters. |
| D003 | accepted 2026-08-25 | Physical/engineering user units; normalize once internally. |
| D004 | accepted 2026-08-25 | Native versioned NetCDF; SFINCS HDF5 adapter. |
| D005 | accepted 2026-08-25 | Python 3.11 minimum. |
| D006 | accepted 2026-08-25 | `argparse` plus Rich. |
| D007 | accepted 2026-08-25 | `src/` and shallow domain packages. |
| D008 | accepted 2026-08-25 | Whole-profile ambipolar/transport/bootstrap first. |
| D009 | accepted 2026-08-25 | SFINCS-matched tested features are stable candidates; other features need further evidence. |
| D010 | accepted 2026-08-25 | Local external-code campaigns; no ordinary CI/runtime dependency. |
| D011 | accepted 2026-08-25 | Laptop CPU official; office NVIDIA GPU maintained. |
| D012 | accepted 2026-08-25 | Full clone and DKX-owned distribution/install artifacts below 20 MiB. |
| D013 | accepted 2026-08-25 | Merge one `plan.md` to `main`; small coherent feature branches. |
| D014 | adopted 2026-08-25 | MyST, Sphinx, Furo. |
| D015 | adopted 2026-08-25 | Compact persistent ledger with release summaries. |
| D016 | accepted 2026-08-25 | Performance and memory are release contracts. |

## 20. Reference set for implementation and review

Implementation should consult primary sources and current code, not rely only on the summaries in this plan.

### 20.1 DKX history and current baseline

- [DKX repository](https://github.com/uwplasma/DKX)
- merged PR #8, `Refactor v3 driver architecture`, including historical `plan_final.md`
- current README, source map, test/validation documentation, benchmark artifacts, and open PRs #66 and #67
- VMEX and Solvax repositories for coupling and reusable numerical infrastructure

### 20.2 Neoclassical codes

- [SFINCS](https://github.com/landreman/sfincs), especially `fortran/version3/solver.F90`, `populateMatrix.F90`, technical documentation, example decks, and output conventions
- [YANCC](https://github.com/f0uriest/yancc) and [arXiv:2607.20861](https://arxiv.org/abs/2607.20861), especially performance, tuning, multigrid, Krylov, warm-start, and GPU documentation
- [MONKES](https://github.com/JavierEscoto/MONKES) and [arXiv:2312.12248](https://arxiv.org/abs/2312.12248), especially the block-tridiagonal Legendre solver and matrix-free variant
- DKES sources in STELLOPT and the variational formulation papers
- KNOSOS and its orbit-averaged low-collisionality formulation
- NEO and NEO-2 papers and documentation
- BOOTSJ and PENTA sources/documentation in the STELLOPT ecosystem
- FORTEC-3D method and validation papers

### 20.3 Linear algebra and solver systems

- [PETSc KSP manual](https://petsc.org/release/manual/ksp/), including exact/preconditioning operator separation, successive systems, convergence norms, runtime options, and matrix-free shell operators
- [PETSc `KSPSetReusePreconditioner`](https://petsc.org/release/manualpages/KSP/KSPSetReusePreconditioner/)
- [PETSc `KSPPIPEFGMRES`](https://petsc.org/release/manualpages/KSP/KSPPIPEFGMRES/)
- [PETSc `MATSHELL`](https://petsc.org/release/manualpages/Mat/MATSHELL/)
- [MUMPS 5.9.1 user guide](https://mumps-solver.org/doc/userguide_5.9.1.pdf), including analysis/factor/solve, ordering, mixed precision, refinement, out-of-core, BLR, tree parallelism, and GPU offload
- [SuperLU_DIST](https://github.com/xiaoyeli/superlu_dist), especially the 3-D communication-avoiding algorithm, OpenMP/MPI/GPU execution, ordering, and triangular solves
- Solvax block-tridiagonal, Krylov, preconditioning, and implicit-differentiation APIs

### 20.4 JAX and packaging

- [JAX distributed arrays and automatic parallelization](https://docs.jax.dev/en/latest/parallel.html)
- [JAX `shard_map`](https://docs.jax.dev/en/latest/notebooks/shard_map.html)
- [JAX persistent compilation cache](https://docs.jax.dev/en/latest/persistent_compilation_cache.html)
- [JAX profiling](https://docs.jax.dev/en/latest/profiling.html)
- [JAX device-memory profiling](https://docs.jax.dev/en/latest/device_memory_profiling.html)
- [JAX asynchronous dispatch](https://docs.jax.dev/en/latest/async_dispatch.html)
- [JAX buffer donation](https://docs.jax.dev/en/latest/buffer_donation.html)
- [JAX `custom_linear_solve`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.custom_linear_solve.html)
- [JAX GPU performance tips](https://docs.jax.dev/en/latest/gpu_performance_tips.html)
- [JAX slow tracing/compilation diagnostics](https://docs.jax.dev/en/latest/debugging/slow_tracing_compilation.html)
- [Python Packaging User Guide](https://packaging.python.org/), including `src/` layout, `pyproject.toml`, wheel/source-distribution testing, and package data
- [PyPI trusted publishing](https://docs.pypi.org/trusted-publishers/)
- [conda-forge package contribution](https://conda-forge.org/docs/maintainer/adding_pkgs/)
- [No AI Slop](https://github.com/petergyang/no-ai-slop) as a prose review aid, subordinate to scientific clarity and human review

### 20.5 Physics references for validation

The capability registry should carry exact bibliographic entries. Initial anchors include:

- Hirshman et al., *Physics of Fluids* 29, 2951 (1986);
- van Rij and Hirshman, *Physics of Fluids B* 1, 563 (1989);
- Shaing and Callen, *Physics of Fluids* 26, 3315 (1983);
- Landreman et al., *Physics of Plasmas* 21, 042503 (2014);
- Beidler et al., *Nuclear Fusion* 51, 076001 (2011);
- Simakov-Helander high-collisionality transport results;
- current MONKES, KNOSOS, YANCC, NEO-2, and FORTEC-3D method papers;
- current literature on electron roots, bootstrap current, impurities, tangential drifts, Phi1, and finite-orbit-width validity selected per capability.

## 21. Execution ledger

Use one row per merged pull request or consequential failed experiment.

| Date | Work item | PR or commit | Result and evidence | Remaining risk or next action |
| --- | --- | --- | --- | --- |
| 2026-08-25 | planning audit | no repository change | Audited DKX `0d5606ce`; reviewed current source, tests, docs, examples, packaging, open PRs, SFINCS v3, PETSc, MUMPS, SuperLU_DIST, MONKES, YANCC, and JAX guidance. | Convert the reviewed artifact into the P0 planning PR. |
| 2026-08-25 | D001-D016 | maintainer decisions | Accepted DKX 3, permanent SFINCS adapters, physical units, native NetCDF, Python 3.11, argparse+Rich, shallow packages, whole-profile flagship, local external-code campaigns, laptop CPU/NVIDIA GPU benchmarks, sub-20-MiB DKX artifacts, and one authoritative plan. | Fill exact hardware and baseline measurements in P0. |
| 2026-08-25 | P0 prior-plan reconciliation | historical PR #8 | Located the prior comprehensive `plan_final.md` in merged/closed PR #8. Integrated its vertical-slice, solver-reuse, bounded-memory, scaling, admission-gate, and no-microtranche requirements. No open plan PR was found in the repository search. | Merge `plan.md`, remove conflicting roadmap/test policy, and treat PR #8 as history. |
| 2026-08-28 | P0.1-P0.8 start | PR #70, `b204851` | Confirmed DKX 2.3.1 drift from the audited 2.3.0 baseline, no competing open planning PR, PR #67 open, and PR #66 closed/superseded. Added the authoritative plan and initial capability, baseline, hardware, and benchmark-schema records without runtime changes. The wheel, sdist, and installed DKX-owned files pass 20 MiB; the 36.46 MiB fresh clone does not. Warning-clean docs pass. The 51m08s serial suite reports 1496 passed, 22 skipped, and two baseline gaps: one reproducible constraintScheme=4 gradient tolerance miss on pristine `main`, and one noisy 5x surrogate timing assertion that passes on repeat. | Review PR #70; identify the official laptop, restore office-GPU access, resolve or explicitly accept the two baseline test gaps, and complete checksum-pinned scientific artifact refresh before merge. |
| 2026-08-28 | P0.1-P0.8 | PR #70, `4fc8cd0` | Merged the authoritative DKX 3 plan and initial evidence registries after 17 green CI/docs checks. | Execute P1 packaging and size contracts; do not begin P2 before the P1 gates are trustworthy. |
| 2026-08-28 | P1.1, P1.3-P1.4, P1.6 | PR #67, `5d7fd8c` | Merged the installed-wheel gate after 20 green checks. A clean Python 3.11 wheel install, run outside the checkout with offline data, produced six finite nonzero monoenergetic coefficients, two finite ambipolar roots, and a 221095-byte panel figure in 28.9 s on the M2 development host. CI preserved its ten-minute required-job contract by rebalancing the unchanged suite from ten to twelve coverage shards; the former heavy shard completed in 10m56s only during a diagnostic 15-minute run. No scientific tolerance changed. | Complete isolated sdist testing, published-artifact smoke, workflow lint, automated size gates, and single-source versioning. |
| 2026-08-28 | P1.7-P1.8 history audit | local destructive simulations only; no remote rewrite | A fresh full clone of merged PR #67 is 36025713 bytes: 22378106 bytes of Git data and 13647607 bytes of working tree. Removing only historical blobs above 250 KiB still left 32.35 MiB. A simulated one-commit history was 20475758 bytes and passed D012 by only 495762 bytes; retaining even the latest ten commits without further content removal measured 21181153 bytes and failed. | Keep the full-clone gate visible but non-enforcing in the P1 follow-up. Meeting D012 requires both current-content margin and an explicitly approved coordinated history rewrite; an ordinary PR cannot close it. |
| 2026-08-28 | P1.2-P1.9 | PR #71, `af0b7b2` | Merged after 21 green checks. Wheel and sdist install in clean environments and run finite science; post-PyPI smoke, checksum-pinned action lint, required-job aggregation, single-source versioning, package-data checks, and exact CI-SHA size measurements are enforced. Hosted measurements: tracked tree 13673224 B, media 3216480 B, wheel 532245 B, sdist 870017 B, installed files 4373541 B; each enforced DKX-owned artifact passes 20 MiB. The exact full clone is 36302007 B and remains visible but non-enforcing. | Begin P2. Do not claim D012 full-history completion or rewrite remote history without a separate coordinated decision. |
| 2026-08-28 | P2.1-P2.3 start | current native-case PR | Added an immutable typed schema-v1 `Case`, TOML/JSON readers, precise path-aware validation, deterministic semantic IDs, complete commented/schema output, and bounded Cartesian/zipped scan preflight without changing numerical kernels. Warm parse/validation measured 0.431 ms per representative case; median cold import measured 0.325 s on merged main and 0.335 s on this branch. Installed wheel/sdist smoke passed at 542212 B/883602 B. | Review/merge the focused contract PR, then normalize and execute one flagship profile through native `Case` and `Result`. |
| 2026-08-28 | P2.1-P2.3 | PR #72, `8bba542` | Merged the immutable native `Case`, schema-v1 TOML/JSON contract, deterministic IDs, bounded scan preflight, and `validate`/`schema` CLI after all 21 required checks passed. | Implement a genuine native normalization/execution path and `Result`; do not serialize `Case` into a compatibility deck. |
| 2026-08-28 | P2.4/P2.7 analytic profile start | current native-result PR | Added direct `Case -> KineticOperator -> solve -> Result` execution for an analytic prescribed-Er profile, with no namelist serialization/parsing. `Result` owns read-only named arrays, native SI observables, certificates, plotting, and schema-v1 NetCDF save/load. Regression tests forbid both namelist conversion calls and match particle flux, heat flux, and parallel current to the accepted kernel path at 2e-12 relative tolerance. The checked three-surface example measured 7.210 s cold, 3.683 s warm in one cache-disabled process, residual 7.51e-15, 1.053 GB process peak after both runs, 0.024 s output time, and a 47285-byte NetCDF file on the M2 development host. Focused architecture/API tests and warning-clean 47-page docs pass. Isolated artifacts are 551699 B (wheel) and 893802 B (sdist); an out-of-tree wheel run produced finite nonzero flux at residual 6.48e-15 and round-tripped a 47202-byte NetCDF file. | Review this bounded route. Next add native VMEC/Boozer normalization and surface-state reuse; the current native route explicitly rejects ambipolar, scans, full tangential drifts, Phi1, convergence refinement, and explicit sharding/reuse rather than downgrading them. |
| 2026-08-28 | P2.4/P2.7 | PR #73, `86f534b` | Merged direct analytic profile `Case -> Result` execution and schema-v1 NetCDF after all 21 CI/docs checks passed. The route never serializes or parses a SFINCS namelist, matches the accepted profile kernel at 2e-12 relative tolerance, and keeps unsupported physics explicit. | Extend the native boundary to VMEC/Boozer geometry and shape-stable reuse; separately remove the reported compatibility scan-5 startup delay under P5.10. |
| 2026-08-28 | P5.2/P5.10 start | current scan-startup PR | Routed compatible single-process RHSMode=1 `Er` compatibility scans through one shared geometry/operator and the bounded batched solve while preserving every per-point input, SFINCS HDF5 output, and solver trace. Matched empty-cache three-point scheme-11 CLI runs on merged main and this branch measured 7.54 s/996311040 B peak RSS versus 4.09 s/632520704 B; second-process populated-cache runs measured 3.05 s/907608064 B versus 1.82 s/538705920 B. Particle flux, heat flux, parallel flow/current, and NTV match scalar outputs at 2e-12 relative tolerance; true residuals are now retained by `BatchedSolveResult`. Process-parallel, transport-matrix, Phi1, non-`Er`, and explicit host-direct cases remain on the scalar path. The isolated wheel (554506 B) and sdist (896960 B) pass size gates; a wheel plus locally cloned SOLVAX installed outside the checkout produced three finite nonzero outputs. | Run hosted CI and merge the focused PR before addressing missing-root plotting status and user-facing solver terminology. |
| 2026-08-28 | P5.2/P5.10 | PR #74, `d7e46ab` | Merged shared-operator compatibility `Er` scans after all 21 required checks passed. The accepted route preserves scalar science/output parity and partial resume while reducing matched empty-cache runtime 45.8% and peak RSS 36.5%; populated-cache runtime fell 40.3% and RSS 40.6%. | Make missing-root profile status explicit and replace user-facing solver-tier jargon. |
| 2026-08-28 | P2.6/P4.4 usability | PR #75, `6391601` | Merged after all 21 required checks passed. Representative radial profiles now distinguish a bracketed root from the sampled point with minimum absolute radial current. When no root is bracketed, Er/bootstrap/flux values remain visible but are labeled closest-scanned, never ambipolar; the HDF5 evidence records the evaluated Er, residual current, and root-status flag. Solver progress names the actual route (structured direct, memory-bounded direct, recycled iterative, host sparse-direct) instead of unexplained tier numbers. Synthetic no-root and legacy-root output/plot tests pass; 63 solver tests pass with only the constraintScheme=4 gradient case already recorded as failing identically on the pre-change baseline deselected. Warning-clean 47-page docs and 59 packaging/planning/source-tree guards pass; `representative.py` remains below its audited line ceiling. Isolated artifacts are 554246 B (wheel) and 897213 B (sdist). | Review the native VMEC profile slice, then resume SOLVAX/downstream integration and native Boozer or ambipolar execution. |
| 2026-08-28 | P2.4/P2.7/P2.8 VMEC profile start | current native-VMEC PR | Extended direct native profile execution to VMEC without a namelist round-trip. A profile resolves and hashes the real `wout`, reads it once, and constructs one shape-stable phase-space grid for all surfaces while interpolating surface-specific geometry and coefficients. The final surface matches the accepted scheme-5 particle flux, heat flux, and parallel current at 2e-12 relative tolerance; tests spy on the single file read/grid construction and pin the physical kV/m-to-ErHat boundary. The checked three-surface VMEC example measured 5.679 s cold, 0.890 s warm, 1.025 GB process peak, and 8.97e-15 maximum residual on the M2 development host with an empty persistent compilation cache. Focused API/normalization tests (48) and warning-clean 47-page docs pass; isolated artifacts are 554907 B (wheel) and 898940 B (sdist). Boozer, ambipolar, full tangential drifts, Phi1, convergence refinement, and explicit sharding remain explicit errors. | Merge profile-status PR #75 first, then rebase and open this coherent VMEC slice. Next implement a dedicated native Boozer reader or native ambipolar continuation, not a compatibility conversion. |
| 2026-08-28 | P2.4/P2.7/P2.8 VMEC profile | PR #76, `d6409ee` | Merged direct native VMEC profile execution after all 19 substantive CI/docs jobs passed. The route reads and hashes the real `wout` once, reuses one shape-stable phase-space grid across surfaces, converts physical electric field only at the execution boundary, and matches the accepted scheme-5 particle flux, heat flux, and parallel current at 2e-12 relative tolerance. The checked three-surface case measured 5.679 s cold, 0.890 s warm, 1.025 GB process peak, and 8.97e-15 maximum residual. | Implement native all-root ambipolar profile execution using shared operators and bounded electric-field batches; preserve explicit no-root status and complete root evidence. |
| 2026-08-28 | P2.3/P2.4/P2.7/P2.8 ambipolar | PR #77, `8e5f254` | Admin squash-merged direct physical-unit all-root profile execution after every substantive producer, all 12 coverage shards, combined coverage, docs, examples, Python-floor, wheel/sdist install, external-data, workflow-lint, and required-job aggregation passed. Each surface builds one shape-stable operator, evaluates the coarse `E_r` search in a bounded batch, refines every sign-changing bracket with real bisection solves, retains all evaluated fields/currents/fluxes/residuals, classifies every resolved root, and selects the root nearest the previous surface while keeping an explicit closest-scanned fallback when no root is bracketed. The checked two-surface kinetic case measured 8.533 s cold, 4.700 s warm, 1851637760 B process peak, and 1.36e-15 maximum primal residual with an empty persistent cache; bisection reduced the same case from 13 to 11 evaluations and cut the measured warm time 23.6%. The final artifacts are 560734 B (wheel) and 906629 B (sdist); an out-of-tree wheel run produced two bracketed roots and round-tripped a 100809 B NetCDF file. Hosted CI initially exposed an obsolete exact `solve.py` line ceiling; `ed56eba` removed only that source-shape assertion as required by section 8.3, and the replacement gate passed. | Add adaptive missed-root/convergence-rung evidence before promoting the all-root workflow beyond `validated_limited`; do not imply that a finite grid proves the absence of an even number of unresolved crossings. |
| 2026-08-28 | P6/P10 downstream strong-root failed hypothesis | SOLVAX PR #91 closed; VMEX phase-6 experiment not retained | Tested whether moving pseudo-transient nonlinear/backtracking decisions to the host would bound compilation memory for VMEX's 24-unknown differentiated strong-force residual. The production fixture took 223.68 s, peaked at 3249111040 B RSS and a 4583156800 B process footprint, rejected stages at alpha 0.5, 0.25, 0.125, and 0.0625, and remained at alpha 0. The prior compiled attempt peaked near 3.18 GB RSS, so the apparent early host-memory reduction was transient. The generic SOLVAX route passed focused tests but failed the downstream admission gate; PR #91 was closed and the uncommitted VMEX integration was removed. | Do not wrap the current full differentiated strong residual in nested PTC/GMRES. Profile and reduce the residual/JVP cost or formulate a demonstrably cheaper physics-aware nonlinear update before reopening continuation work. |
| 2026-08-28 | P10 downstream VMEX strong-root foundation | VMEX PR #166, `980b602a` | Admin-merged the reviewed stacked strong-force root foundation after all substantive API, package, documentation, representative-physics, manifest-parity, device, and changed-line coverage producers passed. The final Linux-only oracle adjustment scales an analytically zero force component to the established physics magnitude instead of relying on a brittle fixed absolute tolerance. This merge does not include the rejected nested continuation experiment above. | Use the merged certified residual, low-order preconditioner, benchmark, and admissibility evidence as the stable base for a cheaper nonlinear update; retain independent final certification as the promotion gate. |
| 2026-08-28 | P8.2/P8.4 adaptive ambipolar evidence | PR #79, `8bdc5b5` | Admin squash-merged the deterministic bounded midpoint hierarchy after both docs triggers, all 12 coverage producers, combined coverage, external-data, installed-wheel/sdist, examples, Python-floor, workflow-lint, and required-job aggregation passed. Every kinetic evaluation retains its reason/level and every rung retains search/total counts, root count, root movement, requested-observable movement, final bracket width, and convergence flag. Synthetic same-sign endpoints expose two hidden crossings after refinement; no-root, exact-root, exhaustion, deterministic-order, evaluation-bound, and memory-preflight gates pass. Result/NetCDF/summary/plot/docs distinguish `resolved`, `refinement_exhausted`, and `no_bracket_observed` while warning that finite sampling cannot exclude hidden even crossings. On the M4/24 GiB host with Python 3.11.15/JAX 0.9.2 and empty external caches, the opt-in checked two-surface case retained 14 of a conservative 153 evaluations per surface, reproduced roots `[-1.796875, -1.8359375]` kV/m, resolved after one midpoint rung with zero root/observable movement and 0.0390625 kV/m brackets, and measured 11.512 s cold, 6.576 s warm, 1656619008 B process peak, and 2.40e-15 maximum primal residual. A matched opt-out run retained 11 evaluations and measured 8.579 s cold/5.508 s warm/1774665728 B process peak; the 34.2% cold and 19.4% warm opt-in cost buys three retained midpoint evaluations per surface and does not affect the default route. Final wheel/sdist artifacts are 563956 B/910426 B; an isolated wheel run resolved both roots and round-tripped a 151746 B NetCDF result. | Implement P8.3 branch identity and creation/loss/merger/crossing events without hiding alternative roots. |
| 2026-08-28 | P8.3 ambipolar branch events | PR #80, `ee77496` | Admin squash-merged after all 12 coverage producers and aggregation, docs, external-data, examples, Python-floor, installed artifacts, workflow-lint, and required-job aggregation passed. Deterministic post-root-discovery continuation uses linear radial prediction and global minimum-cost assignment to give every root a stable ID; selection follows the chosen ID while all alternatives remain visible. Boundary origins, interior creation/loss, discrete merger candidates, ordering crossings, and classification transitions retain participants, root indices, fields, details, and nonsmooth flags in Result/NetCDF/certificates/summary/plot/docs. A synthetic three-surface oracle preserves the selected ion branch through an unambiguous order crossing, detects two classification transitions plus unstable-branch loss/merger, verifies continuation-disabled selection, explicit no-root fallback, input ordering, NetCDF round-trip, and the visually inspected multi-branch plot. The real two-surface kinetic case preserves roots `[-1.796875, -1.8359375]` kV/m, 14 evaluations/surface, and 2.40e-15 maximum residual while assigning `ion-000` across both surfaces with no nonsmooth event. Same-environment empty-cache matched processes measured merged #79 at 10.765 s cold/6.913 s warm/1969537024 B cumulative peak and this branch at 11.074 s cold/6.684 s warm/1966800896 B; the +2.9% cold/-3.3% warm differences are consistent with negligible post-solve overhead, not a performance claim. Final wheel/sdist are 567768 B/915664 B; a clean installed wheel reproduced both roots/branch selections at 2.31e-15 residual and round-tripped a 171910 B NetCDF result. Warning-clean 46-page docs, 134 focused native/batch/scan/source/package tests, and Ruff/diff checks pass. | Independent dense-surface branch-event validation and P8.8 branch-local derivative behavior remain promotion gates; address P8.9 VMEC profile/plot regression next. |
| 2026-08-28 | P8.9 VMEC profile/plot regression | PR #81, `2ee1575` | Admin squash-merged after docs, all 12 coverage producers, combined coverage, examples, external-data, Python-floor, installed-wheel, workflow-lint, and required aggregation passed. The representative command now consumes VMEC's evaluated `presf` independent of its input coefficient representation, retains `pmass_type` provenance, preserves failed surface records, and gives unavailable physics, parser failure, kinetic solve failure, and no-bracket evidence distinct HDF5 and plotted statuses. Synthetic wout cases exercise `power_series`, `gauss_trunc`, `two_power`, `two_lorentz`, `akima_spline`, `cubic_spline`, `pedestal`, `rational`, and `sum_atan`; standard wout density/temperature limitations are stated explicitly instead of inferred. On the checked vacuum tokamak fixture, the real quick command rendered both Er/bootstrap and SI particle/heat-flux panels, retained two `bracketed_root` surfaces and `power_series` provenance, and explicitly recorded the generic-plasma fallback. The empty-cache M4/24 GiB run measured 32.75 s wall, 2175057920 B peak RSS, a 198 KiB visually inspected PNG, and a 22 KiB HDF5 file. Ruff, 57 focused representative tests, 121 representative/planning/docs/package/source-contract tests, and warning-clean 46-page docs passed. Final wheel/sdist artifacts are 569549 B/918675 B, and an out-of-checkout wheel install imported from site-packages and reproduced VMEC profile classification. | A standard wout still cannot supply separate density/temperature profiles, and independent converged stellarator profile comparison remains outstanding. |
| 2026-08-28 | P2.4/P2.7 native Boozer reader | PR #82, `ef58bf9` | Admin squash-merged after both docs builds, all 12 coverage producers, combined coverage, examples, external-data, Python-floor, installed-wheel, workflow-lint, and required aggregation passed. Direct native ``Case`` execution now accepts physical Boozer ``.bc`` inputs without a namelist adapter or user-facing SFINCS geometry number. The dedicated reader auto-detects scheme-11 cosine-only and scheme-12 asymmetric columns from one file read, parses once, and reuses immutable surface tables and one shape-stable phase-space grid. Checked scheme-11/12 detection and a two-surface asymmetric kinetic oracle pass; at a scientifically matched ``1e-10`` linear tolerance, particle flux, heat flux, and parallel current agree with the established scheme-12 route at ``2e-12`` relative tolerance. The inherited ``1e-8`` tolerance produced bit-identical operators but a 0.3206% observable difference from iterative conditioning, so the oracle was tightened rather than weakened. On the M4/24 GiB host with Python 3.11.15/JAX 0.9.2 and an initially empty external compilation cache, the checked two-surface asymmetric profile measured 8.265 s cold and 2.957 s warm inside ``dkx.run``, 1155743744 B cold and 1054703616 B warm maximum RSS, and 1.01e-13 maximum primal residual; its native NetCDF result is 47476 B. Final artifacts are 570570 B (wheel), 920262 B (sdist), and 4750794 B installed, all below the enforced 20 MiB limits; the 36443575 B full clone remains the documented non-enforcing history exception. An out-of-checkout wheel install resolved DKX from ``site-packages`` under JAX 0.10.2, completed the same profile at 2.24e-15 maximum residual, and round-tripped a 47477 B NetCDF result. | Independent SFINCS/MONKES/YANCC validation remains a separate P8.7 gate. |
| 2026-08-28 | P8.7 bounded independent MDKE validation | PR #83, `5d17abd` | Admin squash-merged after both docs builds, all 12 coverage producers, combined coverage, examples, external-data, Python-floor, installed-wheel/sdist, workflow-lint, and required aggregation passed. Added a checksummed machine-readable zero-field PAS/DKES rung for a DSHAPE tokamak, NCSX, and W7-X EIM. The audit maps physical ``nu/v`` through ``nuDHat(x0)``, uses the local rather than LCFS radius in the Beidler references, corrects the raw ``B0`` scale, and records the handedness map. All four coefficients agree within 5.52%: maximum errors are 5.51% DSHAPE versus live YANCC, 1.66% NCSX versus live YANCC, and 5.52% W7-X EIM versus the pinned MONKES row; maximum ``D33*`` error is 0.065%. The artifact pins DKX/YANCC/MONKES/SFINCS commits, source and compact-output checksums, reference residuals, equations, grids, host, timing, RSS, and exclusions. Matched empty-cache M4 processes measured DKX cold/warm/peak RSS as 3.636/0.586 s/642826240 B (DSHAPE), 5.711/1.651 s/4780326912 B (NCSX), and 13.326/9.875 s/1027932160 B (W7-X). Final local review passed the external-input audit, 132 focused validation/release/package/source-tree tests, scoped Ruff/diff checks, warning-clean 46-page docs, 571730 B wheel, 923157 B sdist, ``twine check``, and an isolated wheel normalization smoke. This rung is explicitly not full-Fokker--Planck, finite-``Er``, ambipolar, experimental, or cross-code performance validation. | Add matched full-kinetic SFINCS profile evidence without substituting unmatched equations. |
| 2026-08-29 | P8.7 matched full-kinetic SFINCS validation | PR #84, `cccccc3` | Admin squash-merged after both docs builds, all 12 coverage producers, combined coverage, examples, external-data, Python-floor, installed-wheel, workflow-lint, and required aggregation passed. Built pinned SFINCS `8df5453` against PETSc 3.23.6 with MUMPS 5.8.1 and SuperLU_DIST 9.1.0, without scientific-source edits or link stubs, after the Homebrew fallback LU was independently shown to return an incorrect state. Live DKX and SFINCS then solved the exact checked one-species analytic-tokamak profile with full Fokker--Planck collisions, full trajectories, physical density/temperature gradients, zero Er, automatic constraint 1, and a `1e-10` tolerance. From 6887 to 12509 unknowns, bootstrap/flow moved 0.042% and heat flux 0.280%; the finest nonzero scalar/spectral cross-code error is 2.69e-10 and all completed true residuals are below 1.82e-11. Particle flux and NTV are retained as axisymmetric near-zero absolute gates below 3.63e-13. Exact commits, packages, inputs, compact/raw outputs, logs, checksums, cold/warm DKX timing, SFINCS timing, RSS, and exclusions are recorded; timings are provenance, not a performance claim. Final local review passed both compact and external-raw audits, 86 focused validation/release/package/source-tree tests, scoped Ruff/diff checks, warning-clean 46-page docs, 571730 B wheel, 924102 B sdist, and `twine check`. An isolated wheel install under JAX 0.10.2 imported DKX from `site-packages` and solved the exact high full-FP deck at 2.56e-13 true residual in 25.33 s wall with 1298595840 B peak RSS. | Extend independent full-kinetic evidence to prescribed finite Er and a stellarator before claiming ambipolar-profile validation. |
| 2026-08-29 | P8.7 prescribed finite-Er full-kinetic validation | PR #85, `8eaedd4` | Admin squash-merged after both docs builds, all 12 coverage producers, combined coverage, examples, external-data, Python-floor, installed-wheel, workflow-lint, and required aggregation passed. Live DKX and the same pinned MUMPS-enabled SFINCS build solved exact high/ultra refinements of the upstream one-species tokamak full-FP case at normalized `Er=-30`, with full trajectories, physical density/temperature gradients, automatic constraint 1, and a `1e-13` tolerance. From 6887 to 12509 SFINCS unknowns, flow/current moved 0.0106%, heat flux 0.220%, and momentum flux 0.325%; the maximum retained scalar/spectral cross-code error is 1.88e-9. Intrinsically ambipolar particle flux and NTV stay below 1.07e-11 absolute, and all completed true residuals are below 5.25e-11. Exact inputs, raw/compact outputs, logs, checksums, cold/warm timing, RSS, and explicit claim exclusions are retained outside and inside the checkout as appropriate. Final local review passed compact and external-raw audits, 111 focused validation/release/package/source-tree tests, scoped Ruff/diff checks, warning-clean 46-page docs, 571730 B wheel, 924473 B sdist, and `twine check`. An isolated wheel install under JAX 0.10.2 imported DKX from `site-packages` and solved the exact high deck at 3.04e-15 true residual in 26.12 s wall with 1464057856 B peak RSS. | Add a converged stellarator full-FP comparison without using the compatibility matrix as an ambipolar certificate. |
| 2026-08-29 | P8.7 stellarator full-kinetic validation | PR #86, `ee90f9e` | Admin squash-merged after both docs builds, all 12 coverage producers, combined coverage, examples, external-data, Python-floor, installed-wheel, workflow-lint, and required aggregation passed. Live DKX and pinned MUMPS-enabled SFINCS solved exact high/ultra relative-path decks on the checksummed W7-X SC1 Boozer surface at `rN=0.5`, with physical density/temperature gradients, full Fokker--Planck collisions, full trajectories, zero Er, automatic constraint 1, and a `1e-12` tolerance. From 54407 to 98126 SFINCS unknowns and 87887 to 155994 DKX unknowns, the largest retained movement is 0.4436%; maximum scalar/spectral cross-code error is 1.36e-8, set by an 8.31e-13 absolute NTV difference, and every completed true residual is below 1.82e-12. Momentum flux stays below 1.89e-21 absolute. Serial M4 measurements recorded SFINCS high/ultra at 63.61/145.43 s and 1440448512/2853175296 B peak RSS; DKX high cold/warm at 37.51/30.31 s and ultra at 92.02/95.48 s, with cold peak RSS 6482673664/10272997376 B and peak process footprints 6344529344/15013654288 B. Final local review passed compact and external-raw audits, 80 focused validation/release/package/source tests, scoped Ruff/JSON/diff checks, warning-clean 46-page docs, a 571730 B wheel, a 925189 B sdist, and `twine check`. A fresh out-of-checkout wheel environment under JAX 0.10.2 imported DKX from `site-packages` and solved the exact high deck at 3.64e-16 true residual in 44.15 s with 5170266112 B peak RSS. Timings are provenance, not a cross-code performance claim. | Build a native whole-profile ambipolar certificate without widening this surface-profile claim to experimental, multispecies full-FP, Phi1, or a second stellarator family. |
| 2026-08-29 | VMEX-refined Boozer bridge parity | PR #78, `b9ec2bf` | Admin squash-merged after refreshing the stale branch onto current DKX main and passing both docs builds, all 12 coverage producers, combined coverage, examples, external-data, Python-floor, installed-wheel, workflow-lint, and required aggregation. The integration gate now sends the same Newton-refined VMEX state through the traceable DKX Boozer-table bridge and the host `wout_from_state`/classic Booz_xform route, so it measures mapping parity rather than the stopping displacement between two independently terminated equilibrium solves. No production code or tolerance changed. | Keep this as an integration oracle; production optimization claims still require independently certified final VMEX states. |
| 2026-08-29 | P10 VMEX QA startup isolation | VMEX PR #177, `bca15514` | Admin squash-merged the clean standalone startup/memory slice after both Python API floors, quality/package/docs, two-device AD, changed-executable-line coverage, all representative-physics lanes, all manifest-parity lanes, and the aggregate PR gate passed. The canonical QA example now uses the exact scalarization of its former weighted residual rows and one reverse implicit adjoint, while the pointwise least-squares API remains available. Corrected distinct-point M4/JAX 0.11.1 evidence measured merged-main least-squares at 55.93 s cold/3705.6 MiB peak RSS/15.02 s one-point warm, branch least-squares at 44.73 s/2965.1 MiB/16.57 s median warm, and the scalar example at 32.21 s/2574.0 MiB/17.38 s median warm. The scalar and least-squares initial costs agree within 2.8e-8 absolute and gradient norms within about 2.4e-8 relative. This is a 42.4% cold-start and 30.5% RSS improvement for the example, not a claim that its warm step beats Gauss--Newton. Stacked PR #170 was closed as superseded; uncertified #167--#169 were not merged. | Use this admitted startup foundation only with independently certified final optimizer states; do not revive the rejected nested full-residual PTC hypothesis. |
| 2026-08-29 | P8.2 automatic ambipolar true-residual recovery | PR #87, `cb1ed04` | Admin squash-merged after both docs builds, all 12 coverage producers, combined coverage, examples, external-data, Python-floor, installed-wheel, workflow-lint, and required aggregation passed. A production-grid five-surface W7-X standard-configuration PAS/DKES method case exposed a real structured-batch miss at `rN=0.7`, `Er=0`: `8.23448e-13` true residual versus the unchanged `4.4514e-13` target. The all-GMRES alternative was rejected after 346.94 s because it had not completed the first surface and reached 9855287296 B maximum RSS plus an 81547880576 B process-footprint measurement. The admitted bounded automatic policy retries only failed fields with one scalar GMRES solve and retains both attempts, requested/executed routes, residuals, acceptance, and reason; explicit methods remain fail-closed. On the Apple M4/24 GiB host with macOS 26.6.2, Python 3.11.15, JAX 0.9.2, NumPy 2.4.3, and SciPy 1.17.1, matched empty/populated external-cache processes completed in 307.43/295.04 s wall (305.414/292.936 s inside `dkx.run`), measured 14409515008/14681489408 B maximum RSS and 17540823704/17707186672 B peak footprint, retained 221 structured attempts plus one GMRES recovery, reduced the failed point to `1.93235e-13`, and kept every completed residual below that value. Both runs resolved every adaptive hierarchy, found root counts `[1, 1, 3, 1, 1]`, and selected `[11.3428, 11.6309, 7.4170, -2.8516, -6.7627]` kV/m; every scientific/evidence array is identical across runs except timing. Final local review passed 112 focused native/source/package tests, 27 route-specific native/batch tests, release gates, scoped Ruff/diff checks, warning-clean 46-page docs, a 573480 B wheel, a 927774 B review-head sdist, and `twine check`. A clean out-of-checkout wheel install under JAX 0.10.2 imported DKX from `site-packages`, reproduced both analytic roots at `2.31e-15` true residual with 28 retained attempts, and round-tripped a 197134 B NetCDF result. These are solver/workflow admission results, not phase-space convergence, experiment, full-FP ambipolar, or independent-code validation. | Build the compact checksummed full-profile certificate as its own evidence slice. |
| 2026-08-29 | P8.5 native whole-profile certificate | PR #88, `92d1823` | Admin squash-merged after both docs builds, all 12 coverage producers, combined coverage, examples, external-data, Python-floor, installed-wheel, workflow-lint, and required aggregation passed. The 20154 B machine-readable certificate pins DKX `cb1ed04`, SFINCS geometry source `8df5453`, case ID `f284407a...f096bf`, the 1278 B input (`f2bfee1f...602e7`), the 19416676 B Boozer geometry (`81c686e5...b6062`), and both 339225 B native NetCDF results (`199fcf0e...b8d60`, `a3a790bf...0470c`). Matched empty/populated-cache processes measured 301.64/302.09 s wall, 13588021248/14178992128 B maximum RSS, and 17728371400/17770117688 B peak footprint; the 0.15% slower warm run supports no cache-speedup claim. The compact result retains every selected SI particle/heat flux and current, root counts `[1, 1, 3, 1, 1]`, alternatives and branch identities, seven discrete events including six nonsmooth warnings, two refinement rungs per surface, all 222 solver attempts, and the one rejected/recovered pair. Every final bracket is 0.0048828125 kV/m, maximum selected residual is `5.37e-15`, maximum accepted attempt residual is `1.93e-13`, maximum root-current/slope-bracket fraction is 0.409, and cold/warm scientific arrays are identical except timing. Final local review passed self-contained and external raw-file audits, 108 focused native/validation/manifest/release/package tests, release gates, scoped Ruff/JSON/diff checks, warning-clean 46-page docs, a 573480 B wheel, a 929153 B review-head sdist, and `twine check`. This remains workflow evidence, not phase-space convergence, continuously localized events, experiment, full-FP/Phi1 or independent ambipolar validation, cross-code performance, or a second stellarator family. | Run a separate phase-space root/observable ladder without conflating it with electric-field refinement. |
| 2026-08-29 | P8.4 bounded ambipolar phase-space ladder | PR #89, `a042bd5` | Admin squash-merged after both docs builds, all 12 coverage producers, combined coverage, examples, external-data, Python-floor, installed-wheel, workflow-lint, and required aggregation passed. The separate coarse/reference/fine W7-X PAS/DKES ladder uses resolutions `(13,31,32,5)`, `(15,37,36,6)`, and `(17,37,40,6)` in theta/zeta/pitch/speed with the exact profile physics, electric-field hierarchy, and solver policy held fixed. All three rungs preserve root counts `[1,1,3,1,1]`, classifications, branch IDs, and selected branches, and all accepted true residuals stay below `3.92e-13`. The coarse/reference/fine processes measured 123.33/301.64/440.57 s wall, 5612601344/13588021248/13234585600 B maximum RSS, and 5226239928/17728371400/23418227384 B peak footprint with initially empty matching external caches. Coarse-to-reference movement reaches 1.328125 kV/m, 27.60% selected particle flux, and 25.17% selected heat flux. Reference-to-fine movement still reaches 1.6259765625 kV/m, 4.08% selected particle flux, and 7.81% selected heat flux, failing the unchanged 0.005 kV/m and 2% gates. The 35171 B checked artifact therefore records `refinement_exhausted`, not convergence; the fine rung also does not refine zeta or speed beyond reference. Compact and external-raw audits, 202 focused ambipolar/validation/planning/release/package/source tests, scoped Ruff/JSON/diff checks, a warning-clean 46-page docs build, a 573480 B wheel, a 930621 B sdist, and `twine check` pass locally. | Isolate theta and pitch sensitivity before spending the remaining 24 GiB envelope on another brute-force rung. |
| 2026-08-29 | P8.4 theta/pitch resolution diagnosis | PR #90, `2f6a6e0` | Separate fixed-physics rungs compare the `(15,37,36,6)` reference with theta-only `(17,37,36,6)`, pitch-only `(15,37,40,6)`, and pitch44 `(15,37,44,6)`. Every rung retains root counts `[1,1,3,1,1]`, classifications, branch identities, and selected branches; the maximum accepted true residual is `3.75e-13`. Theta-only measures 0.1611328125 kV/m maximum root, 0.90% selected particle-flux, and 1.28% selected heat-flux movement: flux gates pass but the root gate fails. Pitch40 is dominant at 1.7333984375 kV/m, 3.76%, and 9.47%. Pitch40-to-pitch44 still moves roots by 0.205078125 kV/m and selected particle/heat flux by 13.52%/14.07%; several observables move farther from the reference. Fresh empty-cache reference/theta17/pitch40/pitch44 processes measured 301.64/405.03/326.91/395.70 s wall and 17728371400/21386708200/5633497568/22275409800 B footprints. The 49858 B artifact records `refinement_exhausted`, identifies pitch as the dominant failed direction, and rejects a blind pitch48 escalation near the host limit. Compact and external-raw audits, 207 focused ambipolar/validation/planning/release/package/source tests, scoped Ruff/JSON/diff checks, a warning-clean 46-page docs build, a 573480 B wheel, a 931164 B sdist, and `twine check` pass locally. Every hosted producer, both docs builds, combined coverage, and required aggregation passed before the admin squash merge. | Investigate pitch discretization/quadrature conditioning or a cheaper per-surface resolution strategy before another high-order profile; zeta and speed remain explicit gates. |
| 2026-08-29 | P8.4 native pitch-speed ramp evidence control | PR #91, `7912f04` | Admin squash-merged after both docs builds, all 12 coverage producers, combined coverage, examples, external-data, Python-floor, installed-wheel, workflow-lint, and required aggregation passed. Added optional schema-v1 `resolution.pitch_speed_ramp` values 0/1/2 matching the supported SFINCS speed-local pitch truncation rules, while omitting the historical default 1 from canonical content so implicit and explicit defaults preserve every existing case ID. Native execution consumes the control directly without a namelist path and records the rule, all active pitch counts by speed, and their sum in immutable Result/NetCDF metadata. A small analytic kinetic oracle matches the accepted compatibility path at 2e-12 relative tolerance for both uniform option 0 and default option 1. A deliberately unpromoted three-surface W7-X diagnostic on clean head `2df599f` then compared ramp36 counts `[4,9,17,27,36,36]` (sum 129) with uniform22 `[22,22,22,22,22,22]` (sum 132) using separate empty caches. Ramp36 completed in 184.37 s total with 4556554240 B maximum RSS and 4130181920 B footprint; uniform22 took 341.72 s, 13068550144 B maximum RSS, and 31859925880 B footprint. Both electric-field hierarchies resolved below 1.09e-15 maximum selected residual, but topology changed from `[1,3,1]` to `[3,1,1]` and the middle selected root changed from `+7.4169921875` to `-1.5966796875` kV/m. Exact external input/result SHA-256 values are `162244b5236e65c74af088f67d495729d2ea19b19efbfdd41add07e4d4fcbe6b`/`e80d259d19009a2694b4de18141a02f0fbe6a816af2cfd7919134dee48439220` and `e55c23c117e71e8c2d81daaf801af007d008a1ea0083f6c5cd1f0e47cc3bf415`/`f64517d4e02779f2f43349f1130c37b60005e3ae2b3e1f15afddab4725511f4b`. This proves strong allocation sensitivity and rejects uniform whole-profile escalation; it does not establish convergence of either discretization. Local review passed 59 focused native/ambipolar/schema/package tests, scoped Ruff/diff checks, a warning-clean 46-page docs build, a 573850 B wheel, and a 931609 B sdist. | Package the surface-local diagnostic only after selecting a cheaper refinement strategy; zeta and speed remain open, and the 31.86 GB uniform footprint is outside the host envelope. |
| 2026-08-29 | P5/P8.4 bounded uniform-pitch routing and diagnosis | PR #93, `6c35aa1` | Admin squash-merged after both docs builds, all 12 coverage producers, combined coverage, examples, external-data, Python-floor, installed-wheel, workflow-lint, and required aggregation passed on the exact review head. Corrected the batch memory contract so one explicit budget controls both outer chunking and each element's automatic solver route. On the exact three-surface uniform-pitch-22 W7-X PAS/DKES case, 139 memory-bounded structured evaluations retained every full-factor root and bracket exactly; selected flux differences stay below `3.58e-11` relative, retained evaluation flux differences below `1.53e-10`, and the maximum bounded true residual is `5.28e-14`. The cold/warm bounded processes measured 176.33/184.81 s, 3093299200/3055878144 B maximum RSS, and 2830077576/2923810392 B footprint versus the prior 341.72 s, 13068550144 B RSS, and 31859925880 B footprint. The slower warm run supports no cache-speedup claim. The admitted route then bounded uniform pitch 22/26/30 at 2.92/2.14/2.86 GB maximum recorded footprint, but topology changed `[3,1,1] -> [1,1,1] -> [1,3,1]`; adjacent selected fields moved by 9.599609375/7.7001953125 kV/m and selected heat flux by 55.72%/45.25%. All residuals stayed below `5.65e-14`, so the checked artifact truthfully records `refinement_exhausted` and does not admit uniform pitch 34 or higher. | Use the admitted budget propagation to isolate low/intermediate/high speed-node groups at fixed bounded work. Do not run pitch48 or another uniform whole profile; zeta, speed, continuous events, experiment, full-FP ambipolar, Phi1, and a second stellarator family remain separate gates. |
| 2026-08-29 | P8.4 fixed-work pitch-by-speed diagnosis | PR #94, `32cbef5` | Admin squash-merged after both docs builds, all 12 coverage producers, combined coverage, examples, external-data, Python-floor, installed-wheel, workflow-lint, and required aggregation passed. Compared the supported uniform22, linear-ramp36, and quadratic-ramp44 allocations on the common inner W7-X surface pair. Their six-node pitch counts `[22,22,22,22,22,22]`, `[4,9,17,27,36,36]`, and `[4,5,11,25,44,44]` retain 132/129/133 total modes while shifting work from low toward high speed. Root topology changes `[3,1] -> [1,3] -> [1,1]`; uniform-to-linear selected Er/heat movement reaches 12.20703125 kV/m/68.35%, and linear-to-quadratic reaches 2.177734375 kV/m/17.93%. All accepted residuals remain below `7.04e-14`; measured footprints remain below 4.14 GB. The new quadratic pair measured 140.74 s cold and 148.75 s warm, 2852945920/2946809856 B RSS, and 2398850384/2745945472 B footprint. Its cold/warm scientific arrays are exact except timing, and the slower warm run supports no cache-speedup claim. The self-auditing checked artifact records `diagnostic_complete` with `phase_space_converged=false`. Compact and external-raw audits, 133 focused native/validation/planning/package/source tests, scoped Ruff/JSON/diff checks, a warning-clean 46-page docs build, a 573961 B wheel, a 934072 B sdist, and `twine check` pass locally; tracked worktree/media sizes are 14403033/3216480 B. The final review also repaired Python 3.10 audit-script imports without changing evidence. | Add a narrowly scoped explicit six-node allocation diagnostic to separate low from intermediate sensitivity while holding high-speed work fixed; keep the bounded two-surface scope and do not promote phase-space convergence. |
| 2026-08-29 | P8.4 explicit fixed-high-work pitch diagnosis | PR #95, `08530e4` | Admin squash-merged after both docs builds, every coverage producer, combined coverage, examples, external-data, Python-floor, installed-wheel, workflow-lint, and required aggregation passed. Added optional deterministic `resolution.pitch_modes_by_speed` control without changing the historical default or existing case IDs. The supported linear36, low-heavy, and intermediate-heavy allocations each retain exactly 129 active modes and exactly 72 high-speed modes on the bounded inner W7-X pair. All preserve root topology `[1,3]`, but pairwise selected Er, particle-flux, and heat-flux movement reaches 1.064453125 kV/m, 9.89%, and 9.08%, so the checked artifact records `refinement_exhausted` and `phase_space_converged=false`. The low-heavy run measured 146.45 s inside DKX, 4076601344 B RSS, and 3666596184 B footprint; intermediate-heavy cold/warm measured 147.21/146.38 s, 4020191232/3375595520 B RSS, and 3570897288/4001649080 B footprint. Every new run retained 98 bounded structured solves, maximum accepted residual is `3.05e-14`, and the intermediate-heavy cold/warm scientific arrays are exact except timing. Compact and external-raw audits, 217 focused native/phase-space/ambipolar/validation/planning/release/package/source tests, scoped Ruff/JSON/diff checks, and a warning-clean 46-page docs build pass locally. The isolated 575131 B wheel and 936462 B sdist pass `twine check`; an out-of-checkout wheel installation under JAX 0.10.2 imported from `site-packages` and completed a finite scientific smoke solve. | Raise low and intermediate pitch work together while holding the admitted high-speed group fixed; keep the two-surface bounded scope and do not promote phase-space convergence. |
| 2026-08-29 | P8.4 combined pitch diagnosis and speed-local observability | PR #96, `30cb93e` | Admin squash-merged after both docs builds, all 13 rebalanced coverage producers, combined coverage, examples, external-data, Python-floor, installed-wheel, workflow-lint, and required aggregation passed. The bounded inner W7-X pair raises the low/intermediate/high groups from `8/49/72` to `24/49/72`, then the intermediate group to `24/57/72`. All rungs retain topology `[1,3]`, 98 bounded structured solves, and accepted residuals below `3.05e-14`. Intermediate-heavy to combined movement is only 0.0146484375 kV/m, 0.082% particle flux, and 0.115% heat flux, but combined to intermediate28 moves 0.1513671875 kV/m, 2.64%, and 2.45%, failing all unchanged gates. Combined cold/warm science is exact except timing; measured process wall/RSS/footprint are 184.42 s/3500883968 B/3789001000 B cold and 230.94 s/3143516160 B/4026946000 B warm, so no cache-speedup claim is made. Intermediate28 cold measures 201.55 s/4182147072 B/3755315712 B. The 7576 B checksummed compact/external audit records `refinement_exhausted`. Source review pins SFINCS speed-local pitch rules, YANCC conditioning/convergence guidance, MONKES Legendre block practice, and PENTA root-bracketing conventions. Native results now retain the already-computed per-speed particle/heat-flux contributions on named axes, with their sums required to reproduce integrated fluxes, adding small diagnostic arrays rather than full distribution storage. A real analytic kinetic run retained `(surface,evaluation,speed,species)=(2,14,4,1)`, reproduced integrated particle/heat flux to `6.90e-11`/`5.80e-14` relative, kept the prior roots and `2.40e-15` residual, and round-tripped a 209868 B NetCDF result. Local review passes 240 focused native/phase-space/validation/planning/release/package/source tests, scoped Ruff/JSON/diff checks, a warning-clean 46-page docs build, a 575729 B wheel, a 937794 B sdist, and `twine check`. The initial hosted 12-way shard 10 completed 135 tests (4 skipped) in 9m24s but was canceled before artifact upload by the unchanged ten-minute whole-job limit; the 13-way replacement retained the full suite and limit, with every shard completing in at most 6m53s. | Use the retained speed spectra to localize the sensitive nodes before another bounded rung; zeta and speed convergence remain explicit gates. |
| 2026-08-29 | P8.4 common-field speed-local pitch ceiling diagnosis | PR #97, `b2fdee6` | Admin squash-merged after both docs builds, all 13 coverage producers, combined coverage, examples, external-data, Python-floor, installed-wheel, workflow-lint, and required aggregation passed; the slowest shard completed in 6m51s. Five exact two-surface W7-X probes retain the physical profile gradients but narrow the electric-field grid to `[8.4,8.55,8.7]` kV/m, reducing each diagnostic to 6 or 11 structured evaluations. At the shared second-surface field, changing `[12,12,24,25,36,36]` to `[12,12,28,29,36,36]` moves particle/heat flux by 3.64%/3.09%; speed node 3 at `v/vth=1.4657` accounts for 96.1-98.4% and 98.6-99.4% of their absolute changes across species. Raising only node 3 from 33 to the pitch-36 ceiling still moves fluxes by 9.58%/7.97%. Raising the ceiling with `[12,12,24,40,44,44]` moves them another 24.12%/26.36%, now exposing high-speed-node sensitivity. Every true residual is below `7.04e-16`, peak recorded host memory is below 1.64 GB, and the audit records `phase_space_converged=false`; timings are provenance, not a performance claim. The exact inputs, compact artifact, and optional external-result audit pass; local review passes 73 focused native/phase-space/ambipolar/planning/package tests, scoped Ruff/JSON/diff checks, a warning-clean 46-page docs build, a 575729 B wheel, a 938143 B sdist, and `twine check`. | Treat pitch and speed as coupled open axes; design a bounded joint probe with modal-tail or conditioning evidence before any full root/profile campaign. |
| 2026-08-29 | P8.4 joint pitch-speed and route-aware tail diagnosis | PR #98, `890f3ea` | Admin squash-merged after both docs builds, all 13 coverage producers, combined coverage, examples, external-data, Python-floor, installed-wheel, workflow-lint, and required aggregation passed; the slowest producer completed in 8m31s. At the common W7-X field of 8.55 kV/m, SFINCS-default ramp pitch44 with speed 6/8 uses `[5,12,21,33,44,44]`/`[5,9,16,25,35,44,44,44]`; particle/heat flux moves 2.75%/2.87%, narrowly failing both unchanged 2% gates. At speed 8, raising the ceiling from pitch44 to pitch52 (`[6,11,19,30,42,52,52,52]`) moves particle/heat flux another 7.22%/11.43%. All 18 bounded evaluations retain residuals below `7.44e-16`, recorded peak host memory below 1.01 GB, and explicit no-bracket status for the deliberately narrow field window. A scale-stable, volume- and Legendre-orthogonality-weighted last-two-mode L2 diagnostic is retained for full modal states; the uniform analytic oracle records a 5.12% maximum tail, roots `[-1.8359375,-1.875]` kV/m, and `1.50e-14` maximum residual. The production truncated route returns exact low transport moments but zero-pads eliminated high modes, so it explicitly records the tail as unavailable instead of publishing false zeros. The checked audit records `phase_space_converged=false`; timings are provenance, not a cross-resolution performance claim. Local review passes both compact and optional external-result audits, 85 focused tests, scoped Ruff/JSON/diff checks, a warning-clean 46-page docs build, a 576990 B wheel, a 940368 B sdist, and `twine check`. | Add an opt-in bounded reverse-tail reconstruction, or another independently certified conditioning metric, before increasing W7-X resolution again. Do not start a full root/profile campaign. |

## 22. Current next action

Review and package the bounded joint pitch/speed evidence, then add an opt-in
bounded reverse-tail reconstruction for the memory-lean structured route (or
an independently certified conditioning metric) before increasing W7-X
resolution again. Speed 6-to-8 and pitch 44-to-52 both fail the unchanged flux
gates, so do not run a whole profile or promote phase-space convergence. Zeta
convergence, continuous branch-event localization, experiment, full-FP
ambipolar comparison, Phi1, and a second stellarator family remain separate
gates.
