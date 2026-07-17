## Examples

This directory is the learning surface for `sfincs_jax`. Start with
`tutorials/` if you are starting with the code; jump to the topic folders if you
already know which workflow you need. All first-pass examples avoid a local
SFINCS Fortran v3 executable. Parity and benchmark scripts use frozen
references or optional local Fortran only when explicitly requested.

The machine-readable navigation map lives in `workflow_catalog.json`. It lists
the supported topic folders, first-pass entry points, typical commands, runtime
budgets, and whether a workflow needs a local SFINCS Fortran v3 executable.
Tests keep this catalog synchronized with this README and the documentation.
Use `list_workflows.py` when you want the catalog from the terminal:

```bash
python examples/list_workflows.py --list-topics
python examples/list_workflows.py --topic bootstrap --long
python examples/list_workflows.py --search "VMEC geometry"
```

### One-Command Starts

Use these entries when you want a concrete command before reading the topic
folders. Each entry writes outputs under a script-controlled directory or a
temporary path, and none requires SFINCS Fortran v3 for the first run.

| Goal | Entry script | Typical command |
| --- | --- | --- |
| Write output files and a diagnostics panel | `tutorials/run_quick_output_and_plot.py` | `python examples/tutorials/run_quick_output_and_plot.py --out-dir tutorial_output` |
| Inspect HDF5, NetCDF, NPZ, and plotting | `getting_started/write_and_plot_multiple_formats.py` | `python examples/getting_started/write_and_plot_multiple_formats.py` |
| Load VMEC geometry through `wout_path` | `getting_started/write_sfincs_output_vmec.py` | `python examples/getting_started/write_sfincs_output_vmec.py` |
| Compute a RHSMode=2/3 transport matrix | `transport/transport_matrix_rhsmode2_and_rhsmode3.py` | `python examples/transport/transport_matrix_rhsmode2_and_rhsmode3.py` |
| Differentiate a residual with JAX | `autodiff/autodiff_gradient_nu_n_residual.py` | `python examples/autodiff/autodiff_gradient_nu_n_residual.py` |
| Compare kinetic bootstrap current with Redl | `vmec_jax_finite_beta/compare_qs_paper_sfincs_jax_redl.py` | `python examples/vmec_jax_finite_beta/compare_qs_paper_sfincs_jax_redl.py --case QA --quick --jax-vs-redl --solve-method auto` |
| Time output formats and memory behavior | `performance/benchmark_output_formats.py` | `python examples/performance/benchmark_output_formats.py --repeats 2` |
| Check a frozen Fortran-v3 output fixture | `parity/output_parity_vs_fortran_fixture.py` | `python examples/parity/output_parity_vs_fortran_fixture.py` |

### Run Budgets And Outputs

- Tutorial, getting-started, and frozen-fixture parity entries are designed for
  seconds-scale laptop CPU runs.
- VMEC, Redl, optimization, and performance entries can take longer; use
  `--quick` where available and inspect the generated JSON/HDF5 solver metadata
  before using a result quantitatively.
- Large VMEC and Fortran reference assets are fetched through the package data
  cache or supplied by the user. Generated output directories stay out of git.

### Decision Map

Use this map when you are looking at `examples/` for the first time and want the
shortest path to the relevant workflow.

For an interactive terminal version of this map, run
`python examples/list_workflows.py --list-topics` or filter by task with
`python examples/list_workflows.py --topic transport`.

| Starting question | Go here | Why |
| --- | --- | --- |
| I want to run one case and plot the output. | `tutorials/run_quick_output_and_plot.py` | It writes HDF5, NetCDF, NPZ, and a PDF panel in one bounded command. |
| I want to learn the file formats and CLI/API basics. | `getting_started/` | These scripts isolate input parsing, output writing, VMEC paths, and plotting. |
| I need transport coefficients. | `transport/` | These examples cover RHSMode=2/3 transport matrices and scan postprocessing. |
| I need gradients or optimization hooks. | `autodiff/` then `optimization/` | Start with residual/JVP examples, then move to QA objectives and promotion gates. |
| I need bootstrap current or Redl comparisons. | `vmec_jax_finite_beta/` | This folder owns the VMEC, Redl, ambipolar-root, and bootstrap-current profile scripts. |
| I need to validate against SFINCS Fortran v3 behavior. | `parity/` then `publication_figures/` | The first folder has frozen fixtures; the second regenerates release-facing comparison plots. |
| I need CPU/GPU runtime or memory evidence. | `performance/` | These scripts benchmark output formats, JIT behavior, transport workers, and optional backends. |
| I recognize an upstream SFINCS input name. | `sfincs_examples/` | This folder preserves upstream-style decks for parity and benchmark audits, not first-pass learning. |

### Learning Path

| Step | Goal | Start here |
| --- | --- | --- |
| 0 | Choose the right path for your application | `tutorials/00_start_here.ipynb` |
| 1 | Run the CLI, write output files, and plot diagnostics | `tutorials/01_cli_outputs_and_plots.ipynb`, `tutorials/run_quick_output_and_plot.py` |
| 2 | Compute transport matrices and see autodiff | `tutorials/02_transport_and_autodiff.ipynb` |
| 3 | Compare bootstrap current with Redl and see optimization hooks | `tutorials/03_bootstrap_redl_and_optimization.ipynb` |
| 4 | Choose geometry, validation, and performance workflows | `tutorials/04_geometry_validation_and_performance.ipynb` |
| 5 | Understand grids, geometry, and one operator action | `getting_started/build_grids_and_geometry.py`, `getting_started/apply_collisionless_operator.py` |
| 6 | Compare outputs with frozen SFINCS Fortran v3 references | `parity/output_parity_vs_fortran_fixture.py` |
| 7 | Profile CPU/GPU, JIT, and output formats | `performance/benchmark_output_formats.py` |

### Choose By Task

| If you want to... | Use this first | Then look at |
| --- | --- | --- |
| run one small case from the terminal | `tutorials/run_quick_output_and_plot.py` | `getting_started/write_sfincs_output_cli.py` |
| call SFINCS-JAX from Python | `getting_started/write_sfincs_output_python.py` | `getting_started/write_and_plot_multiple_formats.py` |
| understand transport matrices | `transport/transport_matrix_rhsmode2_and_rhsmode3.py` | `transport/transport_matrix_rhsmode2_scheme11_and_scheme5.py` |
| differentiate a solve or residual | `tutorials/02_transport_and_autodiff.ipynb` | `autodiff/implicit_diff_through_gmres_solve_scheme5.py` |
| compute bootstrap current and compare Redl | `tutorials/03_bootstrap_redl_and_optimization.ipynb` | `vmec_jax_finite_beta/compare_qs_paper_sfincs_jax_redl.py` |
| add neoclassical objectives to optimization | `optimization/qa_nfp2_sfincs_jax_objectives.py` | `optimization/QA_optimization_bootstrap_current.py` |
| choose geometry, validate outputs, and benchmark CPU/GPU | `tutorials/04_geometry_validation_and_performance.ipynb` | `getting_started/write_sfincs_output_vmec.py`, `performance/benchmark_output_formats.py` |
| check CPU/GPU performance or output formats | `performance/benchmark_output_formats.py` | `performance/benchmark_transport_l11_vs_fortran.py` |
| validate against frozen SFINCS Fortran v3 data | `parity/output_parity_vs_fortran_fixture.py` | `publication_figures/` and `sfincs_examples/` |

### Application Recipes

Use this table when you know the physics or software task, but not the folder
name. The first command is the smallest useful run; the follow-up points to the
script or notebook that adds the technical detail needed for research workflows.

| Application | Smallest useful entry point | Research workflow |
| --- | --- | --- |
| CLI output and diagnostics panel | `tutorials/run_quick_output_and_plot.py` | `getting_started/write_and_plot_multiple_formats.py` |
| Analytic tokamak input | `getting_started/write_sfincs_output_tokamak.py` | `sfincs_examples/tokamak_1species_FPCollisions_noEr/input.namelist` |
| VMEC `wout_path` input | `getting_started/write_sfincs_output_vmec.py` | `vmec_jax_finite_beta/finite_beta_vmec_to_sfincs.py` |
| RHSMode=2/3 transport matrix | `transport/transport_matrix_rhsmode2_and_rhsmode3.py` | `transport/transport_matrix_rhsmode2_scheme11_and_scheme5.py` |
| Bootstrap current vs Redl | `vmec_jax_finite_beta/compare_qs_paper_sfincs_jax_redl.py` | `tutorials/03_bootstrap_redl_and_optimization.ipynb` |
| Ambipolar electric-field scan | `vmec_jax_finite_beta/finite_beta_vmec_to_sfincs.py` | `optimization/evaluate_sfincs_jax_promotion_scan.py` |
| Differentiable residual or flux | `autodiff/autodiff_gradient_nu_n_residual.py` | `autodiff/implicit_diff_through_gmres_solve_scheme5.py` |
| VMEC/Boozer/JAX workflow | `autodiff/vmec_jax_to_boozer_sfincs_pipeline.py` | `tutorials/04_geometry_validation_and_performance.ipynb` |
| QA/QI optimization objective | `optimization/qa_nfp2_sfincs_jax_objectives.py` | `optimization/QA_optimization_bootstrap_current.py` |
| CPU/GPU timing and output I/O | `performance/benchmark_output_formats.py` | `performance/benchmark_transport_l11_vs_fortran.py` |
| Frozen Fortran-v3 parity check | `parity/output_parity_vs_fortran_fixture.py` | `sfincs_examples/` for the retained upstream-style decks |

### Canonical Workflow Catalog

These are the recommended first examples for each major capability. They are
kept small enough for learning and CI checks; the final column points to the
heavier workflow when you need release-quality evidence.

| Capability | First-pass example | What it teaches | Production follow-up |
| --- | --- | --- | --- |
| CLI run and plot | `tutorials/run_quick_output_and_plot.py` | Run `sfincs_jax`, write HDF5/NetCDF/NPZ, and create a diagnostics PDF. | `getting_started/write_and_plot_multiple_formats.py` |
| Python API output | `getting_started/write_sfincs_output_python.py` | Call the output writer directly and inspect returned arrays. | `getting_started/write_sfincs_output_vmec.py` |
| Geometry setup | `getting_started/build_grids_and_geometry.py` | Build v3 grids and analytic/VMEC geometry objects. | `tutorials/04_geometry_validation_and_performance.ipynb` |
| Operator action | `getting_started/apply_collisionless_operator.py` | Apply one drift-kinetic operator term on a small grid. | `parity/collisionless_operator_matvec_parity.py` |
| Transport matrix | `transport/transport_matrix_rhsmode2_and_rhsmode3.py` | Run RHSMode=2/3 and read transport coefficients. | `transport/transport_matrix_rhsmode2_scheme11_and_scheme5.py` |
| Autodiff | `tutorials/02_transport_and_autodiff.ipynb` | Differentiate residual/transport quantities with JAX. | `autodiff/implicit_diff_through_gmres_solve_scheme5.py` |
| Bootstrap current and Redl | `tutorials/03_bootstrap_redl_and_optimization.ipynb` | Compare kinetic bootstrap current with a Redl-formula workflow. | `vmec_jax_finite_beta/compare_qs_paper_sfincs_jax_redl.py` |
| Optimization objectives | `optimization/qa_nfp2_sfincs_jax_objectives.py` | Add neoclassical objectives to a QA optimization workflow. | `optimization/QA_optimization_bootstrap_current.py` |
| Frozen Fortran-v3 parity | `parity/output_parity_vs_fortran_fixture.py` | Compare output fields against checked frozen references. | `sfincs_examples/` for retained upstream-style decks |
| CPU/GPU performance | `performance/benchmark_output_formats.py` | Time output formats and inspect memory behavior. | `performance/benchmark_transport_l11_vs_fortran.py` |

### Top-Level Folder Categories

The example tree has a small number of top-level domains. The category tells
you whether a folder is a first-pass learning surface, a capability workflow, a
validation or benchmark workflow, or reference data.

| Category | Folders | Use when |
| --- | --- | --- |
| `learning` | `tutorials/`, `getting_started/` | You want to learn the CLI, Python API, plots, output formats, and first operator/geometry concepts. |
| `capability` | `transport/`, `autodiff/`, `optimization/`, `vmec_jax_finite_beta/` | You need a specific physics or differentiability workflow. |
| `validation` | `parity/`, `performance/`, `publication_figures/`, `paper_benchmarks/` | You need parity checks, runtime/memory evidence, regenerated documentation figures, or methods-paper benchmark cases. |
| `reference` | `data/`, `sfincs_examples/` | You need small shared inputs or recognizable SFINCS Fortran v3 decks for audits. |

### Folder Map

- `tutorials/`: notebook-led learning path plus one fast script that writes
  output files and a diagnostics panel.
- `getting_started/`: minimal CLI and Python workflows, plotting, file formats,
  analytic tokamak geometry, and VMEC `wout_path` usage.
- `transport/`: RHSMode=2/3 transport-matrix workflows and scan-postprocessing
  entry points.
- `autodiff/`: JAX `grad`, JVP/VJP, implicit differentiation, and
  VMEC/Boozer-to-SFINCS differentiable workflow examples.
- `optimization/`: JAX-native proxy objectives, kinetic promotion scripts, and
  QA bootstrap-current optimization helpers.
- `vmec_jax_finite_beta/`: finite-beta VMEC-to-SFINCS radial profiles,
  ambipolar `E_r`, bootstrap-current comparisons, Redl formula checks, and
  convergence plots.
- `parity/`: frozen-reference parity checks against SFINCS Fortran v3 outputs
  without requiring Fortran in CI.
- `performance/`: JIT, memory, output-format, transport-worker scaling, and
  production-floor benchmark drivers. Single-case sharded and multi-GPU
  campaign drivers stay outside the stable example tree until they pass the
  stable-core gates.
- `publication_figures/`: scripts that regenerate documentation and paper
  figures from checked summaries or explicit benchmark runs.
- `paper_benchmarks/`: community-standard benchmark cases for the methods
  paper (ICNTS-style monoenergetic coefficient scans on W7-X, TJ-II, and
  HSX with Fortran v3 cross-checks, the low-collisionality Shaing-Callen
  bootstrap-convergence study, the kinetic-solver-in-the-loop
  bootstrap-consistency workflow on a finite-beta QA equilibrium, plus the
  AD-vs-FD gradient-verification table), each writing a figure and a JSON
  record.
- `sfincs_examples/`: vendored upstream SFINCS v3 example inputs plus helpers
  used for parity and benchmark-suite audits, not the recommended starting
  point for first-time users.
- `data/`: small input data needed by public VMEC and teaching examples.

### Notebook Guides

The tutorial notebooks are the recommended classroom/user-facing guides. They
show commands, equations, code, plotting calls, interpretation notes, and links
to the matching topic scripts. Heavy production and Fortran-overlay commands are
shown explicitly but are not run automatically by CI; fast scripts and notebook
structure are tested so the learning path stays usable.

### Setup

From the repo root:

```bash
cd sfincs_jax
pip install -e ".[dev]"
```

The standard install already includes `matplotlib` and `netCDF4`, so plotting
examples, `sfincs_jax --plot`, and `--out sfincsOutput.nc` work without extra
dependencies.

For optimization examples that use `optax`:

```bash
pip install optax
```

The finite-beta VMEC-JAX example requires an importable `vmec_jax` installation.
If you have a source checkout, point the example at it with:

```bash
export SFINCS_JAX_VMEC_JAX_ROOT=/path/to/vmec_jax
```

### Running

Each example is a standalone script:

```bash
python examples/getting_started/build_grids_and_geometry.py
```

Common entry points:

- Browse examples by task: `examples/list_workflows.py`
- Tutorial notebook index: `examples/tutorials/README.md`
- Fast tutorial output writer/plotter: `examples/tutorials/run_quick_output_and_plot.py`
- Write `sfincsOutput.h5` via Python: `examples/getting_started/write_sfincs_output_python.py`
- Write `sfincsOutput.h5` via CLI: `examples/getting_started/write_sfincs_output_cli.py`
- Write `.h5`, `.nc`, and `.npz`, then build a PDF diagnostics panel: `examples/getting_started/write_and_plot_multiple_formats.py`
- Analytic tokamak example (`geometryScheme=1`): `examples/getting_started/write_sfincs_output_tokamak.py`
- VMEC example (`geometryScheme=5`, `wout_path` override): `examples/getting_started/write_sfincs_output_vmec.py`
- Finite-beta VMEC-JAX to convergence-gated SFINCS radial Er and bootstrap-current profiles: `examples/vmec_jax_finite_beta/finite_beta_vmec_to_sfincs.py`
- Finite-beta kinetic/angular/root-bracket convergence scan from cached outputs: `examples/vmec_jax_finite_beta/plot_convergence_scan.py`
- Plot `.h5`, `.nc`, or `.npz` output: `examples/getting_started/plot_sfincs_output.py`
- Output parity vs Fortran fixture: `examples/parity/output_parity_vs_fortran_fixture.py`
- Transport matrices (RHSMode 2/3): `examples/transport/transport_matrix_rhsmode2_and_rhsmode3.py`
- Differentiate a residual norm w.r.t. `nu_n`: `examples/autodiff/autodiff_gradient_nu_n_residual.py`
- Implicit differentiation through BiCGStab: `examples/autodiff/implicit_diff_through_gmres_solve_scheme5.py --solver bicgstab`
- Output writer/readback benchmark: `examples/performance/benchmark_output_formats.py --repeats 5`

### Validation And Benchmark Sweeps

The example tree includes scripts for parity, performance, and production
benchmark sweeps, but those workflows are not the recommended starting point
for a first run. Use the compact tutorial and topic scripts above for learning;
use the validation and benchmark tooling when you need reproducible evidence
for runtime, memory, parity, or solver-policy changes.

- `examples/parity/`: frozen-reference output checks that do not require a
  local SFINCS Fortran v3 executable.
- `examples/performance/`: CPU/GPU timing, output-format, and transport-worker
  benchmark drivers.
- `examples/publication_figures/`: scripts that rebuild documentation and paper
  figures from checked summaries or explicit benchmark runs.
- `examples/sfincs_examples/`: vendored upstream SFINCS v3 inputs used by the
  scaled-suite runner and release audits.

Detailed release-audit commands live in `docs/parity.rst`,
`docs/performance.rst`, and `docs/fortran_examples.rst`. The scaled-suite
driver is `python -m sfincs_jax.validation.suite scaled`.
