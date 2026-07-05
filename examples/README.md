## Examples

This directory is the learning surface for `sfincs_jax`. Start with
`tutorials/` if you are new to the code; jump to the topic folders if you
already know which workflow you need. All first-pass examples avoid a local
SFINCS Fortran v3 executable. Parity and benchmark scripts use frozen
references or optional local Fortran only when explicitly requested.

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
| 7 | Profile CPU/GPU, JIT, output formats, and parallelism | `performance/benchmark_sharded_solve_scaling.py`, `performance/benchmark_output_formats.py` |

### Choose By Task

| If you want to... | Use this first | Then look at |
| --- | --- | --- |
| run one small case from the terminal | `tutorials/run_quick_output_and_plot.py` | `getting_started/write_sfincs_output_cli.py` |
| call SFINCS-JAX from Python | `getting_started/write_sfincs_output_python.py` | `getting_started/write_and_plot_multiple_formats.py` |
| understand transport matrices | `transport/transport_matrix_rhsmode2_and_rhsmode3.py` | `transport/transport_matrix_recycle_demo.py` |
| differentiate a solve or residual | `tutorials/02_transport_and_autodiff.ipynb` | `autodiff/implicit_diff_through_gmres_solve_scheme5.py` |
| compute bootstrap current and compare Redl | `tutorials/03_bootstrap_redl_and_optimization.ipynb` | `vmec_jax_finite_beta/compare_qs_paper_sfincs_jax_redl.py` |
| add neoclassical objectives to optimization | `optimization/qa_nfp2_sfincs_jax_objectives.py` | `optimization/QA_optimization_bootstrap_current.py` |
| choose geometry, validate outputs, and benchmark CPU/GPU | `tutorials/04_geometry_validation_and_performance.ipynb` | `getting_started/write_sfincs_output_vmec.py`, `performance/benchmark_transport_parallel_scaling.py` |
| check CPU/GPU performance or output formats | `performance/benchmark_output_formats.py` | `performance/benchmark_sharded_solve_scaling.py` |
| validate against frozen SFINCS Fortran v3 data | `parity/output_parity_vs_fortran_fixture.py` | `publication_figures/` and `sfincs_examples/` |
| reproduce the QI benchmark input used in validation reports | `data/qi_nfp2_reference.input.namelist` | `optimization/materialize_qi_nfp2_promotion_input.py` |

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
| VMEC/Boozer/JAX handoff | `autodiff/vmec_jax_to_boozer_sfincs_pipeline.py` | `tutorials/04_geometry_validation_and_performance.ipynb` |
| QA/QI optimization objective | `optimization/qa_nfp2_sfincs_jax_objectives.py` | `optimization/QA_optimization_bootstrap_current.py` |
| CPU/GPU timing and output I/O | `performance/benchmark_output_formats.py` | `performance/benchmark_transport_parallel_scaling.py` |
| Frozen Fortran-v3 parity check | `parity/output_parity_vs_fortran_fixture.py` | `publication_figures/generate_fortran_suite_benchmark_summary.py` |

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
| Frozen Fortran-v3 parity | `parity/output_parity_vs_fortran_fixture.py` | Compare output fields against checked frozen references. | `publication_figures/generate_fortran_suite_benchmark_summary.py` |
| CPU/GPU performance | `performance/benchmark_output_formats.py` | Time output formats and inspect memory behavior. | `performance/benchmark_transport_parallel_scaling.py` |

### Folder Map

- `tutorials/`: notebook-led learning path plus one fast script that writes
  output files and a diagnostics panel.
- `getting_started/`: minimal CLI and Python workflows, plotting, file formats,
  analytic tokamak geometry, and VMEC `wout_path` usage.
- `transport/`: RHSMode=2/3 transport-matrix workflows, Krylov recycling, and
  scan-plot utilities.
- `autodiff/`: JAX `grad`, JVP/VJP, implicit differentiation, and
  VMEC/Boozer-to-SFINCS differentiable handoff examples.
- `optimization/`: JAX-native proxy objectives, kinetic promotion scripts, QI
  electron-root screens, and QA bootstrap-current optimization helpers.
- `vmec_jax_finite_beta/`: finite-beta VMEC-to-SFINCS radial profiles,
  ambipolar `E_r`, bootstrap-current comparisons, Redl formula checks, and
  convergence plots.
- `parity/`: frozen-reference parity checks against SFINCS Fortran v3 outputs
  without requiring Fortran in CI.
- `performance/`: JIT, memory, output-format, sharding, multi-GPU, and
  production-floor benchmark drivers.
- `publication_figures/`: scripts that regenerate documentation and paper
  figures from checked summaries or explicit benchmark runs.
- `sfincs_examples/`: vendored upstream SFINCS v3 example inputs plus helpers
  used for parity and benchmark-suite audits, not the recommended starting
  point for first-time users.
- `upstream/`: curated upstream SFINCS v3 inputs used by tests, docs, and
  validation lanes.
- `data/`: small input data needed by public examples, including the QI/VMEC
  reference namelist used by validation and benchmark scripts.
- `utils/`: helper code shared by example scripts; users normally call the
  topic scripts rather than importing this folder directly.

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
- Transport matrices with Krylov recycling: `examples/transport/transport_matrix_recycle_demo.py`
- Differentiate a residual norm w.r.t. `nu_n`: `examples/autodiff/autodiff_gradient_nu_n_residual.py`
- Implicit differentiation through BiCGStab: `examples/autodiff/implicit_diff_through_gmres_solve_scheme5.py --solver bicgstab`
- CPU sharding benchmark: `examples/performance/benchmark_sharded_solve_scaling.py --backend cpu --devices 1 2 4 8`
- Transport-worker benchmark: `examples/performance/benchmark_transport_parallel_scaling.py --workers 1 2 4`
- Two-GPU throughput benchmark: `examples/performance/benchmark_multi_gpu_case_throughput.py`
- Output writer/readback benchmark: `examples/performance/benchmark_output_formats.py --repeats 5`

### Validation And Benchmark Sweeps

The example tree includes scripts for parity, performance, and production
benchmark sweeps, but those workflows are not the recommended starting point
for a first run. Use the compact tutorial and topic scripts above for learning;
use the validation and benchmark tooling when you need reproducible evidence
for runtime, memory, parity, or solver-policy changes.

- `examples/parity/`: frozen-reference output checks that do not require a
  local SFINCS Fortran v3 executable.
- `examples/performance/`: CPU/GPU timing, output-format, sharding,
  multi-worker, and optional ecosystem benchmark drivers.
- `examples/publication_figures/`: scripts that rebuild documentation and paper
  figures from checked summaries or explicit benchmark runs.
- `examples/sfincs_examples/`: vendored upstream SFINCS v3 inputs used by the
  scaled-suite runner and release audits.

Detailed release-audit commands live in `docs/parity.rst`,
`docs/performance.rst`, and `docs/fortran_examples.rst`. The scaled-suite
driver is `../scripts/run_scaled_example_suite.py`, and the QI robustness
campaign driver is `../scripts/run_qi_seed_robustness.py`.
