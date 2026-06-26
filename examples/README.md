## Examples

This directory is the learning surface for `sfincs_jax`. Start with
`tutorials/` if you are new to the code; jump to the topic folders if you
already know which workflow you need. All first-pass examples avoid a local
SFINCS Fortran v3 executable. Parity and benchmark scripts use frozen
references or optional local Fortran only when explicitly requested.

### Learning Path

| Step | Goal | Start here |
| --- | --- | --- |
| 1 | Run the CLI, write output files, and plot diagnostics | `tutorials/01_cli_outputs_and_plots.ipynb`, `tutorials/run_quick_output_and_plot.py` |
| 2 | Compute transport matrices and see autodiff | `tutorials/02_transport_and_autodiff.ipynb` |
| 3 | Compare bootstrap current with Redl and see optimization hooks | `tutorials/03_bootstrap_redl_and_optimization.ipynb` |
| 4 | Understand grids, geometry, and one operator action | `getting_started/build_grids_and_geometry.py`, `getting_started/apply_collisionless_operator.py` |
| 5 | Compare outputs with frozen SFINCS Fortran v3 references | `parity/output_parity_vs_fortran_fixture.py` |
| 6 | Profile CPU/GPU, JIT, output formats, and parallelism | `performance/benchmark_sharded_solve_scaling.py`, `performance/benchmark_output_formats.py` |

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
  point for new users.
- `upstream/` and `additional_examples/`: curated reference inputs used by
  tests, docs, and validation lanes.

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

### Scaled upstream example sweep

The vendored `examples/sfincs_examples/` inputs currently match the original
Fortran v3 example inputs exactly. For reproducible benchmarking, use the
original upstream tree as the resolution reference and set `--scale-factor`
relative to that baseline. `1.0` means the original v3 example resolution, and
values below `1.0` reduce `NTHETA/NZETA/NX/NXI` consistently from that upstream
reference.

To compare Fortran vs `sfincs_jax` runtime, memory, output parity, and print
parity at the original upstream resolution, use:

```bash
cd /Users/rogeriojorge/local/tests/sfincs_jax
python scripts/run_scaled_example_suite.py \
  --examples-root examples/sfincs_examples \
  --resolution-reference-root /Users/rogeriojorge/local/tests/sfincs_original/fortran/version3/examples \
  --fortran-exe /Users/rogeriojorge/local/tests/sfincs/fortran/version3/sfincs \
  --out-root tests/scaled_example_suite_ref_cpu_local \
  --timeout-s 240 \
  --max-attempts 2 \
  --scale-factor 1.0
```

The script keeps the runnable input text from `--examples-root`, rewrites only
`NTHETA/NZETA/NX/NXI` from the matching case in
`--resolution-reference-root`, includes
`examples/additional_examples/input.namelist`, and writes per-case outputs plus
`suite_report.json`, `suite_report_strict.json`, `suite_status*.rst`,
`run_manifest.json`, and `summary.md` into the chosen `--out-root`.
These suite-level artifacts are checkpointed after every finished case, so an
interrupted long run still leaves a usable partial audit instead of only
per-case directories.
If you restart a long sweep after changing `sfincs_jax` code or after a bad
launch, reuse the same `--out-root` only with `--reset-report`, otherwise the
old case rows remain merged into the new `suite_report*.json` checkpoint files.
For the legacy `examples/upstream/fortran_multispecies` tree, the Fortran lane
also canonicalizes the old pre-v3 namelist groups and aliases into the v3
input shape expected by the reference executable, while the `sfincs_jax` lane
honors those same legacy aliases directly.

To separate reference generation from JAX benchmarking, first create a stable
CPU reference root, then benchmark CPU or GPU JAX runs against that fixed
reference without re-running Fortran:

```bash
cd /Users/rogeriojorge/local/tests/sfincs_jax
python scripts/run_scaled_example_suite.py \
  --examples-root examples/sfincs_examples \
  --resolution-reference-root /Users/rogeriojorge/local/tests/sfincs_original/fortran/version3/examples \
  --fortran-exe /Users/rogeriojorge/local/tests/sfincs/fortran/version3/sfincs \
  --out-root tests/gating_reference_cpu \
  --pattern '^(tokamak_1species_FPCollisions_noEr|inductiveE_noEr)$' \
  --scale-factor 1.0 \
  --max-attempts 1

python scripts/run_scaled_example_suite.py \
  --examples-root examples/sfincs_examples \
  --resolution-reference-root /Users/rogeriojorge/local/tests/sfincs_original/fortran/version3/examples \
  --reference-results-root tests/gating_reference_cpu \
  --out-root tests/gating_cpu_from_ref \
  --pattern '^(tokamak_1species_FPCollisions_noEr|inductiveE_noEr)$' \
  --scale-factor 1.0 \
  --max-attempts 1
```

This keeps the Fortran reference H5/log files fixed across lanes, which is
useful when comparing local CPU and remote GPU runs against the same baseline.

For a full-sweep audit on laptops or workstations where the original v3
resolution is too expensive, keep the upstream resolution ratios but reduce the
global scale factor instead of hand-editing individual examples:

```bash
cd /Users/rogeriojorge/local/tests/sfincs_jax
python scripts/run_scaled_example_suite.py \
  --examples-root examples/sfincs_examples \
  --extra-input examples/additional_examples/input.namelist \
  --resolution-reference-root /Users/rogeriojorge/local/tests/sfincs_original/fortran/version3/examples \
  --fortran-exe /Users/rogeriojorge/local/tests/sfincs/fortran/version3/sfincs \
  --out-root tests/scaled_example_suite_ref_cpu_full \
  --scale-factor 0.75 \
  --timeout-s 3600 \
  --max-attempts 1
```

This preserves the original example mix while reducing `NTHETA/NZETA/NX/NXI`
consistently from the upstream reference tree.

Use these reduced-scale full sweeps as an audit tool for runtime, memory, and
solver-branch fragility. They preserve the upstream resolution ratios, but they
can still shift conditioning enough to expose branch-sensitive mismatches that
do not appear at the original example resolution. The release gate remains the
standard reduced-suite comparisons plus targeted original-resolution examples.

### QI Seed Robustness Lane

The quasi-isodynamic VMEC deck in `examples/additional_examples/input.namelist`
can be expanded into a deterministic multi-seed smoke lane without editing the
source example:

```bash
python scripts/run_qi_seed_robustness.py \
  --out-root tests/qi_seed_robustness \
  --seeds 0 1 2 \
  --execute \
  --max-residual-ratio 1 \
  --require-converged \
  --clean
```

This writes per-seed `input.namelist` files, localizes the QI VMEC equilibrium
next to each case, applies deterministic `nu_n` and `Er` perturbations, and
records the commands in `manifest.json`. With `--execute`, it runs each seed
through `sfincs_jax write-output` and records stdout/stderr, return codes,
output presence, solver-trace residual metadata, aggregate summary fields, and
the optional promotion gates in the same manifest.

The execute smoke defaults to the public `auto` CLI solver policy. The bounded
checked multi-seed smoke now keeps the public `auto` method, auto-selects the
fast dense full-FP path for three neighboring QI seeds at `7 x 13 x 25 x 4`, and
converges below the requested residual target. Check
`execution.gates.passed`, `execution.summary.max_residual_ratio`, and
`execution.results[].solver_trace_summary.converged` before treating any new
result as converged evidence. To probe another solver path explicitly, override
the method:

```bash
python scripts/run_qi_seed_robustness.py \
  --out-root tests/qi_seed_robustness \
  --seeds 0 \
  --execute \
  --solve-method dense \
  --timeout-s 300 \
  --clean
```
