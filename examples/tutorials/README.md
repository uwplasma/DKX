# Tutorials

This folder is the compact learning path for first-time `dkx` users. It does
not replace the topic folders under `examples/`; it points to the best first
scripts and explains the physics and diagnostics in a notebook format.

## Recommended Order

0. `00_start_here.ipynb`
   Choose the right learning path, verify that the first-run assets exist, and
   see a lightweight map from physics goals to example folders.
1. `01_cli_outputs_and_plots.ipynb`
   Run a small input, write HDF5/NetCDF/NPZ outputs, inspect the datasets, and
   build a diagnostics PDF.
2. `02_transport_and_autodiff.ipynb`
   Compute RHSMode=2/3 transport matrices and differentiate a residual-based
   objective with JAX.
3. `03_bootstrap_redl_and_optimization.ipynb`
   Compare kinetic bootstrap current against the Redl fit and see how
   neoclassical objectives enter a QA optimization loop.
4. `04_geometry_validation_and_performance.ipynb`
   Choose analytic, Boozer, or VMEC geometry workflows; connect outputs to
   validation/parity checks; and find CPU/GPU performance and parallelism
   examples without launching long runs from the notebook.

## Runnable Script

For a fast script that writes files to its own output directory:

```bash
python examples/tutorials/run_quick_output_and_plot.py
```

The script writes `sfincsOutput_tutorial.h5`, `.nc`, `.npz`, and
`sfincsOutput_tutorial_summary.pdf` into
`examples/output/run_quick_output_and_plot/`, without requiring SFINCS
Fortran v3.  Edit the `OUTPUT_DIR` parameter at the top to change where they
go.

## Where To Go Next

- Use `examples/getting_started/` for minimal command-line and Python output
  workflows.
- Use `examples/transport/` for RHSMode=2/3 transport-matrix examples.
- Use `examples/autodiff/` for JAX gradient, JVP/VJP, and implicit-solve
  examples.
- Use `examples/vmex_finite_beta/` for VMEC, Redl, ambipolar electric
  field, and bootstrap-current workflows.
- Use `examples/optimization/` for proxy objectives and kinetic promotion
  scripts.
- Use `examples/parity/` and `examples/performance/` for validation and
  benchmarking rather than first-pass learning.

The notebooks avoid large checked-in outputs. Heavy reference data are fetched
through the package data cache or supplied by the user when a workflow needs an
external VMEC/Zenodo/SFINCS Fortran v3 artifact.
