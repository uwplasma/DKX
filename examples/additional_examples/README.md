# Additional Reference Inputs

This folder contains compact standalone input decks that are part of the
validation and benchmark suite but are not intended as the first learning path
for SFINCS-JAX. Start with `../tutorials/` or `../getting_started/` for
walkthrough examples.

## Contents

- `input.namelist`: QI/VMEC RHSMode=1 reference input used by the scaled example
  suite, QI robustness checks, and production-resolution benchmark manifests.
  The referenced VMEC equilibrium is release-hosted through the validation data
  manifest instead of being stored directly in the git repository.

## How To Use

For a quick user-facing run, prefer:

```bash
python examples/tutorials/run_quick_output_and_plot.py
```

Use this folder when you need to reproduce the named `additional_examples`
benchmark case or when a validation script explicitly points here. Large
equilibrium files required by this input should be fetched through
`sfincs_jax.validation.data_fetch`, not copied into this directory.
