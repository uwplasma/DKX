# Upstream Fortran v3 example suite (vendored)

This folder is a **copy** of the upstream SFINCS Fortran v3 example suite
(`sfincs/fortran/version3/examples`) plus the upstream postprocessing scripts in
`sfincs/fortran/version3/utils`.

Goals:

- Let SFINCS users run the **same example inputs** with `dkx`.
- Provide a convenient place to benchmark against the compiled Fortran v3 executable.
- Keep the upstream plotting scripts available for supported output-parity workflows.

## Supported Scope

This folder includes more upstream modes than the first-pass public examples.
The supported scope is the subset documented by the parity and Fortran-example
audit pages.

In general, you should expect:

- `dkx write-output` to work for a broad subset of inputs (geometry + `sfincsOutput.h5` fields).
- `dkx write-output --compute-transport-matrix` to work for the **RHSMode=2/3** examples used by the
  transport-matrix parity fixtures (small cases).

For the support matrix, see:

- `docs/parity.rst`
- `docs/fortran_examples.rst` (auto-generated audit table)

## Running the vendored suite

From the `dkx` repository root:

```bash
python examples/sfincs_examples/run_dkx.py --write-output
```

To run a subset:

```bash
python examples/sfincs_examples/run_dkx.py --write-output --pattern monoenergetic_geometryScheme11
```

To also run the compiled Fortran v3 executable for comparison (slow):

```bash
python examples/sfincs_examples/run_dkx.py \\
  --write-output \\
  --compare-fortran \\
  --fortran-exe ../sfincs/fortran/version3/sfincs
```

## Upstream utils

The scripts in `utils/` are copied from upstream (e.g. `utils/sfincsScanPlot_1`)
and lightly ported for Python 3, noninteractive plotting, and local
`dkx` output generation through the sibling driver helper. They expect
specific fields inside `sfincsOutput.h5`.

Many of these scripts also read default parameters from `globalVariables.F90`. In the upstream
layout, this file lives next to `utils/`, so this repo vendors it as:

- `examples/sfincs_examples/globalVariables.F90`

`dkx` writes the fields needed by the transport-matrix plotting scripts for RHSMode=2/3:
`transportMatrix`, `FSABFlow`, `particleFlux_vm_psiHat`, and `heatFlux_vm_psiHat`.

Broader postprocessing parity is tracked in the validation and output-key
coverage tests before additional datasets are documented as supported.

To run an upstream `utils/` script in a non-interactive way:

```bash
dkx postprocess-upstream --case-dir /path/to/case --util sfincsScanPlot_1 -- pdf
```

If you are not running from a `dkx` repo checkout, set:

```bash
export DKX_UPSTREAM_UTILS_DIR=/path/to/utils
```
