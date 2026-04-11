# sfincs_jax

`sfincs_jax` is a JAX implementation of SFINCS v3 that solves the same neoclassical drift-kinetic problem with matching normalizations, geometry conventions, and output format (`sfincsOutput.h5`).

On the current `main` branch, the full vendored example suite runs cleanly on CPU and GPU with dataset-level parity against SFINCS Fortran v3. The default CLI path is tuned for robust explicit solves and practical throughput, while the Python API can opt into differentiable solve paths when gradients matter.

It is designed for:

- high-performance runs on CPU/GPU,
- memory-efficient large solves,
- end-to-end differentiable workflows.

![Runtime and parity snapshot](docs/_static/figures/sfincs_vs_sfincs_jax_l11_runtime_2x2.png)

The figure above shows a representative transport benchmark. In the full 39-case example-suite audit below, all cases complete on CPU and GPU with no `jax_error`, no `max_attempts`, no practical mismatches, and no strict mismatches.

## Installation

Install from PyPI:

```bash
pip install sfincs_jax
```

Install from source:

```bash
git clone https://github.com/uwplasma/sfincs_jax.git
cd sfincs_jax
pip install .
```

Development install:

```bash
git clone https://github.com/uwplasma/sfincs_jax.git
cd sfincs_jax
pip install -e ".[dev]"
```

## Quick Start (Python)

Read a namelist, run `sfincs_jax`, write `sfincsOutput.h5`, and inspect results directly in memory:

```python
from pathlib import Path

from sfincs_jax.io import write_sfincs_jax_output_h5

input_namelist = Path("input.namelist")
out_path, results = write_sfincs_jax_output_h5(
    input_namelist=input_namelist,
    output_path=Path("sfincsOutput.h5"),
    return_results=True,
)

print("Wrote:", out_path)
print("Available datasets:", len(results))
print("Example key:", "particleFlux_vm_psiHat" in results)
```

If you need to override the equilibrium file without editing the namelist, pass
``equilibrium_file=...`` or the VMEC-friendly alias ``wout_path=...``:

```python
write_sfincs_jax_output_h5(
    input_namelist=input_namelist,
    output_path=Path("sfincsOutput.h5"),
    wout_path=Path("/path/to/wout.nc"),
)
```

`sfincs_jax write-output` and `write_sfincs_jax_output_h5(...)` use the explicit
performance-oriented solve path by default. Request the implicit/differentiable linear-solve path only when
you need it:

```python
write_sfincs_jax_output_h5(
    input_namelist=input_namelist,
    output_path=Path("sfincsOutput.h5"),
    differentiable=True,
)
```

## Executable (CLI)

You can run `sfincs_jax` from anywhere in your terminal. You do not need to be inside the repository folder.

Run an input file (default behavior, same invocation style as Fortran SFINCS):

```bash
sfincs_jax /path/to/input.namelist
```

Write output explicitly:

```bash
sfincs_jax write-output --input /path/to/input.namelist --out /path/to/sfincsOutput.h5
```

Override the equilibrium file at the CLI without changing ``input.namelist``:

```bash
sfincs_jax write-output \
  --input /path/to/input.namelist \
  --out /path/to/sfincsOutput.h5 \
  --wout-path /path/to/wout.nc
```

The bare ``sfincs_jax /path/to/input.namelist`` form accepts the same
``--equilibrium-file`` and ``--wout-path`` overrides.

Parallel CLI controls are now first-class:

```bash
# Multi-core CPU host sharding on one node
sfincs_jax --cores 8 --shard-axis auto /path/to/input.namelist

# Parallel transport-matrix RHS solves
sfincs_jax transport-matrix-v3 \
  --input /path/to/input.namelist \
  --transport-workers 4

# One-node multi-GPU sharded solve (experimental for very large single-RHS cases)
CUDA_VISIBLE_DEVICES=0,1 \
sfincs_jax write-output \
  --input /path/to/input.namelist \
  --shard-axis theta \
  --distributed-gmres auto

# Multi-host JAX distributed bootstrap
sfincs_jax write-output \
  --input /path/to/input.namelist \
  --distributed \
  --process-count 8 \
  --process-id ${RANK} \
  --coordinator-address node0 \
  --coordinator-port 1234
```

Use `-v` to have the executable print the active parallel runtime summary
(cores, shard axis, transport workers, distributed Krylov mode, and multi-host
bootstrap fields) before the solve starts.

Current recommendation:

- CPU host sharding is supported and deterministic, but the measured speedup is
  still case-dependent.
- Use one GPU per case or scan point for production throughput today.
- Multi-GPU single-case sharding is available for benchmarking and very large
  runs, but it remains experimental and is not yet the default recommendation.

Compare two outputs:

```bash
sfincs_jax compare-h5 --a sfincsOutput_jax.h5 --b sfincsOutput_fortran.h5
```

Advanced CLI/solver options are documented in `docs/usage.rst` and `docs/performance_techniques.rst`.

## What Differs From Fortran v3

`sfincs_jax` reproduces the SFINCS v3 equations, normalizations, geometry conventions, and output datasets for the supported examples, but the implementation strategy differs in a few important ways:

- the default CLI path uses an explicit performance-oriented solve strategy instead of trying to mirror every PETSc iteration path exactly,
- the Python API can switch to differentiable solve paths when end-to-end sensitivities are needed,
- CPU runs lean on JIT-cached kernels and selected host sparse factorizations for hard linear branches,
- GPU runs keep operator applications on device, then fall back to accelerator-safe or host rescue paths only when conditioning or memory demands it,
- and terminal output is intentionally a superset of Fortran SFINCS output so debugging information is available without losing Fortran-visible signals.

The detailed equations and normalization conventions are documented in `docs/system_equations.rst`, `docs/normalizations.rst`, and `docs/method.rst`. CPU/GPU-specific implementation notes are documented in `docs/performance.rst` and `docs/performance_techniques.rst`.

## Current Example-Suite Audit

Regenerate this block from the current `main` working tree with:

```bash
python scripts/run_scaled_example_suite.py \
  --examples-root examples/sfincs_examples \
  --resolution-reference-root /Users/rogeriojorge/local/tests/sfincs_original/fortran/version3/examples \
  --fortran-exe /Users/rogeriojorge/local/tests/sfincs/fortran/version3/sfincs \
  --out-root tests/scaled_example_suite_fast_cpu_full_v7_refresh \
  --scale-factor 1.0 \
  --runtime-target-basis fortran \
  --fortran-min-runtime-s 1.0 \
  --fortran-max-runtime-s 20.0 \
  --runtime-adjustment-iters 3
python scripts/generate_readme_fast_branch_audit.py \
  --out-root tests/scaled_example_suite_fast_cpu_full_v7_refresh \
  --gpu-out-root tests/scaled_example_suite_fast_gpu_full_v11_refresh
```

The benchmark policy on `main` is:

- start from the original Fortran v3 example resolution,
- only downscale when a case is too expensive for a practical suite run,
- benchmark JAX CPU and GPU against a frozen CPU-generated Fortran reference root,
- and never intentionally push a reduced case below about `1s` of Fortran wall time unless
  the original example is already that small.

That avoids the misleading sub-second Fortran rows that came from blind global downscaling,
keeps the GPU lane tied to a deterministic reference, and makes the additional example part
of the same artifact set as the standard suite.

<!-- BEGIN FAST_BRANCH_AUDIT -->
Current `main` CPU audit comes from `tests/scaled_example_suite_fast_cpu_full_v7_refresh`.
Matching frozen-reference GPU audit comes from `tests/scaled_example_suite_fast_gpu_full_v11_refresh`.

- Recorded cases: `39/39`
- Practical status counts: `parity_ok=39`
- Strict status counts: `parity_ok=39`
- GPU practical status counts: `parity_ok=39`
- GPU strict status counts: `parity_ok=39`
- Resolution policy: `reference_first_runtime_window, scale_factor=1.0, runtime_basis=fortran, fortran_min=1.0, fortran_max=None, adjust_iters=0`
- Remaining cases: none
- Additional example: `parity_ok` on CPU and `parity_ok` on GPU

Current mismatches:
- CPU practical mismatches: none
- CPU strict mismatches: none
- GPU practical/strict mismatches: none

Full per-case runtime / memory table:
| Case | Fortran CPU(s) | JAX CPU(s) | CPU x | JAX GPU(s) | GPU x | Fortran MB | JAX CPU MB | CPU MB x | JAX GPU MB | GPU MB x | CPU mismatch | GPU mismatch | CPU print | GPU print | CPU status | GPU status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- |
| `HSX_FPCollisions_DKESTrajectories` | 29.664 | 3.381 | 0.11x | 5.203 | 0.18x | 103.0 | 477.0 | 4.63x | 890.2 | 8.64x | 0/193 (strict 0/193) | 0/193 (strict 0/193) | 9/9 | 9/9 | parity_ok | parity_ok |
| `HSX_FPCollisions_fullTrajectories` | 88.504 | 5.132 | 0.06x | 5.254 | 0.06x | 100.8 | 497.0 | 4.93x | 896.9 | 8.90x | 0/193 (strict 0/193) | 0/193 (strict 0/193) | 9/9 | 9/9 | parity_ok | parity_ok |
| `HSX_PASCollisions_DKESTrajectories` | 0.994 | 7.220 | 7.26x | 11.142 | 11.21x | 112.0 | 2086.5 | 18.62x | 1420.9 | 12.68x | 0/123 (strict 0/123) | 0/123 (strict 0/123) | 7/7 | 7/7 | parity_ok | parity_ok |
| `HSX_PASCollisions_fullTrajectories` | 2.510 | 4.567 | 1.82x | 9.584 | 3.82x | 179.2 | 2197.5 | 12.26x | 2031.4 | 11.34x | 0/193 (strict 0/193) | 0/193 (strict 0/193) | 9/9 | 9/9 | parity_ok | parity_ok |
| `additional_examples` | 120.074 | 1.592 | 0.01x | 2.886 | 0.02x | 102.1 | 417.6 | 4.09x | 854.3 | 8.37x | 0/193 (strict 0/193) | 0/193 (strict 0/193) | 9/9 | 9/9 | parity_ok | parity_ok |
| `filteredW7XNetCDF_2species_magneticDrifts_noEr` | 89.052 | 1.787 | 0.02x | 3.339 | 0.04x | 103.2 | 476.5 | 4.62x | 873.2 | 8.46x | 0/193 (strict 0/193) | 0/193 (strict 0/193) | 9/9 | 9/9 | parity_ok | parity_ok |
| `filteredW7XNetCDF_2species_magneticDrifts_withEr` | 95.440 | 2.029 | 0.02x | 3.440 | 0.04x | 96.2 | 513.1 | 5.33x | 881.8 | 9.17x | 0/193 (strict 0/193) | 0/193 (strict 0/193) | 9/9 | 9/9 | parity_ok | parity_ok |
| `filteredW7XNetCDF_2species_noEr` | 128.508 | 1.670 | 0.01x | 2.886 | 0.02x | 100.3 | 452.3 | 4.51x | 862.6 | 8.60x | 0/193 (strict 0/193) | 0/193 (strict 0/193) | 9/9 | 9/9 | parity_ok | parity_ok |
| `geometryScheme4_1species_PAS_withEr_DKESTrajectories` | 1.365 | 2.402 | 1.76x | 4.752 | 3.48x | 127.3 | 1065.5 | 8.37x | 1233.6 | 9.69x | 0/207 (strict 0/207) | 0/207 (strict 0/207) | 9/9 | 9/9 | parity_ok | parity_ok |
| `geometryScheme4_2species_PAS_noEr` | 0.953 | 3.760 | 3.95x | 7.019 | 7.37x | 162.7 | 2907.0 | 17.87x | 2477.1 | 15.22x | 0/207 (strict 0/207) | 0/207 (strict 0/207) | 9/9 | 9/9 | parity_ok | parity_ok |
| `geometryScheme4_2species_noEr` | 139.240 | 1.712 | 0.01x | 3.138 | 0.02x | 92.2 | 439.9 | 4.77x | 883.5 | 9.58x | 0/207 (strict 0/207) | 0/207 (strict 0/207) | 9/9 | 9/9 | parity_ok | parity_ok |
| `geometryScheme4_2species_noEr_withPhi1InDKE` | 293.275 | 2.052 | 0.01x | 3.794 | 0.01x | 100.6 | 470.4 | 4.67x | 915.1 | 9.09x | 0/264 (strict 0/264) | 0/264 (strict 0/264) | 9/9 | 9/9 | parity_ok | parity_ok |
| `geometryScheme4_2species_noEr_withQN` | 146.734 | 1.720 | 0.01x | 3.088 | 0.02x | 95.1 | 450.7 | 4.74x | 899.7 | 9.46x | 0/264 (strict 0/264) | 0/264 (strict 0/264) | 9/9 | 9/9 | parity_ok | parity_ok |
| `geometryScheme4_2species_withEr_fullTrajectories` | 58.053 | 1.795 | 0.03x | 3.087 | 0.05x | 113.4 | 466.9 | 4.12x | 880.5 | 7.77x | 0/193 (strict 0/193) | 0/193 (strict 0/193) | 9/9 | 9/9 | parity_ok | parity_ok |
| `geometryScheme4_2species_withEr_fullTrajectories_withQN` | 211.358 | 1.907 | 0.01x | 3.390 | 0.02x | 98.8 | 481.8 | 4.88x | 907.2 | 9.18x | 0/250 (strict 0/250) | 0/250 (strict 0/250) | 9/9 | 9/9 | parity_ok | parity_ok |
| `geometryScheme5_3species_loRes` | 98.976 | 1.891 | 0.02x | 3.993 | 0.04x | 129.6 | 545.3 | 4.21x | 885.3 | 6.83x | 0/193 (strict 0/193) | 0/193 (strict 0/193) | 9/9 | 9/9 | parity_ok | parity_ok |
| `inductiveE_noEr` | 166.614 | 1.928 | 0.01x | 3.139 | 0.02x | 99.2 | 445.1 | 4.49x | 883.1 | 8.90x | 0/207 (strict 0/207) | 0/207 (strict 0/207) | 9/9 | 9/9 | parity_ok | parity_ok |
| `monoenergetic_geometryScheme1` | 0.795 | 1.759 | 2.21x | 14.621 | 18.39x | 110.2 | 666.2 | 6.04x | 958.6 | 8.70x | 0/203 (strict 0/203) | 0/203 (strict 0/203) | 9/9 | 9/9 | parity_ok | parity_ok |
| `monoenergetic_geometryScheme11` | 0.861 | 3.056 | 3.55x | 5.353 | 6.22x | 118.7 | 1101.3 | 9.28x | 957.8 | 8.07x | 0/208 (strict 0/208) | 0/208 (strict 0/208) | 9/9 | 9/9 | parity_ok | parity_ok |
| `monoenergetic_geometryScheme5_ASCII` | 1.052 | 3.410 | 3.24x | 3.938 | 3.74x | 142.1 | 2916.8 | 20.53x | 940.2 | 6.62x | 0/205 (strict 0/205) | 0/205 (strict 0/205) | 9/9 | 9/9 | parity_ok | parity_ok |
| `monoenergetic_geometryScheme5_netCDF` | 1.029 | 2.425 | 2.36x | 4.296 | 4.17x | 131.4 | 1129.1 | 8.59x | 937.3 | 7.13x | 0/205 (strict 0/205) | 0/205 (strict 0/205) | 9/9 | 9/9 | parity_ok | parity_ok |
| `quick_2species_FPCollisions_noEr` | 166.945 | 1.913 | 0.01x | 3.138 | 0.02x | 97.1 | 441.6 | 4.55x | 883.6 | 9.10x | 0/207 (strict 0/207) | 0/207 (strict 0/207) | 9/9 | 9/9 | parity_ok | parity_ok |
| `sfincsPaperFigure3_geometryScheme11_FPCollisions_2Species_DKESTrajectories` | 76.666 | 1.792 | 0.02x | 3.238 | 0.04x | 106.7 | 462.2 | 4.33x | 887.5 | 8.32x | 0/207 (strict 0/207) | 0/207 (strict 0/207) | 9/9 | 9/9 | parity_ok | parity_ok |
| `sfincsPaperFigure3_geometryScheme11_FPCollisions_2Species_fullTrajectories` | 93.439 | 2.083 | 0.02x | 3.593 | 0.04x | 94.0 | 479.5 | 5.10x | 894.3 | 9.52x | 0/207 (strict 0/207) | 0/207 (strict 0/207) | 9/9 | 9/9 | parity_ok | parity_ok |
| `sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_DKESTrajectories` | 1.104 | 3.028 | 2.74x | 6.314 | 5.72x | 130.7 | 1444.6 | 11.05x | 1559.3 | 11.93x | 0/207 (strict 0/207) | 0/207 (strict 0/207) | 9/9 | 9/9 | parity_ok | parity_ok |
| `sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories` | 1.706 | 3.445 | 2.02x | 7.420 | 4.35x | 144.6 | 2246.1 | 15.54x | 2070.4 | 14.32x | 0/207 (strict 0/207) | 0/207 (strict 0/207) | 9/9 | 9/9 | parity_ok | parity_ok |
| `tokamak_1species_FPCollisions_noEr` | 160.856 | 2.090 | 0.01x | 2.634 | 0.02x | 93.2 | 297.6 | 3.19x | 829.3 | 8.90x | 0/188 (strict 0/188) | 0/188 (strict 0/188) | 9/9 | 9/9 | parity_ok | parity_ok |
| `tokamak_1species_FPCollisions_noEr_withPhi1InDKE` | 259.575 | 2.204 | 0.01x | 3.844 | 0.01x | 89.6 | 444.2 | 4.96x | 904.9 | 10.10x | 0/274 (strict 0/274) | 0/274 (strict 0/274) | 9/9 | 9/9 | parity_ok | parity_ok |
| `tokamak_1species_FPCollisions_noEr_withQN` | 237.879 | 1.686 | 0.01x | 3.189 | 0.01x | 102.6 | 409.1 | 3.99x | 889.5 | 8.67x | 0/274 (strict 0/274) | 0/274 (strict 0/274) | 9/9 | 9/9 | parity_ok | parity_ok |
| `tokamak_1species_FPCollisions_withEr_DKESTrajectories` | 155.955 | 1.662 | 0.01x | 2.981 | 0.02x | 103.1 | 414.3 | 4.02x | 877.3 | 8.51x | 0/214 (strict 0/214) | 0/214 (strict 0/214) | 9/9 | 9/9 | parity_ok | parity_ok |
| `tokamak_1species_FPCollisions_withEr_fullTrajectories` | 154.953 | 1.774 | 0.01x | 3.088 | 0.02x | 101.1 | 421.0 | 4.16x | 884.6 | 8.75x | 0/214 (strict 0/214) | 0/214 (strict 0/214) | 9/9 | 9/9 | parity_ok | parity_ok |
| `tokamak_1species_PASCollisions_noEr` | 0.309 | 2.327 | 7.53x | 5.154 | 16.68x | 114.2 | 568.4 | 4.98x | 953.6 | 8.35x | 0/212 (strict 0/212) | 0/212 (strict 0/212) | 9/9 | 9/9 | parity_ok | parity_ok |
| `tokamak_1species_PASCollisions_noEr_Nx1` | 0.017 | 2.377 | 139.80x | 3.492 | 205.43x | 100.9 | 484.3 | 4.80x | 895.9 | 8.88x | 0/212 (strict 0/212) | 0/212 (strict 0/212) | 9/9 | 9/9 | parity_ok | parity_ok |
| `tokamak_1species_PASCollisions_noEr_withQN` | 0.888 | 2.051 | 2.31x | 3.440 | 3.87x | 120.9 | 493.7 | 4.08x | 953.8 | 7.89x | 0/274 (strict 0/274) | 0/274 (strict 0/274) | 9/9 | 9/9 | parity_ok | parity_ok |
| `tokamak_1species_PASCollisions_withEr_fullTrajectories` | 0.017 | 2.895 | 170.29x | 18.300 | 1076.48x | 102.0 | 583.3 | 5.72x | 987.0 | 9.67x | 0/212 (strict 0/212) | 0/212 (strict 0/212) | 9/9 | 9/9 | parity_ok | parity_ok |
| `tokamak_2species_PASCollisions_noEr` | 0.331 | 4.188 | 12.65x | 9.479 | 28.64x | 123.6 | 2003.6 | 16.21x | 1626.9 | 13.16x | 0/212 (strict 0/212) | 0/212 (strict 0/212) | 9/9 | 9/9 | parity_ok | parity_ok |
| `tokamak_2species_PASCollisions_withEr_fullTrajectories` | 1.330 | 3.361 | 2.53x | 6.917 | 5.20x | 121.8 | 1588.4 | 13.04x | 1216.0 | 9.98x | 0/212 (strict 0/212) | 0/212 (strict 0/212) | 9/9 | 9/9 | parity_ok | parity_ok |
| `transportMatrix_geometryScheme11` | 0.025 | 1.667 | 66.66x | 3.188 | 127.54x | 102.6 | 403.4 | 3.93x | 882.8 | 8.60x | 0/194 (strict 0/194) | 0/194 (strict 0/194) | 9/9 | 9/9 | parity_ok | parity_ok |
| `transportMatrix_geometryScheme2` | 0.031 | 1.600 | 51.62x | 2.987 | 96.36x | 100.5 | 404.8 | 4.03x | 881.0 | 8.77x | 0/194 (strict 0/194) | 0/194 (strict 0/194) | 9/9 | 9/9 | parity_ok | parity_ok |
<!-- END FAST_BRANCH_AUDIT -->

## Documentation

Build docs locally:

```bash
sphinx-build -b html -W docs docs/_build/html
```

Entry points:

- `docs/index.rst`
- `docs/system_equations.rst`
- `docs/method.rst`
- `docs/normalizations.rst`
- `docs/performance.rst`
- `docs/parallelism.rst`

## Testing

```bash
pytest -q
```

## License

See `LICENSE`.
