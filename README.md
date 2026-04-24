# sfincs_jax

[![CI](https://github.com/uwplasma/sfincs_jax/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/uwplasma/sfincs_jax/actions/workflows/ci.yml)
[![Docs](https://github.com/uwplasma/sfincs_jax/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/uwplasma/sfincs_jax/actions/workflows/docs.yml)
[![PyPI](https://img.shields.io/pypi/v/sfincs_jax)](https://pypi.org/project/sfincs_jax/)
[![Coverage](https://codecov.io/gh/uwplasma/sfincs_jax/branch/main/graph/badge.svg)](https://codecov.io/gh/uwplasma/sfincs_jax)
![Python versions](https://img.shields.io/pypi/pyversions/sfincs_jax)
![License](https://img.shields.io/github/license/uwplasma/sfincs_jax)

`sfincs_jax` is a standalone neoclassical transport code for radially local
drift-kinetic calculations in stellarator and tokamak geometry. It combines
high-fidelity kinetic models, CPU/GPU execution, modern matrix-free numerics,
parallel workflows, and optional differentiable solve paths in one codebase.

On the current `main` branch, the full audited example suite runs cleanly on CPU and GPU.
The default CLI path is tuned for robust production solves and practical throughput,
while the Python API can opt into differentiable solve paths when gradients matter.

It is designed for:

- high-performance runs on CPU/GPU,
- research and production transport workflows,
- memory-efficient large solves,
- end-to-end differentiable workflows.

![Runtime and parity snapshot](docs/_static/figures/sfincs_vs_sfincs_jax_l11_runtime_2x2.png)

The figure above shows a representative transport benchmark. The release-facing
validation and benchmark artifacts are documented in the docs and in the audit table
below.

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

After installing from a source checkout, you can run the CLI immediately on the
bundled tiny example:

```bash
cd sfincs_jax
sfincs_jax write-output \
  --input examples/getting_started/input.namelist \
  --out sfincsOutput.h5 \
  --geometry-only
sfincs_jax --plot sfincsOutput.h5
```

This is the fast installation smoke test. It writes `sfincsOutput.h5` and then
writes a compact PNG summary next to it as `sfincsOutput_summary.png`.

## Quick Start (CLI)

You can run `sfincs_jax` from anywhere in your terminal. You do not need to be
inside the repository folder.

Run an input file:

```bash
sfincs_jax /path/to/input.namelist
```

Write output explicitly:

```bash
sfincs_jax write-output --input /path/to/input.namelist --out /path/to/sfincsOutput.h5
```

Plot an existing output file:

```bash
sfincs_jax --plot /path/to/sfincsOutput.h5
```

Override the equilibrium file at the CLI without changing `input.namelist`:

```bash
sfincs_jax write-output \
  --input /path/to/input.namelist \
  --out /path/to/sfincsOutput.h5 \
  --wout-path /path/to/wout.nc
```

The bare `sfincs_jax /path/to/input.namelist` form accepts the same
`--equilibrium-file` and `--wout-path` overrides.

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

Repository examples that map directly onto common first tasks:

- run the bundled tiny CLI example: `sfincs_jax examples/getting_started/input.namelist`
- write a tiny tokamak output: `python examples/getting_started/write_sfincs_output_tokamak.py`
- write a tiny VMEC output with `wout_path`: `python examples/getting_started/write_sfincs_output_vmec.py`
- plot an output file: `python examples/getting_started/plot_sfincs_output.py`
- run autodiff examples: `python examples/autodiff/autodiff_gradient_nu_n_residual.py`
- benchmark CPU/GPU parallel solves: `python examples/performance/benchmark_sharded_solve_scaling.py ...`

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
- The current sharded RHSMode=1 CPU path uses a wider Schwarz patch rule plus a
  bounded multilevel residual correction to avoid the worst 4/8-device
  fragmentation failures seen in earlier releases.
- Use one GPU per case or scan point for production throughput today.
- Multi-GPU single-case sharding is available for benchmarking and very large
  runs, but it remains experimental and is not yet the default recommendation.
- The sharded-solve benchmark helper supports both `--backend cpu` and
  `--backend gpu`; the GPU path uses `CUDA_VISIBLE_DEVICES` and disables JAX
  preallocation in the subprocess, with `cuda_malloc_async` enabled for the
  benchmark subprocess allocator, so one-node GPU scaling experiments are more
  reproducible.
- For practical multi-GPU usage today, the strongest measured path is
  transport-worker parallelism with one worker per GPU on RHSMode=2/3 runs.
  On the fresh office 2-GPU rerun of
  `examples/performance/transport_parallel_2min.input.namelist`, this path
  measured `351.1s -> 237.7s` from `1 -> 2` GPU workers, i.e. `1.48x` speedup
  on a 3-RHS case, essentially at the finite-task ideal of `1.5x`.
- Multi-GPU single-case sharding remains experimental. Use it for research and
  benchmarking, not as the default production scaling path.

You can reproduce the recommended multi-GPU transport-worker benchmark with:

```bash
python examples/performance/benchmark_transport_parallel_scaling.py \
  --input examples/performance/transport_parallel_2min.input.namelist \
  --backend gpu \
  --workers 1 2
```

![GPU transport scaling](docs/_static/figures/parallel/transport_parallel_scaling_gpu.png)

Compare two outputs:

```bash
sfincs_jax compare-h5 --a sfincsOutput_jax.h5 --b sfincsOutput_fortran.h5
```

Advanced CLI, plotting, and solver options are documented in `docs/usage.rst`,
`docs/outputs.rst`, and `docs/performance_techniques.rst`.

## Models, Numerics, and Validation

`sfincs_jax` solves the same class of neoclassical drift-kinetic problems as mature
SFINCS workflows, but it is documented and maintained as its own code. In particular:

- the public executable favors bounded, performance-oriented solve strategies,
- the Python API can switch to differentiable solve paths when end-to-end sensitivities are needed,
- CPU runs lean on JIT-cached kernels and selected host sparse factorizations for hard linear branches,
- repeated RHSMode=1 output-writing runs reuse prebuilt grids, geometry, and operator state to cut setup cost on large HSX/geometry11 cases,
- GPU runs keep operator applications on device, then fall back to accelerator-safe or host rescue paths only when conditioning or memory demands it,
- and the documentation maps the governing equations directly onto the source tree.

The main documentation entry points are:

- physics and equations: `docs/physics_models.rst`, `docs/system_equations.rst`, `docs/physics_reference.rst`
- geometry and numerics: `docs/geometry.rst`, `docs/method.rst`, `docs/numerics.rst`
- inputs and outputs: `docs/inputs.rst`, `docs/outputs.rst`
- parallel and performance workflows: `docs/parallelism.rst`, `docs/performance.rst`
- examples, applications, and testing: `docs/examples.rst`, `docs/applications.rst`, `docs/testing.rst`
- external trust-building comparisons: `docs/fortran_comparison.rst`

## Current Example-Suite Audit

Regenerate this block from the current `main` working tree with:

```bash
python scripts/run_scaled_example_suite.py \
  --examples-root examples/sfincs_examples \
  --resolution-reference-root /Users/rogeriojorge/local/tests/sfincs_original/fortran/version3/examples \
  --fortran-exe /Users/rogeriojorge/local/tests/sfincs/fortran/version3/sfincs \
  --out-root tests/scaled_example_suite_recheck_cpu_frozen_2026-04-23_postkeyfix \
  --scale-factor 1.0 \
  --runtime-target-basis fortran \
  --fortran-min-runtime-s 0.0 \
  --runtime-adjustment-iters 0 \
  --runtime-baseline-report tests/scaled_example_suite_fast_cpu_full_v7_refresh/suite_report.json
python scripts/generate_readme_fast_branch_audit.py \
  --out-root tests/scaled_example_suite_recheck_cpu_frozen_2026-04-23_postkeyfix \
  --gpu-out-root tests/scaled_example_suite_recheck_gpu_frozen_2026-04-23_postruntimefix_mem
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
Current `main` CPU audit comes from `tests/scaled_example_suite_recheck_cpu_frozen_2026-04-23_postkeyfix`.
Matching frozen-reference GPU audit comes from `tests/scaled_example_suite_recheck_gpu_frozen_2026-04-23_postruntimefix_mem`.
The `HSX_PASCollisions_DKESTrajectories` CPU/GPU row,
`HSX_PASCollisions_fullTrajectories` CPU row, and
`tokamak_1species_PASCollisions_withEr_fullTrajectories` GPU row, and
`geometryScheme4_2species_PAS_noEr` CPU/GPU row include the latest focused
current-tip PAS reruns after the guarded PAS auto-selection updates.

- Recorded cases: `39/39`
- Practical status counts: `parity_ok=39`
- Strict status counts: `parity_ok=39`
- GPU practical status counts: `parity_ok=39`
- GPU strict status counts: `parity_ok=39`
- CPU output-key coverage: `missing_total=0, extra_total=70, audited_cases=39, skipped_cases=0`
- GPU output-key coverage: `missing_total=0, extra_total=70, audited_cases=39, skipped_cases=0`
- CPU runtime drift watchlist vs `tests/scaled_example_suite_fast_cpu_full_v7_refresh/suite_report.json`: none
- GPU runtime drift watchlist vs `tests/scaled_example_suite_fast_gpu_full_v11_refresh/suite_report.json`: none
- Resolution policy: `reference_first_runtime_window, scale_factor=1.0, runtime_basis=fortran, fortran_min=0.0, fortran_max=None, adjust_iters=0`
- Remaining cases: none
- Additional example: `parity_ok` on CPU and `parity_ok` on GPU

Current mismatches:
- CPU practical mismatches: none
- CPU strict mismatches: none
- GPU practical/strict mismatches: none

Full per-case runtime / memory table:
| Case | Fortran CPU(s) | JAX CPU(s) | CPU x | JAX GPU(s) | GPU x | Fortran MB | JAX CPU MB | CPU MB x | JAX GPU MB | GPU MB x | CPU mismatch | GPU mismatch | CPU print | GPU print | CPU status | GPU status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- |
| `HSX_FPCollisions_DKESTrajectories` | 29.664 | 3.280 | 0.11x | 5.099 | 0.17x | 103.0 | 510.3 | 4.95x | 919.8 | 8.93x | 0/193 (strict 0/193) | 0/193 (strict 0/193) | 9/9 | 9/9 | parity_ok | parity_ok |
| `HSX_FPCollisions_fullTrajectories` | 88.504 | 3.439 | 0.04x | 5.201 | 0.06x | 100.8 | 525.7 | 5.21x | 923.3 | 9.16x | 0/193 (strict 0/193) | 0/193 (strict 0/193) | 9/9 | 9/9 | parity_ok | parity_ok |
| `HSX_PASCollisions_DKESTrajectories` | 0.994 | 3.940 | 3.96x | 7.627 | 7.67x | 112.0 | 1019.2 | 9.10x | 1174.9 | 10.49x | 0/123 (strict 0/123) | 0/123 (strict 0/123) | 7/7 | 7/7 | parity_ok | parity_ok |
| `HSX_PASCollisions_fullTrajectories` | 2.510 | 4.027 | 1.60x | 9.082 | 3.62x | 179.2 | 1384.1 | 7.72x | 2042.1 | 11.40x | 0/193 (strict 0/193) | 0/193 (strict 0/193) | 9/9 | 9/9 | parity_ok | parity_ok |
| `additional_examples` | 120.074 | 1.883 | 0.02x | 2.684 | 0.02x | 102.1 | 442.4 | 4.33x | 885.4 | 8.67x | 0/193 (strict 0/193) | 0/193 (strict 0/193) | 9/9 | 9/9 | parity_ok | parity_ok |
| `filteredW7XNetCDF_2species_magneticDrifts_noEr` | 89.052 | 2.198 | 0.02x | 3.034 | 0.03x | 103.2 | 501.3 | 4.86x | 899.3 | 8.71x | 0/193 (strict 0/193) | 0/193 (strict 0/193) | 9/9 | 9/9 | parity_ok | parity_ok |
| `filteredW7XNetCDF_2species_magneticDrifts_withEr` | 95.440 | 2.264 | 0.02x | 3.085 | 0.03x | 96.2 | 536.3 | 5.58x | 904.8 | 9.41x | 0/193 (strict 0/193) | 0/193 (strict 0/193) | 9/9 | 9/9 | parity_ok | parity_ok |
| `filteredW7XNetCDF_2species_noEr` | 128.508 | 1.930 | 0.02x | 2.886 | 0.02x | 100.3 | 471.2 | 4.70x | 893.5 | 8.91x | 0/193 (strict 0/193) | 0/193 (strict 0/193) | 9/9 | 9/9 | parity_ok | parity_ok |
| `geometryScheme4_1species_PAS_withEr_DKESTrajectories` | 1.365 | 2.967 | 2.17x | 4.750 | 3.48x | 127.3 | 1064.0 | 8.36x | 1254.4 | 9.86x | 0/207 (strict 0/207) | 0/207 (strict 0/207) | 9/9 | 9/9 | parity_ok | parity_ok |
| `geometryScheme4_2species_PAS_noEr` | 0.953 | 1.962 | 2.06x | 4.774 | 5.01x | 162.7 | 1728.0 | 10.62x | 1817.0 | 11.17x | 0/207 (strict 0/207) | 0/207 (strict 0/207) | 9/9 | 9/9 | parity_ok | parity_ok |
| `geometryScheme4_2species_noEr` | 139.240 | 1.932 | 0.01x | 2.830 | 0.02x | 92.2 | 483.2 | 5.24x | 912.7 | 9.89x | 0/207 (strict 0/207) | 0/207 (strict 0/207) | 9/9 | 9/9 | parity_ok | parity_ok |
| `geometryScheme4_2species_noEr_withPhi1InDKE` | 293.275 | 2.256 | 0.01x | 3.339 | 0.01x | 100.6 | 506.0 | 5.03x | 943.0 | 9.37x | 0/265 (strict 0/265) | 0/265 (strict 0/265) | 9/9 | 9/9 | parity_ok | parity_ok |
| `geometryScheme4_2species_noEr_withQN` | 146.734 | 1.985 | 0.01x | 3.134 | 0.02x | 95.1 | 482.3 | 5.07x | 931.0 | 9.79x | 0/265 (strict 0/265) | 0/265 (strict 0/265) | 9/9 | 9/9 | parity_ok | parity_ok |
| `geometryScheme4_2species_withEr_fullTrajectories` | 58.053 | 2.032 | 0.04x | 3.084 | 0.05x | 113.4 | 498.4 | 4.40x | 908.4 | 8.01x | 0/193 (strict 0/193) | 0/193 (strict 0/193) | 9/9 | 9/9 | parity_ok | parity_ok |
| `geometryScheme4_2species_withEr_fullTrajectories_withQN` | 211.358 | 2.195 | 0.01x | 3.436 | 0.02x | 98.8 | 512.4 | 5.18x | 932.6 | 9.44x | 0/251 (strict 0/251) | 0/251 (strict 0/251) | 9/9 | 9/9 | parity_ok | parity_ok |
| `geometryScheme5_3species_loRes` | 98.976 | 2.101 | 0.02x | 3.792 | 0.04x | 129.6 | 569.4 | 4.39x | 911.0 | 7.03x | 0/193 (strict 0/193) | 0/193 (strict 0/193) | 9/9 | 9/9 | parity_ok | parity_ok |
| `inductiveE_noEr` | 166.614 | 2.147 | 0.01x | 3.036 | 0.02x | 99.2 | 477.5 | 4.81x | 913.8 | 9.21x | 0/207 (strict 0/207) | 0/207 (strict 0/207) | 9/9 | 9/9 | parity_ok | parity_ok |
| `monoenergetic_geometryScheme1` | 0.795 | 2.046 | 2.57x | 14.571 | 18.33x | 110.2 | 710.5 | 6.45x | 996.7 | 9.04x | 0/203 (strict 0/203) | 0/203 (strict 0/203) | 9/9 | 9/9 | parity_ok | parity_ok |
| `monoenergetic_geometryScheme11` | 0.861 | 3.116 | 3.62x | 5.758 | 6.69x | 118.7 | 1201.2 | 10.12x | 1003.6 | 8.46x | 0/210 (strict 0/210) | 0/210 (strict 0/210) | 9/9 | 9/9 | parity_ok | parity_ok |
| `monoenergetic_geometryScheme5_ASCII` | 1.052 | 3.471 | 3.30x | 4.296 | 4.08x | 142.1 | 3066.4 | 21.58x | 989.0 | 6.96x | 0/207 (strict 0/207) | 0/207 (strict 0/207) | 9/9 | 9/9 | parity_ok | parity_ok |
| `monoenergetic_geometryScheme5_netCDF` | 1.029 | 2.522 | 2.45x | 4.196 | 4.08x | 131.4 | 1162.7 | 8.85x | 988.3 | 7.52x | 0/207 (strict 0/207) | 0/207 (strict 0/207) | 9/9 | 9/9 | parity_ok | parity_ok |
| `quick_2species_FPCollisions_noEr` | 166.945 | 2.097 | 0.01x | 2.784 | 0.02x | 97.1 | 478.7 | 4.93x | 913.3 | 9.40x | 0/207 (strict 0/207) | 0/207 (strict 0/207) | 9/9 | 9/9 | parity_ok | parity_ok |
| `sfincsPaperFigure3_geometryScheme11_FPCollisions_2Species_DKESTrajectories` | 76.666 | 2.164 | 0.03x | 3.137 | 0.04x | 106.7 | 492.9 | 4.62x | 917.2 | 8.59x | 0/207 (strict 0/207) | 0/207 (strict 0/207) | 9/9 | 9/9 | parity_ok | parity_ok |
| `sfincsPaperFigure3_geometryScheme11_FPCollisions_2Species_fullTrajectories` | 93.439 | 2.278 | 0.02x | 3.185 | 0.03x | 94.0 | 513.8 | 5.47x | 922.2 | 9.82x | 0/207 (strict 0/207) | 0/207 (strict 0/207) | 9/9 | 9/9 | parity_ok | parity_ok |
| `sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_DKESTrajectories` | 1.104 | 3.068 | 2.78x | 6.458 | 5.85x | 130.7 | 1477.5 | 11.30x | 1586.9 | 12.14x | 0/207 (strict 0/207) | 0/207 (strict 0/207) | 9/9 | 9/9 | parity_ok | parity_ok |
| `sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories` | 1.706 | 3.794 | 2.22x | 7.619 | 4.47x | 144.6 | 2298.6 | 15.90x | 2097.0 | 14.50x | 0/207 (strict 0/207) | 0/207 (strict 0/207) | 9/9 | 9/9 | parity_ok | parity_ok |
| `tokamak_1species_FPCollisions_noEr` | 160.856 | 1.774 | 0.01x | 2.380 | 0.01x | 93.2 | 380.8 | 4.09x | 859.5 | 9.22x | 0/188 (strict 0/188) | 0/188 (strict 0/188) | 9/9 | 9/9 | parity_ok | parity_ok |
| `tokamak_1species_FPCollisions_noEr_withPhi1InDKE` | 259.575 | 2.422 | 0.01x | 3.340 | 0.01x | 89.6 | 474.3 | 5.29x | 933.1 | 10.42x | 0/274 (strict 0/274) | 0/274 (strict 0/274) | 9/9 | 9/9 | parity_ok | parity_ok |
| `tokamak_1species_FPCollisions_noEr_withQN` | 237.879 | 1.989 | 0.01x | 3.038 | 0.01x | 102.6 | 436.8 | 4.26x | 920.1 | 8.97x | 0/274 (strict 0/274) | 0/274 (strict 0/274) | 9/9 | 9/9 | parity_ok | parity_ok |
| `tokamak_1species_FPCollisions_withEr_DKESTrajectories` | 155.955 | 1.987 | 0.01x | 2.683 | 0.02x | 103.1 | 442.7 | 4.29x | 907.2 | 8.80x | 0/214 (strict 0/214) | 0/214 (strict 0/214) | 9/9 | 9/9 | parity_ok | parity_ok |
| `tokamak_1species_FPCollisions_withEr_fullTrajectories` | 154.953 | 2.099 | 0.01x | 3.038 | 0.02x | 101.1 | 459.1 | 4.54x | 911.8 | 9.02x | 0/214 (strict 0/214) | 0/214 (strict 0/214) | 9/9 | 9/9 | parity_ok | parity_ok |
| `tokamak_1species_PASCollisions_noEr` | 0.309 | 2.658 | 8.60x | 4.899 | 15.86x | 114.2 | 612.9 | 5.36x | 985.8 | 8.63x | 0/212 (strict 0/212) | 0/212 (strict 0/212) | 9/9 | 9/9 | parity_ok | parity_ok |
| `tokamak_1species_PASCollisions_noEr_Nx1` | 0.017 | 2.432 | 143.08x | 3.393 | 199.58x | 100.9 | 520.3 | 5.16x | 928.5 | 9.20x | 0/212 (strict 0/212) | 0/212 (strict 0/212) | 9/9 | 9/9 | parity_ok | parity_ok |
| `tokamak_1species_PASCollisions_noEr_withQN` | 0.888 | 2.316 | 2.61x | 3.439 | 3.87x | 120.9 | 526.1 | 4.35x | 987.1 | 8.17x | 0/274 (strict 0/274) | 0/274 (strict 0/274) | 9/9 | 9/9 | parity_ok | parity_ok |
| `tokamak_1species_PASCollisions_withEr_fullTrajectories` | 0.017 | 3.410 | 200.59x | 3.249 | 191.12x | 102.0 | 628.3 | 6.16x | 922.3 | 9.04x | 0/212 (strict 0/212) | 0/212 (strict 0/212) | 9/9 | 9/9 | parity_ok | parity_ok |
| `tokamak_2species_PASCollisions_noEr` | 0.331 | 4.004 | 12.10x | 4.649 | 14.04x | 123.6 | 2088.6 | 16.90x | 1148.7 | 9.29x | 0/212 (strict 0/212) | 0/212 (strict 0/212) | 9/9 | 9/9 | parity_ok | parity_ok |
| `tokamak_2species_PASCollisions_withEr_fullTrajectories` | 1.330 | 3.611 | 2.72x | 7.722 | 5.81x | 121.8 | 1601.9 | 13.15x | 1245.1 | 10.22x | 0/212 (strict 0/212) | 0/212 (strict 0/212) | 9/9 | 9/9 | parity_ok | parity_ok |
| `transportMatrix_geometryScheme11` | 0.025 | 1.806 | 72.23x | 3.289 | 131.57x | 102.6 | 440.9 | 4.30x | 919.1 | 8.96x | 0/194 (strict 0/194) | 0/194 (strict 0/194) | 9/9 | 9/9 | parity_ok | parity_ok |
| `transportMatrix_geometryScheme2` | 0.031 | 1.880 | 60.64x | 3.389 | 109.31x | 100.5 | 436.8 | 4.35x | 918.1 | 9.14x | 0/194 (strict 0/194) | 0/194 (strict 0/194) | 9/9 | 9/9 | parity_ok | parity_ok |
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
