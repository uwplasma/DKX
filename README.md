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

On the current `main` branch, the audited reduced example suite runs cleanly on
CPU and GPU. A separate production-resolution benchmark tier is now being used
for public runtime/memory claims and collaborator workloads; those larger
finite-beta/profile-current cases are tracked separately from the fast smoke
suite. The default CLI path is tuned for robust explicit solves and practical
throughput, while the Python API can opt into differentiable solve paths when
gradients matter.

It is designed for:

- high-performance runs on CPU/GPU,
- research and production transport workflows,
- memory-efficient large solves,
- end-to-end differentiable workflows.

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
bundled tiny example. The suffix of `--out` selects the output format:
`.h5`/`.hdf5` for the Fortran-compatible HDF5 file, `.nc`/`.netcdf` for NetCDF4,
and `.npz` for a fast NumPy archive.

```bash
cd sfincs_jax
sfincs_jax write-output \
  --input examples/getting_started/input.namelist \
  --out sfincsOutput.h5 \
  --solver-trace solver_trace.json \
  --geometry-only
sfincs_jax --plot sfincsOutput.h5
```

This is the fast installation smoke test. It writes `sfincsOutput.h5` and then
writes a multi-page PDF diagnostics panel next to it as
`sfincsOutput_summary.pdf`. The optional `--solver-trace` sidecar records the
backend, selected solve lane, timing, and output metadata without changing the
parity-oriented output file. The same command works for NetCDF and NPZ:

```bash
sfincs_jax write-output --input examples/getting_started/input.namelist --out sfincsOutput.nc --geometry-only
sfincs_jax write-output --input examples/getting_started/input.namelist --out sfincsOutput.npz --geometry-only
sfincs_jax --plot sfincsOutput.nc
```

For larger non-differentiable RHSMode=1 production runs, the CLI can leave the
matrix-free Krylov path and use structural sparse host solves. Large
constrained-PAS profile-current decks now auto-select sparse-PC GMRES when the
problem size is in the validated production window; explicit sparse-host LU is
also available:

```bash
sfincs_jax write-output \
  --input /path/to/input.namelist \
  --out sfincsOutput.h5 \
  --solve-method sparse_host_safe \
  --solver-trace solver_trace.json
```

This path keeps the full SFINCS system, builds a conservative sparse pattern,
probes only colored structural nonzeros, and first tries host sparse LU. The
`sparse_host_safe` mode falls back only for constrained PAS systems where sparse
LU exposes a singular/gauge-sensitive branch; that fallback is labeled as
PETSc-compatible minimum-norm output instead of pretending the true residual
target was met. RHSMode=1 outputs include `linearSolverMethod`,
`linearSolverResidualNorm`, `linearSolverResidualTarget`,
`linearSolverResidualTargetRatio`, `linearSolverConverged`,
`linearSolverAccepted`, and `linearSolverAcceptanceCriterion` in the main output
file. The production benchmark manifest now enforces at least `25 x 31 x 11 x 17`
(`Ntheta x Nzeta x Nx x Nxi`) for 3D cases and `25 x 1 x 11 x 17` for tokamak
cases. Earlier `17 x 21 x 5 x 12` NTX timings were lower-resolution bring-up
checks for this solver lane, not public production baselines.

## Runtime and Memory Summary

![Runtime and memory comparison against SFINCS Fortran v3](docs/_static/figures/paper/sfincs_jax_fortran_suite_benchmark_summary.png)

The release benchmark above compares SFINCS Fortran v3, `sfincs_jax` CPU cold,
`sfincs_jax` CPU warm, `sfincs_jax` GPU cold, and `sfincs_jax` GPU warm for every
audited example-suite case. The left panel shows wall-clock runtime and the right
panel shows peak resident memory, both on log axes. Cold is the first external
suite command. Warm runtime uses `jax_runtime_s_warm` when reports were generated
with `--jax-repeats >= 2`; for the current frozen release reports, it falls back
to the CLI `jax_logged_elapsed_s` field. Cases are ordered by best warm
`sfincs_jax` speedup over the Fortran v3 runtime, so the strongest JAX wins appear
first. Regenerate the plot with
`python examples/publication_figures/generate_fortran_suite_benchmark_summary.py`.

Scope note: this figure is the reduced audited example-suite benchmark, intended
for regression tracking and quick CPU/GPU smoke validation. Production
performance claims should use the higher-resolution benchmark tier in
`benchmarks/production_resolution_inputs_2026-04-30`, which enforces
`25 x 31 x 11 x 17` 3D grids and `25 x 1 x 11 x 17` tokamak grids, including
collaborator/NTX workloads. That production tier is intentionally separated
because it has already exposed larger finite-beta/profile-current and RHSMode=3
transport solver blockers that are not visible in the reduced suite.

## Physics in One Page

`sfincs_jax` solves the radially local, steady, linearized drift-kinetic
equation for the non-adiabatic distribution-function perturbation
`f_s1` on a flux surface. In normalized form the solved kinetic balance is

```text
(parallel streaming + mirror force + E x B drift + magnetic drift
 + energy/pitch-angle drifts - linearized collisions) f_s1 = thermodynamic drives.
```

The unknown distribution can be coupled to the flux-surface electrostatic
potential variation `Phi1(theta,zeta)` through quasineutrality when requested.
The output fluxes, flows, transport matrices, and diagnostics are moments of
this solved `f_s1`. The full equations, normalizations, switches, and source-code
mapping are documented in `docs/system_equations.rst`, `docs/physics_models.rst`,
and `docs/method.rst`.

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

By default this writes `/path/to/sfincsOutput_summary.pdf`, a multi-page panel
with geometry, radial profiles, particle/heat/momentum fluxes, NTV, moments, and
transport-matrix diagnostics when those datasets are present. Use
`sfincs_jax plot-output --input-h5 ... --out custom.pdf` to choose a filename.

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

Read a namelist, run `sfincs_jax`, write an output file, and inspect results directly in memory:

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

Set `output_path=Path("sfincsOutput.nc")` for NetCDF4 or
`output_path=Path("sfincsOutput.npz")` for a fast NumPy archive. The calculation
is identical; only the writer changes.

If you need to override the equilibrium file without editing the namelist, pass
``equilibrium_file=...`` or the VMEC-friendly alias ``wout_path=...``:

```python
write_sfincs_jax_output_h5(
    input_namelist=input_namelist,
    output_path=Path("sfincsOutput.h5"),
    wout_path=Path("/path/to/wout.nc"),
)
```

`sfincs_jax write-output` and the scan utilities use the explicit
performance-oriented solve path by default. When calling
`write_sfincs_jax_output_h5(...)` directly, pass `differentiable=False` for the
same fast path or request the implicit/differentiable linear-solve path only when
you need gradients:

```python
write_sfincs_jax_output_h5(
    input_namelist=input_namelist,
    output_path=Path("sfincsOutput.h5"),
    differentiable=False,
)

write_sfincs_jax_output_h5(
    input_namelist=input_namelist,
    output_path=Path("sfincsOutput.h5"),
    solve_method="sparse_host",
    differentiable=False,
)

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
- run a finite-beta `vmec_jax` equilibrium into convergence-gated `sfincs_jax` radial profiles of ambipolar `E_r` and bootstrap current: `python examples/vmec_jax_finite_beta/finite_beta_vmec_to_sfincs.py`
- plot the finite-beta kinetic/angular/root-bracket convergence scan from cached outputs: `python examples/vmec_jax_finite_beta/plot_convergence_scan.py`
- plot an output file: `python examples/getting_started/plot_sfincs_output.py`
- write HDF5/NetCDF/NPZ and plot a PDF panel: `python examples/getting_started/write_and_plot_multiple_formats.py`
- run autodiff examples: `python examples/autodiff/autodiff_gradient_nu_n_residual.py`
- run the optional VMEC/Boozer differentiable geometry handoff: `python examples/autodiff/vmec_jax_to_boozer_sfincs_pipeline.py --wout /path/to/wout.nc`
- benchmark CPU/GPU parallel solves: `python examples/performance/benchmark_sharded_solve_scaling.py --backend cpu --devices 1 2 --inner-warmup-solves 1 --sample-timeout-s 300 ...`

Parallel CLI controls are now first-class:

```bash
# Multi-core CPU host sharding on one node
sfincs_jax --cores 8 --shard-axis auto /path/to/input.namelist

# Parallel transport-matrix RHS solves
sfincs_jax transport-matrix-v3 \
  --input /path/to/input.namelist \
  --transport-workers 4

# High-nu LHD/W7-X campaign pilot on a dual-GPU node
CUDA_VISIBLE_DEVICES=0,1 \
python examples/publication_figures/generate_sfincs_paper_figs.py \
  --case lhd \
  --collision-operators 0 \
  --nuprime-min 17.78279101649707 \
  --nuprime-max 17.78279101649707 \
  --n-points 1 \
  --transport-workers 2 \
  --transport-parallel-backend gpu \
  --transport-sparse-direct-max 30000 \
  --require-residuals \
  --max-transport-residual 1e-6 \
  --max-transport-relative-residual 1e-6 \
  --scan-only

# The current office dual-GPU LHD pilot for that point is residual-clean in
# ~262 s, compared with ~345 s on one GPU and ~569 s on the older implicit path.
# For the first W7-X FP high-nu point, use the bounded one-worker sparse-LU lane
# below: it closes all three RHS residual gates in ~9.7 min on one office GPU
# with sparse-helper factor reuse, compared with ~33.8 min before reuse.

# W7-X FP high-nu residual-clean pilot, intentionally one worker to limit sparse
# LU memory pressure:
CUDA_VISIBLE_DEVICES=0 \
SFINCS_JAX_TRANSPORT_SPARSE_FACTOR_DTYPE=float32 \
python examples/publication_figures/generate_sfincs_paper_figs.py \
  --case w7x \
  --collision-operators 0 \
  --nuprime-min 17.78332923601508 \
  --nuprime-max 17.78332923601508 \
  --n-points 1 \
  --transport-workers 1 \
  --transport-parallel-backend gpu \
  --transport-sparse-direct-max 40000 \
  --transport-maxiter 800 \
  --require-residuals \
  --max-transport-residual 1e-6 \
  --max-transport-relative-residual 1e-6 \
  --scan-only

# To compare candidate preconditioners before widening W7-X high-nu scans,
# isolate single-RHS behavior:
CUDA_VISIBLE_DEVICES=0 \
python examples/performance/benchmark_w7x_high_nu_preconditioners.py \
  --preconditioners auto,fp_tzfft,xmg \
  --which-rhs 2 \
  --sparse-direct-max 40000 \
  --sparse-factor-dtype float32 \
  --maxiter 800 \
  --timeout-s 900
```

![W7-X high-nu sparse-helper factor reuse](docs/_static/figures/paper/sfincs_jax_w7x_high_nu_performance.png)

The W7-X high-nu figure is generated by
`python examples/publication_figures/generate_w7x_high_nu_performance.py`.
The checked run preserves the previous residual-clean transport matrix exactly,
reduces the full one-point wall time from about `2028 s` to `582 s`, and lowers
measured peak RSS from about `19.9 GB` to `15.3 GB`.

```bash
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
  --reference-results-root tests/scaled_example_suite_recheck_cpu_frozen_2026-04-23_postkeyfix \
  --out-root tests/scaled_example_suite_release_cpu_frozen_2026-04-25_v106 \
  --scale-factor 1.0 \
  --runtime-target-basis fortran \
  --fortran-min-runtime-s 0.0 \
  --runtime-adjustment-iters 0 \
  --runtime-baseline-report tests/scaled_example_suite_recheck_cpu_frozen_2026-04-23_postkeyfix/suite_report.json \
  --jax-profile-marks on
python scripts/generate_readme_fast_branch_audit.py \
  --out-root tests/scaled_example_suite_release_cpu_frozen_2026-04-25_v106 \
  --gpu-out-root tests/scaled_example_suite_gpu_bounded_default_2026-04-28
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
Current `main` CPU audit comes from `tests/scaled_example_suite_release_cpu_frozen_2026-04-25_v106`.
Matching frozen-reference GPU audit comes from `tests/scaled_example_suite_gpu_bounded_default_2026-04-28`.

- Recorded cases: `39/39`
- Practical status counts: `parity_ok=39`
- Strict status counts: `parity_ok=39`
- GPU practical status counts: `parity_ok=39`
- GPU strict status counts: `parity_ok=39`
- CPU output-key coverage: `missing_total=0, extra_total=70, audited_cases=39, skipped_cases=0`
- GPU output-key coverage: `missing_total=0, extra_total=70, audited_cases=39, skipped_cases=0`
- CPU runtime drift watchlist vs `tests/scaled_example_suite_recheck_cpu_frozen_2026-04-23_postkeyfix/suite_report.json`: none
- GPU runtime drift watchlist vs `tests/scaled_example_suite_release_gpu_2026-04-25_v106/suite_report.json`: none
- Resolution policy: `reference_first_runtime_window, scale_factor=1.0, runtime_basis=fortran, fortran_min=0.0, fortran_max=None, adjust_iters=0`
- Remaining cases: none
- Additional example: `parity_ok` on CPU and `parity_ok` on GPU

Current mismatches:
- CPU practical mismatches: none
- CPU strict mismatches: none
- GPU practical/strict mismatches: none

Runtime columns match the summary plot: cold is `jax_runtime_s`; warm/logged is `jax_runtime_s_warm` when available, otherwise `jax_logged_elapsed_s`. The JAX memory columns are the recorded peak-RSS values and are used for both cold and warm memory bars in the plot.

Full per-case runtime / memory table:
| Case | Fortran CPU(s) | JAX CPU cold(s) | CPU cold x | JAX CPU warm/logged(s) | CPU warm/logged x | JAX GPU cold(s) | GPU cold x | JAX GPU warm/logged(s) | GPU warm/logged x | Fortran MB | JAX CPU MB | CPU MB x | JAX GPU MB | GPU MB x | CPU mismatch | GPU mismatch | CPU print | GPU print | CPU status | GPU status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- |
| `HSX_FPCollisions_DKESTrajectories` | 29.664 | 3.060 | 0.10x | 2.438 | 0.08x | 5.298 | 0.18x | 4.514 | 0.15x | 103.0 | 496.9 | 4.82x | 918.3 | 8.91x | 0/193 (strict 0/193) | 0/193 (strict 0/193) | 9/9 | 9/9 | parity_ok | parity_ok |
| `HSX_FPCollisions_fullTrajectories` | 88.504 | 3.054 | 0.03x | 2.414 | 0.03x | 5.247 | 0.06x | 4.493 | 0.05x | 100.8 | 506.8 | 5.03x | 923.7 | 9.16x | 0/193 (strict 0/193) | 0/193 (strict 0/193) | 9/9 | 9/9 | parity_ok | parity_ok |
| `HSX_PASCollisions_DKESTrajectories` | 0.994 | 3.951 | 3.97x | 3.253 | 3.27x | 6.867 | 6.91x | 6.007 | 6.04x | 112.0 | 1146.3 | 10.23x | 1184.1 | 10.57x | 0/123 (strict 0/123) | 0/123 (strict 0/123) | 7/7 | 7/7 | parity_ok | parity_ok |
| `HSX_PASCollisions_fullTrajectories` | 2.510 | 3.724 | 1.48x | 3.071 | 1.22x | 8.469 | 3.37x | 7.574 | 3.02x | 179.2 | 1441.6 | 8.04x | 1577.4 | 8.80x | 0/193 (strict 0/193) | 0/193 (strict 0/193) | 9/9 | 9/9 | parity_ok | parity_ok |
| `additional_examples` | 120.074 | 1.733 | 0.01x | 1.063 | 0.01x | 2.633 | 0.02x | 1.898 | 0.02x | 102.1 | 429.2 | 4.20x | 885.1 | 8.67x | 0/193 (strict 0/193) | 0/193 (strict 0/193) | 9/9 | 9/9 | parity_ok | parity_ok |
| `filteredW7XNetCDF_2species_magneticDrifts_noEr` | 89.052 | 2.069 | 0.02x | 1.417 | 0.02x | 2.834 | 0.03x | 2.115 | 0.02x | 103.2 | 481.3 | 4.66x | 898.0 | 8.70x | 0/193 (strict 0/193) | 0/193 (strict 0/193) | 9/9 | 9/9 | parity_ok | parity_ok |
| `filteredW7XNetCDF_2species_magneticDrifts_withEr` | 95.440 | 2.011 | 0.02x | 1.372 | 0.01x | 3.339 | 0.03x | 2.590 | 0.03x | 96.2 | 512.0 | 5.32x | 904.7 | 9.41x | 0/193 (strict 0/193) | 0/193 (strict 0/193) | 9/9 | 9/9 | parity_ok | parity_ok |
| `filteredW7XNetCDF_2species_noEr` | 128.508 | 1.550 | 0.01x | 0.978 | 0.01x | 2.734 | 0.02x | 1.964 | 0.02x | 100.3 | 457.0 | 4.56x | 892.9 | 8.90x | 0/193 (strict 0/193) | 0/193 (strict 0/193) | 9/9 | 9/9 | parity_ok | parity_ok |
| `geometryScheme4_1species_PAS_withEr_DKESTrajectories` | 1.365 | 2.209 | 1.62x | 1.588 | 1.16x | 4.399 | 3.22x | 3.604 | 2.64x | 127.3 | 1080.1 | 8.49x | 1264.3 | 9.93x | 0/207 (strict 0/207) | 0/207 (strict 0/207) | 9/9 | 9/9 | parity_ok | parity_ok |
| `geometryScheme4_2species_PAS_noEr` | 0.953 | 2.622 | 2.75x | 1.954 | 2.05x | 5.658 | 5.94x | 4.780 | 5.02x | 162.7 | 1808.8 | 11.12x | 1816.7 | 11.17x | 0/207 (strict 0/207) | 0/207 (strict 0/207) | 9/9 | 9/9 | parity_ok | parity_ok |
| `geometryScheme4_2species_noEr` | 139.240 | 1.733 | 0.01x | 1.112 | 0.01x | 2.888 | 0.02x | 2.105 | 0.02x | 92.2 | 468.0 | 5.07x | 913.8 | 9.91x | 0/207 (strict 0/207) | 0/207 (strict 0/207) | 9/9 | 9/9 | parity_ok | parity_ok |
| `geometryScheme4_2species_noEr_withPhi1InDKE` | 293.275 | 1.936 | 0.01x | 1.353 | 0.00x | 3.340 | 0.01x | 2.596 | 0.01x | 100.6 | 480.9 | 4.78x | 942.8 | 9.37x | 0/265 (strict 0/265) | 0/265 (strict 0/265) | 9/9 | 9/9 | parity_ok | parity_ok |
| `geometryScheme4_2species_noEr_withQN` | 146.734 | 1.769 | 0.01x | 1.140 | 0.01x | 3.132 | 0.02x | 2.402 | 0.02x | 95.1 | 468.1 | 4.92x | 930.5 | 9.79x | 0/265 (strict 0/265) | 0/265 (strict 0/265) | 9/9 | 9/9 | parity_ok | parity_ok |
| `geometryScheme4_2species_withEr_fullTrajectories` | 58.053 | 1.710 | 0.03x | 1.144 | 0.02x | 3.032 | 0.05x | 2.258 | 0.04x | 113.4 | 475.5 | 4.19x | 907.9 | 8.01x | 0/193 (strict 0/193) | 0/193 (strict 0/193) | 9/9 | 9/9 | parity_ok | parity_ok |
| `geometryScheme4_2species_withEr_fullTrajectories_withQN` | 211.358 | 1.889 | 0.01x | 1.310 | 0.01x | 3.087 | 0.01x | 2.314 | 0.01x | 98.8 | 486.4 | 4.92x | 932.7 | 9.44x | 0/251 (strict 0/251) | 0/251 (strict 0/251) | 9/9 | 9/9 | parity_ok | parity_ok |
| `geometryScheme5_3species_loRes` | 98.976 | 1.615 | 0.02x | 1.074 | 0.01x | 3.691 | 0.04x | 2.908 | 0.03x | 129.6 | 540.3 | 4.17x | 911.6 | 7.04x | 0/193 (strict 0/193) | 0/193 (strict 0/193) | 9/9 | 9/9 | parity_ok | parity_ok |
| `inductiveE_noEr` | 166.614 | 1.644 | 0.01x | 1.039 | 0.01x | 2.785 | 0.02x | 1.992 | 0.01x | 99.2 | 468.4 | 4.72x | 913.7 | 9.21x | 0/207 (strict 0/207) | 0/207 (strict 0/207) | 9/9 | 9/9 | parity_ok | parity_ok |
| `monoenergetic_geometryScheme1` | 0.795 | 1.924 | 2.42x | 1.291 | 1.62x | 3.541 | 4.45x | 2.731 | 3.44x | 110.2 | 704.0 | 6.39x | 981.0 | 8.90x | 0/203 (strict 0/203) | 0/203 (strict 0/203) | 9/9 | 9/9 | parity_ok | parity_ok |
| `monoenergetic_geometryScheme11` | 0.861 | 2.853 | 3.31x | 2.211 | 2.57x | 5.606 | 6.51x | 4.772 | 5.54x | 118.7 | 1205.2 | 10.16x | 1003.7 | 8.46x | 0/210 (strict 0/210) | 0/210 (strict 0/210) | 9/9 | 9/9 | parity_ok | parity_ok |
| `monoenergetic_geometryScheme5_ASCII` | 1.052 | 1.678 | 1.59x | 1.075 | 1.02x | 4.296 | 4.08x | 3.453 | 3.28x | 142.1 | 497.0 | 3.50x | 990.0 | 6.97x | 0/207 (strict 0/207) | 0/207 (strict 0/207) | 9/9 | 9/9 | parity_ok | parity_ok |
| `monoenergetic_geometryScheme5_netCDF` | 1.029 | 1.884 | 1.83x | 1.256 | 1.22x | 4.141 | 4.02x | 3.307 | 3.21x | 131.4 | 533.7 | 4.06x | 987.8 | 7.52x | 0/207 (strict 0/207) | 0/207 (strict 0/207) | 9/9 | 9/9 | parity_ok | parity_ok |
| `quick_2species_FPCollisions_noEr` | 166.945 | 1.531 | 0.01x | 0.983 | 0.01x | 2.938 | 0.02x | 2.200 | 0.01x | 97.1 | 459.3 | 4.73x | 912.9 | 9.40x | 0/207 (strict 0/207) | 0/207 (strict 0/207) | 9/9 | 9/9 | parity_ok | parity_ok |
| `sfincsPaperFigure3_geometryScheme11_FPCollisions_2Species_DKESTrajectories` | 76.666 | 1.653 | 0.02x | 1.110 | 0.01x | 3.188 | 0.04x | 2.391 | 0.03x | 106.7 | 479.6 | 4.49x | 915.8 | 8.58x | 0/207 (strict 0/207) | 0/207 (strict 0/207) | 9/9 | 9/9 | parity_ok | parity_ok |
| `sfincsPaperFigure3_geometryScheme11_FPCollisions_2Species_fullTrajectories` | 93.439 | 1.767 | 0.02x | 1.177 | 0.01x | 3.138 | 0.03x | 2.363 | 0.03x | 94.0 | 491.9 | 5.24x | 920.2 | 9.79x | 0/207 (strict 0/207) | 0/207 (strict 0/207) | 9/9 | 9/9 | parity_ok | parity_ok |
| `sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_DKESTrajectories` | 1.104 | 2.658 | 2.41x | 2.011 | 1.82x | 5.757 | 5.21x | 4.846 | 4.39x | 130.7 | 1458.0 | 11.16x | 1587.2 | 12.14x | 0/207 (strict 0/207) | 0/207 (strict 0/207) | 9/9 | 9/9 | parity_ok | parity_ok |
| `sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories` | 1.706 | 2.768 | 1.62x | 2.115 | 1.24x | 6.413 | 3.76x | 5.519 | 3.24x | 144.6 | 1482.2 | 10.25x | 1608.5 | 11.13x | 0/207 (strict 0/207) | 0/207 (strict 0/207) | 9/9 | 9/9 | parity_ok | parity_ok |
| `tokamak_1species_FPCollisions_noEr` | 160.856 | 1.395 | 0.01x | 0.841 | 0.01x | 2.534 | 0.02x | 1.794 | 0.01x | 93.2 | 372.9 | 4.00x | 860.4 | 9.23x | 0/188 (strict 0/188) | 0/188 (strict 0/188) | 9/9 | 9/9 | parity_ok | parity_ok |
| `tokamak_1species_FPCollisions_noEr_withPhi1InDKE` | 259.575 | 1.783 | 0.01x | 1.217 | 0.00x | 3.592 | 0.01x | 2.783 | 0.01x | 89.6 | 450.1 | 5.02x | 932.1 | 10.41x | 0/274 (strict 0/274) | 0/274 (strict 0/274) | 9/9 | 9/9 | parity_ok | parity_ok |
| `tokamak_1species_FPCollisions_noEr_withQN` | 237.879 | 1.508 | 0.01x | 0.975 | 0.00x | 3.185 | 0.01x | 2.363 | 0.01x | 102.6 | 421.5 | 4.11x | 919.4 | 8.96x | 0/274 (strict 0/274) | 0/274 (strict 0/274) | 9/9 | 9/9 | parity_ok | parity_ok |
| `tokamak_1species_FPCollisions_withEr_DKESTrajectories` | 155.955 | 1.510 | 0.01x | 0.970 | 0.01x | 2.886 | 0.02x | 2.107 | 0.01x | 103.1 | 431.4 | 4.18x | 906.2 | 8.79x | 0/214 (strict 0/214) | 0/214 (strict 0/214) | 9/9 | 9/9 | parity_ok | parity_ok |
| `tokamak_1species_FPCollisions_withEr_fullTrajectories` | 154.953 | 1.623 | 0.01x | 1.077 | 0.01x | 3.038 | 0.02x | 2.261 | 0.01x | 101.1 | 437.6 | 4.33x | 911.4 | 9.02x | 0/214 (strict 0/214) | 0/214 (strict 0/214) | 9/9 | 9/9 | parity_ok | parity_ok |
| `tokamak_1species_PASCollisions_noEr` | 0.309 | 2.098 | 6.79x | 1.520 | 4.92x | 4.951 | 16.02x | 4.103 | 13.28x | 114.2 | 579.0 | 5.07x | 987.0 | 8.64x | 0/212 (strict 0/212) | 0/212 (strict 0/212) | 9/9 | 9/9 | parity_ok | parity_ok |
| `tokamak_1species_PASCollisions_noEr_Nx1` | 0.017 | 1.843 | 108.41x | 1.249 | 73.47x | 3.443 | 202.56x | 2.614 | 153.76x | 100.9 | 507.6 | 5.03x | 929.8 | 9.21x | 0/212 (strict 0/212) | 0/212 (strict 0/212) | 9/9 | 9/9 | parity_ok | parity_ok |
| `tokamak_1species_PASCollisions_noEr_withQN` | 0.888 | 1.977 | 2.23x | 1.324 | 1.49x | 3.339 | 3.76x | 2.511 | 2.83x | 120.9 | 533.7 | 4.42x | 988.6 | 8.18x | 0/274 (strict 0/274) | 0/274 (strict 0/274) | 9/9 | 9/9 | parity_ok | parity_ok |
| `tokamak_1species_PASCollisions_withEr_fullTrajectories` | 0.017 | 2.770 | 162.92x | 2.125 | 125.00x | 3.794 | 223.18x | 3.024 | 177.88x | 102.0 | 609.1 | 5.97x | 923.7 | 9.05x | 0/212 (strict 0/212) | 0/212 (strict 0/212) | 9/9 | 9/9 | parity_ok | parity_ok |
| `tokamak_2species_PASCollisions_noEr` | 0.331 | 2.436 | 7.36x | 1.745 | 5.27x | 4.250 | 12.84x | 3.384 | 10.22x | 123.6 | 863.5 | 6.99x | 1148.3 | 9.29x | 0/212 (strict 0/212) | 0/212 (strict 0/212) | 9/9 | 9/9 | parity_ok | parity_ok |
| `tokamak_2species_PASCollisions_withEr_fullTrajectories` | 1.330 | 3.247 | 2.44x | 2.538 | 1.91x | 7.777 | 5.85x | 6.866 | 5.16x | 121.8 | 1607.5 | 13.19x | 1245.5 | 10.22x | 0/212 (strict 0/212) | 0/212 (strict 0/212) | 9/9 | 9/9 | parity_ok | parity_ok |
| `transportMatrix_geometryScheme11` | 0.025 | 1.582 | 63.28x | 1.059 | 42.36x | 3.489 | 139.57x | 2.752 | 110.08x | 102.6 | 438.8 | 4.28x | 926.1 | 9.02x | 0/194 (strict 0/194) | 0/194 (strict 0/194) | 9/9 | 9/9 | parity_ok | parity_ok |
| `transportMatrix_geometryScheme2` | 0.031 | 1.576 | 50.85x | 0.996 | 32.13x | 3.191 | 102.93x | 2.360 | 76.13x | 100.5 | 436.4 | 4.34x | 924.5 | 9.20x | 0/194 (strict 0/194) | 0/194 (strict 0/194) | 9/9 | 9/9 | parity_ok | parity_ok |

Largest CPU runtime improvements vs `tests/scaled_example_suite_recheck_cpu_frozen_2026-04-23_postkeyfix/suite_report.json`:
- `monoenergetic_geometryScheme5_ASCII`: 3.5s -> 1.7s (delta=1.8s)
- `tokamak_2species_PASCollisions_noEr`: 4.0s -> 2.4s (delta=1.6s)
- `HSX_PASCollisions_fullTrajectories`: 5.3s -> 3.7s (delta=1.6s)
- `HSX_PASCollisions_DKESTrajectories`: 5.5s -> 4.0s (delta=1.5s)
- `sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories`: 3.8s -> 2.8s (delta=1.0s)

Largest CPU memory improvements vs `tests/scaled_example_suite_recheck_cpu_frozen_2026-04-23_postkeyfix/suite_report.json`:
- `monoenergetic_geometryScheme5_ASCII`: 3066.4 MB -> 497.0 MB (delta=2569.4 MB)
- `tokamak_2species_PASCollisions_noEr`: 2088.6 MB -> 863.5 MB (delta=1225.1 MB)
- `HSX_PASCollisions_DKESTrajectories`: 2053.6 MB -> 1146.3 MB (delta=907.3 MB)
- `sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories`: 2298.6 MB -> 1482.2 MB (delta=816.4 MB)
- `monoenergetic_geometryScheme5_netCDF`: 1162.7 MB -> 533.7 MB (delta=629.0 MB)
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
