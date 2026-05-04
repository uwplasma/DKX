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
for public runtime/memory claims and research-scale workloads; those larger
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
matrix-free Krylov path when a measured policy is safer or faster. Audited CPU
3D full-FP systems auto-select sparse-PC GMRES inside a measured size window
when it is faster and lower-memory than dense FP. Bounded CPU tokamak
electric-field systems auto-select dense LU only in the validated active-size
window where it avoids the slow Krylov/strong/sparse-rescue ladder; this is a
runtime win with a higher transient memory footprint, so it is CPU-only and
size-capped. Large constrained-PAS profile-current decks also auto-select
sparse-PC GMRES when the problem size is in the validated production window.
Explicit sparse-host LU remains available:

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
file. Sparse-PC runs also write setup/solve/factorization timings and
sparse-pattern counters such as `linearSolverMatvecs`,
`linearSolverSetupTime`, `linearSolverSolveTime`,
`linearSolverSparsePCFactorTime`, and `linearSolverSparsePatternNnz`.
The production benchmark manifest now enforces large research-scale floors:
`35 x 43 x 17 x 48` (`Ntheta x Nzeta x Nx x Nxi`) for 3D cases and
`42 x 1 x 16 x 62` for tokamak cases. Public production timing rows target
SFINCS Fortran v3 runtimes of at least `10 s`; earlier `17 x 21 x 5 x 12`
finite-beta/profile-current timings were lower-resolution bring-up checks for
this solver lane, not public production baselines.

## Runtime and Memory Summary

![Runtime and memory comparison against SFINCS Fortran v3](docs/_static/figures/paper/sfincs_jax_fortran_suite_benchmark_summary.png)

The release benchmark above compares SFINCS Fortran v3, `sfincs_jax` CPU cold,
`sfincs_jax` CPU warm, `sfincs_jax` GPU cold, and `sfincs_jax` GPU warm only for
production-scale rows whose Fortran v3 reference runtime is at least `10 s`.
The left panel shows wall-clock runtime and the right panel shows active solver
memory, both on log axes. Fortran memory is process maximum RSS; JAX memory uses
profiler RSS deltas over the fixed Python/JAX/XLA runtime baseline, while the
full process peak remains in the JSON reports as `jax_max_rss_mb`. Cold is the
first external suite command. Warm runtime
uses `jax_runtime_s_warm` when reports were generated with `--jax-repeats >= 2`;
for the current frozen release reports, it falls back to the CLI
`jax_logged_elapsed_s` field. Cases are ordered by best warm `sfincs_jax` speedup
over the Fortran v3 runtime, so the strongest JAX wins appear first. Regenerate
the plot with
`python examples/publication_figures/generate_fortran_suite_benchmark_summary.py`.

Scope note: the full 39-case frozen suite remains the parity and CI smoke audit,
but sub-`10 s` Fortran rows are not shown as public performance comparisons. They
must either be rerun from the higher-resolution benchmark tier in
`benchmarks/production_resolution_inputs_2026-04-30` or remain CI/regression
checks only. The production tier enforces `35 x 43 x 17 x 48` 3D grids and
`42 x 1 x 16 x 62` tokamak grids, including public examples and optional
user-supplied production-resolution workloads. The manual GitHub workflow
`Production Benchmark Inputs` validates and uploads the production input tree
without running expensive solves; full CPU/GPU/Fortran runtime and memory sweeps
should be launched on local, `office`, or cluster hardware with explicit
resource budgets.

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

Production-resolution inputs are generated separately with
`scripts/create_production_benchmark_inputs.py`. When
`scripts/run_scaled_example_suite.py` is pointed at one of those generated
`inputs/` trees, it detects the sibling `manifest.json` and launches only
`bounded_local_ok` rows by default. Use `--max-run-recommendation bounded_remote`,
`--max-run-recommendation remote_or_cluster_only`, or
`--max-run-recommendation all` only on explicitly budgeted remote or cluster
lanes.

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

Runtime columns match the summary plot: cold is `jax_runtime_s`; warm/logged is `jax_runtime_s_warm` when available, otherwise `jax_logged_elapsed_s`. The JAX memory columns match the plot and use profiler active RSS deltas (`jax_incremental_max_rss_mb`) when present; full process peak RSS remains available as `jax_max_rss_mb` in the frozen JSON reports.
README-facing runtime/memory rows are restricted to cases where the SFINCS Fortran v3 reference runtime is at least `10 s`. Excluded lower-resolution CI parity/smoke rows: `HSX_PASCollisions_DKESTrajectories` (0.994s), `HSX_PASCollisions_fullTrajectories` (2.510s), `geometryScheme4_1species_PAS_withEr_DKESTrajectories` (1.365s), `geometryScheme4_2species_PAS_noEr` (0.953s), `monoenergetic_geometryScheme1` (0.795s), `monoenergetic_geometryScheme11` (0.861s), `monoenergetic_geometryScheme5_ASCII` (1.052s), `monoenergetic_geometryScheme5_netCDF` (1.029s), `sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_DKESTrajectories` (1.104s), `sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories` (1.706s), `tokamak_1species_PASCollisions_noEr` (0.309s), `tokamak_1species_PASCollisions_noEr_Nx1` (0.017s), `tokamak_1species_PASCollisions_noEr_withQN` (0.888s), `tokamak_1species_PASCollisions_withEr_fullTrajectories` (0.017s), `tokamak_2species_PASCollisions_noEr` (0.331s), `tokamak_2species_PASCollisions_withEr_fullTrajectories` (1.330s), `transportMatrix_geometryScheme11` (0.025s), `transportMatrix_geometryScheme2` (0.031s).

Full per-case runtime / memory table:
| Case | Fortran CPU(s) | JAX CPU cold(s) | CPU cold x | JAX CPU warm/logged(s) | CPU warm/logged x | JAX GPU cold(s) | GPU cold x | JAX GPU warm/logged(s) | GPU warm/logged x | Fortran MB | JAX CPU active MB | CPU MB x | JAX GPU active MB | GPU MB x | CPU mismatch | GPU mismatch | CPU print | GPU print | CPU status | GPU status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- |
| `HSX_FPCollisions_DKESTrajectories` | 29.664 | 3.060 | 0.10x | 2.438 | 0.08x | 5.298 | 0.18x | 4.514 | 0.15x | 103.0 | 303.6 | 2.95x | 370.1 | 3.59x | 0/193 (strict 0/193) | 0/193 (strict 0/193) | 9/9 | 9/9 | parity_ok | parity_ok |
| `HSX_FPCollisions_fullTrajectories` | 88.504 | 3.054 | 0.03x | 2.414 | 0.03x | 5.247 | 0.06x | 4.493 | 0.05x | 100.8 | 314.1 | 3.12x | 374.9 | 3.72x | 0/193 (strict 0/193) | 0/193 (strict 0/193) | 9/9 | 9/9 | parity_ok | parity_ok |
| `additional_examples` | 120.074 | 1.733 | 0.01x | 1.063 | 0.01x | 2.633 | 0.02x | 1.898 | 0.02x | 102.1 | 237.2 | 2.32x | 336.1 | 3.29x | 0/193 (strict 0/193) | 0/193 (strict 0/193) | 9/9 | 9/9 | parity_ok | parity_ok |
| `filteredW7XNetCDF_2species_magneticDrifts_noEr` | 89.052 | 2.069 | 0.02x | 1.417 | 0.02x | 2.834 | 0.03x | 2.115 | 0.02x | 103.2 | 288.3 | 2.79x | 349.3 | 3.38x | 0/193 (strict 0/193) | 0/193 (strict 0/193) | 9/9 | 9/9 | parity_ok | parity_ok |
| `filteredW7XNetCDF_2species_magneticDrifts_withEr` | 95.440 | 2.011 | 0.02x | 1.372 | 0.01x | 3.339 | 0.03x | 2.590 | 0.03x | 96.2 | 318.7 | 3.31x | 355.4 | 3.69x | 0/193 (strict 0/193) | 0/193 (strict 0/193) | 9/9 | 9/9 | parity_ok | parity_ok |
| `filteredW7XNetCDF_2species_noEr` | 128.508 | 1.550 | 0.01x | 0.978 | 0.01x | 2.734 | 0.02x | 1.964 | 0.02x | 100.3 | 266.9 | 2.66x | 343.8 | 3.43x | 0/193 (strict 0/193) | 0/193 (strict 0/193) | 9/9 | 9/9 | parity_ok | parity_ok |
| `geometryScheme4_2species_noEr` | 139.240 | 1.733 | 0.01x | 1.112 | 0.01x | 2.888 | 0.02x | 2.105 | 0.02x | 92.2 | 276.8 | 3.00x | 364.2 | 3.95x | 0/207 (strict 0/207) | 0/207 (strict 0/207) | 9/9 | 9/9 | parity_ok | parity_ok |
| `geometryScheme4_2species_noEr_withPhi1InDKE` | 293.275 | 1.936 | 0.01x | 1.353 | 0.00x | 3.340 | 0.01x | 2.596 | 0.01x | 100.6 | 288.6 | 2.87x | 394.0 | 3.91x | 0/265 (strict 0/265) | 0/265 (strict 0/265) | 9/9 | 9/9 | parity_ok | parity_ok |
| `geometryScheme4_2species_noEr_withQN` | 146.734 | 1.769 | 0.01x | 1.140 | 0.01x | 3.132 | 0.02x | 2.402 | 0.02x | 95.1 | 276.2 | 2.91x | 381.2 | 4.01x | 0/265 (strict 0/265) | 0/265 (strict 0/265) | 9/9 | 9/9 | parity_ok | parity_ok |
| `geometryScheme4_2species_withEr_fullTrajectories` | 58.053 | 1.710 | 0.03x | 1.144 | 0.02x | 3.032 | 0.05x | 2.258 | 0.04x | 113.4 | 284.6 | 2.51x | 359.1 | 3.17x | 0/193 (strict 0/193) | 0/193 (strict 0/193) | 9/9 | 9/9 | parity_ok | parity_ok |
| `geometryScheme4_2species_withEr_fullTrajectories_withQN` | 211.358 | 1.889 | 0.01x | 1.310 | 0.01x | 3.087 | 0.01x | 2.314 | 0.01x | 98.8 | 295.1 | 2.99x | 384.2 | 3.89x | 0/251 (strict 0/251) | 0/251 (strict 0/251) | 9/9 | 9/9 | parity_ok | parity_ok |
| `geometryScheme5_3species_loRes` | 98.976 | 1.615 | 0.02x | 1.074 | 0.01x | 3.691 | 0.04x | 2.908 | 0.03x | 129.6 | 352.8 | 2.72x | 363.3 | 2.80x | 0/193 (strict 0/193) | 0/193 (strict 0/193) | 9/9 | 9/9 | parity_ok | parity_ok |
| `inductiveE_noEr` | 166.614 | 1.644 | 0.01x | 1.039 | 0.01x | 2.785 | 0.02x | 1.992 | 0.01x | 99.2 | 279.7 | 2.82x | 364.8 | 3.68x | 0/207 (strict 0/207) | 0/207 (strict 0/207) | 9/9 | 9/9 | parity_ok | parity_ok |
| `quick_2species_FPCollisions_noEr` | 166.945 | 1.531 | 0.01x | 0.983 | 0.01x | 2.938 | 0.02x | 2.200 | 0.01x | 97.1 | 269.0 | 2.77x | 363.7 | 3.74x | 0/207 (strict 0/207) | 0/207 (strict 0/207) | 9/9 | 9/9 | parity_ok | parity_ok |
| `sfincsPaperFigure3_geometryScheme11_FPCollisions_2Species_DKESTrajectories` | 76.666 | 1.653 | 0.02x | 1.110 | 0.01x | 3.188 | 0.04x | 2.391 | 0.03x | 106.7 | 294.2 | 2.76x | 367.6 | 3.44x | 0/207 (strict 0/207) | 0/207 (strict 0/207) | 9/9 | 9/9 | parity_ok | parity_ok |
| `sfincsPaperFigure3_geometryScheme11_FPCollisions_2Species_fullTrajectories` | 93.439 | 1.767 | 0.02x | 1.177 | 0.01x | 3.138 | 0.03x | 2.363 | 0.03x | 94.0 | 303.6 | 3.23x | 372.2 | 3.96x | 0/207 (strict 0/207) | 0/207 (strict 0/207) | 9/9 | 9/9 | parity_ok | parity_ok |
| `tokamak_1species_FPCollisions_noEr` | 160.856 | 1.395 | 0.01x | 0.841 | 0.01x | 2.534 | 0.02x | 1.794 | 0.01x | 93.2 | 185.6 | 1.99x | 311.9 | 3.35x | 0/188 (strict 0/188) | 0/188 (strict 0/188) | 9/9 | 9/9 | parity_ok | parity_ok |
| `tokamak_1species_FPCollisions_noEr_withPhi1InDKE` | 259.575 | 1.783 | 0.01x | 1.217 | 0.00x | 3.592 | 0.01x | 2.783 | 0.01x | 89.6 | 258.3 | 2.88x | 382.8 | 4.27x | 0/274 (strict 0/274) | 0/274 (strict 0/274) | 9/9 | 9/9 | parity_ok | parity_ok |
| `tokamak_1species_FPCollisions_noEr_withQN` | 237.879 | 1.508 | 0.01x | 0.975 | 0.00x | 3.185 | 0.01x | 2.363 | 0.01x | 102.6 | 231.6 | 2.26x | 370.5 | 3.61x | 0/274 (strict 0/274) | 0/274 (strict 0/274) | 9/9 | 9/9 | parity_ok | parity_ok |
| `tokamak_1species_FPCollisions_withEr_DKESTrajectories` | 155.955 | 1.510 | 0.01x | 0.970 | 0.01x | 2.886 | 0.02x | 2.107 | 0.01x | 103.1 | 242.6 | 2.35x | 357.7 | 3.47x | 0/214 (strict 0/214) | 0/214 (strict 0/214) | 9/9 | 9/9 | parity_ok | parity_ok |
| `tokamak_1species_FPCollisions_withEr_fullTrajectories` | 154.953 | 1.623 | 0.01x | 1.077 | 0.01x | 3.038 | 0.02x | 2.261 | 0.01x | 101.1 | 247.5 | 2.45x | 362.9 | 3.59x | 0/214 (strict 0/214) | 0/214 (strict 0/214) | 9/9 | 9/9 | parity_ok | parity_ok |

Largest CPU runtime improvements vs `tests/scaled_example_suite_recheck_cpu_frozen_2026-04-23_postkeyfix/suite_report.json`:
- `tokamak_1species_FPCollisions_noEr_withPhi1InDKE`: 2.4s -> 1.8s (delta=0.6s)
- `quick_2species_FPCollisions_noEr`: 2.1s -> 1.5s (delta=0.6s)
- `sfincsPaperFigure3_geometryScheme11_FPCollisions_2Species_fullTrajectories`: 2.3s -> 1.8s (delta=0.5s)
- `sfincsPaperFigure3_geometryScheme11_FPCollisions_2Species_DKESTrajectories`: 2.2s -> 1.7s (delta=0.5s)
- `inductiveE_noEr`: 2.1s -> 1.6s (delta=0.5s)

Largest CPU process peak-RSS improvements vs `tests/scaled_example_suite_recheck_cpu_frozen_2026-04-23_postkeyfix/suite_report.json`:
- `geometryScheme5_3species_loRes`: 569.4 MB -> 540.3 MB (delta=29.1 MB)
- `geometryScheme4_2species_withEr_fullTrajectories_withQN`: 512.4 MB -> 486.4 MB (delta=26.0 MB)
- `geometryScheme4_2species_noEr_withPhi1InDKE`: 506.0 MB -> 480.9 MB (delta=25.1 MB)
- `filteredW7XNetCDF_2species_magneticDrifts_withEr`: 536.3 MB -> 512.0 MB (delta=24.3 MB)
- `tokamak_1species_FPCollisions_noEr_withPhi1InDKE`: 474.3 MB -> 450.1 MB (delta=24.2 MB)
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
