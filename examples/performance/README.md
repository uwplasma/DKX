# Performance Examples

This folder contains benchmark drivers for JIT compilation, memory behavior,
output formats, CPU parallelism, GPU execution, and transport-worker scaling.
Start with the small format and JIT benchmarks, then move to the sharding or
production-floor scripts when you need release evidence.

## Where To Start

- `benchmark_output_formats.py`: compare HDF5, NetCDF, and NPZ output write/read
  costs on a small reproducible case.
- `benchmark_jit_matvec.py`: measure cold and warm JIT matvec timing.
- `benchmark_transport_l11_vs_fortran.py`: reproduce the compact L11
  parity/runtime figure used in the README and docs.
- `profile_reduced_examples.py`: profile the reduced upstream suite and write
  runtime/memory summaries.

## Scaling And Parallelism

- `benchmark_transport_parallel_scaling.py`: benchmark transport-worker scaling
  on CPU or GPU (`--backend cpu|gpu`). The multi-GPU transport-worker path uses
  `--backend gpu` on `transport_parallel_2min.input.namelist`.
- `benchmark_sharded_solve_scaling.py`: benchmark single-case CPU/GPU sharded
  RHSMode=1 solves. For hot-solve scaling checks, use
  `--inner-warmup-solves 1 --sample-timeout-s 300 --rhs1-precond theta_schwarz
  --schwarz-coarse-levels 2`; add `--deterministic-output-probe` to record a
  bounded 1-vs-N residual/digest gate.
- `benchmark_multi_gpu_case_throughput.py`: benchmark one-GPU-per-case
  throughput by comparing sequential one-GPU execution with concurrent runs on
  a two-GPU node.

## Solver And Ecosystem Benchmarks

- `benchmark_optional_lineax_implicit_solve.py`: optional Lineax gate for
  differentiable linear solves. It compares against the in-tree
  `custom_linear_solve` path on a synthetic nonsymmetric system, a tiny SFINCS
  implicit-diff operator, and a repeated-RHS reuse case, and it skips cleanly
  when `lineax` is not installed.
- `benchmark_structured_solve.py`: bounded factor-once/repeated-RHS
  block-tridiagonal benchmark. It supports deterministic synthetic systems and
  `--case sfincs-pas-block`, which extracts a local PAS block from a SFINCS
  fixture.
- `profile_transport_compile_runtime_cache.py`: profile transport-solve
  compile/runtime split with the persistent JAX cache.

## Generated Output Data

Benchmark commands default to `output/` paths so repeated local runs do not
clutter the repository. The directory is generated, ignored, and not a source
of public validation evidence. Checked performance evidence belongs in compact
fixtures, `docs/_static/`, or release assets that are referenced by docs and
tests.
