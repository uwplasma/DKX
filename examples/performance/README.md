# Performance Examples

This folder contains stable benchmark drivers for JIT compilation, memory
behavior, output formats, CPU parallelism, GPU execution, and transport-worker
scaling. Start with the small format and JIT benchmarks, then move to the
transport-worker or production-floor scripts when you need release evidence.

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

Single-case sharded RHSMode=1 and one-GPU-per-case throughput campaign drivers
are research-lane material, not stable examples. They are preserved outside the
stable core until they have production-grid accuracy, runtime, memory, and
documentation gates.

## Solver Benchmarks

- `benchmark_structured_solve.py`: bounded factor-once/repeated-RHS
  block-tridiagonal benchmark. It supports deterministic synthetic systems and
  `--case sfincs-pas-block`, which extracts a local PAS block from a SFINCS
  fixture.
- `profile_transport_compile_runtime_cache.py`: profile transport-solve
  compile/runtime split with the persistent JAX cache.

Optional solver-library adoption studies are research-lane material until they
are promoted through accuracy, runtime, memory, differentiability, and
dependency-policy gates.

## Generated Output Data

Benchmark commands default to `output/` paths so repeated local runs do not
clutter the repository. The directory is generated, ignored, and not a source
of public validation evidence. Checked performance evidence belongs in compact
fixtures, `docs/_static/`, or release assets that are referenced by docs and
tests.
