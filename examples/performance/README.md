# Performance Examples

This folder contains stable benchmark drivers for output formats, CPU/GPU
transport solves, structured solver kernels, and transport-worker scaling.
Start with the small format benchmark, then move to the transport-worker or
production-floor scripts when you need release evidence.

## Where To Start

- `benchmark_output_formats.py`: compare HDF5, NetCDF, and NPZ output write/read
  costs on a small reproducible case.
- `benchmark_transport_l11_vs_fortran.py`: reproduce the compact L11
  parity/runtime figure used in the README and docs.

## Scaling And Parallelism

Solver thread count is configured through the `DKX_CORES` environment knob
(pins the XLA host threadpool; measured optimum 4-8 threads) documented in
`docs/parallelism.rst`; the retired legacy transport-worker and
structured-solve benchmark scripts were deleted with the legacy pipeline.

Single-case sharded RHSMode=1 and one-GPU-per-case throughput campaign drivers
are research-lane material, not stable examples. They are preserved outside the
stable core until they have production-grid accuracy, runtime, memory, and
documentation gates.

## Generated Output Data

Benchmark commands default to `output/` paths so repeated local runs do not
clutter the repository. The directory is generated, ignored, and not a source
of public validation evidence. Checked performance evidence belongs in compact
fixtures, `docs/_static/`, or release assets that are referenced by docs and
tests.
