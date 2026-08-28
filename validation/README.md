# DKX 3 planning evidence

These machine-readable records freeze the DKX 2.3.1 starting point for the
DKX 3 roadmap. They are planning and release evidence, not runtime package
data. Large outputs, profiler traces, and external-code builds stay outside
Git; later records refer to them by checksum or stable artifact identifier.

- `capabilities.toml`: initial capability status and evidence gaps.
- `baseline.toml`: repository, package, API, CI, and evidence baseline.
- `hardware.toml`: named measurement hosts and availability.
- `benchmark_schema.toml`: fields required of comparable benchmark rows.

The controlling definitions and acceptance gates are in `../plan.md`.
