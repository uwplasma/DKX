# Example Data

This folder contains small input files shared by public examples. These files
are intentionally lightweight so a user can clone the repository and run the
first-pass tutorials without downloading external equilibria or benchmark
artifacts.

## Files

- `geometryScheme4_quick_2species.input.namelist`: compact two-species input
  used by tutorial and getting-started workflows that need a fast, deterministic
  DKX run.
- `input.minimal_seed_nfp2`: VMEX's nfp = 2 seed deck (a copy of
  `vmex/examples/data/input.minimal_seed_nfp2`), shaped by
  `examples/optimization/QA_optimization_bootstrap_dkx.py` and the VMEX-chain
  tests.

Large VMEC, Boozer, profiler, and benchmark artifacts should not be added here.
Use the release-data fetcher in `dkx.validation.data_fetch` or point an
example at a user-provided file instead.
