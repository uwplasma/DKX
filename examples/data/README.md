# Example Data

This folder contains small input files shared by public examples. These files
are intentionally lightweight so a user can clone the repository and run the
first-pass tutorials without downloading external equilibria or benchmark
artifacts.

## Files

- `geometryScheme4_quick_2species.input.namelist`: compact two-species input
  used by tutorial and getting-started workflows that need a fast, deterministic
  DKX run.

Large VMEC, Boozer, profiler, and benchmark artifacts should not be added here.
Use the release-data fetcher in `dkx.validation.data_fetch` or point an
example at a user-provided file instead.
