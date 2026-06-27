# Example Data

This folder contains small input files shared by public examples. These files
are intentionally lightweight so a user can clone the repository and run the
first-pass tutorials without downloading external equilibria or benchmark
artifacts.

## Files

- `geometryScheme4_quick_2species.input.namelist`: compact two-species input
  used by tutorial and getting-started workflows that need a fast, deterministic
  SFINCS-JAX run.
- `qi_nfp2_reference.input.namelist`: QI/VMEC RHSMode=1 reference input used by
  scaled-suite audits, QI robustness checks, and production-resolution
  benchmark manifests. The benchmark case label remains `additional_examples`
  in historical reports, but the input lives here because it is shared data
  rather than a standalone learning workflow.

Large VMEC, Boozer, profiler, and benchmark artifacts should not be added here.
Use the release-data fetcher in `sfincs_jax.validation.data_fetch` or point an
example at a user-provided file instead.
