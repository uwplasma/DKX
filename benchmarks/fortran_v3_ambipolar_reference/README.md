# Fortran v3 Ambipolar Reference Decks

This directory contains small and production-resolution SFINCS Fortran v3
namelists used to pin the ambipolar-solver functionality that `sfincs_jax`
will reproduce.

The reference executable is expected at:

```bash
/Users/rogeriojorge/local/sfincs/fortran/version3/sfincs
```

The runner copies a selected namelist into a scratch directory as
`input.namelist`, executes Fortran v3 there, and writes compact profiling
summaries. It intentionally does not commit `sfincsOutput.h5`, PETSc binary
dumps, or verbose MUMPS logs.

```bash
python benchmarks/fortran_v3_ambipolar_reference/run_fortran_v3_ambipolar.py \
  --tier small \
  --scratch /tmp/sfincs_v3_ambipolar_reference \
  --summary-json /tmp/sfincs_v3_ambipolar_reference/summary.json
```

Checked-in compact summaries:

- `small_probe_summary_2026-06-22.json`: option 1, 2, and 3 small W7-X-like
  probes plus a distinct geometry-1 helical Brent probe.
- `production_probe_summary_2026-06-22.json`: a larger W7-X-like Brent probe
  with `Ntheta=13`, `Nzeta=19`, `Nxi=48`, and `Nx=5`.

Some local Fortran v3 runs print successful physical diagnostics and then exit
with code `143` from MPI finalization. Treat the JSON success markers and output
diagnostics as the reference signal; keep the return code as a backend
finalization diagnostic.

Important source-code detail: Fortran v3 documentation says the adjoint-backed
ambipolar Newton paths require `magneticDriftScheme > 0`, but
`validateInput.F90` enforces `magneticDriftScheme == 0` for
`ambipolarSolve=.true.` with `ambipolarSolveOption != 2`. These decks follow
the source code, not the stale manual sentence.
