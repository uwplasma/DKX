# SFINCS Fortran v3 RHSMode 4/5 sensitivity references

This directory stores lightweight reference artifacts for Fortran-v3 adjoint
sensitivity behavior.  It intentionally does not check in `sfincsOutput.h5`
files; summaries pin the fields and values needed by tests while keeping the
repository small.

The checked cases are tiny `geometryScheme=4` W7-X-like RHSMode=4 decks
generated from the small ambipolar reference deck with `ambipolarSolve=.false.`:

- `geometry4_w7x_like_small_rhs4_radial_current` pins
  `dParticleFluxdLambda`, `dParallelFlowdLambda`, and
  `dRadialCurrentdLambda`, including the Fortran relation
  `dRadialCurrentdLambda = dParticleFluxdLambda_ion -
  dParticleFluxdLambda_electron`.
- `geometry4_w7x_like_small_rhs4_heat_flux` pins `dHeatFluxdLambda` and
  `dTotalHeatFluxdLambda`, including the Fortran relation
  `dTotalHeatFluxdLambda = sum_s dHeatFluxdLambda_s`.
