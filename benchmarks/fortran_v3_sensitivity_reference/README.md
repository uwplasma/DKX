# SFINCS Fortran v3 RHSMode 4/5 sensitivity references

This directory stores lightweight reference artifacts for Fortran-v3 adjoint
sensitivity behavior.  It intentionally does not check in `sfincsOutput.h5`
files; summaries pin the fields and values needed by tests while keeping the
repository small.

The checked cases are tiny `geometryScheme=4` W7-X-like decks generated from
the small ambipolar reference deck:

- `geometry4_w7x_like_small_rhs4_radial_current` pins
  `dParticleFluxdLambda`, `dParallelFlowdLambda`, and
  `dRadialCurrentdLambda`, including the Fortran relation
  `dRadialCurrentdLambda = dParticleFluxdLambda_ion -
  dParticleFluxdLambda_electron`.
- `geometry4_w7x_like_small_rhs4_heat_flux` pins `dHeatFluxdLambda` and
  `dTotalHeatFluxdLambda`, including the Fortran relation
  `dTotalHeatFluxdLambda = sum_s dHeatFluxdLambda_s`.
- `geometry4_w7x_like_small_rhs5_heat_flux` pins the corresponding RHSMode=5
  constant-current heat-flux fields plus `dPhidPsidLambda` after the Fortran
  Brent ambipolar solve.
