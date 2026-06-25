# SFINCS Fortran v3 RHSMode 4/5 sensitivity references

This directory stores lightweight reference artifacts for Fortran-v3 adjoint
sensitivity behavior.  It intentionally does not check in `sfincsOutput.h5`
files; summaries pin the fields and values needed by tests while keeping the
repository small.

The first checked case is a tiny `geometryScheme=4` W7-X-like RHSMode=4 radial
current sensitivity deck.  It was generated from the small ambipolar reference
deck, with `ambipolarSolve=.false.` and `adjointRadialCurrentOption=.true.`.

