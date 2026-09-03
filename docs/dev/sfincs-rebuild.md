# Rebuilding SFINCS Fortran v3 for the head-to-head sweep

The sweep in `tools/benchmarks/parity_performance_matrix.py` needs a working
SFINCS binary. This records what it took on macOS/arm64 in September 2026, so
the next rebuild is not a rediscovery.

## Toolchain

    PETSc      3.20.2   /opt/local/lib/petsc   (MUMPS, PARMETIS, SCALAPACK)
    HDF5       1.14                            (libhdf5_fortran.311)
    netcdf-f   local build under ~/local/netcdf-fortran/build
    compilers  macports gfortran 13 / mpif90-mpich-clang16

`SFINCS_SYSTEM=macports`, and `makefiles/makefile.macports` points `PETSC_DIR`
and `NETCDF_DIR` at those.

## Two things that will bite

**Build serially.** `make -j8` dies with `read jobs pipe: Resource temporarily
unavailable` -- the nested `gmake -C mini_libstell` clashes with the parent's
jobserver. Plain `make` works.

**`sfincs.F90` needs a one-line change**, kept here as
`sfincs-local-build-fix.patch`. Under gfortran 13 + MPICH, `sfincs_main`
already pulls in `mpi_base` through PETSc, so a bare `use mpi` makes
`pmpi_wtime` and `pmpi_wtick` ambiguous and the compile fails. Importing only
`MPI_COMM_WORLD`, `MPI_INIT` and `MPI_FINALIZE` fixes it and changes nothing
else. It is a local build fix and is deliberately not upstreamed.

An existing binary can also simply be stale: the one from May 2024 failed at
load with `Library not loaded: @rpath/libhdf5_fortran.310.dylib` because HDF5
had moved to `.311`. Recompiling relinks it.

## One MPI rank only, on this build

SFINCS segfaults at 2 MPI ranks on every deck tried here (4 of 4, no successful
completions), after its matrix pre-assembly inflates about 20x -- 1.13 s to
23.0 s on `tokamak_1species_FPCollisions_noEr`. It still writes an
`sfincsOutput.h5` first: 70 KB against a good 341 KB, carrying classical fluxes
and no neoclassical ones. A parity check will compare against that file rather
than reject it, so verify the reference run *succeeded* and do not just test
that output exists.

This is a property of this toolchain, not a claim about SFINCS: the head-to-head
recorded in `docs/performance.rst` used conda PETSc 3.23 with 2 ranks working.
Run sweeps here with `--ranks 1`.
