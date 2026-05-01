# Scaled Example Suite Summary

- Cases: 39
- Practical status counts: parity_ok=39
- Strict status counts: parity_ok=39

## Runtime offenders (absolute JAX time)

- HSX_PASCollisions_DKESTrajectories: jax=4.845s fortran=0.994s ratio=4.874 res={'NTHETA': 5, 'NZETA': 15, 'NX': 2, 'NXI': 20} status=parity_ok
- HSX_PASCollisions_fullTrajectories: jax=4.527s fortran=2.510s ratio=1.804 res={'NTHETA': 6, 'NZETA': 15, 'NX': 3, 'NXI': 20} status=parity_ok
- tokamak_2species_PASCollisions_withEr_fullTrajectories: jax=4.234s fortran=1.330s ratio=3.183 res={'NTHETA': 14, 'NZETA': 1, 'NX': 5, 'NXI': 31} status=parity_ok
- HSX_FPCollisions_fullTrajectories: jax=3.651s fortran=88.504s ratio=0.041 res={'NTHETA': 5, 'NZETA': 5, 'NX': 2, 'NXI': 4} status=parity_ok
- geometryScheme4_2species_PAS_noEr: jax=3.721s fortran=0.953s ratio=3.905 res={'NTHETA': 8, 'NZETA': 11, 'NX': 4, 'NXI': 25} status=parity_ok
- tokamak_1species_PASCollisions_withEr_fullTrajectories: jax=3.662s fortran=0.017s ratio=215.396 res={'NTHETA': 10, 'NZETA': 1, 'NX': 3, 'NXI': 14} status=parity_ok
- monoenergetic_geometryScheme11: jax=3.639s fortran=0.861s ratio=4.227 res={'NTHETA': 8, 'NZETA': 16, 'NX': 1, 'NXI': 13} status=parity_ok
- HSX_FPCollisions_DKESTrajectories: jax=3.543s fortran=29.664s ratio=0.119 res={'NTHETA': 5, 'NZETA': 5, 'NX': 2, 'NXI': 4} status=parity_ok
- sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories: jax=3.732s fortran=1.706s ratio=2.188 res={'NTHETA': 6, 'NZETA': 19, 'NX': 2, 'NXI': 20} status=parity_ok
- sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_DKESTrajectories: jax=3.541s fortran=1.104s ratio=3.207 res={'NTHETA': 6, 'NZETA': 19, 'NX': 2, 'NXI': 20} status=parity_ok

## Runtime offenders (JAX/Fortran ratio)

- tokamak_1species_PASCollisions_withEr_fullTrajectories: ratio=148.339 jax=3.662s fortran=0.017s res={'NTHETA': 10, 'NZETA': 1, 'NX': 3, 'NXI': 14} status=parity_ok
- tokamak_1species_PASCollisions_noEr_Nx1: ratio=92.553 jax=2.697s fortran=0.017s res={'NTHETA': 21, 'NZETA': 1, 'NX': 1, 'NXI': 31} status=parity_ok
- transportMatrix_geometryScheme11: ratio=41.131 jax=1.909s fortran=0.025s res={'NTHETA': 5, 'NZETA': 5, 'NX': 2, 'NXI': 4} status=parity_ok
- transportMatrix_geometryScheme2: ratio=38.715 jax=2.346s fortran=0.031s res={'NTHETA': 5, 'NZETA': 5, 'NX': 2, 'NXI': 4} status=parity_ok
- tokamak_1species_PASCollisions_noEr: ratio=6.420 jax=3.055s fortran=0.309s res={'NTHETA': 21, 'NZETA': 1, 'NX': 8, 'NXI': 31} status=parity_ok
- tokamak_2species_PASCollisions_noEr: ratio=6.356 jax=3.252s fortran=0.331s res={'NTHETA': 19, 'NZETA': 1, 'NX': 7, 'NXI': 39} status=parity_ok
- HSX_PASCollisions_DKESTrajectories: ratio=3.688 jax=4.845s fortran=0.994s res={'NTHETA': 5, 'NZETA': 15, 'NX': 2, 'NXI': 20} status=parity_ok
- monoenergetic_geometryScheme11: ratio=2.924 jax=3.639s fortran=0.861s res={'NTHETA': 8, 'NZETA': 16, 'NX': 1, 'NXI': 13} status=parity_ok
- geometryScheme4_2species_PAS_noEr: ratio=2.654 jax=3.721s fortran=0.953s res={'NTHETA': 8, 'NZETA': 11, 'NX': 4, 'NXI': 25} status=parity_ok
- tokamak_2species_PASCollisions_withEr_fullTrajectories: ratio=2.266 jax=4.234s fortran=1.330s res={'NTHETA': 14, 'NZETA': 1, 'NX': 5, 'NXI': 31} status=parity_ok

## Memory offenders (absolute JAX RSS)


## Memory offenders (JAX/Fortran ratio)


## Mismatches

- None

## Print parity gaps

- None

## Failures and blockers

- None
