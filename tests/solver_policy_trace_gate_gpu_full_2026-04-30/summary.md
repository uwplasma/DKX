# Scaled Example Suite Summary

- Cases: 39
- Practical status counts: parity_ok=39
- Strict status counts: parity_ok=39

## Runtime offenders (absolute JAX time)

- sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories: jax=24.321s fortran=1.706s ratio=14.256 res={'NTHETA': 6, 'NZETA': 19, 'NX': 2, 'NXI': 20} status=parity_ok
- sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_DKESTrajectories: jax=22.803s fortran=1.104s ratio=20.655 res={'NTHETA': 6, 'NZETA': 19, 'NX': 2, 'NXI': 20} status=parity_ok
- HSX_PASCollisions_fullTrajectories: jax=19.303s fortran=2.510s ratio=7.690 res={'NTHETA': 6, 'NZETA': 15, 'NX': 3, 'NXI': 20} status=parity_ok
- tokamak_2species_PASCollisions_withEr_fullTrajectories: jax=14.314s fortran=1.330s ratio=10.763 res={'NTHETA': 14, 'NZETA': 1, 'NX': 5, 'NXI': 31} status=parity_ok
- geometryScheme4_2species_PAS_noEr: jax=10.288s fortran=0.953s ratio=10.795 res={'NTHETA': 8, 'NZETA': 11, 'NX': 4, 'NXI': 25} status=parity_ok
- tokamak_1species_PASCollisions_noEr: jax=9.481s fortran=0.309s ratio=30.682 res={'NTHETA': 21, 'NZETA': 1, 'NX': 8, 'NXI': 31} status=parity_ok
- HSX_PASCollisions_DKESTrajectories: jax=8.622s fortran=0.994s ratio=8.674 res={'NTHETA': 5, 'NZETA': 15, 'NX': 2, 'NXI': 20} status=parity_ok
- monoenergetic_geometryScheme11: jax=7.767s fortran=0.861s ratio=9.021 res={'NTHETA': 8, 'NZETA': 16, 'NX': 1, 'NXI': 13} status=parity_ok
- tokamak_1species_PASCollisions_withEr_fullTrajectories: jax=7.312s fortran=0.017s ratio=430.118 res={'NTHETA': 10, 'NZETA': 1, 'NX': 3, 'NXI': 14} status=parity_ok
- geometryScheme4_1species_PAS_withEr_DKESTrajectories: jax=6.857s fortran=1.365s ratio=5.023 res={'NTHETA': 8, 'NZETA': 11, 'NX': 3, 'NXI': 24} status=parity_ok

## Runtime offenders (JAX/Fortran ratio)

- tokamak_1species_PASCollisions_withEr_fullTrajectories: ratio=336.299 jax=7.312s fortran=0.017s res={'NTHETA': 10, 'NZETA': 1, 'NX': 3, 'NXI': 14} status=parity_ok
- tokamak_1species_PASCollisions_noEr_Nx1: ratio=182.420 jax=4.698s fortran=0.017s res={'NTHETA': 21, 'NZETA': 1, 'NX': 1, 'NXI': 31} status=parity_ok
- transportMatrix_geometryScheme11: ratio=108.094 jax=4.397s fortran=0.025s res={'NTHETA': 5, 'NZETA': 5, 'NX': 2, 'NXI': 4} status=parity_ok
- transportMatrix_geometryScheme2: ratio=92.502 jax=4.393s fortran=0.031s res={'NTHETA': 5, 'NZETA': 5, 'NX': 2, 'NXI': 4} status=parity_ok
- tokamak_1species_PASCollisions_noEr: ratio=25.563 jax=9.481s fortran=0.309s res={'NTHETA': 21, 'NZETA': 1, 'NX': 8, 'NXI': 31} status=parity_ok
- sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_DKESTrajectories: ratio=19.268 jax=22.803s fortran=1.104s res={'NTHETA': 6, 'NZETA': 19, 'NX': 2, 'NXI': 20} status=parity_ok
- sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories: ratio=13.249 jax=24.321s fortran=1.706s res={'NTHETA': 6, 'NZETA': 19, 'NX': 2, 'NXI': 20} status=parity_ok
- tokamak_2species_PASCollisions_noEr: ratio=12.921 jax=5.865s fortran=0.331s res={'NTHETA': 19, 'NZETA': 1, 'NX': 7, 'NXI': 39} status=parity_ok
- tokamak_2species_PASCollisions_withEr_fullTrajectories: ratio=9.387 jax=14.314s fortran=1.330s res={'NTHETA': 14, 'NZETA': 1, 'NX': 5, 'NXI': 31} status=parity_ok
- geometryScheme4_2species_PAS_noEr: ratio=8.966 jax=10.288s fortran=0.953s res={'NTHETA': 8, 'NZETA': 11, 'NX': 4, 'NXI': 25} status=parity_ok

## Memory offenders (absolute JAX RSS)


## Memory offenders (JAX/Fortran ratio)


## Mismatches

- None

## Print parity gaps

- None

## Failures and blockers

- None
