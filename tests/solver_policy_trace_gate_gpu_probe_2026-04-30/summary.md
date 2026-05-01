# Scaled Example Suite Summary

- Cases: 7
- Practical status counts: parity_ok=7
- Strict status counts: parity_ok=7

## Runtime offenders (absolute JAX time)

- HSX_PASCollisions_fullTrajectories: jax=13.053s fortran=2.510s ratio=5.200 res={'NTHETA': 6, 'NZETA': 15, 'NX': 3, 'NXI': 20} status=parity_ok
- sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories: jax=11.745s fortran=1.706s ratio=6.885 res={'NTHETA': 6, 'NZETA': 19, 'NX': 2, 'NXI': 20} status=parity_ok
- geometryScheme4_2species_PAS_noEr: jax=8.566s fortran=0.953s ratio=8.989 res={'NTHETA': 8, 'NZETA': 11, 'NX': 4, 'NXI': 25} status=parity_ok
- HSX_PASCollisions_DKESTrajectories: jax=7.615s fortran=0.994s ratio=7.661 res={'NTHETA': 5, 'NZETA': 15, 'NX': 2, 'NXI': 20} status=parity_ok
- tokamak_1species_PASCollisions_withEr_fullTrajectories: jax=6.459s fortran=0.017s ratio=379.953 res={'NTHETA': 10, 'NZETA': 1, 'NX': 3, 'NXI': 14} status=parity_ok
- tokamak_1species_FPCollisions_withEr_fullTrajectories: jax=3.336s fortran=154.953s ratio=0.022 res={'NTHETA': 5, 'NZETA': 1, 'NX': 2, 'NXI': 4} status=parity_ok
- tokamak_1species_FPCollisions_noEr: jax=2.936s fortran=160.856s ratio=0.018 res={'NTHETA': 5, 'NZETA': 1, 'NX': 2, 'NXI': 4} status=parity_ok

## Runtime offenders (JAX/Fortran ratio)

- tokamak_1species_PASCollisions_withEr_fullTrajectories: ratio=279.731 jax=6.459s fortran=0.017s res={'NTHETA': 10, 'NZETA': 1, 'NX': 3, 'NXI': 14} status=parity_ok
- geometryScheme4_2species_PAS_noEr: ratio=7.438 jax=8.566s fortran=0.953s res={'NTHETA': 8, 'NZETA': 11, 'NX': 4, 'NXI': 25} status=parity_ok
- HSX_PASCollisions_DKESTrajectories: ratio=6.085 jax=7.615s fortran=0.994s res={'NTHETA': 5, 'NZETA': 15, 'NX': 2, 'NXI': 20} status=parity_ok
- sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories: ratio=5.977 jax=11.745s fortran=1.706s res={'NTHETA': 6, 'NZETA': 19, 'NX': 2, 'NXI': 20} status=parity_ok
- HSX_PASCollisions_fullTrajectories: ratio=4.597 jax=13.053s fortran=2.510s res={'NTHETA': 6, 'NZETA': 15, 'NX': 3, 'NXI': 20} status=parity_ok
- tokamak_1species_FPCollisions_withEr_fullTrajectories: ratio=0.012 jax=3.336s fortran=154.953s res={'NTHETA': 5, 'NZETA': 1, 'NX': 2, 'NXI': 4} status=parity_ok
- tokamak_1species_FPCollisions_noEr: ratio=0.009 jax=2.936s fortran=160.856s res={'NTHETA': 5, 'NZETA': 1, 'NX': 2, 'NXI': 4} status=parity_ok

## Memory offenders (absolute JAX RSS)


## Memory offenders (JAX/Fortran ratio)


## Mismatches

- None

## Print parity gaps

- None

## Failures and blockers

- None
