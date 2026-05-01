# Scaled Example Suite Summary

- Cases: 9
- Practical status counts: parity_ok=9
- Strict status counts: parity_ok=9

## Runtime offenders (absolute JAX time)

- HSX_PASCollisions_DKESTrajectories: jax=4.850s fortran=0.994s ratio=4.880 res={'NTHETA': 5, 'NZETA': 15, 'NX': 2, 'NXI': 20} status=parity_ok
- HSX_PASCollisions_fullTrajectories: jax=4.849s fortran=2.510s ratio=1.932 res={'NTHETA': 6, 'NZETA': 15, 'NX': 3, 'NXI': 20} status=parity_ok
- geometryScheme4_2species_PAS_noEr: jax=3.158s fortran=0.953s ratio=3.314 res={'NTHETA': 8, 'NZETA': 11, 'NX': 4, 'NXI': 25} status=parity_ok
- tokamak_1species_PASCollisions_withEr_fullTrajectories: jax=3.021s fortran=0.017s ratio=177.679 res={'NTHETA': 10, 'NZETA': 1, 'NX': 3, 'NXI': 14} status=parity_ok
- sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories: jax=3.021s fortran=1.706s ratio=1.771 res={'NTHETA': 6, 'NZETA': 19, 'NX': 2, 'NXI': 20} status=parity_ok
- tokamak_1species_FPCollisions_noEr_withPhi1InDKE: jax=2.200s fortran=259.575s ratio=0.008 res={'NTHETA': 5, 'NZETA': 1, 'NX': 2, 'NXI': 4} status=parity_ok
- tokamak_1species_FPCollisions_withEr_fullTrajectories: jax=1.834s fortran=154.953s ratio=0.012 res={'NTHETA': 5, 'NZETA': 1, 'NX': 2, 'NXI': 4} status=parity_ok
- tokamak_1species_FPCollisions_noEr_withQN: jax=1.781s fortran=237.879s ratio=0.007 res={'NTHETA': 5, 'NZETA': 1, 'NX': 2, 'NXI': 4} status=parity_ok
- tokamak_1species_FPCollisions_noEr: jax=1.703s fortran=160.856s ratio=0.011 res={'NTHETA': 5, 'NZETA': 1, 'NX': 2, 'NXI': 4} status=parity_ok

## Runtime offenders (JAX/Fortran ratio)

- tokamak_1species_PASCollisions_withEr_fullTrajectories: ratio=123.584 jax=3.021s fortran=0.017s res={'NTHETA': 10, 'NZETA': 1, 'NX': 3, 'NXI': 14} status=parity_ok
- HSX_PASCollisions_DKESTrajectories: ratio=3.824 jax=4.850s fortran=0.994s res={'NTHETA': 5, 'NZETA': 15, 'NX': 2, 'NXI': 20} status=parity_ok
- geometryScheme4_2species_PAS_noEr: ratio=2.251 jax=3.158s fortran=0.953s res={'NTHETA': 8, 'NZETA': 11, 'NX': 4, 'NXI': 25} status=parity_ok
- HSX_PASCollisions_fullTrajectories: ratio=1.490 jax=4.849s fortran=2.510s res={'NTHETA': 6, 'NZETA': 15, 'NX': 3, 'NXI': 20} status=parity_ok
- sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories: ratio=1.157 jax=3.021s fortran=1.706s res={'NTHETA': 6, 'NZETA': 19, 'NX': 2, 'NXI': 20} status=parity_ok
- tokamak_1species_FPCollisions_withEr_fullTrajectories: ratio=0.006 jax=1.834s fortran=154.953s res={'NTHETA': 5, 'NZETA': 1, 'NX': 2, 'NXI': 4} status=parity_ok
- tokamak_1species_FPCollisions_noEr: ratio=0.005 jax=1.703s fortran=160.856s res={'NTHETA': 5, 'NZETA': 1, 'NX': 2, 'NXI': 4} status=parity_ok
- tokamak_1species_FPCollisions_noEr_withPhi1InDKE: ratio=0.005 jax=2.200s fortran=259.575s res={'NTHETA': 5, 'NZETA': 1, 'NX': 2, 'NXI': 4} status=parity_ok
- tokamak_1species_FPCollisions_noEr_withQN: ratio=0.004 jax=1.781s fortran=237.879s res={'NTHETA': 5, 'NZETA': 1, 'NX': 2, 'NXI': 4} status=parity_ok

## Memory offenders (absolute JAX RSS)


## Memory offenders (JAX/Fortran ratio)


## Mismatches

- None

## Print parity gaps

- None

## Failures and blockers

- None
