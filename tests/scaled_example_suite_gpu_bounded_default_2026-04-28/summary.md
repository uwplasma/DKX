# Scaled Example Suite Summary

- Cases: 39
- Practical status counts: parity_ok=39
- Strict status counts: parity_ok=39

## Runtime offenders (absolute JAX time)

- monoenergetic_geometryScheme1: jax=12.909s fortran=0.795s ratio=16.238 res={'NTHETA': 8, 'NZETA': 9, 'NX': 2, 'NXI': 23} status=parity_ok
- HSX_PASCollisions_fullTrajectories: jax=8.469s fortran=2.510s ratio=3.374 res={'NTHETA': 6, 'NZETA': 15, 'NX': 3, 'NXI': 20} status=parity_ok
- tokamak_2species_PASCollisions_withEr_fullTrajectories: jax=7.777s fortran=1.330s ratio=5.847 res={'NTHETA': 14, 'NZETA': 1, 'NX': 5, 'NXI': 31} status=parity_ok
- HSX_PASCollisions_DKESTrajectories: jax=6.867s fortran=0.994s ratio=6.909 res={'NTHETA': 5, 'NZETA': 15, 'NX': 2, 'NXI': 20} status=parity_ok
- sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories: jax=6.413s fortran=1.706s ratio=3.759 res={'NTHETA': 6, 'NZETA': 19, 'NX': 2, 'NXI': 20} status=parity_ok
- sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_DKESTrajectories: jax=5.757s fortran=1.104s ratio=5.215 res={'NTHETA': 6, 'NZETA': 19, 'NX': 2, 'NXI': 20} status=parity_ok
- geometryScheme4_2species_PAS_noEr: jax=5.658s fortran=0.953s ratio=5.937 res={'NTHETA': 8, 'NZETA': 11, 'NX': 4, 'NXI': 25} status=parity_ok
- monoenergetic_geometryScheme11: jax=5.606s fortran=0.861s ratio=6.511 res={'NTHETA': 8, 'NZETA': 16, 'NX': 1, 'NXI': 13} status=parity_ok
- HSX_FPCollisions_DKESTrajectories: jax=5.298s fortran=29.664s ratio=0.179 res={'NTHETA': 5, 'NZETA': 5, 'NX': 2, 'NXI': 4} status=parity_ok
- HSX_FPCollisions_fullTrajectories: jax=5.247s fortran=88.504s ratio=0.059 res={'NTHETA': 5, 'NZETA': 5, 'NX': 2, 'NXI': 4} status=parity_ok

## Runtime offenders (JAX/Fortran ratio)

- tokamak_1species_PASCollisions_withEr_fullTrajectories: ratio=177.882 jax=3.794s fortran=0.017s res={'NTHETA': 10, 'NZETA': 1, 'NX': 3, 'NXI': 14} status=parity_ok
- tokamak_1species_PASCollisions_noEr_Nx1: ratio=153.765 jax=3.443s fortran=0.017s res={'NTHETA': 21, 'NZETA': 1, 'NX': 1, 'NXI': 31} status=parity_ok
- transportMatrix_geometryScheme11: ratio=110.080 jax=3.489s fortran=0.025s res={'NTHETA': 5, 'NZETA': 5, 'NX': 2, 'NXI': 4} status=parity_ok
- transportMatrix_geometryScheme2: ratio=76.129 jax=3.191s fortran=0.031s res={'NTHETA': 5, 'NZETA': 5, 'NX': 2, 'NXI': 4} status=parity_ok
- monoenergetic_geometryScheme1: ratio=15.152 jax=12.909s fortran=0.795s res={'NTHETA': 8, 'NZETA': 9, 'NX': 2, 'NXI': 23} status=parity_ok
- tokamak_1species_PASCollisions_noEr: ratio=13.278 jax=4.951s fortran=0.309s res={'NTHETA': 21, 'NZETA': 1, 'NX': 8, 'NXI': 31} status=parity_ok
- tokamak_2species_PASCollisions_noEr: ratio=10.224 jax=4.250s fortran=0.331s res={'NTHETA': 19, 'NZETA': 1, 'NX': 7, 'NXI': 39} status=parity_ok
- HSX_PASCollisions_DKESTrajectories: ratio=6.043 jax=6.867s fortran=0.994s res={'NTHETA': 5, 'NZETA': 15, 'NX': 2, 'NXI': 20} status=parity_ok
- monoenergetic_geometryScheme11: ratio=5.542 jax=5.606s fortran=0.861s res={'NTHETA': 8, 'NZETA': 16, 'NX': 1, 'NXI': 13} status=parity_ok
- tokamak_2species_PASCollisions_withEr_fullTrajectories: ratio=5.162 jax=7.777s fortran=1.330s res={'NTHETA': 14, 'NZETA': 1, 'NX': 5, 'NXI': 31} status=parity_ok

## Memory offenders (absolute JAX RSS)

- geometryScheme4_2species_PAS_noEr: jax=1816.7MB fortran=162.7MB ratio=11.166 res={'NTHETA': 8, 'NZETA': 11, 'NX': 4, 'NXI': 25} status=parity_ok
- sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories: jax=1608.5MB fortran=144.6MB ratio=11.125 res={'NTHETA': 6, 'NZETA': 19, 'NX': 2, 'NXI': 20} status=parity_ok
- sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_DKESTrajectories: jax=1587.2MB fortran=130.7MB ratio=12.144 res={'NTHETA': 6, 'NZETA': 19, 'NX': 2, 'NXI': 20} status=parity_ok
- HSX_PASCollisions_fullTrajectories: jax=1577.4MB fortran=179.2MB ratio=8.802 res={'NTHETA': 6, 'NZETA': 15, 'NX': 3, 'NXI': 20} status=parity_ok
- geometryScheme4_1species_PAS_withEr_DKESTrajectories: jax=1264.3MB fortran=127.3MB ratio=9.933 res={'NTHETA': 8, 'NZETA': 11, 'NX': 3, 'NXI': 24} status=parity_ok
- tokamak_2species_PASCollisions_withEr_fullTrajectories: jax=1245.5MB fortran=121.8MB ratio=10.223 res={'NTHETA': 14, 'NZETA': 1, 'NX': 5, 'NXI': 31} status=parity_ok
- HSX_PASCollisions_DKESTrajectories: jax=1184.1MB fortran=112.0MB ratio=10.568 res={'NTHETA': 5, 'NZETA': 15, 'NX': 2, 'NXI': 20} status=parity_ok
- tokamak_2species_PASCollisions_noEr: jax=1148.3MB fortran=123.6MB ratio=9.291 res={'NTHETA': 19, 'NZETA': 1, 'NX': 7, 'NXI': 39} status=parity_ok
- monoenergetic_geometryScheme11: jax=1003.7MB fortran=118.7MB ratio=8.458 res={'NTHETA': 8, 'NZETA': 16, 'NX': 1, 'NXI': 13} status=parity_ok
- monoenergetic_geometryScheme1: jax=996.6MB fortran=110.2MB ratio=9.042 res={'NTHETA': 8, 'NZETA': 9, 'NX': 2, 'NXI': 23} status=parity_ok

## Memory offenders (JAX/Fortran ratio)

- sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_DKESTrajectories: ratio=12.144 jax=1587.2MB fortran=130.7MB res={'NTHETA': 6, 'NZETA': 19, 'NX': 2, 'NXI': 20} status=parity_ok
- geometryScheme4_2species_PAS_noEr: ratio=11.166 jax=1816.7MB fortran=162.7MB res={'NTHETA': 8, 'NZETA': 11, 'NX': 4, 'NXI': 25} status=parity_ok
- sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories: ratio=11.125 jax=1608.5MB fortran=144.6MB res={'NTHETA': 6, 'NZETA': 19, 'NX': 2, 'NXI': 20} status=parity_ok
- HSX_PASCollisions_DKESTrajectories: ratio=10.568 jax=1184.1MB fortran=112.0MB res={'NTHETA': 5, 'NZETA': 15, 'NX': 2, 'NXI': 20} status=parity_ok
- tokamak_1species_FPCollisions_noEr_withPhi1InDKE: ratio=10.405 jax=932.1MB fortran=89.6MB res={'NTHETA': 5, 'NZETA': 1, 'NX': 2, 'NXI': 4} status=parity_ok
- tokamak_2species_PASCollisions_withEr_fullTrajectories: ratio=10.223 jax=1245.5MB fortran=121.8MB res={'NTHETA': 14, 'NZETA': 1, 'NX': 5, 'NXI': 31} status=parity_ok
- geometryScheme4_1species_PAS_withEr_DKESTrajectories: ratio=9.933 jax=1264.3MB fortran=127.3MB res={'NTHETA': 8, 'NZETA': 11, 'NX': 3, 'NXI': 24} status=parity_ok
- geometryScheme4_2species_noEr: ratio=9.906 jax=913.8MB fortran=92.2MB res={'NTHETA': 5, 'NZETA': 5, 'NX': 2, 'NXI': 4} status=parity_ok
- sfincsPaperFigure3_geometryScheme11_FPCollisions_2Species_fullTrajectories: ratio=9.794 jax=920.2MB fortran=94.0MB res={'NTHETA': 5, 'NZETA': 5, 'NX': 2, 'NXI': 4} status=parity_ok
- geometryScheme4_2species_noEr_withQN: ratio=9.788 jax=930.5MB fortran=95.1MB res={'NTHETA': 5, 'NZETA': 5, 'NX': 2, 'NXI': 4} status=parity_ok

## Mismatches

- None

## Print parity gaps

- None

## Failures and blockers

- None
