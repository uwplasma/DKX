# Scaled Example Suite Summary

- Cases: 39
- Practical status counts: parity_ok=39
- Strict status counts: parity_ok=39

## Runtime offenders (absolute JAX time)

- tokamak_1species_FPCollisions_noEr_withPhi1InDKE: jax=12.527s fortran=41.132s ratio=0.305 res={'NTHETA': 25, 'NX': 8, 'NXI': 100, 'NZETA': 1} status=parity_ok
- tokamak_2species_PASCollisions_withEr_fullTrajectories: jax=11.848s fortran=76.530s ratio=0.155 res={'NTHETA': 25, 'NZETA': 1, 'NX': 8, 'NXI': 100} status=parity_ok
- tokamak_1species_FPCollisions_noEr_withQN: jax=8.255s fortran=10.952s ratio=0.754 res={'NTHETA': 25, 'NX': 8, 'NXI': 100, 'NZETA': 1} status=parity_ok
- tokamak_1species_PASCollisions_withEr_fullTrajectories: jax=6.231s fortran=75.698s ratio=0.082 res={'NTHETA': 25, 'NZETA': 1, 'NX': 8, 'NXI': 100} status=parity_ok
- tokamak_1species_PASCollisions_noEr_withQN: jax=4.250s fortran=75.242s ratio=0.056 res={'NTHETA': 25, 'NX': 8, 'NXI': 100, 'NZETA': 1} status=parity_ok
- tokamak_1species_FPCollisions_withEr_fullTrajectories: jax=4.147s fortran=6.736s ratio=0.616 res={'NTHETA': 25, 'NZETA': 1, 'NX': 8, 'NXI': 100} status=parity_ok
- tokamak_1species_FPCollisions_withEr_DKESTrajectories: jax=3.437s fortran=6.958s ratio=0.494 res={'NTHETA': 25, 'NZETA': 1, 'NX': 8, 'NXI': 100} status=parity_ok
- HSX_PASCollisions_DKESTrajectories: jax=3.253s fortran=0.994s ratio=3.273 res={'NTHETA': 5, 'NX': 2, 'NXI': 20, 'NZETA': 15} status=parity_ok
- HSX_PASCollisions_fullTrajectories: jax=3.071s fortran=2.510s ratio=1.224 res={'NTHETA': 6, 'NX': 3, 'NXI': 20, 'NZETA': 15} status=parity_ok
- HSX_FPCollisions_DKESTrajectories: jax=2.438s fortran=29.664s ratio=0.082 res={'NTHETA': 5, 'NX': 2, 'NXI': 4, 'NZETA': 5} status=parity_ok

## Runtime offenders (JAX/Fortran ratio)

- transportMatrix_geometryScheme11: ratio=42.360 jax=1.059s fortran=0.025s res={'NTHETA': 5, 'NX': 2, 'NXI': 4, 'NZETA': 5} status=parity_ok
- transportMatrix_geometryScheme2: ratio=32.129 jax=0.996s fortran=0.031s res={'NTHETA': 5, 'NX': 2, 'NXI': 4, 'NZETA': 5} status=parity_ok
- HSX_PASCollisions_DKESTrajectories: ratio=3.273 jax=3.253s fortran=0.994s res={'NTHETA': 5, 'NX': 2, 'NXI': 20, 'NZETA': 15} status=parity_ok
- monoenergetic_geometryScheme11: ratio=2.568 jax=2.211s fortran=0.861s res={'NTHETA': 8, 'NX': 1, 'NXI': 13, 'NZETA': 16} status=parity_ok
- geometryScheme4_2species_PAS_noEr: ratio=2.050 jax=1.954s fortran=0.953s res={'NTHETA': 8, 'NX': 4, 'NXI': 25, 'NZETA': 11} status=parity_ok
- sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_DKESTrajectories: ratio=1.822 jax=2.011s fortran=1.104s res={'NTHETA': 6, 'NX': 2, 'NXI': 20, 'NZETA': 19} status=parity_ok
- monoenergetic_geometryScheme1: ratio=1.624 jax=1.291s fortran=0.795s res={'NTHETA': 8, 'NX': 2, 'NXI': 23, 'NZETA': 9} status=parity_ok
- sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories: ratio=1.240 jax=2.115s fortran=1.706s res={'NTHETA': 6, 'NX': 2, 'NXI': 20, 'NZETA': 19} status=parity_ok
- HSX_PASCollisions_fullTrajectories: ratio=1.224 jax=3.071s fortran=2.510s res={'NTHETA': 6, 'NX': 3, 'NXI': 20, 'NZETA': 15} status=parity_ok
- monoenergetic_geometryScheme5_netCDF: ratio=1.221 jax=1.256s fortran=1.029s res={'NTHETA': 8, 'NX': 1, 'NXI': 13, 'NZETA': 16} status=parity_ok

## Memory offenders (absolute JAX RSS)

- tokamak_1species_FPCollisions_withEr_fullTrajectories: jax=3621.6MB fortran=139.4MB ratio=25.980 res={'NTHETA': 25, 'NZETA': 1, 'NX': 8, 'NXI': 100} status=parity_ok
- tokamak_1species_FPCollisions_withEr_DKESTrajectories: jax=2906.3MB fortran=105.1MB ratio=27.660 res={'NTHETA': 25, 'NZETA': 1, 'NX': 8, 'NXI': 100} status=parity_ok
- tokamak_2species_PASCollisions_withEr_fullTrajectories: jax=2262.5MB fortran=386.6MB ratio=5.853 res={'NTHETA': 25, 'NZETA': 1, 'NX': 8, 'NXI': 100} status=parity_ok
- tokamak_2species_PASCollisions_noEr: jax=2030.6MB fortran=215.3MB ratio=9.432 res={'NTHETA': 25, 'NX': 8, 'NXI': 100, 'NZETA': 1} status=parity_ok
- geometryScheme4_2species_PAS_noEr: jax=1808.8MB fortran=162.7MB ratio=11.117 res={'NTHETA': 8, 'NX': 4, 'NXI': 25, 'NZETA': 11} status=parity_ok
- sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories: jax=1482.2MB fortran=144.6MB ratio=10.252 res={'NTHETA': 6, 'NX': 2, 'NXI': 20, 'NZETA': 19} status=parity_ok
- sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_DKESTrajectories: jax=1458.0MB fortran=130.7MB ratio=11.155 res={'NTHETA': 6, 'NX': 2, 'NXI': 20, 'NZETA': 19} status=parity_ok
- HSX_PASCollisions_fullTrajectories: jax=1441.6MB fortran=179.2MB ratio=8.045 res={'NTHETA': 6, 'NX': 3, 'NXI': 20, 'NZETA': 15} status=parity_ok
- tokamak_1species_PASCollisions_withEr_fullTrajectories: jax=1319.5MB fortran=248.9MB ratio=5.301 res={'NTHETA': 25, 'NZETA': 1, 'NX': 8, 'NXI': 100} status=parity_ok
- monoenergetic_geometryScheme11: jax=1205.2MB fortran=118.7MB ratio=10.156 res={'NTHETA': 8, 'NX': 1, 'NXI': 13, 'NZETA': 16} status=parity_ok

## Memory offenders (JAX/Fortran ratio)

- tokamak_1species_FPCollisions_withEr_DKESTrajectories: ratio=27.660 jax=2906.3MB fortran=105.1MB res={'NTHETA': 25, 'NZETA': 1, 'NX': 8, 'NXI': 100} status=parity_ok
- tokamak_1species_FPCollisions_withEr_fullTrajectories: ratio=25.980 jax=3621.6MB fortran=139.4MB res={'NTHETA': 25, 'NZETA': 1, 'NX': 8, 'NXI': 100} status=parity_ok
- sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_DKESTrajectories: ratio=11.155 jax=1458.0MB fortran=130.7MB res={'NTHETA': 6, 'NX': 2, 'NXI': 20, 'NZETA': 19} status=parity_ok
- geometryScheme4_2species_PAS_noEr: ratio=11.117 jax=1808.8MB fortran=162.7MB res={'NTHETA': 8, 'NX': 4, 'NXI': 25, 'NZETA': 11} status=parity_ok
- sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories: ratio=10.252 jax=1482.2MB fortran=144.6MB res={'NTHETA': 6, 'NX': 2, 'NXI': 20, 'NZETA': 19} status=parity_ok
- HSX_PASCollisions_DKESTrajectories: ratio=10.231 jax=1146.3MB fortran=112.0MB res={'NTHETA': 5, 'NX': 2, 'NXI': 20, 'NZETA': 15} status=parity_ok
- monoenergetic_geometryScheme11: ratio=10.156 jax=1205.2MB fortran=118.7MB res={'NTHETA': 8, 'NX': 1, 'NXI': 13, 'NZETA': 16} status=parity_ok
- tokamak_2species_PASCollisions_noEr: ratio=9.432 jax=2030.6MB fortran=215.3MB res={'NTHETA': 25, 'NX': 8, 'NXI': 100, 'NZETA': 1} status=parity_ok
- geometryScheme4_1species_PAS_withEr_DKESTrajectories: ratio=8.486 jax=1080.1MB fortran=127.3MB res={'NTHETA': 8, 'NX': 3, 'NXI': 24, 'NZETA': 11} status=parity_ok
- HSX_PASCollisions_fullTrajectories: ratio=8.045 jax=1441.6MB fortran=179.2MB res={'NTHETA': 6, 'NX': 3, 'NXI': 20, 'NZETA': 15} status=parity_ok

## Mismatches

- None

## Reference-quality rows

- None

## Print parity gaps

- tokamak_1species_FPCollisions_withEr_DKESTrajectories: 8/9 missing=residual
- tokamak_1species_FPCollisions_withEr_fullTrajectories: 8/9 missing=residual
- tokamak_1species_PASCollisions_withEr_fullTrajectories: 8/9 missing=residual
- tokamak_2species_PASCollisions_withEr_fullTrajectories: 8/9 missing=residual

## Failures and blockers

- None
