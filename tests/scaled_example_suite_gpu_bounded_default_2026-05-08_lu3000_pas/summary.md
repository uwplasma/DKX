# Scaled Example Suite Summary

- Cases: 39
- Practical status counts: parity_ok=39
- Strict status counts: parity_ok=39

## Runtime offenders (absolute JAX time)

- tokamak_1species_FPCollisions_withEr_DKESTrajectories: jax=30.392s fortran=6.958s ratio=4.368 res={'NTHETA': 25, 'NZETA': 1, 'NX': 8, 'NXI': 100} status=parity_ok
- tokamak_1species_PASCollisions_noEr_Nx1: jax=27.403s fortran=75.533s ratio=0.363 res={'NTHETA': 25, 'NX': 4, 'NXI': 100, 'NZETA': 1} status=parity_ok
- tokamak_2species_PASCollisions_withEr_fullTrajectories: jax=25.021s fortran=76.530s ratio=0.327 res={'NTHETA': 25, 'NZETA': 1, 'NX': 8, 'NXI': 100} status=parity_ok
- tokamak_1species_FPCollisions_noEr_withPhi1InDKE: jax=15.404s fortran=41.132s ratio=0.374 res={'NTHETA': 25, 'NZETA': 1, 'NX': 8, 'NXI': 100} status=parity_ok
- tokamak_2species_PASCollisions_noEr: jax=13.226s fortran=75.362s ratio=0.175 res={'NTHETA': 25, 'NX': 8, 'NXI': 100, 'NZETA': 1} status=parity_ok
- tokamak_1species_PASCollisions_withEr_fullTrajectories: jax=13.193s fortran=75.698s ratio=0.174 res={'NTHETA': 25, 'NZETA': 1, 'NX': 8, 'NXI': 100} status=parity_ok
- tokamak_1species_PASCollisions_noEr: jax=12.654s fortran=75.566s ratio=0.167 res={'NTHETA': 25, 'NX': 8, 'NXI': 100, 'NZETA': 1} status=parity_ok
- tokamak_1species_FPCollisions_withEr_fullTrajectories: jax=12.558s fortran=6.736s ratio=1.864 res={'NTHETA': 25, 'NZETA': 1, 'NX': 8, 'NXI': 100} status=parity_ok
- tokamak_1species_PASCollisions_noEr_withQN: jax=10.019s fortran=75.242s ratio=0.133 res={'NTHETA': 25, 'NX': 8, 'NXI': 100, 'NZETA': 1} status=parity_ok
- HSX_PASCollisions_fullTrajectories: jax=7.574s fortran=2.510s ratio=3.018 res={'NTHETA': 6, 'NX': 3, 'NXI': 20, 'NZETA': 15} status=parity_ok

## Runtime offenders (JAX/Fortran ratio)

- transportMatrix_geometryScheme11: ratio=110.080 jax=2.752s fortran=0.025s res={'NTHETA': 5, 'NX': 2, 'NXI': 4, 'NZETA': 5} status=parity_ok
- transportMatrix_geometryScheme2: ratio=76.129 jax=2.360s fortran=0.031s res={'NTHETA': 5, 'NX': 2, 'NXI': 4, 'NZETA': 5} status=parity_ok
- HSX_PASCollisions_DKESTrajectories: ratio=6.043 jax=6.007s fortran=0.994s res={'NTHETA': 5, 'NX': 2, 'NXI': 20, 'NZETA': 15} status=parity_ok
- monoenergetic_geometryScheme11: ratio=5.542 jax=4.772s fortran=0.861s res={'NTHETA': 8, 'NX': 1, 'NXI': 13, 'NZETA': 16} status=parity_ok
- geometryScheme4_2species_PAS_noEr: ratio=5.016 jax=4.780s fortran=0.953s res={'NTHETA': 8, 'NX': 4, 'NXI': 25, 'NZETA': 11} status=parity_ok
- sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_DKESTrajectories: ratio=4.389 jax=4.846s fortran=1.104s res={'NTHETA': 6, 'NX': 2, 'NXI': 20, 'NZETA': 19} status=parity_ok
- tokamak_1species_FPCollisions_withEr_DKESTrajectories: ratio=4.368 jax=30.392s fortran=6.958s res={'NTHETA': 25, 'NZETA': 1, 'NX': 8, 'NXI': 100} status=parity_ok
- monoenergetic_geometryScheme1: ratio=3.435 jax=2.731s fortran=0.795s res={'NTHETA': 8, 'NX': 2, 'NXI': 23, 'NZETA': 9} status=parity_ok
- monoenergetic_geometryScheme5_ASCII: ratio=3.282 jax=3.453s fortran=1.052s res={'NTHETA': 10, 'NX': 1, 'NXI': 16, 'NZETA': 20} status=parity_ok
- sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories: ratio=3.235 jax=5.519s fortran=1.706s res={'NTHETA': 6, 'NX': 2, 'NXI': 20, 'NZETA': 19} status=parity_ok

## Memory offenders (absolute JAX RSS)

- tokamak_2species_PASCollisions_withEr_fullTrajectories: jax=2322.5MB fortran=386.6MB ratio=6.008 res={'NTHETA': 25, 'NZETA': 1, 'NX': 8, 'NXI': 100} status=parity_ok
- geometryScheme4_2species_PAS_noEr: jax=1816.7MB fortran=162.7MB ratio=11.166 res={'NTHETA': 8, 'NX': 4, 'NXI': 25, 'NZETA': 11} status=parity_ok
- tokamak_2species_PASCollisions_noEr: jax=1664.9MB fortran=215.3MB ratio=7.734 res={'NTHETA': 25, 'NX': 8, 'NXI': 100, 'NZETA': 1} status=parity_ok
- sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories: jax=1608.5MB fortran=144.6MB ratio=11.125 res={'NTHETA': 6, 'NX': 2, 'NXI': 20, 'NZETA': 19} status=parity_ok
- sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_DKESTrajectories: jax=1587.2MB fortran=130.7MB ratio=12.144 res={'NTHETA': 6, 'NX': 2, 'NXI': 20, 'NZETA': 19} status=parity_ok
- HSX_PASCollisions_fullTrajectories: jax=1577.4MB fortran=179.2MB ratio=8.802 res={'NTHETA': 6, 'NX': 3, 'NXI': 20, 'NZETA': 15} status=parity_ok
- tokamak_1species_PASCollisions_withEr_fullTrajectories: jax=1572.1MB fortran=248.9MB ratio=6.316 res={'NTHETA': 25, 'NZETA': 1, 'NX': 8, 'NXI': 100} status=parity_ok
- tokamak_1species_FPCollisions_withEr_DKESTrajectories: jax=1338.7MB fortran=105.1MB ratio=12.741 res={'NTHETA': 25, 'NZETA': 1, 'NX': 8, 'NXI': 100} status=parity_ok
- tokamak_1species_FPCollisions_withEr_fullTrajectories: jax=1335.2MB fortran=139.4MB ratio=9.578 res={'NTHETA': 25, 'NZETA': 1, 'NX': 8, 'NXI': 100} status=parity_ok
- geometryScheme4_1species_PAS_withEr_DKESTrajectories: jax=1264.3MB fortran=127.3MB ratio=9.933 res={'NTHETA': 8, 'NX': 3, 'NXI': 24, 'NZETA': 11} status=parity_ok

## Memory offenders (JAX/Fortran ratio)

- tokamak_1species_FPCollisions_withEr_DKESTrajectories: ratio=12.741 jax=1338.7MB fortran=105.1MB res={'NTHETA': 25, 'NZETA': 1, 'NX': 8, 'NXI': 100} status=parity_ok
- sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_DKESTrajectories: ratio=12.144 jax=1587.2MB fortran=130.7MB res={'NTHETA': 6, 'NX': 2, 'NXI': 20, 'NZETA': 19} status=parity_ok
- geometryScheme4_2species_PAS_noEr: ratio=11.166 jax=1816.7MB fortran=162.7MB res={'NTHETA': 8, 'NX': 4, 'NXI': 25, 'NZETA': 11} status=parity_ok
- sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories: ratio=11.125 jax=1608.5MB fortran=144.6MB res={'NTHETA': 6, 'NX': 2, 'NXI': 20, 'NZETA': 19} status=parity_ok
- HSX_PASCollisions_DKESTrajectories: ratio=10.568 jax=1184.1MB fortran=112.0MB res={'NTHETA': 5, 'NX': 2, 'NXI': 20, 'NZETA': 15} status=parity_ok
- geometryScheme4_1species_PAS_withEr_DKESTrajectories: ratio=9.933 jax=1264.3MB fortran=127.3MB res={'NTHETA': 8, 'NX': 3, 'NXI': 24, 'NZETA': 11} status=parity_ok
- geometryScheme4_2species_noEr: ratio=9.906 jax=913.8MB fortran=92.2MB res={'NTHETA': 5, 'NX': 2, 'NXI': 4, 'NZETA': 5} status=parity_ok
- sfincsPaperFigure3_geometryScheme11_FPCollisions_2Species_fullTrajectories: ratio=9.794 jax=920.2MB fortran=94.0MB res={'NTHETA': 5, 'NX': 2, 'NXI': 4, 'NZETA': 5} status=parity_ok
- geometryScheme4_2species_noEr_withQN: ratio=9.788 jax=930.5MB fortran=95.1MB res={'NTHETA': 5, 'NX': 2, 'NXI': 4, 'NZETA': 5} status=parity_ok
- tokamak_1species_PASCollisions_noEr_Nx1: ratio=9.783 jax=1166.5MB fortran=119.2MB res={'NTHETA': 25, 'NX': 4, 'NXI': 100, 'NZETA': 1} status=parity_ok

## Mismatches

- None

## Reference-quality rows

- None

## Print parity gaps

- tokamak_1species_FPCollisions_noEr: 8/9 missing=residual
- tokamak_1species_FPCollisions_withEr_DKESTrajectories: 8/9 missing=residual
- tokamak_1species_FPCollisions_withEr_fullTrajectories: 8/9 missing=residual
- tokamak_1species_PASCollisions_withEr_fullTrajectories: 8/9 missing=residual
- tokamak_2species_PASCollisions_withEr_fullTrajectories: 8/9 missing=residual

## Failures and blockers

- None
