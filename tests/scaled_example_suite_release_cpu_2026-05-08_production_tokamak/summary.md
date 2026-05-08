# CPU frozen suite with production-floor tokamak FP/PAS refresh

Replacement source: `/Users/rogeriojorge/local/tests/sfincs_jax/tests/production_floor_cpu_tokamak_fp_lu3000_2026-05-08`
- tokamak_1species_FPCollisions_noEr: status=parity_ok jax=2.327s logged=1.555s rss=528.9MB res={'NTHETA': 25, 'NZETA': 1, 'NX': 8, 'NXI': 100}
- tokamak_1species_FPCollisions_noEr_withPhi1InDKE: status=parity_ok jax=13.276s logged=12.527s rss=1024.1MB res={'NTHETA': 25, 'NZETA': 1, 'NX': 8, 'NXI': 100}
- tokamak_1species_FPCollisions_noEr_withQN: status=parity_ok jax=9.019s logged=8.255s rss=1026.6MB res={'NTHETA': 25, 'NZETA': 1, 'NX': 8, 'NXI': 100}
- tokamak_1species_FPCollisions_withEr_DKESTrajectories: status=parity_ok jax=60.348s logged=59.518s rss=3178.1MB res={'NTHETA': 25, 'NZETA': 1, 'NX': 8, 'NXI': 100}
- tokamak_1species_FPCollisions_withEr_fullTrajectories: status=parity_ok jax=73.103s logged=71.969s rss=3983.7MB res={'NTHETA': 25, 'NZETA': 1, 'NX': 8, 'NXI': 100}

Replacement source: `/Users/rogeriojorge/local/tests/sfincs_jax/tests/production_floor_cpu_bounded_xblock_2026-05-08`
- tokamak_1species_PASCollisions_noEr: status=parity_ok jax=3.073s logged=2.220s rss=923.9MB res={'NTHETA': 25, 'NZETA': 1, 'NX': 8, 'NXI': 100}
- tokamak_1species_PASCollisions_noEr_Nx1: status=parity_ok jax=2.519s logged=1.642s rss=688.3MB res={'NTHETA': 25, 'NZETA': 1, 'NX': 4, 'NXI': 100}
- tokamak_1species_PASCollisions_noEr_withQN: status=parity_ok jax=5.147s logged=4.250s rss=833.0MB res={'NTHETA': 25, 'NZETA': 1, 'NX': 8, 'NXI': 100}
- tokamak_1species_PASCollisions_withEr_fullTrajectories: status=parity_ok jax=21.970s logged=21.153s rss=2225.7MB res={'NTHETA': 25, 'NZETA': 1, 'NX': 8, 'NXI': 100}
- tokamak_2species_PASCollisions_noEr: status=parity_ok jax=2.855s logged=1.916s rss=2030.6MB res={'NTHETA': 25, 'NZETA': 1, 'NX': 8, 'NXI': 100}
- tokamak_2species_PASCollisions_withEr_fullTrajectories: status=parity_ok jax=41.632s logged=40.758s rss=4202.3MB res={'NTHETA': 25, 'NZETA': 1, 'NX': 8, 'NXI': 100}
