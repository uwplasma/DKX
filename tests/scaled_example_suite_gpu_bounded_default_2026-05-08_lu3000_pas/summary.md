# GPU frozen suite with production-floor tokamak FP/PAS refresh

Replacement source: `/tmp/sfincs_jax_gpu_pas_remote`
- tokamak_1species_PASCollisions_noEr: status=parity_ok jax=14.166s logged=12.654s rss=1241.2MB res={'NTHETA': 25, 'NZETA': 1, 'NX': 8, 'NXI': 100}
- tokamak_1species_PASCollisions_noEr_Nx1: status=parity_ok jax=28.827s logged=27.403s rss=1166.5MB res={'NTHETA': 25, 'NZETA': 1, 'NX': 4, 'NXI': 100}
- tokamak_1species_PASCollisions_noEr_withQN: status=parity_ok jax=11.444s logged=10.019s rss=1092.4MB res={'NTHETA': 25, 'NZETA': 1, 'NX': 8, 'NXI': 100}
- tokamak_1species_PASCollisions_withEr_fullTrajectories: status=parity_ok jax=46.297s logged=44.905s rss=2185.1MB res={'NTHETA': 25, 'NZETA': 1, 'NX': 8, 'NXI': 100}
- tokamak_2species_PASCollisions_noEr: status=parity_ok jax=14.717s logged=13.226s rss=1664.9MB res={'NTHETA': 25, 'NZETA': 1, 'NX': 8, 'NXI': 100}
- tokamak_2species_PASCollisions_withEr_fullTrajectories: status=parity_ok jax=92.590s logged=91.139s rss=3493.8MB res={'NTHETA': 25, 'NZETA': 1, 'NX': 8, 'NXI': 100}
