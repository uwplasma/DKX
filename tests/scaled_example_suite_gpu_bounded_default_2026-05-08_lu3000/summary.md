# GPU bounded default suite with LU3000 full-FP x-block refresh

This report starts from `tests/scaled_example_suite_gpu_bounded_default_2026-04-28` and replaces the five
`tokamak_1species_FPCollisions*` rows with the 2026-05-08 office GPU rerun at
`Ntheta=25, Nzeta=1, Nx=8, Nxi=100` after promoting the full-FP host x-block exact-LU cap to 3000.

Fresh rerun source: `/tmp/sfincs_jax_gpu_trace_20260508/tests/production_floor_bounded_remote_gpu_xblock_lu3000_2026-05-08` on `office`.

Fresh rows:
- tokamak_1species_FPCollisions_noEr: status=parity_ok jax=4.545s logged=3.230s rss=949.4MB
- tokamak_1species_FPCollisions_noEr_withPhi1InDKE: status=parity_ok jax=16.743s logged=15.433s rss=1246.8MB
- tokamak_1species_FPCollisions_noEr_withQN: status=parity_ok jax=7.372s logged=6.080s rss=1101.3MB
- tokamak_1species_FPCollisions_withEr_DKESTrajectories: status=parity_ok jax=23.799s logged=22.489s rss=1313.1MB
- tokamak_1species_FPCollisions_withEr_fullTrajectories: status=parity_ok jax=44.718s logged=43.342s rss=1380.0MB
