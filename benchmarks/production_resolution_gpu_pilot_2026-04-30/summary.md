# Scaled Example Suite Summary

- Cases: 4
- Practical status counts: max_attempts=1, parity_mismatch=2, parity_ok=1
- Strict status counts: max_attempts=1, parity_mismatch=2, parity_ok=1

## Runtime offenders (absolute JAX time)

- ntx_sfincs_jax_rhsmode1_profile_current_profiling_cpu_17x21x12_deck_finite_beta_qa_pressure_current_rho_0p142857_nu_n_0p00831565: jax=605.619s fortran=79.076s ratio=7.659 res={'NTHETA': 17, 'NZETA': 21, 'NX': 5, 'NXI': 12} status=parity_mismatch
- ntx_outputs_owned_finite_beta_sfincs_jax_profile_current_audit_finite_beta_qa_pressure_current_rho_0p142857_nu_n_0p00831565: jax=43.052s fortran=76.454s ratio=0.563 res={'NTHETA': 13, 'NZETA': 15, 'NX': 5, 'NXI': 8} status=parity_mismatch
- tokamak_1species_PASCollisions_noEr_Nx1: jax=4.897s fortran=75.226s ratio=0.065 res={'NTHETA': 21, 'NZETA': 1, 'NX': 1, 'NXI': 31} status=parity_ok

## Runtime offenders (JAX/Fortran ratio)

- ntx_sfincs_jax_rhsmode1_profile_current_profiling_cpu_17x21x12_deck_finite_beta_qa_pressure_current_rho_0p142857_nu_n_0p00831565: ratio=7.638 jax=605.619s fortran=79.076s res={'NTHETA': 17, 'NZETA': 21, 'NX': 5, 'NXI': 12} status=parity_mismatch
- ntx_outputs_owned_finite_beta_sfincs_jax_profile_current_audit_finite_beta_qa_pressure_current_rho_0p142857_nu_n_0p00831565: ratio=0.539 jax=43.052s fortran=76.454s res={'NTHETA': 13, 'NZETA': 15, 'NX': 5, 'NXI': 8} status=parity_mismatch
- tokamak_1species_PASCollisions_noEr_Nx1: ratio=0.044 jax=4.897s fortran=75.226s res={'NTHETA': 21, 'NZETA': 1, 'NX': 1, 'NXI': 31} status=parity_ok

## Memory offenders (absolute JAX RSS)

- ntx_sfincs_jax_rhsmode1_profile_current_profiling_cpu_17x21x12_deck_finite_beta_qa_pressure_current_rho_0p142857_nu_n_0p00831565: jax=5506.5MB fortran=541.4MB ratio=10.171 res={'NTHETA': 17, 'NZETA': 21, 'NX': 5, 'NXI': 12} status=parity_mismatch
- ntx_outputs_owned_finite_beta_sfincs_jax_profile_current_audit_finite_beta_qa_pressure_current_rho_0p142857_nu_n_0p00831565: jax=2909.3MB fortran=238.8MB ratio=12.185 res={'NTHETA': 13, 'NZETA': 15, 'NX': 5, 'NXI': 8} status=parity_mismatch
- tokamak_1species_PASCollisions_noEr_Nx1: jax=1014.5MB fortran=99.7MB ratio=10.178 res={'NTHETA': 21, 'NZETA': 1, 'NX': 1, 'NXI': 31} status=parity_ok

## Memory offenders (JAX/Fortran ratio)

- ntx_outputs_owned_finite_beta_sfincs_jax_profile_current_audit_finite_beta_qa_pressure_current_rho_0p142857_nu_n_0p00831565: ratio=12.185 jax=2909.3MB fortran=238.8MB res={'NTHETA': 13, 'NZETA': 15, 'NX': 5, 'NXI': 8} status=parity_mismatch
- tokamak_1species_PASCollisions_noEr_Nx1: ratio=10.178 jax=1014.5MB fortran=99.7MB res={'NTHETA': 21, 'NZETA': 1, 'NX': 1, 'NXI': 31} status=parity_ok
- ntx_sfincs_jax_rhsmode1_profile_current_profiling_cpu_17x21x12_deck_finite_beta_qa_pressure_current_rho_0p142857_nu_n_0p00831565: ratio=10.171 jax=5506.5MB fortran=541.4MB res={'NTHETA': 17, 'NZETA': 21, 'NX': 5, 'NXI': 12} status=parity_mismatch

## Mismatches

- ntx_sfincs_jax_rhsmode1_profile_current_profiling_cpu_17x21x12_deck_finite_beta_qa_pressure_current_rho_0p142857_nu_n_0p00831565: practical=34/193 strict=34/193 solver=12 physics=0 sample=FSABFlow,FSABFlow_vs_x,FSABVelocityUsingFSADensity,FSABVelocityUsingFSADensityOverB0
- ntx_outputs_owned_finite_beta_sfincs_jax_profile_current_audit_finite_beta_qa_pressure_current_rho_0p142857_nu_n_0p00831565: practical=30/193 strict=30/193 solver=12 physics=0 sample=FSABFlow,FSABFlow_vs_x,FSABVelocityUsingFSADensity,FSABVelocityUsingFSADensityOverB0

## Print parity gaps

- None

## Failures and blockers

- ntx_grid_35_43_48_finite_beta_qa_pressure_current_rho_0p142857_nuPrime_0p01_EStar_0: status=max_attempts blocker=solver branch mismatch attempts=1 reductions=1 note=Reached max attempts while reducing resolution. Last failure: JAX timeout; reduced largest axis.
