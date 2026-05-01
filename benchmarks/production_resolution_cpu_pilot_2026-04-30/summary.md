# Scaled Example Suite Summary

- Cases: 5
- Practical status counts: max_attempts=2, parity_mismatch=2, parity_ok=1
- Strict status counts: max_attempts=2, parity_mismatch=2, parity_ok=1

## Runtime offenders (absolute JAX time)

- ntx_outputs_owned_finite_beta_sfincs_jax_profile_current_audit_finite_beta_qa_pressure_current_rho_0p142857_nu_n_0p00831565: jax=5.915s fortran=76.454s ratio=0.077 res={'NTHETA': 13, 'NZETA': 15, 'NX': 5, 'NXI': 8} status=parity_mismatch
- tokamak_1species_PASCollisions_noEr_Nx1: jax=2.685s fortran=75.226s ratio=0.036 res={'NTHETA': 21, 'NZETA': 1, 'NX': 1, 'NXI': 31} status=parity_ok
- ntx_grid_35_43_48_finite_beta_qa_pressure_current_rho_0p142857_nuPrime_0p01_EStar_0: jax=2.370s fortran=90.750s ratio=0.026 res={'NTHETA': 35, 'NZETA': 43, 'NX': 1, 'NXI': 48} status=parity_mismatch

## Runtime offenders (JAX/Fortran ratio)

- ntx_outputs_owned_finite_beta_sfincs_jax_profile_current_audit_finite_beta_qa_pressure_current_rho_0p142857_nu_n_0p00831565: ratio=0.056 jax=5.915s fortran=76.454s res={'NTHETA': 13, 'NZETA': 15, 'NX': 5, 'NXI': 8} status=parity_mismatch
- tokamak_1species_PASCollisions_noEr_Nx1: ratio=0.021 jax=2.685s fortran=75.226s res={'NTHETA': 21, 'NZETA': 1, 'NX': 1, 'NXI': 31} status=parity_ok
- ntx_grid_35_43_48_finite_beta_qa_pressure_current_rho_0p142857_nuPrime_0p01_EStar_0: ratio=0.015 jax=2.370s fortran=90.750s res={'NTHETA': 35, 'NZETA': 43, 'NX': 1, 'NXI': 48} status=parity_mismatch

## Memory offenders (absolute JAX RSS)

- ntx_outputs_owned_finite_beta_sfincs_jax_profile_current_audit_finite_beta_qa_pressure_current_rho_0p142857_nu_n_0p00831565: jax=3726.0MB fortran=238.8MB ratio=15.605 res={'NTHETA': 13, 'NZETA': 15, 'NX': 5, 'NXI': 8} status=parity_mismatch
- ntx_grid_35_43_48_finite_beta_qa_pressure_current_rho_0p142857_nuPrime_0p01_EStar_0: jax=616.6MB fortran=1719.4MB ratio=0.359 res={'NTHETA': 35, 'NZETA': 43, 'NX': 1, 'NXI': 48} status=parity_mismatch
- tokamak_1species_PASCollisions_noEr_Nx1: jax=484.2MB fortran=99.7MB ratio=4.858 res={'NTHETA': 21, 'NZETA': 1, 'NX': 1, 'NXI': 31} status=parity_ok

## Memory offenders (JAX/Fortran ratio)

- ntx_outputs_owned_finite_beta_sfincs_jax_profile_current_audit_finite_beta_qa_pressure_current_rho_0p142857_nu_n_0p00831565: ratio=15.605 jax=3726.0MB fortran=238.8MB res={'NTHETA': 13, 'NZETA': 15, 'NX': 5, 'NXI': 8} status=parity_mismatch
- tokamak_1species_PASCollisions_noEr_Nx1: ratio=4.858 jax=484.2MB fortran=99.7MB res={'NTHETA': 21, 'NZETA': 1, 'NX': 1, 'NXI': 31} status=parity_ok
- ntx_grid_35_43_48_finite_beta_qa_pressure_current_rho_0p142857_nuPrime_0p01_EStar_0: ratio=0.359 jax=616.6MB fortran=1719.4MB res={'NTHETA': 35, 'NZETA': 43, 'NX': 1, 'NXI': 48} status=parity_mismatch

## Mismatches

- ntx_grid_35_43_48_finite_beta_qa_pressure_current_rho_0p142857_nuPrime_0p01_EStar_0: practical=33/193 strict=33/193 solver=12 physics=0 sample=FSABFlow,FSABFlow_vs_x,FSABVelocityUsingFSADensity,FSABVelocityUsingFSADensityOverB0
- ntx_outputs_owned_finite_beta_sfincs_jax_profile_current_audit_finite_beta_qa_pressure_current_rho_0p142857_nu_n_0p00831565: practical=30/193 strict=30/193 solver=12 physics=0 sample=FSABFlow,FSABFlow_vs_x,FSABVelocityUsingFSADensity,FSABVelocityUsingFSADensityOverB0

## Print parity gaps

- None

## Failures and blockers

- ntx_sfincs_jax_rhsmode1_profile_current_profiling_cpu_17x21x12_deck_finite_beta_qa_pressure_current_rho_0p142857_nu_n_0p00831565: status=max_attempts blocker=solver branch mismatch attempts=1 reductions=1 note=Reached max attempts while reducing resolution. Last failure: JAX timeout; reduced largest axis.
- tokamak_1species_FPCollisions_noEr: status=max_attempts blocker=solver branch mismatch attempts=1 reductions=1 note=Reached max attempts while reducing resolution. Last failure: Fortran error: RuntimeError: Fortran failed rc=15.
  1769 KSP Residual norm 6.496252467043e-09
  1770 KSP Residual norm 6.399992312916e-09
  1771 KSP Residual norm 6.370184255848e-09
  1772 KSP Residual norm 6.287640345993e-09
  1773 KSP Residual norm 6.263583940533e-09
  1774 KSP Residual norm 6.181159941442e-09
  1775 KSP Residual norm 6.158749102894e-09
  1776 KSP Residual norm 6.074366700301e-09
  1777 KSP Residual norm 6.044502986937e-09
  1778 KSP Residual norm 5.985343920177e-09
  1779 KSP Residual norm 5.945013689446e-09
  1780 KSP Residual norm 5.912129645902e-09
  1781 KSP Residual norm 5.850935825215e-09
  1782 KSP Residual norm 5.828493610395e-09
  1783 KSP Residual norm 5.748407994039e-09
  1784 KSP Residual norm 5.726087773414e-09
  1785 KSP Residual norm 5.624851515314e-09
  1786 KSP Residual norm 5.595210503946e-09
  1787 KSP Residual norm 5.502242660586e-09
  1788 KSP Residual norm 5.468473252971e-09
  1789 KSP Residual norm 5.393382964177e-09
  1790 KSP Residual norm 5.350119510999e-09
  1791 KSP Residual norm 5.291757548643e-09
  1792 KSP Residual norm 5.212211785520e-09
  1793 KSP Residual norm 5.161160098531e-09
  1794 KSP Residual norm 5.063353317028e-09
  1795 KSP Residual norm 5.019299643745e-09
  1796 KSP Residual norm 4.932549665272e-09
  1797 KSP Residual norm 4.891191132919e-09
  1798 KSP Residual norm 4.801901008989e-09
  1799 KSP Residual norm 4.752600639613e-09
  1800 KSP Residual norm 4.670165572574e-09
  1801 KSP Residual norm 4.611850684721e-09
  1802 KSP Residual norm 4.530281555628e-09
  1803 KSP Residual norm 4.463236619998e-09
  1804 KSP Residual norm 4.376924633776e-09
  1805 KSP Residual norm 4.293240583561e-09
  1806 KSP Residual norm 4.223016561707e-09
  1807 KSP Residual norm 4.156317214188e-09
  1808 KSP Residual norm 4.094220001120e-09
  1809 KSP Residual norm 4.020052474879e-09
  1810 KSP Residual norm 3.947797175297e-09
  1811 KSP Residual norm 3.855707461470e-09
  1812 KSP Residual norm 3.789156169771e-09
  1813 KSP Residual norm 3.685803075475e-09
  1814 KSP Residual norm 3.630245557035e-09
  1815 KSP Residual norm 3.552018117676e-09
  1816 KSP Residual norm 3.508402805535e-09
  1817 KSP Residual norm 3.443006382081e-09
  1818 KSP Residual norm 3.392952437890e-09
  1819 KSP Residual norm 3.318716414898e-09
  1820 KSP Residual norm 3.248571585507e-09
  1821 KSP Residual norm 3.169686119728e-09
  1822 KSP Residual norm 3.098515435538e-09
[0]PETSC ERROR: ------------------------------------------------------------------------
[0]PETSC ERROR: Caught signal number 11 SEGV: Segmentation Violation, probably memory access out of range
[0]PETSC ERROR: Try option -start_in_debugger or -on_error_attach_debugger
[0]PETSC ERROR: or see https://petsc.org/release/faq/#valgrind and https://petsc.org/release/faq/
[0]PETSC ERROR: configure using --with-debugging=yes, recompile, link, and run
[0]PETSC ERROR: to get more information on the crash.
[0]PETSC ERROR: Run with -malloc_debug to check if memory corruption is causing the crash.
Abort(59) on node 0 (rank 0 in comm 0): application called MPI_Abort(MPI_COMM_WORLD, 59) - process 0
        2.47 real         2.35 user         0.09 sys
           172343296  maximum resident set size
                   0  average shared memory size
                   0  average unshared data size
                   0  average unshared stack size
               10924  page reclaims
                  40  page faults
                   0  swaps
                   0  block input operations
                   0  block output operations
                   4  messages sent
                   4  messages received
                   1  signals received
                   1  voluntary context switches
                5312  involuntary context switches
         14057350575  instructions retired
          8572111526  cycles elapsed
           147459648  peak memory footprint
