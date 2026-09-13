# Changelog

## Unreleased — candidate v2.4.0-rc1

Draft for the integrated #169–#188 stack, #191 and the reconciled #190 plan.
No release or candidate tag has been created by this change. Release remains
deferred until the maintainer's important-goal and verification requirements are met.

### Correctness and differentiation

- Add the opt-in `SfincsMatrixThreshold` parity switch, which reproduces SFINCS
  v3's `1d-12` matrix sparsification on the Fokker-Planck operator. It accounts
  for a 12-19% bootstrap-current gap on a hot-electron HSX-like deck; the
  default keeps every entry.

- Add opt-in bounded GMRES reuse to the host ambipolar root, with one cold retry
  after failed admission and independent final acceptance. Keep retry policy out
  of root differentiation.
- Correct the ambipolar teaching case to quasineutral analytic W7-X with full-FP
  collisions. JIT the geometry-optimization value/gradient, check its original
  kinetic residual and retain a three-step finite-difference window.

- Keep discrete pitch layouts static under JIT; refresh full-FP density kernels
  and opt-in temperature-dependent coefficients for prepared profile derivatives
  (#173, #174, #178, #186).
- Prepare immutable native profile/field scans with per-input original-equation
  status and complete kinetic-state recovery. Differentiate supported regular
  ambipolar roots with original primal/transpose admission (#182, #184, #187, #188).
- Reject invalid kinetic states, roots and adjoint references; qualify physical
  profile sensitivities with independent cold solves and Taylor checks
  (#180, #184, #188).

### Execution

- Build the transposed coarse preconditioner on its first application, and
  synchronize coarse builds on their factors instead of a discarded zero-vector
  application. Non-differentiable solves no longer pay two transposed coarse
  applications per build; the preconditioner maps are unchanged.
- Factor the dense coarse preconditioner straight from its pinned row generator
  inside one compiled computation, so no band is materialized before
  elimination, and store only the rows each subsystem's `Nxi_for_x` keeps. The
  elimination and both substitution sweeps run one Legendre row at a time,
  batched over the subsystems active at that row, so the kernels launched per
  application scale with the longest chain. The truncated rows are an uncoupled
  `(1 + floor) I` and are applied as such. The map is unchanged: the f-block
  inverse agrees with the reusable Schur-LU route to rounding, GCROT iteration
  counts are identical, and the band guard is left as it was, conservative for
  ramped decks (`tests/test_coarse_ragged_chains.py`).

- Preserve independent-device batch sharding through JIT and gradients, including
  uneven batches (#179). Each complete system still resides on one device.
- Reuse generated SOLVAX Schur factors within supported PAS solve executions for
  multiple right-hand sides, transpose solves and refinement, subject to memory
  policy (#188). Persistent reuse across changed operators is still future work.

### Benchmarking and development

- Supervise benchmark process groups and cancellation; reject invalid reference
  outputs and stale resumed results; retain original-equation checks and evidence
  (#170, #183, #185).
- Preserve requested PETSc options and observed backends; make source inventories
  reproducible; correct sparse reference matrix interpretation (#169, #176, #177).
- Preserve the tested package tree when GitHub merge refs move (#181), and balance
  thirteen coverage shards using measured test durations (#191).
- Adopt one authoritative figure-first plan, a concise workflow-oriented README,
  grouped documentation and citation metadata (#189, #190). Establish decision
  records and an experiment template in Phase 0.

These changes do not certify converged production performance, general persistent
restart reuse, full native Phi1 support or complete equilibrium-boundary optimization.
Historical measurements and their limits remain in the performance documentation.
