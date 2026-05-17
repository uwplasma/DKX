Open research lanes
===================

The current release-facing workflows are parity-clean across the documented
example suite. The lanes below are intentionally not release blockers; they are
the next algorithmic targets for larger, more memory-limited, and more strongly
parallel research workloads.

This page exists to keep those lanes concrete. Each item names the present
implementation, the remaining blocker, the next source-code touchpoints, and the
gate that must pass before the result can be promoted into public performance
claims.

True device-QI
--------------

Current state
~~~~~~~~~~~~~

The large-QI production escape hatch is closed for users who do not need
end-to-end differentiation: explicit large RHSMode=1 QI requests can enter the
non-autodiff host x-block fallback and record that choice in solver metadata.
The true differentiable/device lane remains open because the scale-0.60 hard
seed does not yet have a residual-reducing GPU-compatible preconditioner.

Relevant implementation:

- ``sfincs_jax/rhs1_qi_coarse.py`` builds deterministic QI coarse bases and
  fail-closed Galerkin corrections.
- ``sfincs_jax/rhs1_qi_block_schur.py`` provides the next standalone
  block-Schur/angular/radial coarse-preconditioner primitive. It is
  JAX-compatible, emits rank/conditioning metadata, and has a fail-closed
  true-residual probe. It remains standalone until the residual-deflated driver
  hook below produces hard-seed evidence worth expanding.
- ``sfincs_jax/rhs1_qi_deflation.py`` provides a residual-deflated,
  device-compatible two-level action. It builds a bounded preconditioned Krylov
  basis from the current residual, optionally accepts physics-informed
  block-Schur directions, and fails closed unless the true residual improves.
- ``sfincs_jax/rhs1_qi_two_level.py`` provides the first device-compatible
  local-smoother plus coarse-correction primitive for the next hard-seed probe.
- ``sfincs_jax/rhs1_qi_promotion.py`` defines the production ladder promotion
  gate: every requested seed/backend pair must converge, write output and
  solver trace artifacts, satisfy residual/observable gates, and avoid host
  fallback for a true device-QI claim.
- ``sfincs_jax/rhs1_qi_device_smoother.py`` provides the first standalone
  device-local ``S_local`` candidate: a bounded CSR-backed damped
  Jacobi/stationary smoother with fail-closed diagonal validation.
- ``sfincs_jax/rhs1_device_operator.py`` provides bounded device CSR matvec
  utilities.
- ``sfincs_jax/rhs1_qi_galerkin_policy.py`` rejects Galerkin candidates unless
  a true residual probe improves.
- ``sfincs_jax/v3_driver.py`` wires the x-block sparse-PC, device-Krylov,
  two-level-QI opt-in, residual-deflated QI opt-in, and non-autodiff host
  fallback paths.

Audit conclusion
~~~~~~~~~~~~~~~~

The checked negative artifacts rule out another storage-only or Krylov-label
change. Compact CSR factors, diagonal/one-sided factor applies, simple
Galerkin rank-32 corrections, and LGMRES/GMRES toggles either fail to reduce the
true residual or run mostly on the host. The next candidate must change the
mathematics of the preconditioner.

Next implementation
~~~~~~~~~~~~~~~~~~~

Use the new opt-in two-level RHSMode=1 QI primitive in the hard-seed solver
path:

.. math::

   M^{-1} r =
   S_\mathrm{local}^{-1} r
   + P_c A_c^{-1} R_c \left(r - A S_\mathrm{local}^{-1} r\right),

where ``S_local`` is a device-resident x/species/angular block smoother,
``R_c`` restricts to moment/constraint/global-coupling modes, ``P_c`` prolongs
back to the active vector, and

.. math::

   A_c = R_c A P_c

is a small replicated coarse operator solved with JAX dense linear algebra.

This follows the same structural idea as field-split/Schur preconditioning:
apply cheap local solves, then correct global low-dimensional coupling. It
should be tested first as a preconditioner action on physics load bases before
launching long Krylov solves.

The standalone device-Jacobi smoother is now available as the minimal
operator-reuse ``S_local`` prototype. It deliberately validates diagonal
coverage before construction and performs only JAX CSR matvecs during
application, so it can probe whether a weaker but genuinely device-resident
smoother is preferable to host x-block LU/ILU for the hard GPU seed.

The next candidate is residual-deflated rather than threshold-driven:

.. math::

   Z = \operatorname{orth}\left\{
   S_\mathrm{local}^{-1} r,\,
   \left(S_\mathrm{local}^{-1} A\right) S_\mathrm{local}^{-1} r,\,
   \ldots
   \right\},

followed by the coarse action

.. math::

   c = \arg\min_c \left\|A Z c -
   \left(r - A S_\mathrm{local}^{-1}r\right)\right\|_2 .

This is the first step toward a GCRO/deflated-GMRES style device path: the
small basis targets the current hard residual instead of relying only on fixed
geometry modes.

The current environment switch is::

   SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER=1

The residual-deflated variant is now also wired as an explicit opt-in::

   SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEFLATED_PRECONDITIONER=1

It uses the current x-block preconditioner as ``S_local^{-1}``, builds
``Z`` from the preconditioned residual Krylov sequence, can append smoothed
global load directions, and only replaces the baseline preconditioner after
the true residual probe passes. The public default remains unchanged until the
scale-0.60 CPU/GPU hard seed accepts this gate and converges.

Both QI opt-ins are fail-closed: the driver probes the true physical residual
and keeps the baseline x-block preconditioner unless the candidate action
improves the current seed by a material margin. The default margin is 5%
because a scale-0.60 CPU hard-seed preflight showed that accepting a tiny 0.7%
damped one-step decrease can make the subsequent Krylov phase substantially
worse.
For the older two-level path, the default tests only the requested damping;
damping scans are explicit-only with
``SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_DAMPINGS``.
The opt-in coarse solve can use either the projected Galerkin operator
``Q^T A Q`` or an action least-squares solve against ``A Q``. Residual-adaptive
augmentation can also prepend the current local-smoother correction and
remaining-residual directions before rank gating.

Current evidence
~~~~~~~~~~~~~~~~

- The public automatic scale-0.60 seed-3 path timed out before reaching the
  x-block hook, so it remains blocker evidence for public default promotion.
- The explicit ``xblock_sparse_pc_gmres`` scale-0.60 seed-3 CPU path converged
  in 158.4 s with residual ratio ``5.96e-3`` while the first two-level
  candidate rejected itself fail-closed after worsening the true residual from
  ``3.02e-5`` to ``2.05e-4``.
- A fresh no-two-level baseline on the same seed completed in 225.3 s and
  2723 matvecs. The final single-damping two-level opt-in path rejected the
  candidate, completed in 276.9 s and 3424 matvecs, and remained residual-clean.
  This keeps the implementation safe but not performance-promoted.
- A damped one-step scan found only a non-material 0.7% residual decrease at
  damping ``1e-2`` and increased Krylov work to 288.5 s and 3569 matvecs.
  That result is not a promotion candidate; it is the reason the default gate
  now requires at least 5% true-residual reduction and no longer runs damping
  scans unless explicitly requested.
- Action least-squares without residual augmentation still worsened the
  scale-0.60 seed-3 CPU true residual from ``3.02e-5`` to ``2.00e-4`` and
  completed residual-clean in 313.7 s / 3869 matvecs after fail-closed
  rejection.
- Residual-adaptive augmentation improved the same one-step probe to
  ``2.96e-5`` (ratio ``0.9783``) and completed residual-clean in 195.0 s /
  2295 matvecs, but the reduction is below the 5% material gate. A loosened
  2% acceptance test immediately worsened the side-probe residual ratio from
  ``8.5e4`` to ``2.4e7``, so this remains diagnostic infrastructure rather than
  a GPU-promotion candidate.
- Deeper preconditioned-residual Krylov augmentation is now available and
  records its depth and labels. The first deep probe exposed a rank-gating bug:
  raw high-norm adaptive residual vectors collapsed the retained basis from
  rank 32 to rank 2. Adaptive vectors are now normalized before
  orthonormalization. The corrected scale-0.60 seed-3 CPU probe retained
  rank 39 and stayed residual-clean, but the one-step ratio was still
  ``0.9783`` and the run took 242.6 s / 2942 matvecs. This confirms that
  residual-vector enrichment is safe diagnostic infrastructure, not the
  production device-QI closure.
- A smoothed-load field-split A/B is now wired into the same fail-closed
  two-level hook with
  ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_SMOOTHED_LOAD_BASIS=1``.
  The basis uses source/constraint, flux-surface-average, and low-angular
  loads, then applies the local x-block smoother before rank gating. The
  scale-0.60 seed-3 CPU artifact improved the one-application residual only
  from ``3.0215e-5`` to ``2.9902e-5`` (ratio ``0.9896``), below the 5%
  material gate, and the residual-clean full solve took 285.7 s / 3523
  matvecs. This is useful negative evidence: smoothed physical load directions
  alone are not enough; the remaining algorithmic step is a true
  block-Schur/moment coarse operator.
- The existing ``constraintScheme = 1`` moment-Schur wrapper is now guarded by
  an optional true-residual probe,
  ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_MOMENT_SCHUR_PROBE=1``. On the scale-0.60
  seed-3 CPU hard case the probe rejected the candidate after worsening the
  seed residual from ``3.0215e-5`` to ``2.0325e-4`` (ratio ``6.73``). The
  fail-closed fallback remained residual-clean in 288.2 s / 3502 matvecs. This
  closes the safety/observability gap for the current moment-Schur wrapper, but
  also rules it out as the missing QI residual reducer.
- A standalone block-Schur/angular/radial primitive now exists for the next
  device-QI attempt. Unit tests show residual reduction on synthetic coupled
  angular/radial systems, fail-closed rejection when no reduction is possible,
  stable rank/conditioning metadata, and JAX transform compatibility. This moves
  the implementation surface forward, but it is not a promotion artifact until
  the driver hook and scale-0.60 CPU/GPU hard-seed runs accept and converge.
- A standalone residual-deflated QI primitive now exists. Unit tests show
  reduction of a coupled slow mode beyond the local smoother, JIT-compatible
  application, fail-closed rejection without material improvement, and retention
  of extra block-Schur directions. It is wired into the x-block sparse-PC
  driver behind
  ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEFLATED_PRECONDITIONER=1`` with the same
  5% material true-residual gate. The current default for this opt-in uses a
  seed-only residual-minimizing cycle combination: build fixed-basis correction
  columns ``z_k = M_0^{-1} r_k`` with ``r_{k+1} = r_k - A z_k``, then solve the
  small least-squares problem ``min_c ||A Z c - r_0||`` and use ``Z c`` as a
  physical initial guess. This is GCRO/GMRES-like recycling evidence without
  exposing residual-dependent coefficients as a reusable linear
  preconditioner.
- The first bounded scale-0.60 seed-3 CPU hard-seed artifact for the
  residual-deflated hook is checked in as
  ``docs/_static/qi_seed_robustness_scale060_qi_deflated_seed3_cpu_2026_05_16.json``.
  The solve stayed safe and residual-clean, but the one-application probe only
  improved ``3.0215e-5 -> 2.9842e-5`` (ratio ``0.9876``), below the 5%
  material gate. The candidate therefore rejected before Krylov; no matching
  GPU promotion run is justified from this configuration.
- A stronger 2026-05-17 CPU hard-seed artifact,
  ``docs/_static/qi_seed_robustness_scale060_qi_deflated_minres8_seed3_cpu_2026_05_17.json``,
  shows the residual-minimizing cycle seed accepting:
  ``3.0215e-5 -> 2.8146e-5`` (ratio ``0.9315``), with a residual-clean final
  solve in 266.7 s / 3033 matvecs. The simpler eight-cycle stationary variant
  remained below the gate (ratio ``0.9828``), so the accepted improvement comes
  from the small residual-minimizing combination, not from blindly adding more
  local cycles.
- The matching one-GPU ``office`` artifacts preserve the same seed-level
  residual improvement, but they do not close true device-QI. Default GPU
  side-probe selection timed out at 420 s after 875 matvecs, and forced
  LGMRES-rescue timed out at 420 s after 675 matvecs. Both runs used the host
  x-block factorization path, with low GPU utilization, so the remaining GPU
  blocker is architectural: a device-local local smoother/operator-reuse path
  is required before production-resolution GPU QI can be promoted.
- A standalone device CSR Jacobi/stationary smoother now exists for that
  architectural blocker. Focused tests show CSR diagonal extraction, JIT-safe
  stationary sweeps, fail-closed rejection of missing/invalid diagonals, and
  compatibility with the existing two-level true-residual probe on a synthetic
  coupled global mode. This is implementation infrastructure only: no
  scale-0.60 CPU/GPU hard-seed artifact has been promoted from it.

Promotion gate
~~~~~~~~~~~~~~

- The scale-0.60 hard seed must write HDF5 and solver trace on CPU and one GPU.
- A one-application residual probe must reduce the current true residual by at
  least 5% and beat the existing fail-closed Galerkin probe.
- CPU/GPU observables must remain within the existing parity tolerances.
- Only after the hard seed passes should the five-seed and production-resolution
  QI ladders be launched.

Production-resolution QI ladders
--------------------------------

Current state
~~~~~~~~~~~~~

Bounded QI evidence is strong up to the scale-0.55 CPU/GPU ladders and
selected scale-0.60 probes. Production-resolution ladders remain open because
the scale-0.60 GPU hard seed is not closed by a true device algorithm.

Next implementation
~~~~~~~~~~~~~~~~~~~

After the two-level device preconditioner clears the hard seed, run:

- scale-0.60 five-seed CPU and one-GPU ladders;
- production-resolution proxy ladders at the documented floor, or the
  equivalent production manifest inputs;
- parity and residual audits against the current host fallback and Fortran v3
  reference outputs where available.

The checked promotion helper in ``sfincs_jax/rhs1_qi_promotion.py`` should be
used for every ladder artifact. A production-resolution claim is not a loose
collection of successful runs; it requires complete CPU/GPU seed coverage,
convergence, output and trace provenance, residual gates, observable gates, and
explicit rejection of accidental host fallback in true-device mode.

Promotion gate
~~~~~~~~~~~~~~

No timed-out, nonconverged, host-idle, or trace-missing run counts. The public
claim remains the non-autodiff host fallback until the production ladder exists.

Geometry-rich PAS runtime/RSS promotion
---------------------------------------

Current state
~~~~~~~~~~~~~

PAS byte budgets and fail-closed matrix-free gates prevent unsafe promotions.
However, the real geometry4 and HSX probes were residual-clean but did not
improve runtime or peak memory, so no default promotion is justified.

Relevant implementation:

- ``sfincs_jax/rhs1_pas_matrixfree.py`` contains guarded matrix-free correction
  helpers, candidate-size preflights, and ``PasRuntimeChunkPlan`` for deriving
  bounded reduction chunks from configured byte budgets.
- ``sfincs_jax/rhs1_pas_policy.py`` contains PAS applicability and memory gates.
- ``scripts/benchmark_pas_tz_memory_fallback.py`` records promotion/rejection
  evidence.

Next implementation
~~~~~~~~~~~~~~~~~~~

Promote the new reduction chunk planner from helper-level memory safety into a
measured solve path: use it to bound PAS-heavy diagnostic/reduction live sets,
then benchmark geometry4, HSX, and geometry11 on CPU and GPU. A later algorithm
should still replace dense candidate update materialization with fixed-shape
streamed/chunked correction actions over pitch-angle/angular blocks.

Promotion gate
~~~~~~~~~~~~~~

Geometry4, HSX, and geometry11 artifacts must be residual-clean and improve
warm runtime by at least 20% or active/RSS memory by at least 25%. A candidate
that improves one case but regresses another remains opt-in only.

Single-case multi-GPU strong scaling
------------------------------------

Current state
~~~~~~~~~~~~~

Transport-worker and scan/case-level parallelism are release-facing because
they have deterministic, audited throughput evidence. Single-case multi-GPU
RHSMode=1 sharding remains experimental because current sharded solves are
compile/setup/synchronization dominated.

Relevant implementation:

- ``examples/performance/benchmark_sharded_solve_scaling.py`` and
  ``examples/performance/benchmark_sharded_matvec_scaling.py`` generate
  bounded single-case scaling evidence.
- ``sfincs_jax/transport_parallel_policy.py`` prevents cold or malformed
  scaling payloads from becoming release claims.
- ``sfincs_jax/transport_parallel_sharding.py`` records pure single-case
  sharded-solve plans, caps requested devices to available work, reports
  workload balance, estimates setup/communication amortization, and fail-closes
  release scaling claims for experimental single-case sharding.
- ``sfincs_jax/v3_system.py`` contains the sharded matrix-free operator path.

Next implementation
~~~~~~~~~~~~~~~~~~~

Move from process-per-sample benchmarking to compiled operator reuse:

- shard the state vector over theta/zeta slabs with JAX ``NamedSharding`` or
  ``shard_map``;
- keep the coarse/Schur problem replicated so global reductions are small;
- avoid host collectives inside every Krylov step;
- use the persistent compilation cache on a shared filesystem for repeated
  device-count runs.
- promote single-case sharding only when the amortization model predicts enough
  per-device work to dominate setup, halo exchange, and Krylov collectives.

Promotion gate
~~~~~~~~~~~~~~

The warm two-GPU single-case solve must beat the warm one-GPU solve by at least
1.15x, per-device memory must not increase, and the residual/output comparison
must pass. The amortization gate must also pass with predicted speedup and
efficiency above the configured thresholds and communication below the configured
fraction. Until then, public docs should continue to recommend one GPU per case
or transport RHS.

External numerical anchors
--------------------------

The planned algorithms are anchored in established sparse-solver and JAX
infrastructure:

- `PETSc PCFIELDSPLIT <https://petsc.org/main/manualpages/PC/PCFIELDSPLIT/>`_
  documents block and Schur-complement preconditioners, including local sub-KSP
  solvers and explicit Schur actions.
- Communication-avoiding GMRES and domain-decomposition preconditioners provide
  the scaling model for this lane: fewer global reductions help only when the
  local subdomain work and coarse correction amortize communication.
- The `ICNTS monoenergetic coefficient benchmark
  <https://www.ornl.gov/publication/benchmarking-mono-energetic-transport-coefficients-results-international-collaboration>`_
  remains the relevant physics validation anchor for radial, parallel, and
  bootstrap-current coefficients.
- `MONKES <https://arxiv.org/abs/2312.12248>`_ is the algorithmic comparison
  point for fast monoenergetic stellarator coefficients: spectral
  discretization plus block sparsity is the performance target to keep in mind
  for future QI/PAS paths.
- `Landreman & Catto's quasi-isodynamic theory paper
  <https://arxiv.org/abs/1011.5184>`_ anchors QI-specific flow, radial
  electric-field, and bootstrap-current expectations for hard-seed validation.
- `Landreman & Ernst's velocity-space discretization paper
  <https://arxiv.org/abs/1210.5289>`_ anchors the Fokker-Planck/velocity-grid
  pieces that any production-resolution QI ladder must preserve.
- The SFINCS v3 manual and paper explain the existing performance baseline:
  GMRES/KSP with a cheaper LU-factorized preconditioner formed by dropping speed
  and species coupling.
- JAX ``jax.Array`` and
  `NamedSharding <https://docs.jax.dev/en/latest/jax.sharding.html>`_ provide
  the current sharded-array model; `shard_map
  <https://docs.jax.dev/en/latest/notebooks/shard_map.html>`_ gives explicit
  per-shard code when automatic sharding is not enough.
- The `JAX persistent compilation cache
  <https://docs.jax.dev/en/latest/persistent_compilation_cache.html>`_ is
  required for fair warm scaling studies.
- `jax.experimental.sparse
  <https://docs.jax.dev/en/latest/jax.experimental.sparse.html>`_ remains useful
  for compatibility tests, but the official documentation marks these sparse
  formats as experimental, so they should not be the main performance-critical
  backend by default.
- `Lineax <https://docs.kidger.site/lineax/>`_ remains an optional benchmark
  lane until real SFINCS operators pass residual/error gates; it should not
  become a hard production dependency just because synthetic linear solves look
  good.

Validation order
----------------

1. Add pure unit tests for each new policy/helper before wiring it into
   ``v3_driver.py``.
2. Run focused CPU tests for QI, PAS, sharding, and release-gate policy.
3. Run bounded CPU hard-seed probes.
4. Run one-GPU hard-seed probes on ``office`` only after the CPU preflight
   reduces the true residual.
5. Launch five-seed and production-resolution ladders only after the hard seed
   is closed.
6. Regenerate README/docs plots only after checked artifacts change public
   claims.
