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
seed does not yet have a residual-reducing GPU-compatible preconditioner. The
CPU hard seed now has a bounded orchestration path that writes HDF5 and solver
trace output at ``15 x 31 x 60 x 5`` without entering the old strong-preconditioner
or SciPy-rescue time sinks, but the QI-device correction itself still rejects
that seed by the true-residual gate.

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
  device-local ``S_local`` candidate: a bounded CSR-backed Jacobi smoother with
  fail-closed diagonal validation, an opt-in residual-minimizing step policy, and
  a true-residual seed probe.
- ``sfincs_jax/rhs1_qi_device_preconditioner.py`` provides the first
  production-shaped device-QI state. It combines a device Jacobi smoother with
  a rank-gated coarse basis when device CSR is available, and also provides a
  matrix-free coarse-only path that builds just ``A Q`` by JAX matvec probes
  when full CSR materialization is too expensive. Both paths expose
  setup/apply/probe metadata and keep the timed apply path free of SciPy, host
  LU/ILU, and Python callbacks.
- ``sfincs_jax/rhs1_device_operator.py`` provides bounded device CSR matvec
  utilities.
- ``sfincs_jax/rhs1_qi_galerkin_policy.py`` rejects Galerkin candidates unless
  a true residual probe improves.
- ``sfincs_jax/v3_driver.py`` wires the x-block sparse-PC, device-Krylov,
  two-level-QI opt-in, residual-deflated QI opt-in, device-QI field-split opt-in,
  early matrix-free QI probe, bounded post-xblock acceptance, and non-autodiff
  host fallback paths.

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

The standalone device-QI preconditioner state is now available on top of that
smoother. It implements the same local-plus-coarse equation above with ``A_c``
assembled by device matvec probes. When full device CSR is rejected by the memory
preflight, a second explicit fallback can build only the coarse ``A Q`` matrix
from the matrix-free JAX operator and use the result as a seed-only correction.
Focused tests cover JIT, gradient propagation with respect to the residual,
metadata hygiene, local-only fallback, matrix-free coarse-only setup, and
fail-closed true-residual probing on synthetic global-coupling systems. It is
wired through the driver behind the explicit opt-in::

     SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER=1

The full device-CSR path requires an assembled device CSR operator. The
matrix-free seed-only fallback is separately enabled with::

     SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE=1

The matrix-free path now has an optional residual enrichment that adds bounded
correction-space Krylov vectors without materializing CSR::

     SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_ENRICHMENT=1
     SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_ENRICHMENT_DEPTH=2

This builds ``orth([Q, r, A r, A^2 r, ...])`` and still accepts the candidate
only if a true-residual probe improves by the configured margin. Seed-only
hard-seed experiments may also move the probe ahead of the expensive strong
preconditioner stage with::

     SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_EARLY=1
     SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_SKIP_STRONG=1

The seed-only probe can also run a short bounded sequence of residual-checked
corrections::

     SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_CYCLES=4
     SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_STEP_POLICY=residual_minimizing

The default remains one fixed cycle. With ``STEP_POLICY=residual_minimizing``,
each correction direction ``d`` is scaled by the scalar ``alpha`` that minimizes
``||r - alpha A d||_2`` before the true-residual gate is evaluated. Each
additional cycle recomputes the true residual and stops immediately if the
candidate is non-finite or fails to reduce ``||r||_2`` by the configured
material margin, so the knob is safe for GPU hard-seed experiments without
turning the QI action into an unbounded Krylov preconditioner.

The matrix-free path also has an opt-in recycle enrichment::

     SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RECYCLE_ENRICHMENT=1
     SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RECYCLE_CYCLES=2

This appends the residual left after the current coarse correction,
``r - A Q (A Q)^+ r``, as a new rank-gated candidate direction. It is a
bounded GCRO-style seed construction, not a production Krylov replacement.

The same matrix-free path now has an opt-in residual-polynomial local smoother::

     SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_LOCAL_SMOOTHER=matrix_free_minres
     SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE_SMOOTHER_SWEEPS=2

This applies a fixed number of pure-JAX sweeps using the current residual as the
local direction and the scalar that minimizes ``||r - alpha A r||_2``. The
operation is bounded, device-compatible, and still guarded by the same
true-residual acceptance gate.

The stronger block-projected variant is selected with::

     SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_LOCAL_SMOOTHER=matrix_free_block_minres
     SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE_BLOCK_SMOOTHER_MAX_GROUPS=16

It uses the QI x/species block layout to form residual pieces ``D`` and solves
``min_c ||r - A D c||_2`` as a bounded local action. This is still matrix-free
and device-compatible, but it is a stronger block/angular/radial correction than
the scalar residual-polynomial smoother.
The default grouping keeps contiguous x/species blocks. A stronger experimental
hybrid grouping is available with::

     SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE_BLOCK_SMOOTHER_GROUPING=block_x_species

This augments the local projected space with radial-x and species aggregate
residual directions, giving the least-squares correction a small global-coupling
handle while keeping the matvec count bounded by ``MAX_GROUPS``.

The latest CPU hard-seed evidence uses this early hook plus the guarded
``SFINCS_JAX_RHSMODE1_SKIP_GLOBAL_SPARSE_AFTER_XBLOCK=1`` path. It writes output
with residual ``7.80e-10`` and acceptance criterion
``post_xblock_abs_floor``. This is a bounded CPU orchestration result, not a true
device-QI promotion, because the QI-device probe still reports
``residual_not_reduced``. Matrix-free QI is allowed even when
``precondition_side=none``; only
installation as a Krylov preconditioner is blocked in that case.

The matching one-GPU hard-seed pass on ``office`` no longer reproduces the old
CUDA illegal-address failure. The rank-26/27 matrix-free QI seed reduces
residual from ``2.26e-6`` to ``1.42e-6`` in about ``73 s`` and then fails closed
because the result is still not accurate enough to write production-sized
RHSMode=1 output. Fixed-step and residual-minimizing scalar variants give the
same GPU residual on this seed, so scalar line search is not the missing
physics. The recycle-enriched rank-28 GPU seed improves the same residual to
``1.06e-6`` in about ``74 s`` without a CUDA failure. Adding the matrix-free
minimum-residual local smoother improves the seed further to ``7.42e-7`` with
two sweeps and ``5.17e-7`` with eight sweeps, both in about ``79 s`` with the
same rank, but still misses the ``3.02e-11`` output target by about ``1.7e4``.
A forced non-autodiff host
x-block fallback on the GPU host timed out after ``600 s`` and ``975`` matvecs
in the earlier run; a later explicit QI-as-Krylov-preconditioner attempt also
triggered host fallback, disabled the device-QI preconditioner, and timed out
after ``360 s`` / ``675`` matvecs. The driver now disables only the automatic
host fallback when the user explicitly requests the matrix-free QI-device
preconditioner as the Krylov preconditioner. A bounded GPU probe of that
true-device route ran ``fgmres_jax`` for ``803`` matvecs in ``278 s`` and failed
closed at ``2.83e-5``. This proves the replacement path is reachable and
CUDA-safe, but also shows that the current approximate QI action is not a strong
enough Krylov preconditioner. Host-fallback routes are not promoted as GPU
production fallbacks.
The rank-48/depth-2 experiment timed out before useful metadata and should not
be repeated as the next promotion attempt.
CPU direction probes show that scalar minimization alone is not enough for this
hard seed: rank 12 reduces only ``1.736775e-3`` to ``1.735797e-3``, while rank
27 reduces to ``1.725418e-3``. Both are below the 1% material-improvement gate,
so the next implementation should add a physically stronger local/global
smoother or recycle space rather than tune only scalar damping.
The first block-projected local smoother CPU run at the same scale-0.60 seed
reduces the QI-device seed residual from ``1.736775e-3`` to ``1.649351e-3``
before the existing x-block sparse rescue writes accepted output in ``296.6 s``.
A wider ``max_groups=32`` run keeps the same residual reduction and lowers the
local wall time to ``285.0 s``. This is the first material CPU seed improvement
from a true matrix-free local action, but it is not a true device-QI closure
until the same route passes on GPU and writes output without relying on the
later sparse x-block rescue.
The ``block_x_species`` hybrid grouping also completed on CPU in ``289.5 s``
with the same residual ratio. Since it is slightly slower than contiguous
``max_groups=32`` on this seed, it remains an opt-in GPU/research probe rather
than the preferred CPU setting.

Both paths fail closed when host fallback is active or when the true-residual
probe does not improve. The pre-enrichment scale-0.60 seed-3 CPU artifact
``docs/_static/qi_seed_robustness_scale060_qi_device_matrixfree_seed3_cpu_2026_05_19.json``
shows the matrix-free fallback avoiding full CSR, building a rank-9 coarse
operator in about ``0.57 s``, then rejecting itself because the residual ratio
was ``0.9999998`` rather than a material improvement. It remains unpromoted
until real scale-0.60 CPU/GPU hard-seed artifacts pass the gate below. A
post-enrichment bounded rerun showed that the remaining active blocker is path
ordering in the large active-DOF RHSMode=1 FP branch: the expensive host x-block
rescue can dominate before a bounded QI-device probe produces evidence.

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
- A standalone device CSR Jacobi smoother now exists for that architectural
  blocker. Focused tests show CSR diagonal extraction, JIT-safe stationary
  sweeps, fail-closed rejection of missing/invalid diagonals, and compatibility
  with the existing two-level true-residual probe on a synthetic coupled global
  mode. The residual-minimizing device policy now turns a deliberately bad fixed
  Jacobi step into a measured local reduction on a coupled triangular probe:
  fixed Jacobi worsens the residual norm from ``1.0`` to ``2.0``, while the
  residual-minimizing seed reduces it to ``0.8944`` and rejects itself when a
  stricter 20% material-improvement gate is requested. This is implementation
  infrastructure only: no scale-0.60 CPU/GPU hard-seed artifact has been promoted
  from it.
- A standalone device-QI field-split state now exists and is wired as an explicit
  x-block sparse-PC opt-in. Focused tests show no-host-fallback metadata, JIT
  compatibility, differentiability with respect to the residual, local-only
  fallback behavior, fail-closed residual probing, coarse-shape validation,
  driver metadata for accepted opt-in probes, and fail-closed driver metadata
  when no assembled device CSR operator is available. This advances the
  implementation surface, but it still needs hard-seed evidence before it can be
  used for public claims.

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

The 2026-05-17 clean ``office`` run closed the last unknown in this lane as a
measured blocker, not a promotion. On two RTX A4000 GPUs, a fresh checkout at
``8e34341`` completed the one-GPU hot solve in ``4.047 s`` but timed out the
two-GPU sharded child at the ``300 s`` cap. The checked artifact
``docs/_static/transport_sharded_solve_gpu_1v2_failclosed_2026_05_17.json``
keeps ``release_scaling_claim=false``, marks ``devices=2`` as timed out, and
skips the deterministic-output probe after timing failure. This prevents a
stale dry-run or incomplete trace from being mistaken for strong-scaling
evidence.

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
- ``docs/_static/transport_sharded_solve_gpu_1v2_failclosed_2026_05_17.json``
  is the current negative 1-vs-2 GPU evidence artifact.

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
