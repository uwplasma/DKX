Open research lanes
===================

The documented release-facing workflows are parity-clean across the example
suite. The lanes below are intentionally not release blockers; they are the
algorithmic targets for larger, more memory-limited, and more strongly parallel
research workloads.

Release decision on the native production-preconditioner lane
-------------------------------------------------------------

The lower-memory native sparse-factor/preconditioner campaign for the largest
geometry-rich RHSMode=2/3 and full-grid QA/QH RHSMode=1 cases is deferred as
optimization work. The code now contains tested opt-in infrastructure for
Fortran-reduced direct ``Pmat`` emission, symbolic ordering metadata,
superblock/nested-dissection factors, BLR/HSS-style separator updates, and
strict setup-time true-residual admission. Those pieces are useful research
controls, but production ``geom11`` probes still reject the native
nested-dissection/BLR path on setup time before admission, and full-grid QA/QH
still rely on residual-clean active-LU fallback when the lower-memory native
candidate fails its residual gate.

This means the release scope is closed on correctness and documentation:
promoted defaults keep the residual-clean/Fortran-parity paths, while the
native lower-memory replacement remains nonblocking until a future algorithm
passes the same residual, runtime, and memory gates. Future work should avoid
more smoother/restart tuning and instead target a materially different
separator hierarchy/orderer, matrix-free admitted Schur approximation, or
compiled/JAX-native sparse factorization that avoids Python-level recursive
setup on large fronts.

This page exists to keep those lanes concrete. Each item names the present
implementation, the remaining blocker, the next source-code touchpoints, and the
gate that must pass before the result can be promoted into public performance
claims.

True device-QI
--------------

Evidence State
~~~~~~~~~~~~~~

The large-QI production escape hatch is closed for users who do not need
end-to-end differentiation: explicit large RHSMode=1 QI requests can enter the
non-autodiff host x-block fallback and record that choice in solver metadata.
The true differentiable/device lane remains open because the scale-0.60 hard
seed does not yet pass the production output/write gate on GPU. The CPU hard
seed has a bounded orchestration path that writes HDF5 and solver trace output
at ``15 x 31 x 60 x 5`` without entering host strong-preconditioner or
SciPy-rescue time sinks. The best checked GPU installed-solver evidence is
the 2026-05-20 recycled augmented-Krylov/operator-reuse probe in combined mode.
It keeps the run on device, preserves the reusable QI coarse/operator action in
the FGMRES least-squares problem, avoids the transpose/CUDA crash, and
reduces the true residual from the public-auto ``3.949394e-5`` and earlier
augmented-Krylov ``2.218202e-5`` artifacts to ``7.336295e-6`` in ``158.6 s``.
It still misses the production write gate by roughly five orders of magnitude
relative to the ``3.021487e-11`` write tolerance, so it remains fail-closed
blocker evidence rather than a promotion artifact.
The 2026-05-22 coupled residual-equation GPU probe tested the next
Schur/coarse-space architecture in the right mathematical context: a
validated coupled stage was installed as the Krylov preconditioner after the
one-step seed probe rejected it. This reduced the coupled-probe wall time from
``8:07.61`` to ``3:08.10`` and peak host RSS from ``5.58 GB`` to ``3.90 GB``,
but the final residual was still ``2.450895e-5`` against the
``3.021487e-13`` Krylov target, so it is also fail-closed evidence. Its compact
artifact is
``docs/_static/qi_seed_robustness_scale060_coupled_residual_krylov_install_device_qi_gpu1.json``.
The lower-resolution ``13 x 13 x 15 x 4`` QI ``nfp=2`` operator-reuse route is
checked as a bounded office-GPU artifact:
``docs/_static/figures/optimization/qi_nfp2_electron_root_res13_gpu_operator_reuse_coupled_failclosed.json``.
That rerun activated matrix-free operator reuse, skipped local x-block factors,
kept host fallback disabled, wrote fail-safe solver-trace evidence, and records
the corrected device-cycle count of ``960`` iterations / ``962`` matvecs.
It finished in ``21.44 s`` wall time with peak trace RSS ``1384 MB``, but the
residual was ``9.707076e-6`` against target ``1.466182e-11``. This closes the
route/accounting evidence for the mid-size rung and keeps production true
device-QI convergence explicitly open.
The source tree also includes the fail-closed residual-learning
hook:
``SFINCS_JAX_RHSMODE1_XBLOCK_PC_POST_RESIDUAL_EQUATION``. It builds a bounded
JAX least-squares equation from the final true Krylov residual, optional cached
QI ``(U, A U)`` columns, and fresh physics residual directions. This is
infrastructure for learning from the error mode that Krylov actually leaves
behind; the matching runner preset is ``post-residual-equation-device-qi``. It
has one checked scale-0.60 CPU artifact:
``docs/_static/qi_seed_robustness_scale060_post_residual_equation_device_qi_cpu_2026_05_22.json``.
That artifact reaches the hook and accepts one correction, reducing the final
true residual from ``2.362283e-05`` to ``2.105918e-05`` using ``89`` directions
with cached QI columns. The matching GPU1 artifact,
``docs/_static/qi_seed_robustness_scale060_post_residual_equation_device_qi_gpu1_2026_05_22.json``,
also reaches the hook and reduces ``2.450895e-05 -> 2.142936e-05`` in
``168.120 s``. Both remain fail-closed because the production-sized write gate
is ``3.021487e-11`` and no HDF5/solver trace is written.
Scoped conclusion for this specific hard seed: the current device-QI research
path now gets the residual below ``3e-5`` on CPU and GPU, reproducibly and
without the old CUDA failure, but it does not yet solve the case to production
tolerance. Further research is needed to make this path better: specifically,
the remaining error must be attacked with a stronger residual-equation/coarse
space built from final Krylov error modes and current/constraint/profile
moments, not by further smoother or restart tuning.
The current promotion boundary is intentionally strict: the best checked
one-GPU true-device artifact is the recycled augmented-Krylov probe with final
residual ``7.336295e-6`` against the production write target
``3.021487e-11``. The later coupled/post-residual-equation probes reduce their
own setup residuals but finish at ``O(2e-5)`` and write no HDF5 output or solver
trace. The mid-size operator-reuse artifact writes fail-safe trace evidence and
has a useful lower memory/timing profile, but it is not the production hard seed
and fails the residual gate by ``O(1e6)``. These artifacts are therefore
fail-closed research evidence, not production or documentation-claim evidence.
Separate the closed infrastructure blockers from the open claim blockers:
transpose/VJP safety for the projected block smoother and the prior CUDA
illegal-address crash are closed for the tested paths, while residual
convergence, HDF5/solver-trace output, and installed-Krylov runtime effectiveness
remain open.

Relevant implementation:

- ``sfincs_jax/solvers/preconditioner_qi_basis.py`` builds deterministic
  QI coarse bases, phase-space and residual-region bases, active-pattern
  chunks, global-moment closures, and fail-closed Galerkin corrections.
- ``sfincs_jax/solvers/preconditioner_qi_corrections.py`` owns the
  device-compatible two-level, block-Schur, residual-deflated, multilevel,
  residual-Galerkin, and coupled residual-equation correction primitives.
- ``sfincs_jax/validation/qi_device.py`` defines the production
  ladder promotion gate: every requested seed/backend pair must converge,
  write output and solver trace artifacts, satisfy residual/observable gates,
  and avoid host fallback for a true device-QI claim.
- ``sfincs_jax/solvers/preconditioner_qi_device.py`` provides the standalone
  device-local ``S_local`` candidate and the production-shaped device-QI state.
  It combines a bounded CSR-backed Jacobi smoother with fail-closed diagonal
  validation, an opt-in residual-minimizing step policy, seed probes, and
  a rank-gated coarse basis when device CSR is available, and also provides a
  matrix-free coarse-only path that builds just ``A Q`` by JAX matvec probes
  when full CSR materialization is too expensive. Both paths expose
  setup/apply/probe metadata and keep the timed apply path free of SciPy, host
  LU/ILU, and Python callbacks. The coupled residual-equation setup now batches
  ``A Q`` construction when the operator supports JAX batching and reuses the
  cached action/coarse operator when that stage is installed in Krylov; metadata
  records reuse versus recompute counts and actual JAX array placement. It also
  exposes an opt-in adaptive multilevel residual-equation grouping that forms
  global, aggregate, and block residual source spaces. The first hard-seed GPU
  evidence worsened the residual relative to recycled augmented Krylov, so it is
  retained only as negative evidence and a tested research control.
- ``sfincs_jax/operators/profile_device_sparse.py`` provides bounded device CSR
  matvec utilities and records requested/default backend, available platforms,
  concrete array devices, concrete array platforms, and same-device placement
  for audit trails.
- ``sfincs_jax/solvers/preconditioner_qi_basis.py`` and
  ``sfincs_jax/solvers/preconditioner_qi_device.py`` build and admit QI
  Galerkin, residual-derived, and device-local candidates only when the true
  residual probe improves.
- ``sfincs_jax/problems/profile_solve.py`` wires the x-block sparse-PC,
  device-Krylov, two-level-QI opt-in, residual-deflated QI opt-in,
  device-QI field-split opt-in, early matrix-free QI probe, bounded
  post-xblock acceptance, post-Krylov residual-equation correction, and
  non-autodiff host fallback paths.

Audit conclusion
~~~~~~~~~~~~~~~~

The checked negative artifacts rule out another storage-only, side-threshold,
projected-smoother, or Krylov-label change. Compact CSR factors,
diagonal/one-sided factor applies, simple Galerkin rank-32 corrections,
LGMRES/GMRES toggles, device global-coupling QR, rank-deficient moment-Schur
pseudo-inverse, and a multiplicative base-plus-QI composition either fail to
reduce the true residual or make the residual/runtime worse. The best
checked path changes the coarse architecture with installed depth-64
operator-Krylov plus multilevel coarse reuse, but the small rank increase
(``12`` to ``13``) shows that the remaining error is not captured by the
angular/radial/global-load basis. The source tree includes the
first true augmented-Krylov replacement path: the stored QI coarse basis and
its operator action can be coupled directly to the restart least-squares
problem, ``min ||r - [A U, A Z] c||_2``. The checked
``recycled-augmented-device-qi`` hard-seed artifact is the best GPU evidence so
far, but its slow residual decay shows that a larger Krylov budget alone cannot
close the production tolerance. The source tree also includes
three non-smoother candidate spaces for bounded evidence runs: pitch/xi moments
in the multilevel coarse hierarchy, current/constraint tail moments for
flow/bootstrap-current/nullspace error, and an adjoint-normal Krylov basis
``orth([A^T r, (A^T A)A^T r, ...])`` for non-normal left-error modes.
The first scale-0.60 evidence checks keep them honest: pitch-enabled
multilevel did not change the GPU final residual, current/constraint moments
worsened the GPU residual and runtime, and CPU adjoint-normal depth-2 worsened
the final residual and runtime. None of these variants is promoted. The next
promotable algorithm must change the physics/error space captured by the coarse
solve, not simply add more restart cycles or smoother sweeps. The
post-residual-equation hook is the intended bounded probe for that idea: it
reuses final residual modes and stored operator actions, then fails closed if
the true residual does not decrease.

Next implementation
~~~~~~~~~~~~~~~~~~~

Use the existing two-level/device-QI primitives only as infrastructure for the
next research branch. Another smoother, damping, restart, or rank-only sweep is
not a credible closure route unless it adds genuinely new residual information
beyond the checked bases:

.. math::

   M^{-1} r =
   S_\mathrm{local}^{-1} r
   + P_c A_c^{-1} R_c \left(r - A S_\mathrm{local}^{-1} r\right),

where ``S_local`` is a device-resident x/species/angular block smoother,
``R_c`` restricts to moment/constraint/global-coupling, pitch, and adjoint-error
modes, ``P_c`` prolongs
back to the active vector, and

.. math::

   A_c = R_c A P_c

is a small replicated coarse operator solved with JAX dense linear algebra.

This follows the same structural idea as field-split/Schur preconditioning:
apply cheap local solves, then correct global low-dimensional coupling. It
should be tested first as a preconditioner action on physics load bases before
launching long Krylov solves.

The standalone device-Jacobi smoother is available as the minimal
operator-reuse ``S_local`` prototype. It deliberately validates diagonal
coverage before construction and performs only JAX CSR matvecs during
application, so it can probe whether a weaker but genuinely device-resident
smoother is preferable to host x-block LU/ILU for the hard GPU seed.

The standalone device-QI preconditioner state is available on top of that
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

The matrix-free path has an optional residual enrichment that adds bounded
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

The same matrix-free path has an opt-in residual-polynomial local smoother::

     SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_LOCAL_SMOOTHER=matrix_free_minres
     SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE_SMOOTHER_SWEEPS=2

This applies a fixed number of pure-JAX sweeps using the residual as the
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

The CPU hard-seed evidence uses this early hook plus the guarded
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
so scalar damping and storage-only changes are no longer promotion paths.
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
The 2026-05-20 one-GPU rerun confirms the projected smoother is CUDA-safe and
materially reduces the hard seed without host fallback: contiguous
``max_groups=32`` gives ``2.262e-6 -> 7.004e-7`` in ``75.3 s``; hybrid
``block_x_species`` gives ``6.864e-7`` in ``73.9 s``; and the strongest bounded
hybrid probe tested here, ``max_groups=48`` with four local sweeps, twelve
outer cycles, and residual-minimizing outer steps, gives ``4.689e-7`` in
``74.9 s``. This is a useful improvement over earlier recycle/scalar smoothers,
but it still misses the ``3.021e-11`` production write gate by about four orders
of magnitude. The conclusion is now stronger: projected block smoothing is a
safe device-resident component, but true device-QI closure needs a different
operator-reuse/coarse architecture rather than more tuning of this smoother.
The next operator-reuse pass added two opt-in coarse enrichments. Operator-image
enrichment expands the correction space with ``{Q, A Q, A^2 Q, ...}``; it was
CUDA-safe on the same ``office`` GPU hard seed but did not improve the best
projected-block residual (``4.707e-7`` in ``75.8 s``). The stronger residual
Arnoldi/Krylov enrichment builds ``orth([Q, r, A r, ...])`` from the current
true residual and reuses the resulting ``A Q`` action in the coarse
least-squares solve. With residual/recycle enrichment disabled so the new space
is isolated, depth ``16`` gives ``4.448e-7`` in ``75.2 s``, depth ``32`` gives
``4.199e-7`` in ``75.1 s``, and depth ``64`` gives ``3.627e-7`` in ``76.7 s``.
This was the best checked seed-only GPU evidence for this stage and showed a
real residual trend without memory growth, but it still missed the production
gate and did not close true device-QI. The next true closure candidate had to
install this coarse-reuse state into the actual Krylov solve or add a
mathematically stronger multilevel/coarse-grid correction; further
projected-smoother parameter sweeps were not expected to close the lane.

That installed path is now a named opt-in rather than only a future idea:

.. code-block:: text

   SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV=fgmres-jax
   SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_USE_IN_KRYLOV=1
   SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_OPERATOR_KRYLOV_ENRICHMENT=1
   SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_OPERATOR_KRYLOV_DEPTH=64

The runner preset ``operator-krylov-device-qi`` records the same controls. The
separate ``current-constraint-device-qi`` and ``adjoint-krylov-device-qi``
presets record negative-evidence variants without changing the recommended
operator-Krylov baseline. The operator-Krylov preset remains the correct route
for the next real-push experiment because it keeps the
operator-Krylov coarse state inside the JAX Krylov/preconditioner loop instead
of using it only as a seed-only correction. Promotion still requires the gate
below; merely reaching this path, improving the seed residual, or avoiding CUDA
failures is not enough.
The new ``augmented-krylov-device-qi`` preset is the stricter operator-reuse
variant: it enables cycle-JIT FGMRES and passes the stored QI coarse basis
``U`` and operator action ``A U`` directly to the solver. In its default
``combined`` mode each restart solves the coupled least-squares problem

.. math::

   \min_c \left\|r - [A U, A Z] c\right\|_2,

then applies the update through ``[U, Z] c``. This is stronger than only
projecting the residual over ``A U`` before Arnoldi because the coarse QI
directions remain active inside the Krylov residual equation. The first bounded
CPU/GPU artifacts are still nonpassing because no HDF5 output or solver trace
is written, but they are the best direct installed-solver residual-equation
evidence for this hard seed so far: CPU reaches ``2.218300e-5`` in ``174 s`` and
GPU0 reaches ``2.218202e-5`` in ``145 s``.
That improves on the previous GPU installed operator-Krylov/multilevel
residual ``2.306911e-5`` while keeping the promotion gate open.
The final checked recycled augmented-Krylov GPU0 probe uses the same installed
``(U, A U)`` coarse basis with a larger fixed device-cycle budget. It reduces
the hard-seed residual further to ``7.336295e-6`` in ``158.6 s``. This is the
best checked one-GPU QI residual in the evidence set and the current comparison
target for any future true device-QI work, but it still refuses HDF5 output
because the production write tolerance is ``3.021487e-11``.
The next non-smoother implementation is now wired as
``coarse-residual-device-qi``. It enables a nested multilevel residual equation:
each angular/radial/pitch coarse level gets its own rank budget, solves
``min_c ||r_l - A Q_l c||_2`` against the residual left by prior levels, applies
``Q_l c``, and optionally finishes with the global rank-gated coarse basis. This
is a deeper coarse-grid residual equation rather than another restart,
projection, damping, or local-smoother variant. Focused tests show that it
recovers synthetic angular-radial residual modes discarded by a flat rank-1
coarse gate, remains JIT-compatible, and records driver metadata through the
explicit QI-device hook. The first bounded scale-0.60 CPU hard-seed artifact
exercised the path and accepted the seed correction
``3.021487e-5 -> 2.840364e-5``, but the installed solve ended at
``2.306911e-5`` in ``269 s``. That is worse than the augmented-FGMRES CPU
baseline, so this configuration is negative evidence and is not promoted to a
GPU rerun.
The next checked non-smoother coarse-space path is
``residual-snapshot-device-qi``. It enriches the reusable device-QI coarse
operator with block/aggregate snapshots of the actual current residual and,
when requested, adjoint-normal snapshots ``A^T r_block``. The apply path still
uses the cached ``A Q`` action; the transpose work is setup-time only. The
bounded scale-0.60 CPU hard-seed artifact
``qi_seed_robustness_scale060_residual_snapshot_device_qi_cpu_2026_05_20.json``
is improvement evidence but not promotion evidence: the primal-only snapshot
finished at ``2.447236e-5``, while the adjoint-normal snapshot finished at
``2.103015e-5`` in ``250 s`` after accepting
``3.021487e-5 -> 2.769687e-5``. This is the best checked CPU residual in the
true-device-QI route so far, but it still refuses nonconverged HDF5 output and
keeps the CPU/GPU promotion gate open.
The follow-up ``residual-snapshot-equation-device-qi`` preset converts those
block/aggregate residual snapshots into a staged residual-equation cascade:
each accepted stage solves a setup-time ``min ||r_l - A Q_l c||_2`` problem,
caches ``A Q_l``, and applies through the existing pure-JAX coarse-action path.
The first bounded scale-0.60 CPU artifact
``qi_seed_robustness_scale060_residual_snapshot_equation_device_qi_cpu_2026_05_20.json``
is negative evidence. It accepts the seed correction
``3.021487e-5 -> 2.819970e-5`` and finishes at ``2.320763e-5`` in
``260 s`` before refusing nonconverged output. This is worse than the plain
residual-snapshot CPU artifact, so it is retained as a tested residual-equation
primitive and not promoted to GPU.
The deeper ``block-schur-device-qi`` preset now exercises a staged
block-Schur residual equation. It builds block/aggregate source probes at
setup, accepts only residual-reducing directions, caches their ``A Q_l``
actions, and applies them through the existing pure-JAX residual-equation
cascade. The implementation now also tries a coupled block/aggregate source
space and keeps it only when its measured setup residual is no worse than the
sequential fail-closed construction. The first bounded scale-0.60 CPU artifact
``qi_seed_robustness_scale060_block_schur_device_qi_cpu_2026_05_20.json``
is negative evidence: the seed correction improves only
``3.021487e-5 -> 2.840342e-5`` and the installed solve ends at
``2.275188e-5`` in ``267 s`` before refusing nonconverged output. This is worse
than the residual-snapshot CPU path, so it is kept as a tested research
primitive and not promoted to GPU.
The GPU0 best-of artifact
``qi_seed_robustness_scale060_block_schur_bestof_device_qi_gpu0_2026_05_20.json``
improves the final hard-seed residual to ``1.992464e-5`` in ``292 s``. It is a
bounded one-GPU residual-reduction artifact, but it still misses the production
write gate and remains fail-closed evidence.
The composite coarse-closure GPU1 artifact
``qi_seed_robustness_scale060_composite_closure_device_qi_gpu1_2026_05_20.json``
combines residual snapshots, residual-Galerkin/operator-image stages, and
block-Schur residual equations. It accepts a stronger setup correction
``3.021487e-5 -> 2.575099e-5`` but ends at ``2.305955e-5`` in ``313 s``. Since
that final Krylov residual is worse than the block-Schur best-of artifact, the
composite preset is kept only as audited negative evidence.
The ``global-moment-closure-device-qi`` preset exercises a true global closure
over profile, current, and reduced-tail constraint moments. The first bounded
scale-0.60 CPU artifact
``qi_seed_robustness_scale060_global_moment_closure_device_qi_cpu_2026_05_20.json``
accepts ``3.021487e-5 -> 2.840364e-5`` and ends at ``2.420524e-5`` in
``256 s`` before refusing nonconverged output. It proves the moment-Schur
closure is wired and device-compatible, but it is weaker than the
residual-snapshot evidence and remains fail-closed.
The matched GPU0 rerun
``qi_seed_robustness_scale060_global_moment_closure_device_qi_gpu0_2026_05_20.json``
is CUDA-safe and numerically consistent, ending at the same ``2.420524e-5`` in
``302 s`` before the nonconverged-output guard refuses HDF5 output.
The ``residual-galerkin-device-qi`` preset builds coarse variables from the
actual remaining residual and block residuals. The first bounded scale-0.60 CPU
artifact
``qi_seed_robustness_scale060_residual_galerkin_device_qi_cpu_2026_05_20.json``
accepts ``3.021487e-5 -> 2.766710e-5`` with rank ``16`` from ``21``
candidates and ends at ``2.632208e-5`` in ``244 s`` before refusing
nonconverged output. It is stronger than static global moments at setup, but
weaker than residual snapshots after Krylov; it is retained as tested
architecture rather than validation evidence.
The matched GPU1 rerun
``qi_seed_robustness_scale060_residual_galerkin_device_qi_gpu1_2026_05_20.json``
is also CUDA-safe and numerically consistent, ending at the same
``2.632208e-5`` in ``309 s`` before the nonconverged-output guard refuses HDF5
output.
The inspected 2026-05-20 public-auto-after-transpose and installed
operator-Krylov FGMRES summaries are still nonpassing hard-seed artifacts:
``gates.passed=false``, ``outputs_written=0``, and ``accepted_converged=0``.
They are useful closure evidence for routing/transpose/crash regressions only,
not validated device-QI evidence.

The projected block smoother also has a narrower completed infrastructure item:
its matrix-free block projection is transpose-safe in the tested JAX path. The
block directions are fixed from residual pieces and the least-squares action is
covered by a ``vjp`` regression, so it can remain in differentiable probes. This
does not promote projected smoothing as the closure strategy; the GPU evidence
above shows it is a useful local component but not sufficient by itself.

The alternative coarse direction is now a standalone multilevel/angular-radial
prototype in ``sfincs_jax/solvers/preconditioner_qi_corrections.py``. It constructs radial
aggregate hierarchies, angular harmonics, radial polynomial modes, and
radial-angular products, then applies a pure-JAX action least-squares coarse
correction after a local smoother. Unit tests show deterministic hierarchy
construction, reduction of a synthetic low-frequency angular-radial mode that a
local block smoother misses, fail-closed rejection when the angular space is
removed, and JIT/gradient compatibility. This is deliberately not wired as a
production default or hard-seed claim yet. Its promotion sequence is: integrate
behind an explicit QI-device hook, run the scale-0.60 CPU hard seed, then run
the one-GPU ``office`` hard seed, and only then consider production-resolution
QI ladders.

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
The production-floor follow-up keeps that branch fail-closed but makes it
more bounded and more observable. Host sparse x-block rescue now skips local
blocks above the default ``30000``-unknown cap instead of repeatedly attempting
singular high-fill ILU on the largest speed block, escalates local diagonal
regularization on singular ILU retries, and emits explicit breadcrumbs for the
post-seed refinement and SciPy polish. For very large active systems the default
post-x-block polish is a short probe, not a hidden long solve; users can still
force the historical uncapped local factorization or a longer polish with
environment overrides when running offline experiments. A good explicit x-block
seed also skips full global sparse rescue by default in this narrow branch,
because assembling the full active operator is the dominant memory/runtime
offender; setting ``SFINCS_JAX_RHSMODE1_SKIP_GLOBAL_SPARSE_AFTER_XBLOCK=0``
keeps the old rescue ordering for comparison studies. The same bounded branch
also avoids launching the final SciPy rescue by default; the resulting artifact
remains nonconverged unless the true residual gate is met.

The earlier residual-deflated path remains useful context but is no longer the
preferred next closure path after the operator-Krylov evidence:

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

This was the first step toward a GCRO/deflated-GMRES style device path: the
small basis targets the current hard residual instead of relying only on fixed
geometry modes. The current preferred follow-up is to reuse the stronger
operator-Krylov coarse state inside Krylov, or replace it with a genuine
multilevel/coarse-grid correction if Krylov installation still plateaus.

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
- The coupled residual-equation probe adds one more negative but useful result:
  jointly solving the already accepted multilevel, residual-snapshot, and
  block-Schur spaces avoids the staged-freezing failure mode and is faster when
  installed inside Krylov, but it still leaves the scale-0.60 hard-seed residual
  orders of magnitude above the write gate. The next true-device-QI attempt
  must therefore derive a new coarse equation from the remaining Krylov
  residual/error space itself, or the lane should stay deferred behind the
  documented non-autodiff host fallback for production large-QI use.
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
- Recycled augmented-Krylov coarse reuse is the current best checked one-GPU
  hard-seed evidence. It improves the true residual to ``7.336295e-6`` without
  reverting to host fallback, but it still misses the production output/write
  gate. Treat it as the comparison target for the next physics/error-space
  coarse equation, not as a closed true device-QI claim.
- Residual-region/bounce-region coarse reuse is now recorded as CPU/GPU
  negative evidence rather than a promoted route. The bounded scale-0.60 CPU
  hard seed ended at ``7.833826e-6`` and the matched GPU0 hard seed ended at
  ``8.077991e-6``, both far above the ``3.021487e-11`` write gate and slower
  than the recycled augmented-Krylov incumbent. Keep these artifacts in the
  evidence corpus, but do not use them for release-facing true-device QI
  claims. The aggregate manifest therefore reports these failed/nonconverged
  entries with requested-only classes and separate fail-closed observed
  metadata, preventing installed probe machinery from being counted as
  promotion evidence.
- Coupled residual-equation infrastructure is now the next non-smoother
  promotion attempt. It is wired through the driver, runner, and manifest as
  ``coupled-residual-device-qi`` and solves accepted multilevel/block-Schur/
  residual-snapshot coarse spaces together instead of freezing staged
  coefficients. The preset can also install the validated coupled stage inside
  Krylov after a rejected one-shot seed probe, which tests the route as a true
  preconditioner rather than as a seed-correction heuristic. This is still
  research evidence, not a release claim, until a bounded CPU/GPU hard-seed
  artifact writes converged output and records observed coupled-equation
  metadata.

Promotion gate
~~~~~~~~~~~~~~

- The scale-0.60 hard seed must write HDF5 and solver trace on CPU and one GPU.
- A one-application residual probe must reduce the current true residual by at
  least 5% and beat the existing fail-closed Galerkin probe.
- The Krylov-installed operator-Krylov path must record no host fallback,
  bounded rank/depth controls, and finite residual history through the full
  JAX Krylov loop; a seed-only operator-Krylov artifact cannot satisfy this
  gate by itself.
- The multilevel/angular-radial coarse path must first pass as an explicitly
  gated hard-seed artifact before it can replace the current preferred
  operator-Krylov installation route.
- CPU/GPU observables must remain within the existing parity tolerances.
- Further projected-smoother parameter sweeps are not a preferred closure path;
  install the operator-Krylov coarse state into Krylov or add a true
  multilevel/coarse-grid correction.
- Only after the hard seed passes should the five-seed and production-resolution
  QI ladders be launched.

Production-resolution QI ladders
--------------------------------

Evidence State
~~~~~~~~~~~~~~

Bounded QI evidence is strong up to the scale-0.55 CPU/GPU seed-robustness
ladders, selected scale-0.60 probes, and the first QI ``nfp=2`` kinetic
promotion artifacts. The checked kinetic lane includes a two-species
``7 x 7 x 7 x 4`` CPU/GPU/Fortran electron-root artifact and a refined
``9 x 9 x 11 x 4`` CPU/GPU/Fortran rung. A second bounded
``11 x 11 x 13 x 4`` CPU/GPU/Fortran rung exercises the fixed mid-size
RHSMode=1 full-FP dense policy and preserves CPU/GPU/Fortran root agreement at
fixed resolution. The low-to-refined and refined-to-next-refined root drifts
remain visible, so these rungs are persistence and solver-policy evidence
rather than convergence claims.

The next single-point QI probe at ``13 x 13 x 15 x 4`` exposed and fixed a
multi-device JAX mesh-context bug: transformed RHSMode=1 matvecs must not enter
the cached sharded ``pjit`` path from inside ``vmap`` or
``custom_linear_solve``. The fixed default CPU route writes a converged
``E_r=0.3`` output with active size ``11496``, residual ``2.09e-18`` against
target ``1.47e-11``. The follow-up sparse-LU skip-primary policy bypasses the
theta-line setup, primary Krylov attempt, and stage-2 GMRES for this measured
mid-size window, reducing solver time from about ``108 s`` to about ``35 s``
and peak RSS from about ``2.0 GB`` to about ``1.9 GB`` with identical key
observables. It is deliberately recorded as a bounded single-point
safety/profiling/speedup probe, not as a full resolution ladder: a complete
CPU/GPU/Fortran scan at this size still needs backend coverage before it can be
promoted.

Production-resolution ladders remain open because the scale-0.60 GPU hard seed
is not closed by a true device algorithm and because the QI kinetic
electron-root scan still needs a wider CPU/GPU/Fortran resolution ladder.

Next implementation
~~~~~~~~~~~~~~~~~~~

After a true device-QI hard-seed artifact writes converged output and solver
trace, run:

- the next QI ``nfp=2`` kinetic CPU/GPU/Fortran resolution rungs, with fixed
  profiles, species, electric-field grid, and claim tolerances;
- scale-0.60 five-seed CPU and one-GPU ladders;
- production-resolution proxy ladders at the documented floor, or the
  equivalent production manifest inputs;
- parity and residual audits against the current host fallback and Fortran v3
  reference outputs where available.

The checked promotion helper in ``sfincs_jax/validation/qi_device.py`` should be
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
  evidence. Its dry-runs are explicitly non-promoting; a row becomes
  ``promotion_ready`` only after real child solves pass residual, stall, RSS,
  backend, solver-path, guarded-fallback, and baseline-improvement gates.

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
- ``sfincs_jax/problems/transport_parallel_runtime.py`` prevents cold or
  malformed scaling payloads from becoming release claims, records pure
  single-case sharded-solve plans, caps requested devices to available work,
  reports workload balance, estimates setup/communication amortization, and
  fail-closes release scaling claims for experimental single-case sharding.
- ``sfincs_jax/operators/profile_system.py`` contains the sharded matrix-free operator path.
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

The CI-safe planner now records this target explicitly through
``plan_single_case_operator_coarse_reuse`` and the
``operator_coarse_reuse_plan`` field in
``benchmark_sharded_solve_scaling.py --plan-only`` artifacts. A promotable
artifact must reuse the assembled operator and replicated coarse state, pass the
deterministic output gate, beat the warm one-device solve, and prove that the
multi-device peak memory does not increase. This keeps the next implementation
step concrete without letting dry-run metadata become a scaling claim.

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

1. Add pure unit tests for each new policy/helper before wiring it into the
   canonical profile or transport solve owner.
2. Run focused CPU tests for QI, PAS, sharding, and release-gate policy.
3. Run bounded CPU hard-seed probes.
4. Run one-GPU hard-seed probes on ``office`` only after the CPU preflight
   reduces the true residual.
5. Launch five-seed and production-resolution ladders only after the hard seed
   is closed.
6. Regenerate README/docs plots only after checked artifacts change public
   claims.
