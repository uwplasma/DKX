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
- ``sfincs_jax/rhs1_qi_two_level.py`` provides the first device-compatible
  local-smoother plus coarse-correction primitive for the next hard-seed probe.
- ``sfincs_jax/rhs1_device_operator.py`` provides bounded device CSR matvec
  utilities.
- ``sfincs_jax/rhs1_qi_galerkin_policy.py`` rejects Galerkin candidates unless
  a true residual probe improves.
- ``sfincs_jax/v3_driver.py`` wires the x-block sparse-PC, device-Krylov,
  two-level-QI opt-in, and non-autodiff host fallback paths.

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

The current environment switch is::

   SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER=1

This switch is fail-closed: the driver probes the true physical residual and
keeps the baseline x-block preconditioner unless the two-level action improves
the current seed by a material margin. The default margin is 5% because a
scale-0.60 CPU hard-seed preflight showed that accepting a tiny 0.7% damped
one-step decrease can make the subsequent Krylov phase substantially worse.
The default path tests only the requested damping; damping scans are
explicit-only with
``SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_DAMPINGS``.

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
  helpers and candidate-size preflights.
- ``sfincs_jax/rhs1_pas_policy.py`` contains PAS applicability and memory gates.
- ``scripts/benchmark_pas_tz_memory_fallback.py`` records promotion/rejection
  evidence.

Next implementation
~~~~~~~~~~~~~~~~~~~

Replace dense candidate update materialization with streamed/chunked correction
actions over pitch-angle/angular blocks. The candidate should use
``jax.lax.scan`` or equivalent fixed-shape chunks so the live set is bounded and
the same operation can run on CPU or GPU.

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

Promotion gate
~~~~~~~~~~~~~~

The warm two-GPU single-case solve must beat the warm one-GPU solve by at least
1.15x, per-device memory must not increase, and the residual/output comparison
must pass. Until then, public docs should continue to recommend one GPU per case
or transport RHS.

External numerical anchors
--------------------------

The planned algorithms are anchored in established sparse-solver and JAX
infrastructure:

- `PETSc PCFIELDSPLIT <https://petsc.org/main/manualpages/PC/PCFIELDSPLIT/>`_
  documents block and Schur-complement preconditioners, including local sub-KSP
  solvers and explicit Schur actions.
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
