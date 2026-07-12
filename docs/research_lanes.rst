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
optimization work. The code contains tested opt-in infrastructure for
Fortran-reduced direct ``Pmat`` emission, symbolic ordering metadata,
superblock/nested-dissection factors, and strict setup-time true-residual
admission. Those pieces are useful research controls, but production
``geom11`` probes still reject the native nested-dissection path on setup time
before admission, and full-grid QA/QH still rely on residual-clean active-LU
fallback when the lower-memory native candidate fails its residual gate.

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

QI/device-QI research preservation
-----------------------------------

QI seed-robustness campaigns, true device-QI hard-seed probes, and
production-resolution QI ladders are preserved on the
``research/qi-device-hard-seed`` branch instead of the stable core. The stable
branch keeps the general solver interfaces and claim-boundary documentation,
but removes QI-only generated artifacts, QI-only example inputs, and QI-specific
promotion tests until a candidate implementation passes release gates.

The next promotable QI/device-QI branch must provide compact regenerated
evidence for each claim:

- CPU and GPU strict true-residual convergence,
- HDF5/NetCDF/NPZ output-write success where the output is claimed,
- solver-trace, runtime, and peak-memory metadata,
- CPU/GPU observable parity,
- SFINCS Fortran v3 parity where the model overlap is claimed,
- documented differentiability scope for Python workflows, and
- source/tests/docs small enough to preserve the stable package structure.

Until those gates pass, QI/device-QI material is not a release blocker and must
not appear in README-facing performance, parity, or optimization claims.

Geometry-rich PAS runtime/RSS promotion
---------------------------------------

Current state
~~~~~~~~~~~~~

PAS byte budgets and fail-closed matrix-free gates prevent unsafe promotions.
However, the real geometry4 and HSX probes were residual-clean but did not
improve runtime or peak memory, so no default promotion is justified.

Relevant implementation:

- The retired ``preconditioner_pas_matrix_free`` owner (deleted with the legacy pipeline) contained guarded
  matrix-free correction helpers, candidate-size preflights, and
  ``PasRuntimeChunkPlan`` for deriving bounded reduction chunks from configured
  byte budgets.
- The retired PAS policy owner (deleted with the legacy pipeline) contained PAS
  applicability and memory gates.
- Historical PAS fallback promotion/rejection scripts are preserved on the
  research branch. In the stable core, dry-run evidence is explicitly
  non-promoting; a candidate route becomes default-eligible only after real
  solves pass residual, stall, RSS, backend, solver-path, guarded-fallback, and
  baseline-improvement gates.

Next implementation
~~~~~~~~~~~~~~~~~~~

Promote the reduction chunk planner from helper-level memory safety into a
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
- The retired transport parallel runtime (preserved on the research branch) prevented cold or
  malformed scaling payloads from becoming release claims, records pure
  single-case sharded-solve plans, caps requested devices to available work,
  reports workload balance, estimates setup/communication amortization, and
  fail-closes release scaling claims for experimental single-case sharding.
- The retired legacy operator owner (preserved on the research branch) contained the sharded matrix-free operator path.
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

The CI-safe planner records this target explicitly through
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
