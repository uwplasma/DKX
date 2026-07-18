Parallelism
===========

`dkx` runs on a single node — a multi-core CPU or one GPU — and gets its
throughput from three places: batched ``jax.vmap`` over independent solves,
optional device sharding of a single large solve, and the structured ``solvax``
solve tiers underneath. This covers the same physics that SFINCS Fortran v3
spreads across many nodes with MPI, while keeping the whole path differentiable.

The lever to reach for first is **batching independent solves**. Scanning the
radial electric field, sweeping flux surfaces, or building a monoenergetic
database are all embarrassingly parallel — each point is its own solve — and a
single vmapped call amortizes dispatch on CPU and fills the device on GPU.

Two kinds of parallelism
------------------------

- **Across independent solves.** Many ``E_r`` values, many flux surfaces, many
  ``whichRHS`` right-hand sides, or many optimizer/scan points. They share a
  discretization and differ only in a few physics leaves, so one batched call
  solves them together. This is where the throughput is.
- **Within a single solve.** One large linear system split across host devices
  or GPUs along ``theta``/``zeta``/``x``. Available for cases too large for a
  single device, but secondary to batching for scan-shaped work.

Batching independent solves
---------------------------

The batched API in ``dkx.batch`` productizes ``jax.vmap`` over solves
that share a discretization:

.. code-block:: python

   from dkx.batch import batched_er_scan, batched_surface_scan

   # Scan the radial electric field on one geometry.
   result = batched_er_scan(problem, er_values)
   radial_current = result.radial_current       # J_r for each E_r value

   # Sweep a set of flux surfaces (one KineticOperator each).
   result = batched_surface_scan(operators)

Both return a ``BatchedSolveResult`` carrying the stacked moments, both accept
``differentiable=True`` to keep the batch inside a ``jax.grad`` chain, and both
take optional ``max_batch`` / ``memory_budget_gb`` overrides. Independent solves
— a vector of ``E_r`` values, a set of surfaces, or the ``(nu*, E_r)`` grid of a
monoenergetic database (``dkx.monoenergetic``) — are exactly the
parallel-friendly shape.

**Automatic memory budgeting.** There are no sharding environment variables on
this path. The batch runs in ``jax.lax.map`` chunks sized from the per-solve
tier-1 memory footprint and the resolved device (or host) memory budget, so only
one chunk's intermediates are ever live. ``memory_budget_gb`` overrides the
resolved budget and ``max_batch`` caps the chunk size; the defaults need
neither.

**Measured throughput.** Because ``vmap`` amortizes per-solve dispatch, batching
beats a serial Python loop even on CPU — about ``9.5x`` for an ``E_r`` scan and
``6.4x`` for a surface scan. The larger win is on the GPU, where a single solve
sits at CPU parity but a batch fills the device. Reproduce both with
``python tools/benchmarks/batched_scan.py``.

Solve tiers and where the GPU helps
-----------------------------------

Every solve routes through the three ``solvax``-backed tiers selected by the
``auto`` policy (:doc:`numerics`):

- **Tier 1 — structured direct.** Block-tridiagonal elimination over the
  Legendre index. For the DKES-trajectory / pitch-angle family the system splits
  into independent block-tridiagonal systems (one per species and speed node)
  solved with ``vmap``. This tier is GPU-viable and the one batching
  accelerates.
- **Tier 2 — preconditioned recycled Krylov.** Matrix-free FGMRES with subspace
  recycling, right-preconditioned by an exact tier-1 solve of a simplified
  coarse operator. It carries a recycle pair to warm-start neighbouring points
  in an ``E_r`` scan or a Newton iteration.
- **Tier 3 — host sparse-direct fallback.** A host sparse factorization for
  cases the structured tier cannot admit; non-differentiable, used only on
  ``method="direct"`` or when tier 2 breaches its iteration cap.

The honest headline, measured in :doc:`performance`: the GPU reaches CPU parity
on the **direct** tier and runs the **iterative** and small-system paths 2-5x
*slower*, because those are dominated by serial, dispatch-bound iterations. GPU
wins therefore come from **batched** direct-tier work — multi-``E_r`` or
multi-surface sweeps — not from single solves.

Host devices and CPU threads
----------------------------

Multi-core CPU execution is requested **before JAX is imported**; the CLI
``--cores`` flag sets the environment for you:

.. code-block:: bash

   dkx --cores 8 input.namelist       # or: export DKX_CORES=8

``DKX_CORES`` requests that many host CPU devices and enables auto
sharding, ``DKX_CPU_DEVICES`` sets the device count directly, and
``DKX_XLA_THREADS`` opts into pinning the XLA CPU thread count. Full
semantics and defaults are in the environment-variable reference (:doc:`usage`).

Single-solve sharding
---------------------

When one solve is too large for a single device, the matvec can be sharded
across the host devices or GPUs:

- ``DKX_MATVEC_SHARD_AXIS`` — ``auto``/``off``/``theta``/``zeta``/``x``/``flat``.
- ``DKX_AUTO_SHARD`` and ``DKX_SHARD`` — enable and master-disable
  auto sharding.
- ``DKX_SHARD_PAD`` — neutral padding when a sharded axis is not divisible
  by the device count (does not change outputs).

These mirror the angular domain decomposition an MPI code uses. Their full
behaviour, together with the distributed-Krylov knobs, is documented in
:doc:`usage`.

Multi-host execution
--------------------

For multi-host device pools, JAX distributed initialization is opt-in via
``DKX_DISTRIBUTED`` together with ``DKX_PROCESS_ID``,
``DKX_PROCESS_COUNT``, ``DKX_COORDINATOR_ADDRESS``, and
``DKX_COORDINATOR_PORT`` (or the matching ``--distributed``,
``--process-id``, ``--process-count``, ``--coordinator-address``, and
``--coordinator-port`` CLI flags). Independent transport right-hand sides can
additionally be spread across worker processes with
``DKX_TRANSPORT_PARALLEL`` / ``DKX_TRANSPORT_PARALLEL_WORKERS``
(CLI ``--transport-workers``). See :doc:`usage` for the full list.

Relation to SFINCS Fortran v3
-----------------------------

SFINCS Fortran v3 scales one solve across many nodes with MPI domain
decomposition. `dkx` targets a single node — a multi-core CPU or one GPU
— and recovers scan-level throughput a different way: batched ``vmap`` over
independent solves, subspace recycling across neighbouring points, and exact
gradients that replace finite-difference scans in optimization. Parallel paths
call the same matrix-free operators as the sequential path, so outputs stay
bit-compatible up to floating-point reduction order and a parallel run is a
referee for a serial one.
