Parallelism
===========

`dkx` runs on a single node — a multi-core CPU or the node's local GPUs — and
gets its throughput from two places: batched ``jax.vmap`` over independent
solves (optionally split across the node's devices) and the structured
``solvax`` solve tiers underneath. This covers the same physics that SFINCS
Fortran v3 spreads across many nodes with MPI, while keeping the whole path
differentiable.

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
- **Across the node's devices.** When more than one accelerator is visible, the
  batch of independent solves splits across them (``devices=...``); each device
  runs the same memory-budgeted chunked solve on its shard, so multiple GPUs
  fill with scan-shaped work. A single solve runs whole on one device — there
  is no internal split of one linear system across devices.

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
memory footprint of the route the ``auto`` policy actually takes — a solve
that routes to the memory-lean truncated tier-1 kernel is charged its
truncated working set, not the full-band factorization peak that route never
allocates — and the resolved device (or host) memory budget, so only one
chunk's intermediates are ever live. ``memory_budget_gb`` overrides the
resolved budget and ``max_batch`` caps the chunk size; the defaults need
neither.

**Measured throughput.** Because ``vmap`` amortizes per-solve dispatch, batching
beats a serial Python loop even on CPU — about ``9.5x`` for an ``E_r`` scan and
``6.4x`` for a surface scan. The larger win is on the GPU, where a single solve
sits at CPU parity but a batch fills the device. Reproduce both with
``python tools/benchmarks/batched_scan.py``.

**Multiple devices.** The batch elements are embarrassingly parallel, so the
batched calls also accept a ``devices`` argument that splits the batch across
devices: ``devices="auto"`` uses every local device of the default backend
when more than one is visible, and an explicit sequence of ``jax.Device``
objects selects a subset. Each device receives a contiguous near-equal shard
of the batch, runs the same memory-budgeted chunked solve on it — the budget
bounds each device's chunk, so the auto-chunking arithmetic is per device —
and the results are gathered on the host in batch order. Anything short of
two usable devices, or a batch smaller than the device count, degrades to the
single-device path unchanged, and traced inputs (inside ``jax.jit`` /
``jax.grad``) fall back to the single-device path, which computes the same
answer; keep ``devices=None`` on paths meant for tracing. The per-element
computation is the single-device computation: with matched executed chunk
widths (an explicit ``max_batch``) the results are bitwise identical across
device counts, and the identity gate in ``tests/test_batch.py`` verifies
element-wise identical results for one versus two forced host CPU devices
(``DKX_CPU_DEVICES=2``), which exercises the same split/placement/gather path
as two GPUs. Multi-GPU *speedup* validation is pending access to a
multi-GPU host — the API is measured-correct on multi-device CPU, where no
speedup is possible (forced host devices share one threadpool) and the split
costs one extra per-shard dispatch, so the honest CPU expectation is
neutral-to-slower wall time.

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

Subsystem batching within a tier-1 solve
----------------------------------------

The truncated tier-1 kernel eliminates ``B = n_species * n_x`` independent
``(species, x)`` subsystems. ``solve(subsystem_batch=...)`` sets how many it
eliminates concurrently. An integer fixes the width (clamped to ``[1, B]``;
``1`` is the fully serial, minimum-memory sweep), and any width computes
identical per-subsystem arithmetic — the knob trades memory for batched
parallel work, so the CPU path is byte-identical to the serial sweep.

``subsystem_batch="auto"`` (the default) is backend-aware:

- **CPU backend — width 1.** XLA:CPU runs the batch axis of the LAPACK
  factor/solve custom calls serially per element, so a wider sweep only adds
  memory and cache pressure, not parallelism. Measured on the 336,610-unknown
  mid HSX deck at 8 threads, every width above 1 is neutral-to-slower: the
  ramped deck is 10.3 s at width 1 versus 11.4 s at width 2, and the
  uniform-``Nxi`` variant 16.6 s at width 1 versus 20.5 s at width 10.
- **Accelerator backends — memory-budgeted width.** The widest width whose
  modeled footprint (:func:`dkx.solve.tier1_truncated_peak_memory_bytes`) fits
  the memory budget, because batching raises device occupancy there while the
  budget clamp bounds the working set.

The knob is ignored by the non-truncated tiers.

CPU threads
-----------

The XLA host threadpool is sized once, when the CPU backend initializes, so
thread control must be in place **before JAX is imported**; the CLI
``--cores`` flag sets the environment for you:

.. code-block:: bash

   dkx --cores 4 input.namelist       # or: export DKX_CORES=4

``DKX_CORES=N`` pins the solver threadpool to ``N`` threads (applied as
``NPROC``, the variable XLA reads, plus the host BLAS OpenMP/OpenBLAS pools);
``DKX_CORES=0`` lets XLA size the threadpool itself; when unset the
threadpool is clamped to ``min(8, cpu_count)``. The measured optimum is 4-8
threads on both a 10-core laptop and a 36-core workstation: tier-1 thread
scaling saturates near 2-2.5x and **inverts** beyond the optimum on wide
machines. On the 36-core workstation the mid HSX deck (336,610 unknowns) warm
tier-1 solve measures 9.7 s at 1 thread, 7.8 s at 2, 5.6 s at 4, and 4.87 s at
8 threads — the optimum, a 1.99x speedup at 25% parallel efficiency — then
rises back to 12.2 s at 16, 56.6 s at 32, and 29.3 s at the full 36. The
operator build stays flat near 8 s at every core count, so the inversion is
entirely the XLA fork-join overhead over the sequential Legendre-block sweep
once the pool is too wide (the wide-pool tail carries large run-to-run
variance). The guidance on many-core hosts is to set ``--cores`` to
roughly 4-8, not to ``nproc``. ``DKX_CPU_DEVICES`` is a separate, explicit
opt-in that forces multiple host *devices* for multi-device CPU tests; forced
host devices share one threadpool, so it is not a performance knob. Full
semantics and defaults are in the environment-variable reference
(:doc:`usage`).

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
