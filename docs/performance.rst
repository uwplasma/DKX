Performance and differentiability
=================================

`sfincs_jax` is designed around a few principles that enable both speed and gradients:

1) **Matrix-free operators**: avoid assembling sparse matrices; apply the discrete operator as a pure function.
2) **JIT compilation**: compile hot kernels (matvecs, residuals, linear solves) with `jax.jit`.
3) **Vectorization**: prefer `vmap`, `einsum`, and batched linear algebra over Python loops.
4) **Explicit separations of concerns**: non-differentiable I/O (reading `.bc`/`wout_*.nc`) is isolated from
   the differentiable compute graph.

For a full, technique-by-technique breakdown (equations, derivations, knobs, and
implementation notes), see :doc:`performance_techniques`.


Measured head-to-head: canonical stack vs SFINCS Fortran v3
-----------------------------------------------------------

The canonical-stack benchmark case is ``HSX_PASCollisions_DKESTrajectories``
(RHSMode=1) at ``Ntheta=25, Nzeta=51, Nxi=100, Nx=5`` — 744,610 unknowns —
measured on the same development machine (MacBook, Apple M4, ~10 cores, 24 GB)
for both codes. The Fortran reference is the conda PETSc 3.23 + MUMPS 5.8.2
build of SFINCS v3; ``sfincs_jax`` uses the tier-1 truncated Legendre block
elimination (``solvax`` ``block_thomas_truncated_fn``, blocks assembled on the
fly from the analytic operator coefficients, ``keep_lowest=3`` — exact for
every RHSMode=1 output).

.. figure:: _static/figures/readme/tier1_hsx_runtime_memory.png
   :alt: Runtime and peak memory bars for sfincs_jax and SFINCS Fortran v3 on the 744k-unknown HSX PAS case.
   :align: center
   :width: 90%

   Measured warm solve time and peak process RSS. Regenerate with
   ``python tools/benchmarks/readme_figures.py``; rerun the measurement with
   ``python tools/benchmarks/tier1_hsx_head_to_head.py``.

.. list-table:: Head-to-head (744k unknowns, HSX PAS DKES, RHSMode=1)
   :header-rows: 1

   * - Configuration
     - Warm solve [s]
     - Peak RSS [GB]
   * - ``sfincs_jax`` MacBook M4 CPU, ``Nxi``-for-``x`` ramp
     - 27.2
     - 0.93
   * - ``sfincs_jax`` MacBook M4 CPU, uniform ``Nxi``
     - 44.3
     - 1.16
   * - ``sfincs_jax`` RTX A4000 GPU
     - 45.0
     - 1.88 (0.05 GB VRAM buffers)
   * - SFINCS Fortran v3, 1 MPI rank
     - 463.6
     - 3.98
   * - SFINCS Fortran v3, 2 MPI ranks (measured floor)
     - 229.5
     - 2.86

With the matched ramp discretization this is 17x faster than 1-rank Fortran
and 8.4x faster than Fortran's best measured parallel floor, at roughly 30% of
the memory. Ramp-vs-uniform ``Nxi`` differences on the physics outputs are at
most 0.9% (electrons). GPU time equals M4 CPU time because the Legendre scan
is serial in ``L`` and the A4000 runs FP64 at 1/32 rate; GPU upside requires
batching over (species, ``x``, surfaces/``Er``) or fp32 factors with fp64
refinement. Scope: this is one measured 744k-unknown HSX PAS case; further
cases are promoted as each vertical slice lands with its own evidence.

Fortran strong-scaling baseline (same case, same machine)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - MPI ranks
     - Solve time [s]
     - Speedup
     - Parallel efficiency
     - Peak RSS [GB]
   * - 1
     - 463.6
     - 1.00
     - —
     - 3.98
   * - 2
     - 229.5
     - 2.02
     - 101%
     - 2.86
   * - 4
     - 240.9
     - 1.92
     - 48%
     - 2.88
   * - 8
     - 270.5
     - 1.71
     - 21%
     - 1.61

Fortran/MUMPS saturates at 2 ranks on this machine and degrades beyond
(performance/efficiency core asymmetry plus MUMPS OpenMP contention), so the
practical Fortran floor for this case is about ``230 s``.

Memory findings
~~~~~~~~~~~~~~~

- At the full production resolution of this case
  (``Ntheta=25, Nzeta=115, Nxi=149, Nx=5``; a 2,512,760-unknown system),
  **neither** code fits a global sparse factorization on a 24 GB machine:
  Fortran/MUMPS drove macOS swap to ~46.5 GB during factorization and was
  killed, and the dense/CSR JAX host paths are size-capped well below it.
- The truncated Legendre block elimination is the locally viable direct path:
  its memory is ``O(K m^2)`` with ``m = Ntheta * Nzeta`` (one ~66 MB
  ``2875^2`` block at production resolution, independent of ``Nxi``). On the
  744k case the truncated route needs ~0.3 GB where a full-band tier-1 factor
  would need ~91 GB.

Solver-noise finding
~~~~~~~~~~~~~~~~~~~~

The direct solve is more converged than the Fortran reference: Fortran's own
electron ``FSABFlow`` scatters 51% across its 1/2/4/8-rank runs of this case
(KSP ``rtol=1e-6`` iterative-solver noise), while ``sfincs_jax`` matches the
closest Fortran run to ``2e-10`` and sits inside Fortran's own spread on every
compared quantity.

Parity referees
~~~~~~~~~~~~~~~

.. figure:: _static/figures/readme/canonical_parity.png
   :alt: Measured parity envelopes of the canonical stack against Fortran and recorded references.
   :align: center
   :width: 90%

   Parity envelopes pinned by the CI referee tests
   (``tests/test_run_rhsmode1.py``, ``tests/test_run_transport.py``):
   RHSMode=1 output tables at ``8e-14``, tier-1 state vectors vs recorded
   references at ``1e-11``, RHSMode=2/3 transport matrices vs Fortran golden
   data at ``6e-13 .. 9e-9``.

Known issues
~~~~~~~~~~~~

- **Silently wrong tier-2 adjoint on singular FP systems.** Full
  Fokker-Planck with ``constraintScheme=1`` on the flagship-optimization deck
  yields a numerically singular system (~5 zero singular values, condition
  number ~2e36); the tier-2 GCROT adjoint stagnates and the implicit-diff VJP
  returns a wrong gradient without any error (AD ``-1.7e-3`` vs FD
  ``+2.8e-5`` on the affected dof). PAS+``Er`` tier-2 gradients on the same
  chain are exact (``2.9e-6`` vs FD). Fix direction: surface adjoint-solve
  convergence in ``SolveResult`` and raise/flag when the adjoint residual
  misses tolerance. A reproducer is tracked.
- **Ill-conditioned scheme-1 monoenergetic off-diagonal.** The Fortran build
  itself fails upstream's ``tests.py`` on the ``monoenergetic_geometryScheme1``
  ``transportMatrix[0,1]`` element only (``+1.62`` vs expected ``-1.08`` at
  ``solverTolerance=1e-6``; ``+26.3`` at ``1e-12`` — tolerance-unstable, so
  the element is ill-conditioned in this configuration). Parity tests pin that
  element to upstream's expected value (``-1.07986``), which the ``sfincs_jax``
  direct solve reproduces to ``4.2e-6``.

CPU and GPU: where each wins
----------------------------

The structured tier-1 path is the one that lets ``sfincs_jax`` fit and finish a
production case; the honest headline is that **it reaches CPU/GPU parity on the
direct tier, while every iterative or small-system path is dispatch-bound and
runs slower on the GPU**. The full 744k-unknown HSX case was re-measured on both
backends after the ramp-aware truncated kernel became the canonical route:

.. list-table:: Post-fix 744k HSX PAS/DKES head-to-head (end-to-end = build + solve)
   :header-rows: 1
   :widths: 40 32 28

   * - Backend
     - Runtime
     - Peak RSS
   * - SFINCS Fortran v3 (PETSc + MUMPS, dev MacBook)
     - > 2.6 h, unfinished
     - 3.6-5.7 GB
   * - ``sfincs_jax`` CPU (dev MacBook)
     - 41.4 s e2e (25.0 s warm solve)
     - 1.35 GB
   * - ``sfincs_jax`` GPU (RTX A4000)
     - 59.6 s e2e (26.2 s warm solve)
     - 2.3 GB

Both backends route ``block_tridiagonal_truncated`` and land within a few seconds
of each other on the warm solve; under the tier-2 recycled-Krylov route the same
case ran out of memory on the 16 GB GPU, so the structured tier is what makes it
fit at all. A mid-size HSX case (336k unknowns) is at warm-solve parity as well
(``3.5 s`` CPU versus ``3.3 s`` GPU).

The GPU does **not** help the iterative and small-system paths. Full
Fokker-Planck GCROT, the :math:`\Phi_1` Newton solve, ``value_and_grad``, the
ambipolar Brent root, and a one-shot monoenergetic solve all run 2-5x *slower* on
the A4000: they are dominated by serial iterations (and the tier-1 Legendre
:math:`L`-scan is itself serial, with FP64 at 1/32 rate on this card), so device
dispatch latency dominates. GPU wins require batched work — multi-:math:`E_r` or
multi-surface ``vmap`` sweeps — not single solves.

Production profiling battery
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The per-case picture on CPU (development MacBook, idle, one fresh subprocess per
case, via ``tools/benchmarks/profile_production.py``):

.. list-table::
   :header-rows: 1
   :widths: 30 14 26 16 14

   * - Case
     - Unknowns
     - Method
     - Solve cold / warm
     - Peak RSS
   * - HSX PAS/DKES, mid (ramp)
     - 336,610
     - ``block_tridiagonal_truncated``
     - 4.1 s / 3.5 s
     - 885 MB
   * - W7-X full Fokker-Planck, 2 species
     - 78,628
     - GCROT, 115 iters
     - 17.5 s / 7.6 s
     - 4.1 GB
   * - Monoenergetic RHSMode=3, scheme 1
     - —
     - ``block_tridiagonal``
     - 4.8 s e2e
     - 2.0 GB
   * - PAS gradient (``value_and_grad``)
     - 39,318
     - tier-1 differentiable
     - 5.3 s / 2.1 s
     - 4.3 GB
   * - :math:`\Phi_1` Newton
     - 4,548
     - unpreconditioned GCROT
     - 232 s / 0.04 s
     - 1.4 GB
   * - Ambipolar :math:`E_r`, 2 species
     - —
     - Brent root
     - 31.2 s
     - 3.9 GB

Two lessons stand out. First, promoting the ramp-aware truncated kernel into
``solve(method="auto")`` is what pulls the mid-size HSX case from the tier-2
route's ``10.8 GB`` / ``15.4 s`` warm down to ``885 MB`` / ``3.5 s`` warm — 12x
less memory and roughly 4x faster on the same deck, with the autodiff gradient
through the ramped route still matching finite differences at rtol ``1e-6``.
Second, the cold ``232 s`` :math:`\Phi_1` Newton solve (an unpreconditioned,
restart-capped GCROT inner solve at only 4.5k unknowns) is the top remaining
runtime target; its warm re-solve is already ``0.04 s``, so the cost is entirely
first-call iteration count, not steady state.

Example-suite benchmark
-----------------------

A broader, fast-to-reproduce benchmark complements the single production case: it
runs the full 39-case CPU/GPU example suite against SFINCS Fortran v3 and plots
every row whose Fortran reference runtime clears a ``10 s``
reference-runtime-window, so process launch, filesystem overhead, and JIT
amortization do not dominate the shorter measurements. The full suite stays the
parity audit; only the runtime/memory *plot* applies the window.

.. figure:: _static/figures/paper/sfincs_jax_fortran_suite_benchmark_summary.png
   :alt: Runtime and active-memory comparison for SFINCS Fortran v3 and sfincs_jax across the example suite.
   :align: center
   :width: 92%

   Example-suite benchmark. Runtime bars (left) and active-memory bars (right)
   for each reference-runtime-window case: SFINCS Fortran v3, ``sfincs_jax`` CPU
   cold/warm, and ``sfincs_jax`` GPU cold/warm, ordered by best warm
   ``sfincs_jax`` speedup over the Fortran v3 runtime. Fortran memory is process
   maximum RSS; JAX memory uses profiler RSS deltas over the fixed
   Python/JAX/XLA baseline. Reproduce with
   ``examples/publication_figures/generate_fortran_suite_benchmark_summary.py``.

Across the plotted rows the median cold JAX/Fortran wall-clock ratio is about
``0.021x`` on CPU and ``0.037x`` on GPU. Median active-memory ratios are about
``2.89x`` on CPU and ``3.71x`` on GPU; the full process maximum-RSS ratios, kept
in the summary JSON audit fields, are about ``4.75x`` on CPU and ``8.80x`` on
GPU. The numeric summary and the top runtime/memory cases are recorded in
``examples/publication_figures/artifacts/sfincs_jax_fortran_suite_benchmark_summary.json``.

Compile time vs steady state
----------------------------

JAX pays a one-time compile cost that a persistent cache amortizes across runs.
The checked figure below separates the two for four reference transport cases:
compile estimate (cold first call minus warm first call) versus steady warm solve
time.

.. figure:: _static/figures/transport_compile_runtime_cache_2x2.png
   :alt: Compile estimate versus warm steady-state solve time for four reference cases.
   :align: center
   :width: 95%

   Per case, compile estimate = cold first call - warm first call; steady solve
   is the warm repeated runtime. Warm solves are tens of milliseconds once
   compiled, so repeated scans and optimization loops run at steady-state speed.

For steady-state benchmarking, take repeated JAX runs and report the warm timing;
set a persistent ``JAX_COMPILATION_CACHE_DIR`` to reuse compiled kernels across
processes.

Performance patterns
--------------------

The design choices that produce the numbers above, in one place:

- **Matrix-free operators.** The drift-kinetic Jacobian is applied as a pure
  function (tensor contractions and directional derivatives), never assembled as
  a sparse matrix, so it JIT-compiles for CPU/GPU and differentiates cleanly.
- **Structured direct tier.** The block-tridiagonal-in-:math:`L` elimination with
  truncated storage is the memory lever: ``O(K m^2)`` with ``m = Ntheta*Nzeta``,
  independent of ``Nxi`` (:doc:`numerics`).
- **The** :math:`N_\xi`-**for-**:math:`x` **ramp.** Fewer Legendre modes at high
  speed cut both work and memory; on the 744k HSX case the ramp is the difference
  between ``0.93 GB`` and ``1.16 GB`` at essentially identical physics outputs
  (:math:`\le 0.9\%`).
- **Subspace recycling.** The tier-2 recycle pair warm-starts neighbouring points
  in an :math:`E_r` scan or a :math:`\Phi_1` Newton iteration, so continuation
  converges in a handful of iterations.
- **Preconditioning by a simplified exact solve.** Tier 2 is right-preconditioned
  by an exact tier-1 solve of a collision-/drift-simplified coarse operator (the
  Fortran ``preconditionerOptions`` idiom).
- **Short-recurrence Krylov for transport.** RHSMode=2/3 solves default to
  memory-lean BiCGStab with a collision-diagonal preconditioner, with GMRES as a
  fallback.
- **Gradient checkpointing.** ``jax.checkpoint`` around collision operators and
  transport diagnostics trades recomputation for lower peak memory during
  autodiff on long chains.

For the full technique-by-technique treatment — equations, derivations, tuning
knobs, and implementation notes — see :doc:`performance_techniques`.

Differentiable paths
--------------------

Gradients are exact and cost about one extra solve, because the adjoint reuses the
forward factorization through the implicit function theorem. What is
differentiable (geometry, profiles, the ambipolar :math:`E_r`, the :math:`\Phi_1`
state, the monoenergetic transport matrix), the measured gradient-vs-finite-
difference agreement, and the honest tier-2 singular-Fokker-Planck caveat are all
documented in :doc:`differentiability`.
