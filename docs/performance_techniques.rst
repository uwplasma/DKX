Performance techniques (full detail)
====================================

This page documents **every performance enhancement currently implemented in `sfincs_jax`**,
including the mathematical model context, implementation strategy, tuning knobs, and
how each change differs from (or complements) the original Fortran v3 solver.

Where relevant we reference the upstream SFINCS documentation that defines the
physics and discretization being accelerated. The primary sources are the
vendored v3 manual and technical notes in ``docs/upstream`` (see references below).

Baseline model and linear system (v3)
-------------------------------------

For linear runs (``RHSMode=1``), v3 assembles a linear system of the form

.. math::

   A x = b,

where :math:`x` contains the distribution function unknowns (the **F-block**),
optionally the :math:`\Phi_1(\theta,\zeta)` block (QN) and a constraint scalar
:math:`\lambda`. The operator :math:`A` is the linearized drift-kinetic operator,
and :math:`b` is the drive from thermodynamic gradients, inductive fields, and
other source terms. See :doc:`system_equations` and the upstream manual for the
full normalized expressions and parameter definitions.

For transport-matrix runs (``RHSMode=2/3``), the operator is the same **linear**
operator, but the RHS is overwritten internally for each ``whichRHS`` (v3
``evaluateResidual(f=0)``), and the resulting solutions are postprocessed into
transport coefficients (particle/heat fluxes and FSAB flow).

`sfincs_jax` implements the same model and discretization, but replaces the
matrix assembly with **matrix-free operator application** and JAX-based kernels.

What SFINCS v3 does (for performance context)
---------------------------------------------

The Fortran v3 code uses PETSc/KSP for iterative solves:

- It **assembles sparse matrices** (and sometimes dense blocks) from the discretized
  operator in ``populateMatrix.F90``.
- PETSc performs GMRES/BCGS with user-selected preconditioners, and stores the
  Krylov basis explicitly (memory intensive for large restarts).
- Transport-matrix mode loops over ``whichRHS``, reuses the same matrix, and
  performs multiple solves.

This is a strong baseline for CPU-only runs, but:

- The matrix assembly cost can be large relative to matrix-free matvecs.
- Sparse storage inflates memory, especially in 3D grids or multispecies FP runs.
- Building preconditioners (even block Jacobi) can be expensive in wall time.

`sfincs_jax` retains the same physics but uses a different numerical strategy:
**matrix-free + JIT + JAX-native preconditioning**.

Comparison with Fortran v3 workflow
-----------------------------------

**Fortran v3**

- Assemble sparse matrices in PETSc format.
- Use PETSc/KSP GMRES or BiCGStab with PETSc preconditioners.
- Transport matrices solved by repeated RHS with the same assembled operator.
- Diagnostics evaluated in per-``whichRHS`` loops.

**sfincs_jax**

- Apply the operator matrix-free (no assembly).
- JIT compile matvecs and solver loops; reuse compilation cache across runs.
- Use lightweight JAX-native preconditioners that avoid matvec-assembled blocks.
- Batch transport diagnostics across all ``whichRHS`` and reuse precomputed factors.

These differences are purely algorithmic/performance-oriented; the physics and
normalization remain anchored to the same v3 equations.

Matrix-free operator application (A·x) and caching
--------------------------------------------------

**Technique.** Replace explicit matrix assembly with a matvec:

.. math::

   y = A x,

where the matvec is computed by composing collisionless, drift, and collision
operators directly on the state vector.

**Implementation.**

- Core operator apply: ``sfincs_jax.operators.profile_response.system.apply_v3_full_system_operator_cached``.
- Per-operator **signature cache** prevents re-JITing the matvec when the operator
  shape and static fields are unchanged. See ``_operator_signature_cached``.

**Why it’s fast.**

Matrix-free matvec avoids assembling large sparse matrices and amortizes
operator evaluation over JIT-compiled kernels.

**Impact.**

Reduces matrix assembly overhead and enables kernel fusion in XLA. The effect is
largest in repeated transport solves and in large multispecies FP cases where
assembly dominates total runtime.

**Knobs.**

- ``SFINCS_JAX_SOLVER_JIT``: disable/enable JIT of Krylov solves.
- ``SFINCS_JAX_SOLVER_JIT_MAX_SIZE``: auto-JIT cutoff (default ``2000``).
- ``SFINCS_JAX_TRANSPORT_MATVEC_MODE``: choose base vs RHS operator for transport solves.

**Compared to Fortran.**

Fortran builds PETSc matrices once and multiplies by them; `sfincs_jax` applies
the operator directly, which is often cheaper than assembling matrices,
especially when JIT compilation fuses multiple operator sub-terms.

Structured solve admission gate
-------------------------------

**Technique.** Before changing production preconditioners, benchmark whether a
factor-once / repeated-RHS structured solve is accurate, memory-efficient, and fast on
a bounded block-tridiagonal proxy.

**Implementation.**

- Structured block solver:
  ``sfincs_jax.discretization.structured_velocity.factor_block_tridiagonal``.
- Benchmark harness:
  ``examples/performance/benchmark_structured_solve.py``.
- Focused tests:
  ``tests/test_benchmark_structured_solve.py``.

The harness compares a dense repeated solve with a reusable block-tridiagonal
factorization on deterministic synthetic systems. It reports residuals, maximum
solution error against the dense solve, dense storage bytes, structured storage bytes,
and warm solve timings.

It also has a real-operator mode:

.. code-block:: bash

   python examples/performance/benchmark_structured_solve.py \
     --case sfincs-pas-block \
     --sfincs-input tests/ref/monoenergetic_PAS_tiny_scheme1.input.namelist \
     --n-rhs 4 \
     --out-json examples/performance/output/structured_solve_sfincs_pas_gate.json

This mode fixes one species and one speed index, extracts the active Legendre-mode
chain and full angular grid from the matrix-free SFINCS PAS F-block, checks that the
extracted block is block-tridiagonal, then applies a small diagonal regularization
before comparing dense and structured solves. The regularization is explicit in the
JSON output and exists only to make the local preconditioner-style block nonsingular;
it does not change production solver behavior.

Pinned-offender status
~~~~~~~~~~~~~~~~~~~~~~

The first larger real-block gate used
``tests/reduced_inputs/geometryScheme4_2species_PAS_noEr.input.namelist`` with
species ``0``, speed index ``4``, two right-hand sides, and the explicit
benchmark-only regularization ``1e-4``. The extracted block was exactly
block-tridiagonal in Legendre index (``off_band_norm = 0``), with 14 blocks of
size ``81``.

On the local CPU gate this reduced local-block storage from ``10,287,648`` bytes
for the dense representation to ``2,099,520`` bytes for the structured
representation. Accuracy against the dense regularized block was acceptable for a
preconditioner gate: dense relative residual ``2.25e-13``, structured relative
residual ``9.74e-10``, and max solution difference ``2.94e-7``.

The same local gate did **not** clear the warm-runtime rule: dense solve
``0.1100 s`` versus structured factor+solve ``0.2305 s``. For this reason the
gate validates the memory direction but does not justify a broader production
threshold change by itself. The current production path already uses this idea
where it matters: the user-facing log reports top-level ``schur`` for the
geometry4 PAS case, and the Schur base selector chooses ``pas_tz`` internally for
the pinned offender block.

Run it with:

.. code-block:: bash

   python examples/performance/benchmark_structured_solve.py \
     --nblocks 32 --block-size 8 --n-rhs 8 \
     --out-json examples/performance/output/structured_solve_gate.json

**Admission rule.** A structured algorithm should not be wired into real SFINCS solve
paths unless it is parity-clean on the relevant fixture and this harness, preferably in
``--case sfincs-pas-block`` mode or a direct larger-offender extension of it, shows a
material runtime or memory benefit. The current recommended gate is the same as the
research roadmap: at least ``20%`` warm runtime improvement or ``25%`` memory reduction
on a pinned offender, with no suite drift above ``1.25x``.

JIT compilation and persistent compilation cache
------------------------------------------------

**Technique.** Use `jax.jit` for hot kernels (matvecs, residuals, Krylov loops),
and enable persistent compilation caching for repeated runs.

**Implementation.**

- `sfincs_jax` JITs the matvec and solver wrappers (GMRES/BiCGStab).
- The reduced-suite runner (`scripts/run_reduced_upstream_suite.py`) supports a
  persistent cache via ``--jax-cache-dir``.
- The CLI defaults to a user cache directory (``~/.cache/sfincs_jax/jax_compilation_cache``)
  and enables ``jax.experimental.compilation_cache`` automatically unless disabled.
- You can override the default directory with ``SFINCS_JAX_COMPILATION_CACHE_DIR``.
- Command-line subcommands lazily import heavy modules to reduce startup overhead.

**Why it’s fast.**

JIT amortizes Python overhead and enables XLA fusion. Persistent cache reduces
cold-start overhead in batch runs.

**Impact.**

Removes repeated compilation costs in batch suites and improves throughput for
workflow-style runs (parameter scans, repeated transport matrices).

**Compared to Fortran.**

Fortran has no JIT overhead but also no fusion; JAX replaces repeated Python-side
dispatch with compiled kernels.

Geometry/output caching
-----------------------

**Technique.** Cache geometry arrays and expensive output-only quantities on disk
so repeated cases that share the same equilibrium file reuse the computed data.

**Implementation.**

- ``sfincs_jax.discretization.v3.geometry_from_namelist`` persists the full ``BoozerGeometry`` arrays
  to ``~/.cache/sfincs_jax/geometry_cache`` by default.
- ``sfincs_jax.io.sfincs_jax_output_dict`` caches expensive output-only fields
  (``gpsiHatpsiHat``, ``uHat``, ``diotadpsiHat``, ``VPrimeHat``, ``FSABHat2``,
  ``BDotCurlB``, and the ``classical*NoPhi1_*`` fluxes) in
  ``~/.cache/sfincs_jax/output_cache``.
- Both caches key equilibria by file content rather than staged temporary path, so
  repeated suite reruns and copied-case benchmarks reuse cached HSX/W7-X data even
  when the input tree is localized into a fresh directory.
- The output-cache key also includes the species block plus the static
  classical-transport scalars (``Delta``, ``alpha``, ``nu_n``, ``nuPrime``,
  ``EStar``, and ``RHSMode``), while intentionally ignoring trajectory-model
  switches such as ``useDKESExBDrift``. This allows DKES/full-trajectory pairs
  that share the same static physics inputs to reuse the same cached output-only
  payload safely.
- The RHSMode=1 output-writing path also reuses the already-built ``grids``,
  ``geom``, and full-system operator instead of rebuilding them again during the
  solve handoff. This removes a full operator-construction pass from
  ``write_sfincs_jax_output_h5(...)`` on staged HSX/geometry11 reruns.
- Disable with ``SFINCS_JAX_GEOMETRY_CACHE=0`` / ``SFINCS_JAX_OUTPUT_CACHE=0`` or
  skip disk persistence with ``SFINCS_JAX_GEOMETRY_CACHE_PERSIST=0`` /
  ``SFINCS_JAX_OUTPUT_CACHE_PERSIST=0``.
- Override cache roots with ``SFINCS_JAX_GEOMETRY_CACHE_DIR`` and
  ``SFINCS_JAX_OUTPUT_CACHE_DIR``.

**Impact.**

Reduces ``sfincs_jax_output_dict`` time substantially for repeated runs on the
same equilibrium, especially HSX/W7-X cases where ``gpsiHatpsiHat``, ``uHat``,
and the static classical/no-Phi1 diagnostics are otherwise recomputed each run.
On the copied-HSX offender probe, warming the cache on the original case and
rerunning from a copied ``.bc`` path reduced ``sfincs_jax_output_dict`` from
``1.554 s`` to ``1.257 s`` without changing the solve path or outputs.
On the narrowed RHSMode=1 PAS offenders, the reused setup path reduced the
operator-build stage from ``1.928 s`` to ``0.002 s`` on
``HSX_PASCollisions_DKESTrajectories``, from ``0.583 s`` to ``0.002 s`` on
``HSX_PASCollisions_fullTrajectories``, and from ``0.218 s`` to ``0.002 s`` on
``sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories``,
while keeping the outputs parity-clean against the frozen Fortran references.

Active-DOF reduction (sparse pitch grid)
----------------------------------------

**Technique.** When the pitch-angle basis is truncated (`Nxi_for_x < Nxi`),
construct a reduced system that contains only active degrees of freedom.

**Implementation.**

- Transport solves: ``_transport_active_dof_indices`` in ``sfincs_jax.v3_driver``.
- RHSMode=1: similar logic for active DOFs to reduce matrix-free work.

**Mathematics.**

Let :math:`P` be the selection matrix that extracts active DOFs:

.. math::

   x_{\mathrm{act}} = P x, \qquad A_{\mathrm{act}} = P A P^\top, \qquad b_{\mathrm{act}} = P b.

Solve the reduced system and map back via :math:`x = P^\top x_{\mathrm{act}}`.

**Why it’s fast.**

Reduces problem size and Krylov memory by removing unused Legendre modes.

**Impact.**

Substantial reductions in both memory and Krylov iterations when
``Nxi_for_x`` truncation is active (common in reduced-resolution suites).

**Compared to Fortran.**

Fortran typically keeps the full layout; `sfincs_jax` can safely reduce
if the basis truncation is explicit in the input.

Krylov solver strategy (short recurrence + fallback)
----------------------------------------------------

**Technique.** Use GMRES as the default for RHSMode=1 and BiCGStab
as the default for transport, with GMRES fallback on stagnation or non-finite residuals.

**Motivation.**

GMRES stores a full Krylov basis, with memory ~ :math:`O(n \cdot \text{restart})`.
BiCGStab is short recurrence with memory ~ :math:`O(n)`.
IDR(s) is another short-recurrence family used for nonsymmetric systems and is a
candidate for future low-memory solves.

Host-only LGMRES fast path
--------------------------

For explicit host-side solves, ``sfincs_jax`` now exposes a bounded SciPy
``lgmres`` path. This is intended for the fast CLI / explicit branch where
differentiability is not required, and where restarted GMRES can spend too much
time rebuilding a Krylov basis on hard nonsymmetric systems.

**Implementation.**

- Wrapper: ``sfincs_jax.solver.lgmres_solve_with_history_scipy``.
- Dispatch point: ``sfincs_jax.solver._gmres_solve_core``.
- Accepted methods: ``solve_method in {"lgmres", "lgmres_scipy"}``.

**Behavior.**

- Uses SciPy ``lgmres`` on the host with the same left/right preconditioning
  semantics already used by the existing SciPy GMRES wrappers.
- Remains disabled for distributed solves.
- If the requested solve still routes through an implicit, JITed, or distributed
  context, `sfincs_jax` downgrades to traced-safe ``incremental`` GMRES instead of
  failing at runtime.

This makes the new method safe as a CLI performance option: it can accelerate
hard host-only Krylov runs without changing differentiable workflows or
contaminating the differentiable reference route. Frozen-case offender probes on
``main`` still support keeping this as an explicit tuning knob rather than an
automatic default. On the current pinned heavy cases, ``lgmres`` preserves
parity but is not yet a general win: it is only marginally different on the
tokamak PAS+Er offender and is slower on the current frozen geometry4 and
geometry5 full-system examples. The method is therefore kept available, but
opt-in.

Structural sparse-host RHSMode=1 path
-------------------------------------

For large full-system RHSMode=1 production solves, ``solve_method="sparse_host"``
uses the same matrix-free physics operator but materializes only a conservative
structural sparse pattern:

1. Build a full-system sparsity superset from the v3 layout, theta/zeta
   derivative stencils, dense same-geometry FP velocity/species blocks,
   Phi1/quasineutrality couplings, and constraint rows.
2. Color columns with disjoint row supports.
3. Probe one combined seed vector per color and unpack only declared nonzeros.
4. Drop structural zeros, factor the resulting CSR matrix with host sparse LU,
   and apply iterative refinement against the true sparse operator.

The implementation lives in ``sfincs_jax.operators.profile_response.sparse_pattern`` and
``sfincs_jax.solvers.explicit_sparse.build_operator_from_pattern``. Tests verify that the
conservative pattern covers frozen Fortran PETSc matrices for PAS, FP, and Phi1
tiny systems, and that colored probing reconstructs the PAS tiny matrix to
Fortran tolerances.

This path is non-differentiable and should be used for CLI/Python production
runs, not for gradient tracing. CPU 3D full-FP RHSMode=1 systems now have a
narrow sparse-PC GMRES auto lane in the audited size window: HSX FP full,
HSX FP DKES, and geometryScheme11 FP full probes all stayed Fortran-clean while
using less RSS and less wall time than the dense FP shortcut. The gate excludes
PAS, ``N_zeta=1`` tokamak systems, Phi1/QN, E_parallel, accelerator backends,
implicit/differentiable solves, and explicit user-selected solve methods. Large
constrained-PAS RHSMode=1 profile-current decks also auto-select the sparse-PC
GMRES host lane in the validated production window because that branch
converges the true residual on the finite-beta profile-current bring-up deck in
seconds rather than minutes. Host sparse LU remains explicit because it is only
correct when the constrained RHSMode=1 system has a pinned gauge/nullspace
branch compatible with that factorization. For production experiments that need
an explicit full-system sparse solve, prefer
``solve_method="sparse_host_safe"``: it tries sparse LU first, and if sparse LU
detects a singular constrained-PAS branch it falls back to a PETSc-compatible
minimum-norm solve with separate acceptance metadata. Use ``sparse_lsmr``
directly only as a diagnostic probe. RHSMode=1 outputs record
``linearSolverConverged``, ``linearSolverAccepted``, the acceptance criterion,
and the residual norm/target so nonconverged or branch-compatible diagnostics
are visible in the main file, not only in the optional solver trace. Sparse-PC
GMRES outputs also record setup time, solve time, sparse-pattern build time,
preconditioner factorization time, matvecs, and sparse-pattern row-density
counters, so large-run regressions can be separated into pre-solve setup and
Krylov iteration costs.

For explicit non-differentiable accelerator output runs, ``auto`` may route
moderate full-FP RHSMode=1 systems to the same host sparse rescue when the
active system is too large for the dense accelerator shortcut and device Krylov
has a known slow or fragile tail. This is bounded by
``SFINCS_JAX_RHSMODE1_ACCELERATOR_HOST_SPARSE_RESCUE_MAX`` (default ``30000``
active unknowns) and can be disabled with
``SFINCS_JAX_RHSMODE1_ACCELERATOR_HOST_SPARSE_RESCUE=0``. The QI
``9 x 19 x 35 x 4`` gate records the intended behavior: one RTX A4000 moved
from a rejected ``195 s`` Krylov-tail solve to a converged ``42.8 s`` host-sparse
rescue with true residual ratio ``4.49e-7``.

Tokamak full-FP ``N_zeta=1`` production-floor rows are handled by separate,
narrow GPU/CUDA policies rather than by the CPU 3D full-FP lane above.
``SFINCS_JAX_RHSMODE1_TOKAMAK_FP_NOER_SPARSE_PC`` covers no-Er
``constraintScheme=0`` rows and ``SFINCS_JAX_RHSMODE1_TOKAMAK_FP_ER_SPARSE_PC``
covers electric-field ``constraintScheme=1`` rows. These policies now select
``xblock_sparse_pc_gmres`` for the validated full-FP branch, using compact
per-x/TZ host preconditioners instead of the earlier global dense-velocity
sparse-pattern probe. The audited ``25 x 1 x 8 x 100`` RTX A4000 default-policy
checks stayed parity-clean against Fortran v3: no-Er compared ``188`` datasets
with ``0`` mismatches in ``5.79 s`` wall / ``0.97 GB`` peak RSS, and with-Er
compared ``214`` datasets with ``0`` mismatches in ``1:13.3`` wall / ``1.43 GB``
peak RSS. The no-Er row replaces the previous global sparse-PC default that
took ``2:31.6`` and about ``8.42 GB`` peak RSS.

Controls for the CPU 3D full-FP auto lane are
``SFINCS_JAX_RHSMODE1_FP3D_SPARSE_PC``,
``SFINCS_JAX_RHSMODE1_FP3D_SPARSE_PC_MIN``, and
``SFINCS_JAX_RHSMODE1_FP3D_SPARSE_PC_MAX``.
The production benchmark manifest now enforces at least ``25 x 51 x 4 x 100``
(``Ntheta x Nzeta x Nx x Nxi``) for 3D cases and ``33 x 1 x 12 x 140`` for
tokamak cases. RHSMode=1 PAS/no-``E_r`` tokamak rows use the calibrated
``89 x 1 x 24 x 300`` floor. The manifest also records a ``10 s`` minimum
SFINCS Fortran v3 timing target for public production rows. Performance claims
for this lane should be regenerated from that manifest rather than from earlier
lower-resolution bring-up probes. It is
available via:

.. code-block:: bash

   sfincs_jax write-output --input input.namelist --out sfincsOutput.h5 --solve-method sparse_host_safe

   sfincs_jax write-output --input input.namelist --out sfincsOutput.h5 --solve-method xblock_sparse_pc_gmres

The historical finite-beta ``17 x 21 x 5 x 12`` PAS/profile-current deck remains
useful as a solver bring-up regression for sparse-host correctness, but it is no
longer used as a public production baseline.

Frozen-case variant benchmarking
--------------------------------

Use ``scripts/benchmark_case_variants.py`` to compare solver/preconditioner env
overrides on one promoted suite case without disturbing the full-suite artifacts:

.. code-block:: bash

   python scripts/benchmark_case_variants.py \
     --case-dir tests/scaled_example_suite_fast_cpu_full_v7_refresh/geometryScheme5_3species_loRes \
     --variant 'lgmres=SFINCS_JAX_RHSMODE1_SOLVE_METHOD=lgmres'

The helper runs the default variant plus any requested overrides, records wall
time and ``ru_maxrss``, and compares each output H5 against both the frozen
Fortran output (when present) and the default variant.

For targeted rescue testing, skip the known default route with ``--no-default``:

.. code-block:: bash

   python scripts/benchmark_case_variants.py \
     --case-dir tests/production_floor_cpu_bounded_2026-05-04/tokamak_1species_PASCollisions_withEr_fullTrajectories/variant_prod_case \
     --no-default \
     --variant 'schur=SFINCS_JAX_RHSMODE1_PRECONDITIONER=schur' \
     --timeout-s 300 \
     --profile

Use this mode only for controlled profiling. A variant is not eligible for
automatic selection unless it is residual-clean, parity-clean against the frozen
Fortran output, and passes the documented runtime/memory promotion gates.

Adaptive PAS smoother stage
---------------------------

For PAS-heavy ``RHSMode=1`` solves, the expensive part of the fallback ladder is
often not the base Krylov solve itself, but the sequence of progressively stronger
preconditioner builds that follow when the residual is still above target. The new
adaptive PAS smoother stage inserts a bounded Richardson-like correction before
that escalation:

.. math::

   x_{k+1} = x_k + \omega P^{-1}(b - A x_k),

where :math:`P^{-1}` is the already-built PAS preconditioner application and
:math:`\omega` is a relaxation factor.

**Implementation.**

- Smoother kernel: ``sfincs_jax.pas_smoother.adaptive_pas_smoother``.
- Driver gate: ``sfincs_jax.v3_driver._rhsmode1_pas_adaptive_smoother_allowed``.
- Current integration point: immediately after the base PAS solve and before the
  strong-preconditioner tail in the linear RHSMode=1 driver.

**Why this helps.**

The smoother is much cheaper than building a new ``pas_hybrid`` / ``pas_schur`` /
``xblock_tz`` fallback, because it reuses the preconditioner that is already in
memory. The algorithm keeps the best iterate seen so far and stops as soon as:

- the target residual is reached,
- the residual worsens by more than a configured factor, or
- the relative improvement plateaus.

This makes it safe as a default explicit-path optimization: if the smoother does
not help, the driver falls back to the existing ladder unchanged.

**Knobs.**

- ``SFINCS_JAX_PAS_ADAPTIVE_SMOOTHER``: enable/disable the stage.
- ``SFINCS_JAX_PAS_ADAPTIVE_SMOOTHER_MIN``: minimum active size for activation.
- ``SFINCS_JAX_PAS_ADAPTIVE_SMOOTHER_SWEEPS``: maximum smoothing sweeps.
- ``SFINCS_JAX_PAS_ADAPTIVE_SMOOTHER_OMEGA``: Richardson relaxation factor.

**Compared to Fortran v3.**

This is not a PETSc feature copied from v3. It is an explicit-path optimization
for `sfincs_jax`: the physics operator is unchanged, but the fallback ladder is
made more selective by trying a bounded cheap correction before paying for a much
heavier preconditioner build.

**Implementation.**

- ``_solve_linear`` and ``_solve_linear_with_residual`` in ``sfincs_jax.v3_driver``.
- Fallback controlled by ``SFINCS_JAX_BICGSTAB_FALLBACK``.

**Compared to Fortran.**

Fortran typically uses GMRES via PETSc; `sfincs_jax` keeps GMRES for RHSMode=1 parity
and switches to BiCGStab for transport to reduce memory, with GMRES fallback.

**Impact.**

Lower memory footprint for large systems and improved wall time in many transport
cases. GMRES remains available when BiCGStab stagnates.

Implicit differentiation through linear solves
----------------------------------------------

**Technique.** Use `jax.lax.custom_linear_solve` to differentiate
through linear solves without storing Krylov iterates.

**Math.**

For :math:`A(p)\,x(p) = b(p)`, implicit differentiation yields:

.. math::

   A \frac{dx}{dp} = \frac{db}{dp} - \frac{dA}{dp}x.

The adjoint solve reuses the same linear operator, so gradients are efficient
and memory-bounded.

**Implementation.**

- ``sfincs_jax.solvers.implicit.linear_custom_solve`` and
  ``linear_custom_solve_with_residual``.
- ``sfincs_jax.solvers.implicit.implicit_solve_method_for_custom_linear_solve``
  maps host-only CLI Krylov choices such as ``lgmres_scipy`` to traced-safe
  ``incremental`` GMRES before entering ``jax.lax.custom_linear_solve``.
- Controlled by ``SFINCS_JAX_IMPLICIT_SOLVE``.

Transport preconditioning (RHSMode=2/3)
---------------------------------------

**Technique.** Use analytic, JAX-native preconditioners to reduce Krylov iterations
without matvec-based assembly.

**Collision-diagonal preconditioner (baseline).**

Approximate the operator with its collision diagonal:

.. math::

   P^{-1} \approx \left(\mathrm{diag}(\mathcal{C}) + \alpha I \right)^{-1}.

Includes PAS diagonal and FP self-collision diagonal (per L, per x).

**Species×x block-Jacobi (FP auto / opt-in).**

When the FP operator is available, build per-:math:`L` blocks across species and :math:`x`:

.. math::

   \mathsf{C}^{(L)}_{(a,i),(b,j)} \equiv \mathcal{C}^{\mathrm{FP}}_{ab,ij}(L),

and invert each :math:`\mathsf{C}^{(L)}` (with identity shift + PAS diagonal).
Inactive :math:`x` points (from `Nxi_for_x`) are masked to identity.

**Low-rank Woodbury correction (optional).**

For FP-heavy cases, approximate the dense species×x blocks with a low-rank update
and apply a Woodbury inverse:

.. math::

   \left(D + U V^\top\right)^{-1}
   = D^{-1} - D^{-1} U \left(I + V^\top D^{-1} U\right)^{-1} V^\top D^{-1}.

This reduces both setup and apply costs for the FP preconditioner when
``SFINCS_JAX_TRANSPORT_FP_LOW_RANK_K`` (or ``SFINCS_JAX_FP_LOW_RANK_K``) is set,
including the ``auto`` default for larger FP blocks.

**Measured impact (FP-heavy cases).** With ``SFINCS_JAX_TRANSPORT_FP_LOW_RANK_K=auto``
the low-rank update improved end-to-end wall time by ~9% for
``geometryScheme5_3species_loRes`` and ~21% for ``tokamak_1species_FPCollisions_noEr``
relative to ``SFINCS_JAX_TRANSPORT_FP_LOW_RANK_K=0`` (profiles in
``examples/performance/output/reduced_profiles_fp_*.json``).
See ``docs/references.rst`` for Woodbury/low-rank update references.

**Coarse x-grid additive preconditioner (xmg).**

``SFINCS_JAX_TRANSPORT_PRECOND=xmg`` adds a two-level correction:
fine-grid collision-diagonal smoothing plus a coarse x-grid solve per species/L.
Set ``SFINCS_JAX_XMG_STRIDE`` to control the coarsening.

**Angular domain-decomposition blocks (theta/zeta).**

``SFINCS_JAX_TRANSPORT_PRECOND=theta_dd`` or ``zeta_dd`` builds line-local
block preconditioners in the selected angular direction. Each block inverts a
local subdomain (species × active x/L × line extent) while dropping couplings
to other lines, which keeps the preconditioner apply local and differentiable.
This is useful when experimenting with sharded transport solves.

``SFINCS_JAX_TRANSPORT_PRECOND=theta_schwarz`` or ``zeta_schwarz`` enables a
restricted additive Schwarz (RAS) variant with overlap:

.. math::

   M^{-1}_{RAS} r = \sum_i R_i^T \tilde{A}_i^{-1} R_i r,

where :math:`R_i` restricts to an overlapped patch and the correction is written
back to the non-overlapped core (RAS update). This improves conditioning across
block boundaries compared with pure block-diagonal DD. The path is currently
opt-in because setup/apply overhead can dominate on small-medium runs.

Controls:

- ``SFINCS_JAX_TRANSPORT_DD_BLOCK_T`` (theta block size, default ``8``)
- ``SFINCS_JAX_TRANSPORT_DD_BLOCK_Z`` (zeta block size, default ``8``)
- ``SFINCS_JAX_TRANSPORT_DD_OVERLAP`` (overlap width for Schwarz, default ``1``)
- ``SFINCS_JAX_TRANSPORT_DD_AUTO_MIN`` (optional auto-enable threshold in
  ``TRANSPORT_PRECOND=auto``; default ``0`` disables auto path)

**FP angular Fourier preconditioner (experimental).**

``SFINCS_JAX_TRANSPORT_PRECOND=fp_tzfft`` targets stiff FP
high-collisionality transport matrix solves in three-dimensional geometry.  For
each angular Fourier mode :math:`(k_\theta, k_\zeta)`, it builds a dense block
over Legendre mode, species, and speed:

.. math::

   M_{k_\theta,k_\zeta}
   \approx C_\mathrm{FP}
          + I
          + \widehat{v_\parallel \mathbf{b}\cdot\nabla}_{k_\theta,k_\zeta}
          + \widehat{\mu\,\mathbf{b}\cdot\nabla B\,\partial_{v_\parallel}}_{k_\theta,k_\zeta}
          + \widehat{\mathbf{v}_E\cdot\nabla}_{k_\theta,k_\zeta}.

The FP collision block :math:`C_\mathrm{FP}` is kept dense in species and speed
for each Legendre mode.  Streaming, mirror, and optional :math:`E\times B`
terms use flux-surface-averaged symbols so the angular part diagonalizes by
FFT. The apply stage FFTs the residual over ``(theta,zeta)``, multiplies by the
cached inverse block for each Fourier mode, and inverse-FFTs the correction.

This path is deliberately opt-in. It reduced the reduced W7-X FP RHS2
pre-rescue residual by several orders of magnitude in the benchmark harness, but
the full W7-X high-``nu'`` point still requires the explicit sparse-direct
rescue to pass the strict residual gate. Use ``fp_tzfft`` for candidate
benchmarking before enabling it in a widened publication scan.

Controls:

- ``SFINCS_JAX_TRANSPORT_FP_TZFFT_MAX_MB`` (inverse-table memory cap; default
  ``384`` MB)
- ``SFINCS_JAX_TRANSPORT_FP_TZFFT_REG`` (diagonal regularization; default
  ``1e-10``)
- ``SFINCS_JAX_TRANSPORT_FP_TZFFT_PINV_RCOND`` (pseudo-inverse cutoff; default
  ``1e-12``)

**FP Fourier line preconditioner (bounded native candidate).**

``SFINCS_JAX_TRANSPORT_PRECOND=fp_tzfft_line`` is a lower-memory native
alternative to the dense ``fp_tzfft`` inverse table. Instead of storing one dense
inverse over all :math:`(L,s,x)` unknowns for each Fourier mode, it keeps the
full FP block over :math:`(s,x)` for each Legendre row and solves the
block-tridiagonal Legendre residual equation with block Thomas factors:

.. math::

   A_l^\star = A_l - L_l (A_{l-1}^\star)^{-1} U_{l-1},
   \qquad
   M_{k_\theta,k_\zeta}^{-1} r
   = \operatorname{Thomas}\{L_l,A_l,U_l\}_{l=0}^{N_\xi-1}.

Here :math:`A_l` contains the FP species/speed block plus the identity and
collision diagonal, while :math:`L_l` and :math:`U_l` are the
flux-surface-averaged streaming/mirror links diagonalized in
``(theta,zeta)`` Fourier space. Storage scales like
``Ntheta * Nzeta * Nxi * (Nspecies * Nx)^2`` instead of
``Ntheta * Nzeta * (Nxi * Nspecies * Nx)^2``, so the path avoids the full
SuperLU-style factor fill and the dense ``fp_tzfft`` table.

Current gate status: the factor is implemented and tested as a numerical
preconditioner. On the small FP RHSMode=2 LHD fixture, one application reduces
the true residual by more than ``1e6`` relative to ``sxblock``. On a bounded
``13 x 17 x 30 x 4`` geometry-scheme-2 FP transport probe, however,
``fp_tzfft_line`` and ``sxblock`` still missed the strict true-residual solve
gate without sparse/direct rescue. Therefore ``auto`` does **not** select this
path by default. The next required algorithmic step is a coupled
constraint/source-moment Schur correction on top of the line factor.

``SFINCS_JAX_TRANSPORT_PRECOND=fp_tzfft_line_schur`` adds that first coupled
tail correction. It builds normalized source/constraint-tail response columns
``Z`` and solves a compact tail-restricted true-action equation

.. math::

   (R A Z)c = R\,(r - A M_\mathrm{line}^{-1} r),
   \qquad
   M^{-1}r = M_\mathrm{line}^{-1}r + Zc,

where ``R`` restricts to the source/constraint tail rows. This closes the
source-tail residual in the small FP fixture, including synthetic unit tail
loads, while keeping the acceptance criterion on the full true residual.
However, bounded geometry-scheme-2 probes still retry without the preconditioner
and land at the same kinetic residual as the line-only path. The Schur
candidate is therefore retained as an opt-in diagnostic layer, not a production
default.

Diagnostic variants that restrict the residual to low-order Galerkin moment
directions (``SFINCS_JAX_TRANSPORT_FP_TZFFT_LINE_SCHUR_RESTRICTION=galerkin``
or ``tail_galerkin``) were also tested on bounded geometry-scheme-2 FP probes.
They did not improve the strict true residual and still fell back to the
no-preconditioner result. This points to the remaining bottleneck being the
kinetic FP/streaming residual equation itself, not only the source/constraint
tail coupling.

``SFINCS_JAX_TRANSPORT_PRECOND=fp_local_geom_line`` is a second kinetic
diagnostic candidate. It keeps the local, non-averaged mirror geometry in a
real-space Legendre block-Thomas factor at each ``(theta,zeta)`` point. This
was the expected next low-memory approximation after the flux-surface-averaged
Fourier line factor. The checked tiny FP fixture, however, shows that this
local-only factor amplifies the full kinetic residual and is not viable as a
standalone or additive correction. It remains available only for diagnostic
experiments.

``SFINCS_JAX_TRANSPORT_PRECOND=fp_structured_fblock_lu`` is a stronger
diagnostic baseline. It reuses the migrated structured f-block assembler and
factors the kinetic FP block as a host sparse matrix, retaining the full
non-averaged collisionless streaming/mirror geometry and FP collision couplings.
The full transport residual gate remains unchanged because the factor leaves
the source/constraint tail to the outer Krylov solve. On the tiny LHD FP
fixture, a stabilized factor reduces the kinetic one-apply residual below
``1e-8`` relative. On the bounded ``9 x 11 x 16 x 4`` geometry-scheme-2
one-RHS probe, it reduces the solve residual from the no-preconditioner plateau
(``4.68e-1`` relative) to ``1.32e-9`` relative in ``63.3 s``. The next
``13 x 17 x 30 x 4`` rung did not complete within the bounded runtime budget,
so this path is not promoted to ``auto`` or public benchmark plots. It is a
useful correctness baseline for the lower-memory native coupled block factor
that must replace the host exact factor before production-floor promotion.

``SFINCS_JAX_TRANSPORT_PRECOND=fp_xblock_tz_lu`` is that lower-memory coupled
block-factor candidate. It factors independent sparse blocks for each
``(species,x)`` pair, retaining the non-averaged angular streaming/mirror
geometry and selected drift terms over the coupled ``(L,theta,zeta)`` unknowns
without forming the global kinetic f-block LU. On the bounded
``13 x 17 x 30 x 4`` geometry-scheme-2 FP transport probe with sparse/direct
and dense rescue disabled, the candidate passes a strict ``1e-10`` relative
gate for RHS3 with ``REG=1e-13``:

.. code-block:: text

   whichRHS=3 residual=1.128017e-10
   rhs_norm=1.339822e+00
   relative_residual=8.419155e-11
   elapsed=7.6 s

The same regularization is not sufficient for RHS1 on the all-RHS gate, while
``REG=3e-13`` is better for RHS1/RHS2 but misses RHS3. Therefore the path is
implemented and test-covered, but not promoted to ``auto`` until the coupled
source/tail/kinetic coarse correction closes all three RHS columns under one
policy.

``SFINCS_JAX_TRANSPORT_PRECOND=fp_xblock_tz_lu_schur`` adds that first
source/tail closure. Unlike the older tail-only variant, the current default
``SFINCS_JAX_TRANSPORT_FP_XBLOCK_TZ_LU_SCHUR_RESTRICTION=tail_galerkin`` solves
a compact residual equation using both source/constraint tail rows and kinetic
moment test directions:

.. math::

   (R A Z)c = R\,(r - A M_\mathrm{xblock}^{-1} r),
   \qquad
   M^{-1}r = M_\mathrm{xblock}^{-1}r + Zc.

Here :math:`Z` contains normalized tail response and kinetic moment correction
directions, and :math:`R` contains the selected tail/moment test rows. On the
tiny FP fixture this reduces the tail residual from ``1.95e-12`` to
``4.25e-15`` relative, confirming that the source/tail restriction is active.
The diagnostic residual-coarse controls
``SFINCS_JAX_TRANSPORT_FP_XBLOCK_TZ_LU_SCHUR_KINETIC_RESIDUAL=1`` and
``SFINCS_JAX_TRANSPORT_FP_XBLOCK_TZ_LU_SCHUR_RHS_RESIDUAL=1`` add low-order
kinetic residual-error directions and residual-correction columns built from
the actual RHSMode=2/3 transport drives. With ``DAMPING=0.25`` they reduce the
tiny-fixture one-apply full residual for all three transport RHS columns.
However, the bounded ``13 x 17 x 30 x 4`` geometry-scheme-2 all-RHS gate still
failed promotion: RHS1 took ``375.227 s``, the preconditioned branch produced a
``nan`` residual and retried without the preconditioner, and the strict gate
aborted at residual ``1.811554e-04`` with relative residual ``5.711080e-01``.
The next production step is therefore a stronger kinetic residual equation
inside the local x-block factor or a genuinely coupled coarse residual over the
dominant kinetic subspace, not more tail-Schur damping.

``SFINCS_JAX_TRANSPORT_PRECOND=fp_fortran_reduced_lu`` implements that stronger
global route as a PETSc-like transport preconditioner. It is now the default
``auto`` candidate for eligible non-Phi1 RHSMode=2/3 full-FP transport cases
unless explicitly disabled. It keeps the true SFINCS-JAX matrix-free operator as
:math:`A`, but materializes a separate reduced sparse :math:`P` and factors it
once for all RHSMode=2/3 drives:

.. math::

   A x_i = b_i,\qquad M^{-1}r \approx P^{-1}r.

This mirrors the Fortran v3 PETSc call pattern ``KSPSetOperators(A, P)``.  The
Fortran source uses ``whichMatrix=1`` for the true Jacobian and ``whichMatrix=0``
for the preconditioner matrix, then reuses one LU-factorized preconditioner
across the transport RHS columns.  A fresh bounded profile of SFINCS Fortran v3
on the same ``13 x 17 x 30 x 4`` geometry-scheme-2 input showed:

- true matrix: ``14588 x 14588`` with ``288066`` nonzeros,
- preconditioner matrix: ``263756`` nonzeros,
- PETSc solver: GMRES with ``PCLU`` and MUMPS,
- MUMPS factor entries: ``7204448``,
- MUMPS effective factor memory: ``68 MB`` and allocated memory ``100 MB``,
- PETSc ``PCSetUp``: ``0.187 s``, ``KSPSolve`` over three RHS columns:
  ``0.539 s``, and process wall time: ``1.25 s`` with ``247 MB`` max RSS.

A matching reduced geometry-scheme-11 Fortran v3 profile confirms the same
algorithmic pattern on the 3D FP case: PETSc uses the true Jacobian
(``59526`` nonzeros) for Krylov residuals and the smaller ``whichMatrix=0``
preconditioner (``49320`` nonzeros) for ``PCLU``/MUMPS. MUMPS reported
``N=2900``, ``NNZ=49320``, maximum frontal size ``419``, and the first
transport RHS dropped from ``1.90e-4`` to ``1.58e-10`` in about 50 Krylov
iterations. The measured solve phase was ``0.067 s`` and max RSS was about
``130 MB``. This is the reference behavior the JAX-native production path is
trying to reproduce: lower-NNZ reduced preconditioner, symbolic analysis reused
across RHS columns, and strict residuals always evaluated with the true
operator.

The local Fortran source audit is also important for interpreting failed
low-memory replacements: ``populateMatrix(..., whichMatrix=0)`` still retains
the source and constraint couplings for ``constraintScheme=1`` and
``constraintScheme=2``.  It does not merely factor independent kinetic blocks.
For ``constraintScheme=1`` the particle/energy source amplitudes enter the
L=0 kinetic rows and the density/pressure moment rows test flux-surface
averages.  For ``constraintScheme=2`` each retained speed has a source/constraint
row pair.  Robust Fortran runs therefore rely on a sparse factor of this coupled
reduced matrix, with the true residual evaluated against ``whichMatrix=1``.

The relevant external solver documentation supports the same decomposition:
PETSc `KSPSetOperators <https://petsc.org/release/manualpages/KSP/KSPSetOperators/>`__
accepts separate true and preconditioning matrices, PETSc
`PCSetReusePreconditioner <https://petsc.org/main/manualpages/PC/PCSetReusePreconditioner/>`__
documents explicit preconditioner reuse, the
`MUMPS documentation <https://mumps-solver.org/index.php?page=doc>`__
describes sparse direct analysis/factor/solve phases, and
`SuperLU_DIST <https://github.com/xiaoyeli/superlu_dist>`__ documents
distributed sparse LU and triangular solves. SFINCS-JAX uses these ideas as
design targets, but does not require PETSc, MUMPS, or SuperLU_DIST at runtime.

The SFINCS-JAX implementation exposes two useful variants:

- Fortran-structural mode
  ``PRECONDITIONER_X=1``, ``PRECONDITIONER_XI=1``,
  ``KEEPS_THETA_ZETA=1``: materializes ``263754`` nonzeros, factors an
  ``84.8 MB`` SuperLU factor, and passes the bounded gate at relative residuals
  ``3.69e-10``, ``3.83e-10``, and ``4.56e-10`` in ``91.3 s``.  This is the
  closest structural match to the Fortran v3 ``Pmat``, but the current JAX/Python
  Krylov loop retries because the internal target is much tighter than the
  Fortran input ``solverTolerance=1e-6``.
- Direct term-level ``Pmat`` emission is enabled by default for this
  path when the system is non-Phi1 RHSMode=2/3 FP transport and the active
  unknowns preserve complete zeta blocks plus the complete source/constraint
  tail. It emits the structured kinetic block and analytic source/moment rows
  directly as CSR, avoiding the older pattern-color matrix-free probing step.
  Set ``SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_DIRECT=0`` to force the
  conservative pattern-probe fallback.
- Strong coupled mode, the automatic default for this candidate:
  ``PRECONDITIONER_X=0``, ``PRECONDITIONER_XI=0``,
  ``KEEPS_THETA_ZETA=1``. With direct term-level Pmat emission, active-DOF
  reduction, exact LU, and dense/sparse-direct rescue disabled, it passes both
  bounded geometry-scheme-2 and geometry-scheme-11 all-RHS gates. The bounded
  geometry-scheme-2 gate emits ``55704`` nonzeros, factors a ``13.984 MB``
  SuperLU factor, and reaches max relative residual ``6.98e-11`` in
  ``5.54 s``. The bounded geometry-scheme-11 gate emits ``59526`` nonzeros,
  factors a ``16.976 MB`` SuperLU factor, and reaches max relative residual
  ``5.07e-13`` in ``4.86 s``. RCM symbolic analysis reduces the geom2
  bandwidth/profile from ``2531/2007237`` to ``638/1028290`` and the geom11
  bandwidth/profile from ``2899/2149494`` to ``488/880579``. This closes the
  bounded FP RHSMode=2/3 preconditioner gate, so the exact direct-LU variant is
  promoted into ``auto`` with strict residual admission and memory caps. The
  lower-memory symbolic/native variants remain diagnostic because
  production-floor memory and Krylov gates are still being measured honestly.

On the production-floor ``25 x 51 x 100`` FP transport decks, direct emission
removes the setup bottleneck but does not yet close the full production solve
gate.  The geometry-scheme-11 active structural CSR now emits in about ``8.7 s``
instead of the previous ``287.6 s`` pattern-color materialization, with the same
``8.68e6`` actual nonzeros and a ``~406 MB`` ILU factor estimate.  The first-RHS
Krylov phase still needs a stronger lower-fill/coarse correction before this
preconditioner can be promoted to ``auto``.

The sparse-direct transport route also has a direct active true-operator
emitter for non-Phi1 RHSMode=2/3 FP systems.  This emits the active operator
used by the residual gate, not the reduced preconditioner matrix:

.. math::

   A_{\mathcal A\mathcal A} =
   \begin{bmatrix}
     K_{\mathcal A\mathcal A} & B_{\mathcal A t}\\
     C_{t\mathcal A} & D_{tt}
   \end{bmatrix}.

Here :math:`\mathcal A` is the active kinetic set and :math:`t` is the
source/constraint tail.  On the bounded geometry-scheme-11 gate
(``13 x 17 x 30 x 4``), this direct true-operator path emitted ``288064``
nonzeros in ``0.76 s``, factored an exact LU in ``2.70 s``, and reached true
relative residual ``1.7e-10`` in ``7.3 s`` with ``1.88 GB`` max RSS.  On the
production geometry-scheme-11 gate (``25 x 51 x 100 x 6``), direct true-operator
emission stayed cheap (``10.12e6`` nonzeros, ``82.8 MB`` CSR estimate,
``14.1-14.5 s`` build), but monolithic LU did not finish inside ``600 s`` and
global ILU did not pass the true residual gate.  Therefore this exact emitter is
infrastructure for the next block/reuse preconditioner, not a production default
by itself.

The direct-active true-operator emitter is also now paired with a reusable
symbolic block/coarse factor layer in
``sfincs_jax.problems.transport_matrix.linear_system``. The layer separates symbolic block
ordering over active kinetic unknowns, numerical block inverse plus
source/constraint Schur construction, and setup-time true-residual admission
against the same active operator used by the residual gate. This mirrors the
separation between sparse-direct symbolic analysis, numeric factorization, and
solver admission in PETSc/MUMPS/SuperLU_DIST, while keeping the implementation
Python/JAX-native and dependency-light.

The first direct-active block/reuse probe is available as an opt-in diagnostic:

.. math::

   S = D - C M_K^{-1} B,\qquad
   M^{-1}r =
   \begin{bmatrix}
     M_K^{-1}r_K - M_K^{-1}B S^{-1}(r_t - C M_K^{-1}r_K)\\
     S^{-1}(r_t - C M_K^{-1}r_K)
   \end{bmatrix}.

Here :math:`M_K^{-1}` is built from independent kinetic blocks of the same exact
active operator and :math:`S` closes the retained source/constraint tail.
Select it with
``SFINCS_JAX_TRANSPORT_PRECOND=fp_direct_active_block_schur``. This path is
test-covered and useful for residual-equation experiments, but it is **not** a
production default. Setup admission now requires both a true residual below the
configured gate and a material improvement over the identity preconditioner. On
reduced geometry-scheme-11, both zeta-line and angular-plane symbolic layouts
are rejected before Krylov with deterministic probe metrics
``max_rel=2.05e3`` and ``min_improvement=8.14e-2``. The fallback dense rescue
then solves the reduced fixture cleanly with relative residual ``4.1e-13``.
This result is intentionally negative: simple local active blocks are too weak,
so the next production architecture must retain more of the Fortran-style
reduced preconditioner matrix and reuse symbolic sparse metadata, rather than
factor independent local true-operator blocks.

The stronger existing ``fp_xblock_tz_lu`` and ``fp_xblock_tz_lu_schur`` paths
remain the current bounded RHSMode=2/3 FP candidates. On the reduced
geometry-scheme-11 one-RHS gate they reached true residuals near ``1e-13`` in
about ``2.1-2.3 s``. The production-floor ``25 x 51 x 100 x 6`` probe with
``fp_xblock_tz_lu_schur`` completed setup but did not complete the first RHS
within a 10-minute budget, so it also remains opt-in pending a stronger
reusable block/coarse factor.

Controls:

- ``SFINCS_JAX_TRANSPORT_PRECOND=fp_tzfft_line`` (or aliases
  ``fp_streaming_line``, ``fp_block_thomas``, ``fp_line``) forces the candidate.
- ``SFINCS_JAX_TRANSPORT_FP_TZFFT_LINE_AUTO=1`` allows ``auto`` to try the
  candidate for benchmark campaigns. It is disabled by default.
- ``SFINCS_JAX_TRANSPORT_FP_TZFFT_LINE_AUTO_MIN`` controls the auto-size floor
  when the candidate is explicitly enabled (default ``50000`` unknowns).
- ``SFINCS_JAX_TRANSPORT_FP_TZFFT_LINE_MAX_MB`` caps the factor table memory
  estimate (default ``2048`` MB).
- ``SFINCS_JAX_TRANSPORT_FP_TZFFT_LINE_DTYPE`` can force ``float32`` or
  ``float64`` factor storage.
- The line factor is a left preconditioner. User-forced
  ``SFINCS_JAX_TRANSPORT_PRECONDITION_SIDE=right`` is overridden to ``left`` for
  this candidate so the JAX block-Thomas scan path is not used through an
  unsupported transpose. Acceptance still uses the full operator residual.
- ``SFINCS_JAX_TRANSPORT_FP_TZFFT_LINE_REG`` and
  ``SFINCS_JAX_TRANSPORT_FP_TZFFT_LINE_PINV_RCOND`` control setup
  regularization and pseudo-inverse fallback.
- ``SFINCS_JAX_TRANSPORT_PRECOND=fp_fortran_reduced_lu`` forces the global
  reduced sparse-factor transport preconditioner.  Aliases
  ``fp_global_fortran_reduced_lu``, ``fp_petsc_like_lu``, and
  ``fp_reduced_pmat_lu`` are accepted.
- ``SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_AUTO`` controls ``auto``
  selection for this candidate. It is enabled by default for eligible non-Phi1
  RHSMode=2/3 FP transport cases; set it to ``0``, ``false``, ``no``, or ``off``
  to disable. Explicitly forced FP candidates such as ``fp_tzfft_line`` still
  override this default unless this variable is set to ``1`` explicitly.
- ``SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_PRECONDITIONER_X``,
  ``..._PRECONDITIONER_XI``, ``..._PRECONDITIONER_SPECIES``, and
  ``..._PRECONDITIONER_X_MIN_L`` control the Fortran-style kinetic reduction.
  The opt-in default uses ``X=0`` and ``XI=0`` for the stronger bounded gate;
  set ``X=1`` and ``XI=1`` for the closest Fortran-structural ``Pmat``.
- ``SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_KEEPS_THETA_ZETA`` defaults to
  ``1``, matching Fortran v3 ``preconditioner_theta=0`` and
  ``preconditioner_zeta=0``.  Setting it to ``0`` is a diagnostic angular-drop
  mode and is not promoted.
- ``SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_FACTOR``,
  ``..._FACTOR_DTYPE``, ``..._FACTOR_MAX_MB``, and ``..._SHIFT`` control host
  sparse factor kind, factor precision, factor memory guard, and diagonal shift.
  Supported diagnostic factors are ``lu``, ``ilu``, ``jacobi``,
  ``symbolic_block_lu``, ``symbolic_block_lu_coarse``, and
  ``symbolic_block_schur_lu``, ``symbolic_superblock_lu``, and
  ``symbolic_nd_frontal_schur_lu``.  The symbolic options are native
  lower-memory block/coarse/Schur/frontal experiments and are not automatic
  defaults unless they pass setup-time admission.
- ``SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ORDERING`` controls the
  reusable symbolic analysis ordering metadata. ``mumps_like`` is the default:
  it applies a bounded native nested-dissection-style graph ordering that
  mirrors the role of SCOTCH/PT-SCOTCH/ParMETIS/METIS ordering in the
  PETSc+MUMPS/SuperLU_DIST SFINCS v3 path without adding those packages as
  runtime dependencies.  Aliases ``nested_dissection``, ``scotch``,
  ``ptscotch``, ``parmetis``, and ``metis`` select the same native ordering.
  ``rcm`` selects reverse Cuthill-McKee, and ``natural`` disables structural
  reordering in the analysis report.
- ``SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_BLOCK_SIZE`` controls
  the bounded block plan recorded by the symbolic metadata and used by the
  opt-in ``symbolic_block_lu`` numeric factor path.
- ``SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_AUTO_EXACT_RESCUE_MAX_FACTOR_ENTRIES``
  caps automatic exact-LU rescue by estimated multifrontal factor entries, in
  addition to the existing memory and active-size caps.  This prevents
  production-size cases from spending many minutes in a single-core SuperLU
  setup merely because the memory estimate fits; set it to ``0`` only for an
  explicit diagnostic run where that cost is acceptable.
- ``SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_NUMERIC_PARALLEL_WORKERS``
  controls bounded native parallel numeric setup for symbolic factors.  The
  direct reduced ``whichMatrix=0`` Pmat still uses the same symbolic metadata
  and strict true-residual admission, but independent ``symbolic_superblock_lu``
  superblocks and root children in ``symbolic_nd_frontal_schur_lu`` can be
  factored concurrently before chunked Schur updates are applied.  The generic
  explicit-sparse equivalent is
  ``SFINCS_JAX_EXPLICIT_SPARSE_SYMBOLIC_NUMERIC_PARALLEL_WORKERS``.
- ``SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ND_COMPRESS_UPDATES``
  enables the opt-in BLR/HSS-style separator-update path for
  ``symbolic_nd_frontal_schur_lu``.  Each child-to-separator RHS chunk is
  compressed with an SVD, only the retained basis is solved through the child
  factor, and the separator update is stored as a low-rank ``U V^T``
  contribution to the Schur operator.  The same setup-time true-residual
  admission gate remains mandatory before Krylov can use the factor.
- ``SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ND_PARALLEL_UPDATE_WORKERS``
  controls opt-in parallel execution of separator-update chunks.  This is
  intentionally separate from ``..._SYMBOLIC_NUMERIC_PARALLEL_WORKERS`` because
  production profiles show that sparse triangular solves and BLAS calls can
  contend when too many separator chunks are threaded.  The generic
  explicit-sparse equivalents are
  ``SFINCS_JAX_EXPLICIT_SPARSE_SYMBOLIC_ND_COMPRESS_UPDATES`` and
  ``SFINCS_JAX_EXPLICIT_SPARSE_SYMBOLIC_ND_PARALLEL_UPDATE_WORKERS``.
  These BLR/HSS and nested-dissection controls are intentionally not promoted
  into ``auto`` for the current release: production ``geom11`` probes still
  reject the path on setup-time grounds before admission, so this remains a
  deferred optimization lane rather than a release-blocking correctness issue.
- ``SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_BLOCK_OVERLAP``
  extends each symbolic block by a fixed number of neighboring unknowns before
  factorization and restricts the local solution back to the owned block. This
  is an additive-Schwarz-style diagnostic for retaining boundary couplings.
  The default is ``0``.
- ``SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_COARSE_MAX_COLS``,
  ``..._SYMBOLIC_COARSE_PROBE_COLS``,
  ``..._SYMBOLIC_COARSE_DAMPING``, and
  ``..._SYMBOLIC_COARSE_REG_REL`` control the
  ``symbolic_block_lu_coarse`` residual equation. The coarse basis contains
  sparse block-indicator columns plus optional deterministic residual-derived
  columns, then applies
  ``z = M_0 r + omega B (B^T P B + lambda I)^{-1} B^T (r - P M_0 r)``.
- ``SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_PHYSICS_COARSE`` adds a
  SFINCS-specific direct-Pmat coarse basis for this diagnostic factor.  The
  basis includes source/constraint tail units, flux-surface-average source
  shapes, per-speed L=0 constraint modes, low-order density/pressure/flow/heat
  moments, and approximate tail-Schur response modes built from the actual
  emitted direct ``Pmat`` columns and the local factor.  This is a guarded
  experiment, not a default solver policy.
- ``SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_SCHUR_MAX_SEPARATOR_COLS``,
  ``..._SYMBOLIC_SCHUR_BOUNDARY_WIDTH``,
  ``..._SYMBOLIC_SCHUR_HIGH_DEGREE_COLS``, and
  ``..._SYMBOLIC_SCHUR_REG_REL`` control the
  ``symbolic_block_schur_lu`` candidate.  This candidate uses the reusable
  symbolic ordering, selects source/constraint tail variables, high-degree graph
  nodes, symbolic block boundaries, and actual cross-block nonzero endpoints as
  separator candidates, eliminates local interior blocks, and solves the
  retained separator Schur equation.  It is intended as a Python-native
  analogue of sparse-direct analysis/factor/solve separation, but it still must
  pass setup-time admission before Krylov can use it.
- ``SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_MAX_PERMUTATION_SIZE``
  caps expensive structural permutations such as RCM. The default is
  ``250000`` unknowns, so production-floor matrices report natural ordering
  unless a benchmark campaign explicitly raises the cap.
- ``SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ADMISSION`` defaults
  to ``1`` for symbolic factors. It applies deterministic setup probes to the
  actual emitted ``Pmat`` and rejects the factor unless
  ``||P M^{-1} b - b|| / ||b||`` is small and improves over identity. The
  thresholds are controlled by ``..._SYMBOLIC_ADMISSION_MAX_REL``,
  ``..._SYMBOLIC_ADMISSION_MIN_IMPROVEMENT``, and
  ``..._SYMBOLIC_ADMISSION_PROBES``.
- ``SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ADMISSION_RESCUE_LU``
  defaults to ``1``. If a symbolic factor fails setup admission, the same
  emitted direct ``whichMatrix=0`` Pmat is factored with exact native host LU,
  bounded by
  ``SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ADMISSION_RESCUE_LU_MAX_MB``.
  The rescue is accepted only if it passes the same deterministic true-residual
  admission probes. This keeps the lane accurate instead of silently falling
  back to a weak smoother. Exact ``lu`` is the promoted automatic factor kind for
  this route because the current ``ilu`` and symbolic factors are not
  residual-clean enough for public default use.
- Current production evidence: ``symbolic_block_lu`` is guarded but not
  production-ready. Reduced geometry-scheme-2 and geometry-scheme-11 gates
  reject it by setup residual. Bounded overlap probes with overlap ``64`` and
  ``256`` also reject by setup residual, so simple local overlap is not enough.
  The residual-coarse variant with deterministic residual columns is also
  rejected on reduced and production-floor geometry-scheme-2/11 gates. With a
  relaxed 12 GB factor budget, production-floor geometry-scheme-11 reached
  ``6.98 GB`` estimated factor storage but failed admission with
  ``max_rel≈4.32e8``; geometry-scheme-2 reached ``9.80 GB`` estimated factor
  storage and failed admission with ``max_rel≈4.07e8``. Keep both symbolic
  paths opt-in until a physics/field-moment coarse variant passes strict
  residual, runtime, and RSS gates.
- A bounded physics-aware Schur-coarse retry on the reduced geometry-scheme-2
  and geometry-scheme-11 FP decks built ``18`` and ``16`` SFINCS-specific
  source/moment/tail-response columns, respectively, but still failed setup
  admission before Krylov. Geometry-scheme-2 rejected with
  ``max_rel=1.614e11`` and geometry-scheme-11 rejected with
  ``max_rel=1.645e10``.  The exact direct-Pmat LU path was rechecked after the
  same code change and remains residual-clean: geometry-scheme-2 all-RHS
  ``max_rel=6.98e-11`` in ``4.10 s`` and geometry-scheme-11 all-RHS
  ``max_rel=5.07e-13`` in ``4.32 s``.  Therefore this small coarse correction is
  retained only as tested infrastructure; it is not a production replacement
  for the Fortran/PETSc sparse factor.
- A bounded native ``symbolic_block_schur_lu`` retry now uses actual graph
  cross-block nonzeros to choose separator candidates.  It improved the reduced
  setup residuals but remains non-promotable by itself: geometry-scheme-2 only
  passed admission when the separator cap retained most of the active system
  (``sep=2048``, ``max_rel=2.743e-5``, ``26.17 MB``), while
  geometry-scheme-11 still failed all tested separator/block-size settings.
  With the new admission rescue enabled, both reduced all-RHS gates are clean:
  geometry-scheme-2 reaches relative residuals ``6.98e-11``, ``4.03e-11``, and
  ``7.66e-13`` in ``3.26 s`` after exact-Pmat LU rescue
  (``13.984 MB``), and geometry-scheme-11 reaches ``5.07e-13``,
  ``4.00e-13``, and ``4.14e-13`` in ``3.21 s`` after exact-Pmat LU rescue
  (``16.976 MB``).  This is the current residual-clean bounded path; a genuine
  lower-memory production replacement still needs a stronger hierarchical
  factor retaining dominant kinetic off-diagonal couplings directly.
- ``SFINCS_JAX_TRANSPORT_PRECOND=fp_tzfft_line_schur`` (or
  ``fp_line_schur``) forces the tail-Schur diagnostic candidate.
- ``SFINCS_JAX_TRANSPORT_FP_TZFFT_LINE_SCHUR_AUTO=1`` allows ``auto`` to try
  the Schur candidate in benchmark campaigns. It is disabled by default.
- ``SFINCS_JAX_TRANSPORT_FP_TZFFT_LINE_SCHUR_MAX_COLS``, ``_MAX_MB``,
  ``_DTYPE``, ``_REG``, ``_DAMPING``, and ``_CORRECTION_REL_MAX`` control the
  compact Schur basis, storage cap, pseudoinverse cutoff, and finite-output
  correction limiter.
- ``SFINCS_JAX_TRANSPORT_FP_TZFFT_LINE_SCHUR_RESTRICTION`` can be ``tail``
  (default), ``galerkin``, or ``tail_galerkin`` for diagnostic probes.
- ``SFINCS_JAX_TRANSPORT_PRECOND=fp_local_geom_line`` (or ``fp_geom_line``)
  forces the non-averaged local mirror-geometry diagnostic candidate.
- ``SFINCS_JAX_TRANSPORT_FP_LOCAL_GEOM_LINE_AUTO=1`` allows ``auto`` to try
  this candidate in benchmark campaigns. It is disabled by default.
- ``SFINCS_JAX_TRANSPORT_FP_LOCAL_GEOM_LINE_MAX_MB``, ``_DTYPE``, ``_REG``, and
  ``_PINV_RCOND`` control storage and setup.
- ``SFINCS_JAX_TRANSPORT_PRECOND=fp_structured_fblock_lu`` (or
  ``fp_fblock_lu``) forces the structured kinetic f-block factor diagnostic.
- ``SFINCS_JAX_TRANSPORT_FP_STRUCTURED_FBLOCK_LU_AUTO=1`` allows ``auto`` to
  try this candidate in benchmark campaigns. It is disabled by default.
- ``SFINCS_JAX_TRANSPORT_FP_STRUCTURED_FBLOCK_LU_MAX_MB`` caps assembled CSR
  storage, ``_FACTOR_MAX_MB`` caps post-factor storage, ``_REG`` controls
  diagonal stabilization, and ``_FACTOR`` can be ``lu``, ``ilu``, or ``jacobi``
  for diagnostics.
- ``SFINCS_JAX_TRANSPORT_PRECOND=fp_xblock_tz_lu`` (or ``fp_xblock_lu`` /
  ``fp_angular_xblock_lu``) forces the per-``(species,x)`` coupled
  ``(L,theta,zeta)`` sparse factor.
- ``SFINCS_JAX_TRANSPORT_FP_XBLOCK_TZ_LU_AUTO=1`` allows ``auto`` to try this
  candidate in benchmark campaigns. It is disabled by default.
- ``SFINCS_JAX_TRANSPORT_FP_XBLOCK_TZ_LU_MAX_MB`` caps local-block CSR storage,
  ``_FACTOR_MAX_MB`` caps local-factor storage, ``_REG`` controls diagonal
  stabilization, and ``_FACTOR`` can be ``lu``, ``ilu``, or ``jacobi``.
- ``SFINCS_JAX_TRANSPORT_PRECOND=fp_xblock_tz_lu_schur`` (or
  ``fp_xblock_schur`` / ``fp_xblock_lu_schur``) forces the source/tail Schur
  overlay on top of ``fp_xblock_tz_lu``.
- ``SFINCS_JAX_TRANSPORT_FP_XBLOCK_TZ_LU_SCHUR_AUTO=1`` allows ``auto`` to try
  this Schur overlay in benchmark campaigns. It is disabled by default.
- ``SFINCS_JAX_TRANSPORT_FP_XBLOCK_TZ_LU_SCHUR_MAX_COLS``, ``_MAX_MB``,
  ``_DTYPE``, ``_REG``, ``_DAMPING``, ``_CORRECTION_REL_MAX``, and
  ``_RESTRICTION`` control the compact residual equation. ``_RESTRICTION`` can
  be ``tail``, ``galerkin``, or ``tail_galerkin``; ``tail_galerkin`` is the
  default for this candidate. ``_KINETIC_RESIDUAL=1`` and ``_RHS_RESIDUAL=1``
  enable the opt-in residual-coarse enrichment used in the bounded
  non-promotion probe above.
- ``SFINCS_JAX_TRANSPORT_PRECOND=fp_direct_active_block_schur`` forces the
  experimental direct-active kinetic block plus tail-Schur
  preconditioner. Aliases include ``fp_active_true_block`` and
  ``fp_true_block_lu``.
- ``SFINCS_JAX_TRANSPORT_FP_DIRECT_ACTIVE_BLOCK_SCHUR_MAX_MB`` caps the
  estimated active-block inverse plus Schur storage (default ``2048`` MB).
- ``SFINCS_JAX_TRANSPORT_FP_DIRECT_ACTIVE_BLOCK_SCHUR_BLOCK_KIND`` chooses the
  symbolic kinetic layout. Supported diagnostic values are ``zeta_line``,
  ``theta_line``, ``angular_plane``, and ``ell_band``.
- ``SFINCS_JAX_TRANSPORT_FP_DIRECT_ACTIVE_BLOCK_SCHUR_ELL_BLOCK`` controls the
  number of complete angular planes per ``ell_band`` block.
- ``SFINCS_JAX_TRANSPORT_FP_DIRECT_ACTIVE_BLOCK_SCHUR_MAX_BLOCK`` caps each
  dense symbolic block before factorization.
- ``SFINCS_JAX_TRANSPORT_FP_DIRECT_ACTIVE_BLOCK_SCHUR_ADMISSION`` enables the
  setup-time true-residual admission probe (default enabled).
- ``SFINCS_JAX_TRANSPORT_FP_DIRECT_ACTIVE_BLOCK_SCHUR_ADMISSION_MAX_REL``,
  ``..._ADMISSION_MIN_IMPROVEMENT``, and ``..._ADMISSION_PROBES`` control the
  strict admission gate. A candidate must pass the true-residual gate and
  improve materially over identity before it is used.
- ``SFINCS_JAX_TRANSPORT_FP_DIRECT_ACTIVE_BLOCK_SCHUR_FACTOR_DTYPE`` selects
  ``float64`` or ``float32`` block factors. ``float64`` is the default because
  this path is currently an accuracy diagnostic.

**JAX sparse Jacobi (optional).**

``SFINCS_JAX_TRANSPORT_PRECOND=sparse_jax`` builds a sparsified operator and applies
a few weighted Jacobi sweeps in JAX. This can reduce memory relative to dense
preconditioners while staying differentiable. Controls mirror the RHSMode=1
``sparse_jax`` options (``SFINCS_JAX_TRANSPORT_SPARSE_JAX_*`` and
``SFINCS_JAX_TRANSPORT_SPARSE_DROP_*``).

**Explicit sparse-helper rescue for hard high-nu transport.**

The full W7-X FP high-``nu'`` point is an example where the matrix-free Krylov
routes can stall even with useful preconditioners. The production executable
path can therefore opt into a bounded host sparse-LU rescue:

.. math::

   A X = B,\qquad B = [b_1,\ b_2,\ b_3],

where the transport operator :math:`A` is unchanged across the three
``whichRHS`` drives. ``sfincs_jax`` now exploits that structure in two ways:

- the explicit sparse helper assembles column blocks from local basis matrices
  instead of materializing a full identity matrix,
- the assembled CSR operator and sparse-LU factorization are cached inside the
  transport solve and reused for the later RHS solves when the operator
  signature is unchanged.

This keeps the exact matrix-free operator as the final residual gate: the host
sparse residual is never trusted by itself for accepting a hard transport solve.
The route is non-differentiable and intended for the CLI/executable path, not
for end-to-end autodiff workflows.

Implementation:

- block-basis materialization and factorization helpers:
  ``sfincs_jax.solvers.explicit_sparse.build_operator_from_matvec`` and
  ``factorize_host_sparse_operator``;
- RHSMode=2/3 sparse-direct reuse:
  ``sfincs_jax.v3_driver.solve_v3_transport_matrix_linear_gmres``.

Controls:

- ``SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_MAX`` enables the bounded sparse-direct
  first attempt/rescue when the active system size is below the cap;
- ``SFINCS_JAX_TRANSPORT_SPARSE_FACTOR_DTYPE`` selects host factor precision
  (``float32`` is the measured W7-X high-``nu'`` route);
- ``SFINCS_JAX_TRANSPORT_SPARSE_HELPER_BLOCK_COLS`` tunes sparse-helper
  materialization batch width;
- ``SFINCS_JAX_TRANSPORT_SPARSE_HELPER_DENSE_MAX_MB`` and
  ``SFINCS_JAX_TRANSPORT_SPARSE_HELPER_CSR_MAX_MB`` control storage selection.
- ``SFINCS_JAX_TRANSPORT_SPARSE_PATTERN`` controls the conservative
  sparse-pattern materialization route. In ``auto`` mode, large RHSMode=3
  monoenergetic PAS transport rows use a declared structural pattern instead of
  probing a dense identity matrix.
- ``SFINCS_JAX_TRANSPORT_SPARSE_PATTERN_CSR_MAX_MB`` and
  ``SFINCS_JAX_TRANSPORT_SPARSE_PATTERN_COLOR_BATCH`` bound sparse-pattern CSR
  storage and color-probe batching.
- ``SFINCS_JAX_TRANSPORT_TZFFT_FIRST`` controls the bounded structured
  ``tzfft`` first attempt. In ``auto`` mode this is enabled for explicit
  RHSMode=2/3 mono/PAS transport rows on accelerators when the system size is
  within the production cap.
- ``SFINCS_JAX_TRANSPORT_TZFFT_FIRST_MIN`` /
  ``SFINCS_JAX_TRANSPORT_TZFFT_FIRST_MAX`` bound that automatic structured
  first attempt. The default maximum is ``160000`` active unknowns.
- ``SFINCS_JAX_TRANSPORT_TZFFT_FIRST_RESTART`` /
  ``SFINCS_JAX_TRANSPORT_TZFFT_FIRST_MAXITER`` bound the first Krylov probe
  before sparse-pattern LU rescue is considered. Defaults are ``40`` and ``12``.
- ``SFINCS_JAX_TRANSPORT_FP_DIRECT_ACTIVE_OPERATOR=0`` disables direct active
  true-operator emission for FP transport sparse-direct paths.
- ``SFINCS_JAX_TRANSPORT_FP_DIRECT_ACTIVE_OPERATOR_FACTOR={lu,ilu}`` overrides
  the direct true-operator factor policy. The automatic policy uses exact LU
  for small active systems and ILU for large diagnostics, but production
  evidence so far keeps both global factors out of ``auto`` defaults.

Measured production-floor evidence:

- The ``monoenergetic_geometryScheme11`` row at
  ``25 x 51 x 4 x 100`` previously failed on GPU by entering sparse direct
  without a pattern and attempting a dense ``127501 x 127501`` materialization
  (about ``121 GiB``), then falling back to matrix-free Krylov until timeout.
- Sparse-pattern host LU fixed correctness first: the conservative pattern had
  ``5.67M`` declared nonzeros; color probing built a true CSR with ``3.42M``
  nonzeros in ``7.10 s`` and about ``41.5 MB`` operator storage. Exact LU gave
  residuals ``1.8e-18`` and ``3.6e-15`` but took ``511.1 s`` wall and
  ``25.96 GB`` process RSS, so it is retained as a rescue rather than a default
  performance route.
- The promoted default is now the bounded structured ``tzfft`` first attempt
  with strict true-residual checking. On the same production row it reached
  residuals ``5.3e-18`` and ``1.1e-13`` without invoking sparse LU, completed in
  ``13.6 s`` wall and used ``1.08 GB`` process RSS /
  ``0.52 GB`` incremental RSS. The matched Fortran v3/MUMPS reference took
  ``188.5 s`` and ``3.26 GB`` RSS on the same host. If the structured solve
  misses the true residual gate, the sparse-pattern LU rescue still runs.

**Implementation.**

- ``_build_rhsmode23_sxblock_preconditioner`` in ``sfincs_jax.v3_driver``.
- ``_build_rhsmode23_theta_dd_preconditioner`` / ``_build_rhsmode23_zeta_dd_preconditioner``
  in ``sfincs_jax.v3_driver``.
- ``_build_rhsmode23_theta_schwarz_preconditioner`` /
  ``_build_rhsmode23_zeta_schwarz_preconditioner`` in ``sfincs_jax.v3_driver``.
- ``_build_rhsmode23_fp_tzfft_preconditioner`` in ``sfincs_jax.v3_driver``.
- Controlled by ``SFINCS_JAX_TRANSPORT_PRECOND`` (``auto``, ``sxblock``, ``collision``, etc.).
  ``auto`` picks the collision-diagonal preconditioner for the default BiCGStab transport
  solver and upgrades to species×x blocks for modest FP systems when GMRES is selected.
  For larger FP systems (especially when dense fallbacks are blocked for memory),
  ``auto`` escalates to the matrix-free x-grid multigrid preconditioner (``xmg``).

**Compared to Fortran.**

Fortran often uses PETSc block preconditioners constructed from the assembled matrix.
`sfincs_jax` builds an analytic block preconditioner directly from the FP collision
matrix, avoiding matvec assembly and staying JAX-native.

**Impact.**

Reduces iteration counts in FP transport cases without a heavy preconditioner
build, improving performance in PAS/W7X and FP-heavy benchmarks.

RHSMode=1 preconditioning (matrix-free)
---------------------------------------

`sfincs_jax` includes a family of RHSMode=1 preconditioners to match and extend v3 options:

- **Point-block Jacobi**: local (x,L) blocks at each :math:`(\theta,\zeta)`.
- **Theta-line / Zeta-line / ADI**: 1D line solves across angular dimensions.
- **Theta/Zeta DD + Schwarz**: angular block-Jacobi and overlap-RAS variants
  (``theta_dd``, ``zeta_dd``, ``theta_schwarz``, ``zeta_schwarz``).
- **Species-block (PAS)**: full (x,L,θ,ζ) block per species for strong PAS conditioning.
- **Collision diagonal / xblock / sxblock**: analytic blocks from PAS/FP collisions.
- **Constraint-aware Schur**: enforces constraintScheme=2 source constraints via a
  diagonal or dense Schur complement.

**PAS gauge/constraint projection.** PAS operators admit a nullspace drift in the
flux-surface average. We remove it before Krylov iterations:

.. math::

   f \leftarrow f - \langle f \rangle_{\mathrm{FS}},

applied per species and :math:`x` using the same :math:`(\theta,\zeta)` weights
as diagnostics. This stabilizes PAS tokamak-like cases and pairs with the
``xblock_tz`` preconditioner default.

Implementation: ``sfincs_jax.v3_driver`` (``use_pas_projection`` and
``_project_pas_f``). Control: ``SFINCS_JAX_PAS_PROJECT_CONSTRAINTS`` (auto on for
``N_\zeta=1`` tokamak-like runs **except** ``geometryScheme=1`` analytic tokamak
cases).

**Adaptive PAS smoother gate (standalone helper).** For PAS-heavy runs it is
useful to stop an early smoother pass before it burns time on a bad branch. If
the residual history is :math:`r_0, r_1, \dots, r_k`, define the one-step ratio

.. math::

   \rho_k = \frac{r_k}{r_{k-1}},

and a trailing log-slope over the last :math:`m` ratios,

.. math::

   s_m = \frac{1}{m} \sum_{i=k-m+1}^{k} \log\left(\frac{r_i}{r_{i-1}}\right).

The helper accepts a candidate smoother state when the latest residual is still
improving and stops when either the single-step ratio or trailing log-slope
crosses the configured worsening threshold. This provides a deterministic gate
for the later PAS driver integration without hard-coding case-specific logic.

Implementation: ``sfincs_jax.pas_smoother`` (``append_residual``,
``summarize_residual_history``, ``decide_pas_smoother_action``,
``advance_pas_smoother``, ``adaptive_pas_smoother_allowed``, and
``adaptive_pas_smoother``). The returned metadata includes the residual
history, best-so-far residual, trailing ratio trend, the number of accepted
sweeps, and the stop reason so the driver can decide whether to continue
smoothing or fall back to the next solver stage.

Controls:

- ``PasSmootherConfig.window`` (trailing ratio window, default ``3``)
- ``PasSmootherConfig.accept_ratio`` (minimum ratio for accepting a step, default
  ``1.0``)
- ``PasSmootherConfig.worsen_ratio`` (ratio threshold that triggers early stop,
  default ``1.05``)
- ``PasSmootherConfig.stagnation_ratio`` (window-level stagnation threshold,
  default ``0.995``)
- ``PasSmootherConfig.max_consecutive_increases`` (limit on consecutive worsening
  steps, default ``1``)

**RHSMode=1 angular DD / overlap Schwarz.** ``sfincs_jax`` now exposes
theta/zeta domain-decomposition preconditioners for RHSMode=1:
``SFINCS_JAX_RHSMODE1_PRECONDITIONER=theta_dd`` / ``zeta_dd`` (block-Jacobi) and
``theta_schwarz`` / ``zeta_schwarz`` (overlap-RAS). Controls:

- ``SFINCS_JAX_RHSMODE1_DD_BLOCK_T`` / ``SFINCS_JAX_RHSMODE1_DD_BLOCK_Z``.
- ``SFINCS_JAX_RHSMODE1_DD_OVERLAP`` (or axis-specific ``..._OVERLAP_T/Z``).
- ``SFINCS_JAX_RHSMODE1_SCHWARZ_AUTO_MIN``: auto-select Schwarz on sharded runs
  above the size threshold (default ``4000``).

This keeps preconditioner application local and differentiable while improving
coupling across angular block boundaries relative to pure block-diagonal DD.

**Constraint-aware Schur (constraintScheme=2).** With constraint variables
:math:`c`, the linear system is partitioned as

.. math::

   \begin{bmatrix}
   A_{ff} & A_{fc} \\\\
   A_{cf} & A_{cc}
   \end{bmatrix}
   \begin{bmatrix}
   f \\\\
   c
   \end{bmatrix}
   =
   \begin{bmatrix}
   b_f \\\\
   b_c
   \end{bmatrix}.

The preconditioner uses a Schur complement approximation

.. math::

   S \approx A_{cc} - A_{cf} A_{ff}^{-1} A_{fc},

with diagonal or dense approximations for :math:`S`. This preserves constraint
coupling while improving conditioning in high‑ratio PAS cases.

For x‑block‑diagonal operators (no :math:`x` coupling), :math:`S` is diagonal in the
(species, :math:`x`) constraint index. ``sfincs_jax`` takes advantage of this structure
by recovering the diagonal Schur entries with a **single** base‑preconditioner apply
(injecting all constraint sources at once), rather than constructing the Schur matrix
column‑by‑column. This significantly reduces Schur setup time in tokamak‑like PAS runs
with many :math:`x` points.

Implementation: ``sfincs_jax.v3_driver`` (``_build_rhsmode1_schur_*``).
Controls: ``SFINCS_JAX_RHSMODE1_SCHUR_MODE`` and
``SFINCS_JAX_RHSMODE1_SCHUR_FULL_MAX``. The base preconditioner used inside the
Schur construction can be selected with ``SFINCS_JAX_RHSMODE1_SCHUR_BASE``; the
default ``auto`` path uses a PAS species-block base when the per‑species block
size is modest, then prefers the PAS x-block :math:`(\theta,\zeta)` variant when
the per‑:math:`x` block is still small, and falls back to theta/zeta line bases
otherwise. This avoids dense fallback on PAS stellarator cases without excessive
cost on larger systems.

For PAS cases with ``constraintScheme=2``, ``SFINCS_JAX_RHSMODE1_SCHUR_AUTO_MIN``
can trigger the Schur preconditioner automatically once ``total_size`` exceeds
the threshold (default: ``2500``), which helps HSX-like cases.
See ``docs/references.rst`` for Schur complement references.

For FP-heavy RHSMode=1 systems, the strong-preconditioner fallback is now enabled
automatically once the active system exceeds ``SFINCS_JAX_RHSMODE1_STRONG_PRECOND_MIN``,
so difficult FP cases attempt a stronger angular block preconditioner before dense fallback.
The fallback now uses a residual-ratio gate; tune
``SFINCS_JAX_RHSMODE1_STRONG_PRECOND_RATIO`` (default: ``1e2``) to avoid expensive
fallbacks when the residual is only slightly above target.

For PAS runs that already used a strong base preconditioner (``schur``,
``xblock_tz``, ``xblock_tz_lmax``, ``sxblock_tz``, ``species_block``,
``theta_zeta``, or the ``pas_*`` family), auto mode now skips the extra
strong-preconditioner retry when ``||r|| / target`` is already below
``SFINCS_JAX_PAS_AUTO_STRONG_RATIO`` (default: ``10``). This avoids paying for a
second expensive PAS Krylov cycle in near-converged cases where the base solve
already satisfies the practical parity tolerances.

For **small FP systems**, ``sfincs_jax`` now defaults to a direct dense solve
instead of running GMRES first. This avoids JIT/Krylov overhead in cases where a
direct solve is both faster and more robust for parity. The FP threshold is
``SFINCS_JAX_RHSMODE1_DENSE_FP_CUTOFF`` (default:
``min(SFINCS_JAX_RHSMODE1_DENSE_ACTIVE_CUTOFF, 8000)``). Setting this
environment variable to ``0`` disables the initial dense shortcut on tight-memory
hosts. On GPU/accelerator backends, the default dense shortcut also requires
``active_size >= SFINCS_JAX_RHSMODE1_DENSE_FP_ACCELERATOR_MIN`` (default:
``1000``), because the validated GPU sweep showed tiny FP systems are faster on
the existing matrix-free path while moderate FP systems avoid an expensive
Krylov/preconditioner ladder.

For **small PAS systems** (``constraintScheme=2``), dense solves are enabled up to
``SFINCS_JAX_RHSMODE1_DENSE_PAS_MAX`` (default: ``2500``) to preserve parity, but
larger PAS systems default to Krylov+preconditioning to avoid multi‑GB transient
dense allocations.

When the input requests a fully coupled preconditioner (``preconditioner_species = preconditioner_x = preconditioner_xi = 0``),
``sfincs_jax`` now defaults to the Schur preconditioner for ``constraintScheme=2`` to avoid dense fallbacks while
preserving the constraint coupling. For tokamak-like cases (``N_zeta=1``) with
``|Er|`` below ``SFINCS_JAX_RHSMODE1_SCHUR_ER_ABS_MIN`` (default: ``0``),
the default switches to the cheaper ``xblock_tz`` preconditioner to reduce setup time.
For bounded CPU tokamak PAS+Er branches with magnetic-drift coupling, the default
now also promotes directly to ``xblock_tz`` instead of first forcing ``pas_schur``
when the active system is below
``SFINCS_JAX_RHSMODE1_PAS_TOKAMAK_CPU_XBLOCK_ACTIVE_MAX`` and the angular block
fits inside ``SFINCS_JAX_RHSMODE1_XBLOCK_TZ_MAX``. Set
``SFINCS_JAX_RHSMODE1_SCHUR_TOKAMAK=1`` to force Schur in the no-Er tokamak-like
cases that still use that gate.

For bounded one-GPU analytic tokamak PAS+Er branches, the default takes a
different route: it avoids the expensive ``xblock_tz`` setup and uses
unpreconditioned full-restart GMRES with a tightened tolerance. This was chosen
because the actual ``pas_tokamak_theta`` preconditioner build was not the fast
path in the focused probe, while the tight unpreconditioned solve preserved all
Fortran-output comparisons and reduced the runtime from about ``18.2 s`` to
``3.25 s`` on ``tokamak_1species_PASCollisions_withEr_fullTrajectories``. The
older ``xblock_tz`` GPU route remains available behind an explicit active-size
cap for users who want to benchmark it on a different accelerator.

For bounded near-zero-:math:`E_r` geometryScheme=4 PAS branches, the default now
uses direct top-level ``pas_tz`` instead of the constraint-Schur wrapper. This is
a measured memory policy, not a new physics approximation: the same PAS
angular-block model is used, but the extra Schur applications and buffers are
avoided. On ``geometryScheme4_2species_PAS_noEr`` this preserved all Fortran
output comparisons and reduced focused GPU RSS from about ``2507 MB`` to
``1817 MB`` while also reducing the clean-remote elapsed time from ``5.899 s``
with the disabled Schur route to ``4.774 s``. Set
``SFINCS_JAX_RHSMODE1_GEOM4_PAS_MEMORY_PAS_TZ=0`` to restore the previous Schur
route for comparison.

**Sparse ILU (FP-heavy RHSMode=1).** For FP-heavy RHSMode=1 systems, a PETSc‑like
incomplete factorization is available to avoid dense fallback while retaining
matrix‑free accuracy. We form a sparsified operator :math:`\tilde{A}` and build
an ILU preconditioner :math:`M \approx \tilde{L}\tilde{U}` so that GMRES solves

.. math::

   M^{-1} A x = M^{-1} b,

reducing iterations while keeping the exact operator :math:`A` in the matvec.
When ``SFINCS_JAX_IMPLICIT_SOLVE=1`` (default), the ILU factors are converted to
dense triangular factors and applied with JAX triangular solves to keep
end‑to‑end differentiability. For fully JAX‑native runs (no SciPy), a sparse
Jacobi preconditioner is available that builds a sparsified operator
(:math:`\tilde{A}`) in JAX and applies a few weighted Jacobi sweeps,

.. math::

   x^{(k+1)} = x^{(k)} + \omega D^{-1} (b - \tilde{A} x^{(k)}),

as a differentiable approximation to :math:`\tilde{A}^{-1}`. Explicit solves can
apply SciPy’s sparse ILU and optionally use the sparse operator for matvecs.
References: GMRES [#saad86]_, ILU/Preconditioning surveys [#benzi02]_.

Implementation: ``sfincs_jax.v3_driver`` (``_build_sparse_ilu_from_matvec`` and
the RHSMode=1 sparse fallback). Controls:

- ``SFINCS_JAX_RHSMODE1_SPARSE_PRECOND`` (auto/on/off/jax/scipy)
- ``SFINCS_JAX_RHSMODE1_SPARSE_OPERATOR`` (optional sparse matvec path)
- ``SFINCS_JAX_RHSMODE1_SPARSE_MATVEC`` (CSR matvec in explicit mode)
- ``SFINCS_JAX_RHSMODE1_SPARSE_DROP_TOL`` / ``SFINCS_JAX_RHSMODE1_SPARSE_DROP_REL``
- ``SFINCS_JAX_RHSMODE1_SPARSE_ILU_DROP_TOL`` / ``SFINCS_JAX_RHSMODE1_SPARSE_ILU_FILL_FACTOR``
- ``SFINCS_JAX_RHSMODE1_SPARSE_ILU_DENSE_MAX`` (max size for JAX triangular apply)
- ``SFINCS_JAX_RHSMODE1_SPARSE_DENSE_CACHE_MAX`` (reuse assembled dense operator for fallback solves)
- ``SFINCS_JAX_RHSMODE1_SPARSE_ALLOW_NONDIFF`` (explicit-only override)
- ``SFINCS_JAX_RHSMODE1_SPARSE_JAX_MAX_MB`` (memory guard for JAX sparse assembly)
- ``SFINCS_JAX_RHSMODE1_SPARSE_JAX_SWEEPS`` / ``SFINCS_JAX_RHSMODE1_SPARSE_JAX_OMEGA``
  (Jacobi sweep count and relaxation factor)
- ``SFINCS_JAX_RHSMODE1_SPARSE_JAX_REG`` (diagonal regularization for the sparse Jacobi preconditioner)

**PAS x-block :math:`(\theta,\zeta)` preconditioner.** For PAS cases with
angular grids, ``sfincs_jax`` can build per‑species, per‑:math:`x` blocks over
the full :math:`(L,\theta,\zeta)` space. This captures angular coupling without
forming the full species block, and it is selected automatically when
:math:`L \times N_\theta \times N_\zeta` stays below
``SFINCS_JAX_RHSMODE1_XBLOCK_TZ_MAX``.

Implementation: ``sfincs_jax.v3_driver`` (``_build_rhsmode1_xblock_tz_preconditioner``).

These are cached to avoid recomputation. RHS-only gradients are excluded from the cache key
so scan points can reuse the same preconditioner blocks. Controls:

- ``SFINCS_JAX_RHSMODE1_PRECONDITIONER``
- ``SFINCS_JAX_RHSMODE1_SPECIES_BLOCK_MAX`` (auto cap for PAS species-block preconditioning)
- ``SFINCS_JAX_RHSMODE1_XBLOCK_TZ_MAX`` (auto cap for PAS per‑x :math:`(\theta,\zeta)` preconditioning)
- ``SFINCS_JAX_RHSMODE1_SXBLOCK_MAX`` (auto cap for FP species×(x,L) blocks)
- ``SFINCS_JAX_RHSMODE1_COLLISION_PRECOND_KIND``
- ``SFINCS_JAX_RHSMODE1_COLLISION_SXBLOCK_MAX`` / ``SFINCS_JAX_RHSMODE1_COLLISION_XBLOCK_MAX``
- ``SFINCS_JAX_RHSMODE1_SCHUR_MODE`` / ``SFINCS_JAX_RHSMODE1_SCHUR_FULL_MAX``
- ``SFINCS_JAX_RHSMODE1_SCHUR_AUTO_MIN`` (auto Schur cutoff by total size)
- ``SFINCS_JAX_RHSMODE1_DKES_XBLOCK_TZ_SMALL_MAX`` (DKES PAS only: cap on dense xblock_tz base; default matches ``SFINCS_JAX_RHSMODE1_XBLOCK_TZ_MAX``)
- ``SFINCS_JAX_RHSMODE1_PAS_DKES_CPU_PAS_TZ_MIN`` / ``SFINCS_JAX_RHSMODE1_PAS_DKES_CPU_PAS_TZ_ACTIVE_MAX`` and ``SFINCS_JAX_RHSMODE1_PAS_DKES_GPU_PAS_TZ_MIN`` / ``SFINCS_JAX_RHSMODE1_PAS_DKES_GPU_PAS_TZ_ACTIVE_MAX`` (PAS DKES only: prefer the structured ``pas_tz`` angular block above this backend-specific angular-block size and below this active-DOF cap; defaults ``950`` and ``15000``)
- ``SFINCS_JAX_RHSMODE1_PAS_FULL_CPU_PAS_TZ_NZETA_MAX`` / ``SFINCS_JAX_RHSMODE1_PAS_FULL_CPU_PAS_TZ_MIN`` / ``SFINCS_JAX_RHSMODE1_PAS_FULL_CPU_PAS_TZ_ACTIVE_MAX`` (CPU full-trajectory PAS only: prefer ``pas_tz`` over Schur for bounded geometryScheme=11 cases; defaults ``19``, ``950``, and ``15000``)
- ``SFINCS_JAX_RHSMODE1_PAS_FULL_GPU_PAS_TZ`` plus ``SFINCS_JAX_RHSMODE1_PAS_FULL_GPU_PAS_TZ_NZETA_MAX`` / ``SFINCS_JAX_RHSMODE1_PAS_FULL_GPU_PAS_TZ_MIN`` / ``SFINCS_JAX_RHSMODE1_PAS_FULL_GPU_PAS_TZ_ACTIVE_MAX`` (one-GPU full-trajectory PAS geometryScheme=11 only: prefer ``pas_tz`` over Schur for the measured bounded HSX / SFINCS-paper rows; defaults enabled, ``19``, ``950``, and ``15000``)
- ``SFINCS_JAX_RHSMODE1_PAS_TOKAMAK_GPU_TOL`` (bounded one-GPU tokamak PAS+Er only: default tight-GMRES tolerance ``1e-8`` for cases below the ``xblock_tz`` active-size window; ``0`` disables the tightening; legacy ``SFINCS_JAX_RHSMODE1_PAS_TOKAMAK_GPU_THETA_TOL`` is accepted)
- ``SFINCS_JAX_RHSMODE1_PAS_TOKAMAK_GPU_XBLOCK_ACTIVE_MIN`` / ``SFINCS_JAX_RHSMODE1_PAS_TOKAMAK_GPU_XBLOCK_ACTIVE_MAX`` (bounded one-GPU tokamak PAS+Er only: default ``1000`` / ``8000`` active-DOF window for the structured ``xblock_tz`` branch)
- ``SFINCS_JAX_RHSMODE1_GEOM4_PAS_MEMORY_PAS_TZ`` and
  ``SFINCS_JAX_RHSMODE1_GEOM4_PAS_MEMORY_PAS_TZ_*`` (geometryScheme=4 PAS no-Er
  memory policy: prefer top-level ``pas_tz`` over Schur; default on for the
  bounded measured offender)
- ``SFINCS_JAX_RHSMODE1_PAS_XMG_MIN`` (auto switch to the lightweight PAS x‑multigrid preconditioner for large systems; default ``80000``)
- ``SFINCS_JAX_RHSMODE1_FP_XMG_MAX`` (near-zero-``Er`` FP systems below this size use x‑multigrid preconditioning by default; default ``100000``)
- ``SFINCS_JAX_RHSMODE1_SXBLOCK_TZ_ACTIVE_MAX`` (caps auto ``sxblock_tz`` selection for FP systems to avoid expensive setup on large RHSMode=1 runs; default ``20000``)
- ``SFINCS_JAX_BICGSTAB_FALLBACK_ABS_FLOOR`` (absolute residual floor before forcing BiCGStab→GMRES fallback; default auto floor on distributed PAS runs)
- ``SFINCS_JAX_RHSMODE1_XMG_STRIDE`` (coarse‑x stride for the PAS x‑multigrid preconditioner)
- ``SFINCS_JAX_RHSMODE1_PAS_XDIAG_MIN`` (auto switch to point‑block x‑diagonal preconditioner for large PAS runs; default disabled)
- ``SFINCS_JAX_RHSMODE1_XBLOCK_TZ_LMAX`` (truncate L in PAS x‑block :math:`(\theta,\zeta)` preconditioning)
- ``SFINCS_JAX_RHSMODE1_PAS_TZ_MEMORY_FALLBACK`` (route for memory-unsafe
  ``pas_tz`` builds; default uses the cheap collision fallback when available
  so rejected ``pas_tz`` attempts fail fast, ``hybrid`` restores the historical
  ``pas_hybrid`` fallback for A/B profiling, ``theta``, ``zeta``, or
  ``schwarz`` request a structured additive-Schwarz fallback for bounded
  geometry-rich PAS experiments, and unsafe explicit structured requests are
  demoted to ``tzfft`` only when that experimental builder is available;
  ``tzfft`` itself selects the matrix-free angular-streaming fallback directly
  and remains benchmark-only)
- ``SFINCS_JAX_RHSMODE1_PAS_TZ_SCHWARZ_BLOCK`` /
  ``SFINCS_JAX_RHSMODE1_PAS_TZ_SCHWARZ_OVERLAP`` (shared block and overlap used
  by the opt-in structured PAS-TZ Schwarz fallback when axis-specific
  ``SFINCS_JAX_RHSMODE1_THETA_DD_*`` or ``SFINCS_JAX_RHSMODE1_ZETA_DD_*`` values
  are not set)
- ``SFINCS_JAX_RHSMODE1_PAS_TZ_SCHWARZ_MAX_PATCH_UNKNOWNS`` /
  ``SFINCS_JAX_RHSMODE1_PAS_TZ_SCHWARZ_MAX_INVERSE_ENTRIES`` (guardrails for the
  opt-in structured fallback; default values reject production grids that would
  allocate too many dense Schwarz inverse entries, and ``0`` disables each cap)
- ``SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_STRONG_RETRY`` (default off; set to ``1``
  only when profiling the expensive strong retry after a guarded structured
  PAS-TZ fallback)
- ``SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_STAGE2_RETRY`` (default off; set to
  ``1`` only when profiling stage-2 GMRES after a guarded PAS-TZ fallback; the
  default skip keeps memory-unsafe fallback benchmarks bounded)
- ``SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_CORRECTION`` (default off; set to
  ``tzfft`` to keep the cheap fallback as the Krylov preconditioner and apply a
  bounded matrix-free angular-streaming correction after Krylov; benchmark-only
  until it clears the residual and memory gates)
- ``SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_MINRES_STEPS`` /
  ``SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_MINRES_ALPHA_CLIP`` /
  ``SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_MINRES_MIN_IMPROVEMENT`` (bounded
  matrix-free post-Krylov correction for guarded PAS-TZ fallback; it uses only
  extra matvecs and accepts a correction only when the measured residual
  decreases)
- ``SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_POLY_STEPS`` /
  ``SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_POLY_DAMPING`` (opt-in polynomial
  preconditioner experiment for guarded PAS-TZ fallback; default off because
  the geometry4 smoke probe showed residual growth)
- ``SFINCS_JAX_PAS_STAGE2_WEAK_SKIP_RATIO`` (default ``1e12``) skips stage-2
  for forced weak PAS base preconditioners (``collision``, ``point``, and
  ``xmg``) only when the first residual ratio is so large that the follow-on
  polish solve has been measured to stall; set ``0`` to disable this guard for
  profiling.
- ``SFINCS_JAX_PAS_STRONG_WEAK_SKIP_RATIO`` (default ``1e12``) skips the
  automatic strong-preconditioner retry for the same weak PAS base kinds at
  enormous residual ratios; set ``0`` to force the older retry/search behavior.
- ``SFINCS_JAX_PAS_WEAK_MINRES_RATIO`` / ``SFINCS_JAX_PAS_WEAK_MINRES_STEPS``
  (defaults ``1e6`` / ``2``) apply a bounded matrix-free minimal-residual
  correction to forced weak PAS base solves before stage-2 or strong-retry
  escalation. The correction reuses the existing weak preconditioner and is
  accepted only when the measured residual decreases.
- ``SFINCS_JAX_PAS_WEAK_MINRES_ALPHA_CLIP`` /
  ``SFINCS_JAX_PAS_WEAK_MINRES_MIN_IMPROVEMENT`` control the scalar line-search
  clipping and minimum accepted improvement for that weak-PAS correction.
- ``SFINCS_JAX_PRECOND_MAX_MB`` / ``SFINCS_JAX_PRECOND_CHUNK`` (cap memory during block assembly)
- ``SFINCS_JAX_PRECOND_PAS_MAX_COLS`` (additional column cap for PAS block assembly;
  reduces peak RSS by chunking :math:`(\theta,\zeta)` blocks)
- ``SFINCS_JAX_PRECOND_DTYPE`` (default ``auto``; ``float32`` or ``float64`` to override)
- ``SFINCS_JAX_PRECOND_FP32_MIN_SIZE`` (threshold for auto mixed precision)

The PAS benchmark promotion gate is intentionally stricter than a single
threshold comparison. When ``scripts/benchmark_pas_tz_memory_fallback.py`` is
run with ``--require-default-promotion-gate``, a residual-clean candidate must
not regress either elapsed time or peak RSS against the supplied baseline, and
must still show a material runtime or memory win. This prevents a PAS policy
from being promoted when it is faster but higher-memory, or lower-memory but
slower.

**PAS-lite / PAS-hybrid / PAS-Schur preconditioners.** For PAS-only RHSMode=1
systems, ``sfincs_jax`` now defaults to lightweight PAS-specific preconditioners
before attempting expensive global Schur or dense fallbacks:

- **``pas_lite``**: for large PAS systems, combine a cheap angular/L block
  preconditioner (``pas_tz`` in 3D or ``pas_tokamak_theta`` for tokamak-like cases)
  with an x‑coarse correction (``xmg``) and the collision diagonal. This keeps
  setup cost low while stabilizing Krylov iterations. Auto-triggered when
  ``active_size`` exceeds ``SFINCS_JAX_PAS_LITE_MIN`` and the angular block size
  stays below ``SFINCS_JAX_PAS_LITE_TZ_MAX``.
- **``pas_hybrid``**: line/x‑coarse hybrid for PAS systems that need a stronger
  angular block than ``pas_lite``. Uses a truncated‑:math:`L` angular block
  (``xblock_tz_lmax`` or ``pas_tokamak_theta``) followed by ``xmg``.
- **``pas_schur``**: PAS‑specific block‑Schur composition (angular/L block +
  x‑coarse + collision). This is now the default for tokamak‑like PAS cases
  (``N_\zeta \le 5`` or ``geometryScheme=1``), avoiding the expensive global
  constraint‑Schur fallback when PAS preconditioning is already active.

Key controls:

- ``SFINCS_JAX_PAS_LITE_MIN`` / ``SFINCS_JAX_PAS_LITE_TZ_MAX`` (auto ``pas_lite`` thresholds)
- ``SFINCS_JAX_RHSMODE1_PAS_TZ_LMAX`` (truncate :math:`L` in PAS angular blocks)
- ``SFINCS_JAX_RHSMODE1_PAS_SCHUR_SMALL_MAX`` (enable ``pas_schur`` below this size)
- ``SFINCS_JAX_RHSMODE1_PAS_XMG_MIN`` (switch PAS preconditioning to ``xmg`` for large systems)

**Large tokamak PAS BiCGStab fastpath.** For very large 1-species tokamak PAS
RHSMode=1 systems (:math:`N_\zeta=1`, no :math:`\Phi_1`, ``active_size`` above
``SFINCS_JAX_PAS_LARGE_BICGSTAB_FASTPATH_MIN``), ``auto`` now switches from
GMRES to BiCGStab and skips default stage-2/strong fallback passes. This trims
runtime substantially on the largest PAS no-Er benchmarks while maintaining
output agreement at suite tolerances. Controls:

- ``SFINCS_JAX_PAS_LARGE_BICGSTAB_FASTPATH`` (``auto``/``on``/``off``; default ``auto``)
- ``SFINCS_JAX_PAS_LARGE_BICGSTAB_FASTPATH_MIN`` (default ``80000`` active DOFs)

**PAS sparse per-x LU/ILU preconditioner (PETSc-like).** For PAS-only RHSMode=1
systems *without* :math:`x` coupling (no FP, no Er ``xDot`` term), the operator
is block-diagonal in :math:`x`. In this setting, a robust PETSc-like block-Jacobi
preconditioner is:

- assemble a sparse approximation :math:`\tilde{A}_{s,x}` of each per-(species, :math:`x`)
  :math:`(L,\theta,\zeta)` block using the same stencil structure as the matrix-free matvec,
- factorize :math:`\tilde{A}_{s,x}` once (SciPy sparse LU for small blocks; sparse ILU for larger),
- apply the triangular factors in pure JAX inside GMRES iterations.

This is primarily a fallback for large DKES PAS blocks (or strict memory caps). For
medium DKES PAS systems, the default now prefers dense ``xblock_tz`` blocks when the
estimated dense memory stays within the configured cap, since that path is typically
more robust and faster in practice.

Implementation: ``sfincs_jax.v3_driver`` (``_build_rhsmode1_pas_xblock_ilu_preconditioner``).
Key controls:

- ``SFINCS_JAX_RHSMODE1_SCHUR_BASE=pas_ilu`` (force as Schur base for constraintScheme=2)
- ``SFINCS_JAX_RHSMODE1_PAS_LU_MAX`` (max per-x block size for exact sparse LU; default ``5000``)
- ``SFINCS_JAX_RHSMODE1_PAS_LU_ROW_NNZ_MAX`` (cap stored nnz per row for LU factors; default ``512``)
- ``SFINCS_JAX_RHSMODE1_PAS_ILU_DROP_TOL`` / ``SFINCS_JAX_RHSMODE1_PAS_ILU_FILL_FACTOR``
  (ILU parameters for larger blocks)
- ``SFINCS_JAX_RHSMODE1_PAS_ILU_ROW_NNZ_MAX`` (cap stored nnz per row for ILU factors; default ``64``)

**PAS 3D :math:`(\theta,\zeta)`/L block-tridiagonal preconditioner.** For PAS-only
RHSMode=1 systems without :math:`x` coupling, the collisionless streaming+mirror
operator couples Legendre modes only through :math:`\Delta L=\pm 1`, so for each
species and each :math:`x` we can write a block-tridiagonal system in :math:`L`,

.. math::

   A^{(L)} f^{(L)} + B^{(L)} f^{(L+1)} + C^{(L)} f^{(L-1)} = r^{(L)},

where each block is of size :math:`N_\theta N_\zeta`. The diagonal blocks
:math:`A^{(L)}` include the PAS diagonal (plus any identity shift) and the
E×B advection terms, while :math:`B^{(L)}` and :math:`C^{(L)}`
contain the streaming and mirror couplings (row-scaled finite-difference
operators in :math:`\theta` and :math:`\zeta`).

Rather than assembling/inverting the full per-:math:`x`
:math:`(L,\theta,\zeta)` block (``xblock_tz``), ``sfincs_jax`` precomputes a
block-Thomas (block-tridiagonal) factorization analytically and applies it in
JAX. This avoids the expensive dense-matvec assembly that ``xblock_tz`` uses and
reduces preconditioner build time and peak memory on 3D PAS cases with large
angular blocks.

Implementation: ``sfincs_jax.v3_driver`` (``_build_rhsmode1_pas_tz_preconditioner``).

Auto selection: for 3D PAS-only cases, ``auto`` selects ``pas_tz`` as the Schur
base when :math:`L_{\max} N_\theta N_\zeta` exceeds ``SFINCS_JAX_RHSMODE1_PAS_TZ_MIN``
(default ``800``). You can force it with ``SFINCS_JAX_RHSMODE1_SCHUR_BASE=pas_tz``.

**KSP history cost.** PETSc-style KSP residual histories and iteration counts are
computed via an additional SciPy solve (to match the PETSc text). For large Krylov
counts this can dominate runtime, so the defaults now skip these when the estimated
iteration count exceeds ``SFINCS_JAX_KSP_HISTORY_MAX_ITER`` /
``SFINCS_JAX_SOLVER_ITER_STATS_MAX_ITER``. Raise those caps (or set to ``none``)
only when strict per-iteration history is required.

**Mixed-precision preconditioning.** With ``SFINCS_JAX_PRECOND_DTYPE=auto`` (default),
preconditioner blocks switch to float32 once the estimated system size exceeds
``SFINCS_JAX_PRECOND_FP32_MIN_SIZE`` (global) or the per-block size exceeds
``SFINCS_JAX_PRECOND_FP32_MIN_BLOCK`` (per-block), while Krylov iterations remain
in float64. In addition, the current default auto policy enables float32
preconditioner storage on the near-zero-:math:`E_r`, PAS-only, ``geometryScheme=4``
Schur branch on CPU once the full system reaches
``SFINCS_JAX_PRECOND_FP32_PAS_GEOM4_MIN_SIZE`` (default ``15000``). This is a
measured memory optimization for the Miller-like two-species PAS case family:
the audited ``geometryScheme4_2species_PAS_noEr`` fixture dropped from roughly
``2.95 GB`` RSS to ``1.98 GB`` while keeping ``0`` mismatches against the frozen
Fortran reference. The policy stays off for PAS+DKES and other geometries because
the same blanket float32 switch degraded HSX/geometry11 PAS branches.

**Lightweight profiling.** Set ``SFINCS_JAX_PROFILE=1`` to emit coarse timing and
memory marks during RHSMode=1 solves (operator build, RHS assembly, preconditioner
construction, strong-preconditioner fallback). The output looks like:

.. code-block:: text

   profiling: operator_built dt_s=0.42 total_s=0.42 rss_mb=512.0 drss_mb=35.0 peak_rss_mb=512.0 dpeak_rss_mb=35.0 device_mb=na
   profiling: rhs_assembled dt_s=0.08 total_s=0.50 rss_mb=515.0 drss_mb=38.0 peak_rss_mb=515.0 dpeak_rss_mb=38.0 device_mb=na
   profiling: rhs1_precond_build_start dt_s=0.00 total_s=0.50 ...
   profiling: rhs1_precond_build_done dt_s=1.25 total_s=1.75 ...

.. [#saad86] Y. Saad and M. Schultz, “GMRES: A generalized minimal residual algorithm for
   solving nonsymmetric linear systems,” *SIAM J. Sci. Stat. Comput.* 7(3), 1986.
.. [#benzi02] M. Benzi, “Preconditioning techniques for large linear systems: a survey,”
   *J. Comput. Phys.* 182(2), 2002.

This is intentionally low overhead and does not require external profilers. For
detailed JAX tracing, use ``jax.profiler`` or standard tools, but keep them off
for parity runs.

**XLA dump profiling.** For kernel-level inspection, you can dump HLO/LLVM with
``XLA_FLAGS=--xla_dump_to=/tmp/sfincs_xla`` (optionally add
``--xla_dump_hlo_as_text``). This is heavier and should be used only for
targeted performance investigations.

Matvec fusion for collisionless + drift terms
---------------------------------------------

**Technique.** Accumulate collisionless streaming, ExB, magnetic-drift, and Er
drift contributions in a single static sum expression to reduce Python overhead
while keeping the matvec control‑flow free (required for JAX GMRES/BiCGStab).

**Implementation.**

- ``SFINCS_JAX_FUSED_MATVEC`` (default enabled) in
  ``sfincs_jax.operators.profile_response.fblock``.

**Notes.** Avoid ``lax.scan``/``lax.fori_loop`` inside the matvec used by JAX
iterative solvers: they assert on control‑flow. The collision operators (PAS/FP)
remain separate so remat/checkpointing controls and Phi1‑dependent variants stay intact.

Sparse-row derivative kernels (step 1)
--------------------------------------

**Technique.** Replace dense :math:`d/d\theta` and :math:`d/d\zeta` matrix contractions
in the collisionless block with a sparse-row gather kernel when the differentiation
matrix has a small number of nonzeros per row.

For a derivative matrix :math:`D` and state :math:`f`, the dense apply

.. math::

   y_i = \sum_j D_{ij} f_j

is evaluated as a sparse row sum

.. math::

   y_i = \sum_{k=1}^{K_i} w_{i,k}\, f_{c_{i,k}},

where :math:`c_{i,k}` are the stored nonzero column indices in row :math:`i`.
This reduces derivative cost from :math:`\mathcal{O}(N^2)` to
:math:`\mathcal{O}(N K)` with small :math:`K` (typically 3–5 for v3 finite-difference
schemes used in reduced-suite inputs).

**Implementation.**

- Sparse-row extraction and apply:
  ``sfincs_jax.discretization.periodic_stencil.extract_sparse_row_stencil`` and
  ``sfincs_jax.discretization.periodic_stencil.apply_sparse_row_stencil_gather``.
- Collisionless operator fast path:
  ``sfincs_jax.collisionless.apply_collisionless_v3``.
- Operator build wiring:
  ``sfincs_jax.operators.profile_response.fblock.collisionless_operator_from_namelist``.

**Periodic circulant optimization.** For periodic circulant derivative matrices,
an additional compact roll-based kernel is available and enabled on single-device
runs. On multi-device runs this roll path is disabled by default to avoid
cross-device halo overhead regressions in current JAX CPU sharding.

**Controls.**

- ``SFINCS_JAX_PERIODIC_STENCIL``: enable/disable periodic roll kernel (default on).
- ``SFINCS_JAX_PERIODIC_STENCIL_ON_SHARDED``: force periodic roll kernel on sharded
  multi-device runs (default off).
- ``SFINCS_JAX_DERIV_SPARSE_MAX_ROW_NNZ``: max nonzeros per row for sparse-row
  extraction (default ``9``).

**Validation and measured impact.**

- Unit parity coverage:
  ``tests/test_periodic_stencil.py``,
  ``tests/test_collisionless_operator_parity.py``,
  ``tests/test_fblock_pas_matvec_parity.py``.
- Benchmark (cache-warm, `transport_parallel_xxlarge`, 1 CPU device):
  mean matvec time improved from ``7.49e-4 s`` (stencil off) to
  ``5.87e-4 s`` (stencil on), about **1.28× faster**.

Transport diagnostics: batched + precomputed
--------------------------------------------

**Technique.** For transport matrices, solve all ``whichRHS`` systems, stack solutions,
then compute diagnostics in one batched kernel.

**Transport flux formulas (vm-only).** From v3 diagnostics, the key outputs are:

.. math::

   \Gamma_s^{\mathrm{vm}} \propto \int d\theta\,d\zeta\; \mathcal{F}_{\mathrm{vm}}(\theta,\zeta)\,
   \left[\frac{8}{3}\,\sum_x w_x x^4 f_{s,L=0} + \frac{4}{15}\,\sum_x w_x x^4 f_{s,L=2}\right],

.. math::

   Q_s^{\mathrm{vm}} \propto \int d\theta\,d\zeta\; \mathcal{F}_{\mathrm{vm}}(\theta,\zeta)\,
   \left[\frac{8}{3}\,\sum_x w_x x^6 f_{s,L=0} + \frac{4}{15}\,\sum_x w_x x^6 f_{s,L=2}\right],

.. math::

   \mathrm{FSABFlow}_s \propto \int d\theta\,d\zeta\; \frac{B}{D}\,
   \sum_x w_x x^3 f_{s,L=1}.

These are implemented in ``sfincs_jax.problems.transport_matrix.diagnostics``
with strict-order reductions matching v3 when required.

**Precompute constants + cache.**

Factors depending only on geometry, species normalization, and grids
(:math:`w_x`, :math:`B/D`, prefactors, etc.) are precomputed once per transport run
and reused for all ``whichRHS`` solves.

**Implementation.**

- ``v3_transport_diagnostics_vm_only_precompute`` and
  ``v3_transport_diagnostics_vm_only_batch_op0_precomputed``.
- Cached by operator signature in
  ``sfincs_jax.problems.transport_matrix.diagnostics`` to reuse
  geometry/species factors across repeated transport solves (default cache size: ``4``;
  override with ``SFINCS_JAX_TRANSPORT_DIAG_CACHE_MAX``).
- For large transport solves, diagnostics can be processed in chunks to reduce peak
  memory. Use ``SFINCS_JAX_TRANSPORT_DIAG_CHUNK`` (default: auto for
  ``N * total_size > 2e5``) to set an explicit chunk size.
- Rematerialization for transport diagnostics is enabled automatically at the same
  threshold (override with ``SFINCS_JAX_TRANSPORT_DIAG_REMAT``).

**Compared to Fortran.**

Fortran computes diagnostics per ``whichRHS`` in loops. JAX batches this and
reuses precomputed constants to reduce overhead and JIT work.

**Impact.**

Lower diagnostic overhead in transport workflows and reduced JIT retracing from
repeated reconstruction of geometry-dependent factors.

Recycled Krylov initial guesses for transport
---------------------------------------------

**Technique.** Reuse a small basis from recent solves to warm-start the next RHS:

.. math::

   x_0 \approx U (A U)^{\dagger} b,

where :math:`U` contains recent solution vectors.

**Implementation.**

- ``SFINCS_JAX_TRANSPORT_RECYCLE_K`` in ``sfincs_jax.v3_driver``.
- ``SFINCS_JAX_STATE_IN``/``SFINCS_JAX_STATE_OUT`` (cross-run recycling).
- ``SFINCS_JAX_SCAN_RECYCLE`` (auto-wires state files between scan points).
- ``SFINCS_JAX_RHSMODE1_RECYCLE_K`` (RHSMode=1 scan reuse with least-squares deflation).

**Compared to Fortran.**

Fortran does not reuse Krylov subspaces across ``whichRHS``. This reuse is
lightweight and stays matrix-free.

Weighted reductions and Fortran sum order
-----------------------------------------

**Technique.** Use fused `einsum` for weighted sums by default, but allow
Fortran-like deterministic accumulation order when parity demands it.

**Implementation.**

- ``_weighted_sum_x_fortran`` and ``_weighted_sum_tz_fortran`` in
  ``sfincs_jax.problems.transport_matrix.diagnostics``.
- Strict order controlled by ``SFINCS_JAX_STRICT_SUM_ORDER``.

This reduces Python overhead and improves performance, while still preserving
Fortran parity when needed.

Dense fallbacks (RHSMode=1 vs transport)
----------------------------------------

Dense fallbacks can stabilize difficult RHSMode=1 systems, and dense *retries* can
rescue transport-matrix solves that stall.

**Current default:**

- RHSMode=1 dense fallback is **enabled for modest systems** (``total_size <= 3000``)
  when Krylov iterations stagnate in FP cases. The trigger uses the **true
  (unpreconditioned)** residual norm so the fallback still fires even if a
  left-preconditioned norm appears small (strict residual-guard behavior). PAS fallbacks
  are disabled by default unless ``SFINCS_JAX_RHSMODE1_DENSE_PAS_MAX`` is set.
- For RHSMode=1 runs with ``includePhi1 = .true.``, small systems
  bypass the Newton–Krylov inner GMRES step and take a dense Newton step instead.
  This avoids GMRES setup cost and matches Fortran parity for Phi1‑collision fixtures.
  The cutoff is controlled by ``SFINCS_JAX_PHI1_NK_DENSE_CUTOFF`` (default: ``5000``).
- For RHSMode=2 transport matrices with multiple RHS, ``sfincs_jax`` now
  auto‑selects a **dense batched solve** when the active system size is modest
  (``n \le min(3000, SFINCS_JAX_TRANSPORT_DENSE_RETRY_MAX)``) and the dense
  memory budget allows it. This avoids spending time on Krylov retries when a
  dense solve is likely anyway. For other transport cases, dense fallback remains
  **disabled** unless explicitly requested, but a dense retry is enabled for
  RHSMode=2/3 when the active system size is modest.
- For CPU VMEC monoenergetic transport (``RHSMode=3``, ``geometryScheme=5``,
  PAS/no-FP, small ``Nx``), ``sfincs_jax`` avoids the dense batched fallback by
  default and uses the existing Krylov + ``tzfft`` path. This is a measured
  memory policy: on ``monoenergetic_geometryScheme5_ASCII`` it preserved all
  Fortran output comparisons while reducing CLI-profiled RSS from about
  ``3.0 GB`` to about ``0.5 GB``.

Controls:

- ``SFINCS_JAX_RHSMODE1_DENSE_FALLBACK_MAX`` (default: ``400``).
- ``SFINCS_JAX_RHSMODE1_DENSE_FP_MAX`` (default: ``8000``) for full Fokker–Planck
  cases (``collisionOperator=0``).
- ``SFINCS_JAX_RHSMODE1_DENSE_PAS_MAX`` (default: disabled) for PAS/constraintScheme=2
  cases (notably DKES trajectories); set a positive value to enable.
- ``SFINCS_JAX_RHSMODE1_DENSE_FALLBACK_RATIO`` (default: ``1e2``). Dense fallback
  only triggers when ``||r|| / target`` exceeds this ratio (set ``<= 0`` to always
  allow the fallback).
- ``SFINCS_JAX_RHSMODE1_DENSE_SHORTCUT_RATIO`` (default: ``1e6``). When the residual
  ratio exceeds this threshold, ``sfincs_jax`` skips sparse ILU and other heavy
  fallbacks and goes straight to the dense solve (if enabled). This avoids wasting
  time on ILU builds when dense fallback is inevitable.
- ``SFINCS_JAX_RHSMODE1_DENSE_PROBE`` (default: on). Perform a cheap single-step
  preconditioner probe (one matvec) and, if the residual ratio still exceeds
  ``SFINCS_JAX_RHSMODE1_DENSE_SHORTCUT_RATIO``, skip stage-2/strong Krylov attempts
  and proceed directly to the dense fallback.
- ``SFINCS_JAX_RHSMODE1_TOKAMAK_PAS_NOER_SPARSE_PC`` (default: auto). On CPU/GPU,
  non-differentiable tokamak PAS no-Er RHSMode=1 runs in the measured
  production-floor window use the host sparse-PC GMRES route. This avoids the
  matrix-free Krylov memory cliff for the historically audited two-species
  ``25 x 1 x 8 x 100`` row and the lower-floor GPU runtime cliff for the
  one-species ``25 x 1 x 4 x 100`` row while preserving Fortran parity. The
  default route factors the active
  ``Nxi_for_x`` degrees of freedom rather than the padded full Legendre grid:
  the public two-species CPU row drops from about ``1.75 GB`` active RSS on the
  matrix-free path to about ``0.39 GB`` with sparse-PC GMRES, the two-species
  RTX A4000 row drops from about ``14.7 s`` to about ``5.2 s``, and the
  one-species ``Nx=4`` RTX A4000 row drops from about ``28.8 s`` to about
  ``3.0 s``. At the current raised public floor ``89 x 1 x 24 x 300``, local
  CPU runs are strict-clean and faster than serial SFINCS Fortran v3:
  ``tokamak_1species_PASCollisions_noEr`` takes ``10.38 s`` JAX CPU versus
  ``17.44 s`` Fortran, and ``tokamak_1species_PASCollisions_noEr_Nx1`` takes
  ``8.53 s`` JAX CPU versus ``17.54 s`` Fortran. The refreshed office GPU
  shards select the same route automatically and are practical-parity clean:
  ``tokamak_1species_PASCollisions_noEr`` takes ``19.47 s`` JAX GPU versus
  ``15.92 s`` Fortran with ``4.19 GB`` JAX RSS, and
  ``tokamak_1species_PASCollisions_noEr_Nx1`` takes ``19.22 s`` JAX GPU versus
  ``13.86 s`` reused Fortran with ``4.14 GB`` JAX RSS. Strict GPU mode leaves
  one per-``x`` flow diagnostic mismatch tied to Fortran residual/reference
  sensitivity; integrated transport gates pass. Set this variable to ``0`` to
  force the older matrix-free path, or to ``1`` to force the route while
  retaining the remaining guards.
- ``SFINCS_JAX_RHSMODE1_TOKAMAK_PAS_ER_SPARSE_PC`` (default: auto). On CPU/GPU,
  non-differentiable tokamak PAS+Er full-trajectory RHSMode=1 runs in the
  measured production-floor window use the host sparse-PC GMRES route. This
  avoids the PAS-ILU/Schur stage-2 stall while preserving Fortran parity for the
  audited one- and two-species ``25 x 1 x 8 x 100`` CPU and RTX A4000 GPU cases.
  The route defaults to the lower-fill measured ``MMD_AT_PLUS_A`` SuperLU
  column ordering for the tokamak PAS+Er full-trajectory sparse-PC window,
  ``SFINCS_JAX_RHSMODE1_SPARSE_PC_SHIFT=1e-8`` and
  ``SFINCS_JAX_EXPLICIT_SPARSE_DIAG_PIVOT_THRESH=0`` for constrained PAS unless
  the user overrides those values. In the same measured tokamak PAS+Er window,
  sparse-PC GMRES also factors the active ``Nxi_for_x`` degrees of freedom by
  default rather than the padded full Legendre grid. The active sparse-PC route
  reduces the two-species production row from ``40016`` to ``25466`` linear
  unknowns, from ``7.90M`` to ``3.51M`` probed sparse entries, and from about
  ``11.8 s`` logged / ``2.26 GB`` RSS to about ``9.7 s`` logged /
  ``1.59 GB`` active RSS on CPU. On the audited RTX A4000 row, the same active
  route is strict-clean and reduces the logged time from about ``25.0 s`` to
  ``22.2 s`` with active RSS about ``2.12 GB``. At the current checked-in
  tokamak floor ``33 x 1 x 12 x 140``, the refreshed office GPU shard is
  strict-clean for both bounded PAS+Er rows:
  ``tokamak_1species_PASCollisions_withEr_fullTrajectories`` takes ``19.24 s``
  JAX GPU versus ``13.29 s`` Fortran with ``3.19 GB`` JAX RSS, and
  ``tokamak_2species_PASCollisions_withEr_fullTrajectories`` takes ``33.97 s``
  JAX GPU versus ``18.27 s`` Fortran with ``5.41 GB`` JAX RSS. The same
  checked-in floor remains strict-clean on local CPU from the bounded CPU shard.
  All quoted promoted rows have zero practical and strict Fortran output
  mismatches. The one-species PAS+Er row
  additionally caps the default sparse-PC GMRES restart at ``40`` unless the
  user explicitly sets ``SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_RESTART``; bounded
  CPU and RTX A4000 sweeps preserved output parity while slightly reducing
  time-to-solution and peak memory. The no-Er sparse-PC window keeps
  ``MMD_ATA`` as its measured default.
  The older ``COLAMD`` ordering remains available through
  ``SFINCS_JAX_EXPLICIT_SPARSE_PERMC_SPEC=COLAMD``; it is higher-fill on these
  cases and is retained as an explicit reproducibility knob.
- ``SFINCS_JAX_RHSMODE1_SPARSE_PC_ACTIVE_DOF`` (default: auto for the measured
  tokamak PAS no-Er/PAS+Er constrained-PAS sparse-PC windows). Set to ``0`` to
  factor the padded full system for reproducibility experiments, or ``1`` to
  force the active-DOF sparse-PC reduction on another RHSMode=1 sparse-PC case.
  Forced use outside the measured window should be treated as an experiment
  until the output is compared against the Fortran reference for that geometry.
- ``SFINCS_JAX_RHSMODE1_SPARSE_PC_FACTOR_DTYPE`` (default: ``float64`` for
  sparse-PC GMRES). Set to ``32`` only for controlled memory experiments. When
  single-precision factors are requested, the first Krylov attempt is capped by
  ``SFINCS_JAX_RHSMODE1_SPARSE_PC_FP32_PROBE_MAXITER`` (default: ``2``) before
  falling back to ``float64`` if the true residual target is not met. This
  prevents weak low-precision preconditioners from consuming the full run
  timeout.
- ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_INITIAL_SEED`` (default: off). Set to ``1``
  only for controlled diagnostics of explicit ``xblock_sparse_pc_gmres`` runs.
  It applies the x-block sparse preconditioner once before Krylov and uses that
  vector as the initial guess only if the true residual is lower than the RHS
  norm. The scale-0.50 QI blocker probe rejected this seed and therefore this
  knob is intentionally not a default performance path.
- ``SFINCS_JAX_RHSMODE1_FP3D_XBLOCK_SPARSE_PC`` (default: ``auto``). Controls
  the bounded public-auto promotion for non-differentiable 3D full-FP
  ``RHSMode=1`` output/CLI solves. The default gate is restricted to one-species,
  no-``Phi1``, no-PAS, ``constraintScheme=1`` systems with
  ``Nxi >= 50`` and active size between ``30000`` and ``45000``. The matching
  ``*_MIN``, ``*_MAX``, and ``*_MIN_NXI`` environment variables can widen or
  disable the gate for controlled profiling, but larger promotion requires fresh
  CPU/GPU seed-ladder evidence.
- ``SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_AUTO`` (default: enabled). For large
  non-differentiable RHSMode=1 full-FP, no-``Phi1``, ``constraintScheme=1``
  solves, public ``auto`` can choose ``fortran_reduced_pc_gmres``. This route
  assembles a simplified global sparse preconditioner operator, analogous to
  the SFINCS Fortran v3 preconditioner matrix, while accepting the solve only
  against the full SFINCS-JAX true residual. On the bounded Zenodo QA/QH
  ``s=0.5`` ``13 x 13 x 21 x 5`` gates, this reduced QA auto runtime from the
  previous slow structured path (about ``169 s``) to ``3.82 s`` with residual
  ``3.37e-13`` below target, and QH completed in ``4.21 s`` with residual
  ``6.31e-13`` below target. Disable with ``0``/``false`` only when
  reproducing older policy-ladder behavior.
- For full-grid finite-beta QA/QH ``25 x 39 x 60 x 7`` RHSMode=1 diagnostics,
  the robust non-autodiff reference route is the Fortran-reduced direct-tail
  active LU preconditioner. This Fortran-reduced direct-tail active LU
  preconditioner remains the high-memory fallback. The default ``auto`` route now reaches this
  direct-tail ladder with no manual ``PC_BACKEND=global`` or
  ``DIRECT_TAIL_PC_MAX_MB`` overrides. The checked QA/QH active size remains
  ``507004`` unknowns, and ``auto`` assigns the same adaptive
  ``14708.1 MiB`` cap used by the active-LU reference route: it first tries
  ``active_fortran_v3_reduced_native_stack`` as the lower-memory production
  candidate, requires a true-residual preflight for that candidate, and falls
  back to ``active_fortran_v3_reduced_lu`` when the preflight fails. The checked
  QA full-grid auto audit selected native stack in ``9.17 s`` with
  ``5,090,357,984`` estimated factor bytes, rejected it because the one-apply
  residual worsened, accepted active LU as the no-required-preflight fallback,
  and converged to residual ``9.002525e-13`` in ``343.5 s`` wall with ``46``
  GMRES iterations. Earlier checked QA/QH active-LU reference audits converged
  to ``9.950981e-13`` and ``8.712742e-14`` residual with a
  ``13,303,259,384`` byte active matrix. A stricter guarded rerun with
  ``tol=1e-10`` converged to residual ``7.269598e-16`` in ``354.6 s`` wall
  with ``67`` GMRES iterations and ``74`` matvecs. Native true-coupled rescue is
  intentionally opt-in through
  ``SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_AUTO_NATIVE``
  because the checked full-grid native/coarse probes lower the bad one-apply
  residual but still stall near the RHS norm under long Krylov runs. This route
  is robust and parity-facing; the open performance lane is making the
  lower-memory native block/coarse replacement pass the same true-residual gate
  so active LU is no longer needed at full grid.
- ``SFINCS_JAX_RHSMODE1_XBLOCK_SPARSE_LU_MAX`` (default: ``30000`` for
  non-differentiable full-FP host x-block factors; ``2000`` otherwise). Medium
  full-FP :math:`(x,\theta,\zeta,L)` blocks now use exact SuperLU instead of ILU
  because the scale-0.50 QI blocker showed that weak ILU factors caused the
  residual floor. The checked CPU successor artifact
  ``docs/_static/qi_seed_robustness_scale050_xblock_lu_right_cpu.json`` closes
  the ``13 x 27 x 50 x 4`` seed in ``~12 s`` with residual ratio ``4.16e-2``;
  the matching one-GPU artifact
  ``docs/_static/qi_seed_robustness_scale050_xblock_lu_right_gpu.json`` closes
  the same seed in ``~44.5 s`` with residual ratio ``0.63``. The five-seed
  follow-up artifacts
  ``docs/_static/qi_seed_robustness_scale050_xblock_lu_right_multiseed5_cpu.json``
  and
  ``docs/_static/qi_seed_robustness_scale050_xblock_lu_right_multiseed5_gpu.json``
  pass the public ``auto`` path for seeds ``0..4`` on CPU and one GPU with all
  outputs and solver traces written, maximum elapsed times ``11.58 s`` and
  ``41.18 s``, and maximum residual ratios ``0.966`` and ``0.963``. This closes
  the bounded public-auto route gate; the production-resolution QI ladder remains
  separate. The next-scale
  ``docs/_static/qi_seed_robustness_scale055_auto_cpu_blocker.json`` probe showed
  that the old ``20000`` exact-LU cap sent the largest ``15 x 29 x 55 x 4``
  block into ILU and timed out. Raising only the full-FP host exact-LU cap to
  ``30000`` closes the CPU successor
  ``docs/_static/qi_seed_robustness_scale055_xblock_lu_right_cpu.json`` in
  ``~21.5 s`` with residual ratio ``8.25e-3``.
- ``SFINCS_JAX_RHSMODE1_XBLOCK_RIGHT_PC_MAX`` (default: ``45000`` active
  unknowns for 3D full-FP lanes only). The scale-0.50 QI evidence remains
  faster with right preconditioning, but the harder scale-0.55 seed ``3`` probe
  showed a seed-dependent right-PC slow mode. The size-aware default therefore
  keeps right-PC for the checked scale-0.50 window and automatically uses
  left-PC for larger 3D full-FP active systems unless the user explicitly sets
  ``SFINCS_JAX_GMRES_PRECONDITION_SIDE``. The checked
  ``docs/_static/qi_seed_robustness_scale055_xblock_auto_side_seed3_cpu.json``
  artifact closes that hard CPU seed in ``~47 s`` with residual ratio
  ``2.98e-3`` and records ``precondition_side=left``. The matching five-seed
  CPU and one-GPU artifacts
  ``docs/_static/qi_seed_robustness_scale055_xblock_auto_side_multiseed5_cpu.json``
  and
  ``docs/_static/qi_seed_robustness_scale055_xblock_auto_side_multiseed5_gpu.json``
  pass seeds ``0..4`` with all outputs and solver traces written, maximum
  elapsed times ``44.5 s`` and ``206.7 s``, and maximum residual ratios below
  ``8.3e-3``. The default upper auto-size window remains bounded until the
  next-size and production-resolution ladders pass. The first next-size
  ``15 x 31 x 60 x 5`` seed-0 artifacts
  ``docs/_static/qi_seed_robustness_scale060_xblock_auto_side_seed0_cpu.json``
  and
  ``docs/_static/qi_seed_robustness_scale060_xblock_auto_side_seed0_gpu.json``
  also pass at active size ``81377`` with elapsed times ``42.2 s`` and
  ``145.1 s`` and residual ratios below ``4.7e-3``; this is a bounded seed-0
  probe, not a production-resolution five-seed claim. The harder scale-0.60
  seed-3 rejected-probe summary
  ``docs/_static/qi_seed_robustness_scale060_gpu_rejected_solver_probes_2026_05_13.json``
  keeps two-level x-block, GCROT(m,k), BiCGStab fallback, post-correction-only
  BiCGStab, and JAX-factor/device-Krylov variants out of the default policy; the
  only kept fallback behavior is reusing a non-GMRES candidate as a GMRES initial
  guess when it strictly improves the finite RHS norm and is not a
  right-preconditioned coordinate state.
- Transformed RHSMode=1 matvecs use a different sharding boundary than
  top-level operator applications. Setup probes that call the full-system
  operator inside ``vmap`` and matvecs that run inside ``custom_linear_solve``
  now force the local/unsharded JIT path. This avoids nested
  ``jax.set_mesh``/``pjit`` contexts on multi-device hosts while preserving the
  cached sharded path for top-level CPU/GPU matvecs. The safety gate was added
  after the ``13 x 13 x 15 x 4`` QI single-point probe failed before Krylov
  progress on a multi-device CPU runtime.
- ``SFINCS_JAX_RHSMODE1_SPARSE_LARGE_CPU_SKIP_PRIMARY`` (default: on): bounded
  early-entry policy for mid-size explicit RHSMode=1 full-FP systems that are
  just above the dense cutoff and within the measured exact active sparse-LU
  cap. It skips theta-line/primary-Krylov/stage-2 setup and goes directly to the
  existing active sparse-LU rescue. On the checked QI ``13 x 13 x 15 x 4``
  single point, this reduced solver time from ``107.9 s`` to ``35.3 s`` with
  zero difference in the recorded key HDF5 observables. Controls:
  ``SFINCS_JAX_RHSMODE1_SPARSE_LARGE_CPU_SKIP_PRIMARY=0``,
  ``SFINCS_JAX_RHSMODE1_SPARSE_LARGE_CPU_SKIP_PRIMARY_MIN`` (default
  ``max(sparse_max+1, 8000)``), and
  ``SFINCS_JAX_RHSMODE1_SPARSE_LARGE_CPU_SKIP_PRIMARY_MAX`` (default
  ``30000``).
- ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING`` (default: off): opt-in
  smoothed global-coupling wrapper for explicit ``xblock_sparse_pc_gmres``. It
  builds low-rank load vectors from RHS/source rows, low-L flux-surface-average
  moments, and the lowest theta/zeta Fourier components, applies the existing
  x-block preconditioner to form ``Z = B^{-1}P``, then uses a rank-revealed
  ``A Z`` coarse solve inside each preconditioner application. This is more
  physics-aware than the older fixed two-level basis, and it records rank,
  setup time, basis names, and side-probe switch suppression metadata. The
  ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV=fgmres`` and ``gmres-jax``
  experimental routes now use a JAX-array global-coupling variant for this
  wrapper: ``Z`` and a small coarse least-squares solve stay in JAX arrays
  during the Krylov apply path, so the coarse correction no longer requires
  host QR/SciPy calls on every iteration. The default device coarse solver is a
  rank-revealed QR factorization built once at setup; the older
  ridge-regularized normal-equation solver remains available through
  ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_DEVICE_SOLVER=normal-equations``
  for diagnostics. Device global coupling now defaults to an identity/raw-load
  smoother, while the older preconditioned-load smoothing remains available
  through ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_SMOOTHER=base``.
  This avoids spending hundreds of accelerator matvecs before the Krylov solve
  starts on memory-sensitive QI probes. The setup can also be bounded with
  ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_SETUP_MAX_S``; metadata records
  ``xblock_global_coupling_smoother`` and whether the setup budget was reached.
  This is still opt-in and does not change default solver selection.
  The
  scale-0.60 hard-seed probe
  ``docs/_static/qi_seed_robustness_scale060_global_coupling_rejected_2026_05_13.json``
  rejected it for defaults: CPU reached a near residual only after ``539 s`` and
  ``6671`` matvecs without strict output acceptance, while one-GPU runs timed out
  at ``620 s`` whether the side probe switched right or kept the left
  global-coupling state.
- ``SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR`` (default: off): opt-in
  assembled/operator-reuse path for explicit ``xblock_sparse_pc_gmres``. It
  materializes a conservative sparse full-system operator by graph-colored
  pattern probing, validates sampled matvecs against the matrix-free operator,
  and then reuses the assembled operator for Krylov matvecs. The preflight now
  aborts as soon as the color count exceeds
  ``SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_MAX_COLORS`` so large
  QI/PAS geometries do not spend minutes discovering an already-infeasible
  operator. The CSR memory budget
  ``SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_CSR_MAX_MB`` is now enforced
  as a hard cap when materialization is required. When
  ``SFINCS_JAX_RHSMODE1_XBLOCK_ACTIVE_DOF=1`` is also enabled, the preflight
  builds the conservative pattern in active ``Nxi_for_x`` coordinates before
  applying the cap, and records both full-system and active-DOF byte estimates
  in solver metadata. This prevents reduced QI/PAS systems from rejecting a
  feasible assembled Krylov operator because of inactive pitch modes.
  Large active FP/QI patterns use a structured ``velocity ⊗ angular`` builder
  rather than materializing the full inactive pattern and slicing it. On the
  documented scale-0.60 QI seed-3 preflight, this changed pattern construction
  from a 4.52 GB full CSR estimate and ``250.0 s`` full-pattern build, through a
  ``174.6 s`` Python active-loop build, to a ``1.19 s`` structured active build
  for the same 1.54 GB active CSR pattern. A bounded dummy-matvec probe then
  rejected ``max_colors=64`` graph coloring in ``0.88 s``. A follow-up
  production-color probe on the same active system built the pattern in
  ``1.44 s`` and rejected ``max_colors=512`` coloring in ``2.04 s``. This keeps
  infeasible materialization attempts bounded, but it also means the active
  assembled path remains diagnostic-only until a large case shows strict
  residual parity and lower wall time/memory.
- ``SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_DEVICE`` (default: on for
  experimental JAX device Krylov, off otherwise): copies a successfully
  materialized assembled x-block operator into budgeted JAX CSR arrays and uses a
  JIT-compatible CSR matvec instead of calling SciPy from every Krylov
  iteration. ``SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_DEVICE_REQUIRED=1``
  makes this a hard gate for GPU experiments, so a failed device transfer or
  validation does not silently fall back to host round trips. Solver metadata
  records ``xblock_assembled_operator_device_resident``,
  ``xblock_assembled_operator_device_csr_nbytes_estimate``, validation errors,
  and whether device-Krylov matvec/preconditioner application is
  host-transfer-free. The device CSR stores precomputed row indices in addition
  to ``data``, ``indices``, and ``indptr`` so accelerator matvecs avoid
  per-trace ``repeat``/constant-folding work. This is the current QI
  operator-reuse candidate; it is still opt-in until the bounded ``office``
  hard-seed gate writes a converged HDF5 output and solver trace.
- ``SFINCS_JAX_RHSMODE1_XBLOCK_ACTIVE_DOF`` (default: off): opt-in active-DOF
  reduction for explicit ``xblock_sparse_pc_gmres``. When ``Nxi_for_x`` truncates
  the pitch basis, this route solves the x-block Krylov system on the active
  unknowns and expands the final state back to the full SFINCS vector. This is
  intended as a memory-safety prerequisite for future assembled/operator-reuse
  experiments on large QI grids; it is not enabled by default because the
  release-facing exact-xblock path is already parity-clean on its current
  bounded evidence set.
- ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_JAX_FACTORS`` (default: off): opt-in
  device-factor x-block preconditioner experiment. It exposes the existing JAX
  padded triangular-factor path through the explicit x-block solver so GPU
  diagnostics can distinguish host-SuperLU application overhead from Krylov
  convergence. The 2026-05-13 bounded probe did not promote it: even a small
  local full-FP RHSMode=1 opt-in run was manually killed after about ``76 s`` of
  setup/test time, so this path is evidence-gathering only.
- ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV=fgmres``, ``gmres-jax``,
  ``bicgstab-jax``, or ``tfqmr-jax`` (default:
  off): opt-in JAX-native device Krylov routes for the explicit x-block path.
  ``fgmres`` uses fixed-shape JAX Arnoldi/Hessenberg work arrays, supports
  iteration-dependent right preconditioners, and avoids per-iteration residual
  conversion to Python scalars. ``gmres-jax`` uses the same fixed-shape
  Arnoldi/least-squares primitive with a fixed left preconditioner so measured
  left-preconditioned QI side choices can be tested without SciPy Krylov.
  ``bicgstab-jax`` and ``tfqmr-jax`` are lower-memory short-recurrence paths
  for device assembled-operator experiments; they avoid the :math:`O(n\,m)`
  GMRES restart basis and record ``xblock_estimated_gmres_basis_nbytes``,
  ``xblock_estimated_bicgstab_work_nbytes``, and
  ``xblock_estimated_tfqmr_work_nbytes`` in solver metadata. These routes
  force the JAX-factor x-block apply path and can use the device-resident
  global-coupling correction above. Current tests cover the solver primitive,
  JIT tracing, policy parsing, and small full-system metadata. The checked
  ``docs/_static/qi_seed_robustness_scale060_device_krylov_rejected_2026_05_13.json``
  artifact shows why these routes remain experimental: the one-GPU hard seed no
  longer times out or triggers the earlier illegal-address failure, but it still
  fails strict true-residual acceptance. The follow-up
  ``docs/_static/qi_seed_robustness_scale060_device_operator_rejected_2026_05_13.json``
  artifact verifies the next operator-reuse step: the active device CSR
  assembled with ``2.88e6`` nonzeros in ``83.6 s`` and executed ``400`` device
  matvecs by ``505.7 s``, but full-restart FGMRES still timed out at ``540 s``
  without HDF5 output or a solver trace. The same artifact also records a
  follow-up ``bicgstab-jax`` trial: peak RSS dropped to ``13.6 GB`` and the run
  finished before timeout, but the short-recurrence solve diverged to
  ``2.35e102`` and output was correctly refused. The same artifact records
  ``tfqmr-jax`` and ``tfqmr-jax`` with true-residual replacement every 20
  iterations. Plain TFQMR preserved the low-memory device path but diverged in
  ``8:05``; residual replacement made the failure bounded and faster
  (``4:23``) but still diverged to ``2.35e102``. TFQMR is therefore a useful
  guardrail/diagnostic path, not a production QI replacement solver. The same
  artifact also records the device QR global-coupling FGMRES probe: the
  correction built successfully with ``32`` loads, ``20`` retained directions,
  and rank ``20``, but the run still timed out at ``540 s`` after ``475``
  device matvecs and peaked at ``46.1 GB`` RSS. This rejects QR global coupling
  as a standalone hard-seed fix while keeping the better-conditioned coarse
  implementation for future preconditioner work. A later identity-smoother
  device QR probe reached global-coupling build and Krylov start with ``30``
  retained directions and reduced the probe-coarse seed residual only from
  ``3.021487e-05`` to ``3.019236e-05``; it still timed out after ``400`` device
  matvecs, but peak RSS was lower than the preconditioned-load setup probe.
  This promotes identity smoothing as a bounded setup default, not as a QI
  hard-seed convergence fix. A
  restart-20 device-FGMRES follow-up reached ``500`` device matvecs by
  ``533.2 s`` but still timed out and increased peak host RSS to ``50.4 GB``.
  The cycle-synchronized
  restart-20 follow-up verified the new memory knob and reached ``500`` device
  matvecs by ``528.8 s``, but still timed out and peaked at ``51.0 GB`` RSS.
  That artifact is explicit blocker evidence, not a performance claim. The
  next cycle-JIT implementation changes the device-Krylov execution model: it
  JIT-compiles one restart cycle instead of tracing the entire Python-unrolled
  solver. On the same scale-0.60 seed-3 GPU hard seed, cycle-JIT FGMRES with the
  active device CSR operator finished in ``281.7 s`` and kept peak RSS near
  ``13.9 GB`` instead of the full-solver JIT ``56.6 GB`` failure. Recycled
  cycle-JIT with ``outer_k=6`` also stayed bounded (``386.4 s``, ``13.9 GB``)
  and preserved the explicit right-preconditioned side after a rejected
  right-side probe. Neither reduced the hard-seed residual below
  ``3.019e-5`` versus the ``3.021e-13`` target, so these routes are promoted as
  device/operator infrastructure only, not as QI closure.
- The same blocker artifact records the final conditioning probes for this
  push. Row equilibration solves ``D_r A x = D_r b`` and two-sided
  equilibration solves ``D_r A D_c z = D_r b`` with the physical mapping
  ``x = D_c z``. Both completed the scale-0.60 seed-3 GPU hard seed in about
  ``275 s`` with output/trace enabled, but the physical residual remained near
  ``3.02e-5``. Increasing the x-block JAX factor row cap to ``256`` also left
  the residual unchanged. A closer GPU analogue of the CPU-closing path, exact
  per-x sparse LU with left device GMRES and row cap ``1024``, reached the
  intended factors but timed out at ``540 s`` with no output. These probes
  reject scaling, ILU padding, and naive padded exact-LU transfer as production
  GPU closure strategies. The replacement compact-CSR exact-factor path stores
  only actual SuperLU factor nonzeros instead of padding every row to the widest
  factor row. On the same hard seed it built the full exact per-x factors
  (``1.10e8`` lower nonzeros, ``1.19e8`` upper nonzeros, ``2.76 GB`` estimated
  factor arrays) and reduced the padded-factor memory problem, but the run still
  timed out before writing a solver trace because the first device
  transfer/apply remained too expensive. This keeps compact CSR as useful
  infrastructure, not QI closure. The next QI algorithm must either reduce the
  factor application cost by changing the block factorization itself, or replace
  the x-block preconditioner with a residual-reducing block-Schur/coarse
  operator.
- ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_DEVICE_JIT`` (default: off): opt-in JIT
  execution for JAX-native x-block FGMRES/GMRES probes. With
  ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_DEVICE_JIT_MODE=cycle`` (default), SFINCS_JAX
  compiles a single restarted FGMRES cycle and calls it repeatedly. This avoids
  the large HLO and host-RSS spike observed when the full multi-cycle solver is
  JIT-compiled. ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_DEVICE_JIT_OUTER_K`` adds a
  small LGMRES-style recycled augmentation space for diagnostics. Cycle-JIT now
  exposes per-cycle progress callbacks, so timeout artifacts can distinguish
  setup stalls from a Krylov cycle that is actively reducing the residual. The
  2026-05-15 scale-0.60 QI seed-3 GPU probes showed that compact CSR exact
  factors reach ``solve start`` with about ``5.9--6.4 GB`` of GPU memory when
  ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` is used, but even restart-4 and
  restart-20 cycle-JIT probes did not return one full cycle within the bounded
  window. This rejects compact exact triangular solves inside device Krylov as
  the closure path for this hard seed.
- ConstraintScheme=1 moment-Schur preconditioning is rank-gated. The
  one-species QI hard seed produces a rank-``1`` moment Schur matrix for
  ``extra=2``; SFINCS_JAX now refuses that unstable pseudo-inverse by default
  instead of generating ``1e102`` residual seeds. For compact CSR JAX factors,
  the moment-Schur default is also disabled because bounded GPU evidence showed
  this setup can consume the budget before useful Krylov work. Users can still
  force it with ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_MOMENT_SCHUR=1`` for controlled
  diagnostics.
- ``SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_ROW_EQUILIBRATE`` and
  ``SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_COL_EQUILIBRATE`` (defaults:
  off): opt-in active assembled-operator scaling probes. Row equilibration is a
  left diagonal scaling; column equilibration additionally solves in scaled
  unknowns and maps back before physical diagnostics/output. Both paths are
  metadata-covered and acceptance-gated by the unscaled physical residual.
- ``SFINCS_JAX_RHSMODE1_XBLOCK_SPARSE_JAX_FACTOR_FORMAT=csr`` (default:
  ``padded``): opt-in compact CSR storage for JAX x-block SuperLU factors. This
  avoids the padded max-row storage cliff and is the preferred diagnostic format
  for exact-factor GPU probes, but it is still too expensive to promote on the
  scale-0.60 QI hard seed without a stronger residual-reducing algorithm. The
  current conclusion is explicit: the remaining GPU hard-seed blocker is not
  side selection, LGMRES rescue, QI seed construction, moment-Schur setup, or
  memory preallocation. It is the cost of applying exact compact triangular
  factors inside device Krylov. The next GPU production path should replace that
  per-iteration apply with a GPU-cheap residual-effective smoother/coarse
  preconditioner or use a documented non-autodiff host fallback for this large
  RHSMode=1 lane.
- ``SFINCS_JAX_RHSMODE1_XBLOCK_DEVICE_HOST_FALLBACK`` (default: ``auto``):
  documented non-autodiff fallback for large RHSMode=1, ConstraintScheme=1,
  three-dimensional full-FP/QI solves when the user explicitly requests a
  JAX-native x-block Krylov method (``fgmres``, ``gmres-jax``,
  ``bicgstab-jax``, or ``tfqmr-jax``). In ``auto`` mode this fallback is scoped
  to active systems above
  ``SFINCS_JAX_RHSMODE1_XBLOCK_DEVICE_HOST_FALLBACK_MIN_ACTIVE`` (default:
  ``80000``) and rewrites the requested device solver to the host x-block auto
  policy before x-block factors are built. This avoids constructing JAX factor
  arrays that the accepted path will not use, while preserving the measured
  host side-probe seed and LGMRES rescue that close the CPU hard seed. The
  solver metadata records ``xblock_device_host_fallback_used``,
  ``xblock_device_host_fallback_reason``, the requested method, the effective
  method, and ``xblock_device_host_fallback_non_autodiff=True``. Set the
  fallback to ``0`` to force the experimental device-Krylov path, or to
  ``force``/``host`` to use the host x-block policy even below the automatic
  active-size floor. This is the production-safe escape hatch for the current
  large-QI hard-seed blocker; it is not an end-to-end differentiable solver
  path.
- ``SFINCS_JAX_RHSMODE1_XBLOCK_SPARSE_JAX_FACTOR_APPLY`` (default:
  ``exact``): diagnostic compact-factor apply mode for device Krylov. ``exact``
  applies both SuperLU triangular factors, ``diagonal`` applies only the pivoted
  upper-factor diagonal, ``upper`` and ``lower`` apply one triangular factor,
  and ``identity`` bypasses local factor application. These modes are
  acceptance-gated by the unscaled true residual and are not default
  performance claims. The 2026-05-15 ``office`` GPU hard-seed diagnostics showed
  that ``diagonal`` returns synchronized restart cycles in about
  ``124--153 s`` instead of timing out before the first cycle, but it leaves the
  true residual near ``3.02e-5`` against a ``3.02e-13`` target. A row-cap-16
  exact triangular apply reduced compact factor storage to about ``33 MB`` but
  still needed about ``266 s`` and showed the same residual. This rejects
  diagonal, one-sided, and storage-only triangular approximations as QI closure
  strategies while keeping them as bounded profiling tools.
- ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_FGMRES_BLOCK_BETWEEN_CYCLES`` (default:
  off): opt-in memory diagnostic for the JAX-native FGMRES routes. When enabled,
  SFINCS_JAX synchronizes at GMRES restart boundaries in eager accelerator runs
  so queued work and basis buffers can be released before the next cycle. It is
  ignored while tracing under ``jax.jit`` and is not a convergence fix by itself;
  it exists to test whether hard-seed memory growth comes from cross-cycle
  buffer retention or from the algorithmic basis/preconditioner footprint. The
  scale-0.60 QI hard-seed evidence above shows no peak-memory gain, so this knob
  remains diagnostic-only.
- ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_POST_MINRES_STEPS`` (default: ``0``): opt-in
  matrix-free post-Krylov correction for explicit ``xblock_sparse_pc_gmres``.
  Each accepted step applies the x-block preconditioner to the current residual
  and chooses a scalar minimum-residual step using one extra operator
  application. It did not materially reduce the scale-0.50 QI residual floor
  and remains diagnostic-only.
- ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_PROBE_COARSE`` (default: off): opt-in
  pre-Krylov projected coarse correction for explicit ``xblock_sparse_pc_gmres``.
  It now initializes a zero x-block seed when no side probe or explicit
  ``x0`` is present, records ``xblock_probe_coarse_seed_initialized``, and only
  promotes the projected update if the measured true residual decreases. This
  makes the hook useful as a bounded residual preflight before expensive GPU
  Krylov launches; it is still experimental and does not alter public defaults.
- ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_POST_COARSE`` (default: off): opt-in
  multidirectional post-Krylov coarse correction for explicit
  ``xblock_sparse_pc_gmres``. When enabled, SFINCS_JAX forms a bounded
  matrix-free least-squares problem from the preconditioned residual, optional
  raw residual, flux-surface-averaged low-L residual components, and small
  source/constraint directions. The update is accepted only if the measured true
  residual decreases. This is stronger than the scalar post-minres cleanup, but
  it also did not close the QI floor; the exact-xblock-LU policy above is the
  promoted route.
- ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_POST_RESIDUAL_EQUATION`` (default: off):
  opt-in device-oriented residual-equation correction after a stalled
  x-block Krylov solve. The correction reuses cached QI coarse columns and
  their stored operator actions when available, appends bounded residual-derived
  physics directions, and solves ``min_c ||r - A Q c||_2`` in JAX arrays. The
  candidate is accepted only if the measured true residual decreases, and its
  requested steps, accepted direction count, direction labels, and before/after
  residuals are recorded in solver metadata and output diagnostics. This is the
  preferred place for future QI hard-seed experiments that need the final Krylov
  residual mode without returning to smoother, restart, or active-pattern
  tuning. The first checked scale-0.60 CPU/GPU artifacts accepted one such
  correction with ``89`` directions, reducing ``2.362283e-05 -> 2.105918e-05``
  on CPU and ``2.450895e-05 -> 2.142936e-05`` on GPU1, but both remain
  fail-closed because the production write tolerance is much tighter.
  The scoped claim for this specific hard-seed case is therefore only that the
  research path now gets below ``3e-5`` on both CPU and GPU. Closing the
  remaining gap requires new coarse residual variables, not more smoother,
  restart, or active-pattern tuning.
- ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_PROBE_COARSE`` (default: off): opt-in
  pre-Krylov seed correction for explicit ``xblock_sparse_pc_gmres``. This uses
  the same bounded least-squares correction basis as post-coarse, but applies it
  to the side-probe or user-supplied initial state before the full Krylov solve.
  On active-DOF solves the residual is expanded to full coordinates for
  physics-aware basis construction and every candidate direction is projected
  back to the active ``Nxi_for_x`` coordinate set before acceptance. The
  scale-0.60 QI seed-3 CPU probe
  ``docs/_static/qi_seed_robustness_scale060_probe_coarse_seed3_cpu.json`` used
  this hook with ``32`` requested directions and ``fsavg_lmax=4``. The accepted
  correction reduced the side-probe seed residual from ``2.57e-8`` to
  ``1.43e-8`` in ``0.29 s``; the full solve then converged in ``222.5 s`` with
  residual ratio ``3.43e-3``. This closes the bounded CPU hard-seed timeout, but
  it remains off by default. The matching one-GPU probe
  ``docs/_static/qi_seed_robustness_scale060_probe_coarse_seed3_gpu0_timeout.json``
  still timed out at ``360 s``: the side probe switched from left to right,
  probe-coarse did not apply to a physical seed, and the GPU path advanced only
  to about ``700`` matvecs before the timeout. The remaining GPU blocker is
  therefore a device-resident/preconditioner-application issue, not an illegal
  address crash or CPU convergence issue.
- ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_PROBE_COARSE_ANGULAR_LMAX`` (default:
  ``-1``): optional low-Fourier angular directions for the probe-coarse seed
  basis. These modes target global angular coupling that is invisible to a pure
  per-``x`` block inverse. The basis remains bounded by
  ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_PROBE_COARSE_MAX_DIRECTIONS`` and is accepted
  only when the measured true residual decreases.
- ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_PROBE_COARSE_ANGULAR_RESIDUAL`` (default:
  off): adds residual-weighted angular directions to the same bounded
  probe-coarse seed. This is the current best bounded CPU hard-seed variant:
  ``docs/_static/qi_seed_robustness_scale060_probe_coarse_angular_residual_seed3_cpu_2026_05_14.json``
  passes the scale-0.60 seed-3 QI case in ``170.7 s`` with residual ratio
  ``2.14e-3``, ``2024`` matvecs, active size ``81377``, and measured peak RSS
  about ``4.77 GB``. It improves the probe seed from about ``4.57e-6`` to
  ``2.85e-6`` before Krylov. This knob remains opt-in because the matching
  one-GPU run still times out without HDF5 output.
- ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED`` (default: off): opt-in
  pre-Krylov QI coarse seed. This path builds a deterministic rank-gated basis
  from block constants, species groups, radial ramps, and first angular
  harmonics, pads it into the active ``Nxi_for_x`` Krylov space, and solves a
  small least-squares problem for the seed correction. Metadata records the
  residual before/after, improvement ratio, rank, candidate count, accepted
  labels, rejection reason, and setup time. It is a genuine replacement
  candidate for QI hard seeds. The checked scale-0.60 seed-3 CPU rerun
  ``docs/_static/qi_seed_robustness_scale060_qi_coarse_seed3_cpu_2026_05_14.json``
  passes in ``248.4 s`` with residual ratio ``1.36e-3``. The QI seed correction
  itself is small, while the new angular probe-coarse directions supply the
  material pre-Krylov drop. A matching one-GPU heartbeat probe
  ``docs/_static/qi_seed_robustness_scale060_qi_coarse_seed3_gpu0_heartbeat_timeout_2026_05_14.json``
  kept the process live and bounded for ``420 s`` on the same active
  ``81377``/total ``139502`` system, but it still wrote no HDF5 or solver trace.
  That probe also forced the host-oriented LGMRES rescue on GPU, so it is
  retained only as negative liveness evidence. The follow-up GPU-compatible
  policy probe
  ``docs/_static/qi_seed_robustness_scale060_qi_coarse_seed3_gpu0_no_lgmres_timeout_2026_05_14.json``
  explicitly disabled LGMRES rescue. It switched the side probe from left to
  right GMRES, preserved the physical seed, improved that seed with the
  probe-coarse basis from ``4.57e-6`` to ``2.83e-6``, and reached ``900`` matvecs
  by ``412.7 s`` before the same bounded timeout. This is useful progress
  evidence, but still rejected because no HDF5 output or solver trace was
  written. Keep the route off by default until the GPU-compatible policy writes
  HDF5/trace within budget and is competitive.
- ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_BASIS=enriched``: A/B-test
  basis that adds radial curvature, higher angular/mixed harmonics,
  residual-independent constraint moments, and block-Schur-like species/``x``
  contrasts before rank truncation. It is intentionally not the default. The
  scale-0.60 CPU A/B run converged, but it took ``265.2 s`` and about ``5.34 GB``
  RSS, slower and larger than the accepted residual-weighted angular
  probe-coarse run, so
  ``docs/_static/qi_seed_robustness_scale060_enriched_qi_coarse_seed3_cpu_rejected_2026_05_14.json``
  marks it as a performance rejection. The matching GPU-compatible no-LGMRES
  run reached ``925`` matvecs by ``409.9 s`` and timed out; the compact
  JAX-factor device-Krylov run built ``57 MB`` compact CSR factors but timed out
  before useful Krylov progress. These artifacts are negative evidence, not QI
  promotion.
- ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_LOWER_FILL`` (default: off): local x-block
  factorization A/B hook. ``force`` replaces exact per-x SuperLU factors by
  lower-fill ILU factors even inside the exact-LU size window; companion knobs
  tune ``*_LOWER_FILL_FACTOR``, ``*_LOWER_FILL_ILU_DROP_TOL``,
  ``*_LOWER_FILL_ROW_NNZ_MAX``, and ``*_LOWER_FILL_COMPACT_ROW_NNZ_MAX``. The
  QI hard-seed tests show the tradeoff clearly: lower fill reduced peak RSS to
  about ``3.0 GB`` but stalled at residual ``~6.6e-6`` against a ``3e-11`` target
  after ``14835`` matvecs. The moderate-fill variant also failed. Keep this as
  a memory diagnostic until an adaptive acceptance/probing layer can prove both
  residual convergence and wall-time competitiveness.
- ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL`` (default: off): opt-in two-level
  global-coupling preconditioner for explicit ``xblock_sparse_pc_gmres``. It
  builds a fixed low-dimensional coarse basis from RHS-like directions,
  constraint/source rows, and flux-surface-averaged low-L moments, forms
  ``A Z`` once, and wraps the x-block preconditioner with a coarse inverse during
  Krylov rather than applying a post-hoc cleanup after Krylov stalls. The default
  mode is additive; ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL_MODE`` can be set
  to ``multiplicative`` for diagnostics. When
  ``SFINCS_JAX_RHSMODE1_XBLOCK_ACTIVE_DOF=1`` is also enabled, the same active
  index projector is applied to each coarse vector before ``A Z`` is formed, so
  the two-level wrapper now works on reduced ``Nxi_for_x`` systems instead of
  disabling itself. The scale-0.50 QI probes rejected both historical modes, so
  this remains off by default until a larger active-DOF QI/PAS probe demonstrates
  lower true residual and lower wall time. A bounded scale-0.60 seed-3 CPU probe
  with active DOFs and ``48`` requested directions built a rank-``45`` coarse
  basis in about ``23 s``, but the solve still timed out at ``240 s`` after
  ``2725`` reported matvecs without output, so the current fixed basis is kept
  as opt-in infrastructure rather than a promoted QI hard-seed fix.
- ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV=lgmres`` or ``gcrotmk`` remains a
  diagnostic-only Krylov-method toggle. On the scale-0.50 QI blocker, LGMRES
  stalled at a slightly worse residual than GMRES, fell back to GMRES, doubled
  the matrix-vector count, and ended at the same residual floor. The automatic
  LGMRES rescue is now backend-gated through the tested x-block policy helper:
  it is allowed by default on CPU, disabled by default on GPU, and only runs on
  GPU when the user explicitly forces ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_LGMRES_RESCUE=1``.
  GCROT(m,k) also underperformed right-preconditioned GMRES on the checked QI
  probe.

Current QI promotion gate:

- Further QI promotion must not be justified by another Krylov-name toggle,
  restart-only change, or side-threshold adjustment. Those routes are useful only
  as negative diagnostics unless a new preconditioner/operator path first changes
  the measured true-residual trend.
- The next admissible path must be device-resident or operator-reuse based. Before
  launching the full expensive Krylov loop, it must apply the candidate
  preconditioner, active assembled/reused operator, or physical seed correction to
  the hard-seed state or physics load bases and show a lower true residual than
  the incumbent preflight.
- CPU/GPU parity remains part of the gate. A bounded CPU rescue is not enough to
  widen public QI defaults; the matching one-GPU ``office`` gate must write HDF5
  output and a solver trace, satisfy strict true-residual acceptance, and preserve
  the bounded CPU result before any CPU/GPU multi-seed or production-resolution
  ladder is attempted.

Large geometry-rich PAS closeout:

- The production-resolution ``geometryScheme4_2species_PAS_noEr`` deck
  (``25 x 51 x 100``, ``Nx=5``) has active size ``744610`` and total size
  ``1275010``. Bounded CPU probes of the current default Schur route, generic
  sparse-PC GMRES, ``pas_tz``, ``pas_tz`` with ``Lmax=4``, ``xmg``,
  ``pas_hybrid``, BiCGStab, and LGMRES all hit the ``300 s`` gate before a
  converged output was written. A one-GPU RTX A4000 default probe also hit the
  same gate. These results are checked in as
  ``tests/reference_solver_path_artifacts/geometry4_large_pas_closeout_2026-05-09.json``.
  No threshold-only default promotion is made for this lane; the next valid
  implementation needs a new structured/chunked geometry-aware PAS
  preconditioner that avoids both global conservative sparse patterns and dense
  angular-block storage at ``Ntheta*Nzeta=1275``.
- Dense PAS-TZ memory fallback now prefers the structured ``tzfft`` fallback
  when an explicitly requested theta/zeta Schwarz fallback is rejected by the
  memory guard and the FFT fallback builder is available. This keeps the solve
  on the existing guarded true-residual path while avoiding dense
  ``(Ntheta*Nzeta)^2`` angular-block inverse storage for research-floor
  ``25 x 51 x 100 x 4`` PAS shapes.
- ``SFINCS_JAX_RHSMODE1_TOKAMAK_FP_NOER_SPARSE_PC`` and
  ``SFINCS_JAX_RHSMODE1_TOKAMAK_FP_ER_SPARSE_PC`` (default: auto). On CPU and
  GPU/CUDA, non-differentiable tokamak full-FP RHSMode=1 rows in the measured
  production-floor ``N_zeta=1`` window use the host sparse-PC GMRES route when
  the default matrix-free route is not residual/parity-clean or when theta-line
  is parity-clean but memory-heavy. Set either variable to ``0`` to force the
  older policy, or to ``1`` to force the sparse-PC route while retaining the
  remaining guards. For one-species full-FP Er decks, the Fortran-style
  ``preconditioner_species = 0`` flag is algebraically equivalent to the compact
  per-species x-block because there is no inter-species collision coupling to
  preserve. The host-assembled x-block path is therefore allowed for one species
  and still rejected for coupled multi-species systems. On the audited
  ``25 x 1 x 8 x 100`` CPU rows, this reduces the one-species Er cases from
  multi-GB dense-assembly setup to about ``0.42-0.44 GB`` RSS and ``1 s`` logged
  solve/write time while preserving strict Fortran output parity. On the RTX
  A4000 GPU audit, the same rows are residual-clean at about ``23.3 s`` (DKES)
  and ``11.0 s`` (full trajectories), with active RSS deltas near ``1.1 GB``.
  The default active-size bounds are
  ``SFINCS_JAX_RHSMODE1_TOKAMAK_FP_NOER_SPARSE_PC_MIN/MAX=10000/60000`` and
  ``SFINCS_JAX_RHSMODE1_TOKAMAK_FP_ER_SPARSE_PC_MIN/MAX=10000/60000``.
- ``SFINCS_JAX_RHSMODE1_SPARSE_PC_MAX_ESTIMATED_MB`` (default: unset). When set
  to a positive value, sparse-PC GMRES estimates CSR operator storage, GMRES
  basis storage, and SuperLU/ILU factor fill before factorization. If the
  estimate exceeds the budget, the run raises a clear ``MemoryError`` before
  entering the expensive factorization. Use
  ``SFINCS_JAX_RHSMODE1_SPARSE_PC_FACTOR_FILL_ESTIMATE`` (default: ``8``) to
  adjust the preflight fill multiplier for a known machine/case family.
- ``SFINCS_JAX_LINEAR_STAGE2_RATIO`` (default: ``1e2``). Stage-2 GMRES only runs
  when ``||r|| / target`` exceeds this ratio (set ``<= 0`` to always allow).
- ``SFINCS_JAX_PAS_STAGE2_SKIP_RATIO`` (default: ``1e6``) skips stage-2 for the
  measured PAS-lite/hybrid/tz family when the first residual is far too large
  for a polish solve. ``SFINCS_JAX_PAS_STAGE2_SKIP_EXTENDED=1`` also applies this
  diagnostic skip to ``pas_ilu``, ``schur``, and ``xblock_tz`` routes, but this
  is not a production default because the production-floor tokamak PAS+Er audit
  showed faster completed outputs that were not parity-clean.
- ``SFINCS_JAX_PAS_STAGE2_WEAK_SKIP_RATIO`` and
  ``SFINCS_JAX_PAS_STRONG_WEAK_SKIP_RATIO`` (defaults: ``1e12``) make forced
  weak PAS baselines fail fast when ``collision``, ``point``, or ``xmg`` returns
  an enormous first residual. These guards are deliberately much looser than the
  normal stage-2 trigger so moderate residuals can still use the existing polish
  and retry machinery. Set either value to ``0`` to disable the corresponding
  guard.
- ``SFINCS_JAX_PAS_WEAK_MINRES_*`` controls a cheap accept-only correction for
  the same weak PAS baselines. It performs a small number of matrix-free steps
  :math:`d=P^{-1}r`, chooses the scalar :math:`\alpha` minimizing
  :math:`\|r-\alpha A d\|_2`, and keeps the step only if the residual improves.
- ``SFINCS_JAX_TRANSPORT_DENSE_RETRY_MAX`` (default: ``6000`` for RHSMode=2/3).
- ``SFINCS_JAX_TRANSPORT_DENSE_FALLBACK`` / ``SFINCS_JAX_TRANSPORT_DENSE_FALLBACK_MAX``.
- ``SFINCS_JAX_TRANSPORT_DENSE_MAX_MB`` (default: ``128``). Disable dense transport
  fallbacks when the dense matrix would exceed this memory budget. If float64
  exceeds the limit but float32 does not, the fallback switches to float32 with
  one refinement step.
- ``SFINCS_JAX_TRANSPORT_DENSE_ACCELERATOR_AUTO`` plus
  ``SFINCS_JAX_TRANSPORT_DENSE_ACCELERATOR_AUTO_MAX`` /
  ``SFINCS_JAX_TRANSPORT_DENSE_ACCELERATOR_AUTO_GEOMETRIES`` (defaults on,
  ``2500`` and ``1``). This admits dense GPU transport only for the measured
  bounded RHSMode=3 monoenergetic geometryScheme=1 lane, avoiding the much slower
  Krylov/sparse-rescue path while preserving the explicit opt-out
  ``SFINCS_JAX_TRANSPORT_DENSE_ALLOW_ACCELERATOR=0``.
- ``SFINCS_JAX_TRANSPORT_DENSE_PRECOND_MAX_MB`` (default: ``min(32, dense_max_mb)``).
  Disables dense LU preconditioners when they would exceed the memory budget.
- ``SFINCS_JAX_TRANSPORT_GEOM5_MONO_LOW_MEMORY`` (default: auto). Set ``0`` to
  restore the previous dense batched fallback for CPU VMEC monoenergetic
  comparison runs; set ``1`` to force the low-memory path regardless of the
  backend/size guard.
- ``SFINCS_JAX_TRANSPORT_GEOM5_MONO_LOW_MEMORY_MIN`` /
  ``SFINCS_JAX_TRANSPORT_GEOM5_MONO_LOW_MEMORY_MAX`` bound the automatic
  geometryScheme=5 monoenergetic low-memory policy (defaults ``1000`` and
  ``20000`` total DOFs).
- ``SFINCS_JAX_DENSE_ASSEMBLE_JIT``: JIT-compile dense matrix assembly
  (auto by default: off for ``n<=800``, on for larger matrices).
- ``SFINCS_JAX_DENSE_MAX`` (default: ``8000``): guardrail for dense solves
  (max vector size before dense solve is disallowed).
- ``SFINCS_JAX_DENSE_BLOCK``: column block size for dense assembly (auto block size
  of ``128`` is used when ``n>=1000`` and no explicit block is set).
- ``SFINCS_JAX_DENSE_BLOCK``: assemble dense matrices in column blocks to cap peak memory.

**Impact.**

Keeps transport runs performance-first while improving stability for parity-sensitive
cases where Krylov solvers can stall.

Memory reduction: remat/checkpoint + short recurrence
-----------------------------------------------------

**Rematerialization.**

- ``SFINCS_JAX_REMAT_COLLISIONS`` and ``SFINCS_JAX_REMAT_TRANSPORT_DIAGNOSTICS``
  enable `jax.checkpoint` around large operator/diagnostic kernels to reduce
  peak memory during autodiff.

**Short recurrence.**

BiCGStab avoids storing a full GMRES basis, reducing memory pressure in large
transport and multispecies FP runs.

Memory reduction: operator representation and measured gates
------------------------------------------------------------

The public benchmark table reports active solver memory, but lowering the
actual footprint requires reducing the arrays allocated by the numerical path.
The local Fortran v3 source and PETSc documentation point to the important
distinction: SFINCS v3 keeps memory controlled by using sparse AIJ matrices with
careful nonzero preallocation, thresholded value insertion, PETSc KSP/PC reuse,
fill-reducing orderings for direct solves, and distributed row ownership. JAX
allocator settings can change when memory is reserved, but they do not by
themselves replace dense intermediates, Krylov basis storage, or replicated
preconditioner state.

``sfincs_jax.solvers.memory_model`` provides a small conservative model for the dominant
linear-solve terms:

.. math::

   M_\mathrm{dense} \approx n^2 b
      + (r+1+n_\mathrm{work}) n b
      + M_\mathrm{pc} + M_\mathrm{tmp},

.. math::

   M_\mathrm{csr} \approx \mathrm{nnz}(b+b_i) + (n+1)b_i
      + (r+1+n_\mathrm{work}) n b
      + M_\mathrm{pc} + M_\mathrm{tmp},

where :math:`n` is the active unknown count, :math:`r` is the GMRES restart,
:math:`b` is scalar storage, :math:`b_i` is index storage, and
:math:`M_\mathrm{pc}` / :math:`M_\mathrm{tmp}` are preconditioner and compiled
temporary estimates when available. The GMRES restart cap in
``sfincs_jax.solver`` now uses this shared model, and
``sfincs_jax.solvers.selection_policy`` can compare candidate routes using paired
memory metrics in priority order: device peak memory, active RSS, compiled
temporary memory, and finally legacy process peak RSS. This avoids promoting a
route by comparing GPU device memory against CPU process RSS or by hiding a true
solver-state regression behind the fixed Python/JAX runtime baseline.

The intended production gate for future memory optimizations is:

- preflight the route with ``estimate_linear_solve_memory(...)`` before dense
  assembly, sparse probing, or large-preconditioner construction;
- collect measured active RSS and, on accelerators, JAX device-memory profiles;
- collect compiled temporary estimates from JAX
  ``lower(...).compile().memory_analysis()`` when the backend reports them;
- auto-promote a candidate only when it is residual-clean, parity-clean, and
  materially faster or lower-memory than the incumbent on the same memory metric.

Solver traces now carry both measured and estimated memory fields. The JSON
sidecar records ``active_rss_mb``, ``device_peak_mb`` when available,
``estimated_dense_nbytes``, ``estimated_csr_nbytes``,
``estimated_gmres_basis_nbytes``, and a ``metadata.memory_estimate`` block with
dense/CSR totals and per-device estimates. Sparse-PC traces also include the
actual GMRES restart, maximum iteration count, diagonal shift, sparse pattern
nonzeros, pattern-build time, preconditioner-factor time, and SuperLU ``L``/``U``
factor storage estimates. This makes the next optimization pass auditable: a
claimed memory win must reduce the measured active/device metric and should also
reduce the estimated dominant storage term.

The structured PAS-TZ memory fallback remains benchmark-only until a bounded
route clears both runtime and residual gates. Use
``scripts/benchmark_pas_tz_memory_fallback.py`` to run forced ``collision``,
``hybrid``, ``theta``, ``zeta``, and ``tzfft`` fallback variants in subprocesses
with hard timeouts.
The companion ``scripts/benchmark_rhs1_pas_matrixfree.py`` production-solve
planner can pass the same default-promotion baseline gate into those subprocess
runs via the ``--production-solve-require-default-promotion-gate`` controls.
The fallback benchmark also requires provenance that the guarded PAS-TZ fallback
path actually ran. Rows must contain a guarded PAS-TZ message such as the
structured-fallback guard, guarded correction, or guarded retry-skip trace before
they can pass all gates or appear as promotion-eligible variants. This blocks
false-positive runtime/residual wins from unrelated dispatch paths.
Dry-run JSON is planning evidence only: ``summary.all_gates_passed`` remains
``false`` until at least one real child row is present, and
``summary.promotion_ready`` is true only when all row gates pass and at least one
row records a material runtime or RSS win. Guarded-correction evidence may be
recorded in nested row metadata, including lists of candidate records, but still
must prove ``stream_update_chunks=true`` or
``pas_tz_guarded_correction_streamed=true`` without
``full_update_materialized=true``.
The checked smoke artifact
``tests/reference_solver_path_artifacts/pas_tz_memory_fallback_geometry4_smoke_2026-05-10.json``
records the intended guard behavior on the geometryScheme=4 PAS deck. The
cheap collision and guarded structured rows return in about ``1.6 s`` with
residual ``6.4e5``; the explicit legacy ``hybrid`` row returns in about
``2.3 s`` but with residual ``2.5e16``. The experimental ``tzfft`` row returns
in about ``3.3 s`` with residual ``1.9e-4`` and a higher RSS footprint
(``~944 MB`` in the checked smoke). This is a useful matrix-free residual
improvement, but it still misses the strict residual target and is not promoted
to a default solver path. The next algorithmic step is a genuinely stronger
matrix-free or chunked PAS correction that reduces the residual without
constructing dense angular patch inverses.

A follow-on cheap-base correction probe is also available with
``SFINCS_JAX_RHSMODE1_PAS_TZ_MEMORY_FALLBACK=collision`` and
``SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_CORRECTION=tzfft``. On the same bounded
geometryScheme=4 smoke it returned in about ``2.0 s`` and lowered the residual
from ``6.4e5`` to ``1.3e5`` with RSS about ``728 MB``. Increasing the correction
steps to ``10`` did not materially improve the residual. This is therefore kept
as an explicit profiling tool, not as a promoted default.
The helper rejects exactly zero updates and zero-relaxation attempts before
running a candidate residual matvec, so no-op PAS correction probes now spend one
matrix-free operator application instead of two while preserving the same reject
reason and residual history.
It also preflights candidate vector element and byte budgets before calling the
correction builder. ``scripts/benchmark_rhs1_pas_matrixfree.py`` exposes this as
``--max-candidate-elements`` and ``--max-candidate-bytes`` so production-floor
PAS probes can fail closed before materializing large update or residual arrays.
The byte budget is now also a promotion gate: every planned bounded probe writes
``byte_preflight`` provenance, metadata-backed production-floor probes write a
conservative ``production_floor_byte_preflight`` record using the estimated full
unknown count and float64 item size, and
``--production-solve-require-default-promotion-gate`` refuses to run unless
``--max-candidate-bytes`` is supplied and the selected target is byte-safe. This
keeps diagnostic dry-runs useful without allowing a memory-unsafe PAS candidate
to be promoted from an unrelated runtime/residual artifact.

The real-solve launcher now applies the same rule before *any* opt-in
production-floor PAS probe: ``--run-production-solve-probe`` requires an
explicit ``--max-candidate-bytes`` budget by default.  A developer can still
force an unbudgeted diagnostic with
``--production-solve-allow-unbudgeted-candidate``, but that opt-out is recorded
in the plan JSON and is not suitable for promotion evidence.  The checked
``docs/_static/rhs1_pas_matrixfree_byte_budget_gate_2026_05_15.json`` dry-run
shows the intended state for the current production-floor candidates: geometry4
and HSX have byte-safe, launchable short-probe plans under the configured
budget; geometry11 remains held because the checked artifact evidence is not yet
complete enough for a promotion-facing real solve.

The corresponding byte-budgeted real-solve probes are checked in as negative
promotion evidence, not as defaults.  The geometry4 probe
(``docs/_static/rhs1_pas_production_solve_geometry4.json``) reaches a clean
true residual of about ``5.9e-8`` with stable GMRES solver metadata, but its
``tzfft`` and ``tzfft_lgmres`` candidates both regress runtime and peak RSS
relative to the checked baseline.  The HSX probe
(``docs/_static/rhs1_pas_production_solve_hsx.json``) reaches a clean residual
near ``1.6e-4``; ``tzfft`` is a runtime regression and ``tzfft_lgmres`` is a
memory regression.  These artifacts are useful because they prevent a
memory-safe but slower route from being accidentally promoted.  The matrix-free
PAS artifact preflight treats ``summary.all_gates_passed=false`` as blocking
even when the residual/runtime/RSS absolute gates pass, requires that row-gate
summary for ready evidence, and preserves the row-level promotion failure
reasons so residual-clean regressions remain negative evidence.  The next PAS
optimization must reduce algorithmic cost, for example by chunking/streaming
the correction or by adding a geometry-aware residual-reducing update that does
not materialize dense blocks.

The same fail-fast rule is now applied to forced weak PAS baselines at
astronomical residual ratios, with a cheap accept-only minres correction before
the fail-fast decision. On the geometryScheme=4 smoke deck, forced
``collision``, ``xmg``, and ``point`` preconditioners no longer enter minutes of
stage-2 or strong-preconditioner retry/search after first residual ratios near
``1e15``. ``collision`` returns in about ``1.2 s`` with residual improved from
``1.58e6`` to ``1.27e6`` and about ``0.6 GB`` RSS; ``xmg`` returns in about
``1.35 s`` with residual improved from ``2.53e6`` to ``2.44e6``; ``point``
returns in about ``2.8 s`` and ends at the same ``1.27e6`` residual but still
uses about ``1.9 GB`` RSS. This is retained as an auditable negative benchmark,
not as a production recommendation.

The remaining high-impact memory lanes are algorithmic:

- prefer matrix-free operators with structured preconditioners for production
  RHSMode=1 whenever an explicit dense operator is not provably cheaper;
- build CSR operators only from known/probed sparsity patterns and avoid dense
  ``from_matvec`` materialization for large systems;
- use short-recurrence Krylov methods or smaller restarted GMRES when residual
  quality stays within the physics gate;
- use mixed-precision preconditioners with float64 residual refinement where the
  measured residual and output parity stay clean;
- shard theta/zeta/radius/species work over CPU/GPU meshes so preconditioner
  state and matvec buffers are distributed instead of replicated;
- consider Pallas kernels only for proven stencil/matvec hotspots where XLA
  creates large temporaries that block memory-limited GPUs;
- evaluate Lineax/Equinox operators only behind the same gate, since their main
  value here is a cleaner matrix-free operator abstraction and differentiable
  solver state, not automatic memory reduction.

JAX ecosystem gates
-------------------

External JAX libraries stay behind measured gates until they show a real
reliability or performance advantage on SFINCS-owned workloads. These gates are
optional and must degrade to skipped rows when their packages are unavailable.

Lineax is currently a candidate for differentiable linear-solve experiments, not
a production CLI backend. The lightweight synthetic gate is:

.. code-block:: bash

   python examples/performance/benchmark_optional_lineax_implicit_solve.py \
     --backend all \
     --suite synthetic \
     --size 4 \
     --restart 4 \
     --maxiter 60 \
     --out-json /tmp/sfincs_lineax_synthetic_gate.json

On the local 2026-05-12 smoke, the in-tree solve row was residual-clean
(``relative_residual ~= 1.6e-16``) with gradient error about ``7.8e-12`` and
elapsed time about ``0.89 s``. The Lineax row was also residual-clean
(``relative_residual = 0``) with gradient error about ``2.3e-12`` and elapsed
time about ``0.36 s``. This supports keeping Lineax under evaluation, but it
does not promote it to production because the real tiny SFINCS gate still has to
stay status-clean, parity-clean, and faster or lower-memory.

Equinox and the historical opt-in JAXopt backend are checked as
objective-wrapper tooling around a real ``geometryScheme=4`` differentiable
objective:

.. code-block:: bash

   python examples/optimization/benchmark_optional_eqx_jaxopt_scheme4_gate.py \
     --backend all \
     --n-theta 17 \
     --n-zeta 17 \
     --maxiter 5 \
     --stepsize 0.1 \
     --out-json /tmp/sfincs_eqx_jaxopt_gate.json

On the local 2026-05-12 opt-in smoke, the Equinox wrapper matched a centered
finite-difference directional derivative with about ``1.1e-11`` absolute error
and elapsed time about ``0.038 s``. The JAXopt gradient-descent row reduced the
loss to a ratio of about ``4.1e-14`` and recovered the target harmonic
amplitudes to about ``1.6e-08`` in Euclidean norm with elapsed time about
``0.15 s``. Default CI does not install JAXopt; it verifies that this row skips
cleanly while keeping the Equinox wrapper gate active.

The focused regression test for the measured summaries is:

.. code-block:: bash

   python -m pytest tests/test_optional_ecosystem_gates.py -q

Promotion rule: optional ecosystem libraries may only move closer to production
when the JSON gate contains measured ``ok`` rows for the relevant real SFINCS
case, clean residual or gradient metrics, and no hard dependency on the optional
package in default installs.

Geometry parsing cache
----------------------

`sfincs_jax` caches parsed Boozer geometry files (.bc) by content hash and
geometry scheme to avoid repeated parsing for multiple runs of the same equilibrium.

Implementation: ``sfincs_jax.geometry`` and ``sfincs_jax.discretization.v3``.

F-block operator cache
----------------------

`sfincs_jax` can reuse geometry- and physics-dependent operator blocks across
repeated runs with identical inputs (e.g., scans that only change :math:`E_r`).
This avoids rebuilding collisionless, collision, and magnetic-drift operators.

Controls:

- ``SFINCS_JAX_FBLOCK_CACHE`` (default: enabled)
- ``SFINCS_JAX_FBLOCK_CACHE_MAX`` (max cached entries; default: ``8``)

Implementation: ``sfincs_jax.operators.profile_response.fblock``.

Performance deltas (where measured)
-----------------------------------

The project maintains benchmark scripts and figures in ``docs/_static/figures/``:

- ``transport_compile_runtime_cache_2x2.png``: runtime vs cache effects.
- ``sfincs_vs_sfincs_jax_l11_runtime_2x2.png``: v3 vs JAX runtime comparison.

These figures are updated as part of the benchmarking workflow
(``examples/performance/``).

For quick reproduction:

.. code-block:: bash

   python examples/performance/benchmark_transport_l11_vs_fortran.py --repeats 4
   python examples/performance/profile_transport_compile_runtime_cache.py --repeats 3

Implementation map (source code)
--------------------------------

Key modules and functions referenced above:

- **Operator apply + caching**:

  - ``sfincs_jax/operators/profile_response/system.py``: ``apply_v3_full_system_operator_cached``,
    ``_operator_signature_cached``.

- **Transport solver + preconditioners**:

  - ``sfincs_jax/v3_driver.py``: ``solve_v3_transport_matrix_linear_gmres``,
    ``_build_rhsmode23_sxblock_preconditioner``,
    ``_build_rhsmode23_collision_preconditioner``.

- **Diagnostics and flux formulas**:

  - ``sfincs_jax/problems/transport_matrix/diagnostics.py``:
    ``v3_transport_diagnostics_vm_only_precompute``,
    ``v3_transport_diagnostics_vm_only_batch_op0_precomputed``.

- **Solver backends**:

  - ``sfincs_jax/solver.py``: GMRES/BiCGStab wrappers, dense fallback options,
    memory-aware restart logic.


Summary of tuning knobs
-----------------------

See :doc:`usage` for the full environment variable reference. The most important
performance controls are:

- Solver selection and fallback: ``SFINCS_JAX_RHSMODE1_SOLVE_METHOD``,
  ``SFINCS_JAX_BICGSTAB_FALLBACK``.
- Transport preconditioning: ``SFINCS_JAX_TRANSPORT_PRECOND``.
- Diagnostics precompute: ``SFINCS_JAX_TRANSPORT_DIAG_PRECOMPUTE``.
- Remat thresholds: ``SFINCS_JAX_REMAT_COLLISIONS(_MIN)``,
  ``SFINCS_JAX_REMAT_TRANSPORT_DIAGNOSTICS(_MIN)``.
- Active DOF: ``SFINCS_JAX_ACTIVE_DOF`` and ``SFINCS_JAX_TRANSPORT_ACTIVE_DOF``.

References (vendored)
---------------------

- SFINCS v3 technical manual:
  :download:`20150507-01 Technical documentation for version 3 of SFINCS.pdf <upstream/20150507-01 Technical documentation for version 3 of SFINCS.pdf>`
- Landreman, Smith, Mollen, Helander (2014), PoP 21 042503:
  :download:`LandremanSmithMollenHelander_2014_PoP_v21_p042503_SFINCS.pdf <upstream/LandremanSmithMollenHelander_2014_PoP_v21_p042503_SFINCS.pdf>`
- Technical note on the FP operator:
  :download:`20150402-01 Implementation of the Fokker-Planck operator.pdf <upstream/20150402-01 Implementation of the Fokker-Planck operator.pdf>`
- “Generalized minimal residual method,” Wikipedia (accessed 2025), which cites the original
  GMRES method of Saad & Schultz (1986): https://en.wikipedia.org/wiki/Generalized_minimal_residual_method
- H. A. van der Vorst, “Bi-CGSTAB: A fast and smoothly converging variant of Bi-CG,”
  SIAM J. Sci. Stat. Comput. 13(2):631–644 (1992). DBLP: https://dblp.org/rec/journals/sisc/Vorst92.html
- P. Sonneveld and M. B. van Gijzen, “IDR(s): A family of simple and fast algorithms for
  solving large nonsymmetric linear systems,” SIAM J. Sci. Comput. 31(2):1035–1062 (2008).

External performance references
-------------------------------

- PETSc sparse AIJ preallocation and distributed sparse matrices:
  https://petsc.org/main/manual/mat/
- PETSc matrix-free shell matrices and custom shell preconditioners:
  https://petsc.org/main/manualpages/Mat/MATSHELL/
- PETSc GMRES restart memory tradeoff:
  https://petsc.org/release/manualpages/KSP/KSPGMRESSetRestart/
- JAX GPU memory allocation controls:
  https://docs.jax.dev/en/latest/gpu_memory_allocation.html
- JAX device-memory profiling:
  https://docs.jax.dev/en/latest/device_memory_profiling.html
- JAX compiled executable memory analysis:
  https://docs.jax.dev/en/latest/jax.stages.html
- JAX buffer donation:
  https://docs.jax.dev/en/latest/buffer_donation.html
- JAX rematerialization/checkpointing and activation offload:
  https://docs.jax.dev/en/latest/gradient-checkpointing.html
  and https://docs.jax.dev/en/latest/notebooks/host-offloading.html
- JAX sharding/memory kinds:
  https://docs.jax.dev/en/latest/jax.sharding.html
- JAX Pallas kernels:
  https://docs.jax.dev/en/latest/pallas/index.html
- Lineax matrix-free linear operators and solver state reuse:
  https://docs.kidger.site/lineax/api/operators/
  and https://docs.kidger.site/lineax/api/linear_solve/
  DBLP: https://dblp.org/rec/journals/sisc/SonneveldG08.html
