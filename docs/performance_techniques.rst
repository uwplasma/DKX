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

- Core operator apply: ``sfincs_jax.v3_system.apply_v3_full_system_operator_cached``.
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
  ``sfincs_jax.structured_velocity.factor_block_tridiagonal``.
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

- ``sfincs_jax.v3.geometry_from_namelist`` persists the full ``BoozerGeometry`` arrays
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

The implementation lives in ``sfincs_jax.v3_sparse_pattern`` and
``sfincs_jax.explicit_sparse.build_operator_from_pattern``. Tests verify that the
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
(``Ntheta x Nzeta x Nx x Nxi``) for 3D cases and ``25 x 1 x 4 x 100`` for
tokamak cases, with a ``10 s`` minimum SFINCS Fortran v3 timing target for
public production rows. Performance claims for this lane should be regenerated
from that manifest rather than from earlier lower-resolution bring-up probes. It is
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

- ``sfincs_jax.implicit_solve.linear_custom_solve`` and
  ``linear_custom_solve_with_residual``.
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
  ``sfincs_jax.explicit_sparse.build_operator_from_matvec`` and
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
``min(SFINCS_JAX_RHSMODE1_DENSE_ACTIVE_CUTOFF, 5000)``). Setting this
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
- ``SFINCS_JAX_PRECOND_MAX_MB`` / ``SFINCS_JAX_PRECOND_CHUNK`` (cap memory during block assembly)
- ``SFINCS_JAX_PRECOND_PAS_MAX_COLS`` (additional column cap for PAS block assembly;
  reduces peak RSS by chunking :math:`(\theta,\zeta)` blocks)
- ``SFINCS_JAX_PRECOND_DTYPE`` (default ``auto``; ``float32`` or ``float64`` to override)
- ``SFINCS_JAX_PRECOND_FP32_MIN_SIZE`` (threshold for auto mixed precision)

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

- ``SFINCS_JAX_FUSED_MATVEC`` (default enabled) in ``sfincs_jax.v3_fblock``.

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
  ``sfincs_jax.periodic_stencil.extract_sparse_row_stencil`` and
  ``sfincs_jax.periodic_stencil.apply_sparse_row_stencil_gather``.
- Collisionless operator fast path:
  ``sfincs_jax.collisionless.apply_collisionless_v3``.
- Operator build wiring:
  ``sfincs_jax.v3_fblock.collisionless_operator_from_namelist``.

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

These are implemented in ``sfincs_jax.transport_matrix`` with strict-order
reductions matching v3 when required.

**Precompute constants + cache.**

Factors depending only on geometry, species normalization, and grids
(:math:`w_x`, :math:`B/D`, prefactors, etc.) are precomputed once per transport run
and reused for all ``whichRHS`` solves.

**Implementation.**

- ``v3_transport_diagnostics_vm_only_precompute`` and
  ``v3_transport_diagnostics_vm_only_batch_op0_precomputed``.
- Cached by operator signature in ``sfincs_jax.transport_matrix`` to reuse
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
  ``sfincs_jax.transport_matrix``.
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
- ``SFINCS_JAX_RHSMODE1_DENSE_FP_MAX`` (default: ``5000``) for full Fokker–Planck
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
- ``SFINCS_JAX_RHSMODE1_TOKAMAK_PAS_ER_SPARSE_PC`` (default: auto). On CPU/GPU,
  non-differentiable tokamak PAS+Er full-trajectory RHSMode=1 runs in the
  measured production-floor window use the host sparse-PC GMRES route. This
  avoids the PAS-ILU/Schur stage-2 stall while preserving Fortran parity for the
  audited one- and two-species ``25 x 1 x 8 x 100`` CPU and RTX A4000 GPU cases.
  The route defaults to
  ``SFINCS_JAX_RHSMODE1_SPARSE_PC_SHIFT=1e-8`` and
  ``SFINCS_JAX_EXPLICIT_SPARSE_DIAG_PIVOT_THRESH=0`` for constrained PAS unless
  the user overrides those values. Measured default auto-selection runtimes are
  about ``23 s``/``45 s`` on CPU for one-/two-species and ``44 s``/``86 s`` on an
  RTX A4000 GPU for the same one-/two-species production-floor cases, all with
  zero Fortran output mismatches.
- ``SFINCS_JAX_RHSMODE1_TOKAMAK_FP_NOER_SPARSE_PC`` and
  ``SFINCS_JAX_RHSMODE1_TOKAMAK_FP_ER_SPARSE_PC`` (default: auto). On GPU/CUDA,
  non-differentiable tokamak full-FP RHSMode=1 rows in the measured
  production-floor ``N_zeta=1`` window use the host sparse-PC GMRES route when
  the default matrix-free route is not residual/parity-clean or when theta-line
  is parity-clean but memory-heavy. Set either variable to ``0`` to force the
  older policy, or to ``1`` to include CPU while retaining the remaining guards.
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

``sfincs_jax.memory_model`` provides a small conservative model for the dominant
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
``sfincs_jax.solver_selection_policy`` can compare candidate routes using paired
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

Geometry parsing cache
----------------------

`sfincs_jax` caches parsed Boozer geometry files (.bc) by content hash and
geometry scheme to avoid repeated parsing for multiple runs of the same equilibrium.

Implementation: ``sfincs_jax.geometry`` and ``sfincs_jax.v3``.

F-block operator cache
----------------------

`sfincs_jax` can reuse geometry- and physics-dependent operator blocks across
repeated runs with identical inputs (e.g., scans that only change :math:`E_r`).
This avoids rebuilding collisionless, collision, and magnetic-drift operators.

Controls:

- ``SFINCS_JAX_FBLOCK_CACHE`` (default: enabled)
- ``SFINCS_JAX_FBLOCK_CACHE_MAX`` (max cached entries; default: ``8``)

Implementation: ``sfincs_jax.v3_fblock``.

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

  - ``sfincs_jax/v3_system.py``: ``apply_v3_full_system_operator_cached``,
    ``_operator_signature_cached``.

- **Transport solver + preconditioners**:

  - ``sfincs_jax/v3_driver.py``: ``solve_v3_transport_matrix_linear_gmres``,
    ``_build_rhsmode23_sxblock_preconditioner``,
    ``_build_rhsmode23_collision_preconditioner``.

- **Diagnostics and flux formulas**:

  - ``sfincs_jax/transport_matrix.py``: ``v3_transport_diagnostics_vm_only_precompute``,
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
