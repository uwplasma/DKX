Usage
=====

.. note::

   For RHSMode=1/2/3 cases the recommended entry points are the canonical
   drivers (``dkx.run.run_profile`` and the transport-matrix runners)
   shown in the quickstart on :doc:`index` and in :doc:`examples`. This page
   documents the full CLI and Python surface — the matrix-free operator, the
   three-tier solve, the output writers, ``Er`` scans, and the ambipolar and
   transport-matrix runners — including the advanced controls that most scripts
   do not need.

Parsing an input file
---------------------

.. code-block:: python

   from dkx.namelist import read_sfincs_input

   nml = read_sfincs_input("input.namelist")
   print(nml.group("geometryParameters")["GEOMETRYSCHEME"])

Building v3 grids and geometry
------------------------------

.. code-block:: python

   from dkx.drift_kinetic import kinetic_operator_from_namelist

   op = kinetic_operator_from_namelist(nml)
   print(op.n_theta, op.n_zeta, op.n_x)   # grid sizes
   print(op.b_hat.shape)                  # (Ntheta, Nzeta) geometry arrays

Supported geometry examples
---------------------------

The quickest runnable geometry-specific entry points in the repository are:

.. code-block:: bash

   python examples/getting_started/write_sfincs_output_tokamak.py
   python examples/getting_started/write_sfincs_output_vmec.py

These cover the supported analytic tokamak ``geometryScheme=1`` path and the
VMEC ``geometryScheme=5`` workflow with an explicit ``wout_path`` override.
For simplified Boozer and `.bc` workflows, use the examples under
``examples/sfincs_examples/`` or the tiny parity fixtures under ``tests/ref``.

Applying the operator directly
------------------------------

On the canonical stack the whole drift-kinetic operator is the single
consolidated :class:`dkx.drift_kinetic.KineticOperator`. Build it from a
parsed namelist and apply it matrix-free — no manual assembly of streaming,
mirror, drift, or collision blocks is needed:

.. code-block:: python

   import jax.numpy as jnp
   from pathlib import Path
   from dkx.inputs import parse_sfincs_input_text
   from dkx.drift_kinetic import kinetic_operator_from_namelist

   raw = parse_sfincs_input_text(Path("input.namelist").read_text())
   op = kinetic_operator_from_namelist(raw)

   b = op.rhs()                     # right-hand side S_s (thermodynamic + inductive drives)
   y = op.apply(jnp.zeros_like(b))  # matrix-free A @ v on the full state vector

The collision blocks are assembled into the operator automatically from
``collisionOperator``: pitch-angle scattering (``= 1``) or the full
Fokker--Planck / Rosenbluth operator (``= 0``), both from
:mod:`dkx.collisions` and applied inside ``KineticOperator.apply_f`` (the
distribution-block-only action). To solve the assembled system, hand the
operator to :func:`dkx.solve.solve` (the three-tier auto policy in
:doc:`numerics`), or use the high-level :func:`dkx.run.run_profile`.

Running the Fortran v3 executable
---------------------------------

.. code-block:: bash

   export SFINCS_FORTRAN_EXE=/path/to/sfincs/fortran/version3/sfincs
   dkx run-fortran --input /path/to/input.namelist

.. tip::

   All CLI subcommands support ``-v/--verbose`` (repeatable), ``-q/--quiet``,
   and ``--fortran-stdout``/``--no-fortran-stdout`` for strict stdout mirroring.
   These shared flags can be given either before or after the subcommand.

If you are developing from a source checkout and have not installed the console script,
you can invoke the CLI module directly:

.. code-block:: bash

   python -m dkx run-fortran --input /path/to/input.namelist

First CLI run
-------------

The repository ships a tiny runnable input for quick installation checks:

.. code-block:: bash

   dkx examples/sfincs_examples/quick_2species_FPCollisions_noEr/input.namelist --out sfincsOutput.h5
   dkx --plot sfincsOutput.h5

The first command solves the input with the default ``auto`` policy and writes
``sfincsOutput.h5`` in the current working directory. The second command writes
``sfincsOutput_summary.pdf`` next to it unless ``--out`` is given explicitly.
For normal production use, this is the intended public contract: provide one
input file, optionally override the equilibrium file, and let ``dkx``
choose the validated solve path.
Change only the output suffix to write NetCDF4 or NPZ instead:

.. code-block:: bash

   dkx examples/sfincs_examples/quick_2species_FPCollisions_noEr/input.namelist --out sfincsOutput.nc
   dkx examples/sfincs_examples/quick_2species_FPCollisions_noEr/input.namelist --out sfincsOutput.npz

Advanced linear-state export
----------------------------

.. code-block:: bash

   dkx solve-v3 --input /path/to/input.namelist --out-state stateVector.npy

.. code-block:: bash

   python -m dkx solve-v3 --input /path/to/input.namelist --out-state stateVector.npy

.. note::

   The matrix-free solve path is parity-tested on a growing subset of v3 options.
   In particular, VMEC ``geometryScheme=5`` is supported for the parity-tested tiny PAS case
   (see ``tests/ref/pas_1species_PAS_noEr_tiny_scheme5.input.namelist``).

.. note::

   For end-to-end differentiation, build inputs via the Python API and keep the computation in JAX.
   File I/O, VMEC/Boozer parsing, and SciPy-based solver-history logging use NumPy and are not
   differentiable. Disable strict stdout/history logging with
   ``DKX_FORTRAN_STDOUT=0`` when tracing gradients.

Advanced solver controls
------------------------

Most runs should keep ``--solve-method auto``. The automatic policy is measured
against parity, residual, runtime, and memory gates before a branch is promoted.
Only force a solver when reproducing a benchmark, debugging a path choice, or
running an expert study where the output ``linearSolver*`` diagnostics and
``--solver-trace`` sidecar will be inspected.

The ``dkx.solve`` policy accepts an explicit ``method`` (CLI
``--solve-method``): ``auto`` (the default), ``block_tridiagonal`` and
``block_tridiagonal_truncated`` (the tier-1 structured direct solves),
``gmres`` (the tier-2 recycled Krylov solve), and ``direct`` (the tier-3 host
sparse-direct referee). The three-tier ``auto`` policy owns the supported
surface and routes each deck to the cheapest adequate tier (:doc:`numerics`);
an unrecognized method name raises an error. These names are intentionally
advanced API: scripts for general users should omit them and rely on ``auto``.

Parallel CLI controls
---------------------

The executable path exposes the main parallel runtime controls directly, so
you do not need to rely on undocumented shell environment setup for common
one-node and multi-host runs.

.. code-block:: bash

   # Multi-core CPU host devices + auto sharding
   dkx --cores 8 --shard-axis auto /path/to/input.namelist

   # RHSMode=2/3 transport-matrix run (canonical stack; all whichRHS drives
   # are solved in one shared multi-RHS solve)
   dkx transport-matrix-v3 \
     --input /path/to/input.namelist

   # High-nu publication pilot with one transport RHS worker per visible GPU
   CUDA_VISIBLE_DEVICES=0,1 \
   python examples/publication_figures/generate_sfincs_paper_figs.py \
     --case lhd \
     --collision-operators 0 \
     --nuprime-min 17.78279101649707 \
     --nuprime-max 17.78279101649707 \
     --n-points 1 \
     --transport-workers 2 \
     --transport-parallel-backend gpu \
     --scan-only

   # Experimental one-node single-case multi-GPU sharded solve
   CUDA_VISIBLE_DEVICES=0,1 \
   dkx write-output \
     --input /path/to/input.namelist \
     --shard-axis theta \
     --distributed-gmres auto \
     --distributed-krylov auto

   # RHSMode=2/3 transport-matrix run on a selected GPU
   CUDA_VISIBLE_DEVICES=0 \
   dkx transport-matrix-v3 \
     --input /path/to/input.namelist

   # Multi-host JAX bootstrap for sharded solves
   dkx write-output \
     --input /path/to/input.namelist \
     --distributed \
     --process-count 8 \
     --process-id ${RANK} \
     --coordinator-address node0 \
     --coordinator-port 1234

Relevant CLI flags:

- ``--cores``: request multiple host CPU devices before JAX loads.
- ``--transport-workers``: run independent ``whichRHS`` solves in parallel
  worker processes on the legacy RHSMode=2/3 output path (scan/export_f
  workflows); the canonical ``transport-matrix-v3`` driver solves all drives
  in one shared multi-RHS solve.
- ``--shard-axis {auto,off,theta,zeta,x,flat}``: choose the single-solve sharding
  mode for the executable path.
- ``--distributed-gmres`` and ``--distributed-krylov``: control distributed
  Krylov selection on sharded RHSMode=1 solves.
- ``--distributed``, ``--process-id``, ``--process-count``,
  ``--coordinator-address``, and ``--coordinator-port``: enable one-node or
  multi-host JAX distributed initialization from the CLI.
- ``--shard-pad`` / ``--no-shard-pad``: control neutral padding when the sharded
  dimension is not divisible by the visible device count.
- ``--plot /path/to/sfincsOutput.h5``: top-level shortcut for writing a PDF
  diagnostics panel from an existing output file. HDF5, NetCDF4, and NPZ outputs
  are supported.

For actual scaling measurements, prefer the benchmark scripts in
``examples/performance`` over ad hoc shell timing. They handle warmup, backend
selection, cache reuse, and output JSON/figure generation consistently.

At verbosity level ``-v`` or higher, the CLI prints the active parallel
runtime summary (requested cores, host-device count, shard axis, transport
worker mode, distributed Krylov settings, and multi-host bootstrap fields).
This is the supported way to verify what the executable is actually doing on a
workstation or cluster launch.

Environment variables
---------------------

Defaults are chosen for robust, validated production runs. Solver selection is
governed by ``--solve-method``/``method`` and the ``auto`` three-tier policy
(:doc:`numerics`), not by environment variables. The variables below are
runtime, parallel-execution, cache, diagnostic, and parity overrides; the few
that touch a solve are numerical or parity overrides for reproducing a benchmark
or debugging a path choice, not stable solver selectors.

Precision, cache, and data files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``dkx`` runs in float64 and enables the JAX persistent compilation cache
on import.

- ``JAX_COMPILATION_CACHE_DIR``: standard JAX persistent compilation-cache
  directory, reused across runs (recommended for reduced-suite and batch runs).
  When it is unset, ``dkx`` selects a writable default under ``~/.cache``
  (or ``XDG_CACHE_HOME``).
- ``DKX_COMPILATION_CACHE_DIR``: override for that default cache path when
  ``JAX_COMPILATION_CACHE_DIR`` is unset.
- ``DKX_DISABLE_COMPILATION_CACHE``: set to ``1``/``true`` to skip enabling
  the persistent compilation cache entirely.
- ``DKX_DATA_DIR``: override the cache root for optional release-hosted
  equilibrium data files (default: ``~/.cache/dkx/data``, honoring
  ``XDG_CACHE_HOME``).
- ``DKX_OFFLINE``: set to ``1``/``true`` to forbid network fetches of
  external equilibrium data; an uncached fixture raises instead of downloading.
- ``DKX_EQUILIBRIA_DIRS``: OS-pathsep-separated search directories for
  resolving relative or relocated equilibrium file paths referenced by an input
  deck.
- ``DKX_UPSTREAM_UTILS_DIR``: directory holding the upstream v3 ``utils/``
  plotting scripts, used by ``postprocess-upstream`` and ``scan-er`` when the
  scripts are not found in a repo checkout.

CPU parallelism and host devices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set these **before** JAX is imported (i.e. before running
``python -m dkx``). The CLI ``--cores`` flag sets them for you.

- ``DKX_CORES``: high-level CPU parallelism knob. When set to ``N`` > 1,
  ``dkx`` enables process-parallel ``whichRHS`` solves **and** exposes
  ``N`` host devices for optional sharded matvecs. Set ``DKX_SHARD=0`` to
  keep process parallelism while disabling sharded matvecs. If neither
  ``--cores`` nor ``DKX_CORES`` is set, CLI auto mode uses ``1`` core for
  RHSMode=1 solves and up to ``3`` cores for RHSMode=2/3 transport runs.
- ``DKX_CPU_DEVICES``: request multiple host CPU devices for JAX SPMD
  sharded-JIT execution (sets ``--xla_force_host_platform_device_count``).
- ``DKX_XLA_THREADS``: opt in to setting the XLA CPU thread count from
  ``DKX_CORES``. Some JAX builds do not recognize
  ``--xla_cpu_parallelism_threads``, so this is disabled by default.

Single-solve sharding
~~~~~~~~~~~~~~~~~~~~~~~

- ``DKX_MATVEC_SHARD_AXIS``: shard the matvec along ``theta``, ``zeta``,
  ``x``, ``flat``, or ``auto`` when multiple devices are available (``off``
  disables it). ``auto`` chooses the larger of ``Ntheta``/``Nzeta``; ``flat``
  shards the full state vector evenly across devices.
- ``DKX_AUTO_SHARD``: set to ``0`` to disable auto sharding.
- ``DKX_SHARD``: shorthand to disable auto sharding even when
  ``DKX_CORES`` is set. Use ``0``/``false`` to keep single-device matvecs.
- ``DKX_SHARD_PAD``: pad odd ``Ntheta``/``Nzeta`` (and ``Nx`` when
  x-sharding is requested) internally so sharding can use even device counts
  (default: enabled). Padding adds ghost planes with zero weights and does not
  change outputs.
- ``DKX_GMRES_DISTRIBUTED``: set to ``1`` to run the Krylov solver under
  explicit ``jax.jit`` sharding (vectors kept sharded across devices) when using
  ``flat`` sharding. Default: off (single-device GMRES).
- ``DKX_DISTRIBUTED_KRYLOV``: distributed Krylov preference for
  ``solve_method=auto`` under sharded solves. ``auto`` (default) selects
  communication-reduced BiCGStab; ``gmres`` forces distributed GMRES.

Multi-host distributed initialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``DKX_DISTRIBUTED``: enable JAX multi-host initialization (default:
  off). When set, also provide:

  - ``DKX_PROCESS_ID``: this process rank (0-based).
  - ``DKX_PROCESS_COUNT``: total number of processes.
  - ``DKX_COORDINATOR_ADDRESS``: host (or ``host:port``) of the coordinator.
  - ``DKX_COORDINATOR_PORT``: coordinator port (default: ``1234``).

The CLI flags ``--distributed``, ``--process-id``, ``--process-count``,
``--coordinator-address``, and ``--coordinator-port`` set these for you.

Transport (RHSMode=2/3) worker parallelism
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``DKX_TRANSPORT_PARALLEL``: parallelize RHSMode=2/3 ``whichRHS`` solves
  across worker processes (``off``/``process``/``auto``).
- ``DKX_TRANSPORT_PARALLEL_WORKERS``: number of worker processes for
  parallel transport solves. The CLI ``--transport-workers`` flag sets both.

includePhi1 Newton–Krylov tolerances
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These are numerical overrides for the ``includePhi1 = .true.`` Newton–Krylov
solve. They have no namelist/API equivalent and are overrides for parity or
debugging, not solver selectors.

- ``DKX_PHI1_NEWTON_TOL``: absolute nonlinear (Newton) tolerance for
  includePhi1 solves (default: ``1e-12``). It governs how many Newton iterates
  are accepted and how many entries the ``NIterations`` output axis stores.
- ``DKX_PHI1_GMRES_TOL``: inner GMRES tolerance for the includePhi1
  Newton–Krylov step (default: ``1e-12``).

Parity and physics overrides
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``DKX_ROSENBLUTH_METHOD``: how the Rosenbluth potential response
  matrices are computed for ``collisionOperator=0`` with ``xGridScheme=5/6``.

  - ``quadpack`` (default): match the Fortran v3 QUADPACK-based implementation
    for parity.
  - ``analytic``: faster analytic integrals (may differ at strict parity level).

- ``DKX_FP_STRICT_PARITY``: for ``collisionOperator=0`` multispecies runs,
  force a scalar-ordered accumulation of the FP cross-species coupling to match
  v3 ordering.

  - Default: enabled automatically for RHSMode=1 multispecies cases.
  - ``0``/``false``: disable (use the faster vectorized accumulation).

Diagnostics, profiling, and stdout
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``DKX_FORTRAN_STDOUT``: control strict Fortran-style stdout mirroring
  (``1`` to mirror, ``0`` to silence). The CLI
  ``--fortran-stdout``/``--no-fortran-stdout`` flags set it.
- ``DKX_WRITE_SOLVER_DIAGNOSTICS``: set to ``1`` to add per-``whichRHS``
  residual datasets (``transportResidualNorms``, ``transportRhsNorms``,
  ``transportRelativeResidualNorms``, ``transportMaxResidualNorm``, and
  ``transportMaxRelativeResidualNorm``) to transport H5 output. Publication scan
  scripts use these fields to reject unconverged high-``nu'`` outputs.
- ``DKX_PROFILE``: enable phase-level timing and memory sampling for the
  solve (``1``/``timings``/``full``/``trace`` opt in). It emits ``profiling: ...``
  lines and trace metadata.
- ``DKX_PROFILE_DEVICE_MEM``: also sample device (accelerator) memory in
  the profiler; ``DKX_PROFILE=full``/``device`` implies it.
- ``DKX_DEBUG``: set to ``1``/``true`` to re-raise full tracebacks from
  CLI subcommands instead of printing a short error message.

Internal and test-only variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These are not production tuning knobs and are listed only for completeness:

- ``DKX_CI``: recognized alongside the standard ``CI`` variable to
  suppress the CLI's automatic core selection in continuous-integration runs.
- ``DKX_CLI_BOOTSTRAPPED``: set automatically when the CLI re-executes to
  apply pre-JAX environment defaults; not intended to be set by hand.
- ``DKX_VMEX_WOUT``: points an optional VMEC/Boozer geometry-backend
  integration test at a ``wout`` fixture; unused in normal runs.

Writing output files with `dkx`
--------------------------------------

.. code-block:: bash

   # Default CLI mode (matches Fortran v3 behavior)
   dkx /path/to/input.namelist

   # If --cores is omitted and DKX_CORES is unset, dkx auto-selects
   # 1 core for RHSMode=1 and up to 3 cores for RHSMode=2/3 on non-CI machines.

.. code-block:: bash

   # Parallel CPU run without environment variables
   dkx --cores 4 /path/to/input.namelist

.. code-block:: bash

   dkx write-output --input /path/to/input.namelist --out sfincsOutput.h5
   dkx write-output --input /path/to/input.namelist --out sfincsOutput.nc
   dkx write-output --input /path/to/input.namelist --out sfincsOutput.npz

The suffix selects the writer: ``.h5``/``.hdf5`` for Fortran-compatible HDF5,
``.nc``/``.netcdf`` for NetCDF4, and ``.npz`` for a fast uncompressed NumPy
archive. The solve and diagnostics are identical across these output formats.
Add ``--solver-trace solver_trace.json`` when you want a reproducible JSON
sidecar with backend, selected solve lane, elapsed time, device count, and output
metadata for profiling or regression reports. RHSMode=1 output files also carry
the core convergence fields directly: ``linearSolverMethod``,
``linearSolverResidualNorm``, ``linearSolverResidualTarget``,
``linearSolverResidualTargetRatio``, ``linearSolverConverged``,
``linearSolverAccepted``, and ``linearSolverAcceptanceCriterion``. In
PETSc-compatible constrained-PAS minimum-norm runs, ``linearSolverConverged``
can be false while ``linearSolverAccepted`` is true; that distinction is
intentional and prevents true-residual convergence from being conflated with
Fortran/PETSc-compatible branch acceptance. Sparse-PC runs additionally expose
``linearSolverMatvecs``, setup/solve/elapsed timings, sparse-pattern build time,
sparse preconditioner factorization time, and sparse-pattern nonzero counters in
the same output file.

The solver trace is intentionally separate from the physics output so parity
comparisons against SFINCS Fortran v3 can continue to use byte-stable HDF5,
NetCDF, or NPZ payloads. The sidecar uses a versioned schema and records the
minimum information needed to audit automatic path choices:

.. code-block:: json

   {
     "schema_version": 1,
     "backend": "gpu",
     "rhs_mode": 1,
     "selected_path": "rhsmode1_solution",
     "solve_method": "auto",
     "geometry_scheme": 4,
     "device_count": 1,
     "elapsed_s": 7.088,
     "metadata": {
       "output_format": "h5",
       "compute_solution": true,
       "compute_transport_matrix": false
     }
   }

Use the trace when debugging runtime cliffs: compare ``selected_path``,
``backend``, ``elapsed_s``, and ``device_count`` before changing solver
environment variables or forcing a preconditioner.

.. code-block:: bash

   dkx write-output \
     --input /path/to/input.namelist \
     --out sfincsOutput.h5 \
     --equilibrium-file /path/to/equilibrium.bc

.. code-block:: bash

   dkx /path/to/input.namelist --wout-path /path/to/wout.nc --out sfincsOutput.h5

.. code-block:: bash

   python -m dkx write-output --input /path/to/input.namelist --out sfincsOutput.h5

.. code-block:: python

   from pathlib import Path
   from dkx.api import write_output

   write_output(Path("input.namelist"), Path("sfincsOutput.h5"))

.. code-block:: python

   write_output(
       Path("input.namelist"),
       Path("sfincsOutput.h5"),
       solver_trace_path=Path("solver_trace.json"),
   )

.. code-block:: python

   write_output(Path("input.namelist"), Path("sfincsOutput.nc"))

   write_output(Path("input.namelist"), Path("sfincsOutput.npz"))

.. code-block:: python

   write_output(
       Path("input.namelist"),
       Path("sfincsOutput.h5"),
       wout_path=Path("/path/to/wout.nc"),
   )

The CLI ``write-output`` command uses ``solve_method="auto"`` by default. That
is the recommended production path. ``write_output`` routes the deck through
the canonical RHSMode dispatch (:func:`dkx.run.run_from_namelist`);
for end-to-end differentiation use the pure APIs
(:func:`dkx.solve.solve` with ``differentiable=True`` or
:func:`dkx.er.ambipolar_er`).

Inspect results immediately:

.. code-block:: python

   from dkx.io import read_sfincs_h5

   out_path = write_output(Path("input.namelist"), Path("sfincsOutput.h5"))
   results = read_sfincs_h5(out_path)
   print(out_path)
   print(results["Ntheta"])

When an equilibrium override is used, the embedded ``input.namelist`` dataset or
variable in the output file is patched to reflect the effective file path so downstream
diagnostics and bug reports see the actual run configuration.

Console output is silent by default from Python; pass ``emit=print`` to see the
Fortran-parity console flow.

For transport-matrix runs (``RHSMode=2`` or ``RHSMode=3``) the deck's RHSMode
selects the ``whichRHS`` loop automatically and ``transportMatrix`` is written:

.. code-block:: python

   write_output(Path("input.namelist"), Path("sfincsOutput.h5"))

Running an ``Er`` scan (transport-matrix mode)
----------------------------------------------

To generate a scan directory compatible with upstream plotting scripts like ``sfincsScanPlot_2``,
you can use the ``scan-er`` subcommand:

.. code-block:: bash

   dkx scan-er \
     --input /path/to/input.namelist \
     --out-dir /path/to/scan_dir \
     --min -0.1 --max 0.1 --n 5 \
     --compute-transport-matrix

This creates subdirectories like ``Er0.1/``, each containing ``input.namelist``,
``sfincsOutput.h5``, and ``sfincsOutput.solver_trace.json``.  The scan directory
also gets a scan-style ``input.namelist`` with ``!ss`` directives so the upstream
scan plotting scripts can infer the directory list.  The JSON sidecar is the
auditable record for each point: it stores backend, selected solver lane, active
size, elapsed time, residual target, residual norm, and memory estimates without
changing the Fortran-compatible physics output.

For large scans, you can parallelize scan points:

.. code-block:: bash

   dkx scan-er \
     --input /path/to/input.namelist \
     --out-dir /path/to/scan_dir \
     --min -0.1 --max 0.1 --n 41 \
     --jobs 8

For job arrays, slice the scan values with ``--index`` and ``--stride``:

.. code-block:: bash

   dkx scan-er \
     --input /path/to/input.namelist \
     --out-dir /path/to/scan_dir \
     --min -0.1 --max 0.1 --n 401 \
     --index ${SLURM_ARRAY_TASK_ID} \
     --stride 64

Solving the ambipolar root directly
-----------------------------------

For ``RHSMode=1`` inputs, the ``ambipolar`` command evaluates radial-current
outputs in process and applies a bracketed Brent solve:

.. code-block:: bash

   dkx ambipolar \
     --input /path/to/input.namelist \
     --out-dir /path/to/ambipolar_run \
     --er-min -0.1 --er-max 0.1 --er-initial 0.0

The command routes through the canonical :mod:`dkx.er` slice
(:func:`dkx.er.find_ambipolar_er`) and writes ``ambipolar_result.json``
with the converged flag, the selected root ``root_er``, its ``root_type``
(ion / electron / unstable), the ordered radial-current ``iterations``, and
every classified root in the bracket.  Warm starts and GCROT recycling are
threaded across the :math:`E_r` evaluations internally.

Python workflows can call the same canonical slice directly, and — unlike the
CLI — also obtain a *differentiable* ambipolar :math:`E_r`:

.. code-block:: python

   from dkx import er

   # Fortran-parity Brent root (bracket expansion + classification):
   result = er.find_ambipolar_er(
       "input.namelist", er_bracket=(-0.1, 0.1), er_initial=0.0,
   )
   print(result.er, result.root_type, [r.root_type for r in result.roots])

   # Differentiable ambipolar Er: jax.grad flows through the root via the
   # implicit function theorem (solvax.implicit.root_solve), with dJr/dEr and
   # dJr/dp taken from autodiff of er.radial_current (not finite differences).
   import jax
   root = er.prepare("input.namelist")
   er_star = er.ambipolar_er(root.operator, er0=result.er,
                             dphi_per_er=root.dphi_per_er, z_s=root.z_s)

Running upstream postprocessing scripts (utils/)
------------------------------------------------

The upstream Fortran v3 codebase ships a set of plotting scripts under `utils/`.
This repository vendors those scripts in `examples/sfincs_examples/utils/`.

If you have a directory containing `sfincsOutput.h5`, you can run one of these scripts non-interactively:

.. code-block:: bash

   dkx postprocess-upstream --case-dir /path/to/case --util sfincsScanPlot_1 -- pdf

For example, after running ``scan-er`` you can generate a PDF using the upstream script:

.. code-block:: bash

   dkx postprocess-upstream --case-dir /path/to/scan_dir --util sfincsScanPlot_2 -- pdf
