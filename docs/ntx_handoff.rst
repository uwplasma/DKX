NTX RHSMode=1 Handoff
======================

This page records the SFINCS-JAX state that should be used by the NTX
finite-beta profile-current lane after the RHSMode=1 solver-policy audit.
It is intentionally operational: commands, expected solver metadata, and the
remaining physics caveat are listed in the same place.

Status on 2026-05-01
--------------------

The collaborator-reported runtime cliff for finite-beta RHSMode=1 constrained
PAS profile-current decks is closed for the validated non-differentiable CPU
lane.  Large constrained-PAS decks in the validated size window now route from
``solve_method=auto`` to sparse-preconditioned GMRES instead of the older
matrix-free/PAS fallback that could stall near an ``O(1e-2)`` true residual.

The validated NTX ``17 x 21 x 12, Nx=5`` deck now completes through the
default CLI path in about ``7 s`` on the local CPU, with peak RSS about
``1.6 GB`` and output metadata

.. code-block:: text

   linearSolverMethod = sparse_pc_gmres
   linearSolverConverged = 1
   linearSolverAccepted = 1
   linearSolverResidualNorm = 9.19e-16
   linearSolverResidualTarget = 1.09e-09
   FSABjHat = -1.2981550371185984

This replaces the previous default matrix-free path, which took hundreds of
seconds on the same deck and still stopped near a ``1.9e-2`` residual.

Recommended NTX command pattern
-------------------------------

Use the normal CLI path first.  Keep ``JAX_ENABLE_X64=True`` for production
finite-beta comparisons.

.. code-block:: bash

   JAX_ENABLE_X64=True sfincs_jax input.namelist \
     --output sfincsOutput.h5 \
     --solver-trace sfincsOutput.solver_trace.json

For long audit runs, use the profiling wrapper without a JAX Perfetto/XPlane
trace unless kernel-level profiling is specifically needed:

.. code-block:: bash

   JAX_ENABLE_X64=True python scripts/profile_write_output_trace.py \
     --input input.namelist \
     --out sfincsOutput.h5 \
     --compute-solution \
     --solver-trace \
     --no-jax-trace \
     --phase-log phase_log.jsonl

The phase log is written outside the JAX profiler context, so a solve that
successfully writes HDF5 diagnostics is not reported as failed only because a
profiler-finalization hook fails.

What Is Closed
--------------

- Output writes now include solver metadata fields such as
  ``linearSolverMethod``, ``linearSolverResidualNorm``,
  ``linearSolverResidualTarget``, ``linearSolverConverged``, and
  ``linearSolverAccepted``.
- The default large constrained-PAS RHSMode=1 CPU policy no longer chooses the
  slow residual-stalling path in the validated NTX size window.
- The sparse-PC path converges the same algebraic system to true residuals near
  roundoff on the ``17 x 21 x 12, Nx=5`` deck.
- The ``25 x 31 x 17, Nx=11`` production-resolution NTX deck completes locally
  with ``solve_method=sparse_pc_gmres`` in about ``121 s``, peak RSS about
  ``9.3 GB``, and residual ``4.31e-14`` against a ``2.09e-09`` target.
- Fortran operator-dump audits show SFINCS-JAX applies the same sparse
  RHSMode=1 operator as the clean Fortran v3 PETSc matrix branch to roundoff on
  the audited finite-beta decks.

What Remains Open
-----------------

The finite-beta RHSMode=1 bootstrap-current publication claim is still a
physics-reference audit, not a stalled-solver issue.  The converged
SFINCS-JAX sparse-PC branch matches the clean PETSc sparse-matrix branch, but
some MUMPS/SuperLU_DIST Fortran artifacts and Redl/NTX+NEOPAX comparisons use
different nullspace/reference branches for the profile-current observable.

Before promoting this lane as a publication parity result, NTX and SFINCS-JAX
should pin the physical branch by rerunning the radial/profile ladder with:

- the same finite-beta VMEC geometry,
- the same species/profile/gradient normalization contract,
- the same RHSMode=1 current observable,
- a converged true residual gate in the output metadata,
- and an explicit comparison to Redl and NTX+NEOPAX current profiles.

The GPU-native RHSMode=1 path is also not promoted by this handoff.  The
validated fix is the non-differentiable CPU sparse-PC production lane.  GPU
kernel-level work should be treated as a separate profiling and robustness
campaign.
