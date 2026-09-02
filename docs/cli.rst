Command-line reference
======================

``dkx`` has eleven commands. Every SFINCS-specific operation lives under
``dkx sfincs`` instead of at the top level, so ``dkx --help`` stays short.

.. code-block:: text

   dkx doctor              check that this install can run
   dkx schema              print the case template or JSON Schema
   dkx validate CASE       validate a case file and print its ID
   dkx run CASE            solve one case
   dkx scan CASE           solve every point of a [scan]
   dkx converge CASE       refine each axis and report convergence
   dkx roots RESULT        print the ambipolar root table
   dkx inspect RESULT      print what a result contains
   dkx plot RESULT         write a figure
   dkx compare A B         compare two results
   dkx convert DECK CASE   turn a SFINCS namelist into a case file
   dkx sfincs ...          SFINCS compatibility commands

Every command takes ``-q`` to suppress progress and ``--cores`` to bound the
CPU threadpool.

Starting out
------------

``dkx doctor`` first, when anything behaves strangely. It reports what this
process *observes*, not what it was asked for: the float64 row allocates an
array and reads its dtype rather than trusting ``JAX_ENABLE_X64``, which an
already-initialised backend ignores. That is the failure worth catching --
float64 silently off produces plausible numbers that are wrong.

.. code-block:: bash

   dkx doctor                  # table
   dkx doctor --format json    # machine-readable

It exits non-zero when a check fails, so CI reading only the return code
reaches the same verdict as a person reading the table.

``dkx schema`` prints a commented case template to start from:

.. code-block:: bash

   dkx schema --format toml > case.toml
   dkx schema --format json          # JSON Schema, for editors and validators

Running a case
--------------

``dkx run CASE --out RESULT.nc`` solves one case and writes a ``Result``.
``dkx validate CASE`` checks the file and prints its deterministic ``case_id``
without solving, which is the cheap way to find a problem before a long run.

``validate`` runs the executor's own preflight, not just the schema. The case
schema is deliberately wider than what the solver implements -- it admits
``magnetic_drifts = "full"``, ``workflow = "monoenergetic"`` and
``phi1 = "kinetic"``, none of which execution supports yet -- so a case can
pass the JSON Schema and still be unrunnable. ``validate`` catches that.

``dkx scan CASE`` expands a ``[scan]`` table and solves every point, writing a
single result with a leading ``case`` dimension and the axis values stored
beside the observables. Two behaviours matter:

* A failing point does not discard the points that already succeeded. The
  failure is recorded in that row, the run continues, and the command exits
  ``1`` with the output still written.
* ``resume`` skips points whose ``case_id`` is already in the output. It keys
  on the deterministic id rather than on position, so a scan resumed after the
  case file was edited reruns instead of grafting new physics onto old rows.

``dkx converge CASE`` refines ``theta``, ``zeta``, ``pitch`` and ``speed`` and
reports what the *observables* did, exiting non-zero when any axis is still
moving. It refines the axes jointly as well as one at a time, because one at a
time can mislead: on the shipped analytic tokamak deck the theta axis looks
settled to 0.2% at ``pitch = 8`` and moves the outputs 74% at ``pitch = 40``.

Reading results
---------------

``dkx inspect RESULT`` lists what a result holds.

``dkx plot RESULT`` picks a panel from what the result actually contains: a
radial-profile panel for a profile run, plus an ``E_r`` root panel when the run
was ambipolar, and observables against the axis for a result written by ``dkx
scan``. ``--kind`` forces one instead:

.. code-block:: bash

   dkx plot result.nc                    # auto
   dkx plot result.nc --kind search      # J_r against E_r: every evaluation,
                                         # bracket, root type, selected branch
   dkx plot scan.nc   --kind scan        # observables against the scan axis

The ``search`` panel is the one to look at before believing a root. It draws
every field the search evaluated, so the interval the result is scoped to is
visible rather than implied. An evaluation that failed is marked on the axis
rather than plotted at ``J_r = 0``, which would draw a crossing that never
happened; a scan point that failed is marked rather than interpolated over.

``dkx roots RESULT`` prints the ambipolar roots with their classification,
bracket width and branch, and flags any surface carrying a nonsmooth branch
event -- a root appearing or vanishing makes the output non-differentiable
there, and ``jax.grad`` will still return a number.

A run that admitted no roots says why that is not proof of none existing: sign
sampling cannot see a tangential root, or an even number of crossings between
two samples.

``dkx compare A B`` exits non-zero when two results differ. It dispatches on
the file extension rather than sniffing, because NetCDF4 *is* HDF5 and an
``h5py`` open succeeds on both. Wall-clock time and iteration counts are
reported but do not decide the verdict -- two runs of one case always differ in
timing, and counting it would make the exit status meaningless.

From an equilibrium
-------------------

``dkx wout_XXX.nc`` is a survey, not a study: about a minute, and panels out.
It has to invent a plasma to do it, because a VMEC equilibrium fixes the
*pressure* and nothing else.

The on-axis density is scaled from that pressure against the published reactor
profiles of Landreman, Buller and Drevlak (arXiv:2205.02914), and the
temperature then follows from ``p = 2nT``. The run prints the pair it used.

.. code-block:: bash

   dkx wout_XXX.nc                       # scaled from p(0)
   dkx wout_XXX.nc --density-m3 2.38e20  # pin n(0); T(0) follows
   dkx wout_XXX.nc --quick               # coarser, for a first look

The bootstrap current is sensitive to this. At fixed pressure it moves by a
factor of five across a plausible density range, because the collisionality
that suppresses it goes like ``n/T^2``. A bootstrap current computed from a
pressure profile alone cannot be compared against one from an optimizer that
assumed a different plasma -- pin the density to that design point first.

The electric-field bracket is scaled to the same plasma, since the ambipolar
field goes roughly like ``T/(e L)``: a bracket sized for a 2 keV plasma sits
entirely inside the ion root of a 9 keV one.

For real work, write a case file with your own profiles and resolution, then
``dkx validate`` it and ``dkx converge`` it before trusting a number.

Coming from SFINCS
------------------

``dkx convert deck.namelist case.toml`` turns a SFINCS input namelist into a
native case file. A deck states one surface with prescribed gradients while a
case states a profile, so the conversion emits three surfaces carrying a
profile linear in ``rHat`` -- exact for the ``np.gradient`` the executor uses.

Conversion refuses rather than approximating. A deck using a model the native
route does not implement fails at convert time, naming the key and what would
differ. Most checked-in SFINCS decks refuse; the common reasons are
analytic scheme-1 parameters, ``VMECRadialOption``, and ``RHSMode`` 2 or 3.

The compatibility commands read and write SFINCS files directly, with no
conversion:

.. code-block:: bash

   dkx sfincs --help                  # the full list
   dkx input.namelist --out out.h5    # the implicit form still works

The old top-level spellings (``dkx write-output``, ``dkx compare-h5`` and the
rest) still run. They are hidden from ``dkx --help`` rather than removed,
because they appear in existing scripts.

Exit status
-----------

``0`` success. ``1`` the command ran and the answer is "no": a scan point
failed, a case is not converged, ``doctor`` found a blocking problem, two
results differ. ``2`` the command could not run at all: a missing file, an
invalid case, an unsupported model.
