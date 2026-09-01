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
without solving, which is the cheap way to find a typo before a long run.

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

``dkx inspect RESULT`` lists what a result holds. ``dkx plot RESULT`` writes a
radial-profile panel, plus an ``E_r`` root panel when the result came from the
ambipolar workflow.

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

Coming from SFINCS
------------------

``dkx convert deck.namelist case.toml`` turns a SFINCS input namelist into a
native case file. A deck states one surface with prescribed gradients while a
case states a profile, so the conversion emits three surfaces carrying a
profile linear in ``rHat`` -- exact for the ``np.gradient`` the executor uses.

Conversion refuses rather than approximating. A deck using a model the native
route does not implement fails at convert time, naming the key and what would
differ. Most checked-in SFINCS decks currently refuse; the common reasons are
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
