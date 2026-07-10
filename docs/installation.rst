Installation
============

Standard install
----------------

.. code-block:: bash

   pip install sfincs_jax

Structured solver extra (``solvax``)
------------------------------------

The structured linear-algebra tiers (block-tridiagonal Legendre elimination,
recycled GCROT Krylov, implicit differentiation) live in the external
`solvax <https://github.com/uwplasma/SOLVAX>`_ library. ``sfincs_jax`` imports
it lazily; every module stays importable without it, and the ``auto`` solve
policy falls back to host/direct paths when it is absent. Until the ``solvax``
PyPI release, install it from git (afterwards ``pip install "sfincs_jax[structured]"``
resolves it directly):

.. code-block:: bash

   pip install git+https://github.com/uwplasma/SOLVAX

GPU
---

Install the CUDA build of JAX that matches your driver, for example:

.. code-block:: bash

   pip install -U "jax[cuda12]"

No ``sfincs_jax`` change is needed; the same solves run on the accelerator.

SFINCS Fortran v3 reference build (optional)
--------------------------------------------

Parity and benchmark tooling can compare against a local SFINCS Fortran v3
executable. A reproducible route on macOS/Linux is a conda environment that
provides PETSc and MUMPS (the measured baselines in :doc:`performance` use
conda PETSc 3.23 + MUMPS 5.8.2) together with upstream's
``makefiles/makefile.conda`` in the SFINCS repository
(``fortran/version3``). Modern PETSc ``mpi_f08`` typing needs two local,
version-guarded ``MPIU_Comm`` declaration patches in ``globalVariables.F90``
and ``sfincs_main.F90``; the resulting binary passes upstream's own example
checks to about ``4e-5`` relative. None of this is required to use
``sfincs_jax`` itself — frozen reference outputs ship with the test fixtures.

Release-hosted equilibrium fixtures
-----------------------------------

The package intentionally does not store multi-megabyte public VMEC/Boozer
fixtures in the git history or wheel. Examples and compatibility tests that need
the W7-X, HSX, or QI equilibrium files resolve them by basename and fetch the
``sfincs-jax-data-v1`` GitHub release asset into a user cache on first use.

The default cache root is ``~/.cache/sfincs_jax/data``. To prefetch the release
data explicitly, run:

.. code-block:: bash

   python -m sfincs_jax.validation.data_fetch

Use ``SFINCS_JAX_DATA_DIR=/path/to/cache`` to choose a different cache root. Use
``SFINCS_JAX_OFFLINE=1`` in CI or cluster jobs when a run must fail instead of
downloading missing data.

Editable install (recommended for development)
-----------------------------------------------

.. code-block:: bash

   pip install -e ".[dev]"

Documentation tooling
---------------------

.. code-block:: bash

   pip install -e ".[docs]"

Additional example-only packages
--------------------------------

The core install includes ``matplotlib`` and ``netCDF4``, so plotting examples,
``sfincs_jax --plot``, and ``--out sfincsOutput.nc`` work without any extra plotting
or file-format dependency group.

Some optimization examples use ``optax`` directly. Install it explicitly when
you want those examples:

.. code-block:: bash

   pip install optax

Optional solver-library adoption studies, including Lineax, Equinox-wrapper, and
JAXopt comparisons, are research-lane material. They are not required for the
stable install, stable examples, or default CI.
