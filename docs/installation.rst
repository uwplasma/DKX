Installation
============

Standard install
----------------

.. code-block:: bash

   pip install sfincs_jax

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
