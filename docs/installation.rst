Installation
============

Standard install
----------------

.. code-block:: bash

   pip install sfincs_jax

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

The core install now includes ``matplotlib`` and ``netCDF4``, so plotting examples,
``sfincs_jax --plot``, and ``--out sfincsOutput.nc`` work without any extra plotting
or file-format dependency group.

Some optimization examples use ``optax`` directly. Install it explicitly when
you want those examples:

.. code-block:: bash

   pip install optax

The optional ecosystem benchmark gates also use extra packages when you want to run
them locally:

.. code-block:: bash

   pip install lineax equinox jaxopt
