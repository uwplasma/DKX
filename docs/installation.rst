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

The core install now includes ``matplotlib``, so plotting examples and
``sfincs_jax --plot`` work without any extra plotting dependency group.

Some optimization examples use ``optax`` directly. Install it explicitly when
you want those examples:

.. code-block:: bash

   pip install optax
