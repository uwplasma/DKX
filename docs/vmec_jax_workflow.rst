VMEC JAX workflow
=================

This page is the concrete workflow contract for optional
``vmec_jax -> booz_xform_jax -> sfincs_jax`` coupling. It does not make either
geometry package a hard dependency. Default CI must still pass when both packages
are absent.

The current public lane is a Boozer-spectrum proxy transport-objective gradient
gate. It is not a claim of full VMEC-boundary-to-SFINCS kinetic transport
gradients.

Preflight
---------

Run the lightweight status scaffold first:

.. code-block:: bash

   python examples/optimization/vmec_jax_workflow_status.py --json

This reports shallow importability for ``vmec_jax`` and ``booz_xform_jax``, the
no-overclaim gate, a no-optional-dependency Boozer-spectrum autodiff readiness
gate, and the exact command for the optional proxy-gradient gate.  It does not
import either optional backend.

The existing end-to-end example has the same skip-safe backend contract:

.. code-block:: bash

   python examples/autodiff/vmec_jax_to_boozer_sfincs_pipeline.py --check-backends --json

In ``--check-backends`` mode, the JSON payload includes
``backend_readiness_gate``.  That gate is intentionally synthetic: it evaluates a
small Boozer spectrum through a ``sfincs_jax`` proxy transport objective, checks
the full spectral JAX gradient against centered finite differences, and checks a
JVP against the gradient dot product.  It is a local
backend-readiness/sensitivity check for the downstream differentiable objective,
not evidence that ``vmec_jax`` or ``booz_xform_jax`` executed.

To persist a provenance record without running optional geometry code:

.. code-block:: bash

   python examples/autodiff/vmec_jax_to_boozer_sfincs_pipeline.py \
     --check-backends \
     --summary-json workflow-summary.json

Optional install pattern
------------------------

Use editable installs for local research checkouts, or install equivalent
packages into the active environment:

.. code-block:: bash

   python -m pip install -e /path/to/vmec_jax
   python -m pip install -e /path/to/booz_xform_jax

The status scaffold should then report both optional backends as available.

Proxy-gradient gate
-------------------

Run the documented file-backed handoff with an explicit ``wout`` file:

.. code-block:: bash

   python examples/autodiff/vmec_jax_to_boozer_sfincs_pipeline.py \
     --wout /path/to/wout_circular_tokamak.nc \
     --mboz 3 \
     --nboz 3 \
     --surface 0.5 \
     --steps 0 \
     --summary-json workflow-summary.json

The same input can be supplied as:

.. code-block:: bash

   export SFINCS_JAX_VMEC_JAX_WOUT=/path/to/wout_circular_tokamak.nc
   python examples/autodiff/vmec_jax_to_boozer_sfincs_pipeline.py \
     --mboz 3 \
     --nboz 3 \
     --surface 0.5 \
     --steps 0

If ``vmec_jax`` example decks are available, the example can build the VMEC-like
object before the Boozer transform:

.. code-block:: bash

   python examples/autodiff/vmec_jax_to_boozer_sfincs_pipeline.py \
     --vmec-case circular_tokamak \
     --vmec-max-iter 1 \
     --steps 0

Differentiability contract
--------------------------

Differentiated in this lane:

- scaled VMEC-like magnetic spectral arrays,
- the ``booz_xform_jax`` transform,
- ``sfincs_jax.jax_geometry_adapters.boozer_spectrum_proxy_transport_objective``.

Setup or provenance only, not differentiated:

- VMEC file I/O,
- ``vmec_jax`` fixed-boundary setup used to produce a ``wout``-like object,
- ``sfincs_jax`` VMEC file adapters and scheme-5 parity readers.

Explicit non-claims:

- no full VMEC-boundary-to-SFINCS kinetic transport gradients,
- no gradient through the SFINCS kinetic transport solve in this lane,
- no production solver dependency on ``vmec_jax`` or ``booz_xform_jax``.

Gates
-----

Run the VMEC/Boozer workflow and adapter gates:

.. code-block:: bash

   python -m pytest tests/test_vmec_jax_workflow.py tests/test_jax_geometry_adapters.py -q

Run the optional JAX ecosystem gates that protect future Lineax, Equinox, and
JAXopt adoption:

.. code-block:: bash

   python -m pytest tests/test_optional_ecosystem_gates.py \
     tests/test_optional_lineax_implicit_gate.py \
     tests/test_optional_eqx_jaxopt_scheme4_gate.py -q

The optional ecosystem benchmark CLIs also write measured adoption summaries
without changing their row-list JSON outputs:

.. code-block:: bash

   python examples/performance/benchmark_optional_lineax_implicit_solve.py \
     --backend all \
     --suite synthetic \
     --out-json lineax-rows.json \
     --summary-json lineax-summary.json

   python examples/optimization/benchmark_optional_eqx_jaxopt_scheme4_gate.py \
     --backend all \
     --out-json eqx-jaxopt-rows.json \
     --summary-json eqx-jaxopt-summary.json

Those summaries keep the production decision conservative: optional packages may
be candidates for bounded experiments only when the measured gate is clean; they
do not become hard dependencies or production solver defaults from these gates.

The optional VMEC/Boozer integration tests use ``pytest.importorskip`` or a
skip-status payload. Missing optional packages therefore record a skipped lane,
not a failed default installation.

Promotion rule
--------------

This lane is complete enough for documented research use when:

- ``--check-backends --json`` returns a valid workflow contract,
- ``backend_readiness_gate.status == "pass"`` in the no-optional-dependency
  preflight/backend-contract payload,
- the proxy-gradient gate writes a summary JSON with
  ``numerical_gradient_gate.status == "pass"`` on at least one explicit ``wout``
  fixture when optional packages are installed,
- default tests pass without requiring optional geometry packages,
- docs continue to state the differentiability boundary and non-claims exactly.

The next blocker for a production optimization claim is a pure-JAX scheme-5
geometry path and a transport-objective gradient gate. Until that exists, keep
this page scoped to the Boozer-spectrum proxy workflow.
