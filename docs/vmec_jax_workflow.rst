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

The status JSON also contains ``no_solve_provenance_gate``.  This gate is a
machine-readable assertion that the workflow is still a proxy-gradient lane:
``kinetic_solve_executed`` is false, the differentiated object is the
Boozer-spectrum transport-like scalar, and full VMEC-boundary-to-SFINCS kinetic
gradients remain deferred.  Both workflow examples use the same shared
``sfincs_jax.geometry.jax_adapters`` gate, so the skip-safe status path and the
file-backed proxy-gradient summary enforce the same scalar-contract boundary.

The same payload carries ``kinetic_transport_scalar_contract`` and a
copy of its gate in
``no_solve_provenance_gate.kinetic_transport_scalar_contract_gate``. This is the
forward contract for a future VMEC/Boozer-to-kinetic-transport scalar.  It lists
``required_kinetic_transport_scalar_stages`` in machine-readable form:

- ``vmec_source``,
- ``vmec_equilibrium_or_wout``,
- ``boozer_transform``,
- ``sfincs_geometry_adapter``,
- ``kinetic_operator_assembly``,
- ``linear_kinetic_solve``,
- ``transport_scalar_reduction``,
- ``gradient_validation``.

Each stage records its current public role, differentiability boundary, current
status, and evidence required before the stage can support a full kinetic
transport scalar.  The current public scalar is explicitly
``boozer_spectrum_proxy_not_kinetic``; ``kinetic_transport_scalar_claimed`` and
``kinetic_solve_executed`` must both remain false in this lane.

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

Run the documented file-backed workflow with an explicit ``wout`` file:

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
- ``sfincs_jax.geometry.jax_adapters.boozer_spectrum_proxy_transport_objective``.

Setup or provenance only, not differentiated:

- VMEC file I/O,
- ``vmec_jax`` fixed-boundary setup used to produce a ``wout``-like object,
- ``sfincs_jax`` VMEC file adapters and scheme-5 parity readers.

Explicit non-claims:

- no full VMEC-boundary-to-SFINCS kinetic transport gradients,
- no gradient through the SFINCS kinetic transport solve in this lane,
- no production solver dependency on ``vmec_jax`` or ``booz_xform_jax``.

Future kinetic scalar contract:

- ``kinetic_transport_scalar_contract.required_stages`` is the authoritative
  list of stages that must be present before a VMEC/Boozer-to-SFINCS kinetic
  scalar can be claimed.
- ``kinetic_transport_scalar_contract.current_public_scalar`` separates
  differentiated proxy stages from setup-only and not-covered stages.
- ``kinetic_transport_scalar_contract.no_overclaim_gate.status`` must be
  ``"pass"`` in default CI.  It fails if the proxy lane claims a kinetic solve,
  requires optional geometry packages in default CI, drops a required stage, or
  promotes the full kinetic scalar while deferred stages remain.

Gates
-----

Run the VMEC/Boozer workflow and adapter gates:

.. code-block:: bash

   python -m pytest tests/test_vmec_jax_workflow.py tests/test_jax_geometry_adapters.py -q

These gates also include a no-solve invariant check that the normalized Boozer
proxy transport objective is unchanged by global :math:`|B|` spectrum scaling
and is exactly zero, with zero gradient, for a constant-:math:`B` spectrum.

Run the optional JAX ecosystem gates that protect future Lineax, Equinox, and
opt-in JAXopt adoption:

.. code-block:: bash

   python -m pytest tests/test_optional_ecosystem_gates.py \
     tests/test_optional_lineax_implicit_gate.py \
     tests/test_optional_eqx_jaxopt_scheme4_gate.py -q

The optional ecosystem benchmark CLIs also write measured adoption summaries
without changing their row-list JSON outputs:

.. code-block:: bash

   python examples/performance/benchmark_optional_lineax_implicit_solve.py \
     --backend all \
     --suite all \
     --out-json lineax-rows.json \
     --summary-json lineax-summary.json

   python examples/optimization/benchmark_optional_eqx_jaxopt_scheme4_gate.py \
     --backend all \
     --out-json eqx-jaxopt-rows.json \
     --summary-json eqx-jaxopt-summary.json

Those summaries keep the production decision conservative: optional packages may
be candidates for bounded experiments only when the measured gate is clean; they
do not become hard dependencies or production solver defaults from these gates.
Default CI installs ``lineax`` and ``equinox`` and leaves the JAXopt row
skip-safe unless a developer installs JAXopt explicitly for local comparison.

The optional VMEC/Boozer integration tests use ``pytest.importorskip`` or a
skip-status payload. Missing optional packages therefore record a skipped lane,
not a failed default installation.

In a file-backed optional run, the written ``workflow-summary.json`` must also
show ``no_solve_provenance_gate.status == "pass"``.  For that path the gate
requires explicit provenance fields for the source ``wout`` or in-memory VMEC
object, selected surface, Boozer resolution, objective grid shape, and spectral
scale.  This lets downstream users audit what was differentiated without
mistaking the proxy scalar for a SFINCS kinetic transport solve.
The same JSON must show
``no_solve_provenance_gate.required_kinetic_transport_scalar_stages`` and
``kinetic_transport_scalar_contract.no_overclaim_gate.status == "pass"`` so the
future kinetic-scalar lane cannot silently lose required stages or boundaries.

Promotion rule
--------------

This lane is complete enough for documented research use when:

- ``--check-backends --json`` returns a valid workflow contract,
- ``backend_readiness_gate.status == "pass"`` in the no-optional-dependency
  preflight/backend-contract payload,
- ``no_solve_provenance_gate.status == "pass"`` and
  ``no_solve_provenance_gate.kinetic_solve_executed == false`` in both preflight
  and file-backed summary payloads,
- the proxy-gradient gate writes a summary JSON with
  ``numerical_gradient_gate.status == "pass"`` on at least one explicit ``wout``
  fixture when optional packages are installed,
- default tests pass without requiring optional geometry packages,
- docs continue to state the differentiability boundary and non-claims exactly.

The next blocker for a production optimization claim is a pure-JAX scheme-5
geometry path and a transport-objective gradient gate. Until that exists, keep
this page scoped to the Boozer-spectrum proxy workflow.
