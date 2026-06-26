Geometry models and loading
===========================

`sfincs_jax` solves radially local neoclassical kinetic problems on a single flux
surface. Geometry is therefore not an incidental input: it sets the coefficients of
the streaming term, mirror force, magnetic drifts, :math:`E\times B` drifts, the
Jacobian, the flux-surface averages, and the metric factors that enter diagnostics.

This page summarizes the supported geometry families, the mathematical objects loaded
from each source, and where those quantities are turned into discrete operators in the
source tree.

Supported geometry families
---------------------------

The public `sfincs_jax` workflows support:

- ``geometryScheme=1``: analytic three-helicity straight-field-line tokamak / toroidal
  model.
- ``geometryScheme=2``: analytic LHD-like reduced model.
- ``geometryScheme=4``: analytic W7-X-like reduced model.
- ``geometryScheme=5``: VMEC ``wout`` equilibrium files, in ASCII or netCDF form.
- ``geometryScheme=11``: Boozer ``.bc`` files with stellarator symmetry.
- ``geometryScheme=12``: Boozer ``.bc`` files without stellarator symmetry.

The common goal across all schemes is to provide the normalized fields

.. math::

   \hat B(\theta,\zeta), \qquad
   \hat D(\theta,\zeta), \qquad
   \hat B_\theta, \qquad
   \hat B_\zeta, \qquad
   \hat B_\psi,

their derivatives, and the metric/Jacobian information needed to evaluate
streaming, drift, and moment integrals on a flux surface.

Analytic geometry models
------------------------

The analytic tokamak-like model used in ``geometryScheme=1`` starts from a finite
Fourier representation of the field strength,

.. math::

   \frac{B(\theta,\zeta)}{\bar B}
   =
   \frac{B_0}{\bar B}
   \left[
     1
     + \epsilon_t \cos\theta
     + \epsilon_h \cos(\ell\theta - n\zeta)
     + \epsilon_a \sin(\ell_a\theta - n_a\zeta)
   \right].

This family is useful for:

- controlled tokamak studies,
- monoenergetic regression cases,
- sensitivity studies where a small parameter set is preferable to an equilibrium
  reconstruction,
- and reduced model benchmarking.

The reduced LHD- and W7-X-like schemes (``geometryScheme=2`` and ``4``) use fixed
analytic coefficient sets chosen to reproduce representative magnetic spectra and
metrics without loading an external equilibrium file.

The W7-X-like scheme also exposes a small differentiable harmonic-amplitude hook in
Python:

.. code-block:: python

   import jax
   import jax.numpy as jnp

   from sfincs_jax.geometry import boozer_geometry_scheme4

   theta = jnp.linspace(0.0, 2.0 * jnp.pi, 16, endpoint=False)
   zeta = jnp.linspace(0.0, 2.0 * jnp.pi / 5.0, 12, endpoint=False)
   amp0 = jnp.asarray([0.04645, -0.04351, -0.01902])

   def scalar_objective(a):
       geom = boozer_geometry_scheme4(theta=theta, zeta=zeta, harmonics_amp0=a)
       return jnp.mean(geom.b_hat**2) + 0.1 * jnp.mean(geom.d_hat)

   gradient = jax.grad(scalar_objective)(amp0)

This is not a replacement for VMEC or Boozer-file geometry. It is a deliberately
small end-to-end JAX gate used for optimization and testing: a scalar depending on
the normalized magnetic field and Jacobian-like coefficient is differentiated with
respect to magnetic-spectrum amplitudes and compared against finite differences in
CI.

VMEC workflow
-------------

For ``geometryScheme=5``, `sfincs_jax` reads a VMEC equilibrium and constructs the
single-surface geometric data needed by the kinetic solve. In practice the workflow is:

1. choose a target radius,
2. load equilibrium data from a ``wout`` file,
3. interpolate or evaluate the flux-surface quantities on the requested
   :math:`(\theta,\zeta)` grid,
4. convert the resulting fields to the normalized SFINCS-style arrays used by the
   operator kernels.

The radial coordinate can be requested in any of the supported forms
(:math:`\psi`, :math:`\psi_N`, :math:`\hat r`, :math:`r_N`), but the solve itself is
always local to one surface.

VMEC-centered user workflows typically use either:

- the namelist ``equilibriumFile`` entry, or
- the explicit Python / CLI override ``wout_path=...`` / ``--wout-path ...``.

Large public VMEC fixtures such as ``wout_w7x_standardConfig.nc`` are
release-hosted rather than tracked in the repository. If a namelist or example
references one of these known basenames, `sfincs_jax` resolves it through the
same path search described in :doc:`inputs` and downloads the checked release
asset into the user cache when needed.

Optional JAX-native geometry producers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The standard release path for ``geometryScheme=5`` remains a VMEC ``wout`` file.
For differentiable research workflows, `sfincs_jax` now also includes a small
structural adapter layer in ``sfincs_jax/geometry/jax_adapters.py``. The adapter can
accept VMEC-like in-memory objects, including objects with the field layout used by
``vmec_jax.wout.WoutData``, and normalize them to the internal
``sfincs_jax.geometry.vmec_wout.VmecWout`` convention without making ``vmec_jax`` a required
dependency.

This is the intended staged path for JAX-native equilibrium coupling:

1. solve or update an equilibrium with an optional producer such as ``vmec_jax``,
2. convert the in-memory ``wout``-like object with
   ``vmec_wout_from_wout_like(...)``,
3. evaluate the same scheme-5 geometry formulas with
   ``vmec_geometry_from_wout(...)`` that file-based VMEC inputs use through
   ``vmec_geometry_from_wout_file(...)``,
4. pass the resulting arrays to the kinetic operator and, when the upstream geometry
   producer supports it, differentiate through the outer objective.

The first adapter stage is covered by unit tests that check backend discovery,
``(radius, mode)`` to ``(mode, radius)`` transposition, native ``sfincs_jax`` array
ordering, metadata-only path overrides, strict rejection of missing required field
tables, optional zero-filling of absent covariant/contravariant magnetic-field
tables, invalid-shape rejection, and exact equality between the file wrapper and a
preloaded ``VmecWout`` object. When ``vmec_jax`` is installed, an optional
integration gate also reads a real ``vmec_jax.wout.WoutData`` fixture and checks
exact equality of the converted Fourier coefficients and the evaluated scheme-5
geometry arrays.

Minimal adapter workflow:

.. code-block:: python

   import numpy as np
   from vmec_jax.wout import read_wout as read_vmec_jax_wout

   from sfincs_jax.geometry.jax_adapters import vmec_wout_from_wout_like
   from sfincs_jax.geometry.vmec import vmec_geometry_from_wout

   wout_like = read_vmec_jax_wout("wout_circular_tokamak.nc")
   wout = vmec_wout_from_wout_like(wout_like)

   theta = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
   zeta = np.linspace(0.0, 2.0 * np.pi / wout.nfp, 16, endpoint=False)
   geom = vmec_geometry_from_wout(w=wout, theta=theta, zeta=zeta, psi_n_wish=0.25)

The public optional JAX-native handoff example is:

.. code-block:: bash

   python examples/autodiff/vmec_jax_to_boozer_sfincs_pipeline.py --check-backends

That command only reports importability of ``vmec_jax`` and ``booz_xform_jax``
and prints the current differentiability boundary.  It does not import either
optional backend and should run in a normal ``sfincs_jax`` development
environment.

For automation, dashboards, and lab notebooks that should record the same boundary
without parsing human-readable text, use the JSON status mode:

.. code-block:: bash

   python examples/autodiff/vmec_jax_to_boozer_sfincs_pipeline.py --check-backends --json

The JSON report includes shallow backend importability, runnable setup paths,
gradient-availability labels for each stage, a ``workflow_contract`` block, the
differentiated graph, and the explicit non-claim that this is a geometry-proxy
gradient gate rather than a full transport-gradient workflow.  The contract is
also available directly from
``sfincs_jax.geometry.jax_adapters.geometry_proxy_workflow_contract()`` and is
kept intentionally small enough for tests and notebook provenance:

- default CI does not require ``vmec_jax`` or ``booz_xform_jax``,
- ``--check-backends`` uses shallow importability checks and does not import the
  optional packages,
- the only supported gradient claim is
  ``scaled VMEC-like spectral arrays -> booz_xform_jax -> sfincs_jax
  Boozer-spectrum proxy objective``,
- full VMEC-boundary-to-SFINCS kinetic transport gradients are a forbidden
  overclaim for this lane.

Runs that need provenance for publication artifacts can write a reusable workflow
summary:

.. code-block:: bash

   python examples/autodiff/vmec_jax_to_boozer_sfincs_pipeline.py \
     --check-backends \
     --summary-json workflow-summary.json

The same ``--summary-json`` option works for the full geometry-proxy run.  The
summary records stage names, optional dependency importability, the precise
differentiability status of each stage, the numerical gradient-gate status when
the proxy objective is evaluated, and the explicit non-claim that full kinetic
transport gradients are not covered by this lane.

When both optional packages are installed, run the file-backed VMEC setup path
with an explicit ``wout`` file:

.. code-block:: bash

   python examples/autodiff/vmec_jax_to_boozer_sfincs_pipeline.py \
     --wout /path/to/wout_circular_tokamak.nc \
     --mboz 3 \
     --nboz 3 \
     --surface 0.5 \
     --steps 0

The same file can be supplied with
``SFINCS_JAX_VMEC_JAX_WOUT=/path/to/wout.nc``.  If ``vmec_jax`` example decks are
available locally, the script can also build the VMEC-like object before the
Boozer transform:

.. code-block:: bash

   python examples/autodiff/vmec_jax_to_boozer_sfincs_pipeline.py \
     --vmec-case circular_tokamak \
     --vmec-max-iter 1 \
     --steps 0

This script uses ``vmec_jax`` provenance for a VMEC ``wout`` object,
``booz_xform_jax`` for the Boozer transform, and
``sfincs_jax.geometry.jax_adapters.boozer_spectrum_geometry_proxy_objective``
for a differentiable scalar objective.  It reports the objective, the JAX
gradient with respect to a VMEC magnetic-spectrum scale parameter, a centered
finite-difference check, a pass/fail numerical gradient gate for that proxy
path, and a few gradient-descent steps.

The current example validates the differentiable
``VMEC-like spectral arrays -> booz_xform_jax -> sfincs_jax Boozer-spectrum
objective`` graph.  File I/O and the default ``vmec_geometry_from_wout`` file
adapter remain outside the differentiable graph.  Full VMEC-boundary-to-kinetic
transport optimization is still a larger research workflow, but the public handoff
now has a fast, tested gradient gate.

Current differentiability boundary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The optional ``vmec_jax`` / ``booz_xform_jax`` lane should currently be read as a
geometry-handoff and objective-gradient lane, not as a complete transport
optimization workflow.  The supported public pieces are:

- shallow optional-backend discovery through
  ``optional_jax_geometry_backend_status()``, which checks importability without
  importing either optional package,
- structural conversion of VMEC-like ``wout`` objects into the internal
  ``VmecWout`` representation,
- scheme-5 geometry evaluation from a preloaded ``VmecWout`` object, with the
  same formulas as the file-backed VMEC path,
- a Boozer-spectrum proxy objective that is pure JAX and fast enough for a
  finite-difference gradient gate.

The remaining limitations are deliberate.  Reading or writing equilibrium files is
outside the differentiable graph, the scheme-5 VMEC evaluator is still primarily a
NumPy parity implementation, and the public Boozer-spectrum objective is a
geometry proxy rather than a SFINCS kinetic solve.  The next integration steps are
to keep the adapter contract stable, extend pure-JAX geometry evaluation where it
is useful, and only then wire those arrays into transport objectives with explicit
gradient checks.

A separate finite-beta end-to-end user example uses the direct VMEC ``wout`` lane:

.. code-block:: bash

   python examples/vmec_jax_finite_beta/finite_beta_vmec_to_sfincs.py

That script runs the bundled ``input.nfp2_QA_finite_beta`` deck with ``vmec_jax``,
writes a self-contained ``wout_nfp2_QA_finite_beta_vmec_jax.nc`` file, evaluates
scheme-5 geometry in ``sfincs_jax``, scans normalized ``Er`` on multiple radial
surfaces, and plots core-to-edge profiles of ambipolar ``Er`` and bootstrap
current versus normalized toroidal flux :math:`\psi_N = r_N^2`.  The same panel
also includes representative ``Er`` scans, particle fluxes, and a ``jet``
contour plot of the sampled VMEC magnetic-field strength.  The checked example
uses ``Ntheta=7``, ``Nzeta=7``, ``Nxi=8``, ``NL=6``, and ``Nx=6`` for the radial
profile and overlays a tighter ambipolar-root scan that refines every plotted
surface from a local ``Er`` bracket width of ``1.25`` to ``0.625``.  The
companion convergence-scan figure in :doc:`examples` records the remaining
numerical sensitivity: same-grid root-bracket refinement is tight at ``8/6/6``,
but higher combined ``Nxi``/``Nx`` refinement remains a performance-limited
research lane.  A Boozer transform is not needed for this direct VMEC workflow
because the kinetic solve uses the VMEC geometry coefficients from the generated
``wout`` file.

This finite-beta example is a primal transport workflow.  It records radial
profile provenance in its summary JSON, including the requested ``r_N`` surfaces,
the plotted :math:`\psi_N = r_N^2` values, the all-roots versus selected-branch
policy, and the convergence-overlay status.  It does not claim gradients through
the VMEC file handoff, scheme-5 geometry evaluation, SFINCS kinetic solve, or
radial postprocessing.

Boozer ``.bc`` workflow
-----------------------

For ``geometryScheme=11`` and ``12``, `sfincs_jax` reads Boozer-coordinate Fourier
data and evaluates the fields on the requested angular grid. The retained harmonics
follow the representable-mode policy imposed by the discrete grid:

.. math::

   0 \le m \le \left\lfloor \frac{N_\theta}{2} \right\rfloor,
   \qquad
   |n| \le \left\lfloor \frac{N_\zeta}{2} \right\rfloor,

with the expected Nyquist exclusions when sine/cosine pairs would otherwise be
duplicated. This matters numerically because the resolved harmonic content directly
changes both the geometric coefficients and the trapped-passing boundary structure.

The packaged examples can refer to release-hosted ``.bc`` fixtures by basename,
for example ``hsx3free.bc`` or ``w7x_standardConfig.bc``. The resolver verifies
the release archive checksum and each extracted file checksum before using those
fixtures, so CI and user runs do not depend on unreviewed local copies.

Radial coordinates and geometry-derived scales
----------------------------------------------

The geometry layer defines the conversion between several radial labels used
throughout `sfincs_jax`:

.. math::

   \psi, \qquad
   \psi_N = \psi/\psi_a, \qquad
   \hat r = r/a, \qquad
   r_N.

It also computes surface quantities that enter normalization and diagnostics, including

.. math::

   \hat V' = \frac{dV}{d\hat\psi}, \qquad
   \langle \hat B^2 \rangle, \qquad
   \langle 1/\hat B^2 \rangle,

and the metric contractions needed for classical transport and geometry-dependent
moments.

Geometry in the source tree
---------------------------

The main geometry-related modules are:

- ``sfincs_jax/geometry/__init__.py``: normalized geometric fields and coefficient assembly.
- ``sfincs_jax/geometry/{boozer.py,vmec_wout.py,vmec.py,jax_adapters.py}``: Boozer-file,
  VMEC-file, VMEC Fourier-sum, and JAX-native geometry adapter owners.
- ``sfincs_jax/input_compat.py``: equilibrium-file resolution and namelist overrides.
- ``sfincs_jax/diagnostics.py``: geometry-derived scalar diagnostics and moments.
- ``sfincs_jax/operators/profile_system.py``: insertion of geometry coefficients into the kinetic
  operator.
- ``sfincs_jax/operators/profile_magnetic_drifts.py`` and
  ``sfincs_jax/operators/profile_exb.py``:
  construction of drift coefficients from the geometry arrays.

The operator does not carry an opaque geometry object around. Instead, the solve path
works with explicitly normalized arrays. This is deliberate:

- the arrays are cheap to cache and move between JAX transforms,
- they are straightforward to inspect in tests,
- and they make the code-to-equation correspondence easier to maintain.

Internally these arrays are collected in ``sfincs_jax.geometry.BoozerGeometry``.
The class is a flat, frozen dataclass with ``(Ntheta, Nzeta)`` array layout.  Output
writers may transpose selected datasets to match historical HDF5 conventions, but
operator assembly and differentiability tests use the internal layout directly.

What is not a public geometry mode
----------------------------------

There is currently no separate Miller-parameter public geometry interface in the CLI
or Python API. For tokamak studies, the supported public path is the analytic
straight-field-line model family (primarily ``geometryScheme=1``). If a dedicated
Miller workflow is added later, it should appear here as a first-class geometry mode,
with explicit input definitions and examples.

Worked examples
---------------

The repository includes runnable examples for the main user-facing geometry paths:

- analytic tokamak output: ``examples/getting_started/write_sfincs_output_tokamak.py``
- VMEC output with explicit ``wout_path``:
  ``examples/getting_started/write_sfincs_output_vmec.py``
- mixed transport examples using VMEC and Boozer geometry:
  ``examples/transport/transport_matrix_rhsmode2_scheme11_and_scheme5.py``

For the exact input knobs, see :doc:`inputs`. For the way geometry enters the DKE, see
:doc:`system_equations` and :doc:`numerics`.
