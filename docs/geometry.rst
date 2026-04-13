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

- ``sfincs_jax/geometry.py``: normalized geometric fields and coefficient assembly.
- ``sfincs_jax/input_compat.py``: equilibrium-file resolution and namelist overrides.
- ``sfincs_jax/diagnostics.py``: geometry-derived scalar diagnostics and moments.
- ``sfincs_jax/v3_system.py``: insertion of geometry coefficients into the kinetic
  operator.
- ``sfincs_jax/magnetic_drifts.py`` and ``sfincs_jax/collisionless_exb.py``:
  construction of drift coefficients from the geometry arrays.

The operator does not carry an opaque geometry object around. Instead, the solve path
works with explicitly normalized arrays. This is deliberate:

- the arrays are cheap to cache and move between JAX transforms,
- they are straightforward to inspect in tests,
- and they make the code-to-equation correspondence easier to maintain.

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
