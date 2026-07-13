Theory from the upstream SFINCS notes
=====================================

This page pulls the core narrative from the SFINCS technical notes into the main
documentation set. The goal is to keep the theory readable in the docs themselves. The
original upstream notes are unpublished project documents archived in the upstream
SFINCS project repository (cited from :doc:`upstream_docs`); the prose below is an
original summary rather than a copy of those documents.

Ordering and reduction used by SFINCS v3
----------------------------------------

The SFINCS v3 notes start from the steady drift-kinetic equation in guiding-center
coordinates and then make a sequence of reductions tailored to radially local
neoclassical transport on a flux surface:

.. math::

   v_\parallel \mathbf{b}\cdot\nabla f_{s1}
   + \mathbf{v}_E\cdot\nabla f_{s1}
   + \mathbf{v}_m\cdot\nabla f_{s1}
   + \dot{x}_s \partial_{x_s} f_{s1}
   + \dot{\xi} \partial_\xi f_{s1}
   - \sum_b C_{sb}^{\mathrm{lin}}[f_{b1}]
   = S_s.

Here the unknown is the non-Maxwellian correction :math:`f_{s1}` to a background
Maxwellian :math:`f_{s0}`. The independent velocity variables used in v3 are

.. math::

   x_s = \frac{v}{\sqrt{2 T_s / m_s}},
   \qquad
   \xi = \frac{v_\parallel}{v},
   \qquad
   \mu = \frac{v_\perp^2}{2B}.

The upstream v3 note emphasizes three practical modeling decisions that matter for the
code:

- the kinetic equation is written in a flux-coordinate representation that allows
  Boozer, VMEC-derived, and analytic geometry options,
- magnetic-drift, :math:`E\times B`, :math:`\dot{x}`, and :math:`\dot{\xi}` pieces can be
  toggled to recover full, partial, or DKES-like trajectory models,
- and the same normalized state can be used for direct flux calculations or for
  transport-matrix columns.

Normalization hierarchy
-----------------------

The single- and multi-species notes define the normalized quantities used throughout
SFINCS v3 and preserved in ``sfincs_jax``:

.. math::

   \Delta_s = \frac{v_{\mathrm{th},s}}{\Omega_s R},
   \qquad
   \alpha_s = \frac{Z_s e \Phi}{T_s},
   \qquad
   \nu_{n,s} = \frac{\nu_s R}{v_{\mathrm{th},s}},

with :math:`v_{\mathrm{th},s} = \sqrt{2T_s/m_s}` and
:math:`\Omega_s = Z_s e B / (m_s c)`.

These normalizations are not just bookkeeping. They determine:

- the scale separation assumed by the local model,
- the exact factors that appear in the drift and collision operators,
- and the interpretation of inputs and HDF5 outputs.

This is one of the main reasons ``sfincs_jax`` keeps the v3 conventions instead of
recasting the problem in a more generic linear-algebra normalization.

Drives, fluxes, and transport-matrix columns
--------------------------------------------

The single-species note writes the thermodynamic drive in the familiar form

.. math::

   (\mathbf{v}_m + \mathbf{v}_E)\cdot\nabla r
   \left[
   \frac{1}{n_s}\frac{dn_s}{dr}
   + \frac{Z_s e}{T_s}\frac{d\Phi_0}{dr}
   + \left(x_s^2-\frac{3}{2}\right)\frac{1}{T_s}\frac{dT_s}{dr}
   \right] f_{s0},

with additional terms when :math:`\Phi_1` enters the kinetic equation or collision
operator. In SFINCS v3 these drives are reorganized into internal right-hand-side
families:

- direct solve mode for physical fluxes and flows,
- ``RHSMode=2`` transport-matrix columns,
- ``RHSMode=3`` monoenergetic transport coefficients.

The transport-matrix and Beidler-matrix notes are useful here because they make explicit
that SFINCS is not solving a different kinetic problem in those modes. It is solving the
same discrete operator with a controlled basis of source terms and then post-processing
the resulting flux moments into transport coefficients.

Constraint structure and nullspaces
-----------------------------------

A recurring theme in the upstream notes is conservation. The linearized collision
operator must conserve particles, momentum, and energy, and the discrete solve must
remove nullspaces associated with those conserved quantities. In practice this is why
SFINCS carries explicit constraints and why different ``constraintScheme`` choices can
change the algebraic branch even when the physics is equivalent.

The important implementation consequence for ``sfincs_jax`` is that validation is not
only about matching matrix entries. It is also about matching:

- which nullspace or gauge is selected,
- which moments are constrained directly,
- and which post-processed diagnostics are treated as gauge-sensitive when the solve is
  intentionally rank-deficient.

This is also why the fast CLI/default path and the differentiable Python path are
separated: the fast path can use different linear algebra as long as it converges to the
same physically valid constrained state.

Monoenergetic and DKES-like limits
----------------------------------

The upstream SFINCS paper and the DKES notes explain why monoenergetic limits remain
important even in a multispecies code:

- they provide a bridge to older stellarator databases and optimization workflows,
- they isolate geometry and trajectory effects from full energy-coupling effects,
- and they support transport-matrix workflows where many source columns must be solved
  repeatedly.

In this limit the kinetic problem becomes smaller and more structured. That is exactly
why reduced monoenergetic and DKES-style codes can exploit specialized solvers that are not natural for
the full SFINCS state. For ``sfincs_jax``, this is a strong hint that the worst
monoenergetic and low-energy-coupling offenders should not necessarily be treated with
the same generic flattened solver path used for full FP runs.

Fokker-Planck field terms and Rosenbluth potentials
---------------------------------------------------

The 2014 Rosenbluth-potential implementation note is especially important for performance.
Its key observation is that the dense field-particle part of the linearized
Fokker-Planck operator can be organized as:

1. transform from the collocation speed grid to a modal basis,
2. evaluate Rosenbluth-potential integrals and their derivatives in that basis,
3. map the result back to the collocation grid.

The note describes this using a transform matrix :math:`Y` and integral-evaluation
matrices :math:`R`, so the map from a collocation-grid perturbation to the potentials is
represented schematically by :math:`RY`.

Two consequences matter directly for ``sfincs_jax``:

- the expensive part is dense in speed and species but independent of
  :math:`(\theta,\zeta)` once the geometry factors are separated,
- and those dense blocks are structured enough that factor-and-reuse or low-rank updates
  are often more promising than repeated generic Krylov iterations on the full flattened
  state.

This is exactly the direction suggested by structured block elimination and by the
existing species-by-:math:`x` block preconditioners already present in ``sfincs_jax``.

Phi1, quasineutrality, and poloidally varying collisions
--------------------------------------------------------

The :math:`\Phi_1` notes add two layers of physics beyond the simplest local model:

.. math::

   \Phi(\psi,\theta,\zeta) = \Phi_0(\psi) + \Phi_1(\theta,\zeta).

First, :math:`\Phi_1` modifies the kinetic equation through parallel acceleration,
drift terms, and the background Maxwellian. Second, when enabled, it modifies the
collision operator through a poloidally varying effective density or Boltzmann factor.

The practical message from those notes is subtle:

- some flux definitions that look different in the presence of :math:`\Phi_1` reduce to
  the same physical fluxes once all terms are handled consistently,
- quasineutrality is part of the model closure rather than a cosmetic post-processing
  step,
- and nonlinear :math:`\Phi_1` solves are more than a simple outer loop because the
  Jacobian and source terms both change with the potential.

This is why ``sfincs_jax`` keeps a more conservative, reference-oriented path for
requested differentiable ``\Phi_1`` workflows while allowing the CLI/default path to use
more aggressive explicit strategies when they converge to the same final state.

What this means for sfincs_jax engineering
------------------------------------------

The upstream notes point to a consistent implementation strategy:

- keep the equations, normalizations, and diagnostics aligned with SFINCS v3,
- preserve explicit access to the natural block structure in species, speed, Legendre,
  and angle coordinates,
- and avoid treating every hard case as an unstructured dense or generic Krylov problem.

That leads directly to the current engineering split in ``sfincs_jax``:

- a validation-oriented differentiable path for explicit Python requests and gradient
  workflows,
- a performance-oriented explicit CLI/default path,
- and a continuing push to encode more of the true operator structure in the
  preconditioners and fast solvers.

Primary sources summarized here
-------------------------------

The published reference for the model is M. Landreman, H. M. Smith, A. Mollén, and
P. Helander, *Physics of Plasmas* **21**, 042503 (2014),
`doi:10.1063/1.4870077 <https://doi.org/10.1063/1.4870077>`_.

The narrative above additionally summarizes the following **unpublished upstream
SFINCS project documents** (archived in the upstream repository,
`github.com/landreman/sfincs <https://github.com/landreman/sfincs>`_; see
:doc:`upstream_docs`):

- Technical documentation for version 3 of SFINCS.
- Technical documentation for SFINCS with a single species, and with multiple species.
- Implementation of the Fokker--Planck operator.
- Effects on fluxes of including :math:`\Phi_1`, and the :math:`\Phi_1` implementation note.
- The poloidal-variation-in-collision-operator note.
