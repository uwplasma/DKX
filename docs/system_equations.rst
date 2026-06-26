Drift-kinetic equation and system of equations
==============================================

`sfincs_jax` starts from the radially local, steady drift-kinetic equation on one
flux surface. The unknown is the non-adiabatic part of the species distribution,
``f_s1``; outputs such as particle flux, heat flux, parallel flow, bootstrap-current
proxies, NTV, and transport-matrix coefficients are velocity-space and flux-surface
moments of this solved perturbation.

The continuous equation is treated by expanding pitch-angle dependence in a
Legendre representation, discretizing speed on the ``x`` grid, discretizing
``theta``/``zeta`` on a periodic flux-surface grid, and applying finite-difference
or spectral-style periodic stencils to the streaming, drift, and mirror terms. The
collision block is either the pitch-angle-scattering model or the full linearized
Fokker-Planck/Rosenbluth operator, depending on ``collisionOperator``. Only after
this discretization does the problem become the block linear/nonlinear algebraic
system described below.

In its most general supported configuration, `sfincs_jax` solves a coupled system consisting of:

1) a drift-kinetic equation (DKE) for each kinetic species,
2) an optional quasineutrality equation for the flux-surface variation of the electrostatic potential
   :math:`\Phi_1(\theta,\zeta)`,
3) auxiliary constraints that remove nullspaces and enforce moment conditions.

This page summarizes the block structure and the equations most relevant to the linear
system that `sfincs_jax` assembles and solves.
For physics background and literature references, see :doc:`physics_models`
and :doc:`physics_reference`.

For full upstream context and derivations, see the vendored v3 manual and technical docs in
``docs/upstream/`` (linked from :doc:`upstream_docs`).

Unknown ordering
----------------

The operator uses a global ordering for the state vector and the rows/columns of the
master system that follows the mature SFINCS-style block layout because that structure
is physically natural and convenient for diagnostics, comparison, and testing.

For the common ``readExternalPhi1 = .false.`` case:

- The **F-block** (distribution function) is ordered first, with indices nested as:

  ``species → x → xi/Legendre mode → theta → zeta``.

- If ``includePhi1 = .true.``, the **Phi1 block** (labeled ``BLOCK_QN`` in Fortran) contributes
  :math:`N_\theta N_\zeta` additional unknowns corresponding to :math:`\Phi_1(\theta,\zeta)` on the grid.

- A final scalar unknown ``lambda`` enforces the constraint :math:`\langle \Phi_1 \rangle = 0`.

- Constraint-scheme-dependent source unknowns (and their corresponding constraint rows) are appended last.

Linearized drift-kinetic equation (RHSMode=1)
---------------------------------------------

SFINCS v3 solves a **linearized, steady-state** drift-kinetic equation for the non-adiabatic part
of the distribution function on a single flux surface. We write

.. math::

   f_s = f_{s0} + f_{s1},

where :math:`f_{s0}` is the Maxwellian (optionally modified by :math:`\Phi_1`) and :math:`f_{s1}` is
the unknown. In normalized v3 form, the matrix-free operator assembled by `sfincs_jax` corresponds to

.. math::

   \mathcal{L}_s[f_{s1}] \;=\; S_s,

with

.. math::

   \mathcal{L}_s
   =
   \underbrace{v_\parallel \mathbf{b}\cdot\nabla}_{\text{streaming}}
   +\underbrace{\mathcal{M}}_{\text{mirror}}
   +\underbrace{\mathbf{v}_{E}\cdot\nabla}_{E\times B}
   +\underbrace{\mathbf{v}_{m}\cdot\nabla}_{\text{magnetic drifts}}
   +\underbrace{\dot{x}\,\partial_x + \dot{\xi}\,\partial_\xi}_{\text{energy / pitch-angle drifts}}
   -\underbrace{\sum_b \mathcal{C}^{\mathrm{lin}}_{sb}}_{\text{collisions}},

and a source term :math:`S_s` that includes the background thermodynamic drives
(:math:`\partial_\psi n_s`, :math:`\partial_\psi T_s`), the inductive electric field
(:math:`E_\parallel`), and (in transport-matrix modes) the v3 `whichRHS` overwrites.

The explicit expressions for the ExB and magnetic-drift prefactors are summarized in
:doc:`method`, with longer-form derivations in :doc:`physics_reference`.

Term-to-input switch mapping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The equation above is the *full* v3 linear operator. Each term is enabled (or modified)
by specific namelist parameters:

- **Parallel streaming + mirror**: always included (in Legendre basis, couples :math:`L\leftrightarrow L\pm 1`).
- **ExB drift terms** (:math:`\mathbf{v}_E\cdot\nabla` in :math:`\theta,\zeta`):
  controlled by ``useDKESExBDrift`` (DKES vs default v3 coefficient form) and by
  the radial electric field input (``dPhiHatdpsiHat``).
- **Magnetic drift terms** (:math:`\mathbf{v}_m\cdot\nabla` + associated :math:`\partial_\xi` term):
  enabled when ``magneticDriftScheme > 0`` (scheme ``1`` is fully supported in `sfincs_jax`);
  upwinding is controlled by ``magneticDriftDerivativeScheme``.
- **Energy derivative term** (:math:`\dot{x}\,\partial_x`): enabled with ``includeXDotTerm = .true.``.
- **Pitch-angle drift term** (:math:`\dot{\xi}\,\partial_\xi`): included in v3 with the ExB and magnetic
  drift models; its coefficient matches ``populateMatrix.F90`` for the supported drift schemes.
- **Collisions**: choose with ``collisionOperator``:

  - ``collisionOperator = 1`` → pitch-angle scattering (PAS), diagonal in :math:`L`.
  - ``collisionOperator = 0`` → full linearized Fokker–Planck operator (Rosenbluth form), dense in :math:`x`
    and coupled across species.

  Additional collision options:

  - ``includeTemperatureEquilibrationTerm``: adds the :math:`C_{ab}[f_{aM}, f_{bM}]` term to the RHS.
  - ``includePhi1InCollisionOperator`` (requires ``includePhi1InKineticEquation``): modifies collision
    coefficients with the poloidal density factor :math:`\exp(-Z_s \alpha \Phi_1/T_s)`.

- **Phi1 coupling in the kinetic equation**:
  ``includePhi1 = .true.`` enables the QN block; ``includePhi1InKineticEquation = .true.`` inserts the
  Phi1-dependent terms in the DKE. If ``readExternalPhi1 = .true.``, :math:`\Phi_1` is read from file
  and the kinetic system is solved linearly with fixed :math:`\Phi_1`.
- **Quasineutrality + adiabatic response**:
  ``includePhi1`` + ``quasineutralityOption`` select the QN equation, and ``withAdiabatic = .true.``
  adds adiabatic species contributions.

Transport-matrix modes (RHSMode=2/3)
------------------------------------

For ``RHSMode=2`` (transport matrix) and ``RHSMode=3`` (monoenergetic transport), v3 loops over
``whichRHS`` and **overwrites** selected drives *internally* before forming the RHS:

- ``dnHatdpsiHat`` / ``dTHatdpsiHat`` (thermodynamic gradients),
- ``EParallelHat`` (inductive field),
- and, for ``RHSMode=3``, the single-point ``x=1`` grid used to compute monoenergetic coefficients.

`sfincs_jax` mirrors this behavior via
:func:`sfincs_jax.operators.profile_response.system.with_transport_rhs_settings`, ensuring that the transport-mode
RHS matches v3 ``evaluateResidual(f=0)`` exactly.

Quasineutrality and Phi1 constraint
-----------------------------------

The coupled system includes a quasineutrality condition and the constraint
:math:`\langle \Phi_1 \rangle = 0`:

.. math::

   \lambda + \sum_s Z_s \int d^3v\; (f_{s0} + f_{s1}) = 0,
   \qquad
   \langle \Phi_1 \rangle = 0.

In the fully nonlinear v3 configuration, the quasineutrality equation contains additional
nonlinear dependence through :math:`f_{s0}(\Phi_1)` and (depending on options) adiabatic responses.

Implementation note
-------------------

Not every optional Phi1 coupling that appears in the extended literature is active in
every public workflow. The supported scope is documented in :doc:`inputs`,
:doc:`outputs`, and :doc:`fortran_comparison`, and the code paths that construct these
blocks live primarily in ``sfincs_jax/operators/profile_response/system.py`` and ``sfincs_jax/collisions.py``.
