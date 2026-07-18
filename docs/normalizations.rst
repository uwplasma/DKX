Normalizations and units
========================

SFINCS v3 (and `dkx`) mixes *physical* quantities (:math:`n_s`,
:math:`T_s`, ...) with dimensionless *normalized* quantities written with hats
(``BHat`` :math:`=\hat B`, ...). This page collects the normalization
conventions used throughout the code and the ``sfincsOutput.h5`` fields. The
drift-kinetic equation that uses them is derived in :doc:`physics_reference`.

Reference scales and hats
-------------------------

All hatted fields are normalized to reference scales: lengths to :math:`\bar R`,
magnetic field to :math:`\bar B`, so that

.. math::

   \hat B = \frac{B}{\bar B}, \qquad
   \hat\psi = \frac{\psi}{\bar B\,\bar R^2}, \qquad
   \hat r = \frac{r}{\bar R},

and the Boozer flux functions are :math:`\hat G = \langle B_\zeta\rangle/\bar B\bar R`,
:math:`\hat I = \langle B_\theta\rangle/\bar B\bar R` (``GHat``, ``IHat``).

The drift-kinetic ordering parameters
-------------------------------------

Three scalars set the physical regime and are supplied through the namelist as
``Delta``, ``alpha``, ``nu_n``:

.. math::

   \Delta = \frac{\bar m\,\bar v}{Z\,e\,\bar B\,\bar R}
          \;\sim\; \frac{v_{\mathrm{th}}}{\Omega\,\bar R},
   \qquad
   \alpha = \frac{e\,\bar\Phi}{\bar T},
   \qquad
   \nu_n = \frac{\bar\nu\,\bar R}{\bar v}.

- :math:`\Delta` (``Delta``) — drift/gyroradius ordering parameter; multiplies
  every drift term and every radial flux.
- :math:`\alpha` (``alpha``) — potential-to-energy normalization; appears in the
  :math:`E\times B` drive, the :math:`\Phi_1` Boltzmann factor, and
  quasineutrality.
- :math:`\nu_n` (``nu_n``) — collision-frequency normalization multiplying the
  collision operator.

All three are written verbatim into ``sfincsOutput.h5`` (:doc:`outputs`).

Velocity-space coordinates
--------------------------

The drift-kinetic equation is solved on the 4D phase space
:math:`(\theta,\zeta,x,\xi)`:

- :math:`\theta,\zeta` — Boozer angles (or the appropriate substitutes for the
  geometry scheme; see :doc:`geometry`);
- :math:`x_s = v/\sqrt{2T_s/m_s}` — normalized speed, discretized on the
  Landreman--Ernst grid over a finite interval :math:`0\le x\le x_{\max}`;
- :math:`\xi = v_\parallel/v` — pitch-angle cosine, expanded in Legendre modes.

The species Maxwellian is
:math:`f_M = n_s\,(m_s/2\pi T_s)^{3/2}\,e^{-x_s^2}` and the distribution is split
:math:`f_s=f_{s0}+f_{s1}` (see :doc:`physics_reference`).

Flux-surface averages
---------------------

Constraints and diagnostics use the Jacobian-weighted flux-surface average

.. math::

   \langle g\rangle
   = \frac{\int d\theta\,d\zeta\; g(\theta,\zeta)\,\hat D(\theta,\zeta)^{-1}}
          {\int d\theta\,d\zeta\; \hat D(\theta,\zeta)^{-1}},

evaluated with the quadrature weights ``thetaWeights`` / ``zetaWeights`` and the
geometry factor ``DHat``. Two derived scalars appear throughout:
:math:`\hat V' = \sum_{ij} w^\theta_i w^\zeta_j/\hat D_{ij}` (``VPrimeHat``) and
:math:`\langle\hat B^2\rangle` (``FSABHat2``).

Radial coordinates
------------------

`dkx` supports four radial labels, selected by ``inputRadialCoordinate``
for the flux surface and ``inputRadialCoordinateForGradients`` for the profile
gradients:

.. math::

   \psi \;(\texttt{=0}), \quad
   \psi_N = \psi/\psi_a \;(\texttt{=1}), \quad
   \hat r \;(\texttt{=2}), \quad
   r_N = r/a \;(\texttt{=3}), \quad
   E_r \;(\texttt{=4, gradients}).

Each per-species flux is emitted in all four coordinate variants
(``*_psiHat``, ``*_psiN``, ``*_rHat``, ``*_rN``); the conversion multiplies the
native :math:`\nabla\hat\psi` flux by :math:`d(\text{coord})/d\hat\psi`
(:doc:`outputs`).

Radial electric field
---------------------

The equilibrium radial electric field is :math:`E_r=-d\Phi_0/dr`. The operator
coefficients are written in terms of :math:`d\hat\Phi/d\hat\psi` (the ``Er``,
``dPhiHatdpsiHat``, ``dPhiHatdrHat``, ... inputs are inter-converted through the
radial-coordinate factors). The ambipolar solve drives ``Er`` directly
(``inputRadialCoordinate = 4``); see :doc:`physics_reference` and
:mod:`dkx.er`.

Monoenergetic collisionality and electric field
-----------------------------------------------

For monoenergetic (``RHSMode = 3``) runs the physical collisionality and
electric field are supplied as ``nuPrime`` and ``EStar`` instead of ``nu_n`` and
``dPhiHatdpsiHat``:

.. math::

   \nu' = \frac{(\hat G + \iota\,\hat I)\,\hat\nu}{v\,\hat B_0},
   \qquad
   E^\* = \frac{\hat G}{\iota\,v\,\hat B_0}\,\frac{d\hat\Phi}{d\hat\psi},

with :math:`\hat B_0` the :math:`(0,0)` Fourier mode of :math:`\hat B`. In this
mode the speed grid collapses to a single node at :math:`x=1`, and ``nuPrime`` /
``EStar`` are written to ``sfincsOutput.h5`` alongside the :math:`2\times2`
``transportMatrix``. For the full upstream unit derivations see the SFINCS technical
notes cited from :doc:`upstream_docs`.
