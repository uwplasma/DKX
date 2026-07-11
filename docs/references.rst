References and related work
===========================

This page collects the main literature that informs the physics model, numerics,
validation strategy, and workflow design of `sfincs_jax`.

Foundational neoclassical theory
--------------------------------

- P. Helander and D. J. Sigmar,
  *Collisional Transport in Magnetized Plasmas*, Cambridge University Press (2002).
  The standard textbook derivation of the drift-kinetic equation, the linearized
  Fokker--Planck collision operator, and the neoclassical flux/flow moments that
  underpin :doc:`physics_reference`.
- P. Helander,
  `Theory of plasma confinement in non-axisymmetric magnetic fields <https://doi.org/10.1088/0034-4885/77/8/087001>`_,
  Rep. Prog. Phys. **77**, 087001 (2014). Review of stellarator neoclassical
  theory, ambipolarity, and the :math:`1/\nu`, :math:`\sqrt{\nu}`, and plateau regimes.
- A. H. Boozer,
  `Guiding center drift equations <https://www.osti.gov/biblio/5655342>`_.
- H. Sugama and S. Nishimura,
  `How to calculate the neoclassical viscosity, diffusion, and current coefficients in general toroidal plasmas <https://doi.org/10.1063/1.1512917>`_,
  Phys. Plasmas **9**, 4637 (2002). Trajectory and moment-equation conventions
  relevant to the SFINCS ``full`` and DKES trajectory models.

SFINCS model, collision operator, and speed grid
-------------------------------------------------

- M. Landreman, H. M. Smith, A. Mollén, and P. Helander,
  `Comparison of particle trajectories and collision operators for collisional transport in nonaxisymmetric plasmas <https://doi.org/10.1063/1.4870077>`_,
  Phys. Plasmas **21**, 042503 (2014). The SFINCS paper: the radially local
  drift-kinetic model, the ``Delta``/``alpha``/``nu_n`` normalization, and the
  full-vs-DKES trajectory comparison implemented in
  :mod:`sfincs_jax.drift_kinetic`.
- M. Mollén et al.,
  `Implementation of a full linearized Fokker-Planck collision operator in SFINCS <https://arxiv.org/abs/1504.04810>`_.
  Basis for the Fokker--Planck operator in :mod:`sfincs_jax.collisions`.
- M. Landreman and D. R. Ernst,
  `New velocity-space discretization for continuum kinetic calculations and Fokker--Planck collisions <https://arxiv.org/abs/1210.5289>`_,
  J. Comput. Phys. **243**, 130 (2013). The non-classical orthogonal-polynomial
  speed grid and Rosenbluth-potential field-term treatment implemented in
  :func:`sfincs_jax.phase_space.make_speed_grid` and the Rosenbluth-potential
  terms in :mod:`sfincs_jax.collisions`.
- A. Redl et al.,
  `A new set of analytical formulae for the computation of the bootstrap current and the neoclassical conductivity in tokamaks <https://doi.org/10.1063/5.0012664>`_,
  Phys. Plasmas **28**, 022502 (2021). Bootstrap-current formula used as an
  analytic cross-check for :math:`\langle \mathbf{j}\cdot\mathbf{B}\rangle`.
- O. Sauter, C. Angioni, and Y. R. Lin-Liu,
  `Neoclassical conductivity and bootstrap current formulas for general axisymmetric equilibria and arbitrary collisionality regime <https://doi.org/10.1063/1.873240>`_.
- M. Landreman and E. J. Paul,
  `Magnetic Fields with Precise Quasisymmetry for Plasma Confinement <https://doi.org/10.1103/PhysRevLett.128.035001>`_.

Block-tridiagonal Legendre solver
---------------------------------

The tier-1 structured solve (:doc:`numerics`) eliminates the Legendre chain of
the monoenergetic drift-kinetic equation with a block-tridiagonal factorization
and a truncated-storage back-substitution:

- F. J. Escoto,
  `Fast and accurate calculation of the bootstrap current and radial neoclassical transport in low collisionality stellarator plasmas <https://arxiv.org/abs/2510.27513>`_,
  PhD thesis (2025). Derives the tridiagonal structure of the Legendre-mode
  representation of the monoenergetic drift-kinetic equation and the block
  elimination that :meth:`sfincs_jax.drift_kinetic.KineticOperator.to_block_tridiagonal`
  and :func:`sfincs_jax.solve.solve` exploit.

Geometry and benchmark configurations
-------------------------------------

- C. D. Beidler et al.,
  `Benchmarking of the mono-energetic transport coefficients -- results from the International Collaboration on Neoclassical Transport in Stellarators (ICNTS) <https://doi.org/10.1088/0029-5515/51/7/076001>`_,
  Nucl. Fusion **51**, 076001 (2011). Source of the analytic W7-X / LHD harmonic
  tables used by :meth:`sfincs_jax.magnetic_geometry.FluxSurfaceGeometry.from_scheme`
  (geometry schemes 2/3/4) and of the monoenergetic benchmark coefficients.

Experimental and cross-code validation anchors
----------------------------------------------

- N. A. Pablant et al.,
  `Core radial electric field and transport in Wendelstein 7-X plasmas <https://sites.fusion.ciemat.es/jlvelasco/files/papers/pablant2018er.pdf>`_.
- N. A. Pablant et al.,
  `Investigation of the ion-root solution in Wendelstein 7-X <https://sites.fusion.ciemat.es/jlvelasco/files/papers/pablant2020ionroot.pdf>`_.
- C. D. Beidler et al.,
  `Demonstration of reduced neoclassical energy transport in Wendelstein 7-X <https://www.nature.com/articles/s41586-021-03687-w>`_.
- J. L. Velasco et al.,
  `KNOSOS: A fast orbit-averaging neoclassical code for stellarator optimization studies <https://arxiv.org/abs/1908.11615>`_.

Bundled technical notes and manuals
-----------------------------------

The repository includes the main long-form references under ``docs/upstream``:

- `20150507-01 Technical documentation for version 3 of SFINCS <docs/upstream/20150507-01%20Technical%20documentation%20for%20version%203%20of%20SFINCS.pdf>`_
- `20150402-01 Implementation of the Fokker-Planck operator <docs/upstream/20150402-01%20Implementation%20of%20the%20Fokker-Planck%20operator.pdf>`_
- `20150325-01 Effects on fluxes of including Phi_1 <docs/upstream/20150325-01%20Effects%20on%20fluxes%20of%20including%20Phi_1.pdf>`_
- `SFINCS paper <docs/upstream/sfincsPaper/sfincsPaper.pdf>`_
- `Version 3 user manual sources <docs/upstream/manual/version3/SFINCSUserManual.tex>`_

JAX and differentiable programming
----------------------------------

For implicit differentiation through linear solves (and other solver-aware workflows), see:

- `JAX custom linear solve <https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.custom_linear_solve.html>`_
- `JAX linear transpose <https://jax.readthedocs.io/en/latest/_autosummary/jax.linear_transpose.html>`_
- `JAX sparse linear algebra <https://jax.readthedocs.io/en/latest/jax.scipy.html>`_

Testing, validation, and coverage methodology
---------------------------------------------

The testing strategy is informed by scientific-software verification work and
by empirical evidence that line coverage is useful for finding untested code but
is not sufficient as a quality target:

- U. Kanewala and J. M. Bieman,
  `Testing Scientific Software: A Systematic Literature Review <https://arxiv.org/abs/1804.01954>`_.
- S. Segura et al.,
  `Metamorphic Testing: Testing the Untestable <https://doi.org/10.1109/MS.2018.2875968>`_.
- L. Inozemtseva and R. Holmes,
  `Coverage Is Not Strongly Correlated with Test Suite Effectiveness <https://www.cs.ubc.ca/~rtholmes/papers/icse_2014_inozemtseva.pdf>`_.
- `Hypothesis property-based testing documentation <https://hypothesis.readthedocs.io/>`_.
- `Chex documentation for JAX test variants and assertions <https://chex.readthedocs.io/>`_.
- `JAX checkify documentation for functionalized runtime checks <https://docs.jax.dev/en/latest/jax.experimental.checkify.html>`_.

Linear algebra and preconditioning
----------------------------------

The solver stack in `sfincs_jax` draws on standard Krylov and preconditioning references:

- Y. Saad and M. Schultz, “GMRES: A generalized minimal residual algorithm for solving
  nonsymmetric linear systems,” *SIAM J. Sci. Stat. Comput.* 7(3), 1986.
- H. A. van der Vorst, “Bi-CGSTAB: A fast and smoothly converging variant of Bi-CG,”
  *SIAM J. Sci. Stat. Comput.* 13(2), 1992.
- P. Sonneveld and M. B. van Gijzen, “IDR(s): A family of simple and fast algorithms for
  solving large nonsymmetric systems of linear equations,” *SIAM J. Sci. Comput.* 31(2), 2008.
- M. A. Woodbury, “Inverting modified matrices,” *Statistical Research Group Memo Report*, 1950
  (Woodbury identity / low‑rank updates).
- G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., Johns Hopkins Univ. Press, 2013
  (Schur complements, block factorization).
- M. de Sturler, “Truncation strategies for optimal Krylov subspace methods,”
  *SIAM J. Numer. Anal.* 36(3), 1999 (GCRO/deflation concepts).
- M. Benzi, “Preconditioning techniques for large linear systems: a survey,” *J. Comput. Phys.*
  182(2), 2002.

Optimization-focused neoclassical workflows
-------------------------------------------

Related reduced-model thesis and paper materials are useful for:

- adjoint properties of drift-kinetic equations,
- derivative-aware workflows for optimization,
- and convergence/scaling studies that inform regression tests and benchmarks.

Recent applications (examples to prioritize)
--------------------------------------------

The following papers motivate transport and optimization-oriented examples:

- `Recent progress on neoclassical impurity transport in stellarators with implications for a stellarator reactor <https://doi.org/10.1088/1741-4326/ac2fda>`_
- `Electron root optimisation for stellarator reactor designs <https://arxiv.org/abs/2405.12058>`_
