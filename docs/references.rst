References and related work
===========================

This page collects the main literature that informs the physics model, numerics,
validation strategy, and workflow design of `sfincs_jax`.

Core neoclassical and SFINCS literature
---------------------------------------

- M. Landreman, H. M. Smith, M. Mollén, and P. Helander,
  `Comparison of particle trajectories and collision operators for collisional transport in nonaxisymmetric plasmas <https://doi.org/10.1063/1.4870073>`_.
- M. Mollén et al.,
  `Implementation of a full linearized Fokker-Planck collision operator in SFINCS <https://arxiv.org/abs/1504.04810>`_.
- A. H. Boozer,
  `Guiding center drift equations <https://www.osti.gov/biblio/5655342>`_.

Experimental and cross-code validation anchors
----------------------------------------------

- N. A. Pablant et al.,
  `Core radial electric field and transport in Wendelstein 7-X plasmas <https://sites.fusion.ciemat.es/jlvelasco/files/papers/pablant2018er.pdf>`_.
- N. A. Pablant et al.,
  `Investigation of the ion-root solution in Wendelstein 7-X <https://sites.fusion.ciemat.es/jlvelasco/files/papers/pablant2020ionroot.pdf>`_.
- C. D. Beidler et al.,
  `Demonstration of reduced neoclassical energy transport in Wendelstein 7-X <https://www.nature.com/articles/s41586-021-03687-w>`_.
- F. J. Escoto et al.,
  `MONKES: a fast neoclassical code for the evaluation of monoenergetic transport coefficients <https://arxiv.org/abs/2312.12248>`_.
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
