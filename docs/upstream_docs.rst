Upstream SFINCS sources and primary literature
==============================================

`sfincs_jax` reimplements the radially local, multi-species drift-kinetic model of
the mature SFINCS code. Rather than vendoring copies of the upstream papers and
internal technical notes, this documentation transcribes the physics and numerical
content that `sfincs_jax` depends on into its own pages and cites the primary
sources below.

The most important derived physics and numerical content lives in:

- :doc:`theory_from_upstream`
- :doc:`physics_reference`
- :doc:`system_equations`
- :doc:`method`

Published reference
-------------------

The peer-reviewed description of the SFINCS model — the radially local drift-kinetic
equation, the ``Delta``/``alpha``/``nu_n`` normalization, and the full-vs-DKES
trajectory comparison — is:

- M. Landreman, H. M. Smith, A. Mollén, and P. Helander,
  "Comparison of particle trajectories and collision operators for collisional
  transport in nonaxisymmetric plasmas,"
  *Physics of Plasmas* **21**, 042503 (2014),
  `doi:10.1063/1.4870077 <https://doi.org/10.1063/1.4870077>`_.

See :doc:`references` for the full literature list, including the Fokker--Planck
operator, the orthogonal-polynomial speed grid, and the geometry/benchmark sources.

Upstream project (technical notes and manuals)
----------------------------------------------

The longer-form SFINCS technical documentation — the version-3 technical notes, the
single- and multi-species derivations, the Fokker--Planck implementation note, the
:math:`\Phi_1`/quasineutrality notes, the classical-fluxes and DKES-limit notes, and
the SFINCS user manual — are **unpublished upstream project documents**. They are not
redistributed here. Their archival home is the upstream SFINCS project repository:

- `SFINCS project (github.com/landreman/sfincs) <https://github.com/landreman/sfincs>`_

Readers who need the original derivations should consult that repository directly.
The material that `sfincs_jax` actually relies on is reproduced, in original prose,
on the derived-theory pages linked above.
