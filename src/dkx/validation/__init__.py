"""Runtime validation helpers: Fortran/PETSc fixtures and external equilibria.

Two modules, both with production consumers: :mod:`~dkx.validation.fortran`
drives the reference SFINCS binary and reads its PETSc fixtures (``dkx
run-fortran`` and the optimization workflow import it), and
:mod:`~dkx.validation.data_fetch` resolves large equilibria from release assets
for :mod:`dkx.paths`.

The release-engineering half of this package -- release gates, evidence
registry, benchmark-artifact policy, publication panels and series -- moved to
``tools/release/`` in the checkout.  It reads tracked files and manifests that
no wheel contains, so it could never run from an installed package.
"""

from __future__ import annotations

__all__: tuple[str, ...] = ()
