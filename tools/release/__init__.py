"""Checkout-only release engineering, evidence-registry, and audit tooling.

These modules used to live in ``src/dkx/validation/``, where they shipped in
every wheel and could not run at all for a pip user: each one reads tracked
files, manifests, docs and campaign directories that exist in a repository
checkout and not in ``site-packages``.  ``dkx.paths.repository_root()`` returns
``None`` there, so the whole surface was dead weight in the installed package.

They are unchanged apart from their import lines.  ``tools/`` is not an
installed package; like ``tools/release_contracts.py`` these modules are
imported with the repository root on ``sys.path`` (``tests/conftest.py`` puts it
there) or run directly, for example::

    python -m tools.release.registry
    python -m tools.release.release check-gates

The runtime half of the old package -- ``dkx.validation.fortran`` and
``dkx.validation.data_fetch`` -- stayed behind, because production code imports
both.
"""

from __future__ import annotations

__all__: tuple[str, ...] = ()
