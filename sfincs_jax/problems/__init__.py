"""Physics problem packages that orchestrate reusable operators and solvers.

The ambipolar ``E_r`` problem now lives on the canonical stack in
:mod:`sfincs_jax.er` (``find_ambipolar_er`` / ``ambipolar_er``); the legacy
in-process Brent/Newton owner ``problems/ambipolar.py`` was deleted in that
slice.
"""

from __future__ import annotations

__all__ = ()
