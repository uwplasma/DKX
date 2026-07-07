"""Compatibility alias for the Krylov solver implementation.

The implementation lives in :mod:`sfincs_jax.solvers.krylov`.  This module
keeps the historical ``sfincs_jax.solver`` import path stable for users while
removing the large implementation from the package root.
"""

from __future__ import annotations

import sys as _sys

from .solvers import krylov as _krylov

_sys.modules[__name__] = _krylov
