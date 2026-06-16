"""Compatibility alias for :mod:`sfincs_jax.solvers.preconditioners.pas.xblock_ilu`."""

from __future__ import annotations

import sys as _sys

from .solvers.preconditioners.pas import xblock_ilu as _impl

_sys.modules[__name__] = _impl
