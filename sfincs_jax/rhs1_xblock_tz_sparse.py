"""Compatibility alias for :mod:`sfincs_jax.solvers.preconditioners.xblock.tz_sparse`."""

from __future__ import annotations

import sys as _sys

from .solvers.preconditioners.xblock import tz_sparse as _impl

_sys.modules[__name__] = _impl
