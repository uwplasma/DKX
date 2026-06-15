"""Compatibility alias for :mod:`sfincs_jax.problems.transport_matrix.sparse_direct_solve`."""

from __future__ import annotations

import sys as _sys

from .problems.transport_matrix import sparse_direct_solve as _impl

_sys.modules[__name__] = _impl
