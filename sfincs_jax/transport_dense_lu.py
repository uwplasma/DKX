"""Compatibility alias for :mod:`sfincs_jax.problems.transport_matrix.dense_lu`."""

from __future__ import annotations

import sys as _sys

from .problems.transport_matrix import dense_lu as _impl

_sys.modules[__name__] = _impl
