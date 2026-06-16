"""Compatibility alias for :mod:`sfincs_jax.problems.transport_matrix.direct_block_schur`."""

from __future__ import annotations

import sys as _sys

from .problems.transport_matrix import direct_block_schur as _impl

_sys.modules[__name__] = _impl
