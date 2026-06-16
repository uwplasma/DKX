"""Compatibility alias for :mod:`sfincs_jax.problems.transport_matrix.parallel.execution`."""

from __future__ import annotations

import sys as _sys

from .problems.transport_matrix.parallel import execution as _impl

_sys.modules[__name__] = _impl
