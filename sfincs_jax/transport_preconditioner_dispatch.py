"""Compatibility alias for :mod:`sfincs_jax.problems.transport_matrix.preconditioner_dispatch`."""

from __future__ import annotations

import sys as _sys

from .problems.transport_matrix import preconditioner_dispatch as _impl

_sys.modules[__name__] = _impl
