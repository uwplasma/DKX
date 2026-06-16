"""Compatibility alias for :mod:`sfincs_jax.problems.transport_matrix.fortran_reduced_lu`."""

from __future__ import annotations

import sys as _sys

from .problems.transport_matrix import fortran_reduced_lu as _impl

_sys.modules[__name__] = _impl
