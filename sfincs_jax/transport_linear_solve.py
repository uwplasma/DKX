"""Compatibility alias for transport linear-solve helpers."""

from __future__ import annotations

import sys as _sys

from .problems.transport_matrix import linear_solve as _impl

_sys.modules[__name__] = _impl
