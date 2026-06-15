"""Compatibility alias for transport solve-policy helpers."""

from __future__ import annotations

import sys as _sys

from .problems.transport_matrix import solve_policy as _impl

_sys.modules[__name__] = _impl
