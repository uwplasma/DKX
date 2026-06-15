"""Compatibility alias for transport KSP-iteration diagnostics."""

from __future__ import annotations

import sys as _sys

from .problems.transport_matrix import iteration_stats as _impl

_sys.modules[__name__] = _impl
