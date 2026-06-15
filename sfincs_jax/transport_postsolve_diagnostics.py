"""Compatibility shim for transport postsolve diagnostics helpers."""

from __future__ import annotations

import sys as _sys

from .problems.transport_matrix import postsolve_diagnostics as _impl

_sys.modules[__name__] = _impl
