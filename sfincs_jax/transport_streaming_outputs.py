"""Compatibility shim for streaming transport output helpers."""

from __future__ import annotations

import sys as _sys

from .problems.transport_matrix import streaming_outputs as _impl

_sys.modules[__name__] = _impl
