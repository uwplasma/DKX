"""Compatibility shim for transport active/dense setup helpers."""

from __future__ import annotations

import sys as _sys

from .problems.transport_matrix import active_dense as _impl

_sys.modules[__name__] = _impl
