"""Compatibility shim for transport-matrix setup helpers."""

from __future__ import annotations

import sys as _sys

from .problems.transport_matrix import setup as _impl

_sys.modules[__name__] = _impl
