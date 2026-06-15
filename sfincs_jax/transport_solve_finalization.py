"""Compatibility shim for transport-matrix RHS finalization helpers."""

from __future__ import annotations

import sys as _sys

from .problems.transport_matrix import finalize as _impl

_sys.modules[__name__] = _impl
