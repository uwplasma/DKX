"""Compatibility alias for transport host-GMRES helpers."""

from __future__ import annotations

import sys as _sys

from .problems.transport_matrix import host_gmres as _impl

_sys.modules[__name__] = _impl
