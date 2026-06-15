"""Compatibility alias for transport handoff-policy helpers."""

from __future__ import annotations

import sys as _sys

from .problems.transport_matrix import handoff_policy as _impl

_sys.modules[__name__] = _impl
