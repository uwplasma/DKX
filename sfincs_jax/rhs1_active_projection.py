"""Compatibility alias for :mod:`sfincs_jax.problems.profile_response.active_projection`."""

from __future__ import annotations

import sys as _sys

from .problems.profile_response import active_projection as _impl

_sys.modules[__name__] = _impl
