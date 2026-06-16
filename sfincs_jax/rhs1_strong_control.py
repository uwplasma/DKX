"""Compatibility alias for :mod:`sfincs_jax.problems.profile_response.strong_preconditioning`."""

from __future__ import annotations

import sys as _sys

from .problems.profile_response import strong_preconditioning as _impl

_sys.modules[__name__] = _impl
