"""Compatibility alias for :mod:`sfincs_jax.problems.profile_response.solver_diagnostics`."""

from __future__ import annotations

import sys as _sys

from .problems.profile_response import solver_diagnostics as _impl

_sys.modules[__name__] = _impl
