"""Compatibility aliases for the former ``problems.transport_matrix`` package.

Transport-matrix RHSMode-2/3 implementation modules now live in flat
``sfincs_jax.problems.transport_*`` files.  This shim preserves legacy imports
such as ``sfincs_jax.problems.transport_matrix.solve`` and
``sfincs_jax.problems.transport_matrix.parallel.runtime`` without keeping a
nested source directory.  New code should import the flat modules directly.
"""

from __future__ import annotations

from importlib import import_module
import sys
from types import ModuleType

__path__: list[str] = []

_DIRECT_ALIASES = {
    "diagnostics": "transport_diagnostics",
    "finalize": "transport_finalize",
    "linear_system": "transport_linear_system",
    "policies": "transport_policies",
    "setup": "transport_setup",
    "solve": "transport_solve",
}

_PARALLEL_ALIASES = {
    "runtime": "transport_parallel_runtime",
    "worker": "transport_parallel_worker",
}


def _canonical_module(canonical: str) -> ModuleType:
    return import_module(f"sfincs_jax.problems.{canonical}")


def _install_alias(legacy: str, canonical: str) -> ModuleType:
    module = _canonical_module(canonical)
    sys.modules[f"{__name__}.{legacy}"] = module
    return module


for _legacy, _canonical in _DIRECT_ALIASES.items():
    globals()[_legacy] = _install_alias(_legacy, _canonical)

parallel = ModuleType(f"{__name__}.parallel")
parallel.__path__ = []  # type: ignore[attr-defined]
parallel.__all__ = tuple(_PARALLEL_ALIASES)
sys.modules[parallel.__name__] = parallel

for _legacy, _canonical in _PARALLEL_ALIASES.items():
    _module = _install_alias(f"parallel.{_legacy}", _canonical)
    setattr(parallel, _legacy, _module)

__all__ = (*_DIRECT_ALIASES, "parallel")

del _legacy, _canonical, _module
