"""Compatibility aliases for the former ``problems.profile_response`` package.

The RHSMode-1 profile-response implementation now lives in flat
``sfincs_jax.problems.profile_*`` modules.  This shim keeps legacy imports
working for one release cycle without reintroducing nested source directories.
New code should import the canonical modules directly.
"""

from __future__ import annotations

from importlib import import_module
import sys
from types import ModuleType

__path__: list[str] = []

_DIRECT_ALIASES = {
    "dense": "profile_dense",
    "diagnostics": "profile_diagnostics",
    "phi1_newton": "profile_phi1_newton",
    "policies": "profile_policies",
    "preconditioner_build": "profile_preconditioner_build",
    "residual": "profile_residual",
    "setup": "profile_setup",
    "solve": "profile_solve",
    "solver_diagnostics": "profile_solver_diagnostics",
}

_SPARSE_ALIASES = {
    "direct": "profile_sparse_direct",
    "finalization": "profile_sparse_finalization",
    "fortran_reduced": "profile_sparse_fortran_reduced",
    "handoff": "profile_sparse_handoff",
    "policy": "profile_sparse_policy",
    "qi": "profile_sparse_qi",
    "xblock": "profile_sparse_xblock",
}


def _canonical_module(canonical: str) -> ModuleType:
    return import_module(f"sfincs_jax.problems.{canonical}")


def _install_alias(legacy: str, canonical: str) -> ModuleType:
    module = _canonical_module(canonical)
    sys.modules[f"{__name__}.{legacy}"] = module
    return module


for _legacy, _canonical in _DIRECT_ALIASES.items():
    globals()[_legacy] = _install_alias(_legacy, _canonical)

sparse = ModuleType(f"{__name__}.sparse")
sparse.__path__ = []  # type: ignore[attr-defined]
sparse.__all__ = tuple(_SPARSE_ALIASES)
sys.modules[sparse.__name__] = sparse

for _legacy, _canonical in _SPARSE_ALIASES.items():
    _module = _install_alias(f"sparse.{_legacy}", _canonical)
    setattr(sparse, _legacy, _module)

__all__ = (*_DIRECT_ALIASES, "sparse")

del _legacy, _canonical, _module
