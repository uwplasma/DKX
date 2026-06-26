"""Compatibility aliases for the former ``operators.profile_response`` package.

The canonical modules now live one level below :mod:`sfincs_jax.operators` with
``profile_*`` names. This shim preserves old public imports for one release
cycle without keeping a nested source folder.
"""

from __future__ import annotations

from importlib import import_module
import sys

_ALIASES = {
    "collisionless": "profile_collisionless",
    "collisions": "profile_collisions",
    "compressed_layout": "profile_compressed_layout",
    "device_sparse": "profile_device_sparse",
    "drifts": "profile_drifts",
    "electric_field": "profile_electric_field",
    "exb": "profile_exb",
    "fblock": "profile_fblock",
    "full_system": "profile_full_system",
    "kinetic": "profile_kinetic",
    "layout": "profile_layout",
    "linear_systems": "profile_linear_systems",
    "magnetic_drifts": "profile_magnetic_drifts",
    "reduced_tail": "profile_reduced_tail",
    "sources": "profile_sources",
    "sparse_pattern": "profile_sparse_pattern",
    "structured_csr": "profile_structured_csr",
    "system": "profile_system",
    "true_operator_rescue": "profile_true_operator_rescue",
}

__path__: list[str] = []
__all__ = tuple(_ALIASES)

for _legacy_name, _canonical_name in _ALIASES.items():
    _module = import_module(f"sfincs_jax.operators.{_canonical_name}")
    globals()[_legacy_name] = _module
    sys.modules[f"{__name__}.{_legacy_name}"] = _module

del _canonical_name, _legacy_name, _module
