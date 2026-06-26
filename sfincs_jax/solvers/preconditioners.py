"""Compatibility index for flattened preconditioner modules.

Canonical implementations now live directly under :mod:`sfincs_jax.solvers`
as ``preconditioner_*.py`` files.  This module preserves the former nested
``sfincs_jax.solvers.preconditioners.*`` import paths for one release cycle
while keeping the source tree one level deep.
"""

from __future__ import annotations

from importlib import import_module
import sys
from types import ModuleType
from typing import Iterable

__path__: list[str] = []


def _canonical(module_name: str) -> ModuleType:
    return import_module(f".{module_name}", __package__)


def _public_names(module: ModuleType) -> Iterable[str]:
    exported = getattr(module, "__all__", None)
    if exported is not None:
        return tuple(str(name) for name in exported)
    return tuple(name for name in vars(module) if not name.startswith("_"))


def _register_direct(legacy_name: str, canonical_name: str) -> ModuleType:
    module = _canonical(canonical_name)
    sys.modules[f"{__name__}.{legacy_name}"] = module
    setattr(sys.modules[__name__], legacy_name, module)
    return module


def _register_family(
    family_name: str,
    aliases: dict[str, str],
) -> ModuleType:
    family = ModuleType(f"{__name__}.{family_name}")
    family.__path__ = []
    family.__package__ = family.__name__

    public: list[str] = []
    for legacy_leaf, canonical_name in aliases.items():
        module = _canonical(canonical_name)
        sys.modules[f"{family.__name__}.{legacy_leaf}"] = module
        setattr(family, legacy_leaf, module)
        public.append(legacy_leaf)
        for name in _public_names(module):
            if not hasattr(family, name):
                setattr(family, name, getattr(module, name))
                public.append(name)

    family.__all__ = tuple(dict.fromkeys(public))
    sys.modules[family.__name__] = family
    setattr(sys.modules[__name__], family_name, family)
    return family


dispatch = _register_direct("dispatch", "preconditioner_dispatch")
transport_matrix = _register_direct("transport_matrix", "preconditioner_transport_matrix")
domain_decomposition = _register_direct(
    "domain_decomposition",
    "preconditioner_domain_decomposition",
)

full_fp = _register_family(
    "full_fp",
    {
        "full_csr_kinetic": "preconditioner_full_fp_csr",
        "kinetic_blocks": "preconditioner_full_fp_kinetic",
        "species_blocks": "preconditioner_full_fp_species",
        "structured_fblock": "preconditioner_full_fp_structured",
    },
)

pas = _register_family(
    "pas",
    {
        "angular": "preconditioner_pas_angular",
        "composite": "preconditioner_pas_composite",
        "matrix_free": "preconditioner_pas_matrix_free",
        "policy": "preconditioner_pas_policy",
        "xblock_ilu": "preconditioner_pas_xblock_ilu",
    },
)

qi = _register_family(
    "qi",
    {
        "basis": "preconditioner_qi_basis",
        "corrections": "preconditioner_qi_corrections",
        "device": "preconditioner_qi_device",
        "policy": "preconditioner_qi_policy",
    },
)

schur = _register_family(
    "schur",
    {
        "profile_response": "preconditioner_schur_profile",
    },
)

symbolic_sparse = _register_family(
    "symbolic_sparse",
    {
        "active_factors": "preconditioner_symbolic_active",
        "host_factor": "preconditioner_symbolic_host",
        "policy": "preconditioner_symbolic_policy",
        "profile_response": "preconditioner_symbolic_profile",
    },
)

xblock = _register_family(
    "xblock",
    {
        "active_projected": "preconditioner_xblock_active",
        "block_jacobi": "preconditioner_xblock_block_jacobi",
        "coarse": "preconditioner_xblock_coarse",
        "low_l_schur": "preconditioner_xblock_low_l_schur",
        "policy": "preconditioner_xblock_policy",
        "radial": "preconditioner_xblock_radial",
        "tz_sparse": "preconditioner_xblock_tz_sparse",
    },
)

__all__ = (
    "dispatch",
    "domain_decomposition",
    "full_fp",
    "pas",
    "qi",
    "schur",
    "symbolic_sparse",
    "transport_matrix",
    "xblock",
)
