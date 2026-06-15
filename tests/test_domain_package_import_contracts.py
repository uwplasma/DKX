from __future__ import annotations

import importlib
from types import ModuleType


DOMAIN_PACKAGES = (
    "sfincs_jax.input",
    "sfincs_jax.physics",
    "sfincs_jax.discretization",
    "sfincs_jax.operators",
    "sfincs_jax.problems",
    "sfincs_jax.problems.profile_response",
    "sfincs_jax.problems.transport_matrix",
    "sfincs_jax.solvers",
    "sfincs_jax.solvers.preconditioners",
    "sfincs_jax.solvers.preconditioners.pas",
    "sfincs_jax.solvers.preconditioners.full_fp",
    "sfincs_jax.solvers.preconditioners.qi",
    "sfincs_jax.solvers.preconditioners.schur",
    "sfincs_jax.solvers.preconditioners.domain_decomposition",
    "sfincs_jax.solvers.preconditioners.coarse_space",
    "sfincs_jax.solvers.preconditioners.xblock",
    "sfincs_jax.solvers.preconditioners.symbolic_sparse",
    "sfincs_jax.parallel",
    "sfincs_jax.workflows",
    "sfincs_jax.validation",
    "sfincs_jax.benchmarks",
    "sfincs_jax.compat",
)

LEGACY_MODULES_THAT_KEEP_THEIR_IMPORT_PATHS = (
    "sfincs_jax.input_compat",
    "sfincs_jax.namelist",
    "sfincs_jax.geometry",
    "sfincs_jax.io",
    "sfincs_jax.solver",
    "sfincs_jax.v3_driver",
)

RESERVED_MODULE_NAMES_UNTIL_MIGRATION = (
    "sfincs_jax.geometry",
    "sfincs_jax.io",
)


def _import_module(name: str) -> ModuleType:
    return importlib.import_module(name)


def test_domain_package_skeletons_are_importable_packages() -> None:
    """Phase-A package skeletons must be importable without moving behavior."""

    for module_name in DOMAIN_PACKAGES:
        module = _import_module(module_name)
        assert module.__doc__ is not None, module_name
        assert module.__doc__.strip(), module_name
        assert hasattr(module, "__path__"), module_name
        assert module.__all__ == (), module_name


def test_existing_legacy_modules_keep_their_import_paths() -> None:
    """The package skeleton must not break current public/internal imports."""

    for module_name in LEGACY_MODULES_THAT_KEEP_THEIR_IMPORT_PATHS:
        module = _import_module(module_name)
        assert module.__name__ == module_name


def test_module_names_reserved_for_later_package_migration_still_load_as_modules() -> None:
    """Avoid silently shadowing large legacy modules during Phase A."""

    for module_name in RESERVED_MODULE_NAMES_UNTIL_MIGRATION:
        module = _import_module(module_name)
        assert not hasattr(module, "__path__"), module_name
        assert module.__file__ is not None
        assert module.__file__.endswith(".py"), module.__file__
