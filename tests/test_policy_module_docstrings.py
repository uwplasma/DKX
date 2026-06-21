from __future__ import annotations

import importlib
import inspect
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parents[1] / "sfincs_jax"

POLICY_MODULES = tuple(
    f"sfincs_jax.{path.stem}" for path in sorted(PACKAGE_DIR.glob("*policy*.py"))
)
SOURCE_MAPPED_CONTROL_MODULES = (
    "sfincs_jax.profiling",
    "sfincs_jax.problems.profile_response.handoff",
    "sfincs_jax.problems.profile_response.policies",
    "sfincs_jax.problems.profile_response.strong_preconditioning",
    "sfincs_jax.solvers.preconditioners.pas.xblock_ilu",
    "sfincs_jax.solvers.preconditioners.schur.rhs1_coarse_basis",
    "sfincs_jax.solvers.preconditioners.schur.rhs1_coarse_policy",
    "sfincs_jax.solvers.preconditioners.xblock.tz_sparse",
    "sfincs_jax.transport_parallel_solve",
    "sfincs_jax.transport_postsolve_diagnostics",
)


def _missing_public_api_docstrings(module_names: tuple[str, ...]) -> list[str]:
    missing: list[str] = []
    for module_name in module_names:
        module = importlib.import_module(module_name)
        for name, member in inspect.getmembers(module):
            if name.startswith("_"):
                continue
            if not (inspect.isclass(member) or inspect.isfunction(member)):
                continue
            if getattr(member, "__module__", None) != module.__name__:
                continue
            if not inspect.getdoc(member):
                missing.append(f"{module_name}.{name}")
    return missing


def test_refactored_policy_modules_expose_real_module_docstrings() -> None:
    """Keep every split policy module discoverable in API docs and introspection."""

    for module_name in POLICY_MODULES:
        module = importlib.import_module(module_name)
        assert module.__doc__ is not None, module_name
        assert module.__doc__.strip(), module_name


def test_refactored_policy_public_apis_expose_docstrings() -> None:
    """Keep public policy classes/functions explanatory after extraction."""

    missing = _missing_public_api_docstrings(POLICY_MODULES)

    assert missing == []


def test_source_mapped_driver_control_modules_expose_docstrings() -> None:
    """Keep non-policy driver-control helper modules visible in source docs."""

    for module_name in SOURCE_MAPPED_CONTROL_MODULES:
        module = importlib.import_module(module_name)
        assert module.__doc__ is not None, module_name
        assert module.__doc__.strip(), module_name

    assert _missing_public_api_docstrings(SOURCE_MAPPED_CONTROL_MODULES) == []
