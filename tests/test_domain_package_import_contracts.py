"""Import contracts for the canonical package layout.

After the legacy-stack deletion the package keeps exactly two subpackages
(``workflows`` for scan/optimization orchestration, ``validation`` for
Fortran/release tooling) plus a flat set of canonical root modules.  These
tests pin that layout: the retired legacy packages must never be importable
again, the surviving packages expose their documented facades, and every root
module is classified.
"""

from __future__ import annotations

import importlib
from pathlib import Path


DELETED_LEGACY_PACKAGES = (
    "dkx.physics",
    "dkx.discretization",
    "dkx.geometry",
    "dkx.operators",
    "dkx.problems",
    "dkx.solvers",
    "dkx.outputs",
    "dkx.grids",
    "dkx.diagnostics",
    "dkx.workflows.mapped_xgrid",
)

DOMAIN_PACKAGES = (
    "dkx.workflows",
    "dkx.validation",
)

ACTIVE_PACKAGE_EXPORTS = {
    "dkx.workflows": (
        "geometry_adapters",
        "optimization",
    ),
}

# Canonical root modules and their one-line ownership class.
ROOT_MODULE_CLASSIFICATIONS = {
    "__init__.py": "public package facade",
    "__main__.py": "public entry point",
    "ambipolar.py": "public workflow API",
    "api.py": "public API",
    "batch.py": "public workflow API",
    "bounce_averaged.py": "public physics API",
    "cli.py": "public entry point",
    "collisions.py": "stable physics kernel",
    "compare.py": "public validation API",
    "console.py": "stable support utility",
    "constants.py": "stable physics kernel",
    "drift_kinetic.py": "stable operator kernel",
    "er.py": "public workflow API",
    "impurity.py": "public physics API",
    "input_compat.py": "public compatibility API",
    "inputs.py": "public input API",
    "io.py": "public API",
    "magnetic_geometry.py": "stable geometry kernel",
    "moments.py": "stable physics kernel",
    "momentum_correction.py": "stable physics kernel",
    "monoenergetic.py": "public workflow API",
    "namelist.py": "public input API",
    "paths.py": "stable support utility",
    "phase_space.py": "stable discretization kernel",
    "phi1.py": "stable solver kernel",
    "plotting.py": "public plotting API",
    "profiling.py": "stable support utility",
    "run.py": "public API",
    "sensitivity.py": "public differentiation API",
    "shaing_callen.py": "stable physics kernel",
    "solve.py": "stable solver kernel",
    "solver_trace.py": "stable support utility",
    "species.py": "stable physics kernel",
    "variational.py": "stable physics kernel",
    "writer.py": "public API",
    "xgrid.py": "stable discretization kernel",
}


def _import_module(name: str):
    return importlib.import_module(name)


def test_deleted_legacy_packages_have_no_source_in_this_tree() -> None:
    """The retired legacy stack must never be silently reintroduced.

    Checked on the filesystem (not via import) so a stale editable install of
    another checkout cannot mask a reintroduction in this tree.
    """

    root = Path(__file__).resolve().parents[1] / "dkx"
    for module_name in DELETED_LEGACY_PACKAGES:
        rel = module_name.removeprefix("dkx.").replace(".", "/")
        assert not (root / rel).exists(), module_name
        assert not (root / f"{rel}.py").exists(), module_name


def test_domain_packages_are_importable_with_expected_facades() -> None:
    """Surviving domain packages are importable and expose only intentional facades."""

    for module_name in DOMAIN_PACKAGES:
        module = _import_module(module_name)
        assert module.__doc__ is not None, module_name
        assert module.__doc__.strip(), module_name
        assert hasattr(module, "__path__"), module_name
        expected_exports = ACTIVE_PACKAGE_EXPORTS.get(module_name)
        if expected_exports is not None:
            assert module.__all__ == expected_exports, module_name
            for export_name in expected_exports:
                assert hasattr(module, export_name), f"{module_name}.{export_name}"


def test_workflow_optimization_aliases_resolve_to_owner() -> None:
    """Historical optimization_* module names keep resolving to the durable owner."""

    owner = _import_module("dkx.workflows.optimization")
    for name in (
        "optimization_comparison",
        "optimization_evidence",
        "optimization_ladder",
        "optimization_objectives",
        "optimization_promotion",
        "optimization_workflow",
    ):
        module = _import_module(f"dkx.workflows.{name}")
        assert module is owner, name


def test_root_modules_are_explicitly_classified() -> None:
    """Every remaining package-root module has an owner class; no strays."""

    root = Path(__file__).resolve().parents[1] / "dkx"
    actual = {path.name for path in root.glob("*.py")}
    expected = set(ROOT_MODULE_CLASSIFICATIONS)
    assert actual == expected, (
        f"unclassified: {sorted(actual - expected)}; stale: {sorted(expected - actual)}"
    )


def test_github_workflows_do_not_import_deleted_packages() -> None:
    """CI jobs must follow the canonical import contract."""

    repo_root = Path(__file__).resolve().parents[1]
    workflow_dir = repo_root / ".github" / "workflows"
    for path in workflow_dir.glob("*.yml"):
        text = path.read_text(encoding="utf-8")
        for deleted in DELETED_LEGACY_PACKAGES:
            assert deleted not in text, f"{path.relative_to(repo_root)} references {deleted}"
