from __future__ import annotations

import json
import importlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "sfincs_jax"
EXPECTED_TREE = REPO_ROOT / "tests" / "fixtures" / "source_tree_expected.json"
PACKAGE_README = PACKAGE_ROOT / "README.md"


def _expected_tree() -> dict[str, list[str]]:
    with EXPECTED_TREE.open(encoding="utf-8") as stream:
        return json.load(stream)


def _package_dirs() -> list[Path]:
    return sorted(
        path
        for path in PACKAGE_ROOT.rglob("*")
        if path.is_dir() and path.name != "__pycache__"
    )


def _relative_dir(path: Path) -> str:
    return path.relative_to(PACKAGE_ROOT).as_posix()


def test_source_tree_does_not_gain_new_root_modules_or_packages() -> None:
    expected = _expected_tree()

    root_modules = sorted(path.name for path in PACKAGE_ROOT.glob("*.py"))
    root_packages = sorted(
        path.name
        for path in PACKAGE_ROOT.iterdir()
        if path.is_dir() and path.name != "__pycache__"
    )

    assert root_modules == expected["allowed_root_modules"]
    assert root_packages == expected["allowed_root_packages"]


def test_source_tree_nested_packages_are_explicit_refactor_debt() -> None:
    expected = _expected_tree()

    nested_packages = sorted(
        _relative_dir(path)
        for path in _package_dirs()
        if len(path.relative_to(PACKAGE_ROOT).parts) > 1
    )

    assert nested_packages == expected["temporary_nested_packages"]


def test_source_tree_init_only_packages_are_explicit_refactor_debt() -> None:
    expected = _expected_tree()

    init_only_packages: list[str] = []
    for path in _package_dirs():
        files = sorted(child.name for child in path.iterdir() if child.is_file())
        dirs = sorted(
            child.name
            for child in path.iterdir()
            if child.is_dir() and child.name != "__pycache__"
        )
        if set(files) <= {"__init__.py"} and len(dirs) <= 1:
            init_only_packages.append(_relative_dir(path))

    assert init_only_packages == expected["temporary_init_only_packages"]


def test_source_tree_consolidation_target_is_stricter_than_current_tree() -> None:
    expected = _expected_tree()

    assert set(expected["target_root_modules"]) < set(expected["allowed_root_modules"])
    assert set(expected["target_root_packages"]) <= set(expected["allowed_root_packages"])
    assert expected["temporary_nested_packages"] == []
    assert expected["temporary_init_only_packages"] == []


def test_package_readme_describes_current_source_layout() -> None:
    expected = _expected_tree()
    text = PACKAGE_README.read_text(encoding="utf-8")

    assert "one-level domain structure" in text
    for package in expected["allowed_root_packages"]:
        assert f"`{package}/`" in text
    for module in expected["target_root_modules"]:
        assert f"`{module}`" in text or f"`{module.removesuffix('.py')}`" in text
    for module in sorted(set(expected["allowed_root_modules"]) - set(expected["target_root_modules"])):
        assert f"`{module}`" in text


def test_flattened_operator_legacy_imports_resolve_to_canonical_modules() -> None:
    assert not (PACKAGE_ROOT / "operators" / "profile_response").exists()

    for name in ("collisionless", "fblock", "full_system", "layout", "system"):
        legacy = importlib.import_module(f"sfincs_jax.operators.profile_response.{name}")
        canonical = importlib.import_module(f"sfincs_jax.operators.profile_{name}")
        assert legacy is canonical


def test_flattened_profile_problem_legacy_imports_resolve_to_canonical_modules() -> None:
    assert not (PACKAGE_ROOT / "problems" / "profile_response").exists()

    for name in ("solve", "policies", "residual", "dense", "solver_diagnostics"):
        legacy = importlib.import_module(f"sfincs_jax.problems.profile_response.{name}")
        canonical = importlib.import_module(f"sfincs_jax.problems.profile_{name}")
        assert legacy is canonical

    for name in ("direct", "finalization", "fortran_reduced", "handoff", "policy", "qi", "xblock"):
        legacy = importlib.import_module(f"sfincs_jax.problems.profile_response.sparse.{name}")
        canonical = importlib.import_module(f"sfincs_jax.problems.profile_sparse_{name}")
        assert legacy is canonical


def test_flattened_transport_problem_legacy_imports_resolve_to_canonical_modules() -> None:
    assert not (PACKAGE_ROOT / "problems" / "transport_matrix").exists()

    for name in ("diagnostics", "finalize", "linear_system", "policies", "setup", "solve"):
        legacy = importlib.import_module(f"sfincs_jax.problems.transport_matrix.{name}")
        canonical = importlib.import_module(f"sfincs_jax.problems.transport_{name}")
        assert legacy is canonical

    for name in ("runtime", "worker"):
        legacy = importlib.import_module(f"sfincs_jax.problems.transport_matrix.parallel.{name}")
        canonical = importlib.import_module(f"sfincs_jax.problems.transport_parallel_{name}")
        assert legacy is canonical


def test_flattened_preconditioner_legacy_imports_resolve_to_canonical_modules() -> None:
    assert not (PACKAGE_ROOT / "solvers" / "preconditioners").exists()

    aliases = {
        "dispatch": "preconditioner_dispatch",
        "transport_matrix": "preconditioner_transport_matrix",
        "domain_decomposition": "preconditioner_domain_decomposition",
        "full_fp.full_csr_kinetic": "preconditioner_full_fp_csr",
        "full_fp.kinetic_blocks": "preconditioner_full_fp_kinetic",
        "full_fp.species_blocks": "preconditioner_full_fp_species",
        "full_fp.structured_fblock": "preconditioner_full_fp_structured",
        "pas.angular": "preconditioner_pas_angular",
        "pas.composite": "preconditioner_pas_composite",
        "pas.matrix_free": "preconditioner_pas_matrix_free",
        "pas.policy": "preconditioner_pas_policy",
        "pas.xblock_ilu": "preconditioner_pas_xblock_ilu",
        "qi.basis": "preconditioner_qi_basis",
        "qi.corrections": "preconditioner_qi_corrections",
        "qi.device": "preconditioner_qi_device",
        "qi.policy": "preconditioner_qi_policy",
        "schur.profile_response": "preconditioner_schur_profile",
        "symbolic_sparse.active_factors": "preconditioner_symbolic_active",
        "symbolic_sparse.host_factor": "preconditioner_symbolic_host",
        "symbolic_sparse.policy": "preconditioner_symbolic_policy",
        "symbolic_sparse.profile_response": "preconditioner_symbolic_profile",
        "xblock.active_projected": "preconditioner_xblock_active",
        "xblock.block_jacobi": "preconditioner_xblock_block_jacobi",
        "xblock.coarse": "preconditioner_xblock_coarse",
        "xblock.low_l_schur": "preconditioner_xblock_low_l_schur",
        "xblock.policy": "preconditioner_xblock_policy",
        "xblock.radial": "preconditioner_xblock_radial",
        "xblock.tz_sparse": "preconditioner_xblock_tz_sparse",
    }
    for legacy_suffix, canonical_suffix in aliases.items():
        legacy = importlib.import_module(f"sfincs_jax.solvers.preconditioners.{legacy_suffix}")
        canonical = importlib.import_module(f"sfincs_jax.solvers.{canonical_suffix}")
        assert legacy is canonical
