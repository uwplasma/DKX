from __future__ import annotations

import ast
import json
import importlib
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "sfincs_jax"
SCRIPT_ROOT = REPO_ROOT / "scripts"
EXPECTED_TREE = REPO_ROOT / "tests" / "fixtures" / "source_tree_expected.json"
CORE_SLIM_INVENTORY = REPO_ROOT / "tests" / "fixtures" / "core_slim_inventory.json"
PACKAGE_README = PACKAGE_ROOT / "README.md"
SOURCE_MAP_DOC = REPO_ROOT / "docs" / "source_map.rst"
ACTIVE_PLAN = REPO_ROOT / "plan_final.md"
EXECUTION_LOG = REPO_ROOT / "plan.md"
ALLOWED_ROOT_PLAN_FILES = ("plan.md", "plan_final.md")
ACTIVE_PLAN_MAX_LINES = 420
ACTIVE_PLAN_REQUIRED_SECTIONS = (
    "## One-Sentence Goal",
    "## Current Review State",
    "## Open Lanes",
    "## Source Structure Rules",
    "## Ordered Finish Plan",
    "## Concrete Code-Audit Rules",
    "## Standard Validation Commands",
    "## Completion Gates",
    "## Explicit Deferred Items",
)
PACKAGE_README_REQUIRED_SECTIONS = (
    "## The Canonical Stack (the architecture)",
    "## Other Root Modules",
    "## Remaining Domain Packages",
    "## Design Rules",
    "## Stability And Compatibility",
    "## Generated Files Policy",
    "## Contributor Workflow",
)
DISALLOWED_TRACKED_PACKAGE_PARTS = {
    "__pycache__",
    ".ipynb_checkpoints",
}
DISALLOWED_TRACKED_PACKAGE_SUFFIXES = {
    ".h5",
    ".hdf5",
    ".nc",
    ".netcdf",
    ".npy",
    ".npz",
    ".pb",
    ".prof",
    ".pyc",
    ".pyo",
}
INVENTORY_CATEGORIES = {
    "core",
    "compat",
    "test-fixture",
    "extract-pr",
    "delete",
}
INVENTORY_ACTIONS = {
    "retain",
    "trim",
    "split",
    "extract",
    "delete",
    "promote",
}
INVENTORY_DECISIONS = {
    "keep",
    "merge",
    "delete",
    "extract-pr",
}
REQUIRED_CORE_SLIM_SOURCE_OWNERS = {
    "sfincs_jax/drift_kinetic.py",
    "sfincs_jax/solve.py",
    "sfincs_jax/writer.py",
    "sfincs_jax/magnetic_geometry.py",
    "sfincs_jax/moments.py",
    "sfincs_jax/validation/artifacts.py",
    "sfincs_jax/validation/release.py",
}
REQUIRED_CORE_SLIM_NONPACKAGE_OWNERS = {
    "examples",
    "tests",
    "scripts",
}
REQUIRED_RESEARCH_BRANCHES = {
    "research/parallel-performance",
    "research/publication-audits",
}


def _expected_tree() -> dict[str, list[str]]:
    with EXPECTED_TREE.open(encoding="utf-8") as stream:
        return json.load(stream)


def _core_slim_inventory() -> dict[str, object]:
    with CORE_SLIM_INVENTORY.open(encoding="utf-8") as stream:
        return json.load(stream)


def _package_dirs() -> list[Path]:
    return sorted(
        path
        for path in PACKAGE_ROOT.rglob("*")
        if path.is_dir() and path.name != "__pycache__"
    )


def _relative_dir(path: Path) -> str:
    return path.relative_to(PACKAGE_ROOT).as_posix()


def _tracked_paths() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return sorted(line for line in result.stdout.splitlines() if line)


def _inventory_rule_matches(path: str, rule: dict[str, object]) -> bool:
    for excluded in rule.get("exclude_prefix_any", []):
        if path.startswith(str(excluded)):
            return False

    exact_paths = {str(candidate) for candidate in rule.get("exact_paths", [])}
    if exact_paths:
        return path in exact_paths

    if "prefix_any" in rule:
        return any(path.startswith(str(prefix)) for prefix in rule["prefix_any"])

    if "prefix" in rule:
        return path.startswith(str(rule["prefix"]))

    return False


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


def test_domain_packages_do_not_gain_unplanned_modules() -> None:
    """Keep domain packages from drifting back into many ad-hoc helper files."""

    expected = _expected_tree()
    actual = {
        package: sorted(
            path.name
            for path in (PACKAGE_ROOT / package).glob("*.py")
            if path.name != "__init__.py"
        )
        for package in expected["allowed_root_packages"]
    }

    assert actual == expected["allowed_domain_modules"]


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


def test_source_tree_consolidation_target_matches_current_tree() -> None:
    expected = _expected_tree()

    assert expected["target_root_modules"] == expected["allowed_root_modules"]
    assert expected["target_root_packages"] == expected["allowed_root_packages"]
    assert expected["temporary_nested_packages"] == []
    assert expected["temporary_init_only_packages"] == []


def test_plan_final_is_the_single_authoritative_plan() -> None:
    """Prevent competing root-level plans from drifting into PR review."""

    root_plan_files = sorted(path.name for path in REPO_ROOT.glob("*plan*.md"))
    assert root_plan_files == list(ALLOWED_ROOT_PLAN_FILES)

    active_text = ACTIVE_PLAN.read_text(encoding="utf-8")
    assert "single active plan" in active_text
    assert "`plan.md` is the historical execution log" in active_text
    assert "Do not create another competing plan" in active_text
    assert len(active_text.splitlines()) <= ACTIVE_PLAN_MAX_LINES
    for section in ACTIVE_PLAN_REQUIRED_SECTIONS:
        assert section in active_text

    log_text = EXECUTION_LOG.read_text(encoding="utf-8")
    assert "Historical log only" in log_text
    assert "Authoritative plan: `plan_final.md`" in log_text
    assert "if this file conflicts with `plan_final.md`, follow" in log_text
    assert "active implementation branch" not in log_text.lower()


def test_core_slim_inventory_covers_large_phase_a_owners() -> None:
    """Make Phase A actionable before any broad source deletion happens."""

    payload = _core_slim_inventory()
    assert payload["schema_version"] == 1
    assert set(payload["categories"]) == INVENTORY_CATEGORIES
    assert set(payload["actions"]) == INVENTORY_ACTIONS

    groups = payload["extraction_groups"]
    assert isinstance(groups, list)
    group_branches: set[str] = set()
    for group in groups:
        branch = str(group["branch"])
        assert branch in REQUIRED_RESEARCH_BRANCHES
        group_branches.add(branch)
        assert len(str(group["purpose"])) >= 40
        assert len(str(group["core_removal_gate"])) >= 40
        extracted_paths = {str(path) for path in group.get("core_extracted_paths", [])}
        for key in (
            "source_paths",
            "test_paths",
            "example_paths",
            "script_paths",
            "core_coupling_paths",
            "doc_paths",
        ):
            if key not in group:
                continue
            assert isinstance(group[key], list)
            for referenced in group[key]:
                referenced_path = REPO_ROOT / str(referenced)
                if (
                    (bool(group.get("core_extracted", False)) or str(referenced) in extracted_paths)
                    and key in {"source_paths", "test_paths", "example_paths", "script_paths"}
                    and not referenced_path.exists()
                ):
                    continue
                assert referenced_path.exists(), f"{branch} references missing {referenced}"
        grouped_paths = {
            str(path)
            for key in ("source_paths", "test_paths", "example_paths", "script_paths")
            for path in group.get(key, [])
        }
        assert extracted_paths <= grouped_paths
    assert REQUIRED_RESEARCH_BRANCHES <= group_branches

    entries = payload["entries"]
    assert isinstance(entries, list)
    assert entries

    by_path: dict[str, dict[str, object]] = {}
    extract_branches: set[str] = set()
    for entry in entries:
        path = str(entry["path"])
        assert path not in by_path
        by_path[path] = entry

        category = str(entry["category"])
        action = str(entry["action"])
        assert category in INVENTORY_CATEGORIES
        assert action in INVENTORY_ACTIONS
        assert isinstance(entry["lines_at_audit"], int)
        assert int(entry["lines_at_audit"]) >= 0
        assert isinstance(entry["public_imports"], list)
        assert isinstance(entry["internal_callers"], list)
        assert isinstance(entry["tests"], list)
        assert isinstance(entry["docs_examples"], list)
        assert len(str(entry["retention_reason"])) >= 40
        assert len(str(entry["next_step"])) >= 20

        repo_path = REPO_ROOT / path
        if category in {"core", "compat", "test-fixture"}:
            assert repo_path.exists(), path
        if repo_path.is_file() and repo_path.suffix == ".py":
            current_lines = len(repo_path.read_text(encoding="utf-8").splitlines())
            assert current_lines <= int(entry["lines_at_audit"]), path
        for key in ("tests", "docs_examples", "internal_callers"):
            for referenced in entry[key]:
                referenced_path = REPO_ROOT / str(referenced)
                if category == "extract-pr" and not referenced_path.exists():
                    continue
                assert referenced_path.exists(), f"{path} references missing {referenced}"
        if category == "extract-pr":
            branch = str(entry.get("extract_branch", ""))
            assert branch in REQUIRED_RESEARCH_BRANCHES
            extract_branches.add(branch)

    assert REQUIRED_CORE_SLIM_SOURCE_OWNERS <= set(by_path)
    assert REQUIRED_CORE_SLIM_NONPACKAGE_OWNERS <= set(by_path)
    assert REQUIRED_RESEARCH_BRANCHES <= extract_branches


def test_core_slim_inventory_classifies_every_tracked_file() -> None:
    """Tranche A must cover the whole repo without a huge per-file JSON dump."""

    payload = _core_slim_inventory()
    review = payload["whole_repo_review"]
    assert review["coverage_model"] == "exactly_one_rule_per_git_tracked_path"
    assert review["review_status"] == "first_pass_file_complete"
    assert set(review["decisions"]) == INVENTORY_DECISIONS

    rules = review["rules"]
    assert isinstance(rules, list)
    assert len(rules) >= 12
    seen_rule_names: set[str] = set()
    for rule in rules:
        name = str(rule["name"])
        assert name not in seen_rule_names
        seen_rule_names.add(name)
        assert str(rule["decision"]) in INVENTORY_DECISIONS
        assert isinstance(rule["owner_tags"], list) and rule["owner_tags"]
        assert len(str(rule["proof_owner"])) >= 20
        assert len(str(rule["line_target"])) >= 20
        assert len(str(rule["next_action"])) >= 30

    unmatched: list[str] = []
    ambiguous: dict[str, list[str]] = {}
    for path in _tracked_paths():
        matches = [str(rule["name"]) for rule in rules if _inventory_rule_matches(path, rule)]
        if not matches:
            unmatched.append(path)
        elif len(matches) > 1:
            ambiguous[path] = matches

    assert unmatched == []
    assert ambiguous == {}


def test_package_readme_describes_current_source_layout() -> None:
    expected = _expected_tree()
    text = PACKAGE_README.read_text(encoding="utf-8")

    assert "flat, physics-named root modules" in text
    assert "explicitly transitional" in text
    for section in PACKAGE_README_REQUIRED_SECTIONS:
        assert section in text
    for package in expected["allowed_root_packages"]:
        assert f"`{package}/`" in text
    for module in expected["target_root_modules"]:
        assert f"`{module}`" in text or f"`{module.removesuffix('.py')}`" in text
    canonical_phrases = (
        "`drift_kinetic.py` | The `KineticOperator`",
        "frozen-reference loading, Fortran/PETSc fixture readers",
        "must not be reintroduced",
    )
    for phrase in canonical_phrases:
        assert phrase in text


def test_package_readme_explains_public_surface_and_implementation_boundaries() -> None:
    """The source README should be enough to navigate the package during review."""

    text = PACKAGE_README.read_text(encoding="utf-8")
    expected_phrases = (
        "canonical stack of flat, physics-named root modules",
        "transitional interim owners while the vertical slices landed",
        "one folder below `sfincs_jax/`, no nested",
        "canonical root modules are the stable import surface",
        "Compatibility aliases may remain",
        "fetched through `validation.data_fetch` from release assets",
    )

    for phrase in expected_phrases:
        assert phrase in text


def test_canonical_writer_stays_below_review_size_budget() -> None:
    """Keep output-write orchestration from drifting back into a monolith."""

    writer_path = PACKAGE_ROOT / "writer.py"
    source = writer_path.read_text(encoding="utf-8")
    assert len(source.splitlines()) <= 2300


def test_package_tree_has_no_tracked_generated_or_large_runtime_outputs() -> None:
    """Keep the importable package light and independent of local run artifacts."""

    result = subprocess.run(
        ["git", "ls-files", "sfincs_jax"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    offenders: list[str] = []
    for line in result.stdout.splitlines():
        path = Path(line)
        if DISALLOWED_TRACKED_PACKAGE_PARTS.intersection(path.parts):
            offenders.append(line)
            continue
        if path.suffix in DISALLOWED_TRACKED_PACKAGE_SUFFIXES:
            offenders.append(line)

    assert offenders == []


def test_source_map_doc_describes_current_one_level_layout() -> None:
    """Keep contributor docs synchronized with the flattened package tree."""

    expected = _expected_tree()
    text = SOURCE_MAP_DOC.read_text(encoding="utf-8")

    assert "one level of domain folders" in text
    for package in expected["allowed_root_packages"]:
        assert f"``sfincs_jax/{package}``" in text


def test_source_map_doc_does_not_teach_deleted_file_history() -> None:
    """Keep the active source map focused on canonical owners."""

    text = SOURCE_MAP_DOC.read_text(encoding="utf-8")
    stale_fragments = (
        "historical location",
        "historical locations",
        "former flat",
        "old ``rhs1_",
        "v3_driver.py",
    )

    for fragment in stale_fragments:
        assert fragment not in text

    removed_packages = {
        "benchmarks",
        "compat",
        "input",
        "parallel",
    }
    offenders = [
        package
        for package in sorted(removed_packages)
        if f"``sfincs_jax/{package}``" in text or f"`sfincs_jax/{package}`" in text
    ]
    assert offenders == []


def test_package_sources_do_not_import_deleted_v3_driver() -> None:
    """Keep source imports on canonical problem owners, not deleted driver aliases."""

    offenders: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if path.name == "v3_driver.py" or "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "sfincs_jax.v3_driver":
                        offenders.append(path.relative_to(REPO_ROOT).as_posix())
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module in {"sfincs_jax.v3_driver", "v3_driver"}:
                    offenders.append(path.relative_to(REPO_ROOT).as_posix())

    assert offenders == []


def test_scripts_do_not_import_missing_sibling_modules() -> None:
    """Keep temporary scripts executable while they are promoted or deleted."""

    if not SCRIPT_ROOT.exists():
        return

    available_modules = {path.stem for path in SCRIPT_ROOT.glob("*.py")}
    offenders: list[tuple[str, str]] = []
    for path in sorted(SCRIPT_ROOT.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_name = alias.name.split(".", 1)[0]
                    if root_name in available_modules or root_name.startswith("sfincs_"):
                        continue
                    candidate = SCRIPT_ROOT / f"{root_name}.py"
                    if candidate.parent == SCRIPT_ROOT and root_name.startswith(("audit_", "run_", "generate_")):
                        offenders.append((path.relative_to(REPO_ROOT).as_posix(), root_name))
            elif isinstance(node, ast.ImportFrom):
                if node.level != 0 or node.module is None:
                    continue
                root_name = node.module.split(".", 1)[0]
                if root_name in available_modules or root_name.startswith("sfincs_"):
                    continue
                if root_name.startswith(("audit_", "run_", "generate_")):
                    offenders.append((path.relative_to(REPO_ROOT).as_posix(), root_name))

    assert offenders == []


def test_package_sources_do_not_document_deleted_v3_driver_as_architecture() -> None:
    """Keep production source prose centered on canonical domain owners."""

    offenders = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in sorted(PACKAGE_ROOT.rglob("*.py"))
        if "__pycache__" not in path.parts and "v3_driver" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


def test_package_sources_do_not_use_monolith_state_terms() -> None:
    """Keep active sparse-solve APIs named around solve state, not driver state."""

    forbidden_terms = (
        "driver_state",
        "driver_scope",
        "driver-state",
        "driver-scope",
        "from_driver_state",
        "from_driver_scope",
    )
    offenders: list[tuple[str, str]] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        for term in forbidden_terms:
            if term in text:
                offenders.append((path.relative_to(REPO_ROOT).as_posix(), term))

    assert offenders == []


def test_package_sources_do_not_repeat_top_level_defs() -> None:
    """Repeated top-level definitions hide stale helpers in large owner files."""

    offenders: list[tuple[str, str, int, int]] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        seen: dict[str, int] = {}
        for node in tree.body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                if node.name in seen:
                    offenders.append(
                        (
                            path.relative_to(REPO_ROOT).as_posix(),
                            node.name,
                            seen[node.name],
                            node.lineno,
                        )
                    )
                seen[node.name] = node.lineno

    assert offenders == []


def test_sparse_helper_tests_are_deleted_with_their_owners() -> None:
    """The sparse-helper suites were deleted together with the sparse families."""

    assert not (REPO_ROOT / "tests" / "test_profile_sparse_helper_coverage.py").exists()


def test_sparse_assembly_tests_are_deleted_with_their_owners() -> None:
    """Sparse assembly tests were deleted with preconditioner_xblock_tz_sparse."""

    assert not (REPO_ROOT / "tests" / "test_sparse_assembly.py").exists()


def test_device_operator_tests_are_deleted_with_their_owners() -> None:
    """Device-operator tests were deleted with operators.profile_device_sparse."""

    assert not (REPO_ROOT / "tests" / "test_rhs1_device_operator.py").exists()


def test_structured_csr_docs_tests_are_deleted_with_their_owners() -> None:
    """The structured-CSR documentation contract died with the CSR lanes."""

    assert not (REPO_ROOT / "tests" / "test_structured_csr_docs.py").exists()


def test_rhs1_dispatch_coverage_tests_are_deleted_with_their_owners() -> None:
    """The RHSMode-1 sparse dispatch suites were deleted with the sparse families."""

    assert not (REPO_ROOT / "tests" / "test_profile_rhs1_dispatch_coverage.py").exists()


def test_test_filenames_do_not_reintroduce_deleted_v3_driver_label() -> None:
    """Keep test modules named after the canonical behavior they protect."""

    offenders = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in sorted((REPO_ROOT / "tests").glob("*v3_driver*"))
    ]

    assert offenders == []


def test_test_suite_does_not_import_deleted_v3_driver() -> None:
    """Keep behavior tests on domain modules instead of deleted driver aliases."""

    allowed: set[str] = set()

    offenders: list[str] = []
    for path in sorted((REPO_ROOT / "tests").glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        rel = path.relative_to(REPO_ROOT).as_posix()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(alias.name == "sfincs_jax.v3_driver" for alias in node.names):
                    offenders.append(rel)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imports_driver = module == "sfincs_jax.v3_driver" or (
                    module == "sfincs_jax" and any(alias.name == "v3_driver" for alias in node.names)
                )
                if imports_driver:
                    offenders.append(rel)

    assert sorted(set(offenders)) == sorted(allowed)


def test_deleted_nonroot_compatibility_facades_are_absent() -> None:
    """The flat source tree should not keep package-like one-file facades."""

    deleted_facades = (
        PACKAGE_ROOT / "v3_driver.py",
        PACKAGE_ROOT / "operators" / "profile_response.py",
        PACKAGE_ROOT / "problems" / "profile_response.py",
        PACKAGE_ROOT / "problems" / "transport_matrix.py",
        PACKAGE_ROOT / "solvers" / "preconditioners.py",
    )
    for path in deleted_facades:
        assert not path.exists(), path


def test_deleted_campaign_specific_workflow_module_is_absent() -> None:
    """QI campaign gates are preserved on the QI research branch, not core."""

    assert not (PACKAGE_ROOT / "workflows" / "qi_res15_gpu_campaign.py").exists()
    assert not (PACKAGE_ROOT / "validation" / "qi_device.py").exists()


def test_deleted_tiny_validation_facades_are_absent() -> None:
    """Keep Fortran/PETSc fixture readers in one validation owner."""

    assert not (PACKAGE_ROOT / "validation" / "petsc_binary.py").exists()
    module = importlib.import_module("sfincs_jax.validation.fortran")
    assert hasattr(module, "read_petsc_vec")
    assert hasattr(module, "read_petsc_mat_aij")


def test_deleted_sparse_solver_families_are_absent() -> None:
    """The RHSMode-1 sparse solver families were deleted with the legacy sweep."""

    for name in (
        "problems/profile_sparse_replay.py",
        "problems/profile_sparse_solve.py",
        "problems/profile_sparse_xblock.py",
        "problems/profile_sparse_direct.py",
        "problems/profile_sparse_policy.py",
        "problems/profile_sparse_fortran_reduced.py",
        "problems/profile_sparse_finalization.py",
        "operators/profile_full_system.py",
        "operators/profile_sparse_pattern.py",
        "solvers/explicit_sparse.py",
        "solvers/preconditioner_host_sparse.py",
        "solvers/preconditioner_xblock_tz_sparse.py",
        "solvers/preconditioner_reduced_pmat.py",
    ):
        assert not (PACKAGE_ROOT / name).exists(), f"{name} should stay deleted"


def test_deleted_h5_parity_validation_facade_is_absent() -> None:
    """Strict HDF5 parity is part of the public comparison API."""

    assert not (PACKAGE_ROOT / "validation" / "h5_parity.py").exists()
    module = importlib.import_module("sfincs_jax.compare")
    assert hasattr(module, "compare_h5_outputs")
    assert hasattr(module, "H5DatasetParity")


def test_deleted_qi_promotion_policy_solver_facade_is_absent() -> None:
    """QI promotion evidence is extracted from the stable core."""

    assert not (PACKAGE_ROOT / "solvers" / "preconditioner_qi_policy.py").exists()
    assert not (PACKAGE_ROOT / "validation" / "qi_device.py").exists()


def test_canonical_root_modules_are_importable() -> None:
    """The canonical flat root modules replace the deleted legacy packages."""

    canonical_modules = (
        "sfincs_jax.drift_kinetic",
        "sfincs_jax.solve",
        "sfincs_jax.run",
        "sfincs_jax.writer",
        "sfincs_jax.phase_space",
        "sfincs_jax.magnetic_geometry",
        "sfincs_jax.moments",
        "sfincs_jax.collisions",
        "sfincs_jax.species",
        "sfincs_jax.phi1",
        "sfincs_jax.er",
        "sfincs_jax.solver_trace",
        "sfincs_jax.xgrid",
    )
    for module_name in canonical_modules:
        module = importlib.import_module(module_name)
        assert module.__name__ == module_name
