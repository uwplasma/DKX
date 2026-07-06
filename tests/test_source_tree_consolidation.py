from __future__ import annotations

import ast
import json
import importlib
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "sfincs_jax"
EXPECTED_TREE = REPO_ROOT / "tests" / "fixtures" / "source_tree_expected.json"
PACKAGE_README = PACKAGE_ROOT / "README.md"
SOURCE_MAP_DOC = REPO_ROOT / "docs" / "source_map.rst"
ACTIVE_PLAN = REPO_ROOT / "plan_final.md"
EXECUTION_LOG = REPO_ROOT / "plan.md"
ALLOWED_ROOT_PLAN_FILES = ("plan.md", "plan_final.md")
ACTIVE_PLAN_MAX_LINES = 260
ACTIVE_PLAN_REQUIRED_SECTIONS = (
    "## One-Sentence Goal",
    "## Current Review State",
    "## Open Lanes",
    "## Source Structure Rules",
    "## Ordered Finish Plan",
    "## Standard Validation Commands",
    "## Completion Gates",
    "## Explicit Deferred Items",
)
PACKAGE_README_REQUIRED_SECTIONS = (
    "## Root Modules At A Glance",
    "## Domain Packages At A Glance",
    "## Main Implementation Owners",
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


def test_package_readme_describes_current_source_layout() -> None:
    expected = _expected_tree()
    text = PACKAGE_README.read_text(encoding="utf-8")

    assert "one-level domain structure" in text
    for section in PACKAGE_README_REQUIRED_SECTIONS:
        assert section in text
    for package in expected["allowed_root_packages"]:
        assert f"`{package}/`" in text
    for module in expected["target_root_modules"]:
        assert f"`{module}`" in text or f"`{module.removesuffix('.py')}`" in text
    canonical_owner_phrases = (
        "`operators/profile_system.py`: RHSMode-1 full-system operator",
        "`operators/profile_layout.py`: RHSMode-1 full, active, field-split",
        "`problems/transport_parallel_runtime.py`: RHSMode-2/3 whichRHS",
        "`solvers/preconditioning.py`: shared preconditioner caches",
        "`outputs/formats.py`: HDF5/NetCDF/NPZ schemas",
        "`validation/artifacts.py`: validation artifact manifests",
        "Do not reintroduce helper-only modules",
    )
    for phrase in canonical_owner_phrases:
        assert phrase in text


def test_package_readme_explains_public_surface_and_implementation_boundaries() -> None:
    """The source README should be enough to navigate the package during review."""

    text = PACKAGE_README.read_text(encoding="utf-8")
    expected_phrases = (
        "stable user-facing API",
        "If a new feature is not meant to be imported directly by users",
        "There are no implementation packages nested inside those folders",
        "A folder that contains only `__init__.py`",
        "Public root modules are the stable import surface",
        "Compatibility aliases may remain",
        "Large external data belongs in release-hosted assets",
    )

    for phrase in expected_phrases:
        assert phrase in text


def test_output_writer_stays_below_review_size_budget() -> None:
    """Keep output-write orchestration from drifting back into a monolith."""

    writer_path = PACKAGE_ROOT / "outputs" / "writer.py"
    source = writer_path.read_text(encoding="utf-8")
    line_count = len(source.splitlines())
    assert line_count <= 2300

    tree = ast.parse(source, filename=str(writer_path))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    writer_fn = functions["write_sfincs_jax_output_h5"]
    assert writer_fn.end_lineno is not None
    assert writer_fn.end_lineno - writer_fn.lineno + 1 <= 1500


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


def test_profile_solve_does_not_reexport_low_level_helper_namespaces() -> None:
    """Keep RHSMode-1 orchestration from becoming a private-helper namespace."""

    text = (PACKAGE_ROOT / "problems" / "profile_solve.py").read_text(encoding="utf-8")
    forbidden_fragments = (
        "_dd_core_patch_ranges",
        "_rhs1_dd_auto_block_size",
        "_rhs1_dd_coarse_block_size",
        "_rhs1_dd_coarse_block_sizes",
        "_rhs1_dd_coarse_level_count",
        "block_diagonal_only as _block_diag_only",
        "diagonal_only as _diag_only",
    )

    for fragment in forbidden_fragments:
        assert fragment not in text


def test_policy_tests_import_policy_owners_not_profile_solve() -> None:
    """Policy-only tests should not widen the solve-orchestration API."""

    policy_test_paths = (
        REPO_ROOT / "tests" / "test_profile_solve_policy_helpers.py",
        REPO_ROOT / "tests" / "test_profile_solve_policy_coverage.py",
    )

    for path in policy_test_paths:
        assert "profile_solve" not in path.read_text(encoding="utf-8"), path


def test_sparse_helper_tests_do_not_use_profile_solve_private_aliases() -> None:
    """Sparse-helper tests may run solves, but helpers should use canonical owners."""

    text = (REPO_ROOT / "tests" / "test_profile_sparse_helper_coverage.py").read_text(
        encoding="utf-8"
    )

    assert "profile_solve._" not in text


def test_transport_solve_does_not_import_profile_solve_globals() -> None:
    """Transport orchestration should depend on transport owners explicitly."""

    text = (PACKAGE_ROOT / "problems" / "transport_solve.py").read_text(
        encoding="utf-8"
    )
    forbidden_fragments = (
        "_PROFILE_SOLVE",
        'import_module("sfincs_jax.problems.profile_solve")',
        "vars(_PROFILE_SOLVE)",
        "globals()[_name]",
    )

    for fragment in forbidden_fragments:
        assert fragment not in text


def test_schur_heuristic_tests_use_canonical_policy_and_preconditioner_owners() -> None:
    """Schur/PAS unit tests should not widen profile_solve's private API."""

    text = (REPO_ROOT / "tests" / "test_schur_precond_heuristic.py").read_text(
        encoding="utf-8"
    )

    assert "profile_solve._" not in text
    for owner in (
        "sfincs_jax.problems.profile_policies",
        "sfincs_jax.problems.profile_preconditioner_build",
        "sfincs_jax.solvers.path_policy",
        "sfincs_jax.solvers.preconditioning",
    ):
        assert owner in text


def test_pas_policy_tests_use_pas_policy_owner_not_profile_solve() -> None:
    """PAS applicability and memory tests belong to the PAS policy owner."""

    text = (REPO_ROOT / "tests" / "test_pas_preconditioner_policy.py").read_text(
        encoding="utf-8"
    )

    assert "profile_solve" not in text
    assert "sfincs_jax.solvers.preconditioner_pas_policy" in text


def test_distributed_gmres_axis_tests_use_krylov_dispatch_owner() -> None:
    """Distributed-GMRES axis tests belong to the Krylov dispatch owner."""

    text = (REPO_ROOT / "tests" / "test_distributed_gmres_axis.py").read_text(
        encoding="utf-8"
    )

    assert "profile_solve" not in text
    assert "sfincs_jax.solvers.krylov_dispatch" in text


def test_sparse_assembly_tests_use_sparse_preconditioner_owner() -> None:
    """Sparse assembly tests should not assert profile_solve alias plumbing."""

    text = (REPO_ROOT / "tests" / "test_sparse_assembly.py").read_text(
        encoding="utf-8"
    )

    assert "profile_solve" not in text
    assert "sfincs_jax.solvers.preconditioner_xblock_tz_sparse" in text


def test_device_operator_tests_use_active_layout_and_xblock_policy_owners() -> None:
    """Device-operator tests may run solves but should not test helper aliases."""

    text = (REPO_ROOT / "tests" / "test_rhs1_device_operator.py").read_text(
        encoding="utf-8"
    )

    forbidden = (
        "profile_solve._transport_active_dof_indices",
        "profile_solve._rhs1_xblock_policy",
    )
    for fragment in forbidden:
        assert fragment not in text
    assert "sfincs_jax.problems.transport_linear_system" in text
    assert "sfincs_jax.solvers.preconditioner_xblock_policy" in text


def test_transport_helper_tests_do_not_use_profile_solve_aliases() -> None:
    """RHSMode=2/3 helper tests should import transport owners directly."""

    forbidden_attrs = (
        "_transport_active_dof_indices",
        "_build_rhsmode23",
        "_try_build_rhsmode23",
    )
    offenders: list[tuple[str, str]] = []
    for path in sorted((REPO_ROOT / "tests").glob("test_*.py")):
        if path.name == "test_profile_solve_module_wrappers.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        profile_solve_aliases: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "sfincs_jax.problems.profile_solve":
                        profile_solve_aliases.add(alias.asname or "profile_solve")
            elif isinstance(node, ast.ImportFrom):
                if node.module == "sfincs_jax.problems" and any(
                    alias.name == "profile_solve" for alias in node.names
                ):
                    for alias in node.names:
                        if alias.name == "profile_solve":
                            profile_solve_aliases.add(alias.asname or alias.name)
            elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                if node.value.id in profile_solve_aliases and any(
                    node.attr.startswith(attr) for attr in forbidden_attrs
                ):
                    offenders.append((path.relative_to(REPO_ROOT).as_posix(), node.attr))

    assert offenders == []


def test_structured_csr_docs_tests_use_profile_setup_owner() -> None:
    """Structured-CSR method constants belong to profile setup."""

    text = (REPO_ROOT / "tests" / "test_structured_csr_docs.py").read_text(
        encoding="utf-8"
    )

    assert "profile_solve" not in text
    assert "sfincs_jax.problems.profile_setup" in text


def test_rhs1_dispatch_coverage_uses_canonical_helper_owners() -> None:
    """RHSMode-1 dispatch tests may run solves but should not assert helper aliases."""

    text = (REPO_ROOT / "tests" / "test_profile_rhs1_dispatch_coverage.py").read_text(
        encoding="utf-8"
    )

    assert "profile_solve._" not in text
    assert "sfincs_jax.problems.profile_policies" in text
    assert "sfincs_jax.solvers.path_policy" in text


def test_profile_solve_private_test_usage_is_limited_to_wrapper_contracts() -> None:
    """Only explicit wrapper-contract tests may touch profile_solve private seams."""

    allowed = {
        "tests/test_profile_solve_module_wrappers.py",
        "tests/test_source_tree_consolidation.py",
    }
    offenders = []
    for path in sorted((REPO_ROOT / "tests").glob("test_*.py")):
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel in allowed:
            continue
        if "profile_solve._" in path.read_text(encoding="utf-8"):
            offenders.append(rel)

    assert offenders == []


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
    """Claim-policy gates should live under validation, not workflow sprawl."""

    assert not (PACKAGE_ROOT / "workflows" / "qi_res15_gpu_campaign.py").exists()
    module = importlib.import_module("sfincs_jax.validation.qi_device")
    assert hasattr(module, "evaluate_qi_res15_gpu_campaign_files")


def test_deleted_tiny_validation_facades_are_absent() -> None:
    """Keep Fortran/PETSc fixture readers in one validation owner."""

    assert not (PACKAGE_ROOT / "validation" / "petsc_binary.py").exists()
    module = importlib.import_module("sfincs_jax.validation.fortran")
    assert hasattr(module, "read_petsc_vec")
    assert hasattr(module, "read_petsc_mat_aij")


def test_deleted_sparse_replay_filename_is_absent() -> None:
    """RHSMode-1 sparse-PC orchestration should use the canonical solve owner."""

    assert not (PACKAGE_ROOT / "problems" / "profile_sparse_replay.py").exists()
    module = importlib.import_module("sfincs_jax.problems.profile_sparse_solve")
    assert hasattr(module, "build_sparse_pc_generic_branch_setup")


def test_deleted_h5_parity_validation_facade_is_absent() -> None:
    """Strict HDF5 parity is part of the public comparison API."""

    assert not (PACKAGE_ROOT / "validation" / "h5_parity.py").exists()
    module = importlib.import_module("sfincs_jax.compare")
    assert hasattr(module, "compare_h5_outputs")
    assert hasattr(module, "H5DatasetParity")


def test_deleted_solver_selection_policy_facade_is_absent() -> None:
    """Measured candidate gates live with solver path-policy helpers."""

    assert not (PACKAGE_ROOT / "solvers" / "selection_policy.py").exists()
    module = importlib.import_module("sfincs_jax.solvers.path_policy")
    assert hasattr(module, "SolverCandidateMetrics")
    assert hasattr(module, "solver_candidate_gate")


def test_deleted_qi_promotion_policy_solver_facade_is_absent() -> None:
    """QI promotion evidence gates belong to validation, not solver kernels."""

    assert not (PACKAGE_ROOT / "solvers" / "preconditioner_qi_policy.py").exists()
    module = importlib.import_module("sfincs_jax.validation.qi_device")
    assert hasattr(module, "QIRunEvidence")
    assert hasattr(module, "evaluate_qi_production_ladder_promotion")


def test_deleted_full_fp_species_preconditioner_facade_is_absent() -> None:
    """Full-FP species block preconditioners belong to the kinetic owner."""

    assert not (PACKAGE_ROOT / "solvers" / "preconditioner_full_fp_species.py").exists()
    module = importlib.import_module("sfincs_jax.solvers.preconditioner_full_fp_kinetic")
    assert hasattr(module, "build_rhs1_species_block_preconditioner")
    assert hasattr(module, "build_rhs1_species_xblock_preconditioner")


def test_deleted_profile_linear_systems_facade_is_absent() -> None:
    """Matrix-free residual wrappers belong to the profile-system owner."""

    assert not (PACKAGE_ROOT / "operators" / "profile_linear_systems.py").exists()
    module = importlib.import_module("sfincs_jax.operators.profile_system")
    assert hasattr(module, "V3FBlockLinearSystem")
    assert hasattr(module, "V3FullLinearSystem")


def test_deleted_profile_sources_facade_is_absent() -> None:
    """Constraint-source kernels belong to the profile-system owner."""

    assert not (PACKAGE_ROOT / "operators" / "profile_sources.py").exists()
    module = importlib.import_module("sfincs_jax.operators.profile_system")
    assert hasattr(module, "constraint_scheme1_inject_source")
    assert hasattr(module, "constraint_scheme2_source_from_f")


def test_deleted_profile_compressed_layout_facade_is_absent() -> None:
    """Compressed RHSMode=1 pitch layouts belong to the profile-layout owner."""

    assert not (PACKAGE_ROOT / "operators" / "profile_compressed_layout.py").exists()
    module = importlib.import_module("sfincs_jax.operators.profile_layout")
    assert hasattr(module, "RHS1CompressedPitchLayout")
    assert hasattr(module, "build_rhs1_compressed_pitch_layout")


def test_deleted_transport_parallel_worker_wrapper_is_absent() -> None:
    """Transport subprocess worker CLI belongs to the parallel runtime owner."""

    assert not (PACKAGE_ROOT / "problems" / "transport_parallel_worker.py").exists()
    module = importlib.import_module("sfincs_jax.problems.transport_parallel_runtime")
    assert hasattr(module, "main")
    assert hasattr(module, "transport_parallel_result_to_npz_arrays")


def test_canonical_flat_domain_modules_are_importable() -> None:
    """Canonical owners replace the deleted compatibility import paths."""

    canonical_modules = (
        "sfincs_jax.operators.profile_collisionless",
        "sfincs_jax.operators.profile_fblock",
        "sfincs_jax.operators.profile_full_system",
        "sfincs_jax.operators.profile_layout",
        "sfincs_jax.operators.profile_system",
        "sfincs_jax.problems.profile_solve",
        "sfincs_jax.problems.profile_policies",
        "sfincs_jax.problems.profile_residual",
        "sfincs_jax.problems.profile_dense",
        "sfincs_jax.problems.profile_sparse_xblock",
        "sfincs_jax.problems.transport_diagnostics",
        "sfincs_jax.problems.transport_finalize",
        "sfincs_jax.problems.transport_linear_system",
        "sfincs_jax.problems.transport_policies",
        "sfincs_jax.problems.transport_parallel_runtime",
        "sfincs_jax.solvers.preconditioner_pas_xblock_ilu",
        "sfincs_jax.solvers.preconditioner_xblock_tz_sparse",
        "sfincs_jax.solvers.preconditioner_full_fp_kinetic",
        "sfincs_jax.solvers.preconditioner_qi_basis",
        "sfincs_jax.solvers.preconditioner_schur_profile",
        "sfincs_jax.solvers.preconditioner_symbolic_profile",
    )
    for module_name in canonical_modules:
        module = importlib.import_module(module_name)
        assert module.__name__ == module_name
