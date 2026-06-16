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
    "sfincs_jax.problems.transport_matrix.parallel",
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

TRANSPORT_COMPATIBILITY_IMPORTS = (
    (
        "sfincs_jax.transport_solve_setup",
        "sfincs_jax.problems.transport_matrix.setup",
        "resolve_transport_which_rhs_setup",
    ),
    (
        "sfincs_jax.transport_active_dense_setup",
        "sfincs_jax.problems.transport_matrix.active_dense",
        "resolve_transport_active_dense_setup",
    ),
    (
        "sfincs_jax.transport_loop_support",
        "sfincs_jax.problems.transport_matrix.loop",
        "resolve_transport_recycle_k",
    ),
    (
        "sfincs_jax.transport_solve_finalization",
        "sfincs_jax.problems.transport_matrix.finalize",
        "finalize_full_transport_rhs",
    ),
    (
        "sfincs_jax.transport_streaming_outputs",
        "sfincs_jax.problems.transport_matrix.streaming_outputs",
        "TransportStreamingOutputAccumulator",
    ),
    (
        "sfincs_jax.transport_postsolve_diagnostics",
        "sfincs_jax.problems.transport_matrix.postsolve_diagnostics",
        "compute_transport_postsolve_diagnostics",
    ),
    (
        "sfincs_jax.transport_policy",
        "sfincs_jax.problems.transport_matrix.policies",
        "transport_dense_backend_allowed",
    ),
    (
        "sfincs_jax.transport_solve_policy",
        "sfincs_jax.problems.transport_matrix.solve_policy",
        "resolve_transport_initial_solve_policy",
    ),
    (
        "sfincs_jax.transport_linear_solve",
        "sfincs_jax.problems.transport_matrix.linear_solve",
        "solve_transport_linear_with_residual",
    ),
    (
        "sfincs_jax.transport_host_gmres",
        "sfincs_jax.problems.transport_matrix.host_gmres",
        "transport_host_gmres_solve",
    ),
    (
        "sfincs_jax.transport_handoff_policy",
        "sfincs_jax.problems.transport_matrix.handoff_policy",
        "transport_polish_config_from_env",
    ),
    (
        "sfincs_jax.transport_residual_quality",
        "sfincs_jax.problems.transport_matrix.residual_quality",
        "transport_residual_gate_failure",
    ),
    (
        "sfincs_jax.transport_iteration_stats",
        "sfincs_jax.problems.transport_matrix.iteration_stats",
        "emit_transport_ksp_iteration_stats",
    ),
    (
        "sfincs_jax.transport_dense_lu",
        "sfincs_jax.problems.transport_matrix.dense_lu",
        "dense_solver_for_matvec",
    ),
    (
        "sfincs_jax.transport_dense_batch",
        "sfincs_jax.problems.transport_matrix.dense_batch",
        "solve_transport_dense_batch",
    ),
    (
        "sfincs_jax.transport_active_factor",
        "sfincs_jax.problems.transport_matrix.active_factor",
        "build_active_block_schur_factor",
    ),
    (
        "sfincs_jax.transport_sparse_direct_solve",
        "sfincs_jax.problems.transport_matrix.sparse_direct_solve",
        "transport_sparse_direct_solve",
    ),
    (
        "sfincs_jax.transport_parallel_payload",
        "sfincs_jax.problems.transport_matrix.parallel.payload",
        "solve_transport_parallel_payload",
    ),
    (
        "sfincs_jax.transport_parallel_policy",
        "sfincs_jax.problems.transport_matrix.parallel.policy",
        "transport_parallel_backend",
    ),
    (
        "sfincs_jax.transport_parallel_runtime",
        "sfincs_jax.problems.transport_matrix.parallel.runtime",
        "merge_transport_parallel_results",
    ),
    (
        "sfincs_jax.transport_parallel_pool",
        "sfincs_jax.problems.transport_matrix.parallel.pool",
        "TransportParallelPoolCache",
    ),
    (
        "sfincs_jax.transport_parallel_execution",
        "sfincs_jax.problems.transport_matrix.parallel.execution",
        "run_transport_parallel_payloads",
    ),
    (
        "sfincs_jax.transport_parallel_solve",
        "sfincs_jax.problems.transport_matrix.parallel.solve",
        "maybe_run_transport_parallel_solve",
    ),
    (
        "sfincs_jax.transport_parallel_validation",
        "sfincs_jax.problems.transport_matrix.parallel.validation",
        "validate_transport_worker_result_payload",
    ),
    (
        "sfincs_jax.transport_parallel_sharding",
        "sfincs_jax.problems.transport_matrix.parallel.sharding",
        "plan_single_case_operator_coarse_reuse",
    ),
    (
        "sfincs_jax.transport_parallel_worker",
        "sfincs_jax.problems.transport_matrix.parallel.worker",
        "main",
    ),
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


def test_transport_matrix_package_moves_preserve_legacy_imports() -> None:
    """Moved implementation modules must remain reachable through old names."""

    for legacy_name, new_name, public_name in TRANSPORT_COMPATIBILITY_IMPORTS:
        legacy_module = _import_module(legacy_name)
        new_module = _import_module(new_name)
        assert getattr(legacy_module, public_name) is getattr(new_module, public_name)
        if hasattr(legacy_module, "__all__"):
            assert public_name in legacy_module.__all__
        if hasattr(new_module, "__all__"):
            assert public_name in new_module.__all__
