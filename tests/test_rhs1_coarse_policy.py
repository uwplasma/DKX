from __future__ import annotations

from sfincs_jax.solvers.preconditioners.schur.profile_response import (
    resolve_active_native_field_split_sparse_coarse_policy,
    resolve_active_native_stack_policy,
    resolve_active_sparse_coarse_residual_policy,
)


def test_native_stack_policy_clamps_budget_and_normalizes_solver() -> None:
    policy = resolve_active_native_stack_policy(
        max_factor_nbytes=1000,
        env={
            "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_STACK_BASE_BUDGET_FRACTION": "2.0",
            "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_STACK_SCHWARZ": "yes",
            "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_STACK_SCHWARZ_MAX_SIZE": "42",
            "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_XELL_COARSE_MAX_SIZE": "7",
            "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_STACK_COARSE_SOLVER": "petrov-galerkin",
        },
    )

    assert policy.base_budget_fraction == 1.0
    assert policy.base_budget_nbytes == 1000
    assert policy.schwarz_requested is True
    assert policy.schwarz_max_size == 42
    assert policy.max_coarse_size == 7
    assert policy.coarse_solver_mode == "galerkin"


def test_native_stack_policy_prefers_stack_specific_coarse_cap() -> None:
    policy = resolve_active_native_stack_policy(
        max_factor_nbytes=1000,
        env={
            "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_STACK_BASE_BUDGET_FRACTION": "0.001",
            "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_XELL_COARSE_MAX_SIZE": "7",
            "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_STACK_COARSE_MAX_SIZE": "11",
            "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_STACK_COARSE_SOLVER": "invalid",
        },
    )

    assert policy.base_budget_fraction == 0.05
    assert policy.base_budget_nbytes == 50
    assert policy.max_coarse_size == 11
    assert policy.coarse_solver_mode == "least_squares"


def test_native_field_split_policy_routes_xell_angular_and_admission_controls() -> None:
    policy = resolve_active_native_field_split_sparse_coarse_policy(
        requested_kind="active-xell-angular-field-split-sparse-coarse",
        env={
            "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SPARSE_COARSE_MAX_SIZE": "32",
            "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_XELL_COARSE_SOLVER": "ztaz",
            "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_COARSE_ADMISSION_PROBES": "0",
            "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_COARSE_ADMISSION_MAX_RELATIVE_RESIDUAL": "-1",
            "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_COARSE_ADMISSION_MIN_IMPROVEMENT": "-2",
        },
    )

    assert policy.is_multiline is True
    assert policy.is_angular_only is False
    assert policy.is_coupled_kinetic is False
    assert policy.output_kind == "active_multiline_field_split_sparse_coarse"
    assert policy.requested_base_kind == "active_multiline_xell_angular"
    assert policy.max_coarse_size == 32
    assert policy.coarse_solver_mode == "galerkin"
    assert policy.admission_probes == 1
    assert policy.admission_max_relative_residual == 0.0
    assert policy.admission_min_improvement == 0.0


def test_native_field_split_policy_routes_coupled_kinetic_with_specific_cap() -> None:
    policy = resolve_active_native_field_split_sparse_coarse_policy(
        requested_kind="active-dominant-kinetic-sparse-coarse",
        env={
            "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SPARSE_COARSE_MAX_SIZE": "32",
            "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_COARSE_MAX_SIZE": "9",
        },
    )

    assert policy.is_coupled_kinetic is True
    assert policy.output_kind == "active_coupled_kinetic_field_split_sparse_coarse"
    assert policy.requested_base_kind == "active_coupled_kinetic_block"
    assert policy.max_coarse_size == 9


def test_sparse_coarse_policy_routes_base_kind_and_solver_defaults() -> None:
    filtered = resolve_active_sparse_coarse_residual_policy(
        requested_kind="active-filtered-sparse-factor-sparse-coarse",
        env={},
    )
    schwarz = resolve_active_sparse_coarse_residual_policy(
        requested_kind="active-ras-sparse-coarse",
        env={"SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SPARSE_COARSE_SOLVER": "normal-equations"},
    )
    xblock = resolve_active_sparse_coarse_residual_policy(
        requested_kind="active-xblock-tail-sparse-coarse",
        env={"SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SPARSE_COARSE_SOLVER": "invalid"},
    )

    assert filtered.base_kind == "active_filtered_sparse_factor"
    assert filtered.output_kind == "active_filtered_sparse_coarse"
    assert filtered.coarse_solver_mode == "least_squares"
    assert schwarz.base_kind == "active_overlap_schwarz"
    assert schwarz.output_kind == "active_tail_sparse_coarse"
    assert schwarz.coarse_solver_mode == "least_squares"
    assert xblock.base_kind == "active_xblock"
    assert xblock.coarse_solver_mode == "galerkin"
