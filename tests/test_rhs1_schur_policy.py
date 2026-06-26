from __future__ import annotations

from sfincs_jax.solvers.preconditioner_schur_profile import (
    canonical_schur_base_kind,
    resolve_active_native_field_split_sparse_coarse_policy,
    resolve_active_native_stack_policy,
    resolve_active_sparse_coarse_residual_policy,
    resolve_rhs1_schur_base_kind,
)


def test_canonical_schur_base_kind_aliases() -> None:
    assert canonical_schur_base_kind("theta") == "theta_line"
    assert canonical_schur_base_kind("xblock_theta_zeta") == "xblock_tz"
    assert canonical_schur_base_kind("pas_l_tz") == "pas_tz"
    assert canonical_schur_base_kind("tokamak_theta") == "pas_tokamak_theta"
    assert canonical_schur_base_kind("theta_zeta") == "adi"
    assert canonical_schur_base_kind("unknown") is None


def test_active_native_stack_policy_bounds_memory_and_solver_aliases() -> None:
    policy = resolve_active_native_stack_policy(
        max_factor_nbytes=10_000,
        env={
            "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_STACK_BASE_BUDGET_FRACTION": "2.0",
            "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_STACK_SCHWARZ": "yes",
            "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_STACK_SCHWARZ_MAX_SIZE": "bad",
            "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_STACK_COARSE_MAX_SIZE": "-4",
            "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_STACK_COARSE_SOLVER": "ztaz",
        },
    )

    assert policy.base_budget_fraction == 1.0
    assert policy.base_budget_nbytes == 10_000
    assert policy.schwarz_requested is True
    assert policy.schwarz_max_size == 100_000
    assert policy.max_coarse_size == 1
    assert policy.coarse_solver_mode == "galerkin"

    small_fraction = resolve_active_native_stack_policy(
        max_factor_nbytes=10_000,
        env={"SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_STACK_BASE_BUDGET_FRACTION": "0.01"},
    )
    assert small_fraction.base_budget_fraction == 0.05
    assert small_fraction.base_budget_nbytes == 500


def test_active_field_split_policy_classifies_requested_solver_families() -> None:
    angular = resolve_active_native_field_split_sparse_coarse_policy(
        requested_kind="active-angular-line",
        env={"SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_XELL_COARSE_SOLVER": "petrov_galerkin"},
    )
    assert angular.is_angular_only is True
    assert angular.is_multiline is False
    assert angular.output_kind == "active_angular_line_field_split_sparse_coarse"
    assert angular.requested_base_kind == "active_angular_line"
    assert angular.coarse_solver_mode == "galerkin"

    multiline = resolve_active_native_field_split_sparse_coarse_policy(
        requested_kind="active_multiline_xell_angular",
        env={},
    )
    assert multiline.is_multiline is True
    assert multiline.output_kind == "active_multiline_field_split_sparse_coarse"
    assert multiline.requested_base_kind == "active_multiline_xell_angular"

    coupled = resolve_active_native_field_split_sparse_coarse_policy(
        requested_kind="active_coupled_kinetic_field_split_sparse_coarse",
        env={
            "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_COARSE_MAX_SIZE": "256",
            "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_COARSE_ADMISSION_PROBES": "0",
            "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_COARSE_ADMISSION_MAX_RELATIVE_RESIDUAL": "-1",
            "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_COARSE_ADMISSION_MIN_IMPROVEMENT": "-2",
        },
    )
    assert coupled.is_coupled_kinetic is True
    assert coupled.output_kind == "active_coupled_kinetic_field_split_sparse_coarse"
    assert coupled.requested_base_kind == "active_coupled_kinetic_block"
    assert coupled.max_coarse_size == 256
    assert coupled.admission_probes == 1
    assert coupled.admission_max_relative_residual == 0.0
    assert coupled.admission_min_improvement == 0.0


def test_active_sparse_coarse_residual_policy_maps_physics_families() -> None:
    filtered = resolve_active_sparse_coarse_residual_policy(
        requested_kind="filtered-normal",
        env={
            "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SPARSE_COARSE_MAX_SIZE": "0",
            "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SPARSE_COARSE_SOLVER": "normal_equations",
        },
    )
    assert filtered.base_kind == "active_filtered_sparse_factor"
    assert filtered.output_kind == "active_filtered_sparse_coarse"
    assert filtered.max_coarse_size == 1
    assert filtered.coarse_solver_mode == "least_squares"

    scaled = resolve_active_sparse_coarse_residual_policy(requested_kind="equilibrated_ilu", env={})
    assert scaled.base_kind == "active_scaled_ilu"
    assert scaled.output_kind == "active_tail_sparse_coarse"
    assert scaled.coarse_solver_mode == "galerkin"

    schwarz = resolve_active_sparse_coarse_residual_policy(requested_kind="ras", env={})
    assert schwarz.base_kind == "active_overlap_schwarz"

    xblock = resolve_active_sparse_coarse_residual_policy(requested_kind="xblock", env={})
    assert xblock.base_kind == "active_xblock"


def test_resolve_schur_base_prefers_pas_tz_for_geometry4_offender(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SCHUR_BASE", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_PAS_TZ_MIN", raising=False)

    assert (
        resolve_rhs1_schur_base_kind(
            base_kind_env="",
            n_theta=9,
            n_zeta=9,
            n_species=2,
            total_size=11350,
            nxi_for_x=[4, 4, 8, 12, 14],
            has_pas=True,
            has_fp=False,
            has_er_xdot=False,
            has_er_xidot=False,
            use_dkes_exb=False,
            pas_tokamak_theta_applicable=False,
            pas_tz_applicable=True,
            geom_scheme=4,
        )
        == "pas_tz"
    )


def test_resolve_schur_base_small_pas_fallback_uses_pas_schur(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPECIES_BLOCK_MAX", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_TZ_MAX", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_TZ_PRECOND_MAX", "0")

    assert (
        resolve_rhs1_schur_base_kind(
            base_kind_env="",
            n_theta=5,
            n_zeta=5,
            n_species=2,
            total_size=410,
            nxi_for_x=[4, 4],
            has_pas=True,
            has_fp=False,
            has_er_xdot=False,
            has_er_xidot=False,
            use_dkes_exb=False,
            pas_tokamak_theta_applicable=False,
            pas_tz_applicable=False,
            geom_scheme=4,
        )
        == "pas_schur"
    )


def test_resolve_schur_base_small_pas_prefers_species_block(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SCHUR_BASE", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPECIES_BLOCK_MAX", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_XBLOCK_TZ_MAX", raising=False)

    assert (
        resolve_rhs1_schur_base_kind(
            base_kind_env="",
            n_theta=7,
            n_zeta=7,
            n_species=2,
            total_size=410,
            nxi_for_x=[4, 4],
            has_pas=True,
            has_fp=False,
            has_er_xdot=False,
            has_er_xidot=False,
            use_dkes_exb=False,
            pas_tokamak_theta_applicable=False,
            pas_tz_applicable=False,
            geom_scheme=4,
        )
        == "species_block"
    )


def test_resolve_schur_base_dkes_uses_bounded_xblock_else_pas_ilu(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_DKES_XBLOCK_TZ_MAX_BYTES", raising=False)
    kwargs = dict(
        base_kind_env="",
        n_theta=9,
        n_zeta=9,
        n_species=2,
        total_size=5000,
        nxi_for_x=[3, 3],
        has_pas=True,
        has_fp=False,
        has_er_xdot=False,
        has_er_xidot=False,
        use_dkes_exb=True,
        pas_tokamak_theta_applicable=False,
        pas_tz_applicable=True,
        geom_scheme=11,
    )
    assert resolve_rhs1_schur_base_kind(**kwargs) == "xblock_tz"

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DKES_XBLOCK_TZ_MAX_BYTES", "1")
    assert resolve_rhs1_schur_base_kind(**kwargs) == "pas_ilu"


def test_resolve_schur_base_large_pas_er_prefers_xmg(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_PAS_XMG_MIN", raising=False)

    assert (
        resolve_rhs1_schur_base_kind(
            base_kind_env="",
            n_theta=9,
            n_zeta=9,
            n_species=2,
            total_size=60000,
            nxi_for_x=[4, 4, 8, 12, 14],
            has_pas=True,
            has_fp=False,
            has_er_xdot=True,
            has_er_xidot=False,
            use_dkes_exb=False,
            pas_tokamak_theta_applicable=False,
            pas_tz_applicable=True,
            geom_scheme=4,
        )
        == "xmg"
    )
