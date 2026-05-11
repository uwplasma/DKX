from __future__ import annotations

from sfincs_jax.solver_path_policy import (
    PreconditionerPolicyHints,
    auto_pas_geom4_fp32_precond_allowed,
    is_resource_exhausted_error,
    precond_dtype_name,
    rhs1_dkes_gmres_budget,
    rhs1_residual_needs_rescue,
    rhsmode1_sparse_pc_default_permc_spec,
    rhsmode1_sparse_pc_default_restart,
    sparse_structural_tol,
    use_solver_jit,
)


def _geom4_pas_schur_hints(**overrides: object) -> PreconditionerPolicyHints:
    values = dict(
        size_hint=20_000,
        geom_scheme=4,
        use_dkes=False,
        rhs1_precond_kind="schur",
        has_pas=True,
        has_fp=False,
        include_phi1=False,
        rhs_mode=1,
        er_abs=0.0,
    )
    values.update(overrides)
    return PreconditionerPolicyHints(**values)


def test_use_solver_jit_preserves_cached_size_hint_and_invalid_threshold_fallback() -> None:
    env = {"SFINCS_JAX_SOLVER_JIT_MAX_SIZE": "500"}
    assert use_solver_jit(precond_size_hint=400, env=env)
    assert not use_solver_jit(precond_size_hint=600, env=env)

    # An explicit solve-size hint wins over cached preconditioner metadata.
    assert use_solver_jit(size_hint=100, precond_size_hint=600, env=env)

    invalid_env = {"SFINCS_JAX_SOLVER_JIT_MAX_SIZE": "bad"}
    assert use_solver_jit(size_hint=100_000, env=invalid_env)
    assert not use_solver_jit(size_hint=100_001, env=invalid_env)


def test_use_solver_jit_boolean_env_overrides_large_problem_size() -> None:
    assert use_solver_jit(size_hint=10_000_000, env={"SFINCS_JAX_SOLVER_JIT": "yes"})
    assert not use_solver_jit(size_hint=1, env={"SFINCS_JAX_SOLVER_JIT": "off"})


def test_precond_dtype_keeps_global_and_block_auto_thresholds_separate() -> None:
    hints = PreconditionerPolicyHints(size_hint=64)
    env = {
        "SFINCS_JAX_PRECOND_FP32_MIN_SIZE": "128",
        "SFINCS_JAX_PRECOND_FP32_MIN_BLOCK": "32",
    }
    assert precond_dtype_name(size_hint=None, hints=hints, backend="cpu", env=env) == "float64"
    assert precond_dtype_name(size_hint=64, hints=hints, backend="cpu", env=env) == "float32"


def test_precond_dtype_invalid_explicit_token_is_stability_first_fp64() -> None:
    hints = PreconditionerPolicyHints(size_hint=10_000_000)
    assert (
        precond_dtype_name(
            size_hint=10_000_000,
            hints=hints,
            backend="cpu",
            env={"SFINCS_JAX_PRECOND_DTYPE": "fast"},
        )
        == "float64"
    )


def test_auto_pas_geom4_fp32_preconditioner_is_narrow_and_env_guarded() -> None:
    hints = _geom4_pas_schur_hints()
    assert auto_pas_geom4_fp32_precond_allowed(size_hint=20_000, hints=hints, backend="cpu", env={})
    assert not auto_pas_geom4_fp32_precond_allowed(size_hint=20_000, hints=hints, backend="gpu", env={})
    assert not auto_pas_geom4_fp32_precond_allowed(
        size_hint=20_000,
        hints=_geom4_pas_schur_hints(use_dkes=True),
        backend="cpu",
        env={},
    )
    assert not auto_pas_geom4_fp32_precond_allowed(
        size_hint=20_000,
        hints=_geom4_pas_schur_hints(include_phi1=True),
        backend="cpu",
        env={},
    )
    assert not auto_pas_geom4_fp32_precond_allowed(
        size_hint=20_000,
        hints=_geom4_pas_schur_hints(er_abs=1.0e-10),
        backend="cpu",
        env={"SFINCS_JAX_PRECOND_FP32_PAS_GEOM4_ER_MAX": "1e-12"},
    )
    assert not auto_pas_geom4_fp32_precond_allowed(
        size_hint=20_000,
        hints=hints,
        backend="cpu",
        env={"SFINCS_JAX_PRECOND_FP32_PAS_GEOM4": "off"},
    )


def test_rescue_slack_clamps_negative_values_and_keeps_invalid_default() -> None:
    assert not rhs1_residual_needs_rescue(1.006e-12, 1.0e-12, env={})
    assert rhs1_residual_needs_rescue(1.02e-12, 1.0e-12, env={})
    assert rhs1_residual_needs_rescue(
        1.006e-12,
        1.0e-12,
        env={"SFINCS_JAX_RHSMODE1_RESCUE_TARGET_SLACK": "-1"},
    )
    assert not rhs1_residual_needs_rescue(
        1.006e-12,
        1.0e-12,
        env={"SFINCS_JAX_RHSMODE1_RESCUE_TARGET_SLACK": "bad"},
    )


def test_dkes_budget_defaults_cap_restart_without_overriding_forced_values() -> None:
    assert rhs1_dkes_gmres_budget(
        restart=20,
        maxiter=None,
        restart_forced=False,
        maxiter_forced=False,
        restart_cap_env="90",
    ) == (80, 600, True, True)
    assert rhs1_dkes_gmres_budget(
        restart=20,
        maxiter=10,
        restart_forced=True,
        maxiter_forced=True,
        restart_cap_env="90",
    ) == (20, 10, False, False)
    assert rhs1_dkes_gmres_budget(
        restart=20,
        maxiter=None,
        restart_forced=False,
        maxiter_forced=False,
        restart_cap_env="bad",
    ) == (80, 600, True, True)


def test_sparse_pc_defaults_preserve_measured_restart_and_permutation_paths() -> None:
    assert (
        rhsmode1_sparse_pc_default_permc_spec(
            constrained_pas_pc=True,
            tokamak_pas_er_pc=False,
            n_species=2,
        )
        == "MMD_ATA"
    )
    assert (
        rhsmode1_sparse_pc_default_permc_spec(
            constrained_pas_pc=True,
            tokamak_pas_er_pc=True,
            n_species=1,
        )
        == "MMD_AT_PLUS_A"
    )
    assert (
        rhsmode1_sparse_pc_default_restart(
            requested_restart=120,
            restart_env_value="",
            tokamak_pas_er_pc=True,
            n_species=1,
        )
        == 40
    )
    assert (
        rhsmode1_sparse_pc_default_restart(
            requested_restart=120,
            restart_env_value="120",
            tokamak_pas_er_pc=True,
            n_species=1,
        )
        == 120
    )
    assert (
        rhsmode1_sparse_pc_default_restart(
            requested_restart=120,
            restart_env_value="",
            tokamak_pas_er_pc=True,
            n_species=2,
        )
        == 120
    )


def test_sparse_structural_tol_and_resource_errors_cover_invalid_edge_cases() -> None:
    assert sparse_structural_tol(default_tol=1.0e-14, env={}) == 1.0e-14
    assert sparse_structural_tol(default_tol=1.0e-14, env={"SFINCS_JAX_SPARSE_STRUCTURAL_TOL": "bad"}) == 1.0e-14
    assert sparse_structural_tol(default_tol=1.0e-14, env={"SFINCS_JAX_SPARSE_STRUCTURAL_TOL": "-1"}) == 0.0

    exc = RuntimeError("top")
    exc.__cause__ = MemoryError("resource_exhausted during compile")
    assert is_resource_exhausted_error(exc)
    assert not is_resource_exhausted_error(RuntimeError("shape mismatch"))
