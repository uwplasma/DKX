from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp

import sfincs_jax.problems.profile_policies as profile_policies
import sfincs_jax.problems.transport_policies as transport_policies
import sfincs_jax.solvers.path_policy as path_policy
import sfincs_jax.solvers.preconditioning as preconditioning


def test_use_solver_jit_respects_boolean_env(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_SOLVER_JIT", "1")
    assert preconditioning.use_solver_jit(size_hint=10_000_000)

    monkeypatch.setenv("SFINCS_JAX_SOLVER_JIT", "off")
    assert not preconditioning.use_solver_jit(size_hint=1)


def test_use_solver_jit_uses_threshold_and_invalid_env_fallback(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_SOLVER_JIT", raising=False)
    monkeypatch.setenv("SFINCS_JAX_SOLVER_JIT_MAX_SIZE", "256")
    assert preconditioning.use_solver_jit(size_hint=128)
    assert not preconditioning.use_solver_jit(size_hint=512)

    monkeypatch.setenv("SFINCS_JAX_SOLVER_JIT_MAX_SIZE", "not-an-int")
    assert preconditioning.use_solver_jit(size_hint=1)
    assert not preconditioning.use_solver_jit(size_hint=100_001)


def test_use_solver_jit_falls_back_to_cached_size_hint(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_SOLVER_JIT", raising=False)
    monkeypatch.setenv("SFINCS_JAX_SOLVER_JIT_MAX_SIZE", "500")
    preconditioning.set_precond_size_hint(400)
    try:
        assert preconditioning.use_solver_jit() is True
        preconditioning.set_precond_size_hint(600)
        assert preconditioning.use_solver_jit() is False
    finally:
        preconditioning.set_precond_size_hint(None)


def test_auto_pas_geom4_fp32_precond_allowed_policy_boundaries(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_PRECOND_FP32_PAS_GEOM4", raising=False)
    monkeypatch.delenv("SFINCS_JAX_PRECOND_FP32_PAS_GEOM4_MIN_SIZE", raising=False)
    monkeypatch.delenv("SFINCS_JAX_PRECOND_FP32_PAS_GEOM4_ER_MAX", raising=False)
    monkeypatch.setattr("sfincs_jax.solvers.preconditioning.jax.default_backend", lambda: "cpu")

    preconditioning.set_precond_policy_hints(
        geom_scheme=4,
        use_dkes=False,
        rhs1_precond_kind="schur",
        has_pas=True,
        has_fp=False,
        include_phi1=False,
        rhs_mode=1,
        er_abs=0.0,
    )
    preconditioning.set_precond_size_hint(20_000)
    try:
        assert preconditioning.auto_pas_geom4_fp32_precond_allowed(size_hint=20_000)

        monkeypatch.setenv("SFINCS_JAX_PRECOND_FP32_PAS_GEOM4", "off")
        assert not preconditioning.auto_pas_geom4_fp32_precond_allowed(size_hint=20_000)
        monkeypatch.delenv("SFINCS_JAX_PRECOND_FP32_PAS_GEOM4", raising=False)

        monkeypatch.setenv("SFINCS_JAX_PRECOND_FP32_PAS_GEOM4_ER_MAX", "1e-14")
        preconditioning.set_precond_policy_hints(
            geom_scheme=4,
            use_dkes=False,
            rhs1_precond_kind="schur",
            has_pas=True,
            has_fp=False,
            include_phi1=False,
            rhs_mode=1,
            er_abs=1e-12,
        )
        assert not preconditioning.auto_pas_geom4_fp32_precond_allowed(size_hint=20_000)
    finally:
        preconditioning.set_precond_policy_hints()
        preconditioning.set_precond_size_hint(None)


def test_precond_dtype_respects_explicit_env(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_PRECOND_DTYPE", "fp32")
    assert preconditioning.precond_dtype(size_hint=1) == jnp.float32

    monkeypatch.setenv("SFINCS_JAX_PRECOND_DTYPE", "float64")
    assert preconditioning.precond_dtype(size_hint=10_000_000) == jnp.float64


def test_dense_backend_allowed_defaults_and_env(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_DENSE_ALLOW_ACCELERATOR", raising=False)
    assert profile_policies.rhs1_dense_backend_allowed(backend="cpu")

    assert not profile_policies.rhs1_dense_backend_allowed(backend="gpu")

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DENSE_ALLOW_ACCELERATOR", "on")
    assert profile_policies.rhs1_dense_backend_allowed(backend="gpu")

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DENSE_ALLOW_ACCELERATOR", "0")
    assert not profile_policies.rhs1_dense_backend_allowed(backend="cpu")


def test_resource_exhausted_error_detection_includes_causes() -> None:
    exc = RuntimeError("top level")
    exc.__cause__ = MemoryError("resource_exhausted during allocation")
    assert path_policy.is_resource_exhausted_error(exc)
    assert not path_policy.is_resource_exhausted_error(RuntimeError("solver diverged"))


def test_rhs1_sharded_line_override_allowed_whitelist() -> None:
    assert profile_policies.rhs1_sharded_line_override_allowed(None)
    assert profile_policies.rhs1_sharded_line_override_allowed("theta_line")
    assert not profile_policies.rhs1_sharded_line_override_allowed("schur")


def test_rhs1_pas_dkes_xblock_rejects_invalid_backend_and_zero_limits() -> None:
    assert not profile_policies.rhs1_pas_dkes_xblock_allowed(
        has_pas=True,
        use_dkes=True,
        backend="metal",
        n_theta=9,
        n_zeta=11,
        max_l=4,
        xblock_tz_limit=1000,
    )
    assert not profile_policies.rhs1_pas_dkes_xblock_allowed(
        has_pas=True,
        use_dkes=True,
        backend="gpu",
        n_theta=1,
        n_zeta=11,
        max_l=4,
        xblock_tz_limit=1000,
    )
    assert not profile_policies.rhs1_pas_dkes_xblock_allowed(
        has_pas=True,
        use_dkes=True,
        backend="gpu",
        n_theta=9,
        n_zeta=11,
        max_l=4,
        xblock_tz_limit=0,
    )


def test_pas_tokamak_gpu_policy_handles_invalid_env_bounds(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_TOKAMAK_GPU_THETA_MAX", "bad")
    assert profile_policies.rhs1_pas_tokamak_gpu_theta_allowed(
        has_pas=True,
        has_fp=False,
        backend="gpu",
        tokamak_like=True,
        active_size=500,
        er_abs=1.0e-2,
        schur_er_min=1.0e-12,
        has_magdrift=False,
        has_collisionless=True,
    )

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_TOKAMAK_GPU_XBLOCK_ACTIVE_MIN", "1000")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_TOKAMAK_GPU_XBLOCK_ACTIVE_MAX", "bad")
    assert not profile_policies.rhs1_pas_tokamak_gpu_xblock_preferred(
        has_pas=True,
        has_fp=False,
        backend="gpu",
        tokamak_like=True,
        active_size=500,
        er_abs=1.0e-2,
        schur_er_min=1.0e-12,
        has_magdrift=False,
        has_collisionless=True,
        n_theta=10,
        n_zeta=1,
        max_l=14,
        xblock_tz_limit=1200,
    )
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_TOKAMAK_GPU_XBLOCK_ACTIVE_MAX", "12000")
    assert profile_policies.rhs1_pas_tokamak_gpu_xblock_preferred(
        has_pas=True,
        has_fp=False,
        backend="gpu",
        tokamak_like=True,
        active_size=1500,
        er_abs=1.0e-2,
        schur_er_min=1.0e-12,
        has_magdrift=False,
        has_collisionless=True,
        n_theta=10,
        n_zeta=1,
        max_l=14,
        xblock_tz_limit=1200,
    )


def test_rhs1_gpu_sparse_fallback_skip_invalid_ratio_and_nonpositive_ratio(monkeypatch) -> None:
    op = SimpleNamespace(
        rhs_mode=1,
        include_phi1=False,
        fblock=SimpleNamespace(pas=object()),
    )
    monkeypatch.setattr("sfincs_jax.problems.profile_policies.jax.default_backend", lambda: "gpu")

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_GPU_SPARSE_SKIP_RATIO", "bad")
    assert profile_policies.rhs1_gpu_sparse_fallback_skip_allowed_current_backend(
        op=op,
        rhs1_precond_kind="schur",
        use_active_dof_mode=True,
        residual_norm=5.0,
        target=1.0,
    )

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_GPU_SPARSE_SKIP_RATIO", "0")
    assert not profile_policies.rhs1_gpu_sparse_fallback_skip_allowed_current_backend(
        op=op,
        rhs1_precond_kind="schur",
        use_active_dof_mode=True,
        residual_norm=5.0,
        target=1.0,
    )


def test_sparse_structural_tol_handles_invalid_and_negative_env(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_SPARSE_STRUCTURAL_TOL", raising=False)
    default_tol = preconditioning.sparse_structural_tol()
    assert default_tol >= 0.0

    monkeypatch.setenv("SFINCS_JAX_SPARSE_STRUCTURAL_TOL", "bad")
    assert preconditioning.sparse_structural_tol() == default_tol

    monkeypatch.setenv("SFINCS_JAX_SPARSE_STRUCTURAL_TOL", "-1.0")
    assert preconditioning.sparse_structural_tol() == 0.0


def test_transport_tzfft_accelerator_auto_allowed_boundary_cases(monkeypatch) -> None:
    cpu_op = SimpleNamespace(rhs_mode=3, include_phi1=False, n_x=1, n_theta=2, n_zeta=2, total_size=10, fblock=SimpleNamespace(fp=None))
    assert transport_policies.transport_tzfft_accelerator_auto_allowed(cpu_op, backend="cpu")

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_TZFFT_ACCELERATOR_AUTO_MAX", "bad")
    reject_phi1 = SimpleNamespace(rhs_mode=3, include_phi1=True, n_x=1, n_theta=37, n_zeta=5, total_size=1000, fblock=SimpleNamespace(fp=None))
    reject_rhs = SimpleNamespace(rhs_mode=1, include_phi1=False, n_x=1, n_theta=37, n_zeta=5, total_size=1000, fblock=SimpleNamespace(fp=None))
    reject_nx = SimpleNamespace(rhs_mode=3, include_phi1=False, n_x=3, n_theta=37, n_zeta=5, total_size=1000, fblock=SimpleNamespace(fp=None))
    reject_grid = SimpleNamespace(rhs_mode=3, include_phi1=False, n_x=1, n_theta=7, n_zeta=7, total_size=1000, fblock=SimpleNamespace(fp=None))
    assert not transport_policies.transport_tzfft_accelerator_auto_allowed(reject_phi1, backend="gpu")
    assert not transport_policies.transport_tzfft_accelerator_auto_allowed(reject_rhs, backend="gpu")
    assert not transport_policies.transport_tzfft_accelerator_auto_allowed(reject_nx, backend="gpu")
    assert not transport_policies.transport_tzfft_accelerator_auto_allowed(reject_grid, backend="gpu")


def test_rhsmode1_dense_and_host_dense_policy_envs(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DENSE_KRYLOV", "0")
    assert not profile_policies.rhs1_dense_krylov_allowed()
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DENSE_KRYLOV", "1")
    assert profile_policies.rhs1_dense_krylov_allowed()
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_DENSE_KRYLOV", raising=False)
    assert profile_policies.rhs1_dense_krylov_allowed()


def test_full_fp_dense_auto_route_selects_bounded_cpu_auto(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_DENSE_FP_CUTOFF", raising=False)
    decision = profile_policies.resolve_rhs1_full_fp_dense_auto_route(
        solve_method="auto",
        solve_method_kind="auto",
        use_implicit=False,
        has_fp=True,
        has_pas=False,
        include_phi1=False,
        rhs_mode=1,
        active_size=1200,
        dense_active_cutoff=8000,
        backend="cpu",
    )

    assert decision.selected
    assert decision.solve_method == "dense"
    assert decision.solve_method_kind == "dense"
    assert decision.cutoff == 8000
    assert decision.messages == (
        (
            1,
            "solve_v3_full_system_linear_gmres: auto-selected dense "
            "full-FP solve (size=1200 <= cutoff=8000)",
        ),
    )


def test_full_fp_dense_auto_route_rejects_non_full_fp_or_implicit() -> None:
    base = dict(
        solve_method="default",
        solve_method_kind="default",
        use_implicit=False,
        has_fp=True,
        has_pas=False,
        include_phi1=False,
        rhs_mode=1,
        active_size=1200,
        dense_active_cutoff=8000,
        backend="cpu",
    )

    for override in (
        {"use_implicit": True},
        {"has_fp": False},
        {"has_pas": True},
        {"include_phi1": True},
        {"rhs_mode": 2},
        {"active_size": 9000},
        {"solve_method": "gmres", "solve_method_kind": "gmres"},
    ):
        decision = profile_policies.resolve_rhs1_full_fp_dense_auto_route(
            **{**base, **override}
        )
        assert not decision.selected
        assert decision.solve_method == str(override.get("solve_method", "default"))
        assert decision.solve_method_kind == str(
            override.get("solve_method_kind", "default")
        )
        assert decision.messages == ()


def test_full_fp_dense_auto_route_requires_accelerator_admission(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_DENSE_ALLOW_ACCELERATOR", raising=False)
    decision = profile_policies.resolve_rhs1_full_fp_dense_auto_route(
        solve_method="incremental",
        solve_method_kind="incremental",
        use_implicit=False,
        has_fp=True,
        has_pas=False,
        include_phi1=False,
        rhs_mode=1,
        active_size=1200,
        dense_active_cutoff=8000,
        backend="gpu",
    )
    assert not decision.selected

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DENSE_ALLOW_ACCELERATOR", "1")
    decision = profile_policies.resolve_rhs1_full_fp_dense_auto_route(
        solve_method="incremental",
        solve_method_kind="incremental",
        use_implicit=False,
        has_fp=True,
        has_pas=False,
        include_phi1=False,
        rhs_mode=1,
        active_size=1200,
        dense_active_cutoff=8000,
        backend="gpu",
    )
    assert decision.selected
    assert decision.solve_method == "dense"


def _profile_route_op(
    *,
    has_fp: bool = True,
    has_pas: bool = False,
    rhs_mode: int = 1,
    include_phi1: bool = False,
    constraint_scheme: int = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        rhs_mode=rhs_mode,
        include_phi1=include_phi1,
        constraint_scheme=constraint_scheme,
        fblock=SimpleNamespace(
            fp=object() if has_fp else None,
            pas=object() if has_pas else None,
        ),
    )


def test_initial_sparse_shortcut_route_selects_gpu_dkes_and_clears_preconds() -> None:
    decision = profile_policies.resolve_rhs1_initial_sparse_shortcut_route(
        op=_profile_route_op(),
        rhs1_precond_env_user="auto",
        rhs1_bicgstab_env_user="",
        rhs1_precond_kind="schur",
        rhs1_precond_enabled=True,
        rhs1_bicgstab_kind="rhs1",
        solve_method_kind="auto",
        sparse_precond_mode="auto",
        active_size=1200,
        sparse_max_size=2000,
        use_dkes=True,
        backend="gpu",
    )

    assert decision.gpu_dkes_sparse_shortcut
    assert not decision.cs0_petsc_compat
    assert decision.rhs1_precond_kind is None
    assert not decision.rhs1_precond_enabled
    assert decision.rhs1_bicgstab_kind is None
    assert decision.messages == (
        (
            1,
            "solve_v3_full_system_linear_gmres: GPU DKES auto mode -> sparse ILU shortcut "
            "(size=1200)",
        ),
    )


def test_initial_sparse_shortcut_route_keeps_preconds_when_only_cs0_sparse_first(
    monkeypatch,
) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_CS0_PETSC_COMPAT", raising=False)
    decision = profile_policies.resolve_rhs1_initial_sparse_shortcut_route(
        op=_profile_route_op(constraint_scheme=0),
        rhs1_precond_env_user="user",
        rhs1_bicgstab_env_user="user",
        rhs1_precond_kind="collision",
        rhs1_precond_enabled=True,
        rhs1_bicgstab_kind="collision",
        solve_method_kind="incremental",
        sparse_precond_mode="auto",
        active_size=1200,
        sparse_max_size=2000,
        use_dkes=False,
        backend="gpu",
    )

    assert decision.cs0_sparse_first
    assert not decision.cs0_petsc_compat
    assert not decision.gpu_dkes_sparse_shortcut
    assert decision.rhs1_precond_kind == "collision"
    assert decision.rhs1_precond_enabled
    assert decision.rhs1_bicgstab_kind == "collision"
    assert decision.messages == ()


def test_initial_sparse_shortcut_route_selects_cs0_petsc_compat(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_CS0_PETSC_COMPAT", "1")
    decision = profile_policies.resolve_rhs1_initial_sparse_shortcut_route(
        op=_profile_route_op(constraint_scheme=0),
        rhs1_precond_env_user="user",
        rhs1_bicgstab_env_user="user",
        rhs1_precond_kind="collision",
        rhs1_precond_enabled=True,
        rhs1_bicgstab_kind="collision",
        solve_method_kind="auto",
        sparse_precond_mode="auto",
        active_size=1200,
        sparse_max_size=2000,
        use_dkes=False,
        backend="cpu",
    )

    assert decision.cs0_petsc_compat
    assert not decision.cs0_dense_fallback_allowed
    assert decision.rhs1_precond_kind is None
    assert not decision.rhs1_precond_enabled
    assert decision.rhs1_bicgstab_kind is None
    assert decision.messages == (
        (
            1,
            "solve_v3_full_system_linear_gmres: constraintScheme=0 PETSc-compat auto mode "
            "-> dedicated sparse ILU path",
        ),
    )


def test_rhsmode1_host_dense_fallback_policy_envs(monkeypatch) -> None:
    assert profile_policies.rhs1_host_dense_fallback_allowed(backend="cpu")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DENSE_HOST_LU", "on")
    assert profile_policies.rhs1_host_dense_fallback_allowed(backend="gpu")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DENSE_HOST_LU", "off")
    assert not profile_policies.rhs1_host_dense_fallback_allowed(backend="gpu")
