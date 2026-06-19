from __future__ import annotations

import numpy as np
import pytest
import jax.numpy as jnp

from sfincs_jax.problems.profile_response.dense import (
    HostDenseFullSolveContext,
    HostDenseReducedSolveContext,
    RHS1DenseFallbackThresholds,
    RHS1ReducedDenseFallbackCandidateContext,
    rhs1_dense_fallback_thresholds_from_env,
    rhs1_dense_probe_admission,
    rhs1_dense_probe_enabled_from_env,
    rhs1_dense_probe_shortcut_decision,
    rhs1_dense_shortcut_setup_from_env,
    rhs1_fp_preconditioner_probe_kind_from_env,
    solve_rhs1_reduced_dense_fallback_candidate,
    solve_host_dense_full,
    solve_host_dense_reduced,
)


def test_rhs1_dense_shortcut_setup_from_env_uses_default_ratio(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_DENSE_SHORTCUT_RATIO", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_PAS_DENSE_ALLOW_MAX", raising=False)

    setup = rhs1_dense_shortcut_setup_from_env(
        has_pas=False,
        include_phi1=False,
        constraint_scheme=0,
        active_size=1000,
        dense_fallback_max=5000,
        dense_backend_allowed=True,
        host_dense_fallback_allowed=False,
        dense_krylov_allowed=False,
        backend="cpu",
    )
    assert setup.dense_shortcut_ratio == 1.0e6
    assert setup.dense_fallback_max == 5000
    assert not setup.disable_dense_pas
    assert setup.messages == ()


def test_rhs1_dense_shortcut_setup_from_env_handles_pas_dense_gate(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DENSE_SHORTCUT_RATIO", "bad")
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_PAS_DENSE_ALLOW_MAX", raising=False)

    setup = rhs1_dense_shortcut_setup_from_env(
        has_pas=True,
        include_phi1=False,
        constraint_scheme=1,
        active_size=3000,
        dense_fallback_max=5000,
        dense_backend_allowed=True,
        host_dense_fallback_allowed=False,
        dense_krylov_allowed=False,
        backend="gpu",
    )
    assert setup.dense_shortcut_ratio == 0.0
    assert setup.dense_fallback_max == 5000
    assert not setup.disable_dense_pas

    setup = rhs1_dense_shortcut_setup_from_env(
        has_pas=True,
        include_phi1=False,
        constraint_scheme=1,
        active_size=5000,
        dense_fallback_max=5000,
        dense_backend_allowed=True,
        host_dense_fallback_allowed=False,
        dense_krylov_allowed=False,
        backend="gpu",
    )
    assert setup.dense_shortcut_ratio == 0.0
    assert setup.dense_fallback_max == 0
    assert setup.disable_dense_pas

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_DENSE_ALLOW_MAX", "6000")
    setup = rhs1_dense_shortcut_setup_from_env(
        has_pas=True,
        include_phi1=False,
        constraint_scheme=1,
        active_size=5000,
        dense_fallback_max=5000,
        dense_backend_allowed=True,
        host_dense_fallback_allowed=False,
        dense_krylov_allowed=False,
        backend="gpu",
    )
    assert setup.dense_fallback_max == 5000
    assert not setup.disable_dense_pas


def test_rhs1_dense_shortcut_setup_from_env_reports_backend_disable(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DENSE_SHORTCUT_RATIO", "12")

    setup = rhs1_dense_shortcut_setup_from_env(
        has_pas=False,
        include_phi1=False,
        constraint_scheme=0,
        active_size=1000,
        dense_fallback_max=5000,
        dense_backend_allowed=False,
        host_dense_fallback_allowed=False,
        dense_krylov_allowed=False,
        backend="gpu",
    )
    assert setup.dense_shortcut_ratio == 0.0
    assert setup.dense_fallback_max == 0
    assert setup.messages == ((
        1,
        "solve_v3_full_system_linear_gmres: disabling RHSMode=1 "
        "dense shortcut/fallback on backend=gpu",
    ),)

    setup = rhs1_dense_shortcut_setup_from_env(
        has_pas=False,
        include_phi1=False,
        constraint_scheme=0,
        active_size=1000,
        dense_fallback_max=5000,
        dense_backend_allowed=False,
        host_dense_fallback_allowed=True,
        dense_krylov_allowed=False,
        backend="gpu",
    )
    assert setup.dense_fallback_max == 5000
    assert setup.messages == ((
        1,
        "solve_v3_full_system_linear_gmres: disabling RHSMode=1 "
        "dense shortcut (host dense fallback kept) on backend=gpu",
    ),)

    setup = rhs1_dense_shortcut_setup_from_env(
        has_pas=False,
        include_phi1=False,
        constraint_scheme=0,
        active_size=1000,
        dense_fallback_max=5000,
        dense_backend_allowed=False,
        host_dense_fallback_allowed=False,
        dense_krylov_allowed=True,
        backend="gpu",
    )
    assert setup.dense_fallback_max == 5000
    assert setup.messages == ((
        1,
        "solve_v3_full_system_linear_gmres: disabling RHSMode=1 "
        "dense shortcut disabled (dense Krylov fallback kept) on backend=gpu",
    ),)


def test_rhs1_dense_fallback_thresholds_use_default_ratio_and_huge_limit(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_DENSE_FALLBACK_MAX_HUGE", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_DENSE_FALLBACK_RATIO", raising=False)

    assert rhs1_dense_fallback_thresholds_from_env(
        dense_fallback_max=5000,
        residual_ratio=10.0,
    ) == RHS1DenseFallbackThresholds(
        dense_fallback_max_huge=5000,
        dense_fallback_ratio=100.0,
        dense_fallback_limit=5000,
        dense_fallback_trigger=False,
    )
    assert rhs1_dense_fallback_thresholds_from_env(
        dense_fallback_max=5000,
        residual_ratio=200.0,
    ) == RHS1DenseFallbackThresholds(
        dense_fallback_max_huge=5000,
        dense_fallback_ratio=100.0,
        dense_fallback_limit=5000,
        dense_fallback_trigger=True,
    )


def test_rhs1_dense_fallback_thresholds_respect_env_and_invalid_values(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DENSE_FALLBACK_MAX_HUGE", "12000")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DENSE_FALLBACK_RATIO", "25")

    assert rhs1_dense_fallback_thresholds_from_env(
        dense_fallback_max=5000,
        residual_ratio=30.0,
    ) == RHS1DenseFallbackThresholds(
        dense_fallback_max_huge=12000,
        dense_fallback_ratio=25.0,
        dense_fallback_limit=12000,
        dense_fallback_trigger=True,
    )

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DENSE_FALLBACK_MAX_HUGE", "bad")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DENSE_FALLBACK_RATIO", "bad")
    assert rhs1_dense_fallback_thresholds_from_env(
        dense_fallback_max=5000,
        residual_ratio=200.0,
    ) == RHS1DenseFallbackThresholds(
        dense_fallback_max_huge=5000,
        dense_fallback_ratio=100.0,
        dense_fallback_limit=5000,
        dense_fallback_trigger=True,
    )


def test_rhs1_dense_fallback_thresholds_can_disable_huge_limit(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DENSE_FALLBACK_MAX_HUGE", "12000")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DENSE_FALLBACK_RATIO", "25")

    assert rhs1_dense_fallback_thresholds_from_env(
        dense_fallback_max=5000,
        residual_ratio=30.0,
        allow_huge_limit=False,
    ) == RHS1DenseFallbackThresholds(
        dense_fallback_max_huge=5000,
        dense_fallback_ratio=25.0,
        dense_fallback_limit=5000,
        dense_fallback_trigger=True,
    )


def test_rhs1_fp_preconditioner_probe_kind_from_env_selects_collision(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_FP_PRECOND_PROBE", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_FP_PRECOND_PROBE_MIN", raising=False)

    assert (
        rhs1_fp_preconditioner_probe_kind_from_env(
            rhs1_precond_kind="theta_line",
            rhs1_precond_env="",
            has_fp=True,
            use_dkes=True,
            include_phi1=False,
            dense_fallback_max=6000,
            active_size=3000,
            rhs1_precond_enabled=True,
            solve_method_kind="incremental",
        )
        == "collision"
    )


def test_rhs1_fp_preconditioner_probe_kind_from_env_respects_guards(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_FP_PRECOND_PROBE", raising=False)
    base = dict(
        rhs1_precond_kind="theta_line",
        rhs1_precond_env="",
        has_fp=True,
        use_dkes=True,
        include_phi1=False,
        dense_fallback_max=6000,
        active_size=3000,
        rhs1_precond_enabled=True,
        solve_method_kind="incremental",
    )
    assert rhs1_fp_preconditioner_probe_kind_from_env(**{**base, "use_dkes": False}) == "theta_line"
    assert rhs1_fp_preconditioner_probe_kind_from_env(**{**base, "include_phi1": True}) == "theta_line"
    assert rhs1_fp_preconditioner_probe_kind_from_env(**{**base, "dense_fallback_max": 0}) == "theta_line"
    assert rhs1_fp_preconditioner_probe_kind_from_env(**{**base, "active_size": 2000}) == "theta_line"
    assert rhs1_fp_preconditioner_probe_kind_from_env(**{**base, "active_size": 7000}) == "theta_line"
    assert not rhs1_fp_preconditioner_probe_kind_from_env(**{**base, "rhs1_precond_kind": None})
    assert rhs1_fp_preconditioner_probe_kind_from_env(**{**base, "rhs1_precond_kind": "collision"}) == "collision"
    assert rhs1_fp_preconditioner_probe_kind_from_env(**{**base, "rhs1_precond_env": "schur"}) == "theta_line"
    assert rhs1_fp_preconditioner_probe_kind_from_env(**{**base, "has_fp": False}) == "theta_line"
    assert rhs1_fp_preconditioner_probe_kind_from_env(**{**base, "rhs1_precond_enabled": False}) == "theta_line"
    assert rhs1_fp_preconditioner_probe_kind_from_env(**{**base, "solve_method_kind": "dense"}) == "theta_line"


def test_rhs1_fp_preconditioner_probe_kind_from_env_respects_env(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FP_PRECOND_PROBE", "off")
    assert (
        rhs1_fp_preconditioner_probe_kind_from_env(
            rhs1_precond_kind="theta_line",
            rhs1_precond_env="",
            has_fp=True,
            use_dkes=True,
            include_phi1=False,
            dense_fallback_max=6000,
            active_size=3000,
            rhs1_precond_enabled=True,
            solve_method_kind="incremental",
        )
        == "theta_line"
    )

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FP_PRECOND_PROBE", "on")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FP_PRECOND_PROBE_MIN", "bad")
    assert (
        rhs1_fp_preconditioner_probe_kind_from_env(
            rhs1_precond_kind="theta_line",
            rhs1_precond_env="",
            has_fp=True,
            use_dkes=True,
            include_phi1=False,
            dense_fallback_max=6000,
            active_size=3000,
            rhs1_precond_enabled=True,
            solve_method_kind="incremental",
        )
        == "collision"
    )

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FP_PRECOND_PROBE_MIN", "3500")
    assert (
        rhs1_fp_preconditioner_probe_kind_from_env(
            rhs1_precond_kind="theta_line",
            rhs1_precond_env="",
            has_fp=True,
            use_dkes=True,
            include_phi1=False,
            dense_fallback_max=6000,
            active_size=3000,
            rhs1_precond_enabled=True,
            solve_method_kind="incremental",
        )
        == "theta_line"
    )


def test_host_dense_reduced_row_scaled_lu_solves_square_system() -> None:
    a_np = np.asarray([[2.0, 0.0], [0.0, 4.0]])
    rhs = jnp.asarray([2.0, 8.0])
    result = solve_host_dense_reduced(
        context=HostDenseReducedSolveContext(
            matvec=lambda x: jnp.asarray(a_np) @ x,
            rhs=rhs,
            active_size=2,
            constraint_scheme=0,
            has_fp=False,
            dense_matrix_cache=a_np,
        ),
        x0=jnp.zeros(2),
    )

    assert result.x.tolist() == pytest.approx([1.0, 2.0])
    assert float(result.residual_norm) == pytest.approx(0.0, abs=1.0e-12)


def test_host_dense_reduced_lstsq_handles_rectangular_cache() -> None:
    a_np = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    rhs = jnp.asarray([1.0, 2.0, 3.0])
    result = solve_host_dense_reduced(
        context=HostDenseReducedSolveContext(
            matvec=lambda x: jnp.asarray(a_np) @ x,
            rhs=rhs,
            active_size=2,
            constraint_scheme=2,
            has_fp=False,
            dense_matrix_cache=a_np,
        )
    )

    expected = np.linalg.lstsq(a_np, np.asarray(rhs), rcond=None)[0]
    assert result.x.tolist() == pytest.approx(expected.tolist())
    assert float(result.residual_norm) == pytest.approx(0.0, abs=1.0e-12)


def test_host_dense_full_lu_returns_residual_vector() -> None:
    a_np = np.asarray([[3.0, 0.0], [0.0, 5.0]])
    rhs = jnp.asarray([6.0, 15.0])
    result, residual = solve_host_dense_full(
        context=HostDenseFullSolveContext(
            matvec=lambda x: jnp.asarray(a_np) @ x,
            rhs=rhs,
            total_size=2,
        ),
        x0=jnp.zeros(2),
    )

    assert result.x.tolist() == pytest.approx([2.0, 3.0])
    assert residual.tolist() == pytest.approx([0.0, 0.0])
    assert float(result.residual_norm) == pytest.approx(0.0, abs=1.0e-12)


def test_host_dense_full_lstsq_handles_rectangular_operator(monkeypatch) -> None:
    a_np = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    rhs = jnp.asarray([1.0, 2.0, 3.0])

    def fake_assemble_dense_matrix_from_matvec(**_kwargs):
        return jnp.asarray(a_np)

    monkeypatch.setattr(
        "sfincs_jax.problems.profile_response.dense.assemble_dense_matrix_from_matvec",
        fake_assemble_dense_matrix_from_matvec,
    )
    result, residual = solve_host_dense_full(
        context=HostDenseFullSolveContext(
            matvec=lambda x: jnp.asarray(a_np) @ x,
            rhs=rhs,
            total_size=2,
        )
    )

    expected = np.linalg.lstsq(a_np, np.asarray(rhs), rcond=None)[0]
    assert result.x.tolist() == pytest.approx(expected.tolist())
    assert residual.tolist() == pytest.approx((np.asarray(rhs) - a_np @ expected).tolist())
    assert float(result.residual_norm) == pytest.approx(float(np.linalg.norm(np.asarray(residual))))


def test_rhs1_dense_probe_enabled_from_env_defaults_on_and_respects_false(
    monkeypatch,
) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_DENSE_PROBE", raising=False)
    assert rhs1_dense_probe_enabled_from_env()
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DENSE_PROBE", "off")
    assert not rhs1_dense_probe_enabled_from_env()


def test_rhs1_dense_probe_admission_applies_driver_guards() -> None:
    base = dict(
        probe_enabled=True,
        probe_shortcut=False,
        cs0_petsc_compat=False,
        cs0_sparse_first=False,
        cs0_dense_fallback_allowed=True,
        constraint_scheme=0,
        has_preconditioner=True,
        solve_method_kind="incremental",
    )
    assert rhs1_dense_probe_admission(**base).enabled
    assert not rhs1_dense_probe_admission(**{**base, "probe_enabled": False}).enabled
    assert not rhs1_dense_probe_admission(**{**base, "probe_shortcut": True}).enabled
    assert not rhs1_dense_probe_admission(**{**base, "cs0_petsc_compat": True}).enabled
    assert not rhs1_dense_probe_admission(**{**base, "cs0_sparse_first": True}).enabled
    assert not rhs1_dense_probe_admission(
        **{**base, "cs0_dense_fallback_allowed": False, "constraint_scheme": 0}
    ).enabled
    assert rhs1_dense_probe_admission(
        **{**base, "cs0_dense_fallback_allowed": False, "constraint_scheme": 1}
    ).enabled
    assert not rhs1_dense_probe_admission(**{**base, "has_preconditioner": False}).enabled
    assert not rhs1_dense_probe_admission(**{**base, "solve_method_kind": "dense"}).enabled
    assert not rhs1_dense_probe_admission(**{**base, "solve_method_kind": "dense_ksp"}).enabled


def test_rhs1_dense_probe_shortcut_decision_accepts_or_seeds_probe() -> None:
    decision = rhs1_dense_probe_shortcut_decision(
        dense_shortcut_ratio=10.0,
        probe_ratio=12.0,
        dense_fallback_max=200,
        active_size=100,
        sparse_prefer_over_dense_shortcut=False,
    )
    assert decision.accept_shortcut
    assert not decision.seed_x0_if_missing
    assert decision.messages == ((
        0,
        "solve_v3_full_system_linear_gmres: dense fallback shortcut (probe) "
        "(ratio=1.200e+01 >= 1.0e+01)",
    ),)

    decision = rhs1_dense_probe_shortcut_decision(
        dense_shortcut_ratio=10.0,
        probe_ratio=8.0,
        dense_fallback_max=200,
        active_size=100,
        sparse_prefer_over_dense_shortcut=False,
    )
    assert not decision.accept_shortcut
    assert decision.seed_x0_if_missing
    assert decision.messages == ()


def test_rhs1_dense_probe_shortcut_decision_reports_skip_reasons() -> None:
    decision = rhs1_dense_probe_shortcut_decision(
        dense_shortcut_ratio=10.0,
        probe_ratio=12.0,
        dense_fallback_max=200,
        active_size=100,
        sparse_prefer_over_dense_shortcut=True,
    )
    assert not decision.accept_shortcut
    assert decision.seed_x0_if_missing
    assert decision.messages == ((
        1,
        "solve_v3_full_system_linear_gmres: probe shortcut skipped "
        "(preferring sparse rescue over dense shortcut)",
    ),)

    decision = rhs1_dense_probe_shortcut_decision(
        dense_shortcut_ratio=10.0,
        probe_ratio=12.0,
        dense_fallback_max=50,
        active_size=100,
        sparse_prefer_over_dense_shortcut=False,
    )
    assert not decision.accept_shortcut
    assert decision.seed_x0_if_missing
    assert decision.messages == ((
        1,
        "solve_v3_full_system_linear_gmres: probe shortcut skipped "
        "(size=100 > dense_max=50)",
    ),)


def test_rhs1_reduced_dense_fallback_candidate_host_lu(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DENSE_HOST_LU", "1")
    a_np = np.asarray([[4.0, 1.0], [2.0, 3.0]])
    rhs = jnp.asarray([6.0, 8.0])

    result, elapsed_s = solve_rhs1_reduced_dense_fallback_candidate(
        context=RHS1ReducedDenseFallbackCandidateContext(
            matvec=lambda x: jnp.asarray(a_np) @ x,
            rhs=rhs,
            x0=jnp.zeros(2),
            active_size=2,
            constraint_scheme=2,
            has_fp=False,
            has_pas=False,
            dense_matrix_cache=a_np,
            dense_backend_allowed=True,
            use_implicit=False,
            tol=1.0e-12,
            atol=0.0,
            restart=10,
            maxiter=20,
            gmres_precond_side="left",
            backend="cpu",
        )
    )

    expected = np.linalg.solve(a_np, np.asarray(rhs))
    assert result.x.tolist() == pytest.approx(expected.tolist())
    assert float(result.residual_norm) == pytest.approx(0.0, abs=1.0e-12)
    assert elapsed_s >= 0.0


def test_rhs1_reduced_dense_fallback_candidate_uses_cached_jax_dense(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DENSE_HOST_LU", "0")
    a_np = np.asarray([[2.0, 0.0], [0.0, 5.0]])
    rhs = jnp.asarray([4.0, 15.0])

    result, elapsed_s = solve_rhs1_reduced_dense_fallback_candidate(
        context=RHS1ReducedDenseFallbackCandidateContext(
            matvec=lambda x: jnp.asarray(a_np) @ x,
            rhs=rhs,
            x0=jnp.zeros(2),
            active_size=2,
            constraint_scheme=2,
            has_fp=False,
            has_pas=False,
            dense_matrix_cache=a_np,
            dense_backend_allowed=True,
            use_implicit=False,
            tol=1.0e-12,
            atol=0.0,
            restart=10,
            maxiter=20,
            gmres_precond_side="left",
            backend="cpu",
        )
    )

    assert result.x.tolist() == pytest.approx([2.0, 3.0])
    assert float(result.residual_norm) == pytest.approx(0.0, abs=1.0e-12)
    assert elapsed_s >= 0.0
