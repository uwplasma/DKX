from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp

from sfincs_jax.problems.profile_dense import (
    ProfileLinearSolveContext,
    RHS1Constraint0PETScCompatSolveContext,
    RHS1DenseKSPFullSolveContext,
    RHS1DenseKSPReducedSolveContext,
    RHS1ScipyRescueContext,
    RHS1ScipyRescueStageContext,
    build_profile_linear_solve_dispatch,
    profile_solver_kind,
    rhs1_small_gmres_max_from_env,
    run_rhs1_scipy_rescue,
    run_rhs1_scipy_rescue_stage,
    solve_rhs1_constraint0_petsc_compat,
    solve_rhs1_dense_ksp_full,
    solve_rhs1_dense_ksp_reduced,
    solve_profile_linear,
    solve_profile_linear_with_residual,
)
from sfincs_jax.solver import GMRESSolveResult, assemble_dense_matrix_from_matvec


def _context(
    *,
    rhs_mode: int = 1,
    total_size: int = 1000,
    use_implicit: bool = False,
    use_solver_jit: bool = False,
    distributed_axis: str | None = None,
    distributed_auto_solver: str = "bicgstab",
    small_gmres_max: int = 600,
) -> ProfileLinearSolveContext:
    return ProfileLinearSolveContext(
        rhs_mode=rhs_mode,
        total_size=total_size,
        use_implicit=use_implicit,
        use_solver_jit=use_solver_jit,
        distributed_axis=distributed_axis,
        distributed_auto_solver=distributed_auto_solver,
        small_gmres_max=small_gmres_max,
    )


def test_profile_solver_kind_preserves_rhs1_auto_defaults() -> None:
    assert profile_solver_kind("auto", context=_context(total_size=100)) == ("gmres", "incremental")
    assert profile_solver_kind("default", context=_context(total_size=10000)) == ("gmres", "incremental")
    assert profile_solver_kind("bicgstab", context=_context()) == ("bicgstab", "batched")
    assert profile_solver_kind("lgmres", context=_context()) == ("gmres", "lgmres")


def test_profile_solver_kind_prefers_bicgstab_for_distributed_auto() -> None:
    context = _context(distributed_axis="theta", distributed_auto_solver="bicgstab")

    assert profile_solver_kind("auto", context=context) == ("bicgstab", "batched")


def test_profile_linear_solve_dispatch_builds_context_and_solves_tiny_system() -> None:
    dispatch = build_profile_linear_solve_dispatch(
        rhs_mode=1,
        total_size=2,
        use_implicit=False,
        use_solver_jit=False,
        distributed_axis=None,
        distributed_auto_solver="bicgstab",
        small_gmres_max=10,
    )
    b = jnp.asarray([2.0, -3.0], dtype=jnp.float64)

    result = dispatch.solve(
        matvec_fn=lambda x: x,
        b_vec=b,
        precond_fn=None,
        x0_vec=None,
        tol_val=1.0e-12,
        atol_val=1.0e-12,
        restart_val=4,
        maxiter_val=8,
        solve_method_val="auto",
        precond_side="left",
    )
    residual_result, residual = dispatch.solve_with_residual(
        matvec_fn=lambda x: x,
        b_vec=b,
        precond_fn=None,
        x0_vec=None,
        tol_val=1.0e-12,
        atol_val=1.0e-12,
        restart_val=4,
        maxiter_val=8,
        solve_method_val="auto",
        precond_side="left",
    )

    assert dispatch.solver_kind("auto") == ("gmres", "incremental")
    assert jnp.linalg.norm(result.x - b) < 1.0e-10
    assert jnp.linalg.norm(residual_result.x - b) < 1.0e-10
    assert jnp.linalg.norm(residual) < 1.0e-10


def test_rhs1_small_gmres_max_env_preserves_defaults(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_GMRES_SMALL_MAX", raising=False)
    assert rhs1_small_gmres_max_from_env() == 600

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_GMRES_SMALL_MAX", "42")
    assert rhs1_small_gmres_max_from_env() == 42

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_GMRES_SMALL_MAX", "bad")
    assert rhs1_small_gmres_max_from_env(default=17) == 17


def test_solve_profile_linear_solves_tiny_identity_system() -> None:
    b = jnp.asarray([1.0, -2.0], dtype=jnp.float64)

    result = solve_profile_linear(
        context=_context(total_size=2, small_gmres_max=10),
        matvec_fn=lambda x: x,
        b_vec=b,
        precond_fn=None,
        x0_vec=None,
        tol_val=1.0e-12,
        atol_val=1.0e-12,
        restart_val=4,
        maxiter_val=8,
        solve_method_val="auto",
        precond_side="left",
    )

    assert jnp.linalg.norm(result.x - b) < 1.0e-10


def test_solve_profile_linear_with_residual_solves_tiny_identity_system() -> None:
    b = jnp.asarray([1.0, -2.0], dtype=jnp.float64)

    result, residual = solve_profile_linear_with_residual(
        context=_context(total_size=2, small_gmres_max=10),
        matvec_fn=lambda x: x,
        b_vec=b,
        precond_fn=None,
        x0_vec=None,
        tol_val=1.0e-12,
        atol_val=1.0e-12,
        restart_val=4,
        maxiter_val=8,
        solve_method_val="auto",
        precond_side="left",
    )

    assert jnp.linalg.norm(result.x - b) < 1.0e-10
    assert jnp.linalg.norm(residual) < 1.0e-10


def test_run_rhs1_scipy_rescue_gmres_recomputes_true_residual() -> None:
    a = jnp.asarray([[4.0, 1.0], [1.0, 3.0]], dtype=jnp.float64)
    b = jnp.asarray([1.0, 2.0], dtype=jnp.float64)

    outcome = run_rhs1_scipy_rescue(
        context=RHS1ScipyRescueContext(
            matvec=lambda x: a @ x,
            rhs=b,
            x0=jnp.zeros_like(b),
            preconditioner=None,
            method="gmres",
            tol=1.0e-12,
            atol=1.0e-12,
            restart=4,
            maxiter=12,
            precond_side="left",
        )
    )

    assert jnp.linalg.norm(a @ outcome.result.x - b) < 1.0e-10
    assert jnp.linalg.norm(outcome.residual_vec) < 1.0e-10
    assert outcome.reported_residual < 1.0e-10
    assert outcome.history_len >= 1
    assert outcome.preconditioned_residual is None


def test_run_rhs1_scipy_rescue_bicgstab_recomputes_true_residual() -> None:
    a = jnp.asarray([[5.0, 0.5], [0.25, 2.0]], dtype=jnp.float64)
    b = jnp.asarray([2.0, -1.0], dtype=jnp.float64)

    outcome = run_rhs1_scipy_rescue(
        context=RHS1ScipyRescueContext(
            matvec=lambda x: a @ x,
            rhs=b,
            x0=jnp.zeros_like(b),
            preconditioner=None,
            method="bicgstab",
            tol=1.0e-12,
            atol=1.0e-12,
            restart=4,
            maxiter=20,
            precond_side="left",
        )
    )

    assert jnp.linalg.norm(a @ outcome.result.x - b) < 1.0e-10
    assert jnp.linalg.norm(outcome.residual_vec) < 1.0e-10
    assert outcome.reported_residual < 1.0e-10
    assert outcome.history_len >= 1


def test_run_rhs1_scipy_rescue_stage_accepts_improving_cpu_rescue(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SCIPY_GMRES_RESCUE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SCIPY_GMRES_RESCUE_RATIO", "1")
    rhs = jnp.asarray([1.0, -2.0], dtype=jnp.float64)
    current = GMRESSolveResult(
        x=jnp.zeros_like(rhs),
        residual_norm=jnp.asarray(10.0, dtype=jnp.float64),
    )
    residual_vec = jnp.asarray([10.0, 0.0], dtype=jnp.float64)
    marks: list[str] = []

    stage = run_rhs1_scipy_rescue_stage(
        RHS1ScipyRescueStageContext(
            op=SimpleNamespace(
                rhs_mode=1,
                include_phi1=False,
                fblock=SimpleNamespace(fp=object(), pas=None),
            ),
            result=current,
            residual_vec=residual_vec,
            matvec=lambda x: x,
            rhs=rhs,
            preconditioner=None,
            strong_preconditioner=None,
            preconditioner_name="none",
            strong_preconditioner_name="strong",
            target=1.0e-6,
            tol=1.0e-12,
            atol=1.0e-12,
            restart=4,
            maxiter=20,
            precond_side="left",
            active_size=2,
            used_large_cpu_xblock_shortcut=False,
            used_explicit_fp_xblock_seed=False,
            use_implicit=False,
            skip_global_sparse_after_xblock=False,
            elapsed_s=lambda: 1.0,
            emit=None,
            mark=marks.append,
        )
    )

    assert jnp.linalg.norm(stage.result.x - rhs) < 1.0e-10
    assert float(stage.result.residual_norm) < 1.0e-10
    assert stage.residual_vec is residual_vec
    assert marks == ["rhs1_scipy_rescue_start", "rhs1_scipy_rescue_done"]
    assert stage.metadata["scipy_rescue_attempted"] is True
    assert stage.metadata["scipy_rescue_improved"] is True
    assert stage.metadata["scipy_rescue_method"] == "gmres"


def test_run_rhs1_scipy_rescue_stage_records_active_size_cap_skip(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SCIPY_GMRES_RESCUE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SCIPY_GMRES_RESCUE_MAX_ACTIVE", "10")
    current = GMRESSolveResult(
        x=jnp.asarray([0.0], dtype=jnp.float64),
        residual_norm=jnp.asarray(1.0, dtype=jnp.float64),
    )
    marks: list[str] = []

    stage = run_rhs1_scipy_rescue_stage(
        RHS1ScipyRescueStageContext(
            op=SimpleNamespace(
                rhs_mode=1,
                include_phi1=False,
                fblock=SimpleNamespace(fp=object(), pas=None),
            ),
            result=current,
            residual_vec=None,
            matvec=lambda x: x,
            rhs=jnp.asarray([1.0], dtype=jnp.float64),
            preconditioner=None,
            strong_preconditioner=None,
            preconditioner_name="none",
            strong_preconditioner_name="strong",
            target=1.0e-12,
            tol=1.0e-12,
            atol=1.0e-12,
            restart=4,
            maxiter=20,
            precond_side="left",
            active_size=11,
            used_large_cpu_xblock_shortcut=True,
            used_explicit_fp_xblock_seed=False,
            use_implicit=False,
            skip_global_sparse_after_xblock=False,
            elapsed_s=lambda: 0.0,
            emit=None,
            mark=marks.append,
        )
    )

    assert stage.result is current
    assert stage.residual_vec is None
    assert marks == ["rhs1_scipy_rescue_skipped"]
    assert stage.metadata == {
        "scipy_rescue_attempted": False,
        "scipy_rescue_skipped": True,
        "scipy_rescue_skip_reason": "active_size_cap",
        "scipy_rescue_initial_residual": 1.0,
        "scipy_rescue_target": 1.0e-12,
        "scipy_rescue_threshold": 1.0e-9,
        "scipy_rescue_active_size": 11,
        "scipy_rescue_used_large_cpu_xblock_shortcut": True,
        "scipy_rescue_used_explicit_fp_xblock_seed": False,
    }


def test_solve_rhs1_constraint0_petsc_compat_records_preconditioned_replay() -> None:
    a = jnp.asarray(
        [
            [4.0, 0.2, 0.0],
            [0.1, 3.0, 0.4],
            [0.0, 0.3, 2.0],
        ],
        dtype=jnp.float64,
    )
    rhs = jnp.asarray([1.0, -2.0, 0.5], dtype=jnp.float64)
    messages: list[str] = []

    outcome = solve_rhs1_constraint0_petsc_compat(
        RHS1Constraint0PETScCompatSolveContext(
            matvec=lambda x: a @ x,
            rhs=rhs,
            x0=None,
            active_size=3,
            tol=1.0e-12,
            atol=1.0e-12,
            sparse_drop_tol=0.0,
            sparse_drop_rel=0.0,
            config=SimpleNamespace(
                drop_tol=0.0,
                fill=10.0,
                diag_pivot=0.0,
                restart=4,
                maxiter=12,
            ),
            regularization=lambda _max_abs: 0.0,
        ),
        emit=lambda _level, message: messages.append(message),
    )

    assert jnp.linalg.norm(a @ outcome.result.x - rhs) < 1.0e-10
    assert jnp.linalg.norm(outcome.replay_matvec(outcome.result.x) - outcome.replay_rhs) < 1.0e-10
    assert outcome.true_residual < 1.0e-10
    assert outcome.preconditioned_residual < 1.0e-10
    assert outcome.rhs_pc_norm > 0.0
    assert outcome.drop_threshold == 0.0
    assert outcome.regularization == 0.0
    assert outcome.nnz == 7
    assert any("constraintScheme=0 PETSc-compat sparse ILU solve" in message for message in messages)
    assert any("constraintScheme=0 PETSc-compat residuals" in message for message in messages)


def test_solve_rhs1_dense_ksp_full_solves_species_block_system() -> None:
    a = jnp.asarray(
        [
            [4.0, 0.2, 0.0, 0.0, 0.1, 0.0],
            [0.1, 3.0, 0.0, 0.0, 0.0, 0.2],
            [0.0, 0.0, 5.0, 0.3, 0.1, 0.0],
            [0.0, 0.0, 0.2, 2.5, 0.0, 0.1],
            [0.1, 0.0, 0.2, 0.0, 1.7, 0.0],
            [0.0, 0.2, 0.0, 0.1, 0.0, 1.9],
        ],
        dtype=jnp.float64,
    )
    rhs = jnp.asarray([1.0, -0.5, 0.25, 2.0, -1.0, 0.75], dtype=jnp.float64)
    captured: dict[str, object] = {}

    def solve_linear(**kwargs) -> GMRESSolveResult:
        captured.update(kwargs)
        b_vec = kwargs["b_vec"]
        mat = assemble_dense_matrix_from_matvec(
            matvec=kwargs["matvec_fn"],
            n=int(b_vec.shape[0]),
            dtype=b_vec.dtype,
        )
        x = jnp.linalg.solve(mat, b_vec)
        return GMRESSolveResult(
            x=x,
            residual_norm=jnp.linalg.norm(mat @ x - b_vec),
        )

    outcome = solve_rhs1_dense_ksp_full(
        RHS1DenseKSPFullSolveContext(
            matvec=lambda x: a @ x,
            rhs=rhs,
            x0=None,
            total_size=6,
            phi1_size=0,
            n_species=2,
            n_theta=1,
            n_zeta=1,
            nxi_for_x=jnp.asarray([2]),
            extra_size=2,
            tol=1.0e-12,
            atol=1.0e-12,
            restart=6,
            maxiter=20,
            solve_linear=solve_linear,
        )
    )

    assert jnp.linalg.norm(a @ outcome.result.x - rhs) < 1.0e-10
    assert outcome.result.residual_norm < 1.0e-10
    assert jnp.linalg.norm(outcome.replay_matvec(outcome.result.x) - outcome.replay_rhs) < 1.0e-10
    assert captured["precond_fn"] is None
    assert captured["solve_method_val"] == "incremental"
    assert captured["precond_side"] == "none"


def test_solve_rhs1_dense_ksp_reduced_preserves_preconditioned_result_ready() -> None:
    a = jnp.asarray(
        [
            [4.0, 0.2, 0.0, 0.0, 0.1, 0.0],
            [0.1, 3.0, 0.0, 0.0, 0.0, 0.2],
            [0.0, 0.0, 5.0, 0.3, 0.1, 0.0],
            [0.0, 0.0, 0.2, 2.5, 0.0, 0.1],
            [0.1, 0.0, 0.2, 0.0, 1.7, 0.0],
            [0.0, 0.2, 0.0, 0.1, 0.0, 1.9],
        ],
        dtype=jnp.float64,
    )
    rhs = jnp.asarray([1.0, -0.5, 0.25, 2.0, -1.0, 0.75], dtype=jnp.float64)
    captured: dict[str, object] = {}
    ready_calls: list[GMRESSolveResult] = []

    def solve_linear(**kwargs) -> GMRESSolveResult:
        captured.update(kwargs)
        b_vec = kwargs["b_vec"]
        mat = assemble_dense_matrix_from_matvec(
            matvec=kwargs["matvec_fn"],
            n=int(b_vec.shape[0]),
            dtype=b_vec.dtype,
        )
        x = jnp.linalg.solve(mat, b_vec)
        return GMRESSolveResult(
            x=x,
            residual_norm=jnp.linalg.norm(mat @ x - b_vec),
        )

    def result_ready(result: GMRESSolveResult) -> GMRESSolveResult:
        ready_calls.append(result)
        return GMRESSolveResult(
            x=result.x,
            residual_norm=result.residual_norm + jnp.asarray(0.0, dtype=jnp.float64),
        )

    outcome = solve_rhs1_dense_ksp_reduced(
        RHS1DenseKSPReducedSolveContext(
            matvec=lambda x: a @ x,
            rhs=rhs,
            x0=None,
            active_size=6,
            phi1_size=0,
            n_species=2,
            n_theta=1,
            n_zeta=1,
            nxi_for_x=jnp.asarray([2]),
            extra_size=2,
            tol=1.0e-12,
            atol=1.0e-12,
            restart=6,
            maxiter=20,
            solve_linear=solve_linear,
            result_ready=result_ready,
        )
    )

    assert len(ready_calls) == 1
    assert jnp.linalg.norm(a @ outcome.result.x - rhs) < 1.0e-10
    assert jnp.linalg.norm(outcome.replay_matvec(outcome.result.x) - outcome.replay_rhs) < 1.0e-10
    assert captured["precond_fn"] is None
    assert captured["solve_method_val"] == "incremental"
    assert captured["precond_side"] == "none"
