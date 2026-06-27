from __future__ import annotations

import math
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from sfincs_jax.problems.profile_residual import (
    apply_damped_preconditioned_residual_polish,
    apply_projected_residual_polish,
    l2_norm_float,
    recompute_true_residual_result,
    replay_left_preconditioned_residual_norms,
    residual_converged,
    residual_target,
    result_with_true_residual,
    RHS1FPPostSolvePolishContext,
    run_rhs1_fp_post_solve_polish,
    safe_ratio,
    true_residual_norm_or_inf,
)
from sfincs_jax.solver import GMRESSolveResult


def test_rhs1_residual_target_matches_petsc_style_gate() -> None:
    assert residual_target(atol=1.0e-12, tol=1.0e-6, rhs_norm=3.0) == 3.0e-6
    assert residual_target(atol=1.0e-4, tol=1.0e-6, rhs_norm=3.0) == 1.0e-4


def test_rhs1_l2_norm_float_and_safe_ratio_are_host_scalars() -> None:
    norm = l2_norm_float(jnp.asarray([3.0, 4.0]))

    assert isinstance(norm, float)
    assert norm == 5.0
    assert safe_ratio(2.0, 4.0) == 0.5
    assert safe_ratio(2.0, 0.0) is None
    assert safe_ratio(math.nan, 4.0) is None
    assert safe_ratio(2.0, math.inf) is None


def test_rhs1_residual_converged_requires_finite_residual_and_target() -> None:
    assert residual_converged(1.0e-8, 1.0e-7) is True
    assert residual_converged(1.0e-6, 1.0e-7) is False
    assert residual_converged(math.nan, 1.0e-7) is False
    assert residual_converged(1.0e-8, math.inf) is False


def test_true_residual_norm_or_inf_returns_finite_norm() -> None:
    norm = true_residual_norm_or_inf(
        rhs=jnp.asarray([1.0, 2.0], dtype=jnp.float64),
        matvec=lambda x: jnp.asarray([x[0], 2.0 * x[1]], dtype=jnp.float64),
        x=jnp.asarray([1.0, -1.0], dtype=jnp.float64),
    )

    assert norm == 4.0


def test_true_residual_norm_or_inf_maps_nonfinite_norm_to_infinity() -> None:
    norm = true_residual_norm_or_inf(
        rhs=jnp.asarray([0.0], dtype=jnp.float64),
        matvec=lambda _x: jnp.asarray([jnp.nan], dtype=jnp.float64),
        x=jnp.asarray([1.0], dtype=jnp.float64),
    )

    assert norm == math.inf


def test_result_with_true_residual_returns_result_and_vector() -> None:
    result, residual = result_with_true_residual(
        x=jnp.asarray([1.0, -1.0], dtype=jnp.float64),
        rhs=jnp.asarray([1.0, 2.0], dtype=jnp.float64),
        matvec=lambda x: jnp.asarray([x[0], 2.0 * x[1]], dtype=jnp.float64),
    )

    assert isinstance(result, GMRESSolveResult)
    assert float(result.residual_norm) == 4.0
    assert jnp.array_equal(residual, jnp.asarray([0.0, 4.0], dtype=jnp.float64))
    assert jnp.array_equal(result.x, jnp.asarray([1.0, -1.0], dtype=jnp.float64))


def test_apply_damped_preconditioned_residual_polish_backtracks_to_improvement() -> None:
    result = GMRESSolveResult(
        x=jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        residual_norm=jnp.asarray(5.0, dtype=jnp.float64),
    )
    rhs = jnp.asarray([3.0, 4.0], dtype=jnp.float64)

    polished, improved = apply_damped_preconditioned_residual_polish(
        current_result=result,
        rhs=rhs,
        matvec=lambda x: 2.0 * x,
        preconditioner=lambda r: r,
        target=1.0e-12,
        steps=1,
        omega=1.5,
        backtrack=3,
    )

    assert improved
    assert float(polished.residual_norm) < 5.0
    assert polished.x.tolist() == [pytest.approx(2.25), pytest.approx(3.0)]


def test_apply_damped_preconditioned_residual_polish_rejects_bad_correction() -> None:
    result = GMRESSolveResult(
        x=jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        residual_norm=jnp.asarray(5.0, dtype=jnp.float64),
    )
    rhs = jnp.asarray([3.0, 4.0], dtype=jnp.float64)

    polished, improved = apply_damped_preconditioned_residual_polish(
        current_result=result,
        rhs=rhs,
        matvec=lambda x: x,
        preconditioner=lambda r: -r,
        target=1.0e-12,
        steps=2,
        omega=1.0,
        backtrack=2,
    )

    assert not improved
    assert polished is result


def test_apply_projected_residual_polish_accepts_safe_projected_correction() -> None:
    result = GMRESSolveResult(
        x=jnp.zeros(3, dtype=jnp.float64),
        residual_norm=jnp.asarray(math.sqrt(105.0), dtype=jnp.float64),
    )
    rhs = jnp.asarray([1.0, 10.0, 2.0], dtype=jnp.float64)

    def solve_linear(**kwargs):
        return GMRESSolveResult(
            x=kwargs["b_vec"],
            residual_norm=jnp.asarray(0.0, dtype=jnp.float64),
        )

    outcome = apply_projected_residual_polish(
        current_result=result,
        rhs=rhs,
        matvec=lambda x: x,
        projected_indices=jnp.asarray([0, 2], dtype=jnp.int32),
        active_size=3,
        solve_linear=solve_linear,
        preconditioner=lambda r: r,
        tol=1.0e-12,
        restart=5,
        maxiter=5,
        precond_side="left",
        target=1.0e-12,
        threshold_ratio=1.0,
        abs_threshold=0.0,
        full_accept_ratio=1.2,
        require_full_improvement=True,
    )

    assert outcome.accepted
    assert outcome.result.x.tolist() == pytest.approx([1.0, 0.0, 2.0])
    assert outcome.projected_residual_after == pytest.approx(0.0)
    assert float(outcome.result.residual_norm) == pytest.approx(10.0)


def test_apply_projected_residual_polish_rejects_full_residual_regression() -> None:
    result = GMRESSolveResult(
        x=jnp.zeros(3, dtype=jnp.float64),
        residual_norm=jnp.asarray(math.sqrt(105.0), dtype=jnp.float64),
    )
    rhs = jnp.asarray([1.0, 10.0, 2.0], dtype=jnp.float64)

    def solve_linear(**_kwargs):
        return GMRESSolveResult(
            x=jnp.asarray([20.0, 30.0], dtype=jnp.float64),
            residual_norm=jnp.asarray(0.0, dtype=jnp.float64),
        )

    outcome = apply_projected_residual_polish(
        current_result=result,
        rhs=rhs,
        matvec=lambda x: x,
        projected_indices=jnp.asarray([0, 2], dtype=jnp.int32),
        active_size=3,
        solve_linear=solve_linear,
        preconditioner=lambda r: r,
        tol=1.0e-12,
        restart=5,
        maxiter=5,
        precond_side="left",
        target=1.0e-12,
        threshold_ratio=1.0,
        abs_threshold=0.0,
        full_accept_ratio=1.2,
        require_full_improvement=False,
    )

    assert not outcome.accepted
    assert outcome.result is result
    assert outcome.full_residual_after is not None
    assert outcome.full_residual_after > outcome.full_residual_before


def test_recompute_true_residual_result_replaces_reported_krylov_norm() -> None:
    result = GMRESSolveResult(
        x=jnp.asarray([1.0, -1.0], dtype=jnp.float64),
        residual_norm=jnp.asarray(99.0, dtype=jnp.float64),
    )

    updated, residual_vec, residual_norm = recompute_true_residual_result(
        result=result,
        rhs=jnp.asarray([1.0, 2.0], dtype=jnp.float64),
        matvec=lambda x: jnp.asarray([x[0], 2.0 * x[1]], dtype=jnp.float64),
        residual_vec=None,
        update_residual_vec=False,
    )

    assert updated is not result
    assert float(updated.residual_norm) == 4.0
    assert residual_vec is None
    assert residual_norm == 4.0


def test_recompute_true_residual_result_can_keep_computed_residual_vector() -> None:
    result = GMRESSolveResult(
        x=jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        residual_norm=jnp.asarray(99.0, dtype=jnp.float64),
    )
    supplied_residual = jnp.asarray([3.0, 4.0], dtype=jnp.float64)

    updated, residual_vec, residual_norm = recompute_true_residual_result(
        result=result,
        rhs=jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        matvec=lambda _x: (_ for _ in ()).throw(RuntimeError("should not be called")),
        residual_vec=supplied_residual,
        update_residual_vec=True,
    )

    assert float(updated.residual_norm) == 5.0
    assert residual_vec is not None
    assert jnp.array_equal(residual_vec, supplied_residual)
    assert residual_norm == 5.0


def test_recompute_true_residual_result_keeps_incumbent_on_nonfinite_true_norm() -> None:
    result = GMRESSolveResult(
        x=jnp.asarray([1.0], dtype=jnp.float64),
        residual_norm=jnp.asarray(7.0, dtype=jnp.float64),
    )

    updated, residual_vec, residual_norm = recompute_true_residual_result(
        result=result,
        rhs=jnp.asarray([0.0], dtype=jnp.float64),
        matvec=lambda _x: jnp.asarray([jnp.nan], dtype=jnp.float64),
        residual_vec=None,
        update_residual_vec=True,
    )

    assert updated is result
    assert residual_vec is None
    assert residual_norm == 7.0


def test_replay_left_preconditioned_residual_norms_noops_without_left_preconditioner() -> None:
    result = GMRESSolveResult(
        x=jnp.asarray([1.0], dtype=jnp.float64),
        residual_norm=jnp.asarray(7.0, dtype=jnp.float64),
    )

    residual_vec, true_norm, check_norm = replay_left_preconditioned_residual_norms(
        result=result,
        rhs=jnp.asarray([0.0], dtype=jnp.float64),
        matvec=lambda _x: (_ for _ in ()).throw(RuntimeError("should not be called")),
        residual_vec=None,
        preconditioner=None,
        precondition_side="none",
        update_residual_vec=True,
    )

    assert residual_vec is None
    assert true_norm == 7.0
    assert check_norm == 7.0


def test_replay_left_preconditioned_residual_norms_reports_true_and_preconditioned_norms() -> None:
    result = GMRESSolveResult(
        x=jnp.asarray([1.0, -1.0], dtype=jnp.float64),
        residual_norm=jnp.asarray(99.0, dtype=jnp.float64),
    )

    residual_vec, true_norm, check_norm = replay_left_preconditioned_residual_norms(
        result=result,
        rhs=jnp.asarray([1.0, 2.0], dtype=jnp.float64),
        matvec=lambda x: jnp.asarray([x[0], 2.0 * x[1]], dtype=jnp.float64),
        residual_vec=None,
        preconditioner=lambda r: 0.5 * r,
        precondition_side="left",
        update_residual_vec=True,
    )

    assert residual_vec is not None
    assert jnp.array_equal(residual_vec, jnp.asarray([0.0, 4.0], dtype=jnp.float64))
    assert true_norm == 4.0
    assert check_norm == 2.0


def test_replay_left_preconditioned_residual_norms_keeps_supplied_residual_without_update() -> None:
    result = GMRESSolveResult(
        x=jnp.asarray([1.0, -1.0], dtype=jnp.float64),
        residual_norm=jnp.asarray(99.0, dtype=jnp.float64),
    )
    supplied_residual = jnp.asarray([3.0, 4.0], dtype=jnp.float64)

    residual_vec, true_norm, check_norm = replay_left_preconditioned_residual_norms(
        result=result,
        rhs=jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        matvec=lambda _x: (_ for _ in ()).throw(RuntimeError("should not be called")),
        residual_vec=supplied_residual,
        preconditioner=lambda r: 2.0 * r,
        precondition_side="left",
        update_residual_vec=False,
    )

    assert residual_vec is supplied_residual
    assert true_norm == 5.0
    assert check_norm == 10.0


def test_replay_left_preconditioned_residual_norms_preserves_current_check_when_preconditioned_nonfinite() -> None:
    result = GMRESSolveResult(
        x=jnp.asarray([1.0], dtype=jnp.float64),
        residual_norm=jnp.asarray(7.0, dtype=jnp.float64),
    )

    residual_vec, true_norm, check_norm = replay_left_preconditioned_residual_norms(
        result=result,
        rhs=jnp.asarray([0.0], dtype=jnp.float64),
        matvec=lambda _x: jnp.asarray([jnp.nan], dtype=jnp.float64),
        residual_vec=None,
        preconditioner=lambda r: r,
        precondition_side="left",
        update_residual_vec=True,
    )

    assert residual_vec is not None
    assert true_norm == math.inf
    assert check_norm == 7.0


def _tiny_fp_op() -> SimpleNamespace:
    return SimpleNamespace(
        rhs_mode=1,
        include_phi1=False,
        fblock=SimpleNamespace(
            fp=object(),
            pas=None,
            collisionless=SimpleNamespace(n_xi_for_x=np.asarray([3], dtype=np.int32)),
        ),
        n_species=1,
        n_x=1,
        n_xi=3,
        n_theta=1,
        n_zeta=1,
    )


def _fp_post_solve_context(
    *,
    result: GMRESSolveResult,
    rhs: jnp.ndarray,
    matvec,
    preconditioner=None,
    target: float = 1.0e-12,
    residual_controls=None,
    low_l_controls=None,
    l1_controls=None,
    global_controls=None,
    bicgstab_controls=None,
    targeted=True,
    solve_linear=None,
    emit=None,
) -> RHS1FPPostSolvePolishContext:
    def _raise_collision(**_kwargs):
        raise AssertionError("collision preconditioner should not be built")

    def _raise_lmax(**_kwargs):
        raise AssertionError("low-L preconditioner should not be built")

    def _raise_solve(**_kwargs):
        raise AssertionError("Krylov polish should not be called")

    return RHS1FPPostSolvePolishContext(
        op=_tiny_fp_op(),
        result=result,
        rhs=rhs,
        matvec=matvec,
        preconditioner=preconditioner,
        active_size=int(rhs.size),
        target=float(target),
        tol=1.0e-10,
        atol=0.0,
        restart=5,
        maxiter=5,
        precondition_side="left",
        rhs1_precond_kind="xmg",
        use_implicit=False,
        use_active_dof_mode=False,
        full_to_active=None,
        reduce_full=None,
        expand_reduced=None,
        read_residual_controls=lambda: residual_controls
        or SimpleNamespace(min_size=999, steps=0, hybrid=False, omega=1.0, backtrack=0),
        read_low_l_controls=lambda **_kwargs: low_l_controls
        or SimpleNamespace(lmax_default=0, block_max=0, restart=5, maxiter=5),
        read_l1_controls=lambda: l1_controls
        or SimpleNamespace(
            enabled=False,
            restart=5,
            maxiter=5,
            ratio=1.0,
            abs_threshold=0.0,
            tol=1.0e-10,
            full_accept_ratio=1.2,
        ),
        read_global_low_l_controls=lambda **_kwargs: global_controls
        or SimpleNamespace(
            enabled=False,
            lmax=0,
            max_size=0,
            ratio=1.0,
            restart=5,
            maxiter=5,
            abs_threshold=0.0,
            full_accept_ratio=1.2,
            tol=1.0e-10,
            threshold_ratio=1.0,
        ),
        read_bicgstab_controls=lambda **_kwargs: bicgstab_controls
        or SimpleNamespace(
            enabled=False,
            min_size=1,
            maxiter=5,
            tol=1.0e-10,
            atol=0.0,
        ),
        targeted_polish_allowed=lambda **_kwargs: bool(targeted),
        build_collision_preconditioner=_raise_collision,
        build_lmax_preconditioner=_raise_lmax,
        pitch_mode_active_indices=lambda **kwargs: np.asarray(
            [idx for idx in range(int(kwargs["l_min"]), int(kwargs["l_max"]) + 1)],
            dtype=np.int32,
        ),
        solve_linear=solve_linear or _raise_solve,
        emit=emit,
        label="test_rhs1",
    )


def test_rhs1_fp_post_solve_polish_noops_when_gate_is_closed() -> None:
    result = GMRESSolveResult(
        x=jnp.zeros(3, dtype=jnp.float64),
        residual_norm=jnp.asarray(3.0, dtype=jnp.float64),
    )

    polished = run_rhs1_fp_post_solve_polish(
        _fp_post_solve_context(
            result=result,
            rhs=jnp.ones(3, dtype=jnp.float64),
            matvec=lambda x: x,
            preconditioner=None,
        )
    )

    assert polished is result


def test_rhs1_fp_post_solve_polish_accepts_damped_residual_correction() -> None:
    messages: list[str] = []
    result = GMRESSolveResult(
        x=jnp.zeros(2, dtype=jnp.float64),
        residual_norm=jnp.asarray(5.0, dtype=jnp.float64),
    )

    polished = run_rhs1_fp_post_solve_polish(
        _fp_post_solve_context(
            result=result,
            rhs=jnp.asarray([3.0, 4.0], dtype=jnp.float64),
            matvec=lambda x: 2.0 * x,
            preconditioner=lambda r: r,
            target=1.0e-12,
            residual_controls=SimpleNamespace(
                min_size=1,
                steps=1,
                hybrid=False,
                omega=1.0,
                backtrack=2,
            ),
            targeted=False,
            emit=lambda _level, message: messages.append(message),
        )
    )

    assert polished is not result
    assert float(polished.residual_norm) < float(result.residual_norm)
    assert any("FP polish improved residual" in message for message in messages)


def test_rhs1_fp_post_solve_polish_accepts_projected_l1_correction() -> None:
    result = GMRESSolveResult(
        x=jnp.zeros(3, dtype=jnp.float64),
        residual_norm=jnp.asarray(2.0, dtype=jnp.float64),
    )

    def solve_linear(**kwargs):
        assert kwargs["solve_method_val"] == "incremental"
        return GMRESSolveResult(
            x=kwargs["b_vec"],
            residual_norm=jnp.asarray(0.0, dtype=jnp.float64),
        )

    polished = run_rhs1_fp_post_solve_polish(
        _fp_post_solve_context(
            result=result,
            rhs=jnp.asarray([0.0, 2.0, 0.0], dtype=jnp.float64),
            matvec=lambda x: x,
            preconditioner=lambda r: r,
            target=1.0e-12,
            l1_controls=SimpleNamespace(
                enabled=True,
                restart=5,
                maxiter=5,
                ratio=1.0,
                abs_threshold=0.0,
                tol=1.0e-10,
                full_accept_ratio=1.2,
            ),
            solve_linear=solve_linear,
        )
    )

    assert polished is not result
    assert polished.x.tolist() == pytest.approx([0.0, 2.0, 0.0])
    assert float(polished.residual_norm) == pytest.approx(0.0)


def test_rhs1_fp_post_solve_polish_can_run_bicgstab_after_global_gate() -> None:
    calls: list[str] = []
    result = GMRESSolveResult(
        x=jnp.zeros(3, dtype=jnp.float64),
        residual_norm=jnp.asarray(10.0, dtype=jnp.float64),
    )

    def solve_linear(**kwargs):
        calls.append(kwargs["solve_method_val"])
        return GMRESSolveResult(
            x=jnp.ones(3, dtype=jnp.float64),
            residual_norm=jnp.asarray(1.0, dtype=jnp.float64),
        )

    polished = run_rhs1_fp_post_solve_polish(
        _fp_post_solve_context(
            result=result,
            rhs=jnp.ones(3, dtype=jnp.float64),
            matvec=lambda x: x,
            preconditioner=lambda r: r,
            target=1.0e-12,
            global_controls=SimpleNamespace(
                enabled=True,
                lmax=0,
                max_size=0,
                ratio=1.0,
                restart=5,
                maxiter=5,
                abs_threshold=0.0,
                full_accept_ratio=1.2,
                tol=1.0e-10,
                threshold_ratio=1.0,
            ),
            bicgstab_controls=SimpleNamespace(
                enabled=True,
                min_size=1,
                maxiter=7,
                tol=1.0e-10,
                atol=0.0,
            ),
            solve_linear=solve_linear,
        )
    )

    assert calls == ["bicgstab"]
    assert polished is not result
    assert float(polished.residual_norm) == pytest.approx(1.0)
