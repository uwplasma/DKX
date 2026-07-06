from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp

from sfincs_jax.problems.profile_solver_diagnostics import (
    ProfileResponseLinearFinalizationContext,
    V3LinearSolveResult,
    finalize_profile_response_linear_solve,
    profile_response_post_xblock_accept_floor,
)
from sfincs_jax.problems.profile_solver_diagnostics import RHS1KSPDiagnosticsContext
from sfincs_jax.solver import GMRESSolveResult


def _ksp_context(emit=None) -> RHS1KSPDiagnosticsContext:
    return RHS1KSPDiagnosticsContext(
        emit=emit,
        fortran_stdout=False,
        history_max_size=None,
        history_max_iter=None,
        iter_stats_enabled=False,
        iter_stats_max_size=None,
    )


def _rhs1_op() -> SimpleNamespace:
    return SimpleNamespace(
        rhs_mode=1,
        constraint_scheme=2,
        extra_size=0,
        include_phi1=False,
    )


def test_post_xblock_accept_floor_skips_non_rhs1(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SCIPY_GMRES_RESCUE_ABS", "1e-3")
    op = SimpleNamespace(rhs_mode=2)

    floor = profile_response_post_xblock_accept_floor(
        op=op,
        active_size=10,
        used_large_cpu_xblock_shortcut=True,
        used_explicit_fp_xblock_seed=True,
        use_implicit=False,
        backend="cpu",
    )

    assert floor == 0.0


def test_finalize_profile_response_linear_solve_metadata_and_progress(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SCIPY_GMRES_RESCUE_ABS", "1e-3")
    messages: list[tuple[int, str]] = []

    result = finalize_profile_response_linear_solve(
        ProfileResponseLinearFinalizationContext(
            op=_rhs1_op(),
            rhs=jnp.asarray([1.0, 0.0]),
            result=GMRESSolveResult(
                x=jnp.asarray([2.0, 3.0]),
                residual_norm=jnp.asarray(5.0e-4),
            ),
            residual_vec=None,
            ksp_replay=SimpleNamespace(),
            ksp_diagnostics_context=_ksp_context(),
            tol=1.0e-12,
            atol=1.0e-12,
            solve_method="gmres",
            active_size=100,
            used_large_cpu_xblock_shortcut=False,
            used_explicit_fp_xblock_seed=False,
            use_implicit=False,
            backend="cpu",
            metadata_parts=(
                {"path": "unit"},
                {
                    "extra": 7,
                    "solver_path": "host_dense_shortcut",
                    "solver_kind": "host_dense_lu",
                    "host_dense_shortcut": True,
                },
            ),
            emit=lambda level, message: messages.append((int(level), str(message))),
            elapsed_s=lambda: 12.34567,
        )
    )

    assert isinstance(result, V3LinearSolveResult)
    assert result.metadata is not None
    assert result.metadata["path"] == "unit"
    assert result.metadata["extra"] == 7
    assert result.metadata["solver_path"] == "host_dense_shortcut"
    assert result.metadata["solver_kind"] == "host_dense_lu"
    assert result.metadata["host_dense_shortcut"] is True
    assert result.metadata["accepted_converged"] is True
    assert result.metadata["acceptance_criterion"] == "post_xblock_abs_floor"
    assert result.metadata["true_residual_converged"] is False
    assert result.metadata["accepted_residual_floor"] == 1.0e-3
    assert messages == [
        (0, "solve_v3_full_system_linear_gmres: residual_norm=5.000000e-04"),
        (1, "solve_v3_full_system_linear_gmres: elapsed_s=12.346"),
    ]


def test_finalize_profile_response_linear_solve_omits_empty_metadata() -> None:
    result = finalize_profile_response_linear_solve(
        ProfileResponseLinearFinalizationContext(
            op=_rhs1_op(),
            rhs=jnp.asarray([1.0]),
            result=GMRESSolveResult(
                x=jnp.asarray([1.0]),
                residual_norm=jnp.asarray(0.0),
            ),
            residual_vec=None,
            ksp_replay=SimpleNamespace(),
            ksp_diagnostics_context=_ksp_context(),
            tol=1.0e-8,
            atol=1.0e-8,
            solve_method="gmres",
            active_size=1,
            used_large_cpu_xblock_shortcut=False,
            used_explicit_fp_xblock_seed=False,
            use_implicit=False,
            backend="cpu",
            metadata_parts=(),
        )
    )

    assert result.metadata is None
