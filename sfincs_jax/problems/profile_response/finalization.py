"""Final RHSMode=1 linear-solve handoff.

This module owns the last step of a profile-response solve: cleanup projection,
optional KSP replay diagnostics, final progress lines, branch-acceptance
metadata, and construction of the typed v3-compatible result.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from ...solver import GMRESSolveResult
from ...v3_results import V3LinearSolveResult
from .active_projection import finalize_rhs1_linear_solution_cleanup
from .policies import rhs1_scipy_rescue_abs_floor_after_xblock
from .solver_diagnostics import (
    RHS1KSPDiagnosticsContext,
    build_profile_response_linear_metadata,
    emit_profile_response_ksp_replay_diagnostics,
)


EmitFn = Callable[[int, str], None]


@dataclass(frozen=True)
class ProfileResponseLinearFinalizationContext:
    """Inputs needed to finalize a v3-compatible profile-response linear solve."""

    op: Any
    rhs: jnp.ndarray
    result: GMRESSolveResult
    residual_vec: jnp.ndarray | None
    ksp_replay: Any
    ksp_diagnostics_context: RHS1KSPDiagnosticsContext
    tol: float
    atol: float
    solve_method: str
    active_size: int
    used_large_cpu_xblock_shortcut: bool
    used_explicit_fp_xblock_seed: bool
    use_implicit: bool
    backend: str
    metadata_parts: Sequence[Mapping[str, object]]
    emit: EmitFn | None = None
    elapsed_s: Callable[[], float] | None = None


def profile_response_post_xblock_accept_floor(
    *,
    op: Any,
    active_size: int,
    used_large_cpu_xblock_shortcut: bool,
    used_explicit_fp_xblock_seed: bool,
    use_implicit: bool,
    backend: str,
) -> float:
    """Return the final metadata acceptance floor for post-xblock RHSMode=1 solves."""

    if int(op.rhs_mode) != 1:
        return 0.0
    return rhs1_scipy_rescue_abs_floor_after_xblock(
        op=op,
        active_size=int(active_size),
        used_large_cpu_xblock_shortcut=bool(used_large_cpu_xblock_shortcut),
        used_explicit_fp_xblock_seed=bool(used_explicit_fp_xblock_seed),
        use_implicit=bool(use_implicit),
        backend=str(backend),
    )


def finalize_profile_response_linear_solve(
    context: ProfileResponseLinearFinalizationContext,
) -> V3LinearSolveResult:
    """Apply final cleanup, diagnostics, metadata, and result wrapping."""

    result = finalize_rhs1_linear_solution_cleanup(
        op=context.op,
        result=context.result,
        rhs=context.rhs,
        residual_vec=context.residual_vec,
    )
    emit_profile_response_ksp_replay_diagnostics(
        context=context.ksp_diagnostics_context,
        replay_state=context.ksp_replay,
        tol_val=float(context.tol),
        atol_val=float(context.atol),
        solve_method_val=str(context.solve_method),
    )
    if context.emit is not None:
        context.emit(
            0,
            "solve_v3_full_system_linear_gmres: "
            f"residual_norm={float(result.residual_norm):.6e}",
        )
        if context.elapsed_s is not None:
            context.emit(
                1,
                "solve_v3_full_system_linear_gmres: "
                f"elapsed_s={float(context.elapsed_s()):.3f}",
            )

    post_xblock_accept_floor = profile_response_post_xblock_accept_floor(
        op=context.op,
        active_size=int(context.active_size),
        used_large_cpu_xblock_shortcut=bool(context.used_large_cpu_xblock_shortcut),
        used_explicit_fp_xblock_seed=bool(context.used_explicit_fp_xblock_seed),
        use_implicit=bool(context.use_implicit),
        backend=str(context.backend),
    )
    metadata_out = build_profile_response_linear_metadata(
        rhs_mode=int(context.op.rhs_mode),
        result_residual_norm=float(result.residual_norm),
        rhs=context.rhs,
        tol=float(context.tol),
        atol=float(context.atol),
        metadata_parts=context.metadata_parts,
        post_xblock_accept_floor=float(post_xblock_accept_floor),
    )
    return V3LinearSolveResult(
        op=context.op,
        rhs=context.rhs,
        gmres=result,
        metadata=metadata_out or None,
    )


__all__ = [
    "ProfileResponseLinearFinalizationContext",
    "finalize_profile_response_linear_solve",
    "profile_response_post_xblock_accept_floor",
]
