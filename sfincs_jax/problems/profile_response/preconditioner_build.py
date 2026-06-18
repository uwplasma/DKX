"""RHSMode=1 profile-response preconditioner build orchestration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
import os

import jax
import jax.numpy as jnp


Preconditioner = Callable[[jnp.ndarray], jnp.ndarray]


@dataclass(frozen=True)
class RHS1ReducedPreconditionerBuildContext:
    """Solve-local dependencies needed to build reduced RHSMode=1 preconditioners."""

    op: Any
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray]
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray]
    mv_reduced: Callable[[jnp.ndarray], jnp.ndarray]
    emit: Callable[[int, str], None] | None
    mark: Callable[[str], None]
    progress_preconditioner_build: Callable[[str | None], None] | None
    record_structured_metadata: Callable[[Preconditioner], None]
    wrap_pas_preconditioner: Callable[[Preconditioner], Preconditioner]
    dd_setup: Any
    use_pas_projection: bool
    preconditioner_species: int
    preconditioner_x: int
    preconditioner_xi: int
    build_from_kind: Callable[..., Preconditioner]
    build_collision: Callable[..., Preconditioner]
    build_xmg: Callable[..., Preconditioner]
    compose_residual_correction: Callable[..., Preconditioner]
    compose_multilevel_residual_correction: Callable[..., Preconditioner]
    compose_multilevel_minres_correction: Callable[..., Preconditioner]
    parse_guarded_structured_levels: Callable[[str], tuple[str, ...]]
    resource_exhausted_error: Callable[[BaseException], bool]


@dataclass(frozen=True)
class RHS1FullPreconditionerBuildContext:
    """Solve-local dependencies needed to build full RHSMode=1 preconditioners."""

    op: Any
    emit: Callable[[int, str], None] | None
    mark: Callable[[str], None]
    progress_preconditioner_build: Callable[[str | None], None] | None
    record_structured_metadata: Callable[[Preconditioner], None]
    dd_setup: Any
    preconditioner_species: int
    preconditioner_x: int
    preconditioner_xi: int
    build_from_kind: Callable[..., Preconditioner]


@dataclass(frozen=True)
class RHS1ReducedPreconditionerBuildResult:
    """Result of a reduced preconditioner build attempt."""

    preconditioner: Preconditioner
    rhs1_precond_kind: str | None
    pas_precond_force_collision: bool
    bicgstab_preconditioner: Preconditioner | None
    pas_tz_guarded_fallback: bool
    pas_tz_guarded_axis: str | None


def _parse_adi_sweeps() -> int:
    sweeps_env = os.environ.get("SFINCS_JAX_RHSMODE1_ADI_SWEEPS", "").strip()
    try:
        return int(sweeps_env) if sweeps_env else 2
    except ValueError:
        return 2


def _parse_xblock_tz_lmax(
    *,
    rhs1_precond_kind: str | None,
    rhs1_xblock_tz_lmax: int | None,
) -> int:
    lmax_use = rhs1_xblock_tz_lmax or 0
    if rhs1_precond_kind == "xblock_tz_lmax" and lmax_use <= 0:
        lmax_env = os.environ.get("SFINCS_JAX_RHSMODE1_XBLOCK_TZ_LMAX", "").strip()
        try:
            lmax_use = int(lmax_env) if lmax_env else 0
        except ValueError:
            lmax_use = 0
    return int(lmax_use)


def _wrap_if_needed(context: RHS1ReducedPreconditionerBuildContext, precond: Preconditioner) -> Preconditioner:
    return context.wrap_pas_preconditioner(precond) if context.use_pas_projection else precond


def build_rhs1_reduced_preconditioner(
    *,
    context: RHS1ReducedPreconditionerBuildContext,
    rhs1_precond_kind: str | None,
    rhs1_xblock_tz_lmax: int | None,
) -> RHS1ReducedPreconditionerBuildResult:
    """Build the requested reduced RHSMode=1 preconditioner."""

    context.mark("rhs1_precond_build_start")
    if context.emit is not None:
        context.emit(
            1,
            "solve_v3_full_system_linear_gmres: building RHSMode=1 preconditioner="
            f"{rhs1_precond_kind} (active-DOF)",
        )
        if context.progress_preconditioner_build is not None:
            context.progress_preconditioner_build(rhs1_precond_kind)

    sweeps = _parse_adi_sweeps()
    lmax_use = _parse_xblock_tz_lmax(
        rhs1_precond_kind=rhs1_precond_kind,
        rhs1_xblock_tz_lmax=rhs1_xblock_tz_lmax,
    )

    precond = context.build_from_kind(
        op=context.op,
        rhs1_precond_kind=rhs1_precond_kind,
        reduce_full=context.reduce_full,
        expand_reduced=context.expand_reduced,
        preconditioner_species=int(context.preconditioner_species),
        preconditioner_x=int(context.preconditioner_x),
        preconditioner_xi=int(context.preconditioner_xi),
        rhs1_xblock_tz_lmax=int(lmax_use),
        dd_block_theta=context.dd_setup.block("theta"),
        dd_overlap_theta=context.dd_setup.overlap(
            "theta",
            default=1 if rhs1_precond_kind == "theta_schwarz" else 0,
        ),
        dd_block_zeta=context.dd_setup.block("zeta"),
        dd_overlap_zeta=context.dd_setup.overlap(
            "zeta",
            default=1 if rhs1_precond_kind == "zeta_schwarz" else 0,
        ),
        adi_sweeps=max(1, int(sweeps)),
        emit=context.emit,
    )
    context.record_structured_metadata(precond)
    pas_tz_guarded_fallback = bool(getattr(precond, "_sfincs_jax_pas_tz_guarded_fallback", False))
    pas_tz_guarded_axis = (
        str(getattr(precond, "_sfincs_jax_pas_tz_guarded_axis", "unknown"))
        if pas_tz_guarded_fallback
        else None
    )
    if pas_tz_guarded_fallback and context.emit is not None:
        context.emit(
            1,
            "solve_v3_full_system_linear_gmres: PAS-TZ structured fallback "
            f"guarded out (axis={pas_tz_guarded_axis}); using cheap fallback",
        )

    precond = _wrap_if_needed(context, precond)
    if pas_tz_guarded_fallback:
        precond = _apply_pas_tz_guarded_overlays(context=context, precond=precond)

    context.mark("rhs1_precond_build_done")
    return RHS1ReducedPreconditionerBuildResult(
        preconditioner=precond,
        rhs1_precond_kind=rhs1_precond_kind,
        pas_precond_force_collision=False,
        bicgstab_preconditioner=None,
        pas_tz_guarded_fallback=bool(pas_tz_guarded_fallback),
        pas_tz_guarded_axis=pas_tz_guarded_axis,
    )


def build_rhs1_full_preconditioner(
    *,
    context: RHS1FullPreconditionerBuildContext,
    rhs1_precond_kind: str | None,
    rhs1_xblock_tz_lmax: int | None,
) -> Preconditioner:
    """Build the requested full-system RHSMode=1 preconditioner."""

    context.mark("rhs1_precond_build_start")
    if context.emit is not None:
        context.emit(
            1,
            f"solve_v3_full_system_linear_gmres: building RHSMode=1 preconditioner={rhs1_precond_kind}",
        )
        if context.progress_preconditioner_build is not None:
            context.progress_preconditioner_build(rhs1_precond_kind)

    sweeps = _parse_adi_sweeps()
    lmax_use = _parse_xblock_tz_lmax(
        rhs1_precond_kind=rhs1_precond_kind,
        rhs1_xblock_tz_lmax=rhs1_xblock_tz_lmax,
    )
    precond = context.build_from_kind(
        op=context.op,
        rhs1_precond_kind=rhs1_precond_kind,
        preconditioner_species=int(context.preconditioner_species),
        preconditioner_x=int(context.preconditioner_x),
        preconditioner_xi=int(context.preconditioner_xi),
        rhs1_xblock_tz_lmax=int(lmax_use),
        dd_block_theta=context.dd_setup.block("theta"),
        dd_overlap_theta=context.dd_setup.overlap(
            "theta",
            default=1 if rhs1_precond_kind == "theta_schwarz" else 0,
        ),
        dd_block_zeta=context.dd_setup.block("zeta"),
        dd_overlap_zeta=context.dd_setup.overlap(
            "zeta",
            default=1 if rhs1_precond_kind == "zeta_schwarz" else 0,
        ),
        adi_sweeps=max(1, int(sweeps)),
        emit=context.emit,
    )
    context.record_structured_metadata(precond)
    context.mark("rhs1_precond_build_done")
    return precond


def build_rhs1_reduced_preconditioner_with_fallback(
    *,
    context: RHS1ReducedPreconditionerBuildContext,
    rhs1_precond_kind: str | None,
    rhs1_xblock_tz_lmax: int | None,
    rhs1_bicgstab_kind: str | None,
) -> RHS1ReducedPreconditionerBuildResult:
    """Build a reduced preconditioner, falling back on accelerator PAS OOM."""

    try:
        return build_rhs1_reduced_preconditioner(
            context=context,
            rhs1_precond_kind=rhs1_precond_kind,
            rhs1_xblock_tz_lmax=rhs1_xblock_tz_lmax,
        )
    except Exception as exc:  # noqa: BLE001
        if (
            jax.default_backend() != "cpu"
            and context.op.fblock.pas is not None
            and rhs1_precond_kind not in {"collision", "point"}
            and context.resource_exhausted_error(exc)
        ):
            if context.emit is not None:
                context.emit(
                    0,
                    "solve_v3_full_system_linear_gmres: accelerator PAS preconditioner "
                    f"OOM for kind={rhs1_precond_kind}; falling back to collision preconditioner",
                )
            precond = context.build_collision(
                op=context.op,
                reduce_full=context.reduce_full,
                expand_reduced=context.expand_reduced,
            )
            precond = _wrap_if_needed(context, precond)
            return RHS1ReducedPreconditionerBuildResult(
                preconditioner=precond,
                rhs1_precond_kind="collision",
                pas_precond_force_collision=True,
                bicgstab_preconditioner=precond if rhs1_bicgstab_kind == "rhs1" else None,
                pas_tz_guarded_fallback=False,
                pas_tz_guarded_axis=None,
            )
        raise


def _apply_pas_tz_guarded_overlays(
    *,
    context: RHS1ReducedPreconditionerBuildContext,
    precond: Preconditioner,
) -> Preconditioner:
    poly_steps_env = os.environ.get("SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_POLY_STEPS", "").strip()
    poly_damping_env = os.environ.get("SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_POLY_DAMPING", "").strip()
    try:
        poly_steps = int(poly_steps_env) if poly_steps_env else 0
    except ValueError:
        poly_steps = 0
    try:
        poly_damping = float(poly_damping_env) if poly_damping_env else 0.5
    except ValueError:
        poly_damping = 0.5
    if poly_steps > 0:
        precond = context.compose_residual_correction(
            base=precond,
            coarse=precond,
            matvec=context.mv_reduced,
            damping=float(poly_damping),
            steps=int(poly_steps),
        )
        if context.emit is not None:
            context.emit(
                1,
                "solve_v3_full_system_linear_gmres: PAS-TZ guarded matrix-free "
                f"polynomial correction steps={int(poly_steps)} damping={float(poly_damping):.3g}",
            )

    structured_levels = context.parse_guarded_structured_levels(
        os.environ.get("SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_STRUCTURED_LEVELS", "")
    )
    if not structured_levels:
        return precond

    coarse_preconditioners: list[Preconditioner] = []
    for level in structured_levels:
        if level == "xmg":
            coarse = context.build_xmg(
                op=context.op,
                reduce_full=context.reduce_full,
                expand_reduced=context.expand_reduced,
            )
        elif level == "collision":
            coarse = context.build_collision(
                op=context.op,
                reduce_full=context.reduce_full,
                expand_reduced=context.expand_reduced,
            )
        else:
            continue
        coarse_preconditioners.append(_wrap_if_needed(context, coarse))

    structured_steps_env = os.environ.get("SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_STRUCTURED_STEPS", "").strip()
    structured_damping_env = os.environ.get("SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_STRUCTURED_DAMPING", "").strip()
    structured_mode = (
        os.environ.get("SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_STRUCTURED_MODE", "")
        .strip()
        .lower()
        .replace("-", "_")
    )
    try:
        structured_steps = int(structured_steps_env) if structured_steps_env else 1
    except ValueError:
        structured_steps = 1
    try:
        structured_damping = float(structured_damping_env) if structured_damping_env else 0.7
    except ValueError:
        structured_damping = 0.7
    if not coarse_preconditioners or structured_steps <= 0:
        return precond

    if structured_mode in {"fixed", "damped", "residual"}:
        precond = context.compose_multilevel_residual_correction(
            base=precond,
            coarse_levels=tuple(coarse_preconditioners),
            matvec=context.mv_reduced,
            damping=float(structured_damping),
            steps=int(structured_steps),
        )
        structured_mode_label = f"fixed damping={float(structured_damping):.3g}"
    else:
        alpha_clip_env = os.environ.get("SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_STRUCTURED_ALPHA_CLIP", "").strip()
        min_improve_env = os.environ.get("SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_STRUCTURED_MIN_IMPROVEMENT", "").strip()
        try:
            alpha_clip = float(alpha_clip_env) if alpha_clip_env else 1.0
        except ValueError:
            alpha_clip = 1.0
        try:
            min_improve = float(min_improve_env) if min_improve_env else 0.0
        except ValueError:
            min_improve = 0.0
        precond = context.compose_multilevel_minres_correction(
            base=precond,
            coarse_levels=tuple(coarse_preconditioners),
            matvec=context.mv_reduced,
            alpha_clip=float(alpha_clip),
            min_improvement=float(min_improve),
            steps=int(structured_steps),
        )
        structured_mode_label = (
            f"minres alpha_clip={float(alpha_clip):.3g} "
            f"min_improvement={float(min_improve):.3g}"
        )
    if context.emit is not None:
        context.emit(
            1,
            "solve_v3_full_system_linear_gmres: PAS-TZ guarded structured "
            "residual correction levels="
            f"{','.join(structured_levels)} steps={int(structured_steps)} "
            f"mode={structured_mode_label}",
        )
    return precond


__all__ = [
    "RHS1FullPreconditionerBuildContext",
    "RHS1ReducedPreconditionerBuildContext",
    "RHS1ReducedPreconditionerBuildResult",
    "build_rhs1_full_preconditioner",
    "build_rhs1_reduced_preconditioner",
    "build_rhs1_reduced_preconditioner_with_fallback",
]
