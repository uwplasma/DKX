"""Composite PAS preconditioners for RHSMode=1 profile-response solves."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import os

import jax.numpy as jnp
import numpy as np

from ....problems.profile_response.residual import safe_preconditioner
from ....v3_system import V3FullSystemOperator

Preconditioner = Callable[[jnp.ndarray], jnp.ndarray]
PreconditionerBuilder = Callable[..., Preconditioner]
ApplicabilityPredicate = Callable[[V3FullSystemOperator], bool]

__all__ = (
    "RHS1PasCompositeBuilders",
    "build_rhs1_pas_hybrid_preconditioner",
    "build_rhs1_pas_lite_preconditioner",
    "build_rhs1_pas_schur_preconditioner",
    "compose_preconditioners",
)


@dataclass(frozen=True)
class RHS1PasCompositeBuilders:
    """Builder bundle used by PAS composite preconditioners.

    The composite policies live in this module, while the individual line,
    angular, collision, and x-coarse builders can still be supplied by the
    driver compatibility facade or by tests.
    """

    pas_tokamak_theta_applicable: ApplicabilityPredicate
    pas_tz_applicable: ApplicabilityPredicate
    pas_tokamak_theta_builder: PreconditionerBuilder
    pas_tz_builder: PreconditionerBuilder
    theta_line_builder: PreconditionerBuilder
    zeta_line_builder: PreconditionerBuilder
    xblock_tz_lmax_builder: PreconditionerBuilder
    xmg_builder: PreconditionerBuilder
    xupwind_builder: PreconditionerBuilder
    collision_builder: PreconditionerBuilder


def compose_preconditioners(
    first: Preconditioner,
    second: Preconditioner,
) -> Preconditioner:
    """Return the composition ``second(first(v))`` used by RHSMode=1 policies."""

    def _apply(v: jnp.ndarray) -> jnp.ndarray:
        return second(first(v))

    return _apply


def build_rhs1_pas_lite_preconditioner(
    *,
    op: V3FullSystemOperator,
    builders: RHS1PasCompositeBuilders,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    safe: bool = True,
) -> Preconditioner:
    """Build the lightweight PAS angular/x-coarse/collision composition."""

    angular_precond: Preconditioner | None = None
    if builders.pas_tokamak_theta_applicable(op):
        angular_precond = builders.pas_tokamak_theta_builder(
            op=op,
            reduce_full=reduce_full,
            expand_reduced=expand_reduced,
        )
    elif builders.pas_tz_applicable(op):
        tz_max_env = os.environ.get("SFINCS_JAX_PAS_LITE_TZ_MAX", "").strip()
        try:
            tz_max = int(tz_max_env) if tz_max_env else 256
        except ValueError:
            tz_max = 256
        tz_size = int(op.n_theta) * int(op.n_zeta)
        if tz_max > 0 and tz_size <= tz_max:
            angular_precond = builders.pas_tz_builder(
                op=op,
                reduce_full=reduce_full,
                expand_reduced=expand_reduced,
            )

    x_precond = builders.xmg_builder(
        op=op,
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
    )
    collision_precond = builders.collision_builder(
        op=op,
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
    )
    precond = x_precond
    if angular_precond is not None:
        precond = compose_preconditioners(angular_precond, precond)
    precond = compose_preconditioners(precond, collision_precond)
    return safe_preconditioner(precond) if bool(safe) else precond


def build_rhs1_pas_hybrid_preconditioner(
    *,
    op: V3FullSystemOperator,
    builders: RHS1PasCompositeBuilders,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    safe: bool = True,
) -> Preconditioner:
    """Build the PAS line/truncated-L x-coarse/collision hybrid."""

    line_precond: Preconditioner | None = None
    nxi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
    local_per_species = int(np.sum(nxi_for_x))
    line_size = int(local_per_species * int(op.n_theta))
    line_max_env = os.environ.get("SFINCS_JAX_PAS_HYBRID_LINE_MAX", "").strip()
    try:
        line_max = int(line_max_env) if line_max_env else 512
    except ValueError:
        line_max = 512
    if not line_max_env and int(op.n_zeta) <= 5 and int(op.n_theta) >= 8:
        # Tokamak-like grids benefit from line preconditioning and remain small.
        line_max = max(int(line_max), 4096)
    if line_size <= line_max:
        if int(op.n_theta) >= int(op.n_zeta):
            line_precond = builders.theta_line_builder(
                op=op,
                reduce_full=reduce_full,
                expand_reduced=expand_reduced,
            )
        else:
            line_precond = builders.zeta_line_builder(
                op=op,
                reduce_full=reduce_full,
                expand_reduced=expand_reduced,
            )
    else:
        lmax_env = os.environ.get("SFINCS_JAX_PAS_HYBRID_LMAX", "").strip()
        has_er = op.fblock.er_xdot is not None or op.fblock.er_xidot is not None
        use_dkes_exb = bool(getattr(op.fblock.exb_theta, "use_dkes_exb_drift", False)) or bool(
            getattr(op.fblock.exb_zeta, "use_dkes_exb_drift", False)
        )
        needs_stronger_l = bool(has_er or use_dkes_exb)
        nz = int(op.n_theta) * int(op.n_zeta)
        try:
            if lmax_env:
                lmax = int(lmax_env)
            else:
                lmax = 8 if (needs_stronger_l or nz <= 256) else 2
        except ValueError:
            lmax = 8 if (needs_stronger_l or nz <= 256) else 2
        xblock_max_env = os.environ.get("SFINCS_JAX_PAS_HYBRID_XBLOCK_MAX", "").strip()
        try:
            xblock_max_default = 2048 if (needs_stronger_l or nz <= 256) else 256
            xblock_max = int(xblock_max_env) if xblock_max_env else xblock_max_default
        except ValueError:
            xblock_max = 2048 if (needs_stronger_l or nz <= 256) else 256
        if nz > 0:
            max_allowed = int(xblock_max // nz)
            if max_allowed > 0:
                lmax = min(int(lmax), max_allowed)
        block_size = int(max(0, lmax) * int(op.n_theta) * int(op.n_zeta))
        if lmax > 0 and block_size > 0 and block_size <= xblock_max:
            line_precond = builders.xblock_tz_lmax_builder(
                op=op,
                lmax=lmax,
                reduce_full=reduce_full,
                expand_reduced=expand_reduced,
            )

    if op.fblock.pas is not None and op.fblock.fp is None and op.fblock.er_xdot is not None:
        x_precond = builders.xupwind_builder(
            op=op,
            reduce_full=reduce_full,
            expand_reduced=expand_reduced,
        )
    else:
        x_precond = builders.xmg_builder(
            op=op,
            reduce_full=reduce_full,
            expand_reduced=expand_reduced,
        )
    collision_precond = builders.collision_builder(
        op=op,
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
    )
    precond = x_precond
    if line_precond is not None:
        precond = compose_preconditioners(line_precond, precond)
    precond = compose_preconditioners(precond, collision_precond)
    return safe_preconditioner(precond) if bool(safe) else precond


def build_rhs1_pas_schur_preconditioner(
    *,
    op: V3FullSystemOperator,
    builders: RHS1PasCompositeBuilders,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    safe: bool = True,
) -> Preconditioner:
    """Build the stronger PAS angular/x-coarse/collision Schur-style composition."""

    if builders.pas_tokamak_theta_applicable(op):
        angular_precond = builders.pas_tokamak_theta_builder(
            op=op,
            reduce_full=reduce_full,
            expand_reduced=expand_reduced,
        )
    elif builders.pas_tz_applicable(op):
        angular_precond = builders.pas_tz_builder(
            op=op,
            reduce_full=reduce_full,
            expand_reduced=expand_reduced,
        )
    else:
        angular_precond = build_rhs1_pas_hybrid_preconditioner(
            op=op,
            builders=builders,
            reduce_full=reduce_full,
            expand_reduced=expand_reduced,
            safe=False,
        )

    if op.fblock.pas is not None and op.fblock.fp is None and op.fblock.er_xdot is not None:
        x_precond = builders.xupwind_builder(
            op=op,
            reduce_full=reduce_full,
            expand_reduced=expand_reduced,
        )
    else:
        x_precond = builders.xmg_builder(
            op=op,
            reduce_full=reduce_full,
            expand_reduced=expand_reduced,
        )

    collision_precond = builders.collision_builder(
        op=op,
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
    )
    precond = compose_preconditioners(angular_precond, x_precond)
    precond = compose_preconditioners(precond, collision_precond)
    return safe_preconditioner(precond) if bool(safe) else precond
