"""Composite PAS preconditioners for RHSMode=1 profile-response solves."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import os

import jax.numpy as jnp
import numpy as np

from sfincs_jax.problems.profile_residual import safe_preconditioner
from sfincs_jax.operators.profile_system import V3FullSystemOperator

Preconditioner = Callable[[jnp.ndarray], jnp.ndarray]
PreconditionerBuilder = Callable[..., Preconditioner]
ApplicabilityPredicate = Callable[[V3FullSystemOperator], bool]

__all__ = (
    "RHS1PasCompositeBuilders",
    "RHS1PasFamilyBuilders",
    "build_rhs1_pas_hybrid_preconditioner",
    "build_rhs1_pas_lite_preconditioner",
    "build_rhs1_pas_schur_preconditioner",
    "compose_preconditioners",
)


@dataclass(frozen=True)
class RHS1PasCompositeBuilders:
    """Builder bundle used by PAS composite preconditioners.

    The composite policies live in this module, while the individual line,
    angular, collision, and x-coarse builders are supplied by the solve
    orchestration layer or by tests.
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


@dataclass(frozen=True)
class RHS1PasFamilyBuilders:
    """RHSMode=1 PAS-family dependency bundle.

    This object keeps the PAS preconditioner-family wiring near the PAS
    implementations. The solve orchestration layer supplies the current
    low-level builders so tests can override those callables without changing
    how the PAS family composes tokamak-theta, theta-zeta, lite, hybrid, Schur,
    and x-block ILU variants.
    """

    pas_tokamak_theta_applicable: ApplicabilityPredicate
    pas_tz_applicable: ApplicabilityPredicate
    pas_tz_memory_safe: ApplicabilityPredicate
    matvec_shard_axis: Callable[[object], str | None]
    device_count: Callable[[], int]
    block_preconditioner_builder: PreconditionerBuilder
    theta_schwarz_builder: PreconditionerBuilder
    zeta_schwarz_builder: PreconditionerBuilder
    theta_line_builder: PreconditionerBuilder
    zeta_line_builder: PreconditionerBuilder
    xblock_tz_lmax_builder: PreconditionerBuilder
    xmg_builder: PreconditionerBuilder
    xupwind_builder: PreconditionerBuilder
    collision_builder: PreconditionerBuilder
    tzfft_builder: PreconditionerBuilder
    pas_hybrid_builder: PreconditionerBuilder | None = None

    def composite_builders(self) -> RHS1PasCompositeBuilders:
        """Return the builder subset needed by PAS composite variants."""

        return RHS1PasCompositeBuilders(
            pas_tokamak_theta_applicable=self.pas_tokamak_theta_applicable,
            pas_tz_applicable=self.pas_tz_applicable,
            pas_tokamak_theta_builder=self.build_tokamak_theta,
            pas_tz_builder=self.build_tz,
            theta_line_builder=self.theta_line_builder,
            zeta_line_builder=self.zeta_line_builder,
            xblock_tz_lmax_builder=self.xblock_tz_lmax_builder,
            xmg_builder=self.xmg_builder,
            xupwind_builder=self.xupwind_builder,
            collision_builder=self.collision_builder,
        )

    def build_tokamak_theta(
        self,
        *,
        op: V3FullSystemOperator,
        reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
        expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    ) -> Preconditioner:
        """Build the tokamak-like PAS theta/L preconditioner."""

        from .preconditioner_pas_angular import build_rhs1_pas_tokamak_theta_preconditioner

        return build_rhs1_pas_tokamak_theta_preconditioner(
            op=op,
            reduce_full=reduce_full,
            expand_reduced=expand_reduced,
            block_preconditioner_builder=self.block_preconditioner_builder,
            pas_tokamak_theta_applicable=self.pas_tokamak_theta_applicable,
        )

    def build_tz(
        self,
        *,
        op: V3FullSystemOperator,
        reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
        expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    ) -> Preconditioner:
        """Build the PAS theta-zeta/L preconditioner with memory fallback."""

        from .preconditioner_pas_angular import build_rhs1_pas_tz_preconditioner

        return build_rhs1_pas_tz_preconditioner(
            op=op,
            reduce_full=reduce_full,
            expand_reduced=expand_reduced,
            pas_tz_applicable=self.pas_tz_applicable,
            pas_tz_memory_safe=self.pas_tz_memory_safe,
            matvec_shard_axis=self.matvec_shard_axis,
            device_count=self.device_count,
            theta_schwarz_builder=self.theta_schwarz_builder,
            zeta_schwarz_builder=self.zeta_schwarz_builder,
            pas_hybrid_builder=self.pas_hybrid_builder or self.build_hybrid,
            collision_builder=self.collision_builder,
            tzfft_builder=self.tzfft_builder,
        )

    def build_lite(
        self,
        *,
        op: V3FullSystemOperator,
        reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
        expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
        safe: bool = True,
    ) -> Preconditioner:
        """Build the lightweight PAS composite preconditioner."""

        return build_rhs1_pas_lite_preconditioner(
            op=op,
            builders=self.composite_builders(),
            reduce_full=reduce_full,
            expand_reduced=expand_reduced,
            safe=safe,
        )

    def build_hybrid(
        self,
        *,
        op: V3FullSystemOperator,
        reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
        expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
        safe: bool = True,
    ) -> Preconditioner:
        """Build the PAS line/x-coarse hybrid preconditioner."""

        return build_rhs1_pas_hybrid_preconditioner(
            op=op,
            builders=self.composite_builders(),
            reduce_full=reduce_full,
            expand_reduced=expand_reduced,
            safe=safe,
        )

    def build_schur(
        self,
        *,
        op: V3FullSystemOperator,
        reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
        expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
        safe: bool = True,
    ) -> Preconditioner:
        """Build the strongest PAS composite Schur-style preconditioner."""

        return build_rhs1_pas_schur_preconditioner(
            op=op,
            builders=self.composite_builders(),
            reduce_full=reduce_full,
            expand_reduced=expand_reduced,
            safe=safe,
        )

    def build_xblock_ilu(
        self,
        *,
        op: V3FullSystemOperator,
        reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
        expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    ) -> Preconditioner:
        """Build the PAS x-block sparse ILU/LU preconditioner."""

        from .preconditioner_pas_xblock_ilu import build_rhs1_pas_xblock_ilu_preconditioner

        return build_rhs1_pas_xblock_ilu_preconditioner(
            op=op,
            reduce_full=reduce_full,
            expand_reduced=expand_reduced,
            pas_hybrid_preconditioner=self.pas_hybrid_builder or self.build_hybrid,
        )


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
