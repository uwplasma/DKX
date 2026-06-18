"""Matrix-free QI device seed correction for RHSMode=1 profile solves."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
import os

import jax.numpy as jnp

from sfincs_jax.problems.profile_response.policies import (
    rhs1_qi_device_extra_coarse_controls as _rhs1_qi_device_extra_coarse_controls,
    rhs1_qi_device_extra_coarse_metadata as _rhs1_qi_device_extra_coarse_metadata,
    rhs1_qi_device_extra_coarse_setup_kwargs as _rhs1_qi_device_extra_coarse_setup_kwargs,
    rhs1_qi_device_probe_uses_minres_step as _rhs1_qi_device_probe_uses_minres_step,
    rhs1_qi_device_rank_budget as _rhs1_qi_device_rank_budget,
    rhs1_qi_device_status_fields as _rhs1_qi_device_status_fields,
    rhs1_qi_device_tail_block_required as _rhs1_qi_device_tail_block_required,
)
from sfincs_jax.rhs1_qi_coarse import (
    build_rhs1_xblock_qi_coarse_basis as _rhs1_xblock_qi_coarse_basis,
    rhs1_xblock_qi_block_geometry_metadata as _rhs1_xblock_qi_block_geometry_metadata,
)
from sfincs_jax.rhs1_qi_device_preconditioner import (
    RHS1QIDevicePreconditionerConfig,
    probe_rhs1_qi_device_preconditioner,
    setup_rhs1_qi_device_preconditioner,
)
from sfincs_jax.rhs1_solver_policy import (
    read_bool_env as _rhs1_bool_env,
    read_float_env as _rhs1_float_env,
    read_int_env as _rhs1_int_env,
)
from sfincs_jax.solver import GMRESSolveResult


@dataclass
class MatrixFreeQIDeviceSeedContext:
    """Solve-local state needed by the matrix-free QI device seed attempt."""

    op: Any
    active_size: int
    target_reduced: float
    mv_reduced: Callable[[jnp.ndarray], jnp.ndarray]
    rhs_reduced: jnp.ndarray
    emit: Callable[[int, str], None] | None
    timer_elapsed_s: Callable[[], float]
    rhsmode1_general_metadata: dict[str, object]

    def elapsed_s(self) -> float:
        """Return solve elapsed time using the driver timer."""

        return float(self.timer_elapsed_s())


def attempt_matrixfree_qi_device_seed(
    current_result: GMRESSolveResult,
    *,
    hook: str,
    context: MatrixFreeQIDeviceSeedContext,
) -> GMRESSolveResult:
    """Try the matrix-free QI device seed correction and update metadata."""

    op = context.op
    active_size = int(context.active_size)
    target_reduced = float(context.target_reduced)
    mv_reduced = context.mv_reduced
    rhs_reduced = context.rhs_reduced
    emit = context.emit
    rhsmode1_general_metadata = context.rhsmode1_general_metadata
    t = context
    if not (
        _rhs1_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER",
            default=False,
        )
        and _rhs1_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE",
            default=False,
        )
    ):
        return current_result
    if float(current_result.residual_norm) <= float(target_reduced):
        return current_result

    qi_device_start_s = t.elapsed_s()
    qi_device_reason = "not_attempted"
    qi_device_metadata: dict[str, object] = {
        "hook": hook,
        "operator_source": "matrix_free",
        "active_dof": True,
        "active_size": int(active_size),
    }
    try:
        qi_seed_max_rank = _rhs1_int_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_MAX_RANK",
            default=24,
            minimum=1,
        )
        qi_seed_max_candidates = _rhs1_int_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_MAX_CANDIDATES",
            default=96,
            minimum=1,
        )
        qi_seed_max_angular_mode = _rhs1_int_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_MAX_ANGULAR_MODE",
            default=2,
            minimum=0,
        )
        qi_seed_rank_rtol = _rhs1_float_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_RANK_RTOL",
            default=1.0e-10,
            minimum=0.0,
        )
        qi_device_min_improvement = _rhs1_float_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MIN_IMPROVEMENT",
            default=0.05,
            minimum=0.0,
        )
        qi_device_cycles = _rhs1_int_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_CYCLES",
            default=1,
            minimum=1,
        )
        qi_device_minres_step = _rhs1_qi_device_probe_uses_minres_step()
        qi_device_alpha_clip = _rhs1_float_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_ALPHA_CLIP",
            default=10.0,
            minimum=0.0,
        )
        qi_device_rcond = _rhs1_float_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RCOND",
            default=1.0e-12,
            minimum=0.0,
        )
        qi_device_damping = _rhs1_float_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_DAMPING",
            default=1.0,
            minimum=0.0,
        )
        qi_device_local_smoother_kind = (
            os.environ.get(
                "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_LOCAL_SMOOTHER",
                "none",
            )
            .strip()
            .lower()
            .replace("-", "_")
        )
        qi_device_matrix_free_smoother_sweeps = _rhs1_int_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE_SMOOTHER_SWEEPS",
            default=1,
            minimum=1,
        )
        qi_device_matrix_free_smoother_damping = _rhs1_float_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE_SMOOTHER_DAMPING",
            default=1.0,
            minimum=0.0,
        )
        qi_device_matrix_free_smoother_step_policy = (
            os.environ.get(
                "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE_SMOOTHER_STEP_POLICY",
                "residual_minimizing",
            )
            .strip()
            .lower()
            .replace("-", "_")
        )
        qi_device_matrix_free_smoother_alpha_clip = _rhs1_float_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE_SMOOTHER_ALPHA_CLIP",
            default=10.0,
            minimum=0.0,
        )
        qi_device_matrix_free_block_smoother_max_groups = _rhs1_int_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE_BLOCK_SMOOTHER_MAX_GROUPS",
            default=32,
            minimum=1,
        )
        qi_device_matrix_free_block_smoother_include_tail = _rhs1_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE_BLOCK_SMOOTHER_INCLUDE_TAIL",
            default=True,
        )
        qi_device_matrix_free_block_smoother_rcond = _rhs1_float_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE_BLOCK_SMOOTHER_RCOND",
            default=1.0e-12,
            minimum=0.0,
        )
        qi_device_matrix_free_block_smoother_grouping = (
            os.environ.get(
                "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE_BLOCK_SMOOTHER_GROUPING",
                "contiguous",
            )
            .strip()
            .lower()
            .replace("-", "_")
        )
        qi_device_depth = _rhs1_int_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_ENRICHMENT_DEPTH",
            default=2,
            minimum=0,
        )
        qi_device_include_residual = _rhs1_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_ENRICHMENT_INCLUDE_RESIDUAL",
            default=True,
        )
        qi_device_recycle_enrichment = _rhs1_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RECYCLE_ENRICHMENT",
            default=False,
        )
        qi_device_recycle_cycles = _rhs1_int_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RECYCLE_CYCLES",
            default=1 if bool(qi_device_recycle_enrichment) else 0,
            minimum=0,
        )
        qi_device_operator_krylov_enrichment = _rhs1_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_OPERATOR_KRYLOV_ENRICHMENT",
            default=False,
        )
        qi_device_operator_krylov_depth = _rhs1_int_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_OPERATOR_KRYLOV_DEPTH",
            default=4 if bool(qi_device_operator_krylov_enrichment) else 0,
            minimum=0,
        )
        qi_device_adjoint_krylov_enrichment = _rhs1_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_ADJOINT_KRYLOV_ENRICHMENT",
            default=False,
        )
        qi_device_adjoint_krylov_depth = _rhs1_int_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_ADJOINT_KRYLOV_DEPTH",
            default=4 if bool(qi_device_adjoint_krylov_enrichment) else 0,
            minimum=0,
        )
        qi_device_adjoint_krylov_transpose_source = (
            os.environ.get(
                "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_ADJOINT_KRYLOV_TRANSPOSE",
                "autodiff",
            )
            .strip()
            .lower()
            .replace("-", "_")
        )
        qi_device_operator_action_enrichment = _rhs1_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_OPERATOR_ACTION_ENRICHMENT",
            default=False,
        )
        qi_device_operator_action_depth = _rhs1_int_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_OPERATOR_ACTION_DEPTH",
            default=1 if bool(qi_device_operator_action_enrichment) else 0,
            minimum=0,
        )
        qi_device_multilevel_coarse = _rhs1_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_COARSE",
            default=_rhs1_bool_env(
                "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_REUSE_COARSE_OPERATOR",
                default=False,
            ),
        )
        qi_device_multilevel_max_levels = _rhs1_int_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_MAX_LEVELS",
            default=3 if bool(qi_device_multilevel_coarse) else 1,
            minimum=1,
        )
        qi_device_multilevel_aggregate_factor = _rhs1_int_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_AGGREGATE_FACTOR",
            default=2,
            minimum=2,
        )
        qi_device_multilevel_max_angular_mode = _rhs1_int_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_MAX_ANGULAR_MODE",
            default=1,
            minimum=0,
        )
        qi_device_multilevel_max_radial_degree = _rhs1_int_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_MAX_RADIAL_DEGREE",
            default=2,
            minimum=0,
        )
        qi_device_multilevel_max_pitch_degree = _rhs1_int_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_MAX_PITCH_DEGREE",
            default=0,
            minimum=0,
        )
        qi_device_multilevel_max_rank_env = os.environ.get(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_MAX_RANK",
            "",
        ).strip()
        qi_device_multilevel_max_rank: int | None = None
        if qi_device_multilevel_max_rank_env:
            try:
                qi_device_multilevel_max_rank = max(1, int(qi_device_multilevel_max_rank_env))
            except ValueError:
                qi_device_multilevel_max_rank = None
        qi_device_extra_coarse_controls = _rhs1_qi_device_extra_coarse_controls()
        qi_device_extra_coarse_setup_kwargs = (
            _rhs1_qi_device_extra_coarse_setup_kwargs(qi_device_extra_coarse_controls)
        )
        qi_device_multilevel_current_moments = bool(
            qi_device_extra_coarse_controls["multilevel_current_moments"]
        )
        qi_device_multilevel_species_current_moments = bool(
            qi_device_extra_coarse_controls["multilevel_species_current_moments"]
        )
        qi_device_multilevel_radial_current_moments = bool(
            qi_device_extra_coarse_controls["multilevel_radial_current_moments"]
        )
        qi_device_multilevel_tail_constraint_moments = bool(
            qi_device_extra_coarse_controls["multilevel_tail_constraint_moments"]
        )
        qi_device_multilevel_current_max_pitch_degree = int(
            qi_device_extra_coarse_controls["multilevel_current_max_pitch_degree"]
        )
        qi_device_rank_budget_summary = _rhs1_qi_device_rank_budget(
            seed_max_rank=int(qi_seed_max_rank),
            n_species=int(getattr(op, "n_species", 1)),
            residual_enrichment=True,
            residual_enrichment_depth=int(qi_device_depth),
            residual_enrichment_include_residual=bool(qi_device_include_residual),
            recycle_enrichment=bool(qi_device_recycle_enrichment),
            recycle_cycles=int(qi_device_recycle_cycles),
            operator_krylov_enrichment=bool(qi_device_operator_krylov_enrichment),
            operator_krylov_depth=int(qi_device_operator_krylov_depth),
            adjoint_krylov_enrichment=bool(qi_device_adjoint_krylov_enrichment),
            adjoint_krylov_depth=int(qi_device_adjoint_krylov_depth),
            operator_action_enrichment=bool(qi_device_operator_action_enrichment),
            operator_action_depth=int(qi_device_operator_action_depth),
            multilevel_coarse=bool(qi_device_multilevel_coarse),
            multilevel_max_rank=qi_device_multilevel_max_rank,
            # The seed path historically budgeted only the multilevel basis cap;
            # keep that behavior while the full main-branch policy owns moment ranks.
            multilevel_current_moments=False,
            multilevel_current_max_pitch_degree=int(
                qi_device_multilevel_current_max_pitch_degree
            ),
            multilevel_residual_equation=False,
            multilevel_residual_equation_max_level_rank=0,
            multilevel_max_levels=int(qi_device_multilevel_max_levels),
            global_moment_residual_equation=bool(
                qi_device_extra_coarse_controls["global_moment_residual_equation"]
            ),
            global_moment_residual_equation_max_rank=int(
                qi_device_extra_coarse_controls[
                    "global_moment_residual_equation_max_rank"
                ]
            ),
            residual_galerkin_equation=bool(
                qi_device_extra_coarse_controls["residual_galerkin_equation"]
            ),
            residual_galerkin_equation_max_rank=int(
                qi_device_extra_coarse_controls[
                    "residual_galerkin_equation_max_rank"
                ]
            ),
            phase_space_residual_equation=bool(
                qi_device_extra_coarse_controls["phase_space_residual_equation"]
            ),
            phase_space_residual_equation_max_rank=int(
                qi_device_extra_coarse_controls[
                    "phase_space_residual_equation_max_rank"
                ]
            ),
            residual_region_bounce_coarse=bool(
                qi_device_extra_coarse_controls["residual_region_bounce_coarse"]
            ),
            residual_region_bounce_coarse_max_rank=int(
                qi_device_extra_coarse_controls[
                    "residual_region_bounce_coarse_max_rank"
                ]
            ),
            active_pattern_coarse=bool(
                qi_device_extra_coarse_controls["active_pattern_coarse"]
            ),
            active_pattern_coarse_max_rank=int(
                qi_device_extra_coarse_controls["active_pattern_coarse_max_rank"]
            ),
            block_schur_residual_equation=False,
            block_schur_residual_equation_max_rank=0,
            coupled_residual_equation=False,
            coupled_residual_equation_max_rank=0,
            residual_snapshot_enrichment=False,
            residual_snapshot_max_rank=0,
            residual_snapshot_residual_equation=False,
            residual_snapshot_residual_equation_max_rank=0,
            block_schur_residual_enrichment=False,
            block_schur_residual_max_rank=0,
        )
        qi_device_rank_budget = int(qi_device_rank_budget_summary.rank_budget)
        qi_device_max_rank = max(
            1,
            int(qi_device_rank_budget_summary.max_rank or qi_device_rank_budget),
        )
        qi_basis_kind = (
            os.environ.get(
                "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_BASIS",
                "enriched",
            )
            .strip()
            .lower()
            .replace("-", "_")
        )
        qi_basis = _rhs1_xblock_qi_coarse_basis(
            op=op,
            active_dof=True,
            linear_size=int(active_size),
            max_rank=int(qi_seed_max_rank),
            rank_rtol=float(qi_seed_rank_rtol),
            include_angular=True,
            include_blocks=True,
            basis_kind=qi_basis_kind,
            max_candidates=int(qi_seed_max_candidates),
            max_angular_mode=int(qi_seed_max_angular_mode),
            include_radial=True,
            include_radial_angular=True,
            include_constraint_moments=True,
            include_schur=True,
        )
        residual_seed = rhs_reduced - mv_reduced(jnp.asarray(current_result.x, dtype=jnp.float64))
        qi_device_state = setup_rhs1_qi_device_preconditioner(
            operator=mv_reduced,
            coarse_basis=qi_basis,
            residual_seed=residual_seed,
            total_size=int(active_size),
            dtype=jnp.float64,
            geometry_metadata={
                "rhs_mode": int(op.rhs_mode),
                "n_theta": int(getattr(op, "n_theta", 1)),
                "n_zeta": int(getattr(op, "n_zeta", 1)),
                "n_x": int(getattr(op, "n_x", 1)),
                "n_species": int(getattr(op, "n_species", 1)),
                "active_dof": True,
                **_rhs1_xblock_qi_block_geometry_metadata(
                    op=op,
                    active_dof=True,
                        linear_size=int(active_size),
                        include_tail_block=bool(
                            _rhs1_qi_device_tail_block_required(
                                multilevel_coarse=bool(qi_device_multilevel_coarse),
                                extra_coarse_controls=qi_device_extra_coarse_controls,
                            )
                        ),
                    ),
                },
            config=RHS1QIDevicePreconditionerConfig(
                regularization_rcond=float(qi_device_rcond),
                damping=float(qi_device_damping),
                coarse_solver="action_lstsq",
                local_smoother_kind=qi_device_local_smoother_kind,
                matrix_free_smoother_sweeps=int(qi_device_matrix_free_smoother_sweeps),
                matrix_free_smoother_damping=float(qi_device_matrix_free_smoother_damping),
                matrix_free_smoother_step_policy=qi_device_matrix_free_smoother_step_policy,
                matrix_free_smoother_alpha_clip=float(qi_device_matrix_free_smoother_alpha_clip),
                matrix_free_block_smoother_max_groups=int(
                    qi_device_matrix_free_block_smoother_max_groups
                ),
                matrix_free_block_smoother_include_tail=bool(
                    qi_device_matrix_free_block_smoother_include_tail
                ),
                matrix_free_block_smoother_rcond=float(qi_device_matrix_free_block_smoother_rcond),
                matrix_free_block_smoother_grouping=qi_device_matrix_free_block_smoother_grouping,
                max_rank=int(qi_device_max_rank),
                residual_enrichment=True,
                residual_enrichment_depth=int(qi_device_depth),
                residual_enrichment_include_residual=bool(qi_device_include_residual),
                recycle_enrichment=bool(qi_device_recycle_enrichment),
                recycle_enrichment_cycles=int(qi_device_recycle_cycles),
                operator_krylov_enrichment=bool(qi_device_operator_krylov_enrichment),
                operator_krylov_depth=int(qi_device_operator_krylov_depth),
                adjoint_krylov_enrichment=bool(qi_device_adjoint_krylov_enrichment),
                adjoint_krylov_depth=int(qi_device_adjoint_krylov_depth),
                adjoint_krylov_transpose_source=qi_device_adjoint_krylov_transpose_source,
                operator_action_enrichment=bool(qi_device_operator_action_enrichment),
                operator_action_enrichment_depth=int(qi_device_operator_action_depth),
                multilevel_coarse=bool(qi_device_multilevel_coarse),
                multilevel_max_levels=int(qi_device_multilevel_max_levels),
                multilevel_aggregate_factor=int(qi_device_multilevel_aggregate_factor),
                multilevel_max_rank=qi_device_multilevel_max_rank,
                multilevel_max_angular_mode=int(qi_device_multilevel_max_angular_mode),
                multilevel_max_radial_degree=int(qi_device_multilevel_max_radial_degree),
                multilevel_max_pitch_degree=int(qi_device_multilevel_max_pitch_degree),
                multilevel_current_moments=bool(qi_device_multilevel_current_moments),
                multilevel_species_current_moments=bool(qi_device_multilevel_species_current_moments),
                multilevel_radial_current_moments=bool(qi_device_multilevel_radial_current_moments),
                multilevel_tail_constraint_moments=bool(qi_device_multilevel_tail_constraint_moments),
                multilevel_current_max_pitch_degree=int(qi_device_multilevel_current_max_pitch_degree),
                **qi_device_extra_coarse_setup_kwargs,
            ),
        )
        x_qi_device, qi_device_probe = probe_rhs1_qi_device_preconditioner(
            rhs=rhs_reduced,
            x0=current_result.x,
            state=qi_device_state,
            operator=mv_reduced,
            min_relative_improvement=float(qi_device_min_improvement),
            max_cycles=int(qi_device_cycles),
            residual_minimizing_step=bool(qi_device_minres_step),
            alpha_clip=float(qi_device_alpha_clip),
        )
        qi_device_reason = str(qi_device_probe.reason)
        qi_device_probe_cycles = int(
            getattr(qi_device_probe, "cycles", 1 if bool(qi_device_probe.accepted) else 0)
        )
        qi_device_probe_residual_history = tuple(
            float(value)
            for value in getattr(
                qi_device_probe,
                "residual_history",
                (
                    float(qi_device_probe.residual_before_norm),
                    float(qi_device_probe.residual_after_norm),
                ),
            )
        )
        qi_device_probe_step_history = tuple(
            float(value) for value in getattr(qi_device_probe, "step_history", ())
        )
        qi_device_metadata = {
            **qi_device_probe.metadata.to_dict(),
            "hook": hook,
            "min_improvement": float(qi_device_min_improvement),
            "cycles_requested": int(qi_device_cycles),
            "cycles": int(qi_device_probe_cycles),
            "residual_history": qi_device_probe_residual_history,
            "step_policy": "residual_minimizing" if bool(qi_device_minres_step) else "fixed",
            "alpha_clip": float(qi_device_alpha_clip),
            "step_history": qi_device_probe_step_history,
            "use_in_krylov": False,
            "matrix_free_enabled": True,
            "precondition_side": "seed_only",
            "local_smoother_kind_requested": qi_device_local_smoother_kind,
            "local_smoother_metadata": (
                qi_device_state.local_smoother.metadata.to_dict()
                if qi_device_state.local_smoother is not None
                and hasattr(qi_device_state.local_smoother.metadata, "to_dict")
                else None
            ),
            "residual_enrichment_requested": True,
            "residual_enrichment_depth_requested": int(qi_device_depth),
            "recycle_enrichment_requested": bool(qi_device_recycle_enrichment),
            "recycle_enrichment_cycles_requested": int(qi_device_recycle_cycles),
            "operator_krylov_enrichment_requested": bool(qi_device_operator_krylov_enrichment),
            "operator_krylov_depth_requested": int(qi_device_operator_krylov_depth),
            "adjoint_krylov_enrichment_requested": bool(qi_device_adjoint_krylov_enrichment),
            "adjoint_krylov_depth_requested": int(qi_device_adjoint_krylov_depth),
            "adjoint_krylov_transpose_requested": qi_device_adjoint_krylov_transpose_source,
            "operator_action_enrichment_requested": bool(qi_device_operator_action_enrichment),
            "operator_action_depth_requested": int(qi_device_operator_action_depth),
            "multilevel_coarse_requested": bool(qi_device_multilevel_coarse),
            "multilevel_max_levels_requested": int(qi_device_multilevel_max_levels),
            "multilevel_aggregate_factor_requested": int(qi_device_multilevel_aggregate_factor),
            "multilevel_max_rank_requested": (
                None
                if qi_device_multilevel_max_rank is None
                else int(qi_device_multilevel_max_rank)
            ),
            "multilevel_max_angular_mode_requested": int(qi_device_multilevel_max_angular_mode),
            "multilevel_max_radial_degree_requested": int(qi_device_multilevel_max_radial_degree),
            "multilevel_max_pitch_degree_requested": int(qi_device_multilevel_max_pitch_degree),
            **_rhs1_qi_device_extra_coarse_metadata(qi_device_extra_coarse_controls),
            "max_rank_requested": int(qi_device_max_rank),
            "residual_before_norm": float(qi_device_probe.residual_before_norm),
            "residual_after_norm": float(qi_device_probe.residual_after_norm),
            "improvement_ratio": (
                None
                if qi_device_probe.improvement_ratio is None
                else float(qi_device_probe.improvement_ratio)
            ),
            "accepted": bool(qi_device_probe.accepted),
        }
        qi_device_status_fields = _rhs1_qi_device_status_fields(
            extra_coarse_controls=qi_device_extra_coarse_controls,
            residual_correction_controls={},
            metadata=qi_device_metadata,
        )
        if bool(qi_device_probe.accepted):
            current_result = GMRESSolveResult(
                x=jnp.asarray(x_qi_device, dtype=jnp.float64),
                residual_norm=jnp.asarray(qi_device_probe.residual_after_norm, dtype=jnp.float64),
            )
            if emit is not None:
                emit(
                    0,
                    "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                    "QI device preconditioner accepted "
                    f"residual {float(qi_device_probe.residual_before_norm):.6e} "
                    f"-> {float(qi_device_probe.residual_after_norm):.6e} "
                    f"(rank={int(qi_device_probe.metadata.rank)} "
                    f"cycles={int(qi_device_probe_cycles)} "
                    f"ratio={float(qi_device_probe.improvement_ratio):.6e} "
                    f"operator_krylov={int(bool(qi_device_operator_krylov_enrichment))} "
                    f"coarse_reuse={int(bool(qi_device_multilevel_coarse))} "
                    f"{qi_device_status_fields} "
                    "seed_only=1)",
                )
        elif emit is not None:
            emit(
                1,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                "QI device preconditioner rejected "
                f"reason={qi_device_reason} "
                f"residual {float(qi_device_probe.residual_before_norm):.6e} "
                f"-> {float(qi_device_probe.residual_after_norm):.6e} "
                f"(rank={int(qi_device_probe.metadata.rank)} "
                f"cycles={int(qi_device_probe_cycles)} "
                f"ratio={float(qi_device_probe.improvement_ratio) if qi_device_probe.improvement_ratio is not None else float('nan'):.6e} "
                f"step_policy={qi_device_metadata.get('step_policy', 'fixed')} "
                f"operator_krylov={int(bool(qi_device_operator_krylov_enrichment))} "
                f"coarse_reuse={int(bool(qi_device_multilevel_coarse))} "
                f"{qi_device_status_fields} "
                "seed_only=1)",
            )
    except Exception as exc:  # noqa: BLE001
        qi_device_reason = f"{type(exc).__name__}: {exc}"
        qi_device_metadata = {"hook": hook, "error": qi_device_reason}
        if emit is not None:
            emit(
                1,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                f"QI device preconditioner disabled after build failure ({type(exc).__name__}: {exc})",
            )
    qi_device_setup_s = float(t.elapsed_s() - qi_device_start_s)
    qi_device_metadata["setup_s"] = qi_device_setup_s
    rhsmode1_general_metadata.update(
        {
            "xblock_qi_device_preconditioner_enabled": True,
            "xblock_qi_device_preconditioner_built": "error" not in qi_device_metadata,
            "xblock_qi_device_preconditioner_used": bool(qi_device_metadata.get("accepted", False)),
            "xblock_qi_device_preconditioner_used_in_krylov": False,
            "xblock_qi_device_preconditioner_reason": qi_device_reason,
            "xblock_qi_device_preconditioner_rank": int(qi_device_metadata.get("rank", 0) or 0),
            "xblock_qi_device_preconditioner_candidate_count": int(
                qi_device_metadata.get("residual_enrichment_candidate_count", 0) or 0
            ),
            "xblock_qi_device_preconditioner_residual_before": qi_device_metadata.get(
                "residual_before_norm"
            ),
            "xblock_qi_device_preconditioner_residual_after": qi_device_metadata.get(
                "residual_after_norm"
            ),
            "xblock_qi_device_preconditioner_improvement_ratio": qi_device_metadata.get(
                "improvement_ratio"
            ),
            "xblock_qi_device_preconditioner_metadata": qi_device_metadata,
            "xblock_qi_device_preconditioner_setup_s": qi_device_setup_s,
            "xblock_qi_device_preconditioner_min_improvement": (
                qi_device_metadata.get("min_improvement")
            ),
            "xblock_qi_device_preconditioner_use_in_krylov": False,
            "xblock_qi_device_preconditioner_seed_only": True,
            "xblock_qi_device_preconditioner_operator_krylov_enrichment": bool(
                qi_device_metadata.get("operator_krylov_enrichment_enabled", False)
            ),
            "xblock_qi_device_preconditioner_coarse_reuse": bool(
                qi_device_metadata.get("multilevel_coarse_enabled", False)
            ),
            "xblock_qi_device_preconditioner_phase_space_residual_equation": bool(
                qi_device_metadata.get("phase_space_residual_equation_enabled", False)
            ),
            "xblock_qi_device_preconditioner_phase_space_residual_equation_max_rank": int(
                qi_device_metadata.get(
                    "phase_space_residual_equation_max_rank_requested",
                    qi_device_metadata.get("phase_space_residual_equation_max_rank", 0),
                )
                or 0
            ),
            "xblock_qi_device_preconditioner_phase_space_residual_equation_solver": (
                qi_device_metadata.get("phase_space_residual_equation_solver")
            ),
            "xblock_qi_device_preconditioner_phase_space_residual_equation_candidate_count": int(
                qi_device_metadata.get("phase_space_residual_equation_candidate_count", 0)
                or 0
            ),
            "xblock_qi_device_preconditioner_phase_space_residual_equation_rank": int(
                qi_device_metadata.get("phase_space_residual_equation_rank", 0) or 0
            ),
            "xblock_qi_device_preconditioner_phase_space_residual_equation_stage_count": int(
                qi_device_metadata.get("phase_space_residual_equation_stage_count", 0)
                or 0
            ),
            "xblock_qi_device_preconditioner_phase_space_residual_equation_condition_estimate": float(
                qi_device_metadata.get(
                    "phase_space_residual_equation_condition_estimate", float("inf")
                )
            ),
            "xblock_qi_device_preconditioner_residual_region_bounce_coarse": bool(
                qi_device_metadata.get("residual_region_bounce_coarse_enabled", False)
            ),
            "xblock_qi_device_preconditioner_residual_region_bounce_coarse_max_rank": int(
                qi_device_metadata.get(
                    "residual_region_bounce_coarse_max_rank_requested",
                    qi_device_metadata.get("residual_region_bounce_coarse_max_rank", 0),
                )
                or 0
            ),
            "xblock_qi_device_preconditioner_residual_region_bounce_coarse_solver": (
                qi_device_metadata.get("residual_region_bounce_coarse_solver")
            ),
            "xblock_qi_device_preconditioner_residual_region_bounce_coarse_candidate_count": int(
                qi_device_metadata.get("residual_region_bounce_coarse_candidate_count", 0)
                or 0
            ),
            "xblock_qi_device_preconditioner_residual_region_bounce_coarse_rank": int(
                qi_device_metadata.get("residual_region_bounce_coarse_rank", 0) or 0
            ),
            "xblock_qi_device_preconditioner_residual_region_bounce_coarse_stage_count": int(
                qi_device_metadata.get("residual_region_bounce_coarse_stage_count", 0)
                or 0
            ),
            "xblock_qi_device_preconditioner_residual_region_bounce_coarse_condition_estimate": float(
                qi_device_metadata.get(
                    "residual_region_bounce_coarse_condition_estimate", float("inf")
                )
            ),
            "xblock_qi_device_preconditioner_residual_region_bounce_coarse_min_region_energy_fraction": float(
                qi_device_metadata.get(
                    "residual_region_bounce_coarse_min_region_energy_fraction", float("nan")
                )
            ),
            "xblock_qi_device_preconditioner_active_pattern_coarse": bool(
                qi_device_metadata.get("active_pattern_coarse_enabled", False)
            ),
            "xblock_qi_device_preconditioner_active_pattern_coarse_max_rank": int(
                qi_device_metadata.get(
                    "active_pattern_coarse_max_rank_requested",
                    qi_device_metadata.get("active_pattern_coarse_max_rank", 0),
                )
                or 0
            ),
            "xblock_qi_device_preconditioner_active_pattern_coarse_max_candidates": int(
                qi_device_metadata.get("active_pattern_coarse_max_candidates_requested", 0)
                or 0
            ),
            "xblock_qi_device_preconditioner_active_pattern_coarse_solver": (
                qi_device_metadata.get("active_pattern_coarse_solver")
            ),
            "xblock_qi_device_preconditioner_active_pattern_coarse_candidate_count": int(
                qi_device_metadata.get("active_pattern_coarse_candidate_count", 0) or 0
            ),
            "xblock_qi_device_preconditioner_active_pattern_coarse_rank": int(
                qi_device_metadata.get("active_pattern_coarse_rank", 0) or 0
            ),
            "xblock_qi_device_preconditioner_active_pattern_coarse_stage_count": int(
                qi_device_metadata.get("active_pattern_coarse_stage_count", 0) or 0
            ),
            "xblock_qi_device_preconditioner_active_pattern_coarse_condition_estimate": float(
                qi_device_metadata.get("active_pattern_coarse_condition_estimate", float("inf"))
            ),
            "xblock_qi_device_preconditioner_active_pattern_coarse_min_chunk_energy_fraction": float(
                qi_device_metadata.get(
                    "active_pattern_coarse_min_chunk_energy_fraction", float("nan")
                )
            ),
        }
    )
    return current_result


__all__ = [
    "MatrixFreeQIDeviceSeedContext",
    "attempt_matrixfree_qi_device_seed",
]
