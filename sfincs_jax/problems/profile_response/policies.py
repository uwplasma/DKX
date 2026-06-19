"""RHSMode=1 profile-response solve-routing policy helpers."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Mapping

import jax.numpy as jnp
import numpy as np

from ...pas_smoother import pas_fast_accept as _pas_fast_accept_metric
from ...rhs1_solver_policy import (
    read_bool_env as _read_bool_env,
    read_float_env as _read_float_env,
    read_int_env as _read_int_env,
)

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def _env_token(name: str) -> str:
    return str(os.environ.get(name, "")).strip().lower()


def _env_int(name: str, default: int) -> int:
    raw = str(os.environ.get(name, "")).strip()
    try:
        return int(raw) if raw else int(default)
    except ValueError:
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = str(os.environ.get(name, "")).strip()
    try:
        return float(raw) if raw else float(default)
    except ValueError:
        return float(default)


def parse_rhs1_pas_tz_guarded_structured_levels(raw: str) -> tuple[str, ...]:
    """Parse low-memory coarse levels for guarded PAS-TZ fallback trials."""

    normalized = str(raw or "").strip().lower().replace("-", "_")
    if normalized in {"", "0", "false", "no", "off", "none"}:
        return ()
    for sep in ("+", ";", ":", "|"):
        normalized = normalized.replace(sep, ",")
    aliases = {
        "x": "xmg",
        "x_grid": "xmg",
        "xmultigrid": "xmg",
        "x_mg": "xmg",
        "coll": "collision",
        "collisions": "collision",
        "collision_diag": "collision",
        "collision_diagonal": "collision",
        "diag": "collision",
        "xmg_collision": "xmg,collision",
        "collision_xmg": "collision,xmg",
        "structured": "xmg,collision",
        "default": "xmg,collision",
    }
    expanded_tokens: list[str] = []
    for token in normalized.replace(" ", ",").split(","):
        token = token.strip("_ ")
        if not token:
            continue
        expanded = aliases.get(token, token)
        expanded_tokens.extend(
            part.strip("_ ") for part in expanded.split(",") if part.strip("_ ")
        )

    levels: list[str] = []
    for token in expanded_tokens:
        if token not in {"xmg", "collision"}:
            continue
        if token not in levels:
            levels.append(token)
    return tuple(levels)


def _qi_device_solver_env(name: str, *, default: str) -> str:
    raw = os.environ.get(name, default).strip().lower().replace("-", "_")
    if raw in {"action", "action_ls", "least_squares", "lstsq", "staged"}:
        return "action_lstsq"
    if raw in {"galerkin", "projected", "qtaq", "coarse_grid", "schur"}:
        return "galerkin"
    return default


def rhs1_qi_device_extra_coarse_controls() -> dict[str, object]:
    """Read optional QI device coarse-equation controls shared by seed hooks."""

    return {
        "multilevel_current_moments": _read_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_CURRENT_MOMENTS",
            default=False,
        ),
        "multilevel_species_current_moments": _read_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_SPECIES_CURRENT_MOMENTS",
            default=True,
        ),
        "multilevel_radial_current_moments": _read_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_RADIAL_CURRENT_MOMENTS",
            default=True,
        ),
        "multilevel_tail_constraint_moments": _read_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_TAIL_CONSTRAINT_MOMENTS",
            default=True,
        ),
        "multilevel_current_max_pitch_degree": _read_int_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_CURRENT_MAX_PITCH_DEGREE",
            default=1,
            minimum=0,
        ),
        "global_moment_residual_equation": _read_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_GLOBAL_MOMENT_RESIDUAL_EQUATION",
            default=False,
        ),
        "global_moment_residual_equation_max_rank": _read_int_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_GLOBAL_MOMENT_RESIDUAL_EQUATION_MAX_RANK",
            default=16,
            minimum=1,
        ),
        "global_moment_residual_equation_solver": _qi_device_solver_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_GLOBAL_MOMENT_RESIDUAL_EQUATION_SOLVER",
            default="galerkin",
        ),
        "global_moment_residual_equation_include_profile": _read_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_GLOBAL_MOMENT_RESIDUAL_EQUATION_INCLUDE_PROFILE",
            default=True,
        ),
        "global_moment_residual_equation_include_current": _read_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_GLOBAL_MOMENT_RESIDUAL_EQUATION_INCLUDE_CURRENT",
            default=True,
        ),
        "global_moment_residual_equation_include_tail": _read_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_GLOBAL_MOMENT_RESIDUAL_EQUATION_INCLUDE_TAIL",
            default=True,
        ),
        "residual_galerkin_equation": _read_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_GALERKIN_EQUATION",
            default=False,
        ),
        "residual_galerkin_equation_max_stages": _read_int_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_GALERKIN_EQUATION_MAX_STAGES",
            default=3,
            minimum=1,
        ),
        "residual_galerkin_equation_max_stage_rank": _read_int_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_GALERKIN_EQUATION_MAX_STAGE_RANK",
            default=4,
            minimum=1,
        ),
        "residual_galerkin_equation_max_rank": _read_int_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_GALERKIN_EQUATION_MAX_RANK",
            default=24,
            minimum=1,
        ),
        "residual_galerkin_equation_solver": _qi_device_solver_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_GALERKIN_EQUATION_SOLVER",
            default="action_lstsq",
        ),
        "residual_galerkin_equation_include_global_residual": _read_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_GALERKIN_EQUATION_INCLUDE_GLOBAL_RESIDUAL",
            default=True,
        ),
        "residual_galerkin_equation_include_block_residuals": _read_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_GALERKIN_EQUATION_INCLUDE_BLOCK_RESIDUALS",
            default=True,
        ),
        "residual_galerkin_equation_include_operator_images": _read_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_GALERKIN_EQUATION_INCLUDE_OPERATOR_IMAGES",
            default=False,
        ),
        "phase_space_residual_equation": _read_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_PHASE_SPACE_RESIDUAL_EQUATION",
            default=False,
        ),
        "phase_space_residual_equation_max_rank": _read_int_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_PHASE_SPACE_RESIDUAL_EQUATION_MAX_RANK",
            default=24,
            minimum=1,
        ),
        "phase_space_residual_equation_solver": _qi_device_solver_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_PHASE_SPACE_RESIDUAL_EQUATION_SOLVER",
            default="action_lstsq",
        ),
        "phase_space_residual_equation_include_global": _read_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_PHASE_SPACE_RESIDUAL_EQUATION_INCLUDE_GLOBAL",
            default=False,
        ),
        "phase_space_residual_equation_boundary": _read_float_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_PHASE_SPACE_RESIDUAL_EQUATION_BOUNDARY",
            default=0.35,
            minimum=1.0e-6,
        ),
        "phase_space_residual_equation_include_radial": _read_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_PHASE_SPACE_RESIDUAL_EQUATION_INCLUDE_RADIAL",
            default=True,
        ),
        "phase_space_residual_equation_include_species": _read_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_PHASE_SPACE_RESIDUAL_EQUATION_INCLUDE_SPECIES",
            default=True,
        ),
        "residual_region_bounce_coarse": _read_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_REGION_BOUNCE_COARSE",
            default=False,
        ),
        "residual_region_bounce_coarse_max_rank": _read_int_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_REGION_BOUNCE_COARSE_MAX_RANK",
            default=32,
            minimum=1,
        ),
        "residual_region_bounce_coarse_max_candidates": _read_int_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_REGION_BOUNCE_COARSE_MAX_CANDIDATES",
            default=48,
            minimum=1,
        ),
        "residual_region_bounce_coarse_solver": _qi_device_solver_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_REGION_BOUNCE_COARSE_SOLVER",
            default="action_lstsq",
        ),
        "residual_region_bounce_coarse_include_global": _read_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_REGION_BOUNCE_COARSE_INCLUDE_GLOBAL",
            default=True,
        ),
        "residual_region_bounce_coarse_include_radial": _read_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_REGION_BOUNCE_COARSE_INCLUDE_RADIAL",
            default=True,
        ),
        "residual_region_bounce_coarse_include_species": _read_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_REGION_BOUNCE_COARSE_INCLUDE_SPECIES",
            default=True,
        ),
        "residual_region_bounce_coarse_boundary": _read_float_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_REGION_BOUNCE_COARSE_BOUNCE_BOUNDARY",
            default=0.35,
            minimum=1.0e-6,
        ),
        "residual_region_bounce_coarse_min_energy": _read_float_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_REGION_BOUNCE_COARSE_MIN_REGION_ENERGY_FRACTION",
            default=1.0e-2,
            minimum=0.0,
        ),
        "residual_region_bounce_coarse_region_bands": os.environ.get(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_REGION_BOUNCE_COARSE_REGION_BANDS",
            "bounce,trapped,passing",
        ).strip(),
        "active_pattern_coarse": _read_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_ACTIVE_PATTERN_COARSE",
            default=False,
        ),
        "active_pattern_coarse_max_rank": _read_int_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_ACTIVE_PATTERN_COARSE_MAX_RANK",
            default=32,
            minimum=1,
        ),
        "active_pattern_coarse_max_candidates": _read_int_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_ACTIVE_PATTERN_COARSE_MAX_CANDIDATES",
            default=64,
            minimum=1,
        ),
        "active_pattern_coarse_solver": _qi_device_solver_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_ACTIVE_PATTERN_COARSE_SOLVER",
            default="action_lstsq",
        ),
        "active_pattern_coarse_include_global": _read_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_ACTIVE_PATTERN_COARSE_INCLUDE_GLOBAL",
            default=True,
        ),
        "active_pattern_coarse_min_chunk_energy": _read_float_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_ACTIVE_PATTERN_COARSE_MIN_CHUNK_ENERGY_FRACTION",
            default=1.0e-2,
            minimum=0.0,
        ),
        "active_pattern_coarse_include_block_pitch": _read_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_ACTIVE_PATTERN_COARSE_INCLUDE_BLOCK_PITCH",
            default=True,
        ),
        "active_pattern_coarse_include_block_angular": _read_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_ACTIVE_PATTERN_COARSE_INCLUDE_BLOCK_ANGULAR",
            default=True,
        ),
        "active_pattern_coarse_include_radial_pitch": _read_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_ACTIVE_PATTERN_COARSE_INCLUDE_RADIAL_PITCH",
            default=True,
        ),
        "active_pattern_coarse_include_radial_angular": _read_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_ACTIVE_PATTERN_COARSE_INCLUDE_RADIAL_ANGULAR",
            default=True,
        ),
        "active_pattern_coarse_include_block": _read_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_ACTIVE_PATTERN_COARSE_INCLUDE_BLOCK",
            default=True,
        ),
        "active_pattern_coarse_include_radial": _read_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_ACTIVE_PATTERN_COARSE_INCLUDE_RADIAL",
            default=True,
        ),
        "active_pattern_coarse_include_species": _read_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_ACTIVE_PATTERN_COARSE_INCLUDE_SPECIES",
            default=True,
        ),
    }


def rhs1_qi_device_residual_correction_controls() -> dict[str, object]:
    """Read optional QI device residual-correction enrichment controls."""

    return {
        "block_schur_residual_equation": _read_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_BLOCK_SCHUR_RESIDUAL_EQUATION",
            default=False,
        ),
        "block_schur_residual_equation_max_rank": _read_int_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_BLOCK_SCHUR_RESIDUAL_EQUATION_MAX_RANK",
            default=24,
            minimum=1,
        ),
        "block_schur_residual_equation_include_global": _read_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_BLOCK_SCHUR_RESIDUAL_EQUATION_INCLUDE_GLOBAL",
            default=False,
        ),
        "block_schur_residual_equation_include_blocks": _read_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_BLOCK_SCHUR_RESIDUAL_EQUATION_INCLUDE_BLOCKS",
            default=True,
        ),
        "block_schur_residual_equation_include_aggregates": _read_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_BLOCK_SCHUR_RESIDUAL_EQUATION_INCLUDE_AGGREGATES",
            default=True,
        ),
        "coupled_residual_equation": _read_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_COUPLED_RESIDUAL_EQUATION",
            default=False,
        ),
        "coupled_residual_equation_max_rank": _read_int_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_COUPLED_RESIDUAL_EQUATION_MAX_RANK",
            default=96,
            minimum=1,
        ),
        "coupled_residual_equation_solver": _qi_device_solver_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_COUPLED_RESIDUAL_EQUATION_SOLVER",
            default="action_lstsq",
        ),
        "coupled_residual_equation_include_flat": _read_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_COUPLED_RESIDUAL_EQUATION_INCLUDE_FLAT",
            default=True,
        ),
        "coupled_residual_equation_min_improvement": _read_float_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_COUPLED_RESIDUAL_EQUATION_MIN_RELATIVE_IMPROVEMENT",
            default=0.0,
            minimum=0.0,
        ),
        "coupled_residual_equation_install_on_reject": _read_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_COUPLED_RESIDUAL_EQUATION_INSTALL_IN_KRYLOV_ON_REJECT",
            default=False,
        ),
        "residual_snapshot_enrichment": _read_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_ENRICHMENT",
            default=False,
        ),
        "residual_snapshot_max_rank": _read_int_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_MAX_RANK",
            default=24,
            minimum=1,
        ),
        "residual_snapshot_include_primal": _read_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_INCLUDE_PRIMAL",
            default=True,
        ),
        "residual_snapshot_use_adjoint": _read_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_USE_ADJOINT",
            default=False,
        ),
        "residual_snapshot_include_global": _read_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_INCLUDE_GLOBAL",
            default=False,
        ),
        "residual_snapshot_include_blocks": _read_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_INCLUDE_BLOCKS",
            default=True,
        ),
        "residual_snapshot_include_aggregates": _read_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_INCLUDE_AGGREGATES",
            default=True,
        ),
        "residual_snapshot_residual_equation": _read_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_RESIDUAL_EQUATION",
            default=False,
        ),
        "residual_snapshot_residual_equation_max_rank": _read_int_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_RESIDUAL_EQUATION_MAX_RANK",
            default=24,
            minimum=1,
        ),
        "residual_snapshot_residual_equation_solver": _qi_device_solver_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_RESIDUAL_EQUATION_SOLVER",
            default="action_lstsq",
        ),
        "residual_snapshot_residual_equation_include_global": _read_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_RESIDUAL_EQUATION_INCLUDE_GLOBAL",
            default=False,
        ),
        "block_schur_residual_enrichment": _read_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_BLOCK_SCHUR_RESIDUAL_ENRICHMENT",
            default=False,
        ),
        "block_schur_residual_max_rank": _read_int_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_BLOCK_SCHUR_RESIDUAL_MAX_RANK",
            default=24,
            minimum=1,
        ),
        "block_schur_residual_include_global": _read_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_BLOCK_SCHUR_RESIDUAL_INCLUDE_GLOBAL",
            default=False,
        ),
        "block_schur_residual_include_blocks": _read_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_BLOCK_SCHUR_RESIDUAL_INCLUDE_BLOCKS",
            default=True,
        ),
        "block_schur_residual_include_aggregates": _read_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_BLOCK_SCHUR_RESIDUAL_INCLUDE_AGGREGATES",
            default=True,
        ),
    }


def rhs1_qi_device_extra_coarse_setup_kwargs(
    controls: Mapping[str, object],
) -> dict[str, object]:
    """Return setup kwargs for QI-device extra coarse/residual controls."""

    return {
        "global_moment_residual_equation": bool(
            controls["global_moment_residual_equation"]
        ),
        "global_moment_residual_equation_max_rank": int(
            controls["global_moment_residual_equation_max_rank"]
        ),
        "global_moment_residual_equation_solver": str(
            controls["global_moment_residual_equation_solver"]
        ),
        "global_moment_residual_equation_include_profile": bool(
            controls["global_moment_residual_equation_include_profile"]
        ),
        "global_moment_residual_equation_include_current": bool(
            controls["global_moment_residual_equation_include_current"]
        ),
        "global_moment_residual_equation_include_tail": bool(
            controls["global_moment_residual_equation_include_tail"]
        ),
        "residual_galerkin_equation": bool(controls["residual_galerkin_equation"]),
        "residual_galerkin_equation_max_stages": int(
            controls["residual_galerkin_equation_max_stages"]
        ),
        "residual_galerkin_equation_max_stage_rank": int(
            controls["residual_galerkin_equation_max_stage_rank"]
        ),
        "residual_galerkin_equation_max_rank": int(
            controls["residual_galerkin_equation_max_rank"]
        ),
        "residual_galerkin_equation_solver": str(
            controls["residual_galerkin_equation_solver"]
        ),
        "residual_galerkin_equation_include_global_residual": bool(
            controls["residual_galerkin_equation_include_global_residual"]
        ),
        "residual_galerkin_equation_include_block_residuals": bool(
            controls["residual_galerkin_equation_include_block_residuals"]
        ),
        "residual_galerkin_equation_include_operator_images": bool(
            controls["residual_galerkin_equation_include_operator_images"]
        ),
        "phase_space_residual_equation": bool(
            controls["phase_space_residual_equation"]
        ),
        "phase_space_residual_equation_max_rank": int(
            controls["phase_space_residual_equation_max_rank"]
        ),
        "phase_space_residual_equation_solver": str(
            controls["phase_space_residual_equation_solver"]
        ),
        "phase_space_residual_equation_include_global": bool(
            controls["phase_space_residual_equation_include_global"]
        ),
        "phase_space_residual_equation_trapped_boundary_fraction": float(
            controls["phase_space_residual_equation_boundary"]
        ),
        "phase_space_residual_equation_include_radial": bool(
            controls["phase_space_residual_equation_include_radial"]
        ),
        "phase_space_residual_equation_include_species": bool(
            controls["phase_space_residual_equation_include_species"]
        ),
        "residual_region_bounce_coarse": bool(
            controls["residual_region_bounce_coarse"]
        ),
        "residual_region_bounce_coarse_max_rank": int(
            controls["residual_region_bounce_coarse_max_rank"]
        ),
        "residual_region_bounce_coarse_max_candidates": int(
            controls["residual_region_bounce_coarse_max_candidates"]
        ),
        "residual_region_bounce_coarse_solver": str(
            controls["residual_region_bounce_coarse_solver"]
        ),
        "residual_region_bounce_coarse_include_global": bool(
            controls["residual_region_bounce_coarse_include_global"]
        ),
        "residual_region_bounce_coarse_include_radial": bool(
            controls["residual_region_bounce_coarse_include_radial"]
        ),
        "residual_region_bounce_coarse_include_species": bool(
            controls["residual_region_bounce_coarse_include_species"]
        ),
        "residual_region_bounce_coarse_trapped_boundary_fraction": float(
            controls["residual_region_bounce_coarse_boundary"]
        ),
        "residual_region_bounce_coarse_min_region_energy_fraction": float(
            controls["residual_region_bounce_coarse_min_energy"]
        ),
        "residual_region_bounce_coarse_region_bands": str(
            controls["residual_region_bounce_coarse_region_bands"]
        ),
        "active_pattern_coarse": bool(controls["active_pattern_coarse"]),
        "active_pattern_coarse_max_rank": int(
            controls["active_pattern_coarse_max_rank"]
        ),
        "active_pattern_coarse_max_candidates": int(
            controls["active_pattern_coarse_max_candidates"]
        ),
        "active_pattern_coarse_solver": str(controls["active_pattern_coarse_solver"]),
        "active_pattern_coarse_min_chunk_energy_fraction": float(
            controls["active_pattern_coarse_min_chunk_energy"]
        ),
        "active_pattern_coarse_include_global": bool(
            controls["active_pattern_coarse_include_global"]
        ),
        "active_pattern_coarse_include_block_pitch": bool(
            controls["active_pattern_coarse_include_block_pitch"]
        ),
        "active_pattern_coarse_include_block_angular": bool(
            controls["active_pattern_coarse_include_block_angular"]
        ),
        "active_pattern_coarse_include_radial_pitch": bool(
            controls["active_pattern_coarse_include_radial_pitch"]
        ),
        "active_pattern_coarse_include_radial_angular": bool(
            controls["active_pattern_coarse_include_radial_angular"]
        ),
        "active_pattern_coarse_include_block": bool(
            controls["active_pattern_coarse_include_block"]
        ),
        "active_pattern_coarse_include_radial": bool(
            controls["active_pattern_coarse_include_radial"]
        ),
        "active_pattern_coarse_include_species": bool(
            controls["active_pattern_coarse_include_species"]
        ),
    }


def rhs1_qi_device_residual_correction_setup_kwargs(
    controls: Mapping[str, object],
) -> dict[str, object]:
    """Return setup kwargs for QI-device residual-correction controls."""

    return {
        "block_schur_residual_equation": bool(
            controls["block_schur_residual_equation"]
        ),
        "block_schur_residual_equation_max_rank": int(
            controls["block_schur_residual_equation_max_rank"]
        ),
        "block_schur_residual_equation_include_global": bool(
            controls["block_schur_residual_equation_include_global"]
        ),
        "block_schur_residual_equation_include_blocks": bool(
            controls["block_schur_residual_equation_include_blocks"]
        ),
        "block_schur_residual_equation_include_aggregates": bool(
            controls["block_schur_residual_equation_include_aggregates"]
        ),
        "coupled_residual_equation": bool(controls["coupled_residual_equation"]),
        "coupled_residual_equation_max_rank": int(
            controls["coupled_residual_equation_max_rank"]
        ),
        "coupled_residual_equation_solver": str(
            controls["coupled_residual_equation_solver"]
        ),
        "coupled_residual_equation_include_flat": bool(
            controls["coupled_residual_equation_include_flat"]
        ),
        "coupled_residual_equation_min_relative_improvement": float(
            controls["coupled_residual_equation_min_improvement"]
        ),
        "residual_snapshot_enrichment": bool(
            controls["residual_snapshot_enrichment"]
        ),
        "residual_snapshot_max_rank": int(controls["residual_snapshot_max_rank"]),
        "residual_snapshot_include_primal": bool(
            controls["residual_snapshot_include_primal"]
        ),
        "residual_snapshot_use_adjoint": bool(
            controls["residual_snapshot_use_adjoint"]
        ),
        "residual_snapshot_include_global": bool(
            controls["residual_snapshot_include_global"]
        ),
        "residual_snapshot_include_blocks": bool(
            controls["residual_snapshot_include_blocks"]
        ),
        "residual_snapshot_include_aggregates": bool(
            controls["residual_snapshot_include_aggregates"]
        ),
        "residual_snapshot_residual_equation": bool(
            controls["residual_snapshot_residual_equation"]
        ),
        "residual_snapshot_residual_equation_max_rank": int(
            controls["residual_snapshot_residual_equation_max_rank"]
        ),
        "residual_snapshot_residual_equation_solver": str(
            controls["residual_snapshot_residual_equation_solver"]
        ),
        "residual_snapshot_residual_equation_include_global": bool(
            controls["residual_snapshot_residual_equation_include_global"]
        ),
        "block_schur_residual_enrichment": bool(
            controls["block_schur_residual_enrichment"]
        ),
        "block_schur_residual_max_rank": int(
            controls["block_schur_residual_max_rank"]
        ),
        "block_schur_residual_include_global": bool(
            controls["block_schur_residual_include_global"]
        ),
        "block_schur_residual_include_blocks": bool(
            controls["block_schur_residual_include_blocks"]
        ),
        "block_schur_residual_include_aggregates": bool(
            controls["block_schur_residual_include_aggregates"]
        ),
    }


def rhs1_qi_device_extra_coarse_metadata(
    controls: Mapping[str, object],
) -> dict[str, object]:
    """Return requested-control metadata for QI-device extra coarse spaces."""

    return {
        "global_moment_residual_equation_requested": bool(
            controls["global_moment_residual_equation"]
        ),
        "global_moment_residual_equation_max_rank_requested": int(
            controls["global_moment_residual_equation_max_rank"]
        ),
        "global_moment_residual_equation_solver_requested": str(
            controls["global_moment_residual_equation_solver"]
        ),
        "global_moment_residual_equation_include_profile_requested": bool(
            controls["global_moment_residual_equation_include_profile"]
        ),
        "global_moment_residual_equation_include_current_requested": bool(
            controls["global_moment_residual_equation_include_current"]
        ),
        "global_moment_residual_equation_include_tail_requested": bool(
            controls["global_moment_residual_equation_include_tail"]
        ),
        "residual_galerkin_equation_requested": bool(
            controls["residual_galerkin_equation"]
        ),
        "residual_galerkin_equation_max_stages_requested": int(
            controls["residual_galerkin_equation_max_stages"]
        ),
        "residual_galerkin_equation_max_stage_rank_requested": int(
            controls["residual_galerkin_equation_max_stage_rank"]
        ),
        "residual_galerkin_equation_max_rank_requested": int(
            controls["residual_galerkin_equation_max_rank"]
        ),
        "residual_galerkin_equation_solver_requested": str(
            controls["residual_galerkin_equation_solver"]
        ),
        "residual_galerkin_equation_include_global_residual_requested": bool(
            controls["residual_galerkin_equation_include_global_residual"]
        ),
        "residual_galerkin_equation_include_block_residuals_requested": bool(
            controls["residual_galerkin_equation_include_block_residuals"]
        ),
        "residual_galerkin_equation_include_operator_images_requested": bool(
            controls["residual_galerkin_equation_include_operator_images"]
        ),
        "phase_space_residual_equation_requested": bool(
            controls["phase_space_residual_equation"]
        ),
        "phase_space_residual_equation_max_rank_requested": int(
            controls["phase_space_residual_equation_max_rank"]
        ),
        "phase_space_residual_equation_solver_requested": str(
            controls["phase_space_residual_equation_solver"]
        ),
        "phase_space_residual_equation_include_global_requested": bool(
            controls["phase_space_residual_equation_include_global"]
        ),
        "phase_space_residual_equation_boundary_requested": float(
            controls["phase_space_residual_equation_boundary"]
        ),
        "phase_space_residual_equation_include_radial_requested": bool(
            controls["phase_space_residual_equation_include_radial"]
        ),
        "phase_space_residual_equation_include_species_requested": bool(
            controls["phase_space_residual_equation_include_species"]
        ),
        "residual_region_bounce_coarse_requested": bool(
            controls["residual_region_bounce_coarse"]
        ),
        "residual_region_bounce_coarse_max_rank_requested": int(
            controls["residual_region_bounce_coarse_max_rank"]
        ),
        "residual_region_bounce_coarse_max_candidates_requested": int(
            controls["residual_region_bounce_coarse_max_candidates"]
        ),
        "residual_region_bounce_coarse_solver_requested": str(
            controls["residual_region_bounce_coarse_solver"]
        ),
        "residual_region_bounce_coarse_include_global_requested": bool(
            controls["residual_region_bounce_coarse_include_global"]
        ),
        "residual_region_bounce_coarse_include_radial_requested": bool(
            controls["residual_region_bounce_coarse_include_radial"]
        ),
        "residual_region_bounce_coarse_include_species_requested": bool(
            controls["residual_region_bounce_coarse_include_species"]
        ),
        "residual_region_bounce_coarse_boundary_requested": float(
            controls["residual_region_bounce_coarse_boundary"]
        ),
        "residual_region_bounce_coarse_min_region_energy_fraction_requested": float(
            controls["residual_region_bounce_coarse_min_energy"]
        ),
        "residual_region_bounce_coarse_region_bands_requested": str(
            controls["residual_region_bounce_coarse_region_bands"]
        ),
        "active_pattern_coarse_requested": bool(controls["active_pattern_coarse"]),
        "active_pattern_coarse_max_rank_requested": int(
            controls["active_pattern_coarse_max_rank"]
        ),
        "active_pattern_coarse_max_candidates_requested": int(
            controls["active_pattern_coarse_max_candidates"]
        ),
        "active_pattern_coarse_solver_requested": str(
            controls["active_pattern_coarse_solver"]
        ),
        "active_pattern_coarse_include_global_requested": bool(
            controls["active_pattern_coarse_include_global"]
        ),
        "active_pattern_coarse_min_chunk_energy_fraction_requested": float(
            controls["active_pattern_coarse_min_chunk_energy"]
        ),
        "active_pattern_coarse_include_block_pitch_requested": bool(
            controls["active_pattern_coarse_include_block_pitch"]
        ),
        "active_pattern_coarse_include_block_angular_requested": bool(
            controls["active_pattern_coarse_include_block_angular"]
        ),
        "active_pattern_coarse_include_radial_pitch_requested": bool(
            controls["active_pattern_coarse_include_radial_pitch"]
        ),
        "active_pattern_coarse_include_radial_angular_requested": bool(
            controls["active_pattern_coarse_include_radial_angular"]
        ),
        "active_pattern_coarse_include_block_requested": bool(
            controls["active_pattern_coarse_include_block"]
        ),
        "active_pattern_coarse_include_radial_requested": bool(
            controls["active_pattern_coarse_include_radial"]
        ),
        "active_pattern_coarse_include_species_requested": bool(
            controls["active_pattern_coarse_include_species"]
        ),
    }


def rhs1_qi_device_residual_correction_metadata(
    controls: Mapping[str, object],
) -> dict[str, object]:
    """Return requested-control metadata for QI-device residual corrections."""

    return {
        "block_schur_residual_equation_requested": bool(
            controls["block_schur_residual_equation"]
        ),
        "block_schur_residual_equation_max_rank_requested": int(
            controls["block_schur_residual_equation_max_rank"]
        ),
        "block_schur_residual_equation_include_global_requested": bool(
            controls["block_schur_residual_equation_include_global"]
        ),
        "block_schur_residual_equation_include_blocks_requested": bool(
            controls["block_schur_residual_equation_include_blocks"]
        ),
        "block_schur_residual_equation_include_aggregates_requested": bool(
            controls["block_schur_residual_equation_include_aggregates"]
        ),
        "coupled_residual_equation_requested": bool(
            controls["coupled_residual_equation"]
        ),
        "coupled_residual_equation_max_rank_requested": int(
            controls["coupled_residual_equation_max_rank"]
        ),
        "coupled_residual_equation_solver_requested": str(
            controls["coupled_residual_equation_solver"]
        ),
        "coupled_residual_equation_include_flat_requested": bool(
            controls["coupled_residual_equation_include_flat"]
        ),
        "coupled_residual_equation_min_relative_improvement_requested": float(
            controls["coupled_residual_equation_min_improvement"]
        ),
        "coupled_residual_equation_install_in_krylov_on_reject_requested": bool(
            controls["coupled_residual_equation_install_on_reject"]
        ),
        "residual_snapshot_enrichment_requested": bool(
            controls["residual_snapshot_enrichment"]
        ),
        "residual_snapshot_max_rank_requested": int(
            controls["residual_snapshot_max_rank"]
        ),
        "residual_snapshot_include_primal_requested": bool(
            controls["residual_snapshot_include_primal"]
        ),
        "residual_snapshot_use_adjoint_requested": bool(
            controls["residual_snapshot_use_adjoint"]
        ),
        "residual_snapshot_include_global_requested": bool(
            controls["residual_snapshot_include_global"]
        ),
        "residual_snapshot_include_blocks_requested": bool(
            controls["residual_snapshot_include_blocks"]
        ),
        "residual_snapshot_include_aggregates_requested": bool(
            controls["residual_snapshot_include_aggregates"]
        ),
        "residual_snapshot_residual_equation_requested": bool(
            controls["residual_snapshot_residual_equation"]
        ),
        "residual_snapshot_residual_equation_max_rank_requested": int(
            controls["residual_snapshot_residual_equation_max_rank"]
        ),
        "residual_snapshot_residual_equation_solver_requested": str(
            controls["residual_snapshot_residual_equation_solver"]
        ),
        "residual_snapshot_residual_equation_include_global_requested": bool(
            controls["residual_snapshot_residual_equation_include_global"]
        ),
        "block_schur_residual_enrichment_requested": bool(
            controls["block_schur_residual_enrichment"]
        ),
        "block_schur_residual_max_rank_requested": int(
            controls["block_schur_residual_max_rank"]
        ),
        "block_schur_residual_include_global_requested": bool(
            controls["block_schur_residual_include_global"]
        ),
        "block_schur_residual_include_blocks_requested": bool(
            controls["block_schur_residual_include_blocks"]
        ),
        "block_schur_residual_include_aggregates_requested": bool(
            controls["block_schur_residual_include_aggregates"]
        ),
    }


def rhs1_qi_device_tail_block_required(
    *,
    multilevel_coarse: bool,
    extra_coarse_controls: Mapping[str, object],
) -> bool:
    """Return whether QI-device block metadata needs the tail block retained."""

    return bool(
        multilevel_coarse
        or (
            bool(extra_coarse_controls["global_moment_residual_equation"])
            and bool(extra_coarse_controls["global_moment_residual_equation_include_tail"])
        )
    )


def rhs1_qi_device_coupled_install_on_reject_requested(
    residual_correction_controls: Mapping[str, object],
) -> bool:
    """Return whether an accepted coupled residual stage may install after reject."""

    return bool(residual_correction_controls["coupled_residual_equation_install_on_reject"])


def rhs1_qi_device_status_fields(
    *,
    extra_coarse_controls: Mapping[str, object],
    residual_correction_controls: Mapping[str, object],
    metadata: Mapping[str, object],
) -> str:
    """Format stable QI-device status fields from grouped controls and metadata."""

    def _residual_requested(name: str) -> bool:
        return bool(residual_correction_controls.get(name, False))

    return (
        f"global_moment_equation={int(bool(extra_coarse_controls['global_moment_residual_equation']))} "
        f"global_moment_rank={int(metadata.get('global_moment_residual_equation_rank', 0))} "
        f"global_moment_candidates={int(metadata.get('global_moment_residual_equation_candidate_count', 0))} "
        f"global_moment_cond={float(metadata.get('global_moment_residual_equation_condition_estimate', float('inf'))):.6e} "
        f"residual_galerkin_equation={int(bool(extra_coarse_controls['residual_galerkin_equation']))} "
        f"residual_galerkin_rank={int(metadata.get('residual_galerkin_equation_rank', 0))} "
        f"residual_galerkin_candidates={int(metadata.get('residual_galerkin_equation_candidate_count', 0))} "
        f"phase_space_equation={int(bool(extra_coarse_controls['phase_space_residual_equation']))} "
        f"phase_space_rank={int(metadata.get('phase_space_residual_equation_rank', 0))} "
        f"phase_space_candidates={int(metadata.get('phase_space_residual_equation_candidate_count', 0))} "
        f"residual_region_bounce={int(bool(extra_coarse_controls['residual_region_bounce_coarse']))} "
        f"residual_region_bounce_rank={int(metadata.get('residual_region_bounce_coarse_rank', 0))} "
        f"residual_region_bounce_candidates={int(metadata.get('residual_region_bounce_coarse_candidate_count', 0))} "
        f"active_pattern_coarse={int(bool(extra_coarse_controls['active_pattern_coarse']))} "
        f"active_pattern_rank={int(metadata.get('active_pattern_coarse_rank', 0))} "
        f"active_pattern_candidates={int(metadata.get('active_pattern_coarse_candidate_count', 0))} "
        f"block_schur_equation={int(_residual_requested('block_schur_residual_equation'))} "
        f"coupled_equation={int(_residual_requested('coupled_residual_equation'))} "
        f"coupled_rank={int(metadata.get('coupled_residual_equation_rank', 0))} "
        f"coupled_candidates={int(metadata.get('coupled_residual_equation_candidate_count', 0))} "
        f"residual_snapshot={int(_residual_requested('residual_snapshot_enrichment'))} "
        f"residual_snapshot_equation={int(_residual_requested('residual_snapshot_residual_equation'))} "
        f"block_schur={int(_residual_requested('block_schur_residual_enrichment'))}"
    )


@dataclass(frozen=True)
class RHS1QIDeviceRankBudget:
    """Rank budget and optional rank cap for a QI-device coarse space."""

    rank_budget: int
    max_rank: int | None


@dataclass(frozen=True)
class RHS1QIDeviceSetupSummary:
    """Derived QI-device setup policy shared by the main RHSMode=1 branch."""

    rank_budget: int
    max_rank: int | None
    progress_messages: tuple[str, ...]
    residual_seed_required: bool


def rhs1_qi_device_rank_budget(
    *,
    seed_max_rank: int,
    n_species: int,
    residual_enrichment: bool,
    residual_enrichment_depth: int,
    residual_enrichment_include_residual: bool,
    recycle_enrichment: bool,
    recycle_cycles: int,
    operator_krylov_enrichment: bool,
    operator_krylov_depth: int,
    adjoint_krylov_enrichment: bool,
    adjoint_krylov_depth: int,
    operator_action_enrichment: bool,
    operator_action_depth: int,
    multilevel_coarse: bool,
    multilevel_max_rank: int | None,
    multilevel_current_moments: bool,
    multilevel_current_max_pitch_degree: int,
    multilevel_residual_equation: bool,
    multilevel_residual_equation_max_level_rank: int,
    multilevel_max_levels: int,
    global_moment_residual_equation: bool,
    global_moment_residual_equation_max_rank: int,
    residual_galerkin_equation: bool,
    residual_galerkin_equation_max_rank: int,
    phase_space_residual_equation: bool,
    phase_space_residual_equation_max_rank: int,
    residual_region_bounce_coarse: bool,
    residual_region_bounce_coarse_max_rank: int,
    active_pattern_coarse: bool,
    active_pattern_coarse_max_rank: int,
    block_schur_residual_equation: bool,
    block_schur_residual_equation_max_rank: int,
    coupled_residual_equation: bool,
    coupled_residual_equation_max_rank: int,
    residual_snapshot_enrichment: bool,
    residual_snapshot_max_rank: int,
    residual_snapshot_residual_equation: bool,
    residual_snapshot_residual_equation_max_rank: int,
    block_schur_residual_enrichment: bool,
    block_schur_residual_max_rank: int,
    max_rank_env_value: str | None = None,
) -> RHS1QIDeviceRankBudget:
    """Compute the QI-device coarse-space rank budget and user rank cap."""

    rank_budget = int(seed_max_rank)
    if bool(residual_enrichment):
        rank_budget += int(residual_enrichment_depth)
        if bool(residual_enrichment_include_residual):
            rank_budget += 1
    if bool(recycle_enrichment):
        rank_budget += int(recycle_cycles)
    if bool(operator_krylov_enrichment):
        rank_budget += 1 + int(operator_krylov_depth)
    if bool(adjoint_krylov_enrichment):
        rank_budget += 1 + int(adjoint_krylov_depth)
    if bool(operator_action_enrichment):
        rank_budget *= max(1, 1 + int(operator_action_depth))
    if bool(multilevel_coarse):
        rank_budget += int(multilevel_max_rank or 48)
        if bool(multilevel_current_moments):
            rank_budget += max(1, int(multilevel_current_max_pitch_degree)) * (
                2 * max(1, int(n_species)) + 2
            )
    if bool(multilevel_residual_equation):
        rank_budget += int(multilevel_residual_equation_max_level_rank) * int(
            multilevel_max_levels
        )
    if bool(global_moment_residual_equation):
        rank_budget += int(global_moment_residual_equation_max_rank)
    if bool(residual_galerkin_equation):
        rank_budget += int(residual_galerkin_equation_max_rank)
    if bool(phase_space_residual_equation):
        rank_budget += int(phase_space_residual_equation_max_rank)
    if bool(residual_region_bounce_coarse):
        rank_budget += int(residual_region_bounce_coarse_max_rank)
    if bool(active_pattern_coarse):
        rank_budget += int(active_pattern_coarse_max_rank)
    if bool(block_schur_residual_equation):
        rank_budget += int(block_schur_residual_equation_max_rank)
    if bool(coupled_residual_equation):
        rank_budget += int(coupled_residual_equation_max_rank)
    if bool(residual_snapshot_enrichment):
        rank_budget += int(residual_snapshot_max_rank)
    if bool(residual_snapshot_residual_equation):
        rank_budget += int(residual_snapshot_residual_equation_max_rank)
    if bool(block_schur_residual_enrichment):
        rank_budget += int(block_schur_residual_max_rank)

    raw_max_rank = (
        os.environ.get(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MAX_RANK",
            "",
        )
        if max_rank_env_value is None
        else str(max_rank_env_value)
    ).strip()
    if raw_max_rank:
        try:
            max_rank = max(1, int(raw_max_rank))
        except ValueError:
            max_rank = max(1, int(rank_budget))
    elif (
        bool(residual_enrichment)
        or bool(recycle_enrichment)
        or bool(operator_krylov_enrichment)
        or bool(operator_action_enrichment)
        or bool(multilevel_coarse)
        or bool(multilevel_residual_equation)
        or bool(global_moment_residual_equation)
        or bool(residual_galerkin_equation)
        or bool(phase_space_residual_equation)
        or bool(residual_region_bounce_coarse)
        or bool(active_pattern_coarse)
        or bool(block_schur_residual_equation)
        or bool(coupled_residual_equation)
        or bool(residual_snapshot_enrichment)
        or bool(residual_snapshot_residual_equation)
        or bool(block_schur_residual_enrichment)
    ):
        max_rank = max(1, int(rank_budget))
    else:
        max_rank = None

    return RHS1QIDeviceRankBudget(rank_budget=max(1, int(rank_budget)), max_rank=max_rank)


def rhs1_qi_device_setup_summary(
    *,
    seed_max_rank: int,
    n_species: int,
    assembled_device_operator_available: bool,
    enrichment_config: Any,
    multilevel_config: Any,
    multilevel_max_rank: int | None,
    extra_coarse_controls: Mapping[str, object],
    residual_correction_controls: Mapping[str, object],
    max_rank_env_value: str | None = None,
) -> RHS1QIDeviceSetupSummary:
    """Return rank, progress, and residual-seed policy from resolved controls."""

    rank_budget_setup = rhs1_qi_device_rank_budget(
        seed_max_rank=int(seed_max_rank),
        n_species=int(n_species),
        residual_enrichment=bool(enrichment_config.residual_enrichment),
        residual_enrichment_depth=int(enrichment_config.residual_enrichment_depth),
        residual_enrichment_include_residual=bool(
            enrichment_config.residual_enrichment_include_residual
        ),
        recycle_enrichment=bool(enrichment_config.recycle_enrichment),
        recycle_cycles=int(enrichment_config.recycle_cycles),
        operator_krylov_enrichment=bool(enrichment_config.operator_krylov_enrichment),
        operator_krylov_depth=int(enrichment_config.operator_krylov_depth),
        adjoint_krylov_enrichment=bool(enrichment_config.adjoint_krylov_enrichment),
        adjoint_krylov_depth=int(enrichment_config.adjoint_krylov_depth),
        operator_action_enrichment=bool(enrichment_config.operator_action_enrichment),
        operator_action_depth=int(enrichment_config.operator_action_depth),
        multilevel_coarse=bool(multilevel_config.multilevel_coarse),
        multilevel_max_rank=multilevel_max_rank,
        multilevel_current_moments=bool(multilevel_config.multilevel_current_moments),
        multilevel_current_max_pitch_degree=int(
            multilevel_config.multilevel_current_max_pitch_degree
        ),
        multilevel_residual_equation=bool(
            multilevel_config.multilevel_residual_equation
        ),
        multilevel_residual_equation_max_level_rank=int(
            multilevel_config.multilevel_residual_equation_max_level_rank
        ),
        multilevel_max_levels=int(multilevel_config.multilevel_max_levels),
        global_moment_residual_equation=bool(
            extra_coarse_controls["global_moment_residual_equation"]
        ),
        global_moment_residual_equation_max_rank=int(
            extra_coarse_controls["global_moment_residual_equation_max_rank"]
        ),
        residual_galerkin_equation=bool(
            extra_coarse_controls["residual_galerkin_equation"]
        ),
        residual_galerkin_equation_max_rank=int(
            extra_coarse_controls["residual_galerkin_equation_max_rank"]
        ),
        phase_space_residual_equation=bool(
            extra_coarse_controls["phase_space_residual_equation"]
        ),
        phase_space_residual_equation_max_rank=int(
            extra_coarse_controls["phase_space_residual_equation_max_rank"]
        ),
        residual_region_bounce_coarse=bool(
            extra_coarse_controls["residual_region_bounce_coarse"]
        ),
        residual_region_bounce_coarse_max_rank=int(
            extra_coarse_controls["residual_region_bounce_coarse_max_rank"]
        ),
        active_pattern_coarse=bool(extra_coarse_controls["active_pattern_coarse"]),
        active_pattern_coarse_max_rank=int(
            extra_coarse_controls["active_pattern_coarse_max_rank"]
        ),
        block_schur_residual_equation=bool(
            residual_correction_controls["block_schur_residual_equation"]
        ),
        block_schur_residual_equation_max_rank=int(
            residual_correction_controls["block_schur_residual_equation_max_rank"]
        ),
        coupled_residual_equation=bool(
            residual_correction_controls["coupled_residual_equation"]
        ),
        coupled_residual_equation_max_rank=int(
            residual_correction_controls["coupled_residual_equation_max_rank"]
        ),
        residual_snapshot_enrichment=bool(
            residual_correction_controls["residual_snapshot_enrichment"]
        ),
        residual_snapshot_max_rank=int(
            residual_correction_controls["residual_snapshot_max_rank"]
        ),
        residual_snapshot_residual_equation=bool(
            residual_correction_controls["residual_snapshot_residual_equation"]
        ),
        residual_snapshot_residual_equation_max_rank=int(
            residual_correction_controls["residual_snapshot_residual_equation_max_rank"]
        ),
        block_schur_residual_enrichment=bool(
            residual_correction_controls["block_schur_residual_enrichment"]
        ),
        block_schur_residual_max_rank=int(
            residual_correction_controls["block_schur_residual_max_rank"]
        ),
        max_rank_env_value=max_rank_env_value,
    )
    progress_messages = rhs1_qi_device_progress_messages(
        assembled_device_operator_available=bool(assembled_device_operator_available),
        residual_enrichment=bool(enrichment_config.residual_enrichment),
        residual_enrichment_depth=int(enrichment_config.residual_enrichment_depth),
        operator_action_enrichment=bool(enrichment_config.operator_action_enrichment),
        operator_action_depth=int(enrichment_config.operator_action_depth),
        operator_krylov_enrichment=bool(enrichment_config.operator_krylov_enrichment),
        operator_krylov_depth=int(enrichment_config.operator_krylov_depth),
        adjoint_krylov_enrichment=bool(enrichment_config.adjoint_krylov_enrichment),
        adjoint_krylov_depth=int(enrichment_config.adjoint_krylov_depth),
        adjoint_krylov_transpose_source=str(
            enrichment_config.adjoint_krylov_transpose_source
        ),
        max_rank=rank_budget_setup.max_rank,
        multilevel_coarse=bool(multilevel_config.multilevel_coarse),
        multilevel_max_levels=int(multilevel_config.multilevel_max_levels),
        multilevel_aggregate_factor=int(multilevel_config.multilevel_aggregate_factor),
        multilevel_max_pitch_degree=int(multilevel_config.multilevel_max_pitch_degree),
        multilevel_current_moments=bool(multilevel_config.multilevel_current_moments),
        multilevel_max_rank=multilevel_max_rank,
        multilevel_residual_equation=bool(
            multilevel_config.multilevel_residual_equation
        ),
        multilevel_residual_equation_max_level_rank=int(
            multilevel_config.multilevel_residual_equation_max_level_rank
        ),
        multilevel_residual_equation_order=str(
            multilevel_config.multilevel_residual_equation_order
        ),
        multilevel_residual_equation_solver=str(
            multilevel_config.multilevel_residual_equation_solver
        ),
        multilevel_residual_equation_include_global=bool(
            multilevel_config.multilevel_residual_equation_include_global
        ),
        global_moment_residual_equation=bool(
            extra_coarse_controls["global_moment_residual_equation"]
        ),
        global_moment_residual_equation_max_rank=int(
            extra_coarse_controls["global_moment_residual_equation_max_rank"]
        ),
        global_moment_residual_equation_solver=str(
            extra_coarse_controls["global_moment_residual_equation_solver"]
        ),
        global_moment_residual_equation_include_profile=bool(
            extra_coarse_controls["global_moment_residual_equation_include_profile"]
        ),
        global_moment_residual_equation_include_current=bool(
            extra_coarse_controls["global_moment_residual_equation_include_current"]
        ),
        global_moment_residual_equation_include_tail=bool(
            extra_coarse_controls["global_moment_residual_equation_include_tail"]
        ),
        residual_galerkin_equation=bool(
            extra_coarse_controls["residual_galerkin_equation"]
        ),
        residual_galerkin_equation_max_stages=int(
            extra_coarse_controls["residual_galerkin_equation_max_stages"]
        ),
        residual_galerkin_equation_max_stage_rank=int(
            extra_coarse_controls["residual_galerkin_equation_max_stage_rank"]
        ),
        residual_galerkin_equation_max_rank=int(
            extra_coarse_controls["residual_galerkin_equation_max_rank"]
        ),
        residual_galerkin_equation_solver=str(
            extra_coarse_controls["residual_galerkin_equation_solver"]
        ),
        residual_galerkin_equation_include_global_residual=bool(
            extra_coarse_controls[
                "residual_galerkin_equation_include_global_residual"
            ]
        ),
        residual_galerkin_equation_include_block_residuals=bool(
            extra_coarse_controls["residual_galerkin_equation_include_block_residuals"]
        ),
        residual_galerkin_equation_include_operator_images=bool(
            extra_coarse_controls[
                "residual_galerkin_equation_include_operator_images"
            ]
        ),
        phase_space_residual_equation=bool(
            extra_coarse_controls["phase_space_residual_equation"]
        ),
        phase_space_residual_equation_max_rank=int(
            extra_coarse_controls["phase_space_residual_equation_max_rank"]
        ),
        phase_space_residual_equation_solver=str(
            extra_coarse_controls["phase_space_residual_equation_solver"]
        ),
        phase_space_residual_equation_boundary=float(
            extra_coarse_controls["phase_space_residual_equation_boundary"]
        ),
        phase_space_residual_equation_include_global=bool(
            extra_coarse_controls["phase_space_residual_equation_include_global"]
        ),
        phase_space_residual_equation_include_radial=bool(
            extra_coarse_controls["phase_space_residual_equation_include_radial"]
        ),
        phase_space_residual_equation_include_species=bool(
            extra_coarse_controls["phase_space_residual_equation_include_species"]
        ),
        residual_region_bounce_coarse=bool(
            extra_coarse_controls["residual_region_bounce_coarse"]
        ),
        residual_region_bounce_coarse_max_rank=int(
            extra_coarse_controls["residual_region_bounce_coarse_max_rank"]
        ),
        residual_region_bounce_coarse_solver=str(
            extra_coarse_controls["residual_region_bounce_coarse_solver"]
        ),
        residual_region_bounce_coarse_boundary=float(
            extra_coarse_controls["residual_region_bounce_coarse_boundary"]
        ),
        residual_region_bounce_coarse_min_energy=float(
            extra_coarse_controls["residual_region_bounce_coarse_min_energy"]
        ),
        residual_region_bounce_coarse_include_global=bool(
            extra_coarse_controls["residual_region_bounce_coarse_include_global"]
        ),
        residual_region_bounce_coarse_include_radial=bool(
            extra_coarse_controls["residual_region_bounce_coarse_include_radial"]
        ),
        residual_region_bounce_coarse_include_species=bool(
            extra_coarse_controls["residual_region_bounce_coarse_include_species"]
        ),
        residual_region_bounce_coarse_region_bands=str(
            extra_coarse_controls["residual_region_bounce_coarse_region_bands"]
        ),
        active_pattern_coarse=bool(extra_coarse_controls["active_pattern_coarse"]),
        active_pattern_coarse_max_rank=int(
            extra_coarse_controls["active_pattern_coarse_max_rank"]
        ),
        active_pattern_coarse_max_candidates=int(
            extra_coarse_controls["active_pattern_coarse_max_candidates"]
        ),
        active_pattern_coarse_solver=str(
            extra_coarse_controls["active_pattern_coarse_solver"]
        ),
        active_pattern_coarse_min_chunk_energy=float(
            extra_coarse_controls["active_pattern_coarse_min_chunk_energy"]
        ),
        active_pattern_coarse_include_global=bool(
            extra_coarse_controls["active_pattern_coarse_include_global"]
        ),
        block_schur_residual_equation=bool(
            residual_correction_controls["block_schur_residual_equation"]
        ),
        block_schur_residual_equation_max_rank=int(
            residual_correction_controls["block_schur_residual_equation_max_rank"]
        ),
        block_schur_residual_equation_include_global=bool(
            residual_correction_controls[
                "block_schur_residual_equation_include_global"
            ]
        ),
        block_schur_residual_equation_include_blocks=bool(
            residual_correction_controls[
                "block_schur_residual_equation_include_blocks"
            ]
        ),
        block_schur_residual_equation_include_aggregates=bool(
            residual_correction_controls[
                "block_schur_residual_equation_include_aggregates"
            ]
        ),
        coupled_residual_equation=bool(
            residual_correction_controls["coupled_residual_equation"]
        ),
        coupled_residual_equation_max_rank=int(
            residual_correction_controls["coupled_residual_equation_max_rank"]
        ),
        coupled_residual_equation_solver=str(
            residual_correction_controls["coupled_residual_equation_solver"]
        ),
        coupled_residual_equation_include_flat=bool(
            residual_correction_controls["coupled_residual_equation_include_flat"]
        ),
        coupled_residual_equation_install_on_reject=bool(
            residual_correction_controls[
                "coupled_residual_equation_install_on_reject"
            ]
        ),
        coupled_residual_equation_min_improvement=float(
            residual_correction_controls["coupled_residual_equation_min_improvement"]
        ),
        residual_snapshot_enrichment=bool(
            residual_correction_controls["residual_snapshot_enrichment"]
        ),
        residual_snapshot_max_rank=int(
            residual_correction_controls["residual_snapshot_max_rank"]
        ),
        residual_snapshot_include_primal=bool(
            residual_correction_controls["residual_snapshot_include_primal"]
        ),
        residual_snapshot_use_adjoint=bool(
            residual_correction_controls["residual_snapshot_use_adjoint"]
        ),
        residual_snapshot_include_global=bool(
            residual_correction_controls["residual_snapshot_include_global"]
        ),
        residual_snapshot_include_blocks=bool(
            residual_correction_controls["residual_snapshot_include_blocks"]
        ),
        residual_snapshot_include_aggregates=bool(
            residual_correction_controls["residual_snapshot_include_aggregates"]
        ),
        residual_snapshot_residual_equation=bool(
            residual_correction_controls["residual_snapshot_residual_equation"]
        ),
        residual_snapshot_residual_equation_max_rank=int(
            residual_correction_controls["residual_snapshot_residual_equation_max_rank"]
        ),
        residual_snapshot_residual_equation_solver=str(
            residual_correction_controls["residual_snapshot_residual_equation_solver"]
        ),
        residual_snapshot_residual_equation_include_global=bool(
            residual_correction_controls[
                "residual_snapshot_residual_equation_include_global"
            ]
        ),
        block_schur_residual_enrichment=bool(
            residual_correction_controls["block_schur_residual_enrichment"]
        ),
        block_schur_residual_max_rank=int(
            residual_correction_controls["block_schur_residual_max_rank"]
        ),
        block_schur_residual_include_global=bool(
            residual_correction_controls["block_schur_residual_include_global"]
        ),
        block_schur_residual_include_blocks=bool(
            residual_correction_controls["block_schur_residual_include_blocks"]
        ),
        block_schur_residual_include_aggregates=bool(
            residual_correction_controls["block_schur_residual_include_aggregates"]
        ),
    )
    residual_seed_required = (
        bool(enrichment_config.residual_enrichment)
        or bool(enrichment_config.recycle_enrichment)
        or bool(enrichment_config.operator_krylov_enrichment)
        or bool(multilevel_config.multilevel_residual_equation)
        or bool(extra_coarse_controls["global_moment_residual_equation"])
        or bool(extra_coarse_controls["residual_galerkin_equation"])
        or bool(extra_coarse_controls["phase_space_residual_equation"])
        or bool(extra_coarse_controls["residual_region_bounce_coarse"])
        or bool(extra_coarse_controls["active_pattern_coarse"])
        or bool(residual_correction_controls["block_schur_residual_equation"])
        or bool(residual_correction_controls["coupled_residual_equation"])
        or bool(residual_correction_controls["residual_snapshot_enrichment"])
        or bool(residual_correction_controls["residual_snapshot_residual_equation"])
        or bool(residual_correction_controls["block_schur_residual_enrichment"])
    )
    return RHS1QIDeviceSetupSummary(
        rank_budget=int(rank_budget_setup.rank_budget),
        max_rank=rank_budget_setup.max_rank,
        progress_messages=progress_messages,
        residual_seed_required=bool(residual_seed_required),
    )


def rhs1_qi_device_progress_messages(
    *,
    assembled_device_operator_available: bool,
    residual_enrichment: bool,
    residual_enrichment_depth: int,
    operator_action_enrichment: bool,
    operator_action_depth: int,
    operator_krylov_enrichment: bool,
    operator_krylov_depth: int,
    adjoint_krylov_enrichment: bool,
    adjoint_krylov_depth: int,
    adjoint_krylov_transpose_source: str,
    max_rank: int | None,
    multilevel_coarse: bool,
    multilevel_max_levels: int,
    multilevel_aggregate_factor: int,
    multilevel_max_pitch_degree: int,
    multilevel_current_moments: bool,
    multilevel_max_rank: int | None,
    multilevel_residual_equation: bool,
    multilevel_residual_equation_max_level_rank: int,
    multilevel_residual_equation_order: str,
    multilevel_residual_equation_solver: str,
    multilevel_residual_equation_include_global: bool,
    global_moment_residual_equation: bool,
    global_moment_residual_equation_max_rank: int,
    global_moment_residual_equation_solver: str,
    global_moment_residual_equation_include_profile: bool,
    global_moment_residual_equation_include_current: bool,
    global_moment_residual_equation_include_tail: bool,
    residual_galerkin_equation: bool,
    residual_galerkin_equation_max_stages: int,
    residual_galerkin_equation_max_stage_rank: int,
    residual_galerkin_equation_max_rank: int,
    residual_galerkin_equation_solver: str,
    residual_galerkin_equation_include_global_residual: bool,
    residual_galerkin_equation_include_block_residuals: bool,
    residual_galerkin_equation_include_operator_images: bool,
    phase_space_residual_equation: bool,
    phase_space_residual_equation_max_rank: int,
    phase_space_residual_equation_solver: str,
    phase_space_residual_equation_boundary: float,
    phase_space_residual_equation_include_global: bool,
    phase_space_residual_equation_include_radial: bool,
    phase_space_residual_equation_include_species: bool,
    residual_region_bounce_coarse: bool,
    residual_region_bounce_coarse_max_rank: int,
    residual_region_bounce_coarse_solver: str,
    residual_region_bounce_coarse_boundary: float,
    residual_region_bounce_coarse_min_energy: float,
    residual_region_bounce_coarse_include_global: bool,
    residual_region_bounce_coarse_include_radial: bool,
    residual_region_bounce_coarse_include_species: bool,
    residual_region_bounce_coarse_region_bands: str,
    active_pattern_coarse: bool,
    active_pattern_coarse_max_rank: int,
    active_pattern_coarse_max_candidates: int,
    active_pattern_coarse_solver: str,
    active_pattern_coarse_min_chunk_energy: float,
    active_pattern_coarse_include_global: bool,
    block_schur_residual_equation: bool,
    block_schur_residual_equation_max_rank: int,
    block_schur_residual_equation_include_global: bool,
    block_schur_residual_equation_include_blocks: bool,
    block_schur_residual_equation_include_aggregates: bool,
    coupled_residual_equation: bool,
    coupled_residual_equation_max_rank: int,
    coupled_residual_equation_solver: str,
    coupled_residual_equation_include_flat: bool,
    coupled_residual_equation_install_on_reject: bool,
    coupled_residual_equation_min_improvement: float,
    residual_snapshot_enrichment: bool,
    residual_snapshot_max_rank: int,
    residual_snapshot_include_primal: bool,
    residual_snapshot_use_adjoint: bool,
    residual_snapshot_include_global: bool,
    residual_snapshot_include_blocks: bool,
    residual_snapshot_include_aggregates: bool,
    residual_snapshot_residual_equation: bool,
    residual_snapshot_residual_equation_max_rank: int,
    residual_snapshot_residual_equation_solver: str,
    residual_snapshot_residual_equation_include_global: bool,
    block_schur_residual_enrichment: bool,
    block_schur_residual_max_rank: int,
    block_schur_residual_include_global: bool,
    block_schur_residual_include_blocks: bool,
    block_schur_residual_include_aggregates: bool,
) -> tuple[str, ...]:
    """Return progress messages for optional QI-device coarse/residual features."""

    prefix = "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
    messages: list[str] = []
    if not bool(assembled_device_operator_available):
        messages.append(
            prefix
            + "QI device preconditioner using matrix-free coarse-only operator-on-basis fallback"
        )
    if bool(residual_enrichment):
        messages.append(
            prefix
            + "QI device preconditioner residual enrichment "
            f"(depth={int(residual_enrichment_depth)} max_rank={max_rank})"
        )
    if bool(operator_action_enrichment):
        messages.append(
            prefix
            + "QI device preconditioner operator-action coarse enrichment "
            f"(depth={int(operator_action_depth)} max_rank={max_rank})"
        )
    if bool(operator_krylov_enrichment):
        messages.append(
            prefix
            + "QI device preconditioner operator-Krylov coarse enrichment "
            f"(depth={int(operator_krylov_depth)} max_rank={max_rank})"
        )
    if bool(adjoint_krylov_enrichment):
        messages.append(
            prefix
            + "QI device preconditioner adjoint-normal Krylov coarse enrichment "
            f"(depth={int(adjoint_krylov_depth)} transpose={adjoint_krylov_transpose_source} "
            f"max_rank={max_rank})"
        )
    if bool(multilevel_coarse):
        messages.append(
            prefix
            + "QI device preconditioner multilevel angular-radial coarse reuse "
            f"(levels={int(multilevel_max_levels)} "
            f"aggregate_factor={int(multilevel_aggregate_factor)} "
            f"pitch_degree={int(multilevel_max_pitch_degree)} "
            f"current_moments={int(bool(multilevel_current_moments))} "
            f"max_rank={multilevel_max_rank})"
        )
    if bool(multilevel_residual_equation):
        messages.append(
            prefix
            + "QI device preconditioner multilevel residual equation "
            f"(levels={int(multilevel_max_levels)} "
            f"stage_rank={int(multilevel_residual_equation_max_level_rank)} "
            f"order={multilevel_residual_equation_order} "
            f"solver={multilevel_residual_equation_solver} "
            f"include_global={int(bool(multilevel_residual_equation_include_global))})"
        )
    if bool(global_moment_residual_equation):
        messages.append(
            prefix
            + "QI device preconditioner global moment residual equation "
            f"(max_rank={int(global_moment_residual_equation_max_rank)} "
            f"solver={global_moment_residual_equation_solver} "
            f"profile={int(bool(global_moment_residual_equation_include_profile))} "
            f"current={int(bool(global_moment_residual_equation_include_current))} "
            f"tail={int(bool(global_moment_residual_equation_include_tail))})"
        )
    if bool(residual_galerkin_equation):
        messages.append(
            prefix
            + "QI device preconditioner residual Galerkin equation "
            f"(max_stages={int(residual_galerkin_equation_max_stages)} "
            f"stage_rank={int(residual_galerkin_equation_max_stage_rank)} "
            f"max_rank={int(residual_galerkin_equation_max_rank)} "
            f"solver={residual_galerkin_equation_solver} "
            f"global={int(bool(residual_galerkin_equation_include_global_residual))} "
            f"blocks={int(bool(residual_galerkin_equation_include_block_residuals))} "
            f"images={int(bool(residual_galerkin_equation_include_operator_images))})"
        )
    if bool(phase_space_residual_equation):
        messages.append(
            prefix
            + "QI device preconditioner phase-space residual equation "
            f"(max_rank={int(phase_space_residual_equation_max_rank)} "
            f"solver={phase_space_residual_equation_solver} "
            f"boundary={float(phase_space_residual_equation_boundary):.3e} "
            f"include_global={int(bool(phase_space_residual_equation_include_global))} "
            f"radial={int(bool(phase_space_residual_equation_include_radial))} "
            f"species={int(bool(phase_space_residual_equation_include_species))})"
        )
    if bool(residual_region_bounce_coarse):
        messages.append(
            prefix
            + "QI device preconditioner residual-region/bounce coarse "
            f"(max_rank={int(residual_region_bounce_coarse_max_rank)} "
            f"solver={residual_region_bounce_coarse_solver} "
            f"boundary={float(residual_region_bounce_coarse_boundary):.3e} "
            f"min_energy={float(residual_region_bounce_coarse_min_energy):.3e} "
            f"include_global={int(bool(residual_region_bounce_coarse_include_global))} "
            f"radial={int(bool(residual_region_bounce_coarse_include_radial))} "
            f"species={int(bool(residual_region_bounce_coarse_include_species))} "
            f"bands={residual_region_bounce_coarse_region_bands})"
        )
    if bool(active_pattern_coarse):
        messages.append(
            prefix
            + "QI device preconditioner active-pattern coarse "
            f"(max_rank={int(active_pattern_coarse_max_rank)} "
            f"max_candidates={int(active_pattern_coarse_max_candidates)} "
            f"solver={active_pattern_coarse_solver} "
            f"min_energy={float(active_pattern_coarse_min_chunk_energy):.3e} "
            f"include_global={int(bool(active_pattern_coarse_include_global))})"
        )
    if bool(block_schur_residual_equation):
        messages.append(
            prefix
            + "QI device preconditioner block-Schur residual equation "
            f"(max_rank={int(block_schur_residual_equation_max_rank)} "
            f"include_global={int(bool(block_schur_residual_equation_include_global))} "
            f"include_blocks={int(bool(block_schur_residual_equation_include_blocks))} "
            f"include_aggregates={int(bool(block_schur_residual_equation_include_aggregates))})"
        )
    if bool(coupled_residual_equation):
        messages.append(
            prefix
            + "QI device preconditioner coupled residual equation "
            f"(max_rank={int(coupled_residual_equation_max_rank)} "
            f"solver={coupled_residual_equation_solver} "
            f"include_flat={int(bool(coupled_residual_equation_include_flat))} "
            f"install_on_reject={int(bool(coupled_residual_equation_install_on_reject))} "
            f"min_improvement={float(coupled_residual_equation_min_improvement):.3e})"
        )
    if bool(residual_snapshot_enrichment):
        messages.append(
            prefix
            + "QI device preconditioner residual-snapshot coarse enrichment "
            f"(max_rank={int(residual_snapshot_max_rank)} "
            f"include_primal={int(bool(residual_snapshot_include_primal))} "
            f"use_adjoint={int(bool(residual_snapshot_use_adjoint))} "
            f"include_global={int(bool(residual_snapshot_include_global))} "
            f"include_blocks={int(bool(residual_snapshot_include_blocks))} "
            f"include_aggregates={int(bool(residual_snapshot_include_aggregates))})"
        )
    if bool(residual_snapshot_residual_equation):
        messages.append(
            prefix
            + "QI device preconditioner residual-snapshot residual equation "
            f"(max_rank={int(residual_snapshot_residual_equation_max_rank)} "
            f"solver={residual_snapshot_residual_equation_solver} "
            f"include_global={int(bool(residual_snapshot_residual_equation_include_global))} "
            f"include_primal={int(bool(residual_snapshot_include_primal))} "
            f"use_adjoint={int(bool(residual_snapshot_use_adjoint))} "
            f"include_blocks={int(bool(residual_snapshot_include_blocks))} "
            f"include_aggregates={int(bool(residual_snapshot_include_aggregates))})"
        )
    if bool(block_schur_residual_enrichment):
        messages.append(
            prefix
            + "QI device preconditioner block-Schur residual coarse enrichment "
            f"(max_rank={int(block_schur_residual_max_rank)} "
            f"include_global={int(bool(block_schur_residual_include_global))} "
            f"include_blocks={int(bool(block_schur_residual_include_blocks))} "
            f"include_aggregates={int(bool(block_schur_residual_include_aggregates))})"
        )
    return tuple(messages)


def rhs1_qi_device_probe_uses_minres_step() -> bool:
    """Return whether QI-device seed probes should line-search each correction."""

    raw = (
        os.environ.get(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_STEP_POLICY",
            "fixed",
        )
        .strip()
        .lower()
        .replace("-", "_")
    )
    return raw in {"minres", "residual_minimizing", "line_search", "linesearch"}


def rhs1_xblock_fallback_initial_guess(
    *,
    candidate: np.ndarray,
    original_x0: jnp.ndarray | None,
    rhs_shape: tuple[int, ...],
    candidate_residual_norm: float,
    rhs_norm: float,
    precondition_side: str,
) -> tuple[jnp.ndarray | None, bool, bool]:
    """Return a safe initial guess for x-block Krylov rescues.

    Non-GMRES host Krylov methods can produce a useful physical-space state
    before failing a strict residual gate. Reusing that state is safe only for
    left/no preconditioning when it improves over the zero-state RHS norm.
    Right-preconditioned iteration states are rejected because SciPy stores
    them in preconditioned coordinates.
    """

    candidate_improved_rhs = bool(
        np.isfinite(float(candidate_residual_norm))
        and np.isfinite(float(rhs_norm))
        and float(candidate_residual_norm) < float(rhs_norm)
    )
    if (not candidate_improved_rhs) or str(
        precondition_side
    ).strip().lower() == "right":
        return original_x0, False, candidate_improved_rhs
    try:
        candidate_x0 = jnp.asarray(candidate, dtype=jnp.float64)
        if candidate_x0.shape == tuple(rhs_shape) and bool(
            jnp.all(jnp.isfinite(candidate_x0))
        ):
            return candidate_x0, True, candidate_improved_rhs
    except Exception:
        pass
    return original_x0, False, candidate_improved_rhs


# From sfincs_jax.rhs1_acceptance_policy
def rhs1_pas_fast_accept(
    *,
    op: Any,
    active_size: int,
    residual_norm: float,
    target: float,
    use_implicit: bool,
    backend: str,
) -> bool:
    """Return whether a large explicit CPU PAS solve may be accepted quickly."""
    env = _env_token("SFINCS_JAX_PAS_FAST_ACCEPT")
    if env in _FALSE_VALUES:
        return False
    if bool(use_implicit):
        return False
    if str(backend).strip().lower() != "cpu":
        return False
    if int(op.rhs_mode) != 1 or bool(op.include_phi1):
        return False
    if op.fblock.pas is None:
        return False
    return _pas_fast_accept_metric(
        active_size=int(active_size),
        residual_norm=float(residual_norm),
        target=float(target),
        min_size=_env_int("SFINCS_JAX_PAS_FAST_ACCEPT_MIN", 20000),
        ratio=_env_float("SFINCS_JAX_PAS_FAST_ACCEPT_RATIO", 100.0),
        abs_floor=_env_float("SFINCS_JAX_PAS_FAST_ACCEPT_ABS", 1e-07),
    )


def rhs1_host_factor_probe_ok(*, factor: object | None, block_size: int) -> bool:
    """Return whether a host factor solve passes a bounded unit-vector probe."""
    if factor is None or int(block_size) <= 0:
        return False
    probe_max = max(
        _env_float("SFINCS_JAX_RHSMODE1_XBLOCK_FACTOR_PROBE_MAX", 100000000.0), 1.0
    )
    probe = np.ones((int(block_size),), dtype=np.float64)
    try:
        solved = np.asarray(factor.solve(probe), dtype=np.float64).reshape((-1,))
    except Exception:
        return False
    if solved.shape != probe.shape or not np.all(np.isfinite(solved)):
        return False
    ratio = float(np.linalg.norm(solved)) / max(float(np.linalg.norm(probe)), 1e-300)
    return np.isfinite(ratio) and ratio <= probe_max


# From sfincs_jax.rhs1_constraint0_policy
@dataclass(frozen=True)
class RHS1Constraint0PETScCompatConfig:
    """Host sparse-ILU controls for the constraint-scheme-0 PETSc lane."""

    drop_tol: float
    fill: float
    diag_pivot: float
    restart: int
    maxiter: int


def _has_constraint0_fp_rhs1(op: Any) -> bool:
    if int(op.rhs_mode) != 1 or bool(op.include_phi1):
        return False
    if int(op.constraint_scheme) != 0:
        return False
    return op.fblock.fp is not None


def _sparse_method_allowed(
    *,
    solve_method_kind: str,
    sparse_precond_mode: str,
    active_size: int,
    sparse_max_size: int,
) -> bool:
    if str(solve_method_kind).strip().lower() in {"dense", "dense_ksp"}:
        return False
    if str(sparse_precond_mode).strip().lower() == "off":
        return False
    return int(active_size) <= int(sparse_max_size)


def rhs1_constraint0_sparse_first(
    *,
    op: Any,
    solve_method_kind: str,
    sparse_precond_mode: str,
    active_size: int,
    sparse_max_size: int,
    backend: str,
) -> bool:
    """Return whether constraint-scheme-0 RHSMode=1 should try sparse first.

    The default is accelerator-only because this lane was introduced to avoid
    small/medium GPU dense-LU regressions while retaining CPU dense fallback
    behavior unless the user explicitly opts into sparse-first CPU behavior.
    """
    env = _env_token("SFINCS_JAX_RHSMODE1_CS0_SPARSE_FIRST")
    if env in _FALSE_VALUES:
        return False
    if env not in _TRUE_VALUES and str(backend).strip().lower() == "cpu":
        return False
    if not _has_constraint0_fp_rhs1(op):
        return False
    return _sparse_method_allowed(
        solve_method_kind=solve_method_kind,
        sparse_precond_mode=sparse_precond_mode,
        active_size=active_size,
        sparse_max_size=sparse_max_size,
    )


def rhs1_constraint0_petsc_compat(
    *,
    op: Any,
    solve_method_kind: str,
    sparse_precond_mode: str,
    active_size: int,
    sparse_max_size: int,
) -> bool:
    """Return whether explicit PETSc-compatible sparse behavior is requested."""
    env = _env_token("SFINCS_JAX_RHSMODE1_CS0_PETSC_COMPAT")
    if env in {"", *_FALSE_VALUES}:
        return False
    if not _has_constraint0_fp_rhs1(op):
        return False
    if not _sparse_method_allowed(
        solve_method_kind=solve_method_kind,
        sparse_precond_mode=sparse_precond_mode,
        active_size=active_size,
        sparse_max_size=sparse_max_size,
    ):
        return False
    return env in _TRUE_VALUES


def rhs1_constraint0_dense_fallback_allowed(op: Any) -> bool:
    """Return whether dense fallback is allowed for constraint-scheme-0 solves."""
    if int(op.constraint_scheme) != 0:
        return True
    env = _env_token("SFINCS_JAX_RHSMODE1_CS0_DENSE_FALLBACK")
    return env in _TRUE_VALUES


def rhs1_constraint0_petsc_compat_config_from_env(
    *,
    restart: int,
    maxiter: int | None,
) -> RHS1Constraint0PETScCompatConfig:
    """Parse constraint-scheme-0 PETSc-compatible sparse solve controls."""

    default_restart = max(int(restart), 2000)
    default_maxiter = max(int(maxiter or 1), 1)
    return RHS1Constraint0PETScCompatConfig(
        drop_tol=float(
            _env_float("SFINCS_JAX_RHSMODE1_CS0_PETSC_COMPAT_DROP_TOL", 1.0e-4)
        ),
        fill=float(_env_float("SFINCS_JAX_RHSMODE1_CS0_PETSC_COMPAT_FILL", 10.0)),
        diag_pivot=float(
            _env_float("SFINCS_JAX_RHSMODE1_CS0_PETSC_COMPAT_DIAG_PIVOT", 0.0)
        ),
        restart=int(
            _env_int("SFINCS_JAX_RHSMODE1_CS0_PETSC_COMPAT_RESTART", default_restart)
        ),
        maxiter=int(
            _env_int("SFINCS_JAX_RHSMODE1_CS0_PETSC_COMPAT_MAXITER", default_maxiter)
        ),
    )


def rhs1_constraint0_petsc_compat_regularization(*, max_abs: float) -> float:
    """Parse/floor the PETSc-compatible diagonal regularization."""

    default_reg = 1.0e-12 * float(max_abs)
    regularization = _env_float("SFINCS_JAX_RHSMODE1_CS0_PETSC_COMPAT_REG", default_reg)
    return max(0.0, float(regularization))


# From sfincs_jax.rhs1_post_xblock_policy
def _is_explicit_cpu_rhs1_fp_only(*, op: Any, use_implicit: bool, backend: str) -> bool:
    if bool(use_implicit):
        return False
    if str(backend).strip().lower() != "cpu":
        return False
    if int(op.rhs_mode) != 1 or bool(op.include_phi1):
        return False
    return op.fblock.fp is not None and getattr(op.fblock, "pas", None) is None


def rhs1_fast_post_xblock_polish_allowed(
    *,
    op: Any,
    active_size: int,
    residual_norm: float,
    target: float,
    used_large_cpu_xblock_shortcut: bool,
    use_implicit: bool,
    backend: str,
) -> bool:
    """Return whether a bad large-CPU x-block seed should receive fast polish."""
    env = _env_token("SFINCS_JAX_RHSMODE1_FAST_POST_XBLOCK_POLISH")
    if env in _FALSE_VALUES:
        return False
    if not bool(used_large_cpu_xblock_shortcut):
        return False
    if not _is_explicit_cpu_rhs1_fp_only(
        op=op, use_implicit=use_implicit, backend=backend
    ):
        return False
    polish_min = _env_int("SFINCS_JAX_RHSMODE1_FAST_POST_XBLOCK_POLISH_MIN", 12000)
    if int(active_size) < max(1, int(polish_min)):
        return False
    polish_max = _env_int("SFINCS_JAX_RHSMODE1_FAST_POST_XBLOCK_POLISH_MAX", 200000)
    if int(polish_max) > 0 and int(active_size) > int(polish_max):
        return False
    polish_ratio = _env_float(
        "SFINCS_JAX_RHSMODE1_FAST_POST_XBLOCK_POLISH_RATIO", 1000.0
    )
    polish_abs = _env_float("SFINCS_JAX_RHSMODE1_FAST_POST_XBLOCK_POLISH_ABS", 1e-06)
    threshold = max(float(polish_abs), float(target) * max(1.0, float(polish_ratio)))
    return float(residual_norm) > float(threshold)


@dataclass(frozen=True)
class RHS1FastPostXBlockPolishControls:
    """Bounded Krylov controls for the fast post-xblock polish pass."""

    restart: int
    maxiter: int
    tol: float


def rhs1_fast_post_xblock_polish_controls_from_env(
    *,
    restart: int,
    maxiter: int | None,
    tol: float,
) -> RHS1FastPostXBlockPolishControls:
    """Parse fast post-xblock polish controls with legacy bounds."""

    restart_use = _env_int(
        "SFINCS_JAX_RHSMODE1_FAST_POST_XBLOCK_POLISH_RESTART",
        min(int(restart), 40),
    )
    maxiter_use = _env_int(
        "SFINCS_JAX_RHSMODE1_FAST_POST_XBLOCK_POLISH_MAXITER",
        min(max(40, int(maxiter or 80)), 80),
    )
    tol_use = _env_float(
        "SFINCS_JAX_RHSMODE1_FAST_POST_XBLOCK_POLISH_TOL",
        min(float(tol), 1.0e-10),
    )
    return RHS1FastPostXBlockPolishControls(
        restart=max(5, int(restart_use)),
        maxiter=max(5, int(maxiter_use)),
        tol=float(tol_use),
    )


def rhs1_fp_targeted_polish_allowed(
    *,
    op: Any,
    active_size: int,
    residual_norm: float,
    target: float,
    rhs1_precond_kind: str,
    use_implicit: bool,
    backend: str,
) -> bool:
    """Return whether a medium/large explicit FP xmg solve should be polished."""
    env = _env_token("SFINCS_JAX_RHSMODE1_FP_TARGETED_POLISH")
    if env in _FALSE_VALUES:
        return False
    if not _is_explicit_cpu_rhs1_fp_only(
        op=op, use_implicit=use_implicit, backend=backend
    ):
        return False
    if str(rhs1_precond_kind) != "xmg":
        return False
    polish_min = _env_int("SFINCS_JAX_RHSMODE1_FP_TARGETED_POLISH_MIN", 12000)
    if int(active_size) < max(1, int(polish_min)):
        return False
    polish_ratio = _env_float("SFINCS_JAX_RHSMODE1_FP_TARGETED_POLISH_RATIO", 10.0)
    polish_abs = _env_float("SFINCS_JAX_RHSMODE1_FP_TARGETED_POLISH_ABS", 1e-09)
    threshold = max(float(polish_abs), float(target) * max(1.0, float(polish_ratio)))
    return float(residual_norm) > float(threshold)


@dataclass(frozen=True)
class RHS1FPResidualPolishControls:
    """Controls for the cheap damped FP residual-polish pass."""

    min_size: int
    steps: int
    hybrid: bool
    omega: float
    backtrack: int


@dataclass(frozen=True)
class RHS1FPLowLPolishControls:
    """Controls for the FP low-L angular/x polish pass."""

    lmax_default: int
    block_max: int
    restart: int
    maxiter: int


def rhs1_fp_residual_polish_controls_from_env() -> RHS1FPResidualPolishControls:
    """Parse damped FP residual-polish controls with bounded defaults."""

    hybrid_env = _env_token("SFINCS_JAX_RHSMODE1_FP_POLISH_HYBRID")
    omega = _env_float("SFINCS_JAX_RHSMODE1_FP_POLISH_OMEGA", 1.0)
    backtrack = _env_int("SFINCS_JAX_RHSMODE1_FP_POLISH_BACKTRACK", 3)
    return RHS1FPResidualPolishControls(
        min_size=max(1, _env_int("SFINCS_JAX_RHSMODE1_FP_POLISH_MIN", 80000)),
        steps=max(0, min(_env_int("SFINCS_JAX_RHSMODE1_FP_POLISH_STEPS", 2), 6)),
        hybrid=hybrid_env not in _FALSE_VALUES,
        omega=max(1.0e-3, min(float(omega), 1.5)),
        backtrack=max(0, min(int(backtrack), 6)),
    )


def rhs1_fp_low_l_polish_controls_from_env(
    *,
    has_fp: bool,
    has_pas: bool,
    n_theta: int,
    n_zeta: int,
) -> RHS1FPLowLPolishControls:
    """Parse FP low-L polish controls and the small-angular-grid default bump."""

    lmax_env = _env_token("SFINCS_JAX_RHSMODE1_FP_POLISH_LMAX")
    lmax_default = _env_int("SFINCS_JAX_RHSMODE1_FP_POLISH_LMAX", 2)
    if (
        not lmax_env
        and bool(has_fp)
        and not bool(has_pas)
        and int(n_theta) * int(n_zeta) <= 256
    ):
        lmax_default = max(int(lmax_default), 6)
    return RHS1FPLowLPolishControls(
        lmax_default=int(lmax_default),
        block_max=_env_int("SFINCS_JAX_RHSMODE1_FP_POLISH_LMAX_BLOCK_MAX", 1500),
        restart=_env_int("SFINCS_JAX_RHSMODE1_FP_POLISH_LMAX_RESTART", 80),
        maxiter=_env_int("SFINCS_JAX_RHSMODE1_FP_POLISH_LMAX_MAXITER", 120),
    )


def rhs1_skip_global_sparse_after_xblock_allowed(
    *,
    op: Any,
    active_size: int,
    residual_norm: float,
    target: float,
    used_large_cpu_xblock_shortcut: bool,
    used_explicit_fp_xblock_seed: bool,
    use_implicit: bool,
    backend: str,
) -> bool:
    """Return whether a good x-block seed may skip global sparse rescue."""
    env = _env_token("SFINCS_JAX_RHSMODE1_SKIP_GLOBAL_SPARSE_AFTER_XBLOCK")
    if env in _FALSE_VALUES:
        return False
    if not bool(used_large_cpu_xblock_shortcut) or not bool(
        used_explicit_fp_xblock_seed
    ):
        return False
    if not _is_explicit_cpu_rhs1_fp_only(
        op=op, use_implicit=use_implicit, backend=backend
    ):
        return False
    skip_min = _env_int(
        "SFINCS_JAX_RHSMODE1_SKIP_GLOBAL_SPARSE_AFTER_XBLOCK_MIN", 12000
    )
    if int(active_size) < max(1, int(skip_min)):
        return False
    skip_ratio = _env_float(
        "SFINCS_JAX_RHSMODE1_SKIP_GLOBAL_SPARSE_AFTER_XBLOCK_RATIO", 50000.0
    )
    skip_abs = _env_float(
        "SFINCS_JAX_RHSMODE1_SKIP_GLOBAL_SPARSE_AFTER_XBLOCK_ABS", 0.0005
    )
    threshold = max(float(skip_abs), float(target) * max(1.0, float(skip_ratio)))
    return float(residual_norm) <= float(threshold)


def rhs1_scipy_rescue_abs_floor_after_xblock(
    *,
    op: Any,
    active_size: int,
    used_large_cpu_xblock_shortcut: bool,
    used_explicit_fp_xblock_seed: bool,
    use_implicit: bool,
    backend: str,
) -> float:
    """Return an absolute residual floor below which CPU SciPy rescue is skipped.

    Large explicit FP runs can reach a physically tight residual after the
    x-block seed/refinement while still missing an over-tight relative target
    caused by a small RHS norm.  In that case a full SciPy rescue tends to chase
    roundoff for minutes.  The floor is intentionally limited to the same
    large-CPU, explicit-FP, post-x-block path and remains user-overridable.
    """
    env = _env_token("SFINCS_JAX_RHSMODE1_SCIPY_GMRES_RESCUE_ABS")
    if env:
        try:
            return max(0.0, float(env))
        except ValueError:
            return 0.0
    if not bool(used_large_cpu_xblock_shortcut) or not bool(
        used_explicit_fp_xblock_seed
    ):
        return 0.0
    if not _is_explicit_cpu_rhs1_fp_only(
        op=op, use_implicit=use_implicit, backend=backend
    ):
        return 0.0
    floor_min = _env_int("SFINCS_JAX_RHSMODE1_SCIPY_GMRES_RESCUE_ABS_MIN", 12000)
    if int(active_size) < max(1, int(floor_min)):
        return 0.0
    return 1e-09


def rhs1_scipy_rescue_active_size_allowed(
    *,
    op: Any,
    active_size: int,
    used_large_cpu_xblock_shortcut: bool,
    used_explicit_fp_xblock_seed: bool,
    use_implicit: bool,
    backend: str,
) -> bool:
    """Return whether CPU SciPy rescue may run for this active-system size.

    Production-resolution explicit FP VMEC systems can reach the SciPy rescue
    branch with a very poor seed and then spend minutes in host Krylov without
    producing output.  Keep that rescue for moderate systems and for successful
    x-block-seed follow-up, but make the no-seed large-CPU shortcut fail fast by
    default.  A non-positive max-active override restores the historical
    unbounded behavior.
    """
    if not bool(used_large_cpu_xblock_shortcut):
        return True
    if bool(used_explicit_fp_xblock_seed):
        return True
    if not _is_explicit_cpu_rhs1_fp_only(
        op=op, use_implicit=use_implicit, backend=backend
    ):
        return True
    max_active = _env_int("SFINCS_JAX_RHSMODE1_SCIPY_GMRES_RESCUE_MAX_ACTIVE", 250000)
    if int(max_active) <= 0:
        return True
    return int(active_size) <= int(max_active)


def rhs1_fp_xblock_global_correction_allowed(
    *,
    op: Any,
    active_size: int,
    residual_norm: float,
    target: float,
    used_large_cpu_xblock_shortcut: bool,
    used_explicit_fp_xblock_seed: bool,
    sparse_xblock_candidate_accepted: bool,
    use_implicit: bool,
    backend: str,
) -> bool:
    """Return whether a bounded residual-equation correction may follow x-block.

    This is an opt-in diagnostic path for production-resolution explicit FP
    cases. It reuses the accepted x-block seed and an existing matrix-free
    preconditioner, so it avoids the unbounded host SciPy rescue and avoids
    factorizing the high-x local blocks that were unstable in VMEC finite-beta
    production probes.
    """
    env = _env_token("SFINCS_JAX_RHSMODE1_FP_XBLOCK_GLOBAL_CORRECTION")
    if env not in _TRUE_VALUES:
        return False
    if not bool(used_large_cpu_xblock_shortcut):
        return False
    if not bool(used_explicit_fp_xblock_seed) or not bool(
        sparse_xblock_candidate_accepted
    ):
        return False
    if not _is_explicit_cpu_rhs1_fp_only(
        op=op, use_implicit=use_implicit, backend=backend
    ):
        return False
    if float(residual_norm) <= float(target):
        return False
    active_min = _env_int("SFINCS_JAX_RHSMODE1_FP_XBLOCK_GLOBAL_CORRECTION_MIN", 12000)
    if int(active_size) < max(1, int(active_min)):
        return False
    active_max = _env_int("SFINCS_JAX_RHSMODE1_FP_XBLOCK_GLOBAL_CORRECTION_MAX", 600000)
    if int(active_max) > 0 and int(active_size) > int(active_max):
        return False
    return True


# From sfincs_jax.rhs1_sparse_exact_policy
def rhs1_sparse_exact_lu_requested(
    *,
    op: Any,
    solve_method_kind: str,
    active_size: int,
    sparse_max_size: int,
    full_precond_requested: bool = False,
    preconditioner_x: int,
    use_dkes: bool,
    backend: str,
) -> bool:
    """Return whether the RHSMode=1 sparse exact-LU lane should be attempted.

    ``sparse_max_size`` is accepted to keep the policy signature aligned with the
    driver wrapper.  The exact-LU lane has its own environment-controlled cap
    because it can intentionally exceed the ILU/sparse-preconditioner size cap on
    accelerator DKES or full-x cases.
    """
    env = _env_token("SFINCS_JAX_RHSMODE1_SPARSE_EXACT_LU")
    if env in _FALSE_VALUES:
        return False
    if int(op.rhs_mode) != 1 or bool(op.include_phi1):
        return False
    has_fp = op.fblock.fp is not None
    has_pas = getattr(op.fblock, "pas", None) is not None
    allow_pas_full = bool(has_pas) and (
        bool(full_precond_requested) or env in _TRUE_VALUES
    )
    if not has_fp and (not allow_pas_full):
        return False
    if str(solve_method_kind).strip().lower() in {"dense", "dense_ksp"}:
        return False
    backend_name = str(backend).strip().lower()
    exact_default = 6000 if backend_name == "cpu" else 12000
    exact_max = max(
        0, _env_int("SFINCS_JAX_RHSMODE1_SPARSE_EXACT_LU_MAX", exact_default)
    )
    if int(active_size) > int(exact_max):
        return False
    if env in _TRUE_VALUES:
        return True
    accel_small_max = _env_int(
        "SFINCS_JAX_RHSMODE1_SPARSE_EXACT_LU_ACCEL_SMALL_MAX", 4000
    )
    accel_small_case = backend_name != "cpu" and int(active_size) <= max(
        0, int(accel_small_max)
    )
    return int(preconditioner_x) == 0 or (
        backend_name != "cpu" and (bool(use_dkes) or bool(accel_small_case))
    )


def rhs1_prefer_sparse_over_dense_shortcut(
    *, op: Any, active_size: int, sparse_max_size: int, use_implicit: bool
) -> bool:
    """Return whether a moderate explicit FP solve should keep the sparse path."""
    env = _env_token("SFINCS_JAX_RHSMODE1_SPARSE_PREFER_OVER_DENSE_SHORTCUT")
    if env in _FALSE_VALUES:
        return False
    if int(op.rhs_mode) != 1 or bool(op.include_phi1):
        return False
    if op.fblock.fp is None or bool(use_implicit):
        return False
    if int(active_size) > int(sparse_max_size):
        return False
    min_size = max(
        1, _env_int("SFINCS_JAX_RHSMODE1_SPARSE_PREFER_OVER_DENSE_SHORTCUT_MIN", 2000)
    )
    return int(active_size) >= int(min_size)


def rhs1_sparse_prefer_skips_stage2(
    *, sparse_prefer_over_dense_shortcut: bool, sparse_precond_mode: str
) -> bool:
    """Return whether sparse-prefer routing should skip the stage-2 fallback."""
    env = _env_token("SFINCS_JAX_RHSMODE1_SPARSE_SKIP_STAGE2")
    if env in _FALSE_VALUES:
        return False
    return (
        bool(sparse_prefer_over_dense_shortcut)
        and str(sparse_precond_mode).strip().lower() != "off"
    )


# From sfincs_jax.rhs1_sparse_rescue_policy
@dataclass(frozen=True)
class RHS1SparseRescueOrdering:
    """Resolved sparse-rescue ordering state for one solve branch."""

    enabled: bool
    kind_use: str
    xblock_rescue_active: bool = False
    sxblock_rescue_active: bool = False
    prefer_sparse_exact_over_dense_shortcut: bool = False
    reason_dense_shortcut_skip: bool = False
    reason_size_disabled: bool = False
    reason_size_large_cpu: bool = False
    reason_size_exact_direct: bool = False
    reason_size_targeted: bool = False
    reason_sparse_jax_mem_disabled: bool = False
    reason_large_cpu_exact_skips_targeted: bool = False
    reason_pas_fast_accept: bool = False
    reason_gpu_sparse_skip: bool = False


@dataclass(frozen=True)
class RHS1SparseRescuePolicySetup:
    """Complete sparse-rescue policy setup shared by full and active systems."""

    enabled: bool
    kind_use: str
    ordering: RHS1SparseRescueOrdering
    sparse_jax_est_mb: float | None = None
    sparse_jax_memory_disabled_message: str | None = None


@dataclass(frozen=True)
class RHS1SparseJAXConfig:
    """Environment-controlled sparse-JAX retry controls for RHSMode=1."""

    max_mb: float
    sweeps: int
    omega: float
    reg: float


@dataclass(frozen=True)
class RHS1SparsePreconditionerConfig:
    """Environment-controlled sparse preconditioner policy for RHSMode=1."""

    precond_mode: str
    precond_kind: str
    allow_nondiff: bool
    use_matvec: bool
    operator_mode: str
    max_size: int
    pas_sparse_min: int
    drop_tol: float
    drop_rel: float
    ilu_drop_tol: float
    ilu_fill: float
    ilu_dense_max: int
    dense_cache_max: int


@dataclass(frozen=True)
class RHS1SparseOperatorAdmission:
    """Admission result for replacing a reduced matvec with a sparse operator."""

    use_sparse_operator: bool
    messages: tuple[tuple[int, str], ...] = ()


def rhs1_sparse_enabled_initial(
    *,
    sparse_precond_mode: str,
    has_fp: bool,
    has_pas: bool,
    residual_norm: float,
    target: float,
    rhs_mode: int,
    include_phi1: bool,
) -> bool:
    """Resolve the initial sparse-rescue enable bit before ordering/skip rules."""
    enabled = False
    if sparse_precond_mode == "on":
        enabled = True
    elif sparse_precond_mode == "auto":
        enabled = bool(has_fp) or (
            bool(has_pas) and float(residual_norm) > float(target)
        )
    if enabled:
        enabled = int(rhs_mode) == 1 and (not bool(include_phi1))
    return bool(enabled)


def rhs1_sparse_preconditioner_config_from_env(
    *,
    has_pas: bool,
    use_dkes: bool,
    active_size: int,
    backend: str,
) -> RHS1SparsePreconditionerConfig:
    """Parse RHSMode=1 sparse-preconditioner controls with legacy defaults."""

    sparse_precond_env = _env_token("SFINCS_JAX_RHSMODE1_SPARSE_PRECOND")
    if sparse_precond_env in {"jax", "jax_native", "native"}:
        sparse_precond_mode = "on"
        sparse_precond_kind = "jax"
    elif sparse_precond_env in {"scipy", "ilu", "spilu"}:
        sparse_precond_mode = "on"
        sparse_precond_kind = "scipy"
    elif sparse_precond_env in _TRUE_VALUES:
        sparse_precond_mode = "on"
        sparse_precond_kind = "auto"
    elif sparse_precond_env in _FALSE_VALUES:
        sparse_precond_mode = "off"
        sparse_precond_kind = "auto"
    else:
        sparse_precond_mode = "auto"
        sparse_precond_kind = "auto"

    sparse_allow_nondiff = (
        _env_token("SFINCS_JAX_RHSMODE1_SPARSE_ALLOW_NONDIFF") in _TRUE_VALUES
    )
    sparse_matvec_env = _env_token("SFINCS_JAX_RHSMODE1_SPARSE_MATVEC")
    if sparse_matvec_env in _TRUE_VALUES:
        sparse_use_matvec = True
    elif sparse_matvec_env in _FALSE_VALUES:
        sparse_use_matvec = False
    else:
        sparse_use_matvec = False

    sparse_operator_env = _env_token("SFINCS_JAX_RHSMODE1_SPARSE_OPERATOR")
    if sparse_operator_env in _TRUE_VALUES:
        sparse_operator_mode = "on"
    elif sparse_operator_env in _FALSE_VALUES:
        sparse_operator_mode = "off"
    else:
        sparse_operator_mode = "auto"

    default_sparse_max = 60000 if bool(has_pas) and bool(use_dkes) else 6000
    sparse_max_size = _env_int("SFINCS_JAX_RHSMODE1_SPARSE_MAX", default_sparse_max)
    pas_sparse_min = _env_int("SFINCS_JAX_RHSMODE1_PAS_SPARSE_ILU_MIN", 2000)
    if bool(has_pas) and int(active_size) < max(0, int(pas_sparse_min)):
        sparse_precond_mode = "off"

    default_sparse_ilu_dense_max = 3000 if str(backend).lower() != "cpu" else 2500
    return RHS1SparsePreconditionerConfig(
        precond_mode=str(sparse_precond_mode),
        precond_kind=str(sparse_precond_kind),
        allow_nondiff=bool(sparse_allow_nondiff),
        use_matvec=bool(sparse_use_matvec),
        operator_mode=str(sparse_operator_mode),
        max_size=int(sparse_max_size),
        pas_sparse_min=int(pas_sparse_min),
        drop_tol=float(_env_float("SFINCS_JAX_RHSMODE1_SPARSE_DROP_TOL", 0.0)),
        drop_rel=float(_env_float("SFINCS_JAX_RHSMODE1_SPARSE_DROP_REL", 1.0e-8)),
        ilu_drop_tol=float(
            _env_float("SFINCS_JAX_RHSMODE1_SPARSE_ILU_DROP_TOL", 1.0e-4)
        ),
        ilu_fill=float(_env_float("SFINCS_JAX_RHSMODE1_SPARSE_ILU_FILL_FACTOR", 10.0)),
        ilu_dense_max=int(
            _env_int(
                "SFINCS_JAX_RHSMODE1_SPARSE_ILU_DENSE_MAX",
                default_sparse_ilu_dense_max,
            )
        ),
        dense_cache_max=int(_env_int("SFINCS_JAX_RHSMODE1_SPARSE_DENSE_CACHE_MAX", 3000)),
    )


def rhs1_sparse_operator_admission(
    *,
    operator_mode: str,
    use_matvec: bool,
    has_fp: bool,
    rhs_mode: int,
    include_phi1: bool,
    use_implicit: bool,
    allow_nondiff: bool,
    active_size: int,
    sparse_max_size: int,
) -> RHS1SparseOperatorAdmission:
    """Decide whether to materialize the reduced sparse operator matvec."""

    use_sparse_operator = False
    if str(operator_mode) == "on":
        use_sparse_operator = True
    elif str(operator_mode) == "auto":
        use_sparse_operator = bool(use_matvec) and bool(has_fp)
    if use_sparse_operator:
        use_sparse_operator = int(rhs_mode) == 1 and (not bool(include_phi1))

    messages: list[tuple[int, str]] = []
    if use_sparse_operator:
        if bool(use_implicit) and not bool(allow_nondiff):
            use_sparse_operator = False
            messages.append(
                (
                    1,
                    "sparse_operator: disabled for implicit solves "
                    "(set SFINCS_JAX_RHSMODE1_SPARSE_ALLOW_NONDIFF=1 to override)",
                )
            )
        elif int(active_size) > int(sparse_max_size):
            use_sparse_operator = False
            messages.append(
                (
                    1,
                    f"sparse_operator: disabled (size={int(active_size)} > max={int(sparse_max_size)})",
                )
            )

    return RHS1SparseOperatorAdmission(
        use_sparse_operator=bool(use_sparse_operator),
        messages=tuple(messages),
    )


def rhs1_sparse_jax_config_from_env() -> RHS1SparseJAXConfig:
    """Parse sparse-JAX retry controls with bounded, stable defaults."""

    max_mb = _env_float("SFINCS_JAX_RHSMODE1_SPARSE_JAX_MAX_MB", 128.0)
    sweeps = max(1, _env_int("SFINCS_JAX_RHSMODE1_SPARSE_JAX_SWEEPS", 2))
    omega = _env_float("SFINCS_JAX_RHSMODE1_SPARSE_JAX_OMEGA", 0.8)
    reg = _env_float("SFINCS_JAX_RHSMODE1_SPARSE_JAX_REG", 1.0e-10)
    return RHS1SparseJAXConfig(
        max_mb=float(max_mb),
        sweeps=int(sweeps),
        omega=float(omega),
        reg=float(reg),
    )


def rhs1_sparse_kind_use(*, sparse_precond_kind: str) -> str:
    """Resolve the concrete sparse backend kind used for rescue."""
    return "scipy" if str(sparse_precond_kind) == "auto" else str(sparse_precond_kind)


def rhs1_resolved_sparse_rescue_ordering(
    *,
    sparse_enabled: bool,
    sparse_kind_use: str,
    dense_shortcut: bool = False,
    sparse_exact_direct: bool = False,
    size: int,
    sparse_max_size: int,
    large_cpu_sparse_rescue: bool = False,
    sparse_xblock_rescue_active: bool = False,
    sparse_sxblock_rescue_active: bool = False,
    sparse_jax_est_mb: float | None = None,
    sparse_jax_max_mb: float = 0.0,
    pas_fast_accept: bool = False,
    gpu_sparse_skip: bool = False,
) -> RHS1SparseRescueOrdering:
    """Apply sparse-rescue ordering and skip decisions without side effects."""
    enabled = bool(sparse_enabled)
    kind_use = rhs1_sparse_kind_use(sparse_precond_kind=str(sparse_kind_use))
    xblock_active = bool(sparse_xblock_rescue_active)
    sxblock_active = bool(sparse_sxblock_rescue_active)
    prefer_sparse_exact_over_dense_shortcut = False
    reason_dense_shortcut_skip = False
    reason_size_disabled = False
    reason_size_large_cpu = False
    reason_size_exact_direct = False
    reason_size_targeted = False
    reason_sparse_jax_mem_disabled = False
    reason_large_cpu_exact_skips_targeted = False
    reason_pas_fast_accept = False
    reason_gpu_sparse_skip = False
    if enabled and bool(dense_shortcut):
        if bool(sparse_exact_direct):
            prefer_sparse_exact_over_dense_shortcut = True
        else:
            enabled = False
            reason_dense_shortcut_skip = True
    if enabled and int(size) > int(sparse_max_size):
        if bool(large_cpu_sparse_rescue):
            reason_size_large_cpu = True
        elif bool(sparse_exact_direct):
            reason_size_exact_direct = True
        elif xblock_active or sxblock_active:
            reason_size_targeted = True
        else:
            enabled = False
            reason_size_disabled = True
    if enabled and str(kind_use) == "jax" and (sparse_jax_est_mb is not None):
        if float(sparse_jax_max_mb) > 0.0 and float(sparse_jax_est_mb) > float(
            sparse_jax_max_mb
        ):
            enabled = False
            reason_sparse_jax_mem_disabled = True
    if bool(large_cpu_sparse_rescue) and bool(sparse_exact_direct):
        xblock_active = False
        sxblock_active = False
        reason_large_cpu_exact_skips_targeted = True
    if bool(pas_fast_accept):
        enabled = False
        reason_pas_fast_accept = True
    if bool(gpu_sparse_skip):
        enabled = False
        reason_gpu_sparse_skip = True
    return RHS1SparseRescueOrdering(
        enabled=bool(enabled),
        kind_use=str(kind_use),
        xblock_rescue_active=bool(xblock_active),
        sxblock_rescue_active=bool(sxblock_active),
        prefer_sparse_exact_over_dense_shortcut=bool(
            prefer_sparse_exact_over_dense_shortcut
        ),
        reason_dense_shortcut_skip=bool(reason_dense_shortcut_skip),
        reason_size_disabled=bool(reason_size_disabled),
        reason_size_large_cpu=bool(reason_size_large_cpu),
        reason_size_exact_direct=bool(reason_size_exact_direct),
        reason_size_targeted=bool(reason_size_targeted),
        reason_sparse_jax_mem_disabled=bool(reason_sparse_jax_mem_disabled),
        reason_large_cpu_exact_skips_targeted=bool(
            reason_large_cpu_exact_skips_targeted
        ),
        reason_pas_fast_accept=bool(reason_pas_fast_accept),
        reason_gpu_sparse_skip=bool(reason_gpu_sparse_skip),
    )


def rhs1_sparse_rescue_policy_setup(
    *,
    sparse_precond_mode: str,
    sparse_precond_kind: str,
    has_fp: bool,
    has_pas: bool,
    residual_norm: float,
    target: float,
    rhs_mode: int,
    include_phi1: bool,
    size: int,
    sparse_max_size: int,
    precond_dtype: Any,
    dense_shortcut: bool = False,
    sparse_exact_direct: bool = False,
    large_cpu_sparse_rescue: bool = False,
    sparse_xblock_rescue_active: bool = False,
    sparse_sxblock_rescue_active: bool = False,
    sparse_jax_max_mb: float = 0.0,
    pas_fast_accept: bool = False,
    gpu_sparse_skip: bool = False,
) -> RHS1SparseRescuePolicySetup:
    """Resolve sparse-rescue policy and its JAX dense-memory admission estimate."""

    sparse_enabled = rhs1_sparse_enabled_initial(
        sparse_precond_mode=sparse_precond_mode,
        has_fp=bool(has_fp),
        has_pas=bool(has_pas),
        residual_norm=float(residual_norm),
        target=float(target),
        rhs_mode=int(rhs_mode),
        include_phi1=bool(include_phi1),
    )
    sparse_kind_use = rhs1_sparse_kind_use(sparse_precond_kind=sparse_precond_kind)
    sparse_jax_est_mb: float | None = None
    if sparse_enabled and sparse_kind_use == "jax" and int(size) <= int(sparse_max_size):
        bytes_per = float(np.dtype(precond_dtype).itemsize)
        sparse_jax_est_mb = (int(size) ** 2) * bytes_per / 1.0e6

    ordering = rhs1_resolved_sparse_rescue_ordering(
        sparse_enabled=bool(sparse_enabled),
        sparse_kind_use=sparse_kind_use,
        dense_shortcut=bool(dense_shortcut),
        sparse_exact_direct=bool(sparse_exact_direct),
        size=int(size),
        sparse_max_size=int(sparse_max_size),
        large_cpu_sparse_rescue=bool(large_cpu_sparse_rescue),
        sparse_xblock_rescue_active=bool(sparse_xblock_rescue_active),
        sparse_sxblock_rescue_active=bool(sparse_sxblock_rescue_active),
        sparse_jax_est_mb=sparse_jax_est_mb,
        sparse_jax_max_mb=float(sparse_jax_max_mb),
        pas_fast_accept=bool(pas_fast_accept),
        gpu_sparse_skip=bool(gpu_sparse_skip),
    )
    sparse_jax_memory_disabled_message: str | None = None
    if ordering.reason_sparse_jax_mem_disabled and sparse_jax_est_mb is not None:
        sparse_jax_memory_disabled_message = (
            "sparse_jax: disabled "
            f"(est_mem={sparse_jax_est_mb:.1f} MB > max_mb={float(sparse_jax_max_mb):.1f})"
        )
    return RHS1SparseRescuePolicySetup(
        enabled=bool(ordering.enabled),
        kind_use=str(ordering.kind_use),
        ordering=ordering,
        sparse_jax_est_mb=sparse_jax_est_mb,
        sparse_jax_memory_disabled_message=sparse_jax_memory_disabled_message,
    )


def rhs1_sparse_rescue_initial_messages(
    *,
    ordering: RHS1SparseRescueOrdering,
    size: int,
    sparse_max_size: int,
    sparse_jax_memory_disabled_message: str | None = None,
    large_cpu_sparse_exact_lu: bool | None = None,
    large_cpu_label: str = "large CPU sparse",
    targeted_rescue_kind: str | None = None,
) -> tuple[tuple[int, str], ...]:
    """Format initial sparse-rescue policy progress messages without side effects."""

    messages: list[tuple[int, str]] = []
    if ordering.prefer_sparse_exact_over_dense_shortcut:
        messages.append(
            (
                0,
                "solve_v3_full_system_linear_gmres: preferring sparse exact rescue over dense shortcut",
            )
        )
    if ordering.reason_size_large_cpu:
        sparse_exact_lu = bool(large_cpu_sparse_exact_lu)
        messages.append(
            (
                0,
                "solve_v3_full_system_linear_gmres: "
                f"{large_cpu_label} {'LU' if sparse_exact_lu else 'ILU'} rescue "
                f"(size={int(size)} > max={int(sparse_max_size)})",
            )
        )
    elif ordering.reason_size_exact_direct:
        messages.append(
            (
                0,
                "solve_v3_full_system_linear_gmres: exact sparse LU rescue "
                f"(size={int(size)} > max={int(sparse_max_size)})",
            )
        )
    elif ordering.reason_size_targeted:
        rescue_kind = str(targeted_rescue_kind or "targeted")
        messages.append(
            (
                0,
                "solve_v3_full_system_linear_gmres: targeted sparse "
                f"{rescue_kind} rescue (size={int(size)} > max={int(sparse_max_size)})",
            )
        )
    elif ordering.reason_size_disabled:
        messages.append(
            (1, f"sparse_ilu: disabled (size={int(size)} > max={int(sparse_max_size)})")
        )
    if sparse_jax_memory_disabled_message is not None:
        messages.append((1, sparse_jax_memory_disabled_message))
    return tuple(messages)


def rhs1_sparse_rescue_tail_skip_messages(
    *,
    ordering: RHS1SparseRescueOrdering,
    residual_norm: float,
    rhs1_precond_kind: str,
) -> tuple[tuple[int, str], ...]:
    """Format sparse-rescue tail skip messages without moving driver control flow."""

    messages: list[tuple[int, str]] = []
    if ordering.reason_large_cpu_exact_skips_targeted:
        messages.append(
            (
                1,
                "solve_v3_full_system_linear_gmres: exact large-CPU sparse LU selected "
                "-> skipping targeted sparse xblock/sxblock rescue",
            )
        )
    if ordering.reason_pas_fast_accept:
        messages.append(
            (
                1,
                "solve_v3_full_system_linear_gmres: PAS fast-accept "
                f"(residual={float(residual_norm):.3e}) -> skip sparse rescue tail",
            )
        )
    if ordering.reason_gpu_sparse_skip:
        messages.append(
            (
                1,
                "solve_v3_full_system_linear_gmres: GPU sparse fallback skipped after "
                f"{rhs1_precond_kind} accept (residual={float(residual_norm):.3e})",
            )
        )
    return tuple(messages)


# From sfincs_jax.rhs1_sparse_polish_policy
def rhs1_polish_enabled(*, env_name: str) -> bool:
    """Return whether a polish stage is enabled by its boolean-like env var."""
    env = os.environ.get(env_name, "").strip().lower()
    return env not in {"0", "false", "no", "off"}


def rhs1_parse_accept_ratio(*, env_name: str, default: float) -> float:
    """Parse an acceptance ratio with a floor of 1."""
    env = os.environ.get(env_name, "").strip()
    try:
        value = float(env) if env else float(default)
    except ValueError:
        value = float(default)
    return max(1.0, float(value))


def rhs1_parse_polish_gmres_config(
    *,
    restart_env_name: str,
    maxiter_env_name: str,
    default_restart: int,
    default_maxiter: int,
    min_restart: int = 5,
    min_maxiter: int = 5,
    active_size: int | None = None,
    large_active_min_env_name: str = "",
    large_default_restart_env_name: str = "",
    large_default_maxiter_env_name: str = "",
    default_large_restart: int | None = None,
    default_large_maxiter: int | None = None,
) -> tuple[int, int]:
    """Parse bounded restart/maxiter settings for a short GMRES polish."""
    restart_env = os.environ.get(restart_env_name, "").strip()
    maxiter_env = os.environ.get(maxiter_env_name, "").strip()
    default_restart_use = int(default_restart)
    default_maxiter_use = int(default_maxiter)
    if active_size is not None and (
        default_large_restart is not None or default_large_maxiter is not None
    ):
        large_min_env = (
            os.environ.get(large_active_min_env_name, "").strip()
            if large_active_min_env_name
            else ""
        )
        large_restart_env = (
            os.environ.get(large_default_restart_env_name, "").strip()
            if large_default_restart_env_name
            else ""
        )
        large_default_env = (
            os.environ.get(large_default_maxiter_env_name, "").strip()
            if large_default_maxiter_env_name
            else ""
        )
        try:
            large_min = int(large_min_env) if large_min_env else 200000
        except ValueError:
            large_min = 200000
        if int(active_size) >= max(1, int(large_min)):
            if default_large_restart is not None and (not restart_env):
                try:
                    large_restart = (
                        int(large_restart_env)
                        if large_restart_env
                        else int(default_large_restart)
                    )
                except ValueError:
                    large_restart = int(default_large_restart)
                default_restart_use = min(
                    int(default_restart_use), max(1, int(large_restart))
                )
            if default_large_maxiter is not None and (not maxiter_env):
                try:
                    large_default = (
                        int(large_default_env)
                        if large_default_env
                        else int(default_large_maxiter)
                    )
                except ValueError:
                    large_default = int(default_large_maxiter)
                default_maxiter_use = min(
                    int(default_maxiter_use), max(1, int(large_default))
                )
    try:
        restart = int(restart_env) if restart_env else int(default_restart_use)
    except ValueError:
        restart = int(default_restart_use)
    try:
        maxiter = int(maxiter_env) if maxiter_env else int(default_maxiter_use)
    except ValueError:
        maxiter = int(default_maxiter_use)
    return (max(int(min_restart), int(restart)), max(int(min_maxiter), int(maxiter)))


# From sfincs_jax.rhs1_stage2_policy
_PAS_STAGE2_SKIP_BASE_KINDS = frozenset(
    {"pas_lite", "pas_hybrid", "pas_tz", "pas_schur", "pas_tokamak_theta"}
)

_PAS_STAGE2_EXTENDED_SKIP_BASE_KINDS = frozenset(
    {"pas_ilu", "schur", "xblock_tz", "xblock_tz_lmax"}
)

_PAS_STAGE2_WEAK_SKIP_KINDS = frozenset({"collision", "point", "xmg"})


def rhs1_stage2_ratio(*, use_dkes: bool) -> float:
    """Return the stage-2 residual-ratio trigger with DKES tightening."""
    stage2_ratio_env = os.environ.get("SFINCS_JAX_LINEAR_STAGE2_RATIO", "").strip()
    try:
        stage2_ratio = float(stage2_ratio_env) if stage2_ratio_env else 100.0
    except ValueError:
        stage2_ratio = 100.0
    if use_dkes:
        stage2_ratio = min(float(stage2_ratio), 1.0)
    return float(stage2_ratio)


def rhs1_stage2_trigger(*, res_ratio: float, use_dkes: bool) -> bool:
    """Return whether stage-2 should be considered from the residual ratio."""
    ratio = rhs1_stage2_ratio(use_dkes=use_dkes)
    return bool(res_ratio > ratio) if ratio > 0 else True


def rhs1_fp_force_stage2(
    *, has_fp: bool, include_phi1: bool, residual_norm: float
) -> bool:
    """Return whether FP runs should force a stage-2 polish based on absolute residual."""
    fp_stage2_abs_env = os.environ.get("SFINCS_JAX_FP_STAGE2_ABS", "").strip()
    try:
        fp_stage2_abs = float(fp_stage2_abs_env) if fp_stage2_abs_env else 1e-06
    except ValueError:
        fp_stage2_abs = 1e-06
    return bool(
        has_fp and (not include_phi1) and (float(residual_norm) > float(fp_stage2_abs))
    )


def rhs1_pas_stage2_skip(
    *, has_pas: bool, rhs1_precond_kind: str | None, res_ratio: float
) -> bool:
    """Return whether PAS runs should skip stage-2 and move to later rescue logic.

    Stage-2 GMRES is useful as a polish when the first residual is close enough
    to target. For the historical PAS-lite/hybrid/tz family, very large
    residual ratios should move directly to later rescue logic. Broader skips
    for Schur/xblock/PAS-ILU routes are opt-in only because production-floor
    tests show they can produce faster but non-parity-clean completed outputs.
    """
    if not has_pas:
        return False
    if rhs1_precond_kind not in _PAS_STAGE2_SKIP_BASE_KINDS:
        if rhs1_precond_kind in _PAS_STAGE2_WEAK_SKIP_KINDS:
            weak_skip_env = os.environ.get(
                "SFINCS_JAX_PAS_STAGE2_WEAK_SKIP_RATIO", ""
            ).strip()
            try:
                weak_skip_ratio = (
                    float(weak_skip_env) if weak_skip_env else 1000000000000.0
                )
            except ValueError:
                weak_skip_ratio = 1000000000000.0
            if weak_skip_ratio <= 0.0:
                return False
            return float(res_ratio) >= float(weak_skip_ratio)
        extended_env = (
            os.environ.get("SFINCS_JAX_PAS_STAGE2_SKIP_EXTENDED", "").strip().lower()
        )
        if extended_env not in {"1", "true", "yes", "on"}:
            return False
        if rhs1_precond_kind not in _PAS_STAGE2_EXTENDED_SKIP_BASE_KINDS:
            return False
    pas_stage2_skip_env = os.environ.get("SFINCS_JAX_PAS_STAGE2_SKIP_RATIO", "").strip()
    try:
        pas_stage2_skip_ratio = (
            float(pas_stage2_skip_env) if pas_stage2_skip_env else 1000000.0
        )
    except ValueError:
        pas_stage2_skip_ratio = 1000000.0
    return float(res_ratio) >= float(pas_stage2_skip_ratio)


def rhs1_pas_tz_guarded_stage2_retry() -> bool:
    """Return whether guarded PAS-TZ fallbacks should attempt stage-2 GMRES.

    Guarded PAS-TZ fallbacks are selected after the dense structured PAS-TZ
    builder is rejected by the memory gate. Their purpose is to keep the run
    bounded and diagnostic-rich; strict stage-2 retries can turn an otherwise
    bounded fallback into the same long-running solver-path problem the guard is
    meant to avoid. Users can still opt in when profiling a candidate fallback.
    """
    env = (
        os.environ.get("SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_STAGE2_RETRY", "")
        .strip()
        .lower()
    )
    return env in {"1", "true", "yes", "on"}


__all__ = (
    "RHS1Constraint0PETScCompatConfig",
    "RHS1FastPostXBlockPolishControls",
    "RHS1FPLowLPolishControls",
    "RHS1FPResidualPolishControls",
    "RHS1QIDeviceRankBudget",
    "RHS1QIDeviceSetupSummary",
    "RHS1SparseJAXConfig",
    "RHS1SparseOperatorAdmission",
    "RHS1SparsePreconditionerConfig",
    "RHS1SparseRescueOrdering",
    "RHS1SparseRescuePolicySetup",
    "parse_rhs1_pas_tz_guarded_structured_levels",
    "rhs1_constraint0_dense_fallback_allowed",
    "rhs1_constraint0_petsc_compat",
    "rhs1_constraint0_petsc_compat_config_from_env",
    "rhs1_constraint0_petsc_compat_regularization",
    "rhs1_constraint0_sparse_first",
    "rhs1_fast_post_xblock_polish_allowed",
    "rhs1_fast_post_xblock_polish_controls_from_env",
    "rhs1_fp_force_stage2",
    "rhs1_fp_low_l_polish_controls_from_env",
    "rhs1_fp_residual_polish_controls_from_env",
    "rhs1_fp_targeted_polish_allowed",
    "rhs1_fp_xblock_global_correction_allowed",
    "rhs1_host_factor_probe_ok",
    "rhs1_parse_accept_ratio",
    "rhs1_parse_polish_gmres_config",
    "rhs1_pas_fast_accept",
    "rhs1_pas_stage2_skip",
    "rhs1_pas_tz_guarded_stage2_retry",
    "rhs1_polish_enabled",
    "rhs1_prefer_sparse_over_dense_shortcut",
    "rhs1_qi_device_extra_coarse_controls",
    "rhs1_qi_device_extra_coarse_metadata",
    "rhs1_qi_device_extra_coarse_setup_kwargs",
    "rhs1_qi_device_coupled_install_on_reject_requested",
    "rhs1_qi_device_probe_uses_minres_step",
    "rhs1_qi_device_progress_messages",
    "rhs1_qi_device_rank_budget",
    "rhs1_qi_device_residual_correction_controls",
    "rhs1_qi_device_residual_correction_metadata",
    "rhs1_qi_device_residual_correction_setup_kwargs",
    "rhs1_qi_device_setup_summary",
    "rhs1_qi_device_status_fields",
    "rhs1_qi_device_tail_block_required",
    "rhs1_resolved_sparse_rescue_ordering",
    "rhs1_scipy_rescue_abs_floor_after_xblock",
    "rhs1_scipy_rescue_active_size_allowed",
    "rhs1_skip_global_sparse_after_xblock_allowed",
    "rhs1_sparse_enabled_initial",
    "rhs1_sparse_exact_lu_requested",
    "rhs1_sparse_jax_config_from_env",
    "rhs1_sparse_operator_admission",
    "rhs1_sparse_rescue_initial_messages",
    "rhs1_sparse_kind_use",
    "rhs1_sparse_prefer_skips_stage2",
    "rhs1_sparse_preconditioner_config_from_env",
    "rhs1_sparse_rescue_policy_setup",
    "rhs1_sparse_rescue_tail_skip_messages",
    "rhs1_stage2_ratio",
    "rhs1_stage2_trigger",
    "rhs1_xblock_fallback_initial_guess",
)
