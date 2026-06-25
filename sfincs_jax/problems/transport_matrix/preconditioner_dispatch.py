"""Shared transport preconditioner selection and build helpers.

This module keeps the transport preconditioner decision ladder and builder
dispatch out of ``v3_driver.py`` while preserving the existing runtime
semantics. The goal is structural: make the transport solve orchestration
readable and directly testable without changing parity behavior.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import os
from typing import Any

import jax.numpy as jnp


Preconditioner = Callable[[Any], Any]
Builder = Callable[..., Preconditioner]


@dataclass(frozen=True)
class TransportPreconditionerContext:
    op: Any
    active_size: int
    use_active_dof_mode: bool
    reduce_full: Callable[[Any], Any] | None = None
    expand_reduced: Callable[[Any], Any] | None = None
    active_indices_np: Any | None = None
    emit: Callable[[int, str], None] | None = None


@dataclass(frozen=True)
class TransportPreconditionerDispatchBuilders:
    collision_builder: Builder
    sxblock_builder: Builder
    block_builder: Builder
    xmg_builder: Builder
    theta_dd_builder: Builder
    theta_schwarz_builder: Builder
    zeta_dd_builder: Builder
    zeta_schwarz_builder: Builder
    tzfft_builder: Builder
    sparse_jax_builder: Builder
    sparse_jax_cache_key: Callable[[Any, str], tuple[object, ...]]
    apply_operator_cached: Callable[[Any, jnp.ndarray], jnp.ndarray]
    precond_dtype: Callable[[int], jnp.dtype]
    fp_tzfft_builder: Builder | None = None
    fp_tzfft_line_builder: Builder | None = None
    fp_tzfft_line_schur_builder: Builder | None = None
    fp_local_geom_line_builder: Builder | None = None
    fp_xblock_tz_lu_builder: Builder | None = None
    fp_xblock_tz_lu_schur_builder: Builder | None = None
    fp_structured_fblock_lu_builder: Builder | None = None
    fp_fortran_reduced_lu_builder: Builder | None = None
    fp_direct_active_block_schur_builder: Builder | None = None


@dataclass(frozen=True)
class TransportDDConfig:
    block_theta: int
    overlap_theta: int
    block_zeta: int
    overlap_zeta: int


@dataclass(frozen=True)
class TransportSparseJaxConfig:
    drop_tol: float
    drop_rel: float
    reg: float
    omega: float
    sweeps: int
    max_mb: float


def normalize_transport_preconditioner_kind(*, env_value: str) -> str | None:
    env = str(env_value).strip().lower()
    if env in {"0", "none", "off", "false", "no"}:
        return None
    if env in {
        "collision",
        "block",
        "block_jacobi",
        "sxblock",
        "block_sx",
        "species_x",
        "xmg",
        "multigrid",
        "theta_dd",
        "theta_block",
        "dd_theta",
        "dd_t",
        "theta_schwarz",
        "schwarz_theta",
        "zeta_dd",
        "zeta_block",
        "dd_zeta",
        "dd_z",
        "zeta_schwarz",
        "schwarz_zeta",
        "tzfft",
        "fp_tzfft",
        "fp_tzfft_line",
        "fp_tzfft_line_schur",
        "fp_tzfft_schur",
        "fp_streaming_line_schur",
        "fp_block_thomas_schur",
        "fp_line_schur",
        "fp_local_geom_line",
        "fp_geom_line",
        "fp_local_line",
        "fp_nonavg_line",
        "fp_xblock_tz_lu",
        "fp_xblock_tz_lu_schur",
        "fp_xblock_schur",
        "fp_xblock_lu_schur",
        "fp_tz_xblock_lu_schur",
        "fp_angular_xblock_lu_schur",
        "fp_xblock_lu",
        "fp_tz_xblock_lu",
        "fp_angular_xblock_lu",
        "fp_structured_fblock_lu",
        "fp_fblock_lu",
        "fp_full_fblock_lu",
        "fp_kinetic_lu",
        "fp_fortran_reduced_lu",
        "fp_global_fortran_reduced_lu",
        "fp_petsc_like_lu",
        "fp_reduced_pmat_lu",
        "fp_direct_active_block_schur",
        "fp_direct_active_block_lu",
        "fp_active_true_block_schur",
        "fp_active_true_block",
        "fp_true_block_schur",
        "fp_true_block_lu",
        "fp_streaming_line",
        "fp_block_thomas",
        "fp_line",
        "fp_streaming_fft",
        "streaming_fft",
        "stream_fft",
        "sparse_jax",
    }:
        if env in {"streaming_fft", "stream_fft"}:
            return "tzfft"
        if env in {"fp_streaming_fft"}:
            return "fp_tzfft"
        if env in {"fp_tzfft_schur", "fp_streaming_line_schur", "fp_block_thomas_schur", "fp_line_schur"}:
            return "fp_tzfft_line_schur"
        if env in {"fp_geom_line", "fp_local_line", "fp_nonavg_line"}:
            return "fp_local_geom_line"
        if env in {"fp_xblock_schur", "fp_xblock_lu_schur", "fp_tz_xblock_lu_schur", "fp_angular_xblock_lu_schur"}:
            return "fp_xblock_tz_lu_schur"
        if env in {"fp_xblock_lu", "fp_tz_xblock_lu", "fp_angular_xblock_lu"}:
            return "fp_xblock_tz_lu"
        if env in {"fp_fblock_lu", "fp_full_fblock_lu", "fp_kinetic_lu"}:
            return "fp_structured_fblock_lu"
        if env in {"fp_global_fortran_reduced_lu", "fp_petsc_like_lu", "fp_reduced_pmat_lu"}:
            return "fp_fortran_reduced_lu"
        if env in {
            "fp_direct_active_block_lu",
            "fp_active_true_block_schur",
            "fp_active_true_block",
            "fp_true_block_schur",
            "fp_true_block_lu",
        }:
            return "fp_direct_active_block_schur"
        if env in {"fp_streaming_line", "fp_block_thomas", "fp_line"}:
            return "fp_tzfft_line"
        if env in {"theta_block", "dd_theta", "dd_t"}:
            return "theta_dd"
        if env in {"theta_schwarz", "schwarz_theta"}:
            return "theta_schwarz"
        if env in {"zeta_block", "dd_zeta", "dd_z"}:
            return "zeta_dd"
        if env in {"zeta_schwarz", "schwarz_zeta"}:
            return "zeta_schwarz"
        return env
    return "auto"


def transport_dd_config_from_env(*, op: Any) -> TransportDDConfig:
    block_t_env = os.environ.get("SFINCS_JAX_TRANSPORT_DD_BLOCK_T", "").strip()
    block_z_env = os.environ.get("SFINCS_JAX_TRANSPORT_DD_BLOCK_Z", "").strip()
    overlap_env = os.environ.get("SFINCS_JAX_TRANSPORT_DD_OVERLAP", "").strip()
    try:
        block_t = int(block_t_env) if block_t_env else 8
    except ValueError:
        block_t = 8
    try:
        block_z = int(block_z_env) if block_z_env else 8
    except ValueError:
        block_z = 8
    try:
        overlap = int(overlap_env) if overlap_env else 1
    except ValueError:
        overlap = 1
    block_t = max(1, min(int(getattr(op, "n_theta", 1)), int(block_t)))
    block_z = max(1, min(int(getattr(op, "n_zeta", 1)), int(block_z)))
    overlap_t = max(0, min(int(block_t) - 1, int(overlap)))
    overlap_z = max(0, min(int(block_z) - 1, int(overlap)))
    return TransportDDConfig(
        block_theta=int(block_t),
        overlap_theta=int(overlap_t),
        block_zeta=int(block_z),
        overlap_zeta=int(overlap_z),
    )


def transport_sparse_jax_config_from_env() -> TransportSparseJaxConfig:
    drop_tol_env = os.environ.get("SFINCS_JAX_TRANSPORT_SPARSE_DROP_TOL", "").strip()
    drop_rel_env = os.environ.get("SFINCS_JAX_TRANSPORT_SPARSE_DROP_REL", "").strip()
    reg_env = os.environ.get("SFINCS_JAX_TRANSPORT_SPARSE_JAX_REG", "").strip()
    omega_env = os.environ.get("SFINCS_JAX_TRANSPORT_SPARSE_JAX_OMEGA", "").strip()
    sweeps_env = os.environ.get("SFINCS_JAX_TRANSPORT_SPARSE_JAX_SWEEPS", "").strip()
    max_env = os.environ.get("SFINCS_JAX_TRANSPORT_SPARSE_JAX_MAX_MB", "").strip()
    try:
        drop_tol = float(drop_tol_env) if drop_tol_env else 0.0
    except ValueError:
        drop_tol = 0.0
    try:
        drop_rel = float(drop_rel_env) if drop_rel_env else 1.0e-6
    except ValueError:
        drop_rel = 1.0e-6
    try:
        reg = float(reg_env) if reg_env else 1e-10
    except ValueError:
        reg = 1e-10
    try:
        omega = float(omega_env) if omega_env else 0.8
    except ValueError:
        omega = 0.8
    try:
        sweeps = int(sweeps_env) if sweeps_env else 2
    except ValueError:
        sweeps = 2
    try:
        max_mb = float(max_env) if max_env else 128.0
    except ValueError:
        max_mb = 128.0
    return TransportSparseJaxConfig(
        drop_tol=float(drop_tol),
        drop_rel=float(drop_rel),
        reg=float(reg),
        omega=float(omega),
        sweeps=max(1, int(sweeps)),
        max_mb=float(max_mb),
    )


def auto_transport_preconditioner_choice(
    *,
    op: Any,
    default_solver_kind: str,
    parallel_workers: int,
    dense_mem_block: bool,
    tzfft_backend_allowed: bool,
    shard_axis: str | None,
) -> tuple[str, str | None]:
    block_max_env = os.environ.get("SFINCS_JAX_TRANSPORT_PRECOND_BLOCK_MAX", "").strip()
    sxblock_max_env = os.environ.get("SFINCS_JAX_TRANSPORT_SXBLOCK_MAX", "").strip()
    dd_auto_min_env = os.environ.get("SFINCS_JAX_TRANSPORT_DD_AUTO_MIN", "").strip()
    try:
        block_max = int(block_max_env) if block_max_env else 5000
    except ValueError:
        block_max = 5000
    try:
        sxblock_max = int(sxblock_max_env) if sxblock_max_env else 64
    except ValueError:
        sxblock_max = 64
    try:
        dd_auto_min = int(dd_auto_min_env) if dd_auto_min_env else 0
    except ValueError:
        dd_auto_min = 0

    n_block = int(op.n_species) * int(op.n_x)
    precond_kind: str
    strong_precond_kind: str | None = None
    if op.fblock.fp is not None:
        fp_fortran_env = os.environ.get("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_AUTO", "").strip().lower()
        fp_fortran_disabled = fp_fortran_env in {"0", "false", "no", "off"}
        fp_fortran_forced = fp_fortran_env in {"1", "true", "yes", "on"}
        fp_fortran_min_env = os.environ.get("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_AUTO_MIN", "").strip()
        try:
            fp_fortran_min = int(fp_fortran_min_env) if fp_fortran_min_env else 0
        except ValueError:
            fp_fortran_min = 0
        fp_fblock_env = os.environ.get("SFINCS_JAX_TRANSPORT_FP_STRUCTURED_FBLOCK_LU_AUTO", "").strip().lower()
        fp_fblock_disabled = fp_fblock_env in {"", "0", "false", "no", "off"}
        fp_fblock_forced = fp_fblock_env in {"1", "true", "yes", "on"}
        fp_fblock_min_env = os.environ.get("SFINCS_JAX_TRANSPORT_FP_STRUCTURED_FBLOCK_LU_AUTO_MIN", "").strip()
        try:
            fp_fblock_min = int(fp_fblock_min_env) if fp_fblock_min_env else 50000
        except ValueError:
            fp_fblock_min = 50000
        fp_xblock_env = os.environ.get("SFINCS_JAX_TRANSPORT_FP_XBLOCK_TZ_LU_AUTO", "").strip().lower()
        fp_xblock_disabled = fp_xblock_env in {"", "0", "false", "no", "off"}
        fp_xblock_forced = fp_xblock_env in {"1", "true", "yes", "on"}
        fp_xblock_min_env = os.environ.get("SFINCS_JAX_TRANSPORT_FP_XBLOCK_TZ_LU_AUTO_MIN", "").strip()
        try:
            fp_xblock_min = int(fp_xblock_min_env) if fp_xblock_min_env else 50000
        except ValueError:
            fp_xblock_min = 50000
        fp_xblock_schur_env = os.environ.get("SFINCS_JAX_TRANSPORT_FP_XBLOCK_TZ_LU_SCHUR_AUTO", "").strip().lower()
        fp_xblock_schur_disabled = fp_xblock_schur_env in {"", "0", "false", "no", "off"}
        fp_xblock_schur_forced = fp_xblock_schur_env in {"1", "true", "yes", "on"}
        fp_xblock_schur_min_env = os.environ.get("SFINCS_JAX_TRANSPORT_FP_XBLOCK_TZ_LU_SCHUR_AUTO_MIN", "").strip()
        try:
            fp_xblock_schur_min = int(fp_xblock_schur_min_env) if fp_xblock_schur_min_env else 50000
        except ValueError:
            fp_xblock_schur_min = 50000
        fp_geom_env = os.environ.get("SFINCS_JAX_TRANSPORT_FP_LOCAL_GEOM_LINE_AUTO", "").strip().lower()
        fp_geom_disabled = fp_geom_env in {"", "0", "false", "no", "off"}
        fp_geom_forced = fp_geom_env in {"1", "true", "yes", "on"}
        fp_geom_min_env = os.environ.get("SFINCS_JAX_TRANSPORT_FP_LOCAL_GEOM_LINE_AUTO_MIN", "").strip()
        try:
            fp_geom_min = int(fp_geom_min_env) if fp_geom_min_env else 50000
        except ValueError:
            fp_geom_min = 50000
        fp_schur_env = os.environ.get("SFINCS_JAX_TRANSPORT_FP_TZFFT_LINE_SCHUR_AUTO", "").strip().lower()
        fp_schur_disabled = fp_schur_env in {"", "0", "false", "no", "off"}
        fp_schur_forced = fp_schur_env in {"1", "true", "yes", "on"}
        fp_schur_min_env = os.environ.get("SFINCS_JAX_TRANSPORT_FP_TZFFT_LINE_SCHUR_AUTO_MIN", "").strip()
        try:
            fp_schur_min = int(fp_schur_min_env) if fp_schur_min_env else 50000
        except ValueError:
            fp_schur_min = 50000
        fp_line_env = os.environ.get("SFINCS_JAX_TRANSPORT_FP_TZFFT_LINE_AUTO", "").strip().lower()
        fp_line_disabled = fp_line_env in {"", "0", "false", "no", "off"}
        fp_line_forced = fp_line_env in {"1", "true", "yes", "on"}
        fp_line_min_env = os.environ.get("SFINCS_JAX_TRANSPORT_FP_TZFFT_LINE_AUTO_MIN", "").strip()
        try:
            fp_line_min = int(fp_line_min_env) if fp_line_min_env else 50000
        except ValueError:
            fp_line_min = 50000
        fp_line_candidate = bool(
            (not fp_line_disabled)
            and int(getattr(op, "rhs_mode", 0) or 0) in {2, 3}
            and not bool(getattr(op, "include_phi1", False))
            and int(getattr(op, "n_theta", 0) or 0) * int(getattr(op, "n_zeta", 0) or 0) >= 64
            and (fp_line_forced or int(getattr(op, "total_size", 0) or 0) >= max(1, int(fp_line_min)))
        )
        fp_schur_candidate = bool(
            (not fp_schur_disabled)
            and int(getattr(op, "rhs_mode", 0) or 0) in {2, 3}
            and not bool(getattr(op, "include_phi1", False))
            and int(getattr(op, "n_theta", 0) or 0) * int(getattr(op, "n_zeta", 0) or 0) >= 64
            and (fp_schur_forced or int(getattr(op, "total_size", 0) or 0) >= max(1, int(fp_schur_min)))
        )
        fp_geom_candidate = bool(
            (not fp_geom_disabled)
            and int(getattr(op, "rhs_mode", 0) or 0) in {2, 3}
            and not bool(getattr(op, "include_phi1", False))
            and int(getattr(op, "n_theta", 0) or 0) * int(getattr(op, "n_zeta", 0) or 0) >= 64
            and (fp_geom_forced or int(getattr(op, "total_size", 0) or 0) >= max(1, int(fp_geom_min)))
        )
        fp_fblock_candidate = bool(
            (not fp_fblock_disabled)
            and int(getattr(op, "rhs_mode", 0) or 0) in {2, 3}
            and not bool(getattr(op, "include_phi1", False))
            and (fp_fblock_forced or int(getattr(op, "total_size", 0) or 0) >= max(1, int(fp_fblock_min)))
        )
        fp_xblock_candidate = bool(
            (not fp_xblock_disabled)
            and int(getattr(op, "rhs_mode", 0) or 0) in {2, 3}
            and not bool(getattr(op, "include_phi1", False))
            and int(getattr(op, "n_theta", 0) or 0) * int(getattr(op, "n_zeta", 0) or 0) >= 64
            and (fp_xblock_forced or int(getattr(op, "total_size", 0) or 0) >= max(1, int(fp_xblock_min)))
        )
        fp_xblock_schur_candidate = bool(
            (not fp_xblock_schur_disabled)
            and int(getattr(op, "rhs_mode", 0) or 0) in {2, 3}
            and not bool(getattr(op, "include_phi1", False))
            and int(getattr(op, "n_theta", 0) or 0) * int(getattr(op, "n_zeta", 0) or 0) >= 64
            and (
                fp_xblock_schur_forced
                or int(getattr(op, "total_size", 0) or 0) >= max(1, int(fp_xblock_schur_min))
            )
        )
        other_fp_forced = any(
            (
                fp_line_forced,
                fp_schur_forced,
                fp_geom_forced,
                fp_fblock_forced,
                fp_xblock_forced,
                fp_xblock_schur_forced,
            )
        )
        fp_fortran_candidate = bool(
            (not fp_fortran_disabled)
            and int(getattr(op, "rhs_mode", 0) or 0) in {2, 3}
            and not bool(getattr(op, "include_phi1", False))
            and (
                fp_fortran_forced
                or (
                    (not other_fp_forced)
                    and int(getattr(op, "total_size", 0) or 0) >= max(1, int(fp_fortran_min))
                )
            )
        )
        if fp_fortran_candidate:
            precond_kind = "fp_fortran_reduced_lu"
            strong_precond_kind = "fp_fortran_reduced_lu"
        elif fp_xblock_schur_candidate:
            precond_kind = "fp_xblock_tz_lu_schur"
            strong_precond_kind = "fp_xblock_tz_lu_schur"
        elif fp_xblock_candidate:
            precond_kind = "fp_xblock_tz_lu"
            strong_precond_kind = "fp_xblock_tz_lu"
        elif fp_fblock_candidate:
            precond_kind = "fp_structured_fblock_lu"
            strong_precond_kind = "fp_structured_fblock_lu"
        elif fp_geom_candidate:
            precond_kind = "fp_local_geom_line"
            strong_precond_kind = "fp_local_geom_line"
        elif fp_schur_candidate:
            precond_kind = "fp_tzfft_line_schur"
            strong_precond_kind = "fp_tzfft_line_schur"
        elif fp_line_candidate:
            precond_kind = "fp_tzfft_line"
            strong_precond_kind = "fp_tzfft_line"
        elif n_block <= sxblock_max:
            precond_kind = "sxblock"
        elif int(op.total_size) <= block_max and str(default_solver_kind) != "bicgstab":
            precond_kind = "sxblock"
        else:
            precond_kind = "collision"
        if (
            not fp_line_candidate
            and not fp_schur_candidate
            and not fp_geom_candidate
            and not fp_fblock_candidate
            and not fp_xblock_candidate
            and not fp_xblock_schur_candidate
            and not fp_fortran_candidate
        ):
            if n_block <= sxblock_max:
                strong_precond_kind = "sxblock"
            elif int(op.total_size) <= block_max:
                strong_precond_kind = "block"
            else:
                strong_precond_kind = "xmg"
    else:
        no_fp = op.fblock.fp is None
        small_x = int(op.n_x) <= 2
        multi_angle = int(op.n_theta) * int(op.n_zeta) >= 64
        if no_fp and small_x and multi_angle and tzfft_backend_allowed:
            precond_kind = "tzfft"
            strong_precond_kind = "tzfft"
        elif int(op.total_size) <= block_max:
            precond_kind = "block"
            strong_precond_kind = "block"
        else:
            precond_kind = "collision"
            strong_precond_kind = "collision" if (no_fp and (not tzfft_backend_allowed)) else "xmg"

    if int(parallel_workers) > 1 and dd_auto_min > 0 and int(op.total_size) >= dd_auto_min:
        if shard_axis == "theta":
            precond_kind = "theta_schwarz"
            strong_precond_kind = "theta_schwarz"
        elif shard_axis == "zeta":
            precond_kind = "zeta_schwarz"
            strong_precond_kind = "zeta_schwarz"
    if dense_mem_block and strong_precond_kind is not None:
        precond_kind = strong_precond_kind
    return precond_kind, strong_precond_kind


def resolve_transport_preconditioner_choice(
    *,
    op: Any,
    transport_precond_kind: str | None,
    default_solver_kind: str,
    parallel_workers: int,
    dense_mem_block: bool,
    tzfft_backend_allowed: bool,
    shard_axis: str | None,
    backend: str,
    emit: Callable[[int, str], None] | None = None,
) -> tuple[str | None, str | None]:
    if transport_precond_kind is None:
        return None, None
    precond_kind = transport_precond_kind
    strong_precond_kind: str | None = None
    if precond_kind == "auto":
        precond_kind, strong_precond_kind = auto_transport_preconditioner_choice(
            op=op,
            default_solver_kind=default_solver_kind,
            parallel_workers=parallel_workers,
            dense_mem_block=dense_mem_block,
            tzfft_backend_allowed=tzfft_backend_allowed,
            shard_axis=shard_axis,
        )
    if precond_kind == "tzfft" and (not tzfft_backend_allowed):
        if emit is not None:
            emit(
                1,
                "solve_v3_transport_matrix_linear_gmres: tzfft preconditioner disabled on "
                f"backend={backend}",
            )
        precond_kind = "collision"
        if strong_precond_kind == "tzfft":
            strong_precond_kind = "collision"
    return precond_kind, strong_precond_kind


def resolve_transport_precondition_side_for_kind(
    *,
    kind: str | None,
    requested_side: str,
) -> tuple[str, bool]:
    """Return a preconditioner-side choice that is valid for the selected kind.

    The FP Fourier line factor uses a forward/backward block-Thomas scan.  It is
    intended as a left preconditioner; current JAX transpose rules for the scan
    path make user-forced right preconditioning fragile on some backends.
    Keeping the guard here makes the solver policy explicit and testable.
    """
    side = str(requested_side).strip().lower()
    if side not in {"left", "right", "none"}:
        side = "left"
    if (
        kind
        in {
            "fp_tzfft_line",
            "fp_tzfft_line_schur",
            "fp_local_geom_line",
            "fp_xblock_tz_lu",
            "fp_xblock_tz_lu_schur",
            "fp_structured_fblock_lu",
            "fp_fortran_reduced_lu",
            "fp_direct_active_block_schur",
        }
        and side == "right"
    ):
        return "left", True
    return side, False


def build_transport_preconditioner_from_kind(
    *,
    kind: str,
    context: TransportPreconditionerContext,
    builders: TransportPreconditionerDispatchBuilders,
    dd_config: TransportDDConfig,
    sparse_jax_config: TransportSparseJaxConfig,
    use_reduced: bool,
) -> Preconditioner:
    reduce_full = context.reduce_full if use_reduced else None
    expand_reduced = context.expand_reduced if use_reduced else None
    size_est = int(context.active_size) if use_reduced else int(context.op.total_size)
    if kind in {"xmg", "multigrid"}:
        return builders.xmg_builder(op=context.op, reduce_full=reduce_full, expand_reduced=expand_reduced)
    if kind == "theta_dd":
        return builders.theta_dd_builder(
            op=context.op,
            block=int(dd_config.block_theta),
            reduce_full=reduce_full,
            expand_reduced=expand_reduced,
        )
    if kind == "theta_schwarz":
        return builders.theta_schwarz_builder(
            op=context.op,
            block=int(dd_config.block_theta),
            overlap=int(dd_config.overlap_theta),
            reduce_full=reduce_full,
            expand_reduced=expand_reduced,
        )
    if kind == "zeta_dd":
        return builders.zeta_dd_builder(
            op=context.op,
            block=int(dd_config.block_zeta),
            reduce_full=reduce_full,
            expand_reduced=expand_reduced,
        )
    if kind == "zeta_schwarz":
        return builders.zeta_schwarz_builder(
            op=context.op,
            block=int(dd_config.block_zeta),
            overlap=int(dd_config.overlap_zeta),
            reduce_full=reduce_full,
            expand_reduced=expand_reduced,
        )
    if kind == "tzfft":
        return builders.tzfft_builder(op=context.op, reduce_full=reduce_full, expand_reduced=expand_reduced)
    if kind == "fp_tzfft":
        # Older builder bundles do not know this experimental preconditioner.
        fp_tzfft_builder = getattr(builders, "fp_tzfft_builder", None)
        if fp_tzfft_builder is None:
            return builders.tzfft_builder(op=context.op, reduce_full=reduce_full, expand_reduced=expand_reduced)
        return fp_tzfft_builder(op=context.op, reduce_full=reduce_full, expand_reduced=expand_reduced)
    if kind == "fp_tzfft_line":
        fp_tzfft_line_builder = getattr(builders, "fp_tzfft_line_builder", None)
        if fp_tzfft_line_builder is None:
            fp_tzfft_builder = getattr(builders, "fp_tzfft_builder", None)
            if fp_tzfft_builder is not None:
                return fp_tzfft_builder(op=context.op, reduce_full=reduce_full, expand_reduced=expand_reduced)
            return builders.sxblock_builder(op=context.op, reduce_full=reduce_full, expand_reduced=expand_reduced)
        return fp_tzfft_line_builder(op=context.op, reduce_full=reduce_full, expand_reduced=expand_reduced)
    if kind == "fp_tzfft_line_schur":
        fp_tzfft_line_schur_builder = getattr(builders, "fp_tzfft_line_schur_builder", None)
        if fp_tzfft_line_schur_builder is not None:
            return fp_tzfft_line_schur_builder(op=context.op, reduce_full=reduce_full, expand_reduced=expand_reduced)
        fp_tzfft_line_builder = getattr(builders, "fp_tzfft_line_builder", None)
        if fp_tzfft_line_builder is not None:
            return fp_tzfft_line_builder(op=context.op, reduce_full=reduce_full, expand_reduced=expand_reduced)
        fp_tzfft_builder = getattr(builders, "fp_tzfft_builder", None)
        if fp_tzfft_builder is not None:
            return fp_tzfft_builder(op=context.op, reduce_full=reduce_full, expand_reduced=expand_reduced)
        return builders.sxblock_builder(op=context.op, reduce_full=reduce_full, expand_reduced=expand_reduced)
    if kind == "fp_local_geom_line":
        fp_local_geom_line_builder = getattr(builders, "fp_local_geom_line_builder", None)
        if fp_local_geom_line_builder is not None:
            return fp_local_geom_line_builder(op=context.op, reduce_full=reduce_full, expand_reduced=expand_reduced)
        fp_tzfft_line_builder = getattr(builders, "fp_tzfft_line_builder", None)
        if fp_tzfft_line_builder is not None:
            return fp_tzfft_line_builder(op=context.op, reduce_full=reduce_full, expand_reduced=expand_reduced)
        return builders.sxblock_builder(op=context.op, reduce_full=reduce_full, expand_reduced=expand_reduced)
    if kind == "fp_xblock_tz_lu":
        fp_xblock_tz_lu_builder = getattr(builders, "fp_xblock_tz_lu_builder", None)
        if fp_xblock_tz_lu_builder is not None:
            return fp_xblock_tz_lu_builder(
                op=context.op,
                reduce_full=reduce_full,
                expand_reduced=expand_reduced,
            )
        fp_tzfft_line_builder = getattr(builders, "fp_tzfft_line_builder", None)
        if fp_tzfft_line_builder is not None:
            return fp_tzfft_line_builder(op=context.op, reduce_full=reduce_full, expand_reduced=expand_reduced)
        return builders.sxblock_builder(op=context.op, reduce_full=reduce_full, expand_reduced=expand_reduced)
    if kind == "fp_xblock_tz_lu_schur":
        fp_xblock_tz_lu_schur_builder = getattr(builders, "fp_xblock_tz_lu_schur_builder", None)
        if fp_xblock_tz_lu_schur_builder is not None:
            return fp_xblock_tz_lu_schur_builder(
                op=context.op,
                reduce_full=reduce_full,
                expand_reduced=expand_reduced,
            )
        fp_xblock_tz_lu_builder = getattr(builders, "fp_xblock_tz_lu_builder", None)
        if fp_xblock_tz_lu_builder is not None:
            return fp_xblock_tz_lu_builder(op=context.op, reduce_full=reduce_full, expand_reduced=expand_reduced)
        return builders.sxblock_builder(op=context.op, reduce_full=reduce_full, expand_reduced=expand_reduced)
    if kind == "fp_structured_fblock_lu":
        fp_structured_fblock_lu_builder = getattr(builders, "fp_structured_fblock_lu_builder", None)
        if fp_structured_fblock_lu_builder is not None:
            return fp_structured_fblock_lu_builder(
                op=context.op,
                reduce_full=reduce_full,
                expand_reduced=expand_reduced,
            )
        return builders.sxblock_builder(op=context.op, reduce_full=reduce_full, expand_reduced=expand_reduced)
    if kind == "fp_fortran_reduced_lu":
        fp_fortran_reduced_lu_builder = getattr(builders, "fp_fortran_reduced_lu_builder", None)
        if fp_fortran_reduced_lu_builder is not None:
            if bool(context.use_active_dof_mode) and not use_reduced:
                return builders.sxblock_builder(
                    op=context.op,
                    reduce_full=reduce_full,
                    expand_reduced=expand_reduced,
                )
            return fp_fortran_reduced_lu_builder(
                op=context.op,
                reduce_full=reduce_full,
                expand_reduced=expand_reduced,
                active_indices_np=context.active_indices_np if use_reduced else None,
                emit=context.emit,
            )
        return builders.sxblock_builder(op=context.op, reduce_full=reduce_full, expand_reduced=expand_reduced)
    if kind == "fp_direct_active_block_schur":
        fp_direct_active_block_schur_builder = getattr(builders, "fp_direct_active_block_schur_builder", None)
        if fp_direct_active_block_schur_builder is not None:
            if bool(context.use_active_dof_mode) and not use_reduced:
                return builders.sxblock_builder(
                    op=context.op,
                    reduce_full=reduce_full,
                    expand_reduced=expand_reduced,
                )
            return fp_direct_active_block_schur_builder(
                op=context.op,
                reduce_full=reduce_full,
                expand_reduced=expand_reduced,
                active_indices_np=context.active_indices_np if use_reduced else None,
                emit=context.emit,
            )
        return builders.sxblock_builder(op=context.op, reduce_full=reduce_full, expand_reduced=expand_reduced)
    if kind == "sparse_jax":
        precond_dtype = builders.precond_dtype(int(size_est))
        bytes_per = 4.0 if precond_dtype == jnp.float32 else 8.0
        est_mb = (int(size_est) ** 2) * bytes_per / 1.0e6
        if sparse_jax_config.max_mb > 0.0 and est_mb > sparse_jax_config.max_mb:
            if context.emit is not None:
                context.emit(
                    1,
                    "solve_v3_transport_matrix_linear_gmres: sparse_jax preconditioner disabled "
                    f"(est_mem={est_mb:.1f} MB > max_mb={sparse_jax_config.max_mb:.1f})",
                )
            return builders.collision_builder(
                op=context.op,
                reduce_full=reduce_full,
                expand_reduced=expand_reduced,
            )
        cache_suffix = f"sparse_jax_active_{int(size_est)}" if use_reduced else f"sparse_jax_{int(size_est)}"
        cache_key = builders.sparse_jax_cache_key(context.op, cache_suffix)
        if use_reduced:
            def _mv_sparse_reduced(x_reduced: jnp.ndarray, op=context.op) -> jnp.ndarray:
                assert expand_reduced is not None
                assert reduce_full is not None
                y_full = builders.apply_operator_cached(op, expand_reduced(x_reduced))
                return reduce_full(y_full)

            matvec = _mv_sparse_reduced
        else:
            def _mv_sparse_full(x: jnp.ndarray, op=context.op) -> jnp.ndarray:
                return builders.apply_operator_cached(op, x)

            matvec = _mv_sparse_full
        return builders.sparse_jax_builder(
            matvec=matvec,
            n=int(size_est),
            dtype=precond_dtype,
            cache_key=cache_key,
            drop_tol=float(sparse_jax_config.drop_tol),
            drop_rel=float(sparse_jax_config.drop_rel),
            reg=float(sparse_jax_config.reg),
            omega=float(sparse_jax_config.omega),
            sweeps=int(sparse_jax_config.sweeps),
            emit=context.emit,
        )
    if kind in {"sxblock", "block_sx", "species_x"}:
        return builders.sxblock_builder(op=context.op, reduce_full=reduce_full, expand_reduced=expand_reduced)
    if kind in {"block", "block_jacobi"}:
        return builders.block_builder(op=context.op, reduce_full=reduce_full, expand_reduced=expand_reduced)
    return builders.collision_builder(op=context.op, reduce_full=reduce_full, expand_reduced=expand_reduced)


def build_transport_strong_preconditioner_from_kind(
    *,
    kind: str | None,
    use_reduced: bool,
    precond_kind_used: str | None,
    preconditioner_full: Preconditioner | None,
    preconditioner_reduced: Preconditioner | None,
    context: TransportPreconditionerContext,
    builders: TransportPreconditionerDispatchBuilders,
    dd_config: TransportDDConfig,
    sparse_jax_config: TransportSparseJaxConfig,
) -> Preconditioner | None:
    if kind is None:
        return None
    if precond_kind_used is not None and kind == precond_kind_used:
        return preconditioner_reduced if use_reduced else preconditioner_full
    return build_transport_preconditioner_from_kind(
        kind=kind,
        context=context,
        builders=builders,
        dd_config=dd_config,
        sparse_jax_config=sparse_jax_config,
        use_reduced=use_reduced,
    )


@dataclass
class TransportStrongPreconditionerCache:
    """Lazily build full/reduced strong transport preconditioners once per solve."""

    kind: str | None
    precond_kind_used: str | None
    preconditioner_full: Preconditioner | None
    preconditioner_reduced: Preconditioner | None
    context: TransportPreconditionerContext
    builders: TransportPreconditionerDispatchBuilders
    dd_config: TransportDDConfig
    sparse_jax_config: TransportSparseJaxConfig
    strong_full: Preconditioner | None = None
    strong_reduced: Preconditioner | None = None

    def get(self, *, use_reduced: bool) -> Preconditioner | None:
        if self.kind is None:
            return None
        if use_reduced:
            if self.strong_reduced is None:
                self.strong_reduced = build_transport_strong_preconditioner_from_kind(
                    kind=self.kind,
                    use_reduced=True,
                    precond_kind_used=self.precond_kind_used,
                    preconditioner_full=self.preconditioner_full,
                    preconditioner_reduced=self.preconditioner_reduced,
                    context=self.context,
                    builders=self.builders,
                    dd_config=self.dd_config,
                    sparse_jax_config=self.sparse_jax_config,
                )
            return self.strong_reduced
        if self.strong_full is None:
            self.strong_full = build_transport_strong_preconditioner_from_kind(
                kind=self.kind,
                use_reduced=False,
                precond_kind_used=self.precond_kind_used,
                preconditioner_full=self.preconditioner_full,
                preconditioner_reduced=self.preconditioner_reduced,
                context=self.context,
                builders=self.builders,
                dd_config=self.dd_config,
                sparse_jax_config=self.sparse_jax_config,
            )
        return self.strong_full


__all__ = [
    "TransportDDConfig",
    "TransportPreconditionerContext",
    "TransportPreconditionerDispatchBuilders",
    "TransportStrongPreconditionerCache",
    "TransportSparseJaxConfig",
    "auto_transport_preconditioner_choice",
    "build_transport_preconditioner_from_kind",
    "build_transport_strong_preconditioner_from_kind",
    "normalize_transport_preconditioner_kind",
    "resolve_transport_precondition_side_for_kind",
    "resolve_transport_preconditioner_choice",
    "transport_dd_config_from_env",
    "transport_sparse_jax_config_from_env",
]
