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
        "fp_streaming_fft",
        "streaming_fft",
        "stream_fft",
        "sparse_jax",
    }:
        if env in {"streaming_fft", "stream_fft"}:
            return "tzfft"
        if env in {"fp_streaming_fft"}:
            return "fp_tzfft"
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
        if n_block <= sxblock_max:
            precond_kind = "sxblock"
        elif int(op.total_size) <= block_max and str(default_solver_kind) != "bicgstab":
            precond_kind = "sxblock"
        else:
            precond_kind = "collision"
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


__all__ = [
    "TransportDDConfig",
    "TransportPreconditionerContext",
    "TransportPreconditionerDispatchBuilders",
    "TransportSparseJaxConfig",
    "auto_transport_preconditioner_choice",
    "build_transport_preconditioner_from_kind",
    "build_transport_strong_preconditioner_from_kind",
    "normalize_transport_preconditioner_kind",
    "resolve_transport_preconditioner_choice",
    "transport_dd_config_from_env",
    "transport_sparse_jax_config_from_env",
]
