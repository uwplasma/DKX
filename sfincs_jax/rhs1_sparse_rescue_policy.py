from __future__ import annotations

"""Policy helpers for RHSMode=1 sparse-rescue ordering and skip decisions."""

from dataclasses import dataclass


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
        enabled = bool(has_fp) or (bool(has_pas) and float(residual_norm) > float(target))
    if enabled:
        enabled = int(rhs_mode) == 1 and (not bool(include_phi1))
    return bool(enabled)


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

    if enabled and str(kind_use) == "jax" and sparse_jax_est_mb is not None:
        if float(sparse_jax_max_mb) > 0.0 and float(sparse_jax_est_mb) > float(sparse_jax_max_mb):
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
        prefer_sparse_exact_over_dense_shortcut=bool(prefer_sparse_exact_over_dense_shortcut),
        reason_dense_shortcut_skip=bool(reason_dense_shortcut_skip),
        reason_size_disabled=bool(reason_size_disabled),
        reason_size_large_cpu=bool(reason_size_large_cpu),
        reason_size_exact_direct=bool(reason_size_exact_direct),
        reason_size_targeted=bool(reason_size_targeted),
        reason_sparse_jax_mem_disabled=bool(reason_sparse_jax_mem_disabled),
        reason_large_cpu_exact_skips_targeted=bool(reason_large_cpu_exact_skips_targeted),
        reason_pas_fast_accept=bool(reason_pas_fast_accept),
        reason_gpu_sparse_skip=bool(reason_gpu_sparse_skip),
    )


__all__ = [
    "RHS1SparseRescueOrdering",
    "rhs1_resolved_sparse_rescue_ordering",
    "rhs1_sparse_enabled_initial",
    "rhs1_sparse_kind_use",
]
