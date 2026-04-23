"""Helpers for RHSMode=1 strong-preconditioner fallback builds.

These helpers sit one layer below the main solve orchestration. They keep the
policy adjustment and preconditioner construction for strong fallback retries in
one place while reusing the shared RHSMode=1 dispatch ladder.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def resolve_rhs1_strong_preconditioner_kind_for_build(
    strong_precond_kind: str | None,
    *,
    has_pas: bool,
    base_preconditioner_kind: str | None,
    residual_norm: float,
) -> str | None:
    """Adjust the requested strong-preconditioner kind before building it."""
    if (
        strong_precond_kind == "schur"
        and has_pas
        and base_preconditioner_kind in {"pas_lite", "pas_hybrid"}
        and residual_norm == residual_norm
    ):
        return "pas_hybrid"
    return strong_precond_kind


def build_rhs1_strong_preconditioner_full_from_kind(
    *,
    op: Any,
    strong_precond_kind: str | None,
    base_preconditioner_kind: str | None,
    residual_norm: float,
    rhs1_xblock_tz_lmax: int | None,
    dd_block_theta: int,
    dd_overlap_theta: int,
    dd_block_zeta: int,
    dd_overlap_zeta: int,
    adi_sweeps: int,
    dispatch_builder: Callable[..., Callable],
) -> tuple[str | None, Callable | None]:
    """Build the full-system strong fallback preconditioner via shared dispatch."""
    effective_kind = resolve_rhs1_strong_preconditioner_kind_for_build(
        strong_precond_kind,
        has_pas=getattr(getattr(op, "fblock", None), "pas", None) is not None,
        base_preconditioner_kind=base_preconditioner_kind,
        residual_norm=float(residual_norm),
    )
    if effective_kind is None:
        return None, None
    preconditioner = dispatch_builder(
        op=op,
        rhs1_precond_kind=effective_kind,
        rhs1_xblock_tz_lmax=rhs1_xblock_tz_lmax,
        dd_block_theta=dd_block_theta,
        dd_overlap_theta=dd_overlap_theta,
        dd_block_zeta=dd_block_zeta,
        dd_overlap_zeta=dd_overlap_zeta,
        adi_sweeps=adi_sweeps,
    )
    return effective_kind, preconditioner


__all__ = [
    "build_rhs1_strong_preconditioner_full_from_kind",
    "resolve_rhs1_strong_preconditioner_kind_for_build",
]
