"""Helpers for RHSMode=1 strong-preconditioner fallback builds.

These helpers sit one layer below the main solve orchestration. They keep the
policy adjustment and preconditioner construction for strong fallback retries in
one place while reusing the shared RHSMode=1 dispatch ladder.
"""

from __future__ import annotations

from collections.abc import Callable
import os
from typing import Any


_REDUCED_STRONG_KINDS = frozenset(
    {
        "theta_line",
        "theta_schwarz",
        "theta_line_xdiag",
        "species_block",
        "sxblock",
        "sxblock_tz",
        "theta_zeta",
        "xmg",
        "pas_lite",
        "pas_hybrid",
        "xblock_tz",
        "xblock_tz_lmax",
        "zeta_line",
        "zeta_schwarz",
        "schur",
        "adi",
    }
)


def _parse_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        return int(raw) if raw else int(default)
    except ValueError:
        return int(default)


def _reduced_build_kind(kind: str | None) -> str | None:
    if kind is None:
        return None
    return str(kind) if kind in _REDUCED_STRONG_KINDS else "adi"


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


def build_rhs1_strong_preconditioner_reduced_from_kind(
    *,
    op: Any,
    strong_precond_kind: str | None,
    reduce_full: Callable | None,
    expand_reduced: Callable | None,
    rhs1_xblock_tz_lmax: int | None,
    dd_block_theta: int,
    dd_overlap_theta: int,
    dd_block_zeta: int,
    dd_overlap_zeta: int,
    dispatch_builder: Callable[..., Callable],
) -> Callable | None:
    """Build the reduced active-DOF strong fallback through shared dispatch."""

    effective_kind = _reduced_build_kind(strong_precond_kind)
    if effective_kind is None:
        return None
    lmax_use = int(rhs1_xblock_tz_lmax or 0)
    if effective_kind == "xblock_tz_lmax" and lmax_use <= 0:
        lmax_use = _parse_env_int("SFINCS_JAX_RHSMODE1_XBLOCK_TZ_LMAX", 0)
    return dispatch_builder(
        op=op,
        rhs1_precond_kind=effective_kind,
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
        rhs1_xblock_tz_lmax=int(lmax_use),
        dd_block_theta=int(dd_block_theta),
        dd_overlap_theta=int(dd_overlap_theta),
        dd_block_zeta=int(dd_block_zeta),
        dd_overlap_zeta=int(dd_overlap_zeta),
        adi_sweeps=max(1, _parse_env_int("SFINCS_JAX_RHSMODE1_ADI_SWEEPS", 2)),
    )


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
    "build_rhs1_strong_preconditioner_reduced_from_kind",
    "resolve_rhs1_strong_preconditioner_kind_for_build",
]
