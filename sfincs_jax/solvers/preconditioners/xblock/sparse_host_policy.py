"""RHSMode=1 x-block sparse host-assembly policy helpers.

These helpers keep the host sparse x-block assembly decisions independent from
``v3_driver.py`` objects.  The driver supplies operator metadata, while this
module owns the pure policy choices and their tests.
"""

from __future__ import annotations


def _parse_nonnegative_int(value: str, default: int) -> int:
    try:
        return max(0, int(str(value).strip())) if str(value).strip() else max(0, int(default))
    except ValueError:
        return max(0, int(default))


def rhs1_fp_xblock_species_decoupled_for_host_assembly(
    *,
    n_species: int,
    preconditioner_species: int,
) -> bool:
    """Return whether host assembly preserves the requested species coupling.

    ``preconditioner_species=0`` means the full species coupling is requested.
    That is equivalent to species-decoupled assembly only for one-species
    systems; any explicit species-block preconditioner is already decoupled.
    """

    if int(preconditioner_species) != 0:
        return True
    return int(n_species) == 1


def rhs1_xblock_sparse_lu_default_max(
    *,
    has_fp: bool,
    has_pas: bool,
    build_jax_factors: bool,
) -> int:
    """Return the default exact-LU cap for x-block sparse preconditioners.

    Host SuperLU factors on pure full-FP x-blocks have a measured safe window
    above the generic JAX-factor and PAS paths. Keep the larger cap restricted
    to that narrow host case so medium 3D full-FP blocks avoid weak ILU plateaus
    without promoting exact LU for PAS or device-resident factor paths. The
    30k cutoff covers the bounded scale-0.55 QI x-block probe whose largest
    per-x block is 23925 DOFs; larger production blocks remain ILU/opt-in work.
    """

    if (not bool(build_jax_factors)) and bool(has_fp) and not bool(has_pas):
        return 30000
    return 2000


def rhs1_xblock_sparse_host_block_factor_allowed(
    *,
    block_size: int,
    max_block_size_env_value: str,
    default_max_block_size: int = 30000,
) -> bool:
    """Return whether a host x-block local sparse factor should be attempted.

    Production QI/FP grids can contain very large per-x blocks. Sparse ILU on
    those blocks may spend the entire runtime budget failing singular pivot
    attempts. A positive cap keeps high-resolution rescue bounded; ``0`` keeps
    the historical no-cap behavior for explicit experiments.
    """

    max_block = _parse_nonnegative_int(max_block_size_env_value, default_max_block_size)
    return bool(max_block == 0 or int(block_size) <= int(max_block))


__all__ = [
    "rhs1_fp_xblock_species_decoupled_for_host_assembly",
    "rhs1_xblock_sparse_host_block_factor_allowed",
    "rhs1_xblock_sparse_lu_default_max",
]
