"""RHSMode=1 x-block sparse host-assembly policy helpers.

These helpers keep the host sparse x-block assembly decisions independent from
``v3_driver.py`` objects.  The driver supplies operator metadata, while this
module owns the pure policy choices and their tests.
"""

from __future__ import annotations


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
    without promoting exact LU for PAS or device-resident factor paths.
    """

    if (not bool(build_jax_factors)) and bool(has_fp) and not bool(has_pas):
        return 20000
    return 2000


__all__ = [
    "rhs1_fp_xblock_species_decoupled_for_host_assembly",
    "rhs1_xblock_sparse_lu_default_max",
]
