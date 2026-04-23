"""RHSMode=1 Schur preconditioner base-selection policy.

The Schur preconditioner itself lives in ``v3_driver.py`` because it closes over
many numerical builders. This module keeps the routing policy separate so the
PAS, DKES, and geometry-size rules can be tested without constructing full
preconditioner factors.
"""

from __future__ import annotations

from collections.abc import Sequence
import os

import numpy as np


_SCHUR_BASE_ALIASES: dict[str, str] = {
    "theta": "theta_line",
    "theta_line": "theta_line",
    "line_theta": "theta_line",
    "theta_dd": "theta_dd",
    "theta_block": "theta_dd",
    "dd_theta": "theta_dd",
    "dd_t": "theta_dd",
    "zeta": "zeta_line",
    "zeta_line": "zeta_line",
    "line_zeta": "zeta_line",
    "zeta_dd": "zeta_dd",
    "zeta_block": "zeta_dd",
    "dd_zeta": "zeta_dd",
    "dd_z": "zeta_dd",
    "adi": "adi",
    "adi_line": "adi",
    "theta_zeta": "adi",
    "zeta_theta": "adi",
    "species": "species_block",
    "species_block": "species_block",
    "speciesblock": "species_block",
    "sxblock_tz": "sxblock_tz",
    "sxblock_theta_zeta": "sxblock_tz",
    "species_xblock_tz": "sxblock_tz",
    "sx_tz": "sxblock_tz",
    "xblock_tz": "xblock_tz",
    "xblock": "xblock_tz",
    "x_tz": "xblock_tz",
    "xtz": "xblock_tz",
    "xblock_theta_zeta": "xblock_tz",
    "xblock_tz_lmax": "xblock_tz_lmax",
    "xblock_lmax": "xblock_tz_lmax",
    "xtz_lmax": "xblock_tz_lmax",
    "xblock_theta_zeta_lmax": "xblock_tz_lmax",
    "xmg": "xmg",
    "multigrid": "xmg",
    "x_coarse": "xmg",
    "coarse_x": "xmg",
    "pas_lite": "pas_lite",
    "pas_light": "pas_lite",
    "pas_xmg": "pas_lite",
    "pas_xmg_lite": "pas_lite",
    "pas_hybrid": "pas_hybrid",
    "pas_xline_xcoarse": "pas_hybrid",
    "pas_line_xcoarse": "pas_hybrid",
    "pas_xcoarse_line": "pas_hybrid",
    "pas_schur": "pas_schur",
    "pas_block_schur": "pas_schur",
    "pas_xmg_l": "pas_schur",
    "pas_ilu": "pas_ilu",
    "pas_block_ilu": "pas_ilu",
    "pas_xblock_ilu": "pas_ilu",
    "block_ilu": "pas_ilu",
    "pas_tz": "pas_tz",
    "pas_theta_zeta": "pas_tz",
    "pas_tz_l": "pas_tz",
    "pas_l_tz": "pas_tz",
    "tz_l": "pas_tz",
    "tz_lblock": "pas_tz",
    "pas_tokamak_theta": "pas_tokamak_theta",
    "pas_tokamak": "pas_tokamak_theta",
    "pas_theta": "pas_tokamak_theta",
    "tokamak_theta": "pas_tokamak_theta",
    "theta_tokamak": "pas_tokamak_theta",
    "point": "point",
    "block": "point",
    "jacobi": "point",
}


def canonical_schur_base_kind(base_kind_env: str | None) -> str | None:
    """Return the canonical Schur-base kind for an explicit environment value."""
    key = str(base_kind_env or "").strip().lower()
    if not key:
        return None
    return _SCHUR_BASE_ALIASES.get(key)


def _env_int(name: str, default: int) -> int:
    env = os.environ.get(name, "").strip()
    try:
        return int(env) if env else int(default)
    except ValueError:
        return int(default)


def resolve_rhs1_schur_base_kind(
    *,
    base_kind_env: str | None,
    n_theta: int,
    n_zeta: int,
    n_species: int,
    total_size: int,
    nxi_for_x: Sequence[int],
    has_pas: bool,
    has_fp: bool,
    has_er_xdot: bool,
    has_er_xidot: bool,
    use_dkes_exb: bool,
    pas_tokamak_theta_applicable: bool,
    pas_tz_applicable: bool,
    geom_scheme: int | None = None,
) -> str:
    """Resolve the base preconditioner used inside RHSMode=1 Schur.

    Explicit ``SFINCS_JAX_RHSMODE1_SCHUR_BASE`` aliases win. Otherwise the policy
    mirrors the historical v3-driver ordering: large PAS+Er uses x-coarse,
    DKES PAS prefers dense x-blocks only while bounded, tokamak PAS uses the
    dedicated theta/L base, and large 3D PAS uses the structured PAS-TZ base.
    """
    explicit = canonical_schur_base_kind(base_kind_env)
    if explicit is not None:
        return explicit

    n_theta_i = int(n_theta)
    n_zeta_i = int(n_zeta)
    n_species_i = int(n_species)
    total_size_i = int(total_size)
    geom_scheme_i = int(geom_scheme or 0)
    nxi = np.asarray(nxi_for_x, dtype=np.int64)
    max_l = int(np.max(nxi)) if nxi.size else 0
    local_per_species = int(np.sum(nxi))
    dke_size = int(local_per_species * n_theta_i * n_zeta_i)

    if bool(has_pas) and bool(has_er_xdot) and (not bool(has_fp)):
        xmg_min = _env_int("SFINCS_JAX_RHSMODE1_PAS_XMG_MIN", 50000)
        if total_size_i >= max(1, int(xmg_min)):
            return "xmg"

    if n_theta_i <= 1 and n_zeta_i <= 1:
        return "point"

    tz_max = _env_int("SFINCS_JAX_RHSMODE1_TZ_PRECOND_MAX", 128)
    default_xblock_tz_max = 2000 if bool(has_pas) else 1200
    xblock_tz_max = _env_int("SFINCS_JAX_RHSMODE1_XBLOCK_TZ_MAX", default_xblock_tz_max)
    block_size = int(max_l * n_theta_i * n_zeta_i)
    pas_tz_min = _env_int("SFINCS_JAX_RHSMODE1_PAS_TZ_MIN", 800)

    if bool(has_pas) and (not bool(has_fp)) and bool(use_dkes_exb) and (not bool(has_er_xdot)) and (not bool(has_er_xidot)):
        dense_bytes_max = max(0, _env_int("SFINCS_JAX_RHSMODE1_DKES_XBLOCK_TZ_MAX_BYTES", 512 * 1024 * 1024))
        block_sizes_x = (nxi.astype(np.int64, copy=False) * n_theta_i * n_zeta_i).astype(np.int64, copy=False)
        dense_bytes = int(n_species_i * int(np.sum(block_sizes_x * block_sizes_x)) * 8)
        xblock_tz_small_default = max(0, int(xblock_tz_max))
        xblock_tz_small_max = max(
            0,
            _env_int("SFINCS_JAX_RHSMODE1_DKES_XBLOCK_TZ_SMALL_MAX", xblock_tz_small_default),
        )
        if (
            xblock_tz_small_max > 0
            and xblock_tz_max > 0
            and block_size <= xblock_tz_max
            and block_size <= xblock_tz_small_max
            and dense_bytes <= dense_bytes_max
        ):
            return "xblock_tz"
        return "pas_ilu"

    if bool(has_pas) and bool(pas_tokamak_theta_applicable):
        return "pas_tokamak_theta"

    if bool(has_pas) and (not bool(has_fp)) and bool(pas_tz_applicable) and block_size >= pas_tz_min:
        return "pas_tz"

    species_block_max = _env_int("SFINCS_JAX_RHSMODE1_SPECIES_BLOCK_MAX", 1600)
    if (
        bool(has_pas)
        and n_theta_i > 1
        and n_zeta_i > 1
        and species_block_max > 0
        and dke_size <= species_block_max
    ):
        return "species_block"

    if bool(has_pas) and n_theta_i > 1 and xblock_tz_max > 0 and block_size <= xblock_tz_max:
        return "xblock_tz"

    if bool(has_pas) and n_theta_i > 1 and n_zeta_i > 1 and n_theta_i * n_zeta_i <= tz_max:
        return "theta_zeta"

    if bool(has_pas) and (geom_scheme_i == 1 or n_zeta_i <= 5):
        return "pas_schur"

    return "theta_line" if n_theta_i >= n_zeta_i else "zeta_line"


__all__ = [
    "canonical_schur_base_kind",
    "resolve_rhs1_schur_base_kind",
]
