"""RHSMode=1 automatic preconditioner policy helpers.

These helpers encode small routing decisions used by ``v3_driver.py`` before a
preconditioner builder is called. They are kept free of operator construction so
the policy can be tested directly without running full SFINCS solves.
"""

from __future__ import annotations

import os


PAS_AUTO_STRONG_BASE_KINDS = frozenset(
    {
        "schur",
        "xblock_tz",
        "xblock_tz_lmax",
        "sxblock_tz",
        "species_block",
        "theta_zeta",
        "pas_lite",
        "pas_hybrid",
        "pas_schur",
        "pas_tz",
        "pas_tokamak_theta",
    }
)

_RHS1_PRECONDITIONER_KIND_ALIASES = {
    "0": None,
    "false": None,
    "no": None,
    "off": None,
    "theta": "theta_line",
    "theta_line": "theta_line",
    "line_theta": "theta_line",
    "theta_dd": "theta_dd",
    "theta_block": "theta_dd",
    "dd_theta": "theta_dd",
    "dd_t": "theta_dd",
    "theta_schwarz": "theta_schwarz",
    "schwarz_theta": "theta_schwarz",
    "ras_theta": "theta_schwarz",
    "theta_ras": "theta_schwarz",
    "theta_line_xdiag": "theta_line_xdiag",
    "theta_xdiag": "theta_line_xdiag",
    "theta_line_diagx": "theta_line_xdiag",
    "xdiag": "point_xdiag",
    "point_xdiag": "point_xdiag",
    "block_xdiag": "point_xdiag",
    "species": "species_block",
    "species_block": "species_block",
    "speciesblock": "species_block",
    "sxblock": "sxblock",
    "species_xblock": "sxblock",
    "species_x": "sxblock",
    "sxblock_tz": "sxblock_tz",
    "sxblock_theta_zeta": "sxblock_tz",
    "species_xblock_tz": "sxblock_tz",
    "sx_tz": "sxblock_tz",
    "xblock_tz_lmax": "xblock_tz_lmax",
    "xblock_tz_trunc": "xblock_tz_lmax",
    "xblock_tz_cut": "xblock_tz_lmax",
    "xblock_tz": "xblock_tz",
    "xblock": "xblock_tz",
    "x_tz": "xblock_tz",
    "xtz": "xblock_tz",
    "xblock_theta_zeta": "xblock_tz",
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
    "pas_tz": "pas_tz",
    "pas_3d": "pas_tz",
    "pas_tz_l": "pas_tz",
    "pas_ilu": "pas_ilu",
    "pas_block_ilu": "pas_ilu",
    "pas_xblock_ilu": "pas_ilu",
    "block_ilu": "pas_ilu",
    "theta_zeta": "theta_zeta",
    "theta_zeta_line": "theta_zeta",
    "tz": "theta_zeta",
    "tz_line": "theta_zeta",
    "zeta": "zeta_line",
    "zeta_line": "zeta_line",
    "line_zeta": "zeta_line",
    "zeta_dd": "zeta_dd",
    "zeta_block": "zeta_dd",
    "dd_zeta": "zeta_dd",
    "dd_z": "zeta_dd",
    "zeta_schwarz": "zeta_schwarz",
    "schwarz_zeta": "zeta_schwarz",
    "ras_zeta": "zeta_schwarz",
    "zeta_ras": "zeta_schwarz",
    "adi": "adi",
    "adi_line": "adi",
    "line_adi": "adi",
    "zeta_theta": "adi",
    "1": "point",
    "true": "point",
    "yes": "point",
    "on": "point",
    "point": "point",
    "point_block": "point",
    "schur": "schur",
    "schur_complement": "schur",
    "constraint_schur": "schur",
    "collision": "collision",
    "diag": "collision",
    "collision_diag": "collision",
}


def _env_int(name: str, default: int) -> int:
    env = os.environ.get(name, "").strip()
    try:
        return int(env) if env else int(default)
    except ValueError:
        return int(default)


def _env_float(name: str, default: float) -> float:
    env = os.environ.get(name, "").strip()
    try:
        return float(env) if env else float(default)
    except ValueError:
        return float(default)


def canonical_rhs1_preconditioner_kind(raw: str | None) -> str | None:
    """Canonicalize ``SFINCS_JAX_RHSMODE1_PRECONDITIONER`` aliases.

    Unknown non-empty aliases intentionally return ``None`` to preserve the
    historical driver behavior for unrecognized values.
    """
    key = str(raw or "").strip().lower()
    if not key:
        return None
    return _RHS1_PRECONDITIONER_KIND_ALIASES.get(key)


def rhs1_pas_auto_large_base_kind(*, active_size: int) -> str:
    """Keep large auto-selected PAS solves in the PAS-native preconditioner family."""
    pas_lite_min = _env_int("SFINCS_JAX_PAS_LITE_MIN", 20000)
    if int(active_size) >= max(1, int(pas_lite_min)):
        return "pas_lite"
    return "pas_hybrid"


def pas_auto_skip_strong_retry(
    *,
    has_pas: bool,
    strong_precond_env: str,
    rhs1_precond_kind: str | None,
    residual_norm: float,
    target: float,
    ratio: float,
) -> bool:
    """Skip PAS strong retry when the current strong base already met the relaxed target."""
    if not has_pas or ratio <= 0.0:
        return False
    if strong_precond_env not in {"", "auto"}:
        return False
    if rhs1_precond_kind not in PAS_AUTO_STRONG_BASE_KINDS:
        return False
    return float(residual_norm) <= float(target) * float(ratio)


def rhs1_pas_dkes_xblock_allowed(
    *,
    has_pas: bool,
    use_dkes: bool,
    backend: str,
    n_theta: int,
    n_zeta: int,
    max_l: int,
    xblock_tz_limit: int,
) -> bool:
    """Return whether bounded PAS DKES runs may use dense xblock_tz preconditioning."""
    if not has_pas or not use_dkes:
        return False
    backend_norm = str(backend).strip().lower()
    if backend_norm not in {"cpu", "gpu", "tpu"}:
        return False
    if int(n_theta) <= 1:
        return False
    if int(xblock_tz_limit) <= 0:
        return False
    return int(max_l) * int(n_theta) * int(n_zeta) <= int(xblock_tz_limit)


def rhs1_pas_tokamak_gpu_theta_allowed(
    *,
    has_pas: bool,
    has_fp: bool,
    backend: str,
    tokamak_like: bool,
    active_size: int,
    er_abs: float,
    schur_er_min: float,
    has_magdrift: bool,
    has_collisionless: bool,
) -> bool:
    """Return whether the bounded GPU tokamak PAS theta/L path is eligible."""
    if not has_pas or has_fp:
        return False
    if str(backend).strip().lower() == "cpu":
        return False
    if not tokamak_like or not has_collisionless:
        return False
    if float(er_abs) <= float(schur_er_min):
        return False
    if has_magdrift:
        return False
    theta_max = _env_int("SFINCS_JAX_RHSMODE1_PAS_TOKAMAK_GPU_THETA_MAX", 8000)
    return int(active_size) <= max(1, int(theta_max))


def rhs1_pas_tokamak_gpu_xblock_preferred(
    *,
    has_pas: bool,
    has_fp: bool,
    backend: str,
    tokamak_like: bool,
    active_size: int,
    er_abs: float,
    schur_er_min: float,
    has_magdrift: bool,
    has_collisionless: bool,
    n_theta: int,
    n_zeta: int,
    max_l: int,
    xblock_tz_limit: int,
) -> bool:
    """Prefer xblock_tz over theta/L for bounded GPU tokamak PAS+Er branches."""
    if not rhs1_pas_tokamak_gpu_theta_allowed(
        has_pas=has_pas,
        has_fp=has_fp,
        backend=backend,
        tokamak_like=tokamak_like,
        active_size=active_size,
        er_abs=er_abs,
        schur_er_min=schur_er_min,
        has_magdrift=has_magdrift,
        has_collisionless=has_collisionless,
    ):
        return False
    if int(n_theta) <= 1 or int(xblock_tz_limit) <= 0:
        return False
    prefer_max = _env_int("SFINCS_JAX_RHSMODE1_PAS_TOKAMAK_GPU_XBLOCK_ACTIVE_MAX", 12000)
    if int(active_size) > max(1, int(prefer_max)):
        return False
    return int(max_l) * int(n_theta) * int(n_zeta) <= int(xblock_tz_limit)


def rhs1_pas_tokamak_cpu_xblock_preferred(
    *,
    has_pas: bool,
    has_fp: bool,
    backend: str,
    tokamak_like: bool,
    active_size: int,
    er_abs: float,
    schur_er_min: float,
    has_magdrift: bool,
    has_collisionless: bool,
    n_theta: int,
    n_zeta: int,
    max_l: int,
    xblock_tz_limit: int,
) -> bool:
    """Prefer xblock_tz for bounded CPU tokamak PAS+Er branches before pas_schur."""
    if not has_pas or has_fp:
        return False
    if str(backend).strip().lower() != "cpu":
        return False
    if not tokamak_like or not has_collisionless:
        return False
    if float(er_abs) <= float(schur_er_min) and (not has_magdrift):
        return False
    if int(n_theta) <= 1 or int(xblock_tz_limit) <= 0:
        return False
    prefer_max = _env_int("SFINCS_JAX_RHSMODE1_PAS_TOKAMAK_CPU_XBLOCK_ACTIVE_MAX", 4000)
    if int(active_size) > max(1, int(prefer_max)):
        return False
    return int(max_l) * int(n_theta) * int(n_zeta) <= int(xblock_tz_limit)


def rhs1_gpu_sparse_fallback_skip_allowed(
    *,
    backend: str,
    rhs_mode: int,
    include_phi1: bool,
    has_pas: bool,
    rhs1_precond_kind: str | None,
    use_active_dof_mode: bool,
    residual_norm: float,
    target: float,
) -> bool:
    """Return whether a GPU PAS sparse fallback can be skipped after Schur acceptance."""
    if str(backend).strip().lower() == "cpu":
        return False
    if not bool(use_active_dof_mode):
        return False
    if int(rhs_mode) != 1 or bool(include_phi1):
        return False
    if not has_pas:
        return False
    if str(rhs1_precond_kind or "").strip().lower() not in {"schur", "pas_schur"}:
        return False
    skip_ratio = _env_float("SFINCS_JAX_RHSMODE1_GPU_SPARSE_SKIP_RATIO", 10.0)
    if skip_ratio <= 0.0:
        return False
    return float(residual_norm) <= float(skip_ratio) * max(float(target), 1.0e-300)


def rhs1_sharded_line_override_allowed(rhs1_precond_kind: str | None) -> bool:
    """Return whether sharded auto-selection may demote the current preconditioner to line DD."""
    return rhs1_precond_kind in {
        None,
        "point",
        "point_xdiag",
        "theta_line",
        "theta_line_xdiag",
        "zeta_line",
        "xmg",
        "collision",
        "pas_lite",
        "pas_hybrid",
    }


__all__ = [
    "PAS_AUTO_STRONG_BASE_KINDS",
    "canonical_rhs1_preconditioner_kind",
    "pas_auto_skip_strong_retry",
    "rhs1_gpu_sparse_fallback_skip_allowed",
    "rhs1_pas_auto_large_base_kind",
    "rhs1_pas_dkes_xblock_allowed",
    "rhs1_pas_tokamak_cpu_xblock_preferred",
    "rhs1_pas_tokamak_gpu_theta_allowed",
    "rhs1_pas_tokamak_gpu_xblock_preferred",
    "rhs1_sharded_line_override_allowed",
]
