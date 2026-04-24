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

PAS_WEAK_AUTO_OVERRIDE_KINDS = frozenset(
    {
        None,
        "collision",
        "point",
        "xmg",
        "theta_line",
        "zeta_line",
        "theta_zeta",
        "xblock_tz",
        "xblock_tz_lmax",
        "theta_line_xdiag",
    }
)

FP_FORCE_XMG_WEAK_KINDS = frozenset(
    {
        None,
        "collision",
        "point",
        "theta_line",
        "zeta_line",
        "theta_schwarz",
        "zeta_schwarz",
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


def rhs1_pas_weak_auto_override_kind(
    *,
    rhs1_precond_env: str,
    rhs_mode: int,
    include_phi1: bool,
    has_pas: bool,
    current_kind: str | None,
    active_size: int,
    n_theta: int,
    n_zeta: int,
    max_l: int,
) -> str | None:
    """Promote weak default PAS preconditioners to PAS-aware defaults.

    This mirrors the driver auto-policy used before expensive PAS fallback
    attempts: small angular blocks may use ``xblock_tz``; larger systems stay in
    the PAS-native lite/hybrid family.
    """
    if str(rhs1_precond_env or "").strip().lower():
        return current_kind
    if int(rhs_mode) != 1 or bool(include_phi1) or not has_pas:
        return current_kind
    if current_kind not in PAS_WEAK_AUTO_OVERRIDE_KINDS:
        return current_kind

    xblock_tz_max = _env_int("SFINCS_JAX_RHSMODE1_XBLOCK_TZ_MAX", 1200)
    xblock_small_max = _env_int("SFINCS_JAX_RHSMODE1_XBLOCK_TZ_SMALL_MAX", 4000)
    if (
        int(active_size) <= max(1, int(xblock_small_max))
        and int(xblock_tz_max) > 0
        and int(max_l) * int(n_theta) * int(n_zeta) <= int(xblock_tz_max)
    ):
        return "xblock_tz"
    return rhs1_pas_auto_large_base_kind(active_size=int(active_size))


def rhs1_pas_family_refinement_kind(
    *,
    rhs1_precond_env: str,
    has_pas: bool,
    has_fp: bool,
    current_kind: str | None,
    active_size: int,
    n_zeta: int,
    geom_scheme: int,
    pas_tz_applicable: bool,
    pas_tokamak_theta_applicable: bool,
) -> str | None:
    """Refine automatic PAS lite/hybrid selections to specialized PAS builders."""
    result = current_kind
    env = str(rhs1_precond_env or "").strip().lower()
    tokamak_geometry = int(n_zeta) == 1 or int(geom_scheme) == 1
    tokamak_like = int(geom_scheme) == 1 or int(n_zeta) <= 5

    if result == "pas_lite" and has_pas and tokamak_geometry:
        # GeometryScheme=1 tokamak PAS runs need stronger angular/L coupling
        # than pas_lite provides, but can still avoid the most expensive global
        # blocks by staying in the hybrid family.
        result = "pas_hybrid"
    if env in {"", "auto", "default"} and result in {"pas_lite", "pas_hybrid"} and pas_tokamak_theta_applicable:
        return "pas_tokamak_theta"
    if (
        env in {"", "auto", "default"}
        and result in {"pas_lite", "pas_hybrid"}
        and pas_tz_applicable
        and (not pas_tokamak_theta_applicable)
    ):
        return "pas_tz"
    if tokamak_like and result in {"pas_lite", "pas_hybrid"} and pas_tz_applicable:
        return "pas_tz"
    if env == "" and has_pas and tokamak_like and result in {"pas_lite", "pas_hybrid"}:
        pas_ilu_min = _env_int("SFINCS_JAX_RHSMODE1_PAS_ILU_MIN", 12000)
        if int(active_size) >= max(1, int(pas_ilu_min)):
            return "pas_ilu"
    return result


def rhs1_fp_dkes_env_preconditioner_kind(
    *,
    rhs1_precond_env: str,
    rhs_mode: int,
    include_phi1: bool,
    has_fp: bool,
    use_dkes: bool,
    total_size: int,
    n_theta: int,
    n_zeta: int,
    max_l: int,
) -> str:
    """Return an early env override for bounded FP DKES xblock_tz solves."""
    env = str(rhs1_precond_env or "").strip().lower()
    if env:
        return env
    if int(rhs_mode) != 1 or bool(include_phi1) or (not has_fp) or (not use_dkes):
        return env

    fp_dkes_max = _env_int("SFINCS_JAX_RHSMODE1_FP_DKES_STRONG_MAX", 20000)
    if int(total_size) > max(1, int(fp_dkes_max)):
        return env

    xblock_tz_max = _env_int("SFINCS_JAX_RHSMODE1_XBLOCK_TZ_MAX", 1200)
    if (
        int(n_theta) > 1
        and int(xblock_tz_max) > 0
        and int(max_l) * int(n_theta) * int(n_zeta) <= int(xblock_tz_max)
    ):
        return "xblock_tz"
    return env


def rhs1_fp_dkes_default_kind(
    *,
    active_size: int,
    n_theta: int,
    n_zeta: int,
    max_l: int,
    xblock_tz_limit: int,
) -> str:
    """Select the default RHSMode=1 preconditioner for FP DKES trajectory cases."""
    fp_dkes_strong_max = _env_int("SFINCS_JAX_RHSMODE1_FP_DKES_STRONG_MAX", 20000)
    if int(active_size) > max(1, int(fp_dkes_strong_max)):
        return "collision"
    if (
        int(n_theta) > 1
        and int(xblock_tz_limit) > 0
        and int(max_l) * int(n_theta) * int(n_zeta) <= int(xblock_tz_limit)
    ):
        return "xblock_tz"
    return "xmg"


def rhs1_large_fp_near_zero_er_override_kind(
    *,
    rhs1_precond_env: str,
    rhs_mode: int,
    include_phi1: bool,
    has_fp: bool,
    has_pas: bool,
    current_kind: str | None,
    total_size: int,
    er_abs: float,
    schur_er_min: float,
) -> str | None:
    """Force large near-zero-Er FP-only systems from weak line/point blocks to xmg."""
    if str(rhs1_precond_env or "").strip().lower():
        return current_kind
    if int(rhs_mode) != 1 or bool(include_phi1) or (not has_fp) or has_pas:
        return current_kind
    if float(er_abs) > float(schur_er_min):
        return current_kind
    if current_kind not in FP_FORCE_XMG_WEAK_KINDS:
        return current_kind

    fp_force_xmg_min = _env_int("SFINCS_JAX_RHSMODE1_FP_FORCE_XMG_MIN", 120000)
    if int(total_size) >= max(1, int(fp_force_xmg_min)):
        return "xmg"
    return current_kind


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


def rhs1_pas_dkes_cpu_pas_tz_preferred(
    *,
    has_pas: bool,
    use_dkes: bool,
    backend: str,
    n_theta: int,
    n_zeta: int,
    max_l: int,
    active_size: int,
) -> bool:
    """Return whether CPU PAS DKES auto-selection should prefer ``pas_tz``.

    Dense x-blocks are robust for small DKES angular blocks, but on the HSX
    DKES benchmark the structured PAS angular block is both faster and much
    lower-memory once the block reaches O(10^3) DOFs. Keep the rule CPU-only
    until a matching GPU measurement supports changing that lane.
    """
    if not has_pas or not use_dkes:
        return False
    if str(backend).strip().lower() != "cpu":
        return False
    if int(n_theta) <= 1 or int(n_zeta) <= 1:
        return False
    min_block = _env_int("SFINCS_JAX_RHSMODE1_PAS_DKES_CPU_PAS_TZ_MIN", 950)
    max_active = _env_int("SFINCS_JAX_RHSMODE1_PAS_DKES_CPU_PAS_TZ_ACTIVE_MAX", 15000)
    block_size = int(max_l) * int(n_theta) * int(n_zeta)
    return block_size >= max(1, int(min_block)) and int(active_size) <= max(1, int(max_active))


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
    "FP_FORCE_XMG_WEAK_KINDS",
    "PAS_WEAK_AUTO_OVERRIDE_KINDS",
    "canonical_rhs1_preconditioner_kind",
    "pas_auto_skip_strong_retry",
    "rhs1_fp_dkes_default_kind",
    "rhs1_fp_dkes_env_preconditioner_kind",
    "rhs1_large_fp_near_zero_er_override_kind",
    "rhs1_pas_family_refinement_kind",
    "rhs1_gpu_sparse_fallback_skip_allowed",
    "rhs1_pas_auto_large_base_kind",
    "rhs1_pas_dkes_cpu_pas_tz_preferred",
    "rhs1_pas_dkes_xblock_allowed",
    "rhs1_pas_weak_auto_override_kind",
    "rhs1_pas_tokamak_cpu_xblock_preferred",
    "rhs1_pas_tokamak_gpu_theta_allowed",
    "rhs1_pas_tokamak_gpu_xblock_preferred",
    "rhs1_sharded_line_override_allowed",
]
