"""RHSMode=1 PAS applicability and memory policy helpers.

This module holds the small, pure policy functions that decide whether the
specialized PAS tokamak-theta and PAS-TZ preconditioners are eligible to run.
They are intentionally isolated from the large solve orchestration in
``v3_driver.py`` so they can be tested directly and reused from multiple
dispatch paths without duplicating logic.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
import os

import numpy as np

from sfincs_jax.pas_smoother import adaptive_pas_smoother_allowed

_FALSE_VALUES = {"0", "false", "no", "off"}
_RHS1_PAS_PROBE_HEAVY_PRECONDITIONERS = frozenset(
    {
        "point",
        "theta_line",
        "zeta_line",
        "theta_zeta",
        "adi",
        "xblock_tz",
        "sxblock_tz",
        "species_block",
        "schur",
        "pas_hybrid",
    }
)


@dataclass(frozen=True)
class RHS1PASAdaptiveSmootherControls:
    """Execution controls for the bounded PAS adaptive smoother."""

    max_sweeps: int
    omega: float


@dataclass(frozen=True)
class RHS1PASSchurRescueControls:
    """Admission and Krylov controls for the full-system PAS Schur rescue."""

    run: bool
    ratio: float
    max_active_size: int
    restart: int
    maxiter: int


@dataclass(frozen=True)
class RHS1PASForceFullDecision:
    """Routing decision for forcing a full PAS preconditioner after weak collision."""

    run: bool
    ratio: float
    forced_kind: str | None


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    try:
        return float(raw) if raw else float(default)
    except ValueError:
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        return int(raw) if raw else int(default)
    except ValueError:
        return int(default)


@dataclass(frozen=True)
class RHS1PASPreconditionerProbeConfig:
    """Environment controls for the cheap PAS collision-preconditioner probe."""

    enabled: bool
    rel_max: float
    build_max: int


def rhs1_pas_preconditioner_probe_config_from_env() -> RHS1PASPreconditionerProbeConfig:
    """Read PAS preconditioner-probe controls with the historical defaults."""
    enabled = os.environ.get("SFINCS_JAX_PAS_PRECOND_PROBE", "").strip().lower() not in _FALSE_VALUES
    return RHS1PASPreconditionerProbeConfig(
        enabled=enabled,
        rel_max=_env_float("SFINCS_JAX_PAS_PRECOND_PROBE_REL_MAX", 0.9),
        build_max=_env_int("SFINCS_JAX_PAS_PRECOND_BUILD_MAX", 20000),
    )


def rhs1_pas_default_preconditioner_kind(
    *,
    requested_env: str,
    current_kind: str,
    rhs_mode: int,
    include_phi1: bool,
    has_pas: bool,
    n_species: int,
    n_zeta: int,
    geom_scheme: int,
) -> str:
    """Return the robust default PAS preconditioner for tokamak-like multispecies runs."""
    if (
        str(requested_env).strip().lower() in {"", "auto", "default"}
        and int(rhs_mode) == 1
        and not bool(include_phi1)
        and bool(has_pas)
        and int(n_species) >= 2
        and (int(geom_scheme) == 1 or int(n_zeta) <= 9)
    ):
        return "schur"
    return str(current_kind)


def rhs1_pas_preconditioner_probe_admitted(
    *,
    config: RHS1PASPreconditionerProbeConfig,
    preconditioner_kind: str,
    preconditioner_enabled: bool,
    solve_method_kind: str,
    has_pas: bool,
    use_dkes: bool,
) -> bool:
    """Return whether the cheap collision probe should run before heavy PAS builders."""
    return (
        bool(config.enabled)
        and str(preconditioner_kind).strip().lower() in _RHS1_PAS_PROBE_HEAVY_PRECONDITIONERS
        and bool(preconditioner_enabled)
        and str(solve_method_kind).strip().lower() not in {"dense", "dense_ksp"}
        and bool(has_pas)
        and not bool(use_dkes)
    )


def rhs1_pas_preconditioner_probe_large_collision_skip(
    *,
    config: RHS1PASPreconditionerProbeConfig,
    cached_decision: bool | None,
    total_size: int,
    constraint_scheme: int,
    extra_size: int,
) -> tuple[bool | None, str | None]:
    """Return a fail-fast collision decision for PAS systems too large to probe cheaply."""
    if cached_decision is not None:
        return cached_decision, None
    if int(total_size) < int(config.build_max):
        return None, None
    if int(constraint_scheme) == 2 and int(extra_size) > 0:
        return None, None
    message = (
        "solve_v3_full_system_linear_gmres: PAS precond skip "
        f"(size={int(total_size)} >= {int(config.build_max)}) -> collision"
    )
    return True, message


def rhs1_pas_preconditioner_probe_uses_collision(*, probe_rel: float, rel_max: float) -> bool:
    """Return whether a PAS collision-probe residual is strong enough to accept."""
    return float(probe_rel) <= float(rel_max)


def rhs1_pas_force_full_decision_from_env(
    *,
    enabled: bool,
    has_pas: bool,
    residual_norm: float,
    target: float,
    active_size: int,
    requested_kind: str | None,
) -> RHS1PASForceFullDecision:
    """Return whether a weak collision-PAS solve should force a full preconditioner."""

    ratio = _env_float("SFINCS_JAX_PAS_FORCE_FULL_RATIO", 50.0)
    residual_ratio = float(residual_norm) / max(float(target), 1e-300)
    if not (
        bool(enabled)
        and bool(has_pas)
        and float(residual_norm) > float(target)
        and residual_ratio > float(ratio)
    ):
        return RHS1PASForceFullDecision(run=False, ratio=float(ratio), forced_kind=None)

    forced_kind = str(requested_kind or "xmg")
    if has_pas:
        pas_lite_min = _env_int("SFINCS_JAX_PAS_LITE_MIN", 20000)
        forced_kind = "pas_lite" if int(active_size) >= max(1, int(pas_lite_min)) else "pas_hybrid"
    elif forced_kind in {"collision", "schur", "point"}:
        forced_kind = "xmg"
    return RHS1PASForceFullDecision(
        run=True,
        ratio=float(ratio),
        forced_kind=str(forced_kind),
    )


def rhs1_pas_small_near_zero_er_kind(
    *,
    pas_tz_applicable: bool,
    tz_size: int,
    active_size: int,
) -> str:
    """Return the lightweight PAS default below the x-coarsening size regime.

    Near-zero-Er PAS systems below the ``xmg`` threshold should avoid expensive
    global Schur setup. When PAS-TZ blocks are applicable and angular grids are
    modest, use the PAS-native line/x-coarse family; otherwise fall back to xmg.
    """

    pas_lite_tz_max = _env_int("SFINCS_JAX_PAS_LITE_TZ_MAX", 256)
    if bool(pas_tz_applicable) and int(pas_lite_tz_max) > 0 and int(tz_size) <= int(pas_lite_tz_max):
        pas_lite_min = _env_int("SFINCS_JAX_PAS_LITE_MIN", 20000)
        return (
            "pas_lite"
            if int(active_size) >= max(1, int(pas_lite_min))
            else "pas_hybrid"
        )
    return "xmg"


def pas_tokamak_theta_preconditioner_applicable(op) -> bool:
    """Return whether the PAS tokamak theta/L preconditioner is applicable.

    The tokamak-theta branch is intended for PAS-only RHSMode=1 systems with no
    drift/X coupling terms. ``n_zeta == 1`` is the direct tokamak case, but we
    also admit effectively tokamak-like multi-zeta grids when the geometry is
    zeta-invariant to within ``SFINCS_JAX_PAS_TOKAMAK_TZ_TOL``.
    """
    if int(op.rhs_mode) != 1:
        return False
    if int(op.n_zeta) != 1:
        cl = op.fblock.collisionless
        if cl is None:
            return False
        try:
            b_hat = np.asarray(cl.b_hat, dtype=np.float64)
            b_sup_theta = np.asarray(cl.b_hat_sup_theta, dtype=np.float64)
            b_sup_zeta = np.asarray(cl.b_hat_sup_zeta, dtype=np.float64)
            db_dtheta = np.asarray(cl.db_hat_dtheta, dtype=np.float64)
            db_dzeta = np.asarray(cl.db_hat_dzeta, dtype=np.float64)
        except Exception:
            return False
        tol_env = os.environ.get("SFINCS_JAX_PAS_TOKAMAK_TZ_TOL", "").strip()
        try:
            tol = float(tol_env) if tol_env else 1e-12
        except ValueError:
            tol = 1e-12
        if (
            np.max(np.abs(b_hat - b_hat[:, :1])) > tol
            or np.max(np.abs(b_sup_theta - b_sup_theta[:, :1])) > tol
            or np.max(np.abs(b_sup_zeta - b_sup_zeta[:, :1])) > tol
            or np.max(np.abs(db_dtheta - db_dtheta[:, :1])) > tol
            or np.max(np.abs(db_dzeta - db_dzeta[:, :1])) > tol
        ):
            return False
    fb = op.fblock
    if fb.collisionless is None or fb.pas is None:
        return False
    if (
        fb.exb_theta is not None
        or fb.exb_zeta is not None
        or fb.magdrift_theta is not None
        or fb.magdrift_zeta is not None
        or fb.magdrift_xidot is not None
        or fb.er_xdot is not None
        or fb.er_xidot is not None
        or fb.fp is not None
        or fb.fp_phi1 is not None
    ):
        return False
    return True


def pas_tz_preconditioner_applicable(op) -> bool:
    """Return whether the PAS 3D (theta,zeta)/L preconditioner is applicable."""
    if int(op.rhs_mode) != 1:
        return False
    if int(op.n_theta) <= 1 or int(op.n_zeta) <= 1:
        return False
    if int(op.n_theta) * int(op.n_zeta) < 64:
        return False
    if int(op.n_xi) < 2:
        return False
    fb = op.fblock
    if fb.collisionless is None or fb.pas is None:
        return False
    if fb.fp is not None or fb.fp_phi1 is not None:
        return False
    return True


def rhs1_pas_tz_max_bytes() -> int:
    """Parse the PAS-TZ memory ceiling from the environment with a safe default."""
    env = os.environ.get("SFINCS_JAX_RHSMODE1_PAS_TZ_MAX_BYTES", "").strip()
    try:
        return int(env) if env else 2 * 1024 * 1024 * 1024
    except ValueError:
        return 2 * 1024 * 1024 * 1024


def estimate_rhs1_pas_tz_build_bytes(op) -> int:
    """Estimate the dense PAS-TZ builder memory footprint in bytes.

    This is intentionally conservative. It counts persistent block-Thomas
    factors, the host/device overlap while cached arrays are materialized, and
    the largest dense work arrays used by local inversions. The extra headroom
    keeps routing away from builder choices that are nominally under the hard
    cache footprint but unsafe once transient live arrays are included.
    """
    return int(estimate_rhs1_pas_tz_build_memory(op)["total_nbytes"])


def estimate_rhs1_pas_tz_build_memory(op) -> dict[str, object]:
    """Return structured dense PAS-TZ builder memory preflight metadata.

    The scalar byte estimate is still exposed through
    ``estimate_rhs1_pas_tz_build_bytes`` for existing route gates. This richer
    form lets tests, benchmark manifests, and fail-fast diagnostics explain why
    geometry-rich PAS-TZ builds are accepted or rejected without entering a heavy
    preconditioner construction.
    """
    if not pas_tz_preconditioner_applicable(op):
        max_bytes = max(0, rhs1_pas_tz_max_bytes())
        return {
            "applicable": False,
            "safe": True,
            "reason": "pas-tz-inapplicable",
            "total_nbytes": 0,
            "max_nbytes": int(max_bytes),
        }
    n_species = int(op.n_species)
    n_x = int(op.n_x)
    n_l_full = int(op.n_xi)
    n_theta = int(op.n_theta)
    n_zeta = int(op.n_zeta)
    n_tz = int(n_theta * n_zeta)
    if n_tz <= 1 or n_l_full < 2:
        return 0

    pas_tz_lmax_env = os.environ.get("SFINCS_JAX_RHSMODE1_PAS_TZ_LMAX", "").strip()
    try:
        pas_tz_lmax = int(pas_tz_lmax_env) if pas_tz_lmax_env else 0
    except ValueError:
        pas_tz_lmax = 0
    lmax_source = "env" if pas_tz_lmax > 0 else "default"
    if pas_tz_lmax <= 0:
        if n_tz <= 192:
            pas_tz_lmax = n_l_full
        elif n_tz >= 256:
            pas_tz_lmax = 6
        elif n_tz >= 128:
            pas_tz_lmax = 8
        else:
            pas_tz_lmax = 12
        if (
            getattr(op.fblock, "exb_theta", None) is None
            and getattr(op.fblock, "exb_zeta", None) is None
            and getattr(op.fblock, "magdrift_theta", None) is None
            and getattr(op.fblock, "magdrift_zeta", None) is None
            and getattr(op.fblock, "magdrift_xidot", None) is None
            and getattr(op.fblock, "er_xdot", None) is None
            and getattr(op.fblock, "er_xidot", None) is None
            and n_tz <= 256
        ):
            pas_tz_lmax = n_l_full
            lmax_source = "drift-free-small-tz"
    n_l_use = min(n_l_full, max(2, int(pas_tz_lmax)))
    tz = int(n_tz)
    twotz = int(2 * tz)

    inv_a01 = n_species * n_x * twotz * twotz
    g01 = n_species * n_x * twotz * tz
    inv_a = n_species * n_x * max(n_l_use - 2, 0) * tz * tz
    g = n_species * n_x * max(n_l_use - 3, 0) * tz * tz
    stored_factor_entries = int(inv_a01 + g01 + inv_a + g)

    # Geometry/operator arrays live alongside the factor arrays while the build
    # runs: dtheta_tz, dzeta_tz, m_tz per species, exb_op_tz, and eye_tz.
    geometry_entries = int((4 + n_species) * tz * tz)

    # Local inversion work keeps the input block, inverse/pseudoinverse result,
    # coupling blocks, and effective diagonal blocks live around linalg calls.
    inversion_workspace_entries = int(max(2 * twotz * twotz + 2 * twotz * tz, 6 * tz * tz))

    # Host factor arrays remain live while jnp.asarray materializes device
    # copies. Apply headroom to account for allocator alignment and linalg
    # temporaries that are not visible from Python-level array shapes.
    live_entries = 2 * stored_factor_entries + geometry_entries + inversion_workspace_entries
    headroom = 1.25
    total_nbytes = int(math.ceil(live_entries * 8 * headroom))
    max_nbytes = int(max(0, rhs1_pas_tz_max_bytes()))
    safe = bool(total_nbytes <= max_nbytes)
    return {
        "applicable": True,
        "safe": safe,
        "reason": "within-pas-tz-build-memory-limit" if safe else "pas-tz-build-memory-limit-exceeded",
        "n_species": int(n_species),
        "n_x": int(n_x),
        "n_xi_full": int(n_l_full),
        "n_theta": int(n_theta),
        "n_zeta": int(n_zeta),
        "n_tz": int(n_tz),
        "active_unknowns": int(n_species * n_x * n_l_full * n_tz),
        "lmax": int(n_l_use),
        "lmax_requested": int(pas_tz_lmax),
        "lmax_source": str(lmax_source),
        "stored_factor_entries": int(stored_factor_entries),
        "geometry_entries": int(geometry_entries),
        "inversion_workspace_entries": int(inversion_workspace_entries),
        "live_entries": int(live_entries),
        "headroom": float(headroom),
        "scalar_nbytes": 8,
        "total_nbytes": int(total_nbytes),
        "max_nbytes": int(max_nbytes),
    }


def pas_tz_preconditioner_memory_safe(op) -> bool:
    """Return whether the PAS-TZ builder estimate fits within the memory ceiling."""
    estimate = estimate_rhs1_pas_tz_build_bytes(op)
    if estimate <= 0:
        return True
    return estimate <= max(0, rhs1_pas_tz_max_bytes())


def rhs1_pas_adaptive_smoother_allowed(
    *,
    op,
    active_size: int,
    residual_norm: float,
    target: float,
    use_implicit: bool,
) -> bool:
    """Return whether the adaptive PAS smoother should run before stronger solves."""
    env = os.environ.get("SFINCS_JAX_PAS_ADAPTIVE_SMOOTHER", "").strip().lower()
    enabled = env not in {"0", "false", "no", "off"}
    min_env = os.environ.get("SFINCS_JAX_PAS_ADAPTIVE_SMOOTHER_MIN", "").strip()
    try:
        min_size = int(min_env) if min_env else 2000
    except ValueError:
        min_size = 2000
    return adaptive_pas_smoother_allowed(
        enabled=enabled,
        use_implicit=bool(use_implicit),
        has_pas=op.fblock.pas is not None,
        include_phi1=bool(op.include_phi1),
        residual_norm=float(residual_norm),
        target=float(target),
        active_size=int(active_size),
        min_size=int(min_size),
    )


def rhs1_pas_adaptive_smoother_controls_from_env() -> RHS1PASAdaptiveSmootherControls:
    """Return PAS smoother sweeps and damping controls with legacy defaults."""

    sweeps_env = os.environ.get("SFINCS_JAX_PAS_ADAPTIVE_SMOOTHER_SWEEPS", "").strip()
    omega_env = os.environ.get("SFINCS_JAX_PAS_ADAPTIVE_SMOOTHER_OMEGA", "").strip()
    try:
        max_sweeps = int(sweeps_env) if sweeps_env else 3
    except ValueError:
        max_sweeps = 3
    try:
        omega = float(omega_env) if omega_env else 1.0
    except ValueError:
        omega = 1.0
    return RHS1PASAdaptiveSmootherControls(max_sweeps=int(max_sweeps), omega=float(omega))


def rhs1_pas_schur_rescue_controls_from_env(
    *,
    rhs_mode: int,
    include_phi1: bool,
    has_pas: bool,
    n_species: int,
    residual_norm: float,
    target: float,
    active_size: int,
    restart: int,
    maxiter: int | None,
) -> RHS1PASSchurRescueControls:
    """Return full-system PAS Schur rescue admission and retry controls."""

    ratio_env = os.environ.get("SFINCS_JAX_RHSMODE1_PAS_SCHUR_RESCUE_RATIO", "").strip()
    max_env = os.environ.get("SFINCS_JAX_RHSMODE1_PAS_SCHUR_RESCUE_MAX", "").strip()
    try:
        ratio = float(ratio_env) if ratio_env else 1.0e4
    except ValueError:
        ratio = 1.0e4
    try:
        max_active_size = int(max_env) if max_env else 90000
    except ValueError:
        max_active_size = 90000

    restart_env = os.environ.get("SFINCS_JAX_RHSMODE1_PAS_SCHUR_RESCUE_RESTART", "").strip()
    maxiter_env = os.environ.get("SFINCS_JAX_RHSMODE1_PAS_SCHUR_RESCUE_MAXITER", "").strip()
    try:
        restart_use = int(restart_env) if restart_env else max(120, int(restart))
    except ValueError:
        restart_use = max(120, int(restart))
    try:
        maxiter_use = int(maxiter_env) if maxiter_env else max(1200, int(maxiter or 400) * 3)
    except ValueError:
        maxiter_use = max(1200, int(maxiter or 400) * 3)

    eligible = (
        int(rhs_mode) == 1
        and (not bool(include_phi1))
        and bool(has_pas)
        and int(n_species) >= 2
        and np.isfinite(float(residual_norm))
    )
    run = bool(
        eligible
        and float(ratio) > 0.0
        and int(active_size) <= max(1, int(max_active_size))
        and float(residual_norm) > float(target) * float(ratio)
    )
    return RHS1PASSchurRescueControls(
        run=bool(run),
        ratio=float(ratio),
        max_active_size=int(max_active_size),
        restart=int(restart_use),
        maxiter=int(maxiter_use),
    )


def build_pas_tz_memory_fallback(
    *,
    op,
    matvec_shard_axis: Callable[[object], str | None],
    device_count: Callable[[], int],
    theta_schwarz_builder: Callable[..., Callable],
    zeta_schwarz_builder: Callable[..., Callable],
    hybrid_builder: Callable[..., Callable],
    collision_builder: Callable[..., Callable] | None = None,
    tzfft_builder: Callable[..., Callable] | None = None,
    reduce_full=None,
    expand_reduced=None,
) -> Callable:
    """Build the fallback preconditioner for memory-unsafe PAS-TZ requests.

    On multi-device sharded runs we use the shard-axis-specific Schwarz builder.
    Otherwise we prefer the matrix-free ``tzfft`` fallback when available, unless
    ``SFINCS_JAX_RHSMODE1_PAS_TZ_MEMORY_FALLBACK`` explicitly requests
    ``collision``. The older PAS-hybrid fallback remains available with
    ``SFINCS_JAX_RHSMODE1_PAS_TZ_MEMORY_FALLBACK=hybrid`` for A/B profiling.
    Structured Schwarz builders still allocate dense patch inverses, so they are
    guarded by an explicit patch-work estimate; unsafe structured requests also
    use ``tzfft`` first when available unless collision was requested.
    """
    requested = os.environ.get("SFINCS_JAX_RHSMODE1_PAS_TZ_MEMORY_FALLBACK", "")
    explicit_collision_request = _pas_tz_explicit_collision_fallback_requested(requested)
    shard_axis = matvec_shard_axis(op)
    axis = resolve_pas_tz_memory_fallback_axis(
        op=op,
        requested=requested,
        shard_axis=shard_axis,
        n_devices=device_count(),
    )
    if axis in {"theta", "zeta"}:
        dd_block = _parse_pas_tz_fallback_int(
            f"SFINCS_JAX_RHSMODE1_{axis.upper()}_DD_BLOCK",
            fallback_name="SFINCS_JAX_RHSMODE1_PAS_TZ_SCHWARZ_BLOCK",
            default=64,
        )
        dd_overlap = _parse_pas_tz_fallback_int(
            f"SFINCS_JAX_RHSMODE1_{axis.upper()}_DD_OVERLAP",
            fallback_name="SFINCS_JAX_RHSMODE1_PAS_TZ_SCHWARZ_OVERLAP",
            default=1,
        )
        guard = pas_tz_schwarz_fallback_guard(
            op,
            axis=axis,
            block=dd_block,
            overlap=dd_overlap,
        )
        if not bool(guard["safe"]):
            if tzfft_builder is not None and not explicit_collision_request:
                metadata = dict(guard)
                metadata["reason"] = f"{guard.get('reason', 'schwarz-unsafe')}; using tzfft"
                metadata["requested_axis"] = axis
                precond = tzfft_builder(op=op, reduce_full=reduce_full, expand_reduced=expand_reduced)
                _mark_pas_tz_guarded_fallback(precond, axis="tzfft", metadata=metadata)
            elif collision_builder is not None:
                precond = collision_builder(op=op, reduce_full=reduce_full, expand_reduced=expand_reduced)
                _mark_pas_tz_guarded_fallback(precond, axis=axis, metadata=guard)
            else:
                precond = hybrid_builder(op=op, reduce_full=reduce_full, expand_reduced=expand_reduced)
                _mark_pas_tz_guarded_fallback(precond, axis=axis, metadata=guard)
            return precond
        schwarz_builder = theta_schwarz_builder if axis == "theta" else zeta_schwarz_builder
        return schwarz_builder(
            op=op,
            block=dd_block,
            overlap=dd_overlap,
            reduce_full=reduce_full,
            expand_reduced=expand_reduced,
        )
    cheap_kind = resolve_pas_tz_cheap_fallback_kind(requested=requested)
    if (cheap_kind == "tzfft" and tzfft_builder is not None) or (
        cheap_kind == "collision" and tzfft_builder is not None and not explicit_collision_request
    ):
        precond = tzfft_builder(op=op, reduce_full=reduce_full, expand_reduced=expand_reduced)
        _mark_pas_tz_guarded_fallback(precond, axis="tzfft", metadata={"safe": True, "reason": "cheap-tzfft"})
        return precond
    if cheap_kind == "collision" and collision_builder is not None:
        precond = collision_builder(op=op, reduce_full=reduce_full, expand_reduced=expand_reduced)
        _mark_pas_tz_guarded_fallback(precond, axis="collision", metadata={"safe": True, "reason": "cheap-collision"})
        return precond
    precond = hybrid_builder(op=op, reduce_full=reduce_full, expand_reduced=expand_reduced)
    _mark_pas_tz_guarded_fallback(precond, axis="hybrid", metadata={"safe": True, "reason": "legacy-hybrid"})
    return precond


def _mark_pas_tz_guarded_fallback(precond: Callable, *, axis: str, metadata: dict[str, object] | None = None) -> None:
    """Attach best-effort metadata to a guarded PAS-TZ fallback callable."""
    try:
        setattr(precond, "_sfincs_jax_pas_tz_guarded_fallback", True)
        setattr(precond, "_sfincs_jax_pas_tz_guarded_axis", str(axis))
        setattr(precond, "_sfincs_jax_pas_tz_guarded_metadata", dict(metadata or {}))
    except Exception:
        pass


def _pas_tz_explicit_collision_fallback_requested(requested: str) -> bool:
    """Return whether the PAS-TZ fallback env explicitly asks for collision."""
    req = str(requested or "").strip().lower().replace("-", "_")
    return req in {
        "collision",
        "collisions",
        "collision_diag",
        "pas_collision",
        "cheap_collision",
        "collision_tzfft",
        "collision_tzfft_correction",
    }


def estimate_pas_tz_schwarz_fallback_work(
    op,
    *,
    axis: str,
    block: int,
    overlap: int,
) -> dict[str, int]:
    """Estimate dense-patch work for an opt-in PAS-TZ Schwarz fallback.

    The theta/zeta Schwarz builders currently precompute dense inverses for
    every species and orthogonal angular line. The dominant memory term is the
    number of stored inverse entries, while the dominant setup-time term scales
    cubically with the largest patch unknown count. This estimator is deliberately
    simple and conservative so routing tests can reject known-bad production
    shapes before they enter a long JAX/XLA build.
    """
    axis_l = str(axis).strip().lower()
    if axis_l not in {"theta", "zeta"}:
        axis_l = preferred_pas_tz_schwarz_axis(op)
    n_species = max(1, int(getattr(op, "n_species", 1)))
    n_theta = max(1, int(getattr(op, "n_theta", 1)))
    n_zeta = max(1, int(getattr(op, "n_zeta", 1)))
    block_i = max(1, int(block))
    overlap_i = max(0, int(overlap))
    n_axis = n_theta if axis_l == "theta" else n_zeta
    n_lines = n_zeta if axis_l == "theta" else n_theta
    n_patches_per_line = max(1, int(math.ceil(float(n_axis) / float(block_i))))
    max_patch_extent = min(n_axis, block_i + 2 * overlap_i)
    local_velocity_dof = _pas_tz_local_velocity_dof(op)
    max_patch_unknowns = int(max_patch_extent * local_velocity_dof)
    patch_count = int(n_species * n_lines * n_patches_per_line)
    inverse_entries = int(patch_count * max_patch_unknowns * max_patch_unknowns)
    return {
        "axis": 0 if axis_l == "theta" else 1,
        "block": int(block_i),
        "overlap": int(overlap_i),
        "patch_count": int(patch_count),
        "max_patch_extent": int(max_patch_extent),
        "local_velocity_dof": int(local_velocity_dof),
        "max_patch_unknowns": int(max_patch_unknowns),
        "inverse_entries": int(inverse_entries),
        "inverse_bytes_float64": int(inverse_entries * 8),
    }


def pas_tz_schwarz_fallback_memory_safe(
    op,
    *,
    axis: str,
    block: int,
    overlap: int,
) -> bool:
    """Return whether a structured PAS-TZ Schwarz fallback should be attempted."""
    return bool(pas_tz_schwarz_fallback_guard(op, axis=axis, block=block, overlap=overlap)["safe"])


def pas_tz_schwarz_fallback_guard(
    op,
    *,
    axis: str,
    block: int,
    overlap: int,
) -> dict[str, object]:
    """Return structured PAS-TZ Schwarz guard metadata and the final decision."""
    work = estimate_pas_tz_schwarz_fallback_work(op, axis=axis, block=block, overlap=overlap)
    max_patch_unknowns = _parse_nonnegative_env_int(
        "SFINCS_JAX_RHSMODE1_PAS_TZ_SCHWARZ_MAX_PATCH_UNKNOWNS",
        default=8192,
    )
    max_inverse_entries = _parse_nonnegative_env_int(
        "SFINCS_JAX_RHSMODE1_PAS_TZ_SCHWARZ_MAX_INVERSE_ENTRIES",
        default=100_000_000,
    )
    failures: list[str] = []
    if max_patch_unknowns > 0 and int(work["max_patch_unknowns"]) > max_patch_unknowns:
        failures.append("max-patch-unknowns-exceeded")
    if max_inverse_entries > 0 and int(work["inverse_entries"]) > max_inverse_entries:
        failures.append("max-inverse-entries-exceeded")
    return {
        "safe": not failures,
        "reason": "within-structured-schwarz-guard" if not failures else ",".join(failures),
        "axis": str(axis),
        "block": int(max(1, int(block))),
        "overlap": int(max(0, int(overlap))),
        "max_patch_unknowns_limit": int(max_patch_unknowns),
        "max_inverse_entries_limit": int(max_inverse_entries),
        "work": work,
    }


def _pas_tz_local_velocity_dof(op) -> int:
    """Return the per-angular-line velocity unknown count used by Schwarz patches."""
    try:
        collisionless = getattr(getattr(op, "fblock", None), "collisionless", None)
        values = getattr(collisionless, "n_xi_for_x", None)
        if values is not None:
            arr = np.asarray(values, dtype=np.int64).reshape(-1)
            if arr.size:
                return max(1, int(np.sum(arr)))
    except Exception:
        pass
    n_x = max(1, int(getattr(op, "n_x", 1)))
    n_xi = max(1, int(getattr(op, "n_xi", 1)))
    return int(n_x * n_xi)


def _parse_nonnegative_env_int(name: str, *, default: int) -> int:
    """Parse a non-negative integer env var; non-positive values disable a cap."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return int(default)
    try:
        return max(0, int(raw))
    except ValueError:
        return int(default)


def _parse_pas_tz_fallback_int(name: str, *, fallback_name: str, default: int) -> int:
    """Parse a structured PAS fallback integer env var with a shared fallback."""
    for env_name in (name, fallback_name):
        raw = os.environ.get(env_name, "").strip()
        if raw:
            try:
                value = int(raw)
            except ValueError:
                continue
            if value > 0:
                return value
    return int(default)


def preferred_pas_tz_schwarz_axis(op) -> str:
    """Choose the structured Schwarz axis with the richer angular direction."""
    try:
        n_theta = int(op.n_theta)
        n_zeta = int(op.n_zeta)
    except Exception:
        return "theta"
    return "zeta" if n_zeta >= n_theta else "theta"


def resolve_pas_tz_cheap_fallback_kind(*, requested: str) -> str:
    """Resolve the cheap single-device fallback used after a rejected PAS-TZ build.

    ``collision`` is the default because it is bounded in memory and setup time.
    ``hybrid`` is kept as an explicit compatibility/profiling override.
    ``tzfft`` is an explicit experimental matrix-free angular-streaming route.
    """
    req = str(requested or "").strip().lower().replace("-", "_")
    if req in {"hybrid", "pas_hybrid", "old", "legacy"}:
        return "hybrid"
    if req in {"tzfft", "pas_tzfft", "pas_fft", "pas_stream_fft", "pas_streaming_fft"}:
        return "tzfft"
    return "collision"


def resolve_pas_tz_guarded_correction_kind(*, requested: str) -> str | None:
    """Resolve an optional matrix-free correction after guarded PAS-TZ fallback."""
    req = str(requested or "").strip().lower().replace("-", "_")
    if req in {"", "0", "false", "no", "off", "none"}:
        return None
    if req in {
        "tzfft",
        "pas_tzfft",
        "pas_fft",
        "pas_stream_fft",
        "pas_streaming_fft",
        "collision_tzfft",
        "collision_tzfft_correction",
    }:
        return "tzfft"
    return None


def rhs1_pas_tz_guarded_strong_retry_from_env() -> bool:
    """Return whether guarded PAS-TZ fallback may retry with the strong builder."""
    raw = os.environ.get("SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_STRONG_RETRY", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def resolve_pas_tz_memory_fallback_axis(
    *,
    op,
    requested: str,
    shard_axis: str | None,
    n_devices: int,
) -> str | None:
    """Resolve the memory-unsafe PAS-TZ fallback axis.

    Empty/default requests preserve the historical behavior: only already-sharded
    multi-device runs select a Schwarz fallback automatically. Explicit
    ``theta``, ``zeta``, or ``schwarz`` requests enable the structured route for
    bounded single-device experiments without widening production defaults.
    """
    req = str(requested or "").strip().lower().replace("-", "_")
    if req in {"hybrid", "pas_hybrid", "off", "0", "false", "no"}:
        return None
    if req in {"theta", "theta_schwarz"}:
        return "theta"
    if req in {"zeta", "zeta_schwarz"}:
        return "zeta"
    if req in {"schwarz", "structured", "structured_schwarz", "auto_schwarz"}:
        return preferred_pas_tz_schwarz_axis(op)
    if req:
        return None
    if shard_axis in {"theta", "zeta"} and int(n_devices) > 1:
        return str(shard_axis)
    return None


__all__ = [
    "RHS1PASAdaptiveSmootherControls",
    "RHS1PASForceFullDecision",
    "RHS1PASPreconditionerProbeConfig",
    "RHS1PASSchurRescueControls",
    "build_pas_tz_memory_fallback",
    "estimate_rhs1_pas_tz_build_bytes",
    "estimate_rhs1_pas_tz_build_memory",
    "estimate_pas_tz_schwarz_fallback_work",
    "pas_tokamak_theta_preconditioner_applicable",
    "pas_tz_preconditioner_applicable",
    "pas_tz_schwarz_fallback_guard",
    "pas_tz_schwarz_fallback_memory_safe",
    "pas_tz_preconditioner_memory_safe",
    "preferred_pas_tz_schwarz_axis",
    "resolve_pas_tz_cheap_fallback_kind",
    "resolve_pas_tz_guarded_correction_kind",
    "resolve_pas_tz_memory_fallback_axis",
    "rhs1_pas_adaptive_smoother_allowed",
    "rhs1_pas_adaptive_smoother_controls_from_env",
    "rhs1_pas_default_preconditioner_kind",
    "rhs1_pas_force_full_decision_from_env",
    "rhs1_pas_preconditioner_probe_admitted",
    "rhs1_pas_preconditioner_probe_config_from_env",
    "rhs1_pas_preconditioner_probe_large_collision_skip",
    "rhs1_pas_preconditioner_probe_uses_collision",
    "rhs1_pas_schur_rescue_controls_from_env",
    "rhs1_pas_small_near_zero_er_kind",
    "rhs1_pas_tz_guarded_strong_retry_from_env",
    "rhs1_pas_tz_max_bytes",
]
