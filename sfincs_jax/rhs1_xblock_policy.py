"""Pure RHSMode=1 x-block sparse-PC routing policy helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math

DEFAULT_FULL_FP_3D_RIGHT_PC_MAX_ACTIVE_SIZE = 45_000
DEFAULT_FULL_FP_3D_SIDE_PROBE_MIN_ACTIVE_SIZE = 80_000
DEFAULT_FULL_FP_3D_DEVICE_HOST_FALLBACK_MIN_ACTIVE_SIZE = 80_000
DEFAULT_FULL_FP_3D_SIDE_PROBE_SWITCH_RATIO = 5_000.0
DEFAULT_FULL_FP_3D_LGMRES_RESCUE_MAXITER = 80
DEFAULT_FULL_FP_3D_LGMRES_RESCUE_OUTER_K = 10
DEFAULT_RHS1_XBLOCK_LOCAL_DROP_TOL = 0.0
DEFAULT_RHS1_XBLOCK_LOCAL_DROP_REL = 1.0e-8
DEFAULT_RHS1_XBLOCK_LOCAL_ILU_DROP_TOL = 1.0e-4
DEFAULT_RHS1_XBLOCK_LOCAL_FILL_FACTOR = 10.0
DEFAULT_RHS1_XBLOCK_LOCAL_ROW_NNZ_CAP = 64
DEFAULT_RHS1_XBLOCK_LOCAL_COMPACT_ROW_NNZ_CAP = 0
DEFAULT_RHS1_XBLOCK_LOWER_FILL_DROP_TOL = DEFAULT_RHS1_XBLOCK_LOCAL_DROP_TOL
DEFAULT_RHS1_XBLOCK_LOWER_FILL_DROP_REL = DEFAULT_RHS1_XBLOCK_LOCAL_DROP_REL
DEFAULT_RHS1_XBLOCK_LOWER_FILL_ILU_DROP_TOL = 1.0e-3
DEFAULT_RHS1_XBLOCK_LOWER_FILL_FACTOR = 4.0
DEFAULT_RHS1_XBLOCK_LOWER_FILL_ROW_NNZ_CAP = 32
DEFAULT_RHS1_XBLOCK_LOWER_FILL_COMPACT_ROW_NNZ_CAP = 32
DEFAULT_RHS1_XBLOCK_LOWER_FILL_MAX_BLOCK_SIZE = 250_000
DEFAULT_RHS1_XBLOCK_LOWER_FILL_ACCEPT_MAX_RESIDUAL_RATIO = 100.0
DEFAULT_RHS1_XBLOCK_LOWER_FILL_ACCEPT_MIN_IMPROVEMENT = 1.0
DEFAULT_RHS1_XBLOCK_LOWER_FILL_FACTOR_PROBE_MAX = 1.0e8

_TRUE_VALUES = {"1", "true", "t", "yes", "on", ".true.", ".t."}
_FALSE_VALUES = {"0", "false", "f", "no", "off", ".false.", ".f."}
_DEFAULT_VALUES = {"", "auto", "default"}


@dataclass(frozen=True)
class RHS1XBlockSparsePCPolicy:
    """Resolved x-block sparse preconditioned Krylov policy for one solve."""

    precondition_side: str
    default_right_preconditioned: bool
    krylov_method: str
    ignored_krylov_env: bool
    gmres_restart: int
    restart_capped: bool


@dataclass(frozen=True)
class RHS1XBlockDeviceHostFallbackDecision:
    """Resolved non-autodiff host fallback for large device-Krylov QI solves."""

    mode: str
    used: bool
    reason: str
    requested_device_krylov: bool
    requested_method: str
    effective_krylov_env_value: str
    min_active_size: int
    qi_like_full_fp_3d: bool
    ignored_env: bool
    non_autodiff: bool = True

    def to_metadata(self) -> dict[str, object]:
        """Return JSON-ready fallback metadata."""
        return {
            "mode": self.mode,
            "used": bool(self.used),
            "reason": self.reason,
            "requested_device_krylov": bool(self.requested_device_krylov),
            "requested_method": self.requested_method,
            "effective_krylov_env_value": self.effective_krylov_env_value,
            "min_active_size": int(self.min_active_size),
            "qi_like_full_fp_3d": bool(self.qi_like_full_fp_3d),
            "ignored_env": bool(self.ignored_env),
            "non_autodiff": bool(self.non_autodiff),
        }


@dataclass(frozen=True)
class RHS1XBlockQIDeviceOperatorReuseDecision:
    """Resolved gate for the QI device operator/coarse-reuse x-block route."""

    enabled: bool
    reason: str
    requested: bool
    skip_xblock_factors: bool
    requested_device_krylov: bool
    matrix_free: bool
    use_in_krylov: bool
    precondition_side: str
    qi_like_full_fp_3d: bool

    def to_metadata(self) -> dict[str, object]:
        """Return JSON-ready routing metadata."""
        return {
            "enabled": bool(self.enabled),
            "reason": self.reason,
            "requested": bool(self.requested),
            "skip_xblock_factors": bool(self.skip_xblock_factors),
            "requested_device_krylov": bool(self.requested_device_krylov),
            "matrix_free": bool(self.matrix_free),
            "use_in_krylov": bool(self.use_in_krylov),
            "precondition_side": self.precondition_side,
            "qi_like_full_fp_3d": bool(self.qi_like_full_fp_3d),
        }


@dataclass(frozen=True)
class RHS1XBlockLocalSolveTuning:
    """Sparse local factorization tuning for one x-block candidate."""

    drop_tol: float
    drop_rel: float
    ilu_drop_tol: float
    fill_factor: float
    row_nnz_cap: int
    compact_row_nnz_cap: int

    def to_metadata(self) -> dict[str, object]:
        """Return JSON-ready policy metadata."""
        return {
            "drop_tol": float(self.drop_tol),
            "drop_rel": float(self.drop_rel),
            "ilu_drop_tol": float(self.ilu_drop_tol),
            "fill_factor": float(self.fill_factor),
            "row_nnz_cap": int(self.row_nnz_cap),
            "compact_row_nnz_cap": int(self.compact_row_nnz_cap),
        }


@dataclass(frozen=True)
class RHS1XBlockLocalSolveCandidate:
    """Selected local x-block factorization candidate and metadata labels."""

    block_size: int
    lu_max: int
    mode: str
    factorization: str
    tuning: RHS1XBlockLocalSolveTuning
    metadata_label: str
    selection_reason: str
    exact_lu: bool
    lower_fill: bool
    lower_fill_requested: bool
    ignored_lower_fill_env: bool
    lower_fill_max_block_size: int
    lower_fill_block_size_capped: bool

    def to_metadata(self) -> dict[str, object]:
        """Return JSON-ready policy metadata for traces/manifests."""
        data = {
            "metadata_label": self.metadata_label,
            "block_size": int(self.block_size),
            "lu_max": int(self.lu_max),
            "mode": self.mode,
            "factorization": self.factorization,
            "selection_reason": self.selection_reason,
            "exact_lu": bool(self.exact_lu),
            "lower_fill": bool(self.lower_fill),
            "lower_fill_requested": bool(self.lower_fill_requested),
            "ignored_lower_fill_env": bool(self.ignored_lower_fill_env),
            "lower_fill_max_block_size": int(self.lower_fill_max_block_size),
            "lower_fill_block_size_capped": bool(self.lower_fill_block_size_capped),
        }
        data.update(self.tuning.to_metadata())
        return data


@dataclass(frozen=True)
class RHS1XBlockLowerFillAcceptance:
    """Acceptance/rejection decision for a probed lower-fill local candidate."""

    accepted: bool
    reason: str
    metadata_label: str
    candidate_metadata_label: str
    residual_ratio: float | None
    improvement: float | None
    factor_probe_ratio: float | None
    max_residual_ratio: float
    min_improvement: float
    factor_probe_max: float

    def to_metadata(self) -> dict[str, object]:
        """Return JSON-ready acceptance metadata."""
        return {
            "accepted": bool(self.accepted),
            "reason": self.reason,
            "metadata_label": self.metadata_label,
            "candidate_metadata_label": self.candidate_metadata_label,
            "residual_ratio": None if self.residual_ratio is None else float(self.residual_ratio),
            "improvement": None if self.improvement is None else float(self.improvement),
            "factor_probe_ratio": None if self.factor_probe_ratio is None else float(self.factor_probe_ratio),
            "max_residual_ratio": float(self.max_residual_ratio),
            "min_improvement": float(self.min_improvement),
            "factor_probe_max": float(self.factor_probe_max),
        }


def _normalize_policy_token(value: object) -> str:
    return str(value).strip().lower().replace("-", "_")


def _parse_nonnegative_int_value(env_value: object, default: int) -> int:
    raw = str(env_value).strip()
    if not raw:
        return max(0, int(default))
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return max(0, int(default))


def _parse_finite_float_value(env_value: object, default: float, *, minimum: float = 0.0) -> float:
    raw = str(env_value).strip()
    try:
        value = float(raw) if raw else float(default)
    except (TypeError, ValueError):
        value = float(default)
    if not math.isfinite(value):
        value = float(default)
    return max(float(minimum), float(value))


def _finite_float_or_none(value: object | None) -> float | None:
    if value is None:
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value_f):
        return None
    return float(value_f)


def rhs1_xblock_lower_fill_mode(env_value: str) -> tuple[str, bool]:
    """Return lower-fill local policy mode and whether an env value was ignored.

    The default is ``off`` so the current exact-LU vs ILU split is preserved.
    ``probe`` asks future driver integration to build a bounded lower-fill ILU
    candidate only where legacy exact LU is already inapplicable; ``force``
    permits lower-fill ILU even inside the exact-LU size window.
    """
    raw = _normalize_policy_token(env_value)
    if raw in _DEFAULT_VALUES or raw in _FALSE_VALUES or raw in {"none", "legacy"}:
        return "off", False
    if raw in _TRUE_VALUES or raw in {
        "probe",
        "candidate",
        "lower_fill",
        "lower_fill_ilu",
        "bounded",
        "bounded_ilu",
        "ilu",
    }:
        return "probe", False
    if raw in {"force", "forced", "always", "require", "required", "force_lower_fill"}:
        return "force", False
    return "off", bool(raw)


def rhs1_xblock_local_solve_tuning(
    *,
    drop_tol_env_value: str = "",
    drop_rel_env_value: str = "",
    ilu_drop_tol_env_value: str = "",
    fill_factor_env_value: str = "",
    row_nnz_cap_env_value: str = "",
    compact_row_nnz_cap_env_value: str = "",
) -> RHS1XBlockLocalSolveTuning:
    """Parse legacy x-block local sparse-factor tuning without changing defaults."""
    return RHS1XBlockLocalSolveTuning(
        drop_tol=_parse_finite_float_value(drop_tol_env_value, DEFAULT_RHS1_XBLOCK_LOCAL_DROP_TOL),
        drop_rel=_parse_finite_float_value(drop_rel_env_value, DEFAULT_RHS1_XBLOCK_LOCAL_DROP_REL),
        ilu_drop_tol=_parse_finite_float_value(
            ilu_drop_tol_env_value,
            DEFAULT_RHS1_XBLOCK_LOCAL_ILU_DROP_TOL,
        ),
        fill_factor=_parse_finite_float_value(
            fill_factor_env_value,
            DEFAULT_RHS1_XBLOCK_LOCAL_FILL_FACTOR,
            minimum=1.0,
        ),
        row_nnz_cap=_parse_nonnegative_int_value(row_nnz_cap_env_value, DEFAULT_RHS1_XBLOCK_LOCAL_ROW_NNZ_CAP),
        compact_row_nnz_cap=_parse_nonnegative_int_value(
            compact_row_nnz_cap_env_value,
            DEFAULT_RHS1_XBLOCK_LOCAL_COMPACT_ROW_NNZ_CAP,
        ),
    )


def rhs1_xblock_lower_fill_local_solve_tuning(
    *,
    drop_tol_env_value: str = "",
    drop_rel_env_value: str = "",
    ilu_drop_tol_env_value: str = "",
    fill_factor_env_value: str = "",
    row_nnz_cap_env_value: str = "",
    compact_row_nnz_cap_env_value: str = "",
) -> RHS1XBlockLocalSolveTuning:
    """Parse bounded lower-fill x-block ILU tuning."""
    return RHS1XBlockLocalSolveTuning(
        drop_tol=_parse_finite_float_value(drop_tol_env_value, DEFAULT_RHS1_XBLOCK_LOWER_FILL_DROP_TOL),
        drop_rel=_parse_finite_float_value(drop_rel_env_value, DEFAULT_RHS1_XBLOCK_LOWER_FILL_DROP_REL),
        ilu_drop_tol=_parse_finite_float_value(
            ilu_drop_tol_env_value,
            DEFAULT_RHS1_XBLOCK_LOWER_FILL_ILU_DROP_TOL,
        ),
        fill_factor=_parse_finite_float_value(
            fill_factor_env_value,
            DEFAULT_RHS1_XBLOCK_LOWER_FILL_FACTOR,
            minimum=1.0,
        ),
        row_nnz_cap=_parse_nonnegative_int_value(row_nnz_cap_env_value, DEFAULT_RHS1_XBLOCK_LOWER_FILL_ROW_NNZ_CAP),
        compact_row_nnz_cap=_parse_nonnegative_int_value(
            compact_row_nnz_cap_env_value,
            DEFAULT_RHS1_XBLOCK_LOWER_FILL_COMPACT_ROW_NNZ_CAP,
        ),
    )


def rhs1_xblock_local_solve_metadata_label(*, factorization: str, lower_fill: bool) -> str:
    """Return stable trace labels for x-block local factor candidates."""
    factorization_norm = _normalize_policy_token(factorization)
    if bool(lower_fill):
        return "rhs1_xblock_local_lower_fill_ilu"
    if factorization_norm == "lu":
        return "rhs1_xblock_local_exact_lu"
    return "rhs1_xblock_local_ilu"


def rhs1_xblock_local_solve_candidate(
    *,
    block_size: int,
    lu_max: int,
    lower_fill_env_value: str = "",
    drop_tol_env_value: str = "",
    drop_rel_env_value: str = "",
    ilu_drop_tol_env_value: str = "",
    fill_factor_env_value: str = "",
    row_nnz_cap_env_value: str = "",
    compact_row_nnz_cap_env_value: str = "",
    lower_fill_drop_tol_env_value: str = "",
    lower_fill_drop_rel_env_value: str = "",
    lower_fill_ilu_drop_tol_env_value: str = "",
    lower_fill_factor_env_value: str = "",
    lower_fill_row_nnz_cap_env_value: str = "",
    lower_fill_compact_row_nnz_cap_env_value: str = "",
    lower_fill_max_block_size_env_value: str = "",
) -> RHS1XBlockLocalSolveCandidate:
    """Select exact-LU, legacy ILU, or bounded lower-fill ILU for one local block.

    Blank/default lower-fill controls return the same exact-LU/ILU split used by
    the current driver: exact LU when ``block_size <= lu_max`` and ILU otherwise.
    """
    block_size_i = _parse_nonnegative_int_value(block_size, 0)
    lu_max_i = _parse_nonnegative_int_value(lu_max, 0)
    exact_lu = bool(block_size_i <= lu_max_i)
    mode, ignored_lower_fill_env = rhs1_xblock_lower_fill_mode(lower_fill_env_value)
    lower_fill_requested = mode in {"probe", "force"}
    lower_fill_max_block_size = _parse_nonnegative_int_value(
        lower_fill_max_block_size_env_value,
        DEFAULT_RHS1_XBLOCK_LOWER_FILL_MAX_BLOCK_SIZE,
    )
    lower_fill_block_size_capped = bool(
        lower_fill_requested
        and lower_fill_max_block_size > 0
        and block_size_i > int(lower_fill_max_block_size)
    )
    use_lower_fill = bool(
        lower_fill_requested
        and not lower_fill_block_size_capped
        and (mode == "force" or not exact_lu)
    )

    if use_lower_fill:
        factorization = "ilu"
        tuning = rhs1_xblock_lower_fill_local_solve_tuning(
            drop_tol_env_value=lower_fill_drop_tol_env_value,
            drop_rel_env_value=lower_fill_drop_rel_env_value,
            ilu_drop_tol_env_value=lower_fill_ilu_drop_tol_env_value,
            fill_factor_env_value=lower_fill_factor_env_value,
            row_nnz_cap_env_value=lower_fill_row_nnz_cap_env_value,
            compact_row_nnz_cap_env_value=lower_fill_compact_row_nnz_cap_env_value,
        )
        selection_reason = "lower-fill-forced" if mode == "force" and exact_lu else "lower-fill-requested"
        label = rhs1_xblock_local_solve_metadata_label(factorization=factorization, lower_fill=True)
    else:
        factorization = "lu" if exact_lu else "ilu"
        tuning = rhs1_xblock_local_solve_tuning(
            drop_tol_env_value=drop_tol_env_value,
            drop_rel_env_value=drop_rel_env_value,
            ilu_drop_tol_env_value=ilu_drop_tol_env_value,
            fill_factor_env_value=fill_factor_env_value,
            row_nnz_cap_env_value=row_nnz_cap_env_value,
            compact_row_nnz_cap_env_value=compact_row_nnz_cap_env_value,
        )
        if ignored_lower_fill_env:
            selection_reason = "lower-fill-env-ignored"
        elif lower_fill_block_size_capped:
            selection_reason = "lower-fill-block-size-cap-exceeded"
        elif lower_fill_requested and exact_lu and mode != "force":
            selection_reason = "exact-lu-within-lu-max"
        else:
            selection_reason = "legacy-exact-lu" if exact_lu else "legacy-ilu"
        label = rhs1_xblock_local_solve_metadata_label(
            factorization=factorization,
            lower_fill=False,
        )

    return RHS1XBlockLocalSolveCandidate(
        block_size=int(block_size_i),
        lu_max=int(lu_max_i),
        mode=mode,
        factorization=factorization,
        tuning=tuning,
        metadata_label=label,
        selection_reason=selection_reason,
        exact_lu=bool(factorization == "lu"),
        lower_fill=bool(use_lower_fill),
        lower_fill_requested=bool(lower_fill_requested),
        ignored_lower_fill_env=bool(ignored_lower_fill_env),
        lower_fill_max_block_size=int(lower_fill_max_block_size),
        lower_fill_block_size_capped=bool(lower_fill_block_size_capped),
    )


def rhs1_xblock_lower_fill_acceptance_decision(
    *,
    factorization_ok: bool,
    residual_norm: float | None,
    target: float | None,
    baseline_residual_norm: float | None = None,
    factor_probe_ratio: float | None = None,
    max_residual_ratio_env_value: str = "",
    min_improvement_env_value: str = "",
    factor_probe_max_env_value: str = "",
    candidate_metadata_label: str = "rhs1_xblock_local_lower_fill_ilu",
) -> RHS1XBlockLowerFillAcceptance:
    """Return whether a probed bounded lower-fill local candidate is acceptable."""
    max_residual_ratio = _parse_finite_float_value(
        max_residual_ratio_env_value,
        DEFAULT_RHS1_XBLOCK_LOWER_FILL_ACCEPT_MAX_RESIDUAL_RATIO,
        minimum=1.0,
    )
    min_improvement = _parse_finite_float_value(
        min_improvement_env_value,
        DEFAULT_RHS1_XBLOCK_LOWER_FILL_ACCEPT_MIN_IMPROVEMENT,
        minimum=0.0,
    )
    factor_probe_max = _parse_finite_float_value(
        factor_probe_max_env_value,
        DEFAULT_RHS1_XBLOCK_LOWER_FILL_FACTOR_PROBE_MAX,
        minimum=1.0,
    )

    def _decision(
        *,
        accepted: bool,
        reason: str,
        residual_ratio: float | None = None,
        improvement: float | None = None,
        factor_probe_ratio_use: float | None = None,
    ) -> RHS1XBlockLowerFillAcceptance:
        return RHS1XBlockLowerFillAcceptance(
            accepted=bool(accepted),
            reason=reason,
            metadata_label="rhs1_xblock_lower_fill_acceptance",
            candidate_metadata_label=str(candidate_metadata_label),
            residual_ratio=residual_ratio,
            improvement=improvement,
            factor_probe_ratio=factor_probe_ratio_use,
            max_residual_ratio=float(max_residual_ratio),
            min_improvement=float(min_improvement),
            factor_probe_max=float(factor_probe_max),
        )

    if not bool(factorization_ok):
        return _decision(accepted=False, reason="factorization-failed")

    factor_probe_ratio_f = _finite_float_or_none(factor_probe_ratio)
    if factor_probe_ratio is not None:
        if factor_probe_ratio_f is None or factor_probe_ratio_f < 0.0:
            return _decision(
                accepted=False,
                reason="nonfinite-factor-probe",
                factor_probe_ratio_use=factor_probe_ratio_f,
            )
        if factor_probe_ratio_f > factor_probe_max:
            return _decision(
                accepted=False,
                reason="factor-probe-limit-exceeded",
                factor_probe_ratio_use=factor_probe_ratio_f,
            )

    residual = _finite_float_or_none(residual_norm)
    if residual is None or residual < 0.0:
        return _decision(
            accepted=False,
            reason="nonfinite-residual",
            factor_probe_ratio_use=factor_probe_ratio_f,
        )

    target_f = _finite_float_or_none(target)
    residual_ratio = None
    if target_f is not None and target_f > 0.0:
        residual_ratio = float(residual) / max(float(target_f), 1.0e-300)
        if residual_ratio > max_residual_ratio:
            return _decision(
                accepted=False,
                reason="residual-ratio-limit-exceeded",
                residual_ratio=residual_ratio,
                factor_probe_ratio_use=factor_probe_ratio_f,
            )

    baseline = _finite_float_or_none(baseline_residual_norm)
    improvement = None
    if baseline is not None and baseline > 0.0 and min_improvement > 0.0:
        improvement = float(baseline) / max(float(residual), 1.0e-300)
        if improvement < min_improvement:
            return _decision(
                accepted=False,
                reason="insufficient-improvement",
                residual_ratio=residual_ratio,
                improvement=improvement,
                factor_probe_ratio_use=factor_probe_ratio_f,
            )

    return _decision(
        accepted=True,
        reason="accepted",
        residual_ratio=residual_ratio,
        improvement=improvement,
        factor_probe_ratio_use=factor_probe_ratio_f,
    )


def _full_fp_3d_right_pc_max_active_size(env_value: str) -> int:
    """Return the full-FP 3D active-size limit for default right preconditioning."""
    raw = str(env_value).strip()
    if not raw:
        return DEFAULT_FULL_FP_3D_RIGHT_PC_MAX_ACTIVE_SIZE
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_FULL_FP_3D_RIGHT_PC_MAX_ACTIVE_SIZE


def _active_size_allows_full_fp_3d_right_pc(active_size: int | None, max_active_size: int) -> bool:
    """Gate right-PC defaults to the measured full-FP 3D window."""
    if active_size is None:
        return True
    try:
        return int(active_size) <= int(max_active_size)
    except (TypeError, ValueError):
        return True


def rhs1_xblock_side_probe_min_active_size(env_value: str) -> int:
    """Return the active-size floor for the 3D full-FP side probe."""
    raw = str(env_value).strip()
    if not raw:
        return DEFAULT_FULL_FP_3D_SIDE_PROBE_MIN_ACTIVE_SIZE
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_FULL_FP_3D_SIDE_PROBE_MIN_ACTIVE_SIZE


def rhs1_xblock_side_probe_enabled(
    *,
    env_value: str,
    explicit_side_env_value: str,
    full_fp_3d_pc: bool,
    active_size: int | None,
    min_active_size_env_value: str,
    krylov_method: str,
    precondition_side: str,
) -> bool:
    """Return whether to run the bounded precondition-side probe.

    The automatic probe is deliberately scoped to larger 3D full-FP QI-like
    systems on the host GMRES path, where bounded evidence has shown
    seed-dependent left/right slow modes. Device-resident Krylov methods can
    still opt in explicitly, but they skip the default probe because the probe
    duplicates expensive device setup and can consume the entire bounded GPU
    budget before the actual solve starts.
    """
    raw = str(env_value).strip().lower()
    if raw in {"0", "false", "f", "no", "off", ".false.", ".f."}:
        return False
    explicit_side = str(explicit_side_env_value).strip().lower()
    forced_on = raw in {"1", "true", "t", "yes", "on", ".true.", ".t."}
    if explicit_side in {"left", "right", "none"} and not forced_on:
        return False
    method = str(krylov_method).strip().lower()
    side = str(precondition_side).strip().lower()
    explicit_allowed_methods = {"gmres", "fgmres_jax", "gmres_jax"}
    default_allowed_methods = {"gmres"}
    if method not in explicit_allowed_methods or side not in {"left", "right"}:
        return False
    if forced_on:
        return True
    if raw not in {"", "auto", "default"}:
        return False
    if method not in default_allowed_methods:
        return False
    if not bool(full_fp_3d_pc):
        return False
    min_active_size = rhs1_xblock_side_probe_min_active_size(min_active_size_env_value)
    try:
        return int(active_size) >= int(min_active_size)
    except (TypeError, ValueError):
        return False


def rhs1_xblock_side_probe_should_switch(
    *,
    residual_ratio: float | None,
    switch_ratio_env_value: str,
) -> bool:
    """Return whether a default-side probe is weak enough to try the other side."""
    raw = str(switch_ratio_env_value).strip()
    try:
        threshold = float(raw) if raw else DEFAULT_FULL_FP_3D_SIDE_PROBE_SWITCH_RATIO
    except ValueError:
        threshold = DEFAULT_FULL_FP_3D_SIDE_PROBE_SWITCH_RATIO
    threshold = max(1.0, float(threshold))
    if residual_ratio is None:
        return False
    try:
        value = float(residual_ratio)
    except (TypeError, ValueError):
        return False
    return bool(value == value and value not in {float("inf"), float("-inf")} and value > threshold)


def rhs1_xblock_lgmres_rescue_enabled(*, env_value: str, krylov_env_value: str) -> bool:
    """Return whether a weak large-QI GMRES probe may switch to LGMRES.

    Explicit Krylov method requests are treated as user intent and are not
    rewritten by the automatic rescue. Users can still force this rescue with
    ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_LGMRES_RESCUE=1``.
    """
    raw = str(env_value).strip().lower()
    if raw in {"0", "false", "f", "no", "off", ".false.", ".f."}:
        return False
    if raw in {"1", "true", "t", "yes", "on", ".true.", ".t."}:
        return True
    if raw not in {"", "auto", "default"}:
        return False
    method_env = str(krylov_env_value).strip().lower().replace("-", "_")
    return method_env in {"", "auto", "default"}


def rhs1_xblock_lgmres_rescue_backend_allowed(*, backend: str, env_value: str) -> bool:
    """Return whether the host-oriented LGMRES rescue is allowed on this backend."""
    raw = str(env_value).strip().lower()
    if raw in {"0", "false", "f", "no", "off", ".false.", ".f."}:
        return False
    if raw in {"1", "true", "t", "yes", "on", ".true.", ".t."}:
        return True
    return str(backend).strip().lower() == "cpu"


def rhs1_xblock_device_host_fallback_decision(
    *,
    env_value: str,
    requested_krylov_method: str,
    active_size: int | None,
    min_active_size_env_value: str,
    rhs_mode: int,
    constraint_scheme: int,
    include_phi1: bool,
    has_fp: bool,
    has_pas: bool,
    n_zeta: int,
) -> RHS1XBlockDeviceHostFallbackDecision:
    """Return whether to replace a device-Krylov x-block solve by the host policy.

    This is deliberately scoped to the measured hard-seed family: large
    RHSMode=1, ConstraintScheme=1, three-dimensional full-FP systems without
    Phi1. The replacement is non-autodiff and intended as a robust production
    fallback when the user has requested a JAX-native device Krylov method. It
    enters the host x-block auto policy rather than direct LGMRES, so the
    measured side-probe seed and LGMRES rescue remain available.
    """
    requested_method = _normalize_policy_token(requested_krylov_method)
    device_methods = {"fgmres_jax", "gmres_jax", "bicgstab_jax", "tfqmr_jax"}
    requested_device = requested_method in device_methods
    raw_mode = _normalize_policy_token(env_value)
    ignored_env = False
    if raw_mode in _FALSE_VALUES or raw_mode in {"disabled", "disable"}:
        mode = "off"
    elif raw_mode in _TRUE_VALUES or raw_mode in {
        "force",
        "forced",
        "host",
        "host_fallback",
        "cpu",
        "non_autodiff",
        "nonautodiff",
        "scipy",
        "lgmres",
    }:
        mode = "force"
    elif raw_mode in _DEFAULT_VALUES:
        mode = "auto"
    else:
        mode = "auto"
        ignored_env = bool(raw_mode)

    min_active_size = _parse_nonnegative_int_value(
        min_active_size_env_value,
        DEFAULT_FULL_FP_3D_DEVICE_HOST_FALLBACK_MIN_ACTIVE_SIZE,
    )
    qi_like_full_fp_3d = bool(
        int(rhs_mode) == 1
        and int(constraint_scheme) == 1
        and not bool(include_phi1)
        and bool(has_fp)
        and not bool(has_pas)
        and int(n_zeta) > 1
    )
    try:
        active_ok = int(active_size) >= int(min_active_size)
    except (TypeError, ValueError):
        active_ok = False

    used = False
    reason = "disabled"
    if not requested_device:
        reason = "not-device-krylov"
    elif mode == "off":
        reason = "disabled"
    elif mode == "force":
        used = True
        reason = "forced"
    elif not qi_like_full_fp_3d:
        reason = "not-large-qi-full-fp-3d"
    elif not active_ok:
        reason = "below-active-size-floor"
    else:
        used = True
        reason = "large-qi-full-fp-3d"

    return RHS1XBlockDeviceHostFallbackDecision(
        mode=mode,
        used=bool(used),
        reason=reason,
        requested_device_krylov=bool(requested_device),
        requested_method=requested_method,
        effective_krylov_env_value="auto" if bool(used) else requested_method,
        min_active_size=int(min_active_size),
        qi_like_full_fp_3d=bool(qi_like_full_fp_3d),
        ignored_env=bool(ignored_env),
    )


def rhs1_xblock_qi_device_operator_reuse_decision(
    *,
    env_value: str,
    requested_krylov_method: str,
    host_fallback_used: bool,
    rhs_mode: int,
    constraint_scheme: int,
    include_phi1: bool,
    has_fp: bool,
    has_pas: bool,
    n_zeta: int,
    qi_device_preconditioner_requested: bool,
    qi_device_matrix_free_requested: bool,
    qi_device_use_in_krylov_requested: bool,
    precondition_side: str,
) -> RHS1XBlockQIDeviceOperatorReuseDecision:
    """Return whether QI-device Krylov can skip local x-block factors.

    This is deliberately narrower than the host-fallback policy: it only
    activates for explicit matrix-free QI-device Krylov requests, because those
    runs already build and reuse ``(Q, A Q)`` coarse actions on device. In that
    configuration, building local host/JAX x-block factors first can dominate
    setup time and memory without improving the final device route.
    """
    raw_mode = _normalize_policy_token(env_value)
    ignored_env = False
    if raw_mode in _FALSE_VALUES or raw_mode in {"disabled", "disable"}:
        mode = "off"
    elif raw_mode in _TRUE_VALUES or raw_mode in {"force", "forced", "device", "reuse"}:
        mode = "force"
    elif raw_mode in _DEFAULT_VALUES:
        mode = "auto"
    else:
        mode = "auto"
        ignored_env = bool(raw_mode)

    requested_method = _normalize_policy_token(requested_krylov_method)
    requested_device = requested_method in {"fgmres_jax", "gmres_jax", "bicgstab_jax", "tfqmr_jax"}
    side = _normalize_policy_token(precondition_side)
    qi_like_full_fp_3d = bool(
        int(rhs_mode) == 1
        and int(constraint_scheme) == 1
        and not bool(include_phi1)
        and bool(has_fp)
        and not bool(has_pas)
        and int(n_zeta) > 1
    )
    requested = bool(
        requested_device
        and bool(qi_device_preconditioner_requested)
        and bool(qi_device_matrix_free_requested)
        and bool(qi_device_use_in_krylov_requested)
    )

    enabled = False
    reason = "disabled"
    if ignored_env:
        reason = "ignored-env-auto"
    if mode == "off":
        reason = "disabled"
    elif not requested_device:
        reason = "not-device-krylov"
    elif bool(host_fallback_used):
        reason = "host-fallback-active"
    elif not bool(qi_device_preconditioner_requested):
        reason = "qi-device-preconditioner-not-requested"
    elif not bool(qi_device_matrix_free_requested):
        reason = "matrix-free-not-requested"
    elif not bool(qi_device_use_in_krylov_requested):
        reason = "use-in-krylov-not-requested"
    elif side == "none":
        reason = "precondition-side-none"
    elif mode == "force":
        enabled = True
        reason = "forced"
    elif not qi_like_full_fp_3d:
        reason = "not-qi-full-fp-3d"
    else:
        enabled = True
        reason = "matrix-free-qi-device-krylov"

    return RHS1XBlockQIDeviceOperatorReuseDecision(
        enabled=bool(enabled),
        reason=reason,
        requested=bool(requested),
        skip_xblock_factors=bool(enabled),
        requested_device_krylov=bool(requested_device),
        matrix_free=bool(qi_device_matrix_free_requested),
        use_in_krylov=bool(qi_device_use_in_krylov_requested),
        precondition_side=side,
        qi_like_full_fp_3d=bool(qi_like_full_fp_3d),
    )


def rhs1_xblock_lgmres_rescue_maxiter(env_value: str, current_maxiter: int) -> tuple[int, bool]:
    """Return the bounded LGMRES-rescue outer-iteration limit and cap flag."""
    try:
        requested = int(current_maxiter)
    except (TypeError, ValueError):
        requested = DEFAULT_FULL_FP_3D_LGMRES_RESCUE_MAXITER
    requested = max(1, int(requested))
    raw = str(env_value).strip()
    if raw:
        try:
            selected = max(1, int(raw))
        except ValueError:
            selected = min(requested, DEFAULT_FULL_FP_3D_LGMRES_RESCUE_MAXITER)
    else:
        selected = min(requested, DEFAULT_FULL_FP_3D_LGMRES_RESCUE_MAXITER)
    return selected, bool(selected != requested)


def rhs1_xblock_lgmres_rescue_outer_k(env_value: str) -> int:
    """Return the LGMRES augmentation-space size for the large-QI rescue."""
    raw = str(env_value).strip()
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            return DEFAULT_FULL_FP_3D_LGMRES_RESCUE_OUTER_K
    return DEFAULT_FULL_FP_3D_LGMRES_RESCUE_OUTER_K


def rhs1_xblock_precondition_side(
    *,
    env_value: str,
    tokamak_fp_er_pc: bool,
    full_fp_3d_pc: bool = False,
    active_size: int | None = None,
    full_fp_3d_right_pc_max_env_value: str = "",
    use_dkes: bool,
    include_xdot: bool,
    include_electric_field_xi: bool,
) -> tuple[str, bool]:
    """Return the x-block sparse-PC side and whether right-PC was auto-selected.

    The measured production-floor GPU tokamak full-FP Er full-trajectory row
    and the bounded scale-0.50 3D full-FP QI lane are Krylov dominated and
    benefit from right preconditioning. Larger 3D full-FP QI cases can enter a
    seed-dependent right-PC slow mode, so the 3D default is capped by active
    system size and remains overrideable through
    ``SFINCS_JAX_GMRES_PRECONDITION_SIDE``.
    """
    env_side = str(env_value).strip().lower()
    if env_side in {"left", "right", "none"}:
        return env_side, False
    full_trajectory = bool(include_xdot) or bool(include_electric_field_xi)
    base_path = bool((not bool(use_dkes)) and full_trajectory)
    full_fp_3d_right_pc_max = _full_fp_3d_right_pc_max_active_size(full_fp_3d_right_pc_max_env_value)
    default_right = bool(
        base_path
        and (
            bool(tokamak_fp_er_pc)
            or (
                bool(full_fp_3d_pc)
                and _active_size_allows_full_fp_3d_right_pc(active_size, full_fp_3d_right_pc_max)
            )
        )
    )
    return ("right" if default_right else "left"), default_right


def rhs1_xblock_krylov_method(env_value: str) -> tuple[str, bool]:
    """Canonicalize the x-block sparse-PC Krylov method env value.

    Returns ``(method, ignored_unknown)`` so the driver can preserve its
    historical warning while keeping this normalization pure and directly
    testable.
    """
    env_method = str(env_value).strip().lower()
    method = env_method.replace("-", "_") if env_method else "gmres"
    if method in {"default", "auto"}:
        return "gmres", False
    if method in {"short_recurrence", "shortrecurrence"}:
        return "bicgstab", False
    if method == "lgmres_scipy":
        return "lgmres", False
    if method in {"gcrot", "gcrotmk", "gcrot_mk"}:
        return "gcrotmk", False
    if method in {"gmres_jax", "jax_gmres", "device_gmres"}:
        return "gmres_jax", False
    if method in {"bicgstab_jax", "device_bicgstab", "short_recurrence_jax", "shortrecurrence_jax"}:
        return "bicgstab_jax", False
    if method in {"tfqmr", "tfqmr_jax", "device_tfqmr", "transpose_free_qmr", "transposefree_qmr"}:
        return "tfqmr_jax", False
    if method in {"fgmres", "fgmres_jax", "flexible_gmres", "flexiblegmres"}:
        return "fgmres_jax", False
    if method in {"gmres", "lgmres", "bicgstab"}:
        return method, False
    return "gmres", bool(env_method)


def rhs1_xblock_gmres_restart(
    *,
    requested_restart: int,
    restart_env_value: str,
    krylov_method: str,
    default_right_preconditioned: bool,
    short_restart_default: bool | None = None,
) -> tuple[int, bool]:
    """Return the x-block sparse-PC GMRES restart and whether it was auto-capped.

    The production-floor GPU full-FP Er full-trajectory row converges faster
    with a short restarted GMRES basis once the x-block preconditioner is applied
    on the right. Keep this cap restricted to the measured auto-selected policy;
    explicit user restart overrides and other trajectory branches remain
    untouched.
    """
    restart_use = max(1, int(requested_restart))
    if str(restart_env_value).strip():
        return restart_use, False
    if str(krylov_method).strip().lower() != "gmres":
        return restart_use, False
    short_restart_default = bool(default_right_preconditioned) if short_restart_default is None else bool(
        short_restart_default
    )
    if not short_restart_default:
        return restart_use, False
    capped = min(restart_use, 20)
    return capped, bool(capped != restart_use)


def resolve_rhs1_xblock_sparse_pc_policy(
    *,
    precondition_side_env_value: str,
    krylov_env_value: str,
    requested_restart: int,
    restart_env_value: str,
    tokamak_fp_er_pc: bool,
    full_fp_3d_pc: bool = False,
    active_size: int | None = None,
    full_fp_3d_right_pc_max_env_value: str = "",
    use_dkes: bool,
    include_xdot: bool,
    include_electric_field_xi: bool,
) -> RHS1XBlockSparsePCPolicy:
    """Resolve the full x-block sparse-PC policy used by ``v3_driver.py``."""
    precondition_side, default_right_preconditioned = rhs1_xblock_precondition_side(
        env_value=precondition_side_env_value,
        tokamak_fp_er_pc=tokamak_fp_er_pc,
        full_fp_3d_pc=full_fp_3d_pc,
        active_size=active_size,
        full_fp_3d_right_pc_max_env_value=full_fp_3d_right_pc_max_env_value,
        use_dkes=use_dkes,
        include_xdot=include_xdot,
        include_electric_field_xi=include_electric_field_xi,
    )
    krylov_method, ignored_krylov_env = rhs1_xblock_krylov_method(krylov_env_value)
    gmres_restart, restart_capped = rhs1_xblock_gmres_restart(
        requested_restart=requested_restart,
        restart_env_value=restart_env_value,
        krylov_method=krylov_method,
        default_right_preconditioned=default_right_preconditioned,
        short_restart_default=bool(tokamak_fp_er_pc),
    )
    return RHS1XBlockSparsePCPolicy(
        precondition_side=precondition_side,
        default_right_preconditioned=default_right_preconditioned,
        krylov_method=krylov_method,
        ignored_krylov_env=ignored_krylov_env,
        gmres_restart=gmres_restart,
        restart_capped=restart_capped,
    )


__all__ = [
    "DEFAULT_FULL_FP_3D_DEVICE_HOST_FALLBACK_MIN_ACTIVE_SIZE",
    "DEFAULT_FULL_FP_3D_RIGHT_PC_MAX_ACTIVE_SIZE",
    "DEFAULT_FULL_FP_3D_LGMRES_RESCUE_MAXITER",
    "DEFAULT_FULL_FP_3D_LGMRES_RESCUE_OUTER_K",
    "DEFAULT_FULL_FP_3D_SIDE_PROBE_MIN_ACTIVE_SIZE",
    "DEFAULT_FULL_FP_3D_SIDE_PROBE_SWITCH_RATIO",
    "DEFAULT_RHS1_XBLOCK_LOCAL_COMPACT_ROW_NNZ_CAP",
    "DEFAULT_RHS1_XBLOCK_LOCAL_DROP_REL",
    "DEFAULT_RHS1_XBLOCK_LOCAL_DROP_TOL",
    "DEFAULT_RHS1_XBLOCK_LOCAL_FILL_FACTOR",
    "DEFAULT_RHS1_XBLOCK_LOCAL_ILU_DROP_TOL",
    "DEFAULT_RHS1_XBLOCK_LOCAL_ROW_NNZ_CAP",
    "DEFAULT_RHS1_XBLOCK_LOWER_FILL_ACCEPT_MAX_RESIDUAL_RATIO",
    "DEFAULT_RHS1_XBLOCK_LOWER_FILL_ACCEPT_MIN_IMPROVEMENT",
    "DEFAULT_RHS1_XBLOCK_LOWER_FILL_COMPACT_ROW_NNZ_CAP",
    "DEFAULT_RHS1_XBLOCK_LOWER_FILL_DROP_REL",
    "DEFAULT_RHS1_XBLOCK_LOWER_FILL_DROP_TOL",
    "DEFAULT_RHS1_XBLOCK_LOWER_FILL_FACTOR",
    "DEFAULT_RHS1_XBLOCK_LOWER_FILL_FACTOR_PROBE_MAX",
    "DEFAULT_RHS1_XBLOCK_LOWER_FILL_ILU_DROP_TOL",
    "DEFAULT_RHS1_XBLOCK_LOWER_FILL_MAX_BLOCK_SIZE",
    "DEFAULT_RHS1_XBLOCK_LOWER_FILL_ROW_NNZ_CAP",
    "RHS1XBlockDeviceHostFallbackDecision",
    "RHS1XBlockLocalSolveCandidate",
    "RHS1XBlockLocalSolveTuning",
    "RHS1XBlockLowerFillAcceptance",
    "RHS1XBlockSparsePCPolicy",
    "resolve_rhs1_xblock_sparse_pc_policy",
    "rhs1_xblock_device_host_fallback_decision",
    "rhs1_xblock_qi_device_operator_reuse_decision",
    "rhs1_xblock_gmres_restart",
    "rhs1_xblock_krylov_method",
    "rhs1_xblock_local_solve_candidate",
    "rhs1_xblock_local_solve_metadata_label",
    "rhs1_xblock_local_solve_tuning",
    "rhs1_xblock_lower_fill_acceptance_decision",
    "rhs1_xblock_lower_fill_local_solve_tuning",
    "rhs1_xblock_lower_fill_mode",
    "rhs1_xblock_lgmres_rescue_backend_allowed",
    "rhs1_xblock_lgmres_rescue_enabled",
    "rhs1_xblock_lgmres_rescue_maxiter",
    "rhs1_xblock_lgmres_rescue_outer_k",
    "rhs1_xblock_precondition_side",
    "rhs1_xblock_side_probe_enabled",
    "rhs1_xblock_side_probe_min_active_size",
    "rhs1_xblock_side_probe_should_switch",
]
