from __future__ import annotations

from collections.abc import Callable
import os
from typing import Any

import numpy as np


def transport_dense_backend_allowed(*, backend: str) -> bool:
    env = os.environ.get("SFINCS_JAX_TRANSPORT_DENSE_ALLOW_ACCELERATOR", "").strip().lower()
    if env in {"1", "true", "yes", "on"}:
        return True
    if env in {"0", "false", "no", "off"}:
        return False
    return str(backend).strip().lower() == "cpu"


def transport_dense_accelerator_auto_allowed(
    op: Any,
    *,
    backend: str,
    geometry_scheme: int,
) -> bool:
    """Allow bounded GPU dense transport solves for measured monoenergetic cases."""
    backend_key = str(backend).strip().lower()
    if backend_key == "cpu":
        return False
    if backend_key not in {"gpu", "cuda"}:
        return False
    broad_env = os.environ.get("SFINCS_JAX_TRANSPORT_DENSE_ALLOW_ACCELERATOR", "").strip().lower()
    if broad_env in {"0", "false", "no", "off"}:
        return False
    auto_env = os.environ.get("SFINCS_JAX_TRANSPORT_DENSE_ACCELERATOR_AUTO", "").strip().lower()
    if auto_env in {"0", "false", "no", "off"}:
        return False
    if bool(getattr(op, "include_phi1", False)):
        return False
    if int(getattr(op, "rhs_mode", 0) or 0) != 3:
        return False
    if getattr(getattr(op, "fblock", None), "fp", None) is not None:
        return False
    if int(getattr(op, "n_x", 0) or 0) > 2:
        return False
    allowed_geom_env = os.environ.get("SFINCS_JAX_TRANSPORT_DENSE_ACCELERATOR_AUTO_GEOMETRIES", "").strip()
    allowed_geometries: set[int] = {1}
    if allowed_geom_env:
        parsed: set[int] = set()
        for raw in allowed_geom_env.split(","):
            raw = raw.strip()
            if not raw:
                continue
            try:
                parsed.add(int(raw))
            except ValueError:
                continue
        if parsed:
            allowed_geometries = parsed
    if int(geometry_scheme) not in allowed_geometries:
        return False
    max_env = os.environ.get("SFINCS_JAX_TRANSPORT_DENSE_ACCELERATOR_AUTO_MAX", "").strip()
    try:
        max_size = int(max_env) if max_env else 2500
    except ValueError:
        max_size = 2500
    if int(getattr(op, "total_size", 0) or 0) > max(1, int(max_size)):
        return False
    n_theta = int(getattr(op, "n_theta", 0) or 0)
    n_zeta = int(getattr(op, "n_zeta", 0) or 0)
    return n_theta > 1 and n_zeta > 1 and n_theta * n_zeta >= 64


def transport_tzfft_backend_allowed(*, backend: str) -> bool:
    env = os.environ.get("SFINCS_JAX_TRANSPORT_TZFFT_ALLOW_ACCELERATOR", "").strip().lower()
    if env in {"1", "true", "yes", "on"}:
        return True
    if env in {"0", "false", "no", "off"}:
        return False
    return str(backend).strip().lower() == "cpu"


def transport_tzfft_accelerator_auto_allowed(op: Any, *, backend: str) -> bool:
    """Allow accelerator tzfft only for bounded collisionless transport branches."""
    if str(backend).strip().lower() == "cpu":
        return True
    n_theta = int(getattr(op, "n_theta", 0) or 0)
    n_zeta = int(getattr(op, "n_zeta", 0) or 0)
    total_size = int(getattr(op, "total_size", 0) or 0)
    if bool(op.include_phi1):
        return False
    if int(op.rhs_mode) not in {2, 3}:
        return False
    if op.fblock.fp is not None:
        return False
    if int(op.n_x) > 2:
        return False
    if n_theta <= 0 or n_zeta <= 0 or total_size <= 0:
        return False
    if n_theta * n_zeta < 64:
        return False
    max_env = os.environ.get("SFINCS_JAX_TRANSPORT_TZFFT_ACCELERATOR_AUTO_MAX", "").strip()
    try:
        max_size = int(max_env) if max_env else 5000
    except ValueError:
        max_size = 5000
    return total_size <= max(1, int(max_size))


def transport_sparse_direct_rescue_allowed(
    *,
    op: Any,
    size: int,
    residual_norm: float,
    target: float,
    use_implicit: bool,
    backend: str,
) -> bool:
    env = os.environ.get("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT", "").strip().lower()
    if env in {"0", "false", "no", "off"}:
        return False
    if bool(use_implicit):
        return False
    if int(op.rhs_mode) not in {2, 3} or bool(op.include_phi1):
        return False
    mono_pas_cpu_priority = (
        str(backend).strip().lower() == "cpu"
        and int(op.rhs_mode) == 3
        and getattr(op.fblock, "fp", None) is None
        and int(getattr(op, "n_x", 0) or 0) <= 2
    )
    rescue_max_env = os.environ.get("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_MAX", "").strip()
    rescue_ratio_env = os.environ.get("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_RATIO", "").strip()
    try:
        if rescue_max_env:
            rescue_max = int(rescue_max_env)
        elif mono_pas_cpu_priority:
            rescue_max = 80000
        else:
            rescue_max = 40000
    except ValueError:
        rescue_max = 40000
    try:
        if rescue_ratio_env:
            rescue_ratio = float(rescue_ratio_env)
        elif mono_pas_cpu_priority:
            rescue_ratio = 1.0e4
        else:
            rescue_ratio = 1.0e2
    except ValueError:
        rescue_ratio = 1.0e2
    if int(size) > max(1, int(rescue_max)):
        return False
    if not np.isfinite(float(residual_norm)):
        return True
    if float(target) <= 0.0:
        return True
    return float(residual_norm) > float(target) * float(rescue_ratio)


def transport_sparse_direct_rescue_first(*, sparse_direct_rescue: bool) -> bool:
    env = os.environ.get("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_FIRST", "").strip().lower()
    if env in {"0", "false", "no", "off"}:
        return False
    return bool(sparse_direct_rescue)


def transport_sparse_direct_first_attempt_allowed(
    *,
    op: Any,
    size: int,
    use_implicit: bool,
    backend: str,
) -> bool:
    if bool(use_implicit):
        return False
    if int(op.rhs_mode) not in {2, 3} or bool(op.include_phi1):
        return False
    size_int = int(size)
    backend_norm = str(backend).strip().lower()
    if backend_norm == "cpu":
        if (
            int(op.rhs_mode) == 3
            and getattr(op.fblock, "fp", None) is None
            and int(getattr(op, "n_x", 0) or 0) <= 2
        ):
            return False
        first_min_env = os.environ.get("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_FIRST_CPU_MIN", "").strip()
        first_max_env = os.environ.get("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_FIRST_CPU_MAX", "").strip()
        try:
            first_min = int(first_min_env) if first_min_env else 12000
        except ValueError:
            first_min = 12000
        if size_int < max(1, int(first_min)):
            return False
        allowed = transport_sparse_direct_rescue_allowed(
            op=op,
            size=size_int,
            residual_norm=float("nan"),
            target=1.0,
            use_implicit=use_implicit,
            backend=backend_norm,
        )
        if not allowed:
            return False
        if first_max_env:
            try:
                return size_int <= max(1, int(first_max_env))
            except ValueError:
                return True
        return True
    if backend_norm != "cpu":
        if transport_tzfft_accelerator_auto_allowed(op, backend=backend_norm):
            return False
        return transport_sparse_direct_rescue_allowed(
            op=op,
            size=size_int,
            residual_norm=float("nan"),
            target=1.0,
            use_implicit=use_implicit,
            backend=backend_norm,
        )
    return False


def transport_host_gmres_first_attempt_allowed(
    *,
    op: Any,
    size: int,
    use_implicit: bool,
    backend: str,
) -> bool:
    env = os.environ.get("SFINCS_JAX_TRANSPORT_HOST_GMRES_FIRST", "").strip().lower()
    if env in {"0", "false", "no", "off"}:
        return False
    if bool(use_implicit):
        return False
    if str(backend).strip().lower() != "cpu":
        return False
    if int(op.rhs_mode) not in {2, 3} or bool(op.include_phi1):
        return False
    if getattr(op.fblock, "fp", None) is not None:
        return False
    if int(getattr(op, "n_x", 0) or 0) > 2:
        return False
    if int(op.rhs_mode) == 3:
        max_env = os.environ.get("SFINCS_JAX_TRANSPORT_HOST_GMRES_FIRST_MAX", "").strip()
        try:
            max_size = int(max_env) if max_env else 80000
        except ValueError:
            max_size = 80000
        return int(size) <= max(1, int(max_size))
    if transport_sparse_direct_first_attempt_allowed(
        op=op,
        size=size,
        use_implicit=use_implicit,
        backend=backend,
    ):
        return False
    max_env = os.environ.get("SFINCS_JAX_TRANSPORT_HOST_GMRES_FIRST_MAX", "").strip()
    try:
        max_size = int(max_env) if max_env else 80000
    except ValueError:
        max_size = 80000
    return int(size) <= max(1, int(max_size))


def transport_host_gmres_accepts_preconditioned_residual(
    *,
    op: Any,
    true_residual_norm: float,
    target_true: float,
) -> bool:
    env = os.environ.get("SFINCS_JAX_TRANSPORT_HOST_GMRES_TRUE_RATIO", "").strip()
    try:
        if env:
            max_ratio = float(env)
        elif (
            int(op.rhs_mode) == 3
            and getattr(op.fblock, "fp", None) is None
            and int(getattr(op, "n_x", 0) or 0) <= 2
        ):
            max_ratio = 1.0e4
        else:
            max_ratio = 100.0
    except ValueError:
        max_ratio = 100.0
    max_ratio = max(1.0, float(max_ratio))
    if not np.isfinite(float(true_residual_norm)):
        return False
    if float(true_residual_norm) <= float(target_true):
        return True
    if (
        int(op.rhs_mode) == 3
        and getattr(op.fblock, "fp", None) is None
        and int(getattr(op, "n_x", 0) or 0) <= 2
    ):
        return float(true_residual_norm) <= float(target_true) * float(max_ratio)
    return float(true_residual_norm) <= float(target_true) * min(10.0, float(max_ratio))


def transport_precondition_side(*, op: Any, use_implicit: bool) -> str:
    del op, use_implicit
    env = os.environ.get("SFINCS_JAX_TRANSPORT_PRECONDITION_SIDE", "").strip().lower()
    if env in {"left", "right", "none"}:
        return env
    return "left"


def transport_disable_auto_recycle(*, op: Any, use_implicit: bool, backend: str) -> bool:
    env = os.environ.get("SFINCS_JAX_TRANSPORT_DISABLE_AUTO_RECYCLE", "").strip().lower()
    if env in {"0", "false", "no", "off"}:
        return False
    if env in {"1", "true", "yes", "on"}:
        return True
    return bool(
        (not bool(use_implicit))
        and str(backend).strip().lower() == "cpu"
        and int(op.rhs_mode) == 3
        and int(getattr(op, "constraint_scheme", -1)) == 2
        and getattr(op.fblock, "fp", None) is None
        and int(getattr(op, "n_x", 0) or 0) <= 2
    )


def transport_sparse_direct_needs_float64_retry(
    *,
    factor_dtype: np.dtype,
    residual_norm: float,
    target_true: float,
) -> bool:
    if np.dtype(factor_dtype) != np.dtype(np.float32):
        return False
    if not np.isfinite(float(residual_norm)):
        return True
    env = os.environ.get("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_FLOAT64_RETRY_RATIO", "").strip()
    try:
        ratio = float(env) if env else 10.0
    except ValueError:
        ratio = 10.0
    ratio = max(1.0, float(ratio))
    return float(residual_norm) > float(target_true) * float(ratio)


def transport_sparse_factor_dtype(
    *,
    size: int,
    use_implicit: bool,
    backend: str,
    host_sparse_factor_dtype: Callable[..., np.dtype],
) -> np.dtype:
    env = os.environ.get("SFINCS_JAX_TRANSPORT_SPARSE_FACTOR_DTYPE", "").strip().lower()
    if env == "float64":
        return np.dtype(np.float64)
    if env == "float32":
        return np.dtype(np.float32)
    factor_dtype = host_sparse_factor_dtype(
        size=int(size),
        factorization="lu",
        use_implicit=use_implicit,
    )
    if (
        factor_dtype == np.dtype(np.float32)
        and (not bool(use_implicit))
        and str(backend).strip().lower() == "cpu"
    ):
        min_env = os.environ.get("SFINCS_JAX_TRANSPORT_SPARSE_FLOAT64_MIN", "").strip()
        try:
            float64_min = int(min_env) if min_env else 30000
        except ValueError:
            float64_min = 30000
        if int(size) >= max(1, int(float64_min)):
            return np.dtype(np.float64)
    return factor_dtype


def transport_sparse_direct_use_explicit_helper(*, size: int, backend: str) -> bool:
    env = os.environ.get("SFINCS_JAX_TRANSPORT_SPARSE_HELPER", "").strip().lower()
    if env in {"0", "false", "no", "off"}:
        return False
    if env in {"1", "true", "yes", "on"}:
        return True
    min_env = os.environ.get("SFINCS_JAX_TRANSPORT_SPARSE_HELPER_CPU_MIN", "").strip()
    try:
        cpu_min = int(min_env) if min_env else 12000
    except ValueError:
        cpu_min = 12000
    if str(backend).strip().lower() != "cpu":
        return True
    return int(size) >= max(1, int(cpu_min))
