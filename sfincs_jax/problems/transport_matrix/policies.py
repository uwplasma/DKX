"""Pure transport backend and sparse-host policy helpers.

The transport driver owns operator assembly and residual evaluation; this module
only decides whether specific dense, sparse-direct, host-GMRES, and recycling
paths are eligible for a bounded transport solve.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import os
from typing import Any

import numpy as np


_TRUE_ENV = {"1", "true", "yes", "on"}
_FALSE_ENV = {"0", "false", "no", "off"}


def _env_flag(name: str) -> bool | None:
    raw = os.environ.get(name, "").strip().lower()
    if raw in _TRUE_ENV:
        return True
    if raw in _FALSE_ENV:
        return False
    return None


def transport_dense_backend_allowed(*, backend: str) -> bool:
    """Return whether the dense transport path is allowed on this backend."""
    env = _env_flag("SFINCS_JAX_TRANSPORT_DENSE_ALLOW_ACCELERATOR")
    if env is not None:
        return env
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
    if _env_flag("SFINCS_JAX_TRANSPORT_DENSE_ALLOW_ACCELERATOR") is False:
        return False
    if _env_flag("SFINCS_JAX_TRANSPORT_DENSE_ACCELERATOR_AUTO") is False:
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
    """Return whether the structured ``theta,zeta`` FFT path may run here."""
    env = _env_flag("SFINCS_JAX_TRANSPORT_TZFFT_ALLOW_ACCELERATOR")
    if env is not None:
        return env
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


def transport_tzfft_structured_first_attempt_allowed(
    op: Any,
    *,
    size: int,
    use_implicit: bool,
    backend: str,
) -> bool:
    """Return whether a bounded structured ``tzfft`` Krylov probe should run first.

    This is the production-size counterpart to
    :func:`transport_tzfft_accelerator_auto_allowed`. It intentionally stays
    narrower: automatic promotion is only for explicit RHSMode=2/3 mono/PAS
    transport rows where an exact sparse-pattern LU rescue is still available
    if the true residual gate is not met.
    """
    env = _env_flag("SFINCS_JAX_TRANSPORT_TZFFT_FIRST")
    if env is False:
        return False
    backend_norm = str(backend).strip().lower()
    force = bool(env)
    if (not force) and backend_norm not in {"gpu", "cuda"}:
        return False
    if bool(use_implicit) and not force:
        return False
    if bool(getattr(op, "include_phi1", False)):
        return False
    if int(getattr(op, "rhs_mode", 0) or 0) not in {2, 3}:
        return False
    if getattr(getattr(op, "fblock", None), "fp", None) is not None:
        return False
    if int(getattr(op, "n_x", 0) or 0) > 2:
        return False
    n_theta = int(getattr(op, "n_theta", 0) or 0)
    n_zeta = int(getattr(op, "n_zeta", 0) or 0)
    if n_theta <= 0 or n_zeta <= 0 or n_theta * n_zeta < 64:
        return False
    max_env = os.environ.get("SFINCS_JAX_TRANSPORT_TZFFT_FIRST_MAX", "").strip()
    min_env = os.environ.get("SFINCS_JAX_TRANSPORT_TZFFT_FIRST_MIN", "").strip()
    try:
        max_size = int(max_env) if max_env else 160000
    except ValueError:
        max_size = 160000
    try:
        min_size = int(min_env) if min_env else 5000
    except ValueError:
        min_size = 5000
    size_int = int(size)
    return max(1, int(min_size)) <= size_int <= max(1, int(max_size))


def transport_tzfft_first_attempt_budget(
    *,
    restart: int,
    maxiter: int | None,
) -> tuple[str, int, int]:
    """Resolve the bounded first-pass structured Krylov budget."""
    method_env = os.environ.get("SFINCS_JAX_TRANSPORT_TZFFT_FIRST_METHOD", "").strip().lower()
    method = "incremental" if method_env in {"", "incremental", "gmres"} else "incremental"
    restart_env = os.environ.get("SFINCS_JAX_TRANSPORT_TZFFT_FIRST_RESTART", "").strip()
    maxiter_env = os.environ.get("SFINCS_JAX_TRANSPORT_TZFFT_FIRST_MAXITER", "").strip()
    try:
        restart_use = int(restart_env) if restart_env else min(max(5, int(restart)), 40)
    except ValueError:
        restart_use = min(max(5, int(restart)), 40)
    try:
        maxiter_use = int(maxiter_env) if maxiter_env else min(max(1, int(maxiter or 12)), 12)
    except ValueError:
        maxiter_use = min(max(1, int(maxiter or 12)), 12)
    return method, max(5, int(restart_use)), max(1, int(maxiter_use))


def transport_sparse_direct_rescue_allowed(
    *,
    op: Any,
    size: int,
    residual_norm: float,
    target: float,
    use_implicit: bool,
    backend: str,
) -> bool:
    """Return whether sparse-direct transport rescue is admissible."""
    if _env_flag("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT") is False:
        return False
    if bool(use_implicit):
        return False
    if int(op.rhs_mode) not in {2, 3} or bool(op.include_phi1):
        return False
    backend_norm = str(backend).strip().lower()
    mono_pas_priority = (
        int(op.rhs_mode) == 3
        and getattr(op.fblock, "fp", None) is None
        and int(getattr(op, "n_x", 0) or 0) <= 2
    )
    mono_pas_cpu_priority = backend_norm == "cpu" and mono_pas_priority
    mono_pas_accelerator_priority = backend_norm in {"gpu", "cuda"} and mono_pas_priority
    rescue_max_env = os.environ.get("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_MAX", "").strip()
    rescue_ratio_env = os.environ.get("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_RATIO", "").strip()
    try:
        if rescue_max_env:
            rescue_max = int(rescue_max_env)
        elif mono_pas_accelerator_priority:
            rescue_max = 160000
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
    """Return whether an admitted sparse-direct rescue should run first."""
    if _env_flag("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_FIRST") is False:
        return False
    return bool(sparse_direct_rescue)


def transport_sparse_direct_first_attempt_allowed(
    *,
    op: Any,
    size: int,
    use_implicit: bool,
    backend: str,
) -> bool:
    """Return whether sparse-direct should be the first transport attempt."""
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
        if transport_tzfft_structured_first_attempt_allowed(
            op,
            size=size_int,
            use_implicit=use_implicit,
            backend=backend_norm,
        ):
            return False
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
    """Return whether CPU host-GMRES should be attempted before device GMRES."""
    env_flag = _env_flag("SFINCS_JAX_TRANSPORT_HOST_GMRES_FIRST")
    if env_flag is False:
        return False
    if bool(use_implicit):
        return False
    if str(backend).strip().lower() != "cpu":
        return False
    if int(op.rhs_mode) not in {2, 3} or bool(op.include_phi1):
        return False
    force = bool(env_flag)
    if int(op.rhs_mode) == 3:
        max_env = os.environ.get("SFINCS_JAX_TRANSPORT_HOST_GMRES_FIRST_MAX", "").strip()
        try:
            max_size = int(max_env) if max_env else (int(size) if force else 80000)
        except ValueError:
            max_size = int(size) if force else 80000
        if force:
            return int(size) <= max(1, int(max_size))
        if getattr(op.fblock, "fp", None) is not None:
            return False
        if int(getattr(op, "n_x", 0) or 0) > 2:
            return False
        return int(size) <= max(1, int(max_size))
    if force:
        max_env = os.environ.get("SFINCS_JAX_TRANSPORT_HOST_GMRES_FIRST_MAX", "").strip()
        try:
            max_size = int(max_env) if max_env else int(size)
        except ValueError:
            max_size = int(size)
        return int(size) <= max(1, int(max_size))
    if getattr(op.fblock, "fp", None) is not None:
        return False
    if int(getattr(op, "n_x", 0) or 0) > 2:
        return False
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
    """Return whether host GMRES may accept a preconditioned-residual success."""
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
    """Resolve the transport preconditioner side without inspecting operators."""
    del op, use_implicit
    env = os.environ.get("SFINCS_JAX_TRANSPORT_PRECONDITION_SIDE", "").strip().lower()
    if env in {"left", "right", "none"}:
        return env
    return "left"


def transport_disable_auto_recycle(*, op: Any, use_implicit: bool, backend: str) -> bool:
    """Return whether transport Krylov-state auto-recycling should be disabled."""
    env = _env_flag("SFINCS_JAX_TRANSPORT_DISABLE_AUTO_RECYCLE")
    if env is not None:
        return env
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
    """Return whether a float32 sparse-direct factor needs a float64 retry."""
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
    """Resolve the sparse transport LU factor dtype for host fallback paths."""
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
    """Return whether the explicit sparse host helper should materialize CSR."""
    env = _env_flag("SFINCS_JAX_TRANSPORT_SPARSE_HELPER")
    if env is not None:
        return env
    min_env = os.environ.get("SFINCS_JAX_TRANSPORT_SPARSE_HELPER_CPU_MIN", "").strip()
    try:
        cpu_min = int(min_env) if min_env else 12000
    except ValueError:
        cpu_min = 12000
    if str(backend).strip().lower() != "cpu":
        return True
    return int(size) >= max(1, int(cpu_min))


@dataclass(frozen=True)
class TransportRuntimePolicy:
    """Runtime-bound transport policy facade.

    The pure helpers above take explicit backend and dtype-provider inputs so
    tests and downstream tools can reason about decisions deterministically.
    The CLI/driver needs the same decisions bound to the current JAX backend.
    Keeping that binding here avoids private wrapper functions in
    ``v3_driver.py`` while preserving the same testable policy behavior.
    """

    backend: Callable[[], str]
    host_sparse_factor_dtype: Callable[..., np.dtype]

    def current_backend(self) -> str:
        """Return the active backend key used by runtime policy decisions."""

        return str(self.backend()).strip().lower()

    def dense_backend_allowed(self) -> bool:
        """Return whether dense transport may run on the active backend."""

        return transport_dense_backend_allowed(backend=self.current_backend())

    def dense_accelerator_auto_allowed(self, op: Any, *, geometry_scheme: int) -> bool:
        """Return whether dense transport can auto-promote to accelerator."""

        return transport_dense_accelerator_auto_allowed(
            op,
            backend=self.current_backend(),
            geometry_scheme=int(geometry_scheme),
        )

    def tzfft_backend_allowed(self) -> bool:
        """Return whether structured tzfft transport may run on this backend."""

        return transport_tzfft_backend_allowed(backend=self.current_backend())

    def tzfft_accelerator_auto_allowed(self, op: Any) -> bool:
        """Return whether structured tzfft may auto-run on accelerator."""

        return transport_tzfft_accelerator_auto_allowed(
            op,
            backend=self.current_backend(),
        )

    def tzfft_structured_first_attempt_allowed(
        self,
        op: Any,
        *,
        size: int,
        use_implicit: bool,
    ) -> bool:
        """Return whether the bounded tzfft first attempt is admissible."""

        return transport_tzfft_structured_first_attempt_allowed(
            op,
            size=int(size),
            use_implicit=bool(use_implicit),
            backend=self.current_backend(),
        )

    def sparse_direct_rescue_allowed(
        self,
        *,
        op: Any,
        size: int,
        residual_norm: float,
        target: float,
        use_implicit: bool,
    ) -> bool:
        """Return whether sparse-direct transport rescue is admissible."""

        return transport_sparse_direct_rescue_allowed(
            op=op,
            size=int(size),
            residual_norm=float(residual_norm),
            target=float(target),
            use_implicit=bool(use_implicit),
            backend=self.current_backend(),
        )

    def sparse_direct_first_attempt_allowed(
        self,
        *,
        op: Any,
        size: int,
        use_implicit: bool,
    ) -> bool:
        """Return whether sparse-direct should be attempted before Krylov."""

        return transport_sparse_direct_first_attempt_allowed(
            op=op,
            size=int(size),
            use_implicit=bool(use_implicit),
            backend=self.current_backend(),
        )

    def host_gmres_first_attempt_allowed(
        self,
        *,
        op: Any,
        size: int,
        use_implicit: bool,
    ) -> bool:
        """Return whether host GMRES should be attempted before other rescues."""

        return transport_host_gmres_first_attempt_allowed(
            op=op,
            size=int(size),
            use_implicit=bool(use_implicit),
            backend=self.current_backend(),
        )

    def disable_auto_recycle(self, *, op: Any, use_implicit: bool) -> bool:
        """Return whether transport Krylov recycling should be disabled."""

        return transport_disable_auto_recycle(
            op=op,
            use_implicit=bool(use_implicit),
            backend=self.current_backend(),
        )

    def sparse_factor_dtype(self, *, size: int, use_implicit: bool) -> np.dtype:
        """Resolve sparse transport factor dtype for the active backend."""

        return transport_sparse_factor_dtype(
            size=int(size),
            use_implicit=bool(use_implicit),
            backend=self.current_backend(),
            host_sparse_factor_dtype=self.host_sparse_factor_dtype,
        )

    def sparse_direct_use_explicit_helper(self, *, size: int) -> bool:
        """Return whether to materialize the explicit sparse helper."""

        return transport_sparse_direct_use_explicit_helper(
            size=int(size),
            backend=self.current_backend(),
        )

    def host_gmres_progress_every(self) -> int:
        """Resolve terminal progress cadence for transport host GMRES."""

        env = os.environ.get("SFINCS_JAX_TRANSPORT_HOST_GMRES_PROGRESS_EVERY", "").strip()
        try:
            return max(0, int(env)) if env else 10
        except ValueError:
            return 10


@dataclass(frozen=True)
class TransportPolishConfig:
    """Resolved RHSMode=3 polish settings."""

    enabled: bool
    threshold: float
    ratio: float
    abs_tol: float
    restart: int
    maxiter: int


def transport_residual_value(result: Any) -> float:
    """Return a finite-comparable residual value for a solver result."""

    val = float(result.residual_norm)
    return val if np.isfinite(val) else float("inf")


def transport_result_needs_retry(
    result: Any,
    target: float,
    *,
    result_is_finite,
) -> bool:
    """Shared retry gate for transport solve results."""

    return (not bool(result_is_finite(result))) or (
        transport_residual_value(result) > float(target)
    )


def transport_candidate_is_better(*, candidate: Any, current: Any) -> bool:
    """Compare candidate and current results using the transport residual metric."""

    return transport_residual_value(candidate) < transport_residual_value(current)


def transport_polish_config_from_env(
    *,
    rhs_mode: int,
    residual_norm: float,
    target: float,
    gmres_restart: int,
    maxiter: int | None,
) -> TransportPolishConfig:
    """Resolve the RHSMode=3 GMRES polish trigger and iteration budget."""

    polish_ratio_env = os.environ.get("SFINCS_JAX_TRANSPORT_POLISH_RATIO", "").strip()
    polish_abs_env = os.environ.get("SFINCS_JAX_TRANSPORT_POLISH_ABS", "").strip()
    polish_restart_env = os.environ.get("SFINCS_JAX_TRANSPORT_POLISH_RESTART", "").strip()
    polish_maxiter_env = os.environ.get("SFINCS_JAX_TRANSPORT_POLISH_MAXITER", "").strip()
    try:
        polish_ratio = float(polish_ratio_env) if polish_ratio_env else 2.0
    except ValueError:
        polish_ratio = 2.0
    try:
        polish_abs = float(polish_abs_env) if polish_abs_env else 1e-8
    except ValueError:
        polish_abs = 1e-8
    polish_thresh = max(float(target) * float(polish_ratio), float(polish_abs))
    base_restart = max(int(gmres_restart), 40)
    base_maxiter = int(maxiter) if maxiter is not None else 800
    try:
        polish_restart = (
            int(polish_restart_env)
            if polish_restart_env
            else max(base_restart * 2, 80)
        )
    except ValueError:
        polish_restart = max(base_restart * 2, 80)
    try:
        polish_maxiter = (
            int(polish_maxiter_env)
            if polish_maxiter_env
            else max(base_maxiter * 2, 1200)
        )
    except ValueError:
        polish_maxiter = max(base_maxiter * 2, 1200)
    enabled = int(rhs_mode) == 3 and float(residual_norm) > float(polish_thresh)
    return TransportPolishConfig(
        enabled=bool(enabled),
        threshold=float(polish_thresh),
        ratio=float(polish_ratio),
        abs_tol=float(polish_abs),
        restart=int(polish_restart),
        maxiter=int(polish_maxiter),
    )
