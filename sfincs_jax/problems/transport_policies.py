"""Pure transport backend and sparse-host policy helpers.

The transport driver owns operator assembly and residual evaluation; this module
only decides whether specific dense, sparse-direct, host-GMRES, and recycling
paths are eligible for a bounded transport solve.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import os
from typing import Any

import jax.numpy as jnp
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
    The CLI and transport solve owner need the same decisions bound to the
    current JAX backend. Keeping that binding here avoids private wrapper
    functions while preserving the same testable policy behavior.
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


# Active-DOF, dense-solve, and per-RHS loop policy helpers.
@dataclass(frozen=True)
class TransportActiveDOFDecision:
    """Resolved active-DOF routing decision for one transport solve."""

    use_active_dof_mode: bool
    reason: str | None
    solve_method_use: str
    emit_disabled_hint: bool


@dataclass(frozen=True)
class TransportActiveDOFState:
    """Active-index arrays and size metadata used by reduced transport solves."""

    active_idx_np: np.ndarray | None
    active_idx_jnp: jnp.ndarray | None
    full_to_active_jnp: jnp.ndarray | None
    active_size: int


@dataclass(frozen=True)
class TransportDensePolicy:
    """Resolved dense fallback and dense-preconditioner policy."""

    solve_method_use: str
    dense_fallback: bool
    dense_retry_max: int
    dense_mem_block: bool
    dense_use_mixed: bool
    force_dense: bool
    dense_precond_enabled: bool
    dense_precond_mem_block: bool
    dense_precond_est_mb: float
    dense_precond_mem_max_mb: float
    dense_mem_est_active_mb32: float
    dense_mem_est_active_mb64: float


@dataclass(frozen=True)
class TransportInitialSolvePolicy:
    """Initial RHSMode=2/3 output, dense fallback, and restart policy."""

    geometry_scheme: int
    low_memory_outputs: bool
    stream_diagnostics: bool
    store_state_vectors: bool
    solve_method_use: str
    force_krylov: bool
    force_dense: bool
    dense_fallback: bool
    dense_fallback_max: int
    dense_retry_max: int
    dense_mem_max_mb: float
    dense_mem_est_mb32: float
    dense_mem_est_mb64: float
    dense_mem_block: bool
    dense_use_mixed: bool
    dense_backend_allowed: bool
    dense_accelerator_auto_allowed: bool
    gmres_restart: int
    maxiter: int | None
    notes: tuple[tuple[int, str], ...] = ()


@dataclass(frozen=True)
class TransportPerRHSLoopPolicy:
    """Per-``whichRHS`` transport solve-loop decisions parsed from environment."""

    rhs_mode: int
    constraint_scheme: int
    phi1_size: int
    extra_size: int
    epar_loose_enabled: bool
    epar_krylov_enabled: bool
    project_nullspace_enabled: bool
    iter_stats_enabled: bool
    iter_stats_max_size: int | None
    dense_batch_fallback_enabled: bool

    def rhs3_krylov_flags(self, which_rhs: int) -> tuple[bool, bool]:
        """Return ``(use_loose_tol, force_krylov)`` for the E_parallel RHS."""
        is_epar_rhs = int(self.rhs_mode) == 2 and int(which_rhs) == 3 and int(self.constraint_scheme) == 1
        return bool(self.epar_loose_enabled and is_epar_rhs), bool(self.epar_krylov_enabled and is_epar_rhs)

    def projection_candidate(self, which_rhs: int) -> bool:
        """Return whether this RHS is one of the transport constraint-nullspace RHSs."""
        return (int(self.rhs_mode) == 2 and int(which_rhs) == 3) or (
            int(self.rhs_mode) == 3 and int(which_rhs) == 2
        )

    def projection_needed(self, which_rhs: int) -> bool:
        """Return whether nullspace projection is enabled and relevant for this RHS."""
        return bool(self.project_nullspace_enabled and self.projection_candidate(which_rhs))


def transport_geometry_scheme_from_namelist(nml: Any) -> int:
    """Return ``geometryScheme`` from a namelist-like object, or ``-1``."""
    geom_params = nml.group("geometryParameters")
    try:
        return int(geom_params.get("GEOMETRYSCHEME", geom_params.get("geometryScheme", -1)) or -1)
    except (TypeError, ValueError):
        return -1


def _transport_bool_env(name: str) -> bool | None:
    value = os.environ.get(name, "").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return None


def _transport_true_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _transport_disabled_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"0", "false", "no", "off"}


def _transport_optional_int_env(name: str) -> int | None:
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def resolve_transport_per_rhs_loop_policy(*, op: Any, rhs_mode: int) -> TransportPerRHSLoopPolicy:
    """Resolve branch-local RHSMode=2/3 loop policy before the ``whichRHS`` loop."""
    project_nullspace_enabled = (
        int(op.constraint_scheme) == 1
        and int(op.phi1_size) == 0
        and int(op.extra_size) > 0
        and not _transport_disabled_env("SFINCS_JAX_TRANSPORT_PROJECT_NULLSPACE")
    )
    return TransportPerRHSLoopPolicy(
        rhs_mode=int(rhs_mode),
        constraint_scheme=int(op.constraint_scheme),
        phi1_size=int(op.phi1_size),
        extra_size=int(op.extra_size),
        epar_loose_enabled=_transport_true_env("SFINCS_JAX_TRANSPORT_EPAR_LOOSE"),
        epar_krylov_enabled=_transport_true_env("SFINCS_JAX_TRANSPORT_EPAR_KRYLOV"),
        project_nullspace_enabled=bool(project_nullspace_enabled),
        iter_stats_enabled=_transport_true_env("SFINCS_JAX_SOLVER_ITER_STATS"),
        iter_stats_max_size=_transport_optional_int_env("SFINCS_JAX_SOLVER_ITER_STATS_MAX_SIZE"),
        dense_batch_fallback_enabled=not _transport_disabled_env("SFINCS_JAX_TRANSPORT_DENSE_BATCH_FALLBACK"),
    )


def resolve_transport_initial_solve_policy(
    *,
    op: Any,
    rhs_mode: int,
    n_rhs: int,
    solve_method: str,
    restart: int,
    maxiter: int | None,
    backend: str,
    geometry_scheme: int,
    dense_accelerator_auto_allowed: bool,
    dense_backend_policy_allowed: bool,
    state_out_requested: bool,
    force_stream_diagnostics: bool | None,
    force_store_state: bool | None,
    subset_mode: bool,
) -> TransportInitialSolvePolicy:
    """Resolve initial transport solve policy before active-DOF setup."""
    notes: list[tuple[int, str]] = []

    low_memory_env = _transport_bool_env("SFINCS_JAX_TRANSPORT_LOW_MEMORY")
    if low_memory_env is not None:
        low_memory_outputs = bool(low_memory_env)
    elif transport_geometry5_mono_low_memory_preferred(
        rhs_mode=int(rhs_mode),
        geometry_scheme=int(geometry_scheme),
        backend=str(backend),
        has_fp=op.fblock.fp is not None,
        n_x=int(op.n_x),
        total_size=int(op.total_size),
    ):
        low_memory_outputs = True
        notes.append(
            (
                1,
                "solve_v3_transport_matrix_linear_gmres: geometryScheme=5 RHSMode=3 "
                "auto -> low-memory Krylov transport path",
            )
        )
    else:
        low_memory_outputs = int(op.total_size) * int(n_rhs) >= 200_000

    stream_env = _transport_bool_env("SFINCS_JAX_TRANSPORT_STREAM_DIAGNOSTICS")
    stream_diagnostics = bool(low_memory_outputs) if stream_env is None else bool(stream_env)
    store_state_env = _transport_bool_env("SFINCS_JAX_TRANSPORT_STORE_STATE")
    store_state_vectors = (not stream_diagnostics) if store_state_env is None else bool(store_state_env)
    if state_out_requested:
        store_state_vectors = True
    if (not stream_diagnostics) and (not store_state_vectors):
        store_state_vectors = True
        notes.append((1, "solve_v3_transport_matrix_linear_gmres: forcing state storage (streaming disabled)"))
    if force_stream_diagnostics is not None:
        stream_diagnostics = bool(force_stream_diagnostics)
    if force_store_state is not None:
        store_state_vectors = bool(force_store_state)
    if subset_mode and not stream_diagnostics:
        stream_diagnostics = True
        notes.append((1, "solve_v3_transport_matrix_linear_gmres: streaming diagnostics forced for subset whichRHS"))

    solve_method_use = str(solve_method)
    force_krylov = _transport_bool_env("SFINCS_JAX_TRANSPORT_FORCE_KRYLOV") is True
    force_dense = _transport_bool_env("SFINCS_JAX_TRANSPORT_FORCE_DENSE") is True
    if low_memory_outputs:
        force_krylov = True
        force_dense = False

    dense_fallback_env = os.environ.get("SFINCS_JAX_TRANSPORT_DENSE_FALLBACK", "").strip().lower()
    dense_fallback_max_env = os.environ.get("SFINCS_JAX_TRANSPORT_DENSE_FALLBACK_MAX", "").strip()
    try:
        dense_fallback_max = int(dense_fallback_max_env) if dense_fallback_max_env else 0
    except ValueError:
        dense_fallback_max = 0
    dense_retry_env = os.environ.get("SFINCS_JAX_TRANSPORT_DENSE_RETRY_MAX", "").strip()
    try:
        if dense_retry_env:
            dense_retry_max = int(dense_retry_env)
        else:
            dense_retry_max = 6000 if int(rhs_mode) in {2, 3} else 0
    except ValueError:
        dense_retry_max = 3000 if int(rhs_mode) in {2, 3} else 0
    if low_memory_outputs:
        dense_retry_max = 0
    dense_mem_env = os.environ.get("SFINCS_JAX_TRANSPORT_DENSE_MAX_MB", "").strip()
    try:
        dense_mem_max_mb = float(dense_mem_env) if dense_mem_env else 128.0
    except ValueError:
        dense_mem_max_mb = 128.0
    dense_mem_est_mb64 = (int(op.total_size) ** 2) * 8.0 / 1.0e6
    dense_mem_est_mb32 = (int(op.total_size) ** 2) * 4.0 / 1.0e6
    dense_mem_block64 = bool(dense_mem_max_mb > 0.0 and dense_mem_est_mb64 > dense_mem_max_mb)
    dense_mem_block32 = bool(dense_mem_max_mb > 0.0 and dense_mem_est_mb32 > dense_mem_max_mb)
    dense_mem_block = dense_mem_block32
    dense_use_mixed = dense_mem_block64 and not dense_mem_block32
    dense_fallback_enabled_env = dense_fallback_env in {"1", "true", "yes", "on"}
    dense_fallback_disabled_env = dense_fallback_env in {"0", "false", "no", "off"}
    if dense_fallback_enabled_env:
        dense_fallback = True
        if not dense_fallback_max_env:
            dense_fallback_max = 1600
    elif dense_fallback_disabled_env:
        dense_fallback = False
    else:
        dense_fallback = int(rhs_mode) == 3
        if dense_fallback and not dense_fallback_max_env:
            dense_fallback_max = 6000

    dense_backend_allowed = bool(dense_backend_policy_allowed) or bool(dense_accelerator_auto_allowed)
    if dense_accelerator_auto_allowed:
        notes.append((1, "solve_v3_transport_matrix_linear_gmres: bounded accelerator dense transport auto enabled"))
    if not dense_backend_allowed:
        dense_fallback = False
        dense_retry_max = 0
        force_dense = False
        if str(solve_method_use).lower() == "dense":
            solve_method_use = "incremental"
        notes.append(
            (
                1,
                "solve_v3_transport_matrix_linear_gmres: dense transport path disabled "
                f"on backend={backend}",
            )
        )
    if dense_mem_block:
        dense_fallback = False
        dense_retry_max = 0
        force_dense = False
        notes.append(
            (
                1,
                "solve_v3_transport_matrix_linear_gmres: dense fallback disabled "
                f"(est_mem32={dense_mem_est_mb32:.1f} MB > {dense_mem_max_mb:.1f} MB)",
            )
        )
        if str(solve_method_use).lower() in {"auto", "default", "batched"}:
            solve_method_use = "incremental"
    elif dense_use_mixed:
        notes.append(
            (
                1,
                "solve_v3_transport_matrix_linear_gmres: dense fallback using float32 "
                f"(est_mem64={dense_mem_est_mb64:.1f} MB > {dense_mem_max_mb:.1f} MB)",
            )
        )
    if low_memory_outputs:
        dense_fallback = False

    if int(rhs_mode) in {2, 3}:
        if force_dense:
            solve_method_use = "dense"
            notes.append(
                (
                    0,
                    "solve_v3_transport_matrix_linear_gmres: forced dense solve "
                    f"for RHSMode={rhs_mode} (n={int(op.total_size)})",
                )
            )
        elif (
            int(rhs_mode) == 2
            and (not force_krylov)
            and dense_backend_allowed
            and str(solve_method_use).lower() in {"auto", "default", "batched", "incremental"}
            and int(op.total_size) <= 1500
            and (not dense_mem_block)
        ):
            solve_method_use = "dense"
            notes.append(
                (
                    0,
                    "solve_v3_transport_matrix_linear_gmres: auto dense solve for RHSMode=2 "
                    f"(n={int(op.total_size)})",
                )
            )
        elif (
            dense_fallback
            and (not force_krylov)
            and dense_backend_allowed
            and int(op.total_size) <= int(dense_fallback_max)
            and str(solve_method_use).lower() in {"auto", "default", "batched", "incremental"}
            and (not dense_mem_block)
        ):
            solve_method_use = "dense"
            notes.append(
                (
                    0,
                    "solve_v3_transport_matrix_linear_gmres: dense fallback enabled "
                    f"for RHSMode={rhs_mode} (n={int(op.total_size)})",
                )
            )

    gmres_restart_env = os.environ.get("SFINCS_JAX_TRANSPORT_GMRES_RESTART", "").strip()
    try:
        gmres_restart = int(gmres_restart_env) if gmres_restart_env else min(int(restart), 40)
    except ValueError:
        gmres_restart = min(int(restart), 40)
    if dense_mem_block and gmres_restart < 80:
        gmres_restart = 80

    maxiter_out = maxiter
    if dense_mem_block:
        if maxiter_out is None:
            maxiter_out = 800
        else:
            maxiter_out = max(int(maxiter_out), 800)

    return TransportInitialSolvePolicy(
        geometry_scheme=int(geometry_scheme),
        low_memory_outputs=bool(low_memory_outputs),
        stream_diagnostics=bool(stream_diagnostics),
        store_state_vectors=bool(store_state_vectors),
        solve_method_use=str(solve_method_use),
        force_krylov=bool(force_krylov),
        force_dense=bool(force_dense),
        dense_fallback=bool(dense_fallback),
        dense_fallback_max=int(dense_fallback_max),
        dense_retry_max=int(dense_retry_max),
        dense_mem_max_mb=float(dense_mem_max_mb),
        dense_mem_est_mb32=float(dense_mem_est_mb32),
        dense_mem_est_mb64=float(dense_mem_est_mb64),
        dense_mem_block=bool(dense_mem_block),
        dense_use_mixed=bool(dense_use_mixed),
        dense_backend_allowed=bool(dense_backend_allowed),
        dense_accelerator_auto_allowed=bool(dense_accelerator_auto_allowed),
        gmres_restart=int(gmres_restart),
        maxiter=maxiter_out,
        notes=tuple(notes),
    )


def transport_geometry5_mono_low_memory_preferred(
    *,
    rhs_mode: int,
    geometry_scheme: int,
    backend: str,
    has_fp: bool,
    n_x: int,
    total_size: int,
) -> bool:
    """Return whether VMEC monoenergetic transport should avoid dense fallback.

    The CPU VMEC RHSMode=3 examples are small enough that dense batched solves are
    numerically safe, but the CLI/XLA dense path can transiently retain multi-GB
    allocations. The existing Krylov + ``tzfft`` path is parity-clean on the
    geometryScheme=5 monoenergetic examples and has much lower peak RSS.
    """
    mode = os.environ.get("SFINCS_JAX_TRANSPORT_GEOM5_MONO_LOW_MEMORY", "").strip().lower()
    if mode in {"0", "false", "no", "off"}:
        return False
    if int(rhs_mode) != 3 or int(geometry_scheme) != 5:
        return False
    if bool(has_fp) or int(n_x) > 2:
        return False
    if mode in {"1", "true", "yes", "on"}:
        return True
    if str(backend).strip().lower() != "cpu":
        return False
    min_env = os.environ.get("SFINCS_JAX_TRANSPORT_GEOM5_MONO_LOW_MEMORY_MIN", "").strip()
    max_env = os.environ.get("SFINCS_JAX_TRANSPORT_GEOM5_MONO_LOW_MEMORY_MAX", "").strip()
    try:
        min_size = int(min_env) if min_env else 1000
    except ValueError:
        min_size = 1000
    try:
        max_size = int(max_env) if max_env else 20000
    except ValueError:
        max_size = 20000
    return max(1, int(min_size)) <= int(total_size) <= max(1, int(max_size))


def resolve_transport_active_dof_mode(
    *,
    op: Any,
    rhs_mode: int,
    solve_method_use: str,
    solve_method: str,
    active_dof_env: str,
) -> TransportActiveDOFDecision:
    """Resolve whether transport should compact to the active pitch-angle DOFs."""
    env = str(active_dof_env).strip().lower()
    reason: str | None = None
    if env in {"0", "false", "no", "off"}:
        use_active_dof_mode = False
    elif env in {"1", "true", "yes", "on"}:
        use_active_dof_mode = True
        reason = "env"
    elif int(rhs_mode) in {2, 3}:
        nxi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
        use_active_dof_mode = bool(np.any(nxi_for_x < int(op.n_xi)))
        if use_active_dof_mode:
            reason = "auto"
    else:
        use_active_dof_mode = False
    solve_method_out = str(solve_method_use)
    if use_active_dof_mode and str(solve_method_out).lower() == "dense":
        solve_method_out = str(solve_method)
    emit_disabled_hint = (
        (not use_active_dof_mode)
        and int(rhs_mode) in {2, 3}
        and env not in {"0", "false", "no", "off"}
    )
    return TransportActiveDOFDecision(
        use_active_dof_mode=bool(use_active_dof_mode),
        reason=reason,
        solve_method_use=solve_method_out,
        emit_disabled_hint=bool(emit_disabled_hint),
    )


def build_transport_active_dof_state(
    *,
    op: Any,
    use_active_dof_mode: bool,
    active_dof_indices,
) -> TransportActiveDOFState:
    """Build active-DOF indexing state or a full-size no-op state."""
    if not use_active_dof_mode:
        return TransportActiveDOFState(
            active_idx_np=None,
            active_idx_jnp=None,
            full_to_active_jnp=None,
            active_size=int(op.total_size),
        )
    active_idx_np = np.asarray(active_dof_indices(op), dtype=np.int32)
    active_idx_jnp = jnp.asarray(active_idx_np, dtype=jnp.int32)
    full_to_active_np = np.zeros((int(op.total_size),), dtype=np.int32)
    full_to_active_np[np.asarray(active_idx_np, dtype=np.int32)] = np.arange(1, int(active_idx_np.shape[0]) + 1, dtype=np.int32)
    full_to_active_jnp = jnp.asarray(full_to_active_np, dtype=jnp.int32)
    return TransportActiveDOFState(
        active_idx_np=active_idx_np,
        active_idx_jnp=active_idx_jnp,
        full_to_active_jnp=full_to_active_jnp,
        active_size=int(active_idx_np.shape[0]),
    )


def resolve_transport_dense_policy(
    *,
    rhs_mode: int,
    n_rhs: int,
    total_size: int,
    active_size: int,
    solve_method_use: str,
    force_krylov: bool,
    force_dense: bool,
    dense_fallback: bool,
    dense_retry_max: int,
    dense_mem_max_mb: float,
    dense_mem_block: bool,
    dense_use_mixed: bool,
    low_memory_outputs: bool,
    dense_backend_allowed: bool,
    dense_precond_default: bool,
) -> TransportDensePolicy:
    """Resolve dense fallback/preconditioner admission under memory caps."""
    dense_mem_est_active_mb64 = (int(active_size) ** 2) * 8.0 / 1.0e6
    dense_mem_est_active_mb32 = (int(active_size) ** 2) * 4.0 / 1.0e6
    dense_mem_block_active32 = bool(dense_mem_max_mb > 0.0 and dense_mem_est_active_mb32 > dense_mem_max_mb)
    dense_mem_block_active64 = bool(dense_mem_max_mb > 0.0 and dense_mem_est_active_mb64 > dense_mem_max_mb)

    solve_method_out = str(solve_method_use)
    dense_fallback_out = bool(dense_fallback)
    dense_retry_max_out = int(dense_retry_max)
    dense_mem_block_out = bool(dense_mem_block)
    dense_use_mixed_out = bool(dense_use_mixed)
    force_dense_out = bool(force_dense)

    if dense_mem_block_active32 and not dense_mem_block_out:
        dense_mem_block_out = True
        dense_use_mixed_out = False
        dense_fallback_out = False
        dense_retry_max_out = 0
        force_dense_out = False
        if str(solve_method_out).lower() == "dense":
            solve_method_out = "incremental"
    elif dense_mem_block_active64 and not dense_mem_block_out and not dense_use_mixed_out:
        dense_use_mixed_out = True

    if (
        int(rhs_mode) == 2
        and (not force_krylov)
        and (not force_dense_out)
        and dense_backend_allowed
        and str(solve_method_out).lower() in {"auto", "default", "batched", "incremental"}
    ):
        auto_dense_limit = 1500
        if int(n_rhs) > 1 and dense_retry_max_out > 0:
            auto_dense_limit = max(auto_dense_limit, min(3000, int(dense_retry_max_out)))
        if int(active_size) <= auto_dense_limit and (not dense_mem_block_out):
            solve_method_out = "dense"

    dense_precond_max_env = os.environ.get("SFINCS_JAX_TRANSPORT_DENSE_PRECOND_MAX", "").strip()
    dense_precond_mem_env = os.environ.get("SFINCS_JAX_TRANSPORT_DENSE_PRECOND_MAX_MB", "").strip()
    try:
        dense_precond_max = int(dense_precond_max_env) if dense_precond_max_env else (1600 if int(rhs_mode) == 2 else 600)
    except ValueError:
        dense_precond_max = 1600 if int(rhs_mode) == 2 else 600
    try:
        dense_precond_mem_max_mb = float(dense_precond_mem_env) if dense_precond_mem_env else min(32.0, dense_mem_max_mb or 32.0)
    except ValueError:
        dense_precond_mem_max_mb = min(32.0, dense_mem_max_mb or 32.0)
    dense_precond_size = int(active_size)
    dense_precond_bytes = 4.0 if dense_use_mixed_out else 8.0
    dense_precond_est_mb = (dense_precond_size**2) * dense_precond_bytes / 1.0e6
    dense_precond_mem_block = bool(dense_precond_mem_max_mb > 0.0 and dense_precond_est_mb > dense_precond_mem_max_mb)
    dense_precond_enabled = (
        bool(dense_precond_default)
        and dense_precond_max > 0
        and int(rhs_mode) in {2, 3}
        and int(dense_precond_size) <= dense_precond_max
        and str(solve_method_out).lower() != "dense"
        and (not low_memory_outputs)
        and (not dense_mem_block_out)
        and (not dense_precond_mem_block)
        and dense_backend_allowed
    )
    return TransportDensePolicy(
        solve_method_use=solve_method_out,
        dense_fallback=dense_fallback_out,
        dense_retry_max=int(dense_retry_max_out),
        dense_mem_block=bool(dense_mem_block_out),
        dense_use_mixed=bool(dense_use_mixed_out),
        force_dense=bool(force_dense_out),
        dense_precond_enabled=bool(dense_precond_enabled),
        dense_precond_mem_block=bool(dense_precond_mem_block),
        dense_precond_est_mb=float(dense_precond_est_mb),
        dense_precond_mem_max_mb=float(dense_precond_mem_max_mb),
        dense_mem_est_active_mb32=float(dense_mem_est_active_mb32),
        dense_mem_est_active_mb64=float(dense_mem_est_active_mb64),
    )


# Transport preconditioner selection and builder dispatch helpers.
Preconditioner = Callable[[Any], Any]
Builder = Callable[..., Preconditioner]


@dataclass(frozen=True)
class TransportPreconditionerContext:
    op: Any
    active_size: int
    use_active_dof_mode: bool
    reduce_full: Callable[[Any], Any] | None = None
    expand_reduced: Callable[[Any], Any] | None = None
    active_indices_np: Any | None = None
    emit: Callable[[int, str], None] | None = None


@dataclass(frozen=True)
class TransportPreconditionerDispatchBuilders:
    collision_builder: Builder
    sxblock_builder: Builder
    block_builder: Builder
    xmg_builder: Builder
    theta_dd_builder: Builder
    theta_schwarz_builder: Builder
    zeta_dd_builder: Builder
    zeta_schwarz_builder: Builder
    tzfft_builder: Builder
    sparse_jax_builder: Builder
    sparse_jax_cache_key: Callable[[Any, str], tuple[object, ...]]
    apply_operator_cached: Callable[[Any, jnp.ndarray], jnp.ndarray]
    precond_dtype: Callable[[int], jnp.dtype]
    fp_tzfft_builder: Builder | None = None
    fp_tzfft_line_builder: Builder | None = None
    fp_tzfft_line_schur_builder: Builder | None = None
    fp_local_geom_line_builder: Builder | None = None
    fp_xblock_tz_lu_builder: Builder | None = None
    fp_xblock_tz_lu_schur_builder: Builder | None = None
    fp_structured_fblock_lu_builder: Builder | None = None
    fp_fortran_reduced_lu_builder: Builder | None = None
    fp_direct_active_block_schur_builder: Builder | None = None


@dataclass(frozen=True)
class TransportDDConfig:
    block_theta: int
    overlap_theta: int
    block_zeta: int
    overlap_zeta: int


@dataclass(frozen=True)
class TransportSparseJaxConfig:
    drop_tol: float
    drop_rel: float
    reg: float
    omega: float
    sweeps: int
    max_mb: float


def normalize_transport_preconditioner_kind(*, env_value: str) -> str | None:
    env = str(env_value).strip().lower()
    if env in {"0", "none", "off", "false", "no"}:
        return None
    if env in {
        "collision",
        "block",
        "block_jacobi",
        "sxblock",
        "block_sx",
        "species_x",
        "xmg",
        "multigrid",
        "theta_dd",
        "theta_block",
        "dd_theta",
        "dd_t",
        "theta_schwarz",
        "schwarz_theta",
        "zeta_dd",
        "zeta_block",
        "dd_zeta",
        "dd_z",
        "zeta_schwarz",
        "schwarz_zeta",
        "tzfft",
        "fp_tzfft",
        "fp_tzfft_line",
        "fp_tzfft_line_schur",
        "fp_tzfft_schur",
        "fp_streaming_line_schur",
        "fp_block_thomas_schur",
        "fp_line_schur",
        "fp_local_geom_line",
        "fp_geom_line",
        "fp_local_line",
        "fp_nonavg_line",
        "fp_xblock_tz_lu",
        "fp_xblock_tz_lu_schur",
        "fp_xblock_schur",
        "fp_xblock_lu_schur",
        "fp_tz_xblock_lu_schur",
        "fp_angular_xblock_lu_schur",
        "fp_xblock_lu",
        "fp_tz_xblock_lu",
        "fp_angular_xblock_lu",
        "fp_structured_fblock_lu",
        "fp_fblock_lu",
        "fp_full_fblock_lu",
        "fp_kinetic_lu",
        "fp_fortran_reduced_lu",
        "fp_global_fortran_reduced_lu",
        "fp_petsc_like_lu",
        "fp_reduced_pmat_lu",
        "fp_direct_active_block_schur",
        "fp_direct_active_block_lu",
        "fp_active_true_block_schur",
        "fp_active_true_block",
        "fp_true_block_schur",
        "fp_true_block_lu",
        "fp_streaming_line",
        "fp_block_thomas",
        "fp_line",
        "fp_streaming_fft",
        "streaming_fft",
        "stream_fft",
        "sparse_jax",
    }:
        if env in {"streaming_fft", "stream_fft"}:
            return "tzfft"
        if env in {"fp_streaming_fft"}:
            return "fp_tzfft"
        if env in {"fp_tzfft_schur", "fp_streaming_line_schur", "fp_block_thomas_schur", "fp_line_schur"}:
            return "fp_tzfft_line_schur"
        if env in {"fp_geom_line", "fp_local_line", "fp_nonavg_line"}:
            return "fp_local_geom_line"
        if env in {"fp_xblock_schur", "fp_xblock_lu_schur", "fp_tz_xblock_lu_schur", "fp_angular_xblock_lu_schur"}:
            return "fp_xblock_tz_lu_schur"
        if env in {"fp_xblock_lu", "fp_tz_xblock_lu", "fp_angular_xblock_lu"}:
            return "fp_xblock_tz_lu"
        if env in {"fp_fblock_lu", "fp_full_fblock_lu", "fp_kinetic_lu"}:
            return "fp_structured_fblock_lu"
        if env in {"fp_global_fortran_reduced_lu", "fp_petsc_like_lu", "fp_reduced_pmat_lu"}:
            return "fp_fortran_reduced_lu"
        if env in {
            "fp_direct_active_block_lu",
            "fp_active_true_block_schur",
            "fp_active_true_block",
            "fp_true_block_schur",
            "fp_true_block_lu",
        }:
            return "fp_direct_active_block_schur"
        if env in {"fp_streaming_line", "fp_block_thomas", "fp_line"}:
            return "fp_tzfft_line"
        if env in {"theta_block", "dd_theta", "dd_t"}:
            return "theta_dd"
        if env in {"theta_schwarz", "schwarz_theta"}:
            return "theta_schwarz"
        if env in {"zeta_block", "dd_zeta", "dd_z"}:
            return "zeta_dd"
        if env in {"zeta_schwarz", "schwarz_zeta"}:
            return "zeta_schwarz"
        return env
    return "auto"


def transport_dd_config_from_env(*, op: Any) -> TransportDDConfig:
    block_t_env = os.environ.get("SFINCS_JAX_TRANSPORT_DD_BLOCK_T", "").strip()
    block_z_env = os.environ.get("SFINCS_JAX_TRANSPORT_DD_BLOCK_Z", "").strip()
    overlap_env = os.environ.get("SFINCS_JAX_TRANSPORT_DD_OVERLAP", "").strip()
    try:
        block_t = int(block_t_env) if block_t_env else 8
    except ValueError:
        block_t = 8
    try:
        block_z = int(block_z_env) if block_z_env else 8
    except ValueError:
        block_z = 8
    try:
        overlap = int(overlap_env) if overlap_env else 1
    except ValueError:
        overlap = 1
    block_t = max(1, min(int(getattr(op, "n_theta", 1)), int(block_t)))
    block_z = max(1, min(int(getattr(op, "n_zeta", 1)), int(block_z)))
    overlap_t = max(0, min(int(block_t) - 1, int(overlap)))
    overlap_z = max(0, min(int(block_z) - 1, int(overlap)))
    return TransportDDConfig(
        block_theta=int(block_t),
        overlap_theta=int(overlap_t),
        block_zeta=int(block_z),
        overlap_zeta=int(overlap_z),
    )


def transport_sparse_jax_config_from_env() -> TransportSparseJaxConfig:
    drop_tol_env = os.environ.get("SFINCS_JAX_TRANSPORT_SPARSE_DROP_TOL", "").strip()
    drop_rel_env = os.environ.get("SFINCS_JAX_TRANSPORT_SPARSE_DROP_REL", "").strip()
    reg_env = os.environ.get("SFINCS_JAX_TRANSPORT_SPARSE_JAX_REG", "").strip()
    omega_env = os.environ.get("SFINCS_JAX_TRANSPORT_SPARSE_JAX_OMEGA", "").strip()
    sweeps_env = os.environ.get("SFINCS_JAX_TRANSPORT_SPARSE_JAX_SWEEPS", "").strip()
    max_env = os.environ.get("SFINCS_JAX_TRANSPORT_SPARSE_JAX_MAX_MB", "").strip()
    try:
        drop_tol = float(drop_tol_env) if drop_tol_env else 0.0
    except ValueError:
        drop_tol = 0.0
    try:
        drop_rel = float(drop_rel_env) if drop_rel_env else 1.0e-6
    except ValueError:
        drop_rel = 1.0e-6
    try:
        reg = float(reg_env) if reg_env else 1e-10
    except ValueError:
        reg = 1e-10
    try:
        omega = float(omega_env) if omega_env else 0.8
    except ValueError:
        omega = 0.8
    try:
        sweeps = int(sweeps_env) if sweeps_env else 2
    except ValueError:
        sweeps = 2
    try:
        max_mb = float(max_env) if max_env else 128.0
    except ValueError:
        max_mb = 128.0
    return TransportSparseJaxConfig(
        drop_tol=float(drop_tol),
        drop_rel=float(drop_rel),
        reg=float(reg),
        omega=float(omega),
        sweeps=max(1, int(sweeps)),
        max_mb=float(max_mb),
    )


def auto_transport_preconditioner_choice(
    *,
    op: Any,
    default_solver_kind: str,
    parallel_workers: int,
    dense_mem_block: bool,
    tzfft_backend_allowed: bool,
    shard_axis: str | None,
) -> tuple[str, str | None]:
    block_max_env = os.environ.get("SFINCS_JAX_TRANSPORT_PRECOND_BLOCK_MAX", "").strip()
    sxblock_max_env = os.environ.get("SFINCS_JAX_TRANSPORT_SXBLOCK_MAX", "").strip()
    dd_auto_min_env = os.environ.get("SFINCS_JAX_TRANSPORT_DD_AUTO_MIN", "").strip()
    try:
        block_max = int(block_max_env) if block_max_env else 5000
    except ValueError:
        block_max = 5000
    try:
        sxblock_max = int(sxblock_max_env) if sxblock_max_env else 64
    except ValueError:
        sxblock_max = 64
    try:
        dd_auto_min = int(dd_auto_min_env) if dd_auto_min_env else 0
    except ValueError:
        dd_auto_min = 0

    n_block = int(op.n_species) * int(op.n_x)
    precond_kind: str
    strong_precond_kind: str | None = None
    if op.fblock.fp is not None:
        fp_geom_env = os.environ.get("SFINCS_JAX_TRANSPORT_FP_LOCAL_GEOM_LINE_AUTO", "").strip().lower()
        fp_geom_disabled = fp_geom_env in {"", "0", "false", "no", "off"}
        fp_geom_forced = fp_geom_env in {"1", "true", "yes", "on"}
        fp_geom_min_env = os.environ.get("SFINCS_JAX_TRANSPORT_FP_LOCAL_GEOM_LINE_AUTO_MIN", "").strip()
        try:
            fp_geom_min = int(fp_geom_min_env) if fp_geom_min_env else 50000
        except ValueError:
            fp_geom_min = 50000
        fp_schur_env = os.environ.get("SFINCS_JAX_TRANSPORT_FP_TZFFT_LINE_SCHUR_AUTO", "").strip().lower()
        fp_schur_disabled = fp_schur_env in {"", "0", "false", "no", "off"}
        fp_schur_forced = fp_schur_env in {"1", "true", "yes", "on"}
        fp_schur_min_env = os.environ.get("SFINCS_JAX_TRANSPORT_FP_TZFFT_LINE_SCHUR_AUTO_MIN", "").strip()
        try:
            fp_schur_min = int(fp_schur_min_env) if fp_schur_min_env else 50000
        except ValueError:
            fp_schur_min = 50000
        fp_line_env = os.environ.get("SFINCS_JAX_TRANSPORT_FP_TZFFT_LINE_AUTO", "").strip().lower()
        fp_line_disabled = fp_line_env in {"", "0", "false", "no", "off"}
        fp_line_forced = fp_line_env in {"1", "true", "yes", "on"}
        fp_line_min_env = os.environ.get("SFINCS_JAX_TRANSPORT_FP_TZFFT_LINE_AUTO_MIN", "").strip()
        try:
            fp_line_min = int(fp_line_min_env) if fp_line_min_env else 50000
        except ValueError:
            fp_line_min = 50000
        fp_line_candidate = bool(
            (not fp_line_disabled)
            and int(getattr(op, "rhs_mode", 0) or 0) in {2, 3}
            and not bool(getattr(op, "include_phi1", False))
            and int(getattr(op, "n_theta", 0) or 0) * int(getattr(op, "n_zeta", 0) or 0) >= 64
            and (fp_line_forced or int(getattr(op, "total_size", 0) or 0) >= max(1, int(fp_line_min)))
        )
        fp_schur_candidate = bool(
            (not fp_schur_disabled)
            and int(getattr(op, "rhs_mode", 0) or 0) in {2, 3}
            and not bool(getattr(op, "include_phi1", False))
            and int(getattr(op, "n_theta", 0) or 0) * int(getattr(op, "n_zeta", 0) or 0) >= 64
            and (fp_schur_forced or int(getattr(op, "total_size", 0) or 0) >= max(1, int(fp_schur_min)))
        )
        fp_geom_candidate = bool(
            (not fp_geom_disabled)
            and int(getattr(op, "rhs_mode", 0) or 0) in {2, 3}
            and not bool(getattr(op, "include_phi1", False))
            and int(getattr(op, "n_theta", 0) or 0) * int(getattr(op, "n_zeta", 0) or 0) >= 64
            and (fp_geom_forced or int(getattr(op, "total_size", 0) or 0) >= max(1, int(fp_geom_min)))
        )
        if fp_geom_candidate:
            precond_kind = "fp_local_geom_line"
            strong_precond_kind = "fp_local_geom_line"
        elif fp_schur_candidate:
            precond_kind = "fp_tzfft_line_schur"
            strong_precond_kind = "fp_tzfft_line_schur"
        elif fp_line_candidate:
            precond_kind = "fp_tzfft_line"
            strong_precond_kind = "fp_tzfft_line"
        elif n_block <= sxblock_max:
            precond_kind = "sxblock"
        elif int(op.total_size) <= block_max and str(default_solver_kind) != "bicgstab":
            precond_kind = "sxblock"
        else:
            precond_kind = "collision"
        if (
            not fp_line_candidate
            and not fp_schur_candidate
            and not fp_geom_candidate
        ):
            if n_block <= sxblock_max:
                strong_precond_kind = "sxblock"
            elif int(op.total_size) <= block_max:
                strong_precond_kind = "block"
            else:
                strong_precond_kind = "xmg"
    else:
        no_fp = op.fblock.fp is None
        small_x = int(op.n_x) <= 2
        multi_angle = int(op.n_theta) * int(op.n_zeta) >= 64
        if no_fp and small_x and multi_angle and tzfft_backend_allowed:
            precond_kind = "tzfft"
            strong_precond_kind = "tzfft"
        elif int(op.total_size) <= block_max:
            precond_kind = "block"
            strong_precond_kind = "block"
        else:
            precond_kind = "collision"
            strong_precond_kind = "collision" if (no_fp and (not tzfft_backend_allowed)) else "xmg"

    if int(parallel_workers) > 1 and dd_auto_min > 0 and int(op.total_size) >= dd_auto_min:
        if shard_axis == "theta":
            precond_kind = "theta_schwarz"
            strong_precond_kind = "theta_schwarz"
        elif shard_axis == "zeta":
            precond_kind = "zeta_schwarz"
            strong_precond_kind = "zeta_schwarz"
    if dense_mem_block and strong_precond_kind is not None:
        precond_kind = strong_precond_kind
    return precond_kind, strong_precond_kind


def resolve_transport_preconditioner_choice(
    *,
    op: Any,
    transport_precond_kind: str | None,
    default_solver_kind: str,
    parallel_workers: int,
    dense_mem_block: bool,
    tzfft_backend_allowed: bool,
    shard_axis: str | None,
    backend: str,
    emit: Callable[[int, str], None] | None = None,
) -> tuple[str | None, str | None]:
    if transport_precond_kind is None:
        return None, None
    precond_kind = transport_precond_kind
    strong_precond_kind: str | None = None
    if precond_kind == "auto":
        precond_kind, strong_precond_kind = auto_transport_preconditioner_choice(
            op=op,
            default_solver_kind=default_solver_kind,
            parallel_workers=parallel_workers,
            dense_mem_block=dense_mem_block,
            tzfft_backend_allowed=tzfft_backend_allowed,
            shard_axis=shard_axis,
        )
    if precond_kind == "tzfft" and (not tzfft_backend_allowed):
        if emit is not None:
            emit(
                1,
                "solve_v3_transport_matrix_linear_gmres: tzfft preconditioner disabled on "
                f"backend={backend}",
            )
        precond_kind = "collision"
        if strong_precond_kind == "tzfft":
            strong_precond_kind = "collision"
    return precond_kind, strong_precond_kind


def resolve_transport_precondition_side_for_kind(
    *,
    kind: str | None,
    requested_side: str,
) -> tuple[str, bool]:
    """Return a preconditioner-side choice that is valid for the selected kind.

    The FP Fourier line factor uses a forward/backward block-Thomas scan.  It is
    intended as a left preconditioner; current JAX transpose rules for the scan
    path make user-forced right preconditioning fragile on some backends.
    Keeping the guard here makes the solver policy explicit and testable.
    """
    side = str(requested_side).strip().lower()
    if side not in {"left", "right", "none"}:
        side = "left"
    if (
        kind
        in {
            "fp_tzfft_line",
            "fp_tzfft_line_schur",
            "fp_local_geom_line",
            "fp_xblock_tz_lu",
            "fp_xblock_tz_lu_schur",
            "fp_structured_fblock_lu",
            "fp_fortran_reduced_lu",
            "fp_direct_active_block_schur",
        }
        and side == "right"
    ):
        return "left", True
    return side, False


def build_transport_preconditioner_from_kind(
    *,
    kind: str,
    context: TransportPreconditionerContext,
    builders: TransportPreconditionerDispatchBuilders,
    dd_config: TransportDDConfig,
    sparse_jax_config: TransportSparseJaxConfig,
    use_reduced: bool,
) -> Preconditioner:
    reduce_full = context.reduce_full if use_reduced else None
    expand_reduced = context.expand_reduced if use_reduced else None
    size_est = int(context.active_size) if use_reduced else int(context.op.total_size)
    if kind in {"xmg", "multigrid"}:
        return builders.xmg_builder(op=context.op, reduce_full=reduce_full, expand_reduced=expand_reduced)
    if kind == "theta_dd":
        return builders.theta_dd_builder(
            op=context.op,
            block=int(dd_config.block_theta),
            reduce_full=reduce_full,
            expand_reduced=expand_reduced,
        )
    if kind == "theta_schwarz":
        return builders.theta_schwarz_builder(
            op=context.op,
            block=int(dd_config.block_theta),
            overlap=int(dd_config.overlap_theta),
            reduce_full=reduce_full,
            expand_reduced=expand_reduced,
        )
    if kind == "zeta_dd":
        return builders.zeta_dd_builder(
            op=context.op,
            block=int(dd_config.block_zeta),
            reduce_full=reduce_full,
            expand_reduced=expand_reduced,
        )
    if kind == "zeta_schwarz":
        return builders.zeta_schwarz_builder(
            op=context.op,
            block=int(dd_config.block_zeta),
            overlap=int(dd_config.overlap_zeta),
            reduce_full=reduce_full,
            expand_reduced=expand_reduced,
        )
    if kind == "tzfft":
        return builders.tzfft_builder(op=context.op, reduce_full=reduce_full, expand_reduced=expand_reduced)
    if kind == "fp_tzfft":
        # Older builder bundles do not know this experimental preconditioner.
        fp_tzfft_builder = getattr(builders, "fp_tzfft_builder", None)
        if fp_tzfft_builder is None:
            return builders.tzfft_builder(op=context.op, reduce_full=reduce_full, expand_reduced=expand_reduced)
        return fp_tzfft_builder(op=context.op, reduce_full=reduce_full, expand_reduced=expand_reduced)
    if kind == "fp_tzfft_line":
        fp_tzfft_line_builder = getattr(builders, "fp_tzfft_line_builder", None)
        if fp_tzfft_line_builder is None:
            fp_tzfft_builder = getattr(builders, "fp_tzfft_builder", None)
            if fp_tzfft_builder is not None:
                return fp_tzfft_builder(op=context.op, reduce_full=reduce_full, expand_reduced=expand_reduced)
            return builders.sxblock_builder(op=context.op, reduce_full=reduce_full, expand_reduced=expand_reduced)
        return fp_tzfft_line_builder(op=context.op, reduce_full=reduce_full, expand_reduced=expand_reduced)
    if kind == "fp_tzfft_line_schur":
        fp_tzfft_line_schur_builder = getattr(builders, "fp_tzfft_line_schur_builder", None)
        if fp_tzfft_line_schur_builder is not None:
            return fp_tzfft_line_schur_builder(op=context.op, reduce_full=reduce_full, expand_reduced=expand_reduced)
        fp_tzfft_line_builder = getattr(builders, "fp_tzfft_line_builder", None)
        if fp_tzfft_line_builder is not None:
            return fp_tzfft_line_builder(op=context.op, reduce_full=reduce_full, expand_reduced=expand_reduced)
        fp_tzfft_builder = getattr(builders, "fp_tzfft_builder", None)
        if fp_tzfft_builder is not None:
            return fp_tzfft_builder(op=context.op, reduce_full=reduce_full, expand_reduced=expand_reduced)
        return builders.sxblock_builder(op=context.op, reduce_full=reduce_full, expand_reduced=expand_reduced)
    if kind == "fp_local_geom_line":
        fp_local_geom_line_builder = getattr(builders, "fp_local_geom_line_builder", None)
        if fp_local_geom_line_builder is not None:
            return fp_local_geom_line_builder(op=context.op, reduce_full=reduce_full, expand_reduced=expand_reduced)
        fp_tzfft_line_builder = getattr(builders, "fp_tzfft_line_builder", None)
        if fp_tzfft_line_builder is not None:
            return fp_tzfft_line_builder(op=context.op, reduce_full=reduce_full, expand_reduced=expand_reduced)
        return builders.sxblock_builder(op=context.op, reduce_full=reduce_full, expand_reduced=expand_reduced)
    if kind == "fp_xblock_tz_lu":
        fp_xblock_tz_lu_builder = getattr(builders, "fp_xblock_tz_lu_builder", None)
        if fp_xblock_tz_lu_builder is not None:
            return fp_xblock_tz_lu_builder(
                op=context.op,
                reduce_full=reduce_full,
                expand_reduced=expand_reduced,
            )
        fp_tzfft_line_builder = getattr(builders, "fp_tzfft_line_builder", None)
        if fp_tzfft_line_builder is not None:
            return fp_tzfft_line_builder(op=context.op, reduce_full=reduce_full, expand_reduced=expand_reduced)
        return builders.sxblock_builder(op=context.op, reduce_full=reduce_full, expand_reduced=expand_reduced)
    if kind == "fp_xblock_tz_lu_schur":
        fp_xblock_tz_lu_schur_builder = getattr(builders, "fp_xblock_tz_lu_schur_builder", None)
        if fp_xblock_tz_lu_schur_builder is not None:
            return fp_xblock_tz_lu_schur_builder(
                op=context.op,
                reduce_full=reduce_full,
                expand_reduced=expand_reduced,
            )
        fp_xblock_tz_lu_builder = getattr(builders, "fp_xblock_tz_lu_builder", None)
        if fp_xblock_tz_lu_builder is not None:
            return fp_xblock_tz_lu_builder(op=context.op, reduce_full=reduce_full, expand_reduced=expand_reduced)
        return builders.sxblock_builder(op=context.op, reduce_full=reduce_full, expand_reduced=expand_reduced)
    if kind == "fp_structured_fblock_lu":
        fp_structured_fblock_lu_builder = getattr(builders, "fp_structured_fblock_lu_builder", None)
        if fp_structured_fblock_lu_builder is not None:
            return fp_structured_fblock_lu_builder(
                op=context.op,
                reduce_full=reduce_full,
                expand_reduced=expand_reduced,
            )
        return builders.sxblock_builder(op=context.op, reduce_full=reduce_full, expand_reduced=expand_reduced)
    if kind == "fp_fortran_reduced_lu":
        fp_fortran_reduced_lu_builder = getattr(builders, "fp_fortran_reduced_lu_builder", None)
        if fp_fortran_reduced_lu_builder is not None:
            if bool(context.use_active_dof_mode) and not use_reduced:
                return builders.sxblock_builder(
                    op=context.op,
                    reduce_full=reduce_full,
                    expand_reduced=expand_reduced,
                )
            return fp_fortran_reduced_lu_builder(
                op=context.op,
                reduce_full=reduce_full,
                expand_reduced=expand_reduced,
                active_indices_np=context.active_indices_np if use_reduced else None,
                emit=context.emit,
            )
        return builders.sxblock_builder(op=context.op, reduce_full=reduce_full, expand_reduced=expand_reduced)
    if kind == "fp_direct_active_block_schur":
        fp_direct_active_block_schur_builder = getattr(builders, "fp_direct_active_block_schur_builder", None)
        if fp_direct_active_block_schur_builder is not None:
            if bool(context.use_active_dof_mode) and not use_reduced:
                return builders.sxblock_builder(
                    op=context.op,
                    reduce_full=reduce_full,
                    expand_reduced=expand_reduced,
                )
            return fp_direct_active_block_schur_builder(
                op=context.op,
                reduce_full=reduce_full,
                expand_reduced=expand_reduced,
                active_indices_np=context.active_indices_np if use_reduced else None,
                emit=context.emit,
            )
        return builders.sxblock_builder(op=context.op, reduce_full=reduce_full, expand_reduced=expand_reduced)
    if kind == "sparse_jax":
        precond_dtype = builders.precond_dtype(int(size_est))
        bytes_per = 4.0 if precond_dtype == jnp.float32 else 8.0
        est_mb = (int(size_est) ** 2) * bytes_per / 1.0e6
        if sparse_jax_config.max_mb > 0.0 and est_mb > sparse_jax_config.max_mb:
            if context.emit is not None:
                context.emit(
                    1,
                    "solve_v3_transport_matrix_linear_gmres: sparse_jax preconditioner disabled "
                    f"(est_mem={est_mb:.1f} MB > max_mb={sparse_jax_config.max_mb:.1f})",
                )
            return builders.collision_builder(
                op=context.op,
                reduce_full=reduce_full,
                expand_reduced=expand_reduced,
            )
        cache_suffix = f"sparse_jax_active_{int(size_est)}" if use_reduced else f"sparse_jax_{int(size_est)}"
        cache_key = builders.sparse_jax_cache_key(context.op, cache_suffix)
        if use_reduced:
            def _mv_sparse_reduced(x_reduced: jnp.ndarray, op=context.op) -> jnp.ndarray:
                assert expand_reduced is not None
                assert reduce_full is not None
                y_full = builders.apply_operator_cached(op, expand_reduced(x_reduced))
                return reduce_full(y_full)

            matvec = _mv_sparse_reduced
        else:
            def _mv_sparse_full(x: jnp.ndarray, op=context.op) -> jnp.ndarray:
                return builders.apply_operator_cached(op, x)

            matvec = _mv_sparse_full
        return builders.sparse_jax_builder(
            matvec=matvec,
            n=int(size_est),
            dtype=precond_dtype,
            cache_key=cache_key,
            drop_tol=float(sparse_jax_config.drop_tol),
            drop_rel=float(sparse_jax_config.drop_rel),
            reg=float(sparse_jax_config.reg),
            omega=float(sparse_jax_config.omega),
            sweeps=int(sparse_jax_config.sweeps),
            emit=context.emit,
        )
    if kind in {"sxblock", "block_sx", "species_x"}:
        return builders.sxblock_builder(op=context.op, reduce_full=reduce_full, expand_reduced=expand_reduced)
    if kind in {"block", "block_jacobi"}:
        return builders.block_builder(op=context.op, reduce_full=reduce_full, expand_reduced=expand_reduced)
    return builders.collision_builder(op=context.op, reduce_full=reduce_full, expand_reduced=expand_reduced)


def build_transport_strong_preconditioner_from_kind(
    *,
    kind: str | None,
    use_reduced: bool,
    precond_kind_used: str | None,
    preconditioner_full: Preconditioner | None,
    preconditioner_reduced: Preconditioner | None,
    context: TransportPreconditionerContext,
    builders: TransportPreconditionerDispatchBuilders,
    dd_config: TransportDDConfig,
    sparse_jax_config: TransportSparseJaxConfig,
) -> Preconditioner | None:
    if kind is None:
        return None
    if precond_kind_used is not None and kind == precond_kind_used:
        return preconditioner_reduced if use_reduced else preconditioner_full
    return build_transport_preconditioner_from_kind(
        kind=kind,
        context=context,
        builders=builders,
        dd_config=dd_config,
        sparse_jax_config=sparse_jax_config,
        use_reduced=use_reduced,
    )


@dataclass
class TransportStrongPreconditionerCache:
    """Lazily build full/reduced strong transport preconditioners once per solve."""

    kind: str | None
    precond_kind_used: str | None
    preconditioner_full: Preconditioner | None
    preconditioner_reduced: Preconditioner | None
    context: TransportPreconditionerContext
    builders: TransportPreconditionerDispatchBuilders
    dd_config: TransportDDConfig
    sparse_jax_config: TransportSparseJaxConfig
    strong_full: Preconditioner | None = None
    strong_reduced: Preconditioner | None = None

    def get(self, *, use_reduced: bool) -> Preconditioner | None:
        if self.kind is None:
            return None
        if use_reduced:
            if self.strong_reduced is None:
                self.strong_reduced = build_transport_strong_preconditioner_from_kind(
                    kind=self.kind,
                    use_reduced=True,
                    precond_kind_used=self.precond_kind_used,
                    preconditioner_full=self.preconditioner_full,
                    preconditioner_reduced=self.preconditioner_reduced,
                    context=self.context,
                    builders=self.builders,
                    dd_config=self.dd_config,
                    sparse_jax_config=self.sparse_jax_config,
                )
            return self.strong_reduced
        if self.strong_full is None:
            self.strong_full = build_transport_strong_preconditioner_from_kind(
                kind=self.kind,
                use_reduced=False,
                precond_kind_used=self.precond_kind_used,
                preconditioner_full=self.preconditioner_full,
                preconditioner_reduced=self.preconditioner_reduced,
                context=self.context,
                builders=self.builders,
                dd_config=self.dd_config,
                sparse_jax_config=self.sparse_jax_config,
            )
        return self.strong_full


# Residual-quality gates for transport-matrix solves and workers.
def float_env(name: str) -> float:
    """Parse a positive floating-point threshold from an environment variable."""
    value = os.environ.get(name, "").strip()
    if not value:
        return 0.0
    try:
        return max(0.0, float(value))
    except ValueError:
        return 0.0


def transport_residual_gate_thresholds_from_env(
    *,
    abs_env: str = "SFINCS_JAX_TRANSPORT_ABORT_MAX_RESIDUAL",
    rel_env: str = "SFINCS_JAX_TRANSPORT_ABORT_MAX_RELATIVE_RESIDUAL",
) -> tuple[float, float]:
    """Return absolute and RHS-normalized residual abort thresholds.

    Empty, invalid, or negative values disable the corresponding gate by
    returning ``0.0``. Keeping that normalization here makes downstream failure
    formatting deterministic and avoids treating negative thresholds as a
    separate policy case.
    """
    return float_env(abs_env), float_env(rel_env)


def transport_residual_gate_failure(
    *,
    which_rhs: int,
    residual_norm: float,
    rhs_norm: float,
    max_abs: float,
    max_relative: float,
) -> str | None:
    """Return a diagnostic string when one transport RHS violates residual gates."""
    residual = float(residual_norm)
    rhsn = float(rhs_norm)
    rel = residual / rhsn if np.isfinite(rhsn) and rhsn > 0.0 else float("nan")
    abs_bad = max_abs > 0.0 and (not np.isfinite(residual) or abs(residual) > max_abs)
    rel_bad = max_relative > 0.0 and (not np.isfinite(rel) or abs(rel) > max_relative)
    if not (abs_bad or rel_bad):
        return None
    return (
        f"whichRHS={int(which_rhs)} residual_norm={residual:.6e} "
        f"rhs_norm={rhsn:.6e} relative_residual={rel:.6e}"
    )


def transport_residual_gate_failures_from_arrays(
    *,
    which_rhs_values: Iterable[int],
    residual_norms: Iterable[float],
    rhs_norms: Iterable[float],
    max_abs: float,
    max_relative: float,
) -> list[str]:
    """Return all residual-gate failures in aligned transport worker arrays."""
    failures: list[str] = []
    for which_rhs, residual_norm, rhs_norm in zip(
        which_rhs_values,
        residual_norms,
        rhs_norms,
        strict=False,
    ):
        failure = transport_residual_gate_failure(
            which_rhs=int(which_rhs),
            residual_norm=float(residual_norm),
            rhs_norm=float(rhs_norm),
            max_abs=float(max_abs),
            max_relative=float(max_relative),
        )
        if failure is not None:
            failures.append(failure)
    return failures
