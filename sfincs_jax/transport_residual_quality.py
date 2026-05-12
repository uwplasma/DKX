from __future__ import annotations

from collections.abc import Iterable
import os

import numpy as np


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
