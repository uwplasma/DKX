from __future__ import annotations

"""Policy helpers for RHSMode=1 sparse polish and retry settings."""

import os


def rhs1_polish_enabled(*, env_name: str) -> bool:
    """Return whether a polish stage is enabled by its boolean-like env var."""
    env = os.environ.get(env_name, "").strip().lower()
    return env not in {"0", "false", "no", "off"}


def rhs1_parse_accept_ratio(*, env_name: str, default: float) -> float:
    """Parse an acceptance ratio with a floor of 1."""
    env = os.environ.get(env_name, "").strip()
    try:
        value = float(env) if env else float(default)
    except ValueError:
        value = float(default)
    return max(1.0, float(value))


def rhs1_parse_polish_gmres_config(
    *,
    restart_env_name: str,
    maxiter_env_name: str,
    default_restart: int,
    default_maxiter: int,
    min_restart: int = 5,
    min_maxiter: int = 5,
) -> tuple[int, int]:
    """Parse bounded restart/maxiter settings for a short GMRES polish."""
    restart_env = os.environ.get(restart_env_name, "").strip()
    maxiter_env = os.environ.get(maxiter_env_name, "").strip()
    try:
        restart = int(restart_env) if restart_env else int(default_restart)
    except ValueError:
        restart = int(default_restart)
    try:
        maxiter = int(maxiter_env) if maxiter_env else int(default_maxiter)
    except ValueError:
        maxiter = int(default_maxiter)
    return max(int(min_restart), int(restart)), max(int(min_maxiter), int(maxiter))


__all__ = [
    "rhs1_parse_accept_ratio",
    "rhs1_parse_polish_gmres_config",
    "rhs1_polish_enabled",
]
