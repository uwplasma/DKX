"""Policy helpers for RHSMode=1 sparse polish and retry settings."""

from __future__ import annotations

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
    active_size: int | None = None,
    large_active_min_env_name: str = "",
    large_default_restart_env_name: str = "",
    large_default_maxiter_env_name: str = "",
    default_large_restart: int | None = None,
    default_large_maxiter: int | None = None,
) -> tuple[int, int]:
    """Parse bounded restart/maxiter settings for a short GMRES polish."""
    restart_env = os.environ.get(restart_env_name, "").strip()
    maxiter_env = os.environ.get(maxiter_env_name, "").strip()
    default_restart_use = int(default_restart)
    default_maxiter_use = int(default_maxiter)
    if active_size is not None and (default_large_restart is not None or default_large_maxiter is not None):
        large_min_env = os.environ.get(large_active_min_env_name, "").strip() if large_active_min_env_name else ""
        large_restart_env = (
            os.environ.get(large_default_restart_env_name, "").strip() if large_default_restart_env_name else ""
        )
        large_default_env = (
            os.environ.get(large_default_maxiter_env_name, "").strip() if large_default_maxiter_env_name else ""
        )
        try:
            large_min = int(large_min_env) if large_min_env else 200000
        except ValueError:
            large_min = 200000
        if int(active_size) >= max(1, int(large_min)):
            if default_large_restart is not None and not restart_env:
                try:
                    large_restart = int(large_restart_env) if large_restart_env else int(default_large_restart)
                except ValueError:
                    large_restart = int(default_large_restart)
                default_restart_use = min(int(default_restart_use), max(1, int(large_restart)))
            if default_large_maxiter is not None and not maxiter_env:
                try:
                    large_default = int(large_default_env) if large_default_env else int(default_large_maxiter)
                except ValueError:
                    large_default = int(default_large_maxiter)
                default_maxiter_use = min(int(default_maxiter_use), max(1, int(large_default)))
    try:
        restart = int(restart_env) if restart_env else int(default_restart_use)
    except ValueError:
        restart = int(default_restart_use)
    try:
        maxiter = int(maxiter_env) if maxiter_env else int(default_maxiter_use)
    except ValueError:
        maxiter = int(default_maxiter_use)
    return max(int(min_restart), int(restart)), max(int(min_maxiter), int(maxiter))


__all__ = [
    "rhs1_parse_accept_ratio",
    "rhs1_parse_polish_gmres_config",
    "rhs1_polish_enabled",
]
