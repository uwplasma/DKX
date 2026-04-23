"""Solve-mode policy helpers shared by driver and I/O entry points."""

from __future__ import annotations

import os


_FALSE_VALUES = {"0", "false", "no", "off"}


def resolve_use_implicit(*, differentiable: bool | None = None) -> bool:
    """Resolve whether to use the implicit/differentiable linear-solve path."""
    if differentiable is not None:
        return bool(differentiable)
    implicit_env = os.environ.get("SFINCS_JAX_IMPLICIT_SOLVE", "").strip().lower()
    return implicit_env not in _FALSE_VALUES


__all__ = ["resolve_use_implicit"]
