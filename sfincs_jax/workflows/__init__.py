"""End-to-end workflows for scans, optimization, and publication figures."""

from __future__ import annotations

import sys as _sys

from . import optimization as optimization

# Historical optimization_* modules are compatibility aliases to the durable
# optimization owner. This keeps existing user imports working without retaining
# one implementation file per workflow stage.
for _name in (
    "optimization_comparison",
    "optimization_evidence",
    "optimization_ladder",
    "optimization_objectives",
    "optimization_promotion",
    "optimization_workflow",
):
    _sys.modules[f"{__name__}.{_name}"] = optimization

__all__ = ("optimization",)
