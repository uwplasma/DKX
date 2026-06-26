"""Compatibility shim for the former monolithic v3 driver module.

The RHSMode-1 solve entry points now live in
``sfincs_jax.problems.profile_solve`` and the RHSMode-2/3 transport
entry point lives in ``sfincs_jax.problems.transport_solve``.  This
module intentionally contains no physics equations or solver algorithms; it
only preserves historical imports while the refactor PR migrates tests,
scripts, and docs to the domain-owned modules.
"""

from __future__ import annotations

from importlib import import_module as _import_module
import sys
from typing import Any

_PROFILE_SOLVE = _import_module("sfincs_jax.problems.profile_solve")
_TRANSPORT_SOLVE = _import_module("sfincs_jax.problems.transport_solve")


def _export_public_and_legacy(target: Any, source: Any) -> None:
    """Expose moved public and legacy-private names on the target module."""

    for name, value in vars(source).items():
        if not name.startswith("__"):
            setattr(target, name, value)


_export_public_and_legacy(_PROFILE_SOLVE, _TRANSPORT_SOLVE)


def _transport_parallel_worker(payload: dict[str, object]) -> dict[str, object]:
    """Legacy worker wrapper that honors monkeypatched solve functions."""

    return _PROFILE_SOLVE._solve_transport_parallel_payload(
        payload,
        read_input=_PROFILE_SOLVE.read_sfincs_input,
        solve_transport=_PROFILE_SOLVE.solve_v3_transport_matrix_linear_gmres,
    )


setattr(_PROFILE_SOLVE, "_transport_parallel_worker", _transport_parallel_worker)
setattr(_PROFILE_SOLVE, "__all__", [name for name in vars(_PROFILE_SOLVE) if not name.startswith("__")])

# Make ``import sfincs_jax.v3_driver`` return the real implementation module so
# legacy monkeypatches still mutate the globals used by moved functions.
sys.modules[__name__] = _PROFILE_SOLVE
