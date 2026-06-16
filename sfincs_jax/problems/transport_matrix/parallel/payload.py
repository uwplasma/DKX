"""Payload normalization for transport parallel worker solves.

The in-process CPU worker and the one-GPU-per-worker subprocess both receive a
JSON-like payload describing a subset of ``whichRHS`` values. This module owns
the shared parsing, child-process guards, solve call, and result packing while
receiving the expensive input reader and solver as injected callables. Keeping
those dependencies injected avoids a circular import with ``v3_driver.py`` and
makes the worker contract directly unit-testable.
"""

from __future__ import annotations

from collections.abc import Callable
import os
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np


def solve_transport_parallel_payload(
    payload: dict[str, object],
    *,
    read_input: Callable[[Path], object],
    solve_transport: Callable[..., object],
    emit: Callable[[int, str], None] | None = None,
    set_child_environment: bool = True,
) -> dict[str, object]:
    """Run one transport worker payload and return the merge-ready result dict."""
    input_path = Path(str(payload["input_path"]))
    which_rhs_values = [int(v) for v in payload["which_rhs_values"]]  # type: ignore[assignment]
    tol = float(payload.get("tol", 1e-10))
    atol = float(payload.get("atol", 0.0))
    restart = int(payload.get("restart", 80))
    maxiter = payload.get("maxiter")
    solve_method = str(payload.get("solve_method", "auto"))
    identity_shift = float(payload.get("identity_shift", 0.0))
    differentiable_payload = payload.get("differentiable", None)
    if differentiable_payload is not None:
        differentiable_payload = bool(differentiable_payload)
    phi1_hat_base = payload.get("phi1_hat_base")
    if phi1_hat_base is not None:
        phi1_hat_base = jnp.asarray(phi1_hat_base, dtype=jnp.float64)

    if set_child_environment:
        # Prevent recursive process/GPU worker launches from inside workers.
        os.environ["SFINCS_JAX_TRANSPORT_PARALLEL"] = "off"
        os.environ["SFINCS_JAX_TRANSPORT_PARALLEL_CHILD"] = "1"

    solve_kwargs: dict[str, Any] = {
        "nml": read_input(input_path),
        "tol": tol,
        "atol": atol,
        "restart": restart,
        "maxiter": maxiter,
        "solve_method": solve_method,
        "identity_shift": identity_shift,
        "phi1_hat_base": phi1_hat_base,
        "differentiable": differentiable_payload,
        "input_namelist": input_path,
        "which_rhs_values": which_rhs_values,
        "force_stream_diagnostics": True,
        "force_store_state": True,
        "collect_transport_output_fields": False,
        "parallel_workers": 1,
    }
    if emit is not None:
        solve_kwargs["emit"] = emit
    result = solve_transport(**solve_kwargs)
    return pack_transport_parallel_result(which_rhs_values=which_rhs_values, result=result)


def pack_transport_parallel_result(
    *,
    which_rhs_values: list[int],
    result: object,
) -> dict[str, object]:
    """Convert a transport solve result object to the parent merge payload."""
    return {
        "which_rhs_values": [int(v) for v in which_rhs_values],
        "state_vectors_by_rhs": {
            int(k): np.asarray(v) for k, v in result.state_vectors_by_rhs.items()
        },
        "residual_norms_by_rhs": {
            int(k): float(np.asarray(v)) for k, v in result.residual_norms_by_rhs.items()
        },
        "rhs_norms_by_rhs": {
            int(k): float(np.asarray(v))
            for k, v in (getattr(result, "rhs_norms_by_rhs", None) or {}).items()
        },
        "elapsed_time_s": np.asarray(result.elapsed_time_s, dtype=np.float64),
    }


def transport_parallel_result_to_npz_arrays(result: dict[str, object]) -> dict[str, np.ndarray]:
    """Convert a merge-ready worker result into the subprocess NPZ schema."""
    rhs_values = np.asarray(result.get("which_rhs_values", []), dtype=np.int32)
    if rhs_values.size == 0:
        return {
            "which_rhs_values": rhs_values,
            "state_vectors": np.zeros((0, 0), dtype=np.float64),
            "residual_norms": np.zeros((0,), dtype=np.float64),
            "rhs_norms": np.zeros((0,), dtype=np.float64),
            "elapsed_time_s": np.zeros((0,), dtype=np.float64),
        }

    state_vectors_by_rhs = result.get("state_vectors_by_rhs", {})
    residual_norms_by_rhs = result.get("residual_norms_by_rhs", {})
    rhs_norms_by_rhs = result.get("rhs_norms_by_rhs", {})
    state_vectors = np.stack(
        [np.asarray(state_vectors_by_rhs[int(rhs)], dtype=np.float64) for rhs in rhs_values],
        axis=0,
    )
    residual_norms = np.asarray(
        [float(np.asarray(residual_norms_by_rhs[int(rhs)], dtype=np.float64)) for rhs in rhs_values],
        dtype=np.float64,
    )
    rhs_norms = np.asarray(
        [
            float(np.asarray(rhs_norms_by_rhs.get(int(rhs), np.nan), dtype=np.float64))
            for rhs in rhs_values
        ],
        dtype=np.float64,
    )
    elapsed_full = np.asarray(result.get("elapsed_time_s", np.zeros((0,))), dtype=np.float64)
    if elapsed_full.ndim == 0:
        elapsed_time_s = np.full((rhs_values.size,), float(elapsed_full), dtype=np.float64)
    elif elapsed_full.shape[0] == rhs_values.size:
        elapsed_time_s = elapsed_full.astype(np.float64, copy=False)
    elif elapsed_full.shape[0] > int(np.max(rhs_values - 1)):
        elapsed_time_s = elapsed_full[rhs_values - 1].astype(np.float64, copy=False)
    else:
        elapsed_time_s = np.zeros((rhs_values.size,), dtype=np.float64)
        count = min(rhs_values.size, int(elapsed_full.shape[0]))
        if count > 0:
            elapsed_time_s[:count] = elapsed_full[:count]
    return {
        "which_rhs_values": rhs_values,
        "state_vectors": state_vectors,
        "residual_norms": residual_norms,
        "rhs_norms": rhs_norms,
        "elapsed_time_s": elapsed_time_s,
    }
