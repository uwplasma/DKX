from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from .namelist import read_sfincs_input
from .v3_driver import solve_v3_transport_matrix_linear_gmres


def main() -> int:
    parser = argparse.ArgumentParser(description="GPU transport whichRHS worker.")
    parser.add_argument("--payload", type=Path, required=True, help="Path to worker payload JSON.")
    parser.add_argument("--output", type=Path, required=True, help="Path to output NPZ.")
    args = parser.parse_args()

    payload = json.loads(args.payload.read_text())
    input_path = Path(str(payload["input_path"]))
    which_rhs_values = [int(v) for v in payload["which_rhs_values"]]
    tol = float(payload.get("tol", 1e-10))
    atol = float(payload.get("atol", 0.0))
    restart = int(payload.get("restart", 80))
    maxiter = payload.get("maxiter")
    solve_method = str(payload.get("solve_method", "auto"))
    identity_shift = float(payload.get("identity_shift", 0.0))
    phi1_hat_base = payload.get("phi1_hat_base")
    if phi1_hat_base is not None:
        phi1_hat_base = jnp.asarray(phi1_hat_base, dtype=jnp.float64)

    nml = read_sfincs_input(input_path)
    result = solve_v3_transport_matrix_linear_gmres(
        nml=nml,
        tol=tol,
        atol=atol,
        restart=restart,
        maxiter=maxiter,
        solve_method=solve_method,
        identity_shift=identity_shift,
        phi1_hat_base=phi1_hat_base,
        input_namelist=input_path,
        which_rhs_values=which_rhs_values,
        force_stream_diagnostics=True,
        force_store_state=True,
        collect_transport_output_fields=False,
        parallel_workers=1,
    )

    rhs_values = np.asarray(which_rhs_values, dtype=np.int32)
    if rhs_values.size > 0:
        state_vectors = np.stack(
            [np.asarray(result.state_vectors_by_rhs[int(rhs)], dtype=np.float64) for rhs in rhs_values],
            axis=0,
        )
        residual_norms = np.asarray(
            [float(np.asarray(result.residual_norms_by_rhs[int(rhs)], dtype=np.float64)) for rhs in rhs_values],
            dtype=np.float64,
        )
        elapsed_time_s = np.asarray(
            [float(np.asarray(result.elapsed_time_s[int(rhs) - 1], dtype=np.float64)) for rhs in rhs_values],
            dtype=np.float64,
        )
    else:
        state_vectors = np.zeros((0, 0), dtype=np.float64)
        residual_norms = np.zeros((0,), dtype=np.float64)
        elapsed_time_s = np.zeros((0,), dtype=np.float64)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        which_rhs_values=rhs_values,
        state_vectors=state_vectors,
        residual_norms=residual_norms,
        elapsed_time_s=elapsed_time_s,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
