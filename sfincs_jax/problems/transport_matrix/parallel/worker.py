from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.problems.transport_matrix.parallel.payload import (
    solve_transport_parallel_payload,
    transport_parallel_result_to_npz_arrays,
)
from sfincs_jax.v3_driver import solve_v3_transport_matrix_linear_gmres


def main() -> int:
    parser = argparse.ArgumentParser(description="GPU transport whichRHS worker.")
    parser.add_argument("--payload", type=Path, required=True, help="Path to worker payload JSON.")
    parser.add_argument("--output", type=Path, required=True, help="Path to output NPZ.")
    args = parser.parse_args()

    payload = json.loads(args.payload.read_text())

    def _emit(_level: int, message: str) -> None:
        print(message, flush=True)

    result = solve_transport_parallel_payload(
        payload,
        read_input=read_sfincs_input,
        solve_transport=solve_v3_transport_matrix_linear_gmres,
        emit=_emit,
    )
    arrays = transport_parallel_result_to_npz_arrays(result)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **arrays)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
