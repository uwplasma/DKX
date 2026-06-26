#!/usr/bin/env python
"""Bounded probes for optional JAX ecosystem RHSMode=1 backends."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.operators.profile_layout import RHS1BlockLayout
from sfincs_jax.operators.profile_full_system import (
    build_structured_rhs1_full_csr_preconditioner,
    clear_structured_rhs1_full_csr_cache,
    select_structured_rhs1_full_csr_operator,
)
from sfincs_jax.operators.profile_system import full_system_operator_from_namelist, rhs_v3_full_system


_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = _REPO_ROOT / "tests" / "ref" / "quick_2species_FPCollisions_noEr.input.namelist"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--identity-shift", type=float, default=0.5)
    parser.add_argument("--max-csr-mb", type=float, default=512.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--lineax-max-dense-size", type=int, default=512)
    parser.add_argument("--native-max-factor-mb", type=float, default=256.0)
    parser.add_argument("--json", action="store_true")
    return parser


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _max_nbytes_from_mb(value: float) -> int:
    return int(max(0.0, float(value)) * 1024.0 * 1024.0)


def _block_until_ready(value: Any) -> Any:
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
    return value


def _timed_repeated(fn, *, repeats: int) -> dict[str, Any]:
    times: list[float] = []
    last = None
    for _ in range(max(1, int(repeats))):
        t0 = time.perf_counter()
        last = _block_until_ready(fn())
        times.append(max(0.0, time.perf_counter() - t0))
    return {
        "times_s": times,
        "min_s": min(times),
        "mean_s": float(np.mean(times)),
        "last": last,
    }


def _norm(value: Any) -> float:
    return float(np.linalg.norm(np.asarray(value, dtype=np.float64).reshape((-1,))))


def _optional_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    clear_structured_rhs1_full_csr_cache(clear_fblock_cache=True)
    nml = read_sfincs_input(args.input)
    op = full_system_operator_from_namelist(nml=nml, identity_shift=float(args.identity_shift))
    payload: dict[str, Any] = {
        "input": str(args.input),
        "rhs_mode": int(op.rhs_mode),
        "shape": {
            "total_size": int(op.total_size),
            "f_size": int(op.f_size),
            "n_species": int(op.n_species),
            "n_x": int(op.n_x),
            "n_xi": int(op.n_xi),
            "n_theta": int(op.n_theta),
            "n_zeta": int(op.n_zeta),
        },
        "optional_libraries": {
            name: _optional_available(name)
            for name in ("lineax", "interpax", "equinox", "jaxopt", "optax", "quadax", "orthax")
        },
        "probes": {},
    }
    if int(op.rhs_mode) != 1:
        payload["selected"] = False
        payload["reason"] = f"unsupported_rhs_mode:{int(op.rhs_mode)}"
        return payload

    selected = select_structured_rhs1_full_csr_operator(
        op,
        max_csr_nbytes=_max_nbytes_from_mb(float(args.max_csr_mb)),
    )
    payload["structured_full_csr"] = selected.to_dict()
    if not selected.selected or selected.matrix is None:
        payload["selected"] = False
        payload["reason"] = str(selected.reason)
        return payload

    matrix = selected.matrix.tocsr()
    rhs = np.asarray(rhs_v3_full_system(op), dtype=np.float64)
    rhs_norm = _norm(rhs)
    vector = jnp.asarray(np.sin(0.13 * np.arange(matrix.shape[1], dtype=np.float64)))
    payload["selected"] = True
    payload["rhs_norm"] = rhs_norm
    payload["matrix"] = {"shape": tuple(int(v) for v in matrix.shape), "nnz": int(matrix.nnz)}

    try:
        from jax.experimental import sparse as jsparse

        t0 = time.perf_counter()
        bcoo = jsparse.BCOO.from_scipy_sparse(matrix)
        convert_s = max(0.0, time.perf_counter() - t0)
        matvec = jax.jit(lambda x: bcoo @ x)
        expected = np.asarray(matrix @ np.asarray(vector))
        _block_until_ready(matvec(vector))
        timing = _timed_repeated(lambda: matvec(vector), repeats=int(args.repeats))
        actual = np.asarray(timing.pop("last"))
        payload["probes"]["jax_sparse_bcoo_matvec"] = {
            "selected": True,
            "convert_s": convert_s,
            "nse": int(bcoo.nse),
            "timing": timing,
            "relative_error": _norm(actual - expected) / max(_norm(expected), 1.0e-300),
        }
    except Exception as exc:  # pragma: no cover - diagnostic payload path.
        payload["probes"]["jax_sparse_bcoo_matvec"] = {
            "selected": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }

    layout = RHS1BlockLayout.from_operator(op)
    for preconditioner_kind, payload_key in (
        ("native_xell", "native_xell_preconditioner"),
        ("native_xell_tail_schur", "native_xell_tail_schur_preconditioner"),
    ):
        native_pc = build_structured_rhs1_full_csr_preconditioner(
            matrix=matrix,
            layout=layout,
            kind=preconditioner_kind,
            max_block_inverse_nbytes=_max_nbytes_from_mb(float(args.native_max_factor_mb)),
        )
        native_payload = native_pc.to_dict()
        if native_pc.selected and native_pc.operator is not None:
            _ = native_pc.operator.matvec(rhs)
            timing = _timed_repeated(lambda: native_pc.operator.matvec(rhs), repeats=int(args.repeats))
            correction = np.asarray(timing.pop("last"), dtype=np.float64)
            residual = rhs - np.asarray(matrix @ correction, dtype=np.float64)
            native_payload.update(
                {
                    "timing": timing,
                    "one_step_residual_norm": _norm(residual),
                    "one_step_residual_over_rhs": _norm(residual) / max(rhs_norm, 1.0e-300),
                }
            )
        payload["probes"][payload_key] = native_payload

    if _optional_available("lineax") and int(matrix.shape[0]) <= int(args.lineax_max_dense_size):
        import lineax as lx

        dense = jnp.asarray(matrix.toarray(), dtype=jnp.float64)
        rhs_jax = jnp.asarray(rhs, dtype=jnp.float64)
        operator = lx.MatrixLinearOperator(dense)
        solver = lx.GMRES(rtol=1.0e-10, atol=1.0e-12, max_steps=128, restart=32)
        _ = lx.linear_solve(operator, rhs_jax, solver, throw=True).value.block_until_ready()
        timing = _timed_repeated(
            lambda: lx.linear_solve(operator, rhs_jax, solver, throw=True).value,
            repeats=int(args.repeats),
        )
        solution = np.asarray(timing.pop("last"), dtype=np.float64)
        payload["probes"]["lineax_dense_gmres"] = {
            "selected": True,
            "timing": timing,
            "residual_norm": _norm(rhs - np.asarray(matrix @ solution, dtype=np.float64)),
        }
    else:
        payload["probes"]["lineax_dense_gmres"] = {
            "selected": False,
            "reason": (
                "lineax_unavailable"
                if not _optional_available("lineax")
                else f"dense_size_exceeded:{int(matrix.shape[0])}>{int(args.lineax_max_dense_size)}"
            ),
        }

    return payload


def main() -> None:
    args = _build_parser().parse_args()
    payload = run_probe(args)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")
    if args.json or args.out is None:
        print(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
