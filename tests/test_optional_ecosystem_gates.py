from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


_REPO = Path(__file__).resolve().parents[1]
_LINEAX_SCRIPT = _REPO / "examples" / "performance" / "benchmark_optional_lineax_implicit_solve.py"
_EQX_JAXOPT_SCRIPT = _REPO / "examples" / "optimization" / "benchmark_optional_eqx_jaxopt_scheme4_gate.py"


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _run_json(cmd: list[str], out_json: Path) -> list[dict[str, Any]]:
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    return payload


def _measured_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row["status"]) for row in rows)
    measured_rows = [
        row for row in rows if row.get("status") == "ok" and row.get("elapsed_s") is not None
    ]
    return {
        "rows": len(rows),
        "status_counts": dict(counts),
        "measured_rows": len(measured_rows),
        "backends": sorted(str(row["backend"]) for row in rows),
    }


def test_optional_lineax_synthetic_gate_emits_measured_summary(tmp_path: Path) -> None:
    out_json = tmp_path / "lineax_synthetic_gate.json"
    rows = _run_json(
        [
            sys.executable,
            str(_LINEAX_SCRIPT),
            "--backend",
            "all",
            "--suite",
            "synthetic",
            "--size",
            "4",
            "--restart",
            "4",
            "--maxiter",
            "60",
            "--out-json",
            str(out_json),
        ],
        out_json,
    )
    summary = _measured_summary(rows)
    by_backend = {row["backend"]: row for row in rows}

    assert summary["rows"] == 2
    assert summary["measured_rows"] >= 1
    assert set(summary["backends"]) == {"current_custom_linear_solve", "lineax_gmres"}

    current = by_backend["current_custom_linear_solve"]
    assert current["status"] == "ok"
    assert current["relative_residual"] < 1.0e-8
    assert current["grad_abs_error"] < 1.0e-5
    assert current["elapsed_s"] >= 0.0

    lineax = by_backend["lineax_gmres"]
    if _has_module("lineax"):
        assert lineax["status"] == "ok"
        assert lineax["relative_residual"] < 1.0e-8
        assert lineax["grad_abs_error"] < 1.0e-5
        assert lineax["elapsed_s"] >= 0.0
    else:
        assert lineax["status"] == "skipped"
        assert "Lineax unavailable" in str(lineax["error"])


def test_optional_equinox_jaxopt_gate_emits_measured_summary(tmp_path: Path) -> None:
    out_json = tmp_path / "eqx_jaxopt_gate.json"
    rows = _run_json(
        [
            sys.executable,
            str(_EQX_JAXOPT_SCRIPT),
            "--backend",
            "all",
            "--n-theta",
            "17",
            "--n-zeta",
            "17",
            "--maxiter",
            "5",
            "--stepsize",
            "0.1",
            "--out-json",
            str(out_json),
        ],
        out_json,
    )
    summary = _measured_summary(rows)
    by_backend = {row["backend"]: row for row in rows}

    assert summary["rows"] == 2
    assert set(summary["backends"]) == {"equinox_wrapper", "jaxopt_gradient_descent"}

    eqx = by_backend["equinox_wrapper"]
    if _has_module("equinox"):
        assert eqx["status"] == "ok"
        assert eqx["directional_grad_abs_error"] < 1.0e-6
        assert eqx["elapsed_s"] >= 0.0
    else:
        assert eqx["status"] == "skipped"
        assert "Equinox unavailable" in str(eqx["error"])

    jaxopt = by_backend["jaxopt_gradient_descent"]
    if _has_module("equinox") and _has_module("jaxopt"):
        assert jaxopt["status"] == "ok"
        assert jaxopt["loss_ratio"] < 1.0e-6
        assert jaxopt["final_param_error"] < 1.0e-4
        assert jaxopt["elapsed_s"] >= 0.0
    else:
        assert jaxopt["status"] == "skipped"
        assert "unavailable" in str(jaxopt["error"])
