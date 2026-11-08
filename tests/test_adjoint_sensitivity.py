"""DKX reverse-mode sensitivities against DKX central differences (the RHSMode 4/5 replacement).

The committed record ``validation/adjoint_sensitivity_v1.json`` holds the tokamak-like and W7-X
cases; the tokamak-like one is recomputed here, since it needs no equilibrium file.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools" / "paper_benchmarks"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_committed_record_passes_its_audit_and_rejects_a_tampered_derivative(
    tmp_path: Path,
) -> None:
    audit = _load("audit_adjoint_sensitivity").audit
    record = ROOT / "validation" / "adjoint_sensitivity_v1.json"
    assert audit(record)["pass"]
    payload = json.loads(record.read_text())
    payload["cases"][0]["adjoint"][1][3] *= 1.01
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(payload))
    assert not audit(tampered)["pass"]


def test_tokamak_like_adjoint_matches_central_differences() -> None:
    case = _load("adjoint_sensitivity_verification").run_case("tokamak")
    adjoint = np.asarray(case["adjoint"])
    central = np.asarray(case["central_difference"]["0.0001"])
    assert adjoint.shape == (4, 5)
    assert np.all(
        np.abs(central - adjoint).max(axis=1) < 1e-5 * np.abs(adjoint).max(axis=1)
    )
