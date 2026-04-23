from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def _load_module():
    repo = Path(__file__).resolve().parents[1]
    path = repo / "examples" / "optimization" / "benchmark_optional_eqx_jaxopt_scheme4_gate.py"
    spec = importlib.util.spec_from_file_location("benchmark_optional_eqx_jaxopt_scheme4_gate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_scheme4_problem_is_deterministic() -> None:
    mod = _load_module()
    theta1, zeta1, amps_target1, bhat_target1 = mod.make_scheme4_problem()
    theta2, zeta2, amps_target2, bhat_target2 = mod.make_scheme4_problem()
    np.testing.assert_allclose(np.asarray(theta1), np.asarray(theta2))
    np.testing.assert_allclose(np.asarray(zeta1), np.asarray(zeta2))
    np.testing.assert_allclose(np.asarray(amps_target1), np.asarray(amps_target2))
    np.testing.assert_allclose(np.asarray(bhat_target1), np.asarray(bhat_target2))


def test_equinox_gate_matches_directional_finite_difference() -> None:
    mod = _load_module()
    result = mod.run_equinox_gate(n_theta=17, n_zeta=17)
    assert result.case == "scheme4_geometry_fit"
    assert result.status == "ok"
    assert result.directional_grad is not None and np.isfinite(result.directional_grad)
    assert result.directional_grad_abs_error is not None
    assert result.directional_grad_abs_error < 1.0e-6


def test_jaxopt_gate_reduces_loss_and_recovers_parameters() -> None:
    mod = _load_module()
    result = mod.run_jaxopt_gate(n_theta=17, n_zeta=17, maxiter=5, stepsize=0.1)
    assert result.case == "scheme4_geometry_fit"
    assert result.status == "ok"
    assert result.initial_loss is not None and result.final_loss is not None
    assert result.final_loss < result.initial_loss
    assert result.loss_ratio is not None and result.loss_ratio < 1.0e-6
    assert result.final_param_error is not None and result.final_param_error < 1.0e-4


def test_equinox_gate_skips_when_module_not_supplied() -> None:
    mod = _load_module()
    result = mod.run_equinox_gate(eqx_module=None)
    assert result.status == "skipped"
    assert "Equinox unavailable" in str(result.error)


def test_jaxopt_gate_skips_when_module_not_supplied() -> None:
    mod = _load_module()
    result = mod.run_jaxopt_gate(eqx_module=object(), jaxopt_module=None)
    assert result.status == "skipped"
    assert "JAXopt unavailable" in str(result.error)


def test_optional_eqx_jaxopt_gate_cli_writes_json(tmp_path: Path) -> None:
    mod = _load_module()
    out_json = tmp_path / "eqx_jaxopt_gate.json"
    rc = mod.main(
        [
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
        ]
    )
    assert rc == 0
    payload = json.loads(out_json.read_text())
    backends = {row["backend"] for row in payload}
    assert backends == {"equinox_wrapper", "jaxopt_gradient_descent"}
    assert all(row["status"] == "ok" for row in payload)
