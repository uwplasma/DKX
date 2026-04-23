from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def _load_module():
    repo = Path(__file__).resolve().parents[1]
    path = repo / "examples" / "performance" / "benchmark_optional_lineax_implicit_solve.py"
    spec = importlib.util.spec_from_file_location("benchmark_optional_lineax_implicit_solve", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_nonsymmetric_gate_system_is_deterministic_and_nonsymmetric() -> None:
    mod = _load_module()
    a1, b1 = mod.make_nonsymmetric_system(5)
    a2, b2 = mod.make_nonsymmetric_system(5)
    np.testing.assert_allclose(np.asarray(a1), np.asarray(a2))
    np.testing.assert_allclose(np.asarray(b1), np.asarray(b2))
    assert np.linalg.norm(np.asarray(a1 - a1.T)) > 0.0
    assert np.all(np.diag(np.asarray(a1)) > 2.5)


def test_current_implicit_gate_returns_finite_gradient_and_small_residual() -> None:
    mod = _load_module()
    matrix, rhs = mod.make_nonsymmetric_system(4)
    result = mod.run_current_gate(
        case="synthetic_nonsymmetric",
        matrix=matrix,
        rhs=rhs,
        p0=0.2,
        tol=1.0e-10,
        restart=4,
        maxiter=60,
    )
    assert result.status == "ok"
    assert result.relative_residual is not None
    assert result.relative_residual < 1.0e-8
    assert result.grad is not None and np.isfinite(result.grad)
    assert result.grad_abs_error is not None
    assert result.grad_abs_error < 1.0e-5


def test_lineax_gate_skips_cleanly_when_lineax_is_not_supplied() -> None:
    mod = _load_module()
    matrix, rhs = mod.make_nonsymmetric_system(4)
    result = mod.run_lineax_gate(
        case="synthetic_nonsymmetric",
        matrix=matrix,
        rhs=rhs,
        p0=0.2,
        tol=1.0e-10,
        restart=4,
        maxiter=60,
        lineax_module=None,
    )
    assert result.status == "skipped"
    assert result.backend == "lineax_gmres"
    assert "Lineax unavailable" in str(result.error)


def test_optional_lineax_gate_cli_writes_json_for_current_backend(tmp_path: Path) -> None:
    mod = _load_module()
    out_json = tmp_path / "lineax_gate.json"
    rc = mod.main(
        [
            "--backend",
            "current",
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
        ]
    )
    assert rc == 0
    payload = json.loads(out_json.read_text())
    assert payload[0]["case"] == "synthetic_nonsymmetric"
    assert payload[0]["backend"] == "current_custom_linear_solve"
    assert payload[0]["status"] == "ok"
    assert payload[0]["relative_residual"] < 1.0e-8


def test_load_tiny_sfincs_fixture_returns_scheme5_operator_and_reference_state() -> None:
    mod = _load_module()
    op0, x_ref, nu0 = mod.load_tiny_sfincs_fixture(str(mod._default_sfincs_input()))
    assert int(op0.total_size) == int(x_ref.shape[0])
    assert np.isfinite(float(nu0))
    assert int(x_ref.size) > 0


def test_current_sfincs_implicit_gate_returns_finite_gradient_and_small_residual() -> None:
    mod = _load_module()
    result = mod.run_current_sfincs_implicit_gate(
        input_path=mod._default_sfincs_input(),
        tol=1.0e-10,
        restart=20,
        maxiter=120,
    )
    assert result.case == "sfincs_tiny_implicit"
    assert result.status == "ok"
    assert result.relative_residual is not None
    assert result.relative_residual < 1.0e-8
    assert result.grad is not None and np.isfinite(result.grad)
    assert result.grad_abs_error is not None
    assert result.grad_abs_error < 1.0e-4


def test_current_sfincs_repeated_rhs_gate_returns_small_residual_and_solution_error() -> None:
    mod = _load_module()
    result = mod.run_current_sfincs_repeated_rhs_gate(
        input_path=mod._default_sfincs_input(),
        tol=1.0e-10,
        restart=20,
        maxiter=120,
    )
    assert result.case == "sfincs_tiny_repeated_rhs"
    assert result.status == "ok"
    assert result.n_rhs == 2
    assert result.relative_residual is not None
    assert result.relative_residual < 1.0e-8
    assert result.max_solution_error is not None
    assert result.max_solution_error < 1.0e-6


def test_lineax_sfincs_repeated_rhs_gate_skips_cleanly_when_lineax_missing() -> None:
    mod = _load_module()
    result = mod.run_lineax_sfincs_repeated_rhs_gate(
        input_path=mod._default_sfincs_input(),
        tol=1.0e-10,
        restart=20,
        maxiter=120,
        lineax_module=None,
    )
    assert result.case == "sfincs_tiny_repeated_rhs"
    assert result.status == "skipped"
    assert "Lineax unavailable" in str(result.error)
