from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    repo = Path(__file__).resolve().parents[1]
    path = repo / "utils" / "sfincs_jax_driver.py"
    spec = importlib.util.spec_from_file_location("sfincs_jax_driver", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_run_sfincs_jax_utility_defaults_to_explicit_performance_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_module()
    input_path = tmp_path / "input.namelist"
    output_path = tmp_path / "sfincsOutput.h5"
    input_path.write_text("&general\n  RHSMode = 2\n/\n")
    captured: dict[str, object] = {}

    monkeypatch.setattr(mod, "localize_equilibrium_file_in_place", lambda **_kwargs: None)

    def _fake_write_sfincs_output(**kwargs):
        captured.update(kwargs)
        output_path.write_text("placeholder\n")
        return output_path

    monkeypatch.setattr(mod, "write_sfincs_jax_output_h5", _fake_write_sfincs_output)

    out = mod.run_sfincs_jax(
        input_namelist=input_path,
        output_path=output_path,
        verbose=False,
    )

    assert out == output_path
    assert captured["compute_transport_matrix"] is True
    assert captured["compute_solution"] is False
    assert captured["differentiable"] is False


def test_run_sfincs_jax_utility_can_request_differentiable_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_module()
    input_path = tmp_path / "input.namelist"
    output_path = tmp_path / "sfincsOutput.h5"
    input_path.write_text("&general\n  RHSMode = 1\n/\n")
    captured: dict[str, object] = {}

    monkeypatch.setattr(mod, "localize_equilibrium_file_in_place", lambda **_kwargs: None)

    def _fake_write_sfincs_output(**kwargs):
        captured.update(kwargs)
        return output_path

    monkeypatch.setattr(mod, "write_sfincs_jax_output_h5", _fake_write_sfincs_output)

    mod.run_sfincs_jax(
        input_namelist=input_path,
        output_path=output_path,
        verbose=False,
        differentiable=True,
    )

    assert captured["compute_transport_matrix"] is False
    assert captured["compute_solution"] is True
    assert captured["differentiable"] is True

