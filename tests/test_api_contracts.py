from __future__ import annotations

from dataclasses import FrozenInstanceError, is_dataclass
from pathlib import Path

import pytest

import sfincs_jax
from sfincs_jax.api import (
    BenchmarkReport,
    GeometryState,
    GridState,
    OperatorState,
    OutputSchema,
    PreconditionerState,
    SolveInputs,
    SolverResult,
    TransportResult,
    read_output,
    run_ambipolar_brent,
    write_output,
)


PUBLIC_CONTRACTS = (
    BenchmarkReport,
    GeometryState,
    GridState,
    OperatorState,
    OutputSchema,
    PreconditionerState,
    SolveInputs,
    SolverResult,
    TransportResult,
)


def test_public_api_contracts_are_frozen_dataclasses() -> None:
    """High-level contracts should be stable typed boundaries, not mutable bags."""

    for contract in PUBLIC_CONTRACTS:
        assert is_dataclass(contract), contract.__name__

    inputs = SolveInputs(input_path="input.namelist")
    with pytest.raises(FrozenInstanceError):
        inputs.backend = "gpu"  # type: ignore[misc]


def test_solve_inputs_normalizes_paths_and_freezes_options() -> None:
    inputs = SolveInputs(
        input_path="input.namelist",
        wout_path=Path("wout.nc"),
        output_path="sfincsOutput.h5",
        backend="cpu",
        requires_autodiff=True,
        options={"solver": "auto"},
    )

    assert inputs.input_path == Path("input.namelist")
    assert inputs.wout_path == Path("wout.nc")
    assert inputs.output_path == Path("sfincsOutput.h5")
    assert inputs.backend == "cpu"
    assert inputs.requires_autodiff is True
    assert inputs.options["solver"] == "auto"
    with pytest.raises(TypeError):
        inputs.options["solver"] = "manual"  # type: ignore[index]


def test_state_contract_metadata_is_read_only() -> None:
    geometry = GeometryState(kind="vmec", source_path="wout.nc", radial_coordinate=0.5, metadata={"nfp": 2})
    grid = GridState(n_theta=25, n_zeta=51, n_xi=100, n_x=4, n_species=2, metadata={"layout": "active"})
    operator = OperatorState(rhs_mode=1, size=1000, collision_model="full_fp", include_phi1=True, metadata={"nnz": 9})
    preconditioner = PreconditionerState(kind="auto", differentiable=True, device_safe=True, metadata={"path": "device"})

    assert geometry.source_path == Path("wout.nc")
    assert geometry.metadata["nfp"] == 2
    assert grid.metadata["layout"] == "active"
    assert operator.metadata["nnz"] == 9
    assert preconditioner.metadata["path"] == "device"

    for metadata in (geometry.metadata, grid.metadata, operator.metadata, preconditioner.metadata):
        with pytest.raises(TypeError):
            metadata["new"] = "value"  # type: ignore[index]


def test_result_schema_and_benchmark_contracts_are_immutable_summaries() -> None:
    solver = SolverResult(residual_norm=1.0e-10, converged=True, iterations=12, runtime_s=0.4, metadata={"kind": "gmres"})
    transport = TransportResult(transport_matrix=[[1.0]], solver=solver, metadata={"case": "unit"})
    schema = OutputSchema(format="hdf5", version="1", keys=["FSABFlow", "particleFlux"], path="sfincsOutput.h5")
    report = BenchmarkReport(case="unit", backend="cpu", runtime_s=0.4, peak_memory_mb=128.0, status="pass")

    assert transport.solver is solver
    assert schema.keys == ("FSABFlow", "particleFlux")
    assert schema.path == Path("sfincsOutput.h5")
    assert report.status == "pass"

    with pytest.raises(TypeError):
        solver.metadata["kind"] = "direct"  # type: ignore[index]


def test_contracts_are_reexported_from_top_level_package() -> None:
    assert sfincs_jax.SolveInputs is SolveInputs
    assert sfincs_jax.TransportResult is TransportResult
    assert sfincs_jax.write_output is write_output
    assert sfincs_jax.read_output is read_output
    assert sfincs_jax.run_ambipolar_brent is run_ambipolar_brent
    assert "SolveInputs" in sfincs_jax.__all__
    assert "TransportResult" in sfincs_jax.__all__
    assert "write_output" in sfincs_jax.__all__
    assert "read_output" in sfincs_jax.__all__
    assert "run_ambipolar_brent" in sfincs_jax.__all__


def test_public_write_output_facade_routes_solve_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    def fake_write_sfincs_jax_output_h5(**kwargs):
        calls.append(kwargs)
        return Path(kwargs["output_path"])

    monkeypatch.setattr("sfincs_jax.io.write_sfincs_jax_output_h5", fake_write_sfincs_jax_output_h5)

    request = SolveInputs(
        input_path="input.namelist",
        wout_path="wout.nc",
        output_path="sfincsOutput.h5",
        requires_autodiff=True,
        options={"compute_solution": True, "solve_method": "auto"},
    )

    result = write_output(request)

    assert result == Path("sfincsOutput.h5")
    assert calls == [
        {
            "input_namelist": Path("input.namelist"),
            "output_path": Path("sfincsOutput.h5"),
            "wout_path": Path("wout.nc"),
            "differentiable": True,
            "compute_solution": True,
            "solve_method": "auto",
        }
    ]


def test_public_write_output_requires_input_and_output_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_write_sfincs_jax_output_h5(**_kwargs):
        raise AssertionError("writer should not be called for invalid API requests")

    monkeypatch.setattr("sfincs_jax.io.write_sfincs_jax_output_h5", fake_write_sfincs_jax_output_h5)

    with pytest.raises(ValueError, match="input_path is required"):
        write_output(SolveInputs(output_path="sfincsOutput.h5"))

    with pytest.raises(ValueError, match="output_path is required"):
        write_output(SolveInputs(input_path="input.namelist"))


def test_public_write_output_explicit_kwargs_override_options(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    def fake_write_sfincs_jax_output_h5(**kwargs):
        calls.append(kwargs)
        return Path(kwargs["output_path"])

    monkeypatch.setattr("sfincs_jax.io.write_sfincs_jax_output_h5", fake_write_sfincs_jax_output_h5)

    request = SolveInputs(
        input_path="input.namelist",
        output_path="sfincsOutput.h5",
        requires_autodiff=True,
        options={"solve_method": "auto", "compute_solution": True, "differentiable": False},
    )
    result = write_output(request, solve_method="structured_csr")

    assert result == Path("sfincsOutput.h5")
    assert calls[0]["solve_method"] == "structured_csr"
    assert calls[0]["compute_solution"] is True
    assert calls[0]["differentiable"] is True


def test_public_read_output_facade_routes_output_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Path] = []

    def fake_read_sfincs_output_file(path: Path) -> dict:
        calls.append(path)
        return {"ok": True}

    monkeypatch.setattr("sfincs_jax.outputs.read_sfincs_output_file", fake_read_sfincs_output_file)

    assert read_output("sfincsOutput.npz") == {"ok": True}
    assert calls == [Path("sfincsOutput.npz")]


def test_public_ambipolar_facade_routes_brent_solver(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    def fake_solve_sfincs_jax_ambipolar_brent(**kwargs):
        calls.append(kwargs)
        return "ambipolar-result"

    monkeypatch.setattr(
        "sfincs_jax.problems.ambipolar.solve_sfincs_jax_ambipolar_brent",
        fake_solve_sfincs_jax_ambipolar_brent,
    )

    request = SolveInputs(input_path="input.namelist", requires_autodiff=True, options={"emit": None})

    assert run_ambipolar_brent(
        request,
        work_dir="ambipolar-work",
        er_min=-5.0,
        er_max=5.0,
        max_evaluations=7,
    ) == "ambipolar-result"
    assert calls == [
        {
            "input_namelist": Path("input.namelist"),
            "work_dir": "ambipolar-work",
            "er_min": -5.0,
            "er_max": 5.0,
            "er_initial": 0.0,
            "max_evaluations": 7,
            "current_tolerance": 1.0e-10,
            "step_tolerance": 1.0e-8,
            "solve_method": "auto",
            "differentiable": True,
            "reuse_output_geometry_cache": True,
            "reuse_solver_state": True,
            "emit": None,
        }
    ]


def test_public_ambipolar_facade_honors_explicit_differentiable_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []

    def fake_solve_sfincs_jax_ambipolar_brent(**kwargs):
        calls.append(kwargs)
        return "ambipolar-result"

    monkeypatch.setattr(
        "sfincs_jax.problems.ambipolar.solve_sfincs_jax_ambipolar_brent",
        fake_solve_sfincs_jax_ambipolar_brent,
    )

    request = SolveInputs(input_path="input.namelist", requires_autodiff=True, options={"extra": 3})
    assert run_ambipolar_brent(
        request,
        work_dir="ambipolar-work",
        er_min=-1.0,
        er_max=1.0,
        differentiable=False,
    ) == "ambipolar-result"

    assert calls[0]["differentiable"] is False
    assert calls[0]["extra"] == 3
