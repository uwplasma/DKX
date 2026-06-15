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
    assert "SolveInputs" in sfincs_jax.__all__
    assert "TransportResult" in sfincs_jax.__all__
