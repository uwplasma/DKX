from __future__ import annotations

from dataclasses import FrozenInstanceError, is_dataclass
import importlib
from pathlib import Path
from types import SimpleNamespace

import jax.distributed as jax_distributed
import pytest

import dkx
from dkx.api import (
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
    assert dkx.SolveInputs is SolveInputs
    assert dkx.TransportResult is TransportResult
    assert dkx.write_output is write_output
    assert dkx.read_output is read_output
    assert dkx.run_ambipolar_brent is run_ambipolar_brent
    assert "SolveInputs" in dkx.__all__
    assert "TransportResult" in dkx.__all__
    assert "write_output" in dkx.__all__
    assert "read_output" in dkx.__all__
    assert "run_ambipolar_brent" in dkx.__all__
    assert "initialize_distributed_runtime_from_env" in dkx.__all__
    assert isinstance(dkx.__version__, str)
    assert dkx.__version__


def test_flagship_capabilities_are_exported_from_top_level_package() -> None:
    """The productized API surface: typed inputs, runners, solver knobs, scans."""

    from dkx.api import SolverOptions
    from dkx.inputs import SfincsInput, load_sfincs_input

    assert dkx.SfincsInput is SfincsInput
    assert dkx.load_sfincs_input is load_sfincs_input
    assert dkx.SolverOptions is SolverOptions

    lazy_exports = {
        "run_profile": "dkx.run",
        "run_transport_matrix": "dkx.run",
        "run_from_namelist": "dkx.run",
        "batched_solve": "dkx.batch",
        "monoenergetic_database": "dkx.monoenergetic",
        "ambipolar_er": "dkx.er",
        "find_ambipolar_er": "dkx.er",
        "classical_impurity_flux": "dkx.impurity",
        "build_impurity_plasma": "dkx.impurity",
    }
    for name, owner in lazy_exports.items():
        assert name in dkx.__all__, name
        resolved = getattr(dkx, name)
        assert callable(resolved), name
        assert resolved.__module__ == owner, name
        assert name in dir(dkx), name

    for name in ("SfincsInput", "load_sfincs_input", "SolverOptions", "batched_er_scan",
                 "run_monoenergetic_database"):  # fmt: skip
        assert name in dkx.__all__, name

    with pytest.raises(AttributeError, match="no attribute"):
        getattr(dkx, "no_such_flagship_capability")


def test_solver_options_is_a_frozen_contract_with_solve_kwargs() -> None:
    from dkx.api import SolverOptions

    options = SolverOptions(method="gmres", tol=1e-8, differentiable=True, memory_budget_gb=4.0)
    with pytest.raises(FrozenInstanceError):
        options.tol = 1e-6  # type: ignore[misc]

    kwargs = options.solve_kwargs()
    assert kwargs["method"] == "gmres"
    assert kwargs["tol"] == 1e-8
    assert kwargs["differentiable"] is True
    assert kwargs["tier1_memory_budget_gb"] == 4.0
    assert "cores" not in kwargs  # honest: threads are pinned pre-import (DKX_CORES/--cores)


def test_import_env_controls_solver_threads_and_compilation_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_dir = tmp_path / "jax-cache"
    monkeypatch.setenv("DKX_CORES", "2")
    # DKX_XLA_THREADS is inert: it must neither abort the import nor inject
    # the nonexistent --xla_cpu_parallelism_threads flag.
    monkeypatch.setenv("DKX_XLA_THREADS", "yes")
    monkeypatch.delenv("DKX_CPU_DEVICES", raising=False)
    monkeypatch.delenv("NPROC", raising=False)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    monkeypatch.setenv("DKX_COMPILATION_CACHE_DIR", str(cache_dir))
    monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR", raising=False)
    monkeypatch.setenv("XLA_FLAGS", "")

    reloaded = importlib.reload(dkx)

    assert reloaded is dkx
    # --cores/DKX_CORES pins the XLA host threadpool (NPROC) and the host
    # BLAS pools; it must NOT force host devices or touch XLA_FLAGS.
    assert dkx.os.environ["NPROC"] == "2"
    assert dkx.os.environ["OMP_NUM_THREADS"] == "2"
    assert dkx.os.environ["OPENBLAS_NUM_THREADS"] == "2"
    assert "DKX_CPU_DEVICES" not in dkx.os.environ
    assert dkx.os.environ["XLA_FLAGS"] == ""
    assert dkx.os.environ["JAX_COMPILATION_CACHE_DIR"] == str(cache_dir)
    assert cache_dir.is_dir()

    monkeypatch.delenv("DKX_CORES", raising=False)
    monkeypatch.delenv("DKX_XLA_THREADS", raising=False)
    monkeypatch.delenv("DKX_COMPILATION_CACHE_DIR", raising=False)
    monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR", raising=False)
    monkeypatch.setenv("DKX_DISABLE_COMPILATION_CACHE", "1")
    importlib.reload(dkx)


def test_import_default_thread_clamp_and_zero_opt_out(monkeypatch: pytest.MonkeyPatch) -> None:
    import os as os_mod

    monkeypatch.setenv("DKX_DISABLE_COMPILATION_CACHE", "1")

    # No DKX_CORES and no NPROC: the import clamps the threadpool to
    # min(8, cpu_count) and marks the clamp as dkx-owned.
    monkeypatch.delenv("DKX_CORES", raising=False)
    monkeypatch.delenv("NPROC", raising=False)
    monkeypatch.delenv("_DKX_NPROC_DEFAULTED", raising=False)
    importlib.reload(dkx)
    assert dkx.os.environ["NPROC"] == str(min(8, os_mod.cpu_count() or 1))
    assert dkx.os.environ["_DKX_NPROC_DEFAULTED"] == "1"

    # A user-set NPROC always wins over the default clamp.
    monkeypatch.delenv("_DKX_NPROC_DEFAULTED", raising=False)
    monkeypatch.setenv("NPROC", "3")
    importlib.reload(dkx)
    assert dkx.os.environ["NPROC"] == "3"
    assert "_DKX_NPROC_DEFAULTED" not in dkx.os.environ

    # DKX_CORES=0 means "let XLA size the threadpool": it removes a
    # dkx-defaulted clamp (e.g. inherited across the CLI re-exec) but
    # never a user-set NPROC.
    monkeypatch.setenv("DKX_CORES", "0")
    monkeypatch.setenv("NPROC", "8")
    monkeypatch.setenv("_DKX_NPROC_DEFAULTED", "1")
    importlib.reload(dkx)
    assert "NPROC" not in dkx.os.environ
    monkeypatch.setenv("NPROC", "5")
    importlib.reload(dkx)
    assert dkx.os.environ["NPROC"] == "5"

    monkeypatch.delenv("DKX_CORES", raising=False)


def test_import_env_explicit_cpu_devices_forces_host_device_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DKX_CORES", raising=False)
    monkeypatch.setenv("DKX_CPU_DEVICES", "2")
    monkeypatch.setenv("DKX_DISABLE_COMPILATION_CACHE", "1")
    monkeypatch.setenv("XLA_FLAGS", "")

    importlib.reload(dkx)

    assert "--xla_force_host_platform_device_count=2" in dkx.os.environ["XLA_FLAGS"]


def test_import_env_invalid_cpu_controls_are_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DKX_CORES", "not-an-int")
    monkeypatch.setenv("DKX_CPU_DEVICES", "not-an-int")
    monkeypatch.setenv("DKX_DISABLE_COMPILATION_CACHE", "1")
    monkeypatch.setenv("XLA_FLAGS", "")
    monkeypatch.delenv("NPROC", raising=False)

    importlib.reload(dkx)

    assert dkx.os.environ["XLA_FLAGS"] == ""
    assert "NPROC" not in dkx.os.environ


def test_distributed_runtime_env_bootstrap_is_safe_and_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dkx, "_distributed_runtime_initialized", True)
    assert dkx.initialize_distributed_runtime_from_env() is True

    monkeypatch.setattr(dkx, "_distributed_runtime_initialized", False)
    monkeypatch.delenv("DKX_DISTRIBUTED", raising=False)
    assert dkx.initialize_distributed_runtime_from_env() is False

    monkeypatch.setenv("DKX_DISTRIBUTED", "1")
    monkeypatch.delenv("DKX_COORDINATOR_ADDRESS", raising=False)
    assert dkx.initialize_distributed_runtime_from_env() is False


def test_distributed_runtime_env_bootstrap_parses_and_calls_jax(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_initialize(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(dkx, "_distributed_runtime_initialized", False)
    monkeypatch.setattr(jax_distributed, "initialize", fake_initialize)
    monkeypatch.setenv("DKX_DISTRIBUTED", "true")
    monkeypatch.setenv("DKX_COORDINATOR_ADDRESS", "127.0.0.1")
    monkeypatch.setenv("DKX_COORDINATOR_PORT", "3456")
    monkeypatch.setenv("DKX_PROCESS_COUNT", "4")
    monkeypatch.setenv("DKX_PROCESS_ID", "2")

    assert dkx.initialize_distributed_runtime_from_env() is True
    assert calls == [
        {
            "coordinator_address": "127.0.0.1",
            "coordinator_port": 3456,
            "num_processes": 4,
            "process_id": 2,
        }
    ]
    assert dkx.initialize_distributed_runtime_from_env() is True
    assert len(calls) == 1


def test_distributed_runtime_env_bootstrap_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_initialize(**_kwargs):
        raise RuntimeError("backend unavailable")

    monkeypatch.setattr(dkx, "_distributed_runtime_initialized", False)
    monkeypatch.setattr(jax_distributed, "initialize", fake_initialize)
    monkeypatch.setenv("DKX_DISTRIBUTED", "yes")
    monkeypatch.setenv("DKX_COORDINATOR_ADDRESS", "127.0.0.1")

    assert dkx.initialize_distributed_runtime_from_env() is False
    assert dkx._distributed_runtime_initialized is False

    monkeypatch.setenv("DKX_COORDINATOR_PORT", "not-an-int")
    assert dkx.initialize_distributed_runtime_from_env() is False
    assert dkx._distributed_runtime_initialized is False


def test_public_write_output_facade_routes_solve_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    def fake_run_from_namelist(namelist_path, **kwargs):
        calls.append({"namelist_path": Path(namelist_path), **kwargs})
        return SimpleNamespace(output_path=Path(kwargs["out_path"]))

    monkeypatch.setattr("dkx.run.run_from_namelist", fake_run_from_namelist)

    request = SolveInputs(
        input_path="input.namelist",
        output_path="sfincsOutput.h5",
        options={"solve_method": "auto"},
    )

    result = write_output(request)

    assert result == Path("sfincsOutput.h5")
    assert calls == [
        {
            "namelist_path": Path("input.namelist"),
            "out_path": Path("sfincsOutput.h5"),
            "solve_method": "auto",
        }
    ]


def test_public_write_output_requires_input_and_output_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run_from_namelist(*_args, **_kwargs):
        raise AssertionError("runner should not be called for invalid API requests")

    monkeypatch.setattr("dkx.run.run_from_namelist", fake_run_from_namelist)

    with pytest.raises(ValueError, match="input_path is required"):
        write_output(SolveInputs(output_path="sfincsOutput.h5"))

    with pytest.raises(ValueError, match="output_path is required"):
        write_output(SolveInputs(input_path="input.namelist"))


def test_public_write_output_explicit_kwargs_override_options(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    def fake_run_from_namelist(namelist_path, **kwargs):
        calls.append({"namelist_path": Path(namelist_path), **kwargs})
        return SimpleNamespace(output_path=Path(kwargs["out_path"]))

    monkeypatch.setattr("dkx.run.run_from_namelist", fake_run_from_namelist)

    request = SolveInputs(
        input_path="input.namelist",
        output_path="sfincsOutput.h5",
        options={"solve_method": "auto"},
    )
    result = write_output(request, solve_method="block_tridiagonal")

    assert result == Path("sfincsOutput.h5")
    assert calls[0]["solve_method"] == "block_tridiagonal"


def test_public_read_output_facade_routes_output_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Path] = []

    def fake_read_sfincs_output_file(path: Path) -> dict:
        calls.append(path)
        return {"ok": True}

    monkeypatch.setattr("dkx.io.read_sfincs_output_file", fake_read_sfincs_output_file)

    assert read_output("sfincsOutput.npz") == {"ok": True}
    assert calls == [Path("sfincsOutput.npz")]


def test_public_ambipolar_facade_routes_canonical_er_solver(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_ambipolar_brent routes to the canonical dkx.er slice."""
    calls: list[dict] = []

    def fake_find_ambipolar_er(input_path, **kwargs):
        calls.append({"input_path": input_path, **kwargs})
        return "ambipolar-result"

    monkeypatch.setattr("dkx.er.find_ambipolar_er", fake_find_ambipolar_er)

    request = SolveInputs(input_path="input.namelist", requires_autodiff=True, options={"emit": None})

    assert run_ambipolar_brent(
        request,
        er_min=-5.0,
        er_max=5.0,
        max_evaluations=7,
    ) == "ambipolar-result"
    assert calls == [
        {
            "input_path": Path("input.namelist"),
            "er_bracket": (-5.0, 5.0),
            "er_initial": 0.0,
            "max_iter": 7,
            "current_tol": 1.0e-10,
            "solve_method": "auto",
            "emit": None,
        }
    ]


def test_public_ambipolar_facade_forwards_bracket_and_tolerances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []

    def fake_find_ambipolar_er(input_path, **kwargs):
        calls.append({"input_path": input_path, **kwargs})
        return "ambipolar-result"

    monkeypatch.setattr("dkx.er.find_ambipolar_er", fake_find_ambipolar_er)

    request = SolveInputs(input_path="input.namelist", requires_autodiff=True, options={})
    assert run_ambipolar_brent(
        request,
        er_min=-1.0,
        er_max=1.0,
        er_initial=0.25,
        current_tolerance=1e-9,
        solve_method="gmres",
    ) == "ambipolar-result"

    assert calls[0]["er_bracket"] == (-1.0, 1.0)
    assert calls[0]["er_initial"] == 0.25
    assert calls[0]["current_tol"] == 1e-9
    assert calls[0]["solve_method"] == "gmres"
