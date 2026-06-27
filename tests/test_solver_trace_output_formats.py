from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from sfincs_jax.io import write_sfincs_output_file
from sfincs_jax.outputs.rhsmode1 import (
    _add_rhsmode1_solver_diagnostics,
    _compact_json_metadata,
    _metadata_float,
    _metadata_int,
    _profile_memory_summary,
    _raise_for_nonconverged_rhsmode1_output,
    _rhs1_active_size_for_trace,
    _rhsmode1_result_residual_and_target,
    _should_fail_nonconverged_rhsmode1_output,
    _solver_metadata_dict,
    _solver_trace_memory_estimate,
    _write_nonconverged_rhsmode1_solver_trace_json,
)
from sfincs_jax.solvers.diagnostics import (
    SolverTrace,
    SolverTraceCandidate,
    read_solver_trace_h5,
)


def _trace() -> SolverTrace:
    return SolverTrace(
        backend="gpu",
        rhs_mode=1,
        selected_path="dense",
        solve_method="direct",
        preconditioner="none",
        geometry_scheme=5,
        collision_operator="full-fp",
        total_size=4096,
        active_size=3072,
        device_count=1,
        cold_jit=False,
        residual_norm=4.0e-12,
        residual_target=1.0e-9,
        elapsed_s=3.4,
        setup_s=0.9,
        solve_s=2.5,
        peak_rss_mb=875.0,
        active_rss_mb=275.0,
        device_peak_mb=350.0,
        compiled_temp_mb=80.0,
        estimated_dense_nbytes=4096 * 4096 * 8,
        estimated_csr_nbytes=4_000_000,
        estimated_gmres_basis_nbytes=4096 * 24 * 8,
        matvec_count=24,
        candidate_decisions=(
            SolverTraceCandidate(
                name="dense",
                accepted=True,
                residual_ratio=4.0e-3,
                memory_metric="device_peak_mb",
                device_peak_mb=350.0,
                candidate_setup_s=0.9,
                candidate_solve_s=2.5,
            ),
        ),
        metadata={"case": "solver-trace-output-format"},
    )


def _rhs1_trace_op(
    *,
    total_size: int = 4096,
    constraint_scheme: int = 2,
    include_phi1: bool = False,
    has_pas: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        rhs_mode=1,
        n_species=2,
        n_theta=3,
        n_zeta=4,
        total_size=total_size,
        active_size=123,
        constraint_scheme=constraint_scheme,
        include_phi1=include_phi1,
        phi1_size=7 if include_phi1 else 0,
        extra_size=5,
        collision_operator="full-fp",
        fblock=SimpleNamespace(
            collisionless=SimpleNamespace(n_xi_for_x=np.asarray([3, 2], dtype=np.int32)),
            pas=object() if has_pas else None,
        ),
    )


@pytest.mark.parametrize("suffix", [".h5", ".nc", ".npz"])
def test_write_sfincs_output_file_can_attach_solver_trace(tmp_path: Path, suffix: str) -> None:
    out = tmp_path / f"sfincsOutput{suffix}"
    trace = _trace()

    write_sfincs_output_file(
        path=out,
        data={"scalar": np.asarray(1.0), "vector": np.asarray([1.0, 2.0])},
        fortran_layout=False,
        solver_trace=trace,
    )

    if suffix == ".h5":
        with h5py.File(out, "r") as h5:
            loaded = read_solver_trace_h5(h5)
    elif suffix == ".nc":
        netcdf4 = pytest.importorskip("netCDF4")
        with netcdf4.Dataset(out, "r") as ds:
            loaded = SolverTrace.from_json(ds.getncattr("sfincs_jax_solver_trace_json"))
    else:
        with np.load(out, allow_pickle=False) as npz:
            loaded = SolverTrace.from_json(str(npz["sfincs_jax_solver_trace_json"].reshape(()).item()))

    assert loaded == trace


def test_rhsmode1_active_size_for_trace_respects_projection_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    op = _rhs1_trace_op(total_size=4096, constraint_scheme=2, include_phi1=False, has_pas=True)
    active_f = 2 * (3 + 2) * 3 * 4

    assert _rhs1_active_size_for_trace(op) == active_f

    monkeypatch.setenv("SFINCS_JAX_PAS_PROJECT_MIN", "bad")
    assert _rhs1_active_size_for_trace(op) == active_f

    small_op = _rhs1_trace_op(total_size=128, constraint_scheme=2, include_phi1=False, has_pas=True)
    assert _rhs1_active_size_for_trace(small_op) == active_f + small_op.extra_size

    phi1_op = _rhs1_trace_op(total_size=4096, constraint_scheme=2, include_phi1=True, has_pas=True)
    assert _rhs1_active_size_for_trace(phi1_op) == active_f + phi1_op.phi1_size + phi1_op.extra_size

    assert _rhs1_active_size_for_trace(SimpleNamespace(fblock=SimpleNamespace())) is None


def test_rhsmode1_residual_target_and_fail_gate_are_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    result = SimpleNamespace(residual_norm=np.asarray(2.0e-3), rhs=np.asarray([3.0, 4.0]))
    residual_norm, residual_target = _rhsmode1_result_residual_and_target(result, solver_tol=1.0e-4)

    assert residual_norm == pytest.approx(2.0e-3)
    assert residual_target == pytest.approx(5.0e-4)
    assert _should_fail_nonconverged_rhsmode1_output(
        active_total_size=20_000,
        residual_norm=residual_norm,
        residual_target=residual_target,
    )
    assert not _should_fail_nonconverged_rhsmode1_output(
        active_total_size=20_000,
        residual_norm=residual_norm,
        residual_target=residual_target,
        accepted_converged=True,
    )

    monkeypatch.setenv("SFINCS_JAX_ALLOW_NONCONVERGED_OUTPUT", "1")
    assert not _should_fail_nonconverged_rhsmode1_output(
        active_total_size=20_000,
        residual_norm=residual_norm,
        residual_target=residual_target,
    )
    monkeypatch.delenv("SFINCS_JAX_ALLOW_NONCONVERGED_OUTPUT")
    monkeypatch.setenv("SFINCS_JAX_NONCONVERGED_FAIL_MIN_SIZE", "bad")
    assert not _should_fail_nonconverged_rhsmode1_output(
        active_total_size=128,
        residual_norm=np.inf,
        residual_target=residual_target,
    )

    with pytest.raises(RuntimeError, match="Refusing to write nonconverged RHSMode=1"):
        _raise_for_nonconverged_rhsmode1_output(
            active_total_size=20_000,
            residual_norm=np.inf,
            residual_target=residual_target,
            solve_method="sparse_pc_gmres",
            accepted_converged=False,
            acceptance_criterion="true_residual",
        )

    bad_result = SimpleNamespace(residual_norm=object(), rhs=object())
    assert _rhsmode1_result_residual_and_target(bad_result, solver_tol=1.0e-4) == (None, None)


def test_rhsmode1_solver_metadata_helpers_and_output_fields() -> None:
    metadata = {
        "solve_method_requested": "auto",
        "solver_path": "sparse_pc",
        "solver_kind": "gmres",
        "preconditioner_kind": "xblock",
        "accepted_converged": False,
        "acceptance_criterion": "debug_override",
        "reported_residual_norm": 3.0,
        "iterations": 7,
        "matvecs": 11,
        "info_code": 0,
        "least_squares_converged": True,
        "setup_s": 0.5,
        "solve_s": 1.5,
        "sparse_pattern_nnz": 99,
        "sparse_pattern_avg_row_nnz": 4.5,
        "sparse_pc_xblock_assembled_host": True,
        "sparse_pc_fortran_reduced_direct_tail_structured_pc_metadata": {
            "kind": "native",
            "metadata": {
                "factor_kind": "lu",
                "permc_spec": "COLAMD",
                "permc_spec_requested": "auto",
                "permc_spec_candidates": ("COLAMD", "MMD_AT_PLUS_A"),
                "permc_failures": (),
            },
        },
        "sparse_pc_fortran_reduced_direct_tail_support_mode_preflight_requested": True,
        "sparse_pc_fortran_reduced_direct_tail_support_mode_preflight_selected": True,
        "sparse_pc_fortran_reduced_direct_tail_support_mode_preflight_metadata": {
            "accepted_nonbaseline": True,
            "selected_candidate": "support",
            "candidate_specs": [1, 2],
            "candidates": [{"name": "support", "residual": 0.2}],
            "baseline_residual_after": 2.0,
            "best_residual_after": 0.2,
            "rhs_norm": 10.0,
            "setup_s": 0.25,
        },
        "sparse_pc_fortran_reduced_direct_tail_structured_pc_max_mb": 32.0,
        "sparse_pc_fortran_reduced_direct_tail_structured_pc_max_mb_auto": True,
        "sparse_pc_direct_tail_true_coupled_coarse_requested": True,
        "sparse_pc_direct_tail_true_coupled_coarse_explicit_requested": True,
        "sparse_pc_direct_tail_true_coupled_coarse_auto_enabled": True,
        "sparse_pc_direct_tail_true_coupled_coarse_auto_selected": True,
        "sparse_pc_direct_tail_true_coupled_coarse_selected": True,
        "sparse_pc_direct_tail_true_coupled_coarse_auto_target_ratio": 0.5,
        "sparse_pc_direct_tail_true_coupled_coarse_auto_min_size": 128,
        "sparse_pc_direct_tail_true_coupled_coarse_residual_after": 0.1,
        "sparse_pc_direct_tail_true_coupled_coarse_metadata": {
            "base_residual_after": 2.0,
            "coarse_size": 3,
            "factor_nbytes_estimate": 456,
            "basis_names": ("constant", "current"),
        },
    }
    data: dict[str, object] = {}

    assert _solver_metadata_dict(SimpleNamespace(metadata=metadata)) == metadata
    assert _solver_metadata_dict(SimpleNamespace(metadata=("not", "a", "dict"))) == {}
    assert _metadata_int({"n": "4"}, "n") == 4
    assert _metadata_int({"n": -1}, "n") is None
    assert _metadata_float({"x": "1.25"}, "x") == pytest.approx(1.25)
    assert _metadata_float({"x": np.inf}, "x") is None
    assert _compact_json_metadata({"a": 1}) == '{"a": 1}'
    assert _compact_json_metadata({"payload": "x" * 100}, max_chars=32).endswith("<truncated>")
    assert _compact_json_metadata(object()) is not None

    _add_rhsmode1_solver_diagnostics(
        data,
        residual_norm=2.0,
        residual_target=1.0,
        solve_method="sparse_pc_gmres",
        solver_metadata=metadata,
    )

    assert data["linearSolverMethod"] == "sparse_pc_gmres"
    assert data["linearSolverRequestedMethod"] == "auto"
    assert data["linearSolverPath"] == "sparse_pc"
    assert int(np.asarray(data["linearSolverConverged"])) == -1
    assert int(np.asarray(data["linearSolverAccepted"])) == -1
    assert int(np.asarray(data["linearSolverIterations"])) == 7
    assert int(np.asarray(data["linearSolverMatvecs"])) == 11
    assert data["linearSolverSparsePCSelectedKind"] == "native"
    assert data["linearSolverDirectTailSupportModeSelectedCandidate"] == "support"
    assert int(np.asarray(data["linearSolverDirectTailTrueCoupledCoarseSize"])) == 3
    assert float(np.asarray(data["linearSolverResidualTargetRatio"])) == pytest.approx(2.0)


def test_solver_trace_memory_estimate_uses_sparse_metadata() -> None:
    estimate = _solver_trace_memory_estimate(
        total_size=100,
        active_size=80,
        solver_metadata={
            "sparse_pattern_nnz": 1_200,
            "gmres_restart": 30,
            "sparse_pc_factor_nbytes_estimate": 50_000,
        },
        device_count=2,
    )

    assert estimate is not None
    assert estimate["dense_operator_nbytes"] == 100 * 100 * 8
    assert estimate["csr_operator_nbytes"] == 1_200 * (8 + 4) + 101 * 4
    assert estimate["gmres_basis_nbytes"] == 100 * (30 + 1 + 4) * 8
    assert estimate["preconditioner_nbytes"] == 50_000
    assert estimate["csr_per_device_nbytes"] > 0


def test_solver_trace_memory_estimate_handles_unknown_sizes_and_metadata_fallbacks() -> None:
    assert (
        _solver_trace_memory_estimate(
            total_size=None,
            active_size=None,
            solver_metadata={},
            device_count=None,
        )
        is None
    )

    estimate = _solver_trace_memory_estimate(
        total_size=None,
        active_size=64,
        solver_metadata={
            "restart": 9,
            "csr_nnz": 128,
        },
        device_count=0,
    )

    assert estimate is not None
    assert estimate["dense_operator_nbytes"] == 64 * 64 * 8
    assert estimate["gmres_basis_nbytes"] == 64 * (9 + 1 + 4) * 8


def test_profile_memory_summary_prefers_active_and_device_peaks() -> None:
    profiler = SimpleNamespace(
        entries=[
            {"rss_mb": 100.0, "peak_rss_mb": 120.0, "dpeak_rss_mb": 10.0, "device_mb": 30.0},
            {"rss_mb": 110.0, "peak_rss_mb": 150.0, "dpeak_rss_mb": 42.0, "device_mb": 55.0},
        ]
    )

    active_rss_mb, device_peak_mb, peak_rss_mb = _profile_memory_summary(profiler)

    assert active_rss_mb == 42.0
    assert device_peak_mb == 55.0
    assert peak_rss_mb == 150.0


def test_profile_memory_summary_handles_empty_and_malformed_entries() -> None:
    assert _profile_memory_summary(None) == (None, None, None)
    assert _profile_memory_summary(SimpleNamespace(entries=[])) == (None, None, None)
    profiler = SimpleNamespace(entries=[{"drss_mb": "bad", "device_mb": "bad", "rss_mb": "bad"}])
    assert _profile_memory_summary(profiler) == (None, None, None)


def test_write_nonconverged_rhsmode1_solver_trace_json_records_failure_context(tmp_path: Path) -> None:
    trace_path = tmp_path / "solver_trace.json"
    input_path = tmp_path / "input.namelist"
    output_path = tmp_path / "sfincsOutput.h5"
    input_path.write_text("&general\n RHSMode = 1\n/\n", encoding="utf-8")
    op = _rhs1_trace_op(total_size=2048, constraint_scheme=1, include_phi1=False, has_pas=False)
    result = SimpleNamespace(
        op=op,
        rhs=np.asarray([3.0, 4.0]),
        metadata={
            "preconditioner_kind": "native",
            "gmres_restart": 12,
            "sparse_pattern_nnz": 100,
            "setup_s": 0.1,
            "solve_s": 0.2,
            "matvecs": 13,
            "accepted_converged": False,
            "acceptance_criterion": "true_residual",
        },
    )
    profiler = SimpleNamespace(
        entries=[
            {"rss_mb": 90.0, "peak_rss_mb": 95.0, "dpeak_rss_mb": 5.0, "device_mb": 8.0},
        ]
    )

    _write_nonconverged_rhsmode1_solver_trace_json(
        solver_trace_path=trace_path,
        input_namelist=input_path,
        output_path=output_path,
        output_format="h5",
        rhs_mode=1,
        geom_scheme_hint=5,
        compute_solution=True,
        compute_transport_matrix=False,
        differentiable=False,
        result=result,
        op_fallback=None,
        solver_tol=1.0e-4,
        solve_method="sparse_pc_gmres",
        residual_norm=1.0e-2,
        residual_target=None,
        active_total_size=512,
        run_t0=0.0,
        profiler=profiler,
    )

    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    assert payload["rhs_mode"] == 1
    assert payload["selected_path"] == "rhsmode1_solution"
    assert payload["preconditioner"] == "native"
    assert payload["residual_target"] == pytest.approx(5.0e-4)
    assert payload["metadata"]["failure_reason"] == "nonconverged_rhsmode1_output"
    assert payload["metadata"]["memory_estimate"]["gmres_basis_nbytes"] > 0
