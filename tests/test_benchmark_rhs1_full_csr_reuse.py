from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from scripts.benchmark_rhs1_full_csr_reuse import (
    _direct_factor_storage_metadata,
    _history_summary,
    _preconditioner_storage_metadata,
    _promotion_gate_diagnostics,
    _rate_per_second,
    _residual_reduction,
    run_benchmark,
)


def test_rhs1_full_csr_reuse_benchmark_dry_run_contract(tmp_path: Path) -> None:
    args = argparse.Namespace(
        input=Path("tests/ref/quick_2species_FPCollisions_noEr.input.namelist"),
        out=tmp_path / "bench.json",
        identity_shift=0.5,
        repeats=2,
        max_csr_mb=16.0,
        solve=False,
        solve_tol=1.0e-8,
        solve_atol=1.0e-10,
        solve_restart=80,
        solve_maxiter=20,
        solve_method="gmres",
        active_dof=False,
        min_residual_reduction=1.0e-3,
        preconditioner="auto",
        preconditioner_max_schur_size=2048,
        preconditioner_max_block_inverse_mb=64.0,
        max_preconditioner_setup_s=60.0,
        max_preconditioner_storage_mb=64.0,
        dry_run=True,
        json=False,
    )

    payload = run_benchmark(args)

    assert payload["kind"] == "rhs1_full_csr_reuse_benchmark"
    assert payload["status"] == "planned"
    assert payload["solve"] is False
    assert payload["solve_method"] == "gmres"
    assert payload["active_dof"] is False
    assert payload["preconditioner"] == "auto"
    assert payload["preconditioner_max_block_inverse_mb"] == 64.0
    assert payload["dry_run"] is True
    assert payload["min_residual_reduction"] == 1.0e-3
    assert payload["residual_reduction"] is None
    assert payload["residual_reduction_per_s"] is None
    assert payload["initial_true_residual_norm"] is None
    assert payload["final_true_residual_norm"] is None
    assert payload["preconditioned_residual_history"] == {
        "values": [],
        "count": 0,
        "initial": None,
        "final": None,
        "reduction": None,
    }
    assert payload["initial_preconditioned_residual_norm"] is None
    assert payload["final_preconditioned_residual_norm"] is None
    assert payload["preconditioner_storage"]["setup"]["storage_nbytes_estimate"] == 0
    assert payload["promotion_gate"] is False
    assert "solve_not_selected" in payload["promotion_gate_diagnostics"]["reasons"]


def test_rhs1_full_csr_reuse_residual_rate_and_history_helpers() -> None:
    assert _residual_reduction(10.0, 2.5) == pytest.approx(0.75)
    assert _rate_per_second(0.75, 3.0) == pytest.approx(0.25)

    history = _history_summary((1.0, 0.5, 0.125))

    assert history["values"] == [1.0, 0.5, 0.125]
    assert history["count"] == 3
    assert history["initial"] == 1.0
    assert history["final"] == 0.125
    assert history["reduction"] == pytest.approx(0.875)


def test_rhs1_full_csr_reuse_preconditioner_storage_contract() -> None:
    storage = _preconditioner_storage_metadata(
        {
            "selected": True,
            "kind": "diagonal_schur",
            "reason": "complete",
            "setup_s": 0.5,
            "metadata": {
                "diagonal_size": 4,
                "kinetic_size": 4,
                "tail_size": 2,
                "schur_nbytes": 32,
                "factor_nbytes_actual": 96,
                "work_vector_nbytes": 64,
            },
        },
        total_size=6,
        max_setup_s=1.0,
        max_storage_nbytes=1024,
    )

    assert storage["selected"] is True
    assert storage["kind"] == "diagonal_schur"
    assert storage["setup"]["s"] == 0.5
    assert storage["setup"]["threshold_ok"] is True
    assert storage["setup"]["storage_threshold_ok"] is True
    assert storage["setup"]["storage_components_nbytes"]["inverse_diagonal_nbytes"] == 32
    assert storage["setup"]["storage_components_nbytes"]["schur_nbytes"] == 32
    assert storage["setup"]["storage_components_nbytes"]["factor_nbytes_actual"] == 96
    assert storage["setup"]["storage_nbytes_estimate"] == 168
    assert storage["apply"]["storage_components_nbytes"]["input_vector_nbytes"] == 48
    assert storage["apply"]["storage_components_nbytes"]["output_vector_nbytes"] == 48
    assert storage["apply"]["storage_nbytes_estimate"] == 208


def test_rhs1_full_csr_reuse_direct_factor_storage_contract() -> None:
    storage = _direct_factor_storage_metadata(
        {
            "factor_kind": "splu",
            "factor_s": 0.25,
            "factor_nnz": 12,
            "factor_nbytes_actual": 96,
            "permc_spec": "COLAMD",
        },
        max_setup_s=1.0,
        max_storage_nbytes=128,
    )

    assert storage["selected"] is True
    assert storage["kind"] == "splu"
    assert storage["reason"] == "direct_solve"
    assert storage["setup"]["s"] == 0.25
    assert storage["setup"]["threshold_ok"] is True
    assert storage["setup"]["storage_threshold_ok"] is True
    assert storage["setup"]["storage_components_nbytes"]["splu_factor_nbytes_actual"] == 96
    assert storage["metadata"]["factor_nnz"] == 12


def test_rhs1_full_csr_reuse_promotion_gate_requires_residual_and_resource_gates() -> None:
    storage = _preconditioner_storage_metadata(
        {"selected": True, "kind": "jacobi", "reason": "complete", "setup_s": 0.1, "metadata": {}},
        total_size=2,
        max_setup_s=1.0,
        max_storage_nbytes=16,
    )

    gate = _promotion_gate_diagnostics(
        solve_selected=True,
        solve_converged=False,
        residual_reduction=0.02,
        min_residual_reduction=0.01,
        preconditioner_storage=storage,
    )

    assert gate["passed"] is True
    assert gate["material_residual_improvement"] is True

    storage["setup"]["storage_threshold_ok"] = False
    rejected = _promotion_gate_diagnostics(
        solve_selected=True,
        solve_converged=True,
        residual_reduction=0.0,
        min_residual_reduction=0.01,
        preconditioner_storage=storage,
    )

    assert rejected["passed"] is False
    assert "preconditioner_storage_threshold_exceeded" in rejected["reasons"]
