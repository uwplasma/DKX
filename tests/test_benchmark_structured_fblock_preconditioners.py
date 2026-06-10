from __future__ import annotations

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from scripts.benchmark_structured_fblock_preconditioners import (
    _run_child,
    _summarize,
    build_plan,
    main,
    run_child_payload,
)


def test_structured_fblock_benchmark_dry_run_writes_plan(tmp_path: Path) -> None:
    out = tmp_path / "structured_plan.json"

    rc = main(
        [
            "--dry-run",
            "--out",
            str(out),
            "--cases",
            "fp_phi1_tiny",
            "--preconditioners",
            "fp_radial",
            "fp_lowmode_schur",
            "--solve-cases",
            "fp_phi1_tiny",
            "--timeout-s",
            "12",
            "--restart",
            "7",
            "--maxiter",
            "9",
        ]
    )

    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["kind"] == "structured_fblock_preconditioner_benchmark"
    assert payload["plan"]["cases"] == ["fp_phi1_tiny"]
    assert payload["plan"]["preconditioners"] == ["fp_radial", "fp_lowmode_schur"]
    assert payload["plan"]["solve_cases"] == ["fp_phi1_tiny"]
    assert payload["plan"]["restart"] == 7
    assert payload["plan"]["maxiter"] == 9
    assert payload["plan"]["warm_repeats"] == 2
    assert payload["plan"]["solve_repeats"] == 1
    assert payload["plan"]["max_solve_residual"] == 1.0e-8
    assert payload["plan"]["min_dke_improvement"] == 1.05
    assert payload["plan"]["max_warm_runtime_ratio"] == 1.0
    assert payload["plan"]["max_rss_ratio"] == 1.25
    assert payload["summary"]["result_count"] == 0
    assert payload["summary"]["promotion_ready_cases"] == []
    assert payload["results"] == []


def test_structured_fblock_benchmark_build_plan_records_bounds() -> None:
    args = SimpleNamespace(
        cases=["quick_fp"],
        preconditioners=["fp_radial"],
        solve_cases=[],
        timeout_s=3.0,
        identity_shift=0.25,
        tol=1.0e-7,
        restart=8,
        maxiter=10,
        solve_method="incremental",
        warm_repeats=3,
        solve_repeats=2,
        max_solve_residual=1.0e-9,
        min_dke_improvement=1.25,
        max_warm_runtime_ratio=0.95,
        max_rss_ratio=1.1,
    )

    assert build_plan(args) == {
        "cases": ["quick_fp"],
        "preconditioners": ["fp_radial"],
        "solve_cases": [],
        "timeout_s": 3.0,
        "identity_shift": 0.25,
        "tol": 1.0e-7,
        "restart": 8,
        "maxiter": 10,
        "solve_method": "incremental",
        "warm_repeats": 3,
        "solve_repeats": 2,
        "max_solve_residual": 1.0e-9,
        "min_dke_improvement": 1.25,
        "max_warm_runtime_ratio": 0.95,
        "max_rss_ratio": 1.1,
    }


def test_structured_fblock_benchmark_child_lowmode_reports_matrix_free_metadata() -> None:
    args = SimpleNamespace(
        case="fp_phi1_tiny",
        preconditioner="fp_lowmode_schur",
        run_solve=False,
        identity_shift=0.5,
        tol=1.0e-8,
        restart=20,
        maxiter=40,
        solve_method="incremental",
        warm_repeats=1,
        solve_repeats=1,
    )

    row = run_child_payload(args)

    assert row["ok"] is True
    assert row["case"] == "fp_phi1_tiny"
    assert row["preconditioner"] == "fp_lowmode_schur"
    assert row["metadata"]["coarse"]["kind"] == "matrix_free_galerkin_residual_correction"
    assert row["metadata"]["coarse"]["basis_storage_nbytes"] == 0
    assert row["metadata"]["coarse"]["solver_kind"] == "precomputed_dense_inverse"
    assert row["dke_residual_ratio"] < 6.0e-2
    assert len(row["warm_preconditioner_apply_s"]) == 1
    assert row["warm_preconditioner_apply_s_min"] >= 0.0
    assert row["warm_dke_residual_ratio"] < 6.0e-2


def test_structured_fblock_benchmark_child_moment_reports_compact_metadata() -> None:
    args = SimpleNamespace(
        case="fp_phi1_tiny",
        preconditioner="fp_moment_schur",
        run_solve=False,
        identity_shift=0.5,
        tol=1.0e-8,
        restart=20,
        maxiter=40,
        solve_method="incremental",
        warm_repeats=1,
        solve_repeats=1,
    )

    row = run_child_payload(args)

    assert row["ok"] is True
    assert row["preconditioner"] == "fp_moment_schur"
    assert row["metadata"]["coarse"]["kind"] == "matrix_free_galerkin_residual_correction"
    assert row["metadata"]["coarse"]["basis_storage_nbytes"] == 0
    assert row["metadata"]["coarse"]["n_coarse"] < 80
    assert row["metadata"]["coarse_moment_selection"]["x_moments_retained"] == 2
    assert row["metadata"]["coarse_moment_selection"]["xi_moments_retained"] == 2


def test_structured_fblock_benchmark_child_coupled_reports_full_residual_metadata() -> None:
    args = SimpleNamespace(
        case="fp_phi1_tiny",
        preconditioner="fp_coupled_moment_schur",
        run_solve=False,
        identity_shift=0.5,
        tol=1.0e-8,
        restart=20,
        maxiter=40,
        solve_method="incremental",
        warm_repeats=1,
        solve_repeats=1,
    )

    row = run_child_payload(args)

    assert row["ok"] is True
    assert row["preconditioner"] == "fp_coupled_moment_schur"
    assert row["metadata"]["coarse"]["kind"] == "matrix_free_galerkin_residual_correction"
    assert row["metadata"]["coarse_coupled_selection"]["tail_policy"] == "all_tail"
    assert row["full_residual_ratio"] < 1.0
    assert row["dke_residual_ratio"] > 0.1


def test_structured_fblock_benchmark_child_tail_coupled_reports_tail_metadata() -> None:
    args = SimpleNamespace(
        case="fp_phi1_tiny",
        preconditioner="fp_tail_coupled_schur",
        run_solve=False,
        identity_shift=0.5,
        tol=1.0e-8,
        restart=20,
        maxiter=40,
        solve_method="incremental",
        warm_repeats=1,
        solve_repeats=1,
    )

    row = run_child_payload(args)

    assert row["ok"] is True
    assert row["preconditioner"] == "fp_tail_coupled_schur"
    assert row["metadata"]["coarse"]["kind"] == "matrix_free_least_squares_residual_correction"
    assert row["metadata"]["coarse"]["solver_kind"] == "precomputed_normal_inverse"
    assert row["metadata"]["coarse_tail_selection"]["tail_policy"] == "all_tail"
    assert row["metadata"]["coarse_tail_selection"]["tail_count"] > 0
    assert row["full_residual_ratio"] > 0.0
    assert row["dke_residual_ratio"] > 0.0


def test_structured_fblock_benchmark_child_timeout_becomes_row(monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=["sfincs_jax"], timeout=4.0, output="partial", stderr="trace")

    monkeypatch.setattr("scripts.benchmark_structured_fblock_preconditioners.subprocess.run", fake_run)
    args = SimpleNamespace(
        timeout_s=4.0,
        identity_shift=0.5,
        tol=1.0e-8,
        restart=20,
        maxiter=40,
        solve_method="incremental",
        warm_repeats=1,
        solve_repeats=1,
    )

    row = _run_child(args, case="fp_phi1_tiny", preconditioner="fp_radial", run_solve=False)

    assert row["ok"] is False
    assert row["error"] == "timeout"
    assert row["timeout_s"] == 4.0
    assert row["stdout_tail"] == "partial"
    assert row["stderr_tail"] == "trace"


def test_structured_fblock_benchmark_summary_selects_best_one_step() -> None:
    summary = _summarize(
        [
            {"case": "a", "preconditioner": "slow", "dke_residual_ratio": 2.0, "ok": True},
            {"case": "a", "preconditioner": "fast", "dke_residual_ratio": 1.0, "ok": True},
            {"case": "b", "preconditioner": "bad", "dke_residual_ratio": 0.0, "ok": False},
        ]
    )

    assert summary["result_count"] == 3
    assert summary["ok_count"] == 2
    assert summary["all_ok"] is False
    assert summary["best_one_step_by_case"] == {"a": "fast"}
    assert summary["promotion_ready_cases"] == []


def test_structured_fblock_benchmark_summary_records_promotion_gate() -> None:
    summary = _summarize(
        [
            {
                "case": "fp",
                "preconditioner": "fp_radial",
                "dke_residual_ratio": 0.1,
                "solve": {"residual_norm": 1.0e-12},
                "metadata": {},
                "warm_preconditioner_apply_s_median": 2.0,
                "max_rss_mb": 100.0,
                "ok": True,
            },
            {
                "case": "fp",
                "preconditioner": "fp_lowmode_schur",
                "dke_residual_ratio": 0.05,
                "solve": {"residual_norm": 2.0e-12},
                "metadata": {"coarse": {"basis_storage_nbytes": 0}},
                "warm_preconditioner_apply_s_median": 1.5,
                "max_rss_mb": 110.0,
                "ok": True,
            },
            {
                "case": "fp",
                "preconditioner": "fp_moment_schur",
                "dke_residual_ratio": 0.08,
                "metadata": {"coarse": {"basis_storage_nbytes": 0}},
                "warm_preconditioner_apply_s_median": 0.75,
                "max_rss_mb": 90.0,
                "ok": True,
            },
            {
                "case": "fp",
                "preconditioner": "fp_coupled_moment_schur",
                "dke_residual_ratio": 0.2,
                "full_residual_ratio": 0.01,
                "metadata": {"coarse": {"basis_storage_nbytes": 0}},
                "warm_preconditioner_apply_s_median": 0.7,
                "max_rss_mb": 95.0,
                "ok": True,
            },
        ],
        min_dke_improvement=1.5,
        max_solve_residual=1.0e-8,
        max_warm_runtime_ratio=1.0,
        max_rss_ratio=1.25,
    )

    comparison = summary["promotion_comparisons"]["fp"]
    assert comparison["dke_improvement"] == 2.0
    assert comparison["solve_residual_gate_ok"] is True
    assert comparison["matrix_free_storage_gate_ok"] is True
    assert comparison["warm_runtime_gate_ok"] is True
    assert comparison["rss_gate_ok"] is True
    assert comparison["promotion_ready"] is True
    moment_comparison = summary["promotion_comparisons"]["fp:fp_moment_schur"]
    assert moment_comparison["candidate"] == "fp_moment_schur"
    assert moment_comparison["dke_improvement"] == 1.25
    assert moment_comparison["promotion_ready"] is False
    coupled_comparison = summary["promotion_comparisons"]["fp:fp_coupled_moment_schur"]
    assert coupled_comparison["candidate"] == "fp_coupled_moment_schur"
    assert coupled_comparison["full_residual_improvement"] is None
    assert coupled_comparison["promotion_ready"] is False
    assert summary["promotion_ready_cases"] == ["fp"]
