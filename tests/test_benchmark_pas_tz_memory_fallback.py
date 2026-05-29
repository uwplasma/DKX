from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.benchmark_pas_tz_memory_fallback import (
    _override_namelist_text,
    _run_child,
    _variant_env,
    _variant_provenance,
    _variant_solve_method,
    build_plan,
    main,
    result_gates,
    summarize_results,
)


def test_variant_env_forces_bounded_pas_tz_memory_fallback() -> None:
    env = _variant_env("zeta", block=7, overlap=2, maxiter=5, restart=11)

    assert env["SFINCS_JAX_RHSMODE1_PRECONDITIONER"] == "pas_tz"
    assert env["SFINCS_JAX_RHSMODE1_PAS_TZ_MAX_BYTES"] == "1"
    assert env["SFINCS_JAX_RHSMODE1_PAS_TZ_MEMORY_FALLBACK"] == "zeta"
    assert env["SFINCS_JAX_RHSMODE1_PAS_TZ_SCHWARZ_BLOCK"] == "7"
    assert env["SFINCS_JAX_RHSMODE1_PAS_TZ_SCHWARZ_OVERLAP"] == "2"
    assert env["SFINCS_JAX_GMRES_MAXITER"] == "5"
    assert env["SFINCS_JAX_GMRES_RESTART"] == "11"


def test_variant_env_supports_collision_tzfft_correction() -> None:
    env = _variant_env("collision-tzfft-correction", block=7, overlap=2, maxiter=5, restart=11)

    assert env["SFINCS_JAX_RHSMODE1_PAS_TZ_MEMORY_FALLBACK"] == "collision"
    assert env["SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_CORRECTION"] == "tzfft"
    assert env["SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_STREAM_UPDATE"] == "1"


def test_variant_env_supports_structured_tzfft_correction() -> None:
    env = _variant_env("tzfft-structured", block=7, overlap=2, maxiter=5, restart=11)

    assert env["SFINCS_JAX_RHSMODE1_PAS_TZ_MEMORY_FALLBACK"] == "tzfft"
    assert env["SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_STRUCTURED_LEVELS"] == "xmg,collision"


def test_variant_env_supports_lgmres_suffix_without_changing_fallback() -> None:
    env = _variant_env("tzfft-lgmres", block=7, overlap=2, maxiter=5, restart=11)

    assert env["SFINCS_JAX_RHSMODE1_PAS_TZ_MEMORY_FALLBACK"] == "tzfft"
    assert _variant_solve_method("tzfft-lgmres", "incremental") == "lgmres"
    assert _variant_solve_method("tzfft", "incremental") == "incremental"


def test_variant_provenance_records_lgmres_opt_in_source() -> None:
    provenance = _variant_provenance("tzfft-lgmres", "incremental")

    assert provenance == {
        "variant": "tzfft-lgmres",
        "base_variant": "tzfft",
        "requested_solve_method": "incremental",
        "realized_solve_method": "lgmres",
        "solve_method_source": "variant_suffix",
        "lgmres_opt_in": True,
    }


def test_dry_run_writes_reproducible_plan(tmp_path: Path) -> None:
    out = tmp_path / "pas_tz_plan.json"

    rc = main(
        [
            "--dry-run",
            "--out",
            str(out),
            "--variants",
            "hybrid",
            "zeta",
            "--timeout-s",
            "12",
            "--maxiter",
            "3",
            "--restart",
            "4",
            "--solve-method",
            "lgmres",
            "--block",
            "5",
            "--overlap",
            "1",
            "--Ntheta",
            "31",
            "--Nzeta",
            "41",
            "--Nxi",
            "51",
            "--Nx",
            "7",
        ]
    )

    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["schema_version"] == 2
    assert payload["kind"] == "pas_tz_memory_fallback_benchmark"
    assert payload["plan"]["variants"] == ["hybrid", "zeta"]
    assert payload["plan"]["solve_method"] == "lgmres"
    assert payload["plan"]["variant_methods"] == [
        {
            "variant": "hybrid",
            "base_variant": "hybrid",
            "requested_solve_method": "lgmres",
            "realized_solve_method": "lgmres",
            "solve_method_source": "plan_default",
            "lgmres_opt_in": False,
        },
        {
            "variant": "zeta",
            "base_variant": "zeta",
            "requested_solve_method": "lgmres",
            "realized_solve_method": "lgmres",
            "solve_method_source": "plan_default",
            "lgmres_opt_in": False,
        },
    ]
    assert payload["plan"]["input_overrides"] == {
        "Ntheta": 31,
        "Nzeta": 41,
        "Nxi": 51,
        "Nx": 7,
    }
    assert payload["plan"]["gates"]["timeout_s"] == 12.0
    assert payload["plan"]["gates"]["stall_s"] == 12.0
    assert payload["plan"]["gates"]["max_default_runtime_s"] == 600.0
    assert payload["plan"]["gates"]["max_residual_norm"] == 1.0e-3
    assert "must not regress elapsed_s or max_rss_mb" in payload["plan"]["gates"]["promotion_policy"]
    assert payload["summary"]["result_count"] == 0
    assert payload["summary"]["all_gates_passed"] is False
    assert payload["summary"]["promotion_ready"] is False
    assert payload["results"] == []


def test_build_plan_records_solver_limits() -> None:
    class Args:
        input = Path("case/input.namelist")
        timeout_s = 9.0
        tol = 1.0e-7
        solve_method = "incremental"
        maxiter = 6
        restart = 8
        block = 4
        overlap = 2
        variants = ["theta"]
        Ntheta = 31
        Nzeta = None
        Nxi = 51
        Nx = None

    plan = build_plan(Args())

    assert plan["input_overrides"] == {"Ntheta": 31, "Nxi": 51}
    assert plan["timeout_s"] == 9.0
    assert plan["tol"] == 1.0e-7
    assert plan["solve_method"] == "incremental"
    assert plan["variant_methods"] == [
        {
            "variant": "theta",
            "base_variant": "theta",
            "requested_solve_method": "incremental",
            "realized_solve_method": "incremental",
            "solve_method_source": "plan_default",
            "lgmres_opt_in": False,
        }
    ]
    assert plan["maxiter"] == 6
    assert plan["restart"] == 8
    assert plan["block"] == 4
    assert plan["overlap"] == 2
    assert plan["variants"] == ["theta"]
    assert plan["gates"]["expected_backend"] == "auto"
    assert plan["gates"]["allow_solver_churn"] is False


def test_override_namelist_text_updates_grid_scalars_only() -> None:
    text = """&resolutionParameters
  Ntheta = 13  ! keep comment
  Nzeta = 23
  Nxi = 48
  Nx = 5
/
"""

    updated = _override_namelist_text(text, {"Ntheta": 31, "Nzeta": 41, "Nxi": 51, "Nx": 7})

    assert "  Ntheta = 31  ! keep comment" in updated
    assert "  Nzeta = 41" in updated
    assert "  Nxi = 51" in updated
    assert "  Nx = 7" in updated


def test_run_child_uses_temporary_input_overrides_and_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.input.namelist"
    source.write_text(
        """&resolutionParameters
  Ntheta = 13
  Nzeta = 23
  Nxi = 48
  Nx = 5
/
"""
    )
    out = tmp_path / "out.json"
    captured: dict[str, object] = {}

    def fake_run(
        cmd: list[str],
        *,
        env: dict[str, str],
        text: bool,
        capture_output: bool,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        input_path = Path(cmd[cmd.index("--input") + 1])
        captured["cmd"] = cmd
        captured["env"] = env
        captured["input_path"] = input_path
        captured["input_text"] = input_path.read_text()
        captured["text"] = text
        captured["capture_output"] = capture_output
        captured["timeout"] = timeout
        payload = {
            "status": "ok",
            "elapsed_s": 0.25,
            "residual_norm": 0.0,
            "phase_metadata": [{"name": "solve", "status": "ok", "elapsed_s": 0.25}],
            "solver_provenance": {
                "requested_solve_method": "incremental",
                "realized_solve_method": "incremental",
            },
        }
        stdout = "__SFINCS_JAX_PAS_TZ_RESULT__=" + json.dumps(payload) + "\n"
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    args = type(
        "Args",
        (),
        {
            "input": source,
            "out": out,
            "timeout_s": 9.0,
            "tol": 1.0e-7,
            "solve_method": "incremental",
            "maxiter": 6,
            "restart": 8,
            "block": 4,
            "overlap": 2,
            "variants": ["zeta"],
            "Ntheta": 31,
            "Nzeta": 41,
            "Nxi": 51,
            "Nx": 7,
        },
    )()

    row = _run_child(args, "zeta")

    child_input = captured["input_path"]
    assert isinstance(child_input, Path)
    assert child_input != source
    assert not child_input.exists()
    assert captured["input_text"] == """&resolutionParameters
  Ntheta = 31
  Nzeta = 41
  Nxi = 51
  Nx = 7
/
"""
    assert captured["cmd"][:2] == [sys.executable, str(Path("scripts/benchmark_pas_tz_memory_fallback.py").resolve())]
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["SFINCS_JAX_RHSMODE1_PAS_TZ_MEMORY_FALLBACK"] == "zeta"
    assert env["SFINCS_JAX_GMRES_MAXITER"] == "6"
    assert captured["text"] is True
    assert captured["capture_output"] is True
    assert captured["timeout"] == 9.0
    assert row["status"] == "ok"
    assert row["variant"] == "zeta"
    assert row["variant_provenance"] == {
        "variant": "zeta",
        "base_variant": "zeta",
        "requested_solve_method": "incremental",
        "realized_solve_method": "incremental",
        "solve_method_source": "plan_default",
        "lgmres_opt_in": False,
    }
    assert row["phase_metadata"] == [{"name": "solve", "status": "ok", "elapsed_s": 0.25}]
    assert row["solver_provenance"]["requested_solve_method"] == "incremental"
    assert row["tail_metadata"]["tail_limit_chars"] == 4000


def test_run_child_lgmres_variant_records_variant_method_without_changing_plan_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.input.namelist"
    source.write_text(
        """&resolutionParameters
  Ntheta = 13
  Nzeta = 23
  Nxi = 48
  Nx = 5
/
"""
    )
    captured: dict[str, object] = {}

    def fake_run(
        cmd: list[str],
        *,
        env: dict[str, str],
        text: bool,
        capture_output: bool,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        captured["cmd"] = cmd
        captured["env"] = env
        payload = {
            "status": "ok",
            "elapsed_s": 0.25,
            "residual_norm": 0.0,
            "solver_provenance": {
                "requested_solve_method": "lgmres",
                "realized_solve_method": "lgmres",
            },
        }
        stdout = "__SFINCS_JAX_PAS_TZ_RESULT__=" + json.dumps(payload) + "\n"
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    args = type(
        "Args",
        (),
        {
            "input": source,
            "timeout_s": 9.0,
            "tol": 1.0e-7,
            "solve_method": "incremental",
            "maxiter": 6,
            "restart": 8,
            "block": 4,
            "overlap": 2,
            "variants": ["tzfft-lgmres"],
            "Ntheta": None,
            "Nzeta": None,
            "Nxi": None,
            "Nx": None,
        },
    )()

    row = _run_child(args, "tzfft-lgmres")

    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert cmd[cmd.index("--solve-method") + 1] == "lgmres"
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["SFINCS_JAX_RHSMODE1_PAS_TZ_MEMORY_FALLBACK"] == "tzfft"
    assert row["variant_provenance"] == {
        "variant": "tzfft-lgmres",
        "base_variant": "tzfft",
        "requested_solve_method": "incremental",
        "realized_solve_method": "lgmres",
        "solve_method_source": "variant_suffix",
        "lgmres_opt_in": True,
    }


def test_run_child_gates_stalls_churn_backend_memory_and_residual(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.input.namelist"
    source.write_text("&resolutionParameters\n  Ntheta = 13\n/\n")

    def fake_run(
        cmd: list[str],
        *,
        env: dict[str, str],
        text: bool,
        capture_output: bool,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        del env, text, capture_output, timeout
        payload = {
            "status": "ok",
            "elapsed_s": 2.5,
            "max_rss_mb": 512.0,
            "residual_norm": 1.0e-2,
            "phase_metadata": [{"name": "solve", "status": "ok", "elapsed_s": 2.5}],
            "solver_provenance": {
                "requested_solve_method": "incremental",
                "realized_solve_method": "lgmres",
            },
            "runtime_metadata": {"jax_default_backend": "gpu"},
            "metadata": {"accepted_converged": True},
        }
        stdout = "__SFINCS_JAX_PAS_TZ_RESULT__=" + json.dumps(payload) + "\n"
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    args = type(
        "Args",
        (),
        {
            "input": source,
            "timeout_s": 9.0,
            "stall_s": 1.0,
            "tol": 1.0e-7,
            "solve_method": "incremental",
            "maxiter": 6,
            "restart": 8,
            "block": 4,
            "overlap": 2,
            "variants": ["tzfft"],
            "Ntheta": None,
            "Nzeta": None,
            "Nxi": None,
            "Nx": None,
            "max_rss_mb": 100.0,
            "max_residual_norm": 1.0e-3,
            "expected_backend": "cpu",
            "allow_solver_churn": False,
        },
    )()

    row = _run_child(args, "tzfft")

    assert row["gate"] == "fail"
    assert row["gates"]["stall"]["reason"] == "stall-threshold-exceeded"
    assert row["gates"]["solver_path"]["reason"] == "solver-path-mismatch"
    assert row["gates"]["backend"]["reason"] == "backend-mismatch"
    assert row["gates"]["memory"]["reason"] == "rss-threshold-exceeded"
    assert row["gates"]["residual"]["reason"] == "residual-threshold-exceeded"
    assert summarize_results([row])["all_gates_passed"] is False


def test_result_gates_reject_dense_guarded_correction_without_streaming_evidence() -> None:
    args = type(
        "Args",
        (),
        {
            "timeout_s": 9.0,
            "stall_s": 9.0,
            "max_rss_mb": 0.0,
            "max_residual_norm": 1.0e-3,
            "expected_backend": "auto",
            "allow_solver_churn": False,
            "solve_method": "incremental",
            "require_default_promotion_gate": False,
        },
    )()
    row = {
        "status": "ok",
        "elapsed_s": 1.0,
        "max_rss_mb": 800.0,
        "residual_norm": 1.0e-5,
        "phase_metadata": [{"name": "solve", "status": "ok", "elapsed_s": 1.0}],
        "guarded_pas_tz_seen": True,
        "solver_provenance": {
            "requested_solve_method": "incremental",
            "realized_solve_method": "incremental",
        },
        "metadata": {"accepted_converged": True},
    }

    gates = result_gates(args, row, "collision-tzfft-correction")

    assert gates["guarded_correction_memory"]["status"] == "fail"
    assert gates["guarded_correction_memory"]["reason"] == "dense-guarded-correction-disallowed"
    assert gates["guarded_correction_memory"]["diagnostics"]["full_update_materialized"] is False

    blocked_row = {
        **row,
        "metadata": {
            "accepted_converged": True,
            "pas_tz_guarded_correction_stream_requested": True,
            "pas_tz_guarded_correction_streamed": False,
            "pas_tz_guarded_correction_full_update_materialized": True,
            "pas_tz_guarded_correction_stream_blocker": (
                "production-pas-tz-minres-correction-requires-full-residual-direction"
            ),
        },
    }

    gates = result_gates(args, blocked_row, "collision-tzfft-correction")

    assert gates["guarded_correction_memory"]["status"] == "fail"
    assert gates["guarded_correction_memory"]["diagnostics"]["stream_requested"] is True
    assert gates["guarded_correction_memory"]["diagnostics"]["full_update_materialized"] is True
    assert gates["guarded_correction_memory"]["diagnostics"]["blockers"] == [
        "metadata:production-pas-tz-minres-correction-requires-full-residual-direction"
    ]

    streamed_row = {
        **row,
        "metadata": {
            "accepted_converged": True,
            "pas_tz_guarded_correction_streamed": True,
            "pas_tz_guarded_correction_full_update_materialized": False,
        },
    }

    gates = result_gates(args, streamed_row, "collision-tzfft-correction")

    assert gates["guarded_correction_memory"]["status"] == "pass"
    assert gates["guarded_correction_memory"]["reason"] == "streamed-guarded-correction-evidence-recorded"


def test_guarded_correction_evidence_can_live_in_nested_metadata_lists() -> None:
    args = type(
        "Args",
        (),
        {
            "timeout_s": 9.0,
            "stall_s": 9.0,
            "max_rss_mb": 0.0,
            "max_residual_norm": 1.0e-3,
            "expected_backend": "auto",
            "allow_solver_churn": False,
            "solve_method": "incremental",
            "require_default_promotion_gate": False,
        },
    )()
    row = {
        "status": "ok",
        "elapsed_s": 1.0,
        "max_rss_mb": 800.0,
        "residual_norm": 1.0e-5,
        "phase_metadata": [{"name": "solve", "status": "ok", "elapsed_s": 1.0}],
        "guarded_pas_tz_seen": True,
        "solver_provenance": {
            "requested_solve_method": "incremental",
            "realized_solve_method": "incremental",
        },
        "metadata": {
            "accepted_converged": True,
            "candidate_records": [
                {
                    "matrix_free_metadata": {
                        "stream_update_chunks": True,
                        "full_update_materialized": False,
                    }
                },
                {
                    "pas_tz_guarded_correction_streamed": True,
                    "pas_tz_guarded_correction_full_update_materialized": False,
                },
            ],
        },
    }

    gates = result_gates(args, row, "collision-tzfft-correction")

    assert gates["guarded_correction_memory"]["status"] == "pass"
    assert gates["guarded_correction_memory"]["diagnostics"]["streamed"] is True
    assert gates["guarded_correction_memory"]["diagnostics"]["full_update_materialized"] is False


def test_default_promotion_gate_requires_baseline_and_runtime_or_memory_win() -> None:
    args = type(
        "Args",
        (),
        {
            "timeout_s": 9.0,
            "stall_s": 9.0,
            "max_rss_mb": 0.0,
            "max_residual_norm": 1.0e-3,
            "expected_backend": "auto",
            "allow_solver_churn": False,
            "solve_method": "incremental",
            "require_default_promotion_gate": True,
            "baseline_elapsed_s": None,
            "baseline_rss_mb": 1000.0,
            "min_runtime_speedup": 1.05,
            "min_memory_reduction": 1.05,
        },
    )()
    row = {
        "status": "ok",
        "elapsed_s": 4.0,
        "max_rss_mb": 800.0,
        "residual_norm": 1.0e-5,
        "phase_metadata": [{"name": "solve", "status": "ok", "elapsed_s": 4.0}],
        "solver_provenance": {
            "requested_solve_method": "incremental",
            "realized_solve_method": "incremental",
        },
        "metadata": {"accepted_converged": True},
    }

    gates = result_gates(args, row, "tzfft")

    assert gates["default_promotion"]["status"] == "fail"
    assert gates["default_promotion"]["reason"] == "missing-promotion-baseline"

    args.baseline_elapsed_s = 10.0
    gates = result_gates(args, row, "tzfft")

    assert gates["default_promotion"]["status"] == "pass"
    assert gates["default_promotion"]["reason"] == "promotion-win-recorded"
    assert gates["default_promotion"]["runtime_win"] is True
    assert gates["default_promotion"]["memory_win"] is True


def test_default_promotion_gate_rejects_clean_candidate_without_material_win() -> None:
    args = type(
        "Args",
        (),
        {
            "timeout_s": 9.0,
            "stall_s": 9.0,
            "max_rss_mb": 0.0,
            "max_residual_norm": 1.0e-3,
            "expected_backend": "auto",
            "allow_solver_churn": False,
            "solve_method": "incremental",
            "require_default_promotion_gate": True,
            "baseline_elapsed_s": 10.0,
            "baseline_rss_mb": 1000.0,
            "min_runtime_speedup": 1.05,
            "min_memory_reduction": 1.05,
        },
    )()
    row = {
        "status": "ok",
        "elapsed_s": 9.8,
        "max_rss_mb": 990.0,
        "residual_norm": 1.0e-5,
        "phase_metadata": [{"name": "solve", "status": "ok", "elapsed_s": 9.8}],
        "solver_provenance": {
            "requested_solve_method": "incremental",
            "realized_solve_method": "incremental",
        },
        "metadata": {"accepted_converged": True},
    }

    gates = result_gates(args, row, "tzfft")

    assert gates["default_promotion"]["status"] == "fail"
    assert gates["default_promotion"]["reason"] == "no-runtime-or-memory-win"


def test_default_promotion_gate_rejects_runtime_or_memory_regressions() -> None:
    args = type(
        "Args",
        (),
        {
            "timeout_s": 9.0,
            "stall_s": 9.0,
            "max_rss_mb": 0.0,
            "max_residual_norm": 1.0e-3,
            "expected_backend": "auto",
            "allow_solver_churn": False,
            "solve_method": "incremental",
            "require_default_promotion_gate": True,
            "baseline_elapsed_s": 10.0,
            "baseline_rss_mb": 1000.0,
            "min_runtime_speedup": 1.05,
            "min_memory_reduction": 1.05,
        },
    )()
    base_row = {
        "status": "ok",
        "residual_norm": 1.0e-5,
        "solver_provenance": {
            "requested_solve_method": "incremental",
            "realized_solve_method": "incremental",
        },
        "metadata": {"accepted_converged": True},
    }

    faster_higher_memory = {
        **base_row,
        "elapsed_s": 8.0,
        "max_rss_mb": 1001.0,
        "phase_metadata": [{"name": "solve", "status": "ok", "elapsed_s": 8.0}],
    }
    lower_memory_slower = {
        **base_row,
        "elapsed_s": 10.1,
        "max_rss_mb": 800.0,
        "phase_metadata": [{"name": "solve", "status": "ok", "elapsed_s": 10.1}],
    }
    both_regressed = {
        **base_row,
        "elapsed_s": 10.1,
        "max_rss_mb": 1001.0,
        "phase_metadata": [{"name": "solve", "status": "ok", "elapsed_s": 10.1}],
    }

    gates = result_gates(args, faster_higher_memory, "tzfft")
    assert gates["default_promotion"]["status"] == "fail"
    assert gates["default_promotion"]["reason"] == "memory-regression"

    gates = result_gates(args, lower_memory_slower, "tzfft")
    assert gates["default_promotion"]["status"] == "fail"
    assert gates["default_promotion"]["reason"] == "runtime-regression"

    gates = result_gates(args, both_regressed, "tzfft")
    assert gates["default_promotion"]["status"] == "fail"
    assert gates["default_promotion"]["reason"] == "runtime-and-memory-regression"


def test_summarize_results_lists_only_promotion_win_rows() -> None:
    promoted = {
        "variant": "tzfft",
        "status": "ok",
        "gate": "pass",
        "gate_failures": [],
        "gates": {"default_promotion": {"status": "pass", "reason": "promotion-win-recorded"}},
    }
    not_requested = {
        "variant": "collision",
        "status": "ok",
        "gate": "pass",
        "gate_failures": [],
        "gates": {"default_promotion": {"status": "pass", "reason": "default-promotion-gate-not-requested"}},
    }

    summary = summarize_results([promoted, not_requested])

    assert summary["all_gates_passed"] is True
    assert summary["promotion_eligible_variants"] == ["tzfft"]
    assert summary["promotion_ready"] is True


def test_summarize_results_does_not_mark_empty_dry_run_as_passed() -> None:
    summary = summarize_results([])

    assert summary["result_count"] == 0
    assert summary["all_gates_passed"] is False
    assert summary["promotion_eligible_variants"] == []
    assert summary["promotion_ready"] is False


def test_main_rejects_long_timeout_without_explicit_opt_in(tmp_path: Path) -> None:
    out = tmp_path / "plan.json"

    with pytest.raises(SystemExit) as exc:
        main(["--dry-run", "--out", str(out), "--timeout-s", "601"])

    assert exc.value.code == 2


def test_main_allows_long_timeout_with_explicit_opt_in(tmp_path: Path) -> None:
    out = tmp_path / "plan.json"

    rc = main(["--dry-run", "--out", str(out), "--timeout-s", "601", "--allow-long-run"])

    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["plan"]["gates"]["timeout_s"] == 601.0
    assert payload["plan"]["gates"]["long_run_opt_in"] is True
