from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "reference_solver_path_artifacts"


def _load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def _pas_variant_probe_cases() -> dict[str, dict[str, object]]:
    return _load_json(FIXTURES / "pas_offender_variant_probe_2026-05-01.json")["cases"]


def _fp3d_sparse_pc_probe() -> dict[str, object]:
    return _load_json(FIXTURES / "fp3d_sparse_pc_probe_2026-05-02.json")


def _tokamak_er_dense_probe() -> dict[str, object]:
    return _load_json(FIXTURES / "production_tokamak_er_dense_probe_2026-05-03.json")


def _geometry4_large_pas_closeout() -> dict[str, object]:
    return _load_json(FIXTURES / "geometry4_large_pas_closeout_2026-05-09.json")


def _pas_tz_memory_fallback_smoke() -> dict[str, object]:
    return _load_json(FIXTURES / "pas_tz_memory_fallback_geometry4_smoke_2026-05-10.json")


def test_cpu_solver_path_audit_stays_parity_clean() -> None:
    report = _load_json(FIXTURES / "cpu_audit_snapshot.json")["strict_report"]

    assert report["len"] == 39
    assert set(report["statuses"]) == {"parity_ok"}
    assert report["strict_n_mismatch_common_sum"] == 0
    assert report["print_missing_signals_total"] == 0


def test_cpu_full_fp_cliff_cases_stay_on_dense_audit_path() -> None:
    audit = _load_json(FIXTURES / "cpu_audit_snapshot.json")["full_fp_cases"]
    full_fp_cases = [
        "tokamak_1species_FPCollisions_noEr",
        "tokamak_1species_FPCollisions_withEr_DKESTrajectories",
        "tokamak_1species_FPCollisions_withEr_fullTrajectories",
        "quick_2species_FPCollisions_noEr",
        "inductiveE_noEr",
    ]

    for case in full_fp_cases:
        row = audit[case]
        assert row["dense_auto"] is True
        assert row["default_krylov"] is False
        assert row["dense_fallback"] is False
        assert row["sparse_fallback"] is False
        assert row["profile_peak_rss_mb"] < 650.0
        assert row["rhs1_solve_s"] < 1.0


def test_gpu_neighboring_full_fp_artifacts_reject_slow_krylov_cliff() -> None:
    audits = _load_json(FIXTURES / "gpu_neighboring_full_fp_snapshot.json")
    for rows in audits.values():
        default = rows["default"]
        forced_krylov = rows["forced_krylov"]

        assert default["status"] == "ok"
        assert forced_krylov["status"] == "ok"
        assert default["vs_fortran_count"] == 0
        assert forced_krylov["vs_fortran_count"] == 0
        assert default["used_dense_auto"] is True
        assert default["used_sparse_fallback"] is False
        assert float(default["elapsed_s"]) < 0.5 * float(forced_krylov["elapsed_s"])
        assert float(default["ru_maxrss_mb"]) < float(forced_krylov["ru_maxrss_mb"])


def test_cpu_pas_heavy_audit_cases_stay_on_structured_paths() -> None:
    audit = _load_json(FIXTURES / "cpu_audit_snapshot.json")["pas_heavy_cases"]
    expected_preconditioners = {
        "HSX_PASCollisions_DKESTrajectories": "pas_tz",
        "HSX_PASCollisions_fullTrajectories": "pas_tz",
        "geometryScheme4_1species_PAS_withEr_DKESTrajectories": "pas_tz",
        "geometryScheme4_2species_PAS_noEr": "pas_tz",
        "sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_DKESTrajectories": "pas_tz",
        "sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories": "pas_tz",
        "tokamak_1species_PASCollisions_noEr": "pas_tokamak_theta",
        "tokamak_1species_PASCollisions_noEr_Nx1": "xblock_tz",
        "tokamak_1species_PASCollisions_withEr_fullTrajectories": "xblock_tz",
        "tokamak_2species_PASCollisions_noEr": "schur",
        "tokamak_2species_PASCollisions_withEr_fullTrajectories": "schur",
    }

    for case, expected in expected_preconditioners.items():
        row = audit[case]
        assert row["last_preconditioner"] == expected
        assert row["dense_auto"] is False
        assert row["dense_fallback"] is False
        assert row["sparse_fallback"] is False
        assert row["rhs1_solve_s"] < 2.0
        assert row["profile_peak_rss_mb"] < 2100.0


def test_cpu_pas_heavy_audit_cases_remain_strict_parity_clean() -> None:
    report = _load_json(FIXTURES / "cpu_audit_snapshot.json")["strict_report"]["pas_cases"]
    pas_cases = list(report)

    assert len(pas_cases) >= 13
    for case in pas_cases:
        row = report[case]
        assert row["status"] == "parity_ok"
        assert row["strict_n_mismatch_common"] == 0
        assert row["print_missing_signals"] == []


def test_trace_backed_full_cpu_suite_report_is_strict_clean() -> None:
    report = _load_json(FIXTURES / "trace_artifact_summary.json")["cpu_full_trace"]

    assert report["report_len"] == 39
    assert set(report["statuses"]) == {"parity_ok"}
    assert report["strict_n_mismatch_common_sum"] == 0
    assert report["print_missing_signals_total"] == 0
    assert report["trace_count"] == 39


def test_trace_backed_gpu_probe_report_is_strict_clean() -> None:
    report = _load_json(FIXTURES / "trace_artifact_summary.json")["gpu_probe"]

    assert report["report_len"] == 7
    assert set(report["statuses"]) == {"parity_ok"}
    assert report["strict_n_mismatch_common_sum"] == 0
    assert report["print_missing_signals_total"] == 0
    assert report["trace_count"] == 7
    assert set(report["backends"]) == {"gpu"}
    assert set(report["selected_paths"]) == {"rhsmode1_solution"}


def test_trace_backed_full_gpu_suite_report_is_strict_clean() -> None:
    report = _load_json(FIXTURES / "trace_artifact_summary.json")["gpu_full"]

    assert report["report_len"] == 39
    assert set(report["statuses"]) == {"parity_ok"}
    assert report["strict_n_mismatch_common_sum"] == 0
    assert report["print_missing_signals_total"] == 0
    assert report["trace_count"] == 39
    assert set(report["backends"]) == {"gpu"}
    assert {"rhsmode1_solution", "transport_matrix"}.issuperset(
        set(report["selected_paths"])
    )


def test_pas_offender_variant_probe_keeps_pas_tz_default_on_pareto_front() -> None:
    cases = _pas_variant_probe_cases()

    for case_name in (
        "geometryScheme4_2species_PAS_noEr",
        "sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories",
    ):
        rows = {str(row["variant"]): row for row in cases[case_name]["variants"]}
        default = rows["default"]

        assert default["status"] == "ok"
        assert default["rhs1_preconditioner"] == "pas_tz"
        assert default["vs_fortran_count"] == 0

        default_elapsed = float(default["elapsed_s"])
        default_rss = float(default["ru_maxrss_mb"])
        for variant, row in rows.items():
            if variant == "default" or row["status"] != "ok" or row["vs_fortran_count"] != 0:
                continue
            elapsed = float(row["elapsed_s"])
            rss = float(row["ru_maxrss_mb"])
            assert not (elapsed < default_elapsed and rss <= default_rss), variant
            assert not (elapsed <= default_elapsed and rss < default_rss), variant


def test_pas_offender_variant_probe_rejects_geometry11_xblock_tz_mismatch() -> None:
    cases = _pas_variant_probe_cases()
    rows = {
        str(row["variant"]): row
        for row in cases["sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories"][
            "variants"
        ]
    }

    xblock = rows["xblock_tz"]
    assert xblock["status"] == "ok"
    assert xblock["rhs1_preconditioner"] == "xblock_tz"
    assert xblock["vs_fortran_count"] == 6
    assert "jHat" in xblock["vs_default_sample"]
    assert "flow" in xblock["vs_default_sample"]


def test_pas_offender_restart_variants_are_memory_knobs_not_defaults() -> None:
    cases = _pas_variant_probe_cases()

    geometry4 = {
        str(row["variant"]): row
        for row in cases["geometryScheme4_2species_PAS_noEr_restart_sweep"]["variants"]
    }
    geom4_default = geometry4["default"]
    for variant in ("restart20", "restart30", "restart40", "restart120"):
        row = geometry4[variant]
        assert row["status"] == "ok"
        assert row["vs_fortran_count"] == 0
        assert float(row["ru_maxrss_mb"]) < float(geom4_default["ru_maxrss_mb"])
        assert float(row["elapsed_s"]) > 2.0 * float(geom4_default["elapsed_s"])

    geometry11 = {
        str(row["variant"]): row
        for row in cases["sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories"][
            "variants"
        ]
    }
    geom11_default = geometry11["default"]
    restart40 = geometry11["restart40"]
    assert restart40["status"] == "ok"
    assert restart40["vs_fortran_count"] == 0
    assert float(restart40["ru_maxrss_mb"]) < float(geom11_default["ru_maxrss_mb"])
    assert float(restart40["elapsed_s"]) > 2.0 * float(geom11_default["elapsed_s"])


def test_pas_offender_timed_out_variants_are_not_release_candidates() -> None:
    cases = _pas_variant_probe_cases()

    for case in cases.values():
        timed_out = [row for row in case["variants"] if row["status"] == "timeout"]
        for row in timed_out:
            assert row["elapsed_s"] is None
            assert row["ru_maxrss_mb"] is None
            assert row["rhs1_preconditioner"] is None


def test_geometry4_large_pas_closeout_does_not_promote_timeout_candidates() -> None:
    closeout = _geometry4_large_pas_closeout()

    assert closeout["decision"] == "no_default_promotion"
    assert closeout["system_size"]["active_size"] == 744610
    assert closeout["system_size"]["total_size"] == 1275010
    assert closeout["bounded_gate_s"] == 300

    cpu_candidates = closeout["cpu_candidates"]
    gpu_candidates = closeout["gpu_candidates"]
    assert {row["status"] for row in cpu_candidates} == {"timeout"}
    assert {row["status"] for row in gpu_candidates} == {"timeout"}
    assert {row["preconditioner"] for row in cpu_candidates if "preconditioner" in row}.issuperset(
        {"schur", "sparse_pc_gmres", "pas_tz", "xmg", "pas_hybrid"}
    )
    assert all(float(row["wall_s"]) >= 300.0 for row in [*cpu_candidates, *gpu_candidates])
    assert "structured/chunked geometry-aware PAS preconditioner" in closeout["required_future_algorithm"]


def test_pas_tz_memory_fallback_smoke_keeps_structured_fallback_opt_in() -> None:
    smoke = _pas_tz_memory_fallback_smoke()

    assert smoke["kind"] == "pas_tz_memory_fallback_benchmark"
    assert smoke["plan"]["input"] == "examples/sfincs_examples/geometryScheme4_2species_PAS_noEr/input.namelist"
    assert smoke["plan"]["timeout_s"] == 15.0
    assert smoke["plan"]["variants"] == ["collision", "hybrid", "zeta", "theta"]

    rows = {row["variant"]: row for row in smoke["results"]}
    assert set(rows) == {"collision", "hybrid", "zeta", "theta"}
    for variant in ("collision", "hybrid", "zeta", "theta"):
        row = rows[variant]
        assert row["status"] == "ok"
        assert float(row["elapsed_s"]) < 5.0
        assert float(row["residual_norm"]) > 1.0e3
        messages = "\n".join(row["messages_tail"])
        assert f"guarded out (axis={variant})" in messages
        assert "skipping strong preconditioner after guarded PAS-TZ fallback" in messages
    assert float(rows["collision"]["residual_norm"]) < float(rows["hybrid"]["residual_norm"])
    assert float(rows["collision"]["max_rss_mb"]) <= float(rows["hybrid"]["max_rss_mb"])


def test_fp3d_sparse_pc_probe_beats_dense_default_without_parity_loss() -> None:
    probe = _fp3d_sparse_pc_probe()

    for case_name, rows in probe["fp_cases"].items():
        default = rows["default"]
        sparse_pc = rows["sparse_pc_gmres"]

        assert default["status"] == "ok", case_name
        assert sparse_pc["status"] == "ok", case_name
        assert sparse_pc["vs_fortran_count"] == 0, case_name
        assert float(sparse_pc["elapsed_s"]) < float(default["elapsed_s"]), case_name
        assert float(sparse_pc["ru_maxrss_mb"]) < float(default["ru_maxrss_mb"]), case_name
        assert set(sparse_pc.get("vs_default_sample", [])) <= {"linearSolverResidualTargetRatio"}


def test_fp3d_sparse_pc_probe_does_not_promote_pas_sparse_host() -> None:
    probe = _fp3d_sparse_pc_probe()
    rows = probe["pas_cases"]["HSX_PASCollisions_fullTrajectories"]

    assert rows["default"]["status"] == "ok"
    assert rows["default"]["rhs1_preconditioner"] == "pas_tz"
    assert rows["default"]["vs_fortran_count"] == 0
    assert rows["sparse_host"]["status"] == "error"
    assert rows["sparse_pc_gmres"]["status"] == "error"
    assert rows["sparse_host_safe"]["status"] == "ok"
    assert rows["sparse_host_safe"]["vs_fortran_count"] > 0
    assert "FSABjHat" in rows["sparse_host_safe"]["vs_fortran_sample"]


def test_production_bounded_suite_reclassifies_fortran_snes_divergence() -> None:
    probe = _tokamak_er_dense_probe()
    suite = probe["production_bounded_suite"]["after_fortran_divergence_reclassification"]

    assert suite["status_counts"]["parity_ok"] == 9
    assert suite["status_counts"]["fortran_diverged"] == 1
    assert suite["status_counts"]["parity_mismatch"] == 1
    assert suite["status_counts"]["max_attempts"] == 0
    assert suite["status_counts"]["jax_error"] == 0
    assert suite["blocker_counts"]["reference solver quality"] == 2
    assert suite["output_key_coverage_missing_total"] == 0

    qn_probe = probe["fortran_divergence_reclassification_probe"]
    assert qn_probe["status"] == "fortran_diverged"
    assert qn_probe["blocker_type"] == "reference solver quality"
    assert qn_probe["jax_was_not_run_by_suite"] is True
    assert qn_probe["manual_jax_check"]["NIterations"] == 1
    assert qn_probe["manual_jax_check"]["linearSolverResidualNorm"] < 1.0e-8


def test_tokamak_er_dense_probe_promotes_runtime_win_with_memory_cap() -> None:
    cases = _tokamak_er_dense_probe()["cases"]

    for case_name in (
        "tokamak_1species_FPCollisions_withEr_DKESTrajectories",
        "tokamak_1species_PASCollisions_withEr_fullTrajectories",
    ):
        row = cases[case_name]
        before = row["default_before"]
        after = row["default_after"]

        assert before["status"] == "ok", case_name
        assert after["status"] == "ok", case_name
        assert after["used_dense_auto"] is True, case_name
        assert after["used_sparse_fallback"] is False, case_name
        assert after["rhs1_preconditioner"] is None, case_name
        assert after["vs_fortran_count"] == 0, case_name
        assert float(after["elapsed_s"]) < 0.1 * float(before["elapsed_s"]), case_name
        assert float(after["ru_maxrss_mb"]) > float(before["ru_maxrss_mb"]), case_name

    policy = _tokamak_er_dense_probe()["policy"]
    assert policy["cpu_only"] is True
    assert policy["active_size_min_default"] == 5000
    assert policy["active_size_max_default"] == 6500
    assert policy["dense_matrix_bytes_max_default"] == 350000000


def test_tokamak_er_dense_probe_covers_fp_full_trajectory_case() -> None:
    cases = _tokamak_er_dense_probe()["cases"]
    row = cases["tokamak_1species_FPCollisions_withEr_fullTrajectories"]
    before = row["default_before"]
    after = row["default_after"]
    harness = row["harness_after"]

    assert after["status"] == "ok"
    assert after["used_dense_auto"] is True
    assert after["vs_fortran_count"] == 0
    assert after["rhs1_preconditioner"] is None
    assert float(after["elapsed_s"]) < 0.3 * float(before["elapsed_s"])

    assert harness["status"] == "parity_ok"
    assert harness["strict_n_mismatch_common"] == 0
    assert harness["output_key_missing_total"] == 0


def test_tokamak_er_dense_probe_keeps_noer_and_sparse_failures_out_of_default() -> None:
    cases = _tokamak_er_dense_probe()["cases"]
    control = cases["tokamak_1species_PASCollisions_noEr"]["default_after_control"]

    assert control["status"] == "ok"
    assert control["used_dense_auto"] is False
    assert control["rhs1_preconditioner"] == "pas_tokamak_theta"
    assert control["vs_fortran_count"] == 0
    assert float(control["ru_maxrss_mb"]) < 1000.0

    fp_rejected = cases["tokamak_1species_FPCollisions_withEr_DKESTrajectories"]["rejected_variants"]
    assert fp_rejected["sparse_host"]["status"] == "error"
    assert fp_rejected["sparse_pc_gmres"]["status"] == "error"
    assert fp_rejected["bicgstab"]["elapsed_s"] > 100.0

    pas_rejected = cases["tokamak_1species_PASCollisions_withEr_fullTrajectories"]["rejected_variants"]
    assert pas_rejected["xblock_tz"]["status"] == "timeout"
    assert pas_rejected["pas_tz"]["elapsed_s"] > 80.0
    assert pas_rejected["sparse_host_safe"]["status"] == "error"
