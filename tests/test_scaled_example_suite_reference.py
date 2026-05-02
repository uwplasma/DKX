from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys

import h5py
import pytest


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_scaled_example_suite.py"
sys.path.insert(0, str(_SCRIPT_PATH.parent))
_SPEC = importlib.util.spec_from_file_location("run_scaled_example_suite", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_stage_reference_fortran_artifacts = _MODULE._stage_reference_fortran_artifacts
_run_prepared_case = _MODULE._run_prepared_case
_write_suite_audits = _MODULE._write_suite_audits
_write_suite_outputs = _MODULE._write_suite_outputs
_iter_inputs = _MODULE._iter_inputs
_load_reference_case_metrics = _MODULE._load_reference_case_metrics
_auto_production_manifest_path = _MODULE._auto_production_manifest_path
_filter_inputs_by_production_recommendation = _MODULE._filter_inputs_by_production_recommendation
_load_production_manifest_cases = _MODULE._load_production_manifest_cases
_run_recommendation_allowed = _MODULE._run_recommendation_allowed

from run_reduced_upstream_suite import CaseResult  # noqa: E402
from run_reduced_upstream_suite import _classify_blocker  # noqa: E402
from run_reduced_upstream_suite import _parse_jax_rhs_norm_from_log  # noqa: E402
from run_reduced_upstream_suite import _reference_solve_quality_note  # noqa: E402
from run_reduced_upstream_suite import _runtime_metric_for_basis  # noqa: E402
from run_reduced_upstream_suite import _solver_tolerance_from_namelist  # noqa: E402


def test_stage_reference_fortran_artifacts_uses_last_success(tmp_path: Path) -> None:
    case_name = "tokamak_case"
    ref_root = tmp_path / "reference"
    case_dir = ref_root / case_name / "last_success"
    case_dir.mkdir(parents=True)
    (case_dir / "sfincsOutput_fortran.h5").write_text("fortran-h5", encoding="utf-8")
    (case_dir / "sfincs_fortran.log").write_text("fortran-log", encoding="utf-8")
    (ref_root / case_name / "input.namelist").write_text("&general\n  NTHETA = 21\n/\n", encoding="utf-8")

    case_input = tmp_path / "input.namelist"
    case_input.write_text("&general\n  NTHETA = 21\n/\n", encoding="utf-8")
    out_dir = tmp_path / "out"

    staged, effective_input = _stage_reference_fortran_artifacts(
        case_name=case_name,
        case_input=case_input,
        case_out_dir=out_dir,
        reference_results_root=ref_root,
    )

    assert staged is True
    assert effective_input == case_input
    assert (out_dir / "fortran_run" / "input.namelist").read_text(encoding="utf-8") == case_input.read_text(encoding="utf-8")
    assert (out_dir / "fortran_run" / "sfincsOutput.h5").read_text(encoding="utf-8") == "fortran-h5"
    assert (out_dir / "fortran_run" / "sfincs.log").read_text(encoding="utf-8") == "fortran-log"


def test_stage_reference_fortran_artifacts_reuses_reference_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_name = "tokamak_case"
    ref_root = tmp_path / "reference"
    case_dir = ref_root / case_name / "fortran_run"
    case_dir.mkdir(parents=True)
    (case_dir / "sfincsOutput.h5").write_text("fortran-h5", encoding="utf-8")
    (case_dir / "input.namelist").write_text(
        "&general\n/\n&resolutionParameters\n  NTHETA = 31\n  NZETA = 9\n  NX = 2\n  NXI = 17\n/\n",
        encoding="utf-8",
    )

    case_input = tmp_path / "input.namelist"
    case_input.write_text(
        "&general\n/\n&geometryParameters\n  equilibriumFile = '/office/path/w7x.nc'\n/\n"
        "&resolutionParameters\n  NTHETA = 21\n  NZETA = 7\n  NX = 3\n  NXI = 18\n/\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(_MODULE, "localize_equilibrium_file_in_place", lambda **_: None)

    staged, effective_input = _stage_reference_fortran_artifacts(
        case_name=case_name,
        case_input=case_input,
        case_out_dir=tmp_path / "out",
        reference_results_root=ref_root,
    )

    assert staged is True
    assert effective_input != case_input
    text = effective_input.read_text(encoding="utf-8")
    assert "equilibriumFile = '/office/path/w7x.nc'" in text
    assert "  NTHETA = 31" in text
    assert "  NZETA = 9" in text
    assert "  NX = 2" in text
    assert "  NXI = 17" in text
    assert (tmp_path / "out" / "fortran_run" / "input.namelist").read_text(encoding="utf-8") == text


def test_stage_reference_fortran_artifacts_prefers_fortran_input_over_case_root_input(tmp_path: Path) -> None:
    case_name = "tokamak_case"
    ref_root = tmp_path / "reference"
    case_root = ref_root / case_name
    (case_root / "fortran_run").mkdir(parents=True)
    (case_root / "fortran_run" / "sfincsOutput.h5").write_text("fortran-h5", encoding="utf-8")
    (case_root / "fortran_run" / "input.namelist").write_text(
        "&general\n/\n&resolutionParameters\n  NTHETA = 19\n  NZETA = 1\n  NX = 7\n  NXI = 39\n/\n",
        encoding="utf-8",
    )
    (case_root / "input.namelist").write_text(
        "&general\n/\n&resolutionParameters\n  NTHETA = 21\n  NZETA = 1\n  NX = 8\n  NXI = 40\n/\n",
        encoding="utf-8",
    )

    case_input = tmp_path / "input.namelist"
    case_input.write_text(
        "&general\n/\n&resolutionParameters\n  NTHETA = 21\n  NZETA = 1\n  NX = 8\n  NXI = 40\n/\n",
        encoding="utf-8",
    )

    staged, effective_input = _stage_reference_fortran_artifacts(
        case_name=case_name,
        case_input=case_input,
        case_out_dir=tmp_path / "out",
        reference_results_root=ref_root,
    )

    assert staged is True
    assert effective_input != case_input
    text = effective_input.read_text(encoding="utf-8")
    assert "  NTHETA = 19" in text
    assert "  NX = 7" in text
    assert "  NXI = 39" in text
    assert (tmp_path / "out" / "fortran_run" / "input.namelist").read_text(encoding="utf-8") == text


def test_stage_reference_fortran_artifacts_localizes_staged_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case_name = "tokamak_case"
    ref_root = tmp_path / "reference"
    case_dir = ref_root / case_name / "fortran_run"
    case_dir.mkdir(parents=True)
    (case_dir / "sfincsOutput.h5").write_text("fortran-h5", encoding="utf-8")
    (case_dir / "input.namelist").write_text(
        "&general\n/\n&geometryParameters\n  equilibriumFile = '/office/path/w7x.nc'\n/\n",
        encoding="utf-8",
    )

    def fake_localize(*, input_namelist: Path, overwrite: bool) -> None:
        text = input_namelist.read_text(encoding="utf-8")
        input_namelist.write_text(text.replace("/office/path/w7x.nc", "./w7x.nc"), encoding="utf-8")

    monkeypatch.setattr(_MODULE, "localize_equilibrium_file_in_place", fake_localize)

    case_input = tmp_path / "input.namelist"
    case_input.write_text(
        "&general\n/\n&geometryParameters\n  equilibriumFile = '/office/path/w7x.nc'\n/\n",
        encoding="utf-8",
    )

    staged, _effective_input = _stage_reference_fortran_artifacts(
        case_name=case_name,
        case_input=case_input,
        case_out_dir=tmp_path / "out",
        reference_results_root=ref_root,
    )

    assert staged is True
    assert "./w7x.nc" in (tmp_path / "out" / "fortran_run" / "input.namelist").read_text(encoding="utf-8")


def test_stage_reference_fortran_artifacts_uses_case_search_dir_for_localization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_name = "additional_examples"
    ref_root = tmp_path / "reference"
    case_dir = ref_root / case_name / "fortran_run"
    case_dir.mkdir(parents=True)
    (case_dir / "sfincsOutput.h5").write_text("fortran-h5", encoding="utf-8")
    (case_dir / "input.namelist").write_text(
        "&general\n/\n&geometryParameters\n  equilibriumFile = 'wout_extra.nc'\n/\n",
        encoding="utf-8",
    )

    def fake_localize(*, input_namelist: Path, overwrite: bool) -> None:
        assert os.environ.get("SFINCS_JAX_EQUILIBRIA_DIRS", "") == str(case_input.parent)
        text = input_namelist.read_text(encoding="utf-8")
        input_namelist.write_text(text.replace("wout_extra.nc", "./wout_extra.nc"), encoding="utf-8")

    monkeypatch.setattr(_MODULE, "localize_equilibrium_file_in_place", fake_localize)

    case_input = tmp_path / "case" / "input.namelist"
    case_input.parent.mkdir(parents=True)
    (case_input.parent / "wout_extra.nc").write_text("netcdf", encoding="utf-8")
    case_input.write_text(
        "&general\n/\n&geometryParameters\n  equilibriumFile = 'wout_extra.nc'\n/\n",
        encoding="utf-8",
    )

    staged, _effective_input = _stage_reference_fortran_artifacts(
        case_name=case_name,
        case_input=case_input,
        case_out_dir=tmp_path / "out",
        reference_results_root=ref_root,
    )

    assert staged is True
    assert "./wout_extra.nc" in (tmp_path / "out" / "fortran_run" / "input.namelist").read_text(encoding="utf-8")


def test_iter_inputs_ignores_staged_artifact_directories(tmp_path: Path) -> None:
    (tmp_path / "real_case").mkdir()
    (tmp_path / "real_case" / "input.namelist").write_text("&general\n/\n", encoding="utf-8")
    (tmp_path / "real_case" / "fortran_run").mkdir()
    (tmp_path / "real_case" / "fortran_run" / "input.namelist").write_text("&general\n/\n", encoding="utf-8")
    (tmp_path / "real_case" / "last_success").mkdir()
    (tmp_path / "real_case" / "last_success" / "input.namelist").write_text("&general\n/\n", encoding="utf-8")

    inputs = _iter_inputs(tmp_path)

    assert inputs == [tmp_path / "real_case" / "input.namelist"]


def test_auto_production_manifest_path_detects_generated_inputs_root(tmp_path: Path) -> None:
    inputs_root = tmp_path / "production" / "inputs"
    inputs_root.mkdir(parents=True)
    manifest = inputs_root.parent / "manifest.json"
    manifest.write_text('{"cases": []}', encoding="utf-8")

    assert _auto_production_manifest_path(inputs_root, None) == manifest
    explicit = tmp_path / "explicit.json"
    assert _auto_production_manifest_path(inputs_root, explicit) == explicit
    assert _auto_production_manifest_path(tmp_path / "examples", None) is None


def test_run_recommendation_order_keeps_local_runs_bounded() -> None:
    assert _run_recommendation_allowed("bounded_local_ok", "bounded_local_ok")
    assert not _run_recommendation_allowed("bounded_remote", "bounded_local_ok")
    assert _run_recommendation_allowed("bounded_remote", "bounded_remote")
    assert not _run_recommendation_allowed("remote_or_cluster_only", "bounded_remote")
    assert _run_recommendation_allowed("remote_or_cluster_only", "remote_or_cluster_only")
    assert _run_recommendation_allowed("remote_or_cluster_only", "all")
    assert not _run_recommendation_allowed(None, "bounded_local_ok")


def test_production_manifest_filter_skips_remote_only_cases(tmp_path: Path) -> None:
    prod_root = tmp_path / "production"
    local_input = prod_root / "inputs" / "local_case" / "input.namelist"
    remote_input = prod_root / "inputs" / "remote_case" / "input.namelist"
    local_input.parent.mkdir(parents=True)
    remote_input.parent.mkdir(parents=True)
    local_input.write_text("&general\n/\n", encoding="utf-8")
    remote_input.write_text("&general\n/\n", encoding="utf-8")
    manifest = prod_root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case": "local_case",
                        "input": "inputs/local_case/input.namelist",
                        "size_estimate": {"run_recommendation": "bounded_local_ok"},
                    },
                    {
                        "case": "remote_case",
                        "input": "inputs/remote_case/input.namelist",
                        "size_estimate": {"run_recommendation": "remote_or_cluster_only"},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    _resolved_manifest, cases_by_input = _load_production_manifest_cases(manifest)
    case_names = {local_input: "local_case", remote_input: "remote_case"}

    kept, skipped = _filter_inputs_by_production_recommendation(
        inputs=[local_input, remote_input],
        case_names=case_names,
        manifest_cases_by_input=cases_by_input,
        max_run_recommendation="bounded_local_ok",
    )

    assert kept == [local_input]
    assert skipped == [
        {
            "case": "remote_case",
            "input": str(remote_input),
            "run_recommendation": "remote_or_cluster_only",
            "max_run_recommendation": "bounded_local_ok",
            "reason": "production_run_recommendation_guard",
        }
    ]


def test_main_auto_manifest_guard_records_skipped_cases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prod_root = tmp_path / "production"
    inputs_root = prod_root / "inputs"
    local_input = inputs_root / "local_case" / "input.namelist"
    remote_input = inputs_root / "remote_case" / "input.namelist"
    for path in (local_input, remote_input):
        path.parent.mkdir(parents=True)
        path.write_text(
            "&resolutionParameters\n"
            "  NTHETA = 3\n"
            "  NZETA = 1\n"
            "  NX = 1\n"
            "  NXI = 3\n"
            "/\n",
            encoding="utf-8",
        )
    (prod_root / "manifest.json").write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case": "local_case",
                        "input": "inputs/local_case/input.namelist",
                        "size_estimate": {"run_recommendation": "bounded_local_ok"},
                    },
                    {
                        "case": "remote_case",
                        "input": "inputs/remote_case/input.namelist",
                        "size_estimate": {"run_recommendation": "remote_or_cluster_only"},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    ref_root = tmp_path / "reference"
    ref_root.mkdir()
    out_root = tmp_path / "out"
    ran_cases: list[str] = []

    def fake_run_prepared_case(**kwargs):
        ran_cases.append(kwargs["case_name"])
        return _case_result(kwargs["case_name"])

    monkeypatch.setattr(_MODULE, "_run_prepared_case", fake_run_prepared_case)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_scaled_example_suite.py",
            "--examples-root",
            str(inputs_root),
            "--reference-results-root",
            str(ref_root),
            "--out-root",
            str(out_root),
            "--pattern",
            "_case",
            "--fortran-min-runtime-s",
            "0",
            "--runtime-adjustment-iters",
            "0",
            "--max-attempts",
            "1",
            "--reset-report",
        ],
    )

    assert _MODULE.main() == 0

    run_manifest = json.loads((out_root / "run_manifest.json").read_text(encoding="utf-8"))
    assert ran_cases == ["local_case"]
    assert run_manifest["production_manifest"] == str((prod_root / "manifest.json").resolve())
    assert run_manifest["max_run_recommendation"] == "bounded_local_ok"
    assert [item["case"] for item in run_manifest["skipped_by_recommendation"]] == ["remote_case"]


def test_load_reference_case_metrics_reads_suite_report(tmp_path: Path) -> None:
    ref_root = tmp_path / "reference"
    ref_root.mkdir()
    (ref_root / "suite_report.json").write_text(
        json.dumps(
            [
                {"case": "other", "fortran_runtime_s": 1.0},
                {"case": "target", "fortran_runtime_s": 76.5, "fortran_max_rss_mb": 238.25},
            ]
        ),
        encoding="utf-8",
    )

    metrics = _load_reference_case_metrics(ref_root, "target")

    assert metrics["fortran_runtime_s"] == pytest.approx(76.5)
    assert metrics["fortran_max_rss_mb"] == pytest.approx(238.25)


def _case_result(case: str, *, status: str = "parity_ok", strict_mismatches: int = 0) -> CaseResult:
    return CaseResult(
        case=case,
        status=status,
        blocker_type="",
        message="",
        attempts=1,
        reductions=0,
        fortran_runtime_s=1.0,
        jax_runtime_s=2.0,
        jax_runtime_s_cold=2.1,
        jax_runtime_s_warm=1.9,
        jax_logged_elapsed_s=1.7,
        fortran_max_rss_mb=100.0,
        jax_max_rss_mb=200.0,
        jax_solver_iters_mean=5.0,
        jax_solver_iters_min=5,
        jax_solver_iters_max=5,
        jax_solver_iters_n=1,
        jax_solver_iters_detail=[5],
        jax_solver_kinds=["gmres"],
        print_parity_signals=9,
        print_parity_total=9,
        print_missing_signals=[],
        n_common_keys=10,
        n_mismatch_common=0,
        mismatch_keys_sample=[],
        n_mismatch_solver=0,
        n_mismatch_physics=0,
        mismatch_solver_sample=[],
        mismatch_physics_sample=[],
        max_abs_mismatch=0.0,
        strict_n_common_keys=10,
        strict_n_mismatch_common=strict_mismatches,
        strict_mismatch_keys_sample=["pressureAnisotropy"] if strict_mismatches else [],
        strict_n_mismatch_solver=0,
        strict_n_mismatch_physics=strict_mismatches,
        strict_mismatch_solver_sample=[],
        strict_mismatch_physics_sample=["pressureAnisotropy"] if strict_mismatches else [],
        strict_max_abs_mismatch=1.0e-6 if strict_mismatches else 0.0,
        final_resolution={"NTHETA": 21, "NZETA": 1, "NX": 8, "NXI": 31},
        input_path=f"{case}/input.namelist",
        promoted_input_path=None,
        fortran_h5=f"{case}/fortran.h5",
        jax_h5=f"{case}/jax.h5",
    )


def _write_h5(path: Path, keys: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        for key, value in keys.items():
            h5[key] = float(value)


def test_write_suite_outputs_writes_incremental_reports(tmp_path: Path) -> None:
    rows = [_case_result("b_case"), _case_result("a_case", strict_mismatches=1)]

    _write_suite_outputs(rows, tmp_path)

    report = tmp_path / "suite_report.json"
    report_strict = tmp_path / "suite_report_strict.json"
    report_rst = tmp_path / "suite_status.rst"
    report_rst_strict = tmp_path / "suite_status_strict.rst"
    summary = tmp_path / "summary.md"

    for path in (report, report_strict, report_rst, report_rst_strict, summary):
        assert path.exists(), f"missing {path.name}"

    report_rows = json.loads(report.read_text(encoding="utf-8"))
    strict_rows = json.loads(report_strict.read_text(encoding="utf-8"))

    assert [row["case"] for row in report_rows] == ["a_case", "b_case"]
    assert report_rows[0]["jax_logged_elapsed_s"] == 1.7
    assert strict_rows[0]["status"] == "parity_mismatch"
    assert strict_rows[0]["jax_logged_elapsed_s"] == 1.7
    assert strict_rows[0]["n_mismatch_common"] == 1
    assert "Scaled Example Suite Summary" in summary.read_text(encoding="utf-8")


def test_write_suite_audits_emits_key_coverage_and_runtime_drift(tmp_path: Path) -> None:
    case_dir = tmp_path / "a_case"
    _write_h5(case_dir / "fortran.h5", {"a": 1.0, "b": 2.0})
    _write_h5(case_dir / "jax.h5", {"a": 1.0, "b": 2.0, "c": 3.0})
    rows = [_case_result("a_case")]
    rows[0].fortran_h5 = str(case_dir / "fortran.h5")
    rows[0].jax_h5 = str(case_dir / "jax.h5")
    _write_suite_outputs(rows, tmp_path)

    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps([{"case": "a_case", "jax_runtime_s": 1.0}]), encoding="utf-8")

    audit = _write_suite_audits(
        out_root=tmp_path,
        runtime_baseline_report=baseline,
        runtime_drift_threshold_ratio=1.25,
        runtime_drift_min_baseline_runtime_s=0.0,
    )

    assert (tmp_path / "suite_output_key_coverage.json").exists()
    assert (tmp_path / "suite_output_key_coverage_summary.json").exists()
    assert (tmp_path / "suite_runtime_drift.json").exists()
    assert (tmp_path / "suite_runtime_drift_summary.json").exists()
    assert audit["output_key_coverage"]["missing_total"] == 0
    assert audit["output_key_coverage"]["extra_total"] == 1
    assert audit["runtime_drift"]["flagged_cases"] == 1


def test_classify_blocker_treats_cuda_dense_custom_calls_as_solver_branch(tmp_path: Path) -> None:
    log_path = tmp_path / "sfincs_jax.log"
    log_path.write_text(
        "jaxlib._jax.XlaRuntimeError: UNIMPLEMENTED: No registered implementation for custom call "
        "to cusolver_getrf_ffi for platform CUDA\n",
        encoding="utf-8",
    )

    assert (
        _classify_blocker(
            status="jax_error",
            note="JAX error: CalledProcessError",
            mismatch_keys=[],
            jax_log=log_path,
        )
        == "solver branch mismatch"
    )


def test_reference_solve_quality_uses_rhs_scaled_target(tmp_path: Path) -> None:
    log_path = tmp_path / "sfincs_jax.log"
    log_path.write_text(
        "solve_v3_full_system_linear_gmres: rhs_norm=6.811377e-04\n"
        "solve_v3_full_system_linear_gmres: residual_norm=3.031074e-17\n",
        encoding="utf-8",
    )

    rhs_norm = _parse_jax_rhs_norm_from_log(log_path)
    note = _reference_solve_quality_note(
        final_fortran_residual=1.1769038e-08,
        solver_tolerance=1.0e-6,
        jax_rhs_norm=rhs_norm,
    )

    assert rhs_norm == pytest.approx(6.811377e-04)
    assert note is not None
    assert "solverTolerance*rhs_norm" in note
    assert "reference-solve quality suspect" in note


def test_reference_solve_quality_does_not_warn_for_rhs_scaled_converged_reference() -> None:
    note = _reference_solve_quality_note(
        final_fortran_residual=1.0e-10,
        solver_tolerance=1.0e-6,
        jax_rhs_norm=6.8e-4,
    )

    assert note is None


def test_solver_tolerance_parser_accepts_fortran_uppercase_keys(tmp_path: Path) -> None:
    input_path = tmp_path / "input.namelist"
    input_path.write_text(
        "&resolutionParameters\n  SOLVERTOLERANCE = 1.0d-6\n/\n",
        encoding="utf-8",
    )

    assert _solver_tolerance_from_namelist(input_path) == pytest.approx(1.0e-6)


def test_run_prepared_case_passes_reference_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_run_case(**kwargs):
        seen.update(kwargs)
        return _case_result("tokamak_case")

    monkeypatch.setattr(_MODULE, "_run_case", fake_run_case)

    case_input = tmp_path / "input.scale_seed.namelist"
    case_input.write_text("&general\n/\n", encoding="utf-8")
    reference_input = tmp_path / "input.reference.namelist"
    reference_input.write_text("&general\n/\n", encoding="utf-8")

    result = _run_prepared_case(
        case_name="tokamak_case",
        case_input=case_input,
        reference_input=reference_input,
        case_out_dir=tmp_path / "out",
        fortran_exe=tmp_path / "sfincs",
        timeout_s=10.0,
        rtol=5e-4,
        atol=1e-9,
        max_attempts=1,
        target_runtime_s=1.0,
        target_runtime_max_s=30.0,
        target_runtime_max_iters=2,
        target_runtime_basis="fortran",
        reuse_fortran=False,
        collect_iterations=True,
        jax_repeats=1,
        jax_cache_dir=tmp_path / ".jax_cache",
        jax_profile_mode="off",
        equilibria_search_dir=case_input.parent,
        reference_results_root=None,
    )

    assert result.case == "tokamak_case"
    assert seen["case_input"] == case_input
    assert seen["reference_input"] == reference_input
    assert seen["target_runtime_s"] == 1.0
    assert seen["target_runtime_max_s"] == 30.0
    assert seen["target_runtime_max_iters"] == 2
    assert seen["target_runtime_basis"] == "fortran"
    assert seen["jax_profile_mode"] == "off"


def test_run_prepared_case_preserves_staged_reference_wall_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_name = "ntx_case"
    ref_root = tmp_path / "reference"
    ref_case = ref_root / case_name / "fortran_run"
    ref_case.mkdir(parents=True)
    (ref_case / "sfincsOutput.h5").write_text("fortran-h5", encoding="utf-8")
    (ref_case / "sfincs.log").write_text("Done. Time to solve: 1.31 seconds\n", encoding="utf-8")
    (ref_case / "input.namelist").write_text("&general\n/\n", encoding="utf-8")
    (ref_root / "suite_report.json").write_text(
        json.dumps([{"case": case_name, "fortran_runtime_s": 76.45, "fortran_max_rss_mb": 238.75}]),
        encoding="utf-8",
    )

    def fake_run_case(**kwargs):
        result = _case_result(case_name)
        result.fortran_runtime_s = 1.31
        result.fortran_max_rss_mb = 99.0
        return result

    monkeypatch.setattr(_MODULE, "_run_case", fake_run_case)

    case_input = tmp_path / "input.namelist"
    case_input.write_text("&general\n/\n", encoding="utf-8")

    result = _run_prepared_case(
        case_name=case_name,
        case_input=case_input,
        reference_input=case_input,
        case_out_dir=tmp_path / "out",
        fortran_exe=None,
        timeout_s=10.0,
        rtol=5e-4,
        atol=1e-9,
        max_attempts=1,
        target_runtime_s=None,
        target_runtime_max_s=None,
        target_runtime_max_iters=0,
        target_runtime_basis="fortran",
        reuse_fortran=False,
        collect_iterations=True,
        jax_repeats=1,
        jax_cache_dir=None,
        jax_profile_mode="off",
        equilibria_search_dir=case_input.parent,
        reference_results_root=ref_root,
    )

    assert result.fortran_runtime_s == pytest.approx(76.45)
    assert result.fortran_max_rss_mb == pytest.approx(238.75)


def test_runtime_metric_for_basis_uses_fortran_only_when_requested() -> None:
    assert _runtime_metric_for_basis(fortran_runtime_s=0.8, jax_runtime_s=12.0, basis="fortran") == pytest.approx(0.8)
    assert _runtime_metric_for_basis(fortran_runtime_s=0.8, jax_runtime_s=12.0, basis="max") == pytest.approx(12.0)
    assert _runtime_metric_for_basis(fortran_runtime_s=None, jax_runtime_s=12.0, basis="fortran") is None
