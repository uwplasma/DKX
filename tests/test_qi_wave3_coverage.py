from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "run_qi_seed_robustness.py"


def _load_qi_seed_module() -> object:
    spec = importlib.util.spec_from_file_location("wave3_run_qi_seed_robustness", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_source_input(path: Path) -> Path:
    path.write_text(
        (
            "&resolutionParameters\n"
            "  Ntheta = 10\n"
            "  Nzeta = 20\n"
            "  Nx = 6\n"
            "  Nxi = 8\n"
            "/\n"
        ),
        encoding="utf-8",
    )
    return path


def test_compact_qi_artifact_prefers_trace_and_preserves_progress_fallbacks(
    tmp_path: Path,
) -> None:
    qi_seed = _load_qi_seed_module()
    source_input = _write_source_input(tmp_path / "input.namelist")
    probe_env = {
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER": "1",
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_USE_IN_KRYLOV": "1",
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_OPERATOR_KRYLOV_ENRICHMENT": "1",
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_REUSE_COARSE_OPERATOR": "1",
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_AUGMENTED_KRYLOV": "1",
    }
    progress_only_events = [
        "The matrix is 12 x 12 elements.",
        "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres QI device "
        "preconditioner accepted residual 4.0e-05 -> 2.0e-05 "
        "(rank=5 ratio=5.0e-01 seed_only=1 operator_krylov=0 coarse_reuse=0)",
        "solve_v3_full_system_linear_gmres: QI augmented Krylov enabled rank=7",
    ]
    manifest = {
        "source_input": str(source_input),
        "resolution_scale": 0.5,
        "case_count": 2,
        "solve_method": "xblock_sparse_pc_gmres",
        "probe_preset": "operator-krylov-device-qi",
        "probe_env": probe_env,
        "cases": [
            {
                "case": "qi_seed_0001",
                "resolution": {"NTHETA": 5, "NZETA": 10, "NX": 3, "NXI": 4},
            },
            {
                "case": "qi_seed_0002",
                "resolution": {"NTHETA": 5, "NZETA": 10, "NX": 3, "NXI": 4},
            },
        ],
        "execution": {
            "summary": {"attempted": 2, "process_passed": 1, "timed_out": 1},
            "gates": {"passed": False},
            "results": [
                {
                    "case": "qi_seed_0001",
                    "seed": 1,
                    "returncode": 0,
                    "timed_out": False,
                    "output_exists": True,
                    "solver_trace_exists": True,
                    "elapsed_s": 1.0,
                    "progress_events": [
                        "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres side probe "
                        "keep side=right->right method=gmres->gmres residual=9.0e-01 ratio=9.0e+00"
                    ],
                    "stdout_tail": [],
                    "stderr_tail": [],
                    "solver_trace_summary": {
                        "backend": "gpu",
                        "active_size": 10,
                        "total_size": 100,
                        "selected_path": "trace_selected",
                        "solve_method": "xblock_sparse_pc_gmres",
                        "precondition_side": "left",
                        "converged": True,
                        "accepted_converged": True,
                        "residual_norm": 1.0e-12,
                        "residual_target": 1.0e-11,
                        "residual_ratio": 0.1,
                        "xblock_qi_device_preconditioner_used": True,
                        "xblock_qi_device_preconditioner_use_in_krylov": True,
                        "xblock_qi_device_preconditioner_operator_on_basis_shape": [2, 2],
                        "xblock_qi_device_preconditioner_coarse_operator_shape": [2, 2],
                    },
                },
                {
                    "case": "qi_seed_0002",
                    "seed": 2,
                    "returncode": 124,
                    "timed_out": True,
                    "output_exists": False,
                    "solver_trace_exists": False,
                    "elapsed_s": 9.0,
                    "progress_events": progress_only_events,
                    "stdout_tail": [],
                    "stderr_tail": ["QI seed execution timed out after 9.000 s."],
                    "solver_trace_summary": None,
                },
            ],
        },
    }

    artifact = qi_seed._compact_execution_artifact(manifest)
    first, second = artifact["seeds"]

    assert artifact["public_cli_default_path"] is False
    assert first["selected_path"] == "trace_selected"
    assert first["precondition_side"] == "left"
    assert first["active_size"] == 10
    assert first["evidence_class"] == "device_qi_installed_krylov_coarse_reuse"
    assert first["promotion_eligible"] is True
    assert first["observed_qi_device_operator_krylov"] is True
    assert first["observed_qi_device_coarse_reuse"] is True

    assert second["active_size"] == 12
    assert second["total_size"] is None
    assert second["xblock_qi_device_preconditioner_used"] is True
    assert second["xblock_qi_device_preconditioner_seed_only"] is True
    assert second["xblock_qi_device_preconditioner_use_in_krylov"] is False
    assert second["xblock_device_fgmres_qi_augmented_krylov_used"] is True
    assert second["xblock_device_fgmres_qi_augmented_krylov_rank"] == 7
    assert second["evidence_class"] == "device_qi_seed_only_probe"
    assert second["run_outcome"] == "timed_out"
    assert second["promotion_eligible"] is False
    assert "failed_before_solver_trace_summary" in second["evidence_tags"]

    classification = artifact["evidence_classification"]
    assert classification["classes"] == [
        "device_qi_installed_krylov_coarse_reuse",
        "device_qi_seed_only_probe",
    ]
    assert classification["has_observed_installed_krylov"] is True
    assert classification["has_observed_coarse_reuse"] is True
    assert classification["has_observed_augmented_krylov"] is True
    assert classification["promotion_eligible_seed_count"] == 1


def test_qi_seed_helper_error_branches_and_preset_solve_method_selection() -> None:
    qi_seed = _load_qi_seed_module()

    with pytest.raises(ValueError, match="requires --execute"):
        qi_seed._compact_execution_artifact({})
    with pytest.raises(ValueError, match="no cases"):
        qi_seed._compact_execution_artifact({"execution": {"results": []}, "cases": []})
    with pytest.raises(ValueError, match="no result list"):
        qi_seed._compact_execution_artifact({"execution": {}, "cases": [{"case": "a"}]})
    with pytest.raises(ValueError, match="Unknown QI probe preset"):
        qi_seed._probe_env_for_preset("mystery")

    assert qi_seed._solve_method_for_probe_preset(solve_method="", probe_preset="none") == "auto"
    assert (
        qi_seed._solve_method_for_probe_preset(
            solve_method="auto",
            probe_preset="operator-krylov-device-qi",
        )
        == "xblock_sparse_pc_gmres"
    )
    assert (
        qi_seed._solve_method_for_probe_preset(
            solve_method="dense",
            probe_preset="operator-krylov-device-qi",
        )
        == "dense"
    )


def test_qi_evidence_manifest_classifies_embedded_legacy_and_failed_artifacts(
    tmp_path: Path,
) -> None:
    qi_seed = _load_qi_seed_module()
    source_input = _write_source_input(tmp_path / "input.namelist")
    embedded = tmp_path / "embedded.json"
    host_fallback = tmp_path / "qi_seed_robustness_device_host_fallback.json"
    failed = tmp_path / "qi_seed_robustness_scale060_operator_krylov_device_qi_gpu_timeout.json"

    embedded.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "artifact_kind": "qi_seed_execution_summary",
                "case_count": 1,
                "public_cli_default_path": True,
                "resolution": {"ntheta": 5, "Nzeta": 10, "Nx": 3, "Nxi": 4},
                "active_size_estimate": 42,
                "execution_summary": {"backends": ["cpu"], "max_residual_ratio": 0.25},
                "gates": {"passed": True},
                "evidence_classification": {
                    "classes": ["public_auto_or_legacy"],
                    "tags": ["observed_seed_only"],
                    "outcomes": ["process_passed"],
                    "has_failed_before_summary_json": False,
                    "has_observed_installed_krylov": False,
                    "has_observed_coarse_reuse": False,
                    "promotion_eligible_seed_count": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    host_fallback.write_text(
        json.dumps(
            {
                "passed": 1,
                "failed": 0,
                "backend": "gpu",
                "resolution": {"NTHETA": 10, "NZETA": 20, "NX": 6, "NXI": 8},
                "runs": {
                    "after_patch": {
                        "backend": "gpu",
                        "residual_ratio": 0.2,
                        "elapsed_s": 3.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    failed.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "artifact_kind": "qi_seed_execution_summary",
                "case_count": 1,
                "resolution": {"NTHETA": 12, "NZETA": 24, "NX": 6, "NXI": 8},
                "execution_summary": {
                    "process_failed": 1,
                    "solver_traces_written": 0,
                    "timed_out": 1,
                },
                "gates": {"passed": False},
                "probe_env": {
                    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER": "1",
                    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_USE_IN_KRYLOV": "1",
                    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_OPERATOR_KRYLOV_ENRICHMENT": "1",
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = qi_seed.build_evidence_manifest(
        artifact_paths=[embedded, host_fallback, failed, host_fallback, tmp_path / "missing.json"],
        source_input=source_input,
        production_seed_count=3,
        production_timeout_s=120.0,
    )

    current = manifest["current_evidence"]
    assert current["artifact_count"] == 3
    assert current["passing_artifact_count"] == 2
    assert current["nonpassing_artifact_count"] == 1
    assert current["checked_backends"] == ["cpu", "gpu"]
    assert current["max_checked_total_size"] == 9602
    assert current["max_checked_active_size"] == 42
    assert current["largest_attempted_total_size"] == 13826
    assert current["largest_nonpassing_total_size"] == 13826
    assert current["failed_before_summary_json_count"] == 1
    assert current["evidence_class_counts"]["public_auto_or_legacy"] == 2
    assert current["evidence_class_counts"]["requested_operator_krylov_device_qi"] == 1
    assert current["evidence_tag_counts"]["failed_before_solver_trace_summary"] == 1
    assert current["evidence_tag_counts"]["requested_operator_krylov"] == 1
    assert (
        manifest["release_claims"]["production_non_autodiff_host_fallback"]["claim_status"]
        == "release_ready"
    )
    assert manifest["release_claims"]["true_device_qi"]["claim_status"] == "closed_deferred"
    assert manifest["production_target"]["seed_count"] == 3
    assert "--timeout-s 120.0" in manifest["regeneration_commands"]["production_cpu_seed_ladder"]
