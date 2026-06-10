from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def _load_module():
    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "audit_rhs1_solver_stack_under_test",
        repo / "scripts" / "audit_rhs1_solver_stack.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_rhs1_input(path: Path) -> None:
    path.write_text("&general\n  RHSMode = 1\n/\n", encoding="utf-8")


def test_parse_audit_text_extracts_setup_preflight_and_gmres_lines() -> None:
    module = _load_module()
    text = "\n".join(
        [
            "profiling: rhs1_setup_start dt_s=0.0 total_s=1.0 rss_mb=100.0",
            "solve_v3_full_system_linear_gmres: building RHSMode=1 preconditioner=active_low_l_schur",
            "active_fortran_v3_pc_matrix: factor setup elapsed_s=3.25",
            "support_mode_preflight candidate=ilu rejected budget gate",
            "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres iters=20 ksp_residual=3.0e-4 elapsed_s=12.5",
            "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres matvecs=1980 elapsed_s=473.939",
            "ksp_iterations=7 solver=gmres",
            "solve_v3_full_system_structured_csr: converged=True residual=3.711e-13 solve_s=42.492",
            "solve_v3_full_system_structured_csr: pc_kind=active_fortran_v3_reduced_lu pc_selected=True pc_reason=complete pc_setup_s=35.125 pc_factor_nbytes=3003511928 pc_permc=NATURAL pc_superlu_permc=NATURAL",
            "solve_v3_full_system_linear_gmres: fortran_reduced direct-tail structured preconditioner selected kind=active_fortran_v3_reduced_lu setup_s=279.469 elapsed_s=279.681 reason=complete cache_hit=False factor_nbytes=13303259384 permc=RCM superlu_permc=NATURAL",
            "profiling: rhs1_setup_done dt_s=4.0 total_s=5.0 rss_mb=250.0",
        ]
    )

    parsed = module.parse_audit_text(text)

    assert parsed["last_rhs1_preconditioner"] == "active_low_l_schur"
    assert any("factor setup" in row["text"] for row in parsed["setup_lines"])
    assert any("support_mode_preflight" in row["text"] for row in parsed["preflight_lines"])
    assert any("xblock_sparse_pc_gmres" in row["text"] for row in parsed["gmres_lines"])
    assert {"solver": "gmres", "iterations": 7} in parsed["ksp_iterations"]
    assert {"solver": "xblock_sparse_pc_gmres", "iterations": 20} in parsed["ksp_iterations"]
    assert parsed["gmres_matvecs"] == [{"solver": "xblock_sparse_pc_gmres", "matvecs": 1980}]
    assert parsed["last_structured_solve"] == {
        "converged": True,
        "residual": 3.711e-13,
        "solve_s": 42.492,
    }
    assert parsed["last_structured_preconditioner"] == {
        "kind": "active_fortran_v3_reduced_lu",
        "selected": True,
        "reason": "complete",
        "setup_s": 35.125,
        "factor_nbytes": 3003511928.0,
        "permc_spec": "NATURAL",
        "superlu_permc_spec": "NATURAL",
    }
    assert parsed["last_direct_tail_preconditioner"] == {
        "kind": "active_fortran_v3_reduced_lu",
        "setup_s": 279.469,
        "elapsed_s": 279.681,
        "reason": "complete",
        "cache_hit": False,
        "factor_nbytes": 13303259384.0,
        "permc_spec": "RCM",
        "superlu_permc_spec": "NATURAL",
    }
    assert parsed["profile_stage_durations_s"]["rhs1_setup"] == 4.0
    assert parsed["profile_peak_rss_mb"] == 250.0


def test_main_writes_json_with_mocked_subprocess(monkeypatch, tmp_path: Path) -> None:
    module = _load_module()
    input_path = tmp_path / "qa.input.namelist"
    _write_rhs1_input(input_path)
    out_path = tmp_path / "audit.json"
    calls: list[dict[str, object]] = []

    def fake_run(command, **kwargs):
        calls.append({"command": command, "kwargs": kwargs})
        stdout = "\n".join(
            [
                "solve_v3_full_system_linear_gmres: building RHSMode=1 preconditioner=active_low_l_schur",
                "preflight: active low-l gate passed",
                "solve_v3_full_system_linear_gmres: active_low_l_schur_gmres iters=3 ksp_residual=1.0e-8 elapsed_s=0.2",
                "ksp_iterations=3 solver=gmres",
            ]
        )
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    rc = module.main(
        [
            "--case",
            f"QA={input_path}",
            "--preconditioner",
            "active_low_l_schur",
            "--out",
            str(out_path),
            "--work-dir",
            str(tmp_path / "work"),
            "--timeout-s",
            "5",
            "--restart",
            "4",
            "--maxiter",
            "2",
        ]
    )

    assert rc == 0
    assert len(calls) == 1
    kwargs = calls[0]["kwargs"]
    assert kwargs["timeout"] == 5.0
    assert kwargs["env"]["SFINCS_JAX_RHSMODE1_PRECONDITIONER"] == "active_low_l_schur"
    assert kwargs["env"]["SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_MAXITER"] == "2"
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["kind"] == "rhs1_solver_stack_audit"
    assert payload["summary"]["status_counts"] == {"ok": 1}
    row = payload["rows"][0]
    assert row["case"] == "QA"
    assert row["returncode"] == 0
    assert row["captures"]["last_rhs1_preconditioner"] == "active_low_l_schur"
    assert row["captures"]["ksp_iterations"] == [
        {"solver": "gmres", "iterations": 3},
        {"solver": "active_low_l_schur_gmres", "iterations": 3},
    ]


def test_timeout_result_keeps_partial_capture(tmp_path: Path) -> None:
    module = _load_module()
    input_path = tmp_path / "qh.input.namelist"
    _write_rhs1_input(input_path)
    args = module._build_parser().parse_args(
        [
            "--case",
            f"QH={input_path}",
            "--out",
            str(tmp_path / "audit.json"),
            "--work-dir",
            str(tmp_path / "work"),
            "--timeout-s",
            "3",
        ]
    )
    case = module.AuditCase(label="QH", input_path=input_path)

    def fake_timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=command,
            timeout=kwargs["timeout"],
            output=b"preflight: candidate still running\n",
            stderr=b"solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres matvecs=40\n",
        )

    row = module._run_one(
        repo=Path(__file__).resolve().parents[1],
        case=case,
        preconditioner="auto",
        args=args,
        run_fn=fake_timeout,
    )

    assert row["status"] == "timeout"
    assert row["returncode"] is None
    assert row["timed_out"] is True
    assert "candidate still running" in row["stdout_tail"]
    assert row["captures"]["gmres_matvecs"] == [{"solver": "xblock_sparse_pc_gmres", "matvecs": 40}]


def test_dry_run_writes_planned_rows_without_subprocess(tmp_path: Path) -> None:
    module = _load_module()
    input_path = tmp_path / "qa.input.namelist"
    _write_rhs1_input(input_path)
    out_path = tmp_path / "audit.json"

    rc = module.main(
        [
            "--qa-input",
            str(input_path),
            "--preconditioner",
            "theta_schwarz",
            "--preconditioner",
            "zeta_schwarz",
            "--out",
            str(out_path),
            "--dry-run",
        ]
    )

    assert rc == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["summary"]["status_counts"] == {"planned": 2}
    assert [row["preconditioner"] for row in payload["rows"]] == ["theta_schwarz", "zeta_schwarz"]
