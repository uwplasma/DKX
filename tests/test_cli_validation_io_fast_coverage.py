from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pytest

from sfincs_jax import cli
from sfincs_jax import plotting
from sfincs_jax.benchmark_artifact_policy import (
    check_benchmark_artifact_file,
    check_benchmark_artifact_files,
    fortran_suite_benchmark_summary_errors,
)
from sfincs_jax.validation_figures import (
    build_simakov_helander_high_nu_panel,
    build_w7x_ambipolar_root_provenance_panel,
)


def _write_small_h5(path: Path, data: dict[str, object]) -> None:
    with h5py.File(path, "w") as h5:
        for key, value in data.items():
            h5[key] = np.asarray(value)


def test_normalize_plot_shortcut_preserves_global_flags() -> None:
    argv = [
        "--cores=4",
        "--quiet",
        "--transport-workers",
        "2",
        "--plot",
        "sfincsOutput.npz",
        "--out",
        "summary.pdf",
    ]

    assert cli._normalize_default_argv(argv) == [
        "--cores=4",
        "--quiet",
        "--transport-workers",
        "2",
        "plot-output",
        "--input-h5",
        "sfincsOutput.npz",
        "--out",
        "summary.pdf",
    ]


def test_cmd_plot_output_uses_sfincsoutput_default_pdf(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, Path] = {}

    def fake_plot_sfincs_output_summary(*, input_h5: Path, output_png: Path) -> Path:
        calls["input_h5"] = Path(input_h5)
        calls["output_png"] = Path(output_png)
        output_png.parent.mkdir(parents=True, exist_ok=True)
        output_png.write_bytes(b"%PDF-1.4\n")
        return output_png.resolve()

    monkeypatch.setattr(
        "sfincs_jax.plotting.plot_sfincs_output_summary",
        fake_plot_sfincs_output_summary,
    )

    input_h5 = tmp_path / "case.sfincsOutput.h5"
    rc = cli._cmd_plot_output(
        Namespace(input_h5=str(input_h5), out=None, verbose=0, quiet=True)
    )

    assert rc == 0
    assert calls["input_h5"] == input_h5
    assert calls["output_png"] == tmp_path / "case_summary.pdf"
    assert (tmp_path / "case_summary.pdf").exists()


def test_cmd_dump_h5_keys_only_and_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "sfincsOutput.h5"
    _write_small_h5(
        source,
        {
            "BHat": np.arange(4.0).reshape(2, 2),
            "Ntheta": np.asarray(2, dtype=np.int32),
        },
    )

    assert cli._cmd_dump_h5(
        Namespace(sfincs_output=str(source), out_json=str(tmp_path / "unused.json"), keys_only=True)
    ) == 0
    assert capsys.readouterr().out.splitlines() == ["BHat", "Ntheta"]

    out_json = tmp_path / "dump.json"
    assert cli._cmd_dump_h5(
        Namespace(sfincs_output=str(source), out_json=str(out_json), keys_only=False)
    ) == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["Ntheta"] == 2
    assert payload["BHat"] == [[0.0, 1.0], [2.0, 3.0]]


def test_cmd_ambipolar_summary_records_solver_state_reuse(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from sfincs_jax.problems.ambipolar import AmbipolarIteration, AmbipolarResult, SfincsJaxEvaluationRecord

    input_path = tmp_path / "input.namelist"
    input_path.write_text("&general\n RHSMode = 1\n/\n", encoding="utf-8")
    out_dir = tmp_path / "ambipolar"
    state_path = out_dir / ".sfincs_jax_solver_state" / "rhsmode1_state.npz"

    def fake_solve_sfincs_jax_ambipolar_brent(**kwargs):
        assert kwargs["reuse_output_geometry_cache"] is False
        assert kwargs["reuse_solver_state"] is True
        result = AmbipolarResult(
            converged=True,
            method="brent",
            root_er=0.0,
            root_radial_current=0.0,
            iterations=(
                AmbipolarIteration(index=1, er=0.0, radial_current=0.0, stage="initial"),
            ),
            status="converged",
            root_type="ion",
        )
        record = SfincsJaxEvaluationRecord(
            er=0.0,
            radial_current=0.0,
            input_path=out_dir / "eval_001" / "input.namelist",
            output_path=out_dir / "eval_001" / "sfincsOutput.h5",
            solver_trace_path=out_dir / "eval_001" / "sfincsOutput.solver_trace.json",
            selected_path="rhsmode1_solution",
            solve_method="auto",
            residual_norm=0.0,
            residual_target=1.0e-10,
            converged=True,
            total_size=12,
            active_size=10,
            cache_enabled=False,
            solver_state_reuse_enabled=True,
            solver_state_path=state_path,
            solver_state_input_exists=True,
            solver_state_output_exists=True,
        )
        return result, type("FakeEvaluator", (), {"records": [record]})()

    monkeypatch.setattr(
        "sfincs_jax.problems.ambipolar.solve_sfincs_jax_ambipolar_brent",
        fake_solve_sfincs_jax_ambipolar_brent,
    )

    summary_path = tmp_path / "summary.json"
    rc = cli._cmd_ambipolar(
        Namespace(
            input=str(input_path),
            out_dir=str(out_dir),
            er_min=-1.0,
            er_max=1.0,
            er_initial=0.0,
            max_evaluations=4,
            current_tolerance=1.0e-10,
            step_tolerance=1.0e-8,
            solve_method="auto",
            summary_json=str(summary_path),
            no_output_cache=True,
            no_solver_state=False,
            verbose=0,
            quiet=True,
        )
    )

    assert rc == 0
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["converged"] is True
    assert payload["evaluations"][0]["cache_enabled"] is False
    assert payload["evaluations"][0]["solver_state_reuse_enabled"] is True
    assert payload["evaluations"][0]["solver_state_input_exists"] is True
    assert payload["evaluations"][0]["solver_state_output_exists"] is True
    assert payload["evaluations"][0]["solver_state_path"] == str(state_path)


def test_cmd_compare_h5_honors_tolerance_json(tmp_path: Path) -> None:
    left = tmp_path / "left.h5"
    right = tmp_path / "right.h5"
    tolerances = tmp_path / "tolerances.json"
    _write_small_h5(left, {"transportMatrix": np.asarray([1.0, 2.0])})
    _write_small_h5(right, {"transportMatrix": np.asarray([1.0, 2.1])})
    tolerances.write_text(
        json.dumps({"transportMatrix": {"rtol": 0.0, "atol": 0.2}}),
        encoding="utf-8",
    )

    rc = cli._cmd_compare_h5(
        Namespace(
            a=str(left),
            b=str(right),
            rtol="0.0",
            atol="0.0",
            tolerances_json=str(tolerances),
            show_all=False,
        )
    )

    assert rc == 0


def test_apply_parallel_runtime_settings_shard_axis_off_and_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("SFINCS_JAX_SHARD", "SFINCS_JAX_AUTO_SHARD", "SFINCS_JAX_MATVEC_SHARD_AXIS"):
        monkeypatch.delenv(name, raising=False)

    base = dict(
        transport_workers=None,
        distributed_gmres=None,
        distributed_krylov=None,
        shard_pad=None,
        distributed=False,
        process_id=None,
        process_count=None,
        coordinator_address=None,
        coordinator_port=None,
    )
    cli._apply_parallel_runtime_settings(Namespace(**base, shard_axis="off"))
    assert os_environ_subset() == {
        "SFINCS_JAX_AUTO_SHARD": "0",
        "SFINCS_JAX_MATVEC_SHARD_AXIS": "off",
        "SFINCS_JAX_SHARD": "0",
    }

    cli._apply_parallel_runtime_settings(Namespace(**base, shard_axis="auto"))
    assert os_environ_subset() == {
        "SFINCS_JAX_AUTO_SHARD": "1",
        "SFINCS_JAX_MATVEC_SHARD_AXIS": "auto",
        "SFINCS_JAX_SHARD": "1",
    }


def os_environ_subset() -> dict[str, str]:
    import os

    return {
        key: os.environ[key]
        for key in ("SFINCS_JAX_AUTO_SHARD", "SFINCS_JAX_MATVEC_SHARD_AXIS", "SFINCS_JAX_SHARD")
    }


def test_plotting_shape_helpers_and_missing_panel_paths() -> None:
    assert plotting._select_x_profile(np.arange(24).reshape(2, 3, 4)).tolist() == [0, 12]
    assert plotting._surface(np.arange(3.0)).shape == (1, 3)
    assert plotting._surface(np.arange(24.0).reshape(2, 3, 4)).shape == (2, 3)
    assert plotting._matrix(np.asarray(5.0)).shape == (1, 1)
    assert plotting._matrix(np.arange(3.0)).shape == (3, 1)

    fig, axes = plt.subplots(1, 2)
    try:
        assert plotting._add_heatmap(axes[0], {}, "missing", title="Missing") is False
        assert plotting._add_profile(
            axes[1],
            {"flow": np.asarray([[1.0, 2.0], [3.0, 4.0]])},
            np.asarray([0.0]),
            "flow",
        ) is True
        assert axes[1].get_xlabel() == "index"
    finally:
        plt.close(fig)


def test_plot_sfincs_output_summary_accepts_minimal_npz(tmp_path: Path) -> None:
    source = tmp_path / "minimal.sfincsOutput.npz"
    np.savez(
        source,
        x=np.asarray([0.0, 0.5, 1.0]),
        theta=np.asarray([0.0, 1.0]),
        zeta=np.asarray([0.0, 0.25, 0.5]),
        BHat=np.asarray([[1.0, 1.1, 1.2], [0.9, 1.0, 1.1]]),
        geometryScheme=np.asarray(5),
        RHSMode=np.asarray(1),
        Ntheta=np.asarray(2),
        Nzeta=np.asarray(3),
        Nx=np.asarray(3),
    )

    png = plotting.plot_sfincs_output_summary(input_h5=source, output_png=tmp_path / "summary.png")
    pdf = plotting.plot_sfincs_output_summary(input_h5=source, output_png=tmp_path / "summary.pdf")

    assert png.exists() and png.stat().st_size > 1000
    assert pdf.exists() and pdf.stat().st_size > 1000


def test_validation_figure_guards_reject_duplicate_scan_points() -> None:
    with pytest.raises(ValueError, match="Er scan points must be distinct"):
        build_w7x_ambipolar_root_provenance_panel(
            {
                "runs": [
                    {"er": 0.0, "radial_current": -1.0},
                    {"er": 0.0, "radial_current": 1.0},
                ],
                "ambipolar": {"roots_er": [0.0]},
            }
        )

    with pytest.raises(ValueError, match="nuprime scan points must be distinct"):
        build_simakov_helander_high_nu_panel(
            {
                "runs": [
                    {"nuprime": 10.0, "value": 1.1, "analytic_limit": 1.0},
                    {"nuprime": 10.0, "value": 1.05, "analytic_limit": 1.0},
                ]
            }
        )


def test_simakov_helander_tail_fit_requires_two_points() -> None:
    with pytest.raises(ValueError, match="n_tail_fit must be at least 2"):
        build_simakov_helander_high_nu_panel(
            {
                "runs": [
                    {"nuprime": 10.0, "value": 1.1, "analytic_limit": 1.0},
                    {"nuprime": 100.0, "value": 1.01, "analytic_limit": 1.0},
                ]
            },
            n_tail_fit=1,
        )


def test_benchmark_artifact_file_helpers_report_io_and_json_errors(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")

    missing_errors = check_benchmark_artifact_file(missing)
    aggregate_errors = check_benchmark_artifact_files([missing, malformed])

    assert len(missing_errors) == 1
    assert f"{missing}: could not read JSON file:" in missing_errors[0]
    assert len(aggregate_errors) == 2
    assert f"{missing}: could not read JSON file:" in aggregate_errors[0]
    assert f"{malformed}: invalid JSON:" in aggregate_errors[1]


def test_fortran_suite_summary_rejects_warm_source_count_mismatch() -> None:
    payload = {
        "metadata": {
            "kind": "fortran_v3_suite_benchmark_summary",
            "min_fortran_runtime_s": 10.0,
            "reported_case_counts": {"cpu": 1, "gpu": 1},
            "excluded_low_fortran_runtime_cases": [],
        },
        "reports": {
            backend: {
                "total_cases": 1,
                "parity_ok_cases": 1,
                "strict_mismatch_total": 0,
                "cold_runtime_ratio_summary": {"count": 1},
                "warm_or_logged_runtime_ratio_summary": {"count": 1},
                "active_memory_ratio_summary": {"count": 1},
                "warm_or_logged_runtime_source_counts": {"jax_runtime_s_warm": 0},
                "fastest_jax_vs_fortran_cases": [],
                "slowest_jax_vs_fortran_cases": [],
                "highest_active_jax_memory_cases": [],
            }
            for backend in ("cpu", "gpu")
        },
    }

    errors = fortran_suite_benchmark_summary_errors(payload)

    assert "field reports.cpu.warm_or_logged_runtime_source_counts must sum to total_cases" in errors
    assert "field reports.gpu.warm_or_logged_runtime_source_counts must sum to total_cases" in errors
