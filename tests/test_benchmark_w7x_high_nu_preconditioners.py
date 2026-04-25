from __future__ import annotations

import json
from pathlib import Path

from examples.performance import benchmark_w7x_high_nu_preconditioners as bench


def test_write_w7x_high_nu_input_sets_fp_nu_and_reduced_resolution(tmp_path: Path) -> None:
    path = bench.write_w7x_high_nu_input(
        destination=tmp_path / "input.namelist",
        nuprime=10.0,
        collision_operator=0,
        reduced_resolution=True,
    )
    text = path.read_text()
    assert "nu_n = 1.727145650000e+00" in text
    assert "collisionOperator = 0" in text
    assert "Ntheta = 5" in text
    assert "Nzeta = 7" in text
    assert "!ss scanType" not in text


def test_candidate_environment_records_fp_tzfft_controls() -> None:
    env = bench.candidate_environment(
        preconditioner="fp_tzfft",
        sparse_direct_max=30000,
        sparse_factor_dtype="float32",
        maxiter=800,
        fp_tzfft_max_mb=256.0,
    )
    assert env["SFINCS_JAX_TRANSPORT_PRECOND"] == "fp_tzfft"
    assert env["SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_MAX"] == "30000"
    assert env["SFINCS_JAX_TRANSPORT_SPARSE_FACTOR_DTYPE"] == "float32"
    assert env["SFINCS_JAX_TRANSPORT_MAXITER"] == "800"
    assert env["SFINCS_JAX_TRANSPORT_FP_TZFFT_MAX_MB"] == "256"
    assert env["SFINCS_JAX_IMPLICIT_SOLVE"] == "0"


def test_dry_run_writes_bounded_candidate_summary(tmp_path: Path, capsys) -> None:
    summary = tmp_path / "summary.json"
    rc = bench.main(
        [
            "--work-dir",
            str(tmp_path),
            "--summary",
            str(summary),
            "--preconditioners",
            "auto,fp_tzfft",
            "--which-rhs",
            "2",
            "--reduced-resolution",
            "--sparse-factor-dtype",
            "float32",
            "--dry-run",
        ]
    )
    assert rc == 0
    payload = json.loads(summary.read_text())
    assert payload["status"] == "dry_run"
    assert payload["which_rhs"] == [2]
    assert payload["preconditioners"] == ["auto", "fp_tzfft"]
    assert Path(payload["input"]).exists()
    assert payload["commands"][1]["environment"]["SFINCS_JAX_TRANSPORT_PRECOND"] == "fp_tzfft"
    assert payload["commands"][1]["environment"]["SFINCS_JAX_TRANSPORT_SPARSE_FACTOR_DTYPE"] == "float32"
    assert "fp_tzfft" in capsys.readouterr().out
