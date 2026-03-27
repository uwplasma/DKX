from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_benchmark_case_variants_smoke(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    source = repo / "tests" / "reduced_inputs" / "tokamak_1species_PASCollisions_noEr_Nx1.input.namelist"
    case_dir = tmp_path / "case"
    json_out = tmp_path / "bench.json"
    case_dir.mkdir()
    (case_dir / "input.namelist").write_text(source.read_text())

    proc = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "benchmark_case_variants.py"),
            "--case-dir",
            str(case_dir),
            "--timeout-s",
            "120",
            "--json-out",
            str(json_out),
            "--variant",
            "incremental=SFINCS_JAX_RHSMODE1_SOLVE_METHOD=incremental",
            "--variant",
            "lgmres=SFINCS_JAX_RHSMODE1_SOLVE_METHOD=lgmres",
        ],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "## running default" in proc.stdout
    rows = json.loads(json_out.read_text())
    assert len(rows) == 3
    assert rows[0]["variant"] == "default"
    assert rows[1]["variant"] == "incremental"
    assert rows[2]["variant"] == "lgmres"
    assert rows[0]["status"] == "ok"
    assert rows[1]["status"] == "ok"
    assert rows[2]["status"] == "ok"
    assert rows[1]["vs_default"]["count"] == 0
    assert rows[2]["vs_default"]["count"] == 0
    assert not rows[1]["used_lgmres"]
    assert rows[2]["used_lgmres"]
