from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from sfincs_jax.io import write_sfincs_h5
from sfincs_jax.optimization_promotion import evaluate_sfincs_scan_promotion


_REPO = Path(__file__).resolve().parents[1]


def _write_scan_point(
    scan_dir: Path,
    *,
    er: float,
    current: float,
    residual: float | None = 1.0e-10,
) -> None:
    run_dir = scan_dir / f"Er{er:.4g}"
    run_dir.mkdir(parents=True, exist_ok=True)
    z_s = np.asarray([1.0, -1.0, 6.0], dtype=np.float64)
    gamma_i = 0.02 * er
    gamma_e = 0.10 + 0.01 * er
    gamma_z = (current - gamma_i + gamma_e) / 6.0
    data = {
        "Er": np.asarray(er),
        "Nspecies": np.asarray(3, dtype=np.int32),
        "Zs": z_s,
        "includePhi1": np.asarray(0, dtype=np.int32),
        "particleFlux_vm_rHat": np.asarray([[gamma_i], [gamma_e], [gamma_z]], dtype=np.float64),
        "heatFlux_vm_rHat": np.asarray([[0.03], [0.02], [0.004]], dtype=np.float64),
        "FSABjHatOverRootFSAB2": np.asarray([0.04 + 0.015 * er], dtype=np.float64),
    }
    if residual is not None:
        data["linearSolverResidualNorm"] = np.asarray(residual, dtype=np.float64)
        data["linearSolverResidualTarget"] = np.asarray(1.0e-8, dtype=np.float64)
    write_sfincs_h5(path=run_dir / "sfincsOutput.h5", data=data, overwrite=True, fortran_layout=False)


def test_evaluate_sfincs_scan_promotion_passes_bracketed_electron_root(tmp_path: Path) -> None:
    scan = tmp_path / "scan"
    for er, current in [(-3.0, -2.0), (-1.0, -0.8), (1.0, 0.35), (3.0, 1.9)]:
        _write_scan_point(scan, er=er, current=current)

    summary = evaluate_sfincs_scan_promotion(
        scan,
        require_electron_root=True,
        impurity_species_index=2,
        target_impurity_flux=0.01,
    )
    payload = summary.as_dict()

    assert payload["gate_status"] == "pass"
    assert payload["selected_root"]["root_type"] == "electron"
    assert payload["flux_objective"]["mean_impurity_flux"] > 0.0
    assert payload["bootstrap_objective"] > 0.0


def test_evaluate_sfincs_scan_promotion_fails_unbracketed_or_bad_residual(tmp_path: Path) -> None:
    scan = tmp_path / "scan"
    for er, current in [(-1.0, 1.0), (1.0, 2.0)]:
        _write_scan_point(scan, er=er, current=current, residual=1.0e-4)

    summary = evaluate_sfincs_scan_promotion(scan, require_electron_root=True)
    payload = summary.as_dict()

    assert payload["gate_status"] == "fail"
    assert any("not bracketed" in failure for failure in payload["failures"])
    assert any("residual gate" in failure for failure in payload["failures"])


def test_evaluate_sfincs_scan_promotion_can_allow_reference_outputs_without_residuals(
    tmp_path: Path,
) -> None:
    scan = tmp_path / "fortran_like_scan"
    for er, current in [(-3.0, -2.0), (-1.0, -0.8), (1.0, 0.35), (3.0, 1.9)]:
        _write_scan_point(scan, er=er, current=current, residual=None)

    strict = evaluate_sfincs_scan_promotion(scan, require_electron_root=True)
    relaxed = evaluate_sfincs_scan_promotion(
        scan,
        require_electron_root=True,
        require_residuals=False,
    )

    assert strict.as_dict()["gate_status"] == "fail"
    assert any("missing linear residual" in failure for failure in strict.as_dict()["failures"])
    assert relaxed.as_dict()["gate_status"] == "pass"


def test_public_promotion_example_runs_demo(tmp_path: Path) -> None:
    script = _REPO / "examples" / "optimization" / "evaluate_sfincs_jax_promotion_scan.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--out-dir",
            str(tmp_path),
            "--stem",
            "promotion_test",
            "--impurity-species-index",
            "2",
        ],
        cwd=_REPO,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads((tmp_path / "promotion_test.json").read_text(encoding="utf-8"))
    assert payload["workflow"] == "sfincs_jax_optimization_high_fidelity_promotion"
    assert payload["gate_status"] == "pass"
    assert (tmp_path / "promotion_test.png").exists()
    assert (tmp_path / "promotion_test.pdf").exists()
