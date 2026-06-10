from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from sfincs_jax.h5_parity import compare_h5_outputs, main


def _write_h5(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        for key, value in data.items():
            h5[key] = value


def test_compare_h5_outputs_records_full_numeric_status(tmp_path: Path) -> None:
    reference = tmp_path / "fortran.h5"
    candidate = tmp_path / "jax.h5"
    _write_h5(
        reference,
        {
            "ok_scalar": 1.0,
            "ok_vector": np.array([1.0, 2.0]),
            "value_mismatch": np.array([1.0, 2.0]),
            "missing": 3.0,
            "shape": np.array([1.0, 2.0, 3.0]),
            "text": b"metadata",
        },
    )
    _write_h5(
        candidate,
        {
            "ok_scalar": 1.0,
            "ok_vector": np.array([1.0, 2.0 + 1e-13]),
            "value_mismatch": np.array([1.0, 4.0]),
            "shape": np.array([[1.0, 2.0, 3.0]]),
            "extra": 5.0,
            "text": b"metadata",
        },
    )

    report = compare_h5_outputs(reference_path=reference, candidate_path=candidate, atol=1e-12, rtol=1e-12)
    by_key = {item["key"]: item for item in report["datasets"]}

    assert report["overall_status"] == "fail"
    assert report["status_counts"] == {
        "extra_in_candidate": 1,
        "missing_in_candidate": 1,
        "ok": 2,
        "shape_mismatch": 1,
        "value_mismatch": 1,
    }
    assert by_key["ok_scalar"]["status"] == "ok"
    assert by_key["ok_vector"]["status"] == "ok"
    assert by_key["value_mismatch"]["status"] == "value_mismatch"
    assert by_key["value_mismatch"]["max_abs"] == 2.0
    assert by_key["missing"]["status"] == "missing_in_candidate"
    assert by_key["shape"]["status"] == "shape_mismatch"
    assert by_key["extra"]["status"] == "extra_in_candidate"
    assert "text" not in by_key


def test_compare_h5_outputs_supports_keys_ignore_and_tolerances(tmp_path: Path) -> None:
    reference = tmp_path / "fortran.h5"
    candidate = tmp_path / "jax.h5"
    _write_h5(reference, {"a": 1.0, "b": 10.0, "ignored": 0.0})
    _write_h5(candidate, {"a": 1.0 + 1e-6, "b": 11.0, "ignored": 5.0})

    report = compare_h5_outputs(
        reference_path=reference,
        candidate_path=candidate,
        keys=["a", "ignored"],
        ignore_keys=["ignored"],
        tolerances={"a": {"atol": 1e-5, "rtol": 0.0}},
        include_extra=False,
    )

    assert report["overall_status"] == "pass"
    assert report["compared_dataset_count"] == 1
    assert report["datasets"][0]["key"] == "a"
    assert report["datasets"][0]["status"] == "ok"


def test_h5_parity_cli_writes_report_and_returns_failure(tmp_path: Path) -> None:
    reference = tmp_path / "fortran.h5"
    candidate = tmp_path / "jax.h5"
    out = tmp_path / "report.json"
    _write_h5(reference, {"a": 1.0})
    _write_h5(candidate, {"a": 2.0})

    rc = main([str(reference), str(candidate), "--out", str(out)])

    assert rc == 1
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["overall_status"] == "fail"
    assert payload["datasets"][0]["status"] == "value_mismatch"


def test_h5_parity_cli_passes_identical_outputs(tmp_path: Path) -> None:
    reference = tmp_path / "fortran.h5"
    candidate = tmp_path / "jax.h5"
    _write_h5(reference, {"a": np.array([1.0, 2.0])})
    _write_h5(candidate, {"a": np.array([1.0, 2.0])})

    assert main([str(reference), str(candidate), "--no-extra"]) == 0
