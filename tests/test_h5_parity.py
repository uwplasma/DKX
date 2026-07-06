from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from sfincs_jax.compare import (
    H5DatasetParity,
    _as_numpy,
    _center_fsa,
    _json_default,
    _merge_tolerance_floor,
    _numeric_datasets,
    compare_h5_outputs,
    main,
)


def _write_h5(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        for key, value in data.items():
            h5[key] = value


def test_compare_private_array_and_json_helpers_cover_metadata_edges(tmp_path: Path) -> None:
    assert _as_numpy(np.asarray([1.0])).shape == (1,)
    assert _as_numpy("metadata") is None
    assert _as_numpy(np.asarray("metadata")) is None
    assert _as_numpy([1.0, 2.0]) is None

    centered_4d = _center_fsa(np.arange(16.0).reshape(1, 2, 4, 2))
    np.testing.assert_allclose(centered_4d.mean(axis=(1, 2)), np.zeros((1, 2)))
    centered_3d_surface = _center_fsa(np.arange(12.0).reshape(2, 3, 2))
    np.testing.assert_allclose(centered_3d_surface.mean(axis=(0, 1)), np.zeros((2,)))
    centered_3d_history = _center_fsa(np.arange(6.0).reshape(2, 1, 3))
    np.testing.assert_allclose(centered_3d_history.mean(axis=1), np.zeros((2, 3)))

    tolerances = {"field": {"atol": "bad", "ignore": False}}
    _merge_tolerance_floor(tolerances, "field", {"atol": 1.0e-5, "rtol": "bad", "center_fsa": True})
    assert tolerances["field"]["atol"] == 1.0e-5
    assert tolerances["field"]["center_fsa"] is True
    assert "rtol" not in tolerances["field"]

    assert _json_default(np.float64(2.5)) == 2.5
    assert _json_default(tmp_path) == str(tmp_path)
    with pytest.raises(TypeError, match="not JSON serializable"):
        _json_default(object())


def test_numeric_datasets_ignores_groups_and_non_numeric_values(tmp_path: Path) -> None:
    path = tmp_path / "nested.h5"
    with h5py.File(path, "w") as h5:
        h5.create_group("metadata")
        h5["metadata/label"] = "case"
        h5["metadata/numeric"] = np.asarray([1.0, 2.0])
        h5["root_numeric"] = np.asarray(3.0)

    datasets = _numeric_datasets(path)

    assert set(datasets) == {"metadata/numeric", "root_numeric"}
    np.testing.assert_allclose(datasets["metadata/numeric"], np.asarray([1.0, 2.0]))


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


def test_compare_h5_outputs_records_missing_reference_and_file_errors(tmp_path: Path) -> None:
    reference = tmp_path / "fortran.h5"
    candidate = tmp_path / "jax.h5"
    _write_h5(reference, {"a": 1.0})
    _write_h5(candidate, {"a": 1.0, "candidate_only": 2.0})

    report = compare_h5_outputs(
        reference_path=reference,
        candidate_path=candidate,
        keys=["a", "candidate_only", "missing_everywhere"],
        include_extra=False,
    )
    by_key = {item["key"]: item for item in report["datasets"]}

    assert report["overall_status"] == "fail"
    assert by_key["a"]["status"] == "ok"
    assert by_key["candidate_only"]["status"] == "missing_in_reference"
    assert by_key["missing_everywhere"]["status"] == "missing_in_reference"

    with pytest.raises(FileNotFoundError):
        compare_h5_outputs(reference_path=tmp_path / "missing.h5", candidate_path=candidate)


def test_h5_dataset_parity_json_and_ok_contracts() -> None:
    extra = H5DatasetParity(
        key="candidate_only",
        status="extra_in_candidate",
        reference_shape=None,
        candidate_shape=(2,),
        max_abs=None,
        max_rel=None,
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    non_numeric = H5DatasetParity(
        key="label",
        status="non_numeric",
        reference_shape=(),
        candidate_shape=(),
        max_abs=None,
        max_rel=None,
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    failing = H5DatasetParity(
        key="bad",
        status="missing_in_candidate",
        reference_shape=(1,),
        candidate_shape=None,
        max_abs=None,
        max_rel=None,
        atol=1.0e-12,
        rtol=1.0e-12,
    )

    assert extra.ok
    assert non_numeric.ok
    assert not failing.ok
    assert extra.to_json()["candidate_shape"] == [2]
    assert failing.to_json()["candidate_shape"] is None


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


def test_h5_parity_cli_prints_success_report_to_stdout(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    reference = tmp_path / "fortran.h5"
    candidate = tmp_path / "jax.h5"
    _write_h5(reference, {"a": np.array([1.0, 2.0])})
    _write_h5(candidate, {"a": np.array([1.0, 2.0])})

    assert main([str(reference), str(candidate), "--key", "a", "--ignore-key", "ignored"]) == 0

    captured = capsys.readouterr()
    assert '"overall_status": "pass"' in captured.out
