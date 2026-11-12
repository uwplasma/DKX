"""The shared ambipolar rung-gate audit: committed artifact and edge cases."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "audit_ambipolar_rung_gate", ROOT / "tools/paper_benchmarks/audit_ambipolar_rung_gate.py"
)
gate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gate)

ARTIFACT = ROOT / "validation/hsx_ambipolar_rung_gate_v1.json"


def _root(er: float, flux: float = 1.0) -> dict:
    return {"Er": er, "gamma": [flux, flux], "q": [flux, flux]}


def _artifact(rungs: dict, recorded: float = 0.0) -> dict:
    return {"rungs": rungs, "gate": {"max_relative_difference": recorded}}


def _rung(size: int, *roots: dict) -> dict:
    return {
        "resolution": {"theta": size, "zeta": 2, "pitch": 3, "speed": 4},
        "roots": {f"root:{i}:0": r for i, r in enumerate(roots)},
    }


def _run(tmp_path: Path, payload: dict) -> dict:
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return gate.audit(path)


def test_committed_artifact_passes() -> None:
    report = gate.audit(ARTIFACT)
    assert report["pass"] is True and report["rungs"] == ["A", "B"]


def test_tampered_artifact_file_fails(tmp_path: Path) -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    payload["rungs"]["B"]["roots"]["root:1:0"]["Er"] = 0.0
    report = _run(tmp_path, payload)
    assert report["pass"] is False and any("root:1:0" in e for e in report["errors"])


def test_relative_difference_is_symmetric_and_zero_safe() -> None:
    assert gate.rel(2.0, 1.0) == gate.rel(1.0, 2.0) == 0.5
    assert gate.rel(0.0, 0.0) == 0.0
    assert gate.rel(0.0, 1.0) == 1.0


def test_zero_roots_and_both_zero_pass(tmp_path: Path) -> None:
    payload = _artifact({"A": _rung(10, _root(0.0, 0.0)), "B": _rung(20, _root(0.0, 0.0))})
    assert _run(tmp_path, payload)["pass"] is True


def test_zero_against_nonzero_root_fails(tmp_path: Path) -> None:
    payload = _artifact({"A": _rung(10, _root(0.0)), "B": _rung(20, _root(0.5))})
    assert _run(tmp_path, payload)["pass"] is False


def test_rungs_are_ordered_by_grid_size_not_name(tmp_path: Path) -> None:
    rungs = {
        "A": _rung(30, _root(1.00)),
        "B": _rung(10, _root(1.10)),
        "C": _rung(20, _root(1.00)),
    }
    # By size B -> C -> A: 9.1% then 0%. By name A -> B -> C would give 9.1% then 9.1% too,
    # so the recorded maximum and the failing pair identify the order used.
    report = _run(tmp_path, _artifact(rungs, recorded=gate.rel(1.10, 1.00)))
    assert report["rungs"] == ["B", "C", "A"]
    assert report["pass"] is False and report["errors"] == [f"B->C root:0:0: {gate.rel(1.1, 1.0):.4f} > 0.02"]


def test_recorded_maximum_must_match(tmp_path: Path) -> None:
    payload = _artifact({"A": _rung(10, _root(1.0)), "B": _rung(20, _root(1.0))}, recorded=0.01)
    assert _run(tmp_path, payload)["pass"] is False


def test_root_sets_must_match(tmp_path: Path) -> None:
    payload = _artifact({"A": _rung(10, _root(1.0)), "B": _rung(20, _root(1.0), _root(2.0))})
    assert _run(tmp_path, payload)["pass"] is False


def test_geometry_checksum(tmp_path: Path) -> None:
    geometry = tmp_path / "wout.nc"
    geometry.write_bytes(b"vmec")
    payload = _artifact({"A": _rung(10, _root(1.0)), "B": _rung(20, _root(1.0))})
    payload["geometry_sha256"] = hashlib.sha256(b"vmec").hexdigest()
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert gate.audit(path, geometry)["geometry_checked"] is True
    geometry.write_bytes(b"other")
    report = gate.audit(path, geometry)
    assert report["pass"] is False and "geometry checksum mismatch" in report["errors"]
    payload["geometry_sha256"] = "xyz"
    path.write_text(json.dumps(copy.deepcopy(payload)), encoding="utf-8")
    assert gate.audit(path)["pass"] is False
