"""The W7-X FP ambipolar audit recomputes its accepted rung and is zero-safe."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "audit_w7x_fp_ambipolar_sfincs", ROOT / "tools/paper_benchmarks/audit_w7x_fp_ambipolar_sfincs.py"
)
audit_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit_module)

ARTIFACT = ROOT / "validation/w7x_fp_ambipolar_sfincs_v1.json"


def _run(tmp_path: Path, mutate) -> dict:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    mutate(payload)
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return audit_module.audit(path)


def test_committed_artifact_passes() -> None:
    assert audit_module.audit(ARTIFACT)["pass"] is True


def test_relative_difference_is_symmetric_and_zero_safe() -> None:
    assert audit_module._rel(2.0, 1.0) == audit_module._rel(1.0, 2.0) == 0.5
    assert audit_module._rel(0.0, 0.0) == 0.0


def test_accepted_rung_label_is_not_trusted(tmp_path: Path) -> None:
    def add_finer_rung(payload: dict) -> None:
        finer = json.loads(json.dumps(payload["rungs"]["13x31x24x6"]))
        finer["grid"] = [15, 37, 32, 7]
        finer["sfincs"]["Er"] *= 1.05  # the finest rung disagrees; the recorded label still names the coarse one
        payload["rungs"]["15x37x32x7"] = finer

    report = _run(tmp_path, add_finer_rung)
    assert report["pass"] is False
    assert any("not the finest rung" in e for e in report["errors"])
