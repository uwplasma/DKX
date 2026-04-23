from __future__ import annotations

import json
from pathlib import Path


_VALID_STATUSES = {"implemented", "prototype_artifact", "needs_reaudit", "planned"}
_VALID_KINDS = {
    "literature_reproduction",
    "literature_validation",
    "profile_validation",
    "cross_code_validation",
    "autodiff_validation",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _manifest() -> list[dict[str, object]]:
    path = _repo_root() / "examples" / "publication_figures" / "validation_manifest.json"
    payload = json.loads(path.read_text())
    assert isinstance(payload, list)
    return payload


def _first_command_path(command: str) -> Path:
    return _repo_root() / command.split()[0]


def test_validation_manifest_records_have_research_gate_schema() -> None:
    records = _manifest()
    ids = [str(record["id"]) for record in records]
    assert len(ids) == len(set(ids))
    for record in records:
        assert str(record["status"]) in _VALID_STATUSES
        assert str(record["kind"]) in _VALID_KINDS
        for key in ("literature", "claims", "source_code", "tests", "acceptance_gates"):
            assert key in record, f"{record['id']} missing {key}"
            assert isinstance(record[key], list), f"{record['id']} {key} must be a list"
            assert record[key], f"{record['id']} {key} must not be empty"
        assert all(str(url).startswith("http") for url in record["literature"])
        assert all(str(gate).strip() for gate in record["acceptance_gates"])


def test_validation_manifest_paths_exist_for_nonplanned_lanes() -> None:
    for record in _manifest():
        status = str(record["status"])
        if status == "planned":
            continue
        for command in record.get("scripts", []):
            assert _first_command_path(str(command)).exists(), record["id"]
        for artifact in record.get("artifacts", []):
            assert (_repo_root() / str(artifact)).exists(), f"{record['id']} missing artifact {artifact}"
        for source_path in record["source_code"]:
            assert (_repo_root() / str(source_path)).exists(), f"{record['id']} missing source {source_path}"
        for test_path in record["tests"]:
            assert (_repo_root() / str(test_path)).exists(), f"{record['id']} missing test {test_path}"


def test_validation_manifest_keeps_open_lanes_explicit() -> None:
    planned_or_reaudit = {
        str(record["id"])
        for record in _manifest()
        if str(record["status"]) in {"planned", "needs_reaudit"}
    }
    assert "sfincs2014_fig1_lhd_collisionality" not in planned_or_reaudit
    assert "sfincs2014_fig2_w7x_collisionality" not in planned_or_reaudit
    assert "sfincs2014_fig3_high_collisionality_limit" in planned_or_reaudit
    assert "w7x_ambipolar_er_validation" in planned_or_reaudit
    assert "monkes_monoenergetic_overlap" in planned_or_reaudit
    assert "adjoint_sensitivity_gradient_checks" in planned_or_reaudit
