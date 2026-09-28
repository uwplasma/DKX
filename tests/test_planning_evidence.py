"""Schema checks for the machine-readable DKX 3 planning evidence."""

from __future__ import annotations

from pathlib import Path

import tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_ROOT = REPO_ROOT / "validation"
CAPABILITY_STATUSES = {
    "stable",
    "stable_candidate",
    "validated_limited",
    "experimental",
    "compatibility_only",
    "deprecated",
}


def _load(name: str) -> dict[str, object]:
    with (VALIDATION_ROOT / name).open("rb") as stream:
        return tomllib.load(stream)


def test_capability_registry_has_unique_supported_statuses_and_evidence() -> None:
    payload = _load("capabilities.toml")
    assert set(payload["status_values"]) == CAPABILITY_STATUSES

    capabilities = payload["capability"]
    ids = [capability["id"] for capability in capabilities]
    assert len(ids) == len(set(ids))
    assert capabilities
    for capability in capabilities:
        assert capability["status"] in CAPABILITY_STATUSES
        assert capability["owners"]
        assert capability["evidence"]
        assert capability["gaps"]


def test_benchmark_schema_covers_performance_accuracy_and_failures() -> None:
    payload = _load("benchmark_schema.toml")
    required = set(payload["required_top_level"])
    assert {"timing_s", "memory_bytes", "accuracy", "outcome"} <= required
    assert {"failed", "timed_out", "refused_memory", "out_of_memory"} <= set(
        payload["outcome"]["allowed_status"]
    )


def test_hardware_registry_distinguishes_available_and_official_hosts() -> None:
    hosts = _load("hardware.toml")["host"]
    assert any(host["availability"] == "available" for host in hosts)
    assert any(host["official_release_baseline"] for host in hosts)
    assert all(host["id"] and host["role"] for host in hosts)
