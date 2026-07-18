"""Versioned solver-trace schema and JSON/HDF5 (de)serialization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SolverTraceCandidate:
    """Recorded decision for one candidate solver/preconditioner path."""

    name: str
    accepted: bool
    reasons: tuple[str, ...] = ()
    residual_ratio: float | None = None
    runtime_ratio: float | None = None
    memory_ratio: float | None = None
    memory_metric: str | None = None
    active_rss_mb: float | None = None
    device_peak_mb: float | None = None
    compiled_temp_mb: float | None = None
    candidate_setup_s: float | None = None
    candidate_solve_s: float | None = None


@dataclass(frozen=True)
class SolverTrace:
    """Portable summary of one `dkx` solve-path decision."""

    backend: str
    rhs_mode: int
    selected_path: str
    schema_version: int = SCHEMA_VERSION
    solve_method: str | None = None
    preconditioner: str | None = None
    geometry_scheme: int | None = None
    collision_operator: str | None = None
    total_size: int | None = None
    active_size: int | None = None
    device_count: int | None = None
    cold_jit: bool | None = None
    residual_norm: float | None = None
    residual_target: float | None = None
    converged: bool | None = None
    elapsed_s: float | None = None
    setup_s: float | None = None
    solve_s: float | None = None
    peak_rss_mb: float | None = None
    active_rss_mb: float | None = None
    device_peak_mb: float | None = None
    compiled_temp_mb: float | None = None
    estimated_dense_nbytes: int | None = None
    estimated_csr_nbytes: int | None = None
    estimated_gmres_basis_nbytes: int | None = None
    matvec_count: int | None = None
    candidate_decisions: tuple[SolverTraceCandidate, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        record = asdict(self)
        record["candidate_decisions"] = [asdict(item) for item in self.candidate_decisions]
        return record

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SolverTrace":
        """Build a trace from a dictionary and validate the schema version."""
        schema_version = int(data.get("schema_version", SCHEMA_VERSION))
        if schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported solver trace schema_version={schema_version}")
        candidates = tuple(
            SolverTraceCandidate(
                name=str(item["name"]),
                accepted=bool(item["accepted"]),
                reasons=tuple(str(reason) for reason in item.get("reasons", ())),
                residual_ratio=item.get("residual_ratio"),
                runtime_ratio=item.get("runtime_ratio"),
                memory_ratio=item.get("memory_ratio"),
                memory_metric=item.get("memory_metric"),
                active_rss_mb=item.get("active_rss_mb"),
                device_peak_mb=item.get("device_peak_mb"),
                compiled_temp_mb=item.get("compiled_temp_mb"),
                candidate_setup_s=item.get("candidate_setup_s"),
                candidate_solve_s=item.get("candidate_solve_s"),
            )
            for item in data.get("candidate_decisions", ())
        )
        return cls(
            schema_version=schema_version,
            backend=str(data["backend"]),
            rhs_mode=int(data["rhs_mode"]),
            selected_path=str(data["selected_path"]),
            solve_method=data.get("solve_method"),
            preconditioner=data.get("preconditioner"),
            geometry_scheme=data.get("geometry_scheme"),
            collision_operator=data.get("collision_operator"),
            total_size=data.get("total_size"),
            active_size=data.get("active_size"),
            device_count=data.get("device_count"),
            cold_jit=data.get("cold_jit"),
            residual_norm=data.get("residual_norm"),
            residual_target=data.get("residual_target"),
            converged=data.get("converged"),
            elapsed_s=data.get("elapsed_s"),
            setup_s=data.get("setup_s"),
            solve_s=data.get("solve_s"),
            peak_rss_mb=data.get("peak_rss_mb"),
            active_rss_mb=data.get("active_rss_mb"),
            device_peak_mb=data.get("device_peak_mb"),
            compiled_temp_mb=data.get("compiled_temp_mb"),
            estimated_dense_nbytes=data.get("estimated_dense_nbytes"),
            estimated_csr_nbytes=data.get("estimated_csr_nbytes"),
            estimated_gmres_basis_nbytes=data.get("estimated_gmres_basis_nbytes"),
            matvec_count=data.get("matvec_count"),
            candidate_decisions=candidates,
            metadata=data.get("metadata", {}),
        )

    def to_json(self) -> str:
        """Return a stable JSON representation."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, payload: str | bytes) -> "SolverTrace":
        """Build a trace from a JSON payload."""
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        return cls.from_dict(json.loads(payload))


def write_solver_trace_json(path: str | Path, trace: SolverTrace) -> None:
    """Write a solver trace JSON sidecar."""
    Path(path).write_text(trace.to_json() + "\n", encoding="utf-8")


def read_solver_trace_json(path: str | Path) -> SolverTrace:
    """Read a solver trace JSON sidecar."""
    return SolverTrace.from_json(Path(path).read_text(encoding="utf-8"))


def write_solver_trace_h5(group: Any, trace: SolverTrace) -> None:
    """Write a solver trace into an HDF5 file/group as a JSON attribute."""
    trace_group = group.require_group("solver_trace")
    trace_group.attrs["schema_version"] = int(SCHEMA_VERSION)
    trace_group.attrs["json"] = trace.to_json()


def read_solver_trace_h5(group: Any) -> SolverTrace:
    """Read a solver trace from an HDF5 file/group written by this module."""
    trace_group = group["solver_trace"]
    payload = trace_group.attrs["json"]
    return SolverTrace.from_json(payload)


# Compact Fortran-v3 and DKX solver profile comparisons.
