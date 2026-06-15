"""Stable public data contracts for high-level SFINCS_JAX workflows.

The classes in this module describe orchestration boundaries: user inputs,
geometry/grid/operator summaries, solver metadata, output schemas, and
benchmark reports. They intentionally avoid importing JAX so they stay cheap to
import from the CLI, documentation, tests, and downstream workflow code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence


def _immutable_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


def _path_or_none(value: str | Path | None) -> Path | None:
    return None if value is None else Path(value)


@dataclass(frozen=True, slots=True)
class SolveInputs:
    """Normalized high-level solve request passed across API/CLI boundaries."""

    input_path: Path | None = None
    wout_path: Path | None = None
    output_path: Path | None = None
    backend: str | None = None
    requires_autodiff: bool = False
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_path", _path_or_none(self.input_path))
        object.__setattr__(self, "wout_path", _path_or_none(self.wout_path))
        object.__setattr__(self, "output_path", _path_or_none(self.output_path))
        object.__setattr__(self, "options", _immutable_mapping(self.options))


@dataclass(frozen=True, slots=True)
class GeometryState:
    """Geometry contract exposed to problem setup and validation layers."""

    kind: str
    source_path: Path | None = None
    radial_coordinate: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_path", _path_or_none(self.source_path))
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class GridState:
    """Phase-space grid sizes and layout metadata."""

    n_theta: int
    n_zeta: int
    n_xi: int
    n_x: int
    n_species: int = 1
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class OperatorState:
    """Operator summary shared by problem, solver, and diagnostics layers."""

    rhs_mode: int
    size: int
    collision_model: str
    include_phi1: bool = False
    matrix_free: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class PreconditionerState:
    """Selected preconditioner capability and setup metadata."""

    kind: str
    differentiable: bool
    device_safe: bool
    setup_memory_mb: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class SolverResult:
    """Backend-neutral solver summary for API-level callers."""

    residual_norm: float
    converged: bool
    iterations: int | None = None
    runtime_s: float | None = None
    solution: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class TransportResult:
    """Transport-output summary independent of a specific file format."""

    transport_matrix: Any = None
    particle_flux: Any = None
    heat_flux: Any = None
    bootstrap_current: Any = None
    solver: SolverResult | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class OutputSchema:
    """Versioned output-file schema and field-key contract."""

    format: str
    version: str
    keys: Sequence[str] = ()
    path: Path | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _path_or_none(self.path))
        object.__setattr__(self, "keys", tuple(self.keys))
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """Runtime, memory, and status summary for one benchmark case."""

    case: str
    backend: str
    runtime_s: float
    peak_memory_mb: float
    status: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))


__all__ = [
    "BenchmarkReport",
    "GeometryState",
    "GridState",
    "OperatorState",
    "OutputSchema",
    "PreconditionerState",
    "SolveInputs",
    "SolverResult",
    "TransportResult",
]
