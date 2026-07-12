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


def _solve_request_paths(
    request: SolveInputs | str | Path,
    output_path: str | Path | None = None,
) -> tuple[Path, Path | None, Path | None, str | None, bool, Mapping[str, Any]]:
    if isinstance(request, SolveInputs):
        if request.input_path is None:
            raise ValueError("SolveInputs.input_path is required for this API call.")
        resolved_output = _path_or_none(output_path) if output_path is not None else request.output_path
        return (
            request.input_path,
            request.wout_path,
            resolved_output,
            request.backend,
            bool(request.requires_autodiff),
            request.options,
        )
    return Path(request), None, _path_or_none(output_path), None, False, MappingProxyType({})


def write_output(
    request: SolveInputs | str | Path,
    output_path: str | Path | None = None,
    **kwargs: Any,
) -> Path:
    """Run ``sfincs_jax`` from Python and write an output file.

    ``request`` may be a :class:`SolveInputs` object or an input namelist path.
    Routes to :func:`sfincs_jax.run.run_from_namelist` (the canonical RHSMode
    dispatch behind ``sfincs_jax write-output``).  The implementation imports
    the heavy run stack lazily so importing ``sfincs_jax.api`` stays cheap for
    docs, CLI validation, and downstream workflow planning.
    """

    input_path, wout_path, resolved_output, _backend, _requires_autodiff, options = _solve_request_paths(
        request,
        output_path,
    )
    if resolved_output is None:
        raise ValueError("output_path is required when SolveInputs.output_path is not set.")
    for key, value in options.items():
        kwargs.setdefault(str(key), value)
    wout_path = kwargs.pop("wout_path", wout_path)

    import contextlib  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    from .run import run_from_namelist  # noqa: PLC0415

    @contextlib.contextmanager
    def _namelist_with_override():
        if wout_path is None:
            yield Path(input_path)
            return
        from .input_compat import with_equilibrium_override  # noqa: PLC0415
        from .namelist import read_sfincs_input  # noqa: PLC0415

        nml = with_equilibrium_override(nml=read_sfincs_input(Path(input_path)), wout_path=wout_path)
        if nml.source_text is None:
            raise ValueError("wout_path overrides require a readable input.namelist source text.")
        tmp = tempfile.NamedTemporaryFile(
            "w",
            dir=str(Path(input_path).resolve().parent),
            prefix=f".{Path(input_path).stem}.override.",
            suffix=".namelist",
            delete=False,
            encoding="utf-8",
        )
        try:
            tmp.write(nml.source_text)
            tmp.close()
            yield Path(tmp.name)
        finally:
            Path(tmp.name).unlink(missing_ok=True)

    with _namelist_with_override() as namelist_path:
        run = run_from_namelist(
            namelist_path,
            out_path=resolved_output,
            **kwargs,
        )
    return Path(run.output_path)


def read_output(path: str | Path) -> dict[str, Any]:
    """Read an HDF5, NetCDF, or NPZ ``sfincs_jax`` output file."""

    from .io import read_sfincs_output_file  # noqa: PLC0415

    return read_sfincs_output_file(Path(path))


def run_ambipolar_brent(
    request: SolveInputs | str | Path,
    *,
    work_dir: str | Path | None = None,
    er_min: float,
    er_max: float,
    er_initial: float = 0.0,
    max_evaluations: int = 20,
    current_tolerance: float = 1.0e-10,
    step_tolerance: float = 1.0e-8,
    solve_method: str = "auto",
    differentiable: bool | None = None,
    reuse_output_geometry_cache: bool = True,
    reuse_solver_state: bool = True,
    emit: Any = None,
    **kwargs: Any,
) -> Any:
    """Run the canonical-stack Brent ambipolar ``E_r`` solve (stable public facade).

    Routes to :func:`sfincs_jax.er.find_ambipolar_er` (the canonical
    ``inputs -> drift_kinetic -> solve -> moments`` slice, replacing the legacy
    in-process Brent owner).  ``work_dir``, ``step_tolerance``,
    ``reuse_output_geometry_cache`` and ``reuse_solver_state`` are accepted for
    backwards compatibility; warm starts / recycling are threaded internally.

    Returns:
        A :class:`sfincs_jax.er.AmbipolarResult` (``.er`` is the ambipolar
        ``E_r``; ``.roots`` lists every classified root).
    """
    input_path, _wout_path, _output_path, _backend, _requires_autodiff, _options = (
        _solve_request_paths(request)
    )
    del work_dir, step_tolerance, differentiable
    del reuse_output_geometry_cache, reuse_solver_state, kwargs

    from .er import find_ambipolar_er  # noqa: PLC0415

    return find_ambipolar_er(
        input_path,
        er_bracket=(float(er_min), float(er_max)),
        er_initial=float(er_initial),
        max_iter=int(max_evaluations),
        current_tol=float(current_tolerance),
        solve_method=str(solve_method),
        emit=emit,
    )


def run_monoenergetic_database(
    request: SolveInputs | str | Path,
    nu_prime_grid: Sequence[float],
    e_star_grid: Sequence[float] = (0.0,),
    *,
    output_path: str | Path | None = None,
    solve_method: str = "auto",
    tol: float = 1.0e-10,
    emit: Any = None,
) -> Any:
    """Scan (nuPrime, EStar) monoenergetic coefficients (stable public facade).

    Routes to :func:`sfincs_jax.monoenergetic.monoenergetic_database` (the
    canonical RHSMode=3 database scan) and optionally writes the compact
    ``.npz`` database via :func:`sfincs_jax.monoenergetic.save_database`.
    The heavy stack is imported lazily so ``sfincs_jax.api`` stays cheap to
    import.

    Returns:
        The :class:`sfincs_jax.monoenergetic.MonoenergeticDatabase`.
    """

    input_path, _wout_path, resolved_output, _backend, _requires_autodiff, _options = (
        _solve_request_paths(request, output_path)
    )
    if input_path is None:
        raise ValueError("run_monoenergetic_database requires an input namelist path.")

    from .monoenergetic import monoenergetic_database, save_database  # noqa: PLC0415

    database = monoenergetic_database(
        input_path,
        nu_prime_grid,
        e_star_grid,
        solve_method=str(solve_method),
        tol=float(tol),
        emit=emit,
    )
    if resolved_output is not None:
        save_database(resolved_output, database)
    return database


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
    "read_output",
    "run_ambipolar_brent",
    "run_monoenergetic_database",
    "write_output",
]
