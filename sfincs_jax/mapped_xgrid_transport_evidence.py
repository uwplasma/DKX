"""Solve-facing evidence helpers for opt-in mapped SFINCS speed grids.

The routines here deliberately sit one level above the differentiable moment
objective in :mod:`sfincs_jax.mapped_xgrid_objectives`. Moment matching is a
cheap screening proxy; this module compares mapped-grid candidates against an
actual SFINCS-v3 transport-matrix solve result. The current scope is the PAS
transport path, because the full-FP collision precompute still has additional
scheme assumptions that are not yet mapped-grid compatible.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .adaptive_maps import MappedXGrid
from .mapped_xgrid_objectives import rational_tail_transport_grid, transport_moment_report
from .namelist import Namelist
from .v3_driver import V3TransportMatrixSolveResult, solve_v3_transport_matrix_linear_gmres


SolveFn = Callable[..., V3TransportMatrixSolveResult]


EVIDENCE_CSV_FIELDS = (
    "log_length",
    "moment_objective",
    "moment_loss",
    "regularization_loss",
    "matrix_relative_frobenius_error",
    "matrix_max_abs_error",
    "max_residual_norm",
    "max_relative_residual_norm",
    "total_elapsed_time_s",
    "total_size",
    "active_size",
    "active_fraction",
    "n_x",
    "use_active_dof_mode",
    "solver_kinds",
    "solve_methods",
    "min_dx",
    "width_ratio",
    "smoothness",
    "jac_roughness",
    "tail_mass_proxy",
    "reference_total_size",
    "reference_active_size",
    "reference_active_fraction",
    "reference_n_x",
    "reference_max_residual_norm",
    "reference_max_relative_residual_norm",
    "reference_total_elapsed_time_s",
)


@dataclass(frozen=True)
class TransportSolveSummary:
    """Compact diagnostics extracted from a transport-matrix solve."""

    max_residual_norm: float
    max_relative_residual_norm: float
    total_elapsed_time_s: float
    total_size: int
    active_size: int
    active_fraction: float
    n_x: int
    use_active_dof_mode: bool | None
    solver_kinds: tuple[str, ...]
    solve_methods: tuple[str, ...]


@dataclass(frozen=True)
class TransportMatrixError:
    """Relative and absolute differences between transport matrices."""

    relative_frobenius: float
    max_abs: float
    reference_norm: float


@dataclass(frozen=True)
class MappedTransportEvidenceRow:
    """One candidate mapped-grid comparison against a reference solve."""

    log_length: float
    moment_objective: float
    moment_loss: float
    regularization_loss: float
    matrix_relative_frobenius_error: float
    matrix_max_abs_error: float
    max_residual_norm: float
    max_relative_residual_norm: float
    total_elapsed_time_s: float
    total_size: int
    active_size: int
    active_fraction: float
    n_x: int
    use_active_dof_mode: bool | None
    solver_kinds: tuple[str, ...]
    solve_methods: tuple[str, ...]
    min_dx: float
    width_ratio: float
    smoothness: float
    jac_roughness: float
    tail_mass_proxy: float


@dataclass(frozen=True)
class MappedTransportEvidenceReport:
    """Comparison report for a log-length scan of rational-tail maps."""

    reference_summary: TransportSolveSummary
    rows: tuple[MappedTransportEvidenceRow, ...]

    @property
    def best_by_moment(self) -> MappedTransportEvidenceRow:
        """Return the candidate with the smallest proxy moment objective."""

        return min(self.rows, key=lambda row: row.moment_objective)

    @property
    def best_by_transport_error(self) -> MappedTransportEvidenceRow:
        """Return the candidate closest to the reference transport matrix."""

        return min(self.rows, key=lambda row: row.matrix_relative_frobenius_error)


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_scalar(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_scalar(item) for item in value]
    return value


def _csv_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return ""
    if isinstance(value, (list, tuple)):
        return ";".join(str(item) for item in value)
    if value is None:
        return ""
    return value


def transport_evidence_report_to_dict(
    report: MappedTransportEvidenceReport,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a JSON-serializable mapped transport evidence payload."""

    payload = {
        "kind": "mapped_xgrid_transport_evidence",
        "reference_summary": asdict(report.reference_summary),
        "rows": [asdict(row) for row in report.rows],
        "best_by_moment_log_length": report.best_by_moment.log_length,
        "best_by_transport_error_log_length": report.best_by_transport_error.log_length,
        "metadata": dict(metadata or {}),
    }
    return _json_scalar(payload)


def write_transport_evidence_json(
    report: MappedTransportEvidenceReport,
    path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Write mapped transport evidence as a stable JSON artifact."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = transport_evidence_report_to_dict(report, metadata=metadata)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_transport_evidence_csv(report: MappedTransportEvidenceReport, path: str | Path) -> None:
    """Write candidate mapped-grid evidence rows as a flat CSV artifact."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    reference = report.reference_summary
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVIDENCE_CSV_FIELDS)
        writer.writeheader()
        for row in report.rows:
            record = asdict(row)
            record.update(
                {
                    "reference_total_size": reference.total_size,
                    "reference_active_size": reference.active_size,
                    "reference_active_fraction": reference.active_fraction,
                    "reference_n_x": reference.n_x,
                    "reference_max_residual_norm": reference.max_residual_norm,
                    "reference_max_relative_residual_norm": reference.max_relative_residual_norm,
                    "reference_total_elapsed_time_s": reference.total_elapsed_time_s,
                }
            )
            writer.writerow({key: _csv_scalar(record.get(key)) for key in EVIDENCE_CSV_FIELDS})


def _copy_indexed(indexed: Mapping[str, Mapping[str, Mapping[tuple[int, ...], object]]]) -> dict:
    return {
        group_name: {key: dict(values) for key, values in group.items()}
        for group_name, group in indexed.items()
    }


def _casefold_get(mapping: Mapping[str, object], key: str, default: object) -> object:
    key_lower = key.lower()
    for existing_key, value in mapping.items():
        if existing_key.lower() == key_lower:
            return value
    return default


def copy_namelist_with_mapped_xgrid(
    nml: Namelist,
    *,
    log_length: float,
    family: str = "rational_tail",
    eta_kind: str = "gauss",
    derivative: str = "barycentric",
    eps: float | None = 1.0e-6,
    x_grid_scheme: int = 50,
    extra_options: Mapping[str, object] | None = None,
) -> Namelist:
    """Return a namelist copy with the opt-in mapped speed-grid keys set."""

    groups = {group_name: dict(values) for group_name, values in nml.groups.items()}
    other = dict(groups.get("othernumericalparameters", {}))
    other.update(
        {
            "XGRIDSCHEME": int(x_grid_scheme),
            "MAPPEDXGRIDFAMILY": str(family),
            "MAPPEDXGRIDLOGLENGTH": float(log_length),
            "MAPPEDXGRIDETAKIND": str(eta_kind),
            "MAPPEDXGRIDDERIVATIVE": str(derivative),
        }
    )
    if eps is not None:
        other["MAPPEDXGRIDEPS"] = float(eps)
    if extra_options is not None:
        other.update(dict(extra_options))
    groups["othernumericalparameters"] = other
    return Namelist(
        groups=groups,
        indexed=_copy_indexed(nml.indexed),
        source_path=nml.source_path,
        source_text=nml.source_text,
    )


def copy_namelist_with_resolution(
    nml: Namelist,
    *,
    nx: int | None = None,
    nxi: int | None = None,
    nl: int | None = None,
    ntheta: int | None = None,
    nzeta: int | None = None,
) -> Namelist:
    """Return a namelist copy with selected resolution parameters replaced."""

    groups = {group_name: dict(values) for group_name, values in nml.groups.items()}
    resolution = dict(groups.get("resolutionparameters", {}))
    replacements = {
        "NX": nx,
        "NXI": nxi,
        "NL": nl,
        "NTHETA": ntheta,
        "NZETA": nzeta,
    }
    for key, value in replacements.items():
        if value is not None:
            resolution[key] = int(value)
    groups["resolutionparameters"] = resolution
    return Namelist(
        groups=groups,
        indexed=_copy_indexed(nml.indexed),
        source_path=nml.source_path,
        source_text=nml.source_text,
    )


def transport_solve_summary(result: V3TransportMatrixSolveResult) -> TransportSolveSummary:
    """Extract residual, size, and timing diagnostics from a transport solve."""

    residuals = np.asarray(
        [float(np.asarray(value)) for value in result.residual_norms_by_rhs.values()],
        dtype=np.float64,
    )
    if residuals.size == 0:
        max_residual = 0.0
    else:
        max_residual = float(np.max(residuals))

    max_relative = np.nan
    rhs_norms_by_rhs = getattr(result, "rhs_norms_by_rhs", None)
    if rhs_norms_by_rhs:
        relatives = []
        for rhs, residual in result.residual_norms_by_rhs.items():
            rhs_norm = rhs_norms_by_rhs.get(rhs)
            if rhs_norm is None:
                continue
            denom = max(float(np.asarray(rhs_norm)), 1.0e-300)
            relatives.append(float(np.asarray(residual)) / denom)
        if relatives:
            max_relative = float(np.max(np.asarray(relatives, dtype=np.float64)))

    elapsed = float(np.sum(np.asarray(result.elapsed_time_s, dtype=np.float64)))
    total_size = int(getattr(result.op0, "total_size", np.asarray(result.transport_matrix).shape[0]))
    active_size_raw = getattr(result, "active_size", None)
    active_size = total_size if active_size_raw is None else int(active_size_raw)
    active_fraction = float(active_size / max(total_size, 1))
    n_x = int(getattr(result.op0, "n_x", 0))
    solver_kinds_by_rhs = getattr(result, "solver_kinds_by_rhs", None) or {}
    solve_methods_by_rhs = getattr(result, "solve_methods_by_rhs", None) or {}
    solver_kinds = tuple(sorted({str(value) for value in solver_kinds_by_rhs.values()}))
    solve_methods = tuple(sorted({str(value) for value in solve_methods_by_rhs.values()}))
    use_active = getattr(result, "use_active_dof_mode", None)
    use_active_dof_mode = None if use_active is None else bool(use_active)
    return TransportSolveSummary(
        max_residual_norm=max_residual,
        max_relative_residual_norm=max_relative,
        total_elapsed_time_s=elapsed,
        total_size=total_size,
        active_size=active_size,
        active_fraction=active_fraction,
        n_x=n_x,
        use_active_dof_mode=use_active_dof_mode,
        solver_kinds=solver_kinds,
        solve_methods=solve_methods,
    )


def transport_matrix_error(
    candidate: V3TransportMatrixSolveResult,
    reference: V3TransportMatrixSolveResult,
    *,
    floor: float = 1.0e-300,
) -> TransportMatrixError:
    """Return matrix-level error diagnostics against a reference solve."""

    cand = np.asarray(candidate.transport_matrix, dtype=np.float64)
    ref = np.asarray(reference.transport_matrix, dtype=np.float64)
    if cand.shape != ref.shape:
        raise ValueError("candidate and reference transport matrices must have the same shape")
    diff = cand - ref
    ref_norm = float(np.linalg.norm(ref))
    return TransportMatrixError(
        relative_frobenius=float(np.linalg.norm(diff) / max(ref_norm, floor)),
        max_abs=float(np.max(np.abs(diff))) if diff.size else 0.0,
        reference_norm=ref_norm,
    )


def _grid_diagnostics(grid: MappedXGrid) -> dict[str, float]:
    return {key: float(np.asarray(value)) for key, value in grid.regularization.items()}


def _resolution_nx(nml: Namelist) -> int:
    resolution = nml.group("resolutionParameters")
    return int(_casefold_get(resolution, "Nx", 0))


def _candidate_row(
    *,
    log_length: float,
    grid: MappedXGrid,
    result: V3TransportMatrixSolveResult,
    reference: V3TransportMatrixSolveResult,
    powers: Sequence[float],
    regularization_weights: Mapping[str, float] | None,
) -> MappedTransportEvidenceRow:
    moment = transport_moment_report(
        grid,
        powers=powers,
        regularization_weights=regularization_weights,
    )
    error = transport_matrix_error(result, reference)
    summary = transport_solve_summary(result)
    diag = _grid_diagnostics(grid)
    return MappedTransportEvidenceRow(
        log_length=float(log_length),
        moment_objective=float(np.asarray(moment.objective)),
        moment_loss=float(np.asarray(moment.moment_loss)),
        regularization_loss=float(np.asarray(moment.regularization_loss)),
        matrix_relative_frobenius_error=error.relative_frobenius,
        matrix_max_abs_error=error.max_abs,
        max_residual_norm=summary.max_residual_norm,
        max_relative_residual_norm=summary.max_relative_residual_norm,
        total_elapsed_time_s=summary.total_elapsed_time_s,
        total_size=summary.total_size,
        active_size=summary.active_size,
        active_fraction=summary.active_fraction,
        n_x=summary.n_x,
        use_active_dof_mode=summary.use_active_dof_mode,
        solver_kinds=summary.solver_kinds,
        solve_methods=summary.solve_methods,
        min_dx=diag["min_dx"],
        width_ratio=diag["width_ratio"],
        smoothness=diag["smoothness"],
        jac_roughness=diag["jac_roughness"],
        tail_mass_proxy=diag["tail_mass_proxy"],
    )


def run_rational_tail_transport_comparison(
    nml: Namelist,
    *,
    log_length_values: Sequence[float],
    reference_nml: Namelist | None = None,
    reference_result: V3TransportMatrixSolveResult | None = None,
    solve_fn: SolveFn = solve_v3_transport_matrix_linear_gmres,
    powers: Sequence[float] = (2.0, 4.0, 6.0),
    regularization_weights: Mapping[str, float] | None = None,
    eta_kind: str = "gauss",
    derivative: str = "barycentric",
    eps: float = 1.0e-6,
    solve_kwargs: Mapping[str, object] | None = None,
) -> MappedTransportEvidenceReport:
    """Run a PAS transport-matrix comparison over rational-tail log lengths.

    ``reference_result`` is normally a solve on ``reference_nml`` or the
    original namelist, while each candidate uses ``xGridScheme = 50`` and the
    supplied ``log_length``. Tests can pass a lightweight ``solve_fn`` double;
    production runs should use the default SFINCS-JAX transport solver.
    """

    values = tuple(float(value) for value in log_length_values)
    if not values:
        raise ValueError("log_length_values must contain at least one candidate")
    kwargs = dict(solve_kwargs or {})
    reference_input = nml if reference_nml is None else reference_nml
    reference = reference_result if reference_result is not None else solve_fn(nml=reference_input, **kwargs)
    nx = _resolution_nx(nml)
    if nx < 2:
        raise ValueError("the input namelist must specify resolutionParameters/Nx >= 2")

    rows: list[MappedTransportEvidenceRow] = []
    for log_length in values:
        candidate_nml = copy_namelist_with_mapped_xgrid(
            nml,
            log_length=log_length,
            family="rational_tail",
            eta_kind=eta_kind,
            derivative=derivative,
            eps=eps,
        )
        grid = rational_tail_transport_grid(
            nx,
            log_length,
            eta_kind=eta_kind,
            eps=eps,
            derivative=derivative,
        )
        result = solve_fn(nml=candidate_nml, **kwargs)
        rows.append(
            _candidate_row(
                log_length=log_length,
                grid=grid,
                result=result,
                reference=reference,
                powers=powers,
                regularization_weights=regularization_weights,
            )
        )

    return MappedTransportEvidenceReport(
        reference_summary=transport_solve_summary(reference),
        rows=tuple(rows),
    )
