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

SCORECARD_CSV_FIELDS = (
    "case",
    "input_namelist",
    "source_artifact",
    "best_log_length",
    "best_matrix_relative_frobenius_error",
    "best_matrix_max_abs_error",
    "reference_n_x",
    "candidate_n_x",
    "reference_active_size",
    "best_active_size",
    "active_dof_reduction",
    "active_dof_reduction_fraction",
    "reference_total_size",
    "best_total_size",
    "total_dof_reduction",
    "total_dof_reduction_fraction",
    "candidate_max_residual_norm",
    "candidate_max_relative_residual_norm",
    "reference_max_residual_norm",
    "reference_max_relative_residual_norm",
    "residual_gate_pass",
    "mapped_classification",
    "mapped_useful",
    "mapped_negative",
    "solver_kinds",
    "solve_methods",
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


@dataclass(frozen=True)
class MappedTransportScorecardCase:
    """Reviewer-facing aggregate row for one checked evidence artifact."""

    case: str
    input_namelist: str | None
    source_artifact: str
    best_log_length: float
    best_matrix_relative_frobenius_error: float
    best_matrix_max_abs_error: float
    reference_n_x: int
    candidate_n_x: int
    reference_active_size: int
    best_active_size: int
    active_dof_reduction: int
    active_dof_reduction_fraction: float
    reference_total_size: int
    best_total_size: int
    total_dof_reduction: int
    total_dof_reduction_fraction: float
    candidate_max_residual_norm: float
    candidate_max_relative_residual_norm: float
    reference_max_residual_norm: float
    reference_max_relative_residual_norm: float
    residual_gate_pass: bool
    mapped_classification: str
    mapped_useful: bool
    mapped_negative: bool
    solver_kinds: tuple[str, ...]
    solve_methods: tuple[str, ...]


@dataclass(frozen=True)
class MappedTransportScorecard:
    """Aggregate scorecard over one or more mapped transport evidence artifacts."""

    cases: tuple[MappedTransportScorecardCase, ...]
    useful_error_gate: float
    negative_error_gate: float
    relative_residual_gate: float
    active_dof_reduction_gate: float

    @property
    def case_count(self) -> int:
        """Number of evidence artifacts included in the scorecard."""

        return len(self.cases)

    @property
    def useful_count(self) -> int:
        """Number of artifacts classified as useful mapped-grid evidence."""

        return sum(1 for case in self.cases if case.mapped_useful)

    @property
    def negative_count(self) -> int:
        """Number of artifacts classified as negative mapped-grid evidence."""

        return sum(1 for case in self.cases if case.mapped_negative)

    @property
    def mixed_count(self) -> int:
        """Number of artifacts that pass residual gates but are not decisive."""

        return self.case_count - self.useful_count - self.negative_count

    @property
    def best_case_by_error(self) -> MappedTransportScorecardCase:
        """Return the case with the smallest best-candidate transport error."""

        return min(self.cases, key=lambda case: case.best_matrix_relative_frobenius_error)


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

    best_by_moment = report.best_by_moment
    best_by_transport_error = report.best_by_transport_error
    payload = {
        "kind": "mapped_xgrid_transport_evidence",
        "reference_summary": asdict(report.reference_summary),
        "rows": [asdict(row) for row in report.rows],
        "best_by_moment_log_length": best_by_moment.log_length,
        "best_by_transport_error_log_length": best_by_transport_error.log_length,
        "best_by_moment": asdict(best_by_moment),
        "best_by_transport_error": asdict(best_by_transport_error),
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


def _finite_float_or_nan(value: object) -> float:
    if value is None:
        return float("nan")
    return float(np.asarray(value))


def _passes_relative_residual_gate(value: object, gate: float) -> bool:
    residual = _finite_float_or_nan(value)
    return bool(np.isfinite(residual) and residual <= float(gate))


def _scorecard_case_from_payload(
    payload: Mapping[str, Any],
    *,
    source_artifact: str,
    useful_error_gate: float,
    negative_error_gate: float,
    relative_residual_gate: float,
    active_dof_reduction_gate: float,
) -> MappedTransportScorecardCase:
    if payload.get("kind") != "mapped_xgrid_transport_evidence":
        raise ValueError(f"{source_artifact} is not a mapped transport evidence artifact")
    best = payload.get("best_by_transport_error")
    if best is None:
        rows = payload.get("rows")
        if not isinstance(rows, Sequence) or not rows:
            raise ValueError(f"{source_artifact} is missing candidate rows")
        best = min(
            (row for row in rows if isinstance(row, Mapping)),
            key=lambda row: float(row["matrix_relative_frobenius_error"]),
        )
    reference = payload.get("reference_summary")
    if not isinstance(best, Mapping) or not isinstance(reference, Mapping):
        raise ValueError(f"{source_artifact} is missing best/reference summary blocks")

    metadata = payload.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    case = str(metadata.get("case") or Path(source_artifact).stem)
    input_namelist_raw = metadata.get("input_namelist")
    input_namelist = None if input_namelist_raw is None else str(input_namelist_raw)

    reference_active_size = int(reference["active_size"])
    best_active_size = int(best["active_size"])
    reference_total_size = int(reference["total_size"])
    best_total_size = int(best["total_size"])
    active_dof_reduction = reference_active_size - best_active_size
    total_dof_reduction = reference_total_size - best_total_size
    active_dof_reduction_fraction = float(active_dof_reduction / max(reference_active_size, 1))
    total_dof_reduction_fraction = float(total_dof_reduction / max(reference_total_size, 1))

    best_error = float(best["matrix_relative_frobenius_error"])
    candidate_relative_residual = best.get("max_relative_residual_norm")
    reference_relative_residual = reference.get("max_relative_residual_norm")
    residual_gate_pass = (
        _passes_relative_residual_gate(candidate_relative_residual, relative_residual_gate)
        and _passes_relative_residual_gate(reference_relative_residual, relative_residual_gate)
    )
    reduction_pass = active_dof_reduction_fraction > float(active_dof_reduction_gate)
    mapped_useful = bool(
        residual_gate_pass and reduction_pass and best_error <= float(useful_error_gate)
    )
    mapped_negative = bool(
        (not residual_gate_pass)
        or (not reduction_pass)
        or best_error >= float(negative_error_gate)
    )
    if mapped_useful:
        classification = "useful"
    elif mapped_negative:
        classification = "negative"
    else:
        classification = "mixed"

    return MappedTransportScorecardCase(
        case=case,
        input_namelist=input_namelist,
        source_artifact=source_artifact,
        best_log_length=float(best["log_length"]),
        best_matrix_relative_frobenius_error=best_error,
        best_matrix_max_abs_error=float(best["matrix_max_abs_error"]),
        reference_n_x=int(reference["n_x"]),
        candidate_n_x=int(best["n_x"]),
        reference_active_size=reference_active_size,
        best_active_size=best_active_size,
        active_dof_reduction=active_dof_reduction,
        active_dof_reduction_fraction=active_dof_reduction_fraction,
        reference_total_size=reference_total_size,
        best_total_size=best_total_size,
        total_dof_reduction=total_dof_reduction,
        total_dof_reduction_fraction=total_dof_reduction_fraction,
        candidate_max_residual_norm=float(best["max_residual_norm"]),
        candidate_max_relative_residual_norm=_finite_float_or_nan(candidate_relative_residual),
        reference_max_residual_norm=float(reference["max_residual_norm"]),
        reference_max_relative_residual_norm=_finite_float_or_nan(reference_relative_residual),
        residual_gate_pass=residual_gate_pass,
        mapped_classification=classification,
        mapped_useful=mapped_useful,
        mapped_negative=mapped_negative,
        solver_kinds=tuple(str(value) for value in best.get("solver_kinds", ())),
        solve_methods=tuple(str(value) for value in best.get("solve_methods", ())),
    )


def build_transport_evidence_scorecard(
    artifact_paths: Sequence[str | Path],
    *,
    useful_error_gate: float = 0.1,
    negative_error_gate: float = 0.5,
    relative_residual_gate: float = 1.0e-8,
    active_dof_reduction_gate: float = 0.0,
) -> MappedTransportScorecard:
    """Aggregate checked mapped-grid evidence JSON artifacts into a scorecard.

    The classification is deliberately simple and auditable: a case is
    ``useful`` only if residual gates pass, active DOFs are reduced, and the
    best mapped candidate is below ``useful_error_gate``. A case is ``negative``
    if residuals fail, active DOFs do not decrease, or the best error exceeds
    ``negative_error_gate``. Cases in between are reported as ``mixed``.
    """

    paths = tuple(Path(path) for path in artifact_paths)
    if not paths:
        raise ValueError("artifact_paths must contain at least one evidence JSON artifact")
    if float(negative_error_gate) < float(useful_error_gate):
        raise ValueError("negative_error_gate must be greater than or equal to useful_error_gate")

    cases = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        cases.append(
            _scorecard_case_from_payload(
                payload,
                source_artifact=path.as_posix(),
                useful_error_gate=useful_error_gate,
                negative_error_gate=negative_error_gate,
                relative_residual_gate=relative_residual_gate,
                active_dof_reduction_gate=active_dof_reduction_gate,
            )
        )
    return MappedTransportScorecard(
        cases=tuple(cases),
        useful_error_gate=float(useful_error_gate),
        negative_error_gate=float(negative_error_gate),
        relative_residual_gate=float(relative_residual_gate),
        active_dof_reduction_gate=float(active_dof_reduction_gate),
    )


def transport_scorecard_to_dict(scorecard: MappedTransportScorecard) -> dict[str, Any]:
    """Return a JSON-serializable mapped transport scorecard payload."""

    best_case = scorecard.best_case_by_error
    payload = {
        "kind": "mapped_xgrid_transport_scorecard",
        "schema_version": 1,
        "thresholds": {
            "useful_error_gate": scorecard.useful_error_gate,
            "negative_error_gate": scorecard.negative_error_gate,
            "relative_residual_gate": scorecard.relative_residual_gate,
            "active_dof_reduction_gate": scorecard.active_dof_reduction_gate,
        },
        "summary": {
            "case_count": scorecard.case_count,
            "useful_count": scorecard.useful_count,
            "mixed_count": scorecard.mixed_count,
            "negative_count": scorecard.negative_count,
            "best_case_by_error": best_case.case,
            "best_matrix_relative_frobenius_error": best_case.best_matrix_relative_frobenius_error,
        },
        "cases": [asdict(case) for case in scorecard.cases],
    }
    return _json_scalar(payload)


def write_transport_scorecard_json(scorecard: MappedTransportScorecard, path: str | Path) -> None:
    """Write a stable JSON aggregate scorecard artifact."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(transport_scorecard_to_dict(scorecard), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_transport_scorecard_csv(scorecard: MappedTransportScorecard, path: str | Path) -> None:
    """Write aggregate scorecard cases as flat CSV."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCORECARD_CSV_FIELDS)
        writer.writeheader()
        for case in scorecard.cases:
            record = asdict(case)
            writer.writerow({key: _csv_scalar(record.get(key)) for key in SCORECARD_CSV_FIELDS})


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
