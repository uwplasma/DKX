"""Native physical-unit ambipolar scans and bracket refinement."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np


_MAX_RETAINED_EVALUATIONS = 100_000


@dataclass(frozen=True)
class RootEvaluation:
    """One solved electric-field point retained for root evidence."""

    electric_field_kv_m: float
    radial_current_a_m2: float
    particle_flux_m2_s: np.ndarray
    heat_flux_w_m2: np.ndarray
    parallel_current_a_t_m2: float
    residual_norm: float
    stage: str
    reason: str = "initial_uniform_grid"
    refinement_level: int = 0


@dataclass(frozen=True)
class NativeAmbipolarRoot:
    """One sign-bracketed and physically re-evaluated ambipolar root."""

    electric_field_kv_m: float
    radial_current_a_m2: float
    slope_a_m2_per_kv_m: float
    root_type: str
    bracket_kv_m: tuple[float, float]
    evaluation: RootEvaluation
    movement_kv_m: float = np.nan
    observable_relative_movement: float = np.nan
    branch_id: str = ""


@dataclass(frozen=True)
class BranchEvent:
    """One radially localized branch-continuation event."""

    kind: str
    branch_ids: tuple[str, ...]
    root_indices: tuple[int, ...]
    electric_field_kv_m: float
    detail: str
    nonsmooth: bool = True


@dataclass(frozen=True)
class RefinementEvidence:
    """One deterministic adaptive-search rung retained in the result."""

    level: int
    search_evaluations: int
    total_evaluations: int
    root_count: int
    root_movement_kv_m: float
    observable_relative_movement: float
    max_bracket_width_kv_m: float
    converged: bool


@dataclass(frozen=True)
class NativeAmbipolarSurface:
    """All scan evidence and roots on one flux surface."""

    evaluations: tuple[RootEvaluation, ...]
    roots: tuple[NativeAmbipolarRoot, ...]
    selected_root: int | None
    selected: RootEvaluation
    status: str
    solve_seconds: float
    batch_chunk_size: int
    batch_chunks: int
    refinement: tuple[RefinementEvidence, ...] = ()
    refinement_status: str = "not_requested"
    evaluation_budget: int = 0
    branch_events: tuple[BranchEvent, ...] = ()
    selected_branch_id: str = ""
    selection_reason: str = "not_assigned"


def _classify_root(electric_field_kv_m: float, slope: float) -> str:
    if slope < 0.0:
        return "unstable"
    return "electron" if electric_field_kv_m > 0.0 else "ion"


def _brackets(fields: np.ndarray, currents: np.ndarray) -> list[tuple[int, int]]:
    brackets: list[tuple[int, int]] = []
    for index in range(fields.size - 1):
        left = float(currents[index])
        right = float(currents[index + 1])
        if left == 0.0:
            brackets.append((index, index))
        elif left * right < 0.0:
            brackets.append((index, index + 1))
    if currents[-1] == 0.0:
        brackets.append((fields.size - 1, fields.size - 1))
    return brackets


def _evaluation_budget(
    *,
    search_points: int,
    max_root_iterations: int,
    find_all_roots: bool,
    convergence_enabled: bool,
    max_refinements: int,
) -> tuple[int, int]:
    """Return final hierarchy size and a conservative retained-solve bound."""

    refinements = int(max_refinements) if convergence_enabled else 0
    # With the schema's minimum three search points, level 16 already contains
    # 131073 hierarchy points. Reject it before constructing an arbitrarily
    # large Python integer from an untrusted configuration value.
    if refinements > 15:
        raise ValueError(
            "native ambipolar refinement preflight exceeds 100000 retained "
            "evaluations; reduce convergence.max_refinements"
        )
    levels = refinements + 1
    hierarchy_points = (
        (int(search_points) - 1) * (2**refinements) + 1
        if convergence_enabled
        else int(search_points)
    )
    brackets_per_level = hierarchy_points if find_all_roots else 1
    budget = hierarchy_points + (levels * brackets_per_level * int(max_root_iterations))
    return hierarchy_points, budget


def _observable(evaluation: RootEvaluation, name: str) -> np.ndarray:
    aliases = {
        "electric_field": evaluation.electric_field_kv_m,
        "particle_flux": evaluation.particle_flux_m2_s,
        "heat_flux": evaluation.heat_flux_w_m2,
        "parallel_current": evaluation.parallel_current_a_t_m2,
        "bootstrap_current": evaluation.parallel_current_a_t_m2,
        "radial_current": evaluation.radial_current_a_m2,
    }
    return np.asarray(aliases[name], dtype=np.float64)


def _relative_movement(current: np.ndarray, previous: np.ndarray) -> float:
    scale = np.maximum(np.maximum(np.abs(current), np.abs(previous)), 1.0e-300)
    return float(np.max(np.abs(current - previous) / scale))


def _compare_roots(
    roots: list[NativeAmbipolarRoot],
    previous: tuple[NativeAmbipolarRoot, ...] | None,
    observables: tuple[str, ...],
) -> tuple[list[NativeAmbipolarRoot], float, float]:
    if previous is None or len(roots) != len(previous) or not roots:
        return roots, np.nan, np.nan
    compared: list[NativeAmbipolarRoot] = []
    root_movements: list[float] = []
    observable_movements: list[float] = []
    for root, old_root in zip(roots, previous):
        root_movement = abs(root.electric_field_kv_m - old_root.electric_field_kv_m)
        requested = [
            _relative_movement(
                _observable(root.evaluation, name),
                _observable(old_root.evaluation, name),
            )
            for name in observables
            if name != "electric_field"
        ]
        observable_movement = max(requested, default=0.0)
        compared.append(
            replace(
                root,
                movement_kv_m=float(root_movement),
                observable_relative_movement=float(observable_movement),
            )
        )
        root_movements.append(float(root_movement))
        observable_movements.append(float(observable_movement))
    return compared, max(root_movements), max(observable_movements)


def _branch_prefix(root_type: str) -> str:
    return root_type if root_type in {"ion", "electron", "unstable"} else "branch"


def _predict_branch(history: list[tuple[float, float]], surface: float) -> float:
    if len(history) < 2:
        return history[-1][1]
    previous_surface, previous_field = history[-2]
    last_surface, last_field = history[-1]
    radial_step = last_surface - previous_surface
    if radial_step == 0.0:
        return last_field
    return last_field + (last_field - previous_field) * (
        (surface - last_surface) / radial_step
    )


def continue_ambipolar_branches(
    surface_results: list[NativeAmbipolarSurface] | tuple[NativeAmbipolarSurface, ...],
    *,
    surfaces: np.ndarray,
    electric_field_bounds_kv_m: tuple[float, float],
    continue_selection: bool,
) -> tuple[NativeAmbipolarSurface, ...]:
    """Assign radial branch identity and retain discrete event evidence.

    Matching uses linear prediction from each branch's two most recent points
    and a global minimum-cost assignment. A match is admitted only within one
    quarter of the declared electric-field search span. Events are discrete
    profile evidence; they do not claim a continuously resolved bifurcation.
    """

    from scipy.optimize import linear_sum_assignment  # noqa: PLC0415

    surface_values = np.asarray(surfaces, dtype=np.float64).reshape((-1,))
    if len(surface_results) != surface_values.size:
        raise ValueError("branch continuation needs one result per radial surface")
    if not np.all(np.isfinite(surface_values)) or np.any(
        np.diff(surface_values) <= 0.0
    ):
        raise ValueError(
            "branch continuation needs finite, strictly increasing surfaces"
        )
    span = float(electric_field_bounds_kv_m[1] - electric_field_bounds_kv_m[0])
    if not np.isfinite(span) or span <= 0.0:
        raise ValueError(
            "branch continuation needs increasing finite electric-field bounds"
        )
    gate = 0.25 * span
    counters: dict[str, int] = {"ion": 0, "electron": 0, "unstable": 0, "branch": 0}
    histories: dict[str, list[tuple[float, float]]] = {}
    tracked: list[NativeAmbipolarSurface] = []
    previous_roots: tuple[NativeAmbipolarRoot, ...] = ()

    def new_branch(root: NativeAmbipolarRoot) -> str:
        prefix = _branch_prefix(root.root_type)
        identifier = f"{prefix}-{counters[prefix]:03d}"
        counters[prefix] += 1
        return identifier

    for surface_index, (surface, raw_result) in enumerate(
        zip(surface_values, surface_results)
    ):
        roots = list(raw_result.roots)
        events: list[BranchEvent] = []
        matched_previous: dict[int, int] = {}
        matched_current: dict[int, int] = {}
        predictions: dict[int, float] = {}
        if previous_roots and roots:
            cost = np.empty((len(previous_roots), len(roots)), dtype=np.float64)
            for previous_index, previous_root in enumerate(previous_roots):
                predicted = _predict_branch(
                    histories[previous_root.branch_id], float(surface)
                )
                predictions[previous_index] = predicted
                for root_index, root in enumerate(roots):
                    type_penalty = (
                        0.0
                        if root.root_type == previous_root.root_type
                        else 0.05 * gate
                    )
                    cost[previous_index, root_index] = (
                        abs(root.electric_field_kv_m - predicted)
                        + type_penalty
                        + 1.0e-12 * (previous_index * len(roots) + root_index)
                    )
            previous_indices, root_indices = linear_sum_assignment(cost)
            for previous_index, root_index in zip(previous_indices, root_indices):
                predicted = predictions[previous_index]
                if abs(roots[root_index].electric_field_kv_m - predicted) <= gate:
                    matched_previous[int(previous_index)] = int(root_index)
                    matched_current[int(root_index)] = int(previous_index)

        annotated: list[NativeAmbipolarRoot] = []
        for root_index, root in enumerate(roots):
            if root_index in matched_current:
                previous_index = matched_current[root_index]
                branch_id = previous_roots[previous_index].branch_id
            else:
                branch_id = new_branch(root)
                kind = "boundary_origin" if surface_index == 0 else "creation"
                events.append(
                    BranchEvent(
                        kind=kind,
                        branch_ids=(branch_id,),
                        root_indices=(root_index,),
                        electric_field_kv_m=root.electric_field_kv_m,
                        detail=(
                            "branch first observed at the profile boundary"
                            if kind == "boundary_origin"
                            else "branch first observed between adjacent sampled surfaces"
                        ),
                        nonsmooth=kind != "boundary_origin",
                    )
                )
            annotated.append(replace(root, branch_id=branch_id))

        if previous_roots:
            for previous_index, previous_root in enumerate(previous_roots):
                if previous_index in matched_previous:
                    root_index = matched_previous[previous_index]
                    root = annotated[root_index]
                    if root.root_type != previous_root.root_type:
                        events.append(
                            BranchEvent(
                                kind="classification_transition",
                                branch_ids=(root.branch_id,),
                                root_indices=(root_index,),
                                electric_field_kv_m=root.electric_field_kv_m,
                                detail=(
                                    f"{previous_root.root_type} -> {root.root_type} "
                                    "between adjacent sampled surfaces"
                                ),
                            )
                        )
                    continue
                predicted = _predict_branch(
                    histories[previous_root.branch_id], float(surface)
                )
                events.append(
                    BranchEvent(
                        kind="loss",
                        branch_ids=(previous_root.branch_id,),
                        root_indices=(-1,),
                        electric_field_kv_m=predicted,
                        detail="branch no longer matched within the continuation gate",
                    )
                )
                if annotated:
                    nearest_index = min(
                        range(len(annotated)),
                        key=lambda index: abs(
                            annotated[index].electric_field_kv_m - predicted
                        ),
                    )
                    nearest = annotated[nearest_index]
                    if abs(nearest.electric_field_kv_m - predicted) <= gate:
                        events.append(
                            BranchEvent(
                                kind="merger",
                                branch_ids=(previous_root.branch_id, nearest.branch_id),
                                root_indices=(-1, nearest_index),
                                electric_field_kv_m=nearest.electric_field_kv_m,
                                detail=(
                                    "discrete merger candidate: a lost branch approaches "
                                    "a surviving branch within the continuation gate"
                                ),
                            )
                        )

            matched_items = sorted(matched_previous.items())
            for first in range(len(matched_items)):
                previous_a, current_a = matched_items[first]
                for second in range(first + 1, len(matched_items)):
                    previous_b, current_b = matched_items[second]
                    old_order = (
                        previous_roots[previous_a].electric_field_kv_m
                        - previous_roots[previous_b].electric_field_kv_m
                    )
                    new_order = (
                        annotated[current_a].electric_field_kv_m
                        - annotated[current_b].electric_field_kv_m
                    )
                    if old_order * new_order < 0.0:
                        events.append(
                            BranchEvent(
                                kind="crossing",
                                branch_ids=(
                                    annotated[current_a].branch_id,
                                    annotated[current_b].branch_id,
                                ),
                                root_indices=(current_a, current_b),
                                electric_field_kv_m=0.5
                                * (
                                    annotated[current_a].electric_field_kv_m
                                    + annotated[current_b].electric_field_kv_m
                                ),
                                detail=(
                                    "branch ordering reversed between adjacent sampled surfaces"
                                ),
                            )
                        )

        for root in annotated:
            histories.setdefault(root.branch_id, []).append(
                (float(surface), root.electric_field_kv_m)
            )
        tracked.append(
            replace(
                raw_result,
                roots=tuple(annotated),
                branch_events=tuple(events),
            )
        )
        previous_roots = tuple(annotated)

    selected_branch_id = ""
    previous_selected_field: float | None = None
    selected_results: list[NativeAmbipolarSurface] = []
    for surface_index, result in enumerate(tracked):
        if not result.roots:
            selected_results.append(
                replace(
                    result,
                    selected_root=None,
                    selected_branch_id="",
                    selection_reason="no_bracket_closest_sample",
                )
            )
            selected_branch_id = ""
            continue
        if not continue_selection or previous_selected_field is None:
            selected_root = min(
                range(len(result.roots)),
                key=lambda index: abs(result.roots[index].electric_field_kv_m),
            )
            selection_reason = (
                "nearest_zero_continuation_disabled"
                if not continue_selection and surface_index > 0
                else "nearest_zero_initial"
                if surface_index == 0
                else "nearest_zero_first_observed"
            )
        else:
            continued = next(
                (
                    index
                    for index, root in enumerate(result.roots)
                    if root.branch_id == selected_branch_id
                ),
                None,
            )
            if continued is not None:
                selected_root = continued
                selection_reason = "continued_selected_branch"
            else:
                target = (
                    0.0 if previous_selected_field is None else previous_selected_field
                )
                selected_root = min(
                    range(len(result.roots)),
                    key=lambda index: abs(
                        result.roots[index].electric_field_kv_m - target
                    ),
                )
                selection_reason = "selected_branch_lost_nearest_root"
        selected = result.roots[selected_root]
        selected_branch_id = selected.branch_id
        previous_selected_field = selected.electric_field_kv_m
        selected_results.append(
            replace(
                result,
                selected_root=selected_root,
                selected=selected.evaluation,
                selected_branch_id=selected_branch_id,
                selection_reason=selection_reason,
            )
        )
    return tuple(selected_results)


def solve_native_ambipolar_surface(
    problem: Any,
    *,
    electric_field_bounds_kv_m: tuple[float, float],
    search_points: int,
    root_tolerance_kv_m: float,
    max_root_iterations: int,
    find_all_roots: bool,
    previous_root_kv_m: float | None,
    radial_factor: float,
    solve_method: str,
    solve_tolerance: float,
    memory_budget_gb: float,
    convergence_enabled: bool = False,
    convergence_observables: tuple[str, ...] = (),
    convergence_relative_tolerance: float = 0.02,
    max_refinements: int = 0,
) -> NativeAmbipolarSurface:
    """Scan, refine, classify, and select native ambipolar roots.

    The coarse search is one bounded :func:`dkx.batch.batched_er_scan`. Each
    sign-changing interval is then refined with bracketed bisection. Every
    candidate is a real kinetic solve; no interpolated point is reported as a
    root. Nearby surfaces select the root nearest ``previous_root_kv_m`` while
    preserving every root in the returned evidence.
    """

    import time

    from dkx.batch import batched_er_scan
    from dkx.units import ELEMENTARY_CHARGE, HEAT_FLUX, PARALLEL_CURRENT, PARTICLE_FLUX

    started = time.perf_counter()
    _, evaluation_budget = _evaluation_budget(
        search_points=search_points,
        max_root_iterations=max_root_iterations,
        find_all_roots=find_all_roots,
        convergence_enabled=convergence_enabled,
        max_refinements=max_refinements,
    )
    if evaluation_budget > _MAX_RETAINED_EVALUATIONS:
        raise ValueError(
            "native ambipolar refinement preflight exceeds 100000 retained "
            f"evaluations ({evaluation_budget}); reduce convergence.max_refinements, "
            "electric_field.search_points, or max_root_iterations"
        )
    species_count = max(1, len(np.atleast_1d(getattr(problem, "z_s", [1.0]))))
    retained_bytes = evaluation_budget * (512 + 16 * species_count)
    if retained_bytes > float(memory_budget_gb) * (1024**3):
        raise MemoryError(
            "native ambipolar refinement evidence exceeds the memory preflight: "
            f"estimated={retained_bytes} B, budget={float(memory_budget_gb) * (1024**3):.0f} B"
        )
    fields = np.linspace(
        float(electric_field_bounds_kv_m[0]),
        float(electric_field_bounds_kv_m[1]),
        int(search_points),
        dtype=np.float64,
    )
    evaluations: dict[float, RootEvaluation] = {}
    chunks: list[int] = []
    chunk_sizes: list[int] = []

    def evaluate(
        values: np.ndarray, stage: str, reason: str, refinement_level: int
    ) -> list[RootEvaluation]:
        missing = [float(value) for value in values if float(value) not in evaluations]
        if missing:
            batch = batched_er_scan(
                problem,
                np.asarray(missing, dtype=np.float64),
                solve_method=solve_method,
                tol=solve_tolerance,
                memory_budget_gb=memory_budget_gb,
            )
            particle = (
                np.asarray(batch.moments["particleFlux_vm_psiHat"], dtype=np.float64)
                * float(radial_factor)
                * PARTICLE_FLUX
            )
            heat = (
                np.asarray(batch.moments["heatFlux_vm_psiHat"], dtype=np.float64)
                * float(radial_factor)
                * HEAT_FLUX
            )
            parallel = (
                np.asarray(batch.moments["FSABjHat"], dtype=np.float64)
                * PARALLEL_CURRENT
            )
            radial_current = (
                np.asarray(batch.radial_current, dtype=np.float64)
                * float(radial_factor)
                * PARTICLE_FLUX
                * ELEMENTARY_CHARGE
            )
            residuals = np.asarray(batch.residual_norms, dtype=np.float64).reshape(
                (-1,)
            )
            if not np.all(np.isfinite(residuals)):
                raise RuntimeError(
                    "native ambipolar scan produced a non-finite residual"
                )
            if hasattr(problem, "operator"):
                from dkx.er import operator_at_er

                rhs_norms = np.asarray(
                    [
                        np.linalg.norm(
                            np.asarray(
                                operator_at_er(
                                    problem.operator,
                                    value,
                                    dphi_per_er=problem.dphi_per_er,
                                ).rhs(),
                                dtype=np.float64,
                            )
                        )
                        for value in missing
                    ]
                )
                targets = float(solve_tolerance) * rhs_norms
                failed = np.flatnonzero(residuals > targets)
                if failed.size:
                    index = int(failed[0])
                    raise RuntimeError(
                        "native ambipolar solve did not converge at "
                        f"electric_field={missing[index]:.8g} kV/m: "
                        f"residual={residuals[index]:.6g}, "
                        f"target={targets[index]:.6g}"
                    )
            for index, value in enumerate(missing):
                evaluations[value] = RootEvaluation(
                    electric_field_kv_m=value,
                    radial_current_a_m2=float(radial_current[index]),
                    particle_flux_m2_s=np.asarray(particle[index]),
                    heat_flux_w_m2=np.asarray(heat[index]),
                    parallel_current_a_t_m2=float(parallel[index]),
                    residual_norm=float(residuals[index]),
                    stage=stage,
                    reason=reason,
                    refinement_level=int(refinement_level),
                )
            chunks.append(int(batch.n_chunks))
            chunk_sizes.append(int(batch.chunk_size))
        return [evaluations[float(value)] for value in values]

    evaluate(fields, "coarse_scan", "initial_uniform_grid", 0)
    roots: list[NativeAmbipolarRoot] = []
    previous_roots: tuple[NativeAmbipolarRoot, ...] | None = None
    refinement: list[RefinementEvidence] = []
    refinement_status = "not_requested"
    observables = tuple(convergence_observables) or ("electric_field",)

    for level in range((int(max_refinements) if convergence_enabled else 0) + 1):
        if level:
            midpoints = 0.5 * (fields[:-1] + fields[1:])
            evaluate(
                midpoints,
                "adaptive_refinement",
                "interval_midpoint",
                level,
            )
            fields = np.sort(np.concatenate((fields, midpoints)))
        search = [evaluations[float(value)] for value in fields]
        currents = np.asarray(
            [evaluation.radial_current_a_m2 for evaluation in search],
            dtype=np.float64,
        )
        bracket_indices = _brackets(fields, currents)
        if not find_all_roots and bracket_indices:
            bracket_indices = bracket_indices[:1]

        level_roots: list[NativeAmbipolarRoot] = []
        for left_index, right_index in bracket_indices:
            left = search[left_index]
            right = search[right_index]
            if left_index != right_index:
                for _ in range(int(max_root_iterations)):
                    width = right.electric_field_kv_m - left.electric_field_kv_m
                    if abs(width) <= float(root_tolerance_kv_m):
                        break
                    trial_field = 0.5 * (
                        left.electric_field_kv_m + right.electric_field_kv_m
                    )
                    trial = evaluate(
                        np.asarray([trial_field]),
                        "root_refinement",
                        "bracket_bisection",
                        level,
                    )[0]
                    if trial.radial_current_a_m2 == 0.0:
                        left = right = trial
                        break
                    if left.radial_current_a_m2 * trial.radial_current_a_m2 < 0.0:
                        right = trial
                    else:
                        left = trial
            root_evaluation = min(
                (left, right), key=lambda item: abs(item.radial_current_a_m2)
            )
            delta_field = right.electric_field_kv_m - left.electric_field_kv_m
            slope = (
                0.0
                if delta_field == 0.0
                else (right.radial_current_a_m2 - left.radial_current_a_m2)
                / delta_field
            )
            level_roots.append(
                NativeAmbipolarRoot(
                    electric_field_kv_m=root_evaluation.electric_field_kv_m,
                    radial_current_a_m2=root_evaluation.radial_current_a_m2,
                    slope_a_m2_per_kv_m=float(slope),
                    root_type=_classify_root(
                        root_evaluation.electric_field_kv_m, slope
                    ),
                    bracket_kv_m=(
                        min(left.electric_field_kv_m, right.electric_field_kv_m),
                        max(left.electric_field_kv_m, right.electric_field_kv_m),
                    ),
                    evaluation=root_evaluation,
                )
            )

        roots, root_movement, observable_movement = _compare_roots(
            level_roots, previous_roots, observables
        )
        max_bracket_width = max(
            (root.bracket_kv_m[1] - root.bracket_kv_m[0] for root in roots),
            default=np.nan,
        )
        resolved = bool(
            convergence_enabled
            and previous_roots is not None
            and roots
            and len(roots) == len(previous_roots)
            and root_movement <= float(root_tolerance_kv_m)
            and observable_movement <= float(convergence_relative_tolerance)
            and max_bracket_width <= float(root_tolerance_kv_m)
        )
        refinement.append(
            RefinementEvidence(
                level=level,
                search_evaluations=len(fields),
                total_evaluations=len(evaluations),
                root_count=len(roots),
                root_movement_kv_m=float(root_movement),
                observable_relative_movement=float(observable_movement),
                max_bracket_width_kv_m=float(max_bracket_width),
                converged=resolved,
            )
        )
        previous_roots = tuple(roots)
    if convergence_enabled:
        refinement_status = (
            "no_bracket_observed"
            if not roots
            else "resolved"
            if refinement[-1].converged
            else "refinement_exhausted"
        )

    selected_root: int | None = None
    if roots:
        target = 0.0 if previous_root_kv_m is None else float(previous_root_kv_m)
        selected_root = min(
            range(len(roots)),
            key=lambda index: abs(roots[index].electric_field_kv_m - target),
        )
        selected = roots[selected_root].evaluation
        status = "bracketed_root"
    else:
        selected = min(
            (
                evaluation
                for evaluation in evaluations.values()
                if evaluation.stage != "root_refinement"
            ),
            key=lambda item: abs(item.radial_current_a_m2),
        )
        status = "no_bracketed_root"

    ordered = tuple(evaluations[key] for key in sorted(evaluations))
    return NativeAmbipolarSurface(
        evaluations=ordered,
        roots=tuple(roots),
        selected_root=selected_root,
        selected=selected,
        status=status,
        solve_seconds=time.perf_counter() - started,
        batch_chunk_size=min(chunk_sizes),
        batch_chunks=sum(chunks),
        refinement=tuple(refinement),
        refinement_status=refinement_status,
        evaluation_budget=evaluation_budget,
    )


__all__ = [
    "BranchEvent",
    "NativeAmbipolarRoot",
    "NativeAmbipolarSurface",
    "RefinementEvidence",
    "RootEvaluation",
    "_evaluation_budget",
    "continue_ambipolar_branches",
    "solve_native_ambipolar_surface",
]
