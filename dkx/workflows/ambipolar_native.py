"""Native physical-unit ambipolar scans and bracket refinement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


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


@dataclass(frozen=True)
class NativeAmbipolarRoot:
    """One sign-bracketed and physically re-evaluated ambipolar root."""

    electric_field_kv_m: float
    radial_current_a_m2: float
    slope_a_m2_per_kv_m: float
    root_type: str
    bracket_kv_m: tuple[float, float]
    evaluation: RootEvaluation


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
    fields = np.linspace(
        float(electric_field_bounds_kv_m[0]),
        float(electric_field_bounds_kv_m[1]),
        int(search_points),
        dtype=np.float64,
    )
    evaluations: dict[float, RootEvaluation] = {}
    chunks: list[int] = []
    chunk_sizes: list[int] = []

    def evaluate(values: np.ndarray, stage: str) -> list[RootEvaluation]:
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
                )
            chunks.append(int(batch.n_chunks))
            chunk_sizes.append(int(batch.chunk_size))
        return [evaluations[float(value)] for value in values]

    coarse = evaluate(fields, "coarse_scan")
    currents = np.asarray(
        [evaluation.radial_current_a_m2 for evaluation in coarse], dtype=np.float64
    )
    bracket_indices = _brackets(fields, currents)
    if not find_all_roots and bracket_indices:
        bracket_indices = bracket_indices[:1]

    roots: list[NativeAmbipolarRoot] = []
    for left_index, right_index in bracket_indices:
        left = coarse[left_index]
        right = coarse[right_index]
        if left_index != right_index:
            for _ in range(int(max_root_iterations)):
                width = right.electric_field_kv_m - left.electric_field_kv_m
                if abs(width) <= float(root_tolerance_kv_m):
                    break
                trial_field = 0.5 * (
                    left.electric_field_kv_m + right.electric_field_kv_m
                )
                trial = evaluate(np.asarray([trial_field]), "root_refinement")[0]
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
            else (right.radial_current_a_m2 - left.radial_current_a_m2) / delta_field
        )
        roots.append(
            NativeAmbipolarRoot(
                electric_field_kv_m=root_evaluation.electric_field_kv_m,
                radial_current_a_m2=root_evaluation.radial_current_a_m2,
                slope_a_m2_per_kv_m=float(slope),
                root_type=_classify_root(root_evaluation.electric_field_kv_m, slope),
                bracket_kv_m=(
                    min(left.electric_field_kv_m, right.electric_field_kv_m),
                    max(left.electric_field_kv_m, right.electric_field_kv_m),
                ),
                evaluation=root_evaluation,
            )
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
        selected = min(coarse, key=lambda item: abs(item.radial_current_a_m2))
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
    )


__all__ = [
    "NativeAmbipolarRoot",
    "NativeAmbipolarSurface",
    "RootEvaluation",
    "solve_native_ambipolar_surface",
]
