"""Phase-space convergence study for a case.

``execution.run_case`` already refines the ambipolar ``E_r`` axis when
``[convergence]`` is enabled. This module covers the other half that
``execution.py`` used to point at by name without it existing: refinement of
the phase-space axes ``theta``, ``zeta``, ``pitch`` and ``speed``.

The study reports what plan.md section 7.5 asks for -- convergence of the
requested *observables*, not of a state-vector norm -- and it refines the axes
both independently and jointly. The joint run is not redundant. Refining one
axis at a time can show every axis individually converged while the solution is
not: the axes couple, and a resolution that is adequate in ``theta`` only
because ``pitch`` is too coarse to expose the error will look settled until
both move together. When the joint change exceeds the largest single-axis
change by more than a factor of two, this reports that explicitly rather than
letting the per-axis table imply a convergence the case has not demonstrated.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Mapping

import numpy as np

#: The phase-space axes a study can refine, in the order they are reported.
AXES: tuple[str, ...] = ("theta", "zeta", "pitch", "speed")

#: Result arrays compared entrywise across resolutions, then reduced to worst change.
DEFAULT_OBSERVABLES: tuple[str, ...] = (
    "particle_flux_m2_s",
    "heat_flux_W_m2",
    "parallel_current_A_T_m2",
)


@dataclass(frozen=True)
class AxisRefinement:
    """One refinement run: which axes moved, to what, and what the outputs did."""

    label: str
    resolution: dict[str, int]
    changes: dict[str, float]
    seconds: float

    @property
    def worst(self) -> float:
        """Largest relative change over the observables, or 0.0 if none compared."""
        return max(self.changes.values(), default=0.0)


@dataclass(frozen=True)
class ConvergenceReport:
    baseline: dict[str, int]
    refinements: tuple[AxisRefinement, ...]
    joint: AxisRefinement | None
    tolerance: float

    @property
    def per_axis_worst(self) -> float:
        return max((r.worst for r in self.refinements), default=0.0)

    @property
    def converged(self) -> bool:
        """Every measured change, joint included, is inside the tolerance."""
        worst = self.per_axis_worst
        if self.joint is not None:
            worst = max(worst, self.joint.worst)
        return bool(self.refinements) and np.isfinite(worst) and worst < self.tolerance

    @property
    def axes_understate_the_joint_change(self) -> bool:
        """True when the per-axis table would overstate how settled the case is.

        The factor of two is a reporting threshold, not a physical one: below it
        the joint run is consistent with the axes being effectively separable at
        this resolution, and above it the single-axis numbers are not a safe
        summary of what refinement does.
        """
        if self.joint is None:
            return False
        return self.joint.worst > 2.0 * max(self.per_axis_worst, 1e-300)


def _relative_changes(
    reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray],
    *, tolerance: float = 1.0, absolute_tolerances: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Worst signed, entrywise change per observable on aligned surfaces/species.

    Missing, empty, nonfinite or shape-changing data cannot establish convergence.
    Normalize by max(abs(reference), absolute_tolerance / tolerance), so the
    verdict applies the requested relative OR absolute budget to every entry.
    A zero reference has no implicit unit-dependent absolute allowance.
    """
    changes: dict[str, float] = {}
    for name, ref in reference.items():
        ref = np.asarray(ref, dtype=float)
        got = np.asarray(candidate.get(name, np.nan), dtype=float)
        if (ref.size == 0 or got.shape != ref.shape
                or not np.all(np.isfinite(ref)) or not np.all(np.isfinite(got))):
            changes[name] = float("inf")
            continue
        with np.errstate(over="ignore", invalid="ignore"):
            atol = (absolute_tolerances or {}).get(name, 0.0)
            # Normalize first to avoid overflowing got-ref or atol/tolerance.
            scale = np.maximum(np.abs(ref), atol)
            divisor = np.where(scale > 0.0, scale, 1.0)
            difference = np.abs(got / divisor - ref / divisor)
            budget = np.maximum(tolerance * (np.abs(ref) / divisor), atol / divisor)
            ratio = tolerance / np.where(budget > 0.0, budget, 1.0)
            change = difference * ratio
            change = np.where((scale == 0.0) & (got != ref), np.inf, change)
            changes[name] = float(np.max(change))
    return changes


def _refined(resolution, axis_or_axes: str | tuple[str, ...], factor: float):
    """Return a resolution with the named axes scaled up by ``factor``.

    ``zeta = 1`` marks an axisymmetric case, where the zeta axis is not a
    resolution to refine but a statement that the configuration has no
    toroidal variation. Scaling it would change the physics being solved rather
    than the accuracy of the solve, so it is left alone.
    """
    axes = (axis_or_axes,) if isinstance(axis_or_axes, str) else axis_or_axes
    updates: dict[str, int] = {}
    for axis in axes:
        current = getattr(resolution, axis)
        if axis == "zeta" and current == 1:
            continue
        updates[axis] = max(current + 1, int(round(current * factor)))
    return replace(resolution, **updates) if updates else resolution


def _resolution_dict(resolution) -> dict[str, int]:
    return {axis: int(getattr(resolution, axis)) for axis in AXES}


def _converge(
    resolution, run_at_resolution,
    *,
    axes: tuple[str, ...] = AXES,
    factor: float = 1.5,
    tolerance: float = 0.02,
    observables: tuple[str, ...] = DEFAULT_OBSERVABLES,
    absolute_tolerances: Mapping[str, float] | None = None,
    joint: bool = True,
    emit: Callable[[str], None] | None = None,
) -> ConvergenceReport:
    """Refine each axis of ``case`` and report what the observables did.

    Runs the case once at its stated resolution, once per axis with that axis
    refined, and -- unless ``joint`` is false -- once with every requested axis
    refined together. Costs ``len(axes) + 2`` solves. Optional per-observable
    ``absolute_tolerances`` are in the arrays' physical units and default to zero.
    Each entry must change by less than max(tolerance * abs(reference), atol).
    """
    import time  # noqa: PLC0415

    unknown = sorted(set(axes) - set(AXES))
    if unknown:
        raise ValueError(f"unknown refinement axes {unknown}; choose from {list(AXES)}")
    if not np.isfinite(factor) or factor <= 1.0:
        raise ValueError(f"factor must exceed 1.0 to refine, got {factor}")

    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")

    absolute_tolerances = dict(absolute_tolerances or {})
    if set(absolute_tolerances) - set(observables):
        raise ValueError("absolute_tolerances names must be requested observables")
    if any(not np.isfinite(v) or v < 0.0 for v in absolute_tolerances.values()):
        raise ValueError("absolute tolerances must be finite and nonnegative")

    def _say(message: str) -> None:
        if emit is not None:
            emit(message)

    def _observables_of(result) -> dict[str, np.ndarray]:
        if not result.metadata.get("converged", False):
            raise ValueError("a failed solve cannot establish resolution convergence")
        missing = set(observables) - set(result.arrays)
        if missing:
            raise ValueError(f"requested observables are missing: {sorted(missing)}")
        return {
            name: np.asarray(result.arrays[name], dtype=float).copy()
            for name in observables
        }

    _say(f"baseline {_resolution_dict(resolution)}")
    baseline_result = run_at_resolution(resolution)
    reference = _observables_of(baseline_result)
    if not reference:
        raise ValueError(
            "at least one requested observable is required"
        )

    refinements: list[AxisRefinement] = []
    for axis in axes:
        refined = _refined(resolution, axis, factor)
        if refined == resolution:
            _say(f"{axis}: not refinable for this case, skipped")
            continue
        _say(f"refining {axis} -> {getattr(refined, axis)}")
        started = time.perf_counter()
        result = run_at_resolution(refined)
        refinements.append(
            AxisRefinement(
                label=axis,
                resolution=_resolution_dict(refined),
                changes=_relative_changes(
                    reference, _observables_of(result), tolerance=tolerance,
                    absolute_tolerances=absolute_tolerances,
                ),
                seconds=time.perf_counter() - started,
            )
        )

    joint_refinement: AxisRefinement | None = None
    if joint and len(refinements) > 1:
        refined = _refined(resolution, tuple(axes), factor)
        _say(f"refining every axis together -> {_resolution_dict(refined)}")
        started = time.perf_counter()
        result = run_at_resolution(refined)
        joint_refinement = AxisRefinement(
            label="all axes",
            resolution=_resolution_dict(refined),
            changes=_relative_changes(
                    reference, _observables_of(result), tolerance=tolerance,
                    absolute_tolerances=absolute_tolerances,
                ),
            seconds=time.perf_counter() - started,
        )

    return ConvergenceReport(
        baseline=_resolution_dict(resolution),
        refinements=tuple(refinements),
        joint=joint_refinement,
        tolerance=tolerance,
    )


def converge_case(
    case, *, axes: tuple[str, ...] = AXES, factor: float = 1.5,
    tolerance: float = 0.02, observables: tuple[str, ...] = DEFAULT_OBSERVABLES,
    absolute_tolerances: Mapping[str, float] | None = None,
    joint: bool = True, emit: Callable[[str], None] | None = None,
) -> ConvergenceReport:
    """Refine each axis and optionally all axes jointly; compare signed entries.

    Absolute tolerances are per-observable physical units, defaulting to zero.
    Every entry must change by less than max(tolerance * abs(reference), atol).
    """
    from ..execution import run_case  # noqa: PLC0415

    return _converge(
        case.resolution, lambda r: run_case(replace(case, resolution=r)),
        axes=axes, factor=factor, tolerance=tolerance, observables=observables,
        absolute_tolerances=absolute_tolerances, joint=joint, emit=emit,
    )


def converge_sfincs_input(source, **kwargs) -> ConvergenceReport:
    """Refine a linear RHSMode 1/2/3 namelist without converting its physics.

    Accept a path or SfincsInput. Default observables use SFINCS normalization:
    per-species flow/particle/heat moments for RHSMode 1, the full transport
    matrix for RHSMode 2/3. Every original RHS residual must pass the deck's
    solverTolerance. No files are written; geometry policy and gradients remain
    those of the supplied deck. Refinement options match converge_case.
    """
    from types import SimpleNamespace  # noqa: PLC0415

    from ..config import ResolutionConfig  # noqa: PLC0415
    from ..inputs import SfincsInput, load_sfincs_input  # noqa: PLC0415
    from ..run import run_profile, run_transport_matrix  # noqa: PLC0415

    inp = source if isinstance(source, SfincsInput) else load_sfincs_input(source)
    mode = inp.general.rhs_mode
    if mode not in (1, 2, 3) or inp.physics.include_phi1:
        raise ValueError("namelist convergence requires linear RHSMode 1, 2 or 3 with includePhi1=false")
    names = dict(theta="n_theta", zeta="n_zeta", pitch="n_xi", speed="n_x")
    resolution = ResolutionConfig(**{k: getattr(inp.resolution, v) for k, v in names.items()})
    kwargs.setdefault("observables", ("FSABFlow", "particleFlux_vm_psiHat", "heatFlux_vm_psiHat")
                      if mode == 1 else ("transport_matrix",))

    def run_at_resolution(r):
        updated = replace(inp, resolution=replace(inp.resolution, **{
            v: getattr(r, k) for k, v in names.items()}))
        driver = run_profile if mode == 1 else run_transport_matrix
        run = driver(updated, tol=inp.resolution.solver_tolerance, emit=None)
        states = [run.state_vector] if mode == 1 else run.state_vectors
        accepted = bool(run.solve_result.converged)
        for i, state in enumerate(states, 1):
            rhs = np.asarray(run.operator.rhs(i))
            defect = np.asarray(run.operator.apply(state)) - rhs
            norm_b, norm_r = np.linalg.norm(rhs), np.linalg.norm(defect)
            residual = norm_r / norm_b if norm_b else (0.0 if norm_r == 0 else np.inf)
            accepted = accepted and np.isfinite(residual) and residual <= inp.resolution.solver_tolerance
        arrays = dict(run.moments)
        if mode != 1:
            arrays["transport_matrix"] = run.transport_matrix
        return SimpleNamespace(arrays=arrays, metadata={"converged": accepted})

    return _converge(resolution, run_at_resolution, **kwargs)
