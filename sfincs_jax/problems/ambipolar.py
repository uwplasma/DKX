"""Ambipolar electric-field root solves.

This module owns the in-process ambipolar problem contract.  It is deliberately
small and independent of the full v3 driver so the root-finding policy can be
tested against SFINCS Fortran v3 before it is wired to expensive transport
solves.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any


RadialCurrentEvaluator = Callable[[float], float]


def _immutable_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


@dataclass(frozen=True, slots=True)
class AmbipolarIteration:
    """One radial-current evaluation during an ambipolar root solve."""

    index: int
    er: float
    radial_current: float
    stage: str
    elapsed_s: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "er", float(self.er))
        object.__setattr__(self, "radial_current", float(self.radial_current))
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class AmbipolarProblem:
    """Shape-stable ambipolar root problem.

    The evaluator should update only the radial electric field and reuse any
    fixed-shape geometry, grid, operator, and preconditioner state it owns.
    """

    evaluate_radial_current: RadialCurrentEvaluator
    er_min: float
    er_max: float
    er_initial: float = 0.0
    max_evaluations: int = 20
    current_tolerance: float = 1.0e-10
    step_tolerance: float = 1.0e-8
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "er_min", float(self.er_min))
        object.__setattr__(self, "er_max", float(self.er_max))
        object.__setattr__(self, "er_initial", float(self.er_initial))
        object.__setattr__(self, "max_evaluations", int(self.max_evaluations))
        object.__setattr__(self, "current_tolerance", float(self.current_tolerance))
        object.__setattr__(self, "step_tolerance", float(self.step_tolerance))
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))
        if self.max_evaluations < 3:
            raise ValueError("max_evaluations must be at least 3 for the Fortran-compatible Brent path.")
        if not self.er_min < self.er_max:
            raise ValueError("er_min must be smaller than er_max.")
        if self.current_tolerance <= 0.0:
            raise ValueError("current_tolerance must be positive.")
        if self.step_tolerance <= 0.0:
            raise ValueError("step_tolerance must be positive.")


@dataclass(frozen=True, slots=True)
class AmbipolarResult:
    """Result and certificate for an ambipolar root solve."""

    converged: bool
    method: str
    root_er: float | None
    root_radial_current: float | None
    iterations: tuple[AmbipolarIteration, ...]
    status: str
    message: str = ""
    root_type: str = "unknown"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "iterations", tuple(self.iterations))
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))
        if self.root_er is not None:
            object.__setattr__(self, "root_er", float(self.root_er))
        if self.root_radial_current is not None:
            object.__setattr__(self, "root_radial_current", float(self.root_radial_current))

    @property
    def er_values(self) -> tuple[float, ...]:
        """Electric-field values evaluated in order."""

        return tuple(item.er for item in self.iterations)

    @property
    def radial_currents(self) -> tuple[float, ...]:
        """Radial-current values evaluated in order."""

        return tuple(item.radial_current for item in self.iterations)


def _same_fortran_sign(a: float, b: float) -> bool:
    """Return the sign test used in the Fortran v3 ambipolar solver."""

    return (a > 0.0 and b > 0.0) or (a < 0.0 and b < 0.0)


def _root_type(er: float | None) -> str:
    if er is None:
        return "unknown"
    return "electron" if er > 0.0 else "ion"


def _evaluate(problem: AmbipolarProblem, iterations: list[AmbipolarIteration], er: float, stage: str) -> float:
    current = float(problem.evaluate_radial_current(float(er)))
    iterations.append(
        AmbipolarIteration(
            index=len(iterations) + 1,
            er=float(er),
            radial_current=current,
            stage=stage,
        )
    )
    return current


def solve_ambipolar_brent(problem: AmbipolarProblem) -> AmbipolarResult:
    """Solve ambipolarity with the SFINCS Fortran v3 Brent algorithm.

    The implementation follows ``ambipolarSolver.F90``: evaluate ``Er_min``,
    ``Er_max``, and the initial guess first, then use the Numerical Recipes Brent
    update with the Fortran convergence check.
    """

    iterations: list[AmbipolarIteration] = []
    a = float(problem.er_min)
    fa = _evaluate(problem, iterations, a, "bracket_min")
    c = float(problem.er_max)
    fc = _evaluate(problem, iterations, c, "bracket_max")
    b = float(problem.er_initial)
    fb = _evaluate(problem, iterations, b, "initial")

    if _same_fortran_sign(fa, fc):
        return AmbipolarResult(
            converged=False,
            method="brent",
            root_er=None,
            root_radial_current=None,
            iterations=tuple(iterations),
            status="unbracketed",
            message="Root must be bracketed in Brent solve.",
        )

    if _same_fortran_sign(fa, fb):
        fa = fb
        a = b
    elif _same_fortran_sign(fc, fb):
        fc = fb
        c = b

    d = b - a
    e = d
    eps = 1.0e-15

    for _ in range(4, int(problem.max_evaluations) + 1):
        if _same_fortran_sign(fb, fc):
            c = a
            fc = fa
            e = b - a
            d = e

        if abs(fc) < abs(fb):
            old_b = b
            old_c = c
            old_fb = fb
            old_fc = fc
            a = old_b
            b = old_c
            c = old_b
            fa = old_fb
            fb = old_fc
            fc = old_fb

        # Fortran v3 uses Er_search_tolerance_f in this Brent position. Keep
        # that behavior for parity, even though the name suggests a residual
        # tolerance rather than an electric-field step tolerance.
        tol1 = 2.0 * eps * abs(b) + 0.5 * float(problem.current_tolerance)
        xm = 0.5 * (c - b)
        if abs(xm) <= tol1 or abs(fb) < float(problem.current_tolerance):
            return AmbipolarResult(
                converged=True,
                method="brent",
                root_er=b,
                root_radial_current=fb,
                iterations=tuple(iterations),
                status="converged",
                root_type=_root_type(b),
                metadata={"convergence": "bracket_width" if abs(xm) <= tol1 else "radial_current"},
            )

        if abs(e) >= tol1 and abs(fa) > abs(fb):
            s = fb / fa
            if a == c:
                p = 2.0 * xm * s
                q = 1.0 - s
            else:
                q = fa / fc
                r = fb / fc
                p = s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0.0:
                q = -q
            p = abs(p)
            if 2.0 * p < min(3.0 * xm * q - abs(tol1 * q), abs(e * q)):
                e = d
                d = p / q
            else:
                d = xm
                e = d
        else:
            d = xm
            e = d

        a = b
        fa = fb
        if abs(d) > tol1:
            b = b + d
        else:
            b = b + math.copysign(tol1, xm)
        fb = _evaluate(problem, iterations, b, "brent")

    return AmbipolarResult(
        converged=False,
        method="brent",
        root_er=b,
        root_radial_current=fb,
        iterations=tuple(iterations),
        status="max_evaluations",
        message="The Er search did not converge within max_evaluations.",
        root_type=_root_type(b),
    )


def brent_ambipolar_root(
    evaluate_radial_current: RadialCurrentEvaluator,
    *,
    er_min: float,
    er_max: float,
    er_initial: float = 0.0,
    max_evaluations: int = 20,
    current_tolerance: float = 1.0e-10,
    step_tolerance: float = 1.0e-8,
    metadata: Mapping[str, Any] | None = None,
) -> AmbipolarResult:
    """Convenience wrapper for a Fortran-compatible Brent ambipolar solve."""

    return solve_ambipolar_brent(
        AmbipolarProblem(
            evaluate_radial_current=evaluate_radial_current,
            er_min=er_min,
            er_max=er_max,
            er_initial=er_initial,
            max_evaluations=max_evaluations,
            current_tolerance=current_tolerance,
            step_tolerance=step_tolerance,
            metadata=metadata,
        )
    )


def _lookup_config_value(config: Any, groups: tuple[str, ...], key: str, default: Any = None) -> Any:
    key_upper = key.upper()
    for group in groups:
        group_data: Any
        if hasattr(config, "group"):
            group_data = config.group(group)
        elif isinstance(config, Mapping):
            group_data = config.get(group, config.get(group.lower(), config.get(group.upper(), {})))
        else:
            group_data = {}
        if isinstance(group_data, Mapping):
            if key_upper in group_data:
                return group_data[key_upper]
            if key in group_data:
                return group_data[key]
            lower_map = {str(k).lower(): v for k, v in group_data.items()}
            if key.lower() in lower_map:
                return lower_map[key.lower()]
    if isinstance(config, Mapping):
        if key_upper in config:
            return config[key_upper]
        if key in config:
            return config[key]
        lower_map = {str(k).lower(): v for k, v in config.items()}
        if key.lower() in lower_map:
            return lower_map[key.lower()]
    return default


def validate_fortran_v3_ambipolar_constraints(config: Any, *, option: int | None = None) -> tuple[str, ...]:
    """Return Fortran-v3 compatibility errors for ambipolar solve settings.

    Options 1 and 3 use adjoint radial-current derivatives in Fortran v3 and
    inherit the RHSMode>3 restrictions in ``validateInput.F90``.  Option 2
    Brent does not require the adjoint derivative, but Fortran still rejects
    ``includePhi1`` and RHSMode 2/3 for all ambipolar solves.
    """

    general = ("general", "generalparameters")
    physics = ("physicsParameters", "physicsparameters")
    resolution = ("resolutionParameters", "resolutionparameters")

    opt = int(option if option is not None else _lookup_config_value(config, general, "ambipolarSolveOption", 2))
    rhs_mode = int(_lookup_config_value(config, general, "RHSMode", 1))
    include_phi1 = bool(_lookup_config_value(config, physics, "includePhi1", False))
    magnetic_drift_scheme = int(_lookup_config_value(config, physics, "magneticDriftScheme", 0))
    collision_operator = int(_lookup_config_value(config, physics, "collisionOperator", 0))
    constraint_scheme = int(_lookup_config_value(config, resolution, "constraintScheme", -1))
    e_parallel = float(_lookup_config_value(config, physics, "EParallelHat", 0.0))

    errors: list[str] = []
    if include_phi1:
        errors.append("ambipolarSolve cannot be used with includePhi1 in SFINCS Fortran v3.")
    if rhs_mode in (2, 3):
        errors.append("ambipolarSolve must be used with RHSMode 1, 4, or 5 in SFINCS Fortran v3.")
    if opt in (1, 3):
        if e_parallel != 0.0:
            errors.append("derivative-assisted ambipolar solves cannot use EParallelHat != 0 in SFINCS Fortran v3.")
        if magnetic_drift_scheme > 0:
            errors.append("derivative-assisted ambipolar solves cannot use tangential magnetic drifts in SFINCS Fortran v3.")
        if constraint_scheme not in (-1, 1):
            errors.append("derivative-assisted ambipolar solves require constraintScheme=-1 or 1 in SFINCS Fortran v3.")
        if collision_operator != 0:
            errors.append("derivative-assisted ambipolar solves require collisionOperator=0 in SFINCS Fortran v3.")
    return tuple(errors)


__all__ = [
    "AmbipolarIteration",
    "AmbipolarProblem",
    "AmbipolarResult",
    "RadialCurrentEvaluator",
    "brent_ambipolar_root",
    "solve_ambipolar_brent",
    "validate_fortran_v3_ambipolar_constraints",
]
