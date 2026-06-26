"""Ambipolar electric-field root solves.

This module owns the in-process ambipolar problem contract.  It is deliberately
small and independent of the full v3 driver so the root-finding policy can be
tested against SFINCS Fortran v3 before it is wired to expensive transport
solves.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
import math
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ..input_compat import first_config_value as _first_scalar
from ..input_compat import lookup_config_value as _lookup_config_value


RadialCurrentEvaluator = Callable[[float], float]
RadialCurrentDerivativeEvaluator = Callable[[float], Any]


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


@dataclass(frozen=True, slots=True)
class SfincsJaxEvaluationRecord:
    """One concrete sfincs_jax output evaluation used by an ambipolar solve."""

    er: float
    radial_current: float
    input_path: Path
    output_path: Path
    solver_trace_path: Path | None = None
    selected_path: str | None = None
    solve_method: str | None = None
    preconditioner: str | None = None
    residual_norm: float | None = None
    residual_target: float | None = None
    converged: bool | None = None
    setup_s: float | None = None
    solve_s: float | None = None
    elapsed_s: float | None = None
    total_size: int | None = None
    active_size: int | None = None
    cache_enabled: bool = False
    cache_dir: Path | None = None
    solver_state_reuse_enabled: bool = False
    solver_state_path: Path | None = None
    solver_state_input_exists: bool = False
    solver_state_input_used: bool = False
    solver_state_output_exists: bool = False
    fixed_shape_input_signature: tuple[int, ...] | None = None
    fixed_shape_signature: tuple[int, ...] | None = None
    fixed_shape_reuse_enabled: bool = False
    fixed_shape_reuse_admitted: bool = False
    fixed_shape_reuse_reason: str = "disabled"
    fixed_shape_reuse_count: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "er", float(self.er))
        object.__setattr__(self, "radial_current", float(self.radial_current))
        object.__setattr__(self, "input_path", Path(self.input_path))
        object.__setattr__(self, "output_path", Path(self.output_path))
        if self.solver_trace_path is not None:
            object.__setattr__(self, "solver_trace_path", Path(self.solver_trace_path))
        if self.cache_dir is not None:
            object.__setattr__(self, "cache_dir", Path(self.cache_dir))
        if self.solver_state_path is not None:
            object.__setattr__(self, "solver_state_path", Path(self.solver_state_path))
        if self.fixed_shape_input_signature is not None:
            object.__setattr__(
                self,
                "fixed_shape_input_signature",
                tuple(int(v) for v in self.fixed_shape_input_signature),
            )
        if self.fixed_shape_signature is not None:
            object.__setattr__(
                self,
                "fixed_shape_signature",
                tuple(int(v) for v in self.fixed_shape_signature),
            )
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class RadialCurrentDerivativeResult:
    """Finite-difference derivative certificate for ``dJr/dEr``."""

    er: float
    derivative: float
    step: float
    scheme: str
    evaluations: tuple[AmbipolarIteration, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "er", float(self.er))
        object.__setattr__(self, "derivative", float(self.derivative))
        object.__setattr__(self, "step", float(self.step))
        object.__setattr__(self, "evaluations", tuple(self.evaluations))
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class RHSMode1RadialCurrentResponse:
    """Namelist-backed radial-current response and derivative provider."""

    build_system: Callable[[float], Any]
    finite_difference_step: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.finite_difference_step is not None:
            step = float(self.finite_difference_step)
            if step <= 0.0:
                raise ValueError("finite_difference_step must be positive.")
            object.__setattr__(self, "finite_difference_step", step)
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))

    def radial_current(self, er: float) -> float:
        """Evaluate the matrix-free radial-current observable at ``Er``."""

        from ..sensitivity import evaluate_matrix_free_linear_observable  # noqa: PLC0415

        return evaluate_matrix_free_linear_observable(self.build_system(float(er)))

    def derivative(self, er: float) -> RadialCurrentDerivativeResult:
        """Evaluate ``dJr/dEr`` through the matrix-free implicit certificate."""

        return implicit_matrix_free_radial_current_derivative_from_builder(
            self.build_system,
            er=float(er),
            finite_difference_step=self.finite_difference_step,
            metadata=self.metadata,
        )


@contextmanager
def _temporary_env(overrides: Mapping[str, str | None]):
    previous = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class SfincsJaxRadialCurrentEvaluator:
    """In-process radial-current evaluator backed by ``write_sfincs_jax_output_h5``.

    This is the first real-solve bridge for the canonical ambipolar owner.  It
    keeps template parsing, directory ownership, and call records in one object.
    Later setup reuse should be added behind this class without changing the
    public Brent/Newton root-solving contracts.
    """

    def __init__(
        self,
        *,
        input_namelist: str | Path,
        work_dir: str | Path,
        variable: str | None = None,
        solve_method: str = "auto",
        differentiable: bool = False,
        compute_solution: bool = True,
        overwrite: bool = True,
        reuse_output_geometry_cache: bool = True,
        cache_dir: str | Path | None = None,
        reuse_solver_state: bool = True,
        solver_state_path: str | Path | None = None,
        reuse_fixed_shape_setup: bool = True,
        emit: Callable[[int, str], None] | None = None,
    ) -> None:
        self.input_namelist = Path(input_namelist).resolve()
        self.work_dir = Path(work_dir).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.solve_method = str(solve_method)
        self.differentiable = bool(differentiable)
        self.compute_solution = bool(compute_solution)
        self.overwrite = bool(overwrite)
        self.reuse_output_geometry_cache = bool(reuse_output_geometry_cache)
        self.cache_dir = (
            Path(cache_dir).resolve()
            if cache_dir is not None
            else self.work_dir / ".sfincs_jax_output_cache"
        )
        self.reuse_solver_state = bool(reuse_solver_state)
        self.reuse_fixed_shape_setup = bool(reuse_fixed_shape_setup)
        self.solver_state_path = (
            Path(solver_state_path).resolve()
            if solver_state_path is not None
            else self.work_dir / ".sfincs_jax_solver_state" / "rhsmode1_state.npz"
        )
        self.emit = emit
        self.template_text = self.input_namelist.read_text()
        if variable is None:
            from ..namelist import read_sfincs_input  # noqa: PLC0415
            from ..scans import _er_scan_var_name  # noqa: PLC0415

            variable = _er_scan_var_name(nml=read_sfincs_input(self.input_namelist))
        self.variable = str(variable)
        self.records: list[SfincsJaxEvaluationRecord] = []
        self._last_fixed_shape_signature: tuple[int, ...] | None = None
        self._fixed_shape_reuse_count = 0

    def __call__(self, er: float) -> float:
        from ..ambipolar import radial_current_from_output  # noqa: PLC0415
        from ..io import localize_equilibrium_file_in_place, read_sfincs_h5, write_sfincs_jax_output_h5  # noqa: PLC0415
        from ..scans import _patch_scalar_in_group  # noqa: PLC0415
        from ..solver_state import read_krylov_state_signature  # noqa: PLC0415
        from ..solvers.trace import read_solver_trace_json  # noqa: PLC0415

        index = len(self.records) + 1
        run_dir = self.work_dir / f"eval_{index:03d}_{self.variable}_{float(er):+.8e}".replace("+", "p").replace("-", "m")
        run_dir.mkdir(parents=True, exist_ok=True)
        eval_input = run_dir / "input.namelist"
        eval_output = run_dir / "sfincsOutput.h5"
        solver_trace = run_dir / "sfincsOutput.solver_trace.json"
        patched = _patch_scalar_in_group(
            txt=self.template_text,
            group="physicsParameters",
            key=self.variable,
            value=float(er),
        )
        eval_input.write_text(patched)
        localize_equilibrium_file_in_place(input_namelist=eval_input, overwrite=False)

        env_overrides: dict[str, str | None] = {}
        if self.reuse_output_geometry_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            env_overrides.update(
                {
                    "SFINCS_JAX_OUTPUT_CACHE": "1",
                    "SFINCS_JAX_OUTPUT_CACHE_PERSIST": "1",
                    "SFINCS_JAX_OUTPUT_CACHE_DIR": str(self.cache_dir),
                }
            )
        solver_state_input_exists = bool(self.reuse_solver_state and self.solver_state_path.exists())
        fixed_shape_input_signature = (
            read_krylov_state_signature(self.solver_state_path)
            if self.reuse_solver_state and solver_state_input_exists
            else None
        )
        fixed_shape_reuse_enabled = bool(self.reuse_solver_state and self.reuse_fixed_shape_setup)
        solver_state_input_used = False
        fixed_shape_reuse_admitted = False
        if not fixed_shape_reuse_enabled:
            fixed_shape_reuse_reason = "disabled"
        elif not solver_state_input_exists:
            fixed_shape_reuse_reason = "no_prior_state"
        elif fixed_shape_input_signature is None:
            fixed_shape_reuse_reason = "missing_or_invalid_signature"
        elif self._last_fixed_shape_signature is None:
            fixed_shape_reuse_admitted = True
            fixed_shape_reuse_reason = "external_state_unverified"
        elif tuple(fixed_shape_input_signature) == tuple(self._last_fixed_shape_signature):
            fixed_shape_reuse_admitted = True
            fixed_shape_reuse_reason = "fixed_shape_signature_match"
        else:
            fixed_shape_reuse_reason = "fixed_shape_signature_mismatch"
        solver_state_input_used = bool(solver_state_input_exists and fixed_shape_reuse_admitted)
        if self.reuse_solver_state:
            self.solver_state_path.parent.mkdir(parents=True, exist_ok=True)
            env_overrides["SFINCS_JAX_STATE_OUT"] = str(self.solver_state_path)
            if solver_state_input_used:
                env_overrides["SFINCS_JAX_STATE_IN"] = str(self.solver_state_path)
        with _temporary_env(env_overrides):
            write_sfincs_jax_output_h5(
                input_namelist=eval_input,
                output_path=eval_output,
                overwrite=self.overwrite,
                compute_transport_matrix=False,
                compute_solution=self.compute_solution,
                solver_trace_path=solver_trace,
                solve_method=self.solve_method,
                differentiable=self.differentiable,
                emit=self.emit,
            )
        radial_current = radial_current_from_output(read_sfincs_h5(eval_output))
        trace = read_solver_trace_json(solver_trace) if solver_trace.exists() else None
        solver_state_output_exists = bool(self.reuse_solver_state and self.solver_state_path.exists())
        fixed_shape_signature = (
            read_krylov_state_signature(self.solver_state_path)
            if self.reuse_solver_state and solver_state_output_exists
            else None
        )
        if fixed_shape_reuse_admitted:
            self._fixed_shape_reuse_count += 1
        if fixed_shape_signature is not None:
            self._last_fixed_shape_signature = tuple(fixed_shape_signature)
        self.records.append(
            SfincsJaxEvaluationRecord(
                er=float(er),
                radial_current=float(radial_current),
                input_path=eval_input,
                output_path=eval_output,
                solver_trace_path=solver_trace if solver_trace.exists() else None,
                selected_path=None if trace is None else trace.selected_path,
                solve_method=None if trace is None else trace.solve_method,
                preconditioner=None if trace is None else trace.preconditioner,
                residual_norm=None if trace is None else trace.residual_norm,
                residual_target=None if trace is None else trace.residual_target,
                converged=None if trace is None else trace.converged,
                setup_s=None if trace is None else trace.setup_s,
                solve_s=None if trace is None else trace.solve_s,
                elapsed_s=None if trace is None else trace.elapsed_s,
                total_size=None if trace is None else trace.total_size,
                active_size=None if trace is None else trace.active_size,
                cache_enabled=bool(self.reuse_output_geometry_cache),
                cache_dir=self.cache_dir if self.reuse_output_geometry_cache else None,
                solver_state_reuse_enabled=bool(self.reuse_solver_state),
                solver_state_path=self.solver_state_path if self.reuse_solver_state else None,
                solver_state_input_exists=solver_state_input_exists,
                solver_state_input_used=solver_state_input_used,
                solver_state_output_exists=solver_state_output_exists,
                fixed_shape_input_signature=fixed_shape_input_signature,
                fixed_shape_signature=fixed_shape_signature,
                fixed_shape_reuse_enabled=fixed_shape_reuse_enabled,
                fixed_shape_reuse_admitted=fixed_shape_reuse_admitted,
                fixed_shape_reuse_reason=fixed_shape_reuse_reason,
                fixed_shape_reuse_count=self._fixed_shape_reuse_count,
                metadata={
                    "requested_solve_method": self.solve_method,
                    "solver_state_reuse_enabled": bool(self.reuse_solver_state),
                    "solver_state_input_exists": solver_state_input_exists,
                    "solver_state_input_used": solver_state_input_used,
                    "solver_state_output_exists": solver_state_output_exists,
                    "fixed_shape_input_signature": fixed_shape_input_signature,
                    "fixed_shape_signature": fixed_shape_signature,
                    "fixed_shape_reuse_enabled": fixed_shape_reuse_enabled,
                    "fixed_shape_reuse_admitted": fixed_shape_reuse_admitted,
                    "fixed_shape_reuse_reason": fixed_shape_reuse_reason,
                    "fixed_shape_reuse_count": self._fixed_shape_reuse_count,
                    **({} if trace is None else {"solver_trace_metadata": dict(trace.metadata)}),
                },
            )
        )
        return float(radial_current)


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


def _coerce_derivative_result(
    value: float | RadialCurrentDerivativeResult,
    *,
    er: float,
    source: str,
) -> RadialCurrentDerivativeResult:
    if isinstance(value, RadialCurrentDerivativeResult):
        return value
    return RadialCurrentDerivativeResult(
        er=float(er),
        derivative=float(value),
        step=0.0,
        scheme=str(source),
        evaluations=(),
        metadata={"source": str(source)},
    )


def _evaluate_derivative(
    evaluate_derivative: RadialCurrentDerivativeEvaluator,
    *,
    er: float,
    source: str,
) -> RadialCurrentDerivativeResult:
    result = _coerce_derivative_result(evaluate_derivative(float(er)), er=float(er), source=source)
    if not math.isfinite(float(result.derivative)):
        raise FloatingPointError("dJr/dEr must be finite.")
    return result


def _bracket_from_values(
    *,
    a: float,
    fa: float,
    c: float,
    fc: float,
    b: float,
    fb: float,
) -> tuple[float, float, float, float] | None:
    """Return a sign-changing bracket using the initial point when useful."""

    if _same_fortran_sign(fa, fc):
        return None
    lo = float(a)
    flo = float(fa)
    hi = float(c)
    fhi = float(fc)
    if not _same_fortran_sign(fa, fb):
        hi = float(b)
        fhi = float(fb)
    elif not _same_fortran_sign(fc, fb):
        lo = float(b)
        flo = float(fb)
    if lo > hi:
        lo, hi = hi, lo
        flo, fhi = fhi, flo
    return lo, flo, hi, fhi


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


def solve_ambipolar_safeguarded_newton(
    problem: AmbipolarProblem,
    evaluate_derivative: RadialCurrentDerivativeEvaluator,
    *,
    derivative_source: str = "external",
) -> AmbipolarResult:
    """Solve ambipolarity with derivative-assisted Newton plus bisection guards.

    This is the sfincs_jax owner for the Fortran-v3 option-1 style policy.  The
    derivative provider can be a finite-difference gate today or an exact
    implicit/adjoint derivative later.  The method preserves a sign-changing
    bracket and falls back to bisection whenever the Newton step is unsafe.
    """

    iterations: list[AmbipolarIteration] = []
    a = float(problem.er_min)
    fa = _evaluate(problem, iterations, a, "bracket_min")
    c = float(problem.er_max)
    fc = _evaluate(problem, iterations, c, "bracket_max")
    b = float(problem.er_initial)
    fb = _evaluate(problem, iterations, b, "initial")

    bracket = _bracket_from_values(a=a, fa=fa, c=c, fc=fc, b=b, fb=fb)
    if bracket is None:
        return AmbipolarResult(
            converged=False,
            method="safeguarded_newton",
            root_er=None,
            root_radial_current=None,
            iterations=tuple(iterations),
            status="unbracketed",
            message="Root must be bracketed in safeguarded Newton solve.",
        )
    lo, flo, hi, _ = bracket
    if abs(fb) < float(problem.current_tolerance):
        return AmbipolarResult(
            converged=True,
            method="safeguarded_newton",
            root_er=b,
            root_radial_current=fb,
            iterations=tuple(iterations),
            status="converged",
            root_type=_root_type(b),
            metadata={"convergence": "radial_current"},
        )

    derivative_records: list[RadialCurrentDerivativeResult] = []
    fallback_count = 0
    step = math.inf
    for _ in range(4, int(problem.max_evaluations) + 1):
        try:
            derivative = _evaluate_derivative(evaluate_derivative, er=b, source=derivative_source)
        except FloatingPointError as exc:
            return AmbipolarResult(
                converged=False,
                method="safeguarded_newton",
                root_er=b,
                root_radial_current=fb,
                iterations=tuple(iterations),
                status="invalid_derivative",
                message=str(exc),
                root_type=_root_type(b),
                metadata={"derivative_count": len(derivative_records), "fallback_count": fallback_count},
            )
        derivative_records.append(derivative)
        d_jr_d_er = float(derivative.derivative)
        candidate = math.nan
        if d_jr_d_er != 0.0:
            candidate = b - fb / d_jr_d_er
        use_bisection = (
            (not math.isfinite(candidate))
            or candidate <= lo
            or candidate >= hi
        )
        if use_bisection:
            candidate = 0.5 * (lo + hi)
            fallback_count += 1
        step = abs(candidate - b)
        b = float(candidate)
        fb = _evaluate(problem, iterations, b, "newton_bisection" if use_bisection else "newton")

        if abs(fb) < float(problem.current_tolerance):
            return AmbipolarResult(
                converged=True,
                method="safeguarded_newton",
                root_er=b,
                root_radial_current=fb,
                iterations=tuple(iterations),
                status="converged",
                root_type=_root_type(b),
                metadata={
                    "convergence": "radial_current",
                    "derivative_count": len(derivative_records),
                    "fallback_count": fallback_count,
                    "last_derivative": d_jr_d_er,
                },
            )
        if step <= float(problem.step_tolerance) or abs(hi - lo) <= float(problem.step_tolerance):
            return AmbipolarResult(
                converged=True,
                method="safeguarded_newton",
                root_er=b,
                root_radial_current=fb,
                iterations=tuple(iterations),
                status="converged",
                root_type=_root_type(b),
                metadata={
                    "convergence": "step",
                    "derivative_count": len(derivative_records),
                    "fallback_count": fallback_count,
                    "last_derivative": d_jr_d_er,
                },
            )

        if _same_fortran_sign(flo, fb):
            lo = b
            flo = fb
        else:
            hi = b

    return AmbipolarResult(
        converged=False,
        method="safeguarded_newton",
        root_er=b,
        root_radial_current=fb,
        iterations=tuple(iterations),
        status="max_evaluations",
        message="The derivative-assisted Er search did not converge within max_evaluations.",
        root_type=_root_type(b),
        metadata={
            "derivative_count": len(derivative_records),
            "fallback_count": fallback_count,
            "last_step": float(step) if math.isfinite(step) else None,
        },
    )


def solve_ambipolar_newton(
    problem: AmbipolarProblem,
    evaluate_derivative: RadialCurrentDerivativeEvaluator,
    *,
    derivative_source: str = "external",
) -> AmbipolarResult:
    """Solve ambipolarity with strict Newton steps and trust-region failure certificates."""

    iterations: list[AmbipolarIteration] = []
    b = float(problem.er_initial)
    fb = _evaluate(problem, iterations, b, "initial")
    if abs(fb) < float(problem.current_tolerance):
        return AmbipolarResult(
            converged=True,
            method="newton",
            root_er=b,
            root_radial_current=fb,
            iterations=tuple(iterations),
            status="converged",
            root_type=_root_type(b),
            metadata={"convergence": "radial_current"},
        )

    derivative_records: list[RadialCurrentDerivativeResult] = []
    last_step = math.inf
    for _ in range(2, int(problem.max_evaluations) + 1):
        try:
            derivative = _evaluate_derivative(evaluate_derivative, er=b, source=derivative_source)
        except FloatingPointError as exc:
            return AmbipolarResult(
                converged=False,
                method="newton",
                root_er=b,
                root_radial_current=fb,
                iterations=tuple(iterations),
                status="invalid_derivative",
                message=str(exc),
                root_type=_root_type(b),
                metadata={"derivative_count": len(derivative_records)},
            )
        derivative_records.append(derivative)
        d_jr_d_er = float(derivative.derivative)
        if d_jr_d_er == 0.0:
            return AmbipolarResult(
                converged=False,
                method="newton",
                root_er=b,
                root_radial_current=fb,
                iterations=tuple(iterations),
                status="zero_derivative",
                message="Pure Newton cannot proceed with dJr/dEr=0.",
                root_type=_root_type(b),
                metadata={"derivative_count": len(derivative_records)},
            )
        step = -fb / d_jr_d_er
        if not math.isfinite(step):
            return AmbipolarResult(
                converged=False,
                method="newton",
                root_er=b,
                root_radial_current=fb,
                iterations=tuple(iterations),
                status="invalid_step",
                message="Pure Newton produced a non-finite step.",
                root_type=_root_type(b),
                metadata={"derivative_count": len(derivative_records)},
            )
        last_step = abs(float(step))
        b = float(b + step)
        if b < float(problem.er_min) or b > float(problem.er_max):
            return AmbipolarResult(
                converged=False,
                method="newton",
                root_er=b,
                root_radial_current=None,
                iterations=tuple(iterations),
                status="out_of_bounds",
                message="Pure Newton left the configured Er trust region.",
                root_type=_root_type(b),
                metadata={
                    "derivative_count": len(derivative_records),
                    "last_derivative": d_jr_d_er,
                    "last_step": float(last_step),
                },
            )
        fb = _evaluate(problem, iterations, b, "newton")
        if abs(fb) < float(problem.current_tolerance):
            return AmbipolarResult(
                converged=True,
                method="newton",
                root_er=b,
                root_radial_current=fb,
                iterations=tuple(iterations),
                status="converged",
                root_type=_root_type(b),
                metadata={
                    "convergence": "radial_current",
                    "derivative_count": len(derivative_records),
                    "last_derivative": d_jr_d_er,
                    "last_step": float(last_step),
                },
            )
        if last_step <= float(problem.step_tolerance):
            return AmbipolarResult(
                converged=True,
                method="newton",
                root_er=b,
                root_radial_current=fb,
                iterations=tuple(iterations),
                status="converged",
                root_type=_root_type(b),
                metadata={
                    "convergence": "step",
                    "derivative_count": len(derivative_records),
                    "last_derivative": d_jr_d_er,
                    "last_step": float(last_step),
                },
            )

    return AmbipolarResult(
        converged=False,
        method="newton",
        root_er=b,
        root_radial_current=fb,
        iterations=tuple(iterations),
        status="max_evaluations",
        message="The pure Newton Er search did not converge within max_evaluations.",
        root_type=_root_type(b),
        metadata={
            "derivative_count": len(derivative_records),
            "last_step": float(last_step) if math.isfinite(last_step) else None,
        },
    )


def safeguarded_newton_ambipolar_root(
    evaluate_radial_current: RadialCurrentEvaluator,
    evaluate_derivative: RadialCurrentDerivativeEvaluator,
    *,
    er_min: float,
    er_max: float,
    er_initial: float = 0.0,
    max_evaluations: int = 20,
    current_tolerance: float = 1.0e-10,
    step_tolerance: float = 1.0e-8,
    metadata: Mapping[str, Any] | None = None,
) -> AmbipolarResult:
    """Convenience wrapper for derivative-assisted safeguarded Newton."""

    return solve_ambipolar_safeguarded_newton(
        AmbipolarProblem(
            evaluate_radial_current=evaluate_radial_current,
            er_min=er_min,
            er_max=er_max,
            er_initial=er_initial,
            max_evaluations=max_evaluations,
            current_tolerance=current_tolerance,
            step_tolerance=step_tolerance,
            metadata=metadata,
        ),
        evaluate_derivative,
    )


def newton_ambipolar_root(
    evaluate_radial_current: RadialCurrentEvaluator,
    evaluate_derivative: RadialCurrentDerivativeEvaluator,
    *,
    er_min: float,
    er_max: float,
    er_initial: float = 0.0,
    max_evaluations: int = 20,
    current_tolerance: float = 1.0e-10,
    step_tolerance: float = 1.0e-8,
    metadata: Mapping[str, Any] | None = None,
) -> AmbipolarResult:
    """Convenience wrapper for strict derivative-only Newton."""

    return solve_ambipolar_newton(
        AmbipolarProblem(
            evaluate_radial_current=evaluate_radial_current,
            er_min=er_min,
            er_max=er_max,
            er_initial=er_initial,
            max_evaluations=max_evaluations,
            current_tolerance=current_tolerance,
            step_tolerance=step_tolerance,
            metadata=metadata,
        ),
        evaluate_derivative,
    )


def finite_difference_radial_current_derivative(
    evaluate_radial_current: RadialCurrentEvaluator,
    *,
    er: float,
    step: float,
    scheme: str = "centered",
) -> RadialCurrentDerivativeResult:
    """Estimate ``dJr/dEr`` with a controlled finite-difference stencil."""

    h = float(step)
    if h <= 0.0:
        raise ValueError("step must be positive.")
    scheme_norm = str(scheme).strip().lower()
    evaluations: list[AmbipolarIteration] = []

    def eval_one(value: float, stage: str) -> float:
        current = float(evaluate_radial_current(float(value)))
        evaluations.append(
            AmbipolarIteration(
                index=len(evaluations) + 1,
                er=float(value),
                radial_current=current,
                stage=stage,
            )
        )
        return current

    if scheme_norm in {"centered", "central"}:
        fp = eval_one(float(er) + h, "finite_difference_plus")
        fm = eval_one(float(er) - h, "finite_difference_minus")
        derivative = (fp - fm) / (2.0 * h)
        scheme_out = "centered"
    elif scheme_norm == "forward":
        f0 = eval_one(float(er), "finite_difference_base")
        fp = eval_one(float(er) + h, "finite_difference_plus")
        derivative = (fp - f0) / h
        scheme_out = "forward"
    elif scheme_norm == "backward":
        f0 = eval_one(float(er), "finite_difference_base")
        fm = eval_one(float(er) - h, "finite_difference_minus")
        derivative = (f0 - fm) / h
        scheme_out = "backward"
    else:
        raise ValueError("scheme must be 'centered', 'forward', or 'backward'.")

    return RadialCurrentDerivativeResult(
        er=float(er),
        derivative=float(derivative),
        step=h,
        scheme=scheme_out,
        evaluations=tuple(evaluations),
    )


def implicit_linear_radial_current_derivative(
    *,
    er: float,
    matrix: Any,
    rhs: Any,
    matrix_derivative: Any,
    rhs_derivative: Any,
    radial_current_vector: Any,
    radial_current_vector_derivative: Any | None = None,
    radial_current_offset: float = 0.0,
    radial_current_offset_derivative: float = 0.0,
    finite_difference_radial_current: Callable[[float], float] | None = None,
    finite_difference_step: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RadialCurrentDerivativeResult:
    """Return an implicit/adjoint derivative certificate for ``dJr/dEr``.

    This is the ambipolar-specific adapter around
    :func:`sfincs_jax.sensitivity.implicit_linear_observable_derivative`. The
    full RHSMode-1 wiring supplies the true linearized operator and
    ``Er``-derivative terms; this adapter keeps the root-solver contract stable.
    """

    from ..sensitivity import implicit_linear_observable_derivative  # noqa: PLC0415

    result = implicit_linear_observable_derivative(
        matrix=matrix,
        rhs=rhs,
        matrix_derivative=matrix_derivative,
        rhs_derivative=rhs_derivative,
        observable_vector=radial_current_vector,
        observable_vector_derivative=radial_current_vector_derivative,
        observable_offset=radial_current_offset,
        observable_offset_derivative=radial_current_offset_derivative,
        parameter=float(er),
        finite_difference_observable=finite_difference_radial_current,
        finite_difference_step=finite_difference_step,
        metadata=metadata,
    )
    return _radial_current_derivative_from_linear_certificate(er=er, result=result)


def implicit_linear_radial_current_derivative_from_builder(
    build_system: Callable[[float], Any],
    *,
    er: float,
    solve: Callable[[Any, Any], Any] | None = None,
    transpose_solve: Callable[[Any, Any], Any] | None = None,
    finite_difference_step: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RadialCurrentDerivativeResult:
    """Return ``dJr/dEr`` from a concrete linear-observable system builder.

    The builder should emit the true RHSMode-1 operator, RHS, radial-current
    observable, and their ``Er`` derivatives at the requested electric field.
    This keeps the ambipolar solver independent from the monolithic driver while
    preserving a finite-difference certificate against the same builder.
    """

    from ..sensitivity import implicit_linear_observable_derivative_from_builder  # noqa: PLC0415

    result = implicit_linear_observable_derivative_from_builder(
        build_system,
        parameter=float(er),
        solve=solve,
        transpose_solve=transpose_solve,
        finite_difference_step=finite_difference_step,
        metadata=metadata,
    )
    return _radial_current_derivative_from_linear_certificate(er=er, result=result)


def implicit_matrix_free_radial_current_derivative(
    system: Any,
    *,
    er: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RadialCurrentDerivativeResult:
    """Return ``dJr/dEr`` from a matrix-free linear-observable system."""

    from ..sensitivity import implicit_matrix_free_linear_observable_derivative  # noqa: PLC0415

    result = implicit_matrix_free_linear_observable_derivative(system, metadata=metadata)
    er_value = float(result.parameter if er is None else er)
    return _radial_current_derivative_from_linear_certificate(er=er_value, result=result)


def implicit_matrix_free_radial_current_derivative_from_builder(
    build_system: Callable[[float], Any],
    *,
    er: float,
    finite_difference_step: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RadialCurrentDerivativeResult:
    """Return ``dJr/dEr`` from a production-style matrix-free system builder."""

    from ..sensitivity import implicit_matrix_free_linear_observable_derivative_from_builder  # noqa: PLC0415

    result = implicit_matrix_free_linear_observable_derivative_from_builder(
        build_system,
        parameter=float(er),
        finite_difference_step=finite_difference_step,
        metadata=metadata,
    )
    return _radial_current_derivative_from_linear_certificate(er=er, result=result)


def matrix_free_radial_current_derivative_provider(
    build_system: Callable[[float], Any],
    *,
    finite_difference_step: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RadialCurrentDerivativeEvaluator:
    """Return a root-solver derivative provider backed by matrix-free systems."""

    def evaluate(er: float) -> RadialCurrentDerivativeResult:
        return implicit_matrix_free_radial_current_derivative_from_builder(
            build_system,
            er=float(er),
            finite_difference_step=finite_difference_step,
            metadata=metadata,
        )

    return evaluate


def dense_rhs1_vm_radial_current_linear_observable_system(
    *,
    op: Any,
    op_plus: Any,
    op_minus: Any,
    parameter: float,
    derivative_step: float,
    radial_coordinate: str = "psiHat",
    psi_a_hat: float | None = None,
    a_hat: float | None = None,
    r_n: float | None = None,
    max_size: int = 512,
    observable_chunk_size: int = 128,
    include_jacobian_terms: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> Any:
    """Build a dense RHSMode-1 radial-current derivative system for small gates.

    This validation bridge uses real SFINCS_JAX operators and diagnostics but is
    intentionally size-limited. Production ambipolar derivatives should replace
    the dense assembly with sparse/matrix-free operator derivatives after this
    small-deck gate pins the signs, normalization, and coordinate conversion.
    """

    h = float(derivative_step)
    if h <= 0.0:
        raise ValueError("derivative_step must be positive.")
    total_size = int(getattr(op, "total_size"))
    if total_size <= 0:
        raise ValueError("op.total_size must be positive.")
    if total_size > int(max_size):
        raise ValueError(f"dense RHSMode-1 derivative builder refused size {total_size} > max_size {int(max_size)}.")
    for name, candidate in (("op_plus", op_plus), ("op_minus", op_minus)):
        if int(getattr(candidate, "total_size")) != total_size:
            raise ValueError(f"{name}.total_size must match op.total_size.")

    import jax.numpy as jnp  # noqa: PLC0415

    from ..sensitivity import LinearObservableSystem  # noqa: PLC0415
    from ..solver import assemble_dense_matrix_from_matvec  # noqa: PLC0415
    from sfincs_jax.operators.profile_response.system import apply_v3_full_system_operator_cached, rhs_v3_full_system_jit  # noqa: PLC0415
    from .transport_matrix.diagnostics import radial_current_vm_observable_vector  # noqa: PLC0415

    def assemble_matrix(operator: Any) -> Any:
        return assemble_dense_matrix_from_matvec(
            matvec=lambda state: apply_v3_full_system_operator_cached(
                operator,
                state,
                include_jacobian_terms=bool(include_jacobian_terms),
            ),
            n=total_size,
            dtype=jnp.float64,
        )

    matrix = assemble_matrix(op)
    matrix_plus = assemble_matrix(op_plus)
    matrix_minus = assemble_matrix(op_minus)
    rhs = rhs_v3_full_system_jit(op)
    rhs_plus = rhs_v3_full_system_jit(op_plus)
    rhs_minus = rhs_v3_full_system_jit(op_minus)
    observable_vector, observable_offset = radial_current_vm_observable_vector(
        op,
        radial_coordinate=radial_coordinate,
        psi_a_hat=psi_a_hat,
        a_hat=a_hat,
        r_n=r_n,
        chunk_size=int(observable_chunk_size),
    )
    observable_vector_plus, observable_offset_plus = radial_current_vm_observable_vector(
        op_plus,
        radial_coordinate=radial_coordinate,
        psi_a_hat=psi_a_hat,
        a_hat=a_hat,
        r_n=r_n,
        chunk_size=int(observable_chunk_size),
    )
    observable_vector_minus, observable_offset_minus = radial_current_vm_observable_vector(
        op_minus,
        radial_coordinate=radial_coordinate,
        psi_a_hat=psi_a_hat,
        a_hat=a_hat,
        r_n=r_n,
        chunk_size=int(observable_chunk_size),
    )
    merged_metadata = {
        "builder": "dense_rhs1_vm_radial_current",
        "total_size": total_size,
        "radial_coordinate": str(radial_coordinate),
        "derivative_step": h,
        **dict(metadata or {}),
    }
    return LinearObservableSystem(
        parameter=float(parameter),
        matrix=matrix,
        rhs=rhs,
        matrix_derivative=(matrix_plus - matrix_minus) / (2.0 * h),
        rhs_derivative=(rhs_plus - rhs_minus) / (2.0 * h),
        observable_vector=observable_vector,
        observable_vector_derivative=(observable_vector_plus - observable_vector_minus) / (2.0 * h),
        observable_offset=float(observable_offset),
        observable_offset_derivative=(float(observable_offset_plus) - float(observable_offset_minus)) / (2.0 * h),
        metadata=merged_metadata,
    )


def operator_tangent_from_centered_difference(
    op_plus: Any,
    op_minus: Any,
    derivative_step: float,
) -> Any:
    """Return a JAX-compatible pytree tangent from two same-shape operators."""

    h = float(derivative_step)
    if h <= 0.0:
        raise ValueError("derivative_step must be positive.")

    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    def tangent_leaf(plus: Any, minus: Any) -> Any:
        plus_array = jnp.asarray(plus)
        if jnp.issubdtype(plus_array.dtype, jnp.inexact):
            return (plus_array - jnp.asarray(minus)) / (2.0 * h)
        return jnp.zeros(plus_array.shape, dtype=jax.dtypes.float0)

    return jax.tree_util.tree_map(tangent_leaf, op_plus, op_minus)


def dphi_hat_dpsi_hat_er_derivative_from_namelist(nml: Any) -> float:
    """Return ``d(dPhiHat/dpsiHat)/dEr`` using the v3 radial conversion."""

    from sfincs_jax.operators.profile_response.fblock import _dphi_hat_dpsi_hat_from_er  # noqa: PLC0415

    return float(
        _dphi_hat_dpsi_hat_from_er(nml=nml, er=1.0)
        - _dphi_hat_dpsi_hat_from_er(nml=nml, er=0.0)
    )


def er_operator_tangent_from_dphi_hat_dpsi_hat_derivative(
    op: Any,
    dphi_hat_dpsi_hat_derivative: float,
) -> Any:
    """Build a fixed-shape operator tangent for an ``Er`` perturbation.

    The helper updates existing ``dphi_hat_dpsi_hat`` leaves in the full-system
    operator and f-block suboperators. It intentionally does not activate
    branches that were absent at the base state, so callers should build base
    operators away from branch-changing ``Er=0`` points until the fixed-shape
    zero-Er operator policy is promoted.
    """

    import jax.numpy as jnp  # noqa: PLC0415

    tangent = operator_tangent_from_centered_difference(op, op, 1.0)
    derivative = jnp.asarray(float(dphi_hat_dpsi_hat_derivative), dtype=jnp.float64)

    def replace_dphi_leaf(candidate: Any) -> Any:
        if candidate is None or not hasattr(candidate, "dphi_hat_dpsi_hat"):
            return candidate
        return replace(candidate, dphi_hat_dpsi_hat=derivative)

    fblock = getattr(tangent, "fblock", None)
    if fblock is not None:
        fblock = replace(
            fblock,
            exb_theta=replace_dphi_leaf(getattr(fblock, "exb_theta", None)),
            exb_zeta=replace_dphi_leaf(getattr(fblock, "exb_zeta", None)),
            er_xidot=replace_dphi_leaf(getattr(fblock, "er_xidot", None)),
            er_xdot=replace_dphi_leaf(getattr(fblock, "er_xdot", None)),
        )
    return replace(tangent, fblock=fblock, dphi_hat_dpsi_hat=derivative)


def _namelist_with_er(nml: Any, er: float) -> Any:
    """Return a shallow namelist copy with ``physicsParameters/ER`` updated."""

    from ..namelist import Namelist  # noqa: PLC0415

    groups = {group: dict(values) for group, values in nml.groups.items()}
    indexed = {
        group: {key: dict(values) for key, values in group_values.items()}
        for group, group_values in nml.indexed.items()
    }
    physics = groups.setdefault("physicsparameters", {})
    physics["ER"] = float(er)
    return Namelist(
        groups=groups,
        indexed=indexed,
        source_path=nml.source_path,
        source_text=nml.source_text,
    )


def _dense_validation_linear_algebra_for_operator(
    op: Any,
    *,
    max_size: int,
    include_jacobian_terms: bool,
) -> tuple[Callable[[Any], Any], Callable[[Any], Any], Callable[[Any], Any], dict[str, Any]]:
    """Return dense solve/transpose closures for small validation operators."""

    total_size = int(getattr(op, "total_size"))

    import jax.numpy as jnp  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    from sfincs_jax.operators.profile_response.compressed_layout import build_rhs1_compressed_pitch_layout  # noqa: PLC0415
    from ..solver import assemble_dense_matrix_from_matvec  # noqa: PLC0415
    from sfincs_jax.operators.profile_response.system import apply_v3_full_system_operator_cached  # noqa: PLC0415

    layout = build_rhs1_compressed_pitch_layout(op)
    active_idx_np = np.asarray(layout.active_full_indices, dtype=np.int32)
    use_active_dof = bool(active_idx_np.size < total_size)
    linear_size = int(active_idx_np.size) if use_active_dof else total_size
    if linear_size > int(max_size):
        raise ValueError(
            "bounded dense validation refused RHSMode-1 size "
            f"{linear_size} > max_dense_size {int(max_size)}."
        )
    active_idx = jnp.asarray(active_idx_np, dtype=jnp.int32)

    def reduce_full(vector: Any) -> Any:
        arr = jnp.asarray(vector, dtype=jnp.float64).reshape((total_size,))
        return arr[active_idx] if use_active_dof else arr

    def expand_reduced(vector: Any) -> Any:
        arr = jnp.asarray(vector, dtype=jnp.float64).reshape((linear_size,))
        if not use_active_dof:
            return arr
        out = jnp.zeros((total_size,), dtype=jnp.float64)
        return out.at[active_idx].set(arr, unique_indices=True)

    def matvec_reduced(state: Any) -> Any:
        full_state = expand_reduced(state)
        full_out = apply_v3_full_system_operator_cached(
            op,
            full_state,
            include_jacobian_terms=bool(include_jacobian_terms),
        )
        return reduce_full(full_out)

    matrix = assemble_dense_matrix_from_matvec(
        matvec=matvec_reduced,
        n=linear_size,
        dtype=jnp.float64,
    )

    def solve_dense_system(a: Any, rhs: Any) -> Any:
        rhs_arr = jnp.asarray(rhs, dtype=jnp.float64).reshape((-1,))
        x = jnp.linalg.solve(a, rhs_arr)
        residual = jnp.linalg.norm(a @ x - rhs_arr)
        rhs_norm = jnp.linalg.norm(rhs_arr)
        if bool(jnp.all(jnp.isfinite(x))) and float(residual) <= max(1.0e-10, 1.0e-8 * float(rhs_norm)):
            return x
        # Some Fortran-compatible derivative-assisted RHSMode=1 decks resolve
        # to constraintScheme=1 and are intentionally rank deficient. PETSc
        # handles this through its nullspace-compatible factor/KSP policy; the
        # bounded validation path uses the minimum-norm least-squares solution.
        x_np = np.linalg.lstsq(np.asarray(a), np.asarray(rhs_arr), rcond=None)[0]
        return jnp.asarray(x_np, dtype=jnp.float64)

    return (
        lambda rhs: expand_reduced(solve_dense_system(matrix, reduce_full(rhs))),
        lambda rhs: expand_reduced(solve_dense_system(matrix.T, reduce_full(rhs))),
        lambda vector: expand_reduced(matrix.T @ reduce_full(vector)),
        {
            "linear_algebra": (
                "bounded_dense_active_validation"
                if use_active_dof
                else "bounded_dense_validation"
            ),
            "active_dof": use_active_dof,
            "active_size": linear_size,
            "full_size": total_size,
            "nxi_for_x": tuple(int(v) for v in np.asarray(layout.nxi_for_x, dtype=np.int64)),
        },
    )


def _radial_current_conversion_kwargs_from_namelist(nml: Any) -> dict[str, float]:
    """Infer Fortran v3 radial-current conversion factors from a namelist."""

    from ..boozer_bc import read_boozer_bc_header, selected_r_n_from_bc  # noqa: PLC0415
    from ..input_compat import (  # noqa: PLC0415
        effective_equilibrium_file,
        effective_psi_a_hat,
        effective_psi_n_wish,
    )
    from ..paths import resolve_existing_path  # noqa: PLC0415
    from ..vmec_wout import psi_a_hat_from_wout, read_vmec_wout, vmec_interpolation  # noqa: PLC0415

    def _get_int(group: Mapping[str, Any], key: str, default: int) -> int:
        value = group.get(key.upper(), default)
        if isinstance(value, list):
            value = value[0] if value else default
        return int(value)

    geom_params = nml.group("geometryParameters")
    phys_params = nml.group("physicsParameters")
    geometry_scheme = _get_int(geom_params, "geometryScheme", -1)
    if geometry_scheme == 1:
        psi_a_hat = effective_psi_a_hat(geom_params=geom_params, phys_params=phys_params, default=0.15596)
        a_hat = float(geom_params.get("AHAT", 0.5585))
        r_n = math.sqrt(
            effective_psi_n_wish(
                geom_params=geom_params,
                default_r_n=0.5,
                psi_a_hat=psi_a_hat,
                a_hat=a_hat,
            )
        )
    elif geometry_scheme == 2:
        a_hat = 0.5585
        psi_a_hat = (a_hat * a_hat) / 2.0
        r_n = 0.5
    elif geometry_scheme == 4:
        psi_a_hat = -0.384935
        a_hat = 0.5109
        r_n = 0.5
    elif geometry_scheme in {11, 12}:
        eq = effective_equilibrium_file(geom_params=geom_params)
        if eq is None:
            raise ValueError("geometryScheme=11/12 requires equilibriumFile for radial-current conversion.")
        base_dir = nml.source_path.parent if nml.source_path is not None else None
        repo_root = Path(__file__).resolve().parents[2]
        p = resolve_existing_path(
            str(eq),
            base_dir=base_dir,
            extra_search_dirs=(repo_root / "tests" / "ref", repo_root / "sfincs_jax" / "data" / "equilibria"),
        ).path
        header = read_boozer_bc_header(path=str(p), geometry_scheme=int(geometry_scheme))
        psi_a_hat = float(header.psi_a_hat)
        a_hat = float(header.a_hat)
        psi_n_wish = effective_psi_n_wish(geom_params=geom_params, default_r_n=0.5)
        r_n_wish = math.sqrt(float(psi_n_wish))
        vmecradial_option = _get_int(
            geom_params,
            "VMECRadialOption",
            _get_int(geom_params, "VMECRADIALOPTION", 1),
        )
        r_n = selected_r_n_from_bc(
            path=str(p),
            geometry_scheme=int(geometry_scheme),
            r_n_wish=r_n_wish,
            vmecradial_option=int(vmecradial_option),
        )
    elif geometry_scheme == 5:
        eq = effective_equilibrium_file(geom_params=geom_params)
        if eq is None:
            raise ValueError("geometryScheme=5 requires equilibriumFile for radial-current conversion.")
        base_dir = nml.source_path.parent if nml.source_path is not None else None
        repo_root = Path(__file__).resolve().parents[2]
        p = resolve_existing_path(
            str(eq),
            base_dir=base_dir,
            extra_search_dirs=(repo_root / "tests" / "ref", repo_root / "sfincs_jax" / "data" / "equilibria"),
        ).path
        w = read_vmec_wout(p)
        psi_a_hat = float(psi_a_hat_from_wout(w))
        a_hat = float(w.aminor_p)
        psi_n_wish = effective_psi_n_wish(
            geom_params=geom_params,
            default_r_n=0.5,
            psi_a_hat=psi_a_hat,
            a_hat=a_hat,
        )
        vmecradial_option = _get_int(
            geom_params,
            "VMECRadialOption",
            _get_int(geom_params, "VMECRADIALOPTION", 1),
        )
        interp = vmec_interpolation(w=w, psi_n_wish=psi_n_wish, vmec_radial_option=vmecradial_option)
        r_n = math.sqrt(float(interp.psi_n))
    else:
        raise NotImplementedError(
            "Fortran-compatible radial-current conversion is not implemented for "
            f"geometryScheme={geometry_scheme}."
        )
    return {"psi_a_hat": float(psi_a_hat), "a_hat": float(a_hat), "r_n": float(r_n)}


def rhsmode1_radial_current_response_from_namelist(
    *,
    nml: Any,
    derivative_step: float = 1.0e-5,
    finite_difference_step: float | None = None,
    radial_coordinate: str = "rN",
    psi_a_hat: float | None = None,
    a_hat: float | None = None,
    r_n: float | None = None,
    identity_shift: float = 0.0,
    keep_zero_er_terms: bool = True,
    max_dense_size: int = 512,
    observable_chunk_size: int = 128,
    include_jacobian_terms: bool = True,
    use_analytic_er_tangent: bool = True,
    linear_algebra_factory: Callable[
        [Any],
        tuple[Callable[[Any], Any], Callable[[Any], Any], Callable[[Any], Any]],
    ]
    | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RHSMode1RadialCurrentResponse:
    """Build a namelist-backed RHSMode-1 radial-current response.

    The default backend uses bounded dense factors only for small validation
    decks. Production callers should provide ``linear_algebra_factory`` that
    returns solve, transpose-solve, and transpose-action closures for the
    selected sparse or matrix-free solver path.
    """

    h = float(derivative_step)
    if h <= 0.0:
        raise ValueError("derivative_step must be positive.")
    fd_step = h if finite_difference_step is None else float(finite_difference_step)
    if fd_step <= 0.0:
        raise ValueError("finite_difference_step must be positive.")

    from ..namelist import read_sfincs_input  # noqa: PLC0415
    from sfincs_jax.operators.profile_response.system import full_system_operator_from_namelist  # noqa: PLC0415

    nml_base = read_sfincs_input(nml) if isinstance(nml, (str, Path)) else nml
    rhs_mode = int(nml_base.group("general").get("RHSMODE", 1))
    if rhs_mode != 1:
        raise ValueError("rhsmode1_radial_current_response_from_namelist requires RHSMode=1.")
    dphi_d_er = dphi_hat_dpsi_hat_er_derivative_from_namelist(nml_base)
    source_path = getattr(nml_base, "source_path", None)
    coordinate = str(radial_coordinate)
    if coordinate.strip().lower() not in {"psihat", "psi_hat"} and (
        psi_a_hat is None or a_hat is None or r_n is None
    ):
        inferred_conversion = _radial_current_conversion_kwargs_from_namelist(nml_base)
        psi_a_hat = inferred_conversion["psi_a_hat"] if psi_a_hat is None else psi_a_hat
        a_hat = inferred_conversion["a_hat"] if a_hat is None else a_hat
        r_n = inferred_conversion["r_n"] if r_n is None else r_n

    def build_operator(er: float) -> Any:
        return full_system_operator_from_namelist(
            nml=_namelist_with_er(nml_base, float(er)),
            identity_shift=identity_shift,
            keep_zero_er_terms=bool(keep_zero_er_terms),
        )

    def build_system(er: float) -> Any:
        er_value = float(er)
        op = build_operator(er_value)
        op_plus = build_operator(er_value + h)
        op_minus = build_operator(er_value - h)
        if linear_algebra_factory is None:
            solve, transpose_solve, transpose_apply, linear_algebra_metadata = _dense_validation_linear_algebra_for_operator(
                op,
                max_size=int(max_dense_size),
                include_jacobian_terms=bool(include_jacobian_terms),
            )
        else:
            solve, transpose_solve, transpose_apply = linear_algebra_factory(op)
            linear_algebra_metadata = {"linear_algebra": "caller_supplied"}
        operator_tangent = (
            er_operator_tangent_from_dphi_hat_dpsi_hat_derivative(op, dphi_d_er)
            if bool(use_analytic_er_tangent)
            else None
        )
        return matrix_free_rhs1_vm_radial_current_linear_observable_system(
            op=op,
            op_plus=op_plus,
            op_minus=op_minus,
            parameter=er_value,
            derivative_step=h,
            solve=solve,
            transpose_solve=transpose_solve,
            transpose_apply=transpose_apply,
            operator_tangent=operator_tangent,
            radial_coordinate=radial_coordinate,
            psi_a_hat=psi_a_hat,
            a_hat=a_hat,
            r_n=r_n,
            observable_chunk_size=int(observable_chunk_size),
            include_jacobian_terms=bool(include_jacobian_terms),
            metadata={
                "builder": "rhsmode1_radial_current_response_from_namelist",
                "source_path": None if source_path is None else str(source_path),
                "derivative_step": h,
                "keep_zero_er_terms": bool(keep_zero_er_terms),
                "use_analytic_er_tangent": bool(use_analytic_er_tangent),
                "radial_coordinate": coordinate,
                "psi_a_hat": None if psi_a_hat is None else float(psi_a_hat),
                "a_hat": None if a_hat is None else float(a_hat),
                "r_n": None if r_n is None else float(r_n),
                **linear_algebra_metadata,
                **dict(metadata or {}),
            },
        )

    return RHSMode1RadialCurrentResponse(
        build_system=build_system,
        finite_difference_step=fd_step,
        metadata={
            "response_builder": "rhsmode1_radial_current_response_from_namelist",
            "source_path": None if source_path is None else str(source_path),
            "radial_coordinate": coordinate,
            **dict(metadata or {}),
        },
    )


def solve_rhsmode1_ambipolar_from_namelist(
    *,
    nml: Any,
    derivative_step: float = 1.0e-5,
    finite_difference_step: float | None = None,
    radial_coordinate: str = "rN",
    psi_a_hat: float | None = None,
    a_hat: float | None = None,
    r_n: float | None = None,
    identity_shift: float = 0.0,
    keep_zero_er_terms: bool = True,
    max_dense_size: int = 512,
    observable_chunk_size: int = 128,
    include_jacobian_terms: bool = True,
    use_analytic_er_tangent: bool = True,
    linear_algebra_factory: Callable[
        [Any],
        tuple[Callable[[Any], Any], Callable[[Any], Any], Callable[[Any], Any]],
    ]
    | None = None,
    er_min: float | None = None,
    er_max: float | None = None,
    er_initial: float | None = None,
    max_evaluations: int | None = None,
    current_tolerance: float | None = None,
    step_tolerance: float | None = None,
    ambipolar_solve_option: int | None = None,
    validate_fortran_compatibility: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> AmbipolarResult:
    """Solve a Fortran-compatible RHSMode-1 ambipolar namelist in-process.

    This helper is the small-deck promotion gate for ``ambipolarSolveOption``
    1--3.  It wires the namelist-backed radial-current response into the same
    Brent, safeguarded Newton, and pure Newton policies used by the public
    ambipolar API.  Production callers should provide ``linear_algebra_factory``
    so the response can reuse sparse or matrix-free factors instead of the
    bounded dense validation backend.
    """

    from ..namelist import read_sfincs_input  # noqa: PLC0415

    nml_base = read_sfincs_input(nml) if isinstance(nml, (str, Path)) else nml
    general = nml_base.group("general")
    physics = nml_base.group("physicsParameters")
    option = int(
        _first_scalar(
            ambipolar_solve_option
            if ambipolar_solve_option is not None
            else general.get("AMBIPOLARSOLVEOPTION"),
            2,
        )
    )
    if validate_fortran_compatibility:
        errors = validate_fortran_v3_ambipolar_constraints(nml_base, option=option)
        if errors:
            joined = "; ".join(errors)
            raise ValueError(f"RHSMode-1 ambipolar namelist is not Fortran-v3 compatible: {joined}")

    response = rhsmode1_radial_current_response_from_namelist(
        nml=nml_base,
        derivative_step=derivative_step,
        finite_difference_step=finite_difference_step,
        radial_coordinate=radial_coordinate,
        psi_a_hat=psi_a_hat,
        a_hat=a_hat,
        r_n=r_n,
        identity_shift=identity_shift,
        keep_zero_er_terms=keep_zero_er_terms,
        max_dense_size=max_dense_size,
        observable_chunk_size=observable_chunk_size,
        include_jacobian_terms=include_jacobian_terms,
        use_analytic_er_tangent=use_analytic_er_tangent,
        linear_algebra_factory=linear_algebra_factory,
        metadata=metadata,
    )

    current_cache: dict[float, float] = {}

    def evaluate_current(er: float) -> float:
        key = float(er)
        if key not in current_cache:
            current_cache[key] = float(response.radial_current(key))
        return current_cache[key]

    problem_metadata = {
        "builder": "solve_rhsmode1_ambipolar_from_namelist",
        "source_path": None if getattr(nml_base, "source_path", None) is None else str(nml_base.source_path),
        "ambipolar_solve_option": option,
        "radial_current_response": dict(response.metadata),
        **dict(metadata or {}),
    }
    problem = AmbipolarProblem(
        evaluate_radial_current=evaluate_current,
        er_min=float(_first_scalar(er_min if er_min is not None else general.get("ER_MIN"), -1.0)),
        er_max=float(_first_scalar(er_max if er_max is not None else general.get("ER_MAX"), 1.0)),
        er_initial=float(_first_scalar(er_initial if er_initial is not None else physics.get("ER"), 0.0)),
        max_evaluations=int(
            _first_scalar(max_evaluations if max_evaluations is not None else general.get("NER_AMBIPOLARSOLVE"), 20)
        ),
        current_tolerance=float(
            _first_scalar(
                current_tolerance if current_tolerance is not None else general.get("ER_SEARCH_TOLERANCE_F"),
                1.0e-10,
            )
        ),
        step_tolerance=float(
            _first_scalar(
                step_tolerance if step_tolerance is not None else general.get("ER_SEARCH_TOLERANCE_DX"),
                1.0e-8,
            )
        ),
        metadata=problem_metadata,
    )

    if option == 1:
        result = solve_ambipolar_safeguarded_newton(
            problem,
            response.derivative,
            derivative_source="rhsmode1_implicit_active",
        )
    elif option == 2:
        result = solve_ambipolar_brent(problem)
    elif option == 3:
        result = solve_ambipolar_newton(
            problem,
            response.derivative,
            derivative_source="rhsmode1_implicit_active",
        )
    else:
        raise ValueError(f"Unsupported ambipolarSolveOption={option}; expected 1, 2, or 3.")

    return replace(
        result,
        metadata={
            **problem_metadata,
            **dict(result.metadata),
            "current_cache_size": len(current_cache),
        },
    )


def matrix_free_rhs1_vm_radial_current_linear_observable_system(
    *,
    op: Any,
    op_plus: Any,
    op_minus: Any,
    parameter: float,
    derivative_step: float,
    solve: Callable[[Any], Any],
    transpose_solve: Callable[[Any], Any],
    transpose_apply: Callable[[Any], Any],
    operator_tangent: Any | None = None,
    operator_derivative_apply: Callable[[Any], Any] | None = None,
    rhs_derivative: Any | None = None,
    observable_vector_derivative: Any | None = None,
    observable_offset_derivative: float | None = None,
    radial_coordinate: str = "psiHat",
    psi_a_hat: float | None = None,
    a_hat: float | None = None,
    r_n: float | None = None,
    observable_chunk_size: int = 128,
    include_jacobian_terms: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> Any:
    """Build a matrix-free RHSMode-1 radial-current derivative system.

    The dense validation builder above assembles full matrices and is therefore
    intentionally capped. This builder keeps the same physical contract but
    exposes only operator actions, a caller-provided transpose action, optional
    analytic/JVP derivative actions, and caller-provided solve closures. If no
    derivative action is supplied, it falls back to centered finite differences.
    It is the production-facing seam for ``Er`` derivatives and RHSMode=4/5
    adjoint diagnostics.
    """

    h = float(derivative_step)
    if h <= 0.0:
        raise ValueError("derivative_step must be positive.")
    total_size = int(getattr(op, "total_size"))
    if total_size <= 0:
        raise ValueError("op.total_size must be positive.")
    for name, candidate in (("op_plus", op_plus), ("op_minus", op_minus)):
        if int(getattr(candidate, "total_size")) != total_size:
            raise ValueError(f"{name}.total_size must match op.total_size.")

    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    from ..sensitivity import MatrixFreeLinearObservableSystem  # noqa: PLC0415
    from sfincs_jax.operators.profile_response.system import apply_v3_full_system_operator_cached, rhs_v3_full_system_jit  # noqa: PLC0415
    from .transport_matrix.diagnostics import radial_current_vm_observable_vector  # noqa: PLC0415

    def apply_operator(operator: Any, state: Any) -> Any:
        vector = jnp.asarray(state, dtype=jnp.float64)
        if int(vector.shape[0]) != total_size:
            raise ValueError(f"operator input length must match op.total_size={total_size}.")
        return apply_v3_full_system_operator_cached(
            operator,
            vector,
            include_jacobian_terms=bool(include_jacobian_terms),
        )

    def apply(state: Any) -> Any:
        return apply_operator(op, state)

    def derivative_apply(state: Any) -> Any:
        vector = jnp.asarray(state, dtype=jnp.float64)
        if operator_derivative_apply is not None:
            return jnp.asarray(operator_derivative_apply(vector), dtype=jnp.float64)
        if operator_tangent is not None:
            _, tangent_action = jax.jvp(
                lambda operator: apply_operator(operator, vector),
                (op,),
                (operator_tangent,),
            )
            return jnp.asarray(tangent_action, dtype=jnp.float64)
        return (apply_operator(op_plus, vector) - apply_operator(op_minus, vector)) / (2.0 * h)

    def apply_transpose(state: Any) -> Any:
        vector = jnp.asarray(state, dtype=jnp.float64)
        if int(vector.shape[0]) != total_size:
            raise ValueError(f"transpose input length must match op.total_size={total_size}.")
        return jnp.asarray(transpose_apply(vector), dtype=jnp.float64)

    rhs = rhs_v3_full_system_jit(op)
    rhs_plus = rhs_v3_full_system_jit(op_plus)
    rhs_minus = rhs_v3_full_system_jit(op_minus)
    if rhs_derivative is not None:
        rhs_derivative_value = jnp.asarray(rhs_derivative, dtype=jnp.float64)
        rhs_derivative_kind = "caller_supplied"
    elif operator_tangent is not None:
        _, rhs_derivative_value = jax.jvp(
            rhs_v3_full_system_jit,
            (op,),
            (operator_tangent,),
        )
        rhs_derivative_kind = "jax_jvp_operator_tangent"
    else:
        rhs_derivative_value = (rhs_plus - rhs_minus) / (2.0 * h)
        rhs_derivative_kind = "centered_finite_difference"
    observable_vector, observable_offset = radial_current_vm_observable_vector(
        op,
        radial_coordinate=radial_coordinate,
        psi_a_hat=psi_a_hat,
        a_hat=a_hat,
        r_n=r_n,
        chunk_size=int(observable_chunk_size),
    )
    observable_vector_plus, observable_offset_plus = radial_current_vm_observable_vector(
        op_plus,
        radial_coordinate=radial_coordinate,
        psi_a_hat=psi_a_hat,
        a_hat=a_hat,
        r_n=r_n,
        chunk_size=int(observable_chunk_size),
    )
    observable_vector_minus, observable_offset_minus = radial_current_vm_observable_vector(
        op_minus,
        radial_coordinate=radial_coordinate,
        psi_a_hat=psi_a_hat,
        a_hat=a_hat,
        r_n=r_n,
        chunk_size=int(observable_chunk_size),
    )
    if observable_vector_derivative is None:
        observable_vector_derivative_value = (observable_vector_plus - observable_vector_minus) / (2.0 * h)
        observable_derivative_kind = "centered_finite_difference_vector"
    else:
        observable_vector_derivative_value = jnp.asarray(observable_vector_derivative, dtype=jnp.float64)
        observable_derivative_kind = "caller_supplied"
    if observable_offset_derivative is None:
        observable_offset_derivative_value = (float(observable_offset_plus) - float(observable_offset_minus)) / (2.0 * h)
    else:
        observable_offset_derivative_value = float(observable_offset_derivative)
    if operator_derivative_apply is not None:
        operator_derivative_kind = "caller_supplied"
    elif operator_tangent is not None:
        operator_derivative_kind = "jax_jvp_operator_tangent"
    else:
        operator_derivative_kind = "centered_finite_difference"
    merged_metadata = {
        "builder": "matrix_free_rhs1_vm_radial_current",
        "total_size": total_size,
        "radial_coordinate": str(radial_coordinate),
        "derivative_step": h,
        "operator_action": "v3_full_system_matrix_free",
        "transpose_action": "caller_supplied",
        "operator_derivative_action": operator_derivative_kind,
        "rhs_derivative": rhs_derivative_kind,
        "observable_derivative": observable_derivative_kind,
        "dense_matrix_assembled": False,
        **dict(metadata or {}),
    }
    return MatrixFreeLinearObservableSystem(
        parameter=float(parameter),
        size=total_size,
        rhs=rhs,
        rhs_derivative=rhs_derivative_value,
        apply=apply,
        transpose_apply=apply_transpose,
        derivative_apply=derivative_apply,
        solve=solve,
        transpose_solve=transpose_solve,
        observable_vector=observable_vector,
        observable_vector_derivative=observable_vector_derivative_value,
        observable_offset=float(observable_offset),
        observable_offset_derivative=observable_offset_derivative_value,
        metadata=merged_metadata,
    )


def _radial_current_derivative_from_linear_certificate(
    *,
    er: float,
    result: Any,
) -> RadialCurrentDerivativeResult:
    """Adapt a scalar linear-observable certificate to the ambipolar contract."""

    return RadialCurrentDerivativeResult(
        er=float(er),
        derivative=float(result.derivative),
        step=0.0 if result.finite_difference_step is None else float(result.finite_difference_step),
        scheme="implicit_linear_adjoint",
        evaluations=(),
        metadata={
            **dict(result.metadata),
            "observable": result.observable,
            "tangent_derivative": result.tangent_derivative,
            "adjoint_derivative": result.adjoint_derivative,
            "tangent_adjoint_abs_error": result.tangent_adjoint_abs_error,
            "primal_residual_norm": result.primal_residual_norm,
            "tangent_residual_norm": result.tangent_residual_norm,
            "adjoint_residual_norm": result.adjoint_residual_norm,
            "finite_difference_derivative": result.finite_difference_derivative,
            "finite_difference_abs_error": result.finite_difference_abs_error,
        },
    )


def solve_sfincs_jax_ambipolar_brent(
    *,
    input_namelist: str | Path,
    work_dir: str | Path,
    er_min: float,
    er_max: float,
    er_initial: float = 0.0,
    max_evaluations: int = 20,
    current_tolerance: float = 1.0e-10,
    step_tolerance: float = 1.0e-8,
    solve_method: str = "auto",
    differentiable: bool = False,
    reuse_output_geometry_cache: bool = True,
    reuse_solver_state: bool = True,
    emit: Callable[[int, str], None] | None = None,
) -> tuple[AmbipolarResult, SfincsJaxRadialCurrentEvaluator]:
    """Run a real sfincs_jax-backed Brent ambipolar solve in-process.

    The returned evaluator exposes the concrete per-evaluation input/output
    artifacts.  Keeping it separate from ``AmbipolarResult`` preserves a small
    mathematical result contract while still giving CLI workflows provenance.
    """

    evaluator = SfincsJaxRadialCurrentEvaluator(
        input_namelist=input_namelist,
        work_dir=work_dir,
        solve_method=solve_method,
        differentiable=differentiable,
        reuse_output_geometry_cache=reuse_output_geometry_cache,
        reuse_solver_state=reuse_solver_state,
        emit=emit,
    )
    result = brent_ambipolar_root(
        evaluator,
        er_min=er_min,
        er_max=er_max,
        er_initial=er_initial,
        max_evaluations=max_evaluations,
        current_tolerance=current_tolerance,
        step_tolerance=step_tolerance,
        metadata={"input_namelist": str(Path(input_namelist)), "work_dir": str(Path(work_dir))},
    )
    return result, evaluator


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
    "RadialCurrentDerivativeEvaluator",
    "RadialCurrentEvaluator",
    "RadialCurrentDerivativeResult",
    "RHSMode1RadialCurrentResponse",
    "SfincsJaxEvaluationRecord",
    "SfincsJaxRadialCurrentEvaluator",
    "brent_ambipolar_root",
    "dense_rhs1_vm_radial_current_linear_observable_system",
    "dphi_hat_dpsi_hat_er_derivative_from_namelist",
    "er_operator_tangent_from_dphi_hat_dpsi_hat_derivative",
    "finite_difference_radial_current_derivative",
    "implicit_linear_radial_current_derivative",
    "implicit_linear_radial_current_derivative_from_builder",
    "implicit_matrix_free_radial_current_derivative",
    "implicit_matrix_free_radial_current_derivative_from_builder",
    "matrix_free_radial_current_derivative_provider",
    "matrix_free_rhs1_vm_radial_current_linear_observable_system",
    "newton_ambipolar_root",
    "operator_tangent_from_centered_difference",
    "rhsmode1_radial_current_response_from_namelist",
    "safeguarded_newton_ambipolar_root",
    "solve_ambipolar_brent",
    "solve_ambipolar_newton",
    "solve_ambipolar_safeguarded_newton",
    "solve_rhsmode1_ambipolar_from_namelist",
    "solve_sfincs_jax_ambipolar_brent",
    "validate_fortran_v3_ambipolar_constraints",
]
