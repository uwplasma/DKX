"""Ambipolar electric-field root solves.

This module owns the in-process ambipolar problem contract.  It is deliberately
small and independent of the full v3 driver so the root-finding policy can be
tested against SFINCS Fortran v3 before it is wired to expensive transport
solves.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
import math
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any


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
        from ..solver_trace import read_solver_trace_json  # noqa: PLC0415

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
    from ..v3_system import apply_v3_full_system_operator_cached, rhs_v3_full_system_jit  # noqa: PLC0415
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
    "RadialCurrentDerivativeEvaluator",
    "RadialCurrentEvaluator",
    "RadialCurrentDerivativeResult",
    "SfincsJaxEvaluationRecord",
    "SfincsJaxRadialCurrentEvaluator",
    "brent_ambipolar_root",
    "dense_rhs1_vm_radial_current_linear_observable_system",
    "finite_difference_radial_current_derivative",
    "implicit_linear_radial_current_derivative",
    "implicit_linear_radial_current_derivative_from_builder",
    "newton_ambipolar_root",
    "safeguarded_newton_ambipolar_root",
    "solve_ambipolar_brent",
    "solve_ambipolar_newton",
    "solve_ambipolar_safeguarded_newton",
    "solve_sfincs_jax_ambipolar_brent",
    "validate_fortran_v3_ambipolar_constraints",
]
