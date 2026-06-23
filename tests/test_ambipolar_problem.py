from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from sfincs_jax.ambipolar import radial_current_from_output
from sfincs_jax.io import read_sfincs_h5
from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.problems.ambipolar import (
    AmbipolarProblem,
    SfincsJaxRadialCurrentEvaluator,
    brent_ambipolar_root,
    finite_difference_radial_current_derivative,
    newton_ambipolar_root,
    safeguarded_newton_ambipolar_root,
    solve_ambipolar_brent,
    solve_ambipolar_newton,
    solve_ambipolar_safeguarded_newton,
    validate_fortran_v3_ambipolar_constraints,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = REPO_ROOT / "benchmarks" / "fortran_v3_ambipolar_reference"


def _summary_case(summary_name: str, case_name: str) -> dict:
    data = json.loads((REFERENCE_ROOT / summary_name).read_text())
    for case in data["cases"]:
        if case["case"] == case_name:
            return case
    raise AssertionError(f"Missing reference case {case_name}")


def _table_evaluator(case: dict, *, atol: float = 1.0e-9):
    er_values = [float(v) for v in case["er_values"]]
    currents = [float(v) for v in case["radial_currents"]]
    calls: list[float] = []

    def evaluate(er: float) -> float:
        calls.append(float(er))
        for er_known, current in zip(er_values, currents, strict=True):
            if math.isclose(float(er), er_known, rel_tol=0.0, abs_tol=atol):
                return current
        raise AssertionError(f"Unexpected Er evaluation {er}; known values are {er_values}")

    return evaluate, calls


def test_brent_matches_fortran_v3_small_w7x_early_stop() -> None:
    """Fortran v3 option-2 Brent stops at Er=0 when current is already below tolerance."""

    case = _summary_case("small_probe_summary_2026-06-22.json", "geometry4_w7x_like_small_option2")
    evaluate, calls = _table_evaluator(case)

    result = brent_ambipolar_root(
        evaluate,
        er_min=-20.0,
        er_max=20.0,
        er_initial=0.0,
        max_evaluations=8,
        current_tolerance=1.0e-7,
        step_tolerance=1.0e-6,
        metadata={"reference": case["case"]},
    )

    assert result.converged
    assert result.status == "converged"
    assert result.metadata["convergence"] == "radial_current"
    np.testing.assert_allclose(result.root_er, 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(result.root_radial_current, case["radial_currents"][2], rtol=0.0, atol=1.0e-18)
    np.testing.assert_allclose(result.er_values, case["er_values"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(result.radial_currents, case["radial_currents"], rtol=0.0, atol=1.0e-18)
    np.testing.assert_allclose(calls, case["er_values"], rtol=0.0, atol=0.0)


def test_brent_replays_fortran_v3_geometry1_helical_sequence() -> None:
    """The Brent update sequence matches the distinct geometry-1 helical reference."""

    case = _summary_case("small_probe_summary_2026-06-22.json", "geometry1_helical_small_option2")
    evaluate, calls = _table_evaluator(case)

    problem = AmbipolarProblem(
        evaluate_radial_current=evaluate,
        er_min=-20.0,
        er_max=20.0,
        er_initial=0.0,
        max_evaluations=8,
        current_tolerance=1.0e-7,
        step_tolerance=1.0e-6,
        metadata={"reference": case["case"]},
    )
    result = solve_ambipolar_brent(problem)

    assert result.converged
    assert result.root_type == "ion"
    np.testing.assert_allclose(result.er_values, case["er_values"], rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(result.radial_currents, case["radial_currents"], rtol=0.0, atol=1.0e-18)
    np.testing.assert_allclose(result.root_er, case["er_values"][-1], rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(result.root_radial_current, case["radial_currents"][-1], rtol=0.0, atol=1.0e-18)
    np.testing.assert_allclose(calls, case["er_values"], rtol=0.0, atol=1.0e-12)


def test_brent_replays_fortran_v3_production_w7x_sequence() -> None:
    """Production Brent reference keeps iterating until the stricter current gate passes."""

    case = _summary_case("production_probe_summary_2026-06-22.json", "geometry4_w7x_like_production_option2")
    evaluate, calls = _table_evaluator(case, atol=2.0e-10)

    result = brent_ambipolar_root(
        evaluate,
        er_min=-20.0,
        er_max=20.0,
        er_initial=0.0,
        max_evaluations=12,
        current_tolerance=1.0e-10,
        step_tolerance=1.0e-8,
        metadata={"reference": case["case"]},
    )

    assert result.converged
    assert result.root_type == "ion"
    assert len(result.iterations) == 6
    np.testing.assert_allclose(result.er_values, case["er_values"], rtol=0.0, atol=2.0e-10)
    np.testing.assert_allclose(result.radial_currents, case["radial_currents"], rtol=0.0, atol=1.0e-18)
    np.testing.assert_allclose(result.root_er, -3.5773320425472463, rtol=0.0, atol=2.0e-10)
    np.testing.assert_allclose(result.root_radial_current, 2.206662531726209e-12, rtol=0.0, atol=1.0e-18)
    np.testing.assert_allclose(calls, case["er_values"], rtol=0.0, atol=2.0e-10)


def test_brent_failure_returns_nonconverged_unbracketed_certificate() -> None:
    result = brent_ambipolar_root(
        lambda er: 1.0 + 0.01 * er,
        er_min=-1.0,
        er_max=1.0,
        er_initial=0.0,
        max_evaluations=6,
        current_tolerance=1.0e-10,
    )

    assert not result.converged
    assert result.status == "unbracketed"
    assert result.root_er is None
    assert result.root_radial_current is None
    assert result.er_values == (-1.0, 1.0, 0.0)


def test_fortran_v3_ambipolar_validator_accepts_reference_decks_and_rejects_adjoint_incompatibilities() -> None:
    namelist_dir = REFERENCE_ROOT / "namelists"
    for path in sorted(namelist_dir.glob("*.namelist")):
        nml = read_sfincs_input(path)
        option = int(nml.group("general")["AMBIPOLARSOLVEOPTION"])
        assert validate_fortran_v3_ambipolar_constraints(nml, option=option) == (), path.name

    errors = validate_fortran_v3_ambipolar_constraints(
        {
            "general": {"RHSMode": 1, "ambipolarSolveOption": 1},
            "physicsParameters": {
                "includePhi1": True,
                "EParallelHat": 0.1,
                "magneticDriftScheme": 1,
                "collisionOperator": 1,
            },
            "resolutionParameters": {"constraintScheme": 2},
        },
        option=1,
    )

    assert len(errors) == 5
    assert any("includePhi1" in item for item in errors)
    assert any("EParallelHat" in item for item in errors)
    assert any("tangential magnetic drifts" in item for item in errors)
    assert any("constraintScheme" in item for item in errors)
    assert any("collisionOperator" in item for item in errors)


def test_sfincs_jax_radial_current_evaluator_runs_real_tiny_rhs1_output(tmp_path: Path) -> None:
    """The canonical ambipolar owner can evaluate radial current through sfincs_jax itself."""

    input_path = REPO_ROOT / "tests" / "ref" / "pas_1species_PAS_noEr_tiny_scheme11.input.namelist"
    evaluator = SfincsJaxRadialCurrentEvaluator(
        input_namelist=input_path,
        work_dir=tmp_path / "ambipolar_eval",
        solve_method="auto",
        compute_solution=True,
    )

    radial_current = evaluator(0.0)

    assert len(evaluator.records) == 1
    record = evaluator.records[0]
    assert record.input_path.exists()
    assert record.output_path.exists()
    assert record.solver_trace_path is not None
    assert record.solver_trace_path.exists()
    assert record.selected_path == "rhsmode1_solution"
    assert record.solve_method is not None
    assert record.metadata["requested_solve_method"] == "auto"
    assert record.residual_norm is not None
    assert record.residual_target is not None
    assert record.elapsed_s is not None
    assert record.total_size is not None
    assert record.active_size is not None
    assert record.cache_enabled is True
    assert record.cache_dir is not None
    assert record.cache_dir.exists()
    assert record.solver_state_reuse_enabled is True
    assert record.solver_state_path is not None
    assert record.solver_state_input_exists is False
    assert record.solver_state_output_exists is True
    assert record.solver_state_path.exists()
    assert record.fixed_shape_signature is not None
    data = read_sfincs_h5(record.output_path)
    np.testing.assert_allclose(float(np.asarray(data["Er"]).reshape(())), 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(radial_current, radial_current_from_output(data), rtol=0.0, atol=5.0e-12)

    repeated_radial_current = evaluator(0.0)
    assert len(evaluator.records) == 2
    repeated = evaluator.records[1]
    assert repeated.solver_state_reuse_enabled is True
    assert repeated.solver_state_input_exists is True
    assert repeated.solver_state_output_exists is True
    assert repeated.solver_state_path == record.solver_state_path
    assert repeated.fixed_shape_signature == record.fixed_shape_signature
    np.testing.assert_allclose(repeated_radial_current, radial_current, rtol=0.0, atol=5.0e-12)


def test_finite_difference_radial_current_derivative_matches_smooth_reference() -> None:
    """Finite differences provide the baseline gate for the future implicit dJr/dEr path."""

    def current(er: float) -> float:
        return 2.5 * er**2 - 3.0 * er + 1.25

    result = finite_difference_radial_current_derivative(
        current,
        er=0.75,
        step=1.0e-5,
        scheme="centered",
    )

    expected = 5.0 * 0.75 - 3.0
    np.testing.assert_allclose(result.derivative, expected, rtol=1.0e-10, atol=1.0e-10)
    assert result.scheme == "centered"
    assert len(result.evaluations) == 2
    assert result.evaluations[0].stage == "finite_difference_plus"
    assert result.evaluations[1].stage == "finite_difference_minus"


def test_safeguarded_newton_matches_brent_on_smooth_bracketed_root() -> None:
    def current(er: float) -> float:
        return (er - 0.25) * (1.0 + 0.1 * er * er)

    def derivative(er: float) -> float:
        return (1.0 + 0.1 * er * er) + (er - 0.25) * 0.2 * er

    problem = AmbipolarProblem(
        evaluate_radial_current=current,
        er_min=-1.0,
        er_max=1.0,
        er_initial=0.0,
        max_evaluations=12,
        current_tolerance=1.0e-12,
        step_tolerance=1.0e-12,
    )

    newton = solve_ambipolar_safeguarded_newton(problem, derivative, derivative_source="analytic")
    brent = solve_ambipolar_brent(problem)

    assert newton.converged
    assert newton.method == "safeguarded_newton"
    np.testing.assert_allclose(newton.root_er, 0.25, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(newton.root_er, brent.root_er, rtol=0.0, atol=1.0e-10)
    assert newton.metadata["derivative_count"] >= 1


def test_safeguarded_newton_accepts_finite_difference_derivative_provider() -> None:
    def current(er: float) -> float:
        return er - 0.125

    result = safeguarded_newton_ambipolar_root(
        current,
        lambda er: finite_difference_radial_current_derivative(current, er=er, step=1.0e-5),
        er_min=-1.0,
        er_max=1.0,
        er_initial=0.0,
        max_evaluations=8,
        current_tolerance=1.0e-12,
        step_tolerance=1.0e-12,
    )

    assert result.converged
    np.testing.assert_allclose(result.root_er, 0.125, rtol=0.0, atol=1.0e-12)
    assert result.metadata["derivative_count"] == 1


def test_safeguarded_newton_falls_back_to_bisection_for_unsafe_derivative() -> None:
    result = safeguarded_newton_ambipolar_root(
        lambda er: er,
        lambda er: 0.0,
        er_min=-1.0,
        er_max=1.0,
        er_initial=0.25,
        max_evaluations=60,
        current_tolerance=1.0e-12,
        step_tolerance=1.0e-12,
    )

    assert result.converged
    assert result.metadata["fallback_count"] >= 1
    np.testing.assert_allclose(result.root_er, 0.0, rtol=0.0, atol=1.0e-12)


def test_pure_newton_converges_and_fails_closed_for_zero_derivative() -> None:
    result = newton_ambipolar_root(
        lambda er: er - 0.4,
        lambda er: 1.0,
        er_min=-2.0,
        er_max=2.0,
        er_initial=0.0,
        max_evaluations=5,
        current_tolerance=1.0e-12,
        step_tolerance=1.0e-12,
    )

    assert result.converged
    np.testing.assert_allclose(result.root_er, 0.4, rtol=0.0, atol=1.0e-12)
    assert result.metadata["derivative_count"] == 1

    failed = solve_ambipolar_newton(
        AmbipolarProblem(
            evaluate_radial_current=lambda er: er - 0.4,
            er_min=-2.0,
            er_max=2.0,
            er_initial=0.0,
            max_evaluations=5,
        ),
        lambda er: 0.0,
    )

    assert not failed.converged
    assert failed.status == "zero_derivative"
