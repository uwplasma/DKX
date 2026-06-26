from __future__ import annotations

import json
import math
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from sfincs_jax.ambipolar import radial_current_from_output
from sfincs_jax.io import read_sfincs_h5
from sfincs_jax.namelist import read_sfincs_input
import sfincs_jax.problems.ambipolar as ambipolar_problem
from sfincs_jax.problems.ambipolar import (
    AmbipolarProblem,
    RadialCurrentDerivativeResult,
    RHSMode1RadialCurrentResponse,
    SfincsJaxRadialCurrentEvaluator,
    brent_ambipolar_root,
    finite_difference_radial_current_derivative,
    matrix_free_radial_current_derivative_provider,
    newton_ambipolar_root,
    rhsmode1_radial_current_response_from_namelist,
    safeguarded_newton_ambipolar_root,
    solve_ambipolar_brent,
    solve_ambipolar_newton,
    solve_ambipolar_safeguarded_newton,
    solve_rhsmode1_ambipolar_from_namelist,
    validate_fortran_v3_ambipolar_constraints,
)
from sfincs_jax.sensitivity import MatrixFreeLinearObservableSystem, evaluate_matrix_free_linear_observable


REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = REPO_ROOT / "benchmarks" / "fortran_v3_ambipolar_reference"


def _summary_case(summary_name: str, case_name: str) -> dict:
    data = json.loads((REFERENCE_ROOT / summary_name).read_text())
    for case in data["cases"]:
        if case["case"] == case_name:
            return case
    raise AssertionError(f"Missing reference case {case_name}")


def _profile_case(summary_name: str, case_name: str) -> dict:
    case = _summary_case(summary_name, case_name)
    return case["parsed"]


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


def _nonzero_reference_pairs(case: dict) -> tuple[list[float], list[float]]:
    pairs = [
        (float(er), float(current))
        for er, current in zip(case["er_values"], case["radial_currents"], strict=True)
        if float(er) != 0.0 or float(current) != 0.0
    ]
    return [er for er, _ in pairs], [current for _, current in pairs]


def _newton_derivative_replay_provider(er_values: list[float], currents: list[float]):
    derivatives = {
        er: current / (er - er_next)
        for er, er_next, current in zip(er_values[:-1], er_values[1:], currents[:-1], strict=True)
    }

    def derivative(er: float) -> RadialCurrentDerivativeResult:
        for er_known, value in derivatives.items():
            if math.isclose(float(er), er_known, rel_tol=0.0, abs_tol=1.0e-10):
                return RadialCurrentDerivativeResult(
                    er=float(er),
                    derivative=float(value),
                    step=0.0,
                    scheme="fortran_sequence_replay",
                    evaluations=(),
                    metadata={"reference_er": float(er_known)},
                )
        raise AssertionError(f"Unexpected derivative evaluation {er}; known values are {sorted(derivatives)}")

    return derivative


def test_ambipolar_problem_certificates_validate_and_freeze_metadata() -> None:
    problem = AmbipolarProblem(
        evaluate_radial_current=lambda er: er,
        er_min=-1,
        er_max=1,
        metadata={"case": "metadata_freeze"},
    )

    assert problem.er_min == -1.0
    assert problem.max_evaluations == 20
    with pytest.raises(TypeError):
        problem.metadata["case"] = "mutated"

    with pytest.raises(ValueError, match="max_evaluations"):
        AmbipolarProblem(lambda er: er, er_min=-1.0, er_max=1.0, max_evaluations=2)
    with pytest.raises(ValueError, match="er_min"):
        AmbipolarProblem(lambda er: er, er_min=1.0, er_max=1.0)
    with pytest.raises(ValueError, match="current_tolerance"):
        AmbipolarProblem(lambda er: er, er_min=-1.0, er_max=1.0, current_tolerance=0.0)
    with pytest.raises(ValueError, match="step_tolerance"):
        AmbipolarProblem(lambda er: er, er_min=-1.0, er_max=1.0, step_tolerance=0.0)

    derivative = RadialCurrentDerivativeResult(
        er=1,
        derivative=2,
        step=3,
        scheme="unit",
        evaluations=(),
        metadata={"source": "test"},
    )
    assert derivative.er == 1.0
    assert derivative.derivative == 2.0
    with pytest.raises(TypeError):
        derivative.metadata["source"] = "mutated"

    with pytest.raises(ValueError, match="finite_difference_step"):
        RHSMode1RadialCurrentResponse(lambda er: None, finite_difference_step=0.0)


def test_ambipolar_fortran_sign_root_and_bracket_helpers_cover_endpoint_logic() -> None:
    assert ambipolar_problem._same_fortran_sign(1.0, 2.0)
    assert ambipolar_problem._same_fortran_sign(-1.0, -2.0)
    assert not ambipolar_problem._same_fortran_sign(0.0, 2.0)
    assert not ambipolar_problem._same_fortran_sign(-1.0, 2.0)

    assert ambipolar_problem._root_type(None) == "unknown"
    assert ambipolar_problem._root_type(0.0) == "ion"
    assert ambipolar_problem._root_type(1.0) == "electron"
    assert ambipolar_problem._root_type(-1.0) == "ion"

    assert ambipolar_problem._bracket_from_values(a=-2.0, fa=1.0, c=2.0, fc=3.0, b=0.0, fb=-1.0) is None
    assert ambipolar_problem._bracket_from_values(a=-2.0, fa=-1.0, c=2.0, fc=1.0, b=0.5, fb=0.25) == (
        -2.0,
        -1.0,
        0.5,
        0.25,
    )
    assert ambipolar_problem._bracket_from_values(a=-2.0, fa=-1.0, c=2.0, fc=1.0, b=-0.5, fb=-0.25) == (
        -0.5,
        -0.25,
        2.0,
        1.0,
    )


def test_ambipolar_temporary_env_restores_after_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_AMBIPOLAR_EXISTING", "outer")
    monkeypatch.delenv("SFINCS_JAX_AMBIPOLAR_NEW", raising=False)

    with pytest.raises(RuntimeError, match="trigger cleanup"):
        with ambipolar_problem._temporary_env(
            {
                "SFINCS_JAX_AMBIPOLAR_EXISTING": "inner",
                "SFINCS_JAX_AMBIPOLAR_NEW": "created",
            }
        ):
            assert ambipolar_problem.os.environ["SFINCS_JAX_AMBIPOLAR_EXISTING"] == "inner"
            assert ambipolar_problem.os.environ["SFINCS_JAX_AMBIPOLAR_NEW"] == "created"
            raise RuntimeError("trigger cleanup")

    assert ambipolar_problem.os.environ["SFINCS_JAX_AMBIPOLAR_EXISTING"] == "outer"
    assert "SFINCS_JAX_AMBIPOLAR_NEW" not in ambipolar_problem.os.environ


def test_finite_difference_derivative_schemes_and_validation_cover_one_sided_gates() -> None:
    def current(er: float) -> float:
        return 3.0 * er - 0.75

    for scheme in ("centered", "central", "forward", "backward"):
        result = finite_difference_radial_current_derivative(current, er=0.4, step=1.0e-4, scheme=scheme)
        np.testing.assert_allclose(result.derivative, 3.0, rtol=0.0, atol=1.0e-10)
        assert result.scheme == ("centered" if scheme == "central" else scheme)
        assert len(result.evaluations) == 2

    with pytest.raises(ValueError, match="step must be positive"):
        finite_difference_radial_current_derivative(current, er=0.0, step=0.0)
    with pytest.raises(ValueError, match="scheme must be"):
        finite_difference_radial_current_derivative(current, er=0.0, step=1.0e-3, scheme="complex")


def test_ambipolar_derivative_root_solvers_fail_closed_for_invalid_sensitivity_paths() -> None:
    problem = AmbipolarProblem(
        evaluate_radial_current=lambda er: er - 0.5,
        er_min=-1.0,
        er_max=1.0,
        er_initial=0.0,
        max_evaluations=8,
        current_tolerance=1.0e-12,
        step_tolerance=1.0e-12,
    )

    invalid_derivative = solve_ambipolar_safeguarded_newton(problem, lambda er: math.nan)
    assert not invalid_derivative.converged
    assert invalid_derivative.status == "invalid_derivative"
    assert invalid_derivative.metadata["derivative_count"] == 0

    out_of_bounds = solve_ambipolar_newton(problem, lambda er: 0.01)
    assert not out_of_bounds.converged
    assert out_of_bounds.status == "out_of_bounds"
    assert out_of_bounds.root_radial_current is None
    assert out_of_bounds.metadata["last_step"] > 1.0


def test_rhsmode1_ambipolar_response_rejects_invalid_derivative_steps_before_setup() -> None:
    with pytest.raises(ValueError, match="derivative_step"):
        rhsmode1_radial_current_response_from_namelist(nml={}, derivative_step=0.0)
    with pytest.raises(ValueError, match="finite_difference_step"):
        rhsmode1_radial_current_response_from_namelist(
            nml={},
            derivative_step=1.0e-5,
            finite_difference_step=-1.0,
        )


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


def test_new_profile_summary_replays_geometry1_pure_newton_sequence() -> None:
    """The new helical option-3 profile pins the derivative-assisted Newton owner."""

    case = _profile_case("small_profile_summary_2026-06-23.json", "geometry1_helical_small_option3")
    er_values, currents = _nonzero_reference_pairs(case)
    evaluate, calls = _table_evaluator({"er_values": er_values, "radial_currents": currents}, atol=1.0e-10)

    result = newton_ambipolar_root(
        evaluate,
        _newton_derivative_replay_provider(er_values, currents),
        er_min=-20.0,
        er_max=20.0,
        er_initial=0.0,
        max_evaluations=8,
        current_tolerance=1.0e-7,
        step_tolerance=1.0e-6,
    )

    assert result.converged
    assert result.method == "newton"
    np.testing.assert_allclose(result.er_values, er_values, rtol=0.0, atol=1.0e-10)
    np.testing.assert_allclose(result.radial_currents, currents, rtol=0.0, atol=1.0e-18)
    np.testing.assert_allclose(calls, er_values, rtol=0.0, atol=1.0e-10)


def test_new_profile_summary_replays_geometry1_safeguarded_newton_sequence() -> None:
    """The helical option-1 profile pins the safeguarded Newton/bisection owner."""

    case = _profile_case("small_profile_summary_2026-06-23.json", "geometry1_helical_small_option1")
    er_values, currents = _nonzero_reference_pairs(case)
    evaluate, calls = _table_evaluator({"er_values": er_values, "radial_currents": currents}, atol=1.0e-10)

    result = safeguarded_newton_ambipolar_root(
        evaluate,
        _newton_derivative_replay_provider(er_values, currents),
        er_min=-20.0,
        er_max=20.0,
        er_initial=0.0,
        max_evaluations=8,
        current_tolerance=1.0e-7,
        step_tolerance=1.0e-6,
    )

    assert result.converged
    assert result.method == "safeguarded_newton"
    assert result.metadata["derivative_count"] == 1
    assert result.metadata["fallback_count"] == 0
    np.testing.assert_allclose(result.er_values, er_values, rtol=0.0, atol=1.0e-10)
    np.testing.assert_allclose(result.radial_currents, currents, rtol=0.0, atol=1.0e-18)
    np.testing.assert_allclose(calls, er_values, rtol=0.0, atol=1.0e-10)


def test_new_profile_summaries_preserve_solver_counts_rss_bounds_and_marker_residual_split() -> None:
    """Reference summaries distinguish physical residuals from Fortran success markers."""

    small = json.loads((REFERENCE_ROOT / "small_profile_summary_2026-06-23.json").read_text())
    production = json.loads((REFERENCE_ROOT / "production_profile_summary_2026-06-23.json").read_text())

    assert len(small["cases"]) == 6
    assert len(production["cases"]) == 6
    for payload in (small, production):
        for case in payload["cases"]:
            parsed = case["parsed"]
            case_name = str(case["case"])
            expected_solves = len(_nonzero_reference_pairs(parsed)[0])
            assert parsed["solver_packages"] == ["mumps"]
            assert parsed["petsc_profile_markers"]["ksp_view"] is True
            assert parsed["petsc_profile_markers"]["pc_view"] is True
            assert isinstance(parsed["petsc_profile_markers"]["log_view"], bool)
            assert parsed["max_rss_bytes"] is not None
            assert len(parsed["main_solve_times_s"]) == expected_solves
            assert len(parsed["jacobian_nnz"]) == expected_solves
            assert len(parsed["preconditioner_nnz"]) == expected_solves
            assert max(parsed["jacobian_nnz"] or [0]) > 0
            if case_name.endswith("option2"):
                assert parsed["adjoint_solve_times_s"] == []
            else:
                assert len(parsed["adjoint_solve_times_s"]) == expected_solves
            if payload["tier"] == "small":
                assert parsed["max_rss_bytes"] < 250_000_000
            elif "geometry1_helical" in case_name:
                assert parsed["max_rss_bytes"] < 6_500_000_000
            else:
                assert parsed["max_rss_bytes"] < 1_700_000_000

    helical_brent = _profile_case(
        "production_profile_summary_2026-06-23.json",
        "geometry1_helical_production_option2",
    )
    nonzero_currents = [abs(float(v)) for v in helical_brent["radial_currents"] if float(v) != 0.0]
    assert min(nonzero_currents) < 1.0e-10
    assert helical_brent["success_markers"]["brent_successful"] is False
    assert helical_brent["success_markers"]["goodbye"] is False


def test_production_option13_summaries_pin_derivative_solve_metadata() -> None:
    """Production option-1/3 references include one adjoint derivative solve per physical solve."""

    production = json.loads((REFERENCE_ROOT / "production_profile_summary_2026-06-23.json").read_text())
    option13_cases = [case for case in production["cases"] if str(case["case"]).endswith(("option1", "option3"))]
    assert len(option13_cases) == 4

    for case in option13_cases:
        parsed = case["parsed"]
        solve_count = len(parsed["main_solve_times_s"])
        assert parsed["solver_packages"] == ["mumps"]
        assert parsed["success_markers"]["newton_successful"] is True
        assert parsed["internal_ambipolar_time_s"] > 0.0
        assert parsed["wall_time_s"] >= parsed["internal_ambipolar_time_s"]
        assert solve_count >= 4
        assert len(parsed["adjoint_solve_times_s"]) == solve_count
        assert len(parsed["jacobian_nnz"]) == solve_count
        assert len(parsed["preconditioner_nnz"]) == solve_count
        assert min(parsed["jacobian_nnz"]) > 1_000_000
        assert min(parsed["preconditioner_nnz"]) > 1_000_000
        assert parsed["max_ksp_iteration_index"] >= 3
        assert parsed["petsc_profile_markers"] == {
            "ksp_view": True,
            "log_view": True,
            "pc_view": True,
        }
        assert abs(float(parsed["radial_currents"][-1])) <= 7.0e-11


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


def test_rhsmode1_namelist_ambipolar_option1_replays_fortran_active_root() -> None:
    """The real active response drives the safeguarded Newton option-1 root."""

    input_path = REFERENCE_ROOT / "namelists" / "geometry1_helical_small_option1.namelist"

    result = solve_rhsmode1_ambipolar_from_namelist(
        nml=input_path,
        derivative_step=1.0e-5,
        max_dense_size=1000,
        observable_chunk_size=128,
        metadata={"gate": "fortran_option1_active_root"},
    )

    assert result.converged
    assert result.method == "safeguarded_newton"
    assert result.metadata["builder"] == "solve_rhsmode1_ambipolar_from_namelist"
    assert result.metadata["ambipolar_solve_option"] == 1
    assert result.metadata["radial_current_response"]["response_builder"] == (
        "rhsmode1_radial_current_response_from_namelist"
    )
    assert result.metadata["derivative_count"] == 1
    assert result.metadata["fallback_count"] == 0
    np.testing.assert_allclose(result.root_er, -2.01579684708909, rtol=0.0, atol=2.0e-5)
    np.testing.assert_allclose(result.root_radial_current, -1.0650279455435228e-9, rtol=0.0, atol=1.0e-12)


def test_rhsmode1_namelist_ambipolar_option3_replays_fortran_active_root() -> None:
    """The same active response drives the pure Newton option-3 root."""

    input_path = REFERENCE_ROOT / "namelists" / "geometry1_helical_small_option3.namelist"

    result = solve_rhsmode1_ambipolar_from_namelist(
        nml=input_path,
        derivative_step=1.0e-5,
        max_dense_size=1000,
        observable_chunk_size=128,
        metadata={"gate": "fortran_option3_active_root"},
    )

    assert result.converged
    assert result.method == "newton"
    assert result.metadata["ambipolar_solve_option"] == 3
    assert result.metadata["derivative_count"] == 1
    np.testing.assert_allclose(result.er_values, (0.0, -2.01579684708909), rtol=0.0, atol=2.0e-5)
    np.testing.assert_allclose(result.root_radial_current, -1.0650279455435228e-9, rtol=0.0, atol=1.0e-12)


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
    assert record.solver_state_input_used is False
    assert record.solver_state_output_exists is True
    assert record.solver_state_path.exists()
    assert record.fixed_shape_input_signature is None
    assert record.fixed_shape_signature is not None
    assert record.fixed_shape_reuse_enabled is True
    assert record.fixed_shape_reuse_admitted is False
    assert record.fixed_shape_reuse_reason == "no_prior_state"
    assert record.fixed_shape_reuse_count == 0
    data = read_sfincs_h5(record.output_path)
    np.testing.assert_allclose(float(np.asarray(data["Er"]).reshape(())), 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(radial_current, radial_current_from_output(data), rtol=0.0, atol=5.0e-12)

    repeated_radial_current = evaluator(0.0)
    assert len(evaluator.records) == 2
    repeated = evaluator.records[1]
    assert repeated.solver_state_reuse_enabled is True
    assert repeated.solver_state_input_exists is True
    assert repeated.solver_state_input_used is True
    assert repeated.solver_state_output_exists is True
    assert repeated.solver_state_path == record.solver_state_path
    assert repeated.fixed_shape_input_signature == record.fixed_shape_signature
    assert repeated.fixed_shape_signature == record.fixed_shape_signature
    assert repeated.fixed_shape_reuse_enabled is True
    assert repeated.fixed_shape_reuse_admitted is True
    assert repeated.fixed_shape_reuse_reason == "fixed_shape_signature_match"
    assert repeated.fixed_shape_reuse_count == 1
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


def test_newton_options_accept_matrix_free_implicit_derivative_provider() -> None:
    root_er = 0.2

    def raw_current(er: float) -> float:
        matrix = jnp.asarray(
            [[2.0 + 0.1 * er, 0.2], [0.1, 1.5 - 0.05 * er]],
            dtype=jnp.float64,
        )
        rhs = jnp.asarray([er + 0.1, 1.0 - 0.2 * er], dtype=jnp.float64)
        vector = jnp.asarray([1.2, -0.3 + 0.1 * er], dtype=jnp.float64)
        return float(jnp.vdot(vector, jnp.linalg.solve(matrix, rhs)))

    offset = -raw_current(root_er)

    def build_system(er: float) -> MatrixFreeLinearObservableSystem:
        matrix = jnp.asarray(
            [[2.0 + 0.1 * er, 0.2], [0.1, 1.5 - 0.05 * er]],
            dtype=jnp.float64,
        )
        matrix_derivative = jnp.asarray([[0.1, 0.0], [0.0, -0.05]], dtype=jnp.float64)
        rhs_derivative = jnp.asarray([1.0, -0.2], dtype=jnp.float64)
        vector_derivative = jnp.asarray([0.0, 0.1], dtype=jnp.float64)
        return MatrixFreeLinearObservableSystem(
            parameter=float(er),
            size=2,
            rhs=jnp.asarray([er + 0.1, 1.0 - 0.2 * er], dtype=jnp.float64),
            rhs_derivative=rhs_derivative,
            apply=lambda state: matrix @ state,
            transpose_apply=lambda state: matrix.T @ state,
            derivative_apply=lambda state: matrix_derivative @ state,
            solve=lambda rhs: jnp.linalg.solve(matrix, rhs),
            transpose_solve=lambda rhs: jnp.linalg.solve(matrix.T, rhs),
            observable_vector=jnp.asarray([1.2, -0.3 + 0.1 * er], dtype=jnp.float64),
            observable_vector_derivative=vector_derivative,
            observable_offset=offset,
            metadata={"builder": "matrix_free_option13_unit"},
        )

    def current(er: float) -> float:
        return evaluate_matrix_free_linear_observable(build_system(float(er)))

    provider_records: list[RadialCurrentDerivativeResult] = []
    provider = matrix_free_radial_current_derivative_provider(
        build_system,
        finite_difference_step=1.0e-6,
        metadata={"derivative_provider": "matrix_free_implicit"},
    )

    def recorded_provider(er: float) -> RadialCurrentDerivativeResult:
        result = provider(float(er))
        provider_records.append(result)
        return result

    safeguarded = safeguarded_newton_ambipolar_root(
        current,
        recorded_provider,
        er_min=-1.0,
        er_max=1.0,
        er_initial=0.0,
        max_evaluations=8,
        current_tolerance=1.0e-12,
        step_tolerance=1.0e-12,
    )
    assert safeguarded.converged
    assert safeguarded.method == "safeguarded_newton"
    np.testing.assert_allclose(safeguarded.root_er, root_er, rtol=0.0, atol=1.0e-10)

    newton_records: list[RadialCurrentDerivativeResult] = []

    def recorded_newton_provider(er: float) -> RadialCurrentDerivativeResult:
        result = provider(float(er))
        newton_records.append(result)
        return result

    strict_newton = newton_ambipolar_root(
        current,
        recorded_newton_provider,
        er_min=-1.0,
        er_max=1.0,
        er_initial=0.0,
        max_evaluations=8,
        current_tolerance=1.0e-12,
        step_tolerance=1.0e-12,
    )
    assert strict_newton.converged
    assert strict_newton.method == "newton"
    np.testing.assert_allclose(strict_newton.root_er, root_er, rtol=0.0, atol=1.0e-10)
    for result in [*provider_records, *newton_records]:
        assert result.scheme == "implicit_linear_adjoint"
        assert result.metadata["builder"] == "matrix_free_option13_unit"
        assert result.metadata["derivative_provider"] == "matrix_free_implicit"
        assert result.metadata["system_kind"] == "matrix_free_linear_observable"
        assert result.metadata["finite_difference_abs_error"] < 1.0e-8


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
