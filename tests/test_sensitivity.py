from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sfincs_jax.problems.ambipolar import (
    RadialCurrentDerivativeResult,
    dense_rhs1_vm_radial_current_linear_observable_system,
    dphi_hat_dpsi_hat_er_derivative_from_namelist,
    er_operator_tangent_from_dphi_hat_dpsi_hat_derivative,
    implicit_linear_radial_current_derivative,
    implicit_linear_radial_current_derivative_from_builder,
    implicit_matrix_free_radial_current_derivative_from_builder,
    matrix_free_rhs1_vm_radial_current_linear_observable_system,
    operator_tangent_from_centered_difference,
    rhsmode1_radial_current_response_from_namelist,
)
from sfincs_jax.problems.transport_matrix.diagnostics import (
    radial_current_vm_from_state,
    radial_current_vm_observable_vector,
    radial_current_vm_psi_hat_from_state,
    radial_current_vm_psi_hat_observable_vector,
    v3_transport_diagnostics_vm_only,
)
from sfincs_jax.sensitivity import (
    LinearObservableSystem,
    MatrixFreeLinearObservableSystem,
    adjoint_dot_product_check,
    evaluate_matrix_free_linear_observable,
    fortran_v3_adjoint_sensitivity_output_fields,
    implicit_linear_observable_derivative,
    implicit_linear_observable_derivative_from_builder,
    implicit_matrix_free_linear_observable_derivative_from_builder,
    probe_linear_observable_vector,
    validate_fortran_v3_adjoint_sensitivity_constraints,
)
from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.v3_system import apply_v3_full_system_operator_cached, full_system_operator_from_namelist

REPO_ROOT = Path(__file__).resolve().parents[1]


def _linear_system_components():
    a0 = jnp.asarray(
        [
            [4.0, 0.7, -0.1],
            [-0.3, 3.2, 0.4],
            [0.2, -0.5, 2.7],
        ],
        dtype=jnp.float64,
    )
    ap = jnp.asarray(
        [
            [0.5, -0.05, 0.0],
            [0.1, -0.2, 0.03],
            [0.0, 0.08, 0.25],
        ],
        dtype=jnp.float64,
    )
    b0 = jnp.asarray([1.0, -0.25, 0.75], dtype=jnp.float64)
    bp = jnp.asarray([-0.2, 0.4, 0.1], dtype=jnp.float64)
    c0 = jnp.asarray([0.3, -0.7, 1.1], dtype=jnp.float64)
    cp = jnp.asarray([0.05, 0.0, -0.03], dtype=jnp.float64)
    offset0 = 0.125
    offsetp = -0.4
    p0 = 0.35
    return a0, ap, b0, bp, c0, cp, offset0, offsetp, p0


def _adjoint_config(**overrides):
    config = {
        "general": {"RHSMode": 4},
        "geometryParameters": {"geometryScheme": 4},
        "physicsParameters": {
            "includePhi1": False,
            "EParallelHat": 0.0,
            "magneticDriftScheme": 0,
            "collisionOperator": 0,
            "includeXDotTerm": True,
            "useDKESExBDrift": False,
            "includeElectricFieldTermInXiDot": True,
        },
        "resolutionParameters": {"constraintScheme": -1},
        "adjointOptions": {
            "discreteAdjointOption": True,
            "adjointBootstrapOption": False,
            "adjointRadialCurrentOption": False,
            "adjointTotalHeatFluxOption": False,
            "adjointHeatFluxOption": [False, False],
            "adjointParticleFluxOption": [False, False],
            "adjointParallelFlowOption": [False, False],
            "debugAdjoint": False,
        },
    }
    for group, values in overrides.items():
        config[group].update(values)
    return config


def test_fortran_v3_adjoint_sensitivity_constraints_and_output_fields() -> None:
    config = _adjoint_config(
        adjointOptions={
            "adjointBootstrapOption": True,
            "adjointRadialCurrentOption": True,
            "adjointTotalHeatFluxOption": True,
            "adjointHeatFluxOption": [False, True],
            "adjointParticleFluxOption": [True, False],
            "adjointParallelFlowOption": [False, True],
        }
    )

    assert validate_fortran_v3_adjoint_sensitivity_constraints(config) == ()
    assert fortran_v3_adjoint_sensitivity_output_fields(config) == (
        "dHeatFluxdLambda",
        "dParticleFluxdLambda",
        "dParallelFlowdLambda",
        "dTotalHeatFluxdLambda",
        "dRadialCurrentdLambda",
        "dBootstrapdLambda",
    )


def test_fortran_v3_adjoint_sensitivity_constraints_reject_invalid_source_combinations() -> None:
    config = _adjoint_config(
        general={"RHSMode": 5},
        geometryParameters={"geometryScheme": 5},
        physicsParameters={
            "includePhi1": True,
            "EParallelHat": 0.1,
            "magneticDriftScheme": 1,
            "collisionOperator": 1,
            "includeXDotTerm": False,
            "useDKESExBDrift": False,
            "includeElectricFieldTermInXiDot": True,
        },
        resolutionParameters={"constraintScheme": 2},
        adjointOptions={"adjointRadialCurrentOption": True, "discreteAdjointOption": False},
    )

    errors = validate_fortran_v3_adjoint_sensitivity_constraints(config)

    assert len(errors) == 8
    assert any("adjointRadialCurrentOption" in item for item in errors)
    assert any("Boozer-coordinate" in item for item in errors)
    assert any("includePhi1" in item for item in errors)
    assert any("EParallelHat" in item for item in errors)
    assert any("tangential magnetic drifts" in item for item in errors)
    assert any("constraintScheme" in item for item in errors)
    assert any("collisionOperator" in item for item in errors)
    assert any("discreteAdjointOption=false" in item for item in errors)


def test_fortran_v3_adjoint_output_fields_preserve_parallel_flow_source_gate() -> None:
    """Pin the writeHDF5Output.F90 gate before sfincs_jax adds its own fields."""

    parallel_only = _adjoint_config(
        adjointOptions={
            "adjointParallelFlowOption": [True, False],
            "adjointParticleFluxOption": [False, False],
        }
    )
    debug_rhs5 = _adjoint_config(
        general={"RHSMode": 5},
        adjointOptions={"debugAdjoint": True},
    )

    assert "dParallelFlowdLambda" not in fortran_v3_adjoint_sensitivity_output_fields(parallel_only)
    fields = fortran_v3_adjoint_sensitivity_output_fields(debug_rhs5)
    assert "dParallelFlowdLambda" in fields
    assert "dPhidPsidLambda" in fields
    assert "dPhidPsiPercentError" in fields
    assert "dPhidPsidLambda_finitediff" in fields


def test_fortran_v3_rhs4_reference_summary_pins_radial_current_sensitivity() -> None:
    reference_root = REPO_ROOT / "benchmarks" / "fortran_v3_sensitivity_reference"
    input_path = reference_root / "namelists" / "geometry4_w7x_like_small_rhs4_radial_current.namelist"
    summary = json.loads((reference_root / "small_rhsmode4_summary_2026-06-25.json").read_text())
    case = summary["cases"][0]
    nml = read_sfincs_input(input_path)

    assert validate_fortran_v3_adjoint_sensitivity_constraints(nml) == ()
    assert fortran_v3_adjoint_sensitivity_output_fields(nml) == (
        "dParticleFluxdLambda",
        "dParallelFlowdLambda",
        "dRadialCurrentdLambda",
    )
    assert case["wall_time_s"] < 1.0
    assert case["max_rss_bytes"] < 150_000_000
    assert case["hdf5_fields"]["dParticleFluxdLambda"]["shape"] == [1, 4, 2, 1]
    assert case["hdf5_fields"]["dRadialCurrentdLambda"]["shape"] == [1, 4, 1]

    particle = np.asarray(case["hdf5_fields"]["dParticleFluxdLambda"]["values"], dtype=np.float64)
    radial = np.asarray(case["hdf5_fields"]["dRadialCurrentdLambda"]["values"], dtype=np.float64)
    expected_radial = particle[:, :, 0, :] - particle[:, :, 1, :]
    np.testing.assert_allclose(radial, expected_radial, rtol=0.0, atol=5.0e-18)


def test_implicit_linear_observable_derivative_matches_tangent_adjoint_and_finite_difference() -> None:
    a0, ap, b0, bp, c0, cp, offset0, offsetp, p0 = _linear_system_components()

    def observable(p: float) -> float:
        a = a0 + float(p) * ap
        b = b0 + float(p) * bp
        c = c0 + float(p) * cp
        x = jnp.linalg.solve(a, b)
        return float(jnp.vdot(c, x) + offset0 + float(p) * offsetp)

    result = implicit_linear_observable_derivative(
        matrix=a0 + p0 * ap,
        rhs=b0 + p0 * bp,
        matrix_derivative=ap,
        rhs_derivative=bp,
        observable_vector=c0 + p0 * cp,
        observable_vector_derivative=cp,
        observable_offset=offset0 + p0 * offsetp,
        observable_offset_derivative=offsetp,
        parameter=p0,
        finite_difference_observable=observable,
        finite_difference_step=1.0e-6,
        metadata={"case": "nonsymmetric_dense"},
    )

    assert result.metadata["case"] == "nonsymmetric_dense"
    np.testing.assert_allclose(result.observable, observable(p0), rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(result.tangent_derivative, result.adjoint_derivative, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(result.derivative, result.finite_difference_derivative, rtol=0.0, atol=1.0e-8)
    assert result.primal_residual_norm < 1.0e-12
    assert result.tangent_residual_norm < 1.0e-12
    assert result.adjoint_residual_norm < 1.0e-12
    assert result.tangent_adjoint_abs_error < 1.0e-12
    assert result.finite_difference_abs_error is not None
    assert result.finite_difference_abs_error < 1.0e-8


def test_implicit_linear_radial_current_derivative_adapts_to_ambipolar_contract() -> None:
    a0, ap, b0, bp, c0, cp, offset0, offsetp, p0 = _linear_system_components()

    def radial_current(er: float) -> float:
        a = a0 + float(er) * ap
        b = b0 + float(er) * bp
        c = c0 + float(er) * cp
        x = jnp.linalg.solve(a, b)
        return float(jnp.vdot(c, x) + offset0 + float(er) * offsetp)

    result = implicit_linear_radial_current_derivative(
        er=p0,
        matrix=a0 + p0 * ap,
        rhs=b0 + p0 * bp,
        matrix_derivative=ap,
        rhs_derivative=bp,
        radial_current_vector=c0 + p0 * cp,
        radial_current_vector_derivative=cp,
        radial_current_offset=offset0 + p0 * offsetp,
        radial_current_offset_derivative=offsetp,
        finite_difference_radial_current=radial_current,
        finite_difference_step=1.0e-6,
        metadata={"source": "unit_test"},
    )

    assert isinstance(result, RadialCurrentDerivativeResult)
    assert result.scheme == "implicit_linear_adjoint"
    assert result.step == 1.0e-6
    assert result.metadata["source"] == "unit_test"
    assert result.metadata["finite_difference_abs_error"] < 1.0e-8
    assert result.metadata["tangent_adjoint_abs_error"] < 1.0e-12
    np.testing.assert_allclose(
        result.derivative,
        result.metadata["finite_difference_derivative"],
        rtol=0.0,
        atol=1.0e-8,
    )


def test_implicit_linear_observable_builder_path_matches_direct_path() -> None:
    a0, ap, b0, bp, c0, cp, offset0, offsetp, p0 = _linear_system_components()

    def build_system(p: float) -> LinearObservableSystem:
        return LinearObservableSystem(
            parameter=float(p),
            matrix=a0 + float(p) * ap,
            rhs=b0 + float(p) * bp,
            matrix_derivative=ap,
            rhs_derivative=bp,
            observable_vector=c0 + float(p) * cp,
            observable_vector_derivative=cp,
            observable_offset=offset0 + float(p) * offsetp,
            observable_offset_derivative=offsetp,
            metadata={"builder": "unit_test"},
        )

    result = implicit_linear_observable_derivative_from_builder(
        build_system,
        parameter=p0,
        finite_difference_step=1.0e-6,
        metadata={"caller": "ambipolar_lane"},
    )

    direct = implicit_linear_observable_derivative(
        matrix=a0 + p0 * ap,
        rhs=b0 + p0 * bp,
        matrix_derivative=ap,
        rhs_derivative=bp,
        observable_vector=c0 + p0 * cp,
        observable_vector_derivative=cp,
        observable_offset=offset0 + p0 * offsetp,
        observable_offset_derivative=offsetp,
        parameter=p0,
        finite_difference_step=None,
    )
    assert result.metadata["builder"] == "unit_test"
    assert result.metadata["caller"] == "ambipolar_lane"
    np.testing.assert_allclose(result.observable, direct.observable, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(result.derivative, direct.derivative, rtol=0.0, atol=1.0e-12)
    assert result.finite_difference_abs_error is not None
    assert result.finite_difference_abs_error < 1.0e-8


def test_implicit_linear_radial_current_builder_adapts_to_ambipolar_contract() -> None:
    a0, ap, b0, bp, c0, cp, offset0, offsetp, p0 = _linear_system_components()

    def build_system(er: float) -> LinearObservableSystem:
        return LinearObservableSystem(
            parameter=float(er),
            matrix=a0 + float(er) * ap,
            rhs=b0 + float(er) * bp,
            matrix_derivative=ap,
            rhs_derivative=bp,
            observable_vector=c0 + float(er) * cp,
            observable_vector_derivative=cp,
            observable_offset=offset0 + float(er) * offsetp,
            observable_offset_derivative=offsetp,
            metadata={"operator_owner": "rhsmode1"},
        )

    result = implicit_linear_radial_current_derivative_from_builder(
        build_system,
        er=p0,
        finite_difference_step=1.0e-6,
        metadata={"observable": "caller_metadata_should_not_override"},
    )

    assert isinstance(result, RadialCurrentDerivativeResult)
    assert result.scheme == "implicit_linear_adjoint"
    assert result.metadata["operator_owner"] == "rhsmode1"
    assert result.metadata["observable"] != "caller_metadata_should_not_override"
    assert result.metadata["finite_difference_abs_error"] < 1.0e-8


def test_matrix_free_linear_observable_builder_matches_dense_certificate() -> None:
    a0, ap, b0, bp, c0, cp, offset0, offsetp, p0 = _linear_system_components()

    def build_system(p: float) -> MatrixFreeLinearObservableSystem:
        a = a0 + float(p) * ap
        return MatrixFreeLinearObservableSystem(
            parameter=float(p),
            size=int(a.shape[0]),
            rhs=b0 + float(p) * bp,
            rhs_derivative=bp,
            apply=lambda x: a @ x,
            transpose_apply=lambda x: a.T @ x,
            derivative_apply=lambda x: ap @ x,
            solve=lambda rhs: jnp.linalg.solve(a, rhs),
            transpose_solve=lambda rhs: jnp.linalg.solve(a.T, rhs),
            observable_vector=c0 + float(p) * cp,
            observable_vector_derivative=cp,
            observable_offset=offset0 + float(p) * offsetp,
            observable_offset_derivative=offsetp,
            metadata={"builder": "matrix_free_unit_test"},
        )

    matrix_free = implicit_matrix_free_linear_observable_derivative_from_builder(
        build_system,
        parameter=p0,
        finite_difference_step=1.0e-6,
        metadata={"caller": "production_contract"},
    )
    dense = implicit_linear_observable_derivative(
        matrix=a0 + p0 * ap,
        rhs=b0 + p0 * bp,
        matrix_derivative=ap,
        rhs_derivative=bp,
        observable_vector=c0 + p0 * cp,
        observable_vector_derivative=cp,
        observable_offset=offset0 + p0 * offsetp,
        observable_offset_derivative=offsetp,
        parameter=p0,
        finite_difference_observable=lambda p: jnp.vdot(
            c0 + float(p) * cp,
            jnp.linalg.solve(a0 + float(p) * ap, b0 + float(p) * bp),
        )
        + offset0
        + float(p) * offsetp,
        finite_difference_step=1.0e-6,
    )

    assert matrix_free.metadata["builder"] == "matrix_free_unit_test"
    assert matrix_free.metadata["caller"] == "production_contract"
    assert matrix_free.metadata["system_kind"] == "matrix_free_linear_observable"
    np.testing.assert_allclose(matrix_free.observable, dense.observable, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(matrix_free.derivative, dense.derivative, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        matrix_free.finite_difference_derivative,
        dense.finite_difference_derivative,
        rtol=0.0,
        atol=1.0e-8,
    )
    assert matrix_free.primal_residual_norm < 1.0e-12
    assert matrix_free.tangent_residual_norm < 1.0e-12
    assert matrix_free.adjoint_residual_norm < 1.0e-12
    assert matrix_free.tangent_adjoint_abs_error < 1.0e-12
    assert matrix_free.finite_difference_abs_error is not None
    assert matrix_free.finite_difference_abs_error < 1.0e-8


def test_matrix_free_radial_current_builder_adapts_to_ambipolar_contract() -> None:
    a0, ap, b0, bp, c0, cp, offset0, offsetp, p0 = _linear_system_components()

    def build_system(er: float) -> MatrixFreeLinearObservableSystem:
        a = a0 + float(er) * ap
        return MatrixFreeLinearObservableSystem(
            parameter=float(er),
            size=int(a.shape[0]),
            rhs=b0 + float(er) * bp,
            rhs_derivative=bp,
            apply=lambda x: a @ x,
            transpose_apply=lambda x: a.T @ x,
            derivative_apply=lambda x: ap @ x,
            solve=lambda rhs: jnp.linalg.solve(a, rhs),
            transpose_solve=lambda rhs: jnp.linalg.solve(a.T, rhs),
            observable_vector=c0 + float(er) * cp,
            observable_vector_derivative=cp,
            observable_offset=offset0 + float(er) * offsetp,
            observable_offset_derivative=offsetp,
            metadata={"operator_owner": "matrix_free_rhsmode1"},
        )

    result = implicit_matrix_free_radial_current_derivative_from_builder(
        build_system,
        er=p0,
        finite_difference_step=1.0e-6,
        metadata={"gate": "ambipolar_option13"},
    )

    assert isinstance(result, RadialCurrentDerivativeResult)
    assert result.scheme == "implicit_linear_adjoint"
    assert result.step == 1.0e-6
    assert result.metadata["operator_owner"] == "matrix_free_rhsmode1"
    assert result.metadata["gate"] == "ambipolar_option13"
    assert result.metadata["system_kind"] == "matrix_free_linear_observable"
    assert result.metadata["finite_difference_abs_error"] < 1.0e-8
    assert result.metadata["tangent_adjoint_abs_error"] < 1.0e-12


def test_probe_linear_observable_vector_recovers_chunked_weights_and_offset() -> None:
    weights = jnp.asarray([0.3, -0.4, 1.7, 0.0, -2.0], dtype=jnp.float64)
    offset = 0.125

    def observable(state: jnp.ndarray) -> jnp.ndarray:
        return jnp.vdot(weights, state) + offset

    vector, probed_offset = probe_linear_observable_vector(
        observable,
        size=int(weights.size),
        chunk_size=2,
    )

    np.testing.assert_allclose(vector, weights, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(probed_offset, offset, rtol=0.0, atol=1.0e-12)


def test_rhs1_radial_current_observable_vector_matches_existing_diagnostic() -> None:
    input_path = Path(__file__).parent / "ref" / "pas_1species_PAS_noEr_tiny.input.namelist"
    op = full_system_operator_from_namelist(nml=read_sfincs_input(input_path))
    rng = np.random.default_rng(7)
    state = jnp.asarray(rng.normal(size=(int(op.total_size),)), dtype=jnp.float64)

    vector, offset = radial_current_vm_psi_hat_observable_vector(op, chunk_size=11)
    probed = jnp.vdot(vector, state) + offset
    direct = radial_current_vm_psi_hat_from_state(op, x_full=state)

    np.testing.assert_allclose(probed, direct, rtol=0.0, atol=1.0e-10)

    vector_rhat, offset_rhat = radial_current_vm_observable_vector(
        op,
        radial_coordinate="rHat",
        psi_a_hat=-0.384935,
        a_hat=0.5109,
        r_n=0.5,
        chunk_size=11,
    )
    probed_rhat = jnp.vdot(vector_rhat, state) + offset_rhat
    direct_rhat = radial_current_vm_from_state(
        op,
        x_full=state,
        radial_coordinate="rHat",
        psi_a_hat=-0.384935,
        a_hat=0.5109,
        r_n=0.5,
    )
    np.testing.assert_allclose(probed_rhat, direct_rhat, rtol=0.0, atol=1.0e-10)


def test_dense_rhs1_radial_current_linear_observable_system_matches_finite_difference() -> None:
    input_path = Path(__file__).parent / "ref" / "pas_1species_PAS_noEr_tiny.input.namelist"
    op0 = full_system_operator_from_namelist(nml=read_sfincs_input(input_path))
    step = 1.0e-5
    parameter = 0.2

    def op_at(value: float):
        return replace(
            op0,
            dn_hat_dpsi_hat=op0.dn_hat_dpsi_hat + float(value) * jnp.ones_like(op0.dn_hat_dpsi_hat),
        )

    def build_system(value: float) -> LinearObservableSystem:
        return dense_rhs1_vm_radial_current_linear_observable_system(
            op=op_at(value),
            op_plus=op_at(float(value) + step),
            op_minus=op_at(float(value) - step),
            parameter=float(value),
            derivative_step=step,
            radial_coordinate="rHat",
            psi_a_hat=-0.384935,
            a_hat=0.5109,
            r_n=0.5,
            max_size=400,
            observable_chunk_size=17,
        )

    result = implicit_linear_observable_derivative_from_builder(
        build_system,
        parameter=parameter,
        finite_difference_step=step,
    )

    assert result.metadata["builder"] == "dense_rhs1_vm_radial_current"
    assert result.primal_residual_norm < 1.0e-8
    assert result.tangent_residual_norm < 1.0e-7
    assert result.adjoint_residual_norm < 1.0e-8
    assert result.tangent_adjoint_abs_error < 1.0e-6
    assert result.finite_difference_abs_error is not None
    assert result.finite_difference_abs_error < 1.0e-4


def test_matrix_free_rhs1_radial_current_system_matches_dense_certificate() -> None:
    input_path = Path(__file__).parent / "ref" / "pas_1species_PAS_noEr_tiny.input.namelist"
    op0 = full_system_operator_from_namelist(nml=read_sfincs_input(input_path))
    step = 1.0e-5
    parameter = 0.2

    def op_at(value: float):
        return replace(
            op0,
            dn_hat_dpsi_hat=op0.dn_hat_dpsi_hat + float(value) * jnp.ones_like(op0.dn_hat_dpsi_hat),
        )

    def dense_system(value: float) -> LinearObservableSystem:
        return dense_rhs1_vm_radial_current_linear_observable_system(
            op=op_at(value),
            op_plus=op_at(float(value) + step),
            op_minus=op_at(float(value) - step),
            parameter=float(value),
            derivative_step=step,
            radial_coordinate="rHat",
            psi_a_hat=-0.384935,
            a_hat=0.5109,
            r_n=0.5,
            max_size=400,
            observable_chunk_size=17,
        )

    def matrix_free_system(value: float) -> MatrixFreeLinearObservableSystem:
        # The small-deck test uses dense closures only to validate the
        # matrix-free production contract without relying on iterative solver
        # tolerances.
        dense_for_closures = dense_system(float(value))
        matrix = dense_for_closures.matrix
        op_plus = op_at(float(value) + step)
        op_minus = op_at(float(value) - step)
        return matrix_free_rhs1_vm_radial_current_linear_observable_system(
            op=op_at(value),
            op_plus=op_plus,
            op_minus=op_minus,
            parameter=float(value),
            derivative_step=step,
            solve=lambda rhs: jnp.linalg.solve(matrix, rhs),
            transpose_solve=lambda rhs: jnp.linalg.solve(matrix.T, rhs),
            transpose_apply=lambda vector: matrix.T @ vector,
            operator_tangent=operator_tangent_from_centered_difference(op_plus, op_minus, step),
            radial_coordinate="rHat",
            psi_a_hat=-0.384935,
            a_hat=0.5109,
            r_n=0.5,
            observable_chunk_size=17,
            metadata={"gate": "rhs1_matrix_free_production_contract"},
        )

    matrix_free_result = implicit_matrix_free_linear_observable_derivative_from_builder(
        matrix_free_system,
        parameter=parameter,
        finite_difference_step=step,
    )
    dense_result = implicit_linear_observable_derivative_from_builder(
        dense_system,
        parameter=parameter,
        finite_difference_step=step,
    )

    assert matrix_free_result.metadata["builder"] == "matrix_free_rhs1_vm_radial_current"
    assert matrix_free_result.metadata["gate"] == "rhs1_matrix_free_production_contract"
    assert matrix_free_result.metadata["dense_matrix_assembled"] is False
    assert matrix_free_result.metadata["operator_derivative_action"] == "jax_jvp_operator_tangent"
    assert matrix_free_result.metadata["rhs_derivative"] == "jax_jvp_operator_tangent"
    assert matrix_free_result.primal_residual_norm < 1.0e-8
    assert matrix_free_result.tangent_residual_norm < 1.0e-7
    assert matrix_free_result.adjoint_residual_norm < 1.0e-8
    assert matrix_free_result.tangent_adjoint_abs_error < 1.0e-6
    np.testing.assert_allclose(
        matrix_free_result.observable,
        dense_result.observable,
        rtol=0.0,
        atol=1.0e-8,
    )
    np.testing.assert_allclose(
        matrix_free_result.derivative,
        dense_result.derivative,
        rtol=0.0,
        atol=1.0e-6,
    )


def test_rhs1_operator_tangent_jvp_matches_centered_difference_for_er_xdot() -> None:
    input_path = Path(__file__).parent / "ref" / "er_xdot_1species_tiny.input.namelist"
    op0 = full_system_operator_from_namelist(nml=read_sfincs_input(input_path))
    assert op0.fblock.er_xdot is not None
    step = 1.0e-5

    def op_at(value: float):
        er_xdot = replace(
            op0.fblock.er_xdot,
            dphi_hat_dpsi_hat=op0.fblock.er_xdot.dphi_hat_dpsi_hat + float(value),
        )
        return replace(op0, fblock=replace(op0.fblock, er_xdot=er_xdot))

    op_plus = op_at(step)
    op_minus = op_at(-step)
    op_tangent = operator_tangent_from_centered_difference(op_plus, op_minus, step)
    state = jnp.linspace(0.1, 1.0, int(op0.total_size), dtype=jnp.float64)
    _, jvp_action = jax.jvp(
        lambda operator: apply_v3_full_system_operator_cached(operator, state),
        (op0,),
        (op_tangent,),
    )
    finite_difference_action = (
        apply_v3_full_system_operator_cached(op_plus, state)
        - apply_v3_full_system_operator_cached(op_minus, state)
    ) / (2.0 * step)

    assert float(jnp.linalg.norm(finite_difference_action)) > 0.0
    np.testing.assert_allclose(
        jvp_action,
        finite_difference_action,
        rtol=0.0,
        atol=1.0e-9,
    )


def test_analytic_er_operator_tangent_matches_centered_er_difference() -> None:
    input_path = Path(__file__).parent / "ref" / "er_xdot_1species_tiny.input.namelist"
    nml = read_sfincs_input(input_path)
    op0 = full_system_operator_from_namelist(nml=nml)
    assert op0.fblock.er_xdot is not None
    step = 1.0e-5
    dphi_d_er = dphi_hat_dpsi_hat_er_derivative_from_namelist(nml)

    def replace_dphi(candidate, delta: float):
        if candidate is None or not hasattr(candidate, "dphi_hat_dpsi_hat"):
            return candidate
        return replace(
            candidate,
            dphi_hat_dpsi_hat=candidate.dphi_hat_dpsi_hat + float(delta),
        )

    def op_at_er_delta(delta_er: float):
        dphi_delta = float(delta_er) * float(dphi_d_er)
        fblock = replace(
            op0.fblock,
            exb_theta=replace_dphi(op0.fblock.exb_theta, dphi_delta),
            exb_zeta=replace_dphi(op0.fblock.exb_zeta, dphi_delta),
            er_xidot=replace_dphi(op0.fblock.er_xidot, dphi_delta),
            er_xdot=replace_dphi(op0.fblock.er_xdot, dphi_delta),
        )
        return replace(
            op0,
            fblock=fblock,
            dphi_hat_dpsi_hat=op0.dphi_hat_dpsi_hat + dphi_delta,
        )

    op_plus = op_at_er_delta(step)
    op_minus = op_at_er_delta(-step)
    analytic_tangent = er_operator_tangent_from_dphi_hat_dpsi_hat_derivative(op0, dphi_d_er)
    centered_tangent = operator_tangent_from_centered_difference(op_plus, op_minus, step)
    state = jnp.linspace(0.1, 1.0, int(op0.total_size), dtype=jnp.float64)
    _, analytic_action = jax.jvp(
        lambda operator: apply_v3_full_system_operator_cached(operator, state),
        (op0,),
        (analytic_tangent,),
    )
    _, centered_tangent_action = jax.jvp(
        lambda operator: apply_v3_full_system_operator_cached(operator, state),
        (op0,),
        (centered_tangent,),
    )
    centered_action = (
        apply_v3_full_system_operator_cached(op_plus, state)
        - apply_v3_full_system_operator_cached(op_minus, state)
    ) / (2.0 * step)

    assert float(abs(dphi_d_er)) > 0.0
    assert float(jnp.linalg.norm(centered_action)) > 0.0
    np.testing.assert_allclose(analytic_action, centered_tangent_action, rtol=0.0, atol=1.0e-9)
    np.testing.assert_allclose(analytic_action, centered_action, rtol=0.0, atol=1.0e-9)


def test_zero_er_fixed_shape_branch_tangent_matches_centered_er_difference() -> None:
    input_path = Path(__file__).parent / "ref" / "er_xdot_1species_tiny.input.namelist"
    step = 1.0e-5

    def op_from_er(er: float, *, keep_zero_er_terms: bool = False):
        nml = read_sfincs_input(input_path)
        nml.group("physicsParameters")["ER"] = float(er)
        return full_system_operator_from_namelist(
            nml=nml,
            keep_zero_er_terms=keep_zero_er_terms,
        )

    nml_zero = read_sfincs_input(input_path)
    nml_zero.group("physicsParameters")["ER"] = 0.0
    dphi_d_er = dphi_hat_dpsi_hat_er_derivative_from_namelist(nml_zero)
    op_default_zero = op_from_er(0.0)
    op_fixed_zero = op_from_er(0.0, keep_zero_er_terms=True)
    assert op_default_zero.fblock.er_xdot is None
    assert op_fixed_zero.fblock.er_xdot is not None

    state = jnp.linspace(0.1, 1.0, int(op_fixed_zero.total_size), dtype=jnp.float64)
    np.testing.assert_allclose(
        apply_v3_full_system_operator_cached(op_fixed_zero, state),
        apply_v3_full_system_operator_cached(op_default_zero, state),
        rtol=0.0,
        atol=1.0e-12,
    )

    op_plus = op_from_er(step, keep_zero_er_terms=True)
    op_minus = op_from_er(-step, keep_zero_er_terms=True)
    analytic_tangent = er_operator_tangent_from_dphi_hat_dpsi_hat_derivative(op_fixed_zero, dphi_d_er)
    _, analytic_action = jax.jvp(
        lambda operator: apply_v3_full_system_operator_cached(operator, state),
        (op_fixed_zero,),
        (analytic_tangent,),
    )
    centered_action = (
        apply_v3_full_system_operator_cached(op_plus, state)
        - apply_v3_full_system_operator_cached(op_minus, state)
    ) / (2.0 * step)

    assert float(jnp.linalg.norm(centered_action)) > 0.0
    np.testing.assert_allclose(analytic_action, centered_action, rtol=0.0, atol=1.0e-9)


def test_rhsmode1_namelist_response_uses_fixed_shape_er_derivative_provider() -> None:
    input_path = Path(__file__).parent / "ref" / "pas_1species_PAS_noEr_tiny_scheme1.input.namelist"
    step = 1.0e-5
    response = rhsmode1_radial_current_response_from_namelist(
        nml=input_path,
        derivative_step=step,
        finite_difference_step=step,
        keep_zero_er_terms=True,
        max_dense_size=150,
        observable_chunk_size=17,
        metadata={"gate": "fixed_shape_namelist_response"},
    )

    derivative = response.derivative(0.0)
    centered = (response.radial_current(step) - response.radial_current(-step)) / (2.0 * step)

    assert derivative.scheme == "implicit_linear_adjoint"
    assert derivative.metadata["builder"] == "rhsmode1_radial_current_response_from_namelist"
    assert derivative.metadata["response_builder"] == "rhsmode1_radial_current_response_from_namelist"
    assert derivative.metadata["gate"] == "fixed_shape_namelist_response"
    assert derivative.metadata["keep_zero_er_terms"] is True
    assert derivative.metadata["linear_algebra"] in {
        "bounded_dense_validation",
        "bounded_dense_active_validation",
    }
    assert derivative.metadata["operator_derivative_action"] == "jax_jvp_operator_tangent"
    assert derivative.metadata["rhs_derivative"] == "jax_jvp_operator_tangent"
    assert derivative.metadata["tangent_adjoint_abs_error"] < 1.0e-8
    assert derivative.metadata["finite_difference_abs_error"] < 2.0e-5
    np.testing.assert_allclose(derivative.derivative, centered, rtol=0.0, atol=2.0e-5)


def test_rhsmode1_namelist_response_replays_fortran_active_rn_current() -> None:
    input_path = (
        Path(__file__).parents[1]
        / "benchmarks"
        / "fortran_v3_ambipolar_reference"
        / "namelists"
        / "geometry1_helical_small_option1.namelist"
    )
    response = rhsmode1_radial_current_response_from_namelist(
        nml=input_path,
        derivative_step=1.0e-5,
        max_dense_size=1000,
        observable_chunk_size=128,
        metadata={"gate": "fortran_option1_active_rn_current"},
    )

    system = response.build_system(0.0)
    current = evaluate_matrix_free_linear_observable(system)
    derivative = response.derivative(0.0)
    fortran_implied_newton_slope = 1.078787197904619e-6 / 2.01579684708909

    assert system.metadata["linear_algebra"] == "bounded_dense_active_validation"
    assert system.metadata["active_dof"] is True
    assert system.metadata["active_size"] == 984
    assert system.metadata["full_size"] == 1474
    assert system.metadata["radial_coordinate"] == "rN"
    assert derivative.metadata["tangent_adjoint_abs_error"] < 1.0e-10
    assert derivative.metadata["finite_difference_abs_error"] < 1.0e-10
    np.testing.assert_allclose(current, 1.078787197904619e-6, rtol=2.0e-5, atol=0.0)
    np.testing.assert_allclose(
        derivative.derivative,
        fortran_implied_newton_slope,
        rtol=2.0e-5,
        atol=0.0,
    )


@pytest.mark.parametrize(
    ("case_name", "points"),
    [
        (
            "geometry1_helical_small_option3",
            (
                (0.0, 1.078787197904619e-6),
                (-2.01579684708909, -1.0650279455435228e-9),
            ),
        ),
        (
            "geometry4_w7x_like_small_option3",
            ((0.0, 2.513476802851773e-8),),
        ),
    ],
)
def test_rhsmode1_namelist_response_replays_fortran_option3_currents(
    case_name: str,
    points: tuple[tuple[float, float], ...],
) -> None:
    input_path = (
        Path(__file__).parents[1]
        / "benchmarks"
        / "fortran_v3_ambipolar_reference"
        / "namelists"
        / f"{case_name}.namelist"
    )
    response = rhsmode1_radial_current_response_from_namelist(
        nml=input_path,
        derivative_step=1.0e-5,
        max_dense_size=1000,
        observable_chunk_size=128,
        metadata={"gate": "fortran_option3_active_rn_current"},
    )

    for er, expected in points:
        current = response.radial_current(er)
        np.testing.assert_allclose(current, expected, rtol=2.0e-5, atol=0.0)


def test_rhs1_radial_current_jvp_vjp_dot_product_gate() -> None:
    input_path = Path(__file__).parent / "ref" / "pas_1species_PAS_noEr_tiny.input.namelist"
    op = full_system_operator_from_namelist(nml=read_sfincs_input(input_path))
    rng = np.random.default_rng(11)
    state = jnp.asarray(rng.normal(size=(int(op.total_size),)), dtype=jnp.float64)
    tangent = jnp.asarray(rng.normal(size=(int(op.total_size),)), dtype=jnp.float64)
    cotangent = jnp.asarray(1.25, dtype=jnp.float64)

    result = adjoint_dot_product_check(
        lambda x: radial_current_vm_from_state(
            op,
            x_full=x,
            radial_coordinate="rHat",
            psi_a_hat=-0.384935,
            a_hat=0.5109,
            r_n=0.5,
        ),
        state,
        tangent,
        cotangent,
    )

    assert result.abs_error < 1.0e-10
    np.testing.assert_allclose(result.lhs, result.rhs, rtol=0.0, atol=1.0e-10)


def test_rhs1_transport_diagnostic_jvp_vjp_dot_product_gates() -> None:
    input_path = Path(__file__).parent / "ref" / "pas_1species_PAS_noEr_tiny.input.namelist"
    op = full_system_operator_from_namelist(nml=read_sfincs_input(input_path))
    rng = np.random.default_rng(12)
    state = jnp.asarray(rng.normal(size=(int(op.total_size),)), dtype=jnp.float64)
    tangent = jnp.asarray(rng.normal(size=(int(op.total_size),)), dtype=jnp.float64)
    cotangent = jnp.asarray(0.75, dtype=jnp.float64)

    diagnostic_functions = {
        "particle_flux_vm_psi_hat": lambda x: v3_transport_diagnostics_vm_only(
            op, x_full=x
        ).particle_flux_vm_psi_hat[0],
        "heat_flux_vm_psi_hat": lambda x: v3_transport_diagnostics_vm_only(
            op, x_full=x
        ).heat_flux_vm_psi_hat[0],
        "fsab_flow": lambda x: v3_transport_diagnostics_vm_only(op, x_full=x).fsab_flow[0],
        "fsab_jhat": lambda x: jnp.vdot(
            op.z_s,
            v3_transport_diagnostics_vm_only(op, x_full=x).fsab_flow,
        ),
    }

    for name, diagnostic in diagnostic_functions.items():
        result = adjoint_dot_product_check(diagnostic, state, tangent, cotangent)
        assert result.abs_error < 1.0e-10, name


def test_implicit_linear_observable_derivative_rejects_incompatible_shapes() -> None:
    a0, ap, b0, bp, c0, _cp, _offset0, _offsetp, p0 = _linear_system_components()

    with pytest.raises(ValueError, match="rhs length"):
        implicit_linear_observable_derivative(
            matrix=a0 + p0 * ap,
            rhs=b0[:2],
            matrix_derivative=ap,
            rhs_derivative=bp,
            observable_vector=c0,
        )
