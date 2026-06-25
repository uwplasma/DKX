from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from sfincs_jax.problems.ambipolar import (
    RadialCurrentDerivativeResult,
    dense_rhs1_vm_radial_current_linear_observable_system,
    implicit_linear_radial_current_derivative,
    implicit_linear_radial_current_derivative_from_builder,
)
from sfincs_jax.problems.transport_matrix.diagnostics import (
    radial_current_vm_from_state,
    radial_current_vm_observable_vector,
    radial_current_vm_psi_hat_from_state,
    radial_current_vm_psi_hat_observable_vector,
)
from sfincs_jax.sensitivity import (
    LinearObservableSystem,
    implicit_linear_observable_derivative,
    implicit_linear_observable_derivative_from_builder,
    probe_linear_observable_vector,
)
from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.v3_system import full_system_operator_from_namelist


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
