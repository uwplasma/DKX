from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp

from sfincs_jax.operators.profile_device_sparse import device_csr_from_scipy_csr
from sfincs_jax.solvers.preconditioner_qi_basis import RHS1QICoarseBasis, RHS1QICoarseBasisMetadata
from sfincs_jax.solvers.preconditioner_qi_device import (
    RHS1QIDeviceJacobiSmoother,
    RHS1QIDeviceJacobiSmootherMetadata,
    RHS1QIDeviceSmootherProbe,
    build_rhs1_qi_device_jacobi_smoother,
    extract_device_csr_diagonal,
    probe_rhs1_qi_device_smoother_correction,
    qi_device_solver_env,
    rhs1_qi_device_extra_coarse_controls,
    rhs1_qi_device_residual_correction_controls,
)
from sfincs_jax.solvers.preconditioner_qi_corrections import (
    RHS1QITwoLevelPreconditioner,
    RHS1QITwoLevelProbe,
    build_rhs1_qi_two_level_preconditioner,
    probe_rhs1_qi_two_level_correction,
)


def _basis(vectors: jnp.ndarray, label: str = "global") -> RHS1QICoarseBasis:
    vectors = jnp.asarray(vectors, dtype=jnp.float64)
    labels = tuple(f"{label}:{i}" for i in range(int(vectors.shape[1])))
    return RHS1QICoarseBasis(
        vectors=vectors,
        metadata=RHS1QICoarseBasisMetadata(
            total_size=int(vectors.shape[0]),
            candidate_count=int(vectors.shape[1]),
            rank=int(vectors.shape[1]),
            discarded_count=0,
            candidate_labels=labels,
            accepted_labels=labels,
            candidate_norms=tuple(1.0 for _ in labels),
            accepted_norms=tuple(1.0 for _ in labels),
            rank_rtol=1.0e-12,
            rank_atol=1.0e-14,
        ),
    )


def test_device_jacobi_smoother_reuses_csr_operator_and_is_jittable() -> None:
    matrix = sp.csr_matrix(
        [
            [4.0, 1.0, 0.0],
            [-1.0, 3.0, 0.5],
            [0.0, 0.25, 2.0],
        ],
        dtype=np.float64,
    )
    device_operator = device_csr_from_scipy_csr(matrix, max_csr_mb=1.0)
    smoother = build_rhs1_qi_device_jacobi_smoother(
        device_operator,
        damping=0.6,
        sweeps=3,
    )
    residual = jnp.asarray([1.0, -0.5, 0.75], dtype=jnp.float64)

    correction = jnp.zeros_like(residual)
    remaining = residual
    diagonal = jnp.asarray(matrix.diagonal(), dtype=jnp.float64)
    dense = jnp.asarray(matrix.toarray(), dtype=jnp.float64)
    for _ in range(3):
        step = 0.6 * remaining / diagonal
        correction = correction + step
        remaining = remaining - dense @ step

    eager = smoother.apply(residual)
    compiled = jax.jit(smoother.as_preconditioner())(residual)

    assert isinstance(smoother, RHS1QIDeviceJacobiSmoother)
    assert isinstance(smoother.metadata, RHS1QIDeviceJacobiSmootherMetadata)
    assert smoother.metadata.device_resident is True
    assert smoother.metadata.reason == "built"
    assert smoother.metadata.valid_diagonal_count == 3
    assert smoother.metadata.to_dict()["step_policy"] == "stationary"
    np.testing.assert_allclose(smoother.diagonal, diagonal)
    np.testing.assert_allclose(eager, correction, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(compiled, eager, rtol=1.0e-12, atol=1.0e-12)


def test_device_jacobi_smoother_rejects_invalid_diagonal_by_default() -> None:
    matrix = sp.csr_matrix(
        [
            [0.0, 2.0],
            [1.0, 4.0],
        ],
        dtype=np.float64,
    )
    device_operator = device_csr_from_scipy_csr(matrix, max_csr_mb=1.0)

    diagonal, hit_count = extract_device_csr_diagonal(device_operator)
    np.testing.assert_allclose(diagonal, jnp.asarray([0.0, 4.0], dtype=jnp.float64))
    np.testing.assert_array_equal(hit_count, jnp.asarray([0, 1], dtype=jnp.int32))
    with pytest.raises(ValueError, match="invalid diagonal"):
        build_rhs1_qi_device_jacobi_smoother(device_operator)

    smoother = build_rhs1_qi_device_jacobi_smoother(
        device_operator,
        damping=0.5,
        require_all_diagonal=False,
    )

    assert smoother.metadata.reason == "partial_diagonal"
    assert smoother.metadata.missing_diagonal_count == 1
    np.testing.assert_allclose(
        smoother.apply(jnp.asarray([2.0, 8.0], dtype=jnp.float64)),
        jnp.asarray([0.0, 1.0], dtype=jnp.float64),
    )


def test_device_jacobi_smoother_validates_controls_and_shapes() -> None:
    matrix = sp.csr_matrix(np.eye(2, dtype=np.float64))
    device_operator = device_csr_from_scipy_csr(matrix, max_csr_mb=1.0)

    with pytest.raises(ValueError, match="damping"):
        build_rhs1_qi_device_jacobi_smoother(device_operator, damping=0.0)
    with pytest.raises(ValueError, match="step_policy"):
        build_rhs1_qi_device_jacobi_smoother(device_operator, step_policy="unknown")
    with pytest.raises(ValueError, match="max_step_damping"):
        build_rhs1_qi_device_jacobi_smoother(
            device_operator,
            max_step_damping=0.0,
        )
    with pytest.raises(ValueError, match="min_step_denominator"):
        build_rhs1_qi_device_jacobi_smoother(
            device_operator,
            min_step_denominator=float("inf"),
        )

    smoother = build_rhs1_qi_device_jacobi_smoother(device_operator)
    with pytest.raises(ValueError, match="residual length"):
        smoother.apply(jnp.ones((3,), dtype=jnp.float64))

    bad_policy = replace(
        smoother,
        metadata=replace(smoother.metadata, step_policy="unsupported"),
    )
    with pytest.raises(ValueError, match="unsupported device smoother"):
        bad_policy.apply(jnp.ones((2,), dtype=jnp.float64))


def test_device_jacobi_diagonal_edge_cases_are_reported() -> None:
    rectangular = device_csr_from_scipy_csr(
        sp.csr_matrix(np.ones((2, 3), dtype=np.float64)),
        max_csr_mb=1.0,
    )
    with pytest.raises(ValueError, match="square operator"):
        extract_device_csr_diagonal(rectangular)

    zero_operator = device_csr_from_scipy_csr(
        sp.csr_matrix((2, 2), dtype=np.float64),
        max_csr_mb=1.0,
    )
    diagonal, hit_count = extract_device_csr_diagonal(zero_operator)
    np.testing.assert_allclose(diagonal, jnp.zeros((2,), dtype=jnp.float64), atol=0.0)
    np.testing.assert_array_equal(hit_count, jnp.zeros((2,), dtype=jnp.int32))

    smoother = build_rhs1_qi_device_jacobi_smoother(
        zero_operator,
        require_all_diagonal=False,
    )
    assert smoother.metadata.reason == "empty_or_invalid_diagonal"
    assert smoother.metadata.to_dict()["valid_diagonal_count"] == 0


def test_device_jacobi_smoother_probe_guard_branches() -> None:
    matrix = sp.csr_matrix(np.eye(2, dtype=np.float64))
    device_operator = device_csr_from_scipy_csr(matrix, max_csr_mb=1.0)
    smoother = build_rhs1_qi_device_jacobi_smoother(device_operator)
    rhs = jnp.asarray([1.0, -1.0], dtype=jnp.float64)

    with pytest.raises(ValueError, match="same shape"):
        probe_rhs1_qi_device_smoother_correction(
            rhs=rhs,
            x0=jnp.zeros((3,), dtype=jnp.float64),
            smoother=smoother,
        )

    unchanged, zero_probe = probe_rhs1_qi_device_smoother_correction(
        rhs=rhs,
        x0=rhs,
        smoother=smoother,
    )
    assert zero_probe.reason == "zero_residual"
    assert zero_probe.to_dict()["improvement_ratio"] is None
    np.testing.assert_allclose(unchanged, rhs, atol=0.0)

    unchanged, nonfinite_probe = probe_rhs1_qi_device_smoother_correction(
        rhs=rhs,
        x0=jnp.zeros_like(rhs),
        smoother=smoother,
        operator=lambda x: jnp.where(
            jnp.all(jnp.asarray(x, dtype=jnp.float64) == 0.0),
            jnp.asarray(x, dtype=jnp.float64),
            jnp.full_like(jnp.asarray(x, dtype=jnp.float64), jnp.nan),
        ),
    )
    assert nonfinite_probe.reason == "nonfinite_candidate"
    assert nonfinite_probe.residual_after_norm == pytest.approx(
        nonfinite_probe.residual_before_norm
    )
    np.testing.assert_allclose(unchanged, jnp.zeros_like(rhs), atol=0.0)


def test_qi_device_environment_controls_normalize_all_supported_kinds(monkeypatch) -> None:
    prefix = "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_"
    monkeypatch.setenv(f"{prefix}MULTILEVEL_CURRENT_MOMENTS", "1")
    monkeypatch.setenv(f"{prefix}GLOBAL_MOMENT_RESIDUAL_EQUATION_MAX_RANK", "7")
    monkeypatch.setenv(f"{prefix}GLOBAL_MOMENT_RESIDUAL_EQUATION_SOLVER", "schur")
    monkeypatch.setenv(f"{prefix}PHASE_SPACE_RESIDUAL_EQUATION_BOUNDARY", "0.25")
    monkeypatch.setenv(
        f"{prefix}RESIDUAL_REGION_BOUNCE_COARSE_REGION_BANDS",
        "trapped,passing",
    )
    monkeypatch.setenv(f"{prefix}COUPLED_RESIDUAL_EQUATION_SOLVER", "least-squares")

    extra = rhs1_qi_device_extra_coarse_controls()
    residual = rhs1_qi_device_residual_correction_controls()

    assert extra["multilevel_current_moments"] is True
    assert extra["global_moment_residual_equation_max_rank"] == 7
    assert extra["global_moment_residual_equation_solver"] == "galerkin"
    assert extra["phase_space_residual_equation_boundary"] == pytest.approx(0.25)
    assert extra["residual_region_bounce_coarse_region_bands"] == "trapped,passing"
    assert residual["coupled_residual_equation_solver"] == "action_lstsq"

    monkeypatch.setenv(f"{prefix}COUPLED_RESIDUAL_EQUATION_SOLVER", "unsupported")
    assert (
        qi_device_solver_env(
            f"{prefix}COUPLED_RESIDUAL_EQUATION_SOLVER",
            default="galerkin",
        )
        == "galerkin"
    )


def test_device_jacobi_residual_minimizing_policy_reduces_and_fails_closed() -> None:
    matrix = sp.csr_matrix(
        [
            [1.0, 2.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    device_operator = device_csr_from_scipy_csr(matrix, max_csr_mb=1.0)
    rhs = jnp.asarray([0.0, 1.0], dtype=jnp.float64)
    x0 = jnp.zeros_like(rhs)

    stationary = build_rhs1_qi_device_jacobi_smoother(
        device_operator,
        damping=1.0,
        sweeps=1,
    )
    minres = build_rhs1_qi_device_jacobi_smoother(
        device_operator,
        damping=1.0,
        sweeps=1,
        step_policy="residual_minimizing",
    )

    stationary_residual_norm = float(
        jnp.linalg.norm(rhs - device_operator.matvec(stationary.apply(rhs)))
    )
    x, probe = probe_rhs1_qi_device_smoother_correction(
        rhs=rhs,
        x0=x0,
        smoother=minres,
        min_relative_improvement=0.05,
    )
    rejected_x, rejected_probe = probe_rhs1_qi_device_smoother_correction(
        rhs=rhs,
        x0=x0,
        smoother=minres,
        min_relative_improvement=0.2,
    )
    compiled = jax.jit(minres.as_preconditioner())(rhs)

    assert stationary.metadata.step_policy == "stationary"
    assert minres.metadata.step_policy == "residual_minimizing"
    assert isinstance(probe, RHS1QIDeviceSmootherProbe)
    assert isinstance(rejected_probe, RHS1QIDeviceSmootherProbe)
    assert stationary_residual_norm == pytest.approx(2.0)
    assert probe.accepted is True
    assert probe.reason == "residual_reduced"
    assert probe.improvement_ratio == pytest.approx(np.sqrt(0.8), rel=1.0e-12)
    assert probe.residual_after_norm < 0.9 * probe.residual_before_norm
    assert float(jnp.linalg.norm(rhs - device_operator.matvec(x))) == pytest.approx(
        probe.residual_after_norm
    )
    assert rejected_probe.accepted is False
    assert rejected_probe.reason == "residual_not_reduced"
    assert rejected_probe.improvement_ratio == pytest.approx(probe.improvement_ratio)
    np.testing.assert_allclose(rejected_x, x0, atol=0.0)
    np.testing.assert_allclose(compiled, minres.apply(rhs), rtol=1.0e-12, atol=1.0e-12)


def test_device_jacobi_smoother_feeds_two_level_qi_probe() -> None:
    n = 6
    diag = jnp.linspace(1.5, 3.0, n, dtype=jnp.float64)
    global_mode = jnp.ones((n,), dtype=jnp.float64)
    global_mode = global_mode / jnp.linalg.norm(global_mode)
    dense = jnp.diag(diag) + 0.85 * jnp.outer(global_mode, global_mode)
    matrix = sp.csr_matrix(np.asarray(dense), dtype=np.float64)
    device_operator = device_csr_from_scipy_csr(matrix, max_csr_mb=1.0)
    smoother = build_rhs1_qi_device_jacobi_smoother(
        device_operator,
        damping=1.0,
        sweeps=2,
    )
    rhs = jnp.asarray([1.0, 0.4, -0.6, 0.8, -0.2, 0.5], dtype=jnp.float64)
    x0 = jnp.zeros_like(rhs)
    basis = _basis(global_mode.reshape((-1, 1)))

    local_seed = smoother.apply(rhs)
    local_residual_norm = float(jnp.linalg.norm(rhs - device_operator.matvec(local_seed)))
    preconditioner = build_rhs1_qi_two_level_preconditioner(
        operator=device_operator.matvec,
        local_smoother=smoother.apply,
        basis=basis,
        coarse_solver="action_lstsq",
    )
    x, probe = probe_rhs1_qi_two_level_correction(
        operator=device_operator.matvec,
        rhs=rhs,
        x0=x0,
        preconditioner=preconditioner,
        min_relative_improvement=0.05,
    )

    assert isinstance(preconditioner, RHS1QITwoLevelPreconditioner)
    assert isinstance(probe, RHS1QITwoLevelProbe)
    assert probe.accepted is True
    assert probe.residual_after_norm < 0.75 * local_residual_norm
    assert probe.metadata.rank == 1
    np.testing.assert_allclose(
        jnp.linalg.norm(rhs - device_operator.matvec(x)),
        probe.residual_after_norm,
    )
