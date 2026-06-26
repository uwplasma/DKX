from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

import sfincs_jax.v3_driver as v3_driver
from sfincs_jax.solvers.preconditioner_qi_basis import RHS1QICoarseBasis, RHS1QICoarseBasisMetadata
from sfincs_jax.solvers.preconditioner_qi_corrections import (
    build_rhs1_xblock_device_global_coupling_preconditioner,
    build_rhs1_xblock_smoothed_global_coupling_preconditioner,
    build_rhs1_xblock_two_level_preconditioner,
    build_rhs1_qi_two_level_preconditioner,
    probe_rhs1_qi_two_level_correction,
)


def _basis(vectors: jnp.ndarray) -> RHS1QICoarseBasis:
    vectors = jnp.asarray(vectors, dtype=jnp.float64)
    return RHS1QICoarseBasis(
        vectors=vectors,
        metadata=RHS1QICoarseBasisMetadata(
            total_size=int(vectors.shape[0]),
            candidate_count=int(vectors.shape[1]),
            rank=int(vectors.shape[1]),
            discarded_count=0,
            candidate_labels=("global",),
            accepted_labels=("global",),
            candidate_norms=(1.0,),
            accepted_norms=(1.0,),
            rank_rtol=1.0e-12,
            rank_atol=1.0e-14,
        ),
    )


def _fake_xblock_operator() -> SimpleNamespace:
    n_species = 1
    n_x = 2
    n_xi = 2
    n_theta = 2
    n_zeta = 2
    f_size = n_species * n_x * n_xi * n_theta * n_zeta
    extra_size = 2
    return SimpleNamespace(
        n_species=n_species,
        n_x=n_x,
        n_xi=n_xi,
        n_theta=n_theta,
        n_zeta=n_zeta,
        f_size=f_size,
        phi1_size=0,
        extra_size=extra_size,
        total_size=f_size + extra_size,
        constraint_scheme=1,
        point_at_x0=True,
        x=jnp.asarray([0.0, 1.0], dtype=jnp.float64),
        fblock=SimpleNamespace(f_shape=(n_species, n_x, n_xi, n_theta, n_zeta)),
    )


def _xblock_matrix(size: int) -> jnp.ndarray:
    diagonal = jnp.linspace(1.0, 2.5, int(size), dtype=jnp.float64)
    u = jnp.linspace(-1.0, 1.0, int(size), dtype=jnp.float64)
    return jnp.diag(diagonal) + 0.08 * jnp.outer(u, u)


def test_two_level_qi_preconditioner_reduces_low_rank_residual() -> None:
    u = jnp.ones((6,), dtype=jnp.float64)
    diag = jnp.linspace(1.0, 2.0, 6, dtype=jnp.float64)
    a = jnp.diag(diag) + 0.35 * jnp.outer(u, u)
    q = (u / jnp.linalg.norm(u)).reshape((-1, 1))
    rhs = jnp.arange(1, 7, dtype=jnp.float64)
    x0 = jnp.zeros_like(rhs)

    def operator(x):
        return a @ x

    def local_smoother(r):
        return r / diag

    local_residual = rhs - operator(local_smoother(rhs))
    preconditioner = build_rhs1_qi_two_level_preconditioner(
        operator=operator,
        local_smoother=local_smoother,
        basis=_basis(q),
    )
    x, probe = probe_rhs1_qi_two_level_correction(
        operator=operator,
        rhs=rhs,
        x0=x0,
        preconditioner=preconditioner,
    )

    assert probe.accepted is True
    assert probe.reason == "residual_reduced"
    assert probe.metadata.rank == 1
    assert probe.residual_after_norm < float(jnp.linalg.norm(local_residual))
    assert float(jnp.linalg.norm(rhs - operator(x))) == probe.residual_after_norm


def test_two_level_qi_preconditioner_action_is_jittable() -> None:
    a = jnp.asarray(
        [
            [3.0, 0.5, 0.0],
            [0.5, 2.0, 0.25],
            [0.0, 0.25, 1.5],
        ],
        dtype=jnp.float64,
    )
    q = jnp.asarray([[1.0], [1.0], [1.0]], dtype=jnp.float64)
    q = q / jnp.linalg.norm(q)

    def operator(x):
        return a @ x

    def local_smoother(r):
        return r / jnp.diag(a)

    preconditioner = build_rhs1_qi_two_level_preconditioner(
        operator=operator,
        local_smoother=local_smoother,
        basis=_basis(q),
    )
    residual = jnp.asarray([1.0, -2.0, 0.5], dtype=jnp.float64)

    eager = preconditioner.apply(residual)
    compiled = jax.jit(preconditioner.apply)(residual)

    assert jnp.allclose(compiled, eager)


def test_two_level_qi_action_lstsq_handles_nonnormal_coarse_vectors() -> None:
    a = jnp.asarray(
        [
            [1.0, 100.0],
            [0.0, 1.0],
        ],
        dtype=jnp.float64,
    )
    q = jnp.asarray([[0.0], [1.0]], dtype=jnp.float64)
    rhs = jnp.asarray([1.0, 0.0], dtype=jnp.float64)

    def operator(x):
        return a @ x

    def zero_local_smoother(r):
        return jnp.zeros_like(r)

    projected = build_rhs1_qi_two_level_preconditioner(
        operator=operator,
        local_smoother=zero_local_smoother,
        basis=_basis(q),
        coarse_solver="projected",
    )
    action_lstsq = build_rhs1_qi_two_level_preconditioner(
        operator=operator,
        local_smoother=zero_local_smoother,
        basis=_basis(q),
        coarse_solver="action_lstsq",
    )

    projected_x, projected_probe = probe_rhs1_qi_two_level_correction(
        operator=operator,
        rhs=rhs,
        x0=jnp.zeros_like(rhs),
        preconditioner=projected,
    )
    action_x, action_probe = probe_rhs1_qi_two_level_correction(
        operator=operator,
        rhs=rhs,
        x0=jnp.zeros_like(rhs),
        preconditioner=action_lstsq,
    )

    assert projected_probe.accepted is False
    assert jnp.allclose(projected_x, jnp.zeros_like(rhs))
    assert action_probe.accepted is True
    assert action_probe.metadata.coarse_solver == "action_lstsq"
    assert action_probe.residual_after_norm < 0.02
    assert (
        float(jnp.linalg.norm(rhs - operator(action_x)))
        == action_probe.residual_after_norm
    )


def test_two_level_qi_probe_fails_closed_without_required_improvement() -> None:
    a = jnp.eye(4, dtype=jnp.float64)
    q = jnp.ones((4, 1), dtype=jnp.float64)
    q = q / jnp.linalg.norm(q)
    rhs = jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.float64)
    x0 = jnp.zeros_like(rhs)

    def operator(x):
        return a @ x

    def exact_local_smoother(r):
        return r

    preconditioner = build_rhs1_qi_two_level_preconditioner(
        operator=operator,
        local_smoother=exact_local_smoother,
        basis=_basis(q),
    )
    x, probe = probe_rhs1_qi_two_level_correction(
        operator=operator,
        rhs=rhs,
        x0=x0,
        preconditioner=preconditioner,
        min_relative_improvement=1.1,
    )

    assert probe.accepted is False
    assert probe.reason == "residual_not_reduced"
    assert jnp.allclose(x, x0)


def test_xblock_two_level_wrapper_records_metadata_without_driver_alias() -> None:
    op = _fake_xblock_operator()
    matrix = _xblock_matrix(op.total_size)
    rhs = jnp.arange(1, op.total_size + 1, dtype=jnp.float64)

    def matvec(value):
        return matrix @ value

    def base_preconditioner(value):
        return value / jnp.diag(matrix)

    preconditioner, metadata, stats = build_rhs1_xblock_two_level_preconditioner(
        op=op,
        rhs=rhs,
        matvec=matvec,
        base_preconditioner=base_preconditioner,
        mode="multiplicative",
        fsavg_lmax=1,
        max_extra_units=4,
        max_directions=12,
        rcond=1.0e-12,
        include_rhs=True,
    )
    out = preconditioner(rhs)

    assert out.shape == rhs.shape
    assert metadata["mode"] == "multiplicative"
    assert 1 <= metadata["rank"] <= metadata["basis_size"] <= 12
    assert metadata["expected_size"] == op.total_size
    assert stats["applies"] == 1
    assert stats["coarse_applies"] == 1
    assert np.all(np.isfinite(np.asarray(out)))
    assert not hasattr(v3_driver, "_build_rhs1_xblock_two_level_preconditioner")


def test_host_global_coupling_wrapper_records_metadata_and_stats(monkeypatch) -> None:
    op = _fake_xblock_operator()
    matrix = _xblock_matrix(op.total_size)
    rhs = jnp.arange(1, op.total_size + 1, dtype=jnp.float64)
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_SMOOTHER", "identity"
    )

    def matvec(value):
        return matrix @ value

    def base_preconditioner(value):
        return value / jnp.diag(matrix)

    preconditioner, metadata, stats = (
        build_rhs1_xblock_smoothed_global_coupling_preconditioner(
            op=op,
            rhs=rhs,
            matvec=matvec,
            base_preconditioner=base_preconditioner,
            mode="additive",
            fsavg_lmax=1,
            angular_lmax=1,
            max_extra_units=4,
            max_directions=14,
            rcond=1.0e-12,
            include_rhs=True,
        )
    )
    out = preconditioner(rhs)

    assert out.shape == rhs.shape
    assert metadata["mode"] == "additive"
    assert metadata["smoother"] == "identity"
    assert (
        metadata["load_basis_size"] >= metadata["basis_size"] >= metadata["rank"] >= 1
    )
    assert metadata["setup_budget_reached"] is False
    assert stats["applies"] == 1
    assert stats["coarse_applies"] == 1
    assert np.all(np.isfinite(np.asarray(out)))
    assert not hasattr(
        v3_driver,
        "_build_rhs1_xblock_smoothed_global_coupling_preconditioner",
    )


def test_device_global_coupling_wrapper_supports_qr_and_normal_equations(
    monkeypatch,
) -> None:
    op = _fake_xblock_operator()
    matrix = _xblock_matrix(op.total_size)
    rhs = jnp.arange(1, op.total_size + 1, dtype=jnp.float64)

    def matvec(value):
        return matrix @ value

    def base_preconditioner(value):
        return value / jnp.diag(matrix)

    for solver in ("qr", "normal_equations"):
        monkeypatch.setenv(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_DEVICE_SOLVER",
            solver,
        )
        monkeypatch.setenv(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_SMOOTHER",
            "identity",
        )
        preconditioner, metadata, stats = (
            build_rhs1_xblock_device_global_coupling_preconditioner(
                op=op,
                rhs=rhs,
                matvec=matvec,
                base_preconditioner=base_preconditioner,
                mode="multiplicative",
                fsavg_lmax=1,
                angular_lmax=1,
                max_extra_units=4,
                max_directions=10,
                rcond=1.0e-10,
                include_rhs=True,
            )
        )
        out = preconditioner(rhs)

        assert out.shape == rhs.shape
        assert metadata["device_resident"] is True
        assert metadata["smoother"] == "identity"
        assert metadata["coarse_solver"] == solver
        assert (
            metadata["load_basis_size"]
            >= metadata["basis_size"]
            >= metadata["rank"]
            >= 1
        )
        assert len(metadata["singular_values"]) >= 1
        assert stats["applies"] == 1
        assert stats["coarse_applies"] == 1
        assert np.all(np.isfinite(np.asarray(out)))

    assert not hasattr(
        v3_driver,
        "_build_rhs1_xblock_device_global_coupling_preconditioner",
    )
