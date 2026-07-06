from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sfincs_jax.solvers.preconditioner_qi_corrections import (
    RHS1QIDeflatedPreconditioner,
    RHS1QIDeflationMetadata,
    RHS1QIDeflationProbe,
    build_rhs1_qi_residual_deflated_preconditioner,
    probe_rhs1_qi_deflated_correction,
    probe_rhs1_qi_deflated_minres_seed,
)


def test_residual_deflation_reduces_coupled_slow_mode() -> None:
    diag = jnp.asarray([0.2, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=jnp.float64)
    slow = jnp.asarray([1.0, -0.8, 0.6, -0.4, 0.2, -0.1], dtype=jnp.float64)
    slow = slow / jnp.linalg.norm(slow)
    a = jnp.diag(diag) + 2.5 * jnp.outer(slow, slow)
    rhs = jnp.asarray([1.0, 0.5, -0.25, 0.75, -0.5, 0.25], dtype=jnp.float64)
    x0 = jnp.zeros_like(rhs)

    def operator(x):
        return a @ x

    def local_smoother(r):
        return r / diag

    local_only = local_smoother(rhs)
    local_residual = rhs - operator(local_only)
    preconditioner = build_rhs1_qi_residual_deflated_preconditioner(
        operator=operator,
        local_smoother=local_smoother,
        residual_seed=rhs,
        krylov_depth=4,
        max_rank=5,
    )
    x, probe = probe_rhs1_qi_deflated_correction(
        operator=operator,
        rhs=rhs,
        x0=x0,
        preconditioner=preconditioner,
        min_relative_improvement=0.05,
    )

    assert isinstance(preconditioner, RHS1QIDeflatedPreconditioner)
    assert isinstance(preconditioner.metadata, RHS1QIDeflationMetadata)
    assert isinstance(probe, RHS1QIDeflationProbe)
    assert preconditioner.metadata.to_dict()["reason"] == "built"
    assert probe.to_dict()["metadata"]["rank"] == preconditioner.metadata.rank
    assert preconditioner.metadata.rank >= 2
    assert probe.accepted is True
    assert probe.reason == "residual_reduced"
    assert probe.residual_after_norm < 0.1 * float(jnp.linalg.norm(local_residual))
    direct_residual_norm = float(jnp.linalg.norm(rhs - operator(x)))
    np.testing.assert_allclose(direct_residual_norm, probe.residual_after_norm)


def test_residual_deflation_action_is_jittable() -> None:
    a = jnp.asarray(
        [
            [4.0, 0.5, 0.0],
            [0.2, 2.0, 0.4],
            [0.1, 0.0, 1.5],
        ],
        dtype=jnp.float64,
    )
    residual = jnp.asarray([1.0, -2.0, 0.5], dtype=jnp.float64)

    def operator(x):
        return a @ x

    def local_smoother(r):
        return r / jnp.diag(a)

    preconditioner = build_rhs1_qi_residual_deflated_preconditioner(
        operator=operator,
        local_smoother=local_smoother,
        residual_seed=residual,
        krylov_depth=2,
        max_rank=3,
    )

    eager = preconditioner.apply(residual)
    compiled = jax.jit(preconditioner.apply)(residual)

    assert jnp.allclose(compiled, eager)
    assert preconditioner.metadata.device_resident is True


def test_residual_deflation_fails_closed_without_material_improvement() -> None:
    rhs = jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.float64)
    x0 = jnp.zeros_like(rhs)

    def operator(x):
        return x

    def exact_local_smoother(r):
        return r

    preconditioner = build_rhs1_qi_residual_deflated_preconditioner(
        operator=operator,
        local_smoother=exact_local_smoother,
        residual_seed=rhs,
        krylov_depth=2,
        max_rank=2,
    )
    x, probe = probe_rhs1_qi_deflated_correction(
        operator=operator,
        rhs=rhs,
        x0=x0,
        preconditioner=preconditioner,
        min_relative_improvement=1.1,
    )

    assert probe.accepted is False
    assert probe.reason == "residual_not_reduced"
    assert jnp.allclose(x, x0)


def test_residual_deflation_accepts_extra_block_schur_directions() -> None:
    a = jnp.asarray(
        [
            [2.0, 1.0, 0.0, 0.0],
            [1.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 1.5],
            [0.0, 0.0, 1.5, 3.0],
        ],
        dtype=jnp.float64,
    )
    residual = jnp.asarray([1.0, -1.0, 0.5, -0.5], dtype=jnp.float64)
    extra = (
        ("block0_constant", jnp.asarray([1.0, 1.0, 0.0, 0.0], dtype=jnp.float64)),
        ("block1_constant", jnp.asarray([0.0, 0.0, 1.0, 1.0], dtype=jnp.float64)),
    )

    def operator(x):
        return a @ x

    def diagonal_smoother(r):
        return r / jnp.diag(a)

    preconditioner = build_rhs1_qi_residual_deflated_preconditioner(
        operator=operator,
        local_smoother=diagonal_smoother,
        residual_seed=residual,
        extra_directions=extra,
        krylov_depth=0,
        max_rank=4,
    )

    labels = preconditioner.metadata.accepted_labels
    assert any(label == "extra:block0_constant" for label in labels)
    assert any(label == "extra:block1_constant" for label in labels)
    assert preconditioner.metadata.rank >= 2


def test_residual_deflation_cycles_improve_stationary_slow_mode() -> None:
    a = jnp.asarray(
        [
            [1.0, 0.55, 0.0],
            [0.55, 1.0, 0.25],
            [0.0, 0.25, 0.8],
        ],
        dtype=jnp.float64,
    )
    rhs = jnp.asarray([1.0, -0.4, 0.8], dtype=jnp.float64)

    def operator(x):
        return a @ x

    def underdamped_diagonal_smoother(r):
        return 0.35 * r / jnp.diag(a)

    one_cycle = build_rhs1_qi_residual_deflated_preconditioner(
        operator=operator,
        local_smoother=underdamped_diagonal_smoother,
        residual_seed=rhs,
        krylov_depth=0,
        max_rank=1,
        correction_cycles=1,
    )
    four_cycles = build_rhs1_qi_residual_deflated_preconditioner(
        operator=operator,
        local_smoother=underdamped_diagonal_smoother,
        residual_seed=rhs,
        krylov_depth=0,
        max_rank=1,
        correction_cycles=4,
    )

    residual_one = rhs - operator(one_cycle.apply(rhs))
    residual_four = rhs - operator(four_cycles.apply(rhs))

    assert four_cycles.metadata.correction_cycles == 4
    assert float(jnp.linalg.norm(residual_four)) < 0.6 * float(jnp.linalg.norm(residual_one))


def test_residual_deflation_minres_seed_accelerates_cycle_columns() -> None:
    a = jnp.asarray(
        [
            [3.0, 1.2, 0.0, 0.2],
            [-0.7, 2.2, 0.8, 0.0],
            [0.1, -0.4, 1.6, 0.9],
            [0.0, 0.3, -0.5, 1.4],
        ],
        dtype=jnp.float64,
    )
    rhs = jnp.asarray([1.0, -0.6, 0.8, -0.2], dtype=jnp.float64)
    x0 = jnp.zeros_like(rhs)

    def operator(x):
        return a @ x

    def weak_diagonal_smoother(r):
        return 0.4 * r / jnp.diag(a)

    preconditioner = build_rhs1_qi_residual_deflated_preconditioner(
        operator=operator,
        local_smoother=weak_diagonal_smoother,
        residual_seed=rhs,
        krylov_depth=1,
        max_rank=2,
        correction_cycles=4,
    )
    raw_seed = preconditioner.apply(rhs)
    raw_residual_norm = float(jnp.linalg.norm(rhs - operator(raw_seed)))
    x_minres, probe = probe_rhs1_qi_deflated_minres_seed(
        operator=operator,
        rhs=rhs,
        x0=x0,
        preconditioner=preconditioner,
        cycles=4,
        min_relative_improvement=0.2,
    )
    minres_residual_norm = float(jnp.linalg.norm(rhs - operator(x_minres)))

    assert probe.accepted is True
    assert probe.seed_solver == "cycle_minres"
    assert len(probe.cycle_residual_history) >= 2
    assert len(probe.cycle_coefficients) >= 1
    assert minres_residual_norm < 0.5 * raw_residual_norm


def test_residual_deflation_empty_rank_and_minres_seed_fail_closed() -> None:
    residual = jnp.asarray([1.0, -2.0, 3.0], dtype=jnp.float64)

    def operator(x):
        return jnp.asarray(x, dtype=jnp.float64)

    def zero_smoother(r):
        return jnp.zeros_like(jnp.asarray(r, dtype=jnp.float64))

    preconditioner = build_rhs1_qi_residual_deflated_preconditioner(
        operator=operator,
        local_smoother=zero_smoother,
        residual_seed=residual,
        krylov_depth=2,
        max_rank=4,
        include_raw_residual=False,
    )

    assert preconditioner.metadata.reason == "empty_rank"
    assert preconditioner.solve_coarse(residual).shape == (0,)
    np.testing.assert_allclose(preconditioner.as_preconditioner()(residual), jnp.zeros_like(residual), atol=0.0)
    x, probe = probe_rhs1_qi_deflated_minres_seed(
        operator=operator,
        rhs=residual,
        x0=jnp.zeros_like(residual),
        preconditioner=preconditioner,
        cycles=3,
    )
    assert probe.reason == "empty_minres_seed"
    assert probe.seed_solver == "cycle_minres"
    assert probe.to_dict()["cycle_residual_history"] == (float(jnp.linalg.norm(residual)),)
    np.testing.assert_allclose(x, jnp.zeros_like(residual), atol=0.0)


def test_residual_deflation_validates_inputs_and_composition_names() -> None:
    residual = jnp.asarray([1.0, -1.0], dtype=jnp.float64)

    def operator(x):
        return jnp.asarray(x, dtype=jnp.float64)

    def smoother(r):
        return jnp.asarray(r, dtype=jnp.float64)

    with pytest.raises(ValueError, match="non-empty"):
        build_rhs1_qi_residual_deflated_preconditioner(
            operator=operator,
            local_smoother=smoother,
            residual_seed=jnp.asarray([], dtype=jnp.float64),
        )

    with pytest.raises(ValueError, match="composition"):
        build_rhs1_qi_residual_deflated_preconditioner(
            operator=operator,
            local_smoother=smoother,
            residual_seed=residual,
            composition="unsupported",
        )

    additive = build_rhs1_qi_residual_deflated_preconditioner(
        operator=operator,
        local_smoother=smoother,
        residual_seed=residual,
        composition="add",
        include_raw_residual=True,
        krylov_depth=0,
        max_rank=2,
    )
    multiplicative = build_rhs1_qi_residual_deflated_preconditioner(
        operator=operator,
        local_smoother=smoother,
        residual_seed=residual,
        composition="field-split",
        include_raw_residual=True,
        krylov_depth=0,
        max_rank=2,
    )

    assert additive.metadata.composition == "additive"
    assert multiplicative.metadata.composition == "multiplicative"
    assert additive.metadata.to_dict()["device_resident"] is True
