from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp

from sfincs_jax.problems.profile_response.setup import (
    expand_reduced_with_map,
    finalize_rhs1_linear_solution_cleanup,
    reduce_full_with_indices,
)
from sfincs_jax.solver import GMRESSolveResult


def test_reduce_expand_active_projection_round_trip() -> None:
    v_full = jnp.asarray([10.0, 20.0, 30.0, 40.0])
    active_idx = jnp.asarray([0, 2], dtype=jnp.int32)
    full_to_active = jnp.asarray([1, 0, 2, 0], dtype=jnp.int32)

    reduced = reduce_full_with_indices(v_full, active_idx)

    assert jnp.allclose(reduced, jnp.asarray([10.0, 30.0]))
    assert jnp.allclose(
        expand_reduced_with_map(reduced, full_to_active),
        jnp.asarray([10.0, 0.0, 30.0, 0.0]),
    )


def test_finalize_rhs1_linear_solution_cleanup_skips_non_rhs1() -> None:
    result = GMRESSolveResult(x=jnp.ones(3), residual_norm=jnp.asarray(2.0))
    op = SimpleNamespace(rhs_mode=2, constraint_scheme=1, include_phi1=False, extra_size=1)

    cleaned = finalize_rhs1_linear_solution_cleanup(
        op=op,
        result=result,
        rhs=jnp.zeros(3),
        residual_vec=None,
        source_zero_tolerance=1.0,
    )

    assert cleaned is result


def test_finalize_rhs1_linear_solution_cleanup_respects_projection_env(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PROJECT_NULLSPACE", "off")
    result = GMRESSolveResult(x=jnp.ones(3), residual_norm=jnp.asarray(2.0))
    op = SimpleNamespace(rhs_mode=1, constraint_scheme=1, include_phi1=False, extra_size=1)

    def fail_project(**_kwargs):
        raise AssertionError("disabled projection should not call projector")

    cleaned = finalize_rhs1_linear_solution_cleanup(
        op=op,
        result=result,
        rhs=jnp.zeros(3),
        residual_vec=None,
        project_solution_with_residual=fail_project,
        source_zero_tolerance=0.0,
    )

    assert cleaned is result


def test_finalize_rhs1_linear_solution_cleanup_projects_default_rhs1(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_PROJECT_NULLSPACE", raising=False)
    result = GMRESSolveResult(
        x=jnp.asarray([1.0, 2.0, 3.0]),
        residual_norm=jnp.asarray(9.0),
    )
    op = SimpleNamespace(rhs_mode=1, constraint_scheme=1, include_phi1=False, extra_size=1)

    def fake_project(**kwargs):
        assert kwargs["enabled_env_var"] == "SFINCS_JAX_RHSMODE1_PROJECT_NULLSPACE"
        assert kwargs["residual_vec"] is not None
        return jnp.asarray([1.0, 2.0, 4.0]), jnp.asarray([3.0, 4.0, 0.0])

    cleaned = finalize_rhs1_linear_solution_cleanup(
        op=op,
        result=result,
        rhs=jnp.zeros(3),
        residual_vec=jnp.asarray([9.0, 9.0, 9.0]),
        project_solution_with_residual=fake_project,
        source_zero_tolerance=0.0,
    )

    assert jnp.allclose(cleaned.x, jnp.asarray([1.0, 2.0, 4.0]))
    assert jnp.allclose(cleaned.residual_norm, 5.0)


def test_finalize_rhs1_linear_solution_cleanup_zeroes_tiny_pas_sources() -> None:
    result = GMRESSolveResult(
        x=jnp.asarray([1.0, 2.0, 1.0e-10, -1.0e-10]),
        residual_norm=jnp.asarray(7.0),
    )
    op = SimpleNamespace(rhs_mode=1, constraint_scheme=2, include_phi1=False, extra_size=2)

    cleaned = finalize_rhs1_linear_solution_cleanup(
        op=op,
        result=result,
        rhs=jnp.zeros(4),
        residual_vec=None,
        source_zero_tolerance=2.0e-9,
    )

    assert jnp.allclose(cleaned.x, jnp.asarray([1.0, 2.0, 0.0, 0.0]))
    assert cleaned.residual_norm is result.residual_norm


def test_finalize_rhs1_linear_solution_cleanup_keeps_large_pas_sources() -> None:
    result = GMRESSolveResult(
        x=jnp.asarray([1.0, 2.0, 1.0e-8, -1.0e-10]),
        residual_norm=jnp.asarray(7.0),
    )
    op = SimpleNamespace(rhs_mode=1, constraint_scheme=2, include_phi1=False, extra_size=2)

    cleaned = finalize_rhs1_linear_solution_cleanup(
        op=op,
        result=result,
        rhs=jnp.zeros(4),
        residual_vec=None,
        source_zero_tolerance=2.0e-9,
    )

    assert jnp.allclose(cleaned.x, result.x)
    assert cleaned.residual_norm is result.residual_norm
