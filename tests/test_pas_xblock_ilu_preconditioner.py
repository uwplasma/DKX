from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.operators.profile_system import full_system_operator_from_namelist
from sfincs_jax.solvers.preconditioner_pas_xblock_ilu import (
    build_rhs1_pas_xblock_ilu_preconditioner,
)
from sfincs_jax.solvers.preconditioning import _RHSMODE1_PRECOND_ILU_CACHE


REF = Path(__file__).parent / "ref"


def _tiny_pas_operator():
    nml = read_sfincs_input(REF / "pas_1species_PAS_noEr_tiny.input.namelist")
    return full_system_operator_from_namelist(nml=nml, identity_shift=0.25)


def test_pas_xblock_ilu_falls_back_for_inapplicable_collision_models() -> None:
    calls = {"fallback": 0}

    def fallback(**_kwargs):
        calls["fallback"] += 1
        return lambda x: 2.0 * x

    for fblock in (
        SimpleNamespace(pas=None, fp=None),
        SimpleNamespace(pas=object(), fp=object()),
    ):
        op = SimpleNamespace(fblock=fblock)
        preconditioner = build_rhs1_pas_xblock_ilu_preconditioner(
            op=op,
            pas_hybrid_preconditioner=fallback,
        )

        value = jnp.asarray([1.0, -3.0], dtype=jnp.float64)
        np.testing.assert_allclose(np.asarray(preconditioner(value)), np.asarray(2.0 * value))

    assert calls == {"fallback": 2}


def test_pas_xblock_ilu_builds_finite_padded_factors_on_tiny_pas_fixture(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_ILU_THREADS", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_ILU_DROP_TOL", "not-a-float")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_ILU_FILL_FACTOR", "not-a-float")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_ILU_ROW_NNZ_MAX", "not-an-int")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_ILU_REG", "not-a-float")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_LU_MAX", "not-an-int")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_LU_ROW_NNZ_MAX", "not-an-int")
    _RHSMODE1_PRECOND_ILU_CACHE.clear()
    op = _tiny_pas_operator()
    calls = {"fallback": 0}

    def fallback(**_kwargs):
        calls["fallback"] += 1
        return lambda x: x

    preconditioner = build_rhs1_pas_xblock_ilu_preconditioner(
        op=op,
        pas_hybrid_preconditioner=fallback,
    )

    vector = jnp.sin(0.13 * jnp.arange(op.total_size, dtype=jnp.float64))
    result = preconditioner(vector)

    assert calls == {"fallback": 0}
    assert result.shape == vector.shape
    assert bool(jnp.all(jnp.isfinite(result)))
    assert getattr(preconditioner, "_sfincs_pas_ilu_block_size_max") == op.n_xi * op.n_theta * op.n_zeta
    factors = getattr(preconditioner, "_sfincs_pas_ilu_factors")
    assert len(factors) == 7
    for factor in factors:
        assert factor.shape[:3] == (op.n_species, op.n_x, op.n_xi * op.n_theta * op.n_zeta)
        assert bool(jnp.all(jnp.isfinite(factor)))
    assert len(_RHSMODE1_PRECOND_ILU_CACHE) == 1

    second = build_rhs1_pas_xblock_ilu_preconditioner(
        op=op,
        pas_hybrid_preconditioner=fallback,
    )
    assert len(_RHSMODE1_PRECOND_ILU_CACHE) == 1
    np.testing.assert_allclose(np.asarray(second(vector)), np.asarray(result))


def test_pas_xblock_ilu_reduced_application_matches_full_projection(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_ILU_THREADS", "1")
    _RHSMODE1_PRECOND_ILU_CACHE.clear()
    op = _tiny_pas_operator()
    active = jnp.arange(op.total_size, dtype=jnp.int32)[::3]

    def reduce_full(vector: jnp.ndarray) -> jnp.ndarray:
        return vector[active]

    def expand_reduced(vector: jnp.ndarray) -> jnp.ndarray:
        out = jnp.zeros((op.total_size,), dtype=jnp.float64)
        return out.at[active].set(vector)

    full_preconditioner = build_rhs1_pas_xblock_ilu_preconditioner(
        op=op,
        pas_hybrid_preconditioner=lambda **_kwargs: (lambda x: x),
    )
    reduced_preconditioner = build_rhs1_pas_xblock_ilu_preconditioner(
        op=op,
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
        pas_hybrid_preconditioner=lambda **_kwargs: (lambda x: x),
    )

    reduced_rhs = jnp.cos(0.19 * jnp.arange(active.size, dtype=jnp.float64))
    expected = reduce_full(full_preconditioner(expand_reduced(reduced_rhs)))

    np.testing.assert_allclose(np.asarray(reduced_preconditioner(reduced_rhs)), np.asarray(expected))
