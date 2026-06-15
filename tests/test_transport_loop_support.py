from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from sfincs_jax.transport_loop_support import (
    TransportMatvecCache,
    TransportRecycleState,
    recycled_transport_initial_guess,
    resolve_transport_recycle_k,
)


def _op(signature: tuple[object, ...], *, scale: float = 2.0):
    return SimpleNamespace(signature=signature, scale=float(scale))


def _signature(op) -> tuple[object, ...]:
    return op.signature


def _apply(op, x):
    return op.scale * x


def test_transport_matvec_cache_reuses_full_and_reduced_closures() -> None:
    def reduce_full(x):
        return x[jnp.asarray([0, 2])]

    def expand_reduced(x):
        out = jnp.zeros((4,), dtype=x.dtype)
        return out.at[jnp.asarray([0, 2])].set(x)

    cache = TransportMatvecCache(
        use_active_dof_mode=True,
        active_size=2,
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
        apply_operator=_apply,
        operator_signature=_signature,
    )
    op = _op(("same",), scale=3.0)
    full = cache.get_full(op)
    reduced = cache.get_reduced(op)

    assert cache.get_full(op) is full
    assert cache.get_reduced(op) is reduced
    np.testing.assert_allclose(np.asarray(full(jnp.arange(4.0))), np.asarray([0.0, 3.0, 6.0, 9.0]))
    np.testing.assert_allclose(np.asarray(reduced(jnp.asarray([2.0, 5.0]))), np.asarray([6.0, 15.0]))


def test_recycled_transport_initial_guess_matches_small_subspace() -> None:
    basis = [jnp.asarray([1.0, 0.0]), jnp.asarray([0.0, 1.0])]
    basis_au = [jnp.asarray([2.0, 0.0]), jnp.asarray([0.0, 3.0])]
    x0 = recycled_transport_initial_guess(jnp.asarray([4.0, 9.0]), basis, basis_au)

    assert x0 is not None
    np.testing.assert_allclose(np.asarray(x0), np.asarray([2.0, 3.0]), rtol=1e-10, atol=1e-10)
    assert recycled_transport_initial_guess(jnp.ones((2,)), [], []) is None


def test_transport_recycle_state_trims_and_seeds_full_and_reduced() -> None:
    def reduce_full(x):
        return x[jnp.asarray([0, 2])]

    def expand_reduced(x):
        out = jnp.zeros((4,), dtype=x.dtype)
        return out.at[jnp.asarray([0, 2])].set(x)

    cache = TransportMatvecCache(
        use_active_dof_mode=True,
        active_size=2,
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
        apply_operator=_apply,
        operator_signature=_signature,
    )
    recycle = TransportRecycleState(k=1)
    recycle.seed_from_state(
        state_x_by_rhs={
            1: jnp.asarray([1.0, 2.0, 3.0, 4.0]),
            2: jnp.asarray([5.0, 7.0]),
        },
        total_size=4,
        active_size=2,
        matvec_cache=cache,
        op_ref=_op(("ref",), scale=2.0),
    )

    assert len(recycle.full_basis) == 1
    assert len(recycle.reduced_basis) == 1
    np.testing.assert_allclose(np.asarray(recycle.reduced_basis[0]), np.asarray([5.0, 7.0]))
    np.testing.assert_allclose(np.asarray(recycle.reduced_basis_au[0]), np.asarray([10.0, 14.0]))

    recycle.append_full(jnp.asarray([1.0, 0.0, 0.0, 0.0]), jnp.asarray([2.0, 0.0, 0.0, 0.0]))
    assert len(recycle.full_basis) == 1
    np.testing.assert_allclose(np.asarray(recycle.full_basis[0]), np.asarray([1.0, 0.0, 0.0, 0.0]))


def test_resolve_transport_recycle_k_respects_env_and_operator_variation(monkeypatch) -> None:
    messages: list[tuple[int, str]] = []

    def emit(level: int, message: str) -> None:
        messages.append((int(level), str(message)))

    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_RECYCLE_K", raising=False)
    assert (
        resolve_transport_recycle_k(
            op=object(),
            use_implicit=False,
            op_matvec_by_index=[_op(("a",)), _op(("a",))],
            disable_auto_recycle=lambda **_: False,
            emit=emit,
            operator_signature=_signature,
        )
        == 4
    )

    assert (
        resolve_transport_recycle_k(
            op=object(),
            use_implicit=False,
            op_matvec_by_index=[_op(("a",)), _op(("b",))],
            disable_auto_recycle=lambda **_: False,
            emit=emit,
            operator_signature=_signature,
        )
        == 0
    )
    assert any("matvec operator varies" in message for _, message in messages)

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_RECYCLE_K", "7")
    assert (
        resolve_transport_recycle_k(
            op=object(),
            use_implicit=False,
            op_matvec_by_index=[_op(("a",))],
            disable_auto_recycle=lambda **_: True,
            emit=emit,
            operator_signature=_signature,
        )
        == 0
    )
    assert any("auto recycle disabled" in message for _, message in messages)
