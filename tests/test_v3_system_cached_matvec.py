from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import sfincs_jax.operators.profile_response.system as vs


def test_cached_full_system_matvec_uses_local_path_inside_jax_transform(monkeypatch) -> None:
    calls: dict[str, int] = {"jit": 0}

    monkeypatch.setattr(vs, "_value_contains_tracer", lambda value, *, depth=3: True)
    monkeypatch.setattr(vs, "_operator_signature_cached", lambda _op: ("fake",))

    def _jit_factory(_signature):
        def _apply(_op, vector, include_jacobian_terms=True, pad=0):
            calls["jit"] += 1
            assert include_jacobian_terms is True
            assert pad == 0
            return vector + 1.0

        return _apply

    def _pjit_factory(*_args, **_kwargs):
        raise AssertionError("transformed cached matvec must not enter pjit/set_mesh path")

    monkeypatch.setattr(vs, "_get_apply_full_system_operator_jit", _jit_factory)
    monkeypatch.setattr(vs, "_get_apply_full_system_operator_pjit", _pjit_factory)
    monkeypatch.setattr(vs, "_matvec_shard_axis", lambda _op: "theta")
    monkeypatch.setattr(vs, "_get_matvec_mesh", lambda _axis: object())

    op = SimpleNamespace(total_size=4, n_theta=4, n_zeta=4, n_x=1)
    y = vs.apply_v3_full_system_operator_cached(op, jnp.asarray([0.0, 1.0, 2.0, 3.0]))

    assert calls == {"jit": 1}
    np.testing.assert_allclose(np.asarray(y), np.asarray([1.0, 2.0, 3.0, 4.0]))


@pytest.mark.parametrize("shard_axis", ["theta", "flat"])
@pytest.mark.parametrize("transform", ["vmap", "jit"])
def test_cached_full_system_matvec_detects_real_jax_transform_tracers(
    monkeypatch,
    shard_axis: str,
    transform: str,
) -> None:
    calls: dict[str, int] = {"jit": 0}

    monkeypatch.setattr(vs, "_operator_signature_cached", lambda _op: ("fake", shard_axis))

    def _jit_factory(_signature):
        def _apply(_op, vector, include_jacobian_terms=True, pad=0):
            calls["jit"] += 1
            assert include_jacobian_terms is True
            assert pad == 0
            return vector + 2.0

        return _apply

    def _pjit_factory(*_args, **_kwargs):
        raise AssertionError("real transformed cached matvec must not enter pjit/set_mesh path")

    monkeypatch.setattr(vs, "_get_apply_full_system_operator_jit", _jit_factory)
    monkeypatch.setattr(vs, "_get_apply_full_system_operator_pjit", _pjit_factory)
    monkeypatch.setattr(vs, "_get_apply_full_system_operator_pjit_flat", _pjit_factory)
    monkeypatch.setattr(vs, "_matvec_shard_axis", lambda _op: shard_axis)
    monkeypatch.setattr(vs, "_get_matvec_mesh", lambda _axis: object())

    op = SimpleNamespace(total_size=4, n_theta=4, n_zeta=4, n_x=1)
    x = jnp.asarray([0.0, 1.0, 2.0, 3.0])
    if transform == "vmap":
        y = jax.vmap(lambda row: vs.apply_v3_full_system_operator_cached(op, row))(jnp.stack([x, x + 10.0]))
        expected = np.asarray([[2.0, 3.0, 4.0, 5.0], [12.0, 13.0, 14.0, 15.0]])
    else:
        y = jax.jit(lambda row: vs.apply_v3_full_system_operator_cached(op, row))(x)
        expected = np.asarray([2.0, 3.0, 4.0, 5.0])

    assert calls == {"jit": 1}
    np.testing.assert_allclose(np.asarray(y), expected)
