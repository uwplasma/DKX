from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import sfincs_jax.operators.profile_system as profile_system
from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.operators.profile_system import (
    _fs_average_factor,
    _matvec_shard_axis,
    _operator_signature,
    _operator_signature_cached,
    _pad_full_system_operator,
    _pad_full_vector,
    _shard_pad_enabled,
    _source_basis_constraint_scheme_1,
    _unpad_full_vector,
    full_system_operator_from_namelist,
    sharding_constraints,
    with_transport_rhs_settings,
)


REF = Path(__file__).parent / "ref"


def _deterministic_vector(size: int) -> jnp.ndarray:
    idx = jnp.arange(int(size), dtype=jnp.float64)
    return jnp.sin(0.2 * idx) + 0.1 * jnp.cos(0.7 * idx)


def _tiny_phi1_scheme2_operator():
    nml = read_sfincs_input(REF / "include_phi1_linear_subset_tiny.input.namelist")
    return full_system_operator_from_namelist(nml=nml, identity_shift=0.0)


def test_shard_pad_policy_and_context_restore_global_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_SHARD_PAD", raising=False)
    assert _shard_pad_enabled() is True

    monkeypatch.setenv("SFINCS_JAX_SHARD_PAD", "off")
    assert _shard_pad_enabled() is False

    monkeypatch.setenv("SFINCS_JAX_SHARD_PAD", ".true.")
    assert _shard_pad_enabled() is True

    before = profile_system._SHARDING_CONSTRAINTS_ENABLED
    with sharding_constraints(not before):
        assert profile_system._SHARDING_CONSTRAINTS_ENABLED is (not before)
    assert profile_system._SHARDING_CONSTRAINTS_ENABLED is before


def test_matvec_shard_axis_env_and_auto_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    op = SimpleNamespace(n_theta=25, n_zeta=17, n_x=32)

    monkeypatch.setenv("SFINCS_JAX_MATVEC_SHARD_AXIS", "off")
    assert _matvec_shard_axis(op) is None

    monkeypatch.setenv("SFINCS_JAX_MATVEC_SHARD_AXIS", "vector")
    assert _matvec_shard_axis(op) == "flat"

    monkeypatch.setenv("SFINCS_JAX_MATVEC_SHARD_AXIS", "zeta")
    assert _matvec_shard_axis(op) == "zeta"

    monkeypatch.setenv("SFINCS_JAX_MATVEC_SHARD_AXIS", "unexpected")
    monkeypatch.setenv("SFINCS_JAX_AUTO_SHARD", "off")
    assert _matvec_shard_axis(op) is None

    monkeypatch.setenv("SFINCS_JAX_MATVEC_SHARD_AXIS", "auto")
    monkeypatch.delenv("SFINCS_JAX_AUTO_SHARD", raising=False)
    monkeypatch.setenv("SFINCS_JAX_MATVEC_SHARD_MIN_TZ", "bad")
    assert _matvec_shard_axis(op) == "theta"

    monkeypatch.setenv("SFINCS_JAX_MATVEC_SHARD_PREFER_X", "yes")
    monkeypatch.setenv("SFINCS_JAX_MATVEC_SHARD_MIN_X", "bad")
    assert _matvec_shard_axis(op) == "x"

    small = SimpleNamespace(n_theta=2, n_zeta=2, n_x=2)
    monkeypatch.delenv("SFINCS_JAX_MATVEC_SHARD_MIN_TZ", raising=False)
    monkeypatch.delenv("SFINCS_JAX_MATVEC_SHARD_PREFER_X", raising=False)
    assert _matvec_shard_axis(small) is None


def test_profile_system_moment_helpers_match_analytic_formulas() -> None:
    theta_weights = jnp.asarray([0.25, 0.75])
    zeta_weights = jnp.asarray([0.4, 0.6])
    d_hat = jnp.asarray([[2.0, 4.0], [5.0, 10.0]])
    expected = np.asarray([[0.05, 0.0375], [0.06, 0.045]])
    np.testing.assert_allclose(np.asarray(_fs_average_factor(theta_weights, zeta_weights, d_hat)), expected)

    x = jnp.asarray([0.0, 1.0])
    s1, s2 = _source_basis_constraint_scheme_1(x)
    coef = np.exp(-np.asarray(x) ** 2) / (np.pi * np.sqrt(np.pi))
    np.testing.assert_allclose(np.asarray(s1), (-np.asarray(x) ** 2 + 2.5) * coef)
    np.testing.assert_allclose(np.asarray(s2), ((2.0 / 3.0) * np.asarray(x) ** 2 - 1.0) * coef)


def test_operator_signature_cache_uses_object_identity_and_static_layout() -> None:
    op = _tiny_phi1_scheme2_operator()

    sig = _operator_signature(op)
    cached_first = _operator_signature_cached(op)
    cached_second = _operator_signature_cached(op)
    assert cached_first == sig
    assert cached_second == cached_first

    changed = replace(op, rhs_mode=2)
    assert _operator_signature(changed)[0] == 2
    assert _operator_signature(changed) != sig


@pytest.mark.parametrize("axis", ["theta", "zeta", "x"])
def test_pad_and_unpad_full_vector_roundtrip_for_phi1_scheme2(axis: str) -> None:
    op = _tiny_phi1_scheme2_operator()
    pad = 2
    op_pad = _pad_full_system_operator(op, axis=axis, pad=pad)
    x_full = _deterministic_vector(op.total_size)

    x_pad = _pad_full_vector(x_full, op=op, op_pad=op_pad, axis=axis, pad=pad)
    roundtrip = _unpad_full_vector(x_pad, op=op, op_pad=op_pad, axis=axis, pad=pad)

    assert int(op_pad.total_size) > int(op.total_size)
    assert x_pad.shape == (op_pad.total_size,)
    np.testing.assert_allclose(np.asarray(roundtrip), np.asarray(x_full))

    same_op = _pad_full_system_operator(op, axis=axis, pad=0)
    same_x = _pad_full_vector(x_full, op=op, op_pad=op, axis=axis, pad=0)
    assert same_op is op
    assert same_x is x_full


def test_pad_full_vector_handles_constraint_scheme2_x_sources_without_phi1() -> None:
    base = _tiny_phi1_scheme2_operator()
    op = replace(base, include_phi1=False)
    pad = 3
    op_pad = _pad_full_system_operator(op, axis="x", pad=pad)
    x_full = _deterministic_vector(op.total_size)

    x_pad = _pad_full_vector(x_full, op=op, op_pad=op_pad, axis="x", pad=pad)
    roundtrip = _unpad_full_vector(x_pad, op=op, op_pad=op_pad, axis="x", pad=pad)

    assert x_pad.shape == (op_pad.total_size,)
    np.testing.assert_allclose(np.asarray(roundtrip), np.asarray(x_full))


def test_transport_rhs_settings_cover_mode2_mode3_and_invalid_rhs() -> None:
    op0 = _tiny_phi1_scheme2_operator()

    mono = replace(op0, rhs_mode=3)
    mono_density = with_transport_rhs_settings(mono, which_rhs=1)
    np.testing.assert_allclose(np.asarray(mono_density.dn_hat_dpsi_hat), np.ones(op0.n_species))
    np.testing.assert_allclose(np.asarray(mono_density.dt_hat_dpsi_hat), np.zeros(op0.n_species))
    assert float(mono_density.e_parallel_hat) == pytest.approx(0.0)

    mono_epar = with_transport_rhs_settings(mono, which_rhs=2)
    np.testing.assert_allclose(np.asarray(mono_epar.dn_hat_dpsi_hat), np.zeros(op0.n_species))
    assert float(mono_epar.e_parallel_hat) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="RHSMode=3"):
        with_transport_rhs_settings(mono, which_rhs=3)

    energy = replace(op0, rhs_mode=2)
    temp = with_transport_rhs_settings(energy, which_rhs=2)
    expected_dn = 1.5 * float(op0.n_hat[0]) * float(op0.t_hat[0])
    np.testing.assert_allclose(np.asarray(temp.dn_hat_dpsi_hat), np.full(op0.n_species, expected_dn))
    np.testing.assert_allclose(np.asarray(temp.dt_hat_dpsi_hat), np.ones(op0.n_species))
    with pytest.raises(ValueError, match="RHSMode=2"):
        with_transport_rhs_settings(energy, which_rhs=4)

    unchanged = replace(op0, rhs_mode=1)
    assert with_transport_rhs_settings(unchanged, which_rhs=1) is unchanged


def test_value_contains_tracer_makes_transformed_operator_uncacheable() -> None:
    op = _tiny_phi1_scheme2_operator()

    @jax.jit
    def _inside_transform(alpha):
        transformed = replace(op, alpha=alpha)
        return jnp.asarray(profile_system._op_cacheable(transformed))

    assert bool(profile_system._op_cacheable(op)) is True
    assert bool(_inside_transform(jnp.asarray(1.0))) is False
