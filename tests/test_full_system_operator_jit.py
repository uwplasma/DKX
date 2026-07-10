from __future__ import annotations

from pathlib import Path

import numpy as np
import jax.numpy as jnp

from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.validation.fortran import read_petsc_vec
from sfincs_jax.discretization.v3 import geometry_from_namelist, grids_from_namelist
from sfincs_jax.problems.profile_solve import solve_v3_full_system_linear_gmres
import sfincs_jax.operators.profile_system as profile_system
from sfincs_jax.operators.profile_system import (
    apply_v3_full_system_operator,
    apply_v3_full_system_operator_jit,
    full_system_operator_from_namelist,
)


class _ShardProbe:
    def __init__(self, *, n_theta: int, n_zeta: int, n_x: int) -> None:
        self.n_theta = n_theta
        self.n_zeta = n_zeta
        self.n_x = n_x


def test_full_system_operator_can_jit_compile() -> None:
    """Regression test: V3FullSystemOperator must be a JAX PyTree usable under jit."""
    here = Path(__file__).parent
    input_path = here / "ref" / "pas_1species_PAS_noEr_tiny_scheme1.input.namelist"
    vec_path = here / "ref" / "pas_1species_PAS_noEr_tiny_scheme1.stateVector.petscbin"

    nml = read_sfincs_input(input_path)
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    x = jnp.asarray(read_petsc_vec(vec_path).values)

    y = np.asarray(apply_v3_full_system_operator(op, x))
    y_jit = np.asarray(apply_v3_full_system_operator_jit(op, x))
    np.testing.assert_allclose(y_jit, y, rtol=0, atol=1e-15)


def test_full_system_operator_reuses_prebuilt_grids_and_geometry() -> None:
    here = Path(__file__).parent
    input_path = here / "ref" / "pas_1species_PAS_noEr_tiny_scheme1.input.namelist"
    vec_path = here / "ref" / "pas_1species_PAS_noEr_tiny_scheme1.stateVector.petscbin"

    nml = read_sfincs_input(input_path)
    grids = grids_from_namelist(nml)
    geom = geometry_from_namelist(nml=nml, grids=grids)
    op_default = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    op_reused = full_system_operator_from_namelist(nml=nml, identity_shift=0.0, grids=grids, geom=geom)
    x = jnp.asarray(read_petsc_vec(vec_path).values)

    y_default = np.asarray(apply_v3_full_system_operator(op_default, x))
    y_reused = np.asarray(apply_v3_full_system_operator(op_reused, x))

    np.testing.assert_allclose(y_reused, y_default, rtol=0, atol=1e-15)


def test_full_system_sharding_policy_is_explicit_and_fail_closed(monkeypatch) -> None:
    op = _ShardProbe(n_theta=32, n_zeta=24, n_x=20)

    monkeypatch.setenv("SFINCS_JAX_SHARD_PAD", "off")
    assert profile_system._shard_pad_enabled() is False
    monkeypatch.setenv("SFINCS_JAX_SHARD_PAD", "yes")
    assert profile_system._shard_pad_enabled() is True
    monkeypatch.delenv("SFINCS_JAX_SHARD_PAD", raising=False)
    assert profile_system._shard_pad_enabled() is True

    monkeypatch.setenv("SFINCS_JAX_MATVEC_SHARD_AXIS", "off")
    assert profile_system._matvec_shard_axis(op) is None
    monkeypatch.setenv("SFINCS_JAX_MATVEC_SHARD_AXIS", "flat")
    assert profile_system._matvec_shard_axis(op) == "flat"
    monkeypatch.setenv("SFINCS_JAX_MATVEC_SHARD_AXIS", "zeta")
    assert profile_system._matvec_shard_axis(op) == "zeta"

    monkeypatch.setenv("SFINCS_JAX_MATVEC_SHARD_AXIS", "not-a-real-axis")
    monkeypatch.setenv("SFINCS_JAX_AUTO_SHARD", "off")
    assert profile_system._matvec_shard_axis(op) is None

    monkeypatch.setenv("SFINCS_JAX_MATVEC_SHARD_AXIS", "auto")
    monkeypatch.setenv("SFINCS_JAX_AUTO_SHARD", "on")
    monkeypatch.setenv("SFINCS_JAX_MATVEC_SHARD_MIN_TZ", "bad")
    assert profile_system._matvec_shard_axis(op) == "theta"

    monkeypatch.setenv("SFINCS_JAX_MATVEC_SHARD_PREFER_X", "true")
    monkeypatch.setenv("SFINCS_JAX_MATVEC_SHARD_MIN_X", "bad")
    assert profile_system._matvec_shard_axis(op) == "x"

    monkeypatch.setenv("SFINCS_JAX_MATVEC_SHARD_MIN_TZ", "10000")
    assert profile_system._matvec_shard_axis(op) is None


def test_full_system_padding_round_trips_all_supported_axes() -> None:
    here = Path(__file__).parent
    input_path = here / "ref" / "pas_1species_PAS_noEr_tiny_scheme1.input.namelist"

    nml = read_sfincs_input(input_path)
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    x = jnp.arange(int(op.total_size), dtype=jnp.float64)

    for axis in ("theta", "zeta", "x"):
        op_padded = profile_system._pad_full_system_operator(op, axis=axis, pad=1)
        x_padded = profile_system._pad_full_vector(x, op=op, op_pad=op_padded, axis=axis, pad=1)
        x_unpadded = profile_system._unpad_full_vector(x_padded, op=op, op_pad=op_padded, axis=axis, pad=1)

        if axis == "theta":
            assert op_padded.n_theta == op.n_theta + 1
        elif axis == "zeta":
            assert op_padded.n_zeta == op.n_zeta + 1
        else:
            assert op_padded.n_x == op.n_x + 1
        assert x_padded.shape == (int(op_padded.total_size),)
        np.testing.assert_allclose(np.asarray(x_unpadded), np.asarray(x), rtol=0.0, atol=0.0)


def test_full_system_linear_gmres_reuses_prebuilt_operator() -> None:
    here = Path(__file__).parent
    input_path = here / "ref" / "pas_1species_PAS_noEr_tiny_scheme1.input.namelist"

    nml = read_sfincs_input(input_path)
    grids = grids_from_namelist(nml)
    geom = geometry_from_namelist(nml=nml, grids=grids)
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0, grids=grids, geom=geom)

    res_default = solve_v3_full_system_linear_gmres(nml=nml, tol=1e-10, emit=None)
    res_reused = solve_v3_full_system_linear_gmres(nml=nml, op=op, tol=1e-10, emit=None)

    np.testing.assert_allclose(np.asarray(res_reused.x), np.asarray(res_default.x), rtol=0, atol=1e-11)
    assert float(res_reused.residual_norm) <= float(res_default.residual_norm) * 10.0 + 1e-12
