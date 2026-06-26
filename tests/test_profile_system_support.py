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
    V3FullSystemOperator,
    apply_v3_full_system_operator,
    apply_v3_full_system_jacobian,
    _fs_average_factor,
    _get_bool,
    _get_int,
    _ix_min,
    _matvec_shard_axis,
    _nonlinear_temp_vector,
    _nonlinear_temp_vector_phi1,
    _operator_signature,
    _operator_signature_cached,
    _pad_1d,
    _pad_2d,
    _pad_full_system_operator,
    _pad_full_vector,
    _pad_square,
    _pad_x_1d,
    _pad_x_square,
    _shard_pad_enabled,
    _source_basis_constraint_scheme_1,
    _unpad_full_vector,
    full_system_operator_from_namelist,
    precompile_v3_full_system,
    residual_v3_full_system,
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


def _write_tiny_profile_input(tmp_path: Path, body: str):
    path = tmp_path / "input.namelist"
    path.write_text(body, encoding="utf-8")
    return read_sfincs_input(path)


@pytest.mark.parametrize(
    ("fixture", "expected_active"),
    [
        (
            "er_xdot_1species_tiny.input.namelist",
            {
                "er_xdot",
                "pas",
                "exb_theta",
                "exb_zeta",
            },
        ),
        (
            "er_xidot_1species_tiny.input.namelist",
            {
                "er_xidot",
                "pas",
                "exb_theta",
                "exb_zeta",
            },
        ),
        (
            "exb_theta_1species_tiny.input.namelist",
            {
                "pas",
                "exb_theta",
                "exb_zeta",
            },
        ),
        (
            "magdrift_1species_tiny.input.namelist",
            {
                "magdrift_theta",
                "magdrift_zeta",
                "magdrift_xidot",
                "pas",
                "exb_theta",
                "exb_zeta",
            },
        ),
        (
            "fp_1species_FPCollisions_noEr_tiny_withPhi1_inCollision.input.namelist",
            {
                "fp_phi1",
                "exb_theta",
                "exb_zeta",
            },
        ),
    ],
)
def test_full_system_operator_optional_physics_terms_apply_on_tiny_v3_fixtures(
    fixture: str, expected_active: set[str]
) -> None:
    """Exercise optional v3 physics branches without launching a linear solve."""
    nml = read_sfincs_input(REF / fixture)
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0, keep_zero_er_terms=True)
    fblock = op.fblock
    active_terms = {
        "er_xdot": fblock.er_xdot is not None,
        "er_xidot": fblock.er_xidot is not None,
        "magdrift_theta": fblock.magdrift_theta is not None,
        "magdrift_zeta": fblock.magdrift_zeta is not None,
        "magdrift_xidot": fblock.magdrift_xidot is not None,
        "pas": fblock.pas is not None,
        "fp": fblock.fp is not None,
        "fp_phi1": fblock.fp_phi1 is not None,
        "exb_theta": fblock.exb_theta is not None,
        "exb_zeta": fblock.exb_zeta is not None,
    }

    assert {name for name, is_active in active_terms.items() if is_active} == expected_active
    assert op.total_size > op.f_size
    if "fp_phi1" in expected_active:
        assert op.include_phi1 is True
        assert op.phi1_size == op.n_theta * op.n_zeta + 1
    else:
        assert op.include_phi1 is False

    vector = _deterministic_vector(op.total_size)
    y_jac = apply_v3_full_system_operator(op, vector, include_jacobian_terms=True)
    y_residual_operator = apply_v3_full_system_operator(op, vector, include_jacobian_terms=False)
    residual = residual_v3_full_system(op, vector)

    assert y_jac.shape == (op.total_size,)
    assert y_residual_operator.shape == (op.total_size,)
    assert residual.shape == (op.total_size,)
    assert np.all(np.isfinite(np.asarray(y_jac)))
    assert np.all(np.isfinite(np.asarray(y_residual_operator)))
    assert np.all(np.isfinite(np.asarray(residual)))
    assert float(jnp.linalg.norm(y_jac)) > 0.0


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


def test_profile_system_scalar_helpers_and_ix_min_match_fortran_conventions() -> None:
    group = {
        "A": [7],
        "EMPTY": [],
        "FLAG_TRUE": True,
        "FLAG_FALSE": 0,
    }

    assert _get_int(group, "a", 0) == 7
    assert _get_int(group, "empty", 3) == 3
    assert _get_int(group, "missing", 11) == 11
    assert _get_bool(group, "flag_true") is True
    assert _get_bool(group, "flag_false") is False
    assert _get_bool(group, "missing", default=True) is True
    assert _ix_min(point_at_x0=True) == 1
    assert _ix_min(point_at_x0=False) == 0


def test_full_system_builder_rejects_external_phi1_before_operator_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nml = _write_tiny_profile_input(
        tmp_path,
        "&physicsParameters\n"
        "  includePhi1 = .true.\n"
        "  readExternalPhi1 = .true.\n"
        "/\n",
    )

    def fail_if_called(**_kwargs):
        raise AssertionError("f-block setup should not run before external-Phi1 admission")

    monkeypatch.setattr(profile_system, "fblock_operator_from_namelist", fail_if_called)
    with pytest.raises(NotImplementedError, match="readExternalPhi1"):
        full_system_operator_from_namelist(nml=nml, grids=SimpleNamespace(), geom=SimpleNamespace())


def test_full_system_builder_rejects_invalid_radial_gradient_coordinate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nml = _write_tiny_profile_input(
        tmp_path,
        "&geometryParameters\n"
        "  geometryScheme = 1\n"
        "  inputRadialCoordinateForGradients = 99\n"
        "/\n"
        "&physicsParameters\n"
        "  collisionOperator = 0\n"
        "/\n"
        "&speciesParameters\n"
        "  Zs = 1\n"
        "  mHats = 1\n"
        "  THats = 1\n"
        "  nHats = 1\n"
        "/\n",
    )

    monkeypatch.setattr(profile_system, "fblock_operator_from_namelist", lambda **_kwargs: SimpleNamespace())
    with pytest.raises(NotImplementedError, match="inputRadialCoordinateForGradients"):
        full_system_operator_from_namelist(nml=nml, grids=SimpleNamespace(), geom=SimpleNamespace())


def test_full_system_builder_rejects_unsupported_geometry_scheme_after_fast_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nml = _write_tiny_profile_input(
        tmp_path,
        "&geometryParameters\n"
        "  geometryScheme = 13\n"
        "/\n"
        "&physicsParameters\n"
        "  collisionOperator = 0\n"
        "/\n"
        "&speciesParameters\n"
        "  Zs = 1\n"
        "  mHats = 1\n"
        "  THats = 1\n"
        "  nHats = 1\n"
        "/\n",
    )

    monkeypatch.setattr(profile_system, "fblock_operator_from_namelist", lambda **_kwargs: SimpleNamespace())
    with pytest.raises(NotImplementedError, match="geometryScheme=13"):
        full_system_operator_from_namelist(nml=nml, grids=SimpleNamespace(), geom=SimpleNamespace())


def test_full_system_builder_checks_eparallelhat_spec_species_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nml = _write_tiny_profile_input(
        tmp_path,
        "&general\n"
        "  RHSMode = 1\n"
        "/\n"
        "&geometryParameters\n"
        "  geometryScheme = 1\n"
        "/\n"
        "&physicsParameters\n"
        "  collisionOperator = 0\n"
        "  EParallelHatSpec = 1 2 3\n"
        "/\n"
        "&speciesParameters\n"
        "  Zs = 1 -1\n"
        "  mHats = 1 0.0005446\n"
        "  THats = 1 1\n"
        "  nHats = 1 1\n"
        "/\n",
    )

    monkeypatch.setattr(profile_system, "fblock_operator_from_namelist", lambda **_kwargs: SimpleNamespace())
    with pytest.raises(ValueError, match="EParallelHatSpec"):
        full_system_operator_from_namelist(nml=nml, grids=SimpleNamespace(), geom=SimpleNamespace())


def test_nonlinear_temp_vector_helpers_match_manual_l_coupling() -> None:
    op = SimpleNamespace(
        n_x=2,
        n_xi=2,
        x=jnp.asarray([1.0, 2.0], dtype=jnp.float64),
        ddx=jnp.eye(2, dtype=jnp.float64),
        point_at_x0=True,
        fblock=SimpleNamespace(collisionless=SimpleNamespace(n_xi_for_x=jnp.asarray([1, 2], dtype=jnp.int32))),
    )
    f = jnp.asarray([[[[[2.0]], [[3.0]]], [[[5.0]], [[7.0]]]]], dtype=jnp.float64)

    nonlinear, mask_l, mask_x = _nonlinear_temp_vector(op, f)
    phi1, phi1_mask_l, phi1_mask_x = _nonlinear_temp_vector_phi1(op, f)
    expected = np.asarray([[[[[3.0]], [[2.0]]], [[[14.0 / 3.0]], [[5.0]]]]])

    np.testing.assert_allclose(np.asarray(nonlinear), expected)
    np.testing.assert_allclose(np.asarray(phi1), expected)
    np.testing.assert_allclose(np.asarray(mask_l), np.asarray([[1.0, 0.0], [1.0, 1.0]]))
    np.testing.assert_allclose(np.asarray(phi1_mask_l), np.asarray(mask_l))
    np.testing.assert_allclose(np.asarray(mask_x), np.asarray([0.0, 1.0]))
    np.testing.assert_allclose(np.asarray(phi1_mask_x), np.asarray(mask_x))


def test_profile_system_padding_primitives_preserve_values_and_fill_policy() -> None:
    vector = jnp.asarray([1.0, 2.0], dtype=jnp.float64)
    square = jnp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float64)
    surface = jnp.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=jnp.float64)

    np.testing.assert_allclose(np.asarray(_pad_1d(vector, 2, fill=-1.0)), np.asarray([1.0, 2.0, -1.0, -1.0]))
    np.testing.assert_allclose(
        np.asarray(_pad_square(square, 1)),
        np.asarray([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 0.0]]),
    )
    np.testing.assert_allclose(
        np.asarray(_pad_2d(surface, axis="theta", pad=1, fill=9.0)),
        np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [9.0, 9.0, 9.0]]),
    )
    np.testing.assert_allclose(
        np.asarray(_pad_2d(surface, axis="zeta", pad=2, fill=-2.0)),
        np.asarray([[1.0, 2.0, 3.0, -2.0, -2.0], [4.0, 5.0, 6.0, -2.0, -2.0]]),
    )
    assert _pad_2d(surface, axis="x", pad=5) is surface
    assert _pad_x_1d(vector, 0) is vector
    assert _pad_x_square(square, 0) is square


def test_full_system_operator_properties_and_pytree_roundtrip() -> None:
    op = _tiny_phi1_scheme2_operator()
    children, aux = op.tree_flatten()
    rebuilt = V3FullSystemOperator.tree_unflatten(aux, children)

    assert _operator_signature(rebuilt) == _operator_signature(op)
    assert op.n_species == int(op.fblock.n_species)
    assert op.n_x == int(op.fblock.n_x)
    assert op.n_xi == int(op.fblock.n_xi)
    assert op.n_theta == int(op.fblock.n_theta)
    assert op.n_zeta == int(op.fblock.n_zeta)
    assert op.f_size == int(op.fblock.flat_size)
    assert op.phi1_size == op.n_theta * op.n_zeta + 1
    assert op.total_size == op.f_size + op.phi1_size + op.extra_size

    no_constraints = replace(op, include_phi1=False, constraint_scheme=0)
    assert no_constraints.phi1_size == 0
    assert no_constraints.extra_size == 0
    assert no_constraints.total_size == no_constraints.f_size

    for scheme in (1, 3, 4):
        assert replace(op, constraint_scheme=scheme).extra_size == 2 * op.n_species

    with pytest.raises(NotImplementedError, match="constraintScheme=99"):
        _ = replace(op, constraint_scheme=99).extra_size


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


def test_residual_uses_current_phi1_state_for_nonlinear_operator(monkeypatch: pytest.MonkeyPatch) -> None:
    op = _tiny_phi1_scheme2_operator()
    state = jnp.zeros((op.total_size,), dtype=jnp.float64)
    phi1_state = jnp.arange(op.n_theta * op.n_zeta, dtype=jnp.float64).reshape((op.n_theta, op.n_zeta))
    state = state.at[op.f_size : op.f_size + op.n_theta * op.n_zeta].set(phi1_state.reshape((-1,)))
    captured: dict[str, np.ndarray] = {}

    def fake_matvec(op_use: V3FullSystemOperator, x_full: jnp.ndarray, *, include_jacobian_terms: bool = True) -> jnp.ndarray:
        assert include_jacobian_terms is False
        captured["phi1_hat_base"] = np.asarray(op_use.phi1_hat_base)
        return jnp.asarray(x_full, dtype=jnp.float64)

    monkeypatch.setattr(profile_system, "apply_v3_full_system_operator_cached", fake_matvec)
    monkeypatch.setattr(profile_system, "rhs_v3_full_system_jit", lambda _op: jnp.ones((op.total_size,), dtype=jnp.float64))

    residual = residual_v3_full_system(op, state)

    np.testing.assert_allclose(captured["phi1_hat_base"], np.asarray(phi1_state))
    np.testing.assert_allclose(np.asarray(residual), np.asarray(state) - 1.0)


def test_precompile_v3_full_system_compiles_expected_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    op = _tiny_phi1_scheme2_operator()
    calls: list[tuple[str, bool | None]] = []

    class FakeLowered:
        def __init__(self, name: str, include_jacobian_terms: bool | None) -> None:
            self.name = name
            self.include_jacobian_terms = include_jacobian_terms

        def compile(self) -> None:
            calls.append((self.name, self.include_jacobian_terms))

    class FakeKernel:
        def __init__(self, name: str) -> None:
            self.name = name

        def lower(self, *_args, **kwargs) -> FakeLowered:
            return FakeLowered(self.name, kwargs.get("include_jacobian_terms"))

    monkeypatch.setattr(profile_system, "_get_apply_full_system_operator_jit", lambda _signature: FakeKernel("matvec"))
    monkeypatch.setattr(profile_system, "apply_v3_full_system_jacobian_jit", FakeKernel("jacobian"))
    monkeypatch.setattr(profile_system, "rhs_v3_full_system_jit", FakeKernel("rhs"))

    precompile_v3_full_system(op, include_jacobian=False)
    assert calls == [("matvec", True), ("matvec", False), ("rhs", None)]

    calls.clear()
    precompile_v3_full_system(op, include_jacobian=True)
    assert calls == [("matvec", True), ("matvec", False), ("jacobian", None), ("rhs", None)]


def test_full_system_jacobian_rejects_mismatched_state_shapes() -> None:
    op = _tiny_phi1_scheme2_operator()
    state = jnp.zeros((op.total_size,), dtype=jnp.float64)

    with pytest.raises(ValueError, match="total_size"):
        apply_v3_full_system_jacobian(op, state[:-1], state)

    with pytest.raises(ValueError, match="total_size"):
        apply_v3_full_system_jacobian(op, state, state[:-1])
