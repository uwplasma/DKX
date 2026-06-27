from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

import sfincs_jax.solvers.preconditioner_schur_profile as schur_profile
from sfincs_jax.solvers.preconditioner_schur_profile import (
    RHS1SchurPreconditionerBuilders,
    build_rhs1_schur_preconditioner,
    canonical_schur_base_kind,
    resolve_active_native_field_split_sparse_coarse_policy,
    resolve_active_native_stack_policy,
    resolve_active_sparse_coarse_residual_policy,
    resolve_rhs1_schur_base_kind,
)


def _schur_dispatch_op() -> SimpleNamespace:
    f_shape = (1, 2, 2, 2, 2)
    f_size = int(np.prod(f_shape))
    fblock = SimpleNamespace(
        f_shape=f_shape,
        collisionless=SimpleNamespace(n_xi_for_x=np.asarray([2, 2], dtype=np.int32)),
        exb_theta=None,
        exb_zeta=None,
        pas=None,
        fp=None,
        er_xdot=None,
        er_xidot=None,
    )
    return SimpleNamespace(
        rhs_mode=1,
        constraint_scheme=2,
        phi1_size=0,
        extra_size=0,
        f_size=f_size,
        total_size=f_size,
        n_species=1,
        n_x=2,
        n_theta=2,
        n_zeta=2,
        point_at_x0=False,
        theta_weights=np.ones(2, dtype=np.float64),
        zeta_weights=np.ones(2, dtype=np.float64),
        d_hat=np.ones((2, 2), dtype=np.float64),
        fblock=fblock,
    )


def _dispatch_builders(
    calls: list[tuple[str, dict[str, object]]],
    *,
    pas_tokamak_theta_applicable: bool = True,
    pas_tz_applicable: bool = True,
) -> RHS1SchurPreconditionerBuilders:
    offsets = {
        "theta_line": 1.0,
        "theta_dd": 2.0,
        "species_block": 3.0,
        "sxblock_tz": 4.0,
        "xblock_tz": 5.0,
        "xblock_tz_lmax": 6.0,
        "pas_ilu": 7.0,
        "xmg": 8.0,
        "pas_lite": 9.0,
        "pas_hybrid": 10.0,
        "pas_schur": 11.0,
        "pas_tokamak_theta": 12.0,
        "pas_tz": 13.0,
        "theta_zeta": 14.0,
        "zeta_line": 15.0,
        "zeta_dd": 16.0,
        "block": 17.0,
    }

    def make_builder(name: str):
        def builder(**kwargs):
            calls.append((name, dict(kwargs)))
            return lambda vector: jnp.asarray(vector, dtype=jnp.float64) + offsets[name]

        return builder

    return RHS1SchurPreconditionerBuilders(
        pas_tokamak_theta_applicable=lambda _op: bool(pas_tokamak_theta_applicable),
        pas_tz_applicable=lambda _op: bool(pas_tz_applicable),
        theta_line_builder=make_builder("theta_line"),
        theta_dd_builder=make_builder("theta_dd"),
        species_block_builder=make_builder("species_block"),
        sxblock_tz_builder=make_builder("sxblock_tz"),
        xblock_tz_builder=make_builder("xblock_tz"),
        xblock_tz_lmax_builder=make_builder("xblock_tz_lmax"),
        pas_xblock_ilu_builder=make_builder("pas_ilu"),
        xmg_builder=make_builder("xmg"),
        pas_lite_builder=make_builder("pas_lite"),
        pas_hybrid_builder=make_builder("pas_hybrid"),
        pas_schur_builder=make_builder("pas_schur"),
        pas_tokamak_theta_builder=make_builder("pas_tokamak_theta"),
        pas_tz_builder=make_builder("pas_tz"),
        theta_zeta_builder=make_builder("theta_zeta"),
        zeta_line_builder=make_builder("zeta_line"),
        zeta_dd_builder=make_builder("zeta_dd"),
        block_builder=make_builder("block"),
    )


def _patch_schur_dispatch_cache_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        schur_profile,
        "_rhsmode1_precond_cache_key",
        lambda op, kind: ("schur-dispatch", kind, id(op)),
    )


def _constrained_schur_op(*, point_at_x0: bool = False) -> SimpleNamespace:
    f_shape = (1, 2, 1, 2, 2)
    f_size = int(np.prod(f_shape))
    fblock = SimpleNamespace(
        f_shape=f_shape,
        collisionless=SimpleNamespace(n_xi_for_x=np.asarray([1, 1], dtype=np.int32)),
        exb_theta=None,
        exb_zeta=None,
        pas=None,
        fp=None,
        er_xdot=None,
        er_xidot=None,
    )
    return SimpleNamespace(
        rhs_mode=1,
        constraint_scheme=2,
        phi1_size=0,
        extra_size=2,
        f_size=f_size,
        total_size=f_size + 2,
        n_species=1,
        n_x=2,
        n_theta=2,
        n_zeta=2,
        point_at_x0=point_at_x0,
        theta_weights=np.ones(2, dtype=np.float64),
        zeta_weights=np.ones(2, dtype=np.float64),
        d_hat=np.ones((2, 2), dtype=np.float64),
        fblock=fblock,
    )


def _identity_schur_builders(
    calls: list[str],
    *,
    cache_block_for_diag: bool = False,
) -> RHS1SchurPreconditionerBuilders:
    def identity_builder(name: str):
        def builder(**kwargs):
            op = kwargs["op"]
            calls.append(name)
            if cache_block_for_diag and name == "block":
                key = schur_profile._rhsmode1_precond_cache_key(op, "point")
                schur_profile._RHSMODE1_PRECOND_CACHE[key] = SimpleNamespace(
                    block_inv_jnp=jnp.eye(int(op.n_x), dtype=jnp.float64)[None, :, :]
                )
            return lambda vector: jnp.asarray(vector, dtype=jnp.float64)

        return builder

    return RHS1SchurPreconditionerBuilders(
        pas_tokamak_theta_applicable=lambda _op: False,
        pas_tz_applicable=lambda _op: False,
        theta_line_builder=identity_builder("theta_line"),
        theta_dd_builder=identity_builder("theta_dd"),
        species_block_builder=identity_builder("species_block"),
        sxblock_tz_builder=identity_builder("sxblock_tz"),
        xblock_tz_builder=identity_builder("xblock_tz"),
        xblock_tz_lmax_builder=identity_builder("xblock_tz_lmax"),
        pas_xblock_ilu_builder=identity_builder("pas_ilu"),
        xmg_builder=identity_builder("xmg"),
        pas_lite_builder=identity_builder("pas_lite"),
        pas_hybrid_builder=identity_builder("pas_hybrid"),
        pas_schur_builder=identity_builder("pas_schur"),
        pas_tokamak_theta_builder=identity_builder("pas_tokamak_theta"),
        pas_tz_builder=identity_builder("pas_tz"),
        theta_zeta_builder=identity_builder("theta_zeta"),
        zeta_line_builder=identity_builder("zeta_line"),
        zeta_dd_builder=identity_builder("zeta_dd"),
        block_builder=identity_builder("block"),
    )


def _expected_identity_constraint2_schur(op: SimpleNamespace, vector: np.ndarray) -> np.ndarray:
    f = np.asarray(vector[: op.f_size], dtype=np.float64).reshape(op.fblock.f_shape).copy()
    r_e = np.asarray(vector[op.f_size :], dtype=np.float64).reshape((op.n_species, op.n_x))
    c_y = np.sum(f[:, :, 0, :, :], axis=(-2, -1))
    x_e = (c_y - r_e) / 4.0
    f_corr = np.zeros_like(f)
    f_corr[:, :, 0, :, :] = x_e[:, :, None, None]
    return np.concatenate(((f - f_corr).reshape((-1,)), x_e.reshape((-1,))))


def test_canonical_schur_base_kind_aliases() -> None:
    assert canonical_schur_base_kind("theta") == "theta_line"
    assert canonical_schur_base_kind("xblock_theta_zeta") == "xblock_tz"
    assert canonical_schur_base_kind("pas_l_tz") == "pas_tz"
    assert canonical_schur_base_kind("tokamak_theta") == "pas_tokamak_theta"
    assert canonical_schur_base_kind("theta_zeta") == "adi"
    assert canonical_schur_base_kind("unknown") is None


@pytest.mark.parametrize(
    ("env_value", "expected_builder", "expected_offset"),
    [
        ("theta", "theta_line", 1.0),
        ("theta_dd", "theta_dd", 2.0),
        ("species", "species_block", 3.0),
        ("sx_tz", "sxblock_tz", 4.0),
        ("xblock", "xblock_tz", 5.0),
        ("xblock_lmax", "xblock_tz_lmax", 6.0),
        ("block_ilu", "pas_ilu", 7.0),
        ("xmg", "xmg", 8.0),
        ("pas_light", "pas_lite", 9.0),
        ("pas_line_xcoarse", "pas_hybrid", 10.0),
        ("pas_block_schur", "pas_schur", 11.0),
        ("pas_tokamak", "pas_tokamak_theta", 12.0),
        ("pas_theta_zeta", "pas_tz", 13.0),
        ("line_zeta", "zeta_line", 15.0),
        ("zeta_dd", "zeta_dd", 16.0),
        ("point", "block", 17.0),
    ],
)
def test_build_rhs1_schur_preconditioner_dispatches_explicit_base_builders(
    monkeypatch,
    env_value: str,
    expected_builder: str,
    expected_offset: float,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    op = _schur_dispatch_op()
    _patch_schur_dispatch_cache_key(monkeypatch)
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SCHUR_BASE", env_value)
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DD_BLOCK_T", "bad")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DD_BLOCK_Z", "bad")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SCHUR_XBLOCK_TZ_LMAX", "bad")

    preconditioner = build_rhs1_schur_preconditioner(
        op=op,
        builders=_dispatch_builders(calls),
    )
    vector = jnp.arange(op.total_size, dtype=jnp.float64)

    np.testing.assert_allclose(np.asarray(preconditioner(vector)), np.asarray(vector + expected_offset))
    assert calls[0][0] == expected_builder
    if expected_builder in {"pas_lite", "pas_hybrid", "pas_schur"}:
        assert calls[0][1]["safe"] is False
    if expected_builder == "xblock_tz_lmax":
        assert calls[0][1]["lmax"] == 0
    if expected_builder in {"theta_dd", "zeta_dd"}:
        assert calls[0][1]["block"] == 8


def test_build_rhs1_schur_preconditioner_full_mode_uses_constraint2_schur_shortcut(monkeypatch) -> None:
    calls: list[str] = []
    op = _constrained_schur_op()
    _patch_schur_dispatch_cache_key(monkeypatch)
    schur_profile._RHSMODE1_SCHUR_CACHE.clear()
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SCHUR_BASE", "theta")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SCHUR_MODE", "full")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SCHUR_REG", "0")

    preconditioner = build_rhs1_schur_preconditioner(
        op=op,
        builders=_identity_schur_builders(calls),
    )
    vector = np.linspace(-0.5, 1.0, op.total_size)

    np.testing.assert_allclose(
        np.asarray(preconditioner(jnp.asarray(vector))),
        _expected_identity_constraint2_schur(op, vector),
        rtol=1e-12,
        atol=1e-12,
    )
    # The shortcut should keep the selected base builder unchanged; the
    # constraint algebra is checked by the closed-form expected vector above.
    assert calls == ["theta_line"]


def test_build_rhs1_schur_preconditioner_dense_mode_can_build_column_schur(monkeypatch) -> None:
    calls: list[str] = []
    op = _constrained_schur_op()
    _patch_schur_dispatch_cache_key(monkeypatch)
    schur_profile._RHSMODE1_SCHUR_CACHE.clear()
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SCHUR_BASE", "xmg")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SCHUR_MODE", "dense")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SCHUR_REG", "0")

    preconditioner = build_rhs1_schur_preconditioner(
        op=op,
        builders=_identity_schur_builders(calls),
    )
    vector = np.linspace(-0.25, 0.75, op.total_size)

    np.testing.assert_allclose(
        np.asarray(preconditioner(jnp.asarray(vector))),
        _expected_identity_constraint2_schur(op, vector),
        rtol=1e-12,
        atol=1e-12,
    )
    assert calls == ["xmg"]


def test_build_rhs1_schur_preconditioner_diag_mode_and_reduced_wrapper(monkeypatch) -> None:
    calls: list[str] = []
    op = _constrained_schur_op()
    _patch_schur_dispatch_cache_key(monkeypatch)
    schur_profile._RHSMODE1_SCHUR_CACHE.clear()
    schur_profile._RHSMODE1_PRECOND_CACHE.clear()
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SCHUR_BASE", "theta")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SCHUR_MODE", "not-a-mode")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SCHUR_EPS", "0")

    preconditioner = build_rhs1_schur_preconditioner(
        op=op,
        reduce_full=lambda vector: jnp.asarray(vector, dtype=jnp.float64),
        expand_reduced=lambda vector: jnp.asarray(vector, dtype=jnp.float64),
        builders=_identity_schur_builders(calls, cache_block_for_diag=True),
    )
    vector = np.linspace(0.1, 1.1, op.total_size)

    np.testing.assert_allclose(
        np.asarray(preconditioner(jnp.asarray(vector))),
        _expected_identity_constraint2_schur(op, vector),
        rtol=1e-12,
        atol=1e-12,
    )
    assert calls == ["theta_line", "block"]


def test_build_rhs1_schur_preconditioner_adi_composes_theta_and_zeta(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    op = _schur_dispatch_op()
    _patch_schur_dispatch_cache_key(monkeypatch)
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SCHUR_BASE", "adi")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_ADI_SWEEPS", "bad")
    preconditioner = build_rhs1_schur_preconditioner(
        op=op,
        builders=_dispatch_builders(calls),
    )
    vector = jnp.arange(op.total_size, dtype=jnp.float64)

    # Default bad-env fallback is two ADI sweeps: zeta(theta(v)) twice.
    np.testing.assert_allclose(np.asarray(preconditioner(vector)), np.asarray(vector + 32.0))
    assert [name for name, _kwargs in calls[:2]] == ["theta_line", "zeta_line"]


def test_build_rhs1_schur_preconditioner_auto_can_route_theta_zeta_branch(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    op = _schur_dispatch_op()
    op.fblock.pas = object()
    _patch_schur_dispatch_cache_key(monkeypatch)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SCHUR_BASE", raising=False)
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPECIES_BLOCK_MAX", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_TZ_MAX", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_TZ_PRECOND_MAX", "99")
    preconditioner = build_rhs1_schur_preconditioner(
        op=op,
        builders=_dispatch_builders(calls, pas_tokamak_theta_applicable=False, pas_tz_applicable=False),
        geom_scheme=99,
    )
    vector = jnp.arange(op.total_size, dtype=jnp.float64)

    np.testing.assert_allclose(np.asarray(preconditioner(vector)), np.asarray(vector + 14.0))
    assert calls[0][0] == "theta_zeta"


def test_active_native_stack_policy_bounds_memory_and_solver_aliases() -> None:
    policy = resolve_active_native_stack_policy(
        max_factor_nbytes=10_000,
        env={
            "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_STACK_BASE_BUDGET_FRACTION": "2.0",
            "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_STACK_SCHWARZ": "yes",
            "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_STACK_SCHWARZ_MAX_SIZE": "bad",
            "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_STACK_COARSE_MAX_SIZE": "-4",
            "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_STACK_COARSE_SOLVER": "ztaz",
        },
    )

    assert policy.base_budget_fraction == 1.0
    assert policy.base_budget_nbytes == 10_000
    assert policy.schwarz_requested is True
    assert policy.schwarz_max_size == 100_000
    assert policy.max_coarse_size == 1
    assert policy.coarse_solver_mode == "galerkin"

    small_fraction = resolve_active_native_stack_policy(
        max_factor_nbytes=10_000,
        env={"SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_STACK_BASE_BUDGET_FRACTION": "0.01"},
    )
    assert small_fraction.base_budget_fraction == 0.05
    assert small_fraction.base_budget_nbytes == 500


def test_active_field_split_policy_classifies_requested_solver_families() -> None:
    angular = resolve_active_native_field_split_sparse_coarse_policy(
        requested_kind="active-angular-line",
        env={"SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_XELL_COARSE_SOLVER": "petrov_galerkin"},
    )
    assert angular.is_angular_only is True
    assert angular.is_multiline is False
    assert angular.output_kind == "active_angular_line_field_split_sparse_coarse"
    assert angular.requested_base_kind == "active_angular_line"
    assert angular.coarse_solver_mode == "galerkin"

    multiline = resolve_active_native_field_split_sparse_coarse_policy(
        requested_kind="active_multiline_xell_angular",
        env={},
    )
    assert multiline.is_multiline is True
    assert multiline.output_kind == "active_multiline_field_split_sparse_coarse"
    assert multiline.requested_base_kind == "active_multiline_xell_angular"

    coupled = resolve_active_native_field_split_sparse_coarse_policy(
        requested_kind="active_coupled_kinetic_field_split_sparse_coarse",
        env={
            "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_COARSE_MAX_SIZE": "256",
            "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_COARSE_ADMISSION_PROBES": "0",
            "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_COARSE_ADMISSION_MAX_RELATIVE_RESIDUAL": "-1",
            "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_COARSE_ADMISSION_MIN_IMPROVEMENT": "-2",
        },
    )
    assert coupled.is_coupled_kinetic is True
    assert coupled.output_kind == "active_coupled_kinetic_field_split_sparse_coarse"
    assert coupled.requested_base_kind == "active_coupled_kinetic_block"
    assert coupled.max_coarse_size == 256
    assert coupled.admission_probes == 1
    assert coupled.admission_max_relative_residual == 0.0
    assert coupled.admission_min_improvement == 0.0


def test_active_sparse_coarse_residual_policy_maps_physics_families() -> None:
    filtered = resolve_active_sparse_coarse_residual_policy(
        requested_kind="filtered-normal",
        env={
            "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SPARSE_COARSE_MAX_SIZE": "0",
            "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SPARSE_COARSE_SOLVER": "normal_equations",
        },
    )
    assert filtered.base_kind == "active_filtered_sparse_factor"
    assert filtered.output_kind == "active_filtered_sparse_coarse"
    assert filtered.max_coarse_size == 1
    assert filtered.coarse_solver_mode == "least_squares"

    scaled = resolve_active_sparse_coarse_residual_policy(requested_kind="equilibrated_ilu", env={})
    assert scaled.base_kind == "active_scaled_ilu"
    assert scaled.output_kind == "active_tail_sparse_coarse"
    assert scaled.coarse_solver_mode == "galerkin"

    schwarz = resolve_active_sparse_coarse_residual_policy(requested_kind="ras", env={})
    assert schwarz.base_kind == "active_overlap_schwarz"

    xblock = resolve_active_sparse_coarse_residual_policy(requested_kind="xblock", env={})
    assert xblock.base_kind == "active_xblock"


def test_resolve_schur_base_prefers_pas_tz_for_geometry4_offender(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SCHUR_BASE", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_PAS_TZ_MIN", raising=False)

    assert (
        resolve_rhs1_schur_base_kind(
            base_kind_env="",
            n_theta=9,
            n_zeta=9,
            n_species=2,
            total_size=11350,
            nxi_for_x=[4, 4, 8, 12, 14],
            has_pas=True,
            has_fp=False,
            has_er_xdot=False,
            has_er_xidot=False,
            use_dkes_exb=False,
            pas_tokamak_theta_applicable=False,
            pas_tz_applicable=True,
            geom_scheme=4,
        )
        == "pas_tz"
    )


def test_resolve_schur_base_small_pas_fallback_uses_pas_schur(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPECIES_BLOCK_MAX", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_TZ_MAX", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_TZ_PRECOND_MAX", "0")

    assert (
        resolve_rhs1_schur_base_kind(
            base_kind_env="",
            n_theta=5,
            n_zeta=5,
            n_species=2,
            total_size=410,
            nxi_for_x=[4, 4],
            has_pas=True,
            has_fp=False,
            has_er_xdot=False,
            has_er_xidot=False,
            use_dkes_exb=False,
            pas_tokamak_theta_applicable=False,
            pas_tz_applicable=False,
            geom_scheme=4,
        )
        == "pas_schur"
    )


def test_resolve_schur_base_small_pas_prefers_species_block(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SCHUR_BASE", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPECIES_BLOCK_MAX", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_XBLOCK_TZ_MAX", raising=False)

    assert (
        resolve_rhs1_schur_base_kind(
            base_kind_env="",
            n_theta=7,
            n_zeta=7,
            n_species=2,
            total_size=410,
            nxi_for_x=[4, 4],
            has_pas=True,
            has_fp=False,
            has_er_xdot=False,
            has_er_xidot=False,
            use_dkes_exb=False,
            pas_tokamak_theta_applicable=False,
            pas_tz_applicable=False,
            geom_scheme=4,
        )
        == "species_block"
    )


def test_resolve_schur_base_dkes_uses_bounded_xblock_else_pas_ilu(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_DKES_XBLOCK_TZ_MAX_BYTES", raising=False)
    kwargs = dict(
        base_kind_env="",
        n_theta=9,
        n_zeta=9,
        n_species=2,
        total_size=5000,
        nxi_for_x=[3, 3],
        has_pas=True,
        has_fp=False,
        has_er_xdot=False,
        has_er_xidot=False,
        use_dkes_exb=True,
        pas_tokamak_theta_applicable=False,
        pas_tz_applicable=True,
        geom_scheme=11,
    )
    assert resolve_rhs1_schur_base_kind(**kwargs) == "xblock_tz"

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DKES_XBLOCK_TZ_MAX_BYTES", "1")
    assert resolve_rhs1_schur_base_kind(**kwargs) == "pas_ilu"


def test_resolve_schur_base_large_pas_er_prefers_xmg(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_PAS_XMG_MIN", raising=False)

    assert (
        resolve_rhs1_schur_base_kind(
            base_kind_env="",
            n_theta=9,
            n_zeta=9,
            n_species=2,
            total_size=60000,
            nxi_for_x=[4, 4, 8, 12, 14],
            has_pas=True,
            has_fp=False,
            has_er_xdot=True,
            has_er_xidot=False,
            use_dkes_exb=False,
            pas_tokamak_theta_applicable=False,
            pas_tz_applicable=True,
            geom_scheme=4,
        )
        == "xmg"
    )
