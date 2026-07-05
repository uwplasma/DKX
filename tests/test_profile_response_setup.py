from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from sfincs_jax.problems.profile_setup import (
    ProfileResponseLinearProblemSetupContext,
    RHS1ActiveReducedSystemSetupContext,
    SPARSE_HOST_SAFE_SOLVE_METHODS,
    build_rhs1_active_reduced_system_setup,
    equilibrium_name_hint_from_namelist,
    geometry_scheme_hint_from_namelist,
    materialize_profile_response_linear_problem,
    normalize_profile_solve_method_kind,
    resolve_rhs1_active_problem_setup,
    resolve_rhs1_domain_decomposition_setup,
    resolve_rhs1_dkes_adjustment_setup,
    resolve_rhs1_gmres_budget_setup,
    resolve_rhs1_initial_route_setup,
    resolve_rhs1_physics_flag_setup,
    resolve_rhs1_post_active_solve_policy_setup,
    resolve_rhs1_preconditioner_option_setup,
    resolve_rhs1_recycle_basis_setup,
    resolve_rhs1_reduced_mode_shape_setup,
    resolve_rhs1_tolerance_setup,
    resolve_solve_method_request_flags,
)


class FakeNamelist:
    def __init__(self, groups: dict[str, dict[str, object]]) -> None:
        self._groups = groups

    def group(self, name: str) -> dict[str, object]:
        return dict(self._groups.get(name, {}))


@dataclass(frozen=True)
class FakeFBlock:
    fp: object | None
    pas: object | None


@dataclass(frozen=True)
class FakeActiveFBlock:
    fp: object | None
    pas: object | None
    f_shape: tuple[int, int, int, int, int]


@dataclass(frozen=True)
class FakeOperator:
    rhs_mode: int
    include_phi1: bool
    constraint_scheme: int
    total_size: int
    phi1_size: int
    fblock: FakeFBlock
    n_species: int = 1
    f_size: int = 0
    n_zeta: int = 1


def test_normalize_profile_solve_method_kind_handles_case_and_dashes() -> None:
    assert normalize_profile_solve_method_kind(" Sparse-PC-GMRES ") == "sparse_pc_gmres"
    assert normalize_profile_solve_method_kind("xblock_sparse_pc_gmres") == "xblock_sparse_pc_gmres"


@dataclass(frozen=True)
class FakeActiveOperator:
    rhs_mode: int
    include_phi1: bool
    total_size: int
    f_size: int
    extra_size: int
    n_x: int
    point_at_x0: bool
    theta_weights: jnp.ndarray
    zeta_weights: jnp.ndarray
    d_hat: jnp.ndarray
    fblock: FakeActiveFBlock


class FakeTimer:
    def elapsed_s(self) -> float:
        return 1.25


def test_rhs1_gmres_budget_setup_applies_only_valid_env_overrides() -> None:
    setup = resolve_rhs1_gmres_budget_setup(
        restart=80,
        maxiter=400,
        env={"SFINCS_JAX_GMRES_RESTART": "120", "SFINCS_JAX_GMRES_MAXITER": "bad"},
    )

    assert setup.restart == 120
    assert setup.maxiter == 400
    assert setup.restart_env_forced
    assert not setup.maxiter_env_forced


def test_geometry_hints_accept_v3_case_variants() -> None:
    nml = FakeNamelist(
        {
            "geometryParameters": {
                "GEOMETRYSCHEME": "5",
                "equilibriumFile": "/tmp/wout_w7x.nc",
            }
        }
    )

    assert geometry_scheme_hint_from_namelist(nml) == 5
    assert equilibrium_name_hint_from_namelist(nml) == "wout_w7x.nc"


def test_rhs1_physics_flag_setup_preserves_namelist_spelling_variants() -> None:
    nml = FakeNamelist(
        {
            "physicsParameters": {
                "useDKESExBdrift": ".true.",
                "includeXDotTerm": 1,
                "includeElectricFieldTermInXiDot": "yes",
                "dPhiHatdrN": "-2.5",
                "Er": "not-a-number",
            }
        }
    )

    setup = resolve_rhs1_physics_flag_setup(nml)

    assert setup.use_dkes
    assert setup.include_xdot_sparse_pc
    assert setup.include_electric_field_xi_sparse_pc
    assert setup.er_abs_sparse_pc == 2.5


def test_rhs1_tolerance_setup_tightens_only_matching_physics_lanes() -> None:
    fp_op = FakeOperator(
        rhs_mode=1,
        include_phi1=False,
        constraint_scheme=1,
        total_size=90000,
        phi1_size=0,
        fblock=FakeFBlock(fp=object(), pas=None),
    )
    fp_setup = resolve_rhs1_tolerance_setup(op=fp_op, tol=1e-6, env={})

    assert fp_setup.tol == 1e-8
    assert fp_setup.fp_tightened
    assert not fp_setup.pas_tightened

    pas_op = FakeOperator(
        rhs_mode=1,
        include_phi1=False,
        constraint_scheme=2,
        total_size=2000,
        phi1_size=0,
        fblock=FakeFBlock(fp=None, pas=object()),
    )
    pas_setup = resolve_rhs1_tolerance_setup(
        op=pas_op,
        tol=1e-6,
        env={"SFINCS_JAX_RHSMODE1_PAS_TOL": "5e-9"},
    )

    assert pas_setup.tol == 5e-9
    assert pas_setup.pas_tightened
    assert not pas_setup.fp_tightened


def test_materialize_profile_response_linear_problem_builds_rhs_and_progress() -> None:
    nml = FakeNamelist(
        {
            "geometryParameters": {
                "geometryScheme": 5,
                "equilibriumFile": "/tmp/wout_unit.nc",
            }
        }
    )
    op = FakeOperator(
        rhs_mode=1,
        include_phi1=False,
        constraint_scheme=1,
        total_size=90000,
        phi1_size=0,
        fblock=FakeFBlock(fp=object(), pas=None),
    )
    messages: list[tuple[int, str]] = []
    marks: list[str] = []
    size_hints: list[int] = []
    policy_hints: list[dict[str, object]] = []

    setup = materialize_profile_response_linear_problem(
        ProfileResponseLinearProblemSetupContext(
            nml=nml,
            op=None,
            which_rhs=None,
            restart=80,
            maxiter=400,
            tol=1.0e-6,
            identity_shift=0.25,
            phi1_hat_base="phi1",
            emit=lambda level, message: messages.append((int(level), str(message))),
            mark=marks.append,
            env={"SFINCS_JAX_GMRES_RESTART": "96"},
            timer_factory=FakeTimer,
            build_operator=lambda **_kwargs: op,
            rhs_builder=lambda _op: np.asarray([3.0, 4.0]),
            norm=lambda rhs: np.linalg.norm(rhs),
            with_transport_rhs_settings=lambda op_use, **_kwargs: op_use,
            set_precond_size_hint=size_hints.append,
            set_precond_policy_hints=lambda **kwargs: policy_hints.append(dict(kwargs)),
        )
    )

    assert setup.op is op
    assert setup.which_rhs is None
    assert setup.restart == 96
    assert setup.maxiter == 400
    assert setup.restart_env_forced
    assert not setup.maxiter_env_forced
    assert setup.tol == 1.0e-8
    assert setup.fp_tol == 1.0e-8
    assert setup.rhs_norm == 5.0
    assert marks == ["operator_built", "rhs_assembled"]
    assert size_hints == [90000]
    assert policy_hints == [
        {
            "geom_scheme": 5,
            "has_pas": False,
            "has_fp": True,
            "include_phi1": False,
            "rhs_mode": 1,
        }
    ]
    assert (
        1,
        "solve_v3_full_system_linear_gmres: VMEC operator build start (wout_unit.nc)",
    ) in messages
    assert (
        1,
        "solve_v3_full_system_linear_gmres: VMEC operator build done elapsed_s=1.250",
    ) in messages
    assert any("FP tol tightened" in message for _, message in messages)
    assert messages[-1] == (
        2,
        "solve_v3_full_system_linear_gmres: rhs_norm=5.000000e+00",
    )


def test_materialize_profile_response_linear_problem_applies_transport_rhs_default() -> None:
    nml = FakeNamelist({"geometryParameters": {"geometryScheme": 2}})
    op = FakeOperator(
        rhs_mode=3,
        include_phi1=False,
        constraint_scheme=1,
        total_size=10,
        phi1_size=0,
        fblock=FakeFBlock(fp=None, pas=None),
    )
    transport_calls: list[int] = []

    setup = materialize_profile_response_linear_problem(
        ProfileResponseLinearProblemSetupContext(
            nml=nml,
            op=op,
            which_rhs=None,
            restart=80,
            maxiter=None,
            tol=1.0e-6,
            identity_shift=0.0,
            phi1_hat_base=None,
            emit=None,
            mark=lambda _label: None,
            env={},
            timer_factory=FakeTimer,
            build_operator=lambda **_kwargs: None,
            rhs_builder=lambda _op: np.asarray([2.0]),
            norm=lambda rhs: np.linalg.norm(rhs),
            with_transport_rhs_settings=lambda op_use, *, which_rhs: (
                transport_calls.append(int(which_rhs)) or op_use
            ),
            set_precond_size_hint=lambda _size: None,
            set_precond_policy_hints=lambda **_kwargs: None,
        )
    )

    assert setup.which_rhs == 1
    assert transport_calls == [1]
    assert setup.rhs_norm == 2.0


def test_rhs1_dkes_adjustment_setup_tightens_fp_tol_and_pas_budget() -> None:
    fp_op = FakeOperator(
        rhs_mode=1,
        include_phi1=False,
        constraint_scheme=1,
        total_size=100,
        phi1_size=0,
        fblock=FakeFBlock(fp=object(), pas=None),
    )
    fp_setup = resolve_rhs1_dkes_adjustment_setup(
        op=fp_op,
        tol=1.0e-6,
        fp_tol=1.0e-9,
        restart=40,
        maxiter=80,
        restart_env_forced=False,
        maxiter_env_forced=False,
        use_dkes=True,
        dkes_gmres_budget=lambda **kwargs: (kwargs["restart"], kwargs["maxiter"], False, False),
        env={},
    )

    assert fp_setup.tol == 1.0e-9
    assert fp_setup.restart == 40
    assert any("FP DKES tol tightened" in message for _, message in fp_setup.messages)

    pas_op = FakeOperator(
        rhs_mode=1,
        include_phi1=False,
        constraint_scheme=2,
        total_size=100,
        phi1_size=0,
        fblock=FakeFBlock(fp=None, pas=object()),
    )

    def budget(**_kwargs):
        return 32, 64, True, True

    pas_setup = resolve_rhs1_dkes_adjustment_setup(
        op=pas_op,
        tol=1.0e-6,
        fp_tol=1.0e-9,
        restart=20,
        maxiter=40,
        restart_env_forced=False,
        maxiter_env_forced=False,
        use_dkes=True,
        dkes_gmres_budget=budget,
        env={},
    )

    assert pas_setup.restart == 32
    assert pas_setup.maxiter == 64
    assert any("PAS DKES default GMRES budget" in message for _, message in pas_setup.messages)


def test_solve_method_request_flags_preserve_driver_aliases() -> None:
    assert SPARSE_HOST_SAFE_SOLVE_METHODS == {
        "sparse_host_safe",
        "safe_sparse_host",
        "sparse_host_or_petsc_compat",
    }

    xblock = resolve_solve_method_request_flags(
        solve_method="xblock-sparse-pc-gmres",
        xblock_active_dof_env="true",
    )
    assert xblock.kind == "xblock_sparse_pc_gmres"
    assert xblock.sparse_pc_gmres_requested
    assert xblock.sparse_host_like_requested
    assert xblock.xblock_active_dof_requested

    structured = resolve_solve_method_request_flags(solve_method="structured-full-csr")
    assert structured.structured_full_csr_explicit_requested

    invalid_env = resolve_solve_method_request_flags(
        solve_method="xblock_sparse_pc_gmres",
        xblock_active_dof_env="maybe",
    )
    assert not invalid_env.xblock_active_dof_requested


def test_rhs1_initial_route_setup_controls_structured_auto_admission() -> None:
    nml = FakeNamelist(
        {
            "physicsParameters": {
                "EParallelHat": "bad",
                "eParallelHat": "-0.25",
            }
        }
    )
    op = FakeOperator(
        rhs_mode=1,
        include_phi1=False,
        constraint_scheme=1,
        total_size=4096,
        phi1_size=0,
        fblock=FakeFBlock(fp=object(), pas=None),
    )
    calls: list[dict[str, object]] = []

    def policy(**kwargs: object) -> bool:
        calls.append(kwargs)
        return True

    setup = resolve_rhs1_initial_route_setup(
        nml=nml,
        op=op,
        solve_method="auto",
        xblock_active_dof_env="",
        use_implicit=False,
        force_krylov=False,
        sharded_axis="theta",
        backend="cpu",
        device_count=2,
        structured_auto_allowed=policy,
    )

    assert setup.method_flags.kind == "auto"
    assert setup.structured_eparallel_abs == 0.25
    assert setup.structured_sharded_multidevice
    assert setup.structured_auto_allowed
    assert calls == [
        {
            "op": op,
            "active_size": 4096,
            "use_implicit": False,
            "solve_method_kind": "auto",
            "backend": "cpu",
            "eparallel_abs": 0.25,
        }
    ]

    forced = resolve_rhs1_initial_route_setup(
        nml=nml,
        op=op,
        solve_method="auto",
        xblock_active_dof_env="",
        use_implicit=False,
        force_krylov=True,
        sharded_axis=None,
        backend="gpu",
        device_count=1,
        structured_auto_allowed=policy,
    )

    assert not forced.structured_auto_allowed
    assert len(calls) == 1


def test_rhs1_initial_route_setup_skips_auto_policy_for_explicit_structured() -> None:
    nml = FakeNamelist({"physicsParameters": {"EParallelHat": "1.0"}})
    op = FakeOperator(
        rhs_mode=1,
        include_phi1=False,
        constraint_scheme=1,
        total_size=16,
        phi1_size=0,
        fblock=FakeFBlock(fp=object(), pas=None),
    )

    def policy(**_kwargs: object) -> bool:
        raise AssertionError("explicit structured CSR should not query auto admission")

    setup = resolve_rhs1_initial_route_setup(
        nml=nml,
        op=op,
        solve_method="structured-full-csr",
        xblock_active_dof_env="",
        use_implicit=True,
        force_krylov=False,
        sharded_axis="zeta",
        backend="cpu",
        device_count=4,
        structured_auto_allowed=policy,
    )

    assert setup.method_flags.structured_full_csr_explicit_requested
    assert setup.use_implicit_requested
    assert setup.structured_eparallel_abs == 0.0
    assert not setup.structured_auto_allowed
    assert not setup.structured_sharded_multidevice


def test_rhs1_recycle_basis_setup_filters_shape_and_keeps_latest_vectors() -> None:
    basis = [
        np.asarray([1.0, 0.0]),
        np.asarray([2.0, 0.0, 0.0]),
        np.asarray([3.0, 0.0, 0.0]),
        np.asarray([4.0, 0.0, 0.0]),
    ]

    setup = resolve_rhs1_recycle_basis_setup(
        recycle_basis=basis,
        total_size=3,
        recycle_k_env="2",
        asarray=np.asarray,
    )

    assert setup.recycle_k == 2
    assert len(setup.basis) == 2
    np.testing.assert_allclose(setup.basis[0], np.asarray([3.0, 0.0, 0.0]))
    np.testing.assert_allclose(setup.basis[1], np.asarray([4.0, 0.0, 0.0]))

    invalid = resolve_rhs1_recycle_basis_setup(
        recycle_basis=basis,
        total_size=3,
        recycle_k_env="bad",
        asarray=np.asarray,
    )
    assert invalid.recycle_k == 4
    assert len(invalid.basis) == 3

    disabled = resolve_rhs1_recycle_basis_setup(
        recycle_basis=basis,
        total_size=3,
        recycle_k_env="0",
        asarray=np.asarray,
    )
    assert disabled.basis == ()


def test_rhs1_active_reduced_system_setup_compacts_full_operator() -> None:
    op = FakeActiveOperator(
        rhs_mode=1,
        include_phi1=False,
        total_size=5,
        f_size=5,
        extra_size=0,
        n_x=1,
        point_at_x0=False,
        theta_weights=jnp.ones((1,), dtype=jnp.float64),
        zeta_weights=jnp.ones((1,), dtype=jnp.float64),
        d_hat=jnp.ones((1, 1), dtype=jnp.float64),
        fblock=FakeActiveFBlock(fp=object(), pas=None, f_shape=(1, 1, 5, 1, 1)),
    )
    active_idx = jnp.asarray([0, 2, 4], dtype=jnp.int32)
    full_to_active = jnp.asarray([1, 0, 2, 0, 3], dtype=jnp.int32)
    rhs = jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float64)
    x0 = jnp.asarray([10.0, 11.0, 12.0, 13.0, 14.0], dtype=jnp.float64)

    setup = build_rhs1_active_reduced_system_setup(
        RHS1ActiveReducedSystemSetupContext(
            op=op,
            rhs=rhs,
            x0=x0,
            mv=lambda x: 2.0 * x + 1.0,
            active_idx_jnp=active_idx,
            full_to_active_jnp=full_to_active,
            active_size=3,
            use_pas_projection=False,
            recycle_basis=(),
            tol=1.0e-1,
            atol=0.0,
        )
    )

    np.testing.assert_allclose(np.asarray(setup.rhs_reduced), [1.0, 3.0, 5.0])
    np.testing.assert_allclose(np.asarray(setup.x0_reduced), [10.0, 12.0, 14.0])
    np.testing.assert_allclose(
        np.asarray(setup.expand_reduced(jnp.asarray([7.0, 8.0, 9.0]))),
        [7.0, 0.0, 8.0, 0.0, 9.0],
    )
    np.testing.assert_allclose(
        np.asarray(setup.mv_reduced(jnp.asarray([7.0, 8.0, 9.0]))),
        [15.0, 17.0, 19.0],
    )
    assert np.isclose(setup.target_reduced, 1.0e-1 * np.sqrt(35.0))
    assert np.isclose(setup.target_stage2, 1.0e-1)


def test_rhs1_active_reduced_system_setup_projects_pas_flux_surface_average() -> None:
    op = FakeActiveOperator(
        rhs_mode=1,
        include_phi1=False,
        total_size=9,
        f_size=8,
        extra_size=1,
        n_x=2,
        point_at_x0=False,
        theta_weights=jnp.ones((2,), dtype=jnp.float64),
        zeta_weights=jnp.ones((2,), dtype=jnp.float64),
        d_hat=jnp.ones((2, 2), dtype=jnp.float64),
        fblock=FakeActiveFBlock(fp=None, pas=object(), f_shape=(1, 2, 1, 2, 2)),
    )
    active_idx = jnp.arange(8, dtype=jnp.int32)
    full_to_active = jnp.arange(1, 9, dtype=jnp.int32)
    rhs = jnp.arange(1.0, 10.0, dtype=jnp.float64)

    setup = build_rhs1_active_reduced_system_setup(
        RHS1ActiveReducedSystemSetupContext(
            op=op,
            rhs=rhs,
            x0=None,
            mv=lambda x: x,
            active_idx_jnp=active_idx,
            full_to_active_jnp=full_to_active,
            active_size=8,
            use_pas_projection=True,
            recycle_basis=(rhs,),
            tol=1.0e-3,
            atol=0.0,
        )
    )

    expected_projected = np.asarray(
        [-1.5, -0.5, 0.5, 1.5, -1.5, -0.5, 0.5, 1.5]
    )
    np.testing.assert_allclose(np.asarray(setup.rhs_reduced), expected_projected)
    np.testing.assert_allclose(
        np.asarray(setup.mv_reduced(jnp.asarray(rhs[:8]))), expected_projected
    )
    expanded = np.asarray(setup.expand_reduced(jnp.asarray(rhs[:8])))
    np.testing.assert_allclose(expanded[:8], np.asarray(rhs[:8]))
    assert expanded[8] == 0.0

    projected_precond = setup.wrap_pas_preconditioner(lambda v: v + 2.0)
    np.testing.assert_allclose(
        np.asarray(projected_precond(jnp.asarray(setup.rhs_reduced))),
        expected_projected,
    )
    assert setup.x0_reduced is not None
    np.testing.assert_allclose(np.asarray(setup.x0_reduced), expected_projected)


def test_rhs1_reduced_mode_shape_setup_detects_truncated_pitch_grid() -> None:
    reduced = resolve_rhs1_reduced_mode_shape_setup(
        nxi_for_x=[5, 7, 9],
        n_xi=9,
    )

    assert reduced.nxi_for_x.dtype == np.int32
    assert reduced.max_l == 9
    assert reduced.has_reduced_modes

    full = resolve_rhs1_reduced_mode_shape_setup(
        nxi_for_x=[9, 9],
        n_xi=9,
    )
    assert not full.has_reduced_modes


def test_rhs1_active_problem_setup_combines_dkes_active_dof_and_preconditioner_options() -> None:
    nml = FakeNamelist(
        {
            "physicsParameters": {
                "useDKESExBDrift": ".true.",
                "includeXDotTerm": ".true.",
                "includeElectricFieldTermInXiDot": ".true.",
                "dPhiHatdrN": "-3.0",
            },
            "preconditionerOptions": {
                "PRECONDITIONER_SPECIES": "2",
                "PRECONDITIONER_X": "3",
                "PRECONDITIONER_XI": "4",
                "PRECONDITIONER_X_MIN_L": "1",
            },
            "geometryParameters": {"geometryScheme": "5"},
        }
    )
    op = FakeOperator(
        rhs_mode=1,
        include_phi1=False,
        constraint_scheme=2,
        total_size=6,
        phi1_size=0,
        fblock=FakeFBlock(fp=None, pas=object()),
        f_size=6,
    )

    def budget(**_kwargs: object) -> tuple[int, int, bool, bool]:
        return 24, 48, True, True

    setup = resolve_rhs1_active_problem_setup(
        nml=nml,
        op=op,
        tol=1.0e-6,
        fp_tol=1.0e-9,
        restart=12,
        maxiter=24,
        restart_env_forced=False,
        maxiter_env_forced=False,
        has_reduced_modes=True,
        sparse_host_like_requested=False,
        xblock_active_dof_requested=False,
        dkes_gmres_budget=budget,
        active_dof_indices=lambda _op: np.asarray([0, 2, 4], dtype=np.int32),
        env={"SFINCS_JAX_ACTIVE_DOF": "true"},
    )

    assert setup.restart == 24
    assert setup.maxiter == 48
    assert setup.use_dkes
    assert setup.include_xdot_sparse_pc
    assert setup.include_electric_field_xi_sparse_pc
    assert setup.er_abs_sparse_pc == 3.0
    assert setup.preconditioner_species == 2
    assert setup.preconditioner_x == 3
    assert setup.preconditioner_x_min_l == 1
    assert setup.preconditioner_xi == 4
    assert setup.geom_scheme == 5
    assert setup.use_active_dof_mode
    assert setup.active_size == 3
    np.testing.assert_allclose(np.asarray(setup.active_idx_jnp), np.asarray([0, 2, 4]))
    assert any("PAS DKES default GMRES budget" in message for _, message in setup.messages)


def test_preconditioner_option_setup_controls_pas_projection() -> None:
    nml = FakeNamelist(
        {
            "preconditionerOptions": {
                "PRECONDITIONER_SPECIES": "1",
                "PRECONDITIONER_X": "1",
                "PRECONDITIONER_XI": "1",
                "PRECONDITIONER_X_MIN_L": "2",
            },
            "geometryParameters": {"geometryScheme": "5"},
        }
    )
    op = FakeOperator(
        rhs_mode=1,
        include_phi1=False,
        constraint_scheme=2,
        total_size=3000,
        phi1_size=0,
        fblock=FakeFBlock(fp=None, pas=object()),
    )

    setup = resolve_rhs1_preconditioner_option_setup(
        nml=nml,
        op=op,
        sparse_host_like_requested=False,
        use_active_dof_mode=False,
        env={},
    )

    assert setup.preconditioner_x_min_l == 2
    assert setup.geom_scheme == 5
    assert setup.pas_project_mode == "auto"
    assert setup.use_pas_projection
    assert setup.use_active_dof_mode

    full_precond = FakeNamelist(
        {
            "preconditionerOptions": {
                "PRECONDITIONER_SPECIES": "0",
                "PRECONDITIONER_X": "0",
                "PRECONDITIONER_XI": "0",
            },
            "geometryParameters": {"geometryScheme": "5"},
        }
    )
    disabled = resolve_rhs1_preconditioner_option_setup(
        nml=full_precond,
        op=op,
        sparse_host_like_requested=False,
        use_active_dof_mode=False,
        env={},
    )

    assert disabled.full_preconditioner_requested
    assert not disabled.pas_project_enabled
    assert not disabled.use_pas_projection


def test_rhs1_domain_decomposition_setup_resolves_explicit_and_default_blocks() -> None:
    setup = resolve_rhs1_domain_decomposition_setup(
        n_theta=25,
        n_zeta=39,
        sum_nxi=60,
        distributed_env="off",
        device_count=4,
        auto_axis="theta",
        theta_block_env="11",
        zeta_block_env="bad",
        theta_overlap_env="2",
        zeta_overlap_env="",
        overlap_env="",
        patch_dof_target_env="bad",
    )

    assert setup.sharded_axis is None
    assert setup.patch_dof_target == 1200
    assert setup.block("theta") == 11
    assert setup.block("zeta") == 8
    assert setup.overlap("theta", default=1) == 2
    assert setup.overlap("zeta", default=3) == 3


def test_rhs1_domain_decomposition_setup_auto_blocks_and_overlaps_for_sharded_axis() -> None:
    setup = resolve_rhs1_domain_decomposition_setup(
        n_theta=25,
        n_zeta=39,
        sum_nxi=60,
        distributed_env="auto",
        device_count=2,
        auto_axis="zeta",
        theta_block_env="",
        zeta_block_env="",
        theta_overlap_env="",
        zeta_overlap_env="",
        overlap_env="",
        patch_dof_target_env="700",
    )

    assert setup.sharded_axis == "zeta"
    assert setup.block("theta") == 8
    assert 1 <= setup.block("zeta") <= 39
    assert setup.overlap("theta", default=0) == 0
    assert setup.overlap("zeta", default=0) >= 1


def test_rhs1_post_active_solve_policy_selects_dense_and_sharded_auto() -> None:
    op = FakeOperator(
        rhs_mode=1,
        include_phi1=False,
        constraint_scheme=1,
        total_size=100,
        phi1_size=0,
        fblock=FakeFBlock(fp=object(), pas=None),
        n_zeta=9,
        n_species=2,
    )
    dense = resolve_rhs1_post_active_solve_policy_setup(
        op=op,
        restart=20,
        maxiter=40,
        solve_method="auto",
        active_size=50,
        use_active_dof_mode=True,
        full_precond_requested=True,
        geom_scheme=5,
        dense_backend_allowed=True,
        backend="cpu",
        sharded_axis_hint=None,
        device_count=1,
        env={},
    )

    assert dense.solve_method == "dense"
    assert any("full preconditioner requested" in message for _, message in dense.messages)

    sharded = resolve_rhs1_post_active_solve_policy_setup(
        op=op,
        restart=20,
        maxiter=40,
        solve_method="auto",
        active_size=5000,
        use_active_dof_mode=False,
        full_precond_requested=False,
        geom_scheme=5,
        dense_backend_allowed=True,
        backend="cpu",
        sharded_axis_hint="theta",
        device_count=2,
        env={},
    )

    assert sharded.solve_method == "auto"
    assert any("preserving auto solver selection" in message for _, message in sharded.messages)


def test_rhs1_post_active_solve_policy_selects_pas_large_bicgstab() -> None:
    op = FakeOperator(
        rhs_mode=1,
        include_phi1=False,
        constraint_scheme=2,
        total_size=100000,
        phi1_size=0,
        fblock=FakeFBlock(fp=None, pas=object()),
        n_zeta=1,
        n_species=1,
    )
    setup = resolve_rhs1_post_active_solve_policy_setup(
        op=op,
        restart=20,
        maxiter=40,
        solve_method="auto",
        active_size=90000,
        use_active_dof_mode=False,
        full_precond_requested=False,
        geom_scheme=1,
        dense_backend_allowed=True,
        backend="cpu",
        sharded_axis_hint=None,
        device_count=1,
        env={"SFINCS_JAX_PAS_LARGE_BICGSTAB_FASTPATH_MIN": "80000"},
    )

    assert setup.tokamak_pas
    assert setup.pas_large_bicgstab_fastpath
    assert setup.solve_method == "bicgstab"
    assert setup.pas_large_fastpath_min == 80000
