from __future__ import annotations

from dataclasses import dataclass

from sfincs_jax.problems.profile_response.setup import (
    SPARSE_HOST_SAFE_SOLVE_METHODS,
    equilibrium_name_hint_from_namelist,
    geometry_scheme_hint_from_namelist,
    resolve_rhs1_domain_decomposition_setup,
    resolve_rhs1_dkes_adjustment_setup,
    resolve_rhs1_gmres_budget_setup,
    resolve_rhs1_physics_flag_setup,
    resolve_rhs1_post_active_solve_policy_setup,
    resolve_rhs1_preconditioner_option_setup,
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
class FakeOperator:
    rhs_mode: int
    include_phi1: bool
    constraint_scheme: int
    total_size: int
    phi1_size: int
    fblock: FakeFBlock
    n_zeta: int = 1
    n_species: int = 1


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
