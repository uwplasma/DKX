from __future__ import annotations

from dataclasses import dataclass

from sfincs_jax.problems.profile_response.setup import (
    SPARSE_HOST_SAFE_SOLVE_METHODS,
    equilibrium_name_hint_from_namelist,
    geometry_scheme_hint_from_namelist,
    resolve_rhs1_gmres_budget_setup,
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
    fblock: FakeFBlock


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


def test_rhs1_tolerance_setup_tightens_only_matching_physics_lanes() -> None:
    fp_op = FakeOperator(
        rhs_mode=1,
        include_phi1=False,
        constraint_scheme=1,
        total_size=90000,
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
