from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import sfincs_jax.problems.profile_policies as profile_policies
import sfincs_jax.problems.transport_policies as transport_policies


def _rhs1_fp_op(*, constraint_scheme: int = 1, include_phi1: bool = False, point_at_x0: bool = False):
    return SimpleNamespace(
        rhs_mode=1,
        include_phi1=include_phi1,
        constraint_scheme=constraint_scheme,
        point_at_x0=point_at_x0,
        fblock=SimpleNamespace(fp=object(), pas=None),
    )


def _rhs1_pas_op(*, constraint_scheme: int = 1, include_phi1: bool = False):
    return SimpleNamespace(
        rhs_mode=1,
        include_phi1=include_phi1,
        constraint_scheme=constraint_scheme,
        point_at_x0=False,
        fblock=SimpleNamespace(fp=None, pas=object()),
    )


def _transport_op(
    *,
    rhs_mode: int = 3,
    has_fp: bool = False,
    include_phi1: bool = False,
    n_x: int = 1,
    constraint_scheme: int = 2,
    n_theta: int = 9,
    n_zeta: int = 5,
    total_size: int = 1024,
):
    return SimpleNamespace(
        rhs_mode=rhs_mode,
        include_phi1=include_phi1,
        constraint_scheme=constraint_scheme,
        n_x=n_x,
        n_theta=n_theta,
        n_zeta=n_zeta,
        total_size=total_size,
        fblock=SimpleNamespace(fp=object() if has_fp else None, pas=None),
    )


def test_constraint0_petsc_compat_can_be_enabled_and_respects_guards(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_CS0_PETSC_COMPAT", "1")
    assert profile_policies.rhs1_constraint0_petsc_compat(
        op=_rhs1_fp_op(constraint_scheme=0),
        solve_method_kind="incremental",
        sparse_precond_mode="auto",
        active_size=4096,
        sparse_max_size=6000,
    )
    assert not profile_policies.rhs1_constraint0_petsc_compat(
        op=_rhs1_fp_op(constraint_scheme=1),
        solve_method_kind="incremental",
        sparse_precond_mode="auto",
        active_size=4096,
        sparse_max_size=6000,
    )
    assert not profile_policies.rhs1_constraint0_petsc_compat(
        op=_rhs1_fp_op(constraint_scheme=0),
        solve_method_kind="dense",
        sparse_precond_mode="auto",
        active_size=4096,
        sparse_max_size=6000,
    )
    assert not profile_policies.rhs1_constraint0_petsc_compat(
        op=_rhs1_fp_op(constraint_scheme=0),
        solve_method_kind="incremental",
        sparse_precond_mode="off",
        active_size=4096,
        sparse_max_size=6000,
    )
    assert not profile_policies.rhs1_constraint0_petsc_compat(
        op=_rhs1_fp_op(constraint_scheme=0),
        solve_method_kind="incremental",
        sparse_precond_mode="auto",
        active_size=7000,
        sparse_max_size=6000,
    )


def test_constraint0_sparse_first_defaults_to_accelerator_and_respects_overrides(
    monkeypatch,
) -> None:
    op = _rhs1_fp_op(constraint_scheme=0)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_CS0_SPARSE_FIRST", raising=False)

    assert not profile_policies.rhs1_constraint0_sparse_first(
        op=op,
        solve_method_kind="incremental",
        sparse_precond_mode="auto",
        active_size=4096,
        sparse_max_size=6000,
        backend="cpu",
    )
    assert profile_policies.rhs1_constraint0_sparse_first(
        op=op,
        solve_method_kind="incremental",
        sparse_precond_mode="auto",
        active_size=4096,
        sparse_max_size=6000,
        backend="gpu",
    )

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_CS0_SPARSE_FIRST", "1")
    assert profile_policies.rhs1_constraint0_sparse_first(
        op=op,
        solve_method_kind="incremental",
        sparse_precond_mode="auto",
        active_size=4096,
        sparse_max_size=6000,
        backend="cpu",
    )
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_CS0_SPARSE_FIRST", "0")
    assert not profile_policies.rhs1_constraint0_sparse_first(
        op=op,
        solve_method_kind="incremental",
        sparse_precond_mode="auto",
        active_size=4096,
        sparse_max_size=6000,
        backend="gpu",
    )


def test_constraint0_dense_fallback_policy() -> None:
    assert profile_policies.rhs1_constraint0_dense_fallback_allowed(_rhs1_fp_op(constraint_scheme=1))
    assert not profile_policies.rhs1_constraint0_dense_fallback_allowed(_rhs1_fp_op(constraint_scheme=0))


def test_sparse_pc_default_permc_spec_targets_pas_er_rows() -> None:
    assert (
        profile_policies.rhsmode1_sparse_pc_default_permc_spec(
            constrained_pas_pc=True,
            tokamak_pas_er_pc=True,
            n_species=2,
        )
        == "MMD_AT_PLUS_A"
    )
    assert (
        profile_policies.rhsmode1_sparse_pc_default_permc_spec(
            constrained_pas_pc=True,
            tokamak_pas_er_pc=True,
            n_species=1,
        )
        == "MMD_AT_PLUS_A"
    )
    assert (
        profile_policies.rhsmode1_sparse_pc_default_permc_spec(
            constrained_pas_pc=True,
            tokamak_pas_er_pc=False,
            n_species=2,
        )
        == "MMD_ATA"
    )
    assert (
        profile_policies.rhsmode1_sparse_pc_default_permc_spec(
            constrained_pas_pc=False,
            tokamak_pas_er_pc=False,
            n_species=2,
        )
        == "COLAMD"
    )


def test_sparse_pc_default_restart_caps_one_species_pas_er_without_env() -> None:
    assert (
        profile_policies.rhsmode1_sparse_pc_default_restart(
            requested_restart=80,
            restart_env_value="",
            tokamak_pas_er_pc=True,
            n_species=1,
        )
        == 40
    )
    assert (
        profile_policies.rhsmode1_sparse_pc_default_restart(
            requested_restart=20,
            restart_env_value="",
            tokamak_pas_er_pc=True,
            n_species=1,
        )
        == 20
    )
    assert (
        profile_policies.rhsmode1_sparse_pc_default_restart(
            requested_restart=80,
            restart_env_value="",
            tokamak_pas_er_pc=True,
            n_species=2,
        )
        == 80
    )
    assert (
        profile_policies.rhsmode1_sparse_pc_default_restart(
            requested_restart=80,
            restart_env_value="80",
            tokamak_pas_er_pc=True,
            n_species=1,
        )
        == 80
    )
    assert (
        profile_policies.rhsmode1_sparse_pc_default_restart(
            requested_restart=80,
            restart_env_value="",
            tokamak_pas_er_pc=False,
            n_species=1,
        )
        == 80
    )


def test_pas_tz_guarded_structured_levels_parse_aliases_and_empty_tokens() -> None:
    levels = profile_policies.parse_rhs1_pas_tz_guarded_structured_levels(
        " xmg_collision + , diag ; x_grid | collisions "
    )

    assert levels == ("xmg", "collision")
    assert profile_policies.parse_rhs1_pas_tz_guarded_structured_levels("off") == ()
    assert profile_policies.parse_rhs1_pas_tz_guarded_structured_levels("unknown,,") == ()


def test_sparse_preconditioner_config_parses_scipy_and_false_aliases(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PRECOND", "spilu")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_MATVEC", "false")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_OPERATOR", "yes")

    scipy_config = profile_policies.rhs1_sparse_preconditioner_config_from_env(
        has_pas=False,
        use_dkes=False,
        active_size=3000,
        backend="cpu",
    )

    assert scipy_config.precond_mode == "on"
    assert scipy_config.precond_kind == "scipy"
    assert scipy_config.use_matvec is False
    assert scipy_config.operator_mode == "on"

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PRECOND", "off")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_OPERATOR", "0")
    disabled_config = profile_policies.rhs1_sparse_preconditioner_config_from_env(
        has_pas=False,
        use_dkes=False,
        active_size=3000,
        backend="cpu",
    )

    assert disabled_config.precond_mode == "off"
    assert disabled_config.precond_kind == "auto"
    assert disabled_config.operator_mode == "off"


def test_xblock_fallback_initial_guess_rejects_bad_candidate_fail_closed() -> None:
    class BadArray:
        def __array__(self, dtype=None):  # noqa: ANN001
            raise RuntimeError("bad candidate")

    original = np.ones(3)
    x0, reused, improved = profile_policies.rhs1_xblock_fallback_initial_guess(
        candidate=BadArray(),
        original_x0=original,
        rhs_shape=(3,),
        candidate_residual_norm=0.5,
        rhs_norm=1.0,
        precondition_side="left",
    )

    assert x0 is original
    assert reused is False
    assert improved is True

    x0_right, reused_right, improved_right = profile_policies.rhs1_xblock_fallback_initial_guess(
        candidate=np.zeros(3),
        original_x0=original,
        rhs_shape=(3,),
        candidate_residual_norm=0.5,
        rhs_norm=1.0,
        precondition_side="right",
    )

    assert x0_right is original
    assert reused_right is False
    assert improved_right is True


def test_host_factor_probe_rejects_invalid_solve_and_accepts_bounded_factor(
    monkeypatch,
) -> None:
    class GoodFactor:
        def solve(self, rhs):
            return np.asarray(rhs) * 2.0

    class BadShapeFactor:
        def solve(self, rhs):
            return np.ones((2, 2))

    class RaisingFactor:
        def solve(self, rhs):
            raise RuntimeError("factor failed")

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_FACTOR_PROBE_MAX", "3.0")
    assert profile_policies.rhs1_host_factor_probe_ok(factor=GoodFactor(), block_size=3)
    assert not profile_policies.rhs1_host_factor_probe_ok(factor=None, block_size=3)
    assert not profile_policies.rhs1_host_factor_probe_ok(factor=GoodFactor(), block_size=0)
    assert not profile_policies.rhs1_host_factor_probe_ok(factor=BadShapeFactor(), block_size=3)
    assert not profile_policies.rhs1_host_factor_probe_ok(factor=RaisingFactor(), block_size=3)

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_FACTOR_PROBE_MAX", "1.0")
    assert not profile_policies.rhs1_host_factor_probe_ok(factor=GoodFactor(), block_size=3)


def test_sparse_exact_lu_requested_covers_pas_full_and_accelerator_small_case(monkeypatch) -> None:
    monkeypatch.setattr("sfincs_jax.problems.profile_policies.jax.default_backend", lambda: "gpu")
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_EXACT_LU", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_EXACT_LU_MAX", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_EXACT_LU_ACCEL_SMALL_MAX", raising=False)

    assert profile_policies.rhsmode1_sparse_exact_lu_requested_current_backend(
        op=_rhs1_fp_op(),
        solve_method_kind="incremental",
        active_size=3500,
        sparse_max_size=6000,
        preconditioner_x=1,
        use_dkes=False,
    )

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_EXACT_LU", "1")
    assert profile_policies.rhsmode1_sparse_exact_lu_requested_current_backend(
        op=_rhs1_pas_op(),
        solve_method_kind="incremental",
        active_size=3500,
        sparse_max_size=6000,
        full_precond_requested=False,
        preconditioner_x=1,
        use_dkes=False,
    )


def test_sparse_exact_lu_requested_respects_off_dense_and_size_guards(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_EXACT_LU", "off")
    assert not profile_policies.rhsmode1_sparse_exact_lu_requested_current_backend(
        op=_rhs1_fp_op(),
        solve_method_kind="incremental",
        active_size=3500,
        sparse_max_size=6000,
        preconditioner_x=0,
        use_dkes=False,
    )

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_EXACT_LU", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_EXACT_LU_MAX", "2000")
    assert not profile_policies.rhsmode1_sparse_exact_lu_requested_current_backend(
        op=_rhs1_fp_op(),
        solve_method_kind="incremental",
        active_size=3500,
        sparse_max_size=6000,
        preconditioner_x=0,
        use_dkes=False,
    )
    assert not profile_policies.rhsmode1_sparse_exact_lu_requested_current_backend(
        op=_rhs1_fp_op(),
        solve_method_kind="dense",
        active_size=1000,
        sparse_max_size=6000,
        preconditioner_x=0,
        use_dkes=False,
    )


def test_large_cpu_xblock_skip_primary_allowed_positive_and_guards(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_LARGE_CPU_XBLOCK_SKIP_PRIMARY", raising=False)
    monkeypatch.setattr("sfincs_jax.problems.profile_policies.jax.default_backend", lambda: "cpu")
    assert profile_policies.rhsmode1_large_cpu_xblock_skip_primary_allowed_current_backend(
        op=_rhs1_fp_op(),
        solve_method_kind="incremental",
        active_size=20000,
        sparse_max_size=6000,
        preconditioner_species=1,
        preconditioner_x=1,
        preconditioner_xi=1,
        pre_theta=0,
        pre_zeta=0,
        use_implicit=False,
        rhs1_precond_env="auto",
    )
    assert not profile_policies.rhsmode1_large_cpu_xblock_skip_primary_allowed_current_backend(
        op=_rhs1_fp_op(point_at_x0=True),
        solve_method_kind="incremental",
        active_size=20000,
        sparse_max_size=6000,
        preconditioner_species=1,
        preconditioner_x=1,
        preconditioner_xi=1,
        pre_theta=0,
        pre_zeta=0,
        use_implicit=False,
        rhs1_precond_env="auto",
    )
    assert not profile_policies.rhsmode1_large_cpu_xblock_skip_primary_allowed_current_backend(
        op=_rhs1_fp_op(),
        solve_method_kind="incremental",
        active_size=20000,
        sparse_max_size=6000,
        preconditioner_species=1,
        preconditioner_x=1,
        preconditioner_xi=1,
        pre_theta=0,
        pre_zeta=0,
        use_implicit=False,
        rhs1_precond_env="xblock_tz",
    )


def test_large_cpu_sparse_skip_primary_allowed_positive_and_guards(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_LARGE_CPU_SKIP_PRIMARY", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_LARGE_CPU_SKIP_PRIMARY_MIN", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_LARGE_CPU_SKIP_PRIMARY_MAX", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_LARGE_CPU_RESCUE_EXACT_LU_MAX", raising=False)
    monkeypatch.setattr("sfincs_jax.problems.profile_policies.jax.default_backend", lambda: "cpu")

    assert profile_policies.rhsmode1_large_cpu_sparse_skip_primary_allowed_current_backend(
        op=_rhs1_fp_op(),
        solve_method_kind="incremental",
        active_size=11496,
        sparse_max_size=6000,
        use_implicit=False,
    )
    assert not profile_policies.rhsmode1_large_cpu_sparse_skip_primary_allowed_current_backend(
        op=_rhs1_fp_op(),
        solve_method_kind="dense",
        active_size=11496,
        sparse_max_size=6000,
        use_implicit=False,
    )
    assert not profile_policies.rhsmode1_large_cpu_sparse_skip_primary_allowed_current_backend(
        op=_rhs1_fp_op(include_phi1=True),
        solve_method_kind="incremental",
        active_size=11496,
        sparse_max_size=6000,
        use_implicit=False,
    )
    assert profile_policies.rhsmode1_large_cpu_sparse_skip_primary_allowed_current_backend(
        op=_rhs1_fp_op(),
        solve_method_kind="incremental",
        active_size=11496,
        sparse_max_size=6000,
        use_implicit=True,
    )


def test_scipy_rescue_abs_floor_invalid_override_and_large_cpu_floor(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SCIPY_GMRES_RESCUE_ABS", "bad")
    assert (
        profile_policies.rhs1_scipy_rescue_abs_floor_after_xblock(
            op=_rhs1_fp_op(),
            active_size=20_000,
            used_large_cpu_xblock_shortcut=True,
            used_explicit_fp_xblock_seed=True,
            use_implicit=False,
            backend="cpu",
        )
        == 0.0
    )

    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SCIPY_GMRES_RESCUE_ABS", raising=False)
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SCIPY_GMRES_RESCUE_ABS_MIN", "1000")
    floor = profile_policies.rhs1_scipy_rescue_abs_floor_after_xblock(
        op=_rhs1_fp_op(),
        active_size=20_000,
        used_large_cpu_xblock_shortcut=True,
        used_explicit_fp_xblock_seed=True,
        use_implicit=False,
        backend="cpu",
    )

    assert floor == 1.0e-9


def test_host_dense_shortcut_and_dense_auto_policy_guards(monkeypatch) -> None:
    fp_op = _rhs1_fp_op()
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_HOST_DENSE_SHORTCUT", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_DENSE_HOST_LU", raising=False)
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_HOST_DENSE_SHORTCUT_MAX_BYTES", "0")

    assert profile_policies.rhs1_host_dense_shortcut_allowed(
        op=fp_op,
        active_size=100,
        use_implicit=False,
        solve_method_kind="auto",
        backend="gpu",
        dense_fallback_max=200,
    )
    assert not profile_policies.rhs1_host_dense_shortcut_allowed(
        op=fp_op,
        active_size=100,
        use_implicit=True,
        solve_method_kind="auto",
        backend="gpu",
        dense_fallback_max=200,
    )
    assert not profile_policies.rhs1_host_dense_shortcut_allowed(
        op=fp_op,
        active_size=100,
        use_implicit=False,
        solve_method_kind="auto",
        backend="cpu",
        dense_fallback_max=200,
    )
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_HOST_DENSE_SHORTCUT", "0")
    assert not profile_policies.rhs1_host_dense_shortcut_allowed(
        op=fp_op,
        active_size=100,
        use_implicit=False,
        solve_method_kind="auto",
        backend="gpu",
        dense_fallback_max=200,
    )

    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_DENSE_AUTO_FP_CUTOFF", raising=False)
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DENSE_ALLOW_ACCELERATOR", "0")
    assert not profile_policies.rhs1_dense_auto_fp_allowed(
        backend="gpu",
        active_size=100,
        dense_active_cutoff=200,
    )
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DENSE_ALLOW_ACCELERATOR", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DENSE_FP_ACCELERATOR_MIN", "50")
    assert profile_policies.rhs1_dense_auto_fp_allowed(
        backend="gpu",
        active_size=100,
        dense_active_cutoff=200,
    )


def test_rhs1_sparse_pc_auto_windows_cover_tokamak_and_3d_guards(monkeypatch) -> None:
    pas_tokamak = SimpleNamespace(
        rhs_mode=1,
        include_phi1=False,
        constraint_scheme=2,
        n_zeta=1,
        n_xi=24,
        fblock=SimpleNamespace(fp=None, pas=object()),
    )
    fp_tokamak = SimpleNamespace(
        rhs_mode=1,
        include_phi1=False,
        constraint_scheme=1,
        n_zeta=1,
        n_xi=24,
        fblock=SimpleNamespace(fp=object(), pas=None),
    )
    fp_3d = SimpleNamespace(
        rhs_mode=1,
        include_phi1=False,
        constraint_scheme=1,
        n_zeta=5,
        n_xi=64,
        fblock=SimpleNamespace(fp=object(), pas=None),
    )

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_TOKAMAK_PAS_ER_SPARSE_PC", "1")
    assert profile_policies.rhs1_tokamak_pas_er_sparse_pc_auto_allowed(
        op=pas_tokamak,
        active_size=500,
        use_implicit=False,
        solve_method_kind="auto",
        backend="gpu",
        er_abs=1.0,
        use_dkes=True,
        include_xdot=False,
        include_electric_field_xi=False,
    )
    assert not profile_policies.rhs1_tokamak_pas_er_sparse_pc_auto_allowed(
        op=pas_tokamak,
        active_size=500,
        use_implicit=False,
        solve_method_kind="auto",
        backend="gpu",
        er_abs=0.0,
        use_dkes=True,
        include_xdot=False,
        include_electric_field_xi=False,
    )

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_TOKAMAK_PAS_NOER_SPARSE_PC", "1")
    assert profile_policies.rhs1_tokamak_pas_noer_sparse_pc_auto_allowed(
        op=pas_tokamak,
        active_size=500,
        use_implicit=False,
        solve_method_kind="default",
        backend="cpu",
        er_abs=0.0,
    )
    assert not profile_policies.rhs1_tokamak_pas_noer_sparse_pc_auto_allowed(
        op=pas_tokamak,
        active_size=500,
        use_implicit=False,
        solve_method_kind="default",
        backend="cpu",
        er_abs=1.0,
    )

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_TOKAMAK_FP_ER_SPARSE_PC", "1")
    assert profile_policies.rhs1_tokamak_fp_er_sparse_pc_auto_allowed(
        op=fp_tokamak,
        active_size=500,
        use_implicit=False,
        solve_method_kind="incremental",
        backend="gpu",
        er_abs=1.0,
        use_dkes=False,
        include_xdot=True,
        include_electric_field_xi=False,
    )

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FP3D_SPARSE_PC", "1")
    assert profile_policies.rhs1_fp_3d_sparse_pc_auto_allowed(
        op=fp_3d,
        active_size=6000,
        use_implicit=False,
        solve_method_kind="auto",
        backend="cpu",
        eparallel_abs=0.0,
    )
    assert not profile_policies.rhs1_fp_3d_sparse_pc_auto_allowed(
        op=fp_3d,
        active_size=6000,
        use_implicit=False,
        solve_method_kind="auto",
        backend="gpu",
        eparallel_abs=0.0,
    )


def test_transport_sparse_direct_first_attempt_handles_invalid_cpu_max_env(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT", raising=False)
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_FIRST_CPU_MAX", "bad")
    assert transport_policies.transport_sparse_direct_first_attempt_allowed(
        op=_transport_op(rhs_mode=2),
        size=16382,
        use_implicit=False,
        backend="cpu",
    )


def test_transport_host_gmres_first_attempt_respects_disable_and_invalid_max(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_HOST_GMRES_FIRST", "off")
    assert not transport_policies.transport_host_gmres_first_attempt_allowed(
        op=_transport_op(rhs_mode=3, has_fp=False, n_x=1),
        size=54811,
        use_implicit=False,
        backend="cpu",
    )
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_HOST_GMRES_FIRST", raising=False)
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_HOST_GMRES_FIRST_MAX", "bad")
    assert transport_policies.transport_host_gmres_first_attempt_allowed(
        op=_transport_op(rhs_mode=3, has_fp=False, n_x=1),
        size=54811,
        use_implicit=False,
        backend="cpu",
    )


def test_transport_host_gmres_accepts_preconditioned_residual_handles_invalid_env_and_nonfinite(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_HOST_GMRES_TRUE_RATIO", "bad")
    assert not transport_policies.transport_host_gmres_accepts_preconditioned_residual(
        op=_transport_op(rhs_mode=2, has_fp=False, n_x=3),
        true_residual_norm=float("nan"),
        target_true=1.0e-6,
    )
    assert transport_policies.transport_host_gmres_accepts_preconditioned_residual(
        op=_transport_op(rhs_mode=2, has_fp=False, n_x=3),
        true_residual_norm=8.0e-6,
        target_true=1.0e-6,
    )
    assert not transport_policies.transport_host_gmres_accepts_preconditioned_residual(
        op=_transport_op(rhs_mode=2, has_fp=False, n_x=3),
        true_residual_norm=2.0e-5,
        target_true=1.0e-6,
    )


def test_transport_precondition_side_accepts_none_override(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_PRECONDITION_SIDE", "none")
    assert transport_policies.transport_precondition_side(
        op=_transport_op(),
        use_implicit=False,
    ) == "none"


def test_transport_disable_auto_recycle_forced_on(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_DISABLE_AUTO_RECYCLE", "1")
    assert transport_policies.transport_disable_auto_recycle(
        op=_transport_op(rhs_mode=2, has_fp=True, n_x=4, constraint_scheme=1),
        use_implicit=True,
        backend="cpu",
    )


def test_transport_sparse_direct_needs_float64_retry_nonfinite_and_invalid_ratio(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_FLOAT64_RETRY_RATIO", "bad")
    assert transport_policies.transport_sparse_direct_needs_float64_retry(
        factor_dtype=np.dtype(np.float32),
        residual_norm=float("nan"),
        target_true=1.0e-6,
    )
    assert transport_policies.transport_sparse_direct_needs_float64_retry(
        factor_dtype=np.dtype(np.float32),
        residual_norm=2.0e-5,
        target_true=1.0e-6,
    )


def test_transport_sparse_factor_dtype_respects_explicit_env(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_FACTOR_DTYPE", "float32")
    assert transport_policies.transport_sparse_factor_dtype(
        size=99999,
        use_implicit=False,
        backend="cpu",
        host_sparse_factor_dtype=lambda **_: np.dtype(np.float64),
    ) == np.dtype(np.float32)
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_FACTOR_DTYPE", "float64")
    assert transport_policies.transport_sparse_factor_dtype(
        size=1,
        use_implicit=False,
        backend="cpu",
        host_sparse_factor_dtype=lambda **_: np.dtype(np.float32),
    ) == np.dtype(np.float64)
