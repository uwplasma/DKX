from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp

from sfincs_jax.transport_preconditioner_dispatch import (
    TransportPreconditionerContext,
    TransportPreconditionerDispatchBuilders,
    TransportSparseJaxConfig,
    auto_transport_preconditioner_choice,
    build_transport_preconditioner_from_kind,
    build_transport_strong_preconditioner_from_kind,
    normalize_transport_preconditioner_kind,
    resolve_transport_precondition_side_for_kind,
    resolve_transport_preconditioner_choice,
    transport_dd_config_from_env,
    transport_sparse_jax_config_from_env,
)


def _op(
    *,
    has_fp: bool = True,
    n_species: int = 2,
    n_x: int = 4,
    n_theta: int = 9,
    n_zeta: int = 5,
    total_size: int = 2048,
    rhs_mode: int = 2,
    include_phi1: bool = False,
):
    return SimpleNamespace(
        rhs_mode=rhs_mode,
        include_phi1=include_phi1,
        n_species=n_species,
        n_x=n_x,
        n_theta=n_theta,
        n_zeta=n_zeta,
        total_size=total_size,
        fblock=SimpleNamespace(fp=object() if has_fp else None),
    )


def _builders(calls: list[tuple[str, dict]]) -> TransportPreconditionerDispatchBuilders:
    def _mk(name: str):
        def _builder(**kwargs):
            calls.append((name, kwargs))
            return lambda v, _name=name: (_name, v)

        return _builder

    return TransportPreconditionerDispatchBuilders(
        collision_builder=_mk("collision"),
        sxblock_builder=_mk("sxblock"),
        block_builder=_mk("block"),
        xmg_builder=_mk("xmg"),
        theta_dd_builder=_mk("theta_dd"),
        theta_schwarz_builder=_mk("theta_schwarz"),
        zeta_dd_builder=_mk("zeta_dd"),
        zeta_schwarz_builder=_mk("zeta_schwarz"),
        tzfft_builder=_mk("tzfft"),
        sparse_jax_builder=_mk("sparse_jax"),
        sparse_jax_cache_key=lambda op, key: ("cache", key, int(op.total_size)),
        apply_operator_cached=lambda op, x: x,
        precond_dtype=lambda size: jnp.float32 if int(size) > 100 else jnp.float64,
        fp_tzfft_builder=_mk("fp_tzfft"),
        fp_tzfft_line_builder=_mk("fp_tzfft_line"),
        fp_tzfft_line_schur_builder=_mk("fp_tzfft_line_schur"),
        fp_local_geom_line_builder=_mk("fp_local_geom_line"),
        fp_xblock_tz_lu_builder=_mk("fp_xblock_tz_lu"),
        fp_xblock_tz_lu_schur_builder=_mk("fp_xblock_tz_lu_schur"),
        fp_structured_fblock_lu_builder=_mk("fp_structured_fblock_lu"),
        fp_fortran_reduced_lu_builder=_mk("fp_fortran_reduced_lu"),
        fp_direct_active_block_schur_builder=_mk("fp_direct_active_block_schur"),
    )


def test_normalize_transport_preconditioner_kind_maps_aliases() -> None:
    assert normalize_transport_preconditioner_kind(env_value="stream_fft") == "tzfft"
    assert normalize_transport_preconditioner_kind(env_value="fp_streaming_fft") == "fp_tzfft"
    assert normalize_transport_preconditioner_kind(env_value="fp_streaming_line") == "fp_tzfft_line"
    assert normalize_transport_preconditioner_kind(env_value="fp_block_thomas") == "fp_tzfft_line"
    assert normalize_transport_preconditioner_kind(env_value="fp_line_schur") == "fp_tzfft_line_schur"
    assert normalize_transport_preconditioner_kind(env_value="fp_tzfft_schur") == "fp_tzfft_line_schur"
    assert normalize_transport_preconditioner_kind(env_value="fp_geom_line") == "fp_local_geom_line"
    assert normalize_transport_preconditioner_kind(env_value="fp_nonavg_line") == "fp_local_geom_line"
    assert normalize_transport_preconditioner_kind(env_value="fp_xblock_lu") == "fp_xblock_tz_lu"
    assert normalize_transport_preconditioner_kind(env_value="fp_angular_xblock_lu") == "fp_xblock_tz_lu"
    assert normalize_transport_preconditioner_kind(env_value="fp_xblock_lu_schur") == "fp_xblock_tz_lu_schur"
    assert normalize_transport_preconditioner_kind(env_value="fp_xblock_schur") == "fp_xblock_tz_lu_schur"
    assert normalize_transport_preconditioner_kind(env_value="fp_fblock_lu") == "fp_structured_fblock_lu"
    assert normalize_transport_preconditioner_kind(env_value="fp_kinetic_lu") == "fp_structured_fblock_lu"
    assert normalize_transport_preconditioner_kind(env_value="fp_petsc_like_lu") == "fp_fortran_reduced_lu"
    assert normalize_transport_preconditioner_kind(env_value="fp_reduced_pmat_lu") == "fp_fortran_reduced_lu"
    assert normalize_transport_preconditioner_kind(env_value="fp_active_true_block") == "fp_direct_active_block_schur"
    assert normalize_transport_preconditioner_kind(env_value="fp_true_block_lu") == "fp_direct_active_block_schur"
    assert normalize_transport_preconditioner_kind(env_value="dd_theta") == "theta_dd"
    assert normalize_transport_preconditioner_kind(env_value="schwarz_zeta") == "zeta_schwarz"
    assert normalize_transport_preconditioner_kind(env_value="none") is None
    assert normalize_transport_preconditioner_kind(env_value="weird") == "auto"


def test_transport_sparse_jax_config_and_dd_config_handle_invalid_env(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_DROP_TOL", "bad")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_DROP_REL", "bad")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_JAX_REG", "bad")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_JAX_OMEGA", "bad")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_JAX_SWEEPS", "bad")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_JAX_MAX_MB", "bad")
    cfg = transport_sparse_jax_config_from_env()
    assert cfg.drop_tol == 0.0
    assert cfg.drop_rel == 1.0e-6
    assert cfg.reg == 1.0e-10
    assert cfg.omega == 0.8
    assert cfg.sweeps == 2
    assert cfg.max_mb == 128.0

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_DD_BLOCK_T", "bad")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_DD_BLOCK_Z", "bad")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_DD_OVERLAP", "bad")
    dd = transport_dd_config_from_env(op=_op(n_theta=7, n_zeta=3))
    assert dd.block_theta == 7
    assert dd.overlap_theta == 1
    assert dd.block_zeta == 3
    assert dd.overlap_zeta == 1


def test_auto_transport_preconditioner_choice_prefers_tzfft_for_collisionless_transport() -> None:
    kind, strong = auto_transport_preconditioner_choice(
        op=_op(has_fp=False, n_x=1, n_theta=17, n_zeta=5, total_size=1200),
        default_solver_kind="gmres",
        parallel_workers=1,
        dense_mem_block=False,
        tzfft_backend_allowed=True,
        shard_axis=None,
    )
    assert kind == "tzfft"
    assert strong == "tzfft"


def test_auto_transport_preconditioner_choice_uses_fortran_reduced_lu_for_fp_transport_by_default() -> None:
    kind, strong = auto_transport_preconditioner_choice(
        op=_op(has_fp=True, n_x=8, n_theta=25, n_zeta=51, total_size=1_020_002),
        default_solver_kind="gmres",
        parallel_workers=1,
        dense_mem_block=False,
        tzfft_backend_allowed=False,
        shard_axis=None,
    )
    assert kind == "fp_fortran_reduced_lu"
    assert strong == "fp_fortran_reduced_lu"


def test_auto_transport_preconditioner_choice_prefers_bounded_fp_tzfft_line_when_forced(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_TZFFT_LINE_AUTO", "1")
    kind, strong = auto_transport_preconditioner_choice(
        op=_op(has_fp=True, n_x=8, n_theta=25, n_zeta=51, total_size=1_020_002),
        default_solver_kind="gmres",
        parallel_workers=1,
        dense_mem_block=False,
        tzfft_backend_allowed=False,
        shard_axis=None,
    )
    assert kind == "fp_tzfft_line"
    assert strong == "fp_tzfft_line"


def test_auto_transport_preconditioner_choice_prefers_bounded_fp_tzfft_line_schur_when_forced(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_TZFFT_LINE_SCHUR_AUTO", "1")
    kind, strong = auto_transport_preconditioner_choice(
        op=_op(has_fp=True, n_x=8, n_theta=25, n_zeta=51, total_size=1_020_002),
        default_solver_kind="gmres",
        parallel_workers=1,
        dense_mem_block=False,
        tzfft_backend_allowed=False,
        shard_axis=None,
    )
    assert kind == "fp_tzfft_line_schur"
    assert strong == "fp_tzfft_line_schur"


def test_auto_transport_preconditioner_choice_prefers_fp_local_geom_line_when_forced(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_LOCAL_GEOM_LINE_AUTO", "1")
    kind, strong = auto_transport_preconditioner_choice(
        op=_op(has_fp=True, n_x=8, n_theta=25, n_zeta=51, total_size=1_020_002),
        default_solver_kind="gmres",
        parallel_workers=1,
        dense_mem_block=False,
        tzfft_backend_allowed=False,
        shard_axis=None,
    )
    assert kind == "fp_local_geom_line"
    assert strong == "fp_local_geom_line"


def test_auto_transport_preconditioner_choice_prefers_structured_fblock_lu_when_forced(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_STRUCTURED_FBLOCK_LU_AUTO", "1")
    kind, strong = auto_transport_preconditioner_choice(
        op=_op(has_fp=True, n_x=8, n_theta=25, n_zeta=51, total_size=1_020_002),
        default_solver_kind="gmres",
        parallel_workers=1,
        dense_mem_block=False,
        tzfft_backend_allowed=False,
        shard_axis=None,
    )
    assert kind == "fp_structured_fblock_lu"
    assert strong == "fp_structured_fblock_lu"


def test_auto_transport_preconditioner_choice_prefers_xblock_tz_lu_when_forced(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_XBLOCK_TZ_LU_AUTO", "1")
    kind, strong = auto_transport_preconditioner_choice(
        op=_op(has_fp=True, n_x=8, n_theta=25, n_zeta=51, total_size=1_020_002),
        default_solver_kind="gmres",
        parallel_workers=1,
        dense_mem_block=False,
        tzfft_backend_allowed=False,
        shard_axis=None,
    )
    assert kind == "fp_xblock_tz_lu"
    assert strong == "fp_xblock_tz_lu"


def test_auto_transport_preconditioner_choice_prefers_xblock_tz_lu_schur_when_forced(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_XBLOCK_TZ_LU_SCHUR_AUTO", "1")
    kind, strong = auto_transport_preconditioner_choice(
        op=_op(has_fp=True, n_x=8, n_theta=25, n_zeta=51, total_size=1_020_002),
        default_solver_kind="gmres",
        parallel_workers=1,
        dense_mem_block=False,
        tzfft_backend_allowed=False,
        shard_axis=None,
    )
    assert kind == "fp_xblock_tz_lu_schur"
    assert strong == "fp_xblock_tz_lu_schur"


def test_auto_transport_preconditioner_choice_prefers_fortran_reduced_lu_when_forced(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_AUTO", "1")
    kind, strong = auto_transport_preconditioner_choice(
        op=_op(has_fp=True, n_x=8, n_theta=25, n_zeta=51, total_size=1_020_002),
        default_solver_kind="gmres",
        parallel_workers=1,
        dense_mem_block=False,
        tzfft_backend_allowed=False,
        shard_axis=None,
    )
    assert kind == "fp_fortran_reduced_lu"
    assert strong == "fp_fortran_reduced_lu"


def test_auto_transport_preconditioner_choice_promotes_fortran_reduced_lu_by_default(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_AUTO", raising=False)
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_AUTO_MIN", raising=False)
    kind, strong = auto_transport_preconditioner_choice(
        op=_op(has_fp=True, n_x=8, n_theta=25, n_zeta=51, total_size=1_020_002),
        default_solver_kind="gmres",
        parallel_workers=1,
        dense_mem_block=False,
        tzfft_backend_allowed=False,
        shard_axis=None,
    )

    assert kind == "fp_fortran_reduced_lu"
    assert strong == "fp_fortran_reduced_lu"


def test_auto_transport_preconditioner_choice_can_disable_fortran_reduced_lu(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_AUTO", "0")
    kind, strong = auto_transport_preconditioner_choice(
        op=_op(has_fp=True, n_x=8, n_theta=25, n_zeta=51, total_size=1_020_002),
        default_solver_kind="gmres",
        parallel_workers=1,
        dense_mem_block=False,
        tzfft_backend_allowed=False,
        shard_axis=None,
    )

    assert kind != "fp_fortran_reduced_lu"
    assert strong != "fp_fortran_reduced_lu"


def test_build_transport_fortran_reduced_lu_uses_reduced_factor_only_when_active() -> None:
    calls: list[tuple[str, dict]] = []
    builders = _builders(calls)
    context = TransportPreconditionerContext(
        op=_op(total_size=2048),
        active_size=128,
        use_active_dof_mode=True,
        reduce_full=lambda x: x,
        expand_reduced=lambda x: x,
        active_indices_np=[0, 1, 2],
    )

    build_transport_preconditioner_from_kind(
        kind="fp_fortran_reduced_lu",
        context=context,
        builders=builders,
        dd_config=transport_dd_config_from_env(op=context.op),
        sparse_jax_config=transport_sparse_jax_config_from_env(),
        use_reduced=False,
    )
    build_transport_preconditioner_from_kind(
        kind="fp_fortran_reduced_lu",
        context=context,
        builders=builders,
        dd_config=transport_dd_config_from_env(op=context.op),
        sparse_jax_config=transport_sparse_jax_config_from_env(),
        use_reduced=True,
    )

    assert calls[0][0] == "sxblock"
    assert calls[1][0] == "fp_fortran_reduced_lu"
    assert calls[1][1]["active_indices_np"] == [0, 1, 2]


def test_auto_transport_preconditioner_choice_prefers_sharded_schwarz_when_large_parallel(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_DD_AUTO_MIN", "4000")
    kind, strong = auto_transport_preconditioner_choice(
        op=_op(has_fp=True, total_size=5000),
        default_solver_kind="gmres",
        parallel_workers=2,
        dense_mem_block=False,
        tzfft_backend_allowed=False,
        shard_axis="zeta",
    )
    assert kind == "zeta_schwarz"
    assert strong == "zeta_schwarz"


def test_resolve_transport_preconditioner_choice_disables_tzfft_on_backend() -> None:
    messages: list[str] = []
    kind, strong = resolve_transport_preconditioner_choice(
        op=_op(has_fp=False, n_x=1, total_size=1200),
        transport_precond_kind="tzfft",
        default_solver_kind="gmres",
        parallel_workers=1,
        dense_mem_block=False,
        tzfft_backend_allowed=False,
        shard_axis=None,
        backend="gpu",
        emit=lambda _lvl, msg: messages.append(msg),
    )
    assert kind == "collision"
    assert strong is None
    assert any("tzfft preconditioner disabled" in msg for msg in messages)


def test_resolve_transport_precondition_side_for_kind_keeps_fp_line_left_only() -> None:
    side, changed = resolve_transport_precondition_side_for_kind(
        kind="fp_tzfft_line",
        requested_side="right",
    )
    assert side == "left"
    assert changed

    side, changed = resolve_transport_precondition_side_for_kind(
        kind="fp_tzfft_line_schur",
        requested_side="right",
    )
    assert side == "left"
    assert changed

    side, changed = resolve_transport_precondition_side_for_kind(
        kind="sxblock",
        requested_side="right",
    )
    assert side == "right"
    assert not changed

    side, changed = resolve_transport_precondition_side_for_kind(
        kind="fp_structured_fblock_lu",
        requested_side="right",
    )
    assert side == "left"
    assert changed

    side, changed = resolve_transport_precondition_side_for_kind(
        kind="fp_xblock_tz_lu",
        requested_side="right",
    )
    assert side == "left"
    assert changed

    side, changed = resolve_transport_precondition_side_for_kind(
        kind="fp_xblock_tz_lu_schur",
        requested_side="right",
    )
    assert side == "left"
    assert changed

    side, changed = resolve_transport_precondition_side_for_kind(
        kind="fp_direct_active_block_schur",
        requested_side="right",
    )
    assert side == "left"
    assert changed


def test_build_transport_preconditioner_from_kind_passes_dd_reduced_kwargs() -> None:
    calls: list[tuple[str, dict]] = []
    builders = _builders(calls)
    context = TransportPreconditionerContext(
        op=_op(),
        active_size=128,
        use_active_dof_mode=True,
        reduce_full=lambda x: x,
        expand_reduced=lambda x: x,
    )
    precond = build_transport_preconditioner_from_kind(
        kind="theta_schwarz",
        context=context,
        builders=builders,
        dd_config=transport_dd_config_from_env(op=_op()),
        sparse_jax_config=TransportSparseJaxConfig(0.0, 1.0e-6, 1.0e-10, 0.8, 2, 128.0),
        use_reduced=True,
    )
    assert callable(precond)
    assert calls[0][0] == "theta_schwarz"
    assert "reduce_full" in calls[0][1]
    assert "expand_reduced" in calls[0][1]


def test_build_transport_preconditioner_from_kind_dispatches_fp_tzfft() -> None:
    calls: list[tuple[str, dict]] = []
    builders = _builders(calls)
    context = TransportPreconditionerContext(op=_op(), active_size=128, use_active_dof_mode=False)
    precond = build_transport_preconditioner_from_kind(
        kind="fp_tzfft",
        context=context,
        builders=builders,
        dd_config=transport_dd_config_from_env(op=_op()),
        sparse_jax_config=TransportSparseJaxConfig(0.0, 1.0e-6, 1.0e-10, 0.8, 2, 128.0),
        use_reduced=False,
    )
    assert callable(precond)
    assert calls[0][0] == "fp_tzfft"


def test_build_transport_preconditioner_from_kind_dispatches_fp_tzfft_line() -> None:
    calls: list[tuple[str, dict]] = []
    builders = _builders(calls)
    context = TransportPreconditionerContext(op=_op(), active_size=128, use_active_dof_mode=False)
    precond = build_transport_preconditioner_from_kind(
        kind="fp_tzfft_line",
        context=context,
        builders=builders,
        dd_config=transport_dd_config_from_env(op=_op()),
        sparse_jax_config=TransportSparseJaxConfig(0.0, 1.0e-6, 1.0e-10, 0.8, 2, 128.0),
        use_reduced=False,
    )
    assert callable(precond)
    assert calls[0][0] == "fp_tzfft_line"


def test_build_transport_preconditioner_from_kind_dispatches_fp_tzfft_line_schur() -> None:
    calls: list[tuple[str, dict]] = []
    builders = _builders(calls)
    context = TransportPreconditionerContext(op=_op(), active_size=128, use_active_dof_mode=False)
    precond = build_transport_preconditioner_from_kind(
        kind="fp_tzfft_line_schur",
        context=context,
        builders=builders,
        dd_config=transport_dd_config_from_env(op=_op()),
        sparse_jax_config=TransportSparseJaxConfig(0.0, 1.0e-6, 1.0e-10, 0.8, 2, 128.0),
        use_reduced=False,
    )
    assert callable(precond)
    assert calls[0][0] == "fp_tzfft_line_schur"


def test_build_transport_preconditioner_from_kind_dispatches_fp_local_geom_line() -> None:
    calls: list[tuple[str, dict]] = []
    builders = _builders(calls)
    context = TransportPreconditionerContext(op=_op(), active_size=128, use_active_dof_mode=False)
    precond = build_transport_preconditioner_from_kind(
        kind="fp_local_geom_line",
        context=context,
        builders=builders,
        dd_config=transport_dd_config_from_env(op=_op()),
        sparse_jax_config=TransportSparseJaxConfig(0.0, 1.0e-6, 1.0e-10, 0.8, 2, 128.0),
        use_reduced=False,
    )
    assert callable(precond)
    assert calls[0][0] == "fp_local_geom_line"


def test_build_transport_preconditioner_from_kind_dispatches_fp_structured_fblock_lu() -> None:
    calls: list[tuple[str, dict]] = []
    builders = _builders(calls)
    context = TransportPreconditionerContext(op=_op(), active_size=128, use_active_dof_mode=False)
    precond = build_transport_preconditioner_from_kind(
        kind="fp_structured_fblock_lu",
        context=context,
        builders=builders,
        dd_config=transport_dd_config_from_env(op=_op()),
        sparse_jax_config=TransportSparseJaxConfig(0.0, 1.0e-6, 1.0e-10, 0.8, 2, 128.0),
        use_reduced=False,
    )
    assert callable(precond)
    assert calls[0][0] == "fp_structured_fblock_lu"


def test_build_transport_preconditioner_from_kind_dispatches_fp_xblock_tz_lu() -> None:
    calls: list[tuple[str, dict]] = []
    builders = _builders(calls)
    context = TransportPreconditionerContext(op=_op(), active_size=128, use_active_dof_mode=False)
    precond = build_transport_preconditioner_from_kind(
        kind="fp_xblock_tz_lu",
        context=context,
        builders=builders,
        dd_config=transport_dd_config_from_env(op=_op()),
        sparse_jax_config=TransportSparseJaxConfig(0.0, 1.0e-6, 1.0e-10, 0.8, 2, 128.0),
        use_reduced=False,
    )
    assert callable(precond)
    assert calls[0][0] == "fp_xblock_tz_lu"


def test_build_transport_preconditioner_from_kind_dispatches_fp_xblock_tz_lu_schur() -> None:
    calls: list[tuple[str, dict]] = []
    builders = _builders(calls)
    context = TransportPreconditionerContext(op=_op(), active_size=128, use_active_dof_mode=False)
    precond = build_transport_preconditioner_from_kind(
        kind="fp_xblock_tz_lu_schur",
        context=context,
        builders=builders,
        dd_config=transport_dd_config_from_env(op=_op()),
        sparse_jax_config=TransportSparseJaxConfig(0.0, 1.0e-6, 1.0e-10, 0.8, 2, 128.0),
        use_reduced=False,
    )
    assert callable(precond)
    assert calls[0][0] == "fp_xblock_tz_lu_schur"


def test_build_transport_preconditioner_from_kind_dispatches_fp_direct_active_block_schur() -> None:
    calls: list[tuple[str, dict]] = []
    builders = _builders(calls)
    context = TransportPreconditionerContext(
        op=_op(),
        active_size=128,
        use_active_dof_mode=True,
        reduce_full=lambda x: x,
        expand_reduced=lambda x: x,
        active_indices_np=[0, 1, 2],
    )
    build_transport_preconditioner_from_kind(
        kind="fp_direct_active_block_schur",
        context=context,
        builders=builders,
        dd_config=transport_dd_config_from_env(op=_op()),
        sparse_jax_config=TransportSparseJaxConfig(0.0, 1.0e-6, 1.0e-10, 0.8, 2, 128.0),
        use_reduced=False,
    )
    precond = build_transport_preconditioner_from_kind(
        kind="fp_direct_active_block_schur",
        context=context,
        builders=builders,
        dd_config=transport_dd_config_from_env(op=_op()),
        sparse_jax_config=TransportSparseJaxConfig(0.0, 1.0e-6, 1.0e-10, 0.8, 2, 128.0),
        use_reduced=True,
    )
    assert callable(precond)
    assert calls[0][0] == "sxblock"
    assert calls[1][0] == "fp_direct_active_block_schur"
    assert calls[1][1]["active_indices_np"] == [0, 1, 2]


def test_build_transport_preconditioner_from_kind_falls_back_from_sparse_jax_on_memory_cap() -> None:
    calls: list[tuple[str, dict]] = []
    builders = _builders(calls)
    context = TransportPreconditionerContext(op=_op(total_size=300), active_size=300, use_active_dof_mode=False)
    precond = build_transport_preconditioner_from_kind(
        kind="sparse_jax",
        context=context,
        builders=builders,
        dd_config=transport_dd_config_from_env(op=_op(total_size=300)),
        sparse_jax_config=TransportSparseJaxConfig(0.0, 1.0e-6, 1.0e-10, 0.8, 2, 0.01),
        use_reduced=False,
    )
    assert callable(precond)
    assert calls[0][0] == "collision"


def test_build_transport_strong_preconditioner_from_kind_reuses_primary_when_same_kind() -> None:
    calls: list[tuple[str, dict]] = []
    builders = _builders(calls)
    context = TransportPreconditionerContext(op=_op(), active_size=128, use_active_dof_mode=False)
    primary = lambda v: v
    reused = build_transport_strong_preconditioner_from_kind(
        kind="block",
        use_reduced=False,
        precond_kind_used="block",
        preconditioner_full=primary,
        preconditioner_reduced=None,
        context=context,
        builders=builders,
        dd_config=transport_dd_config_from_env(op=_op()),
        sparse_jax_config=TransportSparseJaxConfig(0.0, 1.0e-6, 1.0e-10, 0.8, 2, 128.0),
    )
    assert reused is primary
    assert calls == []
