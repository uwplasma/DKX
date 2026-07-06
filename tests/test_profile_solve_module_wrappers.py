from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp

import sfincs_jax.problems.profile_solve as profile_solve


def test_profile_solve_distributed_axis_wrapper_uses_profile_system_policy(monkeypatch) -> None:
    captured = {}

    def fake_resolve_impl(**kwargs):
        captured.update(kwargs)
        return kwargs["matvec_shard_axis_fn"](kwargs["op"])

    monkeypatch.setattr(profile_solve, "_resolve_distributed_gmres_axis_impl", fake_resolve_impl)
    monkeypatch.setattr(profile_solve, "_matvec_shard_axis", lambda op: f"axis-{op.name}")
    op = SimpleNamespace(name="fixture")

    assert profile_solve._resolve_distributed_gmres_axis(op=op, emit=None) == "axis-fixture"
    assert captured["op"] is op
    assert captured["emit"] is None


def test_profile_solve_transport_cache_key_wrapper_injects_current_preconditioner_dtype(monkeypatch) -> None:
    op = SimpleNamespace(rhs_mode=1)
    captured = {}

    def fake_precond_dtype():
        return jnp.float32

    def fake_transport_impl(op_arg, kind, *, precond_dtype):
        captured["transport"] = (op_arg, kind, precond_dtype)
        return ("transport", kind, str(precond_dtype))

    monkeypatch.setattr(profile_solve, "_precond_dtype", fake_precond_dtype)
    monkeypatch.setattr(profile_solve, "_transport_precond_cache_key_impl", fake_transport_impl)

    transport = profile_solve._transport_precond_cache_key(op, "collision")

    assert transport == ("transport", "collision", str(jnp.float32))
    assert captured["transport"][0] is op


def test_profile_solve_transport_fp_wrapper_injects_fallback_and_cache_hooks(monkeypatch) -> None:
    captured = {}

    def fake_direct_builder(**kwargs):
        captured["direct"] = kwargs
        return "direct-preconditioner"

    def fake_reduced_builder(**kwargs):
        captured["reduced"] = kwargs
        return "reduced-preconditioner"

    monkeypatch.setattr(
        profile_solve,
        "build_transport_fp_direct_active_block_schur_preconditioner",
        fake_direct_builder,
    )
    monkeypatch.setattr(
        profile_solve,
        "build_transport_fp_fortran_reduced_lu_preconditioner",
        fake_reduced_builder,
    )
    op = SimpleNamespace()

    direct = profile_solve._build_rhsmode23_fp_direct_active_block_schur_preconditioner(
        op=op,
        active_indices_np=jnp.asarray([0, 2]),
        emit=None,
    )
    reduced = profile_solve._build_rhsmode23_fp_fortran_reduced_lu_preconditioner(
        op=op,
        active_indices_np=jnp.asarray([1, 3]),
        emit=None,
    )

    assert direct == "direct-preconditioner"
    assert reduced == "reduced-preconditioner"
    assert captured["direct"]["fallback_builder"] is profile_solve._build_rhsmode23_sxblock_preconditioner
    assert captured["direct"]["transport_precond_cache_key"] is profile_solve._transport_precond_cache_key
    assert captured["reduced"]["fallback_builder"] is profile_solve._build_rhsmode23_sxblock_preconditioner
    assert captured["reduced"]["transport_precond_cache_key"] is profile_solve._transport_precond_cache_key
    assert captured["reduced"]["build_host_sparse_direct_factor_from_matvec"] is (
        profile_solve._build_host_sparse_direct_factor_from_matvec
    )
    assert captured["reduced"]["host_physical_memory_mb"] is profile_solve._host_physical_memory_mb


def test_profile_solve_dense_fallback_max_delegates_to_policy(monkeypatch) -> None:
    op = SimpleNamespace(total_size=123)
    monkeypatch.setattr(profile_solve, "_rhs1_dense_fallback_max_impl", lambda op_arg: op_arg.total_size + 5)

    assert profile_solve._rhsmode1_dense_fallback_max(op) == 128


def test_profile_solve_top_level_orchestrator_can_exit_through_auto_host(monkeypatch) -> None:
    """Lock the no-solve wiring boundary for the large RHSMode=1 orchestrator."""

    captured: dict[str, object] = {}
    nml = SimpleNamespace(label="nml")
    op = SimpleNamespace(
        rhs_mode=1,
        total_size=19,
        fblock=SimpleNamespace(collisionless=SimpleNamespace(n_xi_for_x=jnp.asarray([2, 2]))),
    )
    rhs = jnp.asarray([1.0, 0.0])
    x0 = jnp.asarray([0.25, -0.5])
    recycle_basis = [jnp.asarray([1.0, 0.0])]

    def fake_materialize(context):
        captured["materialize_context"] = context
        assert context.nml is nml
        assert context.op is op
        assert context.which_rhs == 2
        assert context.restart == 37
        assert context.maxiter == 41
        assert context.tol == 1.0e-8
        assert context.identity_shift == 0.125
        assert context.build_operator is profile_solve.full_system_operator_from_namelist
        assert context.rhs_builder is profile_solve.rhs_v3_full_system
        return SimpleNamespace(
            op=op,
            which_rhs=2,
            rhs=rhs,
            rhs_norm=jnp.asarray(1.0),
            tol=2.0e-8,
            fp_tol=3.0e-8,
            restart=23,
            maxiter=29,
            restart_env_forced=True,
            maxiter_env_forced=False,
            geom_scheme_hint=5,
        )

    def fake_route_setup(**kwargs):
        captured["route_setup"] = kwargs
        assert kwargs["nml"] is nml
        assert kwargs["op"] is op
        assert kwargs["solve_method"] == "auto"
        assert kwargs["use_implicit"] is False
        assert kwargs["sharded_axis"] is None
        method_flags = SimpleNamespace(
            kind="auto",
            sparse_host_like_requested=False,
            xblock_active_dof_requested=False,
            structured_full_csr_explicit_requested=False,
        )
        return SimpleNamespace(
            method_flags=method_flags,
            use_implicit_requested=False,
            structured_auto_allowed=False,
            structured_sharded_multidevice=False,
        )

    sentinel = SimpleNamespace(kind="auto-host-result")

    def fake_auto_host(context):
        captured["auto_host_context"] = context
        assert context.nml is nml
        assert context.which_rhs == 2
        assert context.op is op
        assert context.x0 is x0
        assert context.tol == 2.0e-8
        assert context.atol == 1.0e-12
        assert context.restart == 23
        assert context.maxiter == 29
        assert context.solve_method == "auto"
        assert context.identity_shift == 0.125
        assert context.differentiable is False
        assert context.recycle_basis is recycle_basis
        assert context.solve_driver is profile_solve.solve_v3_full_system_linear_gmres
        assert context.solve_method_kind_requested == "auto"
        assert context.structured_full_csr_explicit_requested is False
        assert context.use_implicit is False
        return sentinel

    monkeypatch.setattr(profile_solve, "maybe_profiler", lambda **_kwargs: None)
    monkeypatch.setattr(profile_solve, "materialize_profile_response_linear_problem", fake_materialize)
    monkeypatch.setattr(profile_solve, "resolve_rhs1_initial_route_setup", fake_route_setup)
    monkeypatch.setattr(profile_solve, "try_rhs1_auto_host_solve", fake_auto_host)
    monkeypatch.setattr(profile_solve, "_resolve_use_implicit", lambda *, differentiable: bool(differentiable))
    monkeypatch.setattr(profile_solve, "_matvec_shard_axis", lambda _op: None)
    monkeypatch.setattr(profile_solve.jax, "default_backend", lambda: "cpu")
    monkeypatch.setattr(profile_solve.jax, "device_count", lambda: 1)

    result = profile_solve.solve_v3_full_system_linear_gmres(
        nml=nml,
        which_rhs=2,
        op=op,
        x0=x0,
        tol=1.0e-8,
        atol=1.0e-12,
        restart=37,
        maxiter=41,
        solve_method="auto",
        identity_shift=0.125,
        differentiable=False,
        recycle_basis=recycle_basis,
    )

    assert result is sentinel
    assert set(captured) == {"materialize_context", "route_setup", "auto_host_context"}


def test_profile_solve_top_level_orchestrator_can_exit_through_structured_csr(monkeypatch) -> None:
    """Lock the explicit structured-CSR branch before expensive RHSMode=1 setup."""

    captured: dict[str, object] = {}
    nml = SimpleNamespace(label="nml")
    op = SimpleNamespace(
        rhs_mode=1,
        total_size=31,
        fblock=SimpleNamespace(collisionless=SimpleNamespace(n_xi_for_x=jnp.asarray([3, 3]))),
    )
    x0 = jnp.asarray([0.25, -0.5, 0.75])
    phi1_hat_base = jnp.asarray([[0.0]])
    recycle_basis = [jnp.asarray([1.0, 0.0, 0.0])]
    sentinel = SimpleNamespace(kind="structured-csr-result")

    def fake_materialize(context):
        captured["materialize_context"] = context
        return SimpleNamespace(
            op=op,
            which_rhs=1,
            rhs=jnp.asarray([1.0, 0.0, -1.0]),
            rhs_norm=jnp.asarray(2.0),
            tol=4.0e-9,
            fp_tol=5.0e-9,
            restart=44,
            maxiter=88,
            restart_env_forced=False,
            maxiter_env_forced=True,
            geom_scheme_hint=11,
        )

    def fake_route_setup(**kwargs):
        captured["route_setup"] = kwargs
        assert kwargs["solve_method"] == "structured_full_csr"
        assert kwargs["use_implicit"] is True
        method_flags = SimpleNamespace(
            kind="structured_full_csr",
            sparse_host_like_requested=False,
            xblock_active_dof_requested=False,
            structured_full_csr_explicit_requested=True,
        )
        return SimpleNamespace(
            method_flags=method_flags,
            use_implicit_requested=True,
            structured_auto_allowed=True,
            structured_sharded_multidevice=True,
        )

    def fake_structured(context):
        captured["structured_context"] = context
        assert context.nml is nml
        assert context.op is op
        assert context.x0 is x0
        assert float(context.rhs_norm) == 2.0
        assert context.tol == 4.0e-9
        assert context.atol == 7.0e-12
        assert context.restart == 44
        assert context.maxiter == 88
        assert context.solve_method == "structured_full_csr"
        assert context.identity_shift == 0.25
        assert context.phi1_hat_base is phi1_hat_base
        assert context.differentiable is True
        assert context.structured_solver is profile_solve.solve_v3_full_system_structured_csr
        return sentinel

    monkeypatch.setattr(profile_solve, "maybe_profiler", lambda **_kwargs: None)
    monkeypatch.setattr(profile_solve, "materialize_profile_response_linear_problem", fake_materialize)
    monkeypatch.setattr(profile_solve, "resolve_rhs1_initial_route_setup", fake_route_setup)
    monkeypatch.setattr(profile_solve, "try_rhs1_auto_host_solve", lambda _context: None)
    monkeypatch.setattr(profile_solve, "solve_rhs1_structured_full_csr_explicit", fake_structured)
    monkeypatch.setattr(profile_solve, "_resolve_use_implicit", lambda *, differentiable: bool(differentiable))
    monkeypatch.setattr(profile_solve, "_matvec_shard_axis", lambda _op: "theta")
    monkeypatch.setattr(profile_solve.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(profile_solve.jax, "device_count", lambda: 2)

    result = profile_solve.solve_v3_full_system_linear_gmres(
        nml=nml,
        which_rhs=1,
        op=op,
        x0=x0,
        tol=1.0e-8,
        atol=7.0e-12,
        restart=40,
        maxiter=80,
        solve_method="structured_full_csr",
        identity_shift=0.25,
        phi1_hat_base=phi1_hat_base,
        differentiable=True,
        recycle_basis=recycle_basis,
    )

    assert result is sentinel
    assert set(captured) == {"materialize_context", "route_setup", "structured_context"}
    assert captured["route_setup"]["sharded_axis"] == "theta"
    assert captured["route_setup"]["backend"] == "gpu"
    assert captured["route_setup"]["device_count"] == 2


def _install_profile_solve_sparse_branch_scaffold(
    monkeypatch,
    *,
    solve_method: str,
    use_active_dof_mode: bool = False,
    use_pas_projection: bool = False,
    profiler_marks: list[str] | None = None,
) -> tuple[SimpleNamespace, dict]:
    captured: dict[str, object] = {}
    nml = SimpleNamespace(label="nml", group=lambda _name: {})
    op = SimpleNamespace(
        rhs_mode=1,
        total_size=8,
        n_xi=2,
        fblock=SimpleNamespace(collisionless=SimpleNamespace(n_xi_for_x=jnp.asarray([2, 2]))),
    )
    rhs = jnp.asarray([1.0, 0.0], dtype=jnp.float64)

    def fake_materialize(context):
        captured["materialize_context"] = context
        context.mark("materialized")
        return SimpleNamespace(
            op=op,
            which_rhs=1,
            rhs=rhs,
            rhs_norm=jnp.asarray(1.0, dtype=jnp.float64),
            tol=1.0e-8,
            fp_tol=2.0e-8,
            restart=17,
            maxiter=23,
            restart_env_forced=False,
            maxiter_env_forced=False,
            geom_scheme_hint=1,
        )

    def fake_route_setup(**kwargs):
        captured["route_setup"] = kwargs
        return SimpleNamespace(
            method_flags=SimpleNamespace(
                kind=solve_method,
                sparse_host_like_requested=True,
                xblock_active_dof_requested=False,
                structured_full_csr_explicit_requested=False,
            ),
            use_implicit_requested=False,
            structured_auto_allowed=False,
            structured_sharded_multidevice=False,
        )

    if profiler_marks is None:
        monkeypatch.setattr(profile_solve, "maybe_profiler", lambda **_kwargs: None)
    else:
        monkeypatch.setattr(
            profile_solve,
            "maybe_profiler",
            lambda **_kwargs: SimpleNamespace(mark=lambda label: profiler_marks.append(label)),
        )
    monkeypatch.setattr(profile_solve, "materialize_profile_response_linear_problem", fake_materialize)
    monkeypatch.setattr(profile_solve, "resolve_rhs1_initial_route_setup", fake_route_setup)
    monkeypatch.setattr(profile_solve, "try_rhs1_auto_host_solve", lambda _context: None)
    monkeypatch.setattr(profile_solve, "resolve_rhs1_recycle_basis_setup", lambda **_kwargs: SimpleNamespace(basis=()))
    monkeypatch.setattr(
        profile_solve,
        "resolve_rhs1_reduced_mode_shape_setup",
        lambda **_kwargs: SimpleNamespace(
            nxi_for_x=jnp.asarray([2, 2]),
            max_l=1,
            has_reduced_modes=False,
        ),
    )
    monkeypatch.setattr(
        profile_solve,
        "resolve_rhs1_active_problem_setup",
        lambda **_kwargs: SimpleNamespace(
            tol=1.0e-8,
            restart=19,
            maxiter=29,
            use_dkes=False,
            include_xdot_sparse_pc=False,
            include_electric_field_xi_sparse_pc=False,
            er_abs_sparse_pc=0.0,
            preconditioner_species=1,
            preconditioner_x=1,
            preconditioner_x_min_l=0,
            preconditioner_xi=1,
            full_preconditioner_requested=False,
            geom_scheme=1,
            use_pas_projection=bool(use_pas_projection),
            use_active_dof_mode=bool(use_active_dof_mode),
            active_idx_jnp=jnp.asarray([0, 1], dtype=jnp.int32) if use_active_dof_mode else None,
            full_to_active_jnp=jnp.asarray([0, 1], dtype=jnp.int32) if use_active_dof_mode else None,
            active_size=2 if use_active_dof_mode else 8,
            messages=((1, "active setup"),),
        ),
    )
    monkeypatch.setattr(
        profile_solve,
        "resolve_rhs1_post_active_solve_policy_setup",
        lambda **_kwargs: SimpleNamespace(
            restart=31,
            maxiter=37,
            solve_method=solve_method,
            tokamak_pas=False,
            pas_large_bicgstab_fastpath=False,
            pas_large_fastpath_min=0,
            messages=((1, "post-active setup"),),
        ),
    )
    monkeypatch.setattr(profile_solve, "try_rhs1_sparse_host_safe_solve", lambda _context: None)
    monkeypatch.setattr(profile_solve, "try_run_requested_sparse_pc_gmres_branch", lambda _context: None)
    monkeypatch.setattr(profile_solve, "_resolve_use_implicit", lambda *, differentiable: bool(differentiable))
    monkeypatch.setattr(profile_solve, "_matvec_shard_axis", lambda _op: None)
    monkeypatch.setattr(profile_solve.jax, "default_backend", lambda: "cpu")
    monkeypatch.setattr(profile_solve.jax, "device_count", lambda: 1)
    return nml, captured


def test_profile_solve_top_level_orchestrator_can_exit_through_sparse_minimum_norm(monkeypatch) -> None:
    nml, captured = _install_profile_solve_sparse_branch_scaffold(monkeypatch, solve_method="sparse_minimum_norm")
    sentinel_payload = SimpleNamespace(kind="minimum-norm-payload")
    sentinel_result = SimpleNamespace(kind="minimum-norm-result")

    def fake_minimum_norm(context):
        captured["minimum_norm_context"] = context
        assert context.solve_method_kind == "sparse_minimum_norm"
        assert context.use_active_dof is False
        assert context.tol == 1.0e-8
        assert context.maxiter == 37
        assert context.backend == "cpu"
        assert context.build_operator_from_pattern is profile_solve.build_operator_from_pattern
        return sentinel_payload

    def fake_result(**kwargs):
        captured["result_payload"] = kwargs
        assert kwargs["payload"] is sentinel_payload
        return sentinel_result

    monkeypatch.setattr(profile_solve, "solve_explicit_sparse_minimum_norm_branch", fake_minimum_norm)
    monkeypatch.setattr(profile_solve, "v3_linear_solve_result_from_payload", fake_result)

    result = profile_solve.solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="sparse_minimum_norm",
        differentiable=False,
        atol=1.0e-12,
    )

    assert result is sentinel_result
    assert "minimum_norm_context" in captured
    assert "result_payload" in captured


def test_profile_solve_top_level_orchestrator_can_exit_through_sparse_host_safe(monkeypatch) -> None:
    nml, captured = _install_profile_solve_sparse_branch_scaffold(monkeypatch, solve_method="sparse_host_safe")
    sentinel = SimpleNamespace(kind="sparse-host-safe-result")

    def fake_sparse_host_safe(context):
        captured["sparse_host_safe_context"] = context
        assert context.nml is nml
        assert context.which_rhs == 1
        assert context.tol == 1.0e-8
        assert context.atol == 1.0e-12
        assert context.restart == 31
        assert context.maxiter == 37
        assert context.solve_driver is profile_solve.solve_v3_full_system_linear_gmres
        assert context.solve_method_kind_explicit == "sparse_host_safe"
        assert context.requested is True
        return sentinel

    monkeypatch.setattr(profile_solve, "try_rhs1_sparse_host_safe_solve", fake_sparse_host_safe)

    result = profile_solve.solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="sparse_host_safe",
        differentiable=False,
        atol=1.0e-12,
    )

    assert result is sentinel
    assert "sparse_host_safe_context" in captured
    assert "minimum_norm_context" not in captured


def test_profile_solve_sparse_host_safe_progress_and_profiler_are_forwarded(monkeypatch) -> None:
    messages: list[tuple[int, str]] = []
    profiler_marks: list[str] = []
    nml, captured = _install_profile_solve_sparse_branch_scaffold(
        monkeypatch,
        solve_method="sparse_host_safe",
        use_active_dof_mode=True,
        use_pas_projection=True,
        profiler_marks=profiler_marks,
    )
    sentinel = SimpleNamespace(kind="sparse-host-safe-result")

    def fake_sparse_host_safe(context):
        captured["sparse_host_safe_context"] = context
        assert context.requested is True
        assert context.restart == 31
        assert context.maxiter == 37
        return sentinel

    monkeypatch.setattr(profile_solve, "try_rhs1_sparse_host_safe_solve", fake_sparse_host_safe)

    result = profile_solve.solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="sparse_host_safe",
        differentiable=False,
        atol=1.0e-12,
        emit=lambda level, message: messages.append((int(level), str(message))),
    )

    assert result is sentinel
    assert profiler_marks == ["materialized"]
    emitted = [message for _, message in messages]
    assert "active setup" in emitted
    assert "post-active setup" in emitted
    assert any("PAS constraint projection enabled" in message for message in emitted)
    assert any("GMRES tol=1e-08 atol=1e-12 restart=31 maxiter=37" in message for message in emitted)
    assert any("evaluateJacobian called" in message for message in emitted)


def test_profile_solve_top_level_orchestrator_can_exit_through_sparse_pc_replay(monkeypatch) -> None:
    nml, captured = _install_profile_solve_sparse_branch_scaffold(monkeypatch, solve_method="sparse_pc_gmres")
    sentinel = SimpleNamespace(kind="sparse-pc-result")

    def fake_sparse_pc(context):
        captured["sparse_pc_context"] = context
        assert context.values["nml"] is nml
        assert context.values["solve_method_kind_explicit"] == "sparse_pc_gmres"
        assert context.values["tol"] == 1.0e-8
        assert context.values["restart"] == 31
        assert context.values["maxiter"] == 37
        assert context.values["active_size"] == 8
        return sentinel

    monkeypatch.setattr(profile_solve, "try_run_requested_sparse_pc_gmres_branch", fake_sparse_pc)

    result = profile_solve.solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="sparse_pc_gmres",
        differentiable=False,
        atol=1.0e-12,
    )

    assert result is sentinel
    assert "sparse_pc_context" in captured
    assert "minimum_norm_context" not in captured


def test_profile_solve_top_level_orchestrator_can_exit_through_sparse_host_direct(monkeypatch) -> None:
    nml, captured = _install_profile_solve_sparse_branch_scaffold(monkeypatch, solve_method="sparse_host")
    sentinel_payload = SimpleNamespace(kind="host-direct-payload")
    sentinel_result = SimpleNamespace(kind="host-direct-result")

    def fake_sparse_direct(context):
        captured["sparse_direct_context"] = context
        assert context.use_active_dof is False
        assert context.tol == 1.0e-8
        assert context.atol == 1.0e-12
        assert context.refine_steps >= 0
        assert context.build_host_sparse_direct_factor_from_matvec is profile_solve._build_host_sparse_direct_factor_from_matvec
        return sentinel_payload

    def fake_result(**kwargs):
        captured["result_payload"] = kwargs
        assert kwargs["payload"] is sentinel_payload
        return sentinel_result

    monkeypatch.setattr(profile_solve, "solve_explicit_sparse_host_direct_branch", fake_sparse_direct)
    monkeypatch.setattr(profile_solve, "v3_linear_solve_result_from_payload", fake_result)

    result = profile_solve.solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="sparse_host",
        differentiable=False,
        atol=1.0e-12,
    )

    assert result is sentinel_result
    assert "sparse_direct_context" in captured
    assert "result_payload" in captured
