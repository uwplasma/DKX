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
    monkeypatch.setattr(profile_solve, "_resolve_use_implicit", lambda *, differentiable: bool(differentiable))
    monkeypatch.setattr(profile_solve, "_matvec_shard_axis", lambda _op: None)
    monkeypatch.setattr(profile_solve.jax, "default_backend", lambda: "cpu")
    monkeypatch.setattr(profile_solve.jax, "device_count", lambda: 1)
    return nml, captured


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


