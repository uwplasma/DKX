from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp

import sfincs_jax.problems.profile_solve as profile_solve


def test_profile_solve_gmres_dispatch_wires_module_solver_functions(monkeypatch) -> None:
    captured = {}

    def fake_dispatch_impl(**kwargs):
        captured.update(kwargs)
        return "gmres-result"

    monkeypatch.setattr(profile_solve, "_gmres_solve_dispatch_impl", fake_dispatch_impl)

    out = profile_solve._gmres_solve_dispatch(
        distributed_axis="theta",
        size_hint=17,
        solve_method="incremental",
        rhs=jnp.ones(2),
    )

    assert out == "gmres-result"
    assert captured["distributed_axis"] == "theta"
    assert captured["size_hint"] == 17
    assert captured["gmres_solve_fn"] is profile_solve.gmres_solve
    assert captured["gmres_solve_jit_fn"] is profile_solve.gmres_solve_jit
    assert captured["gmres_solve_distributed_fn"] is profile_solve.gmres_solve_distributed
    assert captured["distributed_gmres_enabled_fn"] is profile_solve.distributed_gmres_enabled
    assert captured["use_solver_jit_fn"] is profile_solve._use_solver_jit
    assert captured["solve_method"] == "incremental"


def test_profile_solve_gmres_residual_dispatch_wires_module_solver_functions(monkeypatch) -> None:
    captured = {}

    def fake_dispatch_impl(**kwargs):
        captured.update(kwargs)
        return "gmres-residual-result"

    monkeypatch.setattr(profile_solve, "_gmres_solve_with_residual_dispatch_impl", fake_dispatch_impl)

    out = profile_solve._gmres_solve_with_residual_dispatch(
        distributed_axis="zeta",
        size_hint=19,
        solve_method="incremental",
    )

    assert out == "gmres-residual-result"
    assert captured["distributed_axis"] == "zeta"
    assert captured["size_hint"] == 19
    assert captured["gmres_solve_with_residual_fn"] is profile_solve.gmres_solve_with_residual
    assert captured["gmres_solve_with_residual_jit_fn"] is profile_solve.gmres_solve_with_residual_jit
    assert captured["gmres_solve_with_residual_distributed_fn"] is profile_solve.gmres_solve_with_residual_distributed
    assert captured["distributed_gmres_enabled_fn"] is profile_solve.distributed_gmres_enabled
    assert captured["use_solver_jit_fn"] is profile_solve._use_solver_jit


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


def test_profile_solve_cache_key_wrappers_inject_current_preconditioner_dtype(monkeypatch) -> None:
    op = SimpleNamespace(rhs_mode=1)
    captured = {}

    def fake_precond_dtype():
        return jnp.float32

    def fake_structured_impl(op_arg, kind, *, precond_dtype, params):
        captured["structured"] = (op_arg, kind, precond_dtype, params)
        return ("structured", kind, str(precond_dtype), params)

    def fake_transport_impl(op_arg, kind, *, precond_dtype):
        captured["transport"] = (op_arg, kind, precond_dtype)
        return ("transport", kind, str(precond_dtype))

    monkeypatch.setattr(profile_solve, "_precond_dtype", fake_precond_dtype)
    monkeypatch.setattr(profile_solve, "_rhs_mode1_structured_fblock_cache_key_impl", fake_structured_impl)
    monkeypatch.setattr(profile_solve, "_transport_precond_cache_key_impl", fake_transport_impl)

    structured = profile_solve._rhsmode1_structured_fblock_cache_key(op, "xblock", params=("a", 2))
    transport = profile_solve._transport_precond_cache_key(op, "collision")

    assert structured == ("structured", "xblock", str(jnp.float32), ("a", 2))
    assert transport == ("transport", "collision", str(jnp.float32))
    assert captured["structured"][0] is op
    assert captured["transport"][0] is op


def test_profile_solve_schur_wrapper_uses_monkeypatchable_builders(monkeypatch) -> None:
    captured = {}
    theta_line = object()
    block = object()

    def fake_build_schur(**kwargs):
        captured.update(kwargs)
        return "schur-preconditioner"

    monkeypatch.setattr(profile_solve, "build_rhs1_schur_preconditioner", fake_build_schur)
    monkeypatch.setattr(profile_solve, "_build_rhsmode1_theta_line_preconditioner", theta_line)
    monkeypatch.setattr(profile_solve, "_build_rhsmode1_block_preconditioner", block)
    op = SimpleNamespace()

    out = profile_solve._build_rhsmode1_schur_preconditioner(
        op=op,
        reduce_full=lambda x: x,
        expand_reduced=lambda x: x,
    )

    assert out == "schur-preconditioner"
    assert captured["op"] is op
    assert captured["builders"].theta_line_builder is theta_line
    assert captured["builders"].block_builder is block


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
