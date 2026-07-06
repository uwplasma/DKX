from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from sfincs_jax.solver import GMRESSolveResult
import sfincs_jax.problems.transport_solve as transport_solve


def test_transport_solve_distributed_axis_wrapper_uses_profile_system_policy(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_resolve_impl(**kwargs):
        captured.update(kwargs)
        return kwargs["matvec_shard_axis_fn"](kwargs["op"])

    monkeypatch.setattr(transport_solve, "_resolve_distributed_gmres_axis_impl", fake_resolve_impl)
    monkeypatch.setattr(transport_solve, "_matvec_shard_axis", lambda op: f"axis-{op.name}")
    op = SimpleNamespace(name="transport-fixture")

    assert transport_solve._resolve_distributed_gmres_axis(op=op, emit=None) == "axis-transport-fixture"
    assert captured["op"] is op
    assert captured["emit"] is None


def test_transport_solve_preconditioner_cache_key_injects_current_dtype(monkeypatch) -> None:
    op = SimpleNamespace(rhs_mode=2)
    captured: dict[str, object] = {}

    def fake_precond_dtype():
        return jnp.float32

    def fake_transport_impl(op_arg, kind, *, precond_dtype):
        captured["args"] = (op_arg, kind, precond_dtype)
        return ("transport", kind, str(precond_dtype))

    monkeypatch.setattr(transport_solve, "_precond_dtype", fake_precond_dtype)
    monkeypatch.setattr(transport_solve, "_transport_precond_cache_key_impl", fake_transport_impl)

    out = transport_solve._transport_precond_cache_key(op, "fp_tzfft")

    assert out == ("transport", "fp_tzfft", str(jnp.float32))
    assert captured["args"][0] is op


def test_transport_solve_fp_preconditioner_wrappers_inject_reusable_hooks(monkeypatch) -> None:
    captured: dict[str, dict[str, object]] = {}

    def fake_direct_builder(**kwargs):
        captured["direct"] = kwargs
        return "direct-pc"

    def fake_reduced_builder(**kwargs):
        captured["reduced"] = kwargs
        return "reduced-pc"

    monkeypatch.setattr(
        transport_solve,
        "build_transport_fp_direct_active_block_schur_preconditioner",
        fake_direct_builder,
    )
    monkeypatch.setattr(
        transport_solve,
        "build_transport_fp_fortran_reduced_lu_preconditioner",
        fake_reduced_builder,
    )
    op = SimpleNamespace(label="op")

    direct = transport_solve._build_rhsmode23_fp_direct_active_block_schur_preconditioner(
        op=op,
        active_indices_np=jnp.asarray([0, 2]),
        emit=None,
    )
    reduced = transport_solve._build_rhsmode23_fp_fortran_reduced_lu_preconditioner(
        op=op,
        active_indices_np=jnp.asarray([1, 3]),
        emit=None,
    )

    assert direct == "direct-pc"
    assert reduced == "reduced-pc"
    assert captured["direct"]["fallback_builder"] is transport_solve._build_rhsmode23_sxblock_preconditioner
    assert captured["direct"]["transport_precond_cache_key"] is transport_solve._transport_precond_cache_key
    assert captured["reduced"]["fallback_builder"] is transport_solve._build_rhsmode23_sxblock_preconditioner
    assert captured["reduced"]["transport_precond_cache_key"] is transport_solve._transport_precond_cache_key
    assert captured["reduced"]["build_host_sparse_direct_factor_from_matvec"] is (
        transport_solve._build_host_sparse_direct_factor_from_matvec
    )
    assert captured["reduced"]["host_physical_memory_mb"] is transport_solve._host_physical_memory_mb


def test_transport_parallel_worker_delegates_payload_with_public_solver(monkeypatch) -> None:
    captured: dict[str, object] = {}
    sentinel = {"ok": True}

    def fake_payload_solver(payload, **kwargs):
        captured["payload"] = payload
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(transport_solve, "_solve_transport_parallel_payload", fake_payload_solver)

    payload = {"input_path": "input.namelist", "which_rhs_values": [1, 2]}
    assert transport_solve._transport_parallel_worker(payload) is sentinel
    assert captured["payload"] is payload
    assert captured["read_input"] is transport_solve.read_sfincs_input
    assert captured["solve_transport"] is transport_solve.solve_v3_transport_matrix_linear_gmres


def test_transport_solve_top_level_can_exit_through_parallel_runtime(monkeypatch) -> None:
    captured: dict[str, object] = {}
    messages: list[tuple[int, str]] = []
    nml = SimpleNamespace(label="nml")
    op = SimpleNamespace(
        rhs_mode=2,
        total_size=11,
        include_phi1=False,
        fblock=SimpleNamespace(pas=object(), fp=None),
    )
    sentinel_result = SimpleNamespace(kind="parallel-result")

    monkeypatch.setattr(
        transport_solve,
        "resolve_transport_maxiter_setup",
        lambda _maxiter: SimpleNamespace(maxiter=17, notes=[(2, "maxiter-note")]),
    )
    monkeypatch.setattr(transport_solve, "full_system_operator_from_namelist", lambda **kwargs: op)
    monkeypatch.setattr(transport_solve, "_set_precond_size_hint", lambda size: captured.setdefault("size_hint", size))
    monkeypatch.setattr(
        transport_solve,
        "_set_precond_policy_hints",
        lambda **kwargs: captured.setdefault("policy_hints", kwargs),
    )
    monkeypatch.setattr(
        transport_solve,
        "resolve_transport_state_setup",
        lambda **kwargs: SimpleNamespace(
            state_in_path=None,
            state_out_path=None,
            x0=kwargs["x0"],
            x0_by_rhs=kwargs["x0_by_rhs"] or {},
            state_x_by_rhs={},
        ),
    )
    monkeypatch.setattr(
        transport_solve,
        "resolve_transport_which_rhs_setup",
        lambda **_kwargs: SimpleNamespace(
            rhs_mode=2,
            n_rhs=3,
            which_rhs_values=[1, 2, 3],
            subset_mode=False,
        ),
    )
    monkeypatch.setattr(transport_solve, "_transport_parallel_backend", lambda: "process")
    monkeypatch.setattr(transport_solve, "_transport_parallel_persistent_pool_enabled", lambda: False)
    monkeypatch.setattr(transport_solve, "_transport_parallel_visible_gpu_ids", lambda _workers: ["0"])
    monkeypatch.setattr(
        transport_solve,
        "resolve_transport_parallel_request",
        lambda **kwargs: SimpleNamespace(
            parallel_child=False,
            parallel_workers=2,
            parallel_backend=kwargs["parallel_backend"],
        ),
    )

    def fake_parallel_solve(**kwargs):
        captured["parallel_kwargs"] = kwargs
        return sentinel_result

    monkeypatch.setattr(transport_solve, "maybe_run_transport_parallel_solve", fake_parallel_solve)

    out = transport_solve.solve_v3_transport_matrix_linear_gmres(
        nml=nml,
        maxiter=23,
        restart=31,
        tol=1.0e-8,
        atol=1.0e-12,
        solve_method="auto",
        input_namelist=None,
        parallel_workers=2,
        collect_transport_output_fields=False,
        emit=lambda level, message: messages.append((int(level), str(message))),
    )

    assert out is sentinel_result
    assert messages[:2] == [(2, "maxiter-note"), (0, "solve_v3_transport_matrix_linear_gmres: starting whichRHS loop")]
    assert captured["size_hint"] == 11
    assert captured["policy_hints"] == {
        "has_pas": True,
        "has_fp": False,
        "include_phi1": False,
        "rhs_mode": 2,
    }
    parallel_kwargs = captured["parallel_kwargs"]
    assert parallel_kwargs["nml"] is nml
    assert parallel_kwargs["op0"] is op
    assert parallel_kwargs["rhs_mode"] == 2
    assert parallel_kwargs["which_rhs_values"] == [1, 2, 3]
    assert parallel_kwargs["parallel_workers"] == 2
    assert parallel_kwargs["parallel_backend"] == "process"
    assert parallel_kwargs["restart"] == 31
    assert parallel_kwargs["maxiter"] == 17
    assert parallel_kwargs["collect_transport_output_fields"] is False
    runtime = parallel_kwargs["runtime"]
    assert runtime.worker is transport_solve._transport_parallel_worker
    assert runtime.persistent_pool_enabled is False


def _install_transport_loop_harness(
    monkeypatch,
    *,
    total_size: int = 3,
    which_rhs_values: tuple[int, ...] = (1, 2),
    dense_retry_max: int = 0,
    solve_method_use: str = "auto",
    use_active_dof_mode: bool = False,
    active_indices: tuple[int, ...] | None = None,
    state_out_path: str = "",
    rhs3_flags: tuple[bool, bool] = (False, False),
) -> dict[str, object]:
    """Install a tiny diagonal transport problem for top-level loop tests."""

    captured: dict[str, object] = {"solve_calls": [], "messages": []}
    if active_indices is None:
        active_indices = tuple(range(int(total_size)))
    active_idx_np = jnp.asarray(active_indices, dtype=jnp.int32)
    full_to_active = [0] * int(total_size)
    for active_position, full_position in enumerate(active_indices, start=1):
        full_to_active[int(full_position)] = int(active_position)
    op = SimpleNamespace(
        rhs_mode=2,
        total_size=int(total_size),
        include_phi1=False,
        include_phi1_in_kinetic=False,
        fblock=SimpleNamespace(pas=None, fp=object()),
        n_species=1,
        n_x=1,
        n_xi=1,
        n_theta=1,
        n_zeta=1,
        constraint_scheme=2,
        quasineutrality_option=1,
        with_adiabatic=False,
        point_at_x0=False,
        phi1_size=0,
        extra_size=0,
        scale=2.0,
    )

    def fake_op_with_rhs(op_in, *, which_rhs):
        return SimpleNamespace(**{**op_in.__dict__, "which_rhs": int(which_rhs)})

    def fake_rhs(op_rhs):
        base = float(getattr(op_rhs, "which_rhs", 1))
        return jnp.asarray([base + i for i in range(int(total_size))], dtype=jnp.float64)

    class FakeCallbacks:
        def __init__(self, *, context):
            captured["linear_context"] = context

        def solve_with_residual(self, **kwargs):
            captured["solve_calls"].append(kwargs)
            x = 0.5 * kwargs["b_vec"]
            return GMRESSolveResult(x=x, residual_norm=jnp.asarray(0.0)), jnp.zeros_like(kwargs["b_vec"])

        def solve(self, **kwargs):
            captured["solve_calls"].append(kwargs)
            x = 0.5 * kwargs["b_vec"]
            return GMRESSolveResult(x=x, residual_norm=jnp.asarray(0.0))

    class FakeMatvecCache:
        def __init__(self, **_kwargs):
            pass

        def get_full(self, op_arg):
            captured.setdefault("full_matvec_ops", []).append(op_arg)
            return lambda x: op_arg.scale * x

        def get_reduced(self, op_arg):
            captured.setdefault("reduced_matvec_ops", []).append(op_arg)
            return lambda x: op_arg.scale * x

    monkeypatch.setattr(
        transport_solve,
        "resolve_transport_maxiter_setup",
        lambda maxiter: SimpleNamespace(maxiter=maxiter, notes=()),
    )
    monkeypatch.setattr(transport_solve, "full_system_operator_from_namelist", lambda **_kwargs: op)
    monkeypatch.setattr(transport_solve, "_set_precond_size_hint", lambda _size: None)
    monkeypatch.setattr(transport_solve, "_set_precond_policy_hints", lambda **_kwargs: None)
    monkeypatch.setattr(
        transport_solve,
        "resolve_transport_state_setup",
        lambda **kwargs: SimpleNamespace(
            state_in_path="",
            state_out_path=str(state_out_path),
            x0=kwargs["x0"],
            x0_by_rhs=kwargs["x0_by_rhs"] or {},
            state_x_by_rhs={},
        ),
    )
    monkeypatch.setattr(
        transport_solve,
        "resolve_transport_which_rhs_setup",
        lambda **_kwargs: SimpleNamespace(
            rhs_mode=2,
            n_rhs=len(which_rhs_values),
            which_rhs_values=list(which_rhs_values),
            subset_mode=False,
        ),
    )
    monkeypatch.setattr(
        transport_solve,
        "resolve_transport_parallel_request",
        lambda **kwargs: SimpleNamespace(
            parallel_child=False,
            parallel_workers=0,
            parallel_backend=kwargs["parallel_backend"],
        ),
    )
    monkeypatch.setattr(transport_solve, "maybe_run_transport_parallel_solve", lambda **_kwargs: None)
    monkeypatch.setattr(transport_solve, "_transport_parallel_backend", lambda: "process")
    monkeypatch.setattr(transport_solve, "_transport_parallel_visible_gpu_ids", lambda _workers=None: [])
    monkeypatch.setattr(transport_solve, "transport_geometry_scheme_from_namelist", lambda _nml: 2)
    monkeypatch.setattr(
        transport_solve,
        "resolve_transport_active_dense_setup",
        lambda **_kwargs: SimpleNamespace(
            initial_notes=(),
            active_notes=(),
            dense_notes=(),
            low_memory_outputs=False,
            stream_diagnostics=False,
            store_state_vectors=True,
            solve_method_use=str(solve_method_use),
            dense_retry_max=int(dense_retry_max),
            dense_mem_block=False,
            dense_use_mixed=False,
            dense_backend_allowed=False,
            gmres_restart=7,
            maxiter=11,
            use_active_dof_mode=bool(use_active_dof_mode),
            active_idx_np=np.asarray(active_indices, dtype=np.int32) if use_active_dof_mode else None,
            active_idx_jnp=active_idx_np if use_active_dof_mode else None,
            full_to_active_jnp=jnp.asarray(full_to_active, dtype=jnp.int32) if use_active_dof_mode else None,
            active_size=int(len(active_indices)) if use_active_dof_mode else int(total_size),
            dense_precond_enabled=False,
        ),
    )
    monkeypatch.setattr(transport_solve, "_resolve_use_implicit", lambda *, differentiable: False)
    monkeypatch.setattr(transport_solve, "_transport_precondition_side", lambda **_kwargs: "left")
    monkeypatch.setattr(transport_solve, "_resolve_distributed_gmres_axis", lambda **_kwargs: None)
    monkeypatch.setattr(transport_solve, "_use_solver_jit", lambda _size: False)
    monkeypatch.setattr(transport_solve, "_transport_sparse_direct_rescue_allowed", lambda **_kwargs: False)
    monkeypatch.setattr(transport_solve, "_transport_sparse_direct_first_attempt_allowed", lambda **_kwargs: False)
    monkeypatch.setattr(transport_solve, "normalize_transport_preconditioner_kind", lambda env_value: None)
    monkeypatch.setattr(transport_solve, "transport_sparse_jax_config_from_env", lambda: SimpleNamespace())
    monkeypatch.setattr(transport_solve, "transport_dd_config_from_env", lambda *, op: SimpleNamespace())
    monkeypatch.setattr(transport_solve, "grids_from_namelist", lambda _nml: SimpleNamespace())
    monkeypatch.setattr(transport_solve, "geometry_from_namelist", lambda **_kwargs: SimpleNamespace())
    monkeypatch.setattr(transport_solve, "with_transport_rhs_settings", fake_op_with_rhs)
    monkeypatch.setattr(transport_solve, "rhs_v3_full_system_jit", fake_rhs)
    monkeypatch.setattr(transport_solve, "transport_residual_gate_thresholds_from_env", lambda: (0.0, 0.0))
    monkeypatch.setattr(transport_solve, "_operator_signature_cached", lambda _op: ("mini-op",))
    monkeypatch.setattr(transport_solve, "apply_v3_full_system_operator_cached", lambda op_arg, x: op_arg.scale * x)
    monkeypatch.setattr(transport_solve, "resolve_transport_recycle_k", lambda **_kwargs: 0)
    monkeypatch.setattr(transport_solve, "TransportMatvecCache", FakeMatvecCache)
    monkeypatch.setattr(
        transport_solve,
        "resolve_transport_per_rhs_loop_policy",
        lambda **_kwargs: SimpleNamespace(
            dense_batch_fallback_enabled=False,
            iter_stats_enabled=False,
            iter_stats_max_size=None,
            rhs3_krylov_flags=lambda _which_rhs: tuple(rhs3_flags),
            projection_candidate=lambda _which_rhs: False,
            projection_needed=lambda _which_rhs: False,
        ),
    )
    monkeypatch.setattr(transport_solve, "TransportLinearSolveCallbacks", FakeCallbacks)
    monkeypatch.setattr(
        transport_solve,
        "_transport_sparse_direct_context_from_env",
        lambda **_kwargs: SimpleNamespace(
            solve=lambda **kwargs: GMRESSolveResult(x=0.5 * kwargs["b_vec"], residual_norm=jnp.asarray(0.0))
        ),
    )
    monkeypatch.setattr(
        transport_solve,
        "compute_transport_postsolve_diagnostics",
        lambda **kwargs: SimpleNamespace(
            transport_matrix=jnp.asarray([[len(kwargs["state_vectors"])]]),
            particle_flux_vm_psi_hat=jnp.ones((1, len(kwargs["which_rhs_values"]))),
            heat_flux_vm_psi_hat=2.0 * jnp.ones((1, len(kwargs["which_rhs_values"]))),
            fsab_flow=3.0 * jnp.ones((1, len(kwargs["which_rhs_values"]))),
            transport_output_fields=None,
        ),
    )
    return captured


def test_transport_solve_loop_falls_back_after_host_gmres_failure(monkeypatch) -> None:
    captured = _install_transport_loop_harness(monkeypatch, which_rhs_values=(1, 2))
    monkeypatch.setattr(transport_solve, "_transport_host_gmres_first_attempt_allowed", lambda **_kwargs: True)

    def fail_host_gmres(**_kwargs):
        raise RuntimeError("host gmres unavailable")

    monkeypatch.setattr(transport_solve, "_transport_host_gmres_solve", fail_host_gmres)

    result = transport_solve.solve_v3_transport_matrix_linear_gmres(
        nml=SimpleNamespace(),
        tol=1e-8,
        atol=1e-12,
        emit=lambda level, message: captured["messages"].append((int(level), str(message))),
    )

    assert result.transport_matrix.tolist() == [[2]]
    assert set(result.state_vectors_by_rhs) == {1, 2}
    assert len(captured["solve_calls"]) == 2
    assert any("host SciPy GMRES first attempt failed" in msg for _level, msg in captured["messages"])


def test_transport_solve_loop_uses_sparse_lu_first_attempt_and_contains_state_write_failure(monkeypatch) -> None:
    captured = _install_transport_loop_harness(
        monkeypatch,
        which_rhs_values=(1,),
        state_out_path="/tmp/sfincs_jax_unit_state.npz",
    )
    monkeypatch.setattr(transport_solve, "_transport_sparse_direct_first_attempt_allowed", lambda **_kwargs: True)

    import sfincs_jax.solvers.diagnostics as diagnostics

    monkeypatch.setattr(diagnostics, "save_krylov_state", lambda **_kwargs: (_ for _ in ()).throw(OSError("no write")))

    result = transport_solve.solve_v3_transport_matrix_linear_gmres(
        nml=SimpleNamespace(),
        tol=1e-8,
        atol=1e-12,
        emit=lambda level, message: captured["messages"].append((int(level), str(message))),
    )

    assert result.solver_kinds_by_rhs == {1: "sparse_lu"}
    assert result.solve_methods_by_rhs == {1: "sparse_lu"}
    assert not captured["solve_calls"]
    assert any("failed to write state" in msg for _level, msg in captured["messages"])


def test_transport_solve_loop_accepts_dense_true_residual_fallback(monkeypatch) -> None:
    captured = _install_transport_loop_harness(monkeypatch, which_rhs_values=(1,), dense_retry_max=8)

    class FailingCallbacks:
        def __init__(self, *, context):
            captured["linear_context"] = context

        def solve_with_residual(self, **kwargs):
            captured["solve_calls"].append(kwargs)
            return GMRESSolveResult(x=jnp.zeros_like(kwargs["b_vec"]), residual_norm=jnp.asarray(10.0)), kwargs["b_vec"]

        def solve(self, **kwargs):
            captured["solve_calls"].append(kwargs)
            return GMRESSolveResult(x=jnp.zeros_like(kwargs["b_vec"]), residual_norm=jnp.asarray(10.0))

    monkeypatch.setattr(transport_solve, "TransportLinearSolveCallbacks", FailingCallbacks)
    monkeypatch.setattr(transport_solve, "_dense_solver_for_matvec", lambda **_kwargs: (lambda rhs: 0.5 * rhs))

    result = transport_solve.solve_v3_transport_matrix_linear_gmres(
        nml=SimpleNamespace(),
        tol=1e-8,
        atol=1e-12,
        emit=lambda level, message: captured["messages"].append((int(level), str(message))),
    )

    assert result.solver_kinds_by_rhs == {1: "dense"}
    assert result.solve_methods_by_rhs == {1: "dense"}
    assert float(result.residual_norms_by_rhs[1]) == 0.0
    assert any("dense fallback" in msg for _level, msg in captured["messages"])


def test_transport_solve_loop_uses_active_dof_reduced_path(monkeypatch) -> None:
    captured = _install_transport_loop_harness(
        monkeypatch,
        total_size=4,
        which_rhs_values=(1,),
        use_active_dof_mode=True,
        active_indices=(0, 2),
    )

    result = transport_solve.solve_v3_transport_matrix_linear_gmres(
        nml=SimpleNamespace(),
        tol=1e-8,
        atol=1e-12,
        emit=lambda level, message: captured["messages"].append((int(level), str(message))),
    )

    assert result.use_active_dof_mode is True
    assert result.active_size == 2
    assert len(captured["solve_calls"]) == 1
    call = captured["solve_calls"][0]
    assert call["b_vec"].shape == (2,)
    assert call["solve_method_val"] == "auto"
    assert result.state_vectors_by_rhs[1].shape == (4,)
    assert result.state_vectors_by_rhs[1].tolist() == [0.5, 0.0, 1.5, 0.0]
    assert result.solver_kinds_by_rhs == {1: "bicgstab"}
    assert result.solve_methods_by_rhs == {1: "auto"}


def test_transport_solve_loop_falls_back_from_bicgstab_to_gmres(monkeypatch) -> None:
    captured = _install_transport_loop_harness(
        monkeypatch,
        which_rhs_values=(1,),
        solve_method_use="bicgstab",
    )

    class BicgstabThenGmresCallbacks:
        def __init__(self, *, context):
            captured["linear_context"] = context

        def solve_with_residual(self, **kwargs):
            captured["solve_calls"].append(kwargs)
            if kwargs["solve_method_val"] == "bicgstab":
                return (
                    GMRESSolveResult(
                        x=jnp.zeros_like(kwargs["b_vec"]),
                        residual_norm=jnp.asarray(99.0),
                    ),
                    kwargs["b_vec"],
                )
            x = 0.5 * kwargs["b_vec"]
            return GMRESSolveResult(x=x, residual_norm=jnp.asarray(0.0)), jnp.zeros_like(kwargs["b_vec"])

        def solve(self, **kwargs):
            captured["solve_calls"].append(kwargs)
            return GMRESSolveResult(
                x=jnp.zeros_like(kwargs["b_vec"]),
                residual_norm=jnp.asarray(99.0),
            )

    monkeypatch.setattr(transport_solve, "TransportLinearSolveCallbacks", BicgstabThenGmresCallbacks)

    result = transport_solve.solve_v3_transport_matrix_linear_gmres(
        nml=SimpleNamespace(),
        tol=1e-8,
        atol=1e-12,
        emit=lambda level, message: captured["messages"].append((int(level), str(message))),
    )

    assert [call["solve_method_val"] for call in captured["solve_calls"]] == ["bicgstab", "incremental"]
    assert result.solver_kinds_by_rhs == {1: "gmres"}
    assert result.solve_methods_by_rhs == {1: "incremental"}
    assert any("BiCGStab fallback to GMRES" in msg for _level, msg in captured["messages"])


@pytest.mark.parametrize(
    ("env_value", "expected_rhs_values"),
    [
        ("rhs", [1, 2]),
        ("base", [None, None]),
    ],
)
def test_transport_solve_loop_respects_matvec_mode_env(
    monkeypatch,
    env_value: str,
    expected_rhs_values: list[int | None],
) -> None:
    captured = _install_transport_loop_harness(monkeypatch, which_rhs_values=(1, 2))
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_MATVEC_MODE", env_value)

    transport_solve.solve_v3_transport_matrix_linear_gmres(
        nml=SimpleNamespace(),
        tol=1e-8,
        atol=1e-12,
        emit=lambda level, message: captured["messages"].append((int(level), str(message))),
    )

    observed = [getattr(op, "which_rhs", None) for op in captured["full_matvec_ops"][:2]]
    assert observed == expected_rhs_values


def test_transport_solve_loop_applies_rhs3_loose_epar_policy(monkeypatch) -> None:
    captured = _install_transport_loop_harness(
        monkeypatch,
        which_rhs_values=(1,),
        solve_method_use="bicgstab",
        rhs3_flags=(True, False),
    )
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_EPAR_TOL", "not-a-number")

    transport_solve.solve_v3_transport_matrix_linear_gmres(
        nml=SimpleNamespace(),
        tol=1e-12,
        atol=1e-14,
        emit=lambda level, message: captured["messages"].append((int(level), str(message))),
    )

    call = captured["solve_calls"][0]
    assert call["solve_method_val"] == "incremental"
    assert call["tol_val"] == pytest.approx(1e-8)


def test_transport_solve_loop_uses_sparse_direct_rescue_after_failed_krylov(monkeypatch) -> None:
    captured = _install_transport_loop_harness(monkeypatch, which_rhs_values=(1,))

    class FailingCallbacks:
        def __init__(self, *, context):
            captured["linear_context"] = context

        def solve_with_residual(self, **kwargs):
            captured["solve_calls"].append(kwargs)
            return GMRESSolveResult(x=jnp.zeros_like(kwargs["b_vec"]), residual_norm=jnp.asarray(50.0)), kwargs["b_vec"]

        def solve(self, **kwargs):
            captured["solve_calls"].append(kwargs)
            return GMRESSolveResult(x=jnp.zeros_like(kwargs["b_vec"]), residual_norm=jnp.asarray(50.0))

    monkeypatch.setattr(transport_solve, "TransportLinearSolveCallbacks", FailingCallbacks)
    monkeypatch.setattr(transport_solve, "_transport_sparse_direct_rescue_allowed", lambda **_kwargs: True)

    result = transport_solve.solve_v3_transport_matrix_linear_gmres(
        nml=SimpleNamespace(),
        tol=1e-8,
        atol=1e-12,
        emit=lambda level, message: captured["messages"].append((int(level), str(message))),
    )

    assert result.solver_kinds_by_rhs == {1: "sparse_lu"}
    assert result.solve_methods_by_rhs == {1: "sparse_lu"}
    assert any("sparse LU direct rescue" in msg for _level, msg in captured["messages"])


def test_transport_solve_loop_retries_with_strong_preconditioner(monkeypatch) -> None:
    def weak_precond(x):
        return x

    def strong_precond(x):
        return 0.5 * x

    captured = _install_transport_loop_harness(
        monkeypatch,
        which_rhs_values=(1,),
        solve_method_use="incremental",
    )

    monkeypatch.setattr(transport_solve, "normalize_transport_preconditioner_kind", lambda env_value: "block")
    monkeypatch.setattr(
        transport_solve,
        "resolve_transport_preconditioner_choice",
        lambda **_kwargs: ("block", "block"),
    )
    monkeypatch.setattr(transport_solve, "build_transport_preconditioner_from_kind", lambda **_kwargs: weak_precond)
    monkeypatch.setattr(transport_solve, "_transport_sparse_direct_rescue_allowed", lambda **_kwargs: False)

    class FakeStrongPreconditionerCache:
        def __init__(self, **_kwargs):
            pass

        def get(self, *, use_reduced: bool):
            assert use_reduced is False
            return strong_precond

    class StrongRetryCallbacks:
        def __init__(self, *, context):
            captured["linear_context"] = context

        def solve_with_residual(self, **kwargs):
            captured["solve_calls"].append(kwargs)
            residual = 0.0 if kwargs["preconditioner_val"] is strong_precond else 50.0
            x = 0.5 * kwargs["b_vec"] if residual == 0.0 else jnp.zeros_like(kwargs["b_vec"])
            return GMRESSolveResult(x=x, residual_norm=jnp.asarray(residual)), kwargs["b_vec"] - 2.0 * x

        def solve(self, **kwargs):
            captured["solve_calls"].append(kwargs)
            residual = 0.0 if kwargs["preconditioner_val"] is strong_precond else 50.0
            x = 0.5 * kwargs["b_vec"] if residual == 0.0 else jnp.zeros_like(kwargs["b_vec"])
            return GMRESSolveResult(x=x, residual_norm=jnp.asarray(residual))

    monkeypatch.setattr(transport_solve, "TransportStrongPreconditionerCache", FakeStrongPreconditionerCache)
    monkeypatch.setattr(transport_solve, "TransportLinearSolveCallbacks", StrongRetryCallbacks)

    result = transport_solve.solve_v3_transport_matrix_linear_gmres(
        nml=SimpleNamespace(),
        tol=1e-8,
        atol=1e-12,
        emit=lambda level, message: captured["messages"].append((int(level), str(message))),
    )

    assert [call["preconditioner_val"] for call in captured["solve_calls"]] == [weak_precond, None, strong_precond]
    assert result.solver_kinds_by_rhs == {1: "gmres"}
    assert result.solve_methods_by_rhs == {1: "incremental"}
    assert any("retry with strong preconditioner" in msg for _level, msg in captured["messages"])


def test_transport_solve_loop_tzfft_first_attempt_records_policy(monkeypatch) -> None:
    def precond(x):
        return x

    captured = _install_transport_loop_harness(monkeypatch, which_rhs_values=(1,))

    monkeypatch.setattr(transport_solve, "_transport_tzfft_structured_first_attempt_allowed", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(transport_solve, "_transport_tzfft_first_attempt_budget", lambda **_kwargs: ("incremental", 13, 17))
    monkeypatch.setattr(transport_solve, "normalize_transport_preconditioner_kind", lambda env_value: "tzfft")
    monkeypatch.setattr(
        transport_solve,
        "resolve_transport_preconditioner_choice",
        lambda **_kwargs: ("tzfft", "tzfft"),
    )
    monkeypatch.setattr(
        transport_solve,
        "resolve_transport_precondition_side_for_kind",
        lambda **_kwargs: ("left", True),
    )
    monkeypatch.setattr(transport_solve, "_transport_precondition_side", lambda **_kwargs: "right")
    monkeypatch.setattr(transport_solve, "build_transport_preconditioner_from_kind", lambda **_kwargs: precond)

    transport_solve.solve_v3_transport_matrix_linear_gmres(
        nml=SimpleNamespace(),
        tol=1e-8,
        atol=1e-12,
        emit=lambda level, message: captured["messages"].append((int(level), str(message))),
    )

    call = captured["solve_calls"][0]
    assert call["solve_method_val"] == "incremental"
    assert call["restart_val"] == 13
    assert call["maxiter_val"] == 17
    assert call["preconditioner_val"] is precond
    assert any("structured tzfft first attempt enabled" in msg for _level, msg in captured["messages"])
    assert any("uses left preconditioning" in msg for _level, msg in captured["messages"])
