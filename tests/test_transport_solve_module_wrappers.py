from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp

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
