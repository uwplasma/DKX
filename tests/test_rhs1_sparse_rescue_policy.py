from __future__ import annotations

from sfincs_jax.rhs1_sparse_rescue_policy import (
    rhs1_resolved_sparse_rescue_ordering,
    rhs1_sparse_jax_config_from_env,
    rhs1_sparse_preconditioner_config_from_env,
    rhs1_sparse_rescue_initial_messages,
    rhs1_sparse_rescue_policy_setup,
    rhs1_sparse_rescue_tail_skip_messages,
    rhs1_sparse_enabled_initial,
    rhs1_sparse_kind_use,
)


_SPARSE_PRECONDITIONER_ENV = (
    "SFINCS_JAX_RHSMODE1_SPARSE_PRECOND",
    "SFINCS_JAX_RHSMODE1_SPARSE_ALLOW_NONDIFF",
    "SFINCS_JAX_RHSMODE1_SPARSE_MATVEC",
    "SFINCS_JAX_RHSMODE1_SPARSE_OPERATOR",
    "SFINCS_JAX_RHSMODE1_SPARSE_MAX",
    "SFINCS_JAX_RHSMODE1_PAS_SPARSE_ILU_MIN",
    "SFINCS_JAX_RHSMODE1_SPARSE_DROP_TOL",
    "SFINCS_JAX_RHSMODE1_SPARSE_DROP_REL",
    "SFINCS_JAX_RHSMODE1_SPARSE_ILU_DROP_TOL",
    "SFINCS_JAX_RHSMODE1_SPARSE_ILU_FILL_FACTOR",
    "SFINCS_JAX_RHSMODE1_SPARSE_ILU_DENSE_MAX",
    "SFINCS_JAX_RHSMODE1_SPARSE_DENSE_CACHE_MAX",
)


def _clear_sparse_preconditioner_env(monkeypatch) -> None:
    for name in _SPARSE_PRECONDITIONER_ENV:
        monkeypatch.delenv(name, raising=False)


def test_rhs1_sparse_enabled_initial_follows_mode_and_rhsmode_guards() -> None:
    assert rhs1_sparse_enabled_initial(
        sparse_precond_mode="on",
        has_fp=False,
        has_pas=False,
        residual_norm=0.0,
        target=1.0,
        rhs_mode=1,
        include_phi1=False,
    )
    assert rhs1_sparse_enabled_initial(
        sparse_precond_mode="auto",
        has_fp=True,
        has_pas=False,
        residual_norm=0.0,
        target=1.0,
        rhs_mode=1,
        include_phi1=False,
    )
    assert not rhs1_sparse_enabled_initial(
        sparse_precond_mode="auto",
        has_fp=False,
        has_pas=True,
        residual_norm=1.0e-8,
        target=1.0e-6,
        rhs_mode=1,
        include_phi1=False,
    )
    assert not rhs1_sparse_enabled_initial(
        sparse_precond_mode="on",
        has_fp=True,
        has_pas=False,
        residual_norm=1.0,
        target=1.0e-6,
        rhs_mode=2,
        include_phi1=False,
    )
    assert not rhs1_sparse_enabled_initial(
        sparse_precond_mode="on",
        has_fp=True,
        has_pas=False,
        residual_norm=1.0,
        target=1.0e-6,
        rhs_mode=1,
        include_phi1=True,
    )


def test_rhs1_sparse_kind_use_normalizes_auto_to_scipy() -> None:
    assert rhs1_sparse_kind_use(sparse_precond_kind="auto") == "scipy"
    assert rhs1_sparse_kind_use(sparse_precond_kind="jax") == "jax"


def test_rhs1_sparse_preconditioner_config_from_env_uses_legacy_defaults(
    monkeypatch,
) -> None:
    _clear_sparse_preconditioner_env(monkeypatch)

    config = rhs1_sparse_preconditioner_config_from_env(
        has_pas=False,
        use_dkes=False,
        active_size=3000,
        backend="cpu",
    )
    assert config.precond_mode == "auto"
    assert config.precond_kind == "auto"
    assert not config.allow_nondiff
    assert not config.use_matvec
    assert config.operator_mode == "auto"
    assert config.max_size == 6000
    assert config.pas_sparse_min == 2000
    assert config.drop_tol == 0.0
    assert config.drop_rel == 1.0e-8
    assert config.ilu_drop_tol == 1.0e-4
    assert config.ilu_fill == 10.0
    assert config.ilu_dense_max == 2500
    assert config.dense_cache_max == 3000


def test_rhs1_sparse_preconditioner_config_from_env_parses_aliases_and_values(
    monkeypatch,
) -> None:
    _clear_sparse_preconditioner_env(monkeypatch)
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PRECOND", "jax_native")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_ALLOW_NONDIFF", "yes")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_MATVEC", "on")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_OPERATOR", "off")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_MAX", "123")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_SPARSE_ILU_MIN", "17")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_DROP_TOL", "1e-3")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_DROP_REL", "2e-3")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_ILU_DROP_TOL", "3e-3")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_ILU_FILL_FACTOR", "4")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_ILU_DENSE_MAX", "55")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_DENSE_CACHE_MAX", "66")

    config = rhs1_sparse_preconditioner_config_from_env(
        has_pas=False,
        use_dkes=False,
        active_size=3000,
        backend="gpu",
    )
    assert config.precond_mode == "on"
    assert config.precond_kind == "jax"
    assert config.allow_nondiff
    assert config.use_matvec
    assert config.operator_mode == "off"
    assert config.max_size == 123
    assert config.pas_sparse_min == 17
    assert config.drop_tol == 1.0e-3
    assert config.drop_rel == 2.0e-3
    assert config.ilu_drop_tol == 3.0e-3
    assert config.ilu_fill == 4.0
    assert config.ilu_dense_max == 55
    assert config.dense_cache_max == 66


def test_rhs1_sparse_preconditioner_config_from_env_handles_pas_dkes_and_invalid(
    monkeypatch,
) -> None:
    _clear_sparse_preconditioner_env(monkeypatch)
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PRECOND", "on")
    for name in (
        "SFINCS_JAX_RHSMODE1_SPARSE_MAX",
        "SFINCS_JAX_RHSMODE1_PAS_SPARSE_ILU_MIN",
        "SFINCS_JAX_RHSMODE1_SPARSE_DROP_TOL",
        "SFINCS_JAX_RHSMODE1_SPARSE_DROP_REL",
        "SFINCS_JAX_RHSMODE1_SPARSE_ILU_DROP_TOL",
        "SFINCS_JAX_RHSMODE1_SPARSE_ILU_FILL_FACTOR",
        "SFINCS_JAX_RHSMODE1_SPARSE_ILU_DENSE_MAX",
        "SFINCS_JAX_RHSMODE1_SPARSE_DENSE_CACHE_MAX",
    ):
        monkeypatch.setenv(name, "bad")

    config = rhs1_sparse_preconditioner_config_from_env(
        has_pas=True,
        use_dkes=True,
        active_size=1999,
        backend="gpu",
    )
    assert config.precond_mode == "off"
    assert config.precond_kind == "auto"
    assert config.max_size == 60000
    assert config.pas_sparse_min == 2000
    assert config.drop_tol == 0.0
    assert config.drop_rel == 1.0e-8
    assert config.ilu_drop_tol == 1.0e-4
    assert config.ilu_fill == 10.0
    assert config.ilu_dense_max == 3000
    assert config.dense_cache_max == 3000


def test_rhs1_sparse_jax_config_from_env_uses_stable_defaults(monkeypatch) -> None:
    for name in (
        "SFINCS_JAX_RHSMODE1_SPARSE_JAX_MAX_MB",
        "SFINCS_JAX_RHSMODE1_SPARSE_JAX_SWEEPS",
        "SFINCS_JAX_RHSMODE1_SPARSE_JAX_OMEGA",
        "SFINCS_JAX_RHSMODE1_SPARSE_JAX_REG",
    ):
        monkeypatch.delenv(name, raising=False)

    config = rhs1_sparse_jax_config_from_env()
    assert config.max_mb == 128.0
    assert config.sweeps == 2
    assert config.omega == 0.8
    assert config.reg == 1.0e-10


def test_rhs1_sparse_jax_config_from_env_parses_values_and_bounds_sweeps(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_JAX_MAX_MB", "64")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_JAX_SWEEPS", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_JAX_OMEGA", "0.65")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_JAX_REG", "1e-8")

    config = rhs1_sparse_jax_config_from_env()
    assert config.max_mb == 64.0
    assert config.sweeps == 1
    assert config.omega == 0.65
    assert config.reg == 1.0e-8


def test_rhs1_sparse_jax_config_from_env_ignores_invalid_values(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_JAX_MAX_MB", "bad")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_JAX_SWEEPS", "bad")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_JAX_OMEGA", "bad")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_JAX_REG", "bad")

    config = rhs1_sparse_jax_config_from_env()
    assert config.max_mb == 128.0
    assert config.sweeps == 2
    assert config.omega == 0.8
    assert config.reg == 1.0e-10


def test_rhs1_sparse_rescue_ordering_handles_dense_shortcut_and_exact_preference() -> None:
    decision = rhs1_resolved_sparse_rescue_ordering(
        sparse_enabled=True,
        sparse_kind_use="auto",
        dense_shortcut=True,
        sparse_exact_direct=False,
        size=1000,
        sparse_max_size=2000,
    )
    assert not decision.enabled
    assert decision.reason_dense_shortcut_skip

    decision = rhs1_resolved_sparse_rescue_ordering(
        sparse_enabled=True,
        sparse_kind_use="auto",
        dense_shortcut=True,
        sparse_exact_direct=True,
        size=1000,
        sparse_max_size=2000,
    )
    assert decision.enabled
    assert decision.prefer_sparse_exact_over_dense_shortcut
    assert decision.kind_use == "scipy"


def test_rhs1_sparse_rescue_ordering_handles_size_routing_and_targeted_disable() -> None:
    decision = rhs1_resolved_sparse_rescue_ordering(
        sparse_enabled=True,
        sparse_kind_use="auto",
        size=5000,
        sparse_max_size=2000,
        large_cpu_sparse_rescue=True,
        sparse_exact_direct=True,
        sparse_xblock_rescue_active=True,
        sparse_sxblock_rescue_active=True,
    )
    assert decision.enabled
    assert decision.reason_size_large_cpu
    assert decision.reason_large_cpu_exact_skips_targeted
    assert not decision.xblock_rescue_active
    assert not decision.sxblock_rescue_active

    decision = rhs1_resolved_sparse_rescue_ordering(
        sparse_enabled=True,
        sparse_kind_use="auto",
        size=5000,
        sparse_max_size=2000,
        large_cpu_sparse_rescue=False,
        sparse_exact_direct=False,
        sparse_xblock_rescue_active=False,
        sparse_sxblock_rescue_active=False,
    )
    assert not decision.enabled
    assert decision.reason_size_disabled


def test_rhs1_sparse_rescue_ordering_disables_jax_path_on_memory_cap() -> None:
    decision = rhs1_resolved_sparse_rescue_ordering(
        sparse_enabled=True,
        sparse_kind_use="jax",
        size=1000,
        sparse_max_size=2000,
        sparse_jax_est_mb=512.0,
        sparse_jax_max_mb=128.0,
    )
    assert not decision.enabled
    assert decision.reason_sparse_jax_mem_disabled


def test_rhs1_sparse_rescue_ordering_disables_after_pas_fast_accept_or_gpu_skip() -> None:
    decision = rhs1_resolved_sparse_rescue_ordering(
        sparse_enabled=True,
        sparse_kind_use="auto",
        size=1000,
        sparse_max_size=2000,
        pas_fast_accept=True,
    )
    assert not decision.enabled
    assert decision.reason_pas_fast_accept

    decision = rhs1_resolved_sparse_rescue_ordering(
        sparse_enabled=True,
        sparse_kind_use="auto",
        size=1000,
        sparse_max_size=2000,
        gpu_sparse_skip=True,
    )
    assert not decision.enabled
    assert decision.reason_gpu_sparse_skip


def test_rhs1_sparse_rescue_policy_setup_computes_jax_memory_admission() -> None:
    setup = rhs1_sparse_rescue_policy_setup(
        sparse_precond_mode="on",
        sparse_precond_kind="jax",
        has_fp=True,
        has_pas=False,
        residual_norm=1.0,
        target=1.0e-8,
        rhs_mode=1,
        include_phi1=False,
        size=1000,
        sparse_max_size=2000,
        precond_dtype="float32",
        sparse_jax_max_mb=1.0,
    )
    assert not setup.enabled
    assert setup.kind_use == "jax"
    assert setup.sparse_jax_est_mb == 4.0
    assert setup.ordering.reason_sparse_jax_mem_disabled
    assert setup.sparse_jax_memory_disabled_message == (
        "sparse_jax: disabled (est_mem=4.0 MB > max_mb=1.0)"
    )


def test_rhs1_sparse_rescue_policy_setup_keeps_ordering_flags() -> None:
    setup = rhs1_sparse_rescue_policy_setup(
        sparse_precond_mode="on",
        sparse_precond_kind="auto",
        has_fp=True,
        has_pas=False,
        residual_norm=1.0,
        target=1.0e-8,
        rhs_mode=1,
        include_phi1=False,
        size=4096,
        sparse_max_size=1024,
        precond_dtype="float64",
        dense_shortcut=True,
        sparse_exact_direct=True,
        large_cpu_sparse_rescue=True,
        sparse_xblock_rescue_active=True,
    )
    assert setup.enabled
    assert setup.kind_use == "scipy"
    assert setup.sparse_jax_est_mb is None
    assert setup.ordering.prefer_sparse_exact_over_dense_shortcut
    assert setup.ordering.reason_size_large_cpu
    assert setup.ordering.reason_large_cpu_exact_skips_targeted
    assert not setup.ordering.xblock_rescue_active


def test_rhs1_sparse_rescue_initial_messages_format_policy_decisions() -> None:
    setup = rhs1_sparse_rescue_policy_setup(
        sparse_precond_mode="on",
        sparse_precond_kind="jax",
        has_fp=True,
        has_pas=False,
        residual_norm=1.0,
        target=1.0e-8,
        rhs_mode=1,
        include_phi1=False,
        size=1000,
        sparse_max_size=2000,
        precond_dtype="float32",
        sparse_jax_max_mb=1.0,
    )
    assert rhs1_sparse_rescue_initial_messages(
        ordering=setup.ordering,
        size=1000,
        sparse_max_size=2000,
        sparse_jax_memory_disabled_message=setup.sparse_jax_memory_disabled_message,
    ) == ((
        1,
        "sparse_jax: disabled (est_mem=4.0 MB > max_mb=1.0)",
    ),)

    decision = rhs1_resolved_sparse_rescue_ordering(
        sparse_enabled=True,
        sparse_kind_use="scipy",
        size=4096,
        sparse_max_size=1024,
        large_cpu_sparse_rescue=True,
    )
    assert rhs1_sparse_rescue_initial_messages(
        ordering=decision,
        size=4096,
        sparse_max_size=1024,
        large_cpu_sparse_exact_lu=False,
        large_cpu_label="gpu host-sparse",
    ) == ((
        0,
        "solve_v3_full_system_linear_gmres: gpu host-sparse ILU rescue "
        "(size=4096 > max=1024)",
    ),)


def test_rhs1_sparse_rescue_tail_skip_messages_format_policy_decisions() -> None:
    decision = rhs1_resolved_sparse_rescue_ordering(
        sparse_enabled=True,
        sparse_kind_use="scipy",
        size=4096,
        sparse_max_size=1024,
        large_cpu_sparse_rescue=True,
        sparse_exact_direct=True,
        pas_fast_accept=True,
    )
    assert rhs1_sparse_rescue_tail_skip_messages(
        ordering=decision,
        residual_norm=1.25e-4,
        rhs1_precond_kind="xblock",
    ) == (
        (
            1,
            "solve_v3_full_system_linear_gmres: exact large-CPU sparse LU selected "
            "-> skipping targeted sparse xblock/sxblock rescue",
        ),
        (
            1,
            "solve_v3_full_system_linear_gmres: PAS fast-accept "
            "(residual=1.250e-04) -> skip sparse rescue tail",
        ),
    )

    decision = rhs1_resolved_sparse_rescue_ordering(
        sparse_enabled=True,
        sparse_kind_use="scipy",
        size=256,
        sparse_max_size=1024,
        gpu_sparse_skip=True,
    )
    assert rhs1_sparse_rescue_tail_skip_messages(
        ordering=decision,
        residual_norm=2.0e-3,
        rhs1_precond_kind="pas_lite",
    ) == ((
        1,
        "solve_v3_full_system_linear_gmres: GPU sparse fallback skipped after "
        "pas_lite accept (residual=2.000e-03)",
    ),)
