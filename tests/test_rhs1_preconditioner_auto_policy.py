from __future__ import annotations

from sfincs_jax.rhs1_preconditioner_auto_policy import (
    pas_auto_skip_strong_retry,
    rhs1_gpu_sparse_fallback_skip_allowed,
    rhs1_pas_auto_large_base_kind,
    rhs1_pas_dkes_xblock_allowed,
    rhs1_pas_tokamak_cpu_xblock_preferred,
    rhs1_pas_tokamak_gpu_theta_allowed,
    rhs1_pas_tokamak_gpu_xblock_preferred,
    rhs1_sharded_line_override_allowed,
)


def test_pas_auto_large_base_kind_respects_threshold(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_PAS_LITE_MIN", raising=False)
    assert rhs1_pas_auto_large_base_kind(active_size=25_000) == "pas_lite"
    assert rhs1_pas_auto_large_base_kind(active_size=5_000) == "pas_hybrid"

    monkeypatch.setenv("SFINCS_JAX_PAS_LITE_MIN", "bad")
    assert rhs1_pas_auto_large_base_kind(active_size=25_000) == "pas_lite"

    monkeypatch.setenv("SFINCS_JAX_PAS_LITE_MIN", "30_000")
    assert rhs1_pas_auto_large_base_kind(active_size=25_000) == "pas_hybrid"


def test_pas_auto_skip_strong_retry_requires_pas_auto_and_strong_base() -> None:
    assert pas_auto_skip_strong_retry(
        has_pas=True,
        strong_precond_env="auto",
        rhs1_precond_kind="schur",
        residual_norm=5.0,
        target=1.0,
        ratio=10.0,
    )
    assert not pas_auto_skip_strong_retry(
        has_pas=True,
        strong_precond_env="auto",
        rhs1_precond_kind="theta_line",
        residual_norm=5.0,
        target=1.0,
        ratio=10.0,
    )
    assert not pas_auto_skip_strong_retry(
        has_pas=True,
        strong_precond_env="pas_tz",
        rhs1_precond_kind="schur",
        residual_norm=5.0,
        target=1.0,
        ratio=10.0,
    )
    assert not pas_auto_skip_strong_retry(
        has_pas=False,
        strong_precond_env="auto",
        rhs1_precond_kind="schur",
        residual_norm=5.0,
        target=1.0,
        ratio=10.0,
    )


def test_dkes_xblock_policy_bounds_backend_and_block_size() -> None:
    assert rhs1_pas_dkes_xblock_allowed(
        has_pas=True,
        use_dkes=True,
        backend="gpu",
        n_theta=9,
        n_zeta=11,
        max_l=21,
        xblock_tz_limit=2500,
    )
    assert not rhs1_pas_dkes_xblock_allowed(
        has_pas=True,
        use_dkes=True,
        backend="metal",
        n_theta=9,
        n_zeta=11,
        max_l=21,
        xblock_tz_limit=2500,
    )
    assert not rhs1_pas_dkes_xblock_allowed(
        has_pas=True,
        use_dkes=True,
        backend="gpu",
        n_theta=17,
        n_zeta=23,
        max_l=36,
        xblock_tz_limit=2500,
    )


def test_tokamak_pas_gpu_and_cpu_xblock_policies(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_PAS_TOKAMAK_GPU_THETA_MAX", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_PAS_TOKAMAK_GPU_XBLOCK_ACTIVE_MAX", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_PAS_TOKAMAK_CPU_XBLOCK_ACTIVE_MAX", raising=False)

    common = dict(
        has_pas=True,
        has_fp=False,
        tokamak_like=True,
        active_size=500,
        er_abs=1.0e-2,
        schur_er_min=1.0e-12,
        has_magdrift=False,
        has_collisionless=True,
    )
    assert rhs1_pas_tokamak_gpu_theta_allowed(backend="gpu", **common)
    assert rhs1_pas_tokamak_gpu_xblock_preferred(
        backend="gpu",
        n_theta=10,
        n_zeta=1,
        max_l=14,
        xblock_tz_limit=1200,
        **common,
    )
    assert not rhs1_pas_tokamak_gpu_xblock_preferred(
        backend="cpu",
        n_theta=10,
        n_zeta=1,
        max_l=14,
        xblock_tz_limit=1200,
        **common,
    )
    assert rhs1_pas_tokamak_cpu_xblock_preferred(
        backend="cpu",
        n_theta=10,
        n_zeta=1,
        max_l=14,
        xblock_tz_limit=1200,
        **common,
    )
    assert not rhs1_pas_tokamak_cpu_xblock_preferred(
        backend="gpu",
        n_theta=10,
        n_zeta=1,
        max_l=14,
        xblock_tz_limit=1200,
        **common,
    )


def test_gpu_sparse_fallback_skip_policy(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_GPU_SPARSE_SKIP_RATIO", raising=False)
    assert rhs1_gpu_sparse_fallback_skip_allowed(
        backend="gpu",
        rhs_mode=1,
        include_phi1=False,
        has_pas=True,
        rhs1_precond_kind="schur",
        use_active_dof_mode=True,
        residual_norm=5.0,
        target=1.0,
    )
    assert not rhs1_gpu_sparse_fallback_skip_allowed(
        backend="cpu",
        rhs_mode=1,
        include_phi1=False,
        has_pas=True,
        rhs1_precond_kind="schur",
        use_active_dof_mode=True,
        residual_norm=5.0,
        target=1.0,
    )

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_GPU_SPARSE_SKIP_RATIO", "0")
    assert not rhs1_gpu_sparse_fallback_skip_allowed(
        backend="gpu",
        rhs_mode=1,
        include_phi1=False,
        has_pas=True,
        rhs1_precond_kind="schur",
        use_active_dof_mode=True,
        residual_norm=5.0,
        target=1.0,
    )


def test_sharded_line_override_policy_preserves_dedicated_pas_preconditioners() -> None:
    assert rhs1_sharded_line_override_allowed(None)
    assert rhs1_sharded_line_override_allowed("theta_line")
    assert rhs1_sharded_line_override_allowed("pas_hybrid")
    assert not rhs1_sharded_line_override_allowed("schur")
    assert not rhs1_sharded_line_override_allowed("pas_tz")
    assert not rhs1_sharded_line_override_allowed("pas_tokamak_theta")
    assert not rhs1_sharded_line_override_allowed("pas_ilu")
