from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import jax.numpy as jnp

from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.v3_driver import (
    _build_rhsmode23_fp_local_geom_line_preconditioner,
    _build_rhsmode23_fp_structured_fblock_lu_preconditioner,
    _build_rhsmode23_fp_fortran_reduced_lu_preconditioner,
    _build_rhsmode23_fp_tzfft_line_preconditioner,
    _build_rhsmode23_fp_tzfft_line_schur_preconditioner,
    _build_rhsmode23_fp_xblock_tz_lu_preconditioner,
    _build_rhsmode23_fp_xblock_tz_lu_schur_preconditioner,
    _build_rhsmode23_sxblock_preconditioner,
    _host_sparse_factor_dtype,
    _transport_dense_accelerator_auto_allowed,
    _transport_tzfft_accelerator_auto_allowed,
    _transport_dense_backend_allowed,
    _transport_host_gmres_accepts_preconditioned_residual,
    _transport_host_gmres_first_attempt_allowed,
    _transport_disable_auto_recycle,
    _transport_precondition_side,
    _transport_sparse_factor_dtype,
    _transport_sparse_direct_needs_float64_retry,
    _transport_sparse_direct_first_attempt_allowed,
    _transport_sparse_direct_rescue_allowed,
    _transport_sparse_direct_rescue_first,
    _transport_sparse_direct_use_explicit_helper,
    _transport_tzfft_backend_allowed,
    _transport_tzfft_first_attempt_budget,
    _transport_tzfft_structured_first_attempt_allowed,
    solve_v3_transport_matrix_linear_gmres,
)
from sfincs_jax.operators.profile_response.system import (
    apply_v3_full_system_operator_cached,
    full_system_operator_from_namelist,
    rhs_v3_full_system,
    with_transport_rhs_settings,
)


def _op(
    *,
    rhs_mode: int = 2,
    has_fp: bool = True,
    has_phi1: bool = False,
    n_x: int = 4,
    constraint_scheme: int = 2,
):
    return SimpleNamespace(
        rhs_mode=rhs_mode,
        include_phi1=has_phi1,
        n_x=n_x,
        constraint_scheme=constraint_scheme,
        fblock=SimpleNamespace(fp=object() if has_fp else None),
    )


def test_transport_sparse_direct_rescue_enabled_for_cpu_fp_transport(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT", raising=False)
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_MAX", raising=False)
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_RATIO", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    assert _transport_sparse_direct_rescue_allowed(
        op=_op(rhs_mode=2),
        size=16382,
        residual_norm=1.0e-3,
        target=1.0e-9,
        use_implicit=False,
    )


def test_transport_sparse_direct_rescue_enabled_for_cpu_collisionless_mono_medium_size(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT", raising=False)
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_MAX", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    assert _transport_sparse_direct_rescue_allowed(
        op=_op(rhs_mode=3, has_fp=False, n_x=1),
        size=54811,
        residual_norm=2.0e-2,
        target=1.0e-6,
        use_implicit=False,
    )


def test_transport_sparse_direct_rescue_respects_guards(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    assert not _transport_sparse_direct_rescue_allowed(
        op=_op(rhs_mode=1),
        size=16382,
        residual_norm=1.0e-3,
        target=1.0e-9,
        use_implicit=False,
    )
    assert not _transport_sparse_direct_rescue_allowed(
        op=_op(rhs_mode=2, has_phi1=True),
        size=16382,
        residual_norm=1.0e-3,
        target=1.0e-9,
        use_implicit=False,
    )
    assert not _transport_sparse_direct_rescue_allowed(
        op=_op(rhs_mode=2),
        size=50000,
        residual_norm=1.0e-3,
        target=1.0e-9,
        use_implicit=False,
    )
    assert not _transport_sparse_direct_rescue_allowed(
        op=_op(rhs_mode=2),
        size=16382,
        residual_norm=1.0e-3,
        target=1.0e-9,
        use_implicit=True,
    )


def test_transport_sparse_direct_rescue_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT", "0")
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    assert not _transport_sparse_direct_rescue_allowed(
        op=_op(rhs_mode=2),
        size=16382,
        residual_norm=1.0e-3,
        target=1.0e-9,
        use_implicit=False,
    )


def test_transport_sparse_direct_rescue_respects_env_max(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT", raising=False)
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_MAX", "12000")
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    assert not _transport_sparse_direct_rescue_allowed(
        op=_op(rhs_mode=2),
        size=12001,
        residual_norm=1.0e-3,
        target=1.0e-9,
        use_implicit=False,
    )


def test_transport_sparse_direct_rescue_enabled_for_gpu_explicit_transport(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "gpu")
    assert _transport_sparse_direct_rescue_allowed(
        op=_op(rhs_mode=2),
        size=16382,
        residual_norm=1.0e-3,
        target=1.0e-9,
        use_implicit=False,
    )


def test_transport_sparse_direct_rescue_enabled_for_gpu_collisionless_transport(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "gpu")
    assert _transport_sparse_direct_rescue_allowed(
        op=_op(rhs_mode=3, has_fp=False, n_x=1),
        size=5383,
        residual_norm=1.0e-3,
        target=1.0e-9,
        use_implicit=False,
    )


def test_transport_sparse_direct_rescue_enabled_for_nonfinite_residual(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "gpu")
    assert _transport_sparse_direct_rescue_allowed(
        op=_op(rhs_mode=3, has_fp=False, n_x=1),
        size=5383,
        residual_norm=float("nan"),
        target=1.0e-9,
        use_implicit=False,
    )


def test_transport_sparse_direct_first_attempt_allowed_for_gpu_explicit_transport(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT", raising=False)
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_MAX", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "gpu")
    assert _transport_sparse_direct_first_attempt_allowed(
        op=_op(rhs_mode=3, has_fp=False, n_x=1),
        size=5383,
        use_implicit=False,
    )


def test_transport_sparse_direct_first_attempt_defers_to_structured_tzfft_on_gpu_mono_pas_floor(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT", raising=False)
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_MAX", raising=False)
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_TZFFT_ACCELERATOR_AUTO_MAX", raising=False)
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_TZFFT_FIRST", raising=False)
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_TZFFT_FIRST_MAX", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "gpu")
    op = SimpleNamespace(
        rhs_mode=3,
        include_phi1=False,
        n_x=1,
        n_theta=25,
        n_zeta=51,
        total_size=127501,
        fblock=SimpleNamespace(fp=None),
    )
    assert _transport_tzfft_structured_first_attempt_allowed(
        op,
        size=127501,
        use_implicit=False,
    )
    assert not _transport_sparse_direct_first_attempt_allowed(
        op=op,
        size=127501,
        use_implicit=False,
    )


def test_transport_structured_tzfft_first_attempt_guards_and_budget(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_TZFFT_FIRST", raising=False)
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_TZFFT_FIRST_MIN", raising=False)
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_TZFFT_FIRST_MAX", raising=False)
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_TZFFT_FIRST_RESTART", raising=False)
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_TZFFT_FIRST_MAXITER", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "gpu")
    op = SimpleNamespace(
        rhs_mode=3,
        include_phi1=False,
        n_x=1,
        n_theta=25,
        n_zeta=51,
        total_size=127501,
        fblock=SimpleNamespace(fp=None),
    )
    assert _transport_tzfft_structured_first_attempt_allowed(
        op,
        size=127501,
        use_implicit=False,
    )
    assert not _transport_tzfft_structured_first_attempt_allowed(
        op,
        size=127501,
        use_implicit=True,
    )
    assert not _transport_tzfft_structured_first_attempt_allowed(
        SimpleNamespace(**{**op.__dict__, "include_phi1": True}),
        size=127501,
        use_implicit=False,
    )
    assert not _transport_tzfft_structured_first_attempt_allowed(
        SimpleNamespace(**{**op.__dict__, "fblock": SimpleNamespace(fp=object())}),
        size=127501,
        use_implicit=False,
    )
    assert not _transport_tzfft_structured_first_attempt_allowed(
        SimpleNamespace(**{**op.__dict__, "n_x": 3}),
        size=127501,
        use_implicit=False,
    )
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_TZFFT_FIRST_MAX", "100000")
    assert not _transport_tzfft_structured_first_attempt_allowed(
        op,
        size=127501,
        use_implicit=False,
    )
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_TZFFT_FIRST_MAX", "bad")
    assert _transport_tzfft_structured_first_attempt_allowed(
        op,
        size=127501,
        use_implicit=False,
    )
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_TZFFT_FIRST", "0")
    assert not _transport_tzfft_structured_first_attempt_allowed(
        op,
        size=127501,
        use_implicit=False,
    )
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_TZFFT_FIRST", "1")
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    assert _transport_tzfft_structured_first_attempt_allowed(
        op,
        size=127501,
        use_implicit=True,
    )
    method, restart, maxiter = _transport_tzfft_first_attempt_budget(restart=80, maxiter=400)
    assert method == "incremental"
    assert restart == 40
    assert maxiter == 12
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_TZFFT_FIRST_RESTART", "23")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_TZFFT_FIRST_MAXITER", "7")
    assert _transport_tzfft_first_attempt_budget(restart=80, maxiter=400) == ("incremental", 23, 7)


def test_transport_sparse_direct_first_attempt_keeps_generic_large_gpu_transport_bounded(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT", raising=False)
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_MAX", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "gpu")
    assert not _transport_sparse_direct_first_attempt_allowed(
        op=_op(rhs_mode=2, has_fp=True, n_x=4),
        size=127501,
        use_implicit=False,
    )


def test_transport_fp_tzfft_line_reduces_one_apply_residual_vs_sxblock(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_TZFFT_LINE_DTYPE", "float64")
    nml = read_sfincs_input("tests/ref/output_scheme2_lhd_1species_tiny.input.namelist")
    op0 = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    op = with_transport_rhs_settings(op0, which_rhs=1)
    assert op0.fblock.fp is not None
    rhs = rhs_v3_full_system(op)
    rhs_norm = float(jnp.linalg.norm(rhs))

    def _relative_one_apply_residual(preconditioner) -> float:
        z = preconditioner(rhs)
        residual = apply_v3_full_system_operator_cached(op, z) - rhs
        return float(jnp.linalg.norm(residual)) / max(rhs_norm, np.finfo(np.float64).tiny)

    sxblock_rel = _relative_one_apply_residual(_build_rhsmode23_sxblock_preconditioner(op=op0))
    line_rel = _relative_one_apply_residual(_build_rhsmode23_fp_tzfft_line_preconditioner(op=op0))

    assert line_rel < sxblock_rel * 1.0e-6
    assert line_rel < 1.0e4


def test_transport_fp_local_geom_line_is_finite_diagnostic_candidate(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_LOCAL_GEOM_LINE_DTYPE", "float64")
    nml = read_sfincs_input("tests/ref/output_scheme2_lhd_1species_tiny.input.namelist")
    op0 = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    op = with_transport_rhs_settings(op0, which_rhs=1)
    rhs = rhs_v3_full_system(op)
    rhs_norm = float(jnp.linalg.norm(rhs))

    def _relative_one_apply_residual(preconditioner) -> float:
        z = preconditioner(rhs)
        residual = apply_v3_full_system_operator_cached(op, z) - rhs
        return float(jnp.linalg.norm(residual)) / max(rhs_norm, np.finfo(np.float64).tiny)

    geom_rel = _relative_one_apply_residual(_build_rhsmode23_fp_local_geom_line_preconditioner(op=op0))

    assert np.isfinite(geom_rel)
    assert geom_rel > 0.0


def test_transport_fp_structured_fblock_lu_reduces_kinetic_residual_vs_line(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_STRUCTURED_FBLOCK_LU_MAX_MB", "256")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_STRUCTURED_FBLOCK_LU_FACTOR_MAX_MB", "512")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_STRUCTURED_FBLOCK_LU_REG", "1e-12")
    nml = read_sfincs_input("tests/ref/output_scheme2_lhd_1species_tiny.input.namelist")
    op0 = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    op = with_transport_rhs_settings(op0, which_rhs=1)
    rhs = rhs_v3_full_system(op)
    rhs_norm = float(jnp.linalg.norm(rhs))

    def _relative_one_apply_residual(preconditioner) -> float:
        z = preconditioner(rhs)
        residual = apply_v3_full_system_operator_cached(op, z) - rhs
        return float(jnp.linalg.norm(residual[: op.f_size])) / max(rhs_norm, np.finfo(np.float64).tiny)

    line_rel = _relative_one_apply_residual(_build_rhsmode23_fp_tzfft_line_preconditioner(op=op0))
    fblock_rel = _relative_one_apply_residual(_build_rhsmode23_fp_structured_fblock_lu_preconditioner(op=op0))

    assert fblock_rel < line_rel * 1.0e-6
    assert fblock_rel < 1.0e-8


def test_transport_fp_xblock_tz_lu_reduces_kinetic_residual_vs_sxblock(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_XBLOCK_TZ_LU_MAX_MB", "256")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_XBLOCK_TZ_LU_FACTOR_MAX_MB", "512")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_XBLOCK_TZ_LU_REG", "1e-12")
    nml = read_sfincs_input("tests/ref/output_scheme2_lhd_1species_tiny.input.namelist")
    op0 = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    op = with_transport_rhs_settings(op0, which_rhs=1)
    rhs = rhs_v3_full_system(op)
    rhs_norm = float(jnp.linalg.norm(rhs))

    def _relative_kinetic_residual(preconditioner) -> float:
        z = preconditioner(rhs)
        residual = apply_v3_full_system_operator_cached(op, z) - rhs
        return float(jnp.linalg.norm(residual[: op.f_size])) / max(rhs_norm, np.finfo(np.float64).tiny)

    sxblock_rel = _relative_kinetic_residual(_build_rhsmode23_sxblock_preconditioner(op=op0))
    xblock_rel = _relative_kinetic_residual(_build_rhsmode23_fp_xblock_tz_lu_preconditioner(op=op0))

    assert np.isfinite(xblock_rel)
    assert xblock_rel < sxblock_rel * 1.0e-6


def test_transport_fp_xblock_tz_lu_schur_does_not_increase_tail_residual(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_XBLOCK_TZ_LU_MAX_MB", "256")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_XBLOCK_TZ_LU_FACTOR_MAX_MB", "512")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_XBLOCK_TZ_LU_REG", "3e-13")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_XBLOCK_TZ_LU_SCHUR_RESTRICTION", "tail_galerkin")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_XBLOCK_TZ_LU_SCHUR_MAX_COLS", "16")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_XBLOCK_TZ_LU_SCHUR_MAX_MB", "128")
    nml = read_sfincs_input("tests/ref/output_scheme2_lhd_1species_tiny.input.namelist")
    op0 = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    op = with_transport_rhs_settings(op0, which_rhs=1)
    rhs = rhs_v3_full_system(op)
    rhs_norm = float(jnp.linalg.norm(rhs))

    def _tail_relative_residual(preconditioner) -> float:
        z = preconditioner(rhs)
        residual = apply_v3_full_system_operator_cached(op, z) - rhs
        return float(jnp.linalg.norm(residual[op.f_size :])) / max(rhs_norm, np.finfo(np.float64).tiny)

    base_rel = _tail_relative_residual(_build_rhsmode23_fp_xblock_tz_lu_preconditioner(op=op0))
    schur_rel = _tail_relative_residual(_build_rhsmode23_fp_xblock_tz_lu_schur_preconditioner(op=op0))

    assert np.isfinite(schur_rel)
    tail_roundoff_floor = 1.0e-10
    assert schur_rel <= max(base_rel, tail_roundoff_floor)
    if base_rel > tail_roundoff_floor:
        assert schur_rel < base_rel * 1.0e-2
    else:
        assert schur_rel < tail_roundoff_floor


def test_transport_fp_xblock_tz_lu_schur_residual_coarse_reduces_all_rhs(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_XBLOCK_TZ_LU_MAX_MB", "256")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_XBLOCK_TZ_LU_FACTOR_MAX_MB", "512")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_XBLOCK_TZ_LU_REG", "3e-13")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_XBLOCK_TZ_LU_SCHUR_RESTRICTION", "tail_galerkin")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_XBLOCK_TZ_LU_SCHUR_MAX_COLS", "16")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_XBLOCK_TZ_LU_SCHUR_MAX_MB", "128")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_XBLOCK_TZ_LU_SCHUR_DAMPING", "0.25")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_XBLOCK_TZ_LU_SCHUR_KINETIC_RESIDUAL", "1")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_XBLOCK_TZ_LU_SCHUR_RHS_RESIDUAL", "1")
    nml = read_sfincs_input("tests/ref/output_scheme2_lhd_1species_tiny.input.namelist")
    op0 = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    base = _build_rhsmode23_fp_xblock_tz_lu_preconditioner(op=op0)
    schur = _build_rhsmode23_fp_xblock_tz_lu_schur_preconditioner(op=op0)

    def _relative_residual(which_rhs: int, preconditioner) -> float:
        op = with_transport_rhs_settings(op0, which_rhs=int(which_rhs))
        rhs = rhs_v3_full_system(op)
        rhs_norm = float(jnp.linalg.norm(rhs))
        z = preconditioner(rhs)
        residual = apply_v3_full_system_operator_cached(op, z) - rhs
        return float(jnp.linalg.norm(residual)) / max(rhs_norm, np.finfo(np.float64).tiny)

    for which_rhs in (1, 2, 3):
        base_rel = _relative_residual(which_rhs, base)
        schur_rel = _relative_residual(which_rhs, schur)
        assert np.isfinite(schur_rel)
        assert schur_rel < base_rel


def test_transport_fp_fortran_reduced_lu_reduces_full_residual_all_rhs(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_EXPLICIT_SPARSE_CSR_MAX_MB", "128")
    monkeypatch.setenv("SFINCS_JAX_EXPLICIT_SPARSE_FACTOR_KIND", "lu")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_FACTOR_MAX_MB", "256")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SHIFT", "1e-12")
    nml = read_sfincs_input("tests/ref/output_scheme2_lhd_1species_tiny.input.namelist")
    op0 = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    precond = _build_rhsmode23_fp_fortran_reduced_lu_preconditioner(op=op0)

    metadata = getattr(precond, "_sfincs_jax_transport_fp_fortran_reduced_lu_metadata")
    assert metadata["factor_kind"] == "lu"
    assert metadata["linear_size"] == int(op0.total_size)

    for which_rhs in (1, 2, 3):
        op = with_transport_rhs_settings(op0, which_rhs=int(which_rhs))
        rhs = rhs_v3_full_system(op)
        rhs_norm = float(jnp.linalg.norm(rhs))
        z = precond(rhs)
        residual = apply_v3_full_system_operator_cached(op, z) - rhs
        rel = float(jnp.linalg.norm(residual)) / max(rhs_norm, np.finfo(np.float64).tiny)
        assert np.isfinite(rel)
        assert rel < 1.0


def test_transport_fp_tzfft_line_schur_reduces_line_tail_residual(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_TZFFT_LINE_DTYPE", "float64")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_TZFFT_LINE_SCHUR_DTYPE", "float64")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_TZFFT_LINE_SCHUR_MAX_COLS", "16")
    nml = read_sfincs_input("tests/ref/output_scheme2_lhd_1species_tiny.input.namelist")
    op0 = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    op = with_transport_rhs_settings(op0, which_rhs=1)
    rhs = rhs_v3_full_system(op)
    rhs_norm = float(jnp.linalg.norm(rhs))

    def _tail_relative_one_apply_residual(preconditioner) -> float:
        z = preconditioner(rhs)
        residual = apply_v3_full_system_operator_cached(op, z) - rhs
        return float(jnp.linalg.norm(residual[op.f_size :])) / max(rhs_norm, np.finfo(np.float64).tiny)

    line_rel = _tail_relative_one_apply_residual(_build_rhsmode23_fp_tzfft_line_preconditioner(op=op0))
    schur = _build_rhsmode23_fp_tzfft_line_schur_preconditioner(op=op0)
    schur_rel = _tail_relative_one_apply_residual(schur)

    assert schur_rel < line_rel * 1.0e-3
    assert schur_rel < 1.0e-8

    for i_extra in range(op.extra_size):
        tail_rhs = jnp.zeros((op.total_size,), dtype=jnp.float64).at[op.f_size + i_extra].set(1.0)
        z_tail = schur(tail_rhs)
        residual_tail = apply_v3_full_system_operator_cached(op, z_tail) - tail_rhs
        assert float(jnp.linalg.norm(residual_tail[op.f_size :])) < 1.0e-8


def test_transport_sparse_direct_first_attempt_disabled_for_gpu_tzfft_auto_case(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT", raising=False)
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_TZFFT_ACCELERATOR_AUTO_MAX", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "gpu")
    op = SimpleNamespace(
        rhs_mode=3,
        include_phi1=False,
        n_x=1,
        n_theta=37,
        n_zeta=5,
        total_size=3697,
        fblock=SimpleNamespace(fp=None),
    )
    assert not _transport_sparse_direct_first_attempt_allowed(
        op=op,
        size=3697,
        use_implicit=False,
    )


def test_transport_sparse_direct_first_attempt_disabled_for_cpu_collisionless_mono_medium_size(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT", raising=False)
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_FIRST_CPU_MIN", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    assert not _transport_sparse_direct_first_attempt_allowed(
        op=_op(rhs_mode=3, has_fp=False, n_x=1),
        size=54811,
        use_implicit=False,
    )


def test_transport_sparse_direct_first_attempt_enabled_for_cpu_transport_fast_path(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT", raising=False)
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_FIRST_CPU_MIN", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    assert _transport_sparse_direct_first_attempt_allowed(
        op=_op(rhs_mode=2),
        size=16382,
        use_implicit=False,
    )


def test_transport_sparse_direct_first_attempt_disabled_for_small_cpu_or_implicit(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT", raising=False)
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_FIRST_CPU_MIN", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    assert not _transport_sparse_direct_first_attempt_allowed(
        op=_op(rhs_mode=2),
        size=8000,
        use_implicit=False,
    )
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "gpu")
    assert not _transport_sparse_direct_first_attempt_allowed(
        op=_op(rhs_mode=2),
        size=16382,
        use_implicit=True,
    )


def test_transport_sparse_direct_helper_auto_prefers_gpu_and_large_cpu(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_SPARSE_HELPER", raising=False)
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_SPARSE_HELPER_CPU_MIN", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "gpu")
    assert _transport_sparse_direct_use_explicit_helper(size=2048)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    assert _transport_sparse_direct_use_explicit_helper(size=12000)
    assert not _transport_sparse_direct_use_explicit_helper(size=8000)


def test_transport_sparse_direct_helper_env_overrides(monkeypatch) -> None:
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_HELPER", "0")
    assert not _transport_sparse_direct_use_explicit_helper(size=50000)
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_HELPER", "1")
    assert _transport_sparse_direct_use_explicit_helper(size=1024)


def test_host_sparse_factor_dtype_defaults_to_float32_for_large_explicit_cpu_lu(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_HOST_SPARSE_FACTOR_DTYPE", raising=False)
    monkeypatch.delenv("SFINCS_JAX_HOST_SPARSE_FACTOR_FLOAT32_MIN", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    assert _host_sparse_factor_dtype(size=16382, factorization="lu", use_implicit=False).name == "float32"
    assert _host_sparse_factor_dtype(size=8000, factorization="lu", use_implicit=False).name == "float64"
    assert _host_sparse_factor_dtype(size=16382, factorization="ilu", use_implicit=False).name == "float64"
    assert _host_sparse_factor_dtype(size=16382, factorization="lu", use_implicit=True).name == "float64"


def test_transport_sparse_factor_dtype_defaults_to_float64_for_large_cpu_transport(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_SPARSE_FACTOR_DTYPE", raising=False)
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_SPARSE_FLOAT64_MIN", raising=False)
    monkeypatch.delenv("SFINCS_JAX_HOST_SPARSE_FACTOR_DTYPE", raising=False)
    monkeypatch.delenv("SFINCS_JAX_HOST_SPARSE_FACTOR_FLOAT32_MIN", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    assert _transport_sparse_factor_dtype(size=35063, use_implicit=False).name == "float64"
    assert _transport_sparse_factor_dtype(size=16382, use_implicit=False).name == "float32"


def test_transport_host_gmres_first_attempt_enabled_for_cpu_collisionless_mono_medium_size(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_HOST_GMRES_FIRST", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    assert _transport_host_gmres_first_attempt_allowed(
        op=_op(rhs_mode=3, has_fp=False, n_x=1),
        size=54811,
        use_implicit=False,
    )


def test_transport_host_gmres_first_attempt_respects_guards(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_HOST_GMRES_FIRST", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    assert not _transport_host_gmres_first_attempt_allowed(
        op=_op(rhs_mode=2),
        size=16382,
        use_implicit=False,
    )
    assert not _transport_host_gmres_first_attempt_allowed(
        op=_op(rhs_mode=3, has_fp=False, n_x=3),
        size=54811,
        use_implicit=False,
    )
    assert not _transport_host_gmres_first_attempt_allowed(
        op=_op(rhs_mode=3, has_fp=False, n_x=1, has_phi1=True),
        size=54811,
        use_implicit=False,
    )
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "gpu")
    assert not _transport_host_gmres_first_attempt_allowed(
        op=_op(rhs_mode=3, has_fp=False, n_x=1),
        size=54811,
        use_implicit=False,
    )


def test_transport_host_gmres_accepts_preconditioned_residual_for_moderate_true_gap(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_HOST_GMRES_TRUE_RATIO", raising=False)
    assert _transport_host_gmres_accepts_preconditioned_residual(
        op=_op(rhs_mode=3, has_fp=False, n_x=1),
        true_residual_norm=5.0e-5,
        target_true=1.0e-6,
    )


def test_transport_precondition_side_defaults_to_left(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_PRECONDITION_SIDE", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    assert _transport_precondition_side(
        op=_op(rhs_mode=3, has_fp=False, n_x=1, constraint_scheme=2),
        use_implicit=False,
    ) == "left"


def test_transport_precondition_side_defaults_to_left_otherwise(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_PRECONDITION_SIDE", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    assert _transport_precondition_side(
        op=_op(rhs_mode=2, has_fp=False, n_x=1, constraint_scheme=1),
        use_implicit=False,
    ) == "left"
    assert _transport_precondition_side(
        op=_op(rhs_mode=3, has_fp=False, n_x=1, constraint_scheme=2),
        use_implicit=True,
    ) == "left"


def test_transport_precondition_side_respects_env_override(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_PRECONDITION_SIDE", "right")
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    assert _transport_precondition_side(
        op=_op(rhs_mode=3, has_fp=False, n_x=1, constraint_scheme=2),
        use_implicit=False,
    ) == "right"


def test_transport_disable_auto_recycle_defaults_on_for_explicit_cpu_mono_pas(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_DISABLE_AUTO_RECYCLE", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    assert _transport_disable_auto_recycle(
        op=_op(rhs_mode=3, has_fp=False, n_x=1, constraint_scheme=2),
        use_implicit=False,
    )


def test_transport_disable_auto_recycle_respects_guards_and_env(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_DISABLE_AUTO_RECYCLE", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    assert not _transport_disable_auto_recycle(
        op=_op(rhs_mode=2, has_fp=False, n_x=1, constraint_scheme=2),
        use_implicit=False,
    )
    assert not _transport_disable_auto_recycle(
        op=_op(rhs_mode=3, has_fp=False, n_x=1, constraint_scheme=2),
        use_implicit=True,
    )
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_DISABLE_AUTO_RECYCLE", "0")
    assert not _transport_disable_auto_recycle(
        op=_op(rhs_mode=3, has_fp=False, n_x=1, constraint_scheme=2),
        use_implicit=False,
    )


def test_transport_sparse_direct_rescue_has_defined_drop_controls(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FORCE_KRYLOV", "1")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT", "1")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_MAX", "2000")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_DENSE_RETRY_MAX", "0")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_MAXITER", "1")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_DROP_TOL", "bad")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_DROP_REL", "bad")

    nml = read_sfincs_input("tests/ref/transportMatrix_PAS_tiny_rhsMode2_scheme2.input.namelist")
    result = solve_v3_transport_matrix_linear_gmres(
        nml=nml,
        tol=1.0e-14,
        maxiter=1,
        which_rhs_values=[2],
        collect_transport_output_fields=False,
        emit=None,
    )
    residual = float(np.asarray(result.residual_norms_by_rhs[2], dtype=np.float64))
    assert np.isfinite(residual)
    assert residual < 1.0e-8


def test_transport_sparse_direct_can_use_pattern_probe_without_dense_identity(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FORCE_KRYLOV", "1")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_PATTERN", "1")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT", "1")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_MAX", "20000")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_FIRST_CPU_MIN", "1")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_HELPER", "0")

    nml = read_sfincs_input("tests/ref/transportMatrix_PAS_tiny_rhsMode2_scheme2.input.namelist")
    messages: list[str] = []
    result = solve_v3_transport_matrix_linear_gmres(
        nml=nml,
        tol=1.0e-10,
        maxiter=5,
        which_rhs_values=[2],
        collect_transport_output_fields=False,
        differentiable=False,
        emit=lambda _level, message: messages.append(message),
    )

    residual = float(np.asarray(result.residual_norms_by_rhs[2], dtype=np.float64))
    assert result.solver_kinds_by_rhs[2] == "sparse_lu"
    assert residual < 1.0e-8
    assert any("source=pattern" in message for message in messages)
    assert any("pattern-probed sparse materialization" in message for message in messages)


def test_transport_host_gmres_accepts_preconditioned_residual_for_branch_sensitive_mono_cpu_gap(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_HOST_GMRES_TRUE_RATIO", raising=False)
    assert _transport_host_gmres_accepts_preconditioned_residual(
        op=_op(rhs_mode=3, has_fp=False, n_x=1),
        true_residual_norm=6.5e-8,
        target_true=5.5e-10,
    )


def test_transport_host_gmres_rejects_preconditioned_residual_for_large_true_gap(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_HOST_GMRES_TRUE_RATIO", raising=False)
    assert not _transport_host_gmres_accepts_preconditioned_residual(
        op=_op(rhs_mode=3, has_fp=False, n_x=1),
        true_residual_norm=2.0e-2,
        target_true=1.0e-6,
    )


def test_transport_sparse_direct_rescue_waits_longer_for_branch_sensitive_mono_cpu_gap(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT", raising=False)
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_RATIO", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    assert not _transport_sparse_direct_rescue_allowed(
        op=_op(rhs_mode=3, has_fp=False, n_x=1),
        size=54811,
        residual_norm=6.5e-8,
        target=5.5e-10,
        use_implicit=False,
    )


def test_transport_sparse_direct_float64_retry_only_triggers_for_large_float32_gap(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_FLOAT64_RETRY_RATIO", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    assert _transport_sparse_direct_needs_float64_retry(
        factor_dtype=_host_sparse_factor_dtype(size=16382, factorization="lu", use_implicit=False),
        residual_norm=2.0e-5,
        target_true=1.0e-6,
    )
    assert not _transport_sparse_direct_needs_float64_retry(
        factor_dtype=_host_sparse_factor_dtype(size=16382, factorization="lu", use_implicit=False),
        residual_norm=5.0e-6,
        target_true=1.0e-6,
    )
    assert not _transport_sparse_direct_needs_float64_retry(
        factor_dtype=_host_sparse_factor_dtype(size=8000, factorization="lu", use_implicit=False),
        residual_norm=2.0e-5,
        target_true=1.0e-6,
    )


def test_transport_dense_backend_allowed_defaults_to_cpu(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_DENSE_ALLOW_ACCELERATOR", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    assert _transport_dense_backend_allowed()
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "gpu")
    assert not _transport_dense_backend_allowed()


def test_transport_tzfft_backend_allowed_defaults_to_cpu(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_TZFFT_ALLOW_ACCELERATOR", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    assert _transport_tzfft_backend_allowed()
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "gpu")
    assert not _transport_tzfft_backend_allowed()


def test_transport_dense_backend_allowed_respects_env(monkeypatch) -> None:
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "gpu")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_DENSE_ALLOW_ACCELERATOR", "1")
    assert _transport_dense_backend_allowed()
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_DENSE_ALLOW_ACCELERATOR", "0")
    assert not _transport_dense_backend_allowed()


def test_transport_dense_accelerator_auto_allowed_for_bounded_mono_case(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_DENSE_ALLOW_ACCELERATOR", raising=False)
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_DENSE_ACCELERATOR_AUTO", raising=False)
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_DENSE_ACCELERATOR_AUTO_GEOMETRIES", raising=False)
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_DENSE_ACCELERATOR_AUTO_MAX", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "gpu")
    op = SimpleNamespace(
        rhs_mode=3,
        include_phi1=False,
        n_x=1,
        n_theta=9,
        n_zeta=9,
        total_size=1864,
        fblock=SimpleNamespace(fp=None),
    )
    assert _transport_dense_accelerator_auto_allowed(op, geometry_scheme=1)
    assert not _transport_dense_accelerator_auto_allowed(op, geometry_scheme=5)

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_DENSE_ACCELERATOR_AUTO_GEOMETRIES", "5")
    assert _transport_dense_accelerator_auto_allowed(op, geometry_scheme=5)

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_DENSE_ACCELERATOR_AUTO_MAX", "1000")
    assert not _transport_dense_accelerator_auto_allowed(op, geometry_scheme=5)

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_DENSE_ACCELERATOR_AUTO_MAX", "2500")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_DENSE_ALLOW_ACCELERATOR", "0")
    assert not _transport_dense_accelerator_auto_allowed(op, geometry_scheme=5)


def test_transport_tzfft_backend_allowed_respects_env(monkeypatch) -> None:
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "gpu")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_TZFFT_ALLOW_ACCELERATOR", "1")
    assert _transport_tzfft_backend_allowed()
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_TZFFT_ALLOW_ACCELERATOR", "0")
    assert not _transport_tzfft_backend_allowed()


def test_transport_tzfft_accelerator_auto_allowed_for_bounded_collisionless_gpu_case(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_TZFFT_ACCELERATOR_AUTO_MAX", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "gpu")
    op = SimpleNamespace(
        rhs_mode=3,
        include_phi1=False,
        n_x=1,
        n_theta=37,
        n_zeta=5,
        total_size=3697,
        fblock=SimpleNamespace(fp=None),
    )
    assert _transport_tzfft_accelerator_auto_allowed(op)


def test_transport_tzfft_accelerator_auto_rejects_large_or_fp_gpu_case(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_TZFFT_ACCELERATOR_AUTO_MAX", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "gpu")
    large_op = SimpleNamespace(
        rhs_mode=3,
        include_phi1=False,
        n_x=1,
        n_theta=37,
        n_zeta=5,
        total_size=7000,
        fblock=SimpleNamespace(fp=None),
    )
    fp_op = SimpleNamespace(
        rhs_mode=2,
        include_phi1=False,
        n_x=1,
        n_theta=37,
        n_zeta=5,
        total_size=3697,
        fblock=SimpleNamespace(fp=object()),
    )
    assert not _transport_tzfft_accelerator_auto_allowed(large_op)
    assert not _transport_tzfft_accelerator_auto_allowed(fp_op)


def test_transport_sparse_direct_rescue_first_defaults_on(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_FIRST", raising=False)
    assert _transport_sparse_direct_rescue_first(sparse_direct_rescue=True)
    assert not _transport_sparse_direct_rescue_first(sparse_direct_rescue=False)


def test_transport_sparse_direct_rescue_first_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_FIRST", "0")
    assert not _transport_sparse_direct_rescue_first(sparse_direct_rescue=True)
