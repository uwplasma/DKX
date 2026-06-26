from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

import sfincs_jax.v3_driver as vd
import sfincs_jax.problems.profile_response.preconditioner_build as pb
import sfincs_jax.problems.profile_response.sparse.direct as sparse_direct
from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.solvers.preconditioning import _RHSMODE1_STRUCTURED_FBLOCK_PRECOND_CACHE
from sfincs_jax.operators.profile_response.system import apply_v3_full_system_operator, full_system_operator_from_namelist


def _op(*, with_pas: bool = False, with_fp: bool = False):
    return SimpleNamespace(
        fblock=SimpleNamespace(
            pas=object() if with_pas else None,
            fp=object() if with_fp else None,
        )
    )


def test_rhs1_dispatch_theta_dd_uses_dd_without_overlap(monkeypatch) -> None:
    sentinel = object()
    seen: dict[str, int] = {}

    def _builder(**kwargs):
        seen["block"] = kwargs["block"]
        return sentinel

    monkeypatch.setattr(pb, "_build_rhsmode1_theta_dd_preconditioner", _builder)
    assert (
        pb._build_rhs1_preconditioner_from_kind(
            op=_op(),
            rhs1_precond_kind="theta_dd",
            dd_block_theta=11,
            dd_overlap_theta=0,
        )
        is sentinel
    )
    assert seen == {"block": 11}


def test_rhs1_dispatch_theta_dd_uses_schwarz_with_overlap(monkeypatch) -> None:
    sentinel = object()
    seen: dict[str, int] = {}

    def _builder(**kwargs):
        seen["block"] = kwargs["block"]
        seen["overlap"] = kwargs["overlap"]
        return sentinel

    monkeypatch.setattr(pb, "_build_rhsmode1_theta_schwarz_preconditioner", _builder)
    assert (
        pb._build_rhs1_preconditioner_from_kind(
            op=_op(),
            rhs1_precond_kind="theta_dd",
            dd_block_theta=13,
            dd_overlap_theta=2,
        )
        is sentinel
    )
    assert seen == {"block": 13, "overlap": 2}


def test_rhs1_dispatch_point_xdiag_forwards_preconditioner_xi(monkeypatch) -> None:
    sentinel = object()
    seen: dict[str, int] = {}

    def _builder(**kwargs):
        seen["preconditioner_xi"] = kwargs["preconditioner_xi"]
        return sentinel

    monkeypatch.setattr(pb, "_build_rhsmode1_block_preconditioner_xdiag", _builder)
    assert (
        pb._build_rhs1_preconditioner_from_kind(
            op=_op(),
            rhs1_precond_kind="point_xdiag",
            preconditioner_xi=7,
        )
        is sentinel
    )
    assert seen == {"preconditioner_xi": 7}


def test_rhs1_dispatch_xblock_tz_lmax_forwards_lmax(monkeypatch) -> None:
    sentinel = object()
    seen: dict[str, int] = {}

    def _builder(**kwargs):
        seen["lmax"] = kwargs["lmax"]
        return sentinel

    monkeypatch.setattr(pb, "_build_rhsmode1_xblock_tz_lmax_preconditioner", _builder)
    assert (
        pb._build_rhs1_preconditioner_from_kind(
            op=_op(),
            rhs1_precond_kind="xblock_tz_lmax",
            rhs1_xblock_tz_lmax=5,
        )
        is sentinel
    )
    assert seen == {"lmax": 5}


def test_rhs1_dispatch_theta_line_xdiag_composes_collision_for_pas(monkeypatch) -> None:
    line = object()
    collision = object()
    sentinel = object()

    monkeypatch.setattr(pb, "_build_rhsmode1_theta_line_xdiag_preconditioner", lambda **kwargs: line)
    monkeypatch.setattr(pb, "_build_rhsmode1_collision_preconditioner", lambda **kwargs: collision)
    monkeypatch.setattr(pb, "_compose_preconditioners", lambda a, b: sentinel if (a, b) == (collision, line) else None)

    assert pb._build_rhs1_preconditioner_from_kind(op=_op(with_pas=True), rhs1_precond_kind="theta_line_xdiag") is sentinel


def test_rhs1_dispatch_default_falls_back_to_block_preconditioner(monkeypatch) -> None:
    sentinel = object()
    seen: dict[str, int] = {}

    def _builder(**kwargs):
        seen["species"] = kwargs["preconditioner_species"]
        seen["x"] = kwargs["preconditioner_x"]
        seen["xi"] = kwargs["preconditioner_xi"]
        return sentinel

    monkeypatch.setattr(pb, "_build_rhsmode1_block_preconditioner", _builder)
    assert (
        pb._build_rhs1_preconditioner_from_kind(
            op=_op(),
            rhs1_precond_kind="unknown-kind",
            preconditioner_species=2,
            preconditioner_x=3,
            preconditioner_xi=4,
        )
        is sentinel
    )
    assert seen == {"species": 2, "x": 3, "xi": 4}


def test_rhs1_dispatch_structured_fblock_jacobi_uses_builder(monkeypatch) -> None:
    sentinel = object()

    monkeypatch.setattr(pb, "_build_rhsmode1_structured_fblock_jacobi_preconditioner", lambda **_kwargs: sentinel)

    assert (
        pb._build_rhs1_preconditioner_from_kind(
            op=_op(with_fp=True),
            rhs1_precond_kind="structured_fblock_jacobi",
        )
        is sentinel
    )


def test_rhs1_dispatch_structured_fblock_angular_jacobi_uses_builder(monkeypatch) -> None:
    sentinel = object()

    monkeypatch.setattr(
        pb,
        "_build_rhsmode1_structured_fblock_angular_jacobi_preconditioner",
        lambda **_kwargs: sentinel,
    )

    assert (
        pb._build_rhs1_preconditioner_from_kind(
            op=_op(with_fp=True),
            rhs1_precond_kind="structured_fblock_angular_jacobi",
        )
        is sentinel
    )


def test_rhs1_dispatch_structured_fblock_xi_angular_jacobi_uses_builder(monkeypatch) -> None:
    sentinel = object()

    monkeypatch.setattr(
        pb,
        "_build_rhsmode1_structured_fblock_xi_angular_jacobi_preconditioner",
        lambda **_kwargs: sentinel,
    )

    assert (
        pb._build_rhs1_preconditioner_from_kind(
            op=_op(with_fp=True),
            rhs1_precond_kind="structured_fblock_xi_angular_jacobi",
        )
        is sentinel
    )


def test_rhs1_dispatch_structured_fblock_fp_radial_jacobi_uses_builder(monkeypatch) -> None:
    sentinel = object()

    monkeypatch.setattr(
        pb,
        "_build_rhsmode1_structured_fblock_fp_radial_jacobi_preconditioner",
        lambda **_kwargs: sentinel,
    )

    assert (
        pb._build_rhs1_preconditioner_from_kind(
            op=_op(with_fp=True),
            rhs1_precond_kind="structured_fblock_fp_radial_jacobi",
        )
        is sentinel
    )


def test_rhs1_dispatch_structured_fblock_fp_lowmode_schur_uses_builder(monkeypatch) -> None:
    sentinel = object()

    monkeypatch.setattr(
        pb,
        "_build_rhsmode1_structured_fblock_fp_lowmode_schur_preconditioner",
        lambda **_kwargs: sentinel,
    )

    assert (
        pb._build_rhs1_preconditioner_from_kind(
            op=_op(with_fp=True),
            rhs1_precond_kind="structured_fblock_fp_lowmode_schur",
        )
        is sentinel
    )


def test_rhs1_dispatch_structured_fblock_fp_moment_schur_uses_builder(monkeypatch) -> None:
    sentinel = object()

    monkeypatch.setattr(
        pb,
        "_build_rhsmode1_structured_fblock_fp_moment_schur_preconditioner",
        lambda **_kwargs: sentinel,
    )

    assert (
        pb._build_rhs1_preconditioner_from_kind(
            op=_op(with_fp=True),
            rhs1_precond_kind="structured_fblock_fp_moment_schur",
        )
        is sentinel
    )


def test_rhs1_dispatch_structured_fblock_fp_coupled_moment_schur_uses_builder(monkeypatch) -> None:
    sentinel = object()

    monkeypatch.setattr(
        pb,
        "_build_rhsmode1_structured_fblock_fp_coupled_moment_schur_preconditioner",
        lambda **_kwargs: sentinel,
    )

    assert (
        pb._build_rhs1_preconditioner_from_kind(
            op=_op(with_fp=True),
            rhs1_precond_kind="structured_fblock_fp_coupled_moment_schur",
        )
        is sentinel
    )


def test_rhs1_dispatch_structured_fblock_fp_tail_coupled_schur_uses_builder(monkeypatch) -> None:
    sentinel = object()

    monkeypatch.setattr(
        pb,
        "_build_rhsmode1_structured_fblock_fp_tail_coupled_schur_preconditioner",
        lambda **_kwargs: sentinel,
    )

    assert (
        pb._build_rhs1_preconditioner_from_kind(
            op=_op(with_fp=True),
            rhs1_precond_kind="structured_fblock_fp_tail_coupled_schur",
        )
        is sentinel
    )


def test_structured_fblock_jacobi_preconditioner_builds_complete_full_vector_action() -> None:
    nml = read_sfincs_input("tests/ref/quick_2species_FPCollisions_noEr.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.5)
    precond = pb._build_rhsmode1_structured_fblock_jacobi_preconditioner(op=op)
    metadata = getattr(precond, "_sfincs_jax_structured_fblock_metadata")

    assert metadata["selected"] is True
    assert metadata["reason"] == "complete"
    assert metadata["assembly"]["is_complete"] is True
    assert metadata["assembly"]["nnz_blocks"] > 0

    rhs = jnp.linspace(-0.25, 0.75, op.total_size, dtype=jnp.float64)
    got = precond(rhs)

    assert got.shape == rhs.shape
    np.testing.assert_allclose(np.asarray(got[op.f_size :]), np.asarray(rhs[op.f_size :]))
    assert np.all(np.isfinite(np.asarray(got)))


def test_structured_fblock_angular_jacobi_preconditioner_builds_complete_full_vector_action() -> None:
    nml = read_sfincs_input("tests/ref/pas_1species_PAS_noEr_small.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.5)
    precond = pb._build_rhsmode1_structured_fblock_angular_jacobi_preconditioner(op=op)
    metadata = getattr(precond, "_sfincs_jax_structured_fblock_metadata")

    assert metadata["selected"] is True
    assert metadata["reason"] == "complete"
    assert metadata["line_kind"] == "fixed_species_x_l_angular"
    assert metadata["blocks_per_line"] == op.n_theta
    assert metadata["factor"]["block_size"] == op.n_theta * op.n_zeta

    rhs = jnp.linspace(-0.25, 0.75, op.total_size, dtype=jnp.float64)
    got = precond(rhs)

    assert got.shape == rhs.shape
    np.testing.assert_allclose(np.asarray(got[op.f_size :]), np.asarray(rhs[op.f_size :]))
    assert np.all(np.isfinite(np.asarray(got)))


def test_structured_fblock_xi_angular_jacobi_preconditioner_builds_complete_full_vector_action() -> None:
    nml = read_sfincs_input("tests/ref/pas_1species_PAS_noEr_small.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.5)
    precond = pb._build_rhsmode1_structured_fblock_xi_angular_jacobi_preconditioner(op=op)
    metadata = getattr(precond, "_sfincs_jax_structured_fblock_metadata")

    assert metadata["selected"] is True
    assert metadata["reason"] == "complete"
    assert metadata["line_kind"] == "fixed_species_x_velocity_angular"
    assert metadata["blocks_per_line"] == op.n_xi * op.n_theta
    assert metadata["factor"]["block_size"] == op.n_xi * op.n_theta * op.n_zeta

    rhs = jnp.linspace(-0.25, 0.75, op.total_size, dtype=jnp.float64)
    got = precond(rhs)

    assert got.shape == rhs.shape
    np.testing.assert_allclose(np.asarray(got[op.f_size :]), np.asarray(rhs[op.f_size :]))
    assert np.all(np.isfinite(np.asarray(got)))


def test_structured_fblock_xi_angular_jacobi_preconditioner_respects_block_guard(monkeypatch) -> None:
    nml = read_sfincs_input("tests/ref/pas_1species_PAS_noEr_small.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.5)
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_STRUCTURED_FBLOCK_XI_ANGULAR_MAX_BLOCK_SIZE", "1")

    with pytest.raises(MemoryError, match="block too large"):
        pb._build_rhsmode1_structured_fblock_xi_angular_jacobi_preconditioner(op=op)


def test_structured_fblock_fp_radial_jacobi_preconditioner_builds_complete_full_vector_action() -> None:
    nml = read_sfincs_input("tests/ref/quick_2species_FPCollisions_noEr.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.5)
    precond = pb._build_rhsmode1_structured_fblock_fp_radial_jacobi_preconditioner(op=op)
    metadata = getattr(precond, "_sfincs_jax_structured_fblock_metadata")

    assert metadata["selected"] is True
    assert metadata["reason"] == "complete"
    assert metadata["line_kind"] == "fixed_l_theta_species_x_zeta"
    assert metadata["blocks_per_group"] == op.n_species * op.n_x
    assert metadata["n_groups"] == op.n_xi * op.n_theta
    assert metadata["factor"]["kind"] == "grouped_block_diagonal"
    assert metadata["factor"]["block_size"] == op.n_species * op.n_x * op.n_zeta

    rhs = jnp.linspace(-0.25, 0.75, op.total_size, dtype=jnp.float64)
    got = precond(rhs)

    assert got.shape == rhs.shape
    np.testing.assert_allclose(np.asarray(got[op.f_size :]), np.asarray(rhs[op.f_size :]))
    assert np.all(np.isfinite(np.asarray(got)))


def test_structured_fblock_fp_radial_jacobi_preconditioner_respects_factor_guard(monkeypatch) -> None:
    nml = read_sfincs_input("tests/ref/quick_2species_FPCollisions_noEr.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.5)
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_STRUCTURED_FBLOCK_FP_RADIAL_MAX_FACTOR_NBYTES", "1")

    with pytest.raises(MemoryError, match="factor too large"):
        pb._build_rhsmode1_structured_fblock_fp_radial_jacobi_preconditioner(op=op)


def test_structured_fblock_fp_radial_jacobi_requires_fp_term() -> None:
    nml = read_sfincs_input("tests/ref/pas_1species_PAS_noEr_small.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.5)

    with pytest.raises(NotImplementedError, match="requires an FP collision term"):
        pb._build_rhsmode1_structured_fblock_fp_radial_jacobi_preconditioner(op=op)


def test_structured_fblock_fp_lowmode_schur_preconditioner_respects_coarse_guard(monkeypatch) -> None:
    nml = read_sfincs_input("tests/ref/fp_1species_FPCollisions_noEr_tiny_withPhi1_inCollision.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.5)
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_STRUCTURED_FBLOCK_FP_LOWMODE_MAX_COARSE", "1")

    with pytest.raises(MemoryError, match="coarse space too large"):
        pb._build_rhsmode1_structured_fblock_fp_lowmode_schur_preconditioner(op=op)


def test_structured_fblock_fp_lowmode_schur_truncates_optional_features(monkeypatch) -> None:
    nml = read_sfincs_input("tests/ref/fp_1species_FPCollisions_noEr_tiny_withPhi1_inCollision.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.5)
    stride = int(op.n_species) * int(op.n_x) * int(op.n_xi)
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_STRUCTURED_FBLOCK_FP_LOWMODE_MAX_COARSE", str(4 * stride))

    precond = pb._build_rhsmode1_structured_fblock_fp_lowmode_schur_preconditioner(op=op)
    metadata = getattr(precond, "_sfincs_jax_structured_fblock_metadata")

    feature_selection = metadata["coarse_feature_selection"]
    assert feature_selection["truncated_features"] is True
    assert feature_selection["requested_features"] == 5
    assert feature_selection["retained_features"] == 4
    assert metadata["coarse"]["n_coarse"] == 4 * stride
    assert metadata["coarse"]["basis_storage_nbytes"] == 0


def test_structured_fblock_xi_angular_jacobi_reduces_dke_residual_vs_block() -> None:
    nml = read_sfincs_input("tests/ref/pas_1species_PAS_noEr_small.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.5)
    rhs = jnp.zeros((op.total_size,), dtype=jnp.float64)
    rhs = rhs.at[: op.f_size].set(jnp.linspace(-0.35, 0.65, op.f_size, dtype=jnp.float64))

    block_precond = pb._build_rhsmode1_structured_fblock_jacobi_preconditioner(op=op)
    xi_angular_precond = pb._build_rhsmode1_structured_fblock_xi_angular_jacobi_preconditioner(op=op)

    def _dke_ratio(precond):
        candidate = precond(rhs)
        residual = rhs[: op.f_size] - apply_v3_full_system_operator(op, candidate)[: op.f_size]
        return float(jnp.linalg.norm(residual) / jnp.linalg.norm(rhs[: op.f_size]))

    block_ratio = _dke_ratio(block_precond)
    xi_angular_ratio = _dke_ratio(xi_angular_precond)

    assert block_ratio > 1.0e-3
    assert xi_angular_ratio < 1.0e-7
    assert xi_angular_ratio < block_ratio * 1.0e-4


def test_structured_fblock_fp_radial_jacobi_reduces_fp_dke_residual_vs_xi_angular() -> None:
    nml = read_sfincs_input("tests/ref/quick_2species_FPCollisions_noEr.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.5)
    rhs = jnp.zeros((op.total_size,), dtype=jnp.float64)
    rhs = rhs.at[: op.f_size].set(jnp.linspace(-0.35, 0.65, op.f_size, dtype=jnp.float64))

    xi_angular_precond = pb._build_rhsmode1_structured_fblock_xi_angular_jacobi_preconditioner(op=op)
    fp_radial_precond = pb._build_rhsmode1_structured_fblock_fp_radial_jacobi_preconditioner(op=op)

    def _dke_ratio(precond):
        candidate = precond(rhs)
        residual = rhs[: op.f_size] - apply_v3_full_system_operator(op, candidate)[: op.f_size]
        return float(jnp.linalg.norm(residual) / jnp.linalg.norm(rhs[: op.f_size]))

    xi_angular_ratio = _dke_ratio(xi_angular_precond)
    fp_radial_ratio = _dke_ratio(fp_radial_precond)

    assert xi_angular_ratio > 1.0
    assert fp_radial_ratio < 5.0e-2
    assert fp_radial_ratio < xi_angular_ratio * 1.0e-2


def test_structured_fblock_fp_radial_jacobi_reuses_same_shape_cache() -> None:
    nml = read_sfincs_input("tests/ref/quick_2species_FPCollisions_noEr.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.5)
    rhs = jnp.zeros((op.total_size,), dtype=jnp.float64)
    rhs = rhs.at[: op.f_size].set(jnp.linspace(-0.35, 0.65, op.f_size, dtype=jnp.float64))
    _RHSMODE1_STRUCTURED_FBLOCK_PRECOND_CACHE.clear()

    first = pb._build_rhsmode1_structured_fblock_fp_radial_jacobi_preconditioner(op=op)
    second = pb._build_rhsmode1_structured_fblock_fp_radial_jacobi_preconditioner(op=op)
    first_metadata = getattr(first, "_sfincs_jax_structured_fblock_metadata")
    second_metadata = getattr(second, "_sfincs_jax_structured_fblock_metadata")

    assert first_metadata["cache_hit"] is False
    assert second_metadata["cache_hit"] is True
    np.testing.assert_allclose(np.asarray(second(rhs)), np.asarray(first(rhs)), rtol=1.0e-12, atol=1.0e-12)


def test_structured_fblock_fp_lowmode_schur_reduces_fp_dke_residual_vs_radial() -> None:
    nml = read_sfincs_input("tests/ref/fp_1species_FPCollisions_noEr_tiny_withPhi1_inCollision.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.5)
    rhs = jnp.zeros((op.total_size,), dtype=jnp.float64)
    rhs = rhs.at[: op.f_size].set(jnp.linspace(-0.35, 0.65, op.f_size, dtype=jnp.float64))

    fp_radial_precond = pb._build_rhsmode1_structured_fblock_fp_radial_jacobi_preconditioner(op=op)
    lowmode_precond = pb._build_rhsmode1_structured_fblock_fp_lowmode_schur_preconditioner(op=op)
    metadata = getattr(lowmode_precond, "_sfincs_jax_structured_fblock_metadata")

    assert metadata["line_kind"] == "fp_radial_plus_low_angular_galerkin"
    assert metadata["coarse"]["kind"] == "matrix_free_galerkin_residual_correction"
    assert metadata["coarse"]["n_coarse"] == op.n_species * op.n_x * op.n_xi * 5
    assert metadata["coarse"]["basis_storage_nbytes"] == 0

    def _dke_ratio(precond):
        candidate = precond(rhs)
        residual = rhs[: op.f_size] - apply_v3_full_system_operator(op, candidate)[: op.f_size]
        return float(jnp.linalg.norm(residual) / jnp.linalg.norm(rhs[: op.f_size]))

    fp_radial_ratio = _dke_ratio(fp_radial_precond)
    lowmode_ratio = _dke_ratio(lowmode_precond)

    assert fp_radial_ratio < 1.0e-1
    assert lowmode_ratio < fp_radial_ratio
    assert lowmode_ratio < 6.0e-2


def test_structured_fblock_fp_moment_schur_is_compact_and_finite() -> None:
    nml = read_sfincs_input("tests/ref/fp_1species_FPCollisions_noEr_tiny_withPhi1_inCollision.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.5)
    rhs = jnp.zeros((op.total_size,), dtype=jnp.float64)
    rhs = rhs.at[: op.f_size].set(jnp.linspace(-0.35, 0.65, op.f_size, dtype=jnp.float64))

    fp_radial_precond = pb._build_rhsmode1_structured_fblock_fp_radial_jacobi_preconditioner(op=op)
    lowmode_precond = pb._build_rhsmode1_structured_fblock_fp_lowmode_schur_preconditioner(op=op)
    moment_precond = pb._build_rhsmode1_structured_fblock_fp_moment_schur_preconditioner(op=op)
    moment_metadata = getattr(moment_precond, "_sfincs_jax_structured_fblock_metadata")
    lowmode_metadata = getattr(lowmode_precond, "_sfincs_jax_structured_fblock_metadata")

    assert moment_metadata["line_kind"] == "fp_radial_plus_low_x_xi_angular_moment_galerkin"
    assert moment_metadata["coarse"]["kind"] == "matrix_free_galerkin_residual_correction"
    assert moment_metadata["coarse"]["basis_storage_nbytes"] == 0
    assert moment_metadata["coarse"]["n_coarse"] < lowmode_metadata["coarse"]["n_coarse"]
    assert moment_metadata["coarse_moment_selection"]["x_moments_retained"] == 2
    assert moment_metadata["coarse_moment_selection"]["xi_moments_retained"] == 2

    def _dke_ratio(precond):
        candidate = precond(rhs)
        residual = rhs[: op.f_size] - apply_v3_full_system_operator(op, candidate)[: op.f_size]
        return float(jnp.linalg.norm(residual) / jnp.linalg.norm(rhs[: op.f_size]))

    fp_radial_ratio = _dke_ratio(fp_radial_precond)
    moment_ratio = _dke_ratio(moment_precond)

    assert np.isfinite(moment_ratio)
    assert moment_ratio < fp_radial_ratio


def test_structured_fblock_fp_coupled_moment_schur_reduces_full_residual() -> None:
    nml = read_sfincs_input("tests/ref/fp_1species_FPCollisions_noEr_tiny_withPhi1_inCollision.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.5)
    rhs = jnp.zeros((op.total_size,), dtype=jnp.float64)
    rhs = rhs.at[: op.f_size].set(jnp.linspace(-0.35, 0.65, op.f_size, dtype=jnp.float64))

    fp_radial_precond = pb._build_rhsmode1_structured_fblock_fp_radial_jacobi_preconditioner(op=op)
    coupled_precond = pb._build_rhsmode1_structured_fblock_fp_coupled_moment_schur_preconditioner(op=op)
    metadata = getattr(coupled_precond, "_sfincs_jax_structured_fblock_metadata")

    assert metadata["line_kind"] == "fp_radial_plus_coupled_tail_moment_galerkin"
    assert metadata["coarse"]["kind"] == "matrix_free_galerkin_residual_correction"
    assert metadata["coarse_coupled_selection"]["tail_policy"] == "all_tail"
    assert metadata["coarse_coupled_selection"]["tail_count"] == op.total_size - op.f_size

    def _ratios(precond):
        candidate = precond(rhs)
        residual = rhs - apply_v3_full_system_operator(op, candidate)
        dke_ratio = float(jnp.linalg.norm(residual[: op.f_size]) / jnp.linalg.norm(rhs[: op.f_size]))
        full_ratio = float(jnp.linalg.norm(residual) / jnp.linalg.norm(rhs))
        return dke_ratio, full_ratio

    radial_dke, radial_full = _ratios(fp_radial_precond)
    coupled_dke, coupled_full = _ratios(coupled_precond)

    assert np.isfinite(coupled_dke)
    assert np.isfinite(coupled_full)
    assert coupled_full < radial_full * 1.0e-2
    assert coupled_dke > radial_dke


def test_structured_fblock_fp_tail_coupled_schur_preserves_f_correction() -> None:
    nml = read_sfincs_input("tests/ref/fp_1species_FPCollisions_noEr_tiny_withPhi1_inCollision.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.5)
    rhs = jnp.zeros((op.total_size,), dtype=jnp.float64)
    rhs = rhs.at[: op.f_size].set(jnp.linspace(-0.35, 0.65, op.f_size, dtype=jnp.float64))

    fp_radial_precond = pb._build_rhsmode1_structured_fblock_fp_radial_jacobi_preconditioner(op=op)
    tail_precond = pb._build_rhsmode1_structured_fblock_fp_tail_coupled_schur_preconditioner(op=op)
    metadata = getattr(tail_precond, "_sfincs_jax_structured_fblock_metadata")

    assert metadata["line_kind"] == "fp_radial_plus_tail_coupled_minres"
    assert metadata["coarse"]["kind"] == "matrix_free_least_squares_residual_correction"
    assert metadata["coarse"]["solver_kind"] == "precomputed_normal_inverse"
    assert metadata["coarse_tail_selection"]["tail_policy"] == "all_tail"
    assert metadata["coarse_tail_selection"]["tail_count"] == op.total_size - op.f_size

    radial_candidate = fp_radial_precond(rhs)
    tail_candidate = tail_precond(rhs)
    tail_residual = rhs - apply_v3_full_system_operator(op, tail_candidate)
    tail_dke_ratio = float(jnp.linalg.norm(tail_residual[: op.f_size]) / jnp.linalg.norm(rhs[: op.f_size]))
    tail_full_ratio = float(jnp.linalg.norm(tail_residual) / jnp.linalg.norm(rhs))

    np.testing.assert_allclose(
        np.asarray(tail_candidate[: op.f_size]),
        np.asarray(radial_candidate[: op.f_size]),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert np.isfinite(tail_dke_ratio)
    assert np.isfinite(tail_full_ratio)


def test_structured_fblock_preconditioner_env_is_honored_for_phi1_solve(monkeypatch) -> None:
    nml = read_sfincs_input("tests/ref/fp_1species_FPCollisions_noEr_tiny_withPhi1_inCollision.input.namelist")
    logs: list[str] = []
    monkeypatch.setenv("SFINCS_JAX_FORTRAN_STDOUT", "0")
    monkeypatch.setenv("SFINCS_JAX_SOLVER_ITER_STATS", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_STRONG_PRECOND", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DENSE_FALLBACK_MAX", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_BICGSTAB_PRECOND", "off")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PRECONDITIONER", "structured_fblock_fp_radial_jacobi")

    result = vd.solve_v3_full_system_linear_gmres(
        nml=nml,
        tol=1.0e-8,
        restart=20,
        maxiter=40,
        solve_method="incremental",
        identity_shift=0.5,
        emit=lambda level, msg: logs.append(str(msg)) if level <= 1 else None,
    )

    assert np.isfinite(float(result.residual_norm))
    assert result.metadata is not None
    assert result.metadata["structured_fblock_preconditioner_enabled"] is True
    assert any(
        "building RHSMode=1 preconditioner=structured_fblock_fp_radial_jacobi" in msg
        for msg in logs
    )


def test_matvec_submatrix_uses_unsharded_operator_inside_vmap(monkeypatch) -> None:
    def _cached_matvec(*_args, **_kwargs):
        raise AssertionError("preconditioner submatrix assembly must not enter cached sharded matvec")

    def _unsharded_matvec(_op, vector, *, include_jacobian_terms=True, allow_sharding=True):
        assert include_jacobian_terms is True
        assert allow_sharding is False
        return 2.0 * vector + jnp.arange(vector.shape[0], dtype=vector.dtype)

    monkeypatch.setattr(vd, "apply_v3_full_system_operator_cached", _cached_matvec)
    monkeypatch.setattr(sparse_direct, "apply_v3_full_system_operator", _unsharded_matvec)

    submatrix = sparse_direct.matvec_submatrix(
        SimpleNamespace(),
        col_idx=np.asarray([0, 2], dtype=np.int32),
        row_idx=np.asarray([0, 2], dtype=np.int32),
        total_size=4,
        chunk_cols=2,
    )

    np.testing.assert_allclose(submatrix, np.asarray([[2.0, 2.0], [0.0, 4.0]]))


def test_rhs1_dkes_gmres_budget_respects_explicit_limits() -> None:
    restart, maxiter, restart_defaulted, maxiter_defaulted = vd._rhs1_dkes_gmres_budget(
        restart=20,
        maxiter=20,
        restart_forced=True,
        maxiter_forced=True,
        restart_cap_env="100",
    )

    assert (restart, maxiter) == (20, 20)
    assert restart_defaulted is False
    assert maxiter_defaulted is False


def test_rhs1_dkes_gmres_budget_applies_defaults_when_unforced() -> None:
    restart, maxiter, restart_defaulted, maxiter_defaulted = vd._rhs1_dkes_gmres_budget(
        restart=20,
        maxiter=20,
        restart_forced=False,
        maxiter_forced=False,
        restart_cap_env="90",
    )

    assert (restart, maxiter) == (80, 600)
    assert restart_defaulted is True
    assert maxiter_defaulted is True


def test_rhs1_dkes_gmres_budget_caps_unforced_restart() -> None:
    restart, maxiter, restart_defaulted, maxiter_defaulted = vd._rhs1_dkes_gmres_budget(
        restart=200,
        maxiter=None,
        restart_forced=False,
        maxiter_forced=False,
        restart_cap_env="100",
    )

    assert (restart, maxiter) == (100, 600)
    assert restart_defaulted is True
    assert maxiter_defaulted is True


def test_rhs1_pas_tz_guarded_structured_levels_parse_aliases() -> None:
    assert vd._rhs1_pas_tz_guarded_structured_levels("") == ()
    assert vd._rhs1_pas_tz_guarded_structured_levels("off") == ()
    assert vd._rhs1_pas_tz_guarded_structured_levels("structured") == ("xmg", "collision")
    assert vd._rhs1_pas_tz_guarded_structured_levels("x+coll+x") == ("xmg", "collision")
    assert vd._rhs1_pas_tz_guarded_structured_levels("unknown,collision_diag") == ("collision",)


def test_ksp_iteration_solver_label_reports_lgmres_method() -> None:
    assert vd._ksp_iteration_solver_label(solver_kind="gmres", solve_method="incremental") == "gmres"
    assert vd._ksp_iteration_solver_label(solver_kind="gmres", solve_method="lgmres") == "lgmres"
    assert vd._ksp_iteration_solver_label(solver_kind="bicgstab", solve_method="lgmres") == "bicgstab"
