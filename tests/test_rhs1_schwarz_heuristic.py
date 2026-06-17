from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import jax.numpy as jnp

from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.problems.profile_response.residual import apply_subspace_minres_correction
import sfincs_jax.v3_driver as vd


def test_rhs1_auto_prefers_theta_schwarz_when_sharded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto RHSMode=1 preconditioner should pick Schwarz on sharded large-system path."""
    input_path = Path(__file__).parent / "ref" / "pas_1species_PAS_noEr_tiny_scheme1.input.namelist"
    assert input_path.exists()
    nml = read_sfincs_input(input_path)

    logs: list[str] = []

    def emit(level: int, msg: str) -> None:
        logs.append(msg)

    monkeypatch.setenv("SFINCS_JAX_FORTRAN_STDOUT", "0")
    monkeypatch.setenv("SFINCS_JAX_SOLVER_ITER_STATS", "0")
    monkeypatch.setenv("SFINCS_JAX_MATVEC_SHARD_AXIS", "theta")
    monkeypatch.setenv("SFINCS_JAX_AUTO_SHARD", "0")
    monkeypatch.setenv("SFINCS_JAX_GMRES_DISTRIBUTED", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SCHWARZ_AUTO_MIN", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_STRONG_PRECOND", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DENSE_FALLBACK_MAX", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_TZ_MAX", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_TZ_PRECOND_MAX", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPECIES_BLOCK_MAX", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_COLLISION_PRECOND_MIN", "1000000000")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_TZ_SCHWARZ_MAX_PATCH_UNKNOWNS", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_TZ_SCHWARZ_MAX_INVERSE_ENTRIES", "0")
    monkeypatch.setattr(vd.jax, "device_count", lambda: 2)

    res = vd.solve_v3_full_system_linear_gmres(nml=nml, tol=1e-8, emit=emit)
    assert np.isfinite(float(res.residual_norm))
    assert any("building RHSMode=1 preconditioner=theta_schwarz" in msg for msg in logs)


def test_rhs1_auto_preserves_auto_solver_selection_on_multidevice_sharded_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = Path(__file__).parent / "ref" / "pas_1species_PAS_noEr_tiny_scheme1.input.namelist"
    assert input_path.exists()
    nml = read_sfincs_input(input_path)

    logs: list[str] = []

    monkeypatch.setenv("SFINCS_JAX_FORTRAN_STDOUT", "0")
    monkeypatch.setenv("SFINCS_JAX_SOLVER_ITER_STATS", "0")
    monkeypatch.setenv("SFINCS_JAX_MATVEC_SHARD_AXIS", "theta")
    monkeypatch.setenv("SFINCS_JAX_AUTO_SHARD", "0")
    monkeypatch.setenv("SFINCS_JAX_GMRES_DISTRIBUTED", "theta")
    monkeypatch.setenv("SFINCS_JAX_GMRES_DISTRIBUTED_ALLOW_ACCELERATOR", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_STRONG_PRECOND", "0")
    monkeypatch.setattr(vd.jax, "device_count", lambda: 2)
    monkeypatch.setattr(vd.jax, "local_device_count", lambda: 2)
    monkeypatch.setattr(vd.jax, "default_backend", lambda: "gpu")

    res = vd.solve_v3_full_system_linear_gmres(
        nml=nml,
        tol=1e-8,
        solve_method="auto",
        differentiable=False,
        emit=lambda _level, msg: logs.append(str(msg)),
    )

    assert np.isfinite(float(res.residual_norm))
    assert any("preserving auto solver selection for multi-device sharded axis=theta" in msg for msg in logs)


def test_pas_tz_memory_estimate_flags_large_case() -> None:
    class _Collisionless:
        n_xi_for_x = np.full((20,), 14, dtype=np.int32)

    class _FBlock:
        collisionless = _Collisionless()
        pas = object()
        fp = None
        fp_phi1 = None
        exb_theta = None
        exb_zeta = None
        magdrift_theta = None
        magdrift_zeta = None
        magdrift_xidot = None
        er_xdot = None
        er_xidot = None

    class _Op:
        rhs_mode = 1
        n_species = 1
        n_x = 20
        n_xi = 14
        n_theta = 127
        n_zeta = 127
        fblock = _FBlock()

    estimate = vd._estimate_rhs1_pas_tz_build_bytes(_Op())
    assert estimate > 100 * 2**30
    assert not vd._pas_tz_preconditioner_memory_safe(_Op())


def test_pas_tz_builder_falls_back_to_theta_schwarz_when_memory_unsafe(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Collisionless:
        n_xi_for_x = np.full((20,), 14, dtype=np.int32)

    class _FBlock:
        collisionless = _Collisionless()
        pas = object()
        fp = None
        fp_phi1 = None
        exb_theta = None
        exb_zeta = None
        magdrift_theta = None
        magdrift_zeta = None
        magdrift_xidot = None
        er_xdot = None
        er_xidot = None

    class _Op:
        rhs_mode = 1
        n_species = 1
        n_x = 20
        n_xi = 14
        n_theta = 127
        n_zeta = 127
        fblock = _FBlock()

    sentinel = object()
    monkeypatch.setenv("SFINCS_JAX_FORTRAN_STDOUT", "0")
    monkeypatch.setenv("SFINCS_JAX_MATVEC_SHARD_AXIS", "theta")
    monkeypatch.setenv("SFINCS_JAX_AUTO_SHARD", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_TZ_SCHWARZ_MAX_PATCH_UNKNOWNS", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_TZ_SCHWARZ_MAX_INVERSE_ENTRIES", "0")
    monkeypatch.setattr(vd.jax, "device_count", lambda: 2)
    monkeypatch.setattr(vd, "_estimate_rhs1_pas_tz_build_bytes", lambda _op: 10 * 2**30)
    monkeypatch.setattr(vd, "_rhs1_pas_tz_max_bytes", lambda: 2 * 2**30)
    monkeypatch.setattr(vd, "_build_rhsmode1_theta_schwarz_preconditioner", lambda **_kwargs: sentinel)

    assert vd._build_rhsmode1_pas_tz_preconditioner(op=_Op()) is sentinel


def test_rhs1_dd_auto_block_size_spans_more_than_one_local_shard() -> None:
    block = vd._rhs1_dd_auto_block_size(
        n=31,
        n_dev=8,
        sum_nxi=144,
        dof_target=1200,
    )
    assert block == 12
    assert block > 4


def test_rhs1_dd_auto_block_size_respects_global_extent() -> None:
    block = vd._rhs1_dd_auto_block_size(
        n=31,
        n_dev=2,
        sum_nxi=144,
        dof_target=1200,
    )
    assert block == 24
    assert block <= 31


def test_rhs1_dd_coarse_block_size_widens_local_patch() -> None:
    coarse = vd._rhs1_dd_coarse_block_size(n=31, block=12, overlap=1)
    assert coarse == 20
    assert coarse > 12


def test_rhs1_dd_coarse_level_count_auto(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SCHWARZ_COARSE_LEVELS", raising=False)
    assert vd._rhs1_dd_coarse_level_count(n_dev=2) == 0
    assert vd._rhs1_dd_coarse_level_count(n_dev=4) == 1
    assert vd._rhs1_dd_coarse_level_count(n_dev=8) == 2


def test_rhs1_dd_coarse_block_sizes_build_multiple_levels() -> None:
    coarse_blocks = vd._rhs1_dd_coarse_block_sizes(n=63, block=12, overlap=1, levels=2)
    assert coarse_blocks == (20, 30)


def test_compose_residual_correction_preconditioner_matches_one_step() -> None:
    def base(v):
        return 0.5 * v

    def coarse(v):
        return 0.25 * v

    def matvec(v):
        return 2.0 * v

    precond = vd._compose_residual_correction_preconditioner(
        base=base,
        coarse=coarse,
        matvec=matvec,
        damping=1.0,
        steps=1,
    )
    out = precond(np.array([4.0]))
    assert np.allclose(np.asarray(out), np.array([2.0]))


def test_compose_multilevel_residual_correction_preconditioner_applies_levels_in_order() -> None:
    def base(v):
        return jnp.zeros_like(v)

    def coarse_1(v):
        return 0.25 * v

    def coarse_2(v):
        return 0.125 * v

    def matvec(v):
        return 2.0 * v

    precond = vd._compose_multilevel_residual_correction_preconditioner(
        base=base,
        coarse_levels=(coarse_1, coarse_2),
        matvec=matvec,
        damping=1.0,
        steps=1,
    )
    out = precond(np.array([4.0]))
    assert np.allclose(np.asarray(out), np.array([1.25]))


def test_compose_multilevel_minres_correction_rejects_zero_direction() -> None:
    def base(v):
        return jnp.zeros_like(v)

    def bad_coarse(v):
        return jnp.zeros_like(v)

    def matvec(v):
        return 2.0 * v

    precond = vd._compose_multilevel_minres_correction_preconditioner(
        base=base,
        coarse_levels=(bad_coarse,),
        matvec=matvec,
        alpha_clip=1.0,
        steps=1,
    )
    out = precond(np.array([4.0]))
    assert np.allclose(np.asarray(out), np.array([0.0]))


def test_compose_multilevel_minres_correction_accepts_better_direction() -> None:
    def base(v):
        return jnp.zeros_like(v)

    def coarse(v):
        return v

    def matvec(v):
        return 2.0 * v

    precond = vd._compose_multilevel_minres_correction_preconditioner(
        base=base,
        coarse_levels=(coarse,),
        matvec=matvec,
        alpha_clip=1.0,
        steps=1,
    )
    out = precond(np.array([4.0]))
    assert np.allclose(np.asarray(out), np.array([2.0]))


def test_preconditioned_minres_correction_accepts_only_residual_improvement() -> None:
    def matvec(v):
        return jnp.asarray([2.0 * v[0], 4.0 * v[1]], dtype=jnp.float64)

    def preconditioner(v):
        return jnp.asarray(v, dtype=jnp.float64)

    rhs = jnp.asarray([2.0, 4.0], dtype=jnp.float64)
    x0 = jnp.zeros((2,), dtype=jnp.float64)

    x, residual, history, alphas = vd._apply_preconditioned_minres_correction(
        matvec=matvec,
        rhs=rhs,
        x0=x0,
        preconditioner=preconditioner,
        steps=2,
    )

    assert len(alphas) >= 1
    assert float(history[-1]) < float(history[0])
    assert float(jnp.linalg.norm(residual)) == pytest.approx(float(history[-1]))
    assert float(jnp.linalg.norm(rhs - matvec(x))) == pytest.approx(float(history[-1]))


def test_subspace_minres_correction_combines_multiple_directions() -> None:
    def matvec(v):
        return jnp.asarray([2.0 * v[0], 5.0 * v[1]], dtype=jnp.float64)

    def directions(_residual):
        return (
            ("x0", jnp.asarray([1.0, 0.0], dtype=jnp.float64)),
            ("x1", jnp.asarray([0.0, 1.0], dtype=jnp.float64)),
        )

    rhs = jnp.asarray([2.0, 10.0], dtype=jnp.float64)
    x0 = jnp.zeros((2,), dtype=jnp.float64)

    assert vd._apply_subspace_minres_correction is apply_subspace_minres_correction
    x, residual, history, counts, names = apply_subspace_minres_correction(
        matvec=matvec,
        rhs=rhs,
        x0=x0,
        direction_builder=directions,
        steps=1,
        max_directions=2,
    )

    assert np.allclose(np.asarray(x), np.asarray([1.0, 2.0]))
    assert float(jnp.linalg.norm(residual)) < 1.0e-12
    assert history[-1] < history[0]
    assert counts == (2,)
    assert names == ("x0", "x1")


def test_subspace_minres_correction_rejects_nonimproving_basis() -> None:
    def matvec(v):
        return jnp.asarray([v[0], v[1]], dtype=jnp.float64)

    def directions(_residual):
        return (("zero", jnp.zeros((2,), dtype=jnp.float64)),)

    rhs = jnp.asarray([1.0, 0.0], dtype=jnp.float64)
    x, residual, history, counts, names = apply_subspace_minres_correction(
        matvec=matvec,
        rhs=rhs,
        x0=jnp.zeros((2,), dtype=jnp.float64),
        direction_builder=directions,
        steps=1,
        max_directions=4,
    )

    assert np.allclose(np.asarray(x), np.zeros((2,)))
    assert np.allclose(np.asarray(residual), np.asarray(rhs))
    assert history == (1.0,)
    assert counts == ()
    assert names == ()
