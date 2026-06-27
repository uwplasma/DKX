from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from sfincs_jax.solver import GMRESSolveResult
import sfincs_jax.problems.profile_solve as profile_solve
from sfincs_jax.problems.profile_residual import (
    compose_multilevel_residual_correction_preconditioner,
    safe_preconditioner,
)


def test_diag_only_preserves_only_point_coupling() -> None:
    mat = jnp.asarray(
        [
            [3.0, 1.0, -2.0],
            [4.0, 5.0, 6.0],
            [-1.0, 2.0, 7.0],
        ],
        dtype=jnp.float64,
    )
    reduced = np.asarray(profile_solve._diag_only(mat))
    np.testing.assert_allclose(reduced, np.diag([3.0, 5.0, 7.0]))


def test_block_diag_only_preserves_local_block_coupling() -> None:
    mat = jnp.asarray(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ],
        dtype=jnp.float64,
    )
    reduced = np.asarray(profile_solve._block_diag_only(mat, block=2))
    expected = np.asarray(
        [
            [1.0, 2.0, 0.0, 0.0],
            [5.0, 6.0, 0.0, 0.0],
            [0.0, 0.0, 11.0, 12.0],
            [0.0, 0.0, 15.0, 16.0],
        ]
    )
    np.testing.assert_allclose(reduced, expected)
    np.testing.assert_allclose(np.asarray(profile_solve._block_diag_only(mat, block=1)), np.asarray(profile_solve._diag_only(mat)))


def test_dd_core_patch_ranges_cover_domain_with_overlap() -> None:
    ranges = profile_solve._dd_core_patch_ranges(n=10, block=4, overlap=1)
    assert ranges == [
        (0, 4, 0, 5),
        (4, 8, 3, 9),
        (8, 10, 7, 10),
    ]


def test_dd_core_patch_ranges_clamp_to_domain_edges() -> None:
    ranges = profile_solve._dd_core_patch_ranges(n=5, block=2, overlap=10)
    assert ranges == [
        (0, 2, 0, 5),
        (2, 4, 0, 5),
        (4, 5, 0, 5),
    ]


def test_rhs1_dd_coarse_level_count_respects_env_override_and_invalid(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SCHWARZ_COARSE_LEVELS", "3")
    assert profile_solve._rhs1_dd_coarse_level_count(n_dev=2) == 3

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SCHWARZ_COARSE_LEVELS", "bad")
    assert profile_solve._rhs1_dd_coarse_level_count(n_dev=8) == 1


def test_rhs1_dd_coarse_block_sizes_stop_at_global_extent() -> None:
    coarse_blocks = profile_solve._rhs1_dd_coarse_block_sizes(n=25, block=12, overlap=4, levels=3)
    assert coarse_blocks[-1] == 25
    assert coarse_blocks == (20, 25)


def test_compose_multilevel_residual_correction_steps_zero_returns_base() -> None:
    calls = {"base": 0, "coarse": 0}

    def base(v: jnp.ndarray) -> jnp.ndarray:
        calls["base"] += 1
        return 2.0 * v

    def coarse(v: jnp.ndarray) -> jnp.ndarray:
        calls["coarse"] += 1
        return -v

    assert (
        profile_solve._compose_multilevel_residual_correction_preconditioner
        is compose_multilevel_residual_correction_preconditioner
    )
    precond = compose_multilevel_residual_correction_preconditioner(
        base=base,
        coarse_levels=(coarse,),
        matvec=lambda v: v,
        steps=0,
    )
    out = np.asarray(precond(jnp.asarray([1.0, -2.0], dtype=jnp.float64)))
    np.testing.assert_allclose(out, np.asarray([2.0, -4.0]))
    assert calls == {"base": 1, "coarse": 0}


def test_safe_preconditioner_zeroes_nonfinite_and_clips() -> None:
    assert profile_solve._safe_preconditioner is safe_preconditioner
    precond = safe_preconditioner(
        lambda v: jnp.asarray([jnp.inf, jnp.nan, -jnp.inf, 9.0 * v[0]], dtype=jnp.float64),
        clip=5.0,
    )
    out = np.asarray(precond(jnp.asarray([2.0], dtype=jnp.float64)))
    np.testing.assert_allclose(out, np.asarray([0.0, 0.0, 0.0, 5.0]))


def test_gmres_result_is_finite_detects_nonfinite_state() -> None:
    good = GMRESSolveResult(
        x=jnp.asarray([1.0, -1.0], dtype=jnp.float64),
        residual_norm=jnp.asarray(1.0e-9, dtype=jnp.float64),
    )
    bad_x = GMRESSolveResult(
        x=jnp.asarray([1.0, jnp.nan], dtype=jnp.float64),
        residual_norm=jnp.asarray(1.0e-9, dtype=jnp.float64),
    )
    bad_r = GMRESSolveResult(
        x=jnp.asarray([1.0, -1.0], dtype=jnp.float64),
        residual_norm=jnp.asarray(jnp.inf, dtype=jnp.float64),
    )
    assert profile_solve._gmres_result_is_finite(good)
    assert not profile_solve._gmres_result_is_finite(bad_x)
    assert not profile_solve._gmres_result_is_finite(bad_r)
