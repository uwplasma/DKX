from __future__ import annotations

import jax.numpy as jnp

from sfincs_jax.solver import GMRESSolveResult
from sfincs_jax.solver import block_gmres_result_ready, gmres_result_is_finite


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

    assert gmres_result_is_finite(good)
    assert not gmres_result_is_finite(bad_x)
    assert not gmres_result_is_finite(bad_r)


def test_block_gmres_result_ready_returns_same_result() -> None:
    result = GMRESSolveResult(
        x=jnp.asarray([1.0], dtype=jnp.float64),
        residual_norm=jnp.asarray(0.0, dtype=jnp.float64),
    )

    assert block_gmres_result_ready(result) is result
