from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp

from sfincs_jax.problems.profile_response.sparse.qi import (
    MatrixFreeQIDeviceSeedContext,
    attempt_matrixfree_qi_device_seed,
    attempt_matrixfree_qi_device_seed_if_requested,
    build_matrixfree_qi_device_seed_setup,
)
from sfincs_jax.solver import GMRESSolveResult


def _context(metadata: dict[str, object]) -> MatrixFreeQIDeviceSeedContext:
    return MatrixFreeQIDeviceSeedContext(
        op=object(),
        active_size=2,
        target_reduced=1.0e-8,
        mv_reduced=lambda x: x,
        rhs_reduced=jnp.asarray([1.0, 2.0], dtype=jnp.float64),
        emit=None,
        timer_elapsed_s=lambda: 0.0,
        rhsmode1_general_metadata=metadata,
    )


def test_matrixfree_qi_device_seed_gate_is_disabled_by_default() -> None:
    metadata: dict[str, object] = {}
    result = GMRESSolveResult(
        x=jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        residual_norm=jnp.asarray(1.0, dtype=jnp.float64),
    )

    out = attempt_matrixfree_qi_device_seed(result, hook="unit", context=_context(metadata))

    assert out is result
    assert metadata == {}


def test_matrixfree_qi_device_seed_skips_when_already_converged(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE", "1")
    metadata: dict[str, object] = {}
    result = GMRESSolveResult(
        x=jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        residual_norm=jnp.asarray(1.0e-10, dtype=jnp.float64),
    )

    out = attempt_matrixfree_qi_device_seed(result, hook="unit", context=_context(metadata))

    assert out is result
    assert metadata == {}


def test_matrixfree_qi_device_seed_setup_resolves_driver_gates(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_EARLY", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_SKIP_STRONG", "1")
    metadata: dict[str, object] = {}

    setup = build_matrixfree_qi_device_seed_setup(
        op=SimpleNamespace(rhs_mode=1),
        active_size=2,
        target_reduced=1.0e-8,
        mv_reduced=lambda x: x,
        rhs_reduced=jnp.asarray([1.0, 2.0], dtype=jnp.float64),
        emit=None,
        timer_elapsed_s=lambda: 1.25,
        rhsmode1_general_metadata=metadata,
    )

    assert setup.early_enabled is True
    assert setup.skip_strong is True
    assert setup.pre_sparse_enabled is True
    assert setup.context.active_size == 2
    assert setup.context.elapsed_s() == 1.25


def test_matrixfree_qi_device_seed_if_requested_keeps_disabled_path_noop() -> None:
    metadata: dict[str, object] = {}
    setup = build_matrixfree_qi_device_seed_setup(
        op=SimpleNamespace(rhs_mode=1),
        active_size=2,
        target_reduced=1.0e-8,
        mv_reduced=lambda x: x,
        rhs_reduced=jnp.asarray([1.0, 2.0], dtype=jnp.float64),
        emit=None,
        timer_elapsed_s=lambda: 0.0,
        rhsmode1_general_metadata=metadata,
    )
    result = GMRESSolveResult(
        x=jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        residual_norm=jnp.asarray(1.0, dtype=jnp.float64),
    )

    attempt = attempt_matrixfree_qi_device_seed_if_requested(
        result,
        hook="unit",
        setup=setup,
        enabled=False,
    )

    assert attempt.result is result
    assert attempt.attempted is False
    assert attempt.improved is False
    assert metadata == {}
