from __future__ import annotations

import jax
import numpy as np
import jax.numpy as jnp
import pytest

from sfincs_jax.grids import uniform_diff_matrices
from sfincs_jax.discretization.periodic_stencil import (
    _periodic_halo_exchange,
    _sharding_active_hint,
    apply_periodic_stencil_roll,
    apply_periodic_stencil_halo,
    apply_sparse_row_stencil_gather,
    extract_sparse_circulant_stencil,
    extract_sparse_row_stencil,
    periodic_stencil_runtime_enabled,
)


def test_extract_sparse_circulant_stencil_scheme10_matches_dense() -> None:
    _, _, ddx, _ = uniform_diff_matrices(n=17, x_min=0.0, x_max=2.0 * np.pi, scheme=10)
    shifts, coeffs = extract_sparse_circulant_stencil(np.asarray(ddx))
    assert shifts
    assert coeffs
    x = np.linspace(0.0, 1.0, 17, dtype=np.float64)
    y_dense = np.asarray(ddx) @ x
    y_stencil = np.zeros_like(x)
    for shift, coeff in zip(shifts, coeffs):
        y_stencil += coeff * np.roll(x, shift)
    np.testing.assert_allclose(y_stencil, y_dense, rtol=0.0, atol=1e-13)


def test_extract_sparse_circulant_stencil_nonperiodic_returns_empty() -> None:
    _, _, ddx, _ = uniform_diff_matrices(n=17, x_min=0.0, x_max=1.0, scheme=12)
    shifts, coeffs = extract_sparse_circulant_stencil(np.asarray(ddx))
    assert shifts == ()
    assert coeffs == ()


def test_apply_periodic_stencil_roll_matches_dense_theta_einsum() -> None:
    _, _, ddx, _ = uniform_diff_matrices(n=19, x_min=0.0, x_max=2.0 * np.pi, scheme=10)
    shifts, coeffs = extract_sparse_circulant_stencil(np.asarray(ddx))
    rng = np.random.default_rng(3)
    f = jnp.asarray(rng.normal(size=(2, 3, 4, 19, 5)), dtype=jnp.float64)
    y_dense = jnp.einsum("ij,sxljz->sxliz", ddx, f)
    y_stencil = apply_periodic_stencil_roll(f, shifts=shifts, coeffs=coeffs, axis=3)
    np.testing.assert_allclose(np.asarray(y_stencil), np.asarray(y_dense), rtol=0.0, atol=1e-12)


def test_apply_sparse_row_stencil_matches_dense_theta_einsum() -> None:
    _, _, ddx, _ = uniform_diff_matrices(n=19, x_min=0.0, x_max=1.0, scheme=2)
    cols, vals = extract_sparse_row_stencil(np.asarray(ddx), max_row_nnz=5)
    assert cols.shape == vals.shape
    rng = np.random.default_rng(7)
    f = jnp.asarray(rng.normal(size=(2, 3, 4, 19, 5)), dtype=jnp.float64)
    y_dense = jnp.einsum("ij,sxljz->sxliz", ddx, f)
    y_sparse = apply_sparse_row_stencil_gather(
        f,
        cols=jnp.asarray(cols, dtype=jnp.int32),
        vals=jnp.asarray(vals, dtype=jnp.float64),
        axis=3,
    )
    np.testing.assert_allclose(np.asarray(y_sparse), np.asarray(y_dense), rtol=0.0, atol=1e-12)


def test_periodic_stencil_auto_enables_on_sharded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_PERIODIC_STENCIL", "1")
    monkeypatch.delenv("SFINCS_JAX_PERIODIC_STENCIL_ON_SHARDED", raising=False)
    monkeypatch.setenv("SFINCS_JAX_MATVEC_SHARD_AXIS", "theta")
    monkeypatch.setenv("SFINCS_JAX_GMRES_DISTRIBUTED", "1")
    monkeypatch.setattr(jax, "local_device_count", lambda: 2)
    assert periodic_stencil_runtime_enabled() is True


def test_periodic_stencil_auto_disables_when_unsharded_multidevice(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_PERIODIC_STENCIL", "1")
    monkeypatch.delenv("SFINCS_JAX_PERIODIC_STENCIL_ON_SHARDED", raising=False)
    monkeypatch.setenv("SFINCS_JAX_MATVEC_SHARD_AXIS", "off")
    monkeypatch.setenv("SFINCS_JAX_GMRES_DISTRIBUTED", "off")
    monkeypatch.setattr(jax, "local_device_count", lambda: 2)
    assert periodic_stencil_runtime_enabled() is False


def test_sharding_active_hint_tracks_env_controls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_MATVEC_SHARD_AXIS", "theta")
    assert _sharding_active_hint() is True
    monkeypatch.setenv("SFINCS_JAX_MATVEC_SHARD_AXIS", "off")
    monkeypatch.setenv("SFINCS_JAX_GMRES_DISTRIBUTED", "auto")
    assert _sharding_active_hint() is False
    monkeypatch.delenv("SFINCS_JAX_MATVEC_SHARD_AXIS", raising=False)
    assert _sharding_active_hint() is True
    monkeypatch.setenv("SFINCS_JAX_GMRES_DISTRIBUTED", "off")
    assert _sharding_active_hint() is False


def test_periodic_stencil_runtime_env_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_PERIODIC_STENCIL", "off")
    assert periodic_stencil_runtime_enabled() is False

    monkeypatch.setenv("SFINCS_JAX_PERIODIC_STENCIL", "1")
    monkeypatch.setenv("SFINCS_JAX_PERIODIC_STENCIL_ON_SHARDED", "0")
    monkeypatch.setattr(jax, "local_device_count", lambda: 2)
    assert periodic_stencil_runtime_enabled() is False

    monkeypatch.setenv("SFINCS_JAX_PERIODIC_STENCIL_ON_SHARDED", "yes")
    assert periodic_stencil_runtime_enabled() is True

    monkeypatch.setattr(jax, "local_device_count", lambda: (_ for _ in ()).throw(RuntimeError("no devices")))
    assert periodic_stencil_runtime_enabled() is True


def test_extract_sparse_circulant_stencil_respects_runtime_disable_and_density_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, ddx, _ = uniform_diff_matrices(n=17, x_min=0.0, x_max=2.0 * np.pi, scheme=10)
    monkeypatch.setenv("SFINCS_JAX_PERIODIC_STENCIL", "0")
    assert extract_sparse_circulant_stencil(np.asarray(ddx)) == ((), ())

    monkeypatch.setenv("SFINCS_JAX_PERIODIC_STENCIL", "1")
    shifts, coeffs = extract_sparse_circulant_stencil(np.asarray(ddx), max_nnz=2)
    assert shifts == ()
    assert coeffs == ()


def test_extract_sparse_circulant_stencil_rejects_invalid_or_non_circulant_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SFINCS_JAX_PERIODIC_STENCIL", "1")
    monkeypatch.setenv("SFINCS_JAX_PERIODIC_STENCIL_ATOL", "not-a-float")
    monkeypatch.setenv("SFINCS_JAX_PERIODIC_STENCIL_MAX_NNZ", "not-an-int")

    assert extract_sparse_circulant_stencil(np.ones((3, 4))) == ((), ())
    assert extract_sparse_circulant_stencil(np.ones((1, 1))) == ((), ())
    assert extract_sparse_circulant_stencil(np.zeros((4, 4))) == ((), ())

    non_circulant = np.eye(4, dtype=np.float64)
    non_circulant[2, 1] = 0.25
    assert extract_sparse_circulant_stencil(non_circulant) == ((), ())


def test_extract_sparse_circulant_stencil_reproduces_fourier_mode_eigenaction() -> None:
    n = 21
    x, _, ddx, _ = uniform_diff_matrices(n=n, x_min=0.0, x_max=2.0 * np.pi, scheme=10)
    shifts, coeffs = extract_sparse_circulant_stencil(np.asarray(ddx))
    assert shifts
    mode = 3
    f = np.exp(1j * mode * np.asarray(x))
    y_dense = np.asarray(ddx) @ f
    y_stencil = np.asarray(apply_periodic_stencil_roll(jnp.asarray(f), shifts=shifts, coeffs=coeffs, axis=0))
    np.testing.assert_allclose(y_stencil, y_dense, rtol=0.0, atol=1e-12)


def test_extract_sparse_row_stencil_returns_empty_for_bad_shape_or_dense_rows() -> None:
    cols, vals = extract_sparse_row_stencil(np.ones((3, 4)))
    assert cols.shape == (0, 0)
    assert vals.shape == (0, 0)

    dense = np.ones((4, 4), dtype=np.float64)
    cols, vals = extract_sparse_row_stencil(dense, max_row_nnz=2)
    assert cols.shape == (0, 0)
    assert vals.shape == (0, 0)


def test_extract_sparse_row_stencil_handles_zero_rows_and_env_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SFINCS_JAX_PERIODIC_STENCIL_ATOL", "bad")
    monkeypatch.setenv("SFINCS_JAX_DERIV_SPARSE_MAX_ROW_NNZ", "bad")

    cols, vals = extract_sparse_row_stencil(np.ones((1, 1)))
    assert cols.shape == (0, 0)
    assert vals.shape == (0, 0)

    sparse = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, -2.0, 0.0],
        ],
        dtype=np.float64,
    )
    cols, vals = extract_sparse_row_stencil(sparse)
    assert cols.shape == (3, 1)
    np.testing.assert_array_equal(cols[:, 0], [0, 0, 1])
    np.testing.assert_allclose(vals[:, 0], [1.0, 0.0, -2.0])


def test_apply_periodic_stencil_halo_falls_back_to_roll_on_small_local_shards() -> None:
    shifts, coeffs = (2, -2), (1.0, -0.5)
    f = jnp.asarray(np.arange(2.0), dtype=jnp.float64)
    y_roll = apply_periodic_stencil_roll(f, shifts=shifts, coeffs=coeffs, axis=0)
    y_halo = apply_periodic_stencil_halo(
        f,
        shifts=shifts,
        coeffs=coeffs,
        axis=0,
        axis_name="theta",
        axis_size=4,
    )
    np.testing.assert_allclose(np.asarray(y_halo), np.asarray(y_roll), rtol=0.0, atol=1e-12)


def test_periodic_halo_exchange_single_axis_and_zero_width_paths() -> None:
    f = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float64)
    left, right = _periodic_halo_exchange(f, axis=0, axis_name="theta", width=0, axis_size=1)
    np.testing.assert_allclose(np.asarray(left), np.zeros(3))
    np.testing.assert_allclose(np.asarray(right), np.zeros(3))

    left, right = _periodic_halo_exchange(f, axis=0, axis_name="theta", width=2, axis_size=1)
    np.testing.assert_allclose(np.asarray(left), [2.0, 3.0])
    np.testing.assert_allclose(np.asarray(right), [1.0, 2.0])

    empty = jnp.asarray([], dtype=jnp.float64)
    left, right = _periodic_halo_exchange(empty, axis=0, axis_name="theta", width=1, axis_size=1)
    assert left.shape == (0,)
    assert right.shape == (0,)


def test_apply_periodic_stencil_halo_zero_and_empty_stencils() -> None:
    f = jnp.asarray([1.0, 2.0, 4.0], dtype=jnp.float64)

    np.testing.assert_allclose(
        np.asarray(apply_periodic_stencil_halo(f, shifts=(), coeffs=(), axis=0, axis_name="theta")),
        np.zeros(3),
    )
    np.testing.assert_allclose(
        np.asarray(apply_periodic_stencil_halo(f, shifts=(0,), coeffs=(2.0,), axis=0, axis_name="theta")),
        [2.0, 4.0, 8.0],
    )


def test_apply_periodic_stencil_roll_skips_zero_coefficients() -> None:
    f = jnp.asarray([1.0, 2.0, 4.0], dtype=jnp.float64)
    out = apply_periodic_stencil_roll(f, shifts=(1, -1), coeffs=(0.0, 2.0), axis=0)
    np.testing.assert_allclose(np.asarray(out), 2.0 * np.roll(np.asarray(f), -1))
