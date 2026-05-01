from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import jax.numpy as jnp
import pytest
from scipy import sparse

import sfincs_jax.v3_driver as v3_driver


class _HalfSolve:
    def solve(self, rhs: np.ndarray) -> np.ndarray:
        return 0.5 * np.asarray(rhs, dtype=np.float64)


def test_host_sparse_direct_allowed_and_sparse_pc_rescue_policy(monkeypatch) -> None:
    assert not v3_driver._rhsmode1_host_sparse_direct_allowed(sparse_exact_lu=False)
    assert not v3_driver._rhsmode1_host_sparse_direct_allowed(sparse_exact_lu=True, use_implicit=True)

    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_DIRECT_HOST", raising=False)
    assert v3_driver._rhsmode1_host_sparse_direct_allowed(sparse_exact_lu=True)
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_DIRECT_HOST", "0")
    assert not v3_driver._rhsmode1_host_sparse_direct_allowed(sparse_exact_lu=True)
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_DIRECT_HOST", "1")
    assert v3_driver._rhsmode1_host_sparse_direct_allowed(sparse_exact_lu=True)

    op = SimpleNamespace(
        rhs_mode=1,
        include_phi1=False,
        constraint_scheme=1,
        fblock=SimpleNamespace(fp=object(), pas=None),
    )
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES", raising=False)
    assert v3_driver._rhsmode1_sparse_operator_preconditioned_rescue_allowed(
        op=op, sparse_exact_lu=True, host_sparse_direct_wanted=True
    )
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES", "off")
    assert not v3_driver._rhsmode1_sparse_operator_preconditioned_rescue_allowed(
        op=op, sparse_exact_lu=True, host_sparse_direct_wanted=True
    )


def test_host_sparse_factor_dtype_and_cache_key(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_HOST_SPARSE_FACTOR_DTYPE", raising=False)
    monkeypatch.delenv("SFINCS_JAX_HOST_SPARSE_FACTOR_FLOAT32_MIN", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    assert v3_driver._host_sparse_factor_dtype(size=20_000, factorization="lu", use_implicit=False) == np.dtype(np.float32)
    assert v3_driver._host_sparse_factor_dtype(size=1_000, factorization="ilu", use_implicit=False) == np.dtype(np.float64)
    assert v3_driver._host_sparse_factor_dtype(size=20_000, factorization="lu", use_implicit=True) == np.dtype(np.float64)

    monkeypatch.setenv("SFINCS_JAX_HOST_SPARSE_FACTOR_DTYPE", "fp64")
    assert v3_driver._host_sparse_factor_dtype(size=20_000, factorization="lu", use_implicit=False) == np.dtype(np.float64)
    monkeypatch.setenv("SFINCS_JAX_HOST_SPARSE_FACTOR_DTYPE", "32")
    assert v3_driver._host_sparse_factor_dtype(size=1, factorization="lu", use_implicit=False) == np.dtype(np.float32)

    key = v3_driver._sparse_factor_cache_key(("a", 1), np.dtype(np.float32))
    assert key == ("a", 1, np.dtype(np.float32).str)


def test_host_sparse_refine_step_parsing_and_skip_dense_ratio(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_DIRECT_SKIP_DENSE_RATIO", raising=False)
    assert v3_driver._rhsmode1_host_sparse_skip_dense_ratio() == pytest.approx(1.0e4)
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_DIRECT_SKIP_DENSE_RATIO", "bad")
    assert v3_driver._rhsmode1_host_sparse_skip_dense_ratio() == pytest.approx(1.0e4)

    monkeypatch.delenv("MY_REFINE_STEPS", raising=False)
    assert v3_driver._host_sparse_direct_refine_steps("MY_REFINE_STEPS", default=3) == 3
    monkeypatch.setenv("MY_REFINE_STEPS", "bad")
    assert v3_driver._host_sparse_direct_refine_steps("MY_REFINE_STEPS", default=2) == 2
    monkeypatch.setenv("MY_REFINE_STEPS", "-4")
    assert v3_driver._host_sparse_direct_refine_steps("MY_REFINE_STEPS", default=2) == 0


def test_host_direct_refinement_helpers_improve_residual() -> None:
    rhs = jnp.asarray([2.0, -4.0], dtype=jnp.float64)
    ident = np.eye(2, dtype=np.float64)

    x_direct, rn_direct = v3_driver._host_direct_solve_with_refinement(
        factor_solve=lambda v: 0.5 * np.asarray(v, dtype=np.float64),
        operator_matrix=ident,
        rhs_vec=rhs,
        factor_dtype=np.dtype(np.float64),
        refine_steps=3,
    )
    np.testing.assert_allclose(x_direct, np.asarray([1.875, -3.75]))
    assert rn_direct < np.linalg.norm(np.asarray(rhs) - 0.5 * np.asarray(rhs))

    x_sparse, rn_sparse = v3_driver._host_sparse_direct_solve_with_refinement(
        ilu=_HalfSolve(),
        a_csr_full=sparse.csr_matrix(ident),
        rhs_vec=rhs,
        factor_dtype=np.dtype(np.float64),
        refine_steps=3,
    )
    np.testing.assert_allclose(x_sparse, x_direct)
    assert rn_sparse == pytest.approx(rn_direct)


def test_host_sparse_direct_polish_uses_preconditioner_and_reports_residual(monkeypatch) -> None:
    seen = {}

    def fake_gmres(**kwargs):
        probe = kwargs["preconditioner"](jnp.asarray([4.0, -2.0], dtype=jnp.float64))
        seen["probe"] = np.asarray(probe)
        return np.asarray([1.0, -2.0]), 0.0, [0.0]

    monkeypatch.setattr(v3_driver, "gmres_solve_with_history_scipy", fake_gmres)
    x, rn = v3_driver._host_sparse_direct_polish(
        matvec_fn=lambda x: x,
        rhs_vec=jnp.asarray([1.0, -2.0], dtype=jnp.float64),
        x0_np=np.zeros(2, dtype=np.float64),
        ilu=_HalfSolve(),
        factor_dtype=np.dtype(np.float64),
        tol=1e-12,
        atol=0.0,
        restart=4,
        maxiter=4,
        precondition_side="left",
    )
    np.testing.assert_allclose(seen["probe"], np.asarray([2.0, -1.0]))
    np.testing.assert_allclose(x, np.asarray([1.0, -2.0]))
    assert rn == 0.0


def test_explicit_sparse_host_direct_allowed_and_env_bounds(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_EXPLICIT_SPARSE_HELPER", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_EXPLICIT_SPARSE_HELPER_MAX", raising=False)
    assert v3_driver._rhsmode1_explicit_sparse_host_direct_allowed(
        sparse_exact_lu=True,
        use_implicit=False,
        active_size=10_000,
    )
    assert not v3_driver._rhsmode1_explicit_sparse_host_direct_allowed(
        sparse_exact_lu=False,
        use_implicit=False,
        active_size=10_000,
    )
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_EXPLICIT_SPARSE_HELPER_MAX", "bad")
    assert v3_driver._rhsmode1_explicit_sparse_host_direct_allowed(
        sparse_exact_lu=True,
        use_implicit=False,
        active_size=10_000,
    )
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_EXPLICIT_SPARSE_HELPER", "off")
    assert not v3_driver._rhsmode1_explicit_sparse_host_direct_allowed(
        sparse_exact_lu=True,
        use_implicit=False,
        active_size=10_000,
    )


def test_build_host_sparse_direct_factor_from_matvec_falls_back_on_invalid_env(monkeypatch) -> None:
    seen: dict[str, object] = {}
    operator_bundle = SimpleNamespace(metadata=SimpleNamespace(storage_kind="csr", reason="fallback"))
    factor_bundle = SimpleNamespace(kind="lu")

    def fake_build_operator_from_matvec(matvec, **kwargs):
        seen["kwargs"] = kwargs
        np.testing.assert_allclose(matvec(np.asarray([1.0, -2.0])), np.asarray([2.0, -4.0]))
        np.testing.assert_allclose(
            kwargs["matmat"](np.asarray([[1.0, 0.0], [0.0, 1.0]])),
            2.0 * np.eye(2),
        )
        return operator_bundle

    def fake_factorize_host_sparse_operator(bundle, *, kind, **kwargs):
        seen["factor"] = (bundle, kind, kwargs)
        return factor_bundle

    monkeypatch.setenv("SFINCS_JAX_EXPLICIT_SPARSE_BLOCK_COLS", "bad")
    monkeypatch.setenv("SFINCS_JAX_EXPLICIT_SPARSE_DENSE_MAX_MB", "bad")
    monkeypatch.setenv("SFINCS_JAX_EXPLICIT_SPARSE_CSR_MAX_MB", "bad")
    monkeypatch.setenv("SFINCS_JAX_EXPLICIT_SPARSE_DROP_TOL", "bad")
    monkeypatch.setattr(v3_driver, "build_operator_from_matvec", fake_build_operator_from_matvec)
    monkeypatch.setattr(v3_driver, "factorize_host_sparse_operator", fake_factorize_host_sparse_operator)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")

    messages: list[tuple[int, str]] = []
    op, fac = v3_driver._build_host_sparse_direct_factor_from_matvec(
        matvec=lambda x: 2.0 * x,
        n=2,
        dtype=jnp.float32,
        factor_dtype=np.dtype(np.float64),
        emit=lambda level, msg: messages.append((level, msg)),
    )

    assert op is operator_bundle
    assert fac is factor_bundle
    factor_bundle_seen, factor_kind_seen, factor_kwargs_seen = seen["factor"]
    assert factor_bundle_seen is operator_bundle
    assert factor_kind_seen == "lu"
    assert factor_kwargs_seen["fill_factor"] == pytest.approx(10.0)
    assert factor_kwargs_seen["drop_tol"] == pytest.approx(1.0e-4)
    assert factor_kwargs_seen["permc_spec"] == "COLAMD"
    assert factor_kwargs_seen["diag_pivot_thresh"] == pytest.approx(1.0)
    kwargs = seen["kwargs"]
    assert kwargs["block_cols"] == 32
    assert kwargs["dense_max_mb"] == pytest.approx(128.0)
    assert kwargs["csr_max_mb"] == pytest.approx(512.0)
    assert kwargs["drop_tol"] == pytest.approx(0.0)
    assert kwargs["prefer_sparse_on_gpu"] is True
    assert kwargs["allow_operator_only"] is False
    assert messages == [(1, "explicit_sparse: storage=csr reason=fallback factor_kind=lu permc=COLAMD")]


def test_build_host_sparse_direct_factor_from_matvec_respects_env_overrides(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_build_operator_from_matvec(matvec, **kwargs):
        seen["kwargs"] = kwargs
        np.testing.assert_allclose(matvec(np.asarray([3.0, -1.0])), np.asarray([9.0, -3.0]))
        return SimpleNamespace(metadata=SimpleNamespace(storage_kind="dense", reason="forced"))

    monkeypatch.setenv("SFINCS_JAX_EXPLICIT_SPARSE_BLOCK_COLS", "7")
    monkeypatch.setenv("SFINCS_JAX_EXPLICIT_SPARSE_DENSE_MAX_MB", "3.5")
    monkeypatch.setenv("SFINCS_JAX_EXPLICIT_SPARSE_CSR_MAX_MB", "9.5")
    monkeypatch.setenv("SFINCS_JAX_EXPLICIT_SPARSE_DROP_TOL", "1e-3")
    monkeypatch.setattr(v3_driver, "build_operator_from_matvec", fake_build_operator_from_matvec)
    monkeypatch.setattr(
        v3_driver,
        "factorize_host_sparse_operator",
        lambda bundle, *, kind, **kwargs: SimpleNamespace(bundle=bundle, kind=kind, kwargs=kwargs),
    )
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "gpu")

    v3_driver._build_host_sparse_direct_factor_from_matvec(
        matvec=lambda x: 3.0 * x,
        n=2,
        dtype=jnp.float64,
        factor_dtype=np.dtype(np.float32),
    )

    kwargs = seen["kwargs"]
    assert kwargs["backend"] == "gpu"
    assert kwargs["block_cols"] == 7
    assert kwargs["dense_max_mb"] == pytest.approx(3.5)
    assert kwargs["csr_max_mb"] == pytest.approx(9.5)
    assert kwargs["drop_tol"] == pytest.approx(1.0e-3)


def test_build_host_sparse_direct_factor_from_matvec_can_use_pattern_probe(monkeypatch) -> None:
    seen: dict[str, object] = {}
    pattern = sparse.eye(2, format="csr")

    def fake_build_operator_from_pattern(matvec, **kwargs):
        seen["kwargs"] = kwargs
        np.testing.assert_allclose(matvec(np.asarray([4.0, -2.0])), np.asarray([8.0, -4.0]))
        return SimpleNamespace(metadata=SimpleNamespace(storage_kind="csr", reason="pattern"))

    monkeypatch.setattr(v3_driver, "build_operator_from_pattern", fake_build_operator_from_pattern)
    monkeypatch.setattr(
        v3_driver,
        "factorize_host_sparse_operator",
        lambda bundle, *, kind, **kwargs: SimpleNamespace(bundle=bundle, kind=kind, kwargs=kwargs),
    )
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")

    v3_driver._build_host_sparse_direct_factor_from_matvec(
        matvec=lambda x: 2.0 * x,
        n=2,
        dtype=jnp.float64,
        factor_dtype=np.dtype(np.float64),
        pattern=pattern,
    )

    kwargs = seen["kwargs"]
    assert kwargs["pattern"] is pattern
    assert kwargs["backend"] == "cpu"
    assert kwargs["allow_operator_only"] is False
