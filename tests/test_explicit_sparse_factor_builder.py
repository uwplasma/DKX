from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from scipy import sparse

from sfincs_jax.solvers.explicit_sparse_factor_builder import build_host_sparse_direct_factor_from_matvec


def test_explicit_sparse_factor_builder_uses_injected_matvec_backend_and_factorizer(monkeypatch) -> None:
    seen: dict[str, object] = {}
    operator_bundle = SimpleNamespace(metadata=SimpleNamespace(storage_kind="csr", reason="unit"))
    factor_bundle = SimpleNamespace(kind="ilu")

    def fake_build_operator_from_matvec(matvec, **kwargs):
        seen["operator_kwargs"] = kwargs
        np.testing.assert_allclose(matvec(np.asarray([1.0, -2.0])), np.asarray([3.0, -6.0]))
        np.testing.assert_allclose(kwargs["matmat"](np.eye(2)), 3.0 * np.eye(2))
        return operator_bundle

    def fake_factorize_host_sparse_operator(bundle, *, kind, **kwargs):
        seen["factor_bundle"] = bundle
        seen["factor_kind"] = kind
        seen["factor_kwargs"] = kwargs
        return factor_bundle

    monkeypatch.setenv("SFINCS_JAX_EXPLICIT_SPARSE_FACTOR_KIND", "spilu")
    op, factor = build_host_sparse_direct_factor_from_matvec(
        matvec=lambda x: 3.0 * x,
        n=2,
        dtype=jnp.float64,
        factor_dtype=np.dtype(np.float32),
        default_backend_callback=lambda: "gpu",
        build_operator_from_matvec_callback=fake_build_operator_from_matvec,
        factorize_host_sparse_operator_callback=fake_factorize_host_sparse_operator,
    )

    assert op is operator_bundle
    assert factor is factor_bundle
    assert seen["factor_bundle"] is operator_bundle
    assert seen["factor_kind"] == "ilu"
    assert seen["operator_kwargs"]["backend"] == "gpu"
    assert seen["operator_kwargs"]["dtype"] == np.dtype(np.float32)
    assert seen["factor_kwargs"]["fill_factor"] == pytest.approx(10.0)
    assert seen["factor_kwargs"]["drop_tol"] == pytest.approx(1.0e-4)
    assert seen["factor_kwargs"]["permc_spec"] == "COLAMD"


def test_explicit_sparse_factor_builder_pattern_probe_forwards_progress(monkeypatch) -> None:
    seen: dict[str, object] = {}
    messages: list[tuple[int, str]] = []
    pattern = sparse.eye(2, format="csr")

    def fake_build_operator_from_pattern(matvec, **kwargs):
        seen["operator_kwargs"] = kwargs
        kwargs["progress_callback"]("pattern progress")
        np.testing.assert_allclose(matvec(np.asarray([2.0, 4.0])), np.asarray([1.0, 2.0]))
        return SimpleNamespace(metadata=SimpleNamespace(storage_kind="csr", reason="pattern"))

    def fake_factorize_host_sparse_operator(bundle, *, kind, **kwargs):
        return SimpleNamespace(kind=kind, bundle=bundle, kwargs=kwargs)

    build_host_sparse_direct_factor_from_matvec(
        matvec=lambda x: 0.5 * x,
        n=2,
        dtype=jnp.float64,
        factor_dtype=np.dtype(np.float64),
        pattern=pattern,
        emit=lambda level, message: messages.append((level, message)),
        default_permc_spec="MMD_AT_PLUS_A",
        default_backend_callback=lambda: "cpu",
        build_operator_from_pattern_callback=fake_build_operator_from_pattern,
        factorize_host_sparse_operator_callback=fake_factorize_host_sparse_operator,
    )

    assert seen["operator_kwargs"]["pattern"] is pattern
    assert seen["operator_kwargs"]["backend"] == "cpu"
    assert seen["operator_kwargs"]["color_batch"] == 1
    assert any(message == "explicit_sparse: pattern progress" for _, message in messages)
    assert any("permc=MMD_AT_PLUS_A" in message for _, message in messages)


def test_explicit_sparse_factor_builder_monolithic_guard_uses_injected_policy() -> None:
    operator_bundle = SimpleNamespace(
        metadata=SimpleNamespace(
            storage_kind="csr",
            reason="large",
            shape=(5, 5),
            nnz_estimate=5,
            csr_nbytes_estimate=128,
        )
    )

    with pytest.raises(MemoryError, match="monolithic factor preflight rejected"):
        build_host_sparse_direct_factor_from_matvec(
            matvec=lambda x: x,
            n=5,
            dtype=jnp.float64,
            factor_dtype=np.dtype(np.float64),
            build_operator_from_matvec_callback=lambda _matvec, **_kwargs: operator_bundle,
            factorize_host_sparse_operator_callback=lambda *_args, **_kwargs: pytest.fail(
                "factorization should be guarded"
            ),
            monolithic_max_size_callback=lambda _kind: 4,
        )
