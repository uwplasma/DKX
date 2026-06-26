from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from scipy import sparse

from sfincs_jax.operators.profile_layout import RHS1BlockLayout
from sfincs_jax.problems.profile_policies import (
    _StructuredHostSparsePreconditionerBundle,
    _direct_tail_structured_pc_cache_key,
    _direct_tail_structured_pc_with_cache_metadata,
    _hash_numpy_array_for_cache,
    _is_direct_reduced_pmat_pc_kind,
    _rhsmode1_fortran_reduced_direct_tail_pc_default_max_mb,
)


class _MatvecOperator:
    def matvec(self, vec: np.ndarray) -> np.ndarray:
        return 2.0 * np.asarray(vec, dtype=np.float64)


class _Preconditioner:
    operator = _MatvecOperator()


@dataclass(frozen=True)
class _CachedPreconditioner:
    setup_s: float
    metadata: dict[str, object] | None = None


def _layout() -> RHS1BlockLayout:
    return RHS1BlockLayout(
        n_species=1,
        n_x=2,
        n_xi=2,
        n_theta=1,
        n_zeta=1,
        f_size=4,
        phi1_size=0,
        extra_size=0,
        total_size=4,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )


def test_structured_host_sparse_bundle_applies_preconditioner_operator() -> None:
    bundle = _StructuredHostSparsePreconditionerBundle(
        preconditioner=_Preconditioner(),
        operator=object(),
        kind="unit",
    )

    np.testing.assert_allclose(bundle.solve(np.asarray([1.0, -3.0])), np.asarray([2.0, -6.0]))

    missing = _StructuredHostSparsePreconditionerBundle(
        preconditioner=object(),
        operator=object(),
        kind="missing",
    )
    with pytest.raises(RuntimeError, match="missing"):
        missing.solve(np.asarray([1.0]))


def test_direct_tail_policy_hash_and_aliases_are_stable() -> None:
    arr = np.asarray([1, 2, 3], dtype=np.int64)
    assert _hash_numpy_array_for_cache(arr) == _hash_numpy_array_for_cache(arr.copy())
    assert _hash_numpy_array_for_cache(arr.astype(np.int32)) != _hash_numpy_array_for_cache(arr)

    assert _is_direct_reduced_pmat_pc_kind("direct-reduced-pmat-lu")
    assert _is_direct_reduced_pmat_pc_kind("ACTIVE_FORTRAN_V3_DIRECT_PMAT_ILU")
    assert not _is_direct_reduced_pmat_pc_kind("active_fortran_v3_reduced_lu")


def test_direct_tail_cache_key_includes_layout_support_and_env(monkeypatch) -> None:
    layout = _layout()
    matrix = sparse.eye(layout.total_size, format="csr", dtype=np.float64)
    active = np.arange(layout.total_size, dtype=np.int64)

    monkeypatch.delenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_RADIAL", raising=False)
    key_default = _direct_tail_structured_pc_cache_key(
        matrix=matrix,
        layout=layout,
        active_indices=active,
        kind="active_fortran_v3_reduced_lu",
        max_factor_nbytes=1_000,
        regularization=1.0e-12,
        support_modes=(1, 1, 1, 0),
    )
    key_different_support = _direct_tail_structured_pc_cache_key(
        matrix=matrix,
        layout=layout,
        active_indices=active,
        kind="active_fortran_v3_reduced_lu",
        max_factor_nbytes=1_000,
        regularization=1.0e-12,
        support_modes=(0, 1, 1, 0),
    )
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_RADIAL", "dense")
    key_env = _direct_tail_structured_pc_cache_key(
        matrix=matrix,
        layout=layout,
        active_indices=active,
        kind="active_fortran_v3_reduced_lu",
        max_factor_nbytes=1_000,
        regularization=1.0e-12,
        support_modes=(1, 1, 1, 0),
    )

    assert key_default != key_different_support
    assert key_default != key_env


def test_direct_tail_cache_metadata_marks_hits_and_misses() -> None:
    miss = _direct_tail_structured_pc_with_cache_metadata(
        _CachedPreconditioner(setup_s=3.0),
        cache_hit=False,
        cache_key=("unit", 1),
    )
    hit = _direct_tail_structured_pc_with_cache_metadata(
        _CachedPreconditioner(setup_s=3.0),
        cache_hit=True,
        cache_key=("unit", 1),
    )

    assert miss.setup_s == pytest.approx(3.0)
    assert miss.metadata["direct_tail_structured_pc_cache_hit"] is False
    assert hit.setup_s == pytest.approx(0.0)
    assert hit.metadata["direct_tail_structured_pc_cache_hit"] is True
    assert hit.metadata["direct_tail_structured_pc_cached_setup_s"] == pytest.approx(3.0)


def test_direct_tail_default_memory_cap_matches_adaptive_policy(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PC_AUTO_BASE_MB", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PC_AUTO_MAX_MB", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PC_AUTO_MB_PER_UNKNOWN", raising=False)

    assert _rhsmode1_fortran_reduced_direct_tail_pc_default_max_mb(
        requested_kind="active_fortran_v3_reduced_lu",
        active_size=604,
    ) == pytest.approx(521.664)
    assert _rhsmode1_fortran_reduced_direct_tail_pc_default_max_mb(
        requested_kind="auto",
        active_size=169_264,
    ) == pytest.approx(3220.224)
    assert _rhsmode1_fortran_reduced_direct_tail_pc_default_max_mb(
        requested_kind="active_fortran_v3_reduced_lu",
        active_size=900_000,
    ) == pytest.approx(16384.0)
    assert _rhsmode1_fortran_reduced_direct_tail_pc_default_max_mb(
        requested_kind="active_xblock",
        active_size=900_000,
    ) == pytest.approx(512.0)
