"""RHSMode=1 direct-tail structured preconditioner cache and policy helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import os
from typing import Any

import numpy as np

from .explicit_sparse import SparseOperatorBundle
from .rhs1_solver_policy import read_float_env, read_int_env


@dataclass(frozen=True)
class _StructuredHostSparsePreconditionerBundle:
    """Adapter exposing structured CSR preconditioners through the factor API."""

    preconditioner: object
    operator: SparseOperatorBundle
    kind: str
    factor_nbytes_estimate: int | None = None
    factor_nnz_estimate: int | None = None
    factor_s: float | None = None

    def solve(self, rhs) -> np.ndarray:
        preconditioner_operator = getattr(self.preconditioner, "operator", None)
        if preconditioner_operator is None:
            raise RuntimeError(f"structured host preconditioner {self.kind!r} was not selected")
        return np.asarray(preconditioner_operator.matvec(np.asarray(rhs, dtype=np.float64).reshape((-1,))))


_DIRECT_TAIL_STRUCTURED_PC_CACHE: dict[tuple[object, ...], object] = {}
_DIRECT_REDUCED_PMAT_PC_KINDS = frozenset(
    {
        "active_fortran_v3_reduced_direct_pmat_lu",
        "active_fortran_v3_reduced_direct_pmat_ilu",
        "active_fortran_v3_direct_pmat_lu",
        "active_fortran_v3_direct_pmat_ilu",
        "direct_reduced_pmat_lu",
        "direct_reduced_pmat_ilu",
    }
)


def _is_direct_reduced_pmat_pc_kind(kind: str | None) -> bool:
    """Return true for explicit preconditioner aliases that avoid active CSR assembly."""

    normalized = str(kind or "").strip().lower().replace("-", "_")
    return normalized in _DIRECT_REDUCED_PMAT_PC_KINDS


def _hash_numpy_array_for_cache(array: Any) -> str:
    """Return a stable content hash for NumPy-like cache-key arrays."""

    arr = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(arr.dtype).encode("ascii", errors="ignore"))
    digest.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    digest.update(arr.view(np.uint8))
    return digest.hexdigest()


def _direct_tail_structured_pc_cache_key(
    *,
    matrix: Any,
    layout: Any,
    active_indices: np.ndarray | None,
    kind: str,
    max_factor_nbytes: int,
    regularization: float,
    support_modes: tuple[int, int, int, int] | None = None,
) -> tuple[object, ...]:
    """Return a robust cache key for an active direct-tail preconditioner."""

    matrix_csr = matrix.tocsr()
    env_signature = tuple(
        sorted(
            (str(key), str(value))
            for key, value in os.environ.items()
            if str(key).startswith("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_")
        )
    )
    active_digest = "all" if active_indices is None else _hash_numpy_array_for_cache(np.asarray(active_indices))
    return (
        "direct_tail_structured_pc_v1",
        tuple(int(v) for v in matrix_csr.shape),
        int(matrix_csr.nnz),
        _hash_numpy_array_for_cache(matrix_csr.indptr),
        _hash_numpy_array_for_cache(matrix_csr.indices),
        _hash_numpy_array_for_cache(matrix_csr.data),
        active_digest,
        tuple(sorted(layout.to_dict().items())),
        str(kind).strip().lower().replace("-", "_"),
        None if support_modes is None else tuple(int(v) for v in support_modes),
        int(max_factor_nbytes),
        float(regularization),
        env_signature,
    )


def _direct_tail_structured_pc_with_cache_metadata(
    preconditioner: object,
    *,
    cache_hit: bool,
    cache_key: tuple[object, ...],
):
    """Attach direct-tail cache diagnostics to an immutable preconditioner object."""

    metadata = dict(getattr(preconditioner, "metadata", None) or {})
    metadata["direct_tail_structured_pc_cache_hit"] = bool(cache_hit)
    metadata["direct_tail_structured_pc_cache_key_digest"] = hashlib.blake2b(
        repr(cache_key).encode("utf-8"),
        digest_size=12,
    ).hexdigest()
    if bool(cache_hit):
        metadata["direct_tail_structured_pc_cached_setup_s"] = float(getattr(preconditioner, "setup_s", 0.0) or 0.0)
        return replace(preconditioner, setup_s=0.0, metadata=metadata)
    return replace(preconditioner, metadata=metadata)


def _rhsmode1_fortran_reduced_direct_tail_pc_default_max_mb(
    *,
    requested_kind: str | None,
    active_size: int,
) -> float:
    """Return the default direct-tail structured-PC memory cap in MiB."""

    base_mb = read_float_env(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PC_AUTO_BASE_MB",
        default=512.0,
        minimum=0.0,
    )
    auto_max_mb = read_float_env(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PC_AUTO_MAX_MB",
        default=4096.0,
        minimum=0.0,
    )
    slope_mb_per_unknown = read_float_env(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PC_AUTO_MB_PER_UNKNOWN",
        default=1.6e-2,
        minimum=0.0,
    )
    requested = str(requested_kind or "").strip().lower().replace("-", "_")
    if requested in {"auto", "active_auto", "structured", "structured_auto"}:
        requested = "active_fortran_v3_reduced_lu"
    if requested != "active_fortran_v3_reduced_lu":
        return float(base_mb)
    active_n = max(0, int(active_size))
    adaptive_mb = max(float(base_mb), float(base_mb) + float(slope_mb_per_unknown) * float(active_n))
    production_min_active = read_int_env(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PC_AUTO_PRODUCTION_MIN_ACTIVE",
        default=400_000,
        minimum=1,
    )
    if active_n >= int(production_min_active):
        production_max_mb = read_float_env(
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PC_AUTO_PRODUCTION_MAX_MB",
            default=16_384.0,
            minimum=0.0,
        )
        production_slope_mb_per_unknown = read_float_env(
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PC_AUTO_PRODUCTION_MB_PER_UNKNOWN",
            default=2.8e-2,
            minimum=0.0,
        )
        adaptive_mb = max(
            float(adaptive_mb),
            float(base_mb) + float(production_slope_mb_per_unknown) * float(active_n),
        )
        if float(production_max_mb) > 0.0:
            auto_max_mb = max(float(auto_max_mb), float(production_max_mb))
    if float(auto_max_mb) > 0.0:
        adaptive_mb = min(float(adaptive_mb), max(float(base_mb), float(auto_max_mb)))
    return float(adaptive_mb)


__all__ = [
    "_DIRECT_TAIL_STRUCTURED_PC_CACHE",
    "_DIRECT_REDUCED_PMAT_PC_KINDS",
    "_StructuredHostSparsePreconditionerBundle",
    "_direct_tail_structured_pc_cache_key",
    "_direct_tail_structured_pc_with_cache_metadata",
    "_hash_numpy_array_for_cache",
    "_is_direct_reduced_pmat_pc_kind",
    "_rhsmode1_fortran_reduced_direct_tail_pc_default_max_mb",
]
