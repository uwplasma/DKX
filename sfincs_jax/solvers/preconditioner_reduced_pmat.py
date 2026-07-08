"""Fortran-v3-style reduced active sparse factors for RHSMode=1.

SFINCS Fortran v3 builds a simplified ``whichMatrix=0`` preconditioning
operator rather than factoring the exact Jacobian used for the residual.  This
module owns the Python-native analogue used by the explicit host CSR lane:
reduced active matrix construction, support-mode selection, equilibration, LU
or ILU setup, memory admission, and JSON-friendly factor metadata.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
import hashlib
from typing import Any
import os
import time

import numpy as np
import scipy.sparse as sp

from sfincs_jax.operators.profile_layout import (
    RHS1ActiveBlockLayout,
    RHS1BlockLayout,
    RHS1CompressedPitchLayout,
    build_rhs1_compressed_pitch_layout,
    infer_rhs1_compressed_pitch_layout_from_active_indices,
)
from .preconditioner_schur_profile import RHS1StructuredFullCSRPreconditioner


def _policy_env_int(env: Mapping[str, str], name: str, default: int) -> int:
    try:
        return int(str(env.get(name, "")).strip() or int(default))
    except ValueError:
        return int(default)


def _policy_env_float(env: Mapping[str, str], name: str, default: float) -> float:
    try:
        return float(str(env.get(name, "")).strip() or float(default))
    except ValueError:
        return float(default)


def _policy_env_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    value = str(env.get(name, "")).strip().lower()
    if not value:
        return bool(default)
    return value in {"1", "true", "yes", "on"}


def _env_float_nonnegative(name: str, default: float) -> float:
    return max(0.0, _env_float(name, default))


@dataclass(frozen=True)
class _StructuredHostSparsePreconditionerBundle:
    """Adapter exposing structured CSR preconditioners through the factor API."""

    preconditioner: object
    operator: object
    kind: str
    factor_nbytes_estimate: int | None = None
    factor_nnz_estimate: int | None = None
    factor_s: float | None = None

    def solve(self, rhs: Any) -> np.ndarray:
        preconditioner_operator = getattr(self.preconditioner, "operator", None)
        if preconditioner_operator is None:
            raise RuntimeError(f"structured host preconditioner {self.kind!r} was not selected")
        rhs_array = np.asarray(rhs, dtype=np.float64).reshape((-1,))
        return np.asarray(preconditioner_operator.matvec(rhs_array), dtype=np.float64).reshape((-1,))


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
    """Return true for aliases that build the reduced Pmat without true CSR first."""

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
    active_digest = (
        "all"
        if active_indices is None
        else _hash_numpy_array_for_cache(np.asarray(active_indices))
    )
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
        metadata["direct_tail_structured_pc_cached_setup_s"] = float(
            getattr(preconditioner, "setup_s", 0.0) or 0.0
        )
        return replace(preconditioner, setup_s=0.0, metadata=metadata)
    return replace(preconditioner, metadata=metadata)


def _rhsmode1_fortran_reduced_direct_tail_pc_default_max_mb(
    *,
    requested_kind: str | None,
    active_size: int,
) -> float:
    """Return the default direct-tail structured-PC memory cap in MiB."""

    base_mb = _env_float_nonnegative("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PC_AUTO_BASE_MB", 512.0)
    auto_max_mb = _env_float_nonnegative("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PC_AUTO_MAX_MB", 4096.0)
    slope_mb_per_unknown = _env_float_nonnegative(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PC_AUTO_MB_PER_UNKNOWN",
        1.6e-2,
    )
    requested = str(requested_kind or "").strip().lower().replace("-", "_")
    if requested in {"auto", "active_auto", "structured", "structured_auto"}:
        requested = "active_fortran_v3_reduced_lu"
    if requested != "active_fortran_v3_reduced_lu":
        return float(max(0.0, base_mb))
    active_n = max(0, int(active_size))
    adaptive_mb = max(float(base_mb), float(base_mb) + float(slope_mb_per_unknown) * float(active_n))
    production_min_active = max(
        1,
        _env_int(
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PC_AUTO_PRODUCTION_MIN_ACTIVE",
            400_000,
        ),
    )
    if active_n >= int(production_min_active):
        production_max_mb = _env_float_nonnegative(
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PC_AUTO_PRODUCTION_MAX_MB",
            16_384.0,
        )
        production_slope_mb_per_unknown = _env_float_nonnegative(
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PC_AUTO_PRODUCTION_MB_PER_UNKNOWN",
            2.8e-2,
        )
        adaptive_mb = max(
            float(adaptive_mb),
            float(base_mb) + float(production_slope_mb_per_unknown) * float(active_n),
        )
        if float(production_max_mb) > 0.0:
            auto_max_mb = max(float(auto_max_mb), float(production_max_mb))
    if float(auto_max_mb) > 0.0:
        adaptive_mb = min(float(adaptive_mb), max(float(base_mb), float(auto_max_mb)))
    return float(max(0.0, adaptive_mb))


@dataclass(frozen=True)
class ActiveFortranV3ReducedFactorPolicy:
    """Resolved host factor settings for the reduced RHSMode=1 Pmat."""

    requested: str
    factor_kind: str
    large_matrix: bool
    ilu_max_size: int
    ilu_size_exceeded: bool
    fill_factor: float
    drop_tol: float
    diag_pivot: float
    permc_requested: str
    permc_candidates: tuple[str, ...]
    permc_spec: str
    scale_norm: str
    max_scale: float
    progress: bool
    lu_large_prefill_size: int
    lu_prefill_safety_factor: float


def resolve_active_fortran_v3_reduced_factor_policy(
    *,
    requested_kind: str,
    matrix_size: int,
    env: Mapping[str, str] | None = None,
) -> ActiveFortranV3ReducedFactorPolicy:
    """Resolve factorization defaults for the Fortran-v3-reduced active Pmat."""

    env_map = os.environ if env is None else env
    requested = str(requested_kind).strip().lower().replace("-", "_")
    factor_kind = str(
        env_map.get("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_FACTOR_KIND", "")
    ).strip().lower()
    if not factor_kind:
        factor_kind = "ilu" if "ilu" in requested or requested.endswith("pc_matrix") else "lu"
    if factor_kind not in {"ilu", "spilu", "lu", "splu"}:
        factor_kind = "ilu"
    factor_kind = "lu" if factor_kind in {"lu", "splu"} else "ilu"

    n = int(matrix_size)
    large_matrix = n >= int(
        _policy_env_int(env_map, "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_LARGE_SIZE", 300_000)
    )
    ilu_max_size = int(
        _policy_env_int(env_map, "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_ILU_MAX_SIZE", 350_000)
    )
    ilu_size_exceeded = bool(factor_kind == "ilu" and int(ilu_max_size) > 0 and n > int(ilu_max_size))

    fill_factor_default = 3.0 if factor_kind == "ilu" else 12.0
    drop_tol_default = 3.0e-3 if factor_kind == "ilu" else 0.0
    if bool(large_matrix) and factor_kind == "ilu":
        fill_factor_default = float(
            _policy_env_float(env_map, "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_LARGE_FILL_FACTOR", 1.2)
        )
        drop_tol_default = float(
            _policy_env_float(env_map, "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_LARGE_DROP_TOL", 5.0e-2)
        )

    fill_factor = max(
        1.0,
        float(
            _policy_env_float(
                env_map,
                "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_FILL_FACTOR",
                fill_factor_default,
            )
        ),
    )
    drop_tol = float(
        _policy_env_float(env_map, "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_DROP_TOL", drop_tol_default)
    )
    diag_pivot = float(
        _policy_env_float(env_map, "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_DIAG_PIVOT_THRESH", 0.0)
    )
    permc_requested = str(
        env_map.get("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_PERMC_SPEC", "")
    ).strip().upper()
    permc_candidates = active_fortran_v3_reduced_permc_candidates(
        requested=permc_requested,
        factor_kind=factor_kind,
    )
    scale_norm = str(env_map.get("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_SCALE_NORM", "l1")).strip().lower()
    if scale_norm not in {"l1", "l2", "max"}:
        scale_norm = "l1"
    max_scale = max(
        1.0,
        float(_policy_env_float(env_map, "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_MAX_SCALE", 1.0e6)),
    )
    lu_large_prefill_size = int(
        _policy_env_int(env_map, "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_LU_LARGE_SIZE", 300_000)
    )
    lu_prefill_default = 4.5
    if factor_kind == "lu" and n >= int(lu_large_prefill_size):
        lu_prefill_default = float(
            _policy_env_float(
                env_map,
                "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_LU_LARGE_PREFILL_SAFETY_FACTOR",
                32.0,
            )
        )
    lu_prefill_safety_factor = max(
        1.0,
        float(
            _policy_env_float(
                env_map,
                "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_LU_PREFILL_SAFETY_FACTOR",
                lu_prefill_default,
            )
        ),
    )
    progress = _policy_env_bool(
        env_map,
        "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_PROGRESS",
        bool(large_matrix),
    )
    return ActiveFortranV3ReducedFactorPolicy(
        requested=str(requested),
        factor_kind=str(factor_kind),
        large_matrix=bool(large_matrix),
        ilu_max_size=int(ilu_max_size),
        ilu_size_exceeded=bool(ilu_size_exceeded),
        fill_factor=float(fill_factor),
        drop_tol=float(drop_tol),
        diag_pivot=float(diag_pivot),
        permc_requested=str(permc_requested),
        permc_candidates=tuple(str(candidate) for candidate in permc_candidates),
        permc_spec=str(permc_candidates[0]),
        scale_norm=str(scale_norm),
        max_scale=float(max_scale),
        progress=bool(progress),
        lu_large_prefill_size=int(lu_large_prefill_size),
        lu_prefill_safety_factor=float(lu_prefill_safety_factor),
    )


def active_fortran_v3_reduced_permc_candidates(*, requested: str, factor_kind: str) -> tuple[str, ...]:
    """Return SuperLU ordering candidates for the active Fortran-v3 factor.

    ``RCM`` is implemented by an explicit symmetric permutation before calling
    SuperLU with ``NATURAL`` ordering. This mirrors SFINCS Fortran v3's PETSc
    serial sparse-direct fallback, where ``MATORDERINGRCM`` is requested for
    the preconditioner factor.
    """

    valid = ("RCM", "NATURAL", "MMD_ATA", "MMD_AT_PLUS_A", "COLAMD")
    requested_use = str(requested or "").strip().upper()
    if requested_use in valid:
        return (requested_use,)
    if requested_use and requested_use not in {"AUTO", "DEFAULT"}:
        return ("COLAMD",)
    if str(factor_kind).strip().lower() == "lu":
        return ("NATURAL", "COLAMD")
    return ("COLAMD",)


# Reduced-Pmat symbolic elimination plan.
@dataclass(frozen=True)
class RHS1ReducedPmatGroup:
    """One contiguous symbolic group in reduced active-pitch ordering."""

    name: str
    kind: str
    indices: np.ndarray

    @property
    def size(self) -> int:
        return int(self.indices.size)

    @property
    def dense_lu_nbytes_estimate(self) -> int:
        return int(self.size * self.size * 8)


@dataclass(frozen=True)
class RHS1ReducedPmatEliminationPlan:
    """Bounded symbolic ordering for direct reduced-Pmat assembly/factorization."""

    layout: RHS1CompressedPitchLayout
    interior_groups: tuple[RHS1ReducedPmatGroup, ...]
    separator_group: RHS1ReducedPmatGroup
    tail_group: RHS1ReducedPmatGroup
    root_group: RHS1ReducedPmatGroup
    permutation: np.ndarray
    inverse_permutation: np.ndarray
    selected_separator_ells: tuple[int, ...]
    selected_separator_x_indices: tuple[int, ...]
    max_interior_group_size: int
    max_separator_size: int

    @property
    def interior_size(self) -> int:
        return int(sum(group.size for group in self.interior_groups))

    @property
    def separator_size(self) -> int:
        return int(self.separator_group.size)

    @property
    def tail_size(self) -> int:
        return int(self.tail_group.size)

    @property
    def root_size(self) -> int:
        return int(self.root_group.size)

    @property
    def root_dense_nbytes_estimate(self) -> int:
        return int(self.root_group.dense_lu_nbytes_estimate)

    @property
    def max_interior_dense_nbytes_estimate(self) -> int:
        if not self.interior_groups:
            return 0
        return int(max(group.dense_lu_nbytes_estimate for group in self.interior_groups))

    def metadata(self) -> dict[str, object]:
        return {
            "reduced_size": int(self.layout.reduced_size),
            "interior_group_count": int(len(self.interior_groups)),
            "interior_size": int(self.interior_size),
            "separator_size": int(self.separator_size),
            "tail_size": int(self.tail_size),
            "root_size": int(self.root_size),
            "selected_separator_ells": tuple(int(v) for v in self.selected_separator_ells),
            "selected_separator_x_indices": tuple(int(v) for v in self.selected_separator_x_indices),
            "max_interior_group_size": int(self.max_interior_group_size),
            "max_separator_size": int(self.max_separator_size),
            "max_interior_dense_nbytes_estimate": int(self.max_interior_dense_nbytes_estimate),
            "root_dense_nbytes_estimate": int(self.root_dense_nbytes_estimate),
        }


def _normalize_separator_ells(values: Iterable[int], *, n_xi: int) -> tuple[int, ...]:
    out = sorted({int(v) for v in values if 0 <= int(v) < int(n_xi)})
    return tuple(out)


def _selected_x_indices(n_x: int, *, n_theta: int, n_zeta: int, ell_count: int, max_separator_size: int) -> tuple[int, ...]:
    if n_x <= 0 or ell_count <= 0 or max_separator_size <= 0:
        return tuple()
    per_x_size = max(1, int(ell_count) * int(n_theta) * int(n_zeta))
    max_x = max(1, int(max_separator_size) // per_x_size)
    if max_x >= int(n_x):
        return tuple(range(int(n_x)))
    # Always keep the endpoints, then add approximately uniform interior x
    # separators.  This mirrors the purpose of nested dissection separators
    # without committing to a numeric factor implementation here.
    raw = np.linspace(0, int(n_x) - 1, num=max_x, dtype=np.int64)
    return tuple(int(v) for v in np.unique(raw))


def _split_group_indices(indices: np.ndarray, *, name_prefix: str, kind: str, max_size: int) -> tuple[RHS1ReducedPmatGroup, ...]:
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0:
        return tuple()
    max_size = max(1, int(max_size))
    groups: list[RHS1ReducedPmatGroup] = []
    for start in range(0, int(indices.size), max_size):
        chunk = indices[start : start + max_size]
        groups.append(
            RHS1ReducedPmatGroup(
                name=f"{name_prefix}_{len(groups)}",
                kind=kind,
                indices=np.asarray(chunk, dtype=np.int64),
            )
        )
    return tuple(groups)


def build_rhs1_reduced_pmat_elimination_plan(
    op_or_layout: object,
    *,
    separator_ells: Iterable[int] = (0,),
    max_interior_group_size: int = 32768,
    max_separator_size: int = 8192,
) -> RHS1ReducedPmatEliminationPlan:
    """Build a bounded symbolic plan over compressed RHSMode=1 active DOFs.

    Parameters
    ----------
    op_or_layout:
      Either a :class:`RHS1CompressedPitchLayout` or an operator accepted by
      :func:`build_rhs1_compressed_pitch_layout`.
    separator_ells:
      Pitch modes retained in the Schur root candidate.  ``ell=0`` is the
      default because density/source/profile moments couple most directly to
      that mode.
    max_interior_group_size:
      Maximum rows in one kinetic interior group before symbolic splitting.
    max_separator_size:
      Maximum selected kinetic separator rows.  Tail rows are appended to the
      root even if this bound is saturated.
    """

    layout = (
        op_or_layout
        if isinstance(op_or_layout, RHS1CompressedPitchLayout)
        else build_rhs1_compressed_pitch_layout(op_or_layout)
    )
    separator_ells_norm = _normalize_separator_ells(separator_ells, n_xi=layout.n_xi)
    selected_x = _selected_x_indices(
        layout.n_x,
        n_theta=layout.n_theta,
        n_zeta=layout.n_zeta,
        ell_count=len(separator_ells_norm),
        max_separator_size=max_separator_size,
    )

    separator_reduced: list[int] = []
    selected_x_set = set(selected_x)
    selected_ell_set = set(separator_ells_norm)
    for species in range(layout.n_species):
        for x_index in selected_x:
            n_active_l = int(layout.nxi_for_x[x_index])
            for ell in separator_ells_norm:
                if ell >= n_active_l:
                    continue
                for theta in range(layout.n_theta):
                    for zeta in range(layout.n_zeta):
                        separator_reduced.append(layout.reduced_kinetic_index(species, x_index, ell, theta, zeta))

    separator_indices = np.asarray(sorted(set(separator_reduced)), dtype=np.int64)
    all_kinetic = np.arange(layout.kinetic_active_size, dtype=np.int64)
    if separator_indices.size:
        interior_mask = np.ones((layout.kinetic_active_size,), dtype=bool)
        interior_mask[separator_indices] = False
        interior_reduced = all_kinetic[interior_mask]
    else:
        interior_reduced = all_kinetic

    interior_groups: list[RHS1ReducedPmatGroup] = []
    for species in range(layout.n_species):
        for x_index in range(layout.n_x):
            block = np.arange(
                layout.species_x_reduced_slice(species, x_index).start,
                layout.species_x_reduced_slice(species, x_index).stop,
                dtype=np.int64,
            )
            if x_index in selected_x_set and separator_ells_norm:
                keep = np.ones((block.size,), dtype=bool)
                for ell in selected_ell_set:
                    if ell >= int(layout.nxi_for_x[x_index]):
                        continue
                    ell_start = (
                        layout.reduced_kinetic_index(species, x_index, ell, 0, 0)
                        - layout.species_x_reduced_slice(species, x_index).start
                    )
                    keep[ell_start : ell_start + layout.n_theta * layout.n_zeta] = False
                block = block[keep]
            interior_groups.extend(
                _split_group_indices(
                    block,
                    name_prefix=f"s{species}_x{x_index}",
                    kind="kinetic_interior",
                    max_size=max_interior_group_size,
                )
            )

    interior_concat = np.concatenate([group.indices for group in interior_groups]) if interior_groups else np.asarray([], dtype=np.int64)
    if not np.array_equal(np.sort(interior_concat), np.sort(interior_reduced)):
        raise ValueError("reduced-Pmat interior groups do not cover the expected kinetic interior")

    tail_indices = np.arange(layout.tail_reduced_start, layout.reduced_size, dtype=np.int64)
    separator_group = RHS1ReducedPmatGroup("kinetic_separator", "kinetic_separator", separator_indices)
    tail_group = RHS1ReducedPmatGroup("tail", "tail", tail_indices)
    root_indices = np.concatenate([separator_indices, tail_indices])
    root_group = RHS1ReducedPmatGroup("schur_root", "schur_root", root_indices)

    permutation = np.concatenate([interior_concat, root_indices]).astype(np.int64, copy=False)
    if permutation.size != layout.reduced_size or np.unique(permutation).size != layout.reduced_size:
        raise ValueError("reduced-Pmat symbolic permutation is not a complete permutation")
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(permutation.size, dtype=np.int64)

    return RHS1ReducedPmatEliminationPlan(
        layout=layout,
        interior_groups=tuple(interior_groups),
        separator_group=separator_group,
        tail_group=tail_group,
        root_group=root_group,
        permutation=permutation,
        inverse_permutation=inverse,
        selected_separator_ells=separator_ells_norm,
        selected_separator_x_indices=selected_x,
        max_interior_group_size=max(1, int(max_interior_group_size)),
        max_separator_size=max(1, int(max_separator_size)),
    )

__all__ = (
    "ActiveFortranV3ReducedFactorPolicy",
    "build_rhs1_reduced_pmat_elimination_plan",
    "RHS1ReducedPmatGroup",
    "RHS1ReducedPmatEliminationPlan",
    "active_fortran_v3_reduced_preconditioner_matrix",
    "active_fortran_v3_reduced_permc_candidates",
    "build_active_fortran_v3_reduced_sparse_factor_preconditioner",
    "estimate_spilu_factor_nbytes",
    "resolve_active_fortran_v3_reduced_factor_policy",
    "sparse_equilibration_scale",
    "sparse_lu_factor_nbytes",
)


def build_active_fortran_v3_reduced_sparse_factor_preconditioner(
    *,
    matrix: Any,
    layout: RHS1BlockLayout,
    active_indices: Any | None,
    requested_kind: str,
    regularization: float,
    max_factor_nbytes: int,
    t0: float,
    preconditioner_x: int | None = None,
    preconditioner_xi: int | None = None,
    preconditioner_species: int | None = None,
    preconditioner_x_min_l: int | None = None,
) -> RHS1StructuredFullCSRPreconditioner:
    """Factor a Fortran-v3-inspired reduced active preconditioner matrix."""

    from scipy.sparse.csgraph import reverse_cuthill_mckee  # noqa: PLC0415
    from scipy.sparse.linalg import LinearOperator, spilu, splu  # noqa: PLC0415

    try:
        reduced, reduction_metadata = active_fortran_v3_reduced_preconditioner_matrix(
            matrix=matrix,
            layout=layout,
            active_indices=active_indices,
            regularization=regularization,
            preconditioner_x=preconditioner_x,
            preconditioner_xi=preconditioner_xi,
            preconditioner_species=preconditioner_species,
            preconditioner_x_min_l=preconditioner_x_min_l,
        )
    except ValueError as exc:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_fortran_v3_pc_matrix",
            reason="active_fortran_v3_pc_matrix_invalid_layout",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={"error": str(exc)},
        )

    requested = str(requested_kind).strip().lower().replace("-", "_")
    use_symbolic_plan = bool(
        _env_bool(
            "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_SYMBOLIC_PLAN",
            "planned" in requested,
        )
    )
    plan_permutation: np.ndarray | None = None
    if use_symbolic_plan:
        separator_ells_raw = os.environ.get(
            "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_SYMBOLIC_PLAN_SEPARATOR_ELLS",
            "0",
        )
        separator_ells: list[int] = []
        for token in separator_ells_raw.replace(";", ",").split(","):
            token = token.strip()
            if not token:
                continue
            try:
                separator_ells.append(int(token))
            except ValueError:
                continue
        if not separator_ells:
            separator_ells = [0]
        try:
            compressed_layout = infer_rhs1_compressed_pitch_layout_from_active_indices(
                layout,
                active_indices,
            )
            plan = build_rhs1_reduced_pmat_elimination_plan(
                compressed_layout,
                separator_ells=tuple(separator_ells),
                max_interior_group_size=max(
                    1,
                    int(
                        _env_int(
                            "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_SYMBOLIC_PLAN_MAX_INTERIOR_GROUP_SIZE",
                            32768,
                        )
                    ),
                ),
                max_separator_size=max(
                    1,
                    int(
                        _env_int(
                            "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_SYMBOLIC_PLAN_MAX_SEPARATOR_SIZE",
                            8192,
                        )
                    ),
                ),
            )
        except ValueError as exc:
            return RHS1StructuredFullCSRPreconditioner(
                operator=None,
                selected=False,
                kind="active_fortran_v3_reduced_planned_matrix",
                reason="active_fortran_v3_reduced_symbolic_plan_invalid",
                setup_s=max(0.0, time.perf_counter() - t0),
                metadata={
                    **reduction_metadata,
                    "symbolic_plan_requested": True,
                    "error": str(exc),
                },
            )
        plan_permutation = np.asarray(plan.permutation, dtype=np.int64)
        if plan_permutation.size != int(reduced.shape[0]) or np.unique(plan_permutation).size != int(
            reduced.shape[0]
        ):
            return RHS1StructuredFullCSRPreconditioner(
                operator=None,
                selected=False,
                kind="active_fortran_v3_reduced_planned_matrix",
                reason="active_fortran_v3_reduced_symbolic_plan_incomplete_permutation",
                setup_s=max(0.0, time.perf_counter() - t0),
                metadata={
                    **reduction_metadata,
                    "symbolic_plan_requested": True,
                    "symbolic_plan_size": int(plan_permutation.size),
                    "matrix_size": int(reduced.shape[0]),
                },
            )
        reduced = reduced[plan_permutation, :][:, plan_permutation].tocsr()
        reduction_metadata = {
            **reduction_metadata,
            "symbolic_plan_requested": True,
            "symbolic_plan_applied": True,
            "symbolic_plan_permutation_nbytes": int(plan_permutation.nbytes),
            "reduced_pmat_symbolic_plan": plan.metadata(),
        }
    else:
        reduction_metadata = {
            **reduction_metadata,
            "symbolic_plan_requested": False,
            "symbolic_plan_applied": False,
        }

    max_size = _env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_MAX_SIZE", 1_000_000)
    n = int(reduced.shape[0])
    if int(max_size) > 0 and n > int(max_size):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_fortran_v3_pc_matrix",
            reason=f"active_fortran_v3_pc_matrix_size_exceeded:{n}>{int(max_size)}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={**reduction_metadata, "max_size": int(max_size)},
        )

    factor_policy = resolve_active_fortran_v3_reduced_factor_policy(
        requested_kind=requested,
        matrix_size=int(n),
    )
    factor_kind = str(factor_policy.factor_kind)
    large_matrix = bool(factor_policy.large_matrix)
    if bool(factor_policy.ilu_size_exceeded):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_fortran_v3_pc_matrix",
            reason=f"active_fortran_v3_pc_matrix_ilu_size_exceeded:{n}>{int(factor_policy.ilu_max_size)}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                **reduction_metadata,
                "factor_kind": str(factor_kind),
                "matrix_size": int(n),
                "ilu_max_size": int(factor_policy.ilu_max_size),
                "note": (
                    "large ILU setup is intentionally fail-closed; raise "
                    "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_ILU_MAX_SIZE "
                    "only for explicit diagnostics"
                ),
            },
        )
    fill_factor = float(factor_policy.fill_factor)
    drop_tol = float(factor_policy.drop_tol)
    diag_pivot = float(factor_policy.diag_pivot)
    permc_env = str(factor_policy.permc_requested)
    permc_candidates = tuple(str(candidate) for candidate in factor_policy.permc_candidates)
    permc_spec = str(permc_candidates[0])
    scale_norm = str(factor_policy.scale_norm)
    max_scale = float(factor_policy.max_scale)

    estimate = estimate_spilu_factor_nbytes(matrix=reduced, fill_factor=fill_factor) + 2 * n * np.dtype(
        np.float64
    ).itemsize
    if bool(factor_policy.progress):
        print(
            "active_fortran_v3_pc_matrix: factor setup "
            f"n={n} nnz={int(reduced.nnz)} factor_kind={factor_kind} "
            f"fill_factor={float(fill_factor):.3g} drop_tol={float(drop_tol):.3g} "
            f"estimate_mb={float(estimate) / (1024.0 * 1024.0):.1f} "
            f"budget_mb={float(max_factor_nbytes) / (1024.0 * 1024.0):.1f}",
            flush=True,
        )
    if estimate > int(max_factor_nbytes):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_fortran_v3_pc_matrix",
            reason=f"active_fortran_v3_pc_matrix_budget_exceeded:{estimate}>{int(max_factor_nbytes)}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                **reduction_metadata,
                "factor_kind": str(factor_kind),
                "factor_nbytes_estimate": int(estimate),
                "max_factor_nbytes": int(max_factor_nbytes),
                "fill_factor": float(fill_factor),
                "drop_tol": float(drop_tol),
                "large_matrix_defaults": bool(large_matrix),
                "permc_spec": str(permc_spec),
                "permc_spec_requested": str(permc_env or "AUTO"),
                "permc_spec_candidates": tuple(str(candidate) for candidate in permc_candidates),
            },
        )
    lu_large_prefill_size = int(factor_policy.lu_large_prefill_size)
    lu_prefill_safety_factor = float(factor_policy.lu_prefill_safety_factor)
    lu_prefill_estimate = int(np.ceil(float(estimate) * float(lu_prefill_safety_factor)))
    if factor_kind == "lu" and lu_prefill_estimate > int(max_factor_nbytes):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_fortran_v3_pc_matrix",
            reason=(
                "active_fortran_v3_pc_matrix_lu_prefill_budget_exceeded:"
                f"{lu_prefill_estimate}>{int(max_factor_nbytes)}"
            ),
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                **reduction_metadata,
                "factor_kind": str(factor_kind),
                "factor_nbytes_estimate": int(estimate),
                "factor_nbytes_prefill_estimate": int(lu_prefill_estimate),
                "lu_prefill_safety_factor": float(lu_prefill_safety_factor),
                "lu_large_prefill_size": int(lu_large_prefill_size),
                "max_factor_nbytes": int(max_factor_nbytes),
                "fill_factor": float(fill_factor),
                "drop_tol": float(drop_tol),
                "large_matrix_defaults": bool(large_matrix),
                "permc_spec": str(permc_spec),
                "permc_spec_requested": str(permc_env or "AUTO"),
                "permc_spec_candidates": tuple(str(candidate) for candidate in permc_candidates),
            },
        )

    row_scale, row_meta = sparse_equilibration_scale(
        reduced,
        axis=1,
        norm=scale_norm,
        max_scale=max_scale,
    )
    scaled = reduced.multiply(row_scale[:, None]).tocsc()
    col_scale, col_meta = sparse_equilibration_scale(
        scaled,
        axis=0,
        norm=scale_norm,
        max_scale=max_scale,
    )
    scaled = scaled.multiply(col_scale[None, :]).tocsc()
    factor = None
    if use_symbolic_plan:
        selected_kind = (
            "active_fortran_v3_reduced_planned_lu"
            if factor_kind == "lu"
            else "active_fortran_v3_reduced_planned_ilu"
        )
    else:
        selected_kind = "active_fortran_v3_reduced_lu" if factor_kind == "lu" else "active_fortran_v3_reduced_ilu"
    permc_failures: list[dict[str, object]] = []
    selected_permutation: np.ndarray | None = None
    selected_superlu_permc = str(permc_spec)
    for candidate_permc in permc_candidates:
        candidate_permc_use = str(candidate_permc).upper()
        factor_matrix = scaled
        factor_permutation: np.ndarray | None = None
        superlu_permc = candidate_permc_use
        if candidate_permc_use == "RCM":
            factor_permutation = np.asarray(
                reverse_cuthill_mckee(scaled.tocsr(), symmetric_mode=False),
                dtype=np.int64,
            )
            factor_matrix = scaled[factor_permutation, :][:, factor_permutation].tocsc()
            superlu_permc = "NATURAL"
        try:
            if factor_kind == "lu":
                factor = splu(factor_matrix, permc_spec=str(superlu_permc), diag_pivot_thresh=float(diag_pivot))
            else:
                factor = spilu(
                    factor_matrix,
                    drop_tol=float(drop_tol),
                    fill_factor=float(fill_factor),
                    permc_spec=str(superlu_permc),
                    diag_pivot_thresh=float(diag_pivot),
                )
            permc_spec = str(candidate_permc_use)
            selected_permutation = factor_permutation
            selected_superlu_permc = str(superlu_permc)
            break
        except Exception as exc:  # noqa: BLE001
            permc_failures.append(
                {
                    "permc_spec": str(candidate_permc_use),
                    "superlu_permc_spec": str(superlu_permc),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    if factor is None:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_fortran_v3_pc_matrix",
            reason="active_fortran_v3_pc_matrix_failed:all_permc_specs",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                **reduction_metadata,
                "error": str(permc_failures[-1]["error"]) if permc_failures else "",
                "permc_failures": tuple(dict(entry) for entry in permc_failures),
                "factor_kind": str(factor_kind),
                "factor_nbytes_estimate": int(estimate),
                "fill_factor": float(fill_factor),
                "drop_tol": float(drop_tol),
                "large_matrix_defaults": bool(large_matrix),
                "permc_spec": str(permc_spec),
                "permc_spec_requested": str(permc_env or "AUTO"),
                "permc_spec_candidates": tuple(str(candidate) for candidate in permc_candidates),
                "row_scaling": row_meta,
                "column_scaling": col_meta,
            },
        )

    factor_nbytes = int(
        sparse_lu_factor_nbytes(factor)
        + row_scale.nbytes
        + col_scale.nbytes
        + (0 if plan_permutation is None else plan_permutation.nbytes)
    )
    if factor_nbytes > int(max_factor_nbytes):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_fortran_v3_pc_matrix",
            reason=f"active_fortran_v3_pc_matrix_factor_budget_exceeded:{factor_nbytes}>{int(max_factor_nbytes)}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                **reduction_metadata,
                "factor_kind": str(factor_kind),
                "factor_nbytes_estimate": int(estimate),
                "factor_nbytes_actual": int(factor_nbytes),
                "max_factor_nbytes": int(max_factor_nbytes),
                "permc_spec": str(permc_spec),
                "permc_spec_requested": str(permc_env or "AUTO"),
                "permc_spec_candidates": tuple(str(candidate) for candidate in permc_candidates),
                "permc_failures": tuple(dict(entry) for entry in permc_failures),
            },
        )

    def apply(x: Any) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64).reshape((-1,))
        arr_factor_order = arr if plan_permutation is None else arr[plan_permutation]
        scaled_rhs = row_scale * arr_factor_order
        if selected_permutation is None:
            scaled_solution = np.asarray(factor.solve(scaled_rhs), dtype=np.float64).reshape((-1,))
        else:
            permuted_solution = np.asarray(
                factor.solve(scaled_rhs[selected_permutation]),
                dtype=np.float64,
            ).reshape((-1,))
            scaled_solution = np.empty_like(permuted_solution)
            scaled_solution[selected_permutation] = permuted_solution
        solution_factor_order = col_scale * scaled_solution
        if plan_permutation is None:
            return solution_factor_order
        solution = np.empty_like(solution_factor_order)
        solution[plan_permutation] = solution_factor_order
        return solution

    operator = LinearOperator(reduced.shape, matvec=apply, dtype=np.float64)
    return RHS1StructuredFullCSRPreconditioner(
        operator=operator,
        selected=True,
        kind=selected_kind,
        reason="complete",
        setup_s=max(0.0, time.perf_counter() - t0),
        metadata={
            **reduction_metadata,
            "requested_kind": str(requested_kind),
            "architecture": "fortran_v3_reduced_active_pc_matrix",
            "factor_kind": str(factor_kind),
            "factor_nnz": int(factor.L.nnz + factor.U.nnz),
            "factor_nbytes_estimate": int(estimate),
            "factor_nbytes_prefill_estimate": int(lu_prefill_estimate) if factor_kind == "lu" else int(estimate),
            "factor_nbytes_actual": int(factor_nbytes),
            "lu_prefill_safety_factor": float(lu_prefill_safety_factor) if factor_kind == "lu" else None,
            "lu_large_prefill_size": int(lu_large_prefill_size),
            "max_factor_nbytes": int(max_factor_nbytes),
            "fill_factor": float(fill_factor),
            "drop_tol": float(drop_tol),
            "large_matrix_defaults": bool(large_matrix),
            "diag_pivot_thresh": float(diag_pivot),
            "permc_spec": str(permc_spec),
            "superlu_permc_spec": str(selected_superlu_permc),
            "explicit_symmetric_ordering": bool(selected_permutation is not None),
            "symbolic_plan_permutation": bool(plan_permutation is not None),
            "permc_spec_requested": str(permc_env or "AUTO"),
            "permc_spec_candidates": tuple(str(candidate) for candidate in permc_candidates),
            "permc_failures": tuple(dict(entry) for entry in permc_failures),
            "scale_norm": str(scale_norm),
            "max_scale": float(max_scale),
            "row_scaling": row_meta,
            "column_scaling": col_meta,
            "requires_preflight": bool(factor_kind == "ilu" or use_symbolic_plan),
            "admission_policy": (
                "external_true_residual_required"
                if bool(factor_kind == "ilu" or use_symbolic_plan)
                else "exact_reduced_matrix_factor"
            ),
        },
    )


def active_fortran_v3_reduced_preconditioner_matrix(
    *,
    matrix: Any,
    layout: RHS1BlockLayout,
    active_indices: Any | None,
    regularization: float,
    preconditioner_x: int | None = None,
    preconditioner_xi: int | None = None,
    preconditioner_species: int | None = None,
    preconditioner_x_min_l: int | None = None,
) -> tuple[Any, dict[str, object]]:
    """Return an active CSR matrix with Fortran-v3-style support reduction."""

    matrix_csr = matrix.tocsr().astype(np.float64)
    active = (
        np.arange(int(matrix_csr.shape[0]), dtype=np.int64)
        if active_indices is None
        else np.asarray(active_indices, dtype=np.int64).reshape((-1,))
    )
    active_layout = RHS1ActiveBlockLayout.from_layout(layout, active)
    if active.size != int(matrix_csr.shape[0]):
        raise ValueError("active_indices size must match active matrix shape")
    if matrix_csr.shape[0] != matrix_csr.shape[1]:
        raise ValueError("active matrix must be square")

    matrix_coo = matrix_csr.tocoo(copy=False)
    row_full = active[np.asarray(matrix_coo.row, dtype=np.int64)]
    col_full = active[np.asarray(matrix_coo.col, dtype=np.int64)]
    kinetic_entry = (row_full < int(layout.f_size)) & (col_full < int(layout.f_size))
    keep = np.ones(matrix_coo.nnz, dtype=bool)
    preconditioner_x_use = (
        int(preconditioner_x)
        if preconditioner_x is not None
        else int(_env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_PRECONDITIONER_X", 1))
    )
    preconditioner_xi_use = (
        int(preconditioner_xi)
        if preconditioner_xi is not None
        else int(_env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_PRECONDITIONER_XI", 1))
    )
    preconditioner_species_use = (
        int(preconditioner_species)
        if preconditioner_species is not None
        else int(_env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_PRECONDITIONER_SPECIES", 1))
    )
    preconditioner_x_min_l_use = max(
        0,
        int(preconditioner_x_min_l)
        if preconditioner_x_min_l is not None
        else int(_env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_PRECONDITIONER_X_MIN_L", 0)),
    )
    dropped: dict[str, int] = {
        "x_nonlocal": 0,
        "x_unsupported": 0,
        "ell_two": 0,
        "ell_outside_support": 0,
        "species_cross": 0,
    }

    if np.any(kinetic_entry):
        kinetic_positions = np.flatnonzero(kinetic_entry)
        row_decoded = layout.decode_kinetic_indices(row_full[kinetic_positions])
        col_decoded = layout.decode_kinetic_indices(col_full[kinetic_positions])

        if int(preconditioner_species_use) > 0:
            mask = row_decoded.species != col_decoded.species
            keep[kinetic_positions[mask]] = False
            dropped["species_cross"] = int(np.count_nonzero(mask))

        if int(preconditioner_x_use) > 0:
            row_x = row_decoded.x.astype(np.int64, copy=False)
            col_x = col_decoded.x.astype(np.int64, copy=False)
            if int(preconditioner_x_use) == 1:
                x_allowed = row_x == col_x
            elif int(preconditioner_x_use) == 2:
                x_allowed = col_x >= row_x
            elif int(preconditioner_x_use) in {3, 5}:
                x_allowed = np.abs(row_x - col_x) <= 1
            elif int(preconditioner_x_use) == 4:
                x_allowed = (col_x == row_x) | (col_x == row_x + 1)
            else:
                x_allowed = row_x == col_x
            if int(preconditioner_x_min_l_use) > 0:
                x_gate = row_decoded.ell >= int(preconditioner_x_min_l_use)
                x_allowed = np.where(x_gate, x_allowed, True)
            mask = ~x_allowed
            keep[kinetic_positions[mask]] = False
            dropped["x_unsupported"] = int(np.count_nonzero(mask))
            dropped["x_nonlocal"] = int(np.count_nonzero(mask & (row_decoded.x != col_decoded.x)))

        ell_distance = np.abs(row_decoded.ell - col_decoded.ell)
        ell_radius = 1 if int(preconditioner_xi_use) > 0 else 2
        mask = ell_distance > int(ell_radius)
        if int(preconditioner_xi_use) > 0:
            dropped["ell_two"] = int(np.count_nonzero(ell_distance == 2))
        if np.any(mask):
            keep[kinetic_positions[mask]] = False
            dropped["ell_outside_support"] = int(np.count_nonzero(mask))

    reduced = sp.coo_matrix(
        (matrix_coo.data[keep], (matrix_coo.row[keep], matrix_coo.col[keep])),
        shape=matrix_csr.shape,
        dtype=np.float64,
    ).tocsr()
    reduced.sum_duplicates()
    reduced.eliminate_zeros()
    diagonal_shift = float(
        _env_float(
            "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_DIAGONAL_SHIFT",
            max(float(abs(regularization)), 1.0e-14),
        )
    )
    if diagonal_shift > 0.0:
        scale = max(float(np.max(np.abs(reduced.data))) if reduced.nnz else 0.0, 1.0)
        reduced = reduced + float(diagonal_shift * scale) * sp.eye(reduced.shape[0], dtype=np.float64, format="csr")
    metadata = {
        "matrix_shape": tuple(int(v) for v in matrix_csr.shape),
        "matrix_nnz": int(matrix_csr.nnz),
        "reduced_matrix_nnz": int(reduced.nnz),
        "reduced_nnz_ratio": float(reduced.nnz / max(int(matrix_csr.nnz), 1)),
        "dropped_entries": {str(k): int(v) for k, v in dropped.items()},
        "diagonal_shift": float(diagonal_shift),
        "preconditioner_x": int(preconditioner_x_use),
        "preconditioner_xi": int(preconditioner_xi_use),
        "preconditioner_species": int(preconditioner_species_use),
        "preconditioner_x_min_l": int(preconditioner_x_min_l_use),
        "fortran_reduced_filter": "layout_decoded_supports",
        "active_layout": active_layout.to_dict(),
        "fortran_v3_source": (
            "solver.F90 uses GMRES+PCLU on whichMatrix=0; populateMatrix.F90 "
            "drops off-by-2 ell terms when preconditioner_xi=1 and createGrids.F90 "
            "defaults preconditioner_x=1 to diagonal x derivative stencils."
        ),
        "implementation_note": (
            "This is an active-CSR reduction of the exact operator. The next "
            "production step is direct term-level assembly of the v3 whichMatrix=0 "
            "operator to avoid first materializing the full true CSR."
        ),
    }
    return reduced, metadata


def estimate_spilu_factor_nbytes(*, matrix: Any, fill_factor: float) -> int:
    """Conservative storage estimate for a SuperLU ILU factorization."""

    matrix = matrix.tocsr()
    nnz_estimate = int(np.ceil(max(1.0, float(fill_factor)) * max(1, int(matrix.nnz))))
    data_nbytes = nnz_estimate * np.dtype(np.float64).itemsize
    index_nbytes = nnz_estimate * np.dtype(np.int32).itemsize
    indptr_nbytes = 2 * (int(matrix.shape[0]) + 1) * np.dtype(np.int32).itemsize
    return int(data_nbytes + index_nbytes + indptr_nbytes)


def sparse_equilibration_scale(
    matrix: Any,
    *,
    axis: int,
    norm: str,
    max_scale: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Return clipped inverse sparse row/column norms for equilibration."""

    matrix_csr = matrix.tocsr() if int(axis) == 1 else matrix.tocsc()
    abs_matrix = matrix_csr.copy()
    abs_matrix.data = np.abs(np.asarray(abs_matrix.data, dtype=np.float64))
    norm_l = str(norm).strip().lower()
    if norm_l == "l2":
        squared = matrix_csr.copy()
        squared.data = np.square(np.asarray(squared.data, dtype=np.float64))
        values = np.sqrt(np.asarray(squared.sum(axis=1 if int(axis) == 1 else 0), dtype=np.float64).reshape((-1,)))
    elif norm_l == "max":
        values = np.asarray(abs_matrix.max(axis=1 if int(axis) == 1 else 0).toarray(), dtype=np.float64).reshape((-1,))
    else:
        norm_l = "l1"
        values = np.asarray(abs_matrix.sum(axis=1 if int(axis) == 1 else 0), dtype=np.float64).reshape((-1,))
    scale = np.ones_like(values, dtype=np.float64)
    good = np.isfinite(values) & (values > 0.0)
    scale[good] = 1.0 / values[good]
    max_scale_use = max(1.0, float(max_scale))
    scale = np.clip(scale, 1.0 / max_scale_use, max_scale_use)
    metadata = {
        "axis": "row" if int(axis) == 1 else "column",
        "norm": str(norm_l),
        "size": int(scale.size),
        "norm_min": float(np.min(values)) if values.size else 0.0,
        "norm_max": float(np.max(values)) if values.size else 0.0,
        "scale_min": float(np.min(scale)) if scale.size else 0.0,
        "scale_max": float(np.max(scale)) if scale.size else 0.0,
        "zero_or_invalid_norm_count": int(np.count_nonzero(~good)),
    }
    return scale, metadata


def sparse_lu_factor_nbytes(factor: Any) -> int:
    """Return actual SuperLU factor storage as CSR data/index bytes."""

    return int(_scipy_csr_nbytes(factor.L.tocsr()) + _scipy_csr_nbytes(factor.U.tocsr()))


def _scipy_csr_nbytes(matrix: Any) -> int:
    matrix_csr = matrix.tocsr()
    return int(matrix_csr.data.nbytes + matrix_csr.indices.nbytes + matrix_csr.indptr.nbytes)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or int(default))
    except ValueError:
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or float(default))
    except ValueError:
        return float(default)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return bool(default)
    return value in {"1", "true", "yes", "on"}
