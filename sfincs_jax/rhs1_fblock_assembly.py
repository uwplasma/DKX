"""Partial RHSMode=1 f-block assembly using JAX-native block operators.

This module is the integration seam between the legacy ``V3FBlockOperator``
term objects and the new structured block-COO architecture.  It intentionally
assembles only terms that have been parity-tested as block stencils; unsupported
terms are reported explicitly instead of silently falling back to dense probing.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
import hashlib
import time

import jax.numpy as jnp
import numpy as np

from .rhs1_block_operator import RHS1BlockCOOBuilder, RHS1BlockCOOOperator, RHS1BlockLayout, RHS1BlockLinearOperator
from .rhs1_collision_stencils import (
    build_fokker_planck_collision_f_block_operator,
    build_fokker_planck_phi1_collision_f_block_operator,
    build_pas_collision_f_block_operator,
)
from .rhs1_collisionless_stencils import (
    build_collisionless_f_block_operator,
    build_er_xdot_f_block_operator,
    build_er_xidot_f_block_operator,
    build_exb_theta_f_block_operator,
    build_exb_zeta_f_block_operator,
    build_magnetic_drift_theta_f_block_operator,
    build_magnetic_drift_xidot_f_block_operator,
    build_magnetic_drift_zeta_f_block_operator,
)

_STRUCTURED_FBLOCK_CSR_CACHE: dict[tuple[object, ...], tuple[Any, dict[str, object]]] = {}
_STRUCTURED_FBLOCK_CSR_OBJECT_CACHE: dict[
    tuple[object, ...], tuple[RHS1StructuredFBlockSelection, Any, dict[str, object]]
] = {}


@dataclass(frozen=True)
class RHS1PartialFBlockAssembly:
    """Assembled block operator plus coverage metadata for migrated f-block terms."""

    layout: RHS1BlockLayout
    operator: RHS1BlockCOOOperator
    included_terms: tuple[str, ...]
    unsupported_terms: tuple[str, ...]
    term_nnz_blocks: dict[str, int]
    term_data_nbytes: dict[str, int]

    @property
    def is_complete(self) -> bool:
        """Whether every nonzero f-block term was assembled by this route."""

        return len(self.unsupported_terms) == 0

    def to_dict(self) -> dict[str, object]:
        """Return JSON-friendly coverage and memory metadata."""

        return {
            "shape": tuple(int(v) for v in self.operator.shape),
            "block_size": int(self.operator.block_size),
            "nnz_blocks": int(self.operator.nnz_blocks),
            "data_nbytes": int(self.operator.data_nbytes),
            "included_terms": tuple(self.included_terms),
            "unsupported_terms": tuple(self.unsupported_terms),
            "is_complete": bool(self.is_complete),
            "term_nnz_blocks": {str(k): int(v) for k, v in self.term_nnz_blocks.items()},
            "term_data_nbytes": {str(k): int(v) for k, v in self.term_data_nbytes.items()},
        }


@dataclass(frozen=True)
class RHS1StructuredFBlockSelection:
    """Fail-closed structured f-block operator selection for solver policy."""

    assembly: RHS1PartialFBlockAssembly
    linear_operator: RHS1BlockLinearOperator | None
    selected: bool
    reason: str

    def matvec(self, x_flat: Any) -> jnp.ndarray:
        """Apply the selected f-block operator to a flat kinetic vector."""

        if self.linear_operator is None:
            raise RuntimeError(f"structured f-block operator was not selected: {self.reason}")
        return self.linear_operator.matvec(x_flat)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-friendly selection and coverage metadata."""

        return {
            "selected": bool(self.selected),
            "reason": str(self.reason),
            "operator": None if self.linear_operator is None else self.linear_operator.to_dict(),
            "assembly": self.assembly.to_dict(),
        }


@dataclass(frozen=True)
class RHS1StructuredFBlockCSRSelection:
    """Host CSR materialization of a complete structured RHSMode=1 f-block.

    This object is deliberately host-side and non-autodiff. It is intended for
    CLI/runtime sparse solve and preconditioner experiments where the user
    wants Fortran-style assembled sparse storage without PETSc/MUMPS/SuperLU_DIST
    dependencies. The JAX-native block-COO operator remains the differentiable
    path.
    """

    selection: RHS1StructuredFBlockSelection
    matrix: Any | None
    selected: bool
    reason: str
    cache_hit: bool
    build_s: float
    metadata: dict[str, object]

    def matvec(self, x: Any) -> np.ndarray:
        """Apply the materialized host CSR matrix."""

        if self.matrix is None:
            raise RuntimeError(f"structured f-block CSR operator was not selected: {self.reason}")
        return np.asarray(self.matrix @ np.asarray(x, dtype=np.float64).reshape((-1,)))

    def to_dict(self) -> dict[str, object]:
        """Return JSON-friendly selection and sparse-storage metadata."""

        return {
            "selected": bool(self.selected),
            "reason": str(self.reason),
            "cache_hit": bool(self.cache_hit),
            "build_s": float(self.build_s),
            "metadata": dict(self.metadata),
            "selection": self.selection.to_dict(),
        }


def rhs1_fblock_layout_from_operator(op: Any) -> RHS1BlockLayout:
    """Build an f-block-only layout from a ``V3FBlockOperator``-like object."""

    return RHS1BlockLayout.from_operator(
        SimpleNamespace(
            n_species=op.n_species,
            n_x=op.n_x,
            n_xi=op.n_xi,
            n_theta=op.n_theta,
            n_zeta=op.n_zeta,
            f_size=op.flat_size,
            phi1_size=0,
            extra_size=0,
            total_size=op.flat_size,
            constraint_scheme=1,
            include_phi1=False,
            include_phi1_in_kinetic=False,
            rhs_mode=1,
        )
    )


def clear_structured_rhs1_fblock_csr_cache() -> None:
    """Clear the host CSR reuse cache used by tests and bounded benchmarks."""

    _STRUCTURED_FBLOCK_CSR_CACHE.clear()
    _STRUCTURED_FBLOCK_CSR_OBJECT_CACHE.clear()


def select_structured_rhs1_fblock_operator(
    op: Any,
    *,
    layout: RHS1BlockLayout | None = None,
    include_identity_shift: bool = True,
    phi1_hat_base: Any | None = None,
    drop_tol: float = 0.0,
    require_complete: bool = True,
) -> RHS1StructuredFBlockSelection:
    """Build a structured f-block matvec when migrated term coverage is complete.

    This is the solver-policy seam for replacing dense-probed f-block paths.
    By default it is fail-closed: unsupported terms produce ``selected=False``
    and no linear operator.  Callers that intentionally want the migrated
    partial operator can set ``require_complete=False``.
    """

    assembly = assemble_partial_rhs1_fblock_operator(
        op,
        layout=layout,
        include_identity_shift=include_identity_shift,
        phi1_hat_base=phi1_hat_base,
        drop_tol=drop_tol,
        strict_complete=False,
    )
    if bool(require_complete) and not assembly.is_complete:
        reason = "unsupported_terms:" + ",".join(assembly.unsupported_terms)
        return RHS1StructuredFBlockSelection(
            assembly=assembly,
            linear_operator=None,
            selected=False,
            reason=reason,
        )

    linear_operator = RHS1BlockLinearOperator(
        layout=assembly.layout,
        matvec_fn=assembly.operator.matvec,
        name="rhs1_structured_fblock",
    )
    reason = "complete" if assembly.is_complete else "partial_allowed"
    return RHS1StructuredFBlockSelection(
        assembly=assembly,
        linear_operator=linear_operator,
        selected=True,
        reason=reason,
    )


def select_structured_rhs1_fblock_csr_operator(
    op: Any,
    *,
    layout: RHS1BlockLayout | None = None,
    include_identity_shift: bool = True,
    phi1_hat_base: Any | None = None,
    drop_tol: float = 0.0,
    require_complete: bool = True,
    max_csr_nbytes: int | None = None,
    use_cache: bool = True,
) -> RHS1StructuredFBlockCSRSelection:
    """Build or reuse a host CSR operator from structured f-block stencils.

    The function is fail-closed: unsupported terms, incomplete coverage, or
    memory-budget violations return ``selected=False`` instead of silently
    falling back to dense probing. The cache key is based on the assembled
    block-COO data, so reuse is safe across repeated same-operator calls and
    conservative when physics coefficients change.
    """

    t0 = time.perf_counter()
    object_cache_key = _structured_fblock_csr_object_cache_key(
        op=op,
        include_identity_shift=include_identity_shift,
        phi1_hat_base=phi1_hat_base,
        drop_tol=drop_tol,
        require_complete=require_complete,
    )
    if bool(use_cache) and object_cache_key in _STRUCTURED_FBLOCK_CSR_OBJECT_CACHE:
        selection, matrix, cached_metadata = _STRUCTURED_FBLOCK_CSR_OBJECT_CACHE[object_cache_key]
        metadata = dict(cached_metadata)
        csr_nbytes_estimate = int(metadata.get("csr_nbytes_estimate", 0) or 0)
        if max_csr_nbytes is not None and csr_nbytes_estimate > int(max_csr_nbytes):
            reason = f"csr_budget_exceeded:{csr_nbytes_estimate}>{int(max_csr_nbytes)}"
            return RHS1StructuredFBlockCSRSelection(
                selection=selection,
                matrix=None,
                selected=False,
                reason=reason,
                cache_hit=True,
                build_s=max(0.0, time.perf_counter() - t0),
                metadata={
                    "selected": False,
                    "reason": reason,
                    "csr_nbytes_estimate": csr_nbytes_estimate,
                    "max_csr_nbytes": int(max_csr_nbytes),
                    "object_cache_hit": True,
                },
            )
        metadata["cache_hit"] = True
        metadata["object_cache_hit"] = True
        return RHS1StructuredFBlockCSRSelection(
            selection=selection,
            matrix=matrix,
            selected=True,
            reason="complete",
            cache_hit=True,
            build_s=max(0.0, time.perf_counter() - t0),
            metadata=metadata,
        )

    selection = select_structured_rhs1_fblock_operator(
        op,
        layout=layout,
        include_identity_shift=include_identity_shift,
        phi1_hat_base=phi1_hat_base,
        drop_tol=drop_tol,
        require_complete=require_complete,
    )
    if not bool(selection.selected):
        return RHS1StructuredFBlockCSRSelection(
            selection=selection,
            matrix=None,
            selected=False,
            reason=str(selection.reason),
            cache_hit=False,
            build_s=max(0.0, time.perf_counter() - t0),
            metadata={"selected": False, "reason": str(selection.reason)},
        )

    operator = selection.assembly.operator
    csr_nbytes_estimate = _estimate_block_coo_as_csr_nbytes(operator)
    if max_csr_nbytes is not None and int(csr_nbytes_estimate) > int(max_csr_nbytes):
        reason = f"csr_budget_exceeded:{int(csr_nbytes_estimate)}>{int(max_csr_nbytes)}"
        return RHS1StructuredFBlockCSRSelection(
            selection=selection,
            matrix=None,
            selected=False,
            reason=reason,
            cache_hit=False,
            build_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "selected": False,
                "reason": reason,
                "csr_nbytes_estimate": int(csr_nbytes_estimate),
                "max_csr_nbytes": int(max_csr_nbytes),
            },
        )

    cache_key = _structured_fblock_csr_cache_key(selection=selection, drop_tol=drop_tol)
    cache_hit = bool(use_cache and cache_key in _STRUCTURED_FBLOCK_CSR_CACHE)
    if cache_hit:
        matrix, cached_metadata = _STRUCTURED_FBLOCK_CSR_CACHE[cache_key]
        metadata = dict(cached_metadata)
    else:
        matrix = operator.to_scipy_csr_matrix()
        metadata = {
            "selected": True,
            "reason": "complete",
            "storage_kind": "csr",
            "shape": tuple(int(v) for v in matrix.shape),
            "nnz": int(matrix.nnz),
            "block_nnz": int(operator.nnz_blocks),
            "block_size": int(operator.block_size),
            "csr_nbytes_estimate": int(csr_nbytes_estimate),
            "csr_nbytes_actual": int(_scipy_csr_nbytes(matrix)),
            "max_csr_nbytes": None if max_csr_nbytes is None else int(max_csr_nbytes),
            "drop_tol": float(drop_tol),
            "operator": operator.to_dict(),
            "assembly": selection.assembly.to_dict(),
        }
        if bool(use_cache):
            _STRUCTURED_FBLOCK_CSR_CACHE[cache_key] = (matrix, dict(metadata))
    metadata["cache_hit"] = bool(cache_hit)
    metadata["object_cache_hit"] = False
    if bool(use_cache):
        _STRUCTURED_FBLOCK_CSR_OBJECT_CACHE[object_cache_key] = (selection, matrix, dict(metadata))
    return RHS1StructuredFBlockCSRSelection(
        selection=selection,
        matrix=matrix,
        selected=True,
        reason="complete",
        cache_hit=bool(cache_hit),
        build_s=max(0.0, time.perf_counter() - t0),
        metadata=metadata,
    )


def assemble_partial_rhs1_fblock_operator(
    op: Any,
    *,
    layout: RHS1BlockLayout | None = None,
    include_identity_shift: bool = True,
    phi1_hat_base: Any | None = None,
    drop_tol: float = 0.0,
    strict_complete: bool = False,
) -> RHS1PartialFBlockAssembly:
    """Assemble migrated ``V3FBlockOperator`` terms into one block-COO operator.

    The assembled set currently covers identity, collisionless streaming/mirror,
    pitch-angle scattering, full-FP collisions, ExB theta/zeta, electric-field
    ``x``/``xi`` drift terms, and magnetic-drift theta/zeta/``xi-dot`` terms.
    Phi1-in-collision terms are assembled only when a frozen
    ``phi1_hat_base`` is supplied; otherwise they are reported in
    ``unsupported_terms`` because the term is state dependent.
    """

    block_layout = rhs1_fblock_layout_from_operator(op) if layout is None else layout
    parts: list[tuple[str, RHS1BlockCOOOperator]] = []
    unsupported: list[str] = []

    shift = float(np.asarray(op.identity_shift, dtype=np.float64).reshape(-1)[0])
    if bool(include_identity_shift) and abs(shift) > 0.0:
        parts.append(("identity_shift", _build_identity_shift_operator(layout=block_layout, shift=shift)))

    parts.append(
        (
            "collisionless",
            build_collisionless_f_block_operator(
                layout=block_layout,
                collisionless_operator=op.collisionless,
                drop_tol=drop_tol,
            ),
        )
    )
    if op.pas is not None:
        parts.append(
            (
                "pas",
                build_pas_collision_f_block_operator(layout=block_layout, pas_operator=op.pas, drop_tol=drop_tol),
            )
        )
    if op.fp is not None:
        parts.append(
            (
                "fp",
                build_fokker_planck_collision_f_block_operator(
                    layout=block_layout,
                    fp_operator=op.fp,
                    drop_tol=drop_tol,
                ),
            )
        )
    if op.fp_phi1 is not None:
        if phi1_hat_base is None:
            unsupported.append("fp_phi1")
        else:
            parts.append(
                (
                    "fp_phi1",
                    build_fokker_planck_phi1_collision_f_block_operator(
                        layout=block_layout,
                        fp_phi1_operator=op.fp_phi1,
                        phi1_hat_base=phi1_hat_base,
                        drop_tol=drop_tol,
                    ),
                )
            )
    if op.exb_theta is not None:
        parts.append(
            (
                "exb_theta",
                build_exb_theta_f_block_operator(
                    layout=block_layout,
                    exb_theta_operator=op.exb_theta,
                    drop_tol=drop_tol,
                ),
            )
        )
    if op.exb_zeta is not None:
        parts.append(
            (
                "exb_zeta",
                build_exb_zeta_f_block_operator(
                    layout=block_layout,
                    exb_zeta_operator=op.exb_zeta,
                    drop_tol=drop_tol,
                ),
            )
        )
    if op.er_xidot is not None:
        parts.append(
            (
                "er_xidot",
                build_er_xidot_f_block_operator(
                    layout=block_layout,
                    er_xidot_operator=op.er_xidot,
                    drop_tol=drop_tol,
                ),
            )
        )
    if op.er_xdot is not None:
        parts.append(
            (
                "er_xdot",
                build_er_xdot_f_block_operator(
                    layout=block_layout,
                    er_xdot_operator=op.er_xdot,
                    drop_tol=drop_tol,
                ),
            )
        )
    if op.magdrift_xidot is not None:
        parts.append(
            (
                "magdrift_xidot",
                build_magnetic_drift_xidot_f_block_operator(
                    layout=block_layout,
                    magdrift_xidot_operator=op.magdrift_xidot,
                    drop_tol=drop_tol,
                ),
            )
        )
    if op.magdrift_theta is not None:
        parts.append(
            (
                "magdrift_theta",
                build_magnetic_drift_theta_f_block_operator(
                    layout=block_layout,
                    magdrift_theta_operator=op.magdrift_theta,
                    drop_tol=drop_tol,
                ),
            )
        )
    if op.magdrift_zeta is not None:
        parts.append(
            (
                "magdrift_zeta",
                build_magnetic_drift_zeta_f_block_operator(
                    layout=block_layout,
                    magdrift_zeta_operator=op.magdrift_zeta,
                    drop_tol=drop_tol,
                ),
            )
        )

    if strict_complete and unsupported:
        raise NotImplementedError(f"RHSMode=1 f-block block-COO assembly does not yet cover: {', '.join(unsupported)}")

    term_nnz_blocks = {name: int(part.nnz_blocks) for name, part in parts}
    term_data_nbytes = {name: int(part.data_nbytes) for name, part in parts}
    return RHS1PartialFBlockAssembly(
        layout=block_layout,
        operator=_sum_block_coo_parts([part for _, part in parts], layout=block_layout),
        included_terms=tuple(name for name, _ in parts),
        unsupported_terms=tuple(unsupported),
        term_nnz_blocks=term_nnz_blocks,
        term_data_nbytes=term_data_nbytes,
    )


def _estimate_block_coo_as_csr_nbytes(operator: RHS1BlockCOOOperator) -> int:
    scalar_nnz_upper = int(operator.nnz_blocks) * int(operator.block_size) * int(operator.block_size)
    dtype = np.dtype(np.asarray(operator.blocks).dtype)
    data_nbytes = scalar_nnz_upper * int(dtype.itemsize)
    index_nbytes = scalar_nnz_upper * np.dtype(np.int32).itemsize
    indptr_nbytes = (int(operator.shape[0]) + 1) * np.dtype(np.int32).itemsize
    return int(data_nbytes + index_nbytes + indptr_nbytes)


def _scipy_csr_nbytes(matrix: Any) -> int:
    return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)


def _structured_fblock_csr_cache_key(
    *,
    selection: RHS1StructuredFBlockSelection,
    drop_tol: float,
) -> tuple[object, ...]:
    operator = selection.assembly.operator
    digest = hashlib.blake2b(digest_size=16)
    for array in (operator.row_blocks, operator.col_blocks, operator.blocks):
        arr = np.asarray(array)
        digest.update(str(arr.dtype).encode("ascii"))
        digest.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(arr).view(np.uint8))
    return (
        "structured_rhs1_fblock_csr",
        tuple(int(v) for v in operator.shape),
        int(operator.block_size),
        float(drop_tol),
        digest.hexdigest(),
    )


def _structured_fblock_csr_object_cache_key(
    *,
    op: Any,
    include_identity_shift: bool,
    phi1_hat_base: Any | None,
    drop_tol: float,
    require_complete: bool,
) -> tuple[object, ...]:
    return (
        "structured_rhs1_fblock_csr_object",
        id(op),
        bool(include_identity_shift),
        id(phi1_hat_base) if phi1_hat_base is not None else None,
        float(drop_tol),
        bool(require_complete),
    )


def _build_identity_shift_operator(*, layout: RHS1BlockLayout, shift: float) -> RHS1BlockCOOOperator:
    builder = RHS1BlockCOOBuilder(
        shape=(int(layout.f_size), int(layout.f_size)),
        block_size=int(layout.n_zeta),
        dtype=np.float64,
    )
    eye = float(shift) * np.eye(int(layout.n_zeta), dtype=np.float64)
    n_blocks = int(layout.f_size) // int(layout.n_zeta)
    for block_id in range(n_blocks):
        builder.add_dense_block(block_id, block_id, eye)
    return builder.build()


def _sum_block_coo_parts(parts: list[RHS1BlockCOOOperator], *, layout: RHS1BlockLayout) -> RHS1BlockCOOOperator:
    if not parts:
        return RHS1BlockCOOOperator.from_blocks(
            row_blocks=jnp.zeros((0,), dtype=jnp.int32),
            col_blocks=jnp.zeros((0,), dtype=jnp.int32),
            blocks=jnp.zeros((0, int(layout.n_zeta), int(layout.n_zeta)), dtype=jnp.float64),
            n_block_rows=int(layout.f_size) // int(layout.n_zeta),
            n_block_cols=int(layout.f_size) // int(layout.n_zeta),
        )

    shape = parts[0].shape
    block_size = int(parts[0].block_size)
    for part in parts:
        if part.shape != shape or int(part.block_size) != block_size:
            raise ValueError("all f-block COO parts must have the same shape and block size")
    return RHS1BlockCOOOperator.from_blocks(
        row_blocks=jnp.concatenate([part.row_blocks for part in parts], axis=0),
        col_blocks=jnp.concatenate([part.col_blocks for part in parts], axis=0),
        blocks=jnp.concatenate([part.blocks for part in parts], axis=0),
        n_block_rows=int(parts[0].n_block_rows),
        n_block_cols=int(parts[0].n_block_cols),
    )


__all__ = [
    "RHS1PartialFBlockAssembly",
    "RHS1StructuredFBlockCSRSelection",
    "RHS1StructuredFBlockSelection",
    "assemble_partial_rhs1_fblock_operator",
    "clear_structured_rhs1_fblock_csr_cache",
    "rhs1_fblock_layout_from_operator",
    "select_structured_rhs1_fblock_csr_operator",
    "select_structured_rhs1_fblock_operator",
]
