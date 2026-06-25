"""Memory-bounded kinetic preconditioner for exact RHSMode=1 full CSR matrices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import time

import numpy as np
import scipy.sparse as sp

from sfincs_jax.operators.profile_response.layout import RHS1BlockLayout

_FLOAT64_NBYTES = int(np.dtype(np.float64).itemsize)
_INT64_NBYTES = int(np.dtype(np.int64).itemsize)
_DEFAULT_MAX_CANDIDATE_NBYTES = 64 * 1024 * 1024
_DEFAULT_MAX_BLOCK_SIZE = 512


@dataclass(frozen=True)
class RHS1FullCSRKineticPreconditioner:
    """Host-side block inverse candidate for a full RHSMode=1 CSR operator."""

    operator: Any | None
    native_factor: Any | None
    selected: bool
    kind: str
    reason: str
    setup_s: float
    metadata: dict[str, object]

    def apply(self, rhs: Any) -> np.ndarray:
        """Apply the preconditioner to a residual-like vector."""

        if self.operator is None:
            raise RuntimeError(f"RHS1 full CSR kinetic preconditioner was not selected: {self.reason}")
        return np.asarray(self.operator.matvec(np.asarray(rhs, dtype=np.float64).reshape((-1,))))

    def apply_native(self, rhs: Any) -> np.ndarray:
        """Apply the optional JAX-native factor and return a NumPy array."""

        if self.native_factor is None:
            raise RuntimeError(f"RHS1 full CSR native kinetic factor is unavailable: {self.reason}")
        from sfincs_jax.native_block_factor import apply_native_x_ell_kinetic_factor  # noqa: PLC0415

        return np.array(
            apply_native_x_ell_kinetic_factor(self.native_factor, np.asarray(rhs, dtype=np.float64)),
            dtype=np.float64,
            copy=True,
        )

    def to_dict(self) -> dict[str, object]:
        """Return JSON-friendly selection and storage metadata."""

        return {
            "selected": bool(self.selected),
            "native_factor_available": self.native_factor is not None,
            "kind": str(self.kind),
            "reason": str(self.reason),
            "setup_s": float(self.setup_s),
            "metadata": dict(self.metadata),
        }


def build_rhs1_full_csr_kinetic_preconditioner(
    *,
    matrix: Any,
    layout: RHS1BlockLayout,
    kind: str | None = "x_ell",
    max_candidate_nbytes: int | None = _DEFAULT_MAX_CANDIDATE_NBYTES,
    max_block_size: int = _DEFAULT_MAX_BLOCK_SIZE,
    regularization: float = 1.0e-12,
    tail_policy: str = "jacobi",
    build_native_factor: bool = False,
) -> RHS1FullCSRKineticPreconditioner:
    """Build a bounded f-block line preconditioner for a full host CSR matrix.

    The default ``x_ell`` candidate groups all ``(x, ell)`` unknowns at fixed
    ``(species, theta, zeta)``. This captures local kinetic coupling from the
    exact f-block while avoiding dense storage for theta/zeta/global couplings.
    The non-kinetic tail is treated with scalar Jacobi by default.
    """

    from scipy.sparse.linalg import LinearOperator  # noqa: PLC0415

    t0 = time.perf_counter()
    kind_l = _normalize_kind(kind)
    tail_policy_l = _normalize_tail_policy(tail_policy)
    if kind_l == "none":
        return _fail(kind=kind_l, reason="disabled", t0=t0, metadata={})
    if kind_l != "x_ell":
        return _fail(kind=kind_l, reason="unsupported_preconditioner", t0=t0, metadata={})
    if tail_policy_l not in {"jacobi", "identity"}:
        return _fail(
            kind=kind_l,
            reason="unsupported_tail_policy",
            t0=t0,
            metadata={"tail_policy": str(tail_policy)},
        )

    validation_reason, validation_metadata = _validate_inputs(matrix=matrix, layout=layout)
    if validation_reason is not None:
        return _fail(kind=kind_l, reason=validation_reason, t0=t0, metadata=validation_metadata)
    csr = matrix.tocsr()
    if csr.nnz and not np.all(np.isfinite(csr.data)):
        return _fail(
            kind=kind_l,
            reason="matrix_data_nonfinite",
            t0=t0,
            metadata={"matrix_shape": tuple(int(v) for v in csr.shape), "nnz": int(csr.nnz)},
        )

    n_blocks, block_size = _x_ell_block_shape(layout)
    if int(max_block_size) >= 0 and block_size > int(max_block_size):
        return _fail(
            kind=kind_l,
            reason=f"kinetic_pc_block_size_exceeded:{block_size}>{int(max_block_size)}",
            t0=t0,
            metadata={
                "block_size": int(block_size),
                "max_block_size": int(max_block_size),
                "n_blocks": int(n_blocks),
                "layout": layout.to_dict(),
            },
        )

    estimate = estimate_rhs1_full_csr_kinetic_preconditioner_nbytes(
        layout,
        kind=kind_l,
        tail_policy=tail_policy_l,
    )
    max_nbytes = None if max_candidate_nbytes is None else max(0, int(max_candidate_nbytes))
    if max_nbytes is not None and estimate > max_nbytes:
        return _fail(
            kind=kind_l,
            reason=f"kinetic_pc_budget_exceeded:{estimate}>{max_nbytes}",
            t0=t0,
            metadata={
                "candidate_nbytes_estimate": int(estimate),
                "max_candidate_nbytes": int(max_nbytes),
                "block_size": int(block_size),
                "n_blocks": int(n_blocks),
                "tail_policy": tail_policy_l,
                "layout": layout.to_dict(),
            },
        )

    block_indices = rhs1_full_csr_x_ell_block_indices(layout)
    inverse_blocks, block_metadata = _build_inverse_blocks(
        matrix=csr,
        block_indices=block_indices,
        regularization=float(regularization),
    )
    if int(block_metadata["block_inverse_nonfinite_count"]) > 0:
        return _fail(
            kind=kind_l,
            reason="block_inverse_nonfinite",
            t0=t0,
            metadata={
                "candidate_nbytes_estimate": int(estimate),
                "max_candidate_nbytes": max_nbytes,
                **block_metadata,
            },
        )

    tail_size = int(layout.total_size) - int(layout.f_size)
    inv_tail = np.empty((0,), dtype=np.float64)
    tail_metadata: dict[str, object] = {"tail_policy": tail_policy_l, "tail_size": int(tail_size)}
    if tail_size > 0 and tail_policy_l == "jacobi":
        inv_tail, tail_metadata_extra = _safe_inverse_diagonal(
            csr.diagonal()[int(layout.f_size) :],
            regularization=float(regularization),
        )
        tail_metadata.update({f"tail_{key}": value for key, value in tail_metadata_extra.items()})

    actual_nbytes = int(inverse_blocks.nbytes + block_indices.nbytes + inv_tail.nbytes)
    if max_nbytes is not None and actual_nbytes > max_nbytes:
        return _fail(
            kind=kind_l,
            reason=f"kinetic_pc_budget_exceeded_actual:{actual_nbytes}>{max_nbytes}",
            t0=t0,
            metadata={
                "candidate_nbytes_estimate": int(estimate),
                "candidate_nbytes_actual": int(actual_nbytes),
                "max_candidate_nbytes": int(max_nbytes),
                **block_metadata,
            },
        )

    n_f = int(layout.f_size)

    def apply(rhs: Any) -> np.ndarray:
        rhs_vec = np.asarray(rhs, dtype=np.float64).reshape((-1,))
        if rhs_vec.shape != (int(layout.total_size),):
            raise ValueError(f"rhs must have shape {(int(layout.total_size),)}, got {rhs_vec.shape}")
        f_out = np.empty((n_f,), dtype=np.float64)
        gathered = rhs_vec[:n_f][block_indices]
        solved = np.einsum("bij,bj->bi", inverse_blocks, gathered, optimize=True)
        f_out[block_indices] = solved
        if tail_size <= 0:
            return f_out
        tail_rhs = rhs_vec[n_f:]
        if tail_policy_l == "jacobi":
            tail_out = inv_tail * tail_rhs
        else:
            tail_out = tail_rhs.copy()
        return np.concatenate((f_out, tail_out))

    operator = LinearOperator(csr.shape, matvec=apply, dtype=np.float64)
    native_factor = None
    native_metadata: dict[str, object] = {"native_factor_available": False}
    if bool(build_native_factor):
        from sfincs_jax.native_block_factor import build_native_x_ell_kinetic_factor  # noqa: PLC0415

        native_factor = build_native_x_ell_kinetic_factor(
            block_inverses=inverse_blocks,
            block_indices=block_indices,
            inv_tail=inv_tail,
            f_size=int(layout.f_size),
            total_size=int(layout.total_size),
        )
        native_metadata = {
            "native_factor_available": True,
            "native_factor_storage_kind": "jax_dense_x_ell_inverse_blocks",
            "native_factor_block_inverse_nbytes": int(inverse_blocks.nbytes),
            "native_factor_block_index_nbytes": int(block_indices.astype(np.int32, copy=False).nbytes),
            "native_factor_tail_inverse_nbytes": int(inv_tail.nbytes),
        }
    metadata: dict[str, object] = {
        "selected": True,
        "reason": "complete",
        "storage_kind": "dense_x_ell_inverse_blocks",
        "matrix_shape": tuple(int(v) for v in csr.shape),
        "matrix_nnz": int(csr.nnz),
        "kinetic_size": int(layout.f_size),
        "tail_size": int(tail_size),
        "line_axes": ("x", "ell"),
        "fixed_axes": ("species", "theta", "zeta"),
        "n_blocks": int(n_blocks),
        "block_size": int(block_size),
        "candidate_nbytes_estimate": int(estimate),
        "candidate_nbytes_actual": int(actual_nbytes),
        "max_candidate_nbytes": max_nbytes,
        "max_block_size": int(max_block_size),
        "regularization": float(regularization),
        "block_inverse_nbytes_actual": int(inverse_blocks.nbytes),
        "block_index_nbytes_actual": int(block_indices.nbytes),
        "tail_inverse_nbytes_actual": int(inv_tail.nbytes),
        "layout": layout.to_dict(),
        **block_metadata,
        **tail_metadata,
        **native_metadata,
    }
    return RHS1FullCSRKineticPreconditioner(
        operator=operator,
        native_factor=native_factor,
        selected=True,
        kind=kind_l,
        reason="complete",
        setup_s=max(0.0, time.perf_counter() - t0),
        metadata=metadata,
    )


def estimate_rhs1_full_csr_kinetic_preconditioner_nbytes(
    layout: RHS1BlockLayout,
    *,
    kind: str | None = "x_ell",
    tail_policy: str = "jacobi",
) -> int:
    """Estimate candidate storage bytes before extracting dense line blocks."""

    kind_l = _normalize_kind(kind)
    if kind_l != "x_ell":
        return 0
    n_blocks, block_size = _x_ell_block_shape(layout)
    tail_size = int(layout.total_size) - int(layout.f_size)
    inverse_nbytes = int(n_blocks * block_size * block_size * _FLOAT64_NBYTES)
    index_nbytes = int(n_blocks * block_size * _INT64_NBYTES)
    tail_nbytes = int(max(0, tail_size) * _FLOAT64_NBYTES) if _normalize_tail_policy(tail_policy) == "jacobi" else 0
    return int(inverse_nbytes + index_nbytes + tail_nbytes)


def rhs1_full_csr_x_ell_block_indices(layout: RHS1BlockLayout) -> np.ndarray:
    """Return f-block indices grouped by fixed ``(species, theta, zeta)``."""

    rows: list[np.ndarray] = []
    for species in range(int(layout.n_species)):
        for theta in range(int(layout.n_theta)):
            for zeta in range(int(layout.n_zeta)):
                rows.append(
                    np.asarray(
                        [
                            layout.kinetic_flat_index(
                                species=species,
                                x=x_index,
                                ell=ell,
                                theta=theta,
                                zeta=zeta,
                            )
                            for x_index in range(int(layout.n_x))
                            for ell in range(int(layout.n_xi))
                        ],
                        dtype=np.int64,
                    )
                )
    return np.asarray(rows, dtype=np.int64)


def _build_inverse_blocks(
    *,
    matrix: sp.csr_matrix,
    block_indices: np.ndarray,
    regularization: float,
) -> tuple[np.ndarray, dict[str, object]]:
    n_blocks, block_size = (int(block_indices.shape[0]), int(block_indices.shape[1]))
    inverse_blocks = np.empty((n_blocks, block_size, block_size), dtype=np.float64)
    regularized_count = 0
    singular_count = 0
    nonfinite_count = 0
    condition_nonfinite_count = 0
    max_block_scale = 0.0
    max_condition_estimate: float | None = 0.0 if block_size <= 64 else None
    for block_id, indices in enumerate(block_indices):
        dense = np.asarray(matrix[indices[:, None], indices].toarray(), dtype=np.float64)
        if not np.all(np.isfinite(dense)):
            nonfinite_count += 1
            inverse_blocks[block_id] = np.nan
            continue
        block_scale = max(float(np.linalg.norm(dense, ord=np.inf)) if dense.size else 0.0, 1.0)
        max_block_scale = max(max_block_scale, block_scale)
        regularization_abs = float(abs(regularization)) * block_scale
        if regularization_abs > 0.0:
            dense = dense + regularization_abs * np.eye(block_size, dtype=np.float64)
            regularized_count += 1
        if max_condition_estimate is not None:
            condition_estimate = float(np.linalg.cond(dense))
            if np.isfinite(condition_estimate):
                max_condition_estimate = max(float(max_condition_estimate), condition_estimate)
            else:
                condition_nonfinite_count += 1
        try:
            inverse = np.linalg.inv(dense)
        except np.linalg.LinAlgError:
            singular_count += 1
            inverse = np.linalg.pinv(dense, rcond=max(float(abs(regularization)), 1.0e-14))
        if not np.all(np.isfinite(inverse)):
            nonfinite_count += 1
        inverse_blocks[block_id] = inverse
    metadata = {
        "block_inverse_regularized_count": int(regularized_count),
        "block_inverse_singular_count": int(singular_count),
        "block_inverse_nonfinite_count": int(nonfinite_count),
        "block_inverse_scale_max": float(max_block_scale),
        "block_inverse_condition_estimate_max": max_condition_estimate,
        "block_inverse_condition_nonfinite_count": int(condition_nonfinite_count),
    }
    return inverse_blocks, metadata


def _safe_inverse_diagonal(diagonal: Any, *, regularization: float) -> tuple[np.ndarray, dict[str, object]]:
    diag = np.asarray(diagonal, dtype=np.float64).reshape((-1,))
    abs_diag = np.abs(diag)
    scale = max(float(np.max(abs_diag)) if abs_diag.size else 0.0, 1.0)
    floor = float(abs(regularization)) * scale
    if floor == 0.0:
        floor = np.finfo(np.float64).tiny
    safe = diag.copy()
    small = abs_diag <= floor
    signs = np.where(safe < 0.0, -1.0, 1.0)
    safe[small] = signs[small] * floor
    inv = 1.0 / safe
    metadata = {
        "diagonal_size": int(diag.size),
        "diagonal_abs_max": float(np.max(abs_diag)) if abs_diag.size else 0.0,
        "diagonal_abs_min": float(np.min(abs_diag)) if abs_diag.size else 0.0,
        "diagonal_floor": float(floor),
        "diagonal_regularized_count": int(np.count_nonzero(small)),
    }
    return inv, metadata


def _validate_inputs(*, matrix: Any, layout: RHS1BlockLayout) -> tuple[str | None, dict[str, object]]:
    if int(layout.rhs_mode) != 1:
        return f"unsupported_rhs_mode:{int(layout.rhs_mode)}", {"layout": layout.to_dict()}
    expected_f_size = int(layout.n_species * layout.n_x * layout.n_xi * layout.n_theta * layout.n_zeta)
    if int(layout.f_size) != expected_f_size:
        return (
            "layout_f_size_mismatch",
            {"layout_f_size": int(layout.f_size), "expected_f_size": int(expected_f_size)},
        )
    if int(layout.total_size) < int(layout.f_size):
        return (
            "layout_total_size_mismatch",
            {"layout_total_size": int(layout.total_size), "layout_f_size": int(layout.f_size)},
        )
    if not sp.issparse(matrix):
        return "matrix_not_sparse", {"matrix_type": type(matrix).__name__}
    if int(matrix.shape[0]) != int(matrix.shape[1]):
        return "matrix_not_square", {"shape": tuple(int(v) for v in matrix.shape)}
    if tuple(int(v) for v in matrix.shape) != (int(layout.total_size), int(layout.total_size)):
        return (
            "layout_size_mismatch",
            {"matrix_shape": tuple(int(v) for v in matrix.shape), "layout_total_size": int(layout.total_size)},
        )
    if int(layout.f_size) <= 0:
        return "empty_kinetic_layout", {"layout": layout.to_dict()}
    return None, {}


def _x_ell_block_shape(layout: RHS1BlockLayout) -> tuple[int, int]:
    block_size = int(layout.n_x) * int(layout.n_xi)
    n_blocks = int(layout.n_species) * int(layout.n_theta) * int(layout.n_zeta)
    return int(n_blocks), int(block_size)


def _normalize_kind(kind: str | None) -> str:
    kind_l = "x_ell" if kind is None else str(kind).strip().lower().replace("-", "_")
    if kind_l in {"", "auto", "xell", "x_ell_line", "x_ell_block", "kinetic_x_ell"}:
        return "x_ell"
    if kind_l in {"none", "false", "off", "disabled"}:
        return "none"
    return kind_l


def _normalize_tail_policy(tail_policy: str) -> str:
    tail_l = str(tail_policy).strip().lower().replace("-", "_")
    if tail_l in {"", "diag", "diagonal"}:
        return "jacobi"
    if tail_l in {"none", "off"}:
        return "identity"
    return tail_l


def _fail(
    *,
    kind: str,
    reason: str,
    t0: float,
    metadata: dict[str, object],
) -> RHS1FullCSRKineticPreconditioner:
    enriched = {"selected": False, "reason": str(reason), **metadata}
    return RHS1FullCSRKineticPreconditioner(
        operator=None,
        native_factor=None,
        selected=False,
        kind=str(kind),
        reason=str(reason),
        setup_s=max(0.0, time.perf_counter() - t0),
        metadata=enriched,
    )


__all__ = [
    "RHS1FullCSRKineticPreconditioner",
    "build_rhs1_full_csr_kinetic_preconditioner",
    "estimate_rhs1_full_csr_kinetic_preconditioner_nbytes",
    "rhs1_full_csr_x_ell_block_indices",
]
