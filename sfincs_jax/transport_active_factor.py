"""Reusable active-operator block factors for RHSMode=2/3 transport.

The helpers in this module deliberately separate three stages that are also
separate in sparse direct packages:

1. symbolic block ordering,
2. numerical block/Schur factor construction,
3. setup-time residual admission against the true active operator.

The module does not depend on PETSc/MUMPS/SuperLU_DIST.  It provides small,
Python-native building blocks that can be reused by the SFINCS-JAX transport
driver while keeping production defaults guarded by strict residual probes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ActiveBlockOrdering:
    """Symbolic block layout for an active transport operator.

    The active operator is assumed to use the SFINCS-JAX active ordering:
    kinetic unknowns first, then the source/constraint tail.  ``blocks`` stores
    kinetic reduced indices only; tail indices are handled by the Schur layer.
    """

    blocks: tuple[np.ndarray, ...]
    block_kind: str
    kinetic_size: int
    tail_size: int
    active_size: int
    block_size_max: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ActiveBlockSchurFactor:
    """Numerical block inverse plus optional exact tail Schur closure."""

    ordering: ActiveBlockOrdering
    block_inverse: tuple[np.ndarray, ...]
    c_tail: Any | None
    mb_tail: np.ndarray | None
    schur_inverse: np.ndarray | None
    dtype: np.dtype
    metadata: dict[str, Any]

    def apply(self, rhs: np.ndarray) -> np.ndarray:
        """Apply the block-Schur inverse approximation to one reduced RHS."""

        dtype = np.dtype(self.dtype)
        ordering = self.ordering
        active_size = int(ordering.active_size)
        kinetic_size = int(ordering.kinetic_size)
        tail_size = int(ordering.tail_size)
        rhs_np = np.asarray(rhs, dtype=dtype).reshape((active_size,))
        y_k = np.zeros((kinetic_size,), dtype=dtype)
        for indices, inverse in zip(ordering.blocks, self.block_inverse, strict=True):
            idx = np.asarray(indices, dtype=np.int64)
            y_k[idx] = np.asarray(inverse @ rhs_np[idx], dtype=dtype)
        if tail_size <= 0 or self.c_tail is None or self.mb_tail is None or self.schur_inverse is None:
            return np.concatenate([y_k, rhs_np[kinetic_size:]], axis=0).astype(np.float64, copy=False)
        rhs_t = rhs_np[kinetic_size:]
        tail_residual = np.asarray(rhs_t - self.c_tail @ y_k, dtype=dtype).reshape((tail_size,))
        y_t = np.asarray(self.schur_inverse @ tail_residual, dtype=dtype).reshape((tail_size,))
        y_k = np.asarray(y_k - self.mb_tail @ y_t, dtype=dtype).reshape((kinetic_size,))
        out = np.concatenate([y_k, y_t], axis=0)
        finite = np.isfinite(out)
        if not np.all(finite):
            out = np.where(finite, out, 0.0)
        return np.asarray(out, dtype=np.float64)


@dataclass(frozen=True)
class ActiveBlockAdmission:
    """Result of setup-time residual admission."""

    accepted: bool
    max_relative_residual: float
    median_relative_residual: float
    min_improvement_vs_identity: float
    probe_count: int
    reason: str


def build_active_block_ordering(
    *,
    kinetic_size: int,
    tail_size: int,
    n_theta: int,
    n_zeta: int,
    block_kind: str = "zeta_line",
    ell_block: int = 1,
    max_block_size: int = 4096,
) -> ActiveBlockOrdering:
    """Build a reusable symbolic ordering over active kinetic unknowns.

    Parameters
    ----------
    kinetic_size:
        Number of retained kinetic unknowns in the active reduced system.
    tail_size:
        Number of retained source/constraint unknowns.
    n_theta, n_zeta:
        Angular grid shape in SFINCS storage order ``(..., theta, zeta)``.
    block_kind:
        ``"zeta_line"`` keeps contiguous zeta lines, ``"theta_line"`` keeps
        one theta line at fixed zeta inside each angular plane, and
        ``"angular_plane"``/``"ell_band"`` keeps one or more complete
        ``(theta,zeta)`` planes.
    ell_block:
        Number of complete angular planes per block for ``"ell_band"``.
    max_block_size:
        Hard memory/speed safety cap.  Oversized symbolic blocks are rejected.
    """

    kinetic_size = int(kinetic_size)
    tail_size = int(tail_size)
    n_theta = int(n_theta)
    n_zeta = int(n_zeta)
    block_kind = str(block_kind).strip().lower()
    max_block_size = max(1, int(max_block_size))
    ell_block = max(1, int(ell_block))
    if kinetic_size <= 0:
        raise ValueError("kinetic_size must be positive")
    if n_theta <= 0 or n_zeta <= 0:
        raise ValueError("n_theta and n_zeta must be positive")
    plane_size = int(n_theta * n_zeta)
    blocks: list[np.ndarray] = []
    if block_kind in {"zeta", "zeta_line", "zeta-line"}:
        if kinetic_size % n_zeta != 0:
            raise ValueError("zeta-line ordering requires kinetic_size divisible by n_zeta")
        for start in range(0, kinetic_size, n_zeta):
            blocks.append(np.arange(start, start + n_zeta, dtype=np.int64))
        canonical = "zeta_line"
    elif block_kind in {"theta", "theta_line", "theta-line"}:
        if kinetic_size % plane_size != 0:
            raise ValueError("theta-line ordering requires complete angular planes")
        for base in range(0, kinetic_size, plane_size):
            for izeta in range(n_zeta):
                blocks.append(base + np.arange(n_theta, dtype=np.int64) * n_zeta + int(izeta))
        canonical = "theta_line"
    elif block_kind in {"plane", "angular_plane", "ell_plane"}:
        if kinetic_size % plane_size != 0:
            raise ValueError("angular-plane ordering requires complete angular planes")
        for start in range(0, kinetic_size, plane_size):
            blocks.append(np.arange(start, start + plane_size, dtype=np.int64))
        canonical = "angular_plane"
    elif block_kind in {"ell_band", "pitch_band", "plane_band"}:
        if kinetic_size % plane_size != 0:
            raise ValueError("ell-band ordering requires complete angular planes")
        block_size = int(ell_block * plane_size)
        for start in range(0, kinetic_size, block_size):
            stop = min(kinetic_size, start + block_size)
            blocks.append(np.arange(start, stop, dtype=np.int64))
        canonical = "ell_band"
    else:
        raise ValueError(f"unsupported active block ordering {block_kind!r}")
    if not blocks:
        raise ValueError("active block ordering produced no blocks")
    block_size_max = max(int(block.size) for block in blocks)
    if block_size_max > max_block_size:
        raise MemoryError(f"active block size {block_size_max} exceeds max_block_size={max_block_size}")
    return ActiveBlockOrdering(
        blocks=tuple(blocks),
        block_kind=canonical,
        kinetic_size=kinetic_size,
        tail_size=tail_size,
        active_size=int(kinetic_size + tail_size),
        block_size_max=int(block_size_max),
        metadata={
            "block_kind": canonical,
            "block_count": int(len(blocks)),
            "block_size_max": int(block_size_max),
            "n_theta": int(n_theta),
            "n_zeta": int(n_zeta),
            "ell_block": int(ell_block),
        },
    )


def _inverse_dense_block(block: np.ndarray, *, reg: float, dtype: np.dtype) -> np.ndarray:
    block_np = np.asarray(block, dtype=dtype)
    if float(reg) > 0.0:
        scale = max(float(np.linalg.norm(block_np, ord=np.inf)), 1.0)
        block_np = block_np + np.asarray(float(reg) * scale, dtype=dtype) * np.eye(block_np.shape[0], dtype=dtype)
    try:
        return np.linalg.inv(block_np)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(block_np, rcond=max(float(abs(reg)), 1.0e-14))


def build_active_block_schur_factor(
    matrix: Any,
    ordering: ActiveBlockOrdering,
    *,
    dtype: np.dtype = np.dtype(np.float64),
    reg: float = 1.0e-12,
    max_mb: float = 2048.0,
) -> ActiveBlockSchurFactor:
    """Build a block inverse and dense tail Schur complement from a CSR matrix."""

    try:
        import scipy.sparse as sp  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - scipy is required by this path
        raise RuntimeError("scipy is required for active block-Schur factors") from exc

    dtype = np.dtype(dtype)
    matrix_csr = matrix.tocsr().astype(dtype, copy=False)
    kinetic_size = int(ordering.kinetic_size)
    tail_size = int(ordering.tail_size)
    active_size = int(ordering.active_size)
    if matrix_csr.shape != (active_size, active_size):
        raise ValueError(f"matrix shape {matrix_csr.shape} does not match active size {active_size}")
    block_inverse: list[np.ndarray] = []
    inverse_nbytes = 0
    for indices in ordering.blocks:
        idx = np.asarray(indices, dtype=np.int64)
        block = np.asarray(matrix_csr[idx[:, None], idx].toarray(), dtype=dtype)
        inverse = _inverse_dense_block(block, reg=float(reg), dtype=dtype)
        inverse_nbytes += int(inverse.nbytes)
        block_inverse.append(np.asarray(inverse, dtype=dtype))
    c_tail = None
    mb_tail = None
    schur_inverse = None
    tail_nbytes = 0
    if tail_size > 0:
        b_tail = np.asarray(matrix_csr[:kinetic_size, kinetic_size:].toarray(), dtype=dtype)
        c_tail = matrix_csr[kinetic_size:, :kinetic_size].tocsr().astype(dtype, copy=False)
        d_tail = np.asarray(matrix_csr[kinetic_size:, kinetic_size:].toarray(), dtype=dtype)
        mb_tail = np.zeros((kinetic_size, tail_size), dtype=dtype)
        for indices, inverse in zip(ordering.blocks, block_inverse, strict=True):
            idx = np.asarray(indices, dtype=np.int64)
            mb_tail[idx, :] = inverse @ b_tail[idx, :]
        schur = np.asarray(d_tail - c_tail @ mb_tail, dtype=dtype)
        schur_inverse = _inverse_dense_block(schur, reg=float(reg), dtype=dtype)
        tail_nbytes = int(b_tail.nbytes + mb_tail.nbytes + schur_inverse.nbytes)
        if not sp.issparse(c_tail):
            c_tail = sp.csr_matrix(c_tail, dtype=dtype)
    matrix_nbytes = int(matrix_csr.data.nbytes + matrix_csr.indices.nbytes + matrix_csr.indptr.nbytes)
    total_nbytes = int(matrix_nbytes + inverse_nbytes + tail_nbytes)
    if float(max_mb) > 0.0 and total_nbytes > int(float(max_mb) * 1.0e6):
        raise MemoryError(f"active block-Schur factor estimate {total_nbytes / 1.0e6:.3f} MB exceeds {max_mb:.3f} MB")
    return ActiveBlockSchurFactor(
        ordering=ordering,
        block_inverse=tuple(block_inverse),
        c_tail=c_tail,
        mb_tail=None if mb_tail is None else np.asarray(mb_tail, dtype=dtype),
        schur_inverse=None if schur_inverse is None else np.asarray(schur_inverse, dtype=dtype),
        dtype=dtype,
        metadata={
            **ordering.metadata,
            "factor_dtype": dtype.name,
            "inverse_nbytes_estimate": int(inverse_nbytes),
            "tail_nbytes_estimate": int(tail_nbytes),
            "matrix_nbytes_estimate": int(matrix_nbytes),
            "total_nbytes_estimate": int(total_nbytes),
            "reg": float(reg),
        },
    )


def deterministic_probe_matrix(
    *,
    active_size: int,
    kinetic_size: int,
    tail_size: int,
    count: int = 4,
) -> np.ndarray:
    """Return deterministic setup probes for preconditioner admission."""

    active_size = int(active_size)
    kinetic_size = int(kinetic_size)
    tail_size = int(tail_size)
    count = max(1, int(count))
    probes: list[np.ndarray] = []
    if kinetic_size > 0:
        probes.append(np.r_[np.ones((kinetic_size,), dtype=np.float64), np.zeros((tail_size,), dtype=np.float64)])
        ramp = np.linspace(-1.0, 1.0, kinetic_size, dtype=np.float64)
        probes.append(np.r_[ramp, np.zeros((tail_size,), dtype=np.float64)])
    for itail in range(min(tail_size, max(0, count - len(probes)))):
        vec = np.zeros((active_size,), dtype=np.float64)
        vec[kinetic_size + int(itail)] = 1.0
        probes.append(vec)
    rng = np.random.default_rng(20260609)
    while len(probes) < count:
        probes.append(rng.normal(size=active_size))
    out = np.column_stack(probes[:count])
    norms = np.linalg.norm(out, axis=0)
    norms = np.where(norms > 0.0, norms, 1.0)
    return out / norms[None, :]


def admit_active_block_schur_factor(
    matrix: Any,
    factor: ActiveBlockSchurFactor,
    probes: np.ndarray | None = None,
    *,
    max_relative_residual: float = 1.0e-2,
    min_improvement_vs_identity: float = 10.0,
) -> ActiveBlockAdmission:
    """Evaluate true residual quality before admitting a block factor."""

    matrix_csr = matrix.tocsr()
    ordering = factor.ordering
    if probes is None:
        probes = deterministic_probe_matrix(
            active_size=int(ordering.active_size),
            kinetic_size=int(ordering.kinetic_size),
            tail_size=int(ordering.tail_size),
            count=4,
        )
    probes_np = np.asarray(probes, dtype=np.float64)
    if probes_np.ndim == 1:
        probes_np = probes_np.reshape((-1, 1))
    rels: list[float] = []
    improvements: list[float] = []
    for icol in range(int(probes_np.shape[1])):
        rhs = probes_np[:, icol]
        rhs_norm = max(float(np.linalg.norm(rhs)), 1.0e-300)
        y = factor.apply(rhs)
        residual = np.asarray(matrix_csr @ y - rhs, dtype=np.float64)
        rel = float(np.linalg.norm(residual) / rhs_norm)
        identity_residual = np.asarray(matrix_csr @ rhs - rhs, dtype=np.float64)
        identity_rel = float(np.linalg.norm(identity_residual) / rhs_norm)
        improvement = identity_rel / max(rel, 1.0e-300)
        rels.append(rel)
        improvements.append(improvement)
    max_rel = float(np.max(rels)) if rels else float("inf")
    med_rel = float(np.median(rels)) if rels else float("inf")
    min_improvement = float(np.min(improvements)) if improvements else 0.0
    accepted = bool(max_rel <= float(max_relative_residual) and min_improvement >= float(min_improvement_vs_identity))
    if accepted:
        reason = "accepted"
    elif max_rel > float(max_relative_residual):
        reason = "relative_residual_gate"
    else:
        reason = "improvement_gate"
    return ActiveBlockAdmission(
        accepted=accepted,
        max_relative_residual=max_rel,
        median_relative_residual=med_rel,
        min_improvement_vs_identity=min_improvement,
        probe_count=int(probes_np.shape[1]),
        reason=reason,
    )


__all__ = [
    "ActiveBlockAdmission",
    "ActiveBlockOrdering",
    "ActiveBlockSchurFactor",
    "admit_active_block_schur_factor",
    "build_active_block_ordering",
    "build_active_block_schur_factor",
    "deterministic_probe_matrix",
]
