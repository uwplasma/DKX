"""Shared preconditioning state, setup, and operator-shaping helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
import hashlib
import os

import jax
import jax.numpy as jnp
import numpy as np

from . import path_policy as solver_path_policy
from sfincs_jax.operators.profile_response.system import V3FullSystemOperator, _THRESHOLD_FOR_INCLUSION


# Cache containers and registries.
@dataclass(frozen=True)
class _RHSMode1PrecondCache:
    idx_map_jnp: jnp.ndarray
    flat_idx_jnp: jnp.ndarray
    block_inv_jnp: jnp.ndarray
    extra_idx_jnp: jnp.ndarray
    extra_inv_jnp: jnp.ndarray | None


_RHSMODE1_PRECOND_CACHE: dict[tuple[object, ...], _RHSMode1PrecondCache] = {}


@dataclass(frozen=True)
class _RHSMode1PrecondListCache:
    block_inv_list: tuple[jnp.ndarray, ...]
    block_slices: tuple[tuple[int, int], ...]
    extra_idx_jnp: jnp.ndarray
    extra_inv_jnp: jnp.ndarray | None


_RHSMODE1_PRECOND_LIST_CACHE: dict[tuple[object, ...], _RHSMode1PrecondListCache] = {}


@dataclass(frozen=True)
class _RHSMode1SchwarzPrecondCache:
    inv_padded_jnp: jnp.ndarray
    patch_idx_padded_jnp: jnp.ndarray
    core_idx_padded_jnp: jnp.ndarray
    core_local_padded_jnp: jnp.ndarray
    core_mask_padded_jnp: jnp.ndarray
    extra_idx_jnp: jnp.ndarray
    extra_inv_jnp: jnp.ndarray | None


_RHSMODE1_SCHWARZ_PRECOND_CACHE: dict[tuple[object, ...], _RHSMode1SchwarzPrecondCache] = {}


@dataclass(frozen=True)
class _RHSMode1PrecondGlobalCache:
    idx_map_jnp: jnp.ndarray
    flat_idx_jnp: jnp.ndarray
    block_inv_jnp: jnp.ndarray
    extra_idx_jnp: jnp.ndarray
    extra_inv_jnp: jnp.ndarray | None


_RHSMODE1_PRECOND_GLOBAL_CACHE: dict[tuple[object, ...], _RHSMode1PrecondGlobalCache] = {}


@dataclass(frozen=True)
class _RHSMode1PrecondDiagXCache:
    block_inv_list: tuple[tuple[jnp.ndarray, ...], ...]
    idx_map_list: tuple[tuple[jnp.ndarray, ...], ...]
    extra_idx_jnp: jnp.ndarray
    extra_inv_jnp: jnp.ndarray | None


_RHSMODE1_PRECOND_DIAGX_CACHE: dict[tuple[object, ...], _RHSMode1PrecondDiagXCache] = {}


@dataclass(frozen=True)
class _RHSMode1PrecondIdxCache:
    block_inv_list: tuple[jnp.ndarray, ...]
    block_idx_list: tuple[jnp.ndarray, ...]
    extra_idx_jnp: jnp.ndarray
    extra_inv_jnp: jnp.ndarray | None


_RHSMODE1_PRECOND_IDX_CACHE: dict[tuple[object, ...], _RHSMode1PrecondIdxCache] = {}
_RHSMODE1_PAS_PRECOND_PROBE_CACHE: dict[tuple[object, ...], bool] = {}


@dataclass(frozen=True)
class _RHSMode1ILUBlockPrecondCache:
    """Sparse ILU block-Jacobi preconditioner cache for PAS-like operators."""

    inv_perm_r_sx: jnp.ndarray
    perm_c_sx: jnp.ndarray
    lower_idx_sx: jnp.ndarray
    lower_val_sx: jnp.ndarray
    upper_idx_sx: jnp.ndarray
    upper_val_sx: jnp.ndarray
    upper_diag_sx: jnp.ndarray
    extra_idx_jnp: jnp.ndarray
    extra_inv_jnp: jnp.ndarray | None


_RHSMODE1_PRECOND_ILU_CACHE: dict[tuple[object, ...], _RHSMode1ILUBlockPrecondCache] = {}


@dataclass(frozen=True)
class _RHSMode1StructuredFBlockPrecondCache:
    """Cached structured f-block factor/coarse data for same-shape solves."""

    operator: object
    metadata: dict[str, object]
    factor: object | None = None
    coarse: object | None = None
    base_preconditioner: Callable[[jnp.ndarray], jnp.ndarray] | None = None


_RHSMODE1_STRUCTURED_FBLOCK_PRECOND_CACHE: dict[tuple[object, ...], _RHSMode1StructuredFBlockPrecondCache] = {}


@dataclass(frozen=True)
class _RHSMode1SparseXBlockPrecondCache:
    """Sparse per-(species,x) block-Jacobi preconditioner cache for FP-like RHSMode=1 operators."""

    perm_r_sx: jnp.ndarray
    inv_perm_c_sx: jnp.ndarray
    lower_idx_sx: jnp.ndarray
    lower_val_sx: jnp.ndarray
    upper_idx_sx: jnp.ndarray
    upper_val_sx: jnp.ndarray
    upper_diag_sx: jnp.ndarray
    extra_idx_jnp: jnp.ndarray
    extra_inv_jnp: jnp.ndarray | None


_RHSMODE1_SPARSE_XBLOCK_PRECOND_CACHE: dict[tuple[object, ...], _RHSMode1SparseXBlockPrecondCache] = {}


@dataclass(frozen=True)
class _RHSMode1SparseXBlockCSRPrecondCache:
    """Compact CSR SuperLU factors for device-side x-block preconditioning."""

    perm_r_sx: jnp.ndarray
    inv_perm_c_sx: jnp.ndarray
    lower_indptr: jnp.ndarray
    lower_indices: jnp.ndarray
    lower_val: jnp.ndarray
    upper_indptr: jnp.ndarray
    upper_indices: jnp.ndarray
    upper_val: jnp.ndarray
    upper_diag_sx: jnp.ndarray
    extra_idx_jnp: jnp.ndarray
    extra_inv_jnp: jnp.ndarray | None
    lower_nnz: int
    upper_nnz: int
    nbytes_estimate: int


_RHSMODE1_SPARSE_XBLOCK_CSR_PRECOND_CACHE: dict[tuple[object, ...], _RHSMode1SparseXBlockCSRPrecondCache] = {}


@dataclass(frozen=True)
class _RHSMode1SparseXBlockHostPrecondCache:
    """Host sparse per-(species,x) block-Jacobi preconditioner cache for explicit solves."""

    block_slices: tuple[tuple[int, int], ...]
    block_factors: tuple[object | None, ...]
    block_diag_inv: tuple[np.ndarray | None, ...]
    extra_idx_np: np.ndarray
    extra_inv_np: np.ndarray | None


_RHSMODE1_SPARSE_XBLOCK_HOST_PRECOND_CACHE: dict[tuple[object, ...], _RHSMode1SparseXBlockHostPrecondCache] = {}


@dataclass(frozen=True)
class _RHSMode1SparseSXBlockHostPrecondCache:
    """Host sparse per-L species/x block preconditioner cache for explicit solves."""

    block_indices: tuple[np.ndarray, ...]
    block_factors: tuple[object | None, ...]
    extra_idx_np: np.ndarray
    extra_inv_np: np.ndarray | None


_RHSMODE1_SPARSE_SXBLOCK_HOST_PRECOND_CACHE: dict[tuple[object, ...], _RHSMode1SparseSXBlockHostPrecondCache] = {}


@dataclass(frozen=True)
class _RHSMode1FPXBlockAssembledHostCache:
    """Cached host-side ingredients for explicit FP x-block assembly."""

    x: np.ndarray
    z_s: np.ndarray
    fp_diag_sxl: np.ndarray
    n_tz_eye: object
    stream_tz_by_species: tuple[object, ...]
    mirror_diag_by_species: tuple[object, ...]
    exb_op_tz: object | None
    mag_theta_m1_tz_by_species: tuple[object, ...] | None
    mag_theta_m2_tz_by_species: tuple[object, ...] | None
    mag_zeta_m1_tz_by_species: tuple[object, ...] | None
    mag_zeta_m2_tz_by_species: tuple[object, ...] | None
    mag_xidot_factor_flat: np.ndarray | None
    er_xidot_factor_flat: np.ndarray | None
    er_xdot_factor_flat: np.ndarray | None
    ddx_plus_diag: np.ndarray | None
    ddx_minus_diag: np.ndarray | None
    identity_shift: float


_RHSMODE1_FP_XBLOCK_ASSEMBLED_HOST_CACHE: dict[tuple[object, ...], _RHSMode1FPXBlockAssembledHostCache] = {}


@dataclass(frozen=True)
class _RHSMode1ThetaLineDiagXCache:
    block_inv: jnp.ndarray
    block_idx: jnp.ndarray
    extra_idx_jnp: jnp.ndarray
    extra_inv_jnp: jnp.ndarray | None


_RHSMODE1_THETA_LINE_DIAGX_CACHE: dict[tuple[object, ...], _RHSMode1ThetaLineDiagXCache] = {}


@dataclass(frozen=True)
class _SparseILUCache:
    a_csr_full: object
    a_csr_drop: object
    ilu: object | None
    a_dense: np.ndarray | None
    l_dense: np.ndarray | None
    u_dense: np.ndarray | None
    l_unit_diag: bool
    perm_r: jnp.ndarray | None = None
    inv_perm_c: jnp.ndarray | None = None
    lower_idx: jnp.ndarray | None = None
    lower_val: jnp.ndarray | None = None
    lower_diag: jnp.ndarray | None = None
    upper_idx: jnp.ndarray | None = None
    upper_val: jnp.ndarray | None = None
    upper_diag: jnp.ndarray | None = None


_RHSMODE1_SPARSE_ILU_CACHE: dict[tuple[object, ...], _SparseILUCache] = {}


@dataclass(frozen=True)
class _TransportPrecondCache:
    inv_diag_f: jnp.ndarray


@dataclass(frozen=True)
class _TransportXBlockPrecondCache:
    inv_xblock: jnp.ndarray


@dataclass(frozen=True)
class _LowRankXBlockPrecondCache:
    d_inv: jnp.ndarray
    d_inv_u: jnp.ndarray
    v: jnp.ndarray
    m_inv: jnp.ndarray


@dataclass(frozen=True)
class _TransportXmgPrecondCache:
    inv_diag_f: jnp.ndarray
    coarse_inv: jnp.ndarray
    coarse_idx: jnp.ndarray
    coarse_inv_lblock: jnp.ndarray | None = None
    lblock: int = 0


@dataclass(frozen=True)
class _XUpwindPrecondCache:
    """Cached factors for a simple x-upwind (bidiagonal) preconditioner."""

    diag: jnp.ndarray
    sub: jnp.ndarray
    lblock: int = 0
    block_inv: jnp.ndarray | None = None
    block_sub: jnp.ndarray | None = None


@dataclass(frozen=True)
class _TransportTzFftPrecondCache:
    subdiag: jnp.ndarray
    diag: jnp.ndarray
    superdiag: jnp.ndarray


@dataclass(frozen=True)
class _TransportFpTzFftPrecondCache:
    inv_mode: jnp.ndarray
    n_block: int


@dataclass(frozen=True)
class _TransportFpTzFftLinePrecondCache:
    """Block-Thomas FP transport factors in Fourier space."""

    inv_eff: jnp.ndarray
    lower_diag: jnp.ndarray
    super_diag: jnp.ndarray
    n_block: int


@dataclass(frozen=True)
class _TransportFpTzFftLineSchurPrecondCache:
    """Small true-action Schur correction on top of FP Fourier line factors."""

    basis: jnp.ndarray
    action: jnp.ndarray
    normal_inv: jnp.ndarray
    restrict_basis: jnp.ndarray | None
    damping: float
    tail0: int
    n_columns: int
    restriction_kind: str
    basis_labels: tuple[str, ...]


@dataclass(frozen=True)
class _TransportFpLocalGeomLinePrecondCache:
    """Block-Thomas FP transport factors with local non-averaged geometry."""

    inv_eff: jnp.ndarray
    lower_diag: jnp.ndarray
    super_diag: jnp.ndarray
    n_block: int


@dataclass(frozen=True)
class _TransportFpStructuredFBlockLuPrecondCache:
    """Host kinetic f-block sparse factor retaining full migrated couplings."""

    factor_bundle: object
    f_size: int
    metadata: dict[str, object]


@dataclass(frozen=True)
class _TransportFpFortranReducedLuPrecondCache:
    """Host sparse factor for a global Fortran-v3-style reduced transport Pmat."""

    factor_bundle: object
    linear_size: int
    metadata: dict[str, object]


@dataclass(frozen=True)
class _TransportFpDirectActiveBlockSchurPrecondCache:
    """Bounded-memory block inverse plus tail Schur factor for active FP transport."""

    block_inverse: object
    block_size: int
    kinetic_size: int
    tail_size: int
    c_tail: object | None
    mb_tail: np.ndarray | None
    schur_inverse: np.ndarray | None
    metadata: dict[str, object]
    factor: object | None = None


@dataclass(frozen=True)
class _TransportFpXBlockTzLuPrecondCache:
    """Per-(species,x) sparse factors over coupled (ell,theta,zeta) blocks."""

    factors: tuple[tuple[object | None, ...], ...]
    diag_inverses: tuple[tuple[np.ndarray | None, ...], ...]
    nxi_for_x: tuple[int, ...]
    factor_nbytes_estimate: int
    factor_nnz_estimate: int
    metadata: dict[str, object]


@dataclass(frozen=True)
class _SparseJaxPrecondCache:
    a_sp: object
    d_inv: jnp.ndarray
    omega: float
    sweeps: int


@dataclass(frozen=True)
class _PasTokamakThetaPrecondCache:
    """PAS tokamak theta/L block-tridiagonal preconditioner factors."""

    inv_a01: jnp.ndarray
    g01: jnp.ndarray
    inv_a: jnp.ndarray
    g: jnp.ndarray
    c_stream: jnp.ndarray
    c_mirror: jnp.ndarray
    m_theta: jnp.ndarray
    mirror_factor: jnp.ndarray
    mask_active: jnp.ndarray
    n_l_build: int
    tail_factors: tuple[tuple[object | None, ...], ...] | None = None


@dataclass(frozen=True)
class _PasTzPrecondCache:
    """PAS 3D (theta,zeta)/L block-tridiagonal preconditioner factors."""

    inv_a01: jnp.ndarray
    g01: jnp.ndarray
    inv_a: jnp.ndarray
    g: jnp.ndarray
    c_stream: jnp.ndarray
    c_mirror: jnp.ndarray
    m_tz: jnp.ndarray
    mirror_factor: jnp.ndarray
    mask_active: jnp.ndarray
    diag_inv: jnp.ndarray
    n_l_use: int


_TRANSPORT_PRECOND_CACHE: dict[tuple[object, ...], _TransportPrecondCache] = {}
_RHSMODE1_DIAG_PRECOND_CACHE: dict[tuple[object, ...], _TransportPrecondCache] = {}
_RHSMODE1_XBLOCK_PRECOND_CACHE: dict[tuple[object, ...], _TransportXBlockPrecondCache] = {}
_RHSMODE1_SCHUR_CACHE: dict[tuple[object, ...], jnp.ndarray] = {}
_TRANSPORT_SXBLOCK_LR_PRECOND_CACHE: dict[tuple[object, ...], _LowRankXBlockPrecondCache] = {}
_RHSMODE1_SXBLOCK_LR_PRECOND_CACHE: dict[tuple[object, ...], _LowRankXBlockPrecondCache] = {}
_TRANSPORT_XMG_PRECOND_CACHE: dict[tuple[object, ...], _TransportXmgPrecondCache] = {}
_RHSMODE1_XMG_PRECOND_CACHE: dict[tuple[object, ...], _TransportXmgPrecondCache] = {}
_RHSMODE1_XUPWIND_PRECOND_CACHE: dict[tuple[object, ...], _XUpwindPrecondCache] = {}
_RHSMODE1_SXBLOCK_PRECOND_CACHE: dict[tuple[object, ...], _TransportXBlockPrecondCache] = {}
_TRANSPORT_XBLOCK_PRECOND_CACHE: dict[tuple[object, ...], _TransportXBlockPrecondCache] = {}
_TRANSPORT_SXBLOCK_PRECOND_CACHE: dict[tuple[object, ...], _TransportXBlockPrecondCache] = {}
_TRANSPORT_TZFFT_PRECOND_CACHE: dict[tuple[object, ...], _TransportTzFftPrecondCache] = {}
_TRANSPORT_FP_TZFFT_PRECOND_CACHE: dict[tuple[object, ...], _TransportFpTzFftPrecondCache] = {}
_TRANSPORT_FP_TZFFT_LINE_PRECOND_CACHE: dict[tuple[object, ...], _TransportFpTzFftLinePrecondCache] = {}
_TRANSPORT_FP_TZFFT_LINE_SCHUR_PRECOND_CACHE: dict[
    tuple[object, ...], _TransportFpTzFftLineSchurPrecondCache
] = {}
_TRANSPORT_FP_LOCAL_GEOM_LINE_PRECOND_CACHE: dict[tuple[object, ...], _TransportFpLocalGeomLinePrecondCache] = {}
_TRANSPORT_FP_STRUCTURED_FBLOCK_LU_PRECOND_CACHE: dict[
    tuple[object, ...], _TransportFpStructuredFBlockLuPrecondCache
] = {}
_TRANSPORT_FP_FORTRAN_REDUCED_LU_PRECOND_CACHE: dict[
    tuple[object, ...], _TransportFpFortranReducedLuPrecondCache
] = {}
_TRANSPORT_FP_DIRECT_ACTIVE_BLOCK_SCHUR_PRECOND_CACHE: dict[
    tuple[object, ...], _TransportFpDirectActiveBlockSchurPrecondCache
] = {}
_TRANSPORT_FP_XBLOCK_TZ_LU_PRECOND_CACHE: dict[tuple[object, ...], _TransportFpXBlockTzLuPrecondCache] = {}
_TRANSPORT_FP_XBLOCK_TZ_LU_SCHUR_PRECOND_CACHE: dict[
    tuple[object, ...], _TransportFpTzFftLineSchurPrecondCache
] = {}
_RHSMODE23_PRECOND_CACHE: dict[tuple[object, ...], _RHSMode1PrecondCache] = {}
_RHSMODE1_SPARSE_JAX_CACHE: dict[tuple[object, ...], _SparseJaxPrecondCache] = {}
_RHSMODE1_PAS_TOKAMAK_THETA_CACHE: dict[tuple[object, ...], _PasTokamakThetaPrecondCache] = {}
_RHSMODE1_PAS_TZ_CACHE: dict[tuple[object, ...], _PasTzPrecondCache] = {}


# Mutable preconditioner policy context.
_PRECOND_SIZE_HINT: int | None = None
_PRECOND_GEOM_SCHEME_HINT: int | None = None
_PRECOND_USE_DKES_HINT: bool | None = None
_PRECOND_RHS1_PRECOND_KIND_HINT: str | None = None
_PRECOND_HAS_PAS_HINT: bool | None = None
_PRECOND_HAS_FP_HINT: bool | None = None
_PRECOND_INCLUDE_PHI1_HINT: bool | None = None
_PRECOND_RHS_MODE_HINT: int | None = None
_PRECOND_ER_ABS_HINT: float | None = None


def set_precond_size_hint(n: int | None) -> None:
    """Cache the current operator size for automatic preconditioner policy."""

    global _PRECOND_SIZE_HINT
    if n is None:
        _PRECOND_SIZE_HINT = None
    else:
        _PRECOND_SIZE_HINT = int(n)


def set_precond_policy_hints(
    *,
    geom_scheme: int | None = None,
    use_dkes: bool | None = None,
    rhs1_precond_kind: str | None = None,
    has_pas: bool | None = None,
    has_fp: bool | None = None,
    include_phi1: bool | None = None,
    rhs_mode: int | None = None,
    er_abs: float | None = None,
) -> None:
    """Cache operator metadata used by stability-first dtype and path policy."""

    global _PRECOND_GEOM_SCHEME_HINT
    global _PRECOND_USE_DKES_HINT
    global _PRECOND_RHS1_PRECOND_KIND_HINT
    global _PRECOND_HAS_PAS_HINT
    global _PRECOND_HAS_FP_HINT
    global _PRECOND_INCLUDE_PHI1_HINT
    global _PRECOND_RHS_MODE_HINT
    global _PRECOND_ER_ABS_HINT
    _PRECOND_GEOM_SCHEME_HINT = None if geom_scheme is None else int(geom_scheme)
    _PRECOND_USE_DKES_HINT = None if use_dkes is None else bool(use_dkes)
    _PRECOND_RHS1_PRECOND_KIND_HINT = None if rhs1_precond_kind is None else str(rhs1_precond_kind)
    _PRECOND_HAS_PAS_HINT = None if has_pas is None else bool(has_pas)
    _PRECOND_HAS_FP_HINT = None if has_fp is None else bool(has_fp)
    _PRECOND_INCLUDE_PHI1_HINT = None if include_phi1 is None else bool(include_phi1)
    _PRECOND_RHS_MODE_HINT = None if rhs_mode is None else int(rhs_mode)
    _PRECOND_ER_ABS_HINT = None if er_abs is None else float(er_abs)


def precond_policy_hints() -> solver_path_policy.PreconditionerPolicyHints:
    """Return the current preconditioner metadata as an immutable policy object."""

    return solver_path_policy.PreconditionerPolicyHints(
        size_hint=_PRECOND_SIZE_HINT,
        geom_scheme=_PRECOND_GEOM_SCHEME_HINT,
        use_dkes=_PRECOND_USE_DKES_HINT,
        rhs1_precond_kind=_PRECOND_RHS1_PRECOND_KIND_HINT,
        has_pas=_PRECOND_HAS_PAS_HINT,
        has_fp=_PRECOND_HAS_FP_HINT,
        include_phi1=_PRECOND_INCLUDE_PHI1_HINT,
        rhs_mode=_PRECOND_RHS_MODE_HINT,
        er_abs=_PRECOND_ER_ABS_HINT,
    )


def use_solver_jit(size_hint: int | None = None) -> bool:
    """Return whether the active solve should use the JIT Krylov wrapper."""

    return solver_path_policy.use_solver_jit(
        size_hint=size_hint,
        precond_size_hint=_PRECOND_SIZE_HINT,
    )


def auto_pas_geom4_fp32_precond_allowed(*, size_hint: int) -> bool:
    """Return whether the narrow PAS geometry-4 fp32 preconditioner path applies."""

    return solver_path_policy.auto_pas_geom4_fp32_precond_allowed(
        size_hint=int(size_hint),
        hints=precond_policy_hints(),
        backend=jax.default_backend(),
    )


def sparse_structural_tol() -> float:
    """Return the sparse structural drop tolerance for pattern extraction."""

    return solver_path_policy.sparse_structural_tol(default_tol=float(_THRESHOLD_FOR_INCLUSION))


def precond_dtype(size_hint: int | None = None) -> jnp.dtype:
    """Return the JAX dtype used for preconditioner factors in this context."""

    dtype_name = solver_path_policy.precond_dtype_name(
        size_hint=size_hint,
        hints=precond_policy_hints(),
        backend=jax.default_backend(),
    )
    return jnp.float32 if dtype_name == "float32" else jnp.float64


# Operator-shaping helpers for v3 preconditioners.
def diagonal_only(matrix: jnp.ndarray) -> jnp.ndarray:
    """Return a diagonal-only copy of a square matrix."""

    return jnp.diag(jnp.diag(matrix))


def block_diagonal_only(matrix: jnp.ndarray, block: int) -> jnp.ndarray:
    """Return a block-diagonal copy of a square matrix."""

    if int(block) <= 1:
        return diagonal_only(matrix)
    matrix_np = np.asarray(matrix, dtype=np.float64)
    n = int(matrix_np.shape[0])
    mask = np.zeros((n, n), dtype=bool)
    for start in range(0, n, int(block)):
        end = min(n, start + int(block))
        mask[start:end, start:end] = True
    matrix_np = np.where(mask, matrix_np, 0.0)
    return jnp.asarray(matrix_np, dtype=matrix.dtype)


_diag_only = diagonal_only
_block_diag_only = block_diagonal_only


def _build_rhsmode1_preconditioner_operator_point(op: V3FullSystemOperator) -> V3FullSystemOperator:
    """Return a simplified RHSMode=1 operator for point-block preconditioning.

    This is the original cheap RHSMode=1 preconditioner: it retains local x/L
    couplings and collisions while dropping theta/zeta derivative couplings
    (streaming, ExB, and magnetic-drift derivatives) by diagonalizing the derivative
    matrices.
    """
    if int(op.rhs_mode) != 1:
        return op

    fblock = op.fblock
    coll = replace(
        fblock.collisionless,
        ddtheta=_diag_only(fblock.collisionless.ddtheta),
        ddzeta=_diag_only(fblock.collisionless.ddzeta),
    )
    exb_theta = None if fblock.exb_theta is None else replace(
        fblock.exb_theta, ddtheta=_diag_only(fblock.exb_theta.ddtheta)
    )
    exb_zeta = None if fblock.exb_zeta is None else replace(
        fblock.exb_zeta, ddzeta=_diag_only(fblock.exb_zeta.ddzeta)
    )
    mag_theta = None
    if fblock.magdrift_theta is not None:
        mag_theta = replace(
            fblock.magdrift_theta,
            ddtheta_plus=_diag_only(fblock.magdrift_theta.ddtheta_plus),
            ddtheta_minus=_diag_only(fblock.magdrift_theta.ddtheta_minus),
        )
    mag_zeta = None
    if fblock.magdrift_zeta is not None:
        mag_zeta = replace(
            fblock.magdrift_zeta,
            ddzeta_plus=_diag_only(fblock.magdrift_zeta.ddzeta_plus),
            ddzeta_minus=_diag_only(fblock.magdrift_zeta.ddzeta_minus),
        )
    fblock_pc = replace(
        fblock,
        collisionless=coll,
        exb_theta=exb_theta,
        exb_zeta=exb_zeta,
        magdrift_theta=mag_theta,
        magdrift_zeta=mag_zeta,
    )
    return replace(op, fblock=fblock_pc)


def _build_rhsmode1_preconditioner_operator_fortran_reduced(
    op: V3FullSystemOperator,
    *,
    preconditioner_x: int = 1,
    preconditioner_xi: int = 1,
    preconditioner_species: int = 1,
    preconditioner_x_min_l: int = 0,
) -> V3FullSystemOperator:
    """Return a Fortran-v3-style reduced global RHSMode=1 preconditioner operator.

    SFINCS Fortran v3's default PETSc preconditioner is not a local point
    smoother: it keeps the angular streaming/drift derivatives and the global
    source/constraint rows, while simplifying selected radial, pitch-angle, and
    species couplings. This function provides the first SFINCS-JAX operator
    shaping for that route. It intentionally preserves theta/zeta coupling and
    only applies the x-diagonal simplification to terms that expose radial
    derivative matrices directly in the current JAX operator tree.

    ``preconditioner_species`` and ``preconditioner_x`` are applied to the full
    Fokker-Planck collision tensor when it is available. ``preconditioner_x_min_l``
    follows the Fortran rule: radial simplification is only applied to rows with
    Legendre index ``L >= preconditioner_x_min_L``. ``preconditioner_xi`` follows
    v3's matrix-0 rule by dropping the collisionless ``L±2`` pitch couplings
    while preserving diagonal-in-``L`` drift/Er terms and streaming ``L±1`` terms.
    """
    if int(op.rhs_mode) != 1:
        return op

    fblock = op.fblock
    fp = fblock.fp
    if fp is not None and hasattr(fp, "mat"):
        mat = jnp.asarray(fp.mat)
        if int(preconditioner_species) > 0 and mat.ndim == 5:
            species_eye = jnp.eye(int(op.n_species), dtype=mat.dtype)
            mat = mat * species_eye[:, :, None, None, None]
        if int(preconditioner_x) > 0 and mat.ndim == 5:
            n_x = int(op.n_x)
            row = jnp.arange(n_x)[:, None]
            col = jnp.arange(n_x)[None, :]
            if int(preconditioner_x) == 1:
                x_mask = row == col
            elif int(preconditioner_x) == 2:
                x_mask = col >= row
            elif int(preconditioner_x) in {3, 5}:
                x_mask = jnp.abs(row - col) <= 1
            elif int(preconditioner_x) == 4:
                x_mask = (col == row) | (col == row + 1)
            else:
                x_mask = row == col
            if int(preconditioner_x_min_l) > 0:
                ell = jnp.arange(int(mat.shape[2]), dtype=jnp.int32)
                l_gate = ell >= int(preconditioner_x_min_l)
                x_mask = jnp.where(l_gate[:, None, None], x_mask[None, :, :], True)
                mat = mat * x_mask[None, None, :, :, :]
            else:
                mat = mat * x_mask[None, None, None, :, :]
        fp = replace(fp, mat=mat)

    drop_l2 = int(preconditioner_xi) > 0

    def _maybe_drop_l2(term):
        if term is None or not drop_l2 or not hasattr(term, "drop_l2_couplings"):
            return term
        return replace(term, drop_l2_couplings=True)

    term_replacements = {
        "fp": fp,
    }
    for name in ("magdrift_theta", "magdrift_zeta", "magdrift_xidot", "er_xidot"):
        if hasattr(fblock, name):
            term_replacements[name] = _maybe_drop_l2(getattr(fblock, name))

    er_xdot = getattr(fblock, "er_xdot", None)
    if er_xdot is not None:
        replacements = {}
        if int(preconditioner_x) > 0 and int(preconditioner_x_min_l) <= 0:
            replacements["ddx_plus"] = _diag_only(er_xdot.ddx_plus)
            replacements["ddx_minus"] = _diag_only(er_xdot.ddx_minus)
        if drop_l2 and hasattr(er_xdot, "drop_l2_couplings"):
            replacements["drop_l2_couplings"] = True
        if replacements:
            er_xdot = replace(er_xdot, **replacements)
    if hasattr(fblock, "er_xdot"):
        term_replacements["er_xdot"] = er_xdot

    fblock_pc = replace(
        fblock,
        # Keep collisionless ddtheta/ddzeta, ExB, magnetic-drift theta/zeta,
        # collisions, source rows, and constraint rows globally coupled.
        **term_replacements,
    )
    return replace(op, fblock=fblock_pc)


def _build_transport_preconditioner_operator_point(op: V3FullSystemOperator) -> V3FullSystemOperator:
    """Return a simplified transport operator for point-block preconditioning.

    This mirrors `_build_rhsmode1_preconditioner_operator_point` but does not
    require RHSMode=1, since RHSMode=2/3 transport solves reuse the same operator
    structure with different right-hand sides.
    """
    fblock = op.fblock
    coll = replace(
        fblock.collisionless,
        ddtheta=_diag_only(fblock.collisionless.ddtheta),
        ddzeta=_diag_only(fblock.collisionless.ddzeta),
    )
    exb_theta = None if fblock.exb_theta is None else replace(
        fblock.exb_theta, ddtheta=_diag_only(fblock.exb_theta.ddtheta)
    )
    exb_zeta = None if fblock.exb_zeta is None else replace(
        fblock.exb_zeta, ddzeta=_diag_only(fblock.exb_zeta.ddzeta)
    )
    mag_theta = None
    if fblock.magdrift_theta is not None:
        mag_theta = replace(
            fblock.magdrift_theta,
            ddtheta_plus=_diag_only(fblock.magdrift_theta.ddtheta_plus),
            ddtheta_minus=_diag_only(fblock.magdrift_theta.ddtheta_minus),
        )
    mag_zeta = None
    if fblock.magdrift_zeta is not None:
        mag_zeta = replace(
            fblock.magdrift_zeta,
            ddzeta_plus=_diag_only(fblock.magdrift_zeta.ddzeta_plus),
            ddzeta_minus=_diag_only(fblock.magdrift_zeta.ddzeta_minus),
        )
    fblock_pc = replace(
        fblock,
        collisionless=coll,
        exb_theta=exb_theta,
        exb_zeta=exb_zeta,
        magdrift_theta=mag_theta,
        magdrift_zeta=mag_zeta,
    )
    return replace(op, fblock=fblock_pc)


def _build_transport_preconditioner_operator_fortran_reduced(
    op: V3FullSystemOperator,
    *,
    preconditioner_x: int = 1,
    preconditioner_xi: int = 1,
    preconditioner_species: int = 1,
    preconditioner_x_min_l: int = 0,
    keep_theta_zeta: bool = True,
) -> V3FullSystemOperator:
    """Return a Fortran-v3-style reduced transport preconditioner operator.

    SFINCS Fortran v3 uses the true matrix as ``Amat`` and a distinct
    ``whichMatrix=0`` reduced matrix as ``Pmat`` for PETSc.  This transport
    helper applies the same x/species/pitch simplifications as the RHSMode=1
    reduced operator but works for RHSMode=2/3.  By default it keeps theta/zeta
    derivative couplings, matching the Fortran v3 defaults
    ``preconditioner_theta=0`` and ``preconditioner_zeta=0``.  Set
    ``keep_theta_zeta=False`` only for a smaller diagnostic Pmat that explicitly
    drops the angular streaming/drift graph.
    """

    rhs_mode_original = int(op.rhs_mode)
    op_rhs1 = replace(op, rhs_mode=1)
    op_pc = _build_rhsmode1_preconditioner_operator_fortran_reduced(
        op_rhs1,
        preconditioner_x=int(preconditioner_x),
        preconditioner_xi=int(preconditioner_xi),
        preconditioner_species=int(preconditioner_species),
        preconditioner_x_min_l=int(preconditioner_x_min_l),
    )
    op_pc = replace(op_pc, rhs_mode=rhs_mode_original)
    if not bool(keep_theta_zeta):
        op_pc = _build_transport_preconditioner_operator_point(op_pc)
    return op_pc


def _build_rhsmode1_preconditioner_operator_theta_line(op: V3FullSystemOperator) -> V3FullSystemOperator:
    """Return a simplified RHSMode=1 operator for theta-line preconditioning.

    Keep full theta derivative couplings but drop zeta derivative couplings. This enables
    a significantly stronger preconditioner than point-block Jacobi, while remaining
    much cheaper than a full (theta,zeta)-coupled preconditioner.
    """
    if int(op.rhs_mode) != 1:
        return op

    fblock = op.fblock
    coll = replace(
        fblock.collisionless,
        ddzeta=_diag_only(fblock.collisionless.ddzeta),
    )
    exb_theta = fblock.exb_theta
    exb_zeta = None if fblock.exb_zeta is None else replace(
        fblock.exb_zeta, ddzeta=_diag_only(fblock.exb_zeta.ddzeta)
    )
    mag_theta = fblock.magdrift_theta
    mag_zeta = None
    if fblock.magdrift_zeta is not None:
        mag_zeta = replace(
            fblock.magdrift_zeta,
            ddzeta_plus=_diag_only(fblock.magdrift_zeta.ddzeta_plus),
            ddzeta_minus=_diag_only(fblock.magdrift_zeta.ddzeta_minus),
        )
    fblock_pc = replace(
        fblock,
        collisionless=coll,
        exb_theta=exb_theta,
        exb_zeta=exb_zeta,
        magdrift_theta=mag_theta,
        magdrift_zeta=mag_zeta,
    )
    return replace(op, fblock=fblock_pc)


def _build_rhsmode1_preconditioner_operator_theta_dd(
    op: V3FullSystemOperator, *, block: int
) -> V3FullSystemOperator:
    """Return a theta-block domain-decomposition operator for preconditioning.

    This operator shaping is used by RHSMode=1 and RHSMode=2/3 transport solves.
    """

    fblock = op.fblock
    coll = replace(
        fblock.collisionless,
        ddtheta=_block_diag_only(fblock.collisionless.ddtheta, block),
        ddzeta=_diag_only(fblock.collisionless.ddzeta),
    )
    exb_theta = fblock.exb_theta
    exb_zeta = None if fblock.exb_zeta is None else replace(
        fblock.exb_zeta, ddzeta=_diag_only(fblock.exb_zeta.ddzeta)
    )
    mag_theta = fblock.magdrift_theta
    if mag_theta is not None:
        mag_theta = replace(
            mag_theta,
            ddtheta_plus=_block_diag_only(mag_theta.ddtheta_plus, block),
            ddtheta_minus=_block_diag_only(mag_theta.ddtheta_minus, block),
        )
    mag_zeta = None
    if fblock.magdrift_zeta is not None:
        mag_zeta = replace(
            fblock.magdrift_zeta,
            ddzeta_plus=_diag_only(fblock.magdrift_zeta.ddzeta_plus),
            ddzeta_minus=_diag_only(fblock.magdrift_zeta.ddzeta_minus),
        )
    fblock_pc = replace(
        fblock,
        collisionless=coll,
        exb_theta=exb_theta,
        exb_zeta=exb_zeta,
        magdrift_theta=mag_theta,
        magdrift_zeta=mag_zeta,
    )
    return replace(op, fblock=fblock_pc)


def _build_rhsmode1_preconditioner_operator_zeta_line(op: V3FullSystemOperator) -> V3FullSystemOperator:
    """Return a simplified RHSMode=1 operator for zeta-line preconditioning.

    Keep full zeta derivative couplings but drop theta derivative couplings. This is the
    zeta-analog of `_build_rhsmode1_preconditioner_operator_theta_line`.
    """
    if int(op.rhs_mode) != 1:
        return op

    fblock = op.fblock
    coll = replace(
        fblock.collisionless,
        ddtheta=_diag_only(fblock.collisionless.ddtheta),
    )
    exb_theta = None if fblock.exb_theta is None else replace(
        fblock.exb_theta, ddtheta=_diag_only(fblock.exb_theta.ddtheta)
    )
    exb_zeta = fblock.exb_zeta
    mag_theta = None
    if fblock.magdrift_theta is not None:
        mag_theta = replace(
            fblock.magdrift_theta,
            ddtheta_plus=_diag_only(fblock.magdrift_theta.ddtheta_plus),
            ddtheta_minus=_diag_only(fblock.magdrift_theta.ddtheta_minus),
        )
    mag_zeta = fblock.magdrift_zeta
    fblock_pc = replace(
        fblock,
        collisionless=coll,
        exb_theta=exb_theta,
        exb_zeta=exb_zeta,
        magdrift_theta=mag_theta,
        magdrift_zeta=mag_zeta,
    )
    return replace(op, fblock=fblock_pc)


def _build_rhsmode1_preconditioner_operator_zeta_dd(
    op: V3FullSystemOperator, *, block: int
) -> V3FullSystemOperator:
    """Return a zeta-block domain-decomposition operator for preconditioning.

    This operator shaping is used by RHSMode=1 and RHSMode=2/3 transport solves.
    """

    fblock = op.fblock
    coll = replace(
        fblock.collisionless,
        ddtheta=_diag_only(fblock.collisionless.ddtheta),
        ddzeta=_block_diag_only(fblock.collisionless.ddzeta, block),
    )
    exb_theta = None if fblock.exb_theta is None else replace(
        fblock.exb_theta, ddtheta=_diag_only(fblock.exb_theta.ddtheta)
    )
    exb_zeta = fblock.exb_zeta
    mag_theta = None
    if fblock.magdrift_theta is not None:
        mag_theta = replace(
            fblock.magdrift_theta,
            ddtheta_plus=_diag_only(fblock.magdrift_theta.ddtheta_plus),
            ddtheta_minus=_diag_only(fblock.magdrift_theta.ddtheta_minus),
        )
    mag_zeta = fblock.magdrift_zeta
    if mag_zeta is not None:
        mag_zeta = replace(
            mag_zeta,
            ddzeta_plus=_block_diag_only(mag_zeta.ddzeta_plus, block),
            ddzeta_minus=_block_diag_only(mag_zeta.ddzeta_minus, block),
        )
    fblock_pc = replace(
        fblock,
        collisionless=coll,
        exb_theta=exb_theta,
        exb_zeta=exb_zeta,
        magdrift_theta=mag_theta,
        magdrift_zeta=mag_zeta,
    )
    return replace(op, fblock=fblock_pc)


# Setup utilities shared by sparse and block preconditioner builders.
def precond_chunk_cols(
    total_size: int,
    n_cols: int,
    *,
    environ: Mapping[str, str] | None = None,
) -> int:
    """Choose how many basis columns to probe at once during setup.

    The explicit column override wins over the memory-budget estimate. Invalid
    environment values deliberately fall back to conservative defaults, matching
    the historical driver behavior.
    """

    env = os.environ if environ is None else environ
    env_cols = env.get("SFINCS_JAX_PRECOND_CHUNK", "").strip()
    if env_cols:
        try:
            cols = int(env_cols)
            if cols > 0:
                return min(cols, n_cols)
        except ValueError:
            pass
    env_max_mb = env.get("SFINCS_JAX_PRECOND_MAX_MB", "").strip()
    try:
        max_mb = float(env_max_mb) if env_max_mb else 256.0
    except ValueError:
        max_mb = 256.0
    if max_mb <= 0:
        return n_cols
    bytes_per_row = int(total_size) * 8
    if bytes_per_row <= 0:
        return n_cols
    max_cols = max(1, int((max_mb * 1e6) // bytes_per_row))
    return min(n_cols, max_cols)


def matvec_submatrix(
    op_pc: object,
    *,
    col_idx: np.ndarray,
    row_idx: np.ndarray,
    total_size: int,
    chunk_cols: int,
    apply_operator_fn: Callable[..., jnp.ndarray],
) -> np.ndarray:
    """Assemble selected rows of selected operator columns by batched probes."""

    col_idx = np.asarray(col_idx, dtype=np.int32)
    row_idx_jnp = jnp.asarray(row_idx, dtype=jnp.int32)
    blocks: list[np.ndarray] = []
    for start in range(0, int(col_idx.shape[0]), int(chunk_cols)):
        idx = col_idx[start : start + int(chunk_cols)]
        basis = jax.nn.one_hot(jnp.asarray(idx, dtype=jnp.int32), total_size, dtype=jnp.float64)
        y = jax.vmap(
            lambda v: apply_operator_fn(
                op_pc,
                v,
                include_jacobian_terms=True,
                allow_sharding=False,
            )
        )(basis)
        y_sub = y[:, row_idx_jnp]
        blocks.append(np.asarray(y_sub, dtype=np.float64))
    if len(blocks) == 1:
        return blocks[0]
    return np.concatenate(blocks, axis=0)


def matvec_submatrix_v3_unsharded(
    op_pc: object,
    *,
    col_idx: np.ndarray,
    row_idx: np.ndarray,
    total_size: int,
    chunk_cols: int,
) -> np.ndarray:
    """Assemble selected V3 operator rows with the unsharded operator apply."""

    from sfincs_jax.operators.profile_response.system import apply_v3_full_system_operator  # noqa: PLC0415

    return matvec_submatrix(
        op_pc,
        col_idx=col_idx,
        row_idx=row_idx,
        total_size=total_size,
        chunk_cols=chunk_cols,
        apply_operator_fn=apply_v3_full_system_operator,
    )


def hash_array(arr: jnp.ndarray | np.ndarray) -> str:
    """Stable short hash for numeric arrays used in preconditioner cache keys."""

    arr_np = np.asarray(arr, dtype=np.float64)
    return hashlib.blake2b(arr_np.tobytes(), digest_size=8).hexdigest()


def rhs_mode1_precond_cache_key(
    op: object,
    kind: str,
    *,
    precond_dtype: object,
) -> tuple[object, ...]:
    """Build the operator-only cache key for RHSMode=1 preconditioners.

    RHS-only gradients are deliberately excluded so preconditioners can be
    reused across whichRHS/profile scan points that share the same linear
    operator.
    """

    nxi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
    return (
        kind,
        str(precond_dtype),
        int(op.rhs_mode),
        int(op.n_species),
        int(op.n_x),
        int(op.n_xi),
        int(op.n_theta),
        int(op.n_zeta),
        int(op.constraint_scheme),
        int(op.quasineutrality_option),
        bool(op.include_phi1),
        bool(op.include_phi1_in_kinetic),
        bool(op.with_adiabatic),
        float(op.alpha),
        float(op.delta),
        float(op.dphi_hat_dpsi_hat),
        hash_array(op.adiabatic_z),
        hash_array(op.adiabatic_nhat),
        hash_array(op.adiabatic_that),
        hash_array(op.z_s),
        hash_array(op.m_hat),
        hash_array(op.t_hat),
        hash_array(op.n_hat),
        hash_array(op.theta_weights),
        hash_array(op.zeta_weights),
        hash_array(op.b_hat),
        hash_array(op.d_hat),
        hash_array(op.b_hat_sub_theta),
        hash_array(op.b_hat_sub_zeta),
        hash_array(op.x),
        hash_array(op.x_weights),
        tuple(nxi_for_x.tolist()),
    )


def rhs_mode1_structured_fblock_cache_key(
    op: object,
    kind: str,
    *,
    precond_dtype: object,
    params: tuple[object, ...] = (),
) -> tuple[object, ...]:
    """Build a cache key for structured RHSMode=1 f-block preconditioners."""

    phi1_hash = None
    if getattr(op, "phi1_hat_base", None) is not None:
        phi1_hash = hash_array(op.phi1_hat_base)
    return (
        *rhs_mode1_precond_cache_key(
            op,
            f"structured_fblock_{kind}",
            precond_dtype=precond_dtype,
        ),
        phi1_hash,
        *tuple(params),
    )


def transport_precond_cache_key(
    op: object,
    kind: str,
    *,
    precond_dtype: object,
) -> tuple[object, ...]:
    """Build the cache key for RHSMode=2/3 transport preconditioners."""

    nxi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
    pas = op.fblock.pas
    fp = op.fblock.fp
    return (
        kind,
        str(precond_dtype),
        int(op.n_species),
        int(op.n_x),
        int(op.n_xi),
        int(op.n_theta),
        int(op.n_zeta),
        float(op.fblock.identity_shift),
        bool(pas is not None),
        float(pas.nu_n) if pas is not None else None,
        float(pas.krook) if pas is not None else None,
        hash_array(pas.nu_d_hat) if pas is not None else None,
        bool(fp is not None),
        hash_array(fp.mat) if fp is not None else None,
        tuple(nxi_for_x.tolist()),
    )
