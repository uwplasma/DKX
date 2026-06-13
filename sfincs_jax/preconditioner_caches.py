"""Preconditioner cache containers shared by the v3 solve driver.

The solve driver owns the numerical setup/apply routines, but these passive
dataclasses and global registries are intentionally kept in a narrow module so
tests can verify cache reuse without importing the full driver implementation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


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
