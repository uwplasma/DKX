"""Fortran-reduced sparse-factor preconditioner for RHSMode=2/3 transport."""

from __future__ import annotations

from collections.abc import Callable
import os

import jax
import jax.numpy as jnp
import numpy as np

from sfincs_jax.explicit_sparse import (
    SparseOperatorBundle,
    admit_sparse_factor_against_operator,
    analyze_sparse_symbolic_structure,
    estimate_multifrontal_direct_lu_nbytes,
    factorize_host_sparse_operator,
    wrap_sparse_factor_with_coarse_correction,
)
from sfincs_jax.preconditioner_caches import (
    _TRANSPORT_FP_FORTRAN_REDUCED_LU_PRECOND_CACHE,
    _TransportFpFortranReducedLuPrecondCache,
)
from sfincs_jax.preconditioner_operators import _build_transport_preconditioner_operator_fortran_reduced
from sfincs_jax.problems.profile_response.policies import _hash_numpy_array_for_cache
from sfincs_jax.problems.transport_matrix.direct_pmat import (
    _build_rhsmode23_direct_pmat_physics_coarse_basis,
    _try_build_rhsmode23_fp_fortran_reduced_direct_pmat_bundle,
)
from sfincs_jax.v3_sparse_pattern import (
    summarize_v3_sparse_pattern,
    v3_full_system_fortran_reduced_preconditioner_sparsity_pattern,
    v3_full_system_fortran_reduced_preconditioner_sparsity_pattern_for_indices,
)
from sfincs_jax.v3_system import V3FullSystemOperator, apply_v3_full_system_operator_cached

__all__ = ["build_transport_fp_fortran_reduced_lu_preconditioner"]


def build_transport_fp_fortran_reduced_lu_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    active_indices_np: np.ndarray | None = None,
    emit: Callable[[int, str], None] | None = None,
    fallback_builder: Callable[..., Callable[[jnp.ndarray], jnp.ndarray]],
    transport_precond_cache_key: Callable[[V3FullSystemOperator, str], tuple[object, ...]],
    build_host_sparse_direct_factor_from_matvec: Callable[..., object],
    host_physical_memory_mb: Callable[[], float | None],
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Global reduced sparse-factor transport preconditioner.

    This is the closest SFINCS-JAX transport analogue to the Fortran v3
    PETSc setup: GMRES still applies the true operator, while this builder
    materializes and factors a separate reduced ``Pmat``.  It is intentionally
    opt-in until production-size residual gates prove that the setup cost and
    memory footprint are justified.
    """

    if int(op.rhs_mode) not in {2, 3} or op.fblock.fp is None:
        return fallback_builder(
            op=op,
            reduce_full=reduce_full,
            expand_reduced=expand_reduced,
        )

    use_reduced = reduce_full is not None and expand_reduced is not None
    if use_reduced and active_indices_np is None:
        return fallback_builder(
            op=op,
            reduce_full=reduce_full,
            expand_reduced=expand_reduced,
        )

    def _int_env(name: str, default: int, *, minimum: int = 0) -> int:
        value = os.environ.get(name, "").strip()
        try:
            parsed = int(value) if value else int(default)
        except ValueError:
            parsed = int(default)
        return max(int(minimum), int(parsed))

    def _float_env(name: str, default: float, *, minimum: float = 0.0) -> float:
        value = os.environ.get(name, "").strip()
        try:
            parsed = float(value) if value else float(default)
        except ValueError:
            parsed = float(default)
        return max(float(minimum), float(parsed))

    def _bool_env(name: str, default: bool) -> bool:
        value = os.environ.get(name, "").strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off"}:
            return False
        return bool(default)

    # Fortran v3 defaults reduce x and xi couplings, but keep the full
    # theta/zeta derivative matrices (preconditioner_theta/zeta=0).  That exact
    # Pmat is available via env overrides, but this opt-in transport candidate
    # defaults to the stronger x/xi-coupled variant because the default-reduced
    # Pmat is too slow for the current FP geometry-rich residual gates.
    preconditioner_x = _int_env("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_PRECONDITIONER_X", 0)
    preconditioner_xi = _int_env("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_PRECONDITIONER_XI", 0)
    preconditioner_species = _int_env("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_PRECONDITIONER_SPECIES", 1)
    preconditioner_x_min_l = _int_env("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_PRECONDITIONER_X_MIN_L", 0)
    keep_theta_zeta = _bool_env("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_KEEPS_THETA_ZETA", True)
    pc_shift = _float_env("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SHIFT", 1.0e-10)
    max_factor_mb = _float_env("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_FACTOR_MAX_MB", 4096.0)
    direct_pmat_enabled = _bool_env("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_DIRECT", True)
    symbolic_ordering = (
        os.environ.get("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ORDERING", "").strip().lower()
        or "mumps_like"
    )
    symbolic_block_size = _int_env("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_BLOCK_SIZE", 4096, minimum=1)
    symbolic_block_overlap = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_BLOCK_OVERLAP",
        0,
        minimum=0,
    )
    symbolic_coarse_max_cols = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_COARSE_MAX_COLS",
        256,
        minimum=1,
    )
    symbolic_coarse_probe_cols = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_COARSE_PROBE_COLS",
        4,
        minimum=0,
    )
    symbolic_coarse_damping = _float_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_COARSE_DAMPING",
        1.0,
        minimum=0.0,
    )
    symbolic_coarse_regularization_rel = _float_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_COARSE_REG_REL",
        1.0e-10,
        minimum=0.0,
    )
    symbolic_physics_coarse_enabled = _bool_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_PHYSICS_COARSE",
        True,
    )
    symbolic_physics_coarse_max_cols = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_PHYSICS_COARSE_MAX_COLS",
        32,
        minimum=1,
    )
    symbolic_schur_max_separator_cols = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_SCHUR_MAX_SEPARATOR_COLS",
        256,
        minimum=0,
    )
    symbolic_schur_boundary_width = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_SCHUR_BOUNDARY_WIDTH",
        1,
        minimum=0,
    )
    symbolic_schur_high_degree_cols = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_SCHUR_HIGH_DEGREE_COLS",
        64,
        minimum=0,
    )
    symbolic_schur_regularization_rel = _float_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_SCHUR_REG_REL",
        1.0e-12,
        minimum=0.0,
    )
    symbolic_frontal_max_separator_cols = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_FRONTAL_MAX_SEPARATOR_COLS",
        1024,
        minimum=0,
    )
    symbolic_frontal_boundary_width = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_FRONTAL_BOUNDARY_WIDTH",
        1,
        minimum=0,
    )
    symbolic_frontal_high_degree_cols = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_FRONTAL_HIGH_DEGREE_COLS",
        128,
        minimum=0,
    )
    symbolic_frontal_max_superblock_size = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_FRONTAL_MAX_SUPERBLOCK_SIZE",
        8192,
        minimum=1,
    )
    symbolic_frontal_max_superblock_blocks = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_FRONTAL_MAX_SUPERBLOCK_BLOCKS",
        8,
        minimum=1,
    )
    symbolic_frontal_min_cross_nnz = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_FRONTAL_MIN_CROSS_NNZ",
        1,
        minimum=1,
    )
    symbolic_frontal_min_cross_separator_fraction = _float_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_FRONTAL_MIN_CROSS_SEPARATOR_FRACTION",
        0.0,
        minimum=0.0,
    )
    symbolic_frontal_regularization_rel = _float_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_FRONTAL_REG_REL",
        1.0e-12,
        minimum=0.0,
    )
    symbolic_frontal_max_dense_rhs_entries = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_FRONTAL_MAX_DENSE_RHS_ENTRIES",
        0,
        minimum=0,
    )
    symbolic_frontal_max_dense_rhs_cols_per_block = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_FRONTAL_MAX_DENSE_RHS_COLS_PER_BLOCK",
        0,
        minimum=0,
    )
    symbolic_blr_frontal_tol = _float_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_BLR_FRONTAL_TOL",
        1.0e-6,
        minimum=0.0,
    )
    symbolic_blr_frontal_max_rank = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_BLR_FRONTAL_MAX_RANK",
        64,
        minimum=1,
    )
    symbolic_blr_frontal_min_cols = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_BLR_FRONTAL_MIN_COLS",
        8,
        minimum=1,
    )
    symbolic_blr_frontal_gmres_rtol = _float_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_BLR_FRONTAL_GMRES_RTOL",
        1.0e-6,
        minimum=0.0,
    )
    symbolic_blr_frontal_gmres_atol = _float_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_BLR_FRONTAL_GMRES_ATOL",
        0.0,
        minimum=0.0,
    )
    symbolic_blr_frontal_gmres_maxiter = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_BLR_FRONTAL_GMRES_MAXITER",
        50,
        minimum=1,
    )
    symbolic_blr_frontal_gmres_restart = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_BLR_FRONTAL_GMRES_RESTART",
        64,
        minimum=1,
    )
    symbolic_blr_frontal_woodbury_max_rank = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_BLR_FRONTAL_WOODBURY_MAX_RANK",
        512,
        minimum=0,
    )
    symbolic_blr_frontal_woodbury_max_condition = _float_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_BLR_FRONTAL_WOODBURY_MAX_CONDITION",
        1.0e8,
        minimum=1.0,
    )
    symbolic_nd_max_leaf_size = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ND_MAX_LEAF_SIZE",
        4096,
        minimum=1,
    )
    symbolic_nd_max_terminal_factor_size = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ND_MAX_TERMINAL_FACTOR_SIZE",
        32768,
        minimum=1,
    )
    symbolic_nd_max_depth = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ND_MAX_DEPTH",
        4,
        minimum=0,
    )
    symbolic_nd_separator_width = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ND_SEPARATOR_WIDTH",
        64,
        minimum=1,
    )
    symbolic_nd_max_separator_cols = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ND_MAX_SEPARATOR_COLS",
        4096,
        minimum=1,
    )
    symbolic_nd_high_degree_cols = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ND_HIGH_DEGREE_COLS",
        64,
        minimum=0,
    )
    symbolic_nd_regularization_rel = _float_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ND_REG_REL",
        1.0e-12,
        minimum=0.0,
    )
    symbolic_nd_max_dense_rhs_entries = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ND_MAX_DENSE_RHS_ENTRIES",
        0,
        minimum=0,
    )
    symbolic_nd_max_dense_rhs_entries_per_child = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ND_MAX_DENSE_RHS_ENTRIES_PER_CHILD",
        0,
        minimum=0,
    )
    symbolic_nd_max_dense_rhs_cols_per_child = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ND_MAX_DENSE_RHS_COLS_PER_CHILD",
        0,
        minimum=0,
    )
    symbolic_nd_max_setup_s = _float_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ND_MAX_SETUP_S",
        0.0,
        minimum=0.0,
    )
    symbolic_nd_compress_updates = _bool_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ND_COMPRESS_UPDATES",
        False,
    )
    symbolic_nd_residual_polish_steps = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ND_RESIDUAL_POLISH_STEPS",
        2,
        minimum=0,
    )
    symbolic_nd_residual_polish_damping = _float_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ND_RESIDUAL_POLISH_DAMPING",
        1.0,
        minimum=0.0,
    )
    symbolic_superblock_max_size = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_SUPERBLOCK_MAX_SIZE",
        32768,
        minimum=1,
    )
    symbolic_superblock_max_blocks = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_SUPERBLOCK_MAX_BLOCKS",
        8,
        minimum=1,
    )
    symbolic_superblock_min_cross_nnz = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_SUPERBLOCK_MIN_CROSS_NNZ",
        1,
        minimum=1,
    )
    symbolic_superblock_min_retained_cross_fraction = _float_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_SUPERBLOCK_MIN_RETAINED_CROSS_FRACTION",
        0.0,
        minimum=0.0,
    )
    symbolic_superblock_regularization_rel = _float_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_SUPERBLOCK_REG_REL",
        1.0e-12,
        minimum=0.0,
    )
    symbolic_numeric_parallel_workers_default = min(4, max(1, int(os.cpu_count() or 1)))
    symbolic_numeric_parallel_workers = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_NUMERIC_PARALLEL_WORKERS",
        symbolic_numeric_parallel_workers_default,
        minimum=1,
    )
    symbolic_nd_parallel_update_workers = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ND_PARALLEL_UPDATE_WORKERS",
        symbolic_numeric_parallel_workers,
        minimum=1,
    )
    symbolic_max_permutation_size = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_MAX_PERMUTATION_SIZE",
        250_000,
        minimum=0,
    )
    symbolic_admission_enabled = _bool_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ADMISSION",
        True,
    )
    symbolic_admission_max_rel = _float_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ADMISSION_MAX_REL",
        1.0e-2,
        minimum=0.0,
    )
    symbolic_admission_min_improvement = _float_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ADMISSION_MIN_IMPROVEMENT",
        10.0,
        minimum=0.0,
    )
    symbolic_admission_probes = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ADMISSION_PROBES",
        4,
        minimum=1,
    )
    symbolic_admission_rescue_lu = _bool_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ADMISSION_RESCUE_LU",
        True,
    )
    symbolic_admission_rescue_lu_max_mb = _float_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ADMISSION_RESCUE_LU_MAX_MB",
        max_factor_mb,
        minimum=0.0,
    )
    auto_exact_rescue_enabled = _bool_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_AUTO_EXACT_RESCUE",
        True,
    )
    auto_exact_rescue_ram_fraction = _float_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_AUTO_EXACT_RESCUE_RAM_FRACTION",
        0.45,
        minimum=0.0,
    )
    host_memory_mb = host_physical_memory_mb()
    auto_exact_rescue_default_max_mb = (
        0.0
        if host_memory_mb is None
        else max(0.0, float(host_memory_mb) * float(auto_exact_rescue_ram_fraction))
    )
    auto_exact_rescue_max_mb = _float_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_AUTO_EXACT_RESCUE_MAX_MB",
        auto_exact_rescue_default_max_mb,
        minimum=0.0,
    )
    auto_exact_rescue_max_size = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_AUTO_EXACT_RESCUE_MAX_SIZE",
        250_000,
        minimum=0,
    )
    auto_exact_rescue_max_factor_entries = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_AUTO_EXACT_RESCUE_MAX_FACTOR_ENTRIES",
        250_000_000,
        minimum=0,
    )
    direct_admission_enabled = _bool_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_DIRECT_ADMISSION",
        True,
    )
    direct_admission_explicit_enabled = _bool_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_DIRECT_ADMISSION_EXPLICIT",
        False,
    )
    factor_dtype_env = os.environ.get("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_FACTOR_DTYPE", "").strip().lower()
    factor_dtype = np.dtype(np.float32) if factor_dtype_env in {"float32", "fp32", "32"} else np.dtype(np.float64)
    factor_kind_env = os.environ.get("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_FACTOR", "").strip().lower()

    active_hash = "full"
    active_indices_use: np.ndarray | None = None
    if use_reduced:
        active_indices_use = np.asarray(active_indices_np, dtype=np.int32).reshape((-1,))
        active_hash = _hash_numpy_array_for_cache(active_indices_use)
        linear_size = int(active_indices_use.size)
    else:
        linear_size = int(op.total_size)
    default_factor_kind = (
        factor_kind_env
        if factor_kind_env
        in {
            "lu",
            "ilu",
            "jacobi",
            "spilu",
            "diag",
            "diagonal",
            "none",
            "symbolic_block_lu",
            "block_lu",
            "native_block_lu",
            "symbolic_lu",
            "symbolic_block_schur_lu",
            "block_schur_lu",
            "native_block_schur_lu",
            "symbolic_schur_lu",
            "symbolic_block_lu_coarse",
            "block_lu_coarse",
            "native_block_lu_coarse",
            "symbolic_lu_coarse",
            "symbolic_frontal_schur_lu",
            "frontal_schur_lu",
            "native_frontal_schur_lu",
            "multifrontal_schur_lu",
            "symbolic_blr_frontal_schur_lu",
            "blr_frontal_schur_lu",
            "native_blr_frontal_schur_lu",
            "compressed_frontal_schur_lu",
            "symbolic_nd_frontal_schur_lu",
            "nd_frontal_schur_lu",
            "nested_dissection_frontal_schur_lu",
            "native_nd_frontal_schur_lu",
            "multilevel_frontal_schur_lu",
            "symbolic_superblock_lu",
            "superblock_lu",
            "native_superblock_lu",
            "block_edge_lu",
        }
        else "lu"
    )
    if default_factor_kind in {"spilu"}:
        default_factor_kind = "ilu"
    elif default_factor_kind in {"diag", "diagonal", "none"}:
        default_factor_kind = "jacobi"
    elif default_factor_kind in {"block_schur_lu", "native_block_schur_lu", "symbolic_schur_lu"}:
        default_factor_kind = "symbolic_block_schur_lu"
    elif default_factor_kind in {"frontal_schur_lu", "native_frontal_schur_lu", "multifrontal_schur_lu"}:
        default_factor_kind = "symbolic_frontal_schur_lu"
    elif default_factor_kind in {
        "blr_frontal_schur_lu",
        "native_blr_frontal_schur_lu",
        "compressed_frontal_schur_lu",
    }:
        default_factor_kind = "symbolic_blr_frontal_schur_lu"
    elif default_factor_kind in {
        "nd_frontal_schur_lu",
        "nested_dissection_frontal_schur_lu",
        "native_nd_frontal_schur_lu",
        "multilevel_frontal_schur_lu",
    }:
        default_factor_kind = "symbolic_nd_frontal_schur_lu"
    elif default_factor_kind in {"superblock_lu", "native_superblock_lu", "block_edge_lu"}:
        default_factor_kind = "symbolic_superblock_lu"
    elif default_factor_kind in {"block_lu_coarse", "native_block_lu_coarse", "symbolic_lu_coarse"}:
        default_factor_kind = "symbolic_block_lu_coarse"
    elif default_factor_kind in {"block_lu", "native_block_lu", "symbolic_lu"}:
        default_factor_kind = "symbolic_block_lu"
    explicit_factor_requested = bool(factor_kind_env) or bool(
        os.environ.get("SFINCS_JAX_EXPLICIT_SPARSE_FACTOR_KIND", "").strip()
    )
    monolithic_auto_guard_size = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_MONOLITHIC_AUTO_MAX_SIZE",
        250_000,
        minimum=0,
    )
    if (
        bool(direct_pmat_enabled)
        and not bool(explicit_factor_requested)
        and default_factor_kind in {"lu", "ilu"}
        and int(monolithic_auto_guard_size) > 0
        and int(linear_size) > int(monolithic_auto_guard_size)
    ):
        default_factor_kind = "symbolic_block_lu_coarse"
        if emit is not None:
            emit(
                1,
                "solve_v3_transport_matrix_linear_gmres: large direct-Pmat auto factor switched "
                "from monolithic LU/ILU to symbolic_block_lu_coarse "
                f"(linear_size={int(linear_size)} max_size={int(monolithic_auto_guard_size)})",
            )

    op_pc = _build_transport_preconditioner_operator_fortran_reduced(
        op,
        preconditioner_x=int(preconditioner_x),
        preconditioner_xi=int(preconditioner_xi),
        preconditioner_species=int(preconditioner_species),
        preconditioner_x_min_l=int(preconditioner_x_min_l),
        keep_theta_zeta=bool(keep_theta_zeta),
    )
    cache_key = (
        *transport_precond_cache_key(
            op,
            "fp_fortran_reduced_lu_"
            f"{default_factor_kind}_{factor_dtype.name}_{float(pc_shift):.3e}_"
            f"{int(preconditioner_x)}_{int(preconditioner_xi)}_{int(preconditioner_species)}_"
            f"{int(preconditioner_x_min_l)}_{int(keep_theta_zeta)}_direct{int(direct_pmat_enabled)}_"
            f"symbolic{str(symbolic_ordering)}_{int(symbolic_block_size)}_{int(symbolic_block_overlap)}_"
            f"coarse{int(symbolic_coarse_max_cols)}_probes{int(symbolic_coarse_probe_cols)}_"
            f"damp{float(symbolic_coarse_damping):.3e}_{float(symbolic_coarse_regularization_rel):.3e}_"
            f"phys{int(symbolic_physics_coarse_enabled)}_{int(symbolic_physics_coarse_max_cols)}_"
            f"schur{int(symbolic_schur_max_separator_cols)}_{int(symbolic_schur_boundary_width)}_"
            f"{int(symbolic_schur_high_degree_cols)}_{float(symbolic_schur_regularization_rel):.3e}_"
            f"frontal{int(symbolic_frontal_max_separator_cols)}_{int(symbolic_frontal_boundary_width)}_"
            f"{int(symbolic_frontal_high_degree_cols)}_{int(symbolic_frontal_max_superblock_size)}_"
            f"{int(symbolic_frontal_max_superblock_blocks)}_{int(symbolic_frontal_min_cross_nnz)}_"
            f"{float(symbolic_frontal_min_cross_separator_fraction):.3e}_"
            f"{float(symbolic_frontal_regularization_rel):.3e}_"
            f"{int(symbolic_frontal_max_dense_rhs_entries)}_"
            f"{int(symbolic_frontal_max_dense_rhs_cols_per_block)}_"
            f"blr{float(symbolic_blr_frontal_tol):.3e}_{int(symbolic_blr_frontal_max_rank)}_"
            f"{int(symbolic_blr_frontal_min_cols)}_{float(symbolic_blr_frontal_gmres_rtol):.3e}_"
            f"{float(symbolic_blr_frontal_gmres_atol):.3e}_{int(symbolic_blr_frontal_gmres_maxiter)}_"
            f"{int(symbolic_blr_frontal_gmres_restart)}_"
            f"wb{int(symbolic_blr_frontal_woodbury_max_rank)}_"
            f"{float(symbolic_blr_frontal_woodbury_max_condition):.3e}_"
            f"nd{int(symbolic_nd_max_leaf_size)}_{int(symbolic_nd_max_depth)}_"
            f"terminal{int(symbolic_nd_max_terminal_factor_size)}_"
            f"{int(symbolic_nd_separator_width)}_{int(symbolic_nd_max_separator_cols)}_"
            f"{int(symbolic_nd_high_degree_cols)}_{float(symbolic_nd_regularization_rel):.3e}_"
            f"{int(symbolic_nd_max_dense_rhs_entries)}_"
            f"{int(symbolic_nd_max_dense_rhs_entries_per_child)}_"
            f"{int(symbolic_nd_max_dense_rhs_cols_per_child)}_"
            f"setups{float(symbolic_nd_max_setup_s):.3e}_"
            f"ndblr{int(symbolic_nd_compress_updates)}_upd{int(symbolic_nd_parallel_update_workers)}_"
            f"polish{int(symbolic_nd_residual_polish_steps)}_"
            f"{float(symbolic_nd_residual_polish_damping):.3e}_"
            f"super{int(symbolic_superblock_max_size)}_{int(symbolic_superblock_max_blocks)}_"
            f"{int(symbolic_superblock_min_cross_nnz)}_"
            f"{float(symbolic_superblock_min_retained_cross_fraction):.3e}_"
            f"{float(symbolic_superblock_regularization_rel):.3e}_"
            f"symworkers{int(symbolic_numeric_parallel_workers)}_"
            f"{int(symbolic_max_permutation_size)}_"
            f"adm{int(symbolic_admission_enabled)}_{float(symbolic_admission_max_rel):.3e}_"
            f"{float(symbolic_admission_min_improvement):.3e}_{int(symbolic_admission_probes)}_"
            f"rescue{int(symbolic_admission_rescue_lu)}_{float(symbolic_admission_rescue_lu_max_mb):.3e}_"
            f"autoexact{int(auto_exact_rescue_enabled)}_{float(auto_exact_rescue_max_mb):.3e}_"
            f"{float(auto_exact_rescue_ram_fraction):.3e}_{int(auto_exact_rescue_max_size)}_"
            f"{int(auto_exact_rescue_max_factor_entries)}_"
            f"directadm{int(direct_admission_enabled)}_{int(direct_admission_explicit_enabled)}_"
            f"maxfactor{float(max_factor_mb):.3e}",
        ),
        str(active_hash),
        int(linear_size),
    )
    cached = _TRANSPORT_FP_FORTRAN_REDUCED_LU_PRECOND_CACHE.get(cache_key)
    if cached is None:
        try:
            direct_operator_bundle: SparseOperatorBundle | None = None
            direct_metadata: dict[str, object] = {}
            if bool(direct_pmat_enabled):
                direct_result = _try_build_rhsmode23_fp_fortran_reduced_direct_pmat_bundle(
                    op_pc=op_pc,
                    active_indices=active_indices_use,
                    factor_dtype=factor_dtype,
                    pc_shift=float(pc_shift),
                    emit=emit,
                )
                if direct_result is not None:
                    direct_operator_bundle, direct_metadata = direct_result

            def _expand(x: jnp.ndarray) -> jnp.ndarray:
                if expand_reduced is None:
                    return x
                return expand_reduced(x)

            def _reduce(y: jnp.ndarray) -> jnp.ndarray:
                if reduce_full is None:
                    return y
                return reduce_full(y)

            def _pc_matvec(x_vec: jnp.ndarray) -> jnp.ndarray:
                x_full = _expand(jnp.asarray(x_vec, dtype=jnp.float64))
                y_full = apply_v3_full_system_operator_cached(op_pc, x_full)
                if float(pc_shift) != 0.0:
                    y_full = y_full + jnp.asarray(float(pc_shift), dtype=jnp.float64) * x_full
                return _reduce(y_full)

            _operator_bundle = None
            factor_bundle = None
            factor_kind_for_build = str(default_factor_kind)
            effective_factor_max_mb = float(max_factor_mb)
            auto_exact_rescue_selected = False
            if direct_operator_bundle is not None:
                direct_csr_nbytes = int(direct_metadata.get("direct_pmat_csr_nbytes_estimate", 0) or 0)
                direct_symbolic_prefill_safety = _float_env(
                    "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_PREFILL_SAFETY_FACTOR",
                    64.0,
                    minimum=1.0,
                )
                direct_symbolic_prefill_estimate = (
                    int(np.ceil(float(direct_csr_nbytes) * float(direct_symbolic_prefill_safety)))
                    if direct_csr_nbytes > 0
                    and factor_kind_for_build
                    in {
                        "symbolic_block_lu",
                        "symbolic_block_lu_coarse",
                        "symbolic_block_schur_lu",
                        "symbolic_frontal_schur_lu",
                        "symbolic_blr_frontal_schur_lu",
                        "symbolic_nd_frontal_schur_lu",
                        "symbolic_superblock_lu",
                    }
                    else 0
                )
                direct_pmat_nnz = int(direct_metadata.get("direct_pmat_nnz", 0) or 0)
                direct_mf_fill_ratio = _float_env(
                    "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_MULTIFRONTAL_FILL_RATIO",
                    104.0,
                    minimum=1.0,
                )
                direct_mf_overhead = _float_env(
                    "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_MULTIFRONTAL_OVERHEAD",
                    1.15,
                    minimum=1.0,
                )
                direct_multifrontal_entries_estimate = (
                    int(np.ceil(float(direct_pmat_nnz) * float(direct_mf_fill_ratio)))
                    if direct_pmat_nnz > 0
                    else 0
                )
                direct_multifrontal_nbytes_estimate = (
                    estimate_multifrontal_direct_lu_nbytes(
                        direct_pmat_nnz,
                        fill_ratio=float(direct_mf_fill_ratio),
                        data_dtype=factor_dtype,
                        overhead=float(direct_mf_overhead),
                    )
                    if direct_pmat_nnz > 0
                    else 0
                )
                direct_metadata.update(
                    {
                        "direct_pmat_multifrontal_fill_ratio_estimate": float(direct_mf_fill_ratio),
                        "direct_pmat_multifrontal_overhead_estimate": float(direct_mf_overhead),
                        "direct_pmat_multifrontal_factor_entries_estimate": int(
                            direct_multifrontal_entries_estimate
                        ),
                        "direct_pmat_multifrontal_factor_nbytes_estimate": int(
                            direct_multifrontal_nbytes_estimate
                        ),
                    }
                )
                max_factor_nbytes = int(float(effective_factor_max_mb) * 1.0e6)
                if emit is not None and direct_multifrontal_nbytes_estimate > 0:
                    emit(
                        1,
                        "solve_v3_transport_matrix_linear_gmres: direct reduced Pmat "
                        "MUMPS-like fill estimate "
                        f"nnz={int(direct_pmat_nnz)} "
                        f"fill_ratio={float(direct_mf_fill_ratio):.3g} "
                        f"factor_mb={float(direct_multifrontal_nbytes_estimate) / 1.0e6:.3f} "
                        f"max_mb={float(effective_factor_max_mb):.3f}",
                    )
                if (
                    factor_kind_for_build in {"lu", "ilu"}
                    and direct_multifrontal_nbytes_estimate > 0
                    and max_factor_nbytes > 0
                    and direct_multifrontal_nbytes_estimate > max_factor_nbytes
                ):
                    if emit is not None:
                        emit(
                            1,
                            "solve_v3_transport_matrix_linear_gmres: direct reduced Pmat exact factor "
                            "rejected by MUMPS-like fill guard "
                            f"factor_kind={factor_kind_for_build} "
                            f"factor_mb={float(direct_multifrontal_nbytes_estimate) / 1.0e6:.3f} "
                            f"max_mb={float(effective_factor_max_mb):.3f}",
                        )
                    return fallback_builder(
                        op=op,
                        reduce_full=reduce_full,
                        expand_reduced=expand_reduced,
                    )
                if (
                    direct_symbolic_prefill_estimate > 0
                    and max_factor_nbytes > 0
                    and direct_symbolic_prefill_estimate > max_factor_nbytes
                ):
                    auto_exact_cap_nbytes = int(float(auto_exact_rescue_max_mb) * 1.0e6)
                    auto_exact_candidate_ok = (
                        bool(auto_exact_rescue_enabled)
                        and not bool(explicit_factor_requested)
                        and factor_kind_for_build == "symbolic_block_lu_coarse"
                        and (
                            int(auto_exact_rescue_max_size) <= 0
                            or int(linear_size) <= int(auto_exact_rescue_max_size)
                        )
                        and (
                            int(auto_exact_rescue_max_factor_entries) <= 0
                            or int(direct_multifrontal_entries_estimate)
                            <= int(auto_exact_rescue_max_factor_entries)
                        )
                        and direct_multifrontal_nbytes_estimate > 0
                        and auto_exact_cap_nbytes > 0
                        and direct_multifrontal_nbytes_estimate <= auto_exact_cap_nbytes
                    )
                    if auto_exact_candidate_ok:
                        factor_kind_for_build = "lu"
                        effective_factor_max_mb = max(float(max_factor_mb), float(auto_exact_rescue_max_mb))
                        max_factor_nbytes = int(float(effective_factor_max_mb) * 1.0e6)
                        auto_exact_rescue_selected = True
                        direct_metadata.update(
                            {
                                "direct_pmat_auto_exact_rescue_selected": True,
                                "direct_pmat_auto_exact_rescue_reason": "symbolic_prefill_guard",
                                "direct_pmat_auto_exact_rescue_max_mb": float(auto_exact_rescue_max_mb),
                            }
                        )
                        if emit is not None:
                            emit(
                                1,
                                "solve_v3_transport_matrix_linear_gmres: direct reduced Pmat symbolic factor "
                                "prefill exceeds default budget; trying exact LU rescue "
                                f"prefill_mb={float(direct_symbolic_prefill_estimate) / 1.0e6:.3f} "
                                f"exact_factor_mb={float(direct_multifrontal_nbytes_estimate) / 1.0e6:.3f} "
                                f"rescue_max_mb={float(auto_exact_rescue_max_mb):.3f}",
                            )
                    else:
                        if emit is not None:
                            emit(
                                1,
                            "solve_v3_transport_matrix_linear_gmres: direct reduced Pmat symbolic factor "
                            "rejected by prefill guard "
                            f"factor_kind={factor_kind_for_build} "
                            f"prefill_mb={float(direct_symbolic_prefill_estimate) / 1.0e6:.3f} "
                            f"max_mb={float(effective_factor_max_mb):.3f} "
                            f"exact_entries={int(direct_multifrontal_entries_estimate)} "
                            f"exact_entries_cap={int(auto_exact_rescue_max_factor_entries)} "
                            f"safety={float(direct_symbolic_prefill_safety):.3g}",
                        )
                        return fallback_builder(
                            op=op,
                            reduce_full=reduce_full,
                            expand_reduced=expand_reduced,
                        )
                try:
                    _operator_bundle, factor_bundle = build_host_sparse_direct_factor_from_matvec(
                        matvec=_pc_matvec,
                        n=int(linear_size),
                        dtype=jnp.float64,
                        factor_dtype=factor_dtype,
                        pattern=None,
                        operator_bundle_override=direct_operator_bundle,
                        emit=emit,
                        default_factor_kind=str(factor_kind_for_build),
                        default_ilu_fill_factor=4.0,
                        default_ilu_drop_tol=1.0e-4,
                        default_permc_spec="MMD_AT_PLUS_A",
                        default_diag_pivot_thresh=0.0,
                        default_pattern_color_batch=8,
                        default_symbolic_ordering_kind=str(symbolic_ordering),
                        default_symbolic_block_size=int(symbolic_block_size),
                        default_symbolic_block_overlap=int(symbolic_block_overlap),
                        default_symbolic_coarse_max_cols=int(symbolic_coarse_max_cols),
                        default_symbolic_coarse_probe_cols=int(symbolic_coarse_probe_cols),
                        default_symbolic_coarse_damping=float(symbolic_coarse_damping),
                        default_symbolic_coarse_regularization_rel=float(symbolic_coarse_regularization_rel),
                        default_symbolic_schur_max_separator_cols=int(symbolic_schur_max_separator_cols),
                        default_symbolic_schur_tail_size=int(direct_metadata.get("direct_pmat_tail_size", 0)),
                        default_symbolic_schur_boundary_width=int(symbolic_schur_boundary_width),
                        default_symbolic_schur_high_degree_cols=int(symbolic_schur_high_degree_cols),
                        default_symbolic_schur_regularization_rel=float(symbolic_schur_regularization_rel),
                        default_symbolic_frontal_max_separator_cols=int(symbolic_frontal_max_separator_cols),
                        default_symbolic_frontal_tail_size=int(direct_metadata.get("direct_pmat_tail_size", 0)),
                        default_symbolic_frontal_boundary_width=int(symbolic_frontal_boundary_width),
                        default_symbolic_frontal_high_degree_cols=int(symbolic_frontal_high_degree_cols),
                        default_symbolic_frontal_max_superblock_size=int(symbolic_frontal_max_superblock_size),
                        default_symbolic_frontal_max_superblock_blocks=int(symbolic_frontal_max_superblock_blocks),
                        default_symbolic_frontal_min_cross_nnz=int(symbolic_frontal_min_cross_nnz),
                        default_symbolic_frontal_min_cross_separator_fraction=float(
                            symbolic_frontal_min_cross_separator_fraction
                        ),
                        default_symbolic_frontal_regularization_rel=float(symbolic_frontal_regularization_rel),
                        default_symbolic_frontal_max_dense_rhs_entries=int(symbolic_frontal_max_dense_rhs_entries),
                        default_symbolic_frontal_max_dense_rhs_cols_per_block=int(
                            symbolic_frontal_max_dense_rhs_cols_per_block
                        ),
                        default_symbolic_blr_frontal_tol=float(symbolic_blr_frontal_tol),
                        default_symbolic_blr_frontal_max_rank=int(symbolic_blr_frontal_max_rank),
                        default_symbolic_blr_frontal_min_cols=int(symbolic_blr_frontal_min_cols),
                        default_symbolic_blr_frontal_gmres_rtol=float(symbolic_blr_frontal_gmres_rtol),
                        default_symbolic_blr_frontal_gmres_atol=float(symbolic_blr_frontal_gmres_atol),
                        default_symbolic_blr_frontal_gmres_maxiter=int(symbolic_blr_frontal_gmres_maxiter),
                        default_symbolic_blr_frontal_gmres_restart=int(symbolic_blr_frontal_gmres_restart),
                        default_symbolic_blr_frontal_woodbury_max_rank=int(
                            symbolic_blr_frontal_woodbury_max_rank
                        ),
                        default_symbolic_blr_frontal_woodbury_max_condition=float(
                            symbolic_blr_frontal_woodbury_max_condition
                        ),
                        default_symbolic_nd_max_leaf_size=int(symbolic_nd_max_leaf_size),
                        default_symbolic_nd_max_terminal_factor_size=int(symbolic_nd_max_terminal_factor_size),
                        default_symbolic_nd_max_depth=int(symbolic_nd_max_depth),
                        default_symbolic_nd_separator_width=int(symbolic_nd_separator_width),
                        default_symbolic_nd_max_separator_cols=int(symbolic_nd_max_separator_cols),
                        default_symbolic_nd_high_degree_cols=int(symbolic_nd_high_degree_cols),
                        default_symbolic_nd_regularization_rel=float(symbolic_nd_regularization_rel),
                        default_symbolic_nd_max_dense_rhs_entries=int(symbolic_nd_max_dense_rhs_entries),
                        default_symbolic_nd_max_dense_rhs_entries_per_child=int(
                            symbolic_nd_max_dense_rhs_entries_per_child
                        ),
                        default_symbolic_nd_max_dense_rhs_cols_per_child=int(
                            symbolic_nd_max_dense_rhs_cols_per_child
                        ),
                        default_symbolic_nd_max_setup_s=float(symbolic_nd_max_setup_s),
                        default_symbolic_nd_compress_updates=bool(symbolic_nd_compress_updates),
                        default_symbolic_nd_parallel_update_workers=int(symbolic_nd_parallel_update_workers),
                        default_symbolic_nd_residual_polish_steps=int(symbolic_nd_residual_polish_steps),
                        default_symbolic_nd_residual_polish_damping=float(symbolic_nd_residual_polish_damping),
                        default_symbolic_superblock_max_size=int(symbolic_superblock_max_size),
                        default_symbolic_superblock_max_blocks=int(symbolic_superblock_max_blocks),
                        default_symbolic_superblock_min_cross_nnz=int(symbolic_superblock_min_cross_nnz),
                        default_symbolic_superblock_min_retained_cross_fraction=float(
                            symbolic_superblock_min_retained_cross_fraction
                        ),
                        default_symbolic_superblock_regularization_rel=float(symbolic_superblock_regularization_rel),
                        default_symbolic_numeric_parallel_workers=int(symbolic_numeric_parallel_workers),
                        default_symbolic_max_permutation_size=int(symbolic_max_permutation_size),
                        default_monolithic_guard_enabled=not bool(auto_exact_rescue_selected),
                    )
                except Exception as exc:  # noqa: BLE001
                    exc_text = str(exc)
                    if bool(auto_exact_rescue_selected):
                        if emit is not None:
                            emit(
                                1,
                                "solve_v3_transport_matrix_linear_gmres: direct reduced Pmat exact LU rescue "
                                f"failed; skipping pattern-probe fallback ({type(exc).__name__}: {exc})",
                            )
                        return fallback_builder(
                            op=op,
                            reduce_full=reduce_full,
                            expand_reduced=expand_reduced,
                        )
                    if factor_kind_for_build == "symbolic_nd_frontal_schur_lu" and (
                        "symbolic_nd_frontal_schur_lu setup time budget exceeded" in exc_text
                        or "symbolic_nd_frontal_schur_lu terminal leaf factor size exceeded" in exc_text
                    ):
                        if emit is not None:
                            emit(
                                1,
                                "solve_v3_transport_matrix_linear_gmres: direct reduced Pmat ND factor "
                                "rejected by setup guard; skipping pattern-probe fallback "
                                f"({type(exc).__name__}: {exc})",
                            )
                        return fallback_builder(
                            op=op,
                            reduce_full=reduce_full,
                            expand_reduced=expand_reduced,
                        )
                    if emit is not None:
                        emit(
                            1,
                            "solve_v3_transport_matrix_linear_gmres: direct reduced Pmat factorization failed; "
                            f"falling back to pattern probe ({type(exc).__name__}: {exc})",
                        )
                    direct_operator_bundle = None
                    direct_metadata = {}

            if factor_bundle is None:
                if active_indices_use is None:
                    pattern = v3_full_system_fortran_reduced_preconditioner_sparsity_pattern(
                        op_pc,
                        preconditioner_x=int(preconditioner_x),
                        preconditioner_xi=int(preconditioner_xi),
                        preconditioner_species=int(preconditioner_species),
                        preconditioner_x_min_l=int(preconditioner_x_min_l),
                    )
                else:
                    pattern = v3_full_system_fortran_reduced_preconditioner_sparsity_pattern_for_indices(
                        op_pc,
                        active_indices_use,
                        preconditioner_x=int(preconditioner_x),
                        preconditioner_xi=int(preconditioner_xi),
                        preconditioner_species=int(preconditioner_species),
                        preconditioner_x_min_l=int(preconditioner_x_min_l),
                    )
                if emit is not None:
                    summary = summarize_v3_sparse_pattern(op_pc, pattern)
                    emit(
                        1,
                        "solve_v3_transport_matrix_linear_gmres: fp_fortran_reduced_lu Pmat pattern "
                        f"scope={'active' if active_indices_use is not None else 'full'} "
                        f"shape={summary.shape} nnz={summary.nnz} avg_row_nnz={summary.avg_row_nnz:.3g}",
                    )

                _operator_bundle, factor_bundle = build_host_sparse_direct_factor_from_matvec(
                    matvec=_pc_matvec,
                    n=int(linear_size),
                    dtype=jnp.float64,
                    factor_dtype=factor_dtype,
                    pattern=pattern,
                    operator_bundle_override=None,
                    emit=emit,
                    default_factor_kind=str(default_factor_kind),
                    default_ilu_fill_factor=4.0,
                    default_ilu_drop_tol=1.0e-4,
                    default_permc_spec="MMD_AT_PLUS_A",
                    default_diag_pivot_thresh=0.0,
                    default_pattern_color_batch=8,
                    default_symbolic_ordering_kind=str(symbolic_ordering),
                    default_symbolic_block_size=int(symbolic_block_size),
                    default_symbolic_block_overlap=int(symbolic_block_overlap),
                    default_symbolic_coarse_max_cols=int(symbolic_coarse_max_cols),
                    default_symbolic_coarse_probe_cols=int(symbolic_coarse_probe_cols),
                    default_symbolic_coarse_damping=float(symbolic_coarse_damping),
                    default_symbolic_coarse_regularization_rel=float(symbolic_coarse_regularization_rel),
                    default_symbolic_schur_max_separator_cols=int(symbolic_schur_max_separator_cols),
                    default_symbolic_schur_tail_size=0,
                    default_symbolic_schur_boundary_width=int(symbolic_schur_boundary_width),
                    default_symbolic_schur_high_degree_cols=int(symbolic_schur_high_degree_cols),
                    default_symbolic_schur_regularization_rel=float(symbolic_schur_regularization_rel),
                    default_symbolic_frontal_max_separator_cols=int(symbolic_frontal_max_separator_cols),
                    default_symbolic_frontal_tail_size=0,
                    default_symbolic_frontal_boundary_width=int(symbolic_frontal_boundary_width),
                    default_symbolic_frontal_high_degree_cols=int(symbolic_frontal_high_degree_cols),
                    default_symbolic_frontal_max_superblock_size=int(symbolic_frontal_max_superblock_size),
                    default_symbolic_frontal_max_superblock_blocks=int(symbolic_frontal_max_superblock_blocks),
                    default_symbolic_frontal_min_cross_nnz=int(symbolic_frontal_min_cross_nnz),
                    default_symbolic_frontal_min_cross_separator_fraction=float(
                        symbolic_frontal_min_cross_separator_fraction
                    ),
                    default_symbolic_frontal_regularization_rel=float(symbolic_frontal_regularization_rel),
                    default_symbolic_frontal_max_dense_rhs_entries=int(symbolic_frontal_max_dense_rhs_entries),
                    default_symbolic_frontal_max_dense_rhs_cols_per_block=int(
                        symbolic_frontal_max_dense_rhs_cols_per_block
                    ),
                    default_symbolic_blr_frontal_tol=float(symbolic_blr_frontal_tol),
                    default_symbolic_blr_frontal_max_rank=int(symbolic_blr_frontal_max_rank),
                    default_symbolic_blr_frontal_min_cols=int(symbolic_blr_frontal_min_cols),
                    default_symbolic_blr_frontal_gmres_rtol=float(symbolic_blr_frontal_gmres_rtol),
                    default_symbolic_blr_frontal_gmres_atol=float(symbolic_blr_frontal_gmres_atol),
                    default_symbolic_blr_frontal_gmres_maxiter=int(symbolic_blr_frontal_gmres_maxiter),
                    default_symbolic_blr_frontal_gmres_restart=int(symbolic_blr_frontal_gmres_restart),
                    default_symbolic_blr_frontal_woodbury_max_rank=int(symbolic_blr_frontal_woodbury_max_rank),
                    default_symbolic_blr_frontal_woodbury_max_condition=float(
                        symbolic_blr_frontal_woodbury_max_condition
                    ),
                    default_symbolic_nd_max_leaf_size=int(symbolic_nd_max_leaf_size),
                    default_symbolic_nd_max_terminal_factor_size=int(symbolic_nd_max_terminal_factor_size),
                    default_symbolic_nd_max_depth=int(symbolic_nd_max_depth),
                    default_symbolic_nd_separator_width=int(symbolic_nd_separator_width),
                    default_symbolic_nd_max_separator_cols=int(symbolic_nd_max_separator_cols),
                    default_symbolic_nd_high_degree_cols=int(symbolic_nd_high_degree_cols),
                    default_symbolic_nd_regularization_rel=float(symbolic_nd_regularization_rel),
                    default_symbolic_nd_max_dense_rhs_entries=int(symbolic_nd_max_dense_rhs_entries),
                    default_symbolic_nd_max_dense_rhs_entries_per_child=int(
                        symbolic_nd_max_dense_rhs_entries_per_child
                    ),
                    default_symbolic_nd_max_dense_rhs_cols_per_child=int(symbolic_nd_max_dense_rhs_cols_per_child),
                    default_symbolic_nd_max_setup_s=float(symbolic_nd_max_setup_s),
                    default_symbolic_nd_compress_updates=bool(symbolic_nd_compress_updates),
                    default_symbolic_nd_parallel_update_workers=int(symbolic_nd_parallel_update_workers),
                    default_symbolic_nd_residual_polish_steps=int(symbolic_nd_residual_polish_steps),
                    default_symbolic_nd_residual_polish_damping=float(symbolic_nd_residual_polish_damping),
                    default_symbolic_superblock_max_size=int(symbolic_superblock_max_size),
                    default_symbolic_superblock_max_blocks=int(symbolic_superblock_max_blocks),
                    default_symbolic_superblock_min_cross_nnz=int(symbolic_superblock_min_cross_nnz),
                    default_symbolic_superblock_min_retained_cross_fraction=float(
                        symbolic_superblock_min_retained_cross_fraction
                    ),
                    default_symbolic_superblock_regularization_rel=float(symbolic_superblock_regularization_rel),
                    default_symbolic_numeric_parallel_workers=int(symbolic_numeric_parallel_workers),
                    default_symbolic_max_permutation_size=int(symbolic_max_permutation_size),
                )
        except Exception as exc:  # noqa: BLE001
            if emit is not None:
                emit(
                    1,
                    "solve_v3_transport_matrix_linear_gmres: fp_fortran_reduced_lu disabled after "
                    f"{type(exc).__name__}: {exc}",
                )
            return fallback_builder(
                op=op,
                reduce_full=reduce_full,
                expand_reduced=expand_reduced,
            )

        physics_coarse_metadata: dict[str, object] = {}
        if (
            str(getattr(factor_bundle, "kind", "")) == "symbolic_block_lu_coarse"
            and bool(symbolic_physics_coarse_enabled)
            and active_indices_use is not None
            and direct_operator_bundle is not None
        ):
            try:
                physics_basis, physics_names = _build_rhsmode23_direct_pmat_physics_coarse_basis(
                    op=op_pc,
                    active_indices=active_indices_use,
                    max_cols=int(symbolic_physics_coarse_max_cols),
                    base_factor_bundle=factor_bundle,
                )
                if physics_basis is not None and int(getattr(physics_basis, "shape", (0, 0))[1]) > 0:
                    factor_bundle = wrap_sparse_factor_with_coarse_correction(
                        factor_bundle,
                        physics_basis,
                        damping=float(symbolic_coarse_damping),
                        regularization_rel=float(symbolic_coarse_regularization_rel),
                    )
                    physics_coarse_metadata = {
                        "symbolic_physics_coarse": True,
                        "symbolic_physics_coarse_cols": int(physics_basis.shape[1]),
                        "symbolic_physics_coarse_nnz": int(physics_basis.nnz),
                        "symbolic_physics_coarse_labels": tuple(str(v) for v in physics_names),
                    }
                    if emit is not None:
                        emit(
                            1,
                            "solve_v3_transport_matrix_linear_gmres: fp_fortran_reduced_lu "
                            f"physics coarse basis cols={int(physics_basis.shape[1])} "
                            f"nnz={int(physics_basis.nnz)}",
                        )
            except Exception as exc:  # noqa: BLE001
                physics_coarse_metadata = {
                    "symbolic_physics_coarse": False,
                    "symbolic_physics_coarse_error": f"{type(exc).__name__}: {exc}",
                }
                if emit is not None:
                    emit(
                        1,
                        "solve_v3_transport_matrix_linear_gmres: fp_fortran_reduced_lu "
                        f"physics coarse disabled ({type(exc).__name__}: {exc})",
                    )

        factor_nbytes = getattr(factor_bundle, "factor_nbytes_estimate", None)
        if (
            float(effective_factor_max_mb) > 0.0
            and factor_nbytes is not None
            and int(factor_nbytes) > int(float(effective_factor_max_mb) * 1.0e6)
        ):
            if emit is not None:
                emit(
                    1,
                    "solve_v3_transport_matrix_linear_gmres: fp_fortran_reduced_lu factor rejected by budget "
                    f"factor_mb={float(factor_nbytes) / 1.0e6:.3f} max_mb={float(effective_factor_max_mb):.3f}",
                )
            return fallback_builder(
                op=op,
                reduce_full=reduce_full,
                expand_reduced=expand_reduced,
            )
        symbolic_metadata: dict[str, object] = {}
        factor_operator = getattr(factor_bundle, "operator", None)
        factor_matrix = None if factor_operator is None else getattr(factor_operator, "matrix", None)
        factor_kind_for_admission = str(getattr(factor_bundle, "kind", ""))
        direct_admission_required = (
            bool(direct_admission_enabled)
            and factor_kind_for_admission in {"lu", "ilu"}
            and (bool(auto_exact_rescue_selected) or bool(direct_admission_explicit_enabled))
        )
        if bool(direct_admission_required):
            direct_admission = admit_sparse_factor_against_operator(
                factor_operator if factor_operator is not None else factor_matrix,
                factor_bundle,
                probe_count=int(symbolic_admission_probes),
                max_relative_residual=float(symbolic_admission_max_rel),
                min_improvement_vs_identity=float(symbolic_admission_min_improvement),
            )
            symbolic_metadata["direct_admission"] = direct_admission.to_dict()
            if emit is not None:
                emit(
                    1,
                    "solve_v3_transport_matrix_linear_gmres: fp_fortran_reduced_lu exact factor admission "
                    f"{'accepted' if direct_admission.accepted else 'rejected'} "
                    f"max_rel={float(direct_admission.max_relative_residual):.3e} "
                    f"median_rel={float(direct_admission.median_relative_residual):.3e} "
                    f"min_improvement={float(direct_admission.min_improvement_vs_identity):.3e} "
                    f"probes={int(direct_admission.probe_count)}",
                )
            if not bool(direct_admission.accepted):
                return fallback_builder(
                    op=op,
                    reduce_full=reduce_full,
                    expand_reduced=expand_reduced,
                )
        if factor_kind_for_admission in {
            "symbolic_block_lu",
            "symbolic_block_lu_coarse",
            "symbolic_block_schur_lu",
            "symbolic_frontal_schur_lu",
            "symbolic_blr_frontal_schur_lu",
            "symbolic_nd_frontal_schur_lu",
            "symbolic_superblock_lu",
        } and bool(symbolic_admission_enabled):
            admission = admit_sparse_factor_against_operator(
                factor_operator if factor_operator is not None else factor_matrix,
                factor_bundle,
                probe_count=int(symbolic_admission_probes),
                max_relative_residual=float(symbolic_admission_max_rel),
                min_improvement_vs_identity=float(symbolic_admission_min_improvement),
            )
            admission_metadata = admission.to_dict()
            symbolic_metadata["symbolic_admission"] = admission_metadata
            if emit is not None:
                admission_label = factor_kind_for_admission
                emit(
                    1,
                    f"solve_v3_transport_matrix_linear_gmres: fp_fortran_reduced_lu {admission_label} admission "
                    f"{'accepted' if admission.accepted else 'rejected'} "
                    f"max_rel={float(admission.max_relative_residual):.3e} "
                    f"median_rel={float(admission.median_relative_residual):.3e} "
                    f"min_improvement={float(admission.min_improvement_vs_identity):.3e} "
                    f"probes={int(admission.probe_count)}",
                )
            if not bool(admission.accepted):
                rescue_bundle = None
                rescue_metadata: dict[str, object] = {}
                if (
                    bool(symbolic_admission_rescue_lu)
                    and factor_operator is not None
                    and getattr(factor_operator, "matrix", None) is not None
                ):
                    try:
                        rescue_candidate = factorize_host_sparse_operator(
                            factor_operator,
                            kind="lu",
                            permc_spec="MMD_AT_PLUS_A",
                            diag_pivot_thresh=0.0,
                        )
                        rescue_nbytes = getattr(rescue_candidate, "factor_nbytes_estimate", None)
                        rescue_budget_ok = (
                            float(symbolic_admission_rescue_lu_max_mb) <= 0.0
                            or rescue_nbytes is None
                            or int(rescue_nbytes) <= int(float(symbolic_admission_rescue_lu_max_mb) * 1.0e6)
                        )
                        if rescue_budget_ok:
                            rescue_admission = admit_sparse_factor_against_operator(
                                factor_operator,
                                rescue_candidate,
                                probe_count=int(symbolic_admission_probes),
                                max_relative_residual=float(symbolic_admission_max_rel),
                                min_improvement_vs_identity=float(symbolic_admission_min_improvement),
                            )
                            rescue_metadata = {
                                "symbolic_admission_rescue_lu": True,
                                "symbolic_admission_rescue_lu_factor_nbytes_estimate": (
                                    None if rescue_nbytes is None else int(rescue_nbytes)
                                ),
                                "symbolic_admission_rescue_lu_factor_nnz_estimate": (
                                    None
                                    if getattr(rescue_candidate, "factor_nnz_estimate", None) is None
                                    else int(rescue_candidate.factor_nnz_estimate)
                                ),
                                "symbolic_admission_rescue_lu_factor_s": (
                                    None
                                    if getattr(rescue_candidate, "factor_s", None) is None
                                    else float(rescue_candidate.factor_s)
                                ),
                                "symbolic_admission_rescue_lu_admission": rescue_admission.to_dict(),
                            }
                            if emit is not None:
                                emit(
                                    1,
                                    "solve_v3_transport_matrix_linear_gmres: fp_fortran_reduced_lu "
                                    "symbolic admission rescue lu "
                                    f"{'accepted' if rescue_admission.accepted else 'rejected'} "
                                    f"max_rel={float(rescue_admission.max_relative_residual):.3e} "
                                    f"factor_mb={float(rescue_nbytes or 0) / 1.0e6:.3f}",
                                )
                            if bool(rescue_admission.accepted):
                                rescue_bundle = rescue_candidate
                        else:
                            rescue_metadata = {
                                "symbolic_admission_rescue_lu": False,
                                "symbolic_admission_rescue_lu_reason": "factor_budget",
                                "symbolic_admission_rescue_lu_factor_nbytes_estimate": (
                                    None if rescue_nbytes is None else int(rescue_nbytes)
                                ),
                                "symbolic_admission_rescue_lu_max_mb": float(symbolic_admission_rescue_lu_max_mb),
                            }
                    except Exception as exc:  # noqa: BLE001
                        rescue_metadata = {
                            "symbolic_admission_rescue_lu": False,
                            "symbolic_admission_rescue_lu_error": f"{type(exc).__name__}: {exc}",
                        }
                        if emit is not None:
                            emit(
                                1,
                                "solve_v3_transport_matrix_linear_gmres: fp_fortran_reduced_lu "
                                f"symbolic admission rescue lu failed ({type(exc).__name__}: {exc})",
                            )
                symbolic_metadata.update(rescue_metadata)
                if rescue_bundle is None:
                    return fallback_builder(
                        op=op,
                        reduce_full=reduce_full,
                        expand_reduced=expand_reduced,
                    )
                factor_bundle = rescue_bundle
                factor_nbytes = getattr(factor_bundle, "factor_nbytes_estimate", None)
                factor_kind_for_admission = str(getattr(factor_bundle, "kind", ""))
        inner_factor_metadata = getattr(getattr(factor_bundle, "factor", None), "metadata", None)
        if isinstance(inner_factor_metadata, dict):
            symbolic_metadata["symbolic_factor_metadata"] = dict(inner_factor_metadata)
        if factor_matrix is not None:
            try:
                symbolic_analysis = getattr(getattr(factor_bundle, "factor", None), "analysis", None)
                if symbolic_analysis is None:
                    symbolic_analysis = analyze_sparse_symbolic_structure(
                        factor_matrix,
                        ordering_kind=str(symbolic_ordering),
                        block_size_target=int(symbolic_block_size),
                        max_permutation_size=int(symbolic_max_permutation_size),
                    )
                symbolic_metadata.update({
                    "symbolic": symbolic_analysis.to_dict(),
                    "symbolic_cache_key": symbolic_analysis.cache_key(),
                    "symbolic_factor_coarse_size": int(getattr(getattr(factor_bundle, "factor", None), "coarse_size", 0)),
                    "symbolic_factor_overlap_size": int(getattr(getattr(factor_bundle, "factor", None), "overlap_size", 0)),
                })
                if emit is not None:
                    emit(
                        1,
                        "solve_v3_transport_matrix_linear_gmres: fp_fortran_reduced_lu symbolic analysis "
                        f"ordering={symbolic_analysis.ordering_kind} "
                        f"pattern_hash={symbolic_analysis.pattern_hash[:12]} "
                        f"nnz={int(symbolic_analysis.nnz)} "
                        f"bandwidth={int(symbolic_analysis.bandwidth)}->{int(symbolic_analysis.permuted_bandwidth)} "
                        f"profile={int(symbolic_analysis.profile)}->{int(symbolic_analysis.permuted_profile)} "
                        f"blocks={int(symbolic_analysis.block_count)}x<= {int(symbolic_analysis.block_size_max)}",
                    )
            except Exception as exc:  # noqa: BLE001
                symbolic_metadata = {
                    "symbolic_error": f"{type(exc).__name__}: {exc}",
                    "symbolic_ordering": str(symbolic_ordering),
                    "symbolic_block_size": int(symbolic_block_size),
                    "symbolic_max_permutation_size": int(symbolic_max_permutation_size),
                    "symbolic_block_overlap": int(symbolic_block_overlap),
                    "symbolic_coarse_max_cols": int(symbolic_coarse_max_cols),
                    "symbolic_coarse_probe_cols": int(symbolic_coarse_probe_cols),
                    "symbolic_coarse_damping": float(symbolic_coarse_damping),
                    "symbolic_coarse_regularization_rel": float(symbolic_coarse_regularization_rel),
                }

        metadata = {
            "factor_kind": str(factor_bundle.kind),
            "factor_dtype": str(factor_dtype.name),
            "factor_nbytes_estimate": None if factor_nbytes is None else int(factor_nbytes),
            "factor_nnz_estimate": None
            if getattr(factor_bundle, "factor_nnz_estimate", None) is None
            else int(factor_bundle.factor_nnz_estimate),
            "factor_s": None if getattr(factor_bundle, "factor_s", None) is None else float(factor_bundle.factor_s),
            "linear_size": int(linear_size),
            "active_dof": bool(active_indices_use is not None),
            "preconditioner_x": int(preconditioner_x),
            "preconditioner_xi": int(preconditioner_xi),
            "preconditioner_species": int(preconditioner_species),
            "preconditioner_x_min_l": int(preconditioner_x_min_l),
            "keeps_theta_zeta": bool(keep_theta_zeta),
            "shift": float(pc_shift),
            "direct_pmat_enabled": bool(direct_pmat_enabled),
            "factor_max_mb": float(max_factor_mb),
            "effective_factor_max_mb": float(effective_factor_max_mb),
            "host_memory_mb": None if host_memory_mb is None else float(host_memory_mb),
            "auto_exact_rescue_enabled": bool(auto_exact_rescue_enabled),
            "auto_exact_rescue_ram_fraction": float(auto_exact_rescue_ram_fraction),
            "auto_exact_rescue_max_mb": float(auto_exact_rescue_max_mb),
            "auto_exact_rescue_max_size": int(auto_exact_rescue_max_size),
            "auto_exact_rescue_max_factor_entries": int(auto_exact_rescue_max_factor_entries),
            "auto_exact_rescue_selected": bool(auto_exact_rescue_selected),
            "direct_admission_enabled": bool(direct_admission_enabled),
            "direct_admission_explicit_enabled": bool(direct_admission_explicit_enabled),
            "direct_admission_required": bool(direct_admission_required),
            "symbolic_ordering": str(symbolic_ordering),
            "symbolic_block_size": int(symbolic_block_size),
            "symbolic_block_overlap": int(symbolic_block_overlap),
            "symbolic_coarse_max_cols": int(symbolic_coarse_max_cols),
            "symbolic_coarse_probe_cols": int(symbolic_coarse_probe_cols),
            "symbolic_coarse_damping": float(symbolic_coarse_damping),
            "symbolic_coarse_regularization_rel": float(symbolic_coarse_regularization_rel),
            "symbolic_physics_coarse_enabled": bool(symbolic_physics_coarse_enabled),
            "symbolic_physics_coarse_max_cols": int(symbolic_physics_coarse_max_cols),
            "symbolic_schur_max_separator_cols": int(symbolic_schur_max_separator_cols),
            "symbolic_schur_boundary_width": int(symbolic_schur_boundary_width),
            "symbolic_schur_high_degree_cols": int(symbolic_schur_high_degree_cols),
            "symbolic_schur_regularization_rel": float(symbolic_schur_regularization_rel),
            "symbolic_frontal_max_separator_cols": int(symbolic_frontal_max_separator_cols),
            "symbolic_frontal_boundary_width": int(symbolic_frontal_boundary_width),
            "symbolic_frontal_high_degree_cols": int(symbolic_frontal_high_degree_cols),
            "symbolic_frontal_max_superblock_size": int(symbolic_frontal_max_superblock_size),
            "symbolic_frontal_max_superblock_blocks": int(symbolic_frontal_max_superblock_blocks),
            "symbolic_frontal_min_cross_nnz": int(symbolic_frontal_min_cross_nnz),
            "symbolic_frontal_min_cross_separator_fraction": float(symbolic_frontal_min_cross_separator_fraction),
            "symbolic_frontal_regularization_rel": float(symbolic_frontal_regularization_rel),
            "symbolic_frontal_max_dense_rhs_entries": int(symbolic_frontal_max_dense_rhs_entries),
            "symbolic_frontal_max_dense_rhs_cols_per_block": int(symbolic_frontal_max_dense_rhs_cols_per_block),
            "symbolic_blr_frontal_tol": float(symbolic_blr_frontal_tol),
            "symbolic_blr_frontal_max_rank": int(symbolic_blr_frontal_max_rank),
            "symbolic_blr_frontal_min_cols": int(symbolic_blr_frontal_min_cols),
            "symbolic_blr_frontal_gmres_rtol": float(symbolic_blr_frontal_gmres_rtol),
            "symbolic_blr_frontal_gmres_atol": float(symbolic_blr_frontal_gmres_atol),
            "symbolic_blr_frontal_gmres_maxiter": int(symbolic_blr_frontal_gmres_maxiter),
            "symbolic_blr_frontal_gmres_restart": int(symbolic_blr_frontal_gmres_restart),
            "symbolic_blr_frontal_woodbury_max_rank": int(symbolic_blr_frontal_woodbury_max_rank),
            "symbolic_blr_frontal_woodbury_max_condition": float(symbolic_blr_frontal_woodbury_max_condition),
            "symbolic_nd_max_leaf_size": int(symbolic_nd_max_leaf_size),
            "symbolic_nd_max_terminal_factor_size": int(symbolic_nd_max_terminal_factor_size),
            "symbolic_nd_max_depth": int(symbolic_nd_max_depth),
            "symbolic_nd_separator_width": int(symbolic_nd_separator_width),
            "symbolic_nd_max_separator_cols": int(symbolic_nd_max_separator_cols),
            "symbolic_nd_high_degree_cols": int(symbolic_nd_high_degree_cols),
            "symbolic_nd_regularization_rel": float(symbolic_nd_regularization_rel),
            "symbolic_nd_max_dense_rhs_entries": int(symbolic_nd_max_dense_rhs_entries),
            "symbolic_nd_max_dense_rhs_entries_per_child": int(symbolic_nd_max_dense_rhs_entries_per_child),
            "symbolic_nd_max_dense_rhs_cols_per_child": int(symbolic_nd_max_dense_rhs_cols_per_child),
            "symbolic_nd_max_setup_s": float(symbolic_nd_max_setup_s),
            "symbolic_nd_compress_updates": bool(symbolic_nd_compress_updates),
            "symbolic_nd_parallel_update_workers": int(symbolic_nd_parallel_update_workers),
            "symbolic_nd_residual_polish_steps": int(symbolic_nd_residual_polish_steps),
            "symbolic_nd_residual_polish_damping": float(symbolic_nd_residual_polish_damping),
            "symbolic_superblock_max_size": int(symbolic_superblock_max_size),
            "symbolic_superblock_max_blocks": int(symbolic_superblock_max_blocks),
            "symbolic_superblock_min_cross_nnz": int(symbolic_superblock_min_cross_nnz),
            "symbolic_superblock_min_retained_cross_fraction": float(symbolic_superblock_min_retained_cross_fraction),
            "symbolic_superblock_regularization_rel": float(symbolic_superblock_regularization_rel),
            "symbolic_numeric_parallel_workers": int(symbolic_numeric_parallel_workers),
            "symbolic_max_permutation_size": int(symbolic_max_permutation_size),
            "symbolic_admission_enabled": bool(symbolic_admission_enabled),
            "symbolic_admission_max_rel": float(symbolic_admission_max_rel),
            "symbolic_admission_min_improvement": float(symbolic_admission_min_improvement),
            "symbolic_admission_probes": int(symbolic_admission_probes),
            "symbolic_admission_rescue_lu_enabled": bool(symbolic_admission_rescue_lu),
            "symbolic_admission_rescue_lu_max_mb": float(symbolic_admission_rescue_lu_max_mb),
        }
        metadata.update(direct_metadata)
        metadata.update(symbolic_metadata)
        metadata.update(physics_coarse_metadata)
        cached = _TransportFpFortranReducedLuPrecondCache(
            factor_bundle=factor_bundle,
            linear_size=int(linear_size),
            metadata=metadata,
        )
        _TRANSPORT_FP_FORTRAN_REDUCED_LU_PRECOND_CACHE[cache_key] = cached

    factor_bundle = cached.factor_bundle
    linear_size_use = int(cached.linear_size)

    def _solve_host(rhs_host: np.ndarray) -> np.ndarray:
        rhs_np = np.asarray(rhs_host, dtype=np.float64).reshape((linear_size_use,))
        try:
            sol = np.asarray(factor_bundle.solve(rhs_np), dtype=np.float64).reshape((linear_size_use,))
        except Exception:
            sol = rhs_np
        finite = np.isfinite(sol)
        if not np.all(finite):
            sol = np.where(finite, sol, 0.0)
        return sol.astype(np.float64, copy=False)

    def _apply(v: jnp.ndarray) -> jnp.ndarray:
        v = jnp.asarray(v, dtype=jnp.float64)
        return jax.pure_callback(
            _solve_host,
            jax.ShapeDtypeStruct((linear_size_use,), jnp.float64),
            v,
        )

    try:
        setattr(_apply, "_sfincs_jax_transport_fp_fortran_reduced_lu_metadata", dict(cached.metadata))
    except Exception:
        pass
    return _apply
