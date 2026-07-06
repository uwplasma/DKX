"""Structured full-Fokker-Planck RHSMode=1 preconditioners."""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np

from sfincs_jax.solvers.preconditioning import (
    _RHSMODE1_STRUCTURED_FBLOCK_PRECOND_CACHE,
    _RHSMode1StructuredFBlockPrecondCache,
)
from sfincs_jax.solvers.preconditioning import precond_dtype
from sfincs_jax.solvers.preconditioning import rhs_mode1_structured_fblock_cache_key
from sfincs_jax.problems.profile_residual import safe_preconditioner
from sfincs_jax.operators.profile_kinetic import select_structured_rhs1_fblock_operator
from sfincs_jax.solvers.preconditioner_xblock_coarse import (
    _build_rhs1_coupled_moment_matrix_free_correction,
    _build_rhs1_lowmode_angular_matrix_free_correction,
    _build_rhs1_moment_angular_matrix_free_correction,
    _build_rhs1_tail_matrix_free_correction,
)
from sfincs_jax.problems.profile_policies import read_float_env as _rhs1_float_env
from sfincs_jax.problems.profile_policies import read_int_env as _rhs1_int_env
from sfincs_jax.solvers.preconditioner_symbolic_host import RHS1FullSystemMatrixFreeOperatorAdapter
from sfincs_jax.operators.profile_system import V3FullSystemOperator, apply_v3_full_system_operator_cached

Preconditioner = Callable[[jnp.ndarray], jnp.ndarray]

__all__ = (
    "build_rhs1_structured_fblock_angular_jacobi_preconditioner",
    "build_rhs1_structured_fblock_fp_coupled_moment_schur_preconditioner",
    "build_rhs1_structured_fblock_fp_lowmode_schur_preconditioner",
    "build_rhs1_structured_fblock_fp_moment_schur_preconditioner",
    "build_rhs1_structured_fblock_fp_radial_jacobi_preconditioner",
    "build_rhs1_structured_fblock_fp_tail_coupled_schur_preconditioner",
    "build_rhs1_structured_fblock_jacobi_preconditioner",
    "build_rhs1_structured_fblock_xi_angular_jacobi_preconditioner",
)


def _structured_fblock_cache_key(
    op: V3FullSystemOperator,
    kind: str,
    *,
    params: tuple[object, ...] = (),
) -> tuple[object, ...]:
    return rhs_mode1_structured_fblock_cache_key(
        op,
        kind,
        precond_dtype=precond_dtype(),
        params=params,
    )


def _attach_structured_metadata(
    preconditioner: Preconditioner,
    metadata: dict[str, object],
) -> Preconditioner:
    setattr(preconditioner, "_sfincs_jax_structured_fblock_metadata", metadata)
    return preconditioner


def _wrap_structured_fblock_preconditioner(
    *,
    apply_full_unchecked: Preconditioner,
    metadata: dict[str, object],
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None,
) -> Preconditioner:
    """Attach diagnostics and optional active-system projection to a builder."""

    apply_full = _attach_structured_metadata(
        safe_preconditioner(apply_full_unchecked),
        metadata,
    )
    if reduce_full is None or expand_reduced is None:
        return apply_full

    def _apply_reduced(r_reduced: jnp.ndarray) -> jnp.ndarray:
        z_full = apply_full(expand_reduced(r_reduced))
        return reduce_full(z_full)

    return _attach_structured_metadata(_apply_reduced, metadata)


def build_rhs1_structured_fblock_jacobi_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build an opt-in block-Jacobi preconditioner from the structured f-block.

    This is a default-off integration path for the JAX-native block-COO
    assembly. It only uses the structured operator when all f-block terms are
    covered; otherwise selection fails before Krylov starts.
    """

    phi1_base = op.phi1_hat_base if getattr(op.fblock, "fp_phi1", None) is not None else None
    selection = select_structured_rhs1_fblock_operator(op.fblock, phi1_hat_base=phi1_base)
    if not bool(selection.selected):
        raise NotImplementedError(f"structured f-block preconditioner unavailable: {selection.reason}")

    regularization = _rhs1_float_env(
        "SFINCS_JAX_RHSMODE1_STRUCTURED_FBLOCK_JACOBI_REG",
        default=1.0e-10,
        minimum=0.0,
    )
    damping = _rhs1_float_env(
        "SFINCS_JAX_RHSMODE1_STRUCTURED_FBLOCK_JACOBI_DAMPING",
        default=1.0,
        minimum=0.0,
    )
    factor = selection.assembly.operator.block_jacobi_factor(
        regularization=float(regularization),
        damping=float(damping),
    )
    metadata = selection.to_dict()
    metadata["factor"] = factor.to_dict()
    metadata["regularization"] = float(regularization)
    metadata["damping"] = float(damping)

    f_size = int(op.f_size)

    def _apply_full_unchecked(r_full: jnp.ndarray) -> jnp.ndarray:
        r_full = jnp.asarray(r_full, dtype=jnp.float64)
        z_f = factor.apply(r_full[:f_size])
        return jnp.concatenate([z_f, r_full[f_size:]], axis=0)

    return _wrap_structured_fblock_preconditioner(
        apply_full_unchecked=_apply_full_unchecked,
        metadata=metadata,
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
    )


def build_rhs1_structured_fblock_angular_jacobi_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build an opt-in angular-line preconditioner from the structured f-block.

    Blocks are grouped over theta for each fixed ``(species, x, L)`` while each
    scalar block already contains all zeta couplings.  This captures the full
    local angular surface coupling without dense probing or host sparse assembly.
    """

    phi1_base = op.phi1_hat_base if getattr(op.fblock, "fp_phi1", None) is not None else None
    selection = select_structured_rhs1_fblock_operator(op.fblock, phi1_hat_base=phi1_base)
    if not bool(selection.selected):
        raise NotImplementedError(f"structured f-block angular preconditioner unavailable: {selection.reason}")

    regularization = _rhs1_float_env(
        "SFINCS_JAX_RHSMODE1_STRUCTURED_FBLOCK_ANGULAR_JACOBI_REG",
        default=1.0e-10,
        minimum=0.0,
    )
    damping = _rhs1_float_env(
        "SFINCS_JAX_RHSMODE1_STRUCTURED_FBLOCK_ANGULAR_JACOBI_DAMPING",
        default=1.0,
        minimum=0.0,
    )
    factor = selection.assembly.operator.line_jacobi_factor(
        blocks_per_line=int(op.n_theta),
        regularization=float(regularization),
        damping=float(damping),
    )
    metadata = selection.to_dict()
    metadata["factor"] = factor.to_dict()
    metadata["regularization"] = float(regularization)
    metadata["damping"] = float(damping)
    metadata["line_kind"] = "fixed_species_x_l_angular"
    metadata["blocks_per_line"] = int(op.n_theta)

    f_size = int(op.f_size)

    def _apply_full_unchecked(r_full: jnp.ndarray) -> jnp.ndarray:
        r_full = jnp.asarray(r_full, dtype=jnp.float64)
        z_f = factor.apply(r_full[:f_size])
        return jnp.concatenate([z_f, r_full[f_size:]], axis=0)

    return _wrap_structured_fblock_preconditioner(
        apply_full_unchecked=_apply_full_unchecked,
        metadata=metadata,
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
    )


def build_rhs1_structured_fblock_xi_angular_jacobi_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build a guarded fixed-``(species,x)`` velocity-angular preconditioner.

    The grouped block contains all pitch/Legendre and angular degrees of
    freedom for one species and one radial grid point.  This is stronger than
    point or angular-line Jacobi for PAS/collisionless coupling, but it is
    explicitly guarded because the grouped block scales as
    ``Nxi * Ntheta * Nzeta``.
    """

    blocks_per_line = int(op.n_xi) * int(op.n_theta)
    scalar_block_size = blocks_per_line * int(op.n_zeta)
    max_block_size = _rhs1_int_env(
        "SFINCS_JAX_RHSMODE1_STRUCTURED_FBLOCK_XI_ANGULAR_MAX_BLOCK_SIZE",
        default=5000,
        minimum=1,
    )
    if int(scalar_block_size) > int(max_block_size):
        raise MemoryError(
            "structured f-block xi-angular preconditioner block too large: "
            f"{int(scalar_block_size)} > {int(max_block_size)}; raise "
            "SFINCS_JAX_RHSMODE1_STRUCTURED_FBLOCK_XI_ANGULAR_MAX_BLOCK_SIZE to override"
        )

    phi1_base = op.phi1_hat_base if getattr(op.fblock, "fp_phi1", None) is not None else None
    selection = select_structured_rhs1_fblock_operator(op.fblock, phi1_hat_base=phi1_base)
    if not bool(selection.selected):
        raise NotImplementedError(f"structured f-block xi-angular preconditioner unavailable: {selection.reason}")

    regularization = _rhs1_float_env(
        "SFINCS_JAX_RHSMODE1_STRUCTURED_FBLOCK_XI_ANGULAR_JACOBI_REG",
        default=1.0e-10,
        minimum=0.0,
    )
    damping = _rhs1_float_env(
        "SFINCS_JAX_RHSMODE1_STRUCTURED_FBLOCK_XI_ANGULAR_JACOBI_DAMPING",
        default=1.0,
        minimum=0.0,
    )
    factor = selection.assembly.operator.line_jacobi_factor(
        blocks_per_line=blocks_per_line,
        regularization=float(regularization),
        damping=float(damping),
    )
    metadata = selection.to_dict()
    metadata["factor"] = factor.to_dict()
    metadata["regularization"] = float(regularization)
    metadata["damping"] = float(damping)
    metadata["line_kind"] = "fixed_species_x_velocity_angular"
    metadata["blocks_per_line"] = int(blocks_per_line)
    metadata["scalar_block_size"] = int(scalar_block_size)
    metadata["max_block_size"] = int(max_block_size)

    f_size = int(op.f_size)

    def _apply_full_unchecked(r_full: jnp.ndarray) -> jnp.ndarray:
        r_full = jnp.asarray(r_full, dtype=jnp.float64)
        z_f = factor.apply(r_full[:f_size])
        return jnp.concatenate([z_f, r_full[f_size:]], axis=0)

    return _wrap_structured_fblock_preconditioner(
        apply_full_unchecked=_apply_full_unchecked,
        metadata=metadata,
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
    )


def build_rhs1_structured_fblock_fp_radial_jacobi_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build a guarded FP-aware species/radial grouped preconditioner.

    The full-Fokker-Planck collision operator is dense in species and radial
    ``x`` for each fixed Legendre index and angular point.  This factor groups
    those non-contiguous blocks directly from the structured block-COO operator,
    capturing the missing FP coupling without a global sparse factor.
    """

    if getattr(op.fblock, "fp", None) is None and getattr(op.fblock, "fp_phi1", None) is None:
        raise NotImplementedError("structured f-block FP-radial preconditioner requires an FP collision term")

    n_species = int(op.n_species)
    n_x = int(op.n_x)
    n_xi = int(op.n_xi)
    n_theta = int(op.n_theta)
    n_zeta = int(op.n_zeta)
    blocks_per_group = n_species * n_x
    scalar_block_size = blocks_per_group * n_zeta
    n_groups = n_xi * n_theta
    max_block_size = _rhs1_int_env(
        "SFINCS_JAX_RHSMODE1_STRUCTURED_FBLOCK_FP_RADIAL_MAX_BLOCK_SIZE",
        default=2500,
        minimum=1,
    )
    if int(scalar_block_size) > int(max_block_size):
        raise MemoryError(
            "structured f-block FP-radial preconditioner block too large: "
            f"{int(scalar_block_size)} > {int(max_block_size)}; raise "
            "SFINCS_JAX_RHSMODE1_STRUCTURED_FBLOCK_FP_RADIAL_MAX_BLOCK_SIZE to override"
        )

    estimated_factor_nbytes = int(n_groups * scalar_block_size * scalar_block_size * np.dtype(np.float64).itemsize)
    max_factor_nbytes = _rhs1_int_env(
        "SFINCS_JAX_RHSMODE1_STRUCTURED_FBLOCK_FP_RADIAL_MAX_FACTOR_NBYTES",
        default=256 * 1024 * 1024,
        minimum=1,
    )
    if estimated_factor_nbytes > int(max_factor_nbytes):
        raise MemoryError(
            "structured f-block FP-radial preconditioner factor too large: "
            f"{estimated_factor_nbytes} > {int(max_factor_nbytes)} bytes; raise "
            "SFINCS_JAX_RHSMODE1_STRUCTURED_FBLOCK_FP_RADIAL_MAX_FACTOR_NBYTES to override"
        )

    regularization = _rhs1_float_env(
        "SFINCS_JAX_RHSMODE1_STRUCTURED_FBLOCK_FP_RADIAL_JACOBI_REG",
        default=1.0e-10,
        minimum=0.0,
    )
    damping = _rhs1_float_env(
        "SFINCS_JAX_RHSMODE1_STRUCTURED_FBLOCK_FP_RADIAL_JACOBI_DAMPING",
        default=1.0,
        minimum=0.0,
    )
    cache_key = _structured_fblock_cache_key(
        op,
        "fp_radial_jacobi",
        params=(
            int(max_block_size),
            int(max_factor_nbytes),
            float(regularization),
            float(damping),
        ),
    )
    cached = _RHSMODE1_STRUCTURED_FBLOCK_PRECOND_CACHE.get(cache_key)
    cache_hit = cached is not None
    if cached is None:
        phi1_base = op.phi1_hat_base if getattr(op.fblock, "fp_phi1", None) is not None else None
        selection = select_structured_rhs1_fblock_operator(op.fblock, phi1_hat_base=phi1_base)
        if not bool(selection.selected):
            raise NotImplementedError(f"structured f-block FP-radial preconditioner unavailable: {selection.reason}")

        groups = np.empty((n_groups, blocks_per_group), dtype=np.int32)
        group_id = 0
        for ell in range(n_xi):
            for theta in range(n_theta):
                offset = 0
                for species in range(n_species):
                    for x_index in range(n_x):
                        groups[group_id, offset] = (((species * n_x + x_index) * n_xi + ell) * n_theta + theta)
                        offset += 1
                group_id += 1

        factor = selection.assembly.operator.grouped_jacobi_factor(
            block_groups=groups,
            regularization=float(regularization),
            damping=float(damping),
        )
        metadata = selection.to_dict()
        metadata["factor"] = factor.to_dict()
        metadata["regularization"] = float(regularization)
        metadata["damping"] = float(damping)
        metadata["line_kind"] = "fixed_l_theta_species_x_zeta"
        metadata["blocks_per_group"] = int(blocks_per_group)
        metadata["n_groups"] = int(n_groups)
        metadata["scalar_block_size"] = int(scalar_block_size)
        metadata["max_block_size"] = int(max_block_size)
        metadata["estimated_factor_nbytes"] = int(estimated_factor_nbytes)
        metadata["max_factor_nbytes"] = int(max_factor_nbytes)
        cached = _RHSMode1StructuredFBlockPrecondCache(
            operator=selection.assembly.operator,
            factor=factor,
            metadata=metadata,
        )
        _RHSMODE1_STRUCTURED_FBLOCK_PRECOND_CACHE[cache_key] = cached
    if cached.factor is None:
        raise RuntimeError("structured f-block FP-radial cache is missing factor")
    factor = cached.factor
    metadata = dict(cached.metadata)
    metadata["cache_hit"] = bool(cache_hit)

    f_size = int(op.f_size)

    def _apply_full_unchecked(r_full: jnp.ndarray) -> jnp.ndarray:
        r_full = jnp.asarray(r_full, dtype=jnp.float64)
        z_f = factor.apply(r_full[:f_size])
        return jnp.concatenate([z_f, r_full[f_size:]], axis=0)

    return _wrap_structured_fblock_preconditioner(
        apply_full_unchecked=_apply_full_unchecked,
        metadata=metadata,
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
    )


def build_rhs1_structured_fblock_fp_lowmode_schur_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build an explicit FP-radial plus low-angular Galerkin correction."""

    if getattr(op.fblock, "fp", None) is None and getattr(op.fblock, "fp_phi1", None) is None:
        raise NotImplementedError("structured f-block FP low-mode Schur preconditioner requires an FP collision term")

    theta_modes = _rhs1_int_env(
        "SFINCS_JAX_RHSMODE1_STRUCTURED_FBLOCK_FP_LOWMODE_THETA",
        default=1,
        minimum=0,
    )
    zeta_modes = _rhs1_int_env(
        "SFINCS_JAX_RHSMODE1_STRUCTURED_FBLOCK_FP_LOWMODE_ZETA",
        default=1,
        minimum=0,
    )
    max_coarse_size = _rhs1_int_env(
        "SFINCS_JAX_RHSMODE1_STRUCTURED_FBLOCK_FP_LOWMODE_MAX_COARSE",
        default=800,
        minimum=1,
    )
    max_basis_nbytes = _rhs1_int_env(
        "SFINCS_JAX_RHSMODE1_STRUCTURED_FBLOCK_FP_LOWMODE_MAX_BASIS_NBYTES",
        default=128 * 1024 * 1024,
        minimum=1,
    )
    basis_batch_size = _rhs1_int_env(
        "SFINCS_JAX_RHSMODE1_STRUCTURED_FBLOCK_FP_LOWMODE_BASIS_BATCH",
        default=32,
        minimum=1,
    )
    regularization = _rhs1_float_env(
        "SFINCS_JAX_RHSMODE1_STRUCTURED_FBLOCK_FP_LOWMODE_SCHUR_REG",
        default=1.0e-10,
        minimum=0.0,
    )
    damping = _rhs1_float_env(
        "SFINCS_JAX_RHSMODE1_STRUCTURED_FBLOCK_FP_LOWMODE_SCHUR_DAMPING",
        default=1.0,
        minimum=0.0,
    )
    cache_key = _structured_fblock_cache_key(
        op,
        "fp_lowmode_schur",
        params=(
            int(theta_modes),
            int(zeta_modes),
            int(max_coarse_size),
            int(max_basis_nbytes),
            int(basis_batch_size),
            float(regularization),
            float(damping),
        ),
    )
    cached = _RHSMODE1_STRUCTURED_FBLOCK_PRECOND_CACHE.get(cache_key)
    cache_hit = cached is not None
    if cached is None:
        base_preconditioner = build_rhs1_structured_fblock_fp_radial_jacobi_preconditioner(op=op)
        base_metadata = getattr(base_preconditioner, "_sfincs_jax_structured_fblock_metadata", {})
        phi1_base = op.phi1_hat_base if getattr(op.fblock, "fp_phi1", None) is not None else None
        selection = select_structured_rhs1_fblock_operator(op.fblock, phi1_hat_base=phi1_base)
        if not bool(selection.selected):
            raise NotImplementedError(
                f"structured f-block FP low-mode Schur preconditioner unavailable: {selection.reason}"
            )
        coarse, feature_metadata = _build_rhs1_lowmode_angular_matrix_free_correction(
            op=op,
            operator=selection.assembly.operator,
            theta_modes=int(theta_modes),
            zeta_modes=int(zeta_modes),
            max_coarse_size=int(max_coarse_size),
            max_basis_batch_nbytes=int(max_basis_nbytes),
            basis_batch_size=int(basis_batch_size),
            regularization=float(regularization),
            damping=float(damping),
        )
        metadata = selection.to_dict()
        metadata["base_preconditioner"] = base_metadata
        metadata["coarse"] = coarse.to_dict()
        metadata["coarse_feature_selection"] = feature_metadata
        metadata["line_kind"] = "fp_radial_plus_low_angular_galerkin"
        metadata["theta_modes"] = int(theta_modes)
        metadata["zeta_modes"] = int(zeta_modes)
        metadata["max_coarse_size"] = int(max_coarse_size)
        metadata["max_basis_batch_nbytes"] = int(max_basis_nbytes)
        metadata["basis_batch_size"] = int(basis_batch_size)
        metadata["regularization"] = float(regularization)
        metadata["damping"] = float(damping)
        cached = _RHSMode1StructuredFBlockPrecondCache(
            operator=selection.assembly.operator,
            coarse=coarse,
            base_preconditioner=base_preconditioner,
            metadata=metadata,
        )
        _RHSMODE1_STRUCTURED_FBLOCK_PRECOND_CACHE[cache_key] = cached
    if cached.base_preconditioner is None or cached.coarse is None:
        raise RuntimeError("structured f-block FP low-mode cache is incomplete")
    base_preconditioner = cached.base_preconditioner
    coarse = cached.coarse
    operator = cached.operator
    metadata = dict(cached.metadata)
    metadata["cache_hit"] = bool(cache_hit)

    f_size = int(op.f_size)

    def _apply_full_unchecked(r_full: jnp.ndarray) -> jnp.ndarray:
        r_full = jnp.asarray(r_full, dtype=jnp.float64)
        z0_full = base_preconditioner(r_full)
        residual_f = r_full[:f_size] - operator.matvec(z0_full[:f_size])
        z_f = z0_full[:f_size] + coarse.apply(residual_f)
        return jnp.concatenate([z_f, z0_full[f_size:]], axis=0)

    return _wrap_structured_fblock_preconditioner(
        apply_full_unchecked=_apply_full_unchecked,
        metadata=metadata,
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
    )


def build_rhs1_structured_fblock_fp_moment_schur_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build an explicit FP-radial plus compact moment-space correction."""

    if getattr(op.fblock, "fp", None) is None and getattr(op.fblock, "fp_phi1", None) is None:
        raise NotImplementedError("structured f-block FP moment Schur preconditioner requires an FP collision term")

    theta_modes = _rhs1_int_env(
        "SFINCS_JAX_RHSMODE1_STRUCTURED_FBLOCK_FP_MOMENT_THETA",
        default=1,
        minimum=0,
    )
    zeta_modes = _rhs1_int_env(
        "SFINCS_JAX_RHSMODE1_STRUCTURED_FBLOCK_FP_MOMENT_ZETA",
        default=1,
        minimum=0,
    )
    x_moments = _rhs1_int_env(
        "SFINCS_JAX_RHSMODE1_STRUCTURED_FBLOCK_FP_MOMENT_X",
        default=2,
        minimum=1,
    )
    xi_moments = _rhs1_int_env(
        "SFINCS_JAX_RHSMODE1_STRUCTURED_FBLOCK_FP_MOMENT_XI",
        default=2,
        minimum=1,
    )
    max_coarse_size = _rhs1_int_env(
        "SFINCS_JAX_RHSMODE1_STRUCTURED_FBLOCK_FP_MOMENT_MAX_COARSE",
        default=256,
        minimum=1,
    )
    max_basis_nbytes = _rhs1_int_env(
        "SFINCS_JAX_RHSMODE1_STRUCTURED_FBLOCK_FP_MOMENT_MAX_BASIS_NBYTES",
        default=32 * 1024 * 1024,
        minimum=1,
    )
    basis_batch_size = _rhs1_int_env(
        "SFINCS_JAX_RHSMODE1_STRUCTURED_FBLOCK_FP_MOMENT_BASIS_BATCH",
        default=32,
        minimum=1,
    )
    regularization = _rhs1_float_env(
        "SFINCS_JAX_RHSMODE1_STRUCTURED_FBLOCK_FP_MOMENT_SCHUR_REG",
        default=1.0e-10,
        minimum=0.0,
    )
    damping = _rhs1_float_env(
        "SFINCS_JAX_RHSMODE1_STRUCTURED_FBLOCK_FP_MOMENT_SCHUR_DAMPING",
        default=1.0,
        minimum=0.0,
    )
    cache_key = _structured_fblock_cache_key(
        op,
        "fp_moment_schur",
        params=(
            int(theta_modes),
            int(zeta_modes),
            int(x_moments),
            int(xi_moments),
            int(max_coarse_size),
            int(max_basis_nbytes),
            int(basis_batch_size),
            float(regularization),
            float(damping),
        ),
    )
    cached = _RHSMODE1_STRUCTURED_FBLOCK_PRECOND_CACHE.get(cache_key)
    cache_hit = cached is not None
    if cached is None:
        base_preconditioner = build_rhs1_structured_fblock_fp_radial_jacobi_preconditioner(op=op)
        base_metadata = getattr(base_preconditioner, "_sfincs_jax_structured_fblock_metadata", {})
        phi1_base = op.phi1_hat_base if getattr(op.fblock, "fp_phi1", None) is not None else None
        selection = select_structured_rhs1_fblock_operator(op.fblock, phi1_hat_base=phi1_base)
        if not bool(selection.selected):
            raise NotImplementedError(
                f"structured f-block FP moment Schur preconditioner unavailable: {selection.reason}"
            )
        coarse, moment_metadata = _build_rhs1_moment_angular_matrix_free_correction(
            op=op,
            operator=selection.assembly.operator,
            theta_modes=int(theta_modes),
            zeta_modes=int(zeta_modes),
            x_moments=int(x_moments),
            xi_moments=int(xi_moments),
            max_coarse_size=int(max_coarse_size),
            max_basis_batch_nbytes=int(max_basis_nbytes),
            basis_batch_size=int(basis_batch_size),
            regularization=float(regularization),
            damping=float(damping),
        )

        metadata = selection.to_dict()
        metadata["base_preconditioner"] = base_metadata
        metadata["coarse"] = coarse.to_dict()
        metadata["coarse_moment_selection"] = moment_metadata
        metadata["line_kind"] = "fp_radial_plus_low_x_xi_angular_moment_galerkin"
        metadata["theta_modes"] = int(theta_modes)
        metadata["zeta_modes"] = int(zeta_modes)
        metadata["x_moments"] = int(x_moments)
        metadata["xi_moments"] = int(xi_moments)
        metadata["max_coarse_size"] = int(max_coarse_size)
        metadata["max_basis_batch_nbytes"] = int(max_basis_nbytes)
        metadata["basis_batch_size"] = int(basis_batch_size)
        metadata["regularization"] = float(regularization)
        metadata["damping"] = float(damping)
        cached = _RHSMode1StructuredFBlockPrecondCache(
            operator=selection.assembly.operator,
            coarse=coarse,
            base_preconditioner=base_preconditioner,
            metadata=metadata,
        )
        _RHSMODE1_STRUCTURED_FBLOCK_PRECOND_CACHE[cache_key] = cached
    if cached.base_preconditioner is None or cached.coarse is None:
        raise RuntimeError("structured f-block FP moment cache is incomplete")
    base_preconditioner = cached.base_preconditioner
    coarse = cached.coarse
    operator = cached.operator
    metadata = dict(cached.metadata)
    metadata["cache_hit"] = bool(cache_hit)

    f_size = int(op.f_size)

    def _apply_full_unchecked(r_full: jnp.ndarray) -> jnp.ndarray:
        r_full = jnp.asarray(r_full, dtype=jnp.float64)
        z0_full = base_preconditioner(r_full)
        residual_f = r_full[:f_size] - operator.matvec(z0_full[:f_size])
        z_f = z0_full[:f_size] + coarse.apply(residual_f)
        return jnp.concatenate([z_f, z0_full[f_size:]], axis=0)

    return _wrap_structured_fblock_preconditioner(
        apply_full_unchecked=_apply_full_unchecked,
        metadata=metadata,
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
    )


def build_rhs1_structured_fblock_fp_coupled_moment_schur_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build an FP-radial base plus coupled full-system moment correction."""

    if getattr(op.fblock, "fp", None) is None and getattr(op.fblock, "fp_phi1", None) is None:
        raise NotImplementedError(
            "structured f-block FP coupled moment Schur preconditioner requires an FP collision term"
        )

    theta_modes = _rhs1_int_env(
        "SFINCS_JAX_RHSMODE1_STRUCTURED_FBLOCK_FP_COUPLED_THETA",
        default=1,
        minimum=0,
    )
    zeta_modes = _rhs1_int_env(
        "SFINCS_JAX_RHSMODE1_STRUCTURED_FBLOCK_FP_COUPLED_ZETA",
        default=1,
        minimum=0,
    )
    x_moments = _rhs1_int_env(
        "SFINCS_JAX_RHSMODE1_STRUCTURED_FBLOCK_FP_COUPLED_X",
        default=2,
        minimum=1,
    )
    xi_moments = _rhs1_int_env(
        "SFINCS_JAX_RHSMODE1_STRUCTURED_FBLOCK_FP_COUPLED_XI",
        default=2,
        minimum=1,
    )
    max_tail_size = _rhs1_int_env(
        "SFINCS_JAX_RHSMODE1_STRUCTURED_FBLOCK_FP_COUPLED_MAX_TAIL",
        default=256,
        minimum=0,
    )
    max_coarse_size = _rhs1_int_env(
        "SFINCS_JAX_RHSMODE1_STRUCTURED_FBLOCK_FP_COUPLED_MAX_COARSE",
        default=512,
        minimum=1,
    )
    max_basis_nbytes = _rhs1_int_env(
        "SFINCS_JAX_RHSMODE1_STRUCTURED_FBLOCK_FP_COUPLED_MAX_BASIS_NBYTES",
        default=64 * 1024 * 1024,
        minimum=1,
    )
    basis_batch_size = _rhs1_int_env(
        "SFINCS_JAX_RHSMODE1_STRUCTURED_FBLOCK_FP_COUPLED_BASIS_BATCH",
        default=32,
        minimum=1,
    )
    regularization = _rhs1_float_env(
        "SFINCS_JAX_RHSMODE1_STRUCTURED_FBLOCK_FP_COUPLED_SCHUR_REG",
        default=1.0e-10,
        minimum=0.0,
    )
    damping = _rhs1_float_env(
        "SFINCS_JAX_RHSMODE1_STRUCTURED_FBLOCK_FP_COUPLED_SCHUR_DAMPING",
        default=1.0,
        minimum=0.0,
    )
    cache_key = _structured_fblock_cache_key(
        op,
        "fp_coupled_moment_schur",
        params=(
            int(theta_modes),
            int(zeta_modes),
            int(x_moments),
            int(xi_moments),
            int(max_tail_size),
            int(max_coarse_size),
            int(max_basis_nbytes),
            int(basis_batch_size),
            float(regularization),
            float(damping),
        ),
    )
    cached = _RHSMODE1_STRUCTURED_FBLOCK_PRECOND_CACHE.get(cache_key)
    cache_hit = cached is not None
    if cached is None:
        base_preconditioner = build_rhs1_structured_fblock_fp_radial_jacobi_preconditioner(op=op)
        base_metadata = getattr(base_preconditioner, "_sfincs_jax_structured_fblock_metadata", {})
        full_operator = RHS1FullSystemMatrixFreeOperatorAdapter(op)
        coarse, coupled_metadata = _build_rhs1_coupled_moment_matrix_free_correction(
            op=op,
            operator=full_operator,
            theta_modes=int(theta_modes),
            zeta_modes=int(zeta_modes),
            x_moments=int(x_moments),
            xi_moments=int(xi_moments),
            max_tail_size=int(max_tail_size),
            max_coarse_size=int(max_coarse_size),
            max_basis_batch_nbytes=int(max_basis_nbytes),
            basis_batch_size=int(basis_batch_size),
            regularization=float(regularization),
            damping=float(damping),
        )
        metadata = {
            "selected": True,
            "reason": "complete",
            "base_preconditioner": base_metadata,
            "coarse": coarse.to_dict(),
            "coarse_coupled_selection": coupled_metadata,
            "line_kind": "fp_radial_plus_coupled_tail_moment_galerkin",
            "theta_modes": int(theta_modes),
            "zeta_modes": int(zeta_modes),
            "x_moments": int(x_moments),
            "xi_moments": int(xi_moments),
            "max_tail_size": int(max_tail_size),
            "max_coarse_size": int(max_coarse_size),
            "max_basis_batch_nbytes": int(max_basis_nbytes),
            "basis_batch_size": int(basis_batch_size),
            "regularization": float(regularization),
            "damping": float(damping),
        }
        cached = _RHSMode1StructuredFBlockPrecondCache(
            operator=full_operator,
            coarse=coarse,
            base_preconditioner=base_preconditioner,
            metadata=metadata,
        )
        _RHSMODE1_STRUCTURED_FBLOCK_PRECOND_CACHE[cache_key] = cached
    if cached.base_preconditioner is None or cached.coarse is None:
        raise RuntimeError("structured f-block FP coupled moment cache is incomplete")
    base_preconditioner = cached.base_preconditioner
    coarse = cached.coarse
    metadata = dict(cached.metadata)
    metadata["cache_hit"] = bool(cache_hit)

    def _apply_full_unchecked(r_full: jnp.ndarray) -> jnp.ndarray:
        r_full = jnp.asarray(r_full, dtype=jnp.float64)
        z0_full = base_preconditioner(r_full)
        residual_full = r_full - apply_v3_full_system_operator_cached(op, z0_full)
        return z0_full + coarse.apply(residual_full)

    return _wrap_structured_fblock_preconditioner(
        apply_full_unchecked=_apply_full_unchecked,
        metadata=metadata,
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
    )


def build_rhs1_structured_fblock_fp_tail_coupled_schur_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build an FP-radial base plus tail-only full-system residual correction."""

    if getattr(op.fblock, "fp", None) is None and getattr(op.fblock, "fp_phi1", None) is None:
        raise NotImplementedError(
            "structured f-block FP tail-coupled Schur preconditioner requires an FP collision term"
        )

    max_tail_size = _rhs1_int_env(
        "SFINCS_JAX_RHSMODE1_STRUCTURED_FBLOCK_FP_TAIL_COUPLED_MAX_TAIL",
        default=256,
        minimum=0,
    )
    max_coarse_size = _rhs1_int_env(
        "SFINCS_JAX_RHSMODE1_STRUCTURED_FBLOCK_FP_TAIL_COUPLED_MAX_COARSE",
        default=512,
        minimum=1,
    )
    max_basis_nbytes = _rhs1_int_env(
        "SFINCS_JAX_RHSMODE1_STRUCTURED_FBLOCK_FP_TAIL_COUPLED_MAX_BASIS_NBYTES",
        default=64 * 1024 * 1024,
        minimum=1,
    )
    max_action_nbytes = _rhs1_int_env(
        "SFINCS_JAX_RHSMODE1_STRUCTURED_FBLOCK_FP_TAIL_COUPLED_MAX_ACTION_NBYTES",
        default=64 * 1024 * 1024,
        minimum=1,
    )
    basis_batch_size = _rhs1_int_env(
        "SFINCS_JAX_RHSMODE1_STRUCTURED_FBLOCK_FP_TAIL_COUPLED_BASIS_BATCH",
        default=32,
        minimum=1,
    )
    regularization = _rhs1_float_env(
        "SFINCS_JAX_RHSMODE1_STRUCTURED_FBLOCK_FP_TAIL_COUPLED_SCHUR_REG",
        default=1.0e-8,
        minimum=0.0,
    )
    damping = _rhs1_float_env(
        "SFINCS_JAX_RHSMODE1_STRUCTURED_FBLOCK_FP_TAIL_COUPLED_SCHUR_DAMPING",
        default=1.0,
        minimum=0.0,
    )
    cache_key = _structured_fblock_cache_key(
        op,
        "fp_tail_coupled_schur",
        params=(
            int(max_tail_size),
            int(max_coarse_size),
            int(max_basis_nbytes),
            int(max_action_nbytes),
            int(basis_batch_size),
            float(regularization),
            float(damping),
        ),
    )
    cached = _RHSMODE1_STRUCTURED_FBLOCK_PRECOND_CACHE.get(cache_key)
    cache_hit = cached is not None
    if cached is None:
        base_preconditioner = build_rhs1_structured_fblock_fp_radial_jacobi_preconditioner(op=op)
        base_metadata = getattr(base_preconditioner, "_sfincs_jax_structured_fblock_metadata", {})
        full_operator = RHS1FullSystemMatrixFreeOperatorAdapter(op)
        coarse, tail_metadata = _build_rhs1_tail_matrix_free_correction(
            op=op,
            operator=full_operator,
            max_tail_size=int(max_tail_size),
            max_coarse_size=int(max_coarse_size),
            max_basis_batch_nbytes=int(max_basis_nbytes),
            max_action_nbytes=int(max_action_nbytes),
            basis_batch_size=int(basis_batch_size),
            regularization=float(regularization),
            damping=float(damping),
        )
        metadata = {
            "selected": True,
            "reason": "complete",
            "base_preconditioner": base_metadata,
            "coarse": coarse.to_dict(),
            "coarse_tail_selection": tail_metadata,
            "line_kind": "fp_radial_plus_tail_coupled_minres",
            "max_tail_size": int(max_tail_size),
            "max_coarse_size": int(max_coarse_size),
            "max_basis_batch_nbytes": int(max_basis_nbytes),
            "max_action_nbytes": int(max_action_nbytes),
            "basis_batch_size": int(basis_batch_size),
            "regularization": float(regularization),
            "damping": float(damping),
        }
        cached = _RHSMode1StructuredFBlockPrecondCache(
            operator=full_operator,
            coarse=coarse,
            base_preconditioner=base_preconditioner,
            metadata=metadata,
        )
        _RHSMODE1_STRUCTURED_FBLOCK_PRECOND_CACHE[cache_key] = cached
    if cached.base_preconditioner is None or cached.coarse is None:
        raise RuntimeError("structured f-block FP tail-coupled cache is incomplete")
    base_preconditioner = cached.base_preconditioner
    coarse = cached.coarse
    metadata = dict(cached.metadata)
    metadata["cache_hit"] = bool(cache_hit)

    def _apply_full_unchecked(r_full: jnp.ndarray) -> jnp.ndarray:
        r_full = jnp.asarray(r_full, dtype=jnp.float64)
        z0_full = base_preconditioner(r_full)
        residual_full = r_full - apply_v3_full_system_operator_cached(op, z0_full)
        return z0_full + coarse.apply(residual_full)

    return _wrap_structured_fblock_preconditioner(
        apply_full_unchecked=_apply_full_unchecked,
        metadata=metadata,
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
    )
