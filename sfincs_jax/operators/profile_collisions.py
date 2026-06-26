"""RHSMode=1 collision-term assembly helpers for JAX-native block operators."""

from __future__ import annotations

from typing import Any

import numpy as np

from sfincs_jax.operators.profile_layout import RHS1BlockCOOBuilder, RHS1BlockCOOOperator, RHS1BlockLayout


def build_pas_collision_f_block_operator(
    *,
    layout: RHS1BlockLayout,
    pas_operator: Any,
    drop_tol: float = 0.0,
) -> RHS1BlockCOOOperator:
    """Assemble the pitch-angle-scattering collision term on the kinetic block.

    The v3 PAS collision term is diagonal in ``(species, x, L, theta, zeta)``.
    Coefficients do not depend on ``theta`` or ``zeta``, so this routine stores
    each fixed ``(species, x, L, theta)`` zeta line as one dense diagonal block.
    It is the first production-shaped bridge from an existing physics operator
    to the JAX-native block-COO representation.
    """

    n_species = int(layout.n_species)
    n_x = int(layout.n_x)
    n_xi = int(layout.n_xi)
    n_theta = int(layout.n_theta)
    n_zeta = int(layout.n_zeta)
    if int(layout.f_size) != n_species * n_x * n_xi * n_theta * n_zeta:
        raise ValueError("layout f_size is inconsistent with kinetic dimensions")

    coef = np.asarray(pas_operator.coef, dtype=np.float64)
    expected_coef_shape = (n_species, n_x, n_xi)
    if coef.shape != expected_coef_shape:
        raise ValueError(f"pas_operator.coef must have shape {expected_coef_shape}, got {coef.shape}")
    mask = np.asarray(pas_operator.mask_xi, dtype=bool)
    expected_mask_shape = (n_x, n_xi)
    if mask.shape != expected_mask_shape:
        raise ValueError(f"pas_operator.mask_xi must have shape {expected_mask_shape}, got {mask.shape}")

    builder = RHS1BlockCOOBuilder(shape=(int(layout.f_size), int(layout.f_size)), block_size=n_zeta, dtype=np.float64)
    eye_zeta = np.eye(n_zeta, dtype=np.float64)
    threshold = float(drop_tol)
    for species in range(n_species):
        for ix in range(n_x):
            for ell in range(n_xi):
                if not bool(mask[ix, ell]):
                    continue
                value = float(coef[species, ix, ell])
                if threshold > 0.0 and abs(value) <= threshold:
                    continue
                block_value = value * eye_zeta
                for theta in range(n_theta):
                    block_id = layout.kinetic_flat_index(
                        species=species,
                        x=ix,
                        ell=ell,
                        theta=theta,
                        zeta=0,
                    ) // n_zeta
                    builder.add_dense_block(block_id, block_id, block_value)
    return builder.build(drop_tol=threshold)


def build_fokker_planck_collision_f_block_operator(
    *,
    layout: RHS1BlockLayout,
    fp_operator: Any,
    drop_tol: float = 0.0,
) -> RHS1BlockCOOOperator:
    """Assemble the no-Phi1 v3 Fokker-Planck collision term on the kinetic block.

    The v3 FP collision term is diagonal in theta, zeta, and Legendre index but
    dense in species and radial ``x``.  The production evaluator masks only
    output rows for inactive ``L`` slots; it does not pre-mask input columns.
    This builder preserves that padded-vector behavior exactly.
    """

    n_species = int(layout.n_species)
    n_x = int(layout.n_x)
    n_xi = int(layout.n_xi)
    n_theta = int(layout.n_theta)
    n_zeta = int(layout.n_zeta)
    if int(layout.f_size) != n_species * n_x * n_xi * n_theta * n_zeta:
        raise ValueError("layout f_size is inconsistent with kinetic dimensions")

    mat = np.asarray(fp_operator.mat, dtype=np.float64)
    expected_mat_shape = (n_species, n_species, n_xi, n_x, n_x)
    if mat.shape != expected_mat_shape:
        raise ValueError(f"fp_operator.mat must have shape {expected_mat_shape}, got {mat.shape}")
    n_xi_for_x = np.asarray(fp_operator.n_xi_for_x, dtype=np.int32).reshape((-1,))
    if n_xi_for_x.shape != (n_x,):
        raise ValueError(f"fp_operator.n_xi_for_x must have shape {(n_x,)}, got {n_xi_for_x.shape}")
    if np.any(n_xi_for_x < 0) or np.any(n_xi_for_x > n_xi):
        raise ValueError("fp_operator.n_xi_for_x contains invalid active-L counts")

    builder = RHS1BlockCOOBuilder(shape=(int(layout.f_size), int(layout.f_size)), block_size=n_zeta, dtype=np.float64)
    eye_zeta = np.eye(n_zeta, dtype=np.float64)
    threshold = float(drop_tol)
    for row_species in range(n_species):
        for row_x in range(n_x):
            for ell in range(min(n_xi, int(n_xi_for_x[row_x]))):
                for theta in range(n_theta):
                    row_block = layout.kinetic_flat_index(
                        species=row_species,
                        x=row_x,
                        ell=ell,
                        theta=theta,
                        zeta=0,
                    ) // n_zeta
                    for col_species in range(n_species):
                        for col_x in range(n_x):
                            value = float(mat[row_species, col_species, ell, row_x, col_x])
                            if value == 0.0 or (threshold > 0.0 and abs(value) <= threshold):
                                continue
                            col_block = layout.kinetic_flat_index(
                                species=col_species,
                                x=col_x,
                                ell=ell,
                                theta=theta,
                                zeta=0,
                            ) // n_zeta
                            builder.add_dense_block(row_block, col_block, value * eye_zeta)
    return builder.build(drop_tol=threshold)


def build_fokker_planck_phi1_collision_f_block_operator(
    *,
    layout: RHS1BlockLayout,
    fp_phi1_operator: Any,
    phi1_hat_base: Any,
    drop_tol: float = 0.0,
) -> RHS1BlockCOOOperator:
    """Assemble the frozen-Phi1 v3 Fokker-Planck collision term on the kinetic block.

    The ``includePhi1InCollisionOperator`` branch is linear in ``f`` only after
    the base ``Phi1Hat(theta,zeta)`` field is fixed.  With that field supplied,
    the term is diagonal in theta/zeta and Legendre index, dense in species and
    radial ``x``, and can be represented as block-COO without dense probing.
    """

    n_species = int(layout.n_species)
    n_x = int(layout.n_x)
    n_xi = int(layout.n_xi)
    n_theta = int(layout.n_theta)
    n_zeta = int(layout.n_zeta)
    if int(layout.f_size) != n_species * n_x * n_xi * n_theta * n_zeta:
        raise ValueError("layout f_size is inconsistent with kinetic dimensions")

    phi1 = np.asarray(phi1_hat_base, dtype=np.float64)
    expected_phi1_shape = (n_theta, n_zeta)
    if phi1.shape != expected_phi1_shape:
        raise ValueError(f"phi1_hat_base must have shape {expected_phi1_shape}, got {phi1.shape}")

    k_nu = np.asarray(fp_phi1_operator.k_nu, dtype=np.float64)
    expected_k_nu_shape = (n_species, n_species, n_x)
    if k_nu.shape != expected_k_nu_shape:
        raise ValueError(f"fp_phi1_operator.k_nu must have shape {expected_k_nu_shape}, got {k_nu.shape}")
    k_cd = np.asarray(fp_phi1_operator.k_cd, dtype=np.float64)
    expected_k_dense_shape = (n_species, n_species, n_x, n_x)
    if k_cd.shape != expected_k_dense_shape:
        raise ValueError(f"fp_phi1_operator.k_cd must have shape {expected_k_dense_shape}, got {k_cd.shape}")
    k_ce = np.asarray(fp_phi1_operator.k_ce, dtype=np.float64)
    if k_ce.shape != expected_k_dense_shape:
        raise ValueError(f"fp_phi1_operator.k_ce must have shape {expected_k_dense_shape}, got {k_ce.shape}")
    k_rosen = np.asarray(fp_phi1_operator.k_rosen, dtype=np.float64)
    nl_declared = int(fp_phi1_operator.nl)
    expected_k_rosen_shape = (n_species, n_species, nl_declared, n_x, n_x)
    if k_rosen.shape != expected_k_rosen_shape:
        raise ValueError(
            f"fp_phi1_operator.k_rosen must have shape {expected_k_rosen_shape}, got {k_rosen.shape}"
        )

    n_xi_for_x = np.asarray(fp_phi1_operator.n_xi_for_x, dtype=np.int32).reshape((-1,))
    if n_xi_for_x.shape != (n_x,):
        raise ValueError(f"fp_phi1_operator.n_xi_for_x must have shape {(n_x,)}, got {n_xi_for_x.shape}")
    if np.any(n_xi_for_x < 0) or np.any(n_xi_for_x > n_xi):
        raise ValueError("fp_phi1_operator.n_xi_for_x contains invalid active-L counts")

    z_s = np.asarray(fp_phi1_operator.z_s, dtype=np.float64).reshape((-1,))
    n_hats = np.asarray(fp_phi1_operator.n_hats, dtype=np.float64).reshape((-1,))
    t_hats = np.asarray(fp_phi1_operator.t_hats, dtype=np.float64).reshape((-1,))
    if z_s.shape != (n_species,) or n_hats.shape != (n_species,) or t_hats.shape != (n_species,):
        raise ValueError("fp_phi1_operator species arrays must have shape (n_species,)")

    alpha = float(np.asarray(fp_phi1_operator.alpha, dtype=np.float64).reshape(-1)[0])
    nu_n = float(np.asarray(fp_phi1_operator.nu_n, dtype=np.float64).reshape(-1)[0])
    krook = float(np.asarray(fp_phi1_operator.krook, dtype=np.float64).reshape(-1)[0])
    n_pol = n_hats[:, None, None] * np.exp(-(z_s[:, None, None] * alpha / t_hats[:, None, None]) * phi1[None, :, :])
    nu_d_hat = np.einsum("bTZ,abx->axTZ", n_pol, k_nu)

    builder = RHS1BlockCOOBuilder(shape=(int(layout.f_size), int(layout.f_size)), block_size=n_zeta, dtype=np.float64)
    threshold = float(drop_tol)
    nl_active = min(nl_declared, n_xi)
    for row_species in range(n_species):
        for row_x in range(n_x):
            for ell in range(min(n_xi, int(n_xi_for_x[row_x]))):
                factor_l = float(ell * (ell + 1) + 2.0 * krook)
                for theta in range(n_theta):
                    row_block = layout.kinetic_flat_index(
                        species=row_species,
                        x=row_x,
                        ell=ell,
                        theta=theta,
                        zeta=0,
                    ) // n_zeta
                    for col_species in range(n_species):
                        for col_x in range(n_x):
                            diagonal = (
                                -nu_n * k_cd[row_species, col_species, row_x, col_x] * n_pol[row_species, theta, :]
                            )
                            if col_species == row_species:
                                diagonal = diagonal - nu_n * np.einsum(
                                    "bz,b->z",
                                    n_pol[:, theta, :],
                                    k_ce[row_species, :, row_x, col_x],
                                )
                                if col_x == row_x:
                                    diagonal = diagonal + 0.5 * nu_n * nu_d_hat[row_species, row_x, theta, :] * factor_l
                            if ell < nl_active:
                                diagonal = diagonal - (
                                    nu_n
                                    * k_rosen[row_species, col_species, ell, row_x, col_x]
                                    * n_pol[row_species, theta, :]
                                )
                            max_abs = float(np.max(np.abs(diagonal)))
                            if max_abs == 0.0 or (threshold > 0.0 and max_abs <= threshold):
                                continue
                            col_block = layout.kinetic_flat_index(
                                species=col_species,
                                x=col_x,
                                ell=ell,
                                theta=theta,
                                zeta=0,
                            ) // n_zeta
                            builder.add_dense_block(row_block, col_block, np.diag(diagonal))
    return builder.build(drop_tol=threshold)


__all__ = [
    "build_fokker_planck_collision_f_block_operator",
    "build_fokker_planck_phi1_collision_f_block_operator",
    "build_pas_collision_f_block_operator",
]
