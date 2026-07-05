"""RHSMode=1 collisionless/drift-term assembly helpers for block operators."""

from __future__ import annotations

from typing import Any

import numpy as np

from sfincs_jax.operators.profile_layout import RHS1BlockCOOBuilder, RHS1BlockCOOOperator, RHS1BlockLayout


def _validate_common_f_layout(layout: RHS1BlockLayout) -> tuple[int, int, int, int, int]:
    n_species = int(layout.n_species)
    n_x = int(layout.n_x)
    n_xi = int(layout.n_xi)
    n_theta = int(layout.n_theta)
    n_zeta = int(layout.n_zeta)
    if int(layout.f_size) != n_species * n_x * n_xi * n_theta * n_zeta:
        raise ValueError("layout f_size is inconsistent with kinetic dimensions")
    return n_species, n_x, n_xi, n_theta, n_zeta


def _validate_n_xi_for_x(value: Any, *, n_x: int, n_xi: int, label: str) -> np.ndarray:
    n_xi_for_x = np.asarray(value, dtype=np.int32).reshape((-1,))
    if n_xi_for_x.shape != (n_x,):
        raise ValueError(f"{label}.n_xi_for_x must have shape {(n_x,)}, got {n_xi_for_x.shape}")
    if np.any(n_xi_for_x < 0) or np.any(n_xi_for_x > n_xi):
        raise ValueError(f"{label}.n_xi_for_x contains invalid active-L counts")
    return n_xi_for_x


def _validate_geometry_field(value: Any, *, shape: tuple[int, int], field_name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != shape:
        raise ValueError(f"{field_name} must have shape {shape}, got {arr.shape}")
    return arr


def _kinetic_zeta_block_id(
    layout: RHS1BlockLayout,
    *,
    species: int,
    x: int,
    ell: int,
    theta: int,
    n_zeta: int,
) -> int:
    """Return the block-COO zeta-line block id for one kinetic coordinate."""

    return layout.kinetic_flat_index(species=species, x=x, ell=ell, theta=theta, zeta=0) // int(n_zeta)


def _add_dense_if_nonzero(
    builder: RHS1BlockCOOBuilder,
    row_block: int,
    col_block: int,
    block: np.ndarray,
    *,
    threshold: float,
) -> None:
    """Add a zeta-line block after the same drop-tolerance check used for scalars."""

    if float(threshold) > 0.0 and not np.any(np.abs(block) > float(threshold)):
        return
    if not np.any(block != 0.0):
        return
    builder.add_dense_block(int(row_block), int(col_block), block)


def build_collisionless_f_block_operator(
    *,
    layout: RHS1BlockLayout,
    collisionless_operator: Any,
    drop_tol: float = 0.0,
) -> RHS1BlockCOOOperator:
    """Assemble the v3 collisionless streaming + mirror f-block term.

    This term is diagonal in species and radial index, couples neighboring
    Legendre/pitch rows, and carries both angular derivatives and mirror-force
    diagonal-in-angle couplings.  The returned block-COO operator uses zeta
    lines as the uniform block unit, so theta and Legendre couplings remain
    explicit while zeta derivatives are grouped into the same block map.
    """

    n_species, n_x, n_xi, n_theta, n_zeta = _validate_common_f_layout(layout)
    x = np.asarray(collisionless_operator.x, dtype=np.float64).reshape((-1,))
    if x.shape != (n_x,):
        raise ValueError(f"collisionless_operator.x must have shape {(n_x,)}, got {x.shape}")
    ddtheta = np.asarray(collisionless_operator.ddtheta, dtype=np.float64)
    if ddtheta.shape != (n_theta, n_theta):
        raise ValueError(f"collisionless_operator.ddtheta must have shape {(n_theta, n_theta)}, got {ddtheta.shape}")
    ddzeta = np.asarray(collisionless_operator.ddzeta, dtype=np.float64)
    if ddzeta.shape != (n_zeta, n_zeta):
        raise ValueError(f"collisionless_operator.ddzeta must have shape {(n_zeta, n_zeta)}, got {ddzeta.shape}")

    geometry_shape = (n_theta, n_zeta)
    b_hat = _validate_geometry_field(collisionless_operator.b_hat, shape=geometry_shape, field_name="b_hat")
    b_hat_sup_theta = _validate_geometry_field(
        collisionless_operator.b_hat_sup_theta,
        shape=geometry_shape,
        field_name="b_hat_sup_theta",
    )
    b_hat_sup_zeta = _validate_geometry_field(
        collisionless_operator.b_hat_sup_zeta,
        shape=geometry_shape,
        field_name="b_hat_sup_zeta",
    )
    db_hat_dtheta = _validate_geometry_field(
        collisionless_operator.db_hat_dtheta,
        shape=geometry_shape,
        field_name="db_hat_dtheta",
    )
    db_hat_dzeta = _validate_geometry_field(
        collisionless_operator.db_hat_dzeta,
        shape=geometry_shape,
        field_name="db_hat_dzeta",
    )
    if np.any(b_hat == 0.0):
        raise ValueError("collisionless_operator.b_hat contains zeros")

    t_hats = np.asarray(collisionless_operator.t_hats, dtype=np.float64).reshape((-1,))
    m_hats = np.asarray(collisionless_operator.m_hats, dtype=np.float64).reshape((-1,))
    if t_hats.shape != (n_species,):
        raise ValueError(f"collisionless_operator.t_hats must have shape {(n_species,)}, got {t_hats.shape}")
    if m_hats.shape != (n_species,):
        raise ValueError(f"collisionless_operator.m_hats must have shape {(n_species,)}, got {m_hats.shape}")
    if np.any(t_hats < 0.0) or np.any(m_hats <= 0.0):
        raise ValueError("collisionless_operator requires nonnegative t_hats and positive m_hats")
    n_xi_for_x = _validate_n_xi_for_x(
        collisionless_operator.n_xi_for_x,
        n_x=n_x,
        n_xi=n_xi,
        label="collisionless_operator",
    )

    ell = np.arange(n_xi, dtype=np.float64)
    coef_plus = (ell + 1.0) / (2.0 * ell + 3.0)
    coef_minus = np.where(ell > 0, ell / (2.0 * ell - 1.0), 0.0)
    coef_mirror_plus = (ell + 1.0) * (ell + 2.0) / (2.0 * ell + 3.0)
    coef_mirror_minus = np.where(ell > 1, -ell * (ell - 1.0) / (2.0 * ell - 1.0), 0.0)

    sqrt_t_over_m = np.sqrt(t_hats / m_hats)
    v_theta = b_hat_sup_theta / b_hat
    v_zeta = b_hat_sup_zeta / b_hat
    mirror_geom = b_hat_sup_theta * db_hat_dtheta + b_hat_sup_zeta * db_hat_dzeta

    builder = RHS1BlockCOOBuilder(shape=(int(layout.f_size), int(layout.f_size)), block_size=n_zeta, dtype=np.float64)
    threshold = float(drop_tol)

    for species in range(n_species):
        species_speed = float(sqrt_t_over_m[species])
        for ix in range(n_x):
            active_l = min(n_xi, int(n_xi_for_x[ix]))
            for row_l in range(active_l):
                legendre_columns = (
                    (row_l + 1, float(x[ix] * coef_plus[row_l])),
                    (row_l - 1, float(x[ix] * coef_minus[row_l])),
                )
                mirror_columns = (
                    (row_l + 1, float(x[ix] * coef_mirror_plus[row_l])),
                    (row_l - 1, float(x[ix] * coef_mirror_minus[row_l])),
                )
                for col_l, legendre_coef in legendre_columns:
                    if col_l < 0 or col_l >= active_l or legendre_coef == 0.0:
                        continue
                    scale = float(legendre_coef * species_speed)
                    for row_theta in range(n_theta):
                        row_block = _kinetic_zeta_block_id(
                            layout,
                            species=species,
                            x=ix,
                            ell=row_l,
                            theta=row_theta,
                            n_zeta=n_zeta,
                        )
                        zeta_block = (scale * v_zeta[row_theta, :])[:, None] * ddzeta
                        col_block = _kinetic_zeta_block_id(
                            layout,
                            species=species,
                            x=ix,
                            ell=col_l,
                            theta=row_theta,
                            n_zeta=n_zeta,
                        )
                        _add_dense_if_nonzero(
                            builder,
                            row_block,
                            col_block,
                            zeta_block,
                            threshold=threshold,
                        )
                        for col_theta in range(n_theta):
                            dd_value = float(ddtheta[row_theta, col_theta])
                            if dd_value == 0.0 or (threshold > 0.0 and abs(dd_value) <= threshold):
                                continue
                            col_block = _kinetic_zeta_block_id(
                                layout,
                                species=species,
                                x=ix,
                                ell=col_l,
                                theta=col_theta,
                                n_zeta=n_zeta,
                            )
                            diagonal = scale * dd_value * v_theta[row_theta, :]
                            _add_dense_if_nonzero(
                                builder,
                                row_block,
                                col_block,
                                np.diag(diagonal),
                                threshold=threshold,
                            )

                for col_l, mirror_coef in mirror_columns:
                    if col_l < 0 or col_l >= active_l or mirror_coef == 0.0:
                        continue
                    for theta in range(n_theta):
                        row_block = _kinetic_zeta_block_id(
                            layout,
                            species=species,
                            x=ix,
                            ell=row_l,
                            theta=theta,
                            n_zeta=n_zeta,
                        )
                        col_block = _kinetic_zeta_block_id(
                            layout,
                            species=species,
                            x=ix,
                            ell=col_l,
                            theta=theta,
                            n_zeta=n_zeta,
                        )
                        mirror_factor = -species_speed * mirror_geom[theta, :] / (2.0 * b_hat[theta, :] ** 2)
                        _add_dense_if_nonzero(
                            builder,
                            row_block,
                            col_block,
                            np.diag(float(mirror_coef) * mirror_factor),
                            threshold=threshold,
                        )

    return builder.build(drop_tol=threshold)


def build_exb_theta_f_block_operator(
    *,
    layout: RHS1BlockLayout,
    exb_theta_operator: Any,
    drop_tol: float = 0.0,
) -> RHS1BlockCOOOperator:
    """Assemble the ExB ``d/dtheta`` f-block term as uniform block-COO.

    The term is off-diagonal in theta through ``ddtheta`` and diagonal in
    species, radial index, Legendre index, and zeta.  With ``block_size = Nzeta``
    each matrix block represents one fixed ``(species, x, L, theta)`` zeta line.
    """

    n_species, n_x, n_xi, n_theta, n_zeta = _validate_common_f_layout(layout)

    ddtheta = np.asarray(exb_theta_operator.ddtheta, dtype=np.float64)
    if ddtheta.shape != (n_theta, n_theta):
        raise ValueError(f"exb_theta_operator.ddtheta must have shape {(n_theta, n_theta)}, got {ddtheta.shape}")
    d_hat = np.asarray(exb_theta_operator.d_hat, dtype=np.float64)
    b_hat = np.asarray(exb_theta_operator.b_hat, dtype=np.float64)
    b_hat_sub_zeta = np.asarray(exb_theta_operator.b_hat_sub_zeta, dtype=np.float64)
    geometry_shape = (n_theta, n_zeta)
    for name, value in (
        ("d_hat", d_hat),
        ("b_hat", b_hat),
        ("b_hat_sub_zeta", b_hat_sub_zeta),
    ):
        if value.shape != geometry_shape:
            raise ValueError(f"exb_theta_operator.{name} must have shape {geometry_shape}, got {value.shape}")
    n_xi_for_x = _validate_n_xi_for_x(exb_theta_operator.n_xi_for_x, n_x=n_x, n_xi=n_xi, label="exb_theta_operator")

    use_dkes = bool(exb_theta_operator.use_dkes_exb_drift)
    if use_dkes:
        denom = float(np.asarray(exb_theta_operator.fsab_hat2, dtype=np.float64).reshape(-1)[0])
        if not np.isfinite(denom) or abs(denom) <= 0.0:
            raise ValueError("exb_theta_operator.fsab_hat2 must be finite and nonzero")
        coef = d_hat * b_hat_sub_zeta / denom
    else:
        if np.any(b_hat == 0.0):
            raise ValueError("exb_theta_operator.b_hat contains zeros")
        coef = d_hat * b_hat_sub_zeta / (b_hat * b_hat)

    factor = (
        float(np.asarray(exb_theta_operator.alpha, dtype=np.float64).reshape(-1)[0])
        * float(np.asarray(exb_theta_operator.delta, dtype=np.float64).reshape(-1)[0])
        * 0.5
        * float(np.asarray(exb_theta_operator.dphi_hat_dpsi_hat, dtype=np.float64).reshape(-1)[0])
    )

    builder = RHS1BlockCOOBuilder(shape=(int(layout.f_size), int(layout.f_size)), block_size=n_zeta, dtype=np.float64)
    threshold = float(drop_tol)
    for species in range(n_species):
        for ix in range(n_x):
            for ell in range(min(n_xi, int(n_xi_for_x[ix]))):
                for row_theta in range(n_theta):
                    row_block = layout.kinetic_flat_index(
                        species=species,
                        x=ix,
                        ell=ell,
                        theta=row_theta,
                        zeta=0,
                    ) // n_zeta
                    for col_theta in range(n_theta):
                        dd_value = float(ddtheta[row_theta, col_theta])
                        if threshold > 0.0 and abs(dd_value) <= threshold:
                            continue
                        diagonal = factor * dd_value * coef[row_theta, :]
                        if threshold > 0.0 and not np.any(np.abs(diagonal) > threshold):
                            continue
                        col_block = layout.kinetic_flat_index(
                            species=species,
                            x=ix,
                            ell=ell,
                            theta=col_theta,
                            zeta=0,
                        ) // n_zeta
                        builder.add_dense_block(row_block, col_block, np.diag(diagonal))
    return builder.build(drop_tol=threshold)


def build_exb_zeta_f_block_operator(
    *,
    layout: RHS1BlockLayout,
    exb_zeta_operator: Any,
    drop_tol: float = 0.0,
) -> RHS1BlockCOOOperator:
    """Assemble the ExB ``d/dzeta`` f-block term as uniform block-COO.

    Unlike the theta term, zeta derivatives couple entries inside each
    ``Nzeta`` block.  We therefore add scalar COO entries into the same
    zeta-line block structure rather than dense theta-to-theta blocks.
    """

    n_species, n_x, n_xi, n_theta, n_zeta = _validate_common_f_layout(layout)

    ddzeta = np.asarray(exb_zeta_operator.ddzeta, dtype=np.float64)
    if ddzeta.shape != (n_zeta, n_zeta):
        raise ValueError(f"exb_zeta_operator.ddzeta must have shape {(n_zeta, n_zeta)}, got {ddzeta.shape}")
    d_hat = np.asarray(exb_zeta_operator.d_hat, dtype=np.float64)
    b_hat = np.asarray(exb_zeta_operator.b_hat, dtype=np.float64)
    b_hat_sub_theta = np.asarray(exb_zeta_operator.b_hat_sub_theta, dtype=np.float64)
    geometry_shape = (n_theta, n_zeta)
    for name, value in (
        ("d_hat", d_hat),
        ("b_hat", b_hat),
        ("b_hat_sub_theta", b_hat_sub_theta),
    ):
        if value.shape != geometry_shape:
            raise ValueError(f"exb_zeta_operator.{name} must have shape {geometry_shape}, got {value.shape}")
    n_xi_for_x = _validate_n_xi_for_x(exb_zeta_operator.n_xi_for_x, n_x=n_x, n_xi=n_xi, label="exb_zeta_operator")

    use_dkes = bool(exb_zeta_operator.use_dkes_exb_drift)
    if use_dkes:
        denom = float(np.asarray(exb_zeta_operator.fsab_hat2, dtype=np.float64).reshape(-1)[0])
        if not np.isfinite(denom) or abs(denom) <= 0.0:
            raise ValueError("exb_zeta_operator.fsab_hat2 must be finite and nonzero")
        coef = d_hat * b_hat_sub_theta / denom
    else:
        if np.any(b_hat == 0.0):
            raise ValueError("exb_zeta_operator.b_hat contains zeros")
        coef = d_hat * b_hat_sub_theta / (b_hat * b_hat)

    factor = (
        -float(np.asarray(exb_zeta_operator.alpha, dtype=np.float64).reshape(-1)[0])
        * float(np.asarray(exb_zeta_operator.delta, dtype=np.float64).reshape(-1)[0])
        * 0.5
        * float(np.asarray(exb_zeta_operator.dphi_hat_dpsi_hat, dtype=np.float64).reshape(-1)[0])
    )

    builder = RHS1BlockCOOBuilder(shape=(int(layout.f_size), int(layout.f_size)), block_size=n_zeta, dtype=np.float64)
    threshold = float(drop_tol)
    for species in range(n_species):
        for ix in range(n_x):
            for ell in range(min(n_xi, int(n_xi_for_x[ix]))):
                for theta in range(n_theta):
                    block_id = _kinetic_zeta_block_id(
                        layout,
                        species=species,
                        x=ix,
                        ell=ell,
                        theta=theta,
                        n_zeta=n_zeta,
                    )
                    block_value = (factor * coef[theta, :])[:, None] * ddzeta
                    _add_dense_if_nonzero(builder, block_id, block_id, block_value, threshold=threshold)
    return builder.build(drop_tol=threshold)


def build_er_xidot_f_block_operator(
    *,
    layout: RHS1BlockLayout,
    er_xidot_operator: Any,
    drop_tol: float = 0.0,
) -> RHS1BlockCOOOperator:
    """Assemble the v3 electric-field ``xi-dot`` f-block term.

    The term is diagonal in species, radial index, theta, and zeta, with
    diagonal and ``L +/- 2`` pitch couplings.  It follows
    ``apply_er_xidot_v3`` exactly: invalid pitch rows are masked, but invalid
    padded pitch columns are not zeroed before applying the operator.
    """

    n_species, n_x, n_xi, n_theta, n_zeta = _validate_common_f_layout(layout)
    n_xi_for_x = _validate_n_xi_for_x(er_xidot_operator.n_xi_for_x, n_x=n_x, n_xi=n_xi, label="er_xidot_operator")
    factor = _er_geometry_factor(
        er_xidot_operator,
        n_theta=n_theta,
        n_zeta=n_zeta,
        sign=1.0,
        label="er_xidot_operator",
    )

    ell = np.arange(n_xi, dtype=np.float64)
    denom0 = (2.0 * ell - 1.0) * (2.0 * ell + 3.0)
    diag_coef = (ell + 1.0) * ell / denom0
    sup2_coef = (ell + 3.0) * (ell + 2.0) * (ell + 1.0) / ((2.0 * ell + 5.0) * (2.0 * ell + 3.0))
    sub2_coef = -ell * (ell - 1.0) * (ell - 2.0) / ((2.0 * ell - 3.0) * (2.0 * ell - 1.0))
    drop_l2 = bool(getattr(er_xidot_operator, "drop_l2_couplings", False))

    builder = RHS1BlockCOOBuilder(shape=(int(layout.f_size), int(layout.f_size)), block_size=n_zeta, dtype=np.float64)
    threshold = float(drop_tol)

    for species in range(n_species):
        for ix in range(n_x):
            for row_l in range(min(n_xi, int(n_xi_for_x[ix]))):
                for theta in range(n_theta):
                    row_block = _kinetic_zeta_block_id(
                        layout,
                        species=species,
                        x=ix,
                        ell=row_l,
                        theta=theta,
                        n_zeta=n_zeta,
                    )
                    for col_l, coef_l in (
                        (row_l, float(diag_coef[row_l])),
                        (row_l + 2, 0.0 if drop_l2 else float(sup2_coef[row_l])),
                        (row_l - 2, 0.0 if drop_l2 else float(sub2_coef[row_l])),
                    ):
                        if col_l < 0 or col_l >= n_xi or coef_l == 0.0:
                            continue
                        col_block = _kinetic_zeta_block_id(
                            layout,
                            species=species,
                            x=ix,
                            ell=col_l,
                            theta=theta,
                            n_zeta=n_zeta,
                        )
                        _add_dense_if_nonzero(
                            builder,
                            row_block,
                            col_block,
                            np.diag(coef_l * factor[theta, :]),
                            threshold=threshold,
                        )
    return builder.build(drop_tol=threshold)


def build_er_xdot_f_block_operator(
    *,
    layout: RHS1BlockLayout,
    er_xdot_operator: Any,
    drop_tol: float = 0.0,
) -> RHS1BlockCOOOperator:
    """Assemble the v3 electric-field ``x-dot`` f-block term.

    The term is diagonal in species, theta, and zeta, dense in radial ``x``
    through the configured derivative matrix, and couples pitch rows through
    diagonal and ``L +/- 2`` coefficients.  Upwind matrix selection follows the
    sign of the local ``xDot`` factor, matching ``apply_er_xdot_v3``.
    """

    n_species, n_x, n_xi, n_theta, n_zeta = _validate_common_f_layout(layout)
    n_xi_for_x = _validate_n_xi_for_x(er_xdot_operator.n_xi_for_x, n_x=n_x, n_xi=n_xi, label="er_xdot_operator")
    x = np.asarray(er_xdot_operator.x, dtype=np.float64).reshape((-1,))
    if x.shape != (n_x,):
        raise ValueError(f"er_xdot_operator.x must have shape {(n_x,)}, got {x.shape}")
    ddx_plus = np.asarray(er_xdot_operator.ddx_plus, dtype=np.float64)
    ddx_minus = np.asarray(er_xdot_operator.ddx_minus, dtype=np.float64)
    for name, value in (("ddx_plus", ddx_plus), ("ddx_minus", ddx_minus)):
        if value.shape != (n_x, n_x):
            raise ValueError(f"er_xdot_operator.{name} must have shape {(n_x, n_x)}, got {value.shape}")
    x_part_plus = x[:, None] * ddx_plus
    x_part_minus = x[:, None] * ddx_minus
    xdot_factor = _er_geometry_factor(
        er_xdot_operator,
        n_theta=n_theta,
        n_zeta=n_zeta,
        sign=1.0,
        label="er_xdot_operator",
    )

    ell = np.arange(n_xi, dtype=np.float64)
    denom = (2.0 * ell + 3.0) * (2.0 * ell - 1.0)
    diag_coef = 2.0 * (3.0 * ell * ell + 3.0 * ell - 2.0) / denom
    sup_coef = np.zeros((n_xi,), dtype=np.float64)
    sub_coef = np.zeros((n_xi,), dtype=np.float64)
    if n_xi >= 3:
        l0 = ell[:-2]
        sup_coef[:-2] = (l0 + 1.0) * (l0 + 2.0) / ((2.0 * l0 + 5.0) * (2.0 * l0 + 3.0))
        l2 = ell[2:]
        sub_coef[2:] = l2 * (l2 - 1.0) / ((2.0 * l2 - 3.0) * (2.0 * l2 - 1.0))
    drop_l2 = bool(getattr(er_xdot_operator, "drop_l2_couplings", False))

    builder = RHS1BlockCOOBuilder(shape=(int(layout.f_size), int(layout.f_size)), block_size=n_zeta, dtype=np.float64)
    threshold = float(drop_tol)

    for species in range(n_species):
        for row_x in range(n_x):
            for row_l in range(min(n_xi, int(n_xi_for_x[row_x]))):
                pitch_columns = [(row_l, float(diag_coef[row_l]))]
                if not drop_l2:
                    pitch_columns.extend(
                        (
                            (row_l + 2, float(sup_coef[row_l])),
                            (row_l - 2, float(sub_coef[row_l])),
                        )
                    )
                for theta in range(n_theta):
                    row_block = _kinetic_zeta_block_id(
                        layout,
                        species=species,
                        x=row_x,
                        ell=row_l,
                        theta=theta,
                        n_zeta=n_zeta,
                    )
                    positive = xdot_factor[theta, :] > 0.0
                    for col_l, pitch_coef in pitch_columns:
                        if col_l < 0 or col_l >= n_xi or pitch_coef == 0.0:
                            continue
                        for col_x in range(n_x):
                            coef_zeta = np.where(
                                positive,
                                x_part_plus[row_x, col_x] * xdot_factor[theta, :],
                                x_part_minus[row_x, col_x] * xdot_factor[theta, :],
                            )
                            diagonal = float(pitch_coef) * coef_zeta
                            col_block = _kinetic_zeta_block_id(
                                layout,
                                species=species,
                                x=col_x,
                                ell=col_l,
                                theta=theta,
                                n_zeta=n_zeta,
                            )
                            _add_dense_if_nonzero(
                                builder,
                                row_block,
                                col_block,
                                np.diag(diagonal),
                                threshold=threshold,
                            )
    return builder.build(drop_tol=threshold)


def build_magnetic_drift_xidot_f_block_operator(
    *,
    layout: RHS1BlockLayout,
    magdrift_xidot_operator: Any,
    drop_tol: float = 0.0,
) -> RHS1BlockCOOOperator:
    """Assemble the v3 magnetic-drift ``xi-dot`` f-block term.

    The production evaluator masks invalid pitch columns before applying this
    operator.  The block assembly therefore restricts both rows and same-``x``
    pitch columns to the active ``n_xi_for_x`` range.
    """

    n_species, n_x, n_xi, n_theta, n_zeta = _validate_common_f_layout(layout)
    n_xi_for_x = _validate_n_xi_for_x(
        magdrift_xidot_operator.n_xi_for_x,
        n_x=n_x,
        n_xi=n_xi,
        label="magdrift_xidot_operator",
    )
    x = np.asarray(magdrift_xidot_operator.x, dtype=np.float64).reshape((-1,))
    if x.shape != (n_x,):
        raise ValueError(f"magdrift_xidot_operator.x must have shape {(n_x,)}, got {x.shape}")
    factor = _magnetic_xidot_geometry_factor(magdrift_xidot_operator, n_theta=n_theta, n_zeta=n_zeta)

    ell = np.arange(n_xi, dtype=np.float64)
    diag_coef = np.where(ell > 0, (ell + 1.0) * ell / ((2.0 * ell - 1.0) * (2.0 * ell + 3.0)), 0.0)
    sup2_coef = (ell + 3.0) * (ell + 2.0) * (ell + 1.0) / ((2.0 * ell + 5.0) * (2.0 * ell + 3.0))
    sub2_coef = np.where(ell > 1, -ell * (ell - 1.0) * (ell - 2.0) / ((2.0 * ell - 3.0) * (2.0 * ell - 1.0)), 0.0)
    drop_l2 = bool(getattr(magdrift_xidot_operator, "drop_l2_couplings", False))

    builder = RHS1BlockCOOBuilder(shape=(int(layout.f_size), int(layout.f_size)), block_size=n_zeta, dtype=np.float64)
    threshold = float(drop_tol)

    for species in range(n_species):
        for ix in range(n_x):
            active_l = min(n_xi, int(n_xi_for_x[ix]))
            x2 = float(x[ix] * x[ix])
            for row_l in range(active_l):
                pitch_columns = [(row_l, float(diag_coef[row_l]))]
                if not drop_l2:
                    pitch_columns.extend(
                        (
                            (row_l + 2, float(sup2_coef[row_l])),
                            (row_l - 2, float(sub2_coef[row_l])),
                        )
                    )
                for theta in range(n_theta):
                    row_block = _kinetic_zeta_block_id(
                        layout,
                        species=species,
                        x=ix,
                        ell=row_l,
                        theta=theta,
                        n_zeta=n_zeta,
                    )
                    for col_l, pitch_coef in pitch_columns:
                        if col_l < 0 or col_l >= active_l or pitch_coef == 0.0:
                            continue
                        col_block = _kinetic_zeta_block_id(
                            layout,
                            species=species,
                            x=ix,
                            ell=col_l,
                            theta=theta,
                            n_zeta=n_zeta,
                        )
                        _add_dense_if_nonzero(
                            builder,
                            row_block,
                            col_block,
                            np.diag(x2 * float(pitch_coef) * factor[theta, :]),
                            threshold=threshold,
                        )
    return builder.build(drop_tol=threshold)


def build_magnetic_drift_theta_f_block_operator(
    *,
    layout: RHS1BlockLayout,
    magdrift_theta_operator: Any,
    drop_tol: float = 0.0,
) -> RHS1BlockCOOOperator:
    """Assemble the v3 magnetic-drift ``d/dtheta`` f-block term."""

    n_species, n_x, n_xi, n_theta, n_zeta = _validate_common_f_layout(layout)
    n_xi_for_x = _validate_n_xi_for_x(
        magdrift_theta_operator.n_xi_for_x,
        n_x=n_x,
        n_xi=n_xi,
        label="magdrift_theta_operator",
    )
    x = np.asarray(magdrift_theta_operator.x, dtype=np.float64).reshape((-1,))
    if x.shape != (n_x,):
        raise ValueError(f"magdrift_theta_operator.x must have shape {(n_x,)}, got {x.shape}")
    ddtheta_plus = np.asarray(magdrift_theta_operator.ddtheta_plus, dtype=np.float64)
    ddtheta_minus = np.asarray(magdrift_theta_operator.ddtheta_minus, dtype=np.float64)
    for name, value in (("ddtheta_plus", ddtheta_plus), ("ddtheta_minus", ddtheta_minus)):
        if value.shape != (n_theta, n_theta):
            raise ValueError(f"magdrift_theta_operator.{name} must have shape {(n_theta, n_theta)}, got {value.shape}")

    gf1, gf2, base = _magnetic_theta_geometry_factors(magdrift_theta_operator, n_theta=n_theta, n_zeta=n_zeta)
    gf12 = gf1 + gf2
    use_plus = (gf1 * float(np.asarray(magdrift_theta_operator.d_hat, dtype=np.float64)[0, 0]) / _charge_scalar(magdrift_theta_operator, "magdrift_theta_operator")) > 0.0
    c1, c2, c3 = _magnetic_diag_coefficients(n_xi)
    del c3  # magneticDriftScheme=1 has geometricFactor3=0 in the production evaluator.
    c_plus = _magnetic_offdiag2_plus(n_xi)
    c_minus = _magnetic_offdiag2_minus(n_xi)
    drop_l2 = bool(getattr(magdrift_theta_operator, "drop_l2_couplings", False))

    builder = RHS1BlockCOOBuilder(shape=(int(layout.f_size), int(layout.f_size)), block_size=n_zeta, dtype=np.float64)
    threshold = float(drop_tol)

    for species in range(n_species):
        for ix in range(n_x):
            active_l = min(n_xi, int(n_xi_for_x[ix]))
            x2 = float(x[ix] * x[ix])
            for row_l in range(active_l):
                pitch_columns = [(row_l, float(c1[row_l]), float(c2[row_l]), 0.0)]
                if not drop_l2:
                    pitch_columns.extend(
                        (
                            (row_l + 2, 0.0, 0.0, float(c_plus[row_l])),
                            (row_l - 2, 0.0, 0.0, float(c_minus[row_l])),
                        )
                    )
                for row_theta in range(n_theta):
                    row_block = _kinetic_zeta_block_id(
                        layout,
                        species=species,
                        x=ix,
                        ell=row_l,
                        theta=row_theta,
                        n_zeta=n_zeta,
                    )
                    derivative_rows = np.where(
                        use_plus[row_theta, :, None],
                        ddtheta_plus[row_theta, :][None, :],
                        ddtheta_minus[row_theta, :][None, :],
                    )
                    for col_l, diag_c1, diag_c2, offdiag_c in pitch_columns:
                        if col_l < 0 or col_l >= active_l:
                            continue
                        if col_l == row_l:
                            pitch_geometry = diag_c1 * gf1[row_theta, :] + diag_c2 * gf2[row_theta, :]
                        else:
                            pitch_geometry = offdiag_c * gf12[row_theta, :]
                        if not np.any(pitch_geometry != 0.0):
                            continue
                        base_pitch = x2 * base[row_theta, :] * pitch_geometry
                        for col_theta in range(n_theta):
                            dd_values = derivative_rows[:, col_theta]
                            if not np.any(dd_values != 0.0):
                                continue
                            col_block = _kinetic_zeta_block_id(
                                layout,
                                species=species,
                                x=ix,
                                ell=col_l,
                                theta=col_theta,
                                n_zeta=n_zeta,
                            )
                            _add_dense_if_nonzero(
                                builder,
                                row_block,
                                col_block,
                                np.diag(base_pitch * dd_values),
                                threshold=threshold,
                            )
    return builder.build(drop_tol=threshold)


def build_magnetic_drift_zeta_f_block_operator(
    *,
    layout: RHS1BlockLayout,
    magdrift_zeta_operator: Any,
    drop_tol: float = 0.0,
) -> RHS1BlockCOOOperator:
    """Assemble the v3 magnetic-drift ``d/dzeta`` f-block term."""

    n_species, n_x, n_xi, n_theta, n_zeta = _validate_common_f_layout(layout)
    n_xi_for_x = _validate_n_xi_for_x(
        magdrift_zeta_operator.n_xi_for_x,
        n_x=n_x,
        n_xi=n_xi,
        label="magdrift_zeta_operator",
    )
    x = np.asarray(magdrift_zeta_operator.x, dtype=np.float64).reshape((-1,))
    if x.shape != (n_x,):
        raise ValueError(f"magdrift_zeta_operator.x must have shape {(n_x,)}, got {x.shape}")
    ddzeta_plus = np.asarray(magdrift_zeta_operator.ddzeta_plus, dtype=np.float64)
    ddzeta_minus = np.asarray(magdrift_zeta_operator.ddzeta_minus, dtype=np.float64)
    for name, value in (("ddzeta_plus", ddzeta_plus), ("ddzeta_minus", ddzeta_minus)):
        if value.shape != (n_zeta, n_zeta):
            raise ValueError(f"magdrift_zeta_operator.{name} must have shape {(n_zeta, n_zeta)}, got {value.shape}")

    gf1, gf2, base = _magnetic_zeta_geometry_factors(magdrift_zeta_operator, n_theta=n_theta, n_zeta=n_zeta)
    gf12 = gf1 + gf2
    use_plus = (gf1 * float(np.asarray(magdrift_zeta_operator.d_hat, dtype=np.float64)[0, 0]) / _charge_scalar(magdrift_zeta_operator, "magdrift_zeta_operator")) > 0.0
    c1, c2, c3 = _magnetic_diag_coefficients(n_xi)
    del c3
    c_plus = _magnetic_offdiag2_plus(n_xi)
    c_minus = _magnetic_offdiag2_minus(n_xi)
    drop_l2 = bool(getattr(magdrift_zeta_operator, "drop_l2_couplings", False))

    builder = RHS1BlockCOOBuilder(shape=(int(layout.f_size), int(layout.f_size)), block_size=n_zeta, dtype=np.float64)
    threshold = float(drop_tol)

    for species in range(n_species):
        for ix in range(n_x):
            active_l = min(n_xi, int(n_xi_for_x[ix]))
            x2 = float(x[ix] * x[ix])
            for row_l in range(active_l):
                pitch_columns = [(row_l, float(c1[row_l]), float(c2[row_l]), 0.0)]
                if not drop_l2:
                    pitch_columns.extend(
                        (
                            (row_l + 2, 0.0, 0.0, float(c_plus[row_l])),
                            (row_l - 2, 0.0, 0.0, float(c_minus[row_l])),
                        )
                    )
                for theta in range(n_theta):
                    row_block = _kinetic_zeta_block_id(
                        layout,
                        species=species,
                        x=ix,
                        ell=row_l,
                        theta=theta,
                        n_zeta=n_zeta,
                    )
                    derivative = np.where(use_plus[theta, :, None], ddzeta_plus, ddzeta_minus)
                    for col_l, diag_c1, diag_c2, offdiag_c in pitch_columns:
                        if col_l < 0 or col_l >= active_l:
                            continue
                        if col_l == row_l:
                            pitch_geometry = diag_c1 * gf1[theta, :] + diag_c2 * gf2[theta, :]
                        else:
                            pitch_geometry = offdiag_c * gf12[theta, :]
                        if not np.any(pitch_geometry != 0.0):
                            continue
                        block_value = (x2 * base[theta, :] * pitch_geometry)[:, None] * derivative
                        col_block = _kinetic_zeta_block_id(
                            layout,
                            species=species,
                            x=ix,
                            ell=col_l,
                            theta=theta,
                            n_zeta=n_zeta,
                        )
                        _add_dense_if_nonzero(builder, row_block, col_block, block_value, threshold=threshold)
    return builder.build(drop_tol=threshold)


def _er_geometry_factor(
    operator: Any,
    *,
    n_theta: int,
    n_zeta: int,
    sign: float,
    label: str,
) -> np.ndarray:
    geometry_shape = (int(n_theta), int(n_zeta))
    d_hat = _validate_geometry_field(operator.d_hat, shape=geometry_shape, field_name=f"{label}.d_hat")
    b_hat = _validate_geometry_field(operator.b_hat, shape=geometry_shape, field_name=f"{label}.b_hat")
    b_hat_sub_theta = _validate_geometry_field(
        operator.b_hat_sub_theta,
        shape=geometry_shape,
        field_name=f"{label}.b_hat_sub_theta",
    )
    b_hat_sub_zeta = _validate_geometry_field(
        operator.b_hat_sub_zeta,
        shape=geometry_shape,
        field_name=f"{label}.b_hat_sub_zeta",
    )
    db_hat_dtheta = _validate_geometry_field(
        operator.db_hat_dtheta,
        shape=geometry_shape,
        field_name=f"{label}.db_hat_dtheta",
    )
    db_hat_dzeta = _validate_geometry_field(
        operator.db_hat_dzeta,
        shape=geometry_shape,
        field_name=f"{label}.db_hat_dzeta",
    )
    if np.any(b_hat == 0.0):
        raise ValueError(f"{label}.b_hat contains zeros")
    temp = b_hat_sub_zeta * db_hat_dtheta - b_hat_sub_theta * db_hat_dzeta
    return (
        float(sign)
        * float(np.asarray(operator.alpha, dtype=np.float64).reshape(-1)[0])
        * float(np.asarray(operator.delta, dtype=np.float64).reshape(-1)[0])
        * float(np.asarray(operator.dphi_hat_dpsi_hat, dtype=np.float64).reshape(-1)[0])
        / 4.0
        * d_hat
        * temp
        / (b_hat**3)
    )


def _magnetic_xidot_geometry_factor(operator: Any, *, n_theta: int, n_zeta: int) -> np.ndarray:
    geometry_shape = (int(n_theta), int(n_zeta))
    d_hat = _validate_geometry_field(operator.d_hat, shape=geometry_shape, field_name="magdrift_xidot_operator.d_hat")
    b_hat = _validate_geometry_field(operator.b_hat, shape=geometry_shape, field_name="magdrift_xidot_operator.b_hat")
    db_hat_dtheta = _validate_geometry_field(
        operator.db_hat_dtheta,
        shape=geometry_shape,
        field_name="magdrift_xidot_operator.db_hat_dtheta",
    )
    db_hat_dzeta = _validate_geometry_field(
        operator.db_hat_dzeta,
        shape=geometry_shape,
        field_name="magdrift_xidot_operator.db_hat_dzeta",
    )
    db_hat_sub_psi_dzeta = _validate_geometry_field(
        operator.db_hat_sub_psi_dzeta,
        shape=geometry_shape,
        field_name="magdrift_xidot_operator.db_hat_sub_psi_dzeta",
    )
    db_hat_sub_zeta_dpsi_hat = _validate_geometry_field(
        operator.db_hat_sub_zeta_dpsi_hat,
        shape=geometry_shape,
        field_name="magdrift_xidot_operator.db_hat_sub_zeta_dpsi_hat",
    )
    db_hat_sub_theta_dpsi_hat = _validate_geometry_field(
        operator.db_hat_sub_theta_dpsi_hat,
        shape=geometry_shape,
        field_name="magdrift_xidot_operator.db_hat_sub_theta_dpsi_hat",
    )
    db_hat_sub_psi_dtheta = _validate_geometry_field(
        operator.db_hat_sub_psi_dtheta,
        shape=geometry_shape,
        field_name="magdrift_xidot_operator.db_hat_sub_psi_dtheta",
    )
    if np.any(b_hat == 0.0):
        raise ValueError("magdrift_xidot_operator.b_hat contains zeros")
    temp = (db_hat_sub_psi_dzeta - db_hat_sub_zeta_dpsi_hat) * db_hat_dtheta
    temp += (db_hat_sub_theta_dpsi_hat - db_hat_sub_psi_dtheta) * db_hat_dzeta
    return (
        -float(np.asarray(operator.delta, dtype=np.float64).reshape(-1)[0])
        * float(np.asarray(operator.t_hat, dtype=np.float64).reshape(-1)[0])
        / (2.0 * float(np.asarray(operator.z, dtype=np.float64).reshape(-1)[0]))
        * d_hat
        * temp
        / (b_hat**3)
    )


def _magnetic_theta_geometry_factors(operator: Any, *, n_theta: int, n_zeta: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    geometry_shape = (int(n_theta), int(n_zeta))
    d_hat = _validate_geometry_field(operator.d_hat, shape=geometry_shape, field_name="magdrift_theta_operator.d_hat")
    b_hat = _validate_geometry_field(operator.b_hat, shape=geometry_shape, field_name="magdrift_theta_operator.b_hat")
    b_hat_sub_zeta = _validate_geometry_field(
        operator.b_hat_sub_zeta,
        shape=geometry_shape,
        field_name="magdrift_theta_operator.b_hat_sub_zeta",
    )
    b_hat_sub_psi = _validate_geometry_field(
        operator.b_hat_sub_psi,
        shape=geometry_shape,
        field_name="magdrift_theta_operator.b_hat_sub_psi",
    )
    db_hat_dzeta = _validate_geometry_field(
        operator.db_hat_dzeta,
        shape=geometry_shape,
        field_name="magdrift_theta_operator.db_hat_dzeta",
    )
    db_hat_dpsi_hat = _validate_geometry_field(
        operator.db_hat_dpsi_hat,
        shape=geometry_shape,
        field_name="magdrift_theta_operator.db_hat_dpsi_hat",
    )
    db_hat_sub_psi_dzeta = _validate_geometry_field(
        operator.db_hat_sub_psi_dzeta,
        shape=geometry_shape,
        field_name="magdrift_theta_operator.db_hat_sub_psi_dzeta",
    )
    db_hat_sub_zeta_dpsi_hat = _validate_geometry_field(
        operator.db_hat_sub_zeta_dpsi_hat,
        shape=geometry_shape,
        field_name="magdrift_theta_operator.db_hat_sub_zeta_dpsi_hat",
    )
    if np.any(b_hat == 0.0):
        raise ValueError("magdrift_theta_operator.b_hat contains zeros")
    z = _charge_scalar(operator, "magdrift_theta_operator")
    gf1 = b_hat_sub_zeta * db_hat_dpsi_hat - b_hat_sub_psi * db_hat_dzeta
    gf2 = 2.0 * b_hat * (db_hat_sub_psi_dzeta - db_hat_sub_zeta_dpsi_hat)
    base = (
        float(np.asarray(operator.delta, dtype=np.float64).reshape(-1)[0])
        * float(np.asarray(operator.t_hat, dtype=np.float64).reshape(-1)[0])
        * d_hat
        / (2.0 * z * (b_hat**3))
    )
    return gf1, gf2, base


def _magnetic_zeta_geometry_factors(operator: Any, *, n_theta: int, n_zeta: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    geometry_shape = (int(n_theta), int(n_zeta))
    d_hat = _validate_geometry_field(operator.d_hat, shape=geometry_shape, field_name="magdrift_zeta_operator.d_hat")
    b_hat = _validate_geometry_field(operator.b_hat, shape=geometry_shape, field_name="magdrift_zeta_operator.b_hat")
    b_hat_sub_theta = _validate_geometry_field(
        operator.b_hat_sub_theta,
        shape=geometry_shape,
        field_name="magdrift_zeta_operator.b_hat_sub_theta",
    )
    b_hat_sub_psi = _validate_geometry_field(
        operator.b_hat_sub_psi,
        shape=geometry_shape,
        field_name="magdrift_zeta_operator.b_hat_sub_psi",
    )
    db_hat_dtheta = _validate_geometry_field(
        operator.db_hat_dtheta,
        shape=geometry_shape,
        field_name="magdrift_zeta_operator.db_hat_dtheta",
    )
    db_hat_dpsi_hat = _validate_geometry_field(
        operator.db_hat_dpsi_hat,
        shape=geometry_shape,
        field_name="magdrift_zeta_operator.db_hat_dpsi_hat",
    )
    db_hat_sub_theta_dpsi_hat = _validate_geometry_field(
        operator.db_hat_sub_theta_dpsi_hat,
        shape=geometry_shape,
        field_name="magdrift_zeta_operator.db_hat_sub_theta_dpsi_hat",
    )
    db_hat_sub_psi_dtheta = _validate_geometry_field(
        operator.db_hat_sub_psi_dtheta,
        shape=geometry_shape,
        field_name="magdrift_zeta_operator.db_hat_sub_psi_dtheta",
    )
    if np.any(b_hat == 0.0):
        raise ValueError("magdrift_zeta_operator.b_hat contains zeros")
    z = _charge_scalar(operator, "magdrift_zeta_operator")
    gf1 = b_hat_sub_psi * db_hat_dtheta - b_hat_sub_theta * db_hat_dpsi_hat
    gf2 = 2.0 * b_hat * (db_hat_sub_theta_dpsi_hat - db_hat_sub_psi_dtheta)
    base = (
        float(np.asarray(operator.delta, dtype=np.float64).reshape(-1)[0])
        * float(np.asarray(operator.t_hat, dtype=np.float64).reshape(-1)[0])
        * d_hat
        / (2.0 * z * (b_hat**3))
    )
    return gf1, gf2, base


def _magnetic_diag_coefficients(n_xi: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ell = np.arange(int(n_xi), dtype=np.float64)
    denom = (2.0 * ell + 3.0) * (2.0 * ell - 1.0)
    c1 = 2.0 * (3.0 * ell * ell + 3.0 * ell - 2.0) / denom
    c2 = (2.0 * ell * ell + 2.0 * ell - 1.0) / denom
    c3 = -2.0 * ell * (ell + 1.0) / denom
    return c1, c2, c3


def _magnetic_offdiag2_plus(n_xi: int) -> np.ndarray:
    ell = np.arange(int(n_xi), dtype=np.float64)
    return (ell + 2.0) * (ell + 1.0) / ((2.0 * ell + 5.0) * (2.0 * ell + 3.0))


def _magnetic_offdiag2_minus(n_xi: int) -> np.ndarray:
    ell = np.arange(int(n_xi), dtype=np.float64)
    return np.where(ell > 1, (ell - 1.0) * ell / ((2.0 * ell - 3.0) * (2.0 * ell - 1.0)), 0.0)


def _charge_scalar(operator: Any, label: str) -> float:
    z = float(np.asarray(operator.z, dtype=np.float64).reshape(-1)[0])
    if z == 0.0:
        raise ValueError(f"{label}.z must be nonzero")
    return z


__all__ = [
    "build_collisionless_f_block_operator",
    "build_er_xdot_f_block_operator",
    "build_er_xidot_f_block_operator",
    "build_exb_theta_f_block_operator",
    "build_exb_zeta_f_block_operator",
    "build_magnetic_drift_theta_f_block_operator",
    "build_magnetic_drift_xidot_f_block_operator",
    "build_magnetic_drift_zeta_f_block_operator",
]
