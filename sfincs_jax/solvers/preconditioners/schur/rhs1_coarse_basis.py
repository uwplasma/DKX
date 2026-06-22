"""Coarse residual bases for RHSMode=1 active Schur preconditioners.

These helpers build the low-pitch, low-angular-mode, and optional targeted
window bases used by the active sparse-coarse and native-stack preconditioner
families. They are host-side setup utilities for non-autodiff sparse
preconditioners; the installed preconditioner action remains a fixed linear
operator once setup is complete.
"""

from __future__ import annotations

from typing import Any
import os

import numpy as np
import scipy.sparse as sp

__all__ = (
    "append_adaptive_residual_basis_csc",
    "build_active_native_xell_coarse_window_basis_csc",
    "build_coarse_residual_basis_csc",
    "coarse_residual_config",
    "coarse_surface_mode_count",
    "coarse_surface_modes",
    "estimate_coarse_residual_nbytes",
    "estimate_xblock_tz_low_l_factor_nbytes",
    "xblock_tz_low_l_config",
)


def xblock_tz_low_l_config(layout: Any) -> dict[str, object]:
    """Resolve low-L ``x``-block sparse-factor controls for coarse estimates."""

    lmax = _env_int("SFINCS_JAX_RHS1_FULL_CSR_XBLOCK_LMAX", 8)
    lmax = max(1, min(int(layout.n_xi), int(lmax)))
    factor_kind = os.environ.get("SFINCS_JAX_RHS1_FULL_CSR_XBLOCK_FACTOR_KIND", "splu").strip().lower()
    if factor_kind not in {"splu", "spilu"}:
        factor_kind = "splu"
    return {
        "lmax": int(lmax),
        "drop_tol": float(_env_float("SFINCS_JAX_RHS1_FULL_CSR_XBLOCK_DROP_TOL", 0.0)),
        "fill_factor": float(_env_float("SFINCS_JAX_RHS1_FULL_CSR_XBLOCK_FILL_FACTOR", 8.0)),
        "factor_kind": factor_kind,
    }


def coarse_residual_config(layout: Any) -> dict[str, object]:
    """Resolve the physics low-mode basis used by RHSMode=1 coarse residual solves."""

    config = xblock_tz_low_l_config(layout)
    coarse_lmax = _env_int("SFINCS_JAX_RHS1_FULL_CSR_COARSE_LMAX", min(4, int(layout.n_xi)))
    coarse_lmax = max(1, min(int(layout.n_xi), int(coarse_lmax)))
    angular_mmax = _env_int("SFINCS_JAX_RHS1_FULL_CSR_COARSE_ANGULAR_MMAX", 1)
    angular_nmax = _env_int("SFINCS_JAX_RHS1_FULL_CSR_COARSE_ANGULAR_NMAX", 1)
    helical_mmax = _env_int("SFINCS_JAX_RHS1_FULL_CSR_COARSE_HELICAL_MMAX", 1)
    helical_nmax = _env_int("SFINCS_JAX_RHS1_FULL_CSR_COARSE_HELICAL_NMAX", min(4, int(layout.n_zeta) // 2))
    angular_mmax = max(0, min(int(layout.n_theta) // 2, int(angular_mmax)))
    angular_nmax = max(0, min(int(layout.n_zeta) // 2, int(angular_nmax)))
    helical_mmax = max(0, min(int(layout.n_theta) // 2, int(helical_mmax)))
    helical_nmax = max(0, min(int(layout.n_zeta) // 2, int(helical_nmax)))
    has_angular_modes = any((angular_mmax, angular_nmax, helical_mmax and helical_nmax))
    config.update(
        {
            "coarse_lmax": int(coarse_lmax),
            "coarse_include_tail": True,
            "coarse_angular_mmax": int(angular_mmax),
            "coarse_angular_nmax": int(angular_nmax),
            "coarse_helical_mmax": int(helical_mmax),
            "coarse_helical_nmax": int(helical_nmax),
            "coarse_basis": (
                "flux_surface_low_l_angular_plus_tail"
                if has_angular_modes
                else "flux_surface_low_l_plus_tail"
            ),
        }
    )
    return config


def estimate_xblock_tz_low_l_factor_nbytes(*, layout: Any, config: dict[str, object]) -> int:
    """Return a conservative sparse-factor memory estimate for low-L x-blocks."""

    block_size = int(config["lmax"]) * int(layout.n_theta) * int(layout.n_zeta)
    n_blocks = int(layout.n_species) * int(layout.n_x)
    # Sparse factors should be much smaller than dense inverse blocks. This
    # conservative cap estimate prevents accidental full-resolution promotion.
    return int(n_blocks * block_size * min(block_size, 64) * np.dtype(np.float64).itemsize)


def estimate_coarse_residual_nbytes(*, layout: Any, config: dict[str, object]) -> int:
    """Estimate sparse basis plus dense coarse-equation storage in bytes."""

    coarse_lmax = int(config["coarse_lmax"])
    surface_mode_count = coarse_surface_mode_count(layout=layout, config=config)
    coarse_kinetic = int(layout.n_species) * int(layout.n_x) * int(coarse_lmax) * int(surface_mode_count)
    coarse_tail = int(layout.total_size) - int(layout.f_size)
    coarse_size = int(coarse_kinetic + coarse_tail)
    basis_nnz = int(coarse_kinetic * layout.n_theta * layout.n_zeta + coarse_tail)
    sparse_bytes = int(basis_nnz * (np.dtype(np.float64).itemsize + np.dtype(np.int32).itemsize))
    sparse_bytes += int((coarse_size + 1) * np.dtype(np.int32).itemsize)
    dense_bytes = int(coarse_size * coarse_size * np.dtype(np.float64).itemsize)
    return int(sparse_bytes + dense_bytes)


def build_coarse_residual_basis_csc(*, layout: Any, config: dict[str, object]) -> Any:
    """Build the sparse full-space low-mode coarse residual basis."""

    coarse_lmax = int(config["coarse_lmax"])
    surface_modes = coarse_surface_modes(layout=layout, config=config)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    col = 0
    ntz = int(layout.n_theta) * int(layout.n_zeta)
    theta = np.arange(int(layout.n_theta), dtype=np.int64)
    zeta = np.arange(int(layout.n_zeta), dtype=np.int64)
    theta_grid, zeta_grid = np.meshgrid(theta, zeta, indexing="ij")
    for species in range(int(layout.n_species)):
        for x in range(int(layout.n_x)):
            for ell in range(coarse_lmax):
                idx = (
                    (((int(species) * int(layout.n_x) + int(x)) * int(layout.n_xi) + int(ell)) * int(layout.n_theta) + theta_grid)
                    * int(layout.n_zeta)
                    + zeta_grid
                ).astype(np.int64, copy=False).reshape((-1,))
                for _mode_name, surface_values in surface_modes:
                    rows.append(idx)
                    cols.append(np.full((ntz,), col, dtype=np.int64))
                    data.append(surface_values)
                    col += 1
    if bool(config.get("coarse_include_tail", True)):
        tail_size = int(layout.total_size) - int(layout.f_size)
        if tail_size > 0:
            tail_rows = int(layout.f_size) + np.arange(tail_size, dtype=np.int64)
            rows.append(tail_rows)
            cols.append(np.arange(col, col + tail_size, dtype=np.int64))
            data.append(np.ones((tail_size,), dtype=np.float64))
            col += tail_size
    if not rows:
        return sp.csc_matrix((int(layout.total_size), 0), dtype=np.float64)
    row = np.concatenate(rows)
    col_idx = np.concatenate(cols)
    values = np.concatenate(data)
    basis = sp.coo_matrix((values, (row, col_idx)), shape=(int(layout.total_size), int(col))).tocsc()
    basis.sum_duplicates()
    basis.eliminate_zeros()
    return basis


def build_active_native_xell_coarse_window_basis_csc(
    *,
    layout: Any,
) -> tuple[Any, dict[str, object]]:
    """Return optional identity columns for targeted active-native coarse windows."""

    spec = os.environ.get("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_XELL_COARSE_WINDOW_SPECS", "").strip()
    max_columns = max(
        0,
        int(_env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_XELL_COARSE_WINDOW_MAX_COLUMNS", 8192)),
    )
    metadata = {
        "window_basis_requested": bool(spec),
        "window_basis_specs": str(spec),
        "window_basis_columns": 0,
        "window_basis_max_columns": int(max_columns),
    }
    if not spec or int(max_columns) <= 0:
        return sp.csc_matrix((int(layout.total_size), 0), dtype=np.float64), metadata

    ell_radius = max(
        0,
        int(_env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_XELL_COARSE_WINDOW_ELL_RADIUS", 1)),
    )
    x_radius = max(
        0,
        int(_env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_XELL_COARSE_WINDOW_X_RADIUS", 0)),
    )

    def parse_axis(raw: str, stop: int) -> tuple[int, ...]:
        item = str(raw).strip().lower()
        if item in {"*", "all"}:
            return tuple(range(int(stop)))
        try:
            value = int(item)
        except ValueError:
            return ()
        if value < 0 or value >= int(stop):
            return ()
        return (int(value),)

    selected: list[int] = []
    skipped_specs = 0
    for raw_item in str(spec).replace(";", ",").split(","):
        item = raw_item.strip()
        if not item:
            continue
        parts = item.replace("/", ":").split(":")
        if len(parts) != 3:
            skipped_specs += 1
            continue
        species_values = parse_axis(parts[0], int(layout.n_species))
        x_centers = parse_axis(parts[1], int(layout.n_x))
        ell_centers = parse_axis(parts[2], int(layout.n_xi))
        if not species_values or not x_centers or not ell_centers:
            skipped_specs += 1
            continue
        for species in species_values:
            for x_center in x_centers:
                x_min = max(0, int(x_center) - int(x_radius))
                x_max = min(int(layout.n_x) - 1, int(x_center) + int(x_radius))
                for ell_center in ell_centers:
                    ell_min = max(0, int(ell_center) - int(ell_radius))
                    ell_max = min(int(layout.n_xi) - 1, int(ell_center) + int(ell_radius))
                    for x_index in range(x_min, x_max + 1):
                        for ell in range(ell_min, ell_max + 1):
                            for theta in range(int(layout.n_theta)):
                                for zeta in range(int(layout.n_zeta)):
                                    selected.append(
                                        layout.kinetic_flat_index(
                                            species=species,
                                            x=x_index,
                                            ell=ell,
                                            theta=theta,
                                            zeta=zeta,
                                        )
                                    )
                                    if len(selected) >= int(max_columns):
                                        break
                                if len(selected) >= int(max_columns):
                                    break
                            if len(selected) >= int(max_columns):
                                break
                        if len(selected) >= int(max_columns):
                            break
                    if len(selected) >= int(max_columns):
                        break
                if len(selected) >= int(max_columns):
                    break
            if len(selected) >= int(max_columns):
                break
        if len(selected) >= int(max_columns):
            break

    if not selected:
        metadata.update(
            {
                "window_basis_skipped_specs": int(skipped_specs),
                "window_basis_truncated": False,
            }
        )
        return sp.csc_matrix((int(layout.total_size), 0), dtype=np.float64), metadata

    rows = np.unique(np.asarray(selected, dtype=np.int64))
    if int(rows.size) > int(max_columns):
        rows = rows[: int(max_columns)]
    cols = np.arange(int(rows.size), dtype=np.int64)
    basis = sp.coo_matrix(
        (np.ones((int(rows.size),), dtype=np.float64), (rows, cols)),
        shape=(int(layout.total_size), int(rows.size)),
    ).tocsc()
    basis.sum_duplicates()
    basis.eliminate_zeros()
    metadata.update(
        {
            "window_basis_columns": int(basis.shape[1]),
            "window_basis_nnz": int(basis.nnz),
            "window_basis_ell_radius": int(ell_radius),
            "window_basis_x_radius": int(x_radius),
            "window_basis_skipped_specs": int(skipped_specs),
            "window_basis_truncated": bool(len(selected) >= int(max_columns)),
        }
    )
    return basis, metadata


def append_adaptive_residual_basis_csc(
    *,
    matrix: Any,
    base_operator: Any,
    basis: Any,
    max_total_columns: int,
) -> tuple[Any, dict[str, object]]:
    """Append bounded residual-derived coarse columns ``z - A M z``.

    The generated vectors are construction-time snapshots of the mismatch
    between the true active operator ``A`` and the selected base preconditioner
    ``M``. They are independent of the Krylov right-hand side, so the resulting
    preconditioner remains linear. Columns are sparsified by relative magnitude
    and capped by both column count and per-column nonzeros.
    """

    enabled = _env_bool("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_ADAPTIVE_RESIDUAL_BASIS", False)
    max_columns = max(0, int(_env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_ADAPTIVE_RESIDUAL_MAX_COLUMNS", 32)))
    max_seed_columns = max(
        0,
        int(_env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_ADAPTIVE_RESIDUAL_MAX_SEED_COLUMNS", 64)),
    )
    max_nnz_per_column = max(
        1,
        int(_env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_ADAPTIVE_RESIDUAL_MAX_NNZ_PER_COLUMN", 4096)),
    )
    drop_rel = max(
        0.0,
        float(_env_float("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_ADAPTIVE_RESIDUAL_DROP_REL", 1.0e-3)),
    )
    min_rel_norm = max(
        0.0,
        float(_env_float("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_ADAPTIVE_RESIDUAL_MIN_REL_NORM", 1.0e-8)),
    )
    metadata = {
        "adaptive_residual_basis_enabled": bool(enabled),
        "adaptive_residual_basis_columns": 0,
        "adaptive_residual_basis_seed_columns": 0,
        "adaptive_residual_basis_max_columns": int(max_columns),
        "adaptive_residual_basis_max_seed_columns": int(max_seed_columns),
        "adaptive_residual_basis_max_nnz_per_column": int(max_nnz_per_column),
        "adaptive_residual_basis_drop_rel": float(drop_rel),
        "adaptive_residual_basis_min_rel_norm": float(min_rel_norm),
    }
    if not bool(enabled) or int(max_columns) <= 0 or int(max_seed_columns) <= 0 or int(basis.shape[1]) <= 0:
        return basis, metadata

    matrix_csr = matrix.tocsr()
    basis_csc = basis.tocsc()
    remaining = max(0, int(max_total_columns) - int(basis_csc.shape[1]))
    max_columns_use = min(int(max_columns), int(remaining))
    if max_columns_use <= 0:
        metadata["adaptive_residual_basis_truncated_by_total_cap"] = True
        return basis_csc, metadata

    seed_count = min(int(basis_csc.shape[1]), int(max_seed_columns))
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    residual_norms: list[float] = []
    skipped_small = 0
    skipped_zero = 0
    for seed_col in range(seed_count):
        if len(rows) >= max_columns_use:
            break
        z = np.asarray(basis_csc[:, seed_col].toarray(), dtype=np.float64).reshape((-1,))
        z_norm = max(float(np.linalg.norm(z)), np.finfo(np.float64).tiny)
        mz = np.asarray(base_operator.matvec(z), dtype=np.float64).reshape((-1,))
        residual = z - np.asarray(matrix_csr @ mz, dtype=np.float64).reshape((-1,))
        residual_norm = float(np.linalg.norm(residual))
        if not np.isfinite(residual_norm) or residual_norm <= 0.0:
            skipped_zero += 1
            continue
        if residual_norm / z_norm < float(min_rel_norm):
            skipped_small += 1
            continue
        abs_residual = np.abs(residual)
        threshold = float(drop_rel) * max(float(np.max(abs_residual)), np.finfo(np.float64).tiny)
        keep = np.flatnonzero(abs_residual >= threshold)
        if keep.size > int(max_nnz_per_column):
            order = np.argpartition(abs_residual[keep], -int(max_nnz_per_column))[-int(max_nnz_per_column) :]
            keep = keep[order]
            keep.sort()
        if keep.size == 0:
            skipped_zero += 1
            continue
        values = residual[keep]
        values_norm = float(np.linalg.norm(values))
        if not np.isfinite(values_norm) or values_norm <= 0.0:
            skipped_zero += 1
            continue
        rows.append(keep.astype(np.int64, copy=False))
        cols.append(np.full((int(keep.size),), len(rows) - 1, dtype=np.int64))
        data.append((values / values_norm).astype(np.float64, copy=False))
        residual_norms.append(float(residual_norm))

    if not rows:
        metadata.update(
            {
                "adaptive_residual_basis_seed_columns": int(seed_count),
                "adaptive_residual_basis_skipped_small": int(skipped_small),
                "adaptive_residual_basis_skipped_zero": int(skipped_zero),
                "adaptive_residual_basis_truncated_by_total_cap": False,
            }
        )
        return basis_csc, metadata

    adaptive = sp.coo_matrix(
        (
            np.concatenate(data),
            (np.concatenate(rows), np.concatenate(cols)),
        ),
        shape=(int(matrix_csr.shape[0]), int(len(rows))),
    ).tocsc()
    adaptive.sum_duplicates()
    adaptive.eliminate_zeros()
    combined = sp.hstack([basis_csc, adaptive], format="csc")
    metadata.update(
        {
            "adaptive_residual_basis_columns": int(adaptive.shape[1]),
            "adaptive_residual_basis_seed_columns": int(seed_count),
            "adaptive_residual_basis_nnz": int(adaptive.nnz),
            "adaptive_residual_basis_skipped_small": int(skipped_small),
            "adaptive_residual_basis_skipped_zero": int(skipped_zero),
            "adaptive_residual_basis_residual_norm_max": float(max(residual_norms)),
            "adaptive_residual_basis_residual_norm_min": float(min(residual_norms)),
            "adaptive_residual_basis_truncated_by_total_cap": bool(len(rows) >= max_columns_use),
        }
    )
    return combined, metadata


def coarse_surface_mode_count(*, layout: Any, config: dict[str, object]) -> int:
    """Return the number of retained normalized angular/helical surface modes."""

    return int(len(coarse_surface_modes(layout=layout, config=config)))


def coarse_surface_modes(*, layout: Any, config: dict[str, object]) -> tuple[tuple[str, np.ndarray], ...]:
    """Return normalized low-angle modes for the RHSMode=1 coarse residual space."""

    n_theta = int(layout.n_theta)
    n_zeta = int(layout.n_zeta)
    theta = 2.0 * np.pi * np.arange(n_theta, dtype=np.float64) / max(1, n_theta)
    zeta = 2.0 * np.pi * np.arange(n_zeta, dtype=np.float64) / max(1, n_zeta)
    theta_grid, zeta_grid = np.meshgrid(theta, zeta, indexing="ij")
    modes: list[tuple[str, np.ndarray]] = []

    def add_mode(name: str, values: np.ndarray) -> None:
        flat = np.asarray(values, dtype=np.float64).reshape((-1,))
        norm = float(np.linalg.norm(flat))
        if not np.isfinite(norm) or norm <= 0.0:
            return
        flat = flat / norm
        for _existing_name, existing in modes:
            # Avoid exact duplicate modes on tiny grids; the coarse solve still
            # has regularization, but removing duplicates keeps conditioning sane.
            if flat.shape == existing.shape and float(abs(np.dot(flat, existing))) > 1.0 - 1.0e-12:
                return
        modes.append((name, flat))

    add_mode("constant", np.ones((n_theta, n_zeta), dtype=np.float64))
    angular_mmax = int(config.get("coarse_angular_mmax", 0) or 0)
    angular_nmax = int(config.get("coarse_angular_nmax", 0) or 0)
    helical_mmax = int(config.get("coarse_helical_mmax", 0) or 0)
    helical_nmax = int(config.get("coarse_helical_nmax", 0) or 0)

    for m in range(1, max(0, angular_mmax) + 1):
        add_mode(f"cos_theta_{m}", np.cos(float(m) * theta_grid))
        add_mode(f"sin_theta_{m}", np.sin(float(m) * theta_grid))
    for n in range(1, max(0, angular_nmax) + 1):
        add_mode(f"cos_zeta_{n}", np.cos(float(n) * zeta_grid))
        add_mode(f"sin_zeta_{n}", np.sin(float(n) * zeta_grid))
    for m in range(1, max(0, helical_mmax) + 1):
        for n in range(1, max(0, helical_nmax) + 1):
            phase = float(m) * theta_grid - float(n) * zeta_grid
            add_mode(f"cos_helical_{m}_{n}", np.cos(phase))
            add_mode(f"sin_helical_{m}_{n}", np.sin(phase))
    return tuple(modes)


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
