"""Small RHSMode=1 residual norm and gate helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ...v3_system import _fs_average_factor, _ix_min, _source_basis_constraint_scheme_1


def build_rhs1_xblock_post_coarse_directions(
    *,
    op: Any,
    residual: jnp.ndarray,
    preconditioner: Callable[[jnp.ndarray], jnp.ndarray],
    direction_projector: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expected_size: int | None = None,
    include_raw: bool,
    fsavg_lmax: int,
    max_extra_units: int,
    max_directions: int,
    angular_lmax: int = -1,
    include_angular_residual: bool = False,
) -> tuple[tuple[str, jnp.ndarray], ...]:
    """Build a small physics-aware correction basis for stalled RHSMode=1 solves.

    The basis is intentionally low-dimensional and matrix-free: residual-like
    directions handle generic error, flux-surface-averaged low-L modes target
    moment/nullspace drift, low Fourier angular modes target global coupling
    missed by an x-local inverse, residual-weighted angular/radial projections
    capture Fourier error with x-dependent amplitudes, and source/constraint
    directions target the constraint rows that are not visible to a pure x-block
    preconditioner.
    """
    residual = jnp.asarray(residual, dtype=jnp.float64)
    total = int(op.total_size)
    expected_size_use = total if expected_size is None else int(expected_size)
    directions: list[tuple[str, jnp.ndarray]] = []

    def _add(name: str, direction: jnp.ndarray) -> None:
        if len(directions) >= int(max_directions):
            return
        vec = jnp.asarray(direction, dtype=jnp.float64).reshape((-1,))
        if vec.shape != (total,):
            return
        if direction_projector is not None:
            vec = jnp.asarray(direction_projector(vec), dtype=jnp.float64).reshape((-1,))
        if vec.shape != (expected_size_use,):
            return
        try:
            norm = float(jnp.linalg.norm(vec))
        except Exception:
            return
        if np.isfinite(norm) and norm > 0.0:
            directions.append((str(name), vec))

    try:
        _add("preconditioned_residual", preconditioner(residual))
    except Exception:
        pass
    if include_raw:
        _add("raw_residual", residual)

    f_res = residual[: op.f_size].reshape(op.fblock.f_shape)
    factor = _fs_average_factor(op.theta_weights, op.zeta_weights, op.d_hat)
    lmax_use = min(max(0, int(fsavg_lmax)), max(0, int(op.n_xi) - 1))
    for il in range(lmax_use + 1):
        if len(directions) >= int(max_directions):
            break
        avg = jnp.einsum("tz,sxtz->sx", factor, f_res[:, :, il, :, :])
        f_dir = jnp.zeros(op.fblock.f_shape, dtype=jnp.float64)
        f_dir = f_dir.at[:, :, il, :, :].set(avg[:, :, None, None])
        tail = jnp.zeros((total - op.f_size,), dtype=jnp.float64)
        _add(f"fsavg_l{il}", jnp.concatenate([f_dir.reshape((-1,)), tail]))

    angular_l_use = min(int(angular_lmax), max(0, int(op.n_xi) - 1))
    if angular_l_use >= 0 and int(op.n_theta) > 1 and int(op.n_zeta) > 1:
        theta = jnp.arange(int(op.n_theta), dtype=jnp.float64)
        zeta = jnp.arange(int(op.n_zeta), dtype=jnp.float64)
        two_pi = float(2.0 * np.pi)
        mode_pairs = (
            (1, 0),
            (0, 1),
            (1, 1),
            (1, -1),
            (2, 0),
            (0, 2),
            (2, 1),
            (1, 2),
        )
        for il in range(angular_l_use + 1):
            for m_mode, n_mode in mode_pairs:
                if len(directions) >= int(max_directions):
                    break
                phase = two_pi * (
                    float(m_mode) * theta[:, None] / float(max(1, int(op.n_theta)))
                    + float(n_mode) * zeta[None, :] / float(max(1, int(op.n_zeta)))
                )
                for parity, pattern in (("cos", jnp.cos(phase)), ("sin", jnp.sin(phase))):
                    pattern_norm = float(jnp.linalg.norm(pattern))
                    if (not np.isfinite(pattern_norm)) or pattern_norm <= 0.0:
                        continue
                    pattern = pattern / pattern_norm
                    if include_angular_residual:
                        weighted_pattern = factor * pattern
                        denom = float(jnp.sum(weighted_pattern * pattern))
                        if np.isfinite(denom) and abs(denom) > 0.0:
                            for s in range(int(op.n_species)):
                                if len(directions) >= int(max_directions):
                                    break
                                coeff = (
                                    jnp.einsum(
                                        "tz,xtz->x",
                                        weighted_pattern,
                                        f_res[s, :, il, :, :],
                                    )
                                    / float(denom)
                                )
                                f_dir = jnp.zeros(op.fblock.f_shape, dtype=jnp.float64)
                                f_dir = f_dir.at[s, :, il, :, :].set(
                                    coeff[:, None, None] * pattern[None, :, :]
                                )
                                tail = jnp.zeros((total - op.f_size,), dtype=jnp.float64)
                                _add(
                                    f"angular_residual_s{s}_l{il}_m{m_mode}_"
                                    f"n{n_mode}_{parity}",
                                    jnp.concatenate([f_dir.reshape((-1,)), tail]),
                                )
                    for s in range(int(op.n_species)):
                        if len(directions) >= int(max_directions):
                            break
                        f_dir = jnp.zeros(op.fblock.f_shape, dtype=jnp.float64)
                        f_dir = f_dir.at[s, :, il, :, :].set(pattern[None, :, :])
                        tail = jnp.zeros((total - op.f_size,), dtype=jnp.float64)
                        _add(
                            f"angular_s{s}_allx_l{il}_m{m_mode}_n{n_mode}_{parity}",
                            jnp.concatenate([f_dir.reshape((-1,)), tail]),
                        )

    extra_start = int(op.f_size + op.phi1_size)
    extra_size = int(op.extra_size)
    if extra_size > 0 and len(directions) < int(max_directions):
        extra_res = residual[extra_start : extra_start + extra_size]
        extra_dir = jnp.zeros((total,), dtype=jnp.float64).at[
            extra_start : extra_start + extra_size
        ].set(extra_res)
        _add("extra_residual", extra_dir)
        if extra_size <= int(max_extra_units):
            for ie in range(extra_size):
                if len(directions) >= int(max_directions):
                    break
                unit = jnp.zeros((total,), dtype=jnp.float64).at[extra_start + ie].set(1.0)
                _add(f"extra_unit_{ie}", unit)

    if int(op.constraint_scheme) == 1 and len(directions) < int(max_directions):
        ix0 = _ix_min(bool(op.point_at_x0))
        source_basis = _source_basis_constraint_scheme_1(op.x)
        for s in range(int(op.n_species)):
            for ibasis, basis in enumerate(source_basis):
                if len(directions) >= int(max_directions):
                    break
                f_dir = jnp.zeros(op.fblock.f_shape, dtype=jnp.float64)
                f_dir = f_dir.at[s, ix0:, 0, :, :].set(basis[ix0:, None, None])
                tail = jnp.zeros((total - op.f_size,), dtype=jnp.float64)
                full = jnp.concatenate([f_dir.reshape((-1,)), tail])
                _add(f"constraint1_source_s{s}_{ibasis}", full)

    return tuple(directions)


def apply_subspace_minres_correction(
    *,
    matvec: Callable[[jnp.ndarray], jnp.ndarray],
    rhs: jnp.ndarray,
    x0: jnp.ndarray,
    direction_builder: Callable[[jnp.ndarray], Sequence[tuple[str, jnp.ndarray]]],
    steps: int,
    max_directions: int,
    alpha_clip: float = 0.0,
    rcond: float = 1.0e-12,
    min_improvement: float = 0.0,
) -> tuple[
    jnp.ndarray, jnp.ndarray, tuple[float, ...], tuple[int, ...], tuple[str, ...]
]:
    """Apply accepted least-squares residual corrections over a bounded basis.

    This matrix-free coarse solve forms only ``A d_i`` for a bounded number of
    candidate directions, solves ``min_alpha ||r - A D alpha||_2``, and accepts
    the update only when the measured true residual decreases.
    """
    rhs = jnp.asarray(rhs, dtype=jnp.float64)
    x = jnp.asarray(x0, dtype=jnp.float64)
    residual = rhs - jnp.asarray(matvec(x), dtype=jnp.float64)
    residual_norm = float(jnp.linalg.norm(residual))
    history: list[float] = [residual_norm]
    accepted_counts: list[int] = []
    accepted_names: list[str] = []
    steps_use = max(0, int(steps))
    max_dirs_use = max(1, int(max_directions))
    alpha_clip_use = max(0.0, float(alpha_clip))
    rcond_use = max(0.0, float(rcond))
    min_improvement_use = max(0.0, float(min_improvement))

    for _ in range(steps_use):
        raw_directions = tuple(direction_builder(residual))[:max_dirs_use]
        names: list[str] = []
        basis_cols: list[np.ndarray] = []
        abasis_cols: list[np.ndarray] = []
        for name, direction in raw_directions:
            direction_np = np.asarray(
                jax.device_get(direction), dtype=np.float64
            ).reshape((-1,))
            if direction_np.shape != (int(x.size),) or not np.all(
                np.isfinite(direction_np)
            ):
                continue
            norm = float(np.linalg.norm(direction_np))
            if (not np.isfinite(norm)) or norm <= 0.0:
                continue
            direction_np = direction_np / norm
            a_direction = np.asarray(
                jax.device_get(matvec(jnp.asarray(direction_np, dtype=jnp.float64))),
                dtype=np.float64,
            ).reshape((-1,))
            if a_direction.shape != direction_np.shape or not np.all(
                np.isfinite(a_direction)
            ):
                continue
            a_norm = float(np.linalg.norm(a_direction))
            if (not np.isfinite(a_norm)) or a_norm <= 0.0:
                continue
            names.append(str(name))
            basis_cols.append(direction_np)
            abasis_cols.append(a_direction)
        if not basis_cols:
            break

        residual_np = np.asarray(jax.device_get(residual), dtype=np.float64).reshape(
            (-1,)
        )
        basis = np.column_stack(basis_cols)
        abasis = np.column_stack(abasis_cols)
        try:
            coeff, *_ = np.linalg.lstsq(
                abasis,
                residual_np,
                rcond=rcond_use if rcond_use > 0.0 else None,
            )
        except np.linalg.LinAlgError:
            coeff = np.linalg.pinv(abasis, rcond=max(rcond_use, 1.0e-12)) @ residual_np
        coeff = np.asarray(coeff, dtype=np.float64).reshape((-1,))
        if alpha_clip_use > 0.0:
            coeff = np.clip(coeff, -alpha_clip_use, alpha_clip_use)
        if not np.all(np.isfinite(coeff)):
            break
        trial_residual_np = residual_np - abasis @ coeff
        trial_norm = float(np.linalg.norm(trial_residual_np))
        if (not np.isfinite(trial_norm)) or trial_norm >= residual_norm * (
            1.0 - min_improvement_use
        ):
            break
        x_np = (
            np.asarray(jax.device_get(x), dtype=np.float64).reshape((-1,))
            + basis @ coeff
        )
        x = jnp.asarray(x_np, dtype=jnp.float64)
        residual = jnp.asarray(trial_residual_np, dtype=jnp.float64)
        residual_norm = trial_norm
        history.append(residual_norm)
        accepted_counts.append(int(len(basis_cols)))
        accepted_names.extend(names)

    return x, residual, tuple(history), tuple(accepted_counts), tuple(accepted_names)


def apply_device_subspace_residual_equation_correction(
    *,
    matvec: Callable[[jnp.ndarray], jnp.ndarray],
    rhs: jnp.ndarray,
    x0: jnp.ndarray,
    direction_builder: Callable[[jnp.ndarray], Sequence[tuple[str, jnp.ndarray]]]
    | None,
    steps: int,
    max_directions: int,
    cached_basis: jnp.ndarray | None = None,
    cached_operator_on_basis: jnp.ndarray | None = None,
    cached_labels: Sequence[str] = (),
    alpha_clip: float = 0.0,
    rcond: float = 1.0e-12,
    min_improvement: float = 0.0,
) -> tuple[
    jnp.ndarray, jnp.ndarray, tuple[float, ...], tuple[int, ...], tuple[str, ...]
]:
    """Apply a device-resident residual-equation correction over a small basis.

    Candidate directions, cached operator actions, and the least-squares solve
    stay as JAX arrays. Cached ``(U, A U)`` columns from a previously built
    coarse space can be mixed with fresh residual-derived directions without
    reapplying the operator to those cached columns.
    """
    rhs = jnp.asarray(rhs, dtype=jnp.float64).reshape((-1,))
    x = jnp.asarray(x0, dtype=jnp.float64).reshape((-1,))
    residual = rhs - jnp.asarray(matvec(x), dtype=jnp.float64).reshape((-1,))
    residual_norm = float(jax.device_get(jnp.linalg.norm(residual)))
    history: list[float] = [residual_norm]
    accepted_counts: list[int] = []
    accepted_names: list[str] = []
    steps_use = max(0, int(steps))
    max_dirs_use = max(1, int(max_directions))
    alpha_clip_use = max(0.0, float(alpha_clip))
    rcond_use = max(0.0, float(rcond))
    min_improvement_use = max(0.0, float(min_improvement))
    n = int(rhs.size)

    cached_basis_jnp: jnp.ndarray | None = None
    cached_action_jnp: jnp.ndarray | None = None
    if cached_basis is not None and cached_operator_on_basis is not None:
        cached_basis_jnp = jnp.asarray(cached_basis, dtype=jnp.float64)
        cached_action_jnp = jnp.asarray(cached_operator_on_basis, dtype=jnp.float64)
        if (
            cached_basis_jnp.ndim != 2
            or cached_action_jnp.ndim != 2
            or int(cached_basis_jnp.shape[0]) != n
            or tuple(cached_basis_jnp.shape) != tuple(cached_action_jnp.shape)
        ):
            cached_basis_jnp = None
            cached_action_jnp = None
    cached_labels_use = tuple(str(label) for label in cached_labels)

    for _ in range(steps_use):
        names: list[str] = []
        basis_cols: list[jnp.ndarray] = []
        action_cols: list[jnp.ndarray] = []

        def _append_candidate(
            name: str,
            direction: jnp.ndarray,
            action: jnp.ndarray | None,
        ) -> None:
            if len(names) >= max_dirs_use:
                return
            vec = jnp.asarray(direction, dtype=jnp.float64).reshape((-1,))
            if int(vec.size) != n:
                return
            if action is None:
                act = jnp.asarray(matvec(vec), dtype=jnp.float64).reshape((-1,))
            else:
                act = jnp.asarray(action, dtype=jnp.float64).reshape((-1,))
            if int(act.size) != n:
                return
            norm = jnp.linalg.norm(vec)
            valid = jnp.logical_and(jnp.isfinite(norm), norm > 0.0)
            safe_norm = jnp.where(valid, norm, jnp.asarray(1.0, dtype=jnp.float64))
            q = jnp.where(valid, vec / safe_norm, jnp.zeros_like(vec))
            aq = jnp.where(valid, act / safe_norm, jnp.zeros_like(act))
            action_norm = jnp.linalg.norm(aq)
            valid = jnp.logical_and(valid, jnp.isfinite(action_norm))
            valid = jnp.logical_and(valid, action_norm > 0.0)
            valid = jnp.logical_and(valid, jnp.all(jnp.isfinite(q)))
            valid = jnp.logical_and(valid, jnp.all(jnp.isfinite(aq)))
            q = jnp.where(valid, q, jnp.zeros_like(q))
            aq = jnp.where(valid, aq, jnp.zeros_like(aq))
            names.append(str(name))
            basis_cols.append(q)
            action_cols.append(aq)

        if cached_basis_jnp is not None and cached_action_jnp is not None:
            cached_rank = min(int(cached_basis_jnp.shape[1]), max_dirs_use)
            for idx in range(cached_rank):
                label = (
                    cached_labels_use[idx]
                    if idx < len(cached_labels_use)
                    else f"cached_qi_{idx}"
                )
                _append_candidate(
                    f"cached_qi:{label}",
                    cached_basis_jnp[:, idx],
                    cached_action_jnp[:, idx],
                )

        if direction_builder is not None and len(names) < max_dirs_use:
            for name, direction in tuple(direction_builder(residual)):
                if len(names) >= max_dirs_use:
                    break
                _append_candidate(str(name), direction, None)

        if not basis_cols:
            break
        basis = jnp.stack(basis_cols, axis=1)
        action_basis = jnp.stack(action_cols, axis=1)
        action_norms = jnp.linalg.norm(action_basis, axis=0)
        valid_cols = jnp.logical_and(jnp.isfinite(action_norms), action_norms > 0.0)
        valid_count = int(jax.device_get(jnp.sum(valid_cols.astype(jnp.int32))))
        if valid_count <= 0:
            break
        basis = jnp.where(valid_cols[None, :], basis, jnp.zeros_like(basis))
        action_basis = jnp.where(
            valid_cols[None, :], action_basis, jnp.zeros_like(action_basis)
        )
        coeff = jnp.linalg.lstsq(
            action_basis,
            residual,
            rcond=rcond_use if rcond_use > 0.0 else None,
        )[0]
        coeff = jnp.where(valid_cols, coeff, jnp.zeros_like(coeff))
        coeff = jnp.nan_to_num(coeff, nan=0.0, posinf=0.0, neginf=0.0)
        if alpha_clip_use > 0.0:
            coeff = jnp.clip(coeff, -alpha_clip_use, alpha_clip_use)
        update = basis @ coeff
        residual_update = action_basis @ coeff
        trial_residual = residual - residual_update
        trial_norm_arr = jnp.linalg.norm(trial_residual)
        trial_norm = float(jax.device_get(trial_norm_arr))
        accept = bool(
            np.isfinite(float(trial_norm))
            and float(trial_norm) < float(residual_norm) * (1.0 - min_improvement_use)
        )
        if not accept:
            break
        x = x + update
        residual = trial_residual
        residual_norm = trial_norm
        history.append(float(residual_norm))
        valid_mask_np = np.asarray(jax.device_get(valid_cols), dtype=bool)
        accepted_counts.append(valid_count)
        accepted_names.extend(
            name
            for name, valid in zip(names, valid_mask_np, strict=True)
            if bool(valid)
        )

    return x, residual, tuple(history), tuple(accepted_counts), tuple(accepted_names)


def l2_norm_float(values: jnp.ndarray) -> float:
    """Return a host ``float`` L2 norm for JAX/NumPy-like vectors."""

    return float(jax.device_get(jnp.linalg.norm(jnp.asarray(values))))


def residual_target(*, atol: float, tol: float, rhs_norm: float) -> float:
    """Return the absolute residual target used by PETSc-style relative gates."""

    return max(float(atol), float(tol) * float(rhs_norm))


def safe_ratio(numerator: float, denominator: float) -> float | None:
    """Return ``numerator / denominator`` only for finite positive denominators."""

    den = float(denominator)
    num = float(numerator)
    if not math.isfinite(num) or not math.isfinite(den) or den <= 0.0:
        return None
    return num / den


def residual_converged(residual_norm: float, target: float) -> bool:
    """Return whether a finite residual satisfies a finite absolute target."""

    residual = float(residual_norm)
    target_use = float(target)
    return bool(
        math.isfinite(residual) and math.isfinite(target_use) and residual <= target_use
    )


__all__ = [
    "apply_device_subspace_residual_equation_correction",
    "apply_subspace_minres_correction",
    "build_rhs1_xblock_post_coarse_directions",
    "l2_norm_float",
    "residual_converged",
    "residual_target",
    "safe_ratio",
]
