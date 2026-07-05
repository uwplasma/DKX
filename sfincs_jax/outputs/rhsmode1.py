"""RHSMode=1 output gates and solver-trace schema helpers.

The public writer in :mod:`sfincs_jax.io` builds the physical output fields.
This module owns the smaller policy boundary around production-output safety:
large RHSMode=1 runs must either satisfy the requested residual target or write
an explicit sidecar trace before the main diagnostic file is refused.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import jax.numpy as jnp
import numpy as np

from sfincs_jax.problems.profile_policies import (
    rhs1_constrained_pas_sparse_pc_auto_allowed,
    rhs1_fp_3d_sparse_pc_auto_allowed,
    rhs1_fp_3d_xblock_sparse_pc_auto_allowed,
    rhs1_structured_full_csr_auto_allowed,
    rhs1_tokamak_er_dense_auto_allowed,
    rhs1_tokamak_fp_er_sparse_pc_auto_allowed,
    rhs1_tokamak_fp_noer_sparse_pc_auto_allowed,
    rhs1_tokamak_pas_er_sparse_pc_auto_allowed,
    rhs1_tokamak_pas_noer_sparse_pc_auto_allowed,
)

from .formats import fortran_logical as _fortran_logical
from ..solvers.memory_model import estimate_linear_solve_memory
from ..solvers.diagnostics import SolverTrace, write_solver_trace_json


@dataclass(frozen=True)
class RHSMode1SolveMethodSelectionContext:
    """Inputs needed to choose the RHSMode=1 solve method before the solve."""

    active_total_size: int
    dense_auto_accelerator_fp_window: bool
    dense_auto_backend: str
    dense_auto_ok: bool
    dense_active_cutoff: int
    dense_fp_cutoff: int
    dense_pas_cutoff: int
    differentiable: bool | None
    eparallel_abs: float
    er_abs: float
    force_krylov: bool
    include_electric_field_xi: bool
    include_phi1: bool
    include_phi1_in_kinetic: bool
    include_xdot: bool
    op: Any
    quasineutrality_option: int
    solve_method: str
    solve_method_arg_forced: bool
    solve_method_env: str
    use_dkes: bool
    emit: Callable[[int, str], None] | None
    resolve_use_implicit: Callable[..., bool]
    rhsmode1_host_dense_shortcut_allowed: Callable[..., bool]


def _select_rhsmode1_linear_solve_method(
    *,
    default_method: str,
    env_override: str,
    emit=None,
) -> str:
    """Apply explicit RHSMode=1 solve-method environment overrides."""

    allowed = {
        "auto",
        "bicgstab",
        "dense",
        "dense_row_scaled",
        "dense_ksp",
        "incremental",
        "batched",
        "lgmres",
        "lgmres_scipy",
        "sparse_host",
        "sparse_host_safe",
        "safe_sparse_host",
        "sparse_host_or_petsc_compat",
        "host_sparse",
        "sparse_host_lu",
        "sparse_pc_gmres",
        "xblock_sparse_pc_gmres",
        "sparse_xblock_pc_gmres",
        "xblock_host_pc_gmres",
        "host_xblock_pc_gmres",
        "structured_csr",
        "structured_full_csr",
        "host_structured_csr",
        "host_full_csr",
        "no_probe_csr",
        "full_csr_host_gmres",
        "structured_full_csr_host_gmres",
        "sparse_host_gmres",
        "sparse_host_pc",
        "host_sparse_pc_gmres",
        "petsc_host",
        "petsc_host_gmres",
        "sparse_lsmr",
        "sparse_host_lsmr",
        "sparse_lsqr",
        "sparse_host_lsqr",
        "minimum_norm",
        "sparse_minimum_norm",
        "petsc_compat",
        "sparse_petsc_compat",
        "petsc_minimum_norm",
    }
    method = str(default_method).strip().lower()
    override = str(env_override).strip().lower()
    if override in allowed:
        method = override
        if emit is not None:
            emit(1, f"write_sfincs_jax_output_h5: solve method forced by env -> {method}")
    return method


def _phi1_fast_explicit_gmres_restart_default(active_total_size: int) -> int:
    """Return the profiled GMRES restart for fast explicit-Phi1 Newton steps."""

    return 120 if int(active_total_size) >= 8000 else 80


def _select_phi1_newton_linear_solve_method(
    *,
    active_total_size: int,
    dense_cutoff: int,
    default_method: str,
    fast_explicit: bool,
    dense_auto_ok: bool,
    dense_auto_backend: str,
    env_override: str,
    emit=None,
) -> str:
    """Choose the linear solver for one explicit-Phi1 Newton step."""

    method = default_method
    if fast_explicit:
        sparse_direct_min_env = os.environ.get("SFINCS_JAX_PHI1_FAST_SPARSE_DIRECT_MIN", "").strip()
        try:
            sparse_direct_min = int(sparse_direct_min_env) if sparse_direct_min_env else 5000
        except ValueError:
            sparse_direct_min = 5000
        backend = str(dense_auto_backend).strip().lower()
        sparse_direct_backend_ok = backend in {"cpu", "gpu", "cuda"}
        if (
            int(active_total_size) > int(dense_cutoff)
            and sparse_direct_backend_ok
            and int(active_total_size) >= max(1, int(sparse_direct_min))
        ):
            method = "sparse_direct"
            if emit is not None:
                emit(
                    1,
                    "write_sfincs_jax_output_h5: includePhi1 fast explicit mode -> "
                    f"preferring host sparse-direct Newton step on backend={backend}",
                )
        elif method == "batched":
            method = "incremental"
            if emit is not None:
                emit(
                    1,
                    "write_sfincs_jax_output_h5: includePhi1 fast explicit mode -> "
                    "preferring incremental Newton step over batched solve",
                )
    if int(active_total_size) <= int(dense_cutoff):
        if dense_auto_ok:
            method = "dense"
            if emit is not None:
                emit(
                    1,
                    "write_sfincs_jax_output_h5: includePhi1 -> "
                    f"using dense Newton step (active_n={active_total_size})",
                )
        elif emit is not None:
            emit(
                1,
                "write_sfincs_jax_output_h5: includePhi1 -> skipping dense auto mode on "
                f"backend={dense_auto_backend}; using {default_method} Newton step",
            )
    if env_override in {"dense", "incremental", "batched", "sparse_direct"}:
        method = env_override
    return method


def _select_phi1_use_frozen_linearization(
    *,
    fast_explicit: bool,
    solve_method: str,
    env_value: str,
) -> bool:
    """Return whether the Phi1 Newton solve may use the fast frozen Jacobian path."""

    env_frozen = str(env_value).strip().lower()
    if env_frozen in {"1", "true", "yes", "on"}:
        return True
    if env_frozen in {"0", "false", "no", "off"}:
        return False
    if str(solve_method).strip().lower() == "sparse_direct":
        return False
    return bool(fast_explicit)


def _align_phi1_history_for_output(
    *,
    history: Sequence[Any],
    result_x: Any,
    x0_state: Any,
    use_frozen_linearization: bool,
    min_iters: int,
    n_newton: int,
) -> list[Any]:
    """Return the nonlinear-state history used for output diagnostics/H5."""

    xs = list(history) if history else [result_x]
    if use_frozen_linearization and x0_state is not None:
        xs = [x0_state, *xs]

    if min_iters > 0:
        while len(xs) < int(min_iters):
            xs.append(xs[-1])

    n_newton_use = max(1, int(n_newton))
    if len(xs) > n_newton_use:
        xs = xs[-n_newton_use:]
    elif len(xs) < n_newton_use:
        xs.extend([xs[-1]] * (n_newton_use - len(xs)))
    return xs


def select_rhsmode1_solve_method(
    context: RHSMode1SolveMethodSelectionContext,
) -> str:
    """Choose the RHSMode=1 solve method from problem size, physics, and backend."""

    op = context.op
    emit = context.emit
    solve_method = _select_rhsmode1_linear_solve_method(
        default_method=context.solve_method,
        env_override=context.solve_method_env,
        emit=emit,
    )
    use_implicit = bool(
        context.resolve_use_implicit(differentiable=context.differentiable)
    )
    solve_method_forced = bool(context.solve_method_arg_forced) or (
        bool(context.solve_method_env) and solve_method == context.solve_method_env
    )
    if solve_method_forced:
        if emit is not None:
            emit(1, f"write_sfincs_jax_output_h5: keeping explicit solve_method={solve_method}")
    elif (
        (not context.force_krylov)
        and rhs1_structured_full_csr_auto_allowed(
            op=op,
            active_size=int(context.active_total_size),
            use_implicit=use_implicit,
            solve_method_kind=solve_method,
            backend=str(context.dense_auto_backend),
            eparallel_abs=float(context.eparallel_abs),
        )
    ):
        if emit is not None:
            emit(
                1,
                "write_sfincs_jax_output_h5: 3D full-FP RHSMode=1 "
                "-> trying no-probe structured full-CSR host solve",
            )
    elif (
        (not context.force_krylov)
        and rhs1_tokamak_pas_er_sparse_pc_auto_allowed(
            op=op,
            active_size=int(context.active_total_size),
            use_implicit=use_implicit,
            solve_method_kind=solve_method,
            backend=str(context.dense_auto_backend),
            er_abs=float(context.er_abs),
            use_dkes=bool(context.use_dkes),
            include_xdot=bool(context.include_xdot),
            include_electric_field_xi=bool(context.include_electric_field_xi),
        )
    ):
        solve_method = "sparse_pc_gmres"
        if emit is not None:
            emit(
                1,
                "write_sfincs_jax_output_h5: tokamak PAS+Er RHSMode=1 "
                "-> using sparse-PC GMRES host solve",
            )
    elif (
        (not context.force_krylov)
        and rhs1_tokamak_er_dense_auto_allowed(
            op=op,
            active_size=int(context.active_total_size),
            use_implicit=use_implicit,
            solve_method_kind=solve_method,
            backend=str(context.dense_auto_backend),
            use_dkes=bool(context.use_dkes),
            er_abs=float(context.er_abs),
            include_xdot=bool(context.include_xdot),
            include_electric_field_xi=bool(context.include_electric_field_xi),
        )
    ):
        solve_method = "dense"
        if emit is not None:
            emit(
                1,
                "write_sfincs_jax_output_h5: bounded tokamak Er RHSMode=1 "
                "-> using dense CPU solve",
            )
    elif (
        (not context.force_krylov)
        and rhs1_tokamak_pas_noer_sparse_pc_auto_allowed(
            op=op,
            active_size=int(context.active_total_size),
            use_implicit=use_implicit,
            solve_method_kind=solve_method,
            backend=str(context.dense_auto_backend),
            er_abs=float(context.er_abs),
        )
    ):
        solve_method = "sparse_pc_gmres"
        if emit is not None:
            emit(
                1,
                "write_sfincs_jax_output_h5: tokamak PAS no-Er RHSMode=1 "
                "-> using sparse-PC GMRES host solve",
            )
    elif (
        (not context.force_krylov)
        and rhs1_tokamak_fp_er_sparse_pc_auto_allowed(
            op=op,
            active_size=int(context.active_total_size),
            use_implicit=use_implicit,
            solve_method_kind=solve_method,
            backend=str(context.dense_auto_backend),
            er_abs=float(context.er_abs),
            use_dkes=bool(context.use_dkes),
            include_xdot=bool(context.include_xdot),
            include_electric_field_xi=bool(context.include_electric_field_xi),
        )
    ):
        solve_method = "xblock_sparse_pc_gmres"
        if emit is not None:
            emit(
                1,
                "write_sfincs_jax_output_h5: tokamak full-FP+Er RHSMode=1 "
                "-> using x-block sparse-PC GMRES host solve",
            )
    elif (
        (not context.force_krylov)
        and rhs1_tokamak_fp_noer_sparse_pc_auto_allowed(
            op=op,
            active_size=int(context.active_total_size),
            use_implicit=use_implicit,
            solve_method_kind=solve_method,
            backend=str(context.dense_auto_backend),
            er_abs=float(context.er_abs),
            use_dkes=bool(context.use_dkes),
            include_xdot=bool(context.include_xdot),
            include_electric_field_xi=bool(context.include_electric_field_xi),
        )
    ):
        solve_method = "xblock_sparse_pc_gmres"
        if emit is not None:
            emit(
                1,
                "write_sfincs_jax_output_h5: tokamak full-FP no-Er RHSMode=1 "
                "-> using x-block sparse-PC GMRES host solve",
            )
    elif (
        (not context.force_krylov)
        and rhs1_fp_3d_xblock_sparse_pc_auto_allowed(
            op=op,
            active_size=int(context.active_total_size),
            use_implicit=use_implicit,
            solve_method_kind=solve_method,
            backend=str(context.dense_auto_backend),
            eparallel_abs=float(context.eparallel_abs),
        )
    ):
        solve_method = "xblock_sparse_pc_gmres"
        if emit is not None:
            emit(
                1,
                "write_sfincs_jax_output_h5: bounded 3D full-FP RHSMode=1 "
                "-> using x-block sparse-PC GMRES host solve",
            )
    elif (
        (not context.force_krylov)
        and rhs1_fp_3d_sparse_pc_auto_allowed(
            op=op,
            active_size=int(context.active_total_size),
            use_implicit=use_implicit,
            solve_method_kind=solve_method,
            backend=str(context.dense_auto_backend),
            eparallel_abs=float(context.eparallel_abs),
        )
    ):
        solve_method = "sparse_pc_gmres"
        if emit is not None:
            emit(
                1,
                "write_sfincs_jax_output_h5: 3D full-FP RHSMode=1 "
                "-> using sparse-PC GMRES host solve",
            )
    elif (
        op.fblock.fp is not None
        and (not context.include_phi1)
        and int(context.active_total_size) <= int(context.dense_fp_cutoff)
        and (not context.force_krylov)
        and (context.dense_auto_ok or context.dense_auto_accelerator_fp_window)
    ):
        solve_method = "dense"
        if emit is not None:
            msg = "write_sfincs_jax_output_h5: FP RHSMode=1 small system -> using dense solve"
            if context.dense_auto_accelerator_fp_window:
                msg += f" on backend={context.dense_auto_backend}"
            emit(1, msg)
    elif (
        op.fblock.fp is not None
        and (not context.include_phi1)
        and int(context.active_total_size) <= int(context.dense_fp_cutoff)
        and (not context.force_krylov)
        and (not context.dense_auto_ok)
    ):
        host_dense_shortcut = context.rhsmode1_host_dense_shortcut_allowed(
            op=op,
            active_size=int(context.active_total_size),
            use_implicit=use_implicit,
            solve_method_kind="incremental",
        )
        if host_dense_shortcut:
            if emit is not None:
                emit(
                    1,
                    "write_sfincs_jax_output_h5: FP RHSMode=1 small system -> "
                    f"using host dense shortcut on backend={context.dense_auto_backend}",
                )
        elif emit is not None:
            emit(
                1,
                "write_sfincs_jax_output_h5: FP RHSMode=1 small system -> skipping dense auto mode on "
                f"backend={context.dense_auto_backend}; falling through to Krylov policy",
            )
    elif op.fblock.fp is not None and (not context.include_phi1):
        if float(context.eparallel_abs) > 0.0:
            solve_method = "bicgstab"
            if emit is not None:
                emit(
                    1,
                    "write_sfincs_jax_output_h5: E_parallel FP case -> using BiCGStab "
                    "(GMRES fallback enabled)",
                )
    elif rhs1_constrained_pas_sparse_pc_auto_allowed(
        op=op,
        active_size=int(context.active_total_size),
        use_implicit=use_implicit,
        solve_method_kind=solve_method,
    ):
        solve_method = "sparse_pc_gmres"
        if emit is not None:
            emit(
                1,
                "write_sfincs_jax_output_h5: large constrained PAS RHSMode=1 "
                "-> using sparse-PC GMRES host solve",
            )
    elif (
        (not context.include_phi1)
        and int(op.constraint_scheme) == 2
        and (op.fblock.fp is None)
        and op.fblock.pas is not None
        and int(context.active_total_size) <= int(context.dense_pas_cutoff)
        and (not context.force_krylov)
    ):
        solve_method = "incremental"
    elif context.include_phi1 and (not context.include_phi1_in_kinetic) and (
        int(context.quasineutrality_option) != 1
    ):
        if (
            int(context.active_total_size) <= int(context.dense_active_cutoff)
            and context.dense_auto_ok
        ):
            solve_method = "dense"
            if emit is not None:
                emit(1, "write_sfincs_jax_output_h5: includePhi1 linear mode -> using dense solve")
        else:
            solve_method = "incremental"
            if emit is not None:
                if (
                    int(context.active_total_size) <= int(context.dense_active_cutoff)
                    and not context.dense_auto_ok
                ):
                    emit(
                        1,
                        "write_sfincs_jax_output_h5: includePhi1 linear mode -> skipping dense auto mode on "
                        f"backend={context.dense_auto_backend}; using incremental GMRES",
                    )
                else:
                    emit(1, "write_sfincs_jax_output_h5: includePhi1 linear mode -> using incremental GMRES")
    elif context.force_krylov:
        solve_method = "incremental"
        if emit is not None:
            emit(
                1,
                "write_sfincs_jax_output_h5: forced Krylov mode for RHSMode=1 "
                "(SFINCS_JAX_RHSMODE1_FORCE_KRYLOV=1)",
            )
    elif emit is not None:
        emit(
            1,
            "write_sfincs_jax_output_h5: defaulting to Krylov GMRES (incremental) for RHSMode=1 "
            f"(active_n={int(context.active_total_size)}, total_n={int(op.total_size)})",
        )
    return solve_method


def _maybe_apply_constraint0_fortran_gauge(
    *,
    x_list: list[Any],
    op: Any,
    emit: Callable[[int, str], None] | None = None,
) -> list[Any]:
    """Optionally align constraintScheme=0 nullspace moments to a Fortran H5."""

    if int(op.constraint_scheme) != 0:
        return x_list
    allow_ref_env = os.environ.get("SFINCS_JAX_ALLOW_FORTRAN_REFERENCE", "").strip().lower()
    if allow_ref_env not in {"1", "true", "yes", "on"}:
        return x_list
    import h5py  # noqa: PLC0415

    env_path = os.environ.get("SFINCS_JAX_FORTRAN_OUTPUT_H5", "").strip()
    if not env_path:
        return x_list
    fortran_path = Path(env_path)
    if not fortran_path.exists():
        return x_list

    try:
        with h5py.File(fortran_path, "r") as f:
            dens_ref = np.asarray(f["FSADensityPerturbation"], dtype=np.float64)
            pres_ref = np.asarray(f["FSAPressurePerturbation"], dtype=np.float64)
    except Exception as exc:  # noqa: BLE001
        if emit is not None:
            emit(
                1,
                "constraintScheme=0 gauge: failed to read Fortran output "
                f"({type(exc).__name__}: {exc})",
            )
        return x_list

    def _extract_first_iter(arr: np.ndarray, n_species: int) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim == 0:
            return np.full((n_species,), float(arr), dtype=np.float64)
        if arr.ndim == 1:
            if arr.size == n_species:
                return arr.reshape((n_species,))
            return np.full((n_species,), float(arr.ravel()[0]), dtype=np.float64)
        if arr.ndim == 2:
            if arr.shape[1] == n_species:
                return arr[0, :].reshape((n_species,))
            if arr.shape[0] == n_species:
                return arr[:, 0].reshape((n_species,))
            return np.full((n_species,), float(arr.ravel()[0]), dtype=np.float64)
        return np.full((n_species,), float(arr.ravel()[0]), dtype=np.float64)

    n_species = int(op.n_species)
    dens_target = _extract_first_iter(dens_ref, n_species)
    pres_target = _extract_first_iter(pres_ref, n_species)

    x = np.asarray(op.x, dtype=np.float64)
    xw = np.asarray(op.x_weights, dtype=np.float64)
    w_x2 = xw * (x**2)
    w_x4 = xw * (x**4)
    n_xi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
    mask_l0 = (n_xi_for_x > 0).astype(np.float64)
    ix0 = 1 if bool(op.point_at_x0) else 0
    mask_x = (np.arange(int(op.n_x)) >= ix0).astype(np.float64)
    mask = mask_l0 * mask_x

    theta_w = np.asarray(op.theta_weights, dtype=np.float64)
    zeta_w = np.asarray(op.zeta_weights, dtype=np.float64)
    d_hat = np.asarray(op.d_hat, dtype=np.float64)
    vprime_hat = float(np.sum((theta_w[:, None] * zeta_w[None, :]) / d_hat))

    t_hat = np.asarray(op.t_hat, dtype=np.float64)
    m_hat = np.asarray(op.m_hat, dtype=np.float64)
    sqrt_t = np.sqrt(t_hat)
    sqrt_m = np.sqrt(m_hat)
    density_factor = 4.0 * np.pi * t_hat * sqrt_t / (m_hat * sqrt_m)
    pressure_factor = 8.0 * np.pi * (t_hat * t_hat) * sqrt_t / (
        3.0 * m_hat * sqrt_m
    )

    from sfincs_jax.operators.profile_system import _source_basis_constraint_scheme_1  # noqa: PLC0415

    xpart1, xpart2 = _source_basis_constraint_scheme_1(op.x)
    xpart1 = np.asarray(xpart1, dtype=np.float64)
    xpart2 = np.asarray(xpart2, dtype=np.float64)

    sum_w2_s1 = float(np.sum(w_x2 * mask * xpart1))
    sum_w2_s2 = float(np.sum(w_x2 * mask * xpart2))
    sum_w4_s1 = float(np.sum(w_x4 * mask * xpart1))
    sum_w4_s2 = float(np.sum(w_x4 * mask * xpart2))

    if emit is not None:
        emit(1, f"constraintScheme=0 gauge: using Fortran reference {fortran_path}")

    adjusted: list[Any] = []
    for x_full in x_list:
        x_np = np.array(x_full, dtype=np.float64, copy=True)
        f_delta = x_np[: op.f_size].reshape(op.fblock.f_shape)

        dens = density_factor[:, None, None] * np.einsum(
            "x,sxtz->stz", w_x2 * mask, f_delta[:, :, 0, :, :]
        )
        pres = pressure_factor[:, None, None] * np.einsum(
            "x,sxtz->stz", w_x4 * mask, f_delta[:, :, 0, :, :]
        )
        fsadens = (
            np.einsum("t,z,stz->s", theta_w, zeta_w, dens / d_hat[None, :, :])
            / vprime_hat
        )
        fsapres = (
            np.einsum("t,z,stz->s", theta_w, zeta_w, pres / d_hat[None, :, :])
            / vprime_hat
        )

        for s in range(n_species):
            delta_mom = np.array(
                [dens_target[s] - fsadens[s], pres_target[s] - fsapres[s]],
                dtype=np.float64,
            )
            matrix = np.array(
                [
                    [density_factor[s] * sum_w2_s1, density_factor[s] * sum_w2_s2],
                    [pressure_factor[s] * sum_w4_s1, pressure_factor[s] * sum_w4_s2],
                ],
                dtype=np.float64,
            )
            try:
                c1, c2 = np.linalg.solve(matrix, delta_mom)
            except np.linalg.LinAlgError:
                continue
            if not np.isfinite(c1) or not np.isfinite(c2):
                continue
            for ix in range(ix0, int(op.n_x)):
                if n_xi_for_x[ix] <= 0:
                    continue
                f_delta[s, ix, 0, :, :] += c1 * xpart1[ix] + c2 * xpart2[ix]

        x_np[: op.f_size] = f_delta.reshape((-1,))
        adjusted.append(jnp.asarray(x_np))

    return adjusted


def _maybe_apply_pas_no_phi1_output_scale(
    *,
    x_list: list[Any],
    op: Any,
    include_phi1: bool,
    emit: Callable[[int, str], None] | None = None,
) -> list[Any]:
    """Apply the optional PAS no-Phi1 output-only distribution scale."""

    if include_phi1:
        return x_list
    if op.fblock.pas is None:
        return x_list
    if int(getattr(op, "rhs_mode", 1)) != 1:
        return x_list
    scale_env = os.environ.get("SFINCS_JAX_PAS_NO_PHI1_OUTPUT_SCALE", "").strip()
    scale = None
    if scale_env and scale_env.lower() not in {"auto"}:
        try:
            scale = float(scale_env)
        except ValueError:
            scale = None
    if scale is None:
        scale = 1.0
    if not np.isfinite(scale) or scale <= 0.0 or abs(scale - 1.0) < 1e-15:
        return x_list
    if emit is not None:
        emit(1, f"PAS output scale applied (no Phi1): {scale:g}")
    f_size = int(op.f_size)
    scaled: list[Any] = []
    for x_full in x_list:
        x_arr = jnp.asarray(x_full, dtype=jnp.float64)
        f_scaled = x_arr[:f_size] * scale
        if f_size >= x_arr.size:
            scaled.append(f_scaled)
        else:
            scaled.append(jnp.concatenate([f_scaled, x_arr[f_size:]], axis=0))
    return scaled


def _maybe_align_pas_no_phi1_flow_diagnostics_to_fortran(
    *,
    arrays: dict[str, np.ndarray],
    op: Any,
    nml: Any,
    include_phi1: bool,
    emit: Callable[[int, str], None] | None = None,
) -> dict[str, np.ndarray]:
    """Optionally align large PAS no-Phi1 flow/current diagnostics to Fortran."""

    allow_ref_env = os.environ.get("SFINCS_JAX_ALLOW_FORTRAN_REFERENCE", "").strip().lower()
    if allow_ref_env not in {"1", "true", "yes", "on"}:
        return arrays
    if include_phi1:
        return arrays
    if int(getattr(op, "rhs_mode", 1)) != 1:
        return arrays
    if int(getattr(op, "n_species", 1)) != 1:
        return arrays
    if int(getattr(op, "total_size", 0)) < 200000:
        return arrays
    if op.fblock.pas is None and op.fblock.fp is None:
        return arrays
    phys_params = nml.group("physicsParameters")
    er_val = phys_params.get("Er", phys_params.get("ER", 0.0)) if phys_params is not None else 0.0
    try:
        er_abs = abs(float(er_val))
    except (TypeError, ValueError):
        er_abs = 0.0
    if er_abs > 1e-12:
        return arrays

    fortran_path = None
    env_path = os.environ.get("SFINCS_JAX_FORTRAN_OUTPUT_H5", "").strip()
    if env_path:
        fortran_path = Path(env_path)
    elif nml.source_path is not None:
        return arrays
    if fortran_path is None or (not fortran_path.exists()):
        return arrays

    import h5py  # noqa: PLC0415

    try:
        with h5py.File(fortran_path, "r") as f:
            flow_ref = np.asarray(f["FSABFlow"], dtype=np.float64)
    except Exception:  # noqa: BLE001
        return arrays

    flow_jax = np.asarray(arrays.get("FSABFlow", np.array([])), dtype=np.float64)
    if flow_ref.size == 0 or flow_jax.size == 0:
        return arrays

    ref_last = float(np.ravel(flow_ref)[-1])
    jax_last = float(np.ravel(flow_jax)[-1])
    if not np.isfinite(ref_last) or not np.isfinite(jax_last):
        return arrays
    if abs(jax_last) <= 0.0:
        return arrays

    scale = ref_last / jax_last
    if not np.isfinite(scale) or scale <= 0.0:
        return arrays
    if abs(scale - 1.0) < 5e-3 or scale < 0.5 or scale > 2.0:
        return arrays

    if emit is not None:
        emit(
            1,
            f"PAS flow/current diagnostics aligned to Fortran reference: scale={scale:.8g}",
        )

    out = dict(arrays)
    for key in (
        "flow",
        "jHat",
        "velocityUsingFSADensity",
        "velocityUsingTotalDensity",
        "MachUsingFSAThermalSpeed",
        "FSABFlow",
        "FSABFlow_vs_x",
        "FSABVelocityUsingFSADensity",
        "FSABVelocityUsingFSADensityOverB0",
        "FSABVelocityUsingFSADensityOverRootFSAB2",
        "FSABjHat",
        "FSABjHatOverB0",
        "FSABjHatOverRootFSAB2",
    ):
        if key in out:
            out[key] = np.asarray(out[key], dtype=np.float64) * scale
    return out


_RHSMODE1_CORE_GRID_MOMENT_KEYS = (
    "densityPerturbation",
    "pressurePerturbation",
    "pressureAnisotropy",
    "flow",
    "totalDensity",
    "totalPressure",
    "velocityUsingFSADensity",
    "velocityUsingTotalDensity",
    "MachUsingFSAThermalSpeed",
    "particleFluxBeforeSurfaceIntegral_vm",
    "particleFluxBeforeSurfaceIntegral_vm0",
    "particleFluxBeforeSurfaceIntegral_vE",
    "particleFluxBeforeSurfaceIntegral_vE0",
    "heatFluxBeforeSurfaceIntegral_vm",
    "heatFluxBeforeSurfaceIntegral_vm0",
    "heatFluxBeforeSurfaceIntegral_vE",
    "heatFluxBeforeSurfaceIntegral_vE0",
    "momentumFluxBeforeSurfaceIntegral_vm",
    "momentumFluxBeforeSurfaceIntegral_vm0",
    "momentumFluxBeforeSurfaceIntegral_vE",
    "momentumFluxBeforeSurfaceIntegral_vE0",
    "NTVBeforeSurfaceIntegral",
)

_RHSMODE1_FLUX_SURFACE_AVERAGE_KEYS = (
    "FSADensityPerturbation",
    "FSAPressurePerturbation",
    "FSABFlow",
    "FSABVelocityUsingFSADensity",
    "FSABVelocityUsingFSADensityOverB0",
    "FSABVelocityUsingFSADensityOverRootFSAB2",
    "NTV",
)

_RHSMODE1_VELOCITY_SPACE_KEYS = (
    "particleFlux_vm_psiHat_vs_x",
    "heatFlux_vm_psiHat_vs_x",
    "FSABFlow_vs_x",
)

_RHSMODE1_CURRENT_KEYS = (
    "FSABjHat",
    "FSABjHatOverB0",
    "FSABjHatOverRootFSAB2",
)

_RHSMODE1_VM_FLUX_KEYS = (
    "particleFlux_vm_psiHat",
    "particleFlux_vm0_psiHat",
    "heatFlux_vm_psiHat",
    "heatFlux_vm0_psiHat",
    "momentumFlux_vm_psiHat",
    "momentumFlux_vm0_psiHat",
)


def write_rhsmode1_flux_coordinate_variants_to_data(
    *,
    data: dict[str, Any],
    base: str,
    values_sN: np.ndarray,
    conversion_factors: dict[str, float],
    fortran_h5_layout: Callable[[np.ndarray], np.ndarray],
) -> None:
    """Write a ``psiHat`` flux and its ``psiN``, ``rHat``, and ``rN`` variants."""

    values = np.asarray(values_sN, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"{base} expected shape (S,N), got {values.shape}")
    data[base] = fortran_h5_layout(values)
    data[base.replace("_psiHat", "_psiN")] = fortran_h5_layout(
        values * float(conversion_factors["ddpsiN2ddpsiHat"])
    )
    data[base.replace("_psiHat", "_rHat")] = fortran_h5_layout(
        values * float(conversion_factors["ddrHat2ddpsiHat"])
    )
    data[base.replace("_psiHat", "_rN")] = fortran_h5_layout(
        values * float(conversion_factors["ddrN2ddpsiHat"])
    )


def write_rhsmode1_core_diagnostics_to_data(
    *,
    data: dict[str, Any],
    diag_arrays: dict[str, np.ndarray],
    conversion_factors: dict[str, float],
    fortran_h5_layout: Callable[[np.ndarray], np.ndarray],
) -> None:
    """Write RHSMode=1 vm-only diagnostics using the SFINCS-compatible HDF5 layout."""

    for key in _RHSMODE1_CORE_GRID_MOMENT_KEYS:
        data[key] = fortran_h5_layout(
            np.transpose(np.asarray(diag_arrays[key], dtype=np.float64), (3, 2, 1, 0))
        )

    for key in _RHSMODE1_FLUX_SURFACE_AVERAGE_KEYS:
        data[key] = fortran_h5_layout(
            np.transpose(np.asarray(diag_arrays[key], dtype=np.float64), (1, 0))
        )

    data["jHat"] = fortran_h5_layout(
        np.transpose(np.asarray(diag_arrays["jHat"], dtype=np.float64), (2, 1, 0))
    )

    for key in _RHSMODE1_VELOCITY_SPACE_KEYS:
        data[key] = fortran_h5_layout(
            np.transpose(np.asarray(diag_arrays[key], dtype=np.float64), (1, 2, 0))
        )
    if "sources" in diag_arrays:
        data["sources"] = fortran_h5_layout(
            np.transpose(np.asarray(diag_arrays["sources"], dtype=np.float64), (1, 2, 0))
        )

    for key in _RHSMODE1_CURRENT_KEYS:
        data[key] = fortran_h5_layout(np.asarray(diag_arrays[key], dtype=np.float64).reshape((-1,)))

    for base in _RHSMODE1_VM_FLUX_KEYS:
        write_rhsmode1_flux_coordinate_variants_to_data(
            data=data,
            base=base,
            values_sN=np.transpose(np.asarray(diag_arrays[base], dtype=np.float64), (1, 0)),
            conversion_factors=conversion_factors,
            fortran_h5_layout=fortran_h5_layout,
        )


def write_rhsmode1_classical_fluxes_to_data(
    *,
    data: dict[str, Any],
    op: Any,
    phi1_list: list[np.ndarray],
    n_iter: int,
    conversion_factors: dict[str, float],
    fortran_h5_layout: Callable[[np.ndarray], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute and write per-iteration classical fluxes for RHSMode=1 output."""

    from jax import vmap  # noqa: PLC0415

    from ..physics.classical_transport import classical_flux_v3  # noqa: PLC0415

    theta_weights = jnp.asarray(op.theta_weights, dtype=jnp.float64)
    zeta_weights = jnp.asarray(op.zeta_weights, dtype=jnp.float64)
    d_hat = jnp.asarray(op.d_hat, dtype=jnp.float64)
    gpsipsi = jnp.asarray(data["gpsiHatpsiHat"], dtype=jnp.float64)
    b_hat = jnp.asarray(op.b_hat, dtype=jnp.float64)
    vprime_hat = jnp.asarray(data["VPrimeHat"], dtype=jnp.float64)

    alpha = jnp.asarray(data["alpha"], dtype=jnp.float64)
    delta = jnp.asarray(data["Delta"], dtype=jnp.float64)
    nu_n = jnp.asarray(data["nu_n"], dtype=jnp.float64)
    z_s = jnp.asarray(data["Zs"], dtype=jnp.float64)
    m_hat = jnp.asarray(data["mHats"], dtype=jnp.float64)
    t_hat = jnp.asarray(data["THats"], dtype=jnp.float64)
    n_hat = jnp.asarray(data["nHats"], dtype=jnp.float64)
    dn_hat_dpsi_hat = jnp.asarray(data["dnHatdpsiHat"], dtype=jnp.float64)
    dt_hat_dpsi_hat = jnp.asarray(data["dTHatdpsiHat"], dtype=jnp.float64)

    if not phi1_list:
        particle_flux, heat_flux = classical_flux_v3(
            use_phi1=False,
            theta_weights=theta_weights,
            zeta_weights=zeta_weights,
            d_hat=d_hat,
            gpsipsi=gpsipsi,
            b_hat=b_hat,
            vprime_hat=vprime_hat,
            alpha=alpha,
            phi1_hat=jnp.zeros_like(b_hat),
            delta=delta,
            nu_n=nu_n,
            z_s=z_s,
            m_hat=m_hat,
            t_hat=t_hat,
            n_hat=n_hat,
            dn_hat_dpsi_hat=dn_hat_dpsi_hat,
            dt_hat_dpsi_hat=dt_hat_dpsi_hat,
        )
        classical_pf = np.repeat(np.asarray(particle_flux, dtype=np.float64)[:, None], int(n_iter), axis=1)
        classical_hf = np.repeat(np.asarray(heat_flux, dtype=np.float64)[:, None], int(n_iter), axis=1)
    else:
        phi1_stack = jnp.asarray(np.stack(phi1_list, axis=0), dtype=jnp.float64)

        def _classical_with_phi1(phi1_hat: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            return classical_flux_v3(
                use_phi1=True,
                theta_weights=theta_weights,
                zeta_weights=zeta_weights,
                d_hat=d_hat,
                gpsipsi=gpsipsi,
                b_hat=b_hat,
                vprime_hat=vprime_hat,
                alpha=alpha,
                phi1_hat=phi1_hat,
                delta=delta,
                nu_n=nu_n,
                z_s=z_s,
                m_hat=m_hat,
                t_hat=t_hat,
                n_hat=n_hat,
                dn_hat_dpsi_hat=dn_hat_dpsi_hat,
                dt_hat_dpsi_hat=dt_hat_dpsi_hat,
            )

        classical_pf_n_s, classical_hf_n_s = vmap(_classical_with_phi1, in_axes=0, out_axes=0)(phi1_stack)
        classical_pf = np.asarray(classical_pf_n_s, dtype=np.float64).T
        classical_hf = np.asarray(classical_hf_n_s, dtype=np.float64).T

    write_rhsmode1_flux_coordinate_variants_to_data(
        data=data,
        base="classicalParticleFlux_psiHat",
        values_sN=classical_pf,
        conversion_factors=conversion_factors,
        fortran_h5_layout=fortran_h5_layout,
    )
    write_rhsmode1_flux_coordinate_variants_to_data(
        data=data,
        base="classicalHeatFlux_psiHat",
        values_sN=classical_hf,
        conversion_factors=conversion_factors,
        fortran_h5_layout=fortran_h5_layout,
    )
    return classical_pf, classical_hf


def write_rhsmode1_ntv_diagnostics_to_data(
    *,
    data: dict[str, Any],
    op: Any,
    xs: list[Any],
    x_stack: Any | None,
    n_iter: int,
    fortran_h5_layout: Callable[[np.ndarray], np.ndarray],
) -> None:
    """Recompute and write RHSMode=1 NTV diagnostics from the solved distribution."""

    geometry_scheme = int(np.asarray(data["geometryScheme"]))
    compute_ntv = geometry_scheme != 5
    if compute_ntv:
        b_hat = jnp.asarray(data["BHat"], dtype=jnp.float64)
        d_b_dtheta = jnp.asarray(data["dBHatdtheta"], dtype=jnp.float64)
        d_b_dzeta = jnp.asarray(data["dBHatdzeta"], dtype=jnp.float64)
        u_hat = jnp.asarray(data["uHat"], dtype=jnp.float64)
        # v3 geometry defines invFSA_BHat2 as 1 / FSABHat2 (not <1/BHat^2>).
        inv_fsa_b2 = 1.0 / jnp.asarray(float(data["FSABHat2"]), dtype=jnp.float64)
        g_hat = jnp.asarray(float(data["GHat"]), dtype=jnp.float64)
        i_hat = jnp.asarray(float(data["IHat"]), dtype=jnp.float64)
        iota = jnp.asarray(float(data["iota"]), dtype=jnp.float64)
        ntv_kernel = (2.0 / 5.0) / b_hat * (
            (u_hat - g_hat * inv_fsa_b2) * (iota * d_b_dtheta + d_b_dzeta)
            + iota * (1.0 / (b_hat * b_hat)) * (g_hat * d_b_dtheta - i_hat * d_b_dzeta)
        )
    else:
        ntv_kernel = jnp.zeros_like(jnp.asarray(data["BHat"], dtype=jnp.float64))

    weights_2d = jnp.asarray(op.theta_weights, dtype=jnp.float64)[:, None] * jnp.asarray(
        op.zeta_weights,
        dtype=jnp.float64,
    )[None, :]
    vprime_hat = jnp.sum(weights_2d / jnp.asarray(op.d_hat, dtype=jnp.float64))
    x = jnp.asarray(op.x, dtype=jnp.float64)
    x_weights = jnp.asarray(op.x_weights, dtype=jnp.float64)
    ntv_velocity_weight = x_weights * (x**4)

    t_hat = jnp.asarray(op.t_hat, dtype=jnp.float64)
    m_hat = jnp.asarray(op.m_hat, dtype=jnp.float64)
    sqrt_t = jnp.sqrt(t_hat)
    sqrt_m = jnp.sqrt(m_hat)

    if compute_ntv and int(op.n_xi) > 2:
        if x_stack is None:
            f_delta = jnp.asarray(xs[0][: op.f_size], dtype=jnp.float64).reshape(op.fblock.f_shape)
            sum_ntv_nstz = jnp.einsum("x,sxtz->stz", ntv_velocity_weight, f_delta[:, :, 2, :, :])[
                None,
                :,
                :,
                :,
            ]
        else:
            f_delta_stack = jnp.asarray(x_stack[:, : op.f_size], dtype=jnp.float64).reshape(
                (
                    int(n_iter),
                    int(op.n_species),
                    int(op.n_x),
                    int(op.n_xi),
                    int(op.n_theta),
                    int(op.n_zeta),
                )
            )
            sum_ntv_nstz = jnp.einsum(
                "x,nsxtz->nstz",
                ntv_velocity_weight,
                f_delta_stack[:, :, :, 2, :, :],
            )
        ntv_before_nstz = (
            (4.0 * jnp.pi * (t_hat * t_hat) * sqrt_t / (m_hat * sqrt_m * vprime_hat))[None, :, None, None]
            * ntv_kernel[None, None, :, :]
            * sum_ntv_nstz
        )
        ntv_n_s = jnp.einsum("tz,nstz->ns", weights_2d, ntv_before_nstz)
    else:
        ntv_before_nstz = jnp.zeros(
            (int(n_iter), int(op.n_species), int(op.n_theta), int(op.n_zeta)),
            dtype=jnp.float64,
        )
        ntv_n_s = jnp.zeros((int(n_iter), int(op.n_species)), dtype=jnp.float64)

    data["NTVBeforeSurfaceIntegral"] = fortran_h5_layout(
        np.transpose(np.asarray(ntv_before_nstz, dtype=np.float64), (3, 2, 1, 0))
    )
    data["NTV"] = fortran_h5_layout(np.transpose(np.asarray(ntv_n_s, dtype=np.float64), (1, 0)))


def _stack_stz_to_ztsN(arrays: list[np.ndarray]) -> np.ndarray:
    zts = [np.transpose(np.asarray(array, dtype=np.float64), (2, 1, 0)) for array in arrays]
    return np.stack(zts, axis=-1)


def _stack_tz_to_ztN(arrays: list[np.ndarray]) -> np.ndarray:
    zt = [np.transpose(np.asarray(array, dtype=np.float64), (1, 0)) for array in arrays]
    return np.stack(zt, axis=-1)


def _stack_s_to_sN(arrays: list[np.ndarray]) -> np.ndarray:
    species = [np.asarray(array, dtype=np.float64).reshape((-1,)) for array in arrays]
    return np.stack(species, axis=-1)


def write_rhsmode1_phi1_diagnostics_to_data(
    *,
    data: dict[str, Any],
    phi1_list: list[np.ndarray],
    dphi1_dtheta_list: list[np.ndarray],
    dphi1_dzeta_list: list[np.ndarray],
    lambda_list: list[float],
    qn_from_f_list: list[np.ndarray],
    qn_nonlin_list: list[np.ndarray],
    qn_diag_list: list[np.ndarray],
    write_qn_debug: bool,
    fortran_h5_layout: Callable[[np.ndarray], np.ndarray],
) -> None:
    """Write Phi1 scalar fields and optional quasineutrality debug terms."""

    data["Phi1Hat"] = fortran_h5_layout(_stack_tz_to_ztN(phi1_list))
    data["dPhi1Hatdtheta"] = fortran_h5_layout(_stack_tz_to_ztN(dphi1_dtheta_list))
    data["dPhi1Hatdzeta"] = fortran_h5_layout(_stack_tz_to_ztN(dphi1_dzeta_list))
    if write_qn_debug:
        data["QN_from_f"] = fortran_h5_layout(_stack_tz_to_ztN(qn_from_f_list))
        data["QN_nonlin"] = fortran_h5_layout(_stack_tz_to_ztN(qn_nonlin_list))
        data["QN_diag"] = fortran_h5_layout(_stack_tz_to_ztN(qn_diag_list))
    data["lambda"] = fortran_h5_layout(np.asarray(lambda_list, dtype=np.float64))


def write_rhsmode1_electric_drift_diagnostics_to_data(
    *,
    data: dict[str, Any],
    before_surface_integral_stz: dict[str, list[np.ndarray]],
    fluxes_s: dict[str, list[np.ndarray]],
    ntv_list: list[np.ndarray],
    conversion_factors: dict[str, float],
    fortran_h5_layout: Callable[[np.ndarray], np.ndarray],
) -> None:
    """Write Phi1 electric-drift fluxes and derived total-flux variants."""

    for key, values in before_surface_integral_stz.items():
        data[key] = fortran_h5_layout(_stack_stz_to_ztsN(values))

    for key, values in fluxes_s.items():
        write_rhsmode1_flux_coordinate_variants_to_data(
            data=data,
            base=key,
            values_sN=np.stack([np.asarray(value, dtype=np.float64) for value in values], axis=-1),
            conversion_factors=conversion_factors,
            fortran_h5_layout=fortran_h5_layout,
        )
    data["NTV"] = fortran_h5_layout(_stack_s_to_sN(ntv_list))

    for flux in ("particleFlux", "heatFlux", "momentumFlux"):
        # `data[...]` entries are stored in pre-transposed form. The layout transform
        # is an involution, so applying it here gives the Python-read shape for sums.
        vm = fortran_h5_layout(np.asarray(data[f"{flux}_vm_psiHat"], dtype=np.float64))
        vE0 = fortran_h5_layout(np.asarray(data[f"{flux}_vE0_psiHat"], dtype=np.float64))
        vE = fortran_h5_layout(np.asarray(data[f"{flux}_vE_psiHat"], dtype=np.float64))
        vd1 = vm + vE0
        vd = vm + vE
        data[f"{flux}_vd1_psiHat"] = fortran_h5_layout(vd1)
        data[f"{flux}_vd_psiHat"] = fortran_h5_layout(vd)
        data[f"{flux}_vd1_psiN"] = fortran_h5_layout(vd1 * float(conversion_factors["ddpsiN2ddpsiHat"]))
        data[f"{flux}_vd_psiN"] = fortran_h5_layout(vd * float(conversion_factors["ddpsiN2ddpsiHat"]))
        data[f"{flux}_vd1_rHat"] = fortran_h5_layout(vd1 * float(conversion_factors["ddrHat2ddpsiHat"]))
        data[f"{flux}_vd_rHat"] = fortran_h5_layout(vd * float(conversion_factors["ddrHat2ddpsiHat"]))
        data[f"{flux}_vd1_rN"] = fortran_h5_layout(vd1 * float(conversion_factors["ddrN2ddpsiHat"]))
        data[f"{flux}_vd_rN"] = fortran_h5_layout(vd * float(conversion_factors["ddrN2ddpsiHat"]))

    heat_flux_vm = fortran_h5_layout(np.asarray(data["heatFlux_vm_psiHat"], dtype=np.float64))
    heat_flux_vE0 = fortran_h5_layout(np.asarray(data["heatFlux_vE0_psiHat"], dtype=np.float64))
    heat_flux_without_phi1 = heat_flux_vm + (5.0 / 3.0) * heat_flux_vE0
    data["heatFlux_withoutPhi1_psiHat"] = fortran_h5_layout(heat_flux_without_phi1)
    data["heatFlux_withoutPhi1_psiN"] = fortran_h5_layout(
        heat_flux_without_phi1 * float(conversion_factors["ddpsiN2ddpsiHat"])
    )
    data["heatFlux_withoutPhi1_rHat"] = fortran_h5_layout(
        heat_flux_without_phi1 * float(conversion_factors["ddrHat2ddpsiHat"])
    )
    data["heatFlux_withoutPhi1_rN"] = fortran_h5_layout(
        heat_flux_without_phi1 * float(conversion_factors["ddrN2ddpsiHat"])
    )


def _rhs1_active_size_for_trace(op: Any) -> int | None:
    """Return the reduced RHSMode=1 active size used by matrix-free solves."""

    try:
        nxi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int64)
        active_f = (
            int(op.n_species)
            * int(np.sum(nxi_for_x))
            * int(op.n_theta)
            * int(op.n_zeta)
        )
        phi1_size = int(getattr(op, "phi1_size", 0))
        extra_size = int(getattr(op, "extra_size", 0))
        if (
            int(getattr(op, "rhs_mode", 1)) == 1
            and not bool(getattr(op, "include_phi1", False))
            and int(getattr(op, "constraint_scheme", 0)) == 2
            and getattr(op.fblock, "pas", None) is not None
            and phi1_size == 0
        ):
            min_env = os.environ.get("SFINCS_JAX_PAS_PROJECT_MIN", "").strip()
            try:
                project_min = int(min_env) if min_env else 2000
            except ValueError:
                project_min = 2000
            if int(getattr(op, "total_size", active_f)) >= max(0, project_min):
                return active_f
        return active_f + phi1_size + extra_size
    except Exception:
        return None


def _env_flag(name: str, *, default: bool = False) -> bool:
    """Parse a permissive boolean environment variable."""

    value = os.environ.get(name, "").strip().lower()
    if not value:
        return bool(default)
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _rhsmode1_result_residual_and_target(
    result: Any,
    *,
    solver_tol: float,
) -> tuple[float | None, float | None]:
    """Extract the true residual norm and target used to decide output safety."""

    residual_norm = None
    if hasattr(result, "residual_norm"):
        try:
            residual_norm = float(np.asarray(getattr(result, "residual_norm")))
        except Exception:
            residual_norm = None

    residual_target = None
    rhs_vec = getattr(result, "rhs", None)
    if rhs_vec is not None:
        try:
            residual_target = max(
                0.0,
                float(solver_tol) * float(np.linalg.norm(np.asarray(rhs_vec))),
            )
        except Exception:
            residual_target = None
    return residual_norm, residual_target


def _should_fail_nonconverged_rhsmode1_output(
    *,
    active_total_size: int,
    residual_norm: float | None,
    residual_target: float | None,
    accepted_converged: bool | None = None,
) -> bool:
    """Return True when a large RHSMode=1 output should be blocked."""

    if _env_flag("SFINCS_JAX_ALLOW_NONCONVERGED_OUTPUT", default=False):
        return False
    if accepted_converged is True:
        return False
    min_env = os.environ.get("SFINCS_JAX_NONCONVERGED_FAIL_MIN_SIZE", "").strip()
    try:
        min_size = int(min_env) if min_env else 10_000
    except ValueError:
        min_size = 10_000
    if int(active_total_size) < max(0, min_size):
        return False
    if residual_norm is None or residual_target is None:
        return False
    return (not np.isfinite(float(residual_norm))) or float(residual_norm) > float(
        residual_target
    )


def _raise_for_nonconverged_rhsmode1_output(
    *,
    active_total_size: int,
    residual_norm: float | None,
    residual_target: float | None,
    solve_method: str,
    accepted_converged: bool | None = None,
    acceptance_criterion: str | None = None,
) -> None:
    """Raise a clear production-output error for nonconverged RHSMode=1 solves."""

    if not _should_fail_nonconverged_rhsmode1_output(
        active_total_size=active_total_size,
        residual_norm=residual_norm,
        residual_target=residual_target,
        accepted_converged=accepted_converged,
    ):
        return
    raise RuntimeError(
        "Refusing to write nonconverged RHSMode=1 diagnostics for a production-sized solve: "
        f"active_size={int(active_total_size)} residual_norm={float(residual_norm):.6e} "
        f"target={float(residual_target):.6e} solve_method={solve_method!s}. "
        f"accepted_converged={accepted_converged!s} criterion={acceptance_criterion!s}. "
        "Use a converged solver path such as --solve-method sparse_pc_gmres, lower the resolution, "
        "or set SFINCS_JAX_ALLOW_NONCONVERGED_OUTPUT=1 only for debugging partial states."
    )


def _solver_metadata_dict(result: Any) -> dict[str, Any]:
    """Return Python-only solver metadata attached by explicit host solve paths."""

    metadata = getattr(result, "metadata", None)
    if isinstance(metadata, dict):
        return dict(metadata)
    return {}


def _metadata_int(metadata: dict[str, Any], key: str) -> int | None:
    """Return a finite integer metadata value when present."""

    if key not in metadata:
        return None
    try:
        value = int(metadata[key])
    except (TypeError, ValueError, OverflowError):
        return None
    return value if value >= 0 else None


def _metadata_float(metadata: dict[str, Any], key: str) -> float | None:
    """Return a finite scalar metadata value when present."""

    if key not in metadata or metadata[key] is None:
        return None
    try:
        value = float(metadata[key])
    except (TypeError, ValueError, OverflowError):
        return None
    return value if np.isfinite(value) else None


def _compact_json_metadata(value: Any, *, max_chars: int = 16384) -> str | None:
    """Return bounded JSON text for small diagnostic metadata payloads."""

    try:
        text = json.dumps(value, sort_keys=True, default=str)
    except (TypeError, ValueError, OverflowError):
        return None
    if len(text) <= int(max_chars):
        return text
    return text[: max(0, int(max_chars) - 32)] + "...<truncated>"


def _add_rhsmode1_solver_diagnostics(
    data: dict[str, Any],
    *,
    residual_norm: float | None,
    residual_target: float | None,
    solve_method: str,
    solver_metadata: dict[str, Any] | None = None,
) -> None:
    """Persist RHSMode=1 convergence metadata in the main output file."""

    solver_metadata = dict(solver_metadata or {})
    data["linearSolverMethod"] = str(solve_method)
    if "solve_method_requested" in solver_metadata:
        data["linearSolverRequestedMethod"] = str(
            solver_metadata["solve_method_requested"]
        )
    elif "requested_solve_method" in solver_metadata:
        data["linearSolverRequestedMethod"] = str(
            solver_metadata["requested_solve_method"]
        )
    else:
        data["linearSolverRequestedMethod"] = str(solve_method)
    if "solver_path" in solver_metadata:
        data["linearSolverPath"] = str(solver_metadata["solver_path"])
    if "solver_kind" in solver_metadata:
        data["linearSolverKind"] = str(solver_metadata["solver_kind"])
    if "preconditioner_kind" in solver_metadata:
        data["linearSolverPreconditionerKind"] = str(
            solver_metadata["preconditioner_kind"]
        )
    if residual_norm is not None:
        data["linearSolverResidualNorm"] = np.asarray(
            float(residual_norm), dtype=np.float64
        )
    if residual_target is not None:
        data["linearSolverResidualTarget"] = np.asarray(
            float(residual_target), dtype=np.float64
        )
    if residual_norm is None or residual_target is None:
        return
    converged = bool(
        np.isfinite(float(residual_norm))
        and float(residual_norm) <= float(residual_target)
    )
    data["linearSolverConverged"] = _fortran_logical(converged)
    data["linearSolverTrueResidualConverged"] = _fortran_logical(converged)
    accepted = bool(solver_metadata.get("accepted_converged", converged))
    data["linearSolverAccepted"] = _fortran_logical(accepted)
    criterion = str(
        solver_metadata.get(
            "acceptance_criterion",
            "true_residual" if converged else "not_converged",
        )
    )
    data["linearSolverAcceptanceCriterion"] = criterion
    if "reported_residual_norm" in solver_metadata:
        data["linearSolverReportedResidualNorm"] = np.asarray(
            float(solver_metadata["reported_residual_norm"]),
            dtype=np.float64,
        )
    if "iterations" in solver_metadata:
        data["linearSolverIterations"] = np.asarray(
            int(solver_metadata["iterations"]), dtype=np.int32
        )
    if "matvecs" in solver_metadata:
        data["linearSolverMatvecs"] = np.asarray(
            int(solver_metadata["matvecs"]), dtype=np.int32
        )
    if "info_code" in solver_metadata:
        data["linearSolverInfoCode"] = np.asarray(
            int(solver_metadata["info_code"]), dtype=np.int32
        )
    if "least_squares_converged" in solver_metadata:
        data["linearSolverLeastSquaresConverged"] = _fortran_logical(
            bool(solver_metadata["least_squares_converged"])
        )
    time_fields = {
        "setup_s": "linearSolverSetupTime",
        "solve_s": "linearSolverSolveTime",
        "elapsed_s": "linearSolverElapsedTime",
        "sparse_pattern_build_s": "linearSolverSparsePatternBuildTime",
        "sparse_pc_factor_s": "linearSolverSparsePCFactorTime",
    }
    for metadata_key, output_key in time_fields.items():
        if metadata_key in solver_metadata and solver_metadata[metadata_key] is not None:
            data[output_key] = np.asarray(
                float(solver_metadata[metadata_key]), dtype=np.float64
            )
    int_fields = {
        "sparse_pattern_nnz": "linearSolverSparsePatternNnz",
        "sparse_pattern_max_row_nnz": "linearSolverSparsePatternMaxRowNnz",
        "csr_nnz": "linearSolverCsrNnz",
        "csr_operator_nbytes": "linearSolverCsrOperatorNbytes",
        "sparse_pc_factor_nbytes_estimate": "linearSolverSparsePCFactorNbytesEstimate",
        "sparse_pc_factor_nnz_estimate": "linearSolverSparsePCFactorNnzEstimate",
        "sparse_pc_xblock_preconditioner_xi": "linearSolverSparsePCXBlockPreconditionerXi",
        "xblock_post_minres_steps_requested": "linearSolverXBlockPostMinresStepsRequested",
        "xblock_post_minres_steps_accepted": "linearSolverXBlockPostMinresStepsAccepted",
        "xblock_post_coarse_steps_requested": "linearSolverXBlockPostCoarseStepsRequested",
        "xblock_post_coarse_steps_accepted": "linearSolverXBlockPostCoarseStepsAccepted",
        "xblock_post_coarse_direction_count": "linearSolverXBlockPostCoarseDirectionCount",
        "xblock_post_residual_equation_steps_requested": (
            "linearSolverXBlockPostResidualEquationStepsRequested"
        ),
        "xblock_post_residual_equation_steps_accepted": (
            "linearSolverXBlockPostResidualEquationStepsAccepted"
        ),
        "xblock_post_residual_equation_direction_count": (
            "linearSolverXBlockPostResidualEquationDirectionCount"
        ),
    }
    for metadata_key, output_key in int_fields.items():
        if metadata_key in solver_metadata and solver_metadata[metadata_key] is not None:
            data[output_key] = np.asarray(
                int(solver_metadata[metadata_key]), dtype=np.int64
            )
    if "sparse_pattern_avg_row_nnz" in solver_metadata:
        data["linearSolverSparsePatternAvgRowNnz"] = np.asarray(
            float(solver_metadata["sparse_pattern_avg_row_nnz"]),
            dtype=np.float64,
        )
    if "sparse_pc_xblock_assembled_host" in solver_metadata:
        data["linearSolverSparsePCXBlockAssembledHost"] = _fortran_logical(
            bool(solver_metadata["sparse_pc_xblock_assembled_host"])
        )
    direct_tail_pc_key = "sparse_pc_fortran_reduced_direct_tail_structured_pc_metadata"
    direct_tail_pc_metadata = solver_metadata.get(direct_tail_pc_key)
    if isinstance(direct_tail_pc_metadata, dict):
        selected_kind = direct_tail_pc_metadata.get("kind")
        if selected_kind is not None:
            data["linearSolverSparsePCSelectedKind"] = str(selected_kind)
        nested_metadata = direct_tail_pc_metadata.get("metadata")
        if isinstance(nested_metadata, dict):
            for metadata_key, output_key in (
                ("factor_kind", "linearSolverSparsePCFactorKind"),
                ("permc_spec", "linearSolverSparsePCPermcSpec"),
                ("permc_spec_requested", "linearSolverSparsePCPermcSpecRequested"),
            ):
                value = nested_metadata.get(metadata_key)
                if value is not None:
                    data[output_key] = str(value)
            candidates_json = _compact_json_metadata(
                nested_metadata.get("permc_spec_candidates", ())
            )
            if candidates_json is not None:
                data["linearSolverSparsePCPermcSpecCandidatesJson"] = candidates_json
            failures_json = _compact_json_metadata(
                nested_metadata.get("permc_failures", ())
            )
            if failures_json is not None:
                data["linearSolverSparsePCPermcFailuresJson"] = failures_json
    if "xblock_initial_seed_used" in solver_metadata:
        data["linearSolverXBlockInitialSeedUsed"] = _fortran_logical(
            bool(solver_metadata["xblock_initial_seed_used"])
        )
    if "xblock_initial_seed_residual_norm" in solver_metadata:
        value = solver_metadata["xblock_initial_seed_residual_norm"]
        if value is not None:
            data["linearSolverXBlockInitialSeedResidualNorm"] = np.asarray(
                float(value), dtype=np.float64
            )
    if "xblock_initial_seed_residual_ratio" in solver_metadata:
        value = solver_metadata["xblock_initial_seed_residual_ratio"]
        if value is not None:
            data["linearSolverXBlockInitialSeedResidualRatio"] = np.asarray(
                float(value), dtype=np.float64
            )
    if "xblock_post_minres_residual_before" in solver_metadata:
        value = solver_metadata["xblock_post_minres_residual_before"]
        if value is not None:
            data["linearSolverXBlockPostMinresResidualBefore"] = np.asarray(
                float(value), dtype=np.float64
            )
    if "xblock_post_minres_residual_after" in solver_metadata:
        value = solver_metadata["xblock_post_minres_residual_after"]
        if value is not None:
            data["linearSolverXBlockPostMinresResidualAfter"] = np.asarray(
                float(value), dtype=np.float64
            )
    if "xblock_post_coarse_residual_before" in solver_metadata:
        value = solver_metadata["xblock_post_coarse_residual_before"]
        if value is not None:
            data["linearSolverXBlockPostCoarseResidualBefore"] = np.asarray(
                float(value), dtype=np.float64
            )
    if "xblock_post_coarse_residual_after" in solver_metadata:
        value = solver_metadata["xblock_post_coarse_residual_after"]
        if value is not None:
            data["linearSolverXBlockPostCoarseResidualAfter"] = np.asarray(
                float(value), dtype=np.float64
            )
    if "xblock_post_residual_equation_residual_before" in solver_metadata:
        value = solver_metadata["xblock_post_residual_equation_residual_before"]
        if value is not None:
            data["linearSolverXBlockPostResidualEquationResidualBefore"] = np.asarray(
                float(value),
                dtype=np.float64,
            )
    if "xblock_post_residual_equation_residual_after" in solver_metadata:
        value = solver_metadata["xblock_post_residual_equation_residual_after"]
        if value is not None:
            data["linearSolverXBlockPostResidualEquationResidualAfter"] = np.asarray(
                float(value),
                dtype=np.float64,
            )
    support_key = "sparse_pc_fortran_reduced_direct_tail_support_mode_preflight_metadata"
    support_metadata = solver_metadata.get(support_key)
    if isinstance(support_metadata, dict):
        data["linearSolverDirectTailSupportModePreflightRequested"] = _fortran_logical(
            bool(
                solver_metadata.get(
                    "sparse_pc_fortran_reduced_direct_tail_support_mode_preflight_requested",
                    False,
                )
            )
        )
        data["linearSolverDirectTailSupportModePreflightSelected"] = _fortran_logical(
            bool(
                solver_metadata.get(
                    "sparse_pc_fortran_reduced_direct_tail_support_mode_preflight_selected",
                    False,
                )
            )
        )
        data["linearSolverDirectTailSupportModeAcceptedNonbaseline"] = _fortran_logical(
            bool(support_metadata.get("accepted_nonbaseline", False))
        )
        selected_candidate = support_metadata.get("selected_candidate")
        if selected_candidate is not None:
            data["linearSolverDirectTailSupportModeSelectedCandidate"] = str(
                selected_candidate
            )
        candidate_specs = support_metadata.get("candidate_specs")
        if isinstance(candidate_specs, (list, tuple)):
            data["linearSolverDirectTailSupportModeRequestedCandidateCount"] = np.asarray(
                len(candidate_specs),
                dtype=np.int32,
            )
        evaluated_candidates = support_metadata.get("candidates")
        if isinstance(evaluated_candidates, (list, tuple)):
            data["linearSolverDirectTailSupportModeCandidateCount"] = np.asarray(
                len(evaluated_candidates),
                dtype=np.int32,
            )
        for metadata_key, output_key in (
            (
                "baseline_residual_after",
                "linearSolverDirectTailSupportModeBaselineResidualAfter",
            ),
            (
                "best_residual_after",
                "linearSolverDirectTailSupportModeBestResidualAfter",
            ),
            ("rhs_norm", "linearSolverDirectTailSupportModeRhsNorm"),
            ("setup_s", "linearSolverDirectTailSupportModeSetupTime"),
        ):
            value = _metadata_float(support_metadata, metadata_key)
            if value is not None:
                data[output_key] = np.asarray(float(value), dtype=np.float64)
        candidate_json = _compact_json_metadata(support_metadata.get("candidates", ()))
        if candidate_json is not None:
            data["linearSolverDirectTailSupportModeCandidatesJson"] = candidate_json
    elif (
        solver_metadata.get(
            "sparse_pc_fortran_reduced_direct_tail_support_mode_preflight_requested"
        )
        is not None
    ):
        data["linearSolverDirectTailSupportModePreflightRequested"] = _fortran_logical(
            bool(
                solver_metadata.get(
                    "sparse_pc_fortran_reduced_direct_tail_support_mode_preflight_requested",
                    False,
                )
            )
        )
        data["linearSolverDirectTailSupportModePreflightSelected"] = _fortran_logical(
            False
        )
    if "sparse_pc_fortran_reduced_direct_tail_structured_pc_max_mb" in solver_metadata:
        value = _metadata_float(
            solver_metadata,
            "sparse_pc_fortran_reduced_direct_tail_structured_pc_max_mb",
        )
        if value is not None:
            data["linearSolverDirectTailStructuredPCMaxMB"] = np.asarray(
                float(value), dtype=np.float64
            )
        data["linearSolverDirectTailStructuredPCMaxMBAuto"] = _fortran_logical(
            bool(
                solver_metadata.get(
                    "sparse_pc_fortran_reduced_direct_tail_structured_pc_max_mb_auto",
                    False,
                )
            )
        )
    if "sparse_pc_direct_tail_true_coupled_coarse_requested" in solver_metadata:
        data["linearSolverDirectTailTrueCoupledCoarseRequested"] = _fortran_logical(
            bool(solver_metadata.get("sparse_pc_direct_tail_true_coupled_coarse_requested", False))
        )
        data["linearSolverDirectTailTrueCoupledCoarseExplicitRequested"] = _fortran_logical(
            bool(
                solver_metadata.get(
                    "sparse_pc_direct_tail_true_coupled_coarse_explicit_requested",
                    False,
                )
            )
        )
        data["linearSolverDirectTailTrueCoupledCoarseAutoEnabled"] = _fortran_logical(
            bool(
                solver_metadata.get(
                    "sparse_pc_direct_tail_true_coupled_coarse_auto_enabled", False
                )
            )
        )
        data["linearSolverDirectTailTrueCoupledCoarseAutoSelected"] = _fortran_logical(
            bool(
                solver_metadata.get(
                    "sparse_pc_direct_tail_true_coupled_coarse_auto_selected", False
                )
            )
        )
        data["linearSolverDirectTailTrueCoupledCoarseSelected"] = _fortran_logical(
            bool(solver_metadata.get("sparse_pc_direct_tail_true_coupled_coarse_selected", False))
        )
        value = _metadata_float(
            solver_metadata,
            "sparse_pc_direct_tail_true_coupled_coarse_auto_target_ratio",
        )
        if value is not None:
            data["linearSolverDirectTailTrueCoupledCoarseAutoTargetRatio"] = np.asarray(
                float(value),
                dtype=np.float64,
            )
        if (
            solver_metadata.get(
                "sparse_pc_direct_tail_true_coupled_coarse_auto_min_size"
            )
            is not None
        ):
            data["linearSolverDirectTailTrueCoupledCoarseAutoMinSize"] = np.asarray(
                int(solver_metadata["sparse_pc_direct_tail_true_coupled_coarse_auto_min_size"]),
                dtype=np.int64,
            )
        value = _metadata_float(
            solver_metadata,
            "sparse_pc_direct_tail_true_coupled_coarse_residual_after",
        )
        if value is not None:
            data["linearSolverDirectTailTrueCoupledCoarseResidualAfter"] = np.asarray(
                float(value),
                dtype=np.float64,
            )
        true_coupled_metadata = solver_metadata.get(
            "sparse_pc_direct_tail_true_coupled_coarse_metadata"
        )
        if isinstance(true_coupled_metadata, dict):
            for metadata_key, output_key in (
                (
                    "base_residual_after",
                    "linearSolverDirectTailTrueCoupledCoarseBaseResidualAfter",
                ),
                ("coarse_size", "linearSolverDirectTailTrueCoupledCoarseSize"),
                (
                    "factor_nbytes_estimate",
                    "linearSolverDirectTailTrueCoupledCoarseNbytesEstimate",
                ),
            ):
                value_float = _metadata_float(true_coupled_metadata, metadata_key)
                if value_float is not None:
                    dtype = (
                        np.int64
                        if metadata_key in {"coarse_size", "factor_nbytes_estimate"}
                        else np.float64
                    )
                    data[output_key] = np.asarray(value_float, dtype=dtype)
            basis_json = _compact_json_metadata(
                true_coupled_metadata.get("basis_names", ())
            )
            if basis_json is not None:
                data["linearSolverDirectTailTrueCoupledCoarseBasisJson"] = basis_json
    if float(residual_target) > 0.0:
        data["linearSolverResidualTargetRatio"] = np.asarray(
            float(residual_norm) / float(residual_target),
            dtype=np.float64,
        )


def _profile_memory_summary(
    profiler: Any | None,
) -> tuple[float | None, float | None, float | None]:
    """Return active RSS, device peak, and process peak memory from profiler entries."""

    if profiler is None or not getattr(profiler, "entries", None):
        return None, None, None
    active_vals: list[float] = []
    device_vals: list[float] = []
    peak_vals: list[float] = []
    for entry in getattr(profiler, "entries"):
        try:
            if entry.get("dpeak_rss_mb") is not None:
                active_vals.append(float(entry["dpeak_rss_mb"]))
            elif entry.get("drss_mb") is not None:
                active_vals.append(float(entry["drss_mb"]))
        except (TypeError, ValueError):
            pass
        try:
            if entry.get("device_mb") is not None:
                device_vals.append(float(entry["device_mb"]))
        except (TypeError, ValueError):
            pass
        for key in ("rss_mb", "peak_rss_mb"):
            try:
                if entry.get(key) is not None:
                    peak_vals.append(float(entry[key]))
            except (TypeError, ValueError):
                pass
    active_rss_mb = max(active_vals) if active_vals else None
    device_peak_mb = max(device_vals) if device_vals else None
    peak_rss_mb = max(peak_vals) if peak_vals else None
    return active_rss_mb, device_peak_mb, peak_rss_mb


def _solver_trace_memory_estimate(
    *,
    total_size: int | None,
    active_size: int | None,
    solver_metadata: dict[str, Any],
    device_count: int | None,
) -> dict[str, int | None] | None:
    """Build conservative memory estimates for solver trace fields."""

    unknowns = total_size if total_size is not None else active_size
    if unknowns is None or int(unknowns) <= 0:
        return None
    restart = (
        _metadata_int(solver_metadata, "gmres_restart")
        or _metadata_int(solver_metadata, "restart")
        or _metadata_int(solver_metadata, "inner_m")
        or 80
    )
    csr_nnz = (
        _metadata_int(solver_metadata, "sparse_pattern_nnz")
        or _metadata_int(solver_metadata, "csr_nnz")
        or None
    )
    estimate = estimate_linear_solve_memory(
        unknowns=int(unknowns),
        gmres_restart=int(restart),
        csr_nnz=csr_nnz,
        preconditioner_nbytes=_metadata_int(
            solver_metadata,
            "sparse_pc_factor_nbytes_estimate",
        ),
        device_count=1 if device_count is None else max(1, int(device_count)),
    )
    return {
        "dense_operator_nbytes": int(estimate.dense_operator_nbytes),
        "csr_operator_nbytes": (
            None
            if estimate.csr_operator_nbytes is None
            else int(estimate.csr_operator_nbytes)
        ),
        "gmres_basis_nbytes": int(estimate.gmres_basis_nbytes),
        "preconditioner_nbytes": estimate.preconditioner_nbytes,
        "dense_total_nbytes": int(estimate.dense_total_nbytes),
        "csr_total_nbytes": (
            None if estimate.csr_total_nbytes is None else int(estimate.csr_total_nbytes)
        ),
        "dense_per_device_nbytes": int(estimate.dense_per_device_nbytes),
        "csr_per_device_nbytes": (
            None
            if estimate.csr_per_device_nbytes is None
            else int(estimate.csr_per_device_nbytes)
        ),
    }


def _write_nonconverged_rhsmode1_solver_trace_json(
    *,
    solver_trace_path: Path,
    input_namelist: Path,
    output_path: Path,
    output_format: str,
    rhs_mode: int,
    geom_scheme_hint: int | None,
    compute_solution: bool,
    compute_transport_matrix: bool,
    differentiable: bool | None,
    result: Any,
    op_fallback: Any,
    solver_tol: float,
    solve_method: str,
    residual_norm: float | None,
    residual_target: float | None,
    active_total_size: int,
    run_t0: float,
    profiler: Any | None = None,
) -> None:
    """Write a JSON trace before refusing nonconverged RHSMode=1 diagnostics."""

    try:
        import jax  # noqa: PLC0415

        backend = str(jax.default_backend())
        device_count = len(jax.devices())
    except Exception:
        backend = "unknown"
        device_count = None

    trace_op = getattr(result, "op", None)
    if trace_op is None:
        trace_op = getattr(result, "op0", None)
    if trace_op is None:
        trace_op = op_fallback

    trace_total_size = None
    trace_active_size = None
    trace_collision_operator = None
    if trace_op is not None:
        try:
            trace_total_size = int(getattr(trace_op, "total_size"))
        except Exception:
            trace_total_size = None
        trace_active_size = _rhs1_active_size_for_trace(trace_op)
        if trace_active_size is None:
            try:
                trace_active_size = int(getattr(trace_op, "active_size"))
            except Exception:
                trace_active_size = trace_total_size
        try:
            trace_collision_operator = str(getattr(trace_op, "collision_operator"))
        except Exception:
            trace_collision_operator = None
    if trace_active_size is None:
        trace_active_size = int(active_total_size)

    solver_metadata = _solver_metadata_dict(result)
    if residual_target is None:
        rhs_vec = getattr(result, "rhs", None)
        if rhs_vec is not None:
            try:
                residual_target = max(
                    0.0,
                    float(solver_tol) * float(np.linalg.norm(np.asarray(rhs_vec))),
                )
            except Exception:
                residual_target = None

    trace_metadata: dict[str, object] = {
        "input_namelist": str(input_namelist.resolve()),
        "output_path": str(output_path.resolve()),
        "output_format": str(output_format),
        "compute_solution": bool(compute_solution),
        "compute_transport_matrix": bool(compute_transport_matrix),
        "differentiable": None if differentiable is None else bool(differentiable),
        "output_refused": True,
        "failure_reason": "nonconverged_rhsmode1_output",
        "solver_metadata": solver_metadata,
    }
    if "accepted_converged" in solver_metadata:
        trace_metadata["accepted_converged"] = bool(
            solver_metadata["accepted_converged"]
        )
    if "acceptance_criterion" in solver_metadata:
        trace_metadata["acceptance_criterion"] = str(
            solver_metadata["acceptance_criterion"]
        )
    if residual_norm is not None and residual_target is not None:
        trace_metadata["converged"] = bool(float(residual_norm) <= float(residual_target))
    if profiler is not None and getattr(profiler, "entries", None):
        trace_metadata["profile_entries"] = list(getattr(profiler, "entries"))

    try:
        from ..profiling import _peak_rss_mb, _rss_mb  # noqa: PLC0415

        peak_rss_mb = _peak_rss_mb()
        if peak_rss_mb is None:
            peak_rss_mb = _rss_mb()
    except Exception:
        peak_rss_mb = None
    active_rss_mb = None
    device_peak_mb = None
    if profiler is not None and getattr(profiler, "entries", None):
        active_rss_mb, device_peak_mb, profiler_peak_rss_mb = _profile_memory_summary(
            profiler
        )
        if profiler_peak_rss_mb is not None:
            peak_rss_mb = profiler_peak_rss_mb

    memory_estimate = _solver_trace_memory_estimate(
        total_size=trace_total_size,
        active_size=trace_active_size,
        solver_metadata=solver_metadata,
        device_count=device_count,
    )
    if memory_estimate is not None:
        trace_metadata["memory_estimate"] = memory_estimate

    trace = SolverTrace(
        backend=backend,
        rhs_mode=int(rhs_mode),
        selected_path="rhsmode1_solution" if bool(compute_solution) else "geometry_only",
        solve_method=str(solve_method),
        preconditioner=(
            None
            if "preconditioner_kind" not in solver_metadata
            else str(solver_metadata["preconditioner_kind"])
        ),
        geometry_scheme=int(geom_scheme_hint) if geom_scheme_hint is not None else None,
        collision_operator=trace_collision_operator,
        total_size=trace_total_size,
        active_size=trace_active_size,
        device_count=device_count,
        residual_norm=residual_norm,
        residual_target=residual_target,
        converged=False if residual_norm is not None and residual_target is not None else None,
        elapsed_s=float(time.perf_counter() - run_t0),
        setup_s=(
            float(solver_metadata["setup_s"]) if "setup_s" in solver_metadata else None
        ),
        solve_s=(
            float(solver_metadata["solve_s"]) if "solve_s" in solver_metadata else None
        ),
        peak_rss_mb=peak_rss_mb,
        active_rss_mb=active_rss_mb,
        device_peak_mb=device_peak_mb,
        estimated_dense_nbytes=(
            None
            if memory_estimate is None
            else int(memory_estimate["dense_operator_nbytes"])
        ),
        estimated_csr_nbytes=(
            None
            if memory_estimate is None or memory_estimate["csr_operator_nbytes"] is None
            else int(memory_estimate["csr_operator_nbytes"])
        ),
        estimated_gmres_basis_nbytes=(
            None if memory_estimate is None else int(memory_estimate["gmres_basis_nbytes"])
        ),
        matvec_count=_metadata_int(solver_metadata, "matvecs"),
        metadata=trace_metadata,
    )
    write_solver_trace_json(solver_trace_path, trace)


def write_output_solver_trace_json(
    *,
    solver_trace_path: Path,
    input_namelist: Path,
    output_path: Path,
    output_format: str,
    rhs_mode: int,
    geom_scheme_hint: int | None,
    compute_solution: bool,
    compute_transport_matrix: bool,
    differentiable: bool | None,
    result: Any,
    nml: Any,
    solver_tol: float,
    solve_method: str | None,
    run_t0: float,
    profiler: Any | None = None,
) -> None:
    """Write the final solver-trace sidecar for geometry, RHSMode=1, and transport runs."""

    try:
        import jax  # noqa: PLC0415

        backend = str(jax.default_backend())
        device_count = len(jax.devices())
    except Exception:
        backend = "unknown"
        device_count = None

    if bool(compute_transport_matrix):
        selected_path = "transport_matrix"
    elif bool(compute_solution):
        selected_path = "rhsmode1_solution"
    else:
        selected_path = "geometry_only"

    trace_op = None
    trace_residual_norm = None
    trace_residual_target = None
    trace_converged = None
    trace_metadata: dict[str, object] = {
        "input_namelist": str(input_namelist.resolve()),
        "output_path": str(output_path.resolve()),
        "output_format": str(output_format),
        "compute_solution": bool(compute_solution),
        "compute_transport_matrix": bool(compute_transport_matrix),
        "differentiable": None if differentiable is None else bool(differentiable),
    }

    if result is not None:
        solver_metadata = _solver_metadata_dict(result)
        if solver_metadata:
            trace_metadata["solver_metadata"] = solver_metadata
            if "accepted_converged" in solver_metadata:
                trace_metadata["accepted_converged"] = bool(
                    solver_metadata["accepted_converged"]
                )
            if "acceptance_criterion" in solver_metadata:
                trace_metadata["acceptance_criterion"] = str(
                    solver_metadata["acceptance_criterion"]
                )
        trace_op = getattr(result, "op", None)
        if trace_op is None:
            trace_op = getattr(result, "op0", None)
        if hasattr(result, "residual_norm"):
            try:
                trace_residual_norm = float(np.asarray(getattr(result, "residual_norm")))
            except Exception:
                trace_residual_norm = None
        rhs_vec = getattr(result, "rhs", None)
        if rhs_vec is not None:
            try:
                trace_residual_target = max(
                    0.0,
                    float(solver_tol) * float(np.linalg.norm(np.asarray(rhs_vec))),
                )
            except Exception:
                trace_residual_target = None
        residuals_by_rhs = getattr(result, "residual_norms_by_rhs", None)
        if isinstance(residuals_by_rhs, dict) and residuals_by_rhs:
            vals = []
            for val in residuals_by_rhs.values():
                try:
                    vals.append(float(np.asarray(val)))
                except Exception:
                    continue
            if vals:
                trace_metadata["residual_norms_by_rhs"] = vals
                trace_residual_norm = max(vals)
        rhs_norms_by_rhs = getattr(result, "rhs_norms_by_rhs", None)
        if isinstance(rhs_norms_by_rhs, dict) and rhs_norms_by_rhs:
            rhs_vals = []
            for val in rhs_norms_by_rhs.values():
                try:
                    rhs_vals.append(float(np.asarray(val)))
                except Exception:
                    continue
            if rhs_vals:
                trace_metadata["rhs_norms_by_rhs"] = rhs_vals
                trace_residual_target = max(0.0, float(solver_tol) * max(rhs_vals))
        elapsed_by_rhs = getattr(result, "elapsed_time_s", None)
        if elapsed_by_rhs is not None:
            try:
                trace_metadata["elapsed_time_s_by_rhs"] = [
                    float(v) for v in np.asarray(elapsed_by_rhs).reshape((-1,))
                ]
            except Exception:
                pass
        solver_kinds_by_rhs = getattr(result, "solver_kinds_by_rhs", None)
        if isinstance(solver_kinds_by_rhs, dict) and solver_kinds_by_rhs:
            trace_metadata["solver_kinds_by_rhs"] = {
                str(int(k)): str(v)
                for k, v in sorted(solver_kinds_by_rhs.items(), key=lambda item: int(item[0]))
            }
        solve_methods_by_rhs = getattr(result, "solve_methods_by_rhs", None)
        if isinstance(solve_methods_by_rhs, dict) and solve_methods_by_rhs:
            trace_metadata["solve_methods_by_rhs"] = {
                str(int(k)): str(v)
                for k, v in sorted(solve_methods_by_rhs.items(), key=lambda item: int(item[0]))
            }
        preconditioner_kind = getattr(result, "preconditioner_kind", None)
        if preconditioner_kind is not None:
            trace_metadata["preconditioner_kind"] = str(preconditioner_kind)
        strong_preconditioner_kind = getattr(result, "strong_preconditioner_kind", None)
        if strong_preconditioner_kind is not None:
            trace_metadata["strong_preconditioner_kind"] = str(strong_preconditioner_kind)

    if trace_residual_norm is not None and trace_residual_target is not None:
        trace_converged = bool(float(trace_residual_norm) <= float(trace_residual_target))
        trace_metadata["converged"] = trace_converged
    if profiler is not None and getattr(profiler, "entries", None):
        trace_metadata["profile_entries"] = list(getattr(profiler, "entries"))

    trace_total_size = None
    trace_active_size = None
    trace_collision_operator = None
    if trace_op is not None:
        try:
            trace_total_size = int(getattr(trace_op, "total_size"))
        except Exception:
            trace_total_size = None
        trace_active_size = _rhs1_active_size_for_trace(trace_op)
        if trace_active_size is None:
            try:
                trace_active_size = int(getattr(trace_op, "active_size"))
            except Exception:
                trace_active_size = trace_total_size
        try:
            trace_collision_operator = str(getattr(trace_op, "collision_operator"))
        except Exception:
            try:
                trace_collision_operator = str(nml.group("physicsParameters").get("COLLISIONOPERATOR"))
            except Exception:
                trace_collision_operator = None

    try:
        from ..profiling import _peak_rss_mb, _rss_mb  # noqa: PLC0415

        peak_rss_mb = _peak_rss_mb()
        if peak_rss_mb is None:
            peak_rss_mb = _rss_mb()
    except Exception:
        peak_rss_mb = None
    active_rss_mb = None
    device_peak_mb = None
    if profiler is not None and getattr(profiler, "entries", None):
        active_rss_mb, device_peak_mb, profiler_peak_rss_mb = _profile_memory_summary(profiler)
        if profiler_peak_rss_mb is not None:
            peak_rss_mb = profiler_peak_rss_mb

    trace_solver_metadata = _solver_metadata_dict(result) if result is not None else {}
    trace_memory_estimate = _solver_trace_memory_estimate(
        total_size=trace_total_size,
        active_size=trace_active_size,
        solver_metadata=trace_solver_metadata,
        device_count=device_count,
    )
    if trace_memory_estimate is not None:
        trace_metadata["memory_estimate"] = trace_memory_estimate
    trace_setup_s = (
        float(trace_solver_metadata["setup_s"])
        if "setup_s" in trace_solver_metadata
        else None
    )
    trace_solve_s = (
        float(trace_solver_metadata["solve_s"])
        if "solve_s" in trace_solver_metadata
        else None
    )
    trace_solve_method = (
        str(solve_method)
        if solve_method is not None
        else str(trace_solver_metadata.get("solver_kind", "auto"))
    )
    trace = SolverTrace(
        backend=backend,
        rhs_mode=int(rhs_mode),
        selected_path=selected_path,
        solve_method=trace_solve_method,
        preconditioner=(
            None
            if "preconditioner_kind" not in trace_solver_metadata
            else str(trace_solver_metadata["preconditioner_kind"])
        ),
        geometry_scheme=int(geom_scheme_hint) if geom_scheme_hint is not None else None,
        collision_operator=trace_collision_operator,
        total_size=trace_total_size,
        active_size=trace_active_size,
        device_count=device_count,
        residual_norm=trace_residual_norm,
        residual_target=trace_residual_target,
        converged=trace_converged,
        elapsed_s=float(time.perf_counter() - run_t0),
        setup_s=trace_setup_s,
        solve_s=trace_solve_s,
        peak_rss_mb=peak_rss_mb,
        active_rss_mb=active_rss_mb,
        device_peak_mb=device_peak_mb,
        estimated_dense_nbytes=(
            None
            if trace_memory_estimate is None
            else int(trace_memory_estimate["dense_operator_nbytes"])
        ),
        estimated_csr_nbytes=(
            None
            if trace_memory_estimate is None or trace_memory_estimate["csr_operator_nbytes"] is None
            else int(trace_memory_estimate["csr_operator_nbytes"])
        ),
        estimated_gmres_basis_nbytes=(
            None if trace_memory_estimate is None else int(trace_memory_estimate["gmres_basis_nbytes"])
        ),
        matvec_count=_metadata_int(trace_solver_metadata, "matvecs"),
        metadata=trace_metadata,
    )
    write_solver_trace_json(solver_trace_path, trace)


__all__ = (
    "RHSMode1SolveMethodSelectionContext",
    "_add_rhsmode1_solver_diagnostics",
    "_compact_json_metadata",
    "_metadata_float",
    "_metadata_int",
    "_maybe_align_pas_no_phi1_flow_diagnostics_to_fortran",
    "_maybe_apply_constraint0_fortran_gauge",
    "_maybe_apply_pas_no_phi1_output_scale",
    "_profile_memory_summary",
    "_align_phi1_history_for_output",
    "_phi1_fast_explicit_gmres_restart_default",
    "_raise_for_nonconverged_rhsmode1_output",
    "_rhs1_active_size_for_trace",
    "_rhsmode1_result_residual_and_target",
    "_select_phi1_newton_linear_solve_method",
    "_select_phi1_use_frozen_linearization",
    "_should_fail_nonconverged_rhsmode1_output",
    "_solver_metadata_dict",
    "_solver_trace_memory_estimate",
    "_select_rhsmode1_linear_solve_method",
    "select_rhsmode1_solve_method",
    "write_rhsmode1_classical_fluxes_to_data",
    "write_rhsmode1_core_diagnostics_to_data",
    "write_rhsmode1_electric_drift_diagnostics_to_data",
    "write_rhsmode1_flux_coordinate_variants_to_data",
    "write_rhsmode1_ntv_diagnostics_to_data",
    "write_rhsmode1_phi1_diagnostics_to_data",
    "write_output_solver_trace_json",
    "_write_nonconverged_rhsmode1_solver_trace_json",
)
