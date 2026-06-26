"""Automatic host-solver routing for RHSMode=1 profile-response solves."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
import os
from typing import Any

import jax.numpy as jnp

from ...namelist import Namelist
from ...operators.profile_response.full_system import solve_structured_rhs1_full_csr
from ...solver import GMRESSolveResult
from sfincs_jax.operators.profile_response.system import (
    V3FullSystemOperator,
    full_system_operator_from_namelist,
    rhs_v3_full_system,
    with_transport_rhs_settings,
)
from .solver_diagnostics import V3LinearSolveResult
from ..transport_matrix.active_dense import transport_active_dof_indices


_FALSE_TOKENS = {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        return int(raw) if raw else int(default)
    except ValueError:
        return int(default)


@dataclass(frozen=True)
class RHS1AutoHostSolveContext:
    """Inputs needed to try non-autodiff RHSMode=1 host solver shortcuts."""

    nml: Any
    which_rhs: int | None
    op: Any
    x0: Any
    tol: float
    atol: float
    restart: int
    maxiter: int | None
    solve_method: str
    identity_shift: float
    phi1_hat_base: Any
    differentiable: bool | None
    emit: Callable[[int, str], None] | None
    recycle_basis: Sequence[Any] | None
    solve_driver: Callable[..., Any]
    solve_method_kind_requested: str
    structured_full_csr_explicit_requested: bool
    use_implicit: bool
    structured_auto_allowed: bool
    structured_sharded_multidevice: bool


@dataclass(frozen=True)
class RHS1StructuredCSRSolveContext:
    """Inputs for the explicit structured full-CSR host solve route."""

    nml: Any
    op: Any
    x0: Any
    rhs_norm: Any
    tol: float
    atol: float
    restart: int
    maxiter: int | None
    solve_method: str
    identity_shift: float
    phi1_hat_base: Any
    differentiable: bool | None
    emit: Callable[[int, str], None] | None
    structured_solver: Callable[..., Any]


@dataclass(frozen=True)
class RHS1SparseHostSafeSolveContext:
    """Inputs for the host sparse solve with constrained-PAS safe fallback."""

    nml: Any
    which_rhs: int | None
    op: Any
    x0: Any
    tol: float
    atol: float
    restart: int
    maxiter: int | None
    identity_shift: float
    phi1_hat_base: Any
    differentiable: bool | None
    emit: Callable[[int, str], None] | None
    recycle_basis: Sequence[Any] | None
    solve_driver: Callable[..., Any]
    solve_method_kind_explicit: str
    requested: bool


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    try:
        return float(raw) if raw else float(default)
    except ValueError:
        return float(default)


def _annotate_auto_result(result: Any, metadata_updates: dict[str, Any]) -> Any:
    metadata = dict(getattr(result, "metadata", None) or {})
    metadata.update(metadata_updates)
    return replace(result, metadata=metadata)


def _try_fortran_reduced_sparse_pc_auto(context: RHS1AutoHostSolveContext) -> Any | None:
    op = context.op
    auto_enabled = os.environ.get("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_AUTO", "").strip().lower()
    if auto_enabled in _FALSE_TOKENS:
        return None

    min_size = max(1, _env_int("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_AUTO_MIN_SIZE", 10_000))
    system_size = int(op.total_size)
    requested = bool(
        context.solve_method_kind_requested in {"auto", "default"}
        and not bool(context.use_implicit)
        and int(op.rhs_mode) == 1
        and not bool(op.include_phi1)
        and int(op.constraint_scheme) == 1
        and op.fblock.fp is not None
        and op.fblock.pas is None
        and abs(float(context.identity_shift)) == 0.0
        and system_size >= min_size
    )
    if not requested:
        return None

    if context.emit is not None:
        context.emit(
            0,
            "solve_v3_full_system_linear_gmres: auto selecting Fortran-reduced "
            "sparse-PC GMRES for large RHSMode=1 full-FP solve "
            f"(system_size={system_size} >= {min_size})",
        )
    result = context.solve_driver(
        nml=context.nml,
        which_rhs=context.which_rhs,
        op=op,
        x0=context.x0,
        tol=context.tol,
        atol=context.atol,
        restart=context.restart,
        maxiter=context.maxiter,
        solve_method="fortran_reduced_pc_gmres",
        identity_shift=context.identity_shift,
        phi1_hat_base=context.phi1_hat_base,
        differentiable=False,
        emit=context.emit,
        recycle_basis=context.recycle_basis,
    )
    return _annotate_auto_result(
        result,
        {
            "solve_method_requested": str(context.solve_method),
            "requested_solve_method": str(context.solve_method),
            "auto_solver_selected": True,
            "auto_solver_policy": "fortran_reduced_pc_gmres",
            "auto_solver_size": system_size,
            "auto_solver_min_size": min_size,
        },
    )


def _try_structured_full_csr_auto(context: RHS1AutoHostSolveContext) -> Any | None:
    if context.structured_full_csr_explicit_requested:
        return None
    requested = bool(
        context.solve_method_kind_requested in {"auto", "default"}
        and not context.structured_sharded_multidevice
        and context.structured_auto_allowed
    )
    if not requested:
        return None

    if context.emit is not None:
        context.emit(0, "solve_v3_full_system_linear_gmres: auto trying structured full CSR host solve")
    try:
        result = context.solve_driver(
            nml=context.nml,
            which_rhs=context.which_rhs,
            op=context.op,
            x0=context.x0,
            tol=context.tol,
            atol=context.atol,
            restart=context.restart,
            maxiter=context.maxiter,
            solve_method="structured_full_csr",
            identity_shift=context.identity_shift,
            phi1_hat_base=context.phi1_hat_base,
            differentiable=False,
            emit=context.emit,
            recycle_basis=context.recycle_basis,
        )
    except RuntimeError as exc:
        if context.emit is not None:
            context.emit(
                1,
                "solve_v3_full_system_linear_gmres: auto structured full CSR skipped "
                f"({exc}); falling back to matrix-free policy",
            )
        return None

    metadata = dict(getattr(result, "metadata", None) or {})
    if bool(metadata.get("accepted_converged", False)):
        return _annotate_auto_result(
            result,
            {
                "solve_method_requested": str(context.solve_method),
                "requested_solve_method": str(context.solve_method),
                "auto_solver_selected": True,
                "auto_solver_policy": "structured_full_csr",
            },
        )
    if context.emit is not None:
        residual = metadata.get("reported_residual_norm", getattr(result.gmres, "residual_norm", 0.0))
        context.emit(
            1,
            "solve_v3_full_system_linear_gmres: auto structured full CSR did not converge "
            f"(residual={float(residual):.3e}); falling back to matrix-free policy",
        )
    return None


def try_rhs1_auto_host_solve(context: RHS1AutoHostSolveContext) -> Any | None:
    """Try early non-autodiff RHSMode=1 host solver routes in priority order."""

    result = _try_fortran_reduced_sparse_pc_auto(context)
    if result is not None:
        return result
    return _try_structured_full_csr_auto(context)


def solve_v3_full_system_structured_csr(
    *,
    nml: Namelist,
    which_rhs: int | None = None,
    op: V3FullSystemOperator | None = None,
    x0: jnp.ndarray | None = None,
    tol: float = 1.0e-10,
    atol: float = 0.0,
    restart: int = 80,
    maxiter: int | None = 400,
    identity_shift: float = 0.0,
    phi1_hat_base: jnp.ndarray | None = None,
    max_csr_nbytes: int | None = None,
    method: str = "gmres",
    preconditioner: str | None = "auto",
    preconditioner_max_schur_size: int = 2048,
    preconditioner_max_block_inverse_nbytes: int = 64 * 1024 * 1024,
    active_dof: bool = False,
    emit: Callable[[int, str], None] | None = None,
) -> V3LinearSolveResult:
    """Solve a supported RHSMode=1 system with explicit host CSR Krylov."""

    if op is None:
        op = full_system_operator_from_namelist(
            nml=nml,
            identity_shift=identity_shift,
            phi1_hat_base=phi1_hat_base,
        )
    if which_rhs is not None:
        op = with_transport_rhs_settings(op, which_rhs=int(which_rhs))
    rhs = rhs_v3_full_system(op)
    active_indices = transport_active_dof_indices(op) if bool(active_dof) else None
    if emit is not None:
        active_msg = (
            f" active_size={int(active_indices.size)}/{int(op.total_size)}"
            if active_indices is not None
            else " full_size"
        )
        emit(
            0,
            "solve_v3_full_system_structured_csr: assembling no-probe host CSR "
            f"(size={int(op.total_size)}{active_msg} method={method} preconditioner={preconditioner})",
        )
    result = solve_structured_rhs1_full_csr(
        op,
        rhs,
        x0=x0,
        tol=tol,
        atol=atol,
        restart=restart,
        maxiter=maxiter,
        method=method,
        preconditioner=preconditioner,
        preconditioner_max_schur_size=preconditioner_max_schur_size,
        preconditioner_max_block_inverse_nbytes=preconditioner_max_block_inverse_nbytes,
        max_csr_nbytes=max_csr_nbytes,
        active_indices=active_indices,
    )
    if emit is not None:
        emit(
            0,
            "solve_v3_full_system_structured_csr: "
            f"converged={bool(result.converged)} residual={float(result.residual_norm):.3e} "
            f"solve_s={float(result.solve_s):.3f}",
        )
        pc_summary = dict(result.metadata.get("preconditioner", {}) or {})
        pc_metadata = dict(pc_summary.get("metadata", {}) or {})
        factor_nbytes = pc_metadata.get("factor_nbytes_actual")
        if factor_nbytes is None:
            factor_nbytes = pc_metadata.get("factor_nbytes_estimate")
        if pc_summary:
            emit(
                0,
                "solve_v3_full_system_structured_csr: "
                f"pc_kind={pc_summary.get('kind', 'unknown')} "
                f"pc_selected={bool(pc_summary.get('selected', False))} "
                f"pc_reason={pc_summary.get('reason', 'unknown')} "
                f"pc_setup_s={float(pc_summary.get('setup_s', 0.0) or 0.0):.3f} "
                f"pc_factor_nbytes={factor_nbytes if factor_nbytes is not None else 'na'} "
                f"pc_permc={pc_metadata.get('permc_spec', 'na')} "
                f"pc_superlu_permc={pc_metadata.get('superlu_permc_spec', 'na')}",
            )
    return V3LinearSolveResult(
        op=op,
        rhs=rhs,
        gmres=GMRESSolveResult(
            x=jnp.asarray(result.x, dtype=jnp.float64),
            residual_norm=jnp.asarray(result.residual_norm, dtype=jnp.float64),
        ),
        metadata={
            "solver_path": "structured_full_csr_host_gmres",
            "structured_full_csr": result.to_dict(),
            "active_dof": bool(active_dof),
        },
    )


def solve_rhs1_structured_full_csr_explicit(context: RHS1StructuredCSRSolveContext) -> Any:
    """Run the explicit host-only structured full-CSR path and normalize metadata."""

    op = context.op
    if context.differentiable is True:
        raise ValueError(
            "solve_method='structured_csr' is host-only/non-differentiable; "
            "use differentiable=False or choose a JAX-native solve method."
        )
    if int(op.rhs_mode) != 1:
        raise ValueError("solve_method='structured_csr' is only implemented for RHSMode=1 full-system solves.")

    csr_max_mb = _env_float("SFINCS_JAX_RHS1_FULL_CSR_MAX_MB", 1024.0)
    pc_max_mb = _env_float("SFINCS_JAX_RHS1_FULL_CSR_PRECONDITIONER_MAX_MB", 128.0)
    pc_kind = os.environ.get("SFINCS_JAX_RHS1_FULL_CSR_PRECONDITIONER", "auto").strip() or "auto"
    pc_schur_max = _env_int("SFINCS_JAX_RHS1_FULL_CSR_PRECONDITIONER_MAX_SCHUR_SIZE", 2048)
    structured_krylov_env = os.environ.get("SFINCS_JAX_RHS1_FULL_CSR_KRYLOV", "").strip().lower()
    structured_krylov_default = "direct" if abs(float(context.identity_shift)) <= 0.0 else "gmres"
    structured_krylov = structured_krylov_env or structured_krylov_default
    active_dof_env = os.environ.get("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_DOF", "").strip().lower()
    if active_dof_env in {"1", "true", "yes", "on", "active"}:
        structured_active_dof = True
    elif active_dof_env in {"0", "false", "no", "off", "full"}:
        structured_active_dof = False
    else:
        structured_active_dof = structured_krylov in {"direct", "splu", "sparse_direct"}

    if context.emit is not None:
        context.emit(
            0,
            "solve_v3_full_system_linear_gmres: using structured full CSR host solve "
            f"(preconditioner={pc_kind} csr_max_mb={csr_max_mb:.3g} pc_max_mb={pc_max_mb:.3g} "
            f"active_dof={structured_active_dof})",
        )
    structured_result = context.structured_solver(
        nml=context.nml,
        which_rhs=None,
        op=op,
        x0=context.x0,
        tol=context.tol,
        atol=context.atol,
        restart=context.restart,
        maxiter=context.maxiter,
        identity_shift=context.identity_shift,
        phi1_hat_base=context.phi1_hat_base,
        max_csr_nbytes=int(max(0.0, float(csr_max_mb)) * 1024.0 * 1024.0),
        method=structured_krylov,
        preconditioner=pc_kind,
        preconditioner_max_schur_size=max(1, int(pc_schur_max)),
        preconditioner_max_block_inverse_nbytes=int(max(0.0, float(pc_max_mb)) * 1024.0 * 1024.0),
        active_dof=bool(structured_active_dof),
        emit=context.emit,
    )

    structured_metadata = dict(structured_result.metadata or {})
    structured_csr_metadata = structured_metadata.get("structured_full_csr", {})
    if not isinstance(structured_csr_metadata, dict):
        structured_csr_metadata = {}
    structured_solve_metadata = structured_csr_metadata.get("metadata", {})
    if not isinstance(structured_solve_metadata, dict):
        structured_solve_metadata = {}
    structured_selection = structured_csr_metadata.get("selection", {})
    if not isinstance(structured_selection, dict):
        structured_selection = {}
    structured_selection_metadata = structured_selection.get("metadata", {})
    if not isinstance(structured_selection_metadata, dict):
        structured_selection_metadata = {}
    structured_preconditioner = structured_solve_metadata.get("preconditioner", {})
    if not isinstance(structured_preconditioner, dict):
        structured_preconditioner = {}
    structured_preconditioner_metadata = structured_preconditioner.get("metadata", {})
    if not isinstance(structured_preconditioner_metadata, dict):
        structured_preconditioner_metadata = {}

    residual_norm = float(structured_csr_metadata.get("residual_norm", structured_result.gmres.residual_norm))
    target = float(structured_solve_metadata.get("target", max(float(context.atol), float(context.tol) * float(context.rhs_norm))))
    converged = bool(structured_csr_metadata.get("converged", residual_norm <= target))
    setup_s = float(structured_preconditioner.get("setup_s", 0.0) or 0.0)
    solve_s = float(structured_csr_metadata.get("solve_s", 0.0) or 0.0)
    direct_factor_s = structured_solve_metadata.get("factor_s", None)
    direct_factor_nbytes = structured_solve_metadata.get("factor_nbytes_actual", None)
    factor_nbytes = structured_preconditioner_metadata.get(
        "factor_nbytes_actual",
        structured_preconditioner_metadata.get("block_inverse_nbytes_actual", direct_factor_nbytes),
    )
    structured_metadata.update(
        {
            "solver_path": "structured_full_csr_host_gmres",
            "solver_kind": "structured_full_csr",
            "solve_method_requested": str(context.solve_method),
            "requested_solve_method": str(context.solve_method),
            "differentiable": False,
            "residual_kind": "true_residual",
            "accepted_converged": bool(converged),
            "acceptance_criterion": "true_residual",
            "reported_residual_norm": float(residual_norm),
            "iterations": len(tuple(structured_csr_metadata.get("residual_history", ()) or ())),
            "info_code": int(structured_csr_metadata.get("info", 0)),
            "setup_s": setup_s,
            "solve_s": solve_s,
            "elapsed_s": setup_s + solve_s,
            "csr_nnz": int(structured_selection_metadata.get("nnz", structured_solve_metadata.get("matrix_nnz", 0)) or 0),
            "csr_operator_nbytes": int(structured_selection_metadata.get("csr_nbytes_actual", 0) or 0),
            "preconditioner_kind": str(structured_preconditioner.get("kind", pc_kind)),
            "sparse_pc_factor_nbytes_estimate": None if factor_nbytes is None else int(factor_nbytes),
            "direct_factor_s": None if direct_factor_s is None else float(direct_factor_s),
            "direct_factor_nbytes_actual": None if direct_factor_nbytes is None else int(direct_factor_nbytes),
            "structured_active_dof": bool(structured_solve_metadata.get("active_dof", False)),
            "structured_active_size": int(structured_solve_metadata.get("active_size", 0) or 0),
            "structured_full_size": int(structured_solve_metadata.get("full_size", 0) or 0),
            "structured_full_csr_env": {
                "csr_max_mb": float(csr_max_mb),
                "preconditioner": str(pc_kind),
                "preconditioner_max_mb": float(pc_max_mb),
                "preconditioner_max_schur_size": int(pc_schur_max),
                "krylov": str(structured_krylov),
                "active_dof": bool(structured_active_dof),
            },
        }
    )
    return replace(structured_result, metadata=structured_metadata)


def try_rhs1_sparse_host_safe_solve(context: RHS1SparseHostSafeSolveContext) -> Any | None:
    """Run ``sparse_host_safe`` or return ``None`` when it was not requested."""

    if not bool(context.requested):
        return None
    try:
        direct_result = context.solve_driver(
            nml=context.nml,
            which_rhs=context.which_rhs,
            op=context.op,
            x0=context.x0,
            tol=context.tol,
            atol=context.atol,
            restart=context.restart,
            maxiter=context.maxiter,
            solve_method="sparse_host",
            identity_shift=context.identity_shift,
            phi1_hat_base=context.phi1_hat_base,
            differentiable=context.differentiable,
            emit=context.emit,
            recycle_basis=context.recycle_basis,
        )
    except RuntimeError as exc:
        if "Host sparse factorization failed" not in str(exc):
            raise
        op = context.op
        constrained_pas = bool(
            int(op.rhs_mode) == 1
            and int(op.constraint_scheme) == 2
            and (not bool(op.include_phi1))
            and op.fblock.pas is not None
        )
        if not constrained_pas:
            raise
        if context.emit is not None:
            context.emit(
                0,
                "solve_v3_full_system_linear_gmres: sparse_host_safe falling back to "
                "PETSc-compatible minimum-norm constrained-PAS branch after sparse LU failure",
            )
        compat_result = context.solve_driver(
            nml=context.nml,
            which_rhs=context.which_rhs,
            op=op,
            x0=context.x0,
            tol=context.tol,
            atol=context.atol,
            restart=context.restart,
            maxiter=context.maxiter,
            solve_method="petsc_compat",
            identity_shift=context.identity_shift,
            phi1_hat_base=context.phi1_hat_base,
            differentiable=context.differentiable,
            emit=context.emit,
            recycle_basis=context.recycle_basis,
        )
        return _annotate_auto_result(
            compat_result,
            {
                "requested_solve_method": str(context.solve_method_kind_explicit),
                "safe_sparse_host_fallback_used": True,
                "sparse_host_failure": str(exc),
            },
        )

    metadata = dict(getattr(direct_result, "metadata", None) or {})
    metadata.update(
        {
            "requested_solve_method": str(context.solve_method_kind_explicit),
            "safe_sparse_host_fallback_used": False,
            "accepted_converged": bool(metadata.get("accepted_converged", True)),
            "acceptance_criterion": metadata.get("acceptance_criterion", "true_residual"),
        }
    )
    return replace(direct_result, metadata=metadata)


__all__ = [
    "RHS1AutoHostSolveContext",
    "RHS1SparseHostSafeSolveContext",
    "RHS1StructuredCSRSolveContext",
    "solve_rhs1_structured_full_csr_explicit",
    "solve_v3_full_system_structured_csr",
    "try_rhs1_sparse_host_safe_solve",
    "try_rhs1_auto_host_solve",
]
