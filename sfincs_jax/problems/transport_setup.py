"""Setup policy for RHSMode=2/3 transport solves."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import math
import os
from typing import Any

import jax.numpy as jnp

from sfincs_jax.operators.profile_system import (
    _operator_signature_cached,
    apply_v3_full_system_operator_cached,
)
from sfincs_jax.problems.transport_diagnostics import transport_matrix_size_from_rhs_mode
from sfincs_jax.problems.transport_policies import transport_residual_gate_failure
from sfincs_jax.solvers.krylov import recycled_initial_guess
from sfincs_jax.solvers.diagnostics import transport_progress_message


StateLoader = Callable[..., dict[str, Any] | None]


@dataclass(frozen=True)
class TransportMaxiterSetup:
    """Resolved transport max-iteration value and user-facing notes."""

    maxiter: int | None
    notes: tuple[tuple[int, str], ...] = ()


@dataclass(frozen=True)
class TransportStateSetup:
    """Resolved transport Krylov state input/output settings."""

    state_in_path: str
    state_out_path: str
    x0: jnp.ndarray | None
    x0_by_rhs: dict[int, jnp.ndarray] | None
    state_x_by_rhs: dict[int, jnp.ndarray] | None


@dataclass(frozen=True)
class TransportWhichRHSSetup:
    """Normalized RHSMode=2/3 transport drive selection."""

    rhs_mode: int
    n_rhs: int
    which_rhs_values: list[int]
    subset_mode: bool


@dataclass(frozen=True)
class TransportParallelRequest:
    """Resolved process/GPU parallel request before parent-side dispatch."""

    parallel_child: bool
    parallel_workers: int
    parallel_backend: str


MatvecFn = Callable[[jnp.ndarray], jnp.ndarray]
EmitFn = Callable[[int, str], None]


@dataclass
class TransportMatvecCache:
    """Cache full and active-DOF transport matvec closures by operator signature."""

    use_active_dof_mode: bool
    active_size: int
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None
    apply_operator: Callable[[Any, jnp.ndarray], jnp.ndarray] = apply_v3_full_system_operator_cached
    operator_signature: Callable[[Any], tuple[object, ...]] = _operator_signature_cached
    full_cache: dict[tuple[object, ...], MatvecFn] = field(default_factory=dict)
    reduced_cache: dict[tuple[object, ...], MatvecFn] = field(default_factory=dict)

    def get_full(self, op_matvec: Any) -> MatvecFn:
        """Return a cached full-space matvec for ``op_matvec``."""
        signature = self.operator_signature(op_matvec)
        fn = self.full_cache.get(signature)
        if fn is None:

            def mv(x: jnp.ndarray, op=op_matvec) -> jnp.ndarray:
                return self.apply_operator(op, x)

            self.full_cache[signature] = mv
            fn = mv
        return fn

    def get_reduced(self, op_matvec: Any) -> MatvecFn:
        """Return a cached active-DOF matvec, or the full matvec when inactive."""
        if not self.use_active_dof_mode or self.reduce_full is None or self.expand_reduced is None:
            return self.get_full(op_matvec)
        signature = self.operator_signature(op_matvec)
        key = (signature, int(self.active_size))
        fn = self.reduced_cache.get(key)
        if fn is None:

            def mv(x_reduced: jnp.ndarray, op=op_matvec) -> jnp.ndarray:
                y_full = self.apply_operator(op, self.expand_reduced(x_reduced))
                return self.reduce_full(y_full)

            self.reduced_cache[key] = mv
            fn = mv
        return fn


def recycled_transport_initial_guess(
    rhs_vec: jnp.ndarray,
    basis: Sequence[jnp.ndarray],
    basis_au: Sequence[jnp.ndarray],
) -> jnp.ndarray | None:
    """Return a residual-minimizing recycled initial guess for one transport RHS."""
    return recycled_initial_guess(rhs_vec, basis, basis_au)


@dataclass
class TransportRecycleState:
    """Bounded recycled Krylov bases for full and active-DOF transport solves."""

    k: int
    full_basis: list[jnp.ndarray] = field(default_factory=list)
    full_basis_au: list[jnp.ndarray] = field(default_factory=list)
    reduced_basis: list[jnp.ndarray] = field(default_factory=list)
    reduced_basis_au: list[jnp.ndarray] = field(default_factory=list)

    @property
    def enabled(self) -> bool:
        """Whether recycle candidates should be used."""
        return int(self.k) > 0

    def candidate_full(self, rhs_vec: jnp.ndarray) -> jnp.ndarray | None:
        """Return a recycled full-space initial guess, if available."""
        if not self.enabled:
            return None
        return recycled_transport_initial_guess(rhs_vec, self.full_basis[-int(self.k) :], self.full_basis_au[-int(self.k) :])

    def candidate_reduced(self, rhs_vec: jnp.ndarray) -> jnp.ndarray | None:
        """Return a recycled active-DOF initial guess, if available."""
        if not self.enabled:
            return None
        return recycled_transport_initial_guess(
            rhs_vec,
            self.reduced_basis[-int(self.k) :],
            self.reduced_basis_au[-int(self.k) :],
        )

    def append_full(self, x_full: jnp.ndarray, ax_full: jnp.ndarray) -> None:
        """Append and trim one full-space recycle vector."""
        if not self.enabled:
            return
        self.full_basis.append(x_full)
        self.full_basis_au.append(ax_full)
        self._trim()

    def append_reduced(
        self,
        x_reduced: jnp.ndarray,
        ax_reduced: jnp.ndarray,
        *,
        x_full: jnp.ndarray | None = None,
        ax_full: jnp.ndarray | None = None,
    ) -> None:
        """Append and trim one reduced recycle vector and optional full vector."""
        if not self.enabled:
            return
        self.reduced_basis.append(x_reduced)
        self.reduced_basis_au.append(ax_reduced)
        if x_full is not None and ax_full is not None:
            self.full_basis.append(x_full)
            self.full_basis_au.append(ax_full)
        self._trim()

    def seed_from_state(
        self,
        *,
        state_x_by_rhs: Mapping[int, jnp.ndarray],
        total_size: int,
        active_size: int,
        matvec_cache: TransportMatvecCache,
        op_ref: Any,
    ) -> None:
        """Seed recycle bases from stored transport Krylov state vectors."""
        if not self.enabled:
            return
        mv_ref_full = matvec_cache.get_full(op_ref)
        mv_ref_reduced = matvec_cache.get_reduced(op_ref)
        for which_rhs in sorted(state_x_by_rhs.keys()):
            x_arr = jnp.asarray(state_x_by_rhs[int(which_rhs)])
            if x_arr.shape == (int(total_size),):
                self.full_basis.append(x_arr)
                self.full_basis_au.append(mv_ref_full(x_arr))
                if matvec_cache.use_active_dof_mode and matvec_cache.reduce_full is not None:
                    x_reduced = matvec_cache.reduce_full(x_arr)
                    self.reduced_basis.append(x_reduced)
                    self.reduced_basis_au.append(mv_ref_reduced(x_reduced))
            elif (
                matvec_cache.use_active_dof_mode
                and x_arr.shape == (int(active_size),)
                and matvec_cache.reduce_full is not None
            ):
                self.reduced_basis.append(x_arr)
                self.reduced_basis_au.append(mv_ref_reduced(x_arr))
        self._trim()

    def _trim(self) -> None:
        k = max(0, int(self.k))
        if k <= 0:
            self.full_basis.clear()
            self.full_basis_au.clear()
            self.reduced_basis.clear()
            self.reduced_basis_au.clear()
            return
        if len(self.full_basis) > k:
            self.full_basis = self.full_basis[-k:]
            self.full_basis_au = self.full_basis_au[-k:]
        if len(self.reduced_basis) > k:
            self.reduced_basis = self.reduced_basis[-k:]
            self.reduced_basis_au = self.reduced_basis_au[-k:]


@dataclass
class TransportLoopProgress:
    """Residual-gate and ETA bookkeeping for sequential transport RHS solves."""

    which_rhs_values: Sequence[int]
    rhs_norms: Mapping[int, Any]
    residual_norms: Mapping[int, Any]
    elapsed_s: Any
    abort_max_residual: float
    abort_max_relative_residual: float
    emit: EmitFn | None = None
    progress_message: Callable[..., str] = transport_progress_message
    residual_gate_failure: Callable[..., str | None] = transport_residual_gate_failure
    elapsed_history: list[float] = field(default_factory=list)

    def relative_residual(self, which_rhs: int) -> float:
        """Return the RHS-normalized residual for one completed transport drive."""
        rhs_norm_val = float(self.rhs_norms[int(which_rhs)])
        residual_norm_val = float(self.residual_norms[int(which_rhs)])
        if math.isfinite(rhs_norm_val) and rhs_norm_val > 0.0:
            return residual_norm_val / rhs_norm_val
        return float("nan")

    def residual_failure(self, which_rhs: int) -> str | None:
        """Return the configured residual-gate failure string, if any."""
        if self.abort_max_residual <= 0.0 and self.abort_max_relative_residual <= 0.0:
            return None
        return self.residual_gate_failure(
            which_rhs=int(which_rhs),
            residual_norm=float(self.residual_norms[int(which_rhs)]),
            rhs_norm=float(self.rhs_norms[int(which_rhs)]),
            max_abs=float(self.abort_max_residual),
            max_relative=float(self.abort_max_relative_residual),
        )

    def finish_rhs(self, *, which_rhs: int, rhs_elapsed_s: float, total_elapsed_s: float) -> None:
        """Record, gate, and report one completed sequential transport RHS solve."""
        which_rhs_i = int(which_rhs)
        elapsed = float(rhs_elapsed_s)
        if self.emit is not None:
            rhs_norm_val = float(self.rhs_norms[which_rhs_i])
            residual_norm_val = float(self.residual_norms[which_rhs_i])
            self.emit(
                0,
                f"whichRHS={which_rhs_i}: residual_norm={residual_norm_val:.6e} "
                f"rhs_norm={rhs_norm_val:.6e} relative_residual={self.relative_residual(which_rhs_i):.6e} "
                f"elapsed_s={elapsed:.3f}",
            )
        self.elapsed_s[which_rhs_i - 1] = elapsed
        failure = self.residual_failure(which_rhs_i)
        if failure is not None:
            if self.emit is not None:
                self.emit(
                    1,
                    "solve_v3_transport_matrix_linear_gmres: transport residual gate failed; "
                    f"aborting remaining whichRHS solves ({failure})",
                )
            raise RuntimeError(f"transport residual gate failed: {failure}")
        self.elapsed_history.append(elapsed)
        if self.emit is not None:
            completed_rhs = len(self.elapsed_history)
            avg_rhs_s = float(sum(self.elapsed_history) / max(1, completed_rhs))
            self.emit(
                0,
                self.progress_message(
                    completed=completed_rhs,
                    total=len(self.which_rhs_values),
                    avg_rhs_s=avg_rhs_s,
                    elapsed_s=float(total_elapsed_s),
                ),
            )


def resolve_transport_recycle_k(
    *,
    op: Any,
    use_implicit: bool,
    op_matvec_by_index: Sequence[Any],
    disable_auto_recycle: Callable[..., bool],
    emit: EmitFn | None,
    operator_signature: Callable[[Any], tuple[object, ...]] = _operator_signature_cached,
) -> int:
    """Resolve the bounded recycle-basis size for the transport solve loop."""
    recycle_k_env = os.environ.get("SFINCS_JAX_TRANSPORT_RECYCLE_K", "").strip()
    try:
        recycle_k = int(recycle_k_env) if recycle_k_env else 4
    except ValueError:
        recycle_k = 4
    recycle_k = max(0, int(recycle_k))
    if recycle_k > 0 and disable_auto_recycle(op=op, use_implicit=bool(use_implicit)):
        recycle_k = 0
        if emit is not None:
            emit(
                1,
                "solve_v3_transport_matrix_linear_gmres: auto recycle disabled "
                "for branch-sensitive explicit mono transport",
            )
    if recycle_k > 0 and op_matvec_by_index:
        signature_ref = operator_signature(op_matvec_by_index[0])
        for op_probe in op_matvec_by_index[1:]:
            if operator_signature(op_probe) != signature_ref:
                recycle_k = 0
                if emit is not None:
                    emit(
                        1,
                        "solve_v3_transport_matrix_linear_gmres: recycle disabled "
                        "(matvec operator varies across whichRHS)",
                    )
                break
    return int(recycle_k)


def resolve_transport_maxiter_setup(
    maxiter: int | None,
    *,
    maxiter_env: str | None = None,
) -> TransportMaxiterSetup:
    """Resolve ``SFINCS_JAX_TRANSPORT_MAXITER`` without side effects."""
    env = (
        os.environ.get("SFINCS_JAX_TRANSPORT_MAXITER", "").strip()
        if maxiter_env is None
        else str(maxiter_env).strip()
    )
    if not env:
        return TransportMaxiterSetup(maxiter=maxiter)
    try:
        maxiter_use = max(1, int(env))
    except ValueError:
        return TransportMaxiterSetup(
            maxiter=maxiter,
            notes=(
                (
                    1,
                    "solve_v3_transport_matrix_linear_gmres: ignoring invalid "
                    f"SFINCS_JAX_TRANSPORT_MAXITER={env!r}",
                ),
            ),
        )
    return TransportMaxiterSetup(
        maxiter=maxiter_use,
        notes=(
            (
                1,
                f"solve_v3_transport_matrix_linear_gmres: maxiter override={int(maxiter_use)}",
            ),
        ),
    )


def resolve_transport_state_setup(
    *,
    op: Any,
    x0: jnp.ndarray | None,
    x0_by_rhs: dict[int, jnp.ndarray] | None,
    state_in_env: str | None = None,
    state_out_env: str | None = None,
    load_state: StateLoader | None = None,
) -> TransportStateSetup:
    """Load optional Krylov state and merge it with explicit initial guesses."""
    state_in_path = (
        os.environ.get("SFINCS_JAX_STATE_IN", "").strip()
        if state_in_env is None
        else str(state_in_env).strip()
    )
    state_out_path = (
        os.environ.get("SFINCS_JAX_STATE_OUT", "").strip()
        if state_out_env is None
        else str(state_out_env).strip()
    )
    state_x_by_rhs: dict[int, jnp.ndarray] | None = None
    x0_use = x0
    x0_by_rhs_use = x0_by_rhs
    if state_in_path:
        loader = _load_transport_krylov_state if load_state is None else load_state
        try:
            state = loader(path=state_in_path, op=op)
        except Exception:  # noqa: BLE001 - state files must never make solves fail.
            state = None
        if state:
            state_x_by_rhs = state.get("x_by_rhs")
            if x0_by_rhs_use is None:
                x0_by_rhs_use = state_x_by_rhs
            if x0_use is None:
                x0_use = state.get("x_full")
    return TransportStateSetup(
        state_in_path=state_in_path,
        state_out_path=state_out_path,
        x0=x0_use,
        x0_by_rhs=x0_by_rhs_use,
        state_x_by_rhs=state_x_by_rhs,
    )


def resolve_transport_which_rhs_setup(
    *,
    rhs_mode: int,
    which_rhs_values: Sequence[int] | None,
) -> TransportWhichRHSSetup:
    """Normalize a requested subset of transport ``whichRHS`` drives."""
    n_rhs = transport_matrix_size_from_rhs_mode(int(rhs_mode))
    if which_rhs_values is None:
        values = list(range(1, int(n_rhs) + 1))
    else:
        values = sorted({int(v) for v in which_rhs_values if 1 <= int(v) <= int(n_rhs)})
    return TransportWhichRHSSetup(
        rhs_mode=int(rhs_mode),
        n_rhs=int(n_rhs),
        which_rhs_values=values,
        subset_mode=len(values) < int(n_rhs),
    )


def resolve_transport_parallel_request(
    *,
    which_rhs_count: int,
    n_rhs: int,
    parallel_workers: int | None,
    parallel_backend: str,
    visible_gpu_ids: Callable[[int], Sequence[str]],
    parallel_child_env: str | None = None,
    parallel_env: str | None = None,
    workers_env: str | None = None,
    cpu_count: int | None = None,
) -> TransportParallelRequest:
    """Resolve parent/worker parallel settings before launching workers."""
    parallel_child_raw = (
        os.environ.get("SFINCS_JAX_TRANSPORT_PARALLEL_CHILD", "").strip().lower()
        if parallel_child_env is None
        else str(parallel_child_env).strip().lower()
    )
    parallel_child = parallel_child_raw in {"1", "true", "yes", "on"}
    parallel_raw = (
        os.environ.get("SFINCS_JAX_TRANSPORT_PARALLEL", "").strip().lower()
        if parallel_env is None
        else str(parallel_env).strip().lower()
    )
    if parallel_workers is None:
        workers_raw = (
            os.environ.get("SFINCS_JAX_TRANSPORT_PARALLEL_WORKERS", "").strip()
            if workers_env is None
            else str(workers_env).strip()
        )
        try:
            workers_val = int(workers_raw) if workers_raw else 0
        except ValueError:
            workers_val = 0
        if parallel_raw in {"", "0", "false", "no", "off"}:
            workers = 1
        elif parallel_raw in {"process", "auto", "1", "true", "yes", "on"}:
            if workers_val > 0:
                workers = workers_val
            else:
                count = int(cpu_count if cpu_count is not None else (os.cpu_count() or 1))
                workers = min(count, int(n_rhs)) if int(n_rhs) > 1 else 1
        else:
            workers = 1
    else:
        workers = max(1, int(parallel_workers))
    if workers > 1:
        workers = min(int(workers), int(which_rhs_count))
    backend = str(parallel_backend)
    if backend == "gpu" and workers > 1:
        gpu_ids = list(visible_gpu_ids(int(workers)))
        workers = min(int(workers), len(gpu_ids))
        if workers <= 1:
            backend = "cpu"
    return TransportParallelRequest(
        parallel_child=bool(parallel_child),
        parallel_workers=max(1, int(workers)),
        parallel_backend=backend,
    )


def _load_transport_krylov_state(*, path: str, op: Any) -> dict[str, Any] | None:
    from sfincs_jax.solvers.diagnostics import load_krylov_state  # noqa: PLC0415

    return load_krylov_state(path=path, op=op)


__all__ = [
    "TransportLoopProgress",
    "TransportMatvecCache",
    "TransportMaxiterSetup",
    "TransportParallelRequest",
    "TransportRecycleState",
    "TransportStateSetup",
    "TransportWhichRHSSetup",
    "recycled_transport_initial_guess",
    "resolve_transport_maxiter_setup",
    "resolve_transport_parallel_request",
    "resolve_transport_recycle_k",
    "resolve_transport_state_setup",
    "resolve_transport_which_rhs_setup",
]
