"""Loop-local matvec caching and recycle-basis helpers for transport solves."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import math
import os
from typing import Any

import jax.numpy as jnp

from sfincs_jax.solver import recycled_initial_guess
from sfincs_jax.solvers.progress import transport_progress_message
from sfincs_jax.problems.transport_matrix.residual_quality import transport_residual_gate_failure
from sfincs_jax.operators.profile_response.system import _operator_signature_cached, apply_v3_full_system_operator_cached


EmitFn = Callable[[int, str], None]
MatvecFn = Callable[[jnp.ndarray], jnp.ndarray]


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


__all__ = [
    "TransportLoopProgress",
    "TransportMatvecCache",
    "TransportRecycleState",
    "recycled_transport_initial_guess",
    "resolve_transport_recycle_k",
]
