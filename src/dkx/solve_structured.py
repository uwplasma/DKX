"""Structured direct route of :func:`dkx.solve.solve` (code name ``tier1``).

Block Thomas over Legendre modes, its truncated and generated-full variants,
and their implicit derivatives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Callable
import functools
import time

# The JAX backend is imported below; dkx/runtime.py explains why this is here.
from .runtime import configure as _configure_runtime

_configure_runtime()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

try:  # noqa: E402
    from solvax.direct import (
        GeneratedBlockTridiagFactors,
        block_thomas_factor_fn,
        block_thomas_selected_tail_fn,
        block_thomas_solve,
        block_thomas_truncated_fn,
    )
    from solvax.refine import iterative_refinement
except ImportError:
    GeneratedBlockTridiagFactors = None  # type: ignore[assignment, misc]
    block_thomas_factor_fn = None  # type: ignore[assignment, misc]
    block_thomas_selected_tail_fn = None  # type: ignore[assignment, misc]
    block_thomas_solve = None  # type: ignore[assignment, misc]
    block_thomas_truncated_fn = None  # type: ignore[assignment, misc]
    iterative_refinement = None  # type: ignore[assignment, misc]

from dkx.coarse_precond import (  # noqa: E402
    _require_solvax,
    _truncated_block_fn,
    _truncated_blocks,
    _truncated_coefficients,
    _truncated_params,
)
from dkx.drift_kinetic import (  # noqa: E402
    KineticOperator,
)
from dkx.solve_common import (  # noqa: E402
    AdjointDiagnostics,
    SolveResult,
    _TIER1_REFINEMENT_SWEEPS,
    _as_columns,
    _column_apply,
    _converged_flag,
    _dispatch_guard,
    _guarded_solve,
    _implicit_solve,
    _operator_apply,
    _operator_norm_estimate,
    _pinned_matvecs,
    _refinement_sweeps,
    _register_guard,
    _residual_guard,
    _residual_norms,
    _say,
    _stop_gradient_tree,
)
from dkx.structured_direct import (  # noqa: E402
    CoupledSolver,
    _TIER1_KEEP_LOWEST_DEFAULT,
    _is_traced,
    _resolve_subsystem_batch,
    _tier1_budget_bytes,
    _truncation_supported,
    _uniform_nxi_for_x,
    build_coupled_solver,
    coupled_available,
    tier1_available,
    tier1_truncated_peak_memory_bytes,
)

def build_structured_solver(op: KineticOperator) -> "Tier1Solver | CoupledSolver":
    """Factor the exact structured direct solver: per ``(species, x)`` for
    pitch-angle scattering, speed-coupled for Fokker-Planck and Sugama."""
    if coupled_available(op)[0]:
        return build_coupled_solver(op)
    return build_tier1_solver(op)

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Tier1Solver:
    """Factored per-(species, x) bordered block-tridiagonal solver.

    Holds the batched block-Thomas factors of the rank-one-regularized
    Legendre bands ``A~ = A + gamma B C`` for every (species, x) subsystem,
    plus the presolved border columns ``z = A~^{-1} B`` (forward) and
    ``z_t = A~^{-T} C^T`` (transpose), so both the forward and the adjoint
    bordered solve reuse the same elimination.

    A pytree (the operator's own static layout is its only static data), so
    the compiled kernels below take it as an argument and ``jit`` caches one
    executable per operator structure rather than one per solve.
    """

    op: KineticOperator
    factors: tuple[jnp.ndarray, jnp.ndarray]  # Schur LUs and pivots, leading batch axis B = S*X
    z_fwd: jnp.ndarray  # (B, L, TZ)
    z_t: jnp.ndarray  # (B, L, TZ)
    gamma: jnp.ndarray  # (B,)
    b0: jnp.ndarray  # (TZ,) source column shape on the l=0 rows
    c0: jnp.ndarray  # (TZ,) constraint row (flux-surface-average weights)

    def solve(self, rhs: jnp.ndarray, transpose: bool = False) -> jnp.ndarray:
        """Solve ``K x = rhs`` (or ``K^T x = rhs``) for flat state vector(s).

        Args:
            rhs: ``(total_size,)`` or ``(total_size, n_rhs)``.
            transpose: solve the transposed bordered system, reusing the same
                factors via ``block_thomas_solve(transpose=True)``.

        Returns:
            Solution(s) with the same shape as ``rhs``.
        """
        op = self.op
        rhs2d, squeeze = _as_columns(rhs)
        n_rhs = rhs2d.shape[1]
        n_s, n_x, n_xi, n_t, n_z = op.f_shape
        batch = n_s * n_x
        n_tz = n_t * n_z

        # f part -> (B, L, TZ, n_rhs)
        b_f = rhs2d[: op.f_size].reshape(n_s, n_x, n_xi, n_tz, n_rhs)
        b_f = b_f.reshape(batch, n_xi, n_tz, n_rhs)

        y = _tier1_f_solve(op, self.factors, self.gamma, b_f, transpose)  # (B, L, TZ, n_rhs)

        if op.constraint_scheme == 0:
            x = y.reshape(op.f_size, n_rhs)
            return x[:, 0] if squeeze else x

        # constraintScheme=2: one bordered unknown per (species, x).
        # Forward:  [[A, b0 e0], [c0^T e0^T, 0]];  transpose swaps b0 <-> c0.
        r_c = rhs2d[op.f_size :].reshape(batch, n_rhs)
        z = self.z_t if transpose else self.z_fwd
        w_row = (
            self.b0 if transpose else self.c0
        )  # constraint row of the (transposed) system
        c_y = jnp.einsum("j,bjr->br", w_row, y[:, 0])  # w·y[l=0], (B, n_rhs)
        c_z = jnp.einsum("j,bj->b", w_row, z[:, 0])  # (B,)
        s = self.gamma[:, None] * r_c + (c_y - r_c) / c_z[:, None]
        shift = s - self.gamma[:, None] * r_c  # (B, n_rhs)
        f = y - shift[:, None, None, :] * z[:, :, :, None]

        x = jnp.concatenate(
            [f.reshape(op.f_size, n_rhs), s.reshape(op.extra_size, n_rhs)], axis=0
        )
        return x[:, 0] if squeeze else x

def build_tier1_solver(op: KineticOperator) -> Tier1Solver:
    """Assemble and factor the batched bordered block-tridiagonal solver.

    This is the full-band structured direct route.

    Uses the analytic (probing-free) :meth:`KineticOperator.to_block_tridiagonal`
    blocks — the replacement for the retired probing-based RHSMode=3 solver POC
    — and absorbs the ``constraintScheme=2`` border with the exact rank-one
    trick ``A~ = A + gamma B C`` documented in the module docstring.

    The assembly and the elimination run as one compiled program
    (:func:`_factor_tier1_compiled`), cached per operator structure, so a caller
    that does not ``jit`` does not dispatch the band assembly and the unrolled
    border presolves one primitive at a time. Under an enclosing trace the
    compiled program nests like any other function.

    Raises:
        NotImplementedError: when :func:`tier1_available` says no.
    """
    _require_solvax()
    ok, reason = tier1_available(op)
    if not ok:
        raise NotImplementedError(
            f"structured direct route unavailable: {reason}"
        )
    if not _uniform_nxi_for_x(op):
        raise NotImplementedError(
            "the full-band structured direct factorization requires uniform "
            "Nxi_for_x (the ramped bands carry singular zero rows on the "
            "truncated DOFs); ramped decks "
            "route through the truncated kernel (method='block_tridiagonal_truncated')"
        )
    return _factor_tier1_compiled(op)

def _tier1_block_fn(op: KineticOperator, k: Any, gamma: Any) -> Callable[[Any], tuple]:
    """Legendre row ``l`` of subsystem ``k`` (one (species, x) pair) of ``A + gamma B C``.

    The full-band route generates its rows on demand, one per scan step, in the
    elimination and in both substitution sweeps: only the Schur LUs are kept
    (a third of the bands), and no program unrolls a loop over L.
    """
    n_tz = op.n_theta * op.n_zeta
    border = jnp.outer(jnp.ones((n_tz,), dtype=jnp.float64), op._fs_average_factor().reshape(-1))

    def block_fn(l: Any) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        blk = op.legendre_blocks(l)
        lower, diag, upper = (a.reshape(-1, n_tz, n_tz)[k] for a in (blk.lower, blk.diag, blk.upper))
        if op.constraint_scheme == 2:
            diag = diag + jnp.where(l == 0, gamma, 0.0) * border
        return lower, diag, upper

    return block_fn

def _tier1_f_solve(op: KineticOperator, factors: tuple, gamma: Any, b_f: Any, transpose: bool) -> Any:
    """Batched f-block solve ``(B, L, TZ, n_rhs)`` from the stored Schur LUs."""
    def solve_one(k: Any, lu: Any, piv: Any, g: Any, r: Any) -> Any:
        state = GeneratedBlockTridiagFactors(lu, piv, _tier1_block_fn(op, k, g), op.n_xi, r.dtype)
        return block_thomas_solve(state, r, transpose=transpose)

    return jax.vmap(solve_one)(jnp.arange(b_f.shape[0]), *factors, gamma, b_f)

def _factor_tier1(op: KineticOperator) -> Tier1Solver:
    """The body of :func:`build_tier1_solver`, after its applicability checks."""
    n_s, n_x, n_xi, n_t, n_z = op.f_shape
    n_tz = n_t * n_z
    batch = n_s * n_x

    b0 = jnp.ones((n_tz,), dtype=jnp.float64)  # source shape on the l=0 rows
    c0 = op._fs_average_factor().reshape(-1)  # flux-surface-average constraint row

    if op.constraint_scheme == 2:
        # Conditioning-friendly rank-one scale per (species, x): mean |diag entry|
        # of the bands over the max magnitude of the rank-one update.
        def sums(l: Any) -> tuple[jnp.ndarray, jnp.ndarray]:
            d = op.legendre_blocks(l).diag.reshape(batch, n_tz, n_tz)
            return jnp.abs(jnp.diagonal(d, axis1=1, axis2=2)).sum(1), jnp.abs(d).sum((1, 2))

        s_diag, s_all = (v.sum(0) for v in jax.lax.map(sums, jnp.arange(n_xi)))
        scale = s_diag / (n_xi * n_tz)
        scale = jnp.where(scale > 0.0, scale, s_all / (n_xi * n_tz * n_tz))
        outer_max = jnp.max(jnp.abs(b0)) * jnp.max(jnp.abs(c0))
        gamma = scale / outer_max
    else:
        gamma = jnp.ones((batch,), dtype=jnp.float64)

    def factor_one(k: jnp.ndarray, g: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        state = block_thomas_factor_fn(_tier1_block_fn(op, k, g), n_xi, store_offdiagonals=False)
        return state.delta_lu, state.delta_piv

    factors = jax.vmap(factor_one)(jnp.arange(batch), gamma)

    e0 = jnp.zeros((batch, n_xi, n_tz, 1), dtype=jnp.float64)
    z_fwd = _tier1_f_solve(op, factors, gamma, e0.at[:, 0, :, 0].set(b0[None, :]), False)[..., 0]
    z_t = _tier1_f_solve(op, factors, gamma, e0.at[:, 0, :, 0].set(c0[None, :]), True)[..., 0]
    return Tier1Solver(
        op=op, factors=factors, z_fwd=z_fwd, z_t=z_t, gamma=gamma, b0=b0, c0=c0
    )

# One executable per operator structure (static layout plus array shapes); the
# physical coefficients are arguments, so an optimizer's or an Er scan's next
# operator reuses it.
_factor_tier1_compiled = jax.jit(_factor_tier1)

def _tier1_refined_solve(
    op: KineticOperator, t1_solver: Tier1Solver, b: jnp.ndarray, transpose: bool,
    fenced: bool = True,
) -> jnp.ndarray:
    """Structured direct substitution plus defect correction against ``op``.

    ``op`` is the operator being solved and ``t1_solver`` the elimination
    applied to it, which is a neighbour's when stored factors are reused.

    Fenced by optimization barriers, so XLA compiles the region the same way in
    every program that contains it: the plain solve and the forward pass of the
    differentiable one return the same bits for the same right-hand side.
    ``fenced=False`` leaves the region to a caller's own ``jit``: fences there
    constrain how XLA fuses it around a sharded batch, and the one-device and
    two-device programs of the same scan then round differently.
    """
    if not fenced:
        return iterative_refinement(
            lambda v: _operator_apply(op, v, transpose),
            b,
            lambda r: t1_solver.solve(r, transpose=transpose),
            iterations=_refinement_sweeps(t1_solver),
        )[0]
    op, t1_solver, b = jax.lax.optimization_barrier((op, t1_solver, b))
    x, _residual_norms = iterative_refinement(
        lambda v: _operator_apply(op, v, transpose),
        b,
        lambda r: t1_solver.solve(r, transpose=transpose),
        iterations=_refinement_sweeps(t1_solver),
    )
    return jax.lax.optimization_barrier(x)

_tier1_refined_solve_compiled = jax.jit(
    _tier1_refined_solve, static_argnames=("transpose", "fenced")
)

def _tier1_implicit(
    op: KineticOperator,
    op_const: KineticOperator,
    t1_solver: Tier1Solver,
    rhs2d: jnp.ndarray,
    token: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """A differentiable structured direct solve of every column, as one program.

    ``custom_linear_solve`` defines the derivative (the adjoint pattern: one
    transposed substitution on the same factors, and the operator's action
    differentiated at the converged state), and ``op`` is the operator the
    gradient flows to. ``op_const`` and ``t1_solver`` are the gradient-free
    operator and elimination the substitutions use. Compiled as a whole, an
    eager caller linearizes and transposes it from cached programs instead of
    re-tracing the implicit solve and evaluating its rules primitive by
    primitive on every call.

    Every substitution measures its true residual and hands each column's to
    the registered guard. Returns the solution and the forward residual norms.
    """

    def solve_measured(b: jnp.ndarray, *, transpose: bool) -> tuple[jnp.ndarray, jnp.ndarray]:
        x = _tier1_refined_solve(op_const, t1_solver, b, transpose)
        apply = _pinned_matvecs(op_const)[1 if transpose else 0]
        residual = _column_apply(apply, x) - b
        norms = [jnp.linalg.norm(v, axis=0) for v in (residual, b, x)]
        op_norm = _operator_norm_estimate(apply, b.shape[0])
        label = "adjoint (transposed)" if transpose else "forward"
        for j in range(b.shape[1]):
            jax.debug.callback(
                functools.partial(_dispatch_guard, label, j), token,
                *(n[j] for n in norms), op_norm, jnp.all(residual[:, j] == 0),
            )  # fmt: skip
        return x, norms[0]

    return _implicit_solve(
        lambda v: _operator_apply(op, v, False),
        lambda v: _operator_apply(op, v, True),
        rhs2d,
        functools.partial(solve_measured, transpose=False),
        functools.partial(solve_measured, transpose=True),
        has_aux=True,
    )

_tier1_implicit_compiled = jax.jit(_tier1_implicit)

def _residual_norms_of(
    op: KineticOperator, x2d: jnp.ndarray, rhs2d: jnp.ndarray, transpose: bool
) -> jnp.ndarray:
    return jnp.linalg.norm(_operator_apply(op, x2d, transpose) - rhs2d, axis=0)

_residual_norms_compiled = jax.jit(_residual_norms_of, static_argnames=("transpose",))

def _generated_full_factor_bytes(op: KineticOperator) -> float:
    m = op.n_theta * op.n_zeta
    return float(op.n_species * np.sum(np.asarray(op.n_xi_for_x)) * (8*m*m + 4*m))

def _build_generated_full_solver(
    op: KineticOperator, subsystem_batch: int | str
) -> Callable[..., jnp.ndarray]:
    """Retain SOLVAX Schur factors for this immutable physical operator.

    Factors stay grouped by active chain length; inactive pitch padding is
    never factored. The returned solve handles matrix RHSs and both borders.
    For the shifted chain K + gamma*b*c^T, the border correction enforces
    c^T*f = r_c; transposition swaps b and c and preserves gamma.
    """
    coef = _truncated_coefficients(op)
    ns, nx, nl, nt, nz = op.f_shape
    m = nt * nz
    bordered = op.constraint_scheme == 2
    width = _resolve_subsystem_batch(op, subsystem_batch, nl)
    groups = {}
    for ix, nb in enumerate(np.asarray(op.n_xi_for_x)):
        groups.setdefault(int(nb), []).append(ix)
    prepared = []
    for nb, indices in groups.items():
        idx = np.asarray(indices)
        count = ns * len(indices)
        inputs = (jnp.repeat(coef["stream"], len(indices), axis=0),
                  jnp.repeat(coef["mirror"], len(indices), axis=0),
                  coef["pas"][:, idx].reshape(count, nl),
                  jnp.tile(op.x[idx], ns), coef["gamma"][:, idx].reshape(count))
        def blocks(values):
            return _truncated_block_fn(coef, nl, *values, shift_border=bordered)
        def factor(values, nb=nb):
            state = block_thomas_factor_fn(blocks(values), nb, store_offdiagonals=False)
            return state.delta_lu, state.delta_piv
        batch_size = None if min(width, count) == 1 else min(width, count)
        lu, piv = jax.lax.map(factor, inputs, batch_size=batch_size)
        prepared.append((nb, idx, inputs, lu, piv, batch_size))

    def solve_prepared(rhs, transpose=False):
        nr = rhs.shape[1]
        rf = rhs[:op.f_size].reshape(ns, nx, nl, m, nr)
        rc = (rhs[op.f_size:].reshape(ns, nx, nr) if bordered
              else jnp.zeros((ns, nx, nr), dtype=rhs.dtype))
        fields, sources = jnp.zeros_like(rf), jnp.zeros_like(rc)
        b, c = (coef["c0"], coef["b0"]) if transpose else (coef["b0"], coef["c0"])
        for nb, idx, inputs, lu, piv, batch_size in prepared:
            count = ns * len(idx)
            def solve_one(values, nb=nb):
                coefficients, lu_one, piv_one, rhs_one, source = values
                block_fn = _truncated_block_fn(coef, nl, *coefficients, shift_border=bordered)
                state = GeneratedBlockTridiagFactors(lu_one, piv_one, block_fn, nb, rhs.dtype)
                if bordered:
                    border = jnp.zeros((nb, m, 1), dtype=rhs.dtype).at[0, :, 0].set(b)
                    solution = block_thomas_solve(state, jnp.concatenate((rhs_one, border), axis=2), transpose=transpose)
                    y, z = solution[:, :, :nr], solution[:, :, nr]
                    shift = (c @ y[0] - source) / (c @ z[0])
                    return y - z[:, :, None] * shift, coefficients[4] * source + shift
                return block_thomas_solve(state, rhs_one, transpose=transpose), source
            f, s = jax.lax.map(solve_one,
                (inputs, lu, piv, rf[:, idx, :nb].reshape(count, nb, m, nr),
                 rc[:, idx].reshape(count, nr)), batch_size=batch_size)
            fields = fields.at[:, idx, :nb].set(f.reshape(ns, len(idx), nb, m, nr))
            sources = sources.at[:, idx].set(s.reshape(ns, len(idx), nr))
        result = fields.reshape(op.f_size, nr)
        return jnp.concatenate((result, sources.reshape(op.extra_size, nr)), axis=0) if bordered else result
    return solve_prepared

def _solve_generated_full(
    op: KineticOperator, rhs2d: jnp.ndarray, *, tol: float, atol: float,
    subsystem_batch: int | str, adjoint_window: int | None,
    check_adjoint: bool, adjoint_residual_factor: float,
    memory_budget_gb: float | None = None,
) -> SolveResult:
    """Implicit full recovery using SOLVAX's existing generated sweeps."""
    start = time.perf_counter()
    diagnostics = AdjointDiagnostics(tol=float(tol), atol=float(atol),
                                     factor=float(adjoint_residual_factor), checked=bool(check_adjoint))
    apply, apply_t = _pinned_matvecs(op)
    mask = op.active_dof_mask()
    # Keep a single RHS vector-shaped in the implicit map. A matrix-shaped
    # map leaves a root bitcast shared with diagnostics, which crashes the
    # GPU multi-output fusion pass in JAX 0.10.2.
    single = rhs2d.shape[1] == 1
    budget, _ = _tier1_budget_bytes(memory_budget_gb)
    # Extend the one-RHS estimate with four buffers/extra column and its 2x slack.
    retained_peak = (tier1_truncated_peak_memory_bytes(op, op.n_xi, subsystem_batch)
                     + 2 * _generated_full_factor_bytes(op)
                     + 64 * op.total_size * (rhs2d.shape[1] - 1))
    prepared = (_build_generated_full_solver(op, subsystem_batch)
                if retained_peak <= budget else None)
    def project(b):
        return b if mask is None else jnp.where(mask if single else mask[:, None], b, 0.)
    def generated(b):
        result = _solve_tier1_truncated(
            op, b[:, None] if single else b, keep=op.n_xi, tol=tol, atol=atol,
            subsystem_batch=subsystem_batch, adjoint_window=adjoint_window,
        )
        # Complete the inverse with identity on inactive padding. The public
        # RHS is projected below, retaining the generated route's zero padding.
        return (result.x[:, 0] if single else result.x) + (b-project(b))
    def transpose(b):
        # Differentiate only the RHS map, not the physics coefficients. The
        # implicit solve differentiates those through the original operator.
        _, pullback = jax.vjp(generated, jnp.zeros_like(b))
        return pullback(b)[0]
    def measured(b, transposed=False):
        if prepared is None:
            x = transpose(b) if transposed else generated(b)
        else:
            action = apply_t if transposed else apply
            action = action if single else jax.vmap(action, in_axes=1, out_axes=1)
            def cached(r):
                value = prepared(r[:, None] if single else r, transpose=transposed)
                return (value[:, 0] if single else value) + (r-project(r))
            x, _ = iterative_refinement(action, b, cached, iterations=_TIER1_REFINEMENT_SWEEPS)
        physical_b, physical_x = project(b), project(x)
        for j in range(rhs2d.shape[1]):
            _guarded_solve(
                "adjoint (transposed)" if transposed else "forward", j,
                apply_t if transposed else apply,
                physical_b if single else physical_b[:, j],
                physical_x if single else physical_x[:, j],
                tol=tol, atol=atol, factor=adjoint_residual_factor,
                raise_on_failure=check_adjoint, diagnostics=diagnostics, solver_name="generated full",
            )
        return x
    # One matrix-RHS solve preserves shared factorization across columns.
    apply2d = apply if single else jax.vmap(apply, in_axes=1, out_axes=1)
    apply_t2d = apply_t if single else jax.vmap(apply_t, in_axes=1, out_axes=1)
    x = _implicit_solve(apply2d, apply_t2d, project(rhs2d[:, 0] if single else rhs2d),
                        measured, lambda b: measured(b, True))
    if single:
        x = x[:, None]
    residuals = _residual_norms(apply, x, rhs2d)
    jax.block_until_ready((x, residuals))
    return SolveResult(x=x, method="block_tridiagonal_truncated", iterations=None,
                       residual_norms=residuals, converged=_converged_flag(residuals, rhs2d, tol, atol),
                       recycle=None, timings={"build_and_solve": time.perf_counter()-start},
                       adjoint=diagnostics)

def _solve_tier1(
    op: KineticOperator,
    rhs2d: jnp.ndarray,
    *,
    tol: float,
    atol: float,
    differentiable: bool,
    check_adjoint: bool,
    adjoint_residual_factor: float,
    factors: Tier1Solver | None = None,
    transpose: bool = False,
    emit: Callable[[str], None] | None = print,
) -> SolveResult:
    diagnostics = (AdjointDiagnostics(tol=float(tol), atol=float(atol),
                                      factor=float(adjoint_residual_factor), checked=bool(check_adjoint))
                   if differentiable else None)
    t0 = time.perf_counter()
    reused = factors is not None
    if reused and factors.op.total_size != op.total_size:
        raise ValueError(
            f"the factors passed in are of size {factors.op.total_size} and "
            f"this operator is {op.total_size}; a factorization is only "
            "reusable for an operator of its own shape"
        )
    # The elimination is not differentiated: ``custom_linear_solve`` discards
    # the tangents of whatever its solve closes over, and handing the
    # factorization a gradient-free operator keeps an eager ``jax.grad`` from
    # linearizing an elimination whose derivative is then thrown away.
    op_const = _stop_gradient_tree(op) if differentiable else op
    t1_solver = factors if reused else build_structured_solver(op_const)
    # Force the async factorization to complete so the "build" timing reflects
    # real compute, not JAX dispatch latency; a no-op under jit/grad tracing.
    jax.block_until_ready(t1_solver)
    t1 = time.perf_counter()

    # The library's own elimination runs through compiled programs: the
    # substitution with its defect correction and residual measurement, and the
    # operator actions the implicit solve differentiates. Anything else in the
    # ``factors`` slot (a test double, say) keeps the uncompiled reference path
    # below, which computes the same thing one primitive at a time.
    compiled = isinstance(t1_solver, (Tier1Solver, CoupledSolver))

    # ``_transposed_apply`` lowers a fresh ``jax.linear_transpose`` of the
    # operator every time it is called, so the refinement, the guard and the
    # residual check must share one rather than build three.
    _applies: dict[bool, Callable[[jnp.ndarray], jnp.ndarray]] = {}

    def _apply_for(transposed: bool) -> Callable[[jnp.ndarray], jnp.ndarray]:
        if not _applies:
            _applies[False], _applies[True] = _pinned_matvecs(op)
        return _applies[transposed]

    def _solve_refined(b: jnp.ndarray, *, transpose: bool = False, rhs_index: int = 0) -> jnp.ndarray:
        """Factor solve plus iterative refinement (defect correction).

        The block-Thomas elimination is backward-stable but its rounding can
        leave the true relative residual a small multiple of eps above the
        strict production gate; refinement ``x += solve(b - A x)`` (one extra
        apply and substitution on the existing factors per sweep) takes it to
        O(1e-16).

        :func:`solvax.refine.iterative_refinement` owns the recurrence.  It is
        the same fixed point as the hand-rolled single pass this replaces --
        one sweep reproduces it exactly -- and it is where the float32-factor
        variant lives (``solvax.refine.as_low_precision``), which is the
        documented route to a GPU that runs FP64 at 1/32 rate.  Reusable
        numerical kernels belong upstream in ``solvax`` rather than here.
        """
        apply = _apply_for(transpose)
        apply2d = apply if b.ndim == 1 else jax.vmap(apply, in_axes=1, out_axes=1)
        x, _residual_norms = iterative_refinement(
            apply2d,
            b,
            lambda r: t1_solver.solve(r, transpose=transpose),
            iterations=_refinement_sweeps(t1_solver),
        )
        if differentiable:
            x = _guarded_solve(
                "adjoint (transposed)" if transpose else "forward", rhs_index,
                apply, b, x, tol=tol, atol=atol, factor=adjoint_residual_factor,
                raise_on_failure=check_adjoint, diagnostics=diagnostics, solver_name="structured",
            )
        return x

    if compiled and differentiable:
        def make_guard(label: str, rhs_index: int) -> Callable[..., None]:
            return _residual_guard(
                label, rhs_index, tol=tol, atol=atol, factor=adjoint_residual_factor,
                raise_on_failure=check_adjoint, diagnostics=diagnostics,
                solver_name="structured",
            )

        x2d, res = _tier1_implicit_compiled(
            op, op_const, t1_solver, rhs2d,
            jnp.asarray(_register_guard(make_guard), dtype=jnp.int32),
        )
    elif compiled:
        x2d = _tier1_refined_solve_compiled(
            op, t1_solver, rhs2d, transpose=transpose, fenced=not _is_traced(rhs2d)
        )
    elif differentiable:
        cols = [
            _implicit_solve(
                _apply_for(False),
                _apply_for(True),
                rhs2d[:, j],
                lambda b, j=j: _solve_refined(b, rhs_index=j),
                lambda b, j=j: _solve_refined(b, transpose=True, rhs_index=j),
            )
            for j in range(rhs2d.shape[1])
        ]
        x2d = jnp.stack(cols, axis=1)
    else:
        x2d = _solve_refined(rhs2d, transpose=transpose)
    x2d = jax.block_until_ready(x2d)  # real solve compute, not just dispatch
    t2 = time.perf_counter()
    # The residual is a diagnostic, not part of the differentiated answer, so
    # it is measured on values alone. An eager ``jax.grad`` holds those
    # concretely, which keeps the convergence flag below a real test there
    # too. The compiled differentiable program measured it already.
    rhs2d_value = jax.lax.stop_gradient(rhs2d)
    if not compiled:
        res = _residual_norms(_apply_for(transpose), x2d, rhs2d)
    elif not differentiable:
        res = _residual_norms_compiled(
            _stop_gradient_tree(op), jax.lax.stop_gradient(x2d), rhs2d_value,
            transpose=transpose,
        )
    if reused and not _converged_flag(res, rhs2d_value, tol, atol):
        # The stored elimination belongs to a neighbouring operator and the
        # refinement above did not make up the difference. The staleness test is
        # this residual; the recovery is one factorization, bounded, not an
        # unbounded wait on a Krylov method that has already been measured to
        # spend 6,000 iterations failing at a distant Er.
        _say(
            emit,
            "[dkx.solve] the stored elimination did not solve this operator "
            f"(residuals={np.asarray(res)}); factorizing it.",
        )
        return _solve_tier1(
            op, rhs2d, tol=tol, atol=atol, differentiable=differentiable,
            check_adjoint=check_adjoint,
            adjoint_residual_factor=adjoint_residual_factor,
            factors=None, transpose=transpose, emit=emit,
        )
    return SolveResult(
        x=x2d,
        method="block_tridiagonal",
        iterations=None,
        residual_norms=res,
        converged=_converged_flag(res, rhs2d_value, tol, atol),
        recycle=None,
        adjoint=diagnostics,
        timings={"build": 0.0 if reused else t1 - t0, "solve": t2 - t1},
        factors=t1_solver,
    )

# =============================================================================
# Structured direct (truncated) — block Thomas over the lowest K Legendre modes
# =============================================================================


@functools.partial(jax.jit, static_argnames=("keep", "adjoint_window", "width"))
def _truncated_sweeps(op: KineticOperator, rhs2d: jnp.ndarray, keep: int, adjoint_window: int | None, width: int):
    """The block-Thomas sweeps of :func:`_solve_tier1_truncated`, compiled once per layout.

    Eagerly, every call retraced ``lax.scan`` over fresh closures and compiled one
    executable per ``Nxi_for_x`` group (8 per solve on the tokamak decks), and every
    Er field of a scan did the same. As a module-level jit the operator is an argument,
    so a repeated solve or another field reuses the executable.
    """
    n_s, n_x, n_xi, n_t, n_z = op.f_shape
    n_tz = n_t * n_z
    batch = n_s * n_x
    n_rhs = rhs2d.shape[1]
    cs = op.constraint_scheme

    coef = _truncated_coefficients(op)

    # Per-subsystem inputs, flattened to batch index b = s * n_x + x (matching
    # the (S, X) reshape used by KineticOperator.apply / build_tier1_solver).
    stream_b = jnp.repeat(coef["stream"], n_x, axis=0)  # (B, TZ, TZ)
    mirror_b = jnp.repeat(coef["mirror"], n_x, axis=0)  # (B, TZ)
    pas_b = coef["pas"].reshape(batch, n_xi)  # (B, L)
    x_b = jnp.tile(op.x, n_s)  # (B,)
    gamma_b = coef["gamma"].reshape(batch)  # (B,)
    b0, c0 = coef["b0"], coef["c0"]

    rhs_f = rhs2d[: op.f_size].reshape(n_s, n_x, n_xi, n_tz, n_rhs)
    rhs_low_b = rhs_f[:, :, :keep].reshape(batch, keep, n_tz, n_rhs)  # (B, keep, TZ, R)
    if cs == 2:
        r_c_b = rhs2d[op.f_size :].reshape(batch, n_rhs)  # (B, R)
    else:
        r_c_b = jnp.zeros((batch, n_rhs), dtype=jnp.float64)

    def solve_one(inputs, n_blocks: int):
        stream, mirror, pas_row, x_val, gamma, rhs_low, r_c = inputs
        local_keep = min(keep, n_blocks)
        rhs_low = rhs_low[:local_keep]

        def truncated_solve(rhs_cols: jnp.ndarray) -> jnp.ndarray:
            if adjoint_window is not None:
                # Structure-preserving bounded adjoint: the
                # differentiable coefficients travel as an explicit params
                # pytree, and reverse mode runs at O((keep + w) * m^2) per
                # subsystem instead of taping the full n_blocks sweep.
                params = _truncated_params(coef, stream, mirror, pas_row, x_val, gamma)
                block_fn_pk = functools.partial(
                    _truncated_blocks, n_xi=n_xi, shift_border=(cs == 2)
                )
                return block_thomas_truncated_fn(
                    block_fn_pk,
                    n_blocks,
                    rhs_cols,
                    local_keep,
                    params=params,
                    adjoint_window=adjoint_window,
                )
            block_fn = _truncated_block_fn(
                coef,
                n_xi,
                stream,
                mirror,
                pas_row,
                x_val,
                gamma,
                shift_border=(cs == 2),
            )
            return block_thomas_truncated_fn(block_fn, n_blocks, rhs_cols, local_keep)

        if cs == 2:
            z_col = jnp.zeros((local_keep, n_tz, 1), dtype=jnp.float64).at[0, :, 0].set(b0)
            rhs_stack = jnp.concatenate([rhs_low, z_col], axis=2)  # (keep, TZ, R+1)
            sol = truncated_solve(rhs_stack)
            y = sol[:, :, :n_rhs]  # (keep, TZ, R)
            z = sol[:, :, n_rhs]  # (keep, TZ)
            c_y0 = c0 @ y[0]  # (R,)
            c_z0 = c0 @ z[0]  # scalar
            shift = (c_y0 - r_c) / c_z0  # (R,)
            s = gamma * r_c + shift  # (R,)
            f_low = y - shift[None, None, :] * z[:, :, None]  # (keep, TZ, R)
        else:
            f_low = truncated_solve(rhs_low)
            s = jnp.zeros((n_rhs,), dtype=jnp.float64)
        return jnp.pad(f_low, ((0, keep - local_keep), (0, 0), (0, 0))), s

    # Concurrency across the B independent subsystems: lax.map with
    # batch_size=w eliminates w subsystems per vmapped chunk (the peak holds w
    # concurrent O(keep * m^2) sweeps); width 1 (batch_size=None) is the fully
    # serial scan-based sweep with a single subsystem's working set.
    def _map_subsystems(fn, inputs, n_sub: int):
        w = min(width, n_sub)
        return jax.lax.map(fn, inputs, batch_size=None if w == 1 else w)

    if _uniform_nxi_for_x(op):
        f_low_b, s_b = _map_subsystems(
            lambda t: solve_one(t, n_xi),
            (stream_b, mirror_b, pas_b, x_b, gamma_b, rhs_low_b, r_c_b),
            batch,
        )
    else:
        # Ramped Nxi_for_x: each (species, x) subsystem is closed, so it is
        # eliminated with its own static n_blocks = Nxi_for_x[ix] (the packed
        # Fortran discretization).  Subsystems are grouped by equal n_blocks —
        # all species at the speed nodes sharing one Nxi_for_x value — and
        # batched within each group, preserving the exact per-subsystem block
        # count at any width.
        rhs_low_sx = rhs_low_b.reshape(n_s, n_x, keep, n_tz, n_rhs)
        r_c_sx = r_c_b.reshape(n_s, n_x, n_rhs)
        groups: dict[int, list[int]] = {}
        for ix, nb in enumerate(int(v) for v in np.asarray(op.n_xi_for_x)):
            groups.setdefault(nb, []).append(ix)
        f_low_sx = jnp.zeros((n_s, n_x, keep, n_tz, n_rhs), dtype=jnp.float64)
        s_sx = jnp.zeros((n_s, n_x, n_rhs), dtype=jnp.float64)
        for nb, ixs in groups.items():
            g = len(ixs)
            idx = np.asarray(ixs)
            if g == 1:
                # Single-speed-node group (all ramp values distinct — the
                # production shape): slice views, no gather/repeat copies.
                ix = ixs[0]
                inputs = (
                    coef["stream"], coef["mirror"], coef["pas"][:, ix],
                    jnp.broadcast_to(op.x[ix], (n_s,)), coef["gamma"][:, ix],
                    rhs_low_sx[:, ix], r_c_sx[:, ix],
                )  # fmt: skip
            else:
                inputs = (
                    jnp.repeat(coef["stream"], g, axis=0),
                    jnp.repeat(coef["mirror"], g, axis=0),
                    coef["pas"][:, idx].reshape(n_s * g, n_xi),
                    jnp.tile(op.x[idx], n_s),
                    coef["gamma"][:, idx].reshape(n_s * g),
                    rhs_low_sx[:, idx].reshape(n_s * g, keep, n_tz, n_rhs),
                    r_c_sx[:, idx].reshape(n_s * g, n_rhs),
                )  # fmt: skip
            f_g, s_g = _map_subsystems(
                lambda t, nb=nb: solve_one(t, nb), inputs, n_s * g
            )
            f_low_sx = f_low_sx.at[:, idx].set(f_g.reshape(n_s, g, keep, n_tz, n_rhs))
            s_sx = s_sx.at[:, idx].set(s_g.reshape(n_s, g, n_rhs))
        f_low_b = f_low_sx.reshape(batch, keep, n_tz, n_rhs)
        s_b = s_sx.reshape(batch, n_rhs)
    return f_low_b, s_b, (coef, stream_b, mirror_b, pas_b, x_b, rhs_low_b, r_c_b)


def _solve_tier1_truncated(
    op: KineticOperator,
    rhs2d: jnp.ndarray,
    *,
    keep: int,
    tol: float,
    atol: float,
    subsystem_batch: int | str = "auto",
    adjoint_window: int | None = None,
) -> SolveResult:
    """Memory-lean structured direct solve: the lowest ``keep`` Legendre blocks.

    Assembles each ``(m, m)`` Legendre block on the fly inside
    ``solvax.direct.block_thomas_truncated_fn`` (peak memory ``O(keep * m^2)``
    per subsystem, independent of ``n_xi``), so the ~39 GB full-band storage is
    never allocated.  The ``constraintScheme=2`` border is absorbed with the
    same exact rank-one trick as :func:`build_tier1_solver`.  The lowest
    ``keep`` blocks (and the source unknowns) are exact; blocks ``l >= keep``
    are zero-padded — valid because the drive and all requested output moments
    live on ``l < keep`` (see :func:`_rhs_confined_to_lowest_blocks`). With full
    recovery, ``keep`` grows with pitch resolution: per-chain memory then grows
    too, although global band arrays are still never materialized.

    Non-uniform ``Nxi_for_x`` (the production speed-dependent Legendre ramp)
    is exact too: each (species, x) subsystem is closed, so it is eliminated
    with its own ``n_blocks = Nxi_for_x[ix]`` — precisely the packed Fortran
    discretization (``indices.F90``), whose truncated DOFs the zero-padded
    tail already covers. Full recovery retains ``min(keep, n_blocks)`` per
    chain and pads only inactive DOFs; partial recovery still requires
    ``keep <= min Nxi_for_x``.

    ``subsystem_batch`` sets how many of the ``B = n_species * n_x``
    independent subsystems are eliminated concurrently (the
    ``jax.lax.map(..., batch_size=w)`` vmapped-chunk axis): width 1 is the
    fully serial minimum-memory sweep, width ``B`` eliminates every subsystem
    at once, and ``"auto"`` resolves backend-aware — width 1 on CPU, the
    widest memory-budget-fitting width on accelerators
    (:func:`_resolve_subsystem_batch`).  On the ramped path subsystems are
    grouped by equal ``n_blocks`` (all species at the speed nodes sharing one
    ``Nxi_for_x`` value) and batched within each group, so every subsystem
    keeps exactly its own static block count at any width — identical
    per-subsystem arithmetic to the serial sweep.

    This raw kernel is a pure-JAX composition of
    ``block_thomas_truncated_fn`` sweeps, so ``jax.grad`` differentiates
    straight through it. Public full recovery with ``differentiable=True``
    instead enters :func:`_solve_generated_full`; the following reduced-system
    discussion applies to partial recovery.  It is *not* wrapped in the full-operator
    implicit-function-theorem adjoint used by the full structured direct and
    recycled Krylov paths: this solve inverts the *reduced* Schur-complemented
    operator on the lowest ``keep`` blocks, not the full band, so a
    full-operator ``A^T`` adjoint would be inconsistent and silently corrupt
    gradients.  Two consistent
    reverse-mode paths exist instead.  The default tapes the generated sweeps
    (exact, but the tape grows with ``n_xi``: ``O(n_xi * m^2)`` per
    subsystem).  With ``adjoint_window=w`` the
    solve uses solvax's structure-preserving custom VJP for generated blocks:
    the right-hand-side gradient is an exactly *generated* truncated solve of
    the transposed operator, and the coefficient gradients are pulled back
    through the block assembly's own derivative on the leading ``keep + w``
    Legendre blocks — reverse mode then runs at ``O((keep + w) * m^2)`` per
    subsystem, independent of ``n_xi``, matching the forward sweep.  The
    window trades nothing on the right-hand-side gradient and has
    ``O(rho^{2w})`` coefficient-gradient error for the block-dominant
    collisional operators this kernel targets; ``w >= n_xi`` reproduces the
    taped gradient exactly (solvax pins full-window bitwise equality).
    """
    _require_solvax()
    n_s, n_x, n_xi, n_t, n_z = op.f_shape
    n_tz = n_t * n_z
    n_rhs = rhs2d.shape[1]

    t0 = time.perf_counter()
    # Concurrency across the B independent subsystems: width w eliminates w subsystems
    # per vmapped chunk (the peak holds w concurrent O(keep * m^2) sweeps); width 1 is
    # the fully serial scan-based sweep with a single subsystem's working set.
    width = _resolve_subsystem_batch(op, subsystem_batch, keep)
    f_low_b, s_b, (coef, stream_b, mirror_b, pas_b, x_b, rhs_low_b, r_c_b) = _truncated_sweeps(
        op, rhs2d, keep, adjoint_window, width
    )
    # Force the async sweep to complete so the timing is real compute, not JAX
    # dispatch latency (a no-op under jit/grad tracing).
    f_low_b, s_b = jax.block_until_ready((f_low_b, s_b))
    t1 = time.perf_counter()

    f_full = jnp.zeros((n_s, n_x, n_xi, n_tz, n_rhs), dtype=jnp.float64)
    f_full = f_full.at[:, :, :keep].set(f_low_b.reshape(n_s, n_x, keep, n_tz, n_rhs))
    parts = [f_full.reshape(op.f_size, n_rhs)]
    if op.extra_size:
        parts.append(s_b.reshape(op.extra_size, n_rhs))
    x2d = jnp.concatenate(parts, axis=0)

    if keep == n_xi:
        res = _residual_norms(op.apply, x2d, rhs2d)
    else:
        res = _truncated_partial_residual(
            op, coef, stream_b, mirror_b, pas_b, x_b, f_low_b, s_b, rhs_low_b, r_c_b, keep
        )
    x2d, res = jax.block_until_ready(
        (x2d, res)
    )  # real residual/assembly time, not dispatch
    t2 = time.perf_counter()
    return SolveResult(
        x=x2d,
        method="block_tridiagonal_truncated",
        iterations=None,
        residual_norms=res,
        converged=_converged_flag(res, rhs2d, tol, atol),
        recycle=None,
        timings={"build": t1 - t0, "solve": t2 - t1},
    )

def tier1_truncated_tail_blocks(
    op: KineticOperator,
    rhs: jnp.ndarray,
    solution: jnp.ndarray,
    *,
    keep_lowest: int = _TIER1_KEEP_LOWEST_DEFAULT,
    keep_highest: int = 2,
) -> jnp.ndarray:
    """Recover exact final Legendre blocks after a truncated structured solve.

    The production truncated solve retains exact low transport modes and the
    ``constraintScheme=2`` source unknowns.  This opt-in diagnostic performs a
    second, forward Schur sweep through the same generated blocks and retains
    only the final ``keep_highest`` modes.  It never materializes full bands or
    a full modal state; dense workspace is independent of ``Nxi``.

    Returns ``(species, speed, keep_highest, theta*zeta, rhs)``.  The tail is
    ordered by increasing Legendre index, and each speed subsystem uses its own
    active ``Nxi_for_x`` count.
    """
    _require_solvax()
    if int(keep_highest) < 1:
        raise ValueError("keep_highest must be at least 1")
    ok, reason = _truncation_supported(op, int(keep_lowest))
    if not ok:
        raise NotImplementedError(
            f"structured direct selected-tail reconstruction unavailable: {reason}"
        )
    if int(np.min(np.asarray(op.n_xi_for_x))) < int(keep_highest):
        raise ValueError("keep_highest exceeds the minimum active Nxi_for_x")

    rhs2d, _ = _as_columns(rhs)
    solution2d, _ = _as_columns(solution)
    if rhs2d.shape != solution2d.shape or rhs2d.shape[0] != op.total_size:
        raise ValueError(
            "rhs and solution must have matching state-vector shapes, got "
            f"{rhs2d.shape} and {solution2d.shape}"
        )

    n_s, n_x, n_xi, n_t, n_z = op.f_shape
    n_tz = n_t * n_z
    n_rhs = rhs2d.shape[1]
    batch = n_s * n_x
    coef = _truncated_coefficients(op)
    rhs_f = rhs2d[: op.f_size].reshape(n_s, n_x, n_xi, n_tz, n_rhs)
    rhs_low_b = rhs_f[:, :, :keep_lowest].reshape(
        batch, keep_lowest, n_tz, n_rhs
    )
    solution_f = solution2d[: op.f_size].reshape(
        n_s, n_x, n_xi, n_tz, n_rhs
    )
    solution_low_b = solution_f[:, :, :keep_lowest].reshape(
        batch, keep_lowest, n_tz, n_rhs
    )
    if op.constraint_scheme == 2:
        r_c_b = rhs2d[op.f_size :].reshape(batch, n_rhs)
        source_b = solution2d[op.f_size :].reshape(batch, n_rhs)
    else:
        r_c_b = jnp.zeros((batch, n_rhs), dtype=jnp.float64)
        source_b = jnp.zeros((batch, n_rhs), dtype=jnp.float64)

    def solve_one(inputs, n_blocks: int):
        stream, mirror, pas_row, x_val, gamma, rhs_low, solution_low, r_c, source = inputs
        block_fn = _truncated_block_fn(
            coef,
            n_xi,
            stream,
            mirror,
            pas_row,
            x_val,
            gamma,
            shift_border=(op.constraint_scheme == 2),
        )
        if op.constraint_scheme == 2:
            shift = source - gamma * r_c
            rhs_low = rhs_low.at[0].add(-coef["b0"][:, None] * shift[None, :])
        return block_thomas_selected_tail_fn(
            block_fn,
            n_blocks,
            rhs_low,
            int(keep_highest),
            solution_low=solution_low,
        )

    groups: dict[int, list[int]] = {}
    for ix, count in enumerate(int(v) for v in np.asarray(op.n_xi_for_x)):
        groups.setdefault(count, []).append(ix)
    tails = jnp.zeros(
        (n_s, n_x, keep_highest, n_tz, n_rhs), dtype=jnp.float64
    )
    rhs_low_sx = rhs_low_b.reshape(n_s, n_x, keep_lowest, n_tz, n_rhs)
    solution_low_sx = solution_low_b.reshape(
        n_s, n_x, keep_lowest, n_tz, n_rhs
    )
    r_c_sx = r_c_b.reshape(n_s, n_x, n_rhs)
    source_sx = source_b.reshape(n_s, n_x, n_rhs)
    for n_blocks, ixs in groups.items():
        count = len(ixs)
        idx = np.asarray(ixs)
        inputs = (
            jnp.repeat(coef["stream"], count, axis=0),
            jnp.repeat(coef["mirror"], count, axis=0),
            coef["pas"][:, idx].reshape(n_s * count, n_xi),
            jnp.tile(op.x[idx], n_s),
            coef["gamma"][:, idx].reshape(n_s * count),
            rhs_low_sx[:, idx].reshape(n_s * count, keep_lowest, n_tz, n_rhs),
            solution_low_sx[:, idx].reshape(
                n_s * count, keep_lowest, n_tz, n_rhs
            ),
            r_c_sx[:, idx].reshape(n_s * count, n_rhs),
            source_sx[:, idx].reshape(n_s * count, n_rhs),
        )
        recovered = jax.lax.map(
            lambda values, n_blocks=n_blocks: solve_one(values, n_blocks), inputs
        )
        tails = tails.at[:, idx].set(
            recovered.reshape(n_s, count, keep_highest, n_tz, n_rhs)
        )
    return tails

def _truncated_partial_residual(
    op: KineticOperator,
    coef: dict[str, jnp.ndarray],
    stream_b: jnp.ndarray,
    mirror_b: jnp.ndarray,
    pas_b: jnp.ndarray,
    x_b: jnp.ndarray,
    f_low_b: jnp.ndarray,
    s_b: jnp.ndarray,
    rhs_low_b: jnp.ndarray,
    r_c_b: jnp.ndarray,
    keep: int,
) -> jnp.ndarray:
    """Residual over the rows fully determined by the computed lowest-K blocks.

    Legendre row ``l`` couples to columns ``l-1, l, l+1``, so rows
    ``l = 0 .. keep-2`` (plus the ``constraintScheme=2`` FSA rows) are entirely
    fixed by the ``keep`` computed blocks and must vanish to machine precision;
    row ``keep-1`` couples to the (deliberately unsolved) block ``keep`` and is
    excluded.  This mirrors ``TruncatedTier1.partial_residual`` and is the
    honest convergence signal for the truncated solve.  Returns per-column
    norms of shape ``(n_rhs,)``.
    """
    cs = op.constraint_scheme
    n_rhs = rhs_low_b.shape[-1]
    b0, c0 = coef["b0"], coef["c0"]

    def per_subsystem(inputs):
        stream, mirror, pas_row, x_val, f_low, s, rhs_low, r_c = inputs
        raw = _truncated_block_fn(
            coef, op.n_xi, stream, mirror, pas_row, x_val, 0.0, shift_border=False
        )
        acc = jnp.zeros((n_rhs,), dtype=jnp.float64)
        for ell in range(keep - 1):
            lo, di, up = raw(jnp.asarray(ell, dtype=jnp.int32))
            r = jnp.einsum("ij,jr->ir", di, f_low[ell]) - rhs_low[ell]
            if ell > 0:
                r = r + jnp.einsum("ij,jr->ir", lo, f_low[ell - 1])
            r = r + jnp.einsum("ij,jr->ir", up, f_low[ell + 1])
            if ell == 0 and cs == 2:
                r = r + b0[:, None] * s[None, :]
            acc = acc + jnp.sum(r * r, axis=0)
        if cs == 2:
            rc = (c0 @ f_low[0]) - r_c  # (R,)
            acc = acc + rc * rc
        return acc

    sq = jax.lax.map(
        per_subsystem, (stream_b, mirror_b, pas_b, x_b, f_low_b, s_b, rhs_low_b, r_c_b)
    )  # (B, R)
    return jnp.sqrt(jnp.sum(sq, axis=0))
