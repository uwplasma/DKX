"""Nonlinear Phi1 / quasineutrality solve on the canonical stack.

``includePhi1`` makes the drift-kinetic equation nonlinear: the electrostatic
potential perturbation ``Phi1(theta, zeta)`` is an additional unknown fixed by
quasineutrality (``Ntheta*Nzeta`` charge-neutrality rows plus one ``<Phi1>=0``
constraint row -- the "lambda" row), and it couples back into the kinetic
equation (``includePhi1InKineticEquation``).  SFINCS Fortran solves this with a
Newton (PETSc ``SNES``) loop wrapping the linear kernel; this module is the
``phi1.py`` slice of ``plan_final.md`` item 5.

Public entry points:

- :func:`solve_phi1` -- the Fortran-parity Newton solve.  Each Newton step
  linearizes the residual (:meth:`sfincs_jax.drift_kinetic.KineticOperator.residual_phi1`)
  and solves ``J dx = -r`` with :func:`sfincs_jax.solve.solve` (tier-2 recycled
  Krylov on the matrix-free Jacobian-vector product), warm-started with the
  GCROT recycle pair across Newton iterations.  Returns a :class:`Phi1Result`
  with the solved state, ``Phi1Hat``, per-iteration diagnostics, and the
  operator carrying the converged linearization.
- :func:`phi1_state` -- the *differentiable* Phi1 state: the nonlinear fixed
  point ``F(x) = 0`` is wrapped with :func:`solvax.implicit.root_solve` so
  ``jax.grad`` of any downstream moment flows through the solve via the implicit
  function theorem ``dx/dp = -(dF/dx)^{-1} dF/dp`` (Jacobians from autodiff of
  the residual, not finite differences).

Fortran correspondence: ``solver.F90`` (the ``SNES``/Newton loop and its KSP
linear solves), ``populateMatrix.F90`` (the quasineutrality block and the Phi1
kinetic coupling), ``evaluateResidual.F90`` (the nonlinear residual and the
Boltzmann quasineutrality drive).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from sfincs_jax.drift_kinetic import KineticOperator, kinetic_operator_from_namelist  # noqa: E402
from sfincs_jax.inputs import (  # noqa: E402
    RawNamelist,
    SfincsInput,
    load_sfincs_input,
    sfincs_input_from_raw,
)
from sfincs_jax.solve import SolveResult, solve  # noqa: E402

__all__ = [
    "NewtonIteration",
    "Phi1Result",
    "operator_from_input",
    "phi1_state",
    "solve_phi1",
    "solve_phi1_history",
]


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NewtonIteration:
    """One accepted Newton step of a Phi1 solve."""

    index: int
    residual_norm: float
    inner_iterations: int | None
    inner_converged: bool
    step_scale: float


@dataclass(frozen=True)
class Phi1Result:
    """Outcome of :func:`solve_phi1`.

    Attributes:
        x: solved state vector, shape ``(total_size,)`` with the
            ``[f | Phi1(theta,zeta) | lambda | sources]`` layout.
        phi1_hat: the solved ``Phi1Hat(theta, zeta)`` field, shape ``(T, Z)``.
        operator: the canonical operator with ``phi1_lin_state`` set to ``x``
            (so ``operator.apply`` is the converged Jacobian and moment/writer
            helpers can read the solved Phi1).
        residual_norm: the final nonlinear residual norm ``||F(x)||``.
        converged: whether the Newton residual dropped below ``tol``.
        n_newton: number of Newton iterations taken.
        iterations: the per-step Newton diagnostics.
        inner_iterations_total: total inner Krylov iterations across all Newton
            steps (``None`` if any step ran a non-counting solve tier).
        timings: wall-clock seconds (``solve``).
    """

    x: jnp.ndarray
    phi1_hat: jnp.ndarray
    operator: KineticOperator
    residual_norm: float
    converged: bool
    n_newton: int
    iterations: tuple[NewtonIteration, ...] = ()
    inner_iterations_total: int | None = None
    timings: dict[str, float] | None = None

    @property
    def residual_norms(self) -> tuple[float, ...]:
        return tuple(it.residual_norm for it in self.iterations)


# ---------------------------------------------------------------------------
# Operator construction
# ---------------------------------------------------------------------------


def _as_input(inp: SfincsInput | RawNamelist | str | Path) -> SfincsInput:
    if isinstance(inp, SfincsInput):
        return inp
    if isinstance(inp, RawNamelist):
        return sfincs_input_from_raw(inp)
    return load_sfincs_input(Path(inp))


def operator_from_input(
    inp: SfincsInput | RawNamelist | str | Path | KineticOperator,
) -> KineticOperator:
    """Build (or accept) the canonical Phi1 operator for a deck.

    Requires ``includePhi1 = .true.`` with a *self-consistent* (solved) Phi1:
    handles ``includePhi1InKineticEquation`` and ``includePhi1InCollisionOperator``
    with ``quasineutralityOption`` 1/2.  ``readExternalPhi1`` is a fixed-field
    LINEAR solve (no Newton) routed through :func:`sfincs_jax.run.run_profile`,
    not this Newton entry point; ``quasineutralityOption`` other than 1/2 raises
    from :func:`sfincs_jax.drift_kinetic.kinetic_operator_from_namelist`.
    """
    if isinstance(inp, KineticOperator):
        op = inp
    else:
        typed = _as_input(inp)
        if typed.general.rhs_mode != 1:
            raise NotImplementedError(
                "Phi1 solves are RHSMode=1 (single-RHS profile drive); "
                f"got RHSMode={typed.general.rhs_mode}."
            )
        raw = typed.raw
        if raw is None:
            raise ValueError("solve_phi1 requires an input parsed from a namelist file.")
        op = kinetic_operator_from_namelist(raw)
    if not op.include_phi1:
        raise ValueError(
            "solve_phi1/phi1_state require includePhi1=.true.; use run_profile "
            "or solve for the linear (no-Phi1) system."
        )
    return op


def _inner_restart(op: KineticOperator, restart: int | None, use_preconditioner: bool) -> int:
    """Krylov restart for the inner Newton solve.

    Phi1 breaks the block-tridiagonal-in-L structure (the quasineutrality rows
    couple all angles at L=0), so the inner solve is tier-2 GCROT on the
    matrix-free Jacobian.

    With the Phi1-aware coarse preconditioner
    (:func:`sfincs_jax.solve.build_coarse_preconditioner`, which Schur-eliminates
    the quasineutrality border) the preconditioned operator clusters tightly
    around the identity, so a *modest* FGMRES cycle converges in a handful of
    iterations at any resolution -- a full-restart basis would waste memory and
    there is no size cap.  Without the preconditioner the inner solve falls back
    to a full restart (``>= total_size``, an exact solve per cycle), which only
    scales to the small Phi1 cases and is capped so a large deck fails loudly.
    """
    if restart is not None:
        return int(restart)
    n = int(op.total_size)
    if use_preconditioner:
        return min(n + 1, 60)
    cap = 6000
    if n > cap:
        raise NotImplementedError(
            f"solve_phi1's unpreconditioned inner Krylov solve targets small Phi1 "
            f"cases (total_size<= {cap}); this deck has total_size={n}. Enable the "
            "Phi1-aware coarse preconditioner (use_preconditioner=True) for "
            "production-scale decks."
        )
    return n + 1


# ---------------------------------------------------------------------------
# Fortran-parity Newton solve (uses solve.solve as the inner linear solve)
# ---------------------------------------------------------------------------


def solve_phi1(
    inp: SfincsInput | RawNamelist | str | Path | KineticOperator,
    *,
    x0: Any | None = None,
    tol: float = 1e-9,
    max_newton: int = 20,
    gmres_tol: float = 1e-11,
    gmres_restart: int | None = None,
    gmres_recycle_dim: int = 16,
    gmres_max_restarts: int = 40,
    warm_start: bool = True,
    line_search: bool = True,
    solve_method: str = "gmres",
    use_preconditioner: bool = True,
    emit: Callable[[str], None] | None = None,
) -> Phi1Result:
    """Solve the nonlinear Phi1 / quasineutrality system with Newton-Krylov.

    Args:
        inp: a deck (``SfincsInput`` / ``RawNamelist`` / path) or a prebuilt
            canonical :class:`~sfincs_jax.drift_kinetic.KineticOperator` with
            ``include_phi1=True``.
        x0: optional warm-start state, shape ``(total_size,)`` (defaults to zero
            -- the linear solution's neighborhood).
        tol: nonlinear residual tolerance ``||F(x)|| < tol``.
        max_newton: Newton iteration cap.
        gmres_tol: inner linear-solve relative tolerance.
        gmres_restart: inner GCROT restart (``None`` -> full restart, exact for
            the small Phi1 cases; see :func:`_inner_restart`).
        gmres_recycle_dim: GCROT recycle directions ``k`` carried across Newton
            steps when ``warm_start``.
        gmres_max_restarts: inner outer-cycle cap.
        warm_start: thread the GCROT recycle pair across Newton iterations.
        line_search: on a Newton step that does not reduce the residual, back
            off the step (halving) until it does -- robustness for the strongly
            nonlinear early iterations.
        solve_method: :func:`sfincs_jax.solve.solve` method for the inner solve
            (``"gmres"``).
        use_preconditioner: use the Phi1-aware coarse bordered-Schur
            preconditioner (:func:`sfincs_jax.solve.build_coarse_preconditioner`)
            for the inner linear solve.  On by default -- it Schur-eliminates the
            quasineutrality border so the inner Krylov solve converges in far
            fewer iterations at production resolution.  Set ``False`` for the
            unpreconditioned full-restart fallback (small decks only).
        emit: optional per-line stdout sink (Fortran-style Newton trace).

    Returns:
        A :class:`Phi1Result`.
    """
    result, _ = _newton_solve_phi1(
        operator_from_input(inp),
        x0=x0,
        tol=tol,
        max_newton=max_newton,
        gmres_tol=gmres_tol,
        gmres_restart=gmres_restart,
        gmres_recycle_dim=gmres_recycle_dim,
        gmres_max_restarts=gmres_max_restarts,
        warm_start=warm_start,
        line_search=line_search,
        solve_method=solve_method,
        use_preconditioner=use_preconditioner,
        emit=emit,
        record_history=False,
    )
    return result


def solve_phi1_history(
    inp: SfincsInput | RawNamelist | str | Path | KineticOperator,
    *,
    x0: Any | None = None,
    tol: float = 1e-9,
    max_newton: int = 20,
    gmres_tol: float = 1e-11,
    gmres_restart: int | None = None,
    gmres_recycle_dim: int = 16,
    gmres_max_restarts: int = 40,
    warm_start: bool = True,
    line_search: bool = True,
    solve_method: str = "gmres",
    use_preconditioner: bool = True,
    emit: Callable[[str], None] | None = None,
) -> tuple[Phi1Result, list[jnp.ndarray]]:
    """:func:`solve_phi1` that also returns the accepted Newton-iterate history.

    The output writer serializes v3's per-Newton-iteration diagnostics
    (the ``NIterations`` axis), so it needs the state after each accepted Newton
    step, not only the converged state.  Because each inner step is a full-restart
    (exact for the small Phi1 cases) Newton update on the same parity residual the
    Fortran ``SNES`` solves, the accepted iterates reproduce the Fortran
    per-iteration states.  Returns ``(result, history)`` where ``history[k]`` is
    the state after accepted Newton step ``k``.
    """
    return _newton_solve_phi1(
        operator_from_input(inp),
        x0=x0,
        tol=tol,
        max_newton=max_newton,
        gmres_tol=gmres_tol,
        gmres_restart=gmres_restart,
        gmres_recycle_dim=gmres_recycle_dim,
        gmres_max_restarts=gmres_max_restarts,
        warm_start=warm_start,
        line_search=line_search,
        solve_method=solve_method,
        use_preconditioner=use_preconditioner,
        emit=emit,
        record_history=True,
    )


def _newton_solve_phi1(
    op: KineticOperator,
    *,
    x0: Any | None,
    tol: float,
    max_newton: int,
    gmres_tol: float,
    gmres_restart: int | None,
    gmres_recycle_dim: int,
    gmres_max_restarts: int,
    warm_start: bool,
    line_search: bool,
    solve_method: str,
    use_preconditioner: bool,
    emit: Callable[[str], None] | None,
    record_history: bool,
) -> tuple[Phi1Result, list[jnp.ndarray]]:
    """Newton-Krylov core shared by :func:`solve_phi1` / :func:`solve_phi1_history`."""
    restart = _inner_restart(op, gmres_restart, use_preconditioner)

    if x0 is None:
        x = jnp.zeros((op.total_size,), dtype=jnp.float64)
    else:
        x = jnp.asarray(x0, dtype=jnp.float64).reshape((-1,))
        if x.shape != (op.total_size,):
            raise ValueError(f"x0 must have shape {(op.total_size,)}, got {x.shape}")

    recycle: tuple[jnp.ndarray, jnp.ndarray] | None = None
    iterations: list[NewtonIteration] = []
    history: list[jnp.ndarray] = []
    inner_total: int | None = 0
    converged = False
    t0 = time.perf_counter()

    for k in range(int(max_newton)):
        r = op.residual_phi1(x)
        rnorm = float(jnp.linalg.norm(r))
        if emit is not None:
            emit(f"{k:4d} SNES Function norm {rnorm: .12e}")
        if rnorm < float(tol):
            converged = True
            iterations.append(NewtonIteration(k, rnorm, 0, True, 0.0))
            break

        op_k = replace(op, phi1_lin_state=x)
        res: SolveResult = solve(
            op_k,
            -r,
            method=solve_method,
            tol=float(gmres_tol),
            use_preconditioner=use_preconditioner,
            restart=int(restart),
            recycle_dim=int(gmres_recycle_dim),
            max_restarts=int(gmres_max_restarts),
            recycle=recycle if warm_start else None,
        )
        recycle = res.recycle
        if res.iterations is None or inner_total is None:
            inner_total = None
        else:
            inner_total += int(res.iterations)
        step = jnp.reshape(res.x, (-1,))

        scale = 1.0
        if line_search:
            scale = _line_search_scale(op, x, step, rnorm)
        x = x + scale * step
        if record_history:
            history.append(x)
        iterations.append(
            NewtonIteration(k, rnorm, res.iterations, bool(res.converged), float(scale))
        )
    else:
        # Loop exhausted without an early break: check the final residual.
        r = op.residual_phi1(x)
        rnorm = float(jnp.linalg.norm(r))
        converged = rnorm < float(tol)

    elapsed = time.perf_counter() - t0
    op_out = replace(op, phi1_lin_state=x)
    phi1_hat = x[op.f_size : op.f_size + op.n_theta * op.n_zeta].reshape((op.n_theta, op.n_zeta))
    final_rnorm = float(jnp.linalg.norm(op.residual_phi1(x)))
    result = Phi1Result(
        x=x,
        phi1_hat=phi1_hat,
        operator=op_out,
        residual_norm=final_rnorm,
        converged=bool(converged),
        n_newton=len([it for it in iterations if it.inner_iterations != 0]),
        iterations=tuple(iterations),
        inner_iterations_total=inner_total,
        timings={"solve": elapsed},
    )
    return result, history


def _line_search_scale(
    op: KineticOperator, x: jnp.ndarray, step: jnp.ndarray, rnorm0: float
) -> float:
    """Backtracking scale in ``{1, 1/2, 1/4, ...}`` that reduces ``||F||``."""
    scale = 1.0
    for _ in range(8):
        r_try = op.residual_phi1(x + scale * step)
        if float(jnp.linalg.norm(r_try)) < rnorm0:
            return scale
        scale *= 0.5
    return 1.0  # no improvement found; take the full step and let Newton retry


# ---------------------------------------------------------------------------
# Differentiable Phi1 state (implicit function theorem)
# ---------------------------------------------------------------------------


def phi1_state(
    op: KineticOperator,
    *,
    x0: Any | None = None,
    tol: float = 1e-12,
    max_newton: int = 40,
) -> jnp.ndarray:
    """Differentiable solved Phi1 state ``x*`` (a JAX array).

    The nonlinear residual ``F(x) = 0`` (:meth:`KineticOperator.residual_phi1`)
    is a differentiable function of the operator's parameters; the root is found
    with a dense-Jacobian Newton loop wrapped by
    :func:`solvax.implicit.root_solve` (``jax.lax.custom_root``), so ``jax.grad``
    of any downstream moment w.r.t. a profile scalar the operator closes over
    flows through ``x*`` via the implicit function theorem -- the Jacobian
    ``dF/dx`` at ``x*`` comes from autodiff of the residual, not finite
    differences.

    The exact dense inner solve requires the un-truncated embedding
    (``Nxi_for_x_option=0``); truncated Phi1 differentiability is deferred.
    """
    from solvax.implicit import root_solve  # noqa: PLC0415

    if op.active_dof_mask() is not None:
        raise NotImplementedError(
            "phi1_state's differentiable dense Newton requires Nxi_for_x_option=0 "
            "(no Legendre truncation); use solve_phi1 for the truncated case."
        )
    n = int(op.total_size)
    if x0 is None:
        x_init = jnp.zeros((n,), dtype=jnp.float64)
    else:
        x_init = jnp.asarray(x0, dtype=jnp.float64).reshape((-1,))

    def residual(x: jnp.ndarray) -> jnp.ndarray:
        return op.residual_phi1(x)

    def solver(f: Callable[[jnp.ndarray], jnp.ndarray], x_seed: jnp.ndarray) -> jnp.ndarray:
        x_seed = jnp.asarray(x_seed, dtype=jnp.float64)

        def cond(state):
            x, r, k = state
            return (k < max_newton) & (jnp.linalg.norm(r) > tol)

        def body(state):
            x, r, k = state
            jac = jax.jacfwd(f)(x)  # (n, n) dense Jacobian (exact, small system)
            dx = jnp.linalg.solve(jac, -r)
            xn = x + dx
            return (xn, f(xn), k + 1)

        x_root, _r, _k = jax.lax.while_loop(cond, body, (x_seed, f(x_seed), 0))
        return x_root

    return root_solve(residual, x_init, solver)
