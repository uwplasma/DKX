from __future__ import annotations

# ruff: noqa: E402

from dataclasses import dataclass
from functools import lru_cache
from inspect import signature
import os
import numpy as np

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from jax import core as jax_core
from jax import tree_util as jtu
from jax import vmap
from jax.scipy.sparse.linalg import bicgstab, gmres

try:  # pragma: no cover - optional for distributed GMRES
    from jax.experimental import pjit as _pjit  # noqa: PLC0415
    from jax.sharding import Mesh, PartitionSpec  # noqa: PLC0415
except Exception:  # noqa: BLE001
    _pjit = None
    Mesh = None
    PartitionSpec = None
from scipy.sparse.linalg import LinearOperator as _LinearOperator
from scipy.sparse.linalg import gmres as _scipy_gmres
from scipy.sparse.linalg import bicgstab as _scipy_bicgstab
from scipy.sparse.linalg import gcrotmk as _scipy_gcrotmk
from scipy.sparse.linalg import lgmres as _scipy_lgmres

from .memory_model import gmres_restart_for_budget


@jtu.register_pytree_node_class
@dataclass(frozen=True)
class GMRESSolveResult:
    x: jnp.ndarray
    residual_norm: jnp.ndarray

    def tree_flatten(self):
        children = (self.x, self.residual_norm)
        aux = None
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        del aux
        x, residual_norm = children
        return cls(x=x, residual_norm=residual_norm)


@jtu.register_pytree_node_class
@dataclass(frozen=True)
class FlexibleGMRESSolveResult:
    """Result from the JAX-native flexible GMRES solver.

    ``residual_history`` stores the working residual norm after the initial
    guess and after every accepted Krylov update; the final entry stores the
    true unpreconditioned residual norm. ``x`` is always the physical solution
    vector for right-, left-, and unpreconditioned solves.
    """

    x: jnp.ndarray
    residual_norm: jnp.ndarray
    residual_history: jnp.ndarray
    n_iterations: jnp.ndarray
    n_restarts: jnp.ndarray
    converged: jnp.ndarray

    def tree_flatten(self):
        children = (
            self.x,
            self.residual_norm,
            self.residual_history,
            self.n_iterations,
            self.n_restarts,
            self.converged,
        )
        aux = None
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        del aux
        x, residual_norm, residual_history, n_iterations, n_restarts, converged = children
        return cls(
            x=x,
            residual_norm=residual_norm,
            residual_history=residual_history,
            n_iterations=n_iterations,
            n_restarts=n_restarts,
            converged=converged,
        )


_HOST_SCIPY_KRYLOV_METHODS = frozenset({"lgmres", "lgmres_scipy"})


def _contains_tracer(*values) -> bool:
    for value in values:
        if value is None:
            continue
        for leaf in jtu.tree_leaves(value):
            if isinstance(leaf, jax_core.Tracer):
                return True
    return False


def _normalize_krylov_method(solve_method: str) -> str:
    method = str(solve_method).strip().lower()
    if method in {"auto", "default"}:
        return "bicgstab"
    return method


def _maybe_limit_restart(n: int, restart: int, dtype: jnp.dtype) -> int:
    if n <= 0 or restart <= 1:
        return restart
    auto_env = os.environ.get("SFINCS_JAX_GMRES_AUTO_RESTART", "").strip().lower()
    if auto_env in {"0", "false", "no", "off"}:
        return restart
    max_mb_env = os.environ.get("SFINCS_JAX_GMRES_MAX_MB", "").strip()
    if max_mb_env:
        try:
            max_mb = float(max_mb_env)
        except ValueError:
            max_mb = 2048.0
    else:
        max_mb = 2048.0
    if max_mb <= 0:
        return restart
    bytes_per_elem = int(np.dtype(dtype).itemsize)
    if bytes_per_elem <= 0:
        return restart
    max_bytes = max_mb * 1e6
    # Estimate Krylov basis storage plus a few work vectors. The conservative
    # model matches the standalone memory preflight helper used by benchmarks.
    return gmres_restart_for_budget(
        n=int(n),
        requested_restart=int(restart),
        dtype=np.dtype(dtype),
        max_bytes=max_bytes,
    )


def _materialize_distributed_input(arr: jnp.ndarray | None, *, dtype: jnp.dtype | None = None) -> jnp.ndarray | None:
    """Return an unsharded host-materialized copy suitable for pjit input resharding."""
    if arr is None:
        return None
    host_arr = jax.device_get(arr)
    return jnp.asarray(host_arr, dtype=dtype)


def _preconditioner_accepts_iteration(preconditioner) -> bool:
    """Return whether ``preconditioner`` appears to accept an iteration index."""
    if preconditioner is None:
        return False
    try:
        sig = signature(preconditioner)
    except (TypeError, ValueError):
        return False
    positional = [
        p
        for p in sig.parameters.values()
        if p.kind in {p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD}
        and p.default is p.empty
    ]
    has_varargs = any(p.kind == p.VAR_POSITIONAL for p in sig.parameters.values())
    return bool(has_varargs or len(positional) >= 2)


def gmres_solve_with_history_scipy(
    *,
    matvec,
    b: jnp.ndarray,
    preconditioner=None,
    x0: jnp.ndarray | None = None,
    tol: float = 1e-10,
    atol: float = 0.0,
    restart: int = 50,
    maxiter: int | None = None,
    precondition_side: str = "left",
) -> tuple[np.ndarray, float, list[float]]:
    """Run SciPy GMRES to collect residual history for Fortran-style logging."""
    b_np = np.array(b, dtype=np.float64, copy=True).reshape((-1,))
    n = int(b_np.size)
    x0_np = np.array(x0, dtype=np.float64, copy=True).reshape((-1,)) if x0 is not None else None
    restart_use = _maybe_limit_restart(n, int(restart), np.dtype(np.float64))

    def _mv(x_np: np.ndarray) -> np.ndarray:
        return np.array(matvec(jnp.asarray(x_np, dtype=jnp.float64)), dtype=np.float64, copy=True)

    def _prec(x_np: np.ndarray) -> np.ndarray:
        if preconditioner is None:
            return np.array(x_np, dtype=np.float64, copy=True)
        return np.array(preconditioner(jnp.asarray(x_np, dtype=jnp.float64)), dtype=np.float64, copy=True)

    side = str(precondition_side).strip().lower()
    if side not in {"left", "right", "none"}:
        side = "left"

    if side == "right" and preconditioner is not None:
        def _mv_right(y_np: np.ndarray) -> np.ndarray:
            return _mv(_prec(y_np))

        A = _LinearOperator((n, n), matvec=_mv_right, dtype=np.float64)
        M = None
    else:
        A = _LinearOperator((n, n), matvec=_mv, dtype=np.float64)
        M = _LinearOperator((n, n), matvec=_prec, dtype=np.float64) if preconditioner is not None else None

    history: list[float] = []

    def _cb(arg):
        # SciPy passes residual norm when callback_type='pr_norm'.
        if np.isscalar(arg):
            history.append(float(arg))
        else:
            history.append(float(np.linalg.norm(arg)))

    x_np, info = _scipy_gmres(
        A,
        b_np,
        x0=x0_np,
        rtol=float(tol),
        atol=float(atol),
        restart=int(restart_use),
        maxiter=int(maxiter) if maxiter is not None else None,
        M=M,
        callback=_cb,
        callback_type="pr_norm",
    )

    if side == "right" and preconditioner is not None:
        x_np = _prec(x_np)

    res = b_np - _mv(x_np)
    rn = float(np.linalg.norm(res))
    return x_np, rn, history


def lgmres_solve_with_history_scipy(
    *,
    matvec,
    b: jnp.ndarray,
    preconditioner=None,
    x0: jnp.ndarray | None = None,
    tol: float = 1e-10,
    atol: float = 0.0,
    restart: int = 50,
    maxiter: int | None = None,
    outer_k: int | None = None,
    precondition_side: str = "left",
) -> tuple[np.ndarray, float, list[float]]:
    """Run SciPy LGMRES for restart-robust host solves on non-differentiable paths."""
    b_np = np.array(b, dtype=np.float64, copy=True).reshape((-1,))
    n = int(b_np.size)
    x0_np = np.array(x0, dtype=np.float64, copy=True).reshape((-1,)) if x0 is not None else None
    restart_use = _maybe_limit_restart(n, int(restart), np.dtype(np.float64))
    if outer_k is None:
        outer_k_env = os.environ.get("SFINCS_JAX_LGMRES_OUTER_K", "").strip()
        try:
            outer_k_use = int(outer_k_env) if outer_k_env else 3
        except ValueError:
            outer_k_use = 3
    else:
        outer_k_use = int(outer_k)
    outer_k_use = max(0, int(outer_k_use))

    def _mv(x_np: np.ndarray) -> np.ndarray:
        return np.array(matvec(jnp.asarray(x_np, dtype=jnp.float64)), dtype=np.float64, copy=True)

    def _prec(x_np: np.ndarray) -> np.ndarray:
        if preconditioner is None:
            return np.array(x_np, dtype=np.float64, copy=True)
        return np.array(preconditioner(jnp.asarray(x_np, dtype=jnp.float64)), dtype=np.float64, copy=True)

    side = str(precondition_side).strip().lower()
    if side not in {"left", "right", "none"}:
        side = "left"

    if side == "right" and preconditioner is not None:

        def _mv_right(y_np: np.ndarray) -> np.ndarray:
            return _mv(_prec(y_np))

        A = _LinearOperator((n, n), matvec=_mv_right, dtype=np.float64)
        M = None
    else:
        A = _LinearOperator((n, n), matvec=_mv, dtype=np.float64)
        M = _LinearOperator((n, n), matvec=_prec, dtype=np.float64) if preconditioner is not None else None

    history: list[float] = []

    def _cb(xk: np.ndarray) -> None:
        x_state = _prec(xk) if side == "right" and preconditioner is not None else xk
        history.append(float(np.linalg.norm(b_np - _mv(x_state))))

    x_np, _info = _scipy_lgmres(
        A,
        b_np,
        x0=x0_np,
        rtol=float(tol),
        atol=float(atol),
        maxiter=int(maxiter) if maxiter is not None else None,
        M=M,
        inner_m=int(restart_use),
        outer_k=int(outer_k_use),
        callback=_cb,
    )

    if side == "right" and preconditioner is not None:
        x_np = _prec(x_np)

    res = b_np - _mv(x_np)
    rn = float(np.linalg.norm(res))
    return x_np, rn, history


def gcrotmk_solve_with_history_scipy(
    *,
    matvec,
    b: jnp.ndarray,
    preconditioner=None,
    x0: jnp.ndarray | None = None,
    tol: float = 1e-10,
    atol: float = 0.0,
    restart: int = 50,
    maxiter: int | None = None,
    precondition_side: str = "left",
) -> tuple[np.ndarray, float, list[float]]:
    """Run SciPy GCROT(m,k) for flexible global-Krylov host solves.

    GCROT keeps a small recycled correction subspace in addition to the inner
    Krylov basis. It is useful for production host fallback paths that plateau
    under restarted GMRES but do not justify assembling a larger dense coarse
    correction.
    """
    b_np = np.array(b, dtype=np.float64, copy=True).reshape((-1,))
    n = int(b_np.size)
    x0_np = np.array(x0, dtype=np.float64, copy=True).reshape((-1,)) if x0 is not None else None
    restart_use = _maybe_limit_restart(n, int(restart), np.dtype(np.float64))

    outer_k_env = os.environ.get("SFINCS_JAX_GCROTMK_OUTER_K", "").strip()
    try:
        outer_k = int(outer_k_env) if outer_k_env else min(20, max(1, int(restart_use) // 2))
    except ValueError:
        outer_k = min(20, max(1, int(restart_use) // 2))
    outer_k = max(0, int(outer_k))

    def _mv(x_np: np.ndarray) -> np.ndarray:
        return np.array(matvec(jnp.asarray(x_np, dtype=jnp.float64)), dtype=np.float64, copy=True)

    def _prec(x_np: np.ndarray) -> np.ndarray:
        if preconditioner is None:
            return np.array(x_np, dtype=np.float64, copy=True)
        return np.array(preconditioner(jnp.asarray(x_np, dtype=jnp.float64)), dtype=np.float64, copy=True)

    side = str(precondition_side).strip().lower()
    if side not in {"left", "right", "none"}:
        side = "left"

    if side == "right" and preconditioner is not None:

        def _mv_right(y_np: np.ndarray) -> np.ndarray:
            return _mv(_prec(y_np))

        A = _LinearOperator((n, n), matvec=_mv_right, dtype=np.float64)
        M = None
    else:
        A = _LinearOperator((n, n), matvec=_mv, dtype=np.float64)
        M = _LinearOperator((n, n), matvec=_prec, dtype=np.float64) if preconditioner is not None else None

    history: list[float] = []

    def _cb(xk: np.ndarray) -> None:
        x_state = _prec(xk) if side == "right" and preconditioner is not None else xk
        history.append(float(np.linalg.norm(b_np - _mv(x_state))))

    x_np, _info = _scipy_gcrotmk(
        A,
        b_np,
        x0=x0_np,
        rtol=float(tol),
        atol=float(atol),
        maxiter=int(maxiter) if maxiter is not None else None,
        M=M,
        callback=_cb,
        m=int(restart_use),
        k=outer_k,
        discard_C=False,
        truncate="oldest",
    )

    if side == "right" and preconditioner is not None:
        x_np = _prec(x_np)

    res = b_np - _mv(x_np)
    rn = float(np.linalg.norm(res))
    return x_np, rn, history


def explicit_left_preconditioned_gmres_scipy(
    *,
    matvec,
    b: jnp.ndarray,
    preconditioner,
    x0: jnp.ndarray | None = None,
    tol: float = 1e-10,
    atol: float = 0.0,
    restart: int = 50,
    maxiter: int | None = None,
) -> tuple[np.ndarray, float, float, list[float]]:
    """Run SciPy GMRES on the explicit left-preconditioned system M^{-1} A x = M^{-1} b."""
    b_np = np.array(b, dtype=np.float64, copy=True).reshape((-1,))
    n = int(b_np.size)
    x0_np = np.array(x0, dtype=np.float64, copy=True).reshape((-1,)) if x0 is not None else None
    restart_use = _maybe_limit_restart(n, int(restart), np.dtype(np.float64))

    def _mv(x_np: np.ndarray) -> np.ndarray:
        return np.array(matvec(jnp.asarray(x_np, dtype=jnp.float64)), dtype=np.float64, copy=True)

    def _prec(x_np: np.ndarray) -> np.ndarray:
        return np.array(preconditioner(jnp.asarray(x_np, dtype=jnp.float64)), dtype=np.float64, copy=True)

    rhs_pc = _prec(b_np)
    rhs_pc_norm = float(np.linalg.norm(rhs_pc))
    if np.isfinite(rhs_pc_norm) and rhs_pc_norm == 0.0:
        x_zero = np.zeros_like(b_np)
        rn_true = float(np.linalg.norm(b_np))
        return x_zero, rn_true, 0.0, [0.0]

    def _mv_pc(x_np: np.ndarray) -> np.ndarray:
        return _prec(_mv(x_np))

    a_pc = _LinearOperator((n, n), matvec=_mv_pc, dtype=np.float64)
    history: list[float] = []

    def _cb(arg):
        if np.isscalar(arg):
            history.append(float(arg))
        else:
            history.append(float(np.linalg.norm(arg)))

    x_np, info = _scipy_gmres(
        a_pc,
        rhs_pc,
        x0=x0_np,
        rtol=float(tol),
        atol=float(atol),
        restart=int(restart_use),
        maxiter=int(maxiter) if maxiter is not None else None,
        M=None,
        callback=_cb,
        callback_type="pr_norm",
    )
    del info

    res_true = b_np - _mv(x_np)
    rn_true = float(np.linalg.norm(res_true))
    res_pc = rhs_pc - _mv_pc(x_np)
    rn_pc = float(np.linalg.norm(res_pc))
    return x_np, rn_true, rn_pc, history


def bicgstab_solve_with_history_scipy(
    *,
    matvec,
    b: jnp.ndarray,
    preconditioner=None,
    x0: jnp.ndarray | None = None,
    tol: float = 1e-10,
    atol: float = 0.0,
    maxiter: int | None = None,
    precondition_side: str = "left",
) -> tuple[np.ndarray, float, list[float]]:
    """Run SciPy BiCGStab to collect residual history for iteration counts."""
    b_np = np.array(b, dtype=np.float64, copy=True).reshape((-1,))
    n = int(b_np.size)
    x0_np = np.array(x0, dtype=np.float64, copy=True).reshape((-1,)) if x0 is not None else None

    def _mv(x_np: np.ndarray) -> np.ndarray:
        return np.array(matvec(jnp.asarray(x_np, dtype=jnp.float64)), dtype=np.float64, copy=True)

    def _prec(x_np: np.ndarray) -> np.ndarray:
        if preconditioner is None:
            return np.array(x_np, dtype=np.float64, copy=True)
        return np.array(preconditioner(jnp.asarray(x_np, dtype=jnp.float64)), dtype=np.float64, copy=True)

    side = str(precondition_side).strip().lower()
    if side not in {"left", "right", "none"}:
        side = "left"

    if side == "right" and preconditioner is not None:
        def _mv_right(y_np: np.ndarray) -> np.ndarray:
            return _mv(_prec(y_np))

        A = _LinearOperator((n, n), matvec=_mv_right, dtype=np.float64)
        M = None
    else:
        A = _LinearOperator((n, n), matvec=_mv, dtype=np.float64)
        M = _LinearOperator((n, n), matvec=_prec, dtype=np.float64) if preconditioner is not None and side == "left" else None

    history: list[float] = []

    def _cb(xk: np.ndarray) -> None:
        rk = b_np - _mv(xk if side != "right" else _prec(xk))
        history.append(float(np.linalg.norm(rk)))

    x_np, _info = _scipy_bicgstab(
        A,
        b_np,
        x0=x0_np,
        rtol=float(tol),
        atol=float(atol),
        maxiter=int(maxiter) if maxiter is not None else None,
        M=M,
        callback=_cb,
    )

    if side == "right" and preconditioner is not None:
        x_np = _prec(x_np)

    res = b_np - _mv(x_np)
    rn = float(np.linalg.norm(res))
    return x_np, rn, history


def _bicgstab_solve_core(
    *,
    matvec,
    b: jnp.ndarray,
    preconditioner=None,
    x0: jnp.ndarray | None = None,
    tol: float = 1e-10,
    atol: float = 0.0,
    maxiter: int | None = None,
    precondition_side: str = "left",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    b = jnp.asarray(b)
    if x0 is not None:
        x0 = jnp.asarray(x0)

    side = str(precondition_side).strip().lower()
    if side not in {"left", "right", "none"}:
        side = "left"

    if side == "right" and preconditioner is not None:
        def matvec_right(y):
            return matvec(preconditioner(y))

        y, _info = bicgstab(
            matvec_right,
            b,
            x0=None,
            tol=float(tol),
            atol=float(atol),
            maxiter=maxiter,
            M=None,
        )
        x = preconditioner(y)
    else:
        M = preconditioner if side == "left" else None
        x, _info = bicgstab(
            matvec,
            b,
            x0=x0,
            tol=float(tol),
            atol=float(atol),
            maxiter=maxiter,
            M=M,
        )

    r = b - matvec(x)
    return x, r


def bicgstab_solve(
    *,
    matvec,
    b: jnp.ndarray,
    preconditioner=None,
    x0: jnp.ndarray | None = None,
    tol: float = 1e-10,
    atol: float = 0.0,
    maxiter: int | None = None,
    precondition_side: str = "left",
) -> GMRESSolveResult:
    """Solve `A x = b` using JAX's BiCGStab (short-recurrence Krylov, O(n) memory)."""
    x, r = _bicgstab_solve_core(
        matvec=matvec,
        b=b,
        preconditioner=preconditioner,
        x0=x0,
        tol=tol,
        atol=atol,
        maxiter=maxiter,
        precondition_side=precondition_side,
    )
    return GMRESSolveResult(x=x, residual_norm=jnp.linalg.norm(r))


def bicgstab_solve_with_residual(
    *,
    matvec,
    b: jnp.ndarray,
    preconditioner=None,
    x0: jnp.ndarray | None = None,
    tol: float = 1e-10,
    atol: float = 0.0,
    maxiter: int | None = None,
    precondition_side: str = "left",
) -> tuple[GMRESSolveResult, jnp.ndarray]:
    """Solve `A x = b` and return both the GMRES-style result and residual vector."""
    x, r = _bicgstab_solve_core(
        matvec=matvec,
        b=b,
        preconditioner=preconditioner,
        x0=x0,
        tol=tol,
        atol=atol,
        maxiter=maxiter,
        precondition_side=precondition_side,
    )
    return GMRESSolveResult(x=x, residual_norm=jnp.linalg.norm(r)), r


bicgstab_solve_jit = jax.jit(
    bicgstab_solve,
    static_argnames=("matvec", "preconditioner", "tol", "atol", "maxiter", "precondition_side"),
)

bicgstab_solve_with_residual_jit = jax.jit(
    bicgstab_solve_with_residual,
    static_argnames=("matvec", "preconditioner", "tol", "atol", "maxiter", "precondition_side"),
)


def assemble_dense_matrix_from_matvec(*, matvec, n: int, dtype: jnp.dtype) -> jnp.ndarray:
    """Assemble a dense matrix from a matrix-free `matvec`."""
    eye = jnp.eye(int(n), dtype=dtype)
    block_env = os.environ.get("SFINCS_JAX_DENSE_BLOCK", "").strip()
    try:
        block = int(block_env) if block_env else 0
    except ValueError:
        block = 0
    if block == 0 and int(n) >= 1000:
        # Limit peak memory for larger dense assemblies.
        block = 128
    jit_env = os.environ.get("SFINCS_JAX_DENSE_ASSEMBLE_JIT", "").strip().lower()
    if jit_env:
        use_jit = jit_env not in {"0", "false", "no", "off"}
    else:
        use_jit = int(n) > 800

    def _assemble(block_cols: jnp.ndarray) -> jnp.ndarray:
        return vmap(matvec, in_axes=1, out_axes=1)(block_cols)

    assemble_fn = jax.jit(_assemble) if use_jit else _assemble
    if block > 0 and block < int(n):
        cols = []
        for start in range(0, int(n), int(block)):
            cols.append(assemble_fn(eye[:, start : start + int(block)]))
        return jnp.concatenate(cols, axis=1)
    return assemble_fn(eye)


def dense_solve_from_matrix(*, a: jnp.ndarray, b: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Solve `A X = B` with v3-compatible singular handling.

    Parameters
    ----------
    a:
        Dense square matrix, shape `(N,N)`.
    b:
        Right-hand side, shape `(N,)` or `(N,K)`.
    """
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"dense_solve_from_matrix expects a square matrix, got shape {a.shape}")
    if b.ndim not in (1, 2):
        raise ValueError(f"dense_solve_from_matrix expects b.ndim in {{1,2}}, got {b.ndim}")
    if b.shape[0] != a.shape[0]:
        raise ValueError(f"dense_solve_from_matrix shape mismatch: a={a.shape}, b={b.shape}")

    n = int(a.shape[0])
    b2 = b[:, None] if b.ndim == 1 else b
    eye = jnp.eye(n, dtype=a.dtype)

    x_direct = jnp.linalg.solve(a, b2)
    direct_finite = jnp.all(jnp.isfinite(x_direct))

    reg_val = 2.2e-10
    env_reg = os.environ.get("SFINCS_JAX_DENSE_REG", "").strip()
    if env_reg:
        reg_val = float(env_reg)
    reg = jnp.asarray(reg_val, dtype=a.dtype)

    singular_mode = os.environ.get("SFINCS_JAX_DENSE_SINGULAR_MODE", "").strip().lower()
    force_reg = os.environ.get("SFINCS_JAX_DENSE_FORCE_REG", "").strip().lower() in {"1", "true", "yes", "on"}
    force_lstsq = singular_mode == "lstsq"

    def _solve_lstsq(_):
        gram = a.T @ a
        rhs = a.T @ b2
        diag = jnp.diag(gram)
        scale = jnp.maximum(jnp.max(jnp.abs(diag)), jnp.asarray(1.0, dtype=a.dtype))
        reg_lstsq = jnp.asarray(1.0e-12, dtype=a.dtype) * scale
        return jnp.linalg.solve(gram + reg_lstsq * eye, rhs)

    if force_reg:
        x2 = jnp.linalg.solve(a + reg * eye, b2)
    elif force_lstsq:
        x2 = _solve_lstsq(None)
    else:
        x2 = jax.lax.cond(direct_finite, lambda _: x_direct, _solve_lstsq, operand=None)

    x2 = jax.lax.cond(jnp.all(jnp.isfinite(x2)), lambda _: x2, _solve_lstsq, operand=None)
    r2 = b2 - a @ x2
    rn = jnp.linalg.norm(r2, axis=0)

    if b.ndim == 1:
        return x2[:, 0], rn[0]
    return x2, rn


def dense_solve_from_matrix_row_scaled(*, a: jnp.ndarray, b: jnp.ndarray, diag_floor: float = 1e-12) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Solve `A X = B` using diagonal row scaling before a direct dense solve.

    This is intended for singular/near-singular systems where solver-dependent
    pivoting choices can shift the nullspace component of the solution. Row
    scaling by the diagonal produces a deterministic gauge that can be used
    for parity-sensitive RHSMode=1 constraintScheme=0 solves.
    """
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"dense_solve_from_matrix_row_scaled expects a square matrix, got shape {a.shape}")
    if b.ndim not in (1, 2):
        raise ValueError(f"dense_solve_from_matrix_row_scaled expects b.ndim in {{1,2}}, got {b.ndim}")
    if b.shape[0] != a.shape[0]:
        raise ValueError(f"dense_solve_from_matrix_row_scaled shape mismatch: a={a.shape}, b={b.shape}")

    b2 = b[:, None] if b.ndim == 1 else b
    diag = jnp.diag(a)
    floor = jnp.asarray(diag_floor, dtype=a.dtype)
    denom = jnp.where(jnp.abs(diag) < floor, jnp.asarray(1.0, dtype=a.dtype), diag)
    a_scaled = a / denom[:, None]
    b_scaled = b2 / denom[:, None]

    x = jnp.linalg.solve(a_scaled, b_scaled)
    r = b2 - a @ x
    rn = jnp.linalg.norm(r, axis=0)
    if b.ndim == 1:
        return x[:, 0], rn[0]
    return x, rn


def dense_krylov_solve_from_matrix_with_residual(
    *,
    a: jnp.ndarray,
    b: jnp.ndarray,
    x0: jnp.ndarray | None = None,
    preconditioner=None,
    tol: float = 1e-10,
    atol: float = 0.0,
    restart: int | None = None,
    maxiter: int | None = None,
    solve_method: str = "incremental",
    precondition_side: str = "left",
    row_scaled: bool = False,
    diag_floor: float = 1e-12,
) -> tuple[GMRESSolveResult, jnp.ndarray]:
    """Solve a dense system with Krylov iterations on an explicit matrix.

    This path is intended for accelerator-safe fallback solves where dense direct
    factorizations are unavailable or undesirable, while keeping the solve in JAX.
    """

    a = jnp.asarray(a)
    b = jnp.asarray(b)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"dense_krylov_solve_from_matrix expects a square matrix, got shape {a.shape}")
    if b.ndim != 1 or b.shape[0] != a.shape[0]:
        raise ValueError(f"dense_krylov_solve_from_matrix expects b.shape {(a.shape[0],)}, got {b.shape}")

    n = int(a.shape[0])
    restart_env = os.environ.get("SFINCS_JAX_DENSE_KRYLOV_RESTART", "").strip()
    if restart_env:
        try:
            restart_use = int(restart_env)
        except ValueError:
            restart_use = 0
    elif restart is None:
        restart_use = min(n, 2000 if row_scaled else 1024)
    else:
        restart_use = int(restart)
    restart_use = max(1, min(int(restart_use), n))

    maxiter_env = os.environ.get("SFINCS_JAX_DENSE_KRYLOV_MAXITER", "").strip()
    if maxiter_env:
        try:
            maxiter_use = int(maxiter_env)
        except ValueError:
            maxiter_use = None
    else:
        maxiter_use = maxiter
    if maxiter_use is None:
        maxiter_use = max(1, int(np.ceil(n / max(1, restart_use))))

    a_use = a
    b_use = b
    x0_use = None if x0 is None else jnp.asarray(x0, dtype=a.dtype)
    preconditioner_use = preconditioner
    if row_scaled:
        diag = jnp.diag(a_use)
        floor = jnp.asarray(diag_floor, dtype=a.dtype)
        denom = jnp.where(jnp.abs(diag) < floor, jnp.asarray(1.0, dtype=a.dtype), diag)
        a_use = a_use / denom[:, None]
        b_use = b_use / denom
        # The supplied preconditioner targets the unscaled system, so disable it here.
        preconditioner_use = None
        x0_use = None if x0_use is None else jnp.asarray(x0_use, dtype=a.dtype)

    def matvec(x: jnp.ndarray) -> jnp.ndarray:
        return a_use @ x

    result, _scaled_residual = gmres_solve_with_residual(
        matvec=matvec,
        b=b_use,
        preconditioner=preconditioner_use,
        x0=x0_use,
        tol=tol,
        atol=atol,
        restart=restart_use,
        maxiter=maxiter_use,
        solve_method=solve_method,
        precondition_side=precondition_side,
    )
    residual = b - a @ result.x
    return GMRESSolveResult(x=result.x, residual_norm=jnp.linalg.norm(residual)), residual


def dense_krylov_solve_from_matrix(**kwargs) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Solve a dense system with Krylov iterations on an explicit matrix."""
    result, residual = dense_krylov_solve_from_matrix_with_residual(**kwargs)
    return result.x, result.residual_norm


def fgmres_solve_with_residual(
    *,
    matvec,
    b: jnp.ndarray,
    preconditioner=None,
    x0: jnp.ndarray | None = None,
    tol: float = 1e-10,
    atol: float = 0.0,
    restart: int = 50,
    maxiter: int | None = None,
    breakdown_tol: float = 1e-14,
    precondition_side: str = "right",
    skip_inactive_work: bool = True,
) -> tuple[FlexibleGMRESSolveResult, jnp.ndarray]:
    """Solve ``A x = b`` with a fixed-shape GMRES/FGMRES implementation in JAX.

    This routine is intended for accelerator-compatible production experiments
    where a fixed linear ``M`` is too restrictive. Unlike JAX's built-in GMRES,
    the right-preconditioned path may vary the preconditioner by iteration:
    pass either ``M(v)`` or ``M(v, iteration)``. The left-preconditioned path is
    available for fixed-preconditioner probes that need to preserve legacy
    side-selection behavior. ``maxiter`` is the total Krylov-iteration budget,
    not the number of restart cycles. ``skip_inactive_work`` avoids expensive
    preconditioner/matvec calls after convergence for trace-safe device
    preconditioners; set it to ``False`` for legacy host preconditioners that
    intentionally call ``device_get`` inside their apply path.

    The implementation deliberately uses fixed-shape JAX arrays for the Arnoldi
    and least-squares state. It does not convert residuals to Python scalars
    inside the iteration loop, so accelerator calls avoid per-iteration
    host/device synchronization. A separate ``jax.jit`` wrapper can trace this
    function when ``matvec`` and ``preconditioner`` are trace-safe callables.
    """
    b = jnp.asarray(b)
    if b.ndim != 1:
        raise ValueError(f"fgmres_solve_with_residual expects 1D b, got shape {b.shape}")
    dtype = b.dtype
    n = int(b.size)
    restart_use = max(1, min(int(restart), max(1, n)))
    if maxiter is None:
        maxiter_use = max(restart_use, n)
    else:
        maxiter_use = max(1, int(maxiter))
    maxiter_use = max(1, int(maxiter_use))
    x = jnp.zeros_like(b) if x0 is None else jnp.asarray(x0, dtype=dtype)
    if x.shape != b.shape:
        raise ValueError(f"fgmres_solve_with_residual x0 shape mismatch: expected {b.shape}, got {x.shape}")

    preconditioner_uses_iteration = _preconditioner_accepts_iteration(preconditioner)
    side = str(precondition_side).strip().lower()
    if side not in {"left", "right", "none"}:
        side = "right"

    def _apply_preconditioner(v: jnp.ndarray, iteration: int) -> jnp.ndarray:
        if preconditioner is None:
            return v
        if preconditioner_uses_iteration:
            return jnp.asarray(preconditioner(v, int(iteration)), dtype=v.dtype)
        return jnp.asarray(preconditioner(v), dtype=v.dtype)

    if side == "left":
        work_b = _apply_preconditioner(b, 0)

        def _work_matvec(v: jnp.ndarray, iteration: int) -> jnp.ndarray:
            return _apply_preconditioner(jnp.asarray(matvec(v), dtype=dtype), iteration)

        def _search_direction(v: jnp.ndarray, iteration: int) -> jnp.ndarray:
            del iteration
            return v

    else:
        work_b = b

        def _work_matvec(v: jnp.ndarray, iteration: int) -> jnp.ndarray:
            del iteration
            return jnp.asarray(matvec(v), dtype=dtype)

        def _search_direction(v: jnp.ndarray, iteration: int) -> jnp.ndarray:
            if side == "none":
                del iteration
                return v
            return _apply_preconditioner(v, iteration)

    rhs_norm = jnp.linalg.norm(work_b)
    target = jnp.maximum(jnp.asarray(atol, dtype=dtype), jnp.asarray(tol, dtype=dtype) * rhs_norm)
    true_target = jnp.maximum(jnp.asarray(atol, dtype=dtype), jnp.asarray(tol, dtype=dtype) * jnp.linalg.norm(b))
    residual = work_b - _work_matvec(x, 0)
    residual_norm = jnp.linalg.norm(residual)
    residual_history = jnp.full((maxiter_use + 1,), residual_norm, dtype=dtype)
    converged = jnp.logical_and(jnp.isfinite(residual_norm), residual_norm <= target)
    breakdown_tol_use = max(0.0, float(breakdown_tol))
    breakdown_threshold = jnp.asarray(breakdown_tol_use, dtype=dtype)

    max_cycles = max(1, int(np.ceil(maxiter_use / restart_use)))
    iteration_index = 0
    for cycle in range(max_cycles):
        beta = jnp.linalg.norm(residual)
        cycle_active = jnp.logical_and(~converged, jnp.logical_and(jnp.isfinite(beta), beta > breakdown_threshold))
        v_basis = jnp.zeros((restart_use + 1, n), dtype=dtype)
        z_basis = jnp.zeros((restart_use, n), dtype=dtype)
        hessenberg = jnp.zeros((restart_use + 1, restart_use), dtype=dtype)
        v0 = jnp.where(cycle_active, residual / jnp.where(beta > 0, beta, jnp.asarray(1.0, dtype=dtype)), 0.0)
        v_basis = v_basis.at[0].set(v0)
        x_cycle_base = x
        cycle_budget = min(restart_use, maxiter_use - iteration_index)
        for j in range(cycle_budget):
            active_step = jnp.logical_and(cycle_active, ~converged)

            if bool(skip_inactive_work):
                z = jax.lax.cond(
                    active_step,
                    lambda v: _search_direction(v, iteration_index),
                    lambda v: jnp.zeros_like(v),
                    v_basis[j],
                )
            else:
                z = _search_direction(v_basis[j], iteration_index)
            z = jnp.asarray(z, dtype=dtype)
            if z.shape != b.shape:
                raise ValueError(
                    "fgmres_solve_with_residual preconditioner shape mismatch: "
                    f"expected {b.shape}, got {z.shape}"
                )
            z = jnp.where(active_step, z, jnp.zeros_like(z))
            z_basis = z_basis.at[j].set(z)

            if bool(skip_inactive_work):
                w = jax.lax.cond(
                    active_step,
                    lambda zz: jnp.asarray(_work_matvec(zz, iteration_index), dtype=dtype),
                    lambda zz: jnp.zeros_like(zz),
                    z,
                )
            else:
                w = jnp.asarray(_work_matvec(z, iteration_index), dtype=dtype)
            if w.shape != b.shape:
                raise ValueError(
                    f"fgmres_solve_with_residual matvec shape mismatch: expected {b.shape}, got {w.shape}"
                )
            for i in range(j + 1):
                hij = jnp.vdot(v_basis[i], w)
                hessenberg = hessenberg.at[i, j].set(hij)
                w = w - hij * v_basis[i]
            h_next = jnp.linalg.norm(w)
            hessenberg = hessenberg.at[j + 1, j].set(h_next)
            arnoldi_ok = jnp.logical_and(jnp.isfinite(h_next), h_next > breakdown_threshold)
            v_next = jnp.where(
                jnp.logical_and(active_step, arnoldi_ok),
                w / jnp.where(h_next > 0, h_next, jnp.asarray(1.0, dtype=dtype)),
                jnp.zeros_like(w),
            )
            v_basis = v_basis.at[j + 1].set(v_next)

            small_rhs = jnp.zeros((j + 2,), dtype=dtype).at[0].set(beta)
            small_h = hessenberg[: j + 2, : j + 1]
            coeff = jnp.linalg.lstsq(small_h, small_rhs, rcond=None)[0]
            candidate_update = coeff @ z_basis[: j + 1]
            candidate_x = x_cycle_base + candidate_update
            if bool(skip_inactive_work):
                candidate_residual = jax.lax.cond(
                    active_step,
                    lambda xx: work_b - jnp.asarray(_work_matvec(xx, iteration_index), dtype=dtype),
                    lambda _xx: residual,
                    candidate_x,
                )
            else:
                candidate_residual = work_b - jnp.asarray(_work_matvec(candidate_x, iteration_index), dtype=dtype)
            candidate_norm = jnp.where(active_step, jnp.linalg.norm(candidate_residual), residual_norm)
            update_ok = jnp.logical_and(active_step, jnp.isfinite(candidate_norm))
            x = jnp.where(update_ok, candidate_x, x)
            residual = jnp.where(update_ok, candidate_residual, residual)
            residual_norm = jnp.where(update_ok, candidate_norm, residual_norm)
            residual_history = residual_history.at[iteration_index + 1].set(residual_norm)
            converged = jnp.logical_or(converged, jnp.logical_and(update_ok, residual_norm <= target))
            cycle_active = jnp.logical_and(cycle_active, arnoldi_ok)
            iteration_index += 1

    residual_final = b - jnp.asarray(matvec(x), dtype=dtype)
    residual_norm_final = jnp.linalg.norm(residual_final)
    residual_history = residual_history.at[-1].set(residual_norm_final)
    history_finite = jnp.isfinite(residual_history)
    history_converged = jnp.logical_and(history_finite, residual_history <= target)
    first_converged = jnp.min(
        jnp.where(history_converged, jnp.arange(maxiter_use + 1, dtype=jnp.int32), maxiter_use + 1)
    )
    true_converged = residual_norm_final <= true_target
    converged_final = jnp.logical_and(first_converged <= maxiter_use, true_converged)
    n_iterations = jnp.where(converged_final, first_converged, maxiter_use)
    n_restarts = jnp.where(n_iterations > 0, (n_iterations - 1) // restart_use, 0)
    result = FlexibleGMRESSolveResult(
        x=x,
        residual_norm=residual_norm_final,
        residual_history=residual_history,
        n_iterations=jnp.asarray(n_iterations, dtype=jnp.int32),
        n_restarts=jnp.asarray(n_restarts, dtype=jnp.int32),
        converged=jnp.asarray(converged_final, dtype=jnp.bool_),
    )
    return result, residual_final


fgmres_solve_with_residual_jit = jax.jit(
    fgmres_solve_with_residual,
    static_argnames=(
        "matvec",
        "preconditioner",
        "tol",
        "atol",
        "restart",
        "maxiter",
        "breakdown_tol",
        "precondition_side",
        "skip_inactive_work",
    ),
)


def _gmres_solve_core(
    *,
    matvec,
    b: jnp.ndarray,
    preconditioner=None,
    x0: jnp.ndarray | None = None,
    tol: float = 1e-10,
    atol: float = 0.0,
    restart: int = 50,
    maxiter: int | None = None,
    solve_method: str = "incremental",
    precondition_side: str = "left",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Solve `A x = b` using JAX's GMRES.

    Notes
    -----
    - `matvec` must be callable like `matvec(x)` and return the same shape as `x`.
    - JAX's `gmres` currently returns `info=None` (SciPy-style iteration info is planned).
    """
    b = jnp.asarray(b)
    if x0 is not None:
        x0 = jnp.asarray(x0)

    method = _normalize_krylov_method(solve_method)
    if method in {"bicgstab", "bicgstab_jax"}:
        res, r = bicgstab_solve_with_residual(
            matvec=matvec,
            b=b,
            preconditioner=preconditioner,
            x0=x0,
            tol=tol,
            atol=atol,
            maxiter=maxiter,
            precondition_side=precondition_side,
        )
        target = jnp.maximum(
            jnp.asarray(atol, dtype=b.dtype),
            jnp.asarray(tol, dtype=b.dtype) * jnp.linalg.norm(b),
        )
        need_fallback = jnp.logical_or(~jnp.isfinite(res.residual_norm), res.residual_norm > target)
        if _contains_tracer(b, x0, res.residual_norm):
            # Keep the BiCGStab->GMRES rescue path available under JIT by avoiding
            # Python-side scalar conversion on traced residuals.
            return jax.lax.cond(
                need_fallback,
                lambda _: _gmres_solve_core(
                    matvec=matvec,
                    b=b,
                    preconditioner=preconditioner,
                    x0=x0,
                    tol=tol,
                    atol=atol,
                    restart=restart,
                    maxiter=maxiter,
                    solve_method="incremental",
                    precondition_side=precondition_side,
                ),
                lambda _: (res.x, r),
                operand=None,
            )
        if bool(need_fallback):
            # Fallback to GMRES when BiCGStab stagnates or returns non-finite residuals.
            return _gmres_solve_core(
                matvec=matvec,
                b=b,
                preconditioner=preconditioner,
                x0=x0,
                tol=tol,
                atol=atol,
                restart=restart,
                maxiter=maxiter,
                solve_method="incremental",
                precondition_side=precondition_side,
            )
        return res.x, r
    if method in _HOST_SCIPY_KRYLOV_METHODS:
        if _contains_tracer(b, x0):
            raise ValueError(f"solve_method={method} is host-only and cannot run inside JIT/differentiable tracing.")
        if maxiter is None:
            maxiter = max(1, int(np.ceil(int(b.size) / max(1, int(restart)))))
        x_np, _rn, _history = lgmres_solve_with_history_scipy(
            matvec=matvec,
            b=b,
            preconditioner=preconditioner,
            x0=x0,
            tol=tol,
            atol=atol,
            restart=restart,
            maxiter=maxiter,
            precondition_side=precondition_side,
        )
        x = jnp.asarray(x_np, dtype=b.dtype)
        r = b - matvec(x)
        return x, r
    if method == "dense":
        n = int(b.size)
        if b.ndim != 1:
            raise ValueError(f"dense solve requires a 1D vector b, got shape {b.shape}")
        # Guardrail: dense assembly is quadratic memory/time.
        dense_max_env = os.environ.get("SFINCS_JAX_DENSE_MAX", "").strip()
        try:
            dense_max = int(dense_max_env) if dense_max_env else 8000
        except ValueError:
            dense_max = 8000
        if n > dense_max:
            raise ValueError(f"dense solve is disabled for n={n} (too large). Use GMRES.")

        a = assemble_dense_matrix_from_matvec(matvec=matvec, n=n, dtype=b.dtype)
        x, _residual_norm = dense_solve_from_matrix(a=a, b=b)
        r = b - a @ x
        return x, r
    if method == "dense_row_scaled":
        n = int(b.size)
        if b.ndim != 1:
            raise ValueError(f"dense solve requires a 1D vector b, got shape {b.shape}")
        # Guardrail: dense assembly is quadratic memory/time.
        dense_max_env = os.environ.get("SFINCS_JAX_DENSE_MAX", "").strip()
        try:
            dense_max = int(dense_max_env) if dense_max_env else 8000
        except ValueError:
            dense_max = 8000
        if n > dense_max:
            raise ValueError(f"dense solve is disabled for n={n} (too large). Use GMRES.")

        a = assemble_dense_matrix_from_matvec(matvec=matvec, n=n, dtype=b.dtype)
        x, _residual_norm = dense_solve_from_matrix_row_scaled(a=a, b=b)
        r = b - a @ x
        return x, r

    restart_use = _maybe_limit_restart(int(b.size), int(restart), b.dtype)

    side = str(precondition_side).strip().lower()
    if side not in {"left", "right", "none"}:
        side = "left"

    if side == "right" and preconditioner is not None:
        # PETSc's GMRES defaults to right preconditioning: solve A P^{-1} y = b, x = P^{-1} y.
        # Here, `preconditioner` is expected to apply P^{-1}.
        def matvec_right(y):
            return matvec(preconditioner(y))

        y, _info = gmres(
            matvec_right,
            b,
            x0=None,
            tol=float(tol),
            atol=float(atol),
            restart=int(restart_use),
            maxiter=maxiter,
            M=None,
            solve_method=solve_method,
        )
        x = preconditioner(y)
    else:
        # Left preconditioning (SciPy-style): solve P^{-1} A x = P^{-1} b.
        M = preconditioner if side == "left" else None
        x, _info = gmres(
            matvec,
            b,
            x0=x0,
            tol=float(tol),
            atol=float(atol),
            restart=int(restart_use),
            maxiter=maxiter,
            M=M,
            solve_method=solve_method,
        )

    r = b - matvec(x)
    return x, r


def gmres_solve(
    *,
    matvec,
    b: jnp.ndarray,
    preconditioner=None,
    x0: jnp.ndarray | None = None,
    tol: float = 1e-10,
    atol: float = 0.0,
    restart: int = 50,
    maxiter: int | None = None,
    solve_method: str = "incremental",
    precondition_side: str = "left",
) -> GMRESSolveResult:
    x, r = _gmres_solve_core(
        matvec=matvec,
        b=b,
        preconditioner=preconditioner,
        x0=x0,
        tol=tol,
        atol=atol,
        restart=restart,
        maxiter=maxiter,
        solve_method=solve_method,
        precondition_side=precondition_side,
    )
    return GMRESSolveResult(x=x, residual_norm=jnp.linalg.norm(r))


def gmres_solve_with_residual(
    *,
    matvec,
    b: jnp.ndarray,
    preconditioner=None,
    x0: jnp.ndarray | None = None,
    tol: float = 1e-10,
    atol: float = 0.0,
    restart: int = 50,
    maxiter: int | None = None,
    solve_method: str = "incremental",
    precondition_side: str = "left",
) -> tuple[GMRESSolveResult, jnp.ndarray]:
    x, r = _gmres_solve_core(
        matvec=matvec,
        b=b,
        preconditioner=preconditioner,
        x0=x0,
        tol=tol,
        atol=atol,
        restart=restart,
        maxiter=maxiter,
        solve_method=solve_method,
        precondition_side=precondition_side,
    )
    return GMRESSolveResult(x=x, residual_norm=jnp.linalg.norm(r)), r


gmres_solve_jit = jax.jit(
    gmres_solve,
    static_argnames=("matvec", "preconditioner", "tol", "atol", "restart", "maxiter", "solve_method", "precondition_side"),
)

gmres_solve_with_residual_jit = jax.jit(
    gmres_solve_with_residual,
    static_argnames=("matvec", "preconditioner", "tol", "atol", "restart", "maxiter", "solve_method", "precondition_side"),
)


def _distributed_gmres_axis() -> str | None:
    env = os.environ.get("SFINCS_JAX_GMRES_DISTRIBUTED", "").strip().lower()
    if env in {"0", "false", "no", "off"}:
        return None
    if env in {"theta", "zeta"}:
        return env
    shard_axis = os.environ.get("SFINCS_JAX_MATVEC_SHARD_AXIS", "").strip().lower()
    if shard_axis in {"flat", "vector", "p"}:
        return "p"
    if env in {"1", "true", "yes", "on"} and shard_axis in {"theta", "zeta"}:
        return shard_axis
    if env in {"1", "true", "yes", "on"} and shard_axis in {"auto", ""}:
        return "p"
    return None


def _distributed_krylov_preference() -> str:
    """Preferred distributed Krylov method for auto/default solves."""
    env = os.environ.get("SFINCS_JAX_DISTRIBUTED_KRYLOV", "").strip().lower()
    if env in {"", "auto", "comm_reduced", "short_recurrence"}:
        return "bicgstab"
    if env in {"bicgstab", "bicgstab_jax"}:
        return "bicgstab"
    if env in {"gmres", "incremental", "batched"}:
        return "gmres"
    return "bicgstab"


def _distributed_solver_kind(solve_method: str) -> tuple[str, str]:
    method = str(solve_method).strip().lower()
    if method in _HOST_SCIPY_KRYLOV_METHODS:
        raise ValueError(f"solve_method={method} is host-only and unsupported for distributed solves.")
    if method in {"bicgstab", "bicgstab_jax"}:
        return "bicgstab", "batched"
    if method in {"auto", "default"}:
        if _distributed_krylov_preference() == "gmres":
            return "gmres", "incremental"
        return "bicgstab", "batched"
    return "gmres", method


@lru_cache(maxsize=4)
def _get_gmres_mesh(axis_name: str) -> Mesh | None:
    if Mesh is None:
        return None
    devices = jax.local_devices()
    if len(devices) <= 1:
        return None
    mesh_devices = np.array(devices)
    return Mesh(mesh_devices, (axis_name,))


def distributed_gmres_enabled() -> bool:
    axis_name = _distributed_gmres_axis()
    if axis_name is None or _pjit is None or PartitionSpec is None:
        return False
    return _get_gmres_mesh(axis_name) is not None


def gmres_solve_distributed(
    *,
    matvec,
    b: jnp.ndarray,
    preconditioner=None,
    x0: jnp.ndarray | None = None,
    axis_name: str | None = None,
    tol: float = 1e-10,
    atol: float = 0.0,
    restart: int = 50,
    maxiter: int | None = None,
    solve_method: str = "incremental",
    precondition_side: str = "left",
) -> GMRESSolveResult:
    axis_name = _distributed_gmres_axis() if axis_name is None else axis_name
    if axis_name is None or _pjit is None or PartitionSpec is None:
        return gmres_solve(
            matvec=matvec,
            b=b,
            preconditioner=preconditioner,
            x0=x0,
            tol=tol,
            atol=atol,
            restart=restart,
            maxiter=maxiter,
            solve_method=solve_method,
            precondition_side=precondition_side,
        )
    mesh = _get_gmres_mesh(axis_name)
    if mesh is None:
        return gmres_solve(
            matvec=matvec,
            b=b,
            preconditioner=preconditioner,
            x0=x0,
            tol=tol,
            atol=atol,
            restart=restart,
            maxiter=maxiter,
            solve_method=solve_method,
            precondition_side=precondition_side,
        )

    solver_kind, solve_method_use = _distributed_solver_kind(solve_method)
    b_use = _materialize_distributed_input(b)
    assert b_use is not None
    x0_input = _materialize_distributed_input(x0, dtype=b_use.dtype)
    x0_use = jnp.zeros_like(b_use) if x0_input is None else x0_input
    n = int(b_use.size)
    n_devices = int(np.prod(mesh.devices.shape))
    pad = (-n) % n_devices if n_devices > 0 else 0
    preconditioner_use = preconditioner
    if pad:
        b_use = jnp.pad(b_use, (0, pad))
        x0_use = jnp.pad(x0_use, (0, pad))

        def matvec_use(x):
            y = matvec(x[:n])
            return jnp.pad(y, (0, pad))
        if preconditioner is not None:
            def preconditioner_use(x):
                y = preconditioner(x[:n])
                return jnp.pad(y, (0, pad))
    else:
        matvec_use = matvec

    def _solve(b_in: jnp.ndarray, x0_in: jnp.ndarray):
        if solver_kind == "bicgstab":
            res = bicgstab_solve(
                matvec=matvec_use,
                b=b_in,
                preconditioner=preconditioner_use,
                x0=x0_in,
                tol=tol,
                atol=atol,
                maxiter=maxiter,
                precondition_side=precondition_side,
            )
        else:
            res = gmres_solve(
                matvec=matvec_use,
                b=b_in,
                preconditioner=preconditioner_use,
                x0=x0_in,
                tol=tol,
                atol=atol,
                restart=restart,
                maxiter=maxiter,
                solve_method=solve_method_use,
                precondition_side=precondition_side,
            )
        return res.x, res.residual_norm

    solve_pjit = _pjit.pjit(
        _solve,
        in_shardings=(PartitionSpec(axis_name), PartitionSpec(axis_name)),
        out_shardings=(PartitionSpec(axis_name), None),
    )
    with mesh:
        x, rn = solve_pjit(b_use, x0_use)
        if pad:
            r_pad = b_use - matvec_use(x)
            r = r_pad[:n]
            rn = jnp.linalg.norm(r)
    if pad:
        x = x[:n]
    return GMRESSolveResult(x=x, residual_norm=rn)


def gmres_solve_with_residual_distributed(
    *,
    matvec,
    b: jnp.ndarray,
    preconditioner=None,
    x0: jnp.ndarray | None = None,
    axis_name: str | None = None,
    tol: float = 1e-10,
    atol: float = 0.0,
    restart: int = 50,
    maxiter: int | None = None,
    solve_method: str = "incremental",
    precondition_side: str = "left",
) -> tuple[GMRESSolveResult, jnp.ndarray]:
    axis_name = _distributed_gmres_axis() if axis_name is None else axis_name
    if axis_name is None or _pjit is None or PartitionSpec is None:
        return gmres_solve_with_residual(
            matvec=matvec,
            b=b,
            preconditioner=preconditioner,
            x0=x0,
            tol=tol,
            atol=atol,
            restart=restart,
            maxiter=maxiter,
            solve_method=solve_method,
            precondition_side=precondition_side,
        )
    mesh = _get_gmres_mesh(axis_name)
    if mesh is None:
        return gmres_solve_with_residual(
            matvec=matvec,
            b=b,
            preconditioner=preconditioner,
            x0=x0,
            tol=tol,
            atol=atol,
            restart=restart,
            maxiter=maxiter,
            solve_method=solve_method,
            precondition_side=precondition_side,
        )

    solver_kind, solve_method_use = _distributed_solver_kind(solve_method)
    b_use = _materialize_distributed_input(b)
    assert b_use is not None
    x0_input = _materialize_distributed_input(x0, dtype=b_use.dtype)
    x0_use = jnp.zeros_like(b_use) if x0_input is None else x0_input
    n = int(b_use.size)
    n_devices = int(np.prod(mesh.devices.shape))
    pad = (-n) % n_devices if n_devices > 0 else 0
    preconditioner_use = preconditioner
    if pad:
        b_use = jnp.pad(b_use, (0, pad))
        x0_use = jnp.pad(x0_use, (0, pad))

        def matvec_use(x):
            y = matvec(x[:n])
            return jnp.pad(y, (0, pad))
        if preconditioner is not None:
            def preconditioner_use(x):
                y = preconditioner(x[:n])
                return jnp.pad(y, (0, pad))
    else:
        matvec_use = matvec

    def _solve(b_in: jnp.ndarray, x0_in: jnp.ndarray):
        if solver_kind == "bicgstab":
            res, r = bicgstab_solve_with_residual(
                matvec=matvec_use,
                b=b_in,
                preconditioner=preconditioner_use,
                x0=x0_in,
                tol=tol,
                atol=atol,
                maxiter=maxiter,
                precondition_side=precondition_side,
            )
        else:
            res, r = gmres_solve_with_residual(
                matvec=matvec_use,
                b=b_in,
                preconditioner=preconditioner_use,
                x0=x0_in,
                tol=tol,
                atol=atol,
                restart=restart,
                maxiter=maxiter,
                solve_method=solve_method_use,
                precondition_side=precondition_side,
            )
        return res.x, res.residual_norm, r

    solve_pjit = _pjit.pjit(
        _solve,
        in_shardings=(PartitionSpec(axis_name), PartitionSpec(axis_name)),
        out_shardings=(PartitionSpec(axis_name), None, PartitionSpec(axis_name)),
    )
    with mesh:
        x, rn, r = solve_pjit(b_use, x0_use)
        if pad:
            r_pad = b_use - matvec_use(x)
            r = r_pad[:n]
            rn = jnp.linalg.norm(r)
    if pad:
        x = x[:n]
    return GMRESSolveResult(x=x, residual_norm=rn), r
