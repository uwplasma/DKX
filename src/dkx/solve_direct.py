"""Sparse direct route of :func:`dkx.solve.solve` (code name ``tier3``).

Host CSR assembly, SuperLU or MUMPS factors, refinement and factor reuse.
Non-differentiable; the referee and the fallback when recycled Krylov stalls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Callable
import inspect
import time

# The JAX backend is imported below; dkx/runtime.py explains why this is here.
from .runtime import configure as _configure_runtime

_configure_runtime()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

try:  # noqa: E402
    from solvax.equilibration import equilibrate
    from solvax.native import SpluFactorization
    from solvax.native_eigen import sparse_operator_matrix
except ImportError:
    SpluFactorization = None  # type: ignore[assignment, misc]
    equilibrate = None  # type: ignore[assignment, misc]
    sparse_operator_matrix = None  # type: ignore[assignment, misc]

from dkx.coarse_precond import (  # noqa: E402
    _available_memory_bytes,
    _require_solvax,
)
from dkx.drift_kinetic import (  # noqa: E402
    KineticOperator,
)
from dkx.solve_common import (  # noqa: E402
    SolveResult,
    _converged_flag,
    _pinned_matvecs,
    _say,
)
from dkx.structured_direct import (  # noqa: E402
    _is_traced,
)

# =============================================================================
# Sparse direct — host fallback and independent cross-check
# =============================================================================


def materialize_csr(
    op: KineticOperator, *, column_chunk: int = 1024, pin_masked_dofs: bool = False
):
    """Materialize the full bordered operator as a scipy CSR matrix.

    ``solvax.native_eigen.sparse_operator_matrix`` owns the sampling: it
    applies the matrix-free operator to identity columns in vmapped chunks and
    sparsifies each chunk as it lands, so no dense ``O(total_size**2)`` array
    is ever formed.  The kinetic operator is a stencil (about 9 of 1121
    entries per angular row; see :mod:`dkx.sparse_precond`), so CSR is what the
    matrix actually is.

    The cost that is *not* removed is the sampling itself: ``total_size``
    operator applications, one per column.  That, not the intermediate, is what
    ``max_dense_size`` guards.

    Args:
        op: the kinetic operator.
        column_chunk: identity columns per vmapped batch.
        pin_masked_dofs: materialize the pinned operator (identity rows and
            columns on the DOFs truncated by ``Nxi_for_x``; see
            :func:`_pinned_matvecs`) instead of the raw rectangular embedding,
            which has exact zero rows on those DOFs.
    """
    _require_solvax()
    apply = _pinned_matvecs(op)[0] if pin_masked_dofs else op.apply
    prototype = jnp.zeros((op.total_size,), dtype=jnp.float64)
    return sparse_operator_matrix(apply, prototype, batch_size=column_chunk)

def materialize_dense(
    op: KineticOperator, *, column_chunk: int = 1024, pin_masked_dofs: bool = False
) -> np.ndarray:
    """:func:`materialize_csr` as a dense numpy matrix, for referee tests.

    Densifying costs ``O(total_size**2)`` memory, which is why the sparse
    direct route uses the CSR form directly and only tests come through here.
    """
    return materialize_csr(
        op, column_chunk=column_chunk, pin_masked_dofs=pin_masked_dofs
    ).toarray()

@dataclass(frozen=True)
class DirectFactors:
    """Sparse factors of the equilibrated assembled operator, kept to reuse.

    The sparse direct route spends nearly all of its time building this: the
    assembly, the Ruiz scaling and the factorization. The solve that follows is
    two triangular substitutions. Production never asks for one solve --- a
    transport matrix is three right-hand sides of one operator and a gradient is
    one transposed solve --- so the factors are returned rather than dropped,
    and :func:`solve` takes them back.

    The transpose comes from the same factors. With ``A_s = D_r A D_c`` the
    scaled system that was factored, ``A^T y = g`` is ``A_s^T z = D_c g`` with
    ``y = D_r z``. The backend applies its transpose solve and the
    factorization is never repeated.

    Attributes:
        lu: the stored :class:`solvax.native.SpluFactorization`.
        row_scale, column_scale: the Ruiz diagonals ``D_r`` and ``D_c``.
        total_size: the operator size these factors were built for; a
            right-hand side of any other length is refused.
        products: operator applications the assembly cost (0 when the matrix
            was sampled column by column instead).
        nnz: nonzeros in the factored matrix.
    """

    lu: Any
    row_scale: np.ndarray
    column_scale: np.ndarray
    total_size: int
    products: int
    nnz: int
    backend: str = "superlu"
    memory_budget_bytes: int | None = None
    workspace_limit_bytes: int | None = None

    @property
    def symbolic_memory_bytes(self) -> int | None:
        """Backend symbolic estimate, when the backend exposes one."""
        return getattr(self.lu, "symbolic_memory_bytes", None)

    @property
    def effective_memory_bytes(self) -> int | None:
        """Backend effective factor memory, when the backend exposes one."""
        return getattr(self.lu, "effective_memory_bytes", None)

    def close(self) -> None:
        """Release native factor storage when the backend supports it."""
        close = getattr(self.lu, "close", None)
        if close is not None:
            close()

    def __enter__(self) -> DirectFactors:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def solve(self, rhs: np.ndarray, *, transpose: bool = False) -> np.ndarray:
        """Apply the stored factors to ``rhs`` (all columns at once).

        Args:
            rhs: ``(total_size,)`` or ``(total_size, n_rhs)``, concrete.
            transpose: solve ``A^T x = rhs`` instead of ``A x = rhs``.

        Returns:
            The solution, shaped like ``rhs``.
        """
        b = np.asarray(rhs)
        squeeze = b.ndim == 1
        if squeeze:
            b = b[:, None]
        if b.shape[0] != self.total_size:
            raise ValueError(
                f"stored factors are of size {self.total_size}; this "
                f"right-hand side has {b.shape[0]} rows"
            )
        pre, post = (
            (self.column_scale, self.row_scale)
            if transpose
            else (self.row_scale, self.column_scale)
        )
        y = np.asarray(self.lu.solve(b * pre[:, None], trans="T" if transpose else "N"))
        x = y * post[:, None]
        return x[:, 0] if squeeze else x

_MUMPS_MEMORY_SAFETY_FACTOR = 1.2

_MUMPS_MINIMUM_BUDGET_BYTES = 1_000_000

# Conservative equivalent full-RHS buffers beyond the caller-owned RHS, which
# current RSS already includes: scaled RHS; native per-column results; native
# stacked result; host/JAX conversion; equilibrated solution; JAX apply output;
# and defect/correction destination.  These do not all peak simultaneously, but
# summing them avoids claiming allocator/JAX lifetime precision we do not have.
_MUMPS_TRANSIENT_RHS_COPIES = 7

def _process_rss_bytes() -> tuple[int | None, str]:
    """RSS evidence in bytes and its metric: current, peak fallback, or unavailable."""
    try:
        import psutil  # type: ignore[import-not-found]  # noqa: PLC0415

        return int(psutil.Process().memory_info().rss), "current RSS"
    except Exception:
        from .profiling import _peak_rss_mb  # noqa: PLC0415

        peak_mib = _peak_rss_mb()
        # _peak_rss_mb converts Linux KiB/macOS bytes to MiB-like units. This
        # fallback is deliberately a high-water value, hence conservative.
        if peak_mib is None:
            return None, "RSS unavailable"
        return int(float(peak_mib) * 2.0**20), "peak RSS fallback"

def _mumps_workspace_limit_bytes(
    scaled_matrix,
    *,
    rhs_nbytes: int,
    memory_budget_bytes: int,
) -> int:
    """MUMPS workspace left after DKX's live host allocations and headroom."""
    from .batch import _RUNTIME_OVERHEAD_BYTES  # local: batch imports solve

    live_rss, rss_metric = _process_rss_bytes()
    available = _available_memory_bytes()
    if live_rss is None or available is None:
        raise RuntimeError(
            f"MUMPS memory admission requires process RSS evidence ({rss_metric}) "
            "and current host available memory; neither physical RAM nor the "
            "process budget alone is a safe shared-host guard"
        )
    index_bytes = int(scaled_matrix.indices.dtype.itemsize)
    coo_bytes = int(scaled_matrix.data.nbytes + 2 * scaled_matrix.nnz * index_bytes)
    # RSS already includes the operator, both CSR matrices, JAX caches, unrelated
    # arrays and any stale native factors. Reserve the future PyMUMPS COO copy,
    # conservative solve transients, and the measured runtime/allocator allowance.
    future_reserve = int(
        coo_bytes
        + _MUMPS_TRANSIENT_RHS_COPIES * rhs_nbytes
        + _RUNTIME_OVERHEAD_BYTES
    )
    process_remaining = int(memory_budget_bytes) - int(live_rss) - future_reserve
    available_remaining = float(available) - future_reserve
    workspace = int(min(process_remaining, available_remaining))
    if workspace < _MUMPS_MINIMUM_BUDGET_BYTES:
        raise MemoryError(
            "MUMPS factorization refused before symbolic analysis: "
            f"process budget {int(memory_budget_bytes)} bytes minus {rss_metric} "
            f"{int(live_rss)} bytes and future COO/RHS/runtime reserve {future_reserve} "
            f"leaves {process_remaining} bytes; current host availability leaves "
            f"{available_remaining:.0f} bytes"
        )
    return workspace

def _admit_mumps_reuse(*, rhs_nbytes: int, memory_budget_bytes: int) -> None:
    """Recheck solve-only headroom when retained MUMPS factors take a new RHS.

    Factor-build admission remains separate because it also reserves projected
    COO storage and derives the native workspace limit after grouped assembly.
    """
    from .batch import _RUNTIME_OVERHEAD_BYTES  # local: batch imports solve

    live_rss, rss_metric = _process_rss_bytes()
    available = _available_memory_bytes()
    if live_rss is None or available is None:
        raise RuntimeError(
            f"MUMPS factor reuse requires process RSS evidence ({rss_metric}) "
            "and current host available memory"
        )
    transient = int(
        _MUMPS_TRANSIENT_RHS_COPIES * rhs_nbytes + _RUNTIME_OVERHEAD_BYTES
    )
    process_remaining = int(memory_budget_bytes) - int(live_rss) - transient
    available_remaining = float(available) - transient
    if process_remaining < 0 or available_remaining < 0:
        raise MemoryError(
            "MUMPS factor reuse refused before solve: "
            f"process budget {int(memory_budget_bytes)} bytes minus {rss_metric} "
            f"{int(live_rss)} bytes and transient RHS/runtime reserve {transient} "
            f"leaves {process_remaining} bytes; current host availability leaves "
            f"{available_remaining:.0f} bytes"
        )

def _factor_direct(
    op: KineticOperator,
    *,
    max_dense_size: int,
    direct_backend: str = "superlu",
    memory_budget_bytes: int | None = None,
    rhs_nbytes: int = 0,
    emit: Callable[[str], None] | None,
) -> DirectFactors:
    """Assemble, equilibrate and factor the operator once."""
    if direct_backend == "mumps":
        if "backend" not in inspect.signature(SpluFactorization).parameters:
            raise ImportError(
                "direct_backend='mumps' requires a SOLVAX release containing "
                "the optional MUMPS adapter; released SOLVAX 0.24 supports the "
                "default SuperLU backend only"
            )
        if memory_budget_bytes is None:
            raise ValueError("direct_backend='mumps' requires a memory budget")
    n = op.total_size
    assembled = None
    if n > max_dense_size:
        # Sampling costs one operator application per column, which is what
        # max_dense_size bounds. The assembly costs one per group of columns
        # that share no row -- 8,800 for 633,600 unknowns -- so that bound does
        # not apply to it, and what the route can afford is the factorization.
        try:
            from dkx.assembly import assemble_operator  # noqa: PLC0415

            assembled = assemble_operator(op)
        except Exception as exc:  # the refusal is the fallback, not a crash
            raise RuntimeError(
                f"sparse direct materialization refused: total_size={n} > "
                f"max_dense_size={max_dense_size}, i.e. {n} operator "
                "applications to sample the matrix column by column, and the "
                f"assembly that would avoid them is unavailable here ({exc}). "
                "Raise max_dense_size explicitly to sample anyway."
            ) from exc
    if assembled is None:
        _say(
            emit,
            f"[dkx.solve] sparse direct route (host {direct_backend}, n={n}): "
            "non-differentiable fallback path.",
        )
        matrix = materialize_csr(op, pin_masked_dofs=True)
    else:
        _say(
            emit,
            f"[dkx.solve] sparse direct route (host {direct_backend}, n={n}): assembled "
            f"from {assembled.products} products, {assembled.matrix.nnz} "
            "nonzeros; non-differentiable fallback path.",
        )
        matrix = assembled.matrix
    # A factorization chooses its pivots from the matrix it is handed, and these
    # rows carry streaming, collision and constraint terms at once. Scaling
    # first is what keeps those pivots meaningful.
    scaling = equilibrate(matrix)
    workspace_limit = None
    if direct_backend == "superlu":
        # Keep the released SOLVAX 0.24 constructor path byte-for-byte simple.
        lu = SpluFactorization(scaling.matrix)
    else:
        workspace_limit = _mumps_workspace_limit_bytes(
            scaling.matrix, rhs_nbytes=rhs_nbytes,
            memory_budget_bytes=memory_budget_bytes,
        )
        lu = SpluFactorization(
            scaling.matrix,
            backend="mumps",
            memory_limit_bytes=workspace_limit,
            memory_safety_factor=_MUMPS_MEMORY_SAFETY_FACTOR,
        )
    return DirectFactors(
        lu=lu,
        row_scale=np.asarray(scaling.row_scale),
        column_scale=np.asarray(scaling.column_scale),
        total_size=n,
        products=0 if assembled is None else int(assembled.products),
        nnz=int(matrix.nnz),
        backend=direct_backend,
        memory_budget_bytes=memory_budget_bytes,
        workspace_limit_bytes=workspace_limit,
    )

#: Defect-correction sweeps the sparse direct route runs on its stored factors.
#: One is what a fresh factorization needs to turn the factorization's residual
#: into the solution's. Reused factors are a solve of a *neighbouring* operator
#: corrected by the current one, so they get a bounded few more before the
#: route gives up on them and factorizes again.
_DIRECT_REFINEMENT_SWEEPS = 1

_DIRECT_REUSE_REFINEMENT_SWEEPS = 4

def _direct_refine(
    op: KineticOperator,
    factors: DirectFactors,
    rhs: np.ndarray,
    *,
    transpose: bool,
    sweeps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve with ``factors`` and correct the defect against ``op`` itself.

    The correction is what makes reuse safe: the defect is measured with the
    *current* operator and only the inverse comes from the stored factors, so a
    solve of a neighbouring operator converges to this one's answer when the
    factors are close enough and visibly fails to when they are not.
    """
    apply = _pinned_matvecs(op)[1 if transpose else 0]
    apply2d = jax.vmap(apply, in_axes=1, out_axes=1)
    x = factors.solve(rhs, transpose=transpose)
    defect = rhs - np.asarray(apply2d(jnp.asarray(x)))
    for _ in range(sweeps):
        x = x + factors.solve(defect, transpose=transpose)
        defect = rhs - np.asarray(apply2d(jnp.asarray(x)))
    # The last sweep already measured the defect of the solution it produced;
    # the caller's convergence check is that defect and costs no further apply.
    return x, defect

def _solve_tier3(
    op: KineticOperator,
    rhs2d: jnp.ndarray,
    *,
    tol: float,
    atol: float,
    max_dense_size: int,
    direct_backend: str | None = None,
    direct_memory_budget_gb: float | None = None,
    factors: DirectFactors | None = None,
    transpose: bool = False,
    emit: Callable[[str], None] | None = print,
) -> SolveResult:
    _require_solvax()
    if _is_traced(rhs2d):
        raise RuntimeError(
            "the host sparse-direct solve is non-differentiable and "
            "cannot run "
            "under jit/vmap/grad; use method='block_tridiagonal' or 'gmres' with "
            "differentiable=True."
        )
    if factors is not None and factors.total_size != op.total_size:
        raise ValueError(
            f"the factors passed in are of size {factors.total_size} and this "
            f"operator is {op.total_size}; a factorization is only reusable "
            "for an operator of its own shape"
        )
    rhs = np.asarray(rhs2d)
    if rhs.ndim == 1:
        rhs = rhs[:, None]
    # The native matrix is float64 in DKX.  Account for conversion to that dtype
    # when a caller supplies narrower RHS storage.
    rhs_work_nbytes = int(rhs.size * max(rhs.dtype.itemsize, np.dtype(np.float64).itemsize))

    requested_budget_bytes = (
        None if direct_memory_budget_gb is None
        else int(float(direct_memory_budget_gb) * 2.0**30)
    )
    if factors is not None:
        if direct_backend is not None and direct_backend != factors.backend:
            raise ValueError(
                f"stored factors use {factors.backend!r}, not requested "
                f"direct_backend={direct_backend!r}"
            )
        if (requested_budget_bytes is not None
                and factors.memory_budget_bytes is not None
                and requested_budget_bytes != factors.memory_budget_bytes):
            raise ValueError("a reused factorization retains its original memory budget")
        backend = factors.backend
        memory_budget_bytes = factors.memory_budget_bytes
    else:
        backend = "superlu" if direct_backend is None else direct_backend
        memory_budget_bytes = requested_budget_bytes
    if backend == "mumps" and memory_budget_bytes is None:
        raise ValueError("direct_backend='mumps' requires a memory budget")

    def _build_factors() -> DirectFactors:
        if backend == "superlu":
            # Preserve the original internal call shape for released defaults
            # and tests/downstream code that monkeypatch the private builder.
            return _factor_direct(op, max_dense_size=max_dense_size, emit=emit)
        return _factor_direct(
            op, max_dense_size=max_dense_size, direct_backend=backend,
            memory_budget_bytes=memory_budget_bytes, rhs_nbytes=rhs_work_nbytes, emit=emit,
        )

    reused = factors is not None
    if reused and backend == "mumps":
        _admit_mumps_reuse(
            rhs_nbytes=rhs_work_nbytes,
            memory_budget_bytes=memory_budget_bytes,
        )
    t0 = time.perf_counter()
    if factors is None:
        factors = _build_factors()
    t1 = time.perf_counter()
    x2d, defect = _direct_refine(
        op, factors, rhs, transpose=transpose,
        sweeps=(
            _DIRECT_REUSE_REFINEMENT_SWEEPS if reused else _DIRECT_REFINEMENT_SWEEPS
        ),
    )
    t2 = time.perf_counter()

    res = jnp.asarray(np.linalg.norm(defect, axis=0))
    if reused and not _converged_flag(res, jnp.asarray(rhs), tol, atol):
        # The staleness test is the original residual, and it costs the applies
        # the correction already paid for. Recovery is one factorization, not an
        # unbounded wait: factors built at one Er have been measured to stall a
        # Krylov solve for 6,000 iterations at another.
        _say(
            emit,
            "[dkx.solve] the stored factors did not solve this operator "
            f"(residuals={np.asarray(res)} after "
            f"{_DIRECT_REUSE_REFINEMENT_SWEEPS} corrections); factorizing it.",
        )
        t0 = time.perf_counter()
        factors = _build_factors()
        t1 = time.perf_counter()
        x2d, defect = _direct_refine(
            op, factors, rhs, transpose=transpose,
            sweeps=_DIRECT_REFINEMENT_SWEEPS,
        )
        t2 = time.perf_counter()
        res = jnp.asarray(np.linalg.norm(defect, axis=0))
        reused = False

    return SolveResult(
        x=jnp.asarray(x2d),
        method="direct",
        iterations=None,
        residual_norms=res,
        converged=_converged_flag(res, jnp.asarray(rhs), tol, atol),
        recycle=None,
        timings={"build": 0.0 if reused else t1 - t0, "solve": t2 - t1},
        factors=factors,
    )
